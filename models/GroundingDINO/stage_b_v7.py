from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from groundingdino.util import box_ops

from .bertwarper import generate_masks_with_special_tokens_and_transfer_map
from .utils import ContrastiveEmbed, MLP


def calibrate_finite_token_logits(
    token_logits: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    *,
    invalid_fill: float = -20.0,
) -> torch.Tensor:
    """Apply learnable calibration without backpropagating through +/-inf padding."""
    finite = torch.isfinite(token_logits)
    safe_logits = torch.where(finite, token_logits, torch.zeros_like(token_logits))
    calibrated = safe_logits * logit_scale.clamp(-2.0, 2.0).exp() + logit_bias
    return calibrated.masked_fill(~finite, float(invalid_fill))


def _pair_stride_values(pair_stride, batch_size: int, device: torch.device) -> torch.Tensor:
    if pair_stride is None:
        return torch.ones((batch_size,), dtype=torch.long, device=device)
    if torch.is_tensor(pair_stride):
        values = pair_stride.detach().reshape(-1).to(device=device, dtype=torch.long)
    else:
        values = torch.as_tensor(pair_stride, dtype=torch.long, device=device).reshape(-1)
    if values.numel() == 0:
        return torch.ones((batch_size,), dtype=torch.long, device=device)
    if values.numel() not in {1, int(batch_size)}:
        raise ValueError(f"pair_stride must be scalar or have B={batch_size} values, got {values.numel()}")
    if values.numel() == 1:
        values = values.expand(batch_size)
    return values.clamp_min(1)


def build_stage_b_v7_candidate_scores(
    predicate_logits: torch.Tensor,
    patch_score: torch.Tensor,
    *,
    candidate_topk: int,
    patch_prior_weight: float = 0.0,
    pair_stride=None,
) -> Dict[str, torch.Tensor]:
    """Gate with Stage-A patch scores, then rank admitted candidates with text."""
    if predicate_logits.dim() != 3:
        raise ValueError(f"predicate_logits must be (B,Q,K), got {tuple(predicate_logits.shape)}")
    if patch_score.dim() == 2:
        patch_score = patch_score.unsqueeze(-1)
    if patch_score.dim() != 3 or patch_score.shape[:2] != predicate_logits.shape[:2]:
        raise ValueError(
            "patch_score must be (B,Q,Kpatch) and align with predicate logits, "
            f"got {tuple(patch_score.shape)} vs {tuple(predicate_logits.shape)}"
        )

    batch_size, num_queries, num_phrase_slots = predicate_logits.shape
    stride_values = _pair_stride_values(pair_stride, batch_size, predicate_logits.device)
    expanded_patch_score = torch.zeros_like(predicate_logits)
    valid_phrase_slots = torch.zeros(
        (batch_size, num_phrase_slots), dtype=torch.bool, device=predicate_logits.device
    )
    for batch_idx, stride_value in enumerate(stride_values.tolist()):
        expanded = patch_score[batch_idx].repeat_interleave(int(stride_value), dim=-1)
        if expanded.shape[-1] > num_phrase_slots:
            raise ValueError(
                "Cannot align patch slots with verifier phrase slots: "
                f"patch={patch_score.shape[-1]}, phrase={num_phrase_slots}, stride={stride_value}"
            )
        num_valid_slots = int(expanded.shape[-1])
        expanded_patch_score[batch_idx, :, :num_valid_slots] = expanded
        valid_phrase_slots[batch_idx, :num_valid_slots] = True

    topk = int(candidate_topk)
    if topk <= 0 or topk >= num_queries:
        candidate_mask = torch.ones_like(predicate_logits, dtype=torch.bool)
    else:
        candidate_idx = torch.topk(expanded_patch_score, k=topk, dim=1, largest=True, sorted=False).indices
        candidate_mask = torch.zeros_like(predicate_logits, dtype=torch.bool)
        candidate_mask.scatter_(1, candidate_idx, True)
    candidate_mask = candidate_mask & valid_phrase_slots[:, None, :]

    eps = max(float(torch.finfo(predicate_logits.dtype).eps), 1e-6)
    patch_prior_logit = torch.logit(expanded_patch_score.clamp(min=eps, max=1.0 - eps))
    final_logits = predicate_logits + float(patch_prior_weight) * patch_prior_logit
    final_logits = final_logits.masked_fill(
        ~candidate_mask, torch.finfo(final_logits.dtype).min
    )
    return {
        "candidate_mask": candidate_mask,
        "expanded_patch_score": expanded_patch_score,
        "final_logits": final_logits,
        "final_score": final_logits.sigmoid(),
    }


class StageBVerifier(nn.Module):
    """Detached post-candidate verifier for Stage B v7."""

    def __init__(
        self,
        *,
        tokenizer,
        bert: nn.Module,
        feat_map: nn.Module,
        hidden_dim: int,
        max_text_len: int = 256,
        sub_sentence_present: bool = True,
        canonical_token_weight: float = 0.15,
        candidate_residual_init: bool = True,
        phrase_agg: str = "mean",
        phrase_mean_weight: float = 0.5,
        phrase_softmin_tau: float = 0.5,
        use_joint_phrase_head: bool = True,
        candidate_topk: int = 50,
        patch_prior_weight: float = 0.0,
        context_scale: float = 2.0,
        use_neighbor_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = copy.deepcopy(bert)
        self.feat_map = copy.deepcopy(feat_map)
        self.hidden_dim = int(hidden_dim)
        self.max_text_len = int(max_text_len)
        self.sub_sentence_present = bool(sub_sentence_present)
        self.canonical_token_weight = float(canonical_token_weight)
        self.candidate_residual_init = bool(candidate_residual_init)
        self.phrase_agg = str(phrase_agg or "mean").lower().strip()
        self.phrase_mean_weight = float(phrase_mean_weight)
        self.phrase_softmin_tau = float(phrase_softmin_tau)
        self.use_joint_phrase_head = bool(use_joint_phrase_head)
        self.candidate_topk = int(candidate_topk)
        self.patch_prior_weight = float(patch_prior_weight)
        self.context_scale = float(context_scale)
        self.use_neighbor_geometry = bool(use_neighbor_geometry)
        if self.use_neighbor_geometry:
            raise NotImplementedError("stage_b_v7_use_neighbor_geometry is reserved for a later ablation.")

        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        self.class_embed = ContrastiveEmbed(max_text_len=self.max_text_len)
        self.query_delta = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.roi_delta = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context_delta = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.box_delta = MLP(4, self.hidden_dim, self.hidden_dim, 2)
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.logit_bias = nn.Parameter(torch.zeros(()))
        if self.use_joint_phrase_head:
            self.phrase_visual_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.phrase_text_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.phrase_joint_norm = nn.LayerNorm(self.hidden_dim)
            self.phrase_score_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.zeros_(self.phrase_score_head[-1].weight)
            nn.init.zeros_(self.phrase_score_head[-1].bias)
        if self.candidate_residual_init:
            nn.init.zeros_(self.query_delta.weight)
            nn.init.zeros_(self.query_delta.bias)
            nn.init.zeros_(self.roi_delta.weight)
            nn.init.zeros_(self.roi_delta.bias)
            nn.init.zeros_(self.context_delta.weight)
            nn.init.zeros_(self.context_delta.bias)
            nn.init.zeros_(self.box_delta.layers[-1].weight)
            nn.init.zeros_(self.box_delta.layers[-1].bias)

        self.freeze_bert()

    @classmethod
    def from_groundingdino(
        cls,
        model,
        *,
        canonical_token_weight: float = 0.15,
        candidate_residual_init: bool = True,
        phrase_agg: str = "mean",
        phrase_mean_weight: float = 0.5,
        phrase_softmin_tau: float = 0.5,
        use_joint_phrase_head: bool = True,
        candidate_topk: int = 50,
        patch_prior_weight: float = 0.0,
        context_scale: float = 2.0,
        use_neighbor_geometry: bool = False,
    ) -> "StageBVerifier":
        verifier = cls(
            tokenizer=model.tokenizer,
            bert=model.bert,
            feat_map=model.feat_map,
            hidden_dim=model.hidden_dim,
            max_text_len=model.max_text_len,
            sub_sentence_present=model.sub_sentence_present,
            canonical_token_weight=canonical_token_weight,
            candidate_residual_init=candidate_residual_init,
            phrase_agg=phrase_agg,
            phrase_mean_weight=phrase_mean_weight,
            phrase_softmin_tau=phrase_softmin_tau,
            use_joint_phrase_head=use_joint_phrase_head,
            candidate_topk=candidate_topk,
            patch_prior_weight=patch_prior_weight,
            context_scale=context_scale,
            use_neighbor_geometry=use_neighbor_geometry,
        )
        verifier.load_from_text_branch(model)
        return verifier

    def load_from_text_branch(self, model) -> None:
        self.bert.load_state_dict(copy.deepcopy(model.bert.state_dict()), strict=True)
        self.feat_map.load_state_dict(copy.deepcopy(model.feat_map.state_dict()), strict=True)
        self.freeze_bert()

    def freeze_bert(self) -> None:
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.bert.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The copied BERT is a frozen feature extractor. Keep its dropout off
        # even when the trainable verifier heads are in training mode.
        self.bert.eval()
        return self

    def encode_text(self, captions: List[str], *, device: torch.device) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        (
            text_self_attention_masks,
            position_ids,
            _cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)
        encoded_text = self.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        return {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

    def extract_roi_features(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        *,
        feature_mask: Optional[torch.Tensor] = None,
        box_scale: float = 1.0,
    ) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(f"feature_map must be (B,C,H,W), got {tuple(feature_map.shape)}")
        if boxes.dim() != 3 or boxes.shape[-1] != 4:
            raise ValueError(f"boxes must be (B,Q,4), got {tuple(boxes.shape)}")
        B, Q, _ = boxes.shape
        _b, _c, H, W = feature_map.shape
        if _b != B:
            raise ValueError(f"feature_map/boxes batch mismatch: {tuple(feature_map.shape)} vs {tuple(boxes.shape)}")

        roi_boxes = boxes.clamp(0.0, 1.0).clone()
        roi_boxes[..., 2:] = (roi_boxes[..., 2:] * float(box_scale)).clamp(max=1.0)
        xyxy = box_ops.box_cxcywh_to_xyxy(roi_boxes).clamp(0.0, 1.0)
        if feature_mask is not None:
            feature_mask = feature_mask.to(device=boxes.device, dtype=torch.bool)
            if feature_mask.shape != (B, H, W):
                raise ValueError(
                    f"feature_mask must be {(B, H, W)}, got {tuple(feature_mask.shape)}"
                )
            valid = ~feature_mask
            valid_h = valid.any(dim=2).sum(dim=1).clamp_min(1).to(dtype=boxes.dtype)
            valid_w = valid.any(dim=1).sum(dim=1).clamp_min(1).to(dtype=boxes.dtype)
            scale = torch.stack([valid_w, valid_h, valid_w, valid_h], dim=-1)[:, None, :]
        else:
            scale = xyxy.new_tensor([float(W), float(H), float(W), float(H)]).view(1, 1, 4)
        xyxy = xyxy * scale
        x0 = torch.minimum(xyxy[..., 0], xyxy[..., 2])
        y0 = torch.minimum(xyxy[..., 1], xyxy[..., 3])
        x1 = torch.maximum(xyxy[..., 0], xyxy[..., 2])
        y1 = torch.maximum(xyxy[..., 1], xyxy[..., 3])
        xyxy = torch.stack([x0, y0, x1, y1], dim=-1)

        batch_idx = torch.arange(B, device=boxes.device, dtype=boxes.dtype)[:, None].expand(B, Q)
        rois = torch.cat([batch_idx.reshape(-1, 1), xyxy.reshape(-1, 4)], dim=1)
        pooled = roi_align(
            feature_map,
            rois,
            output_size=(1, 1),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        return pooled.flatten(1).view(B, Q, -1)

    def _aggregate_phrase_logits(
        self,
        token_logits: torch.Tensor,
        phrase_to_token_mask: Optional[torch.Tensor],
        canonical_to_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_logits.dim() != 3:
            raise ValueError(f"token_logits must be (B,Q,T), got {tuple(token_logits.shape)}")
        B, Q, T = token_logits.shape
        device = token_logits.device
        if phrase_to_token_mask is None:
            phrase_to_token_mask = torch.ones((B, 1, T), dtype=torch.bool, device=device)
        else:
            phrase_to_token_mask = phrase_to_token_mask.to(device=device, dtype=torch.bool)
            if phrase_to_token_mask.dim() != 3:
                raise ValueError(
                    f"phrase_to_token_mask must be (B,K,T), got {tuple(phrase_to_token_mask.shape)}"
                )
            if phrase_to_token_mask.shape[0] != B:
                raise ValueError(
                    f"phrase_to_token_mask batch mismatch: {tuple(phrase_to_token_mask.shape)} vs {tuple(token_logits.shape)}"
                )
            if phrase_to_token_mask.shape[-1] != T:
                phrase_to_token_mask = phrase_to_token_mask[..., :T]
                if phrase_to_token_mask.shape[-1] != T:
                    raise ValueError(
                        f"phrase_to_token_mask token mismatch: {tuple(phrase_to_token_mask.shape)} vs {tuple(token_logits.shape)}"
                    )

        if canonical_to_token_mask is None:
            canonical_to_token_mask = torch.zeros_like(phrase_to_token_mask)
        else:
            canonical_to_token_mask = canonical_to_token_mask.to(device=device, dtype=torch.bool)
            if canonical_to_token_mask.shape != phrase_to_token_mask.shape:
                canonical_to_token_mask = canonical_to_token_mask[..., :T]
                if canonical_to_token_mask.shape != phrase_to_token_mask.shape:
                    raise ValueError(
                        "canonical_to_token_mask must match phrase_to_token_mask, "
                        f"got {tuple(canonical_to_token_mask.shape)} vs {tuple(phrase_to_token_mask.shape)}"
                    )
        canonical_to_token_mask = canonical_to_token_mask & phrase_to_token_mask
        predicate_mask = phrase_to_token_mask & (~canonical_to_token_mask)

        mean_weight = torch.ones_like(phrase_to_token_mask, dtype=token_logits.dtype)
        mean_weight = mean_weight.masked_fill(canonical_to_token_mask, self.canonical_token_weight)
        mean_weight = mean_weight.masked_fill(~phrase_to_token_mask, 0.0)
        valid = mean_weight.sum(dim=-1) > 0
        denom = mean_weight.sum(dim=-1).clamp(min=1e-6)

        logits = torch.nan_to_num(token_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        probability_space = self.phrase_agg in {"prob_mean", "prob_mean_softmin"}
        z = (logits.sigmoid() if probability_space else logits)[:, :, None, :]
        z_mean = (z * mean_weight[:, None, :, :]).sum(dim=-1) / denom[:, None, :]

        use_softmin = self.phrase_agg in {"mean_softmin", "prob_mean_softmin"}
        if not use_softmin:
            if probability_space:
                eps = max(float(torch.finfo(token_logits.dtype).eps), 1e-6)
                return torch.logit(z_mean.clamp(min=eps, max=1.0 - eps)).masked_fill(
                    ~valid[:, None, :], -20.0
                )
            return z_mean.masked_fill(~valid[:, None, :], -20.0)

        tau = max(float(self.phrase_softmin_tau), 1e-6)
        pred_mask = predicate_mask[:, None, :, :]
        n_pred = predicate_mask.sum(dim=-1).clamp_min(1).to(dtype=token_logits.dtype)
        z_softmin = (
            -tau
            * torch.logsumexp(
                (-z / tau).masked_fill(~pred_mask, torch.finfo(token_logits.dtype).min),
                dim=-1,
            )
            + tau * torch.log(n_pred[:, None, :])
        )
        has_predicate = predicate_mask.any(dim=-1)[:, None, :]
        z_softmin = torch.where(has_predicate, z_softmin, z_mean)

        alpha = min(max(float(self.phrase_mean_weight), 0.0), 1.0)
        score = alpha * z_mean + (1.0 - alpha) * z_softmin
        if probability_space:
            eps = max(float(torch.finfo(token_logits.dtype).eps), 1e-6)
            score = torch.logit(score.clamp(min=eps, max=1.0 - eps))
        return score.masked_fill(~valid[:, None, :], -20.0)

    def _pool_phrase_features(
        self,
        encoded_text: torch.Tensor,
        phrase_to_token_mask: Optional[torch.Tensor],
        canonical_to_token_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if phrase_to_token_mask is None:
            return None
        B, T, _D = encoded_text.shape
        phrase_mask = phrase_to_token_mask.to(device=encoded_text.device, dtype=torch.bool)[..., :T]
        if phrase_mask.dim() != 3 or phrase_mask.shape[0] != B or phrase_mask.shape[-1] != T:
            return None
        if canonical_to_token_mask is None:
            canonical_mask = torch.zeros_like(phrase_mask)
        else:
            canonical_mask = canonical_to_token_mask.to(device=encoded_text.device, dtype=torch.bool)[..., :T]
            if canonical_mask.shape != phrase_mask.shape:
                return None
            canonical_mask = canonical_mask & phrase_mask
        weight = torch.ones_like(phrase_mask, dtype=encoded_text.dtype)
        weight = weight.masked_fill(canonical_mask, self.canonical_token_weight)
        weight = weight.masked_fill(~phrase_mask, 0.0)
        denom = weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bkt,btd->bkd", weight / denom, encoded_text)

    def forward(
        self,
        *,
        query_feats: torch.Tensor,
        boxes: torch.Tensor,
        roi_feature_map: torch.Tensor,
        roi_feature_mask: Optional[torch.Tensor] = None,
        predicate_text: List[str],
        phrase_to_token_mask: Optional[torch.Tensor] = None,
        canonical_to_token_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query_feats = query_feats.detach()
        boxes = boxes.detach()
        roi_feature_map = roi_feature_map.detach()
        roi_feats = self.extract_roi_features(
            roi_feature_map,
            boxes,
            feature_mask=roi_feature_mask,
        )
        context_feats = self.extract_roi_features(
            roi_feature_map,
            boxes,
            feature_mask=roi_feature_mask,
            box_scale=self.context_scale,
        )

        candidate = (
            query_feats
            + self.query_delta(query_feats)
            + self.roi_delta(roi_feats)
            + self.context_delta(context_feats)
            + self.box_delta(boxes.to(dtype=query_feats.dtype))
        )

        text_dict = self.encode_text(predicate_text, device=query_feats.device)
        token_logits = self.class_embed(candidate, text_dict)
        token_logits = calibrate_finite_token_logits(token_logits, self.logit_scale, self.logit_bias)
        predicate_logits = self._aggregate_phrase_logits(
            token_logits,
            phrase_to_token_mask=phrase_to_token_mask,
            canonical_to_token_mask=canonical_to_token_mask,
        )
        if self.use_joint_phrase_head:
            phrase_features = self._pool_phrase_features(
                text_dict["encoded_text"],
                phrase_to_token_mask,
                canonical_to_token_mask,
            )
            if phrase_features is not None and phrase_features.shape[1] == predicate_logits.shape[-1]:
                visual = self.phrase_visual_proj(candidate)[:, :, None, :]
                text = self.phrase_text_proj(phrase_features)[:, None, :, :]
                joint = self.phrase_joint_norm(visual + text + visual * text)
                predicate_logits = predicate_logits + self.phrase_score_head(joint).squeeze(-1)

        if patch_mask is not None:
            patch_mask = patch_mask.to(device=predicate_logits.device, dtype=torch.bool)
            if (
                patch_mask.dim() == 2
                and patch_mask.shape[0] == predicate_logits.shape[0]
                and patch_mask.shape[1] == predicate_logits.shape[-1]
            ):
                K = min(int(patch_mask.shape[1]), int(predicate_logits.shape[-1]))
                predicate_logits[:, :, :K] = predicate_logits[:, :, :K].masked_fill(
                    ~patch_mask[:, None, :K],
                    -100.0,
                )
        return {
            "predicate_logits": predicate_logits,
            "predicate_token_logits": token_logits,
            "roi_feats": roi_feats,
            "context_feats": context_feats,
        }

    def score_candidates(
        self,
        predicate_logits: torch.Tensor,
        patch_score: torch.Tensor,
        *,
        pair_stride=None,
    ) -> Dict[str, torch.Tensor]:
        return build_stage_b_v7_candidate_scores(
            predicate_logits,
            patch_score,
            candidate_topk=self.candidate_topk,
            patch_prior_weight=self.patch_prior_weight,
            pair_stride=pair_stride,
        )


class _StageBV7CriterionLegacy(nn.Module):
    def __init__(
        self,
        *,
        patch_criterion,
        min_matched_iou: float = 0.5,
        canonical_token_weight: float = 0.15,
        tn_token_weight: float = 1.0,
        pair_rank_loss_coef: float = 0.0,
        pair_rank_margin: float = 0.18,
        pair_score_tau_pos: float = 0.1,
        pair_score_tau_neg: float = 0.5605,
        pair_pos_weight: float = 0.0,
        pair_neg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_criterion = patch_criterion
        self.matcher = getattr(patch_criterion, "matcher", None)
        self.min_matched_iou = float(min_matched_iou)
        self.canonical_token_weight = float(canonical_token_weight)
        self.tn_token_weight = float(tn_token_weight)
        self.pair_rank_loss_coef = float(pair_rank_loss_coef)
        self.pair_rank_margin = float(pair_rank_margin)
        self.pair_score_tau_pos = float(pair_score_tau_pos)
        self.pair_score_tau_neg = float(pair_score_tau_neg)
        self.pair_pos_weight = float(pair_pos_weight)
        self.pair_neg_weight = float(pair_neg_weight)
        self.weight_dict = {
            "loss_verifier_bce": 1.0,
            "loss_verifier_pair_rank": self.pair_rank_loss_coef,
        }
        if self.pair_pos_weight != 0.0:
            self.weight_dict["loss_verifier_pair_pos"] = self.pair_pos_weight
        if self.pair_neg_weight != 0.0:
            self.weight_dict["loss_verifier_pair_neg"] = self.pair_neg_weight

    def _mask_for_batch(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        key: str,
        batch_idx: int,
        *,
        token_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        value = outputs.get(key, None)
        if value is None and 0 <= batch_idx < len(targets):
            value = targets[batch_idx].get(key, None)
        if not torch.is_tensor(value):
            return None
        if value.dim() == 3:
            value = value[batch_idx]
        elif value.dim() != 2:
            return None
        value = value.to(device=device)
        if dtype is None:
            value = value.to(torch.bool)
        else:
            value = value.to(dtype=dtype)
        if value.shape[-1] != token_len:
            value = value[..., :token_len]
            if value.shape[-1] != token_len:
                raise ValueError(f"{key} token length mismatch: got {tuple(value.shape)}, T={token_len}")
        return value

    def _is_tn_for_slot(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        batch_idx: int,
        slot_idx: int,
        *,
        device: torch.device,
    ) -> bool:
        value = outputs.get("is_tn", None)
        if value is None and 0 <= batch_idx < len(targets):
            value = targets[batch_idx].get("is_tn", None)
        if not torch.is_tensor(value):
            return False
        if value.dim() >= 2:
            value = value[batch_idx]
        value = value.to(device=device, dtype=torch.bool).view(-1)
        if slot_idx < 0 or slot_idx >= int(value.numel()):
            return False
        return bool(value[slot_idx].item())

    def _slot_token_bce(
        self,
        token_logits: torch.Tensor,
        mask: torch.Tensor,
        target_value: float,
        *,
        weight: float = 1.0,
    ) -> Optional[torch.Tensor]:
        if not bool(mask.any().item()):
            return None
        logits = token_logits[mask]
        finite = torch.isfinite(logits)
        if not bool(finite.any().item()):
            return None
        logits = logits[finite]
        labels = torch.full_like(logits, float(target_value))
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return loss.mean() * float(weight)

    @staticmethod
    def _slot_phrase_loss(
        token_logits: torch.Tensor,
        phrase_mask: torch.Tensor,
        target_value: float,
        *,
        canonical_mask: Optional[torch.Tensor] = None,
        canonical_weight: float = 1.0,
    ) -> Optional[torch.Tensor]:
        phrase_mask = phrase_mask.to(torch.bool)
        if not bool(phrase_mask.any().item()):
            return None
        logits = token_logits[phrase_mask]
        finite = torch.isfinite(logits)
        if not bool(finite.any().item()):
            return None
        labels = torch.full_like(logits, float(target_value))
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if canonical_mask is not None and canonical_weight != 1.0:
            canonical_local = canonical_mask.to(device=phrase_mask.device, dtype=torch.bool)[phrase_mask]
            if canonical_local.numel() == loss.numel():
                weight = torch.ones_like(loss)
                weight = weight.masked_fill(canonical_local, float(canonical_weight))
                loss = loss * weight
        return loss.mean()

    def _paired_slots_for_match(
        self,
        is_tn: Optional[torch.Tensor],
        base_slot: int,
        num_slots: int,
        *,
        pair_stride: int = 1,
    ) -> Optional[tuple[int, int]]:
        if is_tn is None:
            return None
        is_tn = is_tn.to(torch.bool).view(-1)
        if int(is_tn.numel()) < 2:
            return None
        if int(pair_stride) > 1:
            pos_slot = int(base_slot) * int(pair_stride)
            neg_slot = pos_slot + 1
            if (
                pos_slot < num_slots
                and neg_slot < num_slots
                and pos_slot < int(is_tn.numel())
                and neg_slot < int(is_tn.numel())
                and (not bool(is_tn[pos_slot].item()))
                and bool(is_tn[neg_slot].item())
            ):
                return pos_slot, neg_slot
            return None
        pos_candidates = torch.nonzero(~is_tn, as_tuple=False).flatten()
        neg_candidates = torch.nonzero(is_tn, as_tuple=False).flatten()
        if int(pos_candidates.numel()) == 0 or int(neg_candidates.numel()) == 0:
            return None
        pos_slot = int(base_slot)
        if pos_slot < 0 or pos_slot >= int(is_tn.numel()) or bool(is_tn[pos_slot].item()):
            pos_slot = int(pos_candidates[0].item())
        neg_slot = int(neg_candidates[0].item())
        if pos_slot >= num_slots or neg_slot >= num_slots:
            return None
        return pos_slot, neg_slot

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        if outputs.get("pred_match_cost", None) is not None or outputs.get("pred_scores_match", None) is not None:
            raise RuntimeError("Stage B v7 matching must use patch/entity logits only.")
        predicate_logits = outputs.get("stage_b_v7_predicate_logits", None)
        if predicate_logits is None:
            raise KeyError("StageBV7Criterion requires outputs['stage_b_v7_predicate_logits'].")
        if predicate_logits.dim() != 3:
            raise ValueError(
                f"stage_b_v7_predicate_logits must be (B,Q,K), got {tuple(predicate_logits.shape)}"
            )
        token_logits = outputs.get("stage_b_v7_predicate_token_logits", None)
        if token_logits is None:
            raise KeyError("StageBV7Criterion requires outputs['stage_b_v7_predicate_token_logits'].")
        if token_logits.dim() != 3:
            raise ValueError(
                f"stage_b_v7_predicate_token_logits must be (B,Q,T), got {tuple(token_logits.shape)}"
            )

        match_outputs = {
            "pred_logits_patch": outputs["pred_logits_patch"],
            "pred_boxes": outputs["pred_boxes"],
        }
        if outputs.get("patch_mask", None) is not None:
            match_outputs["patch_mask"] = outputs["patch_mask"]
        if outputs.get("allow_duplicate_support_classes", None) is not None:
            match_outputs["allow_duplicate_support_classes"] = outputs["allow_duplicate_support_classes"]
        match_ctx = self.patch_criterion.compute_matching(match_outputs, targets)
        pred_boxes = match_ctx["pred_boxes"]
        all_indices = match_ctx["all_indices"]
        matched_slots = match_ctx["matched_patch_idx_list"]
        device = predicate_logits.device
        zero = predicate_logits.sum() * 0.0

        losses: List[torch.Tensor] = []
        iou_values: List[torch.Tensor] = []
        valid_iou_values: List[torch.Tensor] = []
        pred_values: List[torch.Tensor] = []
        entity_values: List[torch.Tensor] = []
        final_values: List[torch.Tensor] = []
        matched_count = 0
        valid_count = 0
        low_iou_count = 0
        pos_token_count = 0
        content_token_count = 0
        canonical_token_count = 0
        tn_token_count = 0
        weighted_tn_token_count = 0.0
        empty_content_mask_count = 0
        empty_tn_mask_count = 0
        pair_count = 0
        pair_rank_losses: List[torch.Tensor] = []
        pair_pos_losses: List[torch.Tensor] = []
        pair_neg_losses: List[torch.Tensor] = []
        pair_pos_scores: List[torch.Tensor] = []
        pair_neg_scores: List[torch.Tensor] = []

        patch_score = outputs.get("patch_score", None)
        if patch_score is not None:
            patch_score = patch_score.to(device=device, dtype=predicate_logits.dtype)
            if patch_score.dim() == 2:
                patch_score = patch_score.unsqueeze(-1)

        for b, ((src_idx, tgt_idx), slot_idx) in enumerate(zip(all_indices, matched_slots)):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            slot_idx = slot_idx.to(device=device, dtype=torch.long)
            matched_count += int(src_idx.numel())

            tgt_boxes = targets[b]["boxes"].to(device=device, dtype=pred_boxes.dtype)
            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[b].index_select(0, src_idx).clamp(0.0, 1.0))
            tgt_xyxy = box_ops.box_cxcywh_to_xyxy(tgt_boxes.index_select(0, tgt_idx).clamp(0.0, 1.0))
            iou_mat, _ = box_ops.box_iou(pred_xyxy, tgt_xyxy)
            iou = torch.diag(iou_mat)
            iou_values.append(iou.detach())
            valid = iou > self.min_matched_iou
            low_iou_count += int((~valid).sum().item())
            if not bool(valid.any().item()):
                continue

            valid_iou_values.append(iou[valid].detach())
            valid_count += int(valid.sum().item())

            T = int(token_logits.shape[-1])
            phrase_to_token_mask = self._mask_for_batch(
                outputs, targets, "phrase_to_token_mask", b, token_len=T, device=device
            )
            canonical_to_token_mask = self._mask_for_batch(
                outputs, targets, "canonical_to_token_mask", b, token_len=T, device=device
            )
            content_to_token_mask = self._mask_for_batch(
                outputs, targets, "content_to_token_mask", b, token_len=T, device=device
            )
            attr_pos_to_token_mask = self._mask_for_batch(
                outputs, targets, "attr_pos_to_token_mask", b, token_len=T, device=device
            )
            attr_neg_to_token_mask = self._mask_for_batch(
                outputs, targets, "attr_neg_to_token_mask", b, token_len=T, device=device
            )
            negative_to_token_mask = self._mask_for_batch(
                outputs, targets, "negative_to_token_mask", b, token_len=T, device=device
            )
            attr_neg_weight_mask = self._mask_for_batch(
                outputs,
                targets,
                "attr_neg_weight_mask",
                b,
                token_len=T,
                device=device,
                dtype=token_logits.dtype,
            )
            if phrase_to_token_mask is None:
                phrase_to_token_mask = torch.ones(
                    (predicate_logits.shape[-1], T), dtype=torch.bool, device=device
                )
            if canonical_to_token_mask is None:
                canonical_to_token_mask = torch.zeros_like(phrase_to_token_mask)
            is_tn_value = outputs.get("is_tn", None)
            if is_tn_value is None and 0 <= b < len(targets):
                is_tn_value = targets[b].get("is_tn", None)
            if torch.is_tensor(is_tn_value):
                if is_tn_value.dim() >= 2:
                    is_tn_for_batch = is_tn_value[b].to(device=device, dtype=torch.bool).view(-1)
                else:
                    is_tn_for_batch = is_tn_value.to(device=device, dtype=torch.bool).view(-1)
            else:
                is_tn_for_batch = None
            pair_stride_value = outputs.get("verifier_pair_stride", None)
            if pair_stride_value is None and 0 <= b < len(targets):
                pair_stride_value = targets[b].get("verifier_pair_stride", None)
            pair_stride = 1
            if torch.is_tensor(pair_stride_value) and pair_stride_value.numel() > 0:
                if pair_stride_value.dim() >= 2:
                    pair_stride_tensor = pair_stride_value[b].detach().view(-1)
                else:
                    pair_stride_tensor = pair_stride_value.detach().view(-1)
                pair_stride = int(pair_stride_tensor[0].item())

            for row_idx, keep in enumerate(valid.tolist()):
                if not bool(keep):
                    continue
                q_idx = int(src_idx[row_idx].item())
                slot = int(slot_idx[row_idx].item())
                if slot < 0 or slot >= int(phrase_to_token_mask.shape[0]):
                    continue
                pair_slots = self._paired_slots_for_match(
                    is_tn_for_batch,
                    base_slot=slot,
                    num_slots=int(phrase_to_token_mask.shape[0]),
                    pair_stride=pair_stride,
                )
                if pair_slots is not None:
                    pos_slot, neg_slot = pair_slots
                    pos_phrase_mask = phrase_to_token_mask[pos_slot].to(torch.bool)
                    pos_canonical_mask = canonical_to_token_mask[pos_slot].to(torch.bool) & pos_phrase_mask
                    neg_phrase_mask = phrase_to_token_mask[neg_slot].to(torch.bool)
                    neg_canonical_mask = canonical_to_token_mask[neg_slot].to(torch.bool) & neg_phrase_mask
                    slot_logits = token_logits[b, q_idx]
                    pos_loss = self._slot_phrase_loss(
                        slot_logits,
                        pos_phrase_mask,
                        1.0,
                        canonical_mask=pos_canonical_mask,
                        canonical_weight=self.canonical_token_weight,
                    )
                    neg_loss = self._slot_phrase_loss(
                        slot_logits,
                        neg_phrase_mask,
                        0.0,
                        canonical_mask=neg_canonical_mask,
                        canonical_weight=self.canonical_token_weight,
                    )
                    slot_had_loss = False
                    pair_bce_terms: List[torch.Tensor] = []
                    if pos_loss is not None:
                        pair_bce_terms.append(pos_loss.reshape(()))
                        count = int(pos_phrase_mask.sum().item())
                        pos_token_count += count
                        content_token_count += max(0, count - int(pos_canonical_mask.sum().item()))
                        canonical_token_count += int(pos_canonical_mask.sum().item())
                        slot_had_loss = True
                    else:
                        empty_content_mask_count += 1
                    if neg_loss is not None:
                        tn_loss = neg_loss * float(max(int(neg_phrase_mask.sum().item()), 1)) * self.tn_token_weight
                        pair_bce_terms.append(tn_loss.reshape(()))
                        count = int(neg_phrase_mask.sum().item())
                        tn_token_count += count
                        weighted_tn_token_count += float(count) * float(max(count, 1)) * float(self.tn_token_weight)
                        slot_had_loss = True
                    else:
                        empty_tn_mask_count += 1
                    if pair_bce_terms:
                        losses.append(torch.stack(pair_bce_terms).sum().reshape(1))
                    if pos_slot < int(predicate_logits.shape[-1]) and neg_slot < int(predicate_logits.shape[-1]):
                        s_pos = predicate_logits[b, q_idx, pos_slot]
                        s_neg = predicate_logits[b, q_idx, neg_slot]
                        pair_rank_losses.append(F.relu(s_neg - s_pos + self.pair_rank_margin).reshape(1))
                        pair_pos_losses.append(F.softplus(self.pair_score_tau_pos - s_pos).reshape(1))
                        pair_neg_losses.append(F.softplus(s_neg - self.pair_score_tau_neg).reshape(1))
                        pair_pos_scores.append(s_pos.sigmoid().detach().reshape(1))
                        pair_neg_scores.append(s_neg.sigmoid().detach().reshape(1))
                        pair_count += 1
                        pred_values.append(s_pos.sigmoid().detach().reshape(1))
                    if slot_had_loss:
                        continue
                phrase_mask = phrase_to_token_mask[slot].to(torch.bool)
                canonical_mask = canonical_to_token_mask[slot].to(torch.bool) & phrase_mask
                if content_to_token_mask is not None:
                    content_mask = content_to_token_mask[slot].to(torch.bool)
                elif attr_pos_to_token_mask is not None:
                    content_mask = attr_pos_to_token_mask[slot].to(torch.bool)
                else:
                    content_mask = phrase_mask

                if attr_neg_to_token_mask is not None:
                    negative_mask = attr_neg_to_token_mask[slot].to(torch.bool)
                elif negative_to_token_mask is not None:
                    negative_mask = negative_to_token_mask[slot].to(torch.bool)
                else:
                    negative_mask = torch.zeros_like(phrase_mask)
                negative_mask = negative_mask & phrase_mask & (~canonical_mask)
                if attr_neg_weight_mask is not None:
                    negative_mask = negative_mask & (attr_neg_weight_mask[slot] > 0)

                content_mask = content_mask & phrase_mask & (~canonical_mask) & (~negative_mask)
                slot_logits = token_logits[b, q_idx]
                slot_had_loss = False

                content_loss = self._slot_token_bce(slot_logits, content_mask, 1.0, weight=1.0)
                if content_loss is not None:
                    losses.append(content_loss.reshape(1))
                    count = int(content_mask.sum().item())
                    content_token_count += count
                    pos_token_count += count
                    slot_had_loss = True
                else:
                    empty_content_mask_count += 1

                canonical_loss = self._slot_token_bce(
                    slot_logits,
                    canonical_mask,
                    1.0,
                    weight=self.canonical_token_weight,
                )
                if canonical_loss is not None:
                    losses.append(canonical_loss.reshape(1))
                    count = int(canonical_mask.sum().item())
                    canonical_token_count += count
                    pos_token_count += count
                    slot_had_loss = True

                is_tn_slot = self._is_tn_for_slot(outputs, targets, b, slot, device=device)
                if is_tn_slot:
                    if not bool(negative_mask.any().item()):
                        empty_tn_mask_count += 1
                    tn_weight = self.tn_token_weight * float(max(int(phrase_mask.sum().item()), 1))
                    tn_loss = self._slot_token_bce(slot_logits, negative_mask, 0.0, weight=tn_weight)
                    if tn_loss is not None:
                        losses.append(tn_loss.reshape(1))
                        count = int(negative_mask.sum().item())
                        tn_token_count += count
                        weighted_tn_token_count += float(count) * float(tn_weight)
                        slot_had_loss = True

                if slot_had_loss:
                    phrase_logit = predicate_logits[b, q_idx, slot]
                    pred_values.append(phrase_logit.sigmoid().detach().reshape(1))
            if patch_score is not None and patch_score.shape[-1] > int(slot_idx.max().item()):
                ent = patch_score[b, src_idx, slot_idx][valid]
                entity_values.append(ent.detach())
                phrase_scores = predicate_logits[b, src_idx, slot_idx][valid].sigmoid()
                final_values.append((ent * phrase_scores).detach())

        if losses:
            loss_verifier_bce = torch.cat([x.reshape(-1) for x in losses]).mean()
        else:
            loss_verifier_bce = zero
        loss_pair_rank = torch.cat(pair_rank_losses).mean() if pair_rank_losses else zero
        loss_pair_pos = torch.cat(pair_pos_losses).mean() if pair_pos_losses else zero
        loss_pair_neg = torch.cat(pair_neg_losses).mean() if pair_neg_losses else zero

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero.detach()
            return torch.cat([v.reshape(-1) for v in values]).mean().detach()

        return {
            "loss_verifier_bce": loss_verifier_bce,
            "loss_verifier_pair_rank": loss_pair_rank,
            "loss_verifier_pair_pos": loss_pair_pos,
            "loss_verifier_pair_neg": loss_pair_neg,
            "stage_b_v7_matched_count": torch.as_tensor(float(matched_count), device=device),
            "stage_b_v7_valid_count": torch.as_tensor(float(valid_count), device=device),
            "stage_b_v7_low_iou_count": torch.as_tensor(float(low_iou_count), device=device),
            "stage_b_v7_pair_count": torch.as_tensor(float(pair_count), device=device),
            "stage_b_v7_pair_rank_loss_raw": loss_pair_rank.detach(),
            "stage_b_v7_pair_pos_loss_raw": loss_pair_pos.detach(),
            "stage_b_v7_pair_neg_loss_raw": loss_pair_neg.detach(),
            "stage_b_v7_pair_pos_score": _mean_or_zero(pair_pos_scores),
            "stage_b_v7_pair_neg_score": _mean_or_zero(pair_neg_scores),
            "stage_b_v7_pair_score_gap": _mean_or_zero(pair_pos_scores) - _mean_or_zero(pair_neg_scores),
            "stage_b_v7_pos_count": torch.as_tensor(float(pos_token_count), device=device),
            "stage_b_v7_tn_count": torch.as_tensor(float(tn_token_count), device=device),
            "stage_b_v7_content_token_count": torch.as_tensor(float(content_token_count), device=device),
            "stage_b_v7_canonical_token_count": torch.as_tensor(float(canonical_token_count), device=device),
            "stage_b_v7_tn_token_count": torch.as_tensor(float(tn_token_count), device=device),
            "stage_b_v7_weighted_tn_token_count": torch.as_tensor(float(weighted_tn_token_count), device=device),
            "stage_b_v7_empty_content_mask_count": torch.as_tensor(float(empty_content_mask_count), device=device),
            "stage_b_v7_empty_tn_mask_count": torch.as_tensor(float(empty_tn_mask_count), device=device),
            "stage_b_v7_matched_iou": _mean_or_zero(iou_values),
            "stage_b_v7_valid_iou": _mean_or_zero(valid_iou_values),
            "stage_b_v7_predicate_score": _mean_or_zero(pred_values),
            "stage_b_v7_entity_score": _mean_or_zero(entity_values),
            "stage_b_v7_final_score": _mean_or_zero(final_values),
            "stage_b_v7_min_matched_iou": torch.as_tensor(float(self.min_matched_iou), device=device),
            "stage_b_v7_canonical_token_weight": torch.as_tensor(float(self.canonical_token_weight), device=device),
            "stage_b_v7_tn_token_weight": torch.as_tensor(float(self.tn_token_weight), device=device),
            "stage_b_v7_pair_rank_weight": torch.as_tensor(float(self.pair_rank_loss_coef), device=device),
            "stage_b_v7_pair_rank_margin": torch.as_tensor(float(self.pair_rank_margin), device=device),
            "stage_b_v7_pair_tau_pos": torch.as_tensor(float(self.pair_score_tau_pos), device=device),
            "stage_b_v7_pair_tau_neg": torch.as_tensor(float(self.pair_score_tau_neg), device=device),
        }


def _sigmoid_focal_loss_no_reduce(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss


class _StageBV7AllQueryCriterion(nn.Module):
    def __init__(
        self,
        *,
        patch_criterion,
        min_matched_iou: float = 0.5,
        canonical_token_weight: float = 0.15,
        tn_token_weight: float = 1.0,
        tn_shared_token_weight: float = 0.25,
        phrase_focal_alpha: float = 0.25,
        phrase_focal_gamma: float = 2.0,
        phrase_focal_coef: float = 1.0,
        token_focal_alpha: float = 0.25,
        token_focal_gamma: float = 2.0,
        token_focal_coef: float = 0.25,
        pair_rank_loss_coef: float = 0.36,
        pair_rank_margin: float = 0.18,
        pair_rank_topk: int = 10,
        pair_rank_lse_tau: float = 0.1,
        tn_pair_rank_loss_coef: float = 0.0,
        tn_pair_rank_margin: float = 0.3,
        tn_pair_rank_topk: int = 10,
        pair_pos_weight: float = 0.0,
        pair_neg_weight: float = 0.0,
        negative_iou_max: float = 0.3,
        phrase_hard_negative_topk: int = 10,
        **_unused,
    ) -> None:
        super().__init__()
        self.patch_criterion = patch_criterion
        self.matcher = getattr(patch_criterion, "matcher", None)
        self.min_matched_iou = float(min_matched_iou)
        self.canonical_token_weight = float(canonical_token_weight)
        self.tn_token_weight = float(tn_token_weight)
        self.tn_shared_token_weight = float(tn_shared_token_weight)
        self.phrase_focal_alpha = float(phrase_focal_alpha)
        self.phrase_focal_gamma = float(phrase_focal_gamma)
        self.phrase_focal_coef = float(phrase_focal_coef)
        self.token_focal_alpha = float(token_focal_alpha)
        self.token_focal_gamma = float(token_focal_gamma)
        self.token_focal_coef = float(token_focal_coef)
        self.pair_rank_loss_coef = float(pair_rank_loss_coef)
        self.pair_rank_margin = float(pair_rank_margin)
        self.pair_rank_topk = int(pair_rank_topk)
        self.pair_rank_lse_tau = float(pair_rank_lse_tau)
        self.tn_pair_rank_loss_coef = float(tn_pair_rank_loss_coef)
        self.tn_pair_rank_margin = float(tn_pair_rank_margin)
        self.tn_pair_rank_topk = int(tn_pair_rank_topk)
        self.pair_pos_weight = float(pair_pos_weight)
        self.pair_neg_weight = float(pair_neg_weight)
        self.negative_iou_max = float(negative_iou_max)
        self.phrase_hard_negative_topk = int(phrase_hard_negative_topk)
        self.weight_dict = {
            "loss_verifier_phrase_focal": self.phrase_focal_coef,
            "loss_verifier_token_focal": self.token_focal_coef,
            "loss_verifier_pair_rank": self.pair_rank_loss_coef,
        }
        if self.tn_pair_rank_loss_coef != 0.0:
            self.weight_dict["loss_verifier_tn_pair_rank"] = self.tn_pair_rank_loss_coef
        if self.pair_pos_weight != 0.0:
            self.weight_dict["loss_verifier_pair_pos"] = self.pair_pos_weight
        if self.pair_neg_weight != 0.0:
            self.weight_dict["loss_verifier_pair_neg"] = self.pair_neg_weight

    def _mask_for_batch(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        key: str,
        batch_idx: int,
        *,
        token_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        value = outputs.get(key, None)
        if value is None and 0 <= batch_idx < len(targets):
            value = targets[batch_idx].get(key, None)
        if not torch.is_tensor(value):
            return None
        if value.dim() == 3:
            value = value[batch_idx]
        elif value.dim() != 2:
            return None
        value = value.to(device=device)
        value = value.to(torch.bool if dtype is None else dtype)
        if value.shape[-1] != token_len:
            value = value[..., :token_len]
            if value.shape[-1] != token_len:
                raise ValueError(f"{key} token length mismatch: got {tuple(value.shape)}, T={token_len}")
        return value

    @staticmethod
    def _vector_for_batch(
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        key: str,
        batch_idx: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        value = outputs.get(key, None)
        if value is None and 0 <= batch_idx < len(targets):
            value = targets[batch_idx].get(key, None)
        if not torch.is_tensor(value):
            return None
        if value.dim() >= 2:
            value = value[batch_idx]
        return value.to(device=device, dtype=dtype).view(-1)

    def _paired_slots_for_match(
        self,
        is_tn: Optional[torch.Tensor],
        base_slot: int,
        num_slots: int,
        *,
        pair_stride: int = 1,
    ) -> Optional[tuple[int, int]]:
        if is_tn is None:
            return None
        is_tn = is_tn.to(torch.bool).view(-1)
        if int(is_tn.numel()) < 2:
            return None
        if int(pair_stride) > 1:
            pos_slot = int(base_slot) * int(pair_stride)
            neg_slot = pos_slot + 1
            if (
                pos_slot < num_slots
                and neg_slot < num_slots
                and pos_slot < int(is_tn.numel())
                and neg_slot < int(is_tn.numel())
                and (not bool(is_tn[pos_slot].item()))
                and bool(is_tn[neg_slot].item())
            ):
                return pos_slot, neg_slot
            return None
        pos_candidates = torch.nonzero(~is_tn, as_tuple=False).flatten()
        neg_candidates = torch.nonzero(is_tn, as_tuple=False).flatten()
        if int(pos_candidates.numel()) == 0 or int(neg_candidates.numel()) == 0:
            return None
        pos_slot = int(base_slot)
        if pos_slot < 0 or pos_slot >= int(is_tn.numel()) or bool(is_tn[pos_slot].item()):
            pos_slot = int(pos_candidates[0].item())
        neg_slot = int(neg_candidates[0].item())
        if pos_slot >= num_slots or neg_slot >= num_slots:
            return None
        return pos_slot, neg_slot

    def _token_focal_mean(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        target_value: float,
        *,
        weight: float,
    ) -> Optional[torch.Tensor]:
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            return None
        selected = logits[mask]
        finite = torch.isfinite(selected)
        if not bool(finite.any().item()):
            return None
        selected = selected[finite]
        targets = torch.full_like(selected, float(target_value))
        return (
            _sigmoid_focal_loss_no_reduce(
                selected,
                targets,
                alpha=self.token_focal_alpha,
                gamma=self.token_focal_gamma,
            ).mean()
            * float(weight)
        )

    def _add_token_loss(
        self,
        bucket: List[torch.Tensor],
        logits: torch.Tensor,
        mask: torch.Tensor,
        target_value: float,
        *,
        weight: float,
    ) -> int:
        loss = self._token_focal_mean(logits, mask, target_value, weight=weight)
        if loss is None:
            return 0
        bucket.append(loss.reshape(1))
        return int(mask.to(torch.bool).sum().item())

    def _final_log_scores(
        self,
        predicate_logits: torch.Tensor,
        patch_score: Optional[torch.Tensor],
        *,
        pair_stride: int = 1,
    ) -> torch.Tensor:
        eps = 1e-6
        if patch_score is None:
            patch_log = torch.zeros_like(predicate_logits)
        else:
            patch_score = patch_score.to(device=predicate_logits.device, dtype=predicate_logits.dtype)
            if patch_score.dim() == 2:
                patch_score = patch_score.unsqueeze(-1)
            if patch_score.shape[-1] != predicate_logits.shape[-1]:
                stride = max(int(pair_stride), 1)
                if patch_score.shape[-1] * stride == predicate_logits.shape[-1]:
                    patch_score = patch_score.repeat_interleave(stride, dim=-1)
                else:
                    patch_score = None
            patch_log = torch.zeros_like(predicate_logits) if patch_score is None else torch.log(patch_score.clamp_min(eps))
        return patch_log + F.logsigmoid(predicate_logits)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        if outputs.get("pred_match_cost", None) is not None or outputs.get("pred_scores_match", None) is not None:
            raise RuntimeError("Stage B v7 matching must use patch/entity logits only.")
        predicate_logits = outputs.get("stage_b_v7_predicate_logits", None)
        if predicate_logits is None or predicate_logits.dim() != 3:
            raise KeyError("StageBV7Criterion requires outputs[stage_b_v7_predicate_logits] as (B,Q,K).")
        token_logits = outputs.get("stage_b_v7_predicate_token_logits", None)
        if token_logits is None or token_logits.dim() != 3:
            raise KeyError("StageBV7Criterion requires outputs[stage_b_v7_predicate_token_logits] as (B,Q,T).")

        match_outputs = {
            "pred_logits_patch": outputs["pred_logits_patch"],
            "pred_boxes": outputs["pred_boxes"],
        }
        if outputs.get("patch_mask", None) is not None:
            match_outputs["patch_mask"] = outputs["patch_mask"]
        if outputs.get("allow_duplicate_support_classes", None) is not None:
            match_outputs["allow_duplicate_support_classes"] = outputs["allow_duplicate_support_classes"]
        match_ctx = self.patch_criterion.compute_matching(match_outputs, targets)
        pred_boxes = match_ctx["pred_boxes"]
        all_indices = match_ctx["all_indices"]
        matched_slots = match_ctx["matched_patch_idx_list"]
        device = predicate_logits.device
        dtype = predicate_logits.dtype
        zero = predicate_logits.sum() * 0.0

        phrase_targets = torch.zeros_like(predicate_logits)
        phrase_supervise = torch.zeros_like(predicate_logits, dtype=torch.bool)
        token_losses: List[torch.Tensor] = []
        pair_rank_losses: List[torch.Tensor] = []
        pair_pos_losses: List[torch.Tensor] = []
        pair_neg_losses: List[torch.Tensor] = []
        iou_values: List[torch.Tensor] = []
        valid_iou_values: List[torch.Tensor] = []
        pair_pos_scores: List[torch.Tensor] = []
        pair_neg_scores: List[torch.Tensor] = []
        entity_values: List[torch.Tensor] = []
        final_values: List[torch.Tensor] = []

        matched_count = 0
        valid_count = 0
        low_iou_count = 0
        pair_count = 0
        phrase_supervise_count = 0
        phrase_positive_count = 0
        token_group_count = 0
        pos_token_count = 0
        content_token_count = 0
        canonical_token_count = 0
        tn_token_count = 0
        tn_shared_token_count = 0
        empty_content_mask_count = 0
        empty_tn_mask_count = 0

        patch_score = outputs.get("patch_score", None)
        if patch_score is not None:
            patch_score = patch_score.to(device=device, dtype=dtype)
            if patch_score.dim() == 2:
                patch_score = patch_score.unsqueeze(-1)
        B, Q, K = predicate_logits.shape
        T = int(token_logits.shape[-1])
        for b, ((src_idx, tgt_idx), slot_idx) in enumerate(zip(all_indices, matched_slots)):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            slot_idx = slot_idx.to(device=device, dtype=torch.long)
            matched_count += int(src_idx.numel())

            tgt_boxes = targets[b]["boxes"].to(device=device, dtype=pred_boxes.dtype)
            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[b].clamp(0.0, 1.0))
            tgt_xyxy_all = box_ops.box_cxcywh_to_xyxy(tgt_boxes.clamp(0.0, 1.0))
            matched_iou_mat, _ = box_ops.box_iou(
                pred_xyxy.index_select(0, src_idx),
                tgt_xyxy_all.index_select(0, tgt_idx),
            )
            matched_iou = torch.diag(matched_iou_mat)
            iou_values.append(matched_iou.detach())
            valid = matched_iou > self.min_matched_iou
            low_iou_count += int((~valid).sum().item())
            if not bool(valid.any().item()):
                continue
            valid_iou_values.append(matched_iou[valid].detach())
            valid_count += int(valid.sum().item())

            phrase_to_token_mask = self._mask_for_batch(
                outputs, targets, "phrase_to_token_mask", b, token_len=T, device=device
            )
            if phrase_to_token_mask is None:
                phrase_to_token_mask = torch.ones((K, T), dtype=torch.bool, device=device)
            canonical_to_token_mask = self._mask_for_batch(
                outputs, targets, "canonical_to_token_mask", b, token_len=T, device=device
            )
            if canonical_to_token_mask is None:
                canonical_to_token_mask = torch.zeros_like(phrase_to_token_mask)
            content_to_token_mask = self._mask_for_batch(
                outputs, targets, "content_to_token_mask", b, token_len=T, device=device
            )
            attr_pos_to_token_mask = self._mask_for_batch(
                outputs, targets, "attr_pos_to_token_mask", b, token_len=T, device=device
            )
            attr_neg_to_token_mask = self._mask_for_batch(
                outputs, targets, "attr_neg_to_token_mask", b, token_len=T, device=device
            )
            negative_to_token_mask = self._mask_for_batch(
                outputs, targets, "negative_to_token_mask", b, token_len=T, device=device
            )
            attr_neg_weight_mask = self._mask_for_batch(
                outputs,
                targets,
                "attr_neg_weight_mask",
                b,
                token_len=T,
                device=device,
                dtype=dtype,
            )
            is_tn_for_batch = self._vector_for_batch(outputs, targets, "is_tn", b, device=device, dtype=torch.bool)
            pair_stride_value = self._vector_for_batch(
                outputs, targets, "verifier_pair_stride", b, device=device, dtype=torch.long
            )
            pair_stride = int(pair_stride_value[0].item()) if pair_stride_value is not None and pair_stride_value.numel() else 1
            log_final_b = self._final_log_scores(
                predicate_logits[b : b + 1],
                None if patch_score is None else patch_score[b : b + 1],
                pair_stride=pair_stride,
            )[0]

            for row_idx, keep in enumerate(valid.tolist()):
                if not bool(keep):
                    continue
                q_idx = int(src_idx[row_idx].item())
                base_slot = int(slot_idx[row_idx].item())
                if base_slot < 0 or base_slot >= K:
                    continue
                pair_slots = self._paired_slots_for_match(
                    is_tn_for_batch,
                    base_slot=base_slot,
                    num_slots=K,
                    pair_stride=pair_stride,
                )
                if pair_slots is None:
                    pair_slots = (base_slot, base_slot)
                pos_slot, neg_slot = pair_slots
                if pos_slot < 0 or pos_slot >= K or neg_slot < 0 or neg_slot >= K:
                    continue

                gt_idx = int(tgt_idx[row_idx].item())
                all_query_iou, _ = box_ops.box_iou(pred_xyxy, tgt_xyxy_all[gt_idx : gt_idx + 1])
                positive_queries = all_query_iou[:, 0] > self.min_matched_iou
                phrase_targets[b, positive_queries, pos_slot] = 1.0
                phrase_supervise[b, :, pos_slot] = True
                phrase_targets[b, :, neg_slot] = 0.0
                phrase_supervise[b, :, neg_slot] = True
                phrase_positive_count += int(positive_queries.sum().item())
                phrase_supervise_count += int(Q * (2 if neg_slot != pos_slot else 1))

                pos_phrase_mask = phrase_to_token_mask[pos_slot].to(torch.bool)
                pos_canonical_mask = canonical_to_token_mask[pos_slot].to(torch.bool) & pos_phrase_mask
                if content_to_token_mask is not None:
                    pos_content_mask = content_to_token_mask[pos_slot].to(torch.bool)
                elif attr_pos_to_token_mask is not None:
                    pos_content_mask = attr_pos_to_token_mask[pos_slot].to(torch.bool)
                else:
                    pos_content_mask = pos_phrase_mask
                pos_content_mask = pos_content_mask & pos_phrase_mask & (~pos_canonical_mask)

                neg_phrase_mask = phrase_to_token_mask[neg_slot].to(torch.bool)
                neg_canonical_mask = canonical_to_token_mask[neg_slot].to(torch.bool) & neg_phrase_mask
                if attr_neg_to_token_mask is not None:
                    neg_negative_mask = attr_neg_to_token_mask[neg_slot].to(torch.bool)
                elif negative_to_token_mask is not None:
                    neg_negative_mask = negative_to_token_mask[neg_slot].to(torch.bool)
                else:
                    neg_negative_mask = torch.zeros_like(neg_phrase_mask)
                neg_negative_mask = neg_negative_mask & neg_phrase_mask & (~neg_canonical_mask)
                if attr_neg_weight_mask is not None:
                    neg_negative_mask = neg_negative_mask & (attr_neg_weight_mask[neg_slot] > 0)
                neg_shared_mask = neg_phrase_mask & (~neg_canonical_mask) & (~neg_negative_mask)

                slot_logits = token_logits[b, q_idx]
                before = len(token_losses)
                count = self._add_token_loss(token_losses, slot_logits, pos_content_mask, 1.0, weight=1.0)
                pos_token_count += count
                content_token_count += count
                count = self._add_token_loss(
                    token_losses,
                    slot_logits,
                    pos_canonical_mask,
                    1.0,
                    weight=self.canonical_token_weight,
                )
                pos_token_count += count
                canonical_token_count += count
                count = self._add_token_loss(
                    token_losses,
                    slot_logits,
                    neg_negative_mask,
                    0.0,
                    weight=self.tn_token_weight,
                )
                tn_token_count += count
                count = self._add_token_loss(
                    token_losses,
                    slot_logits,
                    neg_shared_mask,
                    1.0,
                    weight=self.tn_shared_token_weight,
                )
                tn_shared_token_count += count
                count = self._add_token_loss(
                    token_losses,
                    slot_logits,
                    neg_canonical_mask,
                    1.0,
                    weight=self.canonical_token_weight,
                )
                canonical_token_count += count
                if len(token_losses) == before:
                    empty_content_mask_count += 1
                if not bool(neg_negative_mask.any().item()):
                    empty_tn_mask_count += 1
                token_group_count += len(token_losses) - before

                pos_log_score = log_final_b[q_idx, pos_slot]
                neg_all = log_final_b[:, neg_slot]
                k_top = min(max(int(self.pair_rank_topk), 1), int(neg_all.numel()))
                neg_values = torch.topk(neg_all, k=k_top).values
                tau = max(float(self.pair_rank_lse_tau), 1e-6)
                neg_log_score = tau * torch.logsumexp(neg_values / tau, dim=0) - tau * math.log(float(neg_values.numel()))
                pair_rank_losses.append(F.softplus(neg_log_score - pos_log_score + self.pair_rank_margin).reshape(1))
                pair_pos_scores.append(pos_log_score.exp().detach().reshape(1))
                pair_neg_scores.append(neg_log_score.exp().detach().reshape(1))
                pair_pos_losses.append(F.softplus(-pos_log_score).reshape(1))
                pair_neg_losses.append(F.softplus(neg_log_score).reshape(1))
                pair_count += 1

                if patch_score is not None and patch_score.shape[-1] > base_slot:
                    entity_values.append(patch_score[b, q_idx, base_slot].detach().reshape(1))
                    final_values.append(log_final_b[q_idx, pos_slot].exp().detach().reshape(1))

        phrase_loss_map = _sigmoid_focal_loss_no_reduce(
            predicate_logits,
            phrase_targets,
            alpha=self.phrase_focal_alpha,
            gamma=self.phrase_focal_gamma,
        )
        num_positive_queries = (phrase_targets * phrase_supervise.to(dtype)).sum().clamp_min(1.0)
        loss_phrase_focal = (phrase_loss_map * phrase_supervise.to(dtype)).sum() / num_positive_queries
        loss_token_focal = torch.cat(token_losses).mean() if token_losses else zero
        loss_pair_rank = torch.cat(pair_rank_losses).mean() if pair_rank_losses else zero
        loss_pair_pos = torch.cat(pair_pos_losses).mean() if pair_pos_losses else zero
        loss_pair_neg = torch.cat(pair_neg_losses).mean() if pair_neg_losses else zero

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero.detach()
            return torch.cat([v.reshape(-1) for v in values]).mean().detach()

        return {
            "loss_verifier_phrase_focal": loss_phrase_focal,
            "loss_verifier_token_focal": loss_token_focal,
            "loss_verifier_pair_rank": loss_pair_rank,
            "loss_verifier_pair_pos": loss_pair_pos,
            "loss_verifier_pair_neg": loss_pair_neg,
            "stage_b_v7_matched_count": torch.as_tensor(float(matched_count), device=device),
            "stage_b_v7_valid_count": torch.as_tensor(float(valid_count), device=device),
            "stage_b_v7_low_iou_count": torch.as_tensor(float(low_iou_count), device=device),
            "stage_b_v7_pair_count": torch.as_tensor(float(pair_count), device=device),
            "stage_b_v7_phrase_supervise_count": torch.as_tensor(float(phrase_supervise_count), device=device),
            "stage_b_v7_phrase_positive_count": torch.as_tensor(float(phrase_positive_count), device=device),
            "stage_b_v7_token_group_count": torch.as_tensor(float(token_group_count), device=device),
            "stage_b_v7_pair_rank_loss_raw": loss_pair_rank.detach(),
            "stage_b_v7_pair_pos_loss_raw": loss_pair_pos.detach(),
            "stage_b_v7_pair_neg_loss_raw": loss_pair_neg.detach(),
            "stage_b_v7_pair_pos_score": _mean_or_zero(pair_pos_scores),
            "stage_b_v7_pair_neg_score": _mean_or_zero(pair_neg_scores),
            "stage_b_v7_pair_score_gap": _mean_or_zero(pair_pos_scores) - _mean_or_zero(pair_neg_scores),
            "stage_b_v7_pos_count": torch.as_tensor(float(pos_token_count), device=device),
            "stage_b_v7_tn_count": torch.as_tensor(float(tn_token_count), device=device),
            "stage_b_v7_content_token_count": torch.as_tensor(float(content_token_count), device=device),
            "stage_b_v7_canonical_token_count": torch.as_tensor(float(canonical_token_count), device=device),
            "stage_b_v7_tn_token_count": torch.as_tensor(float(tn_token_count), device=device),
            "stage_b_v7_tn_shared_token_count": torch.as_tensor(float(tn_shared_token_count), device=device),
            "stage_b_v7_weighted_tn_token_count": torch.as_tensor(float(tn_token_count) * float(self.tn_token_weight), device=device),
            "stage_b_v7_empty_content_mask_count": torch.as_tensor(float(empty_content_mask_count), device=device),
            "stage_b_v7_empty_tn_mask_count": torch.as_tensor(float(empty_tn_mask_count), device=device),
            "stage_b_v7_matched_iou": _mean_or_zero(iou_values),
            "stage_b_v7_valid_iou": _mean_or_zero(valid_iou_values),
            "stage_b_v7_predicate_score": predicate_logits.sigmoid().detach().mean(),
            "stage_b_v7_entity_score": _mean_or_zero(entity_values),
            "stage_b_v7_final_score": _mean_or_zero(final_values),
            "stage_b_v7_min_matched_iou": torch.as_tensor(float(self.min_matched_iou), device=device),
            "stage_b_v7_canonical_token_weight": torch.as_tensor(float(self.canonical_token_weight), device=device),
            "stage_b_v7_tn_token_weight": torch.as_tensor(float(self.tn_token_weight), device=device),
            "stage_b_v7_tn_shared_token_weight": torch.as_tensor(float(self.tn_shared_token_weight), device=device),
            "stage_b_v7_phrase_focal_alpha": torch.as_tensor(float(self.phrase_focal_alpha), device=device),
            "stage_b_v7_phrase_focal_gamma": torch.as_tensor(float(self.phrase_focal_gamma), device=device),
            "stage_b_v7_token_focal_weight": torch.as_tensor(float(self.token_focal_coef), device=device),
            "stage_b_v7_pair_rank_weight": torch.as_tensor(float(self.pair_rank_loss_coef), device=device),
            "stage_b_v7_pair_rank_margin": torch.as_tensor(float(self.pair_rank_margin), device=device),
            "stage_b_v7_pair_rank_topk": torch.as_tensor(float(self.pair_rank_topk), device=device),
            "stage_b_v7_pair_rank_lse_tau": torch.as_tensor(float(self.pair_rank_lse_tau), device=device),
        }


class StageBV7Criterion(_StageBV7AllQueryCriterion):
    """Candidate-conditional text reranker; Stage A remains a frozen proposal gate."""

    def _phrase_focal_group(
        self,
        logits: torch.Tensor,
        target_value: float,
        *,
        hard_topk: int = 0,
    ) -> Optional[torch.Tensor]:
        if logits.numel() == 0:
            return None
        targets = torch.full_like(logits, float(target_value))
        losses = _sigmoid_focal_loss_no_reduce(
            logits,
            targets,
            alpha=self.phrase_focal_alpha,
            gamma=self.phrase_focal_gamma,
        ).reshape(-1)
        if int(hard_topk) > 0 and losses.numel() > int(hard_topk):
            losses = torch.topk(losses, k=int(hard_topk), largest=True, sorted=False).values
        return losses.mean()

    @staticmethod
    def _mean_losses(values: List[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
        if not values:
            return zero
        return torch.stack([value.reshape(()) for value in values]).mean()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        if outputs.get("pred_match_cost", None) is not None or outputs.get("pred_scores_match", None) is not None:
            raise RuntimeError("Stage B v7 matching must use patch/entity logits only.")
        predicate_logits = outputs.get("stage_b_v7_predicate_logits", None)
        token_logits = outputs.get("stage_b_v7_predicate_token_logits", None)
        if predicate_logits is None or predicate_logits.dim() != 3:
            raise KeyError("StageBV7Criterion requires stage_b_v7_predicate_logits as (B,Q,K).")
        if token_logits is None or token_logits.dim() != 3:
            raise KeyError("StageBV7Criterion requires stage_b_v7_predicate_token_logits as (B,Q,T).")

        match_outputs = {
            "pred_logits_patch": outputs["pred_logits_patch"],
            "pred_boxes": outputs["pred_boxes"],
        }
        if outputs.get("patch_mask", None) is not None:
            match_outputs["patch_mask"] = outputs["patch_mask"]
        if outputs.get("allow_duplicate_support_classes", None) is not None:
            match_outputs["allow_duplicate_support_classes"] = outputs["allow_duplicate_support_classes"]
        match_ctx = self.patch_criterion.compute_matching(match_outputs, targets)
        pred_boxes = match_ctx["pred_boxes"]
        all_indices = match_ctx["all_indices"]
        matched_slots = match_ctx["matched_patch_idx_list"]

        device = predicate_logits.device
        dtype = predicate_logits.dtype
        zero = predicate_logits.sum() * 0.0
        B, Q, K = predicate_logits.shape
        T = int(token_logits.shape[-1])
        candidate_mask = outputs.get("stage_b_v7_candidate_mask", None)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(predicate_logits, dtype=torch.bool)
        else:
            candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)
            if candidate_mask.shape != predicate_logits.shape:
                raise ValueError(
                    f"stage_b_v7_candidate_mask shape mismatch: {tuple(candidate_mask.shape)} "
                    f"vs {tuple(predicate_logits.shape)}"
                )
        final_logits = outputs.get("stage_b_v7_final_logits", predicate_logits).to(device=device, dtype=dtype)
        if final_logits.shape != predicate_logits.shape:
            raise ValueError("stage_b_v7_final_logits must match predicate logits.")

        expanded_patch_score = outputs.get("stage_b_v7_expanded_patch_score", None)
        if expanded_patch_score is not None:
            expanded_patch_score = expanded_patch_score.to(device=device, dtype=dtype)

        phrase_pos_losses: List[torch.Tensor] = []
        phrase_distractor_losses: List[torch.Tensor] = []
        phrase_tn_losses: List[torch.Tensor] = []
        token_losses: List[torch.Tensor] = []
        pair_rank_losses: List[torch.Tensor] = []
        tn_pair_rank_losses: List[torch.Tensor] = []
        pair_pos_losses: List[torch.Tensor] = []
        pair_neg_losses: List[torch.Tensor] = []
        pair_pos_scores: List[torch.Tensor] = []
        pair_neg_scores: List[torch.Tensor] = []
        tn_pair_pos_scores: List[torch.Tensor] = []
        tn_pair_neg_scores: List[torch.Tensor] = []
        iou_values: List[torch.Tensor] = []
        valid_iou_values: List[torch.Tensor] = []
        entity_values: List[torch.Tensor] = []
        final_values: List[torch.Tensor] = []

        matched_count = 0
        valid_count = 0
        low_iou_count = 0
        candidate_miss_count = 0
        pair_count = 0
        tn_pair_count = 0
        phrase_supervise_count = 0
        phrase_positive_count = 0
        phrase_distractor_count = 0
        phrase_tn_negative_count = 0
        token_group_count = 0
        pos_token_count = 0
        content_token_count = 0
        canonical_token_count = 0
        tn_token_count = 0
        tn_shared_token_count = 0
        empty_content_mask_count = 0
        empty_tn_mask_count = 0

        for b, ((src_idx, tgt_idx), slot_idx) in enumerate(zip(all_indices, matched_slots)):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            slot_idx = slot_idx.to(device=device, dtype=torch.long)
            matched_count += int(src_idx.numel())

            tgt_boxes = targets[b]["boxes"].to(device=device, dtype=pred_boxes.dtype)
            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[b].clamp(0.0, 1.0))
            tgt_xyxy_all = box_ops.box_cxcywh_to_xyxy(tgt_boxes.clamp(0.0, 1.0))
            matched_iou_mat, _ = box_ops.box_iou(
                pred_xyxy.index_select(0, src_idx),
                tgt_xyxy_all.index_select(0, tgt_idx),
            )
            matched_iou = torch.diag(matched_iou_mat)
            iou_values.append(matched_iou.detach())
            low_iou_count += int((matched_iou <= self.min_matched_iou).sum().item())

            phrase_to_token_mask = self._mask_for_batch(
                outputs, targets, "phrase_to_token_mask", b, token_len=T, device=device
            )
            if phrase_to_token_mask is None:
                phrase_to_token_mask = torch.ones((K, T), dtype=torch.bool, device=device)
            canonical_to_token_mask = self._mask_for_batch(
                outputs, targets, "canonical_to_token_mask", b, token_len=T, device=device
            )
            if canonical_to_token_mask is None:
                canonical_to_token_mask = torch.zeros_like(phrase_to_token_mask)
            content_to_token_mask = self._mask_for_batch(
                outputs, targets, "content_to_token_mask", b, token_len=T, device=device
            )
            attr_pos_to_token_mask = self._mask_for_batch(
                outputs, targets, "attr_pos_to_token_mask", b, token_len=T, device=device
            )
            attr_neg_to_token_mask = self._mask_for_batch(
                outputs, targets, "attr_neg_to_token_mask", b, token_len=T, device=device
            )
            negative_to_token_mask = self._mask_for_batch(
                outputs, targets, "negative_to_token_mask", b, token_len=T, device=device
            )
            attr_neg_weight_mask = self._mask_for_batch(
                outputs,
                targets,
                "attr_neg_weight_mask",
                b,
                token_len=T,
                device=device,
                dtype=dtype,
            )
            is_tn_for_batch = self._vector_for_batch(
                outputs, targets, "is_tn", b, device=device, dtype=torch.bool
            )
            pair_stride_value = self._vector_for_batch(
                outputs, targets, "verifier_pair_stride", b, device=device, dtype=torch.long
            )
            pair_stride = (
                int(pair_stride_value[0].item())
                if pair_stride_value is not None and pair_stride_value.numel()
                else 1
            )

            for row_idx in range(int(src_idx.numel())):
                base_slot = int(slot_idx[row_idx].item())
                if base_slot < 0:
                    continue
                pair_slots = self._paired_slots_for_match(
                    is_tn_for_batch,
                    base_slot=base_slot,
                    num_slots=K,
                    pair_stride=pair_stride,
                )
                if pair_slots is None:
                    pos_slot = base_slot
                    neg_slot: Optional[int] = None
                else:
                    pos_slot, neg_slot = pair_slots
                if pos_slot < 0 or pos_slot >= K or (neg_slot is not None and not (0 <= neg_slot < K)):
                    continue

                gt_idx = int(tgt_idx[row_idx].item())
                all_query_iou, _ = box_ops.box_iou(pred_xyxy, tgt_xyxy_all[gt_idx : gt_idx + 1])
                all_query_iou = all_query_iou[:, 0]
                admitted = candidate_mask[b, :, pos_slot]
                positive_queries = admitted & (all_query_iou >= self.min_matched_iou)
                if not bool(positive_queries.any().item()):
                    candidate_miss_count += 1
                    continue
                valid_count += 1
                valid_iou_values.append(all_query_iou[positive_queries].max().detach().reshape(1))

                distractor_queries = admitted & (all_query_iou <= self.negative_iou_max)
                pos_group = self._phrase_focal_group(predicate_logits[b, positive_queries, pos_slot], 1.0)
                if pos_group is not None:
                    phrase_pos_losses.append(pos_group)
                distractor_group = self._phrase_focal_group(
                    predicate_logits[b, distractor_queries, pos_slot],
                    0.0,
                    hard_topk=self.phrase_hard_negative_topk,
                )
                if distractor_group is not None:
                    phrase_distractor_losses.append(distractor_group)
                phrase_positive_count += int(positive_queries.sum().item())
                phrase_distractor_count += int(distractor_queries.sum().item())
                phrase_supervise_count += int(positive_queries.sum().item() + distractor_queries.sum().item())

                if neg_slot is not None:
                    local_tn_queries = positive_queries & candidate_mask[b, :, neg_slot]
                    tn_group = self._phrase_focal_group(predicate_logits[b, local_tn_queries, neg_slot], 0.0)
                    if tn_group is not None:
                        phrase_tn_losses.append(tn_group)
                    phrase_tn_negative_count += int(local_tn_queries.sum().item())
                    phrase_supervise_count += int(local_tn_queries.sum().item())
                else:
                    local_tn_queries = torch.zeros_like(positive_queries)

                positive_indices = torch.nonzero(positive_queries, as_tuple=False).flatten()
                representative_local = torch.argmax(final_logits[b, positive_indices, pos_slot].detach())
                q_idx = int(positive_indices[representative_local].item())
                slot_logits = token_logits[b, q_idx]

                if neg_slot is not None and bool(local_tn_queries.any().item()):
                    paired_pos = final_logits[b, local_tn_queries, pos_slot]
                    paired_neg = final_logits[b, local_tn_queries, neg_slot]
                    violations = paired_neg - paired_pos
                    topk = min(max(self.tn_pair_rank_topk, 1), int(violations.numel()))
                    hard_violations = torch.topk(
                        violations, k=topk, largest=True, sorted=False
                    ).values
                    tn_pair_rank_losses.append(
                        F.softplus(hard_violations + self.tn_pair_rank_margin).mean()
                    )
                    representative_tn = torch.argmax(paired_pos.detach())
                    tn_pair_pos_scores.append(
                        paired_pos[representative_tn].sigmoid().detach().reshape(1)
                    )
                    tn_pair_neg_scores.append(
                        paired_neg[representative_tn].sigmoid().detach().reshape(1)
                    )
                    tn_pair_count += 1

                pos_phrase_mask = phrase_to_token_mask[pos_slot].to(torch.bool)
                pos_canonical_mask = canonical_to_token_mask[pos_slot].to(torch.bool) & pos_phrase_mask
                if content_to_token_mask is not None:
                    pos_content_mask = content_to_token_mask[pos_slot].to(torch.bool)
                elif attr_pos_to_token_mask is not None:
                    pos_content_mask = attr_pos_to_token_mask[pos_slot].to(torch.bool)
                else:
                    pos_content_mask = pos_phrase_mask
                pos_content_mask = pos_content_mask & pos_phrase_mask & (~pos_canonical_mask)

                before = len(token_losses)
                count = self._add_token_loss(token_losses, slot_logits, pos_content_mask, 1.0, weight=1.0)
                pos_token_count += count
                content_token_count += count
                count = self._add_token_loss(
                    token_losses,
                    slot_logits,
                    pos_canonical_mask,
                    1.0,
                    weight=self.canonical_token_weight,
                )
                pos_token_count += count
                canonical_token_count += count
                if len(token_losses) == before:
                    empty_content_mask_count += 1

                if neg_slot is not None:
                    neg_phrase_mask = phrase_to_token_mask[neg_slot].to(torch.bool)
                    neg_canonical_mask = canonical_to_token_mask[neg_slot].to(torch.bool) & neg_phrase_mask
                    if attr_neg_to_token_mask is not None:
                        neg_negative_mask = attr_neg_to_token_mask[neg_slot].to(torch.bool)
                    elif negative_to_token_mask is not None:
                        neg_negative_mask = negative_to_token_mask[neg_slot].to(torch.bool)
                    else:
                        neg_negative_mask = torch.zeros_like(neg_phrase_mask)
                    neg_negative_mask = neg_negative_mask & neg_phrase_mask & (~neg_canonical_mask)
                    if attr_neg_weight_mask is not None:
                        neg_negative_mask = neg_negative_mask & (attr_neg_weight_mask[neg_slot] > 0)
                    neg_shared_mask = neg_phrase_mask & (~neg_canonical_mask) & (~neg_negative_mask)
                    count = self._add_token_loss(
                        token_losses,
                        slot_logits,
                        neg_negative_mask,
                        0.0,
                        weight=self.tn_token_weight,
                    )
                    tn_token_count += count
                    count = self._add_token_loss(
                        token_losses,
                        slot_logits,
                        neg_shared_mask,
                        1.0,
                        weight=self.tn_shared_token_weight,
                    )
                    tn_shared_token_count += count
                    count = self._add_token_loss(
                        token_losses,
                        slot_logits,
                        neg_canonical_mask,
                        1.0,
                        weight=self.canonical_token_weight,
                    )
                    canonical_token_count += count
                    if not bool(neg_negative_mask.any().item()):
                        empty_tn_mask_count += 1
                token_group_count += len(token_losses) - before

                pos_score = final_logits[b, positive_queries, pos_slot].max()
                negative_scores: List[torch.Tensor] = []
                if bool(distractor_queries.any().item()):
                    negative_scores.append(final_logits[b, distractor_queries, pos_slot])
                if negative_scores:
                    negative_values = torch.cat(negative_scores)
                    topk = min(max(self.pair_rank_topk, 1), int(negative_values.numel()))
                    hard_values = torch.topk(negative_values, k=topk, largest=True, sorted=False).values
                    tau = max(float(self.pair_rank_lse_tau), 1e-6)
                    hard_negative = tau * torch.logsumexp(hard_values / tau, dim=0) - tau * math.log(
                        float(hard_values.numel())
                    )
                    pair_rank_losses.append(F.softplus(hard_negative - pos_score + self.pair_rank_margin))
                    pair_pos_losses.append(F.softplus(-pos_score))
                    pair_neg_losses.append(F.softplus(hard_negative))
                    pair_pos_scores.append(pos_score.sigmoid().detach().reshape(1))
                    pair_neg_scores.append(hard_negative.sigmoid().detach().reshape(1))
                    pair_count += 1

                if expanded_patch_score is not None:
                    entity_values.append(expanded_patch_score[b, q_idx, pos_slot].detach().reshape(1))
                final_values.append(pos_score.sigmoid().detach().reshape(1))

        loss_phrase_pos = self._mean_losses(phrase_pos_losses, zero)
        loss_phrase_distractor = self._mean_losses(phrase_distractor_losses, zero)
        loss_phrase_tn = self._mean_losses(phrase_tn_losses, zero)
        loss_phrase_focal = loss_phrase_pos + loss_phrase_distractor + loss_phrase_tn
        loss_token_focal = self._mean_losses(token_losses, zero)
        loss_pair_rank = self._mean_losses(pair_rank_losses, zero)
        loss_tn_pair_rank = self._mean_losses(tn_pair_rank_losses, zero)
        loss_pair_pos = self._mean_losses(pair_pos_losses, zero)
        loss_pair_neg = self._mean_losses(pair_neg_losses, zero)

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero.detach()
            return torch.cat([value.reshape(-1) for value in values]).mean().detach()

        candidate_scores = predicate_logits.sigmoid().detach()[candidate_mask]
        candidate_recall = float(valid_count) / float(max(matched_count, 1))
        return {
            "loss_verifier_phrase_focal": loss_phrase_focal,
            "loss_verifier_token_focal": loss_token_focal,
            "loss_verifier_pair_rank": loss_pair_rank,
            "loss_verifier_tn_pair_rank": loss_tn_pair_rank,
            "loss_verifier_pair_pos": loss_pair_pos,
            "loss_verifier_pair_neg": loss_pair_neg,
            "stage_b_v7_loss_phrase_pos": loss_phrase_pos.detach(),
            "stage_b_v7_loss_phrase_distractor": loss_phrase_distractor.detach(),
            "stage_b_v7_loss_phrase_tn": loss_phrase_tn.detach(),
            "stage_b_v7_matched_count": torch.as_tensor(float(matched_count), device=device),
            "stage_b_v7_valid_count": torch.as_tensor(float(valid_count), device=device),
            "stage_b_v7_low_iou_count": torch.as_tensor(float(low_iou_count), device=device),
            "stage_b_v7_candidate_miss_count": torch.as_tensor(float(candidate_miss_count), device=device),
            "stage_b_v7_candidate_recall": torch.as_tensor(candidate_recall, device=device),
            "stage_b_v7_candidate_count": candidate_mask.sum().detach().to(dtype=torch.float32),
            "stage_b_v7_pair_count": torch.as_tensor(float(pair_count), device=device),
            "stage_b_v7_tn_pair_count": torch.as_tensor(float(tn_pair_count), device=device),
            "stage_b_v7_phrase_supervise_count": torch.as_tensor(float(phrase_supervise_count), device=device),
            "stage_b_v7_phrase_positive_count": torch.as_tensor(float(phrase_positive_count), device=device),
            "stage_b_v7_phrase_distractor_count": torch.as_tensor(float(phrase_distractor_count), device=device),
            "stage_b_v7_phrase_tn_negative_count": torch.as_tensor(float(phrase_tn_negative_count), device=device),
            "stage_b_v7_token_group_count": torch.as_tensor(float(token_group_count), device=device),
            "stage_b_v7_pair_rank_loss_raw": loss_pair_rank.detach(),
            "stage_b_v7_tn_pair_rank_loss_raw": loss_tn_pair_rank.detach(),
            "stage_b_v7_pair_pos_loss_raw": loss_pair_pos.detach(),
            "stage_b_v7_pair_neg_loss_raw": loss_pair_neg.detach(),
            "stage_b_v7_pair_pos_score": _mean_or_zero(pair_pos_scores),
            "stage_b_v7_pair_neg_score": _mean_or_zero(pair_neg_scores),
            "stage_b_v7_pair_score_gap": _mean_or_zero(pair_pos_scores) - _mean_or_zero(pair_neg_scores),
            "stage_b_v7_tn_pair_pos_score": _mean_or_zero(tn_pair_pos_scores),
            "stage_b_v7_tn_pair_neg_score": _mean_or_zero(tn_pair_neg_scores),
            "stage_b_v7_tn_pair_score_gap": _mean_or_zero(tn_pair_pos_scores)
            - _mean_or_zero(tn_pair_neg_scores),
            "stage_b_v7_pos_count": torch.as_tensor(float(pos_token_count), device=device),
            "stage_b_v7_tn_count": torch.as_tensor(float(tn_token_count), device=device),
            "stage_b_v7_content_token_count": torch.as_tensor(float(content_token_count), device=device),
            "stage_b_v7_canonical_token_count": torch.as_tensor(float(canonical_token_count), device=device),
            "stage_b_v7_tn_token_count": torch.as_tensor(float(tn_token_count), device=device),
            "stage_b_v7_tn_shared_token_count": torch.as_tensor(float(tn_shared_token_count), device=device),
            "stage_b_v7_weighted_tn_token_count": torch.as_tensor(
                float(tn_token_count) * float(self.tn_token_weight), device=device
            ),
            "stage_b_v7_empty_content_mask_count": torch.as_tensor(float(empty_content_mask_count), device=device),
            "stage_b_v7_empty_tn_mask_count": torch.as_tensor(float(empty_tn_mask_count), device=device),
            "stage_b_v7_matched_iou": _mean_or_zero(iou_values),
            "stage_b_v7_valid_iou": _mean_or_zero(valid_iou_values),
            "stage_b_v7_predicate_score": (
                candidate_scores.mean() if candidate_scores.numel() else zero.detach()
            ),
            "stage_b_v7_entity_score": _mean_or_zero(entity_values),
            "stage_b_v7_final_score": _mean_or_zero(final_values),
            "stage_b_v7_min_matched_iou": torch.as_tensor(float(self.min_matched_iou), device=device),
            "stage_b_v7_negative_iou_max": torch.as_tensor(float(self.negative_iou_max), device=device),
            "stage_b_v7_canonical_token_weight": torch.as_tensor(float(self.canonical_token_weight), device=device),
            "stage_b_v7_tn_token_weight": torch.as_tensor(float(self.tn_token_weight), device=device),
            "stage_b_v7_tn_shared_token_weight": torch.as_tensor(float(self.tn_shared_token_weight), device=device),
            "stage_b_v7_phrase_focal_alpha": torch.as_tensor(float(self.phrase_focal_alpha), device=device),
            "stage_b_v7_phrase_focal_gamma": torch.as_tensor(float(self.phrase_focal_gamma), device=device),
            "stage_b_v7_token_focal_weight": torch.as_tensor(float(self.token_focal_coef), device=device),
            "stage_b_v7_pair_rank_weight": torch.as_tensor(float(self.pair_rank_loss_coef), device=device),
            "stage_b_v7_pair_rank_margin": torch.as_tensor(float(self.pair_rank_margin), device=device),
            "stage_b_v7_pair_rank_topk": torch.as_tensor(float(self.pair_rank_topk), device=device),
            "stage_b_v7_pair_rank_lse_tau": torch.as_tensor(float(self.pair_rank_lse_tau), device=device),
            "stage_b_v7_tn_pair_rank_weight": torch.as_tensor(
                float(self.tn_pair_rank_loss_coef), device=device
            ),
            "stage_b_v7_tn_pair_rank_margin": torch.as_tensor(
                float(self.tn_pair_rank_margin), device=device
            ),
            "stage_b_v7_tn_pair_rank_topk": torch.as_tensor(
                float(self.tn_pair_rank_topk), device=device
            ),
        }
