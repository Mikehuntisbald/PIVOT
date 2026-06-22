from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from groundingdino.util import box_ops

from .bertwarper import generate_masks_with_special_tokens_and_transfer_map
from .utils import ContrastiveEmbed, MLP


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
        self.use_neighbor_geometry = bool(use_neighbor_geometry)
        if self.use_neighbor_geometry:
            raise NotImplementedError("stage_b_v7_use_neighbor_geometry is reserved for a later ablation.")

        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        self.class_embed = ContrastiveEmbed(max_text_len=self.max_text_len)
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.roi_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.box_proj = MLP(4, self.hidden_dim, self.hidden_dim, 2)
        self.candidate_norm = nn.LayerNorm(self.hidden_dim)

        self.freeze_bert()

    @classmethod
    def from_groundingdino(
        cls,
        model,
        *,
        canonical_token_weight: float = 0.15,
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

    def extract_roi_features(self, feature_map: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(f"feature_map must be (B,C,H,W), got {tuple(feature_map.shape)}")
        if boxes.dim() != 3 or boxes.shape[-1] != 4:
            raise ValueError(f"boxes must be (B,Q,4), got {tuple(boxes.shape)}")
        B, Q, _ = boxes.shape
        _b, _c, H, W = feature_map.shape
        if _b != B:
            raise ValueError(f"feature_map/boxes batch mismatch: {tuple(feature_map.shape)} vs {tuple(boxes.shape)}")

        xyxy = box_ops.box_cxcywh_to_xyxy(boxes.clamp(0.0, 1.0))
        scale = xyxy.new_tensor([float(W), float(H), float(W), float(H)])
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

        weight = torch.ones_like(phrase_to_token_mask, dtype=token_logits.dtype)
        weight = weight.masked_fill(canonical_to_token_mask, self.canonical_token_weight)
        weight = weight.masked_fill(~phrase_to_token_mask, 0.0)
        valid = weight.sum(dim=-1) > 0
        denom = weight.sum(dim=-1).clamp(min=1e-6)
        logits = torch.nan_to_num(token_logits, nan=0.0, posinf=0.0, neginf=0.0)
        score = (logits[:, :, None, :] * weight[:, None, :, :]).sum(dim=-1) / denom[:, None, :]
        return score.masked_fill(~valid[:, None, :], -100.0)

    def forward(
        self,
        *,
        query_feats: torch.Tensor,
        boxes: torch.Tensor,
        roi_feature_map: torch.Tensor,
        predicate_text: List[str],
        phrase_to_token_mask: Optional[torch.Tensor] = None,
        canonical_to_token_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query_feats = query_feats.detach()
        boxes = boxes.detach()
        roi_feature_map = roi_feature_map.detach()
        roi_feats = self.extract_roi_features(roi_feature_map, boxes)

        candidate = (
            self.query_proj(query_feats)
            + self.roi_proj(roi_feats)
            + self.box_proj(boxes.to(dtype=query_feats.dtype))
        )
        candidate = self.candidate_norm(F.gelu(candidate))

        text_dict = self.encode_text(predicate_text, device=query_feats.device)
        token_logits = self.class_embed(candidate, text_dict)
        predicate_logits = self._aggregate_phrase_logits(
            token_logits,
            phrase_to_token_mask=phrase_to_token_mask,
            canonical_to_token_mask=canonical_to_token_mask,
        )

        if patch_mask is not None:
            patch_mask = patch_mask.to(device=predicate_logits.device, dtype=torch.bool)
            if patch_mask.dim() == 2 and patch_mask.shape[0] == predicate_logits.shape[0]:
                K = min(int(patch_mask.shape[1]), int(predicate_logits.shape[-1]))
                predicate_logits[:, :, :K] = predicate_logits[:, :, :K].masked_fill(
                    ~patch_mask[:, None, :K],
                    -100.0,
                )
        return {
            "predicate_logits": predicate_logits,
            "predicate_token_logits": token_logits,
            "roi_feats": roi_feats,
        }


class StageBV7Criterion(nn.Module):
    def __init__(
        self,
        *,
        patch_criterion,
        min_matched_iou: float = 0.5,
        canonical_token_weight: float = 0.15,
        tn_token_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.patch_criterion = patch_criterion
        self.matcher = getattr(patch_criterion, "matcher", None)
        self.min_matched_iou = float(min_matched_iou)
        self.canonical_token_weight = float(canonical_token_weight)
        self.tn_token_weight = float(tn_token_weight)
        self.weight_dict = {"loss_verifier_bce": 1.0}

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

            for row_idx, keep in enumerate(valid.tolist()):
                if not bool(keep):
                    continue
                q_idx = int(src_idx[row_idx].item())
                slot = int(slot_idx[row_idx].item())
                if slot < 0 or slot >= int(phrase_to_token_mask.shape[0]):
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

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero.detach()
            return torch.cat([v.reshape(-1) for v in values]).mean().detach()

        return {
            "loss_verifier_bce": loss_verifier_bce,
            "stage_b_v7_matched_count": torch.as_tensor(float(matched_count), device=device),
            "stage_b_v7_valid_count": torch.as_tensor(float(valid_count), device=device),
            "stage_b_v7_low_iou_count": torch.as_tensor(float(low_iou_count), device=device),
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
        }
