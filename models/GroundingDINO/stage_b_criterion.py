from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from groundingdino.util import box_ops

from .patch_hungarian_criterion import PatchHungarianCriterion
from .stage_b_score import compute_stage_b_slot_logits
from .stage_b_score import compute_stage_b_slot_match_cost


_TN_GROUP_NAMES = ("color_like", "attr_like", "spatial_like", "relation_action_like", "other")
_TN_GROUP_TO_ID = {name: idx for idx, name in enumerate(_TN_GROUP_NAMES)}
_TN_ID_TO_GROUP = {idx: name for name, idx in _TN_GROUP_TO_ID.items()}


def _tn_group_name_from_id(group_id: int) -> str:
    return _TN_ID_TO_GROUP.get(int(group_id), "other")


def _sigmoid_focal_loss_no_reduce(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


class StageBCriterion(nn.Module):
    """
    Stage B criterion:
      - keeps Stage A patch-only matching / patch loss intact
      - adds a text-only token loss on matched (query, slot) pairs
    """

    def __init__(
        self,
        *,
        patch_criterion: PatchHungarianCriterion,
        lambda_patch: float = 1.0,
        lambda_text: float = 0.25,
        canonical_pos_weight: float = 1.0,
        stage_b_text_loss_type: str = "matched_bce",
        stage_b_text_focal_alpha: float = 0.25,
        stage_b_text_focal_gamma: float = 2.0,
        stage_b_extra_iou_match_thr: float = 0.5,
        stage_b_tn_neg_weight: float = 1.0,
        stage_b_tn_content_weight: float = 1.0,
        stage_b_tn_canonical_weight: float = 1.0,
        stage_b_tn_neg_weight_mode: str = "fixed",
        stage_b_tn_content_target: float = 1.0,
        stage_b_tn_canonical_target: float = 1.0,
        stage_b_rank_margin: float = 0.3,
        stage_b_rank_loss_coef: float = 0.0,
        stage_b_rank_detach_patch: bool = True,
        stage_b_rank_beta: float = 1.0,
        stage_b_rank_canonical_weight: float = 1.0,
        stage_b_rank_text_agg: str = "mean",
        stage_b_rank_softmin_tau: float = 0.7,
        stage_b_rank_mean_softmin_alpha: float = 0.5,
        stage_b_score_calib_loss_coef: float = 0.0,
        stage_b_score_calib_tau_pos: float = 0.1,
        stage_b_score_calib_tau_neg: float = 1.4,
        stage_b_score_calib_margin: float = 0.3,
        stage_b_score_calib_topk: int = 10,
        stage_b_score_calib_pos_weight: float = 0.1,
        stage_b_score_calib_neg_weight: float = 0.5,
        stage_b_score_calib_gap_weight: float = 0.1,
        stage_b_score_calib_pos_query_weight: float = 0.1,
        stage_b_score_calib_all_tn_neg_weight: float = 0.0,
        stage_b_score_calib_detach_patch: bool = True,
        stage_b_score_calib_neg_agg: str = "mean",
        stage_b_score_calib_neg_lse_tau: float = 0.5,
        stage_b_score_calib_aux_loss: bool = False,
        stage_b_aux_loss_start_idx: int = 0,
        # Deprecated compatibility args. Softmin phrase TN loss is disabled.
        attr_pos_weight: Optional[float] = None,
        tn_shared_attr_pos_weight: Optional[float] = None,
        attr_neg_weight: Optional[float] = None,
        use_phrase_tn_loss: Optional[bool] = None,
        phrase_score_type: Optional[str] = None,
        softmin_tau: Optional[float] = None,
        lambda_phrase: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.patch_criterion = patch_criterion
        self.matcher = getattr(patch_criterion, "matcher", None)
        self.lambda_patch = float(lambda_patch)
        self.lambda_text = float(lambda_text)
        self.canonical_pos_weight = float(canonical_pos_weight)
        self.stage_b_text_loss_type = str(stage_b_text_loss_type).lower().replace("-", "_").strip()
        if self.stage_b_text_loss_type not in {
            "matched_bce",
            "allquery_focal",
            "allquery_focal_tn_matched_bce",
            "allquery_focal_tn_empty_det",
        }:
            raise ValueError(
                "stage_b_text_loss_type must be 'matched_bce', 'allquery_focal', "
                "'allquery_focal_tn_matched_bce', or 'allquery_focal_tn_empty_det', "
                f"got {stage_b_text_loss_type!r}"
            )
        self.stage_b_text_focal_alpha = float(stage_b_text_focal_alpha)
        self.stage_b_text_focal_gamma = float(stage_b_text_focal_gamma)
        self.stage_b_extra_iou_match_thr = float(stage_b_extra_iou_match_thr)
        self.stage_b_tn_neg_weight = float(stage_b_tn_neg_weight)
        self.stage_b_tn_content_weight = float(stage_b_tn_content_weight)
        self.stage_b_tn_canonical_weight = float(stage_b_tn_canonical_weight)
        self.stage_b_tn_neg_weight_mode = str(stage_b_tn_neg_weight_mode).lower().strip()
        if self.stage_b_tn_neg_weight_mode not in {"fixed", "token_count"}:
            raise ValueError(
                "stage_b_tn_neg_weight_mode must be 'fixed' or 'token_count', "
                f"got {stage_b_tn_neg_weight_mode!r}"
            )
        self.stage_b_tn_content_target = float(stage_b_tn_content_target)
        self.stage_b_tn_canonical_target = float(stage_b_tn_canonical_target)
        self.stage_b_rank_margin = float(stage_b_rank_margin)
        self.stage_b_rank_loss_coef = float(stage_b_rank_loss_coef)
        self.stage_b_rank_detach_patch = bool(stage_b_rank_detach_patch)
        self.stage_b_rank_beta = float(stage_b_rank_beta)
        self.stage_b_rank_canonical_weight = float(stage_b_rank_canonical_weight)
        self.stage_b_rank_text_agg = str(stage_b_rank_text_agg)
        self.stage_b_rank_softmin_tau = float(stage_b_rank_softmin_tau)
        self.stage_b_rank_mean_softmin_alpha = float(stage_b_rank_mean_softmin_alpha)
        self.stage_b_score_calib_loss_coef = float(stage_b_score_calib_loss_coef)
        self.stage_b_score_calib_tau_pos = float(stage_b_score_calib_tau_pos)
        self.stage_b_score_calib_tau_neg = float(stage_b_score_calib_tau_neg)
        self.stage_b_score_calib_margin = float(stage_b_score_calib_margin)
        self.stage_b_score_calib_topk = max(1, int(stage_b_score_calib_topk))
        self.stage_b_score_calib_pos_weight = float(stage_b_score_calib_pos_weight)
        self.stage_b_score_calib_neg_weight = float(stage_b_score_calib_neg_weight)
        self.stage_b_score_calib_gap_weight = float(stage_b_score_calib_gap_weight)
        self.stage_b_score_calib_pos_query_weight = float(stage_b_score_calib_pos_query_weight)
        self.stage_b_score_calib_all_tn_neg_weight = float(stage_b_score_calib_all_tn_neg_weight)
        self.stage_b_score_calib_detach_patch = bool(stage_b_score_calib_detach_patch)
        self.stage_b_score_calib_neg_agg = str(stage_b_score_calib_neg_agg).lower().strip()
        if self.stage_b_score_calib_neg_agg not in {"mean", "max", "logsumexp", "lse"}:
            raise ValueError(
                "stage_b_score_calib_neg_agg must be 'mean', 'max', or 'logsumexp', "
                f"got {stage_b_score_calib_neg_agg!r}"
            )
        self.stage_b_score_calib_neg_lse_tau = max(float(stage_b_score_calib_neg_lse_tau), 1e-6)
        self.stage_b_score_calib_aux_loss = bool(stage_b_score_calib_aux_loss)
        self.stage_b_aux_loss_start_idx = max(0, int(stage_b_aux_loss_start_idx))
        patch_weight_dict = getattr(patch_criterion, "weight_dict", {}) or {}
        self.weight_dict = {
            "loss_patch_ce": float(lambda_patch),
            "loss_text": float(lambda_text),
            "loss_phrase_rank": float(stage_b_rank_loss_coef),
            "loss_score_calib": float(stage_b_score_calib_loss_coef),
            "loss_bbox": float(patch_weight_dict.get("loss_bbox", 0.0)),
            "loss_giou": float(patch_weight_dict.get("loss_giou", 0.0)),
        }
        self.weight_dict.update(
            {
                f"{key}_{i}": value
                for i in range(5)
                if i >= self.stage_b_aux_loss_start_idx
                for key, value in (
                    ("loss_patch_ce", float(lambda_patch)),
                    ("loss_text", float(lambda_text)),
                    ("loss_bbox", float(patch_weight_dict.get("loss_bbox", 0.0))),
                    ("loss_giou", float(patch_weight_dict.get("loss_giou", 0.0))),
                )
            }
        )
        if self.stage_b_score_calib_aux_loss:
            self.weight_dict.update(
                {
                    f"loss_score_calib_{i}": float(stage_b_score_calib_loss_coef)
                    for i in range(5)
                    if i >= self.stage_b_aux_loss_start_idx
                }
            )

    def compute_matching(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        matching_outputs = dict(outputs)
        if outputs.get("pred_logits_text", None) is not None and outputs.get("phrase_to_token_mask", None) is not None:
            with torch.no_grad():
                matching_outputs["pred_match_cost"] = compute_stage_b_slot_match_cost(
                    outputs,
                    beta=self.stage_b_rank_beta,
                    canonical_weight=1.0,
                    focal_alpha=self.stage_b_text_focal_alpha,
                    focal_gamma=self.stage_b_text_focal_gamma,
                    detach_patch=False,
                )
        return self.patch_criterion.compute_matching(matching_outputs, targets)

    def _target_tn_mask(self, target: Dict[str, torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        is_tn = target.get("is_tn", None)
        if not torch.is_tensor(is_tn):
            return None
        return is_tn.to(device=device, dtype=torch.bool).view(-1)

    def _tn_neg_effective_weight(self, token_count: int) -> float:
        if self.stage_b_tn_neg_weight_mode == "token_count":
            return float(max(int(token_count), 1))
        return 1.0

    def _is_tn_target_index(self, target: Dict[str, torch.Tensor], target_idx: int, device: torch.device) -> bool:
        tn_mask = self._target_tn_mask(target, device)
        if tn_mask is None or tn_mask.numel() == 0:
            return False
        if tn_mask.numel() == 1:
            return bool(tn_mask[0].item())
        labels = target.get("labels", None)
        support_classes = target.get("support_classes", None)
        if support_classes is None:
            support_classes = target.get("support_class", None)
        if torch.is_tensor(labels) and torch.is_tensor(support_classes) and 0 <= target_idx < int(labels.numel()):
            label = int(labels.to(device=device, dtype=torch.long).view(-1)[target_idx].item())
            support_classes = support_classes.to(device=device, dtype=torch.long).view(-1)
            matches = torch.nonzero(support_classes == label, as_tuple=False).flatten()
            if matches.numel() > 0:
                slot = int(matches[0].item())
                if slot < int(tn_mask.numel()):
                    return bool(tn_mask[slot].item())
        if 0 <= target_idx < int(tn_mask.numel()):
            return bool(tn_mask[target_idx].item())
        return bool(tn_mask.any().item())

    def _compute_det_box_losses(self, outputs, targets, match_ctx, *, suffix: str = "") -> Dict[str, torch.Tensor]:
        pred_boxes = outputs.get("pred_boxes", None)
        if pred_boxes is None:
            raise KeyError("StageBCriterion requires outputs['pred_boxes'] for bbox/GIoU loss.")
        device = pred_boxes.device
        zero = pred_boxes.sum() * 0.0
        if self.weight_dict.get(f"loss_bbox{suffix}", self.weight_dict.get("loss_bbox", 0.0)) <= 0 and self.weight_dict.get(
            f"loss_giou{suffix}", self.weight_dict.get("loss_giou", 0.0)
        ) <= 0:
            return {f"loss_bbox{suffix}": zero, f"loss_giou{suffix}": zero}

        src_boxes_list: List[torch.Tensor] = []
        tgt_boxes_list: List[torch.Tensor] = []
        det_box_count = 0
        tn_box_filtered_count = 0
        all_indices = match_ctx["all_indices"]
        for b, (src_idx, tgt_idx) in enumerate(all_indices):
            if src_idx.numel() == 0:
                continue
            boxes_b = targets[b]["boxes"].to(device=device, dtype=pred_boxes.dtype)
            keep_src = []
            keep_tgt = []
            for q_t, tgt_t in zip(src_idx.tolist(), tgt_idx.tolist()):
                tgt_i = int(tgt_t)
                if self._is_tn_target_index(targets[b], tgt_i, device):
                    tn_box_filtered_count += 1
                    continue
                if tgt_i < 0 or tgt_i >= int(boxes_b.shape[0]):
                    continue
                keep_src.append(int(q_t))
                keep_tgt.append(tgt_i)
            if not keep_src:
                continue
            src_t = torch.as_tensor(keep_src, dtype=torch.long, device=device)
            tgt_t = torch.as_tensor(keep_tgt, dtype=torch.long, device=device)
            src_boxes_list.append(pred_boxes[b].index_select(0, src_t))
            tgt_boxes_list.append(boxes_b.index_select(0, tgt_t))
            det_box_count += int(src_t.numel())

        if not src_boxes_list:
            out = {f"loss_bbox{suffix}": zero, f"loss_giou{suffix}": zero}
        else:
            src_boxes = torch.cat(src_boxes_list, dim=0)
            tgt_boxes = torch.cat(tgt_boxes_list, dim=0)
            num_boxes = torch.as_tensor([det_box_count], dtype=torch.float32, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(num_boxes)
                world_size = dist.get_world_size()
            else:
                world_size = 1
            num_boxes = torch.clamp(num_boxes / max(1, world_size), min=1.0).item()
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
            loss_giou = 1.0 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                    box_ops.box_cxcywh_to_xyxy(tgt_boxes),
                )
            )
            out = {f"loss_bbox{suffix}": loss_bbox, f"loss_giou{suffix}": loss_giou.sum() / num_boxes}
        if not suffix:
            out["stageb_det_box_count"] = torch.as_tensor(float(det_box_count), device=device)
            out["stageb_tn_box_filtered_count"] = torch.as_tensor(float(tn_box_filtered_count), device=device)
        return out

    def _slot_bce_mean(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        target_value: float,
        weight: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not mask.any():
            return None
        token_logits = logits[mask]
        if token_logits.numel() == 0:
            return None
        finite = torch.isfinite(token_logits)
        if not finite.any():
            return None
        token_logits = token_logits[finite]
        target = torch.full_like(token_logits, float(target_value))
        loss = F.binary_cross_entropy_with_logits(token_logits, target, reduction="none")
        if weight is not None:
            token_weight = weight[mask].to(dtype=loss.dtype, device=loss.device)[finite]
            positive_weight = token_weight > 0
            if not positive_weight.any():
                return None
            loss = loss[positive_weight] * token_weight[positive_weight]
        return loss.mean()

    def _validate_stage_b_token_masks(
        self,
        *,
        pred_logits_text: torch.Tensor,
        target: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        device = pred_logits_text.device
        if "phrase_to_token_mask" not in target:
            raise KeyError("Stage B targets must include 'phrase_to_token_mask'.")
        if "canonical_to_token_mask" not in target:
            raise KeyError("Stage B targets must include 'canonical_to_token_mask'.")

        phrase_to_token_mask = target["phrase_to_token_mask"].to(device=device).to(torch.bool)
        canonical_to_token_mask = target["canonical_to_token_mask"].to(device=device).to(torch.bool)
        content_to_token_mask = target.get("content_to_token_mask", None)
        attr_pos_to_token_mask = target.get("attr_pos_to_token_mask", None)
        attr_neg_to_token_mask = target.get("attr_neg_to_token_mask", None)
        attr_neg_weight_mask = target.get("attr_neg_weight_mask", None)
        is_tn_mask = target.get("is_tn", None)
        negative_to_token_mask = target.get("negative_to_token_mask", None)
        tn_group_ids = target.get("tn_group_ids", None)

        if content_to_token_mask is not None:
            content_to_token_mask = content_to_token_mask.to(device=device).to(torch.bool)
        if attr_pos_to_token_mask is not None:
            attr_pos_to_token_mask = attr_pos_to_token_mask.to(device=device).to(torch.bool)
        if attr_neg_to_token_mask is not None:
            attr_neg_to_token_mask = attr_neg_to_token_mask.to(device=device).to(torch.bool)
        if attr_neg_weight_mask is not None:
            attr_neg_weight_mask = attr_neg_weight_mask.to(device=device, dtype=pred_logits_text.dtype)
        if is_tn_mask is not None:
            is_tn_mask = is_tn_mask.to(device=device).to(torch.bool)
        if negative_to_token_mask is not None:
            negative_to_token_mask = negative_to_token_mask.to(device=device).to(torch.bool)
        if tn_group_ids is not None:
            tn_group_ids = tn_group_ids.to(device=device).to(torch.long)

        T = int(pred_logits_text.shape[-1])
        if phrase_to_token_mask.dim() != 2 or canonical_to_token_mask.dim() != 2:
            raise ValueError("Stage B token masks must be shaped (K,T).")
        if phrase_to_token_mask.shape[-1] != T or canonical_to_token_mask.shape[-1] != T:
            raise ValueError(
                f"Token mask length mismatch for batch {batch_idx}: "
                f"phrase={tuple(phrase_to_token_mask.shape)} "
                f"canonical={tuple(canonical_to_token_mask.shape)} "
                f"pred_logits_text={tuple(pred_logits_text.shape)}"
            )
        optional_masks = {
            "content_to_token_mask": content_to_token_mask,
            "attr_pos_to_token_mask": attr_pos_to_token_mask,
            "attr_neg_to_token_mask": attr_neg_to_token_mask,
            "attr_neg_weight_mask": attr_neg_weight_mask,
            "negative_to_token_mask": negative_to_token_mask,
        }
        for name, mask in optional_masks.items():
            if mask is None:
                continue
            if mask.dim() != 2 or mask.shape != phrase_to_token_mask.shape:
                raise ValueError(
                    f"{name} must match phrase_to_token_mask, got "
                    f"{tuple(mask.shape)} vs {tuple(phrase_to_token_mask.shape)}"
                )
        if is_tn_mask is not None and (is_tn_mask.dim() != 1 or is_tn_mask.shape[0] != phrase_to_token_mask.shape[0]):
            raise ValueError(
                f"is_tn must be shaped (K,), got {tuple(is_tn_mask.shape)} for K={phrase_to_token_mask.shape[0]}"
            )
        if tn_group_ids is not None and (tn_group_ids.dim() != 1 or tn_group_ids.shape[0] != phrase_to_token_mask.shape[0]):
            raise ValueError(
                f"tn_group_ids must be shaped (K,), got {tuple(tn_group_ids.shape)} for K={phrase_to_token_mask.shape[0]}"
            )
        return {
            "phrase_to_token_mask": phrase_to_token_mask,
            "canonical_to_token_mask": canonical_to_token_mask,
            "content_to_token_mask": content_to_token_mask,
            "attr_pos_to_token_mask": attr_pos_to_token_mask,
            "attr_neg_to_token_mask": attr_neg_to_token_mask,
            "attr_neg_weight_mask": attr_neg_weight_mask,
            "is_tn_mask": is_tn_mask,
            "negative_to_token_mask": negative_to_token_mask,
            "tn_group_ids": tn_group_ids,
        }

    def _compute_matched_bce_text_loss(self, outputs, targets, match_ctx):
        pred_logits_text = outputs.get("pred_logits_text", None)
        if pred_logits_text is None:
            raise KeyError("StageBCriterion requires outputs['pred_logits_text'] in Stage B.")
        if pred_logits_text.dim() != 3:
            raise ValueError(
                f"outputs['pred_logits_text'] must be (B,Q,T), got {tuple(pred_logits_text.shape)}"
            )

        device = pred_logits_text.device
        zero = pred_logits_text.new_zeros(())
        content_pos_slot_losses: List[torch.Tensor] = []
        tn_neg_slot_losses: List[torch.Tensor] = []
        canonical_slot_losses: List[torch.Tensor] = []
        tn_group_slot_losses: Dict[str, List[torch.Tensor]] = {name: [] for name in _TN_GROUP_NAMES}
        tn_group_token_counts = {name: 0 for name in _TN_GROUP_NAMES}
        tn_group_nonempty_counts = {name: 0 for name in _TN_GROUP_NAMES}
        tn_group_slot_counts = {name: 0 for name in _TN_GROUP_NAMES}
        valid_slot_count = 0
        empty_content_mask_count = 0
        empty_tn_negative_mask_count = 0
        effective_content_token_count = 0
        effective_tn_negative_token_count = 0

        all_indices = match_ctx["all_indices"]
        matched_patch_idx_list = match_ctx["matched_patch_idx_list"]

        for b, ((src_idx, _tgt_idx), matched_patch_idx) in enumerate(zip(all_indices, matched_patch_idx_list)):
            if src_idx.numel() == 0:
                continue
            masks = self._validate_stage_b_token_masks(
                pred_logits_text=pred_logits_text,
                target=targets[b],
                batch_idx=b,
            )
            phrase_to_token_mask = masks["phrase_to_token_mask"]
            canonical_to_token_mask = masks["canonical_to_token_mask"]
            content_to_token_mask = masks["content_to_token_mask"]
            attr_pos_to_token_mask = masks["attr_pos_to_token_mask"]
            attr_neg_to_token_mask = masks["attr_neg_to_token_mask"]
            attr_neg_weight_mask = masks["attr_neg_weight_mask"]
            is_tn_mask = masks["is_tn_mask"]
            negative_to_token_mask = masks["negative_to_token_mask"]
            tn_group_ids = masks["tn_group_ids"]

            logits_b = pred_logits_text[b, src_idx]
            for row_idx, slot_idx in enumerate(matched_patch_idx.tolist()):
                slot_idx = int(slot_idx)
                if slot_idx < 0 or slot_idx >= int(phrase_to_token_mask.shape[0]):
                    continue

                phrase_mask = phrase_to_token_mask[slot_idx]
                canonical_mask = canonical_to_token_mask[slot_idx] & phrase_mask
                is_tn_slot = bool(is_tn_mask[slot_idx].item()) if is_tn_mask is not None else False
                tn_group_name = (
                    _tn_group_name_from_id(int(tn_group_ids[slot_idx].item()))
                    if tn_group_ids is not None
                    else "other"
                )
                if is_tn_slot:
                    tn_group_slot_counts[tn_group_name] += 1

                if attr_neg_to_token_mask is not None:
                    negative_attr_mask = attr_neg_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)
                elif negative_to_token_mask is not None:
                    negative_attr_mask = negative_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)
                else:
                    negative_attr_mask = torch.zeros_like(phrase_mask)

                if attr_neg_weight_mask is not None:
                    negative_weight = attr_neg_weight_mask[slot_idx].to(dtype=logits_b.dtype) * negative_attr_mask.to(dtype=logits_b.dtype)
                else:
                    negative_weight = negative_attr_mask.to(dtype=logits_b.dtype)
                effective_negative_mask = negative_attr_mask & (negative_weight > 0)

                if is_tn_slot and not effective_negative_mask.any():
                    empty_tn_negative_mask_count += 1

                if content_to_token_mask is not None:
                    content_pos_mask = content_to_token_mask[slot_idx]
                elif attr_pos_to_token_mask is not None:
                    content_pos_mask = attr_pos_to_token_mask[slot_idx]
                else:
                    content_pos_mask = phrase_mask
                content_pos_mask = content_pos_mask & phrase_mask & (~canonical_mask) & (~effective_negative_mask)
                if not content_pos_mask.any():
                    empty_content_mask_count += 1

                slot_logits = logits_b[row_idx]
                slot_had_loss = False

                content_pos_loss = self._slot_bce_mean(slot_logits, content_pos_mask, 1.0)
                if content_pos_loss is not None:
                    content_pos_slot_losses.append(content_pos_loss)
                    effective_content_token_count += int(content_pos_mask.sum().item())
                    slot_had_loss = True

                neg_loss = self._slot_bce_mean(
                    slot_logits,
                    effective_negative_mask,
                    0.0,
                )
                if neg_loss is not None:
                    neg_token_count = int(effective_negative_mask.sum().item())
                    phrase_token_count = int(phrase_mask.sum().item())
                    neg_loss = (
                        neg_loss
                        * self.stage_b_tn_neg_weight
                        * self._tn_neg_effective_weight(phrase_token_count)
                    )
                    tn_neg_slot_losses.append(neg_loss)
                    tn_group_slot_losses[tn_group_name].append(neg_loss)
                    effective_tn_negative_token_count += neg_token_count
                    tn_group_token_counts[tn_group_name] += neg_token_count
                    tn_group_nonempty_counts[tn_group_name] += 1
                    slot_had_loss = True

                canonical_loss = self._slot_bce_mean(slot_logits, canonical_mask, 1.0)
                if canonical_loss is not None:
                    canonical_slot_losses.append(canonical_loss * self.canonical_pos_weight)
                    slot_had_loss = True

                if slot_had_loss:
                    valid_slot_count += 1

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero
            return torch.stack(values).mean()

        content_pos_loss = _mean_or_zero(content_pos_slot_losses)
        tn_neg_loss = _mean_or_zero(tn_neg_slot_losses)
        canonical_loss = _mean_or_zero(canonical_slot_losses)
        token_loss = (
            content_pos_loss
            + tn_neg_loss
            + canonical_loss
        )
        loss_text = token_loss
        metrics = {
            "loss_text": loss_text,
            "text_token_loss_raw": token_loss.detach(),
            "text_phrase_tn_loss_raw": zero.detach(),
            "content_pos_loss": content_pos_loss.detach(),
            "canonical_loss": canonical_loss.detach(),
            "tn_neg_loss": tn_neg_loss.detach(),
            "text_valid_slot_count": torch.as_tensor(float(valid_slot_count), device=device),
            "text_content_pos_slot_count": torch.as_tensor(float(len(content_pos_slot_losses)), device=device),
            "text_tn_neg_slot_count": torch.as_tensor(float(len(tn_neg_slot_losses)), device=device),
            "text_attr_pos_slot_count": torch.as_tensor(float(len(content_pos_slot_losses)), device=device),
            "text_attr_neg_slot_count": torch.as_tensor(float(len(tn_neg_slot_losses)), device=device),
            "text_canonical_slot_count": torch.as_tensor(float(len(canonical_slot_losses)), device=device),
            "text_phrase_tn_slot_count": zero.detach(),
            "text_skipped_tn_slot_count": torch.as_tensor(float(empty_tn_negative_mask_count), device=device),
            "effective_content_token_count": torch.as_tensor(float(effective_content_token_count), device=device),
            "effective_tn_negative_token_count": torch.as_tensor(float(effective_tn_negative_token_count), device=device),
            "empty_content_mask_count": torch.as_tensor(float(empty_content_mask_count), device=device),
            "empty_tn_negative_mask_count": torch.as_tensor(float(empty_tn_negative_mask_count), device=device),
            "text_tn_neg_weight": torch.as_tensor(float(self.stage_b_tn_neg_weight), device=device),
            "text_tn_neg_weight_mode_token_count": torch.as_tensor(
                float(self.stage_b_tn_neg_weight_mode == "token_count"), device=device
            ),
            "text_tn_content_weight": torch.as_tensor(float(self.stage_b_tn_content_weight), device=device),
            "text_tn_canonical_weight": torch.as_tensor(float(self.stage_b_tn_canonical_weight), device=device),
            "spatial_like_tn_count": torch.as_tensor(float(tn_group_slot_counts["spatial_like"]), device=device),
            "relation_action_like_tn_count": torch.as_tensor(
                float(tn_group_slot_counts["relation_action_like"]), device=device
            ),
        }
        for group_name in _TN_GROUP_NAMES:
            group_loss = _mean_or_zero(tn_group_slot_losses[group_name])
            metrics[f"loss_tn_{group_name}"] = group_loss.detach()
            metrics[f"tn_neg_count_{group_name}"] = torch.as_tensor(
                float(tn_group_token_counts[group_name]), device=device
            )
            metrics[f"tn_nonempty_mask_count_{group_name}"] = torch.as_tensor(
                float(tn_group_nonempty_counts[group_name]), device=device
            )
        return metrics

    def _compute_allquery_focal_text_loss(self, outputs, targets, match_ctx):
        pred_logits_text = outputs.get("pred_logits_text", None)
        if pred_logits_text is None:
            raise KeyError("StageBCriterion requires outputs['pred_logits_text'] in Stage B.")
        if pred_logits_text.dim() != 3:
            raise ValueError(
                f"outputs['pred_logits_text'] must be (B,Q,T), got {tuple(pred_logits_text.shape)}"
            )

        device = pred_logits_text.device
        target_map = torch.zeros_like(pred_logits_text, dtype=pred_logits_text.dtype, device=device)
        weight_map = torch.ones_like(pred_logits_text, dtype=pred_logits_text.dtype, device=device)
        text_mask = outputs.get("text_mask", None)
        if text_mask is None:
            valid_text_mask = torch.ones(
                pred_logits_text.shape[0],
                pred_logits_text.shape[-1],
                dtype=torch.bool,
                device=device,
            )
        else:
            valid_text_mask = text_mask.to(device=device, dtype=torch.bool)
            if valid_text_mask.dim() != 2 or valid_text_mask.shape[0] != pred_logits_text.shape[0]:
                raise ValueError(
                    f"outputs['text_mask'] must be (B,T), got {tuple(valid_text_mask.shape)}"
                )
            if valid_text_mask.shape[-1] != pred_logits_text.shape[-1]:
                valid_text_mask = valid_text_mask[:, : pred_logits_text.shape[-1]]
                if valid_text_mask.shape[-1] != pred_logits_text.shape[-1]:
                    raise ValueError(
                        f"outputs['text_mask'] length mismatch: {tuple(valid_text_mask.shape)} "
                        f"vs pred_logits_text={tuple(pred_logits_text.shape)}"
                    )

        zero = pred_logits_text.new_zeros(())
        content_pos_count = 0
        tn_neg_count = 0
        canonical_pos_count = 0
        matched_query_count = 0
        matched_slot_count = 0
        empty_content_mask_count = 0
        empty_tn_negative_mask_count = 0
        tn_group_token_counts = {name: 0 for name in _TN_GROUP_NAMES}
        tn_group_nonempty_counts = {name: 0 for name in _TN_GROUP_NAMES}
        tn_group_slot_counts = {name: 0 for name in _TN_GROUP_NAMES}

        all_indices = match_ctx["all_indices"]
        matched_patch_idx_list = match_ctx["matched_patch_idx_list"]
        for b, ((src_idx, _tgt_idx), matched_patch_idx) in enumerate(zip(all_indices, matched_patch_idx_list)):
            if src_idx.numel() == 0:
                continue
            masks = self._validate_stage_b_token_masks(
                pred_logits_text=pred_logits_text,
                target=targets[b],
                batch_idx=b,
            )
            phrase_to_token_mask = masks["phrase_to_token_mask"]
            canonical_to_token_mask = masks["canonical_to_token_mask"]
            content_to_token_mask = masks["content_to_token_mask"]
            attr_pos_to_token_mask = masks["attr_pos_to_token_mask"]
            attr_neg_to_token_mask = masks["attr_neg_to_token_mask"]
            attr_neg_weight_mask = masks["attr_neg_weight_mask"]
            is_tn_mask = masks["is_tn_mask"]
            negative_to_token_mask = masks["negative_to_token_mask"]
            tn_group_ids = masks["tn_group_ids"]

            for query_idx_t, slot_idx_t in zip(src_idx.tolist(), matched_patch_idx.tolist()):
                query_idx = int(query_idx_t)
                slot_idx = int(slot_idx_t)
                if slot_idx < 0 or slot_idx >= int(phrase_to_token_mask.shape[0]):
                    continue

                phrase_mask = phrase_to_token_mask[slot_idx] & valid_text_mask[b]
                canonical_mask = canonical_to_token_mask[slot_idx] & phrase_mask
                is_tn_slot = bool(is_tn_mask[slot_idx].item()) if is_tn_mask is not None else False
                tn_group_name = (
                    _tn_group_name_from_id(int(tn_group_ids[slot_idx].item()))
                    if tn_group_ids is not None
                    else "other"
                )
                if is_tn_slot:
                    tn_group_slot_counts[tn_group_name] += 1

                if attr_neg_to_token_mask is not None:
                    negative_attr_mask = attr_neg_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)
                elif negative_to_token_mask is not None:
                    negative_attr_mask = negative_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)
                else:
                    negative_attr_mask = torch.zeros_like(phrase_mask)

                if attr_neg_weight_mask is not None:
                    negative_weight = (
                        attr_neg_weight_mask[slot_idx].to(dtype=pred_logits_text.dtype)
                        * negative_attr_mask.to(dtype=pred_logits_text.dtype)
                    )
                    weight_map[b, query_idx, negative_attr_mask] = negative_weight[negative_attr_mask]
                else:
                    negative_weight = negative_attr_mask.to(dtype=pred_logits_text.dtype)
                effective_negative_mask = negative_attr_mask & (negative_weight > 0)
                if is_tn_slot and effective_negative_mask.any():
                    phrase_token_count = int(phrase_mask.sum().item())
                    weight_map[b, query_idx, effective_negative_mask] = torch.maximum(
                        weight_map[b, query_idx, effective_negative_mask],
                        torch.as_tensor(
                            self.stage_b_tn_neg_weight * self._tn_neg_effective_weight(phrase_token_count),
                            dtype=weight_map.dtype,
                            device=device,
                        ),
                    )
                if is_tn_slot and not effective_negative_mask.any():
                    empty_tn_negative_mask_count += 1

                if content_to_token_mask is not None:
                    content_pos_mask = content_to_token_mask[slot_idx]
                elif attr_pos_to_token_mask is not None:
                    content_pos_mask = attr_pos_to_token_mask[slot_idx]
                else:
                    content_pos_mask = phrase_mask
                content_pos_mask = content_pos_mask & phrase_mask & (~canonical_mask) & (~effective_negative_mask)
                if not content_pos_mask.any():
                    empty_content_mask_count += 1

                target_map[b, query_idx, content_pos_mask] = 1.0
                target_map[b, query_idx, canonical_mask] = 1.0
                if canonical_mask.any() and self.canonical_pos_weight != 1.0:
                    weight_map[b, query_idx, canonical_mask] = float(self.canonical_pos_weight)

                content_pos_count += int(content_pos_mask.sum().item())
                canonical_pos_count += int(canonical_mask.sum().item())
                neg_token_count = int(effective_negative_mask.sum().item())
                tn_neg_count += neg_token_count
                if neg_token_count > 0:
                    tn_group_token_counts[tn_group_name] += neg_token_count
                    tn_group_nonempty_counts[tn_group_name] += 1
                matched_query_count += 1
                matched_slot_count += 1

        expanded_text_mask = valid_text_mask[:, None, :].expand_as(pred_logits_text)
        valid_logits = pred_logits_text[expanded_text_mask]
        valid_targets = target_map[expanded_text_mask]
        valid_weights = weight_map[expanded_text_mask]
        if valid_logits.numel() == 0:
            loss_text = zero
        else:
            valid_loss = _sigmoid_focal_loss_no_reduce(
                valid_logits,
                valid_targets,
                alpha=self.stage_b_text_focal_alpha,
                gamma=self.stage_b_text_focal_gamma,
            )
            valid_loss = valid_loss * valid_weights
            normalizer = max(float(matched_query_count), 1.0)
            loss_text = valid_loss.sum() / normalizer

        positive_token_count = int((target_map[expanded_text_mask] > 0.5).sum().item())
        valid_token_count = int(expanded_text_mask.sum().item())
        negative_token_count = max(valid_token_count - positive_token_count, 0)
        metrics = {
            "loss_text": loss_text,
            "text_token_loss_raw": loss_text.detach(),
            "text_phrase_tn_loss_raw": zero.detach(),
            "content_pos_loss": loss_text.detach(),
            "canonical_loss": zero.detach(),
            "tn_neg_loss": zero.detach(),
            "text_valid_slot_count": torch.as_tensor(float(matched_slot_count), device=device),
            "text_content_pos_slot_count": torch.as_tensor(float(matched_query_count), device=device),
            "text_tn_neg_slot_count": torch.as_tensor(float(tn_neg_count), device=device),
            "text_attr_pos_slot_count": torch.as_tensor(float(matched_query_count), device=device),
            "text_attr_neg_slot_count": torch.as_tensor(float(tn_neg_count), device=device),
            "text_canonical_slot_count": torch.as_tensor(float(canonical_pos_count), device=device),
            "text_phrase_tn_slot_count": zero.detach(),
            "text_skipped_tn_slot_count": torch.as_tensor(float(empty_tn_negative_mask_count), device=device),
            "effective_content_token_count": torch.as_tensor(float(content_pos_count), device=device),
            "effective_tn_negative_token_count": torch.as_tensor(float(tn_neg_count), device=device),
            "empty_content_mask_count": torch.as_tensor(float(empty_content_mask_count), device=device),
            "empty_tn_negative_mask_count": torch.as_tensor(float(empty_tn_negative_mask_count), device=device),
            "text_tn_neg_weight": torch.as_tensor(float(self.stage_b_tn_neg_weight), device=device),
            "text_tn_neg_weight_mode_token_count": torch.as_tensor(
                float(self.stage_b_tn_neg_weight_mode == "token_count"), device=device
            ),
            "text_tn_content_weight": torch.as_tensor(float(self.stage_b_tn_content_weight), device=device),
            "text_tn_canonical_weight": torch.as_tensor(float(self.stage_b_tn_canonical_weight), device=device),
            "spatial_like_tn_count": torch.as_tensor(float(tn_group_slot_counts["spatial_like"]), device=device),
            "relation_action_like_tn_count": torch.as_tensor(
                float(tn_group_slot_counts["relation_action_like"]), device=device
            ),
            "text_allquery_focal_loss_raw": loss_text.detach(),
            "text_allquery_positive_token_count": torch.as_tensor(float(positive_token_count), device=device),
            "text_allquery_negative_token_count": torch.as_tensor(float(negative_token_count), device=device),
            "text_allquery_valid_token_count": torch.as_tensor(float(valid_token_count), device=device),
            "text_allquery_matched_query_count": torch.as_tensor(float(matched_query_count), device=device),
            "text_allquery_focal_alpha": torch.as_tensor(float(self.stage_b_text_focal_alpha), device=device),
            "text_allquery_focal_gamma": torch.as_tensor(float(self.stage_b_text_focal_gamma), device=device),
        }
        for group_name in _TN_GROUP_NAMES:
            metrics[f"loss_tn_{group_name}"] = zero.detach()
            metrics[f"tn_neg_count_{group_name}"] = torch.as_tensor(
                float(tn_group_token_counts[group_name]), device=device
            )
            metrics[f"tn_nonempty_mask_count_{group_name}"] = torch.as_tensor(
                float(tn_group_nonempty_counts[group_name]), device=device
            )
        return metrics

    def _get_valid_text_mask(self, outputs, pred_logits_text: torch.Tensor) -> torch.Tensor:
        device = pred_logits_text.device
        text_mask = outputs.get("text_mask", None)
        if text_mask is None:
            return torch.ones(
                pred_logits_text.shape[0],
                pred_logits_text.shape[-1],
                dtype=torch.bool,
                device=device,
            )
        valid_text_mask = text_mask.to(device=device, dtype=torch.bool)
        if valid_text_mask.dim() != 2 or valid_text_mask.shape[0] != pred_logits_text.shape[0]:
            raise ValueError(f"outputs['text_mask'] must be (B,T), got {tuple(valid_text_mask.shape)}")
        if valid_text_mask.shape[-1] != pred_logits_text.shape[-1]:
            valid_text_mask = valid_text_mask[:, : pred_logits_text.shape[-1]]
            if valid_text_mask.shape[-1] != pred_logits_text.shape[-1]:
                raise ValueError(
                    f"outputs['text_mask'] length mismatch: {tuple(valid_text_mask.shape)} "
                    f"vs pred_logits_text={tuple(pred_logits_text.shape)}"
                )
        return valid_text_mask

    def _target_patch_slots_from_labels(self, target, match_ctx, batch_idx: int, device: torch.device) -> Dict[int, int]:
        labels = target["labels"].to(device=device, dtype=torch.long)
        K = int(match_ctx["K"])
        patch_mask = match_ctx.get("patch_mask", None)
        support_classes = self.patch_criterion._get_support_classes(target, K=K, device=device)
        valid_k = support_classes >= 0
        if patch_mask is not None:
            valid_k = valid_k & patch_mask[batch_idx].to(device=device, dtype=torch.bool)
        cid_to_slot = {}
        for slot_idx, cid_t in enumerate(support_classes.tolist()):
            if not bool(valid_k[slot_idx].item()):
                continue
            cid_to_slot[int(cid_t)] = int(slot_idx)
        return {idx: cid_to_slot[int(cid.item())] for idx, cid in enumerate(labels) if int(cid.item()) in cid_to_slot}

    def _iter_stage_b_supervised_pairs(self, outputs, targets, match_ctx):
        pred_boxes = outputs.get("pred_boxes", None)
        if pred_boxes is None:
            raise KeyError("StageBCriterion requires outputs['pred_boxes'] for v5 IoU-augmented text loss.")
        device = pred_boxes.device
        Q = int(match_ctx["Q"])
        all_indices = match_ctx["all_indices"]
        matched_patch_idx_list = match_ctx["matched_patch_idx_list"]
        iou_thr = float(self.stage_b_extra_iou_match_thr)

        pairs_by_batch: List[List[Dict[str, int | bool | float]]] = []
        total_extra = 0
        for b, ((src_idx, tgt_idx), matched_patch_idx) in enumerate(zip(all_indices, matched_patch_idx_list)):
            pairs: List[Dict[str, int | bool | float]] = []
            matched_queries = torch.zeros((Q,), dtype=torch.bool, device=device)
            if src_idx.numel() > 0:
                matched_queries[src_idx.to(device=device)] = True
            for q_t, tgt_t, slot_t in zip(src_idx.tolist(), tgt_idx.tolist(), matched_patch_idx.tolist()):
                pairs.append(
                    {
                        "query": int(q_t),
                        "target": int(tgt_t),
                        "slot": int(slot_t),
                        "is_extra": False,
                        "iou": 1.0,
                    }
                )

            if iou_thr > 0 and targets[b]["boxes"].numel() > 0:
                target_to_slot = self._target_patch_slots_from_labels(targets[b], match_ctx, b, device)
                if target_to_slot:
                    ious, _ = box_ops.box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_boxes[b].detach()),
                        box_ops.box_cxcywh_to_xyxy(targets[b]["boxes"].to(device=device, dtype=pred_boxes.dtype)),
                    )
                    max_iou, max_tgt = ious.max(dim=1)
                    extra_q = torch.nonzero((~matched_queries) & (max_iou > iou_thr), as_tuple=False).flatten()
                    for q_t in extra_q.tolist():
                        tgt = int(max_tgt[q_t].item())
                        slot = target_to_slot.get(tgt, None)
                        if slot is None:
                            continue
                        pairs.append(
                            {
                                "query": int(q_t),
                                "target": tgt,
                                "slot": int(slot),
                                "is_extra": True,
                                "iou": float(max_iou[q_t].item()),
                            }
                        )
                        total_extra += 1
            pairs_by_batch.append(pairs)
        return pairs_by_batch, total_extra

    def _compute_allquery_focal_tn_matched_bce_text_loss(self, outputs, targets, match_ctx):
        pred_logits_text = outputs.get("pred_logits_text", None)
        if pred_logits_text is None:
            raise KeyError("StageBCriterion requires outputs['pred_logits_text'] in Stage B.")
        if pred_logits_text.dim() != 3:
            raise ValueError(
                f"outputs['pred_logits_text'] must be (B,Q,T), got {tuple(pred_logits_text.shape)}"
            )

        device = pred_logits_text.device
        target_map = torch.zeros_like(pred_logits_text, dtype=pred_logits_text.dtype, device=device)
        weight_map = torch.ones_like(pred_logits_text, dtype=pred_logits_text.dtype, device=device)
        valid_text_mask = self._get_valid_text_mask(outputs, pred_logits_text)
        dense_valid_map = valid_text_mask[:, None, :].expand_as(pred_logits_text).clone()
        zero = pred_logits_text.new_zeros(())

        pairs_by_batch, extra_iou_query_count = self._iter_stage_b_supervised_pairs(outputs, targets, match_ctx)

        dense_supervised_query_count = 0
        tn_supervised_query_count = 0
        content_pos_count = 0
        tn_neg_count = 0
        canonical_pos_count = 0
        empty_content_mask_count = 0
        empty_tn_negative_mask_count = 0
        tn_group_token_counts = {name: 0 for name in _TN_GROUP_NAMES}
        tn_group_nonempty_counts = {name: 0 for name in _TN_GROUP_NAMES}
        tn_group_slot_counts = {name: 0 for name in _TN_GROUP_NAMES}
        tn_content_slot_losses: List[torch.Tensor] = []
        tn_neg_slot_losses: List[torch.Tensor] = []
        tn_canonical_slot_losses: List[torch.Tensor] = []
        tn_group_slot_losses: Dict[str, List[torch.Tensor]] = {name: [] for name in _TN_GROUP_NAMES}

        for b, pairs in enumerate(pairs_by_batch):
            if not pairs:
                dense_valid_map[b] = False
                continue
            masks = self._validate_stage_b_token_masks(
                pred_logits_text=pred_logits_text,
                target=targets[b],
                batch_idx=b,
            )
            phrase_to_token_mask = masks["phrase_to_token_mask"]
            canonical_to_token_mask = masks["canonical_to_token_mask"]
            content_to_token_mask = masks["content_to_token_mask"]
            attr_pos_to_token_mask = masks["attr_pos_to_token_mask"]
            attr_neg_to_token_mask = masks["attr_neg_to_token_mask"]
            attr_neg_weight_mask = masks["attr_neg_weight_mask"]
            is_tn_mask = masks["is_tn_mask"]
            negative_to_token_mask = masks["negative_to_token_mask"]
            tn_group_ids = masks["tn_group_ids"]

            batch_has_dense_pair = False
            for pair in pairs:
                query_idx = int(pair["query"])
                slot_idx = int(pair["slot"])
                if slot_idx < 0 or slot_idx >= int(phrase_to_token_mask.shape[0]):
                    continue
                if query_idx < 0 or query_idx >= int(pred_logits_text.shape[1]):
                    continue

                phrase_mask = phrase_to_token_mask[slot_idx] & valid_text_mask[b]
                canonical_mask = canonical_to_token_mask[slot_idx] & phrase_mask
                is_tn_slot = bool(is_tn_mask[slot_idx].item()) if is_tn_mask is not None else False
                tn_group_name = (
                    _tn_group_name_from_id(int(tn_group_ids[slot_idx].item()))
                    if tn_group_ids is not None
                    else "other"
                )

                if attr_neg_to_token_mask is not None:
                    negative_attr_mask = attr_neg_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)
                elif negative_to_token_mask is not None:
                    negative_attr_mask = negative_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)
                else:
                    negative_attr_mask = torch.zeros_like(phrase_mask)
                if attr_neg_weight_mask is not None:
                    negative_weight = (
                        attr_neg_weight_mask[slot_idx].to(dtype=pred_logits_text.dtype)
                        * negative_attr_mask.to(dtype=pred_logits_text.dtype)
                    )
                else:
                    negative_weight = negative_attr_mask.to(dtype=pred_logits_text.dtype)
                effective_negative_mask = negative_attr_mask & (negative_weight > 0)

                if content_to_token_mask is not None:
                    content_pos_mask = content_to_token_mask[slot_idx]
                elif attr_pos_to_token_mask is not None:
                    content_pos_mask = attr_pos_to_token_mask[slot_idx]
                else:
                    content_pos_mask = phrase_mask
                content_pos_mask = content_pos_mask & phrase_mask & (~canonical_mask) & (~effective_negative_mask)

                if is_tn_slot:
                    tn_group_slot_counts[tn_group_name] += 1
                    tn_supervised_query_count += 1
                    dense_valid_map[b, :, phrase_mask] = False
                    if not effective_negative_mask.any():
                        empty_tn_negative_mask_count += 1

                    content_pos_loss = self._slot_bce_mean(
                        pred_logits_text[b, query_idx],
                        content_pos_mask,
                        self.stage_b_tn_content_target,
                    )
                    if content_pos_loss is not None:
                        tn_content_slot_losses.append(content_pos_loss)
                        content_pos_count += int(content_pos_mask.sum().item())

                    neg_loss = self._slot_bce_mean(
                        pred_logits_text[b, query_idx],
                        effective_negative_mask,
                        0.0,
                    )
                    if neg_loss is not None:
                        neg_token_count = int(effective_negative_mask.sum().item())
                        phrase_token_count = int(phrase_mask.sum().item())
                        neg_loss = neg_loss * self._tn_neg_effective_weight(phrase_token_count)
                        tn_neg_slot_losses.append(neg_loss)
                        tn_group_slot_losses[tn_group_name].append(neg_loss)
                        tn_neg_count += neg_token_count
                        tn_group_token_counts[tn_group_name] += neg_token_count
                        tn_group_nonempty_counts[tn_group_name] += 1

                    canonical_loss = self._slot_bce_mean(
                        pred_logits_text[b, query_idx],
                        canonical_mask,
                        self.stage_b_tn_canonical_target,
                    )
                    if canonical_loss is not None:
                        tn_canonical_slot_losses.append(canonical_loss * self.canonical_pos_weight)
                        canonical_pos_count += int(canonical_mask.sum().item())
                    continue

                if not content_pos_mask.any():
                    empty_content_mask_count += 1
                target_map[b, query_idx, content_pos_mask] = 1.0
                target_map[b, query_idx, canonical_mask] = 1.0
                if canonical_mask.any() and self.canonical_pos_weight != 1.0:
                    weight_map[b, query_idx, canonical_mask] = float(self.canonical_pos_weight)
                content_pos_count += int(content_pos_mask.sum().item())
                canonical_pos_count += int(canonical_mask.sum().item())
                dense_supervised_query_count += 1
                batch_has_dense_pair = True

            if not batch_has_dense_pair:
                dense_valid_map[b] = False

        valid_logits = pred_logits_text[dense_valid_map]
        valid_targets = target_map[dense_valid_map]
        valid_weights = weight_map[dense_valid_map]
        if valid_logits.numel() == 0:
            dense_focal_loss = zero
        else:
            valid_loss = _sigmoid_focal_loss_no_reduce(
                valid_logits,
                valid_targets,
                alpha=self.stage_b_text_focal_alpha,
                gamma=self.stage_b_text_focal_gamma,
            )
            valid_loss = valid_loss * valid_weights
            normalizer = max(float(dense_supervised_query_count), 1.0)
            dense_focal_loss = valid_loss.sum() / normalizer

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero
            return torch.stack(values).mean()

        tn_content_bce_loss = _mean_or_zero(tn_content_slot_losses)
        tn_neg_bce_loss = _mean_or_zero(tn_neg_slot_losses)
        tn_canonical_bce_loss = _mean_or_zero(tn_canonical_slot_losses)
        tn_content_bce_loss_weighted = tn_content_bce_loss * self.stage_b_tn_content_weight
        tn_neg_bce_loss_weighted = tn_neg_bce_loss * self.stage_b_tn_neg_weight
        tn_canonical_bce_loss_weighted = tn_canonical_bce_loss * self.stage_b_tn_canonical_weight
        tn_matched_bce_loss = (
            tn_content_bce_loss_weighted
            + tn_neg_bce_loss_weighted
            + tn_canonical_bce_loss_weighted
        )
        loss_text = dense_focal_loss + tn_matched_bce_loss
        positive_token_count = int((target_map[dense_valid_map] > 0.5).sum().item())
        valid_token_count = int(dense_valid_map.sum().item())
        negative_token_count = max(valid_token_count - positive_token_count, 0)
        metrics = {
            "loss_text": loss_text,
            "text_token_loss_raw": loss_text.detach(),
            "text_phrase_tn_loss_raw": zero.detach(),
            "content_pos_loss": dense_focal_loss.detach(),
            "canonical_loss": tn_canonical_bce_loss.detach(),
            "tn_neg_loss": tn_neg_bce_loss.detach(),
            "text_valid_slot_count": torch.as_tensor(float(dense_supervised_query_count + tn_supervised_query_count), device=device),
            "text_content_pos_slot_count": torch.as_tensor(float(dense_supervised_query_count), device=device),
            "text_tn_neg_slot_count": torch.as_tensor(float(len(tn_neg_slot_losses)), device=device),
            "text_attr_pos_slot_count": torch.as_tensor(float(dense_supervised_query_count), device=device),
            "text_attr_neg_slot_count": torch.as_tensor(float(len(tn_neg_slot_losses)), device=device),
            "text_canonical_slot_count": torch.as_tensor(float(canonical_pos_count), device=device),
            "text_phrase_tn_slot_count": zero.detach(),
            "text_skipped_tn_slot_count": torch.as_tensor(float(empty_tn_negative_mask_count), device=device),
            "effective_content_token_count": torch.as_tensor(float(content_pos_count), device=device),
            "effective_tn_negative_token_count": torch.as_tensor(float(tn_neg_count), device=device),
            "empty_content_mask_count": torch.as_tensor(float(empty_content_mask_count), device=device),
            "empty_tn_negative_mask_count": torch.as_tensor(float(empty_tn_negative_mask_count), device=device),
            "text_tn_neg_weight": torch.as_tensor(float(self.stage_b_tn_neg_weight), device=device),
            "text_tn_neg_weight_mode_token_count": torch.as_tensor(
                float(self.stage_b_tn_neg_weight_mode == "token_count"), device=device
            ),
            "text_tn_content_weight": torch.as_tensor(float(self.stage_b_tn_content_weight), device=device),
            "text_tn_canonical_weight": torch.as_tensor(float(self.stage_b_tn_canonical_weight), device=device),
            "spatial_like_tn_count": torch.as_tensor(float(tn_group_slot_counts["spatial_like"]), device=device),
            "relation_action_like_tn_count": torch.as_tensor(
                float(tn_group_slot_counts["relation_action_like"]), device=device
            ),
            "text_allquery_focal_loss_raw": dense_focal_loss.detach(),
            "text_allquery_positive_token_count": torch.as_tensor(float(positive_token_count), device=device),
            "text_allquery_negative_token_count": torch.as_tensor(float(negative_token_count), device=device),
            "text_allquery_valid_token_count": torch.as_tensor(float(valid_token_count), device=device),
            "text_allquery_matched_query_count": torch.as_tensor(float(dense_supervised_query_count), device=device),
            "text_allquery_focal_alpha": torch.as_tensor(float(self.stage_b_text_focal_alpha), device=device),
            "text_allquery_focal_gamma": torch.as_tensor(float(self.stage_b_text_focal_gamma), device=device),
            "text_v5_dense_focal_loss_raw": dense_focal_loss.detach(),
            "text_v5_tn_matched_bce_loss_raw": tn_matched_bce_loss.detach(),
            "text_v5_tn_content_bce_loss_raw": tn_content_bce_loss.detach(),
            "text_v5_tn_neg_bce_loss_raw": tn_neg_bce_loss.detach(),
            "text_v5_tn_canonical_bce_loss_raw": tn_canonical_bce_loss.detach(),
            "text_v5_tn_content_bce_loss_weighted": tn_content_bce_loss_weighted.detach(),
            "text_v5_tn_neg_bce_loss_weighted": tn_neg_bce_loss_weighted.detach(),
            "text_v5_tn_canonical_bce_loss_weighted": tn_canonical_bce_loss_weighted.detach(),
            "text_v5_tn_neg_weight": torch.as_tensor(float(self.stage_b_tn_neg_weight), device=device),
            "text_v5_tn_content_weight": torch.as_tensor(float(self.stage_b_tn_content_weight), device=device),
            "text_v5_tn_canonical_weight": torch.as_tensor(float(self.stage_b_tn_canonical_weight), device=device),
            "text_v5_tn_content_target": torch.as_tensor(float(self.stage_b_tn_content_target), device=device),
            "text_v5_tn_canonical_target": torch.as_tensor(float(self.stage_b_tn_canonical_target), device=device),
            "text_v5_extra_iou_query_count": torch.as_tensor(float(extra_iou_query_count), device=device),
            "text_v5_extra_iou_match_thr": torch.as_tensor(float(self.stage_b_extra_iou_match_thr), device=device),
            "text_v5_dense_supervised_query_count": torch.as_tensor(float(dense_supervised_query_count), device=device),
            "text_v5_tn_supervised_query_count": torch.as_tensor(float(tn_supervised_query_count), device=device),
        }
        for group_name in _TN_GROUP_NAMES:
            group_loss = _mean_or_zero(tn_group_slot_losses[group_name])
            metrics[f"loss_tn_{group_name}"] = group_loss.detach()
            metrics[f"tn_neg_count_{group_name}"] = torch.as_tensor(
                float(tn_group_token_counts[group_name]), device=device
            )
            metrics[f"tn_nonempty_mask_count_{group_name}"] = torch.as_tensor(
                float(tn_group_nonempty_counts[group_name]), device=device
            )
        return metrics

    def _compute_allquery_focal_tn_empty_det_text_loss(self, outputs, targets, match_ctx):
        pred_logits_text = outputs.get("pred_logits_text", None)
        if pred_logits_text is None:
            raise KeyError("StageBCriterion requires outputs['pred_logits_text'] in Stage B.")
        if pred_logits_text.dim() != 3:
            raise ValueError(
                f"outputs['pred_logits_text'] must be (B,Q,T), got {tuple(pred_logits_text.shape)}"
            )

        device = pred_logits_text.device
        target_map = torch.zeros_like(pred_logits_text, dtype=pred_logits_text.dtype, device=device)
        weight_map = torch.ones_like(pred_logits_text, dtype=pred_logits_text.dtype, device=device)
        valid_text_mask = self._get_valid_text_mask(outputs, pred_logits_text)
        dense_valid_map = valid_text_mask[:, None, :].expand_as(pred_logits_text)
        zero = pred_logits_text.new_zeros(())

        det_supervised_query_count = 0
        tn_patch_matched_query_count = 0
        tn_empty_det_sample_count = 0
        tn_text_positive_token_count = 0
        content_pos_count = 0
        canonical_pos_count = 0
        empty_content_mask_count = 0
        tn_group_slot_counts = {name: 0 for name in _TN_GROUP_NAMES}

        all_indices = match_ctx["all_indices"]
        matched_patch_idx_list = match_ctx["matched_patch_idx_list"]
        for b, ((src_idx, _tgt_idx), matched_patch_idx) in enumerate(zip(all_indices, matched_patch_idx_list)):
            sample_has_tn = False
            tn_mask = self._target_tn_mask(targets[b], device)
            if tn_mask is not None and bool(tn_mask.any().item()):
                sample_has_tn = True
                tn_empty_det_sample_count += 1
            if src_idx.numel() == 0:
                continue
            masks = self._validate_stage_b_token_masks(
                pred_logits_text=pred_logits_text,
                target=targets[b],
                batch_idx=b,
            )
            phrase_to_token_mask = masks["phrase_to_token_mask"]
            canonical_to_token_mask = masks["canonical_to_token_mask"]
            content_to_token_mask = masks["content_to_token_mask"]
            attr_pos_to_token_mask = masks["attr_pos_to_token_mask"]
            attr_neg_to_token_mask = masks["attr_neg_to_token_mask"]
            negative_to_token_mask = masks["negative_to_token_mask"]
            attr_neg_weight_mask = masks["attr_neg_weight_mask"]
            is_tn_mask = masks["is_tn_mask"]
            tn_group_ids = masks["tn_group_ids"]

            for query_idx_t, slot_idx_t in zip(src_idx.tolist(), matched_patch_idx.tolist()):
                query_idx = int(query_idx_t)
                slot_idx = int(slot_idx_t)
                if slot_idx < 0 or slot_idx >= int(phrase_to_token_mask.shape[0]):
                    continue

                is_tn_slot = bool(is_tn_mask[slot_idx].item()) if is_tn_mask is not None else sample_has_tn
                tn_group_name = (
                    _tn_group_name_from_id(int(tn_group_ids[slot_idx].item()))
                    if tn_group_ids is not None
                    else "other"
                )
                if is_tn_slot:
                    tn_patch_matched_query_count += 1
                    tn_group_slot_counts[tn_group_name] += 1
                    tn_phrase_mask = phrase_to_token_mask[slot_idx] & valid_text_mask[b]
                    tn_canonical_mask = canonical_to_token_mask[slot_idx] & tn_phrase_mask
                    if attr_neg_to_token_mask is not None:
                        tn_negative_mask = attr_neg_to_token_mask[slot_idx] & tn_phrase_mask & (~tn_canonical_mask)
                    elif negative_to_token_mask is not None:
                        tn_negative_mask = negative_to_token_mask[slot_idx] & tn_phrase_mask & (~tn_canonical_mask)
                    else:
                        tn_negative_mask = torch.zeros_like(tn_phrase_mask)
                    if attr_neg_weight_mask is not None:
                        tn_negative_weight = (
                            attr_neg_weight_mask[slot_idx].to(dtype=pred_logits_text.dtype)
                            * tn_negative_mask.to(dtype=pred_logits_text.dtype)
                        )
                    else:
                        tn_negative_weight = tn_negative_mask.to(dtype=pred_logits_text.dtype)
                    tn_effective_negative_mask = tn_negative_mask & (tn_negative_weight > 0)
                    if tn_effective_negative_mask.any():
                        tn_phrase_token_count = int(tn_phrase_mask.sum().item())
                        tn_weight = self.stage_b_tn_neg_weight * self._tn_neg_effective_weight(
                            tn_phrase_token_count
                        )
                        weight_map[b, :, tn_effective_negative_mask] = torch.maximum(
                            weight_map[b, :, tn_effective_negative_mask],
                            torch.as_tensor(tn_weight, dtype=weight_map.dtype, device=device),
                        )
                    tn_text_positive_token_count += int(target_map[b, query_idx, tn_phrase_mask].sum().item())
                    continue

                phrase_mask = phrase_to_token_mask[slot_idx] & valid_text_mask[b]
                canonical_mask = canonical_to_token_mask[slot_idx] & phrase_mask

                if content_to_token_mask is not None:
                    content_pos_mask = content_to_token_mask[slot_idx]
                elif attr_pos_to_token_mask is not None:
                    content_pos_mask = attr_pos_to_token_mask[slot_idx]
                else:
                    content_pos_mask = phrase_mask
                content_pos_mask = content_pos_mask & phrase_mask & (~canonical_mask)
                if not content_pos_mask.any():
                    empty_content_mask_count += 1

                target_map[b, query_idx, content_pos_mask] = 1.0
                target_map[b, query_idx, canonical_mask] = 1.0

                content_pos_count += int(content_pos_mask.sum().item())
                canonical_pos_count += int(canonical_mask.sum().item())
                det_supervised_query_count += 1

        valid_logits = pred_logits_text[dense_valid_map]
        valid_targets = target_map[dense_valid_map]
        valid_weights = weight_map[dense_valid_map]
        if valid_logits.numel() == 0:
            loss_text = zero
        else:
            valid_loss = _sigmoid_focal_loss_no_reduce(
                valid_logits,
                valid_targets,
                alpha=self.stage_b_text_focal_alpha,
                gamma=self.stage_b_text_focal_gamma,
            )
            valid_loss = valid_loss * valid_weights
            normalizer = max(float(det_supervised_query_count), 1.0)
            loss_text = valid_loss.sum() / normalizer

        positive_token_count = int((target_map[dense_valid_map] > 0.5).sum().item())
        valid_token_count = int(dense_valid_map.sum().item())
        negative_token_count = max(valid_token_count - positive_token_count, 0)
        tn_dense_valid_token_count = int(
            sum(
                int(valid_text_mask[b].sum().item()) * int(pred_logits_text.shape[1])
                for b, target in enumerate(targets)
                if (self._target_tn_mask(target, device) is not None)
                and bool(self._target_tn_mask(target, device).any().item())
            )
        )
        metrics = {
            "loss_text": loss_text,
            "text_token_loss_raw": loss_text.detach(),
            "text_phrase_tn_loss_raw": zero.detach(),
            "content_pos_loss": loss_text.detach(),
            "canonical_loss": zero.detach(),
            "tn_neg_loss": zero.detach(),
            "text_valid_slot_count": torch.as_tensor(float(det_supervised_query_count + tn_patch_matched_query_count), device=device),
            "text_content_pos_slot_count": torch.as_tensor(float(det_supervised_query_count), device=device),
            "text_tn_neg_slot_count": torch.as_tensor(float(tn_patch_matched_query_count), device=device),
            "text_attr_pos_slot_count": torch.as_tensor(float(det_supervised_query_count), device=device),
            "text_attr_neg_slot_count": torch.as_tensor(float(tn_patch_matched_query_count), device=device),
            "text_canonical_slot_count": torch.as_tensor(float(canonical_pos_count), device=device),
            "text_phrase_tn_slot_count": zero.detach(),
            "text_skipped_tn_slot_count": zero.detach(),
            "effective_content_token_count": torch.as_tensor(float(content_pos_count), device=device),
            "effective_tn_negative_token_count": torch.as_tensor(float(tn_dense_valid_token_count), device=device),
            "empty_content_mask_count": torch.as_tensor(float(empty_content_mask_count), device=device),
            "empty_tn_negative_mask_count": zero.detach(),
            "text_tn_neg_weight": torch.as_tensor(float(self.stage_b_tn_neg_weight), device=device),
            "text_tn_neg_weight_mode_token_count": torch.as_tensor(
                float(self.stage_b_tn_neg_weight_mode == "token_count"), device=device
            ),
            "text_tn_content_weight": torch.as_tensor(float(self.stage_b_tn_content_weight), device=device),
            "text_tn_canonical_weight": torch.as_tensor(float(self.stage_b_tn_canonical_weight), device=device),
            "spatial_like_tn_count": torch.as_tensor(float(tn_group_slot_counts["spatial_like"]), device=device),
            "relation_action_like_tn_count": torch.as_tensor(
                float(tn_group_slot_counts["relation_action_like"]), device=device
            ),
            "text_allquery_focal_loss_raw": loss_text.detach(),
            "text_allquery_positive_token_count": torch.as_tensor(float(positive_token_count), device=device),
            "text_allquery_negative_token_count": torch.as_tensor(float(negative_token_count), device=device),
            "text_allquery_valid_token_count": torch.as_tensor(float(valid_token_count), device=device),
            "text_allquery_matched_query_count": torch.as_tensor(float(det_supervised_query_count), device=device),
            "text_allquery_focal_alpha": torch.as_tensor(float(self.stage_b_text_focal_alpha), device=device),
            "text_allquery_focal_gamma": torch.as_tensor(float(self.stage_b_text_focal_gamma), device=device),
            "text_v6_tn_empty_det_sample_count": torch.as_tensor(float(tn_empty_det_sample_count), device=device),
            "text_v6_tn_patch_matched_query_count": torch.as_tensor(float(tn_patch_matched_query_count), device=device),
            "text_v6_tn_text_positive_token_count": torch.as_tensor(float(tn_text_positive_token_count), device=device),
            "text_v6_tn_dense_valid_token_count": torch.as_tensor(float(tn_dense_valid_token_count), device=device),
            "text_v6_det_supervised_query_count": torch.as_tensor(float(det_supervised_query_count), device=device),
        }
        for group_name in _TN_GROUP_NAMES:
            metrics[f"loss_tn_{group_name}"] = zero.detach()
            metrics[f"tn_neg_count_{group_name}"] = zero.detach()
            metrics[f"tn_nonempty_mask_count_{group_name}"] = torch.as_tensor(
                float(tn_group_slot_counts[group_name]), device=device
            )
        return metrics

    def _compute_text_loss(self, outputs, targets, match_ctx):
        if self.stage_b_text_loss_type == "allquery_focal":
            return self._compute_allquery_focal_text_loss(outputs, targets, match_ctx)
        if self.stage_b_text_loss_type == "allquery_focal_tn_matched_bce":
            return self._compute_allquery_focal_tn_matched_bce_text_loss(outputs, targets, match_ctx)
        if self.stage_b_text_loss_type == "allquery_focal_tn_empty_det":
            return self._compute_allquery_focal_tn_empty_det_text_loss(outputs, targets, match_ctx)
        return self._compute_matched_bce_text_loss(outputs, targets, match_ctx)

    def _zero_score_calib_loss_dict(self, zero: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = zero.detach()
        return {
            "loss_score_calib": zero,
            "score_calib_pos_loss": z,
            "score_calib_neg_loss": z,
            "score_calib_gap_loss": z,
            "score_calib_pos_query_loss": z,
            "score_calib_all_tn_neg_loss": z,
            "score_calib_pair_count": z,
            "score_calib_used_pair_count": z,
            "score_calib_skipped_pair_count": z,
            "score_calib_all_tn_neg_count": z,
            "score_calib_pos_score": z,
            "score_calib_neg_matched_score": z,
            "score_calib_neg_topk_score": z,
            "score_calib_pos_other_topk_score": z,
            "score_calib_all_tn_neg_score": z,
            "score_calib_all_tn_neg_topk_max_score": z,
            "score_calib_tau_pos": torch.as_tensor(float(self.stage_b_score_calib_tau_pos), device=zero.device),
            "score_calib_tau_neg": torch.as_tensor(float(self.stage_b_score_calib_tau_neg), device=zero.device),
            "score_calib_margin": torch.as_tensor(float(self.stage_b_score_calib_margin), device=zero.device),
            "score_calib_topk": torch.as_tensor(float(self.stage_b_score_calib_topk), device=zero.device),
            "score_calib_neg_agg_score": z,
            "score_calib_neg_topk_max_score": z,
            "score_calib_neg_lse_tau": torch.as_tensor(float(self.stage_b_score_calib_neg_lse_tau), device=zero.device),
        }

    def _compute_score_calib_loss(self, outputs, targets, match_ctx_neg):
        rank_pos_outputs = outputs.get("rank_pos_outputs", None)
        rank_pos_targets = outputs.get("rank_pos_targets", None)
        rank_pair_map = outputs.get("rank_pair_map", None)
        pred_logits_patch = outputs.get("pred_logits_patch", None)
        if pred_logits_patch is not None:
            zero = pred_logits_patch.sum() * 0.0
        else:
            zero = outputs["pred_boxes"].sum() * 0.0
        if self.stage_b_score_calib_loss_coef <= 0:
            return self._zero_score_calib_loss_dict(zero)

        device = zero.device
        score_neg = compute_stage_b_slot_logits(
            outputs,
            beta=self.stage_b_rank_beta,
            canonical_weight=self.stage_b_rank_canonical_weight,
            text_agg=self.stage_b_rank_text_agg,
            softmin_tau=self.stage_b_rank_softmin_tau,
            mean_softmin_alpha=self.stage_b_rank_mean_softmin_alpha,
            detach_patch=self.stage_b_score_calib_detach_patch,
            normalize_fused_score=True,
        )

        pos_losses: List[torch.Tensor] = []
        neg_losses: List[torch.Tensor] = []
        gap_losses: List[torch.Tensor] = []
        pos_query_losses: List[torch.Tensor] = []
        pos_scores: List[torch.Tensor] = []
        neg_matched_scores: List[torch.Tensor] = []
        neg_topk_scores: List[torch.Tensor] = []
        neg_agg_scores: List[torch.Tensor] = []
        neg_topk_max_scores: List[torch.Tensor] = []
        pos_other_topk_scores: List[torch.Tensor] = []
        all_tn_neg_losses: List[torch.Tensor] = []
        all_tn_neg_scores: List[torch.Tensor] = []
        all_tn_neg_topk_max_scores: List[torch.Tensor] = []
        used_count = 0
        skipped_count = 0
        all_tn_neg_count = 0
        topk = max(1, int(self.stage_b_score_calib_topk))
        pair_count = 0
        if rank_pos_outputs is not None and rank_pos_targets is not None and rank_pair_map is not None:
            rank_pair_map = rank_pair_map.to(device=device, dtype=torch.long).view(-1)
            if len(rank_pos_targets) != int(rank_pair_map.numel()):
                raise ValueError(
                    "rank_pos_targets length must match rank_pair_map, "
                    f"got {len(rank_pos_targets)} vs {rank_pair_map.numel()}"
                )

            match_ctx_pos = self.compute_matching(rank_pos_outputs, rank_pos_targets)
            score_pos = compute_stage_b_slot_logits(
                rank_pos_outputs,
                beta=self.stage_b_rank_beta,
                canonical_weight=self.stage_b_rank_canonical_weight,
                text_agg=self.stage_b_rank_text_agg,
                softmin_tau=self.stage_b_rank_softmin_tau,
                mean_softmin_alpha=self.stage_b_rank_mean_softmin_alpha,
                detach_patch=self.stage_b_score_calib_detach_patch,
                normalize_fused_score=True,
            )
            pair_count = int(rank_pair_map.numel())

            neg_indices = match_ctx_neg["all_indices"]
            neg_slots = match_ctx_neg["matched_patch_idx_list"]
            pos_indices = match_ctx_pos["all_indices"]
            pos_slots = match_ctx_pos["matched_patch_idx_list"]

            for rank_row, batch_idx_t in enumerate(rank_pair_map.tolist()):
                batch_idx = int(batch_idx_t)
                if batch_idx < 0 or batch_idx >= len(targets):
                    skipped_count += 1
                    continue
                rank_source_slot = rank_pos_targets[rank_row].get("rank_source_slot", None)
                if torch.is_tensor(rank_source_slot) and rank_source_slot.numel() > 0:
                    source_slot = int(rank_source_slot.view(-1)[0].item())
                else:
                    source_slot = 0

                src_neg, tgt_neg = neg_indices[batch_idx]
                slot_neg = neg_slots[batch_idx]
                src_pos, tgt_pos = pos_indices[rank_row]
                slot_pos = pos_slots[rank_row]
                if src_neg.numel() == 0 or src_pos.numel() == 0:
                    skipped_count += 1
                    continue

                neg_by_target = {}
                for query_idx, target_idx, slot_idx in zip(src_neg.tolist(), tgt_neg.tolist(), slot_neg.tolist()):
                    if int(slot_idx) == source_slot:
                        neg_by_target[int(target_idx)] = (int(query_idx), int(slot_idx))

                pos_by_target = {}
                rank_target_ids = rank_pos_targets[rank_row].get("rank_target_ids", None)
                if torch.is_tensor(rank_target_ids):
                    rank_target_ids = rank_target_ids.to(device=device, dtype=torch.long).view(-1)
                for query_idx, target_idx, slot_idx in zip(src_pos.tolist(), tgt_pos.tolist(), slot_pos.tolist()):
                    local_target_idx = int(target_idx)
                    if rank_target_ids is not None and local_target_idx < int(rank_target_ids.numel()):
                        original_target_idx = int(rank_target_ids[local_target_idx].item())
                    else:
                        original_target_idx = local_target_idx
                    pos_by_target[original_target_idx] = (int(query_idx), int(slot_idx))

                common_targets = sorted(set(neg_by_target.keys()) & set(pos_by_target.keys()))
                if not common_targets:
                    skipped_count += 1
                    continue

                for target_idx in common_targets:
                    q_neg, k_neg = neg_by_target[target_idx]
                    q_pos, k_pos = pos_by_target[target_idx]
                    if k_neg < 0 or k_neg >= score_neg.shape[2] or k_pos < 0 or k_pos >= score_pos.shape[2]:
                        skipped_count += 1
                        continue

                    s_neg = score_neg[batch_idx, q_neg, k_neg]
                    s_pos = score_pos[rank_row, q_pos, k_pos]
                    pos_losses.append(F.softplus(self.stage_b_score_calib_tau_pos - s_pos))
                    pos_scores.append(s_pos.detach().reshape(1))
                    neg_matched_scores.append(s_neg.detach().reshape(1))

                    neg_scores = score_neg[batch_idx].reshape(-1)
                    neg_k = min(topk, int(neg_scores.numel()))
                    if neg_k > 0:
                        neg_topk = torch.topk(neg_scores, k=neg_k, largest=True).values
                        if self.stage_b_score_calib_neg_agg in {"logsumexp", "lse"}:
                            tau = float(self.stage_b_score_calib_neg_lse_tau)
                            neg_agg = tau * torch.logsumexp(neg_topk / tau, dim=0)
                        elif self.stage_b_score_calib_neg_agg == "max":
                            neg_agg = neg_topk.max()
                        else:
                            neg_agg = neg_topk.mean()
                        neg_losses.append(F.softplus(neg_agg - self.stage_b_score_calib_tau_neg))
                        gap_losses.append(F.softplus(self.stage_b_score_calib_margin - s_pos + neg_agg))
                        neg_topk_scores.append(neg_topk.detach())
                        neg_agg_scores.append(neg_agg.detach().reshape(1))
                        neg_topk_max_scores.append(neg_topk.max().detach().reshape(1))

                    pos_slot_scores = score_pos[rank_row, :, k_pos]
                    if int(pos_slot_scores.numel()) > 1:
                        masked_pos_scores = pos_slot_scores.clone()
                        masked_pos_scores[q_pos] = torch.finfo(masked_pos_scores.dtype).min
                        pos_k = min(topk, int(masked_pos_scores.numel()) - 1)
                        if pos_k > 0:
                            pos_other = torch.topk(masked_pos_scores, k=pos_k, largest=True).values
                            pos_query_losses.append(
                                F.softplus(self.stage_b_score_calib_margin - s_pos + pos_other).mean()
                            )
                            pos_other_topk_scores.append(pos_other.detach())
                    used_count += 1

        if self.stage_b_score_calib_all_tn_neg_weight > 0:
            for batch_idx, target in enumerate(targets):
                is_tn = target.get("is_tn", None)
                if not torch.is_tensor(is_tn):
                    continue
                tn_slots = is_tn.to(device=device, dtype=torch.bool).view(-1)
                if tn_slots.numel() == 0:
                    continue
                tn_slot_idx = torch.nonzero(tn_slots, as_tuple=False).flatten()
                if tn_slot_idx.numel() == 0:
                    continue
                tn_slot_idx = tn_slot_idx[tn_slot_idx < score_neg.shape[2]]
                if tn_slot_idx.numel() == 0:
                    continue
                tn_scores = score_neg[batch_idx].index_select(1, tn_slot_idx).reshape(-1)
                tn_k = min(topk, int(tn_scores.numel()))
                if tn_k <= 0:
                    continue
                tn_topk = torch.topk(tn_scores, k=tn_k, largest=True).values
                if self.stage_b_score_calib_neg_agg in {"logsumexp", "lse"}:
                    tau = float(self.stage_b_score_calib_neg_lse_tau)
                    tn_agg = tau * torch.logsumexp(tn_topk / tau, dim=0)
                elif self.stage_b_score_calib_neg_agg == "max":
                    tn_agg = tn_topk.max()
                else:
                    tn_agg = tn_topk.mean()
                all_tn_neg_losses.append(F.softplus(tn_agg - self.stage_b_score_calib_tau_neg))
                all_tn_neg_scores.append(tn_agg.detach().reshape(1))
                all_tn_neg_topk_max_scores.append(tn_topk.max().detach().reshape(1))
                all_tn_neg_count += 1

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero
            return torch.stack([x.reshape(()) for x in values]).mean()

        pos_loss = _mean_or_zero(pos_losses)
        neg_loss = _mean_or_zero(neg_losses)
        gap_loss = _mean_or_zero(gap_losses)
        pos_query_loss = _mean_or_zero(pos_query_losses)
        all_tn_neg_loss = _mean_or_zero(all_tn_neg_losses)
        total = (
            self.stage_b_score_calib_pos_weight * pos_loss
            + self.stage_b_score_calib_neg_weight * neg_loss
            + self.stage_b_score_calib_gap_weight * gap_loss
            + self.stage_b_score_calib_pos_query_weight * pos_query_loss
            + self.stage_b_score_calib_all_tn_neg_weight * all_tn_neg_loss
        )

        def _cat_mean(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero.detach()
            return torch.cat([x.reshape(-1) for x in values]).mean().detach()

        return {
            "loss_score_calib": total,
            "score_calib_pos_loss": pos_loss.detach(),
            "score_calib_neg_loss": neg_loss.detach(),
            "score_calib_gap_loss": gap_loss.detach(),
            "score_calib_pos_query_loss": pos_query_loss.detach(),
            "score_calib_all_tn_neg_loss": all_tn_neg_loss.detach(),
            "score_calib_pair_count": torch.as_tensor(float(pair_count), device=device),
            "score_calib_used_pair_count": torch.as_tensor(float(used_count), device=device),
            "score_calib_skipped_pair_count": torch.as_tensor(float(skipped_count), device=device),
            "score_calib_all_tn_neg_count": torch.as_tensor(float(all_tn_neg_count), device=device),
            "score_calib_pos_score": _cat_mean(pos_scores),
            "score_calib_neg_matched_score": _cat_mean(neg_matched_scores),
            "score_calib_neg_topk_score": _cat_mean(neg_topk_scores),
            "score_calib_neg_agg_score": _cat_mean(neg_agg_scores),
            "score_calib_neg_topk_max_score": _cat_mean(neg_topk_max_scores),
            "score_calib_pos_other_topk_score": _cat_mean(pos_other_topk_scores),
            "score_calib_all_tn_neg_score": _cat_mean(all_tn_neg_scores),
            "score_calib_all_tn_neg_topk_max_score": _cat_mean(all_tn_neg_topk_max_scores),
            "score_calib_tau_pos": torch.as_tensor(float(self.stage_b_score_calib_tau_pos), device=device),
            "score_calib_tau_neg": torch.as_tensor(float(self.stage_b_score_calib_tau_neg), device=device),
            "score_calib_margin": torch.as_tensor(float(self.stage_b_score_calib_margin), device=device),
            "score_calib_topk": torch.as_tensor(float(topk), device=device),
            "score_calib_neg_lse_tau": torch.as_tensor(float(self.stage_b_score_calib_neg_lse_tau), device=device),
        }

    def _zero_rank_loss_dict(self, zero: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = zero.detach()
        return {
            "loss_phrase_rank": zero,
            "phrase_rank_loss_raw": z,
            "phrase_rank_pair_count": z,
            "phrase_rank_used_pair_count": z,
            "phrase_rank_violation_count": z,
            "phrase_rank_skipped_pair_count": z,
            "phrase_rank_candidate_tn_count": z,
            "phrase_rank_missing_positive_count": z,
            "phrase_rank_invalid_positive_count": z,
            "phrase_rank_margin": torch.as_tensor(float(self.stage_b_rank_margin), device=zero.device),
        }

    def _compute_phrase_rank_loss(self, outputs, targets, match_ctx_neg):
        rank_pos_outputs = outputs.get("rank_pos_outputs", None)
        rank_pos_targets = outputs.get("rank_pos_targets", None)
        rank_pair_map = outputs.get("rank_pair_map", None)
        pred_logits_patch = outputs.get("pred_logits_patch", None)
        if pred_logits_patch is not None:
            zero = pred_logits_patch.sum() * 0.0
        else:
            zero = outputs["pred_boxes"].sum() * 0.0
        device = zero.device
        candidate_tn_count = outputs.get("rank_candidate_tn_count", zero.detach()).to(device=device)
        missing_positive_count = outputs.get("rank_missing_positive_count", zero.detach()).to(device=device)
        invalid_positive_count = outputs.get("rank_invalid_positive_count", zero.detach()).to(device=device)
        if (
            rank_pos_outputs is None
            or rank_pos_targets is None
            or rank_pair_map is None
            or self.stage_b_rank_loss_coef <= 0
        ):
            metrics = self._zero_rank_loss_dict(zero)
            metrics["phrase_rank_candidate_tn_count"] = candidate_tn_count.detach()
            metrics["phrase_rank_missing_positive_count"] = missing_positive_count.detach()
            metrics["phrase_rank_invalid_positive_count"] = invalid_positive_count.detach()
            return metrics

        rank_pair_map = rank_pair_map.to(device=device, dtype=torch.long).view(-1)
        if len(rank_pos_targets) != int(rank_pair_map.numel()):
            raise ValueError(
                f"rank_pos_targets length must match rank_pair_map, got {len(rank_pos_targets)} vs {rank_pair_map.numel()}"
            )

        match_ctx_pos = self.compute_matching(rank_pos_outputs, rank_pos_targets)
        score_neg = compute_stage_b_slot_logits(
            outputs,
            beta=self.stage_b_rank_beta,
            canonical_weight=self.stage_b_rank_canonical_weight,
            text_agg=self.stage_b_rank_text_agg,
            softmin_tau=self.stage_b_rank_softmin_tau,
            mean_softmin_alpha=self.stage_b_rank_mean_softmin_alpha,
            detach_patch=self.stage_b_rank_detach_patch,
            normalize_fused_score=True,
        )
        score_pos = compute_stage_b_slot_logits(
            rank_pos_outputs,
            beta=self.stage_b_rank_beta,
            canonical_weight=self.stage_b_rank_canonical_weight,
            text_agg=self.stage_b_rank_text_agg,
            softmin_tau=self.stage_b_rank_softmin_tau,
            mean_softmin_alpha=self.stage_b_rank_mean_softmin_alpha,
            detach_patch=self.stage_b_rank_detach_patch,
            normalize_fused_score=True,
        )

        losses: List[torch.Tensor] = []
        used_count = 0
        violation_count = 0
        skipped_count = 0
        pair_count = int(rank_pair_map.numel())
        neg_indices = match_ctx_neg["all_indices"]
        neg_slots = match_ctx_neg["matched_patch_idx_list"]
        pos_indices = match_ctx_pos["all_indices"]
        pos_slots = match_ctx_pos["matched_patch_idx_list"]

        for rank_row, batch_idx_t in enumerate(rank_pair_map.tolist()):
            batch_idx = int(batch_idx_t)
            if batch_idx < 0 or batch_idx >= len(targets):
                skipped_count += 1
                continue
            rank_source_slot = rank_pos_targets[rank_row].get("rank_source_slot", None)
            if torch.is_tensor(rank_source_slot) and rank_source_slot.numel() > 0:
                source_slot = int(rank_source_slot.view(-1)[0].item())
            else:
                source_slot = 0

            src_neg, tgt_neg = neg_indices[batch_idx]
            slot_neg = neg_slots[batch_idx]
            src_pos, tgt_pos = pos_indices[rank_row]
            slot_pos = pos_slots[rank_row]
            if src_neg.numel() == 0 or src_pos.numel() == 0:
                skipped_count += 1
                continue

            neg_by_target = {}
            for row_idx, (query_idx, target_idx, slot_idx) in enumerate(
                zip(src_neg.tolist(), tgt_neg.tolist(), slot_neg.tolist())
            ):
                if int(slot_idx) != source_slot:
                    continue
                neg_by_target[int(target_idx)] = (int(query_idx), int(slot_idx))

            pos_by_target = {}
            rank_target_ids = rank_pos_targets[rank_row].get("rank_target_ids", None)
            if torch.is_tensor(rank_target_ids):
                rank_target_ids = rank_target_ids.to(device=device, dtype=torch.long).view(-1)
            for row_idx, (query_idx, target_idx, slot_idx) in enumerate(
                zip(src_pos.tolist(), tgt_pos.tolist(), slot_pos.tolist())
            ):
                local_target_idx = int(target_idx)
                if rank_target_ids is not None and local_target_idx < int(rank_target_ids.numel()):
                    original_target_idx = int(rank_target_ids[local_target_idx].item())
                else:
                    original_target_idx = local_target_idx
                pos_by_target[original_target_idx] = (int(query_idx), int(slot_idx))

            common_targets = sorted(set(neg_by_target.keys()) & set(pos_by_target.keys()))
            if not common_targets:
                skipped_count += 1
                continue

            for target_idx in common_targets:
                q_neg, k_neg = neg_by_target[target_idx]
                q_pos, k_pos = pos_by_target[target_idx]
                if k_neg < 0 or k_neg >= score_neg.shape[2] or k_pos < 0 or k_pos >= score_pos.shape[2]:
                    skipped_count += 1
                    continue
                s_neg = score_neg[batch_idx, q_neg, k_neg]
                s_pos = score_pos[rank_row, q_pos, k_pos]
                rank_value = F.relu(s_neg - s_pos + self.stage_b_rank_margin)
                losses.append(rank_value)
                used_count += 1
                if float(rank_value.detach().item()) > 0:
                    violation_count += 1

        if losses:
            rank_loss = torch.stack(losses).mean()
        else:
            rank_loss = zero
        metrics = {
            "loss_phrase_rank": rank_loss,
            "phrase_rank_loss_raw": rank_loss.detach(),
            "phrase_rank_pair_count": torch.as_tensor(float(pair_count), device=device),
            "phrase_rank_used_pair_count": torch.as_tensor(float(used_count), device=device),
            "phrase_rank_violation_count": torch.as_tensor(float(violation_count), device=device),
            "phrase_rank_skipped_pair_count": torch.as_tensor(float(skipped_count), device=device),
            "phrase_rank_candidate_tn_count": candidate_tn_count.detach(),
            "phrase_rank_missing_positive_count": missing_positive_count.detach(),
            "phrase_rank_invalid_positive_count": invalid_positive_count.detach(),
            "phrase_rank_margin": torch.as_tensor(float(self.stage_b_rank_margin), device=device),
        }
        return metrics

    def _zero_text_loss_dict(self, zero: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = zero.detach()
        out = {
            "loss_text": zero,
            "text_token_loss_raw": z,
            "text_phrase_tn_loss_raw": z,
            "content_pos_loss": z,
            "canonical_loss": z,
            "tn_neg_loss": z,
            "text_valid_slot_count": z,
            "text_content_pos_slot_count": z,
            "text_tn_neg_slot_count": z,
            "text_attr_pos_slot_count": z,
            "text_attr_neg_slot_count": z,
            "text_canonical_slot_count": z,
            "text_phrase_tn_slot_count": z,
            "text_skipped_tn_slot_count": z,
            "effective_content_token_count": z,
            "effective_tn_negative_token_count": z,
            "empty_content_mask_count": z,
            "empty_tn_negative_mask_count": z,
            "text_tn_neg_weight": z,
            "text_tn_neg_weight_mode_token_count": z,
            "text_tn_content_weight": z,
            "text_tn_canonical_weight": z,
            "spatial_like_tn_count": z,
            "relation_action_like_tn_count": z,
        }
        for group_name in _TN_GROUP_NAMES:
            out[f"loss_tn_{group_name}"] = z
            out[f"tn_neg_count_{group_name}"] = z
            out[f"tn_nonempty_mask_count_{group_name}"] = z
        return out

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        match_ctx = self.compute_matching(outputs, targets)
        losses = self.patch_criterion.compute_losses_from_matching(
            match_ctx,
            targets,
            include_box_losses=self.stage_b_text_loss_type != "allquery_focal_tn_empty_det",
        )
        if self.stage_b_text_loss_type == "allquery_focal_tn_empty_det":
            losses.update(self._compute_det_box_losses(outputs, targets, match_ctx))
        if self.lambda_text <= 0:
            pred_logits_patch = outputs.get("pred_logits_patch", None)
            if pred_logits_patch is not None:
                zero = pred_logits_patch.sum() * 0.0
            else:
                zero = outputs["pred_boxes"].sum() * 0.0
            losses.update(self._zero_text_loss_dict(zero))
        else:
            losses.update(self._compute_text_loss(outputs, targets, match_ctx))
        for aux_idx, aux_outputs in enumerate(outputs.get("aux_outputs", []) or []):
            if aux_idx < self.stage_b_aux_loss_start_idx:
                continue
            aux_match_ctx = self.compute_matching(aux_outputs, targets)
            aux_suffix = f"_{aux_idx}"
            aux_patch_losses = self.patch_criterion.compute_losses_from_matching(
                aux_match_ctx,
                targets,
                include_box_losses=self.stage_b_text_loss_type != "allquery_focal_tn_empty_det",
            )
            if "loss_patch_ce" in aux_patch_losses:
                losses[f"loss_patch_ce{aux_suffix}"] = aux_patch_losses["loss_patch_ce"]
            if self.stage_b_text_loss_type == "allquery_focal_tn_empty_det":
                aux_patch_losses.update(self._compute_det_box_losses(aux_outputs, targets, aux_match_ctx))
            for key in ("loss_bbox", "loss_giou"):
                if key in aux_patch_losses:
                    losses[f"{key}{aux_suffix}"] = aux_patch_losses[key]
            if self.lambda_text <= 0:
                zero = aux_outputs["pred_boxes"].sum() * 0.0
                aux_text_losses = self._zero_text_loss_dict(zero)
            else:
                aux_text_losses = self._compute_text_loss(aux_outputs, targets, aux_match_ctx)
            losses[f"loss_text{aux_suffix}"] = aux_text_losses["loss_text"]
            if self.stage_b_score_calib_aux_loss:
                aux_score_outputs = dict(aux_outputs)
                for key in ("rank_pos_outputs", "rank_pos_targets", "rank_pair_map"):
                    if key in outputs:
                        aux_score_outputs[key] = outputs[key]
                aux_score_losses = self._compute_score_calib_loss(aux_score_outputs, targets, aux_match_ctx)
                for key, value in aux_score_losses.items():
                    losses[f"{key}{aux_suffix}"] = value
        losses.update(self._compute_score_calib_loss(outputs, targets, match_ctx))
        losses.update(self._compute_phrase_rank_loss(outputs, targets, match_ctx))
        return losses
