from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .patch_hungarian_criterion import PatchHungarianCriterion
from .stage_b_score import compute_stage_b_slot_logits


_TN_GROUP_NAMES = ("color_like", "attr_like", "spatial_like", "relation_action_like", "other")
_TN_GROUP_TO_ID = {name: idx for idx, name in enumerate(_TN_GROUP_NAMES)}
_TN_ID_TO_GROUP = {idx: name for name, idx in _TN_GROUP_TO_ID.items()}


def _tn_group_name_from_id(group_id: int) -> str:
    return _TN_ID_TO_GROUP.get(int(group_id), "other")


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
        canonical_pos_weight: float = 0.15,
        stage_b_rank_margin: float = 0.3,
        stage_b_rank_loss_coef: float = 1.0,
        stage_b_rank_detach_patch: bool = True,
        stage_b_rank_beta: float = 1.0,
        stage_b_rank_canonical_weight: float = 0.15,
        stage_b_rank_text_agg: str = "mean",
        stage_b_rank_softmin_tau: float = 0.7,
        stage_b_rank_mean_softmin_alpha: float = 0.5,
        # Deprecated compatibility args. Content-positive and TN-negative tokens
        # are fixed at weight 1.0, and softmin phrase TN loss is disabled.
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
        self.stage_b_rank_margin = float(stage_b_rank_margin)
        self.stage_b_rank_loss_coef = float(stage_b_rank_loss_coef)
        self.stage_b_rank_detach_patch = bool(stage_b_rank_detach_patch)
        self.stage_b_rank_beta = float(stage_b_rank_beta)
        self.stage_b_rank_canonical_weight = float(stage_b_rank_canonical_weight)
        self.stage_b_rank_text_agg = str(stage_b_rank_text_agg)
        self.stage_b_rank_softmin_tau = float(stage_b_rank_softmin_tau)
        self.stage_b_rank_mean_softmin_alpha = float(stage_b_rank_mean_softmin_alpha)
        patch_weight_dict = getattr(patch_criterion, "weight_dict", {}) or {}
        self.weight_dict = {
            "loss_patch_ce": float(lambda_patch),
            "loss_text": float(lambda_text),
            "loss_phrase_rank": float(stage_b_rank_loss_coef),
            "loss_bbox": float(patch_weight_dict.get("loss_bbox", 0.0)),
            "loss_giou": float(patch_weight_dict.get("loss_giou", 0.0)),
        }

    def compute_matching(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        return self.patch_criterion.compute_matching(outputs, targets)

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

    def _compute_text_loss(self, outputs, targets, match_ctx):
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
            if "phrase_to_token_mask" not in targets[b]:
                raise KeyError("Stage B targets must include 'phrase_to_token_mask'.")
            if "canonical_to_token_mask" not in targets[b]:
                raise KeyError("Stage B targets must include 'canonical_to_token_mask'.")

            phrase_to_token_mask = targets[b]["phrase_to_token_mask"].to(device=device).to(torch.bool)
            canonical_to_token_mask = targets[b]["canonical_to_token_mask"].to(device=device).to(torch.bool)
            content_to_token_mask = targets[b].get("content_to_token_mask", None)
            attr_pos_to_token_mask = targets[b].get("attr_pos_to_token_mask", None)
            attr_neg_to_token_mask = targets[b].get("attr_neg_to_token_mask", None)
            attr_neg_weight_mask = targets[b].get("attr_neg_weight_mask", None)
            is_tn_mask = targets[b].get("is_tn", None)
            negative_to_token_mask = targets[b].get("negative_to_token_mask", None)
            tn_group_ids = targets[b].get("tn_group_ids", None)

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
                    f"Token mask length mismatch: phrase={tuple(phrase_to_token_mask.shape)} "
                    f"canonical={tuple(canonical_to_token_mask.shape)} pred_logits_text={tuple(pred_logits_text.shape)}"
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
                    tn_neg_slot_losses.append(neg_loss)
                    tn_group_slot_losses[tn_group_name].append(neg_loss)
                    neg_token_count = int(effective_negative_mask.sum().item())
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

        match_ctx_pos = self.patch_criterion.compute_matching(rank_pos_outputs, rank_pos_targets)
        score_neg = compute_stage_b_slot_logits(
            outputs,
            beta=self.stage_b_rank_beta,
            canonical_weight=self.stage_b_rank_canonical_weight,
            text_agg=self.stage_b_rank_text_agg,
            softmin_tau=self.stage_b_rank_softmin_tau,
            mean_softmin_alpha=self.stage_b_rank_mean_softmin_alpha,
            detach_patch=self.stage_b_rank_detach_patch,
        )
        score_pos = compute_stage_b_slot_logits(
            rank_pos_outputs,
            beta=self.stage_b_rank_beta,
            canonical_weight=self.stage_b_rank_canonical_weight,
            text_agg=self.stage_b_rank_text_agg,
            softmin_tau=self.stage_b_rank_softmin_tau,
            mean_softmin_alpha=self.stage_b_rank_mean_softmin_alpha,
            detach_patch=self.stage_b_rank_detach_patch,
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
            "spatial_like_tn_count": z,
            "relation_action_like_tn_count": z,
        }
        for group_name in _TN_GROUP_NAMES:
            out[f"loss_tn_{group_name}"] = z
            out[f"tn_neg_count_{group_name}"] = z
            out[f"tn_nonempty_mask_count_{group_name}"] = z
        return out

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        match_ctx = self.patch_criterion.compute_matching(outputs, targets)
        losses = self.patch_criterion.compute_losses_from_matching(match_ctx, targets)
        if self.lambda_text <= 0:
            pred_logits_patch = outputs.get("pred_logits_patch", None)
            if pred_logits_patch is not None:
                zero = pred_logits_patch.sum() * 0.0
            else:
                zero = outputs["pred_boxes"].sum() * 0.0
            losses.update(self._zero_text_loss_dict(zero))
        else:
            losses.update(self._compute_text_loss(outputs, targets, match_ctx))
        losses.update(self._compute_phrase_rank_loss(outputs, targets, match_ctx))
        return losses
