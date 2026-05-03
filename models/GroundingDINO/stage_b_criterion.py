from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .patch_hungarian_criterion import PatchHungarianCriterion


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
        attr_pos_weight: float = 1.0,
        tn_shared_attr_pos_weight: float = 0.75,
        attr_neg_weight: float = 1.0,
        use_phrase_tn_loss: bool = True,
        phrase_score_type: str = "softmin",
        softmin_tau: float = 0.7,
        lambda_phrase: float = 0.3,
    ) -> None:
        super().__init__()
        self.patch_criterion = patch_criterion
        self.matcher = patch_criterion.matcher
        self.lambda_patch = float(lambda_patch)
        self.lambda_text = float(lambda_text)
        self.canonical_pos_weight = float(canonical_pos_weight)
        self.attr_pos_weight = float(attr_pos_weight)
        self.tn_shared_attr_pos_weight = float(tn_shared_attr_pos_weight)
        self.attr_neg_weight = float(attr_neg_weight)
        self.use_phrase_tn_loss = bool(use_phrase_tn_loss)
        self.phrase_score_type = str(phrase_score_type)
        self.softmin_tau = float(softmin_tau)
        self.lambda_phrase = float(lambda_phrase)
        self.weight_dict = {
            "loss_patch_ce": float(lambda_patch),
            "loss_text": float(lambda_text),
            "loss_bbox": 0.0,
            "loss_giou": 0.0,
        }

    def compute_matching(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        return self.patch_criterion.compute_matching(outputs, targets)

    def _softmin_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.phrase_score_type != "softmin":
            raise ValueError(f"Unsupported phrase_score_type={self.phrase_score_type!r}; expected 'softmin'.")
        tau = max(float(self.softmin_tau), 1e-6)
        return -tau * torch.logsumexp(-logits / tau, dim=-1)

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
        attr_pos_slot_losses: List[torch.Tensor] = []
        attr_neg_slot_losses: List[torch.Tensor] = []
        canonical_slot_losses: List[torch.Tensor] = []
        phrase_tn_slot_losses: List[torch.Tensor] = []
        valid_slot_count = 0
        skipped_tn_slot_count = 0

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
            attr_pos_to_token_mask = targets[b].get("attr_pos_to_token_mask", None)
            attr_neg_to_token_mask = targets[b].get("attr_neg_to_token_mask", None)
            phrase_semantic_token_mask = targets[b].get("phrase_semantic_token_mask", None)
            attr_neg_weight_mask = targets[b].get("attr_neg_weight_mask", None)
            is_tn_mask = targets[b].get("is_tn", None)
            negative_to_token_mask = targets[b].get("negative_to_token_mask", None)

            if attr_pos_to_token_mask is not None:
                attr_pos_to_token_mask = attr_pos_to_token_mask.to(device=device).to(torch.bool)
            if attr_neg_to_token_mask is not None:
                attr_neg_to_token_mask = attr_neg_to_token_mask.to(device=device).to(torch.bool)
            if phrase_semantic_token_mask is not None:
                phrase_semantic_token_mask = phrase_semantic_token_mask.to(device=device).to(torch.bool)
            if attr_neg_weight_mask is not None:
                attr_neg_weight_mask = attr_neg_weight_mask.to(device=device, dtype=pred_logits_text.dtype)
            if is_tn_mask is not None:
                is_tn_mask = is_tn_mask.to(device=device).to(torch.bool)
            if negative_to_token_mask is not None:
                negative_to_token_mask = negative_to_token_mask.to(device=device).to(torch.bool)
            T = int(pred_logits_text.shape[-1])
            if phrase_to_token_mask.dim() != 2 or canonical_to_token_mask.dim() != 2:
                raise ValueError("Stage B token masks must be shaped (K,T).")
            if phrase_to_token_mask.shape[-1] != T or canonical_to_token_mask.shape[-1] != T:
                raise ValueError(
                    f"Token mask length mismatch: phrase={tuple(phrase_to_token_mask.shape)} "
                    f"canonical={tuple(canonical_to_token_mask.shape)} pred_logits_text={tuple(pred_logits_text.shape)}"
                )
            optional_masks = {
                "attr_pos_to_token_mask": attr_pos_to_token_mask,
                "attr_neg_to_token_mask": attr_neg_to_token_mask,
                "phrase_semantic_token_mask": phrase_semantic_token_mask,
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

            logits_b = pred_logits_text[b, src_idx]
            for row_idx, slot_idx in enumerate(matched_patch_idx.tolist()):
                slot_idx = int(slot_idx)
                if slot_idx < 0 or slot_idx >= int(phrase_to_token_mask.shape[0]):
                    continue

                phrase_mask = phrase_to_token_mask[slot_idx]
                canonical_mask = canonical_to_token_mask[slot_idx] & phrase_mask
                is_tn_slot = bool(is_tn_mask[slot_idx].item()) if is_tn_mask is not None else False

                if attr_neg_to_token_mask is not None:
                    negative_attr_mask = attr_neg_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)
                elif negative_to_token_mask is not None:
                    negative_attr_mask = negative_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)
                else:
                    negative_attr_mask = torch.zeros_like(phrase_mask)

                if attr_neg_weight_mask is not None:
                    negative_weight = attr_neg_weight_mask[slot_idx].to(dtype=logits_b.dtype) * negative_attr_mask.to(dtype=logits_b.dtype)
                else:
                    negative_weight = negative_attr_mask.to(dtype=logits_b.dtype) * self.attr_neg_weight
                effective_negative_mask = negative_attr_mask & (negative_weight > 0)

                if is_tn_slot and not effective_negative_mask.any():
                    skipped_tn_slot_count += 1
                    continue

                if attr_pos_to_token_mask is not None:
                    attr_pos_mask = attr_pos_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask) & (~effective_negative_mask)
                else:
                    attr_pos_mask = phrase_mask & (~canonical_mask) & (~effective_negative_mask)

                slot_logits = logits_b[row_idx]
                slot_had_loss = False

                attr_pos_loss = self._slot_bce_mean(slot_logits, attr_pos_mask, 1.0)
                if attr_pos_loss is not None:
                    slot_attr_weight = self.tn_shared_attr_pos_weight if is_tn_slot else self.attr_pos_weight
                    attr_pos_slot_losses.append(attr_pos_loss * float(slot_attr_weight))
                    slot_had_loss = True

                neg_loss = self._slot_bce_mean(
                    slot_logits,
                    effective_negative_mask,
                    0.0,
                    weight=negative_weight,
                )
                if neg_loss is not None:
                    attr_neg_slot_losses.append(neg_loss)
                    slot_had_loss = True

                canonical_loss = self._slot_bce_mean(slot_logits, canonical_mask, 1.0)
                if canonical_loss is not None:
                    canonical_slot_losses.append(canonical_loss * self.canonical_pos_weight)
                    slot_had_loss = True

                if (
                    self.use_phrase_tn_loss
                    and is_tn_slot
                    and effective_negative_mask.any()
                    and phrase_semantic_token_mask is not None
                ):
                    semantic_mask = phrase_semantic_token_mask[slot_idx] & phrase_mask
                    semantic_mask = semantic_mask & (canonical_mask | attr_pos_mask | effective_negative_mask)
                    if semantic_mask.any():
                        phrase_logits = slot_logits[semantic_mask]
                        finite = torch.isfinite(phrase_logits)
                        if finite.any():
                            phrase_logits = phrase_logits[finite]
                            phrase_score = self._softmin_logits(phrase_logits)
                            phrase_loss = F.binary_cross_entropy_with_logits(
                                phrase_score,
                                torch.zeros_like(phrase_score),
                                reduction="mean",
                            )
                            phrase_tn_slot_losses.append(phrase_loss)

                if slot_had_loss:
                    valid_slot_count += 1

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero
            return torch.stack(values).mean()

        token_loss = (
            _mean_or_zero(attr_pos_slot_losses)
            + _mean_or_zero(attr_neg_slot_losses)
            + _mean_or_zero(canonical_slot_losses)
        )
        phrase_loss = _mean_or_zero(phrase_tn_slot_losses)
        loss_text = token_loss + float(self.lambda_phrase) * phrase_loss
        return {
            "loss_text": loss_text,
            "text_token_loss_raw": token_loss.detach(),
            "text_phrase_tn_loss_raw": phrase_loss.detach(),
            "text_valid_slot_count": torch.as_tensor(float(valid_slot_count), device=device),
            "text_attr_pos_slot_count": torch.as_tensor(float(len(attr_pos_slot_losses)), device=device),
            "text_attr_neg_slot_count": torch.as_tensor(float(len(attr_neg_slot_losses)), device=device),
            "text_canonical_slot_count": torch.as_tensor(float(len(canonical_slot_losses)), device=device),
            "text_phrase_tn_slot_count": torch.as_tensor(float(len(phrase_tn_slot_losses)), device=device),
            "text_skipped_tn_slot_count": torch.as_tensor(float(skipped_tn_slot_count), device=device),
        }

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        match_ctx = self.patch_criterion.compute_matching(outputs, targets)
        losses = self.patch_criterion.compute_losses_from_matching(match_ctx, targets)
        losses.update(self._compute_text_loss(outputs, targets, match_ctx))
        return losses
