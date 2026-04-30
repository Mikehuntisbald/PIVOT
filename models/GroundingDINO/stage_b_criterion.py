from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .patch_hungarian_criterion import PatchHungarianCriterion, _sigmoid_focal_loss_no_reduce


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
        attr_neg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.patch_criterion = patch_criterion
        self.matcher = patch_criterion.matcher
        self.lambda_patch = float(lambda_patch)
        self.lambda_text = float(lambda_text)
        self.canonical_pos_weight = float(canonical_pos_weight)
        self.attr_pos_weight = float(attr_pos_weight)
        self.attr_neg_weight = float(attr_neg_weight)
        self.weight_dict = {
            "loss_patch_ce": float(lambda_patch),
            "loss_text": float(lambda_text),
            "loss_bbox": 0.0,
            "loss_giou": 0.0,
        }

    def compute_matching(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        return self.patch_criterion.compute_matching(outputs, targets)

    def _compute_text_loss(self, outputs, targets, match_ctx):
        pred_logits_text = outputs.get("pred_logits_text", None)
        if pred_logits_text is None:
            raise KeyError("StageBCriterion requires outputs['pred_logits_text'] in Stage B.")
        if pred_logits_text.dim() != 3:
            raise ValueError(
                f"outputs['pred_logits_text'] must be (B,Q,T), got {tuple(pred_logits_text.shape)}"
            )

        device = pred_logits_text.device
        total_weighted_loss = torch.zeros((), device=device)
        total_token_weight = torch.zeros((), device=device)
        valid_slot_count = 0

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
            negative_to_token_mask = targets[b].get("negative_to_token_mask", None)
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
            if negative_to_token_mask is not None:
                if negative_to_token_mask.dim() != 2 or negative_to_token_mask.shape != phrase_to_token_mask.shape:
                    raise ValueError(
                        f"negative_to_token_mask must match phrase_to_token_mask, got "
                        f"{tuple(negative_to_token_mask.shape)} vs {tuple(phrase_to_token_mask.shape)}"
                    )

            logits_b = pred_logits_text[b, src_idx]
            for row_idx, slot_idx in enumerate(matched_patch_idx.tolist()):
                slot_idx = int(slot_idx)
                if slot_idx < 0 or slot_idx >= int(phrase_to_token_mask.shape[0]):
                    continue

                phrase_mask = phrase_to_token_mask[slot_idx]
                canonical_mask = canonical_to_token_mask[slot_idx] & phrase_mask
                negative_attr_mask = torch.zeros_like(phrase_mask)
                if negative_to_token_mask is not None:
                    negative_attr_mask = negative_to_token_mask[slot_idx] & phrase_mask & (~canonical_mask)

                attr_mask = phrase_mask & (~canonical_mask) & (~negative_attr_mask)
                positive_token_mask = attr_mask | canonical_mask
                if not positive_token_mask.any():
                    if not negative_attr_mask.any():
                        continue

                positive_token_weight = (
                    attr_mask.to(dtype=logits_b.dtype) * self.attr_pos_weight
                    + canonical_mask.to(dtype=logits_b.dtype) * self.canonical_pos_weight
                )
                negative_token_weight = negative_attr_mask.to(dtype=logits_b.dtype) * self.attr_neg_weight
                token_weight = positive_token_weight + negative_token_weight
                if float(token_weight.sum().item()) <= 0.0:
                    continue

                valid_token_mask = token_weight > 0
                if not valid_token_mask.any():
                    continue

                token_target = positive_token_mask.to(dtype=logits_b.dtype)
                token_logits = logits_b[row_idx][valid_token_mask]
                token_target = token_target[valid_token_mask]
                token_weight_valid = token_weight[valid_token_mask]

                if not torch.isfinite(token_logits).all():
                    finite_mask = torch.isfinite(token_logits)
                    if not finite_mask.any():
                        continue
                    token_logits = token_logits[finite_mask]
                    token_target = token_target[finite_mask]
                    token_weight_valid = token_weight_valid[finite_mask]
                    if token_weight_valid.numel() == 0:
                        continue

                token_loss = _sigmoid_focal_loss_no_reduce(
                    token_logits,
                    token_target,
                    alpha=self.patch_criterion.focal_alpha,
                    gamma=self.patch_criterion.focal_gamma,
                )
                total_weighted_loss = total_weighted_loss + (token_loss * token_weight_valid).sum()
                total_token_weight = total_token_weight + token_weight_valid.sum()
                valid_slot_count += 1

        loss_text = total_weighted_loss / total_token_weight.clamp(min=1e-6)
        return {
            "loss_text": loss_text,
            "text_valid_token_weight": total_token_weight.detach(),
            "text_valid_slot_count": torch.as_tensor(float(valid_slot_count), device=device),
        }

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        match_ctx = self.patch_criterion.compute_matching(outputs, targets)
        losses = self.patch_criterion.compute_losses_from_matching(match_ctx, targets)
        losses.update(self._compute_text_loss(outputs, targets, match_ctx))
        return losses
