from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from groundingdino.util import box_ops
from util.misc import get_world_size, is_dist_avail_and_initialized


def _sigmoid_focal_loss_no_reduce(
    inputs: torch.Tensor,
    targets: torch.Tensor,
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


class PatchHungarianCriterion(nn.Module):
    """
    Patch-only criterion using Hungarian matching cost (classification + bbox + giou),
    while keeping GT labels in canonical-id space.

    Expected outputs:
      - outputs['pred_logits_patch']: (B,Q) or (B,Q,K)
      - outputs['pred_boxes']: (B,Q,4) normalized cxcywh
      - outputs['patch_mask'] (optional): (B,K) bool, True for valid patch channels

    Expected targets (per image):
      - targets[b]['labels']: (N,) canonical class ids
      - targets[b]['boxes']:  (N,4) normalized cxcywh
      - targets[b]['support_classes']: (K,) canonical ids for each patch channel
        (or targets[b]['support_class'] for single-patch)
    """

    def __init__(
        self,
        *,
        matcher,
        weight_dict: Dict[str, float],
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @torch.no_grad()
    def _build_label_map(self, support_classes: torch.Tensor, *, K: int, num_rows: int) -> torch.Tensor:
        """
        Build label_map[cid] -> one-hot over K channels.
        Returned on CPU to match matcher implementation (it indexes with tgt_ids.cpu()).
        """
        label_map = torch.zeros((int(num_rows), int(K)), dtype=torch.float32)
        support_classes = support_classes.view(-1).to(torch.long)
        for k, cid in enumerate(support_classes.tolist()):
            if cid < 0:
                continue
            if cid >= int(num_rows):
                continue
            label_map[int(cid), int(k)] = 1.0
        return label_map

    def _get_support_classes(
        self,
        target: Dict[str, torch.Tensor],
        *,
        K: int,
        device: torch.device,
    ) -> torch.Tensor:
        support_classes = target.get("support_classes", None)
        if support_classes is None:
            support_class = target.get("support_class", None)
            if support_class is None:
                raise KeyError("targets[b] must have 'support_classes' or 'support_class'.")
            support_classes = support_class.view(1).repeat(K)
        support_classes = support_classes.to(device=device).view(-1)
        if support_classes.numel() < K:
            pad = torch.full((K - int(support_classes.numel()),), -1, dtype=support_classes.dtype, device=device)
            support_classes = torch.cat([support_classes, pad], dim=0)
        return support_classes[:K]

    @staticmethod
    def _truthy_flag(value, batch_index: int | None = None) -> bool:
        if value is None:
            return False
        if torch.is_tensor(value):
            if value.numel() == 0:
                return False
            v = value.detach()
            if batch_index is not None and v.dim() > 0 and int(v.shape[0]) > batch_index:
                v = v[batch_index]
            return bool(v.reshape(-1)[0].item())
        return bool(value)

    def _check_duplicate_support_classes(
        self,
        support_classes: torch.Tensor,
        valid_k: torch.Tensor,
        *,
        batch_index: int,
        allow_duplicate: bool,
    ) -> None:
        if allow_duplicate:
            return
        valid_classes = support_classes[valid_k]
        valid_classes = valid_classes[valid_classes >= 0].to(torch.long)
        if valid_classes.numel() <= 1:
            return
        unique, counts = torch.unique(valid_classes, sorted=True, return_counts=True)
        dup = unique[counts > 1]
        if dup.numel() == 0:
            return
        dup_list = [int(x) for x in dup.detach().cpu().tolist()]
        raise ValueError(
            f"Duplicate support class cid(s) {dup_list} in batch index {batch_index}. "
            "If multi-exemplar-per-class is intentional, set allow_duplicate_support_classes=True."
        )

    def compute_matching(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        if "pred_logits_patch" not in outputs or outputs["pred_logits_patch"] is None:
            raise KeyError("PatchHungarianCriterion requires outputs['pred_logits_patch'] (B,Q) or (B,Q,K).")
        if "pred_boxes" not in outputs:
            raise KeyError("PatchHungarianCriterion requires outputs['pred_boxes'] (B,Q,4).")

        pred_logits_patch = outputs["pred_logits_patch"]
        pred_boxes = outputs["pred_boxes"]
        device = pred_boxes.device

        if pred_logits_patch.dim() == 2:
            B, Q = pred_logits_patch.shape
            K = 1
            pred_logits_patch = pred_logits_patch.unsqueeze(-1)
        elif pred_logits_patch.dim() == 3:
            B, Q, K = pred_logits_patch.shape
        else:
            raise ValueError(f"pred_logits_patch must be (B,Q) or (B,Q,K), got {tuple(pred_logits_patch.shape)}")

        patch_mask = outputs.get("patch_mask", None)
        if patch_mask is not None:
            patch_mask = patch_mask.to(device=device).to(torch.bool)
            if patch_mask.shape != (B, K):
                raise ValueError(f"outputs['patch_mask'] must be (B,K)={(B, K)}, got {tuple(patch_mask.shape)}")

        num_boxes = sum(int(t["labels"].numel()) for t in targets)
        num_boxes_t = torch.as_tensor([num_boxes], dtype=torch.float32, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_t)
        num_boxes = float(torch.clamp(num_boxes_t / get_world_size(), min=1.0).item())

        all_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        matched_local_patch_idx_list: List[torch.Tensor] = []
        matched_patch_idx_list: List[torch.Tensor] = []

        for b in range(B):
            tgt_labels = targets[b]["labels"].to(torch.long)
            tgt_boxes = targets[b]["boxes"].to(torch.float32)
            if tgt_labels.numel() == 0:
                empty = torch.zeros((0,), dtype=torch.int64, device=device)
                all_indices.append((empty, empty))
                matched_local_patch_idx_list.append(empty)
                matched_patch_idx_list.append(empty)
                continue

            support_classes = self._get_support_classes(targets[b], K=K, device=device)
            valid_k = support_classes >= 0
            if patch_mask is not None:
                valid_k = valid_k & patch_mask[b]
            allow_duplicate = self._truthy_flag(outputs.get("allow_duplicate_support_classes", None), b) or self._truthy_flag(
                targets[b].get("allow_duplicate_support_classes", None)
            )
            self._check_duplicate_support_classes(
                support_classes,
                valid_k,
                batch_index=b,
                allow_duplicate=allow_duplicate,
            )
            if int(valid_k.sum().item()) == 0:
                empty = torch.zeros((0,), dtype=torch.int64, device=device)
                all_indices.append((empty, empty))
                matched_local_patch_idx_list.append(empty)
                matched_patch_idx_list.append(empty)
                continue

            keep = valid_k.nonzero(as_tuple=False).flatten()
            logits_b = pred_logits_patch[b][:, keep]
            boxes_b = pred_boxes[b]

            max_row = int(max(int(tgt_labels.max().item()), int(support_classes[valid_k].max().item()))) + 1
            label_map = self._build_label_map(support_classes[keep], K=int(keep.numel()), num_rows=max_row)

            for_match = {"pred_logits": logits_b.unsqueeze(0), "pred_boxes": boxes_b.unsqueeze(0)}
            src_idx, tgt_idx = self.matcher(
                for_match, [{"labels": tgt_labels, "boxes": tgt_boxes}], label_map
            )[0]
            all_indices.append((src_idx, tgt_idx))

            cid_to_local_k = {}
            cid_to_full_k = {}
            for local_k, cid in enumerate(support_classes[keep].tolist()):
                cid_to_local_k[int(cid)] = int(local_k)
                cid_to_full_k[int(cid)] = int(keep[local_k].item())
            matched_labels = tgt_labels[tgt_idx]
            matched_local_patch_idx = torch.as_tensor(
                [cid_to_local_k[int(x.item())] for x in matched_labels], device=device, dtype=torch.int64
            )
            matched_patch_idx = torch.as_tensor(
                [cid_to_full_k[int(x.item())] for x in matched_labels], device=device, dtype=torch.int64
            )
            matched_local_patch_idx_list.append(matched_local_patch_idx)
            matched_patch_idx_list.append(matched_patch_idx)

        return {
            "pred_logits_patch": pred_logits_patch,
            "pred_boxes": pred_boxes,
            "patch_mask": patch_mask,
            "num_boxes": num_boxes,
            "all_indices": all_indices,
            "matched_local_patch_idx_list": matched_local_patch_idx_list,
            "matched_patch_idx_list": matched_patch_idx_list,
            "B": B,
            "Q": Q,
            "K": K,
        }

    def compute_losses_from_matching(self, match_ctx, targets: List[Dict[str, torch.Tensor]]):
        pred_logits_patch = match_ctx["pred_logits_patch"]
        pred_boxes = match_ctx["pred_boxes"]
        patch_mask = match_ctx["patch_mask"]
        num_boxes = float(match_ctx["num_boxes"])
        all_indices = match_ctx["all_indices"]
        matched_local_patch_idx_list = match_ctx["matched_local_patch_idx_list"]
        B = int(match_ctx["B"])
        Q = int(match_ctx["Q"])
        K = int(match_ctx["K"])
        device = pred_boxes.device

        loss_cls = torch.zeros((), device=device)
        matched_total = 0
        for b in range(B):
            support_classes = self._get_support_classes(targets[b], K=K, device=device)
            valid_k = support_classes >= 0
            if patch_mask is not None:
                valid_k = valid_k & patch_mask[b]
            keep = valid_k.nonzero(as_tuple=False).flatten()
            if keep.numel() == 0:
                continue

            logits_b = pred_logits_patch[b][:, keep]
            target_b = torch.zeros_like(logits_b)

            src_idx, _tgt_idx = all_indices[b]
            if src_idx.numel() > 0:
                target_b[src_idx, matched_local_patch_idx_list[b]] = 1.0
                matched_total += int(src_idx.numel())

            loss_mat = _sigmoid_focal_loss_no_reduce(
                logits_b, target_b, alpha=self.focal_alpha, gamma=self.focal_gamma
            )
            loss_cls = loss_cls + loss_mat.sum()

        loss_patch_ce = loss_cls / num_boxes

        losses: Dict[str, torch.Tensor] = {
            "loss_patch_ce": loss_patch_ce,
            "patch_match_frac": torch.as_tensor(matched_total / float(B * Q), device=device),
        }

        if ("loss_bbox" in self.weight_dict and self.weight_dict["loss_bbox"] > 0) or (
            "loss_giou" in self.weight_dict and self.weight_dict["loss_giou"] > 0
        ):
            idx = self._get_src_permutation_idx(all_indices)
            src_boxes = pred_boxes[idx]
            tgt_boxes_cat = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, all_indices)], dim=0)
            if tgt_boxes_cat.numel() == 0:
                losses["loss_bbox"] = torch.zeros((), device=device)
                losses["loss_giou"] = torch.zeros((), device=device)
            else:
                losses["loss_bbox"] = F.l1_loss(src_boxes, tgt_boxes_cat, reduction="none").sum() / num_boxes
                giou = 1.0 - torch.diag(
                    box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(src_boxes),
                        box_ops.box_cxcywh_to_xyxy(tgt_boxes_cat),
                    )
                )
                losses["loss_giou"] = giou.sum() / num_boxes
        return losses

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        match_ctx = self.compute_matching(outputs, targets)
        return self.compute_losses_from_matching(match_ctx, targets)
