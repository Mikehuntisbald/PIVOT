from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from groundingdino.util import box_ops


def _sigmoid_focal_loss_no_reduce(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    inputs/targets: same shape, binary targets in {0,1}.
    Returns per-element focal loss with no reduction.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


class PatchOnlyCriterion(nn.Module):
    """
    Stage A (patch-only) losses:
      - patch_ce: IoU-based query labeling on the support class.
      - optional box losses (L1 + GIoU) on IoU-positive queries to keep box head stable.
    """

    def __init__(
        self,
        weight_dict: Dict[str, float],
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        patch_iou_thr: float = 0.5,
        patch_lambda_neg: float = 0.25,
        patch_labeling_mode: str = "iou_thr",  # iou_thr | topk_iou
        patch_topk: int = 50,
        patch_topk_iou_thr: float = 0.05,
    ) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.patch_iou_thr = float(patch_iou_thr)
        self.patch_lambda_neg = float(patch_lambda_neg)
        self.patch_labeling_mode = str(patch_labeling_mode).lower().strip()
        self.patch_topk = int(patch_topk)
        self.patch_topk_iou_thr = float(patch_topk_iou_thr)
        if self.patch_labeling_mode not in {"iou_thr", "topk_iou"}:
            raise ValueError(f"Unsupported patch_labeling_mode={patch_labeling_mode!r} (expected iou_thr|topk_iou).")
        if self.patch_topk < 0:
            raise ValueError("patch_topk must be >= 0.")
        if self.patch_topk_iou_thr < 0:
            raise ValueError("patch_topk_iou_thr must be >= 0.")

    @torch.no_grad()
    def _build_patch_labels(
        self,
        pred_boxes_cxcywh: torch.Tensor,  # (Q,4) in [0,1]
        gt_boxes_cxcywh: torch.Tensor,  # (N,4) in [0,1]
        gt_labels: torch.Tensor,  # (N,)
        support_class: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          label_patch: (Q,) in {0,1}
          max_iou: (Q,) max IoU with any positive GT box (0 if none)
        """
        pos_mask = gt_labels == support_class
        if pos_mask.sum() == 0:
            return torch.zeros((pred_boxes_cxcywh.shape[0],), device=pred_boxes_cxcywh.device), torch.zeros(
                (pred_boxes_cxcywh.shape[0],), device=pred_boxes_cxcywh.device
            )

        pos_boxes = gt_boxes_cxcywh[pos_mask]
        # Geometry ops in fp32 for numerical stability under AMP.
        pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_cxcywh.float())
        pos_xyxy = box_ops.box_cxcywh_to_xyxy(pos_boxes.float())
        iou, _ = box_ops.box_iou(pred_xyxy, pos_xyxy)  # (Q, N_pos)
        max_iou = iou.max(dim=1).values
        if self.patch_labeling_mode == "iou_thr":
            label_patch = (max_iou > self.patch_iou_thr).to(pred_boxes_cxcywh.dtype)
        elif self.patch_labeling_mode == "topk_iou":
            # Mark positives by being among top-k IoU queries (optionally gated by a low IoU threshold).
            # This is useful early in training when a strict threshold yields almost no positives.
            k = int(self.patch_topk) if int(self.patch_topk) > 0 else int(max_iou.numel())
            k = min(k, int(max_iou.numel()))
            topk_idx = torch.topk(max_iou, k=k, largest=True).indices
            keep = max_iou > float(self.patch_topk_iou_thr)
            label_patch = torch.zeros((pred_boxes_cxcywh.shape[0],), device=pred_boxes_cxcywh.device, dtype=pred_boxes_cxcywh.dtype)
            if keep.any():
                pos_idx = topk_idx[keep[topk_idx]]
                if pos_idx.numel() > 0:
                    label_patch[pos_idx] = 1.0
        else:
            raise RuntimeError(f"Unknown patch_labeling_mode={self.patch_labeling_mode}")
        return label_patch, max_iou.to(pred_boxes_cxcywh.dtype)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        if "pred_logits_patch" not in outputs or outputs["pred_logits_patch"] is None:
            raise KeyError("PatchOnlyCriterion requires outputs['pred_logits_patch'] (B,Q) or (B,Q,K).")
        if "pred_boxes" not in outputs:
            raise KeyError("PatchOnlyCriterion requires outputs['pred_boxes'] (B,Q,4).")

        pred_logits_patch = outputs["pred_logits_patch"]  # (B,Q) or (B,Q,K)
        pred_boxes = outputs["pred_boxes"]  # (B,Q,4) normalized cxcywh

        if pred_logits_patch.dim() == 2:
            B, Q = pred_logits_patch.shape
            K = 1
            pred_logits_patch_ = pred_logits_patch.unsqueeze(-1)  # (B,Q,1)
        elif pred_logits_patch.dim() == 3:
            B, Q, K = pred_logits_patch.shape
            pred_logits_patch_ = pred_logits_patch
        else:
            raise ValueError(f"pred_logits_patch must be (B,Q) or (B,Q,K), got {tuple(pred_logits_patch.shape)}")

        device = pred_logits_patch_.device
        label_patch = torch.zeros((B, Q, K), device=device, dtype=pred_logits_patch_.dtype)
        max_iou = torch.zeros((B, Q, K), device=device, dtype=pred_logits_patch_.dtype)

        for b in range(B):
            gt_boxes = targets[b].get("boxes", None)
            gt_labels = targets[b].get("labels", None)
            if gt_boxes is None or gt_labels is None:
                raise KeyError("targets[b] must have 'boxes' and 'labels'.")

            support_classes_t = targets[b].get("support_classes", None)
            if support_classes_t is None:
                support_class_t = targets[b].get("support_class", None)
                if support_class_t is None:
                    raise KeyError(
                        "targets[b] must have 'support_class' (single-patch) or 'support_classes' (multi-patch)."
                    )
                support_classes_t = support_class_t.view(1).repeat(K)
            support_classes_t = support_classes_t.to(device=device).view(-1)
            if support_classes_t.numel() == 1 and K > 1:
                support_classes_t = support_classes_t.repeat(K)
            if support_classes_t.numel() != K:
                raise ValueError(
                    f"targets[b]['support_classes'] must have length K={K}, got shape {tuple(support_classes_t.shape)}"
                )

            for k in range(K):
                support_class = int(support_classes_t[k].item())
                if support_class < 0:
                    continue
                label_b, max_iou_b = self._build_patch_labels(
                    pred_boxes_cxcywh=pred_boxes[b],
                    gt_boxes_cxcywh=gt_boxes,
                    gt_labels=gt_labels,
                    support_class=support_class,
                )
                label_patch[b, :, k] = label_b
                max_iou[b, :, k] = max_iou_b

        loss_mat = _sigmoid_focal_loss_no_reduce(
            pred_logits_patch_, label_patch, alpha=self.focal_alpha, gamma=self.focal_gamma
        )  # (B,Q,K)

        patch_mask = outputs.get("patch_mask", None)
        if patch_mask is not None and torch.is_tensor(patch_mask) and patch_mask.dim() == 2 and patch_mask.shape[0] == B:
            if patch_mask.shape[1] == K:
                loss_mat = loss_mat * patch_mask[:, None, :].to(loss_mat.dtype)
                label_patch = label_patch * patch_mask[:, None, :].to(label_patch.dtype)
                max_iou = max_iou * patch_mask[:, None, :].to(max_iou.dtype)

        pos = label_patch > 0.5
        neg = ~pos

        pos_loss = loss_mat[pos].mean() if pos.any() else torch.zeros((), device=device)
        neg_loss = loss_mat[neg].mean() if neg.any() else torch.zeros((), device=device)
        loss_patch_ce = pos_loss + self.patch_lambda_neg * neg_loss

        losses: Dict[str, torch.Tensor] = {
            "loss_patch_ce": loss_patch_ce,
            "patch_pos_frac": pos.float().mean(),
            "patch_max_iou_mean": max_iou.mean(),
        }

        if ("loss_bbox" in self.weight_dict and self.weight_dict["loss_bbox"] > 0) or (
            "loss_giou" in self.weight_dict and self.weight_dict["loss_giou"] > 0
        ):
            loss_bbox_total = torch.zeros((), device=device)
            loss_giou_total = torch.zeros((), device=device)
            count = 0

            for b in range(B):
                gt_boxes = targets[b]["boxes"]
                gt_labels = targets[b]["labels"]

                support_classes_t = targets[b].get("support_classes", None)
                if support_classes_t is None:
                    support_class_t = targets[b].get("support_class", None)
                    if support_class_t is None:
                        continue
                    support_classes_t = support_class_t.view(1).repeat(K)
                support_classes_t = support_classes_t.to(device=device).view(-1)
                if support_classes_t.numel() == 1 and K > 1:
                    support_classes_t = support_classes_t.repeat(K)

                for k in range(K):
                    support_class = int(support_classes_t[k].item())
                    if support_class < 0:
                        continue
                    pos_gt_mask = gt_labels == support_class
                    if pos_gt_mask.sum() == 0:
                        continue
                    pos_gt_boxes = gt_boxes[pos_gt_mask]

                    pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[b].float())
                    gt_xyxy = box_ops.box_cxcywh_to_xyxy(pos_gt_boxes.float())
                    iou, _ = box_ops.box_iou(pred_xyxy, gt_xyxy)
                    best_iou, best_idx = iou.max(dim=1)
                    pos_q = best_iou > self.patch_iou_thr
                    if pos_q.sum() == 0:
                        continue

                    pred_pos = pred_boxes[b][pos_q].float()
                    tgt_pos = pos_gt_boxes[best_idx[pos_q]].float()

                    loss_bbox_total = loss_bbox_total + F.l1_loss(pred_pos, tgt_pos, reduction="none").sum(-1).mean()

                    pred_xyxy_pos = box_ops.box_cxcywh_to_xyxy(pred_pos)
                    tgt_xyxy_pos = box_ops.box_cxcywh_to_xyxy(tgt_pos)
                    giou = box_ops.generalized_box_iou_pairwise(pred_xyxy_pos, tgt_xyxy_pos)
                    loss_giou_total = loss_giou_total + (1.0 - giou).mean()
                    count += 1

            if count > 0:
                losses["loss_bbox"] = loss_bbox_total / count
                losses["loss_giou"] = loss_giou_total / count
            else:
                losses["loss_bbox"] = torch.zeros((), device=device)
                losses["loss_giou"] = torch.zeros((), device=device)

        return losses
