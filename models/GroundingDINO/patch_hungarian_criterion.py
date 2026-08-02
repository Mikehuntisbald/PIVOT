from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

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
        patch_ce_reduction: str = "legacy",
        patch_lambda_neg: float = 0.25,
        patch_ce_neg_topk: int = 0,
        patch_ce_neg_topk_ratio: float = 0.0,
        patch_rank_margin: float = 0.3,
        patch_rank_hard_negatives: int = 16,
        patch_rank_include_wrong_slots: bool = True,
        patch_rank_wrong_slot_weight: float = 0.5,
        patch_ce_positive_only_for_datasets=(),
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.patch_ce_reduction = str(patch_ce_reduction).lower().strip()
        self.patch_lambda_neg = float(patch_lambda_neg)
        self.patch_ce_neg_topk = max(0, int(patch_ce_neg_topk))
        self.patch_ce_neg_topk_ratio = max(0.0, float(patch_ce_neg_topk_ratio))
        self.patch_rank_margin = float(patch_rank_margin)
        self.patch_rank_hard_negatives = max(0, int(patch_rank_hard_negatives))
        self.patch_rank_include_wrong_slots = bool(patch_rank_include_wrong_slots)
        self.patch_rank_wrong_slot_weight = float(patch_rank_wrong_slot_weight)
        if isinstance(patch_ce_positive_only_for_datasets, str):
            patch_ce_positive_only_for_datasets = [
                x.strip() for x in patch_ce_positive_only_for_datasets.split(",") if x.strip()
            ]
        self.patch_ce_positive_only_for_datasets = tuple(
            str(x).lower().strip().replace("_", "").replace("-", "").replace("+", "plus")
            for x in (patch_ce_positive_only_for_datasets or ())
            if str(x).strip()
        )
        if self.patch_ce_reduction not in {"legacy", "posneg_mean", "posneg_topk"}:
            raise ValueError(
                "patch_ce_reduction must be one of: legacy, posneg_mean, posneg_topk; "
                f"got {patch_ce_reduction!r}."
            )

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

    def _target_uses_positive_only_patch_ce(self, target: Dict[str, torch.Tensor]) -> bool:
        flag = target.get("patch_ce_positive_only", None)
        if torch.is_tensor(flag) and flag.numel() > 0:
            return bool(flag.detach().reshape(-1)[0].item())
        if not self.patch_ce_positive_only_for_datasets:
            return False
        names = []
        for key in ("dataset_name", "source", "pair_source"):
            value = target.get(key, None)
            if value is None:
                continue
            if torch.is_tensor(value):
                continue
            if isinstance(value, (list, tuple, set)):
                names.extend(str(v) for v in value)
            else:
                names.append(str(value))
        for name in names:
            norm = name.lower().replace("_", "").replace("-", "").replace("+", "plus")
            if any(tag in norm for tag in self.patch_ce_positive_only_for_datasets):
                return True
        return False

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

    def compute_matching(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        *,
        sync_num_boxes: bool = True,
    ):
        if "pred_logits_patch" not in outputs or outputs["pred_logits_patch"] is None:
            raise KeyError("PatchHungarianCriterion requires outputs['pred_logits_patch'] (B,Q) or (B,Q,K).")
        if "pred_boxes" not in outputs:
            raise KeyError("PatchHungarianCriterion requires outputs['pred_boxes'] (B,Q,4).")

        pred_logits_patch = outputs["pred_logits_patch"]
        pred_scores_match = outputs.get("pred_scores_match", None)
        pred_match_cost = outputs.get("pred_match_cost", None)
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
        if pred_scores_match is not None:
            if pred_scores_match.dim() == 2:
                pred_scores_match = pred_scores_match.unsqueeze(-1)
            elif pred_scores_match.dim() != 3:
                raise ValueError(
                    f"pred_scores_match must be (B,Q) or (B,Q,K), got {tuple(pred_scores_match.shape)}"
                )
            if pred_scores_match.shape != pred_logits_patch.shape:
                raise ValueError(
                    "pred_scores_match must match pred_logits_patch shape after unsqueeze, got "
                    f"{tuple(pred_scores_match.shape)} vs {tuple(pred_logits_patch.shape)}"
                )
            pred_scores_match = pred_scores_match.to(device=device, dtype=pred_logits_patch.dtype)
        if pred_match_cost is not None:
            if pred_match_cost.dim() == 2:
                pred_match_cost = pred_match_cost.unsqueeze(-1)
            elif pred_match_cost.dim() != 3:
                raise ValueError(
                    f"pred_match_cost must be (B,Q) or (B,Q,K), got {tuple(pred_match_cost.shape)}"
                )
            if pred_match_cost.shape != pred_logits_patch.shape:
                raise ValueError(
                    "pred_match_cost must match pred_logits_patch shape after unsqueeze, got "
                    f"{tuple(pred_match_cost.shape)} vs {tuple(pred_logits_patch.shape)}"
                )
            pred_match_cost = pred_match_cost.to(device=device, dtype=pred_logits_patch.dtype)

        patch_mask = outputs.get("patch_mask", None)
        if patch_mask is not None:
            patch_mask = patch_mask.to(device=device).to(torch.bool)
            if patch_mask.shape != (B, K):
                raise ValueError(f"outputs['patch_mask'] must be (B,K)={(B, K)}, got {tuple(patch_mask.shape)}")

        num_boxes = sum(int(t["labels"].numel()) for t in targets)
        num_boxes_t = torch.as_tensor([num_boxes], dtype=torch.float32, device=device)
        if sync_num_boxes and is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_t)
            world_size = get_world_size()
        else:
            world_size = 1
        num_boxes = float(torch.clamp(num_boxes_t / max(1, world_size), min=1.0).item())

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

            if pred_match_cost is None and pred_scores_match is None:
                for_match = {"pred_logits": logits_b.unsqueeze(0), "pred_boxes": boxes_b.unsqueeze(0)}
                src_idx, tgt_idx = self.matcher(
                    for_match, [{"labels": tgt_labels, "boxes": tgt_boxes}], label_map
                )[0]
            else:
                matched_labels_for_cost = tgt_labels.to(device=device)
                local_slots = []
                for label in matched_labels_for_cost.tolist():
                    slot = (support_classes[keep] == int(label)).nonzero(as_tuple=False).flatten()
                    if slot.numel() == 0:
                        local_slots.append(0)
                    else:
                        local_slots.append(int(slot[0].item()))
                slot_idx = torch.as_tensor(local_slots, dtype=torch.long, device=device)
                if pred_match_cost is not None:
                    match_cost_b = torch.nan_to_num(pred_match_cost[b][:, keep], nan=0.0, posinf=1e6, neginf=-1e6)
                    cost_class = match_cost_b[:, slot_idx]
                else:
                    match_scores_b = torch.nan_to_num(pred_scores_match[b][:, keep], nan=0.0, posinf=1e6, neginf=-1e6)
                    cost_class = -match_scores_b[:, slot_idx]
                cost_bbox = torch.cdist(boxes_b, tgt_boxes, p=1)
                cost_giou = -box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(boxes_b),
                    box_ops.box_cxcywh_to_xyxy(tgt_boxes),
                )
                cost = self.matcher.cost_class * cost_class + self.matcher.cost_bbox * cost_bbox + self.matcher.cost_giou * cost_giou
                cost = cost.detach().cpu()
                cost[torch.isnan(cost)] = 0.0
                cost[torch.isinf(cost)] = 0.0
                src_np, tgt_np = linear_sum_assignment(cost)
                src_idx = torch.as_tensor(src_np, dtype=torch.int64, device=device)
                tgt_idx = torch.as_tensor(tgt_np, dtype=torch.int64, device=device)
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

    def _compute_patch_rank_loss(self, match_ctx, targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if float(self.weight_dict.get("loss_patch_rank", 0.0)) <= 0:
            return {}

        pred_logits_patch = match_ctx["pred_logits_patch"]
        patch_mask = match_ctx["patch_mask"]
        all_indices = match_ctx["all_indices"]
        matched_patch_idx_list = match_ctx["matched_patch_idx_list"]
        B = int(match_ctx["B"])
        Q = int(match_ctx["Q"])
        K = int(match_ctx["K"])
        device = pred_logits_patch.device
        zero = pred_logits_patch.sum() * 0.0

        rank_losses: List[torch.Tensor] = []
        violations: List[torch.Tensor] = []
        pos_scores: List[torch.Tensor] = []
        neg_scores: List[torch.Tensor] = []
        pos_total = 0
        cand_total = 0
        margin = self.patch_rank_margin

        for b in range(B):
            src_idx, _tgt_idx = all_indices[b]
            src_idx = src_idx.to(device=device)
            matched_full_k = matched_patch_idx_list[b].to(device=device)
            if src_idx.numel() == 0 or matched_full_k.numel() == 0:
                continue

            support_classes = self._get_support_classes(targets[b], K=K, device=device)
            valid_k = support_classes >= 0
            if patch_mask is not None:
                valid_k = valid_k & patch_mask[b]
            valid_slots = valid_k.nonzero(as_tuple=False).flatten()
            if valid_slots.numel() == 0:
                continue

            for q_t, k_t in zip(src_idx, matched_full_k):
                q = int(q_t.item())
                k = int(k_t.item())
                if q < 0 or q >= Q or k < 0 or k >= K or not bool(valid_k[k].item()):
                    continue

                pos = pred_logits_patch[b, q, k]
                pos_scores.append(pos.detach().reshape(1))
                pos_total += 1

                same_slot_pos_q = src_idx[matched_full_k == k]
                neg_mask = torch.ones((Q,), dtype=torch.bool, device=device)
                if same_slot_pos_q.numel() > 0:
                    neg_mask[same_slot_pos_q] = False
                same_slot_neg = pred_logits_patch[b, :, k][neg_mask]
                if same_slot_neg.numel() > 0 and self.patch_rank_hard_negatives > 0:
                    topk = min(self.patch_rank_hard_negatives, int(same_slot_neg.numel()))
                    hard_neg = same_slot_neg.topk(topk).values
                    raw = hard_neg - pos + margin
                    rank_losses.append(F.relu(raw))
                    violations.append((raw.detach() > 0).to(torch.float32))
                    neg_scores.append(hard_neg.detach())
                    cand_total += int(hard_neg.numel())

                if self.patch_rank_include_wrong_slots and self.patch_rank_wrong_slot_weight > 0:
                    wrong_slots = valid_slots[valid_slots != k]
                    if wrong_slots.numel() > 0 and self.patch_rank_hard_negatives > 0:
                        wrong_neg = pred_logits_patch[b, q, wrong_slots]
                        topk = min(self.patch_rank_hard_negatives, int(wrong_neg.numel()))
                        hard_wrong = wrong_neg.topk(topk).values
                        raw = hard_wrong - pos + margin
                        rank_losses.append(F.relu(raw) * self.patch_rank_wrong_slot_weight)
                        violations.append((raw.detach() > 0).to(torch.float32))
                        neg_scores.append(hard_wrong.detach())
                        cand_total += int(hard_wrong.numel())

        if rank_losses:
            loss_patch_rank = torch.cat([x.reshape(-1) for x in rank_losses]).mean()
            patch_rank_violation_frac = torch.cat([x.reshape(-1) for x in violations]).mean()
        else:
            loss_patch_rank = zero
            patch_rank_violation_frac = torch.zeros((), device=device)

        if pos_scores:
            patch_rank_pos_score = torch.cat(pos_scores).mean()
        else:
            patch_rank_pos_score = torch.zeros((), device=device)
        if neg_scores:
            patch_rank_neg_score = torch.cat([x.reshape(-1) for x in neg_scores]).mean()
        else:
            patch_rank_neg_score = torch.zeros((), device=device)

        return {
            "loss_patch_rank": loss_patch_rank,
            "patch_rank_used_pairs": torch.as_tensor(float(pos_total), device=device),
            "patch_rank_candidate_count": torch.as_tensor(float(cand_total), device=device),
            "patch_rank_violation_frac": patch_rank_violation_frac,
            "patch_rank_pos_score": patch_rank_pos_score,
            "patch_rank_neg_score": patch_rank_neg_score,
            "patch_rank_margin": torch.as_tensor(float(margin), device=device),
        }

    def compute_losses_from_matching(
        self,
        match_ctx,
        targets: List[Dict[str, torch.Tensor]],
        *,
        include_box_losses: bool = True,
    ):
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

        legacy_loss_cls = torch.zeros((), device=device)
        posneg_loss_cls = torch.zeros((), device=device)
        loss_pos_total = torch.zeros((), device=device)
        loss_neg_total = torch.zeros((), device=device)
        loss_neg_all_total = torch.zeros((), device=device)
        neg_topk_count_total = 0
        neg_count_total = 0
        positive_only_batch_total = 0
        matched_total = 0
        cls_batches = 0
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
            pos = target_b > 0.5
            neg = ~pos
            zero = loss_mat.sum() * 0.0
            positive_only_patch_ce = self._target_uses_positive_only_patch_ce(targets[b])
            if positive_only_patch_ce:
                positive_only_batch_total += 1
                legacy_loss_cls = legacy_loss_cls + loss_mat[pos].sum()
            else:
                legacy_loss_cls = legacy_loss_cls + loss_mat.sum()
            pos_loss = loss_mat[pos].mean() if pos.any() else zero
            neg_values = loss_mat[neg] if not positive_only_patch_ce else loss_mat.new_empty((0,))
            if neg_values.numel() > 0:
                neg_all_loss = neg_values.mean()
                neg_loss = neg_all_loss
                neg_count = int(neg_values.numel())
                topk_count = neg_count
                if self.patch_ce_reduction == "posneg_topk":
                    topk_count = self.patch_ce_neg_topk
                    if self.patch_ce_neg_topk_ratio > 0:
                        ratio_count = int(neg_count * self.patch_ce_neg_topk_ratio + 0.999999)
                        topk_count = max(topk_count, ratio_count)
                    if topk_count <= 0:
                        topk_count = neg_count
                    topk_count = min(topk_count, neg_count)
                    neg_loss = neg_values.topk(k=topk_count, largest=True).values.mean()
            else:
                neg_all_loss = zero
                neg_loss = zero
                neg_count = 0
                topk_count = 0
            posneg_loss_cls = posneg_loss_cls + pos_loss + self.patch_lambda_neg * neg_loss
            loss_pos_total = loss_pos_total + pos_loss.detach()
            loss_neg_total = loss_neg_total + neg_loss.detach()
            loss_neg_all_total = loss_neg_all_total + neg_all_loss.detach()
            neg_topk_count_total += int(topk_count)
            neg_count_total += int(neg_count)
            cls_batches += 1

        cls_den = max(int(cls_batches), 1)
        patch_ce_legacy_dense = legacy_loss_cls / num_boxes
        patch_ce_posneg = posneg_loss_cls / float(cls_den)
        if self.patch_ce_reduction == "legacy":
            loss_patch_ce = patch_ce_legacy_dense
        else:
            loss_patch_ce = patch_ce_posneg

        losses: Dict[str, torch.Tensor] = {
            "loss_patch_ce": loss_patch_ce,
            "patch_match_frac": torch.as_tensor(matched_total / float(B * Q), device=device),
            "patch_ce_pos": loss_pos_total / float(cls_den),
            "patch_ce_neg": loss_neg_total / float(cls_den),
            "patch_ce_neg_all": loss_neg_all_total / float(cls_den),
            "patch_ce_neg_topk_count": torch.as_tensor(float(neg_topk_count_total) / float(cls_den), device=device),
            "patch_ce_neg_count": torch.as_tensor(float(neg_count_total) / float(cls_den), device=device),
            "patch_ce_positive_only_batch_frac": torch.as_tensor(
                float(positive_only_batch_total) / float(cls_den), device=device
            ),
            "patch_ce_legacy_dense": patch_ce_legacy_dense.detach(),
            "patch_ce_posneg": patch_ce_posneg.detach(),
            "patch_lambda_neg": torch.as_tensor(self.patch_lambda_neg, device=device),
        }

        if include_box_losses and (
            ("loss_bbox" in self.weight_dict and self.weight_dict["loss_bbox"] > 0)
            or ("loss_giou" in self.weight_dict and self.weight_dict["loss_giou"] > 0)
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
        losses.update(self._compute_patch_rank_loss(match_ctx, targets))
        return losses

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        match_ctx = self.compute_matching(outputs, targets)
        losses = self.compute_losses_from_matching(match_ctx, targets)

        for aux_idx, aux_outputs in enumerate(outputs.get("aux_outputs", []) or []):
            aux_match_ctx = self.compute_matching(aux_outputs, targets)
            aux_losses = self.compute_losses_from_matching(aux_match_ctx, targets)
            aux_suffix = f"_{aux_idx}"
            for key in ("loss_patch_ce", "loss_bbox", "loss_giou", "loss_patch_rank"):
                if key in aux_losses:
                    losses[f"{key}{aux_suffix}"] = aux_losses[key]

        return losses
