# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import copy
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from util.utils import to_device
import numpy as np
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator
from util.misc import NestedTensor


class GracefulTrainingExit(Exception):
    """Raised after an interrupt checkpoint has been written."""


def _make_grad_scaler(enabled: bool):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=enabled)
        except TypeError:
            try:
                return amp_mod.GradScaler(device_type="cuda", enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


class _IteratorWithLen:
    def __init__(self, iterator, length: int) -> None:
        self.iterator = iterator
        self.length = max(0, int(length))

    def __iter__(self):
        return self.iterator

    def __len__(self) -> int:
        return self.length


def _index_nested_tensor(samples: NestedTensor, indices: List[int]) -> NestedTensor:
    idx = torch.as_tensor(indices, dtype=torch.long, device=samples.tensors.device)
    tensors = samples.tensors.index_select(0, idx)
    mask = samples.mask.index_select(0, idx) if samples.mask is not None else None
    return NestedTensor(tensors, mask)


def _select_rank_patch_rows(value, indices: List[int], slots: List[int]):
    if value is None:
        return None
    batch_idx = torch.as_tensor(indices, dtype=torch.long, device=value.device)
    slot_idx = torch.as_tensor(slots, dtype=torch.long, device=value.device)
    if value.dim() == 2:
        return value.index_select(0, batch_idx)
    if value.dim() == 3:
        return value[batch_idx, slot_idx].unsqueeze(1)
    if value.dim() == 4:
        return value.index_select(0, batch_idx)
    if value.dim() == 5:
        return value[batch_idx, slot_idx].unsqueeze(1)
    raise ValueError(f"Unsupported rank patch tensor shape: {tuple(value.shape)}")


def _build_stage_b_rank_subbatch(args, samples, targets, captions, patches, patch_global, patch_mask):
    if not bool(getattr(args, "stage_b_enable_phrase_rank", True)):
        return None
    device = samples.tensors.device
    rank_indices: List[int] = []
    rank_slots: List[int] = []
    rank_captions: List[str] = []
    rank_targets: List[dict] = []
    rank_candidate_tn_count = 0
    rank_missing_positive_count = 0
    rank_invalid_positive_count = 0
    for batch_idx, target in enumerate(targets):
        is_tn = target.get("is_tn", None)
        if torch.is_tensor(is_tn):
            tn_slots = torch.nonzero(is_tn.to(torch.bool).view(-1), as_tuple=False).flatten()
        else:
            tn_slots = torch.zeros((0,), dtype=torch.long)
        if int(tn_slots.numel()) == 0:
            continue
        rank_candidate_tn_count += 1
        has_rank = target.get("has_rank_positive", None)
        rank_captions_i = target.get("rank_positive_captions", None)
        if (not torch.is_tensor(has_rank)) or rank_captions_i is None:
            rank_missing_positive_count += 1
            continue
        valid_slots = torch.nonzero(has_rank.to(torch.bool).view(-1) & is_tn.to(torch.bool).view(-1), as_tuple=False).flatten()
        if int(valid_slots.numel()) == 0:
            rank_missing_positive_count += 1
            continue
        if int(valid_slots.numel()) != 1:
            rank_invalid_positive_count += 1
            continue
        slot_idx = int(valid_slots[0].item())
        if not isinstance(rank_captions_i, list) or slot_idx >= len(rank_captions_i):
            rank_invalid_positive_count += 1
            continue
        caption = rank_captions_i[slot_idx]
        if not isinstance(caption, str) or not caption.strip():
            rank_missing_positive_count += 1
            continue
        phrase_mask = target.get("rank_positive_phrase_to_token_mask", None)
        canonical_mask = target.get("rank_positive_canonical_to_token_mask", None)
        if (not torch.is_tensor(phrase_mask)) or (not torch.is_tensor(canonical_mask)):
            rank_invalid_positive_count += 1
            continue
        if phrase_mask.dim() != 2 or canonical_mask.dim() != 2 or slot_idx >= int(phrase_mask.shape[0]):
            rank_invalid_positive_count += 1
            continue
        if not bool(phrase_mask[slot_idx].any().item()) or not bool(canonical_mask[slot_idx].any().item()):
            rank_invalid_positive_count += 1
            continue
        rank_indices.append(batch_idx)
        rank_slots.append(slot_idx)
        rank_captions.append(caption)
        rank_target = {
            k: v.to(device, non_blocking=True)
            for k, v in target.items()
            if torch.is_tensor(v)
            and k
            not in {
                "phrase_to_token_mask",
                "canonical_to_token_mask",
                "content_to_token_mask",
                "attr_pos_to_token_mask",
                "attr_neg_to_token_mask",
                "phrase_semantic_token_mask",
                "negative_to_token_mask",
                "attr_neg_weight_mask",
                "rank_positive_phrase_to_token_mask",
                "rank_positive_canonical_to_token_mask",
                "has_rank_positive",
            }
        }
        rank_target["phrase_to_token_mask"] = phrase_mask[slot_idx : slot_idx + 1].to(device, non_blocking=True)
        rank_target["canonical_to_token_mask"] = canonical_mask[slot_idx : slot_idx + 1].to(device, non_blocking=True)
        selected_class = None
        if "support_classes" in rank_target and torch.is_tensor(rank_target["support_classes"]):
            sc = rank_target["support_classes"].view(-1)
            if slot_idx < int(sc.numel()):
                selected_class = int(sc[slot_idx].item())
                rank_target["support_classes"] = sc[slot_idx : slot_idx + 1]
                rank_target["support_class"] = sc[slot_idx : slot_idx + 1]
        elif "support_class" in rank_target and torch.is_tensor(rank_target["support_class"]):
            selected_class = int(rank_target["support_class"].view(-1)[0].item())
            rank_target["support_class"] = rank_target["support_class"].view(-1)[:1]
        if selected_class is not None and "labels" in rank_target and "boxes" in rank_target:
            label_mask = rank_target["labels"].to(torch.long) == int(selected_class)
            if not bool(label_mask.any().item()):
                rank_indices.pop()
                rank_slots.pop()
                rank_captions.pop()
                rank_invalid_positive_count += 1
                continue
            rank_target["rank_target_ids"] = torch.nonzero(label_mask, as_tuple=False).flatten().to(device)
            rank_target["labels"] = rank_target["labels"][label_mask]
            rank_target["boxes"] = rank_target["boxes"][label_mask]
        rank_target["rank_source_slot"] = torch.as_tensor([slot_idx], dtype=torch.long, device=device)
        rank_targets.append(rank_target)
    if rank_candidate_tn_count <= 0:
        return None
    if not rank_indices:
        return {
            "indices": [],
            "rank_candidate_tn_count": rank_candidate_tn_count,
            "rank_missing_positive_count": rank_missing_positive_count,
            "rank_invalid_positive_count": rank_invalid_positive_count,
        }
    rank_patch_mask = None
    if patch_mask is not None:
        rank_patch_mask = torch.ones((len(rank_indices), 1), dtype=torch.bool, device=patch_mask.device)
    return {
        "indices": rank_indices,
        "rank_candidate_tn_count": rank_candidate_tn_count,
        "rank_missing_positive_count": rank_missing_positive_count,
        "rank_invalid_positive_count": rank_invalid_positive_count,
        "samples": _index_nested_tensor(samples, rank_indices),
        "captions": rank_captions,
        "targets": rank_targets,
        "patches": _select_rank_patch_rows(patches, rank_indices, rank_slots),
        "patch_global": _select_rank_patch_rows(patch_global, rank_indices, rank_slots),
        "patch_mask": rank_patch_mask,
    }


def _restore_rng_state(rng_state) -> None:
    if not rng_state:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda", None) is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def _unnormalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    img: (3,H,W) normalized by ImageNet mean/std.
    Returns float tensor in [0,1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device)[:, None, None]
    x = img * std + mean
    return x.clamp(0, 1)


def _cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """
    boxes: (...,4) normalized cxcywh in [0,1]
    returns (...,4) absolute xyxy in pixels.
    """
    cx, cy, bw, bh = boxes.unbind(-1)
    x0 = (cx - 0.5 * bw) * w
    y0 = (cy - 0.5 * bh) * h
    x1 = (cx + 0.5 * bw) * w
    y1 = (cy + 0.5 * bh) * h
    out = torch.stack([x0, y0, x1, y1], dim=-1)
    out[..., 0::2] = out[..., 0::2].clamp(0, w - 1)
    out[..., 1::2] = out[..., 1::2].clamp(0, h - 1)
    return out


@torch.no_grad()
def _maybe_save_patch_sanity(
    *,
    args,
    samples,
    targets,
    outputs,
    criterion=None,
    epoch: int,
    step: int,
) -> None:
    if not bool(getattr(args, "log_patch_sanity", False)):
        return
    if not utils.is_main_process():
        return
    out_dir = getattr(args, "output_dir", None)
    if not out_dir:
        return

    interval = int(getattr(args, "patch_sanity_interval", 500))
    if interval <= 0 or (step % interval) != 0:
        return

    try:
        from PIL import Image, ImageDraw, ImageFont  # pylint: disable=import-error
    except Exception:
        return

    topk = int(getattr(args, "patch_sanity_topk", 20))
    max_images = int(getattr(args, "patch_sanity_max_images", 2))
    topk = max(1, topk)
    max_images = max(1, max_images)

    if "pred_logits_patch" not in outputs or outputs["pred_logits_patch"] is None:
        return
    if "pred_boxes" not in outputs:
        return

    img_t = samples.tensors.detach().float().cpu()  # (B,3,H,W)
    mask = getattr(samples, "mask", None)
    if mask is not None:
        mask = mask.detach().cpu()  # (B,H,W) True for padded

    pred_logits_raw = outputs["pred_logits_patch"].detach().float().cpu()  # (B,Q) or (B,Q,K)
    # Union score for ranking/quick view.
    pred_logits_union = pred_logits_raw
    if pred_logits_union.dim() == 3:
        pred_logits_union = pred_logits_union.max(dim=-1).values  # (B,Q)
    pred_boxes = outputs["pred_boxes"].detach().float().cpu()  # (B,Q,4) cxcywh norm
    patch_mask = outputs.get("patch_mask", None)
    if patch_mask is not None:
        patch_mask = patch_mask.detach().cpu().to(torch.bool)

    save_dir = Path(out_dir) / "patch_sanity"
    save_dir.mkdir(parents=True, exist_ok=True)

    dn_num = int(getattr(args, "patch_dn_num_queries", 0))
    B = int(img_t.shape[0])
    for b in range(min(B, max_images)):
        x = _unnormalize_img(img_t[b])
        if mask is not None:
            m = mask[b]
            x[:, m] = 0.0
        c, h, w = x.shape
        _ = c
        img_u8 = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        pil = Image.fromarray(img_u8)
        draw = ImageDraw.Draw(pil)

        logits_union = pred_logits_union[b]  # (Q,)
        boxes_all = pred_boxes[b]
        Q = int(logits_union.numel())
        dn = max(0, min(int(dn_num), Q))
        is_neg = int(targets[b].get("is_negative_episode", torch.tensor([0])).item())

        def _load_font(size: int, bold: bool = False):
            try:
                candidates = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                ]
                for p in candidates:
                    try:
                        return ImageFont.truetype(p, size=size)
                    except Exception:
                        continue
            except Exception:
                pass
            return ImageFont.load_default()

        font_small = _load_font(14, bold=False)
        font_big = _load_font(22, bold=True)

        def _draw_text_bold(xy, text: str, fill, stroke: int = 1, font=None):
            x0, y0 = xy
            for dx in range(-stroke, stroke + 1):
                for dy in range(-stroke, stroke + 1):
                    draw.text((x0 + dx, y0 + dy), text, fill=fill, font=font)

        # 1) Draw ALL GT boxes (green).
        gt_boxes = targets[b].get("boxes", None)
        gt_labels = targets[b].get("labels", None)
        if gt_boxes is not None and gt_labels is not None:
            gt_boxes = gt_boxes.detach().float().cpu()
            gt_labels = gt_labels.detach().long().cpu()
            gt_xyxy = _cxcywh_norm_to_xyxy_abs(gt_boxes, w=w, h=h).tolist()
            for (x0, y0, x1, y1) in gt_xyxy:
                draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=4)

        # 2) Draw top-k query boxes in red (union score).
        sc = logits_union.sigmoid()
        k = min(topk, int(sc.numel()))
        if k > 0:
            vals, idx = torch.topk(sc, k=k, largest=True)
            q_boxes = boxes_all[idx]
            q_xyxy = _cxcywh_norm_to_xyxy_abs(q_boxes, w=w, h=h).tolist()
            for (x0, y0, x1, y1), v in zip(q_xyxy, vals.tolist()):
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                _draw_text_bold((x0 + 2, max(0, y0 - 14)), f"{v:.2f}", fill=(255, 0, 0), stroke=1, font=font_small)

        # 3) Draw Hungarian-matched query boxes in yellow, and annotate (canonical id, logit).
        if criterion is not None and hasattr(criterion, "matcher") and criterion.matcher is not None:
            # Prepare logits as (Q,K) for matching.
            if pred_logits_raw.dim() == 2:
                logits_qk = pred_logits_raw[b].unsqueeze(-1)  # (Q,1)
                support_classes = targets[b].get("support_class", None)
                if support_classes is not None:
                    support_classes = support_classes.detach().long().cpu().view(-1)
                else:
                    support_classes = torch.full((1,), -1, dtype=torch.long)
            else:
                logits_qk = pred_logits_raw[b]  # (Q,K)
                support_classes = targets[b].get("support_classes", None)
                if support_classes is not None:
                    support_classes = support_classes.detach().long().cpu().view(-1)
                else:
                    support_classes = torch.full((logits_qk.shape[1],), -1, dtype=torch.long)

            K = int(logits_qk.shape[1])
            if support_classes.numel() < K:
                pad = torch.full((K - int(support_classes.numel()),), -1, dtype=support_classes.dtype)
                support_classes = torch.cat([support_classes, pad], dim=0)
            support_classes = support_classes[:K]
            valid_k = support_classes >= 0
            if patch_mask is not None:
                valid_k = valid_k & patch_mask[b].view(-1)[:K]

            if gt_boxes is not None and gt_labels is not None and int(valid_k.sum().item()) > 0:
                keep = valid_k.nonzero(as_tuple=False).flatten()
                logits_b = logits_qk[:, keep]
                support_kept = support_classes[keep].to(torch.long)

                max_row = int(max(int(gt_labels.max().item()), int(support_kept.max().item()))) + 1
                label_map = torch.zeros((max_row, int(keep.numel())), dtype=torch.float32)
                for local_k, cid in enumerate(support_kept.tolist()):
                    if cid >= 0 and cid < max_row:
                        label_map[int(cid), int(local_k)] = 1.0

                try:
                    (src_idx, tgt_idx) = criterion.matcher(
                        {"pred_logits": logits_b.unsqueeze(0), "pred_boxes": boxes_all.unsqueeze(0)},
                        [{"labels": gt_labels, "boxes": gt_boxes}],
                        label_map,
                    )[0]
                except Exception:
                    src_idx = torch.zeros((0,), dtype=torch.int64)
                    tgt_idx = torch.zeros((0,), dtype=torch.int64)

                if src_idx.numel() > 0:
                    cid_to_local = {int(cid): int(i) for i, cid in enumerate(support_kept.tolist())}
                    pred_xyxy = _cxcywh_norm_to_xyxy_abs(boxes_all[src_idx], w=w, h=h).tolist()
                    for m_i, ((x0, y0, x1, y1), gt_i) in enumerate(zip(pred_xyxy, tgt_idx.tolist())):
                        cid = int(gt_labels[gt_i].item())
                        lk = cid_to_local.get(cid, None)
                        if lk is None:
                            continue
                        logit = float(logits_b[int(src_idx[m_i].item()), int(lk)].item())
                        score = 1.0 / (1.0 + torch.exp(torch.tensor(-logit))).item()
                        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)
                        _draw_text_bold((x0 + 2, max(0, y0 - 24)), f"{cid}", fill=(255, 255, 0), stroke=2, font=font_big)

        draw.text(
            (5, 5),
            f"epoch={epoch} step={step} neg={is_neg} dn={dn}",
            fill=(255, 255, 0),
        )
        pil.save(save_dir / f"e{epoch:03d}_s{step:06d}_b{b}.jpg", quality=90)


def _clone_target_value_for_drift(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return copy.deepcopy(value)


def _move_target_value_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _clone_stage_b_drift_batch(samples, targets, captions, patches, patch_global, patch_mask):
    return {
        "samples": utils.NestedTensor(
            samples.tensors.detach().cpu().clone(),
            samples.mask.detach().cpu().clone() if samples.mask is not None else None,
        ),
        "targets": [
            {k: _clone_target_value_for_drift(v) for k, v in t.items()}
            for t in targets
        ],
        "captions": list(captions),
        "patches": patches.detach().cpu().clone() if torch.is_tensor(patches) else None,
        "patch_global": patch_global.detach().cpu().clone() if torch.is_tensor(patch_global) else None,
        "patch_mask": patch_mask.detach().cpu().clone() if torch.is_tensor(patch_mask) else None,
    }


def _move_stage_b_drift_batch_to_device(batch, device):
    return {
        "samples": batch["samples"].to(device),
        "targets": [
            {k: _move_target_value_to_device(v, device) for k, v in t.items()}
            for t in batch["targets"]
        ],
        "captions": list(batch["captions"]),
        "patches": batch["patches"].to(device) if torch.is_tensor(batch["patches"]) else None,
        "patch_global": batch["patch_global"].to(device) if torch.is_tensor(batch["patch_global"]) else None,
        "patch_mask": batch["patch_mask"].to(device) if torch.is_tensor(batch["patch_mask"]) else None,
    }


@torch.no_grad()
def _eval_stage_b_drift_batch(args, model, device, batch):
    cached = _move_stage_b_drift_batch_to_device(batch, device)
    was_training = model.training
    model.eval()
    try:
        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", False)):
            eval_outputs = model(
                cached["samples"],
                targets=cached["targets"],
                captions=cached["captions"],
                patches=cached["patches"],
                patch_global=cached["patch_global"],
                patch_mask=cached["patch_mask"],
                patch_only=True,
                patch_only_compute_text_logits=bool(getattr(args, "patch_only_compute_text_logits", False)),
            )
    finally:
        if was_training:
            model.train()
    return cached, eval_outputs


@torch.no_grad()
def _compute_stage_b_patch_metrics(outputs, targets, criterion, topk: int):
    pred_logits_patch = outputs.get("pred_logits_patch", None)
    if pred_logits_patch is None:
        return None
    pred_logits_patch = pred_logits_patch.detach().float()
    union_logits = pred_logits_patch if pred_logits_patch.dim() == 2 else pred_logits_patch.max(dim=-1).values

    metrics = {
        "patch_logit_mean": float(union_logits.mean().item()),
        "patch_logit_std": float(union_logits.std(unbiased=False).item()),
        "patch_match_topk_recall": 0.0,
    }
    if not hasattr(criterion, "compute_matching"):
        return metrics

    try:
        match_ctx = criterion.compute_matching(outputs, targets)
    except Exception:
        return metrics

    matched = 0.0
    recalled = 0.0
    k = max(1, int(topk))
    for b, (src_idx, _tgt_idx) in enumerate(match_ctx["all_indices"]):
        if src_idx.numel() == 0:
            continue
        topk_idx = torch.topk(union_logits[b], k=min(k, int(union_logits.shape[1])), largest=True).indices
        recalled += float(torch.isin(src_idx.detach().cpu(), topk_idx.detach().cpu()).float().sum().item())
        matched += float(src_idx.numel())
    if matched > 0:
        metrics["patch_match_topk_recall"] = recalled / matched
    return metrics


@torch.no_grad()
def _maybe_log_stage_b_patch_drift(
    *,
    args,
    model,
    criterion,
    device,
    drift_state,
    samples,
    targets,
    captions,
    patches,
    patch_global,
    patch_mask,
    outputs,
    step: int,
    logger=None,
):
    if not bool(getattr(args, "log_stage_b_patch_drift", False)):
        return drift_state
    if not utils.is_main_process():
        return drift_state

    topk = int(getattr(args, "stage_b_patch_drift_topk", 50))
    interval = int(getattr(args, "stage_b_patch_drift_interval", 100))

    if drift_state is None:
        drift_batch = _clone_stage_b_drift_batch(samples, targets, captions, patches, patch_global, patch_mask)
        cached, eval_outputs = _eval_stage_b_drift_batch(args, model, device, drift_batch)
        baseline_metrics = _compute_stage_b_patch_metrics(eval_outputs, cached["targets"], criterion, topk=topk)
        if baseline_metrics is None:
            return None
        drift_state = {
            "baseline_step": int(step),
            "baseline_metrics": baseline_metrics,
            "batch": drift_batch,
        }
        msg = f"Stage B patch drift baseline @step={step}: {baseline_metrics}"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
        return drift_state

    if interval <= 0 or step <= 0 or (step % interval) != 0:
        return drift_state

    cached, eval_outputs = _eval_stage_b_drift_batch(args, model, device, drift_state["batch"])
    cur_metrics = _compute_stage_b_patch_metrics(eval_outputs, cached["targets"], criterion, topk=topk)
    if cur_metrics is None:
        return drift_state
    baseline_metrics = drift_state["baseline_metrics"]
    delta = {k: float(cur_metrics[k] - baseline_metrics[k]) for k in baseline_metrics.keys()}
    msg = (
        f"Stage B patch drift @step={step}: current={cur_metrics} "
        f"baseline={baseline_metrics} delta={delta}"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return drift_state


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    start_iter: int = 0,
                    epoch_rng_state=None,
                    runtime_rng_state=None,
                    iter_checkpoint_fn: Optional[Callable[..., None]] = None):
    if scaler is None:
        scaler = _make_grad_scaler(enabled=args.amp)


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = max(0, int(start_iter))
    drift_state = None

    data_iterable = data_loader
    if _cnt > 0:
        if epoch_rng_state is not None:
            _restore_rng_state(epoch_rng_state)
        raw_iter = iter(data_loader)
        skipped = 0
        total_len = len(data_loader) if hasattr(data_loader, "__len__") else _cnt
        target_skip = min(_cnt, int(total_len))
        def log_skip(message: str) -> None:
            if logger is not None:
                logger.info(message)
            else:
                print(message, flush=True)

        log_every_batches = 100
        log_every_seconds = 30.0
        last_log_t = time.time()
        skip_start_t = last_log_t
        log_skip(
            f"Resuming epoch {epoch}: skipping {target_skip}/{total_len} already-finished batches. "
            "This is CPU/data-loader work, so GPU usage can stay low until the skip finishes."
        )
        for _ in range(target_skip):
            try:
                next(raw_iter)
                skipped += 1
                now_t = time.time()
                if (
                    skipped <= 10
                    or skipped % log_every_batches == 0
                    or skipped == target_skip
                    or (now_t - last_log_t) >= log_every_seconds
                ):
                    elapsed = max(1e-6, now_t - skip_start_t)
                    batches_per_sec = skipped / elapsed
                    remaining = max(0, target_skip - skipped)
                    eta = remaining / max(1e-6, batches_per_sec)
                    log_skip(
                        f"Resume skip progress: {skipped}/{target_skip} batches "
                        f"({batches_per_sec:.2f} batch/s, eta {int(eta)}s)."
                    )
                    last_log_t = now_t
            except StopIteration:
                break
        msg = f"Resuming epoch {epoch} from iteration {skipped}/{total_len}; skipped completed batches."
        log_skip(msg)
        if runtime_rng_state is not None:
            _restore_rng_state(runtime_rng_state)
        data_iterable = _IteratorWithLen(raw_iter, int(total_len) - skipped)
        _cnt = skipped

    for samples, targets in metric_logger.log_every(data_iterable, print_freq, header, logger=logger):

        samples = samples.to(device)
        patch_only = bool(getattr(args, "patch_only", False))
        if patch_only:
            captions = [t.get("caption", "object .") for t in targets]
            patch_mask = None
            patch_global = None
            patches = None
            if all(("patch_global" in t) for t in targets):
                pg0 = targets[0]["patch_global"]
                if (not torch.is_tensor(pg0)) or pg0.dim() not in {1, 2}:
                    raise ValueError("targets[*]['patch_global'] must be (D,) or (K,D) in patch_only mode.")
                if pg0.dim() == 1:
                    patch_global = torch.stack([t["patch_global"] for t in targets], dim=0).to(device, non_blocking=True)
                else:
                    # Pad variable K across batch: (B,Kmax,D) + patch_mask.
                    Kmax = max(int(t["patch_global"].shape[0]) for t in targets)
                    D = int(pg0.shape[1])
                    patch_global = torch.zeros((len(targets), Kmax, D), dtype=pg0.dtype, device=device)
                    patch_mask = torch.zeros((len(targets), Kmax), dtype=torch.bool, device=device)
                    for i, t in enumerate(targets):
                        pg = t["patch_global"]
                        ki = int(pg.shape[0])
                        patch_global[i, :ki] = pg.to(device, non_blocking=True)
                        patch_mask[i, :ki] = True
            elif all(("patches" in t) for t in targets):
                p0 = targets[0]["patches"]
                if (not torch.is_tensor(p0)) or p0.dim() != 4:
                    raise ValueError("targets[*]['patches'] must be (K,3,H,W) in patch_only mode.")
                Kmax = max(int(t["patches"].shape[0]) for t in targets)
                C, H, W = map(int, p0.shape[1:])
                patches = torch.zeros((len(targets), Kmax, C, H, W), dtype=p0.dtype, device=device)
                patch_mask = torch.zeros((len(targets), Kmax), dtype=torch.bool, device=device)
                for i, t in enumerate(targets):
                    p = t["patches"]
                    ki = int(p.shape[0])
                    patches[i, :ki] = p.to(device, non_blocking=True)
                    patch_mask[i, :ki] = True
            else:
                patches = torch.stack([t["patch"] for t in targets], dim=0).to(device, non_blocking=True)
            filtered_targets = []
            Kmax = int(patch_mask.shape[1]) if (patch_mask is not None and torch.is_tensor(patch_mask)) else None
            for t in targets:
                t2 = {k: v.to(device) for k, v in t.items() if torch.is_tensor(v) and k not in {"patch", "patches", "patch_global"}}
                if Kmax is not None and "support_classes" in t2 and torch.is_tensor(t2["support_classes"]):
                    sc = t2["support_classes"].view(-1)
                    if sc.numel() < Kmax:
                        pad = torch.full((Kmax - int(sc.numel()),), -1, dtype=sc.dtype, device=sc.device)
                        t2["support_classes"] = torch.cat([sc, pad], dim=0)
                filtered_targets.append(t2)
                if "rank_positive_captions" in t:
                    t2["rank_positive_captions"] = t["rank_positive_captions"]
            targets = filtered_targets
        else:
            captions = [t["caption"] for t in targets]
            cap_list = [t["cap_list"] for t in targets]
            patches = None
            if all(("patch" in t) for t in targets):
                patches = torch.stack([t["patch"] for t in targets], dim=0).to(device, non_blocking=True)
            patch_global = None
            if all(("patch_global" in t) for t in targets):
                patch_global = torch.stack([t["patch_global"] for t in targets], dim=0).to(device, non_blocking=True)
            targets = [
                {k: v.to(device) for k, v in t.items() if torch.is_tensor(v) and k not in {"patch", "patch_global"}}
                for t in targets
            ]
        with torch.cuda.amp.autocast(enabled=args.amp):
            if patch_only:
                # Pass `targets` so the model can optionally build GT-guided (DN) queries in patch-only mode.
                stage_b_mask_kwargs = {}
                for mask_key in (
                    "canonical_to_token_mask",
                    "content_to_token_mask",
                    "attr_pos_to_token_mask",
                    "attr_neg_to_token_mask",
                    "phrase_semantic_token_mask",
                ):
                    if all(mask_key in t for t in targets):
                        values = [t[mask_key] for t in targets]
                        if all(torch.is_tensor(v) for v in values):
                            if len({tuple(v.shape) for v in values}) == 1:
                                stage_b_mask_kwargs[mask_key] = torch.stack(values, dim=0).to(
                                    device, non_blocking=True
                                )
                            elif all(v.dim() == 2 for v in values):
                                kmax = max(int(v.shape[0]) for v in values)
                                tmax = max(int(v.shape[1]) for v in values)
                                padded = values[0].new_zeros((len(values), kmax, tmax))
                                for i, v in enumerate(values):
                                    padded[i, : int(v.shape[0]), : int(v.shape[1])] = v
                                stage_b_mask_kwargs[mask_key] = padded.to(device, non_blocking=True)
                rank_subbatch = _build_stage_b_rank_subbatch(
                    args,
                    samples,
                    targets,
                    captions,
                    patches,
                    patch_global,
                    patch_mask,
                )
                has_rank_pairs = bool(rank_subbatch is not None and rank_subbatch.get("indices"))
                outputs = model(
                    samples,
                    targets=targets,
                    captions=captions,
                    patches=patches,
                    patch_global=patch_global,
                    patch_mask=patch_mask,
                    patch_only=True,
                    disable_patch_dn=has_rank_pairs,
                    patch_only_compute_text_logits=bool(getattr(args, "patch_only_compute_text_logits", False)),
                    **stage_b_mask_kwargs,
                )
                if rank_subbatch is not None:
                    outputs["rank_candidate_tn_count"] = torch.as_tensor(
                        float(rank_subbatch.get("rank_candidate_tn_count", 0)), device=device
                    )
                    outputs["rank_missing_positive_count"] = torch.as_tensor(
                        float(rank_subbatch.get("rank_missing_positive_count", 0)), device=device
                    )
                    outputs["rank_invalid_positive_count"] = torch.as_tensor(
                        float(rank_subbatch.get("rank_invalid_positive_count", 0)), device=device
                    )
                    if rank_subbatch["indices"]:
                        rank_pos_outputs = model(
                            rank_subbatch["samples"],
                            targets=rank_subbatch["targets"],
                            captions=rank_subbatch["captions"],
                            patches=rank_subbatch["patches"],
                            patch_global=rank_subbatch["patch_global"],
                            patch_mask=rank_subbatch["patch_mask"],
                            patch_only=True,
                            disable_patch_dn=True,
                            patch_only_compute_text_logits=bool(getattr(args, "patch_only_compute_text_logits", False)),
                            canonical_to_token_mask=torch.stack(
                                [t["canonical_to_token_mask"] for t in rank_subbatch["targets"]], dim=0
                            ),
                        )
                        outputs["rank_pos_outputs"] = rank_pos_outputs
                        outputs["rank_pos_targets"] = rank_subbatch["targets"]
                        outputs["rank_pair_map"] = torch.as_tensor(
                            rank_subbatch["indices"], dtype=torch.long, device=device
                        )
                loss_dict = criterion(outputs, targets)
            else:
                outputs = model(samples, captions=captions, patches=patches, patch_global=patch_global)
                loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            if patch_only:
                _maybe_save_patch_sanity(
                    args=args, samples=samples, targets=targets, outputs=outputs, criterion=criterion, epoch=epoch, step=_cnt
                )
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()

        if patch_only and bool(getattr(args, "stage_b", False)):
            drift_state = _maybe_log_stage_b_patch_drift(
                args=args,
                model=model,
                criterion=criterion,
                device=device,
                drift_state=drift_state,
                samples=samples,
                targets=targets,
                captions=captions,
                patches=patches,
                patch_global=patch_global,
                patch_mask=patch_mask,
                outputs=outputs,
                step=_cnt + 1,
                logger=logger,
            )


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        iter_interval = int(getattr(args, "iter_checkpoint_interval", 0) or 0)
        stop_requested = bool(getattr(args, "_stop_requested", False))
        should_save_iter = iter_checkpoint_fn is not None and (
            stop_requested or (iter_interval > 0 and (_cnt % iter_interval == 0))
        )
        if should_save_iter:
            reason = "signal" if stop_requested else "interval"
            iter_checkpoint_fn(
                epoch=epoch,
                iteration=_cnt,
                scaler=scaler,
                epoch_finished=False,
                reason=reason,
            )
        if stop_requested:
            signum = getattr(args, "_stop_signal", None)
            raise GracefulTrainingExit(f"Stop requested by signal {signum}; saved iteration checkpoint.")
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        # 获取所有类别
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list=args.label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs = model(samples, captions=input_captions)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator
