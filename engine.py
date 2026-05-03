# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

from util.utils import to_device
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator


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


def _clone_stage_b_drift_batch(samples, targets, captions, patches, patch_global, patch_mask):
    return {
        "samples": utils.NestedTensor(
            samples.tensors.detach().cpu().clone(),
            samples.mask.detach().cpu().clone() if samples.mask is not None else None,
        ),
        "targets": [{k: v.detach().cpu().clone() for k, v in t.items()} for t in targets],
        "captions": list(captions),
        "patches": patches.detach().cpu().clone() if torch.is_tensor(patches) else None,
        "patch_global": patch_global.detach().cpu().clone() if torch.is_tensor(patch_global) else None,
        "patch_mask": patch_mask.detach().cpu().clone() if torch.is_tensor(patch_mask) else None,
    }


def _move_stage_b_drift_batch_to_device(batch, device):
    return {
        "samples": batch["samples"].to(device),
        "targets": [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]],
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
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    drift_state = None


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

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
                outputs = model(
                    samples,
                    targets=targets,
                    captions=captions,
                    patches=patches,
                    patch_global=patch_global,
                    patch_mask=patch_mask,
                    patch_only=True,
                    patch_only_compute_text_logits=bool(getattr(args, "patch_only_compute_text_logits", False)),
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
