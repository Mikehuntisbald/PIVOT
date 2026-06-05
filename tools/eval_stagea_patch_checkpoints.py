#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from groundingdino.util import box_ops  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _load_model_and_criterion(cfg, ckpt_path: str, device: torch.device):
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={cfg.modelname}")
    model, criterion, _postprocessors = build_func(cfg)
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    state = clean_state_dict(_extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] {ckpt_path}: missing keys={len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] {ckpt_path}: unexpected keys={len(unexpected)}", file=sys.stderr)
    model.to(device).eval()
    criterion.to(device).eval()
    return model, criterion


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _prepare_patch_batch(samples, targets, device: torch.device):
    samples = samples.to(device)
    targets = list(targets)
    captions = [t.get("caption", "object .") for t in targets]
    patch_mask = None
    patch_global = None
    patches = None
    kmax = None

    if all(("patch_global" in t) for t in targets):
        pg0 = targets[0]["patch_global"]
        if pg0.dim() == 1:
            patch_global = torch.stack([t["patch_global"] for t in targets], dim=0).to(device, non_blocking=True)
        else:
            kmax = max(int(t["patch_global"].shape[0]) for t in targets)
            dim = int(pg0.shape[1])
            patch_global = torch.zeros((len(targets), kmax, dim), dtype=pg0.dtype, device=device)
            patch_mask = torch.zeros((len(targets), kmax), dtype=torch.bool, device=device)
            for i, t in enumerate(targets):
                pg = t["patch_global"]
                k = int(pg.shape[0])
                patch_global[i, :k] = pg.to(device, non_blocking=True)
                patch_mask[i, :k] = True
    elif all(("patches" in t) for t in targets):
        p0 = targets[0]["patches"]
        kmax = max(int(t["patches"].shape[0]) for t in targets)
        c, h, w = map(int, p0.shape[1:])
        patches = torch.zeros((len(targets), kmax, c, h, w), dtype=p0.dtype, device=device)
        patch_mask = torch.zeros((len(targets), kmax), dtype=torch.bool, device=device)
        for i, t in enumerate(targets):
            p = t["patches"]
            k = int(p.shape[0])
            patches[i, :k] = p.to(device, non_blocking=True)
            patch_mask[i, :k] = True
    else:
        patches = torch.stack([t["patch"] for t in targets], dim=0).to(device, non_blocking=True)

    filtered_targets = []
    for t in targets:
        out = {
            k: v.to(device)
            for k, v in t.items()
            if torch.is_tensor(v) and k not in {"patch", "patches", "patch_global"}
        }
        if kmax is not None and "support_classes" in out:
            support_classes = out["support_classes"].view(-1)
            if support_classes.numel() < kmax:
                pad = torch.full(
                    (kmax - int(support_classes.numel()),),
                    -1,
                    dtype=support_classes.dtype,
                    device=device,
                )
                out["support_classes"] = torch.cat([support_classes, pad], dim=0)
        filtered_targets.append(out)

    return samples, filtered_targets, captions, patches, patch_global, patch_mask


@torch.no_grad()
def _forward_patch(model, batch, device: torch.device, *, amp: bool):
    samples, targets, captions, patches, patch_global, patch_mask = _prepare_patch_batch(*batch, device)
    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        outputs = model(
            samples,
            targets=targets,
            captions=captions,
            patches=patches,
            patch_global=patch_global,
            patch_mask=patch_mask,
            patch_only=True,
            patch_only_compute_text_logits=False,
        )
    return outputs, targets


def _as_patch_logits(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs["pred_logits_patch"].detach().float()
    if logits.dim() == 2:
        logits = logits.unsqueeze(-1)
    if logits.dim() != 3:
        raise ValueError(f"pred_logits_patch must be (B,Q) or (B,Q,K), got {tuple(logits.shape)}")
    return logits


def _support_classes_from_target(target: Dict[str, torch.Tensor], *, k: int, device: torch.device) -> torch.Tensor:
    support_classes = target.get("support_classes", None)
    if support_classes is None:
        support_class = target.get("support_class", None)
        if support_class is None:
            raise KeyError("target must contain support_classes or support_class")
        support_classes = support_class.view(1).repeat(k)
    support_classes = support_classes.to(device=device).view(-1).to(torch.long)
    if support_classes.numel() < k:
        pad = torch.full((k - int(support_classes.numel()),), -1, dtype=support_classes.dtype, device=device)
        support_classes = torch.cat([support_classes, pad], dim=0)
    return support_classes[:k]


def _to_float(value: float) -> float:
    if math.isnan(float(value)) or math.isinf(float(value)):
        return 0.0
    return float(value)


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


class PatchEvalAccumulator:
    def __init__(
        self,
        topks: Iterable[int],
        *,
        ap_max_dets_per_image: int,
        ap_score_agg: str = "slot",
    ) -> None:
        self.topks = sorted({max(1, int(k)) for k in topks})
        self.ap_max_dets_per_image = max(0, int(ap_max_dets_per_image))
        self.ap_score_agg = str(ap_score_agg).lower().strip()
        if self.ap_score_agg not in {"slot", "class_max"}:
            raise ValueError(f"Unsupported ap_score_agg={ap_score_agg!r}; expected slot or class_max.")
        self.num_images = 0
        self.num_targets = 0
        self.unsupported_targets = 0
        self.box_targets = 0
        self.matched_total = 0
        self.matched_recalled = {k: 0 for k in self.topks}
        self.box_recalled = {k: 0 for k in self.topks}
        self.best_iou_sum = {k: 0.0 for k in self.topks}
        self.match_errors = 0
        self.detections: List[Tuple[float, int, int, float, float, float, float]] = []
        self.gt_by_key: Dict[Tuple[int, int], List[List[float]]] = {}
        self.ap_targets = 0

    def update(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], criterion) -> None:
        logits = _as_patch_logits(outputs)
        pred_boxes = outputs["pred_boxes"].detach().float()
        device = pred_boxes.device
        bsz, num_queries, num_slots = logits.shape
        patch_mask = outputs.get("patch_mask", None)
        if patch_mask is not None:
            patch_mask = patch_mask.to(device=device, dtype=torch.bool)

        union_logits = logits.max(dim=-1).values
        if hasattr(criterion, "compute_matching"):
            try:
                matching_outputs = dict(outputs)
                matching_outputs["pred_logits_patch"] = logits
                matching_outputs["pred_boxes"] = pred_boxes
                match_ctx = criterion.compute_matching(matching_outputs, targets)
            except Exception as e:
                self.match_errors += 1
                print(f"[WARN] compute_matching failed: {e}", file=sys.stderr)
                match_ctx = None
            if match_ctx is not None:
                for b, (src_idx, _tgt_idx) in enumerate(match_ctx["all_indices"]):
                    if src_idx.numel() == 0:
                        continue
                    src_cpu = src_idx.detach().cpu()
                    self.matched_total += int(src_cpu.numel())
                    for k in self.topks:
                        topq = torch.topk(union_logits[b], k=min(k, num_queries), largest=True).indices.detach().cpu()
                        self.matched_recalled[k] += int(torch.isin(src_cpu, topq).sum().item())

        for b in range(bsz):
            image_id = self.num_images
            self.num_images += 1
            target = targets[b]
            labels = target["labels"].detach().to(torch.long)
            gt_boxes = target["boxes"].detach().float()
            if labels.numel() == 0 or gt_boxes.numel() == 0:
                continue

            support_classes = _support_classes_from_target(target, k=num_slots, device=device)
            valid_slots = support_classes >= 0
            if patch_mask is not None:
                valid_slots = valid_slots & patch_mask[b]
            valid_slot_ids = valid_slots.nonzero(as_tuple=False).flatten()
            if valid_slot_ids.numel() == 0:
                self.num_targets += int(labels.numel())
                self.unsupported_targets += int(labels.numel())
                continue

            valid_label_set = set(int(x) for x in support_classes[valid_slot_ids].detach().cpu().tolist())
            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[b]).clamp(0.0, 1.0)
            gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes.to(device)).clamp(0.0, 1.0)
            iou_qt = box_ops.box_iou(pred_xyxy, gt_xyxy)[0]

            for j, label_value in enumerate(labels.detach().cpu().tolist()):
                label = int(label_value)
                self.num_targets += 1
                if label not in valid_label_set:
                    self.unsupported_targets += 1
                    continue
                self.box_targets += 1
                gt_box_list = gt_xyxy[j].detach().cpu().tolist()
                self.gt_by_key.setdefault((image_id, label), []).append([float(x) for x in gt_box_list])
                self.ap_targets += 1

                class_slots = ((support_classes == label) & valid_slots).nonzero(as_tuple=False).flatten()
                if class_slots.numel() == 0:
                    continue
                class_scores = logits[b, :, class_slots].max(dim=-1).values
                for k in self.topks:
                    topq = torch.topk(class_scores, k=min(k, num_queries), largest=True).indices
                    best_iou = float(iou_qt[topq, j].max().item()) if topq.numel() > 0 else 0.0
                    self.best_iou_sum[k] += best_iou
                    if best_iou >= 0.5:
                        self.box_recalled[k] += 1

            self._add_ap_detections(
                image_id=image_id,
                logits_b=logits[b],
                pred_xyxy=pred_xyxy,
                support_classes=support_classes,
                valid_slot_ids=valid_slot_ids,
            )

    def _add_ap_detections(
        self,
        *,
        image_id: int,
        logits_b: torch.Tensor,
        pred_xyxy: torch.Tensor,
        support_classes: torch.Tensor,
        valid_slot_ids: torch.Tensor,
    ) -> None:
        if self.ap_max_dets_per_image <= 0 or valid_slot_ids.numel() == 0:
            return
        if self.ap_score_agg == "class_max":
            det_scores = []
            det_labels = []
            det_query_idx = []
            for label in torch.unique(support_classes[valid_slot_ids]):
                class_slots = valid_slot_ids[support_classes[valid_slot_ids] == label]
                if class_slots.numel() == 0:
                    continue
                class_scores = logits_b[:, class_slots].max(dim=-1).values
                det_scores.append(class_scores)
                det_labels.append(label.expand_as(class_scores))
                det_query_idx.append(torch.arange(class_scores.numel(), device=logits_b.device))
            if not det_scores:
                return
            flat_scores = torch.cat(det_scores, dim=0)
            flat_labels = torch.cat(det_labels, dim=0)
            flat_query_idx = torch.cat(det_query_idx, dim=0)
        else:
            slot_scores = logits_b[:, valid_slot_ids]
            flat_scores = slot_scores.reshape(-1)
            if flat_scores.numel() == 0:
                return
            local_slot = torch.arange(flat_scores.numel(), device=logits_b.device) % int(valid_slot_ids.numel())
            flat_query_idx = torch.div(
                torch.arange(flat_scores.numel(), device=logits_b.device),
                int(valid_slot_ids.numel()),
                rounding_mode="floor",
            )
            flat_labels = support_classes[valid_slot_ids[local_slot]]

        if flat_scores.numel() == 0:
            return
        keep = min(int(self.ap_max_dets_per_image), int(flat_scores.numel()))
        top_scores, top_idx = torch.topk(flat_scores, k=keep, largest=True)
        query_idx = flat_query_idx[top_idx]
        det_labels = flat_labels[top_idx]
        det_boxes = pred_xyxy[query_idx]
        for score, label, box in zip(top_scores.detach().cpu(), det_labels.detach().cpu(), det_boxes.detach().cpu()):
            x1, y1, x2, y2 = [float(x) for x in box.tolist()]
            self.detections.append((float(score.item()), int(image_id), int(label.item()), x1, y1, x2, y2))

    def _compute_ap50(self) -> float:
        if self.ap_targets <= 0 or not self.detections:
            return 0.0
        gt_arrays = {
            key: np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            for key, boxes in self.gt_by_key.items()
            if boxes
        }
        matched = {key: np.zeros((boxes.shape[0],), dtype=bool) for key, boxes in gt_arrays.items()}
        ordered = sorted(self.detections, key=lambda x: x[0], reverse=True)
        tp = np.zeros((len(ordered),), dtype=np.float32)
        fp = np.zeros((len(ordered),), dtype=np.float32)
        for i, det in enumerate(ordered):
            _score, image_id, label, x1, y1, x2, y2 = det
            key = (int(image_id), int(label))
            boxes = gt_arrays.get(key, None)
            if boxes is None or boxes.shape[0] == 0:
                fp[i] = 1.0
                continue
            ious = _iou_one_to_many(np.asarray([x1, y1, x2, y2], dtype=np.float32), boxes)
            order = np.argsort(-ious)
            found = False
            for gt_i in order:
                if float(ious[gt_i]) < 0.5:
                    break
                if not matched[key][gt_i]:
                    matched[key][gt_i] = True
                    tp[i] = 1.0
                    found = True
                    break
            if not found:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(1.0, float(self.ap_targets))
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return _to_float(ap)

    def result(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "num_images": int(self.num_images),
            "num_targets": int(self.num_targets),
            "supported_targets": int(self.box_targets),
            "unsupported_targets": int(self.unsupported_targets),
            "target_support_coverage": self.box_targets / max(1, self.num_targets),
            "matched_total": int(self.matched_total),
            "match_errors": int(self.match_errors),
            "ap_targets": int(self.ap_targets),
            "ap_detections": int(len(self.detections)),
            "ap_score_agg": self.ap_score_agg,
            "patch_ap50": self._compute_ap50(),
        }
        for k in self.topks:
            out[f"matched_query_recall@{k}"] = self.matched_recalled[k] / max(1, self.matched_total)
            out[f"box_recall@{k}"] = self.box_recalled[k] / max(1, self.box_targets)
            out[f"mean_best_iou@{k}"] = self.best_iou_sum[k] / max(1, self.box_targets)
        return out


def _build_loader(
    *,
    cfg,
    datasetinfo: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
):
    _set_seed(seed)
    dataset = build_dataset(image_set="val", args=cfg, datasetinfo=datasetinfo)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _dataset_name(datasetinfo: Dict[str, Any], index: int) -> str:
    name = datasetinfo.get("name") or datasetinfo.get("source") or f"val{index}"
    return str(name).replace("/", "_")


def _checkpoint_label(path: str) -> str:
    p = Path(path)
    return p.stem.replace("/", "_")


@torch.no_grad()
def evaluate_one(
    *,
    cfg,
    model,
    criterion,
    datasetinfo: Dict[str, Any],
    dataset_name: str,
    ckpt_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    topks: List[int],
    ap_max_dets_per_image: int,
    ap_score_agg: str,
    amp: bool,
    max_batches: int,
    max_images: int,
    log_every: int,
) -> Dict[str, Any]:
    loader = _build_loader(
        cfg=cfg,
        datasetinfo=datasetinfo,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )
    acc = PatchEvalAccumulator(topks, ap_max_dets_per_image=ap_max_dets_per_image, ap_score_agg=ap_score_agg)
    start = time.time()
    total_batches = len(loader)
    print(
        f"[INFO] eval ckpt={Path(ckpt_path).name} dataset={dataset_name} "
        f"images={len(loader.dataset)} batches={total_batches} batch_size={batch_size}"
    )
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= max_batches:
            break
        if max_images > 0 and acc.num_images >= max_images:
            break
        outputs, targets = _forward_patch(model, batch, device, amp=amp)
        acc.update(outputs, targets, criterion)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and ((batch_i + 1) % log_every == 0 or batch_i == 0):
            elapsed = time.time() - start
            done_batches = batch_i + 1
            target_batches = min(total_batches, max_batches) if max_batches > 0 else total_batches
            if max_images > 0:
                target_batches = min(target_batches, math.ceil(max_images / max(1, batch_size)))
            eta = elapsed / max(1, done_batches) * max(0, target_batches - done_batches)
            print(
                f"[INFO] {dataset_name} {Path(ckpt_path).name}: "
                f"batch {done_batches}/{target_batches}, images={acc.num_images}, "
                f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
            )
    result = acc.result()
    result.update(
        {
            "checkpoint": str(ckpt_path),
            "checkpoint_name": Path(ckpt_path).name,
            "dataset": dataset_name,
            "seconds": time.time() - start,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "seed": int(seed),
            "max_batches": int(max_batches),
            "max_images": int(max_images),
            "ap_max_dets_per_image": int(ap_max_dets_per_image),
            "ap_score_agg": str(ap_score_agg),
        }
    )
    return result


def _mean_metric(results: List[Dict[str, Any]], ckpt: str, metric: str) -> float:
    vals = [float(r.get(metric, 0.0)) for r in results if r["checkpoint"] == ckpt]
    return sum(vals) / max(1, len(vals))


def _write_outputs(output_dir: Path, results: List[Dict[str, Any]], primary_metric: str) -> None:
    ckpts = []
    seen = set()
    for r in results:
        ckpt = r["checkpoint"]
        if ckpt not in seen:
            seen.add(ckpt)
            ckpts.append(ckpt)
    ranking = [
        {
            "rank": i + 1,
            "checkpoint": ckpt,
            f"mean_{primary_metric}": _mean_metric(results, ckpt, primary_metric),
            "mean_box_recall@50": _mean_metric(results, ckpt, "box_recall@50"),
            "mean_matched_query_recall@50": _mean_metric(results, ckpt, "matched_query_recall@50"),
        }
        for i, ckpt in enumerate(
            sorted(ckpts, key=lambda c: _mean_metric(results, c, primary_metric), reverse=True)
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"primary_metric": primary_metric, "ranking": ranking, "results": results}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    datasets = []
    seen_ds = set()
    for r in results:
        ds = r["dataset"]
        if ds not in seen_ds:
            seen_ds.add(ds)
            datasets.append(ds)

    lines = [
        f"# Stage-A patch eval summary",
        "",
        f"Primary metric: mean `{primary_metric}` across evaluated datasets.",
        "",
        "| rank | checkpoint | mean patch_ap50 | mean box_recall@50 | mean matched_query_recall@50 | "
        + " | ".join(f"{ds} patch_ap50" for ds in datasets)
        + " |",
        "|---:|---|---:|---:|---:|"
        + "|".join("---:" for _ in datasets)
        + "|",
    ]
    by_ckpt_ds = {(r["checkpoint"], r["dataset"]): r for r in results}
    for row in ranking:
        ckpt = row["checkpoint"]
        ds_ap = []
        for ds in datasets:
            r = by_ckpt_ds.get((ckpt, ds), {})
            ds_ap.append(f"{float(r.get('patch_ap50', 0.0)):.6f}")
        lines.append(
            f"| {row['rank']} | `{Path(ckpt).name}` | "
            f"{float(row[f'mean_{primary_metric}']):.6f} | "
            f"{float(row['mean_box_recall@50']):.6f} | "
            f"{float(row['mean_matched_query_recall@50']):.6f} | "
            + " | ".join(ds_ap)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _expand_ckpts(args) -> List[str]:
    ckpts = list(args.ckpts or [])
    if args.ckpt_glob:
        ckpts.extend(str(p) for p in sorted(Path().glob(args.ckpt_glob)))
    if not ckpts:
        raise ValueError("Pass --ckpts or --ckpt_glob.")
    out = []
    for ckpt in ckpts:
        p = Path(ckpt)
        if not p.exists():
            raise FileNotFoundError(str(p))
        out.append(str(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-A patch-only checkpoints on patch episodes.")
    parser.add_argument("--config", default="config/cfg_patch_stage_a.py")
    parser.add_argument("--datasets", default="config/datasets_patch_stage_a_lvis_coco2017_eval_local.json")
    parser.add_argument("--ckpts", nargs="*", default=[])
    parser.add_argument("--ckpt_glob", default=None)
    parser.add_argument("--output_dir", default="outputs/stageA_coco_multipatch_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--ap_max_dets_per_image", type=int, default=100)
    parser.add_argument("--ap_score_agg", default="slot", choices=["slot", "class_max"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--dataset_names", nargs="*", default=None)
    parser.add_argument("--primary_metric", default="patch_ap50")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    ckpts = _expand_ckpts(args)

    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.batch_size = int(args.batch_size)
    cfg.patch_only = True
    cfg.use_coco_eval = False

    with open(args.datasets, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)
    val_infos = list(dataset_meta.get("val", []))
    if not val_infos:
        raise ValueError(f"No val datasets found in {args.datasets}")
    if args.dataset_names:
        wanted = set(args.dataset_names)
        val_infos = [d for i, d in enumerate(val_infos) if _dataset_name(d, i) in wanted]
        if not val_infos:
            raise ValueError(f"No matching val datasets for --dataset_names={args.dataset_names}")

    results: List[Dict[str, Any]] = []
    for ckpt_i, ckpt_path in enumerate(ckpts):
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(ckpts)}: {ckpt_path}")
        _set_seed(int(args.seed))
        model, criterion = _load_model_and_criterion(cfg, ckpt_path, device)
        for ds_i, datasetinfo in enumerate(val_infos):
            ds_name = _dataset_name(datasetinfo, ds_i)
            ds_seed = int(args.seed) + ds_i * 100000
            result = evaluate_one(
                cfg=cfg,
                model=model,
                criterion=criterion,
                datasetinfo=datasetinfo,
                dataset_name=ds_name,
                ckpt_path=ckpt_path,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=ds_seed,
                topks=list(args.topk),
                ap_max_dets_per_image=int(args.ap_max_dets_per_image),
                ap_score_agg=str(args.ap_score_agg),
                amp=bool(args.amp),
                max_batches=int(args.max_batches),
                max_images=int(args.max_images),
                log_every=int(args.log_every),
            )
            results.append(result)
            per_file = output_dir / f"{_checkpoint_label(ckpt_path)}__{ds_name}.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            per_file.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            _write_outputs(output_dir, results, str(args.primary_metric))
            print(
                f"[RESULT] {Path(ckpt_path).name} {ds_name}: "
                f"patch_ap50={result['patch_ap50']:.6f} "
                f"box_recall@50={result.get('box_recall@50', 0.0):.6f} "
                f"matched_query_recall@50={result.get('matched_query_recall@50', 0.0):.6f}"
            )
        del model
        del criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_outputs(output_dir, results, str(args.primary_metric))
    print(f"[INFO] wrote {output_dir / 'summary.json'}")
    print(f"[INFO] wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
