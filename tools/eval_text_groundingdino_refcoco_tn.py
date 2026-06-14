#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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
from tools.eval_refcoco_stageb import (  # noqa: E402
    _build_split_jsonl,
    _ckpt_run_prefix,
    _default_splits,
    _load_canonical_name_maps,
    _load_model,
    _load_phrase_maps,
)
from tools.eval_stagea_patch_checkpoints import _set_seed  # noqa: E402
from tools.eval_stageb_tn_val import _build_tn_eval_jsonl  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _make_datasetinfo(data_root: Path, name: str, anno: Path) -> Dict[str, Any]:
    return {
        "name": name,
        "dataset_mode": "patch_episode",
        "root": "/",
        "anno": str(anno),
        "box_format": "xywh",
        "canonical_classes_json": str(data_root / "canonical_classes_with_aliases.json"),
        "support_patch_tsv": str(data_root / "patches_quality_emb" / "emb_index_from_quality.tsv"),
        "support_patch_bucket": "clean",
        "support_patch_use_embedding": False,
        "support_patch_image_root": str(data_root / "patches_quality"),
        "support_patch_max_per_class": 200,
        "patch_emb_cache_size": 4096,
        "keep_only_support_gt": True,
        "support_min_count": 2,
        "support_patch_size": 224,
        "support_num_patches_min": 1,
        "support_num_patches_max": 1,
        "build_text_token_masks": True,
        "text_mask_skip_invalid_canonical": False,
        "text_mask_warn_limit": 0,
        "tn_balance_sampling": False,
    }


def _build_loader(
    cfg,
    datasetinfo: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
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


def _first_mask(target: Dict[str, Any], key: str, device: torch.device, tmax: int) -> Optional[torch.Tensor]:
    mask = target.get(key)
    if not torch.is_tensor(mask):
        return None
    if mask.dim() == 2:
        mask = mask[0]
    elif mask.dim() != 1:
        return None
    out = torch.zeros((tmax,), dtype=torch.bool, device=device)
    cols = min(tmax, int(mask.shape[-1]))
    if cols > 0:
        out[:cols] = mask[:cols].to(device=device, dtype=torch.bool)
    if not bool(out.any().item()):
        return None
    return out


def _phrase_scores(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]], mask_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = outputs["pred_logits"].detach().float()
    boxes = outputs["pred_boxes"].detach().float()
    device = logits.device
    bsz, num_queries, tmax = logits.shape
    scores = torch.full((bsz, num_queries), -1.0, dtype=torch.float32, device=device)
    valid = torch.zeros((bsz,), dtype=torch.bool, device=device)
    for b, target in enumerate(targets):
        mask = _first_mask(target, mask_key, device, tmax)
        if mask is None:
            continue
        valid[b] = True
        denom = mask.to(torch.float32).sum().clamp(min=1.0)
        scores[b] = logits[b].sigmoid().masked_fill(~mask[None, :], 0.0).sum(dim=-1) / denom
    del boxes
    return scores, valid


def _top_iou(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]], scores: torch.Tensor) -> np.ndarray:
    pred_xyxy = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
    topq = scores.argmax(dim=1)
    ious: List[float] = []
    for b, target in enumerate(targets):
        gt_boxes = target.get("boxes")
        if (not torch.is_tensor(gt_boxes)) or gt_boxes.numel() == 0:
            ious.append(float("nan"))
            continue
        gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1].to(pred_xyxy.device).detach().float()).clamp(0.0, 1.0)[0]
        q = int(topq[b].item())
        iou = box_ops.box_iou(pred_xyxy[b, q : q + 1], gt.view(1, 4))[0].view(-1)[0]
        ious.append(float(iou.item()))
    return np.asarray(ious, dtype=np.float32)


class RefCocoTextAccumulator:
    def __init__(self, topks: Iterable[int]) -> None:
        self.topks = sorted({max(1, int(k)) for k in topks})
        self.total = 0
        self.valid_masks = 0
        self.correct50 = {k: 0 for k in self.topks}
        self.correct25 = {k: 0 for k in self.topks}
        self.iou_sum = {k: 0.0 for k in self.topks}

    def update(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]]) -> None:
        scores, valid = _phrase_scores(outputs, targets, "phrase_to_token_mask")
        pred_xyxy = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
        bsz, num_queries = scores.shape
        max_topk = min(max(self.topks), int(num_queries))
        top_idx = torch.topk(scores, k=max_topk, dim=1, largest=True).indices
        for b, target in enumerate(targets):
            self.total += 1
            if not bool(valid[b].item()):
                continue
            self.valid_masks += 1
            gt_boxes = target.get("boxes")
            if (not torch.is_tensor(gt_boxes)) or gt_boxes.numel() == 0:
                continue
            gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1].to(pred_xyxy.device).detach().float()).clamp(0.0, 1.0)[0]
            query_ious = torch.stack(
                [box_ops.box_iou(pred_xyxy[b, q : q + 1], gt.view(1, 4))[0].view(-1)[0] for q in top_idx[b]]
            )
            for k in self.topks:
                local = query_ious[: min(k, int(query_ious.numel()))]
                best_iou = float(local.max().item()) if local.numel() else 0.0
                self.iou_sum[k] += best_iou
                if best_iou >= 0.5:
                    self.correct50[k] += 1
                if best_iou >= 0.25:
                    self.correct25[k] += 1

    def result(self) -> Dict[str, Any]:
        denom = max(1, int(self.valid_masks))
        out: Dict[str, Any] = {
            "num_expressions": int(self.total),
            "valid_mask_expressions": int(self.valid_masks),
            "invalid_mask_expressions": int(self.total - self.valid_masks),
        }
        for k in self.topks:
            suffix = "" if k == 1 else f"@{k}"
            out[f"acc50{suffix}"] = float(self.correct50[k] / denom)
            out[f"acc25{suffix}"] = float(self.correct25[k] / denom)
            out[f"mean_iou{suffix}"] = float(self.iou_sum[k] / denom)
        return out


def _threshold_for_tpr(pos_scores: np.ndarray, target_tpr: float) -> float:
    pos_scores = pos_scores[np.isfinite(pos_scores)]
    if pos_scores.size == 0:
        return float("inf")
    return float(np.quantile(pos_scores, 1.0 - min(1.0, max(0.0, float(target_tpr)))))


def _summarize_tn_arrays(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    pos_iou: np.ndarray,
    neg_iou: np.ndarray,
    threshold_tprs: List[float],
) -> Dict[str, Any]:
    valid = np.isfinite(pos_scores) & np.isfinite(neg_scores)
    pos_scores = pos_scores[valid]
    neg_scores = neg_scores[valid]
    pos_iou = pos_iou[valid]
    neg_iou = neg_iou[valid]
    gap = pos_scores - neg_scores
    out: Dict[str, Any] = {
        "num_pairs": int(pos_scores.size),
        "pair_win_rate": float(np.mean(pos_scores > neg_scores)) if pos_scores.size else 0.0,
        "pair_tie_rate": float(np.mean(pos_scores == neg_scores)) if pos_scores.size else 0.0,
        "score_gap_mean": float(gap.mean()) if gap.size else 0.0,
        "score_gap_median": float(np.median(gap)) if gap.size else 0.0,
        "pos_score_mean": float(pos_scores.mean()) if pos_scores.size else 0.0,
        "tn_score_mean": float(neg_scores.mean()) if neg_scores.size else 0.0,
        "pos_top1_iou50": float(np.mean(pos_iou >= 0.5)) if pos_iou.size else 0.0,
        "tn_top1_iou50": float(np.mean(neg_iou >= 0.5)) if neg_iou.size else 0.0,
    }
    for tpr in threshold_tprs:
        key = f"{int(round(float(tpr) * 100)):02d}"
        threshold = _threshold_for_tpr(pos_scores, float(tpr))
        out[f"threshold_at_{key}tpr"] = threshold
        out[f"actual_tpr_at_{key}tpr"] = float(np.mean(pos_scores >= threshold)) if pos_scores.size else 0.0
        out[f"fpr{key}tpr"] = float(np.mean(neg_scores >= threshold)) if neg_scores.size else 0.0
    out.setdefault("fpr95tpr", 0.0)
    out["tn_fpr"] = float(out.get("fpr95tpr", 0.0))
    return out


def _summarize_tn_by_meta(records: List[Dict[str, float]], metas: List[Dict[str, Any]], key: str, threshold_tprs: List[float]):
    groups: Dict[str, List[int]] = {}
    for i, meta in enumerate(metas):
        groups.setdefault(str(meta.get(key, "unknown")), []).append(i)
    out: Dict[str, Any] = {}
    for name, idxs in groups.items():
        out[name] = _summarize_tn_arrays(
            np.asarray([records[i]["pos_score"] for i in idxs], dtype=np.float32),
            np.asarray([records[i]["tn_score"] for i in idxs], dtype=np.float32),
            np.asarray([records[i]["pos_iou"] for i in idxs], dtype=np.float32),
            np.asarray([records[i]["tn_iou"] for i in idxs], dtype=np.float32),
            threshold_tprs,
        )
    return out


def _positive_captions(targets: List[Dict[str, Any]]) -> Tuple[List[str], torch.Tensor]:
    captions: List[str] = []
    valid: List[bool] = []
    for target in targets:
        rank_caps = target.get("rank_positive_captions", None)
        has_rank = target.get("has_rank_positive", None)
        cap = None
        ok = False
        if isinstance(rank_caps, list) and rank_caps:
            maybe = rank_caps[0]
            if isinstance(maybe, str) and maybe.strip():
                cap = maybe
                ok = True
        if torch.is_tensor(has_rank):
            ok = ok and bool(has_rank.view(-1)[0].item()) if has_rank.numel() > 0 else False
        captions.append(cap if cap is not None else str(target.get("caption", "object .")))
        valid.append(bool(ok))
    return captions, torch.as_tensor(valid, dtype=torch.bool)


@torch.no_grad()
def evaluate_refcoco_dataset(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    topks: List[int],
    amp: bool,
    max_batches: int,
    log_every: int,
) -> Dict[str, Any]:
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    acc = RefCocoTextAccumulator(topks)
    start = time.time()
    print(
        f"[INFO] text RefCOCO eval ckpt={Path(ckpt_path).name} dataset={dataset_name} "
        f"expressions={len(loader.dataset)} batches={len(loader)} batch_size={batch_size}",
        flush=True,
    )
    for batch_i, (samples, targets) in enumerate(loader):
        if max_batches > 0 and batch_i >= int(max_batches):
            break
        samples = samples.to(device)
        targets = list(targets)
        captions = [str(t.get("caption", "object .")) for t in targets]
        with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
            outputs = model(samples, captions=captions)
        acc.update(outputs, targets)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (batch_i == 0 or (batch_i + 1) % int(log_every) == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            total = min(len(loader), max_batches) if max_batches > 0 else len(loader)
            eta = elapsed / max(1, done) * max(0, total - done)
            print(
                f"[INFO] {dataset_name} {Path(ckpt_path).name}: batch {done}/{total}, "
                f"expressions={acc.total}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                flush=True,
            )
    row = acc.result()
    row.update(
        {
            "run_id": _ckpt_run_prefix(ckpt_path),
            "checkpoint": str(ckpt_path),
            "checkpoint_name": Path(ckpt_path).name,
            "dataset": dataset_name,
            "seconds": float(time.time() - start),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "seed": int(seed),
            "max_batches": int(max_batches),
        }
    )
    return row


@torch.no_grad()
def evaluate_tn_dataset(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    meta_rows: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    threshold_tprs: List[float],
    amp: bool,
    max_batches: int,
    log_every: int,
) -> Dict[str, Any]:
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    records: List[Dict[str, float]] = []
    valid_metas: List[Dict[str, Any]] = []
    invalid_positive = 0
    invalid_negative = 0
    start = time.time()
    print(
        f"[INFO] text TN eval ckpt={Path(ckpt_path).name} pairs={len(loader.dataset)} "
        f"batches={len(loader)} batch_size={batch_size}",
        flush=True,
    )
    offset = 0
    for batch_i, (samples, targets) in enumerate(loader):
        if max_batches > 0 and batch_i >= int(max_batches):
            break
        raw_bsz = len(targets)
        metas = meta_rows[offset : offset + raw_bsz]
        offset += raw_bsz
        samples = samples.to(device)
        targets = list(targets)
        neg_captions = [str(t.get("caption", "object .")) for t in targets]
        pos_captions, valid_pos = _positive_captions(targets)
        with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
            neg_outputs = model(samples, captions=neg_captions)
            pos_outputs = model(samples, captions=pos_captions)
        neg_scores_q, valid_neg = _phrase_scores(neg_outputs, targets, "phrase_to_token_mask")
        pos_scores_q, valid_pos_mask = _phrase_scores(pos_outputs, targets, "rank_positive_phrase_to_token_mask")
        valid = valid_pos.to(valid_neg.device) & valid_neg & valid_pos_mask
        invalid_positive += int((~(valid_pos.to(valid_neg.device) & valid_pos_mask)).sum().item())
        invalid_negative += int((~valid_neg).sum().item())
        neg_best = neg_scores_q.max(dim=1).values.detach().cpu().numpy()
        pos_best = pos_scores_q.max(dim=1).values.detach().cpu().numpy()
        neg_iou = _top_iou(neg_outputs, targets, neg_scores_q)
        pos_iou = _top_iou(pos_outputs, targets, pos_scores_q)
        valid_np = valid.detach().cpu().numpy().astype(bool)
        for i, ok in enumerate(valid_np):
            if not ok:
                continue
            records.append(
                {
                    "pos_score": float(pos_best[i]),
                    "tn_score": float(neg_best[i]),
                    "pos_iou": float(pos_iou[i]),
                    "tn_iou": float(neg_iou[i]),
                }
            )
            valid_metas.append(metas[i])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (batch_i == 0 or (batch_i + 1) % int(log_every) == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            total = min(len(loader), max_batches) if max_batches > 0 else len(loader)
            eta = elapsed / max(1, done) * max(0, total - done)
            print(
                f"[INFO] TN {Path(ckpt_path).name}: batch {done}/{total}, valid_pairs={len(records)}, "
                f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                flush=True,
            )
    row = _summarize_tn_arrays(
        np.asarray([r["pos_score"] for r in records], dtype=np.float32),
        np.asarray([r["tn_score"] for r in records], dtype=np.float32),
        np.asarray([r["pos_iou"] for r in records], dtype=np.float32),
        np.asarray([r["tn_iou"] for r in records], dtype=np.float32),
        threshold_tprs,
    )
    row.update(
        {
            "run_id": _ckpt_run_prefix(ckpt_path),
            "checkpoint": str(ckpt_path),
            "checkpoint_name": Path(ckpt_path).name,
            "seconds": float(time.time() - start),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "seed": int(seed),
            "max_batches": int(max_batches),
            "invalid_positive_pairs": int(invalid_positive),
            "invalid_negative_pairs": int(invalid_negative),
            "by_split": _summarize_tn_by_meta(records, valid_metas, "eval_split", threshold_tprs),
            "by_category": _summarize_tn_by_meta(records, valid_metas, "category", threshold_tprs),
        }
    )
    return row


def _mean_metric(rows: List[Dict[str, Any]], run_id: str, metric: str) -> float:
    vals = [float(row.get(metric, 0.0)) for row in rows if row["run_id"] == run_id]
    return sum(vals) / max(1, len(vals))


def _write_summary(output_dir: Path, ref_rows: List[Dict[str, Any]], tn_rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"refcoco": ref_rows, "tn": tn_rows}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    run_ids: List[str] = []
    seen = set()
    for row in ref_rows:
        if row["run_id"] not in seen:
            seen.add(row["run_id"])
            run_ids.append(row["run_id"])
    datasets: List[str] = []
    seen_ds = set()
    for row in ref_rows:
        if row["dataset"] not in seen_ds:
            seen_ds.add(row["dataset"])
            datasets.append(row["dataset"])
    by_run_ds = {(row["run_id"], row["dataset"]): row for row in ref_rows}
    tn_by_run = {row["run_id"]: row for row in tn_rows}
    ranked = sorted(run_ids, key=lambda rid: _mean_metric(ref_rows, rid, "acc50"), reverse=True)
    lines = [
        "# Text GroundingDINO RefCOCO/TN Evaluation",
        "",
        "| rank | run | mean RefCOCO acc50 | TN fpr@95tpr | TN fpr@90tpr | TN pair win | TN gap | "
        + " | ".join(f"{ds} acc50" for ds in datasets)
        + " |",
        "|---:|---|---:|---:|---:|---:|---:|" + "|".join("---:" for _ in datasets) + "|",
    ]
    for i, run_id in enumerate(ranked, start=1):
        tn = tn_by_run.get(run_id, {})
        ds_vals = [f"{float(by_run_ds.get((run_id, ds), {}).get('acc50', 0.0)):.6f}" for ds in datasets]
        lines.append(
            f"| {i} | `{run_id}` | {float(_mean_metric(ref_rows, run_id, 'acc50')):.6f} | "
            f"{float(tn.get('fpr95tpr', 0.0)):.6f} | {float(tn.get('fpr90tpr', 0.0)):.6f} | "
            f"{float(tn.get('pair_win_rate', 0.0)):.6f} | {float(tn.get('score_gap_mean', 0.0)):.6f} | "
            + " | ".join(ds_vals)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ordinary text GroundingDINO on RefCOCO val and TN pairs.")
    parser.add_argument("--config", default="outputs/ogc_original_finetune_stage_a/cfg_ogc_original_finetune_stage_a.generated.py")
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/text_groundingdino_refcoco_tn_eval")
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))
    parser.add_argument("--tn_jsonl", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ref_splits", nargs="+", default=["refcoco_val", "refcocop_val", "refcocog_val"])
    parser.add_argument("--tn_splits", nargs="+", default=["refcocop_val", "refcocog_val"])
    parser.add_argument("--skip_tn", action="store_true", help="Only run RefCOCO splits and skip TN pair evaluation.")
    parser.add_argument("--topk", nargs="+", type=int, default=[1])
    parser.add_argument("--threshold_tprs", nargs="+", type=float, default=[0.75, 0.9, 0.95])
    parser.add_argument("--max_ref_batches", type=int, default=0)
    parser.add_argument("--max_tn_batches", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    tn_jsonl = Path(args.tn_jsonl) if args.tn_jsonl else data_root / "patch_episode_prebuilt" / "refexp_tn_stageb_v1.jsonl"

    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.patch_only = False
    cfg.use_coco_eval = False
    cfg.batch_size = int(args.batch_size)
    cfg.build_text_token_masks = True
    cfg.text_mask_warn_limit = 0

    canonical_json = data_root / "canonical_classes_with_aliases.json"
    name_to_id, id_to_name = _load_canonical_name_maps(canonical_json)
    phrase_maps = _load_phrase_maps(
        [
            data_root / "refcoco_text_pairs" / "refcoco_unc_pairs.jsonl",
            data_root / "data_proc" / "refcoco_text_pairs" / "refcoco+_unc_pairs.jsonl",
            data_root / "data_proc" / "refcoco_text_pairs" / "refcocog_google_pairs.jsonl",
            data_root / "patch_episode_prebuilt" / "refcocoplus_stageb_phrase_v1.jsonl",
            data_root / "patch_episode_prebuilt" / "refcocog_stageb_phrase_v1.jsonl",
        ]
    )
    split_specs = {spec["name"]: spec for spec in _default_splits()}
    wanted_ref = list(args.ref_splits)
    if wanted_ref == ["all"]:
        wanted_ref = list(split_specs)
    unknown = [name for name in wanted_ref if name not in split_specs]
    if unknown:
        raise KeyError(f"Unknown ref split names: {unknown}; available={list(split_specs)}")

    ref_datasetinfos = []
    for name in wanted_ref:
        spec = split_specs[name]
        jsonl_path, count = _build_split_jsonl(
            data_root=data_root,
            output_dir=output_dir,
            dataset=spec["dataset"],
            splitby=spec["splitby"],
            split=spec["split"],
            phrase_sources=list(spec["sources"]),
            phrase_maps=phrase_maps,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
        )
        print(f"[INFO] built RefCOCO split {name}: {count} expressions -> {jsonl_path}", flush=True)
        ref_datasetinfos.append((name, _make_datasetinfo(data_root, name, jsonl_path)))

    tn_meta_rows: List[Dict[str, Any]] = []
    tn_datasetinfo = None
    if not bool(args.skip_tn):
        tn_eval_jsonl, tn_meta_rows, tn_counts = _build_tn_eval_jsonl(
            data_root=data_root,
            output_dir=output_dir,
            tn_jsonl=tn_jsonl,
            splits=list(args.tn_splits),
            max_pairs=0,
        )
        print(f"[INFO] built TN split rows={len(tn_meta_rows)} counts={tn_counts} -> {tn_eval_jsonl}", flush=True)
        tn_datasetinfo = _make_datasetinfo(data_root, "tn_val", tn_eval_jsonl)

    ref_rows: List[Dict[str, Any]] = []
    tn_rows: List[Dict[str, Any]] = []
    for ckpt_i, ckpt_path in enumerate(args.ckpts):
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(args.ckpts)}: {ckpt_path}", flush=True)
        _set_seed(int(args.seed))
        model = _load_model(cfg, ckpt_path, device)
        for ds_i, (name, datasetinfo) in enumerate(ref_datasetinfos):
            row = evaluate_refcoco_dataset(
                cfg=cfg,
                model=model,
                ckpt_path=ckpt_path,
                datasetinfo=datasetinfo,
                dataset_name=name,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(args.seed) + ds_i * 100000,
                topks=list(args.topk),
                amp=bool(args.amp),
                max_batches=int(args.max_ref_batches),
                log_every=int(args.log_every),
            )
            ref_rows.append(row)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{_ckpt_run_prefix(ckpt_path)}__{name}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _write_summary(output_dir, ref_rows, tn_rows)
            print(f"[RESULT] {row['run_id']} {name}: acc50={row['acc50']:.6f} mean_iou={row['mean_iou']:.6f}", flush=True)
        if not bool(args.skip_tn):
            assert tn_datasetinfo is not None
            tn_row = evaluate_tn_dataset(
                cfg=cfg,
                model=model,
                ckpt_path=ckpt_path,
                datasetinfo=tn_datasetinfo,
                meta_rows=tn_meta_rows,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(args.seed),
                threshold_tprs=list(args.threshold_tprs),
                amp=bool(args.amp),
                max_batches=int(args.max_tn_batches),
                log_every=int(args.log_every),
            )
            tn_rows.append(tn_row)
            (output_dir / f"{_ckpt_run_prefix(ckpt_path)}__tn_val.json").write_text(
                json.dumps(tn_row, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _write_summary(output_dir, ref_rows, tn_rows)
            print(
                f"[RESULT] {tn_row['run_id']} TN: fpr95={tn_row.get('fpr95tpr', 0.0):.6f} "
                f"fpr90={tn_row.get('fpr90tpr', 0.0):.6f} pair_win={tn_row.get('pair_win_rate', 0.0):.6f}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_summary(output_dir, ref_rows, tn_rows)
    print(f"[INFO] wrote {output_dir / 'summary.json'}", flush=True)
    print(f"[INFO] wrote {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
