#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
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
from models.GroundingDINO.stage_b_score import compute_stage_b_slot_logits  # noqa: E402
from tools.eval_refcoco_stageb import _ckpt_run_prefix, _load_model, _safe_name  # noqa: E402
from tools.eval_stagea_patch_checkpoints import _prepare_patch_batch, _set_seed  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _load_ref_split_map(data_root: Path, dataset: str, splitby: str) -> Dict[Tuple[int, int, int], str]:
    refs_path = data_root / "COCO" / dataset / f"refs({splitby}).p"
    refs = pickle.load(refs_path.open("rb"))
    out: Dict[Tuple[int, int, int], str] = {}
    for ref in refs:
        try:
            key = (int(ref["ref_id"]), int(ref["ann_id"]), int(ref["image_id"]))
        except Exception:
            continue
        out[key] = str(ref.get("split", ""))
    return out


def _split_specs() -> List[Dict[str, str]]:
    return [
        {
            "name": "refcocop_val",
            "pair_source": "refcoco+_unc",
            "dataset": "refcoco+",
            "splitby": "unc",
            "split": "val",
        },
        {
            "name": "refcocop_testA",
            "pair_source": "refcoco+_unc",
            "dataset": "refcoco+",
            "splitby": "unc",
            "split": "testA",
        },
        {
            "name": "refcocop_testB",
            "pair_source": "refcoco+_unc",
            "dataset": "refcoco+",
            "splitby": "unc",
            "split": "testB",
        },
        {
            "name": "refcocog_val",
            "pair_source": "refcocog_google",
            "dataset": "refcocog",
            "splitby": "google",
            "split": "val",
        },
    ]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_category(value: Any) -> str:
    values = _as_list(value)
    if not values:
        return "unknown"
    text = str(values[0]).strip()
    return text or "unknown"


def _build_tn_eval_jsonl(
    *,
    data_root: Path,
    output_dir: Path,
    tn_jsonl: Path,
    splits: List[str],
    max_pairs: int,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, int]]:
    specs = {spec["name"]: spec for spec in _split_specs()}
    wanted = list(splits)
    if wanted == ["all"]:
        wanted = list(specs)
    unknown = [name for name in wanted if name not in specs]
    if unknown:
        raise KeyError(f"Unknown TN split names: {unknown}; available={list(specs)} or all")

    split_maps: Dict[Tuple[str, str, str], Dict[Tuple[int, int, int], str]] = {}
    wanted_by_source_split: Dict[Tuple[str, str], str] = {}
    for name in wanted:
        spec = specs[name]
        map_key = (spec["dataset"], spec["splitby"], spec["pair_source"])
        if map_key not in split_maps:
            split_maps[map_key] = _load_ref_split_map(data_root, spec["dataset"], spec["splitby"])
        wanted_by_source_split[(spec["pair_source"], spec["split"])] = name

    out_dir = output_dir / "tn_eval_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_slug = "_".join(_safe_name(x) for x in wanted)
    out_path = out_dir / f"tn_{split_slug}.jsonl"
    meta_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {name: 0 for name in wanted}
    seen = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in _iter_jsonl(tn_jsonl):
            instances = row.get("instances")
            if not isinstance(instances, list) or not instances:
                continue
            inst = instances[0]
            if not isinstance(inst, dict):
                continue
            pair_source = str(inst.get("pair_source") or row.get("pair_source") or row.get("source") or "")
            if not pair_source:
                continue
            try:
                key = (int(row["ref_id"]), int(row["ann_id"]), int(row["image_id"]))
            except Exception:
                continue

            source_split = None
            for (dataset, splitby, source), split_map in split_maps.items():
                if source != pair_source:
                    continue
                source_split = split_map.get(key)
                if source_split is not None:
                    break
            if source_split is None:
                continue
            eval_split = wanted_by_source_split.get((pair_source, source_split))
            if eval_split is None:
                continue

            positive_phrase = inst.get("positive_phrase")
            if not isinstance(positive_phrase, str) or not positive_phrase.strip():
                continue
            inst["text_is_negative"] = True
            out_row = dict(row)
            out_row["tn_eval_split"] = eval_split
            out_row["tn_eval_pair_source"] = pair_source
            out_row["tn_eval_source_split"] = source_split
            out_row["instances"] = [inst]
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            meta_rows.append(
                {
                    "eval_split": eval_split,
                    "pair_source": pair_source,
                    "source_split": source_split,
                    "image_id": int(row["image_id"]),
                    "ann_id": int(row["ann_id"]),
                    "ref_id": int(row["ref_id"]),
                    "sent_id": int(row.get("sent_id", -1)),
                    "negative_phrase": inst.get("raw_phrase") or inst.get("phrase"),
                    "positive_phrase": positive_phrase,
                    "category": _first_category(inst.get("replace_category")),
                }
            )
            counts[eval_split] = counts.get(eval_split, 0) + 1
            seen += 1
            if max_pairs > 0 and seen >= int(max_pairs):
                break
    return out_path, meta_rows, counts


def _make_datasetinfo(data_root: Path, anno: Path) -> Dict[str, Any]:
    return {
        "name": "tn_val",
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
        "build_text_token_masks": True,
        "text_mask_skip_invalid_canonical": False,
        "text_mask_warn_limit": 0,
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


def _pad_target_mask(
    targets: List[Dict[str, Any]],
    key: str,
    kmax: int,
    tmax: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not any(key in target for target in targets):
        return None
    out = torch.zeros((len(targets), kmax, tmax), dtype=torch.bool, device=device)
    for i, target in enumerate(targets):
        mask = target.get(key)
        if not torch.is_tensor(mask):
            continue
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.dim() != 2:
            continue
        rows = min(kmax, int(mask.shape[0]))
        cols = min(tmax, int(mask.shape[-1]))
        if rows > 0 and cols > 0:
            out[i, :rows, :cols] = mask[:rows, :cols].to(device=device, dtype=torch.bool)
    return out


def _inject_text_masks(
    outputs: Dict[str, torch.Tensor],
    raw_targets: List[Dict[str, Any]],
    *,
    phrase_key: str,
    canonical_key: str,
    device: torch.device,
) -> None:
    patch_logits = outputs["pred_logits_patch"]
    kmax = 1 if patch_logits.dim() == 2 else int(patch_logits.shape[-1])
    tmax = int(outputs["pred_logits_text"].shape[-1])
    phrase_mask = _pad_target_mask(raw_targets, phrase_key, kmax, tmax, device)
    canonical_mask = _pad_target_mask(raw_targets, canonical_key, kmax, tmax, device)
    if phrase_mask is not None:
        outputs["phrase_to_token_mask"] = phrase_mask
    if canonical_mask is not None:
        outputs["canonical_to_token_mask"] = canonical_mask


def _rank_positive_captions(raw_targets: List[Dict[str, Any]]) -> Tuple[List[str], torch.Tensor]:
    captions: List[str] = []
    valid: List[bool] = []
    for target in raw_targets:
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
def _forward_pair(model, batch, device: torch.device, *, amp: bool):
    raw_targets = list(batch[1])
    samples, targets, neg_captions, patches, patch_global, patch_mask = _prepare_patch_batch(*batch, device)
    pos_captions, valid_pos = _rank_positive_captions(raw_targets)
    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        neg_outputs = model(
            samples,
            targets=targets,
            captions=neg_captions,
            patches=patches,
            patch_global=patch_global,
            patch_mask=patch_mask,
            patch_only=True,
            patch_only_compute_text_logits=True,
            disable_patch_dn=True,
        )
        pos_outputs = model(
            samples,
            targets=targets,
            captions=pos_captions,
            patches=patches,
            patch_global=patch_global,
            patch_mask=patch_mask,
            patch_only=True,
            patch_only_compute_text_logits=True,
            disable_patch_dn=True,
        )
    _inject_text_masks(
        neg_outputs,
        raw_targets,
        phrase_key="phrase_to_token_mask",
        canonical_key="canonical_to_token_mask",
        device=device,
    )
    _inject_text_masks(
        pos_outputs,
        raw_targets,
        phrase_key="rank_positive_phrase_to_token_mask",
        canonical_key="rank_positive_canonical_to_token_mask",
        device=device,
    )
    return neg_outputs, pos_outputs, targets, valid_pos.to(device=device)


def _slot_scores(outputs: Dict[str, torch.Tensor], cfg, beta: float) -> torch.Tensor:
    return compute_stage_b_slot_logits(
        outputs,
        beta=float(beta),
        canonical_weight=float(getattr(cfg, "stage_b_infer_canonical_weight", 0.15)),
        text_agg=str(getattr(cfg, "stage_b_infer_text_agg", "mean")),
        softmin_tau=float(getattr(cfg, "stage_b_infer_softmin_tau", 0.7)),
        mean_softmin_alpha=float(getattr(cfg, "stage_b_infer_mean_softmin_alpha", 0.5)),
    )


def _best_scores_and_iou(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], cfg, beta: float):
    slot_logits = _slot_scores(outputs, cfg, beta)
    bsz, _q, k = slot_logits.shape
    flat = slot_logits.reshape(bsz, -1)
    score, flat_idx = flat.max(dim=1)
    query_idx = torch.div(flat_idx, k, rounding_mode="floor")

    pred_boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
    ious: List[float] = []
    for b, target in enumerate(targets):
        gt_boxes = target.get("boxes")
        if (not torch.is_tensor(gt_boxes)) or gt_boxes.numel() == 0:
            ious.append(float("nan"))
            continue
        gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1].detach().float()).clamp(0.0, 1.0)[0]
        q = int(query_idx[b].item())
        iou = box_ops.box_iou(pred_boxes[b, q : q + 1], gt.view(1, 4))[0].view(-1)[0]
        ious.append(float(iou.item()))
    return score.detach().float().cpu().numpy(), np.asarray(ious, dtype=np.float32)


def _score_at_best_iou(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], cfg, beta: float):
    slot_logits = _slot_scores(outputs, cfg, beta).detach().float()
    pred_boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
    scores: List[float] = []
    ious: List[float] = []
    for b, target in enumerate(targets):
        gt_boxes = target.get("boxes")
        if (not torch.is_tensor(gt_boxes)) or gt_boxes.numel() == 0:
            scores.append(float("nan"))
            ious.append(float("nan"))
            continue
        gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1].detach().float()).clamp(0.0, 1.0)[0]
        query_ious = box_ops.box_iou(pred_boxes[b], gt.view(1, 4))[0].view(-1)
        q = int(query_ious.argmax().item())
        scores.append(float(slot_logits[b, q, 0].item()))
        ious.append(float(query_ious[q].item()))
    return np.asarray(scores, dtype=np.float32), np.asarray(ious, dtype=np.float32)


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _safe_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _threshold_for_tpr(pos_scores: np.ndarray, target_tpr: float) -> float:
    pos_scores = pos_scores[np.isfinite(pos_scores)]
    if pos_scores.size == 0:
        return float("inf")
    target_tpr = min(1.0, max(0.0, float(target_tpr)))
    return float(np.quantile(pos_scores, 1.0 - target_tpr))


def _summarize_arrays(
    *,
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    pos_iou: np.ndarray,
    neg_iou: np.ndarray,
    pos_iou_score: np.ndarray,
    neg_iou_score: np.ndarray,
    threshold_tprs: List[float],
) -> Dict[str, Any]:
    valid = np.isfinite(pos_scores) & np.isfinite(neg_scores)
    pos_scores = pos_scores[valid]
    neg_scores = neg_scores[valid]
    pos_iou = pos_iou[valid]
    neg_iou = neg_iou[valid]
    pos_iou_score = pos_iou_score[valid]
    neg_iou_score = neg_iou_score[valid]
    gap = pos_scores - neg_scores
    out: Dict[str, Any] = {
        "num_pairs": int(pos_scores.size),
        "pair_win_rate": float(np.mean(pos_scores > neg_scores)) if pos_scores.size else 0.0,
        "pair_tie_rate": float(np.mean(pos_scores == neg_scores)) if pos_scores.size else 0.0,
        "score_gap_mean": _safe_mean(gap),
        "score_gap_median": _safe_median(gap),
        "pos_score_mean": _safe_mean(pos_scores),
        "tn_score_mean": _safe_mean(neg_scores),
        "pos_score_median": _safe_median(pos_scores),
        "tn_score_median": _safe_median(neg_scores),
        "pos_top1_iou50": float(np.mean(pos_iou >= 0.5)) if pos_iou.size else 0.0,
        "tn_top1_iou50": float(np.mean(neg_iou >= 0.5)) if neg_iou.size else 0.0,
        "pos_best_iou_query_score_mean": _safe_mean(pos_iou_score),
        "tn_best_iou_query_score_mean": _safe_mean(neg_iou_score),
        "best_iou_query_pair_win_rate": (
            float(np.mean(pos_iou_score > neg_iou_score)) if pos_iou_score.size else 0.0
        ),
    }
    for tpr in threshold_tprs:
        key = f"{int(round(float(tpr) * 100)):02d}"
        threshold = _threshold_for_tpr(pos_scores, float(tpr))
        actual_tpr = float(np.mean(pos_scores >= threshold)) if pos_scores.size else 0.0
        fpr = float(np.mean(neg_scores >= threshold)) if neg_scores.size else 0.0
        out[f"threshold_at_{key}tpr"] = threshold
        out[f"actual_tpr_at_{key}tpr"] = actual_tpr
        out[f"fpr{key}tpr"] = fpr
    out.setdefault("fpr95tpr", 0.0)
    out["tn_fpr"] = float(out.get("fpr95tpr", 0.0))
    return out


def _summarize_group(
    records: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
    threshold_tprs: List[float],
    key: str,
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[int]] = {}
    for i, meta in enumerate(metas):
        value = str(meta.get(key, "unknown"))
        groups.setdefault(value, []).append(i)
    out: Dict[str, Dict[str, Any]] = {}
    for value, idxs in groups.items():
        if not idxs:
            continue
        idx = np.asarray(idxs, dtype=np.int64)
        out[value] = _summarize_arrays(
            pos_scores=np.asarray([records[i]["pos_score"] for i in idx], dtype=np.float32),
            neg_scores=np.asarray([records[i]["tn_score"] for i in idx], dtype=np.float32),
            pos_iou=np.asarray([records[i]["pos_iou"] for i in idx], dtype=np.float32),
            neg_iou=np.asarray([records[i]["tn_iou"] for i in idx], dtype=np.float32),
            pos_iou_score=np.asarray([records[i]["pos_best_iou_query_score"] for i in idx], dtype=np.float32),
            neg_iou_score=np.asarray([records[i]["tn_best_iou_query_score"] for i in idx], dtype=np.float32),
            threshold_tprs=threshold_tprs,
        )
    return out


class TnPairAccumulator:
    def __init__(self, betas: Iterable[float]) -> None:
        self.betas = [float(beta) for beta in betas]
        self.records: Dict[float, List[Dict[str, float]]] = {beta: [] for beta in self.betas}
        self.metas: List[Dict[str, Any]] = []
        self.invalid_positive = 0

    def update(
        self,
        *,
        neg_outputs: Dict[str, torch.Tensor],
        pos_outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        valid_pos: torch.Tensor,
        metas: List[Dict[str, Any]],
        cfg,
    ) -> None:
        valid_np = valid_pos.detach().cpu().numpy().astype(bool)
        self.invalid_positive += int((~valid_np).sum())
        for beta in self.betas:
            neg_score, neg_iou = _best_scores_and_iou(neg_outputs, targets, cfg, beta)
            pos_score, pos_iou = _best_scores_and_iou(pos_outputs, targets, cfg, beta)
            neg_iou_score, _neg_iou_best = _score_at_best_iou(neg_outputs, targets, cfg, beta)
            pos_iou_score, _pos_iou_best = _score_at_best_iou(pos_outputs, targets, cfg, beta)
            for i, ok in enumerate(valid_np):
                if not ok:
                    continue
                self.records[beta].append(
                    {
                        "pos_score": float(pos_score[i]),
                        "tn_score": float(neg_score[i]),
                        "pos_iou": float(pos_iou[i]),
                        "tn_iou": float(neg_iou[i]),
                        "pos_best_iou_query_score": float(pos_iou_score[i]),
                        "tn_best_iou_query_score": float(neg_iou_score[i]),
                    }
                )
        for i, ok in enumerate(valid_np):
            if ok:
                self.metas.append(metas[i])

    def results(
        self,
        *,
        checkpoint: str,
        threshold_tprs: List[float],
        elapsed: float,
        batch_size: int,
        num_workers: int,
        seed: int,
        max_batches: int,
        max_pairs: int,
    ) -> List[Dict[str, Any]]:
        run_prefix = _ckpt_run_prefix(checkpoint)
        rows: List[Dict[str, Any]] = []
        for beta in self.betas:
            recs = self.records[beta]
            summary = _summarize_arrays(
                pos_scores=np.asarray([r["pos_score"] for r in recs], dtype=np.float32),
                neg_scores=np.asarray([r["tn_score"] for r in recs], dtype=np.float32),
                pos_iou=np.asarray([r["pos_iou"] for r in recs], dtype=np.float32),
                neg_iou=np.asarray([r["tn_iou"] for r in recs], dtype=np.float32),
                pos_iou_score=np.asarray([r["pos_best_iou_query_score"] for r in recs], dtype=np.float32),
                neg_iou_score=np.asarray([r["tn_best_iou_query_score"] for r in recs], dtype=np.float32),
                threshold_tprs=threshold_tprs,
            )
            summary.update(
                {
                    "run_id": f"{run_prefix}:b{beta:g}",
                    "checkpoint": str(checkpoint),
                    "checkpoint_name": Path(checkpoint).name,
                    "checkpoint_run_prefix": run_prefix,
                    "beta": float(beta),
                    "seconds": float(elapsed),
                    "batch_size": int(batch_size),
                    "num_workers": int(num_workers),
                    "seed": int(seed),
                    "max_batches": int(max_batches),
                    "max_pairs": int(max_pairs),
                    "invalid_positive_pairs": int(self.invalid_positive),
                    "by_split": _summarize_group(recs, self.metas, threshold_tprs, "eval_split"),
                    "by_category": _summarize_group(recs, self.metas, threshold_tprs, "category"),
                }
            )
            rows.append(summary)
        return rows


@torch.no_grad()
def evaluate_checkpoint(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    meta_rows: List[Dict[str, Any]],
    device: torch.device,
    betas: List[float],
    threshold_tprs: List[float],
    batch_size: int,
    num_workers: int,
    seed: int,
    amp: bool,
    max_batches: int,
    max_pairs: int,
    log_every: int,
) -> List[Dict[str, Any]]:
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    total_batches = len(loader)
    if max_pairs > 0:
        total_batches = min(total_batches, math.ceil(int(max_pairs) / max(1, int(batch_size))))
    if max_batches > 0:
        total_batches = min(total_batches, int(max_batches))
    acc = TnPairAccumulator(betas)
    start = time.time()
    offset = 0
    print(
        f"[INFO] TN eval ckpt={ckpt_path} pairs={len(loader.dataset)} batches={len(loader)} "
        f"batch_size={batch_size} betas={betas}"
    )
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= int(max_batches):
            break
        raw_bsz = len(batch[1])
        if max_pairs > 0 and offset >= int(max_pairs):
            break
        metas = meta_rows[offset : offset + raw_bsz]
        offset += raw_bsz
        neg_outputs, pos_outputs, targets, valid_pos = _forward_pair(model, batch, device, amp=amp)
        acc.update(
            neg_outputs=neg_outputs,
            pos_outputs=pos_outputs,
            targets=targets,
            valid_pos=valid_pos,
            metas=metas,
            cfg=cfg,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (batch_i == 0 or (batch_i + 1) % int(log_every) == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            eta = elapsed / max(1, done) * max(0, total_batches - done)
            used = sum(len(v) for v in acc.records.values()) // max(1, len(acc.records))
            print(
                f"[INFO] {Path(ckpt_path).parent.name}/{Path(ckpt_path).name}: "
                f"batch {done}/{total_batches}, valid_pairs={used}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
            )
    return acc.results(
        checkpoint=ckpt_path,
        threshold_tprs=threshold_tprs,
        elapsed=time.time() - start,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        max_batches=max_batches,
        max_pairs=max_pairs,
    )


def _write_summary(output_dir: Path, rows: List[Dict[str, Any]], primary_metric: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reverse = primary_metric in {"pair_win_rate", "score_gap_mean", "score_gap_median"}
    ranking = [
        dict(row, rank=i + 1)
        for i, row in enumerate(
            sorted(
                rows,
                key=lambda r: (
                    float(r.get(primary_metric, 0.0)),
                    -float(r.get("pair_win_rate", 0.0)),
                    float(r.get("tn_score_mean", 0.0)),
                ),
                reverse=reverse,
            )
        )
    ]
    payload = {"primary_metric": primary_metric, "ranking": ranking, "results": rows}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    split_names: List[str] = []
    seen = set()
    for row in rows:
        for split in row.get("by_split", {}):
            if split not in seen:
                seen.add(split)
                split_names.append(split)
    header = [
        "# Stage-B TN-Val Rejection Evaluation",
        "",
        f"Primary metric: `{primary_metric}`. Lower is better for FPR metrics.",
        "",
        "| rank | run | beta | fpr@95tpr | fpr@90tpr | pair win | gap mean | pos mean | TN mean | pos IoU50 | TN IoU50 | pairs |"
        + "".join(f" {split} fpr95 |" for split in split_names),
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        + "".join("---:|" for _ in split_names),
    ]
    lines = list(header)
    for row in ranking:
        split_cells = []
        by_split = row.get("by_split", {})
        for split in split_names:
            split_cells.append(f"{float(by_split.get(split, {}).get('fpr95tpr', 0.0)):.6f}")
        lines.append(
            f"| {int(row['rank'])} | `{row['run_id']}` | {float(row['beta']):g} | "
            f"{float(row.get('fpr95tpr', 0.0)):.6f} | "
            f"{float(row.get('fpr90tpr', 0.0)):.6f} | "
            f"{float(row.get('pair_win_rate', 0.0)):.6f} | "
            f"{float(row.get('score_gap_mean', 0.0)):.6f} | "
            f"{float(row.get('pos_score_mean', 0.0)):.6f} | "
            f"{float(row.get('tn_score_mean', 0.0)):.6f} | "
            f"{float(row.get('pos_top1_iou50', 0.0)):.6f} | "
            f"{float(row.get('tn_top1_iou50', 0.0)):.6f} | "
            f"{int(row.get('num_pairs', 0))} | "
            + " | ".join(split_cells)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        checkpoint = str(row.get("checkpoint", ""))
        if not run_id:
            continue
        by_key[(run_id, checkpoint)] = row
    return list(by_key.values())


def _load_existing_rows(output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not output_dir.exists():
        return rows
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            rows.extend([row for row in data if isinstance(row, dict)])
    return _dedupe_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-B true-negative phrase rejection on TN val pairs.")
    parser.add_argument("--config", default="config/cfg_patch_stage_b.py")
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/stageb_tn_val_compare")
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))
    parser.add_argument(
        "--tn_jsonl",
        default=None,
        help="TN patch_episode jsonl. Defaults to DATA_ROOT/patch_episode_prebuilt/refexp_tn_stageb_v1.jsonl.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--threshold_tprs", nargs="+", type=float, default=[0.75, 0.9, 0.95])
    parser.add_argument("--splits", nargs="+", default=["refcocop_val", "refcocog_val"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=0, help="Maximum TN pairs after split filtering; 0 means full.")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--primary_metric", default="fpr95tpr")
    parser.add_argument("--append_existing", action="store_true", help="Include existing per-checkpoint JSON rows in summary.")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    tn_jsonl = Path(args.tn_jsonl) if args.tn_jsonl else data_root / "patch_episode_prebuilt" / "refexp_tn_stageb_v1.jsonl"
    if not tn_jsonl.exists():
        raise FileNotFoundError(tn_jsonl)

    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.patch_only = True
    cfg.patch_only_compute_text_logits = True
    cfg.build_text_token_masks = True
    cfg.use_coco_eval = False
    cfg.batch_size = int(args.batch_size)
    cfg.text_mask_warn_limit = 0

    tn_eval_jsonl, meta_rows, counts = _build_tn_eval_jsonl(
        data_root=data_root,
        output_dir=output_dir,
        tn_jsonl=tn_jsonl,
        splits=list(args.splits),
        max_pairs=int(args.max_pairs),
    )
    if not meta_rows:
        raise RuntimeError(f"No TN rows selected from {tn_jsonl} for splits={args.splits}")
    datasetinfo = _make_datasetinfo(data_root, tn_eval_jsonl)
    print(f"[INFO] built TN eval jsonl: {tn_eval_jsonl} rows={len(meta_rows)} split_counts={counts}")

    rows: List[Dict[str, Any]] = _load_existing_rows(output_dir) if bool(args.append_existing) else []
    if rows:
        print(f"[INFO] loaded {len(rows)} existing result rows from {output_dir}")
    for ckpt_i, ckpt_path in enumerate(args.ckpts):
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(args.ckpts)}: {ckpt_path}")
        _set_seed(int(args.seed))
        model = _load_model(cfg, ckpt_path, device)
        ckpt_rows = evaluate_checkpoint(
            cfg=cfg,
            model=model,
            ckpt_path=ckpt_path,
            datasetinfo=datasetinfo,
            meta_rows=meta_rows,
            device=device,
            betas=list(args.betas),
            threshold_tprs=list(args.threshold_tprs),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            amp=bool(args.amp),
            max_batches=int(args.max_batches),
            max_pairs=int(args.max_pairs),
            log_every=int(args.log_every),
        )
        rows = _dedupe_rows(rows + ckpt_rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{_ckpt_run_prefix(ckpt_path)}.json").write_text(
            json.dumps(ckpt_rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_summary(output_dir, rows, str(args.primary_metric))
        for row in ckpt_rows:
            print(
                f"[RESULT] {row['run_id']}: fpr95={row.get('fpr95tpr', 0.0):.6f} "
                f"fpr90={row.get('fpr90tpr', 0.0):.6f} pair_win={row.get('pair_win_rate', 0.0):.6f} "
                f"gap={row.get('score_gap_mean', 0.0):.6f} "
                f"tn_iou50={row.get('tn_top1_iou50', 0.0):.6f} pairs={row.get('num_pairs', 0)}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_summary(output_dir, rows, str(args.primary_metric))
    print(f"[INFO] wrote {output_dir / 'summary.json'}")
    print(f"[INFO] wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
