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
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from models.GroundingDINO.stage_b_score import compute_stage_b_slot_logits  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from tools.eval_stagea_patch_checkpoints import _prepare_patch_batch, _set_seed  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402

_WS_RE = re.compile(r"\s+")


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


def _load_model(cfg, ckpt_path: str, device: torch.device):
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={cfg.modelname}")
    model, _criterion, _postprocessors = build_func(cfg)
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    state = clean_state_dict(_extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] {ckpt_path}: missing keys={len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] {ckpt_path}: unexpected keys={len(unexpected)}", file=sys.stderr)
    model.to(device).eval()
    return model


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _norm_text(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "").replace("_", " ").replace(".", " ").strip().lower())


def _clean_phrase(value: Any) -> str:
    text = _WS_RE.sub(" ", str(value or "").replace("_", " ").replace(".", " ").strip())
    return text or "object"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _ckpt_run_prefix(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    parent = path.parent.name
    stem = path.stem
    return _safe_name(f"{parent}_{stem}") if parent else _safe_name(stem)


def _load_canonical_name_maps(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for row in data:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        cid = int(row["id"])
        preferred = row.get("base_name") or row.get("norm_name") or row.get("raw_name")
        if isinstance(preferred, str) and preferred.strip():
            id_to_name.setdefault(cid, _clean_phrase(preferred))
        values = [row.get("raw_name"), row.get("norm_name"), row.get("base_name")]
        values.extend(row.get("synonyms") or [])
        for value in values:
            if isinstance(value, str) and value.strip():
                name_to_id.setdefault(_norm_text(value), cid)
    return name_to_id, id_to_name


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _phrase_key(source: str, ref_id: Any, ann_id: Any, image_id: Any, phrase: Any) -> Tuple[str, int, int, int, str]:
    return (str(source), int(ref_id), int(ann_id), int(image_id), _norm_text(phrase))


def _load_phrase_maps(paths: Iterable[Path]) -> Dict[Tuple[str, int, int, int, str], Dict[str, Any]]:
    out: Dict[Tuple[str, int, int, int, str], Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            instances = row.get("instances")
            if isinstance(instances, list) and instances:
                for inst in instances:
                    phrase = inst.get("raw_phrase") or inst.get("phrase") or inst.get("positive_phrase")
                    source = inst.get("pair_source") or row.get("pair_source") or row.get("source")
                    ref_id = row.get("ref_id")
                    ann_id = row.get("ann_id")
                    image_id = row.get("image_id")
                    if source is None or ref_id is None or ann_id is None or image_id is None or not phrase:
                        continue
                    rec = dict(inst)
                    rec.update({"source": source, "ref_id": ref_id, "ann_id": ann_id, "image_id": image_id})
                    out.setdefault(_phrase_key(source, ref_id, ann_id, image_id, phrase), rec)
                continue
            phrase = row.get("raw_phrase") or row.get("phrase") or row.get("head_phrase")
            source = row.get("source") or row.get("pair_source")
            ref_id = row.get("ref_id")
            ann_id = row.get("ann_id")
            image_id = row.get("image_id")
            if source is None or ref_id is None or ann_id is None or image_id is None or not phrase:
                continue
            out.setdefault(_phrase_key(source, ref_id, ann_id, image_id, phrase), dict(row))
    return out


def _load_instances(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    anns = {int(row["id"]): row for row in data.get("annotations", [])}
    images = {int(row["id"]): row for row in data.get("images", [])}
    cats = {int(row["id"]): str(row.get("name", "")) for row in data.get("categories", [])}
    return anns, images, cats


def _image_path(data_root: Path, image: Dict[str, Any]) -> str:
    filename = str(image.get("file_name", ""))
    candidates = [
        data_root / "COCO" / "coco2014" / "train2014" / filename,
        data_root / "COCO" / "coco2014" / "val2014" / filename,
        data_root / "COCO" / "coco2017" / "train2017" / "train2017" / filename.replace("COCO_train2014_", ""),
        data_root / "COCO" / "coco2017" / "val2017" / filename.replace("COCO_val2014_", ""),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _resolve_class_record(
    *,
    phrase_maps: Dict[Tuple[str, int, int, int, str], Dict[str, Any]],
    phrase_sources: List[str],
    ref: Dict[str, Any],
    ann: Dict[str, Any],
    image_id: int,
    phrase: str,
    category_name: str,
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
) -> Dict[str, Any]:
    hit: Optional[Dict[str, Any]] = None
    for source in phrase_sources:
        hit = phrase_maps.get(_phrase_key(source, ref["ref_id"], ann["id"], image_id, phrase))
        if hit:
            break
    if hit is None:
        cid = name_to_id.get(_norm_text(category_name), int(ann.get("category_id", -1)))
        canon = id_to_name.get(int(cid), _clean_phrase(category_name))
        return {
            "class_id": int(cid),
            "head": canon,
            "head_phrase": canon,
            "canonical_name": canon,
            "class_id_source": "category_fallback",
        }
    cid = int(hit.get("class_id", hit.get("head_classifier_class_id", ann.get("category_id", -1))))
    canon = (
        hit.get("canonical_name")
        or hit.get("class_norm_name")
        or hit.get("class_raw_name")
        or id_to_name.get(cid)
        or category_name
    )
    return {
        "class_id": cid,
        "head": hit.get("head") or hit.get("try_tn_head") or canon,
        "head_phrase": hit.get("head_phrase") or hit.get("try_tn_head_phrase") or canon,
        "canonical_name": _clean_phrase(canon),
        "class_id_source": hit.get("class_id_source") or hit.get("label_match_type") or "phrase_map",
    }


def _build_split_jsonl(
    *,
    data_root: Path,
    output_dir: Path,
    dataset: str,
    splitby: str,
    split: str,
    phrase_sources: List[str],
    phrase_maps: Dict[Tuple[str, int, int, int, str], Dict[str, Any]],
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
) -> Tuple[Path, int]:
    ref_root = data_root / "COCO" / dataset
    refs_path = ref_root / f"refs({splitby}).p"
    instances_path = ref_root / "instances.json"
    refs = pickle.load(refs_path.open("rb"))
    anns, images, cats = _load_instances(instances_path)

    out_dir = output_dir / "refcoco_eval_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset.replace('+', 'plus')}_{splitby}_{split}.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ref in refs:
            if str(ref.get("split")) != str(split):
                continue
            ann = anns.get(int(ref["ann_id"]))
            image = images.get(int(ref["image_id"]))
            if ann is None or image is None:
                continue
            category_name = cats.get(int(ann.get("category_id", -1)), "")
            for sent in ref.get("sentences", []) or []:
                phrase = _clean_phrase(sent.get("sent") or sent.get("raw"))
                class_rec = _resolve_class_record(
                    phrase_maps=phrase_maps,
                    phrase_sources=phrase_sources,
                    ref=ref,
                    ann=ann,
                    image_id=int(ref["image_id"]),
                    phrase=phrase,
                    category_name=category_name,
                    name_to_id=name_to_id,
                    id_to_name=id_to_name,
                )
                row = {
                    "filename": _image_path(data_root, image),
                    "source": f"{dataset}_{splitby}_{split}",
                    "image_id": int(ref["image_id"]),
                    "ann_id": int(ref["ann_id"]),
                    "ref_id": int(ref["ref_id"]),
                    "sent_id": int(sent.get("sent_id", count)),
                    "split": split,
                    "instances": [
                        {
                            "bbox": ann["bbox"],
                            "class_id": int(class_rec["class_id"]),
                            "raw_phrase": phrase,
                            "head_phrase": _clean_phrase(class_rec["head_phrase"]),
                            "head": _clean_phrase(class_rec["head"]),
                            "canonical_name": _clean_phrase(class_rec["canonical_name"]),
                            "positive_phrase": phrase,
                            "text_is_negative": False,
                            "pair_source": phrase_sources[0] if phrase_sources else dataset,
                            "category_name": category_name,
                            "class_id_source": class_rec["class_id_source"],
                            "refcoco_category_id": int(ann.get("category_id", -1)),
                        }
                    ],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return out_path, count


def _default_splits() -> List[Dict[str, Any]]:
    return [
        {"name": "refcoco_val", "dataset": "refcoco", "splitby": "unc", "split": "val", "sources": ["refcoco_unc"]},
        {"name": "refcoco_testA", "dataset": "refcoco", "splitby": "unc", "split": "testA", "sources": ["refcoco_unc"]},
        {"name": "refcoco_testB", "dataset": "refcoco", "splitby": "unc", "split": "testB", "sources": ["refcoco_unc"]},
        {"name": "refcocop_val", "dataset": "refcoco+", "splitby": "unc", "split": "val", "sources": ["refcoco+_unc"]},
        {"name": "refcocop_testA", "dataset": "refcoco+", "splitby": "unc", "split": "testA", "sources": ["refcoco+_unc"]},
        {"name": "refcocop_testB", "dataset": "refcoco+", "splitby": "unc", "split": "testB", "sources": ["refcoco+_unc"]},
        {"name": "refcocog_val", "dataset": "refcocog", "splitby": "umd", "split": "val", "sources": ["refcocog_google"]},
        {"name": "refcocog_test", "dataset": "refcocog", "splitby": "umd", "split": "test", "sources": ["refcocog_google"]},
    ]


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
        "build_text_token_masks": True,
        "text_mask_skip_invalid_canonical": False,
    }


def _build_loader(cfg, datasetinfo: Dict[str, Any], batch_size: int, num_workers: int, device: torch.device, seed: int):
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


def _pad_target_mask(targets: List[Dict[str, Any]], key: str, kmax: int, device: torch.device) -> Optional[torch.Tensor]:
    if not any(key in t for t in targets):
        return None
    out = torch.zeros((len(targets), kmax, 256), dtype=torch.bool, device=device)
    for i, target in enumerate(targets):
        mask = target.get(key)
        if not torch.is_tensor(mask):
            continue
        rows = min(kmax, int(mask.shape[0]))
        cols = min(256, int(mask.shape[-1]))
        if rows > 0 and cols > 0:
            out[i, :rows, :cols] = mask[:rows, :cols].to(device=device, dtype=torch.bool)
    return out


@torch.no_grad()
def _forward(model, batch, device: torch.device, *, amp: bool):
    raw_targets = list(batch[1])
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
            patch_only_compute_text_logits=True,
        )
    if outputs["pred_logits_patch"].dim() == 2:
        kmax = 1
    else:
        kmax = int(outputs["pred_logits_patch"].shape[-1])
    phrase_mask = _pad_target_mask(raw_targets, "phrase_to_token_mask", kmax, device)
    canonical_mask = _pad_target_mask(raw_targets, "canonical_to_token_mask", kmax, device)
    if phrase_mask is not None:
        outputs["phrase_to_token_mask"] = phrase_mask
    if canonical_mask is not None:
        outputs["canonical_to_token_mask"] = canonical_mask
    return outputs, targets


def _box_iou_one(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().view(4)
    b = b.detach().float().view(4)
    x1 = torch.maximum(a[0], b[0])
    y1 = torch.maximum(a[1], b[1])
    x2 = torch.minimum(a[2], b[2])
    y2 = torch.minimum(a[3], b[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0)
    area_b = (b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0)
    return float((inter / (area_a + area_b - inter).clamp(min=1e-6)).item())


class RefExpAccumulator:
    def __init__(self, betas: Iterable[float], topks: Iterable[int]) -> None:
        self.betas = [float(b) for b in betas]
        self.topks = sorted({max(1, int(k)) for k in topks})
        self.total = 0
        self.iou_sum = {b: 0.0 for b in self.betas}
        self.correct50 = {b: 0 for b in self.betas}
        self.correct25 = {b: 0 for b in self.betas}
        self.recall = {(b, k): 0 for b in self.betas for k in self.topks}

    def update(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], *, cfg) -> None:
        pred_boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
        for beta in self.betas:
            slot_logits = compute_stage_b_slot_logits(
                outputs,
                beta=float(beta),
                canonical_weight=float(getattr(cfg, "stage_b_infer_canonical_weight", 0.15)),
                text_agg=str(getattr(cfg, "stage_b_infer_text_agg", "mean")),
                softmin_tau=float(getattr(cfg, "stage_b_infer_softmin_tau", 0.7)),
                mean_softmin_alpha=float(getattr(cfg, "stage_b_infer_mean_softmin_alpha", 0.5)),
            )
            bsz, q, k = slot_logits.shape
            flat = slot_logits.reshape(bsz, q * k)
            max_topk = min(max(self.topks), int(q * k))
            top_vals, top_idx = torch.topk(flat, k=max_topk, dim=1, largest=True)
            del top_vals
            query_idx = torch.div(top_idx, k, rounding_mode="floor")
            for b, target in enumerate(targets):
                if beta == self.betas[0]:
                    self.total += 1
                gt_boxes = target["boxes"].detach().float()
                if gt_boxes.numel() == 0:
                    continue
                gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1]).clamp(0.0, 1.0)[0]
                top_queries = query_idx[b]
                ious = torch.stack([box_ops.box_iou(pred_boxes[b, qi : qi + 1], gt.view(1, 4))[0].view(-1)[0] for qi in top_queries])
                best_iou = float(ious[0].item()) if ious.numel() else 0.0
                self.iou_sum[beta] += best_iou
                if best_iou >= 0.5:
                    self.correct50[beta] += 1
                if best_iou >= 0.25:
                    self.correct25[beta] += 1
                for topk in self.topks:
                    if bool((ious[: min(topk, ious.numel())] >= 0.5).any().item()):
                        self.recall[(beta, topk)] += 1

    def results(self) -> List[Dict[str, Any]]:
        out = []
        denom = max(1, int(self.total))
        for beta in self.betas:
            row = {
                "beta": float(beta),
                "num_expressions": int(self.total),
                "acc50": float(self.correct50[beta] / denom),
                "acc25": float(self.correct25[beta] / denom),
                "mean_iou_top1": float(self.iou_sum[beta] / denom),
            }
            for topk in self.topks:
                row[f"recall50@{topk}"] = float(self.recall[(beta, topk)] / denom)
            out.append(row)
        return out


@torch.no_grad()
def evaluate_dataset(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
    betas: List[float],
    topks: List[int],
    batch_size: int,
    num_workers: int,
    seed: int,
    amp: bool,
    max_batches: int,
    max_images: int,
    log_every: int,
) -> List[Dict[str, Any]]:
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    acc = RefExpAccumulator(betas, topks)
    start = time.time()
    total_batches = len(loader)
    print(
        f"[INFO] refexp eval ckpt={Path(ckpt_path).name} dataset={dataset_name} "
        f"expressions={len(loader.dataset)} batches={total_batches} batch_size={batch_size} betas={betas}"
    )
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= max_batches:
            break
        if max_images > 0 and acc.total >= max_images:
            break
        outputs, targets = _forward(model, batch, device, amp=amp)
        acc.update(outputs, targets, cfg=cfg)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (batch_i == 0 or (batch_i + 1) % log_every == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            target_batches = min(total_batches, max_batches) if max_batches > 0 else total_batches
            if max_images > 0:
                target_batches = min(target_batches, math.ceil(max_images / max(1, batch_size)))
            eta = elapsed / max(1, done) * max(0, target_batches - done)
            print(
                f"[INFO] {dataset_name} {Path(ckpt_path).name}: batch {done}/{target_batches}, "
                f"expressions={acc.total}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
            )
    rows = acc.results()
    elapsed = time.time() - start
    run_prefix = _ckpt_run_prefix(ckpt_path)
    for row in rows:
        row.update(
            {
                "run_id": f"{run_prefix}:b{row['beta']:g}",
                "checkpoint": str(ckpt_path),
                "checkpoint_name": Path(ckpt_path).name,
                "checkpoint_run_prefix": run_prefix,
                "dataset": dataset_name,
                "seconds": float(elapsed),
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "seed": int(seed),
                "max_batches": int(max_batches),
                "max_images": int(max_images),
            }
        )
    return rows


def _mean_metric(results: List[Dict[str, Any]], run_id: str, metric: str) -> float:
    vals = [float(r.get(metric, 0.0)) for r in results if r["run_id"] == run_id]
    return sum(vals) / max(1, len(vals))


def _write_summary(output_dir: Path, results: List[Dict[str, Any]], primary_metric: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ids: List[str] = []
    seen = set()
    for row in results:
        run_id = row["run_id"]
        if run_id not in seen:
            seen.add(run_id)
            run_ids.append(run_id)
    datasets: List[str] = []
    seen_ds = set()
    for row in results:
        ds = row["dataset"]
        if ds not in seen_ds:
            seen_ds.add(ds)
            datasets.append(ds)
    ranking = [
        {
            "rank": i + 1,
            "run_id": run_id,
            f"mean_{primary_metric}": _mean_metric(results, run_id, primary_metric),
            "mean_mean_iou_top1": _mean_metric(results, run_id, "mean_iou_top1"),
        }
        for i, run_id in enumerate(
            sorted(run_ids, key=lambda r: _mean_metric(results, r, primary_metric), reverse=True)
        )
    ]
    payload = {"primary_metric": primary_metric, "ranking": ranking, "results": results}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    by_run_ds = {(r["run_id"], r["dataset"]): r for r in results}
    lines = [
        "# RefCOCO Stage-B Evaluation",
        "",
        f"Primary metric: mean `{primary_metric}` across evaluated splits.",
        "",
        "| rank | run | mean acc50 | mean IoU@1 | " + " | ".join(f"{ds} acc50" for ds in datasets) + " |",
        "|---:|---|---:|---:|" + "|".join("---:" for _ in datasets) + "|",
    ]
    for row in ranking:
        run_id = row["run_id"]
        ds_vals = [f"{float(by_run_ds.get((run_id, ds), {}).get('acc50', 0.0)):.6f}" for ds in datasets]
        lines.append(
            f"| {row['rank']} | `{run_id}` | "
            f"{float(row[f'mean_{primary_metric}']):.6f} | "
            f"{float(row['mean_mean_iou_top1']):.6f} | "
            + " | ".join(ds_vals)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-B checkpoints on standard RefCOCO splits.")
    parser.add_argument("--config", default="config/cfg_patch_stage_b.py")
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/refcoco_stageb_eval")
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--splits", nargs="*", default=["refcoco_val", "refcocop_val", "refcocog_val"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0, help="Maximum expression rows per split; 0 means full split.")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--primary_metric", default="acc50")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.patch_only = True
    cfg.build_text_token_masks = True
    cfg.use_coco_eval = False
    cfg.batch_size = int(args.batch_size)

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
    wanted = list(args.splits or [])
    if wanted == ["all"]:
        wanted = list(split_specs)
    unknown = [name for name in wanted if name not in split_specs]
    if unknown:
        raise KeyError(f"Unknown split names: {unknown}; available={list(split_specs)}")

    datasetinfos = []
    for name in wanted:
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
        print(f"[INFO] built {name}: {count} expressions -> {jsonl_path}")
        datasetinfos.append((name, _make_datasetinfo(data_root, name, jsonl_path)))

    results: List[Dict[str, Any]] = []
    for ckpt_i, ckpt_path in enumerate(args.ckpts):
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(args.ckpts)}: {ckpt_path}")
        _set_seed(int(args.seed))
        model = _load_model(cfg, ckpt_path, device)
        for ds_i, (name, datasetinfo) in enumerate(datasetinfos):
            rows = evaluate_dataset(
                cfg=cfg,
                model=model,
                ckpt_path=ckpt_path,
                datasetinfo=datasetinfo,
                dataset_name=name,
                device=device,
                betas=list(args.betas),
                topks=list(args.topk),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(args.seed) + ds_i * 100000,
                amp=bool(args.amp),
                max_batches=int(args.max_batches),
                max_images=int(args.max_images),
                log_every=int(args.log_every),
            )
            results.extend(rows)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{_ckpt_run_prefix(ckpt_path)}__{name}.json").write_text(
                json.dumps(rows, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _write_summary(output_dir, results, str(args.primary_metric))
            for row in rows:
                print(
                    f"[RESULT] {row['run_id']} {name}: "
                    f"acc50={row['acc50']:.6f} mean_iou@1={row['mean_iou_top1']:.6f} "
                    f"recall50@5={row.get('recall50@5', 0.0):.6f}"
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_summary(output_dir, results, str(args.primary_metric))
    print(f"[INFO] wrote {output_dir / 'summary.json'}")
    print(f"[INFO] wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
