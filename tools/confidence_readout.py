"""Versioned confidence-readout intervention over immutable native predictions.

The historical factory, dense head forward and score parameterization are reused
without modification. A readout is the only new computation between dense logits
and the original target loss. Cached geometry never enters a trainable module.
"""
from __future__ import annotations

import copy
import json
import numbers
from pathlib import Path

import torch

from tools.finecops_fixed_rank_targets import make_heads
from tools.train_finecops_bce_l2_heads import (
    CACHE_MANIFEST_SCHEMA as LEGACY_CACHE_SCHEMA, file_sha256, load_cache,
)
from tools.responsibility_isolation_cache import normalized_cxcywh_iou

MMGDINO = "mmgdino_positive"
MDETR = "mdetr_r101_refcoco_ema"
LOCALIZERS = (MMGDINO, MDETR)
GLOBAL = "global_max"
SELECTED = "native_selected"
READOUTS = (GLOBAL, SELECTED)
TARGETS = ("exists", "emit")
CACHE_SCHEMA = "arrow.confidence_readout.cache_manifest/v1"
SHARD_SCHEMA = "arrow.confidence_readout.cache_shard/v1"
ROW_SCHEMA = "arrow.confidence_readout.cache_row/v1"


def training_arms(localizer):
    if localizer not in LOCALIZERS:
        raise ValueError("unknown localizer")
    readouts = (SELECTED,) if localizer == MMGDINO else READOUTS
    return tuple(f"{readout}__{target}" for readout in readouts for target in TARGETS)


def parse_arm(arm):
    parts = arm.split("__")
    if len(parts) != 2 or parts[0] not in READOUTS or parts[1] not in TARGETS:
        raise ValueError("invalid readout/target arm")
    return tuple(parts)


def make_readout_heads(seed, device, localizer):
    """Consume exactly the legacy initialization RNG; copy, never reinitialize."""
    originals = make_heads(seed, device)
    result = {arm: copy.deepcopy(originals[parse_arm(arm)[1]])
              for arm in training_arms(localizer)}
    for model in result.values():
        active = tuple(p for p in model.parameters() if p.requires_grad)
        confidence = model.task_parameters("confidence")
        if len(active) != 8 or sum(p.numel() for p in active) != 50179:
            raise ValueError("confidence capacity drift")
        if {id(p) for p in active} != {id(p) for p in confidence}:
            raise ValueError("confidence ownership drift")
    return result


def _check_dense(logits, mask, selected):
    if not torch.is_tensor(logits) or logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("dense logits must be floating B,Q")
    if mask.dtype != torch.bool or mask.shape != logits.shape or mask.device != logits.device:
        raise ValueError("invalid candidate mask")
    if not mask.numel() or not mask.any(1).all() or not torch.isfinite(logits).all():
        raise ValueError("empty candidate set or nonfinite dense logits")
    if (not torch.is_tensor(selected) or selected.dtype != torch.int64
            or selected.shape != (len(logits),) or selected.device != logits.device):
        raise ValueError("selected query must be aligned int64 B")
    if (selected < 0).any() or (selected >= logits.shape[1]).any():
        raise ValueError("selected query out of range")
    if not mask.gather(1, selected[:, None]).all():
        raise ValueError("selected query is not valid")


def reduce_dense_logits(logits, mask, selected):
    _check_dense(logits, mask, selected)
    maximum = logits.masked_fill(~mask, -torch.inf).max(1)
    return {GLOBAL: maximum.values,
            SELECTED: logits.gather(1, selected[:, None]).squeeze(1),
            "selected_query": logits.gather(1, selected[:, None]).squeeze(1),
            "confidence_winner_index": maximum.indices,
            "native_selected_index": selected,
            "confidence_logits": logits}


def readout_scores(model, features, native, mask, native_selected_index):
    output = model(features, native, mask)
    expected = native.detach().to(output["rank_score"].dtype).masked_fill(
        ~mask, torch.finfo(output["rank_score"].dtype).min)
    if not torch.equal(output["rank_score"], expected) or torch.count_nonzero(output["rank_residual"]):
        raise ValueError("immutable native rank route changed")
    return reduce_dense_logits(output["confidence_score"], mask, native_selected_index)


def _pixel_boxes(row):
    boxes = row["boxes"].detach().cpu().float()
    size = row.get("image_size")
    if not isinstance(size, (tuple, list)) or len(size) != 2 or min(size) <= 0:
        raise ValueError("MDETR requires image_size [height,width] for official tie ordering")
    height, width = map(int, size)
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w/2, cy - h/2, cx + w/2, cy + h/2), -1) * torch.tensor(
        [width, height, width, height], dtype=torch.float32)


def native_selected_index(row, localizer):
    native, mask = row["native_score"], row["candidate_mask"]
    if (native.ndim != 1 or mask.shape != native.shape or mask.dtype != torch.bool
            or not mask.any() or not torch.isfinite(native).all()):
        raise ValueError("invalid native candidate universe")
    if localizer == MMGDINO:
        index = int(native.masked_fill(~mask, -torch.inf).argmax())
    elif localizer == MDETR:
        boxes = _pixel_boxes(row)
        if boxes.shape != (len(native), 4) or not torch.isfinite(boxes).all():
            raise ValueError("invalid MDETR pixel boxes")
        valid = torch.where(mask)[0].tolist()
        # Official tuples are sorted score then pixel xyxy, descending; Python's
        # stable order retains the lowest original query for duplicate tuples.
        index = max(valid, key=lambda j: (float(native[j]), *boxes[j].tolist()))
    else:
        raise ValueError("unknown localizer")
    stored = row.get("native_selected_index")
    if stored is not None:
        if (isinstance(stored, bool) or not isinstance(stored, numbers.Integral)
                or int(stored) != index):
            raise ValueError("stored Native query differs from localizer selector")
    return index


def native_labels(rows, localizer):
    labels, indices = {}, {}
    for row in rows:
        sid = row["sample_id"]
        if sid in labels:
            raise ValueError("duplicate sample identity")
        if row["kind"] not in ("positive", "text"):
            raise ValueError("only positive and negative-text rows are allowed")
        index = native_selected_index(row, localizer)
        if row["kind"] == "positive":
            gt = row["gt_boxes"]
            if gt.ndim != 2 or gt.shape[1] != 4 or not len(gt) or not torch.isfinite(gt).all():
                raise ValueError("positive sample needs actual GT boxes")
            labels[sid] = bool(normalized_cxcywh_iou(row["boxes"], gt)[index].max() >= .5)
        else:
            labels[sid] = None
        indices[sid] = index
    return labels, indices


def load_readout_cache(path, *, split, localizer):
    """Preserve the sealed loader; support a new dynamic-Q MDETR cache ABI."""
    path = Path(path).resolve(strict=True)
    if split not in ("train", "val"):
        raise ValueError("head training never opens held-out or transfer caches")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") == LEGACY_CACHE_SCHEMA:
        if localizer != MMGDINO:
            raise ValueError("legacy MM cache cannot be relabeled as another localizer")
        return load_cache(path, split=split)
    if (manifest.get("schema") != CACHE_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("formal") is not True or manifest.get("split") != split
            or manifest.get("localizer") != localizer):
        raise ValueError("cache manifest contract drift")
    rows = []
    for entry in manifest.get("shards", []):
        shard_path = Path(entry["path"])
        if not shard_path.is_absolute():
            shard_path = path.parent / shard_path
        if file_sha256(shard_path) != entry["sha256"]:
            raise ValueError("cache shard hash drift")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if (shard.get("schema") != SHARD_SCHEMA or shard.get("split") != split
                or shard.get("start") != len(rows) or len(shard.get("rows", [])) != entry["rows"]):
            raise ValueError("cache shard ordering/count drift")
        for row in shard["rows"]:
            if row.get("schema") != ROW_SCHEMA or "native_selected_index" not in row:
                raise ValueError("cache row ABI drift")
            q = len(row["native_score"])
            if (q != (100 if localizer == MDETR else 900)
                    or row["query_features"].shape != (q, 256)
                    or row["query_features"].dtype != torch.float16
                    or row["native_score"].dtype != torch.float32
                    or row["boxes"].shape != (q, 4) or row["boxes"].dtype != torch.float32):
                raise ValueError("cache query count, dtype, or geometry shape drift")
            if row.get("split", split) != split:
                raise ValueError("row split contamination")
            rows.append(row)
    if not rows or len(rows) != manifest.get("records"):
        raise ValueError("cache manifest row count drift")
    return rows, manifest
