#!/usr/bin/env python3
"""Val-only BCE/L2 control derived from the sealed B32A1 trainer.

Original schedule, rank objective, head initialization and task-specific Adam
updates are retained. Only confidence supervision changes. See control_lock.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.b32a1_heads import (
    B32A1AbstentionHead,
    B32A1_HEAD_MODES,
    FACTORIZED_C_ONLY,
    ISOLATED,
    SHARED_WIDE,
    gradient_topology_report,
)
from tools.b32a1_objectives import (
    baseline_preserving_top1_rank_loss,
)
from tools.b32a1_metrics import binary_auroc, selective_localization_aurc
from tools.finecops_bce_l2_control import OBJECTIVE, confidence_objective, activity_health

CACHE_MANIFEST_SCHEMA = "pivot.b32a1.finecops_candidate_manifest/v1"
CACHE_ROW_SCHEMA = "pivot.b32a1.finecops_candidate_row/v1"
CACHE_SHARD_SCHEMA = "pivot.b32a1.finecops_candidate_shard/v1"

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
from tools.responsibility_isolation_cache import normalized_cxcywh_iou


B32A1_HEAD_MODES = (SHARED_WIDE, ISOLATED)
SMOKE_EVENTS = 0
_CACHE_MEMORY = {}
FORMAL_SEEDS = (17, 42, 73)
FORMAL_EPOCHS = 5
RANK_BATCH_SIZE = 32
CONFIDENCE_BATCH_SIZE = 32
RANK_LEARNING_RATE = 3e-5
CONFIDENCE_LEARNING_RATE = 1e-4
CLIP_NORM = 0.1
EXPECTED_TRAIN = {"positive": 83_341, "text": 80_451}
EXPECTED_VAL = {"positive": 9_426, "text": 9_029}
CHECKPOINT_SCHEMA = "arrow.bce_l2.head_checkpoint/v1"
EPOCH_COMPLETION_SCHEMA = "arrow.bce_l2.head_epoch_completion/v1"


class B32A1TrainingError(RuntimeError):
    """Raised when a formal head run violates its frozen-data contract."""


@dataclass
class D3QueueState:
    values: Tensor
    count: int = 0
    cursor: int = 0

    @classmethod
    def empty(cls, *, size: int = 512) -> "D3QueueState":
        return cls(torch.zeros(size, dtype=torch.float32), 0, 0)

    def history(self, *, min_count: int = 256, device: torch.device) -> Tensor | None:
        if self.count < min_count:
            return None
        value = (
            self.values[: self.count]
            if self.count < int(self.values.numel())
            else torch.cat((self.values[self.cursor :], self.values[: self.cursor]))
        )
        return value.detach().to(device=device)

    def append(self, values: Tensor) -> None:
        payload = values.detach().cpu().float().reshape(-1)
        if not payload.numel() or not bool(torch.isfinite(payload).all().item()):
            raise B32A1TrainingError("confidence queue payload must be nonempty and finite")
        size = int(self.values.numel())
        for value in payload:
            self.values[self.cursor] = value
            self.cursor = (self.cursor + 1) % size
            self.count = min(size, self.count + 1)

    def state_dict(self) -> dict[str, Any]:
        return {"values": self.values.clone(), "count": self.count, "cursor": self.cursor}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        values = value.get("values")
        count = value.get("count")
        cursor = value.get("cursor")
        if (
            not torch.is_tensor(values)
            or values.dtype != torch.float32
            or tuple(values.shape) != tuple(self.values.shape)
            or not isinstance(count, int)
            or not 0 <= count <= int(self.values.numel())
            or not isinstance(cursor, int)
            or not 0 <= cursor < int(self.values.numel())
        ):
            raise B32A1TrainingError("confidence queue checkpoint drifted")
        self.values.copy_(values.cpu())
        self.count = count
        self.cursor = cursor


def _stack_rows(
    rows: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.stack([row["query_features"] for row in rows]).to(device=device),
        torch.stack([row["native_score"] for row in rows]).to(device=device),
        torch.stack([row["candidate_mask"] for row in rows]).to(device=device),
    )


def _rank_loss(
    module: B32A1AbstentionHead,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    features, native, mask = _stack_rows(rows, device)
    output = module(features, native, mask)
    candidate_iou = torch.stack(
        [
            normalized_cxcywh_iou(row["boxes"], row["gt_boxes"]).amax(dim=1)
            for row in rows
        ]
    ).to(device=device)
    no_eligible_positive = ~(mask & (candidate_iou >= 0.5)).any(dim=1)
    candidate_iou = torch.where(
        no_eligible_positive[:, None] & (~mask),
        torch.full_like(candidate_iou, -1.0),
        candidate_iou,
    )
    result = baseline_preserving_top1_rank_loss(
        output["rank_score"],
        output["native_score"],
        output["rank_residual"],
        candidate_iou,
        iou_threshold=0.5,
        fix_margin=0.05,
        preserve_margin=0.02,
        temperature=0.1,
        residual_weight=1e-3,
    )
    if not bool(torch.isfinite(result.loss).item()):
        raise B32A1TrainingError("rank objective became non-finite")
    return result.loss, {
        "loss": float(result.loss.detach()),
        "fix_loss": float(result.fix_loss.detach()),
        "preserve_loss": float(result.preserve_loss.detach()),
        "residual_loss": float(result.residual_loss.detach()),
        "base_correct": float(result.base_correct.detach()),
        "adapted_correct": float(result.adapted_correct.detach()),
        "wrong_fixed": float(result.wrong_fixed.detach()),
        "correct_regressed": float(result.correct_regressed.detach()),
        "valid_rows": float(result.valid_rows.detach()),
        "rows_no_positive": float(result.rows_no_positive.detach()),
    }


def _confidence_loss(module, pairs, queue, *, device):
    # No history read or append: the legacy queue shell remains empty for resume ABI.
    del queue
    positive = module(*_stack_rows([p["positive"] for p in pairs], device))
    negative = module(*_stack_rows([p["negative"] for p in pairs], device))
    pos = positive["confidence_score"].masked_fill(~positive["candidate_mask"], -torch.inf).max(1).values
    neg = negative["confidence_score"].masked_fill(~negative["candidate_mask"], -torch.inf).max(1).values
    loss, metrics = confidence_objective(pos, neg)
    return loss, metrics, None


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise B32A1TrainingError(f"stale temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise B32A1TrainingError(f"stale temporary exists: {temporary}")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise B32A1TrainingError(f"torch payload must be a mapping: {path}")
    return value


def _load_manifest(path: Path, *, split: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise B32A1TrainingError(f"could not load cache manifest {path}: {exc}") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != CACHE_MANIFEST_SCHEMA
        or value.get("status") != "complete"
        or value.get("split") != split
        or value.get("formal") is not True
    ):
        raise B32A1TrainingError(f"{split} cache manifest is not formal and complete")
    return value


def load_cache(path: str | Path, *, split: str) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Verify every shard and load one complete frozen cache into host memory."""
    manifest_path = Path(path).resolve(strict=True)
    if split not in ("train", "val"):
        raise B32A1TrainingError("BCE/L2 development never loads Test")
    key = (str(manifest_path), split)
    sha = file_sha256(manifest_path)
    if key in _CACHE_MEMORY:
        old_sha, old_rows, old_manifest = _CACHE_MEMORY[key]
        if sha != old_sha:
            raise B32A1TrainingError("cached manifest drifted")
        return old_rows, old_manifest
    manifest = _load_manifest(manifest_path, split=split)
    rows: list[Mapping[str, Any]] = []
    for expected_index, record in enumerate(manifest.get("shards", [])):
        if not isinstance(record, Mapping):
            raise B32A1TrainingError("cache shard record must be an object")
        shard_path = Path(str(record.get("path", "")))
        if not shard_path.is_file() or file_sha256(shard_path) != record.get("sha256"):
            raise B32A1TrainingError(f"cache shard binding failed: {shard_path}")
        shard = _torch_load(shard_path)
        if (
            shard.get("schema") != CACHE_SHARD_SCHEMA
            or shard.get("split") != split
            or int(shard.get("start", -1)) != len(rows)
        ):
            raise B32A1TrainingError(f"cache shard ordering drifted: {shard_path}")
        shard_rows = shard.get("rows")
        if not isinstance(shard_rows, (list, tuple)) or len(shard_rows) != record.get("rows"):
            raise B32A1TrainingError(f"cache shard row count drifted: {shard_path}")
        for row in shard_rows:
            if row.get("schema") != CACHE_ROW_SCHEMA:
                raise B32A1TrainingError("cache row schema drifted")
            rows.append(row)
        if expected_index % 100 == 0:
            print(f"[B32A1-HEAD] loaded {split} shard {expected_index + 1}", flush=True)
    if len(rows) != int(manifest.get("records", -1)):
        raise B32A1TrainingError(f"{split} cache record count drifted")
    _CACHE_MEMORY[key] = (sha, rows, manifest)
    return rows, manifest


def _validate_population(
    train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Mapping[str, Any]]]]:
    def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            kind: sum(row["kind"] == kind for row in rows) for kind in ("positive", "text")
        }

    if counts(train_rows) != EXPECTED_TRAIN or counts(val_rows) != EXPECTED_VAL:
        raise B32A1TrainingError(
            f"FineCops head population drifted: train={counts(train_rows)}, val={counts(val_rows)}"
        )
    if any(row["kind"] not in {"positive", "text"} for row in (*train_rows, *val_rows)):
        raise B32A1TrainingError("train/validation cache unexpectedly contains image negatives")
    positive = {
        int(row["annotation_id"]): row for row in train_rows if row["kind"] == "positive"
    }
    if len(positive) != EXPECTED_TRAIN["positive"]:
        raise B32A1TrainingError("train positive IDs are not unique")
    pairs = []
    for row in train_rows:
        if row["kind"] != "text":
            continue
        parent_id = int(row["parent_positive_id"])
        if parent_id not in positive:
            raise B32A1TrainingError("train negative lost its positive pair")
        if row["cluster_image_id"] != positive[parent_id]["cluster_image_id"]:
            raise B32A1TrainingError("train pair crossed image clusters")
        pairs.append({"positive": positive[parent_id], "negative": row})
    return list(positive.values()), pairs


def _epoch_schedule(
    *, seed: int, epoch: int, rank_count: int, confidence_count: int
) -> tuple[list[tuple[str, np.ndarray]], Mapping[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(seed * 10_000 + epoch))
    rank_order = rng.permutation(rank_count)
    confidence_order = rng.permutation(confidence_count)
    rank_batches = [
        rank_order[start : start + RANK_BATCH_SIZE]
        for start in range(0, rank_count, RANK_BATCH_SIZE)
    ]
    confidence_batches = [
        confidence_order[start : start + CONFIDENCE_BATCH_SIZE]
        for start in range(0, confidence_count, CONFIDENCE_BATCH_SIZE)
    ]
    events: list[tuple[str, np.ndarray]] = []
    for index in range(max(len(rank_batches), len(confidence_batches))):
        if index < len(rank_batches):
            events.append(("rank", rank_batches[index]))
        if index < len(confidence_batches):
            events.append(("confidence", confidence_batches[index]))
    receipt = {
        "epoch": epoch,
        "generator": "numpy.PCG64(seed*10000+epoch)",
        "rank_permutation_sha256": hashlib.sha256(rank_order.tobytes()).hexdigest(),
        "confidence_permutation_sha256": hashlib.sha256(confidence_order.tobytes()).hexdigest(),
        "rank_batches": len(rank_batches),
        "confidence_batches": len(confidence_batches),
        "events": len(events),
    }
    return events, receipt


def _unique_parameters(values: Sequence[torch.nn.Parameter]) -> tuple[torch.nn.Parameter, ...]:
    result = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return tuple(result)


@contextmanager
def _deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _validation_metrics(
    module: B32A1AbstentionHead,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> Mapping[str, Any]:
    records: list[Mapping[str, Any]] = []
    module.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            features = torch.stack([row["query_features"] for row in selected]).to(device)
            native = torch.stack([row["native_score"] for row in selected]).to(device)
            mask = torch.stack([row["candidate_mask"] for row in selected]).to(device)
            output = module(features, native, mask)
            top = output["rank_score"].argmax(dim=1).detach().cpu()
            confidence = output["confidence_score"].masked_fill(
                ~output["candidate_mask"], -torch.inf
            ).max(dim=1).values.detach().cpu()
            for offset, row in enumerate(selected):
                value: dict[str, Any] = {
                    "annotation_id": int(row["annotation_id"]),
                    "kind": row["kind"],
                    "parent_positive_id": int(row["parent_positive_id"]),
                    "cluster_image_id": row["cluster_image_id"],
                    "level": int(row["level"]),
                    "confidence": float(confidence[offset]),
                }
                if row["kind"] == "positive":
                    iou = normalized_cxcywh_iou(row["boxes"], row["gt_boxes"])
                    value["top1_iou"] = float(iou[int(top[offset])].amax())
                    native_top = int(row["native_score"].masked_fill(~row["candidate_mask"], -torch.inf).argmax())
                    value["native_top1_iou"] = float(iou[native_top].amax())
                records.append(value)
    positives = {
        int(row["annotation_id"]): row for row in records if row["kind"] == "positive"
    }
    by_level = {
        level: [
            float(row["top1_iou"]) >= 0.5
            for row in positives.values()
            if int(row["level"]) == level
        ]
        for level in (1, 2, 3)
    }
    negatives = [row for row in records if row["kind"] == "text"]
    wins = [
        float(positives[int(row["parent_positive_id"])]["top1_iou"]) >= 0.5
        and float(positives[int(row["parent_positive_id"])]["confidence"])
        > float(row["confidence"])
        for row in negatives
    ]
    p_by_level = {str(level): float(np.mean(values)) for level, values in by_level.items()}
    level1_scores = [
        float(row["confidence"]) for row in positives.values() if int(row["level"]) == 1
    ]
    return {
        "activity_health": activity_health(module, rows, device=device, rank_loss_fn=_rank_loss),
        "native_p_at_1_micro": float(np.mean([r["native_top1_iou"] >= .5 for r in positives.values()])),
        "correct_to_wrong": sum(r["native_top1_iou"] >= .5 and r["top1_iou"] < .5 for r in positives.values()),
        "wrong_to_correct": sum(r["native_top1_iou"] < .5 and r["top1_iou"] >= .5 for r in positives.values()),
        "text_auroc_scope": "L1 positives versus text negatives",
        "text_auroc_all_positive": binary_auroc(
            [r["confidence"] for r in positives.values()], [r["confidence"] for r in negatives]),
        "correctness_auroc": binary_auroc(
            [r["confidence"] for r in positives.values() if r["top1_iou"] >= .5],
            [r["confidence"] for r in positives.values() if r["top1_iou"] < .5]),
        "positive": len(positives),
        "text_negative": len(negatives),
        "p_at_1_by_level": p_by_level,
        "p_at_1_macro": float(np.mean(list(p_by_level.values()))),
        "p_at_1_micro": float(
            np.mean([value for values in by_level.values() for value in values])
        ),
        "text_recall_at_1": float(np.mean(wins)),
        "text_auroc": binary_auroc(
            level1_scores, [float(row["confidence"]) for row in negatives]
        ),
        "aurc": selective_localization_aurc(
            [float(row["confidence"]) for row in positives.values()],
            [float(row["top1_iou"]) < 0.5 for row in positives.values()],
        ),
    }


def _load_resume(
    output: Path,
    *,
    seed: int,
    models: Mapping[str, B32A1AbstentionHead],
    rank_optimizers: Mapping[str, torch.optim.Optimizer | None],
    confidence_optimizers: Mapping[str, torch.optim.Optimizer],
    queues: Mapping[str, D3QueueState],
    device: torch.device,
) -> tuple[int, list[Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    del device  # Optimizer.load_state_dict follows the devices of its parameters.
    marker_paths = sorted(output.glob("epoch_completion_*.json"))
    epochs = []
    markers: dict[int, Mapping[str, Any]] = {}
    for path in marker_paths:
        suffix = path.stem.removeprefix("epoch_completion_")
        if not suffix.isdigit():
            raise B32A1TrainingError(f"invalid epoch completion marker: {path}")
        epoch = int(suffix)
        marker = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(marker, Mapping)
            or marker.get("schema") != EPOCH_COMPLETION_SCHEMA
            or marker.get("experiment_id") != "ARROW_BCE_L2_DEV"
            or marker.get("seed") != seed
            or marker.get("epoch") != epoch
            or set(marker.get("arms", {})) != set(B32A1_HEAD_MODES)
        ):
            raise B32A1TrainingError(f"resume epoch marker drifted: {path}")
        epochs.append(epoch)
        markers[epoch] = marker
    if epochs and epochs != list(range(1, max(epochs) + 1)):
        raise B32A1TrainingError("completed head epochs are not consecutive from epoch 1")
    if epochs and max(epochs) > FORMAL_EPOCHS:
        raise B32A1TrainingError("completed head epoch exceeds the fixed endpoint")

    completed_epoch = max(epochs, default=0)
    _archive_uncommitted_artifacts(output, completed_epoch=completed_epoch)
    if completed_epoch == 0:
        return 0, [], {mode: [] for mode in B32A1_HEAD_MODES}

    schedule_receipts: list[Mapping[str, Any]] = []
    validation_history: dict[str, list[Mapping[str, Any]]] = {
        mode: [] for mode in B32A1_HEAD_MODES
    }
    for epoch in epochs:
        marker = markers[epoch]
        schedule = marker.get("schedule")
        if not isinstance(schedule, Mapping) or schedule.get("epoch") != epoch:
            raise B32A1TrainingError(f"resume schedule drifted at epoch {epoch}")
        schedule_receipts.append(schedule)
        for mode in B32A1_HEAD_MODES:
            records = marker["arms"][mode]
            if not isinstance(records, Mapping):
                raise B32A1TrainingError(f"resume arm record drifted for {mode}/epoch{epoch}")
            checkpoint_path = output / mode / f"checkpoint_epoch{epoch}.pt"
            validation_path = output / mode / f"validation_epoch{epoch}.json"
            for label, artifact_path in (
                ("checkpoint", checkpoint_path),
                ("validation", validation_path),
            ):
                artifact = records.get(label)
                if (
                    not isinstance(artifact, Mapping)
                    or Path(str(artifact.get("path", ""))) != artifact_path
                    or not artifact_path.is_file()
                    or file_sha256(artifact_path) != artifact.get("sha256")
                ):
                    raise B32A1TrainingError(
                        f"resume {label} binding failed for {mode}/epoch{epoch}"
                    )
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if not isinstance(validation, Mapping) or validation.get("epoch") != epoch:
                raise B32A1TrainingError(
                    f"resume validation payload drifted for {mode}/epoch{epoch}"
                )
            validation_history[mode].append(validation)

    epoch = completed_epoch
    for mode in B32A1_HEAD_MODES:
        path = output / mode / f"checkpoint_epoch{epoch}.pt"
        payload = _torch_load(path)
        if (
            payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("confidence_objective") != OBJECTIVE
            or payload.get("mode") != mode
            or payload.get("seed") != seed
            or payload.get("epoch") != epoch
        ):
            raise B32A1TrainingError(f"resume checkpoint drifted: {path}")
        if payload.get("schedule") != markers[epoch]["schedule"]:
            raise B32A1TrainingError(f"resume checkpoint schedule drifted: {path}")
        if payload.get("validation") != validation_history[mode][-1]:
            raise B32A1TrainingError(f"resume checkpoint validation drifted: {path}")
        if payload.get("model_state_sha256") != _tensor_state_sha256(
            payload["model_state_dict"]
        ):
            raise B32A1TrainingError(f"resume model state hash drifted: {path}")
        models[mode].load_state_dict(payload["model_state_dict"], strict=True)
        rank_state = payload["rank_optimizer_state_dict"]
        if rank_optimizers[mode] is None:
            if rank_state is not None:
                raise B32A1TrainingError("C-only resume unexpectedly has rank optimizer state")
        else:
            if rank_state is None:
                raise B32A1TrainingError("rank optimizer state is missing")
            rank_optimizers[mode].load_state_dict(rank_state)
        confidence_optimizers[mode].load_state_dict(payload["confidence_optimizer_state_dict"])
        queues[mode].load_state_dict(payload["confidence_queue_state_dict"])
    return epoch, schedule_receipts, validation_history


def _artifact_epoch(path: Path) -> int | None:
    name = path.name
    for prefix in ("checkpoint_epoch", "validation_epoch", "epoch_completion_"):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :].split(".", 1)[0]
        return int(suffix) if suffix.isdigit() else None
    return None


def _archive_uncommitted_artifacts(output: Path, *, completed_epoch: int) -> None:
    candidates: list[Path] = []
    for mode in B32A1_HEAD_MODES:
        if not (output / mode).is_dir():
            continue
        for path in (output / mode).glob("*"):
            epoch = _artifact_epoch(path)
            if epoch is not None and epoch > completed_epoch:
                candidates.append(path)
    for path in output.glob("epoch_completion_*.json.tmp"):
        epoch = _artifact_epoch(path)
        if epoch is not None and epoch > completed_epoch:
            candidates.append(path)
    if not candidates:
        return
    recovery_root = output / "resume_recovery"
    index = 1
    while (recovery_root / f"recovery_{index:03d}").exists():
        index += 1
    archive = recovery_root / f"recovery_{index:03d}"
    records = []
    for source in sorted(set(candidates)):
        relative = source.relative_to(output)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        sha256 = file_sha256(source) if source.is_file() else None
        os.replace(source, destination)
        records.append(
            {"original_path": str(source), "archived_path": str(destination), "sha256": sha256}
        )
    _atomic_json(
        archive / "recovery_receipt.json",
        {
            "schema": "pivot.b32a1.head_resume_recovery/v1",
            "experiment_id": "ARROW_BCE_L2_DEV",
            "completed_epoch": completed_epoch,
            "reason": "artifacts_not_bound_by_an_atomic_epoch_completion_marker",
            "artifacts": records,
        },
    )


def train_seed(
    *,
    train_manifest_path: str | Path,
    val_manifest_path: str | Path,
    output_dir: str | Path,
    seed: int,
    device: str = "cuda",
    resume: bool = False,
) -> Mapping[str, Any]:
    if seed not in FORMAL_SEEDS:
        raise B32A1TrainingError(f"seed must be one of {FORMAL_SEEDS}")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise B32A1TrainingError("CUDA was requested but is unavailable")
    if target.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "training_receipt.json"
    if receipt_path.exists():
        raise B32A1TrainingError("formal training receipt already exists")
    train_rows, train_manifest = load_cache(train_manifest_path, split="train")
    val_rows, val_manifest = load_cache(val_manifest_path, split="val")
    train_checkpoint = train_manifest["model"]["checkpoint"]
    val_checkpoint = val_manifest["model"]["checkpoint"]
    if train_checkpoint != val_checkpoint:
        raise B32A1TrainingError("train/validation caches use different frozen trunks")
    frozen_checkpoint_path = Path(train_checkpoint["path"])
    frozen_checkpoint_sha = file_sha256(frozen_checkpoint_path)
    if frozen_checkpoint_sha != train_checkpoint["sha256"]:
        raise B32A1TrainingError("frozen trunk changed before head training")
    rank_rows, pairs = _validate_population(train_rows, val_rows)

    models: dict[str, B32A1AbstentionHead] = {}
    rank_parameters: dict[str, tuple[torch.nn.Parameter, ...]] = {}
    confidence_parameters: dict[str, tuple[torch.nn.Parameter, ...]] = {}
    rank_optimizers: dict[str, torch.optim.Optimizer | None] = {}
    confidence_optimizers: dict[str, torch.optim.Optimizer] = {}
    queues: dict[str, D3QueueState] = {}
    architectures: dict[str, Any] = {}
    for mode in B32A1_HEAD_MODES:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if target.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = B32A1AbstentionHead(mode=mode).to(target)
        models[mode] = model
        architectures[mode] = model.architecture_report().as_dict()
        rank_parameters[mode] = _unique_parameters(model.task_parameters("rank"))
        confidence_parameters[mode] = _unique_parameters(
            model.task_parameters("confidence")
        )
        rank_optimizers[mode] = (
            None
            if not rank_parameters[mode]
            else torch.optim.AdamW(
                rank_parameters[mode], lr=RANK_LEARNING_RATE,
                weight_decay=0.0, foreach=False
            )
        )
        confidence_optimizers[mode] = torch.optim.AdamW(
            confidence_parameters[mode],
            lr=CONFIDENCE_LEARNING_RATE,
            weight_decay=0.0,
            foreach=False,
        )
        queues[mode] = D3QueueState.empty(size=512)

    start_epoch = 0
    schedule_receipts: list[Mapping[str, Any]] = []
    validation_history: dict[str, list[Mapping[str, Any]]] = {
        mode: [] for mode in B32A1_HEAD_MODES
    }
    if resume:
        start_epoch, schedule_receipts, validation_history = _load_resume(
            output,
            seed=seed,
            models=models,
            rank_optimizers=rank_optimizers,
            confidence_optimizers=confidence_optimizers,
            queues=queues,
            device=target,
        )
    elif any((output / mode).exists() for mode in B32A1_HEAD_MODES):
        raise B32A1TrainingError("head output exists; pass --resume for an epoch-boundary resume")

    with _deterministic_algorithms():
        for epoch in range(start_epoch + 1, FORMAL_EPOCHS + 1):
            events, schedule_receipt = _epoch_schedule(
                seed=seed,
                epoch=epoch,
                rank_count=len(rank_rows),
                confidence_count=len(pairs),
            )
            if SMOKE_EVENTS:
                events = events[:SMOKE_EVENTS]
                schedule_receipt = {**schedule_receipt, "smoke_prefix_events": SMOKE_EVENTS}
            schedule_receipts.append(schedule_receipt)
            task_updates = {"rank": 0, "confidence": 0}
            loss_sums = {
                mode: {"rank": 0.0, "confidence": 0.0} for mode in B32A1_HEAD_MODES
            }
            epoch_artifacts: dict[str, Any] = {}
            for mode in B32A1_HEAD_MODES:
                models[mode].train()
            for event_index, (task, indices) in enumerate(events, 1):
                payload = (
                    [rank_rows[int(index)] for index in indices]
                    if task == "rank"
                    else [pairs[int(index)] for index in indices]
                )
                for mode in B32A1_HEAD_MODES:
                    model = models[mode]
                    model.zero_grad(set_to_none=True)
                    if task == "rank":
                        if rank_optimizers[mode] is None:
                            if mode != FACTORIZED_C_ONLY or rank_parameters[mode]:
                                raise B32A1TrainingError("rank no-op ownership drifted")
                            continue
                        loss, metrics = _rank_loss(model, payload, device=target)
                        optimizer = rank_optimizers[mode]
                        active = rank_parameters[mode]
                        queue_payload = None
                    else:
                        loss, metrics, queue_payload = _confidence_loss(
                            model, payload, queues[mode], device=target
                        )
                        optimizer = confidence_optimizers[mode]
                        active = confidence_parameters[mode]
                    assert optimizer is not None
                    loss.backward()
                    gradients = [parameter.grad for parameter in active]
                    if any(value is None for value in gradients) or not all(
                        bool(torch.isfinite(value).all().item()) for value in gradients
                    ):
                        raise B32A1TrainingError(
                            f"{mode}/{task} lost a finite gradient at epoch {epoch} event {event_index}"
                        )
                    norm = torch.nn.utils.clip_grad_norm_(active, CLIP_NORM)
                    if not bool(torch.isfinite(norm).item()):
                        raise B32A1TrainingError("head gradient norm became non-finite")
                    optimizer.step()
                    if queue_payload is not None:
                        queues[mode].append(queue_payload)
                    loss_sums[mode][task] += float(metrics["loss"])
                task_updates[task] += 1
                if event_index % 250 == 0:
                    print(
                        f"[B32A1-HEAD] seed={seed} epoch={epoch} event={event_index}/{len(events)}",
                        flush=True,
                    )

            for mode in B32A1_HEAD_MODES:
                validation = dict(
                    _validation_metrics(models[mode], val_rows, device=target)
                )
                validation["epoch"] = epoch
                validation_history[mode].append(validation)
                model = models[mode]
                # Probe graph connectivity with nonzero outputs after the epoch.
                sample = rank_rows[:2]
                features = torch.stack([row["query_features"] for row in sample]).to(target)
                native = torch.stack([row["native_score"] for row in sample]).to(target)
                mask = torch.stack([row["candidate_mask"] for row in sample]).to(target)
                probe_output = model(features, native, mask)
                topology = gradient_topology_report(
                    model,
                    probe_output["rank_score"][mask].square().mean(),
                    probe_output["confidence_score"][mask].square().mean(),
                )
                model.zero_grad(set_to_none=True)
                checkpoint_path = output / mode / f"checkpoint_epoch{epoch}.pt"
                if checkpoint_path.exists():
                    raise B32A1TrainingError(f"checkpoint already exists: {checkpoint_path}")
                checkpoint = {
                    "schema": CHECKPOINT_SCHEMA,
                    "experiment_id": "ARROW_BCE_L2_DEV",
                    "mode": mode,
                    "seed": seed,
                    "epoch": epoch,
                    "fixed_endpoint_epoch": FORMAL_EPOCHS,
                    "architecture": architectures[mode],
                    "confidence_objective": OBJECTIVE,
                    "run_role": "smoke_only" if SMOKE_EVENTS else "fixed_epoch5_val_only",
                    "training": {
                        "rank_learning_rate": RANK_LEARNING_RATE,
                        "confidence_learning_rate": CONFIDENCE_LEARNING_RATE,
                        "weight_decay": 0.0,
                        "clip_norm": CLIP_NORM,
                        "rank_batch_size": RANK_BATCH_SIZE,
                        "confidence_batch_size": CONFIDENCE_BATCH_SIZE,
                        "task_specific_adamw_states": True,
                        "updates_this_epoch": task_updates,
                        "mean_losses": {
                            task: loss_sums[mode][task] / task_updates[task]
                            for task in ("rank", "confidence")
                        },
                    },
                    "schedule": schedule_receipt,
                    "validation": validation,
                    "gradient_topology": topology,
                    "frozen_trunk": train_checkpoint,
                    "train_cache_manifest_sha256": file_sha256(Path(train_manifest_path)),
                    "val_cache_manifest_sha256": file_sha256(Path(val_manifest_path)),
                    "model_state_dict": model.state_dict(),
                    "rank_optimizer_state_dict": (
                        None if rank_optimizers[mode] is None
                        else rank_optimizers[mode].state_dict()
                    ),
                    "confidence_optimizer_state_dict": confidence_optimizers[mode].state_dict(),
                    "confidence_queue_state_dict": queues[mode].state_dict(),
                }
                checkpoint["model_state_sha256"] = _tensor_state_sha256(
                    checkpoint["model_state_dict"]
                )
                _atomic_torch(checkpoint_path, checkpoint)
                validation_path = output / mode / f"validation_epoch{epoch}.json"
                _atomic_json(validation_path, validation)
                epoch_artifacts[mode] = {
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": file_sha256(checkpoint_path),
                    },
                    "validation": {
                        "path": str(validation_path),
                        "sha256": file_sha256(validation_path),
                    },
                }
            _atomic_json(
                output / f"epoch_completion_{epoch}.json",
                {
                    "schema": EPOCH_COMPLETION_SCHEMA,
                    "experiment_id": "ARROW_BCE_L2_DEV",
                    "seed": seed,
                    "epoch": epoch,
                    "schedule": schedule_receipt,
                    "arms": epoch_artifacts,
                },
            )

    if file_sha256(frozen_checkpoint_path) != frozen_checkpoint_sha:
        raise B32A1TrainingError("frozen positive trunk changed during head training")
    final_checkpoints = {}
    for mode in B32A1_HEAD_MODES:
        path = output / mode / f"checkpoint_epoch{FORMAL_EPOCHS}.pt"
        payload = _torch_load(path)
        final_checkpoints[mode] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "model_state_sha256": payload["model_state_sha256"],
            "selection": "fixed_epoch_5_no_per_arm_checkpoint_selection",
        }
    receipt = {
        "schema": "arrow.bce_l2.training_receipt/v1",
        "experiment_id": "ARROW_BCE_L2_DEV",
        "status": "complete",
        "confidence_objective": OBJECTIVE,
        "run_role": "smoke_only" if SMOKE_EVENTS else "fixed_epoch5_val_only",
        "test_accessed": False,
        "seed": seed,
        "arms": list(B32A1_HEAD_MODES),
        "architectures": architectures,
        "frozen_trunk": train_checkpoint,
        "frozen_trunk_unchanged": True,
        "train_cache": {
            "path": str(Path(train_manifest_path).resolve()),
            "sha256": file_sha256(Path(train_manifest_path)),
        },
        "val_cache": {
            "path": str(Path(val_manifest_path).resolve()),
            "sha256": file_sha256(Path(val_manifest_path)),
        },
        "schedule": {
            "epochs": FORMAL_EPOCHS,
            "rank_examples_per_epoch": len(rank_rows),
            "confidence_pairs_per_epoch": len(pairs),
            "interleave": "rank_then_confidence_for_each_batch_index",
            "by_epoch": schedule_receipts,
            "semantic_sha256": _canonical_sha256(schedule_receipts),
        },
        "validation_history": validation_history,
        "final_checkpoints": final_checkpoints,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _bound(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def run_sequence(parent_root: Path, output: Path, device: str, resume: bool):
    """One cache allocation; smoke then all three fixed endpoint seeds."""
    global FORMAL_EPOCHS, SMOKE_EVENTS
    parent_root = parent_root.resolve(strict=True)
    output = output.resolve()
    if output == parent_root or parent_root in output.parents:
        raise B32A1TrainingError("new control must use a separate experiment root")
    train_path = parent_root / "formal_cache/train/manifest.json"
    val_path = parent_root / "formal_cache/val/manifest.json"
    pinned = {
        train_path: "23c1e61a33659fddbc153df1f7a13650d110df8c3caa3a17a0f5c2ab238120f0",
        val_path: "c3d30772ed048730b228a35e217e128c17bc3935e2bccb274ec69cbd4ddec085",
    }
    for path, sha in pinned.items():
        if file_sha256(path) != sha:
            raise B32A1TrainingError(f"parent cache binding drift: {path}")
    code_names = ("train_finecops_bce_l2_heads.py", "finecops_bce_l2_control.py",
                  "b32a1_heads.py", "b32a1_objectives.py", "b32a1_metrics.py",
                  "mmgdino_e5_ownership.py", "responsibility_isolation_cache.py")
    protocol = {
        "schema": "arrow.bce_l2.development_lock/v1",
        "status": "locked_before_new_training",
        "role": "posthoc_motivated_val_only_objective_control",
        "confidence_objective": OBJECTIVE,
        "arms": list(B32A1_HEAD_MODES), "seeds": [42, 17, 73],
        "epochs": 5, "rank_updates": 13025, "confidence_updates": 12575,
        "rank_lr": RANK_LEARNING_RATE, "confidence_lr": CONFIDENCE_LEARNING_RATE,
        "weight_decay": 0.0, "clip": CLIP_NORM,
        "rank_batch": RANK_BATCH_SIZE, "confidence_pairs_batch": CONFIDENCE_BATCH_SIZE,
        "smoke": {"seed": 42, "prefix_events": 200, "formal_restarts_from_initialization": True},
        "health_rule": "conf_abs<100,hidden_mean<10000; saturation<.99,span>1e-6,rank_grad>1e-10",
        "health_policy": "smoke numerical/activity gate; formal seeds all complete unless nonfinite/runtime error; endpoint failures reported without selection",
        "code": {name: _bound(REPO_ROOT / "tools" / name) for name in code_names},
        "cache_manifests": {"train": _bound(train_path), "val": _bound(val_path)},
        "parent_test_state": {name: _bound(parent_root / name) for name in
                              ("test_consumed.json", "test_completion_receipt.json")},
        "test_access": "prohibited; cached train/val only",
        "initialization": "same head constructors and per-arm manual_seed as original B32A1; no learned head checkpoint loaded",
        "cache_execution": "single process, one memoized train/val allocation, sequential seeds",
        "precision": "FP32 deterministic, float16 frozen cache storage; no AMP",
    }
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "control_lock.json"
    if lock_path.exists():
        if not resume or json.loads(lock_path.read_text()) != protocol:
            raise B32A1TrainingError("control lock drift or --resume missing")
    else:
        _atomic_json(lock_path, protocol)
    def checked_receipt(path, expected_seed, expected_role):
        value = json.loads(path.read_text())
        if (value.get("status") != "complete" or value.get("seed") != expected_seed
                or value.get("confidence_objective") != OBJECTIVE
                or value.get("run_role") != expected_role
                or value.get("train_cache") != _bound(train_path)
                or value.get("val_cache") != _bound(val_path)):
            raise B32A1TrainingError("completed control receipt drift")
        for record in value["final_checkpoints"].values():
            if file_sha256(Path(record["path"])) != record["sha256"]:
                raise B32A1TrainingError("completed control checkpoint drift")
        return value
    smoke_path = output / "smoke_seed42/training_receipt.json"
    FORMAL_EPOCHS, SMOKE_EVENTS = 1, 200
    print("[BCE-L2] seed42 smoke: 200 prefix events, no Test", flush=True)
    if smoke_path.exists():
        smoke = checked_receipt(smoke_path, 42, "smoke_only")
    else:
        smoke = train_seed(train_manifest_path=train_path, val_manifest_path=val_path,
                           output_dir=smoke_path.parent, seed=42, device=device, resume=resume)
    smoke_health = {mode: smoke["validation_history"][mode][-1]["activity_health"] for mode in B32A1_HEAD_MODES}
    if not all(value["numerical_health"] and value["rank_active"] for value in smoke_health.values()):
        _atomic_json(output / "smoke_health_failure.json", {"health": smoke_health, "formal_started": False})
        raise B32A1TrainingError("smoke activity/numerical gate failed; preserved artifacts, no formal launch")
    print("[BCE-L2] smoke healthy; starting all fixed five-epoch seeds", flush=True)
    FORMAL_EPOCHS, SMOKE_EVENTS = 5, 0
    receipts = {}
    for seed in (42, 17, 73):
        path = output / f"formal/seed{seed}/training_receipt.json"
        if path.exists():
            value = checked_receipt(path, seed, "fixed_epoch5_val_only")
        else:
            print(f"[BCE-L2] starting seed={seed}", flush=True)
            value = train_seed(train_manifest_path=train_path, val_manifest_path=val_path,
                               output_dir=path.parent, seed=seed, device=device, resume=resume)
        receipts[str(seed)] = _bound(path)
        print(f"[BCE-L2] completed seed={seed}; health=" + json.dumps({
            mode: value["validation_history"][mode][-1]["activity_health"] for mode in B32A1_HEAD_MODES
        }), flush=True)
    for record in protocol["parent_test_state"].values():
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise B32A1TrainingError("parent Test state changed")
    final = {"schema": "arrow.bce_l2.sequence_receipt/v1", "status": "complete",
             "control_lock": _bound(lock_path), "smoke": _bound(smoke_path),
             "seeds": receipts, "new_test_forwards": 0, "formal_trajectories": 6}
    _atomic_json(output / "sequence_receipt.json", final)
    print(json.dumps(final, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_sequence(args.parent_root, args.output_dir, args.device, args.resume)


if __name__ == "__main__":
    main()
