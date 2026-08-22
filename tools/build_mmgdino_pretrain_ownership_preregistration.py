#!/usr/bin/env python3
"""Preregister the frozen MM-GDINO-T pretrained ownership replay."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from tools.mmgdino_e5_ownership import MMGDinoE5ResponsibilityOwners
from tools.mmgdino_pretrain_ownership import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    B58_REFERENCE,
    CHECKPOINT,
    CHECKPOINT_META,
    CHECKPOINT_SHA256,
    E5_REFERENCE,
    EVAL_CONFIG,
    EVAL_CONFIG_SHA256,
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    IMAGE_ROOT,
    MMDET_COMMIT,
    MMDET_ROOT,
    OWNERS,
    PREREGISTRATION,
    REC_NONINFERIORITY_MARGIN,
    REF_INPUTS,
    ROOT,
    SCHEDULE_RECEIPT,
    TEST5_SURFACES,
    TESTAB_SURFACES,
    TRUNK_ID,
    eval_cache_path,
    owner_output_dir,
    training_cache_path,
)
from tools.responsibility_isolation_cache import file_sha256


SCHEMA = "arrow.mmgdino_pretrain_ownership.preregistration/v1"
EXPECTED_EFFECTIVE_SCHEMA = (
    "67484843f8942dcf84c06c1b204c9b69e31cc540ec7afe93ced4e872d8c8a596"
)
EXPECTED_FULL_SCHEMA = (
    "911caac7f7306330410fdb03bb308574f7e0f8e260c3ff1b5ebc0010f51a6b87"
)
KNOWN_NONPERSISTENT_BUFFER = (
    "language_model.language_backbone.body.model.embeddings.position_ids"
)


class PreregistrationError(RuntimeError):
    pass


def _record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    result = {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if rows is not None:
        with resolved.open("r", encoding="utf-8") as handle:
            actual = sum(1 for _ in handle)
        if actual != rows:
            raise PreregistrationError(f"row count drift: {resolved}")
        result["rows"] = actual
    return result


def _state_signature(state: Mapping[str, torch.Tensor]) -> str:
    rows = [
        [name, str(state[name].dtype), list(state[name].shape)]
        for name in sorted(state)
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_audit() -> dict[str, Any]:
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise PreregistrationError("pretrained checkpoint SHA drifted")
    try:
        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", mmap=True, weights_only=False
        )
    except (TypeError, RuntimeError):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {})
    for key, expected in CHECKPOINT_META.items():
        if meta.get(key) != expected:
            raise PreregistrationError(f"checkpoint metadata drifted: {key}")
    if "refcoco" in str(meta.get("experiment_name", "")).lower():
        raise PreregistrationError("checkpoint unexpectedly names RefCOCO fine-tuning")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or len(state) != 909:
        raise PreregistrationError("pretrained state-dict tensor count drifted")
    if _state_signature(state) != EXPECTED_FULL_SCHEMA:
        raise PreregistrationError("pretrained full tensor schema drifted")
    if KNOWN_NONPERSISTENT_BUFFER not in state:
        raise PreregistrationError("known BERT position-id buffer is absent")
    effective = {
        name: tensor
        for name, tensor in state.items()
        if name != KNOWN_NONPERSISTENT_BUFFER
    }
    if len(effective) != 908 or _state_signature(effective) != EXPECTED_EFFECTIVE_SCHEMA:
        raise PreregistrationError("effective runtime schema differs from e5/e6")
    if any(not bool(torch.isfinite(value).all().item()) for value in state.values()):
        raise PreregistrationError("pretrained checkpoint contains non-finite tensors")
    return {
        "checkpoint": _record(CHECKPOINT),
        "official_openmmlab_file_id": "b448804b",
        "meta": {key: meta.get(key) for key in (*CHECKPOINT_META, "seed", "time")},
        "state_dict_tensor_count": len(state),
        "state_dict_value_count": sum(int(value.numel()) for value in state.values()),
        "full_state_schema_sha256": EXPECTED_FULL_SCHEMA,
        "effective_runtime_tensor_count": len(effective),
        "effective_runtime_value_count": sum(
            int(value.numel()) for value in effective.values()
        ),
        "effective_runtime_schema_sha256": EXPECTED_EFFECTIVE_SCHEMA,
        "known_nonpersistent_buffer": KNOWN_NONPERSISTENT_BUFFER,
        "all_tensors_finite": True,
        "refcoco_task_specific_finetuning": False,
    }


def _git() -> dict[str, str]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain"), cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise PreregistrationError("preregistration requires a clean worktree")
    return {"commit": commit, "status": "clean"}


def _git_head(path: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"), check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _atomic_json(value: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build(output: Path = PREREGISTRATION) -> dict[str, Any]:
    if output.exists():
        raise PreregistrationError("preregistration already exists")
    if EXPERIMENT_ROOT.exists():
        raise PreregistrationError("experiment output root must not exist")
    if _git_head(MMDET_ROOT) != MMDET_COMMIT:
        raise PreregistrationError("MMDetection checkout drifted")
    if file_sha256(EVAL_CONFIG) != EVAL_CONFIG_SHA256:
        raise PreregistrationError("evaluation config drifted")
    if not IMAGE_ROOT.is_dir():
        raise PreregistrationError("COCO image root is absent")
    schedule_receipt = json.loads(SCHEDULE_RECEIPT.read_text(encoding="utf-8"))
    if schedule_receipt.get("status") != "complete_before_owner_training":
        raise PreregistrationError("schedule receipt drifted")
    schedule_records = {}
    for seed in FORMAL_SEEDS:
        source = schedule_receipt["outputs"][str(seed)]
        schedule_records[str(seed)] = {}
        for name in ("rank", "d3", "schedule"):
            record = _record(Path(source[name]["path"]))
            if record["sha256"] != source[name]["sha256"]:
                raise PreregistrationError(f"seed{seed} {name} bytes drifted")
            schedule_records[str(seed)][name] = record
    checkpoint = _checkpoint_audit()
    architecture = {
        owner: MMGDinoE5ResponsibilityOwners(
            ownership=owner
        ).architecture_report().as_dict()
        for owner in OWNERS
    }
    if architecture[OWNERS[0]]["trainable_parameters"] != 100362:
        raise PreregistrationError("Shared-Wide capacity drifted")
    if architecture[OWNERS[1]]["trainable_parameters"] != 100358:
        raise PreregistrationError("Isolated capacity drifted")
    code_paths = {
        "contract": ROOT / "tools/mmgdino_pretrain_ownership.py",
        "preregistration_builder": Path(__file__),
        "runner": ROOT / "tools/run_mmgdino_pretrain_ownership.py",
        "mature_runner": ROOT / "tools/run_mmgdino_e6_ownership_2x2.py",
        "aggregator": ROOT / "tools/aggregate_mmgdino_pretrain_ownership.py",
        "owners": ROOT / "tools/mmgdino_e5_ownership.py",
        "trainer": ROOT / "tools/train_mmgdino_e5_ownership.py",
        "training_extractor": ROOT / "tools/extract_mmgdino_responsibility_cache.py",
        "evaluation_extractor": ROOT / "tools/extract_mmgdino_e5_eval_cache.py",
        "cache_evaluator": ROOT / "tools/eval_mmgdino_e5_ownership_cache.py",
        "cache_contract": ROOT / "tools/responsibility_isolation_cache.py",
        "score_objectives": (
            ROOT / "models/GroundingDINO/stage_b_gdino_score_adapter.py"
        ),
    }
    payload = {
        "schema": SCHEMA,
        "status": "locked_before_any_pretrained_trunk_gpu_forward",
        "locked_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "git": _git(),
        "research_question": (
            "Without RefCOCO task-specific trunk fine-tuning, what gradient "
            "geometry and deployed REC/rejection tradeoff arise from sharing "
            "versus isolating matched rank and rejection owners?"
        ),
        "trunk": {"id": TRUNK_ID, **checkpoint},
        "matrix": {
            "owners": list(OWNERS),
            "seeds": list(FORMAL_SEEDS),
            "formal_trajectory_count": len(OWNERS) * len(FORMAL_SEEDS),
            "shared_128_excluded": True,
            "architecture": architecture,
        },
        "training": {
            "fixed_endpoint": 150,
            "rank_updates": 100,
            "confidence_updates": 50,
            "interleave": ["rank", "confidence", "rank"],
            "rank_batch_size": 32,
            "confidence_batch_size": 8,
            "rank_learning_rate": 3e-5,
            "confidence_learning_rate": 1e-4,
            "task_specific_adam_states": True,
            "weight_decay": 0.0,
            "clip_norm": 0.1,
            "deterministic_fp32": True,
            "scheduled_rows_without_eligible_positive": (
                "preserved without replacement; zero margin through valid-row mask"
            ),
        },
        "schedule_contract": {
            "receipt": _record(SCHEDULE_RECEIPT),
            "per_seed": schedule_records,
            "same_identity_batch_order_and_losses_as_e5_e6": True,
        },
        "evaluation": {
            "inputs": {
                name: _record(spec["path"], rows=spec["rows"])
                for name, spec in REF_INPUTS.items()
            },
            "test5_surfaces": list(TEST5_SURFACES),
            "testab_surfaces": list(TESTAB_SURFACES),
            "strict_surface": "strict2031",
            "same_cache_for_native_shared_and_isolated": True,
            "no_checkpoint_gap_or_threshold_selection": True,
        },
        "statistics": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_rng": "PCG64",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ref_cluster": "global image_id across Test5",
            "strict_cluster": "image_id carrying the complete pair",
            "fpr95_recomputes_each_owner_seed_positive_q05_per_replicate": True,
            "contrast": "Isolated minus Shared-Wide",
            "rec_noninferiority_margin": REC_NONINFERIORITY_MARGIN,
        },
        "references": {
            "e5": _record(E5_REFERENCE),
            "b58": _record(B58_REFERENCE),
            "reuse_only_no_forward": True,
        },
        "runtime_assets": {
            "mmdetection_root": str(MMDET_ROOT),
            "mmdetection_commit": MMDET_COMMIT,
            "evaluation_config": _record(EVAL_CONFIG),
            "image_root": str(IMAGE_ROOT),
        },
        "code": {name: _record(path) for name, path in code_paths.items()},
        "output_targets_absent": {
            "training_caches": [
                str(training_cache_path(TRUNK_ID, seed)) for seed in FORMAL_SEEDS
            ],
            "formal": [
                str(owner_output_dir(TRUNK_ID, owner, seed))
                for owner in OWNERS for seed in FORMAL_SEEDS
            ],
            "evaluation_caches": [
                str(eval_cache_path(TRUNK_ID, surface)) for surface in REF_INPUTS
            ],
        },
        "prohibitions": [
            "do not update the pretrained trunk",
            "do not add RefCOCO task-specific trunk weights",
            "do not run Shared-128",
            "do not change owner capacity, schedule, losses, seeds, or endpoint",
            "do not select a checkpoint, Gap, or threshold from these results",
            "do not stop after one seed or one surface",
        ],
    }
    for targets in payload["output_targets_absent"].values():
        if any(Path(path).exists() for path in targets):
            raise PreregistrationError("a preregistered output target exists")
    _atomic_json(payload, output)
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
