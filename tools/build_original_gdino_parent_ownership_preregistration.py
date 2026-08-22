#!/usr/bin/env python3
"""Preregister the pure GroundingDINO pre-Stage-B ownership replay."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_FOR_IMPORT))

from tools.mmgdino_e5_ownership import MMGDinoE5ResponsibilityOwners
from tools.original_gdino_parent_ownership import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    B58_CHECKPOINT,
    B58_CHECKPOINT_SHA256,
    B58_REFERENCE,
    CHECKPOINT,
    CHECKPOINT_SHA256,
    E5_REFERENCE,
    EVAL_CONFIG,
    EVAL_CONFIG_SHA256,
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    IMAGE_ROOT,
    MMGDINO_PRETRAIN_REFERENCE,
    OWNERS,
    PARENT_TO_B58_CHANGED_TENSORS,
    PARENT_TO_B58_UNCHANGED_TENSORS,
    PARENT_UNUSED_PATCH_TENSORS,
    PREREGISTRATION,
    PURE_TRUNK_NUMEL,
    PURE_TRUNK_SCHEMA_SHA256,
    PURE_TRUNK_TENSORS,
    REC_NONINFERIORITY_MARGIN,
    REF_INPUTS,
    RELEASE_OGC,
    RELEASE_OGC_SHA256,
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
from util.utils import clean_state_dict


SCHEMA = "arrow.original_gdino_parent_ownership.preregistration/v1"


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
        actual = sum(1 for _ in resolved.open("r", encoding="utf-8"))
        if actual != rows:
            raise PreregistrationError(f"row count drift: {resolved}")
        result["rows"] = actual
    return result


def _schema(state: Mapping[str, torch.Tensor]) -> str:
    rows = [
        [name, str(state[name].dtype), list(state[name].shape)]
        for name in sorted(state)
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise PreregistrationError(f"checkpoint lacks model state: {path}")
    return clean_state_dict(state)


def _parent_unused(name: str) -> bool:
    return (
        name == "patch_logit_scale"
        or name.startswith("patch_encoder.")
        or name.startswith("query_proj_for_patch.")
    )


def _checkpoint_audit() -> dict[str, Any]:
    identities = (
        (RELEASE_OGC, RELEASE_OGC_SHA256, "release_ogc"),
        (CHECKPOINT, CHECKPOINT_SHA256, "pre_stageb_parent"),
        (B58_CHECKPOINT, B58_CHECKPOINT_SHA256, "b58_descendant"),
    )
    for path, expected, label in identities:
        if file_sha256(path) != expected:
            raise PreregistrationError(f"{label} checkpoint SHA drifted")
    release = _load_state(RELEASE_OGC)
    parent = _load_state(CHECKPOINT)
    b58 = _load_state(B58_CHECKPOINT)
    release_effective = {
        key: value for key, value in release.items()
        if key not in {"bert.embeddings.position_ids", "label_enc.weight"}
    }
    parent_effective = {
        key: value for key, value in parent.items() if not _parent_unused(key)
    }
    parent_extra = sorted(set(parent).difference(parent_effective))
    effective = {
        "release_ogc": release_effective,
        "pre_stageb_parent": parent_effective,
        "b58_descendant": b58,
    }
    for label, state in effective.items():
        if (
            len(state) != PURE_TRUNK_TENSORS
            or sum(int(value.numel()) for value in state.values()) != PURE_TRUNK_NUMEL
            or _schema(state) != PURE_TRUNK_SCHEMA_SHA256
        ):
            raise PreregistrationError(f"{label} pure-trunk schema drifted")
        if any(
            value.is_floating_point()
            and not bool(torch.isfinite(value).all().item())
            for value in state.values()
        ):
            raise PreregistrationError(f"{label} contains non-finite tensors")
    if (
        len(parent_extra) != PARENT_UNUSED_PATCH_TENSORS
        or any(not _parent_unused(name) for name in parent_extra)
    ):
        raise PreregistrationError("parent unused patch ownership drifted")

    def difference(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]):
        same = sum(torch.equal(left[key], right[key]) for key in left)
        return {"unchanged_tensors": int(same), "changed_tensors": len(left) - int(same)}

    release_to_parent = difference(release_effective, parent_effective)
    parent_to_b58 = difference(parent_effective, b58)
    if parent_to_b58 != {
        "unchanged_tensors": PARENT_TO_B58_UNCHANGED_TENSORS,
        "changed_tensors": PARENT_TO_B58_CHANGED_TENSORS,
    }:
        raise PreregistrationError("parent-to-B58 tensor delta drifted")
    return {
        "release_ogc": _record(RELEASE_OGC),
        "pre_stageb_parent": _record(CHECKPOINT),
        "b58_descendant": _record(B58_CHECKPOINT),
        "pure_trunk_tensor_count": PURE_TRUNK_TENSORS,
        "pure_trunk_numel": PURE_TRUNK_NUMEL,
        "pure_trunk_schema_sha256": PURE_TRUNK_SCHEMA_SHA256,
        "parent_unused_patch_tensors": len(parent_extra),
        "parent_unused_patch_names_sha256": hashlib.sha256(
            json.dumps(parent_extra, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "release_to_parent": release_to_parent,
        "parent_to_b58": parent_to_b58,
        "all_effective_tensors_finite": True,
    }


def _git() -> dict[str, str]:
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    status = subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()
    if status:
        raise PreregistrationError("preregistration requires a clean worktree")
    return {"commit": commit, "status": "clean"}


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build(output: Path = PREREGISTRATION) -> dict[str, Any]:
    if output.exists():
        raise PreregistrationError("preregistration already exists")
    if EXPERIMENT_ROOT.exists():
        raise PreregistrationError("experiment output root must not exist")
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
    architecture = {
        owner: MMGDinoE5ResponsibilityOwners(
            ownership=owner
        ).architecture_report().as_dict()
        for owner in OWNERS
    }
    code_paths = {
        "contract": ROOT / "tools/original_gdino_parent_ownership.py",
        "config": EVAL_CONFIG,
        "preregistration_builder": Path(__file__),
        "runner": ROOT / "tools/run_original_gdino_parent_ownership.py",
        "mature_runner": ROOT / "tools/run_mmgdino_e6_ownership_2x2.py",
        "extractor": ROOT / "tools/extract_original_gdino_ownership_cache.py",
        "aggregator": ROOT / "tools/aggregate_original_gdino_parent_ownership.py",
        "mature_aggregator": ROOT / "tools/aggregate_mmgdino_pretrain_ownership.py",
        "owners": ROOT / "tools/mmgdino_e5_ownership.py",
        "trainer": ROOT / "tools/train_mmgdino_e5_ownership.py",
        "cache_evaluator": ROOT / "tools/eval_mmgdino_e5_ownership_cache.py",
        "cache_contract": ROOT / "tools/responsibility_isolation_cache.py",
        "score_objectives": ROOT / "models/GroundingDINO/stage_b_gdino_score_adapter.py",
    }
    payload = {
        "schema": SCHEMA,
        "status": "locked_before_any_parent_owner_gpu_forward",
        "locked_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "git": _git(),
        "research_question": (
            "Before the mixed Stage-B continuation that produced B58, does the "
            "same pure GroundingDINO representation benefit from Shared-Wide "
            "or capacity-matched hard isolation?"
        ),
        "causal_axis": {
            "release_initializer": "release_ogc",
            "formal_frozen_trunk": "pre_stageb_parent",
            "descendant_reference": "b58_descendant",
            "formal_trunk_is_direct_b58_parent": True,
            "same_effective_938_tensor_architecture": True,
        },
        "checkpoint_audit": _checkpoint_audit(),
        "matrix": {
            "owners": list(OWNERS),
            "seeds": list(FORMAL_SEEDS),
            "formal_trajectory_count": 6,
            "shared_128_excluded": True,
            "architecture": architecture,
        },
        "native_score": {
            "definition": "mean sigmoid probability over generated full-expression phrase tokens",
            "same_reduction_as_pure_GDINO_B58_base": True,
            "selected_after_results": False,
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
        },
        "schedule_contract": {
            "receipt": _record(SCHEDULE_RECEIPT),
            "per_seed": schedule_records,
            "same_identity_batch_order_and_losses_as_mmgdino_e5_e6_pretrain": True,
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
            "mmgdino_pretrained": _record(MMGDINO_PRETRAIN_REFERENCE),
            "mmgdino_e5": _record(E5_REFERENCE),
            "b58_capacity": _record(B58_REFERENCE),
            "reuse_only_no_forward": True,
        },
        "runtime_assets": {
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
            "do not substitute the earlier release OGC initializer for the direct B58 parent",
            "do not update the parent or import B58 tensors",
            "do not run Shared-128",
            "do not change owner capacity, schedule, losses, seeds, or endpoint",
            "do not select a checkpoint, score reduction, Gap, or threshold from results",
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
