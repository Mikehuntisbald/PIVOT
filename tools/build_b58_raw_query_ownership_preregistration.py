#!/usr/bin/env python3
"""Preregister the strict 100k matched-head replay on B58."""

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

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_FOR_IMPORT))

from tools.b58_raw_query_ownership import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    B58_CAPACITY_REFERENCE,
    CHECKPOINT,
    CHECKPOINT_SHA256,
    E5_REFERENCE,
    EVAL_CONFIG,
    EVAL_CONFIG_SHA256,
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    IMAGE_ROOT,
    OWNERS,
    PARENT_AGGREGATE,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    PARENT_RESULT,
    PARENT_TO_B58_CHANGED_TENSORS,
    PARENT_TO_B58_UNCHANGED_TENSORS,
    PREREGISTRATION,
    PURE_TRUNK_NUMEL,
    PURE_TRUNK_SCHEMA_SHA256,
    PURE_TRUNK_TENSORS,
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
from tools.mmgdino_e5_ownership import MMGDinoE5ResponsibilityOwners
from tools.responsibility_isolation_cache import file_sha256
from util.utils import clean_state_dict


SCHEMA = "arrow.b58_raw_query_ownership.preregistration/v1"


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
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise PreregistrationError("B58 checkpoint SHA drifted")
    if file_sha256(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise PreregistrationError("direct-parent checkpoint SHA drifted")
    parent_all = _load_state(PARENT_CHECKPOINT)
    parent = {
        key: value for key, value in parent_all.items() if not _parent_unused(key)
    }
    b58 = _load_state(CHECKPOINT)
    for label, state in (("parent", parent), ("b58", b58)):
        if (
            len(state) != PURE_TRUNK_TENSORS
            or sum(int(value.numel()) for value in state.values())
            != PURE_TRUNK_NUMEL
            or _schema(state) != PURE_TRUNK_SCHEMA_SHA256
        ):
            raise PreregistrationError(f"{label} pure-trunk schema drifted")
        if any(
            value.is_floating_point() and not bool(torch.isfinite(value).all().item())
            for value in state.values()
        ):
            raise PreregistrationError(f"{label} has non-finite tensors")
    if set(parent) != set(b58):
        raise PreregistrationError("parent/B58 effective tensor names differ")
    unchanged = sum(torch.equal(parent[key], b58[key]) for key in parent)
    if (
        int(unchanged) != PARENT_TO_B58_UNCHANGED_TENSORS
        or len(parent) - int(unchanged) != PARENT_TO_B58_CHANGED_TENSORS
    ):
        raise PreregistrationError("parent-to-B58 tensor delta drifted")
    return {
        "parent": _record(PARENT_CHECKPOINT),
        "b58": _record(CHECKPOINT),
        "same_effective_architecture": True,
        "pure_trunk_tensor_count": PURE_TRUNK_TENSORS,
        "pure_trunk_numel": PURE_TRUNK_NUMEL,
        "pure_trunk_schema_sha256": PURE_TRUNK_SCHEMA_SHA256,
        "unchanged_tensors": int(unchanged),
        "changed_tensors": len(parent) - int(unchanged),
        "all_effective_tensors_finite": True,
    }


def _git() -> dict[str, str]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=ROOT, text=True
    ).strip()
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


def _parent_record_bindings() -> list[dict[str, Any]]:
    receipt = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    routes = receipt.get("artifacts", {}).get("evaluation_routes", [])
    if len(routes) != 42:
        raise PreregistrationError("sealed parent route count drifted")
    bindings = []
    for route in routes:
        record = route["records"]
        path = ROOT / record["path"]
        observed = _record(path)
        if observed["sha256"] != record["sha256"]:
            raise PreregistrationError("sealed parent route bytes drifted")
        bindings.append({
            "surface": route["surface"],
            "route": route["route"],
            "seed": route["seed"],
            **observed,
        })
    return bindings


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
        "contract": ROOT / "tools/b58_raw_query_ownership.py",
        "config": EVAL_CONFIG,
        "preregistration_builder": Path(__file__),
        "runner": ROOT / "tools/run_b58_raw_query_ownership.py",
        "mature_runner": ROOT / "tools/run_mmgdino_e6_ownership_2x2.py",
        "extractor": ROOT / "tools/extract_b58_raw_query_ownership_cache.py",
        "shared_gdino_extraction_engine": ROOT / "tools/extract_original_gdino_ownership_cache.py",
        "aggregator": ROOT / "tools/aggregate_b58_raw_query_ownership.py",
        "mature_aggregator": ROOT / "tools/aggregate_mmgdino_pretrain_ownership.py",
        "owners": ROOT / "tools/mmgdino_e5_ownership.py",
        "trainer": ROOT / "tools/train_mmgdino_e5_ownership.py",
        "cache_evaluator": ROOT / "tools/eval_mmgdino_e5_ownership_cache.py",
        "cache_contract": ROOT / "tools/responsibility_isolation_cache.py",
        "score_objectives": ROOT / "models/GroundingDINO/stage_b_gdino_score_adapter.py",
    }
    payload = {
        "schema": SCHEMA,
        "status": "locked_before_any_b58_raw_query_gpu_forward",
        "locked_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "git": _git(),
        "research_question": (
            "Holding the mature 100k raw-query ownership heads and every "
            "training/evaluation choice fixed, does mixed Stage-B adaptation "
            "change the Shared-Wide versus Isolated effect from the direct "
            "parent to B58?"
        ),
        "checkpoint_audit": _checkpoint_audit(),
        "matrix": {
            "owners": list(OWNERS),
            "seeds": list(FORMAL_SEEDS),
            "formal_trajectory_count": 6,
            "shared_128_excluded": True,
            "architecture": architecture,
        },
        "same_head_axis": {
            "parent_result": _record(PARENT_RESULT),
            "parent_aggregate": _record(PARENT_AGGREGATE),
            "parent_records": _parent_record_bindings(),
            "same_100k_owner_code_and_initialization": True,
            "same_schedule_identity_batch_order_and_losses": True,
            "same_native_score_and_evaluator": True,
            "planned_difference_in_differences": [
                "Test5 ownership effect", "TestAB ownership effect",
                "Strict2031 FPR95-reduction ownership effect",
            ],
        },
        "native_score": {
            "definition": "mean sigmoid probability over generated full-expression phrase tokens",
            "same_reduction_as_parent_and_pure_GDINO_B58_base": True,
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
            "bitwise_reused_from_parent_replay": True,
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
            "same_image_draw_across_parent_b58_owners_and_seeds": True,
            "fpr95_recomputes_each_trunk_owner_seed_positive_q05_per_replicate": True,
            "rec_noninferiority_margin": REC_NONINFERIORITY_MARGIN,
        },
        "references": {
            "e5": _record(E5_REFERENCE),
            "existing_b58_integrated_capacity": _record(B58_CAPACITY_REFERENCE),
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
            "do not import or initialize from parent owner weights",
            "do not update B58 or run Shared-128",
            "do not change owner capacity initialization optimizer schedule losses seeds or endpoint",
            "do not select a checkpoint score reduction Gap or threshold from results",
            "do not stop after one seed or one evaluation surface",
        ],
    }
    for targets in payload["output_targets_absent"].values():
        if any(Path(path).exists() for path in targets):
            raise PreregistrationError("a preregistered output target exists")
    _atomic_json(payload, output)
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
