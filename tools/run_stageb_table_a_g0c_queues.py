#!/usr/bin/env python3
"""Durable formal-training and validation queues for Table-A/G0c.

The controller owns two exact serial queues:

* three G0c B10xA4/U1000 training runs for seeds 17/42/73;
* six Table-A validation runs, candidate then G0c for seeds 17/42/73.

It reuses the repository-wide durable GPU lease and process identity primitives,
but replays G0c and Table-A native postflights instead of pretending those runs
emit the generic paper-matrix sequence schema.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import aggregate_stageb_table_a_results as table_a_aggregate  # noqa: E402
from tools import run_stageb_serial_matrix_queue as shared_queue  # noqa: E402
from tools import run_stageb_table_a_controls as training_runner  # noqa: E402
from tools import run_stageb_table_a_evaluations as evaluator  # noqa: E402
from tools.stageb_dependency_audit import local_python_dependency_paths  # noqa: E402


QUEUE_SCHEMA = "pivot.stageb.table_a_g0c_queue/v1"
PLAN_SCHEMA = "pivot.stageb.table_a_g0c_queue_plan/v1"
JOB_LAUNCH_SCHEMA = "pivot.stageb.table_a_g0c_queue_job_launch/v1"
JOB_STATUS_SCHEMA = "pivot.stageb.table_a_g0c_queue_job_status/v1"
VERIFICATION_SCHEMA = "pivot.stageb.table_a_g0c_queue_verification/v1"
AUDIT_SCHEMA = "pivot.stageb.table_a_g0c_artifact_adoption_audit/v1"
TRAINING_ATTESTATION_SCHEMA = "pivot.stageb.table_a_g0c_training_queue_attestation/v1"

TRAINING_KIND = "g0c_training"
VALIDATION_KIND = "table_a_validation"
FINAL_KIND = "table_a_final"
QUEUE_KINDS = (TRAINING_KIND, VALIDATION_KIND, FINAL_KIND)
FORMAL_SEEDS = (17, 42, 73)
TRAINING_RUN_IDS = tuple(f"G0c:{seed}" for seed in FORMAL_SEEDS)
VALIDATION_RUN_IDS = tuple(
    [f"candidate:{seed}" for seed in FORMAL_SEEDS]
    + [f"g0c:{seed}" for seed in FORMAL_SEEDS]
)
FINAL_RUN_IDS = VALIDATION_RUN_IDS

ITEM_STATUSES = frozenset(
    {"pending", "reserved", "launching", "launched", "completed", "failed"}
)
ACTIVE_STATUSES = frozenset({"reserved", "launching", "launched", "failed"})

DEFAULT_TRAINING_QUEUE_DIR = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_a_g0c_training_u1000_v1"
)
DEFAULT_VALIDATION_QUEUE_DIR = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_a_g0c_validation_v1"
)
DEFAULT_FINAL_QUEUE_DIR = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_a_g0c_final_v1"
)
DEFAULT_LEASE_ROOT = shared_queue.DEFAULT_LEASE_ROOT
DEFAULT_PYTHON = Path(
    os.environ.get(
        "PIVOT_PYTHON", "/home/haoyi/miniconda/envs/gdino5090/bin/python"
    )
)
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))

LEGACY_PLAN_GLOB = "table_a_g0c*.json"
VOLATILE_KEYS = {
    "created_at_utc",
    "started_at_utc",
    "finished_at_utc",
    "verified_at_utc",
    "validated_at_utc",
    "updated_at_utc",
    "observed_at_utc",
    "claimed_at_utc",
}
COMPLETED_ORPHAN_PROCESS_IDENTITY = {
    "pid": 0,
    "available": False,
    "recovery_mode": "completed_output_without_live_child",
}


class G0cQueueError(RuntimeError):
    """A G0c queue or its child evidence violated the formal contract."""


class G0cQueueBusy(G0cQueueError):
    """A supervisor or GPU lease is owned elsewhere."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return shared_queue._sha256_file(path)


def _file_record(path: Path, *roles: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise G0cQueueError(f"queue input is not a regular file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "roles": sorted(set(roles)),
    }


def _merge_record(
    records: MutableMapping[str, dict[str, Any]], path: Path, *roles: str
) -> None:
    record = _file_record(path, *roles)
    existing = records.get(record["path"])
    if existing is None:
        records[record["path"]] = record
        return
    if (
        existing.get("sha256") != record["sha256"]
        or int(existing.get("size_bytes", -1)) != record["size_bytes"]
    ):
        raise G0cQueueError(f"queue input changed while planning: {record['path']}")
    existing["roles"] = sorted(set(existing.get("roles", ())).union(roles))


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise G0cQueueError(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise G0cQueueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise G0cQueueError(f"{label} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    shared_queue._write_json_atomic(path, value)


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _resolve_python(path: Path) -> Path:
    try:
        selected = shared_queue._resolve_executable(path)
        current = Path(sys.executable).resolve(strict=True)
    except (OSError, shared_queue.QueueContractError) as exc:
        raise G0cQueueError(f"invalid G0c queue Python: {exc}") from exc
    if selected != current:
        raise G0cQueueError(
            "G0c queue planning must use its selected Python: "
            f"caller={current}, selected={selected}"
        )
    return selected


def _runtime_contract(
    *, python: Path = DEFAULT_PYTHON, data_root: Path = DEFAULT_DATA_ROOT, gpu_key: str = "0"
) -> dict[str, Any]:
    python = _resolve_python(python)
    data_root = data_root.expanduser().resolve(strict=True)
    gpu_key = str(gpu_key).strip()
    if not data_root.is_dir():
        raise G0cQueueError(f"G0c data root is not a directory: {data_root}")
    if not gpu_key or "," in gpu_key:
        raise G0cQueueError("G0c queues require exactly one GPU key")
    return {
        "python": str(python),
        "python_record": _file_record(python, "queue_python_runtime"),
        "data_root": str(data_root),
        "gpu_key": gpu_key,
        "cuda_visible_devices": gpu_key,
        "training": {
            "micro_batch_size": training_runner.FORMAL_MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": (
                training_runner.FORMAL_GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_global_batch": training_runner.REQUIRED_EFFECTIVE_GLOBAL_BATCH,
            "optimizer_updates": training_runner.FORMAL_OPTIMIZER_UPDATES,
            "num_workers": 8,
        },
        "evaluation": {
            "device": evaluator.FORMAL_EVAL_DEVICE,
            "batch_size": evaluator.FORMAL_EVAL_BATCH_SIZE,
            "num_workers": evaluator.FORMAL_EVAL_NUM_WORKERS,
            "amp": True,
            "eval_seed": evaluator.EVAL_SEED,
        },
    }


def _controller_source_paths() -> tuple[Path, ...]:
    entries = (
        Path(__file__).resolve(),
        Path(training_runner.__file__).resolve(),
        Path(evaluator.__file__).resolve(),
        Path(table_a_aggregate.__file__).resolve(),
        Path(shared_queue.__file__).resolve(),
    )
    try:
        paths = local_python_dependency_paths(entries, root=REPO_ROOT)
    except Exception as exc:
        raise G0cQueueError(f"G0c controller dependency audit failed: {exc}") from exc
    return tuple(paths)


def _controller_source_records() -> list[dict[str, Any]]:
    return [
        _file_record(path, "g0c_queue_controller_source")
        for path in _controller_source_paths()
    ]


def _records_from_training_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    inputs = plan.get("inputs")
    tree = plan.get("source_dependency_tree")
    if not isinstance(inputs, Mapping) or not isinstance(tree, Mapping):
        raise G0cQueueError("G0c training plan lacks input/source closure")
    for label, value in inputs.items():
        if not isinstance(value, Mapping):
            raise G0cQueueError(f"G0c training input {label} is invalid")
        _merge_record(records, Path(str(value.get("path", ""))), f"training_input:{label}")
    tree_records = tree.get("records")
    if not isinstance(tree_records, list) or not tree_records:
        raise G0cQueueError("G0c training source tree is empty")
    for value in tree_records:
        if not isinstance(value, Mapping):
            raise G0cQueueError("G0c training source record is invalid")
        _merge_record(
            records,
            Path(str(value.get("path", ""))),
            "g0c_training_source_dependency",
        )
    return sorted(records.values(), key=lambda value: value["path"])


def _verify_file_record(record: Mapping[str, Any], *, label: str) -> Path:
    try:
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        stat = path.stat()
    except OSError as exc:
        raise G0cQueueError(f"{label} is unavailable: {exc}") from exc
    if (
        not path.is_file()
        or int(record.get("size_bytes", -1)) != int(stat.st_size)
        or int(record.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
        or str(record.get("sha256", "")) != _sha256_file(path)
    ):
        raise G0cQueueError(f"{label} changed after queue planning: {path}")
    return path


def _verify_record_set(values: Any, *, label: str) -> None:
    if not isinstance(values, list) or not values:
        raise G0cQueueError(f"{label} closure is empty")
    paths: set[Path] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise G0cQueueError(f"{label} record {index} is invalid")
        path = _verify_file_record(value, label=f"{label} record {index}")
        if path in paths:
            raise G0cQueueError(f"{label} closure contains duplicate path {path}")
        paths.add(path)


def audit_existing_artifacts() -> dict[str, Any]:
    legacy: list[dict[str, Any]] = []
    for path in sorted(
        (REPO_ROOT / "outputs/paper_cvpr_v1/plans").glob(LEGACY_PLAN_GLOB)
    ):
        try:
            value = _read_json(path, label="legacy G0c plan")
            schema = value.get("schema")
            reasons = []
            if schema != training_runner.PLAN_SCHEMA:
                reasons.append(
                    f"schema={schema!r}, required={training_runner.PLAN_SCHEMA!r}"
                )
            if not value.get("plan_sha256"):
                reasons.append("missing canonical plan SHA-256")
            if value.get("runtime_evidence_required") is not True:
                reasons.append("missing mandatory runtime evidence contract")
            if "python_runtime" not in value.get("inputs", {}):
                reasons.append("selected Python bytes are not bound")
            reasons.append(
                "artifact has no dedicated queue-owned active-item/job identity"
            )
            output_value = value.get("output_dir")
            output_root = (
                Path(str(output_value)).expanduser().resolve(strict=False)
                if isinstance(output_value, str) and output_value
                else None
            )
            legacy.append(
                {
                    "path": str(path.resolve()),
                    "status": "non_adoptable",
                    "reasons": reasons,
                    "output_root": str(output_root) if output_root else None,
                    "output_root_exists": bool(output_root and output_root.exists()),
                }
            )
        except G0cQueueError as exc:
            legacy.append(
                {
                    "path": str(path.resolve(strict=False)),
                    "status": "non_adoptable",
                    "reasons": [str(exc)],
                }
            )
    formal_roots = []
    fresh = True
    for seed in FORMAL_SEEDS:
        root = training_runner.formal_output_root(seed).resolve(strict=False)
        plan = training_runner.formal_plan_path(seed).resolve(strict=False)
        root_exists = root.exists()
        plan_exists = plan.exists()
        if root_exists or plan_exists:
            fresh = False
        formal_roots.append(
            {
                "seed": seed,
                "root": str(root),
                "plan": str(plan),
                "root_exists": root_exists,
                "plan_exists": plan_exists,
                "status": "fresh" if not root_exists and not plan_exists else "non_adoptable",
                "reason": (
                    None
                    if not root_exists and not plan_exists
                    else "artifact lacks a queue-owned active-item/job identity"
                ),
            }
        )
    soak_required = {
        "plan": str(training_runner.DEFAULT_SOAK_PLAN.resolve(strict=False)),
        "root": str(training_runner.DEFAULT_SOAK_ROOT.resolve(strict=False)),
        "seal": str(training_runner.DEFAULT_SOAK_SEAL.resolve(strict=False)),
    }
    soak_ready = False
    soak_error = None
    try:
        training_runner._validate_soak_seal(training_runner.DEFAULT_SOAK_SEAL)
        soak_ready = True
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        soak_error = f"{type(exc).__name__}: {exc}"
    canonical_queue: dict[str, Any] = {
        "path": str(DEFAULT_TRAINING_QUEUE_DIR.resolve(strict=False)),
        "present": DEFAULT_TRAINING_QUEUE_DIR.is_dir(),
        "status": "absent",
    }
    if canonical_queue["present"]:
        try:
            verification = verify_queue(DEFAULT_TRAINING_QUEUE_DIR)
            exact = (
                verification.get("status") == "passed"
                and verification.get("queue_status") == "completed"
                and verification.get("queue_kind") == TRAINING_KIND
                and tuple(verification.get("ordered_run_ids", ()))
                == TRAINING_RUN_IDS
                and len(verification.get("verified_items", ()))
                == len(TRAINING_RUN_IDS)
            )
            canonical_queue.update(
                {
                    "status": "exact_completed" if exact else "non_adoptable",
                    "verification": verification,
                }
            )
            if exact:
                exact_plan_paths = {
                    str(
                        Path(
                            str(
                                evidence["native_completion"]["formal_plan"]["path"]
                            )
                        ).resolve(strict=True)
                    )
                    for evidence in verification["verified_items"]
                }
                for artifact in legacy:
                    if artifact["path"] in exact_plan_paths:
                        artifact.update(
                            {
                                "status": "exact_queue_managed",
                                "reasons": [],
                            }
                        )
                for artifact in formal_roots:
                    if artifact["plan"] in exact_plan_paths:
                        artifact.update(
                            {
                                "status": "exact_queue_managed",
                                "reason": None,
                            }
                        )
        except (G0cQueueError, OSError, KeyError, TypeError, ValueError) as exc:
            canonical_queue.update(
                {
                    "status": "non_adoptable",
                    "verification_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "passed",
        "audited_at_utc": _utc_now(),
        "adoption_policy": "exact_current_schema_plus_queue_owned_job_identity_only",
        "legacy_artifacts": legacy,
        "formal_roots": formal_roots,
        "formal_roots_fresh": fresh,
        "canonical_soak": {
            **soak_required,
            "ready": soak_ready,
            "semantic_replay_error": soak_error,
        },
        "canonical_training_queue": canonical_queue,
        "training_queue_ready": bool(fresh and soak_ready),
    }


def _runtime_environment(runtime: Mapping[str, Any]) -> dict[str, str | None]:
    snapshot = {key: None for key in shared_queue.RUNTIME_ENV_KEYS}
    snapshot.update(
        {
            "PIVOT_PYTHON": str(runtime["python"]),
            "PIVOT_CUDA_VISIBLE_DEVICES": str(runtime["cuda_visible_devices"]),
            "PIVOT_DATA_ROOT": str(runtime["data_root"]),
            "DATA_ROOT": str(runtime["data_root"]),
            "CUDA_VISIBLE_DEVICES": str(runtime["cuda_visible_devices"]),
        }
    )
    return snapshot


def _training_args(seed: int, runtime: Mapping[str, Any]) -> SimpleNamespace:
    contract = runtime["training"]
    return SimpleNamespace(
        purpose="formal",
        checkpoint=str(training_runner.DEFAULT_CHECKPOINT),
        batch_size=int(contract["micro_batch_size"]),
        gradient_accumulation_steps=int(contract["gradient_accumulation_steps"]),
        effective_batch_size=int(contract["effective_global_batch"]),
        updates=int(contract["optimizer_updates"]),
        output_dir=str(training_runner.formal_output_root(seed)),
        python=str(runtime["python"]),
        seed=seed,
        num_workers=int(contract["num_workers"]),
        cuda_visible_devices=str(runtime["cuda_visible_devices"]),
        soak_seal=str(training_runner.DEFAULT_SOAK_SEAL),
    )


def _training_command(
    seed: int, runtime: Mapping[str, Any], plan_path: Path
) -> list[str]:
    contract = runtime["training"]
    return [
        str(runtime["python"]),
        str(Path(training_runner.__file__).resolve()),
        "run",
        "--purpose",
        "formal",
        "--python",
        str(runtime["python"]),
        "--checkpoint",
        str(training_runner.DEFAULT_CHECKPOINT),
        "--batch-size",
        str(contract["micro_batch_size"]),
        "--gradient-accumulation-steps",
        str(contract["gradient_accumulation_steps"]),
        "--effective-batch-size",
        str(contract["effective_global_batch"]),
        "--updates",
        str(contract["optimizer_updates"]),
        "--seed",
        str(seed),
        "--num-workers",
        str(contract["num_workers"]),
        "--output-dir",
        str(training_runner.formal_output_root(seed)),
        "--plan-json",
        str(plan_path),
        "--soak-seal",
        str(training_runner.DEFAULT_SOAK_SEAL),
        "--cuda-visible-devices",
        str(runtime["cuda_visible_devices"]),
    ]


def _build_training_items(runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    audit = audit_existing_artifacts()
    if not audit["formal_roots_fresh"]:
        raise G0cQueueError("canonical G0c formal roots/plans must be fresh")
    if audit["canonical_soak"]["ready"] is not True:
        raise G0cQueueError(
            "canonical telemetry-sealed G0c U50 soak is not complete"
        )
    items = []
    for seed in FORMAL_SEEDS:
        plan_path = training_runner.formal_plan_path(seed).resolve(strict=False)
        expected_plan = training_runner.build_plan(_training_args(seed, runtime))
        training_runner._validate_plan_identity(expected_plan)
        if Path(expected_plan["output_dir"]).resolve(strict=False) != (
            training_runner.formal_output_root(seed).resolve(strict=False)
        ):
            raise G0cQueueError(f"G0c seed {seed} formal output drifted")
        items.append(
            {
                "index": len(items),
                "run_id": f"G0c:{seed}",
                "item_kind": "training",
                "seed": seed,
                "output_root": str(training_runner.formal_output_root(seed)),
                "formal_plan_path": str(plan_path),
                "expected_plan": expected_plan,
                "expected_plan_sha256": expected_plan["plan_sha256"],
                "command": _training_command(seed, runtime, plan_path),
                "input_records": _records_from_training_plan(expected_plan),
            }
        )
    return items


def _stable_evaluation_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "kind",
        "profile",
        "repository_root",
        "evaluation_id",
        "output_dir",
        "source",
        "runtime",
        "instance",
        "final_gate",
        "contract",
        "tn_manifest",
        "tn_inputs",
        "commands",
        "inputs",
    )
    projection = {key: copy.deepcopy(plan.get(key)) for key in keys}
    if projection["schema"] != evaluator.SCHEMA:
        raise G0cQueueError("Table-A evaluation plan schema drifted")
    return projection


def _candidate_training_root(seed: int) -> Path:
    return (
        REPO_ROOT
        / "outputs/paper_cvpr_v1/token_ablation_frozen_v2/L4"
        / f"seed{seed}"
    )


def _evaluation_cli_command(
    *,
    kind: str,
    seed: int,
    profile: str,
    runtime: Mapping[str, Any],
    training_queue_dir: Path,
    final_gate: Path | None = None,
) -> list[str]:
    output = evaluator.canonical_output_dir(kind, profile, seed)
    command = [
        str(runtime["python"]),
        str(Path(evaluator.__file__).resolve()),
        "run",
        "--kind",
        kind,
        "--profile",
        profile,
        "--training-queue-dir",
        str(training_queue_dir),
        "--output-dir",
        str(output),
        "--python",
        str(runtime["python"]),
        "--data-root",
        str(runtime["data_root"]),
        "--device",
        str(runtime["evaluation"]["device"]),
        "--batch-size",
        str(runtime["evaluation"]["batch_size"]),
        "--num-workers",
        str(runtime["evaluation"]["num_workers"]),
    ]
    if profile == evaluator.FINAL_PROFILE:
        if final_gate is None:
            raise G0cQueueError("Table-A final command lacks its sealed gate")
        command.extend(["--final-gate", str(final_gate.resolve(strict=True))])
    elif final_gate is not None:
        raise G0cQueueError("Table-A validation command received a final gate")
    if kind == "candidate":
        command.extend(
            ["--training-run-root", str(_candidate_training_root(seed))]
        )
    else:
        command.extend(
            ["--g0c-training-plan", str(training_runner.formal_plan_path(seed))]
        )
    return command


def _completed_validation_queue_dependency(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    if queue_dir != DEFAULT_VALIDATION_QUEUE_DIR.resolve(strict=True):
        raise G0cQueueError("Table-A final queue requires the canonical validation queue")
    verification = verify_queue(queue_dir)
    if (
        verification.get("status") != "passed"
        or verification.get("queue_status") != "completed"
        or verification.get("queue_kind") != VALIDATION_KIND
        or tuple(verification.get("ordered_run_ids", ())) != VALIDATION_RUN_IDS
        or len(verification.get("verified_items", ())) != len(VALIDATION_RUN_IDS)
    ):
        raise G0cQueueError(
            "Table-A final queue requires the exact completed validation queue"
        )
    queue = load_queue(queue_dir)
    return {
        "queue_dir": str(queue_dir),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "ordered_run_ids": list(VALIDATION_RUN_IDS),
        "queue_manifest": _file_record(
            queue_dir / "queue.json", "table_a_validation_queue_predecessor"
        ),
    }


def _build_evaluation_items(
    runtime: Mapping[str, Any],
    *,
    profile: str,
    training_queue_dir: Path,
    validation_queue_dir: Path = DEFAULT_VALIDATION_QUEUE_DIR,
    final_gate: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if profile not in evaluator.PROFILES:
        raise G0cQueueError(f"unsupported Table-A queue profile: {profile!r}")
    training_queue_dir = training_queue_dir.expanduser().resolve(strict=True)
    training_verification = verify_queue(training_queue_dir)
    if (
        training_verification.get("status") != "passed"
        or training_verification.get("queue_kind") != TRAINING_KIND
        or tuple(training_verification.get("ordered_run_ids", ()))
        != TRAINING_RUN_IDS
    ):
        raise G0cQueueError("G0c training queue is not exactly completed/verified")
    predecessor = None
    resolved_final_gate = None
    if profile == evaluator.FINAL_PROFILE:
        if final_gate is None:
            raise G0cQueueError("Table-A final queue requires its sealed final gate")
        resolved_final_gate = final_gate.expanduser().resolve(strict=True)
        if resolved_final_gate != evaluator.FINAL_GATE_PATH.resolve(strict=True):
            raise G0cQueueError("Table-A final queue gate path is not canonical")
        predecessor = _completed_validation_queue_dependency(validation_queue_dir)
    elif final_gate is not None:
        raise G0cQueueError("Table-A validation queue cannot bind a final gate")
    table_runtime = evaluator.Runtime(
        python=Path(str(runtime["python"])),
        data_root=Path(str(runtime["data_root"])),
        device=str(runtime["evaluation"]["device"]),
        batch_size=int(runtime["evaluation"]["batch_size"]),
        num_workers=int(runtime["evaluation"]["num_workers"]),
        amp=bool(runtime["evaluation"]["amp"]),
    )
    items = []
    for kind in ("candidate", "g0c"):
        for seed in FORMAL_SEEDS:
            output = evaluator.canonical_output_dir(
                kind, profile, seed
            )
            if kind == "candidate":
                candidate_queue = Path(
                    evaluator.LOCKED_CANDIDATE_QUEUES[seed]["path"]
                )
                planned = evaluator.build_candidate_plan(
                    table_runtime,
                    _candidate_training_root(seed),
                    output,
                    profile=profile,
                    training_queue_dir=candidate_queue,
                    final_gate=resolved_final_gate,
                )
                source_queue = candidate_queue.resolve(strict=True)
            else:
                planned = evaluator.build_g0c_plan(
                    table_runtime,
                    training_runner.formal_plan_path(seed),
                    output,
                    profile=profile,
                    training_queue_dir=training_queue_dir,
                    final_gate=resolved_final_gate,
                )
                source_queue = training_queue_dir
            source = planned.get("source")
            if not isinstance(source, Mapping):
                raise G0cQueueError(f"Table-A {kind}:{seed} source is missing")
            items.append(
                {
                    "index": len(items),
                    "run_id": f"{kind}:{seed}",
                    "item_kind": "evaluation",
                    "evaluation_kind": kind,
                    "evaluation_profile": profile,
                    "seed": seed,
                    "output_root": str(output),
                    "command": _evaluation_cli_command(
                        kind=kind,
                        seed=seed,
                        profile=profile,
                        runtime=runtime,
                        training_queue_dir=source_queue,
                        final_gate=resolved_final_gate,
                    ),
                    "evaluation_plan_contract": _stable_evaluation_plan(planned),
                    "instance_sha256": planned["instance"]["instance_sha256"],
                    "training_queue": {
                        "path": str(source_queue),
                        "queue_id": source.get("training_queue_id"),
                        "plan_sha256": source.get("training_queue_plan_sha256"),
                    },
                    "input_records": copy.deepcopy(planned["inputs"]["records"]),
                }
            )
            if profile == evaluator.FINAL_PROFILE:
                items[-1]["final_consumption_path"] = str(
                    evaluator._final_consumption_path(planned["instance"])
                )
    return items, predecessor


def _canonical_queue_dir(queue_kind: str) -> Path:
    if queue_kind == TRAINING_KIND:
        return DEFAULT_TRAINING_QUEUE_DIR
    if queue_kind == VALIDATION_KIND:
        return DEFAULT_VALIDATION_QUEUE_DIR
    if queue_kind == FINAL_KIND:
        return DEFAULT_FINAL_QUEUE_DIR
    raise G0cQueueError(f"unknown G0c queue kind: {queue_kind!r}")


def _evaluation_profile(queue_kind: str) -> str:
    if queue_kind == VALIDATION_KIND:
        return evaluator.VALIDATION_PROFILE
    if queue_kind == FINAL_KIND:
        return evaluator.FINAL_PROFILE
    raise G0cQueueError(f"{queue_kind!r} is not a Table-A evaluation queue")


def build_queue_plan(
    queue_kind: str,
    queue_dir: Path,
    *,
    queue_id: str | None = None,
    python: Path = DEFAULT_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    gpu_key: str = "0",
    training_queue_dir: Path = DEFAULT_TRAINING_QUEUE_DIR,
    validation_queue_dir: Path = DEFAULT_VALIDATION_QUEUE_DIR,
    final_gate: Path = evaluator.FINAL_GATE_PATH,
    require_canonical_path: bool = True,
) -> dict[str, Any]:
    if queue_kind not in QUEUE_KINDS:
        raise G0cQueueError(f"unknown G0c queue kind: {queue_kind!r}")
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    expected_dir = _canonical_queue_dir(queue_kind).resolve(strict=False)
    if require_canonical_path and queue_dir != expected_dir:
        raise G0cQueueError(f"{queue_kind} queue path must be canonical: {expected_dir}")
    if queue_dir.exists():
        raise FileExistsError(f"G0c queue path must be fresh: {queue_dir}")
    runtime = _runtime_contract(python=python, data_root=data_root, gpu_key=gpu_key)
    predecessor = None
    if queue_kind == TRAINING_KIND:
        items = _build_training_items(runtime)
    else:
        profile = _evaluation_profile(queue_kind)
        items, predecessor = _build_evaluation_items(
            runtime,
            profile=profile,
            training_queue_dir=training_queue_dir,
            validation_queue_dir=validation_queue_dir,
            final_gate=(final_gate if queue_kind == FINAL_KIND else None),
        )
    expected_ids = _expected_run_ids(queue_kind)
    if tuple(item["run_id"] for item in items) != expected_ids:
        raise G0cQueueError(f"{queue_kind} item order is not exact")
    lease_root = DEFAULT_LEASE_ROOT.resolve(strict=False)
    selected_id = str(uuid.uuid4()) if queue_id is None else str(queue_id)
    if not selected_id:
        raise G0cQueueError("G0c queue ID must be non-empty")
    plan = {
        "schema": PLAN_SCHEMA,
        "queue_kind": queue_kind,
        "queue_id": selected_id,
        "created_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "repository_root": str(REPO_ROOT),
        "runtime": runtime,
        "runtime_environment": _runtime_environment(runtime),
        "gpu_key": str(runtime["gpu_key"]),
        "lease_root": str(lease_root),
        "lease_path": str(shared_queue._lease_path(lease_root, str(runtime["gpu_key"]))),
        "controller_sources": _controller_source_records(),
        "items": items,
    }
    if predecessor is not None:
        plan["predecessor_validation_queue"] = predecessor
    return plan


def _expected_run_ids(queue_kind: str) -> tuple[str, ...]:
    if queue_kind == TRAINING_KIND:
        return TRAINING_RUN_IDS
    if queue_kind == VALIDATION_KIND:
        return VALIDATION_RUN_IDS
    if queue_kind == FINAL_KIND:
        return FINAL_RUN_IDS
    raise G0cQueueError(f"unknown G0c queue kind: {queue_kind!r}")


def _validate_plan(plan: Mapping[str, Any], queue_dir: Path) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("queue_kind") not in QUEUE_KINDS:
        raise G0cQueueError("G0c queue has no valid immutable plan")
    if Path(str(plan.get("queue_dir", ""))).resolve(strict=False) != queue_dir:
        raise G0cQueueError("G0c queue was opened through a path outside its plan")
    if Path(str(plan.get("repository_root", ""))).resolve(strict=False) != REPO_ROOT:
        raise G0cQueueError("G0c queue repository root drifted")
    queue_id = plan.get("queue_id")
    if not isinstance(queue_id, str) or not queue_id:
        raise G0cQueueError("G0c queue ID is invalid")
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise G0cQueueError("G0c queue runtime is missing")
    python_record = runtime.get("python_record")
    if (
        not isinstance(python_record, Mapping)
        or Path(str(python_record.get("path", ""))).resolve(strict=False)
        != Path(str(runtime.get("python", ""))).resolve(strict=False)
        or runtime.get("cuda_visible_devices") != runtime.get("gpu_key")
    ):
        raise G0cQueueError("G0c queue Python/GPU runtime binding drifted")
    expected_training_runtime = {
        "micro_batch_size": training_runner.FORMAL_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": (
            training_runner.FORMAL_GRADIENT_ACCUMULATION_STEPS
        ),
        "effective_global_batch": training_runner.REQUIRED_EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": training_runner.FORMAL_OPTIMIZER_UPDATES,
        "num_workers": 8,
    }
    expected_evaluation_runtime = {
        "device": evaluator.FORMAL_EVAL_DEVICE,
        "batch_size": evaluator.FORMAL_EVAL_BATCH_SIZE,
        "num_workers": evaluator.FORMAL_EVAL_NUM_WORKERS,
        "amp": True,
        "eval_seed": evaluator.EVAL_SEED,
    }
    if (
        runtime.get("training") != expected_training_runtime
        or runtime.get("evaluation") != expected_evaluation_runtime
        or plan.get("runtime_environment") != _runtime_environment(runtime)
    ):
        raise G0cQueueError("G0c queue exact runtime contract drifted")
    if (
        runtime.get("gpu_key") != plan.get("gpu_key")
        or Path(str(plan.get("lease_path", ""))).resolve(strict=False)
        != shared_queue._lease_path(
            Path(str(plan.get("lease_root", ""))).resolve(strict=False),
            str(plan.get("gpu_key", "")),
        ).resolve(strict=False)
    ):
        raise G0cQueueError("G0c queue GPU lease contract drifted")
    items = plan.get("items")
    expected_ids = _expected_run_ids(str(plan["queue_kind"]))
    if not isinstance(items, list) or tuple(
        item.get("run_id") for item in items if isinstance(item, Mapping)
    ) != expected_ids:
        raise G0cQueueError("G0c queue immutable item order drifted")
    output_roots: set[Path] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping) or item.get("index") != index:
            raise G0cQueueError(f"G0c planned item {index} identity is invalid")
        command = item.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise G0cQueueError(f"G0c planned item {index} command is invalid")
        if command[0] != runtime.get("python"):
            raise G0cQueueError(f"G0c planned item {index} uses another Python")
        output = Path(str(item.get("output_root", ""))).resolve(strict=False)
        if plan["queue_kind"] == TRAINING_KIND:
            expected_kind = "training"
            expected_seed = FORMAL_SEEDS[index]
            expected_runner = Path(training_runner.__file__).resolve()
            expected_plan = item.get("expected_plan")
            if not isinstance(expected_plan, Mapping):
                raise G0cQueueError(
                    f"G0c planned training item {index} plan is missing"
                )
            try:
                training_runner._validate_plan_identity(expected_plan)
            except (KeyError, TypeError, ValueError) as exc:
                raise G0cQueueError(
                    f"G0c planned training item {index} plan identity failed: {exc}"
                ) from exc
            matched = expected_plan.get("matched_contract")
            expected_closure: dict[Path, str] = {}
            inputs = expected_plan.get("inputs")
            tree_records = expected_plan.get("source_dependency_tree", {}).get(
                "records"
            )
            if not isinstance(inputs, Mapping) or not isinstance(tree_records, list):
                raise G0cQueueError(
                    f"G0c planned training item {index} closure is missing"
                )
            for record in (*inputs.values(), *tree_records):
                if not isinstance(record, Mapping):
                    raise G0cQueueError(
                        f"G0c planned training item {index} closure is invalid"
                    )
                path = Path(str(record.get("path", ""))).resolve(strict=False)
                sha256 = str(record.get("sha256", ""))
                if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
                    raise G0cQueueError(
                        f"G0c planned training item {index} closure SHA is invalid"
                    )
                previous = expected_closure.setdefault(path, sha256)
                if previous != sha256:
                    raise G0cQueueError(
                        f"G0c planned training item {index} closure conflicts"
                    )
            actual_closure = {
                Path(str(record.get("path", ""))).resolve(strict=False): str(
                    record.get("sha256", "")
                )
                for record in item.get("input_records", ())
                if isinstance(record, Mapping)
            }
            if (
                item.get("item_kind") != expected_kind
                or item.get("seed") != expected_seed
                or len(command) < 2
                or Path(command[1]).resolve(strict=False) != expected_runner
                or item.get("expected_plan_sha256")
                != expected_plan.get("plan_sha256")
                or expected_plan.get("row_id") != "G0c"
                or expected_plan.get("purpose") != "formal"
                or not isinstance(matched, Mapping)
                or matched.get("seed") != expected_seed
                or matched.get("micro_batch_size_per_rank")
                != training_runner.FORMAL_MICRO_BATCH_SIZE
                or matched.get("gradient_accumulation_steps")
                != training_runner.FORMAL_GRADIENT_ACCUMULATION_STEPS
                or matched.get("effective_global_batch")
                != training_runner.REQUIRED_EFFECTIVE_GLOBAL_BATCH
                or matched.get("optimizer_updates")
                != training_runner.FORMAL_OPTIMIZER_UPDATES
                or Path(str(expected_plan.get("output_dir", ""))).resolve(
                    strict=False
                )
                != output
                or actual_closure != expected_closure
            ):
                raise G0cQueueError(
                    f"G0c planned training item {index} contract drifted"
                )
        else:
            expected_profile = _evaluation_profile(str(plan["queue_kind"]))
            expected_kind = "candidate" if index < len(FORMAL_SEEDS) else "g0c"
            expected_seed = FORMAL_SEEDS[index % len(FORMAL_SEEDS)]
            expected_runner = Path(evaluator.__file__).resolve()
            evaluation_plan = item.get("evaluation_plan_contract")
            expected_consumption = (
                str(
                    evaluator._final_consumption_path(
                        evaluation_plan.get("instance", {})
                    )
                )
                if isinstance(evaluation_plan, Mapping)
                and expected_profile == evaluator.FINAL_PROFILE
                else None
            )
            if (
                item.get("item_kind") != "evaluation"
                or item.get("evaluation_kind") != expected_kind
                or item.get("evaluation_profile") != expected_profile
                or item.get("seed") != expected_seed
                or len(command) < 2
                or Path(command[1]).resolve(strict=False) != expected_runner
                or not isinstance(evaluation_plan, Mapping)
                or dict(evaluation_plan) != _stable_evaluation_plan(evaluation_plan)
                or evaluation_plan.get("kind") != expected_kind
                or evaluation_plan.get("profile") != expected_profile
                or Path(str(evaluation_plan.get("output_dir", ""))).resolve(
                    strict=False
                )
                != output
                or evaluation_plan.get("instance", {}).get("seed") != expected_seed
                or evaluation_plan.get("instance", {}).get("instance_sha256")
                != item.get("instance_sha256")
                or evaluation_plan.get("inputs", {}).get("records")
                != item.get("input_records")
                or (
                    expected_profile == evaluator.FINAL_PROFILE
                    and (
                        not isinstance(evaluation_plan.get("final_gate"), Mapping)
                        or Path(
                            str(evaluation_plan["final_gate"].get("path", ""))
                        ).resolve(strict=False)
                        != evaluator.FINAL_GATE_PATH.resolve(strict=False)
                        or item.get("final_consumption_path")
                        != expected_consumption
                    )
                )
                or (
                    expected_profile == evaluator.VALIDATION_PROFILE
                    and (
                        evaluation_plan.get("final_gate") is not None
                        or "final_consumption_path" in item
                    )
                )
            ):
                raise G0cQueueError(
                    f"Table-A planned {expected_profile} item {index} contract drifted"
                )
        if output in output_roots:
            raise G0cQueueError("G0c queue output roots are duplicated")
        output_roots.add(output)
        records = item.get("input_records")
        if not isinstance(records, list) or not records:
            raise G0cQueueError(f"G0c planned item {index} input closure is empty")
    sources = plan.get("controller_sources")
    if not isinstance(sources, list) or not sources:
        raise G0cQueueError("G0c queue controller source closure is empty")
    source_paths = [
        Path(str(record.get("path", ""))).resolve(strict=False)
        for record in sources
        if isinstance(record, Mapping)
    ]
    if len(source_paths) != len(sources) or len(set(source_paths)) != len(source_paths):
        raise G0cQueueError("G0c queue controller source closure is invalid")
    predecessor = plan.get("predecessor_validation_queue")
    if plan["queue_kind"] == FINAL_KIND:
        if (
            not isinstance(predecessor, Mapping)
            or Path(str(predecessor.get("queue_dir", ""))).resolve(strict=False)
            != DEFAULT_VALIDATION_QUEUE_DIR.resolve(strict=False)
            or predecessor.get("ordered_run_ids") != list(VALIDATION_RUN_IDS)
            or not isinstance(predecessor.get("queue_id"), str)
            or not predecessor.get("queue_id")
            or re.fullmatch(
                r"[0-9a-f]{64}", str(predecessor.get("plan_sha256", ""))
            )
            is None
            or not isinstance(predecessor.get("queue_manifest"), Mapping)
        ):
            raise G0cQueueError(
                "Table-A final queue validation predecessor contract drifted"
            )
    elif predecessor is not None:
        raise G0cQueueError(
            "non-final G0c queue bound a final-validation predecessor"
        )
    canonical_dir = _canonical_queue_dir(str(plan["queue_kind"])).resolve(
        strict=False
    )
    if queue_dir == canonical_dir:
        if Path(str(plan.get("lease_root", ""))).resolve(strict=False) != (
            DEFAULT_LEASE_ROOT.resolve(strict=False)
        ):
            raise G0cQueueError("canonical G0c queue does not use the shared GPU lease")
        for item in items:
            seed = int(item["seed"])
            if plan["queue_kind"] == TRAINING_KIND:
                expected_output = training_runner.formal_output_root(seed)
                expected_plan_path = training_runner.formal_plan_path(seed)
                if (
                    Path(str(item["output_root"])).resolve(strict=False)
                    != expected_output.resolve(strict=False)
                    or Path(str(item["formal_plan_path"])).resolve(strict=False)
                    != expected_plan_path.resolve(strict=False)
                ):
                    raise G0cQueueError("canonical G0c training path drifted")
            else:
                expected_profile = _evaluation_profile(str(plan["queue_kind"]))
                expected_output = evaluator.canonical_output_dir(
                    str(item["evaluation_kind"]),
                    expected_profile,
                    seed,
                )
                if Path(str(item["output_root"])).resolve(strict=False) != (
                    expected_output.resolve(strict=False)
                ):
                    raise G0cQueueError(
                        f"canonical Table-A {expected_profile} path drifted"
                    )


def _job_dir(queue: Mapping[str, Any], item: Mapping[str, Any]) -> Path:
    job_id = item.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise G0cQueueError("G0c active item job ID is invalid")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item["run_id"]))
    return (
        Path(str(queue["plan"]["queue_dir"]))
        / "jobs"
        / f"{int(item['index']):03d}-{slug}"
        / job_id
    ).resolve(strict=False)


def _active_projection(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": item["index"],
        "run_id": item["run_id"],
        "job_id": item["job_id"],
        "job_dir": item["job_dir"],
    }


def _validate_queue(queue: Mapping[str, Any], queue_dir: Path) -> None:
    if queue.get("schema") != QUEUE_SCHEMA:
        raise G0cQueueError("unsupported G0c queue schema")
    plan = queue.get("plan")
    if not isinstance(plan, Mapping):
        raise G0cQueueError("G0c queue immutable plan is missing")
    _validate_plan(plan, queue_dir)
    if queue.get("plan_sha256") != _canonical_sha256(plan):
        raise G0cQueueError("G0c immutable queue plan SHA-256 mismatch")
    status = queue.get("status")
    if status not in {"planned", "running", "completed", "failed"}:
        raise G0cQueueError(f"G0c queue status is invalid: {status!r}")
    planned = plan["items"]
    items = queue.get("items")
    if not isinstance(items, list) or len(items) != len(planned):
        raise G0cQueueError("G0c mutable items differ from immutable plan")
    completed_prefix = True
    active: list[Mapping[str, Any]] = []
    for index, (expected, item) in enumerate(zip(planned, items)):
        if not isinstance(item, Mapping):
            raise G0cQueueError(f"G0c mutable item {index} is invalid")
        if item.get("index") != index or item.get("run_id") != expected["run_id"]:
            raise G0cQueueError(f"G0c mutable item {index} identity drifted")
        item_status = item.get("status")
        if item_status not in ITEM_STATUSES:
            raise G0cQueueError(f"G0c mutable item {index} status is invalid")
        if item_status == "completed":
            if not completed_prefix:
                raise G0cQueueError("G0c completed items must form one prefix")
        else:
            completed_prefix = False
        if item_status in ACTIVE_STATUSES:
            active.append(item)
        if item_status == "pending" and any(
            later.get("status") != "pending" for later in items[index + 1 :]
        ):
            raise G0cQueueError("G0c item advanced past a pending predecessor")
    if len(active) > 1:
        raise G0cQueueError("G0c queue has more than one active item")
    active_record = queue.get("active_item")
    if active:
        item = active[0]
        for field in ("job_id", "job_dir"):
            if not isinstance(item.get(field), str) or not item.get(field):
                raise G0cQueueError(f"G0c active item lacks {field}")
        if Path(str(item["job_dir"])).resolve(strict=False) != _job_dir(queue, item):
            raise G0cQueueError("G0c active item job directory drifted")
        if active_record != _active_projection(item):
            raise G0cQueueError("G0c queue active-item projection drifted")
    elif active_record is not None:
        raise G0cQueueError("G0c queue has stale active-item identity")
    if status == "planned" and any(item["status"] != "pending" for item in items):
        raise G0cQueueError("planned G0c queue has a started item")
    if status == "completed" and any(item["status"] != "completed" for item in items):
        raise G0cQueueError("completed G0c queue has incomplete items")
    if status != "completed" and all(item["status"] == "completed" for item in items):
        raise G0cQueueError("fully completed G0c items require completed queue status")
    failed = [item for item in items if item["status"] == "failed"]
    if status == "failed":
        if len(active) != 1 or len(failed) != 1 or active[0] is not failed[0]:
            raise G0cQueueError("failed G0c queue lacks one failed active item")
    elif failed:
        raise G0cQueueError("non-failed G0c queue contains a failed item")


def create_queue_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(dict(plan))
    queue_dir = Path(str(plan.get("queue_dir", ""))).resolve(strict=False)
    if queue_dir.exists():
        raise FileExistsError(f"G0c queue path must be fresh: {queue_dir}")
    _validate_plan(plan, queue_dir)
    now = _utc_now()
    queue: dict[str, Any] = {
        "schema": QUEUE_SCHEMA,
        "status": "planned",
        "created_at_utc": now,
        "updated_at_utc": now,
        "revision": 0,
        "plan": plan,
        "plan_sha256": _canonical_sha256(plan),
        "items": [
            {
                "index": index,
                "run_id": item["run_id"],
                "status": "pending",
            }
            for index, item in enumerate(plan["items"])
        ],
        "active_item": None,
        "events": [
            {
                "at_utc": now,
                "event": "queue_created",
                "queue_kind": plan["queue_kind"],
                "ordered_run_ids": [item["run_id"] for item in plan["items"]],
            }
        ],
    }
    queue_dir.mkdir(parents=True, exist_ok=False)
    _write_json(queue_dir / "queue.json", queue)
    return load_queue(queue_dir)


def create_queue(
    queue_kind: str,
    queue_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return create_queue_from_plan(build_queue_plan(queue_kind, queue_dir, **kwargs))


def load_queue(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    if not queue_dir.is_dir():
        raise NotADirectoryError(queue_dir)
    queue = _read_json(queue_dir / "queue.json", label="G0c queue state")
    _validate_queue(queue, queue_dir)
    return queue


def _save_queue(queue: MutableMapping[str, Any]) -> None:
    queue_dir = Path(str(queue["plan"]["queue_dir"])).resolve(strict=True)
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_at_utc"] = _utc_now()
    _validate_queue(queue, queue_dir)
    _write_json(queue_dir / "queue.json", queue)


def _event(queue: MutableMapping[str, Any], event: str, **fields: Any) -> None:
    events = queue.setdefault("events", [])
    if not isinstance(events, list):
        raise G0cQueueError("G0c queue events are invalid")
    events.append({"at_utc": _utc_now(), "event": event, **fields})


def _verify_plan_closure(queue: Mapping[str, Any]) -> None:
    _verify_file_record(queue["plan"]["runtime"]["python_record"], label="queue Python")
    recorded_sources = {
        Path(str(record["path"])).resolve(strict=False)
        for record in queue["plan"]["controller_sources"]
    }
    if recorded_sources != set(_controller_source_paths()):
        raise G0cQueueError("G0c queue controller source closure is incomplete")
    _verify_record_set(queue["plan"]["controller_sources"], label="controller source")
    for item in queue["plan"]["items"]:
        _verify_record_set(
            item.get("input_records"), label=f"{item['run_id']} input"
        )
    if queue["plan"]["queue_kind"] == FINAL_KIND:
        predecessor = queue["plan"].get("predecessor_validation_queue")
        if not isinstance(predecessor, Mapping):
            raise G0cQueueError(
                "Table-A final queue predecessor validation queue is missing"
            )
        _verify_file_record(
            predecessor.get("queue_manifest", {}),
            label="Table-A predecessor validation queue manifest",
        )
        predecessor_queue = load_queue(Path(str(predecessor["queue_dir"])))
        if (
            predecessor_queue.get("status") != "completed"
            or predecessor_queue["plan"].get("queue_kind") != VALIDATION_KIND
            or predecessor_queue["plan"].get("queue_id")
            != predecessor.get("queue_id")
            or predecessor_queue.get("plan_sha256")
            != predecessor.get("plan_sha256")
            or [item["run_id"] for item in predecessor_queue["items"]]
            != list(VALIDATION_RUN_IDS)
        ):
            raise G0cQueueError(
                "Table-A predecessor validation queue identity/status drifted"
            )


def _ensure_lease(
    queue: Mapping[str, Any], item: Mapping[str, Any], *, create: bool
) -> None:
    try:
        shared_queue._ensure_lease(queue, item, create=create)
    except shared_queue.QueueBusyError as exc:
        raise G0cQueueBusy(str(exc)) from exc
    except shared_queue.QueueContractError as exc:
        raise G0cQueueError(f"G0c GPU lease verification failed: {exc}") from exc


def _clear_lease(queue: Mapping[str, Any]) -> None:
    try:
        shared_queue._clear_owned_lease(queue)
    except shared_queue.QueueBusyError as exc:
        raise G0cQueueBusy(str(exc)) from exc
    except shared_queue.QueueContractError as exc:
        raise G0cQueueError(f"G0c GPU lease cleanup failed: {exc}") from exc


def _planned_item(queue: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    return queue["plan"]["items"][index]


def _job_identity(queue: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "queue_kind": queue["plan"]["queue_kind"],
        "item_index": item["index"],
        "run_id": item["run_id"],
        "job_id": item["job_id"],
        "job_dir": item["job_dir"],
    }


def _validate_process_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise G0cQueueError(f"{label} process identity is missing")
    pid = value.get("pid")
    start = value.get("start_time_ticks")
    boot = value.get("boot_id")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or value.get("available") is not True
        or isinstance(start, bool)
        or not isinstance(start, int)
        or start <= 0
        or not isinstance(boot, str)
        or not boot
    ):
        raise G0cQueueError(f"{label} lacks exact PID/start-time/boot identity")
    return dict(value)


def _matching_processes(command: Sequence[str]) -> list[tuple[int, dict[str, Any]]]:
    expected = list(command)
    matches = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            observed = [
                value.decode("utf-8", errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except OSError:
            continue
        if observed != expected:
            continue
        pid = int(entry.name)
        identity = shared_queue._read_process_identity(pid)
        if shared_queue._process_running(pid, identity) is True:
            matches.append(
                (pid, _validate_process_identity(identity, label="matching child"))
            )
    return matches


def _reserve(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_plan_closure(queue)
    planned = _planned_item(queue, index)
    output = Path(str(planned["output_root"])).resolve(strict=False)
    if output.exists():
        raise G0cQueueError(f"fresh G0c output already exists: {output}")
    if queue["plan"]["queue_kind"] == FINAL_KIND and Path(
        str(planned.get("final_consumption_path", ""))
    ).exists():
        raise G0cQueueError(
            "final Table-A instance was already consumed before reservation"
        )
    if queue["plan"]["queue_kind"] == TRAINING_KIND and Path(
        str(planned["formal_plan_path"])
    ).exists():
        raise G0cQueueError("fresh G0c formal plan appeared after queue planning")
    item = queue["items"][index]
    _ensure_lease(queue, item, create=index == 0)
    job_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"
    item.update({"status": "reserved", "job_id": job_id})
    item["job_dir"] = str(_job_dir(queue, item))
    item["reserved_at_utc"] = _utc_now()
    queue["status"] = "running"
    queue["active_item"] = _active_projection(item)
    _event(queue, "item_reserved", index=index, run_id=item["run_id"], job_id=job_id)
    _save_queue(queue)


def _prepare_job(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_plan_closure(queue)
    item = queue["items"][index]
    planned = _planned_item(queue, index)
    _ensure_lease(queue, item, create=False)
    job_dir = _job_dir(queue, item)
    if job_dir.exists():
        raise G0cQueueError(f"G0c job directory must be fresh: {job_dir}")
    job_dir.mkdir(parents=True, exist_ok=False)
    identity = _job_identity(queue, item)
    launch = {
        "schema": JOB_LAUNCH_SCHEMA,
        "status": "prepared",
        **identity,
        "command": copy.deepcopy(planned["command"]),
        "output_root": planned["output_root"],
        "prepared_at_utc": _utc_now(),
    }
    status = {
        "schema": JOB_STATUS_SCHEMA,
        "status": "prepared",
        **identity,
        "updated_at_utc": _utc_now(),
    }
    _write_json(job_dir / "launch.json", launch)
    _write_json(job_dir / "status.json", status)
    item["status"] = "launching"
    item["launching_at_utc"] = _utc_now()
    _event(queue, "job_prepared", index=index, run_id=item["run_id"])
    _save_queue(queue)


def _bind_child(
    queue: MutableMapping[str, Any], index: int, pid: int, identity: Mapping[str, Any]
) -> None:
    item = queue["items"][index]
    job_dir = _job_dir(queue, item)
    identity = _validate_process_identity(identity, label="G0c child")
    if identity["pid"] != pid:
        raise G0cQueueError("G0c child PID differs from its process identity")
    try:
        process_group_id = os.getpgid(pid)
    except OSError as exc:
        raise G0cQueueError(f"cannot read G0c child process group: {exc}") from exc
    if process_group_id != pid:
        raise G0cQueueError("G0c child is not its detached process-group leader")
    launch = _read_json(job_dir / "launch.json", label="G0c job launch")
    status = _read_json(job_dir / "status.json", label="G0c job status")
    launch.update(
        {
            "status": "launched",
            "child_pid": pid,
            "child_process_identity": identity,
            "process_group_id": process_group_id,
            "launched_at_utc": _utc_now(),
        }
    )
    status.update(
        {
            "status": "running",
            "child_pid": pid,
            "child_process_identity": identity,
            "process_group_id": process_group_id,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(job_dir / "launch.json", launch)
    _write_json(job_dir / "status.json", status)
    item.update(
        {
            "status": "launched",
            "child_pid": pid,
            "child_process_identity": identity,
            "process_group_id": process_group_id,
            "launched_at_utc": _utc_now(),
        }
    )
    _event(queue, "child_bound", index=index, run_id=item["run_id"], pid=pid)
    _save_queue(queue)


def _bind_completed_orphan(
    queue: MutableMapping[str, Any], index: int
) -> None:
    """Adopt only a fully replayable evaluation that outlived its supervisor."""

    item = queue["items"][index]
    if queue["plan"]["queue_kind"] == TRAINING_KIND:
        raise G0cQueueError("training output cannot use evaluation orphan recovery")
    _verify_evaluation_completion(queue, item)
    job_dir = _job_dir(queue, item)
    launch = _read_json(job_dir / "launch.json", label="G0c job launch")
    status = _read_json(job_dir / "status.json", label="G0c job status")
    identity = copy.deepcopy(COMPLETED_ORPHAN_PROCESS_IDENTITY)
    now = _utc_now()
    launch.update(
        {
            "status": "launched",
            "child_pid": 0,
            "child_process_identity": identity,
            "process_group_id": 0,
            "launched_at_utc": now,
            "recovery_mode": identity["recovery_mode"],
        }
    )
    status.update(
        {
            "status": "running",
            "child_pid": 0,
            "child_process_identity": identity,
            "process_group_id": 0,
            "updated_at_utc": now,
            "recovery_mode": identity["recovery_mode"],
        }
    )
    _write_json(job_dir / "launch.json", launch)
    _write_json(job_dir / "status.json", status)
    item.update(
        {
            "status": "launched",
            "child_pid": 0,
            "child_process_identity": identity,
            "process_group_id": 0,
            "launched_at_utc": now,
            "completed_orphan_recovery": True,
        }
    )
    _event(
        queue,
        "completed_orphan_bound",
        index=index,
        run_id=item["run_id"],
    )
    _save_queue(queue)


def _launch_or_recover(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_plan_closure(queue)
    item = queue["items"][index]
    planned = _planned_item(queue, index)
    _ensure_lease(queue, item, create=False)
    command = [str(value) for value in planned["command"]]
    matches = _matching_processes(command)
    if len(matches) > 1:
        raise G0cQueueError("multiple exact G0c child processes match one job")
    if matches:
        _bind_child(queue, index, *matches[0])
        return
    output = Path(str(planned["output_root"])).resolve(strict=False)
    if output.exists():
        launch_manifest = output / "launch_manifest.json"
        if queue["plan"]["queue_kind"] != TRAINING_KIND and launch_manifest.is_file():
            observed = _read_json(
                launch_manifest, label="completed orphan Table-A launch"
            )
            if observed.get("status") == "completed":
                _bind_completed_orphan(queue, index)
                return
        raise G0cQueueError(
            "G0c output appeared without a recoverable exact child identity"
        )
    if queue["plan"]["queue_kind"] == TRAINING_KIND and Path(
        str(planned["formal_plan_path"])
    ).exists():
        raise G0cQueueError(
            "G0c formal plan appeared without a recoverable exact child identity"
        )
    if queue["plan"]["queue_kind"] == FINAL_KIND and Path(
        str(planned.get("final_consumption_path", ""))
    ).exists():
        raise G0cQueueError(
            "final Table-A consumption exists without a live or completed child; "
            "rerun is forbidden"
        )
    job_dir = _job_dir(queue, item)
    console = job_dir / "console.log"
    try:
        environment = shared_queue._runner_environment(
            queue["plan"]["runtime_environment"]
        )
        with console.open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except (OSError, shared_queue.QueueContractError) as exc:
        raise G0cQueueError(f"cannot launch G0c child: {exc}") from exc
    identity = shared_queue._read_process_identity(process.pid)
    try:
        _bind_child(queue, index, process.pid, identity)
    except BaseException:
        with contextlib.suppress(OSError):
            process.terminate()
        raise


def _validate_job_binding(
    queue: Mapping[str, Any], item: Mapping[str, Any], *, terminal: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    job_dir = _job_dir(queue, item)
    launch = _read_json(job_dir / "launch.json", label="G0c job launch")
    status = _read_json(job_dir / "status.json", label="G0c job status")
    expected = _job_identity(queue, item)
    for label, value in (("launch", launch), ("status", status)):
        if value.get("schema") != (
            JOB_LAUNCH_SCHEMA if label == "launch" else JOB_STATUS_SCHEMA
        ):
            raise G0cQueueError(f"G0c job {label} schema drifted")
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise G0cQueueError(f"G0c job {label} identity drifted at {key}")
    planned = _planned_item(queue, int(item["index"]))
    if (
        launch.get("command") != planned["command"]
        or Path(str(launch.get("output_root", ""))).resolve(strict=False)
        != Path(str(planned["output_root"])).resolve(strict=False)
    ):
        raise G0cQueueError("G0c job launch command/output drifted")
    orphan = item.get("completed_orphan_recovery") is True
    if orphan:
        expected_process = copy.deepcopy(COMPLETED_ORPHAN_PROCESS_IDENTITY)
        if (
            item.get("child_pid") != 0
            or item.get("process_group_id") != 0
            or item.get("child_process_identity") != expected_process
        ):
            raise G0cQueueError("completed orphan process identity drifted")
    else:
        expected_process = _validate_process_identity(
            item.get("child_process_identity"), label="queue item"
        )
        if (
            item.get("process_group_id") != item.get("child_pid")
            or item.get("child_pid") != expected_process.get("pid")
        ):
            raise G0cQueueError("G0c queue item process-group identity drifted")
    for label, value in (("launch", launch), ("status", status)):
        if (
            value.get("child_pid") != item.get("child_pid")
            or value.get("process_group_id") != item.get("process_group_id")
            or value.get("child_process_identity") != expected_process
        ):
            raise G0cQueueError(f"G0c job {label} child identity drifted")
    expected_status = "completed" if terminal else "running"
    if status.get("status") != expected_status:
        raise G0cQueueError(
            f"G0c job status must be {expected_status}, got {status.get('status')!r}"
        )
    expected_launch_status = "completed" if terminal else "launched"
    if launch.get("status") != expected_launch_status:
        raise G0cQueueError(
            "G0c job launch must be "
            f"{expected_launch_status}, got {launch.get('status')!r}"
        )
    if terminal:
        running = shared_queue._process_running(
            item.get("child_pid"), expected_process
        )
        if running is not False:
            raise G0cQueueError("completed G0c job still has a live/unknown child")
        evidence = item.get("completion_evidence")
        if (
            not isinstance(evidence, Mapping)
            or status.get("completion_evidence_sha256")
            != _canonical_sha256(evidence)
            or launch.get("completion_evidence_sha256")
            != _canonical_sha256(evidence)
        ):
            raise G0cQueueError("G0c job completion evidence binding drifted")
    return launch, status


def _verify_training_completion(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    planned = _planned_item(queue, int(item["index"]))
    seed = int(planned["seed"])
    plan_path = Path(str(planned["formal_plan_path"])).resolve(strict=True)
    if plan_path != training_runner.formal_plan_path(seed).resolve(strict=True):
        raise G0cQueueError("completed G0c training plan path is not canonical")
    persisted_plan = _read_json(plan_path, label="completed G0c training plan")
    if persisted_plan != planned["expected_plan"]:
        raise G0cQueueError("completed G0c training plan differs from queue plan")
    try:
        training_runner._validate_plan_identity(persisted_plan)
        replay = training_runner.verify_checkpoint(
            persisted_plan, write_postflight=False
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise G0cQueueError(f"G0c training completion replay failed: {exc}") from exc
    output = Path(str(planned["output_root"])).resolve(strict=True)
    postflight_path = (output / "postflight.json").resolve(strict=True)
    persisted_postflight = _read_json(
        postflight_path, label="completed G0c training postflight"
    )
    if _strip_volatile(persisted_postflight) != _strip_volatile(replay):
        raise G0cQueueError("persisted G0c training postflight differs from replay")
    if (
        persisted_postflight.get("status") != "PASS"
        or persisted_postflight.get("purpose") != "formal"
        or int(persisted_postflight.get("seed", -1)) != seed
        or persisted_postflight.get("plan_sha256")
        != planned["expected_plan_sha256"]
    ):
        raise G0cQueueError("G0c training postflight identity is invalid")
    checkpoint = (output / "checkpoint_iter.pth").resolve(strict=True)
    return {
        "queue_kind": TRAINING_KIND,
        "run_id": item["run_id"],
        "seed": seed,
        "output_root": str(output),
        "formal_plan": _file_record(plan_path, "completed_training_plan"),
        "postflight": _file_record(postflight_path, "completed_training_postflight"),
        "checkpoint": _file_record(checkpoint, "completed_training_checkpoint"),
        "plan_contract_sha256": planned["expected_plan_sha256"],
    }


def _verify_evaluation_completion(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    planned = _planned_item(queue, int(item["index"]))
    profile = _evaluation_profile(str(queue["plan"]["queue_kind"]))
    output = Path(str(planned["output_root"])).resolve(strict=True)
    launch_path = (output / "launch_manifest.json").resolve(strict=True)
    postflight_path = (output / "postflight.json").resolve(strict=True)
    launch = _read_json(launch_path, label="completed Table-A launch")
    persisted_postflight = _read_json(
        postflight_path, label="completed Table-A postflight"
    )
    if (
        launch.get("status") != "completed"
        or launch.get("kind") != planned["evaluation_kind"]
        or launch.get("profile") != profile
        or int(launch.get("instance", {}).get("seed", -1)) != int(planned["seed"])
        or launch.get("instance", {}).get("instance_sha256")
        != planned["instance_sha256"]
    ):
        raise G0cQueueError(
            f"Table-A {profile} launch identity/status drifted"
        )
    if _stable_evaluation_plan(launch) != planned["evaluation_plan_contract"]:
        raise G0cQueueError(
            f"Table-A {profile} launch differs from predeclared plan"
        )
    try:
        replay = evaluator.postflight(launch)
    except Exception as exc:
        raise G0cQueueError(
            f"Table-A {profile} postflight replay failed: {exc}"
        ) from exc
    if _strip_volatile(persisted_postflight) != _strip_volatile(replay):
        raise G0cQueueError(
            f"persisted Table-A {profile} postflight differs from replay"
        )
    evidence = {
        "queue_kind": queue["plan"]["queue_kind"],
        "run_id": item["run_id"],
        "evaluation_kind": planned["evaluation_kind"],
        "evaluation_profile": profile,
        "seed": int(planned["seed"]),
        "output_root": str(output),
        "instance_sha256": planned["instance_sha256"],
        "launch_manifest": _file_record(launch_path, "completed_evaluation_launch"),
        "postflight": _file_record(postflight_path, "completed_evaluation_postflight"),
    }
    if profile == evaluator.FINAL_PROFILE:
        try:
            consumption = evaluator._validate_final_consumption(launch)
        except Exception as exc:
            raise G0cQueueError(
                f"Table-A final consumption replay failed: {exc}"
            ) from exc
        consumption_path = Path(str(consumption["path"])).resolve(strict=True)
        if (
            consumption_path
            != Path(str(planned.get("final_consumption_path", ""))).resolve(
                strict=True
            )
            or persisted_postflight.get("final_consumption") != consumption
        ):
            raise G0cQueueError(
                "Table-A final consumption differs from the predeclared instance"
            )
        gate = planned.get("evaluation_plan_contract", {}).get("final_gate")
        if not isinstance(gate, Mapping):
            raise G0cQueueError("Table-A final completion lacks its gate binding")
        evidence.update(
            {
                "final_gate": _file_record(
                    Path(str(gate["path"])), "completed_final_gate"
                ),
                "final_consumption": _file_record(
                    consumption_path, "completed_final_consumption"
                ),
            }
        )
    return evidence


def _verify_native_completion(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    if queue["plan"]["queue_kind"] == TRAINING_KIND:
        return _verify_training_completion(queue, item)
    return _verify_evaluation_completion(queue, item)


def _completion_evidence(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    native = _verify_native_completion(queue, item)
    job_dir = _job_dir(queue, item)
    return {
        "schema": "pivot.stageb.table_a_g0c_queue_completion_evidence/v1",
        "verified_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "queue_kind": queue["plan"]["queue_kind"],
        "item_index": item["index"],
        "run_id": item["run_id"],
        "job_id": item["job_id"],
        "job_dir": str(job_dir),
        "child_process_identity": copy.deepcopy(item["child_process_identity"]),
        "native_completion": native,
    }


def _release_completed_queue_lease(queue: Mapping[str, Any]) -> None:
    """Release only after durable reload and a fresh replay of every item."""

    queue_dir = Path(str(queue["plan"]["queue_dir"]))
    persisted = load_queue(queue_dir)
    if persisted.get("status") != "completed":
        raise G0cQueueError("GPU lease release requires a completed durable queue")
    _verify_plan_closure(persisted)
    for item in persisted["items"]:
        if item.get("status") != "completed":
            raise G0cQueueError(
                "GPU lease release found an incomplete Table-A queue item"
            )
        _validate_job_binding(persisted, item, terminal=True)
        replay = _completion_evidence(persisted, item)
        evidence = item.get("completion_evidence")
        if (
            not isinstance(evidence, Mapping)
            or _strip_volatile(evidence) != _strip_volatile(replay)
        ):
            raise G0cQueueError(
                f"{item['run_id']} completion evidence drifted before lease release"
            )
    lease_path = Path(str(persisted["plan"]["lease_path"]))
    if not lease_path.is_file():
        return
    lease = _read_json(lease_path, label="shared GPU lease")
    if lease.get("queue_id") != persisted["plan"]["queue_id"]:
        return
    _clear_lease(persisted)


def _mark_completed(
    queue: MutableMapping[str, Any], index: int, evidence: Mapping[str, Any]
) -> None:
    item = queue["items"][index]
    job_dir = _job_dir(queue, item)
    launch, status = _validate_job_binding(queue, item, terminal=False)
    item["status"] = "completed"
    item["completed_at_utc"] = _utc_now()
    item["completion_evidence"] = copy.deepcopy(dict(evidence))
    evidence_sha = _canonical_sha256(evidence)
    launch.update(
        {
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "completion_evidence_sha256": evidence_sha,
        }
    )
    status.update(
        {
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "completion_evidence_sha256": evidence_sha,
        }
    )
    _write_json(job_dir / "launch.json", launch)
    _write_json(job_dir / "status.json", status)
    queue["active_item"] = None
    _event(queue, "item_completed", index=index, run_id=item["run_id"])
    if all(candidate["status"] == "completed" for candidate in queue["items"]):
        queue["status"] = "completed"
        queue["completed_at_utc"] = _utc_now()
        _event(queue, "queue_completed")
    _save_queue(queue)
    if queue["status"] == "completed":
        _release_completed_queue_lease(queue)


def _advance_launched(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_plan_closure(queue)
    item = queue["items"][index]
    _ensure_lease(queue, item, create=False)
    _validate_job_binding(queue, item, terminal=False)
    running = shared_queue._process_running(
        item.get("child_pid"), item.get("child_process_identity")
    )
    if running is True:
        status_path = _job_dir(queue, item) / "status.json"
        status = _read_json(status_path, label="G0c job status")
        status["updated_at_utc"] = _utc_now()
        status["last_observation"] = "child_running"
        _write_json(status_path, status)
        return
    if running is None:
        raise G0cQueueError("G0c child liveness is not observable")
    evidence = _completion_evidence(queue, item)
    _mark_completed(queue, index, evidence)


def _fail_queue(
    queue: MutableMapping[str, Any], index: int, *, error: BaseException | str
) -> None:
    item = queue["items"][index]
    rendered = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    if not isinstance(item.get("job_id"), str):
        item["job_id"] = f"failure-{uuid.uuid4().hex[:12]}"
        item["job_dir"] = str(_job_dir(queue, item))
    job_dir = _job_dir(queue, item)
    job_dir.mkdir(parents=True, exist_ok=True)
    identity = _job_identity(queue, item)
    launch_path = job_dir / "launch.json"
    status_path = job_dir / "status.json"
    launch = (
        _read_json(launch_path, label="failed G0c job launch")
        if launch_path.is_file()
        else {
            "schema": JOB_LAUNCH_SCHEMA,
            **identity,
            "command": copy.deepcopy(_planned_item(queue, index)["command"]),
            "output_root": _planned_item(queue, index)["output_root"],
        }
    )
    status = (
        _read_json(status_path, label="failed G0c job status")
        if status_path.is_file()
        else {"schema": JOB_STATUS_SCHEMA, **identity}
    )
    launch.update({"status": "failed", "failed_at_utc": _utc_now(), "error": rendered})
    status.update(
        {
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "error": rendered,
        }
    )
    _write_json(launch_path, launch)
    _write_json(status_path, status)
    item["status"] = "failed"
    item["failed_at_utc"] = _utc_now()
    item["failure_error"] = rendered
    queue["status"] = "failed"
    queue["active_item"] = _active_projection(item)
    lease_path = Path(str(queue["plan"]["lease_path"]))
    lease_retained = False
    if lease_path.is_file():
        with contextlib.suppress(G0cQueueError):
            lease_retained = (
                _read_json(lease_path, label="shared GPU lease").get("queue_id")
                == queue["plan"]["queue_id"]
            )
    queue["failure"] = {
        "index": index,
        "run_id": item["run_id"],
        "error": rendered,
        "lease_retained_fail_closed": lease_retained,
    }
    _event(queue, "queue_failed", index=index, run_id=item["run_id"], error=rendered)
    _save_queue(queue)


def advance_once(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    if queue["status"] == "completed":
        _release_completed_queue_lease(queue)
        return load_queue(queue_dir)
    if queue["status"] == "failed":
        return queue
    index = next(
        (
            index
            for index, item in enumerate(queue["items"])
            if item["status"] != "completed"
        ),
        None,
    )
    if index is None:
        raise G0cQueueError("G0c queue has no incomplete item but is not completed")
    item = queue["items"][index]
    try:
        if item["status"] == "pending":
            _reserve(queue, index)
        elif item["status"] == "reserved":
            _prepare_job(queue, index)
        elif item["status"] == "launching":
            _launch_or_recover(queue, index)
        elif item["status"] == "launched":
            _advance_launched(queue, index)
        else:
            raise G0cQueueError(f"cannot advance G0c item in {item['status']!r}")
    except (G0cQueueBusy, KeyboardInterrupt):
        raise
    except BaseException as exc:
        current = load_queue(queue_dir)
        if current["status"] == "completed":
            raise
        current_index = next(
            (
                value
                for value, candidate in enumerate(current["items"])
                if candidate["status"] != "completed"
            ),
            index,
        )
        if current["items"][current_index]["status"] != "failed":
            _fail_queue(current, current_index, error=exc)
        return load_queue(queue_dir)
    return load_queue(queue_dir)


def run_queue(
    queue_dir: Path, *, poll_seconds: float = 30.0, once: bool = False
) -> dict[str, Any]:
    if poll_seconds < 0.05:
        raise G0cQueueError("poll interval must be at least 0.05 seconds")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        lock = shared_queue._exclusive_file_lock(
            queue_dir / "supervisor.lock",
            busy_message=f"another G0c queue supervisor is active: {queue_dir}",
        )
        with lock:
            while True:
                queue = advance_once(queue_dir)
                if once or queue["status"] in {"completed", "failed"}:
                    return queue
                time.sleep(poll_seconds)
    except shared_queue.QueueBusyError as exc:
        raise G0cQueueBusy(str(exc)) from exc


def queue_status(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    counts = {
        status: sum(item["status"] == status for item in queue["items"])
        for status in ITEM_STATUSES
    }
    lease_path = Path(str(queue["plan"]["lease_path"]))
    return {
        "schema": "pivot.stageb.table_a_g0c_queue_status/v1",
        "observed_at_utc": _utc_now(),
        "queue_kind": queue["plan"]["queue_kind"],
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "status": queue["status"],
        "revision": queue["revision"],
        "ordered_run_ids": [item["run_id"] for item in queue["items"]],
        "counts": counts,
        "active_item": copy.deepcopy(queue.get("active_item")),
        "failure": copy.deepcopy(queue.get("failure")),
        "lease_path": str(lease_path),
        "lease_present": lease_path.is_file(),
    }


def verify_queue(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    errors = []
    verified = []
    try:
        _verify_plan_closure(queue)
    except G0cQueueError as exc:
        errors.append({"scope": "plan_closure", "error": str(exc)})
    for item in queue["items"]:
        if item["status"] != "completed":
            errors.append(
                {"run_id": item["run_id"], "error": f"status={item['status']}"}
            )
            continue
        try:
            _validate_job_binding(queue, item, terminal=True)
            replay = _completion_evidence(queue, item)
            persisted = item.get("completion_evidence")
            if not isinstance(persisted, Mapping) or _strip_volatile(
                persisted
            ) != _strip_volatile(replay):
                raise G0cQueueError("persisted completion evidence differs from replay")
            verified.append(replay)
        except (G0cQueueError, OSError, ValueError) as exc:
            errors.append({"run_id": item["run_id"], "error": str(exc)})
    lease_path = Path(str(queue["plan"]["lease_path"]))
    if queue["status"] == "completed" and lease_path.exists():
        try:
            lease = _read_json(lease_path, label="shared GPU lease")
        except G0cQueueError as exc:
            errors.append({"scope": "gpu_lease", "error": str(exc)})
        else:
            if lease.get("queue_id") == queue["plan"]["queue_id"]:
                errors.append(
                    {
                        "scope": "gpu_lease",
                        "error": "completed queue retained its GPU lease",
                    }
                )
    passed = queue["status"] == "completed" and not errors
    return {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "status": "passed" if passed else "failed",
        "queue_status": queue["status"],
        "queue_kind": queue["plan"]["queue_kind"],
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "ordered_run_ids": [item["run_id"] for item in queue["items"]],
        "verified_items": verified,
        "errors": errors,
    }


def verify_training_run(
    queue_dir: Path, seed: int, *, require_canonical_path: bool = True
) -> dict[str, Any]:
    seed = int(seed)
    if seed not in FORMAL_SEEDS:
        raise G0cQueueError("G0c training attestation seed is not predeclared")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    if require_canonical_path and queue_dir != DEFAULT_TRAINING_QUEUE_DIR.resolve(
        strict=True
    ):
        raise G0cQueueError("G0c training queue path is not canonical")
    verification = verify_queue(queue_dir)
    if (
        verification["status"] != "passed"
        or verification["queue_kind"] != TRAINING_KIND
        or tuple(verification["ordered_run_ids"]) != TRAINING_RUN_IDS
    ):
        raise G0cQueueError("G0c training queue does not pass the exact contract")
    queue = load_queue(queue_dir)
    matches = [item for item in queue["items"] if item["run_id"] == f"G0c:{seed}"]
    if len(matches) != 1 or matches[0]["status"] != "completed":
        raise G0cQueueError(f"G0c:{seed} is not uniquely completed")
    item = matches[0]
    native = item["completion_evidence"]["native_completion"]
    planned = _planned_item(queue, int(item["index"]))
    if (
        Path(str(native["output_root"])).resolve(strict=True)
        != training_runner.formal_output_root(seed).resolve(strict=True)
        or native["plan_contract_sha256"] != planned["expected_plan_sha256"]
    ):
        raise G0cQueueError("G0c completed training identity drifted")
    job_dir = _job_dir(queue, item)
    return {
        "schema": TRAINING_ATTESTATION_SCHEMA,
        "status": "passed",
        "seed": seed,
        "run_id": item["run_id"],
        "queue_id": queue["plan"]["queue_id"],
        "queue_plan_sha256": queue["plan_sha256"],
        "queue_manifest": str((queue_dir / "queue.json").resolve(strict=True)),
        "job_id": item["job_id"],
        "job_dir": str(job_dir),
        "job_launch": str((job_dir / "launch.json").resolve(strict=True)),
        "job_status": str((job_dir / "status.json").resolve(strict=True)),
        "output_root": native["output_root"],
        "formal_plan": native["formal_plan"],
        "postflight": native["postflight"],
        "checkpoint": native["checkpoint"],
    }


def dry_run(queue_kind: str) -> dict[str, Any]:
    queue_dir = _canonical_queue_dir(queue_kind)
    try:
        plan = build_queue_plan(queue_kind, queue_dir)
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        G0cQueueError,
        ValueError,
    ) as exc:
        return {
            "status": "blocked",
            "queue_kind": queue_kind,
            "queue_dir": str(queue_dir),
            "reason": str(exc),
            "artifact_audit": audit_existing_artifacts(),
            "mutated": False,
        }
    return {
        "status": "ready",
        "queue_kind": queue_kind,
        "queue_dir": str(queue_dir),
        "queue_id": plan["queue_id"],
        "plan_sha256": _canonical_sha256(plan),
        "ordered_run_ids": [item["run_id"] for item in plan["items"]],
        "controller_source_count": len(plan["controller_sources"]),
        "input_record_counts": {
            item["run_id"]: len(item["input_records"]) for item in plan["items"]
        },
        "mutated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("audit")
    dry = modes.add_parser("dry-run")
    dry.add_argument("queue_kind", choices=QUEUE_KINDS)
    create = modes.add_parser("create")
    create.add_argument("queue_kind", choices=QUEUE_KINDS)
    create.add_argument("--queue-dir", type=Path)
    create.add_argument("--gpu-key", default="0")
    reconcile = modes.add_parser("reconcile")
    reconcile.add_argument("queue_dir", type=Path)
    run = modes.add_parser("run")
    run.add_argument("queue_dir", type=Path)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    status = modes.add_parser("status")
    status.add_argument("queue_dir", type=Path)
    verify = modes.add_parser("verify")
    verify.add_argument("queue_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "audit":
            result = audit_existing_artifacts()
            code = 0
        elif args.mode == "dry-run":
            result = dry_run(args.queue_kind)
            code = 0
        elif args.mode == "create":
            expected = _canonical_queue_dir(args.queue_kind)
            queue_dir = expected if args.queue_dir is None else args.queue_dir
            result = create_queue(
                args.queue_kind,
                queue_dir,
                gpu_key=args.gpu_key,
            )
            code = 0
        elif args.mode == "reconcile":
            result = run_queue(args.queue_dir, once=True)
            code = 0 if result["status"] != "failed" else 1
        elif args.mode == "run":
            result = run_queue(
                args.queue_dir, poll_seconds=args.poll_seconds, once=False
            )
            code = 0 if result["status"] != "failed" else 1
        elif args.mode == "status":
            result = queue_status(args.queue_dir)
            code = 0
        elif args.mode == "verify":
            result = verify_queue(args.queue_dir)
            code = 0 if result["status"] == "passed" else 1
        else:  # pragma: no cover
            parser.error(f"unknown mode: {args.mode}")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return code
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        G0cQueueError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
