#!/usr/bin/env python3
"""Run the canonical Table-A G0c U50 soak under the shared GPU lease.

This is a one-item durable queue for the prerequisite that gates formal G0c
training.  It deliberately accepts only the canonical seed-17 B10xA4/U50
soak.  Existing plans, outputs, or seals are never adopted at queue creation.
The queue binds an exact child process/session identity, survives supervisor
restarts, replays the native postflight, and releases the repository-wide GPU
lease only after a durable completion receipt has been reloaded and verified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_serial_matrix_queue as shared_queue  # noqa: E402
from tools import run_stageb_table_a_controls as training_runner  # noqa: E402
from tools.stageb_dependency_audit import local_python_dependency_paths  # noqa: E402


QUEUE_SCHEMA = "pivot.stageb.table_a.g0c_soak_queue/v1"
PLAN_SCHEMA = "pivot.stageb.table_a.g0c_soak_queue_plan/v1"
JOB_LAUNCH_SCHEMA = "pivot.stageb.table_a.g0c_soak_job_launch/v1"
JOB_STATUS_SCHEMA = "pivot.stageb.table_a.g0c_soak_job_status/v1"
CHILD_STATUS_SCHEMA = "pivot.stageb.table_a.g0c_soak_child_status/v1"
SEAL_INTENT_SCHEMA = "pivot.stageb.table_a.g0c_soak_seal_intent/v1"
COMPLETION_CANDIDATE_SCHEMA = (
    "pivot.stageb.table_a.g0c_soak_completion_candidate/v1"
)
COMPLETION_SCHEMA = "pivot.stageb.table_a.g0c_soak_completion/v1"
VERIFICATION_SCHEMA = "pivot.stageb.table_a.g0c_soak_verification/v1"
AUDIT_SCHEMA = "pivot.stageb.table_a.g0c_soak_artifact_audit/v1"

RUN_ID = "G0c-soak:seed17:b10a4:u50"
SEED = 17
MICRO_BATCH_SIZE = 10
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_GLOBAL_BATCH = 40
OPTIMIZER_UPDATES = 50
DEFAULT_NUM_WORKERS = 8

DEFAULT_QUEUE_DIR = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_a_g0c_soak_u50_v1"
)
DEFAULT_LEASE_ROOT = shared_queue.DEFAULT_LEASE_ROOT
DEFAULT_PYTHON = Path(
    os.environ.get(
        "PIVOT_PYTHON", "/home/haoyi/miniconda/envs/gdino5090/bin/python"
    )
)
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))

ITEM_STATUSES = frozenset(
    {"pending", "reserved", "launching", "launched", "completed", "failed"}
)
ACTIVE_STATUSES = frozenset(
    {"reserved", "launching", "launched", "failed"}
)
VOLATILE_KEYS = frozenset(
    {
        "created_at_utc",
        "updated_at_utc",
        "started_at_utc",
        "finished_at_utc",
        "completed_at_utc",
        "failed_at_utc",
        "verified_at_utc",
        "validated_at_utc",
        "sealed_at_utc",
        "observed_at_utc",
        "reserved_at_utc",
        "prepared_at_utc",
        "launching_at_utc",
        "launched_at_utc",
    }
)


class G0cSoakQueueError(RuntimeError):
    """The soak queue or its evidence violates the formal contract."""


class G0cSoakQueueBusy(G0cSoakQueueError):
    """A supervisor or another formal queue currently owns the resource."""


class G0cSoakLeaseLoss(G0cSoakQueueError):
    """An active soak no longer owns its predeclared GPU lease."""


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


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _file_record(path: Path, *roles: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise G0cSoakQueueError(f"queue input is not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "roles": sorted(set(roles)),
    }


def _verify_file_record(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise G0cSoakQueueError(f"{label} record is missing")
    try:
        path = Path(str(value.get("path", ""))).resolve(strict=True)
        size = int(path.stat().st_size)
    except (OSError, TypeError, ValueError) as exc:
        raise G0cSoakQueueError(f"{label} is unavailable: {exc}") from exc
    if (
        not path.is_file()
        or int(value.get("size_bytes", -1)) != size
        or str(value.get("sha256", "")) != _sha256_file(path)
    ):
        raise G0cSoakQueueError(f"{label} changed after queue planning: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise G0cSoakQueueError(f"{label} is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G0cSoakQueueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise G0cSoakQueueError(f"{label} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    shared_queue._write_json_atomic(path, value)


def _write_json_fresh(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if _path_present(path):
            raise FileExistsError(f"refuse to overwrite queue artifact: {path}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


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
        raise G0cSoakQueueError(f"invalid G0c soak Python: {exc}") from exc
    if selected != current:
        raise G0cSoakQueueError(
            "G0c soak planning must run under the selected Python: "
            f"caller={current}, selected={selected}"
        )
    return selected


def _runtime_contract(
    *,
    python: Path = DEFAULT_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    gpu_key: str = "0",
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, Any]:
    python = _resolve_python(python)
    data_root = data_root.expanduser().resolve(strict=True)
    gpu_key = str(gpu_key).strip()
    if not data_root.is_dir():
        raise G0cSoakQueueError(f"G0c soak data root is not a directory: {data_root}")
    if not gpu_key or "," in gpu_key:
        raise G0cSoakQueueError("G0c soak requires exactly one GPU key")
    if int(num_workers) < 0:
        raise G0cSoakQueueError("G0c soak num_workers must be non-negative")
    return {
        "python": str(python),
        "python_record": _file_record(python, "queue_python_runtime"),
        "data_root": str(data_root),
        "gpu_key": gpu_key,
        "cuda_visible_devices": gpu_key,
        "num_workers": int(num_workers),
        "seed": SEED,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": OPTIMIZER_UPDATES,
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


def _soak_args(runtime: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        purpose="soak",
        python=str(runtime["python"]),
        checkpoint=str(training_runner.DEFAULT_CHECKPOINT),
        batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        effective_batch_size=EFFECTIVE_GLOBAL_BATCH,
        updates=OPTIMIZER_UPDATES,
        seed=SEED,
        num_workers=int(runtime["num_workers"]),
        output_dir=str(training_runner.DEFAULT_SOAK_ROOT),
        plan_json=str(training_runner.DEFAULT_SOAK_PLAN),
        soak_seal=str(training_runner.DEFAULT_SOAK_SEAL),
        cuda_visible_devices=str(runtime["cuda_visible_devices"]),
    )


def _soak_command(runtime: Mapping[str, Any]) -> list[str]:
    return [
        str(runtime["python"]),
        str(Path(training_runner.__file__).resolve()),
        "run",
        "--purpose",
        "soak",
        "--python",
        str(runtime["python"]),
        "--checkpoint",
        str(training_runner.DEFAULT_CHECKPOINT),
        "--batch-size",
        str(MICRO_BATCH_SIZE),
        "--gradient-accumulation-steps",
        str(GRADIENT_ACCUMULATION_STEPS),
        "--effective-batch-size",
        str(EFFECTIVE_GLOBAL_BATCH),
        "--updates",
        str(OPTIMIZER_UPDATES),
        "--seed",
        str(SEED),
        "--num-workers",
        str(int(runtime["num_workers"])),
        "--output-dir",
        str(training_runner.DEFAULT_SOAK_ROOT),
        "--plan-json",
        str(training_runner.DEFAULT_SOAK_PLAN),
        "--cuda-visible-devices",
        str(runtime["cuda_visible_devices"]),
    ]


def _expected_soak_plan(runtime: Mapping[str, Any]) -> dict[str, Any]:
    try:
        plan = training_runner.build_plan(_soak_args(runtime))
        training_runner._validate_plan_identity(plan)
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        raise G0cSoakQueueError(f"cannot build exact G0c U50 soak plan: {exc}") from exc
    contract = plan.get("matched_contract")
    expected = {
        "seed": SEED,
        "micro_batch_size_per_rank": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": OPTIMIZER_UPDATES,
    }
    if (
        plan.get("row_id") != "G0c"
        or plan.get("purpose") != "soak"
        or not isinstance(contract, Mapping)
        or any(contract.get(key) != value for key, value in expected.items())
        or Path(str(plan.get("output_dir", ""))).resolve(strict=False)
        != training_runner.DEFAULT_SOAK_ROOT.resolve(strict=False)
    ):
        raise G0cSoakQueueError("native G0c U50 soak plan contract drifted")
    return dict(plan)


def _records_from_soak_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: dict[Path, dict[str, Any]] = {}
    inputs = plan.get("inputs")
    tree = plan.get("source_dependency_tree")
    if not isinstance(inputs, Mapping) or not isinstance(tree, Mapping):
        raise G0cSoakQueueError("G0c soak plan lacks input/source closure")
    values: list[tuple[Mapping[str, Any], str]] = []
    for label, value in inputs.items():
        if not isinstance(value, Mapping):
            raise G0cSoakQueueError(f"G0c soak input {label} is invalid")
        values.append((value, f"soak_input:{label}"))
    tree_records = tree.get("records")
    if not isinstance(tree_records, list) or not tree_records:
        raise G0cSoakQueueError("G0c soak source closure is empty")
    for value in tree_records:
        if not isinstance(value, Mapping):
            raise G0cSoakQueueError("G0c soak source record is invalid")
        values.append((value, "soak_source_dependency"))
    for value, role in values:
        path = Path(str(value.get("path", ""))).resolve(strict=True)
        record = _file_record(path, role)
        if record["sha256"] != value.get("sha256"):
            raise G0cSoakQueueError(f"G0c soak plan input is already stale: {path}")
        existing = records.get(path)
        if existing is None:
            records[path] = record
        elif existing["sha256"] != record["sha256"]:
            raise G0cSoakQueueError(f"G0c soak plan closure conflicts at {path}")
        else:
            existing["roles"] = sorted(set(existing["roles"]).union({role}))
    return [records[path] for path in sorted(records, key=str)]


def _controller_source_paths() -> tuple[Path, ...]:
    try:
        return tuple(
            local_python_dependency_paths([Path(__file__).resolve()], root=REPO_ROOT)
        )
    except Exception as exc:
        raise G0cSoakQueueError(
            f"G0c soak controller dependency audit failed: {exc}"
        ) from exc


def _controller_source_records() -> list[dict[str, Any]]:
    return [
        _file_record(path, "g0c_soak_queue_controller_source")
        for path in _controller_source_paths()
    ]


def _canonical_artifact_paths() -> dict[str, Path]:
    return {
        "plan": training_runner.DEFAULT_SOAK_PLAN.resolve(strict=False),
        "output_root": training_runner.DEFAULT_SOAK_ROOT.resolve(strict=False),
        "seal": training_runner.DEFAULT_SOAK_SEAL.resolve(strict=False),
    }


def _require_fresh_artifacts(paths: Mapping[str, Path] | None = None) -> None:
    selected = _canonical_artifact_paths() if paths is None else paths
    present = [f"{label}={path}" for label, path in selected.items() if _path_present(path)]
    if present:
        raise G0cSoakQueueError(
            "canonical G0c U50 soak artifacts must be fresh; no adoption is allowed: "
            + ", ".join(present)
        )


def audit_existing_artifacts() -> dict[str, Any]:
    paths = _canonical_artifact_paths()
    canonical = {
        label: {"path": str(path), "present": _path_present(path)}
        for label, path in paths.items()
    }
    legacy = []
    plan_root = REPO_ROOT / "outputs/paper_cvpr_v1/plans"
    canonical_plan = paths["plan"]
    if plan_root.is_dir():
        for path in sorted(plan_root.glob("table_a_g0c*.json")):
            resolved = path.resolve(strict=False)
            if resolved == canonical_plan:
                continue
            legacy.append(
                {
                    "path": str(resolved),
                    "status": "non_adoptable",
                    "reason": "not the fresh queue-owned canonical v2 U50 soak plan",
                }
            )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "passed",
        "audited_at_utc": _utc_now(),
        "fresh_only_policy": "all canonical plan/output/seal paths must be absent",
        "canonical_artifacts": canonical,
        "canonical_artifacts_fresh": not any(
            value["present"] for value in canonical.values()
        ),
        "legacy_artifacts": legacy,
        "legacy_adoption_allowed": False,
        "canonical_queue": {
            "path": str(DEFAULT_QUEUE_DIR.resolve(strict=False)),
            "present": DEFAULT_QUEUE_DIR.exists(),
        },
    }


def build_queue_plan(
    queue_dir: Path = DEFAULT_QUEUE_DIR,
    *,
    queue_id: str | None = None,
    python: Path = DEFAULT_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    gpu_key: str = "0",
    num_workers: int = DEFAULT_NUM_WORKERS,
    require_canonical_path: bool = True,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    if require_canonical_path and queue_dir != DEFAULT_QUEUE_DIR.resolve(strict=False):
        raise G0cSoakQueueError(
            f"canonical G0c soak queue path is {DEFAULT_QUEUE_DIR.resolve(strict=False)}"
        )
    if _path_present(queue_dir):
        raise FileExistsError(f"G0c soak queue path must be fresh: {queue_dir}")
    _require_fresh_artifacts()
    runtime = _runtime_contract(
        python=python,
        data_root=data_root,
        gpu_key=gpu_key,
        num_workers=num_workers,
    )
    expected_plan = _expected_soak_plan(runtime)
    selected_id = str(uuid.uuid4()) if queue_id is None else str(queue_id)
    if not selected_id:
        raise G0cSoakQueueError("G0c soak queue ID must be non-empty")
    lease_root = DEFAULT_LEASE_ROOT.resolve(strict=False)
    paths = _canonical_artifact_paths()
    return {
        "schema": PLAN_SCHEMA,
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
        "items": [
            {
                "index": 0,
                "run_id": RUN_ID,
                "item_kind": "g0c_soak",
                "seed": SEED,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
                "optimizer_updates": OPTIMIZER_UPDATES,
                "output_root": str(paths["output_root"]),
                "soak_plan_path": str(paths["plan"]),
                "soak_seal_path": str(paths["seal"]),
                "expected_plan": expected_plan,
                "expected_plan_sha256": expected_plan["plan_sha256"],
                "training_command": _soak_command(runtime),
                "input_records": _records_from_soak_plan(expected_plan),
            }
        ],
    }


def _validate_plan(plan: Mapping[str, Any], queue_dir: Path) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise G0cSoakQueueError("G0c soak queue immutable plan schema drifted")
    if Path(str(plan.get("queue_dir", ""))).resolve(strict=False) != queue_dir:
        raise G0cSoakQueueError("G0c soak queue was opened outside its planned path")
    if Path(str(plan.get("repository_root", ""))).resolve(strict=False) != REPO_ROOT:
        raise G0cSoakQueueError("G0c soak repository root drifted")
    if not isinstance(plan.get("queue_id"), str) or not plan["queue_id"]:
        raise G0cSoakQueueError("G0c soak queue ID is invalid")
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise G0cSoakQueueError("G0c soak runtime is missing")
    exact_runtime = {
        "seed": SEED,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": OPTIMIZER_UPDATES,
    }
    if any(runtime.get(key) != value for key, value in exact_runtime.items()):
        raise G0cSoakQueueError("G0c soak exact runtime contract drifted")
    if (
        runtime.get("gpu_key") != plan.get("gpu_key")
        or runtime.get("cuda_visible_devices") != plan.get("gpu_key")
        or plan.get("runtime_environment") != _runtime_environment(runtime)
    ):
        raise G0cSoakQueueError("G0c soak runtime/GPU binding drifted")
    python_record = runtime.get("python_record")
    if (
        not isinstance(python_record, Mapping)
        or Path(str(python_record.get("path", ""))).resolve(strict=False)
        != Path(str(runtime.get("python", ""))).resolve(strict=False)
    ):
        raise G0cSoakQueueError("G0c soak Python binding drifted")
    lease_root = Path(str(plan.get("lease_root", ""))).resolve(strict=False)
    lease_path = Path(str(plan.get("lease_path", ""))).resolve(strict=False)
    if lease_path != shared_queue._lease_path(
        lease_root, str(plan.get("gpu_key", ""))
    ).resolve(strict=False):
        raise G0cSoakQueueError("G0c soak shared GPU lease path drifted")
    items = plan.get("items")
    if not isinstance(items, list) or len(items) != 1:
        raise G0cSoakQueueError("G0c soak queue must contain exactly one item")
    item = items[0]
    if not isinstance(item, Mapping):
        raise G0cSoakQueueError("G0c soak item is invalid")
    exact_item = {
        "index": 0,
        "run_id": RUN_ID,
        "item_kind": "g0c_soak",
        "seed": SEED,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": OPTIMIZER_UPDATES,
    }
    if any(item.get(key) != value for key, value in exact_item.items()):
        raise G0cSoakQueueError("G0c soak immutable item contract drifted")
    expected_plan = item.get("expected_plan")
    if not isinstance(expected_plan, Mapping):
        raise G0cSoakQueueError("G0c soak expected native plan is missing")
    try:
        training_runner._validate_plan_identity(expected_plan)
    except (KeyError, TypeError, ValueError) as exc:
        raise G0cSoakQueueError(f"G0c soak expected plan identity failed: {exc}") from exc
    contract = expected_plan.get("matched_contract")
    expected_contract = {
        "seed": SEED,
        "micro_batch_size_per_rank": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": OPTIMIZER_UPDATES,
    }
    paths = {
        "output_root": Path(str(item.get("output_root", ""))).resolve(strict=False),
        "plan": Path(str(item.get("soak_plan_path", ""))).resolve(strict=False),
        "seal": Path(str(item.get("soak_seal_path", ""))).resolve(strict=False),
    }
    command = item.get("training_command")
    records = item.get("input_records")
    sources = plan.get("controller_sources")
    if (
        expected_plan.get("row_id") != "G0c"
        or expected_plan.get("purpose") != "soak"
        or not isinstance(contract, Mapping)
        or any(contract.get(key) != value for key, value in expected_contract.items())
        or item.get("expected_plan_sha256") != expected_plan.get("plan_sha256")
        or Path(str(expected_plan.get("output_dir", ""))).resolve(strict=False)
        != paths["output_root"]
        or not isinstance(command, list)
        or len(command) < 3
        or command[0] != runtime.get("python")
        or Path(str(command[1])).resolve(strict=False)
        != Path(training_runner.__file__).resolve(strict=False)
        or command[2] != "run"
        or not isinstance(records, list)
        or not records
        or not isinstance(sources, list)
        or not sources
    ):
        raise G0cSoakQueueError("G0c soak native plan/command closure drifted")
    for label, values in (("controller", sources), ("input", records)):
        observed_paths = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise G0cSoakQueueError(f"G0c soak {label} record {index} is invalid")
            path = Path(str(value.get("path", ""))).resolve(strict=False)
            if re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None:
                raise G0cSoakQueueError(f"G0c soak {label} record SHA is invalid")
            if int(value.get("size_bytes", -1)) < 0:
                raise G0cSoakQueueError(f"G0c soak {label} record size is invalid")
            observed_paths.append(path)
        if len(set(observed_paths)) != len(observed_paths):
            raise G0cSoakQueueError(f"G0c soak {label} closure has duplicates")
    if queue_dir == DEFAULT_QUEUE_DIR.resolve(strict=False):
        canonical = _canonical_artifact_paths()
        if (
            lease_root != DEFAULT_LEASE_ROOT.resolve(strict=False)
            or paths != canonical
        ):
            raise G0cSoakQueueError("canonical G0c soak paths/lease drifted")


def _validate_creation_attestation(plan: Mapping[str, Any]) -> None:
    runtime = plan["runtime"]
    item = plan["items"][0]
    current_plan = _expected_soak_plan(runtime)
    if current_plan != item["expected_plan"]:
        raise G0cSoakQueueError("G0c soak native plan changed during queue creation")
    if _controller_source_records() != plan["controller_sources"]:
        raise G0cSoakQueueError("G0c soak controller closure changed during queue creation")
    if _records_from_soak_plan(current_plan) != item["input_records"]:
        raise G0cSoakQueueError("G0c soak input closure changed during queue creation")


def _validate_queue(queue: Mapping[str, Any], queue_dir: Path) -> None:
    if queue.get("schema") != QUEUE_SCHEMA:
        raise G0cSoakQueueError("unsupported G0c soak queue schema")
    plan = queue.get("plan")
    if not isinstance(plan, Mapping):
        raise G0cSoakQueueError("G0c soak immutable plan is missing")
    _validate_plan(plan, queue_dir)
    if queue.get("plan_sha256") != _canonical_sha256(plan):
        raise G0cSoakQueueError("G0c soak immutable plan SHA-256 mismatch")
    status = queue.get("status")
    if status not in {"planned", "running", "completed", "failed"}:
        raise G0cSoakQueueError(f"G0c soak queue status is invalid: {status!r}")
    items = queue.get("items")
    if not isinstance(items, list) or len(items) != 1:
        raise G0cSoakQueueError("G0c soak mutable item set drifted")
    item = items[0]
    if (
        not isinstance(item, Mapping)
        or item.get("index") != 0
        or item.get("run_id") != RUN_ID
        or item.get("status") not in ITEM_STATUSES
    ):
        raise G0cSoakQueueError("G0c soak mutable item identity/status drifted")
    active = item.get("status") in ACTIVE_STATUSES
    if active:
        if not isinstance(item.get("job_id"), str) or not item["job_id"]:
            raise G0cSoakQueueError("active G0c soak lacks a job ID")
        if Path(str(item.get("job_dir", ""))).resolve(strict=False) != _job_dir(
            queue, item
        ):
            raise G0cSoakQueueError("active G0c soak job directory drifted")
        if queue.get("active_item") != _active_projection(item):
            raise G0cSoakQueueError("G0c soak active-item projection drifted")
    elif queue.get("active_item") is not None:
        raise G0cSoakQueueError("G0c soak has a stale active-item projection")
    if status == "planned" and item["status"] != "pending":
        raise G0cSoakQueueError("planned G0c soak already has an active item")
    if status == "running" and item["status"] not in {
        "reserved",
        "launching",
        "launched",
    }:
        raise G0cSoakQueueError("running G0c soak has no running item")
    if status == "completed" and item["status"] != "completed":
        raise G0cSoakQueueError("completed G0c soak lacks a completed item")
    if status == "failed" and item["status"] != "failed":
        raise G0cSoakQueueError("failed G0c soak lacks a failed item")


def create_queue_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(dict(plan))
    queue_dir = Path(str(plan.get("queue_dir", ""))).resolve(strict=False)
    if _path_present(queue_dir):
        raise FileExistsError(f"G0c soak queue path must be fresh: {queue_dir}")
    _validate_plan(plan, queue_dir)
    _validate_creation_attestation(plan)
    item_plan = plan["items"][0]
    _require_fresh_artifacts(
        {
            "plan": Path(item_plan["soak_plan_path"]),
            "output_root": Path(item_plan["output_root"]),
            "seal": Path(item_plan["soak_seal_path"]),
        }
    )
    now = _utc_now()
    queue = {
        "schema": QUEUE_SCHEMA,
        "status": "planned",
        "created_at_utc": now,
        "updated_at_utc": now,
        "revision": 0,
        "plan": plan,
        "plan_sha256": _canonical_sha256(plan),
        "items": [{"index": 0, "run_id": RUN_ID, "status": "pending"}],
        "active_item": None,
        "events": [{"at_utc": now, "event": "queue_created", "run_id": RUN_ID}],
    }
    queue_dir.mkdir(parents=True, exist_ok=False)
    _write_json(queue_dir / "queue.json", queue)
    return load_queue(queue_dir)


def create_queue(**kwargs: Any) -> dict[str, Any]:
    return create_queue_from_plan(build_queue_plan(**kwargs))


def load_queue(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    queue = _read_json(queue_dir / "queue.json", label="G0c soak queue")
    _validate_queue(queue, queue_dir)
    return queue


def _save_queue(queue: MutableMapping[str, Any]) -> None:
    queue_dir = Path(str(queue["plan"]["queue_dir"])).resolve(strict=True)
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_at_utc"] = _utc_now()
    _validate_queue(queue, queue_dir)
    _write_json(queue_dir / "queue.json", queue)


def _event(queue: MutableMapping[str, Any], event: str, **fields: Any) -> None:
    values = queue.setdefault("events", [])
    if not isinstance(values, list):
        raise G0cSoakQueueError("G0c soak event ledger is invalid")
    values.append({"at_utc": _utc_now(), "event": event, **fields})


def _planned_item(queue: Mapping[str, Any]) -> Mapping[str, Any]:
    return queue["plan"]["items"][0]


def _job_dir(queue: Mapping[str, Any], item: Mapping[str, Any]) -> Path:
    job_id = item.get("job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) is None:
        raise G0cSoakQueueError("G0c soak job ID is invalid")
    return (
        Path(str(queue["plan"]["queue_dir"])) / "jobs" / job_id
    ).resolve(strict=False)


def _active_projection(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": 0,
        "run_id": RUN_ID,
        "job_id": item["job_id"],
        "job_dir": item["job_dir"],
    }


def _job_identity(queue: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "run_id": RUN_ID,
        "item_index": 0,
        "job_id": item["job_id"],
        "job_dir": item["job_dir"],
    }


def _child_command(queue: Mapping[str, Any], item: Mapping[str, Any]) -> list[str]:
    return [
        str(queue["plan"]["runtime"]["python"]),
        str(Path(__file__).resolve()),
        "execute-child",
        "--queue-dir",
        str(Path(queue["plan"]["queue_dir"]).resolve(strict=False)),
        "--job-id",
        str(item["job_id"]),
    ]


def _verify_plan_closure(queue: Mapping[str, Any]) -> None:
    for label, values in (
        ("controller source", queue["plan"]["controller_sources"]),
        ("soak input", _planned_item(queue)["input_records"]),
    ):
        for index, value in enumerate(values):
            _verify_file_record(value, label=f"G0c soak {label} {index}")
    try:
        training_runner._validate_plan_identity(_planned_item(queue)["expected_plan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise G0cSoakQueueError(f"embedded G0c soak plan identity failed: {exc}") from exc


def _ensure_lease(queue: Mapping[str, Any], *, create: bool) -> None:
    try:
        shared_queue._ensure_lease(queue, queue["items"][0], create=create)
    except shared_queue.QueueLeaseOwnershipError as exc:
        if create and queue["items"][0]["status"] == "pending":
            raise G0cSoakQueueBusy(str(exc)) from exc
        raise G0cSoakLeaseLoss(str(exc)) from exc
    except shared_queue.QueueBusyError as exc:
        raise G0cSoakQueueBusy(str(exc)) from exc
    except shared_queue.QueueContractError as exc:
        if create and queue["items"][0]["status"] == "pending":
            raise G0cSoakQueueError(f"cannot acquire G0c soak GPU lease: {exc}") from exc
        raise G0cSoakLeaseLoss(str(exc)) from exc


def _owned_lease_present(queue: Mapping[str, Any]) -> bool:
    path = Path(str(queue["plan"]["lease_path"]))
    if not path.is_file():
        return False
    try:
        lease = _read_json(path, label="G0c soak GPU lease")
    except G0cSoakQueueError:
        return False
    return not shared_queue._lease_identity_mismatches(queue, lease)


def _clear_lease(queue: Mapping[str, Any]) -> None:
    try:
        shared_queue._clear_owned_lease(queue)
    except shared_queue.QueueBusyError as exc:
        raise G0cSoakQueueBusy(str(exc)) from exc
    except shared_queue.QueueContractError as exc:
        raise G0cSoakQueueError(f"cannot release G0c soak GPU lease: {exc}") from exc


def _reserve(queue: MutableMapping[str, Any]) -> None:
    _verify_plan_closure(queue)
    planned = _planned_item(queue)
    _require_fresh_artifacts(
        {
            "plan": Path(planned["soak_plan_path"]),
            "output_root": Path(planned["output_root"]),
            "seal": Path(planned["soak_seal_path"]),
        }
    )
    _ensure_lease(queue, create=True)
    item = queue["items"][0]
    job_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"
    item.update(
        {
            "status": "reserved",
            "job_id": job_id,
            "reserved_at_utc": _utc_now(),
        }
    )
    item["job_dir"] = str(_job_dir(queue, item))
    queue["status"] = "running"
    queue["active_item"] = _active_projection(item)
    _event(queue, "item_reserved", run_id=RUN_ID, job_id=job_id)
    _save_queue(queue)


def _ensure_prepared_record(
    path: Path, expected: Mapping[str, Any], *, label: str
) -> None:
    if path.is_file():
        observed = _read_json(path, label=label)
        if _strip_volatile(observed) != _strip_volatile(expected):
            raise G0cSoakQueueError(f"{label} differs from the recoverable job intent")
        return
    if _path_present(path):
        raise G0cSoakQueueError(f"{label} is not a regular file: {path}")
    _write_json_fresh(path, expected)


def _prepare_job(queue: MutableMapping[str, Any]) -> None:
    _verify_plan_closure(queue)
    _ensure_lease(queue, create=False)
    item = queue["items"][0]
    job_dir = _job_dir(queue, item)
    if _path_present(job_dir):
        if not job_dir.is_dir():
            raise G0cSoakQueueError(
                f"G0c soak job path is not a directory: {job_dir}"
            )
        unexpected = {
            path.name
            for path in job_dir.iterdir()
            if path.name not in {"launch.json", "status.json", "seal_intent.json"}
            and not path.name.startswith(".")
        }
        if unexpected:
            raise G0cSoakQueueError(
                "reserved G0c soak job contains pre-launch artifacts: "
                + ", ".join(sorted(unexpected))
            )
    else:
        job_dir.mkdir(parents=True, exist_ok=False)
    identity = _job_identity(queue, item)
    planned = _planned_item(queue)
    launch = {
        "schema": JOB_LAUNCH_SCHEMA,
        "status": "prepared",
        **identity,
        "child_command": _child_command(queue, item),
        "training_command": copy.deepcopy(planned["training_command"]),
        "output_root": planned["output_root"],
        "soak_plan_path": planned["soak_plan_path"],
        "soak_seal_path": planned["soak_seal_path"],
        "prepared_at_utc": _utc_now(),
    }
    status = {
        "schema": JOB_STATUS_SCHEMA,
        "status": "prepared",
        **identity,
        "updated_at_utc": _utc_now(),
    }
    seal_intent = {
        "schema": SEAL_INTENT_SCHEMA,
        "status": "authorized_after_exact_child_success",
        **identity,
        "expected_plan_sha256": planned["expected_plan_sha256"],
        "soak_plan_path": planned["soak_plan_path"],
        "soak_seal_path": planned["soak_seal_path"],
        "created_at_utc": _utc_now(),
    }
    _ensure_prepared_record(
        job_dir / "launch.json", launch, label="prepared G0c soak launch"
    )
    _ensure_prepared_record(
        job_dir / "status.json", status, label="prepared G0c soak status"
    )
    _ensure_prepared_record(
        job_dir / "seal_intent.json",
        seal_intent,
        label="prepared G0c soak seal intent",
    )
    item["status"] = "launching"
    item["launching_at_utc"] = _utc_now()
    _event(queue, "job_prepared", run_id=RUN_ID, job_id=item["job_id"])
    _save_queue(queue)


def _validate_process_identity(value: Any, *, label: str) -> dict[str, Any]:
    try:
        identity = shared_queue._validated_child_identity(
            value.get("pid") if isinstance(value, Mapping) else None, value
        )
    except shared_queue.QueueContractError as exc:
        raise G0cSoakQueueError(f"{label} process identity is invalid: {exc}") from exc
    if value.get("process_group_id") not in (None, identity["pid"]):
        raise G0cSoakQueueError(f"{label} process-group identity drifted")
    if value.get("session_id") not in (None, identity["pid"]):
        raise G0cSoakQueueError(f"{label} session identity drifted")
    return identity


def _validate_child_record(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    child = _read_json(_job_dir(queue, item) / "child.json", label="G0c soak child")
    if child.get("schema") != CHILD_STATUS_SCHEMA:
        raise G0cSoakQueueError("G0c soak child schema drifted")
    for key, expected in _job_identity(queue, item).items():
        if child.get(key) != expected:
            raise G0cSoakQueueError(f"G0c soak child identity drifted at {key}")
    if child.get("training_command") != _planned_item(queue)["training_command"]:
        raise G0cSoakQueueError("G0c soak child training command drifted")
    identity = _validate_process_identity(child.get("process_identity"), label="child")
    if (
        child.get("child_pid") != identity["pid"]
        or child.get("process_group_id") != identity["pid"]
        or child.get("session_id") != identity["pid"]
    ):
        raise G0cSoakQueueError("G0c soak child PID/session binding drifted")
    if child.get("status") not in {"running", "completed", "failed"}:
        raise G0cSoakQueueError("G0c soak child status is invalid")
    return child


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
                chunk.decode("utf-8", errors="replace")
                for chunk in (entry / "cmdline").read_bytes().split(b"\0")
                if chunk
            ]
        except OSError:
            continue
        if observed != expected:
            continue
        pid = int(entry.name)
        identity = shared_queue._read_process_identity(pid)
        if shared_queue._process_running(pid, identity) is True:
            matches.append((pid, _validate_process_identity(identity, label="matching child")))
    return matches


def _bind_child(
    queue: MutableMapping[str, Any],
    pid: int,
    identity: Mapping[str, Any],
) -> None:
    item = queue["items"][0]
    validated = _validate_process_identity(identity, label="bound child")
    if validated["pid"] != pid:
        raise G0cSoakQueueError("G0c soak child PID differs from its identity")
    running = shared_queue._process_running(pid, validated)
    if running is True:
        try:
            if os.getpgid(pid) != pid or os.getsid(pid) != pid:
                raise G0cSoakQueueError(
                    "G0c soak child is not its session/process-group leader"
                )
        except OSError as exc:
            raise G0cSoakQueueError(f"cannot inspect G0c soak child group: {exc}") from exc
    job_dir = _job_dir(queue, item)
    launch = _read_json(job_dir / "launch.json", label="G0c soak launch")
    status = _read_json(job_dir / "status.json", label="G0c soak status")
    process_identity = {
        **validated,
        "process_group_id": pid,
        "session_id": pid,
    }
    launch.update(
        {
            "status": "launched",
            "child_pid": pid,
            "child_process_identity": process_identity,
            "process_group_id": pid,
            "session_id": pid,
            "launched_at_utc": _utc_now(),
        }
    )
    status.update(
        {
            "status": "running",
            "child_pid": pid,
            "child_process_identity": process_identity,
            "process_group_id": pid,
            "session_id": pid,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(job_dir / "launch.json", launch)
    _write_json(job_dir / "status.json", status)
    item.update(
        {
            "status": "launched",
            "child_pid": pid,
            "child_process_identity": process_identity,
            "process_group_id": pid,
            "session_id": pid,
            "launched_at_utc": _utc_now(),
        }
    )
    _event(queue, "child_bound", run_id=RUN_ID, pid=pid)
    _save_queue(queue)


def _launch_or_recover(queue: MutableMapping[str, Any]) -> None:
    _verify_plan_closure(queue)
    _ensure_lease(queue, create=False)
    item = queue["items"][0]
    job_dir = _job_dir(queue, item)
    child_path = job_dir / "child.json"
    if child_path.is_file():
        child = _validate_child_record(queue, item)
        _bind_child(queue, int(child["child_pid"]), child["process_identity"])
        return
    command = _child_command(queue, item)
    matches = _matching_processes(command)
    if len(matches) > 1:
        raise G0cSoakQueueError("multiple exact G0c soak child processes exist")
    if matches:
        _bind_child(queue, *matches[0])
        return
    planned = _planned_item(queue)
    _require_fresh_artifacts(
        {
            "plan": Path(planned["soak_plan_path"]),
            "output_root": Path(planned["output_root"]),
            "seal": Path(planned["soak_seal_path"]),
        }
    )
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
        shared_queue._LOCAL_DETACH_LAUNCHERS[process.pid] = process
    except (OSError, shared_queue.QueueContractError) as exc:
        raise G0cSoakQueueError(f"cannot launch G0c soak child: {exc}") from exc
    identity = shared_queue._read_process_identity(process.pid)
    if identity.get("available") is True:
        _bind_child(queue, process.pid, identity)
        return
    if child_path.is_file():
        child = _validate_child_record(queue, item)
        _bind_child(queue, int(child["child_pid"]), child["process_identity"])
        return
    if shared_queue._process_running(process.pid, identity) is False:
        return
    raise G0cSoakQueueError("new G0c soak child identity is temporarily unobservable")


def _validate_job_binding(
    queue: Mapping[str, Any], item: Mapping[str, Any], *, terminal: bool | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    job_dir = _job_dir(queue, item)
    launch = _read_json(job_dir / "launch.json", label="G0c soak launch")
    status = _read_json(job_dir / "status.json", label="G0c soak status")
    expected_identity = _job_identity(queue, item)
    for label, value, schema in (
        ("launch", launch, JOB_LAUNCH_SCHEMA),
        ("status", status, JOB_STATUS_SCHEMA),
    ):
        if value.get("schema") != schema:
            raise G0cSoakQueueError(f"G0c soak {label} schema drifted")
        for key, expected in expected_identity.items():
            if value.get(key) != expected:
                raise G0cSoakQueueError(f"G0c soak {label} identity drifted at {key}")
    if (
        launch.get("child_command") != _child_command(queue, item)
        or launch.get("training_command") != _planned_item(queue)["training_command"]
    ):
        raise G0cSoakQueueError("G0c soak launch command binding drifted")
    process_identity = _validate_process_identity(
        item.get("child_process_identity"), label="queue item"
    )
    pid = int(process_identity["pid"])
    for label, value in (("launch", launch), ("status", status)):
        if (
            value.get("child_pid") != pid
            or value.get("process_group_id") != pid
            or value.get("session_id") != pid
            or value.get("child_process_identity") != item["child_process_identity"]
        ):
            raise G0cSoakQueueError(f"G0c soak {label} child binding drifted")
    lifecycle = (launch.get("status"), status.get("status"))
    expected_lifecycle = (
        {
            ("launched", "running"),
            ("completed", "running"),
            ("launched", "completed"),
            ("completed", "completed"),
        }
        if terminal is None
        else {("completed", "completed")}
        if terminal
        else {("launched", "running")}
    )
    if lifecycle not in expected_lifecycle:
        raise G0cSoakQueueError("G0c soak job lifecycle status drifted")
    if terminal is True:
        running = shared_queue._process_running(pid, process_identity)
        group = shared_queue._process_group_exists(pid)
        if running is not False or group is not False:
            raise G0cSoakQueueError("completed G0c soak child/group is still live or unknown")
        evidence = item.get("completion_evidence")
        digest = _canonical_sha256(evidence) if isinstance(evidence, Mapping) else None
        if (
            digest is None
            or launch.get("completion_evidence_sha256") != digest
            or status.get("completion_evidence_sha256") != digest
        ):
            raise G0cSoakQueueError("G0c soak completion binding drifted")
    return launch, status


def _seal_intent(queue: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    value = _read_json(
        _job_dir(queue, item) / "seal_intent.json", label="G0c soak seal intent"
    )
    expected = _job_identity(queue, item)
    planned = _planned_item(queue)
    if (
        value.get("schema") != SEAL_INTENT_SCHEMA
        or value.get("status") != "authorized_after_exact_child_success"
        or value.get("expected_plan_sha256") != planned["expected_plan_sha256"]
        or value.get("soak_plan_path") != planned["soak_plan_path"]
        or value.get("soak_seal_path") != planned["soak_seal_path"]
        or any(value.get(key) != expected_value for key, expected_value in expected.items())
    ):
        raise G0cSoakQueueError("G0c soak seal publication intent drifted")
    return value


def _verify_native_completion(
    queue: Mapping[str, Any], item: Mapping[str, Any], *, publish_seal: bool
) -> dict[str, Any]:
    planned = _planned_item(queue)
    plan_path = Path(str(planned["soak_plan_path"])).resolve(strict=True)
    output_root = Path(str(planned["output_root"])).resolve(strict=True)
    seal_path = Path(str(planned["soak_seal_path"])).resolve(strict=False)
    persisted_plan = _read_json(plan_path, label="completed G0c soak plan")
    if persisted_plan != planned["expected_plan"]:
        raise G0cSoakQueueError("completed G0c soak plan differs from queue plan")
    try:
        replay = training_runner.verify_checkpoint(
            persisted_plan, write_postflight=False
        )
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        raise G0cSoakQueueError(f"G0c soak postflight replay failed: {exc}") from exc
    postflight_path = (output_root / "postflight.json").resolve(strict=True)
    persisted_postflight = _read_json(
        postflight_path, label="completed G0c soak postflight"
    )
    if _strip_volatile(persisted_postflight) != _strip_volatile(replay):
        raise G0cSoakQueueError("persisted G0c soak postflight differs from replay")
    intent = _seal_intent(queue, item)
    if not _path_present(seal_path):
        if not publish_seal:
            raise G0cSoakQueueError("completed G0c soak seal is missing")
        try:
            seal = training_runner.build_soak_seal(plan_path)
            training_runner._write_json_fresh(seal_path, seal)
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
            raise G0cSoakQueueError(f"cannot publish G0c soak seal: {exc}") from exc
    try:
        validated = training_runner._validate_soak_seal(seal_path)
        replay_after_seal = training_runner.verify_checkpoint(
            persisted_plan, write_postflight=False
        )
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        raise G0cSoakQueueError(f"published G0c soak seal replay failed: {exc}") from exc
    persisted_postflight = _read_json(
        postflight_path, label="sealed G0c soak postflight"
    )
    if _strip_volatile(persisted_postflight) != _strip_volatile(replay_after_seal):
        raise G0cSoakQueueError("sealed G0c soak postflight differs from replay")
    if (
        validated.get("path") != str(seal_path.resolve(strict=True))
        or validated.get("plan") != persisted_plan
    ):
        raise G0cSoakQueueError("G0c soak seal validation identity drifted")
    return {
        "schema": COMPLETION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "queue_plan_sha256": queue["plan_sha256"],
        "run_id": RUN_ID,
        "job_id": item["job_id"],
        "seed": SEED,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "output_root": str(output_root),
        "soak_plan": _file_record(plan_path, "completed_soak_plan"),
        "postflight": _file_record(postflight_path, "completed_soak_postflight"),
        "checkpoint": _file_record(
            output_root / "checkpoint_iter.pth", "completed_soak_checkpoint"
        ),
        "soak_seal": _file_record(seal_path, "completed_soak_seal"),
        "seal_intent": _file_record(
            _job_dir(queue, item) / "seal_intent.json", "queue_owned_seal_intent"
        ),
        "seal_intent_payload_sha256": _canonical_sha256(intent),
        "native_plan_sha256": planned["expected_plan_sha256"],
        "fresh_only": True,
        "lease_release_gate": "durable_completion_reload_plus_full_native_replay",
    }


def _completion_candidate_path(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> Path:
    return _job_dir(queue, item) / "completion_candidate.json"


def _completion_candidate_payload(
    queue: Mapping[str, Any], item: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_CANDIDATE_SCHEMA,
        "status": "verified_pending_commit",
        **_job_identity(queue, item),
        "completion_evidence": copy.deepcopy(dict(evidence)),
        "completion_evidence_sha256": _canonical_sha256(evidence),
        "created_at_utc": _utc_now(),
    }


def _load_completion_candidate(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    value = _read_json(
        _completion_candidate_path(queue, item),
        label="G0c soak completion candidate",
    )
    if (
        value.get("schema") != COMPLETION_CANDIDATE_SCHEMA
        or value.get("status") != "verified_pending_commit"
        or any(
            value.get(key) != expected
            for key, expected in _job_identity(queue, item).items()
        )
    ):
        raise G0cSoakQueueError("G0c soak completion candidate identity drifted")
    evidence = value.get("completion_evidence")
    if (
        not isinstance(evidence, Mapping)
        or value.get("completion_evidence_sha256") != _canonical_sha256(evidence)
    ):
        raise G0cSoakQueueError("G0c soak completion candidate digest drifted")
    return value


def _persist_completion_candidate(
    queue: Mapping[str, Any], item: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    path = _completion_candidate_path(queue, item)
    expected = _completion_candidate_payload(queue, item, evidence)
    if path.is_file():
        observed = _load_completion_candidate(queue, item)
        if _strip_volatile(observed) != _strip_volatile(expected):
            raise G0cSoakQueueError(
                "persisted G0c soak completion candidate differs from replay"
            )
        return observed
    if _path_present(path):
        raise G0cSoakQueueError(
            f"G0c soak completion candidate is not a file: {path}"
        )
    _write_json_fresh(path, expected)
    return _load_completion_candidate(queue, item)


def _child_liveness(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> tuple[bool | None, bool | None]:
    process_identity = item["child_process_identity"]
    pid = int(item["child_pid"])
    return (
        shared_queue._process_running(pid, process_identity),
        shared_queue._process_group_exists(pid),
    )


def _require_successful_dead_child(
    queue: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    child = _validate_child_record(queue, item)
    running, group = _child_liveness(queue, item)
    if running is None or group is None:
        raise G0cSoakQueueError("G0c soak child/group liveness is unobservable")
    if running is not False or group is not False:
        raise G0cSoakQueueError("G0c soak child/process group is not fully exited")
    if child.get("status") != "completed" or child.get("returncode") != 0:
        raise G0cSoakQueueError(
            "G0c soak child did not publish an exact zero-return completion"
        )
    return child


def _commit_completed_queue(
    queue: MutableMapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    item = queue["items"][0]
    item["status"] = "completed"
    item["completed_at_utc"] = _utc_now()
    item["completion_evidence"] = copy.deepcopy(dict(evidence))
    queue["status"] = "completed"
    queue["active_item"] = None
    queue["completed_at_utc"] = _utc_now()
    _event(queue, "queue_completed", run_id=RUN_ID)
    _save_queue(queue)

    # Release authority comes only from the exact bytes durably reloaded here.
    durable = load_queue(Path(queue["plan"]["queue_dir"]))
    _verify_completed_item(durable)
    _ensure_lease(durable, create=False)
    _clear_lease(durable)


def _mark_completed(
    queue: MutableMapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    item = queue["items"][0]
    job_dir = _job_dir(queue, item)
    candidate = _persist_completion_candidate(queue, item, evidence)
    persisted_evidence = candidate["completion_evidence"]
    digest = candidate["completion_evidence_sha256"]
    launch, status = _validate_job_binding(queue, item, terminal=None)
    if launch.get("status") == "launched":
        launch.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "completion_evidence_sha256": digest,
            }
        )
        _write_json(job_dir / "launch.json", launch)
    elif launch.get("completion_evidence_sha256") != digest:
        raise G0cSoakQueueError(
            "terminal G0c soak launch differs from its completion candidate"
        )
    if status.get("status") == "running":
        status.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "completion_evidence_sha256": digest,
            }
        )
        _write_json(job_dir / "status.json", status)
    elif status.get("completion_evidence_sha256") != digest:
        raise G0cSoakQueueError(
            "terminal G0c soak status differs from its completion candidate"
        )
    _commit_completed_queue(queue, persisted_evidence)


def _recover_terminal_job(queue: MutableMapping[str, Any]) -> None:
    item = queue["items"][0]
    _require_successful_dead_child(queue, item)
    candidate = _load_completion_candidate(queue, item)
    replay = _verify_native_completion(queue, item, publish_seal=False)
    if _strip_volatile(candidate["completion_evidence"]) != _strip_volatile(replay):
        raise G0cSoakQueueError(
            "terminal G0c soak completion candidate differs from native replay"
        )
    _mark_completed(queue, candidate["completion_evidence"])


def _advance_launched(queue: MutableMapping[str, Any]) -> None:
    _verify_plan_closure(queue)
    _ensure_lease(queue, create=False)
    item = queue["items"][0]
    launch, status = _validate_job_binding(queue, item, terminal=None)
    if "completed" in {launch.get("status"), status.get("status")}:
        _recover_terminal_job(queue)
        return

    child_path = _job_dir(queue, item) / "child.json"
    child = _validate_child_record(queue, item) if child_path.is_file() else None
    running, group = _child_liveness(queue, item)
    item["last_observation"] = {
        "observed_at_utc": _utc_now(),
        "child_status": child["status"] if child is not None else "publication_pending",
        "pid_running": running,
        "process_group_exists": group,
    }
    _save_queue(queue)
    if running is True:
        if group is not True:
            raise G0cSoakQueueError(
                "live G0c soak child has an absent/unobservable process group"
            )
        return
    if running is None or group is None:
        raise G0cSoakQueueError("G0c soak child/group liveness is unobservable")
    if group is not False:
        raise G0cSoakQueueError(
            "G0c soak leader exited while its process group remains live"
        )
    if child is None:
        raise G0cSoakQueueError(
            "dead G0c soak child never published its queue-owned identity"
        )
    if child.get("status") != "completed" or child.get("returncode") != 0:
        raise G0cSoakQueueError(
            "G0c soak child did not publish an exact zero-return completion"
        )
    evidence = _verify_native_completion(queue, item, publish_seal=True)
    _mark_completed(queue, evidence)


def _verify_completed_item(queue: Mapping[str, Any]) -> dict[str, Any]:
    item = queue["items"][0]
    if queue["status"] != "completed" or item["status"] != "completed":
        raise G0cSoakQueueError("G0c soak queue/item is not completed")
    _verify_plan_closure(queue)
    _validate_job_binding(queue, item, terminal=True)
    candidate = _load_completion_candidate(queue, item)
    replay = _verify_native_completion(queue, item, publish_seal=False)
    persisted = item.get("completion_evidence")
    if (
        not isinstance(persisted, Mapping)
        or _strip_volatile(persisted) != _strip_volatile(replay)
        or _strip_volatile(candidate["completion_evidence"])
        != _strip_volatile(replay)
    ):
        raise G0cSoakQueueError("persisted G0c soak completion differs from replay")
    return replay


def _fence_child_if_any(queue: Mapping[str, Any]) -> dict[str, Any] | None:
    item = queue["items"][0]
    identity = item.get("child_process_identity")
    pid = item.get("child_pid")
    if not isinstance(identity, Mapping):
        child_path = _job_dir(queue, item) / "child.json"
        if child_path.is_file():
            child = _validate_child_record(queue, item)
            identity = child["process_identity"]
            pid = child["child_pid"]
        elif item["status"] == "launching":
            matches = _matching_processes(_child_command(queue, item))
            if len(matches) > 1:
                raise G0cSoakQueueError(
                    "cannot fence multiple exact G0c soak children"
                )
            if matches:
                pid, identity = matches[0]
    if not isinstance(identity, Mapping):
        return None
    try:
        return shared_queue._terminate_exact_process_group(
            pid, identity, label="G0c soak child"
        )
    except shared_queue.QueueContractError as exc:
        raise G0cSoakQueueError(f"cannot fence G0c soak child: {exc}") from exc


def _fail_queue(
    queue: MutableMapping[str, Any], *, error: BaseException | str, phase: str
) -> None:
    item = queue["items"][0]
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
        _read_json(launch_path, label="failed G0c soak launch")
        if launch_path.is_file()
        else {
            "schema": JOB_LAUNCH_SCHEMA,
            **identity,
            "child_command": _child_command(queue, item),
            "training_command": _planned_item(queue)["training_command"],
        }
    )
    status = (
        _read_json(status_path, label="failed G0c soak status")
        if status_path.is_file()
        else {"schema": JOB_STATUS_SCHEMA, **identity}
    )
    launch.update(
        {"status": "failed", "failed_at_utc": _utc_now(), "error": rendered}
    )
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
    queue["failure"] = {
        "phase": phase,
        "error": rendered,
        "lease_retained_fail_closed": _owned_lease_present(queue),
    }
    _event(queue, "queue_failed", run_id=RUN_ID, phase=phase, error=rendered)
    _save_queue(queue)


def advance_once(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    if queue["status"] == "completed":
        _verify_completed_item(queue)
        lease_path = Path(queue["plan"]["lease_path"])
        if lease_path.exists():
            _ensure_lease(queue, create=False)
            _clear_lease(queue)
        return load_queue(queue_dir)
    if queue["status"] == "failed":
        return queue
    try:
        status = queue["items"][0]["status"]
        if status == "pending":
            _reserve(queue)
        elif status == "reserved":
            _prepare_job(queue)
        elif status == "launching":
            _launch_or_recover(queue)
        elif status == "launched":
            _advance_launched(queue)
        else:
            raise G0cSoakQueueError(f"cannot advance G0c soak item in {status!r}")
    except G0cSoakQueueBusy:
        raise
    except G0cSoakLeaseLoss as exc:
        current = load_queue(queue_dir)
        try:
            termination = _fence_child_if_any(current)
        except G0cSoakQueueError as fence_error:
            current["lease_loss_fence_blocked"] = {
                "at_utc": _utc_now(),
                "lease_error": str(exc),
                "fence_error": str(fence_error),
            }
            _save_queue(current)
            raise
        if termination is not None:
            current["items"][0]["child_termination"] = termination
        _fail_queue(current, error=exc, phase="lease_ownership_loss")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        current = load_queue(queue_dir)
        if current["status"] == "completed":
            # A crash/error after the durable completion commit is recoverable
            # only if the committed receipt still fully replays.  Never hide a
            # post-commit verification failure behind status="completed".
            _verify_completed_item(current)
            if Path(current["plan"]["lease_path"]).exists():
                _ensure_lease(current, create=False)
                _clear_lease(current)
            return load_queue(queue_dir)
        termination = None
        fence_error = None
        if current["items"][0]["status"] in {"launching", "launched"}:
            try:
                termination = _fence_child_if_any(current)
            except G0cSoakQueueError as observed:
                fence_error = str(observed)
        if termination is not None:
            current["items"][0]["child_termination"] = termination
        rendered = exc
        if fence_error is not None:
            rendered = f"{type(exc).__name__}: {exc}; child fence blocked: {fence_error}"
        _fail_queue(current, error=rendered, phase="advance")
    return load_queue(queue_dir)


def run_queue(
    queue_dir: Path, *, poll_seconds: float = 10.0, once: bool = False
) -> dict[str, Any]:
    if poll_seconds < 0.05:
        raise G0cSoakQueueError("poll interval must be at least 0.05 seconds")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        with shared_queue._exclusive_file_lock(
            queue_dir / "supervisor.lock",
            busy_message=f"another G0c soak supervisor is active: {queue_dir}",
        ):
            while True:
                queue = advance_once(queue_dir)
                if once or queue["status"] in {"completed", "failed"}:
                    return queue
                time.sleep(poll_seconds)
    except shared_queue.QueueBusyError as exc:
        raise G0cSoakQueueBusy(str(exc)) from exc


def queue_status(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    item = queue["items"][0]
    lease_path = Path(queue["plan"]["lease_path"])
    return {
        "schema": "pivot.stageb.table_a.g0c_soak_queue_status/v1",
        "observed_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "status": queue["status"],
        "revision": queue["revision"],
        "run_id": RUN_ID,
        "item_status": item["status"],
        "active_item": copy.deepcopy(queue.get("active_item")),
        "last_observation": copy.deepcopy(item.get("last_observation")),
        "failure": copy.deepcopy(queue.get("failure")),
        "lease_path": str(lease_path),
        "lease_present": lease_path.is_file(),
        "lease_owned": _owned_lease_present(queue),
    }


def verify_queue(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    errors = []
    evidence = None
    try:
        evidence = _verify_completed_item(queue)
    except (G0cSoakQueueError, OSError, ValueError) as exc:
        errors.append(str(exc))
    lease_path = Path(queue["plan"]["lease_path"])
    if queue["status"] == "completed" and lease_path.exists():
        errors.append("completed G0c soak queue retained its GPU lease")
    passed = queue["status"] == "completed" and not errors
    return {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "status": "passed" if passed else "failed",
        "queue_status": queue["status"],
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "run_id": RUN_ID,
        "completion_evidence": evidence,
        "errors": errors,
    }


def dry_run() -> dict[str, Any]:
    try:
        plan = build_queue_plan()
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        G0cSoakQueueError,
        ValueError,
    ) as exc:
        return {
            "status": "blocked",
            "queue_dir": str(DEFAULT_QUEUE_DIR),
            "run_id": RUN_ID,
            "reason": str(exc),
            "artifact_audit": audit_existing_artifacts(),
            "mutated": False,
        }
    item = plan["items"][0]
    return {
        "status": "ready",
        "queue_dir": plan["queue_dir"],
        "queue_id": plan["queue_id"],
        "plan_sha256": _canonical_sha256(plan),
        "run_id": RUN_ID,
        "contract": {
            "seed": SEED,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
            "optimizer_updates": OPTIMIZER_UPDATES,
        },
        "soak_plan_sha256": item["expected_plan_sha256"],
        "controller_source_count": len(plan["controller_sources"]),
        "input_record_count": len(item["input_records"]),
        "lease_path": plan["lease_path"],
        "mutated": False,
    }


def execute_child(queue_dir: Path, job_id: str) -> int:
    queue = load_queue(queue_dir)
    item = queue["items"][0]
    if (
        queue["status"] != "running"
        or item["status"] not in {"launching", "launched"}
        or item.get("job_id") != job_id
    ):
        raise G0cSoakQueueError("G0c soak child was not authorized by the active job")
    _verify_plan_closure(queue)
    _ensure_lease(queue, create=False)
    _seal_intent(queue, item)
    planned = _planned_item(queue)
    _require_fresh_artifacts(
        {
            "plan": Path(planned["soak_plan_path"]),
            "output_root": Path(planned["output_root"]),
            "seal": Path(planned["soak_seal_path"]),
        }
    )
    pid = os.getpid()
    identity = _validate_process_identity(
        {
            **shared_queue._read_process_identity(pid),
            "process_group_id": os.getpgid(pid),
            "session_id": os.getsid(pid),
        },
        label="executing child",
    )
    if os.getpgid(pid) != pid or os.getsid(pid) != pid:
        raise G0cSoakQueueError(
            "G0c soak child must be launched as a new session/process group"
        )
    child_path = _job_dir(queue, item) / "child.json"
    child = {
        "schema": CHILD_STATUS_SCHEMA,
        "status": "running",
        **_job_identity(queue, item),
        "child_pid": pid,
        "process_identity": identity,
        "process_group_id": pid,
        "session_id": pid,
        "training_command": copy.deepcopy(planned["training_command"]),
        "started_at_utc": _utc_now(),
    }
    _write_json_fresh(child_path, child)
    returncode = 126
    error = None
    try:
        environment = shared_queue._runner_environment(
            queue["plan"]["runtime_environment"]
        )
        result = subprocess.run(
            planned["training_command"],
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        returncode = int(result.returncode)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    child.update(
        {
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "finished_at_utc": _utc_now(),
        }
    )
    if error is not None:
        child["error"] = error
    _write_json(child_path, child)
    return returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("audit")
    modes.add_parser("dry-run")
    create = modes.add_parser("create")
    create.add_argument("--gpu-key", default="0")
    create.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    reconcile = modes.add_parser("reconcile")
    reconcile.add_argument("queue_dir", type=Path, nargs="?", default=DEFAULT_QUEUE_DIR)
    run = modes.add_parser("run")
    run.add_argument("queue_dir", type=Path, nargs="?", default=DEFAULT_QUEUE_DIR)
    run.add_argument("--poll-seconds", type=float, default=10.0)
    status = modes.add_parser("status")
    status.add_argument("queue_dir", type=Path, nargs="?", default=DEFAULT_QUEUE_DIR)
    verify = modes.add_parser("verify")
    verify.add_argument("queue_dir", type=Path, nargs="?", default=DEFAULT_QUEUE_DIR)
    child = modes.add_parser("execute-child", help=argparse.SUPPRESS)
    child.add_argument("--queue-dir", type=Path, required=True)
    child.add_argument("--job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "audit":
            result = audit_existing_artifacts()
            code = 0
        elif args.mode == "dry-run":
            result = dry_run()
            code = 0
        elif args.mode == "create":
            result = create_queue(gpu_key=args.gpu_key, num_workers=args.num_workers)
            code = 0
        elif args.mode == "reconcile":
            result = run_queue(args.queue_dir, once=True)
            code = 0 if result["status"] != "failed" else 1
        elif args.mode == "run":
            result = run_queue(args.queue_dir, poll_seconds=args.poll_seconds)
            code = 0 if result["status"] != "failed" else 1
        elif args.mode == "status":
            result = queue_status(args.queue_dir)
            code = 0
        elif args.mode == "verify":
            result = verify_queue(args.queue_dir)
            code = 0 if result["status"] == "passed" else 1
        elif args.mode == "execute-child":
            return execute_child(args.queue_dir, args.job_id)
        else:  # pragma: no cover
            parser.error(f"unknown mode: {args.mode}")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return code
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        G0cSoakQueueError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
