#!/usr/bin/env python3
"""Run the exact three-seed, six-phase formal Table-B v2 validation queue.

Each queue item evaluates one training seed and the matched evaluator executes
the D2m then D3m phases.  The queue retains one shared GPU lease across all
three seeds, hash-binds every evaluation to the immutable queue spec, and only
advances after replaying the completed evaluator postflight.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_paper_evaluations as paper_evaluator  # noqa: E402
from tools import run_stageb_serial_matrix_queue as serial_queue  # noqa: E402
from tools import run_stageb_table_b_matched_evaluations as evaluator  # noqa: E402
from tools import run_stageb_table_b_v2 as training_runner  # noqa: E402
from tools import run_stageb_table_b_v2_queue as training_queue  # noqa: E402
from tools import stageb_profile_dependency_audit as profile_dependency  # noqa: E402
from tools import stageb_table_b_matched_eval_surface as surface  # noqa: E402
from util import stage_b_table_b_v2_contract as training_contract  # noqa: E402


QUEUE_SCHEMA = "pivot.stageb.table_b_v2_validation_queue/v1"
PLAN_SCHEMA = "pivot.stageb.table_b_v2_validation_plan/v1"
SPEC_SCHEMA = "pivot.stageb.table_b_v2_validation_input/v1"
VERIFICATION_SCHEMA = "pivot.stageb.table_b_v2_validation_verification/v1"
SUPERVISOR_SCHEMA = "pivot.stageb.table_b_v2_validation_supervisor/v1"
STATUS_SCHEMA = "pivot.stageb.table_b_v2_validation_status/v1"

SEEDS = tuple(training_contract.SEEDS)
RUN_IDS = tuple(f"seed{seed}" for seed in SEEDS)
PHASE_ORDER = tuple(evaluator.CONDITIONS)
PROFILE = evaluator.EVAL_PROFILE
ITEM_STATUSES = frozenset({"pending", "reserved", "launched", "completed", "failed"})

DEFAULT_QUEUE_DIR = REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_b_v2_validation_v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/evaluations/table_b_v2_matched_formal"
)
DEFAULT_EVALUATION_RUNNER = REPO_ROOT / "tools/run_stageb_table_b_matched_evaluations.py"
DEFAULT_AGGREGATOR = REPO_ROOT / "tools/aggregate_stageb_table_b_v2_validation.py"
DEFAULT_PYTHON = paper_evaluator.DEFAULT_PYTHON
DEFAULT_DATA_ROOT = paper_evaluator.DEFAULT_DATA_ROOT
DEFAULT_LEASE_ROOT = serial_queue.DEFAULT_LEASE_ROOT
VALIDATION_SPEC_NAME = "validation_input_spec.json"

RUNTIME = {
    "device": "cuda:0",
    "batch_size": 16,
    "num_workers": 4,
    "amp": False,
    "log_every": 25,
}

_LOCAL_PROCESSES: dict[int, subprocess.Popen[Any]] = {}


class TableBV2ValidationQueueError(RuntimeError):
    """The exact Table-B v2 validation queue contract was violated."""


class TableBV2ValidationQueueBusy(TableBV2ValidationQueueError):
    """Another supervisor or queue currently owns the requested resource."""


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


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TableBV2ValidationQueueError(
            f"cannot read {label} {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TableBV2ValidationQueueError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _verify_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise TableBV2ValidationQueueError(f"{label} record is invalid")
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    if _file_record(path) != dict(record):
        raise TableBV2ValidationQueueError(f"{label} identity changed: {path}")
    return path


def _source_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [_file_record(path) for path in sorted(set(paths), key=str)]


def _recursive_sources(entry: Path) -> list[Path]:
    try:
        return profile_dependency.recursive_local_python_dependencies(
            (entry.relative_to(REPO_ROOT).as_posix(),),
            repository_root=REPO_ROOT,
        )
    except profile_dependency.ProfileDependencyAuditError as exc:
        raise TableBV2ValidationQueueError(
            f"source dependency audit failed: {exc}"
        ) from exc


def _evaluation_source_paths() -> list[Path]:
    try:
        records = evaluator._code_records()
    except (OSError, ValueError, evaluator.MatchedEvaluationError) as exc:
        raise TableBV2ValidationQueueError(
            f"matched evaluator source closure failed: {exc}"
        ) from exc
    paths = [Path(str(record["path"])).resolve(strict=True) for record in records]
    if not paths or len(paths) != len(set(paths)):
        raise TableBV2ValidationQueueError(
            "matched evaluator source closure is empty or duplicated"
        )
    if DEFAULT_EVALUATION_RUNNER.resolve(strict=True) not in paths:
        raise TableBV2ValidationQueueError(
            "matched evaluator source closure omits its entry point"
        )
    return sorted(paths, key=str)


def _matched_input_records() -> list[dict[str, Any]]:
    try:
        audit = evaluator.verify_panel(surface.DEFAULT_AUDIT)
    except (OSError, KeyError, TypeError, evaluator.MatchedPanelError) as exc:
        raise TableBV2ValidationQueueError(
            f"matched evaluation audit verification failed: {exc}"
        ) from exc
    expected = {
        "audit": surface.DEFAULT_AUDIT.resolve(strict=True),
        "pair_ledger": surface.DEFAULT_LEDGER.resolve(strict=True),
        "d3m_source": surface.DEFAULT_D3M_SOURCE.resolve(strict=True),
        "d2m_source": Path(audit["outputs"]["d2m_calibration"]["path"]).resolve(
            strict=True
        ),
    }
    return [
        {"role": role, **_file_record(path)}
        for role, path in sorted(expected.items())
    ]


def _training_queue_record(queue_dir: Path) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    completion_path = (
        queue_dir / training_queue.COMPLETION_NAME
    ).resolve(strict=True)
    persisted = _read_json(completion_path, label="formal training completion")
    try:
        replayed = training_queue.verify_formal_queue(queue_dir, persist=False)
    except (
        OSError,
        ValueError,
        training_queue.FormalQueueError,
        training_runner.TableBV2RunnerError,
        training_contract.TableBContractError,
        serial_queue.QueueContractError,
    ) as exc:
        raise TableBV2ValidationQueueError(
            f"formal Table-B v2 training queue verification failed: {exc}"
        ) from exc
    if persisted != replayed:
        raise TableBV2ValidationQueueError(
            "formal training completion attestation differs from replay"
        )
    if not (
        replayed.get("status") == "passed"
        and replayed.get("profile") == training_contract.FORMAL_PROFILE
        and replayed.get("ordered_run_ids") == list(training_contract.FORMAL_RUN_IDS)
        and set(replayed.get("runs", {})) == set(training_contract.FORMAL_RUN_IDS)
    ):
        raise TableBV2ValidationQueueError(
            "formal training completion does not attest the exact six runs"
        )
    runs = {
        run_id: str(Path(replayed["runs"][run_id]["run_root"]).resolve(strict=True))
        for run_id in training_contract.FORMAL_RUN_IDS
    }
    queue = replayed.get("queue")
    if not isinstance(queue, Mapping):
        raise TableBV2ValidationQueueError("formal training completion lacks queue identity")
    return {
        "queue_dir": str(queue_dir),
        "queue_id": queue.get("queue_id"),
        "plan_sha256": queue.get("plan_sha256"),
        "profile": replayed["profile"],
        "ordered_run_ids": list(replayed["ordered_run_ids"]),
        "completion_semantic_sha256": replayed.get("semantic_sha256"),
        "manifest": _file_record(queue_dir / "queue.json"),
        "source_plan": _file_record(queue_dir / training_queue.SOURCE_PLAN_NAME),
        "scope_plan": _file_record(queue_dir / training_queue.SCOPE_PLAN_NAME),
        "completion_attestation": _file_record(completion_path),
        "runs": runs,
    }


def _verify_training_queue_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TableBV2ValidationQueueError("formal training queue record is invalid")
    queue_dir = Path(str(record.get("queue_dir", ""))).resolve(strict=True)
    observed = _training_queue_record(queue_dir)
    if observed != dict(record):
        raise TableBV2ValidationQueueError(
            "formal training queue identity changed after validation planning"
        )
    return observed


def _evaluation_root(output_root: Path, seed: int) -> Path:
    return (Path(output_root) / f"seed{seed}").resolve(strict=False)


def _evaluation_command(
    *,
    seed: int,
    evaluation_root: Path,
    training: Mapping[str, Any],
    validation_spec: Path,
) -> list[str]:
    runs = training["runs"]
    return [
        str(DEFAULT_PYTHON.resolve(strict=True)),
        str(DEFAULT_EVALUATION_RUNNER.resolve(strict=True)),
        "run",
        "--d2m-training-run-root",
        str(Path(runs[f"D2m:{seed}"]).resolve(strict=True)),
        "--d3m-training-run-root",
        str(Path(runs[f"D3m:{seed}"]).resolve(strict=True)),
        "--seed",
        str(seed),
        "--output-dir",
        str(evaluation_root),
        "--audit",
        str(surface.DEFAULT_AUDIT.resolve(strict=True)),
        "--pair-ledger",
        str(surface.DEFAULT_LEDGER.resolve(strict=True)),
        "--d3m-source",
        str(surface.DEFAULT_D3M_SOURCE.resolve(strict=True)),
        "--data-root",
        str(DEFAULT_DATA_ROOT.resolve(strict=True)),
        "--python",
        str(DEFAULT_PYTHON.resolve(strict=True)),
        "--device",
        RUNTIME["device"],
        "--batch-size",
        str(RUNTIME["batch_size"]),
        "--num-workers",
        str(RUNTIME["num_workers"]),
        "--log-every",
        str(RUNTIME["log_every"]),
        "--training-queue-dir",
        str(Path(training["queue_dir"]).resolve(strict=True)),
        "--training-source-contract",
        evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT,
        "--validation-queue-spec",
        str(validation_spec),
    ]


def _spec_payload(plan: Mapping[str, Any], plan_sha256: str) -> dict[str, Any]:
    return {
        "schema": SPEC_SCHEMA,
        "queue_dir": str(plan["queue_dir"]),
        "queue_id": str(plan["queue_id"]),
        "plan_sha256": plan_sha256,
        "profile": PROFILE,
        "training_source_contract": evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT,
        "ordered_seeds": list(SEEDS),
        "phase_order_per_seed": list(PHASE_ORDER),
        "total_phase_count": len(SEEDS) * len(PHASE_ORDER),
        "runtime": dict(RUNTIME),
        "training_queue": {
            key: plan["training_queue"][key]
            for key in (
                "queue_dir",
                "queue_id",
                "plan_sha256",
                "profile",
                "completion_semantic_sha256",
            )
        },
        "evaluation_outputs": {
            str(item["seed"]): str(item["evaluation_root"])
            for item in plan["items"]
        },
    }


def build_plan(
    queue_dir: Path,
    *,
    training_queue_dir: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lease_root: Path = DEFAULT_LEASE_ROOT,
    gpu_key: str | None = None,
) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    if queue_dir.exists():
        raise FileExistsError(f"validation queue directory must be fresh: {queue_dir}")
    if output_root.exists():
        raise FileExistsError(f"validation output root must be fresh: {output_root}")
    if (
        output_root == queue_dir
        or output_root in queue_dir.parents
        or queue_dir in output_root.parents
    ):
        raise TableBV2ValidationQueueError(
            "validation queue and output roots must be disjoint"
        )
    python = DEFAULT_PYTHON.resolve(strict=True)
    data_root = DEFAULT_DATA_ROOT.resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise TableBV2ValidationQueueError(
            f"sealed validation Python is not executable: {python}"
        )
    if not data_root.is_dir():
        raise TableBV2ValidationQueueError(f"validation data root is invalid: {data_root}")
    training = _training_queue_record(training_queue_dir)
    selected_gpu = serial_queue._gpu_key_from_environment(
        serial_queue._snapshot_environment(), gpu_key
    )
    lease_root = Path(lease_root).expanduser().resolve(strict=False)
    spec_path = (queue_dir / VALIDATION_SPEC_NAME).resolve(strict=False)
    items = []
    for index, seed in enumerate(SEEDS):
        evaluation_root = _evaluation_root(output_root, seed)
        command = _evaluation_command(
            seed=seed,
            evaluation_root=evaluation_root,
            training=training,
            validation_spec=spec_path,
        )
        items.append(
            {
                "index": index,
                "run_id": RUN_IDS[index],
                "seed": seed,
                "phase_order": list(PHASE_ORDER),
                "phase_count": len(PHASE_ORDER),
                "d2m_training_root": training["runs"][f"D2m:{seed}"],
                "d3m_training_root": training["runs"][f"D3m:{seed}"],
                "evaluation_root": str(evaluation_root),
                "command": command,
                "command_shell": shlex.join(command),
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "queue_id": str(uuid.uuid4()),
        "created_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "repository_root": str(REPO_ROOT),
        "profile": PROFILE,
        "training_source_contract": evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT,
        "ordered_seeds": list(SEEDS),
        "ordered_run_ids": list(RUN_IDS),
        "phase_order_per_seed": list(PHASE_ORDER),
        "total_phase_count": len(SEEDS) * len(PHASE_ORDER),
        "output_root": str(output_root),
        "runner_python": _file_record(python),
        "data_root": str(data_root),
        "runtime": dict(RUNTIME),
        "evaluation_runner": _file_record(DEFAULT_EVALUATION_RUNNER),
        "evaluation_sources": _source_records(_evaluation_source_paths()),
        "controller_sources": _source_records(
            _recursive_sources(Path(__file__).resolve(strict=True))
        ),
        "aggregation_sources": _source_records(
            _recursive_sources(DEFAULT_AGGREGATOR.resolve(strict=True))
        ),
        "matched_inputs": _matched_input_records(),
        "training_queue": training,
        "gpu_key": selected_gpu,
        "gpu_environment": {"CUDA_VISIBLE_DEVICES": selected_gpu},
        "lease_root": str(lease_root),
        "lease_path": str(serial_queue._lease_path(lease_root, selected_gpu)),
        "validation_input_spec": {
            "schema": SPEC_SCHEMA,
            "path": str(spec_path),
        },
        "items": items,
    }


def create_queue(
    queue_dir: Path,
    *,
    training_queue_dir: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lease_root: Path = DEFAULT_LEASE_ROOT,
    gpu_key: str | None = None,
) -> dict[str, Any]:
    plan = build_plan(
        queue_dir,
        training_queue_dir=training_queue_dir,
        output_root=output_root,
        lease_root=lease_root,
        gpu_key=gpu_key,
    )
    queue_dir = Path(plan["queue_dir"])
    plan_sha256 = _canonical_sha(plan)
    now = _utc_now()
    queue = {
        "schema": QUEUE_SCHEMA,
        "status": "planned",
        "created_at_utc": now,
        "updated_at_utc": now,
        "revision": 0,
        "plan": plan,
        "plan_sha256": plan_sha256,
        "items": [
            {"index": index, "run_id": run_id, "status": "pending"}
            for index, run_id in enumerate(RUN_IDS)
        ],
        "events": [
            {
                "at_utc": now,
                "event": "queue_created",
                "ordered_run_ids": list(RUN_IDS),
                "total_phase_count": len(SEEDS) * len(PHASE_ORDER),
            }
        ],
    }
    spec = _spec_payload(plan, plan_sha256)
    queue_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = queue_dir.parent / f".{queue_dir.name}.creating-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_json_atomic(staging / "queue.json", queue)
        _write_json_atomic(staging / VALIDATION_SPEC_NAME, spec)
        os.replace(staging, queue_dir)
        descriptor = os.open(queue_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with contextlib.suppress(OSError):
            staging.rmdir()
    return load_queue(queue_dir)


def _validate_plan(queue: Mapping[str, Any], queue_dir: Path) -> None:
    if queue.get("schema") != QUEUE_SCHEMA:
        raise TableBV2ValidationQueueError("validation queue schema drifted")
    plan = queue.get("plan")
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise TableBV2ValidationQueueError("validation queue plan is invalid")
    if queue.get("plan_sha256") != _canonical_sha(plan):
        raise TableBV2ValidationQueueError("validation queue plan SHA-256 mismatch")
    expected_keys = {
        "schema", "queue_id", "created_at_utc", "queue_dir", "repository_root",
        "profile", "training_source_contract", "ordered_seeds", "ordered_run_ids",
        "phase_order_per_seed", "total_phase_count", "output_root", "runner_python",
        "data_root", "runtime", "evaluation_runner", "evaluation_sources",
        "controller_sources", "aggregation_sources", "matched_inputs",
        "training_queue", "gpu_key", "gpu_environment", "lease_root", "lease_path",
        "validation_input_spec", "items",
    }
    if set(plan) != expected_keys:
        raise TableBV2ValidationQueueError("validation queue plan field set drifted")
    try:
        parsed_id = uuid.UUID(str(plan.get("queue_id")))
    except (ValueError, AttributeError) as exc:
        raise TableBV2ValidationQueueError("validation queue ID is invalid") from exc
    if str(parsed_id) != plan.get("queue_id"):
        raise TableBV2ValidationQueueError("validation queue ID is not canonical")
    if Path(str(plan.get("queue_dir", ""))).resolve(strict=False) != queue_dir:
        raise TableBV2ValidationQueueError("validation queue opened through another path")
    if not (
        Path(str(plan.get("repository_root", ""))).resolve(strict=False) == REPO_ROOT
        and plan.get("profile") == PROFILE
        and plan.get("training_source_contract")
        == evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT
        and plan.get("ordered_seeds") == list(SEEDS)
        and plan.get("ordered_run_ids") == list(RUN_IDS)
        and plan.get("phase_order_per_seed") == list(PHASE_ORDER)
        and plan.get("total_phase_count") == len(SEEDS) * len(PHASE_ORDER)
        and plan.get("runtime") == RUNTIME
        and Path(str(plan.get("data_root", ""))).resolve(strict=False)
        == DEFAULT_DATA_ROOT.resolve(strict=True)
    ):
        raise TableBV2ValidationQueueError("validation identity/runtime/order drifted")
    output_root = Path(str(plan.get("output_root", ""))).resolve(strict=False)
    if (
        not output_root.is_absolute()
        or output_root == queue_dir
        or output_root in queue_dir.parents
        or queue_dir in output_root.parents
    ):
        raise TableBV2ValidationQueueError("validation output root is invalid")
    runner_python = plan.get("runner_python")
    evaluation_runner = plan.get("evaluation_runner")
    if not (
        isinstance(runner_python, Mapping)
        and Path(str(runner_python.get("path", ""))).resolve(strict=False)
        == DEFAULT_PYTHON.resolve(strict=True)
        and isinstance(evaluation_runner, Mapping)
        and Path(str(evaluation_runner.get("path", ""))).resolve(strict=False)
        == DEFAULT_EVALUATION_RUNNER.resolve(strict=True)
    ):
        raise TableBV2ValidationQueueError("validation runner identity drifted")
    training = plan.get("training_queue")
    if not (
        isinstance(training, Mapping)
        and training.get("profile") == training_contract.FORMAL_PROFILE
        and training.get("ordered_run_ids") == list(training_contract.FORMAL_RUN_IDS)
        and set(training.get("runs", {})) == set(training_contract.FORMAL_RUN_IDS)
    ):
        raise TableBV2ValidationQueueError("formal training queue binding drifted")
    gpu_key = plan.get("gpu_key")
    lease_root = Path(str(plan.get("lease_root", ""))).resolve(strict=False)
    if not (
        isinstance(gpu_key, str)
        and bool(gpu_key.strip())
        and "," not in gpu_key
        and plan.get("gpu_environment") == {"CUDA_VISIBLE_DEVICES": gpu_key}
        and Path(str(plan.get("lease_path", ""))).resolve(strict=False)
        == serial_queue._lease_path(lease_root, gpu_key).resolve(strict=False)
    ):
        raise TableBV2ValidationQueueError("validation GPU lease contract drifted")
    spec_path = (queue_dir / VALIDATION_SPEC_NAME).resolve(strict=False)
    if plan.get("validation_input_spec") != {
        "schema": SPEC_SCHEMA,
        "path": str(spec_path),
    }:
        raise TableBV2ValidationQueueError("validation input spec binding drifted")
    plan_items = plan.get("items")
    items = queue.get("items")
    if not (
        isinstance(plan_items, list)
        and len(plan_items) == len(SEEDS)
        and isinstance(items, list)
        and len(items) == len(SEEDS)
    ):
        raise TableBV2ValidationQueueError("validation queue item set is invalid")
    completed_prefix = True
    active = 0
    for index, (planned, item) in enumerate(zip(plan_items, items)):
        seed = SEEDS[index]
        evaluation_root = _evaluation_root(output_root, seed)
        command = _evaluation_command(
            seed=seed,
            evaluation_root=evaluation_root,
            training=training,
            validation_spec=spec_path,
        )
        expected_planned = {
            "index": index,
            "run_id": RUN_IDS[index],
            "seed": seed,
            "phase_order": list(PHASE_ORDER),
            "phase_count": len(PHASE_ORDER),
            "d2m_training_root": training["runs"][f"D2m:{seed}"],
            "d3m_training_root": training["runs"][f"D3m:{seed}"],
            "evaluation_root": str(evaluation_root),
            "command": command,
            "command_shell": shlex.join(command),
        }
        if not (
            isinstance(planned, Mapping)
            and dict(planned) == expected_planned
            and isinstance(item, Mapping)
            and item.get("index") == index
            and item.get("run_id") == RUN_IDS[index]
            and item.get("status") in ITEM_STATUSES
        ):
            raise TableBV2ValidationQueueError(f"validation item {index} drifted")
        status = item["status"]
        if status == "completed":
            if not completed_prefix:
                raise TableBV2ValidationQueueError("completed items are not a prefix")
        else:
            completed_prefix = False
        if status in {"reserved", "launched", "failed"}:
            active += 1
    if active > 1:
        raise TableBV2ValidationQueueError("validation queue has multiple active items")
    status = queue.get("status")
    if status not in {"planned", "running", "completed", "failed"}:
        raise TableBV2ValidationQueueError("validation queue status is invalid")
    if status == "planned" and any(item["status"] != "pending" for item in items):
        raise TableBV2ValidationQueueError("planned validation queue has started items")
    if status == "completed" and any(item["status"] != "completed" for item in items):
        raise TableBV2ValidationQueueError("completed validation queue is incomplete")
    if status == "failed" and sum(item["status"] == "failed" for item in items) != 1:
        raise TableBV2ValidationQueueError("failed validation queue has no failed item")
    expected_spec = _spec_payload(plan, str(queue["plan_sha256"]))
    if _read_json(spec_path, label="validation input spec") != expected_spec:
        raise TableBV2ValidationQueueError("validation input spec differs from queue plan")


def load_queue(queue_dir: Path) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    queue = _read_json(queue_dir / "queue.json", label="validation queue")
    _validate_plan(queue, queue_dir)
    return queue


def _save_queue(queue: MutableMapping[str, Any]) -> None:
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_at_utc"] = _utc_now()
    _write_json_atomic(Path(queue["plan"]["queue_dir"]) / "queue.json", queue)


def _event(queue: MutableMapping[str, Any], name: str, **fields: Any) -> None:
    queue.setdefault("events", []).append({"at_utc": _utc_now(), "event": name, **fields})


def _verify_sources(queue: Mapping[str, Any]) -> None:
    plan = queue["plan"]
    _verify_file_record(plan.get("runner_python"), label="validation Python")
    _verify_file_record(plan.get("evaluation_runner"), label="evaluation runner")
    expected_paths = {
        "evaluation_sources": _evaluation_source_paths(),
        "controller_sources": _recursive_sources(Path(__file__).resolve(strict=True)),
        "aggregation_sources": _recursive_sources(DEFAULT_AGGREGATOR.resolve(strict=True)),
    }
    for key, paths in expected_paths.items():
        records = plan.get(key)
        expected = [str(path.resolve(strict=True)) for path in sorted(set(paths), key=str)]
        observed = [
            str(record.get("path", "")) if isinstance(record, Mapping) else ""
            for record in records or []
        ]
        if not records or observed != expected:
            raise TableBV2ValidationQueueError(
                f"{key} differs from the current recursive dependency closure"
            )
        for index, record in enumerate(records):
            _verify_file_record(record, label=f"{key}[{index}]")
    matched = plan.get("matched_inputs")
    expected_matched = _matched_input_records()
    if matched != expected_matched:
        raise TableBV2ValidationQueueError("matched evaluation inputs drifted")
    for index, record in enumerate(matched):
        identity = {key: record[key] for key in ("path", "sha256", "size_bytes")}
        _verify_file_record(identity, label=f"matched_inputs[{index}]")
    _verify_training_queue_record(plan.get("training_queue"))


def _planned_item(queue: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    return queue["plan"]["items"][index]


def _active_index(queue: Mapping[str, Any]) -> int | None:
    return next(
        (index for index, item in enumerate(queue["items"]) if item["status"] != "completed"),
        None,
    )


def _matching_processes(command: Sequence[str]) -> list[tuple[int, dict[str, Any]]]:
    matches: list[tuple[int, dict[str, Any]]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    expected = list(command)
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
        if observed == expected:
            pid = int(entry.name)
            identity = serial_queue._read_process_identity(pid)
            if serial_queue._process_running(pid, identity) is True:
                matches.append((pid, identity))
    return matches


def _bind_process(
    queue: MutableMapping[str, Any], index: int, pid: int, identity: Mapping[str, Any]
) -> None:
    item = queue["items"][index]
    item["status"] = "launched"
    item["child_pid"] = int(pid)
    item["child_process_identity"] = dict(identity)
    item["launched_at_utc"] = _utc_now()
    _event(queue, "evaluation_launched", index=index, run_id=item["run_id"], pid=pid)
    _save_queue(queue)


def _reserve(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_sources(queue)
    item = queue["items"][index]
    serial_queue._ensure_lease(queue, item, create=True)
    planned = _planned_item(queue, index)
    output = Path(str(planned["evaluation_root"]))
    if output.exists():
        raise TableBV2ValidationQueueError(f"evaluation root must be fresh: {output}")
    log = (
        Path(queue["plan"]["queue_dir"])
        / "logs"
        / f"{index:03d}-{item['run_id']}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    item["status"] = "reserved"
    item["reserved_at_utc"] = _utc_now()
    item["console_log"] = str(log)
    queue["status"] = "running"
    _event(queue, "evaluation_reserved", index=index, run_id=item["run_id"])
    _save_queue(queue)


def _launch_reserved(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_sources(queue)
    item = queue["items"][index]
    serial_queue._ensure_lease(queue, item, create=False)
    planned = _planned_item(queue, index)
    command = list(planned["command"])
    matches = _matching_processes(command)
    if len(matches) > 1:
        raise TableBV2ValidationQueueError("multiple evaluation processes match one item")
    if matches:
        _bind_process(queue, index, *matches[0])
        return
    output = Path(str(planned["evaluation_root"]))
    if output.exists():
        launch_path = output / "launch.json"
        if (
            launch_path.is_file()
            and _read_json(launch_path, label="orphan launch").get("status") == "completed"
        ):
            _bind_process(queue, index, 0, {})
            return
        raise TableBV2ValidationQueueError(
            "evaluation root exists without a recoverable completed launch"
        )
    environment = dict(os.environ)
    environment.update(queue["plan"]["gpu_environment"])
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        with Path(item["console_log"]).open("ab", buffering=0) as log:
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
    except OSError as exc:
        raise TableBV2ValidationQueueError(f"cannot launch evaluation: {exc}") from exc
    pid = int(process.pid)
    _LOCAL_PROCESSES[pid] = process
    _bind_process(queue, index, pid, serial_queue._read_process_identity(pid))


def _launch_source_binding(queue: Mapping[str, Any], launch: Mapping[str, Any]) -> None:
    contract = launch.get("contract")
    records = contract.get("input_records") if isinstance(contract, Mapping) else None
    if not isinstance(records, list):
        raise TableBV2ValidationQueueError("evaluation launch input records are missing")
    observed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or "evaluation_code" not in record.get("roles", []):
            continue
        identity = {key: record.get(key) for key in ("path", "sha256", "size_bytes")}
        path = str(Path(str(identity["path"])).resolve(strict=False))
        if path in observed:
            raise TableBV2ValidationQueueError("evaluation launch duplicates a source")
        identity["path"] = path
        observed[path] = identity
    expected = queue["plan"]["evaluation_sources"]
    if [observed[path] for path in sorted(observed)] != expected:
        raise TableBV2ValidationQueueError(
            "evaluation launch source closure differs from the immutable queue plan"
        )


def _verify_completed(queue: Mapping[str, Any], index: int) -> dict[str, Any]:
    planned = _planned_item(queue, index)
    seed = int(planned["seed"])
    output = Path(str(planned["evaluation_root"])).resolve(strict=True)
    launch_path = (output / "launch.json").resolve(strict=True)
    postflight_path = (output / "postflight.json").resolve(strict=True)
    try:
        postflight = evaluator.verify_completed_output(output)
    except (OSError, ValueError, evaluator.MatchedEvaluationError) as exc:
        raise TableBV2ValidationQueueError(
            f"seed {seed} completed evaluation replay failed: {exc}"
        ) from exc
    launch = _read_json(launch_path, label=f"seed {seed} matched launch")
    contract = launch.get("contract")
    if not (
        launch.get("schema") == evaluator.LAUNCH_SCHEMA
        and launch.get("status") == "completed"
        and Path(str(launch.get("output_dir", ""))).resolve(strict=False) == output
        and isinstance(contract, Mapping)
        and contract.get("seed") == seed
        and contract.get("evaluation_profile") == PROFILE
        and contract.get("evaluation_seed") == evaluator.EVAL_SEED
        and contract.get("conditions") == list(PHASE_ORDER)
        and contract.get("training_source_contract")
        == evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT
        and [phase.get("condition") for phase in contract.get("phases", [])]
        == list(PHASE_ORDER)
        and [phase.get("condition") for phase in launch.get("phases", [])]
        == list(PHASE_ORDER)
        and launch.get("completed_conditions") == list(PHASE_ORDER)
        and all(
            phase.get("status") == "completed" and phase.get("returncode") == 0
            for phase in launch.get("phases", [])
        )
    ):
        raise TableBV2ValidationQueueError(
            f"seed {seed} launch is not the exact completed two-phase contract"
        )
    expected_runtime = {
        "python": _file_record(DEFAULT_PYTHON),
        "data_root": str(DEFAULT_DATA_ROOT.resolve(strict=True)),
        **RUNTIME,
    }
    if contract.get("runtime") != expected_runtime:
        raise TableBV2ValidationQueueError(f"seed {seed} runtime drifted")
    training = contract.get("training")
    plan_training = queue["plan"]["training_queue"]
    if not isinstance(training, Mapping) or set(training) != set(PHASE_ORDER):
        raise TableBV2ValidationQueueError(f"seed {seed} training evidence is incomplete")
    for condition in PHASE_ORDER:
        value = training[condition]
        formal = value.get("formal_v2") if isinstance(value, Mapping) else None
        expected_root = planned[f"{condition.lower()}_training_root"]
        if not (
            isinstance(value, Mapping)
            and Path(str(value.get("training_run_root", ""))).resolve(strict=False)
            == Path(expected_root).resolve(strict=True)
            and value.get("training_run_id") == f"{condition}:{seed}"
            and isinstance(formal, Mapping)
            and formal.get("queue_id") == plan_training["queue_id"]
            and formal.get("queue_plan_sha256") == plan_training["plan_sha256"]
            and formal.get("completion_semantic_sha256")
            == plan_training["completion_semantic_sha256"]
        ):
            raise TableBV2ValidationQueueError(
                f"seed {seed} {condition} formal training binding drifted"
            )
    spec_path = (
        Path(queue["plan"]["queue_dir"]) / VALIDATION_SPEC_NAME
    ).resolve(strict=True)
    expected_spec = evaluator._file_record(
        spec_path, role=evaluator.VALIDATION_QUEUE_SPEC_ROLE
    )
    if contract.get("validation_queue_spec") != expected_spec:
        raise TableBV2ValidationQueueError(
            f"seed {seed} does not hash-bind the immutable validation plan"
        )
    bound_specs = [
        record
        for record in contract.get("input_records", [])
        if isinstance(record, Mapping)
        and evaluator.VALIDATION_QUEUE_SPEC_ROLE in record.get("roles", [])
    ]
    expected_input_spec = {
        key: value for key, value in expected_spec.items() if key != "role"
    }
    expected_input_spec["roles"] = [evaluator.VALIDATION_QUEUE_SPEC_ROLE]
    if bound_specs != [expected_input_spec]:
        raise TableBV2ValidationQueueError(
            f"seed {seed} validation plan input binding drifted"
        )
    _launch_source_binding(queue, launch)
    if not (
        isinstance(postflight, Mapping)
        and postflight.get("schema") == evaluator.POSTFLIGHT_SCHEMA
        and postflight.get("status") == "passed"
        and set(postflight.get("conditions", {})) == set(PHASE_ORDER)
    ):
        raise TableBV2ValidationQueueError(f"seed {seed} postflight is not exact")
    return {
        "run_id": planned["run_id"],
        "seed": seed,
        "phase_order": list(PHASE_ORDER),
        "phase_count": len(PHASE_ORDER),
        "evaluation_root": str(output),
        "launch": _file_record(launch_path),
        "postflight": _file_record(postflight_path),
        "contract_sha256": launch["contract_sha256"],
        "advance_gate": "dead_process_plus_queue_bound_two_phase_postflight_replay",
    }


def _advance_launched(queue: MutableMapping[str, Any], index: int) -> None:
    item = queue["items"][index]
    serial_queue._ensure_lease(queue, item, create=False)
    pid = item.get("child_pid")
    running = False if pid == 0 else serial_queue._process_running(
        pid, item.get("child_process_identity")
    )
    if running is True or running is None:
        return
    matches = _matching_processes(_planned_item(queue, index)["command"])
    if len(matches) > 1:
        raise TableBV2ValidationQueueError(
            "multiple evaluation processes match the launched item"
        )
    if matches:
        _bind_process(queue, index, *matches[0])
        return
    evidence = _verify_completed(queue, index)
    item["status"] = "completed"
    item["completed_at_utc"] = _utc_now()
    item["completion_evidence"] = evidence
    _event(queue, "evaluation_completed", index=index, run_id=item["run_id"])
    if all(candidate["status"] == "completed" for candidate in queue["items"]):
        queue["status"] = "completed"
        queue["completed_at_utc"] = _utc_now()
        _event(queue, "queue_completed")
    _save_queue(queue)
    if queue["status"] == "completed":
        persisted = load_queue(Path(queue["plan"]["queue_dir"]))
        if persisted["items"][index].get("completion_evidence") != evidence:
            raise TableBV2ValidationQueueError(
                "final completion evidence was not durably published"
            )
        for replay_index, replay_item in enumerate(persisted["items"]):
            replayed = _verify_completed(persisted, replay_index)
            if replay_item.get("completion_evidence") != replayed:
                raise TableBV2ValidationQueueError(
                    f"{replay_item['run_id']} completion evidence drifted before lease release"
                )
        serial_queue._clear_owned_lease(persisted)


def _fail(queue: MutableMapping[str, Any], index: int, exc: BaseException) -> None:
    item = queue["items"][index]
    item["status"] = "failed"
    item["failed_at_utc"] = _utc_now()
    item["failure_error"] = f"{type(exc).__name__}: {exc}"
    queue["status"] = "failed"
    lease_path = Path(str(queue["plan"]["lease_path"]))
    lease_retained = False
    if lease_path.is_file():
        with contextlib.suppress(TableBV2ValidationQueueError):
            lease_retained = (
                _read_json(lease_path, label="GPU lease").get("queue_id")
                == queue["plan"]["queue_id"]
            )
    queue["failure"] = {
        "index": index,
        "run_id": item["run_id"],
        "error": item["failure_error"],
        "lease_retained_fail_closed": lease_retained,
    }
    _event(queue, "queue_failed", **queue["failure"])
    _save_queue(queue)


def _clear_completed_queue_lease(queue: Mapping[str, Any]) -> None:
    lease_path = Path(str(queue["plan"]["lease_path"]))
    if not lease_path.is_file():
        return
    lease = _read_json(lease_path, label="GPU lease")
    if lease.get("queue_id") == queue["plan"]["queue_id"]:
        serial_queue._clear_owned_lease(queue)


def advance_once(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    if queue["status"] == "completed":
        _clear_completed_queue_lease(queue)
        return queue
    if queue["status"] == "failed":
        return queue
    index = _active_index(queue)
    if index is None:
        raise TableBV2ValidationQueueError("queue has no active item")
    try:
        status = queue["items"][index]["status"]
        if status == "pending":
            _reserve(queue, index)
        elif status == "reserved":
            _launch_reserved(queue, index)
        elif status == "launched":
            _advance_launched(queue, index)
        else:
            raise TableBV2ValidationQueueError(f"cannot advance status {status!r}")
    except (serial_queue.QueueBusyError, KeyboardInterrupt):
        raise
    except BaseException as exc:
        current = load_queue(queue_dir)
        if current["status"] == "completed":
            _fail(current, len(current["items"]) - 1, exc)
        elif current["status"] != "failed":
            current_index = _active_index(current)
            if current_index is not None:
                _fail(current, current_index, exc)
        return load_queue(queue_dir)
    return load_queue(queue_dir)


def run_queue(queue_dir: Path, *, poll_seconds: float, once: bool = False) -> dict[str, Any]:
    if poll_seconds < 0.05:
        raise TableBV2ValidationQueueError("poll_seconds must be at least 0.05")
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    try:
        lock = serial_queue._exclusive_file_lock(
            queue_dir / "supervisor.lock",
            busy_message=f"another Table-B v2 validation supervisor owns {queue_dir}",
        )
        with lock:
            while True:
                queue = advance_once(queue_dir)
                if once or queue["status"] in {"completed", "failed"}:
                    return queue
                time.sleep(poll_seconds)
    except serial_queue.QueueBusyError as exc:
        raise TableBV2ValidationQueueBusy(str(exc)) from exc


def status(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    current_index = _active_index(queue)
    lease_path = Path(queue["plan"]["lease_path"])
    return {
        "schema": STATUS_SCHEMA,
        "observed_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "status": queue["status"],
        "revision": queue["revision"],
        "ordered_seeds": list(SEEDS),
        "phase_order_per_seed": list(PHASE_ORDER),
        "total_phase_count": len(SEEDS) * len(PHASE_ORDER),
        "counts": {
            item_status: sum(item["status"] == item_status for item in queue["items"])
            for item_status in ITEM_STATUSES
        },
        "current_item": (
            dict(queue["items"][current_index]) if current_index is not None else None
        ),
        "lease": (
            _read_json(lease_path, label="GPU lease")
            if lease_path.is_file()
            else {"present": False}
        ),
        "failure": queue.get("failure"),
    }


def verify_queue(queue_dir: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    try:
        queue = load_queue(queue_dir)
        _verify_sources(queue)
    except (OSError, ValueError, TableBV2ValidationQueueError) as exc:
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "failed",
            "errors": [{"scope": "queue", "error": str(exc)}],
            "verified_items": [],
        }
    for index, item in enumerate(queue["items"]):
        if item["status"] != "completed":
            errors.append({"run_id": item["run_id"], "error": f"status={item['status']}"})
            continue
        try:
            evidence = _verify_completed(queue, index)
            if item.get("completion_evidence") != evidence:
                raise TableBV2ValidationQueueError(
                    f"{item['run_id']} stored completion evidence drifted"
                )
            verified.append(evidence)
        except (OSError, ValueError, TableBV2ValidationQueueError) as exc:
            errors.append({"run_id": item["run_id"], "error": str(exc)})
    lease_path = Path(queue["plan"]["lease_path"])
    if queue["status"] == "completed" and lease_path.is_file():
        try:
            lease = _read_json(lease_path, label="GPU lease")
        except TableBV2ValidationQueueError as exc:
            errors.append({"scope": "lease", "error": str(exc)})
        else:
            if lease.get("queue_id") == queue["plan"]["queue_id"]:
                errors.append({"scope": "lease", "error": "completed queue retained its GPU lease"})
    spec_path = Path(queue["plan"]["queue_dir"]) / VALIDATION_SPEC_NAME
    return {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "status": "passed" if queue["status"] == "completed" and not errors else "failed",
        "queue_status": queue["status"],
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "ordered_seeds": list(SEEDS),
        "phase_order_per_seed": list(PHASE_ORDER),
        "total_phase_count": len(SEEDS) * len(PHASE_ORDER),
        "validation_input_spec": _file_record(spec_path),
        "verified_items": verified,
        "errors": errors,
    }


def detach(queue_dir: Path, *, poll_seconds: float) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    queue = load_queue(queue_dir)
    if queue["status"] in {"completed", "failed"}:
        return {"status": queue["status"], "spawned": False}
    supervisors = queue_dir / "supervisors"
    supervisors.mkdir(parents=True, exist_ok=True)
    current_path = supervisors / "current.json"
    if current_path.is_file():
        current = _read_json(current_path, label="detached supervisor")
        pid = current.get("pid")
        if isinstance(pid, int) and serial_queue._process_running(
            pid, current.get("process_identity")
        ) is True:
            return {**current, "status": "already_running"}
    job_dir = supervisors / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{uuid.uuid4().hex[:8]}"
    )
    job_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(DEFAULT_PYTHON.resolve(strict=True)),
        str(Path(__file__).resolve(strict=True)),
        "_supervise",
        str(queue_dir),
        "--job-dir",
        str(job_dir),
        "--poll-seconds",
        str(poll_seconds),
    ]
    log_path = job_dir / "supervisor.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    record = {
        "schema": SUPERVISOR_SCHEMA,
        "status": "launched",
        "created_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "pid": int(process.pid),
        "process_identity": serial_queue._read_process_identity(int(process.pid)),
        "job_dir": str(job_dir),
        "command": command,
        "command_shell": shlex.join(command),
        "log": str(log_path),
    }
    _LOCAL_PROCESSES[int(process.pid)] = process
    _write_json_atomic(current_path, record)
    _write_json_atomic(job_dir / "launch.json", record)
    return record


def _supervise(queue_dir: Path, job_dir: Path, *, poll_seconds: float) -> int:
    status_path = Path(job_dir) / "status.json"
    result: dict[str, Any] = {
        "schema": SUPERVISOR_SCHEMA,
        "status": "running",
        "started_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "pid": os.getpid(),
    }
    _write_json_atomic(status_path, result)
    try:
        queue = run_queue(queue_dir, poll_seconds=poll_seconds)
        result["status"] = queue["status"]
        result["queue_revision"] = queue["revision"]
        returncode = 0 if queue["status"] == "completed" else 1
    except BaseException as exc:
        result["status"] = "supervisor_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        returncode = 1
    result["finished_at_utc"] = _utc_now()
    _write_json_atomic(status_path, result)
    return returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("--json", action="store_true")
    create = subparsers.add_parser("create")
    create.add_argument("queue_dir", type=Path, nargs="?", default=DEFAULT_QUEUE_DIR)
    create.add_argument("--training-queue-dir", type=Path, required=True)
    create.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    create.add_argument("--lease-root", type=Path, default=DEFAULT_LEASE_ROOT)
    create.add_argument("--gpu-key")
    run = subparsers.add_parser("run")
    run.add_argument("queue_dir", type=Path)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--once", action="store_true")
    detached = subparsers.add_parser("detach")
    detached.add_argument("queue_dir", type=Path)
    detached.add_argument("--poll-seconds", type=float, default=30.0)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("queue_dir", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("queue_dir", type=Path)
    supervise = subparsers.add_parser("_supervise")
    supervise.add_argument("queue_dir", type=Path)
    supervise.add_argument("--job-dir", type=Path, required=True)
    supervise.add_argument("--poll-seconds", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "list":
            result = {
                "schema": "pivot.stageb.table_b_v2_validation_catalog/v1",
                "profile": PROFILE,
                "training_source_contract": evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT,
                "ordered_seeds": list(SEEDS),
                "phase_order_per_seed": list(PHASE_ORDER),
                "total_phase_count": len(SEEDS) * len(PHASE_ORDER),
                "default_queue_dir": str(DEFAULT_QUEUE_DIR),
                "default_output_root": str(DEFAULT_OUTPUT_ROOT),
                "runtime": dict(RUNTIME),
            }
            if not args.json:
                print(" ".join(RUN_IDS))
                return 0
        elif args.mode == "create":
            result = create_queue(
                args.queue_dir,
                training_queue_dir=args.training_queue_dir,
                output_root=args.output_root,
                lease_root=args.lease_root,
                gpu_key=args.gpu_key,
            )
        elif args.mode == "run":
            result = run_queue(args.queue_dir, poll_seconds=args.poll_seconds, once=args.once)
        elif args.mode == "detach":
            result = detach(args.queue_dir, poll_seconds=args.poll_seconds)
        elif args.mode == "status":
            result = status(args.queue_dir)
        elif args.mode == "verify":
            result = verify_queue(args.queue_dir)
        elif args.mode == "_supervise":
            return _supervise(args.queue_dir, args.job_dir, poll_seconds=args.poll_seconds)
        else:
            parser.error(f"unknown mode: {args.mode}")
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.mode == "verify":
            return 0 if result.get("status") == "passed" else 1
        return 0 if result.get("status") != "failed" else 1
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
        TableBV2ValidationQueueError,
        serial_queue.QueueContractError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
