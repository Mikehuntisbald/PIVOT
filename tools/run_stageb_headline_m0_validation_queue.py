#!/usr/bin/env python3
"""Run the exact six M0/M0N validation-only evaluations serially.

The queue accepts only two completed, source-sealed training queues: M0 seeds
17/42/73 and M0N seeds 17/42/73.  It freezes the evaluator source closure and
the downstream aggregation source closure before launch, retains one shared
GPU lease, and advances only after replaying the completed matrix-validation
postflight for the current item.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
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

from tools import run_stageb_headline_m0 as training_runner  # noqa: E402
from tools import run_stageb_paper_evaluations as evaluator  # noqa: E402
from tools import run_stageb_serial_matrix_queue as serial_queue  # noqa: E402
from tools import stageb_profile_dependency_audit as profile_dependency  # noqa: E402


QUEUE_SCHEMA = "pivot.stageb.headline_m0_validation_queue/v1"
PLAN_SCHEMA = "pivot.stageb.headline_m0_validation_plan/v1"
SPEC_SCHEMA = "pivot.stageb.headline_m0_validation_input/v1"
VERIFICATION_SCHEMA = "pivot.stageb.headline_m0_validation_verification/v1"
SUPERVISOR_SCHEMA = "pivot.stageb.headline_m0_validation_supervisor/v1"

CONTRACT_IDS = ("M0", "M0N")
SEEDS = (17, 42, 73)
RUN_IDS = tuple(
    f"{contract_id}:{seed}" for contract_id in CONTRACT_IDS for seed in SEEDS
)
PROFILE = evaluator.MATRIX_PROFILE
ITEM_STATUSES = frozenset({"pending", "reserved", "launched", "completed", "failed"})

DEFAULT_QUEUE_DIR = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/headline_m0_m0n_validation_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/evaluations/headline_m0_objective_control"
)
DEFAULT_EVALUATION_RUNNER = REPO_ROOT / "tools/run_stageb_paper_evaluations.py"
DEFAULT_AGGREGATOR = REPO_ROOT / "tools/aggregate_stageb_headline_m0_validation.py"
DEFAULT_LEASE_ROOT = serial_queue.DEFAULT_LEASE_ROOT
AGGREGATION_SPEC_NAME = "aggregation_input_spec.json"

EVALUATION_SOURCE_FORBIDDEN = frozenset(
    {
        "tools/aggregate_stageb_headline_m0_validation.py",
        "tools/aggregate_stageb_matrix_validation.py",
        "tools/aggregate_stageb_paper_results.py",
        "tools/stageb_headline_release_contract.py",
    }
)

_LOCAL_PROCESSES: dict[int, subprocess.Popen[Any]] = {}


class HeadlineValidationQueueError(RuntimeError):
    """The exact M0/M0N validation queue contract was violated."""


class HeadlineValidationQueueBusy(HeadlineValidationQueueError):
    """Another supervisor or GPU queue currently owns the resource."""


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeadlineValidationQueueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HeadlineValidationQueueError(f"{label} must be a JSON object")
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
        "mtime_ns",
    }:
        raise HeadlineValidationQueueError(f"{label} record is invalid")
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    if _file_record(path) != dict(record):
        raise HeadlineValidationQueueError(f"{label} identity changed: {path}")
    return path


def _source_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [_file_record(path) for path in sorted(set(paths), key=str)]


def _evaluation_source_paths() -> list[Path]:
    common = evaluator.evaluation_common_code_paths()
    provenance = evaluator.evaluation_source_provenance_paths("paper")
    if len(common) != 72 or len(set(common)) != 72:
        raise HeadlineValidationQueueError(
            "paper evaluator common closure must remain exactly 72 files"
        )
    if len(provenance) != 4 or len(set(provenance)) != 4:
        raise HeadlineValidationQueueError(
            "paper evaluator provenance closure must remain exactly 4 files"
        )
    if set(common) & set(provenance):
        raise HeadlineValidationQueueError("paper evaluator source profiles overlap")
    paths = sorted(
        set(common).union(provenance, {Path(training_runner.__file__).resolve()}),
        key=str,
    )
    if len(paths) != 77:
        raise HeadlineValidationQueueError(
            "formal M0 evaluator closure must be exactly 72+4+1 files"
        )
    relative = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in paths
        if path.is_relative_to(REPO_ROOT)
    }
    leaked = sorted(relative & EVALUATION_SOURCE_FORBIDDEN)
    if leaked:
        raise HeadlineValidationQueueError(
            "evaluation closure contains downstream consumers: " + ", ".join(leaked)
        )
    return paths


def _recursive_sources(entry: Path) -> list[Path]:
    try:
        return profile_dependency.recursive_local_python_dependencies(
            (entry.relative_to(REPO_ROOT).as_posix(),),
            repository_root=REPO_ROOT,
        )
    except profile_dependency.ProfileDependencyAuditError as exc:
        raise HeadlineValidationQueueError(f"source dependency audit failed: {exc}") from exc


def _training_queue_record(queue_dir: Path, contract_id: str) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        verification = training_runner.verify_training_queue(
            queue_dir, contract_id, require_completed=True
        )
    except (OSError, ValueError, training_runner.HeadlineM0Error) as exc:
        raise HeadlineValidationQueueError(
            f"{contract_id} training queue verification failed: {exc}"
        ) from exc
    return {
        "contract_id": contract_id,
        "queue_dir": str(queue_dir),
        "queue_id": verification["queue_id"],
        "plan_sha256": verification["plan_sha256"],
        "queue_contract_sha256": verification["queue_contract_sha256"],
        "stable_input_closure_digest": verification[
            "stable_input_closure_digest"
        ],
        "ordered_run_ids": list(training_runner.CONTRACTS[contract_id].dedicated_queue_run_ids),
        "manifest_at_creation": _file_record(queue_dir / "queue.json"),
    }


def _verify_training_queue_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise HeadlineValidationQueueError("training queue record is invalid")
    contract_id = str(record.get("contract_id", ""))
    if contract_id not in CONTRACT_IDS:
        raise HeadlineValidationQueueError("training queue contract ID is invalid")
    queue_dir = Path(str(record.get("queue_dir", ""))).resolve(strict=True)
    try:
        verification = training_runner.verify_training_queue(
            queue_dir, contract_id, require_completed=True
        )
    except (OSError, ValueError, training_runner.HeadlineM0Error) as exc:
        raise HeadlineValidationQueueError(
            f"{contract_id} training queue verification failed: {exc}"
        ) from exc
    expected = {
        "queue_id": verification["queue_id"],
        "plan_sha256": verification["plan_sha256"],
        "queue_contract_sha256": verification["queue_contract_sha256"],
        "stable_input_closure_digest": verification[
            "stable_input_closure_digest"
        ],
        "ordered_run_ids": list(
            training_runner.CONTRACTS[contract_id].dedicated_queue_run_ids
        ),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise HeadlineValidationQueueError(
                f"{contract_id} training queue {key} drifted"
            )
    _verify_file_record(record.get("manifest_at_creation"), label=f"{contract_id} queue")
    return verification


def _evaluation_root(output_root: Path, run_id: str) -> Path:
    contract_id, raw_seed = run_id.split(":", 1)
    return (output_root / contract_id / f"seed{int(raw_seed)}").resolve(strict=False)


def _evaluation_command(
    *,
    run_id: str,
    evaluation_root: Path,
    training_queue_dir: Path,
    matrix_queue_spec: Path,
) -> list[str]:
    contract_id, raw_seed = run_id.split(":", 1)
    seed = int(raw_seed)
    training_root = training_runner.CONTRACTS[contract_id].canonical_training_root(seed)
    return [
        str(evaluator.DEFAULT_PYTHON.resolve(strict=True)),
        str(DEFAULT_EVALUATION_RUNNER.resolve(strict=True)),
        "run",
        "--training-run-root",
        str(training_root),
        "--training-queue-dir",
        str(training_queue_dir),
        "--profile",
        PROFILE,
        "--matrix-queue-spec",
        str(matrix_queue_spec),
        "--python",
        str(evaluator.DEFAULT_PYTHON.resolve(strict=True)),
        "--data-root",
        str(evaluator.DEFAULT_DATA_ROOT.resolve(strict=True)),
        "--device",
        "cuda:0",
        "--batch-size",
        "16",
        "--num-workers",
        "4",
        "--log-every",
        "50",
        "--output-dir",
        str(evaluation_root),
    ]


def _aggregation_spec_payload(plan: Mapping[str, Any], plan_sha256: str) -> dict[str, Any]:
    roots = {str(item["run_id"]): str(item["evaluation_root"]) for item in plan["items"]}
    return {
        "schema": SPEC_SCHEMA,
        "evaluation_queue_dir": str(plan["queue_dir"]),
        "evaluation_queue_id": str(plan["queue_id"]),
        "evaluation_plan_sha256": plan_sha256,
        "expected_train_seeds": list(SEEDS),
        "reference_experiment": "M0",
        "candidate_experiment": "M0N",
        "experiments": [
            {
                "id": contract_id,
                "evaluation_roots": {
                    str(seed): roots[f"{contract_id}:{seed}"] for seed in SEEDS
                },
            }
            for contract_id in CONTRACT_IDS
        ],
    }


def build_plan(
    queue_dir: Path,
    *,
    m0_training_queue: Path,
    m0n_training_queue: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lease_root: Path = DEFAULT_LEASE_ROOT,
    gpu_key: str | None = None,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    output_root = output_root.expanduser().resolve(strict=False)
    if queue_dir.exists():
        raise FileExistsError(f"validation queue directory must be fresh: {queue_dir}")
    if output_root.exists():
        raise FileExistsError(f"validation output root must be fresh: {output_root}")
    if Path(sys.executable).resolve(strict=True) != evaluator.DEFAULT_PYTHON.resolve(
        strict=True
    ):
        raise HeadlineValidationQueueError(
            "validation queue must be created with the sealed GDINO Python"
        )
    training_queues = [
        _training_queue_record(m0_training_queue, "M0"),
        _training_queue_record(m0n_training_queue, "M0N"),
    ]
    if training_queues[0]["queue_dir"] == training_queues[1]["queue_dir"]:
        raise HeadlineValidationQueueError("M0 and M0N require separate training queues")
    selected_gpu = serial_queue._gpu_key_from_environment(
        serial_queue._snapshot_environment(), gpu_key
    )
    lease_root = lease_root.expanduser().resolve(strict=False)
    queue_by_contract = {
        record["contract_id"]: Path(record["queue_dir"]) for record in training_queues
    }
    matrix_queue_spec = (queue_dir / AGGREGATION_SPEC_NAME).resolve(strict=False)
    items = []
    for index, run_id in enumerate(RUN_IDS):
        contract_id, raw_seed = run_id.split(":", 1)
        seed = int(raw_seed)
        evaluation_root = _evaluation_root(output_root, run_id)
        command = _evaluation_command(
            run_id=run_id,
            evaluation_root=evaluation_root,
            training_queue_dir=queue_by_contract[contract_id],
            matrix_queue_spec=matrix_queue_spec,
        )
        items.append(
            {
                "index": index,
                "run_id": run_id,
                "contract_id": contract_id,
                "train_seed": seed,
                "training_root": str(
                    training_runner.CONTRACTS[contract_id].canonical_training_root(seed)
                ),
                "training_queue_dir": str(queue_by_contract[contract_id]),
                "training_queue_id": training_queues[CONTRACT_IDS.index(contract_id)][
                    "queue_id"
                ],
                "training_queue_plan_sha256": training_queues[
                    CONTRACT_IDS.index(contract_id)
                ]["plan_sha256"],
                "evaluation_id": f"{contract_id}_seed{seed}",
                "evaluation_root": str(evaluation_root),
                "command": command,
                "command_shell": shlex.join(command),
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "queue_id": str(uuid.uuid4()),
        "created_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "repository_root": str(REPO_ROOT),
        "profile": PROFILE,
        "ordered_run_ids": list(RUN_IDS),
        "output_root": str(output_root),
        "runner_python": _file_record(evaluator.DEFAULT_PYTHON),
        "evaluation_runner": _file_record(DEFAULT_EVALUATION_RUNNER),
        "evaluation_sources": _source_records(_evaluation_source_paths()),
        "controller_sources": _source_records(_recursive_sources(Path(__file__).resolve())),
        "aggregation_sources": _source_records(
            _recursive_sources(DEFAULT_AGGREGATOR.resolve(strict=True))
        ),
        "training_queues": training_queues,
        "gpu_key": selected_gpu,
        "gpu_environment": {"CUDA_VISIBLE_DEVICES": selected_gpu},
        "lease_root": str(lease_root),
        "lease_path": str(serial_queue._lease_path(lease_root, selected_gpu)),
        "aggregation_input_spec": {
            "schema": SPEC_SCHEMA,
            "path": str(queue_dir / AGGREGATION_SPEC_NAME),
        },
        "items": items,
    }
    return plan


def create_queue(
    queue_dir: Path,
    *,
    m0_training_queue: Path,
    m0n_training_queue: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lease_root: Path = DEFAULT_LEASE_ROOT,
    gpu_key: str | None = None,
) -> dict[str, Any]:
    plan = build_plan(
        queue_dir,
        m0_training_queue=m0_training_queue,
        m0n_training_queue=m0n_training_queue,
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
            {"at_utc": now, "event": "queue_created", "ordered_run_ids": list(RUN_IDS)}
        ],
    }
    spec = _aggregation_spec_payload(plan, plan_sha256)
    queue_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = queue_dir.parent / f".{queue_dir.name}.creating-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_json_atomic(staging / "queue.json", queue)
        _write_json_atomic(staging / AGGREGATION_SPEC_NAME, spec)
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
        raise HeadlineValidationQueueError("validation queue schema drifted")
    plan = queue.get("plan")
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise HeadlineValidationQueueError("validation queue plan is invalid")
    if queue.get("plan_sha256") != _canonical_sha(plan):
        raise HeadlineValidationQueueError("validation queue plan SHA-256 mismatch")
    expected_plan_keys = {
        "schema",
        "queue_id",
        "created_at_utc",
        "queue_dir",
        "repository_root",
        "profile",
        "ordered_run_ids",
        "output_root",
        "runner_python",
        "evaluation_runner",
        "evaluation_sources",
        "controller_sources",
        "aggregation_sources",
        "training_queues",
        "gpu_key",
        "gpu_environment",
        "lease_root",
        "lease_path",
        "aggregation_input_spec",
        "items",
    }
    if set(plan) != expected_plan_keys:
        raise HeadlineValidationQueueError("validation queue plan field set drifted")
    raw_queue_id = plan.get("queue_id")
    try:
        parsed_queue_id = uuid.UUID(str(raw_queue_id))
    except (ValueError, AttributeError) as exc:
        raise HeadlineValidationQueueError("validation queue ID is invalid") from exc
    if str(parsed_queue_id) != raw_queue_id:
        raise HeadlineValidationQueueError("validation queue ID is not canonical")
    if Path(str(plan.get("queue_dir", ""))).resolve(strict=False) != queue_dir:
        raise HeadlineValidationQueueError("validation queue opened through another path")
    if (
        Path(str(plan.get("repository_root", ""))).resolve(strict=False) != REPO_ROOT
        or plan.get("profile") != PROFILE
    ):
        raise HeadlineValidationQueueError("validation repository/profile drifted")
    if plan.get("ordered_run_ids") != list(RUN_IDS):
        raise HeadlineValidationQueueError("validation queue run order drifted")
    output_root = Path(str(plan.get("output_root", ""))).resolve(strict=False)
    if not output_root.is_absolute() or output_root == queue_dir or queue_dir in output_root.parents:
        raise HeadlineValidationQueueError("validation output root is invalid")
    runner_python = plan.get("runner_python")
    evaluation_runner = plan.get("evaluation_runner")
    if (
        not isinstance(runner_python, Mapping)
        or Path(str(runner_python.get("path", ""))).resolve(strict=False)
        != evaluator.DEFAULT_PYTHON.resolve(strict=True)
        or not isinstance(evaluation_runner, Mapping)
        or Path(str(evaluation_runner.get("path", ""))).resolve(strict=False)
        != DEFAULT_EVALUATION_RUNNER.resolve(strict=True)
    ):
        raise HeadlineValidationQueueError("validation runner identity drifted")
    training_queues = plan.get("training_queues")
    if (
        not isinstance(training_queues, list)
        or len(training_queues) != len(CONTRACT_IDS)
        or [
            record.get("contract_id") if isinstance(record, Mapping) else None
            for record in training_queues
        ]
        != list(CONTRACT_IDS)
        or len(
            {
                str(Path(str(record.get("queue_dir", ""))).resolve(strict=False))
                for record in training_queues
                if isinstance(record, Mapping)
            }
        )
        != len(CONTRACT_IDS)
    ):
        raise HeadlineValidationQueueError(
            "validation plan requires separate ordered M0/M0N training queues"
        )
    gpu_key = plan.get("gpu_key")
    lease_root = Path(str(plan.get("lease_root", ""))).resolve(strict=False)
    if (
        not isinstance(gpu_key, str)
        or not gpu_key.strip()
        or "," in gpu_key
        or plan.get("gpu_environment") != {"CUDA_VISIBLE_DEVICES": gpu_key}
        or Path(str(plan.get("lease_path", ""))).resolve(strict=False)
        != serial_queue._lease_path(lease_root, gpu_key).resolve(strict=False)
    ):
        raise HeadlineValidationQueueError("validation GPU lease contract drifted")
    expected_spec_path = (queue_dir / AGGREGATION_SPEC_NAME).resolve(strict=False)
    if plan.get("aggregation_input_spec") != {
        "schema": SPEC_SCHEMA,
        "path": str(expected_spec_path),
    }:
        raise HeadlineValidationQueueError("validation aggregation spec binding drifted")
    plan_items = plan.get("items")
    items = queue.get("items")
    if (
        not isinstance(plan_items, list)
        or len(plan_items) != len(RUN_IDS)
        or not isinstance(items, list)
        or len(items) != len(RUN_IDS)
    ):
        raise HeadlineValidationQueueError("validation queue item set is invalid")
    completed_prefix = True
    active = 0
    for index, (planned, item) in enumerate(zip(plan_items, items)):
        contract_id, raw_seed = RUN_IDS[index].split(":", 1)
        seed = int(raw_seed)
        training_queue = training_queues[CONTRACT_IDS.index(contract_id)]
        evaluation_root = _evaluation_root(output_root, RUN_IDS[index])
        expected_command = _evaluation_command(
            run_id=RUN_IDS[index],
            evaluation_root=evaluation_root,
            training_queue_dir=Path(str(training_queue["queue_dir"])),
            matrix_queue_spec=expected_spec_path,
        )
        if (
            not isinstance(planned, Mapping)
            or not isinstance(item, Mapping)
            or set(planned)
            != {
                "index",
                "run_id",
                "contract_id",
                "train_seed",
                "training_root",
                "training_queue_dir",
                "training_queue_id",
                "training_queue_plan_sha256",
                "evaluation_id",
                "evaluation_root",
                "command",
                "command_shell",
            }
            or planned.get("index") != index
            or item.get("index") != index
            or planned.get("run_id") != RUN_IDS[index]
            or item.get("run_id") != RUN_IDS[index]
            or planned.get("contract_id") != contract_id
            or planned.get("train_seed") != seed
            or Path(str(planned.get("training_root", ""))).resolve(strict=False)
            != training_runner.CONTRACTS[contract_id].canonical_training_root(seed)
            or Path(str(planned.get("training_queue_dir", ""))).resolve(strict=False)
            != Path(str(training_queue["queue_dir"])).resolve(strict=False)
            or planned.get("training_queue_id") != training_queue.get("queue_id")
            or planned.get("training_queue_plan_sha256")
            != training_queue.get("plan_sha256")
            or planned.get("evaluation_id") != f"{contract_id}_seed{seed}"
            or Path(str(planned.get("evaluation_root", ""))).resolve(strict=False)
            != evaluation_root
            or planned.get("command") != expected_command
            or planned.get("command_shell") != shlex.join(expected_command)
            or item.get("status") not in ITEM_STATUSES
        ):
            raise HeadlineValidationQueueError(f"validation item {index} drifted")
        status = item["status"]
        if status == "completed":
            if not completed_prefix:
                raise HeadlineValidationQueueError("completed items are not a prefix")
        else:
            completed_prefix = False
        if status in {"reserved", "launched", "failed"}:
            active += 1
    if active > 1:
        raise HeadlineValidationQueueError("validation queue has multiple active items")
    status = queue.get("status")
    if status not in {"planned", "running", "completed", "failed"}:
        raise HeadlineValidationQueueError("validation queue status is invalid")
    if status == "planned" and any(item["status"] != "pending" for item in items):
        raise HeadlineValidationQueueError("planned validation queue has started items")
    if status == "completed" and any(item["status"] != "completed" for item in items):
        raise HeadlineValidationQueueError("completed validation queue is incomplete")
    if status == "failed" and sum(item["status"] == "failed" for item in items) != 1:
        raise HeadlineValidationQueueError("failed validation queue has no failed item")
    spec_path = queue_dir / AGGREGATION_SPEC_NAME
    expected_spec = _aggregation_spec_payload(plan, str(queue["plan_sha256"]))
    if _read_json(spec_path, label="aggregation input spec") != expected_spec:
        raise HeadlineValidationQueueError("aggregation input spec differs from queue plan")


def load_queue(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
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
        "controller_sources": _recursive_sources(Path(__file__).resolve()),
        "aggregation_sources": _recursive_sources(
            DEFAULT_AGGREGATOR.resolve(strict=True)
        ),
    }
    for key, paths in expected_paths.items():
        records = plan.get(key)
        if not isinstance(records, list) or not records:
            raise HeadlineValidationQueueError(f"{key} is empty")
        expected = [str(path.resolve(strict=True)) for path in sorted(set(paths), key=str)]
        observed = [
            str(record.get("path", "")) if isinstance(record, Mapping) else ""
            for record in records
        ]
        if observed != expected:
            raise HeadlineValidationQueueError(
                f"{key} differs from the current recursive dependency closure"
            )
        for index, record in enumerate(records):
            _verify_file_record(record, label=f"{key}[{index}]")
    for record in plan["training_queues"]:
        _verify_training_queue_record(record)


def _planned_item(queue: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    return queue["plan"]["items"][index]


def _active_index(queue: Mapping[str, Any]) -> int | None:
    return next(
        (
            index
            for index, item in enumerate(queue["items"])
            if item["status"] != "completed"
        ),
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
        raise HeadlineValidationQueueError(f"evaluation root must be fresh: {output}")
    log = Path(queue["plan"]["queue_dir"]) / "logs" / f"{index:03d}-{item['run_id'].replace(':', '_')}.log"
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
        raise HeadlineValidationQueueError("multiple evaluation processes match one item")
    if matches:
        _bind_process(queue, index, *matches[0])
        return
    output = Path(str(planned["evaluation_root"]))
    if output.exists():
        launch = output / "launch_manifest.json"
        if launch.is_file() and _read_json(launch, label="orphan launch").get("status") == "completed":
            _bind_process(queue, index, 0, {})
            return
        raise HeadlineValidationQueueError(
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
        raise HeadlineValidationQueueError(f"cannot launch evaluation: {exc}") from exc
    pid = int(process.pid)
    _LOCAL_PROCESSES[pid] = process
    _bind_process(queue, index, pid, serial_queue._read_process_identity(pid))


def _launch_source_binding(queue: Mapping[str, Any], launch: Mapping[str, Any]) -> None:
    inputs = launch.get("inputs")
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(records, list):
        raise HeadlineValidationQueueError("evaluation launch input records are missing")
    roles = {"evaluation_code_dependency", "source_provenance_dependency"}
    observed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise HeadlineValidationQueueError("evaluation launch input record is invalid")
        record_roles = record.get("roles")
        if not isinstance(record_roles, list) or not roles.intersection(record_roles):
            continue
        identity = {
            key: record.get(key) for key in ("path", "sha256", "size_bytes", "mtime_ns")
        }
        path = str(Path(str(identity["path"])).resolve(strict=False))
        if path in observed:
            raise HeadlineValidationQueueError("evaluation launch duplicates a source")
        identity["path"] = path
        observed[path] = identity
    expected = queue["plan"]["evaluation_sources"]
    rendered = [observed[path] for path in sorted(observed)]
    if rendered != expected:
        raise HeadlineValidationQueueError(
            "evaluation launch source closure differs from the immutable queue plan"
        )


def _verify_completed(queue: Mapping[str, Any], index: int) -> dict[str, Any]:
    planned = _planned_item(queue, index)
    run_id = str(planned["run_id"])
    output = Path(str(planned["evaluation_root"])).resolve(strict=True)
    launch_path = (output / "launch_manifest.json").resolve(strict=True)
    rehash_path = (output / "input_rehash.json").resolve(strict=True)
    postflight_path = (output / "postflight.json").resolve(strict=True)
    launch = _read_json(launch_path, label=f"{run_id} launch")
    if (
        launch.get("schema") != evaluator.SCHEMA
        or launch.get("status") != "completed"
        or launch.get("evaluation_id") != planned["evaluation_id"]
        or Path(str(launch.get("output_dir", ""))).resolve(strict=False) != output
    ):
        raise HeadlineValidationQueueError(f"{run_id} launch is not completed/exact")
    protocol = launch.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("profile") != PROFILE:
        raise HeadlineValidationQueueError(f"{run_id} is not matrix validation")
    spec_path = (
        Path(str(queue["plan"]["queue_dir"])) / AGGREGATION_SPEC_NAME
    ).resolve(strict=True)
    expected_spec_record = {
        **_file_record(spec_path),
        "roles": ["matrix_validation_queue_spec"],
    }
    if launch.get("matrix_validation_queue_spec") != expected_spec_record:
        raise HeadlineValidationQueueError(
            f"{run_id} does not hash-bind the immutable validation queue plan"
        )
    launch_inputs = launch.get("inputs")
    input_records = (
        launch_inputs.get("records") if isinstance(launch_inputs, Mapping) else None
    )
    bound_specs = [
        dict(record)
        for record in input_records or []
        if isinstance(record, Mapping)
        and "matrix_validation_queue_spec" in (record.get("roles") or [])
    ]
    if bound_specs != [expected_spec_record]:
        raise HeadlineValidationQueueError(
            f"{run_id} validation queue input binding drifted"
        )
    _launch_source_binding(queue, launch)
    source = launch.get("source")
    training_queue = queue["plan"]["training_queues"][
        CONTRACT_IDS.index(str(planned["contract_id"]))
    ]
    if (
        not isinstance(source, Mapping)
        or source.get("training_run_id") != run_id
        or source.get("training_seed") != planned["train_seed"]
        or Path(str(source.get("training_run_root", ""))).resolve(strict=False)
        != Path(str(planned["training_root"])).resolve(strict=False)
        or source.get("training_queue_id") != training_queue["queue_id"]
        or source.get("training_queue_plan_sha256") != training_queue["plan_sha256"]
    ):
        raise HeadlineValidationQueueError(f"{run_id} training source drifted")
    completed = launch.get("completed_phases")
    if not (
        isinstance(completed, list)
        and len(completed) == 1
        and isinstance(completed[0], Mapping)
        and completed[0].get("phase_id") == "validation_calibration"
        and completed[0].get("status") == "completed"
        and completed[0].get("returncode") == 0
    ):
        raise HeadlineValidationQueueError(f"{run_id} phase did not complete")
    cache = evaluator.HashCache()
    try:
        declared_rehash = evaluator._verify_declared_file(
            launch.get("input_rehash_artifact"), label="input rehash", cache=cache
        )
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise HeadlineValidationQueueError(f"{run_id} input rehash is invalid: {exc}") from exc
    if declared_rehash != rehash_path:
        raise HeadlineValidationQueueError(
            f"{run_id} input rehash artifact path is not canonical"
        )
    input_rehash = _read_json(rehash_path, label=f"{run_id} input rehash")
    try:
        replay = evaluator._rehash_inputs(launch)
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise HeadlineValidationQueueError(
            f"{run_id} input rehash replay failed: {exc}"
        ) from exc
    for key in ("schema", "status", "records"):
        if replay.get(key) != input_rehash.get(key):
            raise HeadlineValidationQueueError(f"{run_id} input rehash replay drifted")
    try:
        declared_postflight = evaluator._verify_declared_file(
            launch.get("postflight_artifact"), label="postflight", cache=cache
        )
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise HeadlineValidationQueueError(f"{run_id} postflight is invalid: {exc}") from exc
    if declared_postflight != postflight_path:
        raise HeadlineValidationQueueError(
            f"{run_id} postflight artifact path is not canonical"
        )
    postflight = _read_json(postflight_path, label=f"{run_id} postflight")
    if (
        postflight.get("schema") != evaluator.POSTFLIGHT_SCHEMA
        or postflight.get("status") != "passed"
        or postflight.get("profile") != PROFILE
        or postflight.get("evaluation_id") != planned["evaluation_id"]
        or postflight.get("input_rehash") != input_rehash
        or launch.get("postflight") != postflight
    ):
        raise HeadlineValidationQueueError(f"{run_id} postflight drifted")
    try:
        replayed = evaluator._postflight_screen(launch, input_rehash)
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise HeadlineValidationQueueError(
            f"{run_id} postflight replay failed: {exc}"
        ) from exc
    observed = dict(postflight)
    expected = dict(replayed)
    observed.pop("validated_at_utc", None)
    expected.pop("validated_at_utc", None)
    if observed != expected:
        raise HeadlineValidationQueueError(f"{run_id} postflight replay differs")
    return {
        "run_id": run_id,
        "evaluation_root": str(output),
        "launch_manifest": _file_record(launch_path),
        "input_rehash": _file_record(rehash_path),
        "postflight": _file_record(postflight_path),
        "advance_gate": "dead_process_plus_completed_launch_plus_replayed_postflight",
    }


def _advance_launched(queue: MutableMapping[str, Any], index: int) -> None:
    item = queue["items"][index]
    serial_queue._ensure_lease(queue, item, create=False)
    pid = item.get("child_pid")
    running = (
        False
        if pid == 0
        else serial_queue._process_running(pid, item.get("child_process_identity"))
    )
    if running is True or running is None:
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
        serial_queue._clear_owned_lease(queue)


def _fail(queue: MutableMapping[str, Any], index: int, exc: BaseException) -> None:
    item = queue["items"][index]
    item["status"] = "failed"
    item["failed_at_utc"] = _utc_now()
    item["failure_error"] = f"{type(exc).__name__}: {exc}"
    queue["status"] = "failed"
    queue["failure"] = {
        "index": index,
        "run_id": item["run_id"],
        "error": item["failure_error"],
        "lease_retained_fail_closed": True,
    }
    _event(queue, "queue_failed", **queue["failure"])
    _save_queue(queue)


def _clear_completed_queue_lease(queue: Mapping[str, Any]) -> None:
    """Clear only this queue's lease; a later queue may own the same GPU path."""

    lease_path = Path(str(queue["plan"]["lease_path"]))
    if not lease_path.is_file():
        return
    lease = _read_json(lease_path, label="GPU lease")
    if lease.get("queue_id") != queue["plan"]["queue_id"]:
        return
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
        raise HeadlineValidationQueueError("queue has no active item")
    try:
        status = queue["items"][index]["status"]
        if status == "pending":
            _reserve(queue, index)
        elif status == "reserved":
            _launch_reserved(queue, index)
        elif status == "launched":
            _advance_launched(queue, index)
        else:
            raise HeadlineValidationQueueError(f"cannot advance status {status!r}")
    except (serial_queue.QueueBusyError, KeyboardInterrupt):
        raise
    except BaseException as exc:
        current = load_queue(queue_dir)
        if current["status"] not in {"completed", "failed"}:
            current_index = _active_index(current)
            if current_index is not None:
                _fail(current, current_index, exc)
        return load_queue(queue_dir)
    return load_queue(queue_dir)


def run_queue(queue_dir: Path, *, poll_seconds: float, once: bool = False) -> dict[str, Any]:
    if poll_seconds < 0.05:
        raise HeadlineValidationQueueError("poll_seconds must be at least 0.05")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        lock = serial_queue._exclusive_file_lock(
            queue_dir / "supervisor.lock",
            busy_message=f"another validation supervisor owns {queue_dir}",
        )
        with lock:
            while True:
                queue = advance_once(queue_dir)
                if once or queue["status"] in {"completed", "failed"}:
                    return queue
                time.sleep(poll_seconds)
    except serial_queue.QueueBusyError as exc:
        raise HeadlineValidationQueueBusy(str(exc)) from exc


def status(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    current_index = _active_index(queue)
    lease_path = Path(queue["plan"]["lease_path"])
    return {
        "schema": "pivot.stageb.headline_m0_validation_status/v1",
        "observed_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "status": queue["status"],
        "revision": queue["revision"],
        "ordered_run_ids": list(RUN_IDS),
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
    except (OSError, ValueError, HeadlineValidationQueueError) as exc:
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
                raise HeadlineValidationQueueError(
                    f"{item['run_id']} stored completion evidence drifted"
                )
            verified.append(evidence)
        except (OSError, ValueError, HeadlineValidationQueueError) as exc:
            errors.append({"run_id": item["run_id"], "error": str(exc)})
    lease_path = Path(queue["plan"]["lease_path"])
    if queue["status"] == "completed" and lease_path.is_file():
        try:
            lease = _read_json(lease_path, label="GPU lease")
        except HeadlineValidationQueueError as exc:
            errors.append({"scope": "lease", "error": str(exc)})
        else:
            if lease.get("queue_id") == queue["plan"]["queue_id"]:
                errors.append(
                    {"scope": "lease", "error": "completed queue retained its GPU lease"}
                )
    return {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "status": "passed" if queue["status"] == "completed" and not errors else "failed",
        "queue_status": queue["status"],
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "ordered_run_ids": list(RUN_IDS),
        "aggregation_input_spec": _file_record(
            Path(queue["plan"]["queue_dir"]) / AGGREGATION_SPEC_NAME
        ),
        "verified_items": verified,
        "errors": errors,
    }


def detach(queue_dir: Path, *, poll_seconds: float) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
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
        str(evaluator.DEFAULT_PYTHON.resolve(strict=True)),
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
    status_path = job_dir / "status.json"
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
    create.add_argument("queue_dir", type=Path)
    create.add_argument("--m0-training-queue", type=Path, required=True)
    create.add_argument("--m0n-training-queue", type=Path, required=True)
    create.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    create.add_argument("--lease-root", type=Path, default=DEFAULT_LEASE_ROOT)
    create.add_argument("--gpu-key")
    run = subparsers.add_parser("run")
    run.add_argument("queue_dir", type=Path)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--once", action="store_true")
    detach_parser = subparsers.add_parser("detach")
    detach_parser.add_argument("queue_dir", type=Path)
    detach_parser.add_argument("--poll-seconds", type=float, default=30.0)
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
            payload = {
                "schema": "pivot.stageb.headline_m0_validation_catalog/v1",
                "profile": PROFILE,
                "ordered_run_ids": list(RUN_IDS),
                "training_queues_separate": True,
                "reference_experiment": "M0",
                "candidate_experiment": "M0N",
            }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(" ".join(RUN_IDS))
            return 0
        if args.mode == "create":
            result = create_queue(
                args.queue_dir,
                m0_training_queue=args.m0_training_queue,
                m0n_training_queue=args.m0n_training_queue,
                output_root=args.output_root,
                lease_root=args.lease_root,
                gpu_key=args.gpu_key,
            )
        elif args.mode == "run":
            result = run_queue(
                args.queue_dir, poll_seconds=args.poll_seconds, once=args.once
            )
        elif args.mode == "detach":
            result = detach(args.queue_dir, poll_seconds=args.poll_seconds)
        elif args.mode == "status":
            result = status(args.queue_dir)
        elif args.mode == "verify":
            result = verify_queue(args.queue_dir)
        elif args.mode == "_supervise":
            return _supervise(
                args.queue_dir, args.job_dir, poll_seconds=args.poll_seconds
            )
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
        HeadlineValidationQueueError,
        serial_queue.QueueContractError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
