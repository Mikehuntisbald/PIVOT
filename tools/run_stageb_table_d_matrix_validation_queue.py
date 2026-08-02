#!/usr/bin/env python3
"""Run the exact 18 formal Table-D matrix-validation jobs serially."""

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

from tools import run_stageb_serial_matrix_queue as serial_queue  # noqa: E402
from tools import run_stageb_table_d_formal_evaluations as formal_eval  # noqa: E402
from tools import run_stageb_table_d_formal_queue as training_queue  # noqa: E402
from tools import stageb_profile_dependency_audit as dependency_audit  # noqa: E402


QUEUE_SCHEMA = "pivot.stageb.table_d_matrix_validation_queue/v1"
PLAN_SCHEMA = "pivot.stageb.table_d_matrix_validation_plan/v1"
SPEC_SCHEMA = "pivot.stageb.table_d_matrix_validation_input/v1"
VERIFICATION_SCHEMA = "pivot.stageb.table_d_matrix_validation_verification/v1"
FINAL_VERIFICATION_SCHEMA = (
    "pivot.stageb.table_d_matrix_validation_final_verification/v1"
)
PROFILE = formal_eval.PROFILE
AGGREGATION_SPEC_NAME = "aggregation_input_spec.json"
EVALUATION_SCOPE_PLAN_NAME = "evaluation_scope_plan.json"
EVALUATION_SCOPE_PLAN_SCHEMA = (
    "pivot.stageb.table_d_matrix_validation_scope_plan/v1"
)
DEFAULT_LEASE_ROOT = serial_queue.DEFAULT_LEASE_ROOT
DEFAULT_AGGREGATOR = REPO_ROOT / "tools/aggregate_stageb_table_d_formal_matrix.py"

FINAL_JOB_IDS = tuple(f"{run_id}/final" for run_id in training_queue.RUN_IDS)
RANK_JOB_IDS = tuple(f"S3:{seed}/rank" for seed in training_queue.SEEDS)
JOB_IDS = (*FINAL_JOB_IDS, *RANK_JOB_IDS)
ITEM_STATUSES = frozenset(
    {"pending", "reserved", "launching", "launched", "completed", "failed"}
)
_LOCAL_PROCESSES: dict[int, subprocess.Popen[Any]] = {}


class TableDValidationQueueError(RuntimeError):
    """The exact Table-D validation queue contract drifted."""


class TableDValidationQueueBusy(TableDValidationQueueError):
    """Another supervisor or shared GPU queue owns the resource."""


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


def _verify_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TableDValidationQueueError(f"{label} file record is missing")
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if _file_record(path) != dict(record):
        raise TableDValidationQueueError(f"{label} file identity changed: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TableDValidationQueueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TableDValidationQueueError(f"{label} must be a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    rendered = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
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


def _source_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [_file_record(path) for path in sorted(set(paths), key=str)]


def _recursive_sources(entry: Path) -> list[Path]:
    try:
        return dependency_audit.recursive_local_python_dependencies(
            (entry.resolve(strict=True).relative_to(REPO_ROOT).as_posix(),),
            repository_root=REPO_ROOT,
        )
    except dependency_audit.ProfileDependencyAuditError as exc:
        raise TableDValidationQueueError(f"dependency closure failed: {exc}") from exc


def _evaluation_sources() -> list[Path]:
    paths = set(formal_eval.evaluator.evaluation_common_code_paths())
    paths.update(formal_eval.evaluator.evaluation_source_provenance_paths("paper"))
    paths.update(_recursive_sources(Path(formal_eval.__file__)))
    return sorted(paths, key=str)


def _training_queue_record(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        verification = training_queue.verify_training_queue(
            queue_dir, persist=False
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        training_queue.TableDFormalQueueError,
    ) as exc:
        raise TableDValidationQueueError(
            f"formal training queue verification failed: {exc}"
        ) from exc
    attestation = (queue_dir / training_queue.COMPLETION_NAME).resolve(strict=True)
    persisted = _read_json(attestation, label="training completion attestation")
    if persisted != verification:
        raise TableDValidationQueueError("training completion attestation drifted")
    return {
        "queue_dir": str(queue_dir),
        "queue_id": verification["queue"]["queue_id"],
        "plan_sha256": verification["queue"]["plan_sha256"],
        "profile": verification["profile"],
        "ordered_run_ids": list(training_queue.RUN_IDS),
        "completion_semantic_sha256": verification["semantic_sha256"],
        "completion_attestation": _file_record(attestation),
        "source_plan": verification["source_plan"],
        "scope_plan": verification["scope_plan"],
        "run_scope_sha256s": {
            run_id: verification["runs"][run_id]["scope_sha256"]
            for run_id in training_queue.RUN_IDS
        },
        "output_root": str(
            Path(verification["runs"][training_queue.RUN_IDS[0]]["run_root"])
            .resolve(strict=True)
            .parents[1]
        ),
    }


def _job_parts(job_id: str) -> tuple[str, int, str]:
    raw_run_id, phase = job_id.rsplit("/", 1)
    row_id, raw_seed = raw_run_id.split(":", 1)
    return row_id, int(raw_seed), phase


def _evaluation_root(output_root: Path, job_id: str) -> Path:
    row_id, seed, phase = _job_parts(job_id)
    return (output_root / row_id / f"seed{seed}" / phase).resolve(strict=False)


def _training_root(
    training: Mapping[str, Any], job_id: str, *, strict: bool = True
) -> Path:
    row_id, seed, _phase = _job_parts(job_id)
    return (Path(training["output_root"]) / row_id / f"seed{seed}").resolve(
        strict=strict
    )


def _evaluation_command(
    *,
    training: Mapping[str, Any],
    job_id: str,
    evaluation_root: Path,
    spec_path: Path,
    strict_training_root: bool = True,
    strict_runtime_paths: bool = True,
) -> list[str]:
    _row_id, _seed, phase = _job_parts(job_id)
    return [
        str(
            formal_eval.evaluator.DEFAULT_PYTHON.resolve(
                strict=strict_runtime_paths
            )
        ),
        str(Path(formal_eval.__file__).resolve(strict=strict_runtime_paths)),
        "run",
        "--training-queue-dir",
        str(training["queue_dir"]),
        "--training-run-root",
        str(_training_root(training, job_id, strict=strict_training_root)),
        "--training-phase",
        phase,
        "--matrix-queue-spec",
        str(spec_path),
        "--output-dir",
        str(evaluation_root),
        "--python",
        str(
            formal_eval.evaluator.DEFAULT_PYTHON.resolve(
                strict=strict_runtime_paths
            )
        ),
        "--data-root",
        str(
            formal_eval.evaluator.DEFAULT_DATA_ROOT.resolve(
                strict=strict_runtime_paths
            )
        ),
        "--device",
        "cuda:0",
    ]


def _aggregation_spec_payload(
    plan: Mapping[str, Any], plan_sha256: str
) -> dict[str, Any]:
    roots = {
        item["job_id"]: item["evaluation_root"] for item in plan["items"]
    }
    return {
        "schema": SPEC_SCHEMA,
        "profile": PROFILE,
        "evaluation_queue_dir": plan["queue_dir"],
        "evaluation_queue_id": plan["queue_id"],
        "evaluation_plan_sha256": plan_sha256,
        "training_queue": {
            key: plan["training_queue"][key]
            for key in (
                "queue_dir",
                "queue_id",
                "plan_sha256",
                "completion_semantic_sha256",
            )
        },
        "expected_train_seeds": list(training_queue.SEEDS),
        "ordered_job_ids": list(JOB_IDS),
        "final_experiments": [
            {
                "id": row_id,
                "evaluation_roots": {
                    str(seed): roots[f"{row_id}:{seed}/final"]
                    for seed in training_queue.SEEDS
                },
            }
            for row_id in training_queue.ROWS
        ],
        "s3_rank_diagnostics": {
            str(seed): roots[f"S3:{seed}/rank"] for seed in training_queue.SEEDS
        },
    }


def build_plan(
    queue_dir: Path,
    *,
    training_queue_dir: Path,
    output_root: Path,
    lease_root: Path = DEFAULT_LEASE_ROOT,
    gpu_key: str | None = None,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    output_root = output_root.expanduser().resolve(strict=False)
    if queue_dir.exists() or output_root.exists():
        raise FileExistsError("validation queue and output roots must both be fresh")
    if (
        queue_dir == output_root
        or queue_dir in output_root.parents
        or output_root in queue_dir.parents
    ):
        raise TableDValidationQueueError(
            "validation queue and evaluation output roots must be disjoint"
        )
    if Path(sys.executable).resolve(strict=True) != formal_eval.evaluator.DEFAULT_PYTHON.resolve(
        strict=True
    ):
        raise TableDValidationQueueError(
            "validation queue must be created with the sealed GDINO Python"
        )
    training = _training_queue_record(training_queue_dir)
    snapshot = serial_queue._snapshot_environment()
    selected_gpu = serial_queue._gpu_key_from_environment(snapshot, gpu_key)
    lease_root = lease_root.expanduser().resolve(strict=False)
    spec_path = (queue_dir / AGGREGATION_SPEC_NAME).resolve(strict=False)
    queue_id = str(uuid.uuid4())
    items = []
    for index, job_id in enumerate(JOB_IDS):
        row_id, seed, phase = _job_parts(job_id)
        evaluation_root = _evaluation_root(output_root, job_id)
        command = _evaluation_command(
            training=training,
            job_id=job_id,
            evaluation_root=evaluation_root,
            spec_path=spec_path,
        )
        process_environment = {
            "CUDA_VISIBLE_DEVICES": selected_gpu,
            "PIVOT_TABLE_D_VALIDATION_QUEUE_ID": queue_id,
            "PIVOT_TABLE_D_VALIDATION_JOB_ID": job_id,
        }
        items.append(
            {
                "index": index,
                "run_id": job_id,
                "job_id": job_id,
                "training_run_id": f"{row_id}:{seed}",
                "row_id": row_id,
                "train_seed": seed,
                "training_phase": phase,
                "training_root": str(_training_root(training, job_id)),
                "training_scope_sha256": training["run_scope_sha256s"][
                    f"{row_id}:{seed}"
                ],
                "evaluation_root": str(evaluation_root),
                "command": command,
                "command_sha256": _canonical_sha(command),
                "command_shell": shlex.join(command),
                "process_environment": process_environment,
                "process_environment_sha256": _canonical_sha(process_environment),
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "queue_id": queue_id,
        "created_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "repository_root": str(REPO_ROOT),
        "profile": PROFILE,
        "ordered_job_ids": list(JOB_IDS),
        "output_root": str(output_root),
        "runner_python": _file_record(formal_eval.evaluator.DEFAULT_PYTHON),
        "evaluation_wrapper": _file_record(Path(formal_eval.__file__)),
        "evaluation_sources": _source_records(_evaluation_sources()),
        "controller_sources": _source_records(_recursive_sources(Path(__file__))),
        "aggregation_sources": _source_records(
            _recursive_sources(DEFAULT_AGGREGATOR)
        ),
        "training_queue": training,
        "gpu_key": selected_gpu,
        "gpu_environment": {"CUDA_VISIBLE_DEVICES": selected_gpu},
        "lease_root": str(lease_root),
        "lease_path": str(serial_queue._lease_path(lease_root, selected_gpu)),
        "aggregation_input_spec": {
            "schema": SPEC_SCHEMA,
            "path": str(spec_path),
        },
        "items": items,
    }


def _semantic_sha(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("semantic_sha256", None)
    return _canonical_sha(value)


def _evaluation_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "repository_root",
        "evaluation_id",
        "output_dir",
        "matrix_validation_queue_spec",
        "source",
        "runtime",
        "protocol",
        "commands",
        "inputs",
        "table_d_formal",
    )
    return {key: plan.get(key) for key in keys}


def _build_evaluation_scope_plan(
    queue_dir: Path, queue: Mapping[str, Any]
) -> dict[str, Any]:
    plan = queue["plan"]
    spec_path = (queue_dir / AGGREGATION_SPEC_NAME).resolve(strict=True)
    items: dict[str, Any] = {}
    for planned in plan["items"]:
        evaluation_plan, _runtime = formal_eval.build_formal_plan(
            training_queue_dir=Path(plan["training_queue"]["queue_dir"]),
            training_run_root=Path(planned["training_root"]),
            training_phase=planned["training_phase"],
            output_dir=Path(planned["evaluation_root"]),
            matrix_queue_spec=spec_path,
        )
        contract = _evaluation_contract(evaluation_plan)
        scope = {
            "job_id": planned["job_id"],
            "training_run_id": planned["training_run_id"],
            "training_phase": planned["training_phase"],
            "training_scope_sha256": planned["training_scope_sha256"],
            "evaluation_root": planned["evaluation_root"],
            "outer_command_sha256": planned["command_sha256"],
            "evaluation_contract_sha256": _canonical_sha(contract),
            "input_identity_sha256": _canonical_sha(
                evaluation_plan["inputs"]
            ),
            "inner_commands_sha256": _canonical_sha(
                evaluation_plan["commands"]
            ),
            "source_identity_sha256": _canonical_sha(
                evaluation_plan["source"]
            ),
            "formal_binding_sha256": _canonical_sha(
                evaluation_plan["table_d_formal"]
            ),
        }
        items[planned["job_id"]] = {
            **scope,
            "scope_sha256": _canonical_sha(scope),
        }
    payload: dict[str, Any] = {
        "schema": EVALUATION_SCOPE_PLAN_SCHEMA,
        "status": "sealed",
        "profile": PROFILE,
        "queue_id": plan["queue_id"],
        "queue_plan_sha256": queue["plan_sha256"],
        "ordered_job_ids": list(JOB_IDS),
        "aggregation_input_spec": _file_record(spec_path),
        "items": items,
    }
    payload["semantic_sha256"] = _semantic_sha(payload)
    return payload


def _validate_evaluation_scope_plan(
    queue_dir: Path, queue: Mapping[str, Any]
) -> dict[str, Any]:
    path = (queue_dir / EVALUATION_SCOPE_PLAN_NAME).resolve(strict=True)
    payload = _read_json(path, label="evaluation scope plan")
    if not (
        payload.get("schema") == EVALUATION_SCOPE_PLAN_SCHEMA
        and payload.get("status") == "sealed"
        and payload.get("profile") == PROFILE
        and payload.get("queue_id") == queue["plan"]["queue_id"]
        and payload.get("queue_plan_sha256") == queue["plan_sha256"]
        and payload.get("ordered_job_ids") == list(JOB_IDS)
        and payload.get("semantic_sha256") == _semantic_sha(payload)
        and set(payload.get("items", {})) == set(JOB_IDS)
    ):
        raise TableDValidationQueueError("evaluation scope plan identity drifted")
    spec = _verify_file_record(
        payload.get("aggregation_input_spec"),
        label="evaluation scope aggregation specification",
    )
    if spec != (queue_dir / AGGREGATION_SPEC_NAME).resolve(strict=True):
        raise TableDValidationQueueError("evaluation scope specification path drifted")
    for planned in queue["plan"]["items"]:
        item = payload["items"][planned["job_id"]]
        scope = dict(item)
        scope.pop("scope_sha256", None)
        if not (
            item.get("scope_sha256") == _canonical_sha(scope)
            and item.get("job_id") == planned["job_id"]
            and item.get("training_run_id") == planned["training_run_id"]
            and item.get("training_phase") == planned["training_phase"]
            and item.get("training_scope_sha256")
            == planned["training_scope_sha256"]
            and item.get("evaluation_root") == planned["evaluation_root"]
            and item.get("outer_command_sha256") == planned["command_sha256"]
        ):
            raise TableDValidationQueueError(
                f"evaluation scope item {planned['job_id']} drifted"
            )
    return payload


def create_queue(
    queue_dir: Path,
    *,
    training_queue_dir: Path,
    output_root: Path,
    lease_root: Path = DEFAULT_LEASE_ROOT,
    gpu_key: str | None = None,
) -> MutableMapping[str, Any]:
    plan = build_plan(
        queue_dir,
        training_queue_dir=training_queue_dir,
        output_root=output_root,
        lease_root=lease_root,
        gpu_key=gpu_key,
    )
    plan_sha256 = _canonical_sha(plan)
    now = _utc_now()
    queue: dict[str, Any] = {
        "schema": QUEUE_SCHEMA,
        "status": "planned",
        "created_at_utc": now,
        "updated_at_utc": now,
        "revision": 0,
        "plan": plan,
        "plan_sha256": plan_sha256,
        "final_verification": None,
        "items": [
            {"index": index, "run_id": job_id, "status": "pending"}
            for index, job_id in enumerate(JOB_IDS)
        ],
        "events": [
            {
                "at_utc": now,
                "event": "queue_created",
                "ordered_job_ids": list(JOB_IDS),
            }
        ],
    }
    spec = _aggregation_spec_payload(plan, plan_sha256)
    queue_dir = Path(plan["queue_dir"])
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
    scope_plan = _build_evaluation_scope_plan(queue_dir, queue)
    _write_json_atomic(
        queue_dir / EVALUATION_SCOPE_PLAN_NAME, scope_plan
    )
    return load_queue(queue_dir)


def _verify_training_record(record: Any, *, deep: bool = False) -> None:
    if not isinstance(record, Mapping):
        raise TableDValidationQueueError("training queue record is missing")
    queue_dir = Path(str(record.get("queue_dir", ""))).resolve(strict=True)
    if deep:
        observed = _training_queue_record(queue_dir)
        if dict(record) != observed:
            raise TableDValidationQueueError("training queue authority drifted")
        return
    attestation = _verify_file_record(
        record.get("completion_attestation"),
        label="training completion attestation",
    )
    try:
        source, scope, generic = training_queue._load_plans(queue_dir)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        training_queue.TableDFormalQueueError,
    ) as exc:
        raise TableDValidationQueueError(
            f"training source/scope plan replay failed: {exc}"
        ) from exc
    persisted = _read_json(attestation, label="training completion attestation")
    source_path = training_queue._verify_file_record(
        record.get("source_plan"), label="training source plan"
    )
    scope_path = training_queue._verify_file_record(
        record.get("scope_plan"), label="training scope plan"
    )
    expected_output = str(
        Path(persisted["runs"][training_queue.RUN_IDS[0]]["run_root"])
        .resolve(strict=True)
        .parents[1]
    )
    if not (
        source_path == (queue_dir / training_queue.SOURCE_PLAN_NAME).resolve(strict=True)
        and scope_path == (queue_dir / training_queue.SCOPE_PLAN_NAME).resolve(strict=True)
        and persisted.get("schema") == training_queue.COMPLETION_SCHEMA
        and persisted.get("status") == "passed"
        and persisted.get("semantic_sha256") == training_queue._semantic_sha(persisted)
        and persisted.get("queue", {}).get("queue_id") == record.get("queue_id")
        and persisted.get("queue", {}).get("plan_sha256")
        == record.get("plan_sha256")
        and persisted.get("semantic_sha256")
        == record.get("completion_semantic_sha256")
        and persisted.get("source_plan") == record.get("source_plan")
        and persisted.get("scope_plan") == record.get("scope_plan")
        and scope.get("source_plan_semantic_sha256") == source["semantic_sha256"]
        and generic.get("status") == "completed"
        and record.get("profile") == training_queue.PROFILE
        and record.get("ordered_run_ids") == list(training_queue.RUN_IDS)
        and record.get("output_root") == expected_output
    ):
        raise TableDValidationQueueError("training queue structural authority drifted")


def _verify_source_records(records: Any, *, label: str) -> None:
    if not isinstance(records, list) or not records:
        raise TableDValidationQueueError(f"{label} source closure is missing")
    paths: set[Path] = set()
    for index, record in enumerate(records):
        path = _verify_file_record(record, label=f"{label} source {index}")
        if path in paths:
            raise TableDValidationQueueError(f"{label} source closure has duplicates")
        paths.add(path)


def _validate_final_verification_record(queue: Mapping[str, Any]) -> None:
    record = queue.get("final_verification")
    if record is None:
        if queue.get("status") == "completed":
            raise TableDValidationQueueError(
                "completed validation queue lacks its final verification receipt"
            )
        return
    expected_keys = {
        "schema",
        "verified_at_utc",
        "queue_id",
        "plan_sha256",
        "ordered_job_ids",
        "completion_evidence_sha256",
        "training_completion_semantic_sha256",
        "evaluation_scope_plan_semantic_sha256",
    }
    digests = (
        record.get("completion_evidence_sha256"),
        record.get("training_completion_semantic_sha256"),
        record.get("evaluation_scope_plan_semantic_sha256"),
    ) if isinstance(record, Mapping) else ()
    try:
        verified_at = datetime.fromisoformat(str(record.get("verified_at_utc")))
    except (TypeError, ValueError):
        verified_at = None
    if not (
        isinstance(record, Mapping)
        and set(record) == expected_keys
        and record.get("schema") == FINAL_VERIFICATION_SCHEMA
        and verified_at is not None
        and verified_at.tzinfo is not None
        and record.get("queue_id") == queue["plan"]["queue_id"]
        and record.get("plan_sha256") == queue["plan_sha256"]
        and record.get("ordered_job_ids") == list(JOB_IDS)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in digests
        )
        and queue.get("status") == "completed"
    ):
        raise TableDValidationQueueError(
            "validation queue final verification receipt drifted"
        )


def _validate_queue(queue: Mapping[str, Any], queue_dir: Path) -> None:
    plan = queue.get("plan")
    if not (
        queue.get("schema") == QUEUE_SCHEMA
        and isinstance(plan, Mapping)
        and plan.get("schema") == PLAN_SCHEMA
        and queue.get("plan_sha256") == _canonical_sha(plan)
        and plan.get("profile") == PROFILE
        and plan.get("ordered_job_ids") == list(JOB_IDS)
        and Path(str(plan.get("queue_dir", ""))).resolve(strict=False) == queue_dir
        and Path(str(plan.get("repository_root", ""))).resolve(strict=False)
        == REPO_ROOT
        and len(plan.get("items", [])) == len(JOB_IDS)
        and [item.get("job_id") for item in plan.get("items", [])]
        == list(JOB_IDS)
    ):
        raise TableDValidationQueueError("validation queue plan identity drifted")
    output_root = Path(str(plan.get("output_root", ""))).resolve(strict=False)
    lease_root = Path(str(plan.get("lease_root", ""))).resolve(strict=False)
    gpu_key = plan.get("gpu_key")
    spec_path = (queue_dir / AGGREGATION_SPEC_NAME).resolve(strict=False)
    if not (
        isinstance(gpu_key, str)
        and bool(gpu_key)
        and plan.get("gpu_environment") == {"CUDA_VISIBLE_DEVICES": gpu_key}
        and Path(str(plan.get("lease_path", ""))).resolve(strict=False)
        == serial_queue._lease_path(lease_root, gpu_key).resolve(strict=False)
        and plan.get("aggregation_input_spec")
        == {"schema": SPEC_SCHEMA, "path": str(spec_path)}
        and queue_dir != output_root
        and queue_dir not in output_root.parents
        and output_root not in queue_dir.parents
    ):
        raise TableDValidationQueueError(
            "validation runtime/lease/output plan drifted"
        )
    items = queue.get("items")
    if not isinstance(items, list) or len(items) != len(JOB_IDS):
        raise TableDValidationQueueError("validation queue item inventory drifted")
    training = plan.get("training_queue")
    run_scopes = (
        training.get("run_scope_sha256s")
        if isinstance(training, Mapping)
        else None
    )
    if not (
        isinstance(run_scopes, Mapping)
        and set(run_scopes) == set(training_queue.RUN_IDS)
    ):
        raise TableDValidationQueueError("training scope inventory drifted")
    active = 0
    seen_incomplete = False
    for index, (item, planned, job_id) in enumerate(
        zip(items, plan["items"], JOB_IDS)
    ):
        row_id, seed, phase = _job_parts(job_id)
        expected_training_root = (
            Path(str(training.get("output_root", "")))
            / row_id
            / f"seed{seed}"
        ).resolve(strict=False)
        expected_evaluation_root = _evaluation_root(output_root, job_id)
        expected_command = _evaluation_command(
            training=training,
            job_id=job_id,
            evaluation_root=expected_evaluation_root,
            spec_path=spec_path,
            strict_training_root=False,
            strict_runtime_paths=False,
        )
        if not (
            isinstance(item, Mapping)
            and item.get("index") == index
            and item.get("run_id") == job_id
            and planned.get("index") == index
            and planned.get("run_id") == job_id
            and planned.get("training_run_id") == f"{row_id}:{seed}"
            and planned.get("row_id") == row_id
            and planned.get("train_seed") == seed
            and planned.get("training_phase") == phase
            and Path(str(planned.get("training_root", ""))).resolve(strict=False)
            == expected_training_root
            and Path(str(planned.get("evaluation_root", ""))).resolve(strict=False)
            == expected_evaluation_root
            and planned.get("training_scope_sha256")
            == run_scopes[f"{row_id}:{seed}"]
            and planned.get("command") == expected_command
            and item.get("status") in ITEM_STATUSES
            and planned.get("command_sha256")
            == _canonical_sha(planned.get("command"))
            and planned.get("command_shell")
            == shlex.join(list(planned.get("command", [])))
            and planned.get("process_environment")
            == {
                "CUDA_VISIBLE_DEVICES": gpu_key,
                "PIVOT_TABLE_D_VALIDATION_QUEUE_ID": plan["queue_id"],
                "PIVOT_TABLE_D_VALIDATION_JOB_ID": job_id,
            }
            and planned.get("process_environment_sha256")
            == _canonical_sha(planned.get("process_environment"))
        ):
            raise TableDValidationQueueError(f"validation item {index} drifted")
        status = item["status"]
        if status == "launched":
            child_pid = item.get("child_pid")
            child_identity = item.get("child_process_identity")
            if not (
                type(child_pid) is int
                and child_pid >= 0
                and isinstance(child_identity, Mapping)
                and item.get("child_process_group_id") == child_pid
                and item.get("child_session_id") == child_pid
                and (
                    (
                        child_pid > 0
                        and child_identity.get("pid") == child_pid
                        and child_identity.get("available") is True
                        and type(child_identity.get("start_time_ticks")) is int
                        and child_identity["start_time_ticks"] > 0
                        and isinstance(child_identity.get("boot_id"), str)
                        and bool(child_identity["boot_id"])
                    )
                    or (child_identity == {} and child_pid == 0)
                )
            ):
                raise TableDValidationQueueError(
                    f"launched validation item {index} process binding drifted"
                )
        if status == "completed" and not isinstance(
            item.get("completion_evidence"), Mapping
        ):
            raise TableDValidationQueueError(
                f"completed validation item {index} lacks evidence"
            )
        if status != "completed":
            seen_incomplete = True
        elif seen_incomplete:
            raise TableDValidationQueueError("completed validation items are not a prefix")
        if status in {"reserved", "launching", "launched"}:
            active += 1
    if active > 1:
        raise TableDValidationQueueError("multiple validation items are active")
    expected_queue_status = queue.get("status")
    if expected_queue_status not in {
        "planned",
        "running",
        "verifying",
        "completed",
        "failed",
    }:
        raise TableDValidationQueueError("validation queue status is invalid")
    if queue.get("revision") is None or type(queue.get("revision")) is not int:
        raise TableDValidationQueueError("validation queue revision is invalid")
    statuses = [item["status"] for item in items]
    if expected_queue_status == "planned" and any(
        status != "pending" for status in statuses
    ):
        raise TableDValidationQueueError("planned validation queue is not pristine")
    if expected_queue_status == "running" and (
        "failed" in statuses or all(status == "completed" for status in statuses)
    ):
        raise TableDValidationQueueError("running validation queue item state drifted")
    if expected_queue_status in {"verifying", "completed"} and any(
        status != "completed" for status in statuses
    ):
        raise TableDValidationQueueError(
            f"{expected_queue_status} validation queue has incomplete items"
        )
    if expected_queue_status == "failed" and statuses.count("failed") != 1:
        raise TableDValidationQueueError("failed validation queue item state drifted")
    if not isinstance(queue.get("events"), list) or not queue["events"]:
        raise TableDValidationQueueError("validation queue event ledger is missing")
    _validate_final_verification_record(queue)


def _load_queue_structural(queue_dir: Path) -> MutableMapping[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    queue = _read_json(queue_dir / "queue.json", label="validation queue")
    _validate_queue(queue, queue_dir)
    return queue


def load_queue(queue_dir: Path) -> MutableMapping[str, Any]:
    queue = _load_queue_structural(queue_dir)
    if queue["status"] == "completed":
        verified = _verify_full_queue_completion(queue)
        _verify_final_receipt_matches(queue, verified)
    else:
        _verify_sources(queue)
    return queue


def _save_queue(queue: MutableMapping[str, Any]) -> None:
    queue["revision"] = int(queue["revision"]) + 1
    queue["updated_at_utc"] = _utc_now()
    path = Path(queue["plan"]["queue_dir"]) / "queue.json"
    _validate_queue(queue, path.parent.resolve(strict=True))
    _write_json_atomic(path, queue)


def _event(queue: MutableMapping[str, Any], name: str, **fields: Any) -> None:
    queue["events"].append({"at_utc": _utc_now(), "event": name, **fields})


def _active_index(queue: Mapping[str, Any]) -> int | None:
    return next(
        (
            index
            for index, item in enumerate(queue["items"])
            if item["status"] != "completed"
        ),
        None,
    )


def _planned_item(queue: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    return queue["plan"]["items"][index]


def _verify_sources(
    queue: Mapping[str, Any], *, deep_training: bool = False
) -> None:
    plan = queue["plan"]
    python = _verify_file_record(
        plan.get("runner_python"), label="evaluation Python"
    )
    if python != formal_eval.evaluator.DEFAULT_PYTHON.resolve(strict=True):
        raise TableDValidationQueueError("evaluation Python path drifted")
    wrapper = _verify_file_record(
        plan.get("evaluation_wrapper"), label="evaluation wrapper"
    )
    if wrapper != Path(formal_eval.__file__).resolve(strict=True):
        raise TableDValidationQueueError("evaluation wrapper path drifted")
    _verify_training_record(plan["training_queue"], deep=deep_training)
    expected_closures = {
        "evaluation_sources": _source_records(_evaluation_sources()),
        "controller_sources": _source_records(_recursive_sources(Path(__file__))),
        "aggregation_sources": _source_records(
            _recursive_sources(DEFAULT_AGGREGATOR)
        ),
    }
    for key, expected in expected_closures.items():
        _verify_source_records(plan[key], label=key)
        if plan[key] != expected:
            raise TableDValidationQueueError(f"{key} source closure drifted")
    queue_dir = Path(plan["queue_dir"]).resolve(strict=True)
    spec_path = (queue_dir / AGGREGATION_SPEC_NAME).resolve(strict=True)
    spec = _read_json(spec_path, label="aggregation input specification")
    if spec != _aggregation_spec_payload(plan, str(queue["plan_sha256"])):
        raise TableDValidationQueueError("aggregation input specification drifted")
    _validate_evaluation_scope_plan(queue_dir, queue)


def _matching_processes(
    command: Sequence[str], process_environment: Mapping[str, str]
) -> list[tuple[int, dict[str, Any]]]:
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
            try:
                environment = serial_queue._process_environment(pid)
            except UnicodeDecodeError as exc:
                raise TableDValidationQueueError(
                    "matching evaluation process environment is not observable"
                ) from exc
            if environment is None:
                raise TableDValidationQueueError(
                    "matching evaluation process environment is not observable"
                )
            if any(
                environment.get(key) != value
                for key, value in process_environment.items()
            ):
                continue
            try:
                identity = serial_queue._sealed_session_leader_identity(
                    pid, serial_queue._read_process_identity(pid)
                )
            except serial_queue.QueueContractError as exc:
                raise TableDValidationQueueError(
                    "matching evaluation process is not its sealed session leader"
                ) from exc
            matches.append((pid, identity))
    return matches


def _bind_process(
    queue: MutableMapping[str, Any],
    index: int,
    pid: int,
    identity: Mapping[str, Any],
) -> None:
    item = queue["items"][index]
    if pid > 0:
        try:
            identity = serial_queue._sealed_session_leader_identity(pid, identity)
        except serial_queue.QueueContractError as exc:
            raise TableDValidationQueueError(
                "evaluation process identity cannot be sealed"
            ) from exc
    item["status"] = "launched"
    item["child_pid"] = int(pid)
    item["child_process_identity"] = dict(identity)
    item["child_process_group_id"] = int(pid)
    item["child_session_id"] = int(pid)
    item["launched_at_utc"] = _utc_now()
    _event(queue, "evaluation_launched", index=index, run_id=item["run_id"], pid=pid)
    _save_queue(queue)


def _reserve(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_sources(queue, deep_training=True)
    item = queue["items"][index]
    planned = _planned_item(queue, index)
    serial_queue._ensure_lease(queue, item, create=True)
    output = Path(planned["evaluation_root"])
    if output.exists():
        raise TableDValidationQueueError(f"evaluation root must be fresh: {output}")
    log = (
        Path(queue["plan"]["queue_dir"])
        / "logs"
        / f"{index:03d}-{item['run_id'].replace(':', '_').replace('/', '_')}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    item["status"] = "reserved"
    item["reserved_at_utc"] = _utc_now()
    item["console_log"] = str(log)
    queue["status"] = "running"
    _event(queue, "evaluation_reserved", index=index, run_id=item["run_id"])
    _save_queue(queue)


def _mark_launching(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_sources(queue, deep_training=True)
    item = queue["items"][index]
    planned = _planned_item(queue, index)
    serial_queue._ensure_lease(queue, item, create=False)
    command = list(planned["command"])
    if _canonical_sha(command) != planned["command_sha256"]:
        raise TableDValidationQueueError("reserved evaluation command drifted")
    output = Path(planned["evaluation_root"])
    if output.exists():
        raise TableDValidationQueueError(
            "reserved evaluation root became non-fresh before launch"
        )
    item["status"] = "launching"
    item["launching_at_utc"] = _utc_now()
    _event(queue, "evaluation_launching", index=index, run_id=item["run_id"])
    _save_queue(queue)


def _advance_launching(queue: MutableMapping[str, Any], index: int) -> None:
    _verify_sources(queue, deep_training=True)
    item = queue["items"][index]
    planned = _planned_item(queue, index)
    serial_queue._ensure_lease(queue, item, create=False)
    command = list(planned["command"])
    if _canonical_sha(command) != planned["command_sha256"]:
        raise TableDValidationQueueError("launching evaluation command drifted")
    process_environment = planned["process_environment"]
    matches = _matching_processes(command, process_environment)
    if len(matches) > 1:
        raise TableDValidationQueueError("multiple processes match one validation item")
    if matches:
        _bind_process(queue, index, *matches[0])
        return
    output = Path(planned["evaluation_root"])
    if output.exists():
        launch = output / "launch_manifest.json"
        if launch.is_file() and _read_json(launch, label="orphan evaluation launch").get(
            "status"
        ) == "completed":
            _bind_process(queue, index, 0, {})
            return
        raise TableDValidationQueueError(
            "evaluation root exists without a recoverable completed launch"
        )
    environment = dict(os.environ)
    environment.update(queue["plan"]["gpu_environment"])
    environment.update(process_environment)
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
        raise TableDValidationQueueError(f"cannot launch evaluation: {exc}") from exc
    pid = int(process.pid)
    _LOCAL_PROCESSES[pid] = process
    _bind_process(queue, index, pid, serial_queue._read_process_identity(pid))


def _verify_completed_item(
    queue: Mapping[str, Any], index: int
) -> dict[str, Any]:
    planned = _planned_item(queue, index)
    training_root = Path(planned["training_root"]).resolve(strict=True)
    evaluation_root = Path(planned["evaluation_root"]).resolve(strict=True)
    spec_path = (
        Path(queue["plan"]["queue_dir"]) / AGGREGATION_SPEC_NAME
    ).resolve(strict=True)
    try:
        evidence = formal_eval.replay_completed_evaluation(
            training_queue_dir=Path(queue["plan"]["training_queue"]["queue_dir"]),
            training_run_root=training_root,
            training_phase=planned["training_phase"],
            evaluation_root=evaluation_root,
            matrix_queue_spec=spec_path,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        formal_eval.TableDFormalEvaluationError,
    ) as exc:
        raise TableDValidationQueueError(
            f"{planned['job_id']} completion replay failed: {exc}"
        ) from exc
    if not (
        evidence.get("status") == "passed"
        and evidence.get("run_id") == planned["training_run_id"]
        and evidence.get("training_phase") == planned["training_phase"]
        and evidence.get("evaluation_root") == str(evaluation_root)
    ):
        raise TableDValidationQueueError(
            f"{planned['job_id']} completion identity drifted"
        )
    launch = _read_json(
        evaluation_root / "launch_manifest.json",
        label=f"{planned['job_id']} completed launch",
    )
    scope_plan = _validate_evaluation_scope_plan(
        Path(queue["plan"]["queue_dir"]), queue
    )
    scope = scope_plan["items"][planned["job_id"]]
    formal_binding = evidence.get("formal_binding")
    if not (
        _canonical_sha(_evaluation_contract(launch))
        == scope["evaluation_contract_sha256"]
        and _canonical_sha(launch.get("inputs"))
        == scope["input_identity_sha256"]
        and _canonical_sha(launch.get("commands"))
        == scope["inner_commands_sha256"]
        and _canonical_sha(launch.get("source"))
        == scope["source_identity_sha256"]
        and _canonical_sha(launch.get("table_d_formal"))
        == scope["formal_binding_sha256"]
        and isinstance(formal_binding, Mapping)
        and formal_binding.get("scope_sha256")
        == scope["training_scope_sha256"]
    ):
        raise TableDValidationQueueError(
            f"{planned['job_id']} differs from its predeclared evaluation scope"
        )
    return {
        "job_id": planned["job_id"],
        "training_run_id": planned["training_run_id"],
        "training_phase": planned["training_phase"],
        "evaluation_root": str(evaluation_root),
        "command_sha256": planned["command_sha256"],
        "scope_sha256": scope["scope_sha256"],
        "formal_evaluation": evidence,
        "advance_gate": (
            "dead_exact_process_plus_completed_launch_plus_replayed_input_"
            "rehash_and_postflight"
        ),
    }


def _verify_full_queue_completion(
    queue: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _verify_sources(queue, deep_training=True)
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(queue["items"]):
        if item["status"] != "completed":
            raise TableDValidationQueueError(
                f"full validation replay found {item['run_id']} status={item['status']}"
            )
        evidence = _verify_completed_item(queue, index)
        if item.get("completion_evidence") != evidence:
            raise TableDValidationQueueError(
                f"{item['run_id']} stored completion evidence drifted"
            )
        verified.append(evidence)
    return verified


def _verify_final_receipt_matches(
    queue: Mapping[str, Any], verified: Sequence[Mapping[str, Any]]
) -> None:
    _validate_final_verification_record(queue)
    receipt = queue["final_verification"]
    scope = _validate_evaluation_scope_plan(
        Path(queue["plan"]["queue_dir"]), queue
    )
    training = queue["plan"]["training_queue"]
    if not (
        receipt["completion_evidence_sha256"] == _canonical_sha(verified)
        and receipt["training_completion_semantic_sha256"]
        == training["completion_semantic_sha256"]
        and receipt["evaluation_scope_plan_semantic_sha256"]
        == scope["semantic_sha256"]
    ):
        raise TableDValidationQueueError(
            "completed validation queue receipt differs from live replay"
        )


def _advance_final_verification(queue: MutableMapping[str, Any]) -> None:
    if queue["status"] != "verifying":
        raise TableDValidationQueueError(
            "final verification requires verifying queue state"
        )
    last_item = queue["items"][-1]
    serial_queue._ensure_lease(queue, last_item, create=False)
    verified = _verify_full_queue_completion(queue)
    scope = _validate_evaluation_scope_plan(
        Path(queue["plan"]["queue_dir"]), queue
    )
    serial_queue._ensure_lease(queue, last_item, create=False)
    queue["final_verification"] = {
        "schema": FINAL_VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "ordered_job_ids": list(JOB_IDS),
        "completion_evidence_sha256": _canonical_sha(verified),
        "training_completion_semantic_sha256": queue["plan"][
            "training_queue"
        ]["completion_semantic_sha256"],
        "evaluation_scope_plan_semantic_sha256": scope["semantic_sha256"],
    }
    queue["status"] = "completed"
    queue["completed_at_utc"] = _utc_now()
    _event(queue, "queue_final_verification_passed")
    _save_queue(queue)
    durable = load_queue(Path(queue["plan"]["queue_dir"]))
    serial_queue._ensure_lease(durable, durable["items"][-1], create=False)
    serial_queue._clear_owned_lease(durable)


def _advance_launched(queue: MutableMapping[str, Any], index: int) -> None:
    item = queue["items"][index]
    _verify_sources(queue)
    serial_queue._ensure_lease(queue, item, create=False)
    pid = item.get("child_pid")
    running = (
        False
        if pid == 0
        else serial_queue._process_running(pid, item.get("child_process_identity"))
    )
    if running is True or running is None:
        return
    _verify_sources(queue, deep_training=True)
    serial_queue._ensure_lease(queue, item, create=False)
    evidence = _verify_completed_item(queue, index)
    item["status"] = "completed"
    item["completed_at_utc"] = _utc_now()
    item["completion_evidence"] = evidence
    _event(queue, "evaluation_completed", index=index, run_id=item["run_id"])
    if all(candidate["status"] == "completed" for candidate in queue["items"]):
        queue["status"] = "verifying"
        queue["verification_started_at_utc"] = _utc_now()
        _event(queue, "queue_final_verification_pending")
    _save_queue(queue)


def _owned_lease_present(queue: Mapping[str, Any]) -> bool:
    return serial_queue._owned_lease_present(queue)


def _terminate_active_processes(
    queue: Mapping[str, Any], index: int
) -> list[dict[str, Any]]:
    item = queue["items"][index]
    planned = _planned_item(queue, index)
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    pid = item.get("child_pid")
    if item.get("status") == "launched" and isinstance(pid, int) and pid > 0:
        candidates.append((pid, item.get("child_process_identity", {})))
    elif item.get("status") == "launching":
        candidates.extend(
            _matching_processes(
                planned["command"], planned["process_environment"]
            )
        )
    reports: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate_pid, identity in candidates:
        if candidate_pid in seen:
            continue
        seen.add(candidate_pid)
        try:
            report = serial_queue._terminate_exact_process_group(
                candidate_pid,
                identity,
                label=f"Table-D validation child {item['run_id']}",
            )
        except serial_queue.QueueContractError as exc:
            raise TableDValidationQueueError(str(exc)) from exc
        process = _LOCAL_PROCESSES.pop(candidate_pid, None)
        if process is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=0.1)
        reports.append(report)
    return reports


def _fail(
    queue: MutableMapping[str, Any], index: int, exc: BaseException
) -> None:
    item = queue["items"][index]
    item["status"] = "failed"
    item["failed_at_utc"] = _utc_now()
    item["failure_error"] = f"{type(exc).__name__}: {exc}"
    queue["status"] = "failed"
    queue["failure"] = {
        "index": index,
        "run_id": item["run_id"],
        "error": item["failure_error"],
        "lease_retained_fail_closed": _owned_lease_present(queue),
    }
    _event(queue, "queue_failed", **queue["failure"])
    _save_queue(queue)


def _terminalize_failure(
    queue: MutableMapping[str, Any], index: int, exc: BaseException
) -> None:
    item = queue["items"][index]
    if item["status"] in {"launching", "launched"}:
        try:
            termination = _terminate_active_processes(queue, index)
        except BaseException as termination_error:
            item["child_termination_blocked"] = {
                "at_utc": _utc_now(),
                "original_error": f"{type(exc).__name__}: {exc}",
                "termination_error": (
                    f"{type(termination_error).__name__}: {termination_error}"
                ),
            }
            _event(
                queue,
                "evaluation_child_termination_blocked",
                index=index,
                run_id=item["run_id"],
                error=str(termination_error),
            )
            _save_queue(queue)
            raise TableDValidationQueueError(
                "queue remains active because its evaluation child could not be "
                f"proven terminated after {type(exc).__name__}: {exc}: "
                f"{termination_error}"
            ) from termination_error
        item["child_termination"] = termination
    _fail(queue, index, exc)


def advance_once(queue_dir: Path) -> MutableMapping[str, Any]:
    queue = _load_queue_structural(queue_dir)
    if queue["status"] == "completed":
        durable = load_queue(queue_dir)
        lease_path = Path(durable["plan"]["lease_path"])
        if lease_path.is_file():
            serial_queue._ensure_lease(
                durable, durable["items"][-1], create=False
            )
            serial_queue._clear_owned_lease(durable)
        return _load_queue_structural(queue_dir)
    if queue["status"] == "failed":
        return queue
    if queue["status"] == "verifying":
        try:
            _advance_final_verification(queue)
        except (serial_queue.QueueBusyError, KeyboardInterrupt):
            raise
        except BaseException as exc:
            current = _load_queue_structural(queue_dir)
            if current["status"] == "verifying":
                _terminalize_failure(current, len(current["items"]) - 1, exc)
            else:
                raise
        return _load_queue_structural(queue_dir)
    index = _active_index(queue)
    if index is None:
        raise TableDValidationQueueError("validation queue has no active item")
    try:
        status = queue["items"][index]["status"]
        if status == "pending":
            _reserve(queue, index)
        elif status == "reserved":
            _mark_launching(queue, index)
        elif status == "launching":
            _advance_launching(queue, index)
        elif status == "launched":
            _advance_launched(queue, index)
        else:
            raise TableDValidationQueueError(f"cannot advance status {status!r}")
    except (serial_queue.QueueBusyError, KeyboardInterrupt):
        raise
    except BaseException as exc:
        current = _load_queue_structural(queue_dir)
        if current["status"] not in {"completed", "failed"}:
            current_index = _active_index(current)
            if current_index is not None:
                _terminalize_failure(current, current_index, exc)
        return _load_queue_structural(queue_dir)
    return _load_queue_structural(queue_dir)


def run_queue(
    queue_dir: Path, *, poll_seconds: float, once: bool = False
) -> MutableMapping[str, Any]:
    if poll_seconds < 0.05:
        raise TableDValidationQueueError("poll_seconds must be at least 0.05")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        with serial_queue._exclusive_file_lock(
            queue_dir / "supervisor.lock",
            busy_message=f"another validation supervisor owns {queue_dir}",
        ):
            while True:
                queue = advance_once(queue_dir)
                if once or queue["status"] in {"completed", "failed"}:
                    return queue
                time.sleep(poll_seconds)
    except serial_queue.QueueBusyError as exc:
        raise TableDValidationQueueBusy(str(exc)) from exc


def queue_status(queue_dir: Path) -> dict[str, Any]:
    queue = _load_queue_structural(queue_dir)
    try:
        if queue["status"] == "completed":
            verified = _verify_full_queue_completion(queue)
            _verify_final_receipt_matches(queue, verified)
        else:
            _verify_sources(queue)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        TableDValidationQueueError,
    ) as exc:
        provenance = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    else:
        provenance = {"status": "passed"}
    index = _active_index(queue)
    lease_path = Path(queue["plan"]["lease_path"])
    return {
        "schema": "pivot.stageb.table_d_matrix_validation_status/v1",
        "observed_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "profile": PROFILE,
        "status": queue["status"],
        "revision": queue["revision"],
        "ordered_job_ids": list(JOB_IDS),
        "provenance": provenance,
        "counts": {
            status: sum(item["status"] == status for item in queue["items"])
            for status in ITEM_STATUSES
        },
        "current_item": dict(queue["items"][index]) if index is not None else None,
        "lease": (
            _read_json(lease_path, label="shared GPU lease")
            if lease_path.is_file()
            else {"present": False}
        ),
        "failure": queue.get("failure"),
    }


def verify_queue(queue_dir: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    try:
        queue = _load_queue_structural(queue_dir)
        _verify_sources(queue, deep_training=True)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        TableDValidationQueueError,
    ) as exc:
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "failed",
            "errors": [{"scope": "queue", "error": str(exc)}],
            "verified_items": [],
        }
    for index, item in enumerate(queue["items"]):
        if item["status"] != "completed":
            errors.append(
                {"job_id": item["run_id"], "error": f"status={item['status']}"}
            )
            continue
        try:
            evidence = _verify_completed_item(queue, index)
            if item.get("completion_evidence") != evidence:
                raise TableDValidationQueueError(
                    f"{item['run_id']} stored completion evidence drifted"
                )
            verified.append(evidence)
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            TableDValidationQueueError,
        ) as exc:
            errors.append({"job_id": item["run_id"], "error": str(exc)})
    if queue["status"] == "completed" and not errors:
        try:
            _verify_final_receipt_matches(queue, verified)
        except TableDValidationQueueError as exc:
            errors.append({"scope": "final_verification", "error": str(exc)})
    lease_path = Path(queue["plan"]["lease_path"])
    if queue["status"] == "completed" and lease_path.is_file():
        try:
            lease = _read_json(lease_path, label="shared GPU lease")
        except TableDValidationQueueError as exc:
            errors.append({"scope": "lease", "error": str(exc)})
        else:
            if lease.get("queue_id") == queue["plan"]["queue_id"]:
                errors.append(
                    {"scope": "lease", "error": "completed queue retained its lease"}
                )
    return {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "status": "passed" if queue["status"] == "completed" and not errors else "failed",
        "queue_status": queue["status"],
        "profile": PROFILE,
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "ordered_job_ids": list(JOB_IDS),
        "aggregation_input_spec": _file_record(
            Path(queue["plan"]["queue_dir"]) / AGGREGATION_SPEC_NAME
        ),
        "evaluation_scope_plan": _file_record(
            Path(queue["plan"]["queue_dir"]) / EVALUATION_SCOPE_PLAN_NAME
        ),
        "final_verification": queue.get("final_verification"),
        "verified_items": verified,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("queue_dir", type=Path)
    create.add_argument("--training-queue-dir", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--lease-root", type=Path, default=DEFAULT_LEASE_ROOT)
    create.add_argument("--gpu-key")
    run = subparsers.add_parser("run")
    run.add_argument("queue_dir", type=Path)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--once", action="store_true")
    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("queue_dir", type=Path)
    for mode in ("status", "verify"):
        child = subparsers.add_parser(mode)
        child.add_argument("queue_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "create":
            queue = create_queue(
                args.queue_dir,
                training_queue_dir=args.training_queue_dir,
                output_root=args.output_root,
                lease_root=args.lease_root,
                gpu_key=args.gpu_key,
            )
            report = queue_status(Path(queue["plan"]["queue_dir"]))
        elif args.mode == "run":
            queue = run_queue(
                args.queue_dir,
                poll_seconds=args.poll_seconds,
                once=args.once,
            )
            report = queue_status(args.queue_dir)
            if queue.get("status") == "failed":
                print(json.dumps(report, indent=2, sort_keys=True))
                return 1
        elif args.mode == "reconcile":
            run_queue(args.queue_dir, poll_seconds=1.0, once=True)
            report = queue_status(args.queue_dir)
        elif args.mode == "status":
            report = queue_status(args.queue_dir)
        elif args.mode == "verify":
            report = verify_queue(args.queue_dir)
            if report["status"] != "passed":
                print(json.dumps(report, indent=2, sort_keys=True))
                return 1
        else:
            parser.error(f"unsupported mode {args.mode!r}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        TableDValidationQueueError,
        serial_queue.QueueContractError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
