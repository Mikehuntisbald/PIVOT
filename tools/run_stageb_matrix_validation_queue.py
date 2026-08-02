#!/usr/bin/env python3
"""Predeclare and serially run the sealed Table-C validation matrix.

The queue consumes the two immutable Table-C training queue plans.  Creation
is allowed while the second training queue is still running so evaluation
roots and ordering can be committed before any result exists.  No evaluation
is launched until both training queues are completed, their final verification
passes, and a fresh 33-run training-source attestation has been written.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_paper_evaluations as evaluator  # noqa: E402
from tools import run_stageb_serial_matrix_queue as training_queue  # noqa: E402
from tools import stageb_profile_dependency_audit as profile_dependency  # noqa: E402


QUEUE_SCHEMA = "pivot.stageb.matrix_validation_queue/v1"
PLAN_SCHEMA = "pivot.stageb.matrix_validation_queue_plan/v1"
CONTRACT_SCHEMA = "pivot.stageb.matrix_validation_predeclared_contract/v1"
TRAINING_ATTESTATION_SCHEMA = (
    "pivot.stageb.matrix_validation_training_attestation/v2"
)
VERIFICATION_SCHEMA = "pivot.stageb.matrix_validation_queue_verification/v1"
STATUS_SCHEMA = "pivot.stageb.matrix_validation_queue_status/v1"
SUPERVISOR_SCHEMA = "pivot.stageb.matrix_validation_supervisor/v1"
FINAL_VERIFICATION_SCHEMA = (
    "pivot.stageb.matrix_validation_final_verification/v1"
)
FORMAL_PROVENANCE_SCOPE = "formal"
TEST_ONLY_PROVENANCE_SCOPE = "test_only"
_TEST_ONLY_CREATE_CAPABILITY = object()
_AUTHORIZED_TEST_QUEUE_IDS: set[str] = set()

PROFILE = evaluator.MATRIX_PROFILE
ROWS = tuple(f"L{index}" for index in range(11))
SEEDS = (17, 42, 73)
EXPECTED_RUN_IDS = tuple(
    f"{row}:{seed}" for seed in SEEDS for row in ROWS
)

CONTROLLER_SOURCE_RELATIVE_PATHS = (
    "tools/compare_stageb_fpr95_records.py",
    "tools/recover_stageb_serial_matrix_pretraining_failure.py",
    "tools/run_stageb_matrix_validation_queue.py",
    "tools/run_stageb_paper_evaluations.py",
    "tools/run_stageb_serial_matrix_queue.py",
    "tools/run_stageb_token_ablation_matrix.py",
    "tools/stageb_dependency_audit.py",
    "tools/stageb_eval_records.py",
    "tools/stageb_evaluation_source_contracts.py",
    "tools/stageb_profile_dependency_audit.py",
    "tools/stageb_ref_split_contract.py",
    "tools/stageb_screen_calibration.py",
)
CONTROLLER_SOURCE_PRUNED_EDGES = (
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/run_stageb_paper_ablation_matrices.py",
    ),
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/stageb_headline_release_contract.py",
    ),
    (
        "tools/run_stageb_token_ablation_matrix.py",
        "tools/run_stageb_paper_ablation_matrices.py",
    ),
)
PROFILE_SUPPORT_RELATIVE_PATHS = (
    "tools/run_stageb_paper_ablation_matrices.py",
    "tools/stageb_headline_release_contract.py",
)
LATE_BOUND_SOURCE_RELATIVE_PATHS = frozenset(
    {
        "tools/aggregate_stageb_matrix_validation.py",
        "tools/aggregate_stageb_table_d_diagnostics.py",
        "tools/build_stageb_b58_exposure_receipt.py",
        "tools/build_stageb_paper_ablation_completion_receipt.py",
        "tools/stageb_headline_release_contract.py",
    }
)

DEFAULT_QUEUE_DIR = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/queues/table_c_matrix_validation_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/evaluations/matrix_validation"
)
AGGREGATION_INPUT_SCHEMA = "pivot.stageb.matrix_validation_input/v3"
DEFAULT_AGGREGATION_INPUT_SPEC = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/table_c_matrix_validation_input.json"
)
DEFAULT_LEASE_ROOT = training_queue.DEFAULT_LEASE_ROOT
DEFAULT_EVALUATION_RUNNER = REPO_ROOT / "tools/run_stageb_paper_evaluations.py"
DEFAULT_TRAINING_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/token_ablation_frozen_v2"
)
DEFAULT_TRAINING_QUEUE_DIRS = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/queues/"
    "table_c_screen_l0_l4_seed17_b40_u1000_frozen_v2",
    REPO_ROOT
    / "outputs/paper_cvpr_v1/queues/"
    "table_c_remaining_28_b40_u1000_frozen_v2",
)
TABLE_C_PRETRAINING_RECOVERY_RECEIPT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/recovery/table_c_remaining_28/"
    "L2_seed42_attempt000/recovery_receipt.json"
)
LOCKED_TRAINING_QUEUES = {
    "completed_l0_l4_seed17": {
        "queue_id": "3e5a961a-f2da-45ba-8e44-94740f4baee9",
        "plan_sha256": (
            "63619de10c9e41d2ecc5177242b4b3bbf175d57c3c9cdcd7013b1185a53e6cde"
        ),
        "run_ids": tuple(f"L{index}:17" for index in range(5)),
    },
    "remaining_table_c": {
        "queue_id": "ffcc3e46-ca1d-45d0-9fbd-22e5db14ac9f",
        "plan_sha256": (
            "b4b8ef280fcbd67dbf82fc59d6c90f63c9c3573976b8950c06f1e84dbb31c2cc"
        ),
        "run_ids": (
            *(f"L{index}:17" for index in range(5, 11)),
            *(f"L{index}:{seed}" for seed in (42, 73) for index in range(11)),
        ),
    },
}

ITEM_STATUSES = frozenset(
    {"pending", "reserved", "launching", "launched", "completed", "failed"}
)
_LOCAL_EVALUATION_PROCESSES: dict[int, subprocess.Popen[Any]] = {}
_LOCAL_SUPERVISOR_PROCESSES: dict[int, subprocess.Popen[Any]] = {}
_VALIDATED_SOURCE_CONTRACT_CACHE: set[str] = set()
CHILD_TERMINATION_GRACE_SECONDS = 5.0
CHILD_TERMINATION_POLL_SECONDS = 0.05


class MatrixQueueError(RuntimeError):
    """A persisted queue or its evidence violates the sealed contract."""


class MatrixQueueBusy(MatrixQueueError):
    """A second supervisor attempted to mutate the same queue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise MatrixQueueError(f"evidence path is not a file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _content_file_record(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    return {
        "path": str(path.expanduser().resolve(strict=False)),
        "sha256": hashlib.sha256(rendered).hexdigest(),
        "size_bytes": len(rendered),
    }


def _verify_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise MatrixQueueError(f"{label} file record is missing")
    try:
        path = Path(str(record.get("path", ""))).resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise MatrixQueueError(f"{label} file is missing") from exc
    if _file_record(path) != dict(record):
        raise MatrixQueueError(f"{label} file identity drifted: {path}")
    return path


def _verify_content_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise MatrixQueueError(f"{label} content record is invalid")
    try:
        path = Path(str(record["path"])).expanduser().resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise MatrixQueueError(f"{label} file is missing") from exc
    if not path.is_file():
        raise MatrixQueueError(f"{label} is not a file: {path}")
    observed = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }
    if observed != dict(record):
        raise MatrixQueueError(f"{label} content identity drifted: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixQueueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixQueueError(f"{label} is not a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    training_queue._write_json_atomic(path, payload)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise MatrixQueueError(
            "atomic fresh-only publication requires renameat2(RENAME_NOREPLACE)"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(destination)
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise FileExistsError(f"artifact must be fresh: {path}") from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        _fsync_directory(path.parent)


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MatrixQueueBusy(f"another supervisor owns {path.parent}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _exact_run_root(run_id: str) -> Path:
    row, raw_seed = run_id.split(":", 1)
    return (
        DEFAULT_TRAINING_OUTPUT_ROOT / row / f"seed{int(raw_seed)}"
    ).resolve(strict=False)


def _exact_eval_root(output_root: Path, run_id: str) -> Path:
    row, raw_seed = run_id.split(":", 1)
    return (output_root / row / f"seed{int(raw_seed)}").resolve(
        strict=False
    )


def _locked_queue_role(queue: Mapping[str, Any]) -> str:
    plan = queue.get("plan")
    if not isinstance(plan, Mapping):
        raise MatrixQueueError("training queue has no immutable plan")
    matches = [
        role
        for role, expected in LOCKED_TRAINING_QUEUES.items()
        if plan.get("queue_id") == expected["queue_id"]
        and queue.get("plan_sha256") == expected["plan_sha256"]
    ]
    if len(matches) != 1:
        raise MatrixQueueError("training queue ID/plan SHA is not locked Table-C")
    role = matches[0]
    planned = plan.get("items")
    observed = [
        item.get("run_id") if isinstance(item, Mapping) else None
        for item in planned
    ] if isinstance(planned, list) else []
    if observed != list(LOCKED_TRAINING_QUEUES[role]["run_ids"]):
        raise MatrixQueueError(f"training queue {role} run order drifted")
    if any(
        not isinstance(item, Mapping) or item.get("runner") != "token"
        for item in planned or []
    ):
        raise MatrixQueueError(f"training queue {role} contains non-token items")
    return role


def _training_queue_snapshot(queue_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        queue = training_queue.load_queue(queue_dir)
    except (OSError, ValueError, training_queue.QueueContractError) as exc:
        raise MatrixQueueError(f"training queue validation failed: {exc}") from exc
    role = _locked_queue_role(queue)
    plan = queue["plan"]
    record = {
        "role": role,
        "queue_dir": str(queue_dir),
        "queue_id": plan["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "ordered_run_ids": list(LOCKED_TRAINING_QUEUES[role]["run_ids"]),
        "status_at_creation": queue["status"],
        "pretraining_recovery": _verify_pretraining_recovery(
            queue_dir, queue, role
        ),
    }
    return record, queue


def _verify_pretraining_recovery(
    queue_dir: Path, queue: Mapping[str, Any], role: str
) -> dict[str, Any] | None:
    if role != "remaining_table_c":
        return None
    from tools import recover_stageb_serial_matrix_pretraining_failure as recovery

    expected = LOCKED_TRAINING_QUEUES[role]
    try:
        result = recovery.verify_recovery(
            queue_dir,
            TABLE_C_PRETRAINING_RECOVERY_RECEIPT,
        )
    except (OSError, ValueError, recovery.RecoveryError) as exc:
        raise MatrixQueueError(
            f"Table-C pretraining recovery replay failed: {exc}"
        ) from exc
    expected_receipt = _file_record(TABLE_C_PRETRAINING_RECOVERY_RECEIPT)
    expected_verifier = _file_record(Path(recovery.__file__))
    if (
        result.get("status") != "passed"
        or result.get("queue_id") != expected["queue_id"]
        or result.get("plan_sha256") != expected["plan_sha256"]
        or result.get("run_id") != "L2:42"
        or result.get("archived_evidence_verified") is not True
        or result.get("semantic_replay") != recovery.SEMANTIC_REPLAY_PROOF
        or result.get("verifier_source") != expected_verifier
    ):
        raise MatrixQueueError("Table-C pretraining recovery identity drifted")
    if (
        queue.get("status") == "completed"
        and result.get("current_item_status") != "completed"
    ):
        raise MatrixQueueError(
            "completed Table-C queue has an incomplete recovered L2 item"
        )
    events = [
        event
        for event in queue.get("events", [])
        if isinstance(event, Mapping)
        and event.get("event") == recovery.RECOVERY_EVENT
    ]
    recovered_items = [
        item
        for item in queue.get("items", [])
        if isinstance(item, Mapping)
        and item.get("pretraining_recovery_receipts") is not None
    ]
    if (
        len(events) != 1
        or events[0].get("run_id") != "L2:42"
        or events[0].get("failed_revision") != 590
        or events[0].get("receipt") != expected_receipt
        or len(recovered_items) != 1
        or recovered_items[0].get("run_id") != "L2:42"
        or recovered_items[0].get("pretraining_recovery_receipts")
        != [expected_receipt]
    ):
        raise MatrixQueueError(
            "Table-C queue lacks the single sealed pretraining recovery event"
        )
    return {
        "run_id": "L2:42",
        "failed_revision": 590,
        "receipt": expected_receipt,
        "verifier_source": expected_verifier,
        "receipt_sha256": result.get("receipt_sha256"),
    }


def _repo_relative_source_paths(
    paths: Iterable[Path], *, repository_root: Path = REPO_ROOT
) -> frozenset[str]:
    repository_root = repository_root.expanduser().resolve(strict=True)
    relative: set[str] = set()
    for path in paths:
        try:
            relative.add(
                path.resolve(strict=True).relative_to(repository_root).as_posix()
            )
        except ValueError as exc:
            raise MatrixQueueError(
                f"profiled source escapes repository root: {path}"
            ) from exc
    return frozenset(relative)


def _controller_source_paths(
    *, repository_root: Path = REPO_ROOT
) -> list[Path]:
    repository_root = repository_root.expanduser().resolve(strict=True)
    try:
        paths = profile_dependency.recursive_local_python_dependencies(
            ("tools/run_stageb_matrix_validation_queue.py",),
            repository_root=repository_root,
            pruned_edges=CONTROLLER_SOURCE_PRUNED_EDGES,
        )
    except profile_dependency.ProfileDependencyAuditError as exc:
        raise MatrixQueueError(
            f"matrix queue controller dependency profile failed: {exc}"
        ) from exc
    observed = tuple(
        path.relative_to(repository_root).as_posix() for path in paths
    )
    if observed != CONTROLLER_SOURCE_RELATIVE_PATHS:
        raise MatrixQueueError(
            "matrix queue controller dependency closure drifted: "
            f"expected {CONTROLLER_SOURCE_RELATIVE_PATHS}, found {observed}"
        )
    return paths


def _profile_support_source_paths(
    *, repository_root: Path = REPO_ROOT
) -> list[Path]:
    repository_root = repository_root.expanduser().resolve(strict=True)
    configured_edges = {
        *CONTROLLER_SOURCE_PRUNED_EDGES,
        *evaluator.EVAL_COMMON_CODE_PRUNED_EDGES,
        *evaluator.SOURCE_PROVENANCE_PRUNED_EDGES["token"],
    }
    endpoints = {relative for edge in configured_edges for relative in edge}
    execution_relative = set(CONTROLLER_SOURCE_RELATIVE_PATHS)
    support = tuple(sorted(endpoints - execution_relative))
    # Evaluator-only execution paths are removed below after resolving the
    # canonical 75-file profile; these two files are the only pruned targets
    # deliberately absent from both formal execution closures.
    candidates = [
        (repository_root / relative).resolve(strict=True)
        for relative in PROFILE_SUPPORT_RELATIVE_PATHS
    ]
    if support != PROFILE_SUPPORT_RELATIVE_PATHS:
        raise MatrixQueueError(
            "matrix dependency-profile support endpoints drifted: "
            f"expected {PROFILE_SUPPORT_RELATIVE_PATHS}, found {support}"
        )
    return candidates


def _child_evaluation_source_paths(
    *,
    evaluation_runner: Path,
    injected_paths: Sequence[Path] | None,
    repository_root: Path = REPO_ROOT,
) -> list[Path]:
    repository_root = repository_root.expanduser().resolve(strict=True)
    if injected_paths is None:
        try:
            if repository_root == REPO_ROOT:
                common_paths = evaluator.evaluation_common_code_paths()
                provenance_paths = evaluator.evaluation_source_provenance_paths(
                    "token"
                )
            else:
                common_paths = profile_dependency.recursive_local_python_dependencies(
                    evaluator.EVAL_COMMON_CODE_ENTRIES,
                    repository_root=repository_root,
                    include_paths=evaluator.EVAL_COMMON_CODE_INCLUDE,
                    pruned_edges=evaluator.EVAL_COMMON_CODE_PRUNED_EDGES,
                )
                provenance_paths = (
                    profile_dependency.recursive_local_python_dependencies(
                        evaluator.SOURCE_PROVENANCE_ENTRIES["token"],
                        repository_root=repository_root,
                        pruned_edges=evaluator.SOURCE_PROVENANCE_PRUNED_EDGES[
                            "token"
                        ],
                    )
                )
        except (
            OSError,
            evaluator.PaperEvaluationError,
            profile_dependency.ProfileDependencyAuditError,
        ) as exc:
            raise MatrixQueueError(
                f"evaluation source closure failed: {exc}"
            ) from exc
        common_set = set(common_paths)
        provenance_set = set(provenance_paths)
        if (
            len(common_paths) != 72
            or len(common_set) != 72
            or len(provenance_paths) != 3
            or len(provenance_set) != 3
            or common_set & provenance_set
            or len(common_set | provenance_set) != 75
        ):
            raise MatrixQueueError(
                "token evaluation source profile must be exactly 72 disjoint "
                "common files plus 3 provenance files"
            )
        source_paths = [*common_paths, *provenance_paths]
    else:
        source_paths = [
            path.expanduser().resolve(strict=True) for path in injected_paths
        ]
    paths = sorted({*source_paths, evaluation_runner}, key=str)
    if injected_paths is None:
        if len(paths) != 75:
            raise MatrixQueueError(
                "canonical evaluation runner is absent from the exact 75-file "
                "token child source profile"
            )
        relative = _repo_relative_source_paths(
            paths, repository_root=repository_root
        )
        late_bound = sorted(relative & LATE_BOUND_SOURCE_RELATIVE_PATHS)
        if late_bound:
            raise MatrixQueueError(
                "child evaluation source profile contains late-bound artifacts: "
                + ", ".join(late_bound)
            )
        controller = (
            repository_root / "tools/run_stageb_matrix_validation_queue.py"
        ).resolve(strict=True)
        if controller in paths:
            raise MatrixQueueError(
                "queue controller leaked into the child evaluation source profile"
            )
    return paths


def _creation_provenance_scope(
    evaluation_source_paths: Sequence[Path] | None,
    test_only_capability: object | None,
) -> str:
    if evaluation_source_paths is None:
        if test_only_capability is not None:
            raise MatrixQueueError(
                "test-only capability is invalid without an injected source closure"
            )
        return FORMAL_PROVENANCE_SCOPE
    if test_only_capability is not _TEST_ONLY_CREATE_CAPABILITY:
        raise MatrixQueueError(
            "injected evaluation sources require the in-process test-only capability"
        )
    return TEST_ONLY_PROVENANCE_SCOPE


def _require_plan_provenance_scope(plan: Mapping[str, Any]) -> str:
    scope = plan.get("provenance_scope")
    if scope == FORMAL_PROVENANCE_SCOPE:
        return scope
    queue_id = plan.get("queue_id")
    if (
        scope != TEST_ONLY_PROVENANCE_SCOPE
        or not isinstance(queue_id, str)
        or queue_id not in _AUTHORIZED_TEST_QUEUE_IDS
    ):
        raise MatrixQueueError(
            "test-only queue provenance lacks its in-process capability"
        )
    return scope


def _aggregation_input_spec_binding(path: Path) -> dict[str, Any]:
    return {
        "schema": AGGREGATION_INPUT_SCHEMA,
        "path": str(path.expanduser().resolve(strict=False)),
    }


def _aggregation_input_spec_payload(
    plan: Mapping[str, Any], plan_sha: str
) -> dict[str, Any]:
    items = plan.get("items")
    if not isinstance(items, list) or [
        item.get("run_id") if isinstance(item, Mapping) else None
        for item in items
    ] != list(EXPECTED_RUN_IDS):
        raise MatrixQueueError(
            "cannot project aggregation input from a noncanonical matrix plan"
        )
    roots = {
        str(item["run_id"]): str(item["evaluation_root"])
        for item in items
    }
    return {
        "schema": AGGREGATION_INPUT_SCHEMA,
        "expected_train_seeds": list(SEEDS),
        "evaluation_queue_dir": str(plan["queue_dir"]),
        "evaluation_queue_id": str(plan["queue_id"]),
        "evaluation_plan_sha256": plan_sha,
        "evaluation_provenance_scope": plan["provenance_scope"],
        "reference_experiment": "L0",
        "experiments": [
            {
                "id": row,
                "label": row,
                "evaluation_roots": {
                    str(seed): roots[f"{row}:{seed}"] for seed in SEEDS
                },
            }
            for row in ROWS
        ],
    }


def _predeclared_contract(
    plan: Mapping[str, Any],
    plan_sha: str,
    aggregation_input_spec: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "queue_id": plan["queue_id"],
        "plan_sha256": plan_sha,
        "profile": PROFILE,
        "provenance_scope": plan["provenance_scope"],
        "ordered_run_ids": [item["run_id"] for item in plan["items"]],
        "run_to_output_root": {
            item["run_id"]: item["evaluation_root"] for item in plan["items"]
        },
        "training_queues": plan["training_queues"],
        "evaluation_runner": plan["evaluation_runner"],
        "evaluation_sources": plan["evaluation_sources"],
        "controller_sources": plan["controller_sources"],
        "profile_support_sources": plan["profile_support_sources"],
        "aggregation_input_spec": dict(aggregation_input_spec),
        "gpu_lease": {
            "gpu_key": plan["gpu_key"],
            "lease_root": plan["lease_root"],
            "lease_path": plan["lease_path"],
            "gpu_environment": plan["gpu_environment"],
            "policy": "retained_across_items_until_verified_queue_completion",
        },
    }


def _gpu_lease_plan(
    *, lease_root: Path, gpu_key: str | None
) -> dict[str, Any]:
    try:
        selected = training_queue._gpu_key_from_environment(
            training_queue._snapshot_environment(), gpu_key
        )
    except training_queue.QueueContractError as exc:
        raise MatrixQueueError(f"invalid matrix GPU selection: {exc}") from exc
    lease_root = lease_root.expanduser().resolve(strict=False)
    return {
        "gpu_key": selected,
        "lease_root": str(lease_root),
        "lease_path": str(training_queue._lease_path(lease_root, selected)),
        "gpu_environment": {
            "CUDA_VISIBLE_DEVICES": selected,
            "PIVOT_CUDA_VISIBLE_DEVICES": selected,
        },
    }


def _runtime_plan(
    *,
    evaluation_python: Path,
    data_root: Path,
    device: str,
    cuda_visible_devices: str,
) -> dict[str, Any]:
    evaluation_python = evaluation_python.expanduser().resolve(strict=True)
    if not evaluation_python.is_file() or not os.access(evaluation_python, os.X_OK):
        raise MatrixQueueError("evaluation Python is not executable")
    data_root = data_root.expanduser().resolve(strict=True)
    if not data_root.is_dir():
        raise MatrixQueueError("evaluation data root is not a directory")
    if device != "cuda:0":
        raise MatrixQueueError(
            "formal matrix evaluation device must be exactly 'cuda:0'"
        )
    if not isinstance(cuda_visible_devices, str) or not cuda_visible_devices:
        raise MatrixQueueError("matrix CUDA visibility is empty")
    return {
        "python": str(evaluation_python),
        "data_root": str(data_root),
        "device": device,
        "cuda_visible_devices": cuda_visible_devices,
        "batch_size": 16,
        "num_workers": 4,
        "amp": True,
        "log_every": 50,
        "eval_seed": evaluator.EVAL_SEED,
        "max_ref_batches": 0,
        "max_tn_batches": 0,
    }


def _item_command(
    *,
    runner_python: Path,
    evaluation_runner: Path,
    runtime: Mapping[str, Any],
    training_root: Path,
    training_queue_dir: Path,
    evaluation_root: Path,
) -> list[str]:
    return [
        str(runner_python),
        str(evaluation_runner),
        "run",
        "--training-run-root",
        str(training_root),
        "--training-queue-dir",
        str(training_queue_dir),
        "--profile",
        PROFILE,
        "--python",
        str(runtime["python"]),
        "--data-root",
        str(runtime["data_root"]),
        "--device",
        str(runtime["device"]),
        "--batch-size",
        str(runtime["batch_size"]),
        "--num-workers",
        str(runtime["num_workers"]),
        "--log-every",
        str(runtime["log_every"]),
        "--output-dir",
        str(evaluation_root),
    ]


def _creation_stage_dir(queue_dir: Path) -> Path:
    return queue_dir.with_name(f".{queue_dir.name}.creating")


def _discard_uncommitted_creation_stage(stage_dir: Path) -> bool:
    """Remove only a stage that never published its immutable queue plan."""
    queue_path = stage_dir / "queue.json"
    if queue_path.is_file():
        return False
    if not stage_dir.is_dir():
        raise MatrixQueueError(f"matrix queue creation stage is invalid: {stage_dir}")
    allowed_names = {"aggregation_input_spec.json", "predeclared_contract.json"}
    entries = list(stage_dir.iterdir())
    for entry in entries:
        is_known_temporary = entry.name.startswith(".") and ".tmp-" in entry.name
        if (
            entry.is_symlink()
            or not entry.is_file()
            or (entry.name not in allowed_names and not is_known_temporary)
        ):
            raise MatrixQueueError(
                "uncommitted matrix queue creation stage contains unknown state: "
                f"{entry}"
            )
    for entry in entries:
        entry.unlink()
    stage_dir.rmdir()
    _fsync_directory(stage_dir.parent)
    return True


def _ensure_staged_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        try:
            observed = _read_json(path, label=f"staged {path.name}")
        except MatrixQueueError as exc:
            if not isinstance(exc.__cause__, json.JSONDecodeError):
                raise
            path.unlink()
            _fsync_directory(path.parent)
        else:
            if observed != dict(payload):
                raise MatrixQueueError(f"staged creation artifact drifted: {path}")
            return
    elif path.exists():
        raise MatrixQueueError(f"staged creation artifact is not a file: {path}")
    _write_json_exclusive(path, payload)


def _recover_staged_creation(
    queue_dir: Path,
    aggregation_input_spec_path: Path,
    *,
    expected_provenance_scope: str,
    test_only_capability: object | None,
) -> dict[str, Any]:
    stage_dir = _creation_stage_dir(queue_dir)
    staged_queue_path = stage_dir / "queue.json"
    if not staged_queue_path.is_file():
        raise MatrixQueueError(
            "matrix queue creation staging exists without its durable queue plan: "
            f"{stage_dir}"
        )
    queue = _read_json(staged_queue_path, label="staged matrix evaluation queue")
    plan = queue.get("plan")
    if not isinstance(plan, Mapping):
        raise MatrixQueueError("staged matrix queue has no immutable plan")
    if plan.get("provenance_scope") != expected_provenance_scope:
        raise MatrixQueueError("staged matrix queue provenance scope drifted")
    if expected_provenance_scope == TEST_ONLY_PROVENANCE_SCOPE:
        if test_only_capability is not _TEST_ONLY_CREATE_CAPABILITY:
            raise MatrixQueueError(
                "test-only staged recovery lacks its in-process capability"
            )
        _AUTHORIZED_TEST_QUEUE_IDS.add(str(plan.get("queue_id")))
    _require_local_mutation_root(plan)
    plan_sha = _canonical_sha(plan)
    if (
        queue.get("plan_sha256") != plan_sha
        or Path(str(plan.get("queue_dir", ""))).resolve(strict=False) != queue_dir
        or Path(str(plan.get("output_root", ""))).resolve(strict=False)
        != DEFAULT_OUTPUT_ROOT.resolve(strict=False)
        or plan.get("aggregation_input_spec")
        != _aggregation_input_spec_binding(aggregation_input_spec_path)
    ):
        raise MatrixQueueError("staged matrix queue plan identity drifted")
    spec_payload = _aggregation_input_spec_payload(plan, plan_sha)
    spec_record = _content_file_record(
        aggregation_input_spec_path, spec_payload
    )
    if queue.get("aggregation_input_spec") != spec_record:
        raise MatrixQueueError("staged aggregation input record drifted")
    expected_contract = _predeclared_contract(plan, plan_sha, spec_record)
    if (
        queue.get("predeclared_contract_sha256")
        != _canonical_sha(expected_contract)
    ):
        raise MatrixQueueError("staged predeclared contract digest drifted")
    _ensure_staged_json(
        stage_dir / "aggregation_input_spec.json", spec_payload
    )
    _ensure_staged_json(
        stage_dir / "predeclared_contract.json", expected_contract
    )
    _fsync_directory(stage_dir)
    _verify_queue_sources(queue)
    if aggregation_input_spec_path.exists():
        observed_spec = _read_json(
            aggregation_input_spec_path,
            label="predeclared matrix aggregation input",
        )
        if observed_spec != spec_payload:
            raise MatrixQueueError(
                "existing aggregation input differs from staged creation"
            )
    else:
        _write_json_exclusive(aggregation_input_spec_path, spec_payload)
    _verify_content_file_record(
        spec_record, label="predeclared matrix aggregation input"
    )
    if queue_dir.exists():
        raise FileExistsError(
            f"evaluation queue directory must be fresh: {queue_dir}"
        )
    stage_dir.rename(queue_dir)
    _fsync_directory(queue_dir.parent)
    return load_queue(queue_dir)


def _create_queue_locked(
    queue_dir: Path,
    *,
    training_queue_dirs: Sequence[Path],
    output_root: Path,
    runner_python: Path,
    evaluation_runner: Path,
    evaluation_python: Path,
    data_root: Path,
    device: str = "cuda:0",
    evaluation_source_paths: Sequence[Path] | None = None,
    lease_root: Path | None = None,
    gpu_key: str | None = None,
    aggregation_input_spec_path: Path | None = None,
    test_only_capability: object | None = None,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    output_root = output_root.expanduser().resolve(strict=False)
    if queue_dir != DEFAULT_QUEUE_DIR.resolve(strict=False):
        raise MatrixQueueError(
            f"evaluation queue directory must be canonical: {DEFAULT_QUEUE_DIR}"
        )
    if output_root != DEFAULT_OUTPUT_ROOT.resolve(strict=False):
        raise MatrixQueueError(
            f"evaluation output root must be canonical: {DEFAULT_OUTPUT_ROOT}"
        )
    if lease_root is None:
        lease_root = DEFAULT_LEASE_ROOT
    lease_root = lease_root.expanduser().resolve(strict=False)
    if lease_root != DEFAULT_LEASE_ROOT.resolve(strict=False):
        raise MatrixQueueError(
            f"matrix GPU lease root must be canonical: {DEFAULT_LEASE_ROOT}"
        )
    if aggregation_input_spec_path is None:
        aggregation_input_spec_path = DEFAULT_AGGREGATION_INPUT_SPEC
    aggregation_input_spec_path = aggregation_input_spec_path.expanduser().resolve(
        strict=False
    )
    provenance_scope = _creation_provenance_scope(
        evaluation_source_paths, test_only_capability
    )
    if queue_dir.exists():
        raise FileExistsError(f"evaluation queue directory must be fresh: {queue_dir}")
    stage_dir = _creation_stage_dir(queue_dir)
    if stage_dir.exists() and not _discard_uncommitted_creation_stage(stage_dir):
        return _recover_staged_creation(
            queue_dir,
            aggregation_input_spec_path,
            expected_provenance_scope=provenance_scope,
            test_only_capability=test_only_capability,
        )
    if aggregation_input_spec_path.exists():
        raise FileExistsError(
            "predeclared aggregation input must be fresh: "
            f"{aggregation_input_spec_path}"
        )
    if len(training_queue_dirs) != 2:
        raise MatrixQueueError("create requires exactly two training queues")
    snapshots = [
        _training_queue_snapshot(path) for path in training_queue_dirs
    ]
    records = {record["role"]: record for record, _ in snapshots}
    if set(records) != set(LOCKED_TRAINING_QUEUES):
        raise MatrixQueueError("the two locked training queue roles are not exact")
    if any(queue["status"] not in {"running", "completed"} for _, queue in snapshots):
        raise MatrixQueueError("training queues must be running or completed")
    queue_by_role = {
        record["role"]: queue for record, queue in snapshots
    }
    if queue_by_role["completed_l0_l4_seed17"]["status"] != "completed":
        raise MatrixQueueError("the locked L0-L4 seed-17 queue must be completed")
    _verify_training_queue_record(
        records["completed_l0_l4_seed17"], require_completed=True
    )

    runner_python = runner_python.expanduser().resolve(strict=True)
    if not runner_python.is_file() or not os.access(runner_python, os.X_OK):
        raise MatrixQueueError("queue runner Python is not executable")
    evaluation_runner = evaluation_runner.expanduser().resolve(strict=True)
    if not evaluation_runner.is_file():
        raise MatrixQueueError("evaluation runner is not a file")
    lease_plan = _gpu_lease_plan(lease_root=lease_root, gpu_key=gpu_key)
    runtime = _runtime_plan(
        evaluation_python=evaluation_python,
        data_root=data_root,
        device=device,
        cuda_visible_devices=lease_plan["gpu_key"],
    )
    source_paths = _child_evaluation_source_paths(
        evaluation_runner=evaluation_runner,
        injected_paths=evaluation_source_paths,
    )
    controller_source_paths = _controller_source_paths()
    profile_support_source_paths = _profile_support_source_paths()
    queue_by_run = {
        run_id: record
        for record in records.values()
        for run_id in record["ordered_run_ids"]
    }
    if tuple(queue_by_run) != EXPECTED_RUN_IDS:
        # Dict insertion reflects queue partition, not canonical evaluation order.
        if set(queue_by_run) != set(EXPECTED_RUN_IDS):
            raise MatrixQueueError(
                "training queue union is not the exact 33 Table-C runs"
            )
    items = []
    for run_id in EXPECTED_RUN_IDS:
        training_root = _exact_run_root(run_id)
        evaluation_root = _exact_eval_root(output_root, run_id)
        if evaluation_root.exists():
            raise FileExistsError(
                f"predeclared evaluation output already exists: {evaluation_root}"
            )
        queue_record = queue_by_run[run_id]
        command = _item_command(
            runner_python=runner_python,
            evaluation_runner=evaluation_runner,
            runtime=runtime,
            training_root=training_root,
            training_queue_dir=Path(queue_record["queue_dir"]),
            evaluation_root=evaluation_root,
        )
        items.append(
            {
                "run_id": run_id,
                "row_id": run_id.split(":", 1)[0],
                "train_seed": int(run_id.split(":", 1)[1]),
                "training_root": str(training_root),
                "training_queue_role": queue_record["role"],
                "training_queue_dir": queue_record["queue_dir"],
                "training_queue_id": queue_record["queue_id"],
                "training_queue_plan_sha256": queue_record["plan_sha256"],
                "evaluation_root": str(evaluation_root),
                "evaluation_id": run_id.replace(":", "_seed"),
                "profile": PROFILE,
                "command": command,
                "command_shell": shlex.join(command),
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "queue_id": str(uuid.uuid4()),
        "provenance_scope": provenance_scope,
        "created_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "repository_root": str(REPO_ROOT),
        "profile": PROFILE,
        "runner_python": str(runner_python),
        "evaluation_runner": _file_record(evaluation_runner),
        "evaluation_sources": [_file_record(path) for path in source_paths],
        "controller_sources": [
            _file_record(path) for path in controller_source_paths
        ],
        "profile_support_sources": [
            _file_record(path) for path in profile_support_source_paths
        ],
        **lease_plan,
        "runtime": runtime,
        "output_root": str(output_root),
        "aggregation_input_spec": _aggregation_input_spec_binding(
            aggregation_input_spec_path
        ),
        "training_output_root": str(DEFAULT_TRAINING_OUTPUT_ROOT.resolve(strict=False)),
        "training_queues": [records[role] for role in LOCKED_TRAINING_QUEUES],
        "items": items,
    }
    plan_sha = _canonical_sha(plan)
    if provenance_scope == TEST_ONLY_PROVENANCE_SCOPE:
        _AUTHORIZED_TEST_QUEUE_IDS.add(plan["queue_id"])
    aggregation_input = _aggregation_input_spec_payload(plan, plan_sha)
    aggregation_input_record = _content_file_record(
        aggregation_input_spec_path, aggregation_input
    )
    contract = _predeclared_contract(
        plan, plan_sha, aggregation_input_record
    )
    now = _utc_now()
    queue = {
        "schema": QUEUE_SCHEMA,
        "status": "waiting_training",
        "created_at_utc": now,
        "updated_at_utc": now,
        "revision": 0,
        "plan": plan,
        "plan_sha256": plan_sha,
        "predeclared_contract_sha256": _canonical_sha(contract),
        "aggregation_input_spec": aggregation_input_record,
        "training_attestation": None,
        "final_verification": None,
        "items": [
            {
                "index": index,
                "run_id": item["run_id"],
                "evaluation_root": item["evaluation_root"],
                "status": "pending",
            }
            for index, item in enumerate(items)
        ],
        "events": [
            {
                "at_utc": now,
                "event": "queue_predeclared",
                "waiting_training": True,
                "aggregation_input_spec": aggregation_input_record,
                "gpu_key": plan["gpu_key"],
                "lease_path": plan["lease_path"],
            }
        ],
    }
    stage_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(stage_dir.parent)
    _write_json_atomic(stage_dir / "queue.json", queue)
    _ensure_staged_json(
        stage_dir / "aggregation_input_spec.json", aggregation_input
    )
    _ensure_staged_json(stage_dir / "predeclared_contract.json", contract)
    return _recover_staged_creation(
        queue_dir,
        aggregation_input_spec_path,
        expected_provenance_scope=provenance_scope,
        test_only_capability=test_only_capability,
    )


def create_queue(
    queue_dir: Path,
    *,
    training_queue_dirs: Sequence[Path],
    output_root: Path,
    runner_python: Path,
    evaluation_runner: Path,
    evaluation_python: Path,
    data_root: Path,
    device: str = "cuda:0",
    evaluation_source_paths: Sequence[Path] | None = None,
    lease_root: Path | None = None,
    gpu_key: str | None = None,
    aggregation_input_spec_path: Path | None = None,
    test_only_capability: object | None = None,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=False)
    if queue_dir != DEFAULT_QUEUE_DIR.resolve(strict=False):
        raise MatrixQueueError(
            f"evaluation queue directory must be canonical: {DEFAULT_QUEUE_DIR}"
        )
    creation_lock = queue_dir.with_name(f".{queue_dir.name}.create.lock")
    with _exclusive_lock(creation_lock):
        return _create_queue_locked(
            queue_dir,
            training_queue_dirs=training_queue_dirs,
            output_root=output_root,
            runner_python=runner_python,
            evaluation_runner=evaluation_runner,
            evaluation_python=evaluation_python,
            data_root=data_root,
            device=device,
            evaluation_source_paths=evaluation_source_paths,
            lease_root=lease_root,
            gpu_key=gpu_key,
            aggregation_input_spec_path=aggregation_input_spec_path,
            test_only_capability=test_only_capability,
        )


def _source_record_paths(
    plan: Mapping[str, Any], key: str, *, label: str
) -> tuple[Path, ...]:
    records = plan.get(key)
    if not isinstance(records, list) or not records:
        raise MatrixQueueError(f"{label} source closure is empty")
    paths: list[Path] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MatrixQueueError(f"{label} source {index} record is invalid")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise MatrixQueueError(f"{label} source {index} path is invalid")
        paths.append(Path(raw_path).expanduser().resolve(strict=False))
    if len(paths) != len(set(paths)) or paths != sorted(paths, key=str):
        raise MatrixQueueError(
            f"{label} source closure is duplicated or not deterministic"
        )
    return tuple(paths)


def _plan_repository_root(plan: Mapping[str, Any]) -> Path:
    raw_root = plan.get("repository_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise MatrixQueueError("immutable execution repository root is invalid")
    candidate = Path(raw_root).expanduser()
    if not candidate.is_absolute():
        raise MatrixQueueError("immutable execution repository root is not absolute")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise MatrixQueueError(
            f"immutable execution repository root is unavailable: {candidate}"
        ) from exc
    if not root.is_dir() or candidate != root:
        raise MatrixQueueError(
            "immutable execution repository root is not one canonical directory"
        )
    outputs_entry = root / "outputs"
    if not outputs_entry.exists() or not outputs_entry.resolve(strict=True).is_dir():
        raise MatrixQueueError("execution repository has no usable outputs entry")
    if root != REPO_ROOT:
        if not outputs_entry.is_symlink():
            raise MatrixQueueError(
                "relocated execution repository outputs entry is not a symlink"
            )
        if outputs_entry.resolve(strict=True) != (
            REPO_ROOT / "outputs"
        ).resolve(strict=True):
            raise MatrixQueueError(
                "relocated execution repository targets a different outputs root"
            )
    return root


def _require_local_mutation_root(plan: Mapping[str, Any]) -> Path:
    root = _plan_repository_root(plan)
    if root != REPO_ROOT:
        raise MatrixQueueError(
            "queue mutation requires execution from its immutable repository root"
        )
    return root


def _source_contract_cache_key(
    plan: Mapping[str, Any], *, provenance_scope: str, repository_root: Path
) -> str:
    return _canonical_sha(
        {
            "provenance_scope": provenance_scope,
            "repository_root": str(repository_root),
            "evaluation_runner": plan.get("evaluation_runner"),
            "evaluation_sources": plan.get("evaluation_sources"),
            "controller_sources": plan.get("controller_sources"),
            "profile_support_sources": plan.get("profile_support_sources"),
        }
    )


def _validate_source_contract_structure(plan: Mapping[str, Any]) -> None:
    provenance_scope = _require_plan_provenance_scope(plan)
    repository_root = _plan_repository_root(plan)
    cache_key = _source_contract_cache_key(
        plan,
        provenance_scope=provenance_scope,
        repository_root=repository_root,
    )
    if cache_key in _VALIDATED_SOURCE_CONTRACT_CACHE:
        return
    evaluation_paths = _source_record_paths(
        plan, "evaluation_sources", label="evaluation"
    )
    controller_paths = _source_record_paths(
        plan, "controller_sources", label="controller"
    )
    profile_support_paths = _source_record_paths(
        plan, "profile_support_sources", label="profile support"
    )
    expected_controller_paths = tuple(
        _controller_source_paths(repository_root=repository_root)
    )
    if controller_paths != expected_controller_paths:
        raise MatrixQueueError("immutable controller source closure drifted")
    expected_profile_support = tuple(
        _profile_support_source_paths(repository_root=repository_root)
    )
    if profile_support_paths != expected_profile_support:
        raise MatrixQueueError("immutable profile-support source closure drifted")
    if set(profile_support_paths) & (set(evaluation_paths) | set(controller_paths)):
        raise MatrixQueueError(
            "profile-support files leaked into a formal execution source closure"
        )
    runner = plan.get("evaluation_runner")
    if not isinstance(runner, Mapping):
        raise MatrixQueueError("immutable evaluation runner record is missing")
    runner_path = Path(str(runner.get("path", ""))).resolve(strict=False)
    if runner_path not in evaluation_paths:
        raise MatrixQueueError(
            "evaluation runner is absent from the child source closure"
        )
    if provenance_scope == FORMAL_PROVENANCE_SCOPE:
        expected_runner = (
            repository_root / "tools/run_stageb_paper_evaluations.py"
        ).resolve(strict=True)
        if runner_path != expected_runner:
            raise MatrixQueueError(
                "formal evaluation runner is not canonical in its execution root"
            )
        expected_evaluation_paths = tuple(
            _child_evaluation_source_paths(
                evaluation_runner=runner_path,
                injected_paths=None,
                repository_root=repository_root,
            )
        )
        if evaluation_paths != expected_evaluation_paths:
            raise MatrixQueueError(
                "formal child evaluation source closure is not the exact 75-file "
                "canonical profile"
            )
    if (
        repository_root / "tools/run_stageb_matrix_validation_queue.py"
    ).resolve(strict=True) in evaluation_paths:
        raise MatrixQueueError(
            "queue controller leaked into the child evaluation source closure"
        )
    late_bound = sorted(
        path.relative_to(repository_root).as_posix()
        for path in evaluation_paths
        if path.is_relative_to(repository_root)
        and path.relative_to(repository_root).as_posix()
        in LATE_BOUND_SOURCE_RELATIVE_PATHS
    )
    if late_bound:
        raise MatrixQueueError(
            "child evaluation source profile contains late-bound artifacts: "
            + ", ".join(late_bound)
        )
    _VALIDATED_SOURCE_CONTRACT_CACHE.add(cache_key)


def _validate_lifecycle_contract(
    queue: Mapping[str, Any], plan: Mapping[str, Any], plan_sha: str, queue_dir: Path
) -> Mapping[str, Any]:
    gpu_key = plan.get("gpu_key")
    if (
        not isinstance(gpu_key, str)
        or not gpu_key
        or "," in gpu_key
    ):
        raise MatrixQueueError("immutable matrix GPU key is invalid")
    lease_root = Path(str(plan.get("lease_root", ""))).expanduser().resolve(
        strict=False
    )
    expected_lease = training_queue._lease_path(lease_root, gpu_key)
    if Path(str(plan.get("lease_path", ""))).resolve(
        strict=False
    ) != expected_lease.resolve(strict=False):
        raise MatrixQueueError("immutable matrix GPU lease path drifted")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": gpu_key,
        "PIVOT_CUDA_VISIBLE_DEVICES": gpu_key,
    }
    if plan.get("gpu_environment") != expected_environment:
        raise MatrixQueueError("immutable matrix GPU environment drifted")
    runtime = plan.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device") != "cuda:0"
        or runtime.get("cuda_visible_devices") != gpu_key
    ):
        raise MatrixQueueError("immutable matrix runtime/lease GPU binding drifted")

    binding = plan.get("aggregation_input_spec")
    if not isinstance(binding, Mapping) or set(binding) != {"schema", "path"}:
        raise MatrixQueueError("immutable aggregation input binding is invalid")
    spec_path = Path(str(binding.get("path", ""))).expanduser().resolve(
        strict=False
    )
    if binding != _aggregation_input_spec_binding(spec_path):
        raise MatrixQueueError("immutable aggregation input binding drifted")
    expected_payload = _aggregation_input_spec_payload(plan, plan_sha)
    expected_record = _content_file_record(spec_path, expected_payload)
    if queue.get("aggregation_input_spec") != expected_record:
        raise MatrixQueueError("predeclared aggregation input record drifted")
    _verify_content_file_record(
        expected_record, label="predeclared matrix aggregation input"
    )
    if _read_json(
        spec_path, label="predeclared matrix aggregation input"
    ) != expected_payload:
        raise MatrixQueueError("predeclared aggregation input semantics drifted")
    snapshot_path = (queue_dir / "aggregation_input_spec.json").resolve(
        strict=True
    )
    if _read_json(
        snapshot_path, label="matrix aggregation input snapshot"
    ) != expected_payload:
        raise MatrixQueueError("aggregation input snapshot drifted")
    return expected_record


def _validate_final_verification_record(queue: Mapping[str, Any]) -> None:
    record = queue.get("final_verification")
    expected_keys = {
        "schema",
        "verified_at_utc",
        "queue_id",
        "plan_sha256",
        "ordered_run_ids",
        "completion_evidence_sha256",
        "training_attestation_semantic_sha256",
    }
    completion_evidence = [
        item.get("completion_evidence") for item in queue["items"]
    ]
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_keys
        or record.get("schema") != FINAL_VERIFICATION_SCHEMA
        or not _is_utc_timestamp(record.get("verified_at_utc"))
        or record.get("queue_id") != queue["plan"]["queue_id"]
        or record.get("plan_sha256") != queue["plan_sha256"]
        or record.get("ordered_run_ids") != list(EXPECTED_RUN_IDS)
        or record.get("completion_evidence_sha256")
        != _canonical_sha(completion_evidence)
        or not _is_sha256(
            record.get("training_attestation_semantic_sha256")
        )
    ):
        raise MatrixQueueError("final verification record drifted")


def _validate_queue(queue: Mapping[str, Any], queue_dir: Path) -> None:
    if queue.get("schema") != QUEUE_SCHEMA:
        raise MatrixQueueError("unsupported matrix evaluation queue schema")
    plan = queue.get("plan")
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise MatrixQueueError("matrix evaluation queue has no immutable plan")
    plan_sha = _canonical_sha(plan)
    if queue.get("plan_sha256") != plan_sha:
        raise MatrixQueueError("immutable matrix evaluation plan SHA mismatch")
    if Path(str(plan.get("queue_dir", ""))).resolve(strict=False) != queue_dir:
        raise MatrixQueueError("queue opened through a path different from its plan")
    if plan.get("profile") != PROFILE:
        raise MatrixQueueError("immutable plan profile is not matrix_validation")
    _validate_source_contract_structure(plan)
    if Path(str(plan.get("output_root", ""))).resolve(
        strict=False
    ) != DEFAULT_OUTPUT_ROOT.resolve(strict=False):
        raise MatrixQueueError("immutable plan output root is not canonical")
    aggregation_input_record = _validate_lifecycle_contract(
        queue, plan, plan_sha, queue_dir
    )
    plan_items = plan.get("items")
    items = queue.get("items")
    if not isinstance(plan_items, list) or not isinstance(items, list):
        raise MatrixQueueError("matrix evaluation queue items are missing")
    if len(plan_items) != 33 or len(items) != 33:
        raise MatrixQueueError("matrix evaluation queue must contain exactly 33 items")
    if [item.get("run_id") for item in plan_items] != list(EXPECTED_RUN_IDS):
        raise MatrixQueueError("immutable evaluation run order drifted")
    expected_roots = {
        run_id: str(_exact_eval_root(Path(plan["output_root"]), run_id))
        for run_id in EXPECTED_RUN_IDS
    }
    if any(
        not isinstance(item, Mapping)
        or item.get("profile") != PROFILE
        or item.get("evaluation_root") != expected_roots[item.get("run_id")]
        for item in plan_items
    ):
        raise MatrixQueueError("immutable evaluation item identity/root drifted")
    contract_path = (queue_dir / "predeclared_contract.json").resolve(strict=True)
    contract = _read_json(contract_path, label="predeclared evaluation contract")
    expected_contract = _predeclared_contract(
        plan, plan_sha, aggregation_input_record
    )
    if contract != expected_contract or queue.get(
        "predeclared_contract_sha256"
    ) != _canonical_sha(expected_contract):
        raise MatrixQueueError("predeclared evaluation contract drifted")
    status = queue.get("status")
    if status not in {
        "waiting_training",
        "planned",
        "running",
        "verifying",
        "completed",
        "failed",
    }:
        raise MatrixQueueError(f"invalid matrix evaluation queue status: {status!r}")
    active = 0
    failed = 0
    completed_prefix = True
    for index, (planned, item) in enumerate(zip(plan_items, items)):
        if not isinstance(item, Mapping):
            raise MatrixQueueError(f"mutable item {index} is invalid")
        for key in ("run_id", "evaluation_root"):
            if item.get(key) != planned.get(key):
                raise MatrixQueueError(f"mutable item {index} changed {key}")
        if item.get("index") != index or item.get("status") not in ITEM_STATUSES:
            raise MatrixQueueError(f"mutable item {index} status/index is invalid")
        item_status = item["status"]
        if item_status == "completed":
            if not completed_prefix:
                raise MatrixQueueError("completed evaluation items must form a prefix")
        else:
            completed_prefix = False
        if item_status in {"reserved", "launching", "launched", "failed"}:
            active += 1
        if item_status == "failed":
            failed += 1
        if item_status == "pending" and any(
            later.get("status") != "pending" for later in items[index + 1 :]
        ):
            raise MatrixQueueError("later evaluation advanced past pending predecessor")
        child_pid = item.get("child_pid")
        child_identity = item.get("child_process_identity")
        if child_pid is not None:
            _validated_process_identity(
                child_pid,
                child_identity,
                label=f"mutable item {index} child process",
            )
        elif child_identity is not None:
            raise MatrixQueueError(
                f"mutable item {index} has child identity without a PID"
            )
    if active > 1 or failed > 1:
        raise MatrixQueueError("queue contains multiple active/failed evaluations")
    if status in {"waiting_training", "planned"} and any(
        item["status"] != "pending" for item in items
    ):
        raise MatrixQueueError(f"{status} queue contains a started evaluation")
    if status in {"verifying", "completed"} and any(
        item["status"] != "completed" for item in items
    ):
        raise MatrixQueueError(f"{status} queue contains unfinished evaluations")
    if status == "failed" and failed != 1:
        raise MatrixQueueError("failed queue lacks exactly one failed item")
    if status == "running" and failed:
        raise MatrixQueueError("running queue contains a failed item")
    if status == "completed":
        _validate_final_verification_record(queue)
    elif queue.get("final_verification") is not None:
        raise MatrixQueueError(
            "non-completed queue contains a final verification record"
        )
    attestation = queue.get("training_attestation")
    if attestation is not None:
        if not isinstance(attestation, Mapping):
            raise MatrixQueueError("training attestation record is invalid")
        expected_path = (queue_dir / "training_attestation.json").resolve(
            strict=False
        )
        if (
            Path(str(attestation.get("path", ""))).resolve(strict=False)
            != expected_path
        ):
            raise MatrixQueueError("training attestation path is not canonical")
    failed_at_training_gate = (
        status == "failed"
        and attestation is None
        and items[0].get("failure_stage") == "training_gate"
        and items[0].get("status") == "failed"
        and all(item.get("status") == "pending" for item in items[1:])
        and isinstance(queue.get("failure"), Mapping)
        and queue["failure"].get("stage") == "training_gate"
    )
    if (
        status != "waiting_training"
        and attestation is None
        and not failed_at_training_gate
    ):
        raise MatrixQueueError("ready queue has no sealed training attestation")


def _load_queue_structural(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    if not queue_dir.is_dir():
        raise NotADirectoryError(
            f"matrix evaluation queue is not a directory: {queue_dir}"
        )
    queue = _read_json(queue_dir / "queue.json", label="matrix evaluation queue")
    _validate_queue(queue, queue_dir)
    return queue


def load_queue(queue_dir: Path) -> dict[str, Any]:
    queue = _load_queue_structural(queue_dir)
    if queue["status"] == "completed":
        attestation, verified = _verify_full_queue_provenance(queue)
        _verify_final_receipt_matches(queue, attestation, verified)
    else:
        _verify_queue_sources(queue)
    return queue


def _save_queue(queue: MutableMapping[str, Any]) -> None:
    _require_local_mutation_root(queue["plan"])
    queue_dir = Path(str(queue["plan"]["queue_dir"])).resolve(strict=True)
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_at_utc"] = _utc_now()
    _validate_queue(queue, queue_dir)
    _write_json_atomic(queue_dir / "queue.json", queue)


def _event(queue: MutableMapping[str, Any], event: str, **fields: Any) -> None:
    events = queue.setdefault("events", [])
    if not isinstance(events, list):
        raise MatrixQueueError("queue events are invalid")
    events.append({"at_utc": _utc_now(), "event": event, **fields})


def _verify_training_queue_record(
    record: Mapping[str, Any], *, require_completed: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    queue_dir = Path(str(record["queue_dir"])).resolve(strict=True)
    try:
        queue = training_queue.load_queue(queue_dir)
    except (OSError, ValueError, training_queue.QueueContractError) as exc:
        raise MatrixQueueError(f"training queue revalidation failed: {exc}") from exc
    role = _locked_queue_role(queue)
    if (
        role != record["role"]
        or queue["plan"]["queue_id"] != record["queue_id"]
        or queue["plan_sha256"] != record["plan_sha256"]
    ):
        raise MatrixQueueError(f"training queue {record['role']} identity drifted")
    observed_recovery = _verify_pretraining_recovery(queue_dir, queue, role)
    if record.get("pretraining_recovery") != observed_recovery:
        raise MatrixQueueError(
            f"training queue {record['role']} recovery binding drifted"
        )
    if queue["status"] != "completed":
        if require_completed:
            raise MatrixQueueError(
                f"training queue {record['role']} is not completed"
            )
        if queue["status"] != "running":
            raise MatrixQueueError(
                f"training queue {record['role']} is terminal/unexpected: "
                f"{queue['status']}"
            )
        return queue, None
    try:
        verification = training_queue.verify_queue(queue_dir)
    except (OSError, ValueError, training_queue.QueueContractError) as exc:
        raise MatrixQueueError(f"training queue final verify failed: {exc}") from exc
    if (
        verification.get("status") != "passed"
        or verification.get("errors")
        or verification.get("queue_id") != record["queue_id"]
        or verification.get("plan_sha256") != record["plan_sha256"]
    ):
        raise MatrixQueueError(
            f"training queue {record['role']} did not pass final verify"
        )
    return queue, verification


def _resolve_formal_source(
    training_root: Path,
    training_queue_dir: Path,
    cache: evaluator.HashCache,
) -> evaluator.EvaluationSource:
    try:
        source = evaluator._resolve_pivot_source(
            training_root,
            cache,
            training_phase="final",
            training_queue_dir=training_queue_dir,
        )
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise MatrixQueueError(
            f"formal training source resolution failed: {exc}"
        ) from exc
    return source


def _attested_source(
    planned: Mapping[str, Any], source: evaluator.EvaluationSource
) -> dict[str, Any]:
    run_id = str(planned["run_id"])
    if (
        source.kind != "pivot_token_ablation_training_run"
        or source.training_run_id != run_id
        or source.training_seed != planned["train_seed"]
        or source.training_run_root != Path(planned["training_root"])
        or source.training_queue_id != planned["training_queue_id"]
        or source.training_queue_plan_sha256
        != planned["training_queue_plan_sha256"]
        or source.training_phase != "final"
        or source.diagnostic_only
    ):
        raise MatrixQueueError(f"formal training source identity drifted: {run_id}")
    files: dict[str, Any] = {}
    for key, path in (
        ("config", source.config),
        ("checkpoint", source.checkpoint),
        ("sequence_manifest", source.sequence_manifest),
        ("final_phase_manifest", source.final_phase_manifest),
        ("training_postflight", source.training_postflight),
        ("selected_phase_manifest", source.selected_phase_manifest),
        ("selected_training_postflight", source.selected_training_postflight),
        ("training_queue_manifest", source.training_queue_manifest),
        ("training_queue_detached_launch", source.training_queue_detached_launch),
        ("training_queue_detached_status", source.training_queue_detached_status),
    ):
        if path is None:
            raise MatrixQueueError(f"formal source {run_id} lacks {key}")
        files[key] = _file_record(path)
    files["training_data"] = [_file_record(path) for path in source.training_data]
    if not files["training_data"]:
        raise MatrixQueueError(f"formal source {run_id} has no training data")
    if files["checkpoint"]["sha256"] != source.checkpoint_sha256:
        raise MatrixQueueError(f"formal source {run_id} checkpoint SHA drifted")
    resolved_source = asdict(source)
    for key, value in tuple(resolved_source.items()):
        if isinstance(value, Path):
            resolved_source[key] = str(value)
        elif isinstance(value, tuple):
            resolved_source[key] = [
                str(item) if isinstance(item, Path) else item for item in value
            ]
    return {
        "plan_binding": {
            key: planned[key]
            for key in (
                "run_id",
                "row_id",
                "train_seed",
                "training_root",
                "training_queue_role",
                "training_queue_dir",
                "training_queue_id",
                "training_queue_plan_sha256",
            )
        },
        "resolved_source": resolved_source,
        "files": files,
    }


def _training_attestation_semantic_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "semantic_sha256"}
    }


def _build_training_attestation(queue: Mapping[str, Any]) -> dict[str, Any] | None:
    for record in queue["plan"]["training_queues"]:
        _, verification = _verify_training_queue_record(
            record, require_completed=False
        )
        if verification is None:
            return None
    cache = evaluator.HashCache()
    sources = []
    for planned in queue["plan"]["items"]:
        source = _resolve_formal_source(
            Path(planned["training_root"]),
            Path(planned["training_queue_dir"]),
            cache,
        )
        sources.append(_attested_source(planned, source))
    payload = {
        "schema": TRAINING_ATTESTATION_SCHEMA,
        "created_at_utc": _utc_now(),
        "evaluation_queue_id": queue["plan"]["queue_id"],
        "evaluation_plan_sha256": queue["plan_sha256"],
        "training_queues": queue["plan"]["training_queues"],
        "ordered_run_ids": list(EXPECTED_RUN_IDS),
        "sources": sources,
    }
    payload["semantic_sha256"] = _canonical_sha(
        _training_attestation_semantic_payload(payload)
    )
    return payload


def _validate_attested_source_structure(
    planned: Mapping[str, Any], source: Any
) -> None:
    run_id = str(planned["run_id"])
    if not isinstance(source, Mapping) or set(source) != {
        "plan_binding",
        "resolved_source",
        "files",
    }:
        raise MatrixQueueError(f"attested source {run_id} schema is not exact")
    expected_binding = {
        key: planned[key]
        for key in (
            "run_id",
            "row_id",
            "train_seed",
            "training_root",
            "training_queue_role",
            "training_queue_dir",
            "training_queue_id",
            "training_queue_plan_sha256",
        )
    }
    if source.get("plan_binding") != expected_binding:
        raise MatrixQueueError(f"attested source {run_id} plan binding drifted")
    resolved = source.get("resolved_source")
    expected_resolved_keys = set(evaluator.EvaluationSource.__dataclass_fields__)
    if not isinstance(resolved, Mapping) or set(resolved) != expected_resolved_keys:
        raise MatrixQueueError(
            f"attested source {run_id} resolved-source projection is not exact"
        )
    if (
        resolved.get("kind") != "pivot_token_ablation_training_run"
        or resolved.get("training_run_id") != run_id
        or resolved.get("training_seed") != planned["train_seed"]
        or resolved.get("training_run_root") != planned["training_root"]
        or resolved.get("training_queue_id") != planned["training_queue_id"]
        or resolved.get("training_queue_plan_sha256")
        != planned["training_queue_plan_sha256"]
        or resolved.get("training_phase") != "final"
        or resolved.get("diagnostic_only") is not False
        or not isinstance(resolved.get("training_data"), list)
        or not resolved["training_data"]
    ):
        raise MatrixQueueError(f"attested source {run_id} resolved identity drifted")
    files = source.get("files")
    expected_file_keys = {
        "config",
        "checkpoint",
        "sequence_manifest",
        "final_phase_manifest",
        "training_postflight",
        "selected_phase_manifest",
        "selected_training_postflight",
        "training_queue_manifest",
        "training_queue_detached_launch",
        "training_queue_detached_status",
        "training_data",
    }
    if not isinstance(files, Mapping) or set(files) != expected_file_keys:
        raise MatrixQueueError(
            f"attested source {run_id} file-role projection is not exact"
        )
    if not isinstance(files["training_data"], list) or not files["training_data"]:
        raise MatrixQueueError(f"attested source {run_id} training data is empty")


def _validate_training_attestation_payload(
    queue: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema",
        "created_at_utc",
        "evaluation_queue_id",
        "evaluation_plan_sha256",
        "training_queues",
        "ordered_run_ids",
        "sources",
        "semantic_sha256",
    }
    created_at = payload.get("created_at_utc")
    if (
        set(payload) != expected_keys
        or payload.get("schema") != TRAINING_ATTESTATION_SCHEMA
        or not _is_utc_timestamp(created_at)
        or payload.get("evaluation_queue_id") != queue["plan"]["queue_id"]
        or payload.get("evaluation_plan_sha256") != queue["plan_sha256"]
        or payload.get("ordered_run_ids") != list(EXPECTED_RUN_IDS)
        or payload.get("training_queues") != queue["plan"]["training_queues"]
        or payload.get("semantic_sha256")
        != _canonical_sha(_training_attestation_semantic_payload(payload))
    ):
        raise MatrixQueueError("training attestation queue/semantic identity drifted")
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != len(EXPECTED_RUN_IDS):
        raise MatrixQueueError("training attestation source cardinality drifted")
    for planned, source in zip(queue["plan"]["items"], sources):
        _validate_attested_source_structure(planned, source)


def _replay_training_attestation(
    queue: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    _validate_training_attestation_payload(queue, payload)
    expected = _build_training_attestation(queue)
    if expected is None:
        raise MatrixQueueError("training attestation replay found incomplete training")
    if (
        _training_attestation_semantic_payload(payload)
        != _training_attestation_semantic_payload(expected)
        or payload.get("semantic_sha256") != expected.get("semantic_sha256")
    ):
        raise MatrixQueueError(
            "training attestation differs from the canonical 33-source replay"
        )


def _training_attestation_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    record = queue.get("training_attestation")
    path = _verify_file_record(record, label="training attestation")
    payload = _read_json(path, label="training attestation")
    _validate_training_attestation_payload(queue, payload)
    return payload


def _verify_attested_source(queue: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    payload = _training_attestation_payload(queue)
    source = next(
        item
        for item in payload["sources"]
        if item["plan_binding"]["run_id"] == run_id
    )
    files = source["files"]
    for key, value in files.items():
        records = value if isinstance(value, list) else [value]
        for index, record in enumerate(records):
            _verify_file_record(record, label=f"{run_id} {key}[{index}]")
    return source


def _advance_training_gate(queue: MutableMapping[str, Any]) -> bool:
    _verify_queue_sources(queue)
    if queue.get("training_attestation") is not None:
        payload = _training_attestation_payload(queue)
        if queue["status"] == "waiting_training":
            _replay_training_attestation(queue, payload)
            queue["status"] = "planned"
            _event(queue, "training_attestation_recovered")
            _save_queue(queue)
        return True
    path = Path(queue["plan"]["queue_dir"]) / "training_attestation.json"
    if path.is_file():
        payload = _read_json(path, label="recovered training attestation")
        _replay_training_attestation(queue, payload)
        queue["training_attestation"] = _file_record(path)
        queue["status"] = "planned"
        _event(queue, "training_attestation_recovered_after_interruption")
        _save_queue(queue)
        return True
    attestation = _build_training_attestation(queue)
    if attestation is None:
        if queue["status"] != "waiting_training":
            raise MatrixQueueError("training became incomplete after queue readiness")
        return False
    _write_json_exclusive(path, attestation)
    queue["training_attestation"] = _file_record(path)
    queue["status"] = "planned"
    _event(queue, "all_training_completed_and_attested")
    _save_queue(queue)
    return True


def _verify_source_closure(
    queue: Mapping[str, Any], key: str, *, label: str
) -> None:
    plan = queue["plan"]
    sources = plan.get(key)
    if not isinstance(sources, list) or not sources:
        raise MatrixQueueError(f"{label} source closure is empty")
    for index, record in enumerate(sources):
        _verify_file_record(record, label=f"{label} source {index}")


def _verify_evaluation_sources(queue: Mapping[str, Any]) -> None:
    _verify_file_record(
        queue["plan"]["evaluation_runner"], label="evaluation runner"
    )
    _verify_source_closure(
        queue, "evaluation_sources", label="evaluation"
    )


def _verify_controller_sources(queue: Mapping[str, Any]) -> None:
    _verify_source_closure(
        queue, "controller_sources", label="controller"
    )


def _verify_profile_support_sources(queue: Mapping[str, Any]) -> None:
    _verify_source_closure(
        queue, "profile_support_sources", label="profile support"
    )


def _verify_queue_sources(queue: Mapping[str, Any]) -> None:
    _verify_evaluation_sources(queue)
    _verify_controller_sources(queue)
    _verify_profile_support_sources(queue)


def _ensure_gpu_lease(
    queue: Mapping[str, Any], item: Mapping[str, Any], *, create: bool
) -> None:
    try:
        training_queue._ensure_lease(queue, item, create=create)
    except training_queue.QueueLeaseOwnershipError as exc:
        if create:
            raise MatrixQueueBusy(str(exc)) from exc
        raise MatrixQueueError(
            f"matrix GPU lease ownership was lost: {exc}"
        ) from exc
    except training_queue.QueueBusyError as exc:
        raise MatrixQueueBusy(str(exc)) from exc
    except training_queue.QueueContractError as exc:
        raise MatrixQueueError(f"matrix GPU lease verification failed: {exc}") from exc


def _clear_owned_gpu_lease(queue: Mapping[str, Any]) -> None:
    try:
        training_queue._clear_owned_lease(queue)
    except training_queue.QueueLeaseOwnershipError as exc:
        raise MatrixQueueError(
            f"matrix GPU lease ownership was lost before cleanup: {exc}"
        ) from exc
    except training_queue.QueueBusyError as exc:
        raise MatrixQueueBusy(str(exc)) from exc
    except training_queue.QueueContractError as exc:
        raise MatrixQueueError(f"matrix GPU lease cleanup failed: {exc}") from exc
    lease_parent = Path(queue["plan"]["lease_path"]).parent
    parent_fd = os.open(lease_parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _owned_gpu_lease_present(queue: Mapping[str, Any]) -> bool:
    lease_path = Path(queue["plan"]["lease_path"])
    if not lease_path.is_file():
        return False
    try:
        lease = _read_json(lease_path, label="matrix GPU lease")
    except MatrixQueueError:
        return False
    return not training_queue._lease_identity_mismatches(queue, lease)


def _planned_item(queue: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    return queue["plan"]["items"][index]


def _item_work_dir(queue: Mapping[str, Any], index: int) -> Path:
    planned = _planned_item(queue, index)
    slug = planned["run_id"].replace(":", "_")
    return (
        Path(queue["plan"]["queue_dir"]) / "jobs" / f"{index:03d}-{slug}"
    ).resolve(strict=False)


def _validated_process_identity(
    pid: Any, identity: Any, *, label: str
) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise MatrixQueueError(f"{label} PID is invalid")
    if not isinstance(identity, Mapping):
        raise MatrixQueueError(f"{label} identity is missing")
    start_time = identity.get("start_time_ticks")
    boot_id = identity.get("boot_id")
    if (
        identity.get("pid") != pid
        or identity.get("available") is not True
        or isinstance(start_time, bool)
        or not isinstance(start_time, int)
        or start_time <= 0
        or not isinstance(boot_id, str)
        or not boot_id.strip()
    ):
        raise MatrixQueueError(
            f"{label} identity lacks exact PID/start-time/boot binding"
        )
    return dict(identity)


def _evaluation_process_running(pid: Any, expected: Any) -> bool | None:
    running = training_queue._process_running(pid, expected)
    if running is not True:
        return running
    try:
        expected_identity = _validated_process_identity(
            pid, expected, label="stored evaluation child"
        )
        observed_identity = _validated_process_identity(
            pid,
            training_queue._read_process_identity(pid),
            label="observed evaluation child",
        )
    except MatrixQueueError:
        return None
    if (
        observed_identity.get("state") == "Z"
        or observed_identity["start_time_ticks"]
        != expected_identity["start_time_ticks"]
        or observed_identity["boot_id"] != expected_identity["boot_id"]
    ):
        return False
    return True


def _matching_processes(command: Sequence[str]) -> list[tuple[int, dict[str, Any]]]:
    expected = list(command)
    matches: list[tuple[int, dict[str, Any]]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            values = [
                value.decode("utf-8", errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except OSError:
            continue
        if values == expected:
            pid = int(entry.name)
            identity = training_queue._read_process_identity(pid)
            initial_running = training_queue._process_running(pid, identity)
            if initial_running is False:
                continue
            if initial_running is None:
                raise MatrixQueueError(
                    "matching evaluation process identity is not observable"
                )
            identity = _validated_process_identity(
                pid, identity, label="matching evaluation process"
            )
            robust_running = _evaluation_process_running(pid, identity)
            if robust_running is None:
                raise MatrixQueueError(
                    "matching evaluation process identity is not stable"
                )
            if robust_running:
                matches.append((pid, identity))
    return matches


def _reserve(queue: MutableMapping[str, Any], index: int) -> None:
    if not _advance_training_gate(queue):
        return
    _verify_queue_sources(queue)
    planned = _planned_item(queue, index)
    _verify_attested_source(queue, planned["run_id"])
    output_root = Path(planned["evaluation_root"])
    work_dir = _item_work_dir(queue, index)
    if output_root.exists():
        raise MatrixQueueError(
            f"fresh evaluation output already exists: {output_root}"
        )
    if work_dir.exists():
        raise MatrixQueueError(f"fresh evaluation work dir already exists: {work_dir}")
    item = queue["items"][index]
    _ensure_gpu_lease(queue, item, create=index == 0)
    item["status"] = "reserved"
    item["work_dir"] = str(work_dir)
    item["reserved_at_utc"] = _utc_now()
    queue["status"] = "running"
    _event(queue, "evaluation_reserved", index=index, run_id=item["run_id"])
    _save_queue(queue)


def _mark_launching(queue: MutableMapping[str, Any], index: int) -> None:
    planned = _planned_item(queue, index)
    _verify_queue_sources(queue)
    _verify_attested_source(queue, planned["run_id"])
    output_root = Path(planned["evaluation_root"])
    if output_root.exists():
        raise MatrixQueueError(
            f"evaluation output appeared before launch: {output_root}"
        )
    work_dir = Path(queue["items"][index]["work_dir"])
    _ensure_gpu_lease(queue, queue["items"][index], create=False)
    if work_dir.exists():
        if not work_dir.is_dir() or any(work_dir.iterdir()):
            raise MatrixQueueError(
                f"reserved evaluation work dir is not recoverably empty: {work_dir}"
            )
    else:
        work_dir.mkdir(parents=True, exist_ok=False)
    item = queue["items"][index]
    item["status"] = "launching"
    item["console_log"] = str(work_dir / "evaluation_console.log")
    item["launching_at_utc"] = _utc_now()
    _event(queue, "evaluation_launching", index=index, run_id=item["run_id"])
    _save_queue(queue)


def _bind_process(
    queue: MutableMapping[str, Any],
    index: int,
    pid: int,
    identity: Mapping[str, Any],
) -> None:
    identity = _validated_process_identity(
        pid, identity, label="evaluation child process"
    )
    item = queue["items"][index]
    item["status"] = "launched"
    item["child_pid"] = int(pid)
    item["child_process_identity"] = dict(identity)
    item["launched_at_utc"] = _utc_now()
    _event(
        queue,
        "evaluation_process_bound",
        index=index,
        run_id=item["run_id"],
        pid=pid,
    )
    _save_queue(queue)


def _terminate_spawned_process(process: subprocess.Popen[Any]) -> None:
    pid = int(process.pid)
    if process.poll() is None:
        try:
            process_group_id = os.getpgid(pid)
            session_id = os.getsid(pid)
        except ProcessLookupError:
            process_group_id = pid
            session_id = pid
        if process_group_id != pid or session_id != pid:
            raise MatrixQueueError(
                "spawned evaluator is not its sealed session/process-group leader"
            )
    group_exists = _process_group_exists(pid)
    if group_exists is None:
        raise MatrixQueueError("spawned evaluator process group is unobservable")
    if group_exists:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + CHILD_TERMINATION_GRACE_SECONDS
    while _process_group_exists(pid) is True and time.monotonic() < deadline:
        process.poll()
        time.sleep(CHILD_TERMINATION_POLL_SECONDS)
    group_exists = _process_group_exists(pid)
    if group_exists is None:
        raise MatrixQueueError(
            "spawned evaluator shutdown became unobservable after SIGTERM"
        )
    if group_exists:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + CHILD_TERMINATION_GRACE_SECONDS
        while _process_group_exists(pid) is True and time.monotonic() < deadline:
            process.poll()
            time.sleep(CHILD_TERMINATION_POLL_SECONDS)
    process.poll()
    if _process_group_exists(pid) is not False or process.poll() is None:
        raise MatrixQueueError(
            "spawned evaluator process group was not proven terminated/reaped"
        )
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=0)


def _process_group_exists(process_group_id: int) -> bool | None:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    return True


def _reap_local_evaluation_process(pid: int) -> None:
    process = _LOCAL_EVALUATION_PROCESSES.get(pid)
    if process is None:
        return
    if process.poll() is None:
        return
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=0)
    _LOCAL_EVALUATION_PROCESSES.pop(pid, None)


def _wait_for_launched_group_exit(
    pid: int,
    expected_identity: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> bool | None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        _reap_local_evaluation_process(pid)
        running = _evaluation_process_running(pid, expected_identity)
        group_exists = _process_group_exists(pid)
        if running is False and group_exists is False:
            return True
        if running is None or group_exists is None:
            return None
        if time.monotonic() >= deadline:
            return False
        time.sleep(CHILD_TERMINATION_POLL_SECONDS)


def _terminate_launched_process_group(
    queue: Mapping[str, Any], index: int
) -> dict[str, Any]:
    item = queue["items"][index]
    pid = item.get("child_pid")
    expected = item.get("child_process_identity")
    if pid is None and expected is None:
        return {
            "status": "no_bound_process",
            "terminated_at_utc": _utc_now(),
        }
    expected_identity = _validated_process_identity(
        pid, expected, label="terminalizing evaluation child"
    )
    running = _evaluation_process_running(pid, expected_identity)
    if running is None:
        raise MatrixQueueError(
            "launched child identity is unobservable; refusing terminal failure"
        )
    if running is False:
        _reap_local_evaluation_process(pid)
        group_exists = _process_group_exists(pid)
        if group_exists is not False:
            raise MatrixQueueError(
                "launched child leader exited but its exact process group cannot be "
                "proven gone"
            )
        return {
            "status": "already_exited",
            "pid": pid,
            "process_group_id": pid,
            "terminated_at_utc": _utc_now(),
        }

    observed_identity = _validated_process_identity(
        pid,
        training_queue._read_process_identity(pid),
        label="terminalizing observed evaluation child",
    )
    if (
        observed_identity["start_time_ticks"]
        != expected_identity["start_time_ticks"]
        or observed_identity["boot_id"] != expected_identity["boot_id"]
    ):
        raise MatrixQueueError(
            "launched child identity changed before process-group termination"
        )
    try:
        process_group_id = os.getpgid(pid)
        session_id = os.getsid(pid)
    except ProcessLookupError:
        if _wait_for_launched_group_exit(
            pid, expected_identity, timeout_seconds=0.0
        ) is True:
            return {
                "status": "already_exited",
                "pid": pid,
                "process_group_id": pid,
                "terminated_at_utc": _utc_now(),
            }
        raise MatrixQueueError(
            "launched child disappeared before its process group was proven gone"
        )
    if process_group_id != pid or session_id != pid:
        raise MatrixQueueError(
            "launched child is not the sealed session/process-group leader"
        )

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGTERM)
    exited = _wait_for_launched_group_exit(
        pid,
        expected_identity,
        timeout_seconds=CHILD_TERMINATION_GRACE_SECONDS,
    )
    escalated = False
    if exited is not True:
        if exited is None:
            raise MatrixQueueError(
                "launched child shutdown became unobservable after SIGTERM"
            )
        before_kill = training_queue._read_process_identity(pid)
        if before_kill.get("available") is True and (
            before_kill.get("start_time_ticks")
            != expected_identity["start_time_ticks"]
            or before_kill.get("boot_id") != expected_identity["boot_id"]
        ):
            raise MatrixQueueError(
                "refusing SIGKILL because the launched child PID was reused"
            )
        escalated = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
        exited = _wait_for_launched_group_exit(
            pid,
            expected_identity,
            timeout_seconds=CHILD_TERMINATION_GRACE_SECONDS,
        )
    if exited is not True:
        raise MatrixQueueError(
            "launched child process group was not proven gone after termination"
        )
    return {
        "status": "terminated",
        "pid": pid,
        "process_group_id": process_group_id,
        "session_id": session_id,
        "signal": "SIGKILL" if escalated else "SIGTERM",
        "terminated_at_utc": _utc_now(),
    }


def _advance_launching(queue: MutableMapping[str, Any], index: int) -> None:
    execution_root = _require_local_mutation_root(queue["plan"])
    planned = _planned_item(queue, index)
    command = list(planned["command"])
    try:
        matches = _matching_processes(command)
    except MatrixQueueError as exc:
        raise MatrixQueueBusy(
            "launching evaluator recovery is not safely observable"
        ) from exc
    if len(matches) > 1:
        raise MatrixQueueBusy(
            "multiple evaluation processes match one launching queue item"
        )
    if matches:
        try:
            _bind_process(queue, index, *matches[0])
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            raise MatrixQueueBusy(
                "matching evaluator could not be durably bound for recovery"
            ) from exc
        _ensure_gpu_lease(queue, queue["items"][index], create=False)
        _verify_queue_sources(queue)
        _verify_attested_source(queue, planned["run_id"])
        return
    _ensure_gpu_lease(queue, queue["items"][index], create=False)
    _verify_queue_sources(queue)
    _verify_attested_source(queue, planned["run_id"])
    output_root = Path(planned["evaluation_root"])
    launch_path = output_root / "launch_manifest.json"
    if output_root.exists():
        if launch_path.is_file():
            launch = _read_json(launch_path, label="orphan evaluation launch")
            if launch.get("status") == "completed":
                item = queue["items"][index]
                item["status"] = "launched"
                item["child_pid"] = None
                item["child_process_identity"] = None
                item["recovered_completed_launch_at_utc"] = _utc_now()
                _save_queue(queue)
                return
        raise MatrixQueueError(
            "evaluation output exists without one recoverable process/completed launch"
        )
    log_path = Path(queue["items"][index]["console_log"])
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(queue["plan"]["gpu_environment"])
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=execution_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        raise MatrixQueueError(f"cannot launch matrix evaluation: {exc}") from exc
    pid = int(process.pid)
    try:
        identity = _validated_process_identity(
            pid,
            training_queue._read_process_identity(pid),
            label="new evaluation child process",
        )
        _LOCAL_EVALUATION_PROCESSES[pid] = process
        _bind_process(queue, index, pid, identity)
    except BaseException as launch_error:
        _LOCAL_EVALUATION_PROCESSES.pop(pid, None)
        try:
            _terminate_spawned_process(process)
        except BaseException as termination_error:
            raise MatrixQueueBusy(
                "spawn-to-bind failure left an unproven evaluator process group: "
                f"{type(launch_error).__name__}: {launch_error}; "
                f"{type(termination_error).__name__}: {termination_error}"
            ) from termination_error
        raise


def _verify_declared_artifact(
    record: Any, *, expected_path: Path, label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise MatrixQueueError(f"{label} artifact record is missing")
    cache = evaluator.HashCache()
    try:
        path = evaluator._verify_declared_file(record, label=label, cache=cache)
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise MatrixQueueError(f"{label} artifact verification failed: {exc}") from exc
    if path != expected_path.resolve(strict=True):
        raise MatrixQueueError(f"{label} artifact path is not canonical")
    return path


def _verify_launch_evaluation_source_set(
    queue: Mapping[str, Any], launch: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    inputs = launch.get("inputs")
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("algorithm") != "sha256"
        or not isinstance(records, list)
    ):
        raise MatrixQueueError(f"{run_id} launch input records are invalid")
    relevant_roles = {
        "evaluation_code_dependency",
        "source_provenance_dependency",
    }
    observed_by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise MatrixQueueError(f"{run_id} launch contains an invalid input record")
        roles = record.get("roles")
        if not isinstance(roles, list) or not relevant_roles.intersection(roles):
            continue
        try:
            identity = {
                key: record[key]
                for key in ("path", "sha256", "size_bytes", "mtime_ns")
            }
        except KeyError as exc:
            raise MatrixQueueError(
                f"{run_id} launch evaluation source record is incomplete"
            ) from exc
        path = str(Path(str(identity["path"])).resolve(strict=False))
        identity["path"] = path
        if path in observed_by_path:
            raise MatrixQueueError(
                f"{run_id} launch duplicates an evaluation source: {path}"
            )
        observed_by_path[path] = identity
    expected = queue["plan"].get("evaluation_sources")
    if not isinstance(expected, list):
        raise MatrixQueueError("immutable evaluation source closure is missing")
    observed = [observed_by_path[path] for path in sorted(observed_by_path)]
    if observed != expected:
        expected_paths = {str(record.get("path")) for record in expected}
        observed_paths = set(observed_by_path)
        raise MatrixQueueError(
            f"{run_id} launch evaluation source set differs from the immutable plan; "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"extra={sorted(observed_paths - expected_paths)}"
        )
    return {
        "status": "passed",
        "source_count": len(observed),
        "source_set_sha256": _canonical_sha(observed),
        "roles": sorted(relevant_roles),
    }


def _verify_completed_evaluation(
    queue: Mapping[str, Any], index: int
) -> dict[str, Any]:
    planned = _planned_item(queue, index)
    run_id = planned["run_id"]
    attested = _verify_attested_source(queue, run_id)
    output_root = Path(planned["evaluation_root"]).resolve(strict=True)
    launch_path = (output_root / "launch_manifest.json").resolve(strict=True)
    postflight_path = (output_root / "postflight.json").resolve(strict=True)
    rehash_path = (output_root / "input_rehash.json").resolve(strict=True)
    launch = _read_json(launch_path, label=f"{run_id} evaluation launch")
    if (
        launch.get("schema") != evaluator.SCHEMA
        or launch.get("status") != "completed"
        or launch.get("evaluation_id") != planned["evaluation_id"]
        or Path(str(launch.get("output_dir", ""))).resolve(strict=False)
        != output_root
    ):
        raise MatrixQueueError(f"{run_id} launch is not exact/completed")
    protocol = launch.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("profile") != PROFILE:
        raise MatrixQueueError(f"{run_id} launch profile is not matrix_validation")
    launch_source_binding = _verify_launch_evaluation_source_set(
        queue, launch, run_id=run_id
    )
    completed = launch.get("completed_phases")
    if not (
        isinstance(completed, list)
        and len(completed) == 1
        and isinstance(completed[0], Mapping)
        and completed[0].get("phase_id") == "validation_calibration"
        and completed[0].get("status") == "completed"
        and completed[0].get("returncode") == 0
    ):
        raise MatrixQueueError(f"{run_id} evaluation phase did not complete exactly")
    source = launch.get("source")
    checkpoint = attested["files"]["checkpoint"]
    if not isinstance(source, Mapping) or (
        source.get("training_run_id") != run_id
        or source.get("training_seed") != planned["train_seed"]
        or Path(str(source.get("training_run_root", ""))).resolve(strict=False)
        != Path(planned["training_root"])
        or source.get("training_queue_id") != planned["training_queue_id"]
        or source.get("training_queue_plan_sha256")
        != planned["training_queue_plan_sha256"]
        or Path(str(source.get("checkpoint", ""))).resolve(strict=False)
        != Path(checkpoint["path"])
        or source.get("checkpoint_sha256") != checkpoint["sha256"]
    ):
        raise MatrixQueueError(f"{run_id} evaluation source identity drifted")
    _verify_declared_artifact(
        launch.get("input_rehash_artifact"),
        expected_path=rehash_path,
        label=f"{run_id} input rehash",
    )
    input_rehash = _read_json(rehash_path, label=f"{run_id} input rehash")
    if (
        input_rehash.get("schema") != evaluator.INPUT_REHASH_SCHEMA
        or input_rehash.get("status") != "passed"
    ):
        raise MatrixQueueError(f"{run_id} input rehash did not pass")
    try:
        replay = evaluator._rehash_inputs(launch)
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise MatrixQueueError(f"{run_id} input rehash replay failed: {exc}") from exc
    for key in ("schema", "status", "records"):
        if replay.get(key) != input_rehash.get(key):
            raise MatrixQueueError(f"{run_id} input rehash {key} drifted")
    _verify_declared_artifact(
        launch.get("postflight_artifact"),
        expected_path=postflight_path,
        label=f"{run_id} postflight",
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
        raise MatrixQueueError(
            f"{run_id} postflight is not exact/passed matrix evidence"
        )
    try:
        replayed_postflight = evaluator._postflight_screen(launch, input_rehash)
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise MatrixQueueError(
            f"{run_id} matrix postflight replay failed: {exc}"
        ) from exc
    observed_postflight = dict(postflight)
    expected_postflight = dict(replayed_postflight)
    observed_postflight.pop("validated_at_utc", None)
    expected_postflight.pop("validated_at_utc", None)
    if observed_postflight != expected_postflight:
        raise MatrixQueueError(
            f"{run_id} persisted matrix postflight differs from replay"
        )
    postflight_checkpoint = postflight.get("checkpoint")
    if not isinstance(postflight_checkpoint, Mapping) or (
        Path(str(postflight_checkpoint.get("path", ""))).resolve(strict=False)
        != Path(checkpoint["path"])
        or postflight_checkpoint.get("sha256") != checkpoint["sha256"]
    ):
        raise MatrixQueueError(f"{run_id} postflight checkpoint drifted")
    return {
        "run_id": run_id,
        "evaluation_root": str(output_root),
        "launch_manifest": _file_record(launch_path),
        "input_rehash": _file_record(rehash_path),
        "postflight": _file_record(postflight_path),
        "profile": PROFILE,
        "launch_source_binding": launch_source_binding,
        "advance_gate": "completed_launch_passed_postflight_replayed_input_rehash",
    }


def _verify_full_queue_provenance(
    queue: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _verify_queue_sources(queue)
    for record in queue["plan"]["training_queues"]:
        _verify_training_queue_record(record, require_completed=True)
    attestation = _training_attestation_payload(queue)
    _replay_training_attestation(queue, attestation)
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(queue["items"]):
        if item["status"] != "completed":
            raise MatrixQueueError(
                f"full queue verification found {item['run_id']} status={item['status']}"
            )
        evidence = _verify_completed_evaluation(queue, index)
        if item.get("completion_evidence") != evidence:
            raise MatrixQueueError(
                f"{item['run_id']} persisted completion evidence drifted"
            )
        verified.append(evidence)
    return attestation, verified


def _verify_final_receipt_matches(
    queue: Mapping[str, Any],
    attestation: Mapping[str, Any],
    verified: Sequence[Mapping[str, Any]],
) -> None:
    _validate_final_verification_record(queue)
    final_verification = queue["final_verification"]
    if (
        final_verification["completion_evidence_sha256"]
        != _canonical_sha(verified)
        or final_verification["training_attestation_semantic_sha256"]
        != attestation["semantic_sha256"]
    ):
        raise MatrixQueueError(
            "completed queue final verification record differs from live replay"
        )


def _advance_final_verification(queue: MutableMapping[str, Any]) -> None:
    if queue["status"] != "verifying":
        raise MatrixQueueError("final verification requires verifying queue state")
    last_item = queue["items"][-1]
    _ensure_gpu_lease(queue, last_item, create=False)
    attestation, verified = _verify_full_queue_provenance(queue)
    _ensure_gpu_lease(queue, last_item, create=False)
    queue["final_verification"] = {
        "schema": FINAL_VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "ordered_run_ids": list(EXPECTED_RUN_IDS),
        "completion_evidence_sha256": _canonical_sha(verified),
        "training_attestation_semantic_sha256": attestation["semantic_sha256"],
    }
    queue["status"] = "completed"
    queue["completed_at_utc"] = _utc_now()
    _event(queue, "queue_final_verification_passed")
    _save_queue(queue)
    # Re-enter through the public completed-queue verifier after the atomic
    # save.  Lease release is authorized only by the exact persisted bytes,
    # not by the in-memory receipt that preceded publication.
    durable = load_queue(Path(queue["plan"]["queue_dir"]))
    _ensure_gpu_lease(durable, durable["items"][-1], create=False)
    _clear_owned_gpu_lease(durable)


def _advance_launched(queue: MutableMapping[str, Any], index: int) -> None:
    _ensure_gpu_lease(queue, queue["items"][index], create=False)
    planned = _planned_item(queue, index)
    item = queue["items"][index]
    pid = item.get("child_pid")
    running = (
        _evaluation_process_running(pid, item.get("child_process_identity"))
        if isinstance(pid, int)
        else False
    )
    if isinstance(pid, int) and running is False:
        process = _LOCAL_EVALUATION_PROCESSES.pop(pid, None)
        if process is not None:
            _terminate_spawned_process(process)
    launch_path = Path(planned["evaluation_root"]) / "launch_manifest.json"
    launch = (
        _read_json(launch_path, label="evaluation launch")
        if launch_path.is_file()
        else None
    )
    item["last_observation"] = {
        "at_utc": _utc_now(),
        "pid_running": running,
        "launch_status": launch.get("status") if launch else None,
    }
    _save_queue(queue)
    if running is True:
        return
    if running is None:
        return
    if launch is None:
        raise MatrixQueueError("evaluation process exited without launch manifest")
    if launch.get("status") == "failed":
        raise MatrixQueueError(
            f"evaluation runner failed: {launch.get('error', 'no error')}"
        )
    if launch.get("status") != "completed":
        raise MatrixQueueError(
            "evaluation process exited with nonterminal launch manifest"
        )
    evidence = _verify_completed_evaluation(queue, index)
    _ensure_gpu_lease(queue, queue["items"][index], create=False)
    item = queue["items"][index]
    item["status"] = "completed"
    item["completed_at_utc"] = _utc_now()
    item["completion_evidence"] = evidence
    _event(queue, "evaluation_completed", index=index, run_id=item["run_id"])
    if all(candidate["status"] == "completed" for candidate in queue["items"]):
        queue["status"] = "verifying"
        queue["verification_started_at_utc"] = _utc_now()
        _event(queue, "queue_final_verification_pending")
    _save_queue(queue)


def _fail_queue(
    queue: MutableMapping[str, Any],
    index: int,
    error: BaseException | str,
    *,
    failure_stage: str | None = None,
) -> None:
    item = queue["items"][index]
    rendered = (
        str(error)
        if isinstance(error, str)
        else f"{type(error).__name__}: {error}"
    )
    item["status"] = "failed"
    item["failed_at_utc"] = _utc_now()
    item["failure_error"] = rendered
    if failure_stage is not None:
        item["failure_stage"] = failure_stage
    queue["status"] = "failed"
    queue["failure"] = {
        "index": index,
        "run_id": item["run_id"],
        "error": rendered,
        "lease_retained_fail_closed": _owned_gpu_lease_present(queue),
        **({"stage": failure_stage} if failure_stage is not None else {}),
    }
    _event(queue, "queue_failed", index=index, run_id=item["run_id"], error=rendered)
    _save_queue(queue)


def _terminalize_queue_failure(
    queue: MutableMapping[str, Any], index: int, error: BaseException
) -> None:
    item = queue["items"][index]
    if item["status"] == "launched":
        try:
            termination = _terminate_launched_process_group(queue, index)
        except BaseException as termination_error:
            item["child_termination_blocked"] = {
                "at_utc": _utc_now(),
                "original_error": f"{type(error).__name__}: {error}",
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
            raise MatrixQueueError(
                "queue remains launched because its child process group could not "
                f"be proven terminated after {type(error).__name__}: {error}: "
                f"{termination_error}"
            ) from termination_error
        item["child_termination"] = termination
    _fail_queue(queue, index, error)


def advance_once(queue_dir: Path) -> dict[str, Any]:
    queue = _load_queue_structural(queue_dir)
    _require_local_mutation_root(queue["plan"])
    if queue["status"] == "completed":
        lease_path = Path(queue["plan"]["lease_path"])
        if lease_path.exists():
            _ensure_gpu_lease(queue, queue["items"][-1], create=False)
        attestation, verified = _verify_full_queue_provenance(queue)
        _verify_final_receipt_matches(queue, attestation, verified)
        if lease_path.exists():
            _ensure_gpu_lease(queue, queue["items"][-1], create=False)
            _clear_owned_gpu_lease(queue)
        return queue
    if queue["status"] == "failed":
        return queue
    if queue["status"] == "waiting_training":
        try:
            _advance_training_gate(queue)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            # A merely running source queue is represented by a False return,
            # whereas identity/verification drift is terminal.
            current = _load_queue_structural(queue_dir)
            _fail_queue(current, 0, exc, failure_stage="training_gate")
        return _load_queue_structural(queue_dir)
    if queue["status"] == "verifying":
        try:
            _advance_final_verification(queue)
        except (MatrixQueueBusy, KeyboardInterrupt):
            raise
        except BaseException as exc:
            current = _load_queue_structural(queue_dir)
            if current["status"] == "verifying":
                _terminalize_queue_failure(current, len(current["items"]) - 1, exc)
            else:
                raise
        return _load_queue_structural(queue_dir)
    index = next(
        (i for i, item in enumerate(queue["items"]) if item["status"] != "completed"),
        None,
    )
    if index is None:
        raise MatrixQueueError("queue has no unfinished item but is not completed")
    item = queue["items"][index]
    try:
        if item["status"] == "pending":
            _reserve(queue, index)
        elif item["status"] == "reserved":
            _mark_launching(queue, index)
        elif item["status"] == "launching":
            _advance_launching(queue, index)
        elif item["status"] == "launched":
            _advance_launched(queue, index)
        else:
            raise MatrixQueueError(f"cannot advance status {item['status']!r}")
    except (MatrixQueueBusy, KeyboardInterrupt):
        raise
    except BaseException as exc:
        current = _load_queue_structural(queue_dir)
        if current["status"] == "completed":
            return current
        current_index = next(
            (
                i
                for i, candidate in enumerate(current["items"])
                if candidate["status"] != "completed"
            ),
            index,
        )
        if current["status"] != "failed":
            _terminalize_queue_failure(current, current_index, exc)
    return _load_queue_structural(queue_dir)


def run_queue(
    queue_dir: Path, *, poll_seconds: float, once: bool = False
) -> dict[str, Any]:
    if poll_seconds < 0.05:
        raise MatrixQueueError("poll_seconds must be at least 0.05")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    with _exclusive_lock(queue_dir / "supervisor.lock"):
        while True:
            queue = advance_once(queue_dir)
            if once or queue["status"] in {"completed", "failed"}:
                return queue
            time.sleep(poll_seconds)


def queue_status(queue_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_dir)
    current = next(
        (
            item
            for item in queue["items"]
            if item["status"] not in {"completed", "pending"}
        ),
        None,
    )
    observation = None
    if current is not None:
        pid = current.get("child_pid")
        observation = {
            "run_id": current["run_id"],
            "status": current["status"],
            "pid_running": (
                _evaluation_process_running(
                    pid, current.get("child_process_identity")
                )
                if isinstance(pid, int)
                else None
            ),
        }
    lease_path = Path(queue["plan"]["lease_path"])
    lease = (
        _read_json(lease_path, label="matrix GPU lease")
        if lease_path.is_file()
        else {"present": False}
    )
    return {
        "schema": STATUS_SCHEMA,
        "observed_at_utc": _utc_now(),
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "profile": PROFILE,
        "provenance_scope": queue["plan"]["provenance_scope"],
        "status": queue["status"],
        "revision": queue["revision"],
        "gpu_key": queue["plan"]["gpu_key"],
        "lease_path": str(lease_path),
        "lease": lease,
        "aggregation_input_spec": queue["aggregation_input_spec"],
        "counts": {
            status: sum(item["status"] == status for item in queue["items"])
            for status in ITEM_STATUSES
        },
        "current": observation,
        "training_attested": queue.get("training_attestation") is not None,
        "failure": queue.get("failure"),
    }


def verify_queue(queue_dir: Path) -> dict[str, Any]:
    queue = _load_queue_structural(queue_dir)
    errors: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    try:
        _verify_queue_sources(queue)
        for record in queue["plan"]["training_queues"]:
            _verify_training_queue_record(record, require_completed=True)
        payload = _training_attestation_payload(queue)
        _replay_training_attestation(queue, payload)
        lease_path = Path(queue["plan"]["lease_path"])
        if queue["status"] == "completed" and lease_path.exists():
            raise MatrixQueueError(
                "completed matrix queue retained its GPU lease"
            )
    except (MatrixQueueError, OSError, ValueError) as exc:
        errors.append({"scope": "queue_provenance", "error": str(exc)})
    for index, item in enumerate(queue["items"]):
        if item["status"] != "completed":
            errors.append(
                {"run_id": item["run_id"], "error": f"status={item['status']}"}
            )
            continue
        try:
            evidence = _verify_completed_evaluation(queue, index)
            if item.get("completion_evidence") != evidence:
                raise MatrixQueueError(
                    f"{item['run_id']} persisted completion evidence drifted"
                )
            verified.append(evidence)
        except (MatrixQueueError, OSError, ValueError) as exc:
            errors.append({"run_id": item["run_id"], "error": str(exc)})
    if queue["status"] == "completed" and not errors:
        try:
            _verify_final_receipt_matches(queue, payload, verified)
        except MatrixQueueError as exc:
            errors.append(
                {
                    "scope": "queue_provenance",
                    "error": str(exc),
                }
            )
    return {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "status": (
            "passed"
            if queue["status"] == "completed" and not errors
            else "failed"
        ),
        "queue_status": queue["status"],
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "predeclared_contract_sha256": queue["predeclared_contract_sha256"],
        "profile": PROFILE,
        "provenance_scope": queue["plan"]["provenance_scope"],
        "verified_items": verified,
        "errors": errors,
    }


def _supervisor_current_path(queue_dir: Path) -> Path:
    return queue_dir / "supervisors" / "current.json"


def detach_queue(queue_dir: Path, *, poll_seconds: float) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    queue = load_queue(queue_dir)
    execution_root = _require_local_mutation_root(queue["plan"])
    if queue["status"] in {"completed", "failed"}:
        return {
            "schema": SUPERVISOR_SCHEMA,
            "status": queue["status"],
            "queue_id": queue["plan"]["queue_id"],
            "spawned": False,
        }
    with _exclusive_lock(queue_dir / "detach.lock"):
        current_path = _supervisor_current_path(queue_dir)
        if current_path.is_file():
            current = _read_json(current_path, label="detached supervisor")
            pid = current.get("pid")
            if isinstance(pid, int) and training_queue._process_running(
                pid, current.get("process_identity")
            ) is True:
                return {**current, "status": "already_running"}
        job_dir = (
            queue_dir
            / "supervisors"
            / (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
                f"{uuid.uuid4().hex[:8]}"
            )
        )
        job_dir.mkdir(parents=True, exist_ok=False)
        command = [
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            "_supervise",
            str(queue_dir),
            "--job-dir",
            str(job_dir),
            "--poll-seconds",
            str(poll_seconds),
        ]
        log_path = job_dir / "supervisor.log"
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=execution_root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        record = {
            "schema": SUPERVISOR_SCHEMA,
            "status": "launched",
            "created_at_utc": _utc_now(),
            "queue_id": load_queue(queue_dir)["plan"]["queue_id"],
            "job_dir": str(job_dir),
            "pid": int(process.pid),
            "process_identity": training_queue._read_process_identity(int(process.pid)),
            "command": command,
            "command_shell": shlex.join(command),
            "log": str(log_path),
        }
        _LOCAL_SUPERVISOR_PROCESSES[int(process.pid)] = process
        _write_json_atomic(current_path, record)
        _write_json_exclusive(job_dir / "launch.json", record)
        return record


def _supervise(queue_dir: Path, job_dir: Path, *, poll_seconds: float) -> int:
    status_path = job_dir / "status.json"
    status = {
        "schema": SUPERVISOR_SCHEMA,
        "status": "running",
        "started_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "pid": os.getpid(),
    }
    _write_json_atomic(status_path, status)
    try:
        queue = run_queue(queue_dir, poll_seconds=poll_seconds)
        status["status"] = queue["status"]
        status["queue_revision"] = queue["revision"]
        returncode = 0 if queue["status"] != "failed" else 1
    except BaseException as exc:
        status["status"] = "supervisor_failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        returncode = 1
    status["finished_at_utc"] = _utc_now()
    _write_json_atomic(status_path, status)
    return returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    create = modes.add_parser("create", help="fresh-only canonical predeclaration")
    create.add_argument("queue_dir", type=Path, nargs="?", default=DEFAULT_QUEUE_DIR)
    create.add_argument(
        "--training-queue-dir",
        type=Path,
        action="append",
        dest="training_queue_dirs",
    )
    create.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    create.add_argument("--runner-python", type=Path, default=Path(sys.executable))
    create.add_argument(
        "--evaluation-runner", type=Path, default=DEFAULT_EVALUATION_RUNNER
    )
    create.add_argument(
        "--evaluation-python", type=Path, default=evaluator.DEFAULT_PYTHON
    )
    create.add_argument("--data-root", type=Path, default=evaluator.DEFAULT_DATA_ROOT)
    create.add_argument("--device", default="cuda:0")
    create.add_argument("--gpu-key")
    for mode in ("run", "restart", "reconcile"):
        child = modes.add_parser(mode)
        child.add_argument("queue_dir", type=Path)
        child.add_argument("--poll-seconds", type=float, default=30.0)
    detach = modes.add_parser("detach")
    detach.add_argument("queue_dir", type=Path)
    detach.add_argument("--poll-seconds", type=float, default=30.0)
    status = modes.add_parser("status")
    status.add_argument("queue_dir", type=Path)
    verify = modes.add_parser("verify")
    verify.add_argument("queue_dir", type=Path)
    internal = modes.add_parser("_supervise")
    internal.add_argument("queue_dir", type=Path)
    internal.add_argument("--job-dir", type=Path, required=True)
    internal.add_argument("--poll-seconds", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "create":
            queue = create_queue(
                args.queue_dir,
                training_queue_dirs=(
                    args.training_queue_dirs
                    if args.training_queue_dirs is not None
                    else DEFAULT_TRAINING_QUEUE_DIRS
                ),
                output_root=args.output_root,
                runner_python=args.runner_python,
                evaluation_runner=args.evaluation_runner,
                evaluation_python=args.evaluation_python,
                data_root=args.data_root,
                device=args.device,
                gpu_key=args.gpu_key,
            )
            print(json.dumps(queue, indent=2, sort_keys=True))
            return 0
        if args.mode in {"run", "restart", "reconcile"}:
            queue = run_queue(
                args.queue_dir,
                poll_seconds=args.poll_seconds,
                once=args.mode == "reconcile",
            )
            print(json.dumps(queue_status(args.queue_dir), indent=2, sort_keys=True))
            return 1 if queue["status"] == "failed" else 0
        if args.mode == "detach":
            report = detach_queue(
                args.queue_dir, poll_seconds=args.poll_seconds
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.mode == "status":
            print(json.dumps(queue_status(args.queue_dir), indent=2, sort_keys=True))
            return 0
        if args.mode == "verify":
            report = verify_queue(args.queue_dir)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "passed" else 1
        if args.mode == "_supervise":
            return _supervise(
                args.queue_dir,
                args.job_dir,
                poll_seconds=args.poll_seconds,
            )
        parser.error(f"unsupported mode: {args.mode}")
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        MatrixQueueError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
