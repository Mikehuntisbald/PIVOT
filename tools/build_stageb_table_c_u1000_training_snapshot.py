#!/usr/bin/env python3
"""Build or verify the retrospective Table-C U1000 training snapshot.

The snapshot is deliberately outside the historical training dependency
closure.  It preserves the bytes and completed-run evidence that exist after
the two sealed training queues finish; it does not retroactively make the
snapshot builder, or any formerly omitted source, launch-bound.

``build`` first replays the existing live final gates.  It then publishes one
fresh content-addressed directory with Linux ``RENAME_NOREPLACE``.  Ordinary
``verify`` reads the archived 89-source bytes while rehashing the live
completion evidence and ten non-source inputs.  It intentionally does not
require the original 89 source paths.  ``--require-live-source-parity`` adds
those live source rehashes and replays the full final gates.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_SNAPSHOT_SCHEMA = "pivot.stageb.table_c_u1000_training_source_snapshot/v1"
COMPLETION_SUBRECEIPT_SCHEMA = (
    "pivot.stageb.table_c_u1000_training_completion_subreceipt/v1"
)
SOURCE_SNAPSHOT_DIGEST_SCHEMA = (
    "pivot.stageb.table_c_u1000_training_source_snapshot_digest/v1"
)
COMPLETION_SUBRECEIPT_DIGEST_SCHEMA = (
    "pivot.stageb.table_c_u1000_training_completion_subreceipt_digest/v1"
)
VERIFICATION_SCHEMA = (
    "pivot.stageb.table_c_u1000_training_snapshot_verification/v1"
)
ATTESTATION_SCHEMA = "pivot.stageb.table_c_dependency_closure_attestation/v1"
ATTESTATION_DIGEST_SCHEMA = (
    "pivot.stageb.table_c_dependency_closure_attestation_digest/v1"
)
RECOVERY_SCHEMA = "pivot.stageb.serial_queue_pretraining_recovery_receipt/v1"

CANONICAL_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/"
    "table_c_u1000_training_snapshot_v1"
)
DEFAULT_DEPENDENCY_ATTESTATION = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/"
    "table_c_dependency_closure_preflight_20260718.json"
)
DEFAULT_TRAINING_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/token_ablation_frozen_v2"
)
DEFAULT_COMPLETED_QUEUE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/queues/"
    "table_c_screen_l0_l4_seed17_b40_u1000_frozen_v2"
)
DEFAULT_REMAINING_QUEUE = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/queues/"
    "table_c_remaining_28_b40_u1000_frozen_v2"
)
DEFAULT_RECOVERY_RECEIPT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/recovery/table_c_remaining_28/"
    "L2_seed42_attempt000/recovery_receipt.json"
)

ROWS = tuple(f"L{index}" for index in range(11))
SEEDS = (17, 42, 73)
EXPECTED_RUN_IDS = tuple(f"{row}:{seed}" for seed in SEEDS for row in ROWS)
COMPLETED_QUEUE_RUN_IDS = tuple(f"L{index}:17" for index in range(5))
REMAINING_QUEUE_RUN_IDS = tuple(
    [f"L{index}:17" for index in range(5, 11)]
    + [f"L{index}:42" for index in range(11)]
    + [f"L{index}:73" for index in range(11)]
)
LOCKED_QUEUES: Mapping[str, Mapping[str, Any]] = {
    "completed_l0_l4_seed17": {
        "queue_id": "3e5a961a-f2da-45ba-8e44-94740f4baee9",
        "plan_sha256": (
            "63619de10c9e41d2ecc5177242b4b3bbf175d57c3c9cdcd7013b1185a53e6cde"
        ),
        "run_ids": COMPLETED_QUEUE_RUN_IDS,
    },
    "remaining_table_c": {
        "queue_id": "ffcc3e46-ca1d-45d0-9fbd-22e5db14ac9f",
        "plan_sha256": (
            "b4b8ef280fcbd67dbf82fc59d6c90f63c9c3573976b8950c06f1e84dbb31c2cc"
        ),
        "run_ids": REMAINING_QUEUE_RUN_IDS,
    },
}

SOURCE_CLOSURE_COUNT = 85
STATIC_SOURCE_COUNT = 9
AUDITOR_SOURCE_COUNT = 3
SOURCE_UNION_COUNT = 89
SOURCE_UNION_SIZE_BYTES = 1_790_057
NON_SOURCE_INPUT_COUNT = 10
LAUNCH_SOURCE_UNION_COUNT = 36
FULL_LAUNCH_INPUT_UNION_COUNT = 46
PER_RUN_INPUT_REHASH_COUNT = 36
RECOVERY_RUN_ID = "L2:42"
RECOVERY_FAILED_REVISION = 590

POSTFLIGHT_ARTIFACT_FILES: Mapping[str, str] = {
    "checkpoint": "checkpoint_iter.pth",
    "gpu_environment": "gpu_environment.json",
    "gpu_telemetry": "gpu_telemetry.csv",
    "gpu_telemetry_summary": "gpu_telemetry_summary.json",
    "input_rehash": "input_rehash.json",
    "native_info_log": "info.txt",
    "scorer_init_audit": "stage_b_v15_scorer_init_audit.json",
    "train_console_log": "train_console.log",
}
FIRST_CLASS_RUN_FILES: Mapping[str, str] = {
    "sequence_manifest": "sequence_manifest.json",
    "launch_manifest": "launch_manifest.json",
    "postflight": "postflight.json",
    "input_rehash": "input_rehash.json",
    "checkpoint": "checkpoint_iter.pth",
}


class TrainingSnapshotError(RuntimeError):
    """The Table-C training snapshot contract failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TrainingSnapshotError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _self_digest(payload: Mapping[str, Any], key: str) -> str:
    view = dict(payload)
    view.pop(key, None)
    return _canonical_sha256(view)


def _source_snapshot_digest(payload: Mapping[str, Any]) -> str:
    view = dict(payload)
    view.pop("source_snapshot_sha256", None)
    return _canonical_sha256(
        {"schema": SOURCE_SNAPSHOT_DIGEST_SCHEMA, "source_snapshot": view}
    )


def _completion_subreceipt_digest(payload: Mapping[str, Any]) -> str:
    view = dict(payload)
    view.pop("completion_subreceipt_sha256", None)
    return _canonical_sha256(
        {
            "schema": COMPLETION_SUBRECEIPT_DIGEST_SCHEMA,
            "completion_subreceipt": view,
        }
    )


def _strict_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrainingSnapshotError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingSnapshotError(f"{label} is not a JSON object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _regular_path(path: Path, *, label: str) -> Path:
    candidate = path.expanduser().absolute()
    try:
        raw = os.lstat(candidate)
    except OSError as exc:
        raise TrainingSnapshotError(f"cannot stat {label} {candidate}: {exc}") from exc
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode):
        raise TrainingSnapshotError(f"{label} is not a non-symlink regular file: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise TrainingSnapshotError(f"cannot resolve {label} {candidate}: {exc}") from exc
    return resolved


class _Binder:
    """Hash each live path once through a stable descriptor."""

    def __init__(self) -> None:
        self._records: dict[Path, dict[str, Any]] = {}
        self._json: dict[Path, dict[str, Any]] = {}

    def file(self, path: Path, *, label: str) -> dict[str, Any]:
        resolved = _regular_path(path, label=label)
        cached = self._records.get(resolved)
        if cached is not None:
            return dict(cached)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(resolved, flags)
            with os.fdopen(descriptor, "rb") as handle:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise TrainingSnapshotError(f"{label} is not a regular file: {resolved}")
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                after = os.fstat(handle.fileno())
            current = os.stat(resolved, follow_symlinks=False)
        except OSError as exc:
            raise TrainingSnapshotError(f"cannot hash {label} {resolved}: {exc}") from exc
        if _stat_identity(before) != _stat_identity(after) or _stat_identity(after) != _stat_identity(current):
            raise TrainingSnapshotError(f"{label} changed while hashing: {resolved}")
        if size != int(after.st_size):
            raise TrainingSnapshotError(f"{label} size changed while hashing: {resolved}")
        record = {
            "path": str(resolved),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "mtime_ns": int(after.st_mtime_ns),
        }
        self._records[resolved] = record
        return dict(record)

    def json(self, path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
        resolved = _regular_path(path, label=label)
        record = self.file(resolved, label=label)
        cached = self._json.get(resolved)
        if cached is not None:
            return record, dict(cached)
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            raise TrainingSnapshotError(f"cannot read {label} {resolved}: {exc}") from exc
        if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise TrainingSnapshotError(f"{label} changed after hashing: {resolved}")
        value = _strict_json_bytes(data, label=label)
        self._json[resolved] = value
        return record, dict(value)


def _identity(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = {
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
            "size_bytes": int(record["size_bytes"]),
            "mtime_ns": int(record["mtime_ns"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingSnapshotError(f"malformed file record: {record!r}") from exc
    if len(result["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in result["sha256"]):
        raise TrainingSnapshotError(f"malformed SHA-256 in file record: {record!r}")
    if result["size_bytes"] < 0 or result["mtime_ns"] < 0 or not os.path.isabs(result["path"]):
        raise TrainingSnapshotError(f"invalid path/metadata in file record: {record!r}")
    return result


def _bind_declared(
    binder: _Binder, declared: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(declared, Mapping):
        raise TrainingSnapshotError(f"{label} file record is missing")
    expected = _identity(declared)
    observed = binder.file(Path(expected["path"]), label=label)
    if observed != expected:
        raise TrainingSnapshotError(f"{label} file identity drifted")
    return observed


def _same_identity(left: Any, right: Any, *, label: str) -> None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise TrainingSnapshotError(f"{label} file record is missing")
    if _identity(left) != _identity(right):
        raise TrainingSnapshotError(f"{label} file identity differs")


def _attestation_digest(payload: Mapping[str, Any]) -> str:
    view = dict(payload)
    view.pop("attestation_sha256", None)
    return _canonical_sha256(
        {"schema": ATTESTATION_DIGEST_SCHEMA, "attestation": view}
    )


def _recovery_digest(payload: Mapping[str, Any]) -> str:
    view = dict(payload)
    view.pop("receipt_sha256", None)
    return _canonical_sha256({"schema": RECOVERY_SCHEMA, "receipt": view})


def _source_union_from_attestation(
    attestation: Mapping[str, Any], binder: _Binder
) -> list[dict[str, Any]]:
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise TrainingSnapshotError("dependency attestation schema drifted")
    if attestation.get("attestation_sha256") != _attestation_digest(attestation):
        raise TrainingSnapshotError("dependency attestation canonical SHA-256 mismatch")
    closure = attestation.get("dependency_closure", {}).get("file_records")
    static_sources = attestation.get("training_evidence", {}).get(
        "static_repository_sources"
    )
    auditors = attestation.get("auditor_sources")
    if not isinstance(closure, list) or len(closure) != SOURCE_CLOSURE_COUNT:
        raise TrainingSnapshotError("dependency closure must contain exactly 85 records")
    if not isinstance(static_sources, list) or len(static_sources) != STATIC_SOURCE_COUNT:
        raise TrainingSnapshotError("static source list must contain exactly 9 records")
    if not isinstance(auditors, list) or len(auditors) != AUDITOR_SOURCE_COUNT:
        raise TrainingSnapshotError("auditor source list must contain exactly 3 records")

    combined: dict[str, dict[str, Any]] = {}
    groups = (
        ("dependency_closure", closure),
        ("static_repository_source", static_sources),
        ("auditor_source", auditors),
    )
    repository_root = str(attestation.get("repository_root", ""))
    if not os.path.isabs(repository_root):
        raise TrainingSnapshotError("attestation repository_root is not absolute")
    for membership, records in groups:
        for index, raw in enumerate(records):
            observed = _bind_declared(
                binder,
                raw,
                label=f"{membership} source {index}",
            )
            try:
                relative = os.path.relpath(observed["path"], repository_root)
            except ValueError as exc:
                raise TrainingSnapshotError("source path cannot be relativized") from exc
            if relative == ".." or relative.startswith(f"..{os.sep}"):
                raise TrainingSnapshotError(
                    f"training source escapes repository root: {observed['path']}"
                )
            relative = Path(relative).as_posix()
            declared_relative = raw.get("relative_path")
            if declared_relative is not None and declared_relative != relative:
                raise TrainingSnapshotError(
                    f"source relative path drifted for {observed['path']}"
                )
            existing = combined.get(observed["path"])
            if existing is None:
                combined[observed["path"]] = {
                    **observed,
                    "relative_path": relative,
                    "memberships": [membership],
                }
            else:
                _same_identity(existing, observed, label=f"deduplicated source {relative}")
                existing["memberships"].append(membership)
    result = []
    for record in combined.values():
        record["memberships"] = sorted(set(record["memberships"]))
        result.append(record)
    result.sort(key=lambda item: item["relative_path"])
    if len(result) != SOURCE_UNION_COUNT:
        raise TrainingSnapshotError(
            f"source union must contain exactly {SOURCE_UNION_COUNT} paths, got {len(result)}"
        )
    total_size = sum(int(record["size_bytes"]) for record in result)
    if total_size != SOURCE_UNION_SIZE_BYTES:
        raise TrainingSnapshotError(
            "source union byte count drifted: expected "
            f"{SOURCE_UNION_SIZE_BYTES}, got {total_size}"
        )
    return result


def _queue_role_for_ids(run_ids: Sequence[str]) -> str:
    matches = [
        role
        for role, locked in LOCKED_QUEUES.items()
        if list(run_ids) == list(locked["run_ids"])
    ]
    if len(matches) != 1:
        raise TrainingSnapshotError("training queue run order is not a locked partition")
    return matches[0]


def _validate_queue_payload(queue: Mapping[str, Any], *, role: str) -> None:
    locked = LOCKED_QUEUES[role]
    plan = queue.get("plan")
    if not isinstance(plan, Mapping):
        raise TrainingSnapshotError(f"training queue {role} has no plan")
    planned = plan.get("items")
    mutable = queue.get("items")
    if not isinstance(planned, list) or not isinstance(mutable, list):
        raise TrainingSnapshotError(f"training queue {role} items are missing")
    run_ids = [item.get("run_id") if isinstance(item, Mapping) else None for item in planned]
    if run_ids != list(locked["run_ids"]):
        raise TrainingSnapshotError(f"training queue {role} run order drifted")
    if any(not isinstance(item, Mapping) or item.get("runner") != "token" for item in planned):
        raise TrainingSnapshotError(f"training queue {role} has a non-token item")
    if (
        queue.get("schema") != "pivot.stageb.serial_matrix_queue/v1"
        or queue.get("status") != "completed"
        or plan.get("queue_id") != locked["queue_id"]
        or queue.get("plan_sha256") != locked["plan_sha256"]
        or _canonical_sha256(plan) != locked["plan_sha256"]
        or not isinstance(queue.get("revision"), int)
        or int(queue["revision"]) <= 0
        or len(mutable) != len(planned)
    ):
        raise TrainingSnapshotError(f"training queue {role} final identity drifted")
    for index, (expected, item) in enumerate(zip(planned, mutable)):
        if (
            not isinstance(item, Mapping)
            or item.get("index") != index
            or item.get("run_id") != expected.get("run_id")
            or item.get("runner") != "token"
            or item.get("status") != "completed"
        ):
            raise TrainingSnapshotError(
                f"training queue {role} mutable item {index} is not completed and exact"
            )


def _run_live_final_gates(
    *,
    dependency_attestation: Path,
    queue_dirs: Sequence[Path],
    recovery_receipt: Path,
) -> dict[str, Any]:
    """Replay the already-existing final verifiers before snapshotting bytes."""

    from tools import audit_stageb_table_c_dependency_closure as dependency
    from tools import build_stageb_paper_ablation_completion_receipt as completion
    from tools import recover_stageb_serial_matrix_pretraining_failure as recovery
    from tools import run_stageb_serial_matrix_queue as queue_runner

    dependency_result = dependency.verify_attestation(
        dependency_attestation, policy="final"
    )
    if dependency_result.get("status") != "passed":
        raise TrainingSnapshotError("final dependency-closure verification failed")
    sequence_ids, sequence_records = completion._validate_table_c_sequences()
    if sequence_ids != list(EXPECTED_RUN_IDS) or len(sequence_records) != 33:
        raise TrainingSnapshotError("existing B40/U1000 sequence gate is incomplete")
    queue_results = []
    for queue_dir in queue_dirs:
        result = queue_runner.verify_queue(queue_dir)
        if result.get("status") != "passed" or len(result.get("verified_items", [])) not in {5, 28}:
            raise TrainingSnapshotError(
                f"final serial training queue verification failed: {queue_dir}"
            )
        queue_results.append(result)
    recovery_result = recovery.verify_recovery(queue_dirs[1], recovery_receipt)
    if (
        recovery_result.get("status") != "passed"
        or recovery_result.get("run_id") != RECOVERY_RUN_ID
        or recovery_result.get("current_item_status") != "completed"
        or recovery_result.get("archived_evidence_verified") is not True
        or recovery_result.get("semantic_replay") != recovery.SEMANTIC_REPLAY_PROOF
    ):
        raise TrainingSnapshotError("single pretraining recovery replay failed")
    return {
        "dependency_closure_final": {
            "status": "passed",
            "canonical_closure_sha256": dependency_result.get(
                "canonical_closure_sha256"
            ),
        },
        "training_sequences": {"status": "passed", "verified_run_count": 33},
        "training_queues": [
            {
                "status": result["status"],
                "queue_id": result.get("queue_id"),
                "plan_sha256": result.get("plan_sha256"),
                "verified_item_count": len(result.get("verified_items", [])),
            }
            for result in queue_results
        ],
        "single_pretraining_recovery": {
            "status": "passed",
            "run_id": RECOVERY_RUN_ID,
            "failed_revision": RECOVERY_FAILED_REVISION,
            "receipt_sha256": recovery_result.get("receipt_sha256"),
        },
    }


def _validate_budget_and_numerics(
    *,
    run_id: str,
    run_root: str,
    launch: Mapping[str, Any],
    sequence: Mapping[str, Any],
    postflight: Mapping[str, Any],
) -> None:
    row_id, seed_text = run_id.split(":", 1)
    seed = int(seed_text)
    runtime = launch.get("runtime")
    row = launch.get("row")
    inputs = launch.get("inputs")
    if (
        launch.get("schema") != "pivot.stageb.token_ablation_launch/v2"
        or launch.get("status") != "completed"
        or launch.get("returncode") != 0
        or launch.get("run_id") != run_id
        or launch.get("seed") != seed
        or not isinstance(row, Mapping)
        or row.get("row_id") != row_id
        or not isinstance(row.get("config"), str)
        or not isinstance(row.get("token_objective"), str)
        or not isinstance(inputs, Mapping)
        or launch.get("output_dir") != run_root
        or launch.get("output_dir_fresh_at_plan") is not True
        or not isinstance(runtime, Mapping)
        or runtime.get("batch_size") != 40
        or runtime.get("max_train_iters") != 1000
        or runtime.get("iter_checkpoint_interval") != 1000
        or runtime.get("amp") is not True
    ):
        raise TrainingSnapshotError(f"{run_id} launch is not completed B40/U1000")
    budget = sequence.get("equal_budget_contract")
    phases = sequence.get("phases")
    completed = sequence.get("completed_phases")
    if (
        sequence.get("schema") != "pivot.stageb.token_ablation_sequence/v1"
        or sequence.get("status") != "completed"
        or sequence.get("run_id") != run_id
        or sequence.get("seed") != seed
        or sequence.get("row") != dict(row)
        or sequence.get("training_seeds_contract") != list(SEEDS)
        or sequence.get("output_dir") != run_root
        or not isinstance(budget, Mapping)
        or budget.get("batch_size") != 40
        or budget.get("optimizer_updates") != 1000
        or budget.get("contributing_phase_updates") != {"joint": 1000}
        or not isinstance(phases, list)
        or len(phases) != 1
        or phases[0].get("phase_id") != "joint"
        or phases[0].get("optimizer_updates") != 1000
        or phases[0].get("contributes_to_budget") is not True
        or phases[0].get("output_dir") != run_root
        or not isinstance(completed, list)
        or len(completed) != 1
        or completed[0].get("phase_id") != "joint"
        or completed[0].get("status") != "completed"
        or completed[0].get("output_dir") != run_root
    ):
        raise TrainingSnapshotError(f"{run_id} sequence budget/phase contract drifted")
    metadata = postflight.get("checkpoint_metadata")
    numerical = postflight.get("numerical_status")
    checkpoint_args = metadata.get("args") if isinstance(metadata, Mapping) else None
    repository_root = str(launch.get("repository_root", ""))
    expected_paths = {
        "config_file": os.path.normpath(
            os.path.join(repository_root, str(row.get("config", "")))
        ),
        "datasets": str(inputs.get("dataset_manifest", {}).get("path", "")),
        "output_dir": run_root,
        "pretrain_model_path": str(
            inputs.get("stage_a_initializer", {}).get("path", "")
        ),
        "stage_b_v15_scorer_init_checkpoint": str(
            inputs.get("scorer_warmstart", {}).get("path", "")
        ),
    }
    if (
        postflight.get("schema") != "pivot.stageb.token_ablation_postflight/v2"
        or postflight.get("status") != "passed"
        or postflight.get("run_id") != run_id
        or not isinstance(metadata, Mapping)
        or metadata.get("iteration") != 1000
        or metadata.get("epoch") != 0
        or metadata.get("checkpoint_reason") != "max_train_iters"
        or metadata.get("has_complete_training_state") is not True
        or metadata.get("epoch_finished") is not False
        or not isinstance(checkpoint_args, Mapping)
        or checkpoint_args.get("batch_size") != 40
        or checkpoint_args.get("max_train_iters") != 1000
        or checkpoint_args.get("iter_checkpoint_interval") != 1000
        or checkpoint_args.get("seed") != seed
        or checkpoint_args.get("stage_b_v21_token_objective")
        != row.get("token_objective")
        or any(
            not isinstance(checkpoint_args.get(key), str)
            or os.path.normpath(str(checkpoint_args.get(key)))
            != os.path.normpath(expected)
            for key, expected in expected_paths.items()
        )
    ):
        raise TrainingSnapshotError(f"{run_id} checkpoint metadata is not exact U1000")
    if not isinstance(numerical, Mapping):
        raise TrainingSnapshotError(f"{run_id} numerical status is missing")
    scale_min = numerical.get("min_amp_scale")
    scale_max = numerical.get("max_amp_scale")
    if (
        numerical.get("status") != "passed"
        or numerical.get("amp_enabled") is not True
        or numerical.get("loss_values_all_finite") is not True
        or not isinstance(numerical.get("finite_loss_observations"), int)
        or numerical["finite_loss_observations"] <= 0
        or not isinstance(numerical.get("amp_skip_observations"), int)
        or numerical["amp_skip_observations"] <= 0
        or numerical.get("max_amp_step_skipped") != 0.0
        or not isinstance(scale_min, (int, float))
        or isinstance(scale_min, bool)
        or not isinstance(scale_max, (int, float))
        or isinstance(scale_max, bool)
        or not math.isfinite(float(scale_min))
        or not math.isfinite(float(scale_max))
        or float(scale_min) <= 0
        or float(scale_min) != float(scale_max)
    ):
        raise TrainingSnapshotError(
            f"{run_id} numerical evidence lacks finite losses/fixed positive AMP scale/zero skips"
        )


def _normalized_launch_inputs(
    launch: Mapping[str, Any], binder: _Binder, *, run_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = launch.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TrainingSnapshotError(f"{run_id} launch inputs are missing")
    source_raw: list[tuple[str, Any]] = []
    for role, key in (
        ("config_dependency", "config_dependencies"),
        ("repository_source", "repository_sources"),
    ):
        values = inputs.get(key)
        if not isinstance(values, list) or not values:
            raise TrainingSnapshotError(f"{run_id} {key} is missing")
        source_raw.extend((role, value) for value in values)
    non_source_raw: list[tuple[str, Any]] = []
    non_source_raw.append(("dataset_manifest", inputs.get("dataset_manifest")))
    datasets = inputs.get("dataset_source_files")
    if not isinstance(datasets, list) or len(datasets) != 7:
        raise TrainingSnapshotError(f"{run_id} must bind exactly seven dataset sources")
    non_source_raw.extend(("dataset_source_file", value) for value in datasets)
    non_source_raw.extend(
        (
            ("stage_a_initializer", inputs.get("stage_a_initializer")),
            ("scorer_warmstart", inputs.get("scorer_warmstart")),
        )
    )

    def normalize(raw: Sequence[tuple[str, Any]], label: str) -> list[dict[str, Any]]:
        by_path: dict[str, dict[str, Any]] = {}
        for role, value in raw:
            observed = _bind_declared(
                binder, value, label=f"{run_id} {label} {role}"
            )
            existing = by_path.get(observed["path"])
            if existing is None:
                by_path[observed["path"]] = {**observed, "roles": [role]}
            else:
                _same_identity(existing, observed, label=f"{run_id} {label}")
                existing["roles"].append(role)
        result = sorted(by_path.values(), key=lambda record: record["path"])
        for record in result:
            record["roles"] = sorted(set(record["roles"]))
        return result

    return normalize(source_raw, "source input"), normalize(
        non_source_raw, "non-source input"
    )


def _validate_input_rehash(
    *,
    run_id: str,
    launch_sources: Sequence[Mapping[str, Any]],
    non_sources: Sequence[Mapping[str, Any]],
    postflight: Mapping[str, Any],
    input_rehash: Mapping[str, Any],
) -> None:
    if postflight.get("input_rehash") != input_rehash:
        raise TrainingSnapshotError(f"{run_id} embedded and persisted input rehash differ")
    records = input_rehash.get("records")
    if (
        input_rehash.get("status") != "passed"
        or input_rehash.get("algorithm") != "sha256"
        or input_rehash.get("unique_input_count") != PER_RUN_INPUT_REHASH_COUNT
        or not isinstance(records, list)
        or len(records) != PER_RUN_INPUT_REHASH_COUNT
    ):
        raise TrainingSnapshotError(f"{run_id} input rehash is not the exact 36-record gate")
    expected = {
        record["path"]: _identity(record)
        for record in [*launch_sources, *non_sources]
    }
    if len(expected) != PER_RUN_INPUT_REHASH_COUNT:
        raise TrainingSnapshotError(f"{run_id} launch input path union is not 36")
    observed_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TrainingSnapshotError(f"{run_id} input rehash contains a non-object")
        path = str(record.get("path", ""))
        expected_record = expected.get(path)
        if (
            expected_record is None
            or path in observed_paths
            or record.get("passed") is not True
            or record.get("expected_sha256") != expected_record["sha256"]
            or record.get("observed_sha256") != expected_record["sha256"]
            or record.get("observed_size_bytes") != expected_record["size_bytes"]
            or record.get("observed_mtime_ns") != expected_record["mtime_ns"]
        ):
            raise TrainingSnapshotError(f"{run_id} input rehash record drifted for {path}")
        observed_paths.add(path)
    if observed_paths != set(expected):
        raise TrainingSnapshotError(f"{run_id} input rehash path set drifted")


def _collect_live_evidence(
    *,
    dependency_attestation: Path = DEFAULT_DEPENDENCY_ATTESTATION,
    completed_queue: Path = DEFAULT_COMPLETED_QUEUE,
    remaining_queue: Path = DEFAULT_REMAINING_QUEUE,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    recovery_receipt: Path = DEFAULT_RECOVERY_RECEIPT,
) -> dict[str, Any]:
    queue_dirs = (completed_queue, remaining_queue)
    gates = _run_live_final_gates(
        dependency_attestation=dependency_attestation,
        queue_dirs=queue_dirs,
        recovery_receipt=recovery_receipt,
    )
    binder = _Binder()
    attestation_record, attestation = binder.json(
        dependency_attestation, label="Table-C dependency attestation"
    )
    source_union = _source_union_from_attestation(attestation, binder)

    queue_records: list[dict[str, Any]] = []
    queue_payloads: dict[str, dict[str, Any]] = {}
    queue_items: dict[str, tuple[str, dict[str, Any]]] = {}
    for queue_dir in queue_dirs:
        queue_record, queue = binder.json(
            queue_dir / "queue.json", label="completed training queue"
        )
        plan_items = queue.get("plan", {}).get("items", [])
        run_ids = [item.get("run_id") for item in plan_items if isinstance(item, Mapping)]
        role = _queue_role_for_ids(run_ids)
        _validate_queue_payload(queue, role=role)
        queue_payloads[role] = queue
        queue_records.append(
            {
                "role": role,
                "queue_id": queue["plan"]["queue_id"],
                "plan_sha256": queue["plan_sha256"],
                "revision": queue["revision"],
                "ordered_run_ids": list(LOCKED_QUEUES[role]["run_ids"]),
                "queue_json": queue_record,
            }
        )
        for item in queue["items"]:
            run_id = str(item["run_id"])
            if run_id in queue_items:
                raise TrainingSnapshotError(f"duplicate queue item for {run_id}")
            queue_items[run_id] = (role, dict(item))
    queue_records.sort(key=lambda record: list(LOCKED_QUEUES).index(record["role"]))
    if set(queue_items) != set(EXPECTED_RUN_IDS):
        raise TrainingSnapshotError("training queue union is not the exact 33 runs")

    remaining = queue_payloads["remaining_table_c"]
    recovery_events = [
        event
        for event in remaining.get("events", [])
        if isinstance(event, Mapping)
        and event.get("event")
        == "pretraining_environment_failure_archived_and_reopened"
    ]
    recovered_items = [
        item
        for item in remaining.get("items", [])
        if isinstance(item, Mapping)
        and item.get("pretraining_recovery_receipts") is not None
    ]
    recovery_record, recovery = binder.json(
        recovery_receipt, label="L2:42 recovery receipt"
    )
    if (
        recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("status") != "archived_and_eligible_for_fresh_retry"
        or recovery.get("run_id") != RECOVERY_RUN_ID
        or recovery.get("queue", {}).get("failed_revision") != RECOVERY_FAILED_REVISION
        or recovery.get("queue", {}).get("queue_id")
        != LOCKED_QUEUES["remaining_table_c"]["queue_id"]
        or recovery.get("queue", {}).get("plan_sha256")
        != LOCKED_QUEUES["remaining_table_c"]["plan_sha256"]
        or recovery.get("receipt_sha256") != _recovery_digest(recovery)
        or len(recovery_events) != 1
        or recovery_events[0].get("run_id") != RECOVERY_RUN_ID
        or recovery_events[0].get("failed_revision") != RECOVERY_FAILED_REVISION
        or len(recovered_items) != 1
        or recovered_items[0].get("run_id") != RECOVERY_RUN_ID
    ):
        raise TrainingSnapshotError("queue does not bind exactly one L2:42 revision-590 recovery")
    _same_identity(
        recovery_events[0].get("receipt"), recovery_record, label="queue recovery event"
    )
    histories = recovered_items[0].get("pretraining_recovery_receipts")
    if not isinstance(histories, list) or len(histories) != 1:
        raise TrainingSnapshotError("L2:42 recovery history is not singular")
    _same_identity(histories[0], recovery_record, label="L2:42 recovery history")

    training_root_string = str(training_root.expanduser().resolve(strict=True))
    runs: list[dict[str, Any]] = []
    canonical_non_sources: list[dict[str, Any]] | None = None
    all_launch_source_paths: dict[str, dict[str, Any]] = {}
    all_full_input_paths: set[str] = set()
    for run_id in EXPECTED_RUN_IDS:
        row_id, seed_text = run_id.split(":", 1)
        seed = int(seed_text)
        run_root_path = (Path(training_root_string) / row_id / f"seed{seed}").resolve(
            strict=True
        )
        run_root = str(run_root_path)
        first_class: dict[str, dict[str, Any]] = {}
        payloads: dict[str, dict[str, Any]] = {}
        for role, filename in FIRST_CLASS_RUN_FILES.items():
            path = run_root_path / filename
            if filename.endswith(".json"):
                record, payload = binder.json(path, label=f"{run_id} {role}")
                payloads[role] = payload
            else:
                record = binder.file(path, label=f"{run_id} {role}")
            first_class[role] = record
        sequence = payloads["sequence_manifest"]
        launch = payloads["launch_manifest"]
        postflight = payloads["postflight"]
        input_rehash = payloads["input_rehash"]
        _validate_budget_and_numerics(
            run_id=run_id,
            run_root=run_root,
            launch=launch,
            sequence=sequence,
            postflight=postflight,
        )
        if launch.get("postflight") != postflight:
            raise TrainingSnapshotError(f"{run_id} embedded postflight differs from persisted")
        _same_identity(
            launch.get("postflight_artifact"),
            first_class["postflight"],
            label=f"{run_id} launch postflight",
        )
        completed_phase = sequence["completed_phases"][0]
        _same_identity(
            completed_phase.get("checkpoint"),
            first_class["checkpoint"],
            label=f"{run_id} sequence checkpoint",
        )
        _same_identity(
            completed_phase.get("postflight"),
            first_class["postflight"],
            label=f"{run_id} sequence postflight",
        )
        launch_sources, non_sources = _normalized_launch_inputs(
            launch, binder, run_id=run_id
        )
        if canonical_non_sources is None:
            canonical_non_sources = non_sources
        elif canonical_non_sources != non_sources:
            raise TrainingSnapshotError(
                f"{run_id} ten non-source launch inputs differ from prior runs"
            )
        for record in launch_sources:
            existing = all_launch_source_paths.get(record["path"])
            if existing is None:
                all_launch_source_paths[record["path"]] = record
            else:
                _same_identity(existing, record, label="cross-run launch source")
        all_full_input_paths.update(record["path"] for record in launch_sources)
        all_full_input_paths.update(record["path"] for record in non_sources)
        _validate_input_rehash(
            run_id=run_id,
            launch_sources=launch_sources,
            non_sources=non_sources,
            postflight=postflight,
            input_rehash=input_rehash,
        )
        source_union_by_path = {record["path"]: record for record in source_union}
        for record in launch_sources:
            expected = source_union_by_path.get(record["path"])
            if expected is None:
                raise TrainingSnapshotError(
                    f"{run_id} launch source is outside the 89-source union: {record['path']}"
                )
            _same_identity(expected, record, label=f"{run_id} launch source union")

        raw_artifacts = postflight.get("artifacts")
        if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != set(
            POSTFLIGHT_ARTIFACT_FILES
        ):
            raise TrainingSnapshotError(f"{run_id} postflight artifact set is not exact")
        postflight_artifacts: dict[str, dict[str, Any]] = {}
        for role, filename in POSTFLIGHT_ARTIFACT_FILES.items():
            declared = raw_artifacts[role]
            expected_path = run_root_path / filename
            observed = _bind_declared(
                binder, declared, label=f"{run_id} postflight artifact {role}"
            )
            if observed["path"] != str(expected_path):
                raise TrainingSnapshotError(
                    f"{run_id} postflight artifact {role} path drifted"
                )
            postflight_artifacts[role] = observed
        _same_identity(
            postflight_artifacts["checkpoint"],
            first_class["checkpoint"],
            label=f"{run_id} duplicate checkpoint role",
        )
        _same_identity(
            postflight_artifacts["input_rehash"],
            first_class["input_rehash"],
            label=f"{run_id} duplicate input-rehash role",
        )
        launch_manifest_pointer = postflight.get("launch_manifest")
        if (
            not isinstance(launch_manifest_pointer, Mapping)
            or launch_manifest_pointer.get("path") != first_class["launch_manifest"]["path"]
            or launch_manifest_pointer.get("present") is not True
            or launch_manifest_pointer.get("hash_omitted")
            != "manifest embeds postflight and is updated after validation"
        ):
            raise TrainingSnapshotError(
                f"{run_id} launch-manifest cycle omission contract drifted"
            )

        queue_role, queue_item = queue_items[run_id]
        job_dir = str(queue_item.get("job_dir", ""))
        if not os.path.isabs(job_dir):
            raise TrainingSnapshotError(f"{run_id} final detached job_dir is invalid")
        detached_launch_record, detached_launch = binder.json(
            Path(job_dir) / "launch.json", label=f"{run_id} final detached launch"
        )
        detached_status_record, detached_status = binder.json(
            Path(job_dir) / "status.json", label=f"{run_id} final detached status"
        )
        if (
            detached_launch.get("schema")
            != "pivot.stageb.token_ablation_detached_launch/v1"
            or detached_launch.get("status") != "launched"
            or detached_launch.get("run_ids") != [run_id]
            or detached_launch.get("expected_run_roots") != [run_root]
            or detached_launch.get("job_dir") != job_dir
            or detached_status.get("schema")
            != "pivot.stageb.token_ablation_orchestration_status/v1"
            or detached_status.get("status") != "completed"
            or detached_status.get("run_ids") != [run_id]
            or detached_status.get("completed_run_ids") != [run_id]
            or detached_status.get("expected_run_roots") != [run_root]
            or detached_status.get("job_dir") != job_dir
        ):
            raise TrainingSnapshotError(f"{run_id} final detached job evidence drifted")
        completion = queue_item.get("completion_evidence")
        if (
            not isinstance(completion, Mapping)
            or completion.get("run_id") != run_id
            or completion.get("job_dir") != job_dir
            or completion.get("output_root") != run_root
            or completion.get("sequence_manifest")
            != first_class["sequence_manifest"]["path"]
            or completion.get("sequence_sha256")
            != first_class["sequence_manifest"]["sha256"]
            or not isinstance(completion.get("phases"), list)
            or len(completion["phases"]) != 1
            or completion["phases"][0].get("phase_id") != "joint"
            or completion["phases"][0].get("launch_manifest")
            != first_class["launch_manifest"]["path"]
            or completion["phases"][0].get("postflight")
            != first_class["postflight"]["path"]
            or completion["phases"][0].get("postflight_sha256")
            != first_class["postflight"]["sha256"]
        ):
            raise TrainingSnapshotError(f"{run_id} queue completion binding drifted")
        histories = queue_item.get("pretraining_recovery_receipts")
        if run_id == RECOVERY_RUN_ID:
            if not isinstance(histories, list) or len(histories) != 1:
                raise TrainingSnapshotError("recovered L2:42 final item lacks one receipt")
            _same_identity(histories[0], recovery_record, label="L2:42 recovery receipt")
        elif histories is not None:
            raise TrainingSnapshotError(f"unexpected recovery history on {run_id}")

        runs.append(
            {
                "run_id": run_id,
                "row_id": row_id,
                "seed": seed,
                "queue_role": queue_role,
                "queue_item_index": queue_item["index"],
                "training_root": run_root,
                "final_detached_job_dir": job_dir,
                "artifacts": {
                    **first_class,
                    "detached_launch": detached_launch_record,
                    "detached_status": detached_status_record,
                },
                "postflight_artifacts": postflight_artifacts,
            }
        )

    if canonical_non_sources is None or len(canonical_non_sources) != NON_SOURCE_INPUT_COUNT:
        raise TrainingSnapshotError("non-source launch input union is not exactly 10")
    if len(all_launch_source_paths) != LAUNCH_SOURCE_UNION_COUNT:
        raise TrainingSnapshotError(
            f"cross-run launch source union must be 36, got {len(all_launch_source_paths)}"
        )
    if len(all_full_input_paths) != FULL_LAUNCH_INPUT_UNION_COUNT:
        raise TrainingSnapshotError(
            f"cross-run full launch input union must be 46, got {len(all_full_input_paths)}"
        )
    return {
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "training_root": training_root_string,
        "live_verification_gates": gates,
        "dependency_attestation": attestation_record,
        "source_union": source_union,
        "non_source_launch_inputs": canonical_non_sources,
        "launch_source_union": sorted(
            all_launch_source_paths.values(), key=lambda record: record["path"]
        ),
        "training_queues": queue_records,
        "pretraining_recovery": {
            "run_id": RECOVERY_RUN_ID,
            "failed_revision": RECOVERY_FAILED_REVISION,
            "receipt_sha256": recovery["receipt_sha256"],
            "receipt": recovery_record,
        },
        "runs": runs,
    }


def _object_relative_path(sha256: str) -> str:
    return f"objects/sha256/{sha256[:2]}/{sha256}"


def _copy_object(
    source: Path,
    expected: Mapping[str, Any],
    *,
    snapshot_root: Path,
) -> str:
    expected_identity = _identity(expected)
    source = _regular_path(source, label="snapshot object source")
    if str(source) != expected_identity["path"]:
        raise TrainingSnapshotError("snapshot object source path drifted")
    relative = _object_relative_path(expected_identity["sha256"])
    destination = snapshot_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        observed = _Binder().file(destination, label="deduplicated snapshot object")
        if (
            observed["sha256"] != expected_identity["sha256"]
            or observed["size_bytes"] != expected_identity["size_bytes"]
        ):
            raise TrainingSnapshotError("content-addressed object collision")
        current_source = _Binder().file(source, label="deduplicated live object")
        if current_source != expected_identity:
            raise TrainingSnapshotError("deduplicated live object drifted before archive")
        return relative
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as input_handle, temporary.open("xb") as output_handle:
            before = os.fstat(input_handle.fileno())
            for chunk in iter(lambda: input_handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
            after = os.fstat(input_handle.fileno())
        current = os.stat(source, follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
            or digest.hexdigest() != expected_identity["sha256"]
            or size != expected_identity["size_bytes"]
            or int(after.st_mtime_ns) != expected_identity["mtime_ns"]
        ):
            raise TrainingSnapshotError(f"live object drifted while archiving: {source}")
        os.replace(temporary, destination)
        os.chmod(destination, 0o444)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return relative


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered = _canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise TrainingSnapshotError(f"snapshot file must be fresh: {path}") from exc
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(rendered).hexdigest(),
        "size_bytes": len(rendered),
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - required Linux runtime
        raise TrainingSnapshotError(
            "atomic snapshot publication requires Linux renameat2"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    number = ctypes.get_errno()
    if number == errno.EEXIST:
        raise TrainingSnapshotError(
            f"snapshot appeared concurrently and was not replaced: {destination}"
        )
    raise OSError(number, os.strerror(number), str(destination))


def _iter_archivable_records(evidence: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    # Completion evidence remains live-bound.  Only the exact retrospective
    # 89-source union is copied into the content-addressed store.
    yield from evidence["source_union"]


def _with_object(
    record: Mapping[str, Any], object_paths: Mapping[str, str]
) -> dict[str, Any]:
    identity = _identity(record)
    try:
        relative = object_paths[identity["path"]]
    except KeyError as exc:
        raise TrainingSnapshotError(
            f"file was not archived: {identity['path']}"
        ) from exc
    return {**identity, "archive_object": relative}


def _build_payloads(
    evidence: Mapping[str, Any], *, snapshot_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    unique: dict[str, Mapping[str, Any]] = {}
    for record in _iter_archivable_records(evidence):
        identity = _identity(record)
        existing = unique.get(identity["path"])
        if existing is not None:
            _same_identity(existing, identity, label="duplicate archive path")
        else:
            unique[identity["path"]] = identity
    object_paths = {
        path: _copy_object(Path(path), record, snapshot_root=snapshot_root)
        for path, record in sorted(unique.items())
    }
    if len(object_paths) != SOURCE_UNION_COUNT or len(set(object_paths.values())) != SOURCE_UNION_COUNT:
        raise TrainingSnapshotError(
            "source object store must contain 89 unique paths and 89 unique digests"
        )
    source_snapshot: dict[str, Any] = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "status": "retrospective_training_source_snapshot",
        "claim_scope": {
            "historical_launch_manifests_modified": False,
            "retroactively_launch_binds_omitted_files": False,
            "snapshot_builder_is_a_training_source": False,
        },
        "derivation": {
            "dependency_closure_records": SOURCE_CLOSURE_COUNT,
            "static_repository_source_records": STATIC_SOURCE_COUNT,
            "auditor_source_records": AUDITOR_SOURCE_COUNT,
            "path_deduplicated_source_count": SOURCE_UNION_COUNT,
            "path_deduplicated_size_bytes": SOURCE_UNION_SIZE_BYTES,
        },
        "sources": [
            {
                **_with_object(record, object_paths),
                "relative_path": record["relative_path"],
                "memberships": list(record["memberships"]),
            }
            for record in evidence["source_union"]
        ],
        "object_store": {
            "algorithm": "sha256",
            "layout": "objects/sha256/HH/SHA256",
            "object_count": SOURCE_UNION_COUNT,
            "unique_source_path_count": SOURCE_UNION_COUNT,
            "unique_source_digest_count": SOURCE_UNION_COUNT,
        },
    }
    source_snapshot["source_snapshot_sha256"] = _source_snapshot_digest(
        source_snapshot
    )
    source_snapshot_record = _write_json_exclusive(
        snapshot_root / "source_snapshot.json", source_snapshot
    )

    completion_subreceipt: dict[str, Any] = {
        "schema": COMPLETION_SUBRECEIPT_SCHEMA,
        "status": "complete_retrospective_training_completion_subreceipt",
        "created_at_utc": evidence["created_at_utc"],
        "claim_scope": {
            "historical_launch_manifests_modified": False,
            "retroactively_launch_binds_omitted_files": False,
            "outside_historical_dependency_closure": True,
            "purpose": "preserve completed Table-C B40/U1000 training evidence",
        },
        "repository_root_at_build": evidence["repository_root"],
        "training_root_at_build": evidence["training_root"],
        "expected_run_ids": list(EXPECTED_RUN_IDS),
        "live_verification_gates": evidence["live_verification_gates"],
        "dependency_attestation": dict(evidence["dependency_attestation"]),
        "source_snapshot": {
            **source_snapshot_record,
            "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        },
        "non_source_launch_inputs": [
            dict(record) for record in evidence["non_source_launch_inputs"]
        ],
        "launch_source_union": [
            dict(record) for record in evidence["launch_source_union"]
        ],
        "training_queues": [
            {
                **{key: value for key, value in queue.items() if key != "queue_json"},
                "queue_json": dict(queue["queue_json"]),
            }
            for queue in evidence["training_queues"]
        ],
        "pretraining_recovery": {
            **{
                key: value
                for key, value in evidence["pretraining_recovery"].items()
                if key != "receipt"
            },
            "receipt": dict(evidence["pretraining_recovery"]["receipt"]),
        },
        "runs": [
            {
                **{
                    key: value
                    for key, value in run.items()
                    if key not in {"artifacts", "postflight_artifacts"}
                },
                "artifacts": {
                    role: dict(record)
                    for role, record in run["artifacts"].items()
                },
                "postflight_artifacts": {
                    role: dict(record)
                    for role, record in run["postflight_artifacts"].items()
                },
            }
            for run in evidence["runs"]
        ],
    }
    completion_subreceipt["completion_subreceipt_sha256"] = (
        _completion_subreceipt_digest(completion_subreceipt)
    )
    return completion_subreceipt, source_snapshot


def dry_run_snapshot() -> dict[str, Any]:
    evidence = _collect_live_evidence()
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed_live_dry_run",
        "run_count": len(evidence["runs"]),
        "source_count": len(evidence["source_union"]),
        "source_size_bytes": sum(
            record["size_bytes"] for record in evidence["source_union"]
        ),
        "non_source_input_count": len(evidence["non_source_launch_inputs"]),
        "launch_source_union_count": len(evidence["launch_source_union"]),
        "live_verification_gates": evidence["live_verification_gates"],
    }


def build_snapshot(output_root: Path | None = None) -> dict[str, Any]:
    canonical = CANONICAL_OUTPUT_ROOT.resolve(strict=False)
    destination = (CANONICAL_OUTPUT_ROOT if output_root is None else output_root).expanduser().resolve(
        strict=False
    )
    if destination != canonical:
        raise TrainingSnapshotError(
            f"snapshot output is not canonical: expected {canonical}, got {destination}"
        )
    if destination.exists():
        raise TrainingSnapshotError(f"snapshot output must be fresh: {destination}")
    evidence = _collect_live_evidence()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir(exist_ok=False)
    try:
        completion_subreceipt, _ = _build_payloads(
            evidence, snapshot_root=temporary
        )
        _write_json_exclusive(
            temporary / "completion_subreceipt.json", completion_subreceipt
        )
        _fsync_directory(temporary)
        _rename_noreplace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return verify_snapshot(destination)


class _ObjectReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: dict[str, tuple[str, int]] = {}
        self.bytes: dict[str, bytes] = {}

    def read(self, record: Any, *, label: str) -> bytes:
        if not isinstance(record, Mapping):
            raise TrainingSnapshotError(f"{label} archived record is missing")
        identity = _identity(record)
        relative = record.get("archive_object")
        expected_relative = _object_relative_path(identity["sha256"])
        if relative != expected_relative:
            raise TrainingSnapshotError(f"{label} content-addressed path drifted")
        pure = PurePosixPath(str(relative))
        if pure.is_absolute() or ".." in pure.parts:
            raise TrainingSnapshotError(f"{label} archive path escapes snapshot")
        previous = self.records.get(str(relative))
        object_identity = (identity["sha256"], identity["size_bytes"])
        if previous is not None and previous != object_identity:
            raise TrainingSnapshotError(f"{label} object identity collision")
        self.records[str(relative)] = object_identity
        cached = self.bytes.get(str(relative))
        if cached is not None:
            return cached
        path = self.root / pure
        regular = _regular_path(path, label=f"{label} archived object")
        try:
            if regular.relative_to(self.root) != Path(pure):
                raise TrainingSnapshotError(f"{label} archived object resolves unexpectedly")
            data = regular.read_bytes()
        except (OSError, ValueError) as exc:
            raise TrainingSnapshotError(f"cannot read {label} archived object: {exc}") from exc
        if len(data) != identity["size_bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise TrainingSnapshotError(f"{label} archived object bytes drifted")
        self.bytes[str(relative)] = data
        return data

    def json(self, record: Any, *, label: str) -> dict[str, Any]:
        return _strict_json_bytes(self.read(record, label=label), label=label)

    def verify_inventory(self, expected_count: int) -> None:
        object_root = self.root / "objects"
        observed: set[str] = set()
        if object_root.exists():
            for path in object_root.rglob("*"):
                raw = os.lstat(path)
                if stat.S_ISLNK(raw.st_mode):
                    raise TrainingSnapshotError(f"symlink in object store: {path}")
                if stat.S_ISDIR(raw.st_mode):
                    continue
                if not stat.S_ISREG(raw.st_mode):
                    raise TrainingSnapshotError(f"non-regular object-store entry: {path}")
                observed.add(path.relative_to(self.root).as_posix())
        if observed != set(self.records) or len(observed) != expected_count:
            raise TrainingSnapshotError(
                "object-store inventory differs from the referenced content-addressed set"
            )


class _LiveReader:
    """Read identity-bound completion evidence from its original live path."""

    def __init__(self) -> None:
        self.binder = _Binder()
        self.records: dict[str, dict[str, Any]] = {}

    def read(self, record: Any, *, label: str) -> bytes:
        expected = _identity(record)
        observed = self.binder.file(Path(expected["path"]), label=label)
        if observed != expected:
            raise TrainingSnapshotError(f"{label} live identity drifted")
        existing = self.records.get(expected["path"])
        if existing is not None:
            _same_identity(existing, expected, label=label)
        else:
            self.records[expected["path"]] = expected
        try:
            data = Path(expected["path"]).read_bytes()
        except OSError as exc:
            raise TrainingSnapshotError(f"cannot read {label}: {exc}") from exc
        if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise TrainingSnapshotError(f"{label} changed after live rehash")
        return data

    def json(self, record: Any, *, label: str) -> dict[str, Any]:
        return _strict_json_bytes(self.read(record, label=label), label=label)


def _read_root_json(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    regular = _regular_path(path, label=label)
    try:
        data = regular.read_bytes()
    except OSError as exc:
        raise TrainingSnapshotError(f"cannot read {label}: {exc}") from exc
    return data, _strict_json_bytes(data, label=label)


def _verify_source_snapshot(
    source_snapshot: Mapping[str, Any],
    attestation: Mapping[str, Any],
    reader: _ObjectReader,
) -> None:
    if (
        source_snapshot.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or source_snapshot.get("status") != "retrospective_training_source_snapshot"
        or source_snapshot.get("source_snapshot_sha256")
        != _source_snapshot_digest(source_snapshot)
        or source_snapshot.get("claim_scope", {}).get("retroactively_launch_binds_omitted_files")
        is not False
        or source_snapshot.get("claim_scope", {}).get("snapshot_builder_is_a_training_source")
        is not False
    ):
        raise TrainingSnapshotError("source snapshot contract or self-hash drifted")
    derivation = source_snapshot.get("derivation")
    sources = source_snapshot.get("sources")
    if (
        derivation
        != {
            "dependency_closure_records": SOURCE_CLOSURE_COUNT,
            "static_repository_source_records": STATIC_SOURCE_COUNT,
            "auditor_source_records": AUDITOR_SOURCE_COUNT,
            "path_deduplicated_source_count": SOURCE_UNION_COUNT,
            "path_deduplicated_size_bytes": SOURCE_UNION_SIZE_BYTES,
        }
        or not isinstance(sources, list)
        or len(sources) != SOURCE_UNION_COUNT
        or sum(int(record.get("size_bytes", -1)) for record in sources if isinstance(record, Mapping))
        != SOURCE_UNION_SIZE_BYTES
    ):
        raise TrainingSnapshotError("source snapshot count/byte derivation drifted")
    paths: set[str] = set()
    relatives: set[str] = set()
    for index, record in enumerate(sources):
        reader.read(record, label=f"source union {index}")
        identity = _identity(record)
        relative = record.get("relative_path")
        memberships = record.get("memberships")
        if (
            identity["path"] in paths
            or not isinstance(relative, str)
            or not relative
            or relative in relatives
            or not isinstance(memberships, list)
            or memberships != sorted(set(memberships))
            or not set(memberships)
            <= {
                "dependency_closure",
                "static_repository_source",
                "auditor_source",
            }
        ):
            raise TrainingSnapshotError("source snapshot contains invalid dedup metadata")
        paths.add(identity["path"])
        relatives.add(relative)
    if (
        attestation.get("schema") != ATTESTATION_SCHEMA
        or attestation.get("attestation_sha256") != _attestation_digest(attestation)
    ):
        raise TrainingSnapshotError("live dependency attestation self-hash drifted")
    expected: dict[str, dict[str, Any]] = {}
    categories = (
        (
            "dependency_closure",
            attestation.get("dependency_closure", {}).get("file_records"),
            SOURCE_CLOSURE_COUNT,
        ),
        (
            "static_repository_source",
            attestation.get("training_evidence", {}).get("static_repository_sources"),
            STATIC_SOURCE_COUNT,
        ),
        ("auditor_source", attestation.get("auditor_sources"), AUDITOR_SOURCE_COUNT),
    )
    repository_root = str(attestation.get("repository_root", ""))
    for membership, records, count in categories:
        if not isinstance(records, list) or len(records) != count:
            raise TrainingSnapshotError("archived attestation source category drifted")
        for raw in records:
            identity = _identity(raw)
            relative = Path(os.path.relpath(identity["path"], repository_root)).as_posix()
            entry = expected.setdefault(
                identity["path"],
                {**identity, "relative_path": relative, "memberships": []},
            )
            _same_identity(entry, identity, label="archived attestation source union")
            entry["memberships"].append(membership)
    normalized_expected = []
    for record in expected.values():
        record["memberships"] = sorted(set(record["memberships"]))
        normalized_expected.append(record)
    normalized_expected.sort(key=lambda record: record["relative_path"])
    normalized_observed = [
        {
            **_identity(record),
            "relative_path": record["relative_path"],
            "memberships": record["memberships"],
        }
        for record in sources
    ]
    if normalized_observed != normalized_expected:
        raise TrainingSnapshotError(
            "source snapshot is not the exact path-deduplicated attestation union"
        )
    object_store = source_snapshot.get("object_store")
    if (
        not isinstance(object_store, Mapping)
        or object_store.get("algorithm") != "sha256"
        or object_store.get("layout") != "objects/sha256/HH/SHA256"
        or object_store.get("object_count") != SOURCE_UNION_COUNT
        or object_store.get("unique_source_path_count") != SOURCE_UNION_COUNT
        or object_store.get("unique_source_digest_count") != SOURCE_UNION_COUNT
        or len({record["sha256"] for record in sources}) != SOURCE_UNION_COUNT
    ):
        raise TrainingSnapshotError("source snapshot object-store declaration drifted")


def _verify_completion_semantics(
    snapshot: Mapping[str, Any], reader: _LiveReader
) -> None:
    queue_records = snapshot.get("training_queues")
    if not isinstance(queue_records, list) or len(queue_records) != 2:
        raise TrainingSnapshotError("snapshot must bind exactly two training queues")
    queue_payloads: dict[str, dict[str, Any]] = {}
    queue_items: dict[str, tuple[str, dict[str, Any]]] = {}
    for record in queue_records:
        if not isinstance(record, Mapping):
            raise TrainingSnapshotError("training queue snapshot record is invalid")
        role = str(record.get("role", ""))
        if role not in LOCKED_QUEUES or role in queue_payloads:
            raise TrainingSnapshotError("training queue role set drifted")
        queue = reader.json(record.get("queue_json"), label=f"archived queue {role}")
        _validate_queue_payload(queue, role=role)
        if (
            record.get("queue_id") != queue["plan"]["queue_id"]
            or record.get("plan_sha256") != queue["plan_sha256"]
            or record.get("revision") != queue["revision"]
            or record.get("ordered_run_ids") != list(LOCKED_QUEUES[role]["run_ids"])
        ):
            raise TrainingSnapshotError(f"archived queue {role} metadata drifted")
        queue_payloads[role] = queue
        for item in queue["items"]:
            queue_items[item["run_id"]] = (role, item)
    if list(queue_payloads) != list(LOCKED_QUEUES) or set(queue_items) != set(EXPECTED_RUN_IDS):
        raise TrainingSnapshotError("archived queue union/order drifted")

    recovery_record = snapshot.get("pretraining_recovery")
    if not isinstance(recovery_record, Mapping):
        raise TrainingSnapshotError("snapshot recovery binding is missing")
    recovery = reader.json(recovery_record.get("receipt"), label="archived recovery receipt")
    if (
        recovery_record.get("run_id") != RECOVERY_RUN_ID
        or recovery_record.get("failed_revision") != RECOVERY_FAILED_REVISION
        or recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("run_id") != RECOVERY_RUN_ID
        or recovery.get("queue", {}).get("failed_revision") != RECOVERY_FAILED_REVISION
        or recovery.get("receipt_sha256") != _recovery_digest(recovery)
        or recovery_record.get("receipt_sha256") != recovery.get("receipt_sha256")
    ):
        raise TrainingSnapshotError("archived recovery digest/revision binding drifted")
    remaining = queue_payloads["remaining_table_c"]
    events = [
        event
        for event in remaining.get("events", [])
        if isinstance(event, Mapping)
        and event.get("event")
        == "pretraining_environment_failure_archived_and_reopened"
    ]
    recovered = [
        item
        for item in remaining["items"]
        if item.get("pretraining_recovery_receipts") is not None
    ]
    if (
        len(events) != 1
        or events[0].get("run_id") != RECOVERY_RUN_ID
        or events[0].get("failed_revision") != RECOVERY_FAILED_REVISION
        or len(recovered) != 1
        or recovered[0].get("run_id") != RECOVERY_RUN_ID
    ):
        raise TrainingSnapshotError("archived queue recovery history is not exact")
    _same_identity(events[0].get("receipt"), recovery_record["receipt"], label="recovery event")
    histories = recovered[0].get("pretraining_recovery_receipts")
    if not isinstance(histories, list) or len(histories) != 1:
        raise TrainingSnapshotError("archived recovered item receipt history drifted")
    _same_identity(histories[0], recovery_record["receipt"], label="recovered item")

    non_sources = snapshot.get("non_source_launch_inputs")
    launch_source_union = snapshot.get("launch_source_union")
    if (
        not isinstance(non_sources, list)
        or len(non_sources) != NON_SOURCE_INPUT_COUNT
        or not isinstance(launch_source_union, list)
        or len(launch_source_union) != LAUNCH_SOURCE_UNION_COUNT
    ):
        raise TrainingSnapshotError("snapshot launch input union cardinalities drifted")
    non_source_map = {_identity(record)["path"]: _identity(record) for record in non_sources}
    launch_source_map = {
        _identity(record)["path"]: _identity(record) for record in launch_source_union
    }
    if len(non_source_map) != 10 or len(launch_source_map) != 36:
        raise TrainingSnapshotError("snapshot launch input union contains duplicate paths")
    for index, record in enumerate(non_sources):
        reader.read(record, label=f"non-source launch input {index}")

    runs = snapshot.get("runs")
    if not isinstance(runs, list) or [run.get("run_id") for run in runs if isinstance(run, Mapping)] != list(EXPECTED_RUN_IDS):
        raise TrainingSnapshotError("snapshot run order is not exact")
    observed_launch_sources: dict[str, dict[str, Any]] = {}
    for run in runs:
        run_id = str(run["run_id"])
        row_id, seed_text = run_id.split(":", 1)
        seed = int(seed_text)
        artifacts = run.get("artifacts")
        postflight_artifacts = run.get("postflight_artifacts")
        if (
            run.get("row_id") != row_id
            or run.get("seed") != seed
            or not isinstance(artifacts, Mapping)
            or set(artifacts)
            != {*FIRST_CLASS_RUN_FILES, "detached_launch", "detached_status"}
            or not isinstance(postflight_artifacts, Mapping)
            or set(postflight_artifacts) != set(POSTFLIGHT_ARTIFACT_FILES)
        ):
            raise TrainingSnapshotError(f"{run_id} archived role set drifted")
        sequence = reader.json(artifacts["sequence_manifest"], label=f"{run_id} sequence")
        launch = reader.json(artifacts["launch_manifest"], label=f"{run_id} launch")
        postflight = reader.json(artifacts["postflight"], label=f"{run_id} postflight")
        input_rehash = reader.json(artifacts["input_rehash"], label=f"{run_id} input rehash")
        reader.read(artifacts["checkpoint"], label=f"{run_id} checkpoint")
        detached_launch = reader.json(artifacts["detached_launch"], label=f"{run_id} detached launch")
        detached_status = reader.json(artifacts["detached_status"], label=f"{run_id} detached status")
        for role, record in postflight_artifacts.items():
            reader.read(record, label=f"{run_id} postflight artifact {role}")
        run_root = str(run.get("training_root", ""))
        _validate_budget_and_numerics(
            run_id=run_id,
            run_root=run_root,
            launch=launch,
            sequence=sequence,
            postflight=postflight,
        )
        if launch.get("postflight") != postflight:
            raise TrainingSnapshotError(f"{run_id} archived embedded postflight differs")
        _same_identity(launch.get("postflight_artifact"), artifacts["postflight"], label=f"{run_id} launch postflight")
        completed_phase = sequence["completed_phases"][0]
        _same_identity(completed_phase.get("checkpoint"), artifacts["checkpoint"], label=f"{run_id} sequence checkpoint")
        _same_identity(completed_phase.get("postflight"), artifacts["postflight"], label=f"{run_id} sequence postflight")
        raw_postflight_artifacts = postflight.get("artifacts")
        if not isinstance(raw_postflight_artifacts, Mapping) or set(raw_postflight_artifacts) != set(POSTFLIGHT_ARTIFACT_FILES):
            raise TrainingSnapshotError(f"{run_id} archived postflight artifact set drifted")
        for role in POSTFLIGHT_ARTIFACT_FILES:
            _same_identity(raw_postflight_artifacts[role], postflight_artifacts[role], label=f"{run_id} postflight {role}")
        _same_identity(postflight_artifacts["checkpoint"], artifacts["checkpoint"], label=f"{run_id} checkpoint roles")
        _same_identity(postflight_artifacts["input_rehash"], artifacts["input_rehash"], label=f"{run_id} input rehash roles")
        pointer = postflight.get("launch_manifest")
        if (
            not isinstance(pointer, Mapping)
            or pointer.get("path") != artifacts["launch_manifest"]["path"]
            or pointer.get("present") is not True
            or pointer.get("hash_omitted")
            != "manifest embeds postflight and is updated after validation"
        ):
            raise TrainingSnapshotError(f"{run_id} archived launch cycle marker drifted")
        launch_sources, run_non_sources = _normalized_declared_inputs_offline(launch)
        for record in launch_sources:
            expected = launch_source_map.get(record["path"])
            if expected is None:
                raise TrainingSnapshotError(f"{run_id} archived launch source outside union")
            _same_identity(expected, record, label=f"{run_id} archived launch source")
            observed_launch_sources[record["path"]] = _identity(record)
        if {_identity(record)["path"]: _identity(record) for record in run_non_sources} != non_source_map:
            raise TrainingSnapshotError(f"{run_id} archived non-source input set drifted")
        _validate_input_rehash(
            run_id=run_id,
            launch_sources=launch_sources,
            non_sources=run_non_sources,
            postflight=postflight,
            input_rehash=input_rehash,
        )
        role, queue_item = queue_items[run_id]
        if (
            run.get("queue_role") != role
            or run.get("queue_item_index") != queue_item["index"]
            or run.get("final_detached_job_dir") != queue_item.get("job_dir")
            or detached_launch.get("status") != "launched"
            or detached_launch.get("run_ids") != [run_id]
            or detached_launch.get("expected_run_roots") != [run_root]
            or detached_launch.get("job_dir") != run["final_detached_job_dir"]
            or detached_status.get("status") != "completed"
            or detached_status.get("run_ids") != [run_id]
            or detached_status.get("completed_run_ids") != [run_id]
            or detached_status.get("expected_run_roots") != [run_root]
            or detached_status.get("job_dir") != run["final_detached_job_dir"]
        ):
            raise TrainingSnapshotError(f"{run_id} archived final detached job binding drifted")
        completion = queue_item.get("completion_evidence")
        if (
            not isinstance(completion, Mapping)
            or completion.get("job_dir") != run["final_detached_job_dir"]
            or completion.get("sequence_sha256") != artifacts["sequence_manifest"]["sha256"]
            or completion.get("sequence_manifest") != artifacts["sequence_manifest"]["path"]
            or len(completion.get("phases", [])) != 1
            or completion["phases"][0].get("postflight_sha256") != artifacts["postflight"]["sha256"]
        ):
            raise TrainingSnapshotError(f"{run_id} archived queue completion evidence drifted")
        histories = queue_item.get("pretraining_recovery_receipts")
        if run_id == RECOVERY_RUN_ID:
            if not isinstance(histories, list) or len(histories) != 1:
                raise TrainingSnapshotError("archived L2:42 recovery binding drifted")
        elif histories is not None:
            raise TrainingSnapshotError(f"archived unexpected recovery on {run_id}")
    if observed_launch_sources != launch_source_map:
        raise TrainingSnapshotError("archived cross-run launch source union drifted")


def _normalized_declared_inputs_offline(
    launch: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = launch.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TrainingSnapshotError("archived launch inputs are missing")

    def normalize(values: Sequence[tuple[str, Any]]) -> list[dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for role, raw in values:
            if not isinstance(raw, Mapping):
                raise TrainingSnapshotError("archived launch input record is missing")
            identity = _identity(raw)
            existing = records.get(identity["path"])
            if existing is None:
                records[identity["path"]] = {**identity, "roles": [role]}
            else:
                _same_identity(existing, identity, label="archived launch input")
                existing["roles"].append(role)
        result = sorted(records.values(), key=lambda record: record["path"])
        for record in result:
            record["roles"] = sorted(set(record["roles"]))
        return result

    configs = inputs.get("config_dependencies")
    repositories = inputs.get("repository_sources")
    datasets = inputs.get("dataset_source_files")
    if (
        not isinstance(configs, list)
        or not configs
        or not isinstance(repositories, list)
        or not repositories
        or not isinstance(datasets, list)
        or len(datasets) != 7
    ):
        raise TrainingSnapshotError("archived launch input cardinalities drifted")
    sources = normalize(
        [("config_dependency", value) for value in configs]
        + [("repository_source", value) for value in repositories]
    )
    non_sources = normalize(
        [("dataset_manifest", inputs.get("dataset_manifest"))]
        + [("dataset_source_file", value) for value in datasets]
        + [
            ("stage_a_initializer", inputs.get("stage_a_initializer")),
            ("scorer_warmstart", inputs.get("scorer_warmstart")),
        ]
    )
    return sources, non_sources


def _live_source_parity(source_snapshot: Mapping[str, Any]) -> int:
    records: dict[str, dict[str, Any]] = {}

    def add(raw: Any, label: str) -> None:
        identity = _identity(raw)
        existing = records.get(identity["path"])
        if existing is not None:
            _same_identity(existing, identity, label=label)
        else:
            records[identity["path"]] = identity

    for record in source_snapshot["sources"]:
        add(record, "live source parity")
    if len(records) != SOURCE_UNION_COUNT:
        raise TrainingSnapshotError("live source parity set is not exactly 89 paths")
    binder = _Binder()
    for path, expected in sorted(records.items()):
        observed = binder.file(Path(path), label="live source parity")
        if observed != expected:
            raise TrainingSnapshotError(f"live source parity drifted: {path}")
    return len(records)


def verify_snapshot(
    snapshot_root: Path | None = None,
    *,
    require_live_source_parity: bool = False,
) -> dict[str, Any]:
    root = (CANONICAL_OUTPUT_ROOT if snapshot_root is None else snapshot_root).expanduser().resolve(
        strict=True
    )
    if not root.is_dir() or root.is_symlink():
        raise TrainingSnapshotError(f"snapshot root is not a regular directory: {root}")
    completion_bytes, completion = _read_root_json(
        root / "completion_subreceipt.json", label="completion subreceipt"
    )
    if completion_bytes != _canonical_json_bytes(completion) + b"\n":
        raise TrainingSnapshotError("completion subreceipt bytes are not canonical compact JSON")
    if (
        completion.get("schema") != COMPLETION_SUBRECEIPT_SCHEMA
        or completion.get("status")
        != "complete_retrospective_training_completion_subreceipt"
        or completion.get("completion_subreceipt_sha256")
        != _completion_subreceipt_digest(completion)
        or completion.get("expected_run_ids") != list(EXPECTED_RUN_IDS)
        or completion.get("claim_scope", {}).get("outside_historical_dependency_closure")
        is not True
        or completion.get("claim_scope", {}).get("retroactively_launch_binds_omitted_files")
        is not False
    ):
        raise TrainingSnapshotError(
            "completion subreceipt contract or domain-separated self-hash drifted"
        )
    source_record = completion.get("source_snapshot")
    if (
        not isinstance(source_record, Mapping)
        or source_record.get("relative_path") != "source_snapshot.json"
    ):
        raise TrainingSnapshotError("completion source-snapshot binding is missing")
    source_bytes, source_snapshot = _read_root_json(
        root / "source_snapshot.json", label="source snapshot"
    )
    if source_bytes != _canonical_json_bytes(source_snapshot) + b"\n":
        raise TrainingSnapshotError("source snapshot bytes are not canonical compact JSON")
    if (
        source_record.get("sha256") != hashlib.sha256(source_bytes).hexdigest()
        or source_record.get("size_bytes") != len(source_bytes)
        or source_record.get("source_snapshot_sha256")
        != source_snapshot.get("source_snapshot_sha256")
    ):
        raise TrainingSnapshotError("completion source-snapshot file/self-hash binding drifted")

    live_reader = _LiveReader()
    attestation = live_reader.json(
        completion.get("dependency_attestation"),
        label="live dependency attestation",
    )
    object_reader = _ObjectReader(root)
    _verify_source_snapshot(source_snapshot, attestation, object_reader)
    object_reader.verify_inventory(SOURCE_UNION_COUNT)

    archived_source_map = {
        _identity(record)["path"]: _identity(record)
        for record in source_snapshot["sources"]
    }
    launch_source_union = completion.get("launch_source_union")
    if not isinstance(launch_source_union, list) or len(launch_source_union) != LAUNCH_SOURCE_UNION_COUNT:
        raise TrainingSnapshotError("completion launch-source union is not exactly 36")
    for record in launch_source_union:
        identity = _identity(record)
        expected = archived_source_map.get(identity["path"])
        if expected is None:
            raise TrainingSnapshotError(
                f"completion launch source is outside archived source union: {identity['path']}"
            )
        _same_identity(expected, identity, label="completion launch source")
    _verify_completion_semantics(completion, live_reader)

    parity_count = None
    strict_gate_result = None
    if require_live_source_parity:
        parity_count = _live_source_parity(source_snapshot)
        queue_paths = [
            Path(_identity(queue["queue_json"])["path"]).parent
            for queue in completion["training_queues"]
        ]
        strict_gate_result = _run_live_final_gates(
            dependency_attestation=Path(
                _identity(completion["dependency_attestation"])["path"]
            ),
            queue_dirs=queue_paths,
            recovery_receipt=Path(
                _identity(completion["pretraining_recovery"]["receipt"])["path"]
            ),
        )
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "snapshot_root": str(root),
        "completion_subreceipt_file_sha256": hashlib.sha256(
            completion_bytes
        ).hexdigest(),
        "completion_subreceipt_sha256": completion[
            "completion_subreceipt_sha256"
        ],
        "source_snapshot_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "source_count": SOURCE_UNION_COUNT,
        "source_size_bytes": SOURCE_UNION_SIZE_BYTES,
        "run_count": len(EXPECTED_RUN_IDS),
        "object_count": len(object_reader.records),
        "live_completion_record_count": len(live_reader.records),
        "live_source_parity_required": require_live_source_parity,
        "live_parity_record_count": parity_count,
        "strict_live_final_gates": strict_gate_result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("dry-run", help="replay live final gates without writing")
    subparsers.add_parser("build", help="build the one fresh canonical snapshot")
    verify = subparsers.add_parser("verify", help="verify archived bytes and bindings")
    verify.add_argument(
        "--require-live-source-parity",
        action="store_true",
        help="also require every originally bound live path to retain its identity",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "dry-run":
            result = dry_run_snapshot()
        elif args.mode == "build":
            result = build_snapshot()
        else:
            result = verify_snapshot(
                require_live_source_parity=args.require_live_source_parity
            )
    except (OSError, ValueError, TrainingSnapshotError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
