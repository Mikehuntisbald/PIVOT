#!/usr/bin/env python3
"""Archive a proven pretraining launcher failure and reopen one queue item.

This is deliberately narrower than a general retry mechanism.  It accepts only
the Stage-B failure where GPU identity capture failed before ``main.py`` was
spawned.  The failed output and detached-job trees are renamed into an
immutable evidence directory before the existing queue item is reopened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_serial_matrix_queue as queue_runner  # noqa: E402


RECEIPT_SCHEMA = "pivot.stageb.serial_queue_pretraining_recovery_receipt/v1"
COMMIT_SCHEMA = "pivot.stageb.serial_queue_pretraining_recovery_commit/v1"
FAILURE_FRAGMENT = "nvidia-smi identity query failed:"
FAILURE_PHASE = "gpu_telemetry_or_training_process"
QUEUE_FAILURE_EXACT = (
    "QueueContractError: detached runner reached terminal failure failed: "
    "persisted_status_is_explicitly_terminal"
)
RECOVERY_EVENT = "pretraining_environment_failure_archived_and_reopened"
SEMANTIC_REPLAY_PROOF = {
    "queue_before_validated": True,
    "pretraining_failure_rederived": True,
    "archive_move_intent_replayed": True,
    "queue_after_reopen_validated": True,
    "commit_validated": True,
}


class RecoveryError(RuntimeError):
    """The failed queue item is not eligible for the narrow recovery."""


def _utc_now() -> str:
    return queue_runner._utc_now()


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
    if not path.is_file() or path.is_symlink():
        raise RecoveryError(f"evidence is not a regular file: {path}")
    metadata = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
    }


def _verify_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise RecoveryError(f"{label} file record is missing")
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    if _file_record(path) != dict(record):
        raise RecoveryError(f"{label} file identity drifted: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} is not a JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RecoveryError(f"recovery artifact must be fresh: {path}") from exc
    _fsync_directory(path.parent)


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RecoveryError(f"recovery artifact must be fresh: {path}") from exc
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tree_inventory(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RecoveryError(f"evidence root is not a regular directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RecoveryError(f"symlink is forbidden in recovery evidence: {path}")
        if path.is_dir():
            continue
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RecoveryError(f"non-regular recovery evidence is forbidden: {path}")
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": int(metadata.st_size),
            }
        )
    payload = {"root": str(root), "file_count": len(files), "files": files}
    payload["inventory_sha256"] = _canonical_sha(
        {"file_count": len(files), "files": files}
    )
    return payload


def _verify_inventory(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _tree_inventory(root)
    expected_view = {
        "file_count": expected.get("file_count"),
        "files": expected.get("files"),
        "inventory_sha256": expected.get("inventory_sha256"),
    }
    observed_view = {
        "file_count": observed["file_count"],
        "files": observed["files"],
        "inventory_sha256": observed["inventory_sha256"],
    }
    if observed_view != expected_view:
        raise RecoveryError(f"archived evidence inventory drifted: {root}")
    return observed


def _require_failure_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or FAILURE_FRAGMENT not in value:
        raise RecoveryError(f"{label} is not the approved GPU identity failure")
    return value


def _expected_output_root(queue: Mapping[str, Any], run_id: str) -> Path:
    row, raw_seed = run_id.split(":", 1)
    environment = queue.get("plan", {}).get("runtime_environment", {})
    if not isinstance(environment, Mapping):
        raise RecoveryError("queue runtime environment is missing")
    raw_root = environment.get("PIVOT_TOKEN_OUTPUT_ROOT")
    if not isinstance(raw_root, str) or not raw_root:
        raise RecoveryError("queue does not bind PIVOT_TOKEN_OUTPUT_ROOT")
    root = Path(raw_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return (root / row / f"seed{int(raw_seed)}").resolve(strict=False)


def _process_is_dead(
    item: Mapping[str, Any],
    *,
    pid_key: str,
    identity_key: str,
    label: str,
) -> bool:
    pid = item.get(pid_key)
    identity = item.get(identity_key)
    running = queue_runner._process_running(pid, identity)
    if running is not False:
        raise RecoveryError(f"{label} process liveness is not authoritatively false")
    return True


def inspect_failure(
    queue_dir: Path,
    *,
    run_id: str,
    expected_queue_id: str,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    queue = queue_runner.load_queue(queue_dir)
    plan = queue.get("plan", {})
    if (
        plan.get("queue_id") != expected_queue_id
        or queue.get("plan_sha256") != expected_plan_sha256
    ):
        raise RecoveryError("queue identity differs from the approved failed queue")
    if queue.get("status") != "failed":
        raise RecoveryError("queue is not in the failed state")
    failure = queue.get("failure")
    if not isinstance(failure, Mapping) or failure.get("run_id") != run_id:
        raise RecoveryError("queue failure does not identify the requested run")
    item = next(
        (candidate for candidate in queue["items"] if candidate.get("run_id") == run_id),
        None,
    )
    if not isinstance(item, Mapping) or item.get("status") != "failed":
        raise RecoveryError("requested queue item is not the single failed item")
    index = int(item.get("index", -1))
    if failure.get("index") != index:
        raise RecoveryError("queue and item failure indices differ")
    if (
        item.get("failure_error") != QUEUE_FAILURE_EXACT
        or failure.get("error") != QUEUE_FAILURE_EXACT
    ):
        raise RecoveryError("queue failure is not the exact child-terminal wrapper")

    _process_is_dead(
        item,
        pid_key="child_pid",
        identity_key="child_process_identity",
        label="child",
    )
    _process_is_dead(
        item,
        pid_key="detach_launcher_pid",
        identity_key="detach_launcher_identity",
        label="detach launcher",
    )

    job_dir = Path(str(item.get("job_dir", ""))).resolve(strict=True)
    orchestration_root = Path(str(item.get("orchestration_root", ""))).resolve(
        strict=True
    )
    output_root = Path(str(item.get("output_root", ""))).resolve(strict=True)
    if job_dir.parent != orchestration_root:
        raise RecoveryError("failed detached job is outside its orchestration root")
    expected_orchestration = (
        queue_dir / "jobs" / f"{index:03d}-{run_id.replace(':', '_')}"
    ).resolve(strict=False)
    if orchestration_root != expected_orchestration:
        raise RecoveryError("failed orchestration root is not canonical")
    if output_root != _expected_output_root(queue, run_id):
        raise RecoveryError("failed output root is not canonical")

    status = _read_json(job_dir / "status.json", label="detached status")
    launch = _read_json(job_dir / "launch.json", label="detached launch")
    sequence = _read_json(output_root / "sequence_manifest.json", label="sequence")
    phase = _read_json(output_root / "launch_manifest.json", label="phase launch")
    if (
        status.get("status") != "failed"
        or status.get("run_ids") != [run_id]
        or launch.get("status") != "launched"
        or launch.get("run_ids") != [run_id]
        or sequence.get("status") != "failed"
        or sequence.get("run_id") != run_id
        or sequence.get("completed_phases") != []
        or phase.get("status") != "failed"
        or phase.get("run_id") != run_id
        or phase.get("failure_phase") != FAILURE_PHASE
    ):
        raise RecoveryError("failed launcher evidence is not a pretraining failure")
    for label, value in (
        ("detached status", status.get("error")),
        ("sequence", sequence.get("error")),
        ("phase launch", phase.get("failure_error")),
    ):
        _require_failure_text(value, label=label)

    output_inventory = _tree_inventory(output_root)
    output_files = {entry["relative_path"] for entry in output_inventory["files"]}
    if output_files != {"launch_manifest.json", "sequence_manifest.json"}:
        raise RecoveryError("failed output contains training or unexpected artifacts")

    orchestration_inventory = _tree_inventory(orchestration_root)
    relative_job = job_dir.name
    expected_job_files = {
        "detach_launcher.log",
        f"{relative_job}/launch.json",
        f"{relative_job}/orchestrator.log",
        f"{relative_job}/plans/{run_id.split(':', 1)[0]}/seed{run_id.split(':', 1)[1]}.json",
        f"{relative_job}/status.json",
    }
    observed_job_files = {
        entry["relative_path"] for entry in orchestration_inventory["files"]
    }
    if observed_job_files != expected_job_files:
        raise RecoveryError("failed orchestration tree contains unexpected artifacts")

    lease_path = Path(str(plan.get("lease_path", ""))).resolve(strict=True)
    lease = _read_json(lease_path, label="GPU lease")
    if (
        lease.get("status") != "owned"
        or lease.get("queue_id") != expected_queue_id
        or lease.get("plan_sha256") != expected_plan_sha256
    ):
        raise RecoveryError("failed queue no longer owns its exact GPU lease")

    return {
        "schema": "pivot.stageb.serial_queue_pretraining_recovery_inspection/v1",
        "status": "eligible",
        "inspected_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "queue_id": expected_queue_id,
        "plan_sha256": expected_plan_sha256,
        "failed_revision": int(queue.get("revision", -1)),
        "run_id": run_id,
        "index": index,
        "failure_class": "gpu_identity_capture_failed_before_training_process",
        "failure": dict(failure),
        "failed_item": dict(item),
        "job_dir": str(job_dir),
        "orchestration_root": str(orchestration_root),
        "output_root": str(output_root),
        "output_inventory": output_inventory,
        "orchestration_inventory": orchestration_inventory,
        "lease": _file_record(lease_path),
        "proof": {
            "child_process_dead": True,
            "detach_launcher_dead": True,
            "completed_phases": 0,
            "checkpoint_count": 0,
            "postflight_count": 0,
            "training_runtime_artifact_count": 0,
            "output_dir_fresh_at_failed_plan": phase.get("output_dir_fresh_at_plan"),
        },
    }


def _receipt_digest(payload: Mapping[str, Any]) -> str:
    view = dict(payload)
    view.pop("receipt_sha256", None)
    return _canonical_sha({"schema": RECEIPT_SCHEMA, "receipt": view})


def apply_recovery(
    queue_dir: Path,
    *,
    run_id: str,
    archive_root: Path,
    expected_queue_id: str,
    expected_plan_sha256: str,
    expected_failed_revision: int,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    archive_root = archive_root.expanduser().resolve(strict=False)
    if archive_root.exists():
        raise RecoveryError(f"archive root must be fresh: {archive_root}")

    lock_path = queue_dir / "supervisor.lock"
    with queue_runner._exclusive_file_lock(
        lock_path,
        busy_message=f"queue supervisor is active: {queue_dir}",
    ):
        inspection = inspect_failure(
            queue_dir,
            run_id=run_id,
            expected_queue_id=expected_queue_id,
            expected_plan_sha256=expected_plan_sha256,
        )
        if inspection["failed_revision"] != expected_failed_revision:
            raise RecoveryError(
                "failed queue revision differs from the explicitly approved revision"
            )
        queue_path = queue_dir / "queue.json"
        queue_before = queue_path.read_bytes()
        if hashlib.sha256(queue_before).hexdigest() != _sha256_file(queue_path):
            raise RecoveryError("queue changed while its snapshot was read")

        archive_root.mkdir(parents=True, exist_ok=False)
        _fsync_directory(archive_root.parent)
        queue_snapshot = archive_root / "queue_before.json"
        _write_bytes_exclusive(queue_snapshot, queue_before)
        archived_tool = archive_root / "recovery_tool.py"
        _write_bytes_exclusive(archived_tool, Path(__file__).read_bytes())
        archived_output = archive_root / "failed_output_root"
        archived_orchestration = archive_root / "failed_orchestration_root"
        intent = {
            "schema": "pivot.stageb.serial_queue_pretraining_recovery_intent/v1",
            "created_at_utc": _utc_now(),
            "queue_id": expected_queue_id,
            "plan_sha256": expected_plan_sha256,
            "failed_revision": expected_failed_revision,
            "run_id": run_id,
            "index": inspection["index"],
            "queue_before": _file_record(queue_snapshot),
            "recovery_tool": _file_record(archived_tool),
            "moves": [
                {
                    "source": inspection["output_root"],
                    "destination": str(archived_output),
                    "inventory": inspection["output_inventory"],
                },
                {
                    "source": inspection["orchestration_root"],
                    "destination": str(archived_orchestration),
                    "inventory": inspection["orchestration_inventory"],
                },
            ],
        }
        intent_path = archive_root / "recovery_intent.json"
        _write_json_exclusive(intent_path, intent)

        source_output = Path(inspection["output_root"])
        source_orchestration = Path(inspection["orchestration_root"])
        moved_output = False
        moved_orchestration = False
        try:
            os.rename(source_output, archived_output)
            moved_output = True
            os.rename(source_orchestration, archived_orchestration)
            moved_orchestration = True
            _fsync_directory(source_output.parent)
            _fsync_directory(source_orchestration.parent)
            _fsync_directory(archive_root)
            archived_output_inventory = _verify_inventory(
                archived_output, inspection["output_inventory"]
            )
            archived_orchestration_inventory = _verify_inventory(
                archived_orchestration, inspection["orchestration_inventory"]
            )
        except BaseException:
            if moved_orchestration and not source_orchestration.exists():
                os.rename(archived_orchestration, source_orchestration)
            if moved_output and not source_output.exists():
                os.rename(archived_output, source_output)
            raise

        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "status": "archived_and_eligible_for_fresh_retry",
            "created_at_utc": _utc_now(),
            "failure_class": inspection["failure_class"],
            "queue": {
                "queue_dir": str(queue_dir),
                "queue_id": expected_queue_id,
                "plan_sha256": expected_plan_sha256,
                "failed_revision": expected_failed_revision,
                "expected_reopened_revision": expected_failed_revision + 1,
            },
            "run_id": run_id,
            "index": inspection["index"],
            "failure": inspection["failure"],
            "failed_item": inspection["failed_item"],
            "proof": inspection["proof"],
            "archive": {
                "root": str(archive_root),
                "queue_before": _file_record(queue_snapshot),
                "intent": _file_record(intent_path),
                "failed_output_root": archived_output_inventory,
                "failed_orchestration_root": archived_orchestration_inventory,
            },
            "recovery_contract": {
                "immutable_plan_unchanged": True,
                "same_queue_id_retained": True,
                "failed_evidence_renamed_not_deleted": True,
                "fresh_output_and_orchestration_required_for_retry": True,
                "training_dependency_sources_unchanged": True,
            },
            "recovery_tool": _file_record(archived_tool),
        }
        receipt["receipt_sha256"] = _receipt_digest(receipt)
        receipt_path = archive_root / "recovery_receipt.json"
        _write_json_exclusive(receipt_path, receipt)
        receipt_record = _file_record(receipt_path)

        queue = queue_runner.load_queue(queue_dir)
        if int(queue.get("revision", -1)) != expected_failed_revision:
            raise RecoveryError("queue changed after failure evidence was archived")
        index = int(inspection["index"])
        old_item = queue["items"][index]
        if old_item.get("status") != "failed" or old_item.get("run_id") != run_id:
            raise RecoveryError("failed item changed before queue reopen")
        queue["items"][index] = {
            "index": index,
            "run_id": run_id,
            "runner": old_item["runner"],
            "status": "pending",
            "pretraining_recovery_receipts": [receipt_record],
        }
        queue["status"] = "running"
        queue.pop("failure", None)
        queue["events"].append(
            {
                "at_utc": _utc_now(),
                "event": RECOVERY_EVENT,
                "index": index,
                "run_id": run_id,
                "failed_revision": expected_failed_revision,
                "failure_class": inspection["failure_class"],
                "receipt": receipt_record,
            }
        )
        queue_runner._save_queue(queue)

        reopened = queue_runner.load_queue(queue_dir)
        reopened_item = reopened["items"][index]
        if (
            reopened.get("status") != "running"
            or reopened.get("revision") != expected_failed_revision + 1
            or reopened_item.get("status") != "pending"
            or reopened_item.get("pretraining_recovery_receipts") != [receipt_record]
            or source_output.exists()
            or source_orchestration.exists()
        ):
            raise RecoveryError("queue reopen postconditions failed")

        queue_after_snapshot = archive_root / "queue_after_reopen.json"
        _write_bytes_exclusive(queue_after_snapshot, queue_path.read_bytes())
        commit = {
            "schema": COMMIT_SCHEMA,
            "status": "committed",
            "committed_at_utc": _utc_now(),
            "queue_after_reopen": _file_record(queue_after_snapshot),
            "queue_id": expected_queue_id,
            "plan_sha256": expected_plan_sha256,
            "revision": reopened["revision"],
            "run_id": run_id,
            "item_status": reopened_item["status"],
            "receipt": receipt_record,
        }
        commit_path = archive_root / "recovery_commit.json"
        _write_json_exclusive(commit_path, commit)
        return {
            "schema": "pivot.stageb.serial_queue_pretraining_recovery_result/v1",
            "status": "reopened",
            "queue_id": expected_queue_id,
            "plan_sha256": expected_plan_sha256,
            "run_id": run_id,
            "revision": reopened["revision"],
            "receipt": receipt_record,
            "commit": _file_record(commit_path),
            "archive_root": str(archive_root),
        }


def _inventory_content_view(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file_count": value.get("file_count"),
        "files": value.get("files"),
        "inventory_sha256": value.get("inventory_sha256"),
    }


def _semantic_replay_receipt(
    receipt_path: Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    binding = receipt.get("queue")
    archive = receipt.get("archive")
    if not isinstance(binding, Mapping) or not isinstance(archive, Mapping):
        raise RecoveryError("recovery receipt queue/archive binding is missing")
    archive_root = Path(str(archive.get("root", ""))).resolve(strict=True)
    if receipt_path.parent != archive_root:
        raise RecoveryError("recovery receipt is outside its declared archive root")
    queue_dir = Path(str(binding.get("queue_dir", ""))).resolve(strict=True)
    queue_before_path = _verify_file_record(
        archive.get("queue_before"), label="queue-before snapshot"
    )
    queue_before = _read_json(queue_before_path, label="queue-before snapshot")
    try:
        queue_runner._validate_queue(queue_before, queue_dir)
    except queue_runner.QueueContractError as exc:
        raise RecoveryError(f"archived queue-before snapshot is invalid: {exc}") from exc
    run_id = str(receipt.get("run_id"))
    index = receipt.get("index")
    failure = queue_before.get("failure")
    items = queue_before.get("items")
    if not isinstance(index, int) or not isinstance(items, list) or not (0 <= index < len(items)):
        raise RecoveryError("recovery receipt failed item index is invalid")
    failed_item = items[index]
    if (
        queue_before.get("status") != "failed"
        or queue_before.get("revision") != binding.get("failed_revision")
        or queue_before.get("plan", {}).get("queue_id") != binding.get("queue_id")
        or queue_before.get("plan_sha256") != binding.get("plan_sha256")
        or failed_item.get("run_id") != run_id
        or failed_item.get("status") != "failed"
        or failed_item.get("failure_error") != QUEUE_FAILURE_EXACT
        or failure != receipt.get("failure")
        or failed_item != receipt.get("failed_item")
    ):
        raise RecoveryError("recovery receipt differs from archived failed queue state")

    output_record = archive.get("failed_output_root")
    orchestration_record = archive.get("failed_orchestration_root")
    if not isinstance(output_record, Mapping) or not isinstance(
        orchestration_record, Mapping
    ):
        raise RecoveryError("recovery receipt failed-tree inventories are missing")
    archived_output = Path(str(output_record.get("root", ""))).resolve(strict=True)
    archived_orchestration = Path(
        str(orchestration_record.get("root", ""))
    ).resolve(strict=True)
    _verify_inventory(archived_output, output_record)
    _verify_inventory(archived_orchestration, orchestration_record)
    sequence = _read_json(
        archived_output / "sequence_manifest.json", label="archived sequence"
    )
    phase = _read_json(
        archived_output / "launch_manifest.json", label="archived phase launch"
    )
    archived_job = archived_orchestration / Path(str(failed_item["job_dir"])).name
    status = _read_json(archived_job / "status.json", label="archived child status")
    launch = _read_json(archived_job / "launch.json", label="archived child launch")
    for label, value in (
        ("archived child status", status.get("error")),
        ("archived sequence", sequence.get("error")),
        ("archived phase launch", phase.get("failure_error")),
    ):
        _require_failure_text(value, label=label)
    if (
        status.get("status") != "failed"
        or status.get("run_ids") != [run_id]
        or launch.get("status") != "launched"
        or launch.get("run_ids") != [run_id]
        or sequence.get("status") != "failed"
        or sequence.get("run_id") != run_id
        or sequence.get("completed_phases") != []
        or phase.get("status") != "failed"
        or phase.get("run_id") != run_id
        or phase.get("failure_phase") != FAILURE_PHASE
    ):
        raise RecoveryError("archived manifests do not prove a pretraining failure")
    expected_proof = {
        "child_process_dead": True,
        "detach_launcher_dead": True,
        "completed_phases": 0,
        "checkpoint_count": 0,
        "postflight_count": 0,
        "training_runtime_artifact_count": 0,
        "output_dir_fresh_at_failed_plan": phase.get("output_dir_fresh_at_plan"),
    }
    if receipt.get("proof") != expected_proof:
        raise RecoveryError("recovery receipt proof was not derived from archived manifests")

    intent_path = _verify_file_record(archive.get("intent"), label="recovery intent")
    intent = _read_json(intent_path, label="recovery intent")
    if (
        intent.get("schema")
        != "pivot.stageb.serial_queue_pretraining_recovery_intent/v1"
        or intent.get("queue_id") != binding.get("queue_id")
        or intent.get("plan_sha256") != binding.get("plan_sha256")
        or intent.get("failed_revision") != binding.get("failed_revision")
        or intent.get("run_id") != run_id
        or intent.get("index") != index
        or intent.get("queue_before") != archive.get("queue_before")
        or intent.get("recovery_tool") != receipt.get("recovery_tool")
    ):
        raise RecoveryError("recovery intent identity differs from the receipt")
    moves = intent.get("moves")
    expected_moves = (
        (failed_item["output_root"], archived_output, output_record),
        (failed_item["orchestration_root"], archived_orchestration, orchestration_record),
    )
    if not isinstance(moves, list) or len(moves) != len(expected_moves):
        raise RecoveryError("recovery intent move list is invalid")
    for move, (source, destination, inventory) in zip(moves, expected_moves):
        if (
            not isinstance(move, Mapping)
            or move.get("source") != source
            or Path(str(move.get("destination", ""))).resolve(strict=False)
            != destination
            or not isinstance(move.get("inventory"), Mapping)
            or _inventory_content_view(move["inventory"])
            != _inventory_content_view(inventory)
        ):
            raise RecoveryError("recovery intent move differs from archived evidence")

    commit_path = archive_root / "recovery_commit.json"
    commit = _read_json(commit_path, label="recovery commit")
    receipt_record = _file_record(receipt_path)
    queue_after_path = _verify_file_record(
        commit.get("queue_after_reopen"), label="queue-after snapshot"
    )
    queue_after = _read_json(queue_after_path, label="queue-after snapshot")
    try:
        queue_runner._validate_queue(queue_after, queue_dir)
    except queue_runner.QueueContractError as exc:
        raise RecoveryError(f"archived queue-after snapshot is invalid: {exc}") from exc
    reopened_item = queue_after["items"][index]
    recovery_events = [
        event
        for event in queue_after.get("events", [])
        if isinstance(event, Mapping)
        and event.get("event") == RECOVERY_EVENT
        and event.get("run_id") == run_id
    ]
    if (
        commit.get("schema") != COMMIT_SCHEMA
        or commit.get("status") != "committed"
        or commit.get("queue_id") != binding.get("queue_id")
        or commit.get("plan_sha256") != binding.get("plan_sha256")
        or commit.get("revision") != binding.get("expected_reopened_revision")
        or commit.get("run_id") != run_id
        or commit.get("item_status") != "pending"
        or commit.get("receipt") != receipt_record
        or queue_after.get("status") != "running"
        or queue_after.get("revision") != binding.get("expected_reopened_revision")
        or queue_after.get("plan_sha256") != binding.get("plan_sha256")
        or reopened_item.get("status") != "pending"
        or reopened_item.get("pretraining_recovery_receipts") != [receipt_record]
        or len(recovery_events) != 1
        or recovery_events[0].get("failed_revision") != binding.get("failed_revision")
        or recovery_events[0].get("receipt") != receipt_record
    ):
        raise RecoveryError("recovery commit does not prove the exact queue reopen")
    expected_contract = {
        "immutable_plan_unchanged": True,
        "same_queue_id_retained": True,
        "failed_evidence_renamed_not_deleted": True,
        "fresh_output_and_orchestration_required_for_retry": True,
        "training_dependency_sources_unchanged": True,
    }
    if (
        receipt.get("status") != "archived_and_eligible_for_fresh_retry"
        or receipt.get("failure_class")
        != "gpu_identity_capture_failed_before_training_process"
        or receipt.get("recovery_contract") != expected_contract
    ):
        raise RecoveryError("recovery receipt contract is not the sealed narrow policy")
    return dict(SEMANTIC_REPLAY_PROOF)


def verify_recovery(queue_dir: Path, receipt_path: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path, label="recovery receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RecoveryError("unsupported recovery receipt schema")
    if receipt.get("receipt_sha256") != _receipt_digest(receipt):
        raise RecoveryError("recovery receipt canonical SHA-256 mismatch")
    queue = queue_runner.load_queue(queue_dir)
    binding = receipt.get("queue", {})
    if (
        queue.get("plan", {}).get("queue_id") != binding.get("queue_id")
        or queue.get("plan_sha256") != binding.get("plan_sha256")
        or str(queue_dir) != binding.get("queue_dir")
    ):
        raise RecoveryError("recovery receipt queue binding drifted")
    _verify_file_record(receipt.get("recovery_tool"), label="recovery tool snapshot")
    semantic_replay = _semantic_replay_receipt(receipt_path, receipt)
    receipt_record = _file_record(receipt_path)
    run_id = str(receipt.get("run_id"))
    item = next(
        (candidate for candidate in queue["items"] if candidate.get("run_id") == run_id),
        None,
    )
    histories = item.get("pretraining_recovery_receipts") if isinstance(item, Mapping) else None
    if not isinstance(histories, list) or receipt_record not in histories:
        raise RecoveryError("queue item no longer binds the recovery receipt")
    matching_events = [
        event
        for event in queue.get("events", [])
        if isinstance(event, Mapping)
        and event.get("event") == RECOVERY_EVENT
        and event.get("run_id") == run_id
        and event.get("receipt") == receipt_record
    ]
    if len(matching_events) != 1:
        raise RecoveryError("queue does not retain exactly one recovery event")
    return {
        "schema": "pivot.stageb.serial_queue_pretraining_recovery_verification/v1",
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "queue_id": binding.get("queue_id"),
        "plan_sha256": binding.get("plan_sha256"),
        "run_id": run_id,
        "current_item_status": item.get("status"),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "archived_evidence_verified": True,
        "semantic_replay": semantic_replay,
        "verifier_source": _file_record(Path(__file__)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    inspect = subparsers.add_parser("inspect", help="prove failure eligibility")
    apply = subparsers.add_parser("apply", help="archive evidence and reopen item")
    verify = subparsers.add_parser("verify", help="replay a committed recovery")
    for child in (inspect, apply):
        child.add_argument("queue_dir", type=Path)
        child.add_argument("--run-id", required=True)
        child.add_argument("--expected-queue-id", required=True)
        child.add_argument("--expected-plan-sha256", required=True)
    apply.add_argument("--expected-failed-revision", type=int, required=True)
    apply.add_argument("--archive-root", type=Path, required=True)
    verify.add_argument("queue_dir", type=Path)
    verify.add_argument("receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "inspect":
            result = inspect_failure(
                args.queue_dir,
                run_id=args.run_id,
                expected_queue_id=args.expected_queue_id,
                expected_plan_sha256=args.expected_plan_sha256,
            )
        elif args.mode == "apply":
            result = apply_recovery(
                args.queue_dir,
                run_id=args.run_id,
                archive_root=args.archive_root,
                expected_queue_id=args.expected_queue_id,
                expected_plan_sha256=args.expected_plan_sha256,
                expected_failed_revision=args.expected_failed_revision,
            )
        else:
            result = verify_recovery(args.queue_dir, args.receipt)
    except (OSError, RecoveryError, queue_runner.QueueContractError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
