#!/usr/bin/env python3
"""Recover one externally interrupted Table-C validation item without data loss."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_matrix_validation_queue as queue_runner  # noqa: E402


DEFAULT_QUEUE_DIR = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_c_matrix_validation_v1"
)
DEFAULT_ARCHIVE_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/recovery/table_c_matrix_validation_v1/"
    "L3_seed17_attempt000"
)
EXPECTED_QUEUE_ID = "68360aac-cf82-4a9e-a357-04a0c5ccd3b3"
EXPECTED_PLAN_SHA256 = (
    "b238f24b0090323c6a52294592c276b7706662ff9c4b965e84dc775034632078"
)
EXPECTED_RUN_ID = "L3:17"
EXPECTED_INDEX = 3
EXPECTED_INTERRUPTED_REVISION = 968
RECOVERY_EVENT = "matrix_validation_external_interruption_recovered"
RECEIPT_SCHEMA = "pivot.stageb.matrix_validation_interruption_recovery_receipt/v1"
COMMIT_SCHEMA = "pivot.stageb.matrix_validation_interruption_recovery_commit/v1"
RESULT_SCHEMA = "pivot.stageb.matrix_validation_interruption_recovery_result/v1"
VERIFICATION_SCHEMA = (
    "pivot.stageb.matrix_validation_interruption_recovery_verification/v1"
)
SEMANTIC_REPLAY_PROOF = {
    "original_child_identity_proven_gone": True,
    "original_child_process_group_proven_empty": True,
    "original_supervisor_identity_proven_gone": True,
    "original_supervisor_process_group_proven_empty": True,
    "nonterminal_launch_archived": True,
    "partial_output_archived_not_deleted": True,
    "partial_queue_job_archived_not_deleted": True,
    "immutable_plan_and_queue_identity_retained": True,
    "same_item_reopened_from_fresh_paths": True,
    "completed_prefix_retained": True,
    "queue_owned_gpu_lease_retained": True,
}


class RecoveryError(RuntimeError):
    """The interrupted evaluation is not eligible for the narrow retry policy."""


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RecoveryError(f"evidence is not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
    }


def _verify_file_record(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise RecoveryError(f"{label} file record is missing")
    path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    if dict(value) != _file_record(path):
        raise RecoveryError(f"{label} file identity drifted")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


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


def _inventory_view(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file_count": value.get("file_count"),
        "files": value.get("files"),
        "inventory_sha256": value.get("inventory_sha256"),
    }


def _verify_inventory(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _tree_inventory(root)
    if _inventory_view(observed) != _inventory_view(expected):
        raise RecoveryError(f"archived evidence inventory drifted: {root}")
    return observed


def _proc_stat(pid: int) -> dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (FileNotFoundError, ProcessLookupError):
        return None
    right = raw.rfind(")")
    if right < 0:
        raise RecoveryError(f"cannot parse /proc identity for PID {pid}")
    fields = raw[right + 2 :].split()
    if len(fields) < 20:
        raise RecoveryError(f"truncated /proc identity for PID {pid}")
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "process_group": int(fields[2]),
        "session": int(fields[3]),
        "start_time_ticks": int(fields[19]),
        "boot_id": boot_id,
    }


def _require_original_process_gone(
    pid: Any, identity: Any, *, label: str
) -> dict[str, Any]:
    if type(pid) is not int or pid <= 1 or not isinstance(identity, Mapping):
        raise RecoveryError(f"{label} process identity is invalid")
    expected_boot = identity.get("boot_id")
    expected_start = identity.get("start_time_ticks")
    if not isinstance(expected_boot, str) or type(expected_start) is not int:
        raise RecoveryError(f"{label} process identity is incomplete")
    observed = _proc_stat(pid)
    if observed is not None and (
        observed["boot_id"] == expected_boot
        and observed["start_time_ticks"] == expected_start
        and observed["state"] != "Z"
    ):
        raise RecoveryError(f"{label} process is still running")
    return {
        "pid": pid,
        "expected_identity": dict(identity),
        "observed_identity": observed,
        "original_identity_gone": True,
    }


def _process_group_members(process_group: int, session: int) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            value = _proc_stat(int(entry.name))
        except (OSError, RecoveryError):
            continue
        if value is None or value["state"] == "Z":
            continue
        if value["process_group"] == process_group or value["session"] == session:
            members.append(value)
    return sorted(members, key=lambda value: value["pid"])


def _require_process_group_empty(pid: int, *, label: str) -> dict[str, Any]:
    members = _process_group_members(pid, pid)
    if members:
        raise RecoveryError(f"{label} process group/session still has live members")
    return {"process_group": pid, "session": pid, "live_members": []}


def _receipt_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return _canonical_sha({"schema": RECEIPT_SCHEMA, "receipt": payload})


def _expected_work_dir(queue_dir: Path, index: int, run_id: str) -> Path:
    return (queue_dir / "jobs" / f"{index:03d}-{run_id.replace(':', '_')}").resolve(
        strict=False
    )


def inspect_interruption(
    queue_dir: Path,
    *,
    run_id: str,
    expected_queue_id: str,
    expected_plan_sha256: str,
    expected_revision: int,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    try:
        queue = queue_runner.load_queue(queue_dir)
    except Exception as exc:
        raise RecoveryError(f"cannot verify interrupted queue: {exc}") from exc
    plan = queue.get("plan")
    if (
        not isinstance(plan, Mapping)
        or plan.get("queue_id") != expected_queue_id
        or queue.get("plan_sha256") != expected_plan_sha256
        or queue.get("status") != "running"
        or queue.get("revision") != expected_revision
    ):
        raise RecoveryError("queue identity/state differs from the approved interruption")
    items = queue.get("items")
    if not isinstance(items, list) or len(items) != 33:
        raise RecoveryError("interrupted queue item surface is invalid")
    item = items[EXPECTED_INDEX]
    if (
        run_id != EXPECTED_RUN_ID
        or item.get("index") != EXPECTED_INDEX
        or item.get("run_id") != run_id
        or item.get("status") != "launched"
        or any(candidate.get("status") != "completed" for candidate in items[:EXPECTED_INDEX])
        or any(candidate.get("status") != "pending" for candidate in items[EXPECTED_INDEX + 1 :])
    ):
        raise RecoveryError("interrupted item is not the exact first unfinished item")

    child_pid = item.get("child_pid")
    child = _require_original_process_gone(
        child_pid, item.get("child_process_identity"), label="evaluation child"
    )
    child_group = _require_process_group_empty(child_pid, label="evaluation child")

    supervisor_path = (queue_dir / "supervisors/current.json").resolve(strict=True)
    supervisor = _read_json(supervisor_path, label="detached supervisor")
    if supervisor.get("queue_id") != expected_queue_id:
        raise RecoveryError("detached supervisor belongs to another queue")
    supervisor_pid = supervisor.get("pid")
    supervisor_process = _require_original_process_gone(
        supervisor_pid,
        supervisor.get("process_identity"),
        label="detached supervisor",
    )
    supervisor_group = _require_process_group_empty(
        supervisor_pid, label="detached supervisor"
    )
    supervisor_job = Path(str(supervisor.get("job_dir", ""))).resolve(strict=True)
    if supervisor_job.parent != (queue_dir / "supervisors").resolve(strict=True):
        raise RecoveryError("detached supervisor evidence is outside the queue")

    evaluation_root = Path(str(item.get("evaluation_root", ""))).resolve(strict=True)
    expected_root = Path(str(plan["items"][EXPECTED_INDEX]["evaluation_root"])).resolve(
        strict=True
    )
    work_dir = Path(str(item.get("work_dir", ""))).resolve(strict=True)
    if evaluation_root != expected_root or work_dir != _expected_work_dir(
        queue_dir, EXPECTED_INDEX, run_id
    ):
        raise RecoveryError("interrupted evaluation paths are not canonical")
    launch = _read_json(evaluation_root / "launch_manifest.json", label="launch")
    if (
        launch.get("schema") != "pivot.stageb.paper_evaluation_launch/v1"
        or launch.get("status") != "running"
        or (evaluation_root / "postflight.json").exists()
        or (evaluation_root / "input_rehash.json").exists()
    ):
        raise RecoveryError("interrupted evaluation has terminal or postflight evidence")

    lease_path = Path(str(plan.get("lease_path", ""))).resolve(strict=True)
    lease = _read_json(lease_path, label="GPU lease")
    if (
        lease.get("status") != "owned"
        or lease.get("queue_id") != expected_queue_id
        or lease.get("plan_sha256") != expected_plan_sha256
    ):
        raise RecoveryError("interrupted queue no longer owns its exact GPU lease")
    return {
        "schema": "pivot.stageb.matrix_validation_interruption_inspection/v1",
        "status": "eligible",
        "inspected_at_utc": _utc_now(),
        "queue_dir": str(queue_dir),
        "queue_id": expected_queue_id,
        "plan_sha256": expected_plan_sha256,
        "interrupted_revision": expected_revision,
        "run_id": run_id,
        "index": EXPECTED_INDEX,
        "item": dict(item),
        "evaluation_root": str(evaluation_root),
        "work_dir": str(work_dir),
        "evaluation_inventory": _tree_inventory(evaluation_root),
        "work_inventory": _tree_inventory(work_dir),
        "supervisor": {
            "current": _file_record(supervisor_path),
            "job_dir": str(supervisor_job),
            "job_inventory": _tree_inventory(supervisor_job),
        },
        "lease": _file_record(lease_path),
        "proof": {
            "child": child,
            "child_group": child_group,
            "supervisor": supervisor_process,
            "supervisor_group": supervisor_group,
            "launch_status": "running",
            "postflight_present": False,
            "input_rehash_present": False,
            "completed_prefix_count": EXPECTED_INDEX,
        },
    }


def _save_reopened_queue(
    queue: MutableMapping[str, Any], queue_dir: Path, *, expected_revision: int
) -> None:
    queue["revision"] = expected_revision + 1
    queue["updated_at_utc"] = _utc_now()
    queue_runner._validate_queue(queue, queue_dir)
    _write_json_atomic(queue_dir / "queue.json", queue)


def apply_recovery(
    queue_dir: Path,
    *,
    run_id: str,
    archive_root: Path,
    expected_queue_id: str,
    expected_plan_sha256: str,
    expected_revision: int,
) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    archive_root = archive_root.expanduser().resolve(strict=False)
    if archive_root.exists():
        raise RecoveryError(f"archive root must be fresh: {archive_root}")
    with queue_runner._exclusive_lock(queue_dir / "supervisor.lock"):
        inspection = inspect_interruption(
            queue_dir,
            run_id=run_id,
            expected_queue_id=expected_queue_id,
            expected_plan_sha256=expected_plan_sha256,
            expected_revision=expected_revision,
        )
        queue_path = queue_dir / "queue.json"
        queue_before = queue_path.read_bytes()
        if hashlib.sha256(queue_before).hexdigest() != _sha256_file(queue_path):
            raise RecoveryError("queue changed while its snapshot was read")

        archive_root.mkdir(parents=True, exist_ok=False)
        _fsync_directory(archive_root.parent)
        queue_snapshot = archive_root / "queue_before.json"
        tool_snapshot = archive_root / "recovery_tool.py"
        controller_snapshot = archive_root / "matrix_validation_controller.py"
        supervisor_snapshot = archive_root / "supervisor_current.json"
        _write_bytes_exclusive(queue_snapshot, queue_before)
        _write_bytes_exclusive(tool_snapshot, Path(__file__).read_bytes())
        controller_path = Path(queue_runner.__file__).resolve(strict=True)
        _write_bytes_exclusive(controller_snapshot, controller_path.read_bytes())
        supervisor_source = _verify_file_record(
            inspection["supervisor"]["current"], label="supervisor current"
        )
        _write_bytes_exclusive(supervisor_snapshot, supervisor_source.read_bytes())
        archived_supervisor = archive_root / "supervisor_job"
        shutil.copytree(inspection["supervisor"]["job_dir"], archived_supervisor)
        archived_supervisor_inventory = _verify_inventory(
            archived_supervisor, inspection["supervisor"]["job_inventory"]
        )

        archived_evaluation = archive_root / "partial_evaluation_output"
        archived_work = archive_root / "partial_queue_job"
        intent = {
            "schema": "pivot.stageb.matrix_validation_interruption_recovery_intent/v1",
            "created_at_utc": _utc_now(),
            "queue_id": expected_queue_id,
            "plan_sha256": expected_plan_sha256,
            "interrupted_revision": expected_revision,
            "run_id": run_id,
            "index": EXPECTED_INDEX,
            "queue_before": _file_record(queue_snapshot),
            "recovery_tool": _file_record(tool_snapshot),
            "controller": _file_record(controller_snapshot),
            "supervisor_current": _file_record(supervisor_snapshot),
            "supervisor_job": archived_supervisor_inventory,
            "moves": [
                {
                    "source": inspection["evaluation_root"],
                    "destination": str(archived_evaluation),
                    "inventory": inspection["evaluation_inventory"],
                },
                {
                    "source": inspection["work_dir"],
                    "destination": str(archived_work),
                    "inventory": inspection["work_inventory"],
                },
            ],
        }
        intent_path = archive_root / "recovery_intent.json"
        _write_json_exclusive(intent_path, intent)

        source_evaluation = Path(inspection["evaluation_root"])
        source_work = Path(inspection["work_dir"])
        moved_evaluation = False
        moved_work = False
        try:
            os.rename(source_evaluation, archived_evaluation)
            moved_evaluation = True
            os.rename(source_work, archived_work)
            moved_work = True
            _fsync_directory(source_evaluation.parent)
            _fsync_directory(source_work.parent)
            _fsync_directory(archive_root)
            archived_evaluation_inventory = _verify_inventory(
                archived_evaluation, inspection["evaluation_inventory"]
            )
            archived_work_inventory = _verify_inventory(
                archived_work, inspection["work_inventory"]
            )
        except BaseException:
            if moved_work and not source_work.exists():
                os.rename(archived_work, source_work)
            if moved_evaluation and not source_evaluation.exists():
                os.rename(archived_evaluation, source_evaluation)
            raise

        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "status": "archived_and_eligible_for_exact_fresh_retry",
            "created_at_utc": _utc_now(),
            "interruption_class": "external_process_group_disappearance_during_nonterminal_evaluation",
            "queue": {
                "queue_dir": str(queue_dir),
                "queue_id": expected_queue_id,
                "plan_sha256": expected_plan_sha256,
                "interrupted_revision": expected_revision,
                "expected_reopened_revision": expected_revision + 1,
            },
            "run_id": run_id,
            "index": EXPECTED_INDEX,
            "interrupted_item": inspection["item"],
            "proof": inspection["proof"],
            "lease": inspection["lease"],
            "archive": {
                "root": str(archive_root),
                "queue_before": _file_record(queue_snapshot),
                "intent": _file_record(intent_path),
                "recovery_tool": _file_record(tool_snapshot),
                "controller": _file_record(controller_snapshot),
                "supervisor_current": _file_record(supervisor_snapshot),
                "supervisor_job": archived_supervisor_inventory,
                "partial_evaluation_output": archived_evaluation_inventory,
                "partial_queue_job": archived_work_inventory,
            },
            "recovery_contract": {
                "same_immutable_plan_and_queue": True,
                "same_run_checkpoint_config_and_input_contract": True,
                "no_ref_test_or_strict_access": True,
                "nonterminal_partial_output_never_adopted": True,
                "partial_evidence_renamed_not_deleted": True,
                "fresh_canonical_output_and_job_required": True,
                "single_retry_of_exact_interrupted_item": True,
                "retained_queue_owned_lease": True,
            },
        }
        receipt["receipt_sha256"] = _receipt_digest(receipt)
        receipt_path = archive_root / "recovery_receipt.json"
        _write_json_exclusive(receipt_path, receipt)
        receipt_record = _file_record(receipt_path)

        queue = queue_runner.load_queue(queue_dir)
        if queue.get("revision") != expected_revision:
            raise RecoveryError("queue changed after interruption evidence was archived")
        old_item = queue["items"][EXPECTED_INDEX]
        if old_item.get("status") != "launched" or old_item.get("run_id") != run_id:
            raise RecoveryError("interrupted item changed before queue reopen")
        queue["items"][EXPECTED_INDEX] = {
            "evaluation_root": old_item["evaluation_root"],
            "index": EXPECTED_INDEX,
            "run_id": run_id,
            "status": "pending",
            "evaluation_recovery_receipts": [receipt_record],
        }
        queue["events"].append(
            {
                "at_utc": _utc_now(),
                "event": RECOVERY_EVENT,
                "index": EXPECTED_INDEX,
                "run_id": run_id,
                "interrupted_revision": expected_revision,
                "interruption_class": receipt["interruption_class"],
                "receipt": receipt_record,
            }
        )
        _save_reopened_queue(queue, queue_dir, expected_revision=expected_revision)
        reopened = queue_runner.load_queue(queue_dir)
        reopened_item = reopened["items"][EXPECTED_INDEX]
        if (
            reopened.get("status") != "running"
            or reopened.get("revision") != expected_revision + 1
            or reopened_item.get("status") != "pending"
            or reopened_item.get("evaluation_recovery_receipts") != [receipt_record]
            or source_evaluation.exists()
            or source_work.exists()
        ):
            raise RecoveryError("queue reopen postconditions failed")
        queue_after = archive_root / "queue_after_reopen.json"
        _write_bytes_exclusive(queue_after, queue_path.read_bytes())
        commit = {
            "schema": COMMIT_SCHEMA,
            "status": "committed",
            "committed_at_utc": _utc_now(),
            "queue_after_reopen": _file_record(queue_after),
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
            "schema": RESULT_SCHEMA,
            "status": "reopened",
            "queue_id": expected_queue_id,
            "plan_sha256": expected_plan_sha256,
            "run_id": run_id,
            "revision": reopened["revision"],
            "receipt": receipt_record,
            "commit": _file_record(commit_path),
            "archive_root": str(archive_root),
        }


def verify_recovery(queue_dir: Path, receipt_path: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path, label="recovery receipt")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != _receipt_digest(receipt)
        or receipt_path.parent != Path(str(receipt.get("archive", {}).get("root", ""))).resolve(
            strict=True
        )
    ):
        raise RecoveryError("recovery receipt envelope or self digest drifted")
    archive = receipt["archive"]
    for key in (
        "queue_before",
        "intent",
        "recovery_tool",
        "controller",
        "supervisor_current",
    ):
        _verify_file_record(archive.get(key), label=key)
    for key in (
        "supervisor_job",
        "partial_evaluation_output",
        "partial_queue_job",
    ):
        value = archive.get(key)
        if not isinstance(value, Mapping):
            raise RecoveryError(f"{key} inventory is missing")
        _verify_inventory(Path(str(value.get("root", ""))), value)
    _verify_file_record(receipt.get("lease"), label="retained GPU lease")

    queue_before = _read_json(
        _verify_file_record(archive["queue_before"], label="queue before"),
        label="archived queue before",
    )
    binding = receipt.get("queue")
    if not isinstance(binding, Mapping):
        raise RecoveryError("recovery receipt queue binding is missing")
    if (
        queue_before.get("revision") != binding.get("interrupted_revision")
        or queue_before.get("status") != "running"
        or queue_before.get("items", [])[receipt["index"]].get("status") != "launched"
        or queue_before.get("plan", {}).get("queue_id") != binding.get("queue_id")
        or queue_before.get("plan_sha256") != binding.get("plan_sha256")
    ):
        raise RecoveryError("archived interrupted queue state drifted")

    commit_path = receipt_path.parent / "recovery_commit.json"
    commit = _read_json(commit_path, label="recovery commit")
    receipt_record = _file_record(receipt_path)
    if (
        commit.get("schema") != COMMIT_SCHEMA
        or commit.get("status") != "committed"
        or commit.get("queue_id") != binding.get("queue_id")
        or commit.get("plan_sha256") != binding.get("plan_sha256")
        or commit.get("revision") != binding.get("expected_reopened_revision")
        or commit.get("run_id") != receipt.get("run_id")
        or commit.get("item_status") != "pending"
        or commit.get("receipt") != receipt_record
    ):
        raise RecoveryError("recovery commit binding drifted")
    queue_after = _read_json(
        _verify_file_record(commit.get("queue_after_reopen"), label="queue after reopen"),
        label="archived queue after reopen",
    )
    after_item = queue_after["items"][receipt["index"]]
    recovery_events = [
        event
        for event in queue_after.get("events", [])
        if isinstance(event, Mapping) and event.get("event") == RECOVERY_EVENT
    ]
    if (
        queue_after.get("revision") != binding.get("expected_reopened_revision")
        or queue_after.get("status") != "running"
        or after_item.get("status") != "pending"
        or after_item.get("evaluation_recovery_receipts") != [receipt_record]
        or len(recovery_events) != 1
        or recovery_events[0].get("receipt") != receipt_record
    ):
        raise RecoveryError("archived reopened queue state drifted")

    try:
        current = queue_runner.load_queue(queue_dir)
    except Exception as exc:
        raise RecoveryError(f"cannot replay current recovered queue: {exc}") from exc
    current_item = current["items"][receipt["index"]]
    current_events = [
        event
        for event in current.get("events", [])
        if isinstance(event, Mapping) and event.get("event") == RECOVERY_EVENT
    ]
    if (
        current.get("plan", {}).get("queue_id") != binding.get("queue_id")
        or current.get("plan_sha256") != binding.get("plan_sha256")
        or current_item.get("evaluation_recovery_receipts") != [receipt_record]
        or len(current_events) != 1
        or current_events[0].get("receipt") != receipt_record
    ):
        raise RecoveryError("current queue no longer binds the recovery receipt")
    if receipt.get("recovery_contract") != {
        "same_immutable_plan_and_queue": True,
        "same_run_checkpoint_config_and_input_contract": True,
        "no_ref_test_or_strict_access": True,
        "nonterminal_partial_output_never_adopted": True,
        "partial_evidence_renamed_not_deleted": True,
        "fresh_canonical_output_and_job_required": True,
        "single_retry_of_exact_interrupted_item": True,
        "retained_queue_owned_lease": True,
    }:
        raise RecoveryError("recovery contract is not the narrow approved policy")
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "queue_id": binding["queue_id"],
        "plan_sha256": binding["plan_sha256"],
        "run_id": receipt["run_id"],
        "receipt_sha256": receipt["receipt_sha256"],
        "semantic_replay": dict(SEMANTIC_REPLAY_PROOF),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("inspect")
    subparsers.add_parser("apply")
    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "receipt",
        nargs="?",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT / "recovery_receipt.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "inspect":
            result = inspect_interruption(
                DEFAULT_QUEUE_DIR,
                run_id=EXPECTED_RUN_ID,
                expected_queue_id=EXPECTED_QUEUE_ID,
                expected_plan_sha256=EXPECTED_PLAN_SHA256,
                expected_revision=EXPECTED_INTERRUPTED_REVISION,
            )
        elif args.mode == "apply":
            result = apply_recovery(
                DEFAULT_QUEUE_DIR,
                run_id=EXPECTED_RUN_ID,
                archive_root=DEFAULT_ARCHIVE_ROOT,
                expected_queue_id=EXPECTED_QUEUE_ID,
                expected_plan_sha256=EXPECTED_PLAN_SHA256,
                expected_revision=EXPECTED_INTERRUPTED_REVISION,
            )
        else:
            result = verify_recovery(DEFAULT_QUEUE_DIR, args.receipt)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
        RecoveryError,
        queue_runner.MatrixQueueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
