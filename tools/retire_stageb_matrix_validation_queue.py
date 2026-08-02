#!/usr/bin/env python3
"""Retire an interrupted matrix-validation queue without deleting its evidence.

This tool is intentionally narrow.  It releases only a lease owned by the
exact queue being retired, and only after a host reboot proves that both the
recorded evaluator and supervisor process identities are gone.  The immutable
plan, mutable queue ledger, completed evaluations, and partial evaluation are
left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_serial_matrix_queue as serial_queue  # noqa: E402


DEFAULT_QUEUE_DIR = (
    REPO_ROOT / "outputs/paper_cvpr_v1/queues/table_c_matrix_validation_v1"
)
DEFAULT_RETIREMENT_DIR = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/retirements/table_c_matrix_validation_v1"
    / "after_l2_reboot_20260720"
)
EXPECTED_QUEUE_ID = "68360aac-cf82-4a9e-a357-04a0c5ccd3b3"
EXPECTED_PLAN_SHA256 = (
    "b238f24b0090323c6a52294592c276b7706662ff9c4b965e84dc775034632078"
)
EXPECTED_ACTIVE_RUN_ID = "L3:17"
EXPECTED_COMPLETED_PREFIX = ("L0:17", "L1:17", "L2:17")
INTENT_SCHEMA = "pivot.stageb.matrix_validation_retirement_intent/v1"
RECEIPT_SCHEMA = "pivot.stageb.matrix_validation_retirement_receipt/v1"


class RetirementError(RuntimeError):
    """The queue is not eligible for the narrow retirement operation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    metadata = path.stat()
    if not path.is_file() or path.is_symlink():
        raise RetirementError(f"evidence is not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetirementError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RetirementError(f"{label} must be a JSON object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with path.open("x", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RetirementError(f"refusing to overwrite retirement evidence: {path}") from exc


def _current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _require_rebooted_identity(identity: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise RetirementError(f"{label} process identity is missing")
    recorded_boot = identity.get("boot_id")
    recorded_start = identity.get("start_time_ticks")
    if not isinstance(recorded_boot, str) or type(recorded_start) is not int:
        raise RetirementError(f"{label} process identity is incomplete")
    current_boot = _current_boot_id()
    if current_boot == recorded_boot:
        raise RetirementError(
            f"{label} belongs to the current boot; retirement requires a reboot boundary"
        )
    return {
        "recorded": dict(identity),
        "current_boot_id": current_boot,
        "gone_by_reboot_boundary": True,
    }


def inspect_retirement(queue_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    queue_path = queue_dir / "queue.json"
    queue = _read_json(queue_path, label="matrix queue")
    plan = queue.get("plan")
    items = queue.get("items")
    if not isinstance(plan, Mapping) or not isinstance(items, list):
        raise RetirementError("matrix queue plan/items are invalid")
    if (
        plan.get("queue_id") != EXPECTED_QUEUE_ID
        or queue.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or queue.get("status") != "running"
    ):
        raise RetirementError("matrix queue identity or state is not the approved queue")

    completed = [
        str(item.get("run_id"))
        for item in items
        if isinstance(item, Mapping) and item.get("status") == "completed"
    ]
    active = [
        item
        for item in items
        if isinstance(item, Mapping)
        and item.get("status") not in {"completed", "pending"}
    ]
    if tuple(completed) != EXPECTED_COMPLETED_PREFIX or len(active) != 1:
        raise RetirementError("queue is not stopped after the approved completed prefix")
    item = active[0]
    if item.get("run_id") != EXPECTED_ACTIVE_RUN_ID or item.get("status") != "launched":
        raise RetirementError("the sole stale item is not the approved L3 launch")

    supervisor_path = queue_dir / "supervisors/current.json"
    supervisor = _read_json(supervisor_path, label="matrix supervisor")
    child_proof = _require_rebooted_identity(
        item.get("child_process_identity"), label="evaluation child"
    )
    supervisor_proof = _require_rebooted_identity(
        supervisor.get("process_identity"), label="matrix supervisor"
    )

    lease_path = Path(str(plan.get("lease_path", ""))).expanduser().resolve(strict=True)
    lease = _read_json(lease_path, label="GPU lease")
    mismatches = serial_queue._lease_identity_mismatches(queue, lease)
    if mismatches:
        raise RetirementError(f"GPU lease is not owned by this queue: {mismatches}")

    evaluation_root = Path(str(item.get("evaluation_root", ""))).resolve(strict=True)
    work_dir = Path(str(item.get("work_dir", ""))).resolve(strict=True)
    launch_path = evaluation_root / "launch_manifest.json"
    launch = _read_json(launch_path, label="partial evaluation launch")
    if launch.get("status") != "running" or (evaluation_root / "postflight.json").exists():
        raise RetirementError("stale evaluation unexpectedly has terminal evidence")

    return {
        "schema": "pivot.stageb.matrix_validation_retirement_inspection/v1",
        "status": "eligible",
        "inspected_at_utc": _utc_now(),
        "reason": "user_redirected_compute_after_l2_and_host_rebooted",
        "queue_dir": str(queue_dir),
        "queue_id": EXPECTED_QUEUE_ID,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "queue_revision": queue.get("revision"),
        "completed_prefix": list(EXPECTED_COMPLETED_PREFIX),
        "stale_item": {
            "run_id": item.get("run_id"),
            "index": item.get("index"),
            "status": item.get("status"),
            "evaluation_root": str(evaluation_root),
            "work_dir": str(work_dir),
        },
        "proof": {
            "child": child_proof,
            "supervisor": supervisor_proof,
            "partial_launch_status": launch.get("status"),
            "partial_postflight_present": False,
            "queue_evidence_retained": True,
            "partial_evaluation_retained": True,
        },
        "evidence": {
            "queue": _file_record(queue_path),
            "lease": _file_record(lease_path),
            "supervisor": _file_record(supervisor_path),
            "partial_launch": _file_record(launch_path),
        },
    }


def apply_retirement(queue_dir: Path, retirement_dir: Path) -> dict[str, Any]:
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    retirement_dir = retirement_dir.expanduser().resolve(strict=False)
    intent_path = retirement_dir / "intent.json"
    receipt_path = retirement_dir / "receipt.json"
    if retirement_dir.exists() and any(retirement_dir.iterdir()):
        raise RetirementError(f"retirement directory already exists: {retirement_dir}")

    with serial_queue._exclusive_file_lock(
        queue_dir / "supervisor.lock",
        busy_message=f"matrix supervisor lock is busy: {queue_dir}",
    ):
        inspection = inspect_retirement(queue_dir)
        intent = {
            "schema": INTENT_SCHEMA,
            "created_at_utc": _utc_now(),
            "operation": "release_owned_gpu_lease_only",
            "inspection": inspection,
        }
        _write_json_exclusive(intent_path, intent)

        queue = _read_json(queue_dir / "queue.json", label="matrix queue")
        if _file_record(queue_dir / "queue.json") != inspection["evidence"]["queue"]:
            raise RetirementError("matrix queue changed after retirement inspection")
        lease_path = Path(str(queue["plan"]["lease_path"])).resolve(strict=True)
        if _file_record(lease_path) != inspection["evidence"]["lease"]:
            raise RetirementError("GPU lease changed after retirement inspection")
        serial_queue._clear_owned_lease(queue)
        if lease_path.exists():
            raise RetirementError("owned GPU lease still exists after retirement")

        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "retired",
            "retired_at_utc": _utc_now(),
            "operation": "released_owned_gpu_lease_preserved_all_queue_evidence",
            "intent": _file_record(intent_path),
            "queue_after": _file_record(queue_dir / "queue.json"),
            "queue_unchanged": (
                _file_record(queue_dir / "queue.json") == inspection["evidence"]["queue"]
            ),
            "lease_path": str(lease_path),
            "lease_present_after": False,
            "resume_policy": "retired_queue_must_not_be_resumed_without_explicit_reauthorization",
        }
        if not receipt["queue_unchanged"]:
            raise RetirementError("retirement modified the matrix queue ledger")
        _write_json_exclusive(receipt_path, receipt)
    return receipt


def verify_retirement(retirement_dir: Path) -> dict[str, Any]:
    retirement_dir = retirement_dir.expanduser().resolve(strict=True)
    intent_path = retirement_dir / "intent.json"
    receipt_path = retirement_dir / "receipt.json"
    intent = _read_json(intent_path, label="retirement intent")
    receipt = _read_json(receipt_path, label="retirement receipt")
    if intent.get("schema") != INTENT_SCHEMA or receipt.get("schema") != RECEIPT_SCHEMA:
        raise RetirementError("retirement evidence schema is invalid")
    if receipt.get("intent") != _file_record(intent_path):
        raise RetirementError("retirement intent identity drifted")
    inspection = intent.get("inspection")
    if not isinstance(inspection, Mapping):
        raise RetirementError("retirement inspection is missing")
    queue_record = inspection.get("evidence", {}).get("queue")
    if not isinstance(queue_record, Mapping):
        raise RetirementError("retirement queue evidence is missing")
    queue_path = Path(str(queue_record.get("path", ""))).resolve(strict=True)
    if dict(queue_record) != _file_record(queue_path):
        raise RetirementError("retired queue ledger drifted")
    lease_path = Path(str(receipt.get("lease_path", ""))).resolve(strict=False)
    if lease_path.exists() or receipt.get("lease_present_after") is not False:
        raise RetirementError("retired queue GPU lease is present")
    if receipt.get("status") != "retired" or receipt.get("queue_unchanged") is not True:
        raise RetirementError("retirement receipt did not close cleanly")
    return {
        "schema": "pivot.stageb.matrix_validation_retirement_verification/v1",
        "status": "passed",
        "retirement_dir": str(retirement_dir),
        "queue": _file_record(queue_path),
        "lease_path": str(lease_path),
        "lease_present": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    apply_parser.add_argument(
        "--retirement-dir", type=Path, default=DEFAULT_RETIREMENT_DIR
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--retirement-dir", type=Path, default=DEFAULT_RETIREMENT_DIR
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "inspect":
            result = inspect_retirement(args.queue_dir)
        elif args.mode == "apply":
            result = apply_retirement(args.queue_dir, args.retirement_dir)
        else:
            result = verify_retirement(args.retirement_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        ValueError,
        RetirementError,
        serial_queue.QueueContractError,
    ) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
