#!/usr/bin/env python3
"""Drive an immutable matrix-validation controller to an exact run boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPORT_SCHEMA = "pivot.stageb.matrix_validation_boundary_driver/v1"


class BoundaryDriverError(RuntimeError):
    """Raised when the queue cannot be driven without crossing the boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryDriverError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BoundaryDriverError(f"{label} must be a JSON object: {path}")
    return value


def _queue_state(
    queue_dir: Path,
    *,
    expected_queue_id: str,
    expected_plan_sha256: str,
    stop_after_run_id: str,
    expected_next_run_id: str | None,
) -> dict[str, Any]:
    queue = _read_json(queue_dir / "queue.json", label="queue")
    plan = queue.get("plan")
    if not isinstance(plan, dict):
        raise BoundaryDriverError("queue plan is missing or invalid")
    if plan.get("queue_id") != expected_queue_id:
        raise BoundaryDriverError(
            f"queue ID drift: expected {expected_queue_id!r}, "
            f"observed {plan.get('queue_id')!r}"
        )
    if queue.get("plan_sha256") != expected_plan_sha256:
        raise BoundaryDriverError(
            f"plan hash drift: expected {expected_plan_sha256!r}, "
            f"observed {queue.get('plan_sha256')!r}"
        )

    items = queue.get("items")
    if not isinstance(items, list) or not items:
        raise BoundaryDriverError("queue items are missing or invalid")
    matches = [
        index
        for index, item in enumerate(items)
        if isinstance(item, dict) and item.get("run_id") == stop_after_run_id
    ]
    if len(matches) != 1:
        raise BoundaryDriverError(
            f"expected exactly one boundary run {stop_after_run_id!r}, found {len(matches)}"
        )
    boundary_index = matches[0]
    next_run_id = (
        items[boundary_index + 1].get("run_id")
        if boundary_index + 1 < len(items)
        and isinstance(items[boundary_index + 1], dict)
        else None
    )
    if expected_next_run_id is not None and next_run_id != expected_next_run_id:
        raise BoundaryDriverError(
            f"next-run drift: expected {expected_next_run_id!r}, observed {next_run_id!r}"
        )

    crossed = [
        item.get("run_id") if isinstance(item, dict) else "<invalid-item>"
        for item in items[boundary_index + 1 :]
        if not isinstance(item, dict) or item.get("status") != "pending"
    ]
    if crossed:
        raise BoundaryDriverError(
            "pause boundary was already crossed; non-pending suffix items: "
            + ", ".join(map(str, crossed))
        )
    if queue.get("status") == "failed":
        raise BoundaryDriverError(f"queue is failed: {queue.get('failure')!r}")

    boundary = items[boundary_index]
    if not isinstance(boundary, dict):
        raise BoundaryDriverError("boundary item is invalid")
    return {
        "queue": queue,
        "boundary_index": boundary_index,
        "boundary_status": boundary.get("status"),
        "next_run_id": next_run_id,
        "reached": boundary.get("status") == "completed",
    }


def drive(
    queue_dir: Path,
    *,
    controller: Path,
    controller_python: Path,
    expected_controller_sha256: str,
    expected_queue_id: str,
    expected_plan_sha256: str,
    stop_after_run_id: str,
    expected_next_run_id: str | None,
    poll_seconds: float,
    check_only: bool,
) -> dict[str, Any]:
    if poll_seconds < 0.05:
        raise BoundaryDriverError("poll seconds must be at least 0.05")
    queue_dir = queue_dir.expanduser().resolve(strict=True)
    controller = controller.expanduser().resolve(strict=True)
    controller_python = controller_python.expanduser().resolve(strict=True)
    observed_controller_sha256 = _sha256(controller)
    if observed_controller_sha256 != expected_controller_sha256:
        raise BoundaryDriverError(
            "controller hash drift: "
            f"expected {expected_controller_sha256}, observed {observed_controller_sha256}"
        )

    state = _queue_state(
        queue_dir,
        expected_queue_id=expected_queue_id,
        expected_plan_sha256=expected_plan_sha256,
        stop_after_run_id=stop_after_run_id,
        expected_next_run_id=expected_next_run_id,
    )
    initial_revision = state["queue"].get("revision")
    if check_only or state["reached"]:
        return {
            "schema": REPORT_SCHEMA,
            "status": "boundary_reached" if state["reached"] else "ready",
            "observed_at_utc": _utc_now(),
            "queue_id": expected_queue_id,
            "plan_sha256": expected_plan_sha256,
            "controller": str(controller),
            "controller_sha256": observed_controller_sha256,
            "stop_after_run_id": stop_after_run_id,
            "boundary_index": state["boundary_index"],
            "boundary_status": state["boundary_status"],
            "next_run_id": state["next_run_id"],
            "queue_revision": initial_revision,
            "mutated": False,
        }

    last_observation: tuple[Any, Any] | None = None
    reconcile_count = 0
    while True:
        command = [
            str(controller_python),
            str(controller),
            "reconcile",
            str(queue_dir),
            "--poll-seconds",
            str(poll_seconds),
        ]
        result = subprocess.run(
            command,
            cwd=controller.parent.parent,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        reconcile_count += 1
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-4000:]
            raise BoundaryDriverError(
                f"controller reconcile failed with exit {result.returncode}: {detail}"
            )

        state = _queue_state(
            queue_dir,
            expected_queue_id=expected_queue_id,
            expected_plan_sha256=expected_plan_sha256,
            stop_after_run_id=stop_after_run_id,
            expected_next_run_id=expected_next_run_id,
        )
        queue = state["queue"]
        current = next(
            (
                item
                for item in queue["items"]
                if item.get("status") not in {"completed", "pending"}
            ),
            None,
        )
        observation = (
            queue.get("revision"),
            current.get("run_id") if isinstance(current, dict) else None,
        )
        if observation != last_observation:
            print(
                json.dumps(
                    {
                        "observed_at_utc": _utc_now(),
                        "revision": observation[0],
                        "current_run_id": observation[1],
                        "current_status": (
                            current.get("status") if isinstance(current, dict) else None
                        ),
                        "boundary_status": state["boundary_status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_observation = observation
        if state["reached"]:
            return {
                "schema": REPORT_SCHEMA,
                "status": "paused_after_boundary",
                "observed_at_utc": _utc_now(),
                "queue_id": expected_queue_id,
                "plan_sha256": expected_plan_sha256,
                "controller": str(controller),
                "controller_sha256": observed_controller_sha256,
                "stop_after_run_id": stop_after_run_id,
                "boundary_index": state["boundary_index"],
                "boundary_status": state["boundary_status"],
                "next_run_id": state["next_run_id"],
                "queue_revision": queue.get("revision"),
                "initial_queue_revision": initial_revision,
                "reconcile_count": reconcile_count,
                "mutated": True,
            }
        time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--controller-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--expected-queue-id", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--stop-after-run-id", required=True)
    parser.add_argument("--expected-next-run-id")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = drive(
            args.queue_dir,
            controller=args.controller,
            controller_python=args.controller_python,
            expected_controller_sha256=args.expected_controller_sha256,
            expected_queue_id=args.expected_queue_id,
            expected_plan_sha256=args.expected_plan_sha256,
            stop_after_run_id=args.stop_after_run_id,
            expected_next_run_id=args.expected_next_run_id,
            poll_seconds=args.poll_seconds,
            check_only=args.check_only,
        )
    except (BoundaryDriverError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
