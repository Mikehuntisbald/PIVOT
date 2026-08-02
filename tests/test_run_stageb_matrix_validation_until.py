from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_stageb_matrix_validation_until as boundary


QUEUE_ID = "queue-id"
PLAN_SHA256 = "1" * 64


def _write_queue(
    queue_dir: Path,
    *,
    statuses: tuple[str, str, str] = ("completed", "pending", "pending"),
    queue_id: str = QUEUE_ID,
    plan_sha256: str = PLAN_SHA256,
    status: str = "running",
) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "queue.json").write_text(
        json.dumps(
            {
                "status": status,
                "revision": 7,
                "plan_sha256": plan_sha256,
                "plan": {"queue_id": queue_id},
                "items": [
                    {"run_id": "L9:17", "status": statuses[0]},
                    {"run_id": "L10:17", "status": statuses[1]},
                    {"run_id": "L0:42", "status": statuses[2]},
                ],
            }
        ),
        encoding="utf-8",
    )


def _controller(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "execution" / "tools" / "controller.py"
    path.parent.mkdir(parents=True)
    path.write_text("# immutable test controller\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _drive_args(tmp_path: Path) -> dict[str, object]:
    controller, digest = _controller(tmp_path)
    return {
        "controller": controller,
        "controller_python": Path(sys.executable),
        "expected_controller_sha256": digest,
        "expected_queue_id": QUEUE_ID,
        "expected_plan_sha256": PLAN_SHA256,
        "stop_after_run_id": "L10:17",
        "expected_next_run_id": "L0:42",
        "poll_seconds": 0.05,
    }


def test_check_only_binds_exact_boundary_without_mutation(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    _write_queue(queue_dir)

    report = boundary.drive(queue_dir, check_only=True, **_drive_args(tmp_path))

    assert report["status"] == "ready"
    assert report["boundary_status"] == "pending"
    assert report["next_run_id"] == "L0:42"
    assert report["mutated"] is False
    assert json.loads((queue_dir / "queue.json").read_text())["revision"] == 7


def test_drive_stops_immediately_after_boundary_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_dir = tmp_path / "queue"
    _write_queue(queue_dir)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        queue_path = queue_dir / "queue.json"
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        queue["items"][1]["status"] = "completed"
        queue["revision"] += 1
        queue_path.write_text(json.dumps(queue), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(boundary.subprocess, "run", fake_run)
    monkeypatch.setattr(boundary.time, "sleep", lambda _: None)

    report = boundary.drive(queue_dir, check_only=False, **_drive_args(tmp_path))

    queue = json.loads((queue_dir / "queue.json").read_text(encoding="utf-8"))
    assert report["status"] == "paused_after_boundary"
    assert report["reconcile_count"] == 1
    assert len(calls) == 1
    assert queue["items"][1]["status"] == "completed"
    assert queue["items"][2]["status"] == "pending"


def test_refuses_already_crossed_boundary(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    _write_queue(queue_dir, statuses=("completed", "completed", "reserved"))

    with pytest.raises(boundary.BoundaryDriverError, match="already crossed"):
        boundary.drive(queue_dir, check_only=True, **_drive_args(tmp_path))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"queue_id": "other"}, "queue ID drift"),
        ({"plan_sha256": "2" * 64}, "plan hash drift"),
        ({"status": "failed"}, "queue is failed"),
    ],
)
def test_refuses_queue_identity_or_terminal_drift(
    tmp_path: Path, override: dict[str, str], message: str
) -> None:
    queue_dir = tmp_path / "queue"
    _write_queue(queue_dir, **override)

    with pytest.raises(boundary.BoundaryDriverError, match=message):
        boundary.drive(queue_dir, check_only=True, **_drive_args(tmp_path))


def test_refuses_controller_hash_drift(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    _write_queue(queue_dir)
    args = _drive_args(tmp_path)
    args["expected_controller_sha256"] = "0" * 64

    with pytest.raises(boundary.BoundaryDriverError, match="controller hash drift"):
        boundary.drive(queue_dir, check_only=True, **args)


def test_stops_on_reconcile_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_dir = tmp_path / "queue"
    _write_queue(queue_dir)

    monkeypatch.setattr(
        boundary.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(
            command, 3, "", "reconcile failed"
        ),
    )

    with pytest.raises(boundary.BoundaryDriverError, match="exit 3"):
        boundary.drive(queue_dir, check_only=False, **_drive_args(tmp_path))
