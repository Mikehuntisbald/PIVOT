from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import recover_stageb_matrix_validation_interruption as recovery


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RecoveryFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.queue_dir = root / "queue"
        self.output_root = root / "evaluations"
        self.lease = root / "lease.json"
        self.archive = root / "archive"
        self.queue_dir.mkdir()
        (self.queue_dir / "jobs").mkdir()
        (self.queue_dir / "supervisors").mkdir()
        items = []
        plan_items = []
        for index in range(33):
            run_id = recovery.EXPECTED_RUN_ID if index == recovery.EXPECTED_INDEX else f"R{index}:17"
            evaluation_root = self.output_root / f"item{index}"
            plan_items.append(
                {
                    "index": index,
                    "run_id": run_id,
                    "profile": "matrix_validation",
                    "evaluation_root": str(evaluation_root),
                }
            )
            items.append(
                {
                    "index": index,
                    "run_id": run_id,
                    "evaluation_root": str(evaluation_root),
                    "status": "completed" if index < recovery.EXPECTED_INDEX else "pending",
                }
            )
        self.work_dir = (
            self.queue_dir
            / "jobs"
            / f"{recovery.EXPECTED_INDEX:03d}-{recovery.EXPECTED_RUN_ID.replace(':', '_')}"
        )
        self.work_dir.mkdir()
        (self.work_dir / "evaluation_console.log").write_text("partial\n", encoding="utf-8")
        self.evaluation_root = Path(plan_items[recovery.EXPECTED_INDEX]["evaluation_root"])
        (self.evaluation_root / "validation_calibration/refcoco_eval_inputs").mkdir(
            parents=True
        )
        _write_json(
            self.evaluation_root / "launch_manifest.json",
            {
                "schema": "pivot.stageb.paper_evaluation_launch/v1",
                "status": "running",
            },
        )
        (self.evaluation_root / "validation_calibration_console.log").write_text(
            "batch 50\n", encoding="utf-8"
        )
        (self.evaluation_root / "validation_calibration/refcoco_eval_inputs/ref.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
        absent_pid = 999_999_991
        identity = {
            "available": True,
            "pid": absent_pid,
            "boot_id": "fixture-boot",
            "start_time_ticks": 1,
            "state": "R",
        }
        items[recovery.EXPECTED_INDEX] = {
            **items[recovery.EXPECTED_INDEX],
            "status": "launched",
            "child_pid": absent_pid,
            "child_process_identity": identity,
            "work_dir": str(self.work_dir),
            "console_log": str(self.work_dir / "evaluation_console.log"),
        }
        self.supervisor_job = self.queue_dir / "supervisors/supervisor-000"
        self.supervisor_job.mkdir()
        supervisor = {
            "schema": "pivot.stageb.matrix_validation_supervisor/v1",
            "status": "launched",
            "queue_id": recovery.EXPECTED_QUEUE_ID,
            "pid": absent_pid - 1,
            "process_identity": {**identity, "pid": absent_pid - 1},
            "job_dir": str(self.supervisor_job),
        }
        _write_json(self.queue_dir / "supervisors/current.json", supervisor)
        _write_json(self.supervisor_job / "launch.json", supervisor)
        _write_json(
            self.supervisor_job / "status.json",
            {"schema": supervisor["schema"], "status": "running"},
        )
        (self.supervisor_job / "supervisor.log").write_text("running\n", encoding="utf-8")
        _write_json(
            self.lease,
            {
                "schema": "pivot.stageb.serial_matrix_gpu_lease/v1",
                "status": "owned",
                "queue_id": recovery.EXPECTED_QUEUE_ID,
                "plan_sha256": recovery.EXPECTED_PLAN_SHA256,
            },
        )
        self.queue = {
            "schema": "pivot.stageb.matrix_validation_queue/v1",
            "status": "running",
            "revision": recovery.EXPECTED_INTERRUPTED_REVISION,
            "updated_at_utc": "2026-07-19T10:23:20+00:00",
            "plan_sha256": recovery.EXPECTED_PLAN_SHA256,
            "plan": {
                "queue_id": recovery.EXPECTED_QUEUE_ID,
                "queue_dir": str(self.queue_dir),
                "output_root": str(self.output_root),
                "lease_path": str(self.lease),
                "items": plan_items,
            },
            "items": items,
            "events": [],
            "final_verification": None,
        }
        _write_json(self.queue_dir / "queue.json", self.queue)
        (self.queue_dir / "supervisor.lock").touch()

    def load_queue(self, queue_dir: Path) -> dict[str, object]:
        return json.loads((Path(queue_dir) / "queue.json").read_text(encoding="utf-8"))

    def patches(self):
        return (
            patch.object(recovery.queue_runner, "load_queue", side_effect=self.load_queue),
            patch.object(recovery.queue_runner, "_validate_queue", return_value=None),
        )


class MatrixValidationInterruptionRecoveryTests(unittest.TestCase):
    def test_live_original_process_is_rejected(self) -> None:
        observed = recovery._proc_stat(os.getpid())
        self.assertIsNotNone(observed)
        identity = {
            "boot_id": observed["boot_id"],
            "start_time_ticks": observed["start_time_ticks"],
        }
        with self.assertRaisesRegex(recovery.RecoveryError, "still running"):
            recovery._require_original_process_gone(
                os.getpid(), identity, label="fixture"
            )

    def test_apply_archives_and_reopens_exact_item_then_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            first, second = fixture.patches()
            with first, second:
                result = recovery.apply_recovery(
                    fixture.queue_dir,
                    run_id=recovery.EXPECTED_RUN_ID,
                    archive_root=fixture.archive,
                    expected_queue_id=recovery.EXPECTED_QUEUE_ID,
                    expected_plan_sha256=recovery.EXPECTED_PLAN_SHA256,
                    expected_revision=recovery.EXPECTED_INTERRUPTED_REVISION,
                )
                verified = recovery.verify_recovery(
                    fixture.queue_dir, fixture.archive / "recovery_receipt.json"
                )
            queue = fixture.load_queue(fixture.queue_dir)
            item = queue["items"][recovery.EXPECTED_INDEX]
            self.assertEqual(result["status"], "reopened")
            self.assertEqual(verified["status"], "passed")
            self.assertEqual(verified["semantic_replay"], recovery.SEMANTIC_REPLAY_PROOF)
            self.assertEqual(queue["revision"], recovery.EXPECTED_INTERRUPTED_REVISION + 1)
            self.assertEqual(item["status"], "pending")
            self.assertEqual(len(item["evaluation_recovery_receipts"]), 1)
            self.assertFalse(fixture.evaluation_root.exists())
            self.assertFalse(fixture.work_dir.exists())
            self.assertTrue((fixture.archive / "partial_evaluation_output").is_dir())
            self.assertTrue((fixture.archive / "partial_queue_job").is_dir())
            self.assertTrue(fixture.lease.is_file())

    def test_archived_partial_output_tamper_fails_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            first, second = fixture.patches()
            with first, second:
                recovery.apply_recovery(
                    fixture.queue_dir,
                    run_id=recovery.EXPECTED_RUN_ID,
                    archive_root=fixture.archive,
                    expected_queue_id=recovery.EXPECTED_QUEUE_ID,
                    expected_plan_sha256=recovery.EXPECTED_PLAN_SHA256,
                    expected_revision=recovery.EXPECTED_INTERRUPTED_REVISION,
                )
                target = fixture.archive / "partial_evaluation_output/launch_manifest.json"
                target.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(recovery.RecoveryError, "inventory drifted"):
                    recovery.verify_recovery(
                        fixture.queue_dir, fixture.archive / "recovery_receipt.json"
                    )

    def test_terminal_partial_output_and_existing_archive_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            _write_json(fixture.evaluation_root / "postflight.json", {"status": "passed"})
            first, second = fixture.patches()
            with first, second:
                with self.assertRaisesRegex(recovery.RecoveryError, "terminal"):
                    recovery.inspect_interruption(
                        fixture.queue_dir,
                        run_id=recovery.EXPECTED_RUN_ID,
                        expected_queue_id=recovery.EXPECTED_QUEUE_ID,
                        expected_plan_sha256=recovery.EXPECTED_PLAN_SHA256,
                        expected_revision=recovery.EXPECTED_INTERRUPTED_REVISION,
                    )
            fixture.archive.mkdir()
            with self.assertRaisesRegex(recovery.RecoveryError, "must be fresh"):
                recovery.apply_recovery(
                    fixture.queue_dir,
                    run_id=recovery.EXPECTED_RUN_ID,
                    archive_root=fixture.archive,
                    expected_queue_id=recovery.EXPECTED_QUEUE_ID,
                    expected_plan_sha256=recovery.EXPECTED_PLAN_SHA256,
                    expected_revision=recovery.EXPECTED_INTERRUPTED_REVISION,
                )


if __name__ == "__main__":
    unittest.main()
