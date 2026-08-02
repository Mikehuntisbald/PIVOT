from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import recover_stageb_serial_matrix_pretraining_failure as recovery
from tools import run_stageb_serial_matrix_queue as queue_runner


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RecoveryFixture:
    run_id = "L2:42"
    queue_id = "queue-id"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.queue_dir = root / "queue"
        self.output_base = root / "training"
        self.output_root = self.output_base / "L2" / "seed42"
        self.orchestration_root = self.queue_dir / "jobs" / "000-L2_42"
        self.job_dir = self.orchestration_root / "job"
        self.lease_path = root / "gpu.lease.json"
        self.queue_dir.mkdir()
        self.output_root.mkdir(parents=True)
        (self.job_dir / "plans" / "L2").mkdir(parents=True)

        plan = {
            "schema": queue_runner.PLAN_SCHEMA,
            "queue_id": self.queue_id,
            "queue_dir": str(self.queue_dir.resolve()),
            "repository_root": str(root.resolve()),
            "runner_python": "/usr/bin/python3",
            "runners": {},
            "runtime_environment": {
                "PIVOT_TOKEN_OUTPUT_ROOT": str(self.output_base.resolve()),
            },
            "gpu_key": "0",
            "lease_root": str(root.resolve()),
            "lease_path": str(self.lease_path.resolve()),
            "items": [{"run_id": self.run_id, "runner": "token"}],
        }
        self.plan_sha = queue_runner._sha256_bytes(
            queue_runner._canonical_json_bytes(plan)
        )
        failure_text = "RuntimeError: nvidia-smi identity query failed: "
        item = {
            "index": 0,
            "run_id": self.run_id,
            "runner": "token",
            "status": "failed",
            "child_pid": 999_999_991,
            "child_process_identity": {"available": True},
            "detach_launcher_pid": 999_999_992,
            "detach_launcher_identity": {"available": True},
            "job_dir": str(self.job_dir.resolve()),
            "orchestration_root": str(self.orchestration_root.resolve()),
            "output_root": str(self.output_root.resolve()),
            "failure_error": recovery.QUEUE_FAILURE_EXACT,
            "failure_phase": "advance",
        }
        queue = {
            "schema": queue_runner.QUEUE_SCHEMA,
            "status": "failed",
            "created_at_utc": "2026-07-18T00:00:00+00:00",
            "updated_at_utc": "2026-07-18T00:00:01+00:00",
            "revision": 7,
            "plan": plan,
            "plan_sha256": self.plan_sha,
            "items": [item],
            "events": [],
            "failure": {
                "index": 0,
                "run_id": self.run_id,
                "phase": "advance",
                "error": item["failure_error"],
                "lease_retained_fail_closed": True,
            },
        }
        _write_json(self.queue_dir / "queue.json", queue)
        _write_json(
            self.lease_path,
            {
                "schema": "pivot.stageb.serial_matrix_gpu_lease/v1",
                "status": "owned",
                "queue_id": self.queue_id,
                "plan_sha256": self.plan_sha,
            },
        )
        _write_json(
            self.job_dir / "status.json",
            {
                "status": "failed",
                "run_ids": [self.run_id],
                "error": failure_text,
            },
        )
        _write_json(
            self.job_dir / "launch.json",
            {"status": "launched", "run_ids": [self.run_id]},
        )
        (self.job_dir / "orchestrator.log").write_text(
            "nvidia-smi identity query failed:\n", encoding="utf-8"
        )
        _write_json(self.job_dir / "plans" / "L2" / "seed42.json", {})
        (self.orchestration_root / "detach_launcher.log").write_text(
            "launched\n", encoding="utf-8"
        )
        _write_json(
            self.output_root / "sequence_manifest.json",
            {
                "schema": "pivot.stageb.token_ablation_sequence/v1",
                "status": "failed",
                "run_id": self.run_id,
                "completed_phases": [],
                "error": failure_text,
            },
        )
        _write_json(
            self.output_root / "launch_manifest.json",
            {
                "schema": "pivot.stageb.token_ablation_launch/v2",
                "status": "failed",
                "run_id": self.run_id,
                "failure_phase": recovery.FAILURE_PHASE,
                "failure_error": failure_text,
                "output_dir_fresh_at_plan": True,
            },
        )

    def inspect(self):
        return recovery.inspect_failure(
            self.queue_dir,
            run_id=self.run_id,
            expected_queue_id=self.queue_id,
            expected_plan_sha256=self.plan_sha,
        )

    def apply(self, archive: Path, *, revision: int = 7):
        return recovery.apply_recovery(
            self.queue_dir,
            run_id=self.run_id,
            archive_root=archive,
            expected_queue_id=self.queue_id,
            expected_plan_sha256=self.plan_sha,
            expected_failed_revision=revision,
        )


class TestPretrainingFailureRecovery(unittest.TestCase):
    def test_archives_failed_evidence_and_reopens_same_queue_item(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            archive = Path(temporary) / "archive" / "attempt0"
            with patch.object(recovery.queue_runner, "_process_running", return_value=False):
                inspection = fixture.inspect()
                result = fixture.apply(archive)

            self.assertEqual(inspection["status"], "eligible")
            self.assertEqual(inspection["proof"]["checkpoint_count"], 0)
            self.assertEqual(result["status"], "reopened")
            self.assertFalse(fixture.output_root.exists())
            self.assertFalse(fixture.orchestration_root.exists())
            self.assertTrue((archive / "failed_output_root").is_dir())
            self.assertTrue((archive / "failed_orchestration_root").is_dir())

            queue = queue_runner.load_queue(fixture.queue_dir)
            self.assertEqual(queue["status"], "running")
            self.assertEqual(queue["revision"], 8)
            self.assertEqual(queue["plan_sha256"], fixture.plan_sha)
            self.assertEqual(queue["items"][0]["status"], "pending")
            receipt = archive / "recovery_receipt.json"
            verified = recovery.verify_recovery(fixture.queue_dir, receipt)
            self.assertEqual(verified["status"], "passed")

    def test_rejects_any_training_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            (fixture.output_root / "checkpoint_iter.pth").write_bytes(b"checkpoint")
            with patch.object(recovery.queue_runner, "_process_running", return_value=False):
                with self.assertRaisesRegex(recovery.RecoveryError, "training or unexpected"):
                    fixture.inspect()

    def test_rejects_live_or_ambiguous_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            with patch.object(recovery.queue_runner, "_process_running", return_value=True):
                with self.assertRaisesRegex(recovery.RecoveryError, "liveness"):
                    fixture.inspect()

    def test_revision_mismatch_has_no_archive_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            archive = Path(temporary) / "archive"
            with patch.object(recovery.queue_runner, "_process_running", return_value=False):
                with self.assertRaisesRegex(recovery.RecoveryError, "revision"):
                    fixture.apply(archive, revision=6)
            self.assertFalse(archive.exists())
            self.assertTrue(fixture.output_root.is_dir())
            self.assertEqual(queue_runner.load_queue(fixture.queue_dir)["status"], "failed")

    def test_tampered_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            archive = Path(temporary) / "archive"
            with patch.object(recovery.queue_runner, "_process_running", return_value=False):
                fixture.apply(archive)
            receipt_path = archive / "recovery_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["proof"]["checkpoint_count"] = 1
            _write_json(receipt_path, receipt)
            with self.assertRaisesRegex(recovery.RecoveryError, "canonical SHA"):
                recovery.verify_recovery(fixture.queue_dir, receipt_path)


if __name__ == "__main__":
    unittest.main()
