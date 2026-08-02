from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

from tools import audit_stageb_table_c_dependency_closure as audit
from tools import run_stageb_serial_matrix_queue as queue_runner


OLD_MTIME_NS = 1_577_836_800_000_000_000  # 2020-01-01 UTC
POST_START_MTIME_NS = 1_924_992_000_000_000_000  # 2031-01-01 UTC
STARTED_AT = "2030-01-01T00:00:00+00:00"


def _write(path: Path, value: str | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    else:
        rendered = value
    path.write_text(rendered, encoding="utf-8")


def _set_old(path: Path) -> None:
    os.utime(path, ns=(OLD_MTIME_NS, OLD_MTIME_NS))


class SyntheticTableC:
    def __init__(self, root: Path) -> None:
        self.workspace = root
        self.repo = root / "repo"
        self.training = root / "training"
        self.completed_queue = root / "completed_queue"
        self.remaining_queue = root / "remaining_queue"
        self.output = root / "attestation.json"
        self.config_entries = {
            f"L{index}": f"config/l{index}.py" for index in range(11)
        }
        self._build_repository()
        self._build_training()
        self._build_queue(
            self.completed_queue,
            audit.COMPLETED_RUN_IDS,
            status="completed",
        )
        self._build_queue(
            self.remaining_queue,
            audit.REMAINING_RUN_IDS,
            status="running",
        )

    def _source(self, relative: str, content: str) -> Path:
        path = self.repo / relative
        _write(path, content)
        _set_old(path)
        return path

    def _build_repository(self) -> None:
        self._source("main.py", "from pkg import core\n")
        self._source("tools/token_runner.py", "from pkg import core\n")
        self._source("pkg/core.py", "from util import helper\n")
        self._source("util/helper.py", "VALUE = 1\n")
        self._source("config/base.py", "BASE = True\n")
        for index in range(11):
            self._source(
                f"config/l{index}.py",
                "from config.base import *\nROW = " + repr(f"L{index}") + "\n",
            )
        self._source("docs/protocol.md", "synthetic protocol\n")

    def _launch_record(self, path: Path) -> dict[str, Any]:
        return audit._file_record(path)

    def _build_training(self) -> None:
        static_paths = (
            self.repo / "main.py",
            self.repo / "tools/token_runner.py",
            self.repo / "docs/protocol.md",
        )
        chains = audit._config_chains(self.repo.resolve(), self.config_entries)
        for index in range(5):
            row_id = f"L{index}"
            run_id = f"{row_id}:17"
            run_root = self.training / row_id / "seed17"
            run_root.mkdir(parents=True)
            postflight_path = run_root / "postflight.json"
            _write(
                postflight_path,
                {
                    "schema": "pivot.stageb.token_ablation_postflight/v2",
                    "status": "passed",
                    "run_id": run_id,
                    "input_rehash": {"status": "passed"},
                },
            )
            postflight_record = self._launch_record(postflight_path)
            sequence_path = run_root / "sequence_manifest.json"
            _write(
                sequence_path,
                {
                    "schema": "pivot.stageb.token_ablation_sequence/v1",
                    "status": "completed",
                    "run_id": run_id,
                    "started_at_utc": STARTED_AT,
                    "output_dir": str(run_root.resolve()),
                    "phases": [
                        {
                            "phase_id": "joint",
                            "output_dir": str(run_root.resolve()),
                        }
                    ],
                    "completed_phases": [
                        {
                            "phase_id": "joint",
                            "status": "completed",
                            "output_dir": str(run_root.resolve()),
                            "postflight": postflight_record,
                        }
                    ],
                },
            )
            launch_path = run_root / "launch_manifest.json"
            _write(
                launch_path,
                {
                    "schema": "pivot.stageb.token_ablation_launch/v2",
                    "status": "completed",
                    "run_id": run_id,
                    "started_at_utc": STARTED_AT,
                    "repository_root": str(self.repo.resolve()),
                    "output_dir": str(run_root.resolve()),
                    "postflight_artifact": postflight_record,
                    "inputs": {
                        "repository_sources": [
                            self._launch_record(path) for path in static_paths
                        ],
                        "config_dependencies": [
                            self._launch_record(self.repo / relative)
                            for relative in chains[row_id]
                        ],
                    },
                },
            )

    def _build_queue(
        self,
        queue_dir: Path,
        run_ids: tuple[str, ...],
        *,
        status: str,
    ) -> None:
        queue_dir.mkdir(parents=True)
        token_runner = (self.repo / "tools/token_runner.py").resolve()
        plan: dict[str, Any] = {
            "schema": queue_runner.PLAN_SCHEMA,
            "queue_id": f"synthetic-{queue_dir.name}",
            "created_at_utc": "2030-01-01T00:00:00+00:00",
            "queue_dir": str(queue_dir.resolve()),
            "repository_root": str(self.repo.resolve()),
            "runners": {
                "token": {
                    "path": str(token_runner),
                    "sha256": audit._sha256_file(token_runner),
                }
            },
            "items": [
                {"run_id": run_id, "runner": "token"} for run_id in run_ids
            ],
        }
        if status == "completed":
            item_statuses = ["completed"] * len(run_ids)
        else:
            item_statuses = ["launched"] + ["pending"] * (len(run_ids) - 1)
        queue = {
            "schema": queue_runner.QUEUE_SCHEMA,
            "status": status,
            "plan": plan,
            "plan_sha256": audit._canonical_sha256(plan),
            "items": [
                {
                    "index": index,
                    "run_id": run_id,
                    "runner": "token",
                    "status": item_statuses[index],
                }
                for index, run_id in enumerate(run_ids)
            ],
        }
        _write(queue_dir / "queue.json", queue)

    @contextmanager
    def queue_verifier(self) -> Iterator[None]:
        def verify(queue_dir: Path) -> dict[str, Any]:
            queue = queue_runner.load_queue(Path(queue_dir))
            completed = [
                item for item in queue["items"] if item["status"] == "completed"
            ]
            return {
                "schema": "pivot.stageb.serial_matrix_queue_verification/v1",
                "status": "passed",
                "queue_id": queue["plan"]["queue_id"],
                "plan_sha256": queue["plan_sha256"],
                "verified_items": [{} for _ in completed],
                "errors": [],
            }

        with mock.patch.object(audit.queue_runner, "verify_queue", side_effect=verify):
            yield

    def create(self, *, policy: str = "preflight") -> dict[str, Any]:
        with self.queue_verifier():
            return audit.create_attestation(
                self.output,
                repository_root=self.repo,
                completed_queue_dir=self.completed_queue,
                remaining_queue_dir=self.remaining_queue,
                training_root=self.training,
                config_entries=self.config_entries,
                policy=policy,
            )

    def verify(self, *, policy: str = "preflight") -> dict[str, Any]:
        with self.queue_verifier():
            return audit.verify_attestation(
                self.output,
                policy=policy,
                config_entries=self.config_entries,
            )


class StageBTableCDependencyClosureTest(unittest.TestCase):
    def test_create_and_verify_records_supplemental_limit_and_deterministic_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            payload = fixture.create()
            closure = payload["dependency_closure"]
            recursive = closure["recursive_local_python"]
            self.assertEqual(recursive["path_count"], 4)
            self.assertEqual(recursive["static_repository_source_bound_count"], 2)
            self.assertEqual(recursive["supplemental_only_count"], 2)
            self.assertIn("util/helper.py", recursive["supplemental_only_paths"])
            self.assertFalse(
                payload["claim_scope"]["retroactively_launch_binds_omitted_files"]
            )
            self.assertIn(
                "does not retroactively make omitted dependencies launch-bound",
                payload["limitations"]["primary"],
            )
            records = closure["file_records"]
            self.assertTrue(
                all(
                    {"path", "relative_path", "sha256", "size_bytes", "mtime_ns"}
                    <= set(record)
                    for record in records
                )
            )
            result = fixture.verify()
            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                result["canonical_closure_sha256"],
                closure["canonical_closure_sha256"],
            )

    def test_verify_rejects_supplemental_file_content_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            fixture.create()
            helper = fixture.repo / "util/helper.py"
            _write(helper, "VALUE = 2\n")
            _set_old(helper)
            with self.assertRaisesRegex(
                audit.TableCDependencyClosureError,
                "dependency file identity or membership drift.*util/helper.py",
            ):
                fixture.verify()

    def test_create_rejects_supplemental_file_mtime_after_training_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            helper = fixture.repo / "util/helper.py"
            os.utime(helper, ns=(POST_START_MTIME_NS, POST_START_MTIME_NS))
            with fixture.queue_verifier(), self.assertRaisesRegex(
                audit.TableCDependencyClosureError,
                "supplemental file mtime does not predate.*util/helper.py",
            ):
                audit.create_attestation(
                    fixture.output,
                    repository_root=fixture.repo,
                    completed_queue_dir=fixture.completed_queue,
                    remaining_queue_dir=fixture.remaining_queue,
                    training_root=fixture.training,
                    config_entries=fixture.config_entries,
                )
            self.assertFalse(fixture.output.exists())

    def test_verify_rejects_canonically_valid_queue_plan_identity_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            fixture.create()
            queue_path = fixture.remaining_queue / "queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue["plan"]["queue_id"] = "drifted-but-canonical-plan"
            queue["plan_sha256"] = audit._canonical_sha256(queue["plan"])
            _write(queue_path, queue)
            with self.assertRaisesRegex(
                audit.TableCDependencyClosureError,
                "remaining Table-C queue plan identity drift",
            ):
                fixture.verify()

    def test_verify_rejects_missing_completed_training_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            fixture.create()
            (fixture.training / "L3/seed17/postflight.json").unlink()
            with self.assertRaisesRegex(
                audit.TableCDependencyClosureError,
                "L3:17 postflight is missing",
            ):
                fixture.verify()

    def test_verify_rejects_recursive_closure_set_drift_before_identity_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            fixture.create()
            helper = fixture.repo / "util/helper.py"
            extra = fixture.repo / "util/extra.py"
            _write(helper, "from util import extra\nVALUE = 1\n")
            _write(extra, "EXTRA = True\n")
            _set_old(helper)
            _set_old(extra)
            with self.assertRaisesRegex(
                audit.TableCDependencyClosureError,
                "dependency closure-set drift",
            ):
                fixture.verify()

    def test_final_policy_rejects_running_remaining_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            fixture.create()
            with self.assertRaisesRegex(
                audit.TableCDependencyClosureError,
                "remaining_table_c queue status 'running'.*final policy",
            ):
                fixture.verify(policy="final")

    def test_final_create_rejects_running_remaining_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            with self.assertRaisesRegex(
                audit.TableCDependencyClosureError,
                "remaining_table_c queue status 'running'.*final policy",
            ):
                fixture.create(policy="final")
            self.assertFalse(fixture.output.exists())

    def test_final_create_accepts_completed_queue_and_records_required_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            queue_path = fixture.remaining_queue / "queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue["status"] = "completed"
            for item in queue["items"]:
                item["status"] = "completed"
            _write(queue_path, queue)

            payload = fixture.create(policy="final")

            remaining = payload["queues"]["remaining_table_c"]
            self.assertEqual(remaining["observed_status"], "completed")
            self.assertEqual(remaining["status_policy"], "completed_required")
            self.assertEqual(
                remaining["completion_verification"]["verified_item_count"],
                len(audit.REMAINING_RUN_IDS),
            )

    def test_create_cli_forwards_final_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attestation.json"
            _write(output, {})
            payload = {
                "attestation_sha256": "a" * 64,
                "dependency_closure": {
                    "canonical_closure_sha256": "b" * 64,
                    "combined_with_l0_l10_config_chains": {"path_count": 85},
                },
            }
            with mock.patch.object(
                audit, "create_attestation", return_value=payload
            ) as create, mock.patch("builtins.print"):
                result = audit.main(
                    [
                        "create",
                        "--output",
                        str(output),
                        "--policy",
                        "final",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(create.call_args.kwargs["policy"], "final")

    def test_create_rejects_noncanonical_queue_plan_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticTableC(Path(temporary))
            queue_path = fixture.remaining_queue / "queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue["plan"]["queue_id"] = "unhashed-drift"
            _write(queue_path, queue)
            with fixture.queue_verifier(), self.assertRaisesRegex(
                audit.TableCDependencyClosureError,
                "canonical plan verification failed.*immutable queue plan SHA-256 mismatch",
            ):
                audit.create_attestation(
                    fixture.output,
                    repository_root=fixture.repo,
                    completed_queue_dir=fixture.completed_queue,
                    remaining_queue_dir=fixture.remaining_queue,
                    training_root=fixture.training,
                    config_entries=fixture.config_entries,
                )


if __name__ == "__main__":
    unittest.main()
