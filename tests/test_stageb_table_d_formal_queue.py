from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import run_stageb_table_d_formal_queue as queue
from tools import run_stageb_table_d_formal_training as wrapper


class StageBTableDFormalQueueTest(unittest.TestCase):
    def test_exact_training_inventory_and_runtime_contract(self):
        self.assertEqual(wrapper.RUN_IDS, queue.RUN_IDS)
        self.assertEqual(
            queue.RUN_IDS,
            (
                "S0:17", "S0:42", "S0:73",
                "S1:17", "S1:42", "S1:73",
                "S2:17", "S2:42", "S2:73",
                "S3:17", "S3:42", "S3:73",
                "S2F:17", "S2F:42", "S2F:73",
            ),
        )
        self.assertEqual(
            queue.FORMAL_TRAINING_CONTRACT,
            {
                "batch_size": 40,
                "optimizer_updates": 1000,
                "iter_checkpoint_interval": 1000,
                "num_workers": 2,
                "gradient_diagnostic_interval": 100,
                "successful_update_batch_slots_per_run": 40000,
                "successful_update_batch_slots_total": 600000,
                "s3": {
                    "isolation_probe_updates_excluded": 1,
                    "rank_updates": 500,
                    "confidence_updates": 500,
                },
            },
        )

    def test_cpu_preflight_plans_all_phases_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            output_root = root / "output"
            readiness = {
                "row_id": "fixture",
                "batch_size": 40,
                "optimizer_updates": 1,
                "gpu_identity": {
                    "uuid": "GPU-test",
                    "name": "test",
                    "driver_version": "test",
                    "total_memory_mib": 1.0,
                },
            }
            with patch.object(
                queue, "verify_s2_soak_seal", return_value=readiness
            ), patch.object(
                queue, "verify_s2f_confirmation", return_value=readiness
            ):
                report = queue.preflight_training_queue(
                    queue_dir,
                    output_root=output_root,
                    s2_soak_seal=root / "not-read.json",
                    s2f_confirmation_root=root / "not-read",
                    runner_python=queue.paper.DEFAULT_PYTHON,
                )
            self.assertEqual(report["status"], "ready")
            self.assertFalse(report["mutated"])
            self.assertEqual(report["ordered_run_ids"], list(queue.RUN_IDS))
            self.assertEqual(report["common_input_contract"]["phase_count"], 21)
            self.assertFalse(queue_dir.exists())
            self.assertFalse(output_root.exists())

    def test_missing_readiness_fails_before_creating_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            output_root = root / "output"
            with self.assertRaises(FileNotFoundError):
                queue.preflight_training_queue(
                    queue_dir,
                    output_root=output_root,
                    s2_soak_seal=root / "missing-seal.json",
                    s2f_confirmation_root=root / "missing-confirmation",
                )
            self.assertFalse(queue_dir.exists())
            self.assertFalse(output_root.exists())

    def test_preflight_rejects_nonsealed_training_python_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            output_root = root / "output"
            with self.assertRaisesRegex(
                queue.TableDFormalQueueError, "sealed GDINO Python"
            ):
                queue.preflight_training_queue(
                    queue_dir,
                    output_root=output_root,
                    s2_soak_seal=root / "not-read.json",
                    s2f_confirmation_root=root / "not-read",
                    runner_python=Path("/usr/bin/true"),
                )
            self.assertFalse(queue_dir.exists())
            self.assertFalse(output_root.exists())

    def test_create_binds_generic_plan_extension_without_acquiring_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            output_root = root / "output"
            lease_root = root / "leases"
            readiness = {
                "row_id": "fixture",
                "batch_size": 40,
                "optimizer_updates": 1,
                "gpu_identity": {
                    "uuid": "GPU-test",
                    "name": "test",
                    "driver_version": "test",
                    "total_memory_mib": 1.0,
                },
            }
            closure = {
                "status": "sealed",
                "records": [
                    queue._file_record(
                        Path(queue.__file__), roles=("formal_controller_source",)
                    )
                ],
            }
            closure["semantic_sha256"] = queue._canonical_sha(closure["records"])
            with patch.object(
                queue, "verify_s2_soak_seal", return_value=readiness
            ), patch.object(
                queue, "verify_s2f_confirmation", return_value=readiness
            ), patch.object(
                queue, "_source_closure", return_value=closure
            ), patch.object(queue, "_load_plans", return_value=({}, {}, {})):
                report = queue.create_training_queue(
                    queue_dir,
                    output_root=output_root,
                    s2_soak_seal=root / "not-read.json",
                    s2f_confirmation_root=root / "not-read",
                    runner_python=queue.paper.DEFAULT_PYTHON,
                    lease_root=lease_root,
                    gpu_key="0",
                )
            persisted = queue.serial_queue.load_queue(queue_dir)
            plan = persisted["plan"]
            self.assertEqual(report["ordered_run_ids"], list(queue.RUN_IDS))
            self.assertEqual(
                [item["run_id"] for item in plan["items"]], list(queue.RUN_IDS)
            )
            self.assertEqual(
                Path(plan["runners"]["paper"]["path"]),
                queue.FORMAL_TRAINING_WRAPPER,
            )
            self.assertEqual(plan["extensions"]["profile"], queue.PROFILE)
            self.assertEqual(plan["runtime_environment"]["PIVOT_BATCH_SIZE"], "40")
            self.assertEqual(plan["runtime_environment"]["PIVOT_NUM_WORKERS"], "2")
            self.assertFalse(Path(plan["lease_path"]).exists())
            self.assertFalse(output_root.exists())

    def test_s2f_confirmation_is_exact_b40_u2_i2_w2_d1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = {
                "phase": {"diagnostic_interval": 1},
                "runtime": {"num_workers": 2, "iter_checkpoint_interval": 2},
            }
            (root / "launch_manifest.json").write_text(json.dumps(launch))
            artifacts = {}
            for name in ("sequence_manifest", "postflight", "gpu_telemetry_summary"):
                path = root / f"{name}.json"
                path.write_text("{}\n")
                artifacts[name] = queue.probe_sealer._file_record(path)
            observed = {
                "root": str(root),
                "batch_size": 40,
                "optimizer_updates": 2,
                "gpu": {
                    "uuid": "GPU-test",
                    "name": "test",
                    "driver_version": "test",
                    "total_memory_mib": 1.0,
                    "min_free_memory_mib": 2048.0,
                },
                "artifacts": artifacts,
            }
            with patch.object(
                queue.probe_sealer, "_verify_paper_probe_current"
            ) as replay, patch.object(
                queue.probe_sealer, "inspect_probe", return_value=observed
            ):
                record = queue.verify_s2f_confirmation(root)
            replay.assert_called_once_with(
                root.resolve(), expected_row_id="S2F", minimum_soak_updates=50
            )
            self.assertEqual(record["diagnostic_interval"], 1)
            self.assertEqual(record["optimizer_updates"], 2)
            self.assertEqual(record["minimum_headroom_mib"], 1024.0)

    def test_s2f_confirmation_rejects_insufficient_gpu_headroom(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "launch_manifest.json").write_text(
                json.dumps(
                    {
                        "phase": {"diagnostic_interval": 1},
                        "runtime": {
                            "num_workers": 2,
                            "iter_checkpoint_interval": 2,
                        },
                    }
                )
            )
            observed = {
                "batch_size": 40,
                "optimizer_updates": 2,
                "gpu": {
                    "uuid": "GPU-test",
                    "name": "test",
                    "driver_version": "test",
                    "total_memory_mib": 32768.0,
                    "min_free_memory_mib": 512.0,
                },
                "artifacts": {},
            }
            with patch.object(
                queue.probe_sealer, "_verify_paper_probe_current"
            ), patch.object(
                queue.probe_sealer, "inspect_probe", return_value=observed
            ), self.assertRaisesRegex(
                queue.TableDFormalQueueError, "at least 1024.0 MiB headroom"
            ):
                queue.verify_s2f_confirmation(root)

    def test_readiness_pair_rejects_different_gpu_identity(self):
        with self.assertRaisesRegex(
            queue.TableDFormalQueueError, "exact GPU identity"
        ):
            queue._validate_readiness_pair(
                {
                    "s2_b40_u50_soak": {
                        "gpu_identity": {"uuid": "GPU-a"}
                    },
                    "s2f_b40_u2_confirmation": {
                        "gpu_identity": {"uuid": "GPU-b"}
                    },
                }
            )

    def test_authorization_rejects_nonactive_detach(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        queue_dir = Path(temporary.name) / "queue"
        queue_dir.mkdir()
        item_root = queue_dir / "jobs/000-S0_17"
        synthetic = {
            "status": "running",
            "plan": {
                "items": [
                    {"run_id": "S0:17", "runner": "paper"},
                    {"run_id": "S0:42", "runner": "paper"},
                ]
            },
            "items": [
                {
                    "index": 0,
                    "run_id": "S0:17",
                    "runner": "paper",
                    "status": "reserved",
                    "orchestration_root": str(item_root),
                },
                {
                    "index": 1,
                    "run_id": "S0:42",
                    "runner": "paper",
                    "status": "pending",
                },
            ],
        }
        with patch.object(queue, "_load_plans", return_value=({}, {}, synthetic)), patch.object(
            queue.serial_queue, "_ensure_lease"
        ) as lease:
            queue.authorize_wrapper_operation(
                queue_dir,
                run_id="S0:17",
                orchestration_root=item_root,
            )
            with self.assertRaisesRegex(
                queue.TableDFormalQueueError, "active queue item"
            ):
                queue.authorize_wrapper_operation(
                    queue_dir,
                    run_id="S0:42",
                    orchestration_root=item_root,
                )
        lease.assert_called_once()

    def test_s3_completion_invokes_atomic_lineage_and_checkpoint_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "outputs/S3/seed17"
            job_dir = root / "job"
            run_root.mkdir(parents=True)
            job_dir.mkdir()
            (job_dir / "launch.json").write_text("{}\n")
            (job_dir / "status.json").write_text("{}\n")
            rank_checkpoint = root / "rank.pth"
            confidence_checkpoint = root / "confidence.pth"
            rank_checkpoint.write_bytes(b"rank")
            confidence_checkpoint.write_bytes(b"confidence")
            phases = [
                SimpleNamespace(phase_id=value)
                for value in ("isolation_probe", "rank", "confidence")
            ]
            sequence_phases = [
                {"phase": {"phase_id": value}}
                for value in ("isolation_probe", "rank", "confidence")
            ]
            completed = [
                {"phase_id": value}
                for value in ("isolation_probe", "rank", "confidence")
            ]
            current_plan = {
                "schema": "sequence",
                "repository_root": str(queue.REPO_ROOT),
                "run_id": "S3:17",
                "row": {"row_id": "S3"},
                "seed": 17,
                "training_seeds_contract": [17, 42, 73],
                "output_dir": str(run_root),
                "equal_budget_contract": {},
                "phases": sequence_phases,
            }
            sequence = {**current_plan, "status": "completed", "completed_phases": completed}
            (run_root / "sequence_manifest.json").write_text(json.dumps(sequence))
            scope = {
                "runs": {
                    "S3:17": {
                        "sequence_contract_sha256": queue._canonical_sha(
                            queue._immutable_sequence(current_plan)
                        ),
                        "scope_sha256": "a" * 64,
                        "phases": [
                            {"phase_id": value}
                            for value in ("isolation_probe", "rank", "confidence")
                        ],
                    }
                }
            }
            final = SimpleNamespace(
                training_run_id="S3:17",
                training_seed=17,
                training_phase="final",
                diagnostic_only=False,
                final_phase_id="confidence",
                checkpoint=confidence_checkpoint,
            )
            rank = SimpleNamespace(
                training_run_id="S3:17",
                training_seed=17,
                training_phase="rank",
                diagnostic_only=True,
                final_phase_id="confidence",
                checkpoint=rank_checkpoint,
            )
            replay_results = [
                ({"phase_id": "isolation_probe"}, root / "probe.pth"),
                ({"phase_id": "rank"}, rank_checkpoint),
                ({"phase_id": "confidence"}, confidence_checkpoint),
            ]
            (root / "probe.pth").write_bytes(b"probe")
            with patch.object(queue.paper, "build_manifest", return_value=current_plan), patch.object(
                queue.paper, "_phases", return_value=tuple(phases)
            ), patch.object(
                queue, "_replay_phase", side_effect=replay_results
            ), patch.object(
                queue.evaluator,
                "_resolve_paper_source",
                side_effect=[final, rank],
            ), patch.object(
                queue.diagnostics,
                "_verify_s3_training_lineage",
                return_value={"status": "passed"},
            ) as lineage, patch.object(
                queue.diagnostics,
                "checkpoint_allowlist",
                return_value={"status": "passed"},
            ) as allowlist:
                report = queue._verify_completed_training_run(
                    queue_dir=root,
                    source_plan={"runtime": {"output_root": str(root / "outputs")}},
                    scope_plan=scope,
                    generic_item={
                        "index": 9,
                        "run_id": "S3:17",
                        "runner": "paper",
                        "status": "completed",
                    },
                    generic_evidence={
                        "run_id": "S3:17",
                        "runner": "paper",
                        "output_root": str(run_root),
                        "job_dir": str(job_dir),
                    },
                    runtime=SimpleNamespace(),
                    run_id="S3:17",
                )
            self.assertEqual(report["s3_atomic_replay"]["status"], "passed")
            lineage.assert_called_once_with(rank, final)
            allowlist.assert_called_once_with(rank_checkpoint, confidence_checkpoint)


if __name__ == "__main__":
    unittest.main()
