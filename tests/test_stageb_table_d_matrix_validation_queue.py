from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import run_stageb_table_d_formal_evaluations as formal_eval
from tools import run_stageb_table_d_matrix_validation_queue as queue


class StageBTableDMatrixValidationQueueTest(unittest.TestCase):
    def test_exact_eighteen_job_order_and_distinct_s3_phase_ids(self):
        self.assertEqual(len(queue.JOB_IDS), 18)
        self.assertEqual(
            queue.JOB_IDS,
            (
                "S0:17/final", "S0:42/final", "S0:73/final",
                "S1:17/final", "S1:42/final", "S1:73/final",
                "S2:17/final", "S2:42/final", "S2:73/final",
                "S3:17/final", "S3:42/final", "S3:73/final",
                "S2F:17/final", "S2F:42/final", "S2F:73/final",
                "S3:17/rank", "S3:42/rank", "S3:73/rank",
            ),
        )
        self.assertNotEqual("S3:17/final", "S3:17/rank")
        self.assertEqual(queue._job_parts("S3:17/final"), ("S3", 17, "final"))
        self.assertEqual(queue._job_parts("S3:17/rank"), ("S3", 17, "rank"))

    def test_build_plan_predeclares_all_commands_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_root = root / "training"
            for run_id in queue.training_queue.RUN_IDS:
                row, raw_seed = run_id.split(":", 1)
                (training_root / row / f"seed{raw_seed}").mkdir(parents=True)
            training_record = {
                "queue_dir": str(root / "training-queue"),
                "queue_id": "queue-id",
                "plan_sha256": "a" * 64,
                "profile": queue.training_queue.PROFILE,
                "ordered_run_ids": list(queue.training_queue.RUN_IDS),
                "completion_semantic_sha256": "b" * 64,
                "completion_attestation": {},
                "source_plan": {},
                "scope_plan": {},
                "run_scope_sha256s": {
                    run_id: "c" * 64 for run_id in queue.training_queue.RUN_IDS
                },
                "output_root": str(training_root),
            }
            queue_dir = root / "validation-queue"
            output_root = root / "evaluations"
            current = Path(__file__).resolve()
            with patch.object(
                queue, "_training_queue_record", return_value=training_record
            ), patch.object(
                queue, "_evaluation_sources", return_value=[current]
            ), patch.object(
                queue, "_recursive_sources", return_value=[current]
            ), patch.object(
                queue.sys,
                "executable",
                str(queue.formal_eval.evaluator.DEFAULT_PYTHON),
            ):
                plan = queue.build_plan(
                    queue_dir,
                    training_queue_dir=root / "training-queue",
                    output_root=output_root,
                    lease_root=root / "leases",
                    gpu_key="0",
                )
            self.assertEqual(plan["ordered_job_ids"], list(queue.JOB_IDS))
            self.assertEqual(len(plan["items"]), 18)
            self.assertEqual(
                [item["training_phase"] for item in plan["items"][-3:]],
                ["rank", "rank", "rank"],
            )
            for item in plan["items"]:
                self.assertEqual(
                    item["command_sha256"], queue._canonical_sha(item["command"])
                )
                self.assertIn("--matrix-queue-spec", item["command"])
                self.assertIn("--training-queue-dir", item["command"])
                self.assertEqual(
                    item["process_environment"],
                    {
                        "CUDA_VISIBLE_DEVICES": "0",
                        "PIVOT_TABLE_D_VALIDATION_QUEUE_ID": plan["queue_id"],
                        "PIVOT_TABLE_D_VALIDATION_JOB_ID": item["job_id"],
                    },
                )
            self.assertFalse(queue_dir.exists())
            self.assertFalse(output_root.exists())
            self.assertFalse((root / "leases").exists())
            planned_queue = {
                "schema": queue.QUEUE_SCHEMA,
                "status": "planned",
                "revision": 0,
                "plan": plan,
                "plan_sha256": queue._canonical_sha(plan),
                "final_verification": None,
                "items": [
                    {"index": index, "run_id": job_id, "status": "pending"}
                    for index, job_id in enumerate(queue.JOB_IDS)
                ],
                "events": [{"event": "queue_created"}],
            }
            queue._validate_queue(planned_queue, queue_dir)

    def test_build_plan_rejects_nested_control_and_evidence_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            output_root = queue_dir / "evaluations"
            with self.assertRaisesRegex(
                queue.TableDValidationQueueError, "must be disjoint"
            ):
                queue.build_plan(
                    queue_dir,
                    training_queue_dir=root / "not-read",
                    output_root=output_root,
                    lease_root=root / "leases",
                    gpu_key="0",
                )
            self.assertFalse(queue_dir.exists())
            self.assertFalse(output_root.exists())

    def test_formal_source_resolution_adds_authority_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "S0/seed17"
            queue_dir = root / "queue"
            run_root.mkdir(parents=True)
            queue_dir.mkdir()
            (run_root / "sequence_manifest.json").write_text(
                '{"run_id":"S0:17"}\n'
            )
            authority = []
            for name in ("source.json", "scope.json", "completion.json"):
                path = root / name
                path.write_text("{}\n")
                authority.append({"path": str(path)})
            original_data = root / "train.jsonl"
            original_data.write_text("{}\n")
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            config.write_text("value = 1\n")
            checkpoint.write_bytes(b"checkpoint")
            source = formal_eval.evaluator.EvaluationSource(
                kind="pivot_paper_training_run",
                evaluation_id="S0_seed17",
                config=config,
                checkpoint=checkpoint,
                checkpoint_sha256="a" * 64,
                training_run_id="S0:17",
                training_seed=17,
                training_run_root=run_root,
                training_phase="final",
                diagnostic_only=False,
                training_data=(original_data,),
            )
            evidence = {
                "profile": formal_eval.training_queue.PROFILE,
                "run_id": "S0:17",
                "training_phase": "final",
                "queue_id": "queue-id",
                "queue_plan_sha256": "a" * 64,
                "completion_semantic_sha256": "b" * 64,
                "source_plan": authority[0],
                "scope_plan": authority[1],
                "completion_attestation": authority[2],
                "run_verification": {"scope_sha256": "c" * 64},
            }
            with patch.object(
                formal_eval.training_queue,
                "formal_evaluation_evidence",
                return_value=evidence,
            ), patch.object(
                formal_eval.evaluator, "_resolve_paper_source", return_value=source
            ):
                resolved, binding = formal_eval.resolve_formal_source(
                    training_queue_dir=queue_dir,
                    training_run_root=run_root,
                    training_phase="final",
                )
            self.assertEqual(binding["scope_sha256"], "c" * 64)
            self.assertEqual(
                resolved.training_data,
                (original_data, *(Path(value["path"]) for value in authority)),
            )

    def test_evaluation_scope_plan_predeclares_inputs_commands_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / queue.AGGREGATION_SPEC_NAME
            spec.write_text("{}\n")
            items = []
            for index, job_id in enumerate(queue.JOB_IDS):
                row, seed, phase = queue._job_parts(job_id)
                items.append(
                    {
                        "index": index,
                        "job_id": job_id,
                        "run_id": job_id,
                        "training_run_id": f"{row}:{seed}",
                        "training_phase": phase,
                        "training_scope_sha256": "a" * 64,
                        "training_root": str(root),
                        "evaluation_root": str(root / f"evaluation-{index}"),
                        "command_sha256": "b" * 64,
                    }
                )
            persisted = {
                "plan_sha256": "c" * 64,
                "plan": {
                    "queue_id": "queue-id",
                    "training_queue": {"queue_dir": str(root)},
                    "items": items,
                },
            }
            fake_plan = {
                "schema": "evaluation",
                "repository_root": str(queue.REPO_ROOT),
                "evaluation_id": "fixture",
                "output_dir": str(root / "evaluation"),
                "matrix_validation_queue_spec": {},
                "source": {"training_run_id": "fixture"},
                "runtime": {"batch_size": 16},
                "protocol": {"profile": queue.PROFILE},
                "commands": [{"command": ["python"]}],
                "inputs": {"records": [{"path": "fixture"}]},
                "table_d_formal": {"scope_sha256": "a" * 64},
            }
            with patch.object(
                queue.formal_eval,
                "build_formal_plan",
                return_value=(fake_plan, SimpleNamespace()),
            ):
                scope = queue._build_evaluation_scope_plan(root, persisted)
            self.assertEqual(scope["ordered_job_ids"], list(queue.JOB_IDS))
            self.assertEqual(set(scope["items"]), set(queue.JOB_IDS))
            first = scope["items"][queue.JOB_IDS[0]]
            self.assertEqual(
                first["input_identity_sha256"],
                queue._canonical_sha(fake_plan["inputs"]),
            )
            self.assertEqual(
                first["inner_commands_sha256"],
                queue._canonical_sha(fake_plan["commands"]),
            )
            self.assertEqual(scope["semantic_sha256"], queue._semantic_sha(scope))

    def test_rank_source_is_restricted_to_s3(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "S0/seed17"
            queue_dir = root / "queue"
            run_root.mkdir(parents=True)
            queue_dir.mkdir()
            (run_root / "sequence_manifest.json").write_text(
                '{"run_id":"S0:17"}\n'
            )
            with patch.object(
                formal_eval.training_queue,
                "formal_evaluation_evidence",
                side_effect=formal_eval.training_queue.TableDFormalQueueError(
                    "rank evaluation is restricted to S3"
                ),
            ):
                with self.assertRaisesRegex(
                    formal_eval.TableDFormalEvaluationError,
                    "restricted to S3",
                ):
                    formal_eval.resolve_formal_source(
                        training_queue_dir=queue_dir,
                        training_run_root=run_root,
                        training_phase="rank",
                    )

    def test_final_receipt_is_reloaded_before_lease_release(self):
        queue_dir = Path("/formal/table-d-validation")
        state = {
            "status": "verifying",
            "plan_sha256": "a" * 64,
            "plan": {
                "queue_id": "queue-id",
                "queue_dir": str(queue_dir),
                "training_queue": {"completion_semantic_sha256": "b" * 64},
            },
            "items": [
                {
                    "index": index,
                    "run_id": job_id,
                    "status": "completed",
                    "completion_evidence": {"job_id": job_id},
                }
                for index, job_id in enumerate(queue.JOB_IDS)
            ],
            "events": [],
        }
        verified = [item["completion_evidence"] for item in state["items"]]
        calls = []

        def save_queue(observed):
            calls.append("save")
            self.assertEqual(observed["status"], "completed")
            self.assertIsNotNone(observed["final_verification"])

        def reload_queue(path):
            calls.append("reload")
            self.assertEqual(path, queue_dir)
            self.assertEqual(state["status"], "completed")
            return state

        with patch.object(
            queue, "_verify_full_queue_completion", return_value=verified
        ), patch.object(
            queue,
            "_validate_evaluation_scope_plan",
            return_value={"semantic_sha256": "c" * 64},
        ), patch.object(
            queue.serial_queue, "_ensure_lease"
        ) as ensure, patch.object(
            queue, "_save_queue", side_effect=save_queue
        ), patch.object(
            queue, "load_queue", side_effect=reload_queue
        ), patch.object(
            queue.serial_queue,
            "_clear_owned_lease",
            side_effect=lambda _state: calls.append("release"),
        ):
            queue._advance_final_verification(state)

        self.assertEqual(calls, ["save", "reload", "release"])
        self.assertEqual(ensure.call_count, 3)
        receipt = state["final_verification"]
        self.assertEqual(receipt["schema"], queue.FINAL_VERIFICATION_SCHEMA)
        self.assertEqual(
            receipt["completion_evidence_sha256"], queue._canonical_sha(verified)
        )

    def test_final_receipt_tamper_is_rejected(self):
        verified = [{"job_id": job_id} for job_id in queue.JOB_IDS]
        state = {
            "status": "completed",
            "plan_sha256": "a" * 64,
            "plan": {
                "queue_id": "queue-id",
                "queue_dir": "/formal/table-d-validation",
                "training_queue": {"completion_semantic_sha256": "b" * 64},
            },
            "final_verification": {
                "schema": queue.FINAL_VERIFICATION_SCHEMA,
                "verified_at_utc": "2026-07-19T00:00:00+00:00",
                "queue_id": "queue-id",
                "plan_sha256": "a" * 64,
                "ordered_job_ids": list(queue.JOB_IDS),
                "completion_evidence_sha256": queue._canonical_sha(verified),
                "training_completion_semantic_sha256": "b" * 64,
                "evaluation_scope_plan_semantic_sha256": "c" * 64,
            },
        }
        with patch.object(
            queue,
            "_validate_evaluation_scope_plan",
            return_value={"semantic_sha256": "c" * 64},
        ):
            queue._verify_final_receipt_matches(state, verified)
            state["final_verification"]["completion_evidence_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                queue.TableDValidationQueueError, "differs from live replay"
            ):
                queue._verify_final_receipt_matches(state, verified)

    def test_source_drift_terminates_launched_child_before_failure(self):
        state = {
            "status": "running",
            "plan": {"items": [{"command": ["python"], "process_environment": {}}]},
            "items": [
                {
                    "index": 0,
                    "run_id": "S0:17/final",
                    "status": "launched",
                    "child_pid": 123,
                    "child_process_identity": {"pid": 123},
                }
            ],
            "events": [],
        }
        with patch.object(
            queue, "_load_queue_structural", return_value=state
        ), patch.object(
            queue,
            "_advance_launched",
            side_effect=queue.TableDValidationQueueError("source closure drifted"),
        ), patch.object(
            queue,
            "_terminate_active_processes",
            return_value=[{"status": "terminated", "pid": 123}],
        ) as terminate, patch.object(
            queue, "_owned_lease_present", return_value=True
        ), patch.object(
            queue, "_save_queue"
        ):
            observed = queue.advance_once(Path("/not-read"))
        terminate.assert_called_once_with(state, 0)
        self.assertEqual(observed["status"], "failed")
        self.assertTrue(observed["failure"]["lease_retained_fail_closed"])
        self.assertEqual(
            observed["items"][0]["child_termination"][0]["status"],
            "terminated",
        )

    def test_launch_environment_carries_exact_recovery_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "console.log"
            command = ["python", "evaluate.py"]
            markers = {
                "CUDA_VISIBLE_DEVICES": "0",
                "PIVOT_TABLE_D_VALIDATION_QUEUE_ID": "queue-id",
                "PIVOT_TABLE_D_VALIDATION_JOB_ID": "S0:17/final",
            }
            state = {
                "plan": {
                    "gpu_environment": {"CUDA_VISIBLE_DEVICES": "0"},
                    "items": [
                        {
                            "command": command,
                            "command_sha256": queue._canonical_sha(command),
                            "process_environment": markers,
                            "evaluation_root": str(root / "evaluation"),
                        }
                    ],
                },
                "items": [
                    {
                        "run_id": "S0:17/final",
                        "status": "launching",
                        "console_log": str(log),
                    }
                ],
            }
            process = SimpleNamespace(pid=4321)
            with patch.object(queue, "_verify_sources"), patch.object(
                queue.serial_queue, "_ensure_lease"
            ), patch.object(
                queue, "_matching_processes", return_value=[]
            ) as matching, patch.object(
                queue.subprocess, "Popen", return_value=process
            ) as popen, patch.object(
                queue.serial_queue,
                "_read_process_identity",
                return_value={"pid": 4321},
            ), patch.object(queue, "_bind_process"):
                queue._advance_launching(state, 0)
            queue._LOCAL_PROCESSES.pop(4321, None)
            matching.assert_called_once_with(command, markers)
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
            for key, value in markers.items():
                self.assertEqual(environment[key], value)


if __name__ == "__main__":
    unittest.main()
