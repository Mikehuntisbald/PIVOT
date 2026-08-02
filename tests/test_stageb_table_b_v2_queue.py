import copy
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


from tools import run_stageb_table_b_v2 as runner
from tools import run_stageb_table_b_v2_queue as formal_queue
from util import stage_b_table_b_v2_contract as contract


class TableBV2FormalQueueTest(unittest.TestCase):
    def _create(self, root: Path):
        stage_a = root / "stage_a.pth"
        scorer = root / "scorer.pth"
        stage_a.write_bytes(b"stage-a")
        scorer.write_bytes(b"scorer")
        environment = {
            "PIVOT_STAGE_A_INIT": str(stage_a),
            "PIVOT_SCORER_WARMSTART": str(scorer),
            "PIVOT_DATA_ROOT": os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"),
            "PIVOT_NUM_WORKERS": "0",
            "PIVOT_PREFETCH_FACTOR": "1",
            "PIVOT_OMP_NUM_THREADS": "1",
            "PIVOT_MIN_NOFILE": "0",
            "PIVOT_CUDA_VISIBLE_DEVICES": "0",
            "PIVOT_MP_SHARING_STRATEGY": "none",
        }
        queue_dir = root / "queue"
        output_root = root / "runs"
        with patch.dict(os.environ, environment, clear=False):
            report = formal_queue.create_formal_queue(
                queue_dir,
                output_root=output_root,
                runner_python=Path(sys.executable),
                lease_root=root / "leases",
                gpu_key="0",
            )
        return queue_dir, output_root, report, environment

    def test_create_seals_exact_six_formal_runs_and_common_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, output_root, report, _environment = self._create(root)
            self.assertEqual(report["ordered_run_ids"], list(contract.FORMAL_RUN_IDS))
            self.assertEqual(report["status"], "planned")
            generic = json.loads((queue_dir / "queue.json").read_text())
            self.assertEqual(
                [item["run_id"] for item in generic["plan"]["items"]],
                list(contract.FORMAL_RUN_IDS),
            )
            self.assertTrue(
                all(item["runner"] == "paper" for item in generic["plan"]["items"])
            )
            self.assertEqual(
                generic["plan"]["extensions"]["profile"], contract.FORMAL_PROFILE
            )
            self.assertEqual(
                generic["plan"]["extensions"]["ordered_run_ids"],
                list(contract.FORMAL_RUN_IDS),
            )
            self.assertEqual(
                generic["plan"]["runtime_environment"]["PIVOT_TN_OUTPUT_ROOT"],
                str(output_root),
            )
            self.assertEqual(
                generic["plan"]["runtime_environment"]["PIVOT_BATCH_SIZE"], "40"
            )
            self.assertEqual(
                generic["plan"]["runtime_environment"]["PIVOT_MAX_TRAIN_ITERS"],
                "1000",
            )
            source = contract.validate_formal_source_plan(
                queue_dir / formal_queue.SOURCE_PLAN_NAME
            )
            scope = contract.validate_formal_scope_plan(
                queue_dir / formal_queue.SCOPE_PLAN_NAME
            )
            self.assertTrue(
                source["common_input_contract"]["common_inputs_identical"]
            )
            self.assertEqual(set(scope["runs"]), set(contract.FORMAL_RUN_IDS))
            status = formal_queue.queue_status(queue_dir)
            self.assertEqual(status["generic_queue"]["status"], "planned")
            self.assertFalse(status["generic_queue"]["lease"].get("present", True))

    def test_planned_manifest_replays_scope_and_rejects_runtime_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _output_root, _report, base_environment = self._create(root)
            execution = formal_queue._execution_environment(queue_dir)
            with patch.dict(os.environ, {**base_environment, **execution}, clear=False):
                launcher = runner._launcher()
                runtime = launcher.runtime_from_environment()
                manifest = runner.build_manifest(
                    runtime,
                    launcher.ROW_BY_ID["D2m"],
                    17,
                    launcher.token_launcher.HashCache(),
                )
                scope_plan = contract.validate_formal_scope_plan(
                    queue_dir / formal_queue.SCOPE_PLAN_NAME
                )
                self.assertEqual(manifest["profile"], contract.FORMAL_PROFILE)
                self.assertEqual(
                    manifest["table_b_v2_scope_sha256"],
                    scope_plan["runs"]["D2m:17"]["scope_sha256"],
                )
                self.assertEqual(manifest["equal_budget_contract"]["batch_size"], 40)
                completed = copy.deepcopy(manifest)
                completed["status"] = "completed"
                completed["completed_phases"] = [
                    {
                        "phase_id": "joint",
                        "status": "completed",
                        "profile": manifest["profile"],
                        "formal_queue": manifest["formal_queue"],
                        "table_b_v2_scope_sha256": manifest[
                            "table_b_v2_scope_sha256"
                        ],
                        "v2_provenance": {
                            "phase_id": "joint",
                            "scope_sha256": manifest[
                                "table_b_v2_scope_sha256"
                            ],
                            "profile": manifest["profile"],
                            "queue": manifest["formal_queue"],
                            "source_plan_semantic_sha256": manifest[
                                "table_b_v2_scope"
                            ]["source_plan_semantic_sha256"],
                        },
                    }
                ]
                runner._validate_completed_sequence(completed)
                bad_budget = copy.deepcopy(completed)
                bad_budget["equal_budget_contract"]["batch_size"] = 16
                with self.assertRaisesRegex(
                    runner.TableBV2RunnerError, "not B40/U1000"
                ):
                    runner._validate_completed_sequence(bad_budget)
                bad_runtime = replace(runtime, batch_size=16)
                with self.assertRaisesRegex(
                    runner.TableBV2RunnerError, "runtime differs"
                ):
                    runner.build_manifest(
                        bad_runtime,
                        launcher.ROW_BY_ID["D2m"],
                        17,
                        launcher.token_launcher.HashCache(),
                    )

    def test_source_plan_cannot_reauthorize_a_different_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _output_root, _report, _environment = self._create(root)
            path = queue_dir / formal_queue.SOURCE_PLAN_NAME
            payload = json.loads(path.read_text())
            tampered = copy.deepcopy(payload)
            tampered["queue"]["queue_id"] = "different-queue"
            tampered["semantic_sha256"] = formal_queue._semantic_sha256(tampered)
            formal_queue._write_json_atomic(path, tampered)
            with self.assertRaisesRegex(
                contract.TableBContractError, "generic queue identity/order"
            ):
                contract.validate_formal_source_plan(path)

    def test_completion_attestation_replays_all_six_common_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, output_root, _report, base_environment = self._create(root)
            source_path = queue_dir / formal_queue.SOURCE_PLAN_NAME
            launcher = runner._launcher()
            with patch.dict(os.environ, base_environment, clear=False):
                runtime = launcher.runtime_from_environment()
                cache = launcher.token_launcher.HashCache()
                for run_id in contract.FORMAL_RUN_IDS:
                    table_b_id, raw_seed = run_id.split(":", 1)
                    manifest = runner.build_formal_planned_manifest(
                        runtime,
                        launcher.ROW_BY_ID[table_b_id],
                        int(raw_seed),
                        cache,
                        source_plan_path=source_path,
                    )
                    run_root = output_root / table_b_id / f"seed{raw_seed}"
                    run_root.mkdir(parents=True)
                    formal_queue._write_json_atomic(
                        run_root / "launch_manifest.json", manifest["phases"][0]
                    )

            def verified_run(run_root, **_kwargs):
                run_root = Path(run_root).resolve()
                return {
                    "schema": runner.VERIFICATION_SCHEMA,
                    "status": "passed",
                    "run_id": f"{run_root.parent.name}:{run_root.name.removeprefix('seed')}",
                    "run_root": str(run_root),
                }

            with patch.object(
                formal_queue.generic_queue,
                "verify_queue",
                return_value={"status": "passed"},
            ), patch.object(
                formal_queue.v2_runner,
                "verify_completed_run",
                side_effect=verified_run,
            ):
                attestation = formal_queue.verify_formal_queue(
                    queue_dir, persist=True
                )
                self.assertTrue(
                    attestation["common_input_replay"][
                        "all_six_runs_share_identical_common_inputs"
                    ]
                )
                evidence = formal_queue.formal_evaluation_evidence(
                    queue_dir,
                    run_id="D2m:17",
                    run_root=output_root / "D2m/seed17",
                )
                self.assertEqual(evidence["run_id"], "D2m:17")
                self.assertEqual(
                    evidence["completion_semantic_sha256"],
                    attestation["semantic_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
