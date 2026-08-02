import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_stageb_table_a_g0c_queues as queues


def _fixture_plan(root: Path, queue_kind: str) -> dict:
    queue_dir = root / f"{queue_kind}-queue"
    lease_root = root / "leases"
    python = Path(sys.executable).resolve()
    source = root / "input.txt"
    source.write_text("sealed input\n", encoding="ascii")
    runtime = {
        "python": str(python),
        "python_record": queues._file_record(python, "queue_python_runtime"),
        "data_root": str(root.resolve()),
        "gpu_key": "fixture-gpu",
        "cuda_visible_devices": "fixture-gpu",
        "training": {
            "micro_batch_size": queues.training_runner.FORMAL_MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": (
                queues.training_runner.FORMAL_GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_global_batch": (
                queues.training_runner.REQUIRED_EFFECTIVE_GLOBAL_BATCH
            ),
            "optimizer_updates": queues.training_runner.FORMAL_OPTIMIZER_UPDATES,
            "num_workers": 8,
        },
        "evaluation": {
            "device": queues.evaluator.FORMAL_EVAL_DEVICE,
            "batch_size": queues.evaluator.FORMAL_EVAL_BATCH_SIZE,
            "num_workers": queues.evaluator.FORMAL_EVAL_NUM_WORKERS,
            "amp": True,
            "eval_seed": queues.evaluator.EVAL_SEED,
        },
    }
    items = []
    run_ids = queues._expected_run_ids(queue_kind)
    for index, run_id in enumerate(run_ids):
        seed = queues.FORMAL_SEEDS[index % len(queues.FORMAL_SEEDS)]
        output = root / "outputs" / run_id.replace(":", "-")
        if queue_kind == queues.TRAINING_KIND:
            expected_plan = {
                "schema": queues.training_runner.PLAN_SCHEMA,
                "row_id": "G0c",
                "purpose": "formal",
                "output_dir": str(output),
                "matched_contract": {
                    "seed": seed,
                    "micro_batch_size_per_rank": (
                        queues.training_runner.FORMAL_MICRO_BATCH_SIZE
                    ),
                    "gradient_accumulation_steps": (
                        queues.training_runner.FORMAL_GRADIENT_ACCUMULATION_STEPS
                    ),
                    "effective_global_batch": (
                        queues.training_runner.REQUIRED_EFFECTIVE_GLOBAL_BATCH
                    ),
                    "optimizer_updates": (
                        queues.training_runner.FORMAL_OPTIMIZER_UPDATES
                    ),
                },
                "inputs": {
                    "fixture": {
                        "path": str(source.resolve()),
                        "sha256": queues._sha256_file(source),
                    }
                },
                "source_dependency_tree": {
                    "records": [
                        {
                            "path": str(source.resolve()),
                            "sha256": queues._sha256_file(source),
                        }
                    ]
                },
            }
            expected_plan["plan_sha256"] = queues.training_runner._plan_sha256(
                expected_plan
            )
            item = {
                "index": index,
                "run_id": run_id,
                "item_kind": "training",
                "seed": seed,
                "output_root": str(output),
                "formal_plan_path": str(root / f"formal-{seed}.json"),
                "expected_plan": expected_plan,
                "expected_plan_sha256": expected_plan["plan_sha256"],
                "command": [
                    str(python),
                    str(Path(queues.training_runner.__file__).resolve()),
                    "fixture",
                    str(seed),
                ],
            }
            item["input_records"] = queues._records_from_training_plan(
                expected_plan
            )
        else:
            kind = "candidate" if index < len(queues.FORMAL_SEEDS) else "g0c"
            profile = (
                queues.evaluator.FINAL_PROFILE
                if queue_kind == queues.FINAL_KIND
                else queues.evaluator.VALIDATION_PROFILE
            )
            instance_sha256 = f"{index + 11:064x}"
            input_records = [queues._file_record(source, "fixture_input")]
            evaluation_plan = queues._stable_evaluation_plan(
                {
                    "schema": queues.evaluator.SCHEMA,
                    "kind": kind,
                    "profile": profile,
                    "repository_root": str(queues.REPO_ROOT),
                    "evaluation_id": f"{kind}_seed{seed}",
                    "output_dir": str(output),
                    "source": {"kind": "fixture"},
                    "runtime": {"device": queues.evaluator.FORMAL_EVAL_DEVICE},
                    "instance": {
                        "seed": seed,
                        "instance_sha256": instance_sha256,
                    },
                    "final_gate": (
                        {
                            "path": str(
                                queues.evaluator.FINAL_GATE_PATH.resolve(
                                    strict=False
                                )
                            ),
                            "sha256": "f" * 64,
                        }
                        if queue_kind == queues.FINAL_KIND
                        else None
                    ),
                    "contract": {"fixture": True},
                    "tn_manifest": {"fixture": True},
                    "tn_inputs": {},
                    "commands": [],
                    "inputs": {"records": input_records},
                }
            )
            item = {
                "index": index,
                "run_id": run_id,
                "item_kind": "evaluation",
                "evaluation_kind": kind,
                "evaluation_profile": profile,
                "seed": seed,
                "output_root": str(output),
                "instance_sha256": instance_sha256,
                "evaluation_plan_contract": evaluation_plan,
                "command": [
                    str(python),
                    str(Path(queues.evaluator.__file__).resolve()),
                    "fixture",
                    kind,
                    str(seed),
                ],
            }
            item["input_records"] = input_records
            if queue_kind == queues.FINAL_KIND:
                item["final_consumption_path"] = str(
                    queues.evaluator._final_consumption_path(
                        evaluation_plan["instance"]
                    )
                )
        items.append(item)
    plan = {
        "schema": queues.PLAN_SCHEMA,
        "queue_kind": queue_kind,
        "queue_id": f"fixture-{queue_kind}",
        "created_at_utc": "2026-07-19T00:00:00+00:00",
        "queue_dir": str(queue_dir),
        "repository_root": str(queues.REPO_ROOT),
        "runtime": runtime,
        "runtime_environment": queues._runtime_environment(runtime),
        "gpu_key": runtime["gpu_key"],
        "lease_root": str(lease_root),
        "lease_path": str(
            queues.shared_queue._lease_path(lease_root, runtime["gpu_key"])
        ),
        "controller_sources": queues._controller_source_records(),
        "items": items,
    }
    if queue_kind == queues.FINAL_KIND:
        predecessor = root / "validation-queue.json"
        predecessor.write_text("{}\n", encoding="ascii")
        plan["predecessor_validation_queue"] = {
            "queue_dir": str(queues.DEFAULT_VALIDATION_QUEUE_DIR.resolve(strict=False)),
            "queue_id": "fixture-validation-queue",
            "plan_sha256": "e" * 64,
            "ordered_run_ids": list(queues.VALIDATION_RUN_IDS),
            "queue_manifest": queues._file_record(
                predecessor, "table_a_validation_queue_predecessor"
            ),
        }
    return plan


class G0cQueueTest(unittest.TestCase):
    def test_plan_requires_exact_order_and_item_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _fixture_plan(root, queues.VALIDATION_KIND)
            plan["items"][0]["run_id"] = "g0c:17"
            with self.assertRaisesRegex(queues.G0cQueueError, "item order"):
                queues.create_queue_from_plan(plan)

            plan = _fixture_plan(root, queues.VALIDATION_KIND)
            plan["items"][0]["seed"] = 42
            with self.assertRaisesRegex(queues.G0cQueueError, "contract drifted"):
                queues.create_queue_from_plan(plan)

            plan = _fixture_plan(root, queues.VALIDATION_KIND)
            plan["controller_sources"].pop()
            queues.create_queue_from_plan(plan)
            failed = queues.advance_once(Path(plan["queue_dir"]))
            self.assertEqual(failed["status"], "failed")
            self.assertIn("closure is incomplete", failed["failure"]["error"])

    def test_reserve_and_prepare_bind_shared_lease_and_exact_job_without_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _fixture_plan(root, queues.TRAINING_KIND)
            created = queues.create_queue_from_plan(plan)
            with mock.patch.object(queues.subprocess, "Popen") as spawn:
                reserved = queues.advance_once(Path(plan["queue_dir"]))
                prepared = queues.advance_once(Path(plan["queue_dir"]))
            spawn.assert_not_called()

            lease_path = Path(created["plan"]["lease_path"])
            lease = json.loads(lease_path.read_text(encoding="utf-8"))
            self.assertEqual(lease["queue_id"], plan["queue_id"])
            self.assertEqual(lease["first_run_id"], "G0c:17")
            self.assertEqual(reserved["items"][0]["status"], "reserved")
            self.assertEqual(prepared["items"][0]["status"], "launching")
            self.assertEqual(
                prepared["active_item"],
                {
                    key: prepared["items"][0][key]
                    for key in ("index", "run_id", "job_id", "job_dir")
                },
            )
            job_dir = Path(prepared["items"][0]["job_dir"])
            launch = json.loads((job_dir / "launch.json").read_text(encoding="utf-8"))
            status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
            for value in (launch, status):
                self.assertEqual(value["queue_id"], plan["queue_id"])
                self.assertEqual(value["job_id"], prepared["items"][0]["job_id"])
                self.assertEqual(value["run_id"], "G0c:17")
            self.assertEqual(launch["command"], plan["items"][0]["command"])
            Path(plan["items"][0]["formal_plan_path"]).write_text(
                "{}\n", encoding="ascii"
            )
            with mock.patch.object(queues.subprocess, "Popen") as spawn:
                failed = queues.advance_once(Path(plan["queue_dir"]))
            spawn.assert_not_called()
            self.assertEqual(failed["status"], "failed")
            self.assertIn(
                "without a recoverable exact child identity",
                failed["failure"]["error"],
            )
            self.assertTrue(failed["failure"]["lease_retained_fail_closed"])

    def test_active_item_tamper_and_input_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _fixture_plan(root, queues.TRAINING_KIND)
            queues.create_queue_from_plan(plan)
            reserved = queues.advance_once(Path(plan["queue_dir"]))
            manifest = Path(plan["queue_dir"]) / "queue.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["active_item"]["job_id"] = "forged"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(queues.G0cQueueError, "active-item"):
                queues.load_queue(Path(plan["queue_dir"]))
            self.assertTrue(Path(reserved["plan"]["lease_path"]).is_file())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _fixture_plan(root, queues.TRAINING_KIND)
            queues.create_queue_from_plan(plan)
            Path(plan["items"][0]["input_records"][0]["path"]).write_text(
                "drifted input bytes\n", encoding="ascii"
            )
            failed = queues.advance_once(Path(plan["queue_dir"]))
            self.assertEqual(failed["status"], "failed")
            self.assertIn("changed after queue planning", failed["failure"]["error"])
            self.assertFalse(Path(plan["lease_path"]).exists())
            self.assertFalse(failed["failure"]["lease_retained_fail_closed"])

    def test_training_completion_replays_plan_postflight_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "training"
            output.mkdir()
            plan_path = root / "formal.json"
            checkpoint = output / "checkpoint_iter.pth"
            checkpoint.write_bytes(b"checkpoint")
            expected_plan = {"plan_sha256": "a" * 64, "purpose": "formal"}
            plan_path.write_text(json.dumps(expected_plan), encoding="utf-8")
            replay = {
                "status": "PASS",
                "purpose": "formal",
                "seed": 17,
                "plan_sha256": "a" * 64,
                "optimizer_updates": 1000,
                "validated_at_utc": "fresh",
            }
            persisted = {**replay, "validated_at_utc": "persisted"}
            postflight = output / "postflight.json"
            postflight.write_text(json.dumps(persisted), encoding="utf-8")
            planned = {
                "seed": 17,
                "output_root": str(output),
                "formal_plan_path": str(plan_path),
                "expected_plan": expected_plan,
                "expected_plan_sha256": "a" * 64,
            }
            queue = {"plan": {"items": [planned]}}
            item = {"index": 0, "run_id": "G0c:17"}
            with (
                mock.patch.object(
                    queues.training_runner,
                    "formal_plan_path",
                    return_value=plan_path,
                ),
                mock.patch.object(queues.training_runner, "_validate_plan_identity"),
                mock.patch.object(
                    queues.training_runner,
                    "verify_checkpoint",
                    return_value=replay,
                ),
            ):
                evidence = queues._verify_training_completion(queue, item)
                self.assertEqual(evidence["checkpoint"]["path"], str(checkpoint))
                forged = {**persisted, "optimizer_updates": 999}
                postflight.write_text(json.dumps(forged), encoding="utf-8")
                with self.assertRaisesRegex(
                    queues.G0cQueueError, "differs from replay"
                ):
                    queues._verify_training_completion(queue, item)

    def test_validation_completion_replays_predeclared_launch_and_postflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "evaluation"
            output.mkdir()
            launch = {
                "schema": queues.evaluator.SCHEMA,
                "status": "completed",
                "kind": "candidate",
                "profile": queues.evaluator.VALIDATION_PROFILE,
                "repository_root": str(queues.REPO_ROOT),
                "evaluation_id": "L4_seed17",
                "output_dir": str(output),
                "source": {"kind": "fixture"},
                "runtime": {"device": "cuda:0"},
                "instance": {"seed": 17, "instance_sha256": "b" * 64},
                "final_gate": None,
                "contract": {"fixture": True},
                "tn_manifest": {"fixture": True},
                "tn_inputs": {},
                "commands": [],
                "inputs": {"records": []},
            }
            (output / "launch_manifest.json").write_text(
                json.dumps(launch), encoding="utf-8"
            )
            replay = {"status": "passed", "validated_at_utc": "fresh"}
            postflight = output / "postflight.json"
            postflight.write_text(
                json.dumps({**replay, "validated_at_utc": "persisted"}),
                encoding="utf-8",
            )
            planned = {
                "output_root": str(output),
                "evaluation_kind": "candidate",
                "seed": 17,
                "instance_sha256": "b" * 64,
                "evaluation_plan_contract": queues._stable_evaluation_plan(launch),
            }
            queue = {
                "plan": {
                    "queue_kind": queues.VALIDATION_KIND,
                    "items": [planned],
                }
            }
            item = {"index": 0, "run_id": "candidate:17"}
            with mock.patch.object(
                queues.evaluator, "postflight", return_value=replay
            ):
                evidence = queues._verify_evaluation_completion(queue, item)
                self.assertEqual(evidence["run_id"], "candidate:17")
                changed = copy.deepcopy(launch)
                changed["source"] = {"kind": "relabeled"}
                (output / "launch_manifest.json").write_text(
                    json.dumps(changed), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    queues.G0cQueueError, "predeclared plan"
                ):
                    queues._verify_evaluation_completion(queue, item)

    def test_final_completion_binds_gate_and_single_use_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "evaluation"
            output.mkdir()
            gate = root / "gate.json"
            gate.write_text("{}\n", encoding="ascii")
            consumption = root / "consumption.json"
            consumption.write_text("{}\n", encoding="ascii")
            consumption_record = {
                "path": str(consumption.resolve()),
                "sha256": queues._sha256_file(consumption),
            }
            launch = {
                "schema": queues.evaluator.SCHEMA,
                "status": "completed",
                "kind": "candidate",
                "profile": queues.evaluator.FINAL_PROFILE,
                "repository_root": str(queues.REPO_ROOT),
                "evaluation_id": "L4_seed17",
                "output_dir": str(output),
                "source": {"kind": "fixture"},
                "runtime": {"device": "cuda:0"},
                "instance": {"seed": 17, "instance_sha256": "b" * 64},
                "final_gate": {
                    "path": str(gate.resolve()),
                    "sha256": queues._sha256_file(gate),
                },
                "final_consumption": consumption_record,
                "contract": {"fixture": True},
                "tn_manifest": {"fixture": True},
                "tn_inputs": {},
                "commands": [],
                "inputs": {"records": []},
            }
            (output / "launch_manifest.json").write_text(
                json.dumps(launch), encoding="utf-8"
            )
            replay = {
                "status": "passed",
                "verified_at_utc": "fresh",
                "final_consumption": consumption_record,
            }
            (output / "postflight.json").write_text(
                json.dumps({**replay, "verified_at_utc": "persisted"}),
                encoding="utf-8",
            )
            planned = {
                "output_root": str(output),
                "evaluation_kind": "candidate",
                "evaluation_profile": queues.evaluator.FINAL_PROFILE,
                "seed": 17,
                "instance_sha256": "b" * 64,
                "final_consumption_path": str(consumption.resolve()),
                "evaluation_plan_contract": queues._stable_evaluation_plan(launch),
            }
            queue = {
                "plan": {
                    "queue_kind": queues.FINAL_KIND,
                    "items": [planned],
                }
            }
            item = {"index": 0, "run_id": "candidate:17"}
            with (
                mock.patch.object(
                    queues.evaluator, "postflight", return_value=replay
                ),
                mock.patch.object(
                    queues.evaluator,
                    "_validate_final_consumption",
                    return_value=consumption_record,
                ),
            ):
                evidence = queues._verify_evaluation_completion(queue, item)
            self.assertEqual(evidence["queue_kind"], queues.FINAL_KIND)
            self.assertEqual(
                evidence["final_consumption"]["path"], str(consumption.resolve())
            )
            self.assertEqual(evidence["final_gate"]["path"], str(gate.resolve()))

    def test_final_queue_plan_is_exact_fresh_and_does_not_consume_or_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                queues.evaluator,
                "FINAL_CONSUMPTION_ROOT",
                root / "consumptions",
            ):
                plan = _fixture_plan(root, queues.FINAL_KIND)
                created = queues.create_queue_from_plan(plan)
            self.assertEqual(
                [item["run_id"] for item in created["items"]],
                list(queues.FINAL_RUN_IDS),
            )
            self.assertEqual(
                {
                    item["evaluation_profile"]
                    for item in created["plan"]["items"]
                },
                {queues.evaluator.FINAL_PROFILE},
            )
            self.assertFalse(Path(plan["lease_path"]).exists())
            self.assertFalse((root / "consumptions").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                queues.evaluator,
                "FINAL_CONSUMPTION_ROOT",
                root / "consumptions",
            ):
                plan = _fixture_plan(root, queues.FINAL_KIND)
                plan["items"][0]["evaluation_profile"] = (
                    queues.evaluator.VALIDATION_PROFILE
                )
                with self.assertRaisesRegex(
                    queues.G0cQueueError, "planned final item"
                ):
                    queues.create_queue_from_plan(plan)

    def test_final_item_builder_predeclares_gate_profile_and_exact_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_queue = root / "training-queue"
            training_queue.mkdir()
            gate = root / "gate.json"
            gate.write_text("{}\n", encoding="ascii")
            candidate_queues = {}
            locked = {}
            for seed in queues.FORMAL_SEEDS:
                path = root / f"candidate-queue-{seed}"
                path.mkdir()
                candidate_queues[seed] = path
                locked[seed] = {"path": str(path)}
            runtime = {
                "python": str(Path(sys.executable).resolve()),
                "data_root": str(root.resolve()),
                "evaluation": {
                    "device": queues.evaluator.FORMAL_EVAL_DEVICE,
                    "batch_size": queues.evaluator.FORMAL_EVAL_BATCH_SIZE,
                    "num_workers": queues.evaluator.FORMAL_EVAL_NUM_WORKERS,
                    "amp": True,
                },
            }

            def planned(kind, seed, output):
                instance = {
                    "seed": seed,
                    "instance_sha256": f"{seed + (100 if kind == 'g0c' else 0):064x}",
                }
                return {
                    "schema": queues.evaluator.SCHEMA,
                    "kind": kind,
                    "profile": queues.evaluator.FINAL_PROFILE,
                    "repository_root": str(queues.REPO_ROOT),
                    "evaluation_id": f"{kind}_{seed}",
                    "output_dir": str(output),
                    "source": {
                        "training_queue_id": f"{kind}-queue",
                        "training_queue_plan_sha256": "a" * 64,
                    },
                    "runtime": {},
                    "instance": instance,
                    "final_gate": {"path": str(gate), "sha256": "b" * 64},
                    "contract": {},
                    "tn_manifest": {},
                    "tn_inputs": {},
                    "commands": [],
                    "inputs": {
                        "records": [
                            queues._file_record(gate, "table_a_final_gate")
                        ]
                    },
                }

            def candidate_plan(_runtime, _root, output, **_kwargs):
                seed = int(str(output).rsplit("seed", 1)[1])
                return planned("candidate", seed, output)

            def g0c_plan(_runtime, _plan, output, **_kwargs):
                seed = int(str(output).rsplit("seed", 1)[1])
                return planned("g0c", seed, output)

            with (
                mock.patch.object(
                    queues,
                    "verify_queue",
                    return_value={
                        "status": "passed",
                        "queue_kind": queues.TRAINING_KIND,
                        "ordered_run_ids": list(queues.TRAINING_RUN_IDS),
                    },
                ),
                mock.patch.object(
                    queues,
                    "_completed_validation_queue_dependency",
                    return_value={"queue_id": "validation"},
                ),
                mock.patch.object(queues.evaluator, "FINAL_GATE_PATH", gate),
                mock.patch.object(
                    queues.evaluator, "LOCKED_CANDIDATE_QUEUES", locked
                ),
                mock.patch.object(
                    queues.evaluator,
                    "canonical_output_dir",
                    side_effect=lambda kind, profile, seed: (
                        root / "outputs" / profile / kind / f"seed{seed}"
                    ),
                ),
                mock.patch.object(
                    queues.evaluator,
                    "build_candidate_plan",
                    side_effect=candidate_plan,
                ),
                mock.patch.object(
                    queues.evaluator, "build_g0c_plan", side_effect=g0c_plan
                ),
                mock.patch.object(
                    queues.training_runner,
                    "formal_plan_path",
                    side_effect=lambda seed: root / f"g0c-{seed}.json",
                ),
                mock.patch.object(
                    queues,
                    "_candidate_training_root",
                    side_effect=lambda seed: root / f"candidate-{seed}",
                ),
                mock.patch.object(
                    queues.evaluator,
                    "FINAL_CONSUMPTION_ROOT",
                    root / "consumptions",
                ),
            ):
                items, predecessor = queues._build_evaluation_items(
                    runtime,
                    profile=queues.evaluator.FINAL_PROFILE,
                    training_queue_dir=training_queue,
                    final_gate=gate,
                )
            self.assertEqual(
                [item["run_id"] for item in items], list(queues.FINAL_RUN_IDS)
            )
            self.assertEqual(predecessor, {"queue_id": "validation"})
            for item in items:
                self.assertEqual(
                    item["evaluation_profile"], queues.evaluator.FINAL_PROFILE
                )
                command = item["command"]
                self.assertEqual(
                    command[command.index("--final-gate") + 1], str(gate.resolve())
                )

    def test_consumed_incomplete_final_never_relaunches_and_retains_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                queues.evaluator,
                "FINAL_CONSUMPTION_ROOT",
                root / "consumptions",
            ):
                plan = _fixture_plan(root, queues.FINAL_KIND)
                queue_dir = Path(plan["queue_dir"])
                queues.create_queue_from_plan(plan)
                with mock.patch.object(queues, "_verify_plan_closure"):
                    queues.advance_once(queue_dir)
                    prepared = queues.advance_once(queue_dir)
                    consumption = Path(
                        plan["items"][0]["final_consumption_path"]
                    )
                    consumption.parent.mkdir(parents=True)
                    consumption.write_text("{}\n", encoding="ascii")
                    with (
                        mock.patch.object(
                            queues, "_matching_processes", return_value=[]
                        ),
                        mock.patch.object(queues.subprocess, "Popen") as spawn,
                    ):
                        failed = queues.advance_once(queue_dir)
                spawn.assert_not_called()
                self.assertEqual(prepared["items"][0]["status"], "launching")
                self.assertEqual(failed["status"], "failed")
                self.assertIn("rerun is forbidden", failed["failure"]["error"])
                self.assertTrue(failed["failure"]["lease_retained_fail_closed"])
                self.assertTrue(Path(plan["lease_path"]).is_file())

    def test_completed_final_orphan_is_adopted_without_relaunch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                queues.evaluator,
                "FINAL_CONSUMPTION_ROOT",
                root / "consumptions",
            ):
                plan = _fixture_plan(root, queues.FINAL_KIND)
                queue_dir = Path(plan["queue_dir"])
                queues.create_queue_from_plan(plan)
                with mock.patch.object(queues, "_verify_plan_closure"):
                    queues.advance_once(queue_dir)
                    queues.advance_once(queue_dir)
                    output = Path(plan["items"][0]["output_root"])
                    output.mkdir(parents=True)
                    (output / "launch_manifest.json").write_text(
                        json.dumps({"status": "completed"}), encoding="utf-8"
                    )
                    consumption = Path(
                        plan["items"][0]["final_consumption_path"]
                    )
                    consumption.parent.mkdir(parents=True)
                    consumption.write_text("{}\n", encoding="ascii")
                    with (
                        mock.patch.object(
                            queues, "_matching_processes", return_value=[]
                        ),
                        mock.patch.object(
                            queues,
                            "_verify_evaluation_completion",
                            return_value={"status": "passed"},
                        ) as replay,
                        mock.patch.object(queues.subprocess, "Popen") as spawn,
                    ):
                        recovered = queues.advance_once(queue_dir)
                spawn.assert_not_called()
                replay.assert_called_once()
                item = recovered["items"][0]
                self.assertEqual(item["status"], "launched")
                self.assertEqual(item["child_pid"], 0)
                self.assertTrue(item["completed_orphan_recovery"])
                self.assertEqual(
                    item["child_process_identity"],
                    queues.COMPLETED_ORPHAN_PROCESS_IDENTITY,
                )

    def test_final_release_reloads_and_replays_all_six_before_lease_clear(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            queue_dir.mkdir()
            lease_path = root / "gpu.lease.json"
            lease_path.write_text(
                json.dumps({"queue_id": "final-queue"}), encoding="utf-8"
            )
            items = [
                {
                    "index": index,
                    "run_id": run_id,
                    "status": "completed",
                    "completion_evidence": {"run_id": run_id},
                }
                for index, run_id in enumerate(queues.FINAL_RUN_IDS)
            ]
            persisted = {
                "status": "completed",
                "plan": {
                    "queue_dir": str(queue_dir),
                    "queue_id": "final-queue",
                    "lease_path": str(lease_path),
                },
                "items": items,
            }
            calls = []

            def validate(_queue, item, *, terminal):
                self.assertTrue(terminal)
                calls.append(f"validate:{item['run_id']}")
                return {}, {}

            def replay(_queue, item):
                calls.append(f"replay:{item['run_id']}")
                return item["completion_evidence"]

            with (
                mock.patch.object(queues, "load_queue", return_value=persisted),
                mock.patch.object(
                    queues,
                    "_verify_plan_closure",
                    side_effect=lambda _queue: calls.append("closure"),
                ),
                mock.patch.object(
                    queues, "_validate_job_binding", side_effect=validate
                ),
                mock.patch.object(
                    queues, "_completion_evidence", side_effect=replay
                ),
                mock.patch.object(
                    queues,
                    "_clear_lease",
                    side_effect=lambda _queue: calls.append("release"),
                ) as clear,
            ):
                queues._release_completed_queue_lease(persisted)
            clear.assert_called_once_with(persisted)
            self.assertEqual(calls[0], "closure")
            self.assertEqual(calls[-1], "release")
            self.assertEqual(
                [value for value in calls if value.startswith("replay:")],
                [f"replay:{run_id}" for run_id in queues.FINAL_RUN_IDS],
            )

            def drifted_replay(_queue, item):
                if item["run_id"] == queues.FINAL_RUN_IDS[-1]:
                    return {"run_id": "drifted"}
                return item["completion_evidence"]

            with (
                mock.patch.object(queues, "load_queue", return_value=persisted),
                mock.patch.object(queues, "_verify_plan_closure"),
                mock.patch.object(queues, "_validate_job_binding"),
                mock.patch.object(
                    queues,
                    "_completion_evidence",
                    side_effect=drifted_replay,
                ),
                mock.patch.object(queues, "_clear_lease") as clear_drifted,
                self.assertRaisesRegex(
                    queues.G0cQueueError, "drifted before lease release"
                ),
            ):
                queues._release_completed_queue_lease(persisted)
            clear_drifted.assert_not_called()

    def test_completed_queue_verify_ignores_a_later_queues_shared_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lease = root / "gpu.lease.json"
            lease.write_text(
                json.dumps({"queue_id": "later-queue"}), encoding="utf-8"
            )
            items = [
                {
                    "index": index,
                    "run_id": run_id,
                    "status": "completed",
                    "completion_evidence": {"run_id": run_id},
                }
                for index, run_id in enumerate(queues.FINAL_RUN_IDS)
            ]
            persisted = {
                "status": "completed",
                "plan_sha256": "a" * 64,
                "plan": {
                    "queue_kind": queues.FINAL_KIND,
                    "queue_id": "completed-queue",
                    "lease_path": str(lease),
                },
                "items": items,
            }
            with (
                mock.patch.object(queues, "load_queue", return_value=persisted),
                mock.patch.object(queues, "_verify_plan_closure"),
                mock.patch.object(queues, "_validate_job_binding"),
                mock.patch.object(
                    queues,
                    "_completion_evidence",
                    side_effect=lambda _queue, item: item[
                        "completion_evidence"
                    ],
                ),
            ):
                result = queues.verify_queue(root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["errors"], [])

    def test_artifact_audit_is_read_only_and_never_adopts_legacy_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plans = root / "outputs/paper_cvpr_v1/plans"
            output = root / "legacy-output"
            plans.mkdir(parents=True)
            output.mkdir()
            (plans / "table_a_g0c_u50_seed17.json").write_text(
                json.dumps(
                    {
                        "schema": "legacy",
                        "output_dir": str(output),
                        "inputs": {},
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(queues, "REPO_ROOT", root),
                mock.patch.object(
                    queues.training_runner,
                    "formal_output_root",
                    side_effect=lambda seed: root / f"formal/{seed}",
                ),
                mock.patch.object(
                    queues.training_runner,
                    "formal_plan_path",
                    side_effect=lambda seed: root / f"formal/{seed}.json",
                ),
                mock.patch.object(
                    queues.training_runner,
                    "DEFAULT_SOAK_PLAN",
                    root / "soak-plan.json",
                ),
                mock.patch.object(
                    queues.training_runner,
                    "DEFAULT_SOAK_ROOT",
                    root / "soak",
                ),
                mock.patch.object(
                    queues.training_runner,
                    "DEFAULT_SOAK_SEAL",
                    root / "soak-seal.json",
                ),
                mock.patch.object(
                    queues.training_runner,
                    "_validate_soak_seal",
                    side_effect=FileNotFoundError("seal absent"),
                ),
            ):
                audit = queues.audit_existing_artifacts()
            self.assertEqual(audit["legacy_artifacts"][0]["status"], "non_adoptable")
            self.assertTrue(audit["legacy_artifacts"][0]["output_root_exists"])
            self.assertFalse(audit["canonical_soak"]["ready"])
            self.assertFalse(audit["training_queue_ready"])
            self.assertEqual(list(output.iterdir()), [])

    def test_dry_run_reports_blocker_without_mutation(self):
        audit = {"status": "passed", "training_queue_ready": False}
        with (
            mock.patch.object(
                queues,
                "build_queue_plan",
                side_effect=queues.G0cQueueError("sealed soak is incomplete"),
            ),
            mock.patch.object(queues, "audit_existing_artifacts", return_value=audit),
        ):
            result = queues.dry_run(queues.TRAINING_KIND)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["mutated"])
        self.assertEqual(result["artifact_audit"], audit)


if __name__ == "__main__":
    unittest.main()
