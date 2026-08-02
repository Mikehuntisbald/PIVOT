import copy
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import torch

from tools import run_stageb_headline_m0 as runner


class StageBHeadlineM0ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m0 = runpy.run_path(
            str(runner.REPO_ROOT / runner.source_contracts.M0_CONTRACT.config)
        )
        cls.m0n = runpy.run_path(
            str(runner.REPO_ROOT / runner.source_contracts.M0N_CONTRACT.config)
        )

    def test_m0n_changes_the_full_token_objective_and_retains_pair_rank(self):
        self.assertEqual(self.m0["stage_b_v21_token_objective"], "edit_bce")
        self.assertEqual(
            self.m0n["stage_b_v21_token_objective"],
            "targetlocal_allneg_bce",
        )
        self.assertEqual(self.m0n["stage_b_v11_predicate_tn_rank_weight"], 1.0)
        self.assertEqual(
            self.m0n["stage_b_v25_token_objective_scope"],
            "target_local_positive_and_all_negative_token_logits",
        )
        self.assertEqual(
            self.m0n["stage_b_v25_comparison_claim"],
            "full_token_objective_control_not_labels_only",
        )
        self.assertFalse(self.m0n["stage_b_v25_headline_eligible"])
        self.assertTrue(self.m0n["stage_b_v25_matrix_validation_only"])

    def test_m0n_inherits_every_noncontrol_training_knob(self):
        allowed = {
            "stage_b_v25_main_id",
            "stage_b_v25_control_of",
            "stage_b_v25_headline_eligible",
            "stage_b_v25_matrix_validation_only",
            "stage_b_v25_comparison_claim",
            "stage_b_v25_token_objective_scope",
            "stage_b_v21_token_objective",
        }
        left = {
            key: value
            for key, value in self.m0.items()
            if not key.startswith("__") and key not in allowed
        }
        right = {
            key: value
            for key, value in self.m0n.items()
            if not key.startswith("__") and key not in allowed
        }
        self.assertEqual(left, right)

    def test_runner_replays_both_leaf_contracts(self):
        for contract in runner.CONTRACTS.values():
            with self.subTest(contract=contract.id):
                values = runner._validate_config(contract)
                self.assertEqual(values["batch_size"], 40)
                self.assertEqual(
                    values["stage_b_v25_successful_update_batch_slots"],
                    941_280,
                )
                self.assertEqual(
                    Path(values["stage_b_v15_scorer_init_checkpoint"]).resolve(),
                    runner.DEFAULT_STAGE_A_INIT,
                )


class StageBHeadlineM0RunnerTest(unittest.TestCase):
    def _runtime(self, root: Path) -> runner.Runtime:
        root.mkdir(parents=True, exist_ok=True)
        stage_a = root / "checkpoint0004.pth"
        stage_a.write_bytes(b"stage-a-test")
        return runner.Runtime(
            python=Path(sys.executable).resolve(),
            stage_a_init=stage_a.resolve(),
            dataset=runner.DEFAULT_DATASET.resolve(),
            output_root=root / "outputs",
            data_root=runner.DEFAULT_DATA_ROOT.resolve(),
            batch_size=40,
            max_train_iters=23_532,
            iter_checkpoint_interval=500,
            num_workers=2,
            prefetch_factor=1,
            omp_num_threads=8,
            min_nofile=65_536,
            cuda_visible_devices="0",
            mp_sharing_strategy="file_system",
            gradient_diagnostic_interval=100,
            telemetry_interval_seconds=1,
            pin_memory=True,
            persistent_workers=False,
            gradient_accumulation_steps=1,
        )

    def _checkpoint_args(
        self,
        runtime: runner.Runtime,
        contract,
        output: Path,
        *,
        resume: Path | None = None,
    ) -> dict:
        values = {
            "seed": 17,
            "batch_size": 40,
            "max_train_iters": 23_532,
            "iter_checkpoint_interval": 500,
            "gradient_accumulation_steps": 1,
            "num_workers": 2,
            "prefetch_factor": 1,
            "pin_memory": True,
            "persistent_workers": False,
            "world_size": 1,
            "distributed": False,
            "config_file": str(runner._config_path(contract)),
            "datasets": str(runtime.dataset),
            "output_dir": str(output),
            "pretrain_model_path": (
                "" if resume is not None else str(runtime.stage_a_init)
            ),
            "resume": str(resume) if resume is not None else "",
            "stage_b_v15_scorer_init_checkpoint": str(runtime.stage_a_init),
            "stage_b_v25_main_id": contract.id,
            "stage_b_v25_compute_contract": (
                "b58_successful_update_batch_slot_matched"
            ),
            "stage_b_v25_budget_unit": (
                "successful_optimizer_update_global_batch_slots"
            ),
            "stage_b_v25_successful_update_batch_slots": 941_280,
            "stage_b_v25_initializer_contract": (
                "same_stage_a_model_and_scorer_no_b58"
            ),
            "stage_b_v25_strict_resume": True,
            "stage_b_v22_table_id": "S2F",
            "stage_b_v22_objective_fidelity": (
                "full_v19_base_plus_gate_objective"
            ),
            "stage_b_v22_gradient_diagnostic_interval": 100,
            "stage_b_v15_separate_grad_clip": True,
            "stage_b_v21_token_objective": (
                "edit_bce" if contract.id == "M0" else "targetlocal_allneg_bce"
            ),
            "stage_b_v21_token_weight": 1.0,
            "stage_b_v21_token_positive_weight": 1.0,
            "stage_b_v21_token_shared_weight": 0.25,
            "stage_b_v21_token_edit_weight": 1.0,
            "stage_b_v11_predicate_tn_rank_weight": 1.0,
            "stage_b_v21_allow_legacy_token_diff_fallback": False,
            "skip_eval": True,
            "amp": True,
        }
        if contract.id == "M0N":
            values.update(
                {
                    "stage_b_v25_control_of": "M0",
                    "stage_b_v25_headline_eligible": False,
                    "stage_b_v25_matrix_validation_only": True,
                    "stage_b_v25_comparison_claim": (
                        "full_token_objective_control_not_labels_only"
                    ),
                    "stage_b_v25_token_objective_scope": (
                        "target_local_positive_and_all_negative_token_logits"
                    ),
                }
            )
        return values

    def _metadata(
        self,
        runtime: runner.Runtime,
        contract,
        output: Path,
        *,
        updates: int,
        epoch: int,
        iteration: int,
        reason: str,
        epoch_finished: bool = False,
        resume: Path | None = None,
    ) -> dict:
        return {
            "complete_state_components": dict(runner.COMPLETE_STATE_COMPONENTS),
            "optimizer_updates": updates,
            "optimizer_state_count": 94,
            "optimizer_step_values": [updates],
            "epoch": epoch,
            "iteration": iteration,
            "epoch_finished": epoch_finished,
            "checkpoint_reason": reason,
            "checkpoint_cuda_memory": {},
            "args": self._checkpoint_args(
                runtime, contract, output, resume=resume
            ),
        }

    def _queue_extension(self, source: Path, contract_id: str = "M0") -> dict:
        contract = runner.CONTRACTS[contract_id]
        source_record = runner._compact_file_record(runner._file_record(source))
        records = [{**source_record, "roles": ["repository_source"]}]
        digest = runner._sha256_bytes(
            runner._canonical_json_bytes(
                {"schema": runner.STABLE_CLOSURE_SCHEMA, "records": records}
            )
        )
        return {
            "schema": runner.TRAINING_QUEUE_CONTRACT_SCHEMA,
            "contract_id": contract_id,
            "ordered_run_ids": list(contract.dedicated_queue_run_ids),
            "runner": runner._compact_file_record(
                runner._file_record(Path(runner.__file__))
            ),
            "controller_python": runner._compact_file_record(
                runner._file_record(runner.DEFAULT_PYTHON)
            ),
            "stable_input_closure": {
                "schema": runner.STABLE_CLOSURE_SCHEMA,
                "algorithm": "sha256_canonical_path_content_size_roles_v1",
                "digest": digest,
                "records": records,
            },
        }

    def test_catalog_and_queue_specs_are_two_exact_three_seed_queues(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(runner.main(["list", "--json"]), 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            payload["queue_contracts"]["M0"], ["M0:17", "M0:42", "M0:73"]
        )
        self.assertEqual(
            payload["queue_contracts"]["M0N"],
            ["M0N:17", "M0N:42", "M0N:73"],
        )
        self.assertTrue(payload["separate_exact_queues_required"])
        for contract_id in ("M0", "M0N"):
            spec = runner.queue_spec(contract_id)
            self.assertEqual(
                spec["ordered_run_ids"], payload["queue_contracts"][contract_id]
            )
            self.assertTrue(spec["mixed_M0_M0N_queue_forbidden"])
            self.assertEqual(spec["runtime"]["successful_update_batch_slots"], 941_280)

    def test_canonical_training_queue_pins_python_and_source_extension(self):
        from tools import run_stageb_serial_matrix_queue as serial_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root / "runtime")
            extension = {"sealed": True}
            queue = {"status": "planned"}
            with (
                mock.patch.object(
                    runner, "runtime_from_environment", return_value=runtime
                ),
                mock.patch.object(
                    runner,
                    "_training_queue_contract_payload",
                    return_value=extension,
                ),
                mock.patch.object(
                    serial_queue, "create_queue", return_value=queue
                ) as create,
                mock.patch.object(runner, "verify_training_queue") as verify,
            ):
                observed = runner.create_training_queue(
                    root / "queue", "M0", lease_root=root / "leases", gpu_key="0"
                )
            self.assertIs(observed, queue)
            self.assertEqual(create.call_args.kwargs["runner_python"], runner.DEFAULT_PYTHON)
            self.assertEqual(create.call_args.kwargs["paper_runner"], Path(runner.__file__))
            self.assertEqual(
                create.call_args.kwargs["plan_extensions"],
                {runner.TRAINING_QUEUE_EXTENSION_KEY: extension},
            )
            verify.assert_called_once_with(root / "queue", "M0")

    def test_active_training_item_and_lease_are_replayed(self):
        from tools import run_stageb_serial_matrix_queue as serial_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="ascii")
            extension = self._queue_extension(source)
            orchestration_root = root / "queue" / "jobs" / "001-M0_42"
            plan_items = [
                {"run_id": run_id, "runner": "paper"}
                for run_id in runner.CONTRACTS["M0"].dedicated_queue_run_ids
            ]
            mutable_items = [
                {
                    "index": 0,
                    "run_id": "M0:17",
                    "runner": "paper",
                    "status": "completed",
                },
                {
                    "index": 1,
                    "run_id": "M0:42",
                    "runner": "paper",
                    "status": "reserved",
                    "orchestration_root": str(orchestration_root),
                },
                {
                    "index": 2,
                    "run_id": "M0:73",
                    "runner": "paper",
                    "status": "pending",
                },
            ]
            queue = {
                "status": "running",
                "plan_sha256": "a" * 64,
                "plan": {
                    "queue_id": "queue-id",
                    "runner_python": str(runner.DEFAULT_PYTHON),
                    "items": plan_items,
                    "runners": {
                        "paper": {
                            "path": str(Path(runner.__file__).resolve()),
                            "sha256": runner._sha256_file(Path(runner.__file__)),
                        }
                    },
                    "extensions": {
                        runner.TRAINING_QUEUE_EXTENSION_KEY: extension
                    },
                    "runtime_environment": {},
                    "gpu_key": "0",
                    "lease_path": str(root / "leases" / "gpu-0.json"),
                },
                "items": mutable_items,
            }
            with (
                mock.patch.object(serial_queue, "load_queue", return_value=queue),
                mock.patch.object(serial_queue, "_ensure_lease") as ensure_lease,
            ):
                verified = runner.verify_training_queue(
                    root / "queue",
                    "M0",
                    expected_run_id="M0:42",
                    expected_orchestration_root=orchestration_root,
                )
            self.assertEqual(verified["active_item"]["item_index"], 1)
            self.assertEqual(
                verified["active_item"]["orchestration_root"],
                str(orchestration_root.resolve()),
            )
            self.assertEqual(ensure_lease.call_count, 3)
            ensure_lease.assert_has_calls(
                [mock.call(queue, mutable_items[1], create=False)] * 3
            )

            with (
                mock.patch.object(serial_queue, "load_queue", return_value=queue),
                self.assertRaisesRegex(
                    runner.HeadlineM0Error, "orchestration root"
                ),
            ):
                runner.verify_training_queue(
                    root / "queue",
                    "M0",
                    expected_run_id="M0:42",
                    expected_orchestration_root=root / "wrong",
                )

            with (
                mock.patch.object(serial_queue, "load_queue", return_value=queue),
                mock.patch.object(
                    serial_queue,
                    "_ensure_lease",
                    side_effect=serial_queue.QueueContractError("missing lease"),
                ),
                self.assertRaisesRegex(runner.HeadlineM0Error, "lease is invalid"),
            ):
                runner.verify_training_queue(
                    root / "queue",
                    "M0",
                    expected_run_id="M0:42",
                    expected_orchestration_root=orchestration_root,
                )

    def test_existing_active_queue_accepts_launching_to_launched_transition(self):
        from tools import run_stageb_serial_matrix_queue as serial_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="ascii")
            orchestration_root = root / "queue" / "jobs" / "001-M0_42"
            plan_items = [
                {"run_id": run_id, "runner": "paper"}
                for run_id in runner.CONTRACTS["M0"].dedicated_queue_run_ids
            ]
            mutable_items = [
                {
                    "index": 0,
                    "run_id": "M0:17",
                    "runner": "paper",
                    "status": "completed",
                },
                {
                    "index": 1,
                    "run_id": "M0:42",
                    "runner": "paper",
                    "status": "launching",
                    "orchestration_root": str(orchestration_root),
                },
                {
                    "index": 2,
                    "run_id": "M0:73",
                    "runner": "paper",
                    "status": "pending",
                },
            ]
            queue = {
                "status": "running",
                "plan_sha256": "a" * 64,
                "plan": {
                    "queue_id": "queue-id",
                    "runner_python": str(runner.DEFAULT_PYTHON),
                    "items": plan_items,
                    "runners": {
                        "paper": {
                            "path": str(Path(runner.__file__).resolve()),
                            "sha256": runner._sha256_file(Path(runner.__file__)),
                        }
                    },
                    "extensions": {
                        runner.TRAINING_QUEUE_EXTENSION_KEY: self._queue_extension(source)
                    },
                    "runtime_environment": {},
                    "gpu_key": "0",
                    "lease_path": str(root / "leases" / "gpu-0.json"),
                },
                "items": mutable_items,
                "revision": 2,
            }
            launched = copy.deepcopy(queue)
            launched["revision"] = 3
            launched["items"][1].update(
                {
                    "status": "launched",
                    "job_dir": str(orchestration_root / "job-1"),
                    "output_root": str(root / "outputs" / "M0" / "seed42"),
                }
            )
            queue_dir = root / "queue"
            queue_dir.mkdir()
            runner._write_json_atomic(queue_dir / "queue.json", queue)

            def load_queue(path):
                return json.loads((path / "queue.json").read_text(encoding="utf-8"))

            lease_checks = 0

            def ensure_lease(observed_queue, item, *, create):
                nonlocal lease_checks
                self.assertFalse(create)
                self.assertEqual(item["run_id"], "M0:42")
                lease_checks += 1
                if lease_checks == 1:
                    runner._write_json_atomic(queue_dir / "queue.json", launched)

            with (
                mock.patch.object(serial_queue, "load_queue", side_effect=load_queue),
                mock.patch.object(
                    serial_queue, "_ensure_lease", side_effect=ensure_lease
                ),
            ):
                verified = runner.verify_training_queue(
                    queue_dir,
                    "M0",
                    expected_run_id="M0:42",
                    expected_orchestration_root=orchestration_root,
                )

            self.assertEqual(verified["active_item"]["item_status"], "launched")
            self.assertEqual(
                verified["queue_manifest"],
                runner._stable_completed_file_record(queue_dir / "queue.json"),
            )
            self.assertEqual(lease_checks, 3)
            with self.assertRaisesRegex(
                runner.HeadlineM0Error, "status transition"
            ):
                runner._validate_active_training_queue_transition(
                    launched,
                    queue,
                    expected_run_id="M0:42",
                    expected_orchestration_root=orchestration_root,
                )

    def test_one_run_id_parser_has_no_all_or_multi_run_surface(self):
        parser = runner.build_parser()
        parsed = parser.parse_args(["run", "--run-id", "M0:17"])
        self.assertEqual(parsed.run_id, "M0:17")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--all"])
        with self.assertRaises(runner.HeadlineM0Error):
            runner._parse_run_id("M0:19")

    def test_formal_run_requires_serial_queue_binding(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(runner.HeadlineM0Error, "serial queue"):
                runner._run(runner.source_contracts.M0_CONTRACT, 17)

    def test_serial_queue_is_inferred_only_from_jobs_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "queue"
            root = queue / "jobs" / "000-M0_17"
            root.mkdir(parents=True)
            (queue / "queue.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                runner._serial_queue_dir_from_orchestration_root(root),
                queue.resolve(),
            )
            direct = Path(temporary) / "direct"
            direct.mkdir()
            with self.assertRaisesRegex(runner.HeadlineM0Error, "serial_matrix"):
                runner._serial_queue_dir_from_orchestration_root(direct)

    def test_fresh_and_resume_commands_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            output = root / "run"
            recovery = root / "recovery.pth"
            recovery.write_bytes(b"recovery")
            contract = runner.source_contracts.M0_CONTRACT
            fresh = runner.build_command(runtime, contract, 17, output)
            resumed = runner.build_command(
                runtime,
                contract,
                17,
                output,
                resume_checkpoint=recovery,
            )
            self.assertIn("--pretrain_model_path", fresh)
            self.assertNotIn("--resume", fresh)
            self.assertIn("--resume", resumed)
            self.assertNotIn("--pretrain_model_path", resumed)
            for command in (fresh, resumed):
                self.assertIn("--no_persistent_workers", command)
                self.assertIn("--pin_memory", command)
                self.assertIn("--gradient_accumulation_steps", command)
                self.assertEqual(
                    command.count(
                        f"stage_b_v15_scorer_init_checkpoint={runtime.stage_a_init}"
                    ),
                    1,
                )

    def test_manifest_dual_emits_phase_id_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            source = root / "source.py"
            source.write_text("x = 1\n", encoding="utf-8")
            record = runner._file_record(source, role="repository_source")
            closure = {
                "records": [record],
                "normalized_records": [
                    {
                        **runner._compact_file_record(record),
                        "roles": ["repository_source"],
                    }
                ],
                "digest": "a" * 64,
                "dataset_contract": {},
                "config_dependency_count": 1,
            }
            output = root / "formal"
            with (
                mock.patch.object(runner, "_stable_input_closure", return_value=closure),
                mock.patch.object(runner, "output_directory", return_value=output),
            ):
                manifest = runner.build_manifest(
                    runtime,
                    runner.source_contracts.M0N_CONTRACT,
                    17,
                    runner.token_launcher.HashCache(),
                )
            self.assertEqual(manifest["phases"][0]["phase_id"], "joint")
            self.assertEqual(
                manifest["phases"][0]["phase"]["phase_id"], "joint"
            )
            self.assertEqual(
                manifest["equal_budget_contract"],
                {
                    "batch_size": 40,
                    "optimizer_updates": 23_532,
                    "contributing_phase_updates": {"joint": 23_532},
                    "successful_update_batch_slots": 941_280,
                },
            )
            self.assertEqual(
                manifest["row"]["token_objective"], "targetlocal_allneg_bce"
            )

    def test_safe_release_inspector_returns_exact_five_key_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            optimizer_state = {
                index: {"step": torch.tensor(7.0)} for index in range(94)
            }
            torch.save(
                {
                    "model": {},
                    "criterion": {},
                    "optimizer": {"state": optimizer_state, "param_groups": []},
                    "lr_scheduler": {},
                    "scaler": {},
                    "epoch": 0,
                    "iteration": 7,
                    "optimizer_updates": 7,
                    "epoch_finished": False,
                    "rng_state": {},
                    "epoch_rng_state": {},
                    "args": {},
                    "checkpoint_reason": "signal",
                },
                path,
            )
            result = runner.inspect_training_checkpoint_for_release(path)
            self.assertEqual(
                set(result),
                {
                    "optimizer_updates",
                    "optimizer_state_count",
                    "optimizer_step_values",
                    "complete_state_components",
                    "checkpoint_reason",
                },
            )
            self.assertEqual(result["optimizer_updates"], 7)
            self.assertEqual(result["optimizer_state_count"], 94)
            self.assertEqual(result["optimizer_step_values"], [7])
            self.assertEqual(
                result["complete_state_components"],
                runner.COMPLETE_STATE_COMPONENTS,
            )

    def test_final_checkpoint_contract_is_epoch2_iteration6756(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            output = root / "run"
            output.mkdir()
            metadata = self._metadata(
                runtime,
                runner.source_contracts.M0_CONTRACT,
                output,
                updates=23_532,
                epoch=2,
                iteration=6_756,
                reason="max_train_iters",
            )
            self.assertEqual(
                runner._validate_checkpoint_metadata(
                    metadata,
                    runtime=runtime,
                    contract=runner.source_contracts.M0_CONTRACT,
                    seed=17,
                    output_dir=output,
                    source_optimizer_updates=0,
                    resume_checkpoint=None,
                ),
                "max_train_iters",
            )

    def test_only_mid_epoch_signal_is_recoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            output = root / "run"
            output.mkdir()
            contract = runner.source_contracts.M0N_CONTRACT
            signal_metadata = self._metadata(
                runtime,
                contract,
                output,
                updates=100,
                epoch=0,
                iteration=100,
                reason="signal",
            )
            self.assertEqual(
                runner._validate_checkpoint_metadata(
                    signal_metadata,
                    runtime=runtime,
                    contract=contract,
                    seed=17,
                    output_dir=output,
                    source_optimizer_updates=0,
                    resume_checkpoint=None,
                ),
                "signal",
            )
            epoch_boundary = {
                **signal_metadata,
                "iteration": 0,
                "epoch_finished": True,
                "checkpoint_reason": "signal_after_epoch",
            }
            with self.assertRaisesRegex(
                runner.HeadlineM0Error, "signal_after_epoch"
            ):
                runner._validate_checkpoint_metadata(
                    epoch_boundary,
                    runtime=runtime,
                    contract=contract,
                    seed=17,
                    output_dir=output,
                    source_optimizer_updates=0,
                    resume_checkpoint=None,
                )

    def test_signal_checkpoint_is_a_valid_exact_learning_curve_milestone(self):
        metadata = {
            "complete_state_components": dict(runner.COMPLETE_STATE_COMPONENTS),
            "optimizer_updates": 1_000,
            "optimizer_state_count": runner.FORMAL_OPTIMIZER_STATE_COUNT,
            "optimizer_step_values": [1_000],
            "epoch": 0,
            "iteration": 1_000,
            "epoch_finished": False,
            "checkpoint_reason": "signal",
        }
        runner._validate_milestone_metadata(metadata, 1_000)
        with self.assertRaisesRegex(runner.HeadlineM0Error, "milestone"):
            runner._validate_milestone_metadata(
                {**metadata, "checkpoint_reason": "manual"}, 1_000
            )
        with self.assertRaisesRegex(runner.HeadlineM0Error, "milestone"):
            runner._validate_milestone_metadata(
                {
                    **metadata,
                    "optimizer_updates": runner.FORMAL_UPDATES,
                    "optimizer_step_values": [runner.FORMAL_UPDATES],
                    "epoch": 2,
                    "iteration": 6_756,
                },
                runner.FORMAL_UPDATES,
            )

    def test_resume_log_rejects_fresh_optimizer_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.log"
            source = {"epoch": 0, "iteration": 100, "optimizer_updates": 100}
            path.write_text(
                "Restored resume training state: epoch=0, iteration=100, "
                "optimizer_updates=100, epoch_finished=False, scaler_restored=True\n"
                "Resuming mid-epoch from epoch=0\n",
                encoding="utf-8",
            )
            runner._validate_resume_log(path, source_metadata=source)
            path.write_text(
                "Failed to restore; continuing with fresh optimizer state.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(runner.HeadlineM0Error, "fresh"):
                runner._validate_resume_log(path, source_metadata=source)

    def test_resume_authorization_is_one_no_replace_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            control = root / "control"
            control.mkdir(parents=True)
            recovery = root / "recovery.pth"
            recovery.write_bytes(b"recovery")
            recovery_record = runner._compact_file_record(
                runner._file_record(recovery)
            )
            required = {
                "schema": runner.RESUME_REQUEST_SCHEMA,
                "status": "required",
                "created_at_utc": runner._utc_now(),
                "run_id": "M0:17",
                "next_attempt_ordinal": 1,
                "recovery_checkpoint": recovery_record,
                "policy": "explicit_one_attempt_mid_epoch_signal_resume",
            }
            runner._write_json_atomic(runner._resume_required_path(root), required)
            runner._write_json_atomic(
                root / "sequence_manifest.json",
                {
                    "status": "running",
                    "paused_for_resume": {
                        "next_attempt_ordinal": 1,
                        "recovery_checkpoint": recovery_record,
                    },
                },
            )
            result = runner.authorize_resume(root)
            self.assertEqual(result["status"], "authorized")
            request = json.loads(
                runner._resume_request_path(root).read_text(encoding="utf-8")
            )
            self.assertEqual(request["next_attempt_ordinal"], 1)
            with self.assertRaisesRegex(runner.HeadlineM0Error, "overwrite"):
                runner.authorize_resume(root)
            with redirect_stdout(StringIO()):
                authorization_record = runner._wait_for_resume_authorization(
                    run_root=root,
                    run_id="M0:17",
                    next_ordinal=1,
                    recovery_checkpoint=recovery_record,
                    orchestration_status=None,
                )
            archived = root / "control" / "resume_requests" / "001.json"
            self.assertEqual(
                authorization_record,
                runner._compact_file_record(runner._file_record(archived)),
            )
            self.assertFalse(runner._resume_request_path(root).exists())
            self.assertFalse(runner._resume_required_path(root).exists())

    def test_recovery_checkpoint_is_rehashed_before_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            recovery_root = root / "recovery"
            recovery_root.mkdir(parents=True)
            checkpoint = recovery_root / "attempt_001_from_u000100_deadbeef0000.pth"
            checkpoint.write_bytes(b"original")
            record = runner._compact_file_record(runner._file_record(checkpoint))
            checkpoint.write_bytes(b"tampered")
            runtime = self._runtime(Path(temporary) / "runtime")
            with self.assertRaisesRegex(
                runner.HeadlineM0Error, "identity changed"
            ):
                runner._verify_recovery_checkpoint(
                    record,
                    runtime=runtime,
                    run_root=root,
                    ordinal=1,
                    expected_optimizer_updates=100,
                )

    def test_dataset_contract_replays_8388_microbatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self._runtime(Path(temporary))
            contract, paths = runner._validate_dataset(runtime)
            self.assertEqual(contract["total_train_rows"], 335_523)
            self.assertEqual(
                contract["drop_last_microbatches_per_epoch"], 8_388
            )
            self.assertGreater(len(paths), 4)

    def test_native_closure_uses_actual_python311_abi_not_python312_build(self):
        paths = runner._native_runtime_dependency_paths()
        self.assertIn(
            Path(
                runner.importlib.util.find_spec(
                    "MultiScaleDeformableAttention"
                ).origin
            ).resolve(),
            paths,
        )
        self.assertTrue(any(path.name == "setup.py" for path in paths))
        self.assertFalse(any("cpython-312" in str(path) for path in paths))

    def test_recursive_source_closure_includes_dynamic_model_and_package_init(self):
        paths = set(
            runner._repository_dependency_paths(
                runner.source_contracts.M0N_CONTRACT
            )
        )
        self.assertIn(
            (runner.REPO_ROOT / "models/GroundingDINO/groundingdino.py").resolve(),
            paths,
        )
        self.assertIn((runner.REPO_ROOT / "models/__init__.py").resolve(), paths)
        self.assertIn(
            (runner.REPO_ROOT / "models/GroundingDINO/__init__.py").resolve(),
            paths,
        )

    def test_formal_runtime_rejects_budget_drift_before_training(self):
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PIVOT_PYTHON": str(runner.DEFAULT_PYTHON),
            "PIVOT_BATCH_SIZE": "41",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(runner.HeadlineM0Error, "batch size"):
                runner.runtime_from_environment()

    def test_two_attempt_signal_resume_writes_contiguous_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root / "runtime")
            run_root = root / "formal" / "M0" / "seed42"
            source = root / "source.py"
            source.write_text("x = 1\n", encoding="utf-8")
            records = [
                runner._file_record(
                    runtime.stage_a_init, role="stage_a_initializer"
                ),
                runner._file_record(
                    runtime.stage_a_init, role="scorer_warmstart"
                ),
                runner._file_record(source, role="repository_source"),
            ]
            normalized = [
                {
                    **runner._compact_file_record(record),
                    "roles": [record["role"]],
                }
                for record in records
            ]
            normalized.sort(key=lambda value: (value["path"], value["roles"]))
            digest = runner._sha256_bytes(
                runner._canonical_json_bytes(
                    {
                        "schema": runner.STABLE_CLOSURE_SCHEMA,
                        "records": normalized,
                    }
                )
            )
            closure = {
                "records": records,
                "normalized_records": normalized,
                "digest": digest,
                "dataset_contract": {},
                "config_dependency_count": 0,
            }
            with (
                mock.patch.object(runner, "_stable_input_closure", return_value=closure),
                mock.patch.object(runner, "output_directory", return_value=run_root),
            ):
                planned = runner.build_manifest(
                    runtime,
                    runner.source_contracts.M0_CONTRACT,
                    42,
                    runner.token_launcher.HashCache(),
                )

            signal_metadata = {
                "complete_state_components": dict(runner.COMPLETE_STATE_COMPONENTS),
                "optimizer_updates": 100,
                "optimizer_state_count": 94,
                "optimizer_step_values": [100],
                "epoch": 0,
                "iteration": 100,
                "epoch_finished": False,
                "checkpoint_reason": "signal",
                "checkpoint_cuda_memory": {},
                "args": {},
            }
            final_metadata = {
                **signal_metadata,
                "optimizer_updates": 23_532,
                "optimizer_step_values": [23_532],
                "epoch": 2,
                "iteration": 6_756,
                "checkpoint_reason": "max_train_iters",
            }
            process_count = 0

            def fake_process(*_args, **kwargs):
                nonlocal process_count
                attempt_dir = kwargs["attempt_dir"]
                attempt_dir.joinpath("train_console.log").write_text(
                    "synthetic attempt\n", encoding="utf-8"
                )
                kwargs["run_root"].joinpath("checkpoint_iter.pth").write_bytes(
                    f"checkpoint-{process_count}".encode("ascii")
                )
                ordinal = process_count
                process_count += 1
                return {
                    "pid": 1000 + ordinal,
                    "identity": {"pid": 1000 + ordinal},
                    "start_new_session": True,
                    "stdin": "DEVNULL",
                    "stdout_stderr": str(attempt_dir / "train_console.log"),
                    "returncode": 0,
                    "forwarded_signals": [],
                    "started_at_utc": f"2026-07-19T00:0{ordinal}:00+00:00",
                    "finished_at_utc": f"2026-07-19T00:0{ordinal}:30+00:00",
                }

            def fake_archive(_source, **kwargs):
                recovery_root = kwargs["run_root"] / "recovery"
                recovery_root.mkdir(parents=True, exist_ok=True)
                path = recovery_root / "attempt_001_from_u000100_deadbeef0000.pth"
                path.write_bytes(b"recovery")
                return runner._compact_file_record(runner._file_record(path))

            def fake_authorize(**kwargs):
                authorization = {
                    "schema": runner.RESUME_REQUEST_SCHEMA,
                    "status": "authorized",
                    "run_id": kwargs["run_id"],
                    "next_attempt_ordinal": kwargs["next_ordinal"],
                    "recovery_checkpoint": dict(kwargs["recovery_checkpoint"]),
                    "policy": "explicit_one_attempt_mid_epoch_signal_resume",
                    "authorized_at_utc": "2026-07-19T00:00:45+00:00",
                    "authorizer_pid": 4242,
                    "detached_controller_identity": None,
                }
                path = (
                    kwargs["run_root"]
                    / "control"
                    / "resume_requests"
                    / f"{kwargs['next_ordinal']:03d}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(authorization, sort_keys=True) + "\n",
                    encoding="ascii",
                )
                return runner._compact_file_record(runner._file_record(path))

            def fake_archive_telemetry(**kwargs):
                ordinal = kwargs["ordinal"]
                paths = runner._attempt_telemetry_paths(
                    kwargs["run_root"], ordinal
                )
                runner._write_json_atomic(
                    paths["gpu_environment"],
                    {"schema": "pivot.gpu_environment/v1", "attempt": ordinal},
                )
                paths["gpu_telemetry"].write_text(
                    f"attempt-{ordinal}\n", encoding="ascii"
                )
                runner._write_json_atomic(
                    paths["gpu_telemetry_summary"],
                    {
                        "schema": "pivot.gpu_telemetry_summary/v1",
                        "sample_rows": 1,
                        "sampling_interval_ms": 1000,
                        "attempt": ordinal,
                    },
                )
                return {
                    "schema": runner.ATTEMPT_TELEMETRY_SCHEMA,
                    "status": "sealed",
                    "attempt_ordinal": ordinal,
                    "sampling_interval_ms": 1000,
                    "sample_rows": 1,
                    "devices": [
                        {
                            "physical_index": 0,
                            "uuid": "GPU-test",
                            "name": "test-gpu",
                            "driver_version": "test-driver",
                            "total_memory_mib": 1.0,
                        }
                    ],
                    "artifacts": {
                        name: runner._compact_file_record(runner._file_record(path))
                        for name, path in paths.items()
                    },
                }

            postflight = {
                "schema": runner.POSTFLIGHT_SCHEMA,
                "status": "passed",
                "run_id": "M0:42",
                "phase_id": "joint",
            }
            sampler = mock.Mock()
            sampler.stop.return_value = {
                "schema": "pivot.gpu_telemetry_summary/v1",
                "sampling_interval_ms": 1000,
                "sample_rows": 1,
            }
            with (
                mock.patch.object(runner, "runtime_from_environment", return_value=runtime),
                mock.patch.object(runner, "output_directory", return_value=run_root),
                mock.patch.object(
                    runner, "build_manifest", return_value=copy.deepcopy(planned)
                ),
                mock.patch.object(
                    runner.paper_launcher,
                    "_capture_gpu_environment",
                    return_value={"schema": "pivot.gpu_environment/v1"},
                ),
                mock.patch.object(
                    runner.paper_launcher, "_GpuTelemetrySampler", return_value=sampler
                ),
                mock.patch.object(
                    runner,
                    "_archive_attempt_telemetry",
                    side_effect=fake_archive_telemetry,
                ),
                mock.patch.object(
                    runner, "_run_training_process", side_effect=fake_process
                ),
                mock.patch.object(
                    runner,
                    "_inspect_checkpoint_extended",
                    side_effect=[signal_metadata, final_metadata],
                ),
                mock.patch.object(
                    runner,
                    "_validate_checkpoint_metadata",
                    side_effect=["signal", "max_train_iters"],
                ),
                mock.patch.object(
                    runner, "_archive_recovery_checkpoint", side_effect=fake_archive
                ),
                mock.patch.object(
                    runner,
                    "_wait_for_resume_authorization",
                    side_effect=fake_authorize,
                ),
                mock.patch.object(
                    runner,
                    "_verify_recovery_checkpoint",
                    return_value=signal_metadata,
                ),
                mock.patch.object(runner, "_validate_resume_log"),
                mock.patch.object(
                    runner, "_perform_postflight", return_value=postflight
                ),
            ):
                with redirect_stdout(StringIO()):
                    result = runner._run_body(
                        runner.source_contracts.M0_CONTRACT,
                        42,
                        orchestration_status=None,
                    )
            self.assertEqual(result, 0)
            self.assertEqual(process_count, 2)
            sequence = json.loads(
                (run_root / "sequence_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sequence["status"], "completed")
            self.assertEqual(sequence["training_attempt_count"], 2)
            self.assertEqual(sequence["same_run_resume_count"], 1)
            self.assertEqual(sequence["completed_phases"][0]["phase_id"], "joint")
            attempt0 = runner._load_attempt(run_root, 0)
            attempt1 = runner._load_attempt(run_root, 1)
            self.assertEqual(attempt0["termination"]["reason"], "signal")
            self.assertEqual(attempt0["telemetry"]["attempt_ordinal"], 0)
            self.assertEqual(attempt1["telemetry"]["attempt_ordinal"], 1)
            self.assertNotEqual(
                attempt0["telemetry"]["artifacts"]["gpu_telemetry"]["sha256"],
                attempt1["telemetry"]["artifacts"]["gpu_telemetry"]["sha256"],
            )
            self.assertIsNone(attempt0["resume_authorization"])
            self.assertEqual(attempt1["source_optimizer_updates"], 100)
            self.assertEqual(
                attempt1["resume_authorization"],
                runner._compact_file_record(
                    runner._file_record(
                        run_root / "control" / "resume_requests" / "001.json"
                    )
                ),
            )
            self.assertEqual(
                attempt1["parent_attempt_manifest"],
                runner._attempt_manifest_record(run_root, 0),
            )
            launch = json.loads(
                (run_root / "launch_manifest.json").read_text(encoding="utf-8")
            )
            ancestry = runner._build_ancestry(
                phase_manifest=launch,
                runtime=runtime,
                contract=runner.source_contracts.M0_CONTRACT,
                seed=42,
                run_root=run_root,
                final_ordinal=1,
            )
            self.assertEqual(len(ancestry["resume_ancestry"]), 1)
            self.assertEqual(
                ancestry["resume_ancestry"][0]["source_checkpoint_reason"],
                "signal",
            )
            self.assertEqual(
                ancestry["resume_ancestry"][0]["resume_authorization"],
                attempt1["resume_authorization"],
            )


if __name__ == "__main__":
    unittest.main()
