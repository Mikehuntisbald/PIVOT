import copy
import hashlib
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_stageb_headline_m0 as runner


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def _stable_closure(records: list[dict]) -> dict:
    normalized = [
        {**runner._compact_file_record(record), "roles": [record["role"]]}
        for record in records
    ]
    normalized.sort(key=lambda value: (value["path"], value["roles"]))
    digest = runner._sha256_bytes(
        runner._canonical_json_bytes(
            {"schema": runner.STABLE_CLOSURE_SCHEMA, "records": normalized}
        )
    )
    return {"records": normalized, "digest": digest}


def _write_attempt_telemetry(
    run_root: Path, ordinal: int, *, sample_rows: int = 1
) -> dict:
    paths = runner._attempt_telemetry_paths(run_root, ordinal)
    environment = {
        "schema": "pivot.gpu_environment/v1",
        "torch_runtime": {"cuda_available": True},
        "nvidia_devices": [
            {
                "uuid": "GPU-test",
                "name": "test-gpu",
                "driver_version": "test-driver",
                "total_memory_mib": 1000.0,
            }
        ],
    }
    _write_json(paths["gpu_environment"], environment)
    rows = [
        f"2026/07/19 00:00:{index:02d}.000, 0, GPU-test, test-gpu, "
        f"test-driver, 1000, {100 + index}, {900 - index}, {10 + index}"
        for index in range(sample_rows)
    ]
    paths["gpu_telemetry"].write_text(
        "timestamp,index,uuid,name,driver_version,total_memory_mib,"
        "used_memory_mib,free_memory_mib,utilization_percent\n"
        + "\n".join(rows)
        + "\n",
        encoding="ascii",
    )
    summary = runner.paper_launcher._summarize_nvidia_csv(paths["gpu_telemetry"])
    summary["captured_at_utc"] = f"2026-07-19T00:0{ordinal}:00+00:00"
    summary["sampling_interval_ms"] = 1000
    _write_json(paths["gpu_telemetry_summary"], summary)
    return {
        "schema": runner.ATTEMPT_TELEMETRY_SCHEMA,
        "status": "sealed",
        "attempt_ordinal": ordinal,
        "sampling_interval_ms": 1000,
        "sample_rows": sample_rows,
        "devices": runner._gpu_telemetry_device_projection(summary["devices"]),
        "artifacts": {
            name: runner._compact_file_record(runner._file_record(path))
            for name, path in paths.items()
        },
    }


class CompletedTrainingAttemptReplayTest(unittest.TestCase):
    def _scorer_audit(self, stage_a: Path) -> dict:
        return {
            "schema": "stage_b_v15_scorer_init/v1",
            "status": "applied",
            "requested_source_path": str(stage_a.resolve()),
            "resolved_source_path": str(stage_a.resolve()),
            "source_sha256": runner._sha256_file(stage_a),
            "source_size_bytes": stage_a.stat().st_size,
            "source_decoder_num_layers": 6,
            "selected_source_layer_indices": [3, 4, 5],
            "loaded_num_layers": 3,
            "loaded_tensor_count": 90,
            "loaded_components": [
                "decoder.layers[-N:]",
                "decoder.ref_point_head",
                "decoder.norm",
            ],
        }

    def _runtime(self, root: Path, stage_a: Path) -> runner.Runtime:
        return runner.Runtime(
            python=Path(sys.executable).resolve(),
            stage_a_init=stage_a.resolve(),
            dataset=runner.DEFAULT_DATASET.resolve(),
            output_root=root,
            data_root=runner.DEFAULT_DATA_ROOT.resolve(),
            batch_size=runner.FORMAL_BATCH_SIZE,
            max_train_iters=runner.FORMAL_UPDATES,
            iter_checkpoint_interval=runner.FORMAL_CHECKPOINT_INTERVAL,
            num_workers=runner.FORMAL_NUM_WORKERS,
            prefetch_factor=runner.FORMAL_PREFETCH_FACTOR,
            omp_num_threads=8,
            min_nofile=65_536,
            cuda_visible_devices="0",
            mp_sharing_strategy="file_system",
            gradient_diagnostic_interval=runner.FORMAL_GRADIENT_DIAGNOSTIC_INTERVAL,
            telemetry_interval_seconds=runner.FORMAL_TELEMETRY_INTERVAL_SECONDS,
            pin_memory=True,
            persistent_workers=False,
            gradient_accumulation_steps=1,
        )

    def _checkpoint_args(
        self,
        runtime: runner.Runtime,
        contract,
        run_root: Path,
        *,
        seed: int,
        resume: Path | None,
    ) -> dict:
        values = {
            "seed": seed,
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
            "output_dir": str(run_root),
            "pretrain_model_path": "" if resume else str(runtime.stage_a_init),
            "resume": str(resume) if resume else "",
            "stage_b_v15_scorer_init_checkpoint": str(runtime.stage_a_init),
            "stage_b_v15_scorer_init_audit": self._scorer_audit(
                runtime.stage_a_init
            ),
            "stage_b_v25_main_id": contract.id,
            "stage_b_v25_compute_contract": "b58_successful_update_batch_slot_matched",
            "stage_b_v25_budget_unit": "successful_optimizer_update_global_batch_slots",
            "stage_b_v25_successful_update_batch_slots": 941_280,
            "stage_b_v25_initializer_contract": "same_stage_a_model_and_scorer_no_b58",
            "stage_b_v25_strict_resume": True,
            "stage_b_v22_table_id": "S2F",
            "stage_b_v22_objective_fidelity": "full_v19_base_plus_gate_objective",
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
        runtime,
        contract,
        run_root,
        *,
        seed,
        updates,
        epoch,
        iteration,
        reason,
        resume=None,
    ) -> dict:
        return {
            "top_level_keys": [],
            "complete_state_components": dict(runner.COMPLETE_STATE_COMPONENTS),
            "optimizer_updates": updates,
            "optimizer_state_count": runner.FORMAL_OPTIMIZER_STATE_COUNT,
            "optimizer_step_values": [updates],
            "epoch": epoch,
            "iteration": iteration,
            "epoch_finished": False,
            "checkpoint_reason": reason,
            "checkpoint_cuda_memory": {},
            "args": self._checkpoint_args(
                runtime, contract, run_root, seed=seed, resume=resume
            ),
        }

    def _fixture(self, root: Path, contract_id: str = "M0") -> dict:
        contract = runner.CONTRACTS[contract_id]
        seed = 42
        run_root = root / contract_id / f"seed{seed}"
        run_root.mkdir(parents=True)
        stage_a = root / "checkpoint0004.pth"
        stage_a.write_bytes(b"stage-a")
        _write_json(
            run_root / "stage_b_v15_scorer_init_audit.json",
            self._scorer_audit(stage_a),
        )
        runtime = self._runtime(root, stage_a)
        source = root / "source.py"
        source.write_text("VALUE = 1\n", encoding="ascii")
        records = [
            runner._file_record(stage_a, role="stage_a_initializer"),
            runner._file_record(stage_a, role="scorer_warmstart"),
            runner._file_record(source, role="repository_source"),
        ]
        closure = _stable_closure(records)
        launch = {
            "inputs": {
                "records": records,
                "stable_closure_digest": closure["digest"],
            },
            "started_at_utc": "2026-07-19T00:00:00+00:00",
            "finished_at_utc": "2026-07-19T01:00:00+00:00",
        }

        recovery_tmp = run_root / "recovery" / "temporary.pth"
        recovery_tmp.parent.mkdir()
        recovery_tmp.write_bytes(b"recovery-u100")
        recovery_sha = runner._sha256_file(recovery_tmp)
        recovery = recovery_tmp.with_name(
            f"attempt_001_from_u000100_{recovery_sha[:12]}.pth"
        )
        recovery_tmp.rename(recovery)
        recovery_record = runner._compact_file_record(runner._file_record(recovery))
        final_checkpoint = run_root / "checkpoint_iter.pth"
        final_checkpoint.write_bytes(b"final-u23532")
        final_record = runner._compact_file_record(
            runner._file_record(final_checkpoint)
        )

        metadata0 = self._metadata(
            runtime,
            contract,
            run_root,
            seed=seed,
            updates=100,
            epoch=0,
            iteration=100,
            reason="signal",
        )
        metadata1 = self._metadata(
            runtime,
            contract,
            run_root,
            seed=seed,
            updates=23_532,
            epoch=2,
            iteration=6_756,
            reason="max_train_iters",
            resume=recovery,
        )
        attempts = []
        attempt_records = []
        attempt_telemetry = []
        for ordinal in (0, 1):
            attempt_dir, closure_path, attempt_path = runner._attempt_paths(
                run_root, ordinal
            )
            attempt_dir.mkdir(parents=True)
            _write_json(
                closure_path,
                {
                    "schema": runner.STABLE_CLOSURE_SCHEMA,
                    "status": "sealed",
                    "algorithm": "sha256_canonical_path_content_size_roles_v1",
                    "records": closure["records"],
                    "digest": closure["digest"],
                },
            )
            log = attempt_dir / "train_console.log"
            if ordinal == 0:
                log.write_text("signal checkpoint\n", encoding="ascii")
                resume = None
                command = runner.build_command(
                    runtime, contract, seed, run_root
                )
                started = "2026-07-19T00:00:00+00:00"
                finished = "2026-07-19T00:01:00+00:00"
                metadata = metadata0
                checkpoint = recovery_record
                termination = {
                    "kind": "graceful_signal_checkpoint",
                    "reason": "signal",
                }
            else:
                log.write_text(
                    "Restored resume training state: epoch=0, iteration=100, "
                    "optimizer_updates=100, epoch_finished=False, "
                    "scaler_restored=True\nResuming mid-epoch from epoch=0\n",
                    encoding="ascii",
                )
                resume = recovery
                command = runner.build_command(
                    runtime,
                    contract,
                    seed,
                    run_root,
                    resume_checkpoint=recovery,
                )
                started = "2026-07-19T00:02:00+00:00"
                finished = "2026-07-19T01:00:00+00:00"
                metadata = metadata1
                checkpoint = final_record
                termination = {
                    "kind": "target_completed",
                    "reason": "max_train_iters",
                }
            process = {
                "pid": 1000 + ordinal,
                "identity": {"pid": 1000 + ordinal},
                "start_new_session": True,
                "stdin": "DEVNULL",
                "stdout_stderr": str(log.resolve()),
                "returncode": 0,
                "forwarded_signals": [15] if ordinal == 0 else [],
                "started_at_utc": started,
                "finished_at_utc": finished,
            }
            telemetry = _write_attempt_telemetry(run_root, ordinal)
            attempt = {
                "schema": runner.ATTEMPT_SCHEMA,
                "status": "completed",
                "run_id": f"{contract_id}:{seed}",
                "seed": seed,
                "attempt_ordinal": ordinal,
                "initialization_mode": (
                    "fresh_stage_a" if ordinal == 0 else "same_run_resume"
                ),
                "parent_attempt_manifest": (
                    None if ordinal == 0 else attempt_records[0]
                ),
                "resume_checkpoint": None if ordinal == 0 else recovery_record,
                "resume_authorization": None,
                "source_optimizer_updates": 0 if ordinal == 0 else 100,
                "target_optimizer_updates": runner.FORMAL_UPDATES,
                "command": command,
                "command_shell": shlex.join(command),
                "runtime": runner._attempt_runtime(runtime),
                "input_closure_digest": closure["digest"],
                "input_closure": runner._compact_file_record(
                    runner._file_record(closure_path)
                ),
                "telemetry": telemetry,
                "process": process,
                "termination": termination,
                "complete_state_components": dict(runner.COMPLETE_STATE_COMPONENTS),
                "checkpoint_at_exit": checkpoint,
                "checkpoint_metadata": runner._checkpoint_metadata_for_attempt(
                    metadata
                ),
                "started_at_utc": started,
                "finished_at_utc": finished,
            }
            if ordinal == 1:
                authorization_path = (
                    run_root / "control" / "resume_requests" / "001.json"
                )
                _write_json(
                    authorization_path,
                    {
                        "schema": runner.RESUME_REQUEST_SCHEMA,
                        "status": "authorized",
                        "run_id": f"{contract_id}:{seed}",
                        "next_attempt_ordinal": 1,
                        "recovery_checkpoint": recovery_record,
                        "policy": "explicit_one_attempt_mid_epoch_signal_resume",
                        "authorized_at_utc": "2026-07-19T00:01:30+00:00",
                        "authorizer_pid": 4242,
                        "detached_controller_identity": None,
                    },
                )
                attempt["resume_authorization"] = runner._compact_file_record(
                    runner._file_record(authorization_path)
                )
            _write_json(attempt_path, attempt)
            attempts.append(attempt)
            attempt_telemetry.append(telemetry)
            attempt_records.append(
                runner._compact_file_record(runner._file_record(attempt_path))
            )

        launch["command"] = attempts[-1]["command"]
        launch["command_shell"] = attempts[-1]["command_shell"]
        ancestry = runner._build_ancestry(
            phase_manifest=launch,
            runtime=runtime,
            contract=contract,
            seed=seed,
            run_root=run_root,
            final_ordinal=1,
        )
        return {
            "contract": contract,
            "seed": seed,
            "run_root": run_root,
            "runtime": runtime,
            "launch": launch,
            "closure": closure,
            "sequence": {"training_attempt_count": 2, "same_run_resume_count": 1},
            "postflight": {
                "training_attempt_count": 2,
                "same_run_resume_count": 1,
                "model_state_ancestry": ancestry,
                "full_run_telemetry": runner._full_run_telemetry_projection(
                    attempt_telemetry
                ),
            },
            "final_record": final_record,
            "metadata": [metadata0, metadata1],
            "attempt_paths": [
                runner._attempt_paths(run_root, ordinal)[2] for ordinal in (0, 1)
            ],
        }

    def _replay(self, fixture):
        by_path = {
            str(
                Path(
                    fixture["postflight"]["model_state_ancestry"][
                        "resume_ancestry"
                    ][0]["source_checkpoint"]["path"]
                ).resolve()
            ): fixture["metadata"][0],
            str((fixture["run_root"] / "checkpoint_iter.pth").resolve()): fixture[
                "metadata"
            ][1],
        }
        with mock.patch.object(
            runner,
            "_inspect_completed_checkpoint_snapshot",
            side_effect=lambda path, **_kwargs: copy.deepcopy(
                by_path[str(Path(path).resolve())]
            ),
        ):
            return runner._verify_completed_attempts(
                sequence=fixture["sequence"],
                launch=fixture["launch"],
                postflight=fixture["postflight"],
                runtime=fixture["runtime"],
                contract=fixture["contract"],
                seed=fixture["seed"],
                run_root=fixture["run_root"],
                stable_closure=fixture["closure"],
                final_checkpoint=fixture["final_record"],
            )

    def test_two_attempt_m0_and_m0n_replay_to_exact_u23532(self):
        for contract_id in ("M0", "M0N"):
            with self.subTest(contract_id=contract_id), tempfile.TemporaryDirectory() as tmp:
                fixture = self._fixture(Path(tmp), contract_id)
                result = self._replay(fixture)
                self.assertEqual(result["attempt_count"], 2)
                self.assertEqual(result["resume_count"], 1)
                self.assertEqual(
                    result["final_metadata"]["optimizer_updates"], 23_532
                )
                self.assertEqual(result["full_run_telemetry"]["attempt_count"], 2)
                self.assertEqual(result["full_run_telemetry"]["sample_rows"], 2)

    def test_two_attempt_replay_rejects_missing_first_attempt_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            runner._attempt_telemetry_paths(fixture["run_root"], 0)[
                "gpu_telemetry_summary"
            ].unlink()
            with self.assertRaisesRegex(runner.HeadlineM0Error, "artifact is missing"):
                self._replay(fixture)

    def test_two_attempt_replay_rejects_tampered_first_attempt_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            runner._attempt_telemetry_paths(fixture["run_root"], 0)[
                "gpu_telemetry"
            ].write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(runner.HeadlineM0Error, "identity changed"):
                self._replay(fixture)

    def test_final_checkpoint_identity_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            (fixture["run_root"] / "checkpoint_iter.pth").write_bytes(b"tampered")
            with self.assertRaisesRegex(runner.HeadlineM0Error, "identity changed"):
                self._replay(fixture)

    def test_safe_load_metadata_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            fixture["metadata"][1]["optimizer_updates"] = 23_531
            fixture["metadata"][1]["optimizer_step_values"] = [23_531]
            with self.assertRaisesRegex(runner.HeadlineM0Error, "final checkpoint"):
                self._replay(fixture)

    def test_resume_parent_chain_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            path = fixture["attempt_paths"][1]
            attempt = json.loads(path.read_text(encoding="ascii"))
            attempt["parent_attempt_manifest"]["sha256"] = "0" * 64
            _write_json(path, attempt)
            with self.assertRaisesRegex(runner.HeadlineM0Error, "resume edge"):
                self._replay(fixture)

    def test_m0n_and_m0_checkpoint_contracts_cannot_be_swapped(self):
        for contract_id, key, value in (
            ("M0N", "stage_b_v21_token_objective", "edit_bce"),
            ("M0", "stage_b_v25_control_of", "M0"),
        ):
            with self.subTest(contract_id=contract_id), tempfile.TemporaryDirectory() as tmp:
                fixture = self._fixture(Path(tmp), contract_id)
                fixture["metadata"][1]["args"][key] = value
                with self.assertRaises(runner.HeadlineM0Error):
                    self._replay(fixture)


class CompletedTrainingLaunchReplayTest(unittest.TestCase):
    def _fixture(self, root: Path, contract_id: str) -> dict:
        contract = runner.CONTRACTS[contract_id]
        seed = 17
        run_root = root / contract_id / "seed17"
        stage_a = root / "checkpoint0004.pth"
        source = root / "runner_source.py"
        dataset = root / "dataset.json"
        stage_a.write_bytes(b"stage-a-launch")
        source.write_text("VALUE = 1\n", encoding="ascii")
        dataset.write_text("{}\n", encoding="ascii")
        records = [
            runner._file_record(stage_a, role="stage_a_initializer"),
            runner._file_record(stage_a, role="scorer_warmstart"),
            runner._file_record(source, role="repository_source"),
        ]
        closure = _stable_closure(records)
        runtime = runner.Runtime(
            python=Path(sys.executable).resolve(),
            stage_a_init=stage_a.resolve(),
            dataset=dataset.resolve(),
            output_root=root,
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
        queue_binding = {"queue_id": "queue", "plan_sha256": "a" * 64}
        dataset_contract = {"synthetic": True}
        fixed = {
            "architecture_objective": "S2F",
            "compute_contract": "b58_successful_update_batch_slot_matched",
            "successful_update_batch_slots": 941_280,
            "candidate_topk": 50,
            "positive_iou_threshold": 0.5,
            "negative_iou_threshold": 0.499,
            "token_objective": (
                "edit_bce" if contract_id == "M0" else "targetlocal_allneg_bce"
            ),
            "token_objective_scope": contract.token_objective_scope,
            "predicate_pair_rank_weight": 1.0,
            "stage_a_and_scorer_same_source": True,
            "b58_model_ancestry_forbidden": True,
            "dataset": dataset_contract,
            "optimizer_resume": "same_run_mid_epoch_signal_only",
        }
        command = runner.build_command(runtime, contract, seed, run_root)
        common = {
            "schema": runner.TRAINING_PHASE_SCHEMA,
            "created_at_utc": "2026-07-19T00:00:00+00:00",
            "run_id": f"{contract_id}:{seed}",
            "row": contract.expected_row(),
            "seed": seed,
            "phase_id": "joint",
            "phase": runner._phase(contract),
            "output_dir": str(run_root),
            "command": command,
            "command_shell": shlex.join(command),
            "runtime": runner._runtime_payload(runtime),
            "fixed_contract": fixed,
            "inputs": {
                "records": records,
                "stable_closure_digest": closure["digest"],
            },
            "training_queue_binding": queue_binding,
        }
        planned = {**common, "status": "planned"}
        launch = {
            **common,
            "status": "completed",
            "started_at_utc": "2026-07-19T00:01:00+00:00",
            "gpu_environment": {},
            "gpu_telemetry_summary": {},
            "postflight": {},
            "postflight_artifact": {},
            "returncode": 0,
            "finished_at_utc": "2026-07-19T01:00:00+00:00",
        }
        sequence = {
            "schema": runner.TRAINING_SEQUENCE_SCHEMA,
            "status": "completed",
            "created_at_utc": "2026-07-19T00:00:01+00:00",
            "repository_root": str(runner.REPO_ROOT),
            "run_id": f"{contract_id}:{seed}",
            "row": contract.expected_row(),
            "seed": seed,
            "training_seeds_contract": list(contract.seeds),
            "output_dir": str(run_root),
            "output_dir_fresh_at_plan": True,
            "equal_budget_contract": contract.expected_budget(),
            "stable_input_closure_digest": closure["digest"],
            "one_attempt_execution": True,
            "resume_policy": "explicit_authorization_complete_same_run_mid_epoch_signal_only",
            "phases": [planned],
            "training_queue_binding": queue_binding,
            "started_at_utc": "2026-07-19T00:01:00+00:00",
            "finished_at_utc": "2026-07-19T01:00:00+00:00",
            "completed_phases": [],
            "training_attempt_count": 1,
            "same_run_resume_count": 0,
        }
        config = {}
        if contract_id == "M0N":
            config.update(
                {
                    "stage_b_v25_control_of": "M0",
                    "stage_b_v25_headline_eligible": False,
                    "stage_b_v25_matrix_validation_only": True,
                    "stage_b_v25_comparison_claim": "full_token_objective_control_not_labels_only",
                    "stage_b_v25_token_objective_scope": "target_local_positive_and_all_negative_token_logits",
                }
            )
        return {
            "contract": contract,
            "seed": seed,
            "run_root": run_root.resolve(),
            "stage_a": stage_a,
            "source": source,
            "dataset": dataset,
            "sequence": sequence,
            "launch": launch,
            "config": config,
            "dataset_contract": dataset_contract,
        }

    def _replay(self, fixture):
        stage_a_sha = hashlib.sha256(fixture["stage_a"].read_bytes()).hexdigest()
        with (
            mock.patch.object(runner, "DEFAULT_PYTHON", Path(sys.executable).resolve()),
            mock.patch.object(runner, "DEFAULT_STAGE_A_INIT", fixture["stage_a"].resolve()),
            mock.patch.object(runner, "DEFAULT_STAGE_A_SHA256", stage_a_sha),
            mock.patch.object(runner, "DEFAULT_DATASET", fixture["dataset"].resolve()),
            mock.patch.object(runner, "DEFAULT_OUTPUT_ROOT", fixture["run_root"].parent.parent),
            mock.patch.object(
                runner.source_contracts.FormalPaperRunContract,
                "canonical_training_root",
                return_value=fixture["run_root"],
            ),
            mock.patch.object(runner, "_validate_config", return_value=fixture["config"]),
            mock.patch.object(
                runner,
                "_validate_dataset",
                return_value=(fixture["dataset_contract"], []),
            ),
            mock.patch.object(
                runner,
                "_completed_training_verifier_source_paths",
                return_value=[fixture["source"].resolve()],
            ),
        ):
            return runner._verify_completed_launch_contract(
                fixture["sequence"],
                fixture["launch"],
                contract=fixture["contract"],
                seed=fixture["seed"],
                run_root=fixture["run_root"],
            )

    def test_m0_and_m0n_launch_contracts_replay(self):
        for contract_id in ("M0", "M0N"):
            with self.subTest(contract_id=contract_id), tempfile.TemporaryDirectory() as tmp:
                fixture = self._fixture(Path(tmp), contract_id)
                _, closure = self._replay(fixture)
                self.assertEqual(
                    closure["digest"],
                    fixture["sequence"]["stable_input_closure_digest"],
                )

    def test_runtime_budget_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp), "M0")
            fixture["launch"]["runtime"]["batch_size"] = 41
            with self.assertRaisesRegex(runner.HeadlineM0Error, "runtime batch_size"):
                self._replay(fixture)

    def test_m0n_objective_swap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp), "M0N")
            fixture["launch"]["fixed_contract"]["token_objective"] = "edit_bce"
            with self.assertRaisesRegex(runner.HeadlineM0Error, "fixed training"):
                self._replay(fixture)

    def test_source_closure_content_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp), "M0")
            fixture["source"].write_text("VALUE = 2\n", encoding="ascii")
            with self.assertRaisesRegex(runner.HeadlineM0Error, "stable input"):
                self._replay(fixture)


class CompletedTrainingPostflightReplayTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        run_root = root / "M0" / "seed42"
        run_root.mkdir(parents=True)
        source = root / "source.py"
        source.write_text("VALUE = 1\n", encoding="ascii")
        source_record = runner._file_record(source, role="repository_source")
        closure = _stable_closure([source_record])
        launch = {
            "inputs": {
                "records": [source_record],
                "stable_closure_digest": closure["digest"],
            }
        }
        checkpoint = run_root / "checkpoint_iter.pth"
        checkpoint.write_bytes(b"checkpoint")
        info = run_root / "info.txt"
        console = run_root / "train_console.log"
        info.write_text("stage_b_v22_branch_isolation_pass\n", encoding="ascii")
        console.write_text(
            "loss: 1.25 amp_step_skipped: 0 amp_scale: 65536 max mem: 100\n",
            encoding="ascii",
        )
        gpu_environment = {
            "schema": "pivot.gpu_environment/v1",
            "torch_runtime": {"cuda_available": True},
            "nvidia_devices": [
                {
                    "physical_index": 0,
                    "uuid": "GPU-test",
                    "name": "RTX Test",
                    "driver_version": "999.0",
                    "total_memory_mib": 32000.0,
                }
            ],
        }
        _write_json(run_root / "gpu_environment.json", gpu_environment)
        telemetry = run_root / "gpu_telemetry.csv"
        telemetry.write_text(
            runner.paper_launcher._GpuTelemetrySampler.HEADER
            + "2026/07/19 00:00:00.000, 0, GPU-test, RTX Test, 999.0, 32000, 1000, 31000, 5\n",
            encoding="ascii",
        )
        gpu_summary = runner.paper_launcher._summarize_nvidia_csv(telemetry)
        gpu_summary["captured_at_utc"] = "2026-07-19T00:00:01+00:00"
        gpu_summary["sampling_interval_ms"] = 1000
        _write_json(run_root / "gpu_telemetry_summary.json", gpu_summary)
        attempt_telemetry = runner._archive_attempt_telemetry(
            run_root=run_root,
            ordinal=0,
            gpu_environment=gpu_environment,
            gpu_summary=gpu_summary,
        )
        full_run_telemetry = runner._full_run_telemetry_projection(
            [attempt_telemetry]
        )
        scorer = run_root / "stage_b_v15_scorer_init_audit.json"
        _write_json(scorer, {"status": "applied"})
        input_rehash = runner.paper_launcher._rehash_inputs(launch)
        _write_json(run_root / "input_rehash.json", input_rehash)
        numerical = runner.paper_launcher._training_numerical_status(info, console)
        paths = {
            "checkpoint": checkpoint,
            "native_info_log": info,
            "train_console_log": console,
            "gpu_environment": run_root / "gpu_environment.json",
            "gpu_telemetry": telemetry,
            "gpu_telemetry_summary": run_root / "gpu_telemetry_summary.json",
            "scorer_init_audit": scorer,
            "input_rehash": run_root / "input_rehash.json",
        }
        artifacts = {name: runner._file_record(path) for name, path in paths.items()}
        metadata = {
            "optimizer_updates": 23_532,
            "checkpoint_cuda_memory": {},
        }
        progress = {
            "status": "passed",
            "optimizer_updates": 23_532,
            "consumed_microbatches": 23_532,
            "gradient_accumulation_steps": 1,
            "data_loader_microbatches_per_epoch": 8_388,
            "checkpoint_epoch": 2,
            "checkpoint_iteration": 6_756,
            "checkpoint_epoch_finished": False,
            "checkpoint_reason": "max_train_iters",
            "optimizer_state_count": 94,
            "optimizer_step_values": [23_532],
            "checkpoint_optimizer_step": 23_532,
            "successful_update_batch_slots": 941_280,
            "successful_updates_equal_consumed_microbatches": True,
        }
        scorer_wrapper = {"status": "passed"}
        milestones = {"status": "not_required"}
        postflight = {
            "schema": runner.POSTFLIGHT_SCHEMA,
            "status": "passed",
            "validated_at_utc": "2026-07-19T01:00:00+00:00",
            "run_id": "M0:42",
            "seed": 42,
            "phase_id": "joint",
            "checkpoint_metadata": metadata,
            "optimizer_progress": progress,
            "input_rehash": input_rehash,
            "gpu_environment": gpu_environment,
            "gpu_telemetry_summary": gpu_summary,
            "full_run_telemetry": full_run_telemetry,
            "numerical_status": numerical,
            "checkpoint_cuda_memory": {"available": False, "values": {}},
            "artifacts": artifacts,
            "model_state_ancestry": {},
            "scorer_initializer_audit": scorer_wrapper,
            "training_attempt_count": 1,
            "same_run_resume_count": 0,
            "milestones": milestones,
            "formal_claim": "successful_optimizer_update_batch_slot_matched_not_flop_or_wall_clock_matched",
        }
        _write_json(run_root / "postflight.json", postflight)
        postflight_record = runner._file_record(run_root / "postflight.json")
        launch.update(
            {
                "gpu_environment": gpu_environment,
                "gpu_telemetry_summary": gpu_summary,
                "postflight": copy.deepcopy(postflight),
                "postflight_artifact": postflight_record,
            }
        )
        sequence = {
            "completed_phases": [
                {
                    "phase_id": "joint",
                    "status": "completed",
                    "output_dir": str(run_root),
                    "checkpoint": artifacts["checkpoint"],
                    "postflight": postflight_record,
                }
            ]
        }
        runtime = mock.Mock()
        return {
            "run_root": run_root,
            "sequence": sequence,
            "launch": launch,
            "postflight": postflight,
            "runtime": runtime,
            "closure": closure,
            "metadata": metadata,
            "scorer_wrapper": scorer_wrapper,
            "milestones": milestones,
            "attempt_telemetry": attempt_telemetry,
            "full_run_telemetry": full_run_telemetry,
        }

    def _replay(self, fixture):
        attempts = {
            "attempt_count": 1,
            "resume_count": 0,
            "attempt_manifests": [],
            "attempt_telemetry": [fixture["attempt_telemetry"]],
            "full_run_telemetry": fixture["full_run_telemetry"],
            "final_metadata": fixture["metadata"],
            "ancestry": {},
        }
        with (
            mock.patch.object(
                runner, "_verify_completed_attempts", return_value=attempts
            ),
            mock.patch.object(
                runner,
                "_scorer_initializer_wrapper",
                return_value=fixture["scorer_wrapper"],
            ),
            mock.patch.object(
                runner, "_milestone_evidence", return_value=fixture["milestones"]
            ),
        ):
            return runner._verify_completed_postflight(
                sequence=fixture["sequence"],
                launch=fixture["launch"],
                postflight=fixture["postflight"],
                runtime=fixture["runtime"],
                contract=runner.CONTRACTS["M0"],
                seed=42,
                run_root=fixture["run_root"],
                stable_closure=fixture["closure"],
            )

    def test_numerical_amp_and_raw_telemetry_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._replay(self._fixture(Path(tmp)))
            self.assertEqual(result["numerical"]["max_amp_step_skipped"], 0.0)
            self.assertEqual(result["gpu_summary"]["sampling_interval_ms"], 1000)

    def test_embedded_numerical_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            fixture["postflight"]["numerical_status"]["max_amp_step_skipped"] = 1.0
            with self.assertRaisesRegex(runner.HeadlineM0Error, "AMP"):
                self._replay(fixture)

    def test_embedded_telemetry_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            fixture["postflight"]["gpu_telemetry_summary"]["sample_rows"] = 2
            with self.assertRaisesRegex(runner.HeadlineM0Error, "telemetry"):
                self._replay(fixture)


class CompletedTrainingEvidenceSnapshotTest(unittest.TestCase):
    def test_completed_evidence_inventory_includes_every_leaf_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "M0/seed17"
            recovery = run_root / "recovery/attempt_001_from_u000100_deadbeef0000.pth"
            authorization = run_root / "control/resume_requests/001.json"
            attempts = [
                {
                    "checkpoint_at_exit": {"path": str(recovery)},
                    "resume_authorization": None,
                },
                {
                    "checkpoint_at_exit": {
                        "path": str(run_root / "checkpoint_iter.pth")
                    },
                    "resume_authorization": {"path": str(authorization)},
                },
            ]
            observed = set(
                runner._completed_run_evidence_paths(
                    run_root, seed=17, attempts=attempts
                )
            )
            required = {
                run_root / "sequence_manifest.json",
                run_root / "launch_manifest.json",
                run_root / "postflight.json",
                run_root / "attempts/000/attempt_manifest.json",
                run_root / "attempts/000/input_closure.json",
                run_root / "attempts/000/train_console.log",
                run_root / "attempts/000/gpu_environment.json",
                run_root / "attempts/000/gpu_telemetry.csv",
                run_root / "attempts/000/gpu_telemetry_summary.json",
                run_root / "attempts/001/attempt_manifest.json",
                run_root / "attempts/001/gpu_environment.json",
                run_root / "attempts/001/gpu_telemetry.csv",
                run_root / "attempts/001/gpu_telemetry_summary.json",
                authorization,
                recovery,
                run_root / "milestones/checkpoint_iter_001000.pth",
                run_root / "milestones/checkpoint_iter_001000.json",
                run_root / "gpu_telemetry.csv",
                run_root / "input_rehash.json",
            }
            self.assertTrue(required.issubset(observed))

    def test_manifest_persistent_change_after_equality_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name in (
                "sequence_manifest.json",
                "launch_manifest.json",
                "postflight.json",
            ):
                path = root / name
                _write_json(path, {"status": "completed", "name": name})
                paths.append(path)
            before = runner._completed_evidence_snapshot(paths)
            for path in paths:
                runner._read_completed_json_stably(path, label=path.name)
            _write_json(paths[0], {"status": "tampered"})
            after = runner._completed_evidence_snapshot(paths)
            with self.assertRaisesRegex(runner.HeadlineM0Error, "evidence identity"):
                runner._require_same_completed_evidence(
                    before, after, label="completed run"
                )

    def test_every_nonroot_leaf_persistent_change_is_rejected(self):
        names = (
            "attempts/000/attempt_manifest.json",
            "attempts/000/input_closure.json",
            "attempts/000/train_console.log",
            "attempts/000/gpu_environment.json",
            "attempts/000/gpu_telemetry.csv",
            "attempts/000/gpu_telemetry_summary.json",
            "control/resume_requests/001.json",
            "recovery/attempt_001_from_u000100_deadbeef0000.pth",
            "milestones/checkpoint_iter_001000.pth",
            "milestones/checkpoint_iter_001000.json",
        )
        for selected in names:
            with self.subTest(path=selected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = []
                for name in names:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"sealed:{name}".encode("ascii"))
                    paths.append(path)
                before = runner._completed_evidence_snapshot(paths)
                (root / selected).write_bytes(b"persistently-tampered")
                with self.assertRaisesRegex(
                    runner.HeadlineM0Error, "evidence identity"
                ):
                    runner._verify_completed_evidence_current(
                        before, label="completed leaves"
                    )


class CompletedCheckpointSnapshotRaceTest(unittest.TestCase):
    def test_path_swap_and_restore_cannot_feed_different_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint_iter.pth"
            checkpoint.write_bytes(b"sealed-checkpoint")
            expected = runner._compact_file_record(
                runner._stable_completed_file_record(checkpoint)
            )

            def swapped_copy(source, directory):
                source = Path(source)
                source.write_bytes(b"swapped-checkpoint")
                copied = Path(directory) / "private.tmp"
                copied.write_bytes(source.read_bytes())
                digest = runner._sha256_file(copied)
                source.write_bytes(b"sealed-checkpoint")
                return copied, digest

            with (
                mock.patch.object(
                    runner, "_copy_checkpoint_to_temporary", side_effect=swapped_copy
                ),
                mock.patch.object(runner, "_inspect_checkpoint_extended") as inspect,
                self.assertRaisesRegex(
                    runner.HeadlineM0Error, "stable inspection snapshot"
                ),
            ):
                runner._inspect_completed_checkpoint_snapshot(
                    checkpoint,
                    expected_record=expected,
                    python=Path(sys.executable),
                    label="final checkpoint",
                )
            inspect.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"sealed-checkpoint")

    def test_inspector_receives_private_snapshot_not_live_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_iter.pth"
            checkpoint.write_bytes(b"sealed-checkpoint")
            expected = runner._compact_file_record(
                runner._stable_completed_file_record(checkpoint)
            )
            observed = []

            def inspect(path, *, stable_fd=None, **_kwargs):
                observed.append(Path(path).resolve())
                self.assertNotEqual(Path(path).resolve(), checkpoint.resolve())
                self.assertEqual(Path(path).read_bytes(), b"sealed-checkpoint")
                self.assertIsInstance(stable_fd, int)
                return {"optimizer_updates": 1}

            with mock.patch.object(
                runner, "_inspect_checkpoint_extended", side_effect=inspect
            ):
                metadata = runner._inspect_completed_checkpoint_snapshot(
                    checkpoint,
                    expected_record=expected,
                    python=Path(sys.executable),
                    label="final checkpoint",
                )
            self.assertEqual(metadata, {"optimizer_updates": 1})
            self.assertEqual(len(observed), 1)


class CompletedTrainingPublicReplayTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        fixture = CompletedTrainingAttemptReplayTest()._fixture(root, "M0")
        contract = fixture["contract"]
        seed = fixture["seed"]
        run_root = fixture["run_root"]
        runtime = fixture["runtime"]
        closure = fixture["closure"]
        created = "2026-07-18T23:59:58+00:00"
        started = "2026-07-19T00:00:00+00:00"
        finished = "2026-07-19T01:00:00+00:00"
        queue_binding = {
            "schema": "pivot.stageb.headline_m0_queue_verification/v1",
            "status": "passed",
            "contract_id": "M0",
            "queue_status": "running",
            "queue_id": "queue-id",
            "plan_sha256": "a" * 64,
            "ordered_run_ids": list(contract.dedicated_queue_run_ids),
            "queue_contract_sha256": "b" * 64,
            "stable_input_closure_digest": closure["digest"],
            "active_item": {
                "item_index": 1,
                "run_id": "M0:42",
                "item_status": "launched",
                "orchestration_root": str(root / "queue/jobs/001-M0_42"),
                "gpu_key": "0",
                "lease_path": str(root / "leases/gpu-0.json"),
            },
            "completion_verification": None,
            "completed_training_runs": [],
            "completion_semantic_sha256": None,
        }
        dataset_contract = {"synthetic": True}
        fixed = {
            "architecture_objective": "S2F",
            "compute_contract": "b58_successful_update_batch_slot_matched",
            "successful_update_batch_slots": 941_280,
            "candidate_topk": 50,
            "positive_iou_threshold": 0.5,
            "negative_iou_threshold": 0.499,
            "token_objective": "edit_bce",
            "token_objective_scope": None,
            "predicate_pair_rank_weight": 1.0,
            "stage_a_and_scorer_same_source": True,
            "b58_model_ancestry_forbidden": True,
            "dataset": dataset_contract,
            "optimizer_resume": "same_run_mid_epoch_signal_only",
        }
        fresh_command = runner.build_command(runtime, contract, seed, run_root)
        inputs = {
            "records": [
                runner._file_record(fixture["runtime"].stage_a_init, role="stage_a_initializer"),
                runner._file_record(fixture["runtime"].stage_a_init, role="scorer_warmstart"),
                runner._file_record(root / "source.py", role="repository_source"),
            ],
            "stable_closure_digest": closure["digest"],
            "stable_closure_algorithm": "sha256_canonical_path_content_size_roles_v1",
        }
        common = {
            "schema": runner.TRAINING_PHASE_SCHEMA,
            "created_at_utc": created,
            "run_id": "M0:42",
            "row": contract.expected_row(),
            "seed": seed,
            "phase_id": "joint",
            "phase": runner._phase(contract),
            "output_dir": str(run_root),
            "command": fresh_command,
            "command_shell": shlex.join(fresh_command),
            "runtime": runner._runtime_payload(runtime),
            "fixed_contract": fixed,
            "inputs": inputs,
            "training_queue_binding": queue_binding,
        }
        planned = {**common, "status": "planned"}
        final_command = fixture["launch"]["command"]
        launch = {
            **common,
            "status": "completed",
            "command": final_command,
            "command_shell": shlex.join(final_command),
            "started_at_utc": started,
            "gpu_environment": {},
            "gpu_telemetry_summary": {},
            "postflight": {},
            "postflight_artifact": {},
            "returncode": 0,
            "finished_at_utc": finished,
        }
        sequence = {
            "schema": runner.TRAINING_SEQUENCE_SCHEMA,
            "status": "completed",
            "created_at_utc": "2026-07-18T23:59:59+00:00",
            "repository_root": str(runner.REPO_ROOT),
            "run_id": "M0:42",
            "row": contract.expected_row(),
            "seed": seed,
            "training_seeds_contract": list(contract.seeds),
            "output_dir": str(run_root),
            "output_dir_fresh_at_plan": True,
            "equal_budget_contract": contract.expected_budget(),
            "stable_input_closure_digest": closure["digest"],
            "one_attempt_execution": True,
            "resume_policy": "explicit_authorization_complete_same_run_mid_epoch_signal_only",
            "phases": [planned],
            "training_queue_binding": queue_binding,
            "started_at_utc": started,
            "finished_at_utc": finished,
            "completed_phases": [],
            "training_attempt_count": 2,
            "same_run_resume_count": 1,
        }

        info = run_root / "info.txt"
        console = run_root / "train_console.log"
        info.write_text("stage_b_v22_branch_isolation_pass\n", encoding="ascii")
        console.write_text(
            "loss: 1.25 amp_step_skipped: 0 amp_scale: 65536 max mem: 100\n",
            encoding="ascii",
        )
        final_attempt_telemetry = runner._attempt_telemetry_paths(run_root, 1)
        gpu_environment = json.loads(
            final_attempt_telemetry["gpu_environment"].read_text(encoding="ascii")
        )
        _write_json(run_root / "gpu_environment.json", gpu_environment)
        telemetry = run_root / "gpu_telemetry.csv"
        telemetry.write_bytes(
            final_attempt_telemetry["gpu_telemetry"].read_bytes()
        )
        gpu_summary = json.loads(
            final_attempt_telemetry["gpu_telemetry_summary"].read_text(
                encoding="ascii"
            )
        )
        _write_json(run_root / "gpu_telemetry_summary.json", gpu_summary)
        launch["gpu_environment"] = gpu_environment
        launch["gpu_telemetry_summary"] = gpu_summary
        input_rehash = runner.paper_launcher._rehash_inputs(launch)
        _write_json(run_root / "input_rehash.json", input_rehash)
        numerical = runner.paper_launcher._training_numerical_status(info, console)
        paths = {
            "checkpoint": run_root / "checkpoint_iter.pth",
            "native_info_log": info,
            "train_console_log": console,
            "gpu_environment": run_root / "gpu_environment.json",
            "gpu_telemetry": telemetry,
            "gpu_telemetry_summary": run_root / "gpu_telemetry_summary.json",
            "scorer_init_audit": run_root / "stage_b_v15_scorer_init_audit.json",
            "input_rehash": run_root / "input_rehash.json",
        }
        artifacts = {name: runner._file_record(path) for name, path in paths.items()}
        progress = {
            "status": "passed",
            "optimizer_updates": 23_532,
            "consumed_microbatches": 23_532,
            "gradient_accumulation_steps": 1,
            "data_loader_microbatches_per_epoch": 8_388,
            "checkpoint_epoch": 2,
            "checkpoint_iteration": 6_756,
            "checkpoint_epoch_finished": False,
            "checkpoint_reason": "max_train_iters",
            "optimizer_state_count": 94,
            "optimizer_step_values": [23_532],
            "checkpoint_optimizer_step": 23_532,
            "successful_update_batch_slots": 941_280,
            "successful_updates_equal_consumed_microbatches": True,
        }
        stage_a_sha = runner._sha256_file(runtime.stage_a_init)
        scorer_wrapper = {
            "status": "passed",
            "applied": True,
            "source_path": str(runtime.stage_a_init),
            "source_sha256": stage_a_sha,
            "loaded_tensor_count": 90,
            "loaded_num_layers": 3,
            "artifact": artifacts["scorer_init_audit"],
            "same_as_stage_a_initializer": True,
            "b58_source": False,
        }
        milestones = {
            "status": "not_required",
            "reason": "diagnostic learning-curve checkpoints are retained only for seed17",
            "updates": [],
        }
        metadata = fixture["metadata"][-1]
        postflight = {
            "schema": runner.POSTFLIGHT_SCHEMA,
            "status": "passed",
            "validated_at_utc": finished,
            "run_id": "M0:42",
            "seed": seed,
            "phase_id": "joint",
            "checkpoint_metadata": metadata,
            "optimizer_progress": progress,
            "input_rehash": input_rehash,
            "gpu_environment": gpu_environment,
            "gpu_telemetry_summary": gpu_summary,
            "full_run_telemetry": fixture["postflight"]["full_run_telemetry"],
            "numerical_status": numerical,
            "checkpoint_cuda_memory": {"available": False, "values": {}},
            "artifacts": artifacts,
            "model_state_ancestry": fixture["postflight"]["model_state_ancestry"],
            "scorer_initializer_audit": scorer_wrapper,
            "training_attempt_count": 2,
            "same_run_resume_count": 1,
            "milestones": milestones,
            "formal_claim": "successful_optimizer_update_batch_slot_matched_not_flop_or_wall_clock_matched",
        }
        _write_json(run_root / "postflight.json", postflight)
        postflight_record = runner._file_record(run_root / "postflight.json")
        launch["postflight"] = copy.deepcopy(postflight)
        launch["postflight_artifact"] = postflight_record
        sequence["completed_phases"] = [
            {
                "phase_id": "joint",
                "status": "completed",
                "output_dir": str(run_root),
                "checkpoint": artifacts["checkpoint"],
                "postflight": postflight_record,
            }
        ]
        _write_json(run_root / "launch_manifest.json", launch)
        _write_json(run_root / "sequence_manifest.json", sequence)
        fixture.update(
            {
                "launch": launch,
                "sequence": sequence,
                "postflight": postflight,
                "source": root / "source.py",
                "dataset_contract": dataset_contract,
                "stage_a_sha": stage_a_sha,
            }
        )
        return fixture

    def _verify(self, fixture, *, stable_reader=None):
        metadata_by_path = {
            str(
                Path(
                    fixture["postflight"]["model_state_ancestry"][
                        "resume_ancestry"
                    ][0]["source_checkpoint"]["path"]
                ).resolve()
            ): fixture["metadata"][0],
            str((fixture["run_root"] / "checkpoint_iter.pth").resolve()): fixture[
                "metadata"
            ][1],
        }
        contexts = [
            mock.patch.object(runner, "DEFAULT_PYTHON", Path(sys.executable).resolve()),
            mock.patch.object(
                runner, "DEFAULT_STAGE_A_INIT", fixture["runtime"].stage_a_init
            ),
            mock.patch.object(runner, "DEFAULT_STAGE_A_SHA256", fixture["stage_a_sha"]),
            mock.patch.object(runner, "DEFAULT_OUTPUT_ROOT", fixture["run_root"].parent.parent),
            mock.patch.object(
                runner.source_contracts.FormalPaperRunContract,
                "canonical_training_root",
                return_value=fixture["run_root"],
            ),
            mock.patch.object(runner, "_validate_config", return_value={}),
            mock.patch.object(
                runner,
                "_validate_dataset",
                return_value=(fixture["dataset_contract"], []),
            ),
            mock.patch.object(
                runner,
                "_completed_training_verifier_source_paths",
                return_value=[fixture["source"].resolve()],
            ),
            mock.patch.object(
                runner,
                "_inspect_completed_checkpoint_snapshot",
                side_effect=lambda path, **_kwargs: copy.deepcopy(
                    metadata_by_path[str(Path(path).resolve())]
                ),
            ),
        ]
        if stable_reader is not None:
            contexts.append(
                mock.patch.object(
                    runner, "_read_completed_json_stably", side_effect=stable_reader
                )
            )
        with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6], contexts[7], contexts[8]:
            if len(contexts) == 10:
                with contexts[9]:
                    return runner.verify_completed_training_run(
                        fixture["run_root"], "M0", 42
                    )
            return runner.verify_completed_training_run(
                fixture["run_root"], "M0", 42
            )

    def test_public_single_run_success_replays_final_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._verify(fixture)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["run_id"], "M0:42")
            self.assertEqual(
                result["training_queue_binding"]["queue_id"], "queue-id"
            )
            self.assertEqual(
                result["semantic_sha256"],
                runner._sha256_bytes(
                    runner._canonical_json_bytes(
                        {
                            key: value
                            for key, value in result.items()
                            if key != "semantic_sha256"
                        }
                    )
                ),
            )

    def test_public_replay_rejects_persistent_manifest_change_after_equality_reread(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            original = runner._read_completed_json_stably

            def mutate_after_final_read(path, *, label):
                value = original(path, label=label)
                if label.endswith("postflight post-replay"):
                    _write_json(
                        fixture["run_root"] / "sequence_manifest.json",
                        {"status": "persistently-tampered"},
                    )
                return value

            with self.assertRaisesRegex(
                runner.HeadlineM0Error, "evidence identity changed"
            ):
                self._verify(fixture, stable_reader=mutate_after_final_read)


class CompletedTrainingFullQueueReplayTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        contract = runner.CONTRACTS["M0"]
        queue_dir = root / "queue"
        queue_dir.mkdir()
        _write_json(queue_dir / "queue.json", {"sealed": True})
        source = root / "source.py"
        source.write_text("VALUE = 1\n", encoding="ascii")
        source_record = runner._file_record(source, role="repository_source")
        normalized = [{**runner._compact_file_record(source_record), "roles": ["repository_source"]}]
        digest = runner._sha256_bytes(
            runner._canonical_json_bytes(
                {"schema": runner.STABLE_CLOSURE_SCHEMA, "records": normalized}
            )
        )
        queue_contract = {
            "schema": runner.TRAINING_QUEUE_CONTRACT_SCHEMA,
            "contract_id": "M0",
            "ordered_run_ids": list(contract.dedicated_queue_run_ids),
            "runner": runner._compact_file_record(
                runner._file_record(Path(runner.__file__))
            ),
            "controller_python": runner._compact_file_record(
                runner._file_record(Path(sys.executable))
            ),
            "stable_input_closure": {
                "schema": runner.STABLE_CLOSURE_SCHEMA,
                "algorithm": "sha256_canonical_path_content_size_roles_v1",
                "digest": digest,
                "records": normalized,
            },
        }
        queue_contract_sha = runner._sha256_bytes(
            runner._canonical_json_bytes(queue_contract)
        )
        lease_path = root / "leases/gpu-0.json"
        plan_items = [
            {"run_id": run_id, "runner": "paper"}
            for run_id in contract.dedicated_queue_run_ids
        ]
        mutable_items = []
        projections = []
        serial_items = []
        for index, run_id in enumerate(contract.dedicated_queue_run_ids):
            seed = int(run_id.split(":", 1)[1])
            output_root = root / "outputs" / "M0" / f"seed{seed}"
            output_root.mkdir(parents=True)
            orchestration_root = queue_dir / "jobs" / f"{index:03d}-M0_{seed}"
            sequence = {
                "run_id": run_id,
                "status": "completed",
                "stable_input_closure_digest": digest,
            }
            launch = {
                "inputs": {
                    "records": [source_record],
                    "stable_closure_digest": digest,
                }
            }
            _write_json(output_root / "sequence_manifest.json", sequence)
            _write_json(output_root / "launch_manifest.json", launch)
            evidence = runner._completed_evidence_snapshot(
                [
                    output_root / "sequence_manifest.json",
                    output_root / "launch_manifest.json",
                ]
            )
            input_snapshot = runner._completed_stable_input_snapshot(
                normalized, label=f"{run_id} inputs"
            )
            binding = {
                "contract_id": "M0",
                "queue_id": "queue-id",
                "plan_sha256": "a" * 64,
                "queue_contract_sha256": queue_contract_sha,
                "stable_input_closure_digest": digest,
                "ordered_run_ids": list(contract.dedicated_queue_run_ids),
                "active_item": {
                    "item_index": index,
                    "run_id": run_id,
                    "item_status": "launched",
                    "orchestration_root": str(orchestration_root.resolve()),
                    "gpu_key": "0",
                    "lease_path": str(lease_path),
                },
            }
            projection = {
                "schema": runner.COMPLETED_TRAINING_VERIFICATION_SCHEMA,
                "status": "passed",
                "run_id": run_id,
                "contract_id": "M0",
                "seed": seed,
                "training_queue_binding": binding,
                "input_closure": {
                    "digest": digest,
                    "identity_snapshot": input_snapshot,
                },
                "evidence_snapshot": evidence,
            }
            projection["semantic_sha256"] = runner._sha256_bytes(
                runner._canonical_json_bytes(projection)
            )
            projections.append(projection)
            mutable_items.append(
                {
                    "index": index,
                    "run_id": run_id,
                    "runner": "paper",
                    "status": "completed",
                    "output_root": str(output_root),
                    "orchestration_root": str(orchestration_root),
                }
            )
            serial_items.append(
                {
                    "schema": "pivot.stageb.serial_matrix_queue_completion/v1",
                    "verified_at_utc": "2026-07-19T01:00:00+00:00",
                    "run_id": run_id,
                    "runner": "paper",
                    "job_dir": str(orchestration_root / "job"),
                    "output_root": str(output_root),
                    "sequence_manifest": str(output_root / "sequence_manifest.json"),
                    "sequence_sha256": runner._sha256_file(
                        output_root / "sequence_manifest.json"
                    ),
                    "phases": [],
                    "advance_gate": "completed",
                }
            )
        queue = {
            "status": "completed",
            "plan_sha256": "a" * 64,
            "plan": {
                "queue_id": "queue-id",
                "runner_python": str(Path(sys.executable).resolve()),
                "items": plan_items,
                "runners": {
                    "paper": {
                        "path": str(Path(runner.__file__).resolve()),
                        "sha256": runner._sha256_file(Path(runner.__file__)),
                    }
                },
                "extensions": {
                    runner.TRAINING_QUEUE_EXTENSION_KEY: queue_contract
                },
                "runtime_environment": {},
                "gpu_key": "0",
                "lease_path": str(lease_path),
            },
            "items": mutable_items,
        }
        serial = {
            "schema": "pivot.stageb.serial_matrix_queue_verification/v1",
            "verified_at_utc": "2026-07-19T01:00:00+00:00",
            "status": "passed",
            "queue_status": "completed",
            "queue_id": "queue-id",
            "plan_sha256": "a" * 64,
            "verified_items": serial_items,
            "errors": [],
        }
        source_snapshot = {
            "schema": runner.COMPLETED_TRAINING_SOURCE_SNAPSHOT_SCHEMA,
            "algorithm": "sha256_canonical_path_content_size_v1",
            "records": [runner._compact_file_record(source_record)],
            "digest": runner._sha256_bytes(
                runner._canonical_json_bytes(
                    {
                        "schema": runner.COMPLETED_TRAINING_SOURCE_SNAPSHOT_SCHEMA,
                        "records": [runner._compact_file_record(source_record)],
                    }
                )
            ),
        }
        return {
            "queue_dir": queue_dir,
            "queue": queue,
            "serial": serial,
            "projections": projections,
            "source": source,
            "source_snapshot": source_snapshot,
        }

    def _verify(self, fixture, *, run_side_effect=None, serial_side_effect=None):
        from tools import run_stageb_serial_matrix_queue as serial_queue

        runs = run_side_effect if run_side_effect is not None else fixture["projections"]
        serial = serial_side_effect if serial_side_effect is not None else fixture["serial"]
        with (
            mock.patch.object(runner, "DEFAULT_PYTHON", Path(sys.executable).resolve()),
            mock.patch.object(serial_queue, "load_queue", return_value=fixture["queue"]),
            mock.patch.object(serial_queue, "verify_queue", side_effect=([serial] if not callable(serial) else serial)),
            mock.patch.object(runner, "_validate_queue_runtime_snapshot"),
            mock.patch.object(
                runner,
                "_completed_training_verifier_source_snapshot",
                return_value=fixture["source_snapshot"],
            ),
            mock.patch.object(
                runner,
                "verify_completed_training_run",
                side_effect=runs,
            ),
        ):
            return runner.verify_training_queue(
                fixture["queue_dir"], "M0", require_completed=True
            )

    def test_full_three_seed_queue_replays_and_binds_completion_semantic(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._verify(fixture)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["completed_training_runs"]), 3)
            self.assertEqual(result["queue_manifest"]["path"], str(fixture["queue_dir"] / "queue.json"))
            self.assertIsNotNone(result["completion_semantic_sha256"])
            self.assertEqual(
                [
                    item["run_id"]
                    for item in result["serial_completion_evidence"]["verified_items"]
                ],
                ["M0:17", "M0:42", "M0:73"],
            )

    def test_wrong_embedded_queue_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            projections = copy.deepcopy(fixture["projections"])
            projections[1]["training_queue_binding"]["queue_id"] = "other-queue"
            with self.assertRaisesRegex(runner.HeadlineM0Error, "queue queue_id"):
                self._verify(fixture, run_side_effect=projections)

    def test_queue_manifest_change_during_three_seed_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            projections = iter(fixture["projections"])

            def replay(*_args, **_kwargs):
                value = next(projections)
                if value["run_id"] == "M0:42":
                    _write_json(fixture["queue_dir"] / "queue.json", {"sealed": False})
                return value

            with self.assertRaisesRegex(runner.HeadlineM0Error, "queue changed"):
                self._verify(fixture, run_side_effect=replay)

    def test_stable_input_change_during_three_seed_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            projections = iter(fixture["projections"])

            def replay(*_args, **_kwargs):
                value = next(projections)
                if value["run_id"] == "M0:73":
                    fixture["source"].write_text("VALUE = 2\n", encoding="ascii")
                return value

            with self.assertRaisesRegex(runner.HeadlineM0Error, "stable input"):
                self._verify(fixture, run_side_effect=replay)


class CompletedTrainingSourceRaceTest(unittest.TestCase):
    def test_public_verifier_rejects_source_change_during_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "seed17"
            run_root.mkdir()
            before = {
                "schema": runner.COMPLETED_TRAINING_SOURCE_SNAPSHOT_SCHEMA,
                "algorithm": "sha256_canonical_path_content_size_v1",
                "records": [],
                "digest": "a" * 64,
            }
            after = {**before, "digest": "b" * 64}
            with (
                mock.patch.object(
                    runner,
                    "_completed_training_verifier_source_snapshot",
                    side_effect=[before, after],
                ),
                mock.patch.object(
                    runner,
                    "_verify_completed_training_run_replay",
                    return_value={"status": "passed"},
                ),
                self.assertRaisesRegex(runner.HeadlineM0Error, "source identity"),
            ):
                runner.verify_completed_training_run(run_root, "M0", 17)


if __name__ == "__main__":
    unittest.main()
