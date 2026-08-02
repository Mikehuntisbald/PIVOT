import hashlib
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import stageb_headline_release_contract as release
from tools import run_stageb_paper_evaluations as evaluator


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _attempt_telemetry(run_root: Path, ordinal: int) -> dict:
    attempt_dir = run_root / "attempts" / f"{ordinal:03d}"
    environment = attempt_dir / "gpu_environment.json"
    telemetry = attempt_dir / "gpu_telemetry.csv"
    summary = attempt_dir / "gpu_telemetry_summary.json"
    _write_json(environment, {"schema": "pivot.gpu_environment/v1"})
    telemetry.write_text("header\nsample\n", encoding="utf-8")
    _write_json(summary, {"schema": "pivot.gpu_telemetry_summary/v1"})
    return {
        "schema": release.M0_ATTEMPT_TELEMETRY_SCHEMA,
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
            "gpu_environment": release.file_record(environment),
            "gpu_telemetry": release.file_record(telemetry),
            "gpu_telemetry_summary": release.file_record(summary),
        },
    }


class HeadlineReleaseContractTest(unittest.TestCase):
    def test_m0_mid_epoch_reason_rejects_signal_after_epoch(self):
        self.assertEqual(
            release._require_m0_mid_epoch_signal("signal", label="test"), "signal"
        )
        with self.assertRaisesRegex(
            release.HeadlineReleaseError, "mid-epoch signal"
        ):
            release._require_m0_mid_epoch_signal(
                "signal_after_epoch", label="test"
            )

    def test_m0_ancestry_replays_contiguous_same_run_resume_and_rejects_b58(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "M0" / "seed17"
            attempts_root = root / "attempts"
            recovery_root = root / "recovery"
            attempts_root.mkdir(parents=True)
            recovery_root.mkdir(parents=True)
            stage_a = Path(temporary) / "checkpoint0004.pth"
            stage_a.write_bytes(b"stage-a")
            stage_a_record = release.file_record(stage_a)
            final_checkpoint = root / "checkpoint_iter.pth"
            final_checkpoint.write_bytes(b"final-u23532")
            final_record = release.file_record(final_checkpoint)
            scorer_artifact = root / "stage_b_v15_scorer_init_audit.json"
            _write_json(
                scorer_artifact,
                {
                    "schema": "stage_b_v15_scorer_init/v1",
                    "status": "applied",
                    "source_sha256": stage_a_record["sha256"],
                    "resolved_source_path": str(stage_a.resolve()),
                    "loaded_tensor_count": 90,
                    "loaded_num_layers": 3,
                },
            )
            scorer_record = release.file_record(scorer_artifact)
            scorer_wrapper = {
                "status": "passed",
                "applied": True,
                "source_path": str(stage_a.resolve()),
                "source_sha256": stage_a_record["sha256"],
                "loaded_tensor_count": 90,
                "loaded_num_layers": 3,
                "artifact": scorer_record,
                "same_as_stage_a_initializer": True,
                "b58_source": False,
            }
            runtime = {
                "batch_size": 40,
                "num_workers": 2,
                "prefetch_factor": 1,
                "amp": True,
                "gradient_accumulation_steps": 1,
                "max_train_iters": 23532,
                "iter_checkpoint_interval": 500,
                "gradient_diagnostic_interval": 100,
                "telemetry_interval_seconds": 1,
                "pin_memory": True,
                "persistent_workers": False,
            }
            scorer_option = (
                f"stage_b_v15_scorer_init_checkpoint={stage_a.resolve()}"
            )
            stable_input_closure = release._stable_training_input_closure(
                [{**stage_a_record, "role": "stage_a_initializer"}]
            )

            temporary_recovery = recovery_root / "temporary.pth"
            temporary_recovery.write_bytes(b"recovery-u100")
            recovery_sha = release.sha256_file(temporary_recovery)
            recovery = recovery_root / (
                f"attempt_001_from_u000100_{recovery_sha[:12]}.pth"
            )
            temporary_recovery.rename(recovery)
            recovery_record = release.file_record(recovery)
            authorization_path = root / "control" / "resume_requests" / "001.json"
            _write_json(
                authorization_path,
                {
                    "schema": "pivot.stageb.headline_m0_resume_request/v1",
                    "status": "authorized",
                    "run_id": "M0:17",
                    "next_attempt_ordinal": 1,
                    "recovery_checkpoint": recovery_record,
                    "policy": "explicit_one_attempt_mid_epoch_signal_resume",
                    "authorized_at_utc": "2026-07-18T00:01:00+00:00",
                    "authorizer_pid": 4242,
                    "detached_controller_identity": None,
                },
            )
            authorization_record = release.file_record(authorization_path)

            attempt0_dir = attempts_root / "000"
            attempt0_dir.mkdir()
            closure_payload = {
                "schema": release.M0_STABLE_CLOSURE_SCHEMA,
                "status": "sealed",
                "algorithm": "sha256_canonical_path_content_size_roles_v1",
                "records": stable_input_closure["records"],
                "digest": stable_input_closure["digest"],
            }
            closure0_path = attempt0_dir / "input_closure.json"
            _write_json(closure0_path, closure_payload)
            closure0_record = release.file_record(closure0_path)
            command0 = [
                "python",
                "main.py",
                "--pretrain_model_path",
                str(stage_a.resolve()),
                "--options",
                scorer_option,
            ]
            attempt0 = {
                "schema": release.M0_ATTEMPT_SCHEMA,
                "status": "completed",
                "run_id": "M0:17",
                "seed": 17,
                "attempt_ordinal": 0,
                "initialization_mode": "fresh_stage_a",
                "parent_attempt_manifest": None,
                "resume_checkpoint": None,
                "resume_authorization": None,
                "source_optimizer_updates": 0,
                "target_optimizer_updates": 23532,
                "command": command0,
                "command_shell": shlex.join(command0),
                "runtime": runtime,
                "input_closure_digest": stable_input_closure["digest"],
                "input_closure": closure0_record,
                "telemetry": _attempt_telemetry(root, 0),
                "process": {"returncode": 0},
                "termination": {
                    "kind": "graceful_signal_checkpoint",
                    "reason": "signal",
                },
                "complete_state_components": dict(
                    release.M0_COMPLETE_STATE_COMPONENTS
                ),
                "checkpoint_at_exit": recovery_record,
                "checkpoint_metadata": {
                    "optimizer_updates": 100,
                    "optimizer_state_count": 94,
                    "optimizer_step_values": [100],
                    "complete_state_components": dict(
                        release.M0_COMPLETE_STATE_COMPONENTS
                    ),
                    "checkpoint_reason": "signal",
                },
                "started_at_utc": "2026-07-18T00:00:00+00:00",
                "finished_at_utc": "2026-07-18T00:01:00+00:00",
            }
            attempt0_path = attempt0_dir / "attempt_manifest.json"
            _write_json(attempt0_path, attempt0)
            attempt0_record = release.file_record(attempt0_path)

            attempt1_dir = attempts_root / "001"
            attempt1_dir.mkdir()
            closure1_path = attempt1_dir / "input_closure.json"
            _write_json(closure1_path, closure_payload)
            closure1_record = release.file_record(closure1_path)
            command1 = [
                "python",
                "main.py",
                "--resume",
                str(recovery.resolve()),
                "--options",
                scorer_option,
            ]
            attempt1 = {
                **attempt0,
                "attempt_ordinal": 1,
                "initialization_mode": "same_run_resume",
                "parent_attempt_manifest": attempt0_record,
                "resume_checkpoint": recovery_record,
                "resume_authorization": authorization_record,
                "source_optimizer_updates": 100,
                "command": command1,
                "command_shell": shlex.join(command1),
                "input_closure": closure1_record,
                "telemetry": _attempt_telemetry(root, 1),
                "termination": {
                    "kind": "target_completed",
                    "reason": "max_train_iters",
                },
                "checkpoint_at_exit": final_record,
                "checkpoint_metadata": {
                    "optimizer_updates": 23532,
                    "optimizer_state_count": 94,
                    "optimizer_step_values": [23532],
                    "complete_state_components": dict(
                        release.M0_COMPLETE_STATE_COMPONENTS
                    ),
                    "checkpoint_reason": "max_train_iters",
                },
                "started_at_utc": "2026-07-18T00:01:00+00:00",
                "finished_at_utc": "2026-07-18T01:00:00+00:00",
            }
            attempt1_path = attempt1_dir / "attempt_manifest.json"
            _write_json(attempt1_path, attempt1)
            attempt1_record = release.file_record(attempt1_path)

            ancestry = {
                "schema": release.M0_ANCESTRY_SCHEMA,
                "fresh_start": {
                    "run_id": "M0:17",
                    "initialization_mode": (
                        "pretrain_model_path_plus_same_source_scorer_init"
                    ),
                    "pretrain": {
                        **stage_a_record,
                        "role": "stage_a_initializer",
                    },
                    "scorer": {
                        **stage_a_record,
                        "role": "scorer_warmstart",
                    },
                    "same_source": True,
                    "resume_argument": None,
                    "attempt_manifest": attempt0_record,
                },
                "resume_ancestry": [
                    {
                        "ordinal": 1,
                        "run_id": "M0:17",
                        "source_checkpoint": recovery_record,
                        "source_optimizer_updates": 100,
                        "source_checkpoint_reason": "signal",
                        "complete_training_state": True,
                        "same_run": True,
                        "resume_authorization": authorization_record,
                        "attempt_manifest": attempt1_record,
                    }
                ],
                "ultimate_pretrain": {
                    "path": str(stage_a.resolve()),
                    "sha256": stage_a_record["sha256"],
                    "role": "stage_a_initializer",
                },
                "ultimate_scorer": {
                    "path": str(stage_a.resolve()),
                    "sha256": stage_a_record["sha256"],
                    "role": "scorer_warmstart",
                },
                "ultimate_same_stage_a_source": True,
                "resume_chain_contiguous": True,
                "b58_ancestry_count": 0,
                "b58_ancestry_paths": [],
                "b58_ancestry_sha256s": [],
            }
            inspected = {
                recovery.resolve(): attempt0["checkpoint_metadata"],
                final_checkpoint.resolve(): attempt1["checkpoint_metadata"],
            }
            with mock.patch.object(
                release,
                "_inspect_m0_training_checkpoint",
                side_effect=lambda path: inspected[path.resolve()],
            ):
                result = release._validate_m0_ancestry(
                    ancestry,
                    scorer_wrapper,
                    run_id="M0:17",
                    seed=17,
                    run_root=root,
                    stage_a_path=stage_a.resolve(),
                    stage_a_sha256=stage_a_record["sha256"],
                    final_checkpoint=final_record,
                    scorer_audit_artifact=scorer_record,
                    stable_input_closure=stable_input_closure,
                )
            self.assertEqual(result["attempt_count"], 2)
            self.assertEqual(result["resume_count"], 1)

            forged_signal_reason = json.loads(json.dumps(ancestry))
            forged_signal_reason["resume_ancestry"][0][
                "source_checkpoint_reason"
            ] = "signal_after_epoch"
            with self.assertRaisesRegex(
                release.HeadlineReleaseError, "mid-epoch signal"
            ):
                release._validate_m0_ancestry(
                    forged_signal_reason,
                    scorer_wrapper,
                    run_id="M0:17",
                    seed=17,
                    run_root=root,
                    stage_a_path=stage_a.resolve(),
                    stage_a_sha256=stage_a_record["sha256"],
                    final_checkpoint=final_record,
                    scorer_audit_artifact=scorer_record,
                    stable_input_closure=stable_input_closure,
                )

            forged_authorization = json.loads(json.dumps(ancestry))
            forged_authorization["resume_ancestry"][0]["resume_authorization"][
                "sha256"
            ] = "0" * 64
            with (
                mock.patch.object(
                    release,
                    "_inspect_m0_training_checkpoint",
                    side_effect=lambda path: inspected[path.resolve()],
                ),
                self.assertRaisesRegex(
                    release.HeadlineReleaseError, "authorization"
                ),
            ):
                release._validate_m0_ancestry(
                    forged_authorization,
                    scorer_wrapper,
                    run_id="M0:17",
                    seed=17,
                    run_root=root,
                    stage_a_path=stage_a.resolve(),
                    stage_a_sha256=stage_a_record["sha256"],
                    final_checkpoint=final_record,
                    scorer_audit_artifact=scorer_record,
                    stable_input_closure=stable_input_closure,
                )

            forged = json.loads(json.dumps(ancestry))
            forged["b58_ancestry_count"] = 1
            forged["b58_ancestry_sha256s"] = [
                release.FIXED_BASELINE["checkpoint_sha256"]
            ]
            with (
                mock.patch.object(
                    release,
                    "_inspect_m0_training_checkpoint",
                    side_effect=lambda path: inspected[path.resolve()],
                ),
                self.assertRaisesRegex(
                    release.HeadlineReleaseError, "Stage-A-only/no-b58"
                ),
            ):
                release._validate_m0_ancestry(
                    forged,
                    scorer_wrapper,
                    run_id="M0:17",
                    seed=17,
                    run_root=root,
                    stage_a_path=stage_a.resolve(),
                    stage_a_sha256=stage_a_record["sha256"],
                    final_checkpoint=final_record,
                    scorer_audit_artifact=scorer_record,
                    stable_input_closure=stable_input_closure,
                )

            with (
                mock.patch.object(
                    release,
                    "_inspect_m0_training_checkpoint",
                    return_value={
                        **attempt0["checkpoint_metadata"],
                        "optimizer_updates": 99,
                    },
                ),
                self.assertRaisesRegex(
                    release.HeadlineReleaseError, "safe-load replay"
                ),
            ):
                release._validate_m0_ancestry(
                    ancestry,
                    scorer_wrapper,
                    run_id="M0:17",
                    seed=17,
                    run_root=root,
                    stage_a_path=stage_a.resolve(),
                    stage_a_sha256=stage_a_record["sha256"],
                    final_checkpoint=final_record,
                    scorer_audit_artifact=scorer_record,
                    stable_input_closure=stable_input_closure,
                )

            with (
                mock.patch.object(
                    release,
                    "_inspect_m0_training_checkpoint",
                    side_effect=lambda path: inspected[path.resolve()],
                ),
                self.assertRaisesRegex(
                    release.HeadlineReleaseError, "closure replay drifted"
                ),
            ):
                release._validate_m0_ancestry(
                    ancestry,
                    scorer_wrapper,
                    run_id="M0:17",
                    seed=17,
                    run_root=root,
                    stage_a_path=stage_a.resolve(),
                    stage_a_sha256=stage_a_record["sha256"],
                    final_checkpoint=final_record,
                    scorer_audit_artifact=scorer_record,
                    stable_input_closure={
                        **stable_input_closure,
                        "digest": "b" * 64,
                    },
                )

            closure0_path.write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(
                release.HeadlineReleaseError, "digest/size changed"
            ):
                release._validate_m0_ancestry(
                    ancestry,
                    scorer_wrapper,
                    run_id="M0:17",
                    seed=17,
                    run_root=root,
                    stage_a_path=stage_a.resolve(),
                    stage_a_sha256=stage_a_record["sha256"],
                    final_checkpoint=final_record,
                    scorer_audit_artifact=scorer_record,
                    stable_input_closure=stable_input_closure,
                )

    def test_m0_compute_contract_is_not_table_d_u1000(self):
        self.assertEqual(release.CANDIDATE_ID, "M0")
        self.assertEqual(release.CANDIDATE_ARCHITECTURE_OBJECTIVE, "S2F")
        self.assertEqual(release.CANDIDATE_UPDATES, 23532)
        self.assertEqual(
            release.CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS, 941280
        )
        self.assertEqual(release.BASELINE_OPTIMIZER_UPDATES, 49539)
        self.assertEqual(
            release.BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS, 941241
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "M0" / "seed17"
            root.mkdir(parents=True)
            row = {
                "row_id": "M0",
                "config": "config/ablations/cfg_stageb_v25_m0_compute_matched.py",
                "architecture_objective": "S2F",
                "compute_contract": "b58_successful_update_batch_slot_matched",
                "datasets": "config/datasets_stageb_v21_single_edit_train.json",
            }
            sequence = {
                "training_seeds_contract": [17, 42, 73],
                "equal_budget_contract": {
                    "batch_size": 40,
                    "optimizer_updates": 23532,
                    "contributing_phase_updates": {"joint": 23532},
                    "successful_update_batch_slots": 941280,
                },
            }
            with mock.patch.object(
                evaluator.source_contracts.FormalPaperRunContract,
                "canonical_training_root",
                return_value=root,
            ):
                evaluator._validate_paper_formal_run_contract(
                    sequence=sequence,
                    run_root=root,
                    row=row,
                    row_id="M0",
                    seed=17,
                )
                sequence["equal_budget_contract"] = {
                    "batch_size": 40,
                    "optimizer_updates": 1000,
                    "contributing_phase_updates": {"joint": 1000},
                    "successful_update_batch_slots": 40000,
                }
                with self.assertRaisesRegex(
                    evaluator.PaperEvaluationError, "B40/U23532"
                ):
                    evaluator._validate_paper_formal_run_contract(
                        sequence=sequence,
                        run_root=root,
                        row=row,
                        row_id="M0",
                        seed=17,
                    )

    def test_eligibility_is_fixed_s2f_no_fallback_and_uses_point_seed_mean(self):
        baseline = {
            "ref": {split: 0.50 for split in release.VALIDATION_REF_SPLITS},
            "fpr95": 0.50,
            "positive_q05": 0.30,
        }
        candidates = {
            seed: {
                "ref": {
                    "refcoco_val": 0.51,
                    "refcocop_val": 0.52,
                    "refcocog_val": 0.53,
                },
                "fpr95": 0.40,
                "positive_q05": 0.29,
            }
            for seed in release.CANDIDATE_SEEDS
        }
        result = release.evaluate_selection_eligibility(baseline, candidates)
        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate_id"], "M0")
        self.assertEqual(result["candidate_seeds"], [17, 42, 73])
        self.assertEqual(result["selection_policy"], release.SELECTION_POLICY)

        candidates[73]["ref"]["refcocog_val"] = 0.40
        failed = release.evaluate_selection_eligibility(baseline, candidates)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["gates"]["ref_split_noninferiority"]["passed"])
        self.assertEqual(failed["fallback_policy"], "none_close_final_gate")

    def test_fixed_baseline_rejects_any_path_id_or_digest_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            config.write_text("stage_b = False\n", encoding="utf-8")
            checkpoint.write_bytes(b"b58")
            contract = {
                "id": release.BASELINE_ID,
                "train_seed": 42,
                "role": "fixed_historical_checkpoint",
                "config": str(config.resolve()),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            with mock.patch.object(release, "FIXED_BASELINE", contract):
                release.validate_fixed_baseline_identity(
                    evaluation_id=release.BASELINE_ID,
                    config=config,
                    checkpoint=checkpoint,
                    checkpoint_sha256=contract["checkpoint_sha256"],
                )
                with self.assertRaisesRegex(
                    release.HeadlineReleaseError, "fixed b58"
                ):
                    release.validate_fixed_baseline_identity(
                        evaluation_id="other",
                        config=config,
                        checkpoint=checkpoint,
                        checkpoint_sha256=contract["checkpoint_sha256"],
                    )

    def test_gate_consumption_is_canonical_fresh_and_single_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_root = root / "contracts"
            consumption_root = root / "consumptions"
            output = root / "final" / "M0" / "seed17"
            artifacts = {}
            for name in release.RECEIPT_ARTIFACT_NAMES:
                path = contract_root / f"{name}.json"
                payload = {"schema": name, "status": "sealed"}
                _write_json(path, payload)
                artifacts[name] = release.file_record(path)
            completion_path = root / "completion.json"
            completion = {
                "schema": release.COMPLETION_RECEIPT_SCHEMA,
                "status": "completed",
                "completed_before_final_gate": True,
                "all_training_validation_diagnostics_completed": True,
                "completed_blocks": {
                    block: {
                        "status": "completed",
                        "artifacts": [artifacts["selection_receipt"]],
                    }
                    for block in release.COMPLETION_BLOCKS
                },
            }
            completion["receipt_sha256"] = release.canonical_json_sha256(completion)
            _write_json(completion_path, completion)
            completion_record = release.file_record(completion_path)
            exposure_path = root / "exposure.json"
            _write_json(exposure_path, {"status": "verified"})
            exposure_record = release.file_record(exposure_path)
            instance = {
                "schema": release.INSTANCE_SCHEMA,
                "instance_id": "M0:17",
                "role": "candidate",
                "candidate_id": "M0",
                "train_seed": 17,
                "output_dir": str(output.resolve()),
                "profile": "final",
            }
            instance["instance_sha256"] = release.canonical_json_sha256(instance)
            instances = []
            for instance_id in (
                release.BASELINE_ID,
                "M0:17",
                "M0:42",
                "M0:73",
            ):
                value = {
                    **instance,
                    "instance_id": instance_id,
                    "role": (
                        "baseline"
                        if instance_id == release.BASELINE_ID
                        else "candidate"
                    ),
                    "candidate_id": (
                        None
                        if instance_id == release.BASELINE_ID
                        else release.CANDIDATE_ID
                    ),
                    "train_seed": (
                        42
                        if instance_id == release.BASELINE_ID
                        else int(instance_id.rsplit(":", 1)[1])
                    ),
                    "output_dir": (
                        str(output.resolve())
                        if instance_id == "M0:17"
                        else str((root / "other" / instance_id.replace(":", "_")).resolve())
                    ),
                }
                value.pop("instance_sha256", None)
                value["instance_sha256"] = release.canonical_json_sha256(value)
                instances.append(value)
            instance = instances[1]
            gate_payload = {
                "schema": release.FINAL_GATE_SCHEMA,
                "status": "sealed",
                "selection_frozen": True,
                "created_before_first_final_evaluation": True,
                "selection_policy": release.SELECTION_POLICY,
                "fallback_policy": "none_close_final_gate",
                "headline_bootstrap": dict(release.HEADLINE_BOOTSTRAP_CONTRACT),
                "instances": instances,
                "receipt_artifacts": artifacts,
                "paper_ablation_completion_receipt": completion_record,
                "baseline_exposure_receipt": exposure_record,
            }
            gate_payload["gate_sha256"] = release.canonical_json_sha256(gate_payload)
            gate_path = contract_root / "final_gate.json"
            _write_json(gate_path, gate_payload)
            gate_record = release.file_record(gate_path)

            plan = {
                "output_dir": str(output.resolve()),
                "headline_release": {
                    "instance": instance,
                    "final_gate": gate_record,
                    "receipt_artifacts": artifacts,
                    "paper_ablation_completion_receipt": completion_record,
                    "baseline_exposure_receipt": exposure_record,
                },
            }
            with (
                mock.patch.object(release, "FINAL_GATE_PATH", gate_path),
                mock.patch.object(release, "FINAL_CONSUMPTION_ROOT", consumption_root),
                mock.patch.object(release, "SELECTION_RECEIPT_PATH", contract_root / "selection_receipt.json"),
                mock.patch.object(release, "BASELINE_CONTRACT_PATH", contract_root / "baseline_contract.json"),
                mock.patch.object(release, "CANDIDATE_CONTRACT_PATH", contract_root / "candidate_contract.json"),
                mock.patch.object(release, "PARITY_CONTRACT_PATH", contract_root / "evaluation_parity_contract.json"),
                mock.patch.object(release, "PAPER_ABLATION_COMPLETION_RECEIPT_PATH", completion_path),
                mock.patch.object(release, "BASELINE_EXPOSURE_RECEIPT_PATH", exposure_path),
                mock.patch.object(
                    release,
                    "validate_paper_ablation_completion_receipt",
                    return_value=completion,
                ),
                mock.patch.object(
                    release,
                    "validate_baseline_exposure_receipt",
                    return_value={"status": "verified"},
                ),
            ):
                receipt = release.consume_final_instance(plan)
                plan["headline_release"]["final_consumption"] = receipt
                release.validate_final_consumption(plan)
                with self.assertRaisesRegex(
                    release.HeadlineReleaseError, "already consumed"
                ):
                    release.consume_final_instance(plan)

    def test_parity_projection_ignores_model_specific_config_but_not_code_or_data(self):
        common = [
            {
                "path": "/repo/eval.py",
                "sha256": "a" * 64,
                "size_bytes": 10,
                "roles": ["evaluation_code_dependency"],
            },
            {
                "path": "/data/ref.json",
                "sha256": "b" * 64,
                "size_bytes": 20,
                "roles": ["evaluation_data_input"],
            },
        ]
        left = common + [
            {
                "path": "/candidate.py",
                "sha256": "c" * 64,
                "size_bytes": 30,
                "roles": ["evaluation_config", "config_dependency"],
            }
        ]
        right = common + [
            {
                "path": "/baseline.py",
                "sha256": "d" * 64,
                "size_bytes": 40,
                "roles": ["evaluation_config", "config_dependency"],
            }
        ]
        self.assertEqual(
            release.common_evaluation_fingerprints(left),
            release.common_evaluation_fingerprints(right),
        )
        right[0] = {**right[0], "sha256": "e" * 64}
        self.assertNotEqual(
            release.common_evaluation_fingerprints(left),
            release.common_evaluation_fingerprints(right),
        )

    def test_baseline_record_consistency_is_identity_aligned_and_tolerance_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy.jsonl"
            fresh = root / "fresh.jsonl"
            row = {
                "schema": "stageb-eval-record-v1",
                "task": "ref",
                "sample_id": "ref:1",
                "image_id": 1,
                "ann_id": 2,
                "ref_id": 3,
                "sent_id": 4,
                "valid": True,
                "correct50": True,
                "top1_iou": 0.75,
                "all_query_best_iou": 0.80,
            }
            legacy.write_text(json.dumps(row) + "\n", encoding="utf-8")
            fresh.write_text(
                json.dumps({**row, "top1_iou": 0.7500005}) + "\n",
                encoding="utf-8",
            )
            result = release._aligned_record_consistency(
                fresh, legacy, task="ref", label="fixture"
            )
            self.assertTrue(result["identity_order_exact"])
            self.assertLessEqual(
                result["max_abs_observed"]["top1_iou"], 1e-6
            )
            fresh.write_text(
                json.dumps({**row, "top1_iou": 0.750002}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                release.HeadlineReleaseError, "exceeds"
            ):
                release._aligned_record_consistency(
                    fresh, legacy, task="ref", label="fixture"
                )

    def test_builder_provenance_requires_exact_baseline_plus_three_m0_roots(self):
        artifacts = {
            name: {"path": f"/{name}.json", "sha256": "a" * 64, "size_bytes": 1}
            for name in (
                *release.RECEIPT_ARTIFACT_NAMES,
                "final_gate",
                "paper_ablation_completion_receipt",
                "baseline_exposure_receipt",
            )
        }
        parity = {"runtime": {"fixed": True}}
        evaluations = [
            {
                "instance": {"instance_id": release.BASELINE_ID},
                "artifacts": artifacts,
                "parity": parity,
                "baseline_run": {
                    "id": release.BASELINE_ID,
                    "seed": 42,
                    "checkpoint_sha256": release.FIXED_BASELINE[
                        "checkpoint_sha256"
                    ],
                    "role": "fixed_historical_checkpoint",
                    "evaluation_root": "/baseline",
                },
                "candidate_run": None,
            }
        ]
        for seed in release.CANDIDATE_SEEDS:
            evaluations.append(
                {
                    "instance": {"instance_id": f"M0:{seed}"},
                    "artifacts": artifacts,
                    "parity": parity,
                    "baseline_run": None,
                    "candidate_run": {
                        "training_run_id": f"M0:{seed}",
                        "seed": seed,
                        "queue_id": "queue",
                        "queue_plan_sha256": "b" * 64,
                        "checkpoint_sha256": f"{seed:064x}",
                        "evaluation_root": f"/M0/{seed}",
                    },
                }
            )
        provenance = release.build_release_provenance(
            evaluations,
            bootstrap={"iterations": 5000, "confidence": 0.95, "seed": 20260717},
        )
        self.assertEqual(provenance["status"], "passed")
        self.assertEqual(provenance["candidate"]["id"], "M0")
        self.assertEqual(provenance["candidate"]["seeds"], [17, 42, 73])
        with self.assertRaisesRegex(
            release.HeadlineReleaseError, "exactly four"
        ):
            release.build_release_provenance(
                evaluations[:-1],
                bootstrap={
                    "iterations": 5000,
                    "confidence": 0.95,
                    "seed": 20260717,
                },
            )
        with self.assertRaisesRegex(
            release.HeadlineReleaseError, "20260717"
        ):
            release.build_release_provenance(
                evaluations,
                bootstrap={
                    "iterations": 5000,
                    "confidence": 0.95,
                    "seed": 20260718,
                },
            )

    def test_exposure_receipt_delegates_to_canonical_builder(self):
        from tools import build_stageb_b58_exposure_receipt as exposure_builder

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exposure.json"
            _write_json(path, {"placeholder": True})
            expected = {"schema": release.BASELINE_EXPOSURE_RECEIPT_SCHEMA}
            with (
                mock.patch.object(release, "BASELINE_EXPOSURE_RECEIPT_PATH", path),
                mock.patch.object(
                    exposure_builder, "CANONICAL_RECEIPT_PATH", path
                ),
                mock.patch.object(
                    exposure_builder, "verify_receipt", return_value=expected
                ) as verify,
            ):
                self.assertEqual(
                    release.validate_baseline_exposure_receipt(path), expected
                )
            verify.assert_called_once_with(path.resolve())


if __name__ == "__main__":
    unittest.main()
