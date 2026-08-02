import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_stageb_dense_duty_formal as launcher


def _payload(
    spec: launcher.PhaseSpec,
    *,
    updates: int,
    reason: str,
    iteration: int = 4,
    epoch_finished: bool = False,
):
    args = launcher._required_saved_args(spec)
    args.update(
        {
            "resume": "",
            "pretrain_model_path": (
                str(launcher.STAGE_A_CHECKPOINT)
                if spec.phase == "rank"
                else str(
                    (spec.output.parent / "rank" / launcher.CHECKPOINT_NAME).resolve()
                )
            ),
            "start_epoch": 0,
        }
    )
    args.update(
        {
            "stage_b_dense_duty_lineage_audit": {
                "phase": spec.phase,
                "execution_scope": "formal",
                "no_stage_b_teacher": True,
                "dataset_config": {"sha256": spec.dataset_sha256},
            },
            "stage_b_dense_duty_runtime_audit": {
                "schema": launcher.RUNTIME_AUDIT_SCHEMA,
                "optimizer_step_boundaries": updates,
                "successful_optimizer_steps": updates,
                "zero_gradient_successful_steps": 0,
                "amp_skipped_optimizer_steps": 0,
                "nonfinite_gradient_boundaries": 0,
                "max_active_grad_norm_preclip": 1.0,
                "peak_reserved_bytes": 1024,
            },
        }
    )
    return {
        "model": {"weight": 1},
        "criterion": {"queue": 1},
        "optimizer": {"state": 1},
        "lr_scheduler": {"last_epoch": 0},
        "scaler": {"scale": 8.0},
        "epoch": 0,
        "iteration": iteration,
        "optimizer_updates": updates,
        "epoch_finished": epoch_finished,
        "rng_state": {"torch": 1},
        "epoch_rng_state": {"torch": 1},
        "args": args,
        "checkpoint_reason": reason,
    }


def _inspection(phase: str, status: launcher.PhaseStatus):
    return launcher.PhaseInspection(phase=phase, status=status)


class CheckpointClassificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "formal"
        self.rank, self.confidence = launcher.formal_phase_specs(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_classifies_partial_and_terminal_checkpoint_payloads(self):
        partial = launcher.classify_checkpoint_payload(
            _payload(self.rank, updates=100, reason="interval"), self.rank
        )
        terminal = launcher.classify_checkpoint_payload(
            _payload(
                self.rank,
                updates=launcher.RANK_UPDATES,
                reason="max_train_iters",
            ),
            self.rank,
        )
        self.assertEqual(partial.status, launcher.PhaseStatus.PARTIAL)
        self.assertEqual(terminal.status, launcher.PhaseStatus.TERMINAL)

    def test_rejects_missing_training_state_and_formal_argument_drift(self):
        missing = _payload(self.rank, updates=100, reason="interval")
        del missing["optimizer"]
        result = launcher.classify_checkpoint_payload(missing, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("required training state", result.detail)

        empty_state = _payload(self.rank, updates=100, reason="interval")
        empty_state["optimizer"] = {}
        result = launcher.classify_checkpoint_payload(empty_state, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("optimizer", result.detail)
        self.assertIn("empty", result.detail)

        drift = _payload(self.rank, updates=100, reason="interval")
        drift["args"]["num_workers"] = 8
        result = launcher.classify_checkpoint_payload(drift, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("formal arguments drifted", result.detail)

        cli_drift = _payload(self.rank, updates=100, reason="interval")
        cli_drift["args"]["find_unused_params"] = True
        result = launcher.classify_checkpoint_payload(cli_drift, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("formal arguments drifted", result.detail)

    def test_rejects_non_boundary_iteration_and_inconsistent_epoch_boundary(self):
        non_boundary = _payload(
            self.rank, updates=100, reason="interval", iteration=5
        )
        result = launcher.classify_checkpoint_payload(non_boundary, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("accumulation boundary", result.detail)

        inconsistent = _payload(
            self.rank,
            updates=100,
            reason="interval_epoch",
            iteration=4,
            epoch_finished=True,
        )
        result = launcher.classify_checkpoint_payload(inconsistent, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("epoch boundary", result.detail)

    def test_rejects_terminal_count_with_nonterminal_reason(self):
        payload = _payload(
            self.confidence,
            updates=launcher.CONFIDENCE_UPDATES,
            reason="signal",
        )
        result = launcher.classify_checkpoint_payload(payload, self.confidence)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("max_train_iters", result.detail)

    def test_accepts_signal_checkpoint_at_epoch_boundary(self):
        payload = _payload(
            self.rank,
            updates=100,
            reason="signal",
            iteration=0,
            epoch_finished=True,
        )
        result = launcher.classify_checkpoint_payload(payload, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.PARTIAL)

    def test_rejects_interval_checkpoint_off_fixed_cadence(self):
        payload = _payload(self.rank, updates=101, reason="interval")
        result = launcher.classify_checkpoint_payload(payload, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("cadence", result.detail)

    def test_rejects_skipped_or_nonfinite_gradient_runtime(self):
        for field in ("amp_skipped_optimizer_steps", "nonfinite_gradient_boundaries"):
            with self.subTest(field=field):
                payload = _payload(self.confidence, updates=100, reason="interval")
                payload["args"]["stage_b_dense_duty_runtime_audit"][field] = 1
                result = launcher.classify_checkpoint_payload(payload, self.confidence)
                self.assertEqual(result.status, launcher.PhaseStatus.INVALID)

        missing_counter = _payload(self.confidence, updates=100, reason="interval")
        del missing_counter["args"]["stage_b_dense_duty_runtime_audit"][
            "amp_skipped_optimizer_steps"
        ]
        result = launcher.classify_checkpoint_payload(missing_counter, self.confidence)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)

        terminal = _payload(
            self.confidence,
            updates=launcher.CONFIDENCE_UPDATES,
            reason="max_train_iters",
        )
        terminal["args"]["stage_b_dense_duty_runtime_audit"][
            "max_active_grad_norm_preclip"
        ] = float("nan")
        result = launcher.classify_checkpoint_payload(terminal, self.confidence)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)

        boolean_counter = _payload(self.rank, updates=1, reason="signal")
        boolean_counter["args"]["stage_b_dense_duty_runtime_audit"][
            "successful_optimizer_steps"
        ] = True
        result = launcher.classify_checkpoint_payload(boolean_counter, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)

    def test_rejects_checkpoint_when_current_source_closure_drifted(self):
        payload = _payload(self.rank, updates=100, reason="interval")
        with mock.patch.object(
            launcher,
            "build_source_closure",
            return_value={"schema": "simulated-current-source-drift"},
        ):
            result = launcher.classify_checkpoint_payload(payload, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("stage_b_dense_duty_source_closure", result.detail)

    def test_rejects_noncanonical_fresh_or_resume_transition(self):
        fresh = _payload(self.rank, updates=100, reason="interval")
        fresh["args"]["start_epoch"] = 1
        result = launcher.classify_checkpoint_payload(fresh, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("initializer/start-epoch", result.detail)

        resumed = _payload(self.rank, updates=100, reason="interval")
        resumed["args"].update(
            {
                "resume": str(self.rank.checkpoint.resolve()),
                "pretrain_model_path": None,
                "start_epoch": 0,
            }
        )
        result = launcher.classify_checkpoint_payload(resumed, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.PARTIAL)

        resumed["args"]["resume"] = str(
            self.rank.output / "checkpoint.pth"
        )
        result = launcher.classify_checkpoint_payload(resumed, self.rank)
        self.assertEqual(result.status, launcher.PhaseStatus.INVALID)
        self.assertIn("canonical fresh/resume", result.detail)

    def test_directory_is_fresh_only_when_absent_or_empty(self):
        absent = launcher.inspect_phase_directory(
            self.rank, checkpoint_auditor=None
        )
        self.assertEqual(absent.status, launcher.PhaseStatus.FRESH)

        self.rank.output.mkdir(parents=True)
        empty = launcher.inspect_phase_directory(self.rank, checkpoint_auditor=None)
        self.assertEqual(empty.status, launcher.PhaseStatus.FRESH)

        (self.rank.output / "checkpoint.pth").touch()
        ambiguous = launcher.inspect_phase_directory(
            self.rank, checkpoint_auditor=None
        )
        self.assertEqual(ambiguous.status, launcher.PhaseStatus.INVALID)
        self.assertIn("no atomic", ambiguous.detail)

    def test_directory_trusts_only_canonical_checkpoint_and_rejects_temp_bytes(self):
        self.rank.output.mkdir(parents=True)
        self.rank.checkpoint.touch()
        payload = _payload(self.rank, updates=100, reason="interval")
        valid = launcher.inspect_phase_directory(
            self.rank,
            checkpoint_loader=lambda _path: payload,
            checkpoint_auditor=None,
        )
        self.assertEqual(valid.status, launcher.PhaseStatus.PARTIAL)

        (self.rank.output / ".checkpoint_iter.pth.123.tmp").touch()
        invalid = launcher.inspect_phase_directory(
            self.rank,
            checkpoint_loader=lambda _path: payload,
            checkpoint_auditor=None,
        )
        self.assertEqual(invalid.status, launcher.PhaseStatus.INVALID)
        self.assertIn("unpublished checkpoint bytes", invalid.detail)

    def test_production_auditor_reuses_strict_partial_resume_validation(self):
        self.rank.output.mkdir(parents=True)
        self.rank.checkpoint.touch()
        payload = _payload(self.rank, updates=100, reason="interval")
        recorded_contract = {"schema": "synthetic-contract"}
        payload["args"][launcher.TRAINING_CONTRACT_ARG] = recorded_contract
        checkpoint_audit = {
            "schema": launcher.CHECKPOINT_AUDIT_SCHEMA,
            "status": "passed",
            "phase": "rank",
            "optimizer_updates": 100,
        }
        strict_audit = {"phase": "rank", "optimizer_updates": 100}
        with (
            mock.patch.object(
                launcher,
                "build_training_contract",
                return_value=recorded_contract,
            ),
            mock.patch.object(
                launcher,
                "validate_strict_resume_checkpoint_payload",
                return_value=strict_audit,
            ) as strict_validator,
            mock.patch(
                "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                return_value=checkpoint_audit,
            ),
            mock.patch.object(
                launcher,
                "build_source_closure",
                return_value=payload["args"][launcher.SOURCE_CLOSURE_ARG],
            ),
        ):
            launcher._audit_checkpoint(payload, self.rank, self.rank.checkpoint)
        strict_validator.assert_called_once_with(
            payload,
            payload["args"],
            checkpoint_path=self.rank.checkpoint,
        )

    def test_terminal_rank_requires_ogc_scorer_handoff_audit(self):
        self.rank.output.mkdir(parents=True)
        self.rank.checkpoint.touch()
        payload = _payload(
            self.rank,
            updates=launcher.RANK_UPDATES,
            reason="max_train_iters",
        )
        recorded_contract = {"schema": "synthetic-contract"}
        payload["args"][launcher.TRAINING_CONTRACT_ARG] = recorded_contract
        checkpoint_audit = {
            "schema": launcher.CHECKPOINT_AUDIT_SCHEMA,
            "status": "passed",
            "phase": "rank",
            "optimizer_updates": launcher.RANK_UPDATES,
        }
        with (
            mock.patch.object(
                launcher,
                "build_training_contract",
                return_value=recorded_contract,
            ),
            mock.patch(
                "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                return_value=checkpoint_audit,
            ),
            mock.patch.object(
                launcher,
                "build_source_closure",
                return_value=payload["args"][launcher.SOURCE_CLOSURE_ARG],
            ),
        ):
            with self.assertRaisesRegex(
                launcher.FormalLaunchError, "OGC scorer initialization"
            ):
                launcher._audit_checkpoint(
                    payload, self.rank, self.rank.checkpoint
                )


class StateMachineAndCommandTests(unittest.TestCase):
    def test_state_machine_accepts_only_the_linear_two_phase_path(self):
        F = launcher.PhaseStatus.FRESH
        P = launcher.PhaseStatus.PARTIAL
        T = launcher.PhaseStatus.TERMINAL
        expected = {
            (F, F): launcher.LaunchAction.START_RANK,
            (P, F): launcher.LaunchAction.RESUME_RANK,
            (T, F): launcher.LaunchAction.START_CONFIDENCE,
            (T, P): launcher.LaunchAction.RESUME_CONFIDENCE,
            (T, T): launcher.LaunchAction.COMPLETE,
        }
        for rank_status in (F, P, T):
            for confidence_status in (F, P, T):
                with self.subTest(rank=rank_status, confidence=confidence_status):
                    action, _detail = launcher.decide_action(
                        _inspection("rank", rank_status),
                        _inspection("confidence", confidence_status),
                    )
                    self.assertEqual(
                        action,
                        expected.get(
                            (rank_status, confidence_status),
                            launcher.LaunchAction.INVALID,
                        ),
                    )

    def test_commands_fix_runtime_and_use_exclusive_transition_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal"
            rank_spec, confidence_spec = launcher.formal_phase_specs(root)
            cases = {
                launcher.LaunchAction.START_RANK: (
                    "--pretrain_model_path",
                    str(launcher.STAGE_A_CHECKPOINT),
                ),
                launcher.LaunchAction.RESUME_RANK: (
                    "--resume",
                    str(rank_spec.checkpoint.resolve()),
                ),
                launcher.LaunchAction.START_CONFIDENCE: (
                    "--pretrain_model_path",
                    str(rank_spec.checkpoint.resolve()),
                ),
                launcher.LaunchAction.RESUME_CONFIDENCE: (
                    "--resume",
                    str(confidence_spec.checkpoint.resolve()),
                ),
            }
            for action, (transition_flag, transition_value) in cases.items():
                with self.subTest(action=action):
                    command = launcher.build_training_command(action, root)
                    self.assertNotIn("--options", command)
                    self.assertIn("--amp", command)
                    self.assertIn("--pin_memory", command)
                    self.assertIn("--no_persistent_workers", command)
                    self.assertEqual(command[command.index("--num_workers") + 1], "0")
                    self.assertEqual(
                        command[command.index("--prefetch_factor") + 1], "1"
                    )
                    self.assertEqual(
                        command[command.index("--gradient_accumulation_steps") + 1],
                        "4",
                    )
                    self.assertEqual(
                        command.count("--resume")
                        + command.count("--pretrain_model_path"),
                        1,
                    )
                    self.assertEqual(command[command.index(transition_flag) + 1], transition_value)
                    expected_updates = (
                        launcher.RANK_UPDATES
                        if "RANK" in action.name
                        else launcher.CONFIDENCE_UPDATES
                    )
                    self.assertEqual(
                        command[command.index("--max_train_iters") + 1],
                        str(expected_updates),
                    )

    def test_experiment_rejects_unexpected_root_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal"
            root.mkdir()
            (root / "README.txt").write_text("ambiguous", encoding="ascii")
            result = launcher.inspect_experiment(
                root, checkpoint_auditor=None
            )
        self.assertEqual(result.action, launcher.LaunchAction.INVALID)
        self.assertIn("unexpected entries", result.detail)

    def test_exit_zero_does_not_turn_partial_checkpoint_into_success(self):
        start = launcher.ExperimentInspection(
            _inspection("rank", launcher.PhaseStatus.FRESH),
            _inspection("confidence", launcher.PhaseStatus.FRESH),
            launcher.LaunchAction.START_RANK,
        )
        partial = launcher.ExperimentInspection(
            _inspection("rank", launcher.PhaseStatus.PARTIAL),
            _inspection("confidence", launcher.PhaseStatus.FRESH),
            launcher.LaunchAction.RESUME_RANK,
        )
        with (
            mock.patch.object(launcher, "inspect_experiment", side_effect=[start, partial]),
            mock.patch.object(launcher, "validate_launch_inputs"),
            mock.patch.object(
                launcher,
                "run_training_child",
                return_value=launcher.ChildResult(0, ()),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(launcher.FormalLaunchError, "terminal"):
                launcher.run_controller()

    def test_forwarded_signal_stops_after_post_exit_reclassification(self):
        start = launcher.ExperimentInspection(
            _inspection("rank", launcher.PhaseStatus.FRESH),
            _inspection("confidence", launcher.PhaseStatus.FRESH),
            launcher.LaunchAction.START_RANK,
        )
        partial = launcher.ExperimentInspection(
            _inspection("rank", launcher.PhaseStatus.PARTIAL),
            _inspection("confidence", launcher.PhaseStatus.FRESH),
            launcher.LaunchAction.RESUME_RANK,
        )
        with (
            mock.patch.object(launcher, "inspect_experiment", side_effect=[start, partial]),
            mock.patch.object(launcher, "validate_launch_inputs"),
            mock.patch.object(
                launcher,
                "run_training_child",
                return_value=launcher.ChildResult(0, (int(launcher.signal.SIGTERM),)),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(launcher.run_controller(), 128 + int(launcher.signal.SIGTERM))

    def test_launch_input_drift_fails_before_subprocess(self):
        start = launcher.ExperimentInspection(
            _inspection("rank", launcher.PhaseStatus.FRESH),
            _inspection("confidence", launcher.PhaseStatus.FRESH),
            launcher.LaunchAction.START_RANK,
        )
        with (
            mock.patch.object(launcher, "inspect_experiment", return_value=start),
            mock.patch.object(
                launcher,
                "build_source_closure",
                return_value=launcher.build_source_closure(
                    launcher.RANK_CONFIG, repo_root=launcher.REPO_ROOT
                ),
            ),
            mock.patch.object(
                launcher,
                "_stable_regular_file_sha256",
                return_value="0" * 64,
            ),
            mock.patch.object(launcher, "run_training_child") as child,
        ):
            with self.assertRaisesRegex(launcher.FormalLaunchError, "SHA256 drifted"):
                launcher.run_controller()
        child.assert_not_called()


class LockAndCliTests(unittest.TestCase):
    def test_training_child_inherits_controller_lock_fd(self):
        process = mock.Mock()
        process.wait.return_value = 0
        with mock.patch.object(
            launcher.subprocess, "Popen", return_value=process
        ) as popen:
            result = launcher.run_training_child(("command",), lock_fd=17)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(popen.call_args.kwargs["pass_fds"], (17,))
        self.assertTrue(popen.call_args.kwargs["close_fds"])

    def test_subprocess_environment_forces_single_process(self):
        distributed = {
            "RANK": "3",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "3",
            "MASTER_ADDR": "host",
            "MASTER_PORT": "1234",
            "SLURM_PROCID": "3",
            "SLURM_LOCALID": "3",
            "SLURM_NTASKS": "8",
            "SLURM_NODELIST": "node",
        }
        with mock.patch.dict(launcher.os.environ, distributed, clear=False):
            environment = launcher._subprocess_environment()
        for key in distributed:
            self.assertNotIn(key, environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")

    def test_controller_lock_is_nonblocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "controller.lock"
            with launcher.exclusive_controller_lock(lock):
                with self.assertRaises(launcher.FormalLaunchError):
                    with launcher.exclusive_controller_lock(lock):
                        pass

    def test_cli_explicitly_rejects_options(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = launcher.main(["--options", "batch_size=1"])
        self.assertEqual(result, 2)
        self.assertIn("forbids --options", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
