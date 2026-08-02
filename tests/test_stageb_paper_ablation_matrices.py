import hashlib
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools import run_stageb_paper_ablation_matrices as launcher


class StageBPaperMatrixTest(unittest.TestCase):
    def _runtime(self, root: Path, *, total: int = 8) -> launcher.Runtime:
        stage_a = root / "stage_a.pth"
        scorer = root / "scorer.pth"
        stage_a.write_bytes(b"stage-a-test")
        scorer.write_bytes(b"scorer-test")
        return launcher.Runtime(
            python=Path(sys.executable).resolve(),
            stage_a_init=stage_a.resolve(),
            scorer_warmstart=scorer.resolve(),
            tn_output_root=(root / "table_b").resolve(),
            score_output_root=(root / "table_d").resolve(),
            data_root=root.resolve(),
            batch_size=2,
            total_train_iters=total,
            iter_checkpoint_interval=total,
            num_workers=0,
            prefetch_factor=1,
            omp_num_threads=1,
            min_nofile=0,
            cuda_visible_devices="",
            mp_sharing_strategy="none",
            gradient_diagnostic_interval=3,
        )

    def _dataset(
        self,
        root: Path,
        *,
        row_id: str,
        scope: str | None = None,
        table_d: bool = False,
    ) -> Path:
        canonical = root / f"{row_id}_canonical.json"
        support = root / f"{row_id}_support.tsv"
        canonical.write_text("{}\n", encoding="utf-8")
        support.write_text("path\tclass\n", encoding="utf-8")
        train = []
        for index in range(3):
            annotation = root / f"{row_id}_positive_{index}.jsonl"
            annotation.write_text("{}\n", encoding="utf-8")
            train.append(
                {
                    "dataset_mode": "patch_episode",
                    "anno": str(annotation),
                    "canonical_classes_json": str(canonical),
                    "support_patch_tsv": str(support),
                    "mix_weight": 1.0,
                }
            )
        if row_id != "D0":
            annotation = root / f"{row_id}_tn.jsonl"
            annotation.write_text("{}\n", encoding="utf-8")
            source = {
                "dataset_mode": "patch_episode",
                "source": "sam3_tn_pair",
                "anno": str(annotation),
                "canonical_classes_json": str(canonical),
                "support_patch_tsv": str(support),
                "mix_weight": 3.0,
            }
            if table_d:
                source.update(
                    {
                        "require_global_tn_verified": False,
                        "require_single_edit_token_provenance": True,
                        "paper_table_b_id": "D3",
                        "paper_tn_scope": "proposal_covered_verified",
                        "paper_contract_audit": str(launcher.TABLE_B_AUDIT),
                    }
                )
            else:
                source.update(
                    {
                        "require_global_tn_verified": False,
                        "require_single_edit_token_provenance": False,
                        "paper_table_b_id": row_id,
                        "paper_tn_scope": scope,
                        "paper_contract_audit": str(launcher.TABLE_B_AUDIT),
                    }
                )
            train.append(source)
        path = root / f"{row_id}_datasets.json"
        path.write_text(
            json.dumps({"train": train, "val": []}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _detached_job(
        self,
        root: Path,
        *,
        status: str = "running",
        pid: int = 4242,
        sequence_status: str | None = None,
        phase_status: str | None = None,
        orchestrator_log: str = "Start training\n",
        telemetry: str = "timestamp, uuid, memory.used [MiB]\nnow, GPU-0, 1024 MiB\n",
    ) -> tuple[Path, Path]:
        job_dir = root / "job"
        run_root = root / "run"
        job_dir.mkdir(parents=True)
        run_root.mkdir(parents=True)
        log_path = job_dir / "orchestrator.log"
        log_path.write_text(orchestrator_log, encoding="utf-8")
        (run_root / "gpu_telemetry.csv").write_text(
            telemetry, encoding="utf-8"
        )
        (job_dir / "launch.json").write_text(
            json.dumps(
                {
                    "schema": "pivot.stageb.paper_ablation_detached_launch/v1",
                    "status": "launched",
                    "child_pid": pid,
                    "expected_run_roots": [str(run_root)],
                    "orchestrator_log": str(log_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (job_dir / "status.json").write_text(
            json.dumps(
                {
                    "schema": (
                        "pivot.stageb.paper_ablation_orchestration_status/v1"
                    ),
                    "status": status,
                    "pid": pid,
                    "expected_run_roots": [str(run_root)],
                    "current_run_id": "D3m:17",
                    "current_phase_id": "joint",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if sequence_status is not None:
            (run_root / "sequence_manifest.json").write_text(
                json.dumps({"status": sequence_status}) + "\n",
                encoding="utf-8",
            )
        if phase_status is not None:
            (run_root / "launch_manifest.json").write_text(
                json.dumps(
                    {
                        "status": phase_status,
                        "failure_phase": "training_process",
                        "returncode": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return job_dir, run_root

    def test_list_registers_three_seeds_for_all_eleven_rows_without_io(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = launcher.main(["list", "--json"])
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["seeds"], [17, 42, 73])
        self.assertEqual(len(payload["rows"]), 11)
        self.assertEqual(len(payload["run_ids"]), 33)
        self.assertIn("D3:73", payload["run_ids"])
        self.assertIn("S3:17", payload["run_ids"])
        self.assertIn("S2F:42", payload["run_ids"])
        self.assertIn("D3m:73", payload["run_ids"])
        self.assertEqual(launcher._parse_run_id("d2M:17")[0].row_id, "D2m")

    def test_v23_table_b_configs_fix_l4_and_bind_exact_weak_scopes(self):
        root = launcher.REPO_ROOT / "config/ablations"
        expected = {
            "D0": ("cfg_stageb_v23_table_b_d0_no_tn.py", False, []),
            "D1": (
                "cfg_stageb_v23_table_b_d1_unverified_allneg.py",
                True,
                ["unverified_all_negative"],
            ),
            "D2": (
                "cfg_stageb_v23_table_b_d2_traceable_edits.py",
                True,
                ["traceable_counterfactual_edit"],
            ),
            "D3": (
                "cfg_stageb_v23_table_b_d3_proposal_covered.py",
                True,
                ["proposal_covered_verified"],
            ),
        }
        for row_id, (filename, enabled, scopes) in expected.items():
            with self.subTest(row=row_id):
                config = runpy.run_path(str(root / filename))
                self.assertEqual(config["stage_b_v23_table_id"], row_id)
                self.assertEqual(
                    config["stage_b_v19_allow_scope_labeled_tn_ablation"],
                    enabled,
                )
                self.assertEqual(
                    config["stage_b_v19_table_b_scope_allowlist"], scopes
                )
                self.assertEqual(
                    config["stage_b_v19_table_b_audit_sha256"],
                    launcher.TABLE_B_AUDIT_SHA256,
                )
                self.assertEqual(config["stage_b_v21_token_objective"], "edit_bce")
                self.assertEqual(config["stage_b_v11_predicate_tn_rank_weight"], 1.0)
                self.assertTrue(config["stage_b_v20_acc50_aligned_hard_negatives"])
                self.assertEqual(
                    config["stage_b_v16_confidence_output_mode"], "base_plus_gate"
                )

    def test_table_b_manifest_hashes_scope_audit_and_disables_tn_token_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            dataset = self._dataset(
                root,
                row_id="D1",
                scope="unverified_all_negative",
            )
            row = launcher.MatrixRow(
                "D1",
                "B",
                "config/ablations/cfg_stageb_v23_table_b_d1_unverified_allneg.py",
                str(dataset),
                tn_scope="unverified_all_negative",
            )
            manifest = launcher.build_manifest(
                runtime, row, 17, launcher.token_launcher.HashCache()
            )
            self.assertFalse(Path(manifest["output_dir"]).exists())
            self.assertEqual(
                manifest["equal_budget_contract"]["optimizer_updates"], 8
            )
            phase = manifest["phases"][0]
            dataset_contract = phase["fixed_contract"]["dataset"]
            self.assertEqual(dataset_contract["expected_tn_draw_fraction"], 0.5)
            self.assertTrue(
                dataset_contract["token_fairness"][
                    "tn_edit_token_supervision_disabled_for_all_table_b_rows"
                ]
            )
            self.assertFalse(
                dataset_contract["tn_source"]["require_global_tn_verified"]
            )
            self.assertFalse(
                dataset_contract["tn_source"][
                    "require_single_edit_token_provenance"
                ]
            )
            audit_records = [
                record
                for record in phase["inputs"]["records"]
                if Path(record["path"]) == launcher.TABLE_B_AUDIT
            ]
            self.assertTrue(audit_records)
            self.assertTrue(
                all(
                    record["sha256"] == launcher.TABLE_B_AUDIT_SHA256
                    for record in audit_records
                )
            )
            dependency_names = {
                Path(record["path"]).name
                for record in phase["inputs"]["records"]
                if record["role"] == "config_dependency"
            }
            self.assertIn("cfg_stageb_v19_full_text_base_plus_gate.py", dependency_names)
            self.assertIn("cfg_stageb_v21_token_l4_edit_bce_pair_rank.py", dependency_names)

    def test_table_d_preflight_rejects_loss_confound(self):
        values = {"S0": 0.0, "S1": 0.0, "S2": 1.0, "S3": 1.0}

        def fake_run_path(path):
            filename = Path(path).name
            row_id = next(row for row in values if f"_{row.lower()}_" in filename)
            return {"stage_b_v15_tail_queue_positive_trust_weight": values[row_id]}

        with patch.object(launcher.runpy, "run_path", side_effect=fake_run_path):
            with self.assertRaisesRegex(RuntimeError, "ownership comparison is confounded"):
                launcher._validate_table_d_comparison_block()

    def test_actual_table_d_configs_have_common_clean_objective(self):
        self.assertEqual(
            launcher._validate_table_d_comparison_block(),
            {"S0": 0.0, "S1": 0.0, "S2": 0.0, "S3": 0.0},
        )
        full = runpy.run_path(
            str(
                launcher.REPO_ROOT
                / "config/ablations/cfg_stageb_v22_s2_independent_joint_full.py"
            )
        )
        self.assertEqual(full["stage_b_v22_table_id"], "S2F")
        self.assertEqual(full["stage_b_v15_tail_queue_positive_trust_weight"], 1.0)
        launcher._validate_table_d_row_objective(launcher.ROW_BY_ID["S2F"], full)

    def test_table_d_preflight_rejects_wrong_clean_or_full_objective_weight(self):
        values = {"S0": 1.0, "S1": 1.0, "S2": 1.0, "S3": 1.0}

        def fake_run_path(path):
            filename = Path(path).name
            row_id = next(row for row in values if f"_{row.lower()}_" in filename)
            return {"stage_b_v15_tail_queue_positive_trust_weight": values[row_id]}

        with patch.object(launcher.runpy, "run_path", side_effect=fake_run_path):
            with self.assertRaisesRegex(RuntimeError, "clean objective requires"):
                launcher._validate_table_d_comparison_block()
        with self.assertRaisesRegex(RuntimeError, "S2F objective contract requires"):
            launcher._validate_table_d_row_objective(
                launcher.ROW_BY_ID["S2F"],
                {"stage_b_v15_tail_queue_positive_trust_weight": 0.0},
            )

    def test_s2_is_table_d_sustained_worst_memory_probe_row(self):
        def load(name):
            return runpy.run_path(
                str(launcher.REPO_ROOT / "config/ablations" / name)
            )

        s0 = load("cfg_stageb_v22_s0_shared_score.py")
        s1 = load("cfg_stageb_v22_s1_shared_trunk_two_heads.py")
        s2 = load("cfg_stageb_v22_s2_independent_joint.py")
        s2f = load("cfg_stageb_v22_s2_independent_joint_full.py")
        s3_rank = load("cfg_stageb_v22_s3_rank_phase.py")
        s3_confidence = load("cfg_stageb_v22_s3_confidence_phase.py")

        self.assertFalse(s0["stage_b_v15_decoupled_confidence"])
        self.assertFalse(s1["stage_b_v15_decoupled_confidence"])
        self.assertTrue(s2["stage_b_v15_decoupled_confidence"])
        self.assertEqual(
            s2["only_train_keywords"],
            [
                "stage_b_fixed_text_scorer.decoder",
                "stage_b_fixed_text_scorer.validity_head",
            ],
        )
        self.assertEqual(s2f["only_train_keywords"], s2["only_train_keywords"])
        self.assertEqual(
            s3_rank["only_train_keywords"],
            ["stage_b_fixed_text_scorer.decoder"],
        )
        self.assertEqual(
            s3_confidence["only_train_keywords"],
            ["stage_b_fixed_text_scorer.validity_head"],
        )
        self.assertEqual(s2["stage_b_v22_train_phase"], "joint")
        self.assertEqual(s2f["stage_b_v22_train_phase"], "joint")
        self.assertEqual(
            s2["stage_b_v22_gradient_diagnostic_kind"], "branch_isolation"
        )

    def test_s2_memory_readiness_manifests_cover_ladder_and_soak_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self._runtime(root, total=2)
            dataset = self._dataset(root, row_id="S2", table_d=True)
            row = launcher.MatrixRow(
                "S2",
                "D",
                "config/ablations/cfg_stageb_v22_s2_independent_joint.py",
                str(dataset),
                score_ownership="independent_decoders_joint",
                objective_fidelity="common_objective_ownership_ablation",
            )
            profiles = (
                ("b32-ladder", 32, 2, 1),
                ("b40-ladder", 40, 2, 1),
                ("b40-soak", 40, 50, 10),
            )
            cache = launcher.token_launcher.HashCache()
            for name, batch_size, updates, diagnostic_interval in profiles:
                with self.subTest(profile=name):
                    runtime = replace(
                        base,
                        score_output_root=(root / name).resolve(),
                        batch_size=batch_size,
                        total_train_iters=updates,
                        iter_checkpoint_interval=updates,
                        num_workers=2,
                        gradient_diagnostic_interval=diagnostic_interval,
                    )
                    manifest = launcher.build_manifest(runtime, row, 17, cache)
                    phase = manifest["phases"][0]
                    self.assertEqual(manifest["row"]["row_id"], "S2")
                    self.assertEqual(
                        manifest["row"]["score_ownership"],
                        "independent_decoders_joint",
                    )
                    self.assertEqual(
                        manifest["equal_budget_contract"],
                        {
                            "batch_size": batch_size,
                            "optimizer_updates": updates,
                            "s3_probe_updates_excluded": 0,
                            "contributing_phase_updates": {"joint": updates},
                        },
                    )
                    self.assertEqual(phase["phase"]["phase_id"], "joint")
                    self.assertEqual(
                        phase["fixed_contract"]["gradient_diagnostic_interval"],
                        diagnostic_interval,
                    )
                    self.assertIn(
                        f"stage_b_v22_gradient_diagnostic_interval={diagnostic_interval}",
                        phase["command"],
                    )
                    self.assertIn(f"batch_size={runtime.batch_size}", phase["command"])
                    self.assertFalse(Path(manifest["output_dir"]).exists())

    def test_s2_memory_ladder_profile_is_cli_dry_runnable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root, row_id="S2", table_d=True)
            row = launcher.MatrixRow(
                "S2",
                "D",
                "config/ablations/cfg_stageb_v22_s2_independent_joint.py",
                str(dataset),
                score_ownership="independent_decoders_joint",
                objective_fidelity="common_objective_ownership_ablation",
            )
            runtime = replace(
                self._runtime(root, total=2),
                score_output_root=(root / "memory_probes/b32").resolve(),
                batch_size=32,
                iter_checkpoint_interval=2,
                num_workers=2,
                gradient_diagnostic_interval=1,
            )
            manifest_path = root / "plans/table_d_s2_b32_u2.json"
            args = SimpleNamespace(
                mode="dry-run",
                run_id=[(row, 17)],
                all=False,
                table="all",
                manifest=manifest_path,
                manifest_dir=None,
            )
            stdout = StringIO()
            with patch.object(
                launcher, "runtime_from_environment", return_value=runtime
            ):
                with redirect_stdout(stdout):
                    result = launcher._dry_run(args)
            self.assertEqual(result, 0)
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "planned")
            self.assertTrue(manifest["output_dir_fresh_at_plan"])
            self.assertEqual(
                manifest["equal_budget_contract"],
                {
                    "batch_size": 32,
                    "optimizer_updates": 2,
                    "s3_probe_updates_excluded": 0,
                    "contributing_phase_updates": {"joint": 2},
                },
            )
            phase = manifest["phases"][0]
            self.assertEqual(phase["runtime"]["num_workers"], 2)
            self.assertEqual(
                phase["fixed_contract"]["gradient_diagnostic_interval"], 1
            )
            self.assertIn("--max_train_iters 2", phase["command_shell"])
            self.assertIn("batch_size=32", phase["command_shell"])
            self.assertFalse(Path(manifest["output_dir"]).exists())
            self.assertIn("[S2:17/joint]", stdout.getvalue())

    def test_s3_plan_splits_budget_and_never_resumes_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root, total=8)
            dataset = self._dataset(root, row_id="S3", table_d=True)
            row = launcher.MatrixRow(
                "S3",
                "D",
                "config/ablations/cfg_stageb_v22_s3_rank_phase.py",
                str(dataset),
                score_ownership="independent_decoders_two_phase",
                objective_fidelity=(
                    "common_objective_ownership_ablation_split_schedule"
                ),
            )
            with patch.object(
                launcher,
                "_validate_table_d_comparison_block",
                return_value={"S0": 0.0, "S1": 0.0, "S2": 0.0, "S3": 0.0},
            ):
                manifest = launcher.build_manifest(
                    runtime, row, 42, launcher.token_launcher.HashCache()
                )
            phases = {value["phase"]["phase_id"]: value for value in manifest["phases"]}
            self.assertEqual(phases["isolation_probe"]["phase"]["updates"], 1)
            self.assertFalse(
                phases["isolation_probe"]["phase"]["contributes_to_budget"]
            )
            self.assertEqual(phases["rank"]["phase"]["updates"], 4)
            self.assertEqual(phases["confidence"]["phase"]["updates"], 4)
            self.assertEqual(
                manifest["equal_budget_contract"]["optimizer_updates"], 8
            )
            self.assertEqual(
                manifest["equal_budget_contract"]["s3_probe_updates_excluded"], 1
            )
            confidence_command = phases["confidence"]["command"]
            self.assertNotIn("--resume", confidence_command)
            self.assertIn("--pretrain_model_path", confidence_command)
            self.assertTrue(
                any("rank/checkpoint_iter.pth" in part for part in confidence_command)
            )
            self.assertFalse(
                any(
                    part.startswith("stage_b_v15_scorer_init_checkpoint=")
                    for part in confidence_command
                )
            )
            self.assertEqual(
                phases["confidence"]["generated_dependency"]["status"],
                "deferred_until_rank_postflight",
            )
            self.assertEqual(
                phases["isolation_probe"]["fixed_contract"][
                    "gradient_diagnostic_interval"
                ],
                1,
            )

    def test_s3_materialized_rank_dependency_is_hashed_for_confidence_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root, total=8)
            dataset = self._dataset(root, row_id="S3", table_d=True)
            row = launcher.MatrixRow(
                "S3",
                "D",
                "config/ablations/cfg_stageb_v22_s3_rank_phase.py",
                str(dataset),
                score_ownership="independent_decoders_two_phase",
                objective_fidelity=(
                    "common_objective_ownership_ablation_split_schedule"
                ),
            )
            rank_checkpoint = root / "rank" / "checkpoint_iter.pth"
            rank_checkpoint.parent.mkdir()
            rank_checkpoint.write_bytes(b"immutable-rank-model-state")
            phase = launcher._phases(runtime, row)[-1]
            with patch.object(
                launcher,
                "_validate_table_d_comparison_block",
                return_value={"S0": 0.0, "S1": 0.0, "S2": 0.0, "S3": 0.0},
            ):
                manifest = launcher._phase_manifest(
                    runtime,
                    row,
                    17,
                    phase,
                    root / "confidence",
                    launcher.token_launcher.HashCache(),
                    rank_checkpoint=rank_checkpoint,
                )
            dependency = manifest["generated_dependency"]
            expected_sha = hashlib.sha256(rank_checkpoint.read_bytes()).hexdigest()
            self.assertEqual(dependency["status"], "materialized_and_hashed")
            self.assertEqual(dependency["sha256"], expected_sha)
            self.assertTrue(dependency["optimizer_resume_forbidden"])
            rank_records = [
                record
                for record in manifest["inputs"]["records"]
                if record["role"] == "rank_phase_model_state_pretrain"
            ]
            self.assertEqual(len(rank_records), 1)
            self.assertEqual(rank_records[0]["sha256"], expected_sha)

    def test_runtime_rejects_odd_total_update_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            environment = {
                "PIVOT_PYTHON": sys.executable,
                "PIVOT_STAGE_A_INIT": str(runtime.stage_a_init),
                "PIVOT_SCORER_WARMSTART": str(runtime.scorer_warmstart),
                "PIVOT_MAX_TRAIN_ITERS": "7",
            }
            with patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(ValueError, "must be even"):
                    launcher.runtime_from_environment()

    def test_safe_checkpoint_inspector_uses_weights_only_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            checkpoint = root / "checkpoint_iter.pth"
            torch.save(
                {
                    "model": {"weight": torch.ones(1)},
                    "criterion": {},
                    "optimizer": {},
                    "lr_scheduler": {},
                    "scaler": {},
                    "epoch": 0,
                    "iteration": 8,
                    "epoch_finished": False,
                    "checkpoint_reason": "max_train_iters",
                    "args": {"seed": 17},
                },
                checkpoint,
            )
            metadata = launcher._inspect_checkpoint_safely(runtime, checkpoint)
            self.assertTrue(metadata["has_complete_training_state"])
            self.assertEqual(metadata["iteration"], 8)
            self.assertEqual(metadata["checkpoint_reason"], "max_train_iters")

    def test_gpu_and_amp_log_summaries_are_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry = root / "gpu_telemetry.csv"
            telemetry.write_text(
                launcher._GpuTelemetrySampler.HEADER
                + "2026/07/17 12:00:00.000, 0, GPU-test, RTX Test, 999.0, 32000, 1000, 31000, 5\n"
                + "2026/07/17 12:00:01.000, 0, GPU-test, RTX Test, 999.0, 32000, 29000, 3000, 99\n",
                encoding="utf-8",
            )
            summary = launcher._summarize_nvidia_csv(telemetry)
            self.assertEqual(summary["sample_rows"], 2)
            self.assertEqual(summary["devices"][0]["peak_used_memory_mib"], 29000)
            self.assertEqual(summary["devices"][0]["min_free_memory_mib"], 3000)

            info = root / "info.txt"
            console = root / "console.txt"
            info.write_text("training started\n", encoding="utf-8")
            console.write_text(
                "loss: 1.2500 amp_step_skipped: 0.0000 amp_scale: 65536 max mem: 28750\n",
                encoding="utf-8",
            )
            numerical = launcher._training_numerical_status(info, console)
            self.assertEqual(numerical["status"], "passed")
            self.assertEqual(numerical["max_amp_step_skipped"], 0.0)
            self.assertEqual(
                numerical["torch_cuda_max_memory_allocated_mib_from_log"], 28750
            )
            self.assertFalse(
                numerical["torch_cuda_max_memory_reserved"]["available"]
            )

            console.write_text(
                "loss: 1.2500 amp_step_skipped: 0.0000 (0.1000) "
                "amp_scale: 65536 (65536) max mem: 28750\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "numerical/AMP audit failed"):
                launcher._training_numerical_status(info, console)

            environment = {
                "schema": "pivot.gpu_environment/v1",
                "torch_runtime": {"cuda_available": True},
                "nvidia_devices": [
                    {
                        "uuid": "GPU-test",
                        "name": "RTX Test",
                        "driver_version": "999.0",
                        "total_memory_mib": 32000.0,
                    }
                ],
            }
            launcher._validate_gpu_telemetry_contract(environment, summary)
            summary["devices"][0]["uuid"] = "GPU-other"
            with self.assertRaisesRegex(RuntimeError, "UUID set differs"):
                launcher._validate_gpu_telemetry_contract(environment, summary)

    def test_post_run_input_rehash_detects_content_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text('{"value": 1}\n', encoding="utf-8")
            cache = launcher.token_launcher.HashCache()
            record = launcher._file_record(path, cache, role="dataset_source")
            manifest = {"inputs": {"records": [record]}}
            passed = launcher._rehash_inputs(manifest)
            self.assertEqual(passed["status"], "passed")
            path.write_text('{"value": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "post-run input SHA-256 drift"):
                launcher._rehash_inputs(manifest)

    def test_fresh_output_check_happens_before_manifest_or_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            row = launcher.ROW_BY_ID["D0"]
            output = launcher.output_directory(runtime, row, 17)
            output.mkdir(parents=True)
            args = SimpleNamespace(
                mode="run",
                run_id=[(row, 17)],
                all=False,
                table="all",
            )
            with patch.object(launcher, "runtime_from_environment", return_value=runtime):
                with patch.object(launcher, "build_manifest") as build:
                    with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                        launcher._run(args)
                    build.assert_not_called()

    def test_detach_preflights_then_spawns_new_session_with_persistent_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            row = launcher.ROW_BY_ID["D0"]
            args = SimpleNamespace(
                mode="detach",
                run_id=[(row, 17)],
                all=False,
                table="all",
                orchestration_root=root / "jobs",
            )
            planned = {
                "schema": "test.plan/v1",
                "run_id": "D0:17",
                "output_dir": str(launcher.output_directory(runtime, row, 17)),
            }
            stdout = StringIO()
            with patch.object(launcher, "runtime_from_environment", return_value=runtime):
                with patch.object(launcher, "build_manifest", return_value=planned):
                    with patch.object(
                        launcher.subprocess,
                        "Popen",
                        return_value=SimpleNamespace(pid=4242),
                    ) as popen:
                        with redirect_stdout(stdout):
                            result = launcher._detach(args)
            self.assertEqual(result, 0)
            response = json.loads(stdout.getvalue())
            self.assertEqual(response["pid"], 4242)
            job_dir = Path(response["job_dir"])
            launch = json.loads((job_dir / "launch.json").read_text())
            status = json.loads((job_dir / "status.json").read_text())
            self.assertEqual(launch["status"], "launched")
            self.assertTrue(launch["child_start_new_session"])
            self.assertEqual(launch["stdout_stderr"], str(job_dir / "orchestrator.log"))
            self.assertEqual(status["status"], "prepared")
            self.assertTrue((job_dir / "plans/D0/seed17.json").is_file())
            kwargs = popen.call_args.kwargs
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stdin"], launcher.subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], launcher.subprocess.STDOUT)

    def test_status_subcommand_is_read_only_for_live_orchestrator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir, _ = self._detached_job(root)
            status_path = job_dir / "status.json"
            before = status_path.read_bytes()
            alive = {
                "checked_at_utc": "2026-07-17T00:00:00+00:00",
                "pid": 4242,
                "probe": "test",
                "state": "alive",
                "running": True,
            }
            stdout = StringIO()
            with patch.object(
                launcher, "_probe_pid_liveness", return_value=alive
            ):
                with redirect_stdout(stdout):
                    result = launcher.main(["status", str(job_dir)])
            self.assertEqual(result, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["persisted_status"], "running")
            self.assertEqual(report["observed_status"], "running")
            self.assertEqual(report["reason"], "orchestrator_process_is_alive")
            self.assertFalse(report["reconciliation_required"])
            self.assertFalse(report["mutated"])
            self.assertEqual(status_path.read_bytes(), before)
            self.assertEqual(
                report["evidence_tails"]["gpu_telemetry"][0]["tail_lines"][-1],
                "now, GPU-0, 1024 MiB",
            )

    def test_reconcile_dead_job_without_terminal_artifact_is_unknown_not_oom(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir, _ = self._detached_job(
                root,
                orchestrator_log=(
                    "Start training\nCUDA out of memory appeared in text only\n"
                ),
            )
            dead = {
                "checked_at_utc": "2026-07-17T00:00:00+00:00",
                "pid": 4242,
                "probe": "test",
                "state": "not_found",
                "running": False,
            }
            stdout = StringIO()
            with patch.object(
                launcher, "_probe_pid_liveness", return_value=dead
            ):
                with redirect_stdout(stdout):
                    result = launcher.main(["reconcile", str(job_dir)])
            self.assertEqual(result, 0)
            report = json.loads(stdout.getvalue())
            persisted = json.loads((job_dir / "status.json").read_text())
            self.assertTrue(report["mutated"])
            self.assertEqual(report["observed_status"], "hard_terminated_unknown")
            self.assertEqual(persisted["status"], "hard_terminated_unknown")
            self.assertEqual(persisted["reconciled_from_status"], "running")
            reconciliation = persisted["reconciliation"]
            self.assertEqual(reconciliation["termination_cause"], "unknown")
            self.assertEqual(reconciliation["oom_classification"], "not_established")
            self.assertIn(
                "CUDA out of memory",
                "\n".join(
                    reconciliation["evidence_tails"]["orchestrator_log"][
                        "tail_lines"
                    ]
                ),
            )
            self.assertIn(
                "cannot establish OOM", reconciliation["cause_inference_policy"]
            )
            after_first_reconcile = (job_dir / "status.json").read_bytes()
            with patch.object(
                launcher, "_probe_pid_liveness", return_value=dead
            ):
                second = launcher._inspect_or_reconcile_detached_job(
                    job_dir, mutate=True
                )
            self.assertFalse(second["mutated"])
            self.assertEqual(
                (job_dir / "status.json").read_bytes(), after_first_reconcile
            )

    def test_reconcile_uses_only_explicit_artifacts_for_completed_or_failed(self):
        dead = {
            "checked_at_utc": "2026-07-17T00:00:00+00:00",
            "pid": 4242,
            "probe": "test",
            "state": "not_found",
            "running": False,
        }
        cases = (
            ("completed", None, "completed"),
            (None, "failed", "failed"),
        )
        for sequence_status, phase_status, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    job_dir, _ = self._detached_job(
                        Path(temporary),
                        sequence_status=sequence_status,
                        phase_status=phase_status,
                    )
                    with patch.object(
                        launcher, "_probe_pid_liveness", return_value=dead
                    ):
                        report = launcher._inspect_or_reconcile_detached_job(
                            job_dir, mutate=True
                        )
                    persisted = json.loads((job_dir / "status.json").read_text())
                    self.assertEqual(report["observed_status"], expected)
                    self.assertEqual(persisted["status"], expected)
                    self.assertIsNone(report["termination_cause"])
                    self.assertEqual(
                        report["artifact_evidence"]["classification"], expected
                    )

    def test_pid_identity_mismatch_is_reported_as_reuse_not_alive(self):
        expected = {
            "available": True,
            "pid": 4242,
            "state": "R",
            "start_time_ticks": 100,
            "boot_id": "boot-a",
            "command": "old command",
        }
        observed = {
            "available": True,
            "pid": 4242,
            "state": "S",
            "start_time_ticks": 200,
            "boot_id": "boot-a",
            "command": "new command",
        }
        with patch.object(launcher.os, "kill"):
            with patch.object(
                launcher, "_read_process_identity", return_value=observed
            ):
                liveness = launcher._probe_pid_liveness(
                    4242, expected_identity=expected
                )
        self.assertEqual(liveness["state"], "pid_reused")
        self.assertFalse(liveness["running"])

    def test_reconcile_retains_unrecognized_status_conservatively(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_dir, _ = self._detached_job(
                Path(temporary), status="future_terminal_state"
            )
            dead = {
                "checked_at_utc": "2026-07-17T00:00:00+00:00",
                "pid": 4242,
                "probe": "test",
                "state": "not_found",
                "running": False,
            }
            before = (job_dir / "status.json").read_bytes()
            with patch.object(
                launcher, "_probe_pid_liveness", return_value=dead
            ):
                report = launcher._inspect_or_reconcile_detached_job(
                    job_dir, mutate=True
                )
            self.assertFalse(report["mutated"])
            self.assertFalse(report["reconciliation_required"])
            self.assertEqual(report["observed_status"], "future_terminal_state")
            self.assertEqual((job_dir / "status.json").read_bytes(), before)

    def test_s3_confidence_checkpoint_forbids_reapplied_scorer_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            dataset = self._dataset(root, row_id="S3", table_d=True)
            row = launcher.MatrixRow(
                "S3",
                "D",
                "config/ablations/cfg_stageb_v22_s3_rank_phase.py",
                str(dataset),
                score_ownership="independent_decoders_two_phase",
                objective_fidelity=(
                    "common_objective_ownership_ablation_split_schedule"
                ),
            )
            phase = launcher._phases(runtime, row)[-1]
            output = root / "confidence"
            rank_checkpoint = root / "rank" / "checkpoint_iter.pth"
            rank_checkpoint.parent.mkdir()
            rank_checkpoint.write_bytes(b"rank")
            args = {
                "seed": 17,
                "batch_size": 2,
                "max_train_iters": 4,
                "iter_checkpoint_interval": 4,
                "config_file": str(launcher.REPO_ROOT / phase.config),
                "datasets": str(dataset),
                "output_dir": str(output),
                "pretrain_model_path": str(rank_checkpoint),
                "stage_b_v15_scorer_init_checkpoint": str(runtime.scorer_warmstart),
                "stage_b_v15_scorer_init_audit": {"unexpected": True},
                "stage_b_v20_acc50_aligned_hard_negatives": True,
                "stage_b_v11_candidate_topk": 50,
                "stage_b_v11_positive_iou_threshold": 0.5,
                "stage_b_v11_negative_iou_threshold": 0.499,
                "stage_b_v21_token_objective": "edit_bce",
                "stage_b_v21_token_weight": 1.0,
                "stage_b_v21_token_positive_weight": 1.0,
                "stage_b_v21_token_shared_weight": 0.25,
                "stage_b_v21_token_edit_weight": 1.0,
                "stage_b_v21_token_focal_alpha": 0.25,
                "stage_b_v21_token_focal_gamma": 2.0,
                "stage_b_v11_predicate_tn_rank_weight": 1.0,
                "stage_b_v21_allow_legacy_token_diff_fallback": False,
                "stage_b_v15_tail_queue_positive_trust_weight": 0.0,
                "stage_b_v19_allow_scope_labeled_tn_ablation": True,
                "stage_b_v19_table_b_id": "D3",
                "stage_b_v19_table_b_scope_allowlist": [
                    "proposal_covered_verified"
                ],
                "stage_b_v19_table_b_audit": (
                    "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
                ),
                "stage_b_v19_table_b_audit_sha256": (
                    launcher.TABLE_B_AUDIT_SHA256
                ),
                "stage_b_v19_table_b_allow_single_edit_token_provenance": True,
                "stage_b_v22_gradient_diagnostic_interval": 0,
                "stage_b_v22_table_id": "S3-confidence",
                "stage_b_v22_score_ownership": "independent_decoders_two_phase",
                "stage_b_v22_train_phase": "confidence",
                "stage_b_v22_objective_fidelity": (
                    "common_objective_ownership_ablation_split_schedule"
                ),
                "stage_b_v22_phase_index": 2,
                "stage_b_v22_phase_count": 2,
                "stage_b_v22_requires_rank_phase_checkpoint": True,
                "stage_b_v22_probe_only": None,
                "only_train_keywords": [
                    "stage_b_fixed_text_scorer.validity_head"
                ],
                "only_train_exclude_keywords": [],
                "skip_eval": True,
                "amp": True,
            }
            metadata = {
                "has_complete_training_state": True,
                "iteration": 4,
                "checkpoint_reason": "max_train_iters",
                "epoch_finished": False,
                "args": args,
            }
            with self.assertRaisesRegex(RuntimeError, "reapplied scorer warm-start"):
                launcher._validate_checkpoint_metadata(
                    metadata,
                    runtime=runtime,
                    row=row,
                    seed=17,
                    phase=phase,
                    output_dir=output,
                    pretrain_path=rank_checkpoint,
                    scorer_audit=None,
                )
            args["stage_b_v15_scorer_init_checkpoint"] = None
            args["stage_b_v15_scorer_init_audit"] = None
            args["resume"] = str(root / "forbidden_resume.pth")
            with self.assertRaisesRegex(RuntimeError, "resumed optimizer state"):
                launcher._validate_checkpoint_metadata(
                    metadata,
                    runtime=runtime,
                    row=row,
                    seed=17,
                    phase=phase,
                    output_dir=output,
                    pretrain_path=rank_checkpoint,
                    scorer_audit=None,
                )
            args["resume"] = None
            args["only_train_keywords"] = [
                "stage_b_fixed_text_scorer.decoder"
            ]
            with self.assertRaisesRegex(
                RuntimeError, "checkpoint args mismatch for only_train_keywords"
            ):
                launcher._validate_checkpoint_metadata(
                    metadata,
                    runtime=runtime,
                    row=row,
                    seed=17,
                    phase=phase,
                    output_dir=output,
                    pretrain_path=rank_checkpoint,
                    scorer_audit=None,
                )


if __name__ == "__main__":
    unittest.main()
