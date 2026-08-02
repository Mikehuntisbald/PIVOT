import hashlib
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from tools import run_stageb_token_ablation_matrix as launcher
from tools import seal_stageb_memory_probe as memory_seal


class StageBTokenAblationConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.configs = {
            row.row_id: runpy.run_path(str(cls.root / row.config))
            for row in launcher.ROWS
        }

    def test_all_rows_share_the_v20_architecture_and_runtime_contract(self):
        expected = {
            "stage_b_v20_acc50_aligned_hard_negatives": True,
            "stage_b_v11_positive_iou_threshold": 0.5,
            "stage_b_v11_negative_iou_threshold": 0.499,
            "stage_b_v11_candidate_topk": 50,
            "stage_b_v15_decoupled_confidence": True,
            "stage_b_v16_confidence_output_mode": "base_plus_gate",
            "stage_b_v19_explicit_confidence_output_contract": True,
            "data_aug_hflip_prob": 0.0,
            "stage_b_v21_token_weight": 1.0,
            "stage_b_v21_token_focal_alpha": 0.25,
            "stage_b_v21_token_focal_gamma": 2.0,
            "stage_b_v21_allow_legacy_token_diff_fallback": False,
            "skip_eval": True,
        }
        for row_id, config in self.configs.items():
            for key, value in expected.items():
                with self.subTest(row=row_id, key=key):
                    self.assertEqual(config[key], value)

    def test_rows_differ_only_in_predeclared_loss_knobs(self):
        allowlist = {
            "stage_b_v21_token_objective",
            "stage_b_v11_predicate_tn_rank_weight",
            "stage_b_v21_token_positive_weight",
            "stage_b_v21_token_shared_weight",
            "stage_b_v21_token_edit_weight",
        }
        reference = {
            key: value
            for key, value in self.configs["L4"].items()
            if not key.startswith("__") and key not in allowlist
        }
        for row_id, config in self.configs.items():
            comparable = {
                key: value
                for key, value in config.items()
                if not key.startswith("__") and key not in allowlist
            }
            self.assertEqual(comparable, reference, row_id)

    def test_row_objectives_match_the_registered_matrix(self):
        for row in launcher.ROWS:
            config = self.configs[row.row_id]
            with self.subTest(row=row.row_id):
                self.assertEqual(
                    config["stage_b_v21_token_objective"], row.token_objective
                )
                self.assertEqual(
                    config["stage_b_v11_predicate_tn_rank_weight"],
                    row.predicate_pair_rank_weight,
                )
                self.assertEqual(
                    config["stage_b_v21_token_positive_weight"],
                    row.positive_weight,
                )
                self.assertEqual(
                    config["stage_b_v21_token_shared_weight"],
                    row.shared_weight,
                )
                self.assertEqual(
                    config["stage_b_v21_token_edit_weight"], row.edit_weight
                )

    def test_single_edit_dataset_has_exactly_half_expected_tn_exposure(self):
        path = self.root / "config/datasets_stageb_v21_single_edit_train.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        train = payload["train"]
        weights = [float(source["mix_weight"]) for source in train]
        self.assertEqual(weights, [1.0, 1.0, 1.0, 3.0])
        self.assertEqual(weights[-1] / sum(weights), 0.5)
        self.assertEqual(train[-1]["source"], "sam3_tn_pair")
        self.assertFalse(train[-1]["require_global_tn_verified"])
        self.assertTrue(train[-1]["require_single_edit_token_provenance"])
        self.assertEqual(train[-1]["paper_table_b_id"], "D3")
        self.assertEqual(
            train[-1]["paper_tn_scope"], "proposal_covered_verified"
        )
        self.assertTrue(
            train[-1]["anno"].endswith("/d3_proposal_covered_train.jsonl")
        )
        self.assertEqual(payload["val"], [])


class StageBTokenAblationLauncherTest(unittest.TestCase):
    def test_script_context_can_import_shared_paper_runtime(self):
        script = launcher.REPO_ROOT / "tools/run_stageb_token_ablation_matrix.py"
        expression = (
            "import runpy; "
            f"ns = runpy.run_path({str(script)!r}); "
            "ns['_paper_runtime_evidence']()"
        )
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = launcher.subprocess.run(
            [sys.executable, "-c", expression],
            cwd="/tmp",
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def _fake_dataset(self, root: Path) -> Path:
        canonical = root / "canonical.json"
        support = root / "support.tsv"
        audit = root / "table_b_audit.json"
        canonical.write_text("{}\n", encoding="utf-8")
        support.write_text("path\tclass\n", encoding="utf-8")
        audit.write_text("{}\n", encoding="utf-8")
        train = []
        for index in range(4):
            annotation = root / f"train_{index}.jsonl"
            annotation.write_text(
                json.dumps({"id": index}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            source = {
                "dataset_mode": "patch_episode",
                "anno": str(annotation),
                "canonical_classes_json": str(canonical),
                "support_patch_tsv": str(support),
                "mix_weight": 3.0 if index == 3 else 1.0,
            }
            if index == 3:
                source.update(
                    {
                        "source": "sam3_tn_pair",
                        "require_global_tn_verified": False,
                        "require_single_edit_token_provenance": True,
                        "paper_table_b_id": "D3",
                        "paper_tn_scope": "proposal_covered_verified",
                        "paper_contract_audit": str(audit),
                    }
                )
            train.append(source)
        path = root / "datasets.json"
        path.write_text(
            json.dumps({"train": train, "val": []}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def test_list_exposes_all_thirty_three_registered_runs_without_runtime_io(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = launcher.main(["list", "--json"])
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["seeds"], [17, 42, 73])
        self.assertEqual(len(payload["rows"]), 11)
        self.assertEqual(len(payload["run_ids"]), 33)
        self.assertIn("L4:17", payload["run_ids"])
        self.assertIn("L10:73", payload["run_ids"])

    def test_dry_run_records_command_hashes_and_does_not_create_train_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_a = root / "stage_a.pth"
            scorer = root / "scorer.pth"
            stage_a.write_bytes(b"stage-a-test")
            scorer.write_bytes(b"scorer-test")
            dataset = self._fake_dataset(root)
            output_root = root / "paper_outputs"
            manifest_path = root / "planned" / "L3_seed17.json"
            environment = {
                "PIVOT_PYTHON": sys.executable,
                "PIVOT_STAGE_A_INIT": str(stage_a),
                "PIVOT_SCORER_WARMSTART": str(scorer),
                "PIVOT_TOKEN_DATASETS": str(dataset),
                "PIVOT_TOKEN_OUTPUT_ROOT": str(output_root),
                "PIVOT_DATA_ROOT": str(root),
                "PIVOT_BATCH_SIZE": "2",
                "PIVOT_MAX_TRAIN_ITERS": "3",
                "PIVOT_ITER_CHECKPOINT_INTERVAL": "1",
                "PIVOT_NUM_WORKERS": "0",
                "PIVOT_OMP_NUM_THREADS": "1",
                "PIVOT_CUDA_VISIBLE_DEVICES": "7",
            }
            stdout = StringIO()
            with patch.dict(os.environ, environment, clear=False):
                with redirect_stdout(stdout):
                    result = launcher.main(
                        [
                            "dry-run",
                            "--run-id",
                            "L3:17",
                            "--manifest",
                            str(manifest_path),
                        ]
                    )
            self.assertEqual(result, 0)
            self.assertFalse(output_root.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(manifest["run_id"], "L3:17")
            self.assertEqual(manifest["runtime"]["batch_size"], 2)
            self.assertEqual(manifest["runtime"]["max_train_iters"], 3)
            self.assertEqual(manifest["runtime"]["cuda_visible_devices"], "7")
            self.assertEqual(
                manifest["fixed_contract"]["dataset"][
                    "expected_tn_exposure_fraction"
                ],
                0.5,
            )
            command = manifest["command"]
            self.assertIn(str(launcher.REPO_ROOT / "main.py"), command)
            self.assertIn("--amp", command)
            self.assertIn("--save_log", command)
            self.assertIn("--max_train_iters", command)
            self.assertIn("--iter_checkpoint_interval", command)
            self.assertIn("batch_size=2", command)
            self.assertIn(
                f"stage_b_v15_scorer_init_checkpoint={scorer}", command
            )
            self.assertEqual(
                manifest["inputs"]["stage_a_initializer"]["sha256"],
                hashlib.sha256(stage_a.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["inputs"]["scorer_warmstart"]["sha256"],
                hashlib.sha256(scorer.read_bytes()).hexdigest(),
            )
            dependency_names = {
                Path(record["path"]).name
                for record in manifest["inputs"]["config_dependencies"]
            }
            self.assertIn("cfg_stageb_v21_token_l3_edit_bce.py", dependency_names)
            self.assertIn(
                "cfg_stageb_v19_full_text_base_plus_gate.py",
                dependency_names,
            )
            self.assertIn("[L3:17]", stdout.getvalue())

    def test_l1_l9_memory_readiness_plans_cover_ladder_and_soak_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_a = root / "stage_a.pth"
            scorer = root / "scorer.pth"
            stage_a.write_bytes(b"stage-a-test")
            scorer.write_bytes(b"scorer-test")
            dataset = self._fake_dataset(root)
            cache = launcher.HashCache()
            profiles = (
                ("lower-short", 32, 2),
                ("upper-short", 40, 2),
                ("selected-soak", 40, 50),
            )

            for row_id in ("L1", "L9"):
                row = launcher.ROW_BY_ID[row_id]
                for profile, batch_size, updates in profiles:
                    with self.subTest(
                        row=row_id,
                        profile=profile,
                        batch_size=batch_size,
                        updates=updates,
                    ):
                        output_root = (root / profile / row_id).resolve()
                        runtime = launcher.Runtime(
                            python=Path(sys.executable).resolve(),
                            stage_a_init=stage_a.resolve(),
                            scorer_warmstart=scorer.resolve(),
                            dataset=dataset.resolve(),
                            output_root=output_root,
                            data_root=root.resolve(),
                            batch_size=batch_size,
                            max_train_iters=updates,
                            iter_checkpoint_interval=updates,
                            num_workers=0,
                            prefetch_factor=1,
                            omp_num_threads=1,
                            min_nofile=0,
                            cuda_visible_devices="0",
                            mp_sharing_strategy="file_system",
                        )
                        manifest = launcher.build_manifest(
                            runtime, row, 17, cache
                        )
                        sequence = launcher._build_sequence_manifest(manifest)

                        self.assertEqual(
                            Path(manifest["output_dir"]),
                            output_root / row_id / "seed17",
                        )
                        self.assertFalse(Path(manifest["output_dir"]).exists())
                        self.assertEqual(
                            manifest["runtime"]["max_train_iters"], updates
                        )
                        self.assertEqual(
                            manifest["runtime"]["iter_checkpoint_interval"],
                            updates,
                        )
                        self.assertEqual(
                            sequence["equal_budget_contract"],
                            {
                                "batch_size": batch_size,
                                "optimizer_updates": updates,
                                "contributing_phase_updates": {"joint": updates},
                            },
                        )
                        self.assertEqual(
                            sequence["row"]["token_objective"],
                            row.token_objective,
                        )
                        evidence = manifest["runtime_evidence_contract"]
                        self.assertTrue(evidence["all_inputs_rehashed_after_training"])
                        self.assertTrue(evidence["sequence_manifest_required"])
                        self.assertTrue(evidence["zero_amp_skipped_steps_required"])

    def test_postflight_safely_validates_checkpoint_and_native_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_a = root / "stage_a.pth"
            scorer = root / "scorer.pth"
            stage_a.write_bytes(b"stage-a-test")
            scorer.write_bytes(b"scorer-test")
            dataset = self._fake_dataset(root)
            output_dir = root / "outputs" / "L3" / "seed17"
            output_dir.mkdir(parents=True)
            config = (
                launcher.REPO_ROOT
                / "config/ablations/cfg_stageb_v21_token_l3_edit_bce.py"
            ).resolve()
            runtime = launcher.Runtime(
                python=Path(sys.executable).resolve(),
                stage_a_init=stage_a.resolve(),
                scorer_warmstart=scorer.resolve(),
                dataset=dataset.resolve(),
                output_root=(root / "outputs").resolve(),
                data_root=root.resolve(),
                batch_size=2,
                max_train_iters=3,
                iter_checkpoint_interval=1,
                num_workers=0,
                prefetch_factor=1,
                omp_num_threads=1,
                min_nofile=0,
                cuda_visible_devices="",
                mp_sharing_strategy="file_system",
            )
            scorer_sha = hashlib.sha256(scorer.read_bytes()).hexdigest()
            scorer_audit = {
                "schema": "stage_b_v15_scorer_init/v1",
                "status": "applied",
                "source_sha256": scorer_sha,
                "resolved_source_path": str(scorer.resolve()),
                "loaded_num_layers": 3,
            }
            (output_dir / "stage_b_v15_scorer_init_audit.json").write_text(
                json.dumps(scorer_audit) + "\n", encoding="utf-8"
            )
            (output_dir / "info.txt").write_text(
                (
                    "loss: 1.25  amp_step_skipped: 0.0 (0.0)  "
                    "amp_scale: 512.0 (512.0)  max mem: 1024\n"
                ),
                encoding="utf-8",
            )
            (output_dir / "train_console.log").write_text(
                "streamed console output\n", encoding="utf-8"
            )
            (output_dir / "launch_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            gpu_environment = {
                "schema": "pivot.gpu_environment/v1",
                "torch_runtime": {"cuda_available": True},
                "nvidia_devices": [
                    {
                        "uuid": "GPU-test",
                        "name": "Test GPU",
                        "driver_version": "999.0",
                        "total_memory_mib": 32000.0,
                    }
                ],
            }
            gpu_summary = {
                "schema": "pivot.gpu_telemetry_summary/v1",
                "sample_rows": 1,
                "devices": [
                    {
                        "uuid": "GPU-test",
                        "name": "Test GPU",
                        "driver_version": "999.0",
                        "total_memory_mib": 32000.0,
                        "peak_used_memory_mib": 12000.0,
                        "min_free_memory_mib": 20000.0,
                    }
                ],
            }
            (output_dir / "gpu_environment.json").write_text(
                json.dumps(gpu_environment) + "\n", encoding="utf-8"
            )
            (output_dir / "gpu_telemetry.csv").write_text(
                (
                    "timestamp,index,uuid,name,driver_version,total_memory_mib,"
                    "used_memory_mib,free_memory_mib,utilization_percent\n"
                    "now,0,GPU-test,Test GPU,999.0,32000,12000,20000,100\n"
                ),
                encoding="utf-8",
            )
            (output_dir / "gpu_telemetry_summary.json").write_text(
                json.dumps(gpu_summary) + "\n", encoding="utf-8"
            )
            checkpoint_args = {
                "seed": 17,
                "batch_size": 2,
                "max_train_iters": 3,
                "iter_checkpoint_interval": 1,
                "config_file": str(config),
                "datasets": str(dataset.resolve()),
                "output_dir": str(output_dir.resolve()),
                "pretrain_model_path": str(stage_a.resolve()),
                "stage_b_v15_scorer_init_checkpoint": str(scorer.resolve()),
                "stage_b_v15_scorer_init_audit": scorer_audit,
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
                "stage_b_v11_predicate_tn_rank_weight": 0.0,
                "stage_b_v21_allow_legacy_token_diff_fallback": False,
                "stage_b_v19_allow_scope_labeled_tn_ablation": True,
                "stage_b_v19_table_b_id": "D3",
                "stage_b_v19_table_b_scope_allowlist": [
                    "proposal_covered_verified"
                ],
                "stage_b_v19_table_b_audit": (
                    "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
                ),
                "stage_b_v19_table_b_audit_sha256": (
                    "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
                ),
                "stage_b_v19_table_b_allow_single_edit_token_provenance": True,
                "skip_eval": True,
                "amp": True,
            }
            torch.save(
                {
                    "model": {"weight": torch.ones(1)},
                    "criterion": {},
                    "optimizer": {},
                    "lr_scheduler": {},
                    "scaler": {},
                    "args": checkpoint_args,
                    "epoch": 0,
                    "iteration": 3,
                    "epoch_finished": False,
                    "checkpoint_reason": "max_train_iters",
                },
                output_dir / "checkpoint_iter.pth",
            )
            cache = launcher.HashCache()
            manifest = {
                "run_id": "L3:17",
                "output_dir": str(output_dir.resolve()),
                "inputs": {
                    "stage_a_initializer": launcher._file_record(stage_a, cache),
                    "scorer_warmstart": launcher._file_record(scorer, cache),
                    "dataset_manifest": launcher._file_record(dataset, cache),
                    "config_dependencies": [],
                    "dataset_source_files": [],
                    "repository_sources": [],
                },
            }
            postflight = launcher._perform_postflight(
                manifest,
                runtime=runtime,
                row=launcher.ROW_BY_ID["L3"],
                seed=17,
                cache=cache,
            )
            self.assertEqual(postflight["status"], "passed")
            self.assertEqual(
                postflight["schema"], "pivot.stageb.token_ablation_postflight/v2"
            )
            self.assertEqual(postflight["input_rehash"]["status"], "passed")
            self.assertEqual(
                postflight["gpu_telemetry_summary"]["sample_rows"], 1
            )
            self.assertEqual(
                postflight["numerical_status"]["max_amp_step_skipped"], 0.0
            )
            self.assertEqual(
                postflight["checkpoint_metadata"]["checkpoint_reason"],
                "max_train_iters",
            )
            self.assertFalse(postflight["native_epoch_log"]["present"])
            self.assertIn(
                "max_train_iters",
                postflight["native_epoch_log"]["expected_absence_reason"],
            )

    def test_post_run_input_rehash_detects_content_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "input.json"
            path.write_text('{"value": 1}\n', encoding="utf-8")
            record = launcher._file_record(path, launcher.HashCache())
            manifest = {
                "inputs": {
                    "stage_a_initializer": record,
                    "scorer_warmstart": record,
                    "dataset_manifest": record,
                    "config_dependencies": [],
                    "dataset_source_files": [],
                    "repository_sources": [],
                }
            }
            passed = launcher._rehash_inputs(manifest)
            self.assertEqual(passed["status"], "passed")
            self.assertEqual(passed["unique_input_count"], 1)
            self.assertEqual(
                passed["records"][0]["roles"],
                ["dataset_manifest", "scorer_warmstart", "stage_a_initializer"],
            )
            path.write_text('{"value": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "post-run input SHA-256 drift"
            ):
                launcher._rehash_inputs(manifest)

    def test_run_body_writes_sealable_completed_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_a = root / "stage_a.pth"
            scorer = root / "scorer.pth"
            stage_a.write_bytes(b"stage-a-test")
            scorer.write_bytes(b"scorer-test")
            dataset = self._fake_dataset(root)
            runtime = launcher.Runtime(
                python=Path(sys.executable).resolve(),
                stage_a_init=stage_a.resolve(),
                scorer_warmstart=scorer.resolve(),
                dataset=dataset.resolve(),
                output_root=(root / "outputs").resolve(),
                data_root=root.resolve(),
                batch_size=8,
                max_train_iters=50,
                iter_checkpoint_interval=50,
                num_workers=0,
                prefetch_factor=1,
                omp_num_threads=1,
                min_nofile=0,
                cuda_visible_devices="0",
                mp_sharing_strategy="file_system",
            )
            row = launcher.ROW_BY_ID["L1"]
            args = SimpleNamespace(
                mode="run", run_id=[(row, 17)], all=False
            )

            def fake_stream(command, *, runtime, console_log):
                del command, runtime
                console_log.write_text("training output\n", encoding="utf-8")
                output = console_log.parent
                (output / "checkpoint_iter.pth").write_bytes(b"checkpoint")
                return 0

            telemetry_summary = {
                "schema": "pivot.gpu_telemetry_summary/v1",
                "sample_rows": 1,
                "devices": [
                    {
                        "uuid": "GPU-test",
                        "name": "Test GPU",
                        "driver_version": "999.0",
                        "total_memory_mib": 32000.0,
                        "peak_used_memory_mib": 24000.0,
                        "min_free_memory_mib": 8000.0,
                        "sample_count": 1,
                    }
                ],
            }

            def stop_sampler():
                output = launcher.output_directory(runtime, row, 17)
                (output / "gpu_telemetry_summary.json").write_text(
                    json.dumps(telemetry_summary) + "\n", encoding="utf-8"
                )
                return telemetry_summary

            sampler = SimpleNamespace(stop=Mock(side_effect=stop_sampler))
            postflight = {
                "status": "passed",
                "numerical_status": {
                    "status": "passed",
                    "loss_values_all_finite": True,
                    "max_amp_step_skipped": 0.0,
                },
            }
            with patch.object(
                launcher, "runtime_from_environment", return_value=runtime
            ):
                with patch.object(
                    launcher,
                    "_capture_gpu_environment",
                    return_value={"schema": "test.gpu_environment/v1"},
                ):
                    with patch.object(
                        launcher, "_start_gpu_telemetry", return_value=sampler
                    ):
                        with patch.object(
                            launcher, "_stream_subprocess", side_effect=fake_stream
                        ):
                            with patch.object(
                                launcher,
                                "_perform_postflight",
                                return_value=postflight,
                            ):
                                result = launcher._run_body(
                                    args, orchestration_status=None
                                )
            self.assertEqual(result, 0)
            output = launcher.output_directory(runtime, row, 17)
            sequence = json.loads(
                (output / "sequence_manifest.json").read_text(encoding="utf-8")
            )
            launch = json.loads(
                (output / "launch_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sequence["status"], "completed")
            self.assertEqual(
                sequence["equal_budget_contract"],
                {
                    "batch_size": 8,
                    "optimizer_updates": 50,
                    "contributing_phase_updates": {"joint": 50},
                },
            )
            self.assertEqual(sequence["completed_phases"][0]["status"], "completed")
            self.assertEqual(launch["status"], "completed")
            self.assertEqual(sampler.stop.call_count, 1)
            sealed_probe = memory_seal.inspect_probe(output)
            self.assertEqual(sealed_probe["batch_size"], 8)
            self.assertEqual(sealed_probe["optimizer_updates"], 50)

    def test_detach_preflights_and_persists_control_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_a = root / "stage_a.pth"
            scorer = root / "scorer.pth"
            stage_a.write_bytes(b"stage-a-test")
            scorer.write_bytes(b"scorer-test")
            runtime = launcher.Runtime(
                python=Path(sys.executable).resolve(),
                stage_a_init=stage_a.resolve(),
                scorer_warmstart=scorer.resolve(),
                dataset=self._fake_dataset(root).resolve(),
                output_root=(root / "outputs").resolve(),
                data_root=root.resolve(),
                batch_size=8,
                max_train_iters=50,
                iter_checkpoint_interval=50,
                num_workers=0,
                prefetch_factor=1,
                omp_num_threads=1,
                min_nofile=0,
                cuda_visible_devices="0",
                mp_sharing_strategy="file_system",
            )
            row = launcher.ROW_BY_ID["L9"]
            args = SimpleNamespace(
                mode="detach",
                run_id=[(row, 17)],
                all=False,
                orchestration_root=root / "jobs",
            )
            stdout = StringIO()
            with patch.object(
                launcher, "runtime_from_environment", return_value=runtime
            ):
                with patch.object(
                    launcher.subprocess,
                    "Popen",
                    return_value=SimpleNamespace(pid=4242),
                ) as popen:
                    with patch.object(
                        launcher,
                        "_read_process_identity",
                        return_value={
                            "available": True,
                            "pid": 4242,
                            "start_time_ticks": 100,
                            "boot_id": "test-boot",
                        },
                    ):
                        with redirect_stdout(stdout):
                            result = launcher._detach(args)
            self.assertEqual(result, 0)
            response = json.loads(stdout.getvalue())
            job_dir = Path(response["job_dir"])
            launch = json.loads((job_dir / "launch.json").read_text())
            status = json.loads((job_dir / "status.json").read_text())
            self.assertEqual(launch["status"], "launched")
            self.assertEqual(status["status"], "prepared")
            self.assertTrue((job_dir / "plans/L9/seed17.json").is_file())
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(
                popen.call_args.kwargs["stdin"], launcher.subprocess.DEVNULL
            )

    def test_status_and_reconcile_delegate_to_hardened_shared_observer(self):
        for mode, mutate in (("status", False), ("reconcile", True)):
            with self.subTest(mode=mode):
                stdout = StringIO()
                report = {
                    "persisted_status": "running",
                    "observed_status": "running",
                    "mutated": mutate,
                }
                with patch.object(
                    launcher,
                    "_inspect_or_reconcile_detached_job",
                    return_value=report,
                ) as inspect:
                    with redirect_stdout(stdout):
                        result = launcher.main([mode, "/tmp/token-job"])
                self.assertEqual(result, 0)
                self.assertEqual(json.loads(stdout.getvalue()), report)
                inspect.assert_called_once_with(
                    Path("/tmp/token-job"), mutate=mutate
                )


if __name__ == "__main__":
    unittest.main()
