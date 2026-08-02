import hashlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tools import run_stageb_paper_evaluations as evaluator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _artifact(path: Path, *, role: str | None = None) -> dict:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if role is not None:
        result["role"] = role
    return result


class StageBPaperEvaluationTest(unittest.TestCase):
    def test_artifact_root_survives_an_external_outputs_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            storage = root / "storage"
            (repository / "config").mkdir(parents=True)
            (repository / "data").mkdir()
            storage.mkdir()
            os.symlink(storage, repository / "outputs", target_is_directory=True)
            with patch.object(evaluator, "REPO_ROOT", repository):
                self.assertEqual(
                    evaluator._artifact_repository_root(), repository.resolve()
                )

    def test_subprocess_environment_loads_code_before_artifact_configs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution_root = root / "execution"
            artifact_root = root / "artifact"
            existing_root = root / "existing"
            for path in (execution_root, artifact_root, existing_root):
                path.mkdir()
            runtime = self._runtime(root)
            with (
                patch.object(evaluator, "REPO_ROOT", execution_root),
                patch.object(
                    evaluator, "ARTIFACT_REPOSITORY_ROOT", artifact_root
                ),
                patch.dict(
                    os.environ,
                    {"PYTHONPATH": str(existing_root)},
                    clear=False,
                ),
            ):
                environment = evaluator._subprocess_environment(runtime)

            self.assertEqual(
                environment["PYTHONPATH"].split(os.pathsep),
                [str(execution_root), str(artifact_root), str(existing_root)],
            )

    def _training_run(
        self,
        root: Path,
        *,
        row_id: str = "D3m",
        seed: int = 17,
        status: str = "completed",
    ) -> tuple[Path, Path, Path]:
        run_root = root / f"{row_id}_seed{seed}"
        run_root.mkdir()
        config = root / f"{row_id}_config.py"
        config.write_text("stage_b = True\n", encoding="utf-8")
        phase_ids = (
            ("isolation_probe", "rank", "confidence")
            if row_id == "S3"
            else ("joint",)
        )
        completed = []
        planned = []
        final_checkpoint = None
        for phase_id in phase_ids:
            output = run_root if row_id != "S3" else run_root / phase_id
            output.mkdir(exist_ok=True)
            checkpoint = output / "checkpoint_iter.pth"
            checkpoint.write_bytes(f"{row_id}:{seed}:{phase_id}".encode("ascii"))
            checkpoint_record = _artifact(checkpoint)
            postflight = {
                "schema": "pivot.stageb.paper_ablation_phase_postflight/v1",
                "status": "passed",
                "checkpoint_metadata": {
                    "args": {
                        "config_file": str(config.resolve()),
                        "output_dir": str(output.resolve()),
                    }
                },
                "input_rehash": {"status": "passed"},
                "artifacts": {"checkpoint": checkpoint_record},
            }
            postflight_path = output / "postflight.json"
            postflight_path.write_text(
                json.dumps(postflight, indent=2) + "\n", encoding="utf-8"
            )
            phase_manifest = {
                "schema": evaluator.TRAINING_PHASE_SCHEMA,
                "status": "completed",
                "returncode": 0,
                "run_id": f"{row_id}:{seed}",
                "phase": {
                    "phase_id": phase_id,
                    "config": str(config.resolve()),
                },
                "output_dir": str(output.resolve()),
                "inputs": {
                    "records": [_artifact(config, role="config_dependency")]
                },
                "postflight": postflight,
                "postflight_artifact": _artifact(postflight_path),
            }
            (output / "launch_manifest.json").write_text(
                json.dumps(phase_manifest, indent=2) + "\n", encoding="utf-8"
            )
            completed.append(
                {
                    "phase_id": phase_id,
                    "status": "completed",
                    "output_dir": str(output.resolve()),
                    "checkpoint": checkpoint_record,
                }
            )
            planned.append(
                {
                    "phase": {
                        "phase_id": phase_id,
                        "config": str(config.resolve()),
                    },
                    "output_dir": str(output.resolve()),
                }
            )
            final_checkpoint = checkpoint
        assert final_checkpoint is not None
        sequence = {
            "schema": evaluator.TRAINING_SEQUENCE_SCHEMA,
            "status": status,
            "run_id": f"{row_id}:{seed}",
            "row": {"row_id": row_id, "table": "D" if row_id == "S3" else "B"},
            "seed": seed,
            "output_dir": str(run_root.resolve()),
            "phases": planned,
            "completed_phases": completed,
        }
        (run_root / "sequence_manifest.json").write_text(
            json.dumps(sequence, indent=2) + "\n", encoding="utf-8"
        )
        return run_root, config, final_checkpoint

    def _runtime(self, root: Path) -> evaluator.Runtime:
        return evaluator.Runtime(
            python=Path(sys.executable).resolve(),
            data_root=root.resolve(),
            device="cuda:0",
            batch_size=4,
            num_workers=0,
            amp=True,
            log_every=5,
        )

    def _token_training_run(
        self,
        root: Path,
        *,
        row_id: str = "L4",
        seed: int = 17,
        status: str = "completed",
    ) -> tuple[Path, Path, Path]:
        run_root = root / f"{row_id}_seed{seed}"
        run_root.mkdir()
        config = root / f"{row_id}_config.py"
        config.write_text("stage_b = True\n", encoding="utf-8")
        dataset_manifest = root / "datasets.json"
        dataset_manifest.write_text('{"train": []}\n', encoding="utf-8")
        dataset_source = root / "train.jsonl"
        dataset_source.write_text('{"sample": 1}\n', encoding="utf-8")
        checkpoint = run_root / "checkpoint_iter.pth"
        checkpoint.write_bytes(f"{row_id}:{seed}:joint".encode("ascii"))
        checkpoint_record = _artifact(checkpoint)
        row = {
            "row_id": row_id,
            "config": str(config.resolve()),
            "token_objective": "edit_bce",
            "predicate_pair_rank_weight": 1.0,
            "positive_weight": 1.0,
            "shared_weight": 0.25,
            "edit_weight": 1.0,
        }
        postflight = {
            "schema": "pivot.stageb.token_ablation_postflight/v2",
            "status": "passed",
            "checkpoint_metadata": {
                "args": {
                    "config_file": str(config.resolve()),
                    "output_dir": str(run_root.resolve()),
                }
            },
            "input_rehash": {"status": "passed"},
            "artifacts": {"checkpoint": checkpoint_record},
        }
        postflight_path = run_root / "postflight.json"
        postflight_path.write_text(
            json.dumps(postflight, indent=2) + "\n", encoding="utf-8"
        )
        postflight_record = _artifact(postflight_path)
        launch = {
            "schema": evaluator.TOKEN_TRAINING_PHASE_SCHEMA,
            "status": "completed",
            "returncode": 0,
            "run_id": f"{row_id}:{seed}",
            "seed": seed,
            "row": row,
            "output_dir": str(run_root.resolve()),
            "inputs": {
                "config_dependencies": [_artifact(config)],
                "dataset_manifest": _artifact(dataset_manifest),
                "dataset_source_files": [_artifact(dataset_source)],
                "repository_sources": [],
                "stage_a_initializer": _artifact(checkpoint),
                "scorer_warmstart": _artifact(checkpoint),
            },
            "postflight": postflight,
            "postflight_artifact": postflight_record,
        }
        (run_root / "launch_manifest.json").write_text(
            json.dumps(launch, indent=2) + "\n", encoding="utf-8"
        )
        sequence = {
            "schema": evaluator.TOKEN_TRAINING_SEQUENCE_SCHEMA,
            "status": status,
            "run_id": f"{row_id}:{seed}",
            "row": row,
            "seed": seed,
            "output_dir": str(run_root.resolve()),
            "phases": [
                {
                    "phase_id": "joint",
                    "output_dir": str(run_root.resolve()),
                    "optimizer_updates": 1000,
                    "contributes_to_budget": True,
                }
            ],
            "completed_phases": [
                {
                    "phase_id": "joint",
                    "status": "completed",
                    "output_dir": str(run_root.resolve()),
                    "checkpoint": checkpoint_record,
                    "postflight": postflight_record,
                }
            ],
        }
        (run_root / "sequence_manifest.json").write_text(
            json.dumps(sequence, indent=2) + "\n", encoding="utf-8"
        )
        return run_root, config, checkpoint

    def _minimal_plan(self, root: Path) -> tuple[dict, evaluator.Runtime]:
        input_path = root / "input.txt"
        input_path.write_text("sealed\n", encoding="utf-8")
        output = root / "evaluation"
        cache = evaluator.HashCache()
        plan = {
            "schema": evaluator.SCHEMA,
            "status": "planned",
            "evaluation_id": "fixture",
            "output_dir": str(output),
            "source": {
                "checkpoint": str(input_path),
                "checkpoint_sha256": _sha256(input_path),
            },
            "inputs": {
                "records": [
                    evaluator._file_record(input_path, cache, roles=("fixture",))
                ]
            },
            "commands": [
                {
                    "phase_id": "ref8_strict2031",
                    "command": ["fake", "primary"],
                    "console_log": str(output / "primary.log"),
                },
                {
                    "phase_id": "strict1607",
                    "command": ["fake", "supplemental"],
                    "console_log": str(output / "supplemental.log"),
                },
            ],
        }
        return plan, self._runtime(root)

    def test_list_exposes_fixed_protocol_and_all_matrix_runs(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = evaluator.main(["list", "--json"])
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["eval_seed"], 42)
        self.assertEqual(payload["ref_splits"], list(evaluator.REF_SPLITS))
        self.assertEqual(len(payload["paper_training_run_ids"]), 72)
        self.assertIn("L0:17", payload["paper_training_run_ids"])
        self.assertIn("L10:73", payload["paper_training_run_ids"])
        self.assertIn("D3m:17", payload["paper_training_run_ids"])
        self.assertIn("S3:73", payload["paper_training_run_ids"])
        self.assertIn("M0:17", payload["paper_training_run_ids"])
        self.assertIn("M0N:73", payload["paper_training_run_ids"])
        self.assertEqual(payload["formal_paper_training_rows"], ["M0", "M0N"])
        self.assertEqual(
            payload["processes"], ["ref8_strict2031", "strict1607_skip_ref"]
        )
        self.assertEqual(
            payload["profiles"][evaluator.SCREEN_PROFILE]["processes"],
            ["validation_calibration"],
        )

    def test_resolves_completed_single_phase_checkpoint_and_training_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, config, checkpoint = self._training_run(root)
            source = evaluator._resolve_pivot_source(
                run_root, evaluator.HashCache(), allow_nonformal_fixture=True
            )
            self.assertEqual(source.kind, "pivot_paper_training_run")
            self.assertEqual(source.training_run_id, "D3m:17")
            self.assertEqual(source.final_phase_id, "joint")
            self.assertEqual(source.config, config.resolve())
            self.assertEqual(source.checkpoint, checkpoint.resolve())
            self.assertEqual(source.checkpoint_sha256, _sha256(checkpoint))

    def test_s3_resolves_confidence_checkpoint_and_never_rank_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, config, checkpoint = self._training_run(root, row_id="S3")
            source = evaluator._resolve_pivot_source(
                run_root, evaluator.HashCache(), allow_nonformal_fixture=True
            )
            self.assertEqual(source.training_run_id, "S3:17")
            self.assertEqual(source.final_phase_id, "confidence")
            self.assertEqual(source.config, config.resolve())
            self.assertEqual(source.checkpoint, checkpoint.resolve())
            self.assertEqual(source.checkpoint.parent.name, "confidence")
            self.assertNotEqual(source.checkpoint.parent.name, "rank")
            self.assertEqual(source.training_phase, "final")
            self.assertFalse(source.diagnostic_only)
            self.assertEqual(source.selected_phase_id, "confidence")

    def test_s3_rank_phase_is_explicit_diagnostic_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, final_checkpoint = self._training_run(
                root, row_id="S3"
            )
            source = evaluator._resolve_pivot_source(
                run_root,
                evaluator.HashCache(),
                training_phase="rank",
                allow_nonformal_fixture=True,
            )
            self.assertEqual(
                source.kind, "pivot_paper_training_run_rank_diagnostic"
            )
            self.assertEqual(source.training_run_id, "S3:17")
            self.assertEqual(source.training_phase, "rank")
            self.assertTrue(source.diagnostic_only)
            self.assertEqual(source.selected_phase_id, "rank")
            self.assertEqual(source.checkpoint.parent.name, "rank")
            self.assertEqual(source.final_phase_id, "confidence")
            self.assertEqual(source.final_phase_manifest.parent.name, "confidence")
            self.assertEqual(source.selected_phase_manifest.parent.name, "rank")
            self.assertNotEqual(source.checkpoint, final_checkpoint.resolve())

    def test_rank_phase_rejects_non_s3_and_requires_completed_final_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, _ = self._training_run(root, row_id="D3m")
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "restricted.*S3"
            ):
                evaluator._resolve_pivot_source(
                    run_root,
                    evaluator.HashCache(),
                    training_phase="rank",
                    allow_nonformal_fixture=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, _ = self._training_run(
                root, row_id="S3", status="running"
            )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "not completed"
            ):
                evaluator._resolve_pivot_source(
                    run_root,
                    evaluator.HashCache(),
                    training_phase="rank",
                    allow_nonformal_fixture=True,
                )

    def test_rank_phase_rejects_token_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, _ = self._token_training_run(root)
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "restricted.*S3"
            ):
                evaluator._resolve_pivot_source(
                    run_root,
                    evaluator.HashCache(),
                    training_phase="rank",
                    allow_nonformal_fixture=True,
                )

    def test_resolves_completed_token_matrix_checkpoint_config_and_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, config, checkpoint = self._token_training_run(root)
            source = evaluator._resolve_pivot_source(
                run_root, evaluator.HashCache(), allow_nonformal_fixture=True
            )
            self.assertEqual(source.kind, "pivot_token_ablation_training_run")
            self.assertEqual(source.training_run_id, "L4:17")
            self.assertEqual(source.final_phase_id, "joint")
            self.assertEqual(source.config, config.resolve())
            self.assertEqual(source.checkpoint, checkpoint.resolve())
            self.assertEqual(len(source.training_data), 2)
            self.assertEqual(
                {path.name for path in source.training_data},
                {"datasets.json", "train.jsonl"},
            )

    def test_token_source_rejects_noncompleted_or_mismatched_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, _ = self._token_training_run(root, status="running")
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "not completed"
            ):
                evaluator._resolve_pivot_source(
                    run_root,
                    evaluator.HashCache(),
                    allow_nonformal_fixture=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, _ = self._token_training_run(root)
            launch_path = run_root / "launch_manifest.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["row"]["config"] = str(root / "wrong.py")
            launch_path.write_text(
                json.dumps(launch, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "row contracts differ"
            ):
                evaluator._resolve_pivot_source(
                    run_root,
                    evaluator.HashCache(),
                    allow_nonformal_fixture=True,
                )

    def test_formal_contract_rejects_memory_soak_budgets(self):
        from tools import run_stageb_paper_ablation_matrices as paper_launcher
        from tools import run_stageb_token_ablation_matrix as token_launcher

        token_row = token_launcher.ROW_BY_ID["L1"]
        token_root = (
            token_launcher.DEFAULT_OUTPUT_ROOT / "L1" / "seed17"
        ).resolve()
        with self.assertRaisesRegex(
            evaluator.PaperEvaluationError, "1000-update"
        ):
            evaluator._validate_token_formal_run_contract(
                sequence={
                    "training_seeds_contract": list(token_launcher.SEEDS),
                    "equal_budget_contract": {
                        "batch_size": 40,
                        "optimizer_updates": 50,
                        "contributing_phase_updates": {"joint": 50},
                    },
                },
                run_root=token_root,
                row=asdict(token_row),
                row_id="L1",
                seed=17,
            )

        paper_row = paper_launcher.ROW_BY_ID["D3m"]
        paper_root = (
            paper_launcher.DEFAULT_TN_OUTPUT_ROOT / "D3m" / "seed17"
        ).resolve()
        with self.assertRaisesRegex(
            evaluator.PaperEvaluationError, "1000-update"
        ):
            evaluator._validate_paper_formal_run_contract(
                sequence={
                    "training_seeds_contract": list(paper_launcher.SEEDS),
                    "equal_budget_contract": {
                        "batch_size": 40,
                        "optimizer_updates": 50,
                        "s3_probe_updates_excluded": 0,
                        "contributing_phase_updates": {"joint": 50},
                    },
                },
                run_root=paper_root,
                row=asdict(paper_row),
                row_id="D3m",
                seed=17,
            )

    def test_nondefault_token_root_requires_completed_queue_attestation(self):
        from tools import run_stageb_serial_matrix_queue as queue_runner
        from tools import run_stageb_token_ablation_matrix as token_launcher

        row_id = "L1"
        seed = 17
        row = token_launcher.ROW_BY_ID[row_id]
        sequence = {
            "training_seeds_contract": list(token_launcher.SEEDS),
            "equal_budget_contract": {
                "batch_size": 40,
                "optimizer_updates": 1000,
                "contributing_phase_updates": {"joint": 1000},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_base = root / "alternate-token-root"
            run_root = output_base / row_id / f"seed{seed}"
            run_root.mkdir(parents=True)
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "training-queue-dir"
            ):
                evaluator._validate_token_formal_run_contract(
                    sequence=sequence,
                    run_root=run_root,
                    row=asdict(row),
                    row_id=row_id,
                    seed=seed,
                )

            queue_dir = root / "queue"
            job_dir = queue_dir / "jobs" / "001-L1_17" / "job"
            job_dir.mkdir(parents=True)
            (queue_dir / "queue.json").write_text("{}\n", encoding="utf-8")
            (job_dir / "launch.json").write_text("{}\n", encoding="utf-8")
            (job_dir / "status.json").write_text("{}\n", encoding="utf-8")
            run_id = f"{row_id}:{seed}"
            plan_sha = "a" * 64
            queue = {
                "status": "completed",
                "plan_sha256": plan_sha,
                "plan": {
                    "queue_id": "queue-id",
                    "repository_root": str(evaluator.ARTIFACT_REPOSITORY_ROOT),
                    "runtime_environment": {
                        "PIVOT_TOKEN_OUTPUT_ROOT": str(output_base)
                    },
                    "items": [{"run_id": run_id, "runner": "token"}],
                },
                "items": [
                    {
                        "run_id": run_id,
                        "runner": "token",
                        "status": "completed",
                        "output_root": str(run_root),
                    }
                ],
            }
            verification = {
                "status": "passed",
                "queue_id": "queue-id",
                "plan_sha256": plan_sha,
                "verified_items": [
                    {
                        "run_id": run_id,
                        "runner": "token",
                        "output_root": str(run_root),
                        "job_dir": str(job_dir),
                    }
                ],
            }
            with (
                patch.object(queue_runner, "load_queue", return_value=queue),
                patch.object(
                    queue_runner, "verify_queue", return_value=verification
                ),
            ):
                attestation = evaluator._validate_token_formal_run_contract(
                    sequence=sequence,
                    run_root=run_root,
                    row=asdict(row),
                    row_id=row_id,
                    seed=seed,
                    training_queue_dir=queue_dir,
                )
            self.assertIsNotNone(attestation)
            assert attestation is not None
            self.assertEqual(attestation.manifest, (queue_dir / "queue.json").resolve())
            self.assertEqual(attestation.queue_id, "queue-id")
            self.assertEqual(attestation.plan_sha256, plan_sha)
            self.assertEqual(
                attestation.repository_root,
                evaluator.ARTIFACT_REPOSITORY_ROOT,
            )
            self.assertEqual(
                attestation.artifact_outputs_root,
                evaluator.ARTIFACT_OUTPUTS_ROOT,
            )

            wrong_repository = json.loads(json.dumps(queue))
            wrong_repository["plan"]["repository_root"] = str(root / "other")
            with (
                patch.object(
                    queue_runner, "load_queue", return_value=wrong_repository
                ),
                patch.object(
                    queue_runner, "verify_queue", return_value=verification
                ),
                self.assertRaisesRegex(
                    evaluator.PaperEvaluationError, "repository root mismatch"
                ),
            ):
                evaluator._validate_token_formal_run_contract(
                    sequence=sequence,
                    run_root=run_root,
                    row=asdict(row),
                    row_id=row_id,
                    seed=seed,
                    training_queue_dir=queue_dir,
                )

            failed = {**verification, "status": "failed"}
            with (
                patch.object(queue_runner, "load_queue", return_value=queue),
                patch.object(queue_runner, "verify_queue", return_value=failed),
                self.assertRaisesRegex(
                    evaluator.PaperEvaluationError, "not completed and verified"
                ),
            ):
                evaluator._validate_token_formal_run_contract(
                    sequence=sequence,
                    run_root=run_root,
                    row=asdict(row),
                    row_id=row_id,
                    seed=seed,
                    training_queue_dir=queue_dir,
                )

            drifted_queue = json.loads(json.dumps(queue))
            drifted_queue["plan"]["runtime_environment"][
                "PIVOT_TOKEN_OUTPUT_ROOT"
            ] = str(root / "different-root")
            with (
                patch.object(
                    queue_runner, "load_queue", return_value=drifted_queue
                ),
                patch.object(
                    queue_runner, "verify_queue", return_value=verification
                ),
                self.assertRaisesRegex(
                    evaluator.PaperEvaluationError, "verified queue output root"
                ),
            ):
                evaluator._validate_token_formal_run_contract(
                    sequence=sequence,
                    run_root=run_root,
                    row=asdict(row),
                    row_id=row_id,
                    seed=seed,
                    training_queue_dir=queue_dir,
                )

    def test_queue_contract_errors_are_fail_closed(self):
        from tools import run_stageb_serial_matrix_queue as queue_runner

        with tempfile.TemporaryDirectory() as temporary:
            queue_dir = Path(temporary)
            with (
                patch.object(
                    queue_runner,
                    "load_queue",
                    side_effect=queue_runner.QueueContractError("runner drift"),
                ),
                self.assertRaisesRegex(
                    evaluator.PaperEvaluationError, "runner drift"
                ),
            ):
                evaluator._token_queue_attestation(
                    queue_dir,
                    run_root=queue_dir / "L1" / "seed17",
                    row_id="L1",
                    seed=17,
                )

    def test_rejects_noncompleted_sequence_and_checkpoint_hash_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, _ = self._training_run(root, status="running")
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "not completed"
            ):
                evaluator._resolve_pivot_source(
                    run_root,
                    evaluator.HashCache(),
                    allow_nonformal_fixture=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, checkpoint = self._training_run(root)
            checkpoint.write_bytes(b"drifted-after-training")
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "(size_bytes|SHA-256) mismatch"
            ):
                evaluator._resolve_pivot_source(
                    run_root,
                    evaluator.HashCache(),
                    allow_nonformal_fixture=True,
                )

    def test_rejects_checkpoint_metadata_config_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, _, checkpoint = self._training_run(root)
            phase_manifest_path = checkpoint.parent / "launch_manifest.json"
            phase_manifest = json.loads(phase_manifest_path.read_text())
            postflight_path = checkpoint.parent / "postflight.json"
            postflight = json.loads(postflight_path.read_text())
            other = root / "other_config.py"
            other.write_text("stage_b = False\n", encoding="utf-8")
            postflight["checkpoint_metadata"]["args"]["config_file"] = str(
                other.resolve()
            )
            postflight_path.write_text(
                json.dumps(postflight, indent=2) + "\n", encoding="utf-8"
            )
            phase_manifest["postflight"] = postflight
            phase_manifest["postflight_artifact"] = _artifact(postflight_path)
            phase_manifest_path.write_text(
                json.dumps(phase_manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "metadata config"
            ):
                evaluator._resolve_pivot_source(
                    run_root,
                    evaluator.HashCache(),
                    allow_nonformal_fixture=True,
                )

    def test_historical_baseline_requires_explicit_config_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "gdino.py"
            checkpoint = root / "checkpoint.pth"
            config.write_text("stage_b = False\n", encoding="utf-8")
            checkpoint.write_bytes(b"historical-gdino")
            source = evaluator._resolve_baseline_source(
                config, checkpoint, "GDINO Stage-B data FT", evaluator.HashCache()
            )
            self.assertEqual(source.kind, "historical_pure_gdino_explicit")
            self.assertEqual(source.evaluation_id, "GDINO_Stage-B_data_FT")
            self.assertEqual(source.config, config.resolve())
            self.assertEqual(source.checkpoint_sha256, _sha256(checkpoint))
            self.assertEqual(evaluator._config_paths(config), [config.resolve()])

    def test_plan_has_exact_two_process_protocol_and_fixed_full_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            config.write_text("stage_b = True\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            source = evaluator._resolve_baseline_source(
                config, checkpoint, "baseline", evaluator.HashCache()
            )
            runtime = self._runtime(root)
            output = root / "fresh-output"
            strict_records = {
                label: {
                    **evaluator._file_record(
                        Path(spec["path"]),
                        evaluator.HashCache(),
                        roles=(label,),
                    ),
                    "rows": spec["rows"],
                    "source_counts": spec["source_counts"],
                }
                for label, spec in evaluator.STRICT_SPECS.items()
            }
            with (
                patch.object(
                    evaluator,
                    "_strict_manifest_record",
                    side_effect=lambda label, cache: strict_records[label],
                ),
                patch.object(evaluator, "_config_paths", return_value=[config]),
                patch.object(
                    evaluator,
                    "_evaluation_code_paths",
                    return_value=[evaluator.EVALUATOR],
                ),
                patch.object(evaluator, "_data_input_paths", return_value=[]),
            ):
                plan = evaluator.build_plan(
                    runtime, source, output, evaluator.HashCache()
                )
            self.assertFalse(output.exists())
            self.assertEqual(plan["runtime"]["eval_seed"], 42)
            self.assertEqual(plan["runtime"]["max_ref_batches"], 0)
            self.assertEqual(plan["runtime"]["max_tn_batches"], 0)
            self.assertEqual(len(plan["commands"]), 2)
            primary = plan["commands"][0]["command"]
            supplemental = plan["commands"][1]["command"]
            self.assertNotIn("--skip_ref", primary)
            self.assertNotIn("--skip_tn", primary)
            ref_start = primary.index("--ref_splits") + 1
            tn_start = primary.index("--tn_jsonl")
            self.assertEqual(tuple(primary[ref_start:tn_start]), evaluator.REF_SPLITS)
            self.assertIn(str(evaluator.STRICT_SPECS["strict2031"]["path"]), primary)
            self.assertIn("--skip_ref", supplemental)
            self.assertIn(str(evaluator.STRICT_SPECS["strict1607"]["path"]), supplemental)
            for command in (primary, supplemental):
                self.assertEqual(command[command.index("--seed") + 1], "42")
                self.assertEqual(
                    command[command.index("--max_ref_batches") + 1], "0"
                )
                self.assertEqual(
                    command[command.index("--max_tn_batches") + 1], "0"
                )
                self.assertEqual(
                    command[command.index("--config") + 1], str(config.resolve())
                )
                self.assertEqual(
                    command[command.index("--ckpts") + 1], str(checkpoint.resolve())
                )

    def test_screen_plan_exposes_only_ref_validation_and_sealed_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            queue_manifest = root / "queue.json"
            detached_launch = root / "detached_launch.json"
            detached_status = root / "detached_status.json"
            config.write_text("stage_b = True\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            queue_manifest.write_text("{}\n", encoding="utf-8")
            detached_launch.write_text("{}\n", encoding="utf-8")
            detached_status.write_text("{}\n", encoding="utf-8")
            source = evaluator.EvaluationSource(
                kind="pivot_token_ablation_training_run",
                evaluation_id="L0_seed17",
                config=config.resolve(),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=_sha256(checkpoint),
                training_run_id="L0:17",
                training_seed=17,
                training_queue_manifest=queue_manifest.resolve(),
                training_queue_detached_launch=detached_launch.resolve(),
                training_queue_detached_status=detached_status.resolve(),
                training_queue_id="queue-id",
                training_queue_plan_sha256="a" * 64,
            )
            with (
                patch.object(evaluator, "_config_paths", return_value=[config]),
                patch.object(
                    evaluator,
                    "_evaluation_code_paths",
                    return_value=[evaluator.EVALUATOR],
                ),
                patch.object(evaluator, "_data_input_paths", return_value=[]),
            ):
                plan = evaluator.build_plan(
                    self._runtime(root),
                    source,
                    root / "screen-output",
                    evaluator.HashCache(),
                    profile=evaluator.SCREEN_PROFILE,
                )
            self.assertEqual(plan["protocol"]["profile"], evaluator.SCREEN_PROFILE)
            self.assertEqual(
                plan["protocol"]["ref_splits"], list(evaluator.SCREEN_REF_SPLITS)
            )
            self.assertEqual(plan["protocol"]["strict_manifests"], {})
            self.assertEqual(len(plan["commands"]), 1)
            command = plan["commands"][0]["command"]
            ref_start = command.index("--ref_splits") + 1
            tn_start = command.index("--tn_jsonl")
            self.assertEqual(
                tuple(command[ref_start:tn_start]), evaluator.SCREEN_REF_SPLITS
            )
            self.assertIn("--screen_calibration_manifest", command)
            self.assertNotIn("--skip_ref", command)
            self.assertNotIn(str(evaluator.STRICT_SPECS["strict2031"]["path"]), command)
            self.assertNotIn(str(evaluator.STRICT_SPECS["strict1607"]["path"]), command)
            input_roles = {
                role
                for record in plan["inputs"]["records"]
                for role in record["roles"]
            }
            self.assertIn("screen_calibration_source", input_roles)
            self.assertIn("screen_calibration_audit", input_roles)
            self.assertIn("training_queue_manifest", input_roles)
            self.assertIn("training_queue_detached_launch", input_roles)
            self.assertIn("training_queue_detached_status", input_roles)
            self.assertEqual(plan["source"]["training_queue_id"], "queue-id")
            self.assertEqual(
                plan["source"]["training_queue_plan_sha256"], "a" * 64
            )

    def test_matrix_plan_is_validation_only_and_requires_formal_pivot_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            matrix_queue_spec = root / "matrix_queue_spec.json"
            config.write_text("stage_b = True\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            matrix_queue_spec.write_text(
                '{"schema":"fixture.matrix_queue/v1"}\n', encoding="ascii"
            )
            source = evaluator.EvaluationSource(
                kind="pivot_token_ablation_training_run",
                evaluation_id="L10_seed42",
                config=config.resolve(),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=_sha256(checkpoint),
                training_run_id="L10:42",
                training_seed=42,
            )
            with (
                patch.object(evaluator, "_config_paths", return_value=[config]),
                patch.object(
                    evaluator,
                    "_evaluation_code_paths",
                    return_value=[evaluator.EVALUATOR],
                ),
                patch.object(evaluator, "_data_input_paths", return_value=[]),
                patch.object(evaluator, "_revalidate_matrix_source"),
            ):
                plan = evaluator.build_plan(
                    self._runtime(root),
                    source,
                    root / "matrix-output",
                    evaluator.HashCache(),
                    profile=evaluator.MATRIX_PROFILE,
                    matrix_queue_spec=matrix_queue_spec,
                )
            self.assertEqual(plan["protocol"]["profile"], evaluator.MATRIX_PROFILE)
            self.assertEqual(
                plan["protocol"]["ref_splits"], list(evaluator.SCREEN_REF_SPLITS)
            )
            self.assertEqual(plan["protocol"]["strict_manifests"], {})
            self.assertEqual(
                plan["protocol"]["processes"], ["validation_calibration"]
            )
            self.assertEqual(len(plan["commands"]), 1)
            self.assertEqual(
                plan["matrix_validation_queue_spec"]["path"],
                str(matrix_queue_spec.resolve()),
            )
            self.assertEqual(
                [
                    record
                    for record in plan["inputs"]["records"]
                    if "matrix_validation_queue_spec" in record["roles"]
                ],
                [plan["matrix_validation_queue_spec"]],
            )
            command = plan["commands"][0]["command"]
            self.assertNotIn("--skip_ref", command)
            self.assertNotIn(str(evaluator.STRICT_SPECS["strict2031"]["path"]), command)
            self.assertNotIn(str(evaluator.STRICT_SPECS["strict1607"]["path"]), command)
            roles = {
                role
                for record in plan["inputs"]["records"]
                for role in record["roles"]
            }
            self.assertIn("matrix_calibration_source", roles)
            self.assertIn("matrix_calibration_audit", roles)

            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "predeclared L0-L4"
            ):
                evaluator.build_plan(
                    self._runtime(root),
                    source,
                    root / "screen-output",
                    evaluator.HashCache(),
                    profile=evaluator.SCREEN_PROFILE,
                )

            baseline = evaluator.EvaluationSource(
                kind="historical_pure_gdino",
                evaluation_id="baseline",
                config=config.resolve(),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=_sha256(checkpoint),
            )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "completed formal PIVOT"
            ):
                evaluator.build_plan(
                    self._runtime(root),
                    baseline,
                    root / "baseline-matrix-output",
                    evaluator.HashCache(),
                    profile=evaluator.MATRIX_PROFILE,
                )

    def test_matrix_revalidation_rejects_unresolved_and_drifted_source_tags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "rank" / "checkpoint_iter.pth"
            training_root = root / "training"
            config.write_text("stage_b = True\n", encoding="utf-8")
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"rank-checkpoint")
            training_root.mkdir()
            unresolved = evaluator.EvaluationSource(
                kind="pivot_paper_training_run",
                evaluation_id="S3_seed999",
                config=config.resolve(),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=_sha256(checkpoint),
                training_run_id="S3:999",
                training_seed=999,
                training_phase="final",
                diagnostic_only=False,
                final_phase_id="confidence",
                selected_phase_id="rank",
            )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "no formal training run root"
            ):
                evaluator._revalidate_matrix_source(
                    unresolved, evaluator.HashCache()
                )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "no formal training run root"
            ):
                evaluator.build_plan(
                    self._runtime(root),
                    unresolved,
                    root / "forged-matrix-output",
                    evaluator.HashCache(),
                    profile=evaluator.MATRIX_PROFILE,
                )

            claimed = replace(unresolved, training_run_root=training_root.resolve())
            canonical = replace(
                claimed,
                training_run_id="S3:17",
                training_seed=17,
                selected_phase_id="confidence",
            )
            with patch.object(
                evaluator, "_resolve_pivot_source", return_value=canonical
            ):
                with self.assertRaisesRegex(
                    evaluator.PaperEvaluationError,
                    "differs from re-resolved formal provenance",
                ):
                    evaluator._revalidate_matrix_source(
                        claimed, evaluator.HashCache()
                    )

            diagnostic = replace(
                claimed,
                kind="pivot_paper_training_run_rank_diagnostic",
                evaluation_id="S3_seed999_rank_diagnostic",
                training_phase="rank",
                diagnostic_only=True,
                selected_phase_id="rank",
            )
            cache = evaluator.HashCache()
            with patch.object(
                evaluator, "_resolve_pivot_source", return_value=diagnostic
            ) as resolve_source:
                evaluator._revalidate_matrix_source(diagnostic, cache)
            resolve_source.assert_called_once_with(
                training_root.resolve(),
                cache,
                training_phase="rank",
                training_queue_dir=None,
            )

    def test_rank_diagnostic_is_matrix_validation_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            config.write_text("stage_b = True\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            diagnostic = evaluator.EvaluationSource(
                kind="pivot_paper_training_run_rank_diagnostic",
                evaluation_id="S3_seed17_rank_diagnostic",
                config=config.resolve(),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=_sha256(checkpoint),
                training_run_id="S3:17",
                training_seed=17,
                training_phase="rank",
                diagnostic_only=True,
                final_phase_id="confidence",
                selected_phase_id="rank",
            )
            with (
                patch.object(evaluator, "_config_paths", return_value=[config]),
                patch.object(
                    evaluator,
                    "_evaluation_code_paths",
                    return_value=[evaluator.EVALUATOR],
                ),
                patch.object(evaluator, "_data_input_paths", return_value=[]),
                patch.object(evaluator, "_revalidate_matrix_source") as revalidate,
            ):
                plan = evaluator.build_plan(
                    self._runtime(root),
                    diagnostic,
                    root / "matrix-output",
                    evaluator.HashCache(),
                    profile=evaluator.MATRIX_PROFILE,
                )
            revalidate.assert_called_once()
            self.assertEqual(plan["protocol"]["profile"], evaluator.MATRIX_PROFILE)
            self.assertEqual(plan["protocol"]["strict_manifests"], {})
            self.assertEqual(len(plan["commands"]), 1)
            command = plan["commands"][0]["command"]
            self.assertNotIn(
                str(evaluator.STRICT_SPECS["strict2031"]["path"]), command
            )
            self.assertNotIn(
                str(evaluator.STRICT_SPECS["strict1607"]["path"]), command
            )

            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "matrix_validation profile"
            ):
                evaluator.build_plan(
                    self._runtime(root),
                    diagnostic,
                    root / "screen-output",
                    evaluator.HashCache(),
                    profile=evaluator.SCREEN_PROFILE,
                )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "matrix_validation profile"
            ):
                evaluator.build_plan(
                    self._runtime(root),
                    diagnostic,
                    root / "final-output",
                    evaluator.HashCache(),
                    profile=evaluator.FINAL_PROFILE,
                )

            masquerading = evaluator.EvaluationSource(
                kind="pivot_paper_training_run",
                evaluation_id="S3_seed17",
                config=config.resolve(),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=_sha256(checkpoint),
                training_run_id="S3:17",
                training_seed=17,
                training_phase="rank",
                diagnostic_only=False,
                final_phase_id="confidence",
                selected_phase_id="rank",
            )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "cannot be presented"
            ):
                evaluator.build_plan(
                    self._runtime(root),
                    masquerading,
                    root / "main-output",
                    evaluator.HashCache(),
                )
    def test_screen_tn_postflight_replays_binding_records_and_fpr(self):
        from tools.compare_stageb_fpr95_records import exact_fpr95
        from tools.stageb_eval_records import (
            load_eval_manifest,
            make_eval_record,
            write_eval_records,
        )
        from tools.stageb_screen_calibration import (
            build_manifest as build_screen_manifest,
            summary_fields,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            audit_path = root / "audit.json"
            section = root / "validation_calibration"
            derived_path = section / "tn_eval_inputs" / "tn_screen_calibration.jsonl"
            rows = []
            for index in range(2):
                rows.append(
                    {
                        "sample_id": f"screen:{index}",
                        "image_id": 10 + index,
                        "ann_id": 20 + index,
                        "ref_id": 30 + index,
                        "sent_id": 40 + index,
                        "class_id": 7,
                        "file_name": f"COCO_train2014_{index:012d}.jpg",
                        "split": "train",
                        "dataset": "refcocoplus",
                        "pair_source": "refcoco+_unc",
                        "category_name": "person",
                        "class_norm_name": "person",
                        "target_bbox_used": [1.0, 2.0, 30.0, 40.0],
                        "sent": "person in a red shirt",
                        "try_tn": "person in a blue shirt",
                        "try_tn_head": "person",
                        "try_tn_head_phrase": "person in a red shirt",
                        "replace_category": ["color"],
                        "replace_from": ["red"],
                        "replace_to": ["blue"],
                        "replace_span": [[3, 4]],
                        "tn_edits": [
                            {
                                "category": "color",
                                "replace_from": "red",
                                "replace_to": "blue",
                                "replace_span": [3, 4],
                            }
                        ],
                        "table_b_pair_schema": "stage-b-paper-table-b-scope-preserving-pair-v1",
                        "table_b_id": "D3",
                        "tn_scope": "proposal_covered_verified",
                        "proposal_covered_verified": True,
                        "traceable_counterfactual_edit": True,
                        "visual_verified_negative": True,
                        "global_tn_verified": False,
                        "proposalset_proxy_verified": False,
                        "cached_proposal_coverage_only": True,
                        "all_900_gdino_queries_verified": False,
                        "global_max_label_is_semantic_extrapolation": True,
                    }
                )
            source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            audit_path.write_text('{"schema": "fixture"}\n', encoding="utf-8")
            binding = build_screen_manifest(
                source_path=source_path,
                audit_path=audit_path,
                derived_path=derived_path,
                data_root=root,
            )
            manifest = load_eval_manifest(
                derived_path,
                task="tn",
                split="global",
                manifest_key="tn_global",
            )
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            run_id = evaluator._checkpoint_run_id(checkpoint)
            positive = [0.9, 0.8]
            negative = [0.2, 0.3]
            records = [
                make_eval_record(
                    manifest,
                    index=index,
                    run_id=run_id,
                    valid=True,
                    values={
                        "pos_score": positive[index],
                        "neg_score": negative[index],
                    },
                )
                for index in range(2)
            ]
            records_path = section / "per_example_records" / "tn.records.jsonl"
            write_eval_records(records_path, records)
            fpr = float(exact_fpr95(positive, negative)["fpr"])
            row = {
                "checkpoint": str(checkpoint),
                "run_id": run_id,
                "seed": 42,
                "max_batches": 0,
                "invalid_records": 0,
                "manifest_n": 2,
                "num_pairs": 2,
                "manifest_sha256": binding.derived_manifest["sha256"],
                "records_jsonl": str(records_path),
                "fpr95tpr": fpr,
                **summary_fields(binding),
            }
            summary_path = section / "summary.json"
            summary_path.write_text(
                json.dumps({"refcoco": [], "tn": [row]}), encoding="utf-8"
            )
            contract = {
                "source_manifest": dict(binding.source_manifest),
                "source_audit": dict(binding.source_audit),
            }
            result = evaluator._verify_screen_tn_row(
                {"refcoco": [], "tn": [row]},
                summary_path=summary_path,
                section_dir=section,
                checkpoint=checkpoint,
                run_id=run_id,
                cache=evaluator.HashCache(),
                contract=contract,
            )
            self.assertEqual(result["manifest_n"], 2)
            self.assertEqual(result["scope"], "proposal_covered_verified")
            self.assertTrue(result["single_edit"])

    def test_plan_refuses_any_existing_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            config.write_text("stage_b = True\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            source = evaluator._resolve_baseline_source(
                config, checkpoint, "baseline", evaluator.HashCache()
            )
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                evaluator.build_plan(
                    self._runtime(root), source, output, evaluator.HashCache()
                )

    def test_summary_common_rejects_partial_batches_and_checkpoint_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            row = {
                "checkpoint": str(checkpoint),
                "run_id": evaluator._checkpoint_run_id(checkpoint),
                "seed": 42,
                "max_batches": 1,
                "invalid_records": 0,
            }
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "max_batches=0"
            ):
                evaluator._validate_summary_row_common(
                    row,
                    checkpoint=checkpoint,
                    run_id=evaluator._checkpoint_run_id(checkpoint),
                    seed=42,
                )
            row["max_batches"] = 0
            other = root / "other.pth"
            other.write_bytes(b"other")
            row["checkpoint"] = str(other)
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError, "checkpoint mismatch"
            ):
                evaluator._validate_summary_row_common(
                    row,
                    checkpoint=checkpoint,
                    run_id=evaluator._checkpoint_run_id(checkpoint),
                    seed=42,
                )

    def test_execute_persists_completed_manifest_only_after_postflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, runtime = self._minimal_plan(root)
            postflight = {
                "schema": evaluator.POSTFLIGHT_SCHEMA,
                "status": "passed",
            }
            with (
                patch.object(evaluator, "_stream_command", return_value=0) as stream,
                patch.object(
                    evaluator,
                    "_rehash_inputs",
                    return_value={"schema": evaluator.INPUT_REHASH_SCHEMA, "status": "passed"},
                ),
                patch.object(evaluator, "_postflight", return_value=postflight),
            ):
                result = evaluator._execute(plan, runtime)
            self.assertEqual(result, 0)
            self.assertEqual(stream.call_count, 2)
            launch = json.loads(
                (Path(plan["output_dir"]) / "launch_manifest.json").read_text()
            )
            self.assertEqual(launch["status"], "completed")
            self.assertEqual(
                [value["phase_id"] for value in launch["completed_phases"]],
                ["ref8_strict2031", "strict1607"],
            )
            self.assertEqual(launch["postflight"]["status"], "passed")
            self.assertTrue((Path(plan["output_dir"]) / "postflight.json").is_file())
            self.assertTrue((Path(plan["output_dir"]) / "input_rehash.json").is_file())

    def test_execute_failure_leaves_failed_launch_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, runtime = self._minimal_plan(root)
            stderr = StringIO()
            with (
                patch.object(evaluator, "_stream_command", return_value=9),
                redirect_stderr(stderr),
            ):
                result = evaluator._execute(plan, runtime)
            self.assertEqual(result, 1)
            launch = json.loads(
                (Path(plan["output_dir"]) / "launch_manifest.json").read_text()
            )
            self.assertEqual(launch["status"], "failed")
            self.assertEqual(launch["failure_phase"], "ref8_strict2031")
            self.assertIn("exited with code 9", launch["error"])


if __name__ == "__main__":
    unittest.main()
