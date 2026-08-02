import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from tools import aggregate_stageb_table_d_diagnostics as diagnostics


def _probe_line(update: int, *, cosine: float, count: int = 4) -> str:
    values = {
        "grad_cosine": cosine,
        "grad_cosine_defined": 1.0,
        "grad_rank_norm": 2.0 + update / 1000.0,
        "grad_confidence_norm": 3.0 + update / 1000.0,
        "grad_element_conflict_fraction": 0.25,
        "grad_tensor_conflict_fraction": 0.5,
        "grad_shared_parameter_count": float(count),
        "grad_shared_element_count": 100.0,
    }
    metrics = "  ".join(
        f"stage_b_v22_{key}_unscaled: {value:.4f} ({value:.4f})"
        for key, value in values.items()
    )
    return f"INFO | Epoch: [0]  [{update:4d}/8388]  {metrics}"


def _parsed(seed_offset: float = 0.0, *, count: float = 4.0) -> dict:
    metrics = {
        "grad_cosine": -0.2 + seed_offset,
        "grad_cosine_defined": 1.0,
        "grad_rank_norm": 2.0 + seed_offset,
        "grad_confidence_norm": 3.0 + seed_offset,
        "grad_element_conflict_fraction": 0.25 + seed_offset,
        "grad_tensor_conflict_fraction": 0.5 + seed_offset,
        "grad_shared_parameter_count": count,
        "grad_shared_element_count": 100.0,
    }
    return {"cumulative_mean": metrics}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(path: Path, *, roles=()) -> dict:
    return diagnostics.evaluator._file_record(
        path, diagnostics.evaluator.HashCache(), roles=roles
    )


def _seal_evaluation_fixture(fixture: dict) -> None:
    postflight_path = fixture["postflight_path"]
    launch_path = fixture["launch_path"]
    _write_json(postflight_path, fixture["postflight"])
    fixture["launch"]["postflight"] = fixture["postflight"]
    fixture["launch"]["postflight_artifact"] = _record(
        postflight_path, roles=("postflight",)
    )
    _write_json(launch_path, fixture["launch"])


def _sealed_evaluation_fixture(
    root: Path,
    *,
    phase: str = "confidence",
    seed: int = 17,
    shared_input: Path | None = None,
    training_source=None,
) -> dict:
    root.mkdir(parents=True)
    primary = root / "ref8_strict2031"
    supplemental = root / "strict1607"
    primary.mkdir()
    supplemental.mkdir()
    if training_source is None:
        checkpoint = root.parent / "training" / phase / "checkpoint_iter.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{seed}:{phase}".encode("ascii"))
    else:
        checkpoint = training_source.checkpoint.resolve(strict=True)
        seed = int(training_source.training_seed)
    cache = diagnostics.evaluator.HashCache()
    checkpoint_sha = cache.digest(checkpoint)
    run_id = diagnostics.evaluator._checkpoint_run_id(checkpoint)
    if shared_input is None:
        shared_input = root.parent / "shared_eval_input.json"
        shared_input.write_text("{}\n", encoding="utf-8")
    input_record = diagnostics.evaluator._file_record(
        shared_input, cache, roles=("evaluation_data_input",)
    )
    stat = shared_input.stat()
    rehash = {
        "schema": diagnostics.evaluator.INPUT_REHASH_SCHEMA,
        "status": "passed",
        "records": [
            {
                "path": str(shared_input.resolve()),
                "roles": list(input_record["roles"]),
                "expected_sha256": input_record["sha256"],
                "observed_sha256": input_record["sha256"],
                "observed_size_bytes": int(stat.st_size),
                "observed_mtime_ns": int(stat.st_mtime_ns),
                "passed": True,
            }
        ],
    }
    rehash_path = root / "input_rehash.json"
    _write_json(rehash_path, rehash)

    ref_split = "fixture_val"
    ref_sha = "a" * 64
    ref_records = primary / "per_example_records" / "ref.records.jsonl"
    _write_jsonl(
        ref_records,
        [
            {
                "schema": diagnostics.RECORD_SCHEMA,
                "task": "ref",
                "valid": True,
                "run_id": run_id,
                "manifest_index": 0,
                "manifest_n": 1,
                "manifest_sha256": ref_sha,
                "sample_id": "ref:0",
                "split": ref_split,
                "image_id": 1,
                "ann_id": 2,
                "ref_id": 3,
                "sent_id": 4,
                "correct50": True,
                "top1_iou": 0.9,
            }
        ],
    )
    strict_specs = {}
    strict_artifacts = {}
    for index, split in enumerate(diagnostics.TN_SPLITS):
        source = root.parent / f"{split}.jsonl"
        source.write_text("{}\n", encoding="utf-8")
        source_sha = ("b" if index == 0 else "c") * 64
        derived_sha = ("d" if index == 0 else "e") * 64
        section = primary if split == "strict2031" else supplemental
        records = section / "per_example_records" / f"{split}.records.jsonl"
        _write_jsonl(
            records,
            [
                {
                    "schema": diagnostics.RECORD_SCHEMA,
                    "task": "tn",
                    "valid": True,
                    "run_id": run_id,
                    "manifest_index": 0,
                    "manifest_n": 1,
                    "manifest_sha256": derived_sha,
                    "sample_id": f"{split}:0",
                    "split": split,
                    "image_id": 1,
                    "ann_id": 2,
                    "ref_id": 3,
                    "sent_id": 4,
                    "pos_score": 0.9,
                    "neg_score": 0.1,
                }
            ],
        )
        strict_specs[split] = {
            "path": source,
            "rows": 1,
            "sha256": source_sha,
        }
        strict_artifacts[split] = {
            "summary_fpr95": float(
                diagnostics.exact_fpr95([0.9], [0.1])["fpr"]
            ),
            "manifest_binding_mode": "source_to_derived_v1",
            "manifest_n": 1,
            "source_manifest_sha256": source_sha,
            "derived_manifest_sha256": derived_sha,
            "records": _record(records),
        }

    primary_summary = primary / "summary.json"
    supplemental_summary = supplemental / "summary.json"
    primary_summary.write_text("{}\n", encoding="utf-8")
    supplemental_summary.write_text("{}\n", encoding="utf-8")
    artifacts = {
        "primary_summary": _record(primary_summary),
        "supplemental_summary": _record(supplemental_summary),
        "ref8": {
            ref_split: {
                "summary_acc50": 1.0,
                "manifest_n": 1,
                "manifest_sha256": ref_sha,
                "records": _record(ref_records),
            }
        },
        **strict_artifacts,
    }
    expected_diagnostic = phase == "rank"
    if training_source is None:
        kind = (
            "pivot_paper_training_run_rank_diagnostic"
            if expected_diagnostic
            else "pivot_paper_training_run"
        )
        source = {
            "kind": kind,
            "training_run_id": f"S3:{seed}",
            "training_seed": seed,
            "training_phase": "rank" if expected_diagnostic else "final",
            "selected_phase_id": phase,
            "final_phase_id": "confidence",
            "diagnostic_only": expected_diagnostic,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
        }
    else:
        source = asdict(training_source)
        for field in (
            "config",
            "checkpoint",
            "training_run_root",
            "sequence_manifest",
            "final_phase_manifest",
            "training_postflight",
            "selected_phase_manifest",
            "selected_training_postflight",
            "training_queue_manifest",
            "training_queue_detached_launch",
            "training_queue_detached_status",
        ):
            value = source[field]
            source[field] = str(value) if value is not None else None
        source["training_data"] = [str(path) for path in training_source.training_data]
    postflight = {
        "schema": diagnostics.evaluator.POSTFLIGHT_SCHEMA,
        "status": "passed",
        "profile": diagnostics.evaluator.FINAL_PROFILE,
        "evaluation_id": f"fixture_{seed}_{phase}",
        "fixed_runtime": {
            "eval_seed": diagnostics.evaluator.EVAL_SEED,
            "max_ref_batches": 0,
            "max_tn_batches": 0,
        },
        "input_rehash": rehash,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "run_id": run_id,
        },
        "contracts": {
            key: True
            for key in (
                "ref_split_set_exact",
                "full_per_example_records",
                "zero_invalid_records",
                "locked_manifest_binding",
                "checkpoint_consistent_across_all_rows",
                "strict1607_skip_ref_observed",
            )
        },
        "artifacts": artifacts,
    }
    launch = {
        "schema": diagnostics.evaluator.SCHEMA,
        "status": "completed",
        "evaluation_id": postflight["evaluation_id"],
        "output_dir": str(root.resolve()),
        "output_dir_fresh_at_plan": True,
        "completed_phases": [
            {
                "phase_id": "ref8_strict2031",
                "status": "completed",
                "returncode": 0,
            },
            {"phase_id": "strict1607", "status": "completed", "returncode": 0},
        ],
        "protocol": {
            "profile": diagnostics.evaluator.FINAL_PROFILE,
            "ref_splits": [ref_split],
            "processes": ["ref8_strict2031", "strict1607"],
            "strict1607_skip_ref": True,
        },
        "runtime": {
            "eval_seed": diagnostics.evaluator.EVAL_SEED,
            "max_ref_batches": 0,
            "max_tn_batches": 0,
        },
        "source": source,
        "inputs": {"records": [input_record]},
        "input_rehash_artifact": _record(rehash_path),
    }
    fixture = {
        "root": root,
        "launch_path": root / "launch_manifest.json",
        "postflight_path": root / "postflight.json",
        "launch": launch,
        "postflight": postflight,
        "ref_split": ref_split,
        "ref_contract": {ref_split: {"rows": 1, "sha256": ref_sha}},
        "strict_specs": strict_specs,
        "ref_records": ref_records,
        "source": source,
    }
    _seal_evaluation_fixture(fixture)
    return fixture


def _gradient_training_source(root: Path, *, row_id: str, seed: int):
    root.mkdir(parents=True)
    config = root / "config.py"
    checkpoint = root / "checkpoint_iter.pth"
    sequence = root / "sequence_manifest.json"
    launch = root / "launch_manifest.json"
    postflight = root / "postflight.json"
    native_log = root / "info.txt"
    config.write_text("stage_b = True\n", encoding="utf-8")
    checkpoint.write_bytes(f"{row_id}:{seed}".encode("ascii"))
    native_log.write_text("fixture gradient log\n", encoding="utf-8")
    _write_json(sequence, {"status": "completed", "run_id": f"{row_id}:{seed}"})
    _write_json(
        launch,
        {
            "phase": {"phase_id": "joint", "diagnostic_interval": 100},
            "fixed_contract": {"gradient_diagnostic_interval": 100},
        },
    )
    _write_json(
        postflight,
        {
            "checkpoint_metadata": {
                "args": {
                    "stage_b_v22_gradient_diagnostic_interval": 100,
                    "gradient_accumulation_steps": 1,
                }
            },
            "artifacts": {"native_info_log": _record(native_log)},
        },
    )
    return diagnostics.evaluator.EvaluationSource(
        kind="pivot_paper_training_run",
        evaluation_id=f"{row_id}_seed{seed}",
        config=config.resolve(),
        checkpoint=checkpoint.resolve(),
        checkpoint_sha256=diagnostics.evaluator.HashCache().digest(checkpoint),
        training_run_id=f"{row_id}:{seed}",
        training_seed=seed,
        training_run_root=root.resolve(),
        sequence_manifest=sequence.resolve(),
        training_phase="final",
        diagnostic_only=False,
        final_phase_id="joint",
        final_phase_manifest=launch.resolve(),
        training_postflight=postflight.resolve(),
        selected_phase_id="joint",
        selected_phase_manifest=launch.resolve(),
        selected_training_postflight=postflight.resolve(),
    )


def _s3_training_sources(root: Path, *, seed: int):
    root.mkdir(parents=True)
    sequence = root / "sequence_manifest.json"
    data = root / "training_data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    _write_json(sequence, {"status": "completed", "run_id": f"S3:{seed}"})
    paths = {}
    for phase in ("rank", "confidence"):
        phase_root = root / phase
        phase_root.mkdir()
        config = phase_root / "config.py"
        checkpoint = phase_root / "checkpoint_iter.pth"
        launch = phase_root / "launch_manifest.json"
        postflight = phase_root / "postflight.json"
        config.write_text(f"phase = {phase!r}\n", encoding="utf-8")
        checkpoint.write_bytes(f"S3:{seed}:{phase}".encode("ascii"))
        paths[phase] = {
            "root": phase_root,
            "config": config,
            "checkpoint": checkpoint,
            "launch": launch,
            "postflight": postflight,
        }
    rank_sha = diagnostics.evaluator.HashCache().digest(paths["rank"]["checkpoint"])
    _write_json(paths["rank"]["launch"], {"inputs": {"records": []}})
    _write_json(paths["rank"]["postflight"], {"status": "passed"})
    _write_json(
        paths["confidence"]["launch"],
        {
            "inputs": {
                "records": [
                    {
                        **_record(paths["rank"]["checkpoint"]),
                        "role": "rank_phase_model_state_pretrain",
                    }
                ]
            }
        },
    )
    _write_json(
        paths["confidence"]["postflight"],
        {
            "status": "passed",
            "model_state_ancestry": {
                "pretrain_path": str(paths["rank"]["checkpoint"].resolve()),
                "pretrain_sha256": rank_sha,
                "pretrain_manifest_role": "rank_phase_model_state_pretrain",
                "pretrain_mode": "model_state_only_no_optimizer_resume",
                "checkpoint_resume_argument": None,
                "scorer_warmstart_applied": False,
            },
        },
    )
    common = {
        "training_run_id": f"S3:{seed}",
        "training_seed": seed,
        "training_run_root": root.resolve(),
        "sequence_manifest": sequence.resolve(),
        "final_phase_id": "confidence",
        "final_phase_manifest": paths["confidence"]["launch"].resolve(),
        "training_postflight": paths["confidence"]["postflight"].resolve(),
        "training_data": (data.resolve(),),
    }
    rank = diagnostics.evaluator.EvaluationSource(
        kind="pivot_paper_training_run_rank_diagnostic",
        evaluation_id=f"S3_seed{seed}_rank_diagnostic",
        config=paths["rank"]["config"].resolve(),
        checkpoint=paths["rank"]["checkpoint"].resolve(),
        checkpoint_sha256=rank_sha,
        training_phase="rank",
        diagnostic_only=True,
        selected_phase_id="rank",
        selected_phase_manifest=paths["rank"]["launch"].resolve(),
        selected_training_postflight=paths["rank"]["postflight"].resolve(),
        **common,
    )
    confidence = diagnostics.evaluator.EvaluationSource(
        kind="pivot_paper_training_run",
        evaluation_id=f"S3_seed{seed}",
        config=paths["confidence"]["config"].resolve(),
        checkpoint=paths["confidence"]["checkpoint"].resolve(),
        checkpoint_sha256=diagnostics.evaluator.HashCache().digest(
            paths["confidence"]["checkpoint"]
        ),
        training_phase="final",
        diagnostic_only=False,
        selected_phase_id="confidence",
        selected_phase_manifest=paths["confidence"]["launch"].resolve(),
        selected_training_postflight=paths["confidence"]["postflight"].resolve(),
        **common,
    )
    return rank, confidence


def _ref_row(index: int, split: str, *, correct: bool, iou: float) -> dict:
    return {
        "manifest_index": index,
        "manifest_n": 2,
        "manifest_sha256": f"{split}-sha",
        "sample_id": f"{split}:{index}",
        "image_id": index,
        "ann_id": index + 10,
        "ref_id": index + 20,
        "sent_id": index + 30,
        "split": split,
        "correct50": correct,
        "top1_iou": iou,
    }


def _tn_row(index: int, split: str) -> dict:
    return {
        "manifest_index": index,
        "manifest_n": 2,
        "manifest_sha256": f"{split}-sha",
        "sample_id": f"{split}:{index}",
        "image_id": index,
        "ann_id": index + 10,
        "ref_id": index + 20,
        "sent_id": index + 30,
    }


def _evaluation(root: Path, seed: int, phase: str, ref) -> diagnostics.LoadedEvaluation:
    protocol = {
        "profile": "final",
        "ref_splits": list(diagnostics.REF_SPLITS),
        "processes": ["ref8_strict2031", "strict1607"],
    }
    runtime = {"eval_seed": 42, "batch_size": 16}
    launch = {
        "runtime": runtime,
        "protocol": protocol,
        "inputs": {
            "records": [
                {
                    "path": "/data/shared.json",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                    "roles": ["evaluation_data_input"],
                }
            ]
        },
    }
    tn = {
        split: tuple(_tn_row(index, split) for index in range(2))
        for split in diagnostics.TN_SPLITS
    }
    checkpoint = root / f"{phase}.pth"
    checkpoint.write_bytes(phase.encode("ascii"))
    return diagnostics.LoadedEvaluation(
        root=root,
        seed=seed,
        source_kind="fixture",
        phase=phase,
        checkpoint=checkpoint,
        checkpoint_sha256="fixture",
        launch=launch,
        postflight={},
        ref=ref,
        tn=tn,
        fpr95={"strict2031": 0.4, "strict1607": 0.3},
    )


class StageBTableDDiagnosticsTest(unittest.TestCase):
    def _load_fixture(self, fixture: dict):
        with (
            patch.object(
                diagnostics, "REF_SPLITS", (fixture["ref_split"],)
            ),
            patch.object(
                diagnostics, "REF_SPLIT_CONTRACT", fixture["ref_contract"]
            ),
            patch.object(
                diagnostics.evaluator,
                "STRICT_SPECS",
                fixture["strict_specs"],
            ),
        ):
            return diagnostics._load_evaluation(
                fixture["root"],
                seed=17,
                expected_kind="pivot_paper_training_run",
                expected_phase="confidence",
                evidence=diagnostics.Evidence(),
            )

    def test_gradient_parser_uses_fixed_cumulative_probe_means(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "info.txt"
            path.write_text(
                "\n".join(
                    _probe_line(update, cosine=-0.1 - update / 10000.0)
                    for update in diagnostics.PROBE_UPDATES
                )
                + "\n",
                encoding="utf-8",
            )
            result = diagnostics.parse_gradient_log(path)
            self.assertEqual(result["probe_updates"], list(range(0, 1000, 100)))
            self.assertAlmostEqual(result["cumulative_mean"]["grad_cosine"], -0.19)
            self.assertEqual(
                result["cumulative_mean"]["grad_shared_parameter_count"], 4.0
            )

    def test_gradient_parser_fails_closed_on_missing_probe_or_count_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.txt"
            path.write_text(
                "\n".join(
                    _probe_line(update, cosine=-0.1)
                    for update in diagnostics.PROBE_UPDATES[:-1]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError, "fixed probes"
            ):
                diagnostics.parse_gradient_log(path)

            drift = Path(temporary) / "drift.txt"
            drift.write_text(
                "\n".join(
                    _probe_line(
                        update,
                        cosine=-0.1,
                        count=4 if update < 500 else 5,
                    )
                    for update in diagnostics.PROBE_UPDATES
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError, "changed across cumulative"
            ):
                diagnostics.parse_gradient_log(drift)

    def test_gradient_aggregation_is_seed_first_ddof_one_and_marks_na(self):
        rows = {
            row_id: {
                17: _parsed(0.0, count=4.0 if row_id == "S0" else 5.0),
                42: _parsed(0.1, count=4.0 if row_id == "S0" else 5.0),
                73: _parsed(0.2, count=4.0 if row_id == "S0" else 5.0),
            }
            for row_id in ("S0", "S1")
        }
        report = diagnostics.aggregate_gradient_rows(rows)
        cosine = report["rows"]["S0"]["aggregate"]["grad_cosine"]
        self.assertAlmostEqual(cosine["mean"], -0.1)
        self.assertAlmostEqual(cosine["sample_std"], 0.1)
        self.assertEqual(cosine["ddof"], 1)
        self.assertEqual(report["rows"]["S2"]["status"], "not_applicable")
        self.assertEqual(report["rows"]["S3"]["status"], "not_applicable")
        self.assertEqual(
            report["negative_cosine_fraction"]["status"], "not_estimable"
        )

    def test_checkpoint_diff_allows_only_changed_validity_head(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable in this unit-test interpreter")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank = root / "rank.pth"
            confidence = root / "confidence.pth"
            state = {
                "stage_b_fixed_text_scorer.decoder.weight": torch.tensor([1.0]),
                "stage_b_fixed_text_scorer.validity_head.0.weight": torch.tensor([2.0]),
            }
            torch.save({"model": state}, rank)
            torch.save(
                {
                    "model": {
                        **state,
                        "stage_b_fixed_text_scorer.validity_head.0.weight": torch.tensor([3.0]),
                    }
                },
                confidence,
            )
            report = diagnostics.checkpoint_allowlist(rank, confidence)
            self.assertTrue(report["all_non_allowlisted_model_tensors_bitwise_equal"])
            self.assertEqual(report["changed_tensor_count"], 1)

            torch.save(
                {
                    "model": {
                        **state,
                        "stage_b_fixed_text_scorer.decoder.weight": torch.tensor([9.0]),
                        "stage_b_fixed_text_scorer.validity_head.0.weight": torch.tensor([3.0]),
                    }
                },
                confidence,
            )
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError, "outside validity_head"
            ):
                diagnostics.checkpoint_allowlist(rank, confidence)

    def test_checkpoint_diff_is_byte_exact_for_signed_zero(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable in this unit-test interpreter")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank = root / "rank.pth"
            confidence = root / "confidence.pth"
            torch.save(
                {
                    "model": {
                        "stage_b_fixed_text_scorer.decoder.weight": torch.tensor(
                            [0.0]
                        ),
                        "stage_b_fixed_text_scorer.validity_head.0.weight": torch.tensor(
                            [1.0]
                        ),
                    }
                },
                rank,
            )
            torch.save(
                {
                    "model": {
                        "stage_b_fixed_text_scorer.decoder.weight": torch.tensor(
                            [-0.0]
                        ),
                        "stage_b_fixed_text_scorer.validity_head.0.weight": torch.tensor(
                            [2.0]
                        ),
                    }
                },
                confidence,
            )
            self.assertTrue(
                torch.equal(torch.tensor([0.0]), torch.tensor([-0.0]))
            )
            self.assertNotEqual(
                torch.tensor([0.0]).numpy().tobytes(),
                torch.tensor([-0.0]).numpy().tobytes(),
            )
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError, "outside validity_head"
            ):
                diagnostics.checkpoint_allowlist(rank, confidence)

    def test_load_evaluation_closes_launch_postflight_and_artifact_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = _sealed_evaluation_fixture(base / "valid")
            loaded = self._load_fixture(fixture)
            self.assertEqual(loaded.phase, "confidence")

            embedded = _sealed_evaluation_fixture(base / "embedded")
            embedded["launch"]["postflight"] = {"status": "different"}
            _write_json(embedded["launch_path"], embedded["launch"])
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError,
                "embedded and persisted postflight differ",
            ):
                self._load_fixture(embedded)

            bad_hash = _sealed_evaluation_fixture(base / "bad_hash")
            bad_hash["launch"]["postflight_artifact"]["sha256"] = "0" * 64
            _write_json(bad_hash["launch_path"], bad_hash["launch"])
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError,
                "postflight artifact SHA-256 mismatch",
            ):
                self._load_fixture(bad_hash)

            wrong_root = _sealed_evaluation_fixture(base / "wrong_root")
            other = base / "other_root"
            other.mkdir()
            wrong_root["launch"]["output_dir"] = str(other.resolve())
            _write_json(wrong_root["launch_path"], wrong_root["launch"])
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError,
                "not its fresh launch root",
            ):
                self._load_fixture(wrong_root)

            escaped = _sealed_evaluation_fixture(base / "escaped")
            outside = base / "historical_ref.records.jsonl"
            outside.write_bytes(escaped["ref_records"].read_bytes())
            escaped["postflight"]["artifacts"]["ref8"][
                escaped["ref_split"]
            ]["records"] = _record(outside)
            _seal_evaluation_fixture(escaped)
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError,
                "escapes its declared evaluation root",
            ):
                self._load_fixture(escaped)

    def test_s3_comparison_reports_regressions_fixes_and_rejects_misalignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank_ref = {
                split: (
                    _ref_row(0, split, correct=True, iou=0.8),
                    _ref_row(1, split, correct=False, iou=0.2),
                )
                for split in diagnostics.REF_SPLITS
            }
            confidence_ref = {
                split: (
                    _ref_row(0, split, correct=False, iou=0.4),
                    _ref_row(1, split, correct=True, iou=0.7),
                )
                for split in diagnostics.REF_SPLITS
            }
            rank = _evaluation(root, 17, "rank", rank_ref)
            confidence = _evaluation(root, 17, "confidence", confidence_ref)
            report = diagnostics.compare_s3_evaluations(rank, confidence)
            split = diagnostics.REF_SPLITS[0]
            self.assertEqual(
                report["ref8"][split][
                    "rank_correct_to_confidence_wrong_count"
                ],
                1,
            )
            self.assertEqual(report["ref8"][split]["confidence_fixes_count"], 1)
            self.assertEqual(report["ref8"][split]["net_correct_delta_count"], 0)
            self.assertEqual(report["ref8"][split]["top1_iou_changed_count"], 2)

            broken = dict(confidence_ref)
            rows = list(broken[split])
            rows[0] = {**rows[0], "sample_id": "wrong"}
            broken[split] = tuple(rows)
            with self.assertRaisesRegex(
                diagnostics.TableDDiagnosticsError, "alignment failed"
            ):
                diagnostics.compare_s3_evaluations(
                    rank, _evaluation(root, 17, "confidence2", broken)
                )

    def test_aggregate_runs_three_seed_training_and_evaluation_replay_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            gradient_manifest = {"S0": {}, "S1": {}}
            for row_id in ("S0", "S1"):
                for seed in diagnostics.EXPECTED_SEEDS:
                    run_root = root / "gradient" / row_id / f"seed{seed}"
                    source = _gradient_training_source(
                        run_root, row_id=row_id, seed=seed
                    )
                    sources[(run_root.resolve(), "final")] = source
                    gradient_manifest[row_id][str(seed)] = str(run_root)

            shared_input = root / "shared_eval_input.json"
            shared_input.write_text("{}\n", encoding="utf-8")
            s3_manifest = {}
            fixture_contract = None
            for seed in diagnostics.EXPECTED_SEEDS:
                training_root = root / "s3_training" / f"seed{seed}"
                rank_source, confidence_source = _s3_training_sources(
                    training_root, seed=seed
                )
                sources[(training_root.resolve(), "rank")] = rank_source
                sources[(training_root.resolve(), "final")] = confidence_source
                rank_fixture = _sealed_evaluation_fixture(
                    root / "s3_evaluations" / f"seed{seed}" / "rank",
                    phase="rank",
                    seed=seed,
                    shared_input=shared_input,
                    training_source=rank_source,
                )
                confidence_fixture = _sealed_evaluation_fixture(
                    root / "s3_evaluations" / f"seed{seed}" / "confidence",
                    phase="confidence",
                    seed=seed,
                    shared_input=shared_input,
                    training_source=confidence_source,
                )
                fixture_contract = confidence_fixture
                s3_manifest[str(seed)] = {
                    "training_run_root": str(training_root),
                    "rank_evaluation_root": str(rank_fixture["root"]),
                    "confidence_evaluation_root": str(
                        confidence_fixture["root"]
                    ),
                }

            manifest = root / "diagnostics_manifest.json"
            _write_json(
                manifest,
                {
                    "schema": diagnostics.INPUT_SCHEMA,
                    "expected_train_seeds": list(diagnostics.EXPECTED_SEEDS),
                    "gradient_training_runs": gradient_manifest,
                    "s3": s3_manifest,
                },
            )
            assert fixture_contract is not None

            def resolve_source(run_root, _cache, *, training_phase="final", **_):
                return sources[(Path(run_root).resolve(), training_phase)]

            with (
                patch.object(
                    diagnostics, "REF_SPLITS", (fixture_contract["ref_split"],)
                ),
                patch.object(
                    diagnostics,
                    "REF_SPLIT_CONTRACT",
                    fixture_contract["ref_contract"],
                ),
                patch.object(
                    diagnostics.evaluator,
                    "STRICT_SPECS",
                    fixture_contract["strict_specs"],
                ),
                patch.object(
                    diagnostics.evaluator,
                    "_resolve_paper_source",
                    side_effect=resolve_source,
                ),
                patch.object(
                    diagnostics, "parse_gradient_log", return_value=_parsed()
                ),
                patch.object(
                    diagnostics,
                    "checkpoint_allowlist",
                    return_value={
                        "status": "passed",
                        "all_non_allowlisted_model_tensors_bitwise_equal": True,
                    },
                ),
            ):
                report = diagnostics.aggregate(manifest)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                len(report["s3_rank_to_confidence"]["seeds"]), 3
            )
            self.assertEqual(
                report["gradient_conflict"]["rows"]["S0"]["status"],
                "available",
            )

    def test_seed_map_requires_exact_formal_seed_set(self):
        self.assertEqual(
            diagnostics._seed_map(
                {"17": "a", "42": "b", "73": "c"}, label="fixture"
            ),
            {17: "a", 42: "b", 73: "c"},
        )
        with self.assertRaisesRegex(
            diagnostics.TableDDiagnosticsError, "seed set"
        ):
            diagnostics._seed_map({"17": "a", "42": "b"}, label="fixture")


if __name__ == "__main__":
    unittest.main()
