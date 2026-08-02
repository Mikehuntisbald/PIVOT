import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import eval_text_groundingdino_refcoco_tn as joint_eval
from tools import run_stageb_dense_duty_formal_evaluation as evaluator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


class DenseDutyFormalEvaluationTest(unittest.TestCase):
    def _source_fixture(self, root: Path):
        output = root / "formal" / "confidence"
        output.mkdir(parents=True)
        checkpoint = output / "checkpoint_iter.pth"
        checkpoint.write_bytes(b"terminal-confidence")
        config = root / "confidence.py"
        config.write_text("stage_b_dense_duty = True\n", encoding="ascii")
        spec = evaluator.training.PhaseSpec(
            phase="confidence",
            config=config,
            dataset=root / "confidence.json",
            dataset_sha256="d" * 64,
            output=output,
            expected_updates=evaluator.training.CONFIDENCE_UPDATES,
        )
        payload = {
            "args": {
                "batch_size": 16,
                "gradient_accumulation_steps": 4,
                "stage_b_v11_expression_microbatch": 16,
                "stage_b_dense_duty_no_stageb_teacher": True,
                "stage_b_dense_duty_execution_scope": "formal",
                "stage_b_dense_duty_phase": "confidence",
                "stage_b_v22_train_phase": "confidence",
                "stage_b_v22_score_ownership": "independent_decoders_two_phase",
            }
        }
        audit = {
            "schema": "pivot.stageb.dense_duty_checkpoint_audit/v1",
            "status": "passed",
            "phase": "confidence",
            "optimizer_updates": evaluator.training.CONFIDENCE_UPDATES,
            "evaluation_scope": "formal",
            "rank_handoff": {
                "status": "passed",
                "phase": "rank",
                "optimizer_updates": evaluator.training.RANK_UPDATES,
            },
        }
        return checkpoint, config, spec, payload, audit

    def test_formal_source_accepts_only_exact_terminal_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, config, spec, payload, audit = self._source_fixture(root)
            inspection = evaluator.training.PhaseInspection(
                phase="confidence",
                status=evaluator.training.PhaseStatus.TERMINAL,
                checkpoint=checkpoint,
                optimizer_updates=evaluator.training.CONFIDENCE_UPDATES,
                checkpoint_reason="max_train_iters",
            )
            with (
                patch.object(evaluator, "CHECKPOINT", checkpoint),
                patch.object(evaluator, "CONFIG", config),
                patch.object(evaluator.training, "formal_phase_specs", return_value=(None, spec)),
                patch.object(evaluator.training, "classify_checkpoint_payload", return_value=inspection),
                patch.object(evaluator.training, "_audit_checkpoint") as training_audit,
                patch.object(evaluator.SLConfig, "fromfile", return_value=SimpleNamespace()),
                patch.object(evaluator, "build_code_source_closure", return_value={"schema": "fixture"}),
                patch.object(evaluator, "validate_evaluation_checkpoint_payload", return_value=audit) as validate,
            ):
                source, observed = evaluator._resolve_formal_source(
                    evaluator.paper.HashCache(), checkpoint_loader=lambda _: payload
                )

            self.assertEqual(source.checkpoint, checkpoint.resolve())
            self.assertEqual(source.training_phase, "confidence")
            self.assertEqual(observed, audit)
            training_audit.assert_called_once()
            validate.assert_called_once()

    def test_formal_source_rejects_partial_confidence_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, config, spec, payload, _ = self._source_fixture(root)
            inspection = evaluator.training.PhaseInspection(
                phase="confidence",
                status=evaluator.training.PhaseStatus.PARTIAL,
                checkpoint=checkpoint,
                optimizer_updates=100,
                checkpoint_reason="interval",
                detail="only 100 updates",
            )
            with (
                patch.object(evaluator, "CHECKPOINT", checkpoint),
                patch.object(evaluator, "CONFIG", config),
                patch.object(evaluator.training, "formal_phase_specs", return_value=(None, spec)),
                patch.object(evaluator.training, "classify_checkpoint_payload", return_value=inspection),
            ):
                with self.assertRaisesRegex(
                    evaluator.DenseDutyFormalEvaluationError, "not terminal"
                ):
                    evaluator._resolve_formal_source(
                        evaluator.paper.HashCache(), checkpoint_loader=lambda _: payload
                    )

    def _baseline_fixture(self, root: Path):
        checkpoint = root / "baseline.pth"
        config = root / "baseline.py"
        data = root / "baseline_data.jsonl"
        checkpoint.write_bytes(b"baseline-checkpoint")
        config.write_text("baseline = True\n", encoding="ascii")
        data.write_text('{"training": true}\n', encoding="ascii")
        ref_summary = root / "ref_summary.json"
        ref_summary.write_text('{"refcoco": [], "tn": []}\n', encoding="ascii")
        ref_records = {}
        for split in evaluator.paper.REF_SPLITS:
            path = root / f"{split}.jsonl"
            path.write_text(f'{{"split": "{split}"}}\n', encoding="ascii")
            ref_records[split] = _record(path)
        tn = {}
        for label in ("strict2031", "strict1607"):
            summary = root / f"{label}_summary.json"
            records = root / f"{label}.jsonl"
            summary.write_text('{"refcoco": [], "tn": []}\n', encoding="ascii")
            records.write_text(f'{{"label": "{label}"}}\n', encoding="ascii")
            tn[label] = {
                "summary": _record(summary),
                "records": _record(records),
                "run_id": "baseline_run",
            }
        manifest = {
            "schema": "stageb-paper-results-manifest-v1",
            "baseline_experiment": evaluator.headline.BASELINE_ID,
            "expected_train_seeds": [42],
            "experiments": [
                {
                    "id": evaluator.headline.BASELINE_ID,
                    "runs": [
                        {
                            "train_seed": 42,
                            "artifacts": {
                                "checkpoint": _record(checkpoint),
                                "config": _record(config),
                                "data": [_record(data)],
                            },
                            "results": {
                                "ref": {
                                    "summary": _record(ref_summary),
                                    "records": ref_records,
                                    "run_id": "baseline_run",
                                },
                                "tn": tn,
                            },
                        }
                    ],
                }
            ],
            "protocol": {
                "bootstrap": {
                    "confidence": evaluator.BOOTSTRAP_CONFIDENCE,
                    "iterations": evaluator.BOOTSTRAP_ITERATIONS,
                    "seed": evaluator.BOOTSTRAP_SEED,
                },
                "ref_splits": list(evaluator.paper.REF_SPLITS),
                "tn_splits": {
                    label: {
                        "expected_n": specification["rows"],
                        "manifest": {"sha256": specification["sha256"]},
                    }
                    for label, specification in evaluator.paper.STRICT_SPECS.items()
                },
            },
        }
        manifest_path = root / "baseline_manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return manifest_path, checkpoint, config, ref_records

    def test_baseline_contract_rehashes_every_declared_metric_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, checkpoint, config, ref_records = self._baseline_fixture(root)
            fixed = dict(evaluator.headline.FIXED_BASELINE)
            fixed.update(checkpoint=str(checkpoint.resolve()), config=str(config.resolve()))
            with (
                patch.object(evaluator, "BASELINE_MANIFEST", manifest),
                patch.object(evaluator, "BASELINE_MANIFEST_SHA256", _sha256(manifest)),
                patch.object(evaluator, "BASELINE_CHECKPOINT_SHA256", _sha256(checkpoint)),
                patch.object(evaluator.headline, "FIXED_BASELINE", fixed),
            ):
                observed = evaluator._baseline_contract(evaluator.paper.HashCache())
                self.assertEqual(observed["run_id"], "baseline_run")
                tampered_path = Path(ref_records["refcoco_val"]["path"])
                original = tampered_path.read_bytes()
                tampered_path.write_bytes(b"!" + original[1:])
                with self.assertRaisesRegex(
                    evaluator.DenseDutyFormalEvaluationError, "records changed"
                ):
                    evaluator._baseline_contract(evaluator.paper.HashCache())

    def test_commands_are_exact_two_process_full_set_protocol(self):
        source = evaluator.paper.EvaluationSource(
            kind="fixture",
            evaluation_id="fixture",
            config=evaluator.CONFIG.resolve(),
            checkpoint=evaluator.CHECKPOINT.resolve(),
            checkpoint_sha256="a" * 64,
        )
        runtime = evaluator.paper.Runtime(
            python=Path("/fixed/python"),
            data_root=Path("/fixed/data"),
            device="cuda:0",
            batch_size=16,
            num_workers=4,
            amp=True,
            log_every=50,
        )
        commands = evaluator.paper._commands(runtime, source, Path("/fresh/output"))
        self.assertEqual(
            [item["phase_id"] for item in commands],
            ["ref8_strict2031", "strict1607"],
        )
        primary, supplemental = (item["command"] for item in commands)
        for command in (primary, supplemental):
            self.assertIn("--amp", command)
            self.assertEqual(command[command.index("--batch_size") + 1], "16")
            self.assertEqual(command[command.index("--num_workers") + 1], "4")
            self.assertEqual(command[command.index("--seed") + 1], "42")
            self.assertEqual(command[command.index("--topk") + 1], "1")
            self.assertEqual(command[command.index("--max_ref_batches") + 1], "0")
            self.assertEqual(command[command.index("--max_tn_batches") + 1], "0")
        self.assertNotIn("--skip_ref", primary)
        self.assertIn("--skip_ref", supplemental)
        self.assertEqual(
            primary[primary.index("--ref_splits") + 1 : primary.index("--tn_jsonl")],
            list(evaluator.paper.REF_SPLITS),
        )

    def test_dense_summary_provenance_declares_disjoint_score_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            data_root = root / "data"
            config.write_text("stage_b_dense_duty = True\n", encoding="ascii")
            checkpoint.write_bytes(b"checkpoint")
            data_root.mkdir()
            provenance = joint_eval._evaluation_summary_provenance(
                cfg=SimpleNamespace(stage_b_dense_duty=True),
                args=SimpleNamespace(
                    config=str(config), amp=True, device="cuda:0"
                ),
                checkpoint=checkpoint,
                data_root=data_root,
            )
        self.assertEqual(provenance["ref_score_key"], evaluator.REF_SCORE_KEY)
        self.assertEqual(provenance["tn_score_key"], evaluator.TN_SCORE_KEY)
        self.assertEqual(
            provenance["score_ownership"], evaluator.SCORE_OWNERSHIP
        )

    def test_candidate_summary_provenance_rejects_route_drift(self):
        source = {
            "config": str(evaluator.CONFIG.resolve()),
            "config_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
        }
        runtime = {"data_root": str(evaluator.paper.DEFAULT_DATA_ROOT.resolve())}
        row = {
            "config": source["config"],
            "config_sha256": source["config_sha256"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "amp": True,
            "device": "cuda:0",
            "data_root": runtime["data_root"],
            "batch_size": 16,
            "num_workers": 4,
            "ref_score_key": evaluator.REF_SCORE_KEY,
            "tn_score_key": evaluator.TN_SCORE_KEY,
            "score_ownership": evaluator.SCORE_OWNERSHIP,
        }
        primary = {
            "refcoco": [dict(row) for _ in evaluator.paper.REF_SPLITS],
            "tn": [dict(row)],
        }
        supplemental = {"refcoco": [], "tn": [dict(row)]}
        plan = {"source": source, "runtime": runtime}
        evaluator._validate_candidate_summary_provenance(
            plan, primary, supplemental
        )
        supplemental["tn"][0]["tn_score_key"] = evaluator.REF_SCORE_KEY
        with self.assertRaisesRegex(
            evaluator.DenseDutyFormalEvaluationError, "provenance drifted"
        ):
            evaluator._validate_candidate_summary_provenance(
                plan, primary, supplemental
            )

    def test_integer_metric_boundaries_are_exact(self):
        self.assertEqual(
            evaluator._exact_binary_count(
                6993 / 10834, 10834, label="Ref baseline"
            ),
            6993,
        )
        self.assertEqual(
            evaluator._exact_binary_count(
                1040 / 2031, 2031, label="TN baseline"
            ),
            1040,
        )
        with self.assertRaisesRegex(
            evaluator.DenseDutyFormalEvaluationError, "exact integer count"
        ):
            evaluator._exact_binary_count(0.12345, 2031, label="invalid")

    def test_controller_hashes_direct_terminal_and_baseline_dependencies(self):
        dependencies = set(evaluator._direct_controller_dependencies())
        self.assertIn(
            Path(evaluator.training.__file__).resolve(strict=True), dependencies
        )
        self.assertIn(
            Path(evaluator.headline.__file__).resolve(strict=True), dependencies
        )

    def test_paired_report_rejects_identity_and_bootstrap_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            manifest.write_text('{"sample": 1}\n', encoding="ascii")
            baseline.write_text('{"score": 1}\n', encoding="ascii")
            candidate.write_text('{"score": 2}\n', encoding="ascii")
            specification = {
                **evaluator.paper.STRICT_SPECS["strict2031"],
                "rows": 2,
                "sha256": _sha256(manifest),
            }
            report = {
                "validation": {
                    "pass": True,
                    "manifest_n": 2,
                    "valid_n": 2,
                    "invalid_n": 0,
                    "manifest_index_order_match": True,
                    "sample_id_order_match": True,
                    "image_id_order_match": True,
                    "split_order_match": True,
                    "valid_mask_match": True,
                    "baseline_run_ids": ["baseline_run"],
                    "candidate_run_ids": ["candidate_run"],
                    "baseline_manifest_binding_mode": "source_to_derived_v1",
                    "candidate_manifest_binding_mode": "source_to_derived_v1",
                },
                "input_files": {
                    "manifest": _record(manifest),
                    "baseline_records": _record(baseline),
                    "candidate_records": _record(candidate),
                    "identity_is_from_the_same_bytes_used_for_metrics": True,
                },
                "paired_bootstrap": {
                    "unit": "image_cluster",
                    "paired": True,
                    "recomputes_each_model_q05_per_resample": True,
                    "iterations": evaluator.BOOTSTRAP_ITERATIONS,
                    "confidence": evaluator.BOOTSTRAP_CONFIDENCE,
                    "seed": evaluator.BOOTSTRAP_SEED,
                    "valid_records_n": 2,
                    "image_clusters_n": 1,
                },
            }
            with patch.dict(
                evaluator.paper.STRICT_SPECS,
                {"strict2031": specification},
                clear=False,
            ):
                evaluator._validate_paired_report(
                    report,
                    label="strict2031",
                    manifest=manifest,
                    baseline_records=baseline,
                    candidate_records=candidate,
                    baseline_run_id="baseline_run",
                    candidate_run_id="candidate_run",
                    cache=evaluator.paper.HashCache(),
                )
                report["validation"]["candidate_run_ids"] = ["wrong"]
                with self.assertRaisesRegex(
                    evaluator.DenseDutyFormalEvaluationError,
                    "paired validation drifted",
                ):
                    evaluator._validate_paired_report(
                        report,
                        label="strict2031",
                        manifest=manifest,
                        baseline_records=baseline,
                        candidate_records=candidate,
                        baseline_run_id="baseline_run",
                        candidate_run_id="candidate_run",
                        cache=evaluator.paper.HashCache(),
                    )
                report["validation"]["candidate_run_ids"] = ["candidate_run"]
                report["paired_bootstrap"]["seed"] += 1
                with self.assertRaisesRegex(
                    evaluator.DenseDutyFormalEvaluationError,
                    "bootstrap contract drifted",
                ):
                    evaluator._validate_paired_report(
                        report,
                        label="strict2031",
                        manifest=manifest,
                        baseline_records=baseline,
                        candidate_records=candidate,
                        baseline_run_id="baseline_run",
                        candidate_run_id="candidate_run",
                        cache=evaluator.paper.HashCache(),
                    )

    def test_overall_target_requires_every_ref_and_both_fpr_wins(self):
        ref = {split: {"strict_win": True} for split in evaluator.paper.REF_SPLITS}
        fpr = {
            "strict2031": {"strict_win": True},
            "strict1607": {"strict_win": True},
        }
        decision = evaluator._comparison_decision(ref, fpr)
        self.assertTrue(decision["overall_target_met"])
        ref["refcocog_test"]["strict_win"] = False
        decision = evaluator._comparison_decision(ref, fpr)
        self.assertFalse(decision["overall_target_met"])
        self.assertFalse(decision["all_ref8_splits_strictly_higher"])


if __name__ == "__main__":
    unittest.main()
