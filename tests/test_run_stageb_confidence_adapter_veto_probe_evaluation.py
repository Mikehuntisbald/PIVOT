import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_stageb_confidence_adapter_veto_probe_evaluation as controller


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProbeEvaluationControllerTest(unittest.TestCase):
    def test_formal_config_changes_only_registered_promotion_fields(self):
        evidence = controller._validate_formal_config_promotion()
        self.assertTrue(evidence["all_other_config_values_equal"])
        self.assertEqual(
            set(evidence["allowed_overrides"]),
            set(controller.FORMAL_PROMOTION_OVERRIDES),
        )

    def test_command_is_the_exact_tn_only_strict1607_contract(self):
        self.assertEqual(
            controller.build_command(),
            [
                str(controller.FIXED_PYTHON),
                str(controller.EVALUATOR),
                "--config",
                str(controller.CONFIG),
                "--ckpts",
                str(controller.CHECKPOINT),
                "--output_dir",
                str(controller.OUTPUT),
                "--data_root",
                str(controller.DATA_ROOT),
                "--device",
                "cuda:0",
                "--batch_size",
                "16",
                "--num_workers",
                "4",
                "--seed",
                "42",
                "--amp",
                "--skip_ref",
                "--tn_jsonl",
                str(controller.TN_MANIFEST),
                "--tn_splits",
                "refcocop_val",
                "refcocog_umd_val",
                "--partial_dense_duty_confidence_diagnostic",
                "--topk",
                "1",
                "--threshold_tprs",
                "0.75",
                "0.9",
                "0.95",
                "--score_thresholds",
                "0.5",
                "--max_ref_batches",
                "0",
                "--max_tn_batches",
                "0",
                "--log_every",
                "50",
            ],
        )

    def test_nonterminal_and_unhealthy_probe_are_rejected(self):
        rank_sha256 = "a" * 64
        for state in (
            {
                "status": "partial",
                "action": "resume",
                "updates": 299,
                "rank_sha256": rank_sha256,
            },
            {
                "status": "terminal",
                "action": "complete",
                "updates": 299,
                "rank_sha256": rank_sha256,
            },
        ):
            with self.subTest(state=state):
                with self.assertRaises(controller.ProbeEvaluationError):
                    controller._validate_training_state(state)

        unhealthy = {
            "schema": controller.HEALTH_SCHEMA,
            "decision": "unhealthy_do_not_run_strict1607",
            "failed_checks": ["u300_operating_gap_improved"],
        }
        with self.assertRaisesRegex(
            controller.ProbeEvaluationError,
            "did not admit the strict1607 diagnostic",
        ):
            controller._validate_health_report(
                unhealthy,
                checkpoint_record={
                    "path": "/missing",
                    "size_bytes": 1,
                    "sha256": "b" * 64,
                },
            )

    def _fixture(self, root: Path, false_accepts: int):
        root.mkdir(parents=True, exist_ok=True)
        output = root / "output"
        records_dir = output / "per_example_records"
        records_dir.mkdir(parents=True)
        python = root / "python"
        evaluator = root / "evaluator.py"
        config = root / "config.py"
        checkpoint = root / "checkpoint_iter.pth"
        data_root = root / "data"
        source_manifest = root / "strict1607.jsonl"
        data_root.mkdir()
        python.write_text("#!/bin/sh\n", encoding="ascii")
        python.chmod(0o755)
        evaluator.write_text("# evaluator\n", encoding="ascii")
        config.write_text("probe = True\n", encoding="ascii")
        checkpoint.write_bytes(b"terminal-u300")
        source_manifest.write_text('{"source": true}\n', encoding="ascii")
        source_sha = _sha256(source_manifest)
        derived_sha = "d" * 64
        records_path = records_dir / "u300__tn_global.records.jsonl"
        rows = []
        for index in range(controller.EXPECTED_PAIRS):
            rows.append(
                {
                    "schema": controller.RECORD_SCHEMA,
                    "task": "tn",
                    "valid": True,
                    "run_id": "u300",
                    "manifest_key": "tn_global",
                    "manifest_sha256": derived_sha,
                    "source_manifest_sha256": source_sha,
                    "manifest_n": controller.EXPECTED_PAIRS,
                    "source_manifest_n": controller.EXPECTED_PAIRS,
                    "manifest_index": index,
                    "sample_id": f"sample-{index}",
                    "split": (
                        "refcocop_val" if index % 2 == 0 else "refcocog_umd_val"
                    ),
                    "pos_score": 0.9,
                    "neg_score": 0.9 if index < false_accepts else 0.1,
                }
            )
        records_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        inputs = {
            "python": controller._file_record(python, label="python"),
            "evaluator": controller._file_record(evaluator, label="evaluator"),
            "config": controller._file_record(config, label="config"),
            "checkpoint": controller._file_record(checkpoint, label="checkpoint"),
            "tn_manifest": controller._file_record(source_manifest, label="manifest"),
        }
        row = {
            "run_id": "u300",
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "confidence_evaluated": True,
            "training_phase": "confidence",
            "terminal_checkpoint": True,
            "optimizer_updates": 300,
            "expected_optimizer_updates": 300,
            "remaining_optimizer_updates": 0,
            "checkpoint_reason": "max_train_iters",
            "amp": True,
            "device": "cuda:0",
            "batch_size": 16,
            "num_workers": 4,
            "seed": 42,
            "max_batches": 0,
            **controller.EXPECTED_SCORE_ROUTE,
            "config": str(config.resolve()),
            "config_sha256": inputs["config"]["sha256"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": inputs["checkpoint"]["sha256"],
            "data_root": str(data_root.resolve()),
            "num_pairs": controller.EXPECTED_PAIRS,
            "manifest_n": controller.EXPECTED_PAIRS,
            "source_manifest_n": controller.EXPECTED_PAIRS,
            "invalid_positive_pairs": 0,
            "invalid_negative_pairs": 0,
            "invalid_records": 0,
            "source_manifest_sha256": source_sha,
            "manifest_sha256": derived_sha,
            "source_manifest_path": str(source_manifest.resolve()),
            "records_jsonl": str(records_path.resolve()),
            "threshold_at_95tpr": 0.9,
            "fpr95tpr": false_accepts / controller.EXPECTED_PAIRS,
            "tn_fpr": false_accepts / controller.EXPECTED_PAIRS,
            "actual_tpr_at_95tpr": 1.0,
        }
        summary_path = output / "summary.json"
        summary_path.write_text(
            json.dumps({"refcoco": [], "tn": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        constants = {
            "FIXED_PYTHON": python,
            "EVALUATOR": evaluator,
            "CONFIG": config,
            "CHECKPOINT": checkpoint,
            "DATA_ROOT": data_root,
            "TN_MANIFEST": source_manifest,
            "TN_MANIFEST_SHA256": source_sha,
            "DERIVED_TN_MANIFEST_SHA256": derived_sha,
            "OUTPUT": output,
        }
        return constants, inputs, row, summary_path

    def test_exact_false_accept_boundary_800_passes_and_801_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for false_accepts, expected in (
                (800, "admit_to_formal_training"),
                (801, "valid_nonwin_do_not_enter_formal"),
            ):
                with self.subTest(false_accepts=false_accepts):
                    constants, inputs, _, summary = self._fixture(
                        root / str(false_accepts), false_accepts
                    )
                    with patch.multiple(controller, **constants):
                        result = controller.postflight(
                            {"inputs": inputs}, summary_path=summary
                        )
                    self.assertEqual(result["decision"], expected)
                    self.assertEqual(
                        result["strict1607"]["false_accepts"], false_accepts
                    )
                    self.assertTrue(result["diagnostic_only"])
                    self.assertFalse(result["formal_gate_eligible"])
                    self.assertEqual(result["records"]["rows"], 1607)
                    self.assertEqual(len(result["records"]["sha256"]), 64)

    def test_off_grid_rate_and_provenance_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            constants, inputs, row, summary = self._fixture(root, 800)
            with patch.multiple(controller, **constants):
                for field, bad_value in (
                    ("diagnostic_only", False),
                    ("formal_gate_eligible", True),
                    ("terminal_checkpoint", False),
                    ("optimizer_updates", 299),
                    ("remaining_optimizer_updates", 1),
                    ("tn_score_key", "wrong_route"),
                ):
                    with self.subTest(field=field):
                        changed = dict(row)
                        changed[field] = bad_value
                        with self.assertRaises(controller.ProbeEvaluationError):
                            controller._validate_summary_provenance(
                                changed, inputs=inputs
                            )

                payload = json.loads(summary.read_text(encoding="utf-8"))
                payload["tn"][0]["fpr95tpr"] += 1e-7
                summary.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    controller.ProbeEvaluationError, "exact integer count"
                ):
                    controller.postflight({"inputs": inputs}, summary_path=summary)

    def test_non_order_statistic_threshold_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            constants, inputs, _, summary = self._fixture(root, 800)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["tn"][0]["threshold_at_95tpr"] = 0.5
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with patch.multiple(controller, **constants):
                with self.assertRaisesRegex(
                    controller.ProbeEvaluationError, "exact score>= q05"
                ):
                    controller.postflight({"inputs": inputs}, summary_path=summary)

    def test_blocked_preflight_does_not_publish_terminal_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            with (
                patch.object(controller, "REPORT", report),
                patch.object(
                    controller,
                    "preflight",
                    side_effect=controller.ProbeEvaluationError("probe is fresh"),
                ),
            ):
                self.assertEqual(controller.run(), 2)
            self.assertFalse(report.exists())

    def test_formal_admission_replays_health_and_strict1607_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            constants, inputs, _, summary = self._fixture(root, 800)
            console = root / "console.log"
            console.write_text("complete\n", encoding="ascii")
            health_log = root / "health.log"
            health_log.write_text('{"u": 222}\n', encoding="ascii")
            state = {
                "status": "terminal",
                "action": "complete",
                "updates": 300,
                "rank_sha256": "a" * 64,
            }
            health_report = {
                "schema": controller.HEALTH_SCHEMA,
                "decision": "healthy_for_strict1607_diagnostic",
                "failed_checks": [],
                "checks": {"healthy": {"passed": True}},
                "candidate": {
                    "controller": dict(state),
                    "checkpoint": inputs["checkpoint"],
                    "log": controller._file_record(
                        health_log, label="health log"
                    ),
                },
            }
            constants.update(
                {
                    "LOG": console,
                    "REPORT": root / "admission.json",
                }
            )
            with (
                patch.multiple(controller, **constants),
                patch.object(controller.training, "inspect", return_value=state),
                patch.object(
                    controller,
                    "_validate_formal_config_promotion",
                    return_value={"schema": "config-promotion-fixture"},
                ),
            ):
                postflight = controller.postflight(
                    {"inputs": inputs}, summary_path=summary
                )
                launch = {
                    "training": dict(state),
                    "health": health_report,
                    "inputs": inputs,
                    "command": controller.build_command(),
                    "diagnostic_only": True,
                    "formal_gate_eligible": False,
                }
                report = {
                    "schema": controller.SCHEMA,
                    "status": "completed",
                    "decision": "admit_to_formal_training",
                    "diagnostic_only": True,
                    "formal_gate_eligible": False,
                    "preflight": launch,
                    "postflight": postflight,
                    "console_log": controller._file_record(
                        console, label="console"
                    ),
                }
                constants["REPORT"].write_text(
                    json.dumps(report, sort_keys=True), encoding="utf-8"
                )
                admission = controller.verify_admission_report(
                    health_audit=lambda: health_report
                )
                self.assertEqual(admission["status"], "verified")
                self.assertEqual(
                    admission["strict1607"]["false_accepts"], 800
                )

                report["postflight"]["strict1607"]["false_accepts"] = 799
                constants["REPORT"].write_text(
                    json.dumps(report, sort_keys=True), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    controller.ProbeEvaluationError,
                    "differs from the current record replay",
                ):
                    controller.verify_admission_report(
                        health_audit=lambda: health_report
                    )


if __name__ == "__main__":
    unittest.main()
