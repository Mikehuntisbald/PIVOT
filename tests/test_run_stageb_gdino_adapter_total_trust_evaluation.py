import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_stageb_gdino_adapter_total_trust_evaluation as evaluator
from tools import build_stagea_b58_r100_receipt as stagea_receipt


def _record(path: Path, **extra):
    payload = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    payload.update(extra)
    return payload


class TotalTrustHistoricalB58EvaluationTest(unittest.TestCase):
    def test_lineage_verification_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            audit = root / "checkpoint.audit.json"
            output = root / "checkpoint.evaluation.json"
            checkpoint.write_bytes(b"checkpoint")
            audit.write_text("{}\n", encoding="ascii")
            output.write_text("{}\n", encoding="ascii")
            with patch.object(evaluator.subprocess, "run") as run:
                with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                    evaluator._run_candidate_lineage_verification(
                        checkpoint=checkpoint,
                        audit=audit,
                        output=output,
                    )
            run.assert_not_called()

    def test_lineage_binds_historical_b58_and_frozen_r100_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "candidate.pth"
            baseline = root / "b58.pth"
            config = root / "config.py"
            datasets = root / "datasets.json"
            checkpoint.write_bytes(b"candidate")
            baseline.write_bytes(b"historical-b58")
            config.write_text("stage_b_gdino_score_adapter = True\n", encoding="ascii")
            datasets.write_text("{}\n", encoding="ascii")

            rank_sha256 = "7" * 64
            receipt = root / "legacy_receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema": evaluator.LEGACY_RECEIPT_SCHEMA,
                        "checkpoints": {
                            "b58": {"file": _record(baseline)},
                            "rank_r100": {
                                "model": {"rank_tensor_sha256": rank_sha256}
                            },
                        },
                    }
                )
                + "\n",
                encoding="ascii",
            )
            preflight = root / "probe_preflight.json"
            preflight.write_text(
                json.dumps(
                    {
                        "schema": evaluator.TOTAL_TRUST_SCHEMA,
                        "kind": "phase_preflight",
                        "phase": "confidence",
                        "launch": {
                            "initialization": (
                                "sealed_legacy_b58_r100_to_total_trust_"
                                "confidence_pretrain_model_path"
                            )
                        },
                        "initial_audit": _record(receipt),
                    }
                )
                + "\n",
                encoding="ascii",
            )
            audit = root / "candidate.audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema": evaluator.TOTAL_TRUST_SCHEMA,
                        "kind": "milestone_checkpoint",
                        "phase": "confidence",
                        "iteration": 100,
                        "preflight": _record(preflight),
                    }
                )
                + "\n",
                encoding="ascii",
            )
            payload = {
                "schema": evaluator.TOTAL_TRUST_SCHEMA,
                "kind": "evaluation_checkpoint_verified",
                "phase": "confidence",
                "iteration": 100,
                "train_mode": "confidence_only",
                "tn_scope": "benchmark_dataft_alltn",
                "checkpoint": _record(checkpoint, rank_sha256=rank_sha256),
                "audit": _record(audit),
                "config": _record(config),
                "datasets": _record(datasets),
            }
            with patch.object(
                evaluator.dense,
                "_baseline_contract",
                return_value={"checkpoint": baseline.resolve()},
            ):
                observed = evaluator._validate_lineage_payload(
                    payload,
                    checkpoint=checkpoint.resolve(),
                    audit=audit.resolve(),
                    cache=evaluator.paper.HashCache(),
                )
                self.assertTrue(observed["rank_branch_unchanged_from_r100"])
                self.assertEqual(observed["r100_rank_sha256"], rank_sha256)

                payload["checkpoint"]["rank_sha256"] = "8" * 64
                with self.assertRaisesRegex(
                    evaluator.TotalTrustEvaluationError,
                    "not bitwise-identical",
                ):
                    evaluator._validate_lineage_payload(
                        payload,
                        checkpoint=checkpoint.resolve(),
                        audit=audit.resolve(),
                        cache=evaluator.paper.HashCache(),
                    )

    def test_lineage_accepts_sealed_stagea_r100_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "candidate.pth"
            baseline = root / "b58.pth"
            stagea = root / "stagea.pth"
            rank = root / "r100.pth"
            config = root / "config.py"
            datasets = root / "datasets.json"
            for path, contents in (
                (checkpoint, b"candidate"),
                (baseline, b"historical-b58"),
                (stagea, b"stagea"),
                (rank, b"r100"),
            ):
                path.write_bytes(contents)
            config.write_text("stage_b_gdino_score_adapter = True\n", encoding="ascii")
            datasets.write_text("{}\n", encoding="ascii")
            rank_sha256 = "7" * 64
            receipt_payload = {
                "schema": evaluator.STAGEA_R100_RECEIPT_SCHEMA,
                "stagea": {
                    "checkpoint": _record(stagea),
                    "b58_source": _record(baseline),
                },
                "rank_r100": {
                    "checkpoint": _record(rank),
                    "rank_tensor_sha256": rank_sha256,
                },
            }
            receipt = root / "stagea_r100_receipt.json"
            receipt.write_text(json.dumps(receipt_payload) + "\n", encoding="ascii")
            preflight = root / "probe_preflight.json"
            preflight.write_text(
                json.dumps(
                    {
                        "schema": evaluator.TOTAL_TRUST_SCHEMA,
                        "kind": "phase_preflight",
                        "phase": "confidence",
                        "launch": {"initialization": evaluator.STAGEA_R100_INITIALIZATION},
                        "initial_audit": _record(receipt),
                    }
                )
                + "\n",
                encoding="ascii",
            )
            audit = root / "candidate.audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema": evaluator.TOTAL_TRUST_SCHEMA,
                        "kind": "milestone_checkpoint",
                        "phase": "confidence",
                        "iteration": 100,
                        "preflight": _record(preflight),
                    }
                )
                + "\n",
                encoding="ascii",
            )
            payload = {
                "schema": evaluator.TOTAL_TRUST_SCHEMA,
                "kind": "evaluation_checkpoint_verified",
                "phase": "confidence",
                "iteration": 100,
                "train_mode": "confidence_only",
                "tn_scope": "benchmark_dataft_alltn",
                "checkpoint": _record(checkpoint, rank_sha256=rank_sha256),
                "audit": _record(audit),
                "config": _record(config),
                "datasets": _record(datasets),
            }
            with (
                patch.object(
                    evaluator.dense,
                    "_baseline_contract",
                    return_value={"checkpoint": baseline.resolve()},
                ),
                patch.object(stagea_receipt, "verify_receipt", return_value=receipt_payload),
            ):
                observed = evaluator._validate_lineage_payload(
                    payload,
                    checkpoint=checkpoint.resolve(),
                    audit=audit.resolve(),
                    cache=evaluator.paper.HashCache(),
                )
            self.assertEqual(
                observed["lineage_root_schema"], evaluator.STAGEA_R100_RECEIPT_SCHEMA
            )
            self.assertEqual(observed["root_stagea"]["checkpoint"], _record(stagea))
            self.assertTrue(observed["rank_branch_unchanged_from_r100"])

    def test_score_routes_use_rank_for_ref_and_confidence_for_tn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            data_root = root / "data"
            data_root.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            config.write_text(
                "\n".join(
                    (
                        "stage_b_gdino_score_adapter = True",
                        "stage_b_gdino_adapter_train_mode = 'confidence_only'",
                        "stage_b_gdino_tn_scope = 'benchmark_dataft_alltn'",
                        "stage_b_gdino_confidence_objective = "
                        "'detached_recent_q05_total_trust'",
                        "stage_b_gdino_rank_weight = 0.0",
                        "stage_b_gdino_confidence_weight = 1.0",
                    )
                )
                + "\n",
                encoding="ascii",
            )
            runtime = evaluator.paper.Runtime(
                python=Path("/fixed/python"),
                data_root=data_root,
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                amp=True,
                log_every=50,
            )
            routes = evaluator._validate_score_routes(
                config=config,
                checkpoint=checkpoint,
                runtime=runtime,
            )
        self.assertEqual(routes["ref_score_key"], evaluator.REF_SCORE_KEY)
        self.assertEqual(routes["tn_score_key"], evaluator.TN_SCORE_KEY)
        self.assertEqual(routes["score_ownership"], evaluator.SCORE_OWNERSHIP)
        self.assertTrue(routes["summary_route_fields_currently_emitted"])

    def test_summary_provenance_requires_exact_score_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            data_root = root / "data"
            config.write_text("# fixture\n", encoding="ascii")
            data_root.mkdir()
            source = {
                "config": str(config.resolve()),
                "config_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
            }
            runtime = {"data_root": str(data_root.resolve())}
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
            del supplemental["tn"][0]["tn_score_key"]
            with self.assertRaisesRegex(
                evaluator.TotalTrustEvaluationError, "provenance drifted"
            ):
                evaluator._validate_candidate_summary_provenance(
                    plan, primary, supplemental
                )

    def test_strict_fpr_win_is_an_exact_false_accept_boundary(self):
        for label, total, baseline_false_accepts in (
            ("strict2031", 2031, 1040),
            ("strict1607", 1607, 801),
        ):
            with self.subTest(label=label):
                baseline_fpr = baseline_false_accepts / total
                candidate_fpr = (baseline_false_accepts - 1) / total
                report = {
                    "global": {
                        "baseline": {
                            "fpr95": {"fpr": baseline_fpr, "threshold": 0.2}
                        },
                        "candidate": {
                            "fpr95": {"fpr": candidate_fpr, "threshold": 0.3}
                        },
                    },
                    "paired_bootstrap": {"paired": True},
                }
                result = evaluator._strict_fpr_result(
                    label=label,
                    report=report,
                    baseline_summary_fpr95=baseline_fpr,
                    candidate_summary_fpr95=candidate_fpr,
                    baseline_summary_q05=0.2,
                    candidate_summary_q05=0.3,
                )
                self.assertTrue(result["strict_win"])
                self.assertEqual(
                    result["maximum_candidate_false_accepts_for_strict_win"],
                    baseline_false_accepts - 1,
                )

                report["global"]["candidate"]["fpr95"]["fpr"] = baseline_fpr
                tied = evaluator._strict_fpr_result(
                    label=label,
                    report=report,
                    baseline_summary_fpr95=baseline_fpr,
                    candidate_summary_fpr95=baseline_fpr,
                    baseline_summary_q05=0.2,
                    candidate_summary_q05=0.3,
                )
                self.assertFalse(tied["strict_win"])

    def test_existing_evaluation_output_root_fails_before_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            audit = root / "checkpoint.audit.json"
            output = root / "existing"
            checkpoint.write_bytes(b"checkpoint")
            audit.write_text("{}\n", encoding="ascii")
            output.mkdir()
            with patch.object(evaluator, "_fixed_runtime") as runtime:
                with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                    evaluator.build_plan(
                        checkpoint=checkpoint,
                        audit=audit,
                        output_dir=output,
                    )
            runtime.assert_not_called()

    def test_appending_lineage_preserves_sealed_input_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = root / "sealed.txt"
            lineage = root / "lineage.json"
            sealed.write_text("before\n", encoding="ascii")
            lineage.write_text("{}\n", encoding="ascii")
            sealed_record = evaluator.paper._file_record(
                sealed,
                evaluator.paper.HashCache(),
                roles=("sealed",),
            )
            plan = {
                "inputs": {
                    "algorithm": "sha256",
                    "records": [dict(sealed_record)],
                }
            }
            evaluator._append_fresh_lineage_input(plan, lineage)
            observed = next(
                record
                for record in plan["inputs"]["records"]
                if record["path"] == str(sealed.resolve())
            )
            self.assertEqual(observed, sealed_record)
            sealed.write_text("after\n", encoding="ascii")
            with self.assertRaises(evaluator.paper.PaperEvaluationError):
                evaluator.paper._verify_input_identities(plan)


if __name__ == "__main__":
    unittest.main()
