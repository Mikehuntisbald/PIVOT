from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import aggregate_stageb_table_d_formal_matrix as aggregate


class StageBTableDFormalMatrixAggregateTest(unittest.TestCase):
    def test_aggregate_keeps_clean_s2f_and_rank_comparisons_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue_dir = Path(temporary)
            spec_path = queue_dir / aggregate.queue_runner.AGGREGATION_SPEC_NAME
            spec_path.write_text("{}\n")
            plan = {
                "queue_id": "queue-id",
                "training_queue": {"queue_id": "training-id"},
                "aggregation_sources": [],
            }
            queue_payload = {
                "status": "completed",
                "plan": plan,
                "plan_sha256": "a" * 64,
            }
            verification = {
                "schema": aggregate.queue_runner.VERIFICATION_SCHEMA,
                "status": "passed",
                "final_verification": {
                    "schema": aggregate.queue_runner.FINAL_VERIFICATION_SCHEMA,
                },
                "evaluation_scope_plan": {"path": "scope.json"},
            }
            spec = {"schema": aggregate.queue_runner.SPEC_SCHEMA}

            def run(seed: int, prefix: str):
                return SimpleNamespace(
                    seed=seed,
                    checkpoint_sha256=f"{prefix}-{seed}",
                )

            finals = {
                row: {
                    seed: run(seed, row)
                    for seed in aggregate.queue_runner.training_queue.SEEDS
                }
                for row in aggregate.queue_runner.training_queue.ROWS
            }
            ranks = {
                seed: run(seed, "S3-rank")
                for seed in aggregate.queue_runner.training_queue.SEEDS
            }

            comparison_calls = []

            def comparison(**kwargs):
                comparison_calls.append(kwargs)
                return {
                    "reference": kwargs["reference_id"],
                    "candidate": kwargs["candidate_id"],
                }

            with patch.object(
                aggregate.queue_runner, "load_queue", return_value=queue_payload
            ), patch.object(
                aggregate.queue_runner, "verify_queue", return_value=verification
            ), patch.object(
                aggregate, "_verify_aggregation_sources", return_value=[]
            ), patch.object(
                aggregate.queue_runner, "_read_json", return_value=spec
            ), patch.object(
                aggregate.queue_runner, "_aggregation_spec_payload", return_value=spec
            ), patch.object(
                aggregate, "_load_all", return_value=(finals, ranks)
            ), patch.object(
                aggregate.matrix,
                "_aggregate_experiment",
                return_value={"per_seed": {}, "aggregate": {}},
            ), patch.object(
                aggregate.matrix, "_comparison", side_effect=comparison
            ):
                report = aggregate.aggregate(queue_dir)
            self.assertEqual(report["status"], "validated_matrix_validation_only")
            self.assertEqual(
                report["experiments"]["S2F"]["comparison_class"],
                "full_v19_objective_control",
            )
            self.assertEqual(
                report["experiments"]["S3_rank"]["comparison_class"],
                "diagnostic_rank_checkpoint",
            )
            clean = report["comparisons"]["clean_ownership_vs_S0"]
            self.assertEqual(set(clean), {"S1", "S2", "S3"})
            self.assertEqual(
                report["comparisons"]["S2F_minus_S2_full_objective_control"],
                {"reference": "S2", "candidate": "S2F"},
            )
            self.assertEqual(
                report["comparisons"]["S3_confidence_minus_rank_diagnostic"],
                {"reference": "S3_rank", "candidate": "S3"},
            )
            self.assertFalse(
                report["inputs"]["final_diagnostics"]["pooled_into_matrix_results"]
            )
            self.assertEqual(
                report["inputs"]["evaluation_queue"]["final_verification"],
                verification["final_verification"],
            )
            self.assertEqual(len(comparison_calls), 5)
            self.assertTrue(
                all(
                    call["iterations"] == aggregate.FORMAL_BOOTSTRAP_ITERATIONS
                    and call["confidence"] == aggregate.FORMAL_BOOTSTRAP_CONFIDENCE
                    and call["bootstrap_seed"] == aggregate.FORMAL_BOOTSTRAP_SEED
                    for call in comparison_calls
                )
            )

    def test_bootstrap_contract_is_exact(self):
        with self.assertRaisesRegex(
            aggregate.TableDFormalAggregationError, "5000/0.95/20260719"
        ):
            aggregate.aggregate(Path("/not-read"), bootstrap_iterations=4999)

    def test_existing_final_diagnostics_are_replayed_but_not_pooled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "diagnostics-input.json"
            report_path = root / "diagnostics-report.json"
            manifest.write_text("{}\n")
            payload = {
                "schema": aggregate.diagnostics.REPORT_SCHEMA,
                "status": "passed",
                "created_at_utc": "old",
                "expected_train_seeds": list(aggregate.diagnostics.EXPECTED_SEEDS),
                "input_manifest": {"path": str(manifest)},
                "gradient_conflict": {},
                "s3_rank_to_confidence": {},
                "inputs": {},
            }
            report_path.write_text(json.dumps(payload))
            replay = {**payload, "created_at_utc": "new"}
            with patch.object(
                aggregate.diagnostics, "aggregate", return_value=replay
            ):
                binding = aggregate._bind_final_diagnostics(report_path)
            self.assertEqual(binding["status"], "bound_and_replayed")
            self.assertTrue(binding["separate_from_matrix_metrics"])
            self.assertFalse(binding["pooled_into_matrix_results"])


if __name__ == "__main__":
    unittest.main()
