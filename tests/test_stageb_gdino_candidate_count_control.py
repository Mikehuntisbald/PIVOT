import ast
import inspect
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools import eval_text_groundingdino_refcoco_tn as gdino_eval


class CandidateCountControlTest(unittest.TestCase):
    def test_main_evaluator_calls_match_function_signatures(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(gdino_eval.main)))
        signatures = {
            "evaluate_refcoco_dataset": set(
                inspect.signature(gdino_eval.evaluate_refcoco_dataset).parameters
            ),
            "evaluate_tn_dataset": set(
                inspect.signature(gdino_eval.evaluate_tn_dataset).parameters
            ),
        }
        observed = {name: 0 for name in signatures}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name not in signatures:
                continue
            observed[name] += 1
            keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
            self.assertEqual(
                keywords - signatures[name],
                set(),
                msg=f"{name} received unsupported keyword arguments",
            )
        self.assertEqual(observed, {"evaluate_refcoco_dataset": 1, "evaluate_tn_dataset": 1})

    def test_subset_is_deterministic_score_independent_and_bounded(self):
        first = gdino_eval._score_independent_query_subset(
            query_count=900,
            subset_count=50,
            seed=17,
            repeat=2,
            sample_id="sample-a",
        )
        second = gdino_eval._score_independent_query_subset(
            query_count=900,
            subset_count=50,
            seed=17,
            repeat=2,
            sample_id="sample-a",
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.size, 50)
        self.assertEqual(np.unique(first).size, 50)
        self.assertGreaterEqual(int(first.min()), 0)
        self.assertLess(int(first.max()), 900)

    def test_summary_is_explicitly_diagnostic_and_reports_variation(self):
        summary = gdino_eval._summarize_candidate_count_control(
            pos_by_repeat=[[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]],
            neg_by_repeat=[[0.8, 0.2, 0.1], [0.7, 0.6, 0.1]],
            query_count=900,
            subset_count=50,
            seed=17,
            threshold_tprs=[0.9, 0.95],
            score_thresholds=[0.5],
        )
        self.assertTrue(summary["diagnostic_only"])
        self.assertFalse(summary["formal_baseline_eligible"])
        self.assertEqual(summary["query_count"], 900)
        self.assertEqual(summary["subset_count"], 50)
        self.assertEqual(len(summary["repeats"]), 2)
        self.assertIn("fpr95tpr_mean", summary["aggregate"])
        self.assertIn("roc_auc_mean", summary["aggregate"])

    def test_auc_handles_ties(self):
        self.assertEqual(
            gdino_eval._roc_auc(np.array([1.0]), np.array([0.0])), 1.0
        )
        self.assertEqual(
            gdino_eval._roc_auc(np.array([0.0]), np.array([1.0])), 0.0
        )
        self.assertEqual(
            gdino_eval._roc_auc(np.array([0.5]), np.array([0.5])), 0.5
        )

    def test_direct_prebuilt_cli_contract_forbids_partial_or_reused_output(self):
        args = SimpleNamespace(
            direct_prebuilt_tn=True,
            screen_calibration_manifest=False,
            skip_tn=False,
            skip_ref=True,
            tn_jsonl="surface.jsonl",
            direct_prebuilt_tn_binding="surface.binding.json",
            holdout_level="none",
            max_tn_batches=0,
            no_per_example_records=False,
            candidate_count_control=0,
            ckpts=["checkpoint.pth"],
            output_dir="fresh-output",
        )
        with patch.object(Path, "exists", return_value=False):
            gdino_eval._validate_direct_prebuilt_tn_args(args)
        args.max_tn_batches = 1
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaisesRegex(ValueError, "partial TN batches"):
                gdino_eval._validate_direct_prebuilt_tn_args(args)

    def test_direct_prebuilt_rows_require_unique_proposal_covered_surface(self):
        row = {
            "matched_eval_surface_schema": (
                "pivot.stageb.table_b_matched_eval_surface_row/v1"
            ),
            "tn_scope": "proposal_covered_verified",
            "global_tn_verified": False,
            "proposal_covered_verified": True,
            "tn_eval_split": "matched_calibration",
            "table_b_id": "D3m",
            "sample_id": "sample-1",
            "matched_pair_id": "pair-1",
        }
        self.assertEqual(
            gdino_eval._validate_direct_prebuilt_tn_rows(
                [row], declared_scope="proposal_covered_verified"
            ),
            "proposal_covered_verified",
        )
        upgraded = dict(row, global_tn_verified=True)
        with self.assertRaisesRegex(ValueError, "scope"):
            gdino_eval._validate_direct_prebuilt_tn_rows(
                [upgraded], declared_scope="proposal_covered_verified"
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            gdino_eval._validate_direct_prebuilt_tn_rows(
                [row, dict(row)], declared_scope="proposal_covered_verified"
            )


if __name__ == "__main__":
    unittest.main()
