import json
import tempfile
import types
import unittest
from pathlib import Path

import torch

from tools import eval_refcoco_stageb as ref_eval
from tools.stageb_eval_records import EvalManifest


def _diagnostic_cfg(**overrides):
    values = {
        "stage_b_v11_fixed_text": True,
        "stage_b_gdino_score_adapter": False,
        "stage_b_v15_decoupled_confidence": True,
        "stage_b_v15_patch_rank_fusion": True,
        "stage_b_v15_patch_rank_weight": 1.0,
        "stage_b_v11_candidate_topk": 50,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _diagnostic_outputs():
    pred_boxes = torch.tensor(
        [
            [
                [0.15, 0.15, 0.10, 0.10],
                [0.50, 0.50, 0.20, 0.20],
                [0.85, 0.85, 0.10, 0.10],
            ]
        ],
        dtype=torch.float32,
    )
    candidate_idx = torch.tensor([[0, 1]], dtype=torch.int64)
    patch_logits = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
    text_logits = torch.tensor([[[0.0], [3.0]]], dtype=torch.float32)
    fused_logits = text_logits + patch_logits.unsqueeze(-1)
    expression_valid = torch.tensor([[True]], dtype=torch.bool)

    dense_shape = (1, 3, 1)
    scatter_idx = candidate_idx.unsqueeze(-1)
    dense_logits = torch.full(
        dense_shape,
        torch.finfo(fused_logits.dtype).min,
        dtype=fused_logits.dtype,
    )
    dense_logits.scatter_(1, scatter_idx, fused_logits)
    dense_score = torch.zeros(dense_shape, dtype=fused_logits.dtype)
    dense_score.scatter_(1, scatter_idx, fused_logits.sigmoid())
    candidate_mask = torch.zeros(dense_shape, dtype=torch.bool)
    candidate_mask.scatter_(1, scatter_idx, True)
    return {
        "pred_boxes": pred_boxes,
        "stage_b_v11_final_phrase_logits": fused_logits,
        "stage_b_v15_candidate_patch_logits": patch_logits,
        "stage_b_v11_candidate_idx": candidate_idx,
        "stage_b_v11_expression_valid_mask": expression_valid,
        "stage_b_v11_candidate_mask": candidate_mask,
        "stage_b_v15_dense_rank_logits": dense_logits,
        "stage_b_v15_dense_rank_score": dense_score,
    }


def _target():
    return {"boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]])}


def _manifest():
    return EvalManifest(
        path=Path("dummy.jsonl"),
        task="ref",
        manifest_key="ref:val",
        split="val",
        sha256="0" * 64,
        rows=[
            {
                "image_id": 1,
                "ann_id": 2,
                "ref_id": 3,
                "sent_id": 4,
                "sample_id": "ref:val:1:2:3:4",
            }
        ],
    )


class PatchRankDiagnosticTest(unittest.TestCase):
    def test_contract_weight_is_identical_to_default_and_lambda_zero_isolated(self):
        outputs = _diagnostic_outputs()
        logits_by_weight, candidate_idx, _valid = (
            ref_eval._diagnostic_patch_rank_candidate_logits(
                outputs,
                weights=[0.0, 1.0],
                contract_weight=1.0,
            )
        )
        self.assertTrue(
            torch.equal(
                logits_by_weight[1.0],
                outputs["stage_b_v11_final_phrase_logits"],
            )
        )
        self.assertEqual(int(candidate_idx[0, 0]), 0)
        self.assertGreater(float(logits_by_weight[0.0][0, 1, 0]), float(logits_by_weight[0.0][0, 0, 0]))

        formal = ref_eval.RefExpAccumulator(
            [0.0], [1, 2], manifest=_manifest(), run_prefix="formal"
        )
        formal.update(outputs, [_target()], cfg=_diagnostic_cfg())
        diagnostic = ref_eval.DiagnosticPatchRankAccumulator(
            [0.0, 1.0],
            1.0,
            expected_candidate_topk=2,
            contract_selection_topk=2,
        )
        diagnostic.update(outputs, [_target()])
        diagnostic_rows = {
            row["diagnostic_patch_rank_weight"]: row
            for row in diagnostic.results()
        }

        formal_row = formal.results()[0]
        self.assertEqual(diagnostic_rows[1.0]["acc50"], formal_row["acc50"])
        self.assertEqual(
            diagnostic_rows[1.0]["mean_iou_top1"],
            formal_row["mean_iou_top1"],
        )
        self.assertEqual(diagnostic_rows[1.0]["acc50"], 0.0)
        self.assertEqual(diagnostic_rows[0.0]["acc50"], 1.0)
        self.assertAlmostEqual(
            diagnostic_rows[0.0]["mean_iou_top1"], 1.0, places=4
        )
        self.assertEqual(diagnostic_rows[0.0]["candidate_oracle_recall50"], 1.0)
        self.assertEqual(diagnostic_rows[1.0]["candidate_oracle_recall50"], 1.0)

    def test_required_output_keys_fail_closed(self):
        outputs = _diagnostic_outputs()
        for key in tuple(outputs):
            with self.subTest(key=key):
                incomplete = dict(outputs)
                incomplete.pop(key)
                with self.assertRaisesRegex(KeyError, key):
                    ref_eval._diagnostic_patch_rank_candidate_logits(
                        incomplete,
                        weights=[0.0, 1.0],
                        contract_weight=1.0,
                    )

    def test_shapes_and_candidate_mask_contract_fail_closed(self):
        bad_patch_shape = _diagnostic_outputs()
        bad_patch_shape["stage_b_v15_candidate_patch_logits"] = torch.zeros((1, 2, 1))
        with self.assertRaisesRegex(ValueError, "candidate_patch_logits"):
            ref_eval._diagnostic_patch_rank_candidate_logits(
                bad_patch_shape,
                weights=[0.0, 1.0],
                contract_weight=1.0,
            )

        bad_mask = _diagnostic_outputs()
        bad_mask["stage_b_v11_candidate_mask"] = bad_mask[
            "stage_b_v11_candidate_mask"
        ].clone()
        bad_mask["stage_b_v11_candidate_mask"][0, 2, 0] = True
        with self.assertRaisesRegex(ValueError, "does not match candidate indices"):
            ref_eval._diagnostic_patch_rank_candidate_logits(
                bad_mask,
                weights=[0.0, 1.0],
                contract_weight=1.0,
            )

        bad_dense_logits = _diagnostic_outputs()
        bad_dense_logits["stage_b_v15_dense_rank_logits"] = bad_dense_logits[
            "stage_b_v15_dense_rank_logits"
        ].clone()
        bad_dense_logits["stage_b_v15_dense_rank_logits"][0, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "raw fused rank logits"):
            ref_eval._diagnostic_patch_rank_candidate_logits(
                bad_dense_logits,
                weights=[0.0, 1.0],
                contract_weight=1.0,
            )

    def test_default_is_off_and_explicit_mode_requires_fixed_top50_contract(self):
        self.assertIsNone(
            ref_eval._diagnostic_patch_rank_settings(
                None,
                types.SimpleNamespace(),
            )
        )
        settings = ref_eval._diagnostic_patch_rank_settings(
            [0.0, 0.25, 1.0],
            _diagnostic_cfg(),
        )
        self.assertEqual(settings["weights"], [0.0, 0.25, 1.0])
        self.assertEqual(settings["contract_weight"], 1.0)
        self.assertEqual(settings["candidate_topk"], 50)
        with self.assertRaisesRegex(ValueError, "stage_b_v11_fixed_text"):
            ref_eval._diagnostic_patch_rank_settings(
                [0.0, 1.0],
                _diagnostic_cfg(stage_b_v11_fixed_text=False),
            )
        with self.assertRaisesRegex(ValueError, "Top50"):
            ref_eval._diagnostic_patch_rank_settings(
                [0.0, 1.0],
                _diagnostic_cfg(stage_b_v11_candidate_topk=25),
            )
        with self.assertRaisesRegex(ValueError, "must contain"):
            ref_eval._diagnostic_patch_rank_settings(
                [0.0, 0.25],
                _diagnostic_cfg(),
            )

    def test_diagnostic_summary_is_non_formal_and_binds_grid_rows_seeds(self):
        accumulator = ref_eval.DiagnosticPatchRankAccumulator(
            [0.0, 1.0],
            1.0,
            expected_candidate_topk=2,
            contract_selection_topk=1,
        )
        accumulator.update(_diagnostic_outputs(), [_target()])
        rows = accumulator.results()
        for row in rows:
            row.update(
                {
                    "run_id": (
                        "checkpoint:diagnostic_patch_rank_weight="
                        f"{row['diagnostic_patch_rank_weight']:g}"
                    ),
                    "dataset": "refcoco_val",
                    "seed": 42,
                }
            )
        settings = {
            "weights": [0.0, 1.0],
            "contract_weight": 1.0,
            "candidate_topk": 2,
        }
        metadata = ref_eval._diagnostic_summary_metadata(settings, rows)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            ref_eval._write_summary(
                output_dir,
                rows,
                "acc50",
                diagnostic_metadata=metadata,
            )
            payload = json.loads((output_dir / "summary.json").read_text())
            self.assertIs(payload["diagnostic_only"], True)
            self.assertIs(payload["formal_gate_eligible"], False)
            self.assertEqual(payload["fixed_grid"], [0.0, 1.0])
            self.assertEqual(payload["fixed_rows"], {"refcoco_val": 1})
            self.assertEqual(payload["fixed_seeds"], {"refcoco_val": 42})
            self.assertIn(
                "formal_gate_eligible=false",
                (output_dir / "summary.md").read_text(),
            )

            ref_eval._write_summary(output_dir, rows, "acc50")
            default_payload = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(
                set(default_payload),
                {"primary_metric", "ranking", "results"},
            )
            self.assertNotIn(
                "Diagnostic only",
                (output_dir / "summary.md").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
