import unittest

import torch

from models.GroundingDINO.stage_b_data_driven_score import (
    data_driven_category_gate_mask,
)
from models.GroundingDINO.stage_b_native_patch_category_d2 import (
    NATIVE_PATCH_CATEGORY_D2_LOSS,
    NATIVE_PATCH_CATEGORY_D2_MARKER,
    StageBNativePatchCategoryD2Criterion,
    gate_aligned_standardized_patch_score,
)


def _target():
    return {
        "boxes": torch.tensor(
            [[0.20, 0.50, 0.12, 0.20], [0.80, 0.50, 0.12, 0.20]],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([7, 7], dtype=torch.int64),
        "primary_instance_mask": torch.tensor([True, False]),
        "support_class": torch.tensor([7], dtype=torch.int64),
        NATIVE_PATCH_CATEGORY_D2_MARKER: torch.tensor([True]),
    }


def _outputs(patch_score, *, text_requires_grad=False, boxes_requires_grad=False):
    query_count = int(patch_score.shape[1])
    boxes = torch.empty(1, query_count, 4, dtype=torch.float32)
    boxes[0, 0] = torch.tensor([0.20, 0.50, 0.12, 0.20])
    boxes[0, 1] = torch.tensor([0.80, 0.50, 0.12, 0.20])
    boxes[0, 2:] = torch.tensor([0.50, 0.08, 0.04, 0.04])
    boxes.requires_grad_(boxes_requires_grad)
    native = torch.linspace(2.0, -2.0, query_count)
    native[0] = 4.0
    native[1] = 3.0
    native[2] = 5.0
    token_logits = native[None, :, None].expand(1, query_count, 3).clone()
    token_logits.requires_grad_(text_requires_grad)
    return {
        "pred_logits_patch": patch_score,
        "pred_logits_text": token_logits,
        "pred_boxes": boxes,
        "phrase_to_token_mask": torch.tensor([[[True, True, False]]]),
    }


class GateAlignedStandardizationTest(unittest.TestCase):
    def test_forward_matches_exact_inference_standardization(self):
        score = torch.tensor(
            [[20.0, 3.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]],
            requires_grad=True,
        )
        candidate = torch.ones_like(score, dtype=torch.bool)
        observed = gate_aligned_standardized_patch_score(
            score, candidate, clip=1.0
        )
        _eligible, expected = data_driven_category_gate_mask(
            score.detach(), candidate, max_gap=0.5, clip=1.0
        )

        self.assertTrue(torch.equal(observed.detach(), expected))
        observed[0, 0].backward()
        self.assertIsNotNone(score.grad)
        self.assertGreater(float(score.grad.abs().sum()), 0.0)

    def test_invalid_mask_and_nonfinite_score_fail_closed(self):
        score = torch.zeros(1, 4)
        with self.assertRaisesRegex(ValueError, "boolean"):
            gate_aligned_standardized_patch_score(
                score, torch.ones_like(score), clip=5.0
            )
        score[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            gate_aligned_standardized_patch_score(
                score, torch.ones_like(score, dtype=torch.bool), clip=5.0
            )


class StageBNativePatchCategoryD2CriterionTest(unittest.TestCase):
    def setUp(self):
        self.criterion = StageBNativePatchCategoryD2Criterion()

    def test_gate_aligned_geometry_lowers_loss(self):
        bad_score = torch.tensor(
            [[-1.0, -1.0, 4.0] + [0.0] * 17], dtype=torch.float32
        )
        good_score = torch.tensor(
            [[4.0, 4.0] + [-1.0] * 18], dtype=torch.float32
        )
        bad = self.criterion(_outputs(bad_score), [_target()])
        good = self.criterion(_outputs(good_score), [_target()])

        self.assertGreater(
            float(bad[NATIVE_PATCH_CATEGORY_D2_LOSS]),
            float(good[NATIVE_PATCH_CATEGORY_D2_LOSS]),
        )
        self.assertEqual(
            float(good["stage_b_native_patch_category_d2_reachable_instances"]),
            2.0,
        )
        self.assertEqual(
            self.criterion.weight_dict,
            {NATIVE_PATCH_CATEGORY_D2_LOSS: 1.0},
        )

    def test_gradient_only_reaches_patch_score(self):
        patch_score = torch.tensor(
            [[-1.0, -1.0, 4.0] + [0.0] * 17],
            dtype=torch.float32,
            requires_grad=True,
        )
        outputs = _outputs(
            patch_score, text_requires_grad=True, boxes_requires_grad=True
        )
        result = self.criterion(outputs, [_target()])
        result[NATIVE_PATCH_CATEGORY_D2_LOSS].backward()

        self.assertIsNotNone(patch_score.grad)
        self.assertGreater(float(patch_score.grad.abs().sum()), 0.0)
        self.assertIsNone(outputs["pred_logits_text"].grad)
        self.assertIsNone(outputs["pred_boxes"].grad)

    def test_query_permutation_preserves_loss_and_counts(self):
        score = torch.tensor(
            [[3.0, 2.0, 4.0] + [0.0] * 17], dtype=torch.float32
        )
        outputs = _outputs(score)
        baseline = self.criterion(outputs, [_target()])
        permutation = torch.tensor(
            [7, 3, 1, 12, 0, 18, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19]
        )
        permuted = {
            "pred_logits_patch": outputs["pred_logits_patch"][:, permutation],
            "pred_logits_text": outputs["pred_logits_text"][:, permutation],
            "pred_boxes": outputs["pred_boxes"][:, permutation],
            "phrase_to_token_mask": outputs["phrase_to_token_mask"],
        }
        observed = self.criterion(permuted, [_target()])

        self.assertTrue(
            torch.allclose(
                baseline[NATIVE_PATCH_CATEGORY_D2_LOSS],
                observed[NATIVE_PATCH_CATEGORY_D2_LOSS],
                atol=1e-6,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(baseline["stage_b_native_patch_category_d2_drop_queries"]),
            float(observed["stage_b_native_patch_category_d2_drop_queries"]),
        )

    def test_marker_support_and_expression_contract_fail_closed(self):
        score = torch.zeros(1, 20)
        missing_marker = _target()
        del missing_marker[NATIVE_PATCH_CATEGORY_D2_MARKER]
        with self.assertRaisesRegex(ValueError, "D2 marker"):
            self.criterion(_outputs(score), [missing_marker])

        wrong_support = _target()
        wrong_support["support_class"] = torch.tensor([8], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.criterion(_outputs(score), [wrong_support])

        invalid_mask = _outputs(score)
        invalid_mask["phrase_to_token_mask"] = torch.zeros(
            1, 1, 3, dtype=torch.bool
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.criterion(invalid_mask, [_target()])

    def test_padding_nonfinite_is_ignored_but_scored_nonfinite_is_rejected(self):
        score = torch.zeros(1, 20)
        padding_nonfinite = _outputs(score)
        padding_nonfinite["pred_logits_text"][:, :, 2] = -torch.inf
        result = self.criterion(padding_nonfinite, [_target()])
        self.assertTrue(
            torch.isfinite(result[NATIVE_PATCH_CATEGORY_D2_LOSS]).item()
        )

        scored_nonfinite = _outputs(score)
        scored_nonfinite["pred_logits_text"][0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "full-expression token logits"):
            self.criterion(scored_nonfinite, [_target()])


if __name__ == "__main__":
    unittest.main()
