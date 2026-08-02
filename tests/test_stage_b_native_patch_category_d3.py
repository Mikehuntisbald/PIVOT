import inspect
import unittest

import torch
import torch.nn.functional as F

from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO.stage_b_native_patch_category_d3 import (
    NATIVE_PATCH_CATEGORY_D3_LOSS,
    NATIVE_PATCH_CATEGORY_D3_MARKER,
    StageBNativePatchCategoryD3Criterion,
)
from models.GroundingDINO.stage_b_native_patch_category_d2 import (
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
        NATIVE_PATCH_CATEGORY_D3_MARKER: torch.tensor([True]),
    }


def _boxes(batch_size, query_count):
    boxes = torch.empty(batch_size, query_count, 4, dtype=torch.float32)
    boxes[:, 0] = torch.tensor([0.20, 0.50, 0.12, 0.20])
    boxes[:, 1] = torch.tensor([0.80, 0.50, 0.12, 0.20])
    boxes[:, 2:] = torch.tensor([0.50, 0.08, 0.04, 0.04])
    # IoU is about 0.4 with GT 0: neither category-positive nor negative.
    if query_count > 4:
        boxes[:, 4] = torch.tensor([0.2514, 0.50, 0.12, 0.20])
    return boxes


def _outputs(
    patch_score,
    native_logits,
    *,
    text_requires_grad=False,
    boxes_requires_grad=False,
):
    batch_size, query_count = patch_score.shape
    boxes = _boxes(batch_size, query_count).requires_grad_(boxes_requires_grad)
    token_logits = native_logits[:, :, None].expand(
        batch_size, query_count, 3
    ).clone()
    token_logits.requires_grad_(text_requires_grad)
    return {
        "pred_logits_patch": patch_score,
        "pred_logits_text": token_logits,
        "pred_boxes": boxes,
        "phrase_to_token_mask": torch.tensor(
            [[[True, True, False]]], dtype=torch.bool
        ).expand(batch_size, 1, 3),
    }


def _critical_native(query_count):
    native = torch.linspace(1.0, -2.0, query_count)[None]
    native[0, 0] = 4.0
    native[0, 1] = 3.0
    native[0, 2] = 5.0
    return native


def _positive_native(query_count):
    native = torch.linspace(1.0, -2.0, query_count)[None]
    native[0, 0] = 5.0
    native[0, 1] = 3.0
    native[0, 2] = 4.0
    return native


class StageBNativePatchCategoryD3CriterionTest(unittest.TestCase):
    def setUp(self):
        self.criterion = StageBNativePatchCategoryD3Criterion()

    def test_critical_formula_is_exact_and_uses_native_best_positive(self):
        patch = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 0.0, -1.0, -2.0, -3.0]],
            dtype=torch.float32,
        )
        native = _critical_native(patch.shape[1])
        # Query 1, not query 0, is the native-best category-positive query.
        native[0, 1] = 4.5
        result = self.criterion(_outputs(patch, native), [_target()])

        z = gate_aligned_standardized_patch_score(
            patch, torch.ones_like(patch, dtype=torch.bool), clip=5.0
        )
        separation = z[0, 1] - z[0, 2]
        positive_gap = z[0].max().detach() - z[0, 1]
        expected_separation = F.softplus((3.25 - separation) / 0.25)
        expected_keep = F.softplus((positive_gap - 2.75) / 0.25)
        expected = 2.0 * expected_separation + expected_keep

        self.assertTrue(
            torch.allclose(
                result[NATIVE_PATCH_CATEGORY_D3_LOSS],
                expected,
                atol=1e-6,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d3_critical_rows"]),
            1.0,
        )
        self.assertEqual(
            self.criterion.weight_dict,
            {NATIVE_PATCH_CATEGORY_D3_LOSS: 1.0},
        )

    def test_nonwinner_negative_ranking_cannot_create_a_topk_loss(self):
        patch = torch.tensor(
            [[1.0, 0.5, 3.0, -3.0, -2.0, 2.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        native_a = _critical_native(patch.shape[1])
        native_b = native_a.clone()
        # Preserve q_n=2 and q_p=0 while completely reordering other negatives.
        native_a[0, 3:] = torch.tensor([2.9, 2.1, 1.3, 0.2, -1.0])
        native_b[0, 3:] = torch.tensor([-4.0, -3.0, -2.0, 3.7, 3.6])

        result_a = self.criterion(_outputs(patch, native_a), [_target()])
        result_b = self.criterion(_outputs(patch, native_b), [_target()])
        self.assertTrue(
            torch.equal(
                result_a[NATIVE_PATCH_CATEGORY_D3_LOSS],
                result_b[NATIVE_PATCH_CATEGORY_D3_LOSS],
            )
        )
        source = inspect.getsource(StageBNativePatchCategoryD3Criterion.forward)
        self.assertNotIn("topk", source)
        self.assertNotIn("coverage", source)

    def test_three_row_sets_are_averaged_independently(self):
        critical_patch = torch.tensor(
            [[1.0, 0.0, 3.0, 2.0, -1.0, -2.0, -3.0, -4.0]]
        )
        positive_patch_a = torch.tensor(
            [[1.0, 0.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]]
        )
        positive_patch_b = torch.tensor(
            [[4.0, 0.0, 3.0, 1.0, -1.0, -2.0, -3.0, -4.0]]
        )
        patch = torch.cat(
            (critical_patch, positive_patch_a, positive_patch_b), dim=0
        )
        native = torch.cat(
            (
                _critical_native(patch.shape[1]),
                _positive_native(patch.shape[1]),
                _positive_native(patch.shape[1]),
            ),
            dim=0,
        )
        result = self.criterion(
            _outputs(patch, native), [_target(), _target(), _target()]
        )

        expected = (
            2.0
            * result[
                "stage_b_native_patch_category_d3_critical_separation_loss"
            ]
            + result["stage_b_native_patch_category_d3_critical_keep_loss"]
            + result["stage_b_native_patch_category_d3_positive_keep_loss"]
        )
        self.assertTrue(
            torch.allclose(
                result[NATIVE_PATCH_CATEGORY_D3_LOSS],
                expected,
                atol=1e-6,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d3_critical_rows"]),
            1.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d3_positive_native_rows"
                ]
            ),
            2.0,
        )

    def test_gradient_reaches_only_patch_score(self):
        patch = torch.tensor(
            [[1.0, 0.0, 3.0, 2.0, -1.0, -2.0, -3.0, -4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        outputs = _outputs(
            patch,
            _critical_native(patch.shape[1]),
            text_requires_grad=True,
            boxes_requires_grad=True,
        )
        result = self.criterion(outputs, [_target()])
        result[NATIVE_PATCH_CATEGORY_D3_LOSS].backward()

        self.assertIsNotNone(patch.grad)
        self.assertGreater(float(patch.grad.abs().sum()), 0.0)
        self.assertIsNone(outputs["pred_logits_text"].grad)
        self.assertIsNone(outputs["pred_boxes"].grad)

    def test_telemetry_counts_compliance_for_both_winner_types(self):
        # With 20 queries, this pattern yields a standardized separation > 3.25.
        critical_patch = torch.tensor(
            [[4.0, 3.0, -1.0] + [0.0] * 17], dtype=torch.float32
        )
        positive_patch = torch.tensor(
            [[4.0, 3.0, -1.0] + [0.0] * 17], dtype=torch.float32
        )
        patch = torch.cat((critical_patch, positive_patch), dim=0)
        native = torch.cat(
            (_critical_native(20), _positive_native(20)), dim=0
        )
        result = self.criterion(
            _outputs(patch, native), [_target(), _target()]
        )

        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d3_separation_compliant"
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d3_q_p_keep"]), 1.0
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d3_positive_native_keep"
                ]
            ),
            1.0,
        )

    def test_neutral_native_winner_is_not_relabelled_negative(self):
        query_count = 8
        patch = torch.tensor(
            [
                [1.0, 0.0, 3.0, 2.0, -1.0, -2.0, -3.0, -4.0],
                [1.0, 0.0, 3.0, 2.0, -1.0, -2.0, -3.0, -4.0],
            ]
        )
        critical = _critical_native(query_count)
        neutral = _critical_native(query_count)
        neutral[0, 4] = 6.0
        result = self.criterion(
            _outputs(patch, torch.cat((critical, neutral), dim=0)),
            [_target(), _target()],
        )
        baseline = self.criterion(
            _outputs(patch[:1], critical), [_target()]
        )

        self.assertEqual(
            float(result["stage_b_native_patch_category_d3_critical_rows"]),
            1.0,
        )
        self.assertTrue(
            torch.equal(
                result[NATIVE_PATCH_CATEGORY_D3_LOSS],
                baseline[NATIVE_PATCH_CATEGORY_D3_LOSS],
            )
        )

    def test_build_route_and_default_weights_are_explicit(self):
        source = inspect.getsource(build_groundingdino)
        self.assertIn('native_patch_objective == "d3_critical_winner"', source)
        self.assertIn("StageBNativePatchCategoryD3Criterion", source)
        self.assertIn('"stage_b_native_patch_d3_critical_weight",', source)
        self.assertIn('"stage_b_native_patch_d3_critical_keep_weight",', source)
        self.assertIn('"stage_b_native_patch_d3_positive_keep_weight",', source)
        self.assertEqual(self.criterion.critical_weight, 2.0)
        self.assertEqual(self.criterion.critical_keep_weight, 1.0)
        self.assertEqual(self.criterion.positive_keep_weight, 1.0)
        self.assertEqual(self.criterion.temperature, 0.25)


if __name__ == "__main__":
    unittest.main()
