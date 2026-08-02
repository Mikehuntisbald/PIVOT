import unittest

import torch

from models.GroundingDINO.stage_b_native_patch_category import (
    NATIVE_PATCH_CATEGORY_COMPLETE_MARKER,
    NATIVE_PATCH_CATEGORY_LOSS,
    StageBNativePatchCategoryCriterion,
    apply_native_patch_category_gate,
)


def _category_complete_target():
    return {
        "boxes": torch.tensor(
            [[0.25, 0.50, 0.20, 0.20], [0.75, 0.50, 0.20, 0.20]],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([7, 7], dtype=torch.int64),
        "primary_instance_mask": torch.tensor([True, False]),
        NATIVE_PATCH_CATEGORY_COMPLETE_MARKER: torch.tensor([True]),
    }


def _candidate_boxes(*, requires_grad=False):
    return torch.tensor(
        [
            [
                [0.25, 0.50, 0.20, 0.20],
                [0.75, 0.50, 0.20, 0.20],
                [0.50, 0.10, 0.10, 0.10],
            ]
        ],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


class StageBNativePatchCategoryCriterionTest(unittest.TestCase):
    def setUp(self):
        self.criterion = StageBNativePatchCategoryCriterion(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            margin=0.1,
            temperature=0.1,
        )

    def _forward(self, score, *, target=None, boxes=None):
        return self.criterion(
            {
                "pred_logits_patch": score,
                "pred_boxes": _candidate_boxes() if boxes is None else boxes,
            },
            [_category_complete_target() if target is None else target],
        )

    def test_category_margin_rewards_every_instance_over_hard_negative(self):
        bad = self._forward(torch.tensor([[0.0, 0.0, 1.0]]))
        good = self._forward(torch.tensor([[2.0, 2.0, -1.0]]))

        self.assertGreater(
            float(bad[NATIVE_PATCH_CATEGORY_LOSS]),
            float(good[NATIVE_PATCH_CATEGORY_LOSS]),
        )
        self.assertEqual(
            float(good["stage_b_native_patch_category_valid_instances"]), 2.0
        )
        self.assertEqual(
            float(good["stage_b_native_patch_category_skipped_instances"]), 0.0
        )
        self.assertEqual(
            self.criterion.weight_dict, {NATIVE_PATCH_CATEGORY_LOSS: 1.0}
        )

    def test_gradient_reaches_patch_score_but_not_candidate_boxes(self):
        score = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
        boxes = _candidate_boxes(requires_grad=True)
        output = self._forward(score, boxes=boxes)

        output[NATIVE_PATCH_CATEGORY_LOSS].backward()

        self.assertIsNotNone(score.grad)
        self.assertGreater(float(score.grad.abs().sum()), 0.0)
        self.assertLess(float(score.grad[0, 0]), 0.0)
        self.assertLess(float(score.grad[0, 1]), 0.0)
        self.assertGreater(float(score.grad[0, 2]), 0.0)
        self.assertIsNone(boxes.grad)

    def test_unreachable_auxiliary_is_reported_without_discarding_primary(self):
        boxes = _candidate_boxes().clone()
        boxes[0, 1] = torch.tensor([0.50, 0.85, 0.10, 0.10])
        output = self._forward(torch.tensor([[1.0, 0.0, -1.0]]), boxes=boxes)

        self.assertEqual(
            float(output["stage_b_native_patch_category_valid_instances"]), 1.0
        )
        self.assertEqual(
            float(output["stage_b_native_patch_category_skipped_instances"]), 1.0
        )

    def test_category_complete_schema_fails_closed(self):
        cases = []
        mixed = _category_complete_target()
        mixed["labels"] = torch.tensor([7, 8], dtype=torch.int64)
        cases.append((mixed, "multiple categories"))
        no_primary = _category_complete_target()
        no_primary["primary_instance_mask"] = torch.tensor([False, False])
        cases.append((no_primary, "one exact primary"))
        false_marker = _category_complete_target()
        false_marker[NATIVE_PATCH_CATEGORY_COMPLETE_MARKER] = torch.tensor([False])
        cases.append((false_marker, "category-complete marker"))
        float_labels = _category_complete_target()
        float_labels["labels"] = float_labels["labels"].float()
        cases.append((float_labels, "labels must be int64"))
        nan_target = _category_complete_target()
        nan_target["boxes"][0, 0] = torch.nan
        cases.append((nan_target, "finite"))

        for target, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._forward(torch.zeros(1, 3), target=target)

    def test_output_shapes_and_nonfinite_values_fail_closed(self):
        cases = [
            (
                {"pred_logits_patch": torch.zeros(1, 3, 2), "pred_boxes": _candidate_boxes()},
                "one support slot",
            ),
            (
                {"pred_logits_patch": torch.zeros(1, 2), "pred_boxes": _candidate_boxes()},
                "must align",
            ),
            (
                {
                    "pred_logits_patch": torch.tensor([[0.0, torch.nan, 0.0]]),
                    "pred_boxes": _candidate_boxes(),
                },
                "finite",
            ),
            (
                {
                    "pred_logits_patch": torch.zeros(1, 3),
                    "pred_boxes": _candidate_boxes().masked_fill(
                        torch.tensor([[[True, False, False, False]] * 3]),
                        torch.nan,
                    ),
                },
                "finite",
            ),
        ]
        for outputs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.criterion(outputs, [_category_complete_target()])

    def test_batch_without_positive_negative_pair_fails_closed(self):
        boxes = _candidate_boxes().clone()
        boxes[0, 2] = boxes[0, 0]
        with self.assertRaisesRegex(RuntimeError, "no valid positive/negative"):
            self._forward(torch.zeros(1, 3), boxes=boxes)

    def test_caption_context_requires_one_full_expression_per_row(self):
        outputs = {
            "pred_logits_patch": torch.zeros(1, 3),
            "pred_boxes": _candidate_boxes(),
        }
        valid = self.criterion(
            outputs,
            [_category_complete_target()],
            [["the red cup on the left"]],
            ["the red cup on the left ."],
        )
        self.assertIn(NATIVE_PATCH_CATEGORY_LOSS, valid)
        with self.assertRaisesRegex(ValueError, "one full expression"):
            self.criterion(
                outputs,
                [_category_complete_target()],
                [["cup", "red cup"]],
                ["red cup ."],
            )


class NativePatchCategoryGateTest(unittest.TestCase):
    def test_gate_preserves_native_scores_inside_patch_eligible_set(self):
        patch = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        native = torch.tensor([[100.0, -4.0, 20.0, 0.4, 0.5]])
        mask = torch.ones_like(patch, dtype=torch.bool)

        rank, eligible, standardized = apply_native_patch_category_gate(
            native, patch, mask, max_gap=0.8, clip=5.0
        )

        self.assertEqual(eligible.tolist(), [[False, False, False, True, True]])
        self.assertTrue(torch.equal(rank[eligible], native[eligible]))
        self.assertLess(float(rank[~eligible].max()), float(rank[eligible].min()))
        self.assertGreater(float(rank[0, 4]), float(rank[0, 3]))
        self.assertEqual(tuple(standardized.shape), tuple(native.shape))

    def test_patch_positive_affine_transform_does_not_change_gate(self):
        patch = torch.tensor([[0.0, 1.0, 3.0, 5.0, 9.0]])
        transformed = patch * 7.0 + 13.0
        native = torch.tensor([[5.0, -2.0, 7.0, 1.0, 3.0]])
        mask = torch.tensor([[True, True, True, True, False]])

        first = apply_native_patch_category_gate(
            native, patch, mask, max_gap=1.0, clip=5.0
        )
        second = apply_native_patch_category_gate(
            native, transformed, mask, max_gap=1.0, clip=5.0
        )

        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.allclose(first[2], second[2], atol=1e-6, rtol=0.0))

    def test_equal_patch_scores_leave_native_rank_bitwise_unchanged(self):
        native = torch.randn(2, 4)
        patch = torch.ones(2, 4)
        mask = torch.ones(2, 4, dtype=torch.bool)

        rank, eligible, _standardized = apply_native_patch_category_gate(
            native, patch, mask, max_gap=0.0, clip=5.0
        )

        self.assertTrue(bool(eligible.all().item()))
        self.assertTrue(torch.equal(rank, native))

    def test_gate_rank_has_no_gradient_path_into_patch_amplitude(self):
        native = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        patch = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        rank, _eligible, _standardized = apply_native_patch_category_gate(
            native, patch, mask, max_gap=0.5, clip=5.0
        )

        rank.sum().backward()

        self.assertIsNotNone(native.grad)
        self.assertGreater(float(native.grad.abs().sum()), 0.0)
        self.assertIsNone(patch.grad)

    def test_gate_rejects_shape_dtype_and_nonfinite_inputs(self):
        native = torch.zeros(1, 3)
        patch = torch.zeros(1, 3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        cases = [
            (native.unsqueeze(-1), patch, mask, "native score"),
            (native, torch.zeros(1, 2), mask, "patch score"),
            (native, patch, torch.ones(1, 2, dtype=torch.bool), "candidate mask"),
            (native, patch, mask.float(), "candidate mask"),
            (
                native.masked_fill(
                    torch.tensor([[True, False, False]]), torch.nan
                ),
                patch,
                mask,
                "finite",
            ),
            (
                native,
                patch.masked_fill(
                    torch.tensor([[True, False, False]]), torch.nan
                ),
                mask,
                "finite",
            ),
        ]
        for native_value, patch_value, mask_value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    apply_native_patch_category_gate(
                        native_value,
                        patch_value,
                        mask_value,
                        max_gap=1.0,
                        clip=5.0,
                    )

        with self.assertRaisesRegex(ValueError, "requires a candidate"):
            apply_native_patch_category_gate(
                native,
                patch,
                torch.zeros_like(mask),
                max_gap=1.0,
                clip=5.0,
            )
        with self.assertRaisesRegex(ValueError, "gap/clip"):
            apply_native_patch_category_gate(
                native, patch, mask, max_gap=-1.0, clip=5.0
            )


if __name__ == "__main__":
    unittest.main()
