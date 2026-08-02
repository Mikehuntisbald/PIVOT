import inspect
import unittest

import torch
import torch.nn.functional as F

from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO.stage_b_native_patch_category_d2 import (
    gate_aligned_standardized_patch_score,
)
from models.GroundingDINO.stage_b_native_patch_category_d3 import (
    NATIVE_PATCH_CATEGORY_D3_LOSS,
    StageBNativePatchCategoryD3Criterion,
)
from models.GroundingDINO.stage_b_native_patch_category_d5 import (
    NATIVE_PATCH_CATEGORY_D5_CONTRACT_VERSION,
    NATIVE_PATCH_CATEGORY_D5_LOSS,
    NATIVE_PATCH_CATEGORY_D5_MARKER,
    StageBNativePatchCategoryD5Criterion,
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
        NATIVE_PATCH_CATEGORY_D5_MARKER: torch.tensor([True]),
    }


def _outputs(
    patch_score,
    native_winners,
    *,
    require_selector_grads=False,
):
    batch_size, query_count = patch_score.shape
    boxes = torch.empty(batch_size, query_count, 4, dtype=torch.float32)
    boxes[:, 0] = torch.tensor([0.20, 0.50, 0.12, 0.20])
    boxes[:, 1] = torch.tensor([0.80, 0.50, 0.12, 0.20])
    boxes[:, 2:] = torch.tensor([0.50, 0.08, 0.04, 0.04])
    token_logits = native_winners[:, :, None].expand(
        batch_size, query_count, 3
    ).clone()
    boxes.requires_grad_(require_selector_grads)
    token_logits.requires_grad_(require_selector_grads)
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


class StageBNativePatchCategoryD5CriterionTest(unittest.TestCase):
    def test_critical_terms_are_numerically_identical_to_d3(self):
        patch = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 0.0, -1.0, -2.0, -3.0]],
            dtype=torch.float32,
        )
        native = _critical_native(8)
        native[0, 1] = 4.5
        outputs = _outputs(patch, native)

        d3 = StageBNativePatchCategoryD3Criterion()(outputs, [_target()])
        d5 = StageBNativePatchCategoryD5Criterion()(outputs, [_target()])

        self.assertTrue(
            torch.equal(
                d5[NATIVE_PATCH_CATEGORY_D5_LOSS],
                d3[NATIVE_PATCH_CATEGORY_D3_LOSS],
            )
        )
        self.assertTrue(
            torch.equal(
                d5[
                    "stage_b_native_patch_category_d5_critical_separation_loss"
                ],
                d3[
                    "stage_b_native_patch_category_d3_critical_separation_loss"
                ],
            )
        )
        self.assertTrue(
            torch.equal(
                d5["stage_b_native_patch_category_d5_critical_keep_loss"],
                d3["stage_b_native_patch_category_d3_critical_keep_loss"],
            )
        )

    def test_positive_barrier_is_exact_thresholded_active_mean(self):
        # Standardized q0 gaps are about 1.90, 2.28, and 4.00.
        patch = torch.tensor(
            [
                [-1.0, 0.0, 3.0, 4.0, 1.0, -2.0, -3.0, -4.0],
                [-2.0, 0.0, 3.0, 4.0, 1.0, -1.0, -3.0, -4.0],
                [-4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        native = _positive_native(8).expand(3, -1).clone()
        result = StageBNativePatchCategoryD5Criterion()(
            _outputs(patch, native), [_target(), _target(), _target()]
        )

        standardized = gate_aligned_standardized_patch_score(
            patch, torch.ones_like(patch, dtype=torch.bool), clip=5.0
        )
        gaps = standardized.amax(dim=1).detach() - standardized[:, 0]
        active = gaps.detach() > 2.0
        expected_barrier = F.softplus((gaps[active] - 2.5) / 0.25).mean()
        expected = 2.0 * expected_barrier

        self.assertTrue(
            torch.allclose(
                result[NATIVE_PATCH_CATEGORY_D5_LOSS],
                expected,
                atol=1e-6,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d5_positive_native_rows"
                ]
            ),
            3.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d5_positive_active_rows"
                ]
            ),
            2.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d5_positive_target_keep"
                ]
            ),
            2.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d5_positive_deployment_keep"
                ]
            ),
            2.0,
        )

    def test_inactive_positive_rows_do_not_dilute_the_active_mean(self):
        active_patch = torch.tensor(
            [[-2.0, 0.0, 3.0, 4.0, 1.0, -1.0, -3.0, -4.0]],
            dtype=torch.float32,
        )
        inactive_patch = torch.tensor(
            [[-1.0, 0.0, 3.0, 4.0, 1.0, -2.0, -3.0, -4.0]],
            dtype=torch.float32,
        )
        criterion = StageBNativePatchCategoryD5Criterion()
        active_only = criterion(
            _outputs(active_patch, _positive_native(8)), [_target()]
        )
        mixed_patch = torch.cat(
            (active_patch, inactive_patch.expand(4, -1)), dim=0
        )
        mixed_native = _positive_native(8).expand(5, -1).clone()
        mixed = criterion(
            _outputs(mixed_patch, mixed_native), [_target() for _ in range(5)]
        )

        self.assertTrue(
            torch.equal(
                active_only[NATIVE_PATCH_CATEGORY_D5_LOSS],
                mixed[NATIVE_PATCH_CATEGORY_D5_LOSS],
            )
        )
        self.assertEqual(
            float(
                mixed[
                    "stage_b_native_patch_category_d5_positive_active_rows"
                ]
            ),
            1.0,
        )

    def test_no_risk_batch_has_a_differentiable_zero_loss(self):
        patch = torch.tensor(
            [[1.0, 0.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        result = StageBNativePatchCategoryD5Criterion()(
            _outputs(patch, _positive_native(8)), [_target()]
        )
        self.assertEqual(
            float(result[NATIVE_PATCH_CATEGORY_D5_LOSS].detach()), 0.0
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d5_positive_active_rows"
                ]
            ),
            0.0,
        )

        result[NATIVE_PATCH_CATEGORY_D5_LOSS].backward()
        self.assertIsNotNone(patch.grad)
        self.assertEqual(float(patch.grad.abs().sum()), 0.0)

    def test_gradient_isolated_to_patch_score(self):
        patch = torch.tensor(
            [[-4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        outputs = _outputs(
            patch,
            _positive_native(8),
            require_selector_grads=True,
        )
        result = StageBNativePatchCategoryD5Criterion()(outputs, [_target()])
        result[NATIVE_PATCH_CATEGORY_D5_LOSS].backward()

        self.assertIsNotNone(patch.grad)
        self.assertGreater(float(patch.grad.abs().sum()), 0.0)
        self.assertIsNone(outputs["pred_logits_text"].grad)
        self.assertIsNone(outputs["pred_boxes"].grad)

    def test_contract_defaults_geometry_and_build_route_are_d5_specific(self):
        criterion = StageBNativePatchCategoryD5Criterion()
        self.assertEqual(NATIVE_PATCH_CATEGORY_D5_CONTRACT_VERSION, 5)
        self.assertEqual(criterion.critical_weight, 2.0)
        self.assertEqual(criterion.critical_keep_weight, 1.0)
        self.assertEqual(criterion.active_gap, 2.0)
        self.assertEqual(criterion.target_gap, 2.5)
        self.assertEqual(criterion.positive_barrier_weight, 2.0)
        self.assertEqual(
            criterion.weight_dict,
            {NATIVE_PATCH_CATEGORY_D5_LOSS: 1.0},
        )
        with self.assertRaisesRegex(ValueError, "geometry"):
            StageBNativePatchCategoryD5Criterion(active_gap=2.5)
        with self.assertRaisesRegex(ValueError, "geometry"):
            StageBNativePatchCategoryD5Criterion(target_gap=3.0)
        with self.assertRaisesRegex(ValueError, "geometry"):
            StageBNativePatchCategoryD5Criterion(positive_barrier_weight=0.0)

        source = inspect.getsource(build_groundingdino)
        self.assertIn('== "d5_active_tail_positive_barrier"', source)
        self.assertIn("StageBNativePatchCategoryD5Criterion", source)
        for field in (
            "stage_b_native_patch_d5_weight",
            "stage_b_native_patch_d5_keep_gap",
            "stage_b_native_patch_d5_separation_gap",
            "stage_b_native_patch_d5_temperature",
            "stage_b_native_patch_d5_critical_weight",
            "stage_b_native_patch_d5_critical_keep_weight",
            "stage_b_native_patch_d5_active_gap",
            "stage_b_native_patch_d5_target_gap",
            "stage_b_native_patch_d5_positive_barrier_weight",
        ):
            self.assertIn(field, source)
        d5_source = inspect.getsource(StageBNativePatchCategoryD5Criterion.forward)
        self.assertIn("positive_gap > self.active_gap", d5_source)
        self.assertNotIn("positive_keep_losses", d5_source)


if __name__ == "__main__":
    unittest.main()
