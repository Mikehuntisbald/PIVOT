import inspect
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO.stage_b_native_patch_category_d3 import (
    NATIVE_PATCH_CATEGORY_D3_MARKER,
)
from models.GroundingDINO import stage_b_native_patch_category_d6 as d6_module
from models.GroundingDINO.stage_b_native_patch_category_d6 import (
    NATIVE_PATCH_CATEGORY_D6_CONTRACT_VERSION,
    NATIVE_PATCH_CATEGORY_D6_LOSS,
    NATIVE_PATCH_CATEGORY_D6_MARKER,
    StageBNativePatchCategoryD6Criterion,
)


def _target(*, reverse=False, primary=None, include_other_class=False):
    boxes = torch.tensor(
        [[0.20, 0.50, 0.12, 0.20], [0.80, 0.50, 0.12, 0.20]],
        dtype=torch.float32,
    )
    labels = torch.tensor([7, 7], dtype=torch.int64)
    if include_other_class:
        boxes = torch.cat(
            (boxes, torch.tensor([[0.50, 0.08, 0.04, 0.04]])), dim=0
        )
        labels = torch.tensor([7, 7, 9], dtype=torch.int64)
    if reverse:
        order = torch.arange(boxes.shape[0] - 1, -1, -1)
        boxes = boxes[order]
        labels = labels[order]
    target = {
        "boxes": boxes,
        "labels": labels,
        "support_class": torch.tensor([7], dtype=torch.int64),
        NATIVE_PATCH_CATEGORY_D6_MARKER: torch.tensor([True]),
    }
    if primary is not None:
        target["primary_instance_mask"] = primary
    return target


def _boxes(batch_size, query_count):
    boxes = torch.empty(batch_size, query_count, 4, dtype=torch.float32)
    boxes[:, 0] = torch.tensor([0.20, 0.50, 0.12, 0.20])
    boxes[:, 1] = torch.tensor([0.80, 0.50, 0.12, 0.20])
    boxes[:, 2:] = torch.tensor([0.50, 0.08, 0.04, 0.04])
    if query_count > 4:
        # IoU is about 0.4 with GT 0: category-neutral.
        boxes[:, 4] = torch.tensor([0.2514, 0.50, 0.12, 0.20])
    return boxes


def _outputs(z, native, *, selector_grads=False):
    batch_size, query_count = z.shape
    boxes = _boxes(batch_size, query_count).requires_grad_(selector_grads)
    token_logits = native[:, :, None].expand(
        batch_size, query_count, 3
    ).clone()
    token_logits.requires_grad_(selector_grads)
    return {
        "pred_logits_patch": z,
        "pred_logits_text": token_logits,
        "pred_boxes": boxes,
        "phrase_to_token_mask": torch.tensor(
            [[[True, True, False]]], dtype=torch.bool
        ).expand(batch_size, 1, 3),
    }


def _critical_native(query_count=6):
    native = torch.linspace(1.0, -1.0, query_count)[None]
    native[0, 0] = 4.0
    native[0, 1] = 4.5
    native[0, 2] = 5.0
    return native


def _positive_native(query_count=6):
    native = torch.linspace(1.0, -1.0, query_count)[None]
    native[0, 0] = 5.0
    native[0, 1] = 3.0
    native[0, 2] = 4.0
    return native


def _neutral_native(query_count=6):
    native = _critical_native(query_count)
    native[0, 4] = 6.0
    return native


def _identity_standardization():
    return mock.patch.object(
        d6_module,
        "gate_aligned_standardized_patch_score",
        side_effect=lambda patch_score, candidate_mask, *, clip: patch_score,
    )


class StageBNativePatchCategoryD6CriterionTest(unittest.TestCase):
    def setUp(self):
        self.criterion = StageBNativePatchCategoryD6Criterion()

    def test_exact_direct_gap_formula_uses_native_best_positive(self):
        critical_z = torch.tensor([[3.0, 3.2, 1.6, 5.0, -1.0, -2.0]])
        positive_z = torch.tensor([[2.6, 0.0, 1.0, 5.0, -1.0, -2.0]])
        z = torch.cat((critical_z, positive_z), dim=0)
        native = torch.cat((_critical_native(), _positive_native()), dim=0)
        with _identity_standardization():
            result = self.criterion(
                _outputs(z, native), [_target(), _target()]
            )

        negative_gap = 5.0 - critical_z[0, 2]
        positive_gap = 5.0 - critical_z[0, 1]
        winner_gap = 5.0 - positive_z[0, 0]
        expected_drop = F.softplus((3.25 - negative_gap) / 0.25)
        expected_keep = F.softplus((positive_gap - 2.75) / 0.25)
        expected_barrier = F.softplus((winner_gap - 2.5) / 0.25)
        expected = 2.0 * expected_drop + expected_keep + 2.0 * expected_barrier

        self.assertTrue(
            torch.allclose(
                result[NATIVE_PATCH_CATEGORY_D6_LOSS],
                expected,
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d6_critical_rows"]),
            1.0,
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d6_drop_active_rows"]),
            1.0,
        )
        source = inspect.getsource(StageBNativePatchCategoryD6Criterion.forward)
        self.assertNotIn("separation", source)

    def test_drop_is_independent_of_positive_score_at_fixed_negative_gap(self):
        z_a = torch.tensor([[3.0, 4.0, 1.6, 5.0, -1.0, -2.0]])
        z_b = z_a.clone()
        z_b[0, 1] = 2.0
        native = _critical_native()
        with _identity_standardization():
            result_a = self.criterion(_outputs(z_a, native), [_target()])
            result_b = self.criterion(_outputs(z_b, native), [_target()])

        self.assertTrue(
            torch.equal(
                result_a["stage_b_native_patch_category_d6_drop_loss"],
                result_b["stage_b_native_patch_category_d6_drop_loss"],
            )
        )
        self.assertFalse(
            torch.equal(
                result_a["stage_b_native_patch_category_d6_critical_keep_loss"],
                result_b["stage_b_native_patch_category_d6_critical_keep_loss"],
            )
        )

    def test_patch_gradient_directions_and_detached_selectors(self):
        def gradients(drop_weight):
            z = torch.tensor(
                [[3.0, 3.2, 1.6, 5.0, -1.0, -2.0]],
                requires_grad=True,
            )
            outputs = _outputs(z, _critical_native(), selector_grads=True)
            target = _target()
            target["boxes"].requires_grad_(True)
            criterion = StageBNativePatchCategoryD6Criterion(
                drop_weight=drop_weight
            )
            with _identity_standardization():
                loss = criterion(outputs, [target])[NATIVE_PATCH_CATEGORY_D6_LOSS]
            loss.backward()
            self.assertIsNone(outputs["pred_logits_text"].grad)
            self.assertIsNone(outputs["pred_boxes"].grad)
            self.assertIsNone(target["boxes"].grad)
            return z.grad

        default_grad = gradients(2.0)
        stronger_drop_grad = gradients(7.0)
        # Gradient descent lowers q_n and raises q_p.
        self.assertGreater(float(default_grad[0, 2]), 0.0)
        self.assertLess(float(default_grad[0, 1]), 0.0)
        # Changing only the drop weight cannot alter q_p's keep gradient.
        self.assertTrue(
            torch.equal(default_grad[0, 1], stronger_drop_grad[0, 1])
        )
        self.assertGreater(
            float(stronger_drop_grad[0, 2]), float(default_grad[0, 2])
        )
        # b=max(z) is detached, so an unselected best query has no gradient.
        self.assertEqual(float(default_grad[0, 3]), 0.0)

    def test_three_active_sets_have_independent_means_and_empty_zero(self):
        active_critical = torch.tensor(
            [[3.0, 3.2, 1.6, 5.0, -1.0, -2.0]]
        )
        active_positive = torch.tensor(
            [[2.6, 0.0, 1.0, 5.0, -1.0, -2.0]]
        )
        with _identity_standardization():
            baseline = self.criterion(
                _outputs(
                    torch.cat((active_critical, active_positive)),
                    torch.cat((_critical_native(), _positive_native())),
                ),
                [_target(), _target()],
            )

            inactive_drop = torch.tensor(
                [[3.0, 0.0, 1.25, 5.0, -1.0, -2.0]]
            )
            inactive_positive = torch.tensor(
                [[3.0, 0.0, 1.0, 5.0, -1.0, -2.0]]
            )
            mixed = self.criterion(
                _outputs(
                    torch.cat(
                        (
                            active_critical,
                            inactive_drop,
                            active_positive,
                            inactive_positive,
                        )
                    ),
                    torch.cat(
                        (
                            _critical_native(),
                            _critical_native(),
                            _positive_native(),
                            _positive_native(),
                        )
                    ),
                ),
                [_target() for _ in range(4)],
            )

        self.assertTrue(
            torch.equal(
                baseline["stage_b_native_patch_category_d6_drop_loss"],
                mixed["stage_b_native_patch_category_d6_drop_loss"],
            )
        )
        self.assertTrue(
            torch.equal(
                baseline["stage_b_native_patch_category_d6_positive_barrier_loss"],
                mixed["stage_b_native_patch_category_d6_positive_barrier_loss"],
            )
        )
        self.assertEqual(
            float(mixed["stage_b_native_patch_category_d6_critical_rows"]),
            2.0,
        )
        self.assertEqual(
            float(mixed["stage_b_native_patch_category_d6_drop_active_rows"]),
            1.0,
        )
        self.assertEqual(
            float(mixed["stage_b_native_patch_category_d6_positive_native_rows"]),
            2.0,
        )
        self.assertEqual(
            float(mixed["stage_b_native_patch_category_d6_positive_active_rows"]),
            1.0,
        )

        z = torch.tensor(
            [[3.0, 3.2, 1.6, 5.0, -1.0, -2.0]], requires_grad=True
        )
        with _identity_standardization():
            empty = self.criterion(_outputs(z, _neutral_native()), [_target()])
        self.assertEqual(
            float(empty[NATIVE_PATCH_CATEGORY_D6_LOSS].detach()), 0.0
        )
        empty[NATIVE_PATCH_CATEGORY_D6_LOSS].backward()
        self.assertIsNotNone(z.grad)
        self.assertEqual(float(z.grad.abs().sum()), 0.0)

    def test_active_target_and_deployment_boundaries_are_exact(self):
        def critical_at(gap):
            z = torch.tensor(
                [[3.0, 3.2, 5.0 - gap, 5.0, -1.0, -2.0]]
            )
            return self.criterion(_outputs(z, _critical_native()), [_target()])

        def positive_at(gap):
            z = torch.tensor(
                [[5.0 - gap, 0.0, 1.0, 5.0, -1.0, -2.0]]
            )
            return self.criterion(_outputs(z, _positive_native()), [_target()])

        with _identity_standardization():
            drop_release = critical_at(3.75)
            drop_inside = critical_at(3.749)
            drop_target = critical_at(3.25)
            deployment_edge = critical_at(3.0)
            deployment_outside = critical_at(3.001)
            positive_release = positive_at(2.0)
            positive_inside = positive_at(2.001)
            positive_target = positive_at(2.5)

        self.assertEqual(
            float(drop_release["stage_b_native_patch_category_d6_drop_active_rows"]),
            0.0,
        )
        self.assertEqual(
            float(drop_inside["stage_b_native_patch_category_d6_drop_active_rows"]),
            1.0,
        )
        self.assertTrue(
            torch.allclose(
                drop_target["stage_b_native_patch_category_d6_drop_loss"],
                torch.tensor(math.log(2.0)),
                atol=1e-6,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(
                deployment_edge[
                    "stage_b_native_patch_category_d6_negative_deployment_rejected"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            float(
                deployment_outside[
                    "stage_b_native_patch_category_d6_negative_deployment_rejected"
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(
                positive_release[
                    "stage_b_native_patch_category_d6_positive_active_rows"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            float(
                positive_inside[
                    "stage_b_native_patch_category_d6_positive_active_rows"
                ]
            ),
            1.0,
        )
        self.assertTrue(
            torch.allclose(
                positive_target[
                    "stage_b_native_patch_category_d6_positive_barrier_loss"
                ],
                torch.tensor(math.log(2.0)),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_all_same_class_gt_are_used_and_metadata_is_ignored(self):
        z = torch.tensor([[3.0, 3.2, 1.6, 5.0, -1.0, -2.0]])
        native = _critical_native()
        target_a = _target(
            primary=torch.tensor([True, False, False]),
            include_other_class=True,
        )
        target_b = _target(
            reverse=True,
            primary=torch.tensor([False, False, False]),
            include_other_class=True,
        )
        target_a["source_dataset"] = object()
        target_a["stage_b_native_patch_category_variant"] = object()
        target_b["source_dataset"] = "different"
        target_b["stage_b_native_patch_category_variant"] = "different"

        with _identity_standardization():
            result_a = self.criterion(_outputs(z, native), [target_a])
            result_b = self.criterion(_outputs(z, native), [target_b])

        self.assertTrue(
            torch.equal(
                result_a[NATIVE_PATCH_CATEGORY_D6_LOSS],
                result_b[NATIVE_PATCH_CATEGORY_D6_LOSS],
            )
        )
        # The other-class GT exactly overlaps q_n; it must not relabel q_n.
        self.assertEqual(
            float(result_a["stage_b_native_patch_category_d6_critical_rows"]),
            1.0,
        )
        source = inspect.getsource(StageBNativePatchCategoryD6Criterion.forward)
        self.assertNotIn("primary_instance_mask", source)
        self.assertNotIn("source_dataset", source)
        self.assertNotIn("variant", source)

    def test_neutral_winner_is_never_a_drop_label(self):
        z = torch.tensor([[3.0, 3.2, 1.6, 5.0, -1.0, -2.0]])
        with _identity_standardization():
            result = self.criterion(
                _outputs(z, _neutral_native()), [_target()]
            )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d6_critical_rows"]),
            0.0,
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d6_drop_active_rows"]),
            0.0,
        )
        self.assertEqual(float(result[NATIVE_PATCH_CATEGORY_D6_LOSS]), 0.0)

    def test_contract_defaults_geometry_and_build_route_are_explicit(self):
        criterion = StageBNativePatchCategoryD6Criterion()
        self.assertEqual(NATIVE_PATCH_CATEGORY_D6_CONTRACT_VERSION, 6)
        self.assertEqual(NATIVE_PATCH_CATEGORY_D6_MARKER, NATIVE_PATCH_CATEGORY_D3_MARKER)
        self.assertEqual(criterion.keep_gap, 2.75)
        self.assertEqual(criterion.drop_gap, 3.25)
        self.assertEqual(criterion.drop_active_gap, 3.75)
        self.assertEqual(criterion.temperature, 0.25)
        self.assertEqual(criterion.drop_weight, 2.0)
        self.assertEqual(criterion.critical_keep_weight, 1.0)
        self.assertEqual(criterion.positive_active_gap, 2.0)
        self.assertEqual(criterion.positive_target_gap, 2.5)
        self.assertEqual(criterion.positive_barrier_weight, 2.0)
        self.assertEqual(
            criterion.weight_dict, {NATIVE_PATCH_CATEGORY_D6_LOSS: 1.0}
        )
        with self.assertRaisesRegex(ValueError, "geometry"):
            StageBNativePatchCategoryD6Criterion(drop_active_gap=3.25)
        with self.assertRaisesRegex(ValueError, "geometry"):
            StageBNativePatchCategoryD6Criterion(positive_active_gap=2.5)
        with self.assertRaisesRegex(ValueError, "geometry"):
            StageBNativePatchCategoryD6Criterion(drop_weight=0.0)

        source = inspect.getsource(build_groundingdino)
        self.assertIn('native_patch_objective == "d6_direct_deployment_gap"', source)
        self.assertIn("StageBNativePatchCategoryD6Criterion", source)
        for field in (
            "stage_b_native_patch_d6_weight",
            "stage_b_native_patch_d6_keep_gap",
            "stage_b_native_patch_d6_drop_gap",
            "stage_b_native_patch_d6_drop_active_gap",
            "stage_b_native_patch_d6_temperature",
            "stage_b_native_patch_d6_drop_weight",
            "stage_b_native_patch_d6_critical_keep_weight",
            "stage_b_native_patch_d6_positive_active_gap",
            "stage_b_native_patch_d6_positive_target_gap",
            "stage_b_native_patch_d6_positive_barrier_weight",
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
