import inspect
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO import stage_b_native_patch_category_d6 as d6_module
from models.GroundingDINO import stage_b_native_patch_category_d7 as d7_module
from models.GroundingDINO.stage_b_native_patch_category_d6 import (
    NATIVE_PATCH_CATEGORY_D6_LOSS,
    NATIVE_PATCH_CATEGORY_D6_MARKER,
    StageBNativePatchCategoryD6Criterion,
)
from models.GroundingDINO.stage_b_native_patch_category_d7 import (
    NATIVE_PATCH_CATEGORY_D7_CONTRACT_VERSION,
    NATIVE_PATCH_CATEGORY_D7_LOSS,
    NATIVE_PATCH_CATEGORY_D7_MARKER,
    StageBNativePatchCategoryD7Criterion,
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
        NATIVE_PATCH_CATEGORY_D7_MARKER: torch.tensor([True]),
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


def _negative_native(query_count=6):
    native = torch.linspace(1.0, -1.0, query_count)[None]
    native[0, 0] = 4.0
    native[0, 1] = 4.5
    native[0, 2] = 5.0
    return native


def _neutral_native(query_count=6):
    native = _negative_native(query_count)
    native[0, 4] = 6.0
    return native


def _positive_native(query_count=6):
    native = torch.linspace(1.0, -1.0, query_count)[None]
    native[0, 0] = 5.0
    native[0, 1] = 4.5
    native[0, 2] = 4.0
    return native


def _identity_standardization():
    identity = lambda patch_score, candidate_mask, *, clip: patch_score
    return (
        mock.patch.object(
            d6_module,
            "gate_aligned_standardized_patch_score",
            side_effect=identity,
        ),
        mock.patch.object(
            d7_module,
            "gate_aligned_standardized_patch_score",
            side_effect=identity,
        ),
    )


class StageBNativePatchCategoryD7CriterionTest(unittest.TestCase):
    def setUp(self):
        self.criterion = StageBNativePatchCategoryD7Criterion()

    def test_anchor_covers_negative_neutral_and_positive_winner_states(self):
        negative_z = torch.tensor([[3.0, 2.5, 1.6, 5.0, -1.0, -2.0]])
        neutral_z = torch.tensor([[3.0, 2.0, 1.6, 5.0, -1.0, -2.0]])
        positive_z = torch.tensor([[2.75, 2.0, 1.0, 5.0, -1.0, -2.0]])
        z = torch.cat((negative_z, neutral_z, positive_z), dim=0)
        native = torch.cat(
            (_negative_native(), _neutral_native(), _positive_native()), dim=0
        )
        d6 = StageBNativePatchCategoryD6Criterion()
        d6_patch, d7_patch = _identity_standardization()
        with d6_patch, d7_patch:
            outputs = _outputs(z, native)
            targets = [_target(), _target(), _target()]
            base = d6(outputs, targets)
            result = self.criterion(outputs, targets)

        gaps = torch.tensor([2.5, 3.0, 2.25])
        expected_anchor = F.softplus((gaps - 2.5) / 0.25).mean()
        expected = base[NATIVE_PATCH_CATEGORY_D6_LOSS] + 2.0 * expected_anchor
        self.assertTrue(
            torch.allclose(
                result[NATIVE_PATCH_CATEGORY_D7_LOSS],
                expected,
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                result["stage_b_native_patch_category_d7_anchor_loss"],
                expected_anchor,
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d7_anchor_rows"]), 3.0
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d7_anchor_active_rows"]),
            3.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d7_anchor_deployment_keep"
                ]
            ),
            3.0,
        )
        self.assertEqual(
            float(result["stage_b_native_patch_category_d7_anchor_target_keep"]),
            2.0,
        )

    def test_d6_terms_and_telemetry_are_reused_exactly(self):
        z = torch.tensor(
            [
                [3.0, 2.5, 1.6, 5.0, -1.0, -2.0],
                [2.75, 2.0, 1.0, 5.0, -1.0, -2.0],
            ]
        )
        native = torch.cat((_negative_native(), _positive_native()), dim=0)
        d6_patch, d7_patch = _identity_standardization()
        with d6_patch, d7_patch:
            outputs = _outputs(z, native)
            targets = [_target(), _target()]
            d6_result = StageBNativePatchCategoryD6Criterion()(outputs, targets)
            d7_result = self.criterion(outputs, targets)

        for name, value in d6_result.items():
            if name == NATIVE_PATCH_CATEGORY_D6_LOSS:
                continue
            d7_name = name.replace(
                "stage_b_native_patch_category_d6_",
                "stage_b_native_patch_category_d7_",
                1,
            )
            self.assertTrue(torch.equal(d7_result[d7_name], value), d7_name)
        expected = (
            d6_result[NATIVE_PATCH_CATEGORY_D6_LOSS]
            + 2.0 * d7_result["stage_b_native_patch_category_d7_anchor_loss"]
        )
        self.assertTrue(
            torch.equal(d7_result[NATIVE_PATCH_CATEGORY_D7_LOSS], expected)
        )
        source = inspect.getsource(StageBNativePatchCategoryD7Criterion.forward)
        self.assertIn("super().forward", source)

    def test_anchor_active_mean_is_not_diluted_by_inactive_rows(self):
        active = torch.tensor([[3.0, 2.25, 1.6, 5.0, -1.0, -2.0]])
        inactive_negative = torch.tensor(
            [[3.0, 3.0, 1.6, 5.0, -1.0, -2.0]]
        )
        inactive_positive = torch.tensor(
            [[3.0, 2.0, 1.0, 5.0, -1.0, -2.0]]
        )
        d6_patch, d7_patch = _identity_standardization()
        with d6_patch, d7_patch:
            active_only = self.criterion(
                _outputs(active, _neutral_native()), [_target()]
            )
            mixed = self.criterion(
                _outputs(
                    torch.cat((active, inactive_negative, inactive_positive)),
                    torch.cat(
                        (
                            _neutral_native(),
                            _negative_native(),
                            _positive_native(),
                        )
                    ),
                ),
                [_target(), _target(), _target()],
            )

        self.assertTrue(
            torch.equal(
                active_only["stage_b_native_patch_category_d7_anchor_loss"],
                mixed["stage_b_native_patch_category_d7_anchor_loss"],
            )
        )
        self.assertEqual(
            float(mixed["stage_b_native_patch_category_d7_anchor_rows"]), 3.0
        )
        self.assertEqual(
            float(mixed["stage_b_native_patch_category_d7_anchor_active_rows"]),
            1.0,
        )

    def test_anchor_boundary_and_empty_set_are_exact(self):
        def neutral_at(gap):
            z = torch.tensor(
                [[3.0, 5.0 - gap, 1.6, 5.0, -1.0, -2.0]]
            )
            return self.criterion(_outputs(z, _neutral_native()), [_target()])

        d6_patch, d7_patch = _identity_standardization()
        with d6_patch, d7_patch:
            release = neutral_at(2.0)
            inside = neutral_at(2.001)
            target = neutral_at(2.5)
            deployment_edge = neutral_at(3.0)
            deployment_outside = neutral_at(3.001)

        self.assertEqual(
            float(release["stage_b_native_patch_category_d7_anchor_active_rows"]),
            0.0,
        )
        self.assertEqual(
            float(inside["stage_b_native_patch_category_d7_anchor_active_rows"]),
            1.0,
        )
        self.assertTrue(
            torch.allclose(
                target["stage_b_native_patch_category_d7_anchor_loss"],
                torch.tensor(math.log(2.0)),
                atol=1e-6,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(
                deployment_edge[
                    "stage_b_native_patch_category_d7_anchor_deployment_keep"
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(
                deployment_outside[
                    "stage_b_native_patch_category_d7_anchor_deployment_keep"
                ]
            ),
            0.0,
        )

        z = torch.tensor(
            [[3.0, 2.0, 1.6, 5.0, -1.0, -2.0]], requires_grad=True
        )
        outputs = _outputs(z, _neutral_native())
        outputs["pred_boxes"] = torch.tensor(
            [[[0.50, 0.08, 0.04, 0.04]] * 6], dtype=torch.float32
        )
        d6_patch, d7_patch = _identity_standardization()
        with d6_patch, d7_patch:
            empty = self.criterion(outputs, [_target()])
        self.assertEqual(
            float(empty["stage_b_native_patch_category_d7_anchor_rows"]), 0.0
        )
        self.assertEqual(float(empty[NATIVE_PATCH_CATEGORY_D7_LOSS].detach()), 0.0)
        empty[NATIVE_PATCH_CATEGORY_D7_LOSS].backward()
        self.assertIsNotNone(z.grad)
        self.assertEqual(float(z.grad.abs().sum()), 0.0)

    def test_anchor_only_gradient_reaches_patch_and_uses_detached_selectors(self):
        z = torch.tensor(
            [[3.0, 2.0, 1.6, 5.0, -1.0, -2.0]], requires_grad=True
        )
        outputs = _outputs(z, _neutral_native(), selector_grads=True)
        target = _target()
        target["boxes"].requires_grad_(True)
        d6_patch, d7_patch = _identity_standardization()
        with d6_patch, d7_patch:
            result = self.criterion(outputs, [target])
        result[NATIVE_PATCH_CATEGORY_D7_LOSS].backward()

        self.assertLess(float(z.grad[0, 1]), 0.0)
        self.assertEqual(float(z.grad[0, 3]), 0.0)
        self.assertIsNone(outputs["pred_logits_text"].grad)
        self.assertIsNone(outputs["pred_boxes"].grad)
        self.assertIsNone(target["boxes"].grad)

    def test_same_class_gt_order_and_metadata_do_not_change_anchor(self):
        z = torch.tensor([[3.0, 2.0, 1.6, 5.0, -1.0, -2.0]])
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
        d6_patch, d7_patch = _identity_standardization()
        with d6_patch, d7_patch:
            result_a = self.criterion(
                _outputs(z, _negative_native()), [target_a]
            )
            result_b = self.criterion(
                _outputs(z, _negative_native()), [target_b]
            )

        self.assertTrue(
            torch.equal(
                result_a[NATIVE_PATCH_CATEGORY_D7_LOSS],
                result_b[NATIVE_PATCH_CATEGORY_D7_LOSS],
            )
        )
        self.assertEqual(
            float(result_a["stage_b_native_patch_category_d7_critical_rows"]),
            1.0,
        )
        source = inspect.getsource(StageBNativePatchCategoryD7Criterion.forward)
        self.assertNotIn("primary_instance_mask", source)
        self.assertNotIn("source_dataset", source)
        self.assertNotIn("variant", source)

    def test_contract_defaults_geometry_and_build_route_are_explicit(self):
        criterion = StageBNativePatchCategoryD7Criterion()
        self.assertEqual(NATIVE_PATCH_CATEGORY_D7_CONTRACT_VERSION, 7)
        self.assertEqual(NATIVE_PATCH_CATEGORY_D7_MARKER, NATIVE_PATCH_CATEGORY_D6_MARKER)
        self.assertEqual(criterion.anchor_active_gap, 2.0)
        self.assertEqual(criterion.anchor_target_gap, 2.5)
        self.assertEqual(criterion.anchor_weight, 2.0)
        self.assertEqual(
            criterion.weight_dict, {NATIVE_PATCH_CATEGORY_D7_LOSS: 1.0}
        )
        with self.assertRaisesRegex(ValueError, "anchor geometry"):
            StageBNativePatchCategoryD7Criterion(anchor_active_gap=2.5)
        with self.assertRaisesRegex(ValueError, "anchor geometry"):
            StageBNativePatchCategoryD7Criterion(anchor_weight=0.0)

        source = inspect.getsource(build_groundingdino)
        self.assertIn(
            'native_patch_objective == "d7_all_state_positive_anchor"', source
        )
        self.assertIn("StageBNativePatchCategoryD7Criterion", source)
        for field in (
            "stage_b_native_patch_d7_weight",
            "stage_b_native_patch_d7_keep_gap",
            "stage_b_native_patch_d7_drop_gap",
            "stage_b_native_patch_d7_drop_active_gap",
            "stage_b_native_patch_d7_temperature",
            "stage_b_native_patch_d7_drop_weight",
            "stage_b_native_patch_d7_critical_keep_weight",
            "stage_b_native_patch_d7_positive_active_gap",
            "stage_b_native_patch_d7_positive_target_gap",
            "stage_b_native_patch_d7_positive_barrier_weight",
            "stage_b_native_patch_d7_anchor_active_gap",
            "stage_b_native_patch_d7_anchor_target_gap",
            "stage_b_native_patch_d7_anchor_weight",
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
