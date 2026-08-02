import inspect
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO import stage_b_native_patch_category_d6 as d6_module
from models.GroundingDINO import stage_b_native_patch_category_d8 as d8_module
from models.GroundingDINO.stage_b_native_patch_category_d6 import (
    NATIVE_PATCH_CATEGORY_D6_LOSS,
    NATIVE_PATCH_CATEGORY_D6_MARKER,
    StageBNativePatchCategoryD6Criterion,
)
from models.GroundingDINO.stage_b_native_patch_category_d8 import (
    NATIVE_PATCH_CATEGORY_D8_CONTRACT_VERSION,
    NATIVE_PATCH_CATEGORY_D8_LOSS,
    NATIVE_PATCH_CATEGORY_D8_MARKER,
    StageBNativePatchCategoryD8Criterion,
)


def _target(class_id=7, *, reverse=False, primary=None, other_class=False):
    boxes = torch.tensor(
        [[0.20, 0.50, 0.12, 0.20], [0.80, 0.50, 0.12, 0.20]],
        dtype=torch.float32,
    )
    labels = torch.tensor([class_id, class_id], dtype=torch.int64)
    if other_class:
        boxes = torch.cat(
            (boxes, torch.tensor([[0.50, 0.08, 0.04, 0.04]])), dim=0
        )
        labels = torch.tensor(
            [class_id, class_id, class_id + 1000], dtype=torch.int64
        )
    if reverse:
        order = torch.arange(boxes.shape[0] - 1, -1, -1)
        boxes = boxes[order]
        labels = labels[order]
    target = {
        "boxes": boxes,
        "labels": labels,
        "support_class": torch.tensor([class_id], dtype=torch.int64),
        NATIVE_PATCH_CATEGORY_D8_MARKER: torch.tensor([True]),
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


def _z_for_gap(gap, *, positive_state=False):
    z = torch.tensor([[3.0, 5.0 - gap, 1.6, 5.0, -1.0, -2.0]])
    if positive_state:
        z[0, 0] = 5.0 - gap
    return z


def _identity_standardization():
    identity = lambda patch_score, candidate_mask, *, clip: patch_score
    return (
        mock.patch.object(
            d6_module,
            "gate_aligned_standardized_patch_score",
            side_effect=identity,
        ),
        mock.patch.object(
            d8_module,
            "gate_aligned_standardized_patch_score",
            side_effect=identity,
        ),
    )


class StageBNativePatchCategoryD8CriterionTest(unittest.TestCase):
    def setUp(self):
        self.criterion = StageBNativePatchCategoryD8Criterion()

    def test_state_then_class_macro_algebra_and_fixed_weights(self):
        specs = (
            ("negative", 7, 2.25),
            ("negative", 7, 3.00),
            ("negative", 8, 2.50),
            ("neutral", 7, 2.75),
            ("neutral", 9, 3.25),
            ("neutral", 9, 2.25),
            ("positive", 7, 2.50),
        )
        native_by_state = {
            "negative": _negative_native,
            "neutral": _neutral_native,
            "positive": _positive_native,
        }
        z = torch.cat(
            tuple(
                _z_for_gap(gap, positive_state=state == "positive")
                for state, _, gap in specs
            ),
            dim=0,
        )
        native = torch.cat(
            tuple(native_by_state[state]() for state, _, _ in specs), dim=0
        )
        targets = [_target(class_id) for _, class_id, _ in specs]
        d6_patch, d8_patch = _identity_standardization()
        with d6_patch, d8_patch:
            outputs = _outputs(z, native)
            base = StageBNativePatchCategoryD6Criterion()(outputs, targets)
            result = self.criterion(outputs, targets)

        terms = {
            (state, class_id, gap): F.softplus(torch.tensor((gap - 2.5) / 0.25))
            for state, class_id, gap in specs
        }
        negative = (
            (terms[("negative", 7, 2.25)] + terms[("negative", 7, 3.0)])
            / 2.0
            + terms[("negative", 8, 2.5)]
        ) / 2.0
        neutral = (
            terms[("neutral", 7, 2.75)]
            + (
                terms[("neutral", 9, 3.25)]
                + terms[("neutral", 9, 2.25)]
            )
            / 2.0
        ) / 2.0
        positive = terms[("positive", 7, 2.5)]
        expected = (
            base[NATIVE_PATCH_CATEGORY_D6_LOSS]
            + negative
            + 2.0 * neutral
            + 4.0 * positive
        )

        self.assertTrue(
            torch.allclose(
                result[NATIVE_PATCH_CATEGORY_D8_LOSS],
                expected,
                atol=1e-7,
                rtol=0.0,
            )
        )
        for state, expected_loss, rows, classes in (
            ("negative", negative, 3.0, 2.0),
            ("neutral", neutral, 3.0, 2.0),
            ("positive", positive, 1.0, 1.0),
        ):
            prefix = f"stage_b_native_patch_category_d8_anchor_{state}_"
            self.assertTrue(
                torch.allclose(
                    result[prefix + "loss"],
                    expected_loss,
                    atol=1e-7,
                    rtol=0.0,
                )
            )
            self.assertEqual(float(result[prefix + "active_rows"]), rows)
            self.assertEqual(float(result[prefix + "active_classes"]), classes)

        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d8_anchor_negative_deployment_keep"
                ]
            ),
            3.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d8_anchor_neutral_deployment_keep"
                ]
            ),
            2.0,
        )

    def test_empty_states_are_zero_without_present_state_renormalization(self):
        gap = 2.5
        z = _z_for_gap(gap)
        d6_patch, d8_patch = _identity_standardization()
        with d6_patch, d8_patch:
            result = self.criterion(
                _outputs(z, _neutral_native()), [_target(7)]
            )
        anchor = F.softplus(torch.tensor((gap - 2.5) / 0.25))
        # The only present state keeps its fixed weight 2; there is no /1 or /3.
        self.assertTrue(
            torch.allclose(
                result[NATIVE_PATCH_CATEGORY_D8_LOSS],
                2.0 * anchor,
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d8_anchor_negative_loss"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_native_patch_category_d8_anchor_positive_loss"
                ]
            ),
            0.0,
        )

    def test_duplicating_rows_inside_one_class_does_not_change_macro(self):
        row_a = _z_for_gap(2.25)
        row_b = _z_for_gap(3.0)
        d6_patch, d8_patch = _identity_standardization()
        with d6_patch, d8_patch:
            baseline = self.criterion(
                _outputs(
                    torch.cat((row_a, row_b)),
                    _neutral_native().expand(2, -1).clone(),
                ),
                [_target(7), _target(8)],
            )
            duplicated = self.criterion(
                _outputs(
                    torch.cat((row_a.expand(4, -1), row_b)),
                    _neutral_native().expand(5, -1).clone(),
                ),
                [_target(7) for _ in range(4)] + [_target(8)],
            )

        key = "stage_b_native_patch_category_d8_anchor_neutral_loss"
        self.assertTrue(torch.equal(baseline[key], duplicated[key]))
        self.assertTrue(
            torch.equal(
                baseline[NATIVE_PATCH_CATEGORY_D8_LOSS],
                duplicated[NATIVE_PATCH_CATEGORY_D8_LOSS],
            )
        )
        self.assertEqual(
            float(
                duplicated[
                    "stage_b_native_patch_category_d8_anchor_neutral_active_rows"
                ]
            ),
            5.0,
        )
        self.assertEqual(
            float(
                duplicated[
                    "stage_b_native_patch_category_d8_anchor_neutral_active_classes"
                ]
            ),
            2.0,
        )

    def test_d6_terms_are_reused_and_d7_pool_is_not_called(self):
        z = torch.cat((_z_for_gap(2.5), _z_for_gap(2.75)), dim=0)
        native = torch.cat((_negative_native(), _neutral_native()), dim=0)
        targets = [_target(7), _target(8)]
        d6_patch, d8_patch = _identity_standardization()
        with d6_patch, d8_patch:
            outputs = _outputs(z, native)
            d6_result = StageBNativePatchCategoryD6Criterion()(outputs, targets)
            d8_result = self.criterion(outputs, targets)

        for name, value in d6_result.items():
            if name == NATIVE_PATCH_CATEGORY_D6_LOSS:
                continue
            d8_name = name.replace(
                "stage_b_native_patch_category_d6_",
                "stage_b_native_patch_category_d8_",
                1,
            )
            self.assertTrue(torch.equal(d8_result[d8_name], value), d8_name)
        source = inspect.getsource(d8_module)
        self.assertIn("super().forward", source)
        self.assertNotIn("stage_b_native_patch_category_d7", source)
        self.assertNotIn("StageBNativePatchCategoryD7Criterion", source)

    def test_anchor_gradient_only_reaches_patch(self):
        z = _z_for_gap(3.0).requires_grad_(True)
        outputs = _outputs(z, _neutral_native(), selector_grads=True)
        target = _target(7)
        target["boxes"].requires_grad_(True)
        d6_patch, d8_patch = _identity_standardization()
        with d6_patch, d8_patch:
            result = self.criterion(outputs, [target])
        result[NATIVE_PATCH_CATEGORY_D8_LOSS].backward()

        self.assertLess(float(z.grad[0, 1]), 0.0)
        self.assertEqual(float(z.grad[0, 3]), 0.0)
        self.assertIsNone(outputs["pred_logits_text"].grad)
        self.assertIsNone(outputs["pred_boxes"].grad)
        self.assertIsNone(target["boxes"].grad)

    def test_metadata_and_gt_order_are_ignored_and_empty_all_is_differentiable(self):
        z = _z_for_gap(3.0)
        target_a = _target(
            7,
            primary=torch.tensor([True, False, False]),
            other_class=True,
        )
        target_b = _target(
            7,
            reverse=True,
            primary=torch.tensor([False, False, False]),
            other_class=True,
        )
        target_a["source_dataset"] = object()
        target_a["stage_b_native_patch_category_variant"] = object()
        target_b["source_dataset"] = "different"
        target_b["stage_b_native_patch_category_variant"] = "different"
        d6_patch, d8_patch = _identity_standardization()
        with d6_patch, d8_patch:
            result_a = self.criterion(
                _outputs(z, _negative_native()), [target_a]
            )
            result_b = self.criterion(
                _outputs(z, _negative_native()), [target_b]
            )
        self.assertTrue(
            torch.equal(
                result_a[NATIVE_PATCH_CATEGORY_D8_LOSS],
                result_b[NATIVE_PATCH_CATEGORY_D8_LOSS],
            )
        )

        empty_z = _z_for_gap(3.0).requires_grad_(True)
        empty_outputs = _outputs(empty_z, _neutral_native())
        empty_outputs["pred_boxes"] = torch.tensor(
            [[[0.50, 0.08, 0.04, 0.04]] * 6], dtype=torch.float32
        )
        d6_patch, d8_patch = _identity_standardization()
        with d6_patch, d8_patch:
            empty = self.criterion(empty_outputs, [_target(7)])
        self.assertEqual(float(empty[NATIVE_PATCH_CATEGORY_D8_LOSS].detach()), 0.0)
        empty[NATIVE_PATCH_CATEGORY_D8_LOSS].backward()
        self.assertIsNotNone(empty_z.grad)
        self.assertEqual(float(empty_z.grad.abs().sum()), 0.0)

        source = inspect.getsource(StageBNativePatchCategoryD8Criterion.forward)
        self.assertNotIn("primary_instance_mask", source)
        self.assertNotIn("source_dataset", source)
        self.assertNotIn("variant", source)

    def test_contract_defaults_geometry_and_build_route_are_explicit(self):
        criterion = StageBNativePatchCategoryD8Criterion()
        self.assertEqual(NATIVE_PATCH_CATEGORY_D8_CONTRACT_VERSION, 8)
        self.assertEqual(NATIVE_PATCH_CATEGORY_D8_MARKER, NATIVE_PATCH_CATEGORY_D6_MARKER)
        self.assertEqual(criterion.anchor_active_gap, 2.0)
        self.assertEqual(criterion.anchor_target_gap, 2.5)
        self.assertEqual(criterion.negative_weight, 1.0)
        self.assertEqual(criterion.neutral_weight, 2.0)
        self.assertEqual(criterion.positive_weight, 4.0)
        self.assertEqual(
            criterion.weight_dict, {NATIVE_PATCH_CATEGORY_D8_LOSS: 1.0}
        )
        with self.assertRaisesRegex(ValueError, "anchor geometry"):
            StageBNativePatchCategoryD8Criterion(anchor_active_gap=2.5)
        with self.assertRaisesRegex(ValueError, "anchor geometry"):
            StageBNativePatchCategoryD8Criterion(neutral_weight=0.0)

        source = inspect.getsource(build_groundingdino)
        self.assertIn(
            'native_patch_objective == "d8_state_class_macro_anchor"', source
        )
        self.assertIn("StageBNativePatchCategoryD8Criterion", source)
        for field in (
            "stage_b_native_patch_d8_weight",
            "stage_b_native_patch_d8_keep_gap",
            "stage_b_native_patch_d8_drop_gap",
            "stage_b_native_patch_d8_drop_active_gap",
            "stage_b_native_patch_d8_temperature",
            "stage_b_native_patch_d8_drop_weight",
            "stage_b_native_patch_d8_critical_keep_weight",
            "stage_b_native_patch_d8_positive_active_gap",
            "stage_b_native_patch_d8_positive_target_gap",
            "stage_b_native_patch_d8_positive_barrier_weight",
            "stage_b_native_patch_d8_anchor_active_gap",
            "stage_b_native_patch_d8_anchor_target_gap",
            "stage_b_native_patch_d8_anchor_negative_weight",
            "stage_b_native_patch_d8_anchor_neutral_weight",
            "stage_b_native_patch_d8_anchor_positive_weight",
        ):
            self.assertIn(field, source)
        for obsolete_field in (
            "stage_b_native_patch_d8_negative_weight",
            "stage_b_native_patch_d8_neutral_weight",
            "stage_b_native_patch_d8_positive_weight",
        ):
            self.assertNotIn(obsolete_field, source)


if __name__ == "__main__":
    unittest.main()
