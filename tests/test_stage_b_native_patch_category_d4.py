import inspect
import unittest

import torch

from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO.stage_b_native_patch_category_d3 import (
    NATIVE_PATCH_CATEGORY_D3_LOSS,
    StageBNativePatchCategoryD3Criterion,
)
from models.GroundingDINO.stage_b_native_patch_category_d4 import (
    NATIVE_PATCH_CATEGORY_D4_CONTRACT_VERSION,
    NATIVE_PATCH_CATEGORY_D4_LOSS,
    NATIVE_PATCH_CATEGORY_D4_MARKER,
    StageBNativePatchCategoryD4Criterion,
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
        NATIVE_PATCH_CATEGORY_D4_MARKER: torch.tensor([True]),
    }


def _outputs(patch_score, native_winners, *, require_selector_grads=False):
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


class StageBNativePatchCategoryD4CriterionTest(unittest.TestCase):
    def test_formula_is_d3_with_only_positive_keep_scaled_by_32(self):
        patch = torch.tensor(
            [
                [1.0, 0.0, 3.0, 2.0, -1.0, -2.0, -3.0, -4.0],
                [1.0, 0.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
            ],
            dtype=torch.float32,
        )
        native = torch.cat(
            (_critical_native(8), _positive_native(8)), dim=0
        )
        outputs = _outputs(patch, native)
        targets = [_target(), _target()]

        d3 = StageBNativePatchCategoryD3Criterion()(outputs, targets)
        d4 = StageBNativePatchCategoryD4Criterion()(outputs, targets)
        expected = (
            d3[NATIVE_PATCH_CATEGORY_D3_LOSS]
            + 31.0
            * d3["stage_b_native_patch_category_d3_positive_keep_loss"]
        )

        self.assertTrue(
            torch.allclose(
                d4[NATIVE_PATCH_CATEGORY_D4_LOSS],
                expected,
                atol=1e-6,
                rtol=0.0,
            )
        )
        self.assertFalse(any("_d3_" in name for name in d4))
        self.assertIn(
            "stage_b_native_patch_category_d4_positive_keep_loss", d4
        )
        for d3_name, d3_value in d3.items():
            if d3_name == NATIVE_PATCH_CATEGORY_D3_LOSS:
                continue
            d4_name = d3_name.replace(
                "stage_b_native_patch_category_d3_",
                "stage_b_native_patch_category_d4_",
                1,
            )
            self.assertTrue(torch.equal(d4[d4_name], d3_value), d4_name)
        source = inspect.getsource(StageBNativePatchCategoryD4Criterion.forward)
        self.assertIn("super().forward", source)
        self.assertNotIn("softplus", source)

    def test_gradient_isolated_to_patch_score(self):
        patch = torch.tensor(
            [
                [1.0, 0.0, 3.0, 2.0, -1.0, -2.0, -3.0, -4.0],
                [1.0, 0.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        native = torch.cat(
            (_critical_native(8), _positive_native(8)), dim=0
        )
        outputs = _outputs(patch, native, require_selector_grads=True)
        result = StageBNativePatchCategoryD4Criterion()(
            outputs, [_target(), _target()]
        )
        result[NATIVE_PATCH_CATEGORY_D4_LOSS].backward()

        self.assertIsNotNone(patch.grad)
        self.assertGreater(float(patch.grad.abs().sum()), 0.0)
        self.assertIsNone(outputs["pred_logits_text"].grad)
        self.assertIsNone(outputs["pred_boxes"].grad)

    def test_contract_defaults_and_build_route_are_d4_specific(self):
        criterion = StageBNativePatchCategoryD4Criterion()
        self.assertEqual(NATIVE_PATCH_CATEGORY_D4_CONTRACT_VERSION, 4)
        self.assertEqual(criterion.critical_weight, 2.0)
        self.assertEqual(criterion.critical_keep_weight, 1.0)
        self.assertEqual(criterion.positive_keep_weight, 32.0)
        self.assertEqual(
            criterion.weight_dict,
            {NATIVE_PATCH_CATEGORY_D4_LOSS: 1.0},
        )

        source = inspect.getsource(build_groundingdino)
        self.assertIn(
            '== "d4_positive_protected_critical_winner"', source
        )
        self.assertIn("StageBNativePatchCategoryD4Criterion", source)
        for field in (
            "stage_b_native_patch_d4_weight",
            "stage_b_native_patch_d4_keep_gap",
            "stage_b_native_patch_d4_separation_gap",
            "stage_b_native_patch_d4_temperature",
            "stage_b_native_patch_d4_critical_weight",
            "stage_b_native_patch_d4_critical_keep_weight",
            "stage_b_native_patch_d4_positive_keep_weight",
        ):
            self.assertIn(field, source)

    def test_d3_defaults_and_outputs_remain_unchanged(self):
        d3 = StageBNativePatchCategoryD3Criterion()
        patch = torch.tensor(
            [[1.0, 0.0, 3.0, 2.0, -1.0, -2.0, -3.0, -4.0]],
            dtype=torch.float32,
        )
        result = d3(_outputs(patch, _critical_native(8)), [_target()])

        self.assertEqual(d3.critical_weight, 2.0)
        self.assertEqual(d3.critical_keep_weight, 1.0)
        self.assertEqual(d3.positive_keep_weight, 1.0)
        self.assertEqual(
            d3.weight_dict, {NATIVE_PATCH_CATEGORY_D3_LOSS: 1.0}
        )
        self.assertIn(NATIVE_PATCH_CATEGORY_D3_LOSS, result)
        self.assertTrue(all("_d4_" not in name for name in result))


if __name__ == "__main__":
    unittest.main()
