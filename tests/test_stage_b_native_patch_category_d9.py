import inspect
import unittest

import torch

from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO.stage_b_native_patch_category_d2 import (
    gate_aligned_standardized_patch_score,
)
from models.GroundingDINO.stage_b_native_patch_category_d8 import (
    NATIVE_PATCH_CATEGORY_D8_LOSS,
    StageBNativePatchCategoryD8Criterion,
)
from models.GroundingDINO.stage_b_native_patch_category_d9 import (
    NATIVE_PATCH_CATEGORY_D9_CONTRACT_VERSION,
    NATIVE_PATCH_CATEGORY_D9_LOSS,
    NATIVE_PATCH_CATEGORY_D9_MARKER,
    StageBNativePatchCategoryD9Criterion,
    loss_gradient_localized_standardized_patch_score,
)


def _target(class_id: int = 7):
    return {
        "boxes": torch.tensor(
            [[0.20, 0.50, 0.12, 0.20], [0.80, 0.50, 0.12, 0.20]],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([class_id, class_id], dtype=torch.int64),
        "support_class": torch.tensor([class_id], dtype=torch.int64),
        NATIVE_PATCH_CATEGORY_D9_MARKER: torch.tensor([True]),
    }


def _outputs(patch_score: torch.Tensor, *, selector_grads: bool = False):
    query_count = int(patch_score.shape[1])
    boxes = torch.empty(1, query_count, 4, dtype=torch.float32)
    boxes[:, 0] = torch.tensor([0.20, 0.50, 0.12, 0.20])
    boxes[:, 1] = torch.tensor([0.80, 0.50, 0.12, 0.20])
    boxes[:, 2:] = torch.tensor([0.50, 0.08, 0.04, 0.04])
    boxes.requires_grad_(selector_grads)
    native = torch.tensor([[4.0, 4.5, 5.0, 1.0, 0.0, -1.0]])
    token_logits = native[:, :, None].expand(1, query_count, 3).clone()
    token_logits.requires_grad_(selector_grads)
    return {
        "pred_logits_patch": patch_score,
        "pred_logits_text": token_logits,
        "pred_boxes": boxes,
        "phrase_to_token_mask": torch.tensor(
            [[[True, True, False]]], dtype=torch.bool
        ),
    }


class StageBNativePatchCategoryD9Test(unittest.TestCase):
    def test_localized_standardization_is_bitwise_forward_equal(self):
        generator = torch.Generator().manual_seed(20260725)
        for shape in ((4, 17), (3, 29, 1)):
            with self.subTest(shape=shape):
                score = torch.randn(shape, generator=generator)
                mask_shape = shape[:2]
                mask = torch.rand(mask_shape, generator=generator) > 0.2
                mask[:, 0] = True
                expected = gate_aligned_standardized_patch_score(
                    score, mask, clip=5.0
                )
                observed = loss_gradient_localized_standardized_patch_score(
                    score, mask, clip=5.0
                )
                self.assertTrue(torch.equal(observed, expected))

    def test_localized_jacobian_has_no_cross_query_gradient(self):
        score = torch.tensor(
            [[-1.1, 0.4, 1.2, -0.7, 0.9, 0.1]],
            requires_grad=True,
        )
        mask = torch.tensor([[True, True, True, True, True, False]])
        standardized = loss_gradient_localized_standardized_patch_score(
            score, mask, clip=5.0
        )
        (standardized[0, 1] + 2.0 * standardized[0, 3]).backward()
        nonzero = set(
            torch.nonzero(score.grad[0], as_tuple=False).flatten().tolist()
        )
        self.assertEqual(nonzero, {1, 3})
        self.assertEqual(float(score.grad[0, 5]), 0.0)

        dense_score = score.detach().clone().requires_grad_(True)
        dense = gate_aligned_standardized_patch_score(
            dense_score, mask, clip=5.0
        )
        dense[0, 1].backward()
        dense_nonzero = set(
            torch.nonzero(dense_score.grad[0], as_tuple=False)
            .flatten()
            .tolist()
        )
        self.assertGreater(len(dense_nonzero), 1)

    def test_full_d9_forward_equals_d8_and_renames_all_outputs(self):
        patch_score = torch.tensor(
            [[0.0, -1.0, -0.5, 3.0, -2.0, -2.5]],
            dtype=torch.float32,
        )
        outputs = _outputs(patch_score)
        target = _target()
        d8 = StageBNativePatchCategoryD8Criterion()(outputs, [target])
        d9 = StageBNativePatchCategoryD9Criterion()(outputs, [target])

        for d8_name, d8_value in d8.items():
            d9_name = (
                NATIVE_PATCH_CATEGORY_D9_LOSS
                if d8_name == NATIVE_PATCH_CATEGORY_D8_LOSS
                else d8_name.replace(
                    "stage_b_native_patch_category_d8_",
                    "stage_b_native_patch_category_d9_",
                    1,
                )
            )
            self.assertIn(d9_name, d9)
            self.assertTrue(torch.equal(d9[d9_name], d8_value), d9_name)
        self.assertEqual(len(d9), len(d8))

    def test_full_loss_gradient_is_limited_to_selected_queries(self):
        patch_score = torch.tensor(
            [[0.0, -1.0, -0.5, 3.0, -2.0, -2.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        outputs = _outputs(patch_score, selector_grads=True)
        target = _target()
        target["boxes"].requires_grad_(True)
        result = StageBNativePatchCategoryD9Criterion()(outputs, [target])
        result[NATIVE_PATCH_CATEGORY_D9_LOSS].backward()

        nonzero = set(
            torch.nonzero(patch_score.grad[0], as_tuple=False)
            .flatten()
            .tolist()
        )
        # q1 is the native-best category-positive anchor; q2 is the native
        # negative winner. q3 is the detached row maximum.
        self.assertEqual(nonzero, {1, 2})
        self.assertIsNone(outputs["pred_logits_text"].grad)
        self.assertIsNone(outputs["pred_boxes"].grad)
        self.assertIsNone(target["boxes"].grad)

    def test_contract_and_builder_route_are_explicit(self):
        criterion = StageBNativePatchCategoryD9Criterion()
        self.assertEqual(NATIVE_PATCH_CATEGORY_D9_CONTRACT_VERSION, 9)
        self.assertTrue(criterion.detach_row_stats)
        self.assertEqual(
            criterion.weight_dict, {NATIVE_PATCH_CATEGORY_D9_LOSS: 1.0}
        )
        with self.assertRaisesRegex(ValueError, "detach_row_stats=True"):
            StageBNativePatchCategoryD9Criterion(detach_row_stats=False)

        source = inspect.getsource(build_groundingdino)
        self.assertIn(
            'native_patch_objective == "d9_loss_gradient_localized"', source
        )
        self.assertIn("StageBNativePatchCategoryD9Criterion", source)
        self.assertIn("stage_b_native_patch_d9_detach_row_stats", source)

        forward_source = inspect.getsource(
            StageBNativePatchCategoryD9Criterion.forward
        )
        self.assertNotIn("primary_instance_mask", forward_source)
        self.assertNotIn("source_dataset", forward_source)
        self.assertNotIn("variant", forward_source)


if __name__ == "__main__":
    unittest.main()
