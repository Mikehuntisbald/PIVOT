import unittest

import torch

from models.GroundingDINO.stage_b_data_driven_score import (
    data_driven_category_gate_mask,
)
from models.GroundingDINO.stage_b_native_patch_category_d9 import (
    loss_gradient_localized_standardized_patch_score,
)
from models.GroundingDINO.stage_b_u0_gate_aligned_d10 import (
    STAGE_B_U0_GATE_ALIGNED_D10_CONTRACT_VERSION,
    STAGE_B_U0_GATE_ALIGNED_D10_LOSS,
    STAGE_B_U0_GATE_ALIGNED_D10_MARKER,
    StageBU0GateAlignedD10Criterion,
)
from models.GroundingDINO.stage_b_u0_patch_rank import StageBU0PatchRankAdapter


def _target(two_instances: bool = True):
    boxes = torch.tensor(
        [[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]],
        dtype=torch.float32,
    )
    if not two_instances:
        boxes = boxes[:1]
    primary = torch.zeros((len(boxes),), dtype=torch.bool)
    primary[0] = True
    return {
        "boxes": boxes,
        "labels": torch.full((len(boxes),), 7, dtype=torch.int64),
        "support_class": torch.tensor([7], dtype=torch.int64),
        "primary_instance_mask": primary,
        STAGE_B_U0_GATE_ALIGNED_D10_MARKER: torch.tensor([True]),
    }


def _outputs(patch_score: torch.Tensor, rank_score: torch.Tensor):
    return {
        "pred_logits_patch": patch_score,
        "pred_boxes": torch.tensor(
            [
                [
                    [0.50, 0.10, 0.10, 0.10],
                    [0.25, 0.50, 0.20, 0.20],
                    [0.75, 0.50, 0.20, 0.20],
                    [0.50, 0.90, 0.10, 0.10],
                ]
            ],
            dtype=torch.float32,
        ),
        "stage_b_u0_teacher_rank_score": rank_score,
        "stage_b_u0_candidate_mask": torch.ones((1, 4), dtype=torch.bool),
    }


class StageBU0GateAlignedD10Test(unittest.TestCase):
    def test_contract_and_gradient_ownership(self):
        patch = torch.tensor(
            [[2.0, 0.0, -1.0, -2.0]], requires_grad=True
        )
        rank = torch.tensor(
            [[4.0, 3.0, 2.0, 1.0]], requires_grad=True
        )
        criterion = StageBU0GateAlignedD10Criterion()
        result = criterion(_outputs(patch, rank), [_target()])
        self.assertEqual(
            int(criterion.criterion_contract_version.item()),
            STAGE_B_U0_GATE_ALIGNED_D10_CONTRACT_VERSION,
        )
        self.assertEqual(
            criterion.weight_dict,
            {STAGE_B_U0_GATE_ALIGNED_D10_LOSS: 1.0},
        )
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d10_deployed_negative_rows"]),
            1,
        )
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d10_reachable_instances"]),
            2,
        )
        result[STAGE_B_U0_GATE_ALIGNED_D10_LOSS].backward()
        self.assertIsNone(rank.grad)
        self.assertGreater(float(patch.grad[0, 0]), 0.0)
        self.assertLess(float(patch.grad[0, 1]), 0.0)
        self.assertLess(float(patch.grad[0, 2]), 0.0)

    def test_loss_standardization_matches_deployment_forward(self):
        patch = torch.tensor([[8.0, 2.0, -1.0, -9.0]], requires_grad=True)
        mask = torch.ones_like(patch, dtype=torch.bool)
        loss_score = loss_gradient_localized_standardized_patch_score(
            patch, mask, clip=5.0
        )
        adapter_score = StageBU0PatchRankAdapter._standardize(
            patch, mask, clip=5.0
        )
        _eligible, deployed_score = data_driven_category_gate_mask(
            patch.detach(), mask, max_gap=2.0, clip=5.0
        )
        self.assertTrue(torch.equal(loss_score.detach(), adapter_score.detach()))
        self.assertTrue(torch.equal(loss_score.detach(), deployed_score))

    def test_small_tensor_optimization_corrects_deployed_category_winner(self):
        patch = torch.nn.Parameter(torch.tensor([[2.0, 0.0, -2.0, -3.0]]))
        rank = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        criterion = StageBU0GateAlignedD10Criterion(
            instance_coverage_weight=1.0
        )
        optimizer = torch.optim.Adam([patch], lr=0.05)
        for _ in range(160):
            optimizer.zero_grad(set_to_none=True)
            result = criterion(_outputs(patch, rank), [_target(False)])
            result[STAGE_B_U0_GATE_ALIGNED_D10_LOSS].backward()
            optimizer.step()
        eligible, _ = data_driven_category_gate_mask(
            patch.detach(),
            torch.ones_like(patch, dtype=torch.bool),
            max_gap=2.0,
            clip=5.0,
        )
        winner = int(rank.masked_fill(~eligible, -torch.inf).argmax(dim=1)[0])
        self.assertEqual(winner, 1)

    def test_rejects_missing_category_complete_marker(self):
        target = _target()
        target.pop(STAGE_B_U0_GATE_ALIGNED_D10_MARKER)
        patch = torch.tensor([[2.0, 0.0, -1.0, -2.0]])
        rank = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "category-complete data marker"):
            StageBU0GateAlignedD10Criterion()(
                _outputs(patch, rank), [target]
            )

    def test_zero_instance_coverage_weight_is_a_valid_ablation(self):
        criterion = StageBU0GateAlignedD10Criterion(
            instance_coverage_weight=0.0
        )
        patch = torch.tensor(
            [[2.0, 0.0, -1.0, -2.0]], requires_grad=True
        )
        rank = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        loss = criterion(_outputs(patch, rank), [_target()])[
            STAGE_B_U0_GATE_ALIGNED_D10_LOSS
        ]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(patch.grad)


if __name__ == "__main__":
    unittest.main()
