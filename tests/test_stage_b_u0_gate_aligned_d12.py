import unittest

import torch

from models.GroundingDINO.stage_b_u0_gate_aligned_d11 import (
    STAGE_B_U0_GATE_ALIGNED_D11_MARKER,
)
from models.GroundingDINO.stage_b_u0_gate_aligned_d12 import (
    STAGE_B_U0_GATE_ALIGNED_D12_CONTRACT_VERSION,
    STAGE_B_U0_GATE_ALIGNED_D12_LOSS,
    StageBU0GateAlignedD12Criterion,
    StageBU0GateAlignedD12RankResidual,
)


def _target():
    return {
        "boxes": torch.tensor(
            [[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([7, 7], dtype=torch.int64),
        "support_class": torch.tensor([7], dtype=torch.int64),
        "primary_instance_mask": torch.tensor([True, False]),
        STAGE_B_U0_GATE_ALIGNED_D11_MARKER: torch.tensor([True]),
    }


_BOXES = torch.tensor(
    [
        [
            [0.50, 0.10, 0.10, 0.10],
            [0.25, 0.50, 0.20, 0.20],
            [0.75, 0.50, 0.20, 0.20],
            [0.50, 0.90, 0.10, 0.10],
        ]
    ],
    dtype=torch.float32,
)
_ELIGIBLE = torch.tensor([[True, True, True, False]])


def _outputs(adapted, teacher, residual):
    return {
        "stage_b_u0_d12_rank_score": adapted,
        "stage_b_u0_d12_teacher_rank_score": teacher,
        "stage_b_u0_d12_rank_residual": residual,
        "stage_b_u0_category_gate_eligible_mask": _ELIGIBLE,
        "pred_boxes": _BOXES,
    }


class StageBU0GateAlignedD12Test(unittest.TestCase):
    def test_zero_initialization_is_bitwise_teacher_exact(self):
        adapter = StageBU0GateAlignedD12RankResidual(
            feature_dim=4, hidden_dim=8
        )
        feature = torch.randn(2, 5, 4)
        teacher = torch.randn(2, 5)
        result = adapter(feature, teacher)
        self.assertEqual(
            int(adapter.contract_version.item()),
            STAGE_B_U0_GATE_ALIGNED_D12_CONTRACT_VERSION,
        )
        self.assertTrue(torch.equal(result["rank_score"], teacher))
        self.assertTrue(
            torch.equal(result["rank_residual"], torch.zeros_like(teacher))
        )

    def test_fix_loss_has_no_teacher_gradient(self):
        teacher = torch.tensor([[3.0, 2.0, 1.0, 4.0]], requires_grad=True)
        residual = torch.zeros_like(teacher, requires_grad=True)
        adapted = teacher.detach() + residual
        criterion = StageBU0GateAlignedD12Criterion()
        result = criterion(_outputs(adapted, teacher, residual), [_target()])
        self.assertEqual(
            criterion.weight_dict,
            {STAGE_B_U0_GATE_ALIGNED_D12_LOSS: 1.0},
        )
        self.assertEqual(int(result["stage_b_u0_gate_aligned_d12_fix_rows"]), 1)
        result[STAGE_B_U0_GATE_ALIGNED_D12_LOSS].backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(residual.grad[0, 0]), 0.0)
        self.assertLess(float(residual.grad[0, 1]), 0.0)
        self.assertEqual(float(residual.grad[0, 3]), 0.0)

    def test_frozen_teacher_correct_margin_detects_regression(self):
        teacher = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
        adapted = torch.tensor([[3.0, 2.0, 1.0, 4.0]], requires_grad=True)
        residual = adapted - teacher
        result = StageBU0GateAlignedD12Criterion()(
            _outputs(adapted, teacher, residual), [_target()]
        )
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d12_correct_regressed"]), 1
        )
        self.assertGreater(
            float(result["stage_b_u0_gate_aligned_d12_preserve_loss"]), 0.0
        )

    def test_conditional_residual_can_repair_a_wrong_winner(self):
        torch.manual_seed(4)
        adapter = StageBU0GateAlignedD12RankResidual(
            feature_dim=4, hidden_dim=8, residual_limit=0.5
        )
        feature = torch.randn(1, 4, 4)
        teacher = torch.tensor([[0.10, 0.00, -0.10, 1.0]])
        criterion = StageBU0GateAlignedD12Criterion(
            fix_margin=0.05, residual_weight=0.0
        )
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.03)
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            rank = adapter(feature, teacher)
            loss = criterion(
                _outputs(
                    rank["rank_score"],
                    rank["teacher_rank_score"],
                    rank["rank_residual"],
                ),
                [_target()],
            )[STAGE_B_U0_GATE_ALIGNED_D12_LOSS]
            loss.backward()
            optimizer.step()
        rank = adapter(feature, teacher)["rank_score"].detach()
        winner = int(rank.masked_fill(~_ELIGIBLE, -torch.inf).argmax())
        self.assertEqual(winner, 1)


if __name__ == "__main__":
    unittest.main()
