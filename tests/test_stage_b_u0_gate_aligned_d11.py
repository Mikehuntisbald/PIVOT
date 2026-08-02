import unittest

import torch

from models.GroundingDINO.stage_b_u0_gate_aligned_d11 import (
    STAGE_B_U0_GATE_ALIGNED_D11_CONTRACT_VERSION,
    STAGE_B_U0_GATE_ALIGNED_D11_LOSS,
    STAGE_B_U0_GATE_ALIGNED_D11_MARKER,
    StageBU0GateAlignedD11Criterion,
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


def _outputs(rank_score, eligible=None):
    if eligible is None:
        eligible = torch.tensor([[True, True, True, False]])
    return {
        "stage_b_gdino_rank_score": rank_score,
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
        "stage_b_u0_category_gate_eligible_mask": eligible,
    }


class StageBU0GateAlignedD11Test(unittest.TestCase):
    def test_contract_gradient_and_gate_ownership(self):
        rank = torch.tensor([[3.0, 2.0, 1.0, 4.0]], requires_grad=True)
        criterion = StageBU0GateAlignedD11Criterion()
        result = criterion(_outputs(rank), [_target()])
        self.assertEqual(
            int(criterion.criterion_contract_version.item()),
            STAGE_B_U0_GATE_ALIGNED_D11_CONTRACT_VERSION,
        )
        self.assertEqual(
            criterion.weight_dict,
            {STAGE_B_U0_GATE_ALIGNED_D11_LOSS: 1.0},
        )
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d11_fix_rows"]), 1
        )
        result[STAGE_B_U0_GATE_ALIGNED_D11_LOSS].backward()
        self.assertGreater(float(rank.grad[0, 0]), 0.0)
        self.assertLess(float(rank.grad[0, 1]), 0.0)
        self.assertEqual(float(rank.grad[0, 2]), 0.0)
        # The highest score is ineligible and must have no rank gradient.
        self.assertEqual(float(rank.grad[0, 3]), 0.0)

    def test_patch_unreachable_row_is_skipped(self):
        rank = torch.tensor([[3.0, 2.0, 1.0, 4.0]], requires_grad=True)
        eligible = torch.tensor([[True, False, True, False]])
        result = StageBU0GateAlignedD11Criterion()(
            _outputs(rank, eligible), [_target()]
        )
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d11_unreachable_rows"]), 1
        )
        self.assertEqual(
            float(result[STAGE_B_U0_GATE_ALIGNED_D11_LOSS].detach()), 0.0
        )
        result[STAGE_B_U0_GATE_ALIGNED_D11_LOSS].backward()
        self.assertTrue(torch.equal(rank.grad, torch.zeros_like(rank)))

    def test_tiny_optimization_repairs_deployed_winner(self):
        rank = torch.nn.Parameter(torch.tensor([[3.0, 2.0, 1.0, 4.0]]))
        criterion = StageBU0GateAlignedD11Criterion()
        optimizer = torch.optim.SGD([rank], lr=0.1)
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_outputs(rank), [_target()])[
                STAGE_B_U0_GATE_ALIGNED_D11_LOSS
            ]
            loss.backward()
            optimizer.step()
        eligible = _outputs(rank)["stage_b_u0_category_gate_eligible_mask"]
        winner = int(rank.detach().masked_fill(~eligible, -torch.inf).argmax())
        self.assertEqual(winner, 1)

    def test_rejects_non_d2_target(self):
        target = _target()
        target.pop(STAGE_B_U0_GATE_ALIGNED_D11_MARKER)
        rank = torch.tensor([[3.0, 2.0, 1.0, 4.0]])
        with self.assertRaisesRegex(ValueError, "exact D2"):
            StageBU0GateAlignedD11Criterion()(_outputs(rank), [target])

    def test_rejects_missing_hard_gate_mask(self):
        rank = torch.tensor([[3.0, 2.0, 1.0, 4.0]])
        outputs = _outputs(rank)
        outputs.pop("stage_b_u0_category_gate_eligible_mask")
        with self.assertRaisesRegex(ValueError, "hard-gate mask"):
            StageBU0GateAlignedD11Criterion()(outputs, [_target()])


if __name__ == "__main__":
    unittest.main()
