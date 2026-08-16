import unittest

import torch

from models.GroundingDINO.stage_b_u0_gate_aligned_d11 import (
    STAGE_B_U0_GATE_ALIGNED_D11_MARKER,
)
from models.GroundingDINO.stage_b_u0_patch_rank import StageBU0PatchRankAdapter
from models.GroundingDINO.stage_b_u2v2_rank_residual import (
    STAGE_B_U2V2_LOSS,
    StageBU2V2RankResidual,
    StageBU2V2RankResidualCriterion,
)


def _target():
    return {
        "boxes": torch.tensor([[0.25, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "labels": torch.tensor([7]),
        "support_class": torch.tensor([7]),
        "primary_instance_mask": torch.tensor([True]),
        STAGE_B_U0_GATE_ALIGNED_D11_MARKER: torch.tensor([True]),
    }


class StageBU2V2Test(unittest.TestCase):
    def test_zero_init_is_eligible_identity(self):
        module = StageBU2V2RankResidual(feature_dim=4, hidden_dim=8)
        feature = torch.randn(2, 5, 4, requires_grad=True)
        teacher = torch.randn(2, 5, requires_grad=True)
        eligible = torch.tensor(
            [[True, False, True, False, False], [False, True, True, True, False]]
        )
        output = module(feature, teacher, eligible)
        self.assertTrue(torch.equal(output["pre_demotion_rank_score"], teacher.detach()))
        self.assertTrue(torch.equal(output["rank_residual"], torch.zeros_like(teacher)))

    def test_residual_is_bounded_and_eligible_only(self):
        module = StageBU2V2RankResidual(
            feature_dim=4, hidden_dim=8, residual_limit=0.1
        )
        with torch.no_grad():
            module.output.weight.fill_(1e4)
        eligible = torch.tensor([[True, False, True]])
        result = module(torch.randn(1, 3, 4), torch.randn(1, 3), eligible)
        residual = result["rank_residual"]
        self.assertLessEqual(float(residual.detach().abs().max()), 0.10000001)
        self.assertEqual(float(residual.detach()[0, 1]), 0.0)

    def test_empty_mask_fails_closed(self):
        module = StageBU2V2RankResidual(feature_dim=4, hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            module(torch.randn(1, 3, 4), torch.randn(1, 3), torch.zeros(1, 3, dtype=torch.bool))

    def test_lexicographic_demotion_always_selects_eligible(self):
        gate = StageBU0PatchRankAdapter(
            query_count=4, category_preserving_gate=True, category_gate_max_gap=0.5
        ).eval()
        patch = torch.tensor([[0.0, 3.0, 2.8, -4.0]])
        teacher = torch.tensor([[100.0, -3.0, -2.0, 200.0]])
        score, eligible = gate.apply_category_preserving_gate(
            patch, teacher, torch.ones_like(teacher, dtype=torch.bool)
        )
        self.assertTrue(bool(eligible[0, int(score.argmax())]))
        self.assertTrue(torch.equal(score[eligible], teacher[eligible]))

    def test_loss_connects_only_to_residual(self):
        module = StageBU2V2RankResidual(feature_dim=4, hidden_dim=8)
        feature = torch.randn(1, 4, 4, requires_grad=True)
        teacher = torch.tensor([[3.0, 2.0, 1.0, 4.0]], requires_grad=True)
        eligible = torch.tensor([[True, True, True, False]])
        result = module(feature, teacher, eligible)
        outputs = {
            "stage_b_u2v2_rank_score": result["pre_demotion_rank_score"],
            "stage_b_u2v2_teacher_rank_score": result["teacher_rank_score"],
            "stage_b_u2v2_rank_residual": result["rank_residual"],
            "stage_b_u2v2_eligible_mask": eligible,
            "pred_boxes": torch.tensor([[[.5,.1,.1,.1],[.25,.5,.2,.2],[.8,.5,.1,.1],[.5,.9,.1,.1]]]),
        }
        loss = StageBU2V2RankResidualCriterion()(outputs, [_target()])[STAGE_B_U2V2_LOSS]
        loss.backward()
        self.assertIsNone(feature.grad)
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(module.output.weight.grad).all())
        self.assertGreater(float(module.output.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
