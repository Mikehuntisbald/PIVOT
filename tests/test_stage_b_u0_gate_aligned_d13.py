import unittest

import torch

from models.GroundingDINO.stage_b_u0_gate_aligned_d13 import (
    STAGE_B_U0_GATE_ALIGNED_D13_CONTRACT_VERSION,
    STAGE_B_U0_GATE_ALIGNED_D13_LOSS,
    StageBU0GateAlignedD13Criterion,
    StageBU0GateAlignedD13PatchResidual,
)


def _target() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "labels": torch.tensor([7], dtype=torch.int64),
        "support_class": torch.tensor([7], dtype=torch.int64),
        "primary_instance_mask": torch.tensor([True]),
        "stage_b_native_patch_category_d2": torch.tensor([True]),
    }


def _boxes() -> torch.Tensor:
    return torch.tensor(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.2, 0.2],
            [0.9, 0.9, 0.1, 0.1],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


def _criterion_outputs(
    *, teacher_patch: torch.Tensor, adapted_patch: torch.Tensor, rank: torch.Tensor
) -> dict[str, torch.Tensor]:
    mask = torch.ones_like(teacher_patch, dtype=torch.bool)
    teacher_best = teacher_patch.amax(dim=1, keepdim=True)
    adapted_best = adapted_patch.amax(dim=1, keepdim=True)
    return {
        "stage_b_u0_d13_patch_score": adapted_patch,
        "stage_b_u0_d13_teacher_patch_score": teacher_patch,
        "stage_b_u0_d13_patch_residual": adapted_patch - teacher_patch,
        "stage_b_u0_d13_teacher_rank_score": rank,
        "stage_b_u0_d13_teacher_eligible_mask": (
            teacher_best - teacher_patch <= 2.0
        ),
        "stage_b_u0_category_gate_eligible_mask": (
            adapted_best - adapted_patch <= 2.0
        ),
        "stage_b_u0_candidate_mask": mask,
        "pred_boxes": _boxes(),
    }


class StageBU0GateAlignedD13Test(unittest.TestCase):
    def test_zero_initializer_is_exact_teacher_gate(self) -> None:
        module = StageBU0GateAlignedD13PatchResidual(
            feature_dim=4, hidden_dim=3, residual_limit=0.25, gate_max_gap=2.0
        )
        module.train()
        query = torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1)
        support = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
        patch = torch.tensor([[0.0, -1.0, -2.1], [0.2, -2.0, -3.0]])
        rank = torch.tensor([[0.1, 0.3, 0.2], [0.7, 0.4, 0.9]])
        result = module(query, support, patch, rank)
        self.assertTrue(torch.equal(result["patch_residual"], torch.zeros_like(patch)))
        self.assertTrue(torch.equal(result["patch_score"], patch))
        self.assertTrue(
            torch.equal(result["eligible_mask"], result["teacher_eligible_mask"])
        )
        self.assertTrue(
            torch.equal(result["rank_score"], result["teacher_gated_rank_score"])
        )
        self.assertEqual(int(module.contract_version), STAGE_B_U0_GATE_ALIGNED_D13_CONTRACT_VERSION)
        self.assertEqual(len(module.trainable_parameters()), 7)

    def test_actionable_negative_winner_has_gradient(self) -> None:
        module = StageBU0GateAlignedD13PatchResidual(
            feature_dim=4, hidden_dim=3
        )
        query = torch.nn.functional.normalize(torch.randn(1, 3, 4), dim=-1)
        support = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
        patch = torch.tensor([[0.0, -1.9, -3.0]])
        rank = torch.tensor([[1.0, 0.5, 0.0]])
        adapted = module(query, support, patch, rank)
        outputs = {
            "stage_b_u0_d13_patch_score": adapted["patch_score"],
            "stage_b_u0_d13_teacher_patch_score": adapted["teacher_patch_score"],
            "stage_b_u0_d13_patch_residual": adapted["patch_residual"],
            "stage_b_u0_d13_teacher_rank_score": adapted["teacher_rank_score"],
            "stage_b_u0_d13_teacher_eligible_mask": adapted[
                "teacher_eligible_mask"
            ],
            "stage_b_u0_category_gate_eligible_mask": adapted["eligible_mask"],
            "stage_b_u0_candidate_mask": adapted["candidate_mask"],
            "pred_boxes": _boxes(),
        }
        result = StageBU0GateAlignedD13Criterion()(outputs, [_target()])
        self.assertEqual(
            result.keys() & {STAGE_B_U0_GATE_ALIGNED_D13_LOSS},
            {STAGE_B_U0_GATE_ALIGNED_D13_LOSS},
        )
        self.assertEqual(int(result["stage_b_u0_gate_aligned_d13_actionable_rows"]), 1)
        self.assertEqual(int(result["stage_b_u0_gate_aligned_d13_selected_blockers"]), 1)
        result[STAGE_B_U0_GATE_ALIGNED_D13_LOSS].backward()
        self.assertGreater(float(module.output.weight.grad.abs().sum()), 0.0)

    def test_correct_teacher_has_zero_preserve_loss_at_initializer(self) -> None:
        teacher = torch.tensor([[-2.1, 0.0, -3.0]])
        rank = torch.tensor([[2.0, 1.0, 0.0]])
        result = StageBU0GateAlignedD13Criterion()(
            _criterion_outputs(
                teacher_patch=teacher,
                adapted_patch=teacher.clone().requires_grad_(True),
                rank=rank,
            ),
            [_target()],
        )
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d13_teacher_correct_rows"]), 1
        )
        self.assertEqual(
            float(result["stage_b_u0_gate_aligned_d13_preserve_loss"]), 0.0
        )

    def test_correct_teacher_regression_is_penalized(self) -> None:
        teacher = torch.tensor([[-2.1, 0.0, -3.0]])
        adapted = torch.tensor([[-1.9, 0.0, -3.0]], requires_grad=True)
        rank = torch.tensor([[2.0, 1.0, 0.0]])
        result = StageBU0GateAlignedD13Criterion()(
            _criterion_outputs(
                teacher_patch=teacher, adapted_patch=adapted, rank=rank
            ),
            [_target()],
        )
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d13_correct_regressed"]), 1
        )
        self.assertGreater(
            float(result["stage_b_u0_gate_aligned_d13_preserve_loss"]), 0.0
        )
        result[STAGE_B_U0_GATE_ALIGNED_D13_LOSS].backward()
        self.assertGreater(float(adapted.grad.abs().sum()), 0.0)

    def test_neutral_winner_is_not_relabelled_as_category_negative(self) -> None:
        boxes = _boxes()
        boxes[0, 0] = torch.tensor([0.5, 0.5, 0.29, 0.29])
        teacher = torch.tensor([[0.0, -1.0, -3.0]])
        rank = torch.tensor([[2.0, 1.0, 0.0]])
        outputs = _criterion_outputs(
            teacher_patch=teacher,
            adapted_patch=teacher.clone().requires_grad_(True),
            rank=rank,
        )
        outputs["pred_boxes"] = boxes
        result = StageBU0GateAlignedD13Criterion()(outputs, [_target()])
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d13_actionable_rows"]), 0
        )
        self.assertEqual(
            int(result["stage_b_u0_gate_aligned_d13_skipped_neutral_rows"]), 1
        )


if __name__ == "__main__":
    unittest.main()
