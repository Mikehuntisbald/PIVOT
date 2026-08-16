import unittest
from types import SimpleNamespace

import torch

from main import (
    _freeze_and_audit_stage_b_u2v3_category_admission,
    _stage_b_u2v3_category_admission_optimizer_groups,
)
from models.GroundingDINO.stage_b_u0_gate_aligned_d10 import (
    STAGE_B_U0_GATE_ALIGNED_D10_MARKER,
)
from models.GroundingDINO.stage_b_u2v3_category_admission import (
    STAGE_B_U2V3_CATEGORY_ADMISSION_CONTRACT_VERSION,
    STAGE_B_U2V3_CATEGORY_ADMISSION_LOSS,
    StageBU2V3CategoryAdmissionCriterion,
)


def _target():
    return {
        "boxes": torch.tensor([[0.25, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "labels": torch.tensor([7], dtype=torch.int64),
        "support_class": torch.tensor([7], dtype=torch.int64),
        "primary_instance_mask": torch.tensor([True]),
        STAGE_B_U0_GATE_ALIGNED_D10_MARKER: torch.tensor([True]),
    }


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.patch_encoder = torch.nn.Module()
        self.patch_encoder.backbone = self.backbone
        self.patch_encoder.input_proj = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.LayerNorm(2)
        )
        self.patch_encoder.norm = torch.nn.LayerNorm(2)
        self.query_proj_for_patch = torch.nn.Linear(2, 2)
        self.stage_b_u0_patch_rank_adapter = torch.nn.Linear(2, 2)
        self.stage_b_gdino_score_adapter = torch.nn.Linear(2, 2)
        self.stage_b_u2v2_rank_residual = None
        self.patch_logit_scale = torch.nn.Parameter(torch.ones(()))


class StageBU2V3CategoryAdmissionTest(unittest.TestCase):
    def test_criterion_has_independent_contract_and_patch_only_gradient(self):
        patch = torch.tensor([[2.0, 0.0, -1.0, -2.0]], requires_grad=True)
        rank = torch.tensor([[4.0, 3.0, 2.0, 1.0]], requires_grad=True)
        outputs = {
            "pred_logits_patch": patch,
            "pred_boxes": torch.tensor(
                [[[0.5, 0.1, 0.1, 0.1], [0.25, 0.5, 0.2, 0.2],
                  [0.8, 0.5, 0.1, 0.1], [0.5, 0.9, 0.1, 0.1]]]
            ),
            "stage_b_u0_teacher_rank_score": rank,
            "stage_b_u0_candidate_mask": torch.ones((1, 4), dtype=torch.bool),
        }
        criterion = StageBU2V3CategoryAdmissionCriterion(
            gate_max_gap=3.0,
            keep_gap=2.75,
            drop_gap=3.25,
            drop_active_gap=3.75,
            positive_active_gap=2.25,
            positive_target_gap=2.5,
            instance_active_gap=2.25,
            instance_target_gap=2.5,
        )
        result = criterion(outputs, [_target()])
        self.assertEqual(
            int(criterion.criterion_contract_version.item()),
            STAGE_B_U2V3_CATEGORY_ADMISSION_CONTRACT_VERSION,
        )
        self.assertEqual(
            criterion.weight_dict,
            {STAGE_B_U2V3_CATEGORY_ADMISSION_LOSS: 1.0},
        )
        self.assertIn("stage_b_u2v3_deployed_negative_rows", result)
        result[STAGE_B_U2V3_CATEGORY_ADMISSION_LOSS].backward()
        self.assertIsNone(rank.grad)
        self.assertIsNotNone(patch.grad)
        self.assertTrue(torch.isfinite(patch.grad).all())
        self.assertGreater(float(patch.grad.abs().sum()), 0.0)

    def test_freeze_and_optimizer_own_exactly_eight_projection_tensors(self):
        model = _TinyModel()
        count = _freeze_and_audit_stage_b_u2v3_category_admission(model)
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(len(trainable), 8)
        self.assertGreater(count, 0)
        self.assertTrue(
            all(
                name.startswith(
                    ("patch_encoder.input_proj.", "patch_encoder.norm.",
                     "query_proj_for_patch.")
                )
                for name in trainable
            )
        )
        groups = _stage_b_u2v3_category_admission_optimizer_groups(
            model, lr=5e-5
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["stage_b_u2v3_branch"],
            "category_admission_projection",
        )
        self.assertEqual(len(groups[0]["params"]), 8)


if __name__ == "__main__":
    unittest.main()
