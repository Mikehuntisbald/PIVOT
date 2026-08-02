import json
import unittest
from pathlib import Path

import torch
from torch import nn

from models.GroundingDINO.stage_b_gdino_score_adapter import (
    StageBGDINOScoreAdapter,
)
from models.GroundingDINO.stage_b_u0_patch_rank import (
    StageBU0PatchRankAdapter,
    StageBU0PatchRankCriterion,
    validate_stage_b_u0_patch_rank_checkpoint,
)
from main import _stage_b_u0_patch_rank_optimizer_groups
from util.slconfig import SLConfig


class _U0Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_b_u0_patch_rank_adapter = StageBU0PatchRankAdapter(
            query_count=5, hidden_dim=8
        )


class StageBU0PatchRankTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_zero_initialization_is_bitwise_r100_identity(self):
        adapter = StageBU0PatchRankAdapter(query_count=7, hidden_dim=12)
        patch_score = torch.randn(3, 7)
        teacher_rank = torch.randn(3, 7)

        output = adapter(patch_score, teacher_rank)

        self.assertTrue(
            torch.equal(output["patch_rank_residual"], torch.zeros_like(teacher_rank))
        )
        self.assertTrue(torch.equal(output["rank_score"], teacher_rank))
        self.assertTrue(torch.equal(output["teacher_rank_score"], teacher_rank))

    def test_patch_rank_gradient_reaches_patch_but_not_r100_or_p50(self):
        gdino = StageBGDINOScoreAdapter(
            hidden_dim=8, adapter_dim=6, gate_hidden_dim=5
        )
        with torch.no_grad():
            gdino.rank_output.weight.normal_()
            gdino.confidence_gate[-1].weight.normal_()
        u0 = StageBU0PatchRankAdapter(query_count=5, hidden_dim=8)
        with torch.no_grad():
            u0.output.weight.normal_(std=0.2)
        query_hs = torch.randn(2, 5, 8, requires_grad=True)
        base_score = torch.randn(2, 5, requires_grad=True)
        patch_score = torch.randn(2, 5, requires_grad=True)
        teacher = gdino(query_hs, base_score)
        confidence_before = teacher["confidence_score"].detach().clone()

        output = u0(patch_score, teacher["rank_score"])
        loss = output["rank_score"].square().mean()
        loss.backward()

        self.assertIsNone(query_hs.grad)
        self.assertIsNone(base_score.grad)
        self.assertIsNotNone(patch_score.grad)
        self.assertGreater(float(patch_score.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in gdino.parameters()))
        self.assertGreater(float(u0.output.weight.grad.abs().sum()), 0.0)
        confidence_after = gdino(query_hs, base_score)["confidence_score"]
        self.assertTrue(torch.equal(confidence_before, confidence_after.detach()))

    def test_nonzero_patch_residual_changes_rank_but_not_sealed_confidence(self):
        gdino = StageBGDINOScoreAdapter(
            hidden_dim=8, adapter_dim=6, gate_hidden_dim=5
        )
        u0 = StageBU0PatchRankAdapter(query_count=5, hidden_dim=8)
        with torch.no_grad():
            u0.output.weight.normal_(std=0.2)
        query_hs = torch.randn(2, 5, 8)
        base_score = torch.randn(2, 5)
        sealed = gdino(query_hs, base_score)
        confidence = sealed["confidence_score"].clone()

        first = u0(torch.randn(2, 5), sealed["rank_score"])
        second = u0(torch.randn(2, 5), sealed["rank_score"])

        self.assertFalse(torch.equal(first["rank_score"], second["rank_score"]))
        self.assertTrue(torch.equal(gdino(query_hs, base_score)["confidence_score"], confidence))

    def test_category_gate_reuses_u2_state_without_new_checkpoint_keys(self):
        trained = StageBU0PatchRankAdapter(query_count=5, hidden_dim=8)
        gated = StageBU0PatchRankAdapter(
            query_count=5,
            hidden_dim=8,
            category_preserving_gate=True,
            category_gate_max_gap=0.75,
        )
        self.assertEqual(set(trained.state_dict()), set(gated.state_dict()))
        gated.load_state_dict(trained.state_dict(), strict=True)

    def test_category_gate_has_separate_config_and_u2_stays_default_off(self):
        u2 = SLConfig.fromfile(
            "config/ablations/cfg_stageb_u2_category_complete_patch_rank.py"
        )
        gated = SLConfig.fromfile(
            "config/ablations/cfg_stageb_u2_category_preserving_patch_gate.py"
        )
        self.assertFalse(
            getattr(u2, "stage_b_u0_category_preserving_patch_gate", False)
        )
        self.assertTrue(gated.stage_b_u0_category_preserving_patch_gate)
        self.assertEqual(gated.stage_b_u0_category_gate_max_gap, 1.0)

    def test_category_gate_is_patch_only_and_preserves_teacher_inside_eligible(self):
        adapter = StageBU0PatchRankAdapter(
            query_count=5,
            hidden_dim=8,
            category_preserving_gate=True,
            category_gate_max_gap=0.8,
        ).eval()
        with torch.no_grad():
            adapter.output.weight.normal_(std=10.0)
            adapter.output.bias.fill_(100.0)
        patch = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        teacher = torch.tensor([[100.0, -4.0, 20.0, 0.4, 0.5]])

        output = adapter(patch, teacher)
        eligible = output["category_gate_eligible_mask"]
        self.assertEqual(eligible.tolist(), [[False, False, False, True, True]])
        self.assertTrue(torch.equal(output["rank_score"][eligible], teacher[eligible]))
        self.assertLess(
            float(output["rank_score"][~eligible].max()),
            float(output["rank_score"][eligible].min()),
        )
        self.assertGreater(
            float(output["rank_score"][0, 4]),
            float(output["rank_score"][0, 3]),
        )
        self.assertFalse(
            torch.equal(output["pre_category_gate_rank_score"], output["rank_score"])
        )

        other_teacher = torch.tensor([[-10.0, 9.0, 8.0, 7.0, 6.0]])
        other = adapter(patch, other_teacher)
        self.assertTrue(
            torch.equal(other["category_gate_eligible_mask"], eligible)
        )

    def test_category_gate_equal_patch_scores_leave_teacher_bitwise_unchanged(self):
        adapter = StageBU0PatchRankAdapter(
            query_count=4,
            hidden_dim=8,
            category_preserving_gate=True,
            category_gate_max_gap=0.0,
        ).eval()
        patch = torch.ones(2, 4)
        teacher = torch.randn(2, 4)

        output = adapter(patch, teacher)

        self.assertTrue(bool(output["category_gate_eligible_mask"].all().item()))
        self.assertTrue(torch.equal(output["rank_score"], teacher))
        self.assertTrue(
            torch.equal(output["patch_rank_residual"], torch.zeros_like(teacher))
        )

    def test_category_gate_selects_only_inside_existing_candidate_mask(self):
        adapter = StageBU0PatchRankAdapter(
            query_count=4,
            hidden_dim=8,
            category_preserving_gate=True,
            category_gate_max_gap=0.0,
        ).eval()
        patch = torch.tensor([[100.0, 1.0, 3.0, 2.0]])
        teacher = torch.tensor([[50.0, 40.0, 0.2, 30.0]])
        candidate_mask = torch.tensor([[False, True, True, True]])

        output = adapter(patch, teacher, candidate_mask)

        self.assertEqual(
            output["category_gate_eligible_mask"].tolist(),
            [[False, False, True, False]],
        )
        self.assertTrue(
            torch.equal(output["rank_score"][0, 2], teacher[0, 2])
        )
        self.assertLess(
            float(output["rank_score"][0, [0, 1, 3]].max()),
            float(output["rank_score"][0, 2]),
        )

    def test_category_gate_fails_closed_in_training_mode(self):
        adapter = StageBU0PatchRankAdapter(
            query_count=3,
            hidden_dim=8,
            category_preserving_gate=True,
        )
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            adapter(torch.randn(1, 3), torch.randn(1, 3))

    def test_category_gate_rejects_negative_gap(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StageBU0PatchRankAdapter(
                query_count=3,
                hidden_dim=8,
                category_preserving_gate=True,
                category_gate_max_gap=-0.1,
            )

    def test_u1_zero_direct_gain_is_bitwise_u0_identity(self):
        u0 = StageBU0PatchRankAdapter(query_count=7, hidden_dim=12)
        u1 = StageBU0PatchRankAdapter(
            query_count=7,
            hidden_dim=12,
            direct_patch_skip=True,
            direct_patch_gain_limit=0.5,
        )
        with torch.no_grad():
            u0.output.weight.normal_(std=0.2)
            u0.output.bias.normal_(std=0.2)
        u1.trunk.load_state_dict(u0.trunk.state_dict(), strict=True)
        u1.output.load_state_dict(u0.output.state_dict(), strict=True)
        patch = torch.randn(3, 7)
        teacher = torch.randn(3, 7)

        expected = u0(patch, teacher)
        observed = u1(patch, teacher)

        self.assertEqual(float(observed["direct_patch_gain"].detach()), 0.0)
        self.assertTrue(
            torch.equal(observed["rank_score"], expected["rank_score"])
        )
        self.assertTrue(
            torch.equal(
                observed["patch_rank_residual"],
                expected["patch_rank_residual"],
            )
        )

    def test_u1_direct_patch_skip_is_monotonic_and_has_zero_init_gradient(self):
        adapter = StageBU0PatchRankAdapter(
            query_count=5,
            hidden_dim=8,
            direct_patch_skip=True,
            direct_patch_gain_limit=0.5,
        )
        patch = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        teacher = torch.zeros_like(patch)
        zero = adapter(patch, teacher)
        objective = -zero["rank_score"][0, -1] + zero["rank_score"][0, 0]
        objective.backward()
        self.assertIsNotNone(adapter.direct_patch_gain.grad)
        self.assertNotEqual(float(adapter.direct_patch_gain.grad), 0.0)

        with torch.no_grad():
            adapter.direct_patch_gain.fill_(0.2)
        ranked = adapter(patch, teacher)
        direct = ranked["direct_patch_rank_residual"][0]
        self.assertTrue(bool(torch.all(direct[1:] > direct[:-1])))

    def test_u1_direct_gain_has_its_own_optimizer_group(self):
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.stage_b_u0_patch_rank_adapter = StageBU0PatchRankAdapter(
                    query_count=5,
                    hidden_dim=8,
                    direct_patch_skip=True,
                )
                self.patch_projection = nn.Linear(4, 4)

        model = Root()
        groups = _stage_b_u0_patch_rank_optimizer_groups(
            model,
            residual_lr=3e-4,
            patch_projection_lr=3e-5,
            direct_patch_gain_lr=5e-2,
        )
        by_name = {group["stage_b_u0_branch"]: group for group in groups}
        self.assertEqual(set(by_name), {
            "patch_rank_residual",
            "direct_patch_gain",
            "patch_projection",
        })
        self.assertEqual(by_name["direct_patch_gain"]["lr"], 5e-2)
        self.assertEqual(by_name["direct_patch_gain"]["weight_decay"], 0.0)
        self.assertEqual(
            by_name["direct_patch_gain"]["params"],
            [model.stage_b_u0_patch_rank_adapter.direct_patch_gain],
        )

    def test_criterion_uses_r100_as_preservation_baseline(self):
        criterion = StageBU0PatchRankCriterion(residual_weight=0.0)
        teacher = torch.tensor([[0.2, 0.8, 0.1]])
        residual = torch.zeros_like(teacher, requires_grad=True)
        outputs = {
            "stage_b_u0_teacher_rank_score": teacher,
            "stage_b_u0_patch_rank_residual": residual,
            "stage_b_u0_rank_score": teacher + residual,
            "pred_boxes": torch.tensor(
                [[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]]
            ),
        }
        targets = [{"boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2]])}]

        losses = criterion(outputs, targets)
        losses["loss_stage_b_u0_patch_rank"].backward()

        self.assertGreater(float(residual.grad.abs().sum()), 0.0)
        self.assertIn("stage_b_u0_teacher_correct", losses)
        self.assertNotIn("loss_stage_b_gdino_confidence", losses)

    def test_u2_keeps_referent_rank_strict_and_trains_all_category_instances(self):
        criterion = StageBU0PatchRankCriterion(
            residual_weight=0.0,
            category_complete_supervision=True,
            category_loss_weight=1.0,
        )
        teacher = torch.tensor([[0.2, 0.8, 0.1]])
        residual = torch.zeros_like(teacher, requires_grad=True)
        patch_score = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
        boxes = torch.tensor(
            [[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]]
        )
        outputs = {
            "stage_b_u0_teacher_rank_score": teacher,
            "stage_b_u0_patch_rank_residual": residual,
            "stage_b_u0_rank_score": teacher + residual,
            "pred_logits_patch": patch_score,
            "pred_boxes": boxes,
        }
        targets = [
            {
                "boxes": torch.cat(
                    (
                        boxes[0, :2].clone(),
                        torch.tensor([[0.05, 0.95, 0.02, 0.02]]),
                    ),
                    dim=0,
                ),
                "primary_instance_mask": torch.tensor([True, False, False]),
                "stage_b_u2_category_complete": torch.tensor([True]),
            }
        ]

        losses = criterion(outputs, targets)
        losses["loss_stage_b_u0_patch_rank"].backward()

        self.assertEqual(float(losses["stage_b_u0_teacher_correct"]), 0.0)
        self.assertEqual(
            float(losses["stage_b_u2_category_teacher_correct"]), 1.0
        )
        self.assertEqual(
            float(losses["stage_b_u2_category_valid_instances"]), 2.0
        )
        self.assertEqual(
            float(losses["stage_b_u2_category_skipped_instances"]), 1.0
        )
        self.assertGreater(float(losses["stage_b_u2_category_patch_loss"]), 0.0)
        self.assertGreater(float(patch_score.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(patch_score.grad).all().item()))
        self.assertEqual(float(residual.grad.abs().sum()), 0.0)

    def test_u2_category_loss_requires_exact_dataset_marker(self):
        criterion = StageBU0PatchRankCriterion(
            category_complete_supervision=True,
            category_loss_weight=1.0,
        )
        outputs = {
            "stage_b_u0_teacher_rank_score": torch.zeros(1, 2),
            "stage_b_u0_patch_rank_residual": torch.zeros(1, 2),
            "stage_b_u0_rank_score": torch.zeros(1, 2),
            "pred_logits_patch": torch.zeros(1, 2),
            "pred_boxes": torch.tensor(
                [[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]
            ),
        }
        targets = [
            {
                "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1]]),
                "primary_instance_mask": torch.tensor([True]),
            }
        ]
        with self.assertRaisesRegex(ValueError, "exact true"):
            criterion(outputs, targets)

    def test_contract_validator_requires_complete_exact_state(self):
        model = _U0Root()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        validate_stage_b_u0_patch_rank_checkpoint(
            model, state, checkpoint_label="complete"
        )
        missing = dict(state)
        missing.pop("stage_b_u0_patch_rank_adapter.output.weight")
        with self.assertRaisesRegex(ValueError, "missing="):
            validate_stage_b_u0_patch_rank_checkpoint(
                model, missing, checkpoint_label="missing"
            )
        drift = dict(state)
        drift["stage_b_u0_patch_rank_adapter._contract_query_count"] = torch.tensor(6)
        with self.assertRaisesRegex(ValueError, "contract_mismatches"):
            validate_stage_b_u0_patch_rank_checkpoint(
                model, drift, checkpoint_label="drift"
            )

    def test_requires_exact_all_query_shape_and_finite_scores(self):
        adapter = StageBU0PatchRankAdapter(query_count=5, hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            adapter(torch.randn(1, 4), torch.randn(1, 4))
        patch = torch.randn(1, 5)
        patch[0, 2] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter(patch, torch.randn(1, 5))

    def test_training_dataset_is_explicitly_positive_only(self):
        config = json.loads(
            Path("config/datasets_stageb_u0_patch_rank_three_ref.json").read_text()
        )
        self.assertEqual(len(config["train"]), 3)
        for source in config["train"]:
            self.assertEqual(source["dataset_mode"], "patch_episode")
            self.assertEqual(source["neg_episode_prob"], 0.0)
            self.assertIs(source["lvis_neg_category_only"], False)
            self.assertEqual(source["mix_weight"], 2.0)


if __name__ == "__main__":
    unittest.main()
