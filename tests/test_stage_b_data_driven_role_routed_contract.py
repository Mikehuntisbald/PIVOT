import unittest

import torch
from torch import nn

from main import (
    _freeze_and_audit_stage_b_data_driven,
    _stage_b_data_driven_parameter_groups,
)
from models.GroundingDINO.stage_b_data_driven_score import (
    DATA_DRIVEN_PATCH_DROP_ANCHOR_GLOBAL_MAX,
    DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED,
    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED,
    StageBDataDrivenCriterion,
    StageBDataDrivenScoreHeads,
    _active_unsafe_fixed_denominator_severity_loss,
    _active_unsafe_row_fraction_loss,
    _dense_fixed_denominator_softplus_loss,
    deployment_gate_category_patch_loss,
    role_routed_official_assignment_top1_loss,
    validate_data_driven_criterion_checkpoint_state,
)


class _PatchEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


class _RoleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.patch_encoder = _PatchEncoder(self.backbone)
        self.query_proj_for_patch = nn.Linear(4, 4)
        self.patch_logit_scale = nn.Parameter(torch.tensor(2.0))
        self.stage_b_data_driven_score_heads = StageBDataDrivenScoreHeads(
            4,
            rank_dim=3,
            confidence_dim=3,
            gate_hidden_dim=4,
        )


def _category_target(*, include_unassigned=True):
    boxes = [
        [0.2, 0.5, 0.2, 0.2],
        [0.8, 0.5, 0.2, 0.2],
    ]
    roles = [0, 1]
    if include_unassigned:
        boxes.append([0.5, 0.8, 0.2, 0.2])
        roles.append(-1)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.ones(len(boxes), dtype=torch.int64),
        "primary_instance_mask": torch.tensor(
            [True] + [False] * (len(boxes) - 1)
        ),
        "stage_b_u2_category_complete": torch.tensor([True]),
        "stage_b_data_driven_assignment_valid": torch.tensor([True]),
        "stage_b_data_driven_assignment_role": torch.tensor(
            roles, dtype=torch.int64
        ),
    }


class StageBDataDrivenRoleRoutedContractTest(unittest.TestCase):
    def test_active_unsafe_surrogate_tracks_fraction_and_row_normalizes(self):
        one_of_four = torch.tensor(
            [0.2, -0.2, -0.2, -0.2], requires_grad=True
        )
        four_of_four = torch.full((4,), 0.2, requires_grad=True)
        duplicated_half = torch.tensor(
            [0.2, -0.2, 0.2, -0.2], requires_grad=True
        )
        one_loss, one_active = _active_unsafe_row_fraction_loss(
            one_of_four, temperature=0.1
        )
        four_loss, four_active = _active_unsafe_row_fraction_loss(
            four_of_four, temperature=0.1
        )
        half_loss, half_active = _active_unsafe_row_fraction_loss(
            duplicated_half, temperature=0.1
        )
        pair_loss, _ = _active_unsafe_row_fraction_loss(
            duplicated_half[:2], temperature=0.1
        )
        self.assertEqual(int(one_active.sum()), 1)
        self.assertEqual(int(four_active.sum()), 4)
        self.assertEqual(int(half_active.sum()), 2)
        self.assertTrue(torch.allclose(four_loss, 4.0 * one_loss))
        self.assertTrue(torch.allclose(half_loss, pair_loss))
        one_loss.backward()
        self.assertGreater(float(one_of_four.grad[0]), 0.0)
        self.assertEqual(float(one_of_four.grad[1:].abs().sum()), 0.0)

    def test_active_unsafe_severity_keeps_tail_gradient_and_fixed_denominator(self):
        violations = torch.tensor(
            [0.2, -0.2, 0.8, -0.8], requires_grad=True
        )
        loss, active = _active_unsafe_fixed_denominator_severity_loss(
            violations, temperature=0.1
        )
        self.assertEqual(active.tolist(), [True, False, True, False])
        loss.backward()
        self.assertGreater(float(violations.grad[0]), 0.0)
        self.assertGreater(float(violations.grad[2]), 0.0)
        self.assertGreater(
            float(violations.grad[2]), float(violations.grad[0])
        )
        self.assertEqual(float(violations.grad[[1, 3]].abs().sum()), 0.0)

    def test_patch_dense_tail_default_zero_and_weighted_contract(self):
        target = {"boxes": torch.tensor([[0.2, 0.5, 0.2, 0.2]])}
        boxes = torch.tensor(
            [
                [
                    [0.2, 0.5, 0.2, 0.2],
                    [0.8, 0.1, 0.1, 0.1],
                    [0.7, 0.1, 0.1, 0.1],
                ]
                + [[0.2, 0.5, 0.13, 0.13]] * 9
            ],
            dtype=torch.float32,
        )
        candidate = torch.ones((1, 12), dtype=torch.bool)
        scores = torch.tensor([[10.0, 1.5, 1.08] + [0.0] * 9])
        kwargs = dict(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            boundary_margin=0.25,
            temperature=0.1,
            dense_category_focal_weight=0.0,
        )

        default_scores = scores.clone().requires_grad_(True)
        zero_scores = scores.clone().requires_grad_(True)
        weighted_scores = scores.clone().requires_grad_(True)
        default = deployment_gate_category_patch_loss(
            default_scores, boxes, [target], candidate, **kwargs
        )
        explicit_zero = deployment_gate_category_patch_loss(
            zero_scores,
            boxes,
            [target],
            candidate,
            drop_dense_tail_weight=0.0,
            **kwargs,
        )
        weight = 1.7
        weighted = deployment_gate_category_patch_loss(
            weighted_scores,
            boxes,
            [target],
            candidate,
            drop_dense_tail_weight=weight,
            **kwargs,
        )

        for key in (
            "loss",
            "keep_objective_component",
            "drop_objective_component",
        ):
            self.assertTrue(torch.equal(default[key], explicit_zero[key]), key)
        default_grad = torch.autograd.grad(default["loss"], default_scores)[0]
        zero_grad = torch.autograd.grad(explicit_zero["loss"], zero_scores)[0]
        self.assertTrue(torch.equal(default_grad, zero_grad))

        for key in (
            "standardized_patch_score",
            "deployed_gate",
            "category_positive_mask",
            "category_negative_mask",
            "category_neutral_mask",
            "role_exclusive_positive_mask",
        ):
            self.assertTrue(torch.equal(explicit_zero[key], weighted[key]), key)
        self.assertTrue(
            torch.equal(
                explicit_zero["drop_dense_tail_component"],
                weighted["drop_dense_tail_component"],
            )
        )
        torch.testing.assert_close(
            weighted["drop_objective_component"]
            - explicit_zero["drop_objective_component"],
            weight * weighted["drop_dense_tail_component"],
        )
        torch.testing.assert_close(
            weighted["loss"] - explicit_zero["loss"],
            0.5 * weight * weighted["drop_dense_tail_component"],
        )

        for invalid_weight in (-0.1, float("nan"), float("inf")):
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaisesRegex(ValueError, "thresholds are invalid"):
                    deployment_gate_category_patch_loss(
                        scores,
                        boxes,
                        [target],
                        candidate,
                        drop_dense_tail_weight=invalid_weight,
                        **kwargs,
                    )

    def test_patch_dense_tail_reaches_a_slightly_safe_negative(self):
        target = {"boxes": torch.tensor([[0.2, 0.5, 0.2, 0.2]])}
        boxes = torch.tensor(
            [
                [
                    [0.2, 0.5, 0.2, 0.2],
                    [0.8, 0.1, 0.1, 0.1],
                    [0.7, 0.1, 0.1, 0.1],
                ]
                + [[0.2, 0.5, 0.13, 0.13]] * 9
            ],
            dtype=torch.float32,
        )
        scores = torch.tensor(
            [[10.0, 1.5, 1.08] + [0.0] * 9], requires_grad=True
        )
        result = deployment_gate_category_patch_loss(
            scores,
            boxes,
            [target],
            torch.ones_like(scores, dtype=torch.bool),
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            boundary_margin=0.25,
            temperature=0.1,
            drop_dense_tail_weight=1.0,
            dense_category_focal_weight=0.0,
        )
        standardized = result["standardized_patch_score"][0]
        violations = 3.25 - (standardized[0] - standardized[1:3])
        self.assertGreater(float(violations[0]), 0.0)
        self.assertLess(float(violations[1]), 0.0)
        self.assertGreater(float(violations[1]), -0.02)
        self.assertEqual(float(result["active_unsafe_drop_queries"]), 1.0)
        torch.testing.assert_close(
            result["drop_dense_tail_component"],
            _dense_fixed_denominator_softplus_loss(
                violations, temperature=0.1
            ),
        )

        active_grad = torch.autograd.grad(
            result["drop_active_unsafe_component"], scores, retain_graph=True
        )[0]
        dense_grad = torch.autograd.grad(
            result["drop_dense_tail_component"], scores
        )[0]
        self.assertEqual(float(active_grad[0, 2]), 0.0)
        self.assertGreater(float(dense_grad[0, 2]), 0.0)
        self.assertLess(float(dense_grad[0, 0]), 0.0)
        self.assertEqual(float(dense_grad[0, 3:].abs().sum()), 0.0)
        self.assertAlmostEqual(float(dense_grad.sum()), 0.0, places=6)

    def test_role_truth_table_and_rank_gradient_ownership(self):
        rank = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ],
            requires_grad=True,
        )
        assignment_iou = torch.tensor(
            [
                [
                    [0.8, 0.1],  # exclusive role 0
                    [0.1, 0.9],  # exclusive role 1
                    [0.0, 0.0],  # certain category negative
                    [0.4, 0.1],  # 0.3--0.5 ambiguous
                    [0.6, 0.6],  # double overlap
                    [0.0, 0.0],  # unassigned same-category instance
                    [0.8, 0.1],  # own plus unassigned double overlap
                ]
            ],
            dtype=torch.float32,
        )
        other_iou = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7]],
            dtype=torch.float32,
        )
        patch = torch.zeros(1, 7, 1, requires_grad=True)
        result = role_routed_official_assignment_top1_loss(
            rank,
            assignment_iou,
            other_iou,
            torch.tensor([True]),
            torch.ones(1, 7, 2, dtype=torch.bool),
            patch,
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            margin=0.1,
            temperature=0.1,
            include_all_exclusive_nonowned=True,
        )
        self.assertEqual(
            result["owned_mask"].nonzero(as_tuple=False).tolist(),
            [[0, 0, 0], [0, 1, 1]],
        )
        self.assertEqual(
            result["safe_nonowned_mask"].nonzero(as_tuple=False).tolist(),
            [
                [0, 0, 1],
                [0, 1, 0],
                [0, 5, 0],
                [0, 5, 1],
                [0, 6, 1],
            ],
        )
        self.assertEqual(
            result["category_negative_mask"].nonzero(
                as_tuple=False
            ).tolist(),
            [[0, 2]],
        )
        self.assertEqual(
            result["neutral_mask"].nonzero(as_tuple=False).tolist(),
            [
                [0, 3, 0],
                [0, 3, 1],
                [0, 4, 0],
                [0, 4, 1],
                [0, 6, 0],
            ],
        )
        result["loss"].backward()
        self.assertLess(float(rank.grad[0, 0, 0]), 0.0)
        self.assertEqual(float(rank.grad[0, 1, 0]), 0.0)
        self.assertEqual(float(rank.grad[0, 0, 1]), 0.0)
        self.assertLess(float(rank.grad[0, 1, 1]), 0.0)
        self.assertEqual(float(rank.grad[0, 2:5].abs().sum()), 0.0)
        self.assertGreater(float(rank.grad[0, 5, 0]), 0.0)
        self.assertEqual(float(rank.grad[0, 5, 1]), 0.0)
        self.assertEqual(float(rank.grad[0, 6, 0]), 0.0)
        self.assertGreater(float(rank.grad[0, 6, 1]), 0.0)
        self.assertIsNone(patch.grad)

    def test_patch_objective_is_affine_invariant_and_focal_is_diagnostic_only(self):
        target = _category_target(include_unassigned=False)
        boxes = torch.tensor(
            [
                [
                    [0.2, 0.5, 0.2, 0.2],
                    [0.8, 0.5, 0.2, 0.2],
                    [0.5, 0.1, 0.1, 0.1],
                    [0.2, 0.5, 0.13, 0.13],
                    [0.9, 0.1, 0.1, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
        candidate = torch.tensor([[True, True, True, True, False]])
        patch = torch.tensor(
            [[0.4, -0.2, 2.0, 0.1, 9.0]], requires_grad=True
        )
        kwargs = dict(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            boundary_margin=0.25,
            temperature=0.1,
            role_exclusive_keep=True,
            dense_category_focal_weight=0.0,
        )
        base = deployment_gate_category_patch_loss(
            patch, boxes, [target], candidate, **kwargs
        )
        transformed = deployment_gate_category_patch_loss(
            patch.detach() * 3.7 + 8.0,
            boxes,
            [target],
            candidate,
            **kwargs,
        )
        self.assertTrue(
            torch.allclose(
                base["standardized_patch_score"],
                transformed["standardized_patch_score"],
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.equal(base["deployed_gate"], transformed["deployed_gate"])
        )
        for key in (
            "loss",
            "keep_component",
            "keep_objective_component",
            "drop_component",
            "drop_objective_component",
        ):
            self.assertTrue(
                torch.allclose(base[key], transformed[key], atol=2e-6)
            )
        self.assertFalse(
            torch.allclose(
                base["dense_category_focal_component"],
                transformed["dense_category_focal_component"],
            )
        )
        self.assertTrue(
            torch.allclose(
                base["loss"],
                0.5
                * (
                    base["keep_objective_component"]
                    + base["drop_objective_component"]
                ),
                atol=1e-7,
                rtol=1e-7,
            )
        )
        self.assertGreaterEqual(
            float(base["keep_component"].detach()),
            float(base["keep_mean_component"].detach()),
        )
        self.assertEqual(
            base["category_positive_mask"].tolist(),
            [[True, True, False, False, False]],
        )
        self.assertEqual(
            base["category_negative_mask"].tolist(),
            [[False, False, True, False, False]],
        )
        self.assertEqual(
            base["category_neutral_mask"].tolist(),
            [[False, False, False, True, False]],
        )
        base["dense_category_focal_component"].backward(
            retain_graph=True
        )
        self.assertLess(float(patch.grad[0, 0]), 0.0)
        self.assertLess(float(patch.grad[0, 1]), 0.0)
        self.assertGreater(float(patch.grad[0, 2]), 0.0)
        self.assertEqual(float(patch.grad[0, 3]), 0.0)
        self.assertEqual(float(patch.grad[0, 4]), 0.0)
        patch.grad.zero_()
        base["loss"].backward()
        self.assertLess(float(patch.grad[0, 0]), 0.0)
        self.assertLess(float(patch.grad[0, 1]), 0.0)
        self.assertGreater(float(patch.grad[0, 2]), 0.0)
        self.assertEqual(float(patch.grad[0, 3]), 0.0)
        self.assertEqual(float(patch.grad[0, 4]), 0.0)

    def test_patch_straight_through_clip_is_bitwise_exact_for_900_queries(self):
        target = _category_target(include_unassigned=False)
        boxes = torch.full((1, 900, 4), 0.05, dtype=torch.float32)
        boxes[0, 0] = target["boxes"][0]
        boxes[0, 1] = target["boxes"][1]
        patch = torch.zeros((1, 900), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            patch[0, :4] = 1.0
        result = deployment_gate_category_patch_loss(
            patch,
            boxes,
            [target],
            torch.ones((1, 900), dtype=torch.bool),
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            boundary_margin=0.25,
            temperature=0.1,
            role_exclusive_keep=True,
        )
        self.assertEqual(float(result["standardized_patch_score"][0, 0]), 5.0)
        result["loss"].backward()
        self.assertTrue(bool(torch.isfinite(patch.grad).all()))
        self.assertGreater(float(patch.grad.abs().sum()), 0.0)

    def test_patch_active_unsafe_auxiliary_densifies_only_violations(self):
        drop_target = {
            "boxes": torch.tensor([[0.2, 0.5, 0.2, 0.2]])
        }
        drop_boxes = torch.tensor(
            [
                [
                    [0.2, 0.5, 0.2, 0.2],
                    [0.8, 0.1, 0.1, 0.1],
                    [0.7, 0.1, 0.1, 0.1],
                ]
                + [[0.2, 0.5, 0.13, 0.13]] * 9
            ],
            dtype=torch.float32,
        )
        drop_patch = torch.tensor(
            [[10.0, 1.5, 1.3] + [0.0] * 9], requires_grad=True
        )
        kwargs = dict(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            boundary_margin=0.25,
            temperature=0.1,
        )
        drop = deployment_gate_category_patch_loss(
            drop_patch,
            drop_boxes,
            [drop_target],
            torch.ones_like(drop_patch, dtype=torch.bool),
            **kwargs,
        )
        self.assertEqual(float(drop["active_unsafe_drop_queries"]), 2.0)
        self.assertTrue(
            torch.allclose(
                drop["drop_objective_component"],
                drop["drop_component"]
                + drop["drop_active_unsafe_component"],
            )
        )
        drop["drop_component"].backward(retain_graph=True)
        self.assertLess(float(drop_patch.grad[0, 0]), 0.0)
        self.assertGreater(float(drop_patch.grad[0, 1]), 0.0)
        self.assertEqual(float(drop_patch.grad[0, 2]), 0.0)
        self.assertEqual(float(drop_patch.grad[0, 3:].abs().sum()), 0.0)
        self.assertAlmostEqual(float(drop_patch.grad.sum()), 0.0, places=6)
        drop_patch.grad.zero_()
        drop["drop_active_unsafe_component"].backward()
        self.assertLess(float(drop_patch.grad[0, 0]), 0.0)
        self.assertGreater(float(drop_patch.grad[0, 1]), 0.0)
        self.assertGreater(float(drop_patch.grad[0, 2]), 0.0)
        self.assertEqual(float(drop_patch.grad[0, 3:].abs().sum()), 0.0)
        self.assertAlmostEqual(float(drop_patch.grad.sum()), 0.0, places=6)

        keep_target = _category_target(include_unassigned=False)
        keep_boxes = torch.full((1, 12, 4), 0.05, dtype=torch.float32)
        keep_boxes[0, 0] = torch.tensor([0.5, 0.1, 0.1, 0.1])
        keep_boxes[0, 1] = keep_target["boxes"][0]
        keep_boxes[0, 2] = keep_target["boxes"][1]
        keep_patch = torch.zeros((1, 12), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            keep_patch[0, 0] = 10.0
        keep = deployment_gate_category_patch_loss(
            keep_patch,
            keep_boxes,
            [keep_target],
            torch.ones_like(keep_patch, dtype=torch.bool),
            role_exclusive_keep=True,
            **kwargs,
        )
        self.assertEqual(
            float(keep["active_unsafe_generic_keep_constraints"]), 2.0
        )
        self.assertEqual(
            float(
                keep[
                    "active_unsafe_role_exclusive_keep_constraints"
                ]
            ),
            2.0,
        )
        self.assertTrue(
            torch.allclose(
                keep["keep_objective_component"],
                keep["keep_component"]
                + keep["keep_active_unsafe_component"],
            )
        )
        self.assertTrue(
            torch.allclose(
                keep["keep_active_unsafe_component"],
                0.5
                * (
                    keep["generic_keep_active_unsafe_component"]
                    + keep[
                        "role_exclusive_keep_active_unsafe_component"
                    ]
                ),
            )
        )
        keep["keep_active_unsafe_component"].backward()
        self.assertGreater(float(keep_patch.grad[0, 0]), 0.0)
        self.assertLess(float(keep_patch.grad[0, 1]), 0.0)
        self.assertLess(float(keep_patch.grad[0, 2]), 0.0)
        self.assertEqual(float(keep_patch.grad[0, 3:].abs().sum()), 0.0)

        safe_patch = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
        safe = deployment_gate_category_patch_loss(
            safe_patch,
            keep_target["boxes"].reshape(1, 2, 4),
            [keep_target],
            torch.ones_like(safe_patch, dtype=torch.bool),
            role_exclusive_keep=True,
            **kwargs,
        )
        self.assertEqual(
            float(safe["keep_active_unsafe_component"].detach()), 0.0
        )
        self.assertEqual(
            float(safe["drop_active_unsafe_component"].detach()), 0.0
        )
        self.assertTrue(bool(torch.isfinite(safe["loss"]).item()))

    def test_balanced_drop_anchor_is_forward_exact_and_reaches_every_instance(self):
        target = _category_target(include_unassigned=False)
        boxes = torch.tensor(
            [
                [
                    [0.2, 0.5, 0.2, 0.2],
                    [0.8, 0.5, 0.2, 0.2],
                    [0.5, 0.1, 0.1, 0.1],
                    [0.7, 0.1, 0.1, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
        candidate = torch.ones((1, 4), dtype=torch.bool)
        kwargs = dict(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            boundary_margin=0.25,
            temperature=0.1,
            role_exclusive_keep=True,
            dense_category_focal_weight=0.0,
        )
        global_patch = torch.tensor(
            [[2.0, 0.5, 1.5, 1.0]], requires_grad=True
        )
        balanced_patch = global_patch.detach().clone().requires_grad_(True)
        global_result = deployment_gate_category_patch_loss(
            global_patch,
            boxes,
            [target],
            candidate,
            drop_positive_anchor_gradient_policy=(
                DATA_DRIVEN_PATCH_DROP_ANCHOR_GLOBAL_MAX
            ),
            **kwargs,
        )
        balanced_result = deployment_gate_category_patch_loss(
            balanced_patch,
            boxes,
            [target],
            candidate,
            drop_positive_anchor_gradient_policy=(
                DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED
            ),
            **kwargs,
        )
        for key in (
            "standardized_patch_score",
            "deployed_gate",
            "category_positive_mask",
            "category_negative_mask",
            "loss",
            "keep_component",
            "keep_objective_component",
            "drop_component",
            "drop_objective_component",
            "drop_active_unsafe_component",
        ):
            self.assertTrue(
                torch.equal(global_result[key], balanced_result[key]), key
            )

        global_result["drop_component"].backward(retain_graph=True)
        balanced_result["drop_component"].backward(retain_graph=True)
        self.assertLess(float(global_patch.grad[0, 0]), 0.0)
        self.assertEqual(float(global_patch.grad[0, 1]), 0.0)
        self.assertLess(float(balanced_patch.grad[0, 0]), 0.0)
        self.assertLess(float(balanced_patch.grad[0, 1]), 0.0)
        self.assertAlmostEqual(float(balanced_patch.grad.sum()), 0.0, places=6)

        global_patch.grad.zero_()
        balanced_patch.grad.zero_()
        global_result["drop_active_unsafe_component"].backward()
        balanced_result["drop_active_unsafe_component"].backward()
        self.assertLess(float(balanced_patch.grad[0, 0]), 0.0)
        self.assertLess(float(balanced_patch.grad[0, 1]), 0.0)
        self.assertGreater(float(balanced_patch.grad[0, 2]), 0.0)
        self.assertGreater(float(balanced_patch.grad[0, 3]), 0.0)
        self.assertAlmostEqual(float(balanced_patch.grad.sum()), 0.0, places=6)

    def test_patch_keep_requires_role_exclusive_geometry(self):
        target = {
            **_category_target(include_unassigned=False),
            "boxes": torch.tensor(
                [
                    [0.4, 0.5, 0.25, 0.2],
                    [0.6, 0.5, 0.25, 0.2],
                ],
                dtype=torch.float32,
            ),
        }
        boxes = torch.full((1, 12, 4), 0.05, dtype=torch.float32)
        boxes[0, 0] = torch.tensor([0.5, 0.5, 0.45, 0.2])
        boxes[0, 1] = target["boxes"][0]
        boxes[0, 2] = target["boxes"][1]
        patch = torch.zeros((1, 12), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            patch[0, 0] = 10.0
        result = deployment_gate_category_patch_loss(
            patch,
            boxes,
            [target],
            torch.ones((1, 12), dtype=torch.bool),
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            boundary_margin=0.25,
            temperature=0.1,
            role_exclusive_keep=True,
        )
        exclusive = result["role_exclusive_positive_mask"]
        self.assertEqual(
            exclusive.nonzero(as_tuple=False).tolist(),
            [[0, 1, 0], [0, 2, 1]],
        )
        self.assertEqual(
            float(result["role_exclusive_reachable_instances"]), 2.0
        )
        self.assertEqual(
            float(result["role_exclusive_unreachable_instances"]), 0.0
        )
        self.assertEqual(
            float(result["role_exclusive_keep_deployed_instances"]), 0.0
        )
        self.assertGreater(
            float(result["role_exclusive_keep_component"].detach()),
            float(result["keep_mean_component"].detach()),
        )
        result["loss"].backward()
        self.assertGreater(float(patch.grad[0, 0]), 0.0)
        self.assertLess(float(patch.grad[0, 1:3].sum()), 0.0)

    def test_role_routed_rejects_patch_rank_geometry_threshold_drift(self):
        with self.assertRaisesRegex(ValueError, "thresholds must match"):
            StageBDataDrivenCriterion(
                train_mode="rank_patch_only",
                category_complete=True,
                rank_supervision=(
                    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED
                ),
                assignment_weight=1.0,
                rank_negative_iou_threshold=0.3,
                patch_negative_iou_threshold=0.2,
            )

    def test_role_exclusive_keep_skips_unreachable_and_invalid_directions(self):
        target = {
            **_category_target(include_unassigned=False),
            "boxes": torch.tensor(
                [
                    [0.4, 0.5, 0.25, 0.2],
                    [0.6, 0.5, 0.25, 0.2],
                ],
                dtype=torch.float32,
            ),
        }
        boxes = torch.full((1, 4, 4), 0.05, dtype=torch.float32)
        boxes[0, 0] = torch.tensor([0.5, 0.5, 0.45, 0.2])
        boxes[0, 1] = target["boxes"][0]
        boxes[0, 2] = target["boxes"][1]
        candidate = torch.tensor([[True, False, True, True]])
        kwargs = dict(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            boundary_margin=0.25,
            temperature=0.1,
            role_exclusive_keep=True,
        )
        one_sided = deployment_gate_category_patch_loss(
            torch.zeros((1, 4)), boxes, [target], candidate, **kwargs
        )
        self.assertEqual(
            one_sided["role_exclusive_positive_mask"]
            .nonzero(as_tuple=False)
            .tolist(),
            [[0, 2, 1]],
        )
        self.assertEqual(
            float(one_sided["role_exclusive_reachable_instances"]), 1.0
        )
        self.assertEqual(
            float(one_sided["role_exclusive_unreachable_instances"]), 1.0
        )
        invalid_target = dict(target)
        invalid_target["stage_b_data_driven_assignment_valid"] = torch.tensor(
            [False]
        )
        invalid = deployment_gate_category_patch_loss(
            torch.zeros((1, 4)),
            boxes,
            [invalid_target],
            candidate,
            **kwargs,
        )
        self.assertEqual(
            float(invalid["role_exclusive_reachable_instances"]), 0.0
        )
        self.assertEqual(
            float(invalid["role_exclusive_unreachable_instances"]), 0.0
        )
        self.assertTrue(bool(torch.isfinite(invalid["loss"]).item()))

    def test_criterion_separates_text_and_patch_roles(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=(
                DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED
            ),
            assignment_weight=1.0,
            deployment_weight=0.0,
            patch_dense_category_focal_weight=0.0,
            patch_drop_positive_anchor_gradient_policy=(
                DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED
            ),
        )
        self.assertEqual(int(criterion.criterion_contract_version.item()), 18)
        self.assertEqual(int(criterion.rank_supervision_contract_id.item()), 6)
        self.assertEqual(
            criterion.patch_active_unsafe_auxiliary_weight, 1.0
        )
        self.assertEqual(
            (
                criterion.patch_dense_category_focal_weight,
                criterion.patch_dense_category_focal_alpha,
                criterion.patch_dense_category_focal_gamma,
                criterion.patch_dense_category_focal_negative_weight,
            ),
            (0.0, 0.25, 2.0, 1.0),
        )
        self.assertEqual(
            set(criterion.weight_dict),
            {
                "loss_stage_b_data_driven_role_routed_rank",
                "loss_stage_b_data_driven_patch",
            },
        )
        rank = torch.zeros(1, 4, 2, requires_grad=True)
        patch = torch.tensor(
            [[[0.5], [0.2], [0.1], [2.0]]], requires_grad=True
        )
        outputs = {
            "stage_b_data_driven_text_rank_score": rank,
            "stage_b_data_driven_candidate_mask": torch.ones(
                1, 4, 2, dtype=torch.bool
            ),
            "pred_logits_patch": patch,
            "pred_boxes": torch.tensor(
                [
                    [
                        [0.2, 0.5, 0.2, 0.2],
                        [0.8, 0.5, 0.2, 0.2],
                        [0.5, 0.8, 0.2, 0.2],
                        [0.5, 0.1, 0.1, 0.1],
                    ]
                ],
                dtype=torch.float32,
            ),
        }
        result = criterion(outputs, [_category_target()])
        result["loss_stage_b_data_driven_role_routed_rank"].backward()
        self.assertGreater(float(rank.grad.abs().sum()), 0.0)
        self.assertIsNone(patch.grad)
        rank.grad.zero_()
        result["loss_stage_b_data_driven_patch"].backward()
        self.assertEqual(float(rank.grad.abs().sum()), 0.0)
        self.assertGreater(float(patch.grad.abs().sum()), 0.0)
        self.assertEqual(
            float(
                result[
                    "stage_b_data_driven_assignment_category_negative_queries"
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(result["stage_b_data_driven_assignment_neutral_queries"]),
            0.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_data_driven_assignment_safe_sibling_queries"
                ]
            ),
            4.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_data_driven_assignment_paired_sibling_queries"
                ]
            ),
            2.0,
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_data_driven_patch_role_exclusive_reachable_instances"
                ]
            ),
            2.0,
        )

    def test_optimizer_excludes_and_freezes_deployment_inert_scale(self):
        model = _RoleModel()
        groups, _active = _stage_b_data_driven_parameter_groups(
            model, "rank_patch_only"
        )
        scale_id = id(model.patch_logit_scale)
        self.assertFalse(
            any(
                id(parameter) == scale_id
                for parameters in groups.values()
                for parameter in parameters
            )
        )
        _freeze_and_audit_stage_b_data_driven(model, "rank_patch_only")
        self.assertFalse(model.patch_logit_scale.requires_grad)

    def test_criterion_contract_rejects_cross_mode_same_shape_state(self):
        role = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=(
                DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED
            ),
            assignment_weight=1.0,
            patch_drop_positive_anchor_gradient_policy=(
                DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED
            ),
        )
        state = role.state_dict()
        self.assertEqual(
            validate_data_driven_criterion_checkpoint_state(
                state,
                expected_rank_supervision=(
                    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED
                ),
                checkpoint_label="role",
                allow_legacy_eval_contract=False,
                require_cold_rank_queue=True,
            ),
            (18, 6),
        )
        crossed = dict(state)
        crossed["criterion_contract_version"] = torch.tensor(
            4, dtype=torch.int64
        )
        crossed["rank_supervision_contract_id"] = torch.tensor(
            4, dtype=torch.int64
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_data_driven_criterion_checkpoint_state(
                crossed,
                expected_rank_supervision=(
                    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED
                ),
                checkpoint_label="crossed",
                allow_legacy_eval_contract=False,
                require_cold_rank_queue=True,
            )

    def test_global_max_confidence_rejects_proposal_scoped_tn(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="confidence_pair",
            category_complete=True,
            token_weight=0.0,
            positive_queue_size=0,
        )
        outputs = {
            "stage_b_data_driven_confidence_score": torch.zeros(1, 2, 2),
            "stage_b_data_driven_confidence_token_logits": torch.zeros(
                1, 2, 2, 3
            ),
            "stage_b_data_driven_expression_token_mask": torch.ones(
                1, 2, 3, dtype=torch.bool
            ),
            "stage_b_data_driven_expression_input_ids": torch.ones(
                1, 2, 3, dtype=torch.int64
            ),
            "stage_b_data_driven_candidate_mask": torch.ones(
                1, 2, 2, dtype=torch.bool
            ),
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]
            ),
        }
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "stage_b_data_driven_expression_captions": [
                "a pink box .",
                "a blue box .",
            ],
            "stage_b_data_driven_trace": {
                "category": "color",
                "replace_from": "pink",
                "replace_to": "blue",
                "replace_span": [1, 2],
            },
            "tn_scope": "proposal_covered_verified",
            "global_tn_verified": torch.tensor([False]),
        }
        with self.assertRaisesRegex(ValueError, "image-global"):
            criterion(outputs, [target])
        target["tn_scope"] = "image_global_topk_verified"
        target["global_tn_verified"] = torch.tensor([True])
        result = criterion(outputs, [target])
        self.assertTrue(
            torch.isfinite(
                result["loss_stage_b_data_driven_confidence"]
            ).item()
        )
        target["stage_b_data_driven_trace"]["replace_span"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "exactly reconstruct"):
            criterion(outputs, [target])

    def test_paired_slot_scoring_equals_duplicated_single_scoring(self):
        torch.manual_seed(17)
        heads = StageBDataDrivenScoreHeads(
            4,
            rank_dim=3,
            confidence_dim=3,
            gate_hidden_dim=4,
        ).eval()
        query = torch.randn(1, 5, 4)
        text = torch.randn(2, 4, 4)
        mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )
        duplicated = heads(
            query.expand(2, -1, -1).contiguous(), text, mask
        )
        singles = [
            heads(query, text[index : index + 1], mask[index : index + 1])
            for index in range(2)
        ]
        for key in (
            "text_rank_score",
            "confidence_score",
            "confidence_token_logits",
        ):
            expected = torch.cat([item[key] for item in singles], dim=0)
            self.assertTrue(torch.equal(duplicated[key], expected))


if __name__ == "__main__":
    unittest.main()
