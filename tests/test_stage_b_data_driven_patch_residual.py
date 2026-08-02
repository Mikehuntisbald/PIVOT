import unittest

import torch
from torch import nn

from engine import _set_stage_b_data_driven_training_mode
from main import (
    _freeze_and_audit_stage_b_data_driven,
    _stage_b_data_driven_parameter_groups,
)
from models.GroundingDINO.stage_b_data_driven_patch_residual import (
    StageBDataDrivenPatchResidualMatcher,
    StageBDataDrivenTopKPatchResidualMatcher,
)
from models.GroundingDINO.stage_b_data_driven_score import (
    StageBDataDrivenScoreHeads,
)


class _PatchEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.ModuleList(
            [nn.Linear(256, 256), nn.Linear(256, 256)]
        )
        self.norm = nn.LayerNorm(256)


class _ResidualRoleModel(nn.Module):
    def __init__(self, residual=None):
        super().__init__()
        self.backbone = nn.Linear(256, 256)
        self.bert = nn.Linear(256, 256)
        self.transformer = nn.Linear(256, 256)
        self.patch_encoder = _PatchEncoder(self.backbone)
        self.query_proj_for_patch = nn.Linear(256, 256)
        self.patch_logit_scale = nn.Parameter(torch.tensor(2.0))
        self.stage_b_data_driven_score_heads = StageBDataDrivenScoreHeads(
            256,
            rank_dim=8,
            confidence_dim=8,
            gate_hidden_dim=8,
        )
        self.stage_b_data_driven_patch_residual = (
            residual or StageBDataDrivenPatchResidualMatcher()
        )


class StageBDataDrivenPatchResidualTest(unittest.TestCase):
    def test_topk_semantic_u0_multi_patch_and_legacy_migration_are_exact(self):
        legacy = StageBDataDrivenPatchResidualMatcher(center_raw=True)
        module = StageBDataDrivenTopKPatchResidualMatcher()
        parameters = module.trainable_parameters()
        self.assertEqual(len(parameters), 6)
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 135488)
        self.assertTrue(
            torch.equal(module.input.weight, legacy.input.weight)
        )
        self.assertTrue(torch.equal(module.input.bias, legacy.input.bias))
        self.assertTrue(
            torch.equal(module.output.weight, legacy.output.weight)
        )

        query = torch.randn(2, 7, 256)
        patch = torch.randn(2, 3, 256)
        base = torch.randn(2, 7, 3)
        residual = module(query, patch, base)
        self.assertTrue(torch.equal(residual, torch.zeros_like(base)))
        self.assertTrue(torch.equal(base.detach() + 14.0 * residual, base))
        permutation = torch.tensor([4, 0, 6, 1, 5, 3, 2])
        torch.testing.assert_close(
            module(
                query[:, permutation],
                patch,
                base[:, permutation],
            ),
            residual[:, permutation],
            rtol=0.0,
            atol=1e-7,
        )

        single_patch = patch[:, 0]
        single_base = base[:, :, 0]
        self.assertTrue(
            torch.equal(
                module(query, single_patch, single_base),
                torch.zeros_like(single_base),
            )
        )

    def test_topk_semantic_detaches_inputs_and_two_step_bootstraps(self):
        module = StageBDataDrivenTopKPatchResidualMatcher()
        query = torch.randn(2, 7, 256, requires_grad=True)
        patch = torch.randn(2, 256, requires_grad=True)
        base = torch.randn(2, 7, requires_grad=True)
        weights = torch.arange(7, dtype=query.dtype)[None, :]
        (module(query, patch, base) * weights).sum().backward()
        self.assertIsNone(query.grad)
        self.assertIsNone(patch.grad)
        self.assertIsNone(base.grad)
        self.assertGreater(
            float(module.output.weight.grad.detach().abs().sum()), 0.0
        )
        self.assertGreater(
            float(module.context_output.weight.grad.detach().abs().sum()), 0.0
        )
        for trunk in (module.input, module.context_input):
            self.assertEqual(
                sum(
                    float(parameter.grad.detach().abs().sum())
                    for parameter in trunk.parameters()
                ),
                0.0,
            )

        with torch.no_grad():
            module.output.weight.copy_(module.output.weight.grad * 1e-3)
            module.context_output.weight.copy_(
                module.context_output.weight.grad * 1e-3
            )
        module.zero_grad(set_to_none=True)
        (module(query, patch, base) * weights).sum().backward()
        for trunk in (module.input, module.context_input):
            self.assertGreater(
                sum(
                    float(parameter.grad.detach().abs().sum())
                    for parameter in trunk.parameters()
                ),
                0.0,
            )

    def test_topk_semantic_validates_base_score_alignment(self):
        module = StageBDataDrivenTopKPatchResidualMatcher()
        query = torch.randn(2, 7, 256)
        patch = torch.randn(2, 256)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            module(query, patch, torch.randn(2, 6))

    def test_topk_semantic_parameter_group_is_six_tensor_residual_only(self):
        matcher = StageBDataDrivenTopKPatchResidualMatcher()
        model = _ResidualRoleModel(residual=matcher)
        groups, active = _stage_b_data_driven_parameter_groups(
            model, "rank_patch_only"
        )
        self.assertEqual(
            {id(parameter) for parameter in groups["patch"]},
            {id(parameter) for parameter in matcher.parameters()},
        )
        self.assertEqual(len(groups["patch"]), 6)
        trainable = _freeze_and_audit_stage_b_data_driven(
            model, "rank_patch_only"
        )
        self.assertEqual(trainable, sum(parameter.numel() for parameter in active))
        self.assertTrue(all(parameter.requires_grad for parameter in matcher.parameters()))

    def test_zero_init_is_exact_bounded_and_permutation_equivariant(self):
        module = StageBDataDrivenPatchResidualMatcher()
        parameters = module.trainable_parameters()
        self.assertEqual(len(parameters), 3)
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 131328)
        query = torch.randn(2, 7, 256)
        patch = torch.randn(2, 3, 256)
        residual = module(query, patch)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        base = torch.randn_like(residual)
        adapted = base.detach() + torch.tensor(14.0) * residual
        self.assertTrue(torch.equal(adapted, base))
        permutation = torch.tensor([4, 0, 6, 1, 5, 3, 2])
        self.assertTrue(
            torch.equal(
                module(query[:, permutation], patch),
                residual[:, permutation],
            )
        )
        module.eval()
        self.assertTrue(torch.equal(module(query, patch), residual))
        with torch.no_grad():
            module.output.weight.fill_(1000.0)
        bounded = module(query, patch)
        self.assertLessEqual(float(bounded.detach().abs().max()), 0.25)

    def test_inputs_are_detached_and_zero_output_bootstraps_trunk_on_second_step(self):
        module = StageBDataDrivenPatchResidualMatcher()
        query = torch.randn(2, 5, 256, requires_grad=True)
        patch = torch.randn(2, 256, requires_grad=True)
        module(query, patch).sum().backward()
        self.assertIsNone(query.grad)
        self.assertIsNone(patch.grad)
        self.assertGreater(
            float(module.output.weight.grad.detach().abs().sum()), 0.0
        )
        self.assertEqual(
            float(module.input.weight.grad.detach().abs().sum()), 0.0
        )
        self.assertEqual(
            float(module.input.bias.grad.detach().abs().sum()), 0.0
        )

        with torch.no_grad():
            module.output.weight.copy_(module.output.weight.grad * 1e-3)
        module.zero_grad(set_to_none=True)
        module(query, patch).square().sum().backward()
        self.assertGreater(
            float(module.input.weight.grad.detach().abs().sum()), 0.0
        )
        self.assertGreater(
            float(module.input.bias.grad.detach().abs().sum()), 0.0
        )

    def test_raw_centering_precedes_tanh_and_preserves_zero_initializer(self):
        module = StageBDataDrivenPatchResidualMatcher(center_raw=True)
        query = torch.randn(2, 7, 256)
        patch = torch.randn(2, 256)
        self.assertEqual(
            module.architecture()["query_centering"],
            "raw_mean_before_tanh_v1",
        )
        self.assertTrue(
            torch.equal(module(query, patch), torch.zeros(2, 7))
        )
        with torch.no_grad():
            module.output.weight.normal_(mean=0.0, std=0.2)
            features = module._pair_features(query, patch)
            raw = module.output(
                torch.nn.functional.gelu(module.input(features))
            ).squeeze(-1)
            centered_raw = raw - raw.mean(dim=1, keepdim=True)
            expected = 0.25 * torch.tanh(centered_raw / 0.25)
        observed = module(query, patch)
        self.assertTrue(torch.equal(observed, expected))
        permutation = torch.tensor([4, 0, 6, 1, 5, 3, 2])
        torch.testing.assert_close(
            module(query[:, permutation], patch),
            observed[:, permutation],
            rtol=0.0,
            atol=1e-7,
        )

    def test_raw_centered_zero_output_bootstraps_with_relative_gradient(self):
        module = StageBDataDrivenPatchResidualMatcher(center_raw=True)
        query = torch.randn(2, 5, 256, requires_grad=True)
        patch = torch.randn(2, 256, requires_grad=True)
        weights = torch.arange(5, dtype=query.dtype)[None, :]
        (module(query, patch) * weights).sum().backward()
        self.assertIsNone(query.grad)
        self.assertIsNone(patch.grad)
        self.assertGreater(
            float(module.output.weight.grad.detach().abs().sum()), 0.0
        )
        self.assertEqual(
            float(module.input.weight.grad.detach().abs().sum()), 0.0
        )

    def test_residual_only_parameter_and_training_mode_contract(self):
        model = _ResidualRoleModel()
        groups, active = _stage_b_data_driven_parameter_groups(
            model, "rank_patch_only"
        )
        residual_ids = {
            id(parameter)
            for parameter in model.stage_b_data_driven_patch_residual.parameters()
        }
        self.assertEqual(
            {id(parameter) for parameter in groups["patch"]}, residual_ids
        )
        self.assertEqual(len(groups["patch"]), 3)
        trainable = _freeze_and_audit_stage_b_data_driven(
            model, "rank_patch_only"
        )
        self.assertEqual(trainable, sum(parameter.numel() for parameter in active))
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.stage_b_data_driven_patch_residual.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (
                    model.patch_encoder.input_proj,
                    model.patch_encoder.norm,
                    model.query_proj_for_patch,
                )
                for parameter in module.parameters()
            )
        )
        _set_stage_b_data_driven_training_mode(model, "rank_patch_only")
        self.assertTrue(model.stage_b_data_driven_patch_residual.training)
        self.assertFalse(model.patch_encoder.input_proj.training)
        self.assertFalse(model.patch_encoder.norm.training)
        self.assertFalse(model.query_proj_for_patch.training)


if __name__ == "__main__":
    unittest.main()
