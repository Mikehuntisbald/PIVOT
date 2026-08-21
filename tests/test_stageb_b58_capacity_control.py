import unittest

import torch

from models.GroundingDINO.stage_b_gdino_score_adapter import StageBGDINOScoreAdapter
from tools.train_stageb_u2v5_ownership import (
    _TaskSpecificOptimizer,
    _load_capacity_initializer,
)


class _AdapterModel(torch.nn.Module):
    def __init__(self, adapter_dim, gate_dim, ownership):
        super().__init__()
        self.stage_b_gdino_score_adapter = StageBGDINOScoreAdapter(
            256,
            adapter_dim=adapter_dim,
            gate_hidden_dim=gate_dim,
            u2v5_score_ownership=ownership,
        )


class B58CapacityControlTests(unittest.TestCase):
    def test_shared_wide_matches_isolated_capacity_and_macs(self):
        wide = _AdapterModel(163, 62, "shared_wide_two_heads")
        isolated = _AdapterModel(128, 128, "isolated_heads")
        wide_adapter = wide.stage_b_gdino_score_adapter
        isolated_adapter = isolated.stage_b_gdino_score_adapter
        wide_parameters = sum(
            value.numel()
            for value in (
                *wide_adapter.rank_parameters(),
                *wide_adapter.confidence_gate.parameters(),
            )
        )
        isolated_parameters = sum(
            value.numel() for value in isolated_adapter.gate_parameters()
        )
        self.assertEqual(wide_parameters, 83971)
        self.assertEqual(isolated_parameters, 83969)
        self.assertEqual(wide_parameters - isolated_parameters, 2)
        self.assertEqual(83007 - 82944, 63)

    def test_zero_padded_transplant_preserves_rank_and_identity_confidence(self):
        torch.manual_seed(7)
        source = _AdapterModel(128, 128, "shared_trunk_two_heads")
        torch.nn.init.normal_(source.stage_b_gdino_score_adapter.rank_output.weight)
        torch.nn.init.normal_(source.stage_b_gdino_score_adapter.rank_output.bias)
        torch.manual_seed(11)
        target = _AdapterModel(163, 62, "shared_wide_two_heads")
        normalized, audit = _load_capacity_initializer(
            target, {"model": source.state_dict()}, "B58_SHARED_WIDE"
        )
        self.assertEqual(audit["mode"], "zero_padded_R100_transplant")
        self.assertLessEqual(audit["rank_residual_max_abs_error"], 1e-6)
        self.assertTrue(audit["confidence_identity_bitwise"])
        self.assertEqual(set(normalized), set(target.state_dict()))

    def test_two_optimizers_keep_independent_moments_for_shared_parameter(self):
        shared = torch.nn.Parameter(torch.tensor([1.0]))
        rank_head = torch.nn.Parameter(torch.tensor([1.0]))
        confidence_head = torch.nn.Parameter(torch.tensor([1.0]))
        admission = torch.optim.AdamW(
            [{"params": [shared, rank_head]}], lr=0.1, weight_decay=0.0, foreach=False
        )
        confidence = torch.optim.AdamW(
            [{"params": [shared, confidence_head]}], lr=0.1, weight_decay=0.0, foreach=False
        )
        wrapper = _TaskSpecificOptimizer(admission, confidence)
        wrapper.set_task("admission")
        (shared + rank_head).backward()
        wrapper.step()
        wrapper.zero_grad()
        self.assertIn(shared, admission.state)
        self.assertNotIn(shared, confidence.state)
        wrapper.set_task("confidence")
        (2.0 * shared + confidence_head).backward()
        wrapper.step()
        wrapper.zero_grad()
        self.assertIn(shared, confidence.state)
        self.assertIsNot(admission.state[shared], confidence.state[shared])
        state = wrapper.state_dict()
        self.assertEqual(state["schema"], "arrow.stageb.task_specific_adamw/v1")
        self.assertTrue(all(group["weight_decay"] == 0.0 for group in admission.param_groups))
        self.assertTrue(all(group["weight_decay"] == 0.0 for group in confidence.param_groups))


if __name__ == "__main__":
    unittest.main()
