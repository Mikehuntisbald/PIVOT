import unittest

import torch

from models.GroundingDINO.stage_b_gdino_score_adapter import StageBGDINOScoreAdapter
from models.GroundingDINO.stage_b_u0_patch_rank import stage_b_u0_tensor_state_sha256
from tools.stageb_b58_capacity_control_contract import (
    CapacityControlContractError,
    validate_b58_capacity_runtime_payload,
)
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

    def _capacity_payload(self, model, row_id="B58_SHARED_WIDE"):
        state = model.state_dict()
        frozen = sorted(state)
        expected = {
            "B58_SHARED_WIDE": {
                "config": "config/ablations/cfg_stageb_b58_capacity_shared_wide.py",
                "ownership": "shared_wide_two_heads",
                "capacity": {
                    "trainable_parameters": 352138,
                    "score_owner_parameters": 83971,
                    "score_macs_per_query_and_output": 83007,
                    "representation_dim": 163,
                    "gate_hidden_dim": 62,
                },
            },
            "B58_ISOLATED_REPLAY": {
                "config": "config/ablations/cfg_stageb_b58_capacity_isolated_replay.py",
                "ownership": "isolated_heads",
                "capacity": {
                    "trainable_parameters": 352136,
                    "score_owner_parameters": 83969,
                    "score_macs_per_query_and_output": 82944,
                    "representation_dim": 128,
                    "gate_hidden_dim": 128,
                },
            },
        }[row_id]
        gradient = (
            {"diagnostic_pairs": 1}
            if row_id == "B58_SHARED_WIDE"
            else {"structural_isolation_checks": 150, "structural_cross_gradients": 0}
        )
        return {
            "model": state,
            "u2v5_ownership": {
                "schema": "arrow.stageb.b58_capacity_control_checkpoint/v1",
                "row": {
                    "schema": "arrow.stageb.b58_capacity_control_row/v1",
                    "row_id": row_id,
                    "config": expected["config"],
                    "ownership": expected["ownership"],
                    "updates": 150,
                    "batch_size": 56,
                    "parent": "clean_initializer",
                },
                "frozen_keys": frozen,
                "trainable_keys": [],
                "frozen_tensor_sha256": stage_b_u0_tensor_state_sha256(state, frozen),
                "c100_confidence_imported": False,
                "exposure": {"admission": 100, "confidence": 50},
                "runtime_audit": {
                    "successful_optimizer_steps": 150,
                    "task_successful_steps": {"admission": 100, "confidence": 50},
                    "amp_skipped_optimizer_steps": 0,
                    "nonfinite_gradient_boundaries": 0,
                },
                "optimizer_ownership": {
                    "task_specific_states": True,
                    "weight_decay": 0.0,
                },
                "parameter_accounting": {
                    "capacity_control": expected["capacity"],
                    "trainable": expected["capacity"]["trainable_parameters"],
                },
                "gradient_audit": gradient,
            },
        }

    def test_capacity_runtime_contract_accepts_locked_wide_payload(self):
        model = _AdapterModel(163, 62, "shared_wide_two_heads")
        contract = validate_b58_capacity_runtime_payload(
            model,
            self._capacity_payload(model),
            row_id="B58_SHARED_WIDE",
            checkpoint_label="fixture",
        )
        self.assertEqual(contract["row"]["row_id"], "B58_SHARED_WIDE")

    def test_capacity_runtime_contract_rejects_weight_decay(self):
        model = _AdapterModel(163, 62, "shared_wide_two_heads")
        payload = self._capacity_payload(model)
        payload["u2v5_ownership"]["optimizer_ownership"]["weight_decay"] = 1e-4
        with self.assertRaises(CapacityControlContractError):
            validate_b58_capacity_runtime_payload(
                model,
                payload,
                row_id="B58_SHARED_WIDE",
                checkpoint_label="fixture",
            )


if __name__ == "__main__":
    unittest.main()
