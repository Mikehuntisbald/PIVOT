import unittest
from collections import OrderedDict

import torch

from models.GroundingDINO.stage_b_u0_patch_rank import (
    U1_DIRECT_PATCH_ADDED_KEYS,
    U1_DIRECT_PATCH_INITIALIZER_SCHEMA,
    U1_DIRECT_PATCH_REPLACED_KEYS,
    stage_b_u0_tensor_state_sha256,
    validate_stage_b_u1_direct_patch_initializer_payload,
)
from tools.build_stageb_u1_direct_patch_initializer import (
    U1InitializerError,
    compose_u1_model_state,
)


class BuildStageBU1DirectPatchInitializerTest(unittest.TestCase):
    def _states(self):
        source = OrderedDict(
            {
                "backbone.weight": torch.tensor([1.0]),
                "stage_b_u0_patch_rank_adapter.output.weight": torch.tensor([2.0]),
                "stage_b_u0_patch_rank_adapter._contract_version": torch.tensor(1),
            }
        )
        template = OrderedDict(
            (key, torch.zeros_like(value)) for key, value in source.items()
        )
        template["stage_b_u0_patch_rank_adapter._contract_version"] = torch.tensor(2)
        template["stage_b_u0_patch_rank_adapter.direct_patch_gain"] = torch.tensor(0.0)
        template[
            "stage_b_u0_patch_rank_adapter._contract_direct_patch_gain_limit"
        ] = torch.tensor(0.5)
        return template, source

    def _payload(self):
        template, source = self._states()
        state, roles = compose_u1_model_state(template, source)
        contract = {
            "schema": U1_DIRECT_PATCH_INITIALIZER_SCHEMA,
            "role_keys": roles,
            "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, list(state)
            ),
            "source_preserved_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, roles["source_preserved"]
            ),
            "u1_added_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, roles["u1_added"]
            ),
            "u1_replaced_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, roles["u1_replaced"]
            ),
            "u100_functional_bitwise": {
                "teacher_rank_score": True,
                "patch_rank_residual": True,
                "rank_score": True,
            },
            "invariants": {
                "u100_source_preserved_bitwise": True,
                "direct_patch_gain_zero": True,
                "u1_rank_equals_u100_at_initialization": True,
                "r100_p50_b58_frozen_source_unchanged": True,
            },
        }
        return template, {"model": state, "u1_initializer": contract}

    def test_compose_preserves_source_and_only_adds_declared_u1_state(self):
        template, source = self._states()
        state, roles = compose_u1_model_state(template, source)
        self.assertTrue(torch.equal(state["backbone.weight"], source["backbone.weight"]))
        self.assertEqual(set(roles["u1_added"]), set(U1_DIRECT_PATCH_ADDED_KEYS))
        self.assertEqual(set(roles["u1_replaced"]), set(U1_DIRECT_PATCH_REPLACED_KEYS))
        self.assertEqual(int(state[next(iter(U1_DIRECT_PATCH_REPLACED_KEYS))]), 2)

    def test_compose_rejects_unbound_or_unconsumed_state(self):
        template, source = self._states()
        template["unknown.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(U1InitializerError, "unbound"):
            compose_u1_model_state(template, source)
        template, source = self._states()
        source["stale.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(U1InitializerError, "coverage"):
            compose_u1_model_state(template, source)

    def test_runtime_validator_accepts_exact_payload_and_rejects_drift(self):
        expected, payload = self._payload()
        validate_stage_b_u1_direct_patch_initializer_payload(
            expected, payload, checkpoint_label="valid"
        )
        drift = {
            "model": OrderedDict(
                (key, value.clone()) for key, value in payload["model"].items()
            ),
            "u1_initializer": dict(payload["u1_initializer"]),
        }
        drift["model"]["stage_b_u0_patch_rank_adapter.direct_patch_gain"].fill_(0.1)
        with self.assertRaisesRegex(ValueError, "full_model_tensor_sha256"):
            validate_stage_b_u1_direct_patch_initializer_payload(
                expected, drift, checkpoint_label="drift"
            )


if __name__ == "__main__":
    unittest.main()
