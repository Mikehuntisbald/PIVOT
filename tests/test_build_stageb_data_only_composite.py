import unittest
from collections import OrderedDict

import torch

from models.GroundingDINO.stage_b_u0_patch_rank import U0_PATCH_SOURCE_KEYS
from tools.build_stageb_data_only_composite import (
    CONTRACT_KEY,
    DataOnlyCompositeError,
    SCHEMA,
    compose_data_only_model_state,
    validate_data_only_composite_payload,
)
from models.GroundingDINO.stage_b_u0_patch_rank import (
    stage_b_u0_tensor_state_sha256,
)


class BuildStageBDataOnlyCompositeTest(unittest.TestCase):
    def _source_states(self):
        d9 = OrderedDict(
            {
                "backbone.weight": torch.tensor([1.0, 2.0]),
                "patch_encoder.backbone.weight": torch.tensor([1.0, 2.0]),
            }
        )
        for index, key in enumerate(sorted(U0_PATCH_SOURCE_KEYS)):
            d9[key] = torch.full((1,), float(index + 10))
        u0 = OrderedDict((key, value.clone()) for key, value in d9.items())
        for key in U0_PATCH_SOURCE_KEYS:
            u0[key].add_(100.0)
        u0["stage_b_gdino_score_adapter.rank_output.weight"] = torch.tensor(
            [[3.0]]
        )
        u0["stage_b_u0_patch_rank_adapter._contract_version"] = torch.tensor(1)
        u0["stage_b_u0_patch_rank_adapter.output.weight"] = torch.zeros(1)
        u0["stage_b_u0_patch_rank_adapter.output.bias"] = torch.zeros(1)
        return u0, d9

    def _payload(self):
        u0, d9 = self._source_states()
        state, _roles = compose_data_only_model_state(u0, d9)
        roles = {
            "legacy_r100_p50": [
                "backbone.weight",
                "stage_b_gdino_score_adapter.rank_output.weight",
            ],
            "shared_backbone_alias": ["patch_encoder.backbone.weight"],
            "d9_patch": sorted(U0_PATCH_SOURCE_KEYS),
            "u0_adapter": [
                "stage_b_u0_patch_rank_adapter._contract_version",
                "stage_b_u0_patch_rank_adapter.output.weight",
                "stage_b_u0_patch_rank_adapter.output.bias",
            ],
        }
        config_contract = {
            "stage_b_data_only_composite": True,
            "stage_b_data_only_composite_contract_version": 1,
            "stage_b_u0_patch_rank": True,
            "stage_b_gdino_score_adapter": True,
            "enable_patch_branch": True,
            "patch_gate_with_text": False,
            "stage_b_u0_category_preserving_patch_gate": True,
            "stage_b_u0_category_gate_max_gap": 3.0,
            "stage_b_u2_category_complete_supervision": False,
        }
        contract = {
            "schema": SCHEMA,
            "eval_only": True,
            "resumable": False,
            "single_checkpoint": True,
            "single_model_root": True,
            "external_score_source_required": False,
            "model_state_keys": len(state),
            "role_keys": roles,
            "role_key_counts": {key: len(value) for key, value in roles.items()},
            "config_contract": config_contract,
            "invariants": {"all_sources_are_stage_b_data": True},
        }
        for role, keys in roles.items():
            contract[f"{role}_tensor_sha256"] = stage_b_u0_tensor_state_sha256(
                state, keys
            )
        contract["full_model_tensor_sha256"] = stage_b_u0_tensor_state_sha256(
            state, state.keys()
        )
        return state, {"model": state, CONTRACT_KEY: contract}

    def test_compose_replaces_exactly_nine_patch_tensors(self):
        u0, d9 = self._source_states()
        state, roles = compose_data_only_model_state(u0, d9)
        self.assertEqual(set(roles["d9_patch"]), set(U0_PATCH_SOURCE_KEYS))
        for key in U0_PATCH_SOURCE_KEYS:
            self.assertTrue(torch.equal(state[key], d9[key]))
        for key in set(u0) - set(U0_PATCH_SOURCE_KEYS):
            self.assertTrue(torch.equal(state[key], u0[key]))

    def test_compose_rejects_nonpatch_shared_difference(self):
        u0, d9 = self._source_states()
        d9["backbone.weight"].add_(1.0)
        with self.assertRaisesRegex(DataOnlyCompositeError, "exactly the nine"):
            compose_data_only_model_state(u0, d9)

    def test_compose_rejects_missing_patch_tensor(self):
        u0, d9 = self._source_states()
        d9.pop(next(iter(U0_PATCH_SOURCE_KEYS)))
        with self.assertRaisesRegex(DataOnlyCompositeError, "missing patch"):
            compose_data_only_model_state(u0, d9)

    def test_fast_validator_accepts_sealed_single_checkpoint(self):
        state, payload = self._payload()
        validate_data_only_composite_payload(
            state, payload, checkpoint_label="valid synthetic composite"
        )

    def test_fast_validator_rejects_tensor_and_top_level_drift(self):
        state, payload = self._payload()
        drifted_state = OrderedDict(
            (key, value.clone()) for key, value in state.items()
        )
        drifted_state[next(iter(U0_PATCH_SOURCE_KEYS))].add_(1.0)
        drifted = {"model": drifted_state, CONTRACT_KEY: payload[CONTRACT_KEY]}
        with self.assertRaisesRegex(ValueError, "tensor hash drifted"):
            validate_data_only_composite_payload(
                state, drifted, checkpoint_label="drifted composite"
            )

        with self.assertRaisesRegex(ValueError, "top-level keys"):
            validate_data_only_composite_payload(
                state,
                {"model": state},
                checkpoint_label="missing contract",
            )


if __name__ == "__main__":
    unittest.main()
