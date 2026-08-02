import unittest
from collections import OrderedDict

import torch

from tools.build_stageb_u0_initializer import (
    PATCH_SOURCE_KEYS,
    SCHEMA,
    U0InitializerError,
    compose_u0_model_state,
    tensor_state_sha256,
)
from models.GroundingDINO.stage_b_u0_patch_rank import (
    U0_SEALED_TEACHER_ARCHITECTURE_FIELDS,
    U0_TEACHER_FUNCTIONAL_FIELDS,
    validate_stage_b_u0_initializer_payload,
)


class BuildStageBU0InitializerTest(unittest.TestCase):
    def _states(self):
        merged = OrderedDict(
            {
                "backbone.block.weight": torch.tensor([1.0, 2.0]),
                "transformer.weight": torch.tensor([3.0]),
                "stage_b_gdino_score_adapter.rank_output.weight": torch.tensor(
                    [[4.0]]
                ),
            }
        )
        stagea = {
            key: torch.full((1,), float(index + 10))
            for index, key in enumerate(sorted(PATCH_SOURCE_KEYS))
        }
        stagea["patch_encoder.backbone.block.weight"] = torch.tensor(
            [-1.0, -2.0]
        )
        template = OrderedDict()
        template.update({key: torch.zeros_like(value) for key, value in merged.items()})
        template["patch_encoder.backbone.block.weight"] = torch.zeros(2)
        for key in sorted(PATCH_SOURCE_KEYS):
            template[key] = torch.zeros_like(stagea[key])
        template["stage_b_u0_patch_rank_adapter.output.weight"] = torch.zeros(1)
        template["stage_b_u0_patch_rank_adapter.output.bias"] = torch.zeros(1)
        return template, merged, stagea

    def _initializer_payload(self):
        template, merged, stagea = self._states()
        state, roles = compose_u0_model_state(template, merged, stagea)
        teacher_architecture = {
            key: False if key == "enable_patch_branch" else 1
            for key in U0_SEALED_TEACHER_ARCHITECTURE_FIELDS
        }
        u0_architecture = dict(teacher_architecture)
        u0_architecture["enable_patch_branch"] = True
        contract = {
            "schema": SCHEMA,
            "sealed_teacher_architecture": teacher_architecture,
            "u0_architecture": u0_architecture,
            "teacher_functional_bitwise": {
                key: True for key in U0_TEACHER_FUNCTIONAL_FIELDS
            },
            "role_keys": roles,
            "full_model_tensor_sha256": tensor_state_sha256(state, state.keys()),
            "merged_teacher_tensor_sha256": tensor_state_sha256(
                state, roles["merged"]
            ),
            "stagea_patch_tensor_sha256": tensor_state_sha256(
                state, roles["stagea_patch"]
            ),
            "shared_backbone_alias_tensor_sha256": tensor_state_sha256(
                state, roles["shared_backbone_alias"]
            ),
            "u0_zero_tensor_sha256": tensor_state_sha256(
                state, roles["u0_zero"]
            ),
            "invariants": {
                "merged_teacher_copied_bitwise": True,
                "stagea_patch_specific_keys_only": True,
                "stagea_patch_backbone_imported": False,
                "patch_backbone_aliases_source_b58": True,
                "u0_output_exactly_zero": True,
                "u0_rank_equals_r100_at_initialization": True,
                "p50_confidence_unchanged": True,
            },
        }
        return template, {"model": state, "u0_initializer": contract}

    def test_compose_uses_b58_for_alias_and_only_patch_specific_stagea_keys(self):
        template, merged, stagea = self._states()

        state, roles = compose_u0_model_state(template, merged, stagea)

        self.assertTrue(
            torch.equal(
                state["patch_encoder.backbone.block.weight"],
                merged["backbone.block.weight"],
            )
        )
        self.assertFalse(
            torch.equal(
                state["patch_encoder.backbone.block.weight"],
                stagea["patch_encoder.backbone.block.weight"],
            )
        )
        self.assertEqual(set(roles["stagea_patch"]), set(PATCH_SOURCE_KEYS))
        self.assertEqual(
            roles["shared_backbone_alias"],
            ["patch_encoder.backbone.block.weight"],
        )

    def test_compose_rejects_missing_patch_specific_tensor(self):
        template, merged, stagea = self._states()
        stagea.pop(next(iter(PATCH_SOURCE_KEYS)))
        with self.assertRaisesRegex(U0InitializerError, "key contract mismatch"):
            compose_u0_model_state(template, merged, stagea)

    def test_compose_rejects_unbound_template_tensor(self):
        template, merged, stagea = self._states()
        template["unowned.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(U0InitializerError, "unbound tensor"):
            compose_u0_model_state(template, merged, stagea)

    def test_runtime_validator_requires_full_sealed_initializer(self):
        expected, payload = self._initializer_payload()

        validate_stage_b_u0_initializer_payload(
            expected, payload, checkpoint_label="valid"
        )

        partial = dict(payload)
        partial["model"] = OrderedDict(payload["model"])
        partial["model"].pop("backbone.block.weight")
        with self.assertRaisesRegex(ValueError, "full-model key coverage"):
            validate_stage_b_u0_initializer_payload(
                expected, partial, checkpoint_label="partial"
            )

    def test_runtime_validator_rejects_tensor_or_contract_drift(self):
        expected, payload = self._initializer_payload()
        drifted = {
            "model": OrderedDict(
                (key, value.clone()) for key, value in payload["model"].items()
            ),
            "u0_initializer": dict(payload["u0_initializer"]),
        }
        drifted["model"]["transformer.weight"].add_(1.0)
        with self.assertRaisesRegex(ValueError, "full_model_tensor_sha256"):
            validate_stage_b_u0_initializer_payload(
                expected, drifted, checkpoint_label="drifted"
            )

        no_contract = {"model": payload["model"]}
        with self.assertRaisesRegex(ValueError, "top-level keys"):
            validate_stage_b_u0_initializer_payload(
                expected, no_contract, checkpoint_label="no-contract"
            )


if __name__ == "__main__":
    unittest.main()
