import unittest
from collections import OrderedDict
from unittest import mock

import torch

from models.GroundingDINO.stage_b_data_driven_score import (
    data_driven_tensor_state_sha256,
)
from tools.build_stageb_data_driven_relational_initializer import (
    RelationalInitializerError,
    compose_relational_state,
)


class BuildStageBDataDrivenRelationalInitializerTest(unittest.TestCase):
    def _states(self):
        b58 = OrderedDict(
            [("backbone.weight", torch.tensor([1.0, 2.0]))]
        )
        template = OrderedDict(
            [
                ("backbone.weight", torch.zeros(2)),
                ("patch_encoder.backbone.weight", torch.zeros(2)),
                ("patch_logit_scale", torch.tensor(0.25)),
                (
                    "stage_b_data_driven_score_heads.rank_branch.weight",
                    torch.tensor([0.5]),
                ),
                (
                    "stage_b_data_driven_score_heads.confidence_branch.weight",
                    torch.tensor([0.75]),
                ),
                (
                    "stage_b_data_driven_score_heads._contract_version",
                    torch.tensor(3, dtype=torch.int64),
                ),
            ]
        )
        return template, b58

    def test_exact_roles_and_b58_alias(self):
        template, b58 = self._states()
        patch_hash = data_driven_tensor_state_sha256(
            template, ["patch_logit_scale"]
        )
        confidence_hash = data_driven_tensor_state_sha256(
            template,
            ["stage_b_data_driven_score_heads.confidence_branch.weight"],
        )
        with mock.patch(
            "tools.build_stageb_data_driven_relational_initializer."
            "EXPECTED_A0_PATCH_TENSOR_SHA256",
            patch_hash,
        ), mock.patch(
            "tools.build_stageb_data_driven_relational_initializer."
            "EXPECTED_A0_CONFIDENCE_TENSOR_SHA256",
            confidence_hash,
        ):
            state, roles = compose_relational_state(template, b58)
        self.assertEqual(set().union(*map(set, roles.values())), set(template))
        self.assertEqual(sum(map(len, roles.values())), len(template))
        self.assertTrue(
            torch.equal(
                state["backbone.weight"],
                state["patch_encoder.backbone.weight"],
            )
        )
        self.assertEqual(
            roles["random_relational_rank"],
            ["stage_b_data_driven_score_heads.rank_branch.weight"],
        )
        self.assertEqual(
            roles["random_absolute_confidence"],
            ["stage_b_data_driven_score_heads.confidence_branch.weight"],
        )

    def test_unknown_or_teacher_tensor_fails_closed(self):
        for key in (
            "unknown_projection.weight",
            "stage_b_gdino_score_adapter.rank.weight",
            "stage_b_u0_patch_rank_adapter.weight",
        ):
            template, b58 = self._states()
            template[key] = torch.ones(1)
            with self.assertRaisesRegex(
                RelationalInitializerError, "unbound template tensor"
            ):
                compose_relational_state(template, b58)

    def test_patch_initialization_must_match_absolute_control(self):
        template, b58 = self._states()
        with self.assertRaisesRegex(
            RelationalInitializerError, "differs from the absolute A0"
        ):
            compose_relational_state(template, b58)

    def test_confidence_initialization_must_match_absolute_control(self):
        template, b58 = self._states()
        patch_hash = data_driven_tensor_state_sha256(
            template, ["patch_logit_scale"]
        )
        with mock.patch(
            "tools.build_stageb_data_driven_relational_initializer."
            "EXPECTED_A0_PATCH_TENSOR_SHA256",
            patch_hash,
        ), self.assertRaisesRegex(
            RelationalInitializerError, "confidence initialization differs"
        ):
            compose_relational_state(template, b58)


if __name__ == "__main__":
    unittest.main()
