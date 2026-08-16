import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from main import _freeze_and_audit_stage_b_u2v4_legacy_training_replay
from models.GroundingDINO.stage_b_u0_patch_rank import StageBU0PatchRankAdapter
from tools.stageb_u2v4_legacy_training_contract import (
    AUXILIARY_RESIDUAL_KEYS,
    SURFACE_PARAMETER_KEYS,
    TRAINABLE_KEYS,
    build_training_contract,
)
from tools.stageb_gdino_adapter_probe_audit import file_record


class _TinyReplayModel(torch.nn.Module):
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
        self.patch_logit_scale = torch.nn.Parameter(torch.ones(()))
        self.stage_b_u0_patch_rank_adapter = StageBU0PatchRankAdapter(
            query_count=4, hidden_dim=2
        )
        self.stage_b_gdino_score_adapter = torch.nn.Linear(2, 2)


class U2V4LegacyTrainingContractTest(unittest.TestCase):
    def test_freeze_exposes_surface_and_auxiliary_as_one_subsystem(self):
        model = _TinyReplayModel()
        count = _freeze_and_audit_stage_b_u2v4_legacy_training_replay(model)
        observed = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(observed, set(TRAINABLE_KEYS))
        self.assertEqual(len(observed), 16)
        self.assertGreater(count, 0)
        self.assertFalse(model.patch_logit_scale.requires_grad)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertFalse(
            any(
                p.requires_grad
                for p in model.stage_b_gdino_score_adapter.parameters()
            )
        )

    @mock.patch(
        "tools.stageb_u2v4_legacy_training_contract.validate_training_initializer_payload",
        return_value={"schema": "pivot.stageb.u2v4_training_initializer/v1"},
    )
    def test_contract_partitions_sixteen_trainable_tensors(self, _validate):
        state = {
            key: torch.zeros(1)
            for key in TRAINABLE_KEYS
        }
        for index in range(1165 - len(state)):
            state[f"frozen.{index:04d}"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initializer.pth"
            torch.save({"model": state, "u2v4_training_initializer": {}}, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            contract = build_training_contract(
                payload,
                initializer_path=path,
                initializer_sha256=file_record(path)["sha256"],
            )
        self.assertEqual(contract["trainable_tensor_count"], 16)
        self.assertEqual(contract["frozen_tensor_count"], 1149)
        self.assertEqual(
            contract["surface_parameter_keys"], list(SURFACE_PARAMETER_KEYS)
        )
        self.assertEqual(
            contract["auxiliary_residual_keys"], list(AUXILIARY_RESIDUAL_KEYS)
        )
        self.assertTrue(
            contract["training_mechanism"]["auxiliary_residual_trains_surface"]
        )


if __name__ == "__main__":
    unittest.main()
