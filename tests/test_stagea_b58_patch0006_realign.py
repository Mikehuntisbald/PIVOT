from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn

import main as main_module
from engine import _set_stage_a_b58_patch_realign_training_mode
from tools.build_stagea_b58_patch0006_initializer import (
    compose_model_state,
)
from tools import seal_stagea_launch_source as launch_seal
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stagea_b58_trunk_patch0006_realign_20260814.py"
)


def _states():
    b58 = OrderedDict(
        {
            "backbone.layer.weight": torch.tensor([58.0]),
            "transformer.decoder.layer.weight": torch.tensor([5.8]),
        }
    )
    patch0006 = OrderedDict(
        {
            "backbone.layer.weight": torch.tensor([6.0]),
            "transformer.decoder.layer.weight": torch.tensor([0.6]),
            "patch_logit_scale": torch.tensor(2.0),
            "patch_dn_tgt": torch.tensor([6.0]),
            "patch_encoder.backbone.layer.weight": torch.tensor([600.0]),
            "patch_encoder.input_proj.0.weight": torch.tensor([1.0]),
            "patch_encoder.norm.weight": torch.tensor([2.0]),
            "query_proj_for_patch.weight": torch.tensor([3.0]),
        }
    )
    return b58, patch0006


def test_initializer_preserves_b58_and_transfers_only_independent_patch_state():
    state, roles = compose_model_state(*_states())
    assert torch.equal(state["backbone.layer.weight"], torch.tensor([58.0]))
    assert torch.equal(
        state["transformer.decoder.layer.weight"], torch.tensor([5.8])
    )
    assert torch.equal(
        state["patch_encoder.backbone.layer.weight"], torch.tensor([58.0])
    )
    assert torch.equal(
        state["query_proj_for_patch.weight"], torch.tensor([3.0])
    )
    assert "patch_dn_tgt" not in state
    assert roles["disabled_query_state"] == ["patch_dn_tgt"]
    assert roles["patch0006_transfer"] == [
        "patch_logit_scale",
        "patch_encoder.input_proj.0.weight",
        "patch_encoder.norm.weight",
        "query_proj_for_patch.weight",
    ]


def test_initializer_rejects_unowned_patch_tensor():
    b58, patch0006 = _states()
    patch0006["patch_encoder.uncontracted.weight"] = torch.tensor([1.0])
    with pytest.raises(RuntimeError, match="unowned checkpoint0006 tensor"):
        compose_model_state(b58, patch0006)


class _PatchEncoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.bert = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.transformer = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.patch_encoder = _PatchEncoder(self.backbone)
        self.query_proj_for_patch = nn.Linear(2, 2)
        self.patch_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.patch_dn_tgt = None


def _set_projection_only_grad(model: _Model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.patch_encoder.input_proj,
        model.patch_encoder.norm,
        model.query_proj_for_patch,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.patch_logit_scale.requires_grad_(True)


def test_training_mode_keeps_b58_deterministic_and_patch_projection_trainable():
    model = _Model()
    _set_projection_only_grad(model)
    model.train()
    _set_stage_a_b58_patch_realign_training_mode(model)
    assert not model.training
    assert not model.backbone.training
    assert not model.bert.training
    assert not model.transformer.training
    assert model.patch_encoder.input_proj.training
    assert model.patch_encoder.norm.training
    assert model.query_proj_for_patch.training


def test_training_mode_rejects_trainable_decoder_state():
    model = _Model()
    _set_projection_only_grad(model)
    next(model.transformer.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable ownership drifted"):
        _set_stage_a_b58_patch_realign_training_mode(model)


def test_config_freezes_query_semantics_and_disables_dn_queries():
    cfg = SLConfig.fromfile(str(CONFIG))
    assert cfg.stage_a_b58_patch_realign is True
    assert cfg.patch_only is True
    assert cfg.patch_dn_num_queries == 0
    assert cfg.unfreeze_decoder_last_n_layers == 0
    assert cfg.batch_size == 38
    assert cfg.amp_init_scale == 65_536.0
    assert cfg.amp_growth_interval == 1_000_000
    assert cfg.only_train_keywords == [
        "patch_encoder.input_proj",
        "patch_encoder.norm",
        "query_proj_for_patch",
        "patch_logit_scale",
    ]


def test_stagea_grad_scaler_disables_growth_probes(monkeypatch):
    observed = {}

    class _Scaler:
        def __init__(self, device, **kwargs):
            observed["device"] = device
            observed.update(kwargs)

    monkeypatch.setattr(main_module.torch.amp, "GradScaler", _Scaler)
    main_module._make_grad_scaler(
        enabled=True,
        init_scale=65_536.0,
        growth_interval=1_000_000,
    )
    assert observed == {
        "device": "cuda",
        "enabled": True,
        "init_scale": 65_536.0,
        "growth_interval": 1_000_000,
    }


@pytest.mark.parametrize("value", [True, 0, -1, 2.5])
def test_grad_scaler_rejects_invalid_growth_interval(value):
    with pytest.raises(ValueError, match="positive integer"):
        main_module._make_grad_scaler(enabled=True, growth_interval=value)


def test_launch_source_manifest_rejects_sealed_artifact_drift(tmp_path):
    info = tmp_path / "info.txt"
    patch = tmp_path / launch_seal.PATCH_NAME
    manifest = tmp_path / launch_seal.MANIFEST_NAME
    info.write_text("launch evidence\n", encoding="utf-8")
    patch.write_text("tracked diff\n", encoding="utf-8")
    payload = {
        "schema": launch_seal.SCHEMA,
        "launch_artifacts": [launch_seal._file_record(info)],
        "tracked_worktree_patch": launch_seal._file_record(patch),
    }
    payload["manifest_sha256"] = launch_seal._canonical_sha256(payload)
    manifest.write_text(
        launch_seal.json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert launch_seal.verify_manifest(manifest) == payload
    patch.write_text("drifted diff\n", encoding="utf-8")
    with pytest.raises(launch_seal.LaunchSourceSealError, match="drifted"):
        launch_seal.verify_manifest(manifest)
