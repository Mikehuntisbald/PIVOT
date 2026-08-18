from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.GroundingDINO.groundingdino import GroundingDINO
from models.GroundingDINO.stage_b_u0_patch_rank import StageBU0PatchRankAdapter
from tools.stageb_arrow_admission_contract import (
    SOURCES,
    null_sentinel,
    null_sentinel_sha256,
)
from tools.run_arrow_admission_matrix import _env, _parse
from tools.merge_arrow_admission_confidence import merge
from tools.eval_arrow_admission_panel import _summary as panel_summary
from tools.aggregate_arrow_admission_results import _panel_contrast
import numpy as np


class _Surface(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Conv2d(768, 256, 1), nn.GroupNorm(32, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(256)


def _stub():
    return SimpleNamespace(
        stage_b_arrow_admission_input_ablation=True,
        stage_b_arrow_source_spatial_size=7,
        patch_encoder=_Surface(),
        query_proj_for_patch=nn.Linear(256, 256),
        bert=SimpleNamespace(config=SimpleNamespace(hidden_size=768)),
    )


def test_shared_surface_projects_all_sources_to_one_normalized_token():
    stub = _stub()
    source = torch.randn(3, 768, requires_grad=True)
    projected = GroundingDINO._project_stage_b_arrow_source(stub, source)
    assert projected.shape == (3, 256)
    assert torch.allclose(projected.norm(dim=-1), torch.ones(3), atol=1e-5)
    projected.sum().backward()
    assert stub.patch_encoder.input_proj[0].weight.grad is not None


def test_null_sentinel_is_dense_deterministic_and_matches_model_route():
    expected = "c0afa83779cbb2cb1bec71ee42264973c14423a114b9da13aea8cc428bb0299a"
    assert null_sentinel_sha256() == expected
    assert int(torch.count_nonzero(null_sentinel())) == 768
    model_value = GroundingDINO._stage_b_arrow_null_source(
        _stub(), 2, torch.device("cpu"), torch.float32
    )
    digest = hashlib.sha256(model_value[0].contiguous().numpy().tobytes()).hexdigest()
    assert digest == expected
    assert torch.equal(model_value[0], model_value[1])


def test_canonical_source_pools_only_phrase_tokens():
    stub = _stub()
    encoded = torch.arange(2 * 4 * 768, dtype=torch.float32).view(2, 4, 768)
    mask = torch.tensor([[False, True, True, False], [False, False, True, False]])
    stub._encode_stage_b_v11_captions = lambda captions, device, apply_feat_map: (
        {"encoded_text": encoded}, mask
    )
    pooled = GroundingDINO._stage_b_arrow_canonical_source(
        stub, ["traffic light .", "person ."], torch.device("cpu")
    )
    assert torch.equal(pooled[0], encoded[0, 1:3].mean(0))
    assert torch.equal(pooled[1], encoded[1, 2])
    with pytest.raises(ValueError, match="non-empty"):
        GroundingDINO._stage_b_arrow_canonical_source(
            stub, ["", "person ."], torch.device("cpu")
        )


def test_generic_admission_adapter_exposes_standardized_score():
    adapter = StageBU0PatchRankAdapter(query_count=4, hidden_dim=4)
    result = adapter(torch.randn(2, 4), torch.randn(2, 4))
    assert result["admission_standardized_score"].shape == (2, 4)
    assert torch.isfinite(result["admission_standardized_score"]).all()


def test_arrow_registry_and_env_alias_fail_closed(monkeypatch):
    assert SOURCES == {
        "AR_A_PATCH": "support_patch",
        "AR_B_TEXT": "canonical_text",
        "AR_C_NULL": "learned_null",
    }
    assert _parse("AR_B_TEXT:17") == ("AR_B_TEXT", 17)
    with pytest.raises(ValueError, match="invalid ARROW seed"):
        _parse("AR_C_NULL:99")
    monkeypatch.setenv("ARROW_TEST_VALUE", "new")
    monkeypatch.setenv("PIVOT_TEST_VALUE", "old")
    with pytest.raises(RuntimeError, match="conflicts"):
        _env("ARROW_TEST_VALUE", "PIVOT_TEST_VALUE", "default")


def test_confidence_overlay_changes_exactly_twelve_tensors(tmp_path):
    confidence_keys = [f"confidence.{index}" for index in range(12)]
    state = {key: torch.zeros(1) for key in confidence_keys}
    state["frozen"] = torch.ones(1)
    admission = {
        "model": state,
        "arrow_admission_input": {
            "schema": "arrow.stageb.admission_input_ablation/v1",
            "frozen_keys": ["frozen", *confidence_keys],
        },
        "optimizer": {}, "lr_scheduler": {}, "scaler": {},
    }
    confidence = {
        "model": {**state, **{key: torch.ones(1) for key in confidence_keys}},
        "u2v5_clean_confidence": {
            "schema": "pivot.stageb.u2v5_clean_confidence_handoff/v1",
            "identity_confidence_keys": confidence_keys,
            "scope": "proposal_covered_verified", "table_b_id": "D3",
            "c100_confidence_imported": False,
        },
    }
    admission_path, confidence_path = tmp_path / "a.pth", tmp_path / "c.pth"
    torch.save(admission, admission_path)
    torch.save(confidence, confidence_path)
    result = merge(admission_path, confidence_path)
    assert result["arrow_confidence_overlay"]["changed_keys"] == sorted(confidence_keys)
    assert result["model"]["frozen"].item() == 1
    assert all(result["model"][key].item() == 1 for key in confidence_keys)
    assert "optimizer" not in result and "scaler" not in result


def test_panel_pair_success_requires_both_directional_arms():
    rows = [
        {"pair_id": "p", "active_score_wins": True,
         "active_eligible_recall50": True, "counterfactual_eligible_leakage": False,
         "eligible_query_count": 2, "has_both_oracle_query_sets": True},
        {"pair_id": "p", "active_score_wins": False,
         "active_eligible_recall50": True, "counterfactual_eligible_leakage": True,
         "eligible_query_count": 4, "has_both_oracle_query_sets": True},
    ]
    result = panel_summary(rows)
    assert result["pair_switch_success"] == 0
    assert result["mean_eligible_queries"] == 3


def test_panel_bootstrap_uses_same_pairs_for_all_seeds():
    candidate = {
        seed: {"p1": True, "p2": True} for seed in (17, 42, 73)
    }
    reference = {
        seed: {"p1": False, "p2": True} for seed in (17, 42, 73)
    }
    first = _panel_contrast(
        candidate, reference, rng=np.random.default_rng(9), iterations=20
    )
    second = _panel_contrast(
        candidate, reference, rng=np.random.default_rng(9), iterations=20
    )
    assert first == second
    assert first["gain"] == 0.5
