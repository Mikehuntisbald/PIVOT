import copy
from pathlib import Path

import pytest
import torch

from engine import _build_stage_b_gdino_adapter_pair_captions
from models.GroundingDINO.stage_b_gdino_score_adapter import (
    stage_b_gdino_confidence_objective_code,
    stage_b_gdino_tn_scope_code,
)
from tools.build_stageb_u2v5_clean_initializer import (
    U2V5CleanInitializerError,
    validate_initializer_payload,
)


def _target():
    return {
        "tn_scope": "proposal_covered_verified",
        "proposalset_proxy_verified": torch.tensor([False]),
        "global_tn_verified": torch.tensor([False]),
        "benchmark_dataft_alltn": torch.tensor([False]),
        "table_b_id": "D3",
        "table_b_audit_sha256": "a" * 64,
        "verifier_num_patch_slots": torch.tensor([1]),
        "verifier_pair_stride": torch.tensor([2]),
        "cap_list": ["the red cup", "the blue cup"],
        "is_tn": torch.tensor([False, True]),
    }


def test_d3_scope_and_objective_have_distinct_codes():
    assert stage_b_gdino_tn_scope_code("proposal_covered_verified") == 3
    assert stage_b_gdino_confidence_objective_code(
        "detached_recent_q05_proposal_covered"
    ) == 4


def test_d3_pair_contract_accepts_weak_scope_without_global_upgrade():
    positive, negative, codes = _build_stage_b_gdino_adapter_pair_captions(
        [_target()], "proposal_covered_verified"
    )
    assert positive == ["the red cup ."]
    assert negative == ["the blue cup ."]
    assert torch.equal(codes, torch.tensor([3]))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda target: target.update(global_tn_verified=torch.tensor([True])),
        lambda target: target.update(table_b_id="D2"),
        lambda target: target.update(table_b_audit_sha256=""),
    ],
)
def test_d3_pair_contract_rejects_scope_upgrade_or_unbound_audit(mutation):
    target = _target()
    mutation(target)
    with pytest.raises(RuntimeError, match="audited D3"):
        _build_stage_b_gdino_adapter_pair_captions(
            [target], "proposal_covered_verified"
        )


def test_clean_initializer_validator_rejects_c100_source():
    state = {"only": torch.zeros(1)}
    payload = {
        "model": state,
        "u2v5_clean_initializer": {
            "schema": "pivot.stageb.u2v5_clean_initializer/v1",
            "model_state_keys": 1,
            "sources": {"c100": {"path": str(Path("missing"))}},
            "invariants": {"c100_confidence_imported": False},
            "role_keys": {"identity_confidence": ["only"]},
            "role_tensor_sha256": {"identity_confidence": "invalid"},
        },
    }
    with pytest.raises(U2V5CleanInitializerError):
        validate_initializer_payload(copy.deepcopy(payload), verify_sources=False)


def test_clean_configs_lock_seeds_to_external_runner_and_never_name_c100_path():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "cfg_stageb_u2v5_clean_admission_u100.py",
        "cfg_stageb_u2v5_clean_confidence_d3_u100.py",
    ):
        text = (root / "config/ablations" / name).read_text(encoding="utf-8")
        assert "checkpoint_iter_000100.pth" not in text
        assert "c9737d6" not in text
    confidence = (
        root
        / "config/ablations/cfg_stageb_u2v5_clean_confidence_d3_u100.py"
    ).read_text(encoding="utf-8")
    assert 'stage_b_gdino_tn_scope = "proposal_covered_verified"' in confidence
    assert "detached_recent_q05_proposal_covered" in confidence
