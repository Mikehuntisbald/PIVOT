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
    validate_confidence_runtime_payload,
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

    from config.ablations import cfg_stageb_u2v5_clean_admission_eval_gap3 as eval_cfg
    from config.ablations import cfg_stageb_u2v5_clean_confidence_d3_u100 as conf_cfg

    for cfg in (eval_cfg, conf_cfg):
        assert cfg.stage_b_u2v2_c100_checkpoint is None
        assert cfg.stage_b_u2v2_c100_sha256 is None
    assert eval_cfg.stage_b_u2v4_checkpoint_eval is True
    assert eval_cfg.stage_b_u2v4_legacy_training_replay is True
    assert conf_cfg.stage_b_u2v4_checkpoint_eval is False


def test_d3_screen_calibration_eval_is_narrowly_allowlisted():
    from types import SimpleNamespace

    from tools.eval_stageb_tn_val import _validate_adapter_tn_eval_manifest

    audit = "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
    cfg = SimpleNamespace(
        stage_b_gdino_score_adapter=True,
        stage_b_data_driven_score=False,
        stage_b_gdino_tn_scope="proposal_covered_verified",
        stage_b_u2v5_clean_confidence=True,
        stage_b_v19_table_b_audit_sha256=audit,
    )
    row = {
        "tn_scope": "proposal_covered_verified",
        "table_b_id": "D3",
        "tn_eval_split": "screen_calibration",
        "tn_eval_source_split": "sealed_image_disjoint_calibration",
        "proposal_covered_verified": True,
        "global_tn_verified": False,
        "benchmark_dataft_alltn": False,
        "proposalset_proxy_verified": False,
        "cached_proposal_coverage_only": True,
        "all_900_gdino_queries_verified": False,
        "global_max_label_is_semantic_extrapolation": True,
        "table_b_audit_sha256": audit,
    }
    with pytest.raises(ValueError, match="proposal-covered"):
        _validate_adapter_tn_eval_manifest(cfg, [row])
    assert (
        _validate_adapter_tn_eval_manifest(
            cfg, [row], allow_proposal_covered_calibration=True
        )
        == "proposal_covered_verified"
    )
    changed = dict(row, table_b_audit_sha256="0" * 64)
    with pytest.raises(ValueError, match="audit-bound"):
        _validate_adapter_tn_eval_manifest(
            cfg, [changed], allow_proposal_covered_calibration=True
        )


def test_clean_confidence_runtime_rejects_missing_contract():
    model = torch.nn.Linear(1, 1)
    with pytest.raises(U2V5CleanInitializerError, match="provenance"):
        validate_confidence_runtime_payload(
            model,
            {"model": model.state_dict()},
            checkpoint_label="test checkpoint",
        )


def test_preregistration_uses_robust_cross_seed_milestone_rule():
    from tools.build_stageb_u2v5_preregistration import select_confidence

    values = {
        25: (0.56, 0.58, 0.57),
        50: (0.54, 0.52, 0.542),
        100: (0.53, 0.527, 0.545),
    }
    rows = []
    for update, scores in values.items():
        for seed, score in zip((17, 42, 73), scores):
            rows.append(
                {
                    "run_id": f"confidence_seed{seed}_u{update}_checkpoint_iter",
                    "fpr95tpr": score,
                }
            )
    result = select_confidence({"tn": rows})
    assert result["selected_update"] == 50
    assert result["candidates"][0]["worst_seed_fpr95"] == pytest.approx(0.542)
