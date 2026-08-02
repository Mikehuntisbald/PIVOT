from pathlib import Path

import pytest

import main as training_main
from tools import (
    run_stageb_confidence_adapter_candidate_gate_zero_offset_probe_u0400 as v39_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_tn_only_carrier_pair_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "tn_only_carrier_pair_probe_u0400_20260801.py"
)


def test_v42_is_an_exact_single_delta_over_v39():
    v39 = SLConfig.fromfile(str(v39_probe.CONFIG))._cfg_dict.to_dict()
    v42 = SLConfig.fromfile(str(PROBE_CONFIG))._cfg_dict.to_dict()
    allowed = {
        "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }
    assert {
        key for key in set(v39) | set(v42) if v39.get(key) != v42.get(key)
    } == allowed
    assert v42["stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract"] == (
        "tn_only_positive_detached_v2"
    )
    assert v42.get("stage_b_v21_token_edit_query_scope", "target_iou_v1") == (
        "target_iou_v1"
    )
    assert v42["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_asymmetric_confidence_v32"
    )
    assert v42["stage_b_v11_trainable_params_min"] == v39[
        "stage_b_v11_trainable_params_min"
    ]
    assert v42["stage_b_v11_trainable_params_max"] == v39[
        "stage_b_v11_trainable_params_max"
    ]


def test_v42_gets_a_gradient_route_bound_v24_training_contract():
    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v24"
    assert contract["values"][
        "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract"
    ] == "tn_only_positive_detached_v2"


def test_v39_contract_remains_unchanged_and_omits_v42_key():
    contract = build_training_contract(v39_probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v21"
    assert (
        "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract"
        not in contract["values"]
    )


def test_v42_scope_validation_fails_closed_on_token_carrier_drift():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    cfg.stage_b_v21_token_edit_query_scope = (
        "target_iou_union_detached_final_confidence_base_argmax_v2"
    )
    with pytest.raises(RuntimeError, match="exact v39"):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v42_probe_paths_and_update_budget_are_isolated():
    assert probe.UPDATES == 400
    assert probe.CONFIG.resolve(strict=True) == PROBE_CONFIG.resolve(strict=True)
    assert "tn_only_carrier_pair" in str(probe.OUTPUT)
    assert probe.OUTPUT != v39_probe.OUTPUT
