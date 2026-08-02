from pathlib import Path

import pytest

import main as training_main
from tools import (
    run_stageb_confidence_adapter_candidate_deployed_routing_probe_u0400 as probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_gate_zero_offset_probe_u0400 as v39_probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "deployed_routing_probe_u0400_20260801.py"
)


def test_v43_is_a_bounded_deployed_routing_delta_over_v39():
    v39 = SLConfig.fromfile(str(v39_probe.CONFIG))._cfg_dict.to_dict()
    v43 = SLConfig.fromfile(str(PROBE_CONFIG))._cfg_dict.to_dict()
    allowed = {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_confidence_gate_gradient_contract",
        "stage_b_dense_duty_deployed_veto_routing_weight",
        "stage_b_dense_duty_deployed_veto_positive_max",
        "stage_b_dense_duty_deployed_veto_tn_min",
        "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }
    assert {
        key for key in set(v39) | set(v43) if v39.get(key) != v43.get(key)
    } == allowed
    assert v43["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_asymmetric_deployed_routing_v43"
    )
    assert v43["stage_b_dense_duty_confidence_gate_gradient_contract"] == (
        "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
    )
    assert v43["stage_b_dense_duty_deployed_veto_routing_weight"] == 0.1
    assert v43["stage_b_dense_duty_deployed_veto_positive_max"] == 0.1
    assert v43["stage_b_dense_duty_deployed_veto_tn_min"] == 0.9
    assert v43.get(
        "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
        "bidirectional_v1",
    ) == "bidirectional_v1"
    assert v43["stage_b_v11_trainable_params_min"] == v39[
        "stage_b_v11_trainable_params_min"
    ]
    assert v43["stage_b_v11_trainable_params_max"] == v39[
        "stage_b_v11_trainable_params_max"
    ]


def test_v43_routing_fields_are_bound_into_v25_training_contract():
    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v25"
    values = contract["values"]
    assert values["stage_b_dense_duty_confidence_gate_gradient_contract"] == (
        "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
    )
    assert values["stage_b_dense_duty_deployed_veto_routing_weight"] == 0.1
    assert values["stage_b_dense_duty_deployed_veto_positive_max"] == 0.1
    assert values["stage_b_dense_duty_deployed_veto_tn_min"] == 0.9


def test_v43_validation_fails_closed_on_routing_margin_drift():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    cfg.stage_b_dense_duty_deployed_veto_tn_min = 0.8
    with pytest.raises(RuntimeError, match="v43 deployed routing"):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v43_probe_is_isolated_and_disables_formal_admission():
    assert probe.UPDATES == 400
    assert probe.CONFIG.resolve(strict=True) == PROBE_CONFIG.resolve(strict=True)
    assert "candidate_deployed_routing" in str(probe.OUTPUT)
    assert probe.OUTPUT != v39_probe.OUTPUT
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert training_main._bind_stage_b_confidence_probe_admission(cfg) is None
