from tools import (
    audit_stageb_confidence_adapter_veto_gated_pool_calibrated_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_calibrated_highmem_formal as formal,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_calibrated_probe as probe,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_calibrated_probe_evaluation as promotion,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_calibrated_probe_u0050 as probe_u0050,
)


def _admission_fixture():
    return {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "diagnostic_only": True,
        "formal_training_admitted": True,
    }


def test_calibrated_formal_contract_binds_v6_and_u6551(monkeypatch):
    monkeypatch.setattr(
        formal._BASE,
        "FORMAL_ADMISSION_VALIDATOR",
        _admission_fixture,
    )
    values = formal._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_gated_pool_calibrated_v6"
    )
    assert values["stage_b_dense_duty_confidence_phrase_aggregation"] == (
        "trace_activated_word_veto_gated_pool_absolute_cap_v5"
    )
    assert values["stage_b_dense_duty_confidence_veto_gate_offset"] == 0.02
    assert values["stage_b_dense_duty_confidence_veto_gate_scale"] == 0.03
    assert values["stage_b_dense_duty_raw_veto_positive_margin"] == 0.1
    assert values["stage_b_dense_duty_raw_veto_tn_margin"] == 0.15
    assert values["stage_b_dense_duty_rank_source_optimizer_updates"] == 6551
    assert values["stage_b_v11_trainable_params_min"] == 185_925
    assert values["stage_b_v11_trainable_params_max"] == 185_925
    assert formal._BASE.build_training_contract(values)["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v9"
    )


def test_calibrated_probe_promotion_and_health_are_exact_contracts():
    values = probe._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 300
    assert values["stage_b_dense_duty_execution_scope"] == "probe"
    argv = probe.command("start")
    assert argv[argv.index("--max_train_iters") + 1] == "300"
    assert "--resume" not in argv
    u50_argv = probe_u0050.command("start")
    assert u50_argv[u50_argv.index("--max_train_iters") + 1] == "50"
    assert promotion.HEALTH_SCHEMA == health.SCHEMA
    assert promotion.FORMAL_PROMOTION_OVERRIDES[
        "stage_b_dense_duty_confidence_probe_admission_contract"
    ] == (
        "disabled_for_probe_v1",
        "u300_word_veto_gated_pool_calibrated_strict1607_v6",
    )


def test_calibrated_health_keeps_original_preregistered_gate_thresholds(monkeypatch):
    monkeypatch.setattr(
        health,
        "_BASE_HEALTH_CHECKS",
        lambda trajectory, endpoint, runtime: {},
    )
    trajectory = {
        "train_stage_b_dense_confidence_positive_veto_coverage_mean_unscaled": 0.9,
        "train_stage_b_dense_confidence_tn_veto_coverage_mean_unscaled": 0.1,
        "train_stage_b_dense_confidence_positive_veto_sample_gate_mean_unscaled": 0.05,
        "train_stage_b_dense_confidence_tn_veto_sample_gate_mean_unscaled": 0.8,
        "train_stage_b_dense_confidence_veto_ceiling_unscaled": -0.1,
    }
    checks = health._health_checks(trajectory, {}, {})
    assert checks["u222_positive_carrier_gate_closed"]["passed"]
    assert checks["u222_tn_carrier_gate_open"]["passed"]
    assert checks["u222_absolute_ceiling_nonpositive"]["passed"]
