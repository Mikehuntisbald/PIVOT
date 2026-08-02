from tools import (
    audit_stageb_confidence_adapter_veto_gated_pool_carrier_balanced_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_carrier_balanced_highmem_formal as formal,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_carrier_balanced_probe as probe,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_carrier_balanced_probe_evaluation as promotion,
)
from tools import (
    run_stageb_confidence_adapter_veto_gated_pool_carrier_balanced_probe_u0050 as probe_u0050,
)


def _admission_fixture():
    return {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "diagnostic_only": True,
        "formal_training_admitted": True,
    }


def test_carrier_balanced_formal_contract_binds_v7_and_u6551(monkeypatch):
    monkeypatch.setattr(
        formal._BASE,
        "FORMAL_ADMISSION_VALIDATOR",
        _admission_fixture,
    )
    values = formal._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_gated_pool_carrier_balanced_v7"
    )
    assert values["stage_b_dense_duty_raw_veto_query_scope"] == (
        "tn_all_admitted_carrier_balanced_positive_carrier_v3"
    )
    assert values["stage_b_dense_duty_raw_veto_tn_carrier_balance"] == 0.5
    assert values["stage_b_dense_duty_confidence_carrier_selector_contract"] == (
        "final_layer_reference_argmax_exact_eligible_v1"
    )
    assert values["stage_b_dense_duty_confidence_veto_gate_offset"] == 0.02
    assert values["stage_b_dense_duty_confidence_veto_gate_scale"] == 0.03
    assert values["stage_b_dense_duty_rank_source_optimizer_updates"] == 6551
    assert formal._BASE.build_training_contract(values)["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v10"
    )


def test_carrier_balanced_probe_and_promotion_are_fresh_exact_contracts():
    values = probe._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 300
    assert values["stage_b_dense_duty_execution_scope"] == "probe"
    argv = probe.command("start")
    assert argv[argv.index("--max_train_iters") + 1] == "300"
    assert "--resume" not in argv
    u50_argv = probe_u0050.command("start")
    assert u50_argv[u50_argv.index("--max_train_iters") + 1] == "50"
    assert "--resume" not in u50_argv
    assert promotion.HEALTH_SCHEMA == health.SCHEMA
    assert promotion.FORMAL_PROMOTION_OVERRIDES[
        "stage_b_dense_duty_confidence_probe_admission_contract"
    ] == (
        "disabled_for_probe_v1",
        "u300_word_veto_gated_pool_carrier_balanced_strict1607_v7",
    )


def _healthy_trajectory():
    return {
        "train_stage_b_dense_confidence_positive_veto_coverage_mean_unscaled": 0.1,
        "train_stage_b_dense_confidence_tn_veto_coverage_mean_unscaled": 0.9,
        "train_stage_b_dense_confidence_positive_veto_sample_gate_mean_unscaled": 0.05,
        "train_stage_b_dense_confidence_tn_veto_sample_gate_mean_unscaled": 0.8,
        "train_stage_b_dense_confidence_veto_ceiling_unscaled": -0.1,
        "train_fixed_text_raw_veto_positive_sample_count_unscaled": 16.0,
        "train_fixed_text_raw_veto_tn_sample_count_unscaled": 15.5,
        "train_fixed_text_raw_veto_positive_all_hinge_mean_unscaled": 0.03,
        "train_fixed_text_raw_veto_positive_loss_mean_unscaled": 0.03,
        "train_fixed_text_raw_veto_tn_all_hinge_mean_unscaled": 0.04,
        "train_fixed_text_raw_veto_tn_carrier_hinge_mean_unscaled": 0.02,
        "train_fixed_text_raw_veto_tn_balanced_loss_mean_unscaled": 0.03,
        "train_fixed_text_raw_veto_positive_carrier_sample_count_unscaled": 16.0,
        "train_fixed_text_raw_veto_positive_carrier_source_mean_unscaled": -0.08,
        "train_fixed_text_raw_veto_positive_carrier_violation_rate_unscaled": 0.2,
        "train_fixed_text_raw_veto_tn_carrier_sample_count_unscaled": 15.5,
        "train_fixed_text_raw_veto_tn_carrier_source_mean_unscaled": 0.12,
        "train_fixed_text_raw_veto_tn_carrier_violation_rate_unscaled": 0.2,
        "train_fixed_text_raw_veto_tn_carrier_changed_gate_mean_unscaled": 0.85,
        "train_fixed_text_raw_veto_tn_carrier_full_open_rate_unscaled": 0.8,
    }


def test_carrier_balanced_health_preregisters_routing_and_attribution(monkeypatch):
    monkeypatch.setattr(
        health,
        "_BASE_HEALTH_CHECKS",
        lambda trajectory, endpoint, runtime: {},
    )
    checks = health._health_checks(_healthy_trajectory(), {}, {})
    assert all(check["passed"] for check in checks.values())

    wrong_mix = _healthy_trajectory()
    wrong_mix["train_fixed_text_raw_veto_tn_balanced_loss_mean_unscaled"] = 0.04
    checks = health._health_checks(wrong_mix, {}, {})
    assert not checks["u222_tn_balance_exact"]["passed"]

    shared_word_only = _healthy_trajectory()
    shared_word_only[
        "train_fixed_text_raw_veto_tn_carrier_changed_gate_mean_unscaled"
    ] = 0.2
    checks = health._health_checks(shared_word_only, {}, {})
    assert not checks["u222_tn_changed_carrier_gate_open"]["passed"]
