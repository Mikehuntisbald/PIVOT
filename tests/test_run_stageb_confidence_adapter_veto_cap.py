from tools import (
    run_stageb_confidence_adapter_veto_cap_highmem_formal as formal,
)
from tools import run_stageb_confidence_adapter_veto_cap_probe as probe
from tools import (
    run_stageb_confidence_adapter_veto_cap_probe_evaluation as promotion,
)


def _admission_fixture():
    return {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "diagnostic_only": True,
        "formal_training_admitted": True,
    }


def test_veto_cap_formal_contract_binds_v4_and_u6551(monkeypatch):
    monkeypatch.setattr(
        formal._BASE,
        "FORMAL_ADMISSION_VALIDATOR",
        _admission_fixture,
    )
    values = formal._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_coverage_absolute_cap_v4"
    )
    assert values["stage_b_dense_duty_confidence_phrase_aggregation"] == (
        "trace_activated_word_veto_absolute_cap_v4"
    )
    assert values["stage_b_dense_duty_raw_veto_query_scope"] == (
        "tn_all_admitted_positive_carrier_v2"
    )
    assert values["stage_b_dense_duty_confidence_veto_gate_offset"] == 0.05
    assert values["stage_b_dense_duty_confidence_veto_gate_scale"] == 0.1
    assert values["stage_b_dense_duty_confidence_veto_coverage_offset"] == 0.1
    assert values["stage_b_dense_duty_confidence_veto_coverage_ramp"] == 0.8
    assert values["stage_b_dense_duty_confidence_veto_cap_temperature"] == 0.1
    assert values["stage_b_dense_duty_confidence_veto_cap_initial_ceiling"] == -0.1
    assert values["stage_b_v11_trainable_params_min"] == 185_925
    assert values["stage_b_v11_trainable_params_max"] == 185_925
    assert values["stage_b_dense_duty_rank_source_optimizer_updates"] == 6551
    assert values["stage_b_dense_duty_confidence_probe_admission_audit"] == (
        _admission_fixture()
    )
    assert formal._BASE.build_training_contract(values)["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v9"
    )


def test_veto_cap_probe_and_promotion_are_exact_u300_contracts():
    values = probe._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 300
    assert values["stage_b_dense_duty_execution_scope"] == "probe"
    assert values["stage_b_dense_duty_evaluation_scope"] == "probe"
    argv = probe.command("start")
    assert argv[argv.index("--max_train_iters") + 1] == "300"
    assert "--resume" not in argv

    assert promotion.HEALTH_SCHEMA.endswith("v4")
    assert promotion.FORMAL_PROMOTION_OVERRIDES[
        "stage_b_dense_duty_confidence_probe_admission_contract"
    ] == (
        "disabled_for_probe_v1",
        "u300_word_veto_absolute_cap_strict1607_v4",
    )
