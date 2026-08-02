from tools import (
    run_stageb_confidence_adapter_veto_gate_highmem_formal as formal,
)
from tools import run_stageb_confidence_adapter_veto_gate_probe as probe
from tools import (
    run_stageb_confidence_adapter_veto_gate_probe_evaluation as promotion,
)


def _admission_fixture():
    return {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "diagnostic_only": True,
        "formal_training_admitted": True,
    }


def test_veto_gate_formal_contract_binds_raw_gate_and_u6551(monkeypatch):
    monkeypatch.setattr(
        formal._BASE,
        "FORMAL_ADMISSION_VALIDATOR",
        _admission_fixture,
    )
    values = formal._formal_current_args()
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_raw_gate_margin_v3"
    )
    assert values["stage_b_dense_duty_confidence_phrase_aggregation"] == (
        "trace_activated_word_veto_penalty_v2"
    )
    assert values["stage_b_dense_duty_raw_veto_gate_weight"] == 1.0
    assert values["stage_b_dense_duty_raw_veto_positive_margin"] == 0.1
    assert values["stage_b_dense_duty_raw_veto_tn_margin"] == 0.1
    assert values["stage_b_dense_duty_rank_source_optimizer_updates"] == 6551
    assert values["stage_b_dense_duty_rank_source_checkpoint_sha256"] == (
        "50e60a1314f7f2908bee5eea84ede5549b908177b367609efdec1682caa67ed3"
    )
    assert values["stage_b_dense_duty_confidence_probe_admission_audit"] == (
        _admission_fixture()
    )
    assert formal._BASE.build_training_contract(values)["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v8"
    )


def test_veto_gate_probe_is_an_exact_u300_scope_promotion():
    values = probe._BASE._formal_current_args()
    assert values["stage_b_dense_duty_confidence_expected_optimizer_updates"] == 300
    assert values["stage_b_dense_duty_execution_scope"] == "probe"
    assert values["stage_b_dense_duty_evaluation_scope"] == "probe"
    assert values["stage_b_dense_duty_confidence_probe_admission_contract"] == (
        "disabled_for_probe_v1"
    )
    argv = probe.command("start")
    assert argv[argv.index("--max_train_iters") + 1] == "300"
    assert argv[argv.index("--gradient_accumulation_steps") + 1] == "2"
    assert argv[argv.index("--pretrain_model_path") + 1].endswith(
        "dense_duty_20260728/formal/rank/checkpoint_iter.pth"
    )
    assert "--resume" not in argv


def test_veto_gate_promotion_uses_v3_health_and_exact_config_delta():
    assert promotion.HEALTH_SCHEMA.endswith("v3")
    assert promotion.FORMAL_PROMOTION_OVERRIDES == {
        "epochs": (2, 24),
        "stage_b_dense_duty_confidence_expected_optimizer_updates": (300, 4412),
        "stage_b_dense_duty_evaluation_scope": ("probe", "formal"),
        "stage_b_dense_duty_execution_scope": ("probe", "formal"),
        "stage_b_dense_duty_confidence_probe_admission_contract": (
            "disabled_for_probe_v1",
            "u300_word_veto_gate_strict1607_v3",
        ),
        "stage_b_dense_duty_confidence_probe_admission_report": (
            "",
            str(promotion.REPORT),
        ),
    }
