from pathlib import Path

import pytest

from tools import run_stageb_confidence_adapter_veto_highmem_formal as formal
from tools import run_stageb_confidence_adapter_veto_probe as probe


def _admission_fixture():
    return {
        "status": "verified",
        "decision": "admit_to_formal_training",
        "diagnostic_only": True,
        "formal_training_admitted": True,
    }


def test_veto_formal_contract_is_isolated_and_removes_duplicate_tn_tail(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        formal._BASE,
        "FORMAL_ADMISSION_VALIDATOR",
        _admission_fixture,
    )
    values = formal._formal_current_args()
    assert values["stage_b_dense_duty_confidence_phrase_aggregation"] == (
        "trace_activated_word_veto_product_v1"
    )
    assert values["stage_b_dense_duty_positive_trust_contract"] == (
        "net_total_confidence_delta_v1"
    )
    assert values["stage_b_dense_duty_confidence_tn_scope"] == (
        "direct_trace_valid_v1"
    )
    assert values["stage_b_v11_global_tn_negative_weight"] == 0.25
    assert values["stage_b_v11_global_tn_tail_weight"] == 0.0
    assert values["stage_b_v11_trainable_params_min"] == 185_924
    assert values["stage_b_v11_trainable_params_max"] == 185_924
    assert values[formal.SOURCE_CLOSURE_ARG]["config"]["entry"].endswith(
        "cfg_stageb_dense_duty_confidence_adapter_veto_20260730.py"
    )
    assert "adapter_veto_trace_audit" in values[
        "stage_b_dense_duty_trace_audit_path"
    ]
    assert values["stage_b_dense_duty_trace_audit_sha256"] == (
        "60047aa1a8b89e6789ef4e4817e4ce9c23e8fdb457e25bdde65daada07250da6"
    )
    assert "adapter_veto_highmem" in str(formal.OUTPUT)
    assert values[
        "stage_b_dense_duty_confidence_probe_admission_audit"
    ] == _admission_fixture()
    assert formal._BASE.build_training_contract(values)["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v7"
    )


def test_veto_formal_recipe_fails_closed_without_probe_admission(
    monkeypatch: pytest.MonkeyPatch,
):
    def reject():
        raise RuntimeError("U300 admission is unavailable")

    monkeypatch.setattr(
        formal._BASE,
        "FORMAL_ADMISSION_VALIDATOR",
        reject,
    )
    with pytest.raises(RuntimeError, match="U300 admission"):
        formal._formal_current_args()


def test_veto_probe_preserves_highmem_geometry_and_is_bounded():
    values = probe._BASE._formal_current_args()
    assert values[
        "stage_b_dense_duty_confidence_expected_optimizer_updates"
    ] == 300
    argv = probe.command("start")
    assert argv[argv.index("--max_train_iters") + 1] == "300"
    assert argv[argv.index("--gradient_accumulation_steps") + 1] == "2"
    assert argv[argv.index("--config_file") + 1] == str(probe.CONFIG)
    assert argv[argv.index("--output_dir") + 1] == str(probe.OUTPUT)
    assert "probe/u000300" in str(probe.OUTPUT)
    assert "--pretrain_model_path" in argv
    assert "--resume" not in argv


def test_veto_probe_terminal_inspection_preserves_probe_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "probe"
    output.mkdir()
    checkpoint = output / "checkpoint_iter.pth"
    checkpoint.touch()
    base = probe._BASE
    saved_args = {
        "stage_b_v22_score_ownership": (
            "rank_tower_stopgrad_token_adapter_two_phase"
        ),
        base.TRAINING_CONTRACT_ARG: {"schema": "sentinel"},
    }
    payload = {
        "args": saved_args,
        "optimizer_updates": probe.UPDATES,
        "model": {
            "stage_b_fixed_text_scorer.rank_tower.weight": object(),
        },
    }
    current_args = {
        "stage_b_dense_duty_evaluation_scope": "probe",
        formal.SOURCE_CLOSURE_ARG: {"code": {"sha256": "0" * 64}},
    }
    observed = {}

    def validate_evaluation(value, cfg, **kwargs):
        observed["scope"] = cfg.stage_b_dense_duty_evaluation_scope
        return {"evaluation_scope": observed["scope"]}

    monkeypatch.setattr(base, "OUTPUT", output)
    monkeypatch.setattr(base, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(base, "_load", lambda _path: payload)
    monkeypatch.setattr(base, "_validate_migration", lambda _args: {})
    monkeypatch.setattr(base, "_formal_current_args", lambda: current_args)
    monkeypatch.setattr(base, "_validate_terminal_training_state", lambda *_: None)
    monkeypatch.setattr(
        base, "build_training_contract", lambda _args: {"schema": "sentinel"}
    )
    monkeypatch.setattr(base, "validate_resume_training_contract", lambda *_: None)
    monkeypatch.setattr(
        base, "validate_evaluation_checkpoint_payload", validate_evaluation
    )
    monkeypatch.setattr(
        base,
        "fingerprint_named_tensors",
        lambda *_args, **_kwargs: {"sha256": base.RANK_SHA256},
    )

    state = probe.inspect()
    assert state["status"] == "terminal"
    assert state["checkpoint_audit"]["evaluation_scope"] == "probe"
    assert observed["scope"] == "probe"
