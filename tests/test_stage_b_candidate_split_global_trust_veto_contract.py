from __future__ import annotations

import math
from pathlib import Path

import pytest

import main as training_main
from tools import (
    audit_stageb_confidence_adapter_candidate_split_global_trust_veto_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_boundary_routing_probe_u0400 as v47_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_global_trust_veto_highmem_formal as formal,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_global_trust_veto_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_global_trust_veto_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


def _configs():
    return (
        SLConfig.fromfile(str(v47_probe.CONFIG))._cfg_dict.to_dict(),
        SLConfig.fromfile(str(probe.CONFIG))._cfg_dict.to_dict(),
    )


def test_v49_is_v47_plus_only_split_global_heads_and_provenance():
    v47, v49 = _configs()
    changed = {key for key in set(v47) | set(v49) if v47.get(key) != v49.get(key)}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_confidence_head_gradient_contract",
        "stage_b_v11_trainable_params_min",
        "stage_b_v11_trainable_params_max",
        "stage_b_v15_tail_queue_negative_reduction_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }
    assert v49["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_global_trust_veto_v49"
    )
    assert v49["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_trust_veto_v4"
    )
    assert v49["stage_b_v11_trainable_params_min"] == 669_322
    assert v49["stage_b_v11_trainable_params_max"] == 669_322
    assert v49.get(
        "stage_b_v15_tail_queue_negative_reduction_contract",
        "all_mean_v1",
    ) == "all_mean_v1"
    assert Path(v49["stage_b_dense_duty_trace_audit_path"]).is_file()
    assert len(v49["stage_b_dense_duty_trace_audit_sha256"]) == 64


def test_v49_training_contract_is_exact_v31():
    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v31"
    values = contract["values"]
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_global_trust_veto_v49"
    )
    assert values["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_trust_veto_v4"
    )
    assert values["stage_b_v11_trainable_params_min"] == 669_322
    assert values["stage_b_v11_trainable_params_max"] == 669_322


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_v2",
        ),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "balanced_mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_positive_trust_reduction_contract",
            "mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_negative_reduction_contract",
            "exact_fpr95_active_set_mean_v1",
        ),
    ),
)
def test_v49_validation_fails_closed_on_contract_drift(field, value):
    cfg = SLConfig.fromfile(str(probe.CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v49_probe_is_fresh_and_reuses_only_the_frozen_rank_source():
    assert probe.UPDATES == 400
    assert probe.OUTPUT != v47_probe.OUTPUT
    command = probe.command("start")
    assert "--resume" not in command
    assert command[command.index("--pretrain_model_path") + 1] == str(
        probe._BASE.RANK_SOURCE
    )
    assert command[command.index("--max_train_iters") + 1] == "400"


def test_v49_health_closes_token_trust_and_veto_owners():
    runtime = {
        "optimizer_step_boundaries": 400,
        "successful_optimizer_steps": 400,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "nonfinite_token_veto_gradient_boundaries": 0,
        "nonfinite_global_absolute_gradient_boundaries": 0,
        "nonfinite_global_trust_gradient_boundaries": 0,
        "nonfinite_global_veto_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "zero_token_veto_gradient_successful_steps": 0,
        "zero_global_absolute_gradient_successful_steps": 0,
        "zero_global_trust_gradient_successful_steps": 0,
        "zero_global_veto_gradient_successful_steps": 0,
        "last_token_veto_grad_norm_preclip": 20.0,
        "max_token_veto_grad_norm_preclip": 30.0,
        "last_global_absolute_grad_norm_preclip": 22.0,
        "max_global_absolute_grad_norm_preclip": 31.0,
        "last_global_trust_grad_norm_preclip": 18.0,
        "max_global_trust_grad_norm_preclip": 27.0,
        "last_global_veto_grad_norm_preclip": 13.0,
        "max_global_veto_grad_norm_preclip": 19.0,
        "min_amp_scale": 256.0,
        "last_amp_scale": 256.0,
    }
    trajectory = {
        "train_grad_norm_dense_duty_active_preclip": 30.0,
        "train_grad_tensor_count_dense_duty_active": 73.0,
        "train_grad_norm_dense_duty_token_veto_preclip": 20.0,
        "train_grad_tensor_count_dense_duty_token_veto": 21.0,
        "train_grad_norm_dense_duty_global_absolute_preclip": 22.0,
        "train_grad_tensor_count_dense_duty_global_absolute": 52.0,
        "train_grad_norm_dense_duty_global_trust_preclip": 18.0,
        "train_grad_tensor_count_dense_duty_global_trust": 46.0,
        "train_grad_norm_dense_duty_global_veto_preclip": 13.0,
        "train_grad_tensor_count_dense_duty_global_veto": 6.0,
        "train_grad_norm_dense_duty_token_veto_postclip": 0.1,
        "train_grad_norm_dense_duty_global_absolute_postclip": 0.1,
        "train_grad_norm_dense_duty_global_trust_postclip": 0.08,
        "train_grad_norm_dense_duty_global_veto_postclip": 0.06,
        "train_grad_norm_dense_duty_active_postclip": math.sqrt(0.02),
        "train_amp_step_skipped": 0.0,
    }
    checks = health._health_checks(runtime, trajectory)
    assert all(check["passed"] for check in checks.values())

    # Epoch summaries average per-step norms. A changing trust/veto ownership
    # ratio obeys triangle bounds but does not obey Pythagoras after averaging.
    trajectory["train_grad_norm_dense_duty_global_trust_postclip"] = (
        0.08084657197600012
    )
    trajectory["train_grad_norm_dense_duty_global_veto_postclip"] = (
        0.034661713718968395
    )
    checks = health._health_checks(runtime, trajectory)
    assert checks["u222_global_union_clip_preserves_both_subowners"]["passed"]

    trajectory["train_grad_norm_dense_duty_global_veto_postclip"] = 0.01
    checks = health._health_checks(runtime, trajectory)
    assert not checks["u222_global_union_clip_preserves_both_subowners"]["passed"]


def test_v49_health_negative_reduction_reads_saved_contract_values(monkeypatch):
    key = "stage_b_v15_tail_queue_negative_reduction_contract"
    monkeypatch.setattr(
        health,
        "_BASE_AUDIT_TRAINING_CONTRACT",
        lambda _args: {"schema": health.TRAINING_CONTRACT_SCHEMA},
    )
    args = {
        key: "all_mean_v1",
        health.TRAINING_CONTRACT_ARG: {
            "values": {key: "all_mean_v1"},
        },
    }

    observed = health._audit_training_contract(args)

    assert observed["negative_reduction_contract"] == "all_mean_v1"


@pytest.mark.parametrize("drift_location", ("args", "saved_contract"))
def test_v49_health_negative_reduction_fails_closed_on_drift(
    monkeypatch, drift_location
):
    key = "stage_b_v15_tail_queue_negative_reduction_contract"
    monkeypatch.setattr(
        health,
        "_BASE_AUDIT_TRAINING_CONTRACT",
        lambda _args: {"schema": health.TRAINING_CONTRACT_SCHEMA},
    )
    args = {
        key: "all_mean_v1",
        health.TRAINING_CONTRACT_ARG: {
            "values": {key: "all_mean_v1"},
        },
    }
    if drift_location == "args":
        args[key] = "exact_fpr95_active_set_mean_v1"
    else:
        args[health.TRAINING_CONTRACT_ARG]["values"][key] = (
            "exact_fpr95_active_set_mean_v1"
        )

    with pytest.raises(health.ProbeHealthEvidenceError):
        health._audit_training_contract(args)


def _write_admission_binder(path: Path, *, complete: bool) -> None:
    negative = (
        '    negative = "all_mean_v1"\n'
        if complete
        else '    negative = "exact_fpr95_active_set_mean_v1"\n'
    )
    path.write_text(
        "def _bind_stage_b_confidence_probe_admission(args):\n"
        "    from tools import "
        "run_stageb_confidence_adapter_candidate_split_global_trust_veto_"
        "probe_evaluation as promotion\n"
        '    revision = "word_veto_candidate_split_global_trust_veto_v49"\n'
        '    head = "split_token_veto_global_trust_veto_v4"\n'
        '    routing = "balanced_top_quarter_cvar_v2"\n'
        '    trust = "top_quarter_cvar_v2"\n'
        + negative
        + '    admission = "u400_word_veto_candidate_split_global_trust_veto_'
        'confidence_strict1607_v49"\n'
        "    return promotion, revision, head, routing, trust, negative, admission\n",
        encoding="utf-8",
    )


def test_v49_strict_controller_is_fixed_and_ast_admission_is_fail_closed(tmp_path):
    command = evaluation.build_command()
    assert command[command.index("--config") + 1] == str(evaluation.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(evaluation.CHECKPOINT)
    assert command[command.index("--batch_size") + 1] == "16"
    assert command[command.index("--num_workers") + 1] == "4"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--topk") + 1] == "1"
    assert "--amp" in command and "--skip_ref" in command

    incomplete = tmp_path / "incomplete_main.py"
    complete = tmp_path / "complete_main.py"
    _write_admission_binder(incomplete, complete=False)
    _write_admission_binder(complete, complete=True)
    assert evaluation._formal_main_admission_is_wired(incomplete) is False
    assert evaluation._formal_main_admission_is_wired(complete) is True


def test_v49_formal_runner_starts_fresh_from_rank_source_not_probe_checkpoint():
    assert formal.UPDATES == 4_412
    assert formal.OUTPUT != probe.OUTPUT
    command = formal.command("start")
    assert "--resume" not in command
    assert command[command.index("--pretrain_model_path") + 1] == str(
        formal._BASE.RANK_SOURCE
    )
    assert command[command.index("--max_train_iters") + 1] == "4412"
    assert str(probe.CHECKPOINT) not in command
