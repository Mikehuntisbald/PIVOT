from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from tools import (
    audit_stageb_confidence_adapter_candidate_split_fpr_active_set_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_boundary_routing_probe_u0400 as v47_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_fpr_active_set_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_fpr_active_set_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return (
        SLConfig.fromfile(str(v47_probe.CONFIG))._cfg_dict.to_dict(),
        SLConfig.fromfile(str(probe.CONFIG))._cfg_dict.to_dict(),
    )


def test_v48_is_v47_plus_only_fpr_active_set_and_provenance():
    v47, v48 = _configs()
    changed = {key for key in set(v47) | set(v48) if v47.get(key) != v48.get(key)}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_v15_tail_queue_negative_reduction_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }
    assert v48["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_fpr_active_set_v48"
    )
    assert v48["stage_b_v15_tail_queue_negative_reduction_contract"] == (
        "exact_fpr95_active_set_mean_v1"
    )
    assert v47.get(
        "stage_b_v15_tail_queue_negative_reduction_contract",
        "all_mean_v1",
    ) == "all_mean_v1"
    assert v48["stage_b_dense_duty_trace_audit_sha256"] != "0" * 64


def test_v48_training_contract_is_exact_v30():
    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v30"
    assert contract["values"][
        "stage_b_v15_tail_queue_negative_reduction_contract"
    ] == "exact_fpr95_active_set_mean_v1"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_v15_tail_queue_negative_reduction_contract",
            "all_mean_v1",
        ),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 1.0),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "balanced_mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_positive_trust_reduction_contract",
            "mean_v1",
        ),
    ),
)
def test_v48_validation_fails_closed_on_contract_drift(field, value):
    cfg = SLConfig.fromfile(str(probe.CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError, match="v48 FPR active set"):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v47_rejects_the_v48_active_set_field():
    cfg = SLConfig.fromfile(str(v47_probe.CONFIG))
    cfg.stage_b_v15_tail_queue_negative_reduction_contract = (
        "exact_fpr95_active_set_mean_v1"
    )
    with pytest.raises(RuntimeError, match="v47 boundary routing"):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v48_probe_is_fresh_and_reuses_only_the_frozen_rank_source():
    assert probe.UPDATES == 400
    assert probe.OUTPUT != v47_probe.OUTPUT
    assert probe.inspect() == {"status": "fresh", "action": "start"}
    command = probe.command("start")
    assert "--resume" not in command
    assert command[command.index("--pretrain_model_path") + 1] == str(
        probe._BASE.RANK_SOURCE
    )
    current = probe._BASE._formal_current_args()
    current["resume"] = None
    current["pretrain_model_path"] = str(probe._BASE.RANK_SOURCE)
    training_main._validate_stage_b_dense_duty_args(SimpleNamespace(**current))
    probe.validate_inputs()


@pytest.mark.parametrize(("active", "fraction"), ((19.0, 19.0 / 32.0), (32.0, 1.0)))
def test_v48_health_requires_exact_active_set_telemetry(active, fraction):
    runtime = {
        "optimizer_step_boundaries": 400,
        "successful_optimizer_steps": 400,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "nonfinite_token_veto_gradient_boundaries": 0,
        "nonfinite_global_absolute_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "zero_token_veto_gradient_successful_steps": 0,
        "zero_global_absolute_gradient_successful_steps": 0,
        "last_token_veto_grad_norm_preclip": 20.0,
        "max_token_veto_grad_norm_preclip": 30.0,
        "last_global_absolute_grad_norm_preclip": 22.0,
        "max_global_absolute_grad_norm_preclip": 31.0,
        "min_amp_scale": 256.0,
        "last_amp_scale": 256.0,
    }
    trajectory = {
        "train_grad_norm_dense_duty_active_preclip": 30.0,
        "train_grad_tensor_count_dense_duty_active": 67.0,
        "train_grad_norm_dense_duty_token_veto_preclip": 20.0,
        "train_grad_tensor_count_dense_duty_token_veto": 21.0,
        "train_grad_norm_dense_duty_global_absolute_preclip": 22.0,
        "train_grad_tensor_count_dense_duty_global_absolute": 46.0,
        "train_grad_norm_dense_duty_token_veto_postclip": 0.1,
        "train_grad_norm_dense_duty_global_absolute_postclip": 0.1,
        "train_grad_norm_dense_duty_active_postclip": math.sqrt(0.02),
        "train_amp_step_skipped": 0.0,
        "train_fixed_text_tail_queue_negative_total_count_unscaled": 32.0,
        "train_fixed_text_tail_queue_negative_active_count_unscaled": active,
        "train_fixed_text_tail_queue_negative_selected_count_unscaled": active,
        "train_fixed_text_tail_queue_negative_active_fraction_unscaled": fraction,
        "train_fixed_text_tail_queue_positive_threshold_unscaled": -0.2,
        "train_fixed_text_tail_queue_negative_active_min_logit_unscaled": -0.2,
        "train_fixed_text_tail_queue_negative_inactive_max_logit_unscaled": -0.21,
        "train_fixed_text_tail_queue_negative_loss_unscaled": 0.4,
    }
    checks = health._health_checks(runtime, trajectory)
    assert all(check["passed"] for check in checks.values())


def test_v48_strict_controller_is_fixed_and_unbound():
    command = evaluation.build_command()
    assert command[command.index("--config") + 1] == str(evaluation.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(evaluation.CHECKPOINT)
    assert command[command.index("--batch_size") + 1] == "16"
    assert command[command.index("--num_workers") + 1] == "4"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--topk") + 1] == "1"
    assert "--amp" in command and "--skip_ref" in command
    assert evaluation._formal_main_admission_is_wired(REPO_ROOT / "main.py") is False
    formal = SLConfig.fromfile(str(evaluation.FORMAL_CONFIG))
    assert training_main._bind_stage_b_confidence_probe_admission(formal) is None
