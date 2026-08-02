from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from tools import (
    audit_stageb_confidence_adapter_candidate_split_positive_tail_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_heads_probe_u0400 as v44_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_positive_tail_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_positive_tail_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return (
        SLConfig.fromfile(str(v44_probe.CONFIG))._cfg_dict.to_dict(),
        SLConfig.fromfile(str(probe.CONFIG))._cfg_dict.to_dict(),
    )


def test_v46_is_v44_plus_only_positive_tail_trust_and_provenance():
    v44, v46 = _configs()
    changed = {key for key in set(v44) | set(v46) if v44.get(key) != v46.get(key)}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
        "stage_b_v15_tail_queue_positive_trust_reduction_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }
    assert v46["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_absolute_v2"
    )
    assert v46["stage_b_dense_duty_deployed_veto_routing_weight"] == 0.1
    assert v46[
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract"
    ] == "balanced_mean_v1"
    assert v46[
        "stage_b_v15_tail_queue_positive_trust_reduction_contract"
    ] == "top_quarter_cvar_v2"
    assert v46["stage_b_dense_duty_trace_audit_sha256"] != "0" * 64


def test_v46_training_contract_is_exact_v28():
    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v28"
    values = contract["values"]
    assert values["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_absolute_v2"
    )
    assert values[
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract"
    ] == "balanced_mean_v1"
    assert values[
        "stage_b_v15_tail_queue_positive_trust_reduction_contract"
    ] == "top_quarter_cvar_v2"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_joint_clip_v3",
            "exact v45 tail-aligned surface",
        ),
        (
            "stage_b_dense_duty_deployed_veto_routing_weight",
            1.0,
            "v46 positive-tail alignment",
        ),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "balanced_top_quarter_cvar_v2",
            "v46 positive-tail alignment",
        ),
        (
            "stage_b_v15_tail_queue_positive_trust_reduction_contract",
            "mean_v1",
            "v46 positive-tail alignment",
        ),
    ),
)
def test_v46_validation_fails_closed_on_contract_drift(field, value, message):
    cfg = SLConfig.fromfile(str(probe.CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError, match=message):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v46_probe_is_fresh_and_does_not_resume_v45_or_v44():
    assert probe.UPDATES == 400
    assert probe.OUTPUT != v44_probe.OUTPUT
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


def test_v46_health_requires_independent_owner_clips():
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
    }
    checks = health._health_checks(runtime, trajectory)
    assert all(check["passed"] for check in checks.values())
    trajectory["train_grad_norm_dense_duty_active_postclip"] = 0.1
    checks = health._health_checks(runtime, trajectory)
    assert checks["u222_independent_clip_v2_evidence"]["passed"] is False


def test_v46_strict_controller_is_fixed_and_unbound():
    command = evaluation.build_command()
    assert command[command.index("--config") + 1] == str(evaluation.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(evaluation.CHECKPOINT)
    assert command[command.index("--batch_size") + 1] == "16"
    assert command[command.index("--num_workers") + 1] == "4"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--topk") + 1] == "1"
    assert "--amp" in command and "--skip_ref" in command
    assert evaluation._formal_main_admission_is_wired(REPO_ROOT / "main.py") is False
