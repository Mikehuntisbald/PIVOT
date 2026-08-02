from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tools import (
    audit_stageb_confidence_adapter_candidate_split_tail_aligned_probe_health as health,
)


def _criterion(**overrides):
    value = {
        "tail_positive_queue": torch.linspace(-1.0, 1.0, health.QUEUE_SIZE),
        "tail_negative_queue": torch.linspace(-2.0, 0.0, health.QUEUE_SIZE),
        "tail_positive_ptr": torch.tensor(3, dtype=torch.int64),
        "tail_negative_ptr": torch.tensor(4, dtype=torch.int64),
        "tail_positive_count": torch.tensor(
            health.QUEUE_SIZE, dtype=torch.int64
        ),
        "tail_negative_count": torch.tensor(
            health.QUEUE_SIZE, dtype=torch.int64
        ),
    }
    value.update(overrides)
    return value


def _runtime(**overrides):
    value = {
        "schema": health.RUNTIME_SCHEMA,
        "optimizer_step_boundaries": 400,
        "successful_optimizer_steps": 400,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "last_active_grad_norm_preclip": 20.0,
        "max_active_grad_norm_preclip": 40.0,
        "last_token_veto_grad_norm_preclip": 12.0,
        "max_token_veto_grad_norm_preclip": 25.0,
        "last_global_absolute_grad_norm_preclip": 16.0,
        "max_global_absolute_grad_norm_preclip": 30.0,
        "last_amp_scale": 256.0,
        "min_amp_scale": 256.0,
    }
    value.update(overrides)
    return value


def _trajectory(**overrides):
    value = {
        "train_optimizer_updates": 222,
        "train_grad_norm_dense_duty_active_preclip": 30.0,
        "train_grad_tensor_count_dense_duty_active": 67.0,
        "train_grad_norm_dense_duty_token_veto_preclip": 20.0,
        "train_grad_tensor_count_dense_duty_token_veto": 21.0,
        "train_grad_norm_dense_duty_global_absolute_preclip": 22.0,
        "train_grad_tensor_count_dense_duty_global_absolute": 46.0,
        "train_grad_norm_dense_duty_token_veto_postclip": 0.07,
        "train_grad_norm_dense_duty_global_absolute_postclip": 0.065,
        "train_grad_norm_dense_duty_active_postclip": 0.1,
        "train_amp_step_skipped": 0.0,
    }
    value.update(overrides)
    return value


def test_full_finite_tail_queues_are_admitted():
    endpoint = health._audit_tail_queues(_criterion())
    assert endpoint["positive_count"] == health.QUEUE_SIZE
    assert endpoint["tn_count"] == health.QUEUE_SIZE
    assert endpoint["all_finite"] is True
    assert math.isfinite(endpoint["operating_gap"])


@pytest.mark.parametrize(
    "overrides,error",
    [
        (
            {"tail_positive_queue": torch.zeros(4096, dtype=torch.float64)},
            "full finite float32",
        ),
        (
            {
                "tail_negative_queue": torch.full(
                    (4096,), float("nan"), dtype=torch.float32
                )
            },
            "full finite float32",
        ),
        (
            {"tail_positive_count": torch.tensor(4095, dtype=torch.int64)},
            "must be full",
        ),
        (
            {"tail_negative_ptr": torch.tensor(4096, dtype=torch.int64)},
            "outside the ring",
        ),
    ],
)
def test_tail_queue_drift_fails_closed(overrides, error):
    with pytest.raises(health.ProbeHealthEvidenceError, match=error):
        health._audit_tail_queues(_criterion(**overrides))


def test_owner_classifier_matches_the_two_explicit_apis():
    assert health._owner_for_name(
        "stage_b_fixed_text_scorer.confidence_adapter.query_projection.weight"
    ) == "token_veto"
    assert health._owner_for_name(
        "stage_b_fixed_text_scorer.confidence_adapter.cross_attention.in_proj_weight"
    ) == "global_absolute"
    assert health._owner_for_name(
        "stage_b_fixed_text_scorer.confidence_pool.residual.0.weight"
    ) == "global_absolute"
    with pytest.raises(health.ProbeHealthEvidenceError, match="unknown"):
        health._owner_for_name(
            "stage_b_fixed_text_scorer.confidence_adapter.shared_trunk.weight"
        )


def test_exact_v27_contract_replays(monkeypatch: pytest.MonkeyPatch):
    closure = {
        "schema": "pivot.stageb.dense_duty_source_closure/v1",
        "sha256": "a" * 64,
        "config": {"entry": health.EXPECTED_CONFIG_ENTRY},
    }
    values = {**health.EXPECTED_CONTRACT_VALUES}
    values["stage_b_dense_duty_source_closure"] = closure
    contract = {
        "schema": health.TRAINING_CONTRACT_SCHEMA,
        "sha256": "b" * 64,
        "values": values,
    }
    args = {
        **health.EXPECTED_CONTRACT_VALUES,
        "stage_b_dense_duty_source_closure": closure,
        "stage_b_dense_duty_evaluation_scope": "probe",
        "stage_b_dense_duty_confidence_probe_admission_contract": (
            "disabled_for_probe_v1"
        ),
        "stage_b_dense_duty_confidence_probe_admission_report": "",
        health.TRAINING_CONTRACT_ARG: contract,
    }
    monkeypatch.setattr(health, "build_training_contract", lambda _: contract)
    observed = health._audit_training_contract(args)
    assert observed["schema"] == health.TRAINING_CONTRACT_SCHEMA
    assert observed["head_gradient_contract"] == health.EXPECTED_HEAD_CONTRACT
    assert observed["routing_reduction_contract"] == (
        "balanced_top_quarter_cvar_v2"
    )


def test_v27_contract_or_probe_admission_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    closure = {"sha256": "a" * 64, "config": {"entry": health.EXPECTED_CONFIG_ENTRY}}
    values = {**health.EXPECTED_CONTRACT_VALUES}
    values["stage_b_dense_duty_source_closure"] = closure
    contract = {
        "schema": health.TRAINING_CONTRACT_SCHEMA,
        "sha256": "b" * 64,
        "values": values,
    }
    args = {
        **health.EXPECTED_CONTRACT_VALUES,
        "stage_b_dense_duty_evaluation_scope": "probe",
        "stage_b_dense_duty_confidence_probe_admission_contract": "formal",
        "stage_b_dense_duty_confidence_probe_admission_report": "report.json",
        health.TRAINING_CONTRACT_ARG: contract,
    }
    monkeypatch.setattr(health, "build_training_contract", lambda _: contract)
    with pytest.raises(health.ProbeHealthEvidenceError, match="formal admission"):
        health._audit_training_contract(args)


def test_joint_clip_and_runtime_health_pass():
    runtime = health._audit_runtime(_runtime())
    checks = health._health_checks(runtime, _trajectory())
    assert all(check["passed"] for check in checks.values())
    assert runtime["nonfinite_token_veto_gradient_boundaries"] == 0
    assert runtime["zero_global_absolute_gradient_successful_steps"] == 0


def test_amp_skip_and_independent_head_clip_are_rejected_as_unhealthy():
    runtime = health._audit_runtime(_runtime(amp_skipped_optimizer_steps=1))
    trajectory = _trajectory(
        train_grad_norm_dense_duty_token_veto_postclip=0.1,
        train_grad_norm_dense_duty_global_absolute_postclip=0.1,
    )
    checks = health._health_checks(runtime, trajectory)
    assert checks["runtime_amp_skips_zero"]["passed"] is False
    assert checks["u222_joint_clip_v3_evidence"]["passed"] is False


def test_terminal_cursor_requires_exact_u400_geometry():
    payload = {
        "epoch": 1,
        "iteration": 356,
        "optimizer_updates": 400,
        "epoch_finished": False,
        "checkpoint_reason": "max_train_iters",
    }
    assert health._audit_terminal_cursor(payload) == payload
    payload["iteration"] = 354
    with pytest.raises(health.ProbeHealthEvidenceError, match="cursor drifted"):
        health._audit_terminal_cursor(payload)


def test_default_report_is_outside_atomic_training_directory():
    assert health._default_output().parent == Path(health.training.OUTPUT).parent
    assert health._default_output().parent != Path(health.training.OUTPUT)
