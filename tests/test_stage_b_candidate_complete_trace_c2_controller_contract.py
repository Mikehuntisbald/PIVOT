from __future__ import annotations

import pytest

from tools import (
    audit_stageb_confidence_candidate_complete_trace_c2_probe_health as health,
)
from tools import (
    run_stageb_confidence_candidate_complete_trace_c2_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_candidate_complete_trace_c2_probe_u0400 as training,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


def test_c2_health_seals_token_only_v28_capacity_and_one_owner_clip():
    assert health.TRAINING_CONTRACT_SCHEMA.endswith("/v43")
    assert health.MIGRATION_SCHEMA.endswith("candidate_complete_monotone/v28")
    assert health.FRESH_CONFIDENCE_CONTRACT.endswith("token_entailment_v26")
    assert health.EXPECTED_ACTIVE_TENSORS == health.EXPECTED_TOKEN_TENSORS == 356
    assert health.EXPECTED_ACTIVE_ELEMENTS == health.EXPECTED_TOKEN_ELEMENTS == 25_464_320
    assert health.EXPECTED_GLOBAL_TENSORS == health.EXPECTED_GLOBAL_ELEMENTS == 0
    assert health.ONE_OWNER_CLIP_CONTRACT_SCHEMA.endswith("one_owner_clip_contract/v1")
    assert health._C2_EXPECTED_CONTRACT_VALUES[
        "stage_b_dense_duty_raw_veto_gate_weight"
    ] == 0.0
    assert health._C2_EXPECTED_CONTRACT_VALUES[
        "stage_b_dense_duty_raw_veto_carrier_pair_weight"
    ] == 0.0
    assert all("raw_veto" not in field for field in health._C2_REQUIRED_TRAJECTORY_FIELDS)
    assert all("global_absolute" not in field for field in health._C2_JOINT_CLIP_FIELDS)


def test_c2_config_and_replayed_contract_bind_monotone_scope():
    cfg = SLConfig.fromfile(str(training.CONFIG))
    assert cfg.stage_b_dense_duty_confidence_candidate_trace_contract == "candidate_complete_monotone_token_entailment_v2"
    assert cfg.stage_b_dense_duty_confidence_capacity_contract == health.EXPECTED_CAPACITY_CONTRACT
    assert cfg.stage_b_dense_duty_raw_veto_gate_weight == 0.0
    assert cfg.stage_b_dense_duty_raw_veto_carrier_pair_weight == 0.0
    args = training._BASE._formal_current_args()
    contract = build_training_contract(args)
    assert contract["schema"] == health.TRAINING_CONTRACT_SCHEMA
    assert contract["values"]["stage_b_v21_token_edit_query_scope"] == "candidate_complete_trace_v4"
    assert "stage_b_v21_target_iou" not in contract["values"]
    args["stage_b_dense_duty_training_contract"] = contract
    assert health._audit_c2_training_contract(args)["schema"] == (
        health.TRAINING_CONTRACT_SCHEMA
    )
    args["stage_b_dense_duty_raw_veto_gate_weight"] = 1.0
    with pytest.raises(health.ProbeHealthEvidenceError, match="C2 training contract"):
        health._audit_c2_training_contract(args)


def test_c2_health_candidate_depth_checks_require_coverage_and_positive_protection(monkeypatch):
    trajectory = {
        "train_grad_norm_dense_duty_active_preclip": 0.2,
        "train_grad_tensor_count_dense_duty_active": 356.0,
        "train_grad_tensor_count_dense_duty_token_veto": 356.0,
        "train_grad_norm_dense_duty_token_veto_preclip": 0.2,
        "train_grad_norm_dense_duty_token_veto_postclip": 0.1,
        "train_grad_norm_dense_duty_active_postclip": 0.1,
        "train_amp_step_skipped": 0.0,
        "train_fixed_text_token_trace_deployed_candidate_count_unscaled": 50.0,
        "train_fixed_text_candidate_depth_tn_sample_count_unscaled": 1.0,
        "train_fixed_text_candidate_depth_tn_query_count_unscaled": 50.0,
        "train_fixed_text_token_trace_broadcast_candidate_count_unscaled": 0.0,
        "train_fixed_text_token_trace_candidate_coverage_unscaled": 0.0,
        "train_fixed_text_candidate_depth_tn_zero_rate_unscaled": 0.005,
        "train_fixed_text_candidate_depth_tn_min_mean_unscaled": 0.2,
        "train_fixed_text_candidate_depth_positive_sample_count_unscaled": 1.0,
        "train_fixed_text_candidate_depth_positive_query_count_unscaled": 2.0,
        "train_fixed_text_candidate_depth_positive_min_mean_unscaled": 0.08,
    }
    runtime = {
        "clip_contract_schema": health.ONE_OWNER_CLIP_CONTRACT_SCHEMA,
        "optimizer_step_boundaries": 400,
        "successful_optimizer_steps": 400,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "nonfinite_token_veto_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "zero_token_veto_gradient_successful_steps": 0,
        "last_token_veto_grad_norm_preclip": 0.2,
        "max_token_veto_grad_norm_preclip": 0.3,
        "min_amp_scale": 256.0,
        "last_amp_scale": 256.0,
    }
    checks = health._health_checks_c2(runtime, trajectory)
    assert all(check["passed"] for check in checks.values())
    trajectory["train_fixed_text_token_trace_candidate_coverage_unscaled"] = 1.0
    assert not health._health_checks_c2(runtime, trajectory)["u222_c2_no_token_broadcast_or_coverage"]["passed"]


def test_c2_scope_temporarily_installs_one_owner_log_schema():
    baseline = health._CORE._JOINT_CLIP_FIELDS
    with health._c2_scope():
        assert health._CORE._JOINT_CLIP_FIELDS == health._C2_JOINT_CLIP_FIELDS
    assert health._CORE._JOINT_CLIP_FIELDS == baseline


def test_c2_split_ownership_returns_and_seals_frozen_rank_identity(monkeypatch):
    active_name = (
        "stage_b_fixed_text_scorer.confidence_verifier_tower."
        "decoder.layers.0.weight"
    )
    rank_name = f"{health._CORE._RANK_PREFIX}decoder.layers.0.weight"
    model = {
        active_name: health.torch.tensor([1.0]),
        rank_name: health.torch.tensor([2.0]),
    }
    current_active = health.fingerprint_named_tensors(model, [active_name])
    rank = health.fingerprint_named_tensors(model, [rank_name])
    initial_active = {**current_active, "sha256": "0" * 64}

    monkeypatch.setattr(health, "EXPECTED_ACTIVE_TENSORS", 1)
    monkeypatch.setattr(health, "EXPECTED_ACTIVE_ELEMENTS", 1)
    monkeypatch.setattr(health, "EXPECTED_TOKEN_ELEMENTS", 1)
    monkeypatch.setattr(health._CORE, "EXPECTED_RANK_SHA256", rank["sha256"])
    monkeypatch.setattr(health, "_audit_c2_migration", lambda _args: {"schema": "test"})
    monkeypatch.setattr(
        health,
        "validate_initial_fingerprint",
        lambda *_args, **_kwargs: {
            "active_parameter_names": [active_name],
            "active": initial_active,
        },
    )
    payload = {
        "model": model,
        "optimizer": {"param_groups": [{"params": [0]}], "state": {0: {}}},
    }
    args = {"stage_b_dense_duty_rank_source_rank_sha256": rank["sha256"]}

    ownership = health._audit_c2_split_ownership(payload, args)
    assert ownership["rank"] == rank

    args["stage_b_dense_duty_rank_source_rank_sha256"] = "f" * 64
    with pytest.raises(
        health.ProbeHealthEvidenceError,
        match="C2 frozen rank tower identity drifted",
    ):
        health._audit_c2_split_ownership(payload, args)


def test_c2_strict1607_output_is_fresh_and_c2_scoped(monkeypatch):
    command = evaluation.build_command()
    assert str(training.CONFIG) in command
    assert str(training.CHECKPOINT) in command
    assert "c2_monotone_token_entailment" in str(evaluation.OUTPUT)
    assert evaluation.OUTPUT.name == "u000400_strict1607"
    monkeypatch.setattr(
        evaluation.v62,
        "_v62_postflight",
        lambda *_args, **_kwargs: {"contracts": {}},
    )
    contracts = evaluation._c2_postflight({})["contracts"]
    assert contracts["token_entailment_is_the_only_active_confidence_owner"] is True
    assert contracts["global_absolute_owner_is_absent"] is True
