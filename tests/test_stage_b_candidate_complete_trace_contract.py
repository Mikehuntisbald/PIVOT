from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from tools import (
    audit_stageb_confidence_candidate_complete_trace_c1_probe_health as c1_health,
)
from tools import (
    audit_stageb_confidence_full_decoder_verifier_probe_health as v61_health,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    run_stageb_confidence_candidate_complete_trace_c1_probe_u0400 as c1_training,
)
from tools import (
    run_stageb_confidence_candidate_complete_trace_c2_probe_u0400 as c2_training,
)
from tools import (
    run_stageb_confidence_candidate_complete_trace_c1_probe_evaluation as c1_evaluation,
)
from tools import (
    run_stageb_confidence_full_decoder_verifier_probe_evaluation as v61_evaluation,
)
from tools import (
    run_stageb_confidence_full_decoder_verifier_probe_u0400 as v61_training,
)
from tools import (
    run_stageb_confidence_full_decoder_patch_softmin_veto_probe_u0400 as v62_training,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(c1_training.CONFIG),
        output_dir=str(tmp_path / "candidate-complete-c1-strict1607"),
        ckpts=["candidate-complete-c1-u400-checkpoint.pth"],
        tn_jsonl=str(
            combined_eval._PARTIAL_CONFIDENCE_TN_SPECS["strict1607"]["path"]
        ),
        tn_splits=["refcocop_val", "refcocog_umd_val"],
        skip_tn=False,
        skip_ref=True,
        device="cuda:0",
        batch_size=16,
        num_workers=4,
        seed=42,
        amp=True,
        topk=[1],
        threshold_tprs=[0.75, 0.9, 0.95],
        score_thresholds=[0.5],
        max_ref_batches=0,
        max_tn_batches=0,
        no_per_example_records=False,
        screen_calibration_manifest=False,
        direct_prebuilt_tn=False,
        category_gate_max_gaps=None,
        category_gate_include_base_expert=False,
        candidate_count_control=0,
        holdout_level="none",
        exclude_train_jsonl=[],
    )


def test_c1_config_is_distinct_from_sealed_v62():
    cfg = SLConfig.fromfile(str(c1_training.CONFIG))
    assert ref_eval._validate_c1_candidate_complete_trace_config(cfg)
    assert not ref_eval._validate_v62_patch_softmin_veto_config(cfg)
    assert cfg.stage_b_dense_duty_confidence_candidate_trace_contract == (
        "candidate_complete_free_head_coverage_v1"
    )
    assert cfg.stage_b_v21_token_edit_query_scope == "candidate_complete_trace_v4"
    assert cfg.stage_b_v21_token_objective == "edit_bce_group_balanced"


def test_c1_resume_contract_is_v43_and_binds_every_candidate_depth_setting():
    args = c1_training._BASE._formal_current_args()
    contract = build_training_contract(args)
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v43"
    expected = {
        "stage_b_dense_duty_confidence_candidate_trace_contract": (
            "candidate_complete_free_head_coverage_v1"
        ),
        "stage_b_dense_duty_confidence_token_depth_base_scale": 1.0,
        "stage_b_v21_token_edit_query_scope": "candidate_complete_trace_v4",
        "stage_b_dense_duty_candidate_depth_all_weight": 1.0,
        "stage_b_dense_duty_candidate_depth_escape_weight": 1.0,
        "stage_b_dense_duty_candidate_depth_positive_weight": 1.0,
        "stage_b_dense_duty_candidate_depth_tn_margin": 0.5,
        "stage_b_dense_duty_candidate_depth_escape_margin": 0.5,
        "stage_b_dense_duty_candidate_depth_positive_max": 0.05,
        "stage_b_dense_duty_candidate_depth_temperature": 0.1,
    }
    assert all(contract["values"][key] == value for key, value in expected.items())

    changed = dict(args)
    changed["stage_b_dense_duty_confidence_token_depth_base_scale"] = 2.0
    assert build_training_contract(changed)["sha256"] != contract["sha256"]


def test_c1_main_preflight_does_not_evaluate_unrelated_v32_admission():
    values = c1_training._BASE._formal_current_args()
    with Path(values["stage_b_dense_duty_trace_audit_path"]).open(
        "r", encoding="ascii"
    ) as handle:
        archived_receipt = json.load(handle)
    values["stage_b_dense_duty_source_closure"] = {
        **values["stage_b_dense_duty_source_closure"],
        "code": {
            **values["stage_b_dense_duty_source_closure"]["code"],
            "sha256": archived_receipt["code_source_closure"]["sha256"],
        },
    }
    values["resume"] = ""
    values["pretrain_model_path"] = str(c1_training._BASE.RANK_SOURCE)
    args = SimpleNamespace(**values)
    training_main._validate_stage_b_dense_duty_args(args)


def test_sealed_v62_contract_remains_v42():
    contract = build_training_contract(v62_training._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v42"
    assert "stage_b_dense_duty_candidate_depth_all_weight" not in contract["values"]


def test_c1_is_registered_by_combined_evaluator(tmp_path):
    cfg = SLConfig.fromfile(str(c1_training.CONFIG))
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


def test_c2_exact_config_is_registered_by_combined_evaluator(tmp_path):
    cfg = SLConfig.fromfile(str(c2_training.CONFIG))
    assert ref_eval._validate_c2_candidate_complete_trace_config(cfg)
    assert not ref_eval._validate_c1_candidate_complete_trace_config(cfg)
    args = _diagnostic_args(tmp_path)
    args.config = str(c2_training.CONFIG)
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        args, cfg
    )


@pytest.mark.parametrize(
    ("name", "mutated"),
    (
        (
            "stage_b_dense_duty_confidence_candidate_trace_contract",
            "off_v1",
        ),
        ("stage_b_dense_duty_confidence_capacity_contract", "drifted"),
        ("stage_b_dense_duty_confidence_variant", "drifted"),
        ("stage_b_v11_trainable_params_min", 25_464_319),
        ("stage_b_v11_trainable_params_max", 25_464_321),
        ("stage_b_dense_duty_raw_veto_gate_weight", 1.0),
        ("stage_b_dense_duty_raw_veto_carrier_pair_weight", 0.25),
        ("stage_b_v21_token_edit_query_scope", "target_iou_v1"),
        ("stage_b_v21_token_objective", "edit_bce"),
        ("stage_b_dense_duty_confidence_token_depth_base_scale", 2.0),
        ("stage_b_dense_duty_candidate_depth_all_weight", 0.0),
        ("stage_b_dense_duty_candidate_depth_escape_weight", 0.0),
        ("stage_b_dense_duty_candidate_depth_positive_weight", 0.0),
        ("stage_b_dense_duty_candidate_depth_tn_margin", 0.4),
        ("stage_b_dense_duty_candidate_depth_escape_margin", 0.4),
        ("stage_b_dense_duty_candidate_depth_positive_max", 0.1),
        ("stage_b_dense_duty_candidate_depth_temperature", 0.2),
    ),
)
def test_c2_validator_fails_closed_on_contract_mutation(name, mutated):
    exact = SLConfig.fromfile(str(c2_training.CONFIG))
    cfg = SimpleNamespace(**dict(exact._cfg_dict))
    setattr(cfg, name, mutated)
    try:
        accepted = ref_eval._validate_c2_candidate_complete_trace_config(cfg)
    except RuntimeError:
        accepted = False
    assert accepted is False


def _exact_c2_migration_audit():
    return {
        "schema": "pivot.stageb.rank_to_full_decoder_candidate_complete_monotone/v28",
        "source_optimizer_updates": 6551,
        "fresh_confidence_contract": (
            "rank_cloned_full_decoder_candidate_complete_monotone_"
            "token_entailment_v26"
        ),
        "token_logit_contract": (
            "independent_rank_cloned_full_decoder_token_entailment_v2"
        ),
        "candidate_trace_contract": (
            "candidate_complete_monotone_token_entailment_v2"
        ),
        "deployed_u0_contract": (
            "rank_cloned_absolute_token_mismatch_nonzero_allowed_v1"
        ),
        "zero_output_scope": "serialized_dormant_free_head_and_pool_only_v1",
        "verifier_parameter_ownership_contract": (
            "tower_owned_parameters_exact_active_trainable_v1"
        ),
        "frozen_unowned_verifier_scope": (
            "encoder_visual_layers_and_level_embed_frozen_unowned_v1"
        ),
        "active_confidence_parameter_tensor_count": 356,
        "active_confidence_parameter_element_count": 25_464_320,
        "active_verifier_parameter_tensor_count": 356,
        "active_verifier_parameter_element_count": 25_464_320,
        "active_verifier_requires_grad_count": 356,
        "active_verifier_veto_head_parameter_tensor_count": 0,
        "active_verifier_veto_head_parameter_element_count": 0,
        "active_pool_parameter_tensor_count": 0,
        "active_pool_parameter_element_count": 0,
        "frozen_unowned_verifier_parameter_tensor_count": 97,
        "frozen_unowned_verifier_parameter_element_count": 7_694_080,
        "frozen_unowned_verifier_requires_grad_count": 0,
        "hidden_dim": 256,
        "decoder_num_layers": 6,
        "verifier_tensor_count": 453,
        "verifier_element_count": 33_158_400,
        "verifier_matches_rank": True,
        "verifier_veto_output_nonzero_count": 0,
        "pool_output_nonzero_count": 0,
        "retired_confidence_loaded_tensor_count": 0,
        "patch_softmin_veto_only": True,
    }


@pytest.mark.parametrize(
    ("name", "mutated"),
    (
        ("schema", "pivot.stageb.rank_to_full_decoder_patch_softmin_veto/v27"),
        ("active_confidence_parameter_tensor_count", 362),
        ("active_confidence_parameter_element_count", 25_530_881),
        ("active_verifier_requires_grad_count", 355),
        ("active_verifier_veto_head_parameter_tensor_count", 6),
        ("active_pool_parameter_tensor_count", 6),
        ("candidate_trace_contract", "candidate_complete_free_head_coverage_v1"),
        ("verifier_parameter_ownership_contract", "namespace_all_active_v1"),
    ),
)
def test_c2_checkpoint_migration_admission_fails_closed(name, mutated):
    exact = _exact_c2_migration_audit()
    ref_eval._validate_c2_candidate_complete_migration_audit(exact)
    exact[name] = mutated
    with pytest.raises(RuntimeError, match="token-only verifier migration"):
        ref_eval._validate_c2_candidate_complete_migration_audit(exact)


def _exact_c2_runtime_audit():
    return {
        "schema": "pivot.stageb.dense_duty_runtime_audit/v1",
        "clip_contract_schema": (
            "pivot.stageb.dense_duty_one_owner_clip_contract/v1"
        ),
        "successful_optimizer_steps": 400,
        "optimizer_step_boundaries": 400,
        "clip_contract_checked_steps": 400,
        "nonfinite_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "max_active_grad_norm_preclip": 16.0,
        "last_token_veto_grad_norm_preclip": 10.0,
        "max_token_veto_grad_norm_preclip": 16.0,
        "expected_token_veto_tensor_count": 356,
        "last_observed_token_veto_tensor_count": 356,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
        "clip_contract_tolerance": 1e-4,
        "clip_contract_max_norm": 0.1,
        "max_active_pre_decomposition_residual": 0.0,
        "max_active_post_decomposition_residual": 0.0,
        "max_owner_clip_residual": 1e-9,
        "max_active_monotonic_residual": 0.0,
    }


@pytest.mark.parametrize(
    ("name", "mutated"),
    (
        ("clip_contract_schema", "pivot.stageb.dense_duty_two_owner_clip_contract/v1"),
        ("clip_contract_checked_steps", 399),
        ("expected_token_veto_tensor_count", 355),
        ("last_observed_token_veto_tensor_count", 355),
        ("live_tensor_count_violation_steps", 1),
    ),
)
def test_c2_checkpoint_runtime_admission_fails_closed(name, mutated):
    exact = _exact_c2_runtime_audit()
    ref_eval._validate_c2_one_owner_runtime_audit(exact, optimizer_updates=400)
    exact[name] = mutated
    with pytest.raises(RuntimeError):
        ref_eval._validate_c2_one_owner_runtime_audit(
            exact, optimizer_updates=400
        )


def test_c2_checkpoint_runtime_rejects_a_second_owner():
    runtime = _exact_c2_runtime_audit()
    runtime["expected_global_absolute_tensor_count"] = 6
    with pytest.raises(RuntimeError, match="second confidence owner"):
        ref_eval._validate_c2_one_owner_runtime_audit(
            runtime, optimizer_updates=400
        )


def test_c1_health_uses_exact_unscaled_training_log_fields(monkeypatch):
    required = set(c1_health._C1_REQUIRED_TRAJECTORY_FIELDS)
    assert all(field.endswith("_unscaled") for field in required)

    monkeypatch.setattr(
        c1_health.v62,
        "_health_checks_v62",
        lambda _runtime, _trajectory: {
            "u222_v62_owner_live_counts_exact": {
                "passed": True,
                "observed": None,
                "requirement": "base",
            },
            "u222_v62_two_independent_clips_exact": {
                "passed": True,
                "observed": None,
                "requirement": "base",
            },
        },
    )
    trajectory = {
        "train_fixed_text_token_trace_deployed_candidate_count_unscaled": 50.0,
        "train_fixed_text_candidate_depth_tn_sample_count_unscaled": 1.0,
        "train_fixed_text_candidate_depth_tn_query_count_unscaled": 50.0,
        "train_fixed_text_token_trace_broadcast_candidate_count_unscaled": 0.0,
        "train_fixed_text_token_trace_candidate_coverage_unscaled": 0.0,
        "train_fixed_text_candidate_depth_tn_zero_rate_unscaled": 0.005,
        "train_fixed_text_candidate_depth_tn_min_mean_unscaled": 0.2,
        "train_fixed_text_candidate_depth_positive_sample_count_unscaled": 1.0,
        "train_fixed_text_candidate_depth_positive_query_count_unscaled": 2.0,
        "train_fixed_text_candidate_depth_positive_zero_rate_unscaled": 0.0,
        "train_fixed_text_candidate_depth_positive_min_mean_unscaled": 0.08,
    }
    checks = c1_health._health_checks_c1({}, trajectory)
    assert all(check["passed"] for check in checks.values())

    trajectory[
        "train_fixed_text_candidate_depth_tn_query_count_unscaled"
    ] = 49.0
    checks = c1_health._health_checks_c1({}, trajectory)
    assert not checks["u222_c1_candidate_depth_deployed_set_exact"]["passed"]


def test_c1_health_replays_its_v43_contract_without_v45_scope_fallback():
    args = c1_training._BASE._formal_current_args()
    args["stage_b_dense_duty_training_contract"] = build_training_contract(args)
    result = c1_health._audit_c1_training_contract(args)
    assert result["schema"] == "pivot.stageb.dense_duty_training_contract/v43"
    assert result["candidate_trace_contract"] == (
        "candidate_complete_free_head_coverage_v1"
    )
    assert result["token_edit_query_scope"] == "candidate_complete_trace_v4"


def test_c1_wrappers_restore_v61_health_and_evaluation_state():
    assert v61_health.TRAINING_CONTRACT_SCHEMA.endswith("/v42")
    assert v61_health.EXPECTED_CAPACITY_CONTRACT == (
        "rank_cloned_full_decoder_6layer_256d_v1"
    )
    assert v61_health._CORE.EXPECTED_CONTRACT_VALUES[
        "stage_b_v11_trainable_params_min"
    ] == 25_664_258

    c1_command = c1_evaluation.build_command()
    assert str(c1_training.CONFIG) in c1_command
    assert str(c1_training.CHECKPOINT) in c1_command

    v61_command = v61_evaluation.build_command()
    assert str(v61_training.CONFIG) in v61_command
    assert str(v61_training.CHECKPOINT) in v61_command
    assert v61_evaluation.EXPECTED_CAPACITY_CONTRACT == (
        "rank_cloned_full_decoder_6layer_256d_v1"
    )
