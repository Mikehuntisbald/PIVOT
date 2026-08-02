from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import (
    FINGERPRINT_ARG,
    SOURCE_CLOSURE_ARG,
    _RESUME_CONTRACT_KEYS,
    _validate_fulltext_global_absolute_runtime_audit,
    audit_checkpoint_payload,
    build_source_closure,
    build_training_contract,
    fingerprint_state,
    validate_evaluation_checkpoint_payload,
    validate_formal_invocation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V52_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_sample_calibrator_"
    "20260802.py"
)
V53_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "20260802.py"
)
V53_PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "probe_u0400_20260802.py"
)
V54_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "exact_residual_20260802.py"
)
V54_PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "exact_residual_probe_u0400_20260802.py"
)
V55_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_independent_"
    "absolute_20260802.py"
)
V55_PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_independent_"
    "absolute_probe_u0400_20260802.py"
)


def _training_args(config_path: Path) -> dict:
    values = SLConfig.fromfile(str(config_path))._cfg_dict.to_dict()
    for key in _RESUME_CONTRACT_KEYS:
        values.setdefault(key, 1)
    values[SOURCE_CLOSURE_ARG] = build_source_closure(
        config_path, repo_root=REPO_ROOT
    )
    values.setdefault(
        "stage_b_dense_duty_confidence_probe_admission_audit",
        {"status": "test"},
    )
    return values


def _healthy_v53_runtime(steps: int = 400) -> dict:
    runtime = {
        "schema": "pivot.stageb.dense_duty_runtime_audit/v1",
        "optimizer_step_boundaries": steps,
        "successful_optimizer_steps": steps,
        "nonfinite_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "max_active_grad_norm_preclip": 3.0,
        "clip_contract_schema": (
            "pivot.stageb.dense_duty_two_owner_clip_contract/v1"
        ),
        "clip_contract_checked_steps": steps,
        "clip_contract_max_norm": 0.1,
        "clip_contract_tolerance": 1.0e-5,
        "max_active_pre_decomposition_residual": 0.0,
        "max_active_post_decomposition_residual": 0.0,
        "max_owner_clip_residual": 0.0,
        "max_active_monotonic_residual": 0.0,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
    }
    for owner, count in (("token_veto", 21), ("global_absolute", 44)):
        runtime[f"last_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"max_{owner}_grad_norm_preclip"] = 2.0
        runtime[f"nonfinite_{owner}_gradient_boundaries"] = 0
        runtime[f"zero_{owner}_gradient_successful_steps"] = 0
        runtime[f"expected_{owner}_tensor_count"] = count
        runtime[f"last_observed_{owner}_tensor_count"] = count
    return runtime


def _v53_state(*, tensor_count: int = 65, element_count: int = 534_725):
    if tensor_count <= 0 or element_count < tensor_count:
        raise ValueError("test state requires at least one element per tensor")
    state = {
        "stage_b_fixed_text_scorer._dense_duty_contract_version": torch.tensor(2),
        "stage_b_fixed_text_scorer.rank_tower.frozen": torch.zeros(1),
    }
    active_names = []
    for index in range(tensor_count):
        name = f"stage_b_fixed_text_scorer.confidence_adapter.v53_{index:02d}"
        size = 1 if index < tensor_count - 1 else element_count - tensor_count + 1
        state[name] = torch.zeros(size)
        active_names.append(name)
    return state, active_names


def _v53_checkpoint_payload(*, tensor_count: int = 65) -> dict:
    state, active_names = _v53_state(tensor_count=tensor_count)
    initial = fingerprint_state(
        state, active_parameter_names=active_names, phase="confidence"
    )
    current = {name: value.clone() for name, value in state.items()}
    current[active_names[0]].add_(1.0)
    return {
        "model": current,
        "optimizer_updates": 1,
        "args": {
            "stage_b_dense_duty": True,
            "stage_b_dense_duty_phase": "confidence",
            "stage_b_dense_duty_execution_scope": "formal",
            "stage_b_v22_score_ownership": (
                "rank_tower_stopgrad_token_adapter_two_phase"
            ),
            "stage_b_dense_duty_confidence_revision": (
                "word_veto_rank_full_expression_global_absolute_v53"
            ),
            "stage_b_dense_duty_confidence_head_gradient_contract": (
                "split_token_veto_fulltext_global_absolute_v7"
            ),
            "stage_b_dense_duty_confidence_pool_feature_contract": (
                "detached_rank_full_expression_candidate_residual_global_pool_v10"
            ),
            "stage_b_dense_duty_confidence_gate_gradient_contract": (
                "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
            ),
            "stage_b_dense_duty_confidence_probe_admission_contract": (
                "u400_word_veto_rank_full_expression_global_absolute_"
                "confidence_strict1607_v53"
            ),
            "stage_b_v11_trainable_params_min": 534_725,
            "stage_b_v11_trainable_params_max": 534_725,
            "stage_b_dense_duty_runtime_audit": _healthy_v53_runtime(1),
            FINGERPRINT_ARG: initial,
        },
    }


def _v55_checkpoint_payload(*, tensor_count: int = 65) -> dict:
    payload = _v53_checkpoint_payload(tensor_count=tensor_count)
    args = payload["args"]
    args["stage_b_dense_duty_confidence_revision"] = (
        "word_veto_rank_full_expression_global_independent_absolute_v55"
    )
    args["stage_b_dense_duty_confidence_head_gradient_contract"] = (
        "split_token_veto_local_candidate_global_absolute_v8"
    )
    args["stage_b_dense_duty_confidence_pool_feature_contract"] = (
        "detached_rank_full_expression_local_candidate_"
        "frozen_rank_global_pool_v12"
    )
    args["stage_b_dense_duty_positive_trust_contract"] = (
        "absolute_global_pool_logit_v4"
    )
    args["stage_b_dense_duty_confidence_probe_admission_contract"] = (
        "u400_word_veto_rank_full_expression_global_independent_absolute_"
        "confidence_strict1607_v55"
    )
    return payload


def _v55_probe_evaluation_fixture() -> tuple[dict, SLConfig]:
    cfg = SLConfig.fromfile(str(V55_CONFIG))
    cfg.stage_b_dense_duty_evaluation_scope = "probe"
    saved_args = cfg._cfg_dict.to_dict()
    saved_args["stage_b_dense_duty_execution_scope"] = "probe"

    payload = _v55_checkpoint_payload()
    saved_args[FINGERPRINT_ARG] = payload["args"][FINGERPRINT_ARG]
    saved_args["stage_b_dense_duty_runtime_audit"] = _healthy_v53_runtime(1)
    payload["args"] = saved_args
    payload["checkpoint_reason"] = "max_train_iters"
    return payload, cfg


def test_v53_training_contract_is_exact_v35_and_seals_full_surface():
    contract = build_training_contract(_training_args(V53_CONFIG))

    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v35"
    assert contract["values"]["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_rank_full_expression_global_absolute_v53"
    )
    assert contract["values"][
        "stage_b_dense_duty_confidence_head_gradient_contract"
    ] == "split_token_veto_fulltext_global_absolute_v7"
    assert contract["values"][
        "stage_b_dense_duty_confidence_pool_feature_contract"
    ] == "detached_rank_full_expression_candidate_residual_global_pool_v10"
    assert contract["values"][
        "stage_b_dense_duty_confidence_gate_gradient_contract"
    ] == "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
    assert contract["values"]["stage_b_v11_trainable_params_min"] == 534_725
    assert contract["values"]["stage_b_v11_trainable_params_max"] == 534_725


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_candidate_absolute_sample_calibrator_v6",
        ),
        (
            "stage_b_dense_duty_confidence_pool_feature_contract",
            "detached_candidate_absolute_sample_calibrator_global_pool_v9",
        ),
        (
            "stage_b_dense_duty_confidence_gate_gradient_contract",
            "candidate_patch_asymmetric_monotone_veto_absolute_logit_st_v15",
        ),
        ("stage_b_v11_trainable_params_min", 534_724),
        ("stage_b_v11_trainable_params_max", 534_726),
        (
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "u400_word_veto_candidate_sample_calibrator_confidence_strict1607_v52",
        ),
    ),
)
def test_v53_training_contract_fails_closed_on_exact_surface_drift(field, value):
    args = _training_args(V53_CONFIG)
    args[field] = value

    with pytest.raises(
        RuntimeError, match="V53 full-text global-absolute confidence contract drifted"
    ):
        build_training_contract(args)


def test_v53_probe_keeps_v35_while_disabling_formal_admission():
    contract = build_training_contract(_training_args(V53_PROBE_CONFIG))
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v35"
    assert (
        contract["values"].get(
            "stage_b_dense_duty_confidence_probe_admission_contract"
        )
        is None
    )


def test_v53_formal_config_is_allowlisted_and_closed_over():
    validate_formal_invocation(
        SimpleNamespace(
            stage_b_dense_duty_execution_scope="formal",
            stage_b_dense_duty_phase="confidence",
            stage_b_v22_score_ownership=(
                "rank_tower_stopgrad_token_adapter_two_phase"
            ),
            options=None,
            config_file=str(V53_CONFIG),
        ),
        repo_root=REPO_ROOT,
    )
    closure = build_source_closure(V53_CONFIG, repo_root=REPO_ROOT)
    config_paths = {item["path"] for item in closure["config"]["files"]}
    assert closure["config"]["entry"] == V53_CONFIG.relative_to(REPO_ROOT).as_posix()
    assert V53_CONFIG.relative_to(REPO_ROOT).as_posix() in config_paths
    assert V52_CONFIG.relative_to(REPO_ROOT).as_posix() in config_paths


def test_v52_training_schema_is_preserved():
    assert build_training_contract(_training_args(V52_CONFIG))["schema"] == (
        "pivot.stageb.dense_duty_training_contract/v34"
    )


def test_v53_runtime_accepts_exact_two_owner_evidence():
    runtime = _healthy_v53_runtime()
    assert _validate_fulltext_global_absolute_runtime_audit(
        runtime, expected_steps=400
    ) == runtime


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("successful_optimizer_steps", 399),
        ("clip_contract_checked_steps", 399),
        (
            "clip_contract_schema",
            "pivot.stageb.dense_duty_three_owner_clip_contract/v1",
        ),
        ("owner_clip_violation_steps", 1),
        ("nonfinite_gradient_boundaries", 1),
        ("zero_gradient_successful_steps", 1),
        ("last_observed_token_veto_tensor_count", 20),
        ("expected_global_absolute_tensor_count", 43),
        ("nonfinite_global_absolute_gradient_boundaries", 1),
        ("zero_token_veto_gradient_successful_steps", 1),
        ("last_token_veto_grad_norm_preclip", 0.0),
        ("max_global_absolute_grad_norm_preclip", float("nan")),
        ("max_owner_clip_residual", 1.0e-3),
        ("clip_contract_max_norm", 0.2),
    ),
)
def test_v53_runtime_fails_closed_on_incomplete_or_bad_evidence(field, value):
    runtime = _healthy_v53_runtime()
    runtime[field] = value

    with pytest.raises(RuntimeError, match="V53"):
        _validate_fulltext_global_absolute_runtime_audit(
            runtime, expected_steps=400
        )


def test_v53_checkpoint_audit_enforces_exact_production_ownership():
    audit = audit_checkpoint_payload(_v53_checkpoint_payload())
    assert audit["status"] == "passed"
    assert audit["ownership"]["active_parameter_count"] == 65
    assert audit["current"]["active"]["element_count"] == 534_725

    with pytest.raises(RuntimeError, match="65-tensor/534725-element"):
        audit_checkpoint_payload(_v53_checkpoint_payload(tensor_count=64))


def test_v53_checkpoint_audit_requires_every_successful_step_checked():
    payload = _v53_checkpoint_payload()
    payload["args"]["stage_b_dense_duty_runtime_audit"][
        "clip_contract_checked_steps"
    ] = 0

    with pytest.raises(RuntimeError, match="one two-owner clip check"):
        audit_checkpoint_payload(payload)


def test_v54_training_contract_is_v36_and_seals_exact_residual_trust():
    contract = build_training_contract(_training_args(V54_CONFIG))

    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v36"
    values = contract["values"]
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
    )
    assert values["stage_b_dense_duty_confidence_pool_feature_contract"] == (
        "detached_rank_full_expression_candidate_residual_global_pool_"
        "exact_rank_max_reference_v11"
    )
    assert values["stage_b_dense_duty_positive_trust_contract"] == (
        "exact_frozen_rank_max_confidence_delta_v3"
    )
    assert values["stage_b_v11_trainable_params_min"] == 534_725
    assert values["stage_b_v11_trainable_params_max"] == 534_725


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_pool_feature_contract",
            "detached_rank_full_expression_candidate_residual_global_pool_v10",
        ),
        (
            "stage_b_dense_duty_positive_trust_contract",
            "absolute_global_confidence_logit_v2",
        ),
        (
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "u400_word_veto_rank_full_expression_global_absolute_"
            "confidence_strict1607_v53",
        ),
    ),
)
def test_v54_training_contract_rejects_v53_contract_mixing(field, value):
    args = _training_args(V54_CONFIG)
    args[field] = value

    with pytest.raises(RuntimeError, match="V54 exact-residual"):
        build_training_contract(args)


def test_v54_probe_keeps_v36_while_disabling_formal_admission():
    contract = build_training_contract(_training_args(V54_PROBE_CONFIG))
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v36"
    assert (
        contract["values"].get(
            "stage_b_dense_duty_confidence_probe_admission_contract"
        )
        is None
    )


def test_v54_formal_config_is_allowlisted_and_closed_over():
    validate_formal_invocation(
        SimpleNamespace(
            stage_b_dense_duty_execution_scope="formal",
            stage_b_dense_duty_phase="confidence",
            stage_b_v22_score_ownership=(
                "rank_tower_stopgrad_token_adapter_two_phase"
            ),
            options=None,
            config_file=str(V54_CONFIG),
        ),
        repo_root=REPO_ROOT,
    )
    closure = build_source_closure(V54_CONFIG, repo_root=REPO_ROOT)
    config_paths = {item["path"] for item in closure["config"]["files"]}
    assert closure["config"]["entry"] == V54_CONFIG.relative_to(REPO_ROOT).as_posix()
    assert V54_CONFIG.relative_to(REPO_ROOT).as_posix() in config_paths
    assert V53_CONFIG.relative_to(REPO_ROOT).as_posix() in config_paths


def test_v54_checkpoint_audit_reuses_exact_two_owner_ownership():
    payload = _v53_checkpoint_payload()
    args = payload["args"]
    args["stage_b_dense_duty_confidence_revision"] = (
        "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
    )
    args["stage_b_dense_duty_confidence_pool_feature_contract"] = (
        "detached_rank_full_expression_candidate_residual_global_pool_"
        "exact_rank_max_reference_v11"
    )
    args["stage_b_dense_duty_positive_trust_contract"] = (
        "exact_frozen_rank_max_confidence_delta_v3"
    )
    args["stage_b_dense_duty_confidence_probe_admission_contract"] = (
        "u400_word_veto_rank_full_expression_global_absolute_exact_residual_"
        "confidence_strict1607_v54"
    )

    audit = audit_checkpoint_payload(payload)
    assert audit["status"] == "passed"
    assert audit["ownership"]["active_parameter_count"] == 65
    assert audit["current"]["active"]["element_count"] == 534_725


def test_v55_training_contract_is_v37_and_seals_independent_global_absolute():
    contract = build_training_contract(_training_args(V55_CONFIG))

    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v37"
    values = contract["values"]
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_rank_full_expression_global_independent_absolute_v55"
    )
    assert values["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_local_candidate_global_absolute_v8"
    )
    assert values["stage_b_dense_duty_confidence_pool_feature_contract"] == (
        "detached_rank_full_expression_local_candidate_"
        "frozen_rank_global_pool_v12"
    )
    assert values["stage_b_dense_duty_positive_trust_contract"] == (
        "absolute_global_pool_logit_v4"
    )
    assert values["stage_b_v11_trainable_params_min"] == 534_725
    assert values["stage_b_v11_trainable_params_max"] == 534_725


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_fulltext_global_absolute_v7",
        ),
        (
            "stage_b_dense_duty_confidence_pool_feature_contract",
            "detached_rank_full_expression_candidate_residual_global_pool_"
            "exact_rank_max_reference_v11",
        ),
        (
            "stage_b_dense_duty_positive_trust_contract",
            "exact_frozen_rank_max_confidence_delta_v3",
        ),
        (
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "u400_word_veto_rank_full_expression_global_absolute_exact_residual_"
            "confidence_strict1607_v54",
        ),
    ),
)
def test_v55_training_contract_rejects_v54_contract_mixing(field, value):
    args = _training_args(V55_CONFIG)
    args[field] = value

    with pytest.raises(RuntimeError, match="V55 independent global-absolute"):
        build_training_contract(args)


def test_v55_probe_keeps_v37_while_disabling_formal_admission():
    contract = build_training_contract(_training_args(V55_PROBE_CONFIG))
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v37"
    assert (
        contract["values"].get(
            "stage_b_dense_duty_confidence_probe_admission_contract"
        )
        is None
    )


def test_v55_formal_config_is_allowlisted_and_closed_over_its_ancestors():
    validate_formal_invocation(
        SimpleNamespace(
            stage_b_dense_duty_execution_scope="formal",
            stage_b_dense_duty_phase="confidence",
            stage_b_v22_score_ownership=(
                "rank_tower_stopgrad_token_adapter_two_phase"
            ),
            options=None,
            config_file=str(V55_CONFIG),
        ),
        repo_root=REPO_ROOT,
    )
    closure = build_source_closure(V55_CONFIG, repo_root=REPO_ROOT)
    config_paths = {item["path"] for item in closure["config"]["files"]}
    assert closure["config"]["entry"] == V55_CONFIG.relative_to(
        REPO_ROOT
    ).as_posix()
    assert V55_CONFIG.relative_to(REPO_ROOT).as_posix() in config_paths
    assert V54_CONFIG.relative_to(REPO_ROOT).as_posix() in config_paths
    assert V53_CONFIG.relative_to(REPO_ROOT).as_posix() in config_paths
    assert V52_CONFIG.relative_to(REPO_ROOT).as_posix() in config_paths


def test_v55_checkpoint_audit_reuses_exact_two_owner_ownership():
    audit = audit_checkpoint_payload(_v55_checkpoint_payload())
    assert audit["status"] == "passed"
    assert audit["ownership"]["active_parameter_count"] == 65
    assert audit["current"]["active"]["element_count"] == 534_725

    with pytest.raises(RuntimeError, match="65-tensor/534725-element"):
        audit_checkpoint_payload(_v55_checkpoint_payload(tensor_count=64))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_gate_gradient_contract",
            "drifted_gate_contract",
        ),
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_fulltext_global_absolute_v7",
        ),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "drifted_tail_reduction",
        ),
        ("stage_b_v15_tail_queue_positive_trust_reduction_contract", "drifted"),
        ("stage_b_v15_tail_queue_negative_reduction_contract", "drifted"),
        ("stage_b_v21_token_edit_query_scope", "drifted_query_scope"),
        (
            "stage_b_dense_duty_confidence_probe_admission_report",
            "/tmp/drifted-v55-admission-report.json",
        ),
    ),
)
def test_v55_evaluation_checkpoint_config_comparison_fails_closed(
    field, value
):
    payload, cfg = _v55_probe_evaluation_fixture()
    setattr(cfg, field, value)

    with pytest.raises(
        RuntimeError, match="evaluation configuration drifted from training"
    ):
        validate_evaluation_checkpoint_payload(payload, cfg)
