from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util import stage_b_dense_duty_audit as dense_duty_audit
from util.slconfig import SLConfig


V53_CONFIG = combined_eval._FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_U0400_CONFIG


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V53_CONFIG),
        output_dir=str(tmp_path / "v53-strict1607"),
        ckpts=["fulltext-global-u400-checkpoint.pth"],
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


def _checkpoint_payload(tmp_path: Path, *, training_schema: str) -> dict:
    return {
        "args": {
            "output_dir": str(tmp_path),
            "stage_b_dense_duty_training_contract": {
                "schema": training_schema,
                "values": {},
            },
        }
    }


def _migration_audit() -> dict:
    return {
        "schema": ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
        "source_optimizer_updates": 6551,
        "fresh_confidence_contract": (
            ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
        ),
        "head_gradient_contract": (
            ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_HEAD_CONTRACT
        ),
        "pool_feature_contract": (
            ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
        ),
    }


def _runtime_audit() -> dict:
    runtime = {
        "clip_contract_schema": ref_eval._V53_TWO_OWNER_CLIP_CONTRACT_SCHEMA,
        "clip_contract_checked_steps": 400,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
        "clip_contract_tolerance": 1e-6,
        "clip_contract_max_norm": 0.1,
        "max_active_pre_decomposition_residual": 0.0,
        "max_active_post_decomposition_residual": 0.0,
        "max_owner_clip_residual": 0.0,
        "max_active_monotonic_residual": 0.0,
    }
    for owner, count in (
        ("token_veto", 21),
        ("global_absolute", 44),
    ):
        runtime[f"last_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"max_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"expected_{owner}_tensor_count"] = count
        runtime[f"last_observed_{owner}_tensor_count"] = count
    return runtime


def test_v53_probe_config_is_registered_and_admitted_by_both_evaluators(
    tmp_path,
):
    cfg = SLConfig.fromfile(str(V53_CONFIG))

    assert ref_eval._validate_v53_fulltext_global_absolute_config(cfg) is True
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_candidate_absolute_sample_calibrator_v6",
        ),
        (
            "stage_b_dense_duty_confidence_gate_gradient_contract",
            "candidate_raw_patch_asymmetric_deployed_routing_st_v15",
        ),
        (
            "stage_b_dense_duty_confidence_pool_feature_contract",
            "detached_candidate_absolute_raw_patch_asymmetric_veto_logits_v8",
        ),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.1),
        ("stage_b_dense_duty_deployed_veto_positive_max", 0.2),
        ("stage_b_dense_duty_deployed_veto_tn_min", 0.8),
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
        ("stage_b_v21_token_edit_query_scope", "detached_argmax_v2"),
        (
            "stage_b_dense_duty_positive_trust_contract",
            "net_total_confidence_delta_v1",
        ),
        (
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "tn_only_positive_detached_v2",
        ),
        (
            "stage_b_dense_duty_confidence_carrier_selector_contract",
            "final_layer_confidence_argmax_v1",
        ),
        (
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5",
        ),
        ("stage_b_dense_duty_rank_source_optimizer_updates", 400),
        ("stage_b_v11_trainable_params_min", 535_945),
        ("stage_b_v11_trainable_params_max", 535_945),
    ),
)
def test_v53_evaluators_reject_contract_drift(tmp_path, field, value):
    cfg = SLConfig.fromfile(str(V53_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(
        RuntimeError,
        match="v53 fulltext/global-absolute confidence config drifted",
    ):
        ref_eval._validate_v53_fulltext_global_absolute_config(cfg)
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v53_evaluators_reject_unknown_revision(tmp_path):
    cfg = SLConfig.fromfile(str(V53_CONFIG))
    cfg.stage_b_dense_duty_confidence_revision = (
        "word_veto_rank_full_expression_global_absolute"
    )

    assert ref_eval._validate_v53_fulltext_global_absolute_config(cfg) is False
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v53_checkpoint_rejects_pre_v35_training_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v53-schema-contract-check")
    cfg = SLConfig.fromfile(str(V53_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema="pivot.stageb.dense_duty_training_contract/v34",
    )
    monkeypatch.setattr(
        dense_duty_audit,
        "audit_checkpoint_payload",
        lambda *_args, **_kwargs: {"status": "passed", "phase": "confidence"},
    )

    with pytest.raises(RuntimeError, match="exact v35 training contract"):
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )


def test_v53_checkpoint_rejects_non_v20_migration(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v53-migration-contract-check")
    cfg = SLConfig.fromfile(str(V53_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema=ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINING_CONTRACT_SCHEMA,
    )
    payload["args"][
        "stage_b_dense_duty_confidence_adapter_migration_audit"
    ] = {
        **_migration_audit(),
        "schema": ref_eval._V52_CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA,
    }
    payload["model"] = {}
    monkeypatch.setattr(
        dense_duty_audit,
        "audit_checkpoint_payload",
        lambda *_args, **_kwargs: {"status": "passed", "phase": "confidence"},
    )

    with pytest.raises(RuntimeError, match="exact fresh-U6551"):
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize(
    "forbidden_name",
    (
        "stage_b_fixed_text_scorer.confidence_adapter.deployed_router.weight",
        "stage_b_fixed_text_scorer.confidence_adapter.patch_residual.weight",
        "stage_b_fixed_text_scorer.confidence_adapter.global_query_norm.weight",
        "stage_b_fixed_text_scorer.confidence_adapter.veto_cap_raw_ceiling",
        "stage_b_fixed_text_scorer.confidence_adapter.candidate_coverage_depth_raw",
    ),
)
def test_v53_checkpoint_rejects_non_v53_parameter_surface(
    tmp_path, monkeypatch, forbidden_name
):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v53-two-owner-contract-check")
    cfg = SLConfig.fromfile(str(V53_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema=ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINING_CONTRACT_SCHEMA,
    )
    payload["args"][
        "stage_b_dense_duty_confidence_adapter_migration_audit"
    ] = _migration_audit()
    payload["model"] = {forbidden_name: None}
    monkeypatch.setattr(
        dense_duty_audit,
        "audit_checkpoint_payload",
        lambda *_args, **_kwargs: {"status": "passed", "phase": "confidence"},
    )

    with pytest.raises(RuntimeError, match="complete two-owner parameter surface"):
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )


def test_v53_runtime_audit_accepts_complete_two_owner_evidence():
    ref_eval._validate_v53_two_owner_runtime_audit(
        _runtime_audit(), optimizer_updates=400
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("clip_contract_schema", "pivot.stageb.dense_duty_runtime_audit/v1"),
        ("clip_contract_checked_steps", 399),
        ("owner_clip_violation_steps", 1),
        ("last_token_veto_grad_norm_preclip", 0.0),
        ("max_global_absolute_grad_norm_preclip", 0.0),
        ("nonfinite_global_absolute_gradient_boundaries", 1),
        ("zero_token_veto_gradient_successful_steps", 1),
        ("expected_token_veto_tensor_count", 20),
        ("last_observed_global_absolute_tensor_count", 43),
        ("clip_contract_max_norm", 1.0),
        ("max_owner_clip_residual", 2e-6),
        ("max_candidate_absolute_grad_norm_preclip", 1.0),
    ),
)
def test_v53_runtime_audit_fails_closed(field, value):
    runtime = _runtime_audit()
    runtime[field] = value

    with pytest.raises(RuntimeError, match="two-owner gradient/clip"):
        ref_eval._validate_v53_two_owner_runtime_audit(
            runtime, optimizer_updates=400
        )
