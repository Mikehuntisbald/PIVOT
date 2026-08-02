from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    audit_stageb_confidence_adapter_candidate_sample_calibrator_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_sample_calibrator_probe_evaluation
    as probe_evaluation,
)
from util import stage_b_dense_duty_audit as dense_duty_audit
from util.slconfig import SLConfig


V52_CONFIG = combined_eval._CANDIDATE_SAMPLE_CALIBRATOR_CONFIDENCE_U0400_CONFIG


def test_v52_probe_controller_loads_health_audit_callable():
    assert probe_evaluation._CORE._load_health_audit() is health.audit


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V52_CONFIG),
        output_dir=str(tmp_path / "v52-strict1607"),
        ckpts=["candidate-u400-checkpoint.pth"],
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
        "schema": ref_eval._V52_CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA,
        "source_optimizer_updates": 6551,
        "head_gradient_contract": (
            ref_eval._V52_CANDIDATE_SAMPLE_CALIBRATOR_HEAD_CONTRACT
        ),
    }


def _runtime_audit() -> dict:
    runtime = {
        "clip_contract_schema": ref_eval._V52_THREE_OWNER_CLIP_CONTRACT_SCHEMA,
        "clip_contract_checked_steps": 400,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
    }
    for owner in ("token_veto", "candidate_absolute", "sample_calibrator"):
        runtime[f"max_{owner}_grad_norm_preclip"] = 1.0
    return runtime


def test_v52_probe_config_is_registered_and_admitted_by_both_evaluators(
    tmp_path,
):
    cfg = SLConfig.fromfile(str(V52_CONFIG))

    assert ref_eval._validate_v52_candidate_sample_calibrator_config(cfg) is True
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_deployed_router_global_absolute_v5",
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
            "stage_b_dense_duty_confidence_gate_gradient_contract",
            "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13",
        ),
        (
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "tn_only_positive_detached_v2",
        ),
        (
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5",
        ),
        (
            "stage_b_dense_duty_confidence_pool_feature_contract",
            "detached_candidate_absolute_patch_invariant_monotone_veto_logits_v6",
        ),
        ("stage_b_dense_duty_rank_source_optimizer_updates", 400),
        ("stage_b_v11_trainable_params_min", 536_734),
        ("stage_b_v11_trainable_params_max", 536_734),
    ),
)
def test_v52_evaluators_reject_contract_drift(tmp_path, field, value):
    cfg = SLConfig.fromfile(str(V52_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(
        RuntimeError,
        match="v52 candidate/sample-calibrator confidence config drifted",
    ):
        ref_eval._validate_v52_candidate_sample_calibrator_config(cfg)
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v52_evaluators_reject_unknown_revision(tmp_path):
    cfg = SLConfig.fromfile(str(V52_CONFIG))
    cfg.stage_b_dense_duty_confidence_revision = (
        "word_veto_candidate_sample_calibrator_v52"
    )

    assert ref_eval._validate_v52_candidate_sample_calibrator_config(cfg) is False
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v52_checkpoint_rejects_pre_v34_training_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v52-schema-contract-check")
    cfg = SLConfig.fromfile(str(V52_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema="pivot.stageb.dense_duty_training_contract/v33",
    )
    monkeypatch.setattr(
        dense_duty_audit,
        "audit_checkpoint_payload",
        lambda *_args, **_kwargs: {"status": "passed", "phase": "confidence"},
    )

    with pytest.raises(RuntimeError, match="exact v34 training contract"):
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )


def test_v52_checkpoint_rejects_non_v19_migration(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v52-migration-contract-check")
    cfg = SLConfig.fromfile(str(V52_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema=ref_eval._V52_CANDIDATE_SAMPLE_CALIBRATOR_TRAINING_CONTRACT_SCHEMA,
    )
    payload["args"]["stage_b_dense_duty_confidence_adapter_migration_audit"] = {
        **_migration_audit(),
        "schema": "pivot.stageb.rank_to_token_confidence_adapter_deployed_router/v18",
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


def test_v52_checkpoint_rejects_deployed_router_parameters(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v52-no-router-contract-check")
    cfg = SLConfig.fromfile(str(V52_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema=ref_eval._V52_CANDIDATE_SAMPLE_CALIBRATOR_TRAINING_CONTRACT_SCHEMA,
    )
    payload["args"][
        "stage_b_dense_duty_confidence_adapter_migration_audit"
    ] = _migration_audit()
    payload["model"] = {
        "stage_b_fixed_text_scorer.confidence_adapter.deployed_router.weight": None
    }
    monkeypatch.setattr(
        dense_duty_audit,
        "audit_checkpoint_payload",
        lambda *_args, **_kwargs: {"status": "passed", "phase": "confidence"},
    )

    with pytest.raises(RuntimeError, match="without deployed-router parameters"):
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )


def test_v52_runtime_audit_accepts_complete_three_owner_evidence():
    ref_eval._validate_v52_three_owner_runtime_audit(
        _runtime_audit(), optimizer_updates=400
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("clip_contract_schema", "pivot.stageb.dense_duty_runtime_audit/v1"),
        ("clip_contract_checked_steps", 399),
        ("owner_clip_violation_steps", 1),
        ("max_token_veto_grad_norm_preclip", 0.0),
        ("nonfinite_candidate_absolute_gradient_boundaries", 1),
        ("zero_sample_calibrator_gradient_successful_steps", 1),
        ("max_deployed_router_grad_norm_preclip", 1.0),
    ),
)
def test_v52_runtime_audit_fails_closed(field, value):
    runtime = _runtime_audit()
    runtime[field] = value

    with pytest.raises(RuntimeError, match="no-router three-owner"):
        ref_eval._validate_v52_three_owner_runtime_audit(
            runtime, optimizer_updates=400
        )
