from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util import stage_b_dense_duty_audit as dense_duty_audit
from util.slconfig import SLConfig


V54_CONFIG = (
    combined_eval._FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_CONFIDENCE_U0400_CONFIG
)


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V54_CONFIG),
        output_dir=str(tmp_path / "v54-strict1607"),
        ckpts=["fulltext-global-exact-residual-u400-checkpoint.pth"],
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
        "schema": (
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
        ),
        "source_optimizer_updates": 6551,
        "fresh_confidence_contract": (
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
        ),
        "head_gradient_contract": (
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_HEAD_CONTRACT
        ),
        "pool_feature_contract": (
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
        ),
    }


def _runtime_audit() -> dict:
    runtime = {
        "clip_contract_schema": ref_eval._V54_TWO_OWNER_CLIP_CONTRACT_SCHEMA,
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
    for owner, count in (("token_veto", 21), ("global_absolute", 44)):
        runtime[f"last_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"max_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"expected_{owner}_tensor_count"] = count
        runtime[f"last_observed_{owner}_tensor_count"] = count
    return runtime


def test_v54_probe_config_is_registered_and_admitted_by_both_evaluators(tmp_path):
    cfg = SLConfig.fromfile(str(V54_CONFIG))

    assert ref_eval._validate_v54_fulltext_global_absolute_exact_residual_config(cfg)
    assert not ref_eval._validate_v53_fulltext_global_absolute_config(cfg)
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_pool_feature_contract",
            ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT,
        ),
        (
            "stage_b_dense_duty_positive_trust_contract",
            ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_POSITIVE_TRUST,
        ),
        ("stage_b_dense_duty_rank_source_optimizer_updates", 400),
        ("stage_b_v11_trainable_params_max", 534_726),
    ),
)
def test_v54_evaluators_reject_contract_drift(tmp_path, field, value):
    cfg = SLConfig.fromfile(str(V54_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(RuntimeError, match="v54 fulltext/global-absolute"):
        ref_eval._validate_v54_fulltext_global_absolute_exact_residual_config(cfg)
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v54_checkpoint_rejects_pre_v36_training_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v54-schema-contract-check")
    cfg = SLConfig.fromfile(str(V54_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema=ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINING_CONTRACT_SCHEMA,
    )
    monkeypatch.setattr(
        dense_duty_audit,
        "audit_checkpoint_payload",
        lambda *_args, **_kwargs: {"status": "passed", "phase": "confidence"},
    )

    with pytest.raises(RuntimeError, match="exact v36 training contract"):
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA),
        (
            "fresh_confidence_contract",
            ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
        ),
        (
            "pool_feature_contract",
            ref_eval._V53_FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT,
        ),
    ),
)
def test_v54_checkpoint_rejects_stale_migration_lineage(
    tmp_path, monkeypatch, field, value
):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v54-migration-contract-check")
    cfg = SLConfig.fromfile(str(V54_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema=(
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_TRAINING_CONTRACT_SCHEMA
        ),
    )
    payload["args"][
        "stage_b_dense_duty_confidence_adapter_migration_audit"
    ] = {**_migration_audit(), field: value}
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


def test_v54_runtime_audit_accepts_unchanged_two_owner_counts():
    ref_eval._validate_v54_two_owner_runtime_audit(
        _runtime_audit(), optimizer_updates=400
    )
