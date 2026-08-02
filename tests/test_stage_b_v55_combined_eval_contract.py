from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util import stage_b_dense_duty_audit as dense_duty_audit
from util.slconfig import SLConfig


V55_CONFIG = (
    combined_eval._FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_CONFIDENCE_U0400_CONFIG
)
V55_FORMAL_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "fulltext_global_independent_absolute_20260802.py"
)


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V55_CONFIG),
        output_dir=str(tmp_path / "v55-strict1607"),
        ckpts=["fulltext-global-independent-u400-checkpoint.pth"],
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
            ref_eval._V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA
        ),
        "source_optimizer_updates": 6551,
        "fresh_confidence_contract": (
            ref_eval._V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
        ),
        "head_gradient_contract": (
            ref_eval._V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_CONTRACT
        ),
        "pool_feature_contract": (
            ref_eval._V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT
        ),
    }


def _runtime_audit() -> dict:
    runtime = {
        "clip_contract_schema": ref_eval._V55_TWO_OWNER_CLIP_CONTRACT_SCHEMA,
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


def test_v55_probe_config_is_registered_and_admitted_by_both_evaluators(tmp_path):
    cfg = SLConfig.fromfile(str(V55_CONFIG))

    assert ref_eval._validate_v55_fulltext_global_independent_absolute_config(cfg)
    assert not ref_eval._validate_v54_fulltext_global_absolute_exact_residual_config(
        cfg
    )
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_HEAD_CONTRACT,
        ),
        (
            "stage_b_dense_duty_confidence_pool_feature_contract",
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT,
        ),
        (
            "stage_b_dense_duty_positive_trust_contract",
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POSITIVE_TRUST,
        ),
        ("stage_b_dense_duty_rank_source_optimizer_updates", 400),
        ("stage_b_v11_trainable_params_max", 534_726),
    ),
)
def test_v55_evaluators_reject_carrier_or_surface_drift(tmp_path, field, value):
    cfg = SLConfig.fromfile(str(V55_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(RuntimeError, match="v55 fulltext/global-absolute"):
        ref_eval._validate_v55_fulltext_global_independent_absolute_config(cfg)
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v55_checkpoint_rejects_pre_v37_training_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v55-schema-contract-check")
    cfg = SLConfig.fromfile(str(V55_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema=(
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_TRAINING_CONTRACT_SCHEMA
        ),
    )
    monkeypatch.setattr(
        dense_duty_audit,
        "audit_checkpoint_payload",
        lambda *_args, **_kwargs: {"status": "passed", "phase": "confidence"},
    )

    with pytest.raises(RuntimeError, match="exact v37 training contract"):
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "schema",
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA,
        ),
        (
            "fresh_confidence_contract",
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT,
        ),
        (
            "head_gradient_contract",
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_HEAD_CONTRACT,
        ),
        (
            "pool_feature_contract",
            ref_eval._V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT,
        ),
    ),
)
def test_v55_checkpoint_rejects_stale_carrier_lineage(
    tmp_path, monkeypatch, field, value
):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v55-migration-contract-check")
    cfg = SLConfig.fromfile(str(V55_CONFIG))
    payload = _checkpoint_payload(
        tmp_path,
        training_schema=(
            ref_eval._V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_TRAINING_CONTRACT_SCHEMA
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


def test_v55_runtime_audit_requires_unchanged_two_owner_counts():
    runtime = _runtime_audit()
    ref_eval._validate_v55_two_owner_runtime_audit(runtime, optimizer_updates=400)

    runtime["last_observed_global_absolute_tensor_count"] = 43
    with pytest.raises(RuntimeError, match="v55 confidence checkpoint"):
        ref_eval._validate_v55_two_owner_runtime_audit(
            runtime, optimizer_updates=400
        )


def test_v55_formal_admission_binds_its_canonical_controller(tmp_path, monkeypatch):
    from tools import (
        run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_evaluation
        as promotion,
    )

    report = tmp_path / "u000400_strict1607_report.json"
    report.write_text("{}\n", encoding="utf-8")
    sentinel = {"schema": "v55-admission-sentinel"}
    monkeypatch.setattr(promotion, "REPORT", report)
    monkeypatch.setattr(
        promotion,
        "verify_admission_report",
        lambda path: sentinel if Path(path) == report else None,
    )
    cfg = SLConfig.fromfile(str(V55_FORMAL_CONFIG))
    cfg.stage_b_dense_duty_confidence_probe_admission_report = str(report)

    ref_eval._bind_dense_duty_formal_probe_admission(cfg)

    assert cfg.stage_b_dense_duty_confidence_probe_admission_audit == sentinel
