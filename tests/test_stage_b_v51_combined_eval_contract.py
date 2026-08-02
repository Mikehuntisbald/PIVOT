from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util import stage_b_dense_duty_audit as dense_duty_audit
from util.slconfig import SLConfig


V51_CONFIG = (
    combined_eval._CANDIDATE_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_CONFIDENCE_U0400_CONFIG
)


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V51_CONFIG),
        output_dir=str(tmp_path / "v51-strict1607"),
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


def test_v51_probe_config_is_registered_and_admitted_by_both_evaluators(
    tmp_path,
):
    cfg = SLConfig.fromfile(str(V51_CONFIG))

    assert (
        ref_eval._validate_v51_split_independent_deployed_router_config(cfg)
        is True
    )
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_v2",
        ),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.25),
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
        (
            "stage_b_v21_token_edit_query_scope",
            "target_iou_union_detached_final_confidence_base_argmax_v2",
        ),
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
    ),
)
def test_v51_evaluators_reject_contract_drift(tmp_path, field, value):
    cfg = SLConfig.fromfile(str(V51_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(
        RuntimeError,
        match="v51 split-independent-deployed-router confidence config drifted",
    ):
        ref_eval._validate_v51_split_independent_deployed_router_config(cfg)
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v51_evaluators_reject_the_pre_rename_revision(tmp_path):
    cfg = SLConfig.fromfile(str(V51_CONFIG))
    cfg.stage_b_dense_duty_confidence_revision = (
        "word_veto_candidate_split_deployed_router_v51"
    )

    assert (
        ref_eval._validate_v51_split_independent_deployed_router_config(cfg)
        is False
    )
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v51_checkpoint_rejects_pre_v33_training_contract(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"v51-schema-contract-check")
    cfg = SLConfig.fromfile(str(V51_CONFIG))
    payload = {
        "args": {
            "output_dir": str(tmp_path),
            "stage_b_dense_duty_training_contract": {
                "schema": "pivot.stageb.dense_duty_training_contract/v32",
                "values": {},
            },
        }
    }
    monkeypatch.setattr(
        dense_duty_audit,
        "audit_checkpoint_payload",
        lambda *_args, **_kwargs: {"status": "passed", "phase": "confidence"},
    )

    with pytest.raises(RuntimeError, match="exact v33 training contract"):
        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
            payload,
            cfg,
            checkpoint_path=checkpoint,
        )


def test_v51_formal_ref_evaluator_binds_exact_promotion_report(monkeypatch):
    from tools import (
        run_stageb_confidence_adapter_candidate_split_independent_deployed_router_probe_evaluation as promotion,
    )

    cfg = SLConfig.fromfile(str(promotion.FORMAL_CONFIG))
    sentinel = {"status": "verified"}
    monkeypatch.setattr(promotion, "verify_admission_report", lambda path: sentinel)
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)

    ref_eval._bind_dense_duty_formal_probe_admission(cfg)

    assert cfg.stage_b_dense_duty_confidence_probe_admission_contract == (
        "u400_word_veto_candidate_split_independent_deployed_router_"
        "confidence_strict1607_v51"
    )
    assert cfg.stage_b_dense_duty_confidence_probe_admission_audit == sentinel
