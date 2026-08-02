from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    run_stageb_confidence_adapter_candidate_split_strong_boundary_routing_probe_evaluation as promotion,
)
from util.slconfig import SLConfig


V50_CONFIG = (
    combined_eval._CANDIDATE_SPLIT_STRONG_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG
)


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V50_CONFIG),
        output_dir=str(tmp_path / "v50-strict1607"),
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


def test_v50_probe_config_is_admitted_by_both_evaluators(tmp_path):
    cfg = SLConfig.fromfile(str(V50_CONFIG))
    assert ref_eval._validate_v50_split_strong_boundary_routing_config(cfg) is True
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.1),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 1.0),
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
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_joint_clip_v3",
        ),
        (
            "stage_b_v21_token_edit_query_scope",
            "target_iou_union_detached_final_confidence_base_argmax_v2",
        ),
    ),
)
def test_v50_training_and_eval_validators_reject_drift(
    tmp_path, field, value
):
    cfg = SLConfig.fromfile(str(V50_CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError):
        training_main._validate_stage_b_dense_duty_args(cfg)
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v50_formal_ref_evaluator_binds_exact_promotion_report(monkeypatch):
    cfg = SLConfig.fromfile(str(promotion.FORMAL_CONFIG))
    sentinel = {"status": "verified"}
    monkeypatch.setattr(promotion, "verify_admission_report", lambda path: sentinel)
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)

    ref_eval._bind_dense_duty_formal_probe_admission(cfg)

    assert cfg.stage_b_dense_duty_confidence_probe_admission_audit == sentinel
