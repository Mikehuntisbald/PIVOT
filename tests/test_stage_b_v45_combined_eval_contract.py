from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util.slconfig import SLConfig


V45_CONFIG = (
    combined_eval._CANDIDATE_SPLIT_TAIL_ALIGNED_CONFIDENCE_U0400_CONFIG
)


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V45_CONFIG),
        output_dir=str(tmp_path / "v45-strict1607"),
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


def test_v45_probe_config_is_admitted_by_ref_and_combined_validators(tmp_path):
    cfg = SLConfig.fromfile(str(V45_CONFIG))

    assert ref_eval._validate_v45_split_tail_aligned_config(cfg) is True
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


V45_ARCHITECTURE_DRIFT = (
    (
        "stage_b_dense_duty_confidence_revision",
        "word_veto_candidate_asymmetric_deployed_routing_v43",
    ),
    (
        "stage_b_dense_duty_confidence_gate_gradient_contract",
        "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13",
    ),
    ("stage_b_dense_duty_deployed_veto_routing_weight", 0.1),
    ("stage_b_dense_duty_deployed_veto_positive_max", 0.2),
    ("stage_b_dense_duty_deployed_veto_tn_min", 0.8),
    (
        "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
        "tn_only_positive_detached_v2",
    ),
    (
        "stage_b_v21_token_edit_query_scope",
        "target_iou_union_detached_final_confidence_base_argmax_v2",
    ),
    (
        "stage_b_v15_tail_queue_positive_gradient_contract",
        "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5",
    ),
    (
        "stage_b_dense_duty_confidence_head_gradient_contract",
        "split_token_veto_global_absolute_v2",
    ),
    (
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
        "balanced_mean_v1",
    ),
    (
        "stage_b_v15_tail_queue_positive_trust_reduction_contract",
        "mean_v1",
    ),
    (
        "stage_b_dense_duty_positive_trust_contract",
        "net_total_confidence_delta_v1",
    ),
)


@pytest.mark.parametrize(("field", "value"), V45_ARCHITECTURE_DRIFT)
def test_v45_combined_eval_rejects_architecture_drift(
    tmp_path, field, value
):
    cfg = SLConfig.fromfile(str(V45_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


@pytest.mark.parametrize(("field", "value"), V45_ARCHITECTURE_DRIFT)
def test_ref_evaluator_v45_architecture_validator_fails_closed(field, value):
    cfg = SLConfig.fromfile(str(V45_CONFIG))
    setattr(cfg, field, value)

    if field == "stage_b_dense_duty_confidence_revision":
        assert ref_eval._validate_v45_split_tail_aligned_config(cfg) is False
    else:
        with pytest.raises(RuntimeError, match="v45 split-tail-aligned"):
            ref_eval._validate_v45_split_tail_aligned_config(cfg)
