from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util.slconfig import SLConfig


V43_CONFIG = combined_eval._CANDIDATE_DEPLOYED_ROUTING_CONFIDENCE_U0400_CONFIG


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V43_CONFIG),
        output_dir=str(tmp_path / "v43-strict1607"),
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


def test_v43_probe_config_is_admitted_without_immutable_checkpoint_identity(
    tmp_path,
):
    cfg = SLConfig.fromfile(str(V43_CONFIG))

    assert ref_eval._validate_v43_deployed_routing_config(cfg) is True
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )

    assert not hasattr(ref_eval, "_V43_IMMUTABLE_ARCHIVED_SNAPSHOTS")
    assert not hasattr(combined_eval, "_V43_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS")


@pytest.mark.parametrize(
    "field,value",
    (
        (
            "stage_b_dense_duty_confidence_revision",
            "word_veto_candidate_asymmetric_confidence_v32",
        ),
        (
            "stage_b_dense_duty_confidence_gate_gradient_contract",
            "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13",
        ),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.2),
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
    ),
)
def test_v43_combined_eval_rejects_architecture_drift(
    tmp_path, field, value
):
    cfg = SLConfig.fromfile(str(V43_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


@pytest.mark.parametrize(
    "field,value",
    (
        (
            "stage_b_dense_duty_confidence_gate_gradient_contract",
            "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13",
        ),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.0),
        ("stage_b_dense_duty_deployed_veto_positive_max", 0.0),
        ("stage_b_dense_duty_deployed_veto_tn_min", 1.0),
        (
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "tn_only_positive_detached_v2",
        ),
        (
            "stage_b_v21_token_edit_query_scope",
            "target_iou_union_detached_final_confidence_base_argmax_v2",
        ),
    ),
)
def test_ref_evaluator_v43_architecture_validator_fails_closed(field, value):
    cfg = SLConfig.fromfile(str(V43_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(RuntimeError, match="v43 deployed-routing"):
        ref_eval._validate_v43_deployed_routing_config(cfg)
