from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
V44_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_heads_probe_u0400_20260801.py"
)
V46_BASE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_positive_tail_20260801.py"
)
V46_CONFIG = combined_eval._CANDIDATE_SPLIT_POSITIVE_TAIL_CONFIDENCE_U0400_CONFIG


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V46_CONFIG),
        output_dir=str(tmp_path / "v46-strict1607"),
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


def test_v46_is_fresh_v44_plus_only_positive_trust_tail_behavior():
    v44 = SLConfig.fromfile(str(V44_CONFIG))._cfg_dict.to_dict()
    v46 = SLConfig.fromfile(str(V46_CONFIG))._cfg_dict.to_dict()

    changed = {
        key for key in set(v44) | set(v46) if v44.get(key) != v46.get(key)
    }
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
        "stage_b_v15_tail_queue_positive_trust_reduction_contract",
    }
    assert v46["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_positive_tail_v46"
    )
    assert v46["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_absolute_v2"
    )
    assert v46["stage_b_dense_duty_deployed_veto_routing_weight"] == 0.1
    assert v46[
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract"
    ] == "balanced_mean_v1"
    assert v46[
        "stage_b_v15_tail_queue_positive_trust_reduction_contract"
    ] == "top_quarter_cvar_v2"
    assert v46["stage_b_dense_duty_positive_trust_contract"] == (
        "absolute_global_confidence_logit_v2"
    )


def test_v46_uses_isolated_trace_and_admission_paths():
    v44 = SLConfig.fromfile(str(V44_CONFIG))
    v46_probe = SLConfig.fromfile(str(V46_CONFIG))
    v46_base = SLConfig.fromfile(str(V46_BASE_CONFIG))

    assert "candidate_split_positive_tail" in str(
        v46_probe.stage_b_dense_duty_trace_audit_path
    )
    assert v46_probe.stage_b_dense_duty_trace_audit_path != (
        v44.stage_b_dense_duty_trace_audit_path
    )
    assert v46_probe.stage_b_dense_duty_confidence_probe_admission_contract == (
        "disabled_for_probe_v1"
    )
    assert v46_probe.stage_b_dense_duty_confidence_probe_admission_report == ""
    assert v46_base.stage_b_dense_duty_confidence_probe_admission_contract == (
        "u400_word_veto_candidate_split_positive_tail_"
        "confidence_strict1607_v46"
    )
    assert "candidate_split_positive_tail" in str(
        v46_base.stage_b_dense_duty_confidence_probe_admission_report
    )


def test_v46_probe_config_is_admitted_by_ref_and_combined_validators(tmp_path):
    cfg = SLConfig.fromfile(str(V46_CONFIG))

    assert ref_eval._validate_v46_split_positive_tail_config(cfg) is True
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


V46_ARCHITECTURE_DRIFT = (
    (
        "stage_b_dense_duty_confidence_revision",
        "word_veto_candidate_asymmetric_deployed_routing_v43",
    ),
    (
        "stage_b_dense_duty_confidence_gate_gradient_contract",
        "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13",
    ),
    ("stage_b_dense_duty_deployed_veto_routing_weight", 1.0),
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
    ("stage_b_dense_duty_confidence_veto_gate_offset", 0.02),
    (
        "stage_b_dense_duty_confidence_head_gradient_contract",
        "split_token_veto_global_absolute_joint_clip_v3",
    ),
    (
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
        "balanced_top_quarter_cvar_v2",
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


@pytest.mark.parametrize(("field", "value"), V46_ARCHITECTURE_DRIFT)
def test_ref_evaluator_v46_architecture_validator_fails_closed(field, value):
    cfg = SLConfig.fromfile(str(V46_CONFIG))
    setattr(cfg, field, value)

    if field == "stage_b_dense_duty_confidence_revision":
        assert ref_eval._validate_v46_split_positive_tail_config(cfg) is False
    else:
        with pytest.raises(RuntimeError, match="v46 split-positive-tail"):
            ref_eval._validate_v46_split_positive_tail_config(cfg)


@pytest.mark.parametrize(("field", "value"), V46_ARCHITECTURE_DRIFT)
def test_v46_combined_eval_rejects_architecture_drift(
    tmp_path, field, value
):
    cfg = SLConfig.fromfile(str(V46_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stage_b_dense_duty_deployed_veto_routing_weight", 1.0),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "balanced_top_quarter_cvar_v2",
        ),
        (
            "stage_b_v15_tail_queue_positive_trust_reduction_contract",
            "mean_v1",
        ),
    ),
)
def test_training_validator_rejects_v46_objective_drift_before_artifacts(
    field, value
):
    cfg = SLConfig.fromfile(str(V46_CONFIG))
    setattr(cfg, field, value)

    with pytest.raises(RuntimeError, match="v46 positive-tail alignment"):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v46_formal_admission_is_intentionally_unbound():
    probe = SLConfig.fromfile(str(V46_CONFIG))
    formal = SLConfig.fromfile(str(V46_BASE_CONFIG))

    assert training_main._bind_stage_b_confidence_probe_admission(probe) is None
    assert training_main._bind_stage_b_confidence_probe_admission(formal) is None
