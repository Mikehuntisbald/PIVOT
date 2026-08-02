from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
V46_CONFIG = combined_eval._CANDIDATE_SPLIT_POSITIVE_TAIL_CONFIDENCE_U0400_CONFIG
V47_BASE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_boundary_routing_20260801.py"
)
V47_CONFIG = combined_eval._CANDIDATE_SPLIT_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(V47_CONFIG),
        output_dir=str(tmp_path / "v47-strict1607"),
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


def test_v47_is_exact_single_behavior_delta_over_v46():
    v46 = SLConfig.fromfile(str(V46_CONFIG))._cfg_dict.to_dict()
    v47 = SLConfig.fromfile(str(V47_CONFIG))._cfg_dict.to_dict()
    changed = {key for key in set(v46) | set(v47) if v46.get(key) != v47.get(key)}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }


def test_v47_probe_config_is_admitted_by_both_evaluators(tmp_path):
    cfg = SLConfig.fromfile(str(V47_CONFIG))
    assert ref_eval._validate_v47_split_boundary_routing_config(cfg) is True
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
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
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_joint_clip_v3",
        ),
    ),
)
def test_v47_training_and_eval_validators_reject_drift(tmp_path, field, value):
    cfg = SLConfig.fromfile(str(V47_CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError, match="v47 boundary routing"):
        training_main._validate_stage_b_dense_duty_args(cfg)
    with pytest.raises(ValueError, match="contract failed"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )


def test_v47_formal_admission_is_intentionally_unbound():
    probe = SLConfig.fromfile(str(V47_CONFIG))
    formal = SLConfig.fromfile(str(V47_BASE_CONFIG))
    assert training_main._bind_stage_b_confidence_probe_admission(probe) is None
    assert training_main._bind_stage_b_confidence_probe_admission(formal) is None
