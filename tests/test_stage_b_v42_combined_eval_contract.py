from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util.slconfig import SLConfig


V39_TO_V41_CONFIGS = (
    combined_eval._CANDIDATE_GATE_ZERO_OFFSET_CONFIDENCE_U0400_CONFIG,
    combined_eval._CANDIDATE_HARDEST_EDIT_CONFIDENCE_U0400_CONFIG,
    combined_eval._CANDIDATE_ROLE_COMPLETE_CARRIER_CONFIDENCE_U0400_CONFIG,
)
V42_CONFIG = (
    combined_eval._CANDIDATE_TN_ONLY_CARRIER_PAIR_CONFIDENCE_U0400_CONFIG
)


def _diagnostic_args(tmp_path: Path, config: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(config),
        output_dir=str(tmp_path / config.stem),
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


def _validate(tmp_path: Path, config: Path, cfg: SLConfig) -> None:
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path, config), cfg
    )


def test_v42_strict1607_probe_is_allowlisted_with_tn_only_gradient_contract(
    tmp_path,
):
    cfg = SLConfig.fromfile(str(V42_CONFIG))
    assert cfg.stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract == (
        "tn_only_positive_detached_v2"
    )
    _validate(tmp_path, V42_CONFIG, cfg)


def test_v42_strict1607_probe_rejects_bidirectional_gradient_drift(tmp_path):
    cfg = SLConfig.fromfile(str(V42_CONFIG))
    cfg.stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract = (
        "bidirectional_v1"
    )
    with pytest.raises(ValueError, match="v42 requires.*tn_only_positive_detached_v2"):
        _validate(tmp_path, V42_CONFIG, cfg)


@pytest.mark.parametrize("config", V39_TO_V41_CONFIGS)
def test_v39_to_v41_remain_bidirectional_and_reject_v42_cross_admission(
    tmp_path, config
):
    cfg = SLConfig.fromfile(str(config))
    assert (
        getattr(
            cfg,
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "bidirectional_v1",
        )
        == "bidirectional_v1"
    )
    _validate(tmp_path, config, cfg)

    cfg.stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract = (
        "tn_only_positive_detached_v2"
    )
    with pytest.raises(ValueError, match="v39-v41 require bidirectional_v1"):
        _validate(tmp_path, config, cfg)
