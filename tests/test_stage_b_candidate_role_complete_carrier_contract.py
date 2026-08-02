from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from main import _bind_stage_b_confidence_probe_admission
from tools import (
    run_stageb_confidence_adapter_candidate_hardest_edit_probe_u0400 as v40_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_role_complete_carrier_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_role_complete_carrier_probe_u0400 as probe,
)
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_role_complete_carrier_"
    "probe_u0400_20260801.py"
)
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_role_complete_carrier_"
    "20260801.py"
)
SCOPE = "target_iou_union_detached_role_complete_confidence_base_argmax_v3"


def test_role_complete_config_gets_scope_bound_v23_training_contract():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert cfg.stage_b_v21_token_edit_query_scope == SCOPE
    assert cfg.stage_b_v21_token_objective == "edit_bce"
    assert cfg.stage_b_dense_duty_confidence_revision == (
        "word_veto_candidate_asymmetric_confidence_v32"
    )
    assert cfg.stage_b_v15_tail_queue_positive_gradient_contract == (
        "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
    )
    assert cfg.stage_b_dense_duty_confidence_veto_gate_offset == 0.0

    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v23"
    assert contract["values"]["stage_b_v21_token_edit_query_scope"] == SCOPE


def test_v40_contract_remains_v22():
    contract = build_training_contract(v40_probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v22"
    assert contract["values"]["stage_b_v21_token_edit_query_scope"] == (
        "target_iou_union_detached_final_confidence_base_argmax_v2"
    )


def test_role_complete_scope_is_restricted_to_exact_v41_surface():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    cfg.stage_b_v15_tail_queue_positive_gradient_contract = (
        "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5"
    )
    with pytest.raises(RuntimeError, match="exact v41"):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_role_complete_probe_and_promotion_are_isolated_and_fixed():
    assert probe.UPDATES == 400
    assert evaluation._BASE.EXPECTED_UPDATES == 400
    assert evaluation.CONFIG.resolve(strict=True) == PROBE_CONFIG.resolve(strict=True)
    assert evaluation.FORMAL_CONFIG.resolve(strict=True) == FORMAL_CONFIG.resolve(
        strict=True
    )
    promotion = evaluation._BASE._validate_formal_config_promotion()
    assert promotion["all_other_config_values_equal"] is True
    assert promotion["allowed_overrides"][
        "stage_b_dense_duty_confidence_expected_optimizer_updates"
    ] == {"probe": 400, "formal": 4412}


def test_role_complete_probe_disables_formal_admission():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert _bind_stage_b_confidence_probe_admission(cfg) is None


def test_role_complete_formal_binding_fails_closed_until_report_exists():
    cfg = SLConfig.fromfile(str(FORMAL_CONFIG))
    report = Path(cfg.stage_b_dense_duty_confidence_probe_admission_report)
    if report.exists():
        pytest.skip("a completed v41 probe report is already present")
    with pytest.raises(FileNotFoundError):
        _bind_stage_b_confidence_probe_admission(cfg)


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(PROBE_CONFIG),
        output_dir=str(tmp_path / "strict1607"),
        ckpts=["v41-u400-checkpoint.pth"],
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


def test_role_complete_strict1607_diagnostic_config_is_allowlisted(tmp_path):
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


def test_role_complete_strict1607_diagnostic_rejects_scope_drift(tmp_path):
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    cfg.stage_b_v21_token_edit_query_scope = (
        "target_iou_union_detached_final_confidence_base_argmax_v2"
    )
    with pytest.raises(ValueError, match="role-complete base-logit carrier scope"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _diagnostic_args(tmp_path), cfg
        )
