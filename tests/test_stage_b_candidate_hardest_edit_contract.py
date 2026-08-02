from pathlib import Path

import pytest

import main as training_main
from main import _bind_stage_b_confidence_probe_admission
from tools import (
    run_stageb_confidence_adapter_candidate_gate_zero_offset_probe_u0400 as v39_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_hardest_edit_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_hardest_edit_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_hardest_edit_"
    "probe_u0400_20260801.py"
)
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_hardest_edit_20260801.py"
)
SCOPE = "target_iou_union_detached_final_confidence_base_argmax_v2"


def test_hardest_edit_config_gets_a_scope_bound_v22_training_contract():
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
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v22"
    assert contract["values"]["stage_b_v21_token_edit_query_scope"] == SCOPE


def test_v39_default_scope_remains_a_v21_contract_without_new_key():
    contract = build_training_contract(v39_probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v21"
    assert "stage_b_v21_token_edit_query_scope" not in contract["values"]


def test_hardest_edit_scope_is_restricted_to_the_exact_v40_surface():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    cfg.stage_b_v15_tail_queue_positive_gradient_contract = (
        "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5"
    )
    with pytest.raises(RuntimeError, match="exact v40"):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_hardest_edit_probe_and_promotion_are_isolated_and_fixed():
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


def test_hardest_edit_probe_disables_formal_admission():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert _bind_stage_b_confidence_probe_admission(cfg) is None


def test_hardest_edit_formal_binding_fails_closed_until_report_exists():
    cfg = SLConfig.fromfile(str(FORMAL_CONFIG))
    report = Path(cfg.stage_b_dense_duty_confidence_probe_admission_report)
    if report.exists():
        pytest.skip("a completed v40 probe report is already present")
    with pytest.raises(FileNotFoundError):
        _bind_stage_b_confidence_probe_admission(cfg)
