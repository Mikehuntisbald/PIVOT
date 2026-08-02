from pathlib import Path

import pytest

from main import _bind_stage_b_confidence_probe_admission
from tools import (
    run_stageb_confidence_adapter_candidate_gate_zero_offset_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_gate_zero_offset_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_gate_zero_offset_"
    "probe_u0400_20260801.py"
)
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_gate_zero_offset_20260801.py"
)


def test_gate_zero_offset_config_and_training_contract_are_bound():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert cfg.stage_b_v15_tail_queue_positive_gradient_contract == (
        "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
    )
    assert cfg.stage_b_dense_duty_confidence_veto_gate_offset == 0.0
    assert cfg.stage_b_dense_duty_confidence_veto_gate_scale == 0.03
    values = probe._BASE._formal_current_args()
    contract = build_training_contract(values)
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v21"
    assert contract["values"][
        "stage_b_dense_duty_confidence_veto_gate_offset"
    ] == 0.0


def test_gate_zero_offset_probe_and_promotion_are_isolated_and_fixed():
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


def test_gate_zero_offset_probe_disables_formal_admission():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert _bind_stage_b_confidence_probe_admission(cfg) is None


def test_gate_zero_offset_formal_binding_fails_closed_until_report_exists():
    cfg = SLConfig.fromfile(str(FORMAL_CONFIG))
    report = Path(cfg.stage_b_dense_duty_confidence_probe_admission_report)
    if report.exists():
        pytest.skip("a completed v39 probe report is already present")
    with pytest.raises(FileNotFoundError):
        _bind_stage_b_confidence_probe_admission(cfg)
