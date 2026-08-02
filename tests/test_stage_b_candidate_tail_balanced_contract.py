from pathlib import Path

import pytest

from main import _bind_stage_b_confidence_probe_admission
from tools import (
    run_stageb_confidence_adapter_candidate_q05_probe_evaluation as q05_evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_tail_balanced_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_tail_balanced_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_tail_balanced_"
    "probe_u0400_20260801.py"
)
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_tail_balanced_20260801.py"
)


def test_tail_balanced_probe_preserves_v32_surface_and_adds_combined_carrier():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert cfg.stage_b_dense_duty_confidence_revision == (
        "word_veto_candidate_asymmetric_confidence_v32"
    )
    assert cfg.stage_b_dense_duty_confidence_pool_feature_contract == (
        "detached_candidate_absolute_raw_patch_asymmetric_veto_logits_v8"
    )
    assert cfg.stage_b_v15_tail_queue_positive_gradient_contract == (
        "mean_plus_exact_lower_tail_st_v3"
    )
    assert cfg.stage_b_v15_tail_queue_positive_trust_weight == 1.0
    assert cfg.stage_b_v14_tail_queue_weight == 1.0
    assert cfg.stage_b_dense_duty_confidence_expected_optimizer_updates == 400


def test_tail_balanced_probe_and_promotion_share_fixed_u400_contract():
    assert probe.UPDATES == 400
    assert evaluation._BASE.EXPECTED_UPDATES == 400
    assert evaluation.health.EXPECTED_UPDATES == 400
    assert evaluation.CONFIG.resolve(strict=True) == PROBE_CONFIG.resolve(strict=True)
    assert evaluation.FORMAL_CONFIG.resolve(strict=True) == FORMAL_CONFIG.resolve(
        strict=True
    )
    promotion = evaluation._BASE._validate_formal_config_promotion()
    assert promotion["all_other_config_values_equal"] is True
    assert promotion["allowed_overrides"][
        "stage_b_dense_duty_confidence_expected_optimizer_updates"
    ] == {"probe": 400, "formal": 4412}


def test_tail_balanced_evaluator_does_not_mutate_v34_controller_modules():
    assert evaluation._BASE is not q05_evaluation._BASE
    assert evaluation.health is not q05_evaluation.health
    assert "candidate_q05" in str(q05_evaluation._BASE.CONFIG)
    assert "candidate_tail_balanced" in str(evaluation._BASE.CONFIG)


def test_tail_balanced_training_contract_binds_gradient_contract():
    values = probe._BASE._formal_current_args()
    contract = build_training_contract(values)
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v21"
    assert contract["values"][
        "stage_b_v15_tail_queue_positive_gradient_contract"
    ] == (
        "mean_plus_exact_lower_tail_st_v3"
    )


def test_tail_balanced_probe_disables_formal_admission():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert _bind_stage_b_confidence_probe_admission(cfg) is None
    assert cfg.stage_b_dense_duty_confidence_probe_admission_contract == (
        "disabled_for_probe_v1"
    )


def test_tail_balanced_formal_binding_fails_closed_until_report_exists():
    cfg = SLConfig.fromfile(str(FORMAL_CONFIG))
    report = Path(cfg.stage_b_dense_duty_confidence_probe_admission_report)
    if report.exists():
        pytest.skip("a completed v35 probe report is already present")
    with pytest.raises(FileNotFoundError):
        _bind_stage_b_confidence_probe_admission(cfg)
