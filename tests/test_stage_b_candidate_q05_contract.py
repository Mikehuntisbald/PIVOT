from pathlib import Path

import pytest

from main import _bind_stage_b_confidence_probe_admission
from tools import (
    run_stageb_confidence_adapter_candidate_q05_probe_evaluation as evaluation,
)
from tools import run_stageb_confidence_adapter_candidate_q05_probe_u0400 as probe
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import (
    _POSITIVE_TAIL_GRADIENT_RESUME_CONTRACT_KEYS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_q05_"
    "probe_u0400_20260801.py"
)
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_q05_20260801.py"
)


def test_q05_probe_preserves_v32_surface_and_changes_only_tail_gradient():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert cfg.stage_b_dense_duty_confidence_revision == (
        "word_veto_candidate_asymmetric_confidence_v32"
    )
    assert cfg.stage_b_dense_duty_confidence_pool_feature_contract == (
        "detached_candidate_absolute_raw_patch_asymmetric_veto_logits_v8"
    )
    assert cfg.stage_b_v15_tail_queue_positive_gradient_contract == (
        "exact_batch_lower_tail_st_v2"
    )
    assert cfg.stage_b_v15_tail_queue_positive_trust_weight == 1.0
    assert cfg.stage_b_v14_tail_queue_weight == 1.0
    assert cfg.stage_b_dense_duty_confidence_expected_optimizer_updates == 400


def test_q05_gradient_contract_is_bound_by_strict_resume():
    assert "stage_b_v15_tail_queue_positive_gradient_contract" in (
        _POSITIVE_TAIL_GRADIENT_RESUME_CONTRACT_KEYS
    )


def test_q05_probe_and_promotion_share_the_fixed_u400_contract():
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


def test_q05_probe_disables_formal_admission_but_matches_main_binding():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert _bind_stage_b_confidence_probe_admission(cfg) is None
    assert cfg.stage_b_dense_duty_confidence_probe_admission_contract == (
        "disabled_for_probe_v1"
    )


def test_q05_formal_binding_fails_closed_until_report_exists():
    cfg = SLConfig.fromfile(str(FORMAL_CONFIG))
    report = Path(cfg.stage_b_dense_duty_confidence_probe_admission_report)
    if report.exists():
        pytest.skip("a completed v34 probe report is already present")
    with pytest.raises(FileNotFoundError):
        _bind_stage_b_confidence_probe_admission(cfg)
