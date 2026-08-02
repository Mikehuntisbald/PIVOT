from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import (
    audit_stageb_confidence_adapter_deployment_owned_global_probe_health as health,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    run_stageb_confidence_adapter_deployment_owned_global_highmem_formal as formal,
)
from tools import (
    run_stageb_confidence_adapter_deployment_owned_global_probe_evaluation
    as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_deployment_owned_global_probe_u0400 as training,
)
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(training.CONFIG),
        output_dir=str(tmp_path / "v56-strict1607"),
        ckpts=["deployment-owned-u400-checkpoint.pth"],
        tn_jsonl=str(combined_eval._PARTIAL_CONFIDENCE_TN_SPECS["strict1607"]["path"]),
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


def test_v56_config_transfers_ownership_and_disables_local_loss():
    cfg = SLConfig.fromfile(str(training.CONFIG))
    assert ref_eval._validate_v56_deployment_owned_global_config(cfg)
    assert cfg.stage_b_v14_local_absolute_weight == 0.0
    assert cfg.stage_b_v11_trainable_params_min == 468_164
    assert cfg.stage_b_v11_trainable_params_max == 468_164


def test_v56_is_registered_by_combined_evaluator(tmp_path):
    cfg = SLConfig.fromfile(str(training.CONFIG))
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )


def test_v56_health_seals_exact_active_and_diagnostic_surfaces():
    assert health.TRAINING_CONTRACT_SCHEMA == "pivot.stageb.dense_duty_training_contract/v38"
    assert health.EXPECTED_ACTIVE_TENSORS == 59
    assert health.EXPECTED_ACTIVE_ELEMENTS == 468_164
    assert health.EXPECTED_TOKEN_TENSORS == 21
    assert health.EXPECTED_GLOBAL_TENSORS == 38
    assert health.EXPECTED_ADAPTER_TENSORS == 53
    assert health._CORE.EXPECTED_CONTRACT_VALUES[
        "stage_b_v14_local_absolute_weight"
    ] == 0.0


def test_v56_runtime_rejects_old_v55_global_owner_count():
    runtime = {
        "clip_contract_schema": ref_eval._V56_TWO_OWNER_CLIP_CONTRACT_SCHEMA,
        "clip_contract_checked_steps": 400,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
        "clip_contract_tolerance": 1e-6,
        "clip_contract_max_norm": 0.1,
        "max_active_pre_decomposition_residual": 0.0,
        "max_active_post_decomposition_residual": 0.0,
        "max_owner_clip_residual": 0.0,
        "max_active_monotonic_residual": 0.0,
    }
    for owner, count in (("token_veto", 21), ("global_absolute", 38)):
        runtime[f"last_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"max_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"expected_{owner}_tensor_count"] = count
        runtime[f"last_observed_{owner}_tensor_count"] = count
    ref_eval._validate_v56_two_owner_runtime_audit(runtime, optimizer_updates=400)
    runtime["last_observed_global_absolute_tensor_count"] = 44
    with pytest.raises(RuntimeError, match="v56 confidence checkpoint"):
        ref_eval._validate_v56_two_owner_runtime_audit(runtime, optimizer_updates=400)


def test_v56_controllers_are_fresh_and_main_admission_is_wired(monkeypatch):
    assert training.UPDATES == 400
    assert "--resume" not in training.command("start")
    assert evaluation._formal_main_admission_is_wired(REPO_ROOT / "main.py")
    assert formal.UPDATES == 4412
    assert "--resume" not in formal.command("start")
    sentinel = {"formal_training_admitted": True}
    monkeypatch.setattr(evaluation, "verify_admission_report", lambda: sentinel)
    assert formal.verify_probe_admission() is sentinel


def test_v56_postflight_replaces_v55_representation_claims(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "_BASE_POSTFLIGHT",
        lambda *_args, **_kwargs: {
            "contracts": {
                "v55_rank_full_expression_global_independent_absolute_v37": True,
                "same_v53_v54_data_loss_update_parameter_surface": True,
            }
        },
    )
    contracts = evaluation._v56_postflight({})["contracts"]
    assert contracts["v56_deployment_owned_global_representation_v38"] is True
    assert contracts["candidate_head_is_frozen_diagnostic_only"] is True
    assert contracts["fpr95_and_positive_q05_use_true_deployed_global_logit"] is True
    assert "same_v53_v54_data_loss_update_parameter_surface" not in contracts


def test_v56_ref_evaluator_binds_formal_admission(tmp_path, monkeypatch):
    report = tmp_path / "u000400_strict1607_report.json"
    report.write_text("{}\n", encoding="utf-8")
    sentinel = {"schema": "v56-admission-sentinel"}
    monkeypatch.setattr(evaluation, "REPORT", report)
    monkeypatch.setattr(
        evaluation,
        "verify_admission_report",
        lambda path: sentinel if Path(path) == report else None,
    )
    cfg = SLConfig.fromfile(str(evaluation.FORMAL_CONFIG))
    cfg.stage_b_dense_duty_confidence_probe_admission_report = str(report)
    ref_eval._bind_dense_duty_formal_probe_admission(cfg)
    assert cfg.stage_b_dense_duty_confidence_probe_admission_audit == sentinel
