from pathlib import Path
from types import SimpleNamespace

from tools import (
    audit_stageb_confidence_adapter_deployment_owned_global_stable_fpr95_active_set_probe_health
    as health,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    run_stageb_confidence_adapter_deployment_owned_global_stable_fpr95_active_set_highmem_formal
    as formal,
)
from tools import (
    run_stageb_confidence_adapter_deployment_owned_global_stable_fpr95_active_set_probe_evaluation
    as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_deployment_owned_global_stable_fpr95_active_set_probe_u0400
    as training,
)
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(training.CONFIG),
        output_dir=str(tmp_path / "v58-strict1607"),
        ckpts=["deployment-owned-stable-active-u400.pth"],
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


def test_v58_config_and_combined_evaluator_are_bound(tmp_path):
    cfg = SLConfig.fromfile(str(training.CONFIG))
    assert ref_eval._validate_v58_deployment_owned_stable_fpr95_active_set_config(cfg)
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )
    assert cfg.epochs == 2
    assert "output_dir" not in cfg
    assert cfg.stage_b_v15_tail_queue_negative_reduction_contract == (
        "exact_fpr95_active_set_all_count_mean_v2"
    )
    assert cfg.stage_b_v14_local_absolute_weight == 0.0
    assert cfg.stage_b_dense_duty_deployed_global_absolute_weight == 0.0


def test_v58_controllers_bind_the_single_treatment(monkeypatch):
    assert health.TRAINING_CONTRACT_SCHEMA == "pivot.stageb.dense_duty_training_contract/v40"
    assert health.v56._CORE.EXPECTED_CONTRACT_VALUES[
        "stage_b_v15_tail_queue_negative_reduction_contract"
    ] == "exact_fpr95_active_set_all_count_mean_v2"
    assert training.UPDATES == 400
    assert "--resume" not in training.command("start")
    assert evaluation._CORE._load_health_audit() is health.audit
    assert evaluation._formal_main_admission_is_wired(REPO_ROOT / "main.py")
    assert formal.UPDATES == 4412
    assert "--resume" not in formal.command("start")
    sentinel = {"formal_training_admitted": True}
    monkeypatch.setattr(evaluation, "verify_admission_report", lambda: sentinel)
    assert formal.verify_probe_admission() is sentinel


def test_v58_postflight_replaces_v56_result_label(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "_BASE_V56_POSTFLIGHT",
        lambda *_args, **_kwargs: {
            "contracts": {
                "v56_deployment_owned_global_representation_v38": True,
                "candidate_head_is_frozen_diagnostic_only": True,
                "all_mean_v1": True,
            }
        },
    )
    contracts = evaluation._v58_postflight({})["contracts"]
    assert contracts["v58_deployment_owned_stable_fpr95_active_set_v40"] is True
    assert contracts["fpr95_negative_gradients_are_exactly_active_set_only"] is True
    assert contracts["active_set_normalization_uses_all_valid_tn_count"] is True
    assert contracts["exact_fpr95_active_set_all_count_mean_v2"] is True
    assert "all_mean_v1" not in contracts
    assert "v56_deployment_owned_global_representation_v38" not in contracts
