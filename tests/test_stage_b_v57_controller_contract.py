from pathlib import Path
from types import SimpleNamespace

from tools import (
    audit_stageb_confidence_adapter_deployed_global_balanced_absolute_probe_health
    as health,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    run_stageb_confidence_adapter_deployed_global_balanced_absolute_highmem_formal
    as formal,
)
from tools import (
    run_stageb_confidence_adapter_deployed_global_balanced_absolute_probe_evaluation
    as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_deployed_global_balanced_absolute_probe_u0400
    as training,
)
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _diagnostic_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(training.CONFIG),
        output_dir=str(tmp_path / "v57-strict1607"),
        ckpts=["deployed-global-balanced-absolute-u400.pth"],
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


def test_v57_config_and_combined_evaluator_are_bound(tmp_path):
    cfg = SLConfig.fromfile(str(training.CONFIG))
    assert ref_eval._validate_v57_deployed_global_balanced_absolute_config(cfg)
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _diagnostic_args(tmp_path), cfg
    )
    assert cfg.epochs == 2
    assert "output_dir" not in cfg


def test_v57_health_and_controllers_bind_the_single_treatment(monkeypatch):
    assert health.TRAINING_CONTRACT_SCHEMA == "pivot.stageb.dense_duty_training_contract/v39"
    assert health.v56._CORE.EXPECTED_CONTRACT_VALUES[
        "stage_b_dense_duty_deployed_global_absolute_weight"
    ] == 1.0
    assert health.v56._CORE.EXPECTED_CONTRACT_VALUES[
        "stage_b_dense_duty_deployed_global_absolute_gamma"
    ] == 1.0
    assert training.UPDATES == 400
    assert "--resume" not in training.command("start")
    assert evaluation._formal_main_admission_is_wired(REPO_ROOT / "main.py")
    assert formal.UPDATES == 4412
    assert "--resume" not in formal.command("start")
    sentinel = {"formal_training_admitted": True}
    monkeypatch.setattr(evaluation, "verify_admission_report", lambda: sentinel)
    assert formal.verify_probe_admission() is sentinel


def test_v57_postflight_replaces_v56_result_label(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "_BASE_V56_POSTFLIGHT",
        lambda *_args, **_kwargs: {
            "contracts": {
                "v56_deployment_owned_global_representation_v38": True,
                "candidate_head_is_frozen_diagnostic_only": True,
            }
        },
    )
    contracts = evaluation._v57_postflight({})["contracts"]
    assert contracts["v57_deployed_global_balanced_absolute_v39"] is True
    assert contracts["balanced_absolute_loss_uses_true_deployed_global_logits"] is True
    assert "v56_deployment_owned_global_representation_v38" not in contracts
