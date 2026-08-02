from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    run_stageb_confidence_adapter_candidate_deployed_routing_probe_u0400 as v43_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_heads_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_heads_probe_u0400_20260801.py"
)


def _strict1607_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partial_dense_duty_rank_diagnostic=False,
        partial_dense_duty_confidence_diagnostic=True,
        config=str(PROBE_CONFIG),
        output_dir=str(tmp_path / "split-heads-strict1607"),
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


def test_split_heads_is_an_exact_single_delta_over_v43():
    v43 = SLConfig.fromfile(str(v43_probe.CONFIG))._cfg_dict.to_dict()
    split = SLConfig.fromfile(str(PROBE_CONFIG))._cfg_dict.to_dict()

    changed = {
        key for key in set(v43) | set(split) if v43.get(key) != split.get(key)
    }
    assert changed == {
        "stage_b_dense_duty_confidence_head_gradient_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }
    # The extra two fields are provenance-only: the new code source closure
    # needs its own immutable direct-trace receipt. All train/eval behavior has
    # exactly one delta, the split head-gradient contract.
    assert split["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_asymmetric_deployed_routing_v43"
    )
    assert split["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_absolute_v2"
    )
    assert split["stage_b_v11_trainable_params_min"] == v43[
        "stage_b_v11_trainable_params_min"
    ]
    assert split["stage_b_v11_trainable_params_max"] == v43[
        "stage_b_v11_trainable_params_max"
    ]


def test_split_heads_are_resume_bound_without_changing_v43_contract():
    split = build_training_contract(probe._BASE._formal_current_args())
    assert split["schema"] == "pivot.stageb.dense_duty_training_contract/v26"
    assert split["values"][
        "stage_b_dense_duty_confidence_head_gradient_contract"
    ] == "split_token_veto_global_absolute_v2"

    v43 = build_training_contract(v43_probe._BASE._formal_current_args())
    assert v43["schema"] == "pivot.stageb.dense_duty_training_contract/v25"
    assert (
        "stage_b_dense_duty_confidence_head_gradient_contract"
        not in v43["values"]
    )


def test_split_heads_validation_fails_closed_outside_v43_surface():
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    cfg.stage_b_dense_duty_confidence_revision = (
        "word_veto_candidate_asymmetric_confidence_v32"
    )
    with pytest.raises(RuntimeError, match="split token-veto/global-absolute"):
        training_main._validate_stage_b_dense_duty_args(cfg)


@pytest.mark.parametrize("clip_max_norm", [0.0, float("nan")])
def test_split_heads_validation_requires_finite_positive_clip(clip_max_norm):
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    cfg.clip_max_norm = clip_max_norm
    with pytest.raises(RuntimeError, match="finite positive clip_max_norm"):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_split_heads_probe_is_fresh_isolated_and_uses_u6551_source():
    assert probe.UPDATES == 400
    assert probe.CONFIG.resolve(strict=True) == PROBE_CONFIG.resolve(strict=True)
    assert "candidate_split_heads" in str(probe.OUTPUT)
    assert probe.OUTPUT != v43_probe.OUTPUT
    state = probe.inspect()
    assert state["status"] in {"fresh", "terminal", "invalid"}
    if state["status"] != "invalid":
        assert state["action"] == (
            "start" if state["status"] == "fresh" else "complete"
        )
    else:
        reason = state["reason"].lower()
        assert "source closure" in reason or "source_closure" in reason
    if state["status"] == "terminal":
        assert state["updates"] == 400
        assert state["checkpoint_audit"]["status"] == "passed"

    command = probe.command("start")
    assert "--resume" not in command
    assert command[command.index("--pretrain_model_path") + 1] == str(
        probe._BASE.RANK_SOURCE
    )
    assert command[command.index("--max_train_iters") + 1] == "400"

    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    assert training_main._bind_stage_b_confidence_probe_admission(cfg) is None


def test_split_heads_strict1607_config_is_allowlisted(tmp_path):
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
        _strict1607_args(tmp_path), cfg
    )


def test_split_heads_strict1607_rejects_head_contract_drift(tmp_path):
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    cfg.stage_b_dense_duty_confidence_head_gradient_contract = (
        "shared_token_veto_global_absolute_v1"
    )
    with pytest.raises(ValueError, match="head-gradient contract"):
        combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
            _strict1607_args(tmp_path), cfg
        )
