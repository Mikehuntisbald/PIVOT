from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from tools import (
    run_stageb_confidence_adapter_candidate_split_heads_probe_u0400 as v44_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_tail_aligned_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_tail_aligned_probe_u0400_20260801.py"
)


def test_v45_is_a_bounded_tail_alignment_delta_over_v44():
    v44 = SLConfig.fromfile(str(v44_probe.CONFIG))._cfg_dict.to_dict()
    v45 = SLConfig.fromfile(str(PROBE_CONFIG))._cfg_dict.to_dict()

    changed = {key for key in set(v44) | set(v45) if v44.get(key) != v45.get(key)}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_confidence_head_gradient_contract",
        "stage_b_dense_duty_deployed_veto_routing_weight",
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
        "stage_b_v15_tail_queue_positive_trust_reduction_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
    }
    assert v45["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_tail_aligned_v45"
    )
    assert v45["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_absolute_joint_clip_v3"
    )
    assert v45["stage_b_dense_duty_deployed_veto_routing_weight"] == 1.0
    assert v45["stage_b_dense_duty_positive_trust_contract"] == (
        "absolute_global_confidence_logit_v2"
    )


def test_v45_training_contract_binds_both_head_reductions():
    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v27"
    values = contract["values"]
    assert values["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_absolute_joint_clip_v3"
    )
    assert values[
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract"
    ] == "balanced_top_quarter_cvar_v2"
    assert values[
        "stage_b_v15_tail_queue_positive_trust_reduction_contract"
    ] == "top_quarter_cvar_v2"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_v2",
            "split token-veto/global-absolute",
        ),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.1, "v45 tail alignment"),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "balanced_mean_v1",
            "v45 tail alignment",
        ),
        (
            "stage_b_v15_tail_queue_positive_trust_reduction_contract",
            "mean_v1",
            "v45 tail alignment",
        ),
        (
            "stage_b_dense_duty_positive_trust_contract",
            "pool_residual_v1",
            "v45 tail alignment",
        ),
    ),
)
def test_v45_validation_fails_closed_on_contract_drift(field, value, message):
    cfg = SLConfig.fromfile(str(PROBE_CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError, match=message):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v45_probe_is_fresh_isolated_and_uses_u6551_source():
    assert probe.UPDATES == 400
    assert probe.CONFIG.resolve(strict=True) == PROBE_CONFIG.resolve(strict=True)
    assert probe.OUTPUT != v44_probe.OUTPUT
    assert "split_tail_aligned" in str(probe.OUTPUT)
    assert probe.inspect() == {"status": "fresh", "action": "start"}

    command = probe.command("start")
    assert "--resume" not in command
    assert command[command.index("--pretrain_model_path") + 1] == str(
        probe._BASE.RANK_SOURCE
    )
    assert command[command.index("--max_train_iters") + 1] == "400"

    current = probe._BASE._formal_current_args()
    current["resume"] = None
    current["pretrain_model_path"] = str(probe._BASE.RANK_SOURCE)
    training_main._validate_stage_b_dense_duty_args(SimpleNamespace(**current))
    probe.validate_inputs()
