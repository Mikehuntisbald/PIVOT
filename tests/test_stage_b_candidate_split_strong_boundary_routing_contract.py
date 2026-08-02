from __future__ import annotations

import math
from pathlib import Path

import pytest

import main as training_main
from tools import (
    audit_stageb_confidence_adapter_candidate_split_strong_boundary_routing_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_boundary_routing_probe_u0400 as v47_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_strong_boundary_routing_highmem_formal as formal,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_strong_boundary_routing_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_strong_boundary_routing_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return (
        SLConfig.fromfile(str(v47_probe.CONFIG))._cfg_dict.to_dict(),
        SLConfig.fromfile(str(probe.CONFIG))._cfg_dict.to_dict(),
    )


def test_v50_is_v47_plus_one_behavior_delta_and_sealed_defaults():
    v47, v50 = _configs()
    changed = {key for key in set(v47) | set(v50) if v47.get(key) != v50.get(key)}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_deployed_veto_routing_weight",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
        "stage_b_v15_tail_queue_negative_reduction_contract",
        "stage_b_v21_token_edit_query_scope",
    }
    assert v50["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_strong_boundary_routing_v50"
    )
    assert v50["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_global_absolute_v2"
    )
    assert v50["stage_b_dense_duty_deployed_veto_routing_weight"] == 0.25
    assert v50[
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract"
    ] == "balanced_top_quarter_cvar_v2"
    assert v50[
        "stage_b_v15_tail_queue_positive_trust_reduction_contract"
    ] == "top_quarter_cvar_v2"
    assert v50["stage_b_v15_tail_queue_negative_reduction_contract"] == (
        "all_mean_v1"
    )
    assert v50["stage_b_v21_token_edit_query_scope"] == "target_iou_v1"
    receipt = Path(v50["stage_b_dense_duty_trace_audit_path"])
    assert receipt.is_file()
    assert len(v50["stage_b_dense_duty_trace_audit_sha256"]) == 64
    assert v50["stage_b_dense_duty_trace_audit_sha256"] != "0" * 64


def test_v50_training_contract_is_exact_v32_and_seals_defaults():
    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v32"
    values = contract["values"]
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_strong_boundary_routing_v50"
    )
    assert values["stage_b_dense_duty_deployed_veto_routing_weight"] == 0.25
    assert values["stage_b_v15_tail_queue_negative_reduction_contract"] == (
        "all_mean_v1"
    )
    assert values["stage_b_v21_token_edit_query_scope"] == "target_iou_v1"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_joint_clip_v3",
        ),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.1),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 1.0),
        (
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
            "balanced_mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_positive_trust_reduction_contract",
            "mean_v1",
        ),
        (
            "stage_b_v15_tail_queue_negative_reduction_contract",
            "exact_fpr95_active_set_mean_v1",
        ),
        (
            "stage_b_v21_token_edit_query_scope",
            "target_iou_union_detached_final_confidence_base_argmax_v2",
        ),
    ),
)
def test_v50_training_validation_fails_closed_on_contract_drift(field, value):
    cfg = SLConfig.fromfile(str(probe.CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v50_probe_is_fresh_from_rank_source_only():
    assert probe.UPDATES == 400
    assert probe.OUTPUT != v47_probe.OUTPUT
    observed = probe.inspect()
    if probe.CHECKPOINT.exists():
        assert observed["status"] == "terminal"
        assert observed["action"] == "complete"
        assert observed["updates"] == 400
    else:
        assert observed == {"status": "fresh", "action": "start"}
    command = probe.command("start")
    assert "--resume" not in command
    assert command[command.index("--pretrain_model_path") + 1] == str(
        probe._BASE.RANK_SOURCE
    )
    assert command[command.index("--max_train_iters") + 1] == "400"
    probe.validate_inputs()


def test_v50_health_preserves_two_independent_point_one_clips():
    runtime = {
        "optimizer_step_boundaries": 400,
        "successful_optimizer_steps": 400,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "nonfinite_token_veto_gradient_boundaries": 0,
        "nonfinite_global_absolute_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "zero_token_veto_gradient_successful_steps": 0,
        "zero_global_absolute_gradient_successful_steps": 0,
        "last_token_veto_grad_norm_preclip": 20.0,
        "max_token_veto_grad_norm_preclip": 30.0,
        "last_global_absolute_grad_norm_preclip": 22.0,
        "max_global_absolute_grad_norm_preclip": 31.0,
        "min_amp_scale": 256.0,
        "last_amp_scale": 256.0,
    }
    trajectory = {
        "train_grad_norm_dense_duty_active_preclip": 30.0,
        "train_grad_tensor_count_dense_duty_active": 67.0,
        "train_grad_norm_dense_duty_token_veto_preclip": 20.0,
        "train_grad_tensor_count_dense_duty_token_veto": 21.0,
        "train_grad_norm_dense_duty_global_absolute_preclip": 22.0,
        "train_grad_tensor_count_dense_duty_global_absolute": 46.0,
        "train_grad_norm_dense_duty_token_veto_postclip": 0.1,
        "train_grad_norm_dense_duty_global_absolute_postclip": 0.1,
        "train_grad_norm_dense_duty_active_postclip": math.sqrt(0.02),
        "train_amp_step_skipped": 0.0,
    }
    checks = health._health_checks(runtime, trajectory)
    assert all(check["passed"] for check in checks.values())


def _write_admission_binder(path: Path, *, routing_weight: float) -> None:
    path.write_text(
        "def _bind_stage_b_confidence_probe_admission(args):\n"
        "    revision = getattr(args, 'revision', '')\n"
        "    if (\n"
        '        revision == "word_veto_candidate_split_strong_boundary_routing_v50"\n'
        "        and str(getattr(args, "
        "'stage_b_dense_duty_confidence_head_gradient_contract', '')) "
        '== "split_token_veto_global_absolute_v2"\n'
        "        and str(getattr(args, "
        "'stage_b_dense_duty_deployed_veto_routing_reduction_contract', '')) "
        '== "balanced_top_quarter_cvar_v2"\n'
        "        and str(getattr(args, "
        "'stage_b_v15_tail_queue_positive_trust_reduction_contract', '')) "
        '== "top_quarter_cvar_v2"\n'
        "        and str(getattr(args, "
        "'stage_b_v15_tail_queue_negative_reduction_contract', '')) "
        '== "all_mean_v1"\n'
        "        and str(getattr(args, "
        "'stage_b_dense_duty_positive_trust_contract', '')) "
        '== "absolute_global_confidence_logit_v2"\n'
        "        and str(getattr(args, 'stage_b_v21_token_edit_query_scope', '')) "
        '== "target_iou_v1"\n'
        "        and str(getattr(args, "
        "'stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract', '')) "
        '== "bidirectional_v1"\n'
        "        and float(getattr(args, "
        "'stage_b_dense_duty_deployed_veto_positive_max', -1.0)) == 0.1\n"
        "        and float(getattr(args, "
        "'stage_b_dense_duty_deployed_veto_tn_min', -1.0)) == 0.9\n"
        "        and float(getattr(args, "
        "'stage_b_dense_duty_confidence_veto_gate_offset', -1.0)) == 0.0\n"
        "        and float(getattr(args, "
        "'stage_b_dense_duty_deployed_veto_routing_weight', 0.0)) "
        f"== {routing_weight!r}\n"
        "    ):\n"
        '        formal_contract = "u400_word_veto_candidate_split_strong_'
        'boundary_routing_confidence_strict1607_v50"\n'
        "        from tools import "
        "run_stageb_confidence_adapter_candidate_split_strong_boundary_"
        "routing_probe_evaluation as promotion\n"
        "        return formal_contract, promotion\n"
        "    return None\n",
        encoding="utf-8",
    )


def test_v50_strict_controller_and_branch_local_ast_are_fail_closed(tmp_path):
    command = evaluation.build_command()
    assert command[command.index("--config") + 1] == str(evaluation.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(evaluation.CHECKPOINT)
    assert command[command.index("--batch_size") + 1] == "16"
    assert command[command.index("--num_workers") + 1] == "4"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--topk") + 1] == "1"
    assert "--amp" in command and "--skip_ref" in command

    good = tmp_path / "good_main.py"
    bad = tmp_path / "bad_main.py"
    _write_admission_binder(good, routing_weight=0.25)
    _write_admission_binder(bad, routing_weight=1.0)
    assert evaluation._formal_main_admission_is_wired(REPO_ROOT / "main.py")
    assert evaluation._formal_main_admission_is_wired(good)
    assert not evaluation._formal_main_admission_is_wired(bad)


def test_v50_formal_runner_is_fresh_from_rank_source_not_probe_checkpoint():
    assert formal.UPDATES == 4_412
    assert formal.OUTPUT != probe.OUTPUT
    command = formal.command("start")
    assert "--resume" not in command
    assert command[command.index("--pretrain_model_path") + 1] == str(
        formal._BASE.RANK_SOURCE
    )
    assert command[command.index("--max_train_iters") + 1] == "4412"
    assert str(probe.CHECKPOINT) not in command
