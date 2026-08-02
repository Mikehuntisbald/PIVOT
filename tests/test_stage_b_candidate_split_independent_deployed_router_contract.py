from __future__ import annotations

import math
from pathlib import Path

import pytest

import main as training_main
from tools import (
    audit_stageb_confidence_adapter_candidate_split_independent_deployed_router_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_boundary_routing_probe_u0400 as v47_probe,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_independent_deployed_router_highmem_formal as formal,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_independent_deployed_router_probe_evaluation as evaluation,
)
from tools import (
    run_stageb_confidence_adapter_candidate_split_independent_deployed_router_probe_u0400 as probe,
)
from util.slconfig import SLConfig
from util.stage_b_dense_duty_audit import build_training_contract


REPO_ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return (
        SLConfig.fromfile(str(v47_probe.CONFIG))._cfg_dict.to_dict(),
        SLConfig.fromfile(str(probe.CONFIG))._cfg_dict.to_dict(),
    )


def test_v51_is_v47_boundary_routing_with_only_an_independent_router_head():
    v47, v51 = _configs()
    changed = {key for key in set(v47) | set(v51) if v47.get(key) != v51.get(key)}
    assert changed == {
        "stage_b_dense_duty_confidence_revision",
        "stage_b_dense_duty_confidence_head_gradient_contract",
        "stage_b_dense_duty_trace_audit_path",
        "stage_b_dense_duty_trace_audit_sha256",
        "stage_b_v15_tail_queue_negative_reduction_contract",
        "stage_b_v21_token_edit_query_scope",
        "stage_b_v11_trainable_params_min",
        "stage_b_v11_trainable_params_max",
    }
    assert v51["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_independent_deployed_router_v51"
    )
    assert v51["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_deployed_router_global_absolute_v5"
    )
    assert v51["stage_b_dense_duty_deployed_veto_routing_weight"] == 0.1
    assert v51[
        "stage_b_dense_duty_deployed_veto_routing_reduction_contract"
    ] == "balanced_top_quarter_cvar_v2"
    assert v51[
        "stage_b_v15_tail_queue_positive_trust_reduction_contract"
    ] == "top_quarter_cvar_v2"
    assert v51["stage_b_v15_tail_queue_negative_reduction_contract"] == (
        "all_mean_v1"
    )
    assert v51["stage_b_v21_token_edit_query_scope"] == "target_iou_v1"
    assert v51["stage_b_v11_trainable_params_min"] == 536_734
    assert v51["stage_b_v11_trainable_params_max"] == 536_734
    assert v51["stage_b_dense_duty_trace_audit_sha256"] == (
        "4a51a2d9a79284763922747ba80f9588aafa2675467283ea06cd51798cebd027"
    )
    assert v51["stage_b_dense_duty_trace_audit_path"].endswith(
        "candidate_split_independent_deployed_router_trace_audit_20260802/"
        "receipt.json"
    )


def test_v51_training_contract_is_exact_v33():
    contract = build_training_contract(probe._BASE._formal_current_args())
    assert contract["schema"] == "pivot.stageb.dense_duty_training_contract/v33"
    values = contract["values"]
    assert values["stage_b_dense_duty_confidence_revision"] == (
        "word_veto_candidate_split_independent_deployed_router_v51"
    )
    assert values["stage_b_dense_duty_confidence_head_gradient_contract"] == (
        "split_token_veto_deployed_router_global_absolute_v5"
    )
    assert values["stage_b_dense_duty_deployed_veto_routing_weight"] == 0.1
    assert values["stage_b_v15_tail_queue_negative_reduction_contract"] == (
        "all_mean_v1"
    )
    assert values["stage_b_v21_token_edit_query_scope"] == "target_iou_v1"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "split_token_veto_global_absolute_v2",
        ),
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.25),
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
def test_v51_training_validation_fails_closed_on_contract_drift(field, value):
    cfg = SLConfig.fromfile(str(probe.CONFIG))
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError):
        training_main._validate_stage_b_dense_duty_args(cfg)


def test_v51_probe_and_formal_controllers_are_fresh_from_rank_only():
    assert probe.UPDATES == 400
    assert formal.UPDATES == 4_412
    assert probe.OUTPUT != v47_probe.OUTPUT
    assert formal.OUTPUT != probe.OUTPUT
    assert "independent_deployed_router_highmem_20260802" in str(probe.OUTPUT)

    probe_command = probe.command("start")
    assert "--resume" not in probe_command
    assert probe_command[probe_command.index("--pretrain_model_path") + 1] == str(
        probe._BASE.RANK_SOURCE
    )
    assert probe_command[probe_command.index("--max_train_iters") + 1] == "400"

    formal_command = formal.command("start")
    assert "--resume" not in formal_command
    assert formal_command[
        formal_command.index("--pretrain_model_path") + 1
    ] == str(formal._BASE.RANK_SOURCE)
    assert formal_command[formal_command.index("--max_train_iters") + 1] == (
        "4412"
    )
    assert str(probe.CHECKPOINT) not in formal_command


def test_v51_health_seals_v33_and_three_independent_point_one_clips():
    assert health.TRAINING_CONTRACT_SCHEMA == (
        "pivot.stageb.dense_duty_training_contract/v33"
    )
    assert health.EXPECTED_HEAD_CONTRACT == (
        "split_token_veto_deployed_router_global_absolute_v5"
    )
    assert health.MIGRATION_FRESH_TENSOR_COUNT == 81
    assert health.MIGRATION_FRESH_ELEMENT_COUNT == 536_735
    assert len(health.MIGRATION_FRESH_SHA256) == 64
    assert health.MIGRATION_FINGERPRINT_IS_PLACEHOLDER == (
        health.MIGRATION_FRESH_SHA256 == "0" * 64
    )
    assert health._CORE.EXPECTED_CONTRACT_VALUES[
        "stage_b_dense_duty_deployed_veto_routing_weight"
    ] == 0.1

    runtime = {
        "optimizer_step_boundaries": 400,
        "successful_optimizer_steps": 400,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "nonfinite_token_veto_gradient_boundaries": 0,
        "nonfinite_deployed_router_gradient_boundaries": 0,
        "nonfinite_global_absolute_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "zero_token_veto_gradient_successful_steps": 0,
        "zero_deployed_router_gradient_successful_steps": 0,
        "zero_global_absolute_gradient_successful_steps": 0,
        "last_token_veto_grad_norm_preclip": 20.0,
        "max_token_veto_grad_norm_preclip": 30.0,
        "last_deployed_router_grad_norm_preclip": 18.0,
        "max_deployed_router_grad_norm_preclip": 29.0,
        "last_global_absolute_grad_norm_preclip": 22.0,
        "max_global_absolute_grad_norm_preclip": 31.0,
        "min_amp_scale": 256.0,
        "last_amp_scale": 256.0,
    }
    trajectory = {
        "train_grad_norm_dense_duty_active_preclip": 40.0,
        "train_grad_tensor_count_dense_duty_active": 73.0,
        "train_grad_norm_dense_duty_token_veto_preclip": 20.0,
        "train_grad_tensor_count_dense_duty_token_veto": 21.0,
        "train_grad_norm_dense_duty_deployed_router_preclip": 18.0,
        "train_grad_tensor_count_dense_duty_deployed_router": 6.0,
        "train_grad_norm_dense_duty_global_absolute_preclip": 22.0,
        "train_grad_tensor_count_dense_duty_global_absolute": 46.0,
        "train_grad_norm_dense_duty_token_veto_postclip": 0.1,
        "train_grad_norm_dense_duty_deployed_router_postclip": 0.1,
        "train_grad_norm_dense_duty_global_absolute_postclip": 0.1,
        "train_grad_norm_dense_duty_active_postclip": math.sqrt(0.03),
        "train_amp_step_skipped": 0.0,
    }
    checks = health._health_checks(runtime, trajectory)
    assert all(check["passed"] for check in checks.values())
    assert "u222_three_independent_clips_v5_evidence" in checks

    # A small router gradient must remain unchanged instead of being scaled by
    # the much larger token/global owners' joint norm.
    trajectory["train_grad_norm_dense_duty_deployed_router_preclip"] = 0.02
    trajectory["train_grad_norm_dense_duty_deployed_router_postclip"] = 0.02
    trajectory["train_grad_norm_dense_duty_active_postclip"] = math.sqrt(
        0.1**2 + 0.02**2 + 0.1**2
    )
    checks = health._health_checks(runtime, trajectory)
    assert checks["u222_three_independent_clips_v5_evidence"]["passed"] is True

    # U222 stores epoch means. In general, the mean of each step's combined
    # norm is slightly larger than the norm of the three owner means.
    trajectory.update(
        {
            "train_grad_norm_dense_duty_active_preclip": 14.34833695437457,
            "train_grad_norm_dense_duty_token_veto_preclip": 2.4052601070017428,
            "train_grad_norm_dense_duty_deployed_router_preclip": 0.11860100683328267,
            "train_grad_norm_dense_duty_global_absolute_preclip": 13.964743779183507,
            "train_grad_norm_dense_duty_token_veto_postclip": 0.09999994953741899,
            "train_grad_norm_dense_duty_deployed_router_postclip": 0.09933323151356466,
            "train_grad_norm_dense_duty_global_absolute_postclip": 0.09999999041492874,
            "train_grad_norm_dense_duty_active_postclip": 0.17284257528749672,
        }
    )
    checks = health._health_checks(runtime, trajectory)
    assert checks["u222_three_independent_clips_v5_evidence"]["passed"] is True

    post_l2_lower = math.sqrt(
        trajectory["train_grad_norm_dense_duty_token_veto_postclip"] ** 2
        + trajectory["train_grad_norm_dense_duty_deployed_router_postclip"] ** 2
        + trajectory["train_grad_norm_dense_duty_global_absolute_postclip"] ** 2
    )
    trajectory["train_grad_norm_dense_duty_active_postclip"] = (
        post_l2_lower - 1e-4
    )
    checks = health._health_checks(runtime, trajectory)
    assert checks["u222_three_independent_clips_v5_evidence"]["passed"] is False

    trajectory["train_grad_norm_dense_duty_active_postclip"] = 0.17284257528749672
    trajectory["train_grad_norm_dense_duty_deployed_router_postclip"] = 0.1001
    checks = health._health_checks(runtime, trajectory)
    assert checks["u222_three_independent_clips_v5_evidence"]["passed"] is False


def _write_admission_binder(path: Path, *, routing_weight: float) -> None:
    path.write_text(
        "def _bind_stage_b_confidence_probe_admission(args):\n"
        "    revision = getattr(args, 'revision', '')\n"
        "    if (\n"
        '        revision == "word_veto_candidate_split_independent_'
        'deployed_router_v51"\n'
        "        and str(getattr(args, "
        "'stage_b_dense_duty_confidence_head_gradient_contract', '')) "
        '== "split_token_veto_deployed_router_global_absolute_v5"\n'
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
        "'stage_b_v15_tail_queue_positive_gradient_contract', '')) "
        '== "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"\n'
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
        '        formal_contract = "u400_word_veto_candidate_split_independent_'
        'deployed_router_confidence_strict1607_v51"\n'
        "        from tools import "
        "run_stageb_confidence_adapter_candidate_split_independent_deployed_"
        "router_probe_evaluation as promotion\n"
        "        return formal_contract, promotion\n"
        "    return None\n",
        encoding="utf-8",
    )


def test_v51_strict_controller_and_branch_local_ast_are_fail_closed(tmp_path):
    command = evaluation.build_command()
    assert command[command.index("--config") + 1] == str(evaluation.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(evaluation.CHECKPOINT)
    assert command[command.index("--batch_size") + 1] == "16"
    assert command[command.index("--num_workers") + 1] == "4"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--topk") + 1] == "1"
    assert "--amp" in command and "--skip_ref" in command
    assert evaluation._CORE._load_health_audit() is health.audit

    good = tmp_path / "good_main.py"
    bad = tmp_path / "bad_main.py"
    _write_admission_binder(good, routing_weight=0.1)
    _write_admission_binder(bad, routing_weight=0.25)
    assert evaluation._formal_main_admission_is_wired(REPO_ROOT / "main.py")
    assert evaluation._formal_main_admission_is_wired(good)
    assert not evaluation._formal_main_admission_is_wired(bad)
