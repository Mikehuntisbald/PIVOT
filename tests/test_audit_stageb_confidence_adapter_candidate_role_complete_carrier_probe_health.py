import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import (
    audit_stageb_confidence_adapter_candidate_role_complete_carrier_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_role_complete_carrier_probe_evaluation as evaluation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(**overrides):
    value = {
        "schema": health.RUNTIME_SCHEMA,
        "optimizer_step_boundaries": 400,
        "successful_optimizer_steps": 400,
        "amp_skipped_optimizer_steps": 0,
        "nonfinite_gradient_boundaries": 0,
        "zero_gradient_successful_steps": 0,
        "min_amp_scale": 256.0,
        "last_amp_scale": 256.0,
        "max_active_grad_norm_preclip": 2.0,
        "last_active_grad_norm_preclip": 1.0,
    }
    value.update(overrides)
    return value


def _criterion(**overrides):
    value = {
        "tail_positive_queue": torch.zeros(4096, dtype=torch.float32),
        "tail_negative_queue": torch.full((4096,), -1.0, dtype=torch.float32),
        "tail_positive_ptr": torch.tensor(0, dtype=torch.int64),
        "tail_negative_ptr": torch.tensor(0, dtype=torch.int64),
        "tail_positive_count": torch.tensor(4096, dtype=torch.int64),
        "tail_negative_count": torch.tensor(4096, dtype=torch.int64),
    }
    value.update(overrides)
    return value


def _log(**overrides):
    value = {
        "train_optimizer_updates": 222,
        "train_loss_fixed_text_token_unscaled": 1.0,
        "train_loss_fixed_text_local_absolute_unscaled": 0.5,
        "train_loss_fixed_text_global_tn_negative_unscaled": 0.4,
        "train_loss_fixed_text_tail_queue_unscaled": 0.3,
        "train_loss_fixed_text_global_tn_tail": 0.0,
        "train_fixed_text_tail_queue_positive_count_unscaled": 4096.0,
        "train_fixed_text_tail_queue_negative_count_unscaled": 4096.0,
        "train_fixed_text_tail_queue_threshold_valid_unscaled": 1.0,
        "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled": 0.2,
        "train_stage_b_dense_confidence_positive_delta_mean_unscaled": -1.0,
        "train_stage_b_dense_confidence_tn_delta_mean_unscaled": -2.0,
        "train_stage_b_dense_confidence_positive_veto_gate_mean_unscaled": 0.2,
        "train_stage_b_dense_confidence_tn_veto_gate_mean_unscaled": 0.8,
        "train_amp_step_skipped": 0.0,
        "train_fixed_text_token_role_carrier_pair_selected_count_unscaled": 12.0,
        "train_fixed_text_token_role_carrier_positive_added_count_unscaled": 4.0,
        "train_fixed_text_token_role_carrier_tn_added_count_unscaled": 7.0,
        "train_fixed_text_token_role_carrier_positive_target_overlap_count_unscaled": 8.0,
        "train_fixed_text_token_role_carrier_tn_target_overlap_count_unscaled": 5.0,
        "train_fixed_text_token_edit_carrier_selected_count_unscaled": 12.0,
        "train_fixed_text_token_edit_carrier_added_count_unscaled": 7.0,
        "train_fixed_text_token_edit_carrier_target_overlap_count_unscaled": 5.0,
        "train_fixed_text_confidence_tn_train_eligible_count_unscaled": 12.0,
        "train_fixed_text_confidence_tn_train_excluded_count_unscaled": 4.0,
        "train_fixed_text_token_all_negative_count_unscaled": 0.0,
        "train_fixed_text_token_provenance_valid_count_unscaled": 16.0,
        "train_fixed_text_token_direct_trace_valid_count_unscaled": 12.0,
        "train_fixed_text_global_tn_sample_count_unscaled": 12.0,
    }
    value.update(overrides)
    return value


def _checkpoint_args(*, runtime=None, **overrides):
    files = [
        {
            "path": "engine.py",
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
        {
            "path": "main.py",
            "size_bytes": 1,
            "sha256": "2" * 64,
        },
    ]
    code = {
        "schema": health.CODE_SOURCE_CLOSURE_SCHEMA,
        "file_count": len(files),
        "files": files,
        "sha256": health._canonical_sha256(files, label="test code ledger"),
    }
    config_files = [
        {
            "path": "config/test_role_complete_carrier.py",
            "size_bytes": 1,
            "sha256": "3" * 64,
        }
    ]
    config_entry = config_files[0]["path"]
    config = {
        "schema": health.CONFIG_SOURCE_CLOSURE_SCHEMA,
        "entry": config_entry,
        "file_count": len(config_files),
        "files": config_files,
        "sha256": health._canonical_sha256(
            {"entry": config_entry, "files": config_files},
            label="test config closure",
        ),
    }
    closure = {
        "schema": health.SOURCE_CLOSURE_SCHEMA,
        "code": code,
        "config": config,
        "sha256": health._canonical_sha256(
            {
                "code_sha256": code["sha256"],
                "config_sha256": config["sha256"],
            },
            label="test phase closure",
        ),
    }
    values = {
        **health.EXPECTED_PACKED_VALUES,
        "stage_b_v22_score_ownership": health.EXPECTED_OWNERSHIP,
        "stage_b_v21_token_edit_query_scope": health.EXPECTED_SCOPE,
        "stage_b_dense_duty_source_closure": closure,
    }
    values.update(overrides)
    contract = {
        "schema": health.TRAINING_CONTRACT_SCHEMA,
        "sha256": health._canonical_sha256(
            values, label="test training contract"
        ),
        "values": values,
    }
    return {
        **values,
        "stage_b_dense_duty_training_contract": contract,
        "stage_b_dense_duty_runtime_audit": (
            _runtime() if runtime is None else runtime
        ),
    }


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline_log = tmp_path / "old_log.txt"
    baseline_u222 = tmp_path / "old_u222.pth"
    baseline_u444 = tmp_path / "old_u444.pth"
    baseline_log.write_text("sealed old log\n", encoding="utf-8")
    baseline_u222.write_bytes(b"sealed old U222 checkpoint")
    baseline_u444.write_bytes(b"sealed old U444 checkpoint")

    output = tmp_path / "probe" / "u000400"
    output.mkdir(parents=True)
    checkpoint = output / "checkpoint_iter.pth"
    log = output / "log.txt"

    monkeypatch.setattr(health, "BASELINE_LOG", baseline_log)
    monkeypatch.setattr(health, "BASELINE_U222_CHECKPOINT", baseline_u222)
    monkeypatch.setattr(health, "BASELINE_U444_CHECKPOINT", baseline_u444)
    monkeypatch.setattr(health, "BASELINE_LOG_SHA256", _sha256(baseline_log))
    monkeypatch.setattr(
        health, "BASELINE_U222_CHECKPOINT_SHA256", _sha256(baseline_u222)
    )
    monkeypatch.setattr(
        health, "BASELINE_U444_CHECKPOINT_SHA256", _sha256(baseline_u444)
    )
    monkeypatch.setattr(health.probe, "OUTPUT", output)
    monkeypatch.setattr(health.probe, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(
        health.probe,
        "inspect",
        lambda: {
            "status": "terminal",
            "action": "complete",
            "updates": 400,
            "rank_sha256": "5" * 64,
        },
    )

    def write_checkpoint(
        *,
        criterion=None,
        runtime=None,
        args_overrides=None,
        payload_overrides=None,
        contract_schema=None,
    ):
        args = _checkpoint_args(
            runtime=runtime,
            **({} if args_overrides is None else args_overrides),
        )
        if contract_schema is not None:
            args["stage_b_dense_duty_training_contract"]["schema"] = contract_schema
        payload = {
            "epoch": 1,
            "iteration": 356,
            "optimizer_updates": 400,
            "epoch_finished": False,
            "checkpoint_reason": "max_train_iters",
            "criterion": _criterion() if criterion is None else criterion,
            "args": args,
        }
        payload.update({} if payload_overrides is None else payload_overrides)
        torch.save(payload, checkpoint)

    def write_log(*, remove=(), **overrides):
        payload = _log(**overrides)
        for field in remove:
            payload.pop(field)
        log.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    write_checkpoint()
    write_log()
    return {
        "output": output,
        "checkpoint": checkpoint,
        "log": log,
        "write_checkpoint": write_checkpoint,
        "write_log": write_log,
    }


def _run(evidence, name="report.json"):
    report = evidence["output"] / name
    code = health.run(["--output", str(report)])
    return code, json.loads(report.read_text(encoding="utf-8"))


def test_healthy_v41_u400_binds_role_carriers_and_packed_cursor(evidence):
    code, report = _run(evidence)

    assert code == 0
    assert report["schema"] == health.SCHEMA
    assert report["decision"] == "healthy_for_strict1607_diagnostic"
    assert report["trajectory_u222"]["train_optimizer_updates"] == 222
    assert "endpoint_u300" not in report
    assert report["endpoint_u400"]["positive_q05"] == 0.0
    assert report["endpoint_u400"]["tn_q95"] == -1.0
    assert report["packed_forward"]["replay"] == {
        "physical_forwards": 800,
        "logical_batches": 1599,
    }
    assert report["packed_forward"]["geometry"]["effective_batch_size"] == 64
    assert report["failed_checks"] == []
    assert all(check["passed"] for check in report["checks"].values())


def test_u400_positive_q05_retains_exact_score_ge_order_statistic():
    criterion = _criterion(
        tail_positive_queue=torch.arange(4096, dtype=torch.float32)
    )

    endpoint = health._audit_endpoint_u400(criterion)

    assert endpoint["positive_q05"] == 204.0
    assert endpoint["positive_q05"] != torch.quantile(
        criterion["tail_positive_queue"], 0.05
    ).item()


def test_malformed_u400_queue_is_invalid_evidence(evidence):
    evidence["write_checkpoint"](
        criterion=_criterion(
            tail_negative_queue=torch.zeros(4096, dtype=torch.float64)
        )
    )
    code, report = _run(evidence)

    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert "must be float32" in report["error"]["message"]


def test_realistic_terminal_endpoint_is_valid_but_q05_unhealthy(evidence):
    evidence["write_checkpoint"](
        criterion=_criterion(
            tail_positive_queue=torch.full(
                (4096,), -1.257617712020874, dtype=torch.float32
            ),
            tail_negative_queue=torch.full(
                (4096,), 0.7815719246864319, dtype=torch.float32
            ),
        )
    )
    code, report = _run(evidence)

    assert code == 1
    assert report["decision"] == "unhealthy_do_not_run_diagnostic"
    assert report["failed_checks"] == [
        "u400_positive_q05_in_operating_range",
        "u400_tn_q95_in_operating_range",
        "u400_operating_gap_in_operating_range",
    ]
    assert all(
        check["passed"]
        for name, check in report["checks"].items()
        if "role_carrier" in name
    )


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "train_fixed_text_token_role_carrier_pair_selected_count_unscaled",
            float("nan"),
        ),
        ("train_fixed_text_token_role_carrier_tn_added_count_unscaled", None),
    ],
)
def test_missing_or_nonfinite_role_evidence_is_invalid(
    evidence, field, value
):
    if value is None:
        evidence["write_log"](remove=(field,))
    else:
        evidence["write_log"](**{field: value})
    code, report = _run(evidence)

    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert field in report["error"]["message"]


@pytest.mark.parametrize(
    "overrides,failed_check",
    [
        (
            {
                "train_fixed_text_token_role_carrier_positive_added_count_unscaled": 3.0
            },
            "u222_role_carrier_positive_partition_exact",
        ),
        (
            {"train_fixed_text_token_all_negative_count_unscaled": 1.0},
            "u222_role_carrier_trace_partition_exact",
        ),
        (
            {
                "train_fixed_text_confidence_tn_train_eligible_count_unscaled": 11.0
            },
            "u222_role_carrier_full_eligible_coverage",
        ),
    ],
)
def test_role_carrier_conservation_drift_is_unhealthy(
    evidence, overrides, failed_check
):
    evidence["write_log"](**overrides)
    code, report = _run(evidence)

    assert code == 1
    assert failed_check in report["failed_checks"]


def test_role_carrier_conservation_uses_float_tolerance(evidence):
    evidence["write_log"](
        train_fixed_text_token_role_carrier_positive_added_count_unscaled=(
            4.0 + 5e-7
        )
    )
    code, report = _run(evidence)

    assert code == 0
    assert report["checks"][
        "u222_role_carrier_positive_partition_exact"
    ]["passed"] is True


@pytest.mark.parametrize(
    "checkpoint_kwargs,error",
    [
        (
            {"payload_overrides": {"iteration": 354}},
            "cursor does not replay",
        ),
        (
            {"contract_schema": "pivot.stageb.dense_duty_training_contract/v22"},
            "schema must be",
        ),
        (
            {
                "args_overrides": {
                    "stage_b_dense_duty_forward_pack_factor": 1
                }
            },
            "forward_pack_factor=2",
        ),
        (
            {
                "args_overrides": {
                    "stage_b_v21_token_edit_query_scope": (
                        "target_iou_union_detached_final_confidence_base_argmax_v2"
                    )
                }
            },
            "token_edit_query_scope",
        ),
    ],
)
def test_packed_contract_or_cursor_drift_is_invalid(
    evidence, checkpoint_kwargs, error
):
    evidence["write_checkpoint"](**checkpoint_kwargs)
    code, report = _run(evidence)

    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert error in report["error"]["message"]


def test_runtime_failures_remain_health_failures(evidence):
    evidence["write_checkpoint"](
        runtime=_runtime(amp_skipped_optimizer_steps=1)
    )
    code, report = _run(evidence)

    assert code == 1
    assert report["failed_checks"] == ["runtime_amp_skips_zero"]


def test_v41_evaluation_binds_only_the_dedicated_health_schema(
    monkeypatch: pytest.MonkeyPatch,
):
    assert evaluation.health is health
    assert evaluation.HEALTH_SCHEMA == health.SCHEMA
    assert evaluation._BASE.HEALTH_SCHEMA == health.SCHEMA
    assert evaluation._BASE.EXPECTED_UPDATES == 400
    assert evaluation._BASE._load_health_audit() is health.audit
    assert "role_complete_carrier" in evaluation.SCHEMA
    assert "role_complete_carrier" in evaluation.POSTFLIGHT_SCHEMA
    assert "role_complete_carrier" in evaluation.ADMISSION_SCHEMA

    monkeypatch.setattr(
        evaluation,
        "_BASE_POSTFLIGHT",
        lambda *_args, **_kwargs: {
            "contracts": {
                "terminal_u300_diagnostic": True,
                "full_strict1607": True,
            }
        },
    )
    postflight = evaluation._v41_postflight({})
    assert "terminal_u300_diagnostic" not in postflight["contracts"]
    assert postflight["contracts"]["terminal_u400_diagnostic"] is True
