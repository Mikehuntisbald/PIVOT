import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import (
    audit_stageb_confidence_adapter_candidate_tn_only_carrier_pair_probe_health as health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_tn_only_carrier_pair_highmem_formal as formal,
)
from tools import (
    run_stageb_confidence_adapter_candidate_tn_only_carrier_pair_probe_evaluation as evaluation,
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
        "train_fixed_text_raw_veto_positive_sample_count_unscaled": 16.0,
        "train_fixed_text_raw_veto_positive_carrier_sample_count_unscaled": 16.0,
        "train_fixed_text_raw_veto_tn_sample_count_unscaled": 12.0,
        "train_fixed_text_raw_veto_tn_carrier_sample_count_unscaled": 12.0,
        "train_fixed_text_raw_veto_carrier_pair_sample_count_unscaled": 12.0,
        "train_fixed_text_confidence_tn_train_eligible_count_unscaled": 12.0,
        "train_fixed_text_confidence_tn_train_excluded_count_unscaled": 4.0,
        "train_fixed_text_token_all_negative_count_unscaled": 0.0,
        "train_fixed_text_token_provenance_valid_count_unscaled": 16.0,
        "train_fixed_text_token_direct_trace_valid_count_unscaled": 12.0,
        "train_fixed_text_global_tn_sample_count_unscaled": 12.0,
        "train_fixed_text_raw_veto_carrier_pair_violation_rate_unscaled": 0.8,
        "train_fixed_text_raw_veto_tail_pair_violation_rate_unscaled": 0.9,
        "train_loss_fixed_text_raw_veto_carrier_pair": 0.05,
        "train_loss_fixed_text_raw_veto_carrier_pair_unscaled": 0.2,
        "train_fixed_text_raw_veto_carrier_pair_gap_mean_unscaled": 0.1,
        "train_fixed_text_raw_veto_carrier_pair_hinge_mean_unscaled": 0.15,
        "train_fixed_text_raw_veto_tail_pair_gap_mean_unscaled": 0.05,
        "train_fixed_text_raw_veto_tail_pair_hinge_mean_unscaled": 0.2,
        "train_fixed_text_raw_veto_tail_pair_effective_sample_count_unscaled": 8.0,
        "train_fixed_text_raw_veto_tn_tail_effective_sample_count_unscaled": 8.0,
    }
    value.update(overrides)
    return value


def _source_closure():
    paths = (
        "engine.py",
        "main.py",
        "models/GroundingDINO/groundingdino.py",
        "models/GroundingDINO/stage_b_fixed_text_criterion.py",
        "util/stage_b_dense_duty_audit.py",
    )
    files = [
        {"path": path, "size_bytes": index + 1, "sha256": str(index + 1) * 64}
        for index, path in enumerate(paths)
    ]
    code = {
        "schema": health.CODE_SOURCE_CLOSURE_SCHEMA,
        "file_count": len(files),
        "files": files,
        "sha256": health._canonical_sha256(files, label="test code ledger"),
    }
    config_files = [
        {
            "path": health.EXPECTED_CONFIG_ENTRY,
            "size_bytes": 1,
            "sha256": "a" * 64,
        }
    ]
    config = {
        "schema": health.CONFIG_SOURCE_CLOSURE_SCHEMA,
        "entry": health.EXPECTED_CONFIG_ENTRY,
        "file_count": len(config_files),
        "files": config_files,
        "sha256": health._canonical_sha256(
            {"entry": health.EXPECTED_CONFIG_ENTRY, "files": config_files},
            label="test config closure",
        ),
    }
    return {
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


def _checkpoint_args(*, runtime=None, contract_overrides=None, args_overrides=None):
    values = {
        **health.EXPECTED_PACKED_VALUES,
        **health.EXPECTED_V42_VALUES,
        "stage_b_dense_duty_source_closure": _source_closure(),
    }
    values.update({} if contract_overrides is None else contract_overrides)
    contract = {
        "schema": health.TRAINING_CONTRACT_SCHEMA,
        "values": values,
        "sha256": health._canonical_sha256(values, label="test training contract"),
    }
    model = {
        "stage_b_fixed_text_scorer.rank_tower.weight": torch.tensor([1.0])
    }
    rank = health.fingerprint_named_tensors(model, sorted(model))
    args = {
        **values,
        "stage_b_dense_duty_training_contract": contract,
        "stage_b_dense_duty_runtime_audit": (
            _runtime() if runtime is None else runtime
        ),
        "stage_b_dense_duty_rank_source_rank_sha256": rank["sha256"],
    }
    args.update({} if args_overrides is None else args_overrides)
    return args, model


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

    def write_checkpoint(
        *,
        criterion=None,
        runtime=None,
        contract_overrides=None,
        args_overrides=None,
        payload_overrides=None,
        contract_schema=None,
    ):
        args, model = _checkpoint_args(
            runtime=runtime,
            contract_overrides=contract_overrides,
            args_overrides=args_overrides,
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
            "model": model,
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


def test_healthy_v42_checkpoint_binds_v24_pair_route_without_controller_inspect(
    evidence, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        health.probe,
        "inspect",
        lambda: pytest.fail("health must not inspect mutable resume state"),
    )

    code, report = _run(evidence)

    assert code == 0
    assert report["schema"] == health.SCHEMA
    assert report["decision"] == "healthy_for_strict1607_diagnostic"
    assert report["candidate"]["controller"] == {
        "status": "terminal",
        "action": "complete",
        "updates": 400,
        "rank_sha256": report["rank"]["sha256"],
        "evidence_source": "immutable_checkpoint_v24",
        "current_resume_controller_required": False,
    }
    assert report["packed_forward"]["training_contract_schema"] == (
        health.TRAINING_CONTRACT_SCHEMA
    )
    assert report["packed_forward"]["token_edit_query_scope"] == "target_iou_v1"
    assert report["packed_forward"]["token_edit_query_scope_source"] == (
        "checkpoint_args_default"
    )
    assert report["packed_forward"]["gradient_contract"] == (
        "tn_only_positive_detached_v2"
    )
    assert report["packed_forward"]["terminal_cursor"]["iteration"] == 356
    assert report["failed_checks"] == []


def test_realistic_v42_endpoint_is_valid_evidence_but_expectedly_unhealthy(evidence):
    evidence["write_checkpoint"](
        criterion=_criterion(
            tail_positive_queue=torch.full(
                (4096,), -1.3100193738937378, dtype=torch.float32
            ),
            tail_negative_queue=torch.full(
                (4096,), 0.7819646596908569, dtype=torch.float32
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
    assert report["checks"]["u222_pair_support_conservation_exact"]["passed"]
    assert report["checks"]["packed_v24_tn_only_gradient_contract_bound"][
        "passed"
    ]


@pytest.mark.parametrize(
    "checkpoint_kwargs,error",
    [
        (
            {"contract_schema": "pivot.stageb.dense_duty_training_contract/v23"},
            "schema must be",
        ),
        (
            {
                "contract_overrides": {
                    "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract": (
                        "bidirectional_v1"
                    )
                },
                "args_overrides": {
                    "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract": (
                        "bidirectional_v1"
                    )
                },
            },
            "tn_only_positive_detached_v2",
        ),
        (
            {
                "contract_overrides": {
                    "stage_b_dense_duty_raw_veto_carrier_pair_weight": 0.5
                },
                "args_overrides": {
                    "stage_b_dense_duty_raw_veto_carrier_pair_weight": 0.5
                },
            },
            "carrier_pair_weight=0.25",
        ),
        (
            {
                "contract_overrides": {
                    "stage_b_dense_duty_raw_veto_tail_quantile": 0.9
                },
                "args_overrides": {
                    "stage_b_dense_duty_raw_veto_tail_quantile": 0.9
                },
            },
            "raw_veto_tail_quantile=0.95",
        ),
        (
            {
                "args_overrides": {
                    "stage_b_v21_token_edit_query_scope": (
                        "target_iou_union_detached_final_confidence_base_argmax_v2"
                    )
                }
            },
            "target_iou_v1",
        ),
        ({"payload_overrides": {"iteration": 354}}, "cursor does not replay"),
    ],
)
def test_v42_checkpoint_contract_drift_is_invalid(
    evidence, checkpoint_kwargs, error
):
    evidence["write_checkpoint"](**checkpoint_kwargs)

    code, report = _run(evidence)

    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert error in report["error"]["message"]


def test_pair_conservation_drift_is_a_health_failure(evidence):
    evidence["write_log"](
        train_fixed_text_raw_veto_carrier_pair_sample_count_unscaled=11.0
    )

    code, report = _run(evidence)

    assert code == 1
    assert report["failed_checks"] == [
        "u222_pair_support_conservation_exact"
    ]


def test_missing_pair_evidence_is_invalid(evidence):
    field = "train_fixed_text_raw_veto_tail_pair_hinge_mean_unscaled"
    evidence["write_log"](remove=(field,))

    code, report = _run(evidence)

    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert field in report["error"]["message"]


def test_runtime_failure_remains_a_health_failure(evidence):
    evidence["write_checkpoint"](
        runtime=_runtime(amp_skipped_optimizer_steps=1)
    )

    code, report = _run(evidence)

    assert code == 1
    assert report["failed_checks"] == ["runtime_amp_skips_zero"]


def test_v42_evaluation_binds_dedicated_health_and_exact_promotion(
    monkeypatch: pytest.MonkeyPatch,
):
    assert evaluation.health is health
    assert evaluation.HEALTH_SCHEMA == health.SCHEMA
    assert evaluation._BASE.HEALTH_SCHEMA == health.SCHEMA
    assert evaluation._BASE.EXPECTED_UPDATES == 400
    assert evaluation._BASE._load_health_audit() is health.audit
    assert evaluation.FORMAL_ADMISSION_CONTRACT == (
        "u400_word_veto_candidate_tn_only_carrier_pair_"
        "confidence_strict1607_v42"
    )
    promotion = evaluation._BASE._validate_formal_config_promotion()
    assert promotion["all_other_config_values_equal"] is True

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
    postflight = evaluation._v42_postflight({})
    assert "terminal_u300_diagnostic" not in postflight["contracts"]
    assert postflight["contracts"]["terminal_u400_diagnostic"] is True
    assert postflight["contracts"]["tn_only_positive_detached_v2"] is True


def test_v42_promotion_and_formal_controllers_remain_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        evaluation.training,
        "inspect",
        lambda: {"status": "invalid", "reason": "source closure drifted"},
    )
    with pytest.raises(evaluation._BASE.ProbeEvaluationError, match="status"):
        evaluation.preflight(health_audit=lambda: {})

    assert formal.CONFIG.resolve(strict=True) == evaluation.FORMAL_CONFIG.resolve(
        strict=True
    )
    assert formal._BASE.FORMAL_ADMISSION_VALIDATOR is formal.verify_probe_admission
