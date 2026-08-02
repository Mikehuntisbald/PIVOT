import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import audit_stageb_confidence_adapter_veto_probe_health as health


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(**overrides):
    value = {
        "schema": health.RUNTIME_SCHEMA,
        "optimizer_step_boundaries": 300,
        "successful_optimizer_steps": 300,
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
        "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled": 0.05,
        "train_stage_b_dense_confidence_positive_delta_mean_unscaled": 0.0,
        "train_stage_b_dense_confidence_tn_delta_mean_unscaled": -0.1,
        "train_stage_b_dense_confidence_positive_veto_gate_mean_unscaled": 0.2,
        "train_stage_b_dense_confidence_tn_veto_gate_mean_unscaled": 0.8,
        "train_amp_step_skipped": 0.0,
    }
    value.update(overrides)
    return value


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline_log = tmp_path / "old_log.txt"
    baseline_u222 = tmp_path / "old_u222.pth"
    baseline_u444 = tmp_path / "old_u444.pth"
    baseline_log.write_text("sealed old log\n", encoding="utf-8")
    baseline_u222.write_bytes(b"sealed old U222 checkpoint")
    baseline_u444.write_bytes(b"sealed old U444 checkpoint")

    output = tmp_path / "probe" / "u000300"
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
        lambda: {"status": "terminal", "action": "complete", "updates": 300},
    )

    def write_checkpoint(*, criterion=None, runtime=None):
        torch.save(
            {
                "optimizer_updates": 300,
                "checkpoint_reason": "max_train_iters",
                "criterion": _criterion() if criterion is None else criterion,
                "args": {
                    "stage_b_dense_duty_runtime_audit": (
                        _runtime() if runtime is None else runtime
                    )
                },
            },
            checkpoint,
        )

    def write_log(**overrides):
        log.write_text(json.dumps(_log(**overrides)) + "\n", encoding="utf-8")

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


def test_default_report_is_outside_atomic_training_directory(evidence):
    assert health._default_output().parent == evidence["output"].parent
    assert health._default_output().parent != evidence["output"]


def test_healthy_u300_with_only_one_u222_log_exits_zero(evidence):
    code, report = _run(evidence)
    assert code == 0
    assert report["schema"] == health.SCHEMA
    assert report["decision"] == "healthy_for_strict1607_diagnostic"
    assert report["trajectory_u222"]["train_optimizer_updates"] == 222
    assert report["endpoint_u300"]["positive_q05"] == 0.0
    assert report["endpoint_u300"]["tn_q95"] == -1.0
    assert report["failed_checks"] == []
    assert all(value["passed"] for value in report["checks"].values())


def test_positive_q05_matches_exact_score_ge_order_statistic():
    values = torch.arange(4096, dtype=torch.float32)
    exact = health._exact_lower_tail_operating_threshold(values, 0.05)
    assert exact.item() == 204.0
    assert exact.item() != torch.quantile(values, 0.05).item()


def test_valid_but_unhealthy_loss_exits_one(evidence):
    evidence["write_log"](train_loss_fixed_text_token_unscaled=3.0)
    code, report = _run(evidence)
    assert code == 1
    assert report["decision"] == "unhealthy_do_not_run_diagnostic"
    assert report["failed_checks"] == ["u222_token_below_old_u444"]


def test_nonterminal_controller_is_invalid_evidence_exit_two(evidence, monkeypatch):
    monkeypatch.setattr(
        health.probe,
        "inspect",
        lambda: {"status": "partial", "action": "resume", "updates": 299},
    )
    code, report = _run(evidence)
    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert "status='terminal'" in report["error"]["message"]


def test_nan_required_log_value_is_invalid_evidence(evidence):
    evidence["write_log"](
        train_stage_b_dense_confidence_tn_delta_mean_unscaled=float("nan")
    )
    code, report = _run(evidence)
    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert "must be finite" in report["error"]["message"]


@pytest.mark.parametrize(
    "criterion,error",
    [
        (
            _criterion(
                tail_negative_queue=torch.zeros(4096, dtype=torch.float64)
            ),
            "must be float32",
        ),
        (
            _criterion(tail_positive_count=torch.tensor(4095, dtype=torch.int64)),
            "must have count 4096",
        ),
        (
            _criterion(tail_negative_ptr=torch.tensor(4096, dtype=torch.int64)),
            "pointer must be in",
        ),
        (
            _criterion(
                tail_positive_queue=torch.full(
                    (4096,), float("nan"), dtype=torch.float32
                )
            ),
            "must be finite",
        ),
    ],
)
def test_malformed_or_nonfinite_queue_is_invalid_evidence(
    evidence, criterion, error
):
    evidence["write_checkpoint"](criterion=criterion)
    code, report = _run(evidence)
    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert error in report["error"]["message"]


def test_finite_unhealthy_runtime_counter_exits_one(evidence):
    evidence["write_checkpoint"](runtime=_runtime(amp_skipped_optimizer_steps=1))
    code, report = _run(evidence)
    assert code == 1
    assert report["failed_checks"] == ["runtime_amp_skips_zero"]


def test_nonfinite_runtime_value_is_invalid_evidence(evidence):
    evidence["write_checkpoint"](runtime=_runtime(min_amp_scale=float("nan")))
    code, report = _run(evidence)
    assert code == 2
    assert report["decision"] == "invalid_evidence"
    assert "runtime.min_amp_scale must be finite" in report["error"]["message"]


def test_extra_or_missing_candidate_log_line_is_invalid(evidence):
    line = json.dumps(_log())
    evidence["log"].write_text(f"{line}\n{line}\n", encoding="utf-8")
    code, report = _run(evidence)
    assert code == 2
    assert "exactly one" in report["error"]["message"]


def test_fixed_baseline_hash_mismatch_is_invalid(evidence, monkeypatch):
    monkeypatch.setattr(health, "BASELINE_LOG_SHA256", "0" * 64)
    code, report = _run(evidence)
    assert code == 2
    assert "SHA-256 mismatch" in report["error"]["message"]
