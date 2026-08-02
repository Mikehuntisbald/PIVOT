#!/usr/bin/env python3
"""Audit the U300 word-veto confidence probe before formal training.

Exit codes:
  0: all evidence is valid and every pre-registered health check passes;
  1: evidence is valid, but at least one loss-health check fails;
  2: evidence is missing, malformed, unbound, or not the terminal U300 probe.

This is a training-health gate, not a RefCOCO or TN performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_confidence_adapter_veto_probe as probe  # noqa: E402


SCHEMA = "pivot.stageb.confidence_adapter_veto_probe_health/v1"
EXPECTED_UPDATES = 300
EXPECTED_LOG_UPDATES = 222
QUEUE_SIZE = 4096

BASELINE_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_packed_highmem_20260730/formal/confidence"
)
BASELINE_LOG = BASELINE_ROOT / "log.txt"
BASELINE_U222_CHECKPOINT = BASELINE_ROOT / "checkpoint0000.pth"
BASELINE_U444_CHECKPOINT = BASELINE_ROOT / "checkpoint0001.pth"
BASELINE_LOG_SHA256 = (
    "b81185fbb28c2c136e0f6545579b79b472473995107b98c7af945300c1f25658"
)
BASELINE_U222_CHECKPOINT_SHA256 = (
    "4d6179e9161c0efc48a3be5cf7bd36b718bdefd14d8946fd6a40c85819f7de49"
)
BASELINE_U444_CHECKPOINT_SHA256 = (
    "609cd32d80ee00f185222704ff8d5e16414047a6304fed256e9cd79460c25bee"
)

OLD_U444_TOKEN_LOSS = 2.3132341
OLD_U222_POSITIVE_Q05 = -4.6322799
OLD_U222_TN_Q95 = 0.88048965
OLD_U222_OPERATING_GAP = -5.5127695
OLD_U222_TRUST_VIOLATION = 0.5708145

REQUIRED_U222_FIELDS = (
    "train_loss_fixed_text_token_unscaled",
    "train_loss_fixed_text_local_absolute_unscaled",
    "train_loss_fixed_text_global_tn_negative_unscaled",
    "train_loss_fixed_text_tail_queue_unscaled",
    "train_loss_fixed_text_global_tn_tail",
    "train_fixed_text_tail_queue_positive_count_unscaled",
    "train_fixed_text_tail_queue_negative_count_unscaled",
    "train_fixed_text_tail_queue_threshold_valid_unscaled",
    "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled",
    "train_stage_b_dense_confidence_positive_delta_mean_unscaled",
    "train_stage_b_dense_confidence_tn_delta_mean_unscaled",
    "train_stage_b_dense_confidence_positive_veto_gate_mean_unscaled",
    "train_stage_b_dense_confidence_tn_veto_gate_mean_unscaled",
    "train_amp_step_skipped",
)
QUEUE_KEYS = frozenset(
    {
        "tail_positive_queue",
        "tail_negative_queue",
        "tail_positive_ptr",
        "tail_negative_ptr",
        "tail_positive_count",
        "tail_negative_count",
    }
)
RUNTIME_SCHEMA = "pivot.stageb.dense_duty_runtime_audit/v1"


class ProbeHealthEvidenceError(ValueError):
    """Raised when the audit inputs cannot support a health decision."""


def _candidate_log() -> Path:
    return Path(probe.OUTPUT) / "log.txt"


def _candidate_checkpoint() -> Path:
    return Path(probe.CHECKPOINT)


def _default_output() -> Path:
    # Never publish preflight evidence inside the atomic training directory.
    # A fresh/invalid audit must not turn a resumable fresh run into a
    # non-empty output without a checkpoint.
    return Path(probe.OUTPUT).parent / "u000300_probe_health_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if path.is_symlink():
        raise ProbeHealthEvidenceError(f"{label} must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ProbeHealthEvidenceError(f"{label} is missing or is not a real file: {path}")
    sha256 = _sha256(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ProbeHealthEvidenceError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {sha256}"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
    }


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ProbeHealthEvidenceError(f"{label} must be an exact integer")
    return int(value)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeHealthEvidenceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProbeHealthEvidenceError(f"{label} must be finite")
    return result


def _strict_u222_log(path: Path) -> tuple[dict[str, float | int], dict[str, Any]]:
    record = _file_record(path, label="candidate U222 log")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProbeHealthEvidenceError(f"could not read candidate U222 log: {error}") from error
    if len(lines) != 1 or not lines[0].strip():
        raise ProbeHealthEvidenceError(
            "candidate log must contain exactly one non-empty U222 JSON line"
        )
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ProbeHealthEvidenceError(f"candidate U222 log is invalid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ProbeHealthEvidenceError("candidate U222 log line must be a JSON object")
    updates = _exact_int(payload.get("train_optimizer_updates"), "train_optimizer_updates")
    if updates != EXPECTED_LOG_UPDATES:
        raise ProbeHealthEvidenceError(
            f"candidate log update must be U{EXPECTED_LOG_UPDATES}, got U{updates}"
        )
    values: dict[str, float | int] = {"train_optimizer_updates": updates}
    for field in REQUIRED_U222_FIELDS:
        values[field] = _finite_float(payload.get(field), field)

    for field in (
        "train_fixed_text_tail_queue_positive_count_unscaled",
        "train_fixed_text_tail_queue_negative_count_unscaled",
    ):
        if not 0.0 <= float(values[field]) <= float(QUEUE_SIZE):
            raise ProbeHealthEvidenceError(f"{field} must be in [0, {QUEUE_SIZE}]")
    for field in (
        "train_fixed_text_tail_queue_threshold_valid_unscaled",
        "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled",
        "train_stage_b_dense_confidence_positive_veto_gate_mean_unscaled",
        "train_stage_b_dense_confidence_tn_veto_gate_mean_unscaled",
    ):
        if not 0.0 <= float(values[field]) <= 1.0:
            raise ProbeHealthEvidenceError(f"{field} must be in [0, 1]")
    if float(values["train_amp_step_skipped"]) < 0.0:
        raise ProbeHealthEvidenceError("train_amp_step_skipped must be non-negative")
    return values, record


def _load_checkpoint_mmap(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    record = _file_record(path, label="candidate U300 checkpoint")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:
        raise ProbeHealthEvidenceError(
            f"could not mmap-load candidate U300 checkpoint: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ProbeHealthEvidenceError("candidate U300 checkpoint must be a mapping")
    return payload, record


def _queue_scalar(state: Mapping[str, Any], name: str) -> int:
    value = state.get(name)
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.int64
        or value.numel() != 1
    ):
        raise ProbeHealthEvidenceError(f"criterion.{name} must be one int64 tensor")
    return int(value.item())


def _queue_tensor(state: Mapping[str, Any], name: str) -> torch.Tensor:
    value = state.get(name)
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or tuple(value.shape) != (QUEUE_SIZE,)
    ):
        raise ProbeHealthEvidenceError(
            f"criterion.{name} must be float32 with shape ({QUEUE_SIZE},)"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ProbeHealthEvidenceError(f"criterion.{name} must be finite")
    return value


def _exact_lower_tail_operating_threshold(
    values: torch.Tensor,
    lower_tail_fraction: float,
) -> torch.Tensor:
    """Equivalent to the criterion's exact score>=threshold order statistic."""
    values = values.float().reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all().item()):
        raise ProbeHealthEvidenceError("positive tail queue must be non-empty and finite")
    target_tpr = 1.0 - float(lower_tail_fraction)
    accepted = max(1, int(math.ceil(target_tpr * int(values.numel()))))
    kth = int(values.numel()) - accepted + 1
    return torch.kthvalue(values, kth).values


def _audit_endpoint(criterion: Any) -> dict[str, Any]:
    if not isinstance(criterion, Mapping) or set(criterion) != QUEUE_KEYS:
        raise ProbeHealthEvidenceError(
            "candidate criterion must contain the exact six tail-queue tensors"
        )
    positive = _queue_tensor(criterion, "tail_positive_queue")
    negative = _queue_tensor(criterion, "tail_negative_queue")
    counts = {
        "positive": _queue_scalar(criterion, "tail_positive_count"),
        "tn": _queue_scalar(criterion, "tail_negative_count"),
    }
    pointers = {
        "positive": _queue_scalar(criterion, "tail_positive_ptr"),
        "tn": _queue_scalar(criterion, "tail_negative_ptr"),
    }
    if counts != {"positive": QUEUE_SIZE, "tn": QUEUE_SIZE}:
        raise ProbeHealthEvidenceError(
            f"both U300 tail queues must have count {QUEUE_SIZE}, got {counts}"
        )
    for label, pointer in pointers.items():
        if not 0 <= pointer < QUEUE_SIZE:
            raise ProbeHealthEvidenceError(
                f"U300 {label} tail queue pointer must be in [0, {QUEUE_SIZE})"
            )

    positive_q05 = float(
        _exact_lower_tail_operating_threshold(positive, 0.05).item()
    )
    tn_q95 = float(torch.quantile(negative.float(), 0.95).item())
    operating_gap = positive_q05 - tn_q95
    if not all(math.isfinite(value) for value in (positive_q05, tn_q95, operating_gap)):
        raise ProbeHealthEvidenceError("derived U300 tail metrics must be finite")
    return {
        "queue_size": QUEUE_SIZE,
        "queue_counts": counts,
        "queue_pointers": pointers,
        "positive_q05": positive_q05,
        "tn_q95": tn_q95,
        "operating_gap": operating_gap,
        "positive_q05_contract": "exact_score_ge_order_statistic_tpr95",
        "tn_q95_contract": "torch.quantile_linear_q0.95",
    }


def _audit_runtime(args: Any) -> dict[str, Any]:
    if not isinstance(args, Mapping):
        raise ProbeHealthEvidenceError("candidate checkpoint args must be a mapping")
    runtime = args.get("stage_b_dense_duty_runtime_audit")
    if not isinstance(runtime, Mapping):
        raise ProbeHealthEvidenceError("candidate checkpoint lacks the runtime audit")
    if runtime.get("schema") != RUNTIME_SCHEMA:
        raise ProbeHealthEvidenceError(
            f"runtime audit schema must be {RUNTIME_SCHEMA!r}"
        )
    counter_fields = (
        "optimizer_step_boundaries",
        "successful_optimizer_steps",
        "amp_skipped_optimizer_steps",
        "nonfinite_gradient_boundaries",
        "zero_gradient_successful_steps",
    )
    result: dict[str, Any] = {"schema": RUNTIME_SCHEMA}
    for field in counter_fields:
        value = _exact_int(runtime.get(field), f"runtime.{field}")
        if value < 0:
            raise ProbeHealthEvidenceError(f"runtime.{field} must be non-negative")
        result[field] = value
    result["min_amp_scale"] = _finite_float(
        runtime.get("min_amp_scale"), "runtime.min_amp_scale"
    )
    for field in (
        "last_amp_scale",
        "max_active_grad_norm_preclip",
        "last_active_grad_norm_preclip",
    ):
        if field in runtime:
            result[field] = _finite_float(runtime[field], f"runtime.{field}")
    return result


def _check(observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def _health_checks(
    trajectory: Mapping[str, float | int],
    endpoint: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    token = float(trajectory["train_loss_fixed_text_token_unscaled"])
    trust = float(
        trajectory[
            "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled"
        ]
    )
    positive_delta = float(
        trajectory["train_stage_b_dense_confidence_positive_delta_mean_unscaled"]
    )
    tn_delta = float(
        trajectory["train_stage_b_dense_confidence_tn_delta_mean_unscaled"]
    )
    separation = positive_delta - tn_delta
    positive_gate = float(
        trajectory[
            "train_stage_b_dense_confidence_positive_veto_gate_mean_unscaled"
        ]
    )
    tn_gate = float(
        trajectory["train_stage_b_dense_confidence_tn_veto_gate_mean_unscaled"]
    )
    global_tail = float(trajectory["train_loss_fixed_text_global_tn_tail"])
    amp_log_skips = float(trajectory["train_amp_step_skipped"])
    positive_q05 = float(endpoint["positive_q05"])
    tn_q95 = float(endpoint["tn_q95"])
    gap = float(endpoint["operating_gap"])

    return {
        "u222_token_below_old_u444": _check(
            token, f"< {OLD_U444_TOKEN_LOSS}", token < OLD_U444_TOKEN_LOSS
        ),
        "u222_positive_trust_violation": _check(
            trust, "<= 0.1", trust <= 0.1
        ),
        "u222_positive_delta_preserved": _check(
            positive_delta, ">= -0.05", positive_delta >= -0.05
        ),
        "u222_tn_delta_negative": _check(tn_delta, "< 0", tn_delta < 0.0),
        "u222_delta_separation": _check(
            separation, ">= 0.02", separation >= 0.02
        ),
        "u222_tn_gate_exceeds_positive_gate": _check(
            {"positive": positive_gate, "tn": tn_gate},
            "tn > positive",
            tn_gate > positive_gate,
        ),
        "u222_duplicate_global_tn_tail_disabled": _check(
            global_tail, "== 0", global_tail == 0.0
        ),
        "u222_log_amp_skips_zero": _check(
            amp_log_skips, "== 0", amp_log_skips == 0.0
        ),
        "u300_positive_q05_improved": _check(
            positive_q05,
            f"> {OLD_U222_POSITIVE_Q05}",
            positive_q05 > OLD_U222_POSITIVE_Q05,
        ),
        "u300_tn_q95_improved": _check(
            tn_q95, f"< {OLD_U222_TN_Q95}", tn_q95 < OLD_U222_TN_Q95
        ),
        "u300_operating_gap_improved": _check(
            gap,
            f"> {OLD_U222_OPERATING_GAP}",
            gap > OLD_U222_OPERATING_GAP,
        ),
        "runtime_all_boundaries_succeeded": _check(
            {
                "boundaries": runtime["optimizer_step_boundaries"],
                "successful": runtime["successful_optimizer_steps"],
            },
            f"boundaries == successful == {EXPECTED_UPDATES}",
            runtime["optimizer_step_boundaries"] == EXPECTED_UPDATES
            and runtime["successful_optimizer_steps"] == EXPECTED_UPDATES,
        ),
        "runtime_amp_skips_zero": _check(
            runtime["amp_skipped_optimizer_steps"],
            "== 0",
            runtime["amp_skipped_optimizer_steps"] == 0,
        ),
        "runtime_nonfinite_gradients_zero": _check(
            runtime["nonfinite_gradient_boundaries"],
            "== 0",
            runtime["nonfinite_gradient_boundaries"] == 0,
        ),
        "runtime_zero_gradient_steps_zero": _check(
            runtime["zero_gradient_successful_steps"],
            "== 0",
            runtime["zero_gradient_successful_steps"] == 0,
        ),
        "runtime_amp_scale_floor": _check(
            runtime["min_amp_scale"],
            ">= 256",
            float(runtime["min_amp_scale"]) >= 256.0,
        ),
    }


def audit() -> dict[str, Any]:
    state = probe.inspect()
    if not isinstance(state, Mapping):
        raise ProbeHealthEvidenceError("probe controller inspect result must be a mapping")
    for field, expected in (
        ("status", "terminal"),
        ("action", "complete"),
        ("updates", EXPECTED_UPDATES),
    ):
        if state.get(field) != expected:
            raise ProbeHealthEvidenceError(
                f"probe controller requires {field}={expected!r}, got {state.get(field)!r}"
            )

    baseline = {
        "id": "u4388_pre_veto_packed_highmem_20260730",
        "log": _file_record(
            BASELINE_LOG,
            label="fixed old baseline log",
            expected_sha256=BASELINE_LOG_SHA256,
        ),
        "u222_checkpoint": _file_record(
            BASELINE_U222_CHECKPOINT,
            label="fixed old U222 checkpoint",
            expected_sha256=BASELINE_U222_CHECKPOINT_SHA256,
        ),
        "u444_checkpoint": _file_record(
            BASELINE_U444_CHECKPOINT,
            label="fixed old U444 checkpoint",
            expected_sha256=BASELINE_U444_CHECKPOINT_SHA256,
        ),
        "metrics": {
            "u444_token_loss": OLD_U444_TOKEN_LOSS,
            "u222_positive_q05": OLD_U222_POSITIVE_Q05,
            "u222_tn_q95": OLD_U222_TN_Q95,
            "u222_operating_gap": OLD_U222_OPERATING_GAP,
            "u222_positive_trust_violation_rate": OLD_U222_TRUST_VIOLATION,
        },
    }

    trajectory, log_record = _strict_u222_log(_candidate_log())
    payload, checkpoint_record = _load_checkpoint_mmap(_candidate_checkpoint())
    checkpoint_updates = _exact_int(
        payload.get("optimizer_updates"), "checkpoint.optimizer_updates"
    )
    if checkpoint_updates != EXPECTED_UPDATES:
        raise ProbeHealthEvidenceError("candidate checkpoint optimizer_updates must be 300")
    if payload.get("checkpoint_reason") != "max_train_iters":
        raise ProbeHealthEvidenceError(
            "candidate checkpoint reason must be max_train_iters"
        )
    endpoint = _audit_endpoint(payload.get("criterion"))
    runtime = _audit_runtime(payload.get("args"))
    checks = _health_checks(trajectory, endpoint, runtime)
    failed = [name for name, value in checks.items() if not value["passed"]]

    return {
        "schema": SCHEMA,
        "baseline": baseline,
        "candidate": {
            "controller": {
                "status": state["status"],
                "action": state["action"],
                "updates": state["updates"],
                **(
                    {"rank_sha256": state["rank_sha256"]}
                    if isinstance(state.get("rank_sha256"), str)
                    else {}
                ),
            },
            "log": log_record,
            "checkpoint": checkpoint_record,
        },
        "trajectory_u222": trajectory,
        "endpoint_u300": endpoint,
        "runtime": runtime,
        "checks": checks,
        "failed_checks": failed,
        "decision": (
            "healthy_for_strict1607_diagnostic"
            if not failed
            else "unhealthy_do_not_run_diagnostic"
        ),
        "scope": "training_health_only_not_ref_or_tn_performance",
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise OSError(f"audit output must not be a symlink: {path}")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_default_output())
    args = parser.parse_args(argv)
    try:
        report = audit()
        exit_code = (
            0
            if report["decision"] == "healthy_for_strict1607_diagnostic"
            else 1
        )
    except (ProbeHealthEvidenceError, OSError, RuntimeError) as error:
        report = {
            "schema": SCHEMA,
            "decision": "invalid_evidence",
            "failed_checks": [],
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "scope": "training_health_only_not_ref_or_tn_performance",
        }
        exit_code = 2
    try:
        _atomic_write_json(args.output, report)
    except OSError as error:
        print(f"[FAIL] could not publish audit report: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return exit_code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
