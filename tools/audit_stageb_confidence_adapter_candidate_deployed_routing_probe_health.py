#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Audit v43 U400 training health and diagnose adapter-capacity evidence.

This is a checkpoint/log audit only. It does not run RefCOCO or strict1607.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_deployed_routing_probe_u0400 as probe,
)


_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_candidate_"
    "tn_only_carrier_pair_probe_health.py"
)
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_pivot_v43_deployed_routing_health_base", _BASE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load health-audit base: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)


SCHEMA = "pivot.stageb.confidence_adapter_candidate_deployed_routing_probe_health/v1"
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v25"
EXPECTED_REVISION = "word_veto_candidate_asymmetric_deployed_routing_v43"
EXPECTED_GATE_CONTRACT = "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
EXPECTED_PAIR_GRADIENT_CONTRACT = "bidirectional_v1"
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "deployed_routing_probe_u0400_20260801.py"
)
EXPECTED_RANK_SHA256 = (
    "e03219d5868004aa5cb9ff4fe68f1aa94d33f1f0f6e1290cb251d12f9c914045"
)
STRICT_MANIFEST_SHA256 = (
    "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25"
)
EXPECTED_ACTIVE_ELEMENTS = 535_945
EXPECTED_UPDATES = 400
QUEUE_SIZE = 4096

PROBE_ROOT = Path(probe.OUTPUT).parent
CHECKPOINTS = {
    100: PROBE_ROOT / "intermediate_snapshots/u000100/checkpoint_iter.pth",
    200: PROBE_ROOT / "intermediate_snapshots/u000200/checkpoint_iter.pth",
    300: PROBE_ROOT / "intermediate_snapshots/u000300/checkpoint_iter.pth",
    400: Path(probe.CHECKPOINT),
}
EXPECTED_CURSORS = {
    100: (0, 200, "interval"),
    200: (0, 400, "interval"),
    300: (1, 156, "interval"),
    400: (1, 356, "max_train_iters"),
}
LOG_PATH = Path(probe.OUTPUT) / "log.txt"
INFO_PATH = Path(probe.OUTPUT) / "info.txt"

ROUTING_COUNT_FIELDS = (
    "train_fixed_text_deployed_veto_routing_positive_sample_count_unscaled",
    "train_fixed_text_deployed_veto_routing_tn_sample_count_unscaled",
)
ROUTING_RATE_FIELDS = (
    "train_fixed_text_deployed_veto_routing_positive_winner_violation_rate_unscaled",
    "train_fixed_text_deployed_veto_routing_tn_winner_violation_rate_unscaled",
    "train_fixed_text_deployed_veto_routing_positive_coverage_violation_rate_unscaled",
    "train_fixed_text_deployed_veto_routing_tn_coverage_violation_rate_unscaled",
)
ROUTING_VALUE_FIELDS = (
    "train_loss_fixed_text_deployed_veto_routing",
    "train_loss_fixed_text_deployed_veto_routing_unscaled",
    "train_fixed_text_deployed_veto_routing_positive_winner_gate_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_tn_winner_gate_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_positive_coverage_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_tn_coverage_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_positive_winner_hinge_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_tn_winner_hinge_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_positive_coverage_hinge_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_tn_coverage_hinge_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_winner_loss_mean_unscaled",
    "train_fixed_text_deployed_veto_routing_coverage_loss_mean_unscaled",
)

COMPARATOR_SUMMARIES = {
    "v30_calibrated_535k": REPO_ROOT / (
        "outputs/paper_cvpr_v1/"
        "dense_duty_adapter_candidate_calibrated_highmem_20260731/"
        "probe_evaluation/u000400_strict1607/summary.json"
    ),
    "v32_asymmetric_535k": REPO_ROOT / (
        "outputs/paper_cvpr_v1/"
        "dense_duty_adapter_candidate_asymmetric_highmem_20260731/"
        "probe_evaluation/u000400_strict1607/summary.json"
    ),
    "v33_set_attention_1329k": REPO_ROOT / (
        "outputs/paper_cvpr_v1/"
        "dense_duty_adapter_candidate_set_attention_highmem_20260731/"
        "probe_evaluation/u000400_strict1607/summary.json"
    ),
}
TAIL_COMPARATORS = {
    "v39_gate_zero_offset": REPO_ROOT / (
        "outputs/paper_cvpr_v1/"
        "dense_duty_adapter_candidate_gate_zero_offset_highmem_20260801/"
        "probe/u000400_fresh/checkpoint_iter.pth"
    ),
    "v42_tn_only_pair": REPO_ROOT / (
        "outputs/paper_cvpr_v1/"
        "dense_duty_adapter_candidate_tn_only_carrier_pair_highmem_20260801/"
        "probe/u000400_fresh/checkpoint_iter.pth"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"missing audit evidence: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"checkpoint is not a mapping: {path}")
    return payload


def _tail_endpoint(criterion: Any) -> dict[str, Any]:
    if not isinstance(criterion, Mapping):
        raise RuntimeError("checkpoint criterion state is missing")
    positive = criterion["tail_positive_queue"].float()
    negative = criterion["tail_negative_queue"].float()
    positive_count = int(criterion["tail_positive_count"])
    negative_count = int(criterion["tail_negative_count"])
    if positive_count != QUEUE_SIZE or negative_count != QUEUE_SIZE:
        raise RuntimeError("both tail queues must be full at U400")
    rank = max(1, math.ceil(0.05 * positive.numel())) - 1
    positive_q05 = float(positive.sort().values[rank].item())
    negative_q95 = float(torch.quantile(negative, 0.95).item())
    return {
        "positive_count": positive_count,
        "tn_count": negative_count,
        "positive_pointer": int(criterion["tail_positive_ptr"]),
        "tn_pointer": int(criterion["tail_negative_ptr"]),
        "positive_q05": positive_q05,
        "tn_q95": negative_q95,
        "operating_gap": positive_q05 - negative_q95,
    }


def _configure_base_audit() -> None:
    _BASE.probe = probe
    _BASE.SCHEMA = SCHEMA
    _BASE.TRAINING_CONTRACT_SCHEMA = TRAINING_CONTRACT_SCHEMA
    _BASE.EXPECTED_GRADIENT_CONTRACT = EXPECTED_PAIR_GRADIENT_CONTRACT
    _BASE.EXPECTED_CONFIG_ENTRY = EXPECTED_CONFIG_ENTRY
    expected = dict(_BASE.EXPECTED_V42_VALUES)
    expected.update(
        {
            "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
            "stage_b_dense_duty_confidence_gate_gradient_contract": (
                EXPECTED_GATE_CONTRACT
            ),
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract": (
                EXPECTED_PAIR_GRADIENT_CONTRACT
            ),
            "stage_b_dense_duty_deployed_veto_routing_weight": 0.1,
            "stage_b_dense_duty_deployed_veto_positive_max": 0.1,
            "stage_b_dense_duty_deployed_veto_tn_min": 0.9,
        }
    )
    _BASE.EXPECTED_V42_VALUES = expected
    _BASE.PAIR_COUNT_FIELDS = tuple(
        dict.fromkeys((*_BASE.PAIR_COUNT_FIELDS, *ROUTING_COUNT_FIELDS))
    )
    _BASE.PAIR_RATE_FIELDS = tuple(
        dict.fromkeys((*_BASE.PAIR_RATE_FIELDS, *ROUTING_RATE_FIELDS))
    )
    _BASE.PAIR_VALUE_FIELDS = tuple(
        dict.fromkeys((*_BASE.PAIR_VALUE_FIELDS, *ROUTING_VALUE_FIELDS))
    )
    _BASE.REQUIRED_U222_FIELDS = tuple(
        dict.fromkeys(
            (
                *_BASE.REQUIRED_U222_FIELDS,
                *ROUTING_COUNT_FIELDS,
                *ROUTING_RATE_FIELDS,
                *ROUTING_VALUE_FIELDS,
            )
        )
    )


def _checkpoint_trajectory() -> tuple[dict[str, Any], Mapping[str, Any]]:
    loaded = {update: _load_checkpoint(path) for update, path in CHECKPOINTS.items()}
    source_sha = None
    rank_sha = None
    records: dict[str, Any] = {}
    for update, payload in loaded.items():
        epoch, iteration, reason = EXPECTED_CURSORS[update]
        observed = (
            payload.get("epoch"),
            payload.get("iteration"),
            payload.get("checkpoint_reason"),
        )
        if payload.get("optimizer_updates") != update or observed != (
            epoch,
            iteration,
            reason,
        ):
            raise RuntimeError(f"U{update} checkpoint cursor drifted: {observed}")
        args = payload.get("args")
        if not isinstance(args, Mapping):
            raise RuntimeError(f"U{update} checkpoint args are missing")
        closure = args.get("stage_b_dense_duty_source_closure")
        if not isinstance(closure, Mapping):
            raise RuntimeError(f"U{update} source closure is missing")
        current_source_sha = closure.get("sha256")
        current_rank_sha = args.get("stage_b_dense_duty_rank_source_rank_sha256")
        source_sha = current_source_sha if source_sha is None else source_sha
        rank_sha = current_rank_sha if rank_sha is None else rank_sha
        if current_source_sha != source_sha or current_rank_sha != rank_sha:
            raise RuntimeError("snapshot lineage differs across U100-U400")
        records[str(update)] = {
            "file": _file_record(CHECKPOINTS[update]),
            "epoch": epoch,
            "iteration": iteration,
            "checkpoint_reason": reason,
        }
    if rank_sha != EXPECTED_RANK_SHA256:
        raise RuntimeError("snapshot rank lineage SHA drifted")

    terminal = loaded[400]
    args = terminal["args"]
    fingerprint = args.get("stage_b_dense_duty_initial_state_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise RuntimeError("terminal active-state fingerprint is missing")
    active = fingerprint.get("active")
    active_names = fingerprint.get("active_parameter_names")
    if (
        not isinstance(active, Mapping)
        or active.get("element_count") != EXPECTED_ACTIVE_ELEMENTS
        or not isinstance(active_names, list)
    ):
        raise RuntimeError("terminal active parameter contract drifted")
    active_set = set(active_names)
    model = terminal.get("model")
    optimizer = terminal.get("optimizer")
    if not isinstance(model, Mapping) or not isinstance(optimizer, Mapping):
        raise RuntimeError("terminal model or optimizer state is missing")
    ordered_names = [name for name in model if name in active_set]
    if len(ordered_names) != len(active_names):
        raise RuntimeError("could not recover active parameter order")
    state_ids = set(optimizer.get("state", {}))
    effective_names = [
        name for index, name in enumerate(ordered_names) if index in state_ids
    ]
    dormant_names = [
        name for index, name in enumerate(ordered_names) if index not in state_ids
    ]
    effective_elements = sum(model[name].numel() for name in effective_names)
    dormant_elements = sum(model[name].numel() for name in dormant_names)

    intervals: dict[str, Any] = {}
    for left, right in ((100, 200), (200, 300), (300, 400)):
        delta_square = 0.0
        base_square = 0.0
        changed = 0
        total = 0
        max_abs = 0.0
        for name in effective_names:
            before = loaded[left]["model"][name].float()
            after = loaded[right]["model"][name].float()
            delta = after - before
            delta_square += float(delta.square().sum().item())
            base_square += float(before.square().sum().item())
            changed += int((delta != 0).sum().item())
            total += delta.numel()
            max_abs = max(max_abs, float(delta.abs().max().item()))
        intervals[f"u{left}_to_u{right}"] = {
            "l2_delta": math.sqrt(delta_square),
            "relative_l2_delta": math.sqrt(delta_square / base_square),
            "changed_elements": changed,
            "total_effective_elements": total,
            "changed_fraction": changed / total,
            "max_abs_delta": max_abs,
        }

    return (
        {
            "snapshots": records,
            "source_closure_sha256": source_sha,
            "rank_sha256": rank_sha,
            "declared_active_parameter_tensors": len(active_names),
            "declared_active_elements": int(active["element_count"]),
            "optimizer_state_parameter_tensors": len(effective_names),
            "effective_elements": effective_elements,
            "effective_fraction": effective_elements / EXPECTED_ACTIVE_ELEMENTS,
            "dormant_elements": dormant_elements,
            "dormant_parameters": [
                {
                    "name": name,
                    "elements": model[name].numel(),
                    "shape": list(model[name].shape),
                }
                for name in dormant_names
            ],
            "parameter_motion": intervals,
        },
        terminal,
    )


_METRIC_RE = re.compile(
    r"(?<!\S)([A-Za-z0-9_]+):\s+(-?\d+(?:\.\d+)?)"
    r"\s+\((-?\d+(?:\.\d+)?)\)"
)


def _near_terminal_metrics() -> dict[str, Any]:
    lines = [
        line for line in INFO_PATH.read_text(encoding="utf-8").splitlines()
        if "Epoch:" in line
    ]
    if not lines:
        raise RuntimeError("training info log has no metric lines")
    last = lines[-1]
    cursor = re.search(r"Epoch: \[(\d+)\]\s+\[\s*(\d+)/444\]", last)
    if cursor is None:
        raise RuntimeError("could not parse the final logged cursor")
    metrics = {
        match.group(1): {
            "current": float(match.group(2)),
            "epoch_mean": float(match.group(3)),
        }
        for match in _METRIC_RE.finditer(last)
    }
    return {
        "epoch": int(cursor.group(1)),
        "physical_forward_index": int(cursor.group(2)),
        "approximate_optimizer_update": (
            int(cursor.group(1)) * 444 + int(cursor.group(2))
        )
        // 2,
        "metrics": metrics,
        "evidence": _file_record(INFO_PATH),
    }


def _capacity_comparators() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, summary_path in COMPARATOR_SUMMARIES.items():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        runs = payload.get("tn")
        if not isinstance(runs, list) or len(runs) != 1:
            raise RuntimeError(f"invalid comparator summary: {summary_path}")
        run = runs[0]
        if (
            run.get("manifest_n") != 1607
            or run.get("source_manifest_sha256") != STRICT_MANIFEST_SHA256
        ):
            raise RuntimeError(f"comparator is not strict1607-bound: {summary_path}")
        checkpoint = _load_checkpoint(REPO_ROOT / run["checkpoint"])
        active = checkpoint["args"][
            "stage_b_dense_duty_initial_state_fingerprint"
        ]["active"]
        fpr = float(run["fpr95tpr"])
        records[name] = {
            "active_elements": int(active["element_count"]),
            "fpr95": fpr,
            "false_accepts": int(round(fpr * 1607)),
            "optimizer_updates": int(checkpoint["optimizer_updates"]),
            "summary": _file_record(summary_path),
        }
    return records


def _tail_comparators(v43_endpoint: Mapping[str, Any]) -> dict[str, Any]:
    result = {"v43_deployed_routing": dict(v43_endpoint)}
    for name, path in TAIL_COMPARATORS.items():
        payload = _load_checkpoint(path)
        result[name] = _tail_endpoint(payload.get("criterion"))
    return result


def audit() -> dict[str, Any]:
    _configure_base_audit()
    base = _BASE.audit()
    trajectory, terminal = _checkpoint_trajectory()
    runtime = terminal["args"].get("stage_b_dense_duty_runtime_audit")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("terminal runtime audit is missing")
    u222 = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    if u222.get("train_optimizer_updates") != 222:
        raise RuntimeError("epoch-zero trajectory log is not U222")
    near_terminal = _near_terminal_metrics()
    endpoint = _tail_endpoint(terminal.get("criterion"))
    comparators = _capacity_comparators()
    tail_comparators = _tail_comparators(endpoint)

    mechanical_exclusions = {
        "u400_positive_q05_in_operating_range",
        "u400_tn_q95_in_operating_range",
        "u400_operating_gap_in_operating_range",
    }
    mechanical_failed = [
        name for name in base["failed_checks"] if name not in mechanical_exclusions
    ]
    final_metrics = near_terminal["metrics"]
    routing_unresolved = (
        endpoint["operating_gap"] < -0.2
        and final_metrics[
            "fixed_text_deployed_veto_routing_tn_winner_gate_mean_unscaled"
        ]["epoch_mean"]
        < 0.9
        and final_metrics[
            "fixed_text_deployed_veto_routing_tn_coverage_mean_unscaled"
        ]["epoch_mean"]
        < 0.9
    )
    capacity_engaged = (
        trajectory["effective_fraction"] > 0.99
        and trajectory["parameter_motion"]["u300_to_u400"][
            "changed_fraction"
        ]
        > 0.99
        and runtime.get("successful_optimizer_steps") == EXPECTED_UPDATES
        and runtime.get("zero_gradient_successful_steps") == 0
        and float(runtime.get("last_active_grad_norm_preclip", 0.0)) > 1.0
    )
    larger = comparators["v33_set_attention_1329k"]
    smaller = comparators["v32_asymmetric_535k"]
    larger_capacity_not_better = (
        larger["active_elements"] > 2 * smaller["active_elements"]
        and larger["false_accepts"] >= smaller["false_accepts"]
    )
    mechanically_healthy = not mechanical_failed
    verdict = (
        "optimization_and_absolute_readout_conflict_not_raw_capacity"
        if mechanically_healthy
        and routing_unresolved
        and capacity_engaged
        and larger_capacity_not_better
        else "capacity_cause_not_resolved_by_audit"
    )

    return {
        "schema": SCHEMA,
        "scope": "training_health_and_capacity_evidence_only_no_new_dataset_evaluation",
        "checkpoint": _file_record(Path(probe.CHECKPOINT)),
        "log": _file_record(LOG_PATH),
        "base_health_audit": {
            "failed_checks": base["failed_checks"],
            "mechanical_failed_checks": mechanical_failed,
            "mechanically_healthy": mechanically_healthy,
            "endpoint_checks_are_training_objective_diagnostics": sorted(
                mechanical_exclusions
            ),
        },
        "runtime": dict(runtime),
        "trajectory_u222": {
            key: value
            for key, value in u222.items()
            if key.startswith("train_loss_fixed_text_")
            or key.startswith("train_fixed_text_deployed_veto_routing_")
            or key.startswith("train_fixed_text_tail_queue_")
            or key.startswith("train_grad_")
            or key in {"train_optimizer_updates", "train_amp_step_skipped"}
        },
        "near_terminal": near_terminal,
        "tail_endpoint_u400": endpoint,
        "tail_endpoint_comparison": tail_comparators,
        "parameter_utilization": trajectory,
        "historical_capacity_control": comparators,
        "diagnosis": {
            "mechanically_healthy": mechanically_healthy,
            "routing_objective_unresolved": routing_unresolved,
            "adapter_capacity_engaged": capacity_engaged,
            "larger_capacity_not_better_in_historical_control": (
                larger_capacity_not_better
            ),
            "adapter_too_light_is_primary_bottleneck": False,
            "raw_parameter_capacity_verdict": "not_supported",
            "structural_verdict": (
                "the likely bottleneck is the absolute-confidence readout, "
                "routing, and competing gradients inside the confidence path; "
                "blindly widening the adapter is not supported"
            ),
            "primary_diagnosis": verdict,
        },
        "decision": "pause_before_strict1607_as_requested",
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROBE_ROOT / "u000400_deployed_routing_health_capacity_audit.json",
    )
    args = parser.parse_args(argv)
    try:
        report = audit()
        exit_code = 0
    except Exception as error:
        report = {
            "schema": SCHEMA,
            "decision": "invalid_evidence",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        exit_code = 2
    _atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
