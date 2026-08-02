#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Audit terminal U400 health for the v41 role-complete carrier probe.

This is a training-health gate. It validates the sealed v23 training surface,
replays packed-forward geometry from the terminal cursor, and verifies that the
role-complete carrier supervision covered every eligible direct-trace row.
It does not make a RefCOCO or TN performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_role_complete_carrier_probe_u0400 as probe,
)
from util.stage_b_dense_duty_audit import (  # noqa: E402
    CONFIG_SOURCE_CLOSURE_SCHEMA,
    validate_source_closure,
)


_BASE_PATH = REPO_ROOT / "tools/audit_stageb_confidence_adapter_veto_probe_health.py"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_role_complete_carrier_health_base",
    _BASE_PATH,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load health auditor base: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)


SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_role_complete_carrier_"
    "probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v23"
SOURCE_CLOSURE_SCHEMA = "pivot.stageb.dense_duty_source_closure/v1"
CODE_SOURCE_CLOSURE_SCHEMA = "pivot.stageb.dense_duty_code_source_closure/v1"
RUNTIME_SCHEMA = _BASE.RUNTIME_SCHEMA
EXPECTED_UPDATES = 400
EXPECTED_LOG_UPDATES = 222
QUEUE_SIZE = 4096
EXPECTED_SCOPE = (
    "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
)
EXPECTED_OWNERSHIP = "rank_tower_stopgrad_token_adapter_two_phase"
EXPECTED_PACKED_VALUES = {
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "stage_b_dense_duty_forward_pack_factor": 2,
    "stage_b_dense_duty_logical_loss_batch_size": 16,
    "stage_b_dense_duty_expected_forward_batch_size": 32,
    "stage_b_dense_duty_expected_logical_batches_per_epoch": 887,
    "stage_b_dense_duty_expected_physical_forwards_per_epoch": 444,
    "stage_b_dense_duty_expected_gradient_accumulation_steps": 2,
    "stage_b_dense_duty_confidence_expected_optimizer_updates": 400,
    "max_train_iters": 400,
}
EXPECTED_TERMINAL_EPOCH = 1
EXPECTED_TERMINAL_ITERATION = 356
EXPECTED_TOTAL_PHYSICAL_FORWARDS = 800
EXPECTED_TOTAL_LOGICAL_BATCHES = 1599

ROLE_CARRIER_FIELDS = (
    "train_fixed_text_token_role_carrier_pair_selected_count_unscaled",
    "train_fixed_text_token_role_carrier_positive_added_count_unscaled",
    "train_fixed_text_token_role_carrier_tn_added_count_unscaled",
    "train_fixed_text_token_role_carrier_positive_target_overlap_count_unscaled",
    "train_fixed_text_token_role_carrier_tn_target_overlap_count_unscaled",
    "train_fixed_text_token_edit_carrier_selected_count_unscaled",
    "train_fixed_text_token_edit_carrier_added_count_unscaled",
    "train_fixed_text_token_edit_carrier_target_overlap_count_unscaled",
    "train_fixed_text_confidence_tn_train_eligible_count_unscaled",
    "train_fixed_text_confidence_tn_train_excluded_count_unscaled",
    "train_fixed_text_token_all_negative_count_unscaled",
    "train_fixed_text_token_provenance_valid_count_unscaled",
    "train_fixed_text_token_direct_trace_valid_count_unscaled",
    "train_fixed_text_global_tn_sample_count_unscaled",
)
REQUIRED_U222_FIELDS = _BASE.REQUIRED_U222_FIELDS + ROLE_CARRIER_FIELDS

BASELINE_LOG = _BASE.BASELINE_LOG
BASELINE_U222_CHECKPOINT = _BASE.BASELINE_U222_CHECKPOINT
BASELINE_U444_CHECKPOINT = _BASE.BASELINE_U444_CHECKPOINT
BASELINE_LOG_SHA256 = _BASE.BASELINE_LOG_SHA256
BASELINE_U222_CHECKPOINT_SHA256 = _BASE.BASELINE_U222_CHECKPOINT_SHA256
BASELINE_U444_CHECKPOINT_SHA256 = _BASE.BASELINE_U444_CHECKPOINT_SHA256
OLD_U444_TOKEN_LOSS = _BASE.OLD_U444_TOKEN_LOSS

ProbeHealthEvidenceError = _BASE.ProbeHealthEvidenceError
_file_record = _BASE._file_record
_exact_int = _BASE._exact_int
_finite_float = _BASE._finite_float
_queue_scalar = _BASE._queue_scalar
_queue_tensor = _BASE._queue_tensor
_exact_lower_tail_operating_threshold = (
    _BASE._exact_lower_tail_operating_threshold
)
_audit_runtime = _BASE._audit_runtime
_check = _BASE._check
_atomic_write_json = _BASE._atomic_write_json


def _candidate_log() -> Path:
    return Path(probe.OUTPUT) / "log.txt"


def _candidate_checkpoint() -> Path:
    return Path(probe.CHECKPOINT)


def _default_output() -> Path:
    return Path(probe.OUTPUT).parent / (
        "u000400_role_complete_carrier_probe_health_audit.json"
    )


def _canonical_sha256(value: Any, *, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ProbeHealthEvidenceError(
            f"{label} cannot be encoded canonically: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_u222_log(path: Path) -> tuple[dict[str, float | int], dict[str, Any]]:
    record = _file_record(path, label="candidate v41 U222 log")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProbeHealthEvidenceError(
            f"could not read candidate v41 U222 log: {error}"
        ) from error
    if len(lines) != 1 or not lines[0].strip():
        raise ProbeHealthEvidenceError(
            "candidate v41 log must contain exactly one non-empty U222 JSON line"
        )
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ProbeHealthEvidenceError(
            f"candidate v41 U222 log is invalid JSON: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ProbeHealthEvidenceError(
            "candidate v41 U222 log line must be a JSON object"
        )
    updates = _exact_int(
        payload.get("train_optimizer_updates"), "train_optimizer_updates"
    )
    if updates != EXPECTED_LOG_UPDATES:
        raise ProbeHealthEvidenceError(
            f"candidate v41 log update must be U{EXPECTED_LOG_UPDATES}, got U{updates}"
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
    for field in ROLE_CARRIER_FIELDS:
        if not 0.0 <= float(values[field]) <= 16.0:
            raise ProbeHealthEvidenceError(f"{field} must be in [0, 16]")
    if float(values["train_amp_step_skipped"]) < 0.0:
        raise ProbeHealthEvidenceError("train_amp_step_skipped must be non-negative")
    return values, record


def _load_checkpoint_mmap(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    record = _file_record(path, label="candidate v41 U400 checkpoint")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:
        raise ProbeHealthEvidenceError(
            "could not mmap-load candidate v41 U400 checkpoint: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ProbeHealthEvidenceError(
            "candidate v41 U400 checkpoint must be a mapping"
        )
    return payload, record


def _audit_endpoint_u400(criterion: Any) -> dict[str, Any]:
    if not isinstance(criterion, Mapping) or set(criterion) != _BASE.QUEUE_KEYS:
        raise ProbeHealthEvidenceError(
            "candidate v41 criterion must contain the exact six tail-queue tensors"
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
            f"both v41 U400 tail queues must have count {QUEUE_SIZE}, got {counts}"
        )
    for label, pointer in pointers.items():
        if not 0 <= pointer < QUEUE_SIZE:
            raise ProbeHealthEvidenceError(
                f"v41 U400 {label} tail queue pointer must be in [0, {QUEUE_SIZE})"
            )

    positive_q05 = float(
        _exact_lower_tail_operating_threshold(positive, 0.05).item()
    )
    tn_q95 = float(torch.quantile(negative.float(), 0.95).item())
    operating_gap = positive_q05 - tn_q95
    if not all(
        math.isfinite(value) for value in (positive_q05, tn_q95, operating_gap)
    ):
        raise ProbeHealthEvidenceError(
            "derived v41 U400 tail metrics must be finite"
        )
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


def _exact_contract_value(
    args: Mapping[str, Any],
    values: Mapping[str, Any],
    name: str,
    expected: Any,
) -> None:
    for label, source in (("checkpoint args", args), ("training contract", values)):
        observed = source.get(name)
        if type(observed) is not type(expected) or observed != expected:
            raise ProbeHealthEvidenceError(
                f"{label} requires {name}={expected!r}, got {observed!r}"
            )


def _audit_packed_forward(
    payload: Mapping[str, Any],
    args: Any,
    *,
    trajectory_updates: int,
) -> dict[str, Any]:
    if not isinstance(args, Mapping):
        raise ProbeHealthEvidenceError("candidate checkpoint args must be a mapping")
    contract = args.get("stage_b_dense_duty_training_contract")
    if not isinstance(contract, Mapping):
        raise ProbeHealthEvidenceError("candidate v41 checkpoint lacks training contract")
    if set(contract) != {"schema", "sha256", "values"}:
        raise ProbeHealthEvidenceError(
            "candidate v41 training contract has an invalid structure"
        )
    if contract.get("schema") != TRAINING_CONTRACT_SCHEMA:
        raise ProbeHealthEvidenceError(
            f"v41 training contract schema must be {TRAINING_CONTRACT_SCHEMA!r}"
        )
    values = contract.get("values")
    if not isinstance(values, Mapping):
        raise ProbeHealthEvidenceError("v41 training contract values are missing")
    contract_sha256 = contract.get("sha256")
    if not _valid_sha256(contract_sha256) or contract_sha256 != _canonical_sha256(
        values, label="v41 training contract values"
    ):
        raise ProbeHealthEvidenceError("v41 training contract SHA-256 is invalid")

    _exact_contract_value(
        args,
        values,
        "stage_b_v22_score_ownership",
        EXPECTED_OWNERSHIP,
    )
    _exact_contract_value(
        args,
        values,
        "stage_b_v21_token_edit_query_scope",
        EXPECTED_SCOPE,
    )
    for name, expected in EXPECTED_PACKED_VALUES.items():
        _exact_contract_value(args, values, name, expected)

    closure_value = values.get("stage_b_dense_duty_source_closure")
    if not isinstance(closure_value, Mapping) or args.get(
        "stage_b_dense_duty_source_closure"
    ) != closure_value:
        raise ProbeHealthEvidenceError(
            "v41 training contract source closure is missing or unbound"
        )
    try:
        closure = validate_source_closure(closure_value)
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(
            f"v41 source closure is invalid: {error}"
        ) from error
    code = closure.get("code")
    if not isinstance(code, Mapping):
        raise ProbeHealthEvidenceError("v41 code source closure is missing")
    files = code.get("files")
    if not isinstance(files, list):
        raise ProbeHealthEvidenceError("v41 code source file ledger is missing")
    source_paths = {
        record.get("path")
        for record in files
        if isinstance(record, Mapping)
    }
    if not {"engine.py", "main.py"}.issubset(source_paths):
        raise ProbeHealthEvidenceError(
            "v41 code source closure does not bind engine.py and main.py"
        )

    checkpoint_updates = _exact_int(
        payload.get("optimizer_updates"), "checkpoint.optimizer_updates"
    )
    epoch = _exact_int(payload.get("epoch"), "checkpoint.epoch")
    iteration = _exact_int(payload.get("iteration"), "checkpoint.iteration")
    epoch_finished = payload.get("epoch_finished")
    if type(epoch_finished) is not bool:
        raise ProbeHealthEvidenceError("checkpoint.epoch_finished must be a bool")
    if checkpoint_updates != EXPECTED_UPDATES:
        raise ProbeHealthEvidenceError(
            f"candidate v41 checkpoint optimizer_updates must be {EXPECTED_UPDATES}"
        )
    if payload.get("checkpoint_reason") != "max_train_iters":
        raise ProbeHealthEvidenceError(
            "candidate v41 checkpoint reason must be max_train_iters"
        )

    pack_factor = EXPECTED_PACKED_VALUES[
        "stage_b_dense_duty_forward_pack_factor"
    ]
    logical_per_epoch = EXPECTED_PACKED_VALUES[
        "stage_b_dense_duty_expected_logical_batches_per_epoch"
    ]
    physical_per_epoch = EXPECTED_PACKED_VALUES[
        "stage_b_dense_duty_expected_physical_forwards_per_epoch"
    ]
    accumulation = EXPECTED_PACKED_VALUES["gradient_accumulation_steps"]
    first_epoch_updates = math.ceil(physical_per_epoch / accumulation)
    expected_iteration = (
        checkpoint_updates - epoch * first_epoch_updates
    ) * accumulation
    replayed_physical = epoch * physical_per_epoch + iteration
    replayed_logical = epoch * logical_per_epoch + iteration * pack_factor
    effective_batch = (
        EXPECTED_PACKED_VALUES["batch_size"] * pack_factor * accumulation
    )
    if (
        trajectory_updates != first_epoch_updates
        or epoch != EXPECTED_TERMINAL_EPOCH
        or iteration != expected_iteration
        or iteration != EXPECTED_TERMINAL_ITERATION
        or epoch_finished is not False
        or replayed_physical != EXPECTED_TOTAL_PHYSICAL_FORWARDS
        or replayed_logical != EXPECTED_TOTAL_LOGICAL_BATCHES
        or effective_batch != 64
    ):
        raise ProbeHealthEvidenceError(
            "v41 packed-forward cursor does not replay the sealed U400 geometry"
        )

    return {
        "evidence_contract": "v23_source_closure_plus_checkpoint_cursor_v1",
        "training_contract_schema": TRAINING_CONTRACT_SCHEMA,
        "training_contract_sha256": contract_sha256,
        "code_source_sha256": code["sha256"],
        "score_ownership": EXPECTED_OWNERSHIP,
        "token_edit_query_scope": EXPECTED_SCOPE,
        "geometry": {
            "logical_batch_size": EXPECTED_PACKED_VALUES[
                "stage_b_dense_duty_logical_loss_batch_size"
            ],
            "forward_pack_factor": pack_factor,
            "forward_batch_size": EXPECTED_PACKED_VALUES[
                "stage_b_dense_duty_expected_forward_batch_size"
            ],
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": effective_batch,
            "logical_batches_per_epoch": logical_per_epoch,
            "physical_forwards_per_epoch": physical_per_epoch,
            "optimizer_updates_after_first_epoch": first_epoch_updates,
        },
        "terminal_cursor": {
            "epoch": epoch,
            "iteration": iteration,
            "optimizer_updates": checkpoint_updates,
            "epoch_finished": epoch_finished,
            "checkpoint_reason": payload["checkpoint_reason"],
        },
        "replay": {
            "physical_forwards": replayed_physical,
            "logical_batches": replayed_logical,
        },
    }


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)


def _health_checks(
    trajectory: Mapping[str, float | int],
    endpoint: Mapping[str, Any],
    runtime: Mapping[str, Any],
    packed: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    token = float(trajectory["train_loss_fixed_text_token_unscaled"])
    trust = float(
        trajectory[
            "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled"
        ]
    )
    threshold_valid = float(
        trajectory["train_fixed_text_tail_queue_threshold_valid_unscaled"]
    )
    pair_selected = float(
        trajectory[
            "train_fixed_text_token_role_carrier_pair_selected_count_unscaled"
        ]
    )
    positive_added = float(
        trajectory[
            "train_fixed_text_token_role_carrier_positive_added_count_unscaled"
        ]
    )
    tn_added = float(
        trajectory["train_fixed_text_token_role_carrier_tn_added_count_unscaled"]
    )
    positive_overlap = float(
        trajectory[
            "train_fixed_text_token_role_carrier_positive_target_overlap_count_unscaled"
        ]
    )
    tn_overlap = float(
        trajectory[
            "train_fixed_text_token_role_carrier_tn_target_overlap_count_unscaled"
        ]
    )
    edit_selected = float(
        trajectory["train_fixed_text_token_edit_carrier_selected_count_unscaled"]
    )
    edit_added = float(
        trajectory["train_fixed_text_token_edit_carrier_added_count_unscaled"]
    )
    edit_overlap = float(
        trajectory[
            "train_fixed_text_token_edit_carrier_target_overlap_count_unscaled"
        ]
    )
    eligible = float(
        trajectory["train_fixed_text_confidence_tn_train_eligible_count_unscaled"]
    )
    excluded = float(
        trajectory["train_fixed_text_confidence_tn_train_excluded_count_unscaled"]
    )
    all_negative = float(
        trajectory["train_fixed_text_token_all_negative_count_unscaled"]
    )
    provenance = float(
        trajectory["train_fixed_text_token_provenance_valid_count_unscaled"]
    )
    direct = float(
        trajectory["train_fixed_text_token_direct_trace_valid_count_unscaled"]
    )
    global_tn = float(
        trajectory["train_fixed_text_global_tn_sample_count_unscaled"]
    )
    positive_q05 = float(endpoint["positive_q05"])
    tn_q95 = float(endpoint["tn_q95"])
    gap = float(endpoint["operating_gap"])

    return {
        "u222_token_below_old_u444": _check(
            token, f"< {OLD_U444_TOKEN_LOSS}", token < OLD_U444_TOKEN_LOSS
        ),
        "u222_q05_gradient_path_healthy": _check(
            trust, "<= 0.4", trust <= 0.4
        ),
        "u222_tail_queue_ready": _check(
            threshold_valid, ">= 0.9", threshold_valid >= 0.9
        ),
        "u222_duplicate_global_tn_tail_disabled": _check(
            trajectory["train_loss_fixed_text_global_tn_tail"],
            "== 0",
            float(trajectory["train_loss_fixed_text_global_tn_tail"]) == 0.0,
        ),
        "u222_log_amp_skips_zero": _check(
            trajectory["train_amp_step_skipped"],
            "== 0",
            float(trajectory["train_amp_step_skipped"]) == 0.0,
        ),
        "u222_role_carrier_support_nonzero": _check(
            pair_selected, "> 0", pair_selected > 0.0
        ),
        "u222_role_carrier_full_eligible_coverage": _check(
            {
                "pair_selected": pair_selected,
                "tn_eligible": eligible,
                "direct_trace_valid": direct,
                "global_tn_samples": global_tn,
            },
            "pair_selected == TN eligible == direct trace valid == global TN samples",
            _close(pair_selected, eligible)
            and _close(pair_selected, direct)
            and _close(pair_selected, global_tn),
        ),
        "u222_role_carrier_trace_partition_exact": _check(
            {
                "eligible": eligible,
                "excluded": excluded,
                "provenance_valid": provenance,
                "all_negative": all_negative,
            },
            "eligible + excluded == provenance valid == 16 and all-negative == 0",
            _close(eligible + excluded, provenance)
            and _close(provenance, 16.0)
            and _close(all_negative, 0.0),
        ),
        "u222_role_carrier_positive_partition_exact": _check(
            {
                "selected": pair_selected,
                "added": positive_added,
                "target_overlap": positive_overlap,
            },
            "positive added + target overlap == pair selected",
            _close(positive_added + positive_overlap, pair_selected),
        ),
        "u222_role_carrier_tn_partition_exact": _check(
            {
                "selected": pair_selected,
                "added": tn_added,
                "target_overlap": tn_overlap,
            },
            "TN added + target overlap == pair selected",
            _close(tn_added + tn_overlap, pair_selected),
        ),
        "u222_role_carrier_edit_alias_exact": _check(
            {
                "role_selected": pair_selected,
                "edit_selected": edit_selected,
                "role_tn_added": tn_added,
                "edit_added": edit_added,
                "role_tn_overlap": tn_overlap,
                "edit_overlap": edit_overlap,
            },
            "edit counters exactly alias the role-complete TN carrier",
            _close(edit_selected, pair_selected)
            and _close(edit_added, tn_added)
            and _close(edit_overlap, tn_overlap),
        ),
        "u222_role_carrier_expands_both_roles": _check(
            {"positive_added": positive_added, "tn_added": tn_added},
            "positive added > 0 and TN added > 0",
            positive_added > 0.0 and tn_added > 0.0,
        ),
        "u400_positive_q05_in_operating_range": _check(
            positive_q05, "> -0.1", positive_q05 > -0.1
        ),
        "u400_tn_q95_in_operating_range": _check(
            tn_q95, "< 0.1", tn_q95 < 0.1
        ),
        "u400_operating_gap_in_operating_range": _check(
            gap, "> -0.2", gap > -0.2
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
        "packed_v23_role_complete_contract_bound": _check(
            {
                "schema": packed["training_contract_schema"],
                "scope": packed["token_edit_query_scope"],
            },
            "v23 role-complete scope",
            packed["training_contract_schema"] == TRAINING_CONTRACT_SCHEMA
            and packed["token_edit_query_scope"] == EXPECTED_SCOPE,
        ),
        "packed_u400_cursor_replay_exact": _check(
            {
                "cursor": packed["terminal_cursor"],
                "replay": packed["replay"],
            },
            "epoch 1 / iteration 356 replays 800 physical and 1599 logical batches",
            packed["terminal_cursor"]
            == {
                "epoch": EXPECTED_TERMINAL_EPOCH,
                "iteration": EXPECTED_TERMINAL_ITERATION,
                "optimizer_updates": EXPECTED_UPDATES,
                "epoch_finished": False,
                "checkpoint_reason": "max_train_iters",
            }
            and packed["replay"]
            == {
                "physical_forwards": EXPECTED_TOTAL_PHYSICAL_FORWARDS,
                "logical_batches": EXPECTED_TOTAL_LOGICAL_BATCHES,
            },
        ),
    }


def _baseline() -> dict[str, Any]:
    return {
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
            "u222_positive_q05": _BASE.OLD_U222_POSITIVE_Q05,
            "u222_tn_q95": _BASE.OLD_U222_TN_Q95,
            "u222_operating_gap": _BASE.OLD_U222_OPERATING_GAP,
            "u222_positive_trust_violation_rate": _BASE.OLD_U222_TRUST_VIOLATION,
        },
    }


def audit() -> dict[str, Any]:
    state = probe.inspect()
    if not isinstance(state, Mapping):
        raise ProbeHealthEvidenceError(
            "v41 probe controller inspect result must be a mapping"
        )
    for field, expected in (
        ("status", "terminal"),
        ("action", "complete"),
        ("updates", EXPECTED_UPDATES),
    ):
        if state.get(field) != expected:
            raise ProbeHealthEvidenceError(
                f"v41 probe controller requires {field}={expected!r}, "
                f"got {state.get(field)!r}"
            )

    trajectory, log_record = _strict_u222_log(_candidate_log())
    payload, checkpoint_record = _load_checkpoint_mmap(_candidate_checkpoint())
    packed = _audit_packed_forward(
        payload,
        payload.get("args"),
        trajectory_updates=int(trajectory["train_optimizer_updates"]),
    )
    endpoint = _audit_endpoint_u400(payload.get("criterion"))
    runtime = _audit_runtime(payload.get("args"))
    checks = _health_checks(trajectory, endpoint, runtime, packed)
    failed = [name for name, value in checks.items() if not value["passed"]]

    return {
        "schema": SCHEMA,
        "baseline": _baseline(),
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
        "endpoint_u400": endpoint,
        "packed_forward": packed,
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
