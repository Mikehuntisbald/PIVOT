#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Audit terminal U400 health for the v42 TN-only carrier-pair probe.

The immutable checkpoint is the evidence source.  Post-training source changes
may make the training controller intentionally reject resume, but they do not
invalidate the checkpoint-recorded v24 contract or its sealed source closure.
This audit is a training-health gate, not a RefCOCO or TN performance claim.
"""

from __future__ import annotations

import argparse
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
    run_stageb_confidence_adapter_candidate_tn_only_carrier_pair_probe_u0400 as probe,
)
from util.stage_b_dense_duty_audit import (  # noqa: E402
    CONFIG_SOURCE_CLOSURE_SCHEMA,
    fingerprint_named_tensors,
    validate_source_closure,
)


_V41_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_candidate_"
    "role_complete_carrier_probe_health.py"
)
_V41_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_tn_only_pair_health_helpers",
    _V41_PATH,
)
if _V41_SPEC is None or _V41_SPEC.loader is None:
    raise RuntimeError(f"cannot load health-audit helpers: {_V41_PATH}")
_V41 = importlib.util.module_from_spec(_V41_SPEC)
_V41_SPEC.loader.exec_module(_V41)


SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_tn_only_carrier_pair_"
    "probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v24"
SOURCE_CLOSURE_SCHEMA = "pivot.stageb.dense_duty_source_closure/v1"
CODE_SOURCE_CLOSURE_SCHEMA = "pivot.stageb.dense_duty_code_source_closure/v1"
RUNTIME_SCHEMA = _V41.RUNTIME_SCHEMA
EXPECTED_UPDATES = 400
EXPECTED_LOG_UPDATES = 222
QUEUE_SIZE = 4096
EXPECTED_TOKEN_SCOPE = "target_iou_v1"
EXPECTED_OWNERSHIP = "rank_tower_stopgrad_token_adapter_two_phase"
EXPECTED_GRADIENT_CONTRACT = "tn_only_positive_detached_v2"
EXPECTED_RAW_VETO_SCOPE = (
    "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
)
EXPECTED_CARRIER_SELECTOR = "final_layer_reference_argmax_exact_eligible_v1"
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "tn_only_carrier_pair_probe_u0400_20260801.py"
)
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
EXPECTED_V42_VALUES = {
    "stage_b_v22_score_ownership": EXPECTED_OWNERSHIP,
    "stage_b_dense_duty_raw_veto_query_scope": EXPECTED_RAW_VETO_SCOPE,
    "stage_b_dense_duty_confidence_carrier_selector_contract": (
        EXPECTED_CARRIER_SELECTOR
    ),
    "stage_b_dense_duty_raw_veto_tn_carrier_balance": 0.25,
    "stage_b_dense_duty_raw_veto_carrier_pair_weight": 0.25,
    "stage_b_dense_duty_raw_veto_carrier_pair_margin": 0.25,
    "stage_b_dense_duty_raw_veto_tail_quantile": 0.95,
    "stage_b_dense_duty_raw_veto_tail_temperature": 0.1,
    "stage_b_dense_duty_raw_veto_tail_min_count": 256,
    "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract": (
        EXPECTED_GRADIENT_CONTRACT
    ),
}
EXPECTED_TERMINAL_EPOCH = 1
EXPECTED_TERMINAL_ITERATION = 356
EXPECTED_TOTAL_PHYSICAL_FORWARDS = 800
EXPECTED_TOTAL_LOGICAL_BATCHES = 1599

PAIR_COUNT_FIELDS = (
    "train_fixed_text_raw_veto_positive_sample_count_unscaled",
    "train_fixed_text_raw_veto_positive_carrier_sample_count_unscaled",
    "train_fixed_text_raw_veto_tn_sample_count_unscaled",
    "train_fixed_text_raw_veto_tn_carrier_sample_count_unscaled",
    "train_fixed_text_raw_veto_carrier_pair_sample_count_unscaled",
    "train_fixed_text_confidence_tn_train_eligible_count_unscaled",
    "train_fixed_text_confidence_tn_train_excluded_count_unscaled",
    "train_fixed_text_token_all_negative_count_unscaled",
    "train_fixed_text_token_provenance_valid_count_unscaled",
    "train_fixed_text_token_direct_trace_valid_count_unscaled",
    "train_fixed_text_global_tn_sample_count_unscaled",
)
PAIR_RATE_FIELDS = (
    "train_fixed_text_raw_veto_carrier_pair_violation_rate_unscaled",
    "train_fixed_text_raw_veto_tail_pair_violation_rate_unscaled",
)
PAIR_VALUE_FIELDS = (
    "train_loss_fixed_text_raw_veto_carrier_pair",
    "train_loss_fixed_text_raw_veto_carrier_pair_unscaled",
    "train_fixed_text_raw_veto_carrier_pair_gap_mean_unscaled",
    "train_fixed_text_raw_veto_carrier_pair_hinge_mean_unscaled",
    "train_fixed_text_raw_veto_tail_pair_gap_mean_unscaled",
    "train_fixed_text_raw_veto_tail_pair_hinge_mean_unscaled",
    "train_fixed_text_raw_veto_tail_pair_effective_sample_count_unscaled",
    "train_fixed_text_raw_veto_tn_tail_effective_sample_count_unscaled",
)
REQUIRED_U222_FIELDS = (
    _V41._BASE.REQUIRED_U222_FIELDS
    + PAIR_COUNT_FIELDS
    + PAIR_RATE_FIELDS
    + PAIR_VALUE_FIELDS
)

BASELINE_LOG = _V41.BASELINE_LOG
BASELINE_U222_CHECKPOINT = _V41.BASELINE_U222_CHECKPOINT
BASELINE_U444_CHECKPOINT = _V41.BASELINE_U444_CHECKPOINT
BASELINE_LOG_SHA256 = _V41.BASELINE_LOG_SHA256
BASELINE_U222_CHECKPOINT_SHA256 = _V41.BASELINE_U222_CHECKPOINT_SHA256
BASELINE_U444_CHECKPOINT_SHA256 = _V41.BASELINE_U444_CHECKPOINT_SHA256
OLD_U444_TOKEN_LOSS = _V41.OLD_U444_TOKEN_LOSS

ProbeHealthEvidenceError = _V41.ProbeHealthEvidenceError
_file_record = _V41._file_record
_exact_int = _V41._exact_int
_finite_float = _V41._finite_float
_queue_scalar = _V41._queue_scalar
_queue_tensor = _V41._queue_tensor
_exact_lower_tail_operating_threshold = (
    _V41._exact_lower_tail_operating_threshold
)
_audit_runtime = _V41._audit_runtime
_check = _V41._check
_atomic_write_json = _V41._atomic_write_json
_canonical_sha256 = _V41._canonical_sha256
_valid_sha256 = _V41._valid_sha256


def _candidate_log() -> Path:
    return Path(probe.OUTPUT) / "log.txt"


def _candidate_checkpoint() -> Path:
    return Path(probe.CHECKPOINT)


def _default_output() -> Path:
    return Path(probe.OUTPUT).parent / (
        "u000400_tn_only_carrier_pair_probe_health_audit.json"
    )


def _strict_u222_log(path: Path) -> tuple[dict[str, float | int], dict[str, Any]]:
    record = _file_record(path, label="candidate v42 U222 log")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProbeHealthEvidenceError(
            f"could not read candidate v42 U222 log: {error}"
        ) from error
    if len(lines) != 1 or not lines[0].strip():
        raise ProbeHealthEvidenceError(
            "candidate v42 log must contain exactly one non-empty U222 JSON line"
        )
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ProbeHealthEvidenceError(
            f"candidate v42 U222 log is invalid JSON: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ProbeHealthEvidenceError(
            "candidate v42 U222 log line must be a JSON object"
        )
    updates = _exact_int(
        payload.get("train_optimizer_updates"), "train_optimizer_updates"
    )
    if updates != EXPECTED_LOG_UPDATES:
        raise ProbeHealthEvidenceError(
            f"candidate v42 log update must be U{EXPECTED_LOG_UPDATES}, got U{updates}"
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
        *PAIR_RATE_FIELDS,
    ):
        if not 0.0 <= float(values[field]) <= 1.0:
            raise ProbeHealthEvidenceError(f"{field} must be in [0, 1]")
    for field in PAIR_COUNT_FIELDS:
        if not 0.0 <= float(values[field]) <= 16.0:
            raise ProbeHealthEvidenceError(f"{field} must be in [0, 16]")
    for field in (
        "train_fixed_text_raw_veto_tail_pair_effective_sample_count_unscaled",
        "train_fixed_text_raw_veto_tn_tail_effective_sample_count_unscaled",
    ):
        if not 0.0 <= float(values[field]) <= 16.0:
            raise ProbeHealthEvidenceError(f"{field} must be in [0, 16]")
    for field in (
        "train_loss_fixed_text_raw_veto_carrier_pair",
        "train_loss_fixed_text_raw_veto_carrier_pair_unscaled",
        "train_fixed_text_raw_veto_carrier_pair_hinge_mean_unscaled",
        "train_fixed_text_raw_veto_tail_pair_hinge_mean_unscaled",
    ):
        if float(values[field]) < 0.0:
            raise ProbeHealthEvidenceError(f"{field} must be non-negative")
    if float(values["train_amp_step_skipped"]) < 0.0:
        raise ProbeHealthEvidenceError("train_amp_step_skipped must be non-negative")
    return values, record


def _load_checkpoint_mmap(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    record = _file_record(path, label="candidate v42 U400 checkpoint")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:
        raise ProbeHealthEvidenceError(
            "could not mmap-load candidate v42 U400 checkpoint: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ProbeHealthEvidenceError(
            "candidate v42 U400 checkpoint must be a mapping"
        )
    return payload, record


def _audit_endpoint_u400(criterion: Any) -> dict[str, Any]:
    if not isinstance(criterion, Mapping) or set(criterion) != _V41._BASE.QUEUE_KEYS:
        raise ProbeHealthEvidenceError(
            "candidate v42 criterion must contain the exact six tail-queue tensors"
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
            f"both v42 U400 tail queues must have count {QUEUE_SIZE}, got {counts}"
        )
    for label, pointer in pointers.items():
        if not 0 <= pointer < QUEUE_SIZE:
            raise ProbeHealthEvidenceError(
                f"v42 U400 {label} tail queue pointer must be in [0, {QUEUE_SIZE})"
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
            "derived v42 U400 tail metrics must be finite"
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
        raise ProbeHealthEvidenceError("candidate v42 checkpoint lacks training contract")
    if set(contract) != {"schema", "sha256", "values"}:
        raise ProbeHealthEvidenceError(
            "candidate v42 training contract has an invalid structure"
        )
    if contract.get("schema") != TRAINING_CONTRACT_SCHEMA:
        raise ProbeHealthEvidenceError(
            f"v42 training contract schema must be {TRAINING_CONTRACT_SCHEMA!r}"
        )
    values = contract.get("values")
    if not isinstance(values, Mapping):
        raise ProbeHealthEvidenceError("v42 training contract values are missing")
    contract_sha256 = contract.get("sha256")
    if not _valid_sha256(contract_sha256) or contract_sha256 != _canonical_sha256(
        values, label="v42 training contract values"
    ):
        raise ProbeHealthEvidenceError("v42 training contract SHA-256 is invalid")

    token_scope = args.get("stage_b_v21_token_edit_query_scope", EXPECTED_TOKEN_SCOPE)
    if token_scope != EXPECTED_TOKEN_SCOPE:
        raise ProbeHealthEvidenceError(
            "checkpoint args require the target_iou_v1 token query scope, "
            f"got {token_scope!r}"
        )
    if (
        "stage_b_v21_token_edit_query_scope" in values
        and values.get("stage_b_v21_token_edit_query_scope") != EXPECTED_TOKEN_SCOPE
    ):
        raise ProbeHealthEvidenceError(
            "v42 training contract records a non-default token query scope"
        )
    for name, expected in EXPECTED_PACKED_VALUES.items():
        _exact_contract_value(args, values, name, expected)
    for name, expected in EXPECTED_V42_VALUES.items():
        _exact_contract_value(args, values, name, expected)

    closure_value = values.get("stage_b_dense_duty_source_closure")
    if not isinstance(closure_value, Mapping) or args.get(
        "stage_b_dense_duty_source_closure"
    ) != closure_value:
        raise ProbeHealthEvidenceError(
            "v42 training contract source closure is missing or unbound"
        )
    try:
        closure = validate_source_closure(closure_value)
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(
            f"v42 source closure is invalid: {error}"
        ) from error
    config = closure.get("config")
    if not isinstance(config, Mapping) or config.get("entry") != EXPECTED_CONFIG_ENTRY:
        raise ProbeHealthEvidenceError(
            "v42 source closure does not bind the exact U400 probe config"
        )
    code = closure.get("code")
    if not isinstance(code, Mapping):
        raise ProbeHealthEvidenceError("v42 code source closure is missing")
    files = code.get("files")
    if not isinstance(files, list):
        raise ProbeHealthEvidenceError("v42 code source file ledger is missing")
    source_paths = {
        record.get("path") for record in files if isinstance(record, Mapping)
    }
    required_sources = {
        "engine.py",
        "main.py",
        "models/GroundingDINO/groundingdino.py",
        "models/GroundingDINO/stage_b_fixed_text_criterion.py",
        "util/stage_b_dense_duty_audit.py",
    }
    if not required_sources.issubset(source_paths):
        raise ProbeHealthEvidenceError(
            "v42 code source closure omits a gradient-route implementation file"
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
            f"candidate v42 checkpoint optimizer_updates must be {EXPECTED_UPDATES}"
        )
    if payload.get("checkpoint_reason") != "max_train_iters":
        raise ProbeHealthEvidenceError(
            "candidate v42 checkpoint reason must be max_train_iters"
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
            "v42 packed-forward cursor does not replay the sealed U400 geometry"
        )

    return {
        "evidence_contract": "v24_source_closure_plus_checkpoint_cursor_v1",
        "training_contract_schema": TRAINING_CONTRACT_SCHEMA,
        "training_contract_sha256": contract_sha256,
        "code_source_sha256": code["sha256"],
        "config_source_sha256": config["sha256"],
        "score_ownership": EXPECTED_OWNERSHIP,
        "token_edit_query_scope": token_scope,
        "token_edit_query_scope_source": (
            "checkpoint_args_explicit"
            if "stage_b_v21_token_edit_query_scope" in args
            else "checkpoint_args_default"
        ),
        "gradient_contract": EXPECTED_GRADIENT_CONTRACT,
        "raw_veto_contract": {
            name: expected for name, expected in EXPECTED_V42_VALUES.items()
        },
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


def _audit_rank(payload: Mapping[str, Any], args: Mapping[str, Any]) -> dict[str, Any]:
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ProbeHealthEvidenceError("candidate v42 checkpoint lacks model state")
    names = sorted(
        str(name)
        for name in model
        if str(name).startswith("stage_b_fixed_text_scorer.rank_tower.")
    )
    if not names:
        raise ProbeHealthEvidenceError("candidate v42 checkpoint has no rank tower")
    try:
        rank = fingerprint_named_tensors(model, names)
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(
            f"candidate v42 rank fingerprint is invalid: {error}"
        ) from error
    expected = args.get("stage_b_dense_duty_rank_source_rank_sha256")
    if not _valid_sha256(expected) or rank.get("sha256") != expected:
        raise ProbeHealthEvidenceError(
            "candidate v42 checkpoint changed the frozen rank tower"
        )
    if rank.get("nonfinite_count") != 0:
        raise ProbeHealthEvidenceError("candidate v42 rank tower is non-finite")
    return rank


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)


def _health_checks(
    trajectory: Mapping[str, float | int],
    endpoint: Mapping[str, Any],
    runtime: Mapping[str, Any],
    packed: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    def value(name: str) -> float:
        return float(trajectory[name])

    pair_samples = value(
        "train_fixed_text_raw_veto_carrier_pair_sample_count_unscaled"
    )
    tn_samples = value("train_fixed_text_raw_veto_tn_sample_count_unscaled")
    tn_carriers = value(
        "train_fixed_text_raw_veto_tn_carrier_sample_count_unscaled"
    )
    eligible = value(
        "train_fixed_text_confidence_tn_train_eligible_count_unscaled"
    )
    excluded = value(
        "train_fixed_text_confidence_tn_train_excluded_count_unscaled"
    )
    provenance = value(
        "train_fixed_text_token_provenance_valid_count_unscaled"
    )
    direct = value("train_fixed_text_token_direct_trace_valid_count_unscaled")
    global_tn = value("train_fixed_text_global_tn_sample_count_unscaled")
    all_negative = value("train_fixed_text_token_all_negative_count_unscaled")
    positive_samples = value(
        "train_fixed_text_raw_veto_positive_sample_count_unscaled"
    )
    positive_carriers = value(
        "train_fixed_text_raw_veto_positive_carrier_sample_count_unscaled"
    )
    weighted_pair_loss = value("train_loss_fixed_text_raw_veto_carrier_pair")
    pair_loss = value(
        "train_loss_fixed_text_raw_veto_carrier_pair_unscaled"
    )
    tail_pair_hinge = value(
        "train_fixed_text_raw_veto_tail_pair_hinge_mean_unscaled"
    )
    tail_pair_effective = value(
        "train_fixed_text_raw_veto_tail_pair_effective_sample_count_unscaled"
    )
    tn_tail_effective = value(
        "train_fixed_text_raw_veto_tn_tail_effective_sample_count_unscaled"
    )
    positive_q05 = float(endpoint["positive_q05"])
    tn_q95 = float(endpoint["tn_q95"])
    gap = float(endpoint["operating_gap"])

    return {
        "u222_token_below_old_u444": _check(
            value("train_loss_fixed_text_token_unscaled"),
            f"< {OLD_U444_TOKEN_LOSS}",
            value("train_loss_fixed_text_token_unscaled") < OLD_U444_TOKEN_LOSS,
        ),
        "u222_q05_gradient_path_healthy": _check(
            value(
                "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled"
            ),
            "<= 0.4",
            value(
                "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled"
            )
            <= 0.4,
        ),
        "u222_tail_queue_ready": _check(
            value("train_fixed_text_tail_queue_threshold_valid_unscaled"),
            ">= 0.9",
            value("train_fixed_text_tail_queue_threshold_valid_unscaled") >= 0.9,
        ),
        "u222_duplicate_global_tn_tail_disabled": _check(
            value("train_loss_fixed_text_global_tn_tail"),
            "== 0",
            value("train_loss_fixed_text_global_tn_tail") == 0.0,
        ),
        "u222_log_amp_skips_zero": _check(
            value("train_amp_step_skipped"),
            "== 0",
            value("train_amp_step_skipped") == 0.0,
        ),
        "u222_pair_support_nonzero": _check(
            pair_samples, "> 0", pair_samples > 0.0
        ),
        "u222_pair_support_conservation_exact": _check(
            {
                "pairs": pair_samples,
                "tn_samples": tn_samples,
                "tn_carriers": tn_carriers,
                "eligible": eligible,
                "direct_trace": direct,
                "global_tn": global_tn,
            },
            "pairs == TN samples == TN carriers == eligible == direct trace == global TN",
            all(
                _close(pair_samples, observed)
                for observed in (
                    tn_samples,
                    tn_carriers,
                    eligible,
                    direct,
                    global_tn,
                )
            ),
        ),
        "u222_pair_trace_partition_exact": _check(
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
        "u222_positive_carrier_support_complete": _check(
            {
                "positive_samples": positive_samples,
                "positive_carriers": positive_carriers,
            },
            "positive samples == positive carriers == 16",
            _close(positive_samples, positive_carriers)
            and _close(positive_samples, 16.0),
        ),
        "u222_tail_pair_support_conservation_exact": _check(
            {
                "tail_pair_effective": tail_pair_effective,
                "tn_tail_effective": tn_tail_effective,
                "pairs": pair_samples,
            },
            "0 < tail-pair effective == TN-tail effective <= pairs",
            tail_pair_effective > 0.0
            and _close(tail_pair_effective, tn_tail_effective)
            and tail_pair_effective <= pair_samples + 1e-6,
        ),
        "u222_tail_pair_loss_attributed_exact": _check(
            {"pair_loss": pair_loss, "tail_pair_hinge": tail_pair_hinge},
            "unscaled pair loss == tail-weighted pair hinge",
            _close(pair_loss, tail_pair_hinge),
        ),
        "u222_pair_weight_applied_exact": _check(
            {"weighted": weighted_pair_loss, "unscaled": pair_loss},
            "weighted pair loss == 0.25 * unscaled pair loss",
            _close(weighted_pair_loss, 0.25 * pair_loss),
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
        "packed_v24_tn_only_gradient_contract_bound": _check(
            {
                "schema": packed["training_contract_schema"],
                "token_scope": packed["token_edit_query_scope"],
                "gradient_contract": packed["gradient_contract"],
            },
            "v24 + target_iou_v1 + tn_only_positive_detached_v2",
            packed["training_contract_schema"] == TRAINING_CONTRACT_SCHEMA
            and packed["token_edit_query_scope"] == EXPECTED_TOKEN_SCOPE
            and packed["gradient_contract"] == EXPECTED_GRADIENT_CONTRACT,
        ),
        "packed_u400_cursor_replay_exact": _check(
            {"cursor": packed["terminal_cursor"], "replay": packed["replay"]},
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
            "u222_positive_q05": _V41._BASE.OLD_U222_POSITIVE_Q05,
            "u222_tn_q95": _V41._BASE.OLD_U222_TN_Q95,
            "u222_operating_gap": _V41._BASE.OLD_U222_OPERATING_GAP,
            "u222_positive_trust_violation_rate": (
                _V41._BASE.OLD_U222_TRUST_VIOLATION
            ),
        },
    }


def audit() -> dict[str, Any]:
    trajectory, log_record = _strict_u222_log(_candidate_log())
    payload, checkpoint_record = _load_checkpoint_mmap(_candidate_checkpoint())
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise ProbeHealthEvidenceError("candidate checkpoint args must be a mapping")
    packed = _audit_packed_forward(
        payload,
        args,
        trajectory_updates=int(trajectory["train_optimizer_updates"]),
    )
    endpoint = _audit_endpoint_u400(payload.get("criterion"))
    runtime = _audit_runtime(args)
    rank = _audit_rank(payload, args)
    checks = _health_checks(trajectory, endpoint, runtime, packed)
    failed = [name for name, value in checks.items() if not value["passed"]]

    return {
        "schema": SCHEMA,
        "baseline": _baseline(),
        "candidate": {
            "controller": {
                "status": "terminal",
                "action": "complete",
                "updates": EXPECTED_UPDATES,
                "rank_sha256": rank["sha256"],
                "evidence_source": "immutable_checkpoint_v24",
                "current_resume_controller_required": False,
            },
            "log": log_record,
            "checkpoint": checkpoint_record,
        },
        "trajectory_u222": trajectory,
        "endpoint_u400": endpoint,
        "packed_forward": packed,
        "runtime": runtime,
        "rank": rank,
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
            "error": {"type": type(error).__name__, "message": str(error)},
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
