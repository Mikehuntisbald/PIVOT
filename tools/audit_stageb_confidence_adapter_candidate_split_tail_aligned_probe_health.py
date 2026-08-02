#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed mechanical health audit for the V45 terminal U400 probe.

This audit does not judge RefCOCO or TN quality. It only admits the immutable
terminal checkpoint to the fixed strict1607 diagnostic when the V45/v27
training, ownership, clipping, queue, AMP, and frozen-rank evidence all hold.

Exit codes:
  0: all mechanical checks pass;
  1: evidence is valid but a runtime health check fails;
  2: evidence is missing, malformed, nonterminal, or contract-drifted.
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

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_split_tail_aligned_probe_u0400 as training,
)
from util.stage_b_dense_duty_audit import (  # noqa: E402
    TRAINING_CONTRACT_ARG,
    build_training_contract,
    fingerprint_named_tensors,
    validate_initial_fingerprint,
)


SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_split_tail_aligned_"
    "probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v27"
RUNTIME_SCHEMA = "pivot.stageb.dense_duty_runtime_audit/v1"
EXPECTED_REVISION = "word_veto_candidate_split_tail_aligned_v45"
EXPECTED_HEAD_CONTRACT = "split_token_veto_global_absolute_joint_clip_v3"
EXPECTED_RANK_SHA256 = (
    "e03219d5868004aa5cb9ff4fe68f1aa94d33f1f0f6e1290cb251d12f9c914045"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_tail_aligned_probe_u0400_20260801.py"
)
EXPECTED_UPDATES = 400
EXPECTED_LOG_UPDATES = 222
EXPECTED_TERMINAL_EPOCH = 1
EXPECTED_TERMINAL_ITERATION = 356
EXPECTED_ACTIVE_TENSORS = 74
EXPECTED_ACTIVE_ELEMENTS = 535_945
EXPECTED_TOKEN_TENSORS = 21
EXPECTED_TOKEN_ELEMENTS = 51_267
EXPECTED_GLOBAL_TENSORS = 53
EXPECTED_GLOBAL_ELEMENTS = 484_678
EXPECTED_LIVE_TOKEN_TENSORS = 21
EXPECTED_LIVE_GLOBAL_TENSORS = 46
QUEUE_SIZE = 4096
CLIP_MAX_NORM = 0.1

LOG_PATH = Path(training.OUTPUT) / "log.txt"

EXPECTED_CONTRACT_VALUES = {
    "stage_b_dense_duty": True,
    "stage_b_dense_duty_phase": "confidence",
    "stage_b_v22_train_phase": "confidence",
    "stage_b_v22_score_ownership": (
        "rank_tower_stopgrad_token_adapter_two_phase"
    ),
    "stage_b_dense_duty_execution_scope": "probe",
    "stage_b_dense_duty_confidence_expected_optimizer_updates": 400,
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_confidence_head_gradient_contract": (
        EXPECTED_HEAD_CONTRACT
    ),
    "stage_b_dense_duty_confidence_gate_gradient_contract": (
        "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
    ),
    "stage_b_dense_duty_deployed_veto_routing_weight": 1.0,
    "stage_b_dense_duty_deployed_veto_positive_max": 0.1,
    "stage_b_dense_duty_deployed_veto_tn_min": 0.9,
    "stage_b_dense_duty_deployed_veto_routing_reduction_contract": (
        "balanced_top_quarter_cvar_v2"
    ),
    "stage_b_v15_tail_queue_positive_trust_reduction_contract": (
        "top_quarter_cvar_v2"
    ),
    "stage_b_dense_duty_positive_trust_contract": (
        "absolute_global_confidence_logit_v2"
    ),
    "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract": (
        "bidirectional_v1"
    ),
    "stage_b_v15_tail_queue_positive_gradient_contract": (
        "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
    ),
    "stage_b_dense_duty_confidence_veto_gate_offset": 0.0,
    "stage_b_dense_duty_forward_pack_factor": 2,
    "stage_b_dense_duty_logical_loss_batch_size": 16,
    "stage_b_dense_duty_expected_forward_batch_size": 32,
    "stage_b_dense_duty_expected_logical_batches_per_epoch": 887,
    "stage_b_dense_duty_expected_physical_forwards_per_epoch": 444,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "amp": True,
    "clip_max_norm": CLIP_MAX_NORM,
    "max_train_iters": EXPECTED_UPDATES,
    "stage_b_v14_tail_queue_size": QUEUE_SIZE,
}

_CONFIDENCE_ADAPTER_PREFIX = "stage_b_fixed_text_scorer.confidence_adapter."
_CONFIDENCE_POOL_PREFIX = "stage_b_fixed_text_scorer.confidence_pool."
_RANK_PREFIX = "stage_b_fixed_text_scorer.rank_tower."
_TOKEN_MODULE_PREFIXES = (
    "query_norm.",
    "text_norm.",
    "query_projection.",
    "text_projection.",
    "query_bias.",
    "token_bias.",
    "carrier_rank_slope.",
    "rank_channel_norm.",
    "rank_channel_projection.",
    "rank_channel_logit_projection.",
    "rank_channel_output.",
)
_TOKEN_OPTIONAL_PARAMETERS = frozenset(
    {"rank_evidence_residual_scale", "rank_evidence_residual_bias"}
)
_GLOBAL_MODULE_PREFIXES = (
    "patch_residual.",
    "patch_feature.",
    "feature_norm.",
    "global_query_norm.",
    "global_query_trunk.",
    "cross_query_norm.",
    "cross_text_norm.",
    "cross_query_projection.",
    "cross_text_projection.",
    "cross_evidence_projection.",
    "cross_attention.",
    "cross_ffn.",
    "cross_output_projection.",
    "candidate_absolute_head.",
)
_GLOBAL_OPTIONAL_PARAMETERS = frozenset(
    {
        "veto_cap_raw_ceiling",
        "candidate_patch_scale_raw",
        "candidate_veto_depth_raw",
        "candidate_coverage_depth_raw",
    }
)


class ProbeHealthEvidenceError(RuntimeError):
    """The terminal V45 health evidence cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ProbeHealthEvidenceError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ProbeHealthEvidenceError(f"{label} is unavailable: {path}") from error
    if not resolved.is_file():
        raise ProbeHealthEvidenceError(f"{label} is not a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _load_checkpoint(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    before = _file_record(path, label="terminal V45 checkpoint")
    payload = torch.load(
        Path(before["path"]), map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, Mapping):
        raise ProbeHealthEvidenceError("terminal V45 checkpoint is not a mapping")
    after = _file_record(path, label="terminal V45 checkpoint")
    if before != after:
        raise ProbeHealthEvidenceError("terminal V45 checkpoint changed while loading")
    return payload, before


def _exact_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise ProbeHealthEvidenceError(f"{label} must be an exact integer")
    return int(value)


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeHealthEvidenceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProbeHealthEvidenceError(f"{label} must be finite")
    return result


def _scalar_int_tensor(value: Any, *, label: str) -> int:
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.int64
        or value.numel() != 1
    ):
        raise ProbeHealthEvidenceError(f"{label} must be one int64 scalar")
    return int(value.item())


def _check(observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def _audit_controller_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeHealthEvidenceError("training controller state is not a mapping")
    expected = {
        "status": "terminal",
        "action": "complete",
        "updates": EXPECTED_UPDATES,
        "rank_sha256": EXPECTED_RANK_SHA256,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ProbeHealthEvidenceError(
                f"training controller requires {key}={item!r}, got {value.get(key)!r}"
            )
    return dict(value)


def _audit_terminal_cursor(payload: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "epoch": _exact_int(payload.get("epoch"), label="checkpoint.epoch"),
        "iteration": _exact_int(
            payload.get("iteration"), label="checkpoint.iteration"
        ),
        "optimizer_updates": _exact_int(
            payload.get("optimizer_updates"),
            label="checkpoint.optimizer_updates",
        ),
        "epoch_finished": payload.get("epoch_finished"),
        "checkpoint_reason": payload.get("checkpoint_reason"),
    }
    expected = {
        "epoch": EXPECTED_TERMINAL_EPOCH,
        "iteration": EXPECTED_TERMINAL_ITERATION,
        "optimizer_updates": EXPECTED_UPDATES,
        "epoch_finished": False,
        "checkpoint_reason": "max_train_iters",
    }
    if type(observed["epoch_finished"]) is not bool or observed != expected:
        raise ProbeHealthEvidenceError(
            f"terminal V45 U400 cursor drifted: {observed}"
        )
    return observed


def _audit_training_contract(args: Mapping[str, Any]) -> dict[str, Any]:
    saved = args.get(TRAINING_CONTRACT_ARG)
    if not isinstance(saved, Mapping):
        raise ProbeHealthEvidenceError("checkpoint lacks its saved v27 contract")
    try:
        rebuilt = build_training_contract(args)
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(
            f"V45 training contract/source closure is invalid: {error}"
        ) from error
    if dict(saved) != rebuilt:
        raise ProbeHealthEvidenceError("saved V45 training contract does not replay")
    if rebuilt.get("schema") != TRAINING_CONTRACT_SCHEMA:
        raise ProbeHealthEvidenceError("training contract is not exact v27")
    values = rebuilt.get("values")
    if not isinstance(values, Mapping):
        raise ProbeHealthEvidenceError("v27 training contract has no values mapping")
    for key, expected in EXPECTED_CONTRACT_VALUES.items():
        if values.get(key) != expected or args.get(key) != expected:
            raise ProbeHealthEvidenceError(
                f"V45 contract requires {key}={expected!r}"
            )

    if str(args.get("stage_b_v21_token_edit_query_scope", "target_iou_v1")) != (
        "target_iou_v1"
    ):
        raise ProbeHealthEvidenceError("V45 token edit query scope drifted")
    if args.get("stage_b_dense_duty_evaluation_scope") != "probe":
        raise ProbeHealthEvidenceError("V45 evaluation scope is not probe")
    if args.get("stage_b_dense_duty_confidence_probe_admission_contract") != (
        "disabled_for_probe_v1"
    ) or str(args.get("stage_b_dense_duty_confidence_probe_admission_report", "")):
        raise ProbeHealthEvidenceError(
            "V45 probe checkpoint must not carry formal admission"
        )

    closure = values.get("stage_b_dense_duty_source_closure")
    if not isinstance(closure, Mapping):
        raise ProbeHealthEvidenceError("v27 source closure is missing")
    config = closure.get("config")
    if not isinstance(config, Mapping) or config.get("entry") != EXPECTED_CONFIG_ENTRY:
        raise ProbeHealthEvidenceError("v27 closure does not bind the exact V45 config")
    return {
        "schema": rebuilt["schema"],
        "sha256": rebuilt["sha256"],
        "source_closure_sha256": closure.get("sha256"),
        "config_entry": config.get("entry"),
        "revision": values["stage_b_dense_duty_confidence_revision"],
        "head_gradient_contract": values[
            "stage_b_dense_duty_confidence_head_gradient_contract"
        ],
        "routing_reduction_contract": values[
            "stage_b_dense_duty_deployed_veto_routing_reduction_contract"
        ],
        "positive_trust_reduction_contract": values[
            "stage_b_v15_tail_queue_positive_trust_reduction_contract"
        ],
        "positive_trust_contract": values[
            "stage_b_dense_duty_positive_trust_contract"
        ],
        "token_edit_query_scope": "target_iou_v1",
    }


def _owner_for_name(name: str) -> str:
    if name.startswith(_CONFIDENCE_POOL_PREFIX):
        suffix = name[len(_CONFIDENCE_POOL_PREFIX) :]
        if suffix.startswith("residual."):
            return "global_absolute"
        raise ProbeHealthEvidenceError(f"unknown confidence-pool owner: {name}")
    if not name.startswith(_CONFIDENCE_ADAPTER_PREFIX):
        raise ProbeHealthEvidenceError(f"active parameter is outside confidence: {name}")
    suffix = name[len(_CONFIDENCE_ADAPTER_PREFIX) :]
    if suffix.startswith(_TOKEN_MODULE_PREFIXES) or suffix in (
        _TOKEN_OPTIONAL_PARAMETERS
    ):
        return "token_veto"
    if suffix.startswith(_GLOBAL_MODULE_PREFIXES) or suffix in (
        _GLOBAL_OPTIONAL_PARAMETERS
    ):
        return "global_absolute"
    raise ProbeHealthEvidenceError(f"unknown split-head parameter owner: {name}")


def _audit_split_ownership(
    payload: Mapping[str, Any], args: Mapping[str, Any]
) -> dict[str, Any]:
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    if not isinstance(model, Mapping) or not isinstance(optimizer, Mapping):
        raise ProbeHealthEvidenceError("checkpoint model/optimizer state is missing")
    try:
        initial = validate_initial_fingerprint(
            args.get("stage_b_dense_duty_initial_state_fingerprint"),
            expected_phase="confidence",
        )
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(str(error)) from error
    names = initial["active_parameter_names"]
    if (
        len(names) != EXPECTED_ACTIVE_TENSORS
        or initial["active"]["tensor_count"] != EXPECTED_ACTIVE_TENSORS
        or initial["active"]["element_count"] != EXPECTED_ACTIVE_ELEMENTS
        or initial["active"]["nonfinite_count"] != 0
    ):
        raise ProbeHealthEvidenceError("V45 active parameter fingerprint drifted")
    if any(name not in model for name in names):
        raise ProbeHealthEvidenceError("V45 active parameter is absent from model state")

    owners = {"token_veto": [], "global_absolute": []}
    for name in names:
        owners[_owner_for_name(name)].append(name)
    token = owners["token_veto"]
    global_absolute = owners["global_absolute"]
    token_elements = sum(int(model[name].numel()) for name in token)
    global_elements = sum(int(model[name].numel()) for name in global_absolute)
    if (
        len(token) != EXPECTED_TOKEN_TENSORS
        or token_elements != EXPECTED_TOKEN_ELEMENTS
        or len(global_absolute) != EXPECTED_GLOBAL_TENSORS
        or global_elements != EXPECTED_GLOBAL_ELEMENTS
        or set(token) & set(global_absolute)
        or set(token) | set(global_absolute) != set(names)
    ):
        raise ProbeHealthEvidenceError(
            "V45 token-veto/global-absolute ownership is not exact and disjoint"
        )
    for name in names:
        value = model[name]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
            raise ProbeHealthEvidenceError(f"active parameter is non-finite: {name}")
    current_active = fingerprint_named_tensors(model, names)
    if (
        current_active["nonfinite_count"] != 0
        or current_active["sha256"] == initial["active"]["sha256"]
    ):
        raise ProbeHealthEvidenceError("V45 active state is non-finite or unchanged")

    param_groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(param_groups, list) or not isinstance(state, Mapping):
        raise ProbeHealthEvidenceError("V45 optimizer state is malformed")
    optimizer_ids: list[int] = []
    for group in param_groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ProbeHealthEvidenceError("V45 optimizer parameter group is malformed")
        for value in group["params"]:
            optimizer_ids.append(_exact_int(value, label="optimizer parameter id"))
    if (
        len(optimizer_ids) != EXPECTED_ACTIVE_TENSORS
        or len(set(optimizer_ids)) != EXPECTED_ACTIVE_TENSORS
        or set(optimizer_ids) != set(range(EXPECTED_ACTIVE_TENSORS))
    ):
        raise ProbeHealthEvidenceError("optimizer does not own the exact active set")

    active_set = set(names)
    ordered_names = [str(name) for name in model if str(name) in active_set]
    if len(ordered_names) != EXPECTED_ACTIVE_TENSORS:
        raise ProbeHealthEvidenceError("cannot replay optimizer parameter order")
    state_ids = {
        _exact_int(value, label="optimizer state id") for value in state.keys()
    }
    if not state_ids.issubset(set(optimizer_ids)):
        raise ProbeHealthEvidenceError("optimizer state references an unknown parameter")
    live = [ordered_names[index] for index in sorted(state_ids)]
    live_token = sum(_owner_for_name(name) == "token_veto" for name in live)
    live_global = sum(
        _owner_for_name(name) == "global_absolute" for name in live
    )
    if (
        live_token != EXPECTED_LIVE_TOKEN_TENSORS
        or live_global != EXPECTED_LIVE_GLOBAL_TENSORS
    ):
        raise ProbeHealthEvidenceError("both split owners were not optimized as sealed")

    rank_names = sorted(
        str(name) for name in model if str(name).startswith(_RANK_PREFIX)
    )
    if not rank_names or set(rank_names) & active_set:
        raise ProbeHealthEvidenceError("rank tower is absent or marked trainable")
    rank = fingerprint_named_tensors(model, rank_names)
    if (
        rank.get("sha256") != EXPECTED_RANK_SHA256
        or rank.get("nonfinite_count") != 0
        or args.get("stage_b_dense_duty_rank_source_rank_sha256")
        != EXPECTED_RANK_SHA256
    ):
        raise ProbeHealthEvidenceError("frozen rank tower identity drifted")
    return {
        "active": current_active,
        "initial_active_sha256": initial["active"]["sha256"],
        "token_veto": {
            "tensor_count": len(token),
            "element_count": token_elements,
            "optimizer_state_tensor_count": live_token,
            "parameter_names_sha256": hashlib.sha256(
                json.dumps(sorted(token), separators=(",", ":")).encode("ascii")
            ).hexdigest(),
        },
        "global_absolute": {
            "tensor_count": len(global_absolute),
            "element_count": global_elements,
            "optimizer_state_tensor_count": live_global,
            "parameter_names_sha256": hashlib.sha256(
                json.dumps(
                    sorted(global_absolute), separators=(",", ":")
                ).encode("ascii")
            ).hexdigest(),
        },
        "disjoint_and_complete": True,
        "rank": rank,
    }


def _audit_tail_queues(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeHealthEvidenceError("checkpoint criterion state is missing")
    expected_keys = {
        "tail_positive_queue",
        "tail_negative_queue",
        "tail_positive_ptr",
        "tail_negative_ptr",
        "tail_positive_count",
        "tail_negative_count",
    }
    if set(value) != expected_keys:
        raise ProbeHealthEvidenceError("criterion tail-queue schema drifted")
    queues = {}
    for label, key in (
        ("positive", "tail_positive_queue"),
        ("tn", "tail_negative_queue"),
    ):
        queue = value[key]
        if (
            not torch.is_tensor(queue)
            or queue.dtype != torch.float32
            or tuple(queue.shape) != (QUEUE_SIZE,)
            or not bool(torch.isfinite(queue).all().item())
        ):
            raise ProbeHealthEvidenceError(f"{label} tail queue is not full finite float32")
        queues[label] = queue.float()
    positive_count = _scalar_int_tensor(
        value["tail_positive_count"], label="positive queue count"
    )
    tn_count = _scalar_int_tensor(
        value["tail_negative_count"], label="TN queue count"
    )
    positive_ptr = _scalar_int_tensor(
        value["tail_positive_ptr"], label="positive queue pointer"
    )
    tn_ptr = _scalar_int_tensor(
        value["tail_negative_ptr"], label="TN queue pointer"
    )
    if positive_count != QUEUE_SIZE or tn_count != QUEUE_SIZE:
        raise ProbeHealthEvidenceError("both V45 tail queues must be full at U400")
    if not 0 <= positive_ptr < QUEUE_SIZE or not 0 <= tn_ptr < QUEUE_SIZE:
        raise ProbeHealthEvidenceError("V45 tail queue pointer is outside the ring")
    lower_index = max(1, math.ceil(0.05 * QUEUE_SIZE)) - 1
    positive_q05 = float(queues["positive"].sort().values[lower_index].item())
    tn_q95 = float(torch.quantile(queues["tn"], 0.95).item())
    return {
        "positive_count": positive_count,
        "tn_count": tn_count,
        "positive_pointer": positive_ptr,
        "tn_pointer": tn_ptr,
        "positive_q05": positive_q05,
        "tn_q95": tn_q95,
        "operating_gap": positive_q05 - tn_q95,
        "all_finite": True,
    }


def _audit_runtime(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != RUNTIME_SCHEMA:
        raise ProbeHealthEvidenceError("terminal V45 runtime audit schema is invalid")
    result = dict(value)
    integer_fields = (
        "optimizer_step_boundaries",
        "successful_optimizer_steps",
        "amp_skipped_optimizer_steps",
        "nonfinite_gradient_boundaries",
        "zero_gradient_successful_steps",
    )
    for field in integer_fields:
        result[field] = _exact_int(result.get(field), label=f"runtime.{field}")
    for head in ("token_veto", "global_absolute"):
        for field in (
            f"last_{head}_grad_norm_preclip",
            f"max_{head}_grad_norm_preclip",
        ):
            result[field] = _finite_float(result.get(field), label=f"runtime.{field}")
        for field in (
            f"nonfinite_{head}_gradient_boundaries",
            f"zero_{head}_gradient_successful_steps",
        ):
            raw = result.get(field, 0)
            result[field] = _exact_int(raw, label=f"runtime.{field}")
    for field in (
        "last_active_grad_norm_preclip",
        "max_active_grad_norm_preclip",
        "last_amp_scale",
        "min_amp_scale",
    ):
        result[field] = _finite_float(result.get(field), label=f"runtime.{field}")
    return result


_JOINT_CLIP_FIELDS = (
    "train_grad_norm_dense_duty_active_preclip",
    "train_grad_tensor_count_dense_duty_active",
    "train_grad_norm_dense_duty_token_veto_preclip",
    "train_grad_tensor_count_dense_duty_token_veto",
    "train_grad_norm_dense_duty_global_absolute_preclip",
    "train_grad_tensor_count_dense_duty_global_absolute",
    "train_grad_norm_dense_duty_token_veto_postclip",
    "train_grad_norm_dense_duty_global_absolute_postclip",
    "train_grad_norm_dense_duty_active_postclip",
    "train_amp_step_skipped",
)


def _audit_u222_log(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _file_record(path, label="V45 trajectory log")
    try:
        lines = [
            line
            for line in Path(before["path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise ProbeHealthEvidenceError(f"V45 trajectory log is unreadable: {error}") from error
    if len(lines) != 1:
        raise ProbeHealthEvidenceError("V45 trajectory log must contain exactly U222")
    try:
        row = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ProbeHealthEvidenceError(f"V45 U222 log is invalid JSON: {error}") from error
    if not isinstance(row, Mapping) or row.get("train_optimizer_updates") != (
        EXPECTED_LOG_UPDATES
    ):
        raise ProbeHealthEvidenceError("V45 trajectory row is not exact U222")
    result = dict(row)
    for field in _JOINT_CLIP_FIELDS:
        result[field] = _finite_float(result.get(field), label=f"U222.{field}")
    after = _file_record(path, label="V45 trajectory log")
    if before != after:
        raise ProbeHealthEvidenceError("V45 trajectory log changed while auditing")
    return result, before


def _health_checks(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    value = lambda name: float(trajectory[name])
    active_post = value("train_grad_norm_dense_duty_active_postclip")
    token_post = value("train_grad_norm_dense_duty_token_veto_postclip")
    global_post = value("train_grad_norm_dense_duty_global_absolute_postclip")
    return {
        "runtime_all_u400_boundaries_succeeded": _check(
            {
                "boundaries": runtime["optimizer_step_boundaries"],
                "successful": runtime["successful_optimizer_steps"],
            },
            "boundaries == successful == 400",
            runtime["optimizer_step_boundaries"] == EXPECTED_UPDATES
            and runtime["successful_optimizer_steps"] == EXPECTED_UPDATES,
        ),
        "runtime_amp_skips_zero": _check(
            runtime["amp_skipped_optimizer_steps"],
            "== 0",
            runtime["amp_skipped_optimizer_steps"] == 0,
        ),
        "runtime_nonfinite_gradients_zero": _check(
            {
                "active": runtime["nonfinite_gradient_boundaries"],
                "token_veto": runtime["nonfinite_token_veto_gradient_boundaries"],
                "global_absolute": runtime[
                    "nonfinite_global_absolute_gradient_boundaries"
                ],
            },
            "all == 0",
            runtime["nonfinite_gradient_boundaries"] == 0
            and runtime["nonfinite_token_veto_gradient_boundaries"] == 0
            and runtime["nonfinite_global_absolute_gradient_boundaries"] == 0,
        ),
        "runtime_no_zero_gradient_owner_steps": _check(
            {
                "active": runtime["zero_gradient_successful_steps"],
                "token_veto": runtime["zero_token_veto_gradient_successful_steps"],
                "global_absolute": runtime[
                    "zero_global_absolute_gradient_successful_steps"
                ],
            },
            "all == 0",
            runtime["zero_gradient_successful_steps"] == 0
            and runtime["zero_token_veto_gradient_successful_steps"] == 0
            and runtime["zero_global_absolute_gradient_successful_steps"] == 0,
        ),
        "runtime_both_split_owners_engaged": _check(
            {
                "last_token": runtime["last_token_veto_grad_norm_preclip"],
                "max_token": runtime["max_token_veto_grad_norm_preclip"],
                "last_global": runtime["last_global_absolute_grad_norm_preclip"],
                "max_global": runtime["max_global_absolute_grad_norm_preclip"],
            },
            "all > 0",
            all(
                float(runtime[name]) > 0.0
                for name in (
                    "last_token_veto_grad_norm_preclip",
                    "max_token_veto_grad_norm_preclip",
                    "last_global_absolute_grad_norm_preclip",
                    "max_global_absolute_grad_norm_preclip",
                )
            ),
        ),
        "runtime_amp_scale_positive": _check(
            {
                "minimum": runtime["min_amp_scale"],
                "last": runtime["last_amp_scale"],
            },
            "minimum and last >= 1",
            float(runtime["min_amp_scale"]) >= 1.0
            and float(runtime["last_amp_scale"]) >= 1.0,
        ),
        "u222_log_amp_skips_zero": _check(
            value("train_amp_step_skipped"),
            "== 0",
            value("train_amp_step_skipped") == 0.0,
        ),
        "u222_split_owner_live_counts_exact": _check(
            {
                "active": value("train_grad_tensor_count_dense_duty_active"),
                "token_veto": value(
                    "train_grad_tensor_count_dense_duty_token_veto"
                ),
                "global_absolute": value(
                    "train_grad_tensor_count_dense_duty_global_absolute"
                ),
            },
            "active=67, token-veto=21, global-absolute=46",
            math.isclose(
                value("train_grad_tensor_count_dense_duty_active"),
                EXPECTED_LIVE_TOKEN_TENSORS + EXPECTED_LIVE_GLOBAL_TENSORS,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                value("train_grad_tensor_count_dense_duty_token_veto"),
                EXPECTED_LIVE_TOKEN_TENSORS,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                value("train_grad_tensor_count_dense_duty_global_absolute"),
                EXPECTED_LIVE_GLOBAL_TENSORS,
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
        ),
        "u222_joint_clip_v3_evidence": _check(
            {
                "active_preclip": value(
                    "train_grad_norm_dense_duty_active_preclip"
                ),
                "token_preclip": value(
                    "train_grad_norm_dense_duty_token_veto_preclip"
                ),
                "global_preclip": value(
                    "train_grad_norm_dense_duty_global_absolute_preclip"
                ),
                "active_postclip": active_post,
                "token_postclip": token_post,
                "global_postclip": global_post,
            },
            "both owners live and one union clip yields total 0.1",
            value("train_grad_norm_dense_duty_active_preclip") > CLIP_MAX_NORM
            and value("train_grad_norm_dense_duty_token_veto_preclip") > 0.0
            and value("train_grad_norm_dense_duty_global_absolute_preclip") > 0.0
            and 0.099 <= active_post <= CLIP_MAX_NORM + 1e-6
            and 0.0 < token_post < active_post
            and 0.0 < global_post < active_post,
        ),
    }


def audit() -> dict[str, Any]:
    controller = _audit_controller_state(training.inspect())
    payload, checkpoint_record = _load_checkpoint(Path(training.CHECKPOINT))
    cursor = _audit_terminal_cursor(payload)
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise ProbeHealthEvidenceError("terminal V45 checkpoint args are missing")
    contract = _audit_training_contract(args)
    ownership = _audit_split_ownership(payload, args)
    if ownership["rank"]["sha256"] != controller["rank_sha256"]:
        raise ProbeHealthEvidenceError("controller/checkpoint rank identities differ")
    endpoint = _audit_tail_queues(payload.get("criterion"))
    runtime = _audit_runtime(args.get("stage_b_dense_duty_runtime_audit"))
    trajectory, log_record = _audit_u222_log(LOG_PATH)
    checks = _health_checks(runtime, trajectory)
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema": SCHEMA,
        "candidate": {
            "controller": controller,
            "checkpoint": checkpoint_record,
            "log": log_record,
        },
        "terminal_cursor_u400": cursor,
        "training_contract": contract,
        "split_head_ownership": ownership,
        "endpoint_u400": endpoint,
        "runtime": runtime,
        "trajectory_u222": {
            key: trajectory[key]
            for key in ("train_optimizer_updates", *_JOINT_CLIP_FIELDS)
        },
        "checks": checks,
        "failed_checks": failed,
        "decision": (
            "healthy_for_strict1607_diagnostic"
            if not failed
            else "unhealthy_do_not_run_diagnostic"
        ),
        "scope": "training_health_only_not_ref_or_tn_performance",
    }


def _default_output() -> Path:
    return Path(training.OUTPUT).parent / (
        "u000400_split_tail_aligned_health_audit.json"
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise OSError(f"health report destination must not be a symlink: {path}")
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
    except (OSError, RuntimeError, ValueError) as error:
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
        print(f"[FAIL] could not publish V45 health report: {error}", file=sys.stderr)
        return 2
    stream = sys.stdout if exit_code in (0, 1) else sys.stderr
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), file=stream)
    return exit_code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
