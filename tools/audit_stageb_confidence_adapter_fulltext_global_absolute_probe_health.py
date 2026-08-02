#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V53 terminal U400 confidence probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_candidate_sample_calibrator_"
    "probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_fulltext_global_absolute_probe_health_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_BASE_PATH}")
_V52 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V52)
_BASE = _V52._BASE
_CORE = _V52._CORE

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_fulltext_global_absolute_probe_u0400 as training,
)
from util.stage_b_confidence_adapter_migration import (  # noqa: E402
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT,
    FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
    FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT,
    validate_confidence_adapter_migration_audit,
)
from util.stage_b_dense_duty_audit import (  # noqa: E402
    fingerprint_named_tensors,
    validate_initial_fingerprint,
)


SCHEMA = "pivot.stageb.confidence_adapter_fulltext_global_absolute_probe_health/v1"
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v35"
EXPECTED_REVISION = "word_veto_rank_full_expression_global_absolute_v53"
EXPECTED_HEAD_CONTRACT = FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
EXPECTED_POOL_CONTRACT = FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
EXPECTED_GATE_CONTRACT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "probe_u0400_20260802.py"
)
EXPECTED_SOURCE_UPDATES = 6551
EXPECTED_UPDATES = 400
LOG_PATH = Path(training.OUTPUT) / "log.txt"

EXPECTED_ACTIVE_TENSORS = 65
EXPECTED_ACTIVE_ELEMENTS = 534_725
EXPECTED_TOKEN_TENSORS = 21
EXPECTED_TOKEN_ELEMENTS = 51_267
EXPECTED_GLOBAL_TENSORS = 44
EXPECTED_GLOBAL_ELEMENTS = 483_458
EXPECTED_LIVE_TOKEN_TENSORS = EXPECTED_TOKEN_TENSORS
EXPECTED_LIVE_GLOBAL_TENSORS = EXPECTED_GLOBAL_TENSORS

EXPECTED_ADAPTER_TENSORS = EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT
EXPECTED_POOL_TENSORS = EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT
MIGRATION_FRESH_TENSOR_COUNT = EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT
MIGRATION_FRESH_ELEMENT_COUNT = EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT
MIGRATION_FRESH_STORAGE_BYTES = EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES
MIGRATION_FRESH_SHA256 = EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256
MIGRATION_FINGERPRINT_IS_PLACEHOLDER = MIGRATION_FRESH_SHA256 == "0" * 64
TWO_OWNER_CLIP_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_two_owner_clip_contract/v1"
)

for _name in (
    "SCHEMA",
    "TRAINING_CONTRACT_SCHEMA",
    "EXPECTED_REVISION",
    "EXPECTED_HEAD_CONTRACT",
    "EXPECTED_CONFIG_ENTRY",
    "EXPECTED_UPDATES",
    "LOG_PATH",
    "EXPECTED_ACTIVE_TENSORS",
    "EXPECTED_ACTIVE_ELEMENTS",
    "EXPECTED_TOKEN_TENSORS",
    "EXPECTED_TOKEN_ELEMENTS",
    "EXPECTED_GLOBAL_TENSORS",
    "EXPECTED_GLOBAL_ELEMENTS",
    "EXPECTED_LIVE_TOKEN_TENSORS",
    "EXPECTED_LIVE_GLOBAL_TENSORS",
):
    setattr(_BASE, _name, globals()[_name])
    setattr(_CORE, _name, globals()[_name])
_BASE.training = training
_CORE.training = training
_CORE.EXPECTED_CONTRACT_VALUES = {
    **_CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_confidence_head_gradient_contract": EXPECTED_HEAD_CONTRACT,
    "stage_b_dense_duty_confidence_pool_feature_contract": EXPECTED_POOL_CONTRACT,
    "stage_b_dense_duty_confidence_gate_gradient_contract": EXPECTED_GATE_CONTRACT,
    "stage_b_dense_duty_deployed_veto_routing_weight": 0.0,
    "stage_b_dense_duty_deployed_veto_routing_reduction_contract": (
        "balanced_top_quarter_cvar_v2"
    ),
    "stage_b_v15_tail_queue_positive_trust_reduction_contract": (
        "top_quarter_cvar_v2"
    ),
    "stage_b_v15_tail_queue_negative_reduction_contract": "all_mean_v1",
    "stage_b_v21_token_edit_query_scope": "target_iou_v1",
    "stage_b_v11_trainable_params_min": EXPECTED_ACTIVE_ELEMENTS,
    "stage_b_v11_trainable_params_max": EXPECTED_ACTIVE_ELEMENTS,
}

_BASE_AUDIT = _V52._BASE_AUDIT
_BASE_AUDIT_RUNTIME = _V52._BASE_AUDIT_RUNTIME
_BASE_HEALTH_CHECKS = _V52._BASE_HEALTH_CHECKS
ProbeHealthEvidenceError = _V52.ProbeHealthEvidenceError

_CORE._JOINT_CLIP_FIELDS = (
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

_FORBIDDEN_PARAMETER_FRAGMENTS = (
    "deployed_router",
    ".patch_residual.",
    ".global_query_norm.",
    "veto_cap_raw_ceiling",
    "candidate_patch_scale_raw",
    "candidate_veto_depth_raw",
    "candidate_coverage_depth_raw",
)
_FORBIDDEN_RUNTIME_OWNER_FRAGMENTS = (
    "candidate_absolute",
    "sample_calibrator",
    "deployed_router",
)


def _owner_for_name(name: str) -> str:
    if name.startswith(_CORE._CONFIDENCE_POOL_PREFIX):
        suffix = name[len(_CORE._CONFIDENCE_POOL_PREFIX) :]
        if suffix.startswith("residual."):
            return "global_absolute"
        raise ProbeHealthEvidenceError(f"unknown confidence-pool owner: {name}")
    if not name.startswith(_CORE._CONFIDENCE_ADAPTER_PREFIX):
        raise ProbeHealthEvidenceError(
            f"active parameter is outside confidence: {name}"
        )
    suffix = name[len(_CORE._CONFIDENCE_ADAPTER_PREFIX) :]
    if suffix.startswith(_CORE._TOKEN_MODULE_PREFIXES) or suffix in (
        _CORE._TOKEN_OPTIONAL_PARAMETERS
    ):
        return "token_veto"
    if suffix.startswith(_CORE._GLOBAL_MODULE_PREFIXES) or suffix in (
        _CORE._GLOBAL_OPTIONAL_PARAMETERS
    ):
        return "global_absolute"
    raise ProbeHealthEvidenceError(f"unknown V53 parameter owner: {name}")


def _owner_summary(
    *, names: Sequence[str], live: Sequence[str], model: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "tensor_count": len(names),
        "element_count": sum(int(model[name].numel()) for name in names),
        "optimizer_state_tensor_count": len(live),
        "parameter_names_sha256": hashlib.sha256(
            json.dumps(sorted(names), separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


def _audit_v53_migration(args: Mapping[str, Any]) -> dict[str, Any]:
    if args.get("stage_b_dense_duty_rank_source_optimizer_updates") != (
        EXPECTED_SOURCE_UPDATES
    ):
        raise ProbeHealthEvidenceError("V53 migration is not fresh from U6551")
    try:
        audit = validate_confidence_adapter_migration_audit(
            args.get("stage_b_dense_duty_confidence_adapter_migration_audit"),
            source_checkpoint_sha256=str(
                args.get("stage_b_dense_duty_rank_source_checkpoint_sha256", "")
            ),
            source_optimizer_updates=EXPECTED_SOURCE_UPDATES,
            source_checkpoint_reason=str(
                args.get("stage_b_dense_duty_rank_source_checkpoint_reason", "")
            ),
            rank_sha256=str(
                args.get("stage_b_dense_duty_rank_source_rank_sha256", "")
            ),
            transferred_sha256=str(
                args.get("stage_b_dense_duty_rank_source_transferred_sha256", "")
            ),
        )
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(str(error)) from error
    fresh = audit.get("fresh_confidence")
    if (
        audit.get("schema") != FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
        or audit.get("fresh_confidence_contract")
        != FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
        or audit.get("head_gradient_contract") != EXPECTED_HEAD_CONTRACT
        or audit.get("pool_feature_contract") != EXPECTED_POOL_CONTRACT
        or audit.get("confidence_parameter_tensor_count")
        != EXPECTED_ACTIVE_TENSORS
        or audit.get("confidence_parameter_element_count")
        != EXPECTED_ACTIVE_ELEMENTS
        or not isinstance(fresh, Mapping)
        or fresh.get("tensor_count") != MIGRATION_FRESH_TENSOR_COUNT
        or fresh.get("element_count") != MIGRATION_FRESH_ELEMENT_COUNT
        or fresh.get("storage_bytes") != MIGRATION_FRESH_STORAGE_BYTES
        or fresh.get("sha256") != MIGRATION_FRESH_SHA256
    ):
        raise ProbeHealthEvidenceError("V53 migration v20 surface drifted")
    return dict(audit)


def _audit_split_ownership(
    payload: Mapping[str, Any], args: Mapping[str, Any]
) -> dict[str, Any]:
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    if not isinstance(model, Mapping) or not isinstance(optimizer, Mapping):
        raise ProbeHealthEvidenceError("checkpoint model/optimizer state is missing")
    migration = _audit_v53_migration(args)
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
        or any(name not in model for name in names)
        or any(
            fragment in name
            for name in names
            for fragment in _FORBIDDEN_PARAMETER_FRAGMENTS
        )
    ):
        raise ProbeHealthEvidenceError("V53 active parameter fingerprint drifted")

    adapter_names = [
        name for name in names if name.startswith(_CORE._CONFIDENCE_ADAPTER_PREFIX)
    ]
    pool_names = [
        name for name in names if name.startswith(_CORE._CONFIDENCE_POOL_PREFIX)
    ]
    if (
        len(adapter_names) != EXPECTED_ADAPTER_TENSORS
        or len(pool_names) != EXPECTED_POOL_TENSORS
        or set(adapter_names) | set(pool_names) != set(names)
    ):
        raise ProbeHealthEvidenceError("V53 adapter/pool parameter surface drifted")

    owners = {"token_veto": [], "global_absolute": []}
    for name in names:
        owners[_owner_for_name(name)].append(name)
    expected = {
        "token_veto": (EXPECTED_TOKEN_TENSORS, EXPECTED_TOKEN_ELEMENTS),
        "global_absolute": (EXPECTED_GLOBAL_TENSORS, EXPECTED_GLOBAL_ELEMENTS),
    }
    if (
        any(
            len(owners[label]) != tensor_count
            or sum(int(model[name].numel()) for name in owners[label]) != element_count
            for label, (tensor_count, element_count) in expected.items()
        )
        or set(owners["token_veto"]) & set(owners["global_absolute"])
        or set(owners["token_veto"]) | set(owners["global_absolute"]) != set(names)
    ):
        raise ProbeHealthEvidenceError("V53 two-owner partition is not exact")
    for name in names:
        value = model[name]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
            raise ProbeHealthEvidenceError(f"active parameter is non-finite: {name}")
    current_active = fingerprint_named_tensors(model, names)
    if (
        current_active["nonfinite_count"] != 0
        or current_active["sha256"] == initial["active"]["sha256"]
    ):
        raise ProbeHealthEvidenceError("V53 active state is non-finite or unchanged")

    param_groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(param_groups, list) or not isinstance(state, Mapping):
        raise ProbeHealthEvidenceError("V53 optimizer state is malformed")
    optimizer_ids: list[int] = []
    for group in param_groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ProbeHealthEvidenceError("V53 optimizer parameter group is malformed")
        optimizer_ids.extend(
            _CORE._exact_int(value, label="optimizer parameter id")
            for value in group["params"]
        )
    exact_ids = set(range(EXPECTED_ACTIVE_TENSORS))
    if (
        len(optimizer_ids) != EXPECTED_ACTIVE_TENSORS
        or len(set(optimizer_ids)) != EXPECTED_ACTIVE_TENSORS
        or set(optimizer_ids) != exact_ids
    ):
        raise ProbeHealthEvidenceError("optimizer does not own the exact V53 set")

    active_set = set(names)
    ordered_names = [str(name) for name in model if str(name) in active_set]
    if len(ordered_names) != EXPECTED_ACTIVE_TENSORS:
        raise ProbeHealthEvidenceError("cannot replay V53 optimizer parameter order")
    state_ids = {
        _CORE._exact_int(value, label="optimizer state id") for value in state.keys()
    }
    if state_ids != exact_ids:
        raise ProbeHealthEvidenceError(
            "all 65 V53 parameters must have optimizer state"
        )
    live = [ordered_names[index] for index in sorted(state_ids)]
    live_by_owner = {
        label: [name for name in live if _owner_for_name(name) == label]
        for label in owners
    }
    if {label: len(group) for label, group in live_by_owner.items()} != {
        "token_veto": EXPECTED_LIVE_TOKEN_TENSORS,
        "global_absolute": EXPECTED_LIVE_GLOBAL_TENSORS,
    }:
        raise ProbeHealthEvidenceError("both V53 owners were not optimized exactly")

    rank_names = sorted(
        str(name) for name in model if str(name).startswith(_CORE._RANK_PREFIX)
    )
    if not rank_names or set(rank_names) & active_set:
        raise ProbeHealthEvidenceError("rank tower is absent or marked trainable")
    rank = fingerprint_named_tensors(model, rank_names)
    if (
        rank.get("sha256") != _CORE.EXPECTED_RANK_SHA256
        or rank.get("nonfinite_count") != 0
        or args.get("stage_b_dense_duty_rank_source_rank_sha256")
        != _CORE.EXPECTED_RANK_SHA256
    ):
        raise ProbeHealthEvidenceError("frozen rank tower identity drifted")
    return {
        "active": current_active,
        "initial_active_sha256": initial["active"]["sha256"],
        "migration": migration,
        **{
            label: _owner_summary(
                names=owners[label], live=live_by_owner[label], model=model
            )
            for label in owners
        },
        "adapter_tensor_count": len(adapter_names),
        "pool_tensor_count": len(pool_names),
        "disjoint_and_complete": True,
        "rank": rank,
    }


def _audit_runtime(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        fragment in str(key)
        for key in value
        for fragment in _FORBIDDEN_RUNTIME_OWNER_FRAGMENTS
    ):
        raise ProbeHealthEvidenceError(
            "V53 runtime must contain only token-veto/global-absolute owners"
        )
    result = _BASE_AUDIT_RUNTIME(value)
    if result.get("clip_contract_schema") != TWO_OWNER_CLIP_CONTRACT_SCHEMA:
        raise ProbeHealthEvidenceError("V53 two-owner clip-contract schema is invalid")
    integer_fields = (
        "clip_contract_checked_steps",
        "owner_clip_violation_steps",
        "active_pre_decomposition_violation_steps",
        "active_post_decomposition_violation_steps",
        "live_tensor_count_violation_steps",
        "active_monotonic_violation_steps",
    )
    for field in integer_fields:
        result[field] = _CORE._exact_int(
            result.get(field, 0), label=f"runtime.{field}"
        )
    float_fields = (
        "max_active_pre_decomposition_residual",
        "max_active_post_decomposition_residual",
        "max_owner_clip_residual",
        "max_active_monotonic_residual",
        "clip_contract_tolerance",
        "clip_contract_max_norm",
    )
    for field in float_fields:
        result[field] = _CORE._finite_float(
            result.get(field), label=f"runtime.{field}"
        )
    for owner in ("token_veto", "global_absolute"):
        for prefix in ("expected", "last_observed"):
            field = f"{prefix}_{owner}_tensor_count"
            result[field] = _CORE._exact_int(
                result.get(field), label=f"runtime.{field}"
            )

    tolerance = result["clip_contract_tolerance"]
    violations = integer_fields[1:]
    owner_counts = {
        "token_veto": EXPECTED_LIVE_TOKEN_TENSORS,
        "global_absolute": EXPECTED_LIVE_GLOBAL_TENSORS,
    }
    exact_runtime = (
        result["optimizer_step_boundaries"] == EXPECTED_UPDATES
        and result["successful_optimizer_steps"] == EXPECTED_UPDATES
        and result["amp_skipped_optimizer_steps"] == 0
        and result["nonfinite_gradient_boundaries"] == 0
        and result["zero_gradient_successful_steps"] == 0
        and result["clip_contract_checked_steps"] == EXPECTED_UPDATES
        and all(result[field] == 0 for field in violations)
        and tolerance > 0.0
        and math.isclose(
            result["clip_contract_max_norm"],
            _CORE.CLIP_MAX_NORM,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and all(
            result[f"expected_{owner}_tensor_count"] == count
            and result[f"last_observed_{owner}_tensor_count"] == count
            and result[f"last_{owner}_grad_norm_preclip"] > 0.0
            and result[f"max_{owner}_grad_norm_preclip"] > 0.0
            and result[f"nonfinite_{owner}_gradient_boundaries"] == 0
            and result[f"zero_{owner}_gradient_successful_steps"] == 0
            for owner, count in owner_counts.items()
        )
        and all(result[field] <= tolerance for field in float_fields[:4])
    )
    if not exact_runtime:
        raise ProbeHealthEvidenceError(
            "V53 runtime lacks exact 400-step two-owner clip evidence"
        )
    return result


def _health_checks(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = dict(_BASE_HEALTH_CHECKS(runtime, trajectory))
    for name in (
        "u222_split_owner_live_counts_exact",
        "u222_joint_clip_v3_evidence",
        "u222_independent_clip_v2_evidence",
    ):
        checks.pop(name, None)

    owners = ("token_veto", "global_absolute")
    expected_live = {
        "token_veto": EXPECTED_LIVE_TOKEN_TENSORS,
        "global_absolute": EXPECTED_LIVE_GLOBAL_TENSORS,
    }
    value = lambda name: float(trajectory[name])
    checks["u222_v53_owner_live_counts_exact"] = _CORE._check(
        {
            "active": value("train_grad_tensor_count_dense_duty_active"),
            **{
                owner: value(f"train_grad_tensor_count_dense_duty_{owner}")
                for owner in owners
            },
        },
        "active=65, token-veto=21, global-absolute=44",
        math.isclose(
            value("train_grad_tensor_count_dense_duty_active"),
            EXPECTED_ACTIVE_TENSORS,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and all(
            math.isclose(
                value(f"train_grad_tensor_count_dense_duty_{owner}"),
                count,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for owner, count in expected_live.items()
        ),
    )

    pre = {
        owner: value(f"train_grad_norm_dense_duty_{owner}_preclip")
        for owner in owners
    }
    post = {
        owner: value(f"train_grad_norm_dense_duty_{owner}_postclip")
        for owner in owners
    }
    active_pre = value("train_grad_norm_dense_duty_active_preclip")
    active_post = value("train_grad_norm_dense_duty_active_postclip")
    pre_l2 = math.sqrt(sum(item * item for item in pre.values()))
    post_l2 = math.sqrt(sum(item * item for item in post.values()))
    checks["u222_v53_two_independent_clips_exact"] = _CORE._check(
        {
            "active_preclip": active_pre,
            "active_postclip": active_post,
            "owner_preclip_l2": pre_l2,
            "owner_postclip_l2": post_l2,
            **{f"{owner}_preclip": item for owner, item in pre.items()},
            **{f"{owner}_postclip": item for owner, item in post.items()},
        },
        "two positive owners obey independent 0.1 clips and aggregate bounds",
        value("train_amp_step_skipped") == 0.0
        and all(
            pre[owner] > 0.0
            and post[owner] > 0.0
            and post[owner]
            <= min(pre[owner], _CORE.CLIP_MAX_NORM) + 1e-6
            for owner in owners
        )
        and pre_l2 <= active_pre + 1e-6 <= sum(pre.values()) + 1e-6
        and post_l2 <= active_post + 1e-6 <= sum(post.values()) + 1e-6
        and active_post
        <= min(
            active_pre,
            math.sqrt(len(owners)) * _CORE.CLIP_MAX_NORM,
            sum(post.values()),
        )
        + 1e-6,
    )
    return checks


_CORE._audit_split_ownership = _audit_split_ownership
_CORE._audit_runtime = _audit_runtime
_CORE._health_checks = _health_checks
_CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_fulltext_global_absolute_health_audit.json"
)


def audit() -> dict[str, Any]:
    if MIGRATION_FINGERPRINT_IS_PLACEHOLDER:
        raise ProbeHealthEvidenceError(
            "V53 migration fresh-state SHA256 is still the all-zero placeholder"
        )
    return _BASE_AUDIT()


_CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
