#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V51 terminal U400 confidence probe."""

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
    "tools/audit_stageb_confidence_adapter_candidate_split_boundary_routing_"
    "probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_split_independent_deployed_router_probe_health_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
_CORE = _BASE._CORE

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_split_independent_deployed_router_probe_u0400 as training,
)
from util.stage_b_confidence_adapter_migration import (  # noqa: E402
    EXPECTED_DEPLOYED_ROUTER_FRESH_ELEMENT_COUNT,
    EXPECTED_DEPLOYED_ROUTER_FRESH_SHA256,
    EXPECTED_DEPLOYED_ROUTER_FRESH_TENSOR_COUNT,
)
from util.stage_b_dense_duty_audit import (  # noqa: E402
    fingerprint_named_tensors,
    validate_initial_fingerprint,
)


SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_split_independent_deployed_"
    "router_probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v33"
EXPECTED_REVISION = (
    "word_veto_candidate_split_independent_deployed_router_v51"
)
EXPECTED_HEAD_CONTRACT = (
    "split_token_veto_deployed_router_global_absolute_v5"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_split_independent_"
    "deployed_router_probe_u0400_20260802.py"
)
LOG_PATH = Path(training.OUTPUT) / "log.txt"

# V51 adds six independently owned router tensors (789 fp32 elements) to V47.
EXPECTED_ACTIVE_TENSORS = 80
EXPECTED_ACTIVE_ELEMENTS = 536_734
EXPECTED_TOKEN_TENSORS = 21
EXPECTED_TOKEN_ELEMENTS = 51_267
EXPECTED_ROUTER_TENSORS = 6
EXPECTED_ROUTER_ELEMENTS = 789
EXPECTED_GLOBAL_TENSORS = 53
EXPECTED_GLOBAL_ELEMENTS = 484_678
EXPECTED_LIVE_TOKEN_TENSORS = 21
EXPECTED_LIVE_ROUTER_TENSORS = 6
EXPECTED_LIVE_GLOBAL_TENSORS = 46

# The deterministic migration fingerprint is sealed in migration v18. Retain an
# explicit placeholder guard so an unsealed future edit cannot launch training.
MIGRATION_FRESH_TENSOR_COUNT = EXPECTED_DEPLOYED_ROUTER_FRESH_TENSOR_COUNT
MIGRATION_FRESH_ELEMENT_COUNT = EXPECTED_DEPLOYED_ROUTER_FRESH_ELEMENT_COUNT
MIGRATION_FRESH_SHA256 = EXPECTED_DEPLOYED_ROUTER_FRESH_SHA256
MIGRATION_FINGERPRINT_IS_PLACEHOLDER = MIGRATION_FRESH_SHA256 == "0" * 64

for _name in (
    "SCHEMA",
    "TRAINING_CONTRACT_SCHEMA",
    "EXPECTED_REVISION",
    "EXPECTED_HEAD_CONTRACT",
    "EXPECTED_CONFIG_ENTRY",
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
    "stage_b_dense_duty_confidence_head_gradient_contract": (
        EXPECTED_HEAD_CONTRACT
    ),
    "stage_b_dense_duty_deployed_veto_routing_weight": 0.1,
    "stage_b_dense_duty_deployed_veto_routing_reduction_contract": (
        "balanced_top_quarter_cvar_v2"
    ),
    "stage_b_v15_tail_queue_positive_trust_reduction_contract": (
        "top_quarter_cvar_v2"
    ),
    "stage_b_v15_tail_queue_negative_reduction_contract": "all_mean_v1",
    "stage_b_v21_token_edit_query_scope": "target_iou_v1",
}

_ROUTER_MODULE_PREFIXES = (
    "deployed_router_norm.",
    "deployed_router_residual.",
)
_BASE_AUDIT = _CORE.audit
_BASE_AUDIT_RUNTIME = _CORE._audit_runtime
_BASE_HEALTH_CHECKS = _BASE._health_checks
_CORE._JOINT_CLIP_FIELDS = tuple(
    dict.fromkeys(
        (*_CORE._JOINT_CLIP_FIELDS,
         "train_grad_norm_dense_duty_deployed_router_preclip",
         "train_grad_tensor_count_dense_duty_deployed_router",
         "train_grad_norm_dense_duty_deployed_router_postclip")
    )
)

ProbeHealthEvidenceError = _BASE.ProbeHealthEvidenceError


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
    if suffix.startswith(_ROUTER_MODULE_PREFIXES):
        return "deployed_router"
    if suffix.startswith(_CORE._TOKEN_MODULE_PREFIXES) or suffix in (
        _CORE._TOKEN_OPTIONAL_PARAMETERS
    ):
        return "token_veto"
    if suffix.startswith(_CORE._GLOBAL_MODULE_PREFIXES) or suffix in (
        _CORE._GLOBAL_OPTIONAL_PARAMETERS
    ):
        return "global_absolute"
    raise ProbeHealthEvidenceError(f"unknown three-head parameter owner: {name}")


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
        or any(name not in model for name in names)
    ):
        raise ProbeHealthEvidenceError("V51 active parameter fingerprint drifted")

    owners = {
        "token_veto": [],
        "deployed_router": [],
        "global_absolute": [],
    }
    for name in names:
        owners[_owner_for_name(name)].append(name)
    expected = {
        "token_veto": (EXPECTED_TOKEN_TENSORS, EXPECTED_TOKEN_ELEMENTS),
        "deployed_router": (EXPECTED_ROUTER_TENSORS, EXPECTED_ROUTER_ELEMENTS),
        "global_absolute": (EXPECTED_GLOBAL_TENSORS, EXPECTED_GLOBAL_ELEMENTS),
    }
    if any(
        len(owners[label]) != tensor_count
        or sum(int(model[name].numel()) for name in owners[label]) != element_count
        for label, (tensor_count, element_count) in expected.items()
    ) or set().union(*(set(group) for group in owners.values())) != set(names):
        raise ProbeHealthEvidenceError(
            "V51 token-veto/router/global-absolute ownership is not exact"
        )
    if any(
        set(owners[left]) & set(owners[right])
        for left, right in (
            ("token_veto", "deployed_router"),
            ("token_veto", "global_absolute"),
            ("deployed_router", "global_absolute"),
        )
    ):
        raise ProbeHealthEvidenceError("V51 confidence owners overlap")
    for name in names:
        value = model[name]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
            raise ProbeHealthEvidenceError(f"active parameter is non-finite: {name}")
    current_active = fingerprint_named_tensors(model, names)
    if (
        current_active["nonfinite_count"] != 0
        or current_active["sha256"] == initial["active"]["sha256"]
    ):
        raise ProbeHealthEvidenceError("V51 active state is non-finite or unchanged")

    param_groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(param_groups, list) or not isinstance(state, Mapping):
        raise ProbeHealthEvidenceError("V51 optimizer state is malformed")
    optimizer_ids: list[int] = []
    for group in param_groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ProbeHealthEvidenceError("V51 optimizer parameter group is malformed")
        optimizer_ids.extend(
            _CORE._exact_int(value, label="optimizer parameter id")
            for value in group["params"]
        )
    if (
        len(optimizer_ids) != EXPECTED_ACTIVE_TENSORS
        or len(set(optimizer_ids)) != EXPECTED_ACTIVE_TENSORS
        or set(optimizer_ids) != set(range(EXPECTED_ACTIVE_TENSORS))
    ):
        raise ProbeHealthEvidenceError("optimizer does not own the exact V51 set")

    active_set = set(names)
    ordered_names = [str(name) for name in model if str(name) in active_set]
    if len(ordered_names) != EXPECTED_ACTIVE_TENSORS:
        raise ProbeHealthEvidenceError("cannot replay V51 optimizer parameter order")
    state_ids = {
        _CORE._exact_int(value, label="optimizer state id") for value in state.keys()
    }
    if not state_ids.issubset(set(optimizer_ids)):
        raise ProbeHealthEvidenceError("optimizer state references an unknown parameter")
    live = [ordered_names[index] for index in sorted(state_ids)]
    live_by_owner = {
        label: [name for name in live if _owner_for_name(name) == label]
        for label in owners
    }
    if {
        label: len(group) for label, group in live_by_owner.items()
    } != {
        "token_veto": EXPECTED_LIVE_TOKEN_TENSORS,
        "deployed_router": EXPECTED_LIVE_ROUTER_TENSORS,
        "global_absolute": EXPECTED_LIVE_GLOBAL_TENSORS,
    }:
        raise ProbeHealthEvidenceError("all three V51 owners were not optimized")

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
        **{
            label: _owner_summary(
                names=owners[label], live=live_by_owner[label], model=model
            )
            for label in owners
        },
        "disjoint_and_complete": True,
        "rank": rank,
    }


def _audit_runtime(value: Any) -> dict[str, Any]:
    result = _BASE_AUDIT_RUNTIME(value)
    for field in (
        "last_deployed_router_grad_norm_preclip",
        "max_deployed_router_grad_norm_preclip",
    ):
        result[field] = _CORE._finite_float(
            result.get(field), label=f"runtime.{field}"
        )
    for field in (
        "nonfinite_deployed_router_gradient_boundaries",
        "zero_deployed_router_gradient_successful_steps",
    ):
        result[field] = _CORE._exact_int(
            result.get(field, 0), label=f"runtime.{field}"
        )
    return result


def _health_checks(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = dict(_BASE_HEALTH_CHECKS(runtime, trajectory))
    for name in (
        "runtime_nonfinite_gradients_zero",
        "runtime_no_zero_gradient_owner_steps",
        "runtime_both_split_owners_engaged",
        "u222_split_owner_live_counts_exact",
        "u222_independent_clip_v2_evidence",
    ):
        checks.pop(name, None)

    heads = ("token_veto", "deployed_router", "global_absolute")
    checks["runtime_nonfinite_gradients_zero"] = _CORE._check(
        {
            "active": runtime["nonfinite_gradient_boundaries"],
            **{
                head: runtime[f"nonfinite_{head}_gradient_boundaries"]
                for head in heads
            },
        },
        "active and all three owners == 0",
        runtime["nonfinite_gradient_boundaries"] == 0
        and all(
            runtime[f"nonfinite_{head}_gradient_boundaries"] == 0
            for head in heads
        ),
    )
    checks["runtime_no_zero_gradient_owner_steps"] = _CORE._check(
        {
            "active": runtime["zero_gradient_successful_steps"],
            **{
                head: runtime[f"zero_{head}_gradient_successful_steps"]
                for head in heads
            },
        },
        "active and all three owners == 0",
        runtime["zero_gradient_successful_steps"] == 0
        and all(
            runtime[f"zero_{head}_gradient_successful_steps"] == 0
            for head in heads
        ),
    )
    checks["runtime_three_split_owners_engaged"] = _CORE._check(
        {
            head: {
                "last": runtime[f"last_{head}_grad_norm_preclip"],
                "max": runtime[f"max_{head}_grad_norm_preclip"],
            }
            for head in heads
        },
        "last and max gradient norms for all three owners > 0",
        all(
            float(runtime[f"last_{head}_grad_norm_preclip"]) > 0.0
            and float(runtime[f"max_{head}_grad_norm_preclip"]) > 0.0
            for head in heads
        ),
    )

    value = lambda name: float(trajectory[name])
    expected_counts = {
        "token_veto": EXPECTED_LIVE_TOKEN_TENSORS,
        "deployed_router": EXPECTED_LIVE_ROUTER_TENSORS,
        "global_absolute": EXPECTED_LIVE_GLOBAL_TENSORS,
    }
    checks["u222_three_owner_live_counts_exact"] = _CORE._check(
        {
            "active": value("train_grad_tensor_count_dense_duty_active"),
            **{
                head: value(f"train_grad_tensor_count_dense_duty_{head}")
                for head in heads
            },
        },
        "active=73, token-veto=21, deployed-router=6, global-absolute=46",
        math.isclose(
            value("train_grad_tensor_count_dense_duty_active"),
            sum(expected_counts.values()),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and all(
            math.isclose(
                value(f"train_grad_tensor_count_dense_duty_{head}"),
                count,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for head, count in expected_counts.items()
        ),
    )

    pre = {
        head: value(f"train_grad_norm_dense_duty_{head}_preclip")
        for head in heads
    }
    post = {
        head: value(f"train_grad_norm_dense_duty_{head}_postclip")
        for head in heads
    }
    active_pre = value("train_grad_norm_dense_duty_active_preclip")
    active_post = value("train_grad_norm_dense_duty_active_postclip")
    aggregate_pre_l2_lower = math.sqrt(
        sum(item * item for item in pre.values())
    )
    aggregate_post_l2_lower = math.sqrt(
        sum(item * item for item in post.values())
    )
    per_step_post_l2_upper = math.sqrt(len(heads)) * _CORE.CLIP_MAX_NORM
    checks["u222_three_independent_clips_v5_evidence"] = _CORE._check(
        {
            "active_preclip": active_pre,
            "active_postclip": active_post,
            "aggregate_pre_l2_lower": aggregate_pre_l2_lower,
            "aggregate_post_l2_lower": aggregate_post_l2_lower,
            "per_step_post_l2_upper": per_step_post_l2_upper,
            **{f"{head}_preclip": item for head, item in pre.items()},
            **{f"{head}_postclip": item for head, item in post.items()},
        },
        (
            "U222 aggregate obeys the Jensen and upper bounds implied by "
            "three independent piecewise 0.1 clips"
        ),
        active_pre > _CORE.CLIP_MAX_NORM
        and all(
            pre[head] > 0.0
            and post[head] > 0.0
            and post[head] <= min(pre[head], _CORE.CLIP_MAX_NORM) + 1e-6
            for head in heads
        )
        and active_pre + 1e-6 >= aggregate_pre_l2_lower
        and active_post + 1e-6 >= aggregate_post_l2_lower
        and active_post <= per_step_post_l2_upper + 1e-6,
    )
    return checks


_CORE._audit_split_ownership = _audit_split_ownership
_CORE._audit_runtime = _audit_runtime
_CORE._health_checks = _health_checks
_CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_split_independent_deployed_router_health_audit.json"
)


def audit() -> dict[str, Any]:
    if MIGRATION_FINGERPRINT_IS_PLACEHOLDER:
        raise ProbeHealthEvidenceError(
            "V51 migration fresh-state SHA256 is still the all-zero placeholder"
        )
    return _BASE_AUDIT()


_CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
