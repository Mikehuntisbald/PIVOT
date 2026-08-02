#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V52 terminal U400 confidence probe."""

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
    "tools/audit_stageb_confidence_adapter_candidate_split_independent_"
    "deployed_router_probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_candidate_sample_calibrator_probe_health_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_BASE_PATH}")
_V51 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V51)
_BASE = _V51._BASE
_CORE = _V51._CORE

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_sample_calibrator_probe_u0400 as training,
)
from util.stage_b_confidence_adapter_migration import (  # noqa: E402
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_ELEMENT_COUNT,
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_SHA256,
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_TENSOR_COUNT,
)
from util.stage_b_dense_duty_audit import (  # noqa: E402
    fingerprint_named_tensors,
    validate_initial_fingerprint,
)


SCHEMA = "pivot.stageb.confidence_adapter_candidate_sample_calibrator_probe_health/v1"
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v34"
EXPECTED_REVISION = "word_veto_candidate_sample_calibrator_split_v52"
EXPECTED_HEAD_CONTRACT = (
    "split_token_veto_candidate_absolute_sample_calibrator_v6"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_sample_calibrator_"
    "probe_u0400_20260802.py"
)
LOG_PATH = Path(training.OUTPUT) / "log.txt"

EXPECTED_ACTIVE_TENSORS = 74
EXPECTED_ACTIVE_ELEMENTS = 535_945
EXPECTED_TOKEN_TENSORS = 21
EXPECTED_TOKEN_ELEMENTS = 51_267
EXPECTED_CANDIDATE_TENSORS = 46
EXPECTED_CANDIDATE_ELEMENTS = 351_300
EXPECTED_SAMPLE_TENSORS = 7
EXPECTED_SAMPLE_ELEMENTS = 133_378
EXPECTED_LIVE_TOKEN_TENSORS = 21
EXPECTED_LIVE_CANDIDATE_TENSORS = 39
EXPECTED_LIVE_SAMPLE_TENSORS = 7

MIGRATION_FRESH_TENSOR_COUNT = EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_TENSOR_COUNT
MIGRATION_FRESH_ELEMENT_COUNT = EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_ELEMENT_COUNT
MIGRATION_FRESH_SHA256 = EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_SHA256
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
    "EXPECTED_LIVE_TOKEN_TENSORS",
):
    setattr(_BASE, _name, globals()[_name])
    setattr(_CORE, _name, globals()[_name])
_BASE.training = training
_CORE.training = training
_CORE.EXPECTED_CONTRACT_VALUES = {
    **_CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_confidence_head_gradient_contract": EXPECTED_HEAD_CONTRACT,
    "stage_b_dense_duty_deployed_veto_routing_weight": 0.0,
    "stage_b_dense_duty_deployed_veto_routing_reduction_contract": (
        "balanced_top_quarter_cvar_v2"
    ),
    "stage_b_v15_tail_queue_positive_trust_reduction_contract": (
        "top_quarter_cvar_v2"
    ),
    "stage_b_v15_tail_queue_negative_reduction_contract": "all_mean_v1",
    "stage_b_v21_token_edit_query_scope": "target_iou_v1",
}

_BASE_AUDIT = _V51._BASE_AUDIT
_BASE_AUDIT_RUNTIME = _V51._BASE_AUDIT_RUNTIME
_BASE_HEALTH_CHECKS = _V51._BASE_HEALTH_CHECKS
ProbeHealthEvidenceError = _V51.ProbeHealthEvidenceError

_CORE._JOINT_CLIP_FIELDS = (
    "train_grad_norm_dense_duty_active_preclip",
    "train_grad_tensor_count_dense_duty_active",
    "train_grad_norm_dense_duty_token_veto_preclip",
    "train_grad_tensor_count_dense_duty_token_veto",
    "train_grad_norm_dense_duty_candidate_absolute_preclip",
    "train_grad_tensor_count_dense_duty_candidate_absolute",
    "train_grad_norm_dense_duty_sample_calibrator_preclip",
    "train_grad_tensor_count_dense_duty_sample_calibrator",
    "train_grad_norm_dense_duty_token_veto_postclip",
    "train_grad_norm_dense_duty_candidate_absolute_postclip",
    "train_grad_norm_dense_duty_sample_calibrator_postclip",
    "train_grad_norm_dense_duty_active_postclip",
    "train_amp_step_skipped",
)


def _owner_for_name(name: str) -> str:
    if name.startswith(_CORE._CONFIDENCE_POOL_PREFIX):
        suffix = name[len(_CORE._CONFIDENCE_POOL_PREFIX) :]
        if suffix.startswith("residual."):
            return "sample_calibrator"
        raise ProbeHealthEvidenceError(f"unknown confidence-pool owner: {name}")
    if not name.startswith(_CORE._CONFIDENCE_ADAPTER_PREFIX):
        raise ProbeHealthEvidenceError(f"active parameter is outside confidence: {name}")
    suffix = name[len(_CORE._CONFIDENCE_ADAPTER_PREFIX) :]
    if suffix.startswith(_CORE._TOKEN_MODULE_PREFIXES) or suffix in (
        _CORE._TOKEN_OPTIONAL_PARAMETERS
    ):
        return "token_veto"
    if suffix == "candidate_coverage_depth_raw":
        return "sample_calibrator"
    if suffix.startswith(_CORE._GLOBAL_MODULE_PREFIXES) or suffix in (
        _CORE._GLOBAL_OPTIONAL_PARAMETERS - {"candidate_coverage_depth_raw"}
    ):
        return "candidate_absolute"
    raise ProbeHealthEvidenceError(f"unknown V52 parameter owner: {name}")


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
        raise ProbeHealthEvidenceError("V52 active parameter fingerprint drifted")

    owners = {
        "token_veto": [],
        "candidate_absolute": [],
        "sample_calibrator": [],
    }
    for name in names:
        owners[_owner_for_name(name)].append(name)
    expected = {
        "token_veto": (EXPECTED_TOKEN_TENSORS, EXPECTED_TOKEN_ELEMENTS),
        "candidate_absolute": (
            EXPECTED_CANDIDATE_TENSORS,
            EXPECTED_CANDIDATE_ELEMENTS,
        ),
        "sample_calibrator": (EXPECTED_SAMPLE_TENSORS, EXPECTED_SAMPLE_ELEMENTS),
    }
    if any(
        len(owners[label]) != tensor_count
        or sum(int(model[name].numel()) for name in owners[label]) != element_count
        for label, (tensor_count, element_count) in expected.items()
    ) or set().union(*(set(group) for group in owners.values())) != set(names):
        raise ProbeHealthEvidenceError("V52 three-owner partition is not exact")
    if any(
        set(owners[left]) & set(owners[right])
        for left, right in (
            ("token_veto", "candidate_absolute"),
            ("token_veto", "sample_calibrator"),
            ("candidate_absolute", "sample_calibrator"),
        )
    ):
        raise ProbeHealthEvidenceError("V52 confidence owners overlap")
    for name in names:
        value = model[name]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
            raise ProbeHealthEvidenceError(f"active parameter is non-finite: {name}")
    current_active = fingerprint_named_tensors(model, names)
    if (
        current_active["nonfinite_count"] != 0
        or current_active["sha256"] == initial["active"]["sha256"]
    ):
        raise ProbeHealthEvidenceError("V52 active state is non-finite or unchanged")

    param_groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(param_groups, list) or not isinstance(state, Mapping):
        raise ProbeHealthEvidenceError("V52 optimizer state is malformed")
    optimizer_ids: list[int] = []
    for group in param_groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ProbeHealthEvidenceError("V52 optimizer parameter group is malformed")
        optimizer_ids.extend(
            _CORE._exact_int(value, label="optimizer parameter id")
            for value in group["params"]
        )
    if (
        len(optimizer_ids) != EXPECTED_ACTIVE_TENSORS
        or len(set(optimizer_ids)) != EXPECTED_ACTIVE_TENSORS
        or set(optimizer_ids) != set(range(EXPECTED_ACTIVE_TENSORS))
    ):
        raise ProbeHealthEvidenceError("optimizer does not own the exact V52 set")

    active_set = set(names)
    ordered_names = [str(name) for name in model if str(name) in active_set]
    if len(ordered_names) != EXPECTED_ACTIVE_TENSORS:
        raise ProbeHealthEvidenceError("cannot replay V52 optimizer parameter order")
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
    expected_live = {
        "token_veto": EXPECTED_LIVE_TOKEN_TENSORS,
        "candidate_absolute": EXPECTED_LIVE_CANDIDATE_TENSORS,
        "sample_calibrator": EXPECTED_LIVE_SAMPLE_TENSORS,
    }
    if {label: len(group) for label, group in live_by_owner.items()} != expected_live:
        raise ProbeHealthEvidenceError("all three V52 owners were not optimized exactly")

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
    if not isinstance(value, Mapping):
        raise ProbeHealthEvidenceError("V52 runtime audit is missing")
    compat = dict(value)
    for prefix in ("last", "max"):
        candidate = _CORE._finite_float(
            compat.get(f"{prefix}_candidate_absolute_grad_norm_preclip"),
            label=f"runtime.{prefix}_candidate_absolute_grad_norm_preclip",
        )
        sample = _CORE._finite_float(
            compat.get(f"{prefix}_sample_calibrator_grad_norm_preclip"),
            label=f"runtime.{prefix}_sample_calibrator_grad_norm_preclip",
        )
        compat[f"{prefix}_global_absolute_grad_norm_preclip"] = math.hypot(
            candidate, sample
        )
    for prefix in ("nonfinite", "zero"):
        suffix = (
            "gradient_boundaries"
            if prefix == "nonfinite"
            else "gradient_successful_steps"
        )
        compat[f"{prefix}_global_absolute_{suffix}"] = sum(
            _CORE._exact_int(
                compat.get(f"{prefix}_{owner}_{suffix}", 0),
                label=f"runtime.{prefix}_{owner}_{suffix}",
            )
            for owner in ("candidate_absolute", "sample_calibrator")
        )
    result = _BASE_AUDIT_RUNTIME(compat)
    for owner in ("candidate_absolute", "sample_calibrator"):
        for prefix in ("last", "max"):
            field = f"{prefix}_{owner}_grad_norm_preclip"
            result[field] = _CORE._finite_float(
                value.get(field), label=f"runtime.{field}"
            )
        for prefix, suffix in (
            ("nonfinite", "gradient_boundaries"),
            ("zero", "gradient_successful_steps"),
        ):
            field = f"{prefix}_{owner}_{suffix}"
            result[field] = _CORE._exact_int(
                value.get(field, 0), label=f"runtime.{field}"
            )
    if result.get("clip_contract_schema") != (
        "pivot.stageb.dense_duty_three_owner_clip_contract/v1"
    ):
        raise ProbeHealthEvidenceError("V52 clip-contract schema is invalid")
    for field in (
        "clip_contract_checked_steps",
        "owner_clip_violation_steps",
        "active_pre_decomposition_violation_steps",
        "active_post_decomposition_violation_steps",
        "live_tensor_count_violation_steps",
        "active_monotonic_violation_steps",
    ):
        result[field] = _CORE._exact_int(
            result.get(field, 0), label=f"runtime.{field}"
        )
    for field in (
        "max_active_pre_decomposition_residual",
        "max_active_post_decomposition_residual",
        "max_owner_clip_residual",
        "max_active_monotonic_residual",
        "clip_contract_tolerance",
        "clip_contract_max_norm",
    ):
        result[field] = _CORE._finite_float(
            result.get(field), label=f"runtime.{field}"
        )
    for owner in ("token_veto", "candidate_absolute", "sample_calibrator"):
        for prefix in ("expected", "last_observed"):
            field = f"{prefix}_{owner}_tensor_count"
            result[field] = _CORE._exact_int(
                result.get(field), label=f"runtime.{field}"
            )
    return result


def _health_checks(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    compat_runtime = dict(runtime)
    compat_trajectory = dict(trajectory)
    for side in ("preclip", "postclip"):
        candidate = float(
            trajectory[f"train_grad_norm_dense_duty_candidate_absolute_{side}"]
        )
        sample = float(
            trajectory[f"train_grad_norm_dense_duty_sample_calibrator_{side}"]
        )
        compat_trajectory[f"train_grad_norm_dense_duty_global_absolute_{side}"] = (
            math.hypot(candidate, sample)
        )
    compat_trajectory["train_grad_tensor_count_dense_duty_global_absolute"] = (
        float(trajectory["train_grad_tensor_count_dense_duty_candidate_absolute"])
        + float(trajectory["train_grad_tensor_count_dense_duty_sample_calibrator"])
    )
    checks = dict(_BASE_HEALTH_CHECKS(compat_runtime, compat_trajectory))
    for name in (
        "runtime_nonfinite_gradients_zero",
        "runtime_no_zero_gradient_owner_steps",
        "runtime_both_split_owners_engaged",
        "u222_split_owner_live_counts_exact",
        "u222_joint_clip_v3_evidence",
        "u222_independent_clip_v2_evidence",
    ):
        checks.pop(name, None)

    owners = ("token_veto", "candidate_absolute", "sample_calibrator")
    checks["runtime_nonfinite_gradients_zero"] = _CORE._check(
        {
            "active": runtime["nonfinite_gradient_boundaries"],
            **{
                owner: runtime[f"nonfinite_{owner}_gradient_boundaries"]
                for owner in owners
            },
        },
        "active and all three owners == 0",
        runtime["nonfinite_gradient_boundaries"] == 0
        and all(
            runtime[f"nonfinite_{owner}_gradient_boundaries"] == 0
            for owner in owners
        ),
    )
    checks["runtime_no_zero_gradient_owner_steps"] = _CORE._check(
        {
            "active": runtime["zero_gradient_successful_steps"],
            **{
                owner: runtime[f"zero_{owner}_gradient_successful_steps"]
                for owner in owners
            },
        },
        "active and all three owners == 0",
        runtime["zero_gradient_successful_steps"] == 0
        and all(
            runtime[f"zero_{owner}_gradient_successful_steps"] == 0
            for owner in owners
        ),
    )
    checks["runtime_three_split_owners_engaged"] = _CORE._check(
        {
            owner: {
                "last": runtime[f"last_{owner}_grad_norm_preclip"],
                "max": runtime[f"max_{owner}_grad_norm_preclip"],
            }
            for owner in owners
        },
        "last and max gradient norms for all three owners > 0",
        all(
            float(runtime[f"last_{owner}_grad_norm_preclip"]) > 0.0
            and float(runtime[f"max_{owner}_grad_norm_preclip"]) > 0.0
            for owner in owners
        ),
    )
    expected_live = {
        "token_veto": EXPECTED_LIVE_TOKEN_TENSORS,
        "candidate_absolute": EXPECTED_LIVE_CANDIDATE_TENSORS,
        "sample_calibrator": EXPECTED_LIVE_SAMPLE_TENSORS,
    }
    violation_fields = (
        "owner_clip_violation_steps",
        "active_pre_decomposition_violation_steps",
        "active_post_decomposition_violation_steps",
        "live_tensor_count_violation_steps",
        "active_monotonic_violation_steps",
    )
    tolerance = float(runtime["clip_contract_tolerance"])
    checks["runtime_three_owner_clip_contract_exact"] = _CORE._check(
        {
            "checked_steps": runtime["clip_contract_checked_steps"],
            "violations": {field: runtime[field] for field in violation_fields},
            "expected": {
                owner: runtime[f"expected_{owner}_tensor_count"]
                for owner in owners
            },
            "observed": {
                owner: runtime[f"last_observed_{owner}_tensor_count"]
                for owner in owners
            },
        },
        "all 400 steps checked, zero violations, exact 21/39/7 live tensors",
        runtime["clip_contract_checked_steps"] == _CORE.EXPECTED_UPDATES
        and all(runtime[field] == 0 for field in violation_fields)
        and math.isclose(
            float(runtime["clip_contract_max_norm"]),
            _CORE.CLIP_MAX_NORM,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and all(
            runtime[f"expected_{owner}_tensor_count"] == count
            and runtime[f"last_observed_{owner}_tensor_count"] == count
            for owner, count in expected_live.items()
        )
        and float(runtime["max_active_pre_decomposition_residual"])
        <= tolerance
        and float(runtime["max_active_post_decomposition_residual"])
        <= tolerance
        and float(runtime["max_owner_clip_residual"]) <= tolerance
        and float(runtime["max_active_monotonic_residual"]) <= tolerance,
    )

    value = lambda name: float(trajectory[name])
    checks["u222_three_owner_live_counts_exact"] = _CORE._check(
        {
            "active": value("train_grad_tensor_count_dense_duty_active"),
            **{
                owner: value(f"train_grad_tensor_count_dense_duty_{owner}")
                for owner in owners
            },
        },
        "active=67, token-veto=21, candidate-absolute=39, sample-calibrator=7",
        math.isclose(
            value("train_grad_tensor_count_dense_duty_active"),
            sum(expected_live.values()),
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
    pre_l1 = sum(pre.values())
    post_l1 = sum(post.values())
    checks["u222_three_independent_clips_v6_evidence"] = _CORE._check(
        {
            "active_preclip": active_pre,
            "active_postclip": active_post,
            "owner_preclip_l2": pre_l2,
            "owner_preclip_l1": pre_l1,
            "owner_postclip_l2": post_l2,
            "owner_postclip_l1": post_l1,
            **{f"{owner}_preclip": item for owner, item in pre.items()},
            **{f"{owner}_postclip": item for owner, item in post.items()},
        },
        "three positive owners obey independent 0.1 clips and aggregate bounds",
        all(
            pre[owner] > 0.0
            and post[owner] > 0.0
            and post[owner]
            <= min(pre[owner], _CORE.CLIP_MAX_NORM) + 1e-6
            for owner in owners
        )
        and pre_l2 <= active_pre + 1e-6 <= pre_l1 + 1e-6
        and post_l2 <= active_post + 1e-6 <= post_l1 + 1e-6
        and active_post
        <= min(
            active_pre,
            math.sqrt(len(owners)) * _CORE.CLIP_MAX_NORM,
            post_l1,
        )
        + 1e-6,
    )
    return checks


_CORE._audit_split_ownership = _audit_split_ownership
_CORE._audit_runtime = _audit_runtime
_CORE._health_checks = _health_checks
_CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_candidate_sample_calibrator_health_audit.json"
)


def audit() -> dict[str, Any]:
    if MIGRATION_FINGERPRINT_IS_PLACEHOLDER:
        raise ProbeHealthEvidenceError(
            "V52 migration fresh-state SHA256 is still the all-zero placeholder"
        )
    return _BASE_AUDIT()


_CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
