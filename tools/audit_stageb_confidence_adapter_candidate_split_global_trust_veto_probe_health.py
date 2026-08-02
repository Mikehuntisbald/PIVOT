#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed mechanical health audit for the V49 terminal U400 probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_candidate_"
    "split_boundary_routing_probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_split_global_trust_veto_probe_health_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
_CORE = _BASE._CORE

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_split_global_trust_veto_probe_u0400 as training,
)


SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_split_global_trust_veto_"
    "probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v31"
TRAINING_CONTRACT_ARG = "stage_b_dense_duty_training_contract"
EXPECTED_REVISION = "word_veto_candidate_split_global_trust_veto_v49"
EXPECTED_HEAD_CONTRACT = "split_token_veto_global_trust_veto_v4"
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_global_trust_veto_probe_u0400_20260801.py"
)
LOG_PATH = Path(training.OUTPUT) / "log.txt"

EXPECTED_ACTIVE_TENSORS = 80
EXPECTED_ACTIVE_ELEMENTS = 669_322
EXPECTED_TOKEN_TENSORS = 21
EXPECTED_TOKEN_ELEMENTS = 51_267
EXPECTED_GLOBAL_TENSORS = 59
EXPECTED_GLOBAL_ELEMENTS = 618_055
EXPECTED_TRUST_TENSORS = 53
EXPECTED_TRUST_ELEMENTS = 484_678
EXPECTED_VETO_TENSORS = 6
EXPECTED_VETO_ELEMENTS = 133_377
EXPECTED_LIVE_TOKEN_TENSORS = 21
EXPECTED_LIVE_GLOBAL_TENSORS = 52
EXPECTED_LIVE_TRUST_TENSORS = 46
EXPECTED_LIVE_VETO_TENSORS = 6

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
    "EXPECTED_TRUST_TENSORS",
    "EXPECTED_TRUST_ELEMENTS",
    "EXPECTED_VETO_TENSORS",
    "EXPECTED_VETO_ELEMENTS",
    "EXPECTED_LIVE_TOKEN_TENSORS",
    "EXPECTED_LIVE_GLOBAL_TENSORS",
    "EXPECTED_LIVE_TRUST_TENSORS",
    "EXPECTED_LIVE_VETO_TENSORS",
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
    "stage_b_v11_trainable_params_min": EXPECTED_ACTIVE_ELEMENTS,
    "stage_b_v11_trainable_params_max": EXPECTED_ACTIVE_ELEMENTS,
}

ProbeHealthEvidenceError = _BASE.ProbeHealthEvidenceError
_VETO_POOL_PREFIX = "stage_b_fixed_text_scorer.confidence_veto_pool."
_BASE_OWNER_FOR_NAME = _CORE._owner_for_name
_BASE_AUDIT_SPLIT_OWNERSHIP = _CORE._audit_split_ownership
_BASE_AUDIT_RUNTIME = _CORE._audit_runtime
_BASE_AUDIT_TRAINING_CONTRACT = _CORE._audit_training_contract
_BASE_HEALTH_CHECKS = _BASE._health_checks


def _audit_training_contract(args: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_BASE_AUDIT_TRAINING_CONTRACT(args))
    saved = args.get(TRAINING_CONTRACT_ARG)
    values = saved.get("values") if isinstance(saved, Mapping) else None
    key = "stage_b_v15_tail_queue_negative_reduction_contract"
    observed = str(args.get(key, "")).strip()
    if (
        not isinstance(values, Mapping)
        or values.get(key) != "all_mean_v1"
        or observed != "all_mean_v1"
    ):
        raise ProbeHealthEvidenceError(
            "V49 training contract requires all_mean_v1 negative reduction"
        )
    result["negative_reduction_contract"] = observed
    return result


def _owner_for_name(name: str) -> str:
    if name.startswith(_VETO_POOL_PREFIX):
        suffix = name[len(_VETO_POOL_PREFIX) :]
        if suffix.startswith("residual."):
            return "global_absolute"
        raise ProbeHealthEvidenceError(f"unknown confidence-veto-pool owner: {name}")
    return _BASE_OWNER_FOR_NAME(name)


def _audit_split_ownership(
    payload: Mapping[str, Any], args: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(_BASE_AUDIT_SPLIT_OWNERSHIP(payload, args))
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    if not isinstance(model, Mapping) or not isinstance(optimizer, Mapping):
        raise ProbeHealthEvidenceError("V49 model/optimizer state is missing")
    initial = _CORE.validate_initial_fingerprint(
        args.get("stage_b_dense_duty_initial_state_fingerprint"),
        expected_phase="confidence",
    )
    names = [str(name) for name in initial["active_parameter_names"]]
    token = [name for name in names if _owner_for_name(name) == "token_veto"]
    veto = [name for name in names if name.startswith(_VETO_POOL_PREFIX)]
    trust = [
        name
        for name in names
        if _owner_for_name(name) == "global_absolute" and name not in set(veto)
    ]
    counts = {
        "token": (len(token), sum(int(model[name].numel()) for name in token)),
        "trust": (len(trust), sum(int(model[name].numel()) for name in trust)),
        "veto": (len(veto), sum(int(model[name].numel()) for name in veto)),
    }
    expected = {
        "token": (EXPECTED_TOKEN_TENSORS, EXPECTED_TOKEN_ELEMENTS),
        "trust": (EXPECTED_TRUST_TENSORS, EXPECTED_TRUST_ELEMENTS),
        "veto": (EXPECTED_VETO_TENSORS, EXPECTED_VETO_ELEMENTS),
    }
    owner_sets = (set(token), set(trust), set(veto))
    if (
        counts != expected
        or any(owner_sets[i] & owner_sets[j] for i in range(3) for j in range(i + 1, 3))
        or set().union(*owner_sets) != set(names)
    ):
        raise ProbeHealthEvidenceError(
            "V49 token/global-trust/global-veto ownership is not exact and disjoint"
        )

    state = optimizer.get("state")
    if not isinstance(state, Mapping):
        raise ProbeHealthEvidenceError("V49 optimizer state is malformed")
    active_set = set(names)
    ordered_names = [str(name) for name in model if str(name) in active_set]
    state_ids = {
        _CORE._exact_int(value, label="optimizer state id") for value in state.keys()
    }
    live = [ordered_names[index] for index in sorted(state_ids)]
    live_trust = sum(name in set(trust) for name in live)
    live_veto = sum(name in set(veto) for name in live)
    if (
        live_trust != EXPECTED_LIVE_TRUST_TENSORS
        or live_veto != EXPECTED_LIVE_VETO_TENSORS
    ):
        raise ProbeHealthEvidenceError("both V49 global subowners were not optimized")

    def record(group: list[str], live_count: int) -> dict[str, Any]:
        return {
            "tensor_count": len(group),
            "element_count": sum(int(model[name].numel()) for name in group),
            "optimizer_state_tensor_count": live_count,
            "parameter_names_sha256": hashlib.sha256(
                json.dumps(sorted(group), separators=(",", ":")).encode("ascii")
            ).hexdigest(),
        }

    result["global_trust"] = record(trust, live_trust)
    result["global_veto"] = record(veto, live_veto)
    result["three_way_disjoint_and_complete"] = True
    return result


def _audit_runtime(value: Any) -> dict[str, Any]:
    result = dict(_BASE_AUDIT_RUNTIME(value))
    for head in ("global_trust", "global_veto"):
        for field in (
            f"last_{head}_grad_norm_preclip",
            f"max_{head}_grad_norm_preclip",
        ):
            result[field] = _CORE._finite_float(
                result.get(field), label=f"runtime.{field}"
            )
        for field in (
            f"nonfinite_{head}_gradient_boundaries",
            f"zero_{head}_gradient_successful_steps",
        ):
            result[field] = _CORE._exact_int(
                result.get(field, 0), label=f"runtime.{field}"
            )
    return result


_EXTRA_TRAJECTORY_FIELDS = (
    "train_grad_norm_dense_duty_global_trust_preclip",
    "train_grad_tensor_count_dense_duty_global_trust",
    "train_grad_norm_dense_duty_global_veto_preclip",
    "train_grad_tensor_count_dense_duty_global_veto",
    "train_grad_norm_dense_duty_global_trust_postclip",
    "train_grad_norm_dense_duty_global_veto_postclip",
)
_CORE._JOINT_CLIP_FIELDS = tuple(
    dict.fromkeys((*_CORE._JOINT_CLIP_FIELDS, *_EXTRA_TRAJECTORY_FIELDS))
)


def _health_checks(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = dict(_BASE_HEALTH_CHECKS(runtime, trajectory))
    value = lambda name: float(trajectory[name])
    trust_pre = value("train_grad_norm_dense_duty_global_trust_preclip")
    veto_pre = value("train_grad_norm_dense_duty_global_veto_preclip")
    trust_post = value("train_grad_norm_dense_duty_global_trust_postclip")
    veto_post = value("train_grad_norm_dense_duty_global_veto_postclip")
    global_post = value("train_grad_norm_dense_duty_global_absolute_postclip")
    checks["runtime_global_subowners_engaged"] = _CORE._check(
        {
            head: {
                "last": runtime[f"last_{head}_grad_norm_preclip"],
                "max": runtime[f"max_{head}_grad_norm_preclip"],
                "nonfinite": runtime[f"nonfinite_{head}_gradient_boundaries"],
                "zero": runtime[f"zero_{head}_gradient_successful_steps"],
            }
            for head in ("global_trust", "global_veto")
        },
        "trust and veto are finite and nonzero on every successful step",
        all(
            float(runtime[f"last_{head}_grad_norm_preclip"]) > 0.0
            and float(runtime[f"max_{head}_grad_norm_preclip"]) > 0.0
            and runtime[f"nonfinite_{head}_gradient_boundaries"] == 0
            and runtime[f"zero_{head}_gradient_successful_steps"] == 0
            for head in ("global_trust", "global_veto")
        ),
    )
    checks["u222_global_subowner_live_counts_exact"] = _CORE._check(
        {
            "global": value("train_grad_tensor_count_dense_duty_global_absolute"),
            "trust": value("train_grad_tensor_count_dense_duty_global_trust"),
            "veto": value("train_grad_tensor_count_dense_duty_global_veto"),
        },
        "global=52, trust=46, veto=6",
        math.isclose(
            value("train_grad_tensor_count_dense_duty_global_trust"),
            EXPECTED_LIVE_TRUST_TENSORS,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            value("train_grad_tensor_count_dense_duty_global_veto"),
            EXPECTED_LIVE_VETO_TENSORS,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            value("train_grad_tensor_count_dense_duty_global_absolute"),
            EXPECTED_LIVE_TRUST_TENSORS + EXPECTED_LIVE_VETO_TENSORS,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
    )
    checks["u222_global_union_clip_preserves_both_subowners"] = _CORE._check(
        {
            "trust_preclip": trust_pre,
            "veto_preclip": veto_pre,
            "trust_postclip": trust_post,
            "veto_postclip": veto_post,
            "global_postclip": global_post,
        },
        (
            "both subowners remain live and their mean norms satisfy the triangle "
            "bounds of the 0.1 global union clip"
        ),
        trust_pre > 0.0
        and veto_pre > 0.0
        and trust_post > 0.0
        and veto_post > 0.0
        # The persisted trajectory contains means of per-step norms. Norms of
        # those means need not close by Pythagoras when ownership ratios vary.
        and max(trust_post, veto_post) <= global_post + 1e-6
        and global_post <= trust_post + veto_post + 1e-6
        and 0.099 <= global_post <= _CORE.CLIP_MAX_NORM + 1e-6,
    )
    return checks


_CORE._owner_for_name = _owner_for_name
_CORE._audit_split_ownership = _audit_split_ownership
_CORE._audit_runtime = _audit_runtime
_CORE._audit_training_contract = _audit_training_contract
_CORE._health_checks = _health_checks
_CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_split_global_trust_veto_health_audit.json"
)


def audit() -> dict[str, Any]:
    return _CORE.audit()


def run(argv: Sequence[str] | None = None) -> int:
    return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
