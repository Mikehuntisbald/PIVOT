#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for candidate-complete trace C1 U400."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

from tools import (
    audit_stageb_confidence_full_decoder_verifier_probe_health as _v61_baseline,
)


_SHARED_OVERRIDE_NAMES = (
    "SCHEMA",
    "TRAINING_CONTRACT_SCHEMA",
    "EXPECTED_CAPACITY_CONTRACT",
    "EXPECTED_VARIANT",
    "EXPECTED_CONFIG_ENTRY",
    "EXPECTED_UPDATES",
    "EXPECTED_TERMINAL_EPOCH",
    "EXPECTED_TERMINAL_ITERATION",
    "LOG_PATH",
    "EXPECTED_ACTIVE_TENSORS",
    "EXPECTED_ACTIVE_ELEMENTS",
    "EXPECTED_TOKEN_TENSORS",
    "EXPECTED_TOKEN_ELEMENTS",
    "EXPECTED_GLOBAL_TENSORS",
    "EXPECTED_GLOBAL_ELEMENTS",
    "EXPECTED_LIVE_TOKEN_TENSORS",
    "EXPECTED_LIVE_GLOBAL_TENSORS",
    "EXPECTED_ADAPTER_TENSORS",
    "EXPECTED_POOL_TENSORS",
    "MIGRATION_SCHEMA",
    "FRESH_CONFIDENCE_CONTRACT",
    "FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA",
    "FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT",
    "training",
)
_BASELINE_MODULES = (
    _v61_baseline,
    _v61_baseline._V60,
    _v61_baseline._V59,
    _v61_baseline._V56,
    _v61_baseline._V55,
    _v61_baseline._V53,
    _v61_baseline._BASE,
    _v61_baseline._CORE,
)
_MISSING = object()
_BASELINE_MODULE_STATE = {
    id(module): (
        module,
        {
            name: getattr(module, name, _MISSING)
            for name in _SHARED_OVERRIDE_NAMES
        },
    )
    for module in _BASELINE_MODULES
}
_BASELINE_CORE_STATE = {
    name: getattr(_v61_baseline._CORE, name)
    for name in (
        "_CONFIDENCE_ADAPTER_PREFIX",
        "_CONFIDENCE_POOL_PREFIX",
        "_default_output",
        "EXPECTED_CONTRACT_VALUES",
        "_audit_split_ownership",
        "_audit_runtime",
        "_health_checks",
        "_audit_training_contract",
        "audit",
    )
}
_BASELINE_HOOK_STATE = {
    (_v61_baseline._V53, "_owner_for_name"): _v61_baseline._V53._owner_for_name,
    (_v61_baseline._V53, "_audit_v53_migration"): (
        _v61_baseline._V53._audit_v53_migration
    ),
    (_v61_baseline._V55, "_audit_v55_migration"): (
        _v61_baseline._V55._audit_v55_migration
    ),
}

from tools import (  # noqa: E402
    audit_stageb_confidence_full_decoder_patch_softmin_veto_probe_health as v62,
)
from tools import (
    run_stageb_confidence_candidate_complete_trace_c1_probe_u0400 as training,
)
from util.stage_b_dense_duty_audit import build_training_contract


SCHEMA = "pivot.stageb.confidence_candidate_complete_trace_c1_probe_health/v1"
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v43"
EXPECTED_CAPACITY_CONTRACT = (
    "rank_cloned_full_decoder_candidate_complete_free_head_v3"
)
EXPECTED_VARIANT = "candidate_complete_trace_free_head_coverage_c1"
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_candidate_complete_trace_c1_"
    "probe_u0400_20260803.py"
)
EXPECTED_UPDATES = 400
EXPECTED_TERMINAL_EPOCH = 1
EXPECTED_TERMINAL_ITERATION = 712
LOG_PATH = Path(training.OUTPUT) / "log.txt"

EXPECTED_ACTIVE_TENSORS = 362
EXPECTED_ACTIVE_ELEMENTS = 25_530_881
EXPECTED_TOKEN_TENSORS = 356
EXPECTED_TOKEN_ELEMENTS = 25_464_320
EXPECTED_GLOBAL_TENSORS = 6
EXPECTED_GLOBAL_ELEMENTS = 66_561
EXPECTED_LIVE_TOKEN_TENSORS = EXPECTED_TOKEN_TENSORS
EXPECTED_LIVE_GLOBAL_TENSORS = EXPECTED_GLOBAL_TENSORS
EXPECTED_ADAPTER_TENSORS = 362
EXPECTED_POOL_TENSORS = 0
MIGRATION_SCHEMA = v62.MIGRATION_SCHEMA
FRESH_CONFIDENCE_CONTRACT = v62.FRESH_CONFIDENCE_CONTRACT
ProbeHealthEvidenceError = v62.ProbeHealthEvidenceError

_C1_REQUIRED_TRAJECTORY_FIELDS = (
    "train_fixed_text_token_trace_deployed_candidate_count_unscaled",
    "train_fixed_text_candidate_depth_tn_sample_count_unscaled",
    "train_fixed_text_candidate_depth_tn_query_count_unscaled",
    "train_fixed_text_token_trace_broadcast_candidate_count_unscaled",
    "train_fixed_text_token_trace_candidate_coverage_unscaled",
    "train_fixed_text_candidate_depth_tn_zero_rate_unscaled",
    "train_fixed_text_candidate_depth_tn_min_mean_unscaled",
    "train_fixed_text_candidate_depth_positive_sample_count_unscaled",
    "train_fixed_text_candidate_depth_positive_query_count_unscaled",
    "train_fixed_text_candidate_depth_positive_zero_rate_unscaled",
    "train_fixed_text_candidate_depth_positive_min_mean_unscaled",
)

_CORE = v62.v61._CORE
_MODULES = (v62, v62.v61, *v62._MODULES)
_OVERRIDES = {
    "SCHEMA": SCHEMA,
    "TRAINING_CONTRACT_SCHEMA": TRAINING_CONTRACT_SCHEMA,
    "EXPECTED_CAPACITY_CONTRACT": EXPECTED_CAPACITY_CONTRACT,
    "EXPECTED_VARIANT": EXPECTED_VARIANT,
    "EXPECTED_CONFIG_ENTRY": EXPECTED_CONFIG_ENTRY,
    "EXPECTED_UPDATES": EXPECTED_UPDATES,
    "EXPECTED_TERMINAL_EPOCH": EXPECTED_TERMINAL_EPOCH,
    "EXPECTED_TERMINAL_ITERATION": EXPECTED_TERMINAL_ITERATION,
    "LOG_PATH": LOG_PATH,
    "EXPECTED_ACTIVE_TENSORS": EXPECTED_ACTIVE_TENSORS,
    "EXPECTED_ACTIVE_ELEMENTS": EXPECTED_ACTIVE_ELEMENTS,
    "EXPECTED_TOKEN_TENSORS": EXPECTED_TOKEN_TENSORS,
    "EXPECTED_TOKEN_ELEMENTS": EXPECTED_TOKEN_ELEMENTS,
    "EXPECTED_GLOBAL_TENSORS": EXPECTED_GLOBAL_TENSORS,
    "EXPECTED_GLOBAL_ELEMENTS": EXPECTED_GLOBAL_ELEMENTS,
    "EXPECTED_LIVE_TOKEN_TENSORS": EXPECTED_LIVE_TOKEN_TENSORS,
    "EXPECTED_LIVE_GLOBAL_TENSORS": EXPECTED_LIVE_GLOBAL_TENSORS,
    "EXPECTED_ADAPTER_TENSORS": EXPECTED_ADAPTER_TENSORS,
    "EXPECTED_POOL_TENSORS": EXPECTED_POOL_TENSORS,
    "MIGRATION_SCHEMA": MIGRATION_SCHEMA,
    "FRESH_CONFIDENCE_CONTRACT": FRESH_CONFIDENCE_CONTRACT,
    "FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA": MIGRATION_SCHEMA,
    "FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT": (
        FRESH_CONFIDENCE_CONTRACT
    ),
    "training": training,
}
_C1_EXPECTED_CONTRACT_VALUES = {
    **_CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_v21_token_edit_query_scope": "candidate_complete_trace_v4",
    "stage_b_v11_trainable_params_min": EXPECTED_ACTIVE_ELEMENTS,
    "stage_b_v11_trainable_params_max": EXPECTED_ACTIVE_ELEMENTS,
}


def _health_checks_c1(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = dict(v62._health_checks_v62(runtime, trajectory))
    owner = checks.pop("u222_v62_owner_live_counts_exact")
    owner["requirement"] = "active=362, token-veto=356, free-depth-head=6"
    checks["u222_c1_owner_live_counts_exact"] = owner
    clip = checks.pop("u222_v62_two_independent_clips_exact")
    checks["u222_c1_two_independent_clips_exact"] = clip

    def value(name: str) -> float:
        raw = trajectory.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProbeHealthEvidenceError(
                f"candidate-complete C1 trajectory field is missing or not numeric: "
                f"{name}"
            )
        observed = float(raw)
        if not math.isfinite(observed):
            raise ProbeHealthEvidenceError(
                f"candidate-complete C1 trajectory field is non-finite: {name}"
            )
        return observed

    deployed_count = value(
        "train_fixed_text_token_trace_deployed_candidate_count_unscaled"
    )
    tn_sample_count = value(
        "train_fixed_text_candidate_depth_tn_sample_count_unscaled"
    )
    tn_query_count = value(
        "train_fixed_text_candidate_depth_tn_query_count_unscaled"
    )
    broadcast_count = value(
        "train_fixed_text_token_trace_broadcast_candidate_count_unscaled"
    )
    candidate_coverage = value(
        "train_fixed_text_token_trace_candidate_coverage_unscaled"
    )
    tn_zero_rate = value(
        "train_fixed_text_candidate_depth_tn_zero_rate_unscaled"
    )
    tn_min_depth = value(
        "train_fixed_text_candidate_depth_tn_min_mean_unscaled"
    )
    positive_sample_count = value(
        "train_fixed_text_candidate_depth_positive_sample_count_unscaled"
    )
    positive_query_count = value(
        "train_fixed_text_candidate_depth_positive_query_count_unscaled"
    )
    positive_zero_rate = value(
        "train_fixed_text_candidate_depth_positive_zero_rate_unscaled"
    )
    positive_min_depth = value(
        "train_fixed_text_candidate_depth_positive_min_mean_unscaled"
    )

    checks["u222_c1_candidate_depth_deployed_set_exact"] = _CORE._check(
        {
            "deployed_candidate_count": deployed_count,
            "tn_depth_sample_count": tn_sample_count,
            "tn_depth_query_count": tn_query_count,
        },
        "every non-empty deployed TN candidate receives depth supervision",
        deployed_count > 0.0
        and tn_sample_count > 0.0
        and math.isclose(
            deployed_count,
            tn_query_count,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
    )
    checks["u222_c1_no_token_broadcast_or_coverage"] = _CORE._check(
        {
            "deployed_candidate_count": deployed_count,
            "broadcast_candidate_count": broadcast_count,
            "candidate_token_coverage": candidate_coverage,
        },
        "expression-only C1 has exact zero token broadcast and coverage",
        math.isclose(broadcast_count, 0.0, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(candidate_coverage, 0.0, rel_tol=0.0, abs_tol=1e-12),
    )
    checks["u222_c1_tn_depth_released_from_zero"] = _CORE._check(
        {"tn_query_count": tn_query_count, "tn_zero_depth_rate": tn_zero_rate},
        "TN zero-depth rate is at most 1%",
        tn_query_count > 0.0 and 0.0 <= tn_zero_rate <= 0.01,
    )
    checks["u222_c1_tn_min_depth_finite"] = _CORE._check(
        tn_min_depth,
        "mean shallowest TN depth is finite and non-negative",
        tn_min_depth >= 0.0,
    )
    checks["u222_c1_positive_depth_protection"] = _CORE._check(
        {
            "positive_sample_count": positive_sample_count,
            "positive_query_count": positive_query_count,
            "positive_zero_depth_rate": positive_zero_rate,
            "positive_min_depth": positive_min_depth,
        },
        "localized positive minimum depth remains in [0, 0.10]",
        positive_sample_count > 0.0
        and positive_query_count > 0.0
        and 0.0 <= positive_min_depth <= 0.10,
    )
    return checks


def _audit_c1_training_contract(args: Mapping[str, Any]) -> dict[str, Any]:
    direct_expected = {
        "stage_b_dense_duty_confidence_full_decoder_verifier": True,
        "stage_b_dense_duty_confidence_veto_only_patch_softmin": True,
        "stage_b_dense_duty_confidence_candidate_trace_contract": (
            "candidate_complete_free_head_coverage_v1"
        ),
        "stage_b_dense_duty_confidence_capacity_contract": (
            EXPECTED_CAPACITY_CONTRACT
        ),
        "stage_b_dense_duty_confidence_variant": EXPECTED_VARIANT,
        "stage_b_v21_token_objective": "edit_bce_group_balanced",
        "stage_b_v21_token_edit_query_scope": "candidate_complete_trace_v4",
        "stage_b_dense_duty_candidate_depth_all_weight": 1.0,
        "stage_b_dense_duty_candidate_depth_escape_weight": 1.0,
        "stage_b_dense_duty_candidate_depth_positive_weight": 1.0,
        "stage_b_dense_duty_candidate_depth_tn_margin": 0.5,
        "stage_b_dense_duty_candidate_depth_escape_margin": 0.5,
        "stage_b_dense_duty_candidate_depth_positive_max": 0.05,
        "stage_b_dense_duty_candidate_depth_temperature": 0.1,
        "stage_b_dense_duty_deployed_global_absolute_weight": 0.0,
        "stage_b_dense_duty_forward_pack_factor": 1,
        "stage_b_dense_duty_logical_loss_batch_size": 16,
        "stage_b_dense_duty_expected_forward_batch_size": 16,
        "stage_b_dense_duty_expected_logical_batches_per_epoch": 887,
        "stage_b_dense_duty_expected_physical_forwards_per_epoch": 887,
    }
    drift = {
        key: (args.get(key), value)
        for key, value in direct_expected.items()
        if args.get(key) != value
    }
    if drift:
        raise ProbeHealthEvidenceError(
            f"candidate-complete C1 training contract drifted: {drift}"
        )

    saved = args.get("stage_b_dense_duty_training_contract")
    if not isinstance(saved, Mapping):
        raise ProbeHealthEvidenceError(
            "candidate-complete C1 checkpoint lacks its saved v43 contract"
        )
    try:
        rebuilt = build_training_contract(args)
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(
            f"candidate-complete C1 training contract/source closure is invalid: "
            f"{error}"
        ) from error
    if dict(saved) != rebuilt:
        raise ProbeHealthEvidenceError(
            "saved candidate-complete C1 training contract does not replay"
        )
    if rebuilt.get("schema") != TRAINING_CONTRACT_SCHEMA:
        raise ProbeHealthEvidenceError(
            "candidate-complete C1 training contract is not exact v43"
        )
    values = rebuilt.get("values")
    if not isinstance(values, Mapping):
        raise ProbeHealthEvidenceError(
            "candidate-complete C1 v43 contract has no values mapping"
        )
    for key, expected in _C1_EXPECTED_CONTRACT_VALUES.items():
        if values.get(key) != expected or args.get(key) != expected:
            raise ProbeHealthEvidenceError(
                f"candidate-complete C1 contract requires {key}={expected!r}"
            )
    if args.get("stage_b_dense_duty_evaluation_scope") != "probe":
        raise ProbeHealthEvidenceError(
            "candidate-complete C1 evaluation scope is not probe"
        )
    if args.get("stage_b_dense_duty_confidence_probe_admission_contract") != (
        "disabled_for_probe_v1"
    ) or str(args.get("stage_b_dense_duty_confidence_probe_admission_report", "")):
        raise ProbeHealthEvidenceError(
            "candidate-complete C1 probe checkpoint must not carry formal admission"
        )

    closure = values.get("stage_b_dense_duty_source_closure")
    if not isinstance(closure, Mapping):
        raise ProbeHealthEvidenceError(
            "candidate-complete C1 v43 source closure is missing"
        )
    config = closure.get("config")
    if not isinstance(config, Mapping) or config.get("entry") != EXPECTED_CONFIG_ENTRY:
        raise ProbeHealthEvidenceError(
            "candidate-complete C1 closure does not bind the exact probe config"
        )
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
        "candidate_trace_contract": values[
            "stage_b_dense_duty_confidence_candidate_trace_contract"
        ],
        "token_edit_query_scope": values[
            "stage_b_v21_token_edit_query_scope"
        ],
    }


def _activate_c1() -> None:
    for _module in _MODULES:
        for _name, _value in _OVERRIDES.items():
            setattr(_module, _name, _value)
    _CORE._CONFIDENCE_ADAPTER_PREFIX = (
        v62._VERIFIER_TOWER_PREFIX,
        v62._VERIFIER_HEAD_PREFIX,
    )
    _CORE._CONFIDENCE_POOL_PREFIX = ()
    _CORE._default_output = lambda: Path(training.OUTPUT).parent / (
        "u000400_candidate_complete_trace_c1_health_audit.json"
    )
    _CORE.EXPECTED_CONTRACT_VALUES = _C1_EXPECTED_CONTRACT_VALUES
    v62.v61._V53._owner_for_name = v62._owner_for_name
    v62.v61._V53._audit_v53_migration = v62._audit_v62_migration
    v62.v61._V55._audit_v55_migration = v62._audit_v62_migration
    _CORE._audit_split_ownership = v62.v61._V53._audit_split_ownership
    _CORE._audit_runtime = v62.v61._V53._audit_runtime
    _CORE._health_checks = _health_checks_c1
    _CORE._audit_training_contract = _audit_c1_training_contract
    _CORE.audit = audit


def _restore_shared_state() -> None:
    for _module, _values in _BASELINE_MODULE_STATE.values():
        for _name, _value in _values.items():
            if _value is _MISSING:
                _module.__dict__.pop(_name, None)
            else:
                setattr(_module, _name, _value)
    for _name, _value in _BASELINE_CORE_STATE.items():
        setattr(_CORE, _name, _value)
    for (_module, _name), _value in _BASELINE_HOOK_STATE.items():
        setattr(_module, _name, _value)


_C1_SCOPE_DEPTH = 0


@contextmanager
def _c1_scope():
    global _C1_SCOPE_DEPTH
    outermost = _C1_SCOPE_DEPTH == 0
    if outermost:
        _activate_c1()
    _C1_SCOPE_DEPTH += 1
    try:
        yield
    finally:
        _C1_SCOPE_DEPTH -= 1
        if outermost:
            _restore_shared_state()


def audit() -> dict[str, Any]:
    with _c1_scope():
        result = v62.v61._V53._BASE_AUDIT()
        result["candidate_trace_provenance_claim"] = (
            "expression_depth_complete_token_broadcast_provenance_gated"
        )
        return result


def run(argv: Sequence[str] | None = None) -> int:
    with _c1_scope():
        return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()


_restore_shared_state()
