#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for candidate-complete monotone C2 U400."""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (
    audit_stageb_confidence_full_decoder_verifier_probe_health as _v61_baseline,
)


_SHARED_NAMES = (
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
_SHARED_STATE = {
    id(module): (
        module,
        {name: getattr(module, name, _MISSING) for name in _SHARED_NAMES},
    )
    for module in _BASELINE_MODULES
}
_CORE_STATE = {
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
        "_JOINT_CLIP_FIELDS",
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
    run_stageb_confidence_candidate_complete_trace_c2_probe_u0400 as training,
)
from util.stage_b_confidence_adapter_migration import (
    FULL_DECODER_CANDIDATE_COMPLETE_MONOTONE_FRESH_CONFIDENCE_CONTRACT,
    FULL_DECODER_CANDIDATE_COMPLETE_MONOTONE_MIGRATION_SCHEMA,
    validate_confidence_adapter_migration_audit,
)
from util.stage_b_dense_duty_audit import (
    _validate_fulltext_global_absolute_runtime_audit,
    build_training_contract,
    fingerprint_named_tensors,
    validate_initial_fingerprint,
)


SCHEMA = "pivot.stageb.confidence_candidate_complete_trace_c2_probe_health/v1"
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v43"
EXPECTED_CAPACITY_CONTRACT = "rank_cloned_full_decoder_candidate_complete_monotone_v4"
EXPECTED_VARIANT = "candidate_complete_trace_monotone_token_entailment_c2"
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_candidate_complete_trace_c2_"
    "probe_u0400_20260803.py"
)
EXPECTED_UPDATES = 400
EXPECTED_TERMINAL_EPOCH = 1
EXPECTED_TERMINAL_ITERATION = 712
LOG_PATH = Path(training.OUTPUT) / "log.txt"

EXPECTED_ACTIVE_TENSORS = 356
EXPECTED_ACTIVE_ELEMENTS = 25_464_320
EXPECTED_TOKEN_TENSORS = 356
EXPECTED_TOKEN_ELEMENTS = 25_464_320
EXPECTED_GLOBAL_TENSORS = 0
EXPECTED_GLOBAL_ELEMENTS = 0
EXPECTED_LIVE_TOKEN_TENSORS = EXPECTED_TOKEN_TENSORS
EXPECTED_ADAPTER_TENSORS = EXPECTED_TOKEN_TENSORS
EXPECTED_POOL_TENSORS = 0
ONE_OWNER_CLIP_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_one_owner_clip_contract/v1"
MIGRATION_SCHEMA = FULL_DECODER_CANDIDATE_COMPLETE_MONOTONE_MIGRATION_SCHEMA
FRESH_CONFIDENCE_CONTRACT = (
    FULL_DECODER_CANDIDATE_COMPLETE_MONOTONE_FRESH_CONFIDENCE_CONTRACT
)
ProbeHealthEvidenceError = v62.ProbeHealthEvidenceError

_VERIFIER_TOWER_PREFIX = "stage_b_fixed_text_scorer.confidence_verifier_tower."
_C2_REQUIRED_TRAJECTORY_FIELDS = (
    "train_fixed_text_token_trace_deployed_candidate_count_unscaled",
    "train_fixed_text_candidate_depth_tn_sample_count_unscaled",
    "train_fixed_text_candidate_depth_tn_query_count_unscaled",
    "train_fixed_text_token_trace_broadcast_candidate_count_unscaled",
    "train_fixed_text_token_trace_candidate_coverage_unscaled",
    "train_fixed_text_candidate_depth_tn_zero_rate_unscaled",
    "train_fixed_text_candidate_depth_tn_min_mean_unscaled",
    "train_fixed_text_candidate_depth_positive_sample_count_unscaled",
    "train_fixed_text_candidate_depth_positive_query_count_unscaled",
    "train_fixed_text_candidate_depth_positive_min_mean_unscaled",
)
_C2_JOINT_CLIP_FIELDS = (
    "train_grad_norm_dense_duty_active_preclip",
    "train_grad_tensor_count_dense_duty_active",
    "train_grad_norm_dense_duty_token_veto_preclip",
    "train_grad_tensor_count_dense_duty_token_veto",
    "train_grad_norm_dense_duty_token_veto_postclip",
    "train_grad_norm_dense_duty_active_postclip",
    "train_amp_step_skipped",
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
    "EXPECTED_LIVE_GLOBAL_TENSORS": 0,
    "EXPECTED_ADAPTER_TENSORS": EXPECTED_ADAPTER_TENSORS,
    "EXPECTED_POOL_TENSORS": EXPECTED_POOL_TENSORS,
    "MIGRATION_SCHEMA": MIGRATION_SCHEMA,
    "FRESH_CONFIDENCE_CONTRACT": FRESH_CONFIDENCE_CONTRACT,
    "training": training,
}
_C2_EXPECTED_CONTRACT_VALUES = {
    **_CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_v21_token_edit_query_scope": "candidate_complete_trace_v4",
    "stage_b_dense_duty_raw_veto_gate_weight": 0.0,
    "stage_b_dense_duty_raw_veto_carrier_pair_weight": 0.0,
    "stage_b_v11_trainable_params_min": EXPECTED_ACTIVE_ELEMENTS,
    "stage_b_v11_trainable_params_max": EXPECTED_ACTIVE_ELEMENTS,
}


def _audit_c2_migration(args: Mapping[str, Any]) -> dict[str, Any]:
    try:
        audit = validate_confidence_adapter_migration_audit(
            args.get("stage_b_dense_duty_confidence_adapter_migration_audit"),
            source_checkpoint_sha256=str(
                args.get("stage_b_dense_duty_rank_source_checkpoint_sha256", "")
            ),
            source_optimizer_updates=6551,
            source_checkpoint_reason=str(
                args.get("stage_b_dense_duty_rank_source_checkpoint_reason", "")
            ),
            rank_sha256=str(args.get("stage_b_dense_duty_rank_source_rank_sha256", "")),
            transferred_sha256=str(
                args.get("stage_b_dense_duty_rank_source_transferred_sha256", "")
            ),
        )
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(str(error)) from error
    expected = {
        "schema": MIGRATION_SCHEMA,
        "fresh_confidence_contract": FRESH_CONFIDENCE_CONTRACT,
        "candidate_trace_contract": "candidate_complete_monotone_token_entailment_v2",
        "active_confidence_parameter_tensor_count": EXPECTED_ACTIVE_TENSORS,
        "active_confidence_parameter_element_count": EXPECTED_ACTIVE_ELEMENTS,
        "active_verifier_parameter_tensor_count": EXPECTED_TOKEN_TENSORS,
        "active_verifier_parameter_element_count": EXPECTED_TOKEN_ELEMENTS,
        "active_verifier_veto_head_parameter_tensor_count": 0,
        "active_pool_parameter_tensor_count": 0,
    }
    drift = {key: (audit.get(key), value) for key, value in expected.items() if audit.get(key) != value}
    if drift:
        raise ProbeHealthEvidenceError(f"candidate-complete C2 migration drifted: {drift}")
    return dict(audit)


def _audit_c2_split_ownership(
    payload: Mapping[str, Any], args: Mapping[str, Any]
) -> dict[str, Any]:
    model, optimizer = payload.get("model"), payload.get("optimizer")
    if not isinstance(model, Mapping) or not isinstance(optimizer, Mapping):
        raise ProbeHealthEvidenceError("C2 checkpoint model/optimizer state is missing")
    migration = _audit_c2_migration(args)
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
        or any(not str(name).startswith(_VERIFIER_TOWER_PREFIX) for name in names)
        or any(name not in model for name in names)
    ):
        raise ProbeHealthEvidenceError("C2 active state is not the token-only verifier tower")
    if any("confidence_verifier_veto_head." in str(name) or "confidence_pool." in str(name) for name in names):
        raise ProbeHealthEvidenceError("C2 active state includes a global owner")
    if any(not torch.is_tensor(model[name]) or not bool(torch.isfinite(model[name]).all().item()) for name in names):
        raise ProbeHealthEvidenceError("C2 active parameter is non-finite")
    current = fingerprint_named_tensors(model, names)
    if current["sha256"] == initial["active"]["sha256"]:
        raise ProbeHealthEvidenceError("C2 token-only active state is unchanged")
    groups, state = optimizer.get("param_groups"), optimizer.get("state")
    if not isinstance(groups, list) or not isinstance(state, Mapping):
        raise ProbeHealthEvidenceError("C2 optimizer state is malformed")
    ids = [
        _CORE._exact_int(identifier, label="optimizer parameter id")
        for group in groups
        if isinstance(group, Mapping) and isinstance(group.get("params"), list)
        for identifier in group["params"]
    ]
    exact_ids = set(range(EXPECTED_ACTIVE_TENSORS))
    if len(ids) != EXPECTED_ACTIVE_TENSORS or set(ids) != exact_ids:
        raise ProbeHealthEvidenceError("C2 optimizer does not own the exact token-only set")
    state_ids = {_CORE._exact_int(identifier, label="optimizer state id") for identifier in state}
    if state_ids != exact_ids:
        raise ProbeHealthEvidenceError("C2 token-only optimizer state is incomplete")
    rank_names = sorted(
        str(name) for name in model if str(name).startswith(_CORE._RANK_PREFIX)
    )
    if not rank_names or set(rank_names) & set(names):
        raise ProbeHealthEvidenceError("C2 rank tower is not frozen and disjoint")
    rank = fingerprint_named_tensors(model, rank_names)
    if (
        rank.get("sha256") != _CORE.EXPECTED_RANK_SHA256
        or rank.get("nonfinite_count") != 0
        or args.get("stage_b_dense_duty_rank_source_rank_sha256")
        != _CORE.EXPECTED_RANK_SHA256
    ):
        raise ProbeHealthEvidenceError("C2 frozen rank tower identity drifted")
    return {
        "active": current,
        "initial_active_sha256": initial["active"]["sha256"],
        "migration": migration,
        "token_veto": {"tensor_count": len(names), "element_count": EXPECTED_TOKEN_ELEMENTS, "live_tensor_count": len(names)},
        "global_absolute": None,
        "adapter_tensor_count": len(names),
        "pool_tensor_count": 0,
        "disjoint_and_complete": True,
        "rank": rank,
    }


def _audit_c2_runtime(value: Any) -> dict[str, Any]:
    try:
        return _validate_fulltext_global_absolute_runtime_audit(
            value,
            expected_steps=EXPECTED_UPDATES,
            expected_owner_tensor_counts={"token_veto": EXPECTED_TOKEN_TENSORS},
            contract_label="C2",
        )
    except RuntimeError as error:
        raise ProbeHealthEvidenceError(str(error)) from error


def _health_checks_c2(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = {
        "runtime_all_u400_boundaries_succeeded": _CORE._check(
            {
                "boundaries": runtime.get("optimizer_step_boundaries"),
                "successful": runtime.get("successful_optimizer_steps"),
            },
            "boundaries == successful == 400",
            runtime.get("optimizer_step_boundaries") == EXPECTED_UPDATES
            and runtime.get("successful_optimizer_steps") == EXPECTED_UPDATES,
        ),
        "runtime_amp_skips_zero": _CORE._check(
            runtime.get("amp_skipped_optimizer_steps"),
            "== 0",
            runtime.get("amp_skipped_optimizer_steps") == 0,
        ),
        "runtime_nonfinite_gradients_zero": _CORE._check(
            {
                "active": runtime.get("nonfinite_gradient_boundaries"),
                "token_veto": runtime.get(
                    "nonfinite_token_veto_gradient_boundaries"
                ),
            },
            "active and token-veto == 0",
            runtime.get("nonfinite_gradient_boundaries") == 0
            and runtime.get("nonfinite_token_veto_gradient_boundaries", 0) == 0,
        ),
        "runtime_no_zero_gradient_owner_steps": _CORE._check(
            {
                "active": runtime.get("zero_gradient_successful_steps"),
                "token_veto": runtime.get(
                    "zero_token_veto_gradient_successful_steps"
                ),
            },
            "active and token-veto == 0",
            runtime.get("zero_gradient_successful_steps") == 0
            and runtime.get("zero_token_veto_gradient_successful_steps", 0) == 0,
        ),
        "runtime_token_owner_engaged": _CORE._check(
            {
                "last": runtime.get("last_token_veto_grad_norm_preclip"),
                "maximum": runtime.get("max_token_veto_grad_norm_preclip"),
            },
            "last and maximum token-veto gradient norms > 0",
            float(runtime.get("last_token_veto_grad_norm_preclip", 0.0)) > 0.0
            and float(runtime.get("max_token_veto_grad_norm_preclip", 0.0)) > 0.0,
        ),
        "runtime_amp_scale_positive": _CORE._check(
            {
                "minimum": runtime.get("min_amp_scale"),
                "last": runtime.get("last_amp_scale"),
            },
            "minimum and last >= 1",
            isinstance(runtime.get("min_amp_scale"), (int, float))
            and not isinstance(runtime.get("min_amp_scale"), bool)
            and isinstance(runtime.get("last_amp_scale"), (int, float))
            and not isinstance(runtime.get("last_amp_scale"), bool)
            and math.isfinite(float(runtime["min_amp_scale"]))
            and math.isfinite(float(runtime["last_amp_scale"]))
            and float(runtime["min_amp_scale"]) >= 1.0
            and float(runtime["last_amp_scale"]) >= 1.0,
        ),
    }

    def value(name: str) -> float:
        raw = trajectory.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProbeHealthEvidenceError(
                f"candidate-complete C2 trajectory field is missing or not numeric: "
                f"{name}"
            )
        observed = float(raw)
        if not math.isfinite(observed):
            raise ProbeHealthEvidenceError(
                f"candidate-complete C2 trajectory field is non-finite: {name}"
            )
        return observed

    checks["u222_log_amp_skips_zero"] = _CORE._check(
        value("train_amp_step_skipped"),
        "== 0",
        value("train_amp_step_skipped") == 0.0,
    )
    checks["u222_c2_token_only_owner_exact"] = _CORE._check(
        {"active": value("train_grad_tensor_count_dense_duty_active"), "token_veto": value("train_grad_tensor_count_dense_duty_token_veto")},
        "active=356, token-veto=356, global owner absent",
        math.isclose(value("train_grad_tensor_count_dense_duty_active"), EXPECTED_ACTIVE_TENSORS, abs_tol=1e-6)
        and math.isclose(value("train_grad_tensor_count_dense_duty_token_veto"), EXPECTED_TOKEN_TENSORS, abs_tol=1e-6),
    )
    checks["u222_c2_one_owner_clip_exact"] = _CORE._check(
        {"schema": runtime.get("clip_contract_schema"), "preclip": value("train_grad_norm_dense_duty_token_veto_preclip"), "postclip": value("train_grad_norm_dense_duty_token_veto_postclip")},
        "one positive token owner obeys the 0.1 clip",
        runtime.get("clip_contract_schema") == ONE_OWNER_CLIP_CONTRACT_SCHEMA
        and value("train_amp_step_skipped") == 0.0
        and value("train_grad_norm_dense_duty_token_veto_preclip") > 0.0
        and 0.0 < value("train_grad_norm_dense_duty_token_veto_postclip") <= min(value("train_grad_norm_dense_duty_token_veto_preclip"), _CORE.CLIP_MAX_NORM) + 1e-6,
    )
    fields = {name: value(name) for name in _C2_REQUIRED_TRAJECTORY_FIELDS}
    deployed = fields["train_fixed_text_token_trace_deployed_candidate_count_unscaled"]
    tn_samples = fields["train_fixed_text_candidate_depth_tn_sample_count_unscaled"]
    tn_queries = fields["train_fixed_text_candidate_depth_tn_query_count_unscaled"]
    broadcast = fields["train_fixed_text_token_trace_broadcast_candidate_count_unscaled"]
    coverage = fields["train_fixed_text_token_trace_candidate_coverage_unscaled"]
    checks["u222_c2_candidate_depth_deployed_set_exact"] = _CORE._check(fields, "every deployed TN candidate receives monotone token depth supervision", deployed > 0.0 and tn_samples > 0.0 and math.isclose(deployed, tn_queries, abs_tol=1e-6))
    checks["u222_c2_no_token_broadcast_or_coverage"] = _CORE._check(
        {"broadcast": broadcast, "coverage": coverage},
        "expression-only C2 has exact zero token broadcast and coverage",
        math.isclose(broadcast, 0.0, abs_tol=1e-6)
        and math.isclose(coverage, 0.0, abs_tol=1e-12),
    )
    checks["u222_c2_tn_depth_released_from_zero"] = _CORE._check({"queries": tn_queries, "zero_rate": fields["train_fixed_text_candidate_depth_tn_zero_rate_unscaled"], "min_mean": fields["train_fixed_text_candidate_depth_tn_min_mean_unscaled"]}, "TN depth is finite, nonnegative, and released from zero", tn_queries > 0.0 and 0.0 <= fields["train_fixed_text_candidate_depth_tn_zero_rate_unscaled"] <= 0.01 and fields["train_fixed_text_candidate_depth_tn_min_mean_unscaled"] >= 0.0)
    checks["u222_c2_positive_depth_protection"] = _CORE._check({"samples": fields["train_fixed_text_candidate_depth_positive_sample_count_unscaled"], "queries": fields["train_fixed_text_candidate_depth_positive_query_count_unscaled"], "min_mean": fields["train_fixed_text_candidate_depth_positive_min_mean_unscaled"]}, "localized positive minimum depth remains in [0, 0.10]", fields["train_fixed_text_candidate_depth_positive_sample_count_unscaled"] > 0.0 and fields["train_fixed_text_candidate_depth_positive_query_count_unscaled"] > 0.0 and 0.0 <= fields["train_fixed_text_candidate_depth_positive_min_mean_unscaled"] <= 0.10)
    return checks


def _audit_c2_training_contract(args: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "stage_b_dense_duty_confidence_full_decoder_verifier": True,
        "stage_b_dense_duty_confidence_veto_only_patch_softmin": True,
        "stage_b_dense_duty_confidence_candidate_trace_contract": "candidate_complete_monotone_token_entailment_v2",
        "stage_b_dense_duty_confidence_capacity_contract": EXPECTED_CAPACITY_CONTRACT,
        "stage_b_dense_duty_confidence_variant": EXPECTED_VARIANT,
        "stage_b_v21_token_edit_query_scope": "candidate_complete_trace_v4",
        "stage_b_dense_duty_raw_veto_gate_weight": 0.0,
        "stage_b_dense_duty_raw_veto_carrier_pair_weight": 0.0,
        "stage_b_dense_duty_deployed_global_absolute_weight": 0.0,
        "stage_b_dense_duty_forward_pack_factor": 1,
        "stage_b_dense_duty_logical_loss_batch_size": 16,
        "stage_b_dense_duty_expected_forward_batch_size": 16,
        "stage_b_dense_duty_expected_logical_batches_per_epoch": 887,
        "stage_b_dense_duty_expected_physical_forwards_per_epoch": 887,
    }
    drift = {key: (args.get(key), value) for key, value in expected.items() if args.get(key) != value}
    if drift:
        raise ProbeHealthEvidenceError(f"candidate-complete C2 training contract drifted: {drift}")
    saved = args.get("stage_b_dense_duty_training_contract")
    rebuilt = build_training_contract(args)
    if not isinstance(saved, Mapping) or dict(saved) != rebuilt or rebuilt.get("schema") != TRAINING_CONTRACT_SCHEMA:
        raise ProbeHealthEvidenceError("C2 saved v43 training contract does not replay")
    return {"schema": rebuilt["schema"], "sha256": rebuilt["sha256"], "candidate_trace_contract": expected["stage_b_dense_duty_confidence_candidate_trace_contract"], "token_edit_query_scope": expected["stage_b_v21_token_edit_query_scope"]}


def _activate_c2() -> None:
    for _module in _MODULES:
        for _name, _value in _OVERRIDES.items():
            setattr(_module, _name, _value)
    _CORE._CONFIDENCE_ADAPTER_PREFIX = (_VERIFIER_TOWER_PREFIX,)
    _CORE._CONFIDENCE_POOL_PREFIX = ()
    _CORE._default_output = lambda: Path(training.OUTPUT).parent / (
        "u000400_candidate_complete_trace_c2_health_audit.json"
    )
    _CORE.EXPECTED_CONTRACT_VALUES = _C2_EXPECTED_CONTRACT_VALUES
    _CORE._audit_split_ownership = _audit_c2_split_ownership
    _CORE._audit_runtime = _audit_c2_runtime
    _CORE._health_checks = _health_checks_c2
    _CORE._audit_training_contract = _audit_c2_training_contract
    _CORE._JOINT_CLIP_FIELDS = _C2_JOINT_CLIP_FIELDS
    _CORE.audit = audit


def _restore_shared_state() -> None:
    for _module, _values in _SHARED_STATE.values():
        for _name, _value in _values.items():
            if _value is _MISSING:
                _module.__dict__.pop(_name, None)
            else:
                setattr(_module, _name, _value)
    for _name, _value in _CORE_STATE.items():
        setattr(_CORE, _name, _value)
    for (_module, _name), _value in _BASELINE_HOOK_STATE.items():
        setattr(_module, _name, _value)


_C2_SCOPE_DEPTH = 0


@contextmanager
def _c2_scope():
    global _C2_SCOPE_DEPTH
    outermost = _C2_SCOPE_DEPTH == 0
    if outermost:
        _activate_c2()
    _C2_SCOPE_DEPTH += 1
    try:
        yield
    finally:
        _C2_SCOPE_DEPTH -= 1
        if outermost:
            _restore_shared_state()


def audit() -> dict[str, Any]:
    with _c2_scope():
        result = v62.v61._V53._BASE_AUDIT()
        result["candidate_trace_provenance_claim"] = (
            "monotone_token_entailment_all_candidate_depth_without_token_broadcast"
        )
        return result


def run(argv: Sequence[str] | None = None) -> int:
    with _c2_scope():
        return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()


_restore_shared_state()
