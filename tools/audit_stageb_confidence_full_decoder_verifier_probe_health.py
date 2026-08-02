#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V61 full-decoder verifier U400 probe."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    run_stageb_confidence_full_decoder_verifier_probe_u0400 as training,
)
from util.stage_b_confidence_adapter_migration import (  # noqa: E402
    FULL_DECODER_VERIFIER_FRESH_CONFIDENCE_CONTRACT,
    FULL_DECODER_VERIFIER_MIGRATION_SCHEMA,
    FULL_DECODER_VERIFIER_TOKEN_LOGIT_CONTRACT,
    validate_confidence_adapter_migration_audit,
)


_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_deployment_owned_query_veto_"
    "probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_full_decoder_verifier_probe_health_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load V60 probe health audit: {_BASE_PATH}")
_V60 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V60)


_V59 = _V60._V59
_V56 = _V59._V56
_V55 = _V59._V55
_V53 = _V59._V53
_BASE = _V59._BASE
_CORE = _V59._CORE

SCHEMA = "pivot.stageb.confidence_full_decoder_verifier_probe_health/v1"
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v42"
EXPECTED_REVISION = (
    "word_veto_rank_full_expression_deployment_owned_query_veto_v60"
)
EXPECTED_HEAD_CONTRACT = (
    "split_token_veto_deployment_owned_query_veto_global_absolute_v11"
)
EXPECTED_POOL_CONTRACT = (
    "detached_rank_full_expression_token_conditioned_query_veto_"
    "deployment_owned_global_pool_v15"
)
EXPECTED_GATE_CONTRACT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
EXPECTED_POSITIVE_TRUST_CONTRACT = "absolute_global_confidence_logit_v2"
EXPECTED_CAPACITY_CONTRACT = "rank_cloned_full_decoder_6layer_256d_v1"
EXPECTED_VARIANT = (
    "full_decoder_token_entailment_nonnegative_veto_capacity_upper_bound_v61"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_full_decoder_verifier_"
    "probe_u0400_20260803.py"
)
EXPECTED_SOURCE_UPDATES = 6551
EXPECTED_UPDATES = 400
# V45 used pack2/accum2 and ended at epoch1/iteration356. V61 preserves the
# same effective optimizer batch with pack1/accum4, so its exact terminal
# physical-forward cursor is epoch1/iteration712.
EXPECTED_TERMINAL_EPOCH = 1
EXPECTED_TERMINAL_ITERATION = 712
LOG_PATH = Path(training.OUTPUT) / "log.txt"

EXPECTED_ACTIVE_TENSORS = 368
EXPECTED_ACTIVE_ELEMENTS = 25_664_258
EXPECTED_TOKEN_TENSORS = 356
EXPECTED_TOKEN_ELEMENTS = 25_464_320
EXPECTED_GLOBAL_TENSORS = 12
EXPECTED_GLOBAL_ELEMENTS = 199_938
EXPECTED_LIVE_TOKEN_TENSORS = EXPECTED_TOKEN_TENSORS
EXPECTED_LIVE_GLOBAL_TENSORS = EXPECTED_GLOBAL_TENSORS
EXPECTED_ADAPTER_TENSORS = 362
EXPECTED_POOL_TENSORS = 6
EXPECTED_VERIFIER_STATE_TENSORS = 453
EXPECTED_VERIFIER_STATE_ELEMENTS = 33_158_400
EXPECTED_FRESH_TENSORS = 72
EXPECTED_FRESH_ELEMENTS = 601_287
EXPECTED_FRESH_SHA256 = (
    "6c33647364fcc5f1e3f91fbe7e932e549de2e06015803dfa3dfc30758ffc63a0"
)
MIGRATION_SCHEMA = FULL_DECODER_VERIFIER_MIGRATION_SCHEMA
FRESH_CONFIDENCE_CONTRACT = FULL_DECODER_VERIFIER_FRESH_CONFIDENCE_CONTRACT
TOKEN_LOGIT_CONTRACT = FULL_DECODER_VERIFIER_TOKEN_LOGIT_CONTRACT
ProbeHealthEvidenceError = _V60.ProbeHealthEvidenceError

_VERIFIER_TOWER_PREFIX = "stage_b_fixed_text_scorer.confidence_verifier_tower."
_VERIFIER_HEAD_PREFIX = "stage_b_fixed_text_scorer.confidence_verifier_veto_head."
_POOL_PREFIX = "stage_b_fixed_text_scorer.confidence_pool."


_OVERRIDES = {
    "SCHEMA": SCHEMA,
    "TRAINING_CONTRACT_SCHEMA": TRAINING_CONTRACT_SCHEMA,
    "EXPECTED_REVISION": EXPECTED_REVISION,
    "EXPECTED_HEAD_CONTRACT": EXPECTED_HEAD_CONTRACT,
    "EXPECTED_POOL_CONTRACT": EXPECTED_POOL_CONTRACT,
    "EXPECTED_GATE_CONTRACT": EXPECTED_GATE_CONTRACT,
    "EXPECTED_POSITIVE_TRUST_CONTRACT": EXPECTED_POSITIVE_TRUST_CONTRACT,
    "EXPECTED_CONFIG_ENTRY": EXPECTED_CONFIG_ENTRY,
    "EXPECTED_SOURCE_UPDATES": EXPECTED_SOURCE_UPDATES,
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
for _module in (_V60, _V59, _V56, _V55, _V53, _BASE, _CORE):
    for _name, _value in _OVERRIDES.items():
        setattr(_module, _name, _value)

# Reuse the mature two-owner audit machinery while making its active adapter
# namespace the independent verifier tower plus its one-sided veto head.
_CORE._CONFIDENCE_ADAPTER_PREFIX = (
    _VERIFIER_TOWER_PREFIX,
    _VERIFIER_HEAD_PREFIX,
)
_CORE._CONFIDENCE_POOL_PREFIX = _POOL_PREFIX


def _owner_for_name(name: str) -> str:
    if name.startswith(_VERIFIER_TOWER_PREFIX):
        return "token_veto"
    if name.startswith((_VERIFIER_HEAD_PREFIX, _POOL_PREFIX)):
        return "global_absolute"
    raise ProbeHealthEvidenceError(f"unknown V61 active parameter owner: {name}")


_V53._owner_for_name = _owner_for_name

_CORE.EXPECTED_CONTRACT_VALUES = {
    **{
        key: value
        for key, value in _CORE.EXPECTED_CONTRACT_VALUES.items()
        if key
        not in {
            "stage_b_dense_duty_deployed_global_absolute_weight",
            # Schema v42 intentionally does not serialize these runtime-shape
            # fields. They are checked directly against checkpoint args below.
            "stage_b_dense_duty_forward_pack_factor",
            "stage_b_dense_duty_logical_loss_batch_size",
            "stage_b_dense_duty_expected_forward_batch_size",
            "stage_b_dense_duty_expected_logical_batches_per_epoch",
            "stage_b_dense_duty_expected_physical_forwards_per_epoch",
        }
    },
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_confidence_head_gradient_contract": EXPECTED_HEAD_CONTRACT,
    "stage_b_dense_duty_confidence_pool_feature_contract": EXPECTED_POOL_CONTRACT,
    "stage_b_dense_duty_confidence_gate_gradient_contract": EXPECTED_GATE_CONTRACT,
    "stage_b_dense_duty_positive_trust_contract": EXPECTED_POSITIVE_TRUST_CONTRACT,
    "stage_b_v15_tail_queue_negative_reduction_contract": "all_mean_v1",
    "stage_b_v14_local_absolute_weight": 0.0,
    "stage_b_v11_trainable_params_min": EXPECTED_ACTIVE_ELEMENTS,
    "stage_b_v11_trainable_params_max": EXPECTED_ACTIVE_ELEMENTS,
    "gradient_accumulation_steps": 4,
}
_CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_full_decoder_verifier_health_audit.json"
)


def _audit_v61_migration(args: Mapping[str, Any]) -> dict[str, Any]:
    if args.get("stage_b_dense_duty_rank_source_optimizer_updates") != (
        EXPECTED_SOURCE_UPDATES
    ):
        raise ProbeHealthEvidenceError("V61 migration is not fresh from U6551")
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
    expected = {
        "schema": MIGRATION_SCHEMA,
        "fresh_confidence_contract": FRESH_CONFIDENCE_CONTRACT,
        "token_logit_contract": TOKEN_LOGIT_CONTRACT,
        "active_confidence_parameter_tensor_count": EXPECTED_ACTIVE_TENSORS,
        "active_confidence_parameter_element_count": EXPECTED_ACTIVE_ELEMENTS,
        "verifier_tensor_count": EXPECTED_VERIFIER_STATE_TENSORS,
        "verifier_element_count": EXPECTED_VERIFIER_STATE_ELEMENTS,
        "hidden_dim": 256,
        "decoder_num_layers": 6,
        "verifier_matches_rank": True,
        "verifier_veto_output_nonzero_count": 0,
        "pool_output_nonzero_count": 0,
        "retired_confidence_loaded_tensor_count": 0,
    }
    drift = {
        key: (audit.get(key), value)
        for key, value in expected.items()
        if audit.get(key) != value
    }
    fresh = audit.get("fresh_confidence")
    if (
        drift
        or not isinstance(fresh, Mapping)
        or fresh.get("tensor_count") != EXPECTED_FRESH_TENSORS
        or fresh.get("element_count") != EXPECTED_FRESH_ELEMENTS
        or fresh.get("sha256") != EXPECTED_FRESH_SHA256
    ):
        raise ProbeHealthEvidenceError(
            f"V61 rank-cloned verifier migration surface drifted: {drift}"
        )
    return dict(audit)


_V53._audit_v53_migration = _audit_v61_migration
_V55._audit_v55_migration = _audit_v61_migration
_CORE._audit_split_ownership = _V53._audit_split_ownership
_CORE._audit_runtime = _V53._audit_runtime
_BASE_HEALTH_CHECKS = _V53._health_checks


def _health_checks_v61(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = dict(_BASE_HEALTH_CHECKS(runtime, trajectory))
    owner = checks.pop("u222_v53_owner_live_counts_exact")
    owner["requirement"] = "active=368, token-veto=356, global-absolute=12"
    checks["u222_v61_owner_live_counts_exact"] = owner
    clip = checks.pop("u222_v53_two_independent_clips_exact")
    checks["u222_v61_two_independent_clips_exact"] = clip
    return checks


_CORE._health_checks = _health_checks_v61
_BASE_AUDIT_TRAINING_CONTRACT = _CORE._audit_training_contract


def _audit_v61_training_contract(args: Mapping[str, Any]) -> dict[str, Any]:
    direct_expected = {
        "stage_b_dense_duty_confidence_full_decoder_verifier": True,
        "stage_b_dense_duty_confidence_capacity_contract": (
            EXPECTED_CAPACITY_CONTRACT
        ),
        "stage_b_dense_duty_confidence_variant": EXPECTED_VARIANT,
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
            f"V61 direct training contract drifted: {drift}"
        )
    return _BASE_AUDIT_TRAINING_CONTRACT(args)


_CORE._audit_training_contract = _audit_v61_training_contract


def audit() -> dict[str, Any]:
    return _V53._BASE_AUDIT()


_CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
