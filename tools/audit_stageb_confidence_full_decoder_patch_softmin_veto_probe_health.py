#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V62 patch-softmin veto-only U400 probe."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    audit_stageb_confidence_full_decoder_verifier_probe_health as v61,
)
from tools import (  # noqa: E402
    run_stageb_confidence_full_decoder_patch_softmin_veto_probe_u0400 as training,
)
from util.stage_b_confidence_adapter_migration import (  # noqa: E402
    FULL_DECODER_PATCH_SOFTMIN_VETO_FRESH_CONFIDENCE_CONTRACT,
    FULL_DECODER_PATCH_SOFTMIN_VETO_MIGRATION_SCHEMA,
)


SCHEMA = "pivot.stageb.confidence_full_decoder_patch_softmin_veto_probe_health/v1"
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v42"
EXPECTED_CAPACITY_CONTRACT = "rank_cloned_full_decoder_patch_softmin_veto_v2"
EXPECTED_VARIANT = (
    "full_decoder_token_entailment_patch_weighted_existential_veto_v62"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_full_decoder_patch_softmin_veto_"
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
MIGRATION_SCHEMA = FULL_DECODER_PATCH_SOFTMIN_VETO_MIGRATION_SCHEMA
FRESH_CONFIDENCE_CONTRACT = (
    FULL_DECODER_PATCH_SOFTMIN_VETO_FRESH_CONFIDENCE_CONTRACT
)
ProbeHealthEvidenceError = v61.ProbeHealthEvidenceError

_VERIFIER_TOWER_PREFIX = "stage_b_fixed_text_scorer.confidence_verifier_tower."
_VERIFIER_HEAD_PREFIX = "stage_b_fixed_text_scorer.confidence_verifier_veto_head."

_MODULES = (v61, v61._V60, v61._V59, v61._V56, v61._V55, v61._V53, v61._BASE, v61._CORE)
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
    "FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT": FRESH_CONFIDENCE_CONTRACT,
    "training": training,
}
for _module in _MODULES:
    for _name, _value in _OVERRIDES.items():
        setattr(_module, _name, _value)

v61._CORE._CONFIDENCE_ADAPTER_PREFIX = (
    _VERIFIER_TOWER_PREFIX,
    _VERIFIER_HEAD_PREFIX,
)
v61._CORE._CONFIDENCE_POOL_PREFIX = ()
v61._CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_full_decoder_patch_softmin_veto_health_audit.json"
)
v61._CORE.EXPECTED_CONTRACT_VALUES = {
    **v61._CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_v11_trainable_params_min": EXPECTED_ACTIVE_ELEMENTS,
    "stage_b_v11_trainable_params_max": EXPECTED_ACTIVE_ELEMENTS,
}


def _owner_for_name(name: str) -> str:
    if name.startswith(_VERIFIER_TOWER_PREFIX):
        return "token_veto"
    if name.startswith(_VERIFIER_HEAD_PREFIX):
        return "global_absolute"
    raise ProbeHealthEvidenceError(f"unknown V62 active parameter owner: {name}")


v61._V53._owner_for_name = _owner_for_name


def _audit_v62_migration(args: Mapping[str, Any]) -> dict[str, Any]:
    audit = v61._audit_v61_migration(args)
    if audit.get("patch_softmin_veto_only") is not True:
        raise ProbeHealthEvidenceError(
            "V62 migration lacks the exact patch-softmin veto-only marker"
        )
    return audit


v61._V53._audit_v53_migration = _audit_v62_migration
v61._V55._audit_v55_migration = _audit_v62_migration
v61._CORE._audit_split_ownership = v61._V53._audit_split_ownership
v61._CORE._audit_runtime = v61._V53._audit_runtime


def _health_checks_v62(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = dict(v61._BASE_HEALTH_CHECKS(runtime, trajectory))
    owner = checks.pop("u222_v53_owner_live_counts_exact")
    owner["requirement"] = "active=362, token-veto=356, global-absolute=6"
    checks["u222_v62_owner_live_counts_exact"] = owner
    clip = checks.pop("u222_v53_two_independent_clips_exact")
    checks["u222_v62_two_independent_clips_exact"] = clip
    return checks


v61._CORE._health_checks = _health_checks_v62


def _audit_v62_training_contract(args: Mapping[str, Any]) -> dict[str, Any]:
    direct_expected = {
        "stage_b_dense_duty_confidence_full_decoder_verifier": True,
        "stage_b_dense_duty_confidence_veto_only_patch_softmin": True,
        "stage_b_dense_duty_confidence_capacity_contract": EXPECTED_CAPACITY_CONTRACT,
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
        raise ProbeHealthEvidenceError(f"V62 direct training contract drifted: {drift}")
    return v61._BASE_AUDIT_TRAINING_CONTRACT(args)


v61._CORE._audit_training_contract = _audit_v62_training_contract


def audit() -> dict[str, Any]:
    return v61._V53._BASE_AUDIT()


v61._CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return v61._CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
