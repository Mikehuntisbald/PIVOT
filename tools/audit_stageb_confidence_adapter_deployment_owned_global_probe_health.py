#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V56 terminal U400 probe."""

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
    run_stageb_confidence_adapter_deployment_owned_global_probe_u0400 as training,
)
from util.stage_b_confidence_adapter_migration import (  # noqa: E402
    DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
    DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT,
    DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
    DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT,
    EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_ELEMENT_COUNT,
    EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_TENSOR_COUNT,
    EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_ELEMENT_COUNT,
    EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT,
    validate_confidence_adapter_migration_audit,
)


_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_fulltext_global_independent_"
    "absolute_probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_deployment_owned_global_probe_health_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_BASE_PATH}")
_V55 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V55)
_V53 = _V55._V53
_BASE = _V55._BASE
_CORE = _V55._CORE


SCHEMA = "pivot.stageb.confidence_adapter_deployment_owned_global_probe_health/v1"
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v38"
EXPECTED_REVISION = "word_veto_rank_full_expression_deployment_owned_global_v56"
EXPECTED_HEAD_CONTRACT = DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
EXPECTED_POOL_CONTRACT = DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
EXPECTED_GATE_CONTRACT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
EXPECTED_POSITIVE_TRUST_CONTRACT = "absolute_global_pool_logit_v4"
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_"
    "probe_u0400_20260802.py"
)
EXPECTED_SOURCE_UPDATES = 6551
EXPECTED_UPDATES = 400
LOG_PATH = Path(training.OUTPUT) / "log.txt"

EXPECTED_ACTIVE_TENSORS = EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_TENSOR_COUNT
EXPECTED_ACTIVE_ELEMENTS = EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_ELEMENT_COUNT
EXPECTED_TOKEN_TENSORS = 21
EXPECTED_TOKEN_ELEMENTS = 51_267
EXPECTED_GLOBAL_TENSORS = 38
EXPECTED_GLOBAL_ELEMENTS = 416_897
EXPECTED_LIVE_TOKEN_TENSORS = EXPECTED_TOKEN_TENSORS
EXPECTED_LIVE_GLOBAL_TENSORS = EXPECTED_GLOBAL_TENSORS
EXPECTED_ADAPTER_TENSORS = 53
EXPECTED_POOL_TENSORS = 6
MIGRATION_SCHEMA = DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
FRESH_CONFIDENCE_CONTRACT = (
    DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
)
ProbeHealthEvidenceError = _V55.ProbeHealthEvidenceError


_OVERRIDES = {
    "SCHEMA": SCHEMA,
    "TRAINING_CONTRACT_SCHEMA": TRAINING_CONTRACT_SCHEMA,
    "EXPECTED_REVISION": EXPECTED_REVISION,
    "EXPECTED_HEAD_CONTRACT": EXPECTED_HEAD_CONTRACT,
    "EXPECTED_POOL_CONTRACT": EXPECTED_POOL_CONTRACT,
    "EXPECTED_GATE_CONTRACT": EXPECTED_GATE_CONTRACT,
    "EXPECTED_CONFIG_ENTRY": EXPECTED_CONFIG_ENTRY,
    "EXPECTED_SOURCE_UPDATES": EXPECTED_SOURCE_UPDATES,
    "EXPECTED_UPDATES": EXPECTED_UPDATES,
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
    "FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA": MIGRATION_SCHEMA,
    "FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT": (
        FRESH_CONFIDENCE_CONTRACT
    ),
    "training": training,
}
for _module in (_V55, _V53, _BASE, _CORE):
    for _name, _value in _OVERRIDES.items():
        setattr(_module, _name, _value)

_CORE.EXPECTED_CONTRACT_VALUES = {
    **_CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_confidence_head_gradient_contract": EXPECTED_HEAD_CONTRACT,
    "stage_b_dense_duty_confidence_pool_feature_contract": EXPECTED_POOL_CONTRACT,
    "stage_b_dense_duty_confidence_gate_gradient_contract": EXPECTED_GATE_CONTRACT,
    "stage_b_dense_duty_positive_trust_contract": EXPECTED_POSITIVE_TRUST_CONTRACT,
    "stage_b_v14_local_absolute_weight": 0.0,
    "stage_b_v11_trainable_params_min": EXPECTED_ACTIVE_ELEMENTS,
    "stage_b_v11_trainable_params_max": EXPECTED_ACTIVE_ELEMENTS,
}
_CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_deployment_owned_global_health_audit.json"
)


def _audit_v56_migration(args: Mapping[str, Any]) -> dict[str, Any]:
    if args.get("stage_b_dense_duty_rank_source_optimizer_updates") != (
        EXPECTED_SOURCE_UPDATES
    ):
        raise ProbeHealthEvidenceError("V56 migration is not fresh from U6551")
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
        "head_gradient_contract": EXPECTED_HEAD_CONTRACT,
        "pool_feature_contract": EXPECTED_POOL_CONTRACT,
        "confidence_parameter_tensor_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT
        ),
        "confidence_parameter_element_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT
        ),
        "active_confidence_parameter_tensor_count": EXPECTED_ACTIVE_TENSORS,
        "active_confidence_parameter_element_count": EXPECTED_ACTIVE_ELEMENTS,
        "diagnostic_candidate_parameter_tensor_count": (
            EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_TENSOR_COUNT
        ),
        "diagnostic_candidate_parameter_element_count": (
            EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_ELEMENT_COUNT
        ),
        "diagnostic_candidate_requires_grad_count": 0,
    }
    drift = {
        key: (audit.get(key), value)
        for key, value in expected.items()
        if audit.get(key) != value
    }
    fresh = audit.get("fresh_confidence")
    if drift or not isinstance(fresh, Mapping) or fresh.get("sha256") != (
        _V55.MIGRATION_FRESH_SHA256
    ):
        raise ProbeHealthEvidenceError(
            f"V56 deployment-owned migration surface drifted: {drift}"
        )
    return dict(audit)


_V53._audit_v53_migration = _audit_v56_migration
_V55._audit_v55_migration = _audit_v56_migration
_CORE._audit_split_ownership = _V53._audit_split_ownership
_CORE._audit_runtime = _V53._audit_runtime
_CORE._health_checks = _V53._health_checks


def audit() -> dict[str, Any]:
    return _V53._BASE_AUDIT()


_CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
