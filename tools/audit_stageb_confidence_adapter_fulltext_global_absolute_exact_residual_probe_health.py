#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V54 terminal U400 confidence probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_probe_u0400
    as training,
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
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA,
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT,
)


_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_fulltext_global_absolute_"
    "probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_fulltext_global_absolute_exact_residual_probe_health_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_BASE_PATH}")
_V53 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V53)
_BASE = _V53._BASE
_CORE = _V53._CORE


SCHEMA = (
    "pivot.stageb.confidence_adapter_fulltext_global_absolute_exact_residual_"
    "probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v36"
EXPECTED_REVISION = (
    "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
)
EXPECTED_HEAD_CONTRACT = FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
EXPECTED_POOL_CONTRACT = (
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
)
EXPECTED_GATE_CONTRACT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
EXPECTED_POSITIVE_TRUST_CONTRACT = (
    "exact_frozen_rank_max_confidence_delta_v3"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "exact_residual_probe_u0400_20260802.py"
)
EXPECTED_SOURCE_UPDATES = 6551
EXPECTED_UPDATES = 400
LOG_PATH = Path(training.OUTPUT) / "log.txt"

# V54 deliberately keeps the V53 parameter and optimizer-owner surface exact.
EXPECTED_ACTIVE_TENSORS = (
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT
)
EXPECTED_ACTIVE_ELEMENTS = (
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT
)
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
MIGRATION_FRESH_STORAGE_BYTES = (
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES
)
MIGRATION_FRESH_SHA256 = EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256
MIGRATION_FINGERPRINT_IS_PLACEHOLDER = MIGRATION_FRESH_SHA256 == "0" * 64
MIGRATION_SCHEMA = FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
FRESH_CONFIDENCE_CONTRACT = (
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
)
TWO_OWNER_CLIP_CONTRACT_SCHEMA = _V53.TWO_OWNER_CLIP_CONTRACT_SCHEMA
ProbeHealthEvidenceError = _V53.ProbeHealthEvidenceError


# The inherited functions resolve these globals at call time. They are loaded
# into an isolated module object so the V53 controller remains untouched.
_V54_OVERRIDES = {
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
    "MIGRATION_FRESH_TENSOR_COUNT": MIGRATION_FRESH_TENSOR_COUNT,
    "MIGRATION_FRESH_ELEMENT_COUNT": MIGRATION_FRESH_ELEMENT_COUNT,
    "MIGRATION_FRESH_STORAGE_BYTES": MIGRATION_FRESH_STORAGE_BYTES,
    "MIGRATION_FRESH_SHA256": MIGRATION_FRESH_SHA256,
    "MIGRATION_FINGERPRINT_IS_PLACEHOLDER": MIGRATION_FINGERPRINT_IS_PLACEHOLDER,
    "FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA": MIGRATION_SCHEMA,
    "FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT": (
        FRESH_CONFIDENCE_CONTRACT
    ),
    "training": training,
}
for _name, _value in _V54_OVERRIDES.items():
    setattr(_V53, _name, _value)
for _module in (_BASE, _CORE):
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
        setattr(_module, _name, globals()[_name])
    _module.training = training

_CORE.EXPECTED_CONTRACT_VALUES = {
    **_CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_confidence_head_gradient_contract": (
        EXPECTED_HEAD_CONTRACT
    ),
    "stage_b_dense_duty_confidence_pool_feature_contract": (
        EXPECTED_POOL_CONTRACT
    ),
    "stage_b_dense_duty_confidence_gate_gradient_contract": (
        EXPECTED_GATE_CONTRACT
    ),
    "stage_b_dense_duty_positive_trust_contract": (
        EXPECTED_POSITIVE_TRUST_CONTRACT
    ),
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
_CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_fulltext_global_absolute_exact_residual_health_audit.json"
)


def _audit_v54_migration(args: Mapping[str, Any]) -> dict[str, Any]:
    return _V53._audit_v53_migration(args)


_audit_split_ownership = _V53._audit_split_ownership
_audit_runtime = _V53._audit_runtime
_health_checks = _V53._health_checks


def audit() -> dict[str, Any]:
    if MIGRATION_FINGERPRINT_IS_PLACEHOLDER:
        raise ProbeHealthEvidenceError(
            "V54 migration fresh-state SHA256 is still the all-zero placeholder"
        )
    return _V53._BASE_AUDIT()


_CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
