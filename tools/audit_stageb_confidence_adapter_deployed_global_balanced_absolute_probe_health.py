#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V57 terminal U400 probe."""

from __future__ import annotations

import json
import importlib.util
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_deployed_global_balanced_absolute_probe_u0400
    as training,
)


_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_deployment_owned_global_"
    "probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_deployed_global_balanced_absolute_health_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load V56 health base: {_BASE_PATH}")
v56 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v56)


SCHEMA = (
    "pivot.stageb.confidence_adapter_deployed_global_balanced_absolute_"
    "probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v39"
EXPECTED_REVISION = (
    "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployed_global_"
    "balanced_absolute_probe_u0400_20260802.py"
)
LOG_PATH = Path(training.OUTPUT) / "log.txt"
ProbeHealthEvidenceError = v56.ProbeHealthEvidenceError


_OVERRIDES = {
    "SCHEMA": SCHEMA,
    "TRAINING_CONTRACT_SCHEMA": TRAINING_CONTRACT_SCHEMA,
    "EXPECTED_REVISION": EXPECTED_REVISION,
    "EXPECTED_CONFIG_ENTRY": EXPECTED_CONFIG_ENTRY,
    "LOG_PATH": LOG_PATH,
    "training": training,
}
for _module in (v56, v56._V55, v56._V53, v56._BASE, v56._CORE):
    for _name, _value in _OVERRIDES.items():
        setattr(_module, _name, _value)

v56._CORE.EXPECTED_CONTRACT_VALUES = {
    **v56._CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_deployed_global_absolute_weight": 1.0,
    "stage_b_dense_duty_deployed_global_absolute_gamma": 1.0,
}
v56._CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_deployed_global_balanced_absolute_health_audit.json"
)


def _load_logged_trajectory() -> dict[str, Any]:
    if not LOG_PATH.is_file():
        raise ProbeHealthEvidenceError(f"V57 training log is missing: {LOG_PATH}")
    rows = []
    with LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and "train_optimizer_updates" in row:
                rows.append(dict(row))
    if not rows:
        raise ProbeHealthEvidenceError("V57 log has no optimizer trajectory row")
    row = rows[-1]
    required = {
        "train_loss_fixed_text_deployed_global_absolute_unscaled",
        "train_fixed_text_deployed_global_absolute_positive_sample_count_unscaled",
        "train_fixed_text_deployed_global_absolute_tn_sample_count_unscaled",
        "train_fixed_text_deployed_global_absolute_positive_loss_unscaled",
        "train_fixed_text_deployed_global_absolute_tn_loss_unscaled",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ProbeHealthEvidenceError(
            f"V57 log lacks deployed-global absolute evidence: {missing}"
        )
    for key in required:
        value = row[key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ProbeHealthEvidenceError(f"V57 log field {key} is not finite")
    if (
        float(row["train_loss_fixed_text_deployed_global_absolute_unscaled"])
        <= 0.0
        or float(
            row[
                "train_fixed_text_deployed_global_absolute_positive_sample_count_unscaled"
            ]
        )
        <= 0.0
        or float(
            row[
                "train_fixed_text_deployed_global_absolute_tn_sample_count_unscaled"
            ]
        )
        <= 0.0
    ):
        raise ProbeHealthEvidenceError(
            "V57 deployed-global balanced loss was not live on both classes"
        )
    return {key: row[key] for key in sorted(required)}


def audit() -> dict[str, Any]:
    result = dict(v56._V53._BASE_AUDIT())
    result["schema"] = SCHEMA
    result["deployed_global_balanced_absolute"] = _load_logged_trajectory()
    checks = dict(result.get("checks", {}))
    checks["deployed_global_balanced_absolute_live"] = {
        "passed": True,
        "requirement": (
            "the exact deployed sample-global positive and TN logits both "
            "contribute finite balanced focal-BCE"
        ),
        "observed": result["deployed_global_balanced_absolute"],
    }
    result["checks"] = checks
    return result


v56._CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return v56._CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
