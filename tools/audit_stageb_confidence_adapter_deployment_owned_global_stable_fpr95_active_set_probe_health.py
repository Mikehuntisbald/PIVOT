#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed health audit for the V58 terminal U400 probe."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_deployment_owned_global_stable_fpr95_active_set_probe_u0400
    as training,
)


_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_deployment_owned_global_"
    "probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_deployment_owned_stable_fpr95_active_set_health_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load V56 health base: {_BASE_PATH}")
v56 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v56)


SCHEMA = (
    "pivot.stageb.confidence_adapter_deployment_owned_global_"
    "stable_fpr95_active_set_probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v40"
EXPECTED_REVISION = (
    "word_veto_rank_full_expression_deployment_owned_global_"
    "stable_fpr95_active_set_v58"
)
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_"
    "stable_fpr95_active_set_probe_u0400_20260802.py"
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
    "stage_b_v15_tail_queue_negative_reduction_contract": (
        "exact_fpr95_active_set_all_count_mean_v2"
    ),
}
v56._CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_deployment_owned_stable_fpr95_active_set_health_audit.json"
)


def _load_active_set_trajectory() -> dict[str, float]:
    if not LOG_PATH.is_file():
        raise ProbeHealthEvidenceError(f"V58 training log is missing: {LOG_PATH}")
    rows = []
    with LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and "train_optimizer_updates" in row:
                rows.append(row)
    if not rows:
        raise ProbeHealthEvidenceError("V58 log has no optimizer trajectory row")
    row = rows[-1]
    keys = (
        "train_fixed_text_tail_queue_negative_total_count_unscaled",
        "train_fixed_text_tail_queue_negative_active_count_unscaled",
        "train_fixed_text_tail_queue_negative_selected_count_unscaled",
        "train_fixed_text_tail_queue_negative_active_fraction_unscaled",
        "train_fixed_text_tail_queue_negative_loss_unscaled",
    )
    values = {}
    for key in keys:
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ProbeHealthEvidenceError(f"V58 log field {key} is not finite")
        values[key] = float(value)
    total = values[keys[0]]
    active = values[keys[1]]
    selected = values[keys[2]]
    if not (
        total > 0.0
        and 0.0 < active < total
        and math.isclose(selected, active, rel_tol=1e-7, abs_tol=1e-7)
        and values[keys[4]] > 0.0
    ):
        raise ProbeHealthEvidenceError(
            "V58 active-set loss was not live, selective, and exactly selected"
        )
    return values


def audit() -> dict[str, Any]:
    result = dict(v56._V53._BASE_AUDIT())
    result["schema"] = SCHEMA
    result["stable_fpr95_active_set"] = _load_active_set_trajectory()
    checks = dict(result.get("checks", {}))
    checks["stable_fpr95_active_set_live"] = {
        "passed": True,
        "requirement": (
            "selected==active<all valid TNs with finite positive loss"
        ),
        "observed": result["stable_fpr95_active_set"],
    }
    result["checks"] = checks
    return result


v56._CORE.audit = audit


def run(argv: Sequence[str] | None = None) -> int:
    return v56._CORE.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
