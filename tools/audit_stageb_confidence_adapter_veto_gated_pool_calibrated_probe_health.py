#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Audit U300 health for calibrated carrier-gated confidence."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_confidence_adapter_veto_gated_pool_calibrated_probe as probe  # noqa: E402


_BASE_PATH = REPO_ROOT / "tools/audit_stageb_confidence_adapter_veto_gate_probe_health.py"
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_confidence_adapter_veto_gated_pool_calibrated_health_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load health auditor: {_BASE_PATH}")
_GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE)
_BASE = _GATE._BASE

SCHEMA = "pivot.stageb.confidence_adapter_veto_gated_pool_probe_health/v6"
_GATED_POOL_FIELDS = (
    "train_stage_b_dense_confidence_positive_veto_coverage_mean_unscaled",
    "train_stage_b_dense_confidence_tn_veto_coverage_mean_unscaled",
    "train_stage_b_dense_confidence_positive_veto_sample_gate_mean_unscaled",
    "train_stage_b_dense_confidence_tn_veto_sample_gate_mean_unscaled",
    "train_stage_b_dense_confidence_veto_ceiling_unscaled",
)

_BASE.probe = probe
_BASE.SCHEMA = SCHEMA
_BASE.REQUIRED_U222_FIELDS = _BASE.REQUIRED_U222_FIELDS + _GATED_POOL_FIELDS
_BASE_HEALTH_CHECKS = _GATE._health_checks


def _health_checks(
    trajectory: Mapping[str, float | int],
    endpoint: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    checks = _BASE_HEALTH_CHECKS(trajectory, endpoint, runtime)
    positive_gate = float(
        trajectory[
            "train_stage_b_dense_confidence_positive_veto_sample_gate_mean_unscaled"
        ]
    )
    tn_gate = float(
        trajectory[
            "train_stage_b_dense_confidence_tn_veto_sample_gate_mean_unscaled"
        ]
    )
    ceiling = float(
        trajectory["train_stage_b_dense_confidence_veto_ceiling_unscaled"]
    )
    checks.update(
        {
            "u222_positive_carrier_gate_closed": _BASE._check(
                positive_gate, "<= 0.05", positive_gate <= 0.05
            ),
            "u222_tn_carrier_gate_open": _BASE._check(
                tn_gate, ">= 0.80", tn_gate >= 0.80
            ),
            "u222_absolute_ceiling_nonpositive": _BASE._check(
                ceiling, "<= -0.05", ceiling <= -0.05
            ),
        }
    )
    return checks


_BASE._health_checks = _health_checks


def audit() -> dict[str, Any]:
    return _BASE.audit()


def run(argv: Sequence[str] | None = None) -> int:
    return _BASE.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
