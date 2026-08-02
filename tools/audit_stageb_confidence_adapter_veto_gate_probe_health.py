#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Audit U300 training health for the raw-gate word-veto adapter."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_confidence_adapter_veto_gate_probe as probe  # noqa: E402


_BASE_PATH = REPO_ROOT / "tools/audit_stageb_confidence_adapter_veto_probe_health.py"
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_confidence_adapter_veto_gate_health_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load health auditor: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

SCHEMA = "pivot.stageb.confidence_adapter_veto_gate_probe_health/v3"
_RAW_GATE_FIELDS = (
    "train_loss_fixed_text_raw_veto_gate_unscaled",
    "train_fixed_text_raw_veto_positive_sample_count_unscaled",
    "train_fixed_text_raw_veto_tn_sample_count_unscaled",
    "train_fixed_text_raw_veto_positive_query_count_unscaled",
    "train_fixed_text_raw_veto_tn_query_count_unscaled",
    "train_fixed_text_raw_veto_positive_source_mean_unscaled",
    "train_fixed_text_raw_veto_tn_changed_source_mean_unscaled",
    "train_fixed_text_raw_veto_source_separation_unscaled",
)

_BASE.probe = probe
_BASE.SCHEMA = SCHEMA
_BASE.REQUIRED_U222_FIELDS = _BASE.REQUIRED_U222_FIELDS + _RAW_GATE_FIELDS
_BASE_HEALTH_CHECKS = _BASE._health_checks


def _health_checks(
    trajectory: Mapping[str, float | int],
    endpoint: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    checks = _BASE_HEALTH_CHECKS(trajectory, endpoint, runtime)
    raw_loss = float(trajectory["train_loss_fixed_text_raw_veto_gate_unscaled"])
    positive_samples = float(
        trajectory["train_fixed_text_raw_veto_positive_sample_count_unscaled"]
    )
    tn_samples = float(
        trajectory["train_fixed_text_raw_veto_tn_sample_count_unscaled"]
    )
    positive_queries = float(
        trajectory["train_fixed_text_raw_veto_positive_query_count_unscaled"]
    )
    tn_queries = float(
        trajectory["train_fixed_text_raw_veto_tn_query_count_unscaled"]
    )
    positive_source = float(
        trajectory["train_fixed_text_raw_veto_positive_source_mean_unscaled"]
    )
    tn_source = float(
        trajectory["train_fixed_text_raw_veto_tn_changed_source_mean_unscaled"]
    )
    raw_separation = float(
        trajectory["train_fixed_text_raw_veto_source_separation_unscaled"]
    )
    checks.update(
        {
            "u222_raw_veto_loss_bounded": _BASE._check(
                raw_loss, "< 0.1", raw_loss < 0.1
            ),
            "u222_raw_veto_positive_support": _BASE._check(
                {"samples": positive_samples, "queries": positive_queries},
                "samples > 0 and queries > 0",
                positive_samples > 0.0 and positive_queries > 0.0,
            ),
            "u222_raw_veto_tn_support": _BASE._check(
                {"samples": tn_samples, "queries": tn_queries},
                "samples > 0 and queries > 0",
                tn_samples > 0.0 and tn_queries > 0.0,
            ),
            "u222_raw_veto_positive_source_closed": _BASE._check(
                positive_source, "< 0", positive_source < 0.0
            ),
            "u222_raw_veto_changed_tn_source_open": _BASE._check(
                tn_source, "> 0", tn_source > 0.0
            ),
            "u222_raw_veto_source_separation": _BASE._check(
                raw_separation, ">= 0.02", raw_separation >= 0.02
            ),
        }
    )
    return checks


_BASE._health_checks = _health_checks


def audit() -> dict[str, Any]:
    return _BASE.audit()


def run(argv: Sequence[str] | None = None) -> int:
    return _BASE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
