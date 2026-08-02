#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run and audit the v4 absolute-cap U300 strict1607 diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import audit_stageb_confidence_adapter_veto_cap_probe_health as health  # noqa: E402
from tools import run_stageb_confidence_adapter_veto_cap_probe as training  # noqa: E402


_BASE_PATH = REPO_ROOT / "tools/run_stageb_confidence_adapter_veto_probe_evaluation.py"
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_confidence_adapter_veto_cap_evaluation_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load promotion evaluator: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

SCHEMA = "pivot.stageb.confidence_adapter_veto_cap_probe_evaluation/v4"
POSTFLIGHT_SCHEMA = "pivot.stageb.confidence_adapter_veto_cap_probe_evaluation_postflight/v4"
ADMISSION_SCHEMA = "pivot.stageb.confidence_adapter_veto_cap_formal_admission/v4"
HEALTH_SCHEMA = health.SCHEMA
CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_cap_probe_20260731.py"
FORMAL_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_cap_20260731.py"
CHECKPOINT = REPO_ROOT / "outputs/paper_cvpr_v1/dense_duty_adapter_veto_cap_highmem_20260731/probe/u000300/checkpoint_iter.pth"
OUTPUT = REPO_ROOT / "outputs/paper_cvpr_v1/dense_duty_adapter_veto_cap_highmem_20260731/probe_evaluation/u000300_strict1607"
REPORT = OUTPUT.parent / "u000300_strict1607_report.json"
LOG = OUTPUT.parent / "u000300_strict1607_console.log"
FORMAL_PROMOTION_OVERRIDES = {
    "epochs": (2, 24),
    "stage_b_dense_duty_confidence_expected_optimizer_updates": (300, 4412),
    "stage_b_dense_duty_evaluation_scope": ("probe", "formal"),
    "stage_b_dense_duty_execution_scope": ("probe", "formal"),
    "stage_b_dense_duty_confidence_probe_admission_contract": (
        "disabled_for_probe_v1",
        "u300_word_veto_absolute_cap_strict1607_v4",
    ),
    "stage_b_dense_duty_confidence_probe_admission_report": ("", str(REPORT)),
}

for _name in (
    "SCHEMA",
    "POSTFLIGHT_SCHEMA",
    "ADMISSION_SCHEMA",
    "HEALTH_SCHEMA",
    "CONFIG",
    "FORMAL_CONFIG",
    "CHECKPOINT",
    "OUTPUT",
    "REPORT",
    "LOG",
    "FORMAL_PROMOTION_OVERRIDES",
):
    setattr(_BASE, _name, globals()[_name])
_BASE.training = training


def _load_health_audit():
    return health.audit


_BASE._load_health_audit = _load_health_audit


def build_command() -> list[str]:
    return _BASE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    return _BASE.preflight(health_audit=health_audit)


def postflight(preflight_report: Mapping[str, Any], *, summary_path=None):
    return _BASE.postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(report_path=None, *, health_audit=None):
    return _BASE.verify_admission_report(report_path, health_audit=health_audit)


def run() -> int:
    return _BASE.run()


def status():
    return _BASE.status()


def main(argv: Sequence[str] | None = None) -> int:
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
