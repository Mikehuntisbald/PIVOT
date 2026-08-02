#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run, audit, and promote the v41 U400 role-complete strict1607 probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    audit_stageb_confidence_adapter_candidate_role_complete_carrier_probe_health as health,
)
from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_role_complete_carrier_probe_u0400 as training,
)

_BASE_PATH = REPO_ROOT / "tools/run_stageb_confidence_adapter_veto_probe_evaluation.py"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_role_complete_carrier_probe_evaluation_base",
    _BASE_PATH,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load probe evaluator controller: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_role_complete_carrier_"
    "probe_evaluation/v1"
)
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_role_complete_carrier_"
    "probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_role_complete_carrier_"
    "formal_admission/v1"
)
HEALTH_SCHEMA = health.SCHEMA

CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "role_complete_carrier_20260801.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_role_complete_carrier_highmem_20260801/"
    "probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_candidate_role_complete_carrier_confidence_strict1607_v41"
)
EXPECTED_UPDATES = 400

_BASE.training = training
_BASE.EXPECTED_UPDATES = EXPECTED_UPDATES
for _name in (
    "SCHEMA",
    "POSTFLIGHT_SCHEMA",
    "ADMISSION_SCHEMA",
    "HEALTH_SCHEMA",
    "CONFIG",
    "FORMAL_CONFIG",
    "CHECKPOINT",
    "FIXED_PYTHON",
    "OUTPUT",
    "REPORT",
    "LOG",
):
    setattr(_BASE, _name, globals()[_name])
_BASE.FORMAL_PROMOTION_OVERRIDES = {
    "epochs": (2, 24),
    "stage_b_dense_duty_confidence_expected_optimizer_updates": (400, 4412),
    "stage_b_dense_duty_evaluation_scope": ("probe", "formal"),
    "stage_b_dense_duty_execution_scope": ("probe", "formal"),
    "stage_b_dense_duty_confidence_probe_admission_contract": (
        "disabled_for_probe_v1",
        FORMAL_ADMISSION_CONTRACT,
    ),
    "stage_b_dense_duty_confidence_probe_admission_report": ("", str(REPORT)),
}
_BASE._load_health_audit = lambda: health.audit
_BASE_POSTFLIGHT = _BASE.postflight


def _v41_postflight(preflight_report: Mapping[str, Any], *, summary_path=None):
    result = dict(
        _BASE_POSTFLIGHT(preflight_report, summary_path=summary_path)
    )
    contracts = result.get("contracts")
    if (
        not isinstance(contracts, Mapping)
        or contracts.get("terminal_u300_diagnostic") is not True
    ):
        raise _BASE.ProbeEvaluationError(
            "v41 postflight lacks the inherited terminal diagnostic contract"
        )
    contracts = dict(contracts)
    del contracts["terminal_u300_diagnostic"]
    contracts["terminal_u400_diagnostic"] = True
    result["contracts"] = contracts
    return result


_BASE.postflight = _v41_postflight


def build_command() -> list[str]:
    return _BASE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    return _BASE.preflight(health_audit=health_audit)


def postflight(preflight_report: Mapping[str, Any], *, summary_path=None):
    return _BASE.postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(path: Path | None = None, *, health_audit=None):
    return _BASE.verify_admission_report(path, health_audit=health_audit)


def run() -> int:
    return _BASE.run()


def status():
    return _BASE.status()


def main(argv: Sequence[str] | None = None) -> int:
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
