#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run, audit, and promote the v35 U400 strict1607 probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (
    run_stageb_confidence_adapter_candidate_q05_probe_evaluation as q05_health,
)
from tools import (
    run_stageb_confidence_adapter_candidate_tail_balanced_probe_u0400 as training,
)


_BASE_PATH = REPO_ROOT / "tools/run_stageb_confidence_adapter_veto_probe_evaluation.py"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_candidate_tail_balanced_probe_evaluation_base",
    _BASE_PATH,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load probe evaluator controller: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

_HEALTH_PATH = REPO_ROOT / "tools/audit_stageb_confidence_adapter_veto_probe_health.py"
_HEALTH_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_candidate_tail_balanced_probe_health",
    _HEALTH_PATH,
)
if _HEALTH_SPEC is None or _HEALTH_SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_HEALTH_PATH}")
health = importlib.util.module_from_spec(_HEALTH_SPEC)
_HEALTH_SPEC.loader.exec_module(health)
health._health_checks = q05_health._q05_health_checks

CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_tail_balanced_probe_u0400_20260801.py"
)
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_tail_balanced_20260801.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_tail_balanced_highmem_20260801/"
    "probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_candidate_tail_balanced_confidence_strict1607_v35"
)
EXPECTED_UPDATES = 400

_BASE.training = training
_BASE.EXPECTED_UPDATES = EXPECTED_UPDATES
health.probe = training
health.EXPECTED_UPDATES = EXPECTED_UPDATES
for _name in (
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


def verify_admission_report(path: Path | None = None):
    return _BASE.verify_admission_report(path)


def main(argv: Sequence[str] | None = None) -> int:
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
