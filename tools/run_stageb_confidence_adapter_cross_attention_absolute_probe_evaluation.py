#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run and audit the v28 U300 strict1607 confidence diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import audit_stageb_confidence_adapter_veto_probe_health as health
from tools import run_stageb_confidence_adapter_cross_attention_absolute_probe_u0300 as training
from tools import run_stageb_confidence_adapter_veto_probe_evaluation as _BASE


CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "cross_attention_absolute_probe_u0300_20260731.py"
)
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "cross_attention_absolute_20260731.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_cross_attention_absolute_highmem_20260731/"
    "probe_evaluation/u000300_strict1607"
)
REPORT = OUTPUT.parent / "u000300_strict1607_report.json"
LOG = OUTPUT.parent / "u000300_strict1607_console.log"
FORMAL_ADMISSION_CONTRACT = (
    "u300_word_veto_cross_attention_absolute_confidence_strict1607_v28"
)

_BASE.training = training
health.probe = training
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
    "stage_b_dense_duty_confidence_expected_optimizer_updates": (300, 4412),
    "stage_b_dense_duty_evaluation_scope": ("probe", "formal"),
    "stage_b_dense_duty_execution_scope": ("probe", "formal"),
    "stage_b_dense_duty_confidence_probe_admission_contract": (
        "disabled_for_probe_v1",
        FORMAL_ADMISSION_CONTRACT,
    ),
    "stage_b_dense_duty_confidence_probe_admission_report": (
        "",
        str(REPORT),
    ),
}
_BASE._load_health_audit = lambda: health.audit


def main(argv: Sequence[str] | None = None) -> int:
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
