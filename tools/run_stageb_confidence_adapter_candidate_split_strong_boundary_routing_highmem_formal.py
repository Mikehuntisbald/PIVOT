#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run fresh V50 formal confidence training after strict1607 admission."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = REPO_ROOT / "tools/run_stageb_confidence_adapter_formal.py"
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_confidence_adapter_candidate_split_strong_boundary_routing_formal_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load formal controller: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_strong_boundary_routing_20260801.py"
)
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_strong_boundary_routing_highmem_20260801/"
    "formal/confidence"
)
CHECKPOINT = OUTPUT / "checkpoint_iter.pth"
LOCK = OUTPUT.parent.parent / (
    ".formal_confidence_adapter_split_strong_boundary_routing.lock"
)
LOG = OUTPUT.parent / "controller.log"

for _name in ("CONFIG", "OUTPUT", "CHECKPOINT", "LOCK", "LOG"):
    setattr(_BASE, _name, globals()[_name])

UPDATES = _BASE.UPDATES
SOURCE_CLOSURE_ARG = _BASE.SOURCE_CLOSURE_ARG


def verify_probe_admission():
    from tools import (
        run_stageb_confidence_adapter_candidate_split_strong_boundary_routing_probe_evaluation as promotion,
    )

    if promotion.FORMAL_CONFIG.resolve(strict=True) != CONFIG.resolve(strict=True):
        raise _BASE.ControllerError(
            "probe admission promotes a different formal config"
        )
    return promotion.verify_admission_report()


_BASE.FORMAL_ADMISSION_VALIDATOR = verify_probe_admission


def _formal_current_args():
    return _BASE._formal_current_args()


def inspect():
    return _BASE.inspect()


def command(action: str):
    return _BASE.command(action)


def validate_inputs() -> None:
    _BASE.validate_inputs()


def main(argv: Sequence[str] | None = None) -> int:
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
