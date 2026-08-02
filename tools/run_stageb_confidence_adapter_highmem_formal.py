#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed controller for the high-memory confidence-adapter recipe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = REPO_ROOT / "tools/run_stageb_confidence_adapter_formal.py"
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_confidence_adapter_highmem_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load formal controller: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_20260730.py"
)
OUTPUT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_packed_highmem_20260730/formal/confidence"
)
CHECKPOINT = OUTPUT / "checkpoint_iter.pth"
LOCK = OUTPUT.parent.parent / ".formal_confidence_adapter.lock"
LOG = OUTPUT.parent / "controller.log"

# Load the audited controller in an isolated module so this recipe can select a
# fresh output without mutating the original controller module in test runners.
_BASE.CONFIG = CONFIG
_BASE.OUTPUT = OUTPUT
_BASE.CHECKPOINT = CHECKPOINT
_BASE.LOCK = LOCK
_BASE.LOG = LOG

UPDATES = _BASE.UPDATES
SOURCE_CLOSURE_ARG = _BASE.SOURCE_CLOSURE_ARG


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
