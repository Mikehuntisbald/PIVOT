#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the U300 ungated monotone tail-veto confidence screen."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = REPO_ROOT / "tools/run_stageb_confidence_adapter_formal.py"
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_confidence_adapter_ungated_monotone_tail_veto_u0300_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load confidence controller: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "ungated_monotone_tail_veto_probe_u0300_20260731.py"
)
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_ungated_monotone_tail_veto_highmem_20260731/"
    "probe/u000300"
)
CHECKPOINT = OUTPUT / "checkpoint_iter.pth"
LOCK = OUTPUT.parent.parent / ".confidence_adapter_ungated_monotone_tail_veto_u0300.lock"
LOG = OUTPUT.parent / "u000300_controller.log"
UPDATES = 300

for _name in ("CONFIG", "OUTPUT", "CHECKPOINT", "LOCK", "LOG", "UPDATES"):
    setattr(_BASE, _name, globals()[_name])


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
