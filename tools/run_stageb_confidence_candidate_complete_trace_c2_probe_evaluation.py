#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the terminal-U400 strict1607 gate for candidate-complete C2."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (
    run_stageb_confidence_full_decoder_verifier_probe_evaluation as _v61_baseline,
)


_SHARED_OVERRIDE_NAMES = (
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
    "EXPECTED_UPDATES",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "EXPECTED_CAPACITY_CONTRACT",
    "FORMAL_ADMISSION_CONTRACT",
    "CONTROLLER_IMPORT",
    "MAIN_SOURCE",
    "training",
    "health",
)
_BASELINE_MODULES = (
    _v61_baseline,
    _v61_baseline._V60,
    _v61_baseline._V60._V59,
    _v61_baseline._V60._V59._V56,
    _v61_baseline._V60._V59._V55,
    _v61_baseline._V60._V59._V54,
    _v61_baseline._V60._V59._V53,
    _v61_baseline._V60._V59._BASE,
    _v61_baseline._CORE,
)
_MISSING = object()
_SHARED_STATE = {
    id(module): (
        module,
        {
            name: getattr(module, name, _MISSING)
            for name in _SHARED_OVERRIDE_NAMES
        },
    )
    for module in _BASELINE_MODULES
}
_BASE_POSTFLIGHT = _v61_baseline._V60._V59._BASE.postflight
_CORE_STATE = {
    "FORMAL_PROMOTION_OVERRIDES": _v61_baseline._CORE.FORMAL_PROMOTION_OVERRIDES,
    "_load_health_audit": _v61_baseline._CORE._load_health_audit,
    "postflight": _v61_baseline._CORE.postflight,
}

from tools import (  # noqa: E402
    audit_stageb_confidence_candidate_complete_trace_c2_probe_health as health,
)
from tools import (
    run_stageb_confidence_candidate_complete_trace_c2_probe_u0400 as training,
)
from tools import (
    run_stageb_confidence_full_decoder_patch_softmin_veto_probe_evaluation as v62,
)


SCHEMA = "pivot.stageb.confidence_candidate_complete_trace_c2_probe_evaluation/v1"
POSTFLIGHT_SCHEMA = "pivot.stageb.confidence_candidate_complete_trace_c2_probe_evaluation_postflight/v1"
ADMISSION_SCHEMA = "pivot.stageb.confidence_candidate_complete_trace_c2_formal_admission/v1"
HEALTH_SCHEMA = health.SCHEMA
CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_dense_duty_confidence_candidate_complete_trace_c2_formal_20260803.py"
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / "outputs/paper_cvpr_v1/dense_duty_candidate_complete_trace_20260803/c2_monotone_token_entailment/probe_evaluation/u000400_strict1607"
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
EXPECTED_UPDATES = 400
EXPECTED_TRAINABLE_PARAMETERS = 25_464_320
EXPECTED_CAPACITY_CONTRACT = health.EXPECTED_CAPACITY_CONTRACT
FORMAL_ADMISSION_CONTRACT = "u400_candidate_complete_trace_c2_strict1607_v1"
CONTROLLER_IMPORT = "run_stageb_confidence_candidate_complete_trace_c2_probe_evaluation"
MAIN_SOURCE = REPO_ROOT / "main.py"
ProbeEvaluationError = v62.ProbeEvaluationError

_CORE = v62._CORE
_MODULES = (v62, *v62._MODULES)
_OVERRIDES = {name: globals()[name] for name in ("SCHEMA", "POSTFLIGHT_SCHEMA", "ADMISSION_SCHEMA", "HEALTH_SCHEMA", "CONFIG", "FORMAL_CONFIG", "CHECKPOINT", "FIXED_PYTHON", "OUTPUT", "REPORT", "LOG", "EXPECTED_UPDATES", "EXPECTED_TRAINABLE_PARAMETERS", "EXPECTED_CAPACITY_CONTRACT", "FORMAL_ADMISSION_CONTRACT", "CONTROLLER_IMPORT", "MAIN_SOURCE")}
_OVERRIDES.update({"training": training, "health": health})
_C2_FORMAL_PROMOTION_OVERRIDES = {
    "epochs": (2, 24),
    "stage_b_dense_duty_confidence_expected_optimizer_updates": (400, 4412),
    "stage_b_dense_duty_evaluation_scope": ("probe", "formal"),
    "stage_b_dense_duty_execution_scope": ("probe", "formal"),
    "stage_b_dense_duty_confidence_probe_admission_contract": ("disabled_for_probe_v1", FORMAL_ADMISSION_CONTRACT),
    "stage_b_dense_duty_confidence_probe_admission_report": ("", str(REPORT)),
}


def _c2_postflight(preflight_report: Mapping[str, Any], *, summary_path: Path | None = None) -> dict[str, Any]:
    result = dict(v62._v62_postflight(preflight_report, summary_path=summary_path))
    result["contracts"] = {
        "tn_only": True,
        "full_strict1607": True,
        "terminal_u400_diagnostic": True,
        "candidate_complete_trace_c2_training_contract_v43": True,
        EXPECTED_CAPACITY_CONTRACT: True,
        "rank_tower_is_frozen_and_parameter_disjoint": True,
        "token_entailment_is_the_only_active_confidence_owner": True,
        "global_absolute_owner_is_absent": True,
        "one_owner_clip_contract_is_exact": True,
        "exact_deployed_top50_depth_supervision": True,
        "positive_depth_protection_is_target_iou_existential": True,
        "still_not_formal_evaluation": True,
    }
    return result


def _activate_c2() -> None:
    for _module in _MODULES:
        for _name, _value in _OVERRIDES.items():
            setattr(_module, _name, _value)
    _CORE.FORMAL_PROMOTION_OVERRIDES = _C2_FORMAL_PROMOTION_OVERRIDES
    _CORE._load_health_audit = lambda: health.audit
    v62.v61._V60._V59._BASE.postflight = _c2_postflight
    _CORE.postflight = _c2_postflight


def _restore_shared_state() -> None:
    for _module, _values in _SHARED_STATE.values():
        for _name, _value in _values.items():
            if _value is _MISSING:
                _module.__dict__.pop(_name, None)
            else:
                setattr(_module, _name, _value)
    v62.v61._V60._V59._BASE.postflight = _BASE_POSTFLIGHT
    for _name, _value in _CORE_STATE.items():
        setattr(_CORE, _name, _value)


_C2_SCOPE_DEPTH = 0


@contextmanager
def _c2_scope():
    global _C2_SCOPE_DEPTH
    outermost = _C2_SCOPE_DEPTH == 0
    if outermost:
        _activate_c2()
    _C2_SCOPE_DEPTH += 1
    try:
        yield
    finally:
        _C2_SCOPE_DEPTH -= 1
        if outermost:
            _restore_shared_state()


def build_command() -> list[str]:
    with _c2_scope(): return _CORE.build_command()
def preflight(*, health_audit=None) -> dict[str, Any]:
    with _c2_scope(): return _CORE.preflight(health_audit=health_audit)
def postflight(preflight_report: Mapping[str, Any], *, summary_path: Path | None = None) -> dict[str, Any]: return _c2_postflight(preflight_report, summary_path=summary_path)
def verify_admission_report(path: Path | None = None, *, health_audit=None) -> dict[str, Any]:
    with _c2_scope(): return _CORE.verify_admission_report(path, health_audit=health_audit)
def run() -> int:
    with _c2_scope(): return _CORE.run()
def status():
    with _c2_scope(): return _CORE.status()
def main(argv: Sequence[str] | None = None) -> int:
    with _c2_scope(): return _CORE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


_restore_shared_state()
