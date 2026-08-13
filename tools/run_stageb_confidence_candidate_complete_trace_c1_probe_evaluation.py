#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the terminal-U400 strict1607 gate for candidate-complete C1."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

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
_BASELINE_MODULE_STATE = {
    id(module): (
        module,
        {
            name: getattr(module, name, _MISSING)
            for name in _SHARED_OVERRIDE_NAMES
        },
    )
    for module in _BASELINE_MODULES
}
_BASELINE_CORE_STATE = {
    "FORMAL_PROMOTION_OVERRIDES": _v61_baseline._CORE.FORMAL_PROMOTION_OVERRIDES,
    "_load_health_audit": _v61_baseline._CORE._load_health_audit,
    "postflight": _v61_baseline._CORE.postflight,
}
_BASELINE_POSTFLIGHT = _v61_baseline._V60._V59._BASE.postflight

from tools import (
    audit_stageb_confidence_candidate_complete_trace_c1_probe_health as health,
)
from tools import (  # noqa: E402
    run_stageb_confidence_full_decoder_patch_softmin_veto_probe_evaluation as v62,
)
from tools import (
    run_stageb_confidence_candidate_complete_trace_c1_probe_u0400 as training,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot.stageb.confidence_candidate_complete_trace_c1_probe_evaluation/v1"
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_candidate_complete_trace_c1_"
    "probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_candidate_complete_trace_c1_formal_admission/v1"
)
HEALTH_SCHEMA = health.SCHEMA
CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_candidate_complete_trace_c1_"
    "formal_20260803.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/dense_duty_candidate_complete_trace_20260803/"
    "c1_free_head_coverage/probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
EXPECTED_UPDATES = 400
EXPECTED_TRAINABLE_PARAMETERS = 25_530_881
EXPECTED_CAPACITY_CONTRACT = (
    "rank_cloned_full_decoder_candidate_complete_free_head_v3"
)
FORMAL_ADMISSION_CONTRACT = (
    "u400_candidate_complete_trace_c1_strict1607_v1"
)
CONTROLLER_IMPORT = (
    "run_stageb_confidence_candidate_complete_trace_c1_probe_evaluation"
)
MAIN_SOURCE = REPO_ROOT / "main.py"
ProbeEvaluationError = v62.ProbeEvaluationError

_CORE = v62._CORE
_MODULES = (v62, *v62._MODULES)
_OVERRIDES = {
    "SCHEMA": SCHEMA,
    "POSTFLIGHT_SCHEMA": POSTFLIGHT_SCHEMA,
    "ADMISSION_SCHEMA": ADMISSION_SCHEMA,
    "HEALTH_SCHEMA": HEALTH_SCHEMA,
    "CONFIG": CONFIG,
    "FORMAL_CONFIG": FORMAL_CONFIG,
    "CHECKPOINT": CHECKPOINT,
    "FIXED_PYTHON": FIXED_PYTHON,
    "OUTPUT": OUTPUT,
    "REPORT": REPORT,
    "LOG": LOG,
    "EXPECTED_UPDATES": EXPECTED_UPDATES,
    "EXPECTED_TRAINABLE_PARAMETERS": EXPECTED_TRAINABLE_PARAMETERS,
    "EXPECTED_CAPACITY_CONTRACT": EXPECTED_CAPACITY_CONTRACT,
    "FORMAL_ADMISSION_CONTRACT": FORMAL_ADMISSION_CONTRACT,
    "CONTROLLER_IMPORT": CONTROLLER_IMPORT,
    "MAIN_SOURCE": MAIN_SOURCE,
    "training": training,
    "health": health,
}
_C1_FORMAL_PROMOTION_OVERRIDES = {
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


def _c1_postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    result = dict(v62._v62_postflight(preflight_report, summary_path=summary_path))
    result["contracts"] = {
        "tn_only": True,
        "full_strict1607": True,
        "terminal_u400_diagnostic": True,
        "candidate_complete_trace_c1_training_contract_v43": True,
        EXPECTED_CAPACITY_CONTRACT: True,
        "rank_tower_is_frozen_and_parameter_disjoint": True,
        "free_veto_head_retained_only_for_coverage_hypothesis": True,
        "exact_deployed_top50_depth_supervision": True,
        "shallowest_tn_escape_receives_direct_loss": True,
        "positive_depth_protection_is_target_iou_existential": True,
        "changed_token_broadcast_is_provenance_gated": True,
        "current_manifest_candidate_token_broadcast_rows_are_zero": True,
        "deployed_global_logit_is_exactly_negative_patch_softmin_veto": True,
        "still_not_formal_evaluation": True,
    }
    return result


def _activate_c1() -> None:
    for _module in _MODULES:
        for _name, _value in _OVERRIDES.items():
            setattr(_module, _name, _value)
    _CORE.FORMAL_PROMOTION_OVERRIDES = _C1_FORMAL_PROMOTION_OVERRIDES
    _CORE._load_health_audit = lambda: health.audit
    v62.v61._V60._V59._BASE.postflight = _c1_postflight
    _CORE.postflight = _c1_postflight


def _restore_shared_state() -> None:
    for _module, _values in _BASELINE_MODULE_STATE.values():
        for _name, _value in _values.items():
            if _value is _MISSING:
                _module.__dict__.pop(_name, None)
            else:
                setattr(_module, _name, _value)
    v62.v61._V60._V59._BASE.postflight = _BASELINE_POSTFLIGHT
    for _name, _value in _BASELINE_CORE_STATE.items():
        setattr(_CORE, _name, _value)


_C1_SCOPE_DEPTH = 0


@contextmanager
def _c1_scope():
    global _C1_SCOPE_DEPTH
    outermost = _C1_SCOPE_DEPTH == 0
    if outermost:
        _activate_c1()
    _C1_SCOPE_DEPTH += 1
    try:
        yield
    finally:
        _C1_SCOPE_DEPTH -= 1
        if outermost:
            _restore_shared_state()


def build_command() -> list[str]:
    with _c1_scope():
        return _CORE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    with _c1_scope():
        return _CORE.preflight(health_audit=health_audit)


def postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    with _c1_scope():
        return _c1_postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(
    path: Path | None = None, *, health_audit=None
) -> dict[str, Any]:
    with _c1_scope():
        return _CORE.verify_admission_report(path, health_audit=health_audit)


def run() -> int:
    with _c1_scope():
        return _CORE.run()


def status():
    with _c1_scope():
        return _CORE.status()


def main(argv: Sequence[str] | None = None) -> int:
    with _c1_scope():
        return _CORE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


_restore_shared_state()
