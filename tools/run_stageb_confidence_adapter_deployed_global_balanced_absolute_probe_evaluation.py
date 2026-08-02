#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the fixed V57 terminal-U400 strict1607 diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    audit_stageb_confidence_adapter_deployed_global_balanced_absolute_probe_health
    as health,
)
from tools import (  # noqa: E402
    run_stageb_confidence_adapter_deployed_global_balanced_absolute_probe_u0400
    as training,
)


_BASE_PATH = REPO_ROOT / (
    "tools/run_stageb_confidence_adapter_deployment_owned_global_"
    "probe_evaluation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_deployed_global_balanced_absolute_probe_evaluation_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe evaluator controller: {_BASE_PATH}")
_V56 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V56)
_V55 = _V56._V55
_V54 = _V56._V54
_V53 = _V56._V53
_BASE = _V56._BASE
_CORE = _V56._CORE


SCHEMA = (
    "pivot.stageb.confidence_adapter_deployed_global_balanced_absolute_"
    "probe_evaluation/v1"
)
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_adapter_deployed_global_balanced_absolute_"
    "probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_adapter_deployed_global_balanced_absolute_"
    "formal_admission/v1"
)
HEALTH_SCHEMA = health.SCHEMA
CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployed_global_"
    "balanced_absolute_20260802.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_deployed_global_balanced_absolute_highmem_20260802/"
    "probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
EXPECTED_UPDATES = 400
EXPECTED_REVISION = (
    "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57"
)
FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_deployed_global_balanced_absolute_"
    "confidence_strict1607_v57"
)
CONTROLLER_IMPORT = (
    "run_stageb_confidence_adapter_deployed_global_balanced_absolute_"
    "probe_evaluation"
)
MAIN_SOURCE = REPO_ROOT / "main.py"
ProbeEvaluationError = _CORE.ProbeEvaluationError


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
    "EXPECTED_REVISION": EXPECTED_REVISION,
    "FORMAL_ADMISSION_CONTRACT": FORMAL_ADMISSION_CONTRACT,
    "CONTROLLER_IMPORT": CONTROLLER_IMPORT,
    "MAIN_SOURCE": MAIN_SOURCE,
    "training": training,
    "health": health,
}
for _module in (_V56, _V55, _V54, _V53, _BASE, _CORE):
    for _name, _value in _OVERRIDES.items():
        setattr(_module, _name, _value)

_CORE.FORMAL_PROMOTION_OVERRIDES = {
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
_CORE._load_health_audit = lambda: health.audit
_BASE_V56_POSTFLIGHT = _V56._v56_postflight


def _v57_postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    result = dict(
        _BASE_V56_POSTFLIGHT(preflight_report, summary_path=summary_path)
    )
    contracts = dict(result.get("contracts", {}))
    if contracts.pop("v56_deployment_owned_global_representation_v38", None) is not True:
        raise ProbeEvaluationError("V57 postflight lacks inherited V56 ownership")
    contracts.update(
        {
            "v57_deployed_global_balanced_absolute_v39": True,
            "candidate_local_absolute_weight_is_exactly_zero": True,
            "deployed_global_absolute_weight_is_exactly_one": True,
            "deployed_global_absolute_gamma_is_exactly_one": True,
            "balanced_absolute_loss_uses_true_deployed_global_logits": True,
        }
    )
    result["contracts"] = contracts
    return result


_BASE.postflight = _v57_postflight
_CORE.postflight = _v57_postflight


def _formal_main_admission_is_wired(path: Path | None = None) -> bool:
    return _V54._formal_main_admission_is_wired(path)


def build_command() -> list[str]:
    return _CORE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    return _CORE.preflight(health_audit=health_audit)


def postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    return _v57_postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(path: Path | None = None, *, health_audit=None):
    if not _formal_main_admission_is_wired():
        raise ProbeEvaluationError(
            "V57 strict1607 evidence cannot promote formal training until main "
            "has the exact deployed-global balanced-absolute binding"
        )
    return _CORE.verify_admission_report(path, health_audit=health_audit)


def run() -> int:
    return _CORE.run()


def status():
    return _CORE.status()


def main(argv: Sequence[str] | None = None) -> int:
    return _CORE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
