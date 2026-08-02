#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the fixed V49 terminal-U400 strict1607 diagnostic."""

from __future__ import annotations

import ast
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    audit_stageb_confidence_adapter_candidate_split_global_trust_veto_probe_health as health,
)
from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_split_global_trust_veto_probe_u0400 as training,
)


_BASE_PATH = REPO_ROOT / (
    "tools/run_stageb_confidence_adapter_candidate_split_boundary_routing_"
    "probe_evaluation.py"
)
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_split_global_trust_veto_probe_evaluation_base", _BASE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load probe evaluator controller: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)
_CORE = _BASE._CORE


SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_split_global_trust_veto_"
    "probe_evaluation/v1"
)
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_split_global_trust_veto_"
    "probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_split_global_trust_veto_"
    "formal_admission/v1"
)
HEALTH_SCHEMA = health.SCHEMA

CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_global_trust_veto_20260801.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_global_trust_veto_highmem_20260801/"
    "probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
EXPECTED_UPDATES = 400
EXPECTED_REVISION = "word_veto_candidate_split_global_trust_veto_v49"
EXPECTED_HEAD_CONTRACT = "split_token_veto_global_trust_veto_v4"
EXPECTED_ROUTING_REDUCTION = "balanced_top_quarter_cvar_v2"
EXPECTED_TRUST_REDUCTION = "top_quarter_cvar_v2"
EXPECTED_NEGATIVE_REDUCTION = "all_mean_v1"
FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_candidate_split_global_trust_veto_confidence_"
    "strict1607_v49"
)
CONTROLLER_IMPORT = (
    "run_stageb_confidence_adapter_candidate_split_global_trust_veto_"
    "probe_evaluation"
)
MAIN_SOURCE = REPO_ROOT / "main.py"
ProbeEvaluationError = _CORE.ProbeEvaluationError


_BASE.training = training
_BASE.health = health
_BASE.EXPECTED_UPDATES = EXPECTED_UPDATES
_CORE.training = training
_CORE.EXPECTED_UPDATES = EXPECTED_UPDATES
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
    setattr(_CORE, _name, globals()[_name])
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
_BASE_POSTFLIGHT = _BASE.postflight


def _v49_postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    result = dict(_BASE_POSTFLIGHT(preflight_report, summary_path=summary_path))
    contracts = result.get("contracts")
    if (
        not isinstance(contracts, Mapping)
        or contracts.get("terminal_u400_diagnostic") is not True
        or contracts.get("v47_split_boundary_routing_v29") is not True
    ):
        raise ProbeEvaluationError(
            "V49 postflight lacks the inherited terminal-U400 diagnostic contract"
        )
    contracts = dict(contracts)
    del contracts["v47_split_boundary_routing_v29"]
    contracts.update(
        {
            "v49_split_global_trust_veto_v31": True,
            "all_mean_v1": True,
            "split_token_veto_global_trust_veto_v4": True,
            "formal_admission_requires_separate_main_binding": True,
        }
    )
    result["contracts"] = contracts
    return result


_BASE.postflight = _v49_postflight
_CORE.postflight = _v49_postflight


def _formal_main_admission_is_wired(path: Path | None = None) -> bool:
    """Prove that main's admission dispatcher owns an exact V49 branch."""
    source = MAIN_SOURCE if path is None else Path(path)
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return False
    binder = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_bind_stage_b_confidence_probe_admission"
        ),
        None,
    )
    if binder is None:
        return False
    strings = {
        node.value
        for node in ast.walk(binder)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    tool_imports = {
        alias.name
        for node in ast.walk(binder)
        if isinstance(node, ast.ImportFrom) and node.module == "tools"
        for alias in node.names
    }
    return {
        EXPECTED_REVISION,
        EXPECTED_HEAD_CONTRACT,
        EXPECTED_ROUTING_REDUCTION,
        EXPECTED_TRUST_REDUCTION,
        EXPECTED_NEGATIVE_REDUCTION,
        FORMAL_ADMISSION_CONTRACT,
    }.issubset(strings) and CONTROLLER_IMPORT in tool_imports


def build_command() -> list[str]:
    return _CORE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    return _CORE.preflight(health_audit=health_audit)


def postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    return _v49_postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(
    path: Path | None = None, *, health_audit=None
) -> dict[str, Any]:
    if not _formal_main_admission_is_wired():
        raise ProbeEvaluationError(
            "V49 strict1607 evidence cannot promote formal training until "
            "main._bind_stage_b_confidence_probe_admission has the exact V49 "
            "split-global-trust/veto contract and this controller verifier"
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
