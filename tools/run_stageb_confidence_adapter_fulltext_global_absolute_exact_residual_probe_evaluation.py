#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the fixed V54 terminal-U400 strict1607 diagnostic."""

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
    audit_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_probe_health
    as health,
)
from tools import (  # noqa: E402
    run_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_probe_u0400
    as training,
)


_BASE_PATH = REPO_ROOT / (
    "tools/run_stageb_confidence_adapter_fulltext_global_absolute_"
    "probe_evaluation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_fulltext_global_absolute_exact_residual_probe_evaluation_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe evaluator controller: {_BASE_PATH}")
_V53 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V53)
_BASE = _V53._BASE
_CORE = _V53._CORE


SCHEMA = (
    "pivot.stageb.confidence_adapter_fulltext_global_absolute_exact_residual_"
    "probe_evaluation/v1"
)
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_adapter_fulltext_global_absolute_exact_residual_"
    "probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_adapter_fulltext_global_absolute_exact_residual_"
    "formal_admission/v1"
)
HEALTH_SCHEMA = health.SCHEMA
CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "exact_residual_20260802.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_fulltext_global_absolute_exact_residual_highmem_"
    "20260802/probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
EXPECTED_UPDATES = 400
EXPECTED_REVISION = (
    "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
)
EXPECTED_HEAD_CONTRACT = "split_token_veto_fulltext_global_absolute_v7"
EXPECTED_POOL_CONTRACT = (
    "detached_rank_full_expression_candidate_residual_global_pool_"
    "exact_rank_max_reference_v11"
)
EXPECTED_GATE_CONTRACT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
EXPECTED_AGGREGATION = "trace_activated_word_veto_gated_pool_absolute_cap_v5"
EXPECTED_ROUTING_WEIGHT = 0.0
EXPECTED_ROUTING_REDUCTION = "balanced_top_quarter_cvar_v2"
EXPECTED_TRUST_REDUCTION = "top_quarter_cvar_v2"
EXPECTED_NEGATIVE_REDUCTION = "all_mean_v1"
EXPECTED_TOKEN_EDIT_QUERY_SCOPE = "target_iou_v1"
EXPECTED_POSITIVE_GRADIENT_CONTRACT = (
    "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
)
EXPECTED_POSITIVE_TRUST_CONTRACT = (
    "exact_frozen_rank_max_confidence_delta_v3"
)
EXPECTED_TRAINABLE_PARAMETERS = 534_725
FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_global_absolute_exact_residual_"
    "confidence_strict1607_v54"
)
CONTROLLER_IMPORT = (
    "run_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_"
    "probe_evaluation"
)
MAIN_SOURCE = REPO_ROOT / "main.py"
BASELINE_FALSE_ACCEPTS = _CORE.BASELINE_FALSE_ACCEPTS
MAX_ADMITTED_FALSE_ACCEPTS = _CORE.MAX_ADMITTED_FALSE_ACCEPTS
ProbeEvaluationError = _CORE.ProbeEvaluationError


_V54_OVERRIDES = {
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
    "EXPECTED_HEAD_CONTRACT": EXPECTED_HEAD_CONTRACT,
    "EXPECTED_POOL_CONTRACT": EXPECTED_POOL_CONTRACT,
    "EXPECTED_GATE_CONTRACT": EXPECTED_GATE_CONTRACT,
    "EXPECTED_AGGREGATION": EXPECTED_AGGREGATION,
    "EXPECTED_ROUTING_WEIGHT": EXPECTED_ROUTING_WEIGHT,
    "EXPECTED_ROUTING_REDUCTION": EXPECTED_ROUTING_REDUCTION,
    "EXPECTED_TRUST_REDUCTION": EXPECTED_TRUST_REDUCTION,
    "EXPECTED_NEGATIVE_REDUCTION": EXPECTED_NEGATIVE_REDUCTION,
    "EXPECTED_TOKEN_EDIT_QUERY_SCOPE": EXPECTED_TOKEN_EDIT_QUERY_SCOPE,
    "EXPECTED_POSITIVE_GRADIENT_CONTRACT": (
        EXPECTED_POSITIVE_GRADIENT_CONTRACT
    ),
    "EXPECTED_TRAINABLE_PARAMETERS": EXPECTED_TRAINABLE_PARAMETERS,
    "FORMAL_ADMISSION_CONTRACT": FORMAL_ADMISSION_CONTRACT,
    "CONTROLLER_IMPORT": CONTROLLER_IMPORT,
    "MAIN_SOURCE": MAIN_SOURCE,
    "training": training,
    "health": health,
}
for _name, _value in _V54_OVERRIDES.items():
    setattr(_V53, _name, _value)
for _module in (_BASE, _CORE):
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
        setattr(_module, _name, globals()[_name])
    _module.training = training
    _module.health = health
    _module.EXPECTED_UPDATES = EXPECTED_UPDATES

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
_BASE_POSTFLIGHT = _V53._v53_postflight


def _v54_postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    result = dict(_BASE_POSTFLIGHT(preflight_report, summary_path=summary_path))
    contracts = result.get("contracts")
    if (
        not isinstance(contracts, Mapping)
        or contracts.get("terminal_u400_diagnostic") is not True
        or contracts.get("v53_rank_full_expression_global_absolute_v35") is not True
    ):
        raise ProbeEvaluationError(
            "V54 postflight lacks the inherited terminal-U400 diagnostic contract"
        )
    contracts = dict(contracts)
    contracts.pop("v53_rank_full_expression_global_absolute_v35", None)
    contracts.update(
        {
            "v54_rank_full_expression_global_absolute_exact_residual_v36": True,
            EXPECTED_HEAD_CONTRACT: True,
            EXPECTED_POOL_CONTRACT: True,
            EXPECTED_GATE_CONTRACT: True,
            EXPECTED_POSITIVE_TRUST_CONTRACT: True,
            "deployed_global_confidence_remains_absolute": True,
            "positive_tail_trust_uses_exact_frozen_rank_max_residual": True,
            "tn_pair_queue_and_inference_remain_absolute": True,
            "same_v53_data_loss_update_parameter_surface": True,
            "formal_admission_requires_separate_main_binding": True,
        }
    )
    result["contracts"] = contracts
    return result


_BASE.postflight = _v54_postflight
_CORE.postflight = _v54_postflight


def _string_constants(node: ast.AST) -> set[str]:
    return {
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def _has_exact_numeric_field_equality(
    test: ast.AST, *, field: str, expected: float
) -> bool:
    for node in ast.walk(test):
        if (
            not isinstance(node, ast.Compare)
            or len(node.ops) != 1
            or not isinstance(node.ops[0], ast.Eq)
            or len(node.comparators) != 1
        ):
            continue
        comparator = node.comparators[0]
        if (
            isinstance(comparator, ast.Constant)
            and not isinstance(comparator.value, bool)
            and isinstance(comparator.value, (int, float))
            and float(comparator.value) == float(expected)
            and field in _string_constants(node.left)
        ):
            return True
    return False


def _formal_main_admission_is_wired(path: Path | None = None) -> bool:
    """Prove that main's admission dispatcher owns one exact V54 branch."""
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
    candidates = [
        node
        for node in ast.walk(binder)
        if isinstance(node, ast.If)
        and EXPECTED_REVISION in _string_constants(node.test)
    ]
    if len(candidates) != 1:
        return False
    branch = candidates[0]
    required_test_strings = {
        EXPECTED_AGGREGATION,
        EXPECTED_REVISION,
        EXPECTED_HEAD_CONTRACT,
        EXPECTED_POOL_CONTRACT,
        EXPECTED_GATE_CONTRACT,
        EXPECTED_ROUTING_REDUCTION,
        EXPECTED_TRUST_REDUCTION,
        EXPECTED_NEGATIVE_REDUCTION,
        EXPECTED_TOKEN_EDIT_QUERY_SCOPE,
        EXPECTED_POSITIVE_GRADIENT_CONTRACT,
        EXPECTED_POSITIVE_TRUST_CONTRACT,
        "bidirectional_v1",
    }
    if not required_test_strings.issubset(_string_constants(branch.test)):
        return False
    for field, expected in (
        ("stage_b_dense_duty_deployed_veto_routing_weight", 0.0),
        ("stage_b_dense_duty_deployed_veto_positive_max", 0.1),
        ("stage_b_dense_duty_deployed_veto_tn_min", 0.9),
        ("stage_b_dense_duty_confidence_veto_gate_offset", 0.0),
    ):
        if not _has_exact_numeric_field_equality(
            branch.test, field=field, expected=expected
        ):
            return False
    body_module = ast.Module(body=branch.body, type_ignores=[])
    body_strings = _string_constants(body_module)
    tool_imports = {
        alias.name
        for node in ast.walk(body_module)
        if isinstance(node, ast.ImportFrom) and node.module == "tools"
        for alias in node.names
    }
    return (
        FORMAL_ADMISSION_CONTRACT in body_strings
        and CONTROLLER_IMPORT in tool_imports
    )


def build_command() -> list[str]:
    return _CORE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    return _CORE.preflight(health_audit=health_audit)


def postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    return _v54_postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(
    path: Path | None = None, *, health_audit=None
) -> dict[str, Any]:
    if not _formal_main_admission_is_wired():
        raise ProbeEvaluationError(
            "V54 strict1607 evidence cannot promote formal training until main "
            "has the exact V54 revision/pool/exact-residual-trust contract and "
            "this verifier"
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
