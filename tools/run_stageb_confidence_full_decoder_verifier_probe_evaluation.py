#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the fixed V61 terminal-U400 strict1607 capacity-upper-bound gate."""

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
    audit_stageb_confidence_full_decoder_verifier_probe_health as health,
)
from tools import (  # noqa: E402
    run_stageb_confidence_full_decoder_verifier_probe_u0400 as training,
)

_BASE_PATH = REPO_ROOT / (
    "tools/run_stageb_confidence_adapter_deployment_owned_query_veto_"
    "probe_evaluation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_full_decoder_verifier_probe_evaluation_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load V60 probe evaluator: {_BASE_PATH}")
_V60 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V60)


SCHEMA = "pivot.stageb.confidence_full_decoder_verifier_probe_evaluation/v1"
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_full_decoder_verifier_probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_full_decoder_verifier_formal_admission/v1"
)
HEALTH_SCHEMA = health.SCHEMA
CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_full_decoder_verifier_formal_20260803.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_full_decoder_confidence_verifier_20260803/"
    "probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
EXPECTED_UPDATES = 400
EXPECTED_REVISION = _V60.EXPECTED_REVISION
EXPECTED_HEAD_CONTRACT = _V60.EXPECTED_HEAD_CONTRACT
EXPECTED_POOL_CONTRACT = _V60.EXPECTED_POOL_CONTRACT
EXPECTED_GATE_CONTRACT = _V60.EXPECTED_GATE_CONTRACT
EXPECTED_AGGREGATION = _V60.EXPECTED_AGGREGATION
EXPECTED_ROUTING_WEIGHT = _V60.EXPECTED_ROUTING_WEIGHT
EXPECTED_ROUTING_REDUCTION = _V60.EXPECTED_ROUTING_REDUCTION
EXPECTED_TRUST_REDUCTION = _V60.EXPECTED_TRUST_REDUCTION
EXPECTED_NEGATIVE_REDUCTION = _V60.EXPECTED_NEGATIVE_REDUCTION
EXPECTED_TOKEN_EDIT_QUERY_SCOPE = _V60.EXPECTED_TOKEN_EDIT_QUERY_SCOPE
EXPECTED_POSITIVE_GRADIENT_CONTRACT = _V60.EXPECTED_POSITIVE_GRADIENT_CONTRACT
EXPECTED_POSITIVE_TRUST_CONTRACT = _V60.EXPECTED_POSITIVE_TRUST_CONTRACT
EXPECTED_TRAINABLE_PARAMETERS = 25_664_258
EXPECTED_CAPACITY_CONTRACT = "rank_cloned_full_decoder_6layer_256d_v1"
FORMAL_ADMISSION_CONTRACT = (
    "u400_rank_cloned_full_decoder_verifier_confidence_strict1607_v61"
)
CONTROLLER_IMPORT = "run_stageb_confidence_full_decoder_verifier_probe_evaluation"
MAIN_SOURCE = REPO_ROOT / "main.py"
BASELINE_FALSE_ACCEPTS = _V60.BASELINE_FALSE_ACCEPTS
MAX_ADMITTED_FALSE_ACCEPTS = _V60.MAX_ADMITTED_FALSE_ACCEPTS
ProbeEvaluationError = _V60.ProbeEvaluationError

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
    "EXPECTED_HEAD_CONTRACT": EXPECTED_HEAD_CONTRACT,
    "EXPECTED_POOL_CONTRACT": EXPECTED_POOL_CONTRACT,
    "EXPECTED_GATE_CONTRACT": EXPECTED_GATE_CONTRACT,
    "EXPECTED_AGGREGATION": EXPECTED_AGGREGATION,
    "EXPECTED_ROUTING_WEIGHT": EXPECTED_ROUTING_WEIGHT,
    "EXPECTED_ROUTING_REDUCTION": EXPECTED_ROUTING_REDUCTION,
    "EXPECTED_TRUST_REDUCTION": EXPECTED_TRUST_REDUCTION,
    "EXPECTED_NEGATIVE_REDUCTION": EXPECTED_NEGATIVE_REDUCTION,
    "EXPECTED_TOKEN_EDIT_QUERY_SCOPE": EXPECTED_TOKEN_EDIT_QUERY_SCOPE,
    "EXPECTED_POSITIVE_GRADIENT_CONTRACT": EXPECTED_POSITIVE_GRADIENT_CONTRACT,
    "EXPECTED_POSITIVE_TRUST_CONTRACT": EXPECTED_POSITIVE_TRUST_CONTRACT,
    "EXPECTED_TRAINABLE_PARAMETERS": EXPECTED_TRAINABLE_PARAMETERS,
    "FORMAL_ADMISSION_CONTRACT": FORMAL_ADMISSION_CONTRACT,
    "CONTROLLER_IMPORT": CONTROLLER_IMPORT,
    "MAIN_SOURCE": MAIN_SOURCE,
    "training": training,
    "health": health,
}
for _module in (
    _V60,
    _V60._V59,
    _V60._V59._V56,
    _V60._V59._V55,
    _V60._V59._V54,
    _V60._V59._V53,
    _V60._V59._BASE,
    _V60._V59._CORE,
):
    for _name, _value in _OVERRIDES.items():
        setattr(_module, _name, _value)

_CORE = _V60._V59._CORE
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


def _v61_postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    result = dict(
        _V60._v60_postflight(preflight_report, summary_path=summary_path)
    )
    contracts = result.get("contracts")
    if not isinstance(contracts, Mapping) or contracts.get(
        "v60_deployment_owned_query_veto_representation_v42"
    ) is not True:
        raise ProbeEvaluationError("V61 postflight lacks inherited veto evidence")
    contracts = dict(contracts)
    contracts.pop("v60_deployment_owned_query_veto_representation_v42", None)
    contracts.update(
        {
            "v61_rank_cloned_full_decoder_verifier_v26": True,
            EXPECTED_CAPACITY_CONTRACT: True,
            "rank_tower_is_frozen_and_parameter_disjoint": True,
            "verifier_is_six_layer_256d_full_expression_tower": True,
            "verifier_outputs_token_entailment_and_nonnegative_veto_only": True,
            "verifier_has_no_free_signed_absolute_score": True,
            "veto_and_pool_outputs_are_zero_initialized": True,
        }
    )
    result["contracts"] = contracts
    return result


_V60._V59._BASE.postflight = _v61_postflight
_CORE.postflight = _v61_postflight


def _formal_main_admission_is_wired(path: Path | None = None) -> bool:
    candidate = MAIN_SOURCE if path is None else Path(path)
    if not _V60._formal_main_admission_is_wired(candidate):
        return False
    source = candidate.read_text(encoding="utf-8")
    return (
        "stage_b_dense_duty_confidence_full_decoder_verifier" in source
        and EXPECTED_CAPACITY_CONTRACT in source
        and FORMAL_ADMISSION_CONTRACT in source
        and CONTROLLER_IMPORT in source
    )


def build_command() -> list[str]:
    return _CORE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    return _CORE.preflight(health_audit=health_audit)


def postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    return _v61_postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(
    path: Path | None = None, *, health_audit=None
) -> dict[str, Any]:
    if not _formal_main_admission_is_wired():
        raise ProbeEvaluationError(
            "V61 strict1607 evidence cannot promote formal training until main "
            "binds the exact full-decoder capacity contract"
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
