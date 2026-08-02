#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the fixed V55 terminal-U400 strict1607 diagnostic."""

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
    audit_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_health
    as health,
)
from tools import (  # noqa: E402
    run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_u0400
    as training,
)


_BASE_PATH = REPO_ROOT / (
    "tools/run_stageb_confidence_adapter_fulltext_global_absolute_"
    "exact_residual_probe_evaluation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_fulltext_global_independent_absolute_probe_evaluation_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe evaluator controller: {_BASE_PATH}")
_V54 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V54)
_V53 = _V54._V53
_BASE = _V54._BASE
_CORE = _V54._CORE


SCHEMA = (
    "pivot.stageb.confidence_adapter_fulltext_global_independent_absolute_"
    "probe_evaluation/v1"
)
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_adapter_fulltext_global_independent_absolute_"
    "probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_adapter_fulltext_global_independent_absolute_"
    "formal_admission/v1"
)
HEALTH_SCHEMA = health.SCHEMA
CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_fulltext_global_"
    "independent_absolute_20260802.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_fulltext_global_independent_absolute_highmem_"
    "20260802/probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
EXPECTED_UPDATES = 400
EXPECTED_REVISION = (
    "word_veto_rank_full_expression_global_independent_absolute_v55"
)
EXPECTED_HEAD_CONTRACT = "split_token_veto_local_candidate_global_absolute_v8"
EXPECTED_POOL_CONTRACT = (
    "detached_rank_full_expression_local_candidate_"
    "frozen_rank_global_pool_v12"
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
EXPECTED_POSITIVE_TRUST_CONTRACT = "absolute_global_pool_logit_v4"
EXPECTED_TRAINABLE_PARAMETERS = 534_725
FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_global_independent_absolute_"
    "confidence_strict1607_v55"
)
CONTROLLER_IMPORT = (
    "run_stageb_confidence_adapter_fulltext_global_independent_absolute_"
    "probe_evaluation"
)
MAIN_SOURCE = REPO_ROOT / "main.py"
BASELINE_FALSE_ACCEPTS = _CORE.BASELINE_FALSE_ACCEPTS
MAX_ADMITTED_FALSE_ACCEPTS = _CORE.MAX_ADMITTED_FALSE_ACCEPTS
ProbeEvaluationError = _CORE.ProbeEvaluationError


_V55_OVERRIDES = {
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
    "EXPECTED_POSITIVE_TRUST_CONTRACT": EXPECTED_POSITIVE_TRUST_CONTRACT,
    "EXPECTED_TRAINABLE_PARAMETERS": EXPECTED_TRAINABLE_PARAMETERS,
    "FORMAL_ADMISSION_CONTRACT": FORMAL_ADMISSION_CONTRACT,
    "CONTROLLER_IMPORT": CONTROLLER_IMPORT,
    "MAIN_SOURCE": MAIN_SOURCE,
    "training": training,
    "health": health,
}
for _module in (_V54, _V53):
    for _name, _value in _V55_OVERRIDES.items():
        setattr(_module, _name, _value)
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
_BASE_POSTFLIGHT = _V54._v54_postflight


def _v55_postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    result = dict(_BASE_POSTFLIGHT(preflight_report, summary_path=summary_path))
    contracts = result.get("contracts")
    if (
        not isinstance(contracts, Mapping)
        or contracts.get("terminal_u400_diagnostic") is not True
        or contracts.get(
            "v54_rank_full_expression_global_absolute_exact_residual_v36"
        )
        is not True
    ):
        raise ProbeEvaluationError(
            "V55 postflight lacks the inherited terminal-U400 diagnostic contract"
        )
    contracts = dict(contracts)
    for stale in (
        "v54_rank_full_expression_global_absolute_exact_residual_v36",
        "deployed_global_confidence_remains_absolute",
        "positive_tail_trust_uses_exact_frozen_rank_max_residual",
        "tn_pair_queue_and_inference_remain_absolute",
        "same_v53_data_loss_update_parameter_surface",
        "candidate_and_sample_losses_share_live_global_owner",
        "frozen_rank_full_expression_u0_carrier",
    ):
        contracts.pop(stale, None)
    contracts.update(
        {
            "v55_rank_full_expression_global_independent_absolute_v37": True,
            EXPECTED_HEAD_CONTRACT: True,
            EXPECTED_POOL_CONTRACT: True,
            EXPECTED_GATE_CONTRACT: True,
            EXPECTED_POSITIVE_TRUST_CONTRACT: True,
            "deployed_global_confidence_is_pool_absolute_only": True,
            "local_candidate_logits_excluded_from_deployed_global": True,
            "rank_logits_are_detached_pool_inputs_and_reference_only": True,
            "global_tn_pair_queue_and_inference_use_pool_absolute_only": True,
            "local_candidate_and_global_pool_logits_are_distinct": True,
            "same_v53_v54_data_loss_update_parameter_surface": True,
            "formal_admission_requires_separate_main_binding": True,
        }
    )
    result["contracts"] = contracts
    return result


_BASE.postflight = _v55_postflight
_CORE.postflight = _v55_postflight


def _formal_main_admission_is_wired(path: Path | None = None) -> bool:
    """Prove that main's admission dispatcher owns one exact V55 branch."""
    return _V54._formal_main_admission_is_wired(path)


def build_command() -> list[str]:
    return _CORE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    return _CORE.preflight(health_audit=health_audit)


def postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    return _v55_postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(
    path: Path | None = None, *, health_audit=None
) -> dict[str, Any]:
    if not _formal_main_admission_is_wired():
        raise ProbeEvaluationError(
            "V55 strict1607 evidence cannot promote formal training until main "
            "has the exact V55 independent-pool/trust contract and this verifier"
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
