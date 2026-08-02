#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the fixed V62 terminal-U400 strict1607 veto-only gate."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (  # noqa: E402
    run_stageb_confidence_full_decoder_verifier_probe_evaluation as v61,
)
from tools import (  # noqa: E402
    audit_stageb_confidence_full_decoder_patch_softmin_veto_probe_health as health,
)
from tools import (  # noqa: E402
    run_stageb_confidence_full_decoder_patch_softmin_veto_probe_u0400 as training,
)


SCHEMA = "pivot.stageb.confidence_full_decoder_patch_softmin_veto_probe_evaluation/v1"
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_full_decoder_patch_softmin_veto_"
    "probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_full_decoder_patch_softmin_veto_formal_admission/v1"
)
HEALTH_SCHEMA = health.SCHEMA
CONFIG = training.CONFIG
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_full_decoder_patch_softmin_veto_"
    "formal_20260803.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_full_decoder_patch_softmin_veto_20260803/"
    "probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
EXPECTED_UPDATES = 400
EXPECTED_TRAINABLE_PARAMETERS = 25_530_881
EXPECTED_CAPACITY_CONTRACT = "rank_cloned_full_decoder_patch_softmin_veto_v2"
FORMAL_ADMISSION_CONTRACT = (
    "u400_full_decoder_patch_softmin_veto_strict1607_v62"
)
CONTROLLER_IMPORT = (
    "run_stageb_confidence_full_decoder_patch_softmin_veto_probe_evaluation"
)
MAIN_SOURCE = REPO_ROOT / "main.py"
ProbeEvaluationError = v61.ProbeEvaluationError

_CORE = v61._CORE
_MODULES = (
    v61,
    v61._V60,
    v61._V60._V59,
    v61._V60._V59._V56,
    v61._V60._V59._V55,
    v61._V60._V59._V54,
    v61._V60._V59._V53,
    v61._V60._V59._BASE,
    _CORE,
)
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
for _module in _MODULES:
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


def _v62_postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    result = dict(v61._v61_postflight(preflight_report, summary_path=summary_path))
    contracts = result.get("contracts")
    if not isinstance(contracts, Mapping) or contracts.get(
        "v61_rank_cloned_full_decoder_verifier_v26"
    ) is not True:
        raise ProbeEvaluationError("V62 postflight lacks inherited verifier evidence")
    # Do not retain inherited V60/V61 architectural prose: those controllers
    # describe a deployed absolute pool, which is deliberately absent in V62.
    result["contracts"] = {
        "tn_only": True,
        "full_strict1607": True,
        "zero_invalid_records": True,
        "terminal_u400_diagnostic": True,
        "full_per_example_records_bound": True,
        "fpr95_replayed_as_exact_integer_count": True,
        "still_not_formal_evaluation": True,
        "v62_patch_softmin_veto_only_migration_v27": True,
        EXPECTED_CAPACITY_CONTRACT: True,
        "rank_tower_is_frozen_and_parameter_disjoint": True,
        "verifier_is_six_layer_256d_full_expression_tower": True,
        "verifier_outputs_token_entailment_and_nonnegative_veto_only": True,
        "verifier_has_no_free_signed_absolute_score": True,
        "patch_top50_is_detached_existential_weight_only": True,
        "absolute_confidence_pool_is_frozen_dormant_and_zero": True,
        "deployed_global_logit_is_exactly_negative_veto": True,
        "veto_is_nonnegative_patch_weighted_existential_softmin": True,
        "no_pool_veto_compensation_coordinate": True,
    }
    return result


v61._V60._V59._BASE.postflight = _v62_postflight
_CORE.postflight = _v62_postflight


def build_command() -> list[str]:
    return _CORE.build_command()


def preflight(*, health_audit=None) -> dict[str, Any]:
    return _CORE.preflight(health_audit=health_audit)


def postflight(
    preflight_report: Mapping[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    return _v62_postflight(preflight_report, summary_path=summary_path)


def verify_admission_report(
    path: Path | None = None, *, health_audit=None
) -> dict[str, Any]:
    return _CORE.verify_admission_report(path, health_audit=health_audit)


def run() -> int:
    return _CORE.run()


def status():
    return _CORE.status()


def main(argv: Sequence[str] | None = None) -> int:
    return _CORE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
