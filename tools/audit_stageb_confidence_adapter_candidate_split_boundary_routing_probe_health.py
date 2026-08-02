#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed mechanical health audit for the V47 terminal U400 probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_candidate_"
    "split_positive_tail_probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_split_boundary_routing_probe_health_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
_CORE = _BASE._BASE

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_split_boundary_routing_probe_u0400 as training,
)


SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_split_boundary_routing_"
    "probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v29"
EXPECTED_REVISION = "word_veto_candidate_split_boundary_routing_v47"
EXPECTED_HEAD_CONTRACT = "split_token_veto_global_absolute_v2"
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_boundary_routing_probe_u0400_20260801.py"
)
LOG_PATH = Path(training.OUTPUT) / "log.txt"

for _name in (
    "SCHEMA",
    "TRAINING_CONTRACT_SCHEMA",
    "EXPECTED_REVISION",
    "EXPECTED_HEAD_CONTRACT",
    "EXPECTED_CONFIG_ENTRY",
    "LOG_PATH",
):
    setattr(_BASE, _name, globals()[_name])
    setattr(_CORE, _name, globals()[_name])
_BASE.training = training
_CORE.training = training
_CORE.EXPECTED_CONTRACT_VALUES = {
    **_CORE.EXPECTED_CONTRACT_VALUES,
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_confidence_head_gradient_contract": (
        EXPECTED_HEAD_CONTRACT
    ),
    "stage_b_dense_duty_deployed_veto_routing_weight": 0.1,
    "stage_b_dense_duty_deployed_veto_routing_reduction_contract": (
        "balanced_top_quarter_cvar_v2"
    ),
    "stage_b_v15_tail_queue_positive_trust_reduction_contract": (
        "top_quarter_cvar_v2"
    ),
}

ProbeHealthEvidenceError = _BASE.ProbeHealthEvidenceError


def _health_checks(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return _BASE._health_checks(runtime, trajectory)


_CORE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_split_boundary_routing_health_audit.json"
)


def audit() -> dict[str, Any]:
    return _CORE.audit()


def run(argv: Sequence[str] | None = None) -> int:
    return _CORE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
