#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Fail-closed mechanical health audit for the V46 terminal U400 probe."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = REPO_ROOT / (
    "tools/audit_stageb_confidence_adapter_candidate_"
    "split_tail_aligned_probe_health.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_split_positive_tail_probe_health_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

from tools import (  # noqa: E402
    run_stageb_confidence_adapter_candidate_split_positive_tail_probe_u0400 as training,
)


SCHEMA = (
    "pivot.stageb.confidence_adapter_candidate_split_positive_tail_"
    "probe_health/v1"
)
TRAINING_CONTRACT_SCHEMA = "pivot.stageb.dense_duty_training_contract/v28"
EXPECTED_REVISION = "word_veto_candidate_split_positive_tail_v46"
EXPECTED_HEAD_CONTRACT = "split_token_veto_global_absolute_v2"
EXPECTED_CONFIG_ENTRY = (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_"
    "split_positive_tail_probe_u0400_20260801.py"
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
_BASE.training = training
_BASE.EXPECTED_CONTRACT_VALUES = {
    **_BASE.EXPECTED_CONTRACT_VALUES,
    "stage_b_dense_duty_confidence_revision": EXPECTED_REVISION,
    "stage_b_dense_duty_confidence_head_gradient_contract": (
        EXPECTED_HEAD_CONTRACT
    ),
    "stage_b_dense_duty_deployed_veto_routing_weight": 0.1,
    "stage_b_dense_duty_deployed_veto_routing_reduction_contract": (
        "balanced_mean_v1"
    ),
    "stage_b_v15_tail_queue_positive_trust_reduction_contract": (
        "top_quarter_cvar_v2"
    ),
}


def _health_checks(
    runtime: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = dict(_BASE_HEALTH_CHECKS(runtime, trajectory))
    checks.pop("u222_joint_clip_v3_evidence", None)
    value = lambda name: float(trajectory[name])
    token_post = value("train_grad_norm_dense_duty_token_veto_postclip")
    global_post = value("train_grad_norm_dense_duty_global_absolute_postclip")
    active_post = value("train_grad_norm_dense_duty_active_postclip")
    expected_active = math.hypot(token_post, global_post)
    checks["u222_independent_clip_v2_evidence"] = _BASE._check(
        {
            "active_preclip": value("train_grad_norm_dense_duty_active_preclip"),
            "token_preclip": value(
                "train_grad_norm_dense_duty_token_veto_preclip"
            ),
            "global_preclip": value(
                "train_grad_norm_dense_duty_global_absolute_preclip"
            ),
            "active_postclip": active_post,
            "token_postclip": token_post,
            "global_postclip": global_post,
        },
        "both disjoint owners independently clip to 0.1",
        value("train_grad_norm_dense_duty_token_veto_preclip")
        > _BASE.CLIP_MAX_NORM
        and value("train_grad_norm_dense_duty_global_absolute_preclip")
        > _BASE.CLIP_MAX_NORM
        and 0.099 <= token_post <= _BASE.CLIP_MAX_NORM + 1e-6
        and 0.099 <= global_post <= _BASE.CLIP_MAX_NORM + 1e-6
        and math.isclose(active_post, expected_active, rel_tol=0.0, abs_tol=1e-6),
    )
    return checks


_BASE_HEALTH_CHECKS = _BASE._health_checks
_BASE._health_checks = _health_checks
_BASE._default_output = lambda: Path(training.OUTPUT).parent / (
    "u000400_split_positive_tail_health_audit.json"
)

ProbeHealthEvidenceError = _BASE.ProbeHealthEvidenceError


def audit() -> dict[str, Any]:
    return _BASE.audit()


def run(argv: Sequence[str] | None = None) -> int:
    return _BASE.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
