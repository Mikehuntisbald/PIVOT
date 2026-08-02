#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run, audit, and promote the v34 U400 exact-q05 strict1607 probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_confidence_adapter_candidate_q05_probe_u0400 as training

_BASE_PATH = REPO_ROOT / "tools/run_stageb_confidence_adapter_veto_probe_evaluation.py"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_candidate_q05_probe_evaluation_base",
    _BASE_PATH,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load probe evaluator controller: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

_HEALTH_PATH = REPO_ROOT / "tools/audit_stageb_confidence_adapter_veto_probe_health.py"
_HEALTH_SPEC = importlib.util.spec_from_file_location(
    "_pivot_stageb_candidate_q05_probe_health",
    _HEALTH_PATH,
)
if _HEALTH_SPEC is None or _HEALTH_SPEC.loader is None:
    raise RuntimeError(f"cannot load probe health audit: {_HEALTH_PATH}")
health = importlib.util.module_from_spec(_HEALTH_SPEC)
_HEALTH_SPEC.loader.exec_module(health)


def _q05_health_checks(trajectory, endpoint, runtime):
    """Health checks aligned with v34's q05 carrier, not legacy mean delta."""

    def finite(value):
        return isinstance(value, (int, float)) and value == value and abs(value) < float("inf")

    checks = {
        "u222_token_below_old_u444": {
            "passed": finite(trajectory.get("train_loss_fixed_text_token_unscaled"))
            and trajectory["train_loss_fixed_text_token_unscaled"]
            < health.OLD_U444_TOKEN_LOSS,
            "observed": trajectory.get("train_loss_fixed_text_token_unscaled"),
            "requirement": f"< {health.OLD_U444_TOKEN_LOSS}",
        },
        "u222_q05_gradient_path_healthy": {
            "passed": finite(
                trajectory.get(
                    "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled"
                )
            )
            and trajectory[
                "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled"
            ]
            <= 0.4,
            "observed": trajectory.get(
                "train_fixed_text_tail_queue_positive_trust_violation_rate_unscaled"
            ),
            "requirement": "<= 0.4",
        },
        "u222_tail_queue_ready": {
            "passed": finite(
                trajectory.get("train_fixed_text_tail_queue_threshold_valid_unscaled")
            )
            and trajectory["train_fixed_text_tail_queue_threshold_valid_unscaled"]
            >= 0.9,
            "observed": trajectory.get(
                "train_fixed_text_tail_queue_threshold_valid_unscaled"
            ),
            "requirement": ">= 0.9",
        },
        "u300_positive_q05_in_operating_range": {
            "passed": finite(endpoint.get("positive_q05"))
            and endpoint["positive_q05"] > -0.1,
            "observed": endpoint.get("positive_q05"),
            "requirement": "> -0.1",
        },
        "u300_tn_q95_in_operating_range": {
            "passed": finite(endpoint.get("tn_q95"))
            and endpoint["tn_q95"] < 0.1,
            "observed": endpoint.get("tn_q95"),
            "requirement": "< 0.1",
        },
        "u300_operating_gap_in_operating_range": {
            "passed": finite(endpoint.get("operating_gap"))
            and endpoint["operating_gap"] > -0.2,
            "observed": endpoint.get("operating_gap"),
            "requirement": "> -0.2",
        },
        "runtime_all_boundaries_succeeded": {
            "passed": runtime.get("optimizer_step_boundaries")
            == runtime.get("successful_optimizer_steps")
            == EXPECTED_UPDATES,
            "observed": {
                "boundaries": runtime.get("optimizer_step_boundaries"),
                "successful": runtime.get("successful_optimizer_steps"),
            },
            "requirement": f"boundaries == successful == {EXPECTED_UPDATES}",
        },
        "runtime_amp_skips_zero": {
            "passed": runtime.get("amp_skipped_optimizer_steps") == 0,
            "observed": runtime.get("amp_skipped_optimizer_steps"),
            "requirement": "== 0",
        },
        "runtime_nonfinite_gradients_zero": {
            "passed": runtime.get("nonfinite_gradient_boundaries") == 0,
            "observed": runtime.get("nonfinite_gradient_boundaries"),
            "requirement": "== 0",
        },
        "runtime_zero_gradient_steps_zero": {
            "passed": runtime.get("zero_gradient_successful_steps") == 0,
            "observed": runtime.get("zero_gradient_successful_steps"),
            "requirement": "== 0",
        },
        "runtime_amp_scale_floor": {
            "passed": finite(runtime.get("min_amp_scale"))
            and runtime["min_amp_scale"] >= 256.0,
            "observed": runtime.get("min_amp_scale"),
            "requirement": ">= 256",
        },
    }
    return checks


health._health_checks = _q05_health_checks


CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_q05_probe_u0400_20260801.py"
)
FORMAL_CONFIG = REPO_ROOT / (
    "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_q05_20260801.py"
)
CHECKPOINT = training.CHECKPOINT
FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python3.11")
OUTPUT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_q05_highmem_20260801/"
    "probe_evaluation/u000400_strict1607"
)
REPORT = OUTPUT.parent / "u000400_strict1607_report.json"
LOG = OUTPUT.parent / "u000400_strict1607_console.log"
FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_candidate_q05_confidence_strict1607_v34"
)
EXPECTED_UPDATES = 400

_BASE.training = training
_BASE.EXPECTED_UPDATES = EXPECTED_UPDATES
health.probe = training
health.EXPECTED_UPDATES = EXPECTED_UPDATES
for _name in (
    "CONFIG",
    "FORMAL_CONFIG",
    "CHECKPOINT",
    "FIXED_PYTHON",
    "OUTPUT",
    "REPORT",
    "LOG",
):
    setattr(_BASE, _name, globals()[_name])
_BASE.FORMAL_PROMOTION_OVERRIDES = {
    "epochs": (2, 24),
    "stage_b_dense_duty_confidence_expected_optimizer_updates": (400, 4412),
    "stage_b_dense_duty_evaluation_scope": ("probe", "formal"),
    "stage_b_dense_duty_execution_scope": ("probe", "formal"),
    "stage_b_dense_duty_confidence_probe_admission_contract": (
        "disabled_for_probe_v1",
        FORMAL_ADMISSION_CONTRACT,
    ),
    "stage_b_dense_duty_confidence_probe_admission_report": (
        "",
        str(REPORT),
    ),
}
_BASE._load_health_audit = lambda: health.audit


def verify_admission_report(path: Path | None = None):
    return _BASE.verify_admission_report(path)


def main(argv: Sequence[str] | None = None) -> int:
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
