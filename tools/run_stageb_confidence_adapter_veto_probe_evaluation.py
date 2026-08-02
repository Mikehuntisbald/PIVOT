#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run the fixed U300 word-veto strict1607 diagnostic and audit its result.

Exit codes:
  0: valid diagnostic with at most 800 FPR95 false accepts;
  1: valid diagnostic that does not beat the fixed 801/1607 baseline;
  2: invalid, incomplete, unhealthy, or unbound evidence.

This controller never emits a formal evaluation claim.  A passing result only
admits the architecture to a fresh formal-training run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_confidence_adapter_veto_probe as training  # noqa: E402


SCHEMA = "pivot.stageb.confidence_adapter_veto_probe_evaluation/v1"
POSTFLIGHT_SCHEMA = (
    "pivot.stageb.confidence_adapter_veto_probe_evaluation_postflight/v1"
)
ADMISSION_SCHEMA = (
    "pivot.stageb.confidence_adapter_veto_formal_admission/v1"
)
HEALTH_SCHEMA = "pivot.stageb.confidence_adapter_veto_probe_health/v1"
RECORD_SCHEMA = "stageb-eval-record-v1"

FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
EVALUATOR = REPO_ROOT / "tools/eval_text_groundingdino_refcoco_tn.py"
CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_probe_20260730.py"
)
FORMAL_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_20260730.py"
)
CHECKPOINT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_highmem_20260730/probe/u000300/checkpoint_iter.pth"
)
DATA_ROOT = Path("/media/haoyi/T9/data")
TN_MANIFEST = (
    REPO_ROOT
    / "data/eval_manifests/"
    "stageb_vlm_verified_strict_ann_umd_val_20260711/"
    "semantic_stageb_union_image_disjoint_manifest.jsonl"
)
TN_MANIFEST_SHA256 = (
    "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25"
)
DERIVED_TN_MANIFEST_SHA256 = (
    "3572e14206bb3b0ddbb14da0d4efe67cf218fa16ff1710fab6728184c107feca"
)
OUTPUT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_highmem_20260730/"
    "probe_evaluation/u000300_strict1607"
)
REPORT = OUTPUT.parent / "u000300_strict1607_report.json"
LOG = OUTPUT.parent / "u000300_strict1607_console.log"

EXPECTED_UPDATES = 300
EXPECTED_PAIRS = 1607
BASELINE_FALSE_ACCEPTS = 801
MAX_ADMITTED_FALSE_ACCEPTS = 800
EXPECTED_SPLITS = frozenset({"refcocop_val", "refcocog_umd_val"})
EXPECTED_SCORE_ROUTE = {
    "ref_score_key": "stage_b_v15_dense_rank_score",
    "tn_score_key": "stage_b_v7_final_score",
    "score_ownership": "rank_tower_stopgrad_token_adapter_two_phase",
}
FORMAL_PROMOTION_OVERRIDES = {
    "epochs": (2, 24),
    "stage_b_dense_duty_confidence_expected_optimizer_updates": (300, 4412),
    "stage_b_dense_duty_evaluation_scope": ("probe", "formal"),
    "stage_b_dense_duty_execution_scope": ("probe", "formal"),
    "stage_b_dense_duty_confidence_probe_admission_contract": (
        "disabled_for_probe_v1",
        "u300_word_veto_strict1607_v1",
    ),
    "stage_b_dense_duty_confidence_probe_admission_report": (
        "",
        str(REPORT),
    ),
}


class ProbeEvaluationError(RuntimeError):
    """The fixed probe-evaluation contract cannot be proven."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ProbeEvaluationError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ProbeEvaluationError(f"{label} is unavailable: {path}") from error
    if not resolved.is_file():
        raise ProbeEvaluationError(f"{label} is not a file: {resolved}")
    sha256 = _sha256(resolved)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ProbeEvaluationError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {sha256}"
        )
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256,
    }


def _record_matches(record: Any, observed: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(record, Mapping):
        raise ProbeEvaluationError(f"{label} file record is missing")
    try:
        record_path = Path(str(record.get("path", ""))).expanduser().resolve(
            strict=True
        )
        record_size = int(record.get("size_bytes"))
    except (OSError, TypeError, ValueError) as error:
        raise ProbeEvaluationError(f"{label} file record is malformed") from error
    if (
        record_path != Path(str(observed["path"]))
        or record_size != int(observed["size_bytes"])
        or record.get("sha256") != observed["sha256"]
    ):
        raise ProbeEvaluationError(f"{label} file record does not bind the artifact")


def _exact_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise ProbeEvaluationError(f"{label} must be an exact integer")
    return int(value)


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeEvaluationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProbeEvaluationError(f"{label} must be finite")
    return result


def _exact_binary_count(rate: Any, total: int, *, label: str) -> int:
    value = _finite_float(rate, label=label)
    if not 0.0 <= value <= 1.0 or total <= 0:
        raise ProbeEvaluationError(f"{label} must be a binary rate")
    count = int(round(value * total))
    if not math.isclose(value, count / total, rel_tol=0.0, abs_tol=1e-12):
        raise ProbeEvaluationError(
            f"{label} does not replay to an exact integer count over N={total}"
        )
    return count


def _exact_tpr_operating_threshold(
    scores: Sequence[float], *, target_tpr: float
) -> float:
    values = sorted(float(score) for score in scores)
    if (
        not values
        or not 0.0 < float(target_tpr) <= 1.0
        or not all(math.isfinite(value) for value in values)
    ):
        raise ProbeEvaluationError("positive scores cannot define a TPR threshold")
    accepted = max(1, int(math.ceil(float(target_tpr) * len(values))))
    return values[len(values) - accepted]


def build_command() -> list[str]:
    """Return the single fixed evaluator command; there are no runtime overrides."""
    return [
        str(FIXED_PYTHON),
        str(EVALUATOR),
        "--config",
        str(CONFIG),
        "--ckpts",
        str(CHECKPOINT),
        "--output_dir",
        str(OUTPUT),
        "--data_root",
        str(DATA_ROOT),
        "--device",
        "cuda:0",
        "--batch_size",
        "16",
        "--num_workers",
        "4",
        "--seed",
        "42",
        "--amp",
        "--skip_ref",
        "--tn_jsonl",
        str(TN_MANIFEST),
        "--tn_splits",
        "refcocop_val",
        "refcocog_umd_val",
        "--partial_dense_duty_confidence_diagnostic",
        "--topk",
        "1",
        "--threshold_tprs",
        "0.75",
        "0.9",
        "0.95",
        "--score_thresholds",
        "0.5",
        "--max_ref_batches",
        "0",
        "--max_tn_batches",
        "0",
        "--log_every",
        "50",
    ]


def _load_health_audit() -> Callable[[], Mapping[str, Any]]:
    try:
        module = importlib.import_module(
            "tools.audit_stageb_confidence_adapter_veto_probe_health"
        )
    except (ImportError, OSError) as error:
        raise ProbeEvaluationError(
            "the U300 loss-health audit is unavailable"
        ) from error
    audit = getattr(module, "audit", None)
    if not callable(audit):
        raise ProbeEvaluationError("the U300 loss-health audit has no audit() entry")
    return audit


def _validate_training_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ProbeEvaluationError("training inspect result must be a mapping")
    expected = {
        "status": "terminal",
        "action": "complete",
        "updates": EXPECTED_UPDATES,
    }
    for field, value in expected.items():
        if state.get(field) != value:
            raise ProbeEvaluationError(
                f"U300 training probe requires {field}={value!r}, "
                f"got {state.get(field)!r}"
            )
    rank_sha256 = state.get("rank_sha256")
    if (
        not isinstance(rank_sha256, str)
        or len(rank_sha256) != 64
        or any(character not in "0123456789abcdef" for character in rank_sha256)
    ):
        raise ProbeEvaluationError("U300 training probe has no valid rank SHA-256")
    return dict(state)


def _validate_health_report(
    report: Any,
    *,
    checkpoint_record: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, Mapping) or report.get("schema") != HEALTH_SCHEMA:
        raise ProbeEvaluationError("loss-health audit schema is invalid")
    if report.get("decision") != "healthy_for_strict1607_diagnostic":
        raise ProbeEvaluationError(
            "loss-health audit did not admit the strict1607 diagnostic"
        )
    if report.get("failed_checks") != []:
        raise ProbeEvaluationError("loss-health audit reports failed checks")
    checks = report.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(
            not isinstance(check, Mapping) or check.get("passed") is not True
            for check in checks.values()
        )
    ):
        raise ProbeEvaluationError("loss-health audit checks are incomplete")
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ProbeEvaluationError("loss-health audit has no candidate evidence")
    controller = candidate.get("controller")
    if not isinstance(controller, Mapping) or any(
        controller.get(field) != expected
        for field, expected in (
            ("status", "terminal"),
            ("action", "complete"),
            ("updates", EXPECTED_UPDATES),
        )
    ):
        raise ProbeEvaluationError("loss-health audit is not bound to terminal U300")
    _record_matches(
        candidate.get("checkpoint"), checkpoint_record, label="health checkpoint"
    )
    log_record = candidate.get("log")
    if not isinstance(log_record, Mapping):
        raise ProbeEvaluationError("loss-health audit has no candidate log identity")
    observed_log = _file_record(
        Path(str(log_record.get("path", ""))), label="health candidate log"
    )
    _record_matches(log_record, observed_log, label="health candidate log")
    return dict(report)


def _output_is_fresh() -> bool:
    if not OUTPUT.exists():
        return True
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        return False
    return not any(OUTPUT.iterdir())


def preflight(
    *,
    health_audit: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay terminal training and loss-health evidence before launching CUDA."""
    state = _validate_training_state(training.inspect())
    if not FIXED_PYTHON.is_file() or not os.access(FIXED_PYTHON, os.X_OK):
        raise ProbeEvaluationError(
            f"fixed gdino5090 Python is unavailable: {FIXED_PYTHON}"
        )
    if not DATA_ROOT.resolve(strict=True).is_dir():
        raise ProbeEvaluationError(f"fixed data root is unavailable: {DATA_ROOT}")
    inputs = {
        "python": _file_record(FIXED_PYTHON, label="fixed Python"),
        "evaluator": _file_record(EVALUATOR, label="fixed evaluator"),
        "config": _file_record(CONFIG, label="fixed probe config"),
        "checkpoint": _file_record(CHECKPOINT, label="terminal U300 checkpoint"),
        "tn_manifest": _file_record(
            TN_MANIFEST,
            label="fixed strict1607 source manifest",
            expected_sha256=TN_MANIFEST_SHA256,
        ),
    }
    audit = _load_health_audit() if health_audit is None else health_audit
    health = _validate_health_report(
        audit(), checkpoint_record=inputs["checkpoint"]
    )
    health_rank = health["candidate"]["controller"].get("rank_sha256")
    if health_rank is not None and health_rank != state["rank_sha256"]:
        raise ProbeEvaluationError("training and loss-health rank identities differ")
    if not _output_is_fresh():
        raise ProbeEvaluationError("fixed evaluation output is not absent or empty")
    if REPORT.exists() or LOG.exists():
        raise ProbeEvaluationError("fixed report/log destination is not fresh")
    return {
        "checked_at_utc": _utc_now(),
        "training": state,
        "health": health,
        "inputs": inputs,
        "command": build_command(),
        "diagnostic_only": True,
        "formal_gate_eligible": False,
    }


def _assert_inputs_unchanged(inputs: Any) -> None:
    if not isinstance(inputs, Mapping):
        raise ProbeEvaluationError("preflight input identities are missing")
    fixed = {
        "python": (FIXED_PYTHON, None),
        "evaluator": (EVALUATOR, None),
        "config": (CONFIG, None),
        "checkpoint": (CHECKPOINT, None),
        "tn_manifest": (TN_MANIFEST, TN_MANIFEST_SHA256),
    }
    for label, (path, expected_sha256) in fixed.items():
        observed = _file_record(
            path, label=f"postflight {label}", expected_sha256=expected_sha256
        )
        _record_matches(inputs.get(label), observed, label=f"preflight {label}")


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeEvaluationError(f"{label} is not readable JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ProbeEvaluationError(f"{label} must be a JSON object")
    return payload


def _summary_artifact_path(raw: Any, *, summary_path: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ProbeEvaluationError(f"summary has no {label} path")
    path = Path(raw).expanduser()
    candidates = (
        (path,)
        if path.is_absolute()
        else (REPO_ROOT / path, summary_path.parent / path)
    )
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if not existing:
        raise ProbeEvaluationError(f"summary {label} is unavailable: {raw}")
    expected_root = OUTPUT.resolve()
    for candidate in existing:
        if candidate.is_relative_to(expected_root):
            return candidate.resolve(strict=True)
    raise ProbeEvaluationError(f"summary {label} is outside the fixed output")


def _read_and_verify_records(
    path: Path,
    *,
    summary_row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    before = _file_record(path, label="strict1607 per-example records")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ProbeEvaluationError(f"could not read strict1607 records: {error}") from error
    if len(lines) != EXPECTED_PAIRS or any(not line.strip() for line in lines):
        raise ProbeEvaluationError(
            f"strict1607 records must contain exactly {EXPECTED_PAIRS} JSON lines"
        )
    rows: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    split_set: set[str] = set()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProbeEvaluationError(
                f"strict1607 record {index} is invalid JSON: {error}"
            ) from error
        if not isinstance(row, Mapping):
            raise ProbeEvaluationError(f"strict1607 record {index} is not an object")
        if (
            row.get("schema") != RECORD_SCHEMA
            or row.get("task") != "tn"
            or row.get("valid") is not True
            or row.get("run_id") != summary_row.get("run_id")
            or row.get("manifest_key") != "tn_global"
            or row.get("manifest_sha256") != DERIVED_TN_MANIFEST_SHA256
            or row.get("source_manifest_sha256") != TN_MANIFEST_SHA256
            or _exact_int(row.get("manifest_n"), label=f"record {index}.manifest_n")
            != EXPECTED_PAIRS
            or _exact_int(
                row.get("source_manifest_n"),
                label=f"record {index}.source_manifest_n",
            )
            != EXPECTED_PAIRS
            or _exact_int(
                row.get("manifest_index"), label=f"record {index}.manifest_index"
            )
            != index
        ):
            raise ProbeEvaluationError(
                f"strict1607 record {index} violates the manifest/provenance contract"
            )
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise ProbeEvaluationError(
                f"strict1607 record {index} has a missing/duplicate sample_id"
            )
        seen_ids.add(sample_id)
        split = row.get("split")
        if split not in EXPECTED_SPLITS:
            raise ProbeEvaluationError(
                f"strict1607 record {index} has unexpected split {split!r}"
            )
        split_set.add(str(split))
        _finite_float(row.get("pos_score"), label=f"record {index}.pos_score")
        _finite_float(row.get("neg_score"), label=f"record {index}.neg_score")
        rows.append(row)
    if split_set != EXPECTED_SPLITS:
        raise ProbeEvaluationError("strict1607 records do not cover both fixed splits")
    after = _file_record(path, label="strict1607 per-example records")
    if before != after:
        raise ProbeEvaluationError("strict1607 records changed while being audited")
    return {**before, "rows": len(rows)}, rows


def _validate_summary_provenance(
    row: Mapping[str, Any], *, inputs: Mapping[str, Any]
) -> None:
    expected = {
        "diagnostic_only": True,
        "formal_gate_eligible": False,
        "confidence_evaluated": True,
        "training_phase": "confidence",
        "terminal_checkpoint": True,
        "optimizer_updates": EXPECTED_UPDATES,
        "expected_optimizer_updates": EXPECTED_UPDATES,
        "remaining_optimizer_updates": 0,
        "checkpoint_reason": "max_train_iters",
        "amp": True,
        "device": "cuda:0",
        "batch_size": 16,
        "num_workers": 4,
        "seed": 42,
        "max_batches": 0,
        **EXPECTED_SCORE_ROUTE,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ProbeEvaluationError(
                f"summary provenance requires {field}={value!r}, got {row.get(field)!r}"
            )
    try:
        config = Path(str(row.get("config", ""))).expanduser().resolve(strict=True)
        checkpoint = Path(str(row.get("checkpoint", ""))).expanduser().resolve(
            strict=True
        )
        data_root = Path(str(row.get("data_root", ""))).expanduser().resolve(
            strict=True
        )
    except OSError as error:
        raise ProbeEvaluationError("summary provenance paths are unavailable") from error
    if (
        config != CONFIG.resolve(strict=True)
        or checkpoint != CHECKPOINT.resolve(strict=True)
        or data_root != DATA_ROOT.resolve(strict=True)
        or row.get("config_sha256") != inputs["config"]["sha256"]
        or row.get("checkpoint_sha256") != inputs["checkpoint"]["sha256"]
    ):
        raise ProbeEvaluationError("summary config/checkpoint/runtime identity drifted")


def postflight(
    preflight_report: Mapping[str, Any],
    *,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Replay the strict1607 result and return a diagnostic admission decision."""
    inputs = preflight_report.get("inputs")
    _assert_inputs_unchanged(inputs)
    assert isinstance(inputs, Mapping)
    path = OUTPUT / "summary.json" if summary_path is None else Path(summary_path)
    summary_record = _file_record(path, label="strict1607 summary")
    summary = _read_json(Path(summary_record["path"]), label="strict1607 summary")
    if set(summary) != {"refcoco", "tn"}:
        raise ProbeEvaluationError("summary must contain only refcoco and tn rows")
    if summary.get("refcoco") != []:
        raise ProbeEvaluationError("TN-only diagnostic unexpectedly contains Ref rows")
    tn_rows = summary.get("tn")
    if not isinstance(tn_rows, list) or len(tn_rows) != 1:
        raise ProbeEvaluationError("summary must contain exactly one TN row")
    row = tn_rows[0]
    if not isinstance(row, Mapping):
        raise ProbeEvaluationError("TN summary row must be an object")
    _validate_summary_provenance(row, inputs=inputs)

    exact_counts = {
        "num_pairs": EXPECTED_PAIRS,
        "manifest_n": EXPECTED_PAIRS,
        "source_manifest_n": EXPECTED_PAIRS,
        "invalid_positive_pairs": 0,
        "invalid_negative_pairs": 0,
        "invalid_records": 0,
    }
    for field, expected in exact_counts.items():
        if _exact_int(row.get(field), label=f"summary.{field}") != expected:
            raise ProbeEvaluationError(
                f"summary {field} must equal {expected}, got {row.get(field)!r}"
            )
    if (
        row.get("source_manifest_sha256") != TN_MANIFEST_SHA256
        or row.get("manifest_sha256") != DERIVED_TN_MANIFEST_SHA256
    ):
        raise ProbeEvaluationError("strict1607 source/derived manifest identity drifted")
    try:
        source_manifest = Path(
            str(row.get("source_manifest_path", ""))
        ).expanduser().resolve(strict=True)
    except OSError as error:
        raise ProbeEvaluationError("summary source manifest is unavailable") from error
    if source_manifest != TN_MANIFEST.resolve(strict=True):
        raise ProbeEvaluationError("summary used a different strict1607 source manifest")

    records_path = _summary_artifact_path(
        row.get("records_jsonl"), summary_path=Path(summary_record["path"]), label="records"
    )
    expected_records_root = (OUTPUT / "per_example_records").resolve()
    if records_path.parent != expected_records_root:
        raise ProbeEvaluationError("records are not in the fixed per-example directory")
    record_artifact, records = _read_and_verify_records(
        records_path, summary_row=row
    )

    threshold = _finite_float(
        row.get("threshold_at_95tpr"), label="summary.threshold_at_95tpr"
    )
    positive_scores = [
        _finite_float(record.get("pos_score"), label="record.pos_score")
        for record in records
    ]
    exact_threshold = _exact_tpr_operating_threshold(
        positive_scores, target_tpr=0.95
    )
    if not math.isclose(
        threshold, exact_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ProbeEvaluationError(
            "summary threshold_at_95tpr is not the exact score>= q05 "
            "order statistic"
        )
    false_accepts_from_records = sum(
        _finite_float(record.get("neg_score"), label="record.neg_score")
        >= threshold
        for record in records
    )
    false_accepts_from_rate = _exact_binary_count(
        row.get("fpr95tpr"), EXPECTED_PAIRS, label="summary.fpr95tpr"
    )
    tn_fpr_count = _exact_binary_count(
        row.get("tn_fpr"), EXPECTED_PAIRS, label="summary.tn_fpr"
    )
    if not (
        false_accepts_from_records == false_accepts_from_rate == tn_fpr_count
    ):
        raise ProbeEvaluationError(
            "FPR95 false-accept count does not replay from per-example scores"
        )
    actual_positive_accepts = sum(score >= threshold for score in positive_scores)
    reported_positive_accepts = _exact_binary_count(
        row.get("actual_tpr_at_95tpr"),
        EXPECTED_PAIRS,
        label="summary.actual_tpr_at_95tpr",
    )
    if actual_positive_accepts != reported_positive_accepts:
        raise ProbeEvaluationError(
            "TPR95 positive-accept count does not replay from per-example scores"
        )

    admitted = false_accepts_from_rate <= MAX_ADMITTED_FALSE_ACCEPTS
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "passed",
        "validated_at_utc": _utc_now(),
        "decision": (
            "admit_to_formal_training"
            if admitted
            else "valid_nonwin_do_not_enter_formal"
        ),
        "diagnostic_only": True,
        "formal_gate_eligible": False,
        "checkpoint": dict(inputs["checkpoint"]),
        "summary": summary_record,
        "records": record_artifact,
        "score_route": dict(EXPECTED_SCORE_ROUTE),
        "strict1607": {
            "pairs": EXPECTED_PAIRS,
            "fpr95": false_accepts_from_rate / EXPECTED_PAIRS,
            "false_accepts": false_accepts_from_rate,
            "baseline_false_accepts": BASELINE_FALSE_ACCEPTS,
            "maximum_admitted_false_accepts": MAX_ADMITTED_FALSE_ACCEPTS,
            "strict_win": admitted,
            "positive_accepts": actual_positive_accepts,
        },
        "contracts": {
            "tn_only": True,
            "full_strict1607": True,
            "zero_invalid_records": True,
            "terminal_u300_diagnostic": True,
            "full_per_example_records_bound": True,
            "fpr95_replayed_as_exact_integer_count": True,
            "still_not_formal_evaluation": True,
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise OSError(f"JSON destination must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_command_atomic(command: Sequence[str], log_path: Path) -> int:
    log_path = Path(log_path)
    if log_path.exists() or log_path.is_symlink():
        raise ProbeEvaluationError(f"console log destination is not fresh: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        log_path.parent
        / f".{log_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                list(command),
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, log_path)
        return int(completed.returncode)
    finally:
        if temporary.exists():
            temporary.unlink()


def _existing_report_exit_code(report: Mapping[str, Any]) -> int:
    decision = report.get("decision")
    if decision == "admit_to_formal_training":
        return 0
    if decision == "valid_nonwin_do_not_enter_formal":
        return 1
    return 2


def _postflight_stable_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeEvaluationError("controller report lacks postflight evidence")
    required = (
        "schema",
        "status",
        "decision",
        "diagnostic_only",
        "formal_gate_eligible",
        "checkpoint",
        "summary",
        "records",
        "score_route",
        "strict1607",
        "contracts",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ProbeEvaluationError(
            f"controller postflight evidence is incomplete: {missing}"
        )
    return {key: value[key] for key in required}


def _validate_formal_config_promotion() -> dict[str, Any]:
    from util.slconfig import SLConfig

    probe_values = SLConfig.fromfile(str(CONFIG))._cfg_dict.to_dict()
    formal_values = SLConfig.fromfile(str(FORMAL_CONFIG))._cfg_dict.to_dict()
    observed_differences = {
        key: (probe_values.get(key), formal_values.get(key))
        for key in sorted(set(probe_values) | set(formal_values))
        if probe_values.get(key) != formal_values.get(key)
    }
    if observed_differences != FORMAL_PROMOTION_OVERRIDES:
        raise ProbeEvaluationError(
            "formal config is not an exact scope/update promotion of the U300 "
            f"probe: {observed_differences}"
        )
    return {
        "schema": "pivot.stageb.confidence_adapter_veto_config_promotion/v1",
        "probe_config": _file_record(CONFIG, label="U300 probe config"),
        "formal_config": _file_record(
            FORMAL_CONFIG, label="promoted formal config"
        ),
        "allowed_overrides": {
            key: {"probe": values[0], "formal": values[1]}
            for key, values in sorted(FORMAL_PROMOTION_OVERRIDES.items())
        },
        "all_other_config_values_equal": True,
    }


def verify_admission_report(
    report_path: Path | None = None,
    *,
    health_audit: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recompute every U300 selection gate before formal training starts."""
    path = REPORT if report_path is None else Path(report_path)
    report_record_before = _file_record(
        path, label="U300 strict1607 admission report"
    )
    report = _read_json(
        Path(report_record_before["path"]),
        label="U300 strict1607 admission report",
    )
    expected_top = {
        "schema": SCHEMA,
        "status": "completed",
        "decision": "admit_to_formal_training",
        "diagnostic_only": True,
        "formal_gate_eligible": False,
    }
    for field, expected in expected_top.items():
        if report.get(field) != expected:
            raise ProbeEvaluationError(
                f"formal admission report requires {field}={expected!r}, "
                f"got {report.get(field)!r}"
            )

    launch = report.get("preflight")
    if not isinstance(launch, Mapping):
        raise ProbeEvaluationError("formal admission report lacks preflight evidence")
    if (
        launch.get("diagnostic_only") is not True
        or launch.get("formal_gate_eligible") is not False
        or launch.get("command") != build_command()
    ):
        raise ProbeEvaluationError("formal admission preflight contract drifted")
    inputs = launch.get("inputs")
    _assert_inputs_unchanged(inputs)
    if not isinstance(inputs, Mapping):
        raise ProbeEvaluationError("formal admission input identities are missing")

    state = _validate_training_state(training.inspect())
    saved_state = launch.get("training")
    if not isinstance(saved_state, Mapping) or any(
        saved_state.get(key) != state.get(key)
        for key in ("status", "action", "updates", "rank_sha256")
    ):
        raise ProbeEvaluationError(
            "formal admission training state differs from the selected U300 probe"
        )

    checkpoint_record = _file_record(
        CHECKPOINT, label="formal admission U300 checkpoint"
    )
    _record_matches(
        inputs.get("checkpoint"),
        checkpoint_record,
        label="formal admission U300 checkpoint",
    )
    audit = _load_health_audit() if health_audit is None else health_audit
    current_health = _validate_health_report(
        audit(), checkpoint_record=checkpoint_record
    )
    saved_health = launch.get("health")
    if saved_health != current_health:
        raise ProbeEvaluationError(
            "formal admission loss-health evidence changed after selection"
        )
    health_rank = current_health["candidate"]["controller"].get("rank_sha256")
    if health_rank is not None and health_rank != state["rank_sha256"]:
        raise ProbeEvaluationError(
            "formal admission health and training rank identities differ"
        )

    recomputed = postflight(launch)
    if recomputed.get("decision") != "admit_to_formal_training":
        raise ProbeEvaluationError(
            "current strict1607 replay no longer admits formal training"
        )
    stored_postflight = _postflight_stable_evidence(report.get("postflight"))
    current_postflight = _postflight_stable_evidence(recomputed)
    if stored_postflight != current_postflight:
        raise ProbeEvaluationError(
            "stored strict1607 postflight differs from the current record replay"
        )

    console_record = _file_record(LOG, label="strict1607 evaluator console log")
    _record_matches(
        report.get("console_log"),
        console_record,
        label="strict1607 evaluator console log",
    )
    report_record_after = _file_record(
        path, label="U300 strict1607 admission report"
    )
    if report_record_before != report_record_after:
        raise ProbeEvaluationError(
            "U300 strict1607 admission report changed while being verified"
        )
    return {
        "schema": ADMISSION_SCHEMA,
        "status": "verified",
        "decision": "admit_to_formal_training",
        "verifier": _file_record(
            Path(__file__), label="formal admission verifier source"
        ),
        "report": report_record_after,
        "probe_checkpoint": checkpoint_record,
        "probe_rank_sha256": state["rank_sha256"],
        "probe_optimizer_updates": EXPECTED_UPDATES,
        "health_decision": current_health["decision"],
        "strict1607": dict(recomputed["strict1607"]),
        "config_promotion": _validate_formal_config_promotion(),
        "diagnostic_only": True,
        "formal_training_admitted": True,
    }


def run() -> int:
    if REPORT.is_file() and not REPORT.is_symlink():
        try:
            report = _read_json(REPORT, label="existing controller report")
        except ProbeEvaluationError as error:
            print(f"[FAIL] {error}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return _existing_report_exit_code(report)

    report: dict[str, Any]
    try:
        launch = preflight()
    except (ProbeEvaluationError, OSError, RuntimeError) as error:
        report = {
            "schema": SCHEMA,
            "status": "invalid",
            "decision": "invalid_evidence",
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        # A preflight failure has not launched evaluation and must remain
        # retryable. Publishing it at the terminal report path would poison a
        # future run after the U300 checkpoint becomes available.
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    try:
        returncode = _run_command_atomic(launch["command"], LOG)
        if returncode != 0:
            raise ProbeEvaluationError(
                f"fixed strict1607 evaluator exited with code {returncode}"
            )
        result = postflight(launch)
        log_record = _file_record(LOG, label="atomic evaluator console log")
        report = {
            "schema": SCHEMA,
            "status": "completed",
            "decision": result["decision"],
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "preflight": launch,
            "postflight": result,
            "console_log": log_record,
        }
        exit_code = _existing_report_exit_code(report)
    except (ProbeEvaluationError, OSError, RuntimeError) as error:
        report = {
            "schema": SCHEMA,
            "status": "invalid",
            "decision": "invalid_evidence",
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "preflight": launch,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        if LOG.is_file() and not LOG.is_symlink():
            try:
                report["console_log"] = _file_record(
                    LOG, label="atomic evaluator console log"
                )
            except ProbeEvaluationError:
                pass
        exit_code = 2
    try:
        _atomic_write_json(REPORT, report)
    except OSError as error:
        print(f"[FAIL] could not publish report: {error}", file=sys.stderr)
        return 2
    stream = sys.stdout if exit_code in (0, 1) else sys.stderr
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), file=stream)
    return exit_code


def status() -> tuple[dict[str, Any], int]:
    if REPORT.exists():
        if REPORT.is_symlink() or not REPORT.is_file():
            return {"status": "invalid", "reason": "report is not a real file"}, 2
        try:
            report = _read_json(REPORT, label="controller report")
        except ProbeEvaluationError as error:
            return {"status": "invalid", "reason": str(error)}, 2
        if report.get("schema") != SCHEMA:
            return {"status": "invalid", "reason": "report schema drifted"}, 2
        return dict(report), _existing_report_exit_code(report)
    if OUTPUT.exists() and not _output_is_fresh():
        return {
            "status": "invalid",
            "reason": "evaluation output exists without an atomic controller report",
        }, 2
    try:
        launch = preflight()
    except (ProbeEvaluationError, OSError, RuntimeError) as error:
        return {"status": "blocked", "reason": str(error)}, 2
    return {
        "schema": SCHEMA,
        "status": "ready",
        "decision": "ready_for_strict1607_diagnostic",
        "diagnostic_only": True,
        "formal_gate_eligible": False,
        "preflight": launch,
    }, 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.status:
        payload, exit_code = status()
        stream = sys.stdout if exit_code in (0, 1) else sys.stderr
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), file=stream)
        return exit_code
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
