#!/usr/bin/env python3
"""Aggregate the formal Table-D mechanism diagnostics with replay checks.

The input manifest binds the completed S0/S1 training roots and paired S3
rank/final evaluation roots for every formal training seed::

    {
      "schema": "pivot.stageb.table_d_diagnostics_input/v1",
      "expected_train_seeds": [17, 42, 73],
      "gradient_training_runs": {
        "S0": {"17": "...", "42": "...", "73": "..."},
        "S1": {"17": "...", "42": "...", "73": "..."}
      },
      "s3": {
        "17": {
          "training_run_root": ".../S3/seed17",
          "rank_evaluation_root": "...",
          "confidence_evaluation_root": "..."
        }
      }
    }

S0/S1 diagnostics are the cumulative means printed at fixed updates
0,100,...,900.  The logger does not retain individual probe values, so this
tool deliberately does not report a negative-cosine frequency.  S3 compares
the diagnostic rank checkpoint with the main confidence checkpoint on the
same complete Ref8/strict2031/strict1607 surfaces and verifies that model-state
changes are confined to ``stage_b_fixed_text_scorer.validity_head.*``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_paper_evaluations as evaluator  # noqa: E402
from tools.compare_stageb_fpr95_records import exact_fpr95  # noqa: E402
from tools.stageb_eval_records import RECORD_SCHEMA  # noqa: E402
from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT,
    REF_SPLITS,
)


INPUT_SCHEMA = "pivot.stageb.table_d_diagnostics_input/v1"
REPORT_SCHEMA = "pivot.stageb.table_d_diagnostics_report/v1"
EXPECTED_SEEDS = (17, 42, 73)
PROBE_UPDATES = tuple(range(0, 1000, 100))
TN_SPLITS = ("strict2031", "strict1607")
GRADIENT_METRICS = (
    "grad_cosine",
    "grad_cosine_defined",
    "grad_rank_norm",
    "grad_confidence_norm",
    "grad_element_conflict_fraction",
    "grad_tensor_conflict_fraction",
    "grad_shared_parameter_count",
    "grad_shared_element_count",
)
COUNT_METRICS = (
    "grad_shared_parameter_count",
    "grad_shared_element_count",
)
VALIDITY_HEAD_PREFIX = "stage_b_fixed_text_scorer.validity_head."
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_STEP_RE = re.compile(r"Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)\s*/\s*\d+\s*\]")


class TableDDiagnosticsError(ValueError):
    """Raised when a Table-D diagnostic contract cannot be proven."""


class Evidence:
    def __init__(self) -> None:
        self._records: dict[Path, dict[str, Any]] = {}
        self._cache = evaluator.HashCache()

    @property
    def cache(self) -> evaluator.HashCache:
        return self._cache

    def add(self, path: Path, role: str) -> None:
        path = path.resolve(strict=True)
        record = self._records.get(path)
        if record is None:
            stat = path.stat()
            record = {
                "path": str(path),
                "sha256": self._cache.digest(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "roles": [],
            }
            self._records[path] = record
        roles = record["roles"]
        if role not in roles:
            roles.append(role)
            roles.sort()

    def records(self) -> list[dict[str, Any]]:
        return [self._records[path] for path in sorted(self._records)]


@dataclass(frozen=True)
class LoadedEvaluation:
    root: Path
    seed: int
    source_kind: str
    phase: str
    checkpoint: Path
    checkpoint_sha256: str
    launch: Mapping[str, Any]
    postflight: Mapping[str, Any]
    ref: Mapping[str, tuple[Mapping[str, Any], ...]]
    tn: Mapping[str, tuple[Mapping[str, Any], ...]]
    fpr95: Mapping[str, float]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TableDDiagnosticsError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TableDDiagnosticsError(f"{label} must be a JSON object")
    return payload


def _resolve_path(value: Any, *, base: Path, label: str, directory: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TableDDiagnosticsError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        path = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise TableDDiagnosticsError(f"{label} does not exist: {path}") from exc
    if directory != path.is_dir():
        expected = "directory" if directory else "file"
        raise TableDDiagnosticsError(f"{label} is not a {expected}: {path}")
    return path


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise TableDDiagnosticsError(
            f"{label} escapes its declared evaluation root: {path}"
        ) from exc


def _verify_file_record(
    record: Any,
    *,
    label: str,
    evidence: Evidence,
    role: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise TableDDiagnosticsError(f"{label} has no file record")
    try:
        return evaluator._verify_declared_file(
            record,
            label=label,
            cache=evidence.cache,
        )
    except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
        raise TableDDiagnosticsError(str(exc)) from exc
    finally:
        raw = record.get("path") if isinstance(record, Mapping) else None
        if isinstance(raw, str) and Path(raw).expanduser().is_file():
            evidence.add(Path(raw).expanduser(), role)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise TableDDiagnosticsError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TableDDiagnosticsError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise TableDDiagnosticsError(f"{label} must be finite")
    return result


def parse_gradient_log(path: Path) -> dict[str, Any]:
    """Read cumulative diagnostic means at exactly the ten formal probes."""
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TableDDiagnosticsError(f"cannot read gradient log {path}: {exc}") from exc
    by_update: dict[int, dict[str, float]] = {}
    for line in lines:
        step_match = _STEP_RE.search(line)
        if step_match is None:
            continue
        epoch = int(step_match.group(1))
        update = int(step_match.group(2))
        if update not in PROBE_UPDATES:
            continue
        if epoch != 0:
            raise TableDDiagnosticsError(
                f"gradient probe update {update} appeared outside epoch 0"
            )
        values: dict[str, float] = {}
        for metric in GRADIENT_METRICS:
            match = re.search(
                rf"\bstage_b_v22_{re.escape(metric)}_unscaled:\s*"
                rf"({_FLOAT})\s*\(({_FLOAT})\)",
                line,
            )
            if match is not None:
                values[metric] = _finite_number(
                    match.group(2), label=f"update {update} {metric}"
                )
        if not values:
            continue
        if set(values) != set(GRADIENT_METRICS):
            missing = sorted(set(GRADIENT_METRICS) - set(values))
            raise TableDDiagnosticsError(
                f"gradient probe update {update} is missing metrics {missing}"
            )
        if update in by_update:
            raise TableDDiagnosticsError(
                f"gradient probe update {update} appears more than once"
            )
        by_update[update] = values
    if tuple(sorted(by_update)) != PROBE_UPDATES:
        raise TableDDiagnosticsError(
            "gradient log does not contain exactly the fixed probes "
            f"{PROBE_UPDATES}; found {tuple(sorted(by_update))}"
        )

    for update, values in by_update.items():
        if not math.isclose(
            values["grad_cosine_defined"], 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise TableDDiagnosticsError(
                f"gradient cosine was undefined by update {update}"
            )
        if not -1.0001 <= values["grad_cosine"] <= 1.0001:
            raise TableDDiagnosticsError(f"invalid gradient cosine at {update}")
        for metric in ("grad_element_conflict_fraction", "grad_tensor_conflict_fraction"):
            if not -1e-9 <= values[metric] <= 1.0 + 1e-9:
                raise TableDDiagnosticsError(f"invalid {metric} at {update}")
        for metric in ("grad_rank_norm", "grad_confidence_norm"):
            if values[metric] < 0.0:
                raise TableDDiagnosticsError(f"negative {metric} at {update}")

    for metric in COUNT_METRICS:
        values = [by_update[update][metric] for update in PROBE_UPDATES]
        if any(value <= 0.0 or not float(value).is_integer() for value in values):
            raise TableDDiagnosticsError(f"{metric} is not a positive integer")
        if len(set(values)) != 1:
            raise TableDDiagnosticsError(
                f"{metric} changed across cumulative probes: {values}"
            )
    return {
        "probe_updates": list(PROBE_UPDATES),
        "cumulative_by_update": {
            str(update): dict(by_update[update]) for update in PROBE_UPDATES
        },
        "cumulative_mean": dict(by_update[PROBE_UPDATES[-1]]),
        "estimator": "logger_global_average_over_fixed_probe_updates",
    }


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) < 2:
        raise TableDDiagnosticsError("sample standard deviation requires >=2 seeds")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)),
        "n": len(values),
        "ddof": 1,
    }


def aggregate_gradient_rows(
    rows: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> dict[str, Any]:
    if set(rows) != {"S0", "S1"}:
        raise TableDDiagnosticsError("gradient rows must be exactly S0 and S1")
    output: dict[str, Any] = {}
    for row_id in ("S0", "S1"):
        seed_rows = rows[row_id]
        if tuple(sorted(seed_rows)) != EXPECTED_SEEDS:
            raise TableDDiagnosticsError(f"{row_id} gradient seed set mismatch")
        counts_by_metric: dict[str, set[float]] = {metric: set() for metric in COUNT_METRICS}
        rendered_seeds = []
        for seed in EXPECTED_SEEDS:
            parsed = seed_rows[seed]
            cumulative = parsed.get("cumulative_mean")
            if not isinstance(cumulative, Mapping) or set(cumulative) != set(
                GRADIENT_METRICS
            ):
                raise TableDDiagnosticsError(f"{row_id}:{seed} metrics are incomplete")
            metrics = {
                metric: _finite_number(
                    cumulative[metric], label=f"{row_id}:{seed}.{metric}"
                )
                for metric in GRADIENT_METRICS
            }
            for metric in COUNT_METRICS:
                counts_by_metric[metric].add(metrics[metric])
            rendered_seeds.append({"train_seed": seed, "cumulative_mean": metrics})
        for metric, values in counts_by_metric.items():
            if len(values) != 1:
                raise TableDDiagnosticsError(
                    f"{row_id} {metric} changed across seeds: {sorted(values)}"
                )
        output[row_id] = {
            "status": "available",
            "seeds": rendered_seeds,
            "aggregate": {
                metric: _sample_summary(
                    [
                        seed_rows[seed]["cumulative_mean"][metric]
                        for seed in EXPECTED_SEEDS
                    ]
                )
                for metric in GRADIENT_METRICS
                if metric != "grad_cosine_defined"
            },
            "cosine_defined_at_every_probe": True,
            "shared_counts_stable_across_probes_and_seeds": True,
        }
    output["S2"] = {
        "status": "not_applicable",
        "reason": "independent branches use structural isolation, not shared-gradient cosine",
    }
    output["S3"] = {
        "status": "not_applicable",
        "reason": "two-phase independent branches have no jointly trained shared parameters",
    }
    return {
        "probe_updates": list(PROBE_UPDATES),
        "seed_estimator": "per_seed_cumulative_mean_then_cross_seed_mean_and_sample_std",
        "metric_definition": {
            "grad_cosine": (
                "cosine between the fixed weighted rank-task and confidence-task "
                "gradients on parameters structurally shared by both losses"
            ),
            "conflict_fractions": "elementwise and per-tensor negative dot-product fractions",
            "norms": "L2 norms of the same weighted task gradients",
        },
        "rows": output,
        "negative_cosine_fraction": {
            "status": "not_estimable",
            "reason": (
                "the sealed native logger retains cumulative means, not every "
                "individual cosine sign"
            ),
        },
    }


def _read_records(
    path: Path,
    *,
    task: str,
    split: str,
    expected_n: int,
    expected_manifest_sha256: str,
    expected_run_id: str,
) -> tuple[Mapping[str, Any], ...]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TableDDiagnosticsError(f"cannot read records {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise TableDDiagnosticsError(f"{path}:{line_number}: blank JSONL row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TableDDiagnosticsError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise TableDDiagnosticsError(f"{path}:{line_number}: row is not an object")
        if (
            row.get("schema") != RECORD_SCHEMA
            or row.get("task") != task
            or row.get("valid") is not True
            or row.get("run_id") != expected_run_id
            or int(row.get("manifest_index", -1)) != len(rows)
            or int(row.get("manifest_n", -1)) != expected_n
            or str(row.get("manifest_sha256", "")).lower()
            != expected_manifest_sha256.lower()
        ):
            raise TableDDiagnosticsError(
                f"{path}:{line_number}: record surface contract mismatch"
            )
        if task == "ref" and row.get("split") != split:
            raise TableDDiagnosticsError(f"{path}:{line_number}: Ref split mismatch")
        if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
            raise TableDDiagnosticsError(f"{path}:{line_number}: missing sample_id")
        rows.append(row)
    if len(rows) != expected_n:
        raise TableDDiagnosticsError(
            f"{path}: expected {expected_n} records, found {len(rows)}"
        )
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise TableDDiagnosticsError(f"{path}: duplicate sample_id")
    return tuple(rows)


def _verify_eval_input_rehash(
    launch: Mapping[str, Any], *, root: Path, evidence: Evidence
) -> Mapping[str, Any]:
    inputs = launch.get("inputs")
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    rehash = launch.get("input_rehash_artifact")
    if not isinstance(records, list) or not records or not isinstance(rehash, Mapping):
        raise TableDDiagnosticsError("evaluation launch input contract is incomplete")
    rehash_path = _verify_file_record(
        rehash,
        label="evaluation input rehash",
        evidence=evidence,
        role="evaluation_input_rehash",
    )
    if rehash_path != (root / "input_rehash.json").resolve(strict=True):
        raise TableDDiagnosticsError("evaluation input rehash path is not canonical")
    payload = _read_json(rehash_path, label="evaluation input rehash")
    if payload.get("schema") != evaluator.INPUT_REHASH_SCHEMA or payload.get("status") != "passed":
        raise TableDDiagnosticsError("evaluation input rehash did not pass")
    replay_rows = payload.get("records")
    if not isinstance(replay_rows, list) or len(replay_rows) != len(records):
        raise TableDDiagnosticsError("evaluation rehash coverage mismatch")
    for index, (record, replay) in enumerate(zip(records, replay_rows)):
        if not isinstance(record, Mapping) or not isinstance(replay, Mapping):
            raise TableDDiagnosticsError(f"evaluation input {index} is invalid")
        path = _verify_file_record(
            record,
            label=f"evaluation input {index}",
            evidence=evidence,
            role="evaluation_launch_input",
        )
        stat = path.stat()
        expected = {
            "path": str(path),
            "roles": list(record.get("roles", [])),
            "expected_sha256": record.get("sha256"),
            "observed_sha256": record.get("sha256"),
            "observed_size_bytes": int(stat.st_size),
            "observed_mtime_ns": int(stat.st_mtime_ns),
            "passed": True,
        }
        if {key: replay.get(key) for key in expected} != expected:
            raise TableDDiagnosticsError(
                f"evaluation input rehash row {index} does not replay launch input"
            )
    return payload


def _load_evaluation(
    root: Path,
    *,
    seed: int,
    expected_kind: str,
    expected_phase: str,
    evidence: Evidence,
) -> LoadedEvaluation:
    root = root.resolve(strict=True)
    launch_path = (root / "launch_manifest.json").resolve(strict=True)
    postflight_path = (root / "postflight.json").resolve(strict=True)
    evidence.add(launch_path, f"S3_{expected_phase}_evaluation_launch")
    evidence.add(postflight_path, f"S3_{expected_phase}_evaluation_postflight")
    launch = _read_json(launch_path, label=f"S3 {expected_phase} evaluation launch")
    postflight = _read_json(
        postflight_path, label=f"S3 {expected_phase} evaluation postflight"
    )
    if launch.get("schema") != evaluator.SCHEMA or launch.get("status") != "completed":
        raise TableDDiagnosticsError(f"S3 {expected_phase} evaluation is not completed")
    if (
        launch.get("output_dir_fresh_at_plan") is not True
        or _resolve_path(
            launch.get("output_dir"),
            base=REPO_ROOT,
            label=f"S3 {expected_phase} evaluation output_dir",
            directory=True,
        )
        != root
    ):
        raise TableDDiagnosticsError(
            f"S3 {expected_phase} evaluation root is not its fresh launch root"
        )
    completed = launch.get("completed_phases")
    if (
        not isinstance(completed, list)
        or len(completed) != 2
        or any(not isinstance(row, Mapping) for row in completed)
        or [row.get("phase_id") for row in completed] != [
            "ref8_strict2031",
            "strict1607",
        ]
        or any(
            row.get("status") != "completed" or row.get("returncode") != 0
            for row in completed
        )
    ):
        raise TableDDiagnosticsError(f"S3 {expected_phase} evaluation phases are incomplete")
    if launch.get("postflight") != postflight:
        raise TableDDiagnosticsError(
            f"S3 {expected_phase} embedded and persisted postflight differ"
        )
    bound_postflight = _verify_file_record(
        launch.get("postflight_artifact"),
        label=f"S3 {expected_phase} evaluation postflight artifact",
        evidence=evidence,
        role=f"S3_{expected_phase}_evaluation_postflight",
    )
    if bound_postflight != postflight_path:
        raise TableDDiagnosticsError(
            f"S3 {expected_phase} postflight artifact path is not canonical"
        )
    if (
        postflight.get("schema") != evaluator.POSTFLIGHT_SCHEMA
        or postflight.get("status") != "passed"
        or postflight.get("profile") != evaluator.FINAL_PROFILE
        or postflight.get("evaluation_id") != launch.get("evaluation_id")
    ):
        raise TableDDiagnosticsError(f"S3 {expected_phase} postflight did not pass")
    contracts = postflight.get("contracts")
    required_contracts = {
        "ref_split_set_exact",
        "full_per_example_records",
        "zero_invalid_records",
        "locked_manifest_binding",
        "checkpoint_consistent_across_all_rows",
        "strict1607_skip_ref_observed",
    }
    if not isinstance(contracts, Mapping) or any(
        contracts.get(key) is not True for key in required_contracts
    ):
        raise TableDDiagnosticsError(f"S3 {expected_phase} contracts are incomplete")
    fixed_runtime = postflight.get("fixed_runtime")
    if (
        not isinstance(fixed_runtime, Mapping)
        or int(fixed_runtime.get("eval_seed", -1)) != evaluator.EVAL_SEED
        or int(fixed_runtime.get("max_ref_batches", -1)) != 0
        or int(fixed_runtime.get("max_tn_batches", -1)) != 0
    ):
        raise TableDDiagnosticsError(
            f"S3 {expected_phase} postflight runtime surface drifted"
        )
    protocol = launch.get("protocol")
    runtime = launch.get("runtime")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("profile") != evaluator.FINAL_PROFILE
        or tuple(protocol.get("ref_splits", ())) != tuple(REF_SPLITS)
        or protocol.get("processes") != ["ref8_strict2031", "strict1607"]
        or protocol.get("strict1607_skip_ref") is not True
        or not isinstance(runtime, Mapping)
        or int(runtime.get("eval_seed", -1)) != evaluator.EVAL_SEED
        or int(runtime.get("max_ref_batches", -1)) != 0
        or int(runtime.get("max_tn_batches", -1)) != 0
    ):
        raise TableDDiagnosticsError(f"S3 {expected_phase} evaluation surface drifted")
    source = launch.get("source")
    if not isinstance(source, Mapping):
        raise TableDDiagnosticsError("evaluation source is missing")
    expected_diagnostic = expected_phase == "rank"
    expected_training_phase = "rank" if expected_diagnostic else "final"
    if (
        source.get("kind") != expected_kind
        or source.get("training_run_id") != f"S3:{seed}"
        or int(source.get("training_seed", -1)) != seed
        or source.get("training_phase") != expected_training_phase
        or source.get("selected_phase_id") != expected_phase
        or source.get("final_phase_id") != "confidence"
        or source.get("diagnostic_only") is not expected_diagnostic
    ):
        raise TableDDiagnosticsError(f"S3 {expected_phase} source contract mismatch")
    if expected_phase == "confidence" and expected_kind != "pivot_paper_training_run":
        raise TableDDiagnosticsError("confidence evaluation is not the main S3 source")
    replayed_rehash = _verify_eval_input_rehash(
        launch, root=root, evidence=evidence
    )
    if postflight.get("input_rehash") != replayed_rehash:
        raise TableDDiagnosticsError(
            "evaluation postflight and persisted input rehash differ"
        )

    checkpoint = _resolve_path(
        source.get("checkpoint"), base=REPO_ROOT, label="evaluation checkpoint", directory=False
    )
    checkpoint_sha = evidence.cache.digest(checkpoint)
    checkpoint_record = postflight.get("checkpoint")
    if (
        checkpoint_sha != source.get("checkpoint_sha256")
        or not isinstance(checkpoint_record, Mapping)
        or checkpoint_record.get("path") != str(checkpoint)
        or checkpoint_record.get("sha256") != checkpoint_sha
    ):
        raise TableDDiagnosticsError("evaluation checkpoint binding mismatch")
    evidence.add(checkpoint, f"S3_{expected_phase}_checkpoint")
    run_id = str(checkpoint_record.get("run_id", ""))
    if not run_id:
        raise TableDDiagnosticsError("evaluation checkpoint run_id is missing")

    artifacts = postflight.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "primary_summary",
        "supplemental_summary",
        "ref8",
        "strict2031",
        "strict1607",
    }:
        raise TableDDiagnosticsError("evaluation postflight artifacts are missing")
    primary_root = (root / "ref8_strict2031").resolve(strict=True)
    supplemental_root = (root / "strict1607").resolve(strict=True)
    expected_summaries = {
        "primary_summary": primary_root / "summary.json",
        "supplemental_summary": supplemental_root / "summary.json",
    }
    for summary_name, expected_summary in expected_summaries.items():
        summary_path = _verify_file_record(
            artifacts.get(summary_name),
            label=f"evaluation {summary_name}",
            evidence=evidence,
            role=f"S3_{expected_phase}_{summary_name}",
        )
        if summary_path != expected_summary:
            raise TableDDiagnosticsError(
                f"evaluation {summary_name} path is not canonical"
            )

    ref_artifacts = artifacts.get("ref8")
    if not isinstance(ref_artifacts, Mapping) or set(ref_artifacts) != set(REF_SPLITS):
        raise TableDDiagnosticsError("evaluation Ref8 artifact set mismatch")
    ref_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for split in REF_SPLITS:
        artifact = ref_artifacts[split]
        if not isinstance(artifact, Mapping):
            raise TableDDiagnosticsError(f"{split} artifact is invalid")
        contract = REF_SPLIT_CONTRACT[split]
        expected_n = int(contract["rows"])
        expected_sha = str(contract["sha256"])
        if int(artifact.get("manifest_n", -1)) != expected_n or str(
            artifact.get("manifest_sha256", "")
        ).lower() != expected_sha.lower():
            raise TableDDiagnosticsError(f"{split} manifest surface mismatch")
        records_path = _verify_file_record(
            artifact.get("records"),
            label=f"{split} records",
            evidence=evidence,
            role=f"S3_{expected_phase}_{split}_records",
        )
        _require_within(records_path, primary_root, label=f"{split} records")
        rows = _read_records(
            records_path,
            task="ref",
            split=split,
            expected_n=expected_n,
            expected_manifest_sha256=expected_sha,
            expected_run_id=run_id,
        )
        correct = []
        for index, row in enumerate(rows):
            if not isinstance(row.get("correct50"), bool):
                raise TableDDiagnosticsError(f"{split}:{index}: correct50 is invalid")
            iou = _finite_number(row.get("top1_iou"), label=f"{split}:{index}.top1_iou")
            if row["correct50"] != (iou >= 0.5):
                raise TableDDiagnosticsError(f"{split}:{index}: correct50 disagrees with IoU")
            correct.append(bool(row["correct50"]))
        measured = float(sum(correct) / len(correct))
        reported = _finite_number(
            artifact.get("summary_acc50"), label=f"{split}.summary_acc50"
        )
        if not math.isclose(measured, reported, rel_tol=0.0, abs_tol=1e-12):
            raise TableDDiagnosticsError(f"{split} Acc50 does not replay records")
        ref_rows[split] = rows

    tn_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    fpr95: dict[str, float] = {}
    for split in TN_SPLITS:
        artifact = artifacts.get(split)
        specification = evaluator.STRICT_SPECS[split]
        if not isinstance(artifact, Mapping):
            raise TableDDiagnosticsError(f"{split} artifact is invalid")
        expected_n = int(specification["rows"])
        if (
            int(artifact.get("manifest_n", -1)) != expected_n
            or str(artifact.get("source_manifest_sha256", "")).lower()
            != str(specification["sha256"]).lower()
            or artifact.get("manifest_binding_mode") != "source_to_derived_v1"
        ):
            raise TableDDiagnosticsError(f"{split} source surface mismatch")
        derived_sha = str(artifact.get("derived_manifest_sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", derived_sha) is None:
            raise TableDDiagnosticsError(f"{split} derived manifest SHA is invalid")
        records_path = _verify_file_record(
            artifact.get("records"),
            label=f"{split} records",
            evidence=evidence,
            role=f"S3_{expected_phase}_{split}_records",
        )
        expected_section = primary_root if split == "strict2031" else supplemental_root
        _require_within(
            records_path, expected_section, label=f"{split} records"
        )
        rows = _read_records(
            records_path,
            task="tn",
            split=split,
            expected_n=expected_n,
            expected_manifest_sha256=derived_sha,
            expected_run_id=run_id,
        )
        positive = np.asarray(
            [_finite_number(row.get("pos_score"), label=f"{split}.pos_score") for row in rows],
            dtype=np.float64,
        )
        negative = np.asarray(
            [_finite_number(row.get("neg_score"), label=f"{split}.neg_score") for row in rows],
            dtype=np.float64,
        )
        measured = float(exact_fpr95(positive, negative)["fpr"])
        reported = _finite_number(
            artifact.get("summary_fpr95"), label=f"{split}.summary_fpr95"
        )
        if not math.isclose(measured, reported, rel_tol=0.0, abs_tol=1e-12):
            raise TableDDiagnosticsError(f"{split} FPR95 does not replay records")
        tn_rows[split] = rows
        fpr95[split] = measured

    return LoadedEvaluation(
        root=root,
        seed=seed,
        source_kind=expected_kind,
        phase=expected_phase,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        launch=launch,
        postflight=postflight,
        ref=ref_rows,
        tn=tn_rows,
        fpr95=fpr95,
    )


def _identity(row: Mapping[str, Any], *, task: str) -> tuple[Any, ...]:
    common = (
        row.get("manifest_index"),
        row.get("sample_id"),
        row.get("image_id"),
        row.get("ann_id"),
        row.get("ref_id"),
        row.get("sent_id"),
    )
    return common + ((row.get("split"),) if task == "ref" else ())


def compare_s3_evaluations(
    rank: LoadedEvaluation, confidence: LoadedEvaluation
) -> dict[str, Any]:
    if rank.seed != confidence.seed:
        raise TableDDiagnosticsError("S3 paired evaluation seed mismatch")
    if tuple(rank.ref) != tuple(confidence.ref) or tuple(rank.tn) != tuple(confidence.tn):
        raise TableDDiagnosticsError("S3 paired evaluation split set mismatch")
    if rank.launch.get("runtime") != confidence.launch.get("runtime") or rank.launch.get(
        "protocol"
    ) != confidence.launch.get("protocol"):
        raise TableDDiagnosticsError("S3 paired runtime/protocol surfaces differ")

    def surface(launch: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
        selected_roles = {
            "evaluation_data_input",
            "evaluation_code_dependency",
            "strict2031",
            "strict1607",
        }
        result = {}
        for record in launch["inputs"]["records"]:
            roles = set(record.get("roles", []))
            if not roles.intersection(selected_roles):
                continue
            result[str(record["path"])] = (
                record.get("sha256"),
                record.get("size_bytes"),
                tuple(sorted(roles.intersection(selected_roles))),
            )
        return result

    if surface(rank.launch) != surface(confidence.launch):
        raise TableDDiagnosticsError("S3 rank/confidence evaluation inputs differ")

    ref_output: dict[str, Any] = {}
    rank_acc = []
    confidence_acc = []
    for split in REF_SPLITS:
        rank_rows = rank.ref[split]
        confidence_rows = confidence.ref[split]
        if len(rank_rows) != len(confidence_rows) or any(
            _identity(left, task="ref") != _identity(right, task="ref")
            or left.get("manifest_sha256") != right.get("manifest_sha256")
            or left.get("manifest_n") != right.get("manifest_n")
            for left, right in zip(rank_rows, confidence_rows)
        ):
            raise TableDDiagnosticsError(f"{split} paired record alignment failed")
        regressions = sum(
            bool(left["correct50"]) and not bool(right["correct50"])
            for left, right in zip(rank_rows, confidence_rows)
        )
        fixes = sum(
            not bool(left["correct50"]) and bool(right["correct50"])
            for left, right in zip(rank_rows, confidence_rows)
        )
        changed = sum(
            float(left["top1_iou"]) != float(right["top1_iou"])
            for left, right in zip(rank_rows, confidence_rows)
        )
        n = len(rank_rows)
        rank_correct = sum(bool(row["correct50"]) for row in rank_rows)
        confidence_correct = sum(bool(row["correct50"]) for row in confidence_rows)
        rank_value = rank_correct / n
        confidence_value = confidence_correct / n
        rank_acc.append(rank_value)
        confidence_acc.append(confidence_value)
        ref_output[split] = {
            "n": n,
            "rank_acc50": rank_value,
            "confidence_acc50": confidence_value,
            "rank_correct_to_confidence_wrong_count": regressions,
            "rank_correct_to_confidence_wrong_rate": regressions / n,
            "confidence_fixes_count": fixes,
            "net_correct_delta_count": confidence_correct - rank_correct,
            "acc50_delta": confidence_value - rank_value,
            "top1_iou_changed_count": changed,
        }
    rank_mean = statistics.fmean(rank_acc)
    confidence_mean = statistics.fmean(confidence_acc)

    tn_output: dict[str, Any] = {}
    for split in TN_SPLITS:
        rank_rows = rank.tn[split]
        confidence_rows = confidence.tn[split]
        if len(rank_rows) != len(confidence_rows) or any(
            _identity(left, task="tn") != _identity(right, task="tn")
            or left.get("manifest_sha256") != right.get("manifest_sha256")
            or left.get("manifest_n") != right.get("manifest_n")
            for left, right in zip(rank_rows, confidence_rows)
        ):
            raise TableDDiagnosticsError(f"{split} paired record alignment failed")
        tn_output[split] = {
            "n": len(rank_rows),
            "rank_fpr95": rank.fpr95[split],
            "confidence_fpr95": confidence.fpr95[split],
            "fpr95_delta": confidence.fpr95[split] - rank.fpr95[split],
            "aligned_records": True,
        }
    return {
        "train_seed": rank.seed,
        "ref8": ref_output,
        "ref8_unweighted_split_mean": {
            "rank": rank_mean,
            "confidence": confidence_mean,
            "regression": confidence_mean - rank_mean,
        },
        "tn": tn_output,
        "surface_alignment": {
            "runtime_and_protocol_identical": True,
            "evaluation_data_code_and_strict_inputs_identical": True,
            "ref_and_tn_records_aligned": True,
        },
    }


def checkpoint_allowlist(
    rank_checkpoint: Path, confidence_checkpoint: Path
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise TableDDiagnosticsError(
            "checkpoint allowlist requires the formal PyTorch environment"
        ) from exc

    def load(path: Path) -> Mapping[str, Any]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise TableDDiagnosticsError(f"cannot load checkpoint {path}: {exc}") from exc
        state = payload.get("model") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping) or not state:
            raise TableDDiagnosticsError(f"checkpoint {path} has no model state")
        return state

    rank_state = load(rank_checkpoint)
    confidence_state = load(confidence_checkpoint)
    if set(rank_state) != set(confidence_state):
        missing = sorted(set(rank_state) ^ set(confidence_state))
        raise TableDDiagnosticsError(f"checkpoint model key set changed: {missing[:8]}")
    changed = []
    for key in sorted(rank_state):
        left = rank_state[key]
        right = confidence_state[key]
        if torch.is_tensor(left) != torch.is_tensor(right):
            raise TableDDiagnosticsError(f"checkpoint value type changed for {key}")
        if torch.is_tensor(left):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise TableDDiagnosticsError(f"checkpoint tensor surface changed for {key}")
            if left.layout != torch.strided or right.layout != torch.strided:
                raise TableDDiagnosticsError(
                    f"checkpoint tensor layout is unsupported for byte audit: {key}"
                )
            try:
                left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
                right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
            except (RuntimeError, TypeError) as exc:
                raise TableDDiagnosticsError(
                    f"cannot expose checkpoint tensor bytes for {key}: {exc}"
                ) from exc
            equal = torch.equal(left_bytes, right_bytes)
        else:
            equal = left == right
        if not equal:
            changed.append(key)
    if not changed:
        raise TableDDiagnosticsError("S3 confidence phase changed no model tensors")
    disallowed = [key for key in changed if not key.startswith(VALIDITY_HEAD_PREFIX)]
    if disallowed:
        raise TableDDiagnosticsError(
            "S3 confidence checkpoint changed tensors outside validity_head: "
            f"{disallowed[:8]}"
        )
    return {
        "status": "passed",
        "allowlist_prefix": VALIDITY_HEAD_PREFIX,
        "changed_tensor_count": len(changed),
        "changed_tensors": changed,
        "all_non_allowlisted_model_tensors_bitwise_equal": True,
        "at_least_one_validity_head_tensor_changed": True,
    }


def _verify_s3_training_lineage(
    rank: evaluator.EvaluationSource,
    confidence: evaluator.EvaluationSource,
) -> dict[str, Any]:
    if (
        rank.training_run_id != confidence.training_run_id
        or rank.training_seed != confidence.training_seed
        or rank.training_run_root != confidence.training_run_root
        or rank.selected_phase_id != "rank"
        or confidence.selected_phase_id != "confidence"
    ):
        raise TableDDiagnosticsError("S3 training source lineage mismatch")
    if confidence.selected_phase_manifest is None or confidence.selected_training_postflight is None:
        raise TableDDiagnosticsError("S3 confidence provenance paths are missing")
    launch = _read_json(
        confidence.selected_phase_manifest, label="S3 confidence phase launch"
    )
    postflight = _read_json(
        confidence.selected_training_postflight, label="S3 confidence postflight"
    )
    inputs = launch.get("inputs")
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    matches = [
        record
        for record in records or []
        if isinstance(record, Mapping)
        and record.get("role") == "rank_phase_model_state_pretrain"
    ]
    ancestry = postflight.get("model_state_ancestry")
    if (
        len(matches) != 1
        or matches[0].get("path") != str(rank.checkpoint)
        or matches[0].get("sha256") != rank.checkpoint_sha256
        or not isinstance(ancestry, Mapping)
        or ancestry.get("pretrain_path") != str(rank.checkpoint)
        or ancestry.get("pretrain_sha256") != rank.checkpoint_sha256
        or ancestry.get("pretrain_manifest_role")
        != "rank_phase_model_state_pretrain"
        or ancestry.get("pretrain_mode") != "model_state_only_no_optimizer_resume"
        or ancestry.get("checkpoint_resume_argument") is not None
        or ancestry.get("scorer_warmstart_applied") is not False
    ):
        raise TableDDiagnosticsError(
            "S3 confidence phase does not prove rank-checkpoint model-state lineage"
        )
    return {
        "rank_to_confidence_model_state_lineage": True,
        "optimizer_scheduler_scaler_not_resumed": True,
        "scorer_warmstart_not_reapplied": True,
    }


def _verify_evaluation_source_matches_training(
    loaded: LoadedEvaluation, source: evaluator.EvaluationSource
) -> None:
    recorded = loaded.launch.get("source")
    if not isinstance(recorded, Mapping):
        raise TableDDiagnosticsError("S3 evaluation source is missing")
    expected_paths = {
        "config": source.config,
        "checkpoint": source.checkpoint,
        "training_run_root": source.training_run_root,
        "sequence_manifest": source.sequence_manifest,
        "final_phase_manifest": source.final_phase_manifest,
        "training_postflight": source.training_postflight,
        "selected_phase_manifest": source.selected_phase_manifest,
        "selected_training_postflight": source.selected_training_postflight,
    }
    for field, expected in expected_paths.items():
        if expected is None:
            if recorded.get(field) is not None:
                raise TableDDiagnosticsError(
                    f"S3 evaluation source unexpectedly records {field}"
                )
            continue
        try:
            observed = Path(str(recorded.get(field, ""))).expanduser().resolve(
                strict=True
            )
        except (FileNotFoundError, OSError) as exc:
            raise TableDDiagnosticsError(
                f"S3 evaluation source {field} is invalid"
            ) from exc
        if observed != expected.resolve(strict=True):
            raise TableDDiagnosticsError(
                f"S3 evaluation source {field} differs from training provenance"
            )
    expected_scalars = {
        "kind": source.kind,
        "training_run_id": source.training_run_id,
        "training_seed": source.training_seed,
        "training_phase": source.training_phase,
        "diagnostic_only": source.diagnostic_only,
        "final_phase_id": source.final_phase_id,
        "selected_phase_id": source.selected_phase_id,
        "checkpoint_sha256": source.checkpoint_sha256,
    }
    if any(recorded.get(field) != value for field, value in expected_scalars.items()):
        raise TableDDiagnosticsError(
            "S3 evaluation source scalar provenance differs from training"
        )


def _seed_map(value: Any, *, label: str) -> Mapping[int, Any]:
    if not isinstance(value, Mapping):
        raise TableDDiagnosticsError(f"{label} must be an object keyed by seed")
    converted: dict[int, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"\d+", key):
            raise TableDDiagnosticsError(f"{label} has invalid seed key {key!r}")
        seed = int(key)
        if seed in converted:
            raise TableDDiagnosticsError(f"{label} has duplicate seed {seed}")
        converted[seed] = item
    if tuple(sorted(converted)) != EXPECTED_SEEDS:
        raise TableDDiagnosticsError(
            f"{label} seed set must be exactly {EXPECTED_SEEDS}"
        )
    return converted


def aggregate(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = _read_json(manifest_path, label="Table-D diagnostics manifest")
    if set(manifest) != {
        "schema",
        "expected_train_seeds",
        "gradient_training_runs",
        "s3",
    }:
        raise TableDDiagnosticsError("Table-D diagnostics manifest fields mismatch")
    if manifest.get("schema") != INPUT_SCHEMA or manifest.get(
        "expected_train_seeds"
    ) != list(EXPECTED_SEEDS):
        raise TableDDiagnosticsError("Table-D diagnostics seed/schema contract mismatch")
    gradient_runs = manifest.get("gradient_training_runs")
    if not isinstance(gradient_runs, Mapping) or set(gradient_runs) != {"S0", "S1"}:
        raise TableDDiagnosticsError("gradient_training_runs must contain exactly S0/S1")
    gradient_maps = {
        row_id: _seed_map(gradient_runs[row_id], label=f"gradient {row_id}")
        for row_id in ("S0", "S1")
    }
    s3_map = _seed_map(manifest.get("s3"), label="S3 diagnostics")

    evidence = Evidence()
    evidence.add(manifest_path, "diagnostics_input_manifest")
    evidence.add(Path(__file__), "diagnostics_aggregator_code")
    gradient_parsed: dict[str, dict[int, Mapping[str, Any]]] = {
        "S0": {},
        "S1": {},
    }
    base = manifest_path.parent
    for row_id in ("S0", "S1"):
        for seed in EXPECTED_SEEDS:
            run_root = _resolve_path(
                gradient_maps[row_id][seed],
                base=base,
                label=f"{row_id}:{seed} training root",
                directory=True,
            )
            try:
                source = evaluator._resolve_paper_source(
                    run_root, evidence.cache, training_phase="final"
                )
            except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
                raise TableDDiagnosticsError(f"{row_id}:{seed}: {exc}") from exc
            if source.training_run_id != f"{row_id}:{seed}" or source.training_seed != seed:
                raise TableDDiagnosticsError(f"{row_id}:{seed} source identity mismatch")
            if source.selected_phase_manifest is None or source.selected_training_postflight is None:
                raise TableDDiagnosticsError(f"{row_id}:{seed} provenance is missing")
            launch = _read_json(source.selected_phase_manifest, label=f"{row_id}:{seed} launch")
            postflight = _read_json(
                source.selected_training_postflight,
                label=f"{row_id}:{seed} postflight",
            )
            phase = launch.get("phase")
            fixed = launch.get("fixed_contract")
            metadata = postflight.get("checkpoint_metadata")
            args = metadata.get("args") if isinstance(metadata, Mapping) else None
            if (
                not isinstance(phase, Mapping)
                or phase.get("phase_id") != "joint"
                or int(phase.get("diagnostic_interval", -1)) != 100
                or not isinstance(fixed, Mapping)
                or int(fixed.get("gradient_diagnostic_interval", -1)) != 100
                or not isinstance(args, Mapping)
                or int(args.get("stage_b_v22_gradient_diagnostic_interval", -1)) != 100
                or int(args.get("gradient_accumulation_steps", 1)) != 1
            ):
                raise TableDDiagnosticsError(
                    f"{row_id}:{seed} is not the fixed 100-update gradient surface"
                )
            artifacts = postflight.get("artifacts")
            native_log = _verify_file_record(
                artifacts.get("native_info_log") if isinstance(artifacts, Mapping) else None,
                label=f"{row_id}:{seed} native info log",
                evidence=evidence,
                role=f"{row_id}_gradient_log",
            )
            if native_log != (run_root / "info.txt").resolve(strict=True):
                raise TableDDiagnosticsError(f"{row_id}:{seed} log path is not canonical")
            evidence.add(source.sequence_manifest, f"{row_id}_training_sequence")
            evidence.add(source.selected_phase_manifest, f"{row_id}_training_launch")
            evidence.add(source.selected_training_postflight, f"{row_id}_training_postflight")
            evidence.add(source.config, f"{row_id}_training_config")
            evidence.add(source.checkpoint, f"{row_id}_training_checkpoint")
            for training_path in source.training_data:
                evidence.add(training_path, f"{row_id}_training_data")
            gradient_parsed[row_id][seed] = parse_gradient_log(native_log)
    gradient_report = aggregate_gradient_rows(gradient_parsed)

    s3_seed_reports = []
    for seed in EXPECTED_SEEDS:
        specification = s3_map[seed]
        if not isinstance(specification, Mapping) or set(specification) != {
            "training_run_root",
            "rank_evaluation_root",
            "confidence_evaluation_root",
        }:
            raise TableDDiagnosticsError(f"S3:{seed} manifest fields mismatch")
        training_root = _resolve_path(
            specification["training_run_root"],
            base=base,
            label=f"S3:{seed} training root",
            directory=True,
        )
        try:
            rank_source = evaluator._resolve_paper_source(
                training_root, evidence.cache, training_phase="rank"
            )
            confidence_source = evaluator._resolve_paper_source(
                training_root, evidence.cache, training_phase="final"
            )
        except (evaluator.PaperEvaluationError, OSError, ValueError) as exc:
            raise TableDDiagnosticsError(f"S3:{seed}: {exc}") from exc
        lineage = _verify_s3_training_lineage(rank_source, confidence_source)
        for source, phase in (
            (rank_source, "rank"),
            (confidence_source, "confidence"),
        ):
            evidence.add(source.config, f"S3_{phase}_training_config")
            evidence.add(source.checkpoint, f"S3_{phase}_training_checkpoint")
            for training_path in source.training_data:
                evidence.add(training_path, "S3_training_data")
        for path, role in (
            (rank_source.sequence_manifest, "S3_training_sequence"),
            (rank_source.selected_phase_manifest, "S3_rank_training_launch"),
            (rank_source.selected_training_postflight, "S3_rank_training_postflight"),
            (confidence_source.selected_phase_manifest, "S3_confidence_training_launch"),
            (confidence_source.selected_training_postflight, "S3_confidence_training_postflight"),
        ):
            if path is not None:
                evidence.add(path, role)
        rank_eval_root = _resolve_path(
            specification["rank_evaluation_root"],
            base=base,
            label=f"S3:{seed} rank evaluation root",
            directory=True,
        )
        confidence_eval_root = _resolve_path(
            specification["confidence_evaluation_root"],
            base=base,
            label=f"S3:{seed} confidence evaluation root",
            directory=True,
        )
        rank_eval = _load_evaluation(
            rank_eval_root,
            seed=seed,
            expected_kind="pivot_paper_training_run_rank_diagnostic",
            expected_phase="rank",
            evidence=evidence,
        )
        confidence_eval = _load_evaluation(
            confidence_eval_root,
            seed=seed,
            expected_kind="pivot_paper_training_run",
            expected_phase="confidence",
            evidence=evidence,
        )
        _verify_evaluation_source_matches_training(rank_eval, rank_source)
        _verify_evaluation_source_matches_training(
            confidence_eval, confidence_source
        )
        if (
            rank_eval.checkpoint != rank_source.checkpoint
            or rank_eval.checkpoint_sha256 != rank_source.checkpoint_sha256
            or confidence_eval.checkpoint != confidence_source.checkpoint
            or confidence_eval.checkpoint_sha256 != confidence_source.checkpoint_sha256
        ):
            raise TableDDiagnosticsError(f"S3:{seed} evaluation/training checkpoint mismatch")
        comparison = compare_s3_evaluations(rank_eval, confidence_eval)
        comparison["checkpoint_diff"] = checkpoint_allowlist(
            rank_source.checkpoint, confidence_source.checkpoint
        )
        comparison["training_lineage"] = lineage
        s3_seed_reports.append(comparison)

    s3_aggregate = {
        "ref8_unweighted_split_mean_regression": _sample_summary(
            [row["ref8_unweighted_split_mean"]["regression"] for row in s3_seed_reports]
        ),
        "ref8_by_split": {
            split: {
                "n": sum(row["ref8"][split]["n"] for row in s3_seed_reports),
                "rank_correct_to_confidence_wrong_count": sum(
                    row["ref8"][split][
                        "rank_correct_to_confidence_wrong_count"
                    ]
                    for row in s3_seed_reports
                ),
                "rank_correct_to_confidence_wrong_rate": (
                    sum(
                        row["ref8"][split][
                            "rank_correct_to_confidence_wrong_count"
                        ]
                        for row in s3_seed_reports
                    )
                    / sum(row["ref8"][split]["n"] for row in s3_seed_reports)
                ),
                "confidence_fixes_count": sum(
                    row["ref8"][split]["confidence_fixes_count"]
                    for row in s3_seed_reports
                ),
                "net_correct_delta_count": sum(
                    row["ref8"][split]["net_correct_delta_count"]
                    for row in s3_seed_reports
                ),
                "top1_iou_changed_count": sum(
                    row["ref8"][split]["top1_iou_changed_count"]
                    for row in s3_seed_reports
                ),
                "acc50_delta": _sample_summary(
                    [row["ref8"][split]["acc50_delta"] for row in s3_seed_reports]
                ),
            }
            for split in REF_SPLITS
        },
        "strict_fpr95_delta": {
            split: _sample_summary(
                [row["tn"][split]["fpr95_delta"] for row in s3_seed_reports]
            )
            for split in TN_SPLITS
        },
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "created_at_utc": _utc_now(),
        "input_manifest": {
            "path": str(manifest_path),
            "sha256": evidence.cache.digest(manifest_path),
        },
        "expected_train_seeds": list(EXPECTED_SEEDS),
        "gradient_conflict": gradient_report,
        "s3_rank_to_confidence": {
            "interpretation": (
                "paired rank-to-confidence regression after structural branch-isolation; "
                "not a pre-rank-phase regression estimate"
            ),
            "seeds": s3_seed_reports,
            "aggregate": s3_aggregate,
        },
        "inputs": {
            "algorithm": "sha256",
            "records": evidence.records(),
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    if path.exists():
        raise FileExistsError(f"diagnostics output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = aggregate(args.manifest)
        _write_json_atomic(args.output, report)
    except (
        TableDDiagnosticsError,
        evaluator.PaperEvaluationError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"[OK] Table-D diagnostics: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
