#!/usr/bin/env python3
"""Compare paired Stage-B TN records under the exact FPR@95TPR protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # Import-only acceptance binding uses no numeric replay.
    np = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_eval_records import (  # noqa: E402
    RECORD_SCHEMA,
    TN_DERIVED_MANIFEST_BINDING_SCHEMA,
    load_tn_derived_manifest_binding,
    sample_id_from_meta,
)


TARGET_TPR = 0.95
QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)


class RecordComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestRows:
    path: Path
    sha256: str
    file_record: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    sample_ids: tuple[str, ...]
    image_ids: tuple[int, ...]
    splits: tuple[str, ...]
    taxonomies: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TNRecords:
    path: Path
    file_record: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    valid: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    run_ids: tuple[str, ...]
    manifest_binding_mode: str


def _read_jsonl_with_record(
    path: Path,
) -> tuple[tuple[tuple[int, Dict[str, Any]], ...], Dict[str, Any]]:
    try:
        raw = path.read_bytes()
        rendered = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RecordComparisonError(f"could not read {path}: {error}") from error
    rows = []
    for line_number, line in enumerate(rendered.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecordComparisonError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise RecordComparisonError(
                f"expected a JSON object at {path}:{line_number}"
            )
        rows.append((line_number, row))
    return tuple(rows), {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    rows, _ = _read_jsonl_with_record(path)
    yield from rows


def _required_int(value: Any, *, field: str, location: str) -> int:
    if isinstance(value, bool):
        raise RecordComparisonError(f"{location}: {field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RecordComparisonError(
            f"{location}: {field} must be an integer"
        ) from error
    return result


def _finite_float(value: Any, *, field: str, location: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RecordComparisonError(
            f"{location}: {field} must be a finite number"
        ) from error
    if not math.isfinite(result):
        raise RecordComparisonError(f"{location}: {field} must be finite")
    return result


def _manifest_split(row: Mapping[str, Any], *, index: int) -> str:
    split = str(
        row.get("eval_split")
        or row.get("tn_eval_split")
        or row.get("split")
        or ""
    ).strip()
    if not split:
        raise RecordComparisonError(f"manifest row {index} has no evaluation split")
    return split


def _taxonomy_values(row: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    value = row.get("replace_category", None)
    if value is None:
        instances = row.get("instances", None)
        if isinstance(instances, list) and instances and isinstance(instances[0], dict):
            value = instances[0].get("replace_category", None)
    if value is None:
        return ("unknown",)
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise RecordComparisonError(
            f"manifest row {index} replace_category must be a string or list"
        )
    normalized = []
    for raw in values:
        text = " ".join(str(raw or "").strip().lower().split())
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized or ["unknown"])


def load_manifest(path: str | Path) -> ManifestRows:
    path = Path(path)
    parsed, file_record = _read_jsonl_with_record(path)
    rows = tuple(row for _, row in parsed)
    if not rows:
        raise RecordComparisonError(f"manifest is empty: {path}")
    sample_ids = tuple(
        sample_id_from_meta(row, task="tn", split=_manifest_split(row, index=i), index=i)
        for i, row in enumerate(rows)
    )
    if len(set(sample_ids)) != len(sample_ids):
        raise RecordComparisonError(f"manifest contains duplicate sample_id values: {path}")
    image_ids = tuple(
        _required_int(row.get("image_id"), field="image_id", location=f"manifest row {i}")
        for i, row in enumerate(rows)
    )
    return ManifestRows(
        path=path,
        sha256=str(file_record["sha256"]),
        file_record=file_record,
        rows=rows,
        sample_ids=sample_ids,
        image_ids=image_ids,
        splits=tuple(_manifest_split(row, index=i) for i, row in enumerate(rows)),
        taxonomies=tuple(_taxonomy_values(row, index=i) for i, row in enumerate(rows)),
    )


def load_tn_records(
    path: str | Path,
    manifest: ManifestRows,
    *,
    label: str,
) -> TNRecords:
    path = Path(path)
    parsed, file_record = _read_jsonl_with_record(path)
    rows = tuple(row for _, row in parsed)
    if len(rows) != len(manifest.rows):
        raise RecordComparisonError(
            f"{label}: records N={len(rows)} != manifest N={len(manifest.rows)}"
        )

    valid_values = []
    positive_values = []
    negative_values = []
    run_ids = set()
    binding_flags = {
        row.get("manifest_binding_schema") == TN_DERIVED_MANIFEST_BINDING_SCHEMA
        for row in rows
    }
    if len(binding_flags) != 1:
        raise RecordComparisonError(
            f"{label}: mixed legacy and two-layer TN manifest records"
        )
    manifest_binding = None
    if binding_flags == {True}:
        binding_paths = {str(row.get("manifest_binding_path", "")) for row in rows}
        if len(binding_paths) != 1 or not next(iter(binding_paths)):
            raise RecordComparisonError(f"{label}: no unique TN manifest binding path")
        try:
            manifest_binding = load_tn_derived_manifest_binding(
                Path(next(iter(binding_paths)))
            )
        except (OSError, TypeError, ValueError) as error:
            raise RecordComparisonError(
                f"{label}: invalid source-to-derived TN manifest binding: {error}"
            ) from error
        expected_source = {**dict(manifest.file_record), "rows": len(manifest.rows)}
        if dict(manifest_binding.source_manifest) != expected_source:
            raise RecordComparisonError(
                f"{label}: TN record binding source is not the comparison manifest"
            )
        if len(manifest_binding.row_mapping) != len(rows):
            raise RecordComparisonError(f"{label}: TN row mapping length mismatch")
        binding_mode = "source_to_derived_v1"
    else:
        # Explicit compatibility for records produced when the evaluator read
        # the locked source directly.  A legacy record whose hash names some
        # unbound derived file is intentionally rejected below.
        binding_mode = "legacy_direct_source_v1"
    for index, row in enumerate(rows):
        location = f"{label} record {index}"
        if row.get("schema") != RECORD_SCHEMA:
            raise RecordComparisonError(
                f"{location}: schema must be exactly {RECORD_SCHEMA!r}"
            )
        if row.get("task") != "tn" or row.get("manifest_key") != "tn_global":
            raise RecordComparisonError(
                f"{location}: expected task='tn' and manifest_key='tn_global'"
            )
        expected_manifest_hash = (
            str(manifest_binding.derived_manifest["sha256"])
            if manifest_binding is not None
            else manifest.sha256
        )
        if str(row.get("manifest_sha256", "")).lower() != expected_manifest_hash:
            legacy_detail = (
                "; legacy records are accepted only when they directly name the "
                "locked source, not an unbound derived manifest"
                if manifest_binding is None
                else ""
            )
            raise RecordComparisonError(
                f"{location}: manifest hash mismatch{legacy_detail}"
            )
        if _required_int(
            row.get("manifest_n"), field="manifest_n", location=location
        ) != len(manifest.rows):
            raise RecordComparisonError(f"{location}: manifest_n mismatch")
        if _required_int(
            row.get("manifest_index"), field="manifest_index", location=location
        ) != index:
            raise RecordComparisonError(
                f"{location}: manifest indices are not exact 0..N-1 order"
            )
        mapping = (
            manifest_binding.row_mapping[index]
            if manifest_binding is not None
            else None
        )
        expected_sample_id = (
            str(mapping["sample_id"]) if mapping is not None else manifest.sample_ids[index]
        )
        if str(row.get("sample_id", "")) != expected_sample_id:
            raise RecordComparisonError(f"{location}: sample_id/order mismatch")
        if _required_int(
            row.get("image_id"), field="image_id", location=location
        ) != manifest.image_ids[index]:
            raise RecordComparisonError(f"{location}: image_id/order mismatch")
        expected_split = (
            str(mapping["eval_split"]) if mapping is not None else manifest.splits[index]
        )
        if str(row.get("split", "")).strip() != expected_split:
            raise RecordComparisonError(f"{location}: split/order mismatch")
        if manifest_binding is not None:
            expected_binding_fields = {
                "manifest_path": str(manifest_binding.derived_manifest["path"]),
                "manifest_size_bytes": int(
                    manifest_binding.derived_manifest["size_bytes"]
                ),
                "manifest_derivation_algorithm": str(
                    manifest_binding.derivation["contract"]["algorithm"]
                ),
                "manifest_binding_sha256": manifest_binding.sha256,
                "manifest_binding_size_bytes": manifest_binding.size_bytes,
                "manifest_row_mapping_sha256": manifest_binding.row_mapping_sha256,
                "source_manifest_path": str(manifest_binding.source_manifest["path"]),
                "source_manifest_sha256": manifest.sha256,
                "source_manifest_size_bytes": int(manifest.file_record["size_bytes"]),
                "source_manifest_n": len(manifest.rows),
                "source_manifest_index": int(mapping["source_index"]),
            }
            for field, expected in expected_binding_fields.items():
                if row.get(field) != expected:
                    raise RecordComparisonError(
                        f"{location}: {field} does not match TN manifest binding"
                    )
        valid = row.get("valid", None)
        if type(valid) is not bool:
            raise RecordComparisonError(f"{location}: valid must be an exact boolean")
        valid_values.append(valid)
        if valid:
            positive_values.append(
                _finite_float(row.get("pos_score"), field="pos_score", location=location)
            )
            negative_values.append(
                _finite_float(row.get("neg_score"), field="neg_score", location=location)
            )
        else:
            positive_values.append(float("nan"))
            negative_values.append(float("nan"))
        run_ids.add(str(row.get("run_id", "")).strip() or "unknown")

    return TNRecords(
        path=path,
        file_record=file_record,
        rows=rows,
        valid=np.asarray(valid_values, dtype=np.bool_),
        positive=np.asarray(positive_values, dtype=np.float64),
        negative=np.asarray(negative_values, dtype=np.float64),
        run_ids=tuple(sorted(run_ids)),
        manifest_binding_mode=binding_mode,
    )


def exact_fpr_at_tpr(
    positive_scores: Sequence[float] | np.ndarray,
    negative_scores: Sequence[float] | np.ndarray,
    *,
    target_tpr: float,
) -> Dict[str, Any]:
    positive = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    target_tpr = float(target_tpr)
    if (
        positive.size == 0
        or negative.size == 0
        or not np.isfinite(positive).all()
        or not np.isfinite(negative).all()
    ):
        raise RecordComparisonError(
            "exact FPR@TPR requires non-empty finite positive and negative arrays"
        )
    if not 0.0 < target_tpr <= 1.0:
        raise RecordComparisonError("target_tpr must be in (0, 1]")
    accepted = max(1, int(math.ceil(target_tpr * int(positive.size))))
    ascending_index = int(positive.size) - accepted
    threshold = float(np.partition(positive, ascending_index)[ascending_index])
    return {
        "target_tpr": target_tpr,
        "threshold": threshold,
        "accepted_positive_n": accepted,
        "actual_tpr": float(np.mean(positive >= threshold)),
        "fpr": float(np.mean(negative >= threshold)),
        "tie_policy": ">=",
    }


def exact_fpr95(
    positive_scores: Sequence[float] | np.ndarray,
    negative_scores: Sequence[float] | np.ndarray,
) -> Dict[str, Any]:
    return exact_fpr_at_tpr(
        positive_scores,
        negative_scores,
        target_tpr=TARGET_TPR,
    )


def exact_binary_auroc(
    positive_scores: Sequence[float] | np.ndarray,
    negative_scores: Sequence[float] | np.ndarray,
) -> float:
    """Return Mann-Whitney AUROC with half credit for exact score ties."""

    positive = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if (
        positive.size == 0
        or negative.size == 0
        or not np.isfinite(positive).all()
        or not np.isfinite(negative).all()
    ):
        raise RecordComparisonError(
            "AUROC requires non-empty finite positive and negative arrays"
        )
    scores = np.concatenate((positive, negative))
    labels = np.concatenate(
        (
            np.ones(positive.size, dtype=np.bool_),
            np.zeros(negative.size, dtype=np.bool_),
        )
    )
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Average of the one-indexed ranks occupied by this exact-tie block.
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    mann_whitney = positive_rank_sum - (
        float(positive.size) * float(positive.size + 1) / 2.0
    )
    return float(mann_whitney / (float(positive.size) * float(negative.size)))


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    result = np.quantile(np.asarray(values, dtype=np.float64), QUANTILES)
    return {
        f"q{int(round(quantile * 100)):02d}": float(value)
        for quantile, value in zip(QUANTILES, result)
    }


def _score_summary(positive: np.ndarray, negative: np.ndarray) -> Dict[str, Any]:
    return {
        "n": int(positive.size),
        "fpr90": exact_fpr_at_tpr(positive, negative, target_tpr=0.90),
        "fpr95": exact_fpr95(positive, negative),
        "auroc": exact_binary_auroc(positive, negative),
        "positive_quantiles": _quantiles(positive),
        "negative_quantiles": _quantiles(negative),
        "pair_win_rate": float(np.mean(positive > negative)),
        "pair_tie_rate": float(np.mean(positive == negative)),
        "positive_score_mean": float(np.mean(positive)),
        "negative_score_mean": float(np.mean(negative)),
        "paired_score_gap_mean": float(np.mean(positive - negative)),
    }


def _at_threshold(
    positive: np.ndarray,
    negative: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    return {
        "threshold": float(threshold),
        "positive_tpr": float(np.mean(positive >= float(threshold))),
        "negative_fpr": float(np.mean(negative >= float(threshold))),
    }


def _group_comparison(
    indices: np.ndarray,
    *,
    baseline: TNRecords,
    candidate: TNRecords,
    baseline_global_threshold: float,
    candidate_global_threshold: float,
) -> Dict[str, Any]:
    baseline_positive = baseline.positive[indices]
    baseline_negative = baseline.negative[indices]
    candidate_positive = candidate.positive[indices]
    candidate_negative = candidate.negative[indices]
    baseline_summary = _score_summary(baseline_positive, baseline_negative)
    candidate_summary = _score_summary(candidate_positive, candidate_negative)
    baseline_at_global = _at_threshold(
        baseline_positive, baseline_negative, baseline_global_threshold
    )
    candidate_at_global = _at_threshold(
        candidate_positive, candidate_negative, candidate_global_threshold
    )
    return {
        "n": int(indices.size),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "candidate_minus_baseline_fpr95": float(
            candidate_summary["fpr95"]["fpr"] - baseline_summary["fpr95"]["fpr"]
        ),
        "candidate_minus_baseline_fpr90": float(
            candidate_summary["fpr90"]["fpr"] - baseline_summary["fpr90"]["fpr"]
        ),
        "candidate_minus_baseline_auroc": float(
            candidate_summary["auroc"] - baseline_summary["auroc"]
        ),
        "candidate_minus_baseline_pair_win_rate": float(
            candidate_summary["pair_win_rate"]
            - baseline_summary["pair_win_rate"]
        ),
        "at_each_model_global_threshold": {
            "baseline": baseline_at_global,
            "candidate": candidate_at_global,
            "candidate_minus_baseline_negative_fpr": float(
                candidate_at_global["negative_fpr"]
                - baseline_at_global["negative_fpr"]
            ),
        },
    }


def _image_clusters(image_ids: np.ndarray) -> list[np.ndarray]:
    grouped: Dict[int, list[int]] = defaultdict(list)
    order = []
    for index, raw_image_id in enumerate(image_ids.tolist()):
        image_id = int(raw_image_id)
        if image_id not in grouped:
            order.append(image_id)
        grouped[image_id].append(index)
    return [np.asarray(grouped[image_id], dtype=np.int64) for image_id in order]


def paired_fpr95_bootstrap(
    baseline_positive: np.ndarray,
    baseline_negative: np.ndarray,
    candidate_positive: np.ndarray,
    candidate_negative: np.ndarray,
    image_ids: np.ndarray,
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    if int(iterations) <= 0:
        raise RecordComparisonError("bootstrap iterations must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise RecordComparisonError("bootstrap confidence must be in (0, 1)")
    size = int(baseline_positive.size)
    if not all(
        int(values.size) == size
        for values in (
            baseline_negative,
            candidate_positive,
            candidate_negative,
            image_ids,
        )
    ):
        raise RecordComparisonError("paired bootstrap arrays must have equal length")
    clusters = _image_clusters(image_ids)
    if not clusters:
        raise RecordComparisonError("cannot bootstrap an empty valid record set")

    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        sampled_clusters = rng.integers(0, len(clusters), size=len(clusters))
        sampled = np.concatenate([clusters[index] for index in sampled_clusters])
        baseline_fpr = exact_fpr95(
            baseline_positive[sampled], baseline_negative[sampled]
        )["fpr"]
        candidate_fpr = exact_fpr95(
            candidate_positive[sampled], candidate_negative[sampled]
        )["fpr"]
        deltas[iteration] = candidate_fpr - baseline_fpr

    alpha = (1.0 - float(confidence)) / 2.0
    ci_low, ci_high = np.quantile(deltas, [alpha, 1.0 - alpha])
    observed = exact_fpr95(candidate_positive, candidate_negative)["fpr"] - exact_fpr95(
        baseline_positive, baseline_negative
    )["fpr"]
    return {
        "unit": "image_cluster",
        "paired": True,
        "recomputes_each_model_q05_per_resample": True,
        "iterations": int(iterations),
        "confidence": float(confidence),
        "seed": int(seed),
        "valid_records_n": size,
        "image_clusters_n": len(clusters),
        "observed_candidate_minus_baseline_fpr95": float(observed),
        "delta_mean": float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "delta_std": float(deltas.std(ddof=0)),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "probability_delta_below_zero": float(np.mean(deltas < 0.0)),
        "ci_supports_lower_candidate_fpr95": bool(ci_high < 0.0),
    }


def compare_records(
    baseline: TNRecords,
    candidate: TNRecords,
    manifest: ManifestRows,
    *,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260711,
) -> Dict[str, Any]:
    if baseline.valid.shape != candidate.valid.shape or not np.array_equal(
        baseline.valid, candidate.valid
    ):
        raise RecordComparisonError("baseline/candidate valid mask mismatch")
    valid_indices = np.flatnonzero(baseline.valid)
    if valid_indices.size == 0:
        raise RecordComparisonError("paired record set has zero common valid rows")

    baseline_global = _score_summary(
        baseline.positive[valid_indices], baseline.negative[valid_indices]
    )
    candidate_global = _score_summary(
        candidate.positive[valid_indices], candidate.negative[valid_indices]
    )
    baseline_threshold = baseline_global["fpr95"]["threshold"]
    candidate_threshold = candidate_global["fpr95"]["threshold"]

    split_indices: Dict[str, list[int]] = defaultdict(list)
    taxonomy_indices: Dict[str, list[int]] = defaultdict(list)
    valid_set = set(valid_indices.tolist())
    for index in valid_indices.tolist():
        split_indices[manifest.splits[index]].append(index)
        for taxonomy in manifest.taxonomies[index]:
            taxonomy_indices[taxonomy].append(index)

    def build_groups(groups: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
        return {
            name: _group_comparison(
                np.asarray(indices, dtype=np.int64),
                baseline=baseline,
                candidate=candidate,
                baseline_global_threshold=baseline_threshold,
                candidate_global_threshold=candidate_threshold,
            )
            for name, indices in sorted(groups.items())
            if indices and set(indices).issubset(valid_set)
        }

    bootstrap = paired_fpr95_bootstrap(
        baseline.positive[valid_indices],
        baseline.negative[valid_indices],
        candidate.positive[valid_indices],
        candidate.negative[valid_indices],
        np.asarray(manifest.image_ids, dtype=np.int64)[valid_indices],
        iterations=bootstrap_iterations,
        confidence=confidence,
        seed=seed,
    )
    return {
        "schema": "stageb-fpr95-record-comparison-v1",
        "validation": {
            "pass": True,
            "manifest_path": str(manifest.path),
            "manifest_sha256": manifest.sha256,
            "manifest_n": len(manifest.rows),
            "valid_n": int(valid_indices.size),
            "invalid_n": int(len(manifest.rows) - valid_indices.size),
            "manifest_index_order_match": True,
            "sample_id_order_match": True,
            "image_id_order_match": True,
            "split_order_match": True,
            "valid_mask_match": True,
            "baseline_records": str(baseline.path),
            "candidate_records": str(candidate.path),
            "baseline_run_ids": list(baseline.run_ids),
            "candidate_run_ids": list(candidate.run_ids),
            "baseline_manifest_binding_mode": baseline.manifest_binding_mode,
            "candidate_manifest_binding_mode": candidate.manifest_binding_mode,
        },
        "input_files": {
            "manifest": dict(manifest.file_record),
            "baseline_records": dict(baseline.file_record),
            "candidate_records": dict(candidate.file_record),
            "identity_is_from_the_same_bytes_used_for_metrics": True,
        },
        "global": {
            "baseline": baseline_global,
            "candidate": candidate_global,
            "candidate_minus_baseline_fpr95": float(
                candidate_global["fpr95"]["fpr"]
                - baseline_global["fpr95"]["fpr"]
            ),
            "candidate_minus_baseline_fpr90": float(
                candidate_global["fpr90"]["fpr"]
                - baseline_global["fpr90"]["fpr"]
            ),
            "candidate_minus_baseline_auroc": float(
                candidate_global["auroc"] - baseline_global["auroc"]
            ),
            "candidate_minus_baseline_pair_win_rate": float(
                candidate_global["pair_win_rate"]
                - baseline_global["pair_win_rate"]
            ),
        },
        "paired_bootstrap": bootstrap,
        "by_split": build_groups(split_indices),
        "by_taxonomy": build_groups(taxonomy_indices),
        "taxonomy_note": (
            "A manifest row may contribute to multiple taxonomy groups; group counts "
            "therefore need not sum to valid_n."
        ),
    }


def compare_record_files(
    *,
    baseline_records: str | Path,
    candidate_records: str | Path,
    manifest_path: str | Path,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260711,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    baseline = load_tn_records(baseline_records, manifest, label="baseline")
    candidate = load_tn_records(candidate_records, manifest, label="candidate")
    return compare_records(
        baseline,
        candidate,
        manifest,
        bootstrap_iterations=bootstrap_iterations,
        confidence=confidence,
        seed=seed,
    )


def _fmt(value: Any) -> str:
    return f"{float(value):.6f}"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    global_result = report["global"]
    baseline = global_result["baseline"]
    candidate = global_result["candidate"]
    bootstrap = report["paired_bootstrap"]
    lines = [
        "# Stage-B Paired FPR95 Record Comparison",
        "",
        f"Manifest: `{_markdown_cell(validation['manifest_sha256'])}` "
        f"({validation['manifest_n']} rows; {validation['valid_n']} valid; "
        f"{validation['invalid_n']} invalid)",
        "",
        "Exact operating rule: positive q05 order statistic with `score >= threshold`.",
        "",
        "## Global",
        "",
        "| model | valid N | q05 threshold | actual TPR | FPR90 | FPR95 | AUROC | pair win |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, values in (("baseline", baseline), ("candidate", candidate)):
        lines.append(
            f"| {label} | {values['n']} | {_fmt(values['fpr95']['threshold'])} | "
            f"{_fmt(values['fpr95']['actual_tpr'])} | "
            f"{_fmt(values['fpr90']['fpr'])} | {_fmt(values['fpr95']['fpr'])} | "
            f"{_fmt(values['auroc'])} | {_fmt(values['pair_win_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Candidate minus baseline FPR95: "
            f"`{_fmt(global_result['candidate_minus_baseline_fpr95'])}`.",
            "",
            "## Score Quantiles",
            "",
            "| model/distribution | q01 | q05 | q50 | q95 | q99 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, values in (("baseline", baseline), ("candidate", candidate)):
        for distribution, key in (
            ("positive", "positive_quantiles"),
            ("TN", "negative_quantiles"),
        ):
            quantiles = values[key]
            lines.append(
                f"| {label} {distribution} | {_fmt(quantiles['q01'])} | "
                f"{_fmt(quantiles['q05'])} | {_fmt(quantiles['q50'])} | "
                f"{_fmt(quantiles['q95'])} | {_fmt(quantiles['q99'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired Bootstrap",
            "",
            f"Image-cluster bootstrap ({bootstrap['iterations']} iterations, "
            f"{bootstrap['confidence']:.1%} CI): candidate-minus-baseline "
            f"`[{_fmt(bootstrap['delta_ci_low'])}, "
            f"{_fmt(bootstrap['delta_ci_high'])}]`; "
            f"P(delta < 0) = `{_fmt(bootstrap['probability_delta_below_zero'])}`.",
            "",
            "Each resample recomputes each model's positive q05 threshold.",
            "",
        ]
    )

    def add_group_table(title: str, groups: Mapping[str, Any]) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| group | N | baseline FPR95 | candidate FPR95 | delta | "
                "baseline AUROC | candidate AUROC | baseline FPR at global threshold | "
                "candidate FPR at global threshold |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        ordered = sorted(groups.items(), key=lambda item: (-int(item[1]["n"]), item[0]))
        for name, values in ordered:
            at_global = values["at_each_model_global_threshold"]
            lines.append(
                f"| {_markdown_cell(name)} | {values['n']} | "
                f"{_fmt(values['baseline']['fpr95']['fpr'])} | "
                f"{_fmt(values['candidate']['fpr95']['fpr'])} | "
                f"{_fmt(values['candidate_minus_baseline_fpr95'])} | "
                f"{_fmt(values['baseline']['auroc'])} | "
                f"{_fmt(values['candidate']['auroc'])} | "
                f"{_fmt(at_global['baseline']['negative_fpr'])} | "
                f"{_fmt(at_global['candidate']['negative_fpr'])} |"
            )
        lines.append("")

    add_group_table("By Split", report["by_split"])
    add_group_table("By Taxonomy", report["by_taxonomy"])
    return "\n".join(lines).rstrip() + "\n"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-records", required=True)
    parser.add_argument("--candidate-records", required=True)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Exact TN manifest whose SHA-256 is declared by both record files.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = compare_record_files(
            baseline_records=args.baseline_records,
            candidate_records=args.candidate_records,
            manifest_path=args.manifest,
            bootstrap_iterations=args.bootstrap_iterations,
            confidence=args.confidence,
            seed=args.seed,
        )
    except RecordComparisonError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    rendered_json = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json:
        _write_atomic(Path(args.output_json), rendered_json)
    if args.output_markdown:
        _write_atomic(Path(args.output_markdown), render_markdown(report))
    print(rendered_json, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
