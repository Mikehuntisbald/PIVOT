#!/usr/bin/env python3
"""Aggregate multi-seed Stage-B paper results under an explicit contract.

The input manifest is deliberately small and path based.  Paths are resolved
relative to the manifest file.  Every result artifact may be either a path
string or ``{"path": ..., "sha256": ..., "size_bytes": ...}``; declared
hashes and sizes are mandatory bindings, not annotations.

Minimal manifest shape::

    {
      "schema": "stageb-paper-results-manifest-v1",
      "expected_train_seeds": [41, 42, 43],
      "baseline_experiment": "gdino_data_ft",
      "protocol": {
        "ref_splits": [
          "refcoco_val", "refcoco_testA", "refcoco_testB",
          "refcocop_val", "refcocop_testA", "refcocop_testB",
          "refcocog_val", "refcocog_test"
        ],
        "tn_splits": {
          "strict2031": {
            "manifest": {"path": "strict2031.jsonl", "sha256": "..."},
            "expected_n": 2031
          },
          "strict1607": {
            "manifest": {"path": "strict1607.jsonl", "sha256": "..."},
            "expected_n": 1607
          }
        },
        "bootstrap": {"iterations": 5000, "confidence": 0.95, "seed": 20260717}
      },
      "experiments": [{
        "id": "gdino_data_ft",
        "label": "GDINO Stage-B data FT",
        "runs": [{
          "train_seed": 41,
          "artifacts": {
            "checkpoint": {"path": "checkpoint.pth", "sha256": "..."},
            "config": {"path": "config.py", "sha256": "..."},
            "data": [{"path": "train.jsonl", "sha256": "..."}]
          },
          "results": {
            "ref": {
              "summary": "ref8/summary.json",
              "records": {"refcoco_val": "ref8/refcoco_val.records.jsonl"}
            },
            "tn": {
              "strict2031": {
                "summary": "strict2031/summary.json",
                "records": "strict2031/tn.records.jsonl"
              },
              "strict1607": {
                "summary": "strict1607/summary.json",
                "records": "strict1607/tn.records.jsonl"
              }
            }
          }
        }]
      }]
    }

The ``records`` map in the example is abbreviated; complete reports require
all eight protocol keys.  ``--allow-incomplete`` is only for progress reports:
it preserves usable metrics but marks every output INCOMPLETE and lists the
failed contracts.  Without it, the first contract violation exits non-zero and
no result files are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compare_stageb_fpr95_records import (  # noqa: E402
    ManifestRows,
    RecordComparisonError,
    TNRecords,
    exact_binary_auroc,
    exact_fpr_at_tpr,
    exact_fpr95,
    load_manifest,
    load_tn_records,
    paired_fpr95_bootstrap,
)
from tools.stageb_eval_records import (  # noqa: E402
    RECORD_SCHEMA,
    RefRecordContractError,
    RefRecords,
    load_formal_ref_records,
)
from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT,
    REF_SPLITS,
)


SCHEMA = "stageb-paper-results-manifest-v1"
REPORT_SCHEMA = "stageb-paper-results-report-v1"
TN_SPLITS = ("strict2031", "strict1607")
TN_METRICS = (
    "fpr95",
    "fpr90",
    "threshold_at_95tpr",
    "actual_tpr_at_95tpr",
    "auroc",
    "pair_win_rate",
    "pair_tie_rate",
    "positive_score_mean",
    "negative_score_mean",
    "paired_score_gap_mean",
)
REF_SPLIT_NONINFERIORITY_MARGIN = 0.01
POSITIVE_Q05_NONINFERIORITY_MARGIN = 0.02


class PaperAggregationError(ValueError):
    """Raised when a paper-result contract cannot be proven."""


@dataclass
class LoadedRun:
    train_seed: int
    artifacts: Mapping[str, Any]
    ref_metrics: MutableMapping[str, float]
    ref_records: MutableMapping[str, RefRecords]
    tn_metrics: MutableMapping[str, float]
    tn_metric_details: MutableMapping[str, Mapping[str, Any]]
    tn_records: MutableMapping[str, TNRecords]
    inputs: MutableMapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_artifact_unchanged(
    record: Mapping[str, Any], *, label: str
) -> None:
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    expected_size = _required_int(
        record.get("size_bytes"), label=f"{label}.size_bytes"
    )
    expected_sha = str(record.get("sha256", ""))
    stat = path.stat()
    if int(stat.st_size) != expected_size or _sha256(path) != expected_sha:
        raise PaperAggregationError(
            f"{label}: artifact changed between identity verification and parsing"
        )


def _resolve_path(path: Any, base_dir: Path, *, label: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise PaperAggregationError(f"{label}: path must be a non-empty string")
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = base_dir / result
    return result.resolve()


def _artifact_record(
    value: Any,
    base_dir: Path,
    *,
    label: str,
    always_hash: bool,
) -> Dict[str, Any]:
    if isinstance(value, str):
        specification: Mapping[str, Any] = {"path": value}
    elif isinstance(value, Mapping):
        specification = value
    else:
        raise PaperAggregationError(
            f"{label}: artifact must be a path string or object"
        )
    unexpected = set(specification) - {"path", "sha256", "size_bytes", "label"}
    if unexpected:
        raise PaperAggregationError(
            f"{label}: unexpected artifact fields {sorted(unexpected)}"
        )
    path = _resolve_path(specification.get("path"), base_dir, label=label)
    if not path.is_file():
        raise PaperAggregationError(f"{label}: file does not exist: {path}")
    size = int(path.stat().st_size)
    if "size_bytes" in specification:
        try:
            expected_size = int(specification["size_bytes"])
        except (TypeError, ValueError) as error:
            raise PaperAggregationError(
                f"{label}: size_bytes must be an integer"
            ) from error
        if size != expected_size:
            raise PaperAggregationError(
                f"{label}: size mismatch, expected {expected_size}, found {size}"
            )
    expected_sha = specification.get("sha256")
    if expected_sha is not None:
        expected_sha = str(expected_sha).strip().lower()
        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise PaperAggregationError(f"{label}: sha256 must be 64 lowercase hex characters")
    actual_sha = _sha256(path) if always_hash or expected_sha is not None else None
    if expected_sha is not None and actual_sha != expected_sha:
        raise PaperAggregationError(
            f"{label}: SHA-256 mismatch, expected {expected_sha}, found {actual_sha}"
        )
    result: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": size,
        "sha256": actual_sha,
        "hash_verified": expected_sha is not None,
    }
    if "label" in specification:
        result["label"] = str(specification["label"])
    return result


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperAggregationError(f"{label}: could not read JSON: {error}") from error


def _read_jsonl(path: Path, *, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        rendered = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PaperAggregationError(f"{label}: could not read JSONL: {error}") from error
    for line_number, line in enumerate(rendered.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PaperAggregationError(
                f"{label}:{line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise PaperAggregationError(f"{label}:{line_number}: expected an object")
        rows.append(row)
    return rows


def _finite_unit(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PaperAggregationError(f"{label}: expected a finite number") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise PaperAggregationError(f"{label}: expected a finite number in [0, 1]")
    return result


def _finite_number(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PaperAggregationError(f"{label}: expected a finite number") from error
    if not math.isfinite(result):
        raise PaperAggregationError(f"{label}: expected a finite number")
    return result


def _required_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise PaperAggregationError(f"{label}: expected an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        rendered = value.strip()
        if rendered and rendered.lstrip("+-").isdigit():
            return int(rendered)
    raise PaperAggregationError(f"{label}: expected an integer")


def _summary_record_path_matches(
    reported: Any,
    explicit: Path,
    *,
    summary_path: Path,
    manifest_base: Path,
) -> bool:
    if not isinstance(reported, str) or not reported.strip():
        return False
    raw = Path(reported).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        REPO_ROOT / raw,
        summary_path.parent / raw,
        manifest_base / raw,
    ]
    return any(candidate.resolve() == explicit.resolve() for candidate in candidates)


def _summary_checkpoint_matches(row: Mapping[str, Any], checkpoint: Optional[Path]) -> bool:
    if checkpoint is None:
        return True
    reported = row.get("checkpoint")
    if not isinstance(reported, str) or not reported.strip():
        return False
    path = Path(reported).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve() == checkpoint.resolve()


def _select_summary_row(
    rows: Any,
    *,
    label: str,
    run_id: Optional[str],
    checkpoint: Optional[Path],
    dataset: Optional[str] = None,
) -> Mapping[str, Any]:
    if not isinstance(rows, list):
        raise PaperAggregationError(f"{label}: summary section must be a list")
    candidates = [row for row in rows if isinstance(row, Mapping)]
    if dataset is not None:
        candidates = [row for row in candidates if row.get("dataset") == dataset]
    if run_id is not None:
        candidates = [row for row in candidates if row.get("run_id") == run_id]
    if checkpoint is not None:
        candidates = [row for row in candidates if _summary_checkpoint_matches(row, checkpoint)]
    if len(candidates) != 1:
        details = []
        if dataset is not None:
            details.append(f"dataset={dataset!r}")
        if run_id is not None:
            details.append(f"run_id={run_id!r}")
        if checkpoint is not None:
            details.append(f"checkpoint={str(checkpoint)!r}")
        raise PaperAggregationError(
            f"{label}: expected exactly one summary row ({', '.join(details)}), "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _load_ref_records(
    artifact: Any,
    *,
    base_dir: Path,
    label: str,
    split: str,
    summary_row: Mapping[str, Any],
    summary_path: Path,
) -> RefRecords:
    try:
        return load_formal_ref_records(
            artifact,
            base_dir=base_dir,
            label=label,
            split=split,
            summary_row=summary_row,
            summary_path=summary_path,
            split_contract=REF_SPLIT_CONTRACT,
        )
    except RefRecordContractError as error:
        raise PaperAggregationError(str(error)) from error


def _mean_std(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise PaperAggregationError("mean/std requires non-empty finite values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size >= 2 else None,
        "std_ddof": 1,
    }


def _tn_score_metrics(
    positive_scores: Sequence[float] | np.ndarray,
    negative_scores: Sequence[float] | np.ndarray,
) -> Dict[str, Any]:
    positive = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if positive.size != negative.size:
        raise PaperAggregationError("paired TN score arrays must have equal length")
    fpr95 = exact_fpr95(positive, negative)
    fpr90 = exact_fpr_at_tpr(positive, negative, target_tpr=0.90)
    return {
        "n": int(positive.size),
        "fpr95": float(fpr95["fpr"]),
        "fpr90": float(fpr90["fpr"]),
        "threshold_at_95tpr": float(fpr95["threshold"]),
        "actual_tpr_at_95tpr": float(fpr95["actual_tpr"]),
        "auroc": exact_binary_auroc(positive, negative),
        "pair_win_rate": float(np.mean(positive > negative)),
        "pair_tie_rate": float(np.mean(positive == negative)),
        "positive_score_mean": float(np.mean(positive)),
        "negative_score_mean": float(np.mean(negative)),
        "paired_score_gap_mean": float(np.mean(positive - negative)),
        "operating_rule": {
            "target_tpr": 0.95,
            "accepted_positive_n": int(fpr95["accepted_positive_n"]),
            "tie_policy": str(fpr95["tie_policy"]),
        },
    }


def _paper_tn_taxonomy_group(value: Any) -> str:
    rendered = str(value or "").lower().replace("_", " ").replace("/", " ")
    words = set(rendered.split())
    groups = set()
    if "relation" in words or "relational" in words:
        groups.add("relation")
    elif words.intersection(
        {
            "spatial",
            "position",
            "location",
            "direction",
            "orientation",
            "distance",
            "side",
            "placement",
            "proximity",
            "depth",
        }
    ):
        groups.add("spatial")
    if words.intersection({"color", "colour", "shade", "brightness", "tone"}):
        groups.add("color")
    if words.intersection(
        {
            "size",
            "height",
            "width",
            "length",
            "scale",
            "quantity",
            "count",
            "number",
            "dimension",
        }
    ):
        groups.add("size")
    if words.intersection(
        {
            "action",
            "activity",
            "motion",
            "behavior",
            "behaviour",
            "gesture",
            "posture",
            "pose",
            "sport",
        }
    ):
        groups.add("action")
    if words.intersection(
        {
            "noun",
            "category",
            "object",
            "type",
            "species",
            "animal",
            "vehicle",
            "food",
            "identity",
            "entity",
            "item",
            "subject",
            "person",
            "breed",
        }
    ):
        groups.add("noun_category")
    if not groups:
        return "other"
    if len(groups) > 1:
        return "mixed"
    return next(iter(groups))


def _tn_metric_details(
    records: TNRecords,
    manifest: ManifestRows,
) -> Dict[str, Any]:
    if len(records.rows) != len(manifest.rows):
        raise PaperAggregationError("TN records/manifest row-count mismatch")
    valid = np.flatnonzero(records.valid)
    if valid.size == 0:
        raise PaperAggregationError("TN metric details require valid records")
    by_taxonomy: Dict[str, List[int]] = defaultdict(list)
    by_taxonomy_group: Dict[str, List[int]] = defaultdict(list)
    valid_set = set(valid.tolist())
    for index, taxonomies in enumerate(manifest.taxonomies):
        if index not in valid_set:
            continue
        for taxonomy in taxonomies:
            by_taxonomy[str(taxonomy)].append(index)
        for group in {
            _paper_tn_taxonomy_group(taxonomy) for taxonomy in taxonomies
        }:
            by_taxonomy_group[group].append(index)
    return {
        "global": _tn_score_metrics(
            records.positive[valid], records.negative[valid]
        ),
        "by_taxonomy": {
            taxonomy: _tn_score_metrics(
                records.positive[np.asarray(indices, dtype=np.int64)],
                records.negative[np.asarray(indices, dtype=np.int64)],
            )
            for taxonomy, indices in sorted(by_taxonomy.items())
        },
        "by_taxonomy_group": {
            taxonomy: _tn_score_metrics(
                records.positive[np.asarray(indices, dtype=np.int64)],
                records.negative[np.asarray(indices, dtype=np.int64)],
            )
            for taxonomy, indices in sorted(by_taxonomy_group.items())
        },
        "taxonomy_note": (
            "Rows with multiple replace_category values contribute to each declared "
            "group, so taxonomy counts need not sum to the global count."
        ),
    }


def _clusters(image_ids: np.ndarray) -> List[np.ndarray]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    order: List[int] = []
    for index, raw_image_id in enumerate(image_ids.tolist()):
        image_id = int(raw_image_id)
        if image_id not in grouped:
            order.append(image_id)
        grouped[image_id].append(index)
    return [np.asarray(grouped[image_id], dtype=np.int64) for image_id in order]


def _bootstrap_summary(
    deltas: np.ndarray,
    *,
    observed: float,
    iterations: int,
    confidence: float,
    seed: int,
    unit: str,
    clusters_n: int,
) -> Dict[str, Any]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(deltas, [alpha, 1.0 - alpha])
    return {
        "paired": True,
        "unit": unit,
        "iterations": int(iterations),
        "confidence": float(confidence),
        "seed": int(seed),
        "image_clusters_n": int(clusters_n),
        "observed_candidate_minus_baseline": float(observed),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=0)),
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
        "probability_delta_below_zero": float(np.mean(deltas < 0.0)),
    }


def paired_accuracy_bootstrap(
    baseline: RefRecords,
    candidate: RefRecords,
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    if baseline.identities != candidate.identities:
        raise PaperAggregationError("Ref record identity/order mismatch")
    if not np.array_equal(baseline.image_ids, candidate.image_ids):
        raise PaperAggregationError("Ref image identity/order mismatch")
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise PaperAggregationError("invalid bootstrap iterations/confidence")
    clusters = _clusters(baseline.image_ids)
    if not clusters:
        raise PaperAggregationError("cannot bootstrap empty Ref records")
    base = baseline.correct50.astype(np.float64)
    cand = candidate.correct50.astype(np.float64)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in chosen])
        deltas[iteration] = float(cand[indices].mean() - base[indices].mean())
    return _bootstrap_summary(
        deltas,
        observed=float(cand.mean() - base.mean()),
        iterations=iterations,
        confidence=confidence,
        seed=seed,
        unit="image_cluster",
        clusters_n=len(clusters),
    )


def paired_ref8_bootstrap(
    baseline: Mapping[str, RefRecords],
    candidate: Mapping[str, RefRecords],
    splits: Sequence[str],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    prepared = []
    total_clusters = 0
    for split in splits:
        base = baseline[split]
        cand = candidate[split]
        if base.identities != cand.identities or not np.array_equal(base.image_ids, cand.image_ids):
            raise PaperAggregationError(f"{split}: Ref record identity/order mismatch")
        clusters = _clusters(base.image_ids)
        total_clusters += len(clusters)
        prepared.append(
            (base.correct50.astype(np.float64), cand.correct50.astype(np.float64), clusters)
        )
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        split_deltas = []
        for base, cand, clusters in prepared:
            chosen = rng.integers(0, len(clusters), size=len(clusters))
            indices = np.concatenate([clusters[index] for index in chosen])
            split_deltas.append(float(cand[indices].mean() - base[indices].mean()))
        deltas[iteration] = float(np.mean(split_deltas))
    observed = float(
        np.mean([cand.mean() - base.mean() for base, cand, _ in prepared])
    )
    result = _bootstrap_summary(
        deltas,
        observed=observed,
        iterations=iterations,
        confidence=confidence,
        seed=seed,
        unit="image_cluster_within_split",
        clusters_n=total_clusters,
    )
    result["split_weighting"] = "equal_weight_over_eight_ref_splits"
    return result


def paired_ref_seed_first_bootstrap(
    baseline: Mapping[str, RefRecords],
    candidates: Mapping[int, Mapping[str, RefRecords]],
    splits: Sequence[str],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap the seed-mean Ref result with one global COCO-image draw.

    The same image multiplicities are applied to the fixed baseline and every
    candidate training seed.  Per-seed paired deltas are computed first, then
    averaged with equal seed weight; training seeds themselves are not
    resampled because their sample standard deviation is reported separately.
    """

    ordered_splits = tuple(splits)
    ordered_candidates = tuple(sorted(candidates))
    if not ordered_splits or not ordered_candidates:
        raise PaperAggregationError(
            "seed-first Ref bootstrap requires splits and candidate seeds"
        )
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise PaperAggregationError("invalid bootstrap iterations/confidence")

    image_ids = set()
    for split in ordered_splits:
        if split not in baseline:
            raise PaperAggregationError(f"missing baseline Ref records for {split}")
        base = baseline[split]
        image_ids.update(int(value) for value in base.image_ids.tolist())
        for train_seed in ordered_candidates:
            candidate = candidates[train_seed].get(split)
            if candidate is None:
                raise PaperAggregationError(
                    f"candidate seed {train_seed} lacks Ref records for {split}"
                )
            if base.identities != candidate.identities:
                raise PaperAggregationError(
                    f"{split}: Ref record identity/order mismatch for seed {train_seed}"
                )
            if not np.array_equal(base.image_ids, candidate.image_ids):
                raise PaperAggregationError(
                    f"{split}: Ref image identity/order mismatch for seed {train_seed}"
                )
    canonical_images = tuple(sorted(image_ids))
    if not canonical_images:
        raise PaperAggregationError("cannot bootstrap empty Ref records")
    image_position = {
        image_id: index for index, image_id in enumerate(canonical_images)
    }
    split_count_rows = []
    baseline_hit_rows = []
    candidate_hit_rows: Dict[int, list[np.ndarray]] = {
        train_seed: [] for train_seed in ordered_candidates
    }
    for split in ordered_splits:
        base = baseline[split]
        positions = np.asarray(
            [image_position[int(value)] for value in base.image_ids.tolist()],
            dtype=np.int64,
        )
        split_count_rows.append(
            np.bincount(positions, minlength=len(canonical_images)).astype(
                np.float64
            )
        )
        baseline_hit_rows.append(
            np.bincount(
                positions,
                weights=base.correct50.astype(np.float64),
                minlength=len(canonical_images),
            )
        )
        for train_seed in ordered_candidates:
            candidate_hit_rows[train_seed].append(
                np.bincount(
                    positions,
                    weights=candidates[train_seed][split].correct50.astype(
                        np.float64
                    ),
                    minlength=len(canonical_images),
                )
            )

    split_counts = np.stack(split_count_rows, axis=0)
    baseline_hits = np.stack(baseline_hit_rows, axis=0)
    candidate_hits = {
        train_seed: np.stack(rows, axis=0)
        for train_seed, rows in candidate_hit_rows.items()
    }
    split_deltas = {
        train_seed: np.empty(
            (int(iterations), len(ordered_splits)), dtype=np.float64
        )
        for train_seed in ordered_candidates
    }
    rng = np.random.default_rng(int(seed))
    for iteration in range(int(iterations)):
        for _attempt in range(1000):
            multiplicity = np.bincount(
                rng.integers(
                    0,
                    len(canonical_images),
                    size=len(canonical_images),
                ),
                minlength=len(canonical_images),
            ).astype(np.float64)
            denominators = split_counts @ multiplicity
            if bool(np.all(denominators > 0.0)):
                break
        else:
            raise PaperAggregationError(
                "could not draw global Ref image clusters covering every split"
            )
        baseline_accuracy = (baseline_hits @ multiplicity) / denominators
        for train_seed in ordered_candidates:
            candidate_accuracy = (
                candidate_hits[train_seed] @ multiplicity
            ) / denominators
            split_deltas[train_seed][iteration] = (
                candidate_accuracy - baseline_accuracy
            )

    observed_by_seed: Dict[int, np.ndarray] = {}
    for train_seed in ordered_candidates:
        observed_by_seed[train_seed] = np.asarray(
            [
                float(
                    candidates[train_seed][split].correct50.mean()
                    - baseline[split].correct50.mean()
                )
                for split in ordered_splits
            ],
            dtype=np.float64,
        )
    per_seed: Dict[str, Any] = {}
    for train_seed in ordered_candidates:
        per_seed[str(train_seed)] = {
            "splits": {
                split: _bootstrap_summary(
                    split_deltas[train_seed][:, split_index],
                    observed=float(observed_by_seed[train_seed][split_index]),
                    iterations=iterations,
                    confidence=confidence,
                    seed=seed,
                    unit="global_canonical_coco_image_cluster",
                    clusters_n=len(canonical_images),
                )
                for split_index, split in enumerate(ordered_splits)
            },
            "mean8_acc50": {
                **_bootstrap_summary(
                    split_deltas[train_seed].mean(axis=1),
                    observed=float(observed_by_seed[train_seed].mean()),
                    iterations=iterations,
                    confidence=confidence,
                    seed=seed,
                    unit="global_canonical_coco_image_cluster",
                    clusters_n=len(canonical_images),
                ),
                "split_weighting": "equal_weight_over_eight_ref_splits",
            },
        }

    seed_mean_split_deltas = np.mean(
        np.stack(
            [split_deltas[train_seed] for train_seed in ordered_candidates],
            axis=0,
        ),
        axis=0,
    )
    seed_mean_observed = np.mean(
        np.stack(
            [observed_by_seed[train_seed] for train_seed in ordered_candidates],
            axis=0,
        ),
        axis=0,
    )
    return {
        "paired": True,
        "unit": "global_canonical_coco_image_cluster",
        "draw_sharing": (
            "one_global_image_draw_shared_by_fixed_baseline_and_all_"
            "candidate_train_seeds"
        ),
        "seed_aggregation": (
            "per_seed_delta_then_equal_weight_mean_without_seed_resampling"
        ),
        "zero_split_draw_policy": "redraw_until_every_ref_split_has_records",
        "candidate_train_seeds": list(ordered_candidates),
        "iterations": int(iterations),
        "confidence": float(confidence),
        "seed": int(seed),
        "image_clusters_n": len(canonical_images),
        "per_seed": per_seed,
        "mean_across_candidate_train_seeds": {
            "splits": {
                split: _bootstrap_summary(
                    seed_mean_split_deltas[:, split_index],
                    observed=float(seed_mean_observed[split_index]),
                    iterations=iterations,
                    confidence=confidence,
                    seed=seed,
                    unit="global_canonical_coco_image_cluster",
                    clusters_n=len(canonical_images),
                )
                for split_index, split in enumerate(ordered_splits)
            },
            "mean8_acc50": {
                **_bootstrap_summary(
                    seed_mean_split_deltas.mean(axis=1),
                    observed=float(seed_mean_observed.mean()),
                    iterations=iterations,
                    confidence=confidence,
                    seed=seed,
                    unit="global_canonical_coco_image_cluster",
                    clusters_n=len(canonical_images),
                ),
                "split_weighting": "equal_weight_over_eight_ref_splits",
            },
        },
    }


def paired_tn_seed_first_bootstrap(
    baseline: TNRecords,
    candidates: Mapping[int, TNRecords],
    image_ids: np.ndarray,
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap seed-mean FPR95 and positive-q05 deltas on shared draws."""

    ordered_candidates = tuple(sorted(candidates))
    if not ordered_candidates:
        raise PaperAggregationError(
            "seed-first TN bootstrap requires candidate seeds"
        )
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise PaperAggregationError("invalid bootstrap iterations/confidence")
    if not bool(baseline.valid.all()):
        raise PaperAggregationError(
            "seed-first TN bootstrap requires complete valid baseline records"
        )
    size = int(baseline.positive.size)
    images = np.asarray(image_ids, dtype=np.int64).reshape(-1)
    if size == 0 or images.size != size:
        raise PaperAggregationError("TN image and score arrays must be non-empty/aligned")
    for train_seed in ordered_candidates:
        candidate = candidates[train_seed]
        if (
            int(candidate.positive.size) != size
            or not np.array_equal(candidate.valid, baseline.valid)
            or not bool(candidate.valid.all())
        ):
            raise PaperAggregationError(
                f"candidate seed {train_seed} TN records are not aligned/complete"
            )

    canonical_images = tuple(sorted(set(int(value) for value in images.tolist())))
    image_position = {
        image_id: index for index, image_id in enumerate(canonical_images)
    }
    record_cluster = np.asarray(
        [image_position[int(value)] for value in images.tolist()], dtype=np.int64
    )
    fpr_deltas = {
        train_seed: np.empty(int(iterations), dtype=np.float64)
        for train_seed in ordered_candidates
    }
    threshold_deltas = {
        train_seed: np.empty(int(iterations), dtype=np.float64)
        for train_seed in ordered_candidates
    }
    rng = np.random.default_rng(int(seed))
    for iteration in range(int(iterations)):
        multiplicity = np.bincount(
            rng.integers(
                0,
                len(canonical_images),
                size=len(canonical_images),
            ),
            minlength=len(canonical_images),
        )
        sampled = np.repeat(
            np.arange(size, dtype=np.int64), multiplicity[record_cluster]
        )
        if sampled.size == 0:
            raise PaperAggregationError("TN image-cluster draw was empty")
        baseline_operating = exact_fpr95(
            baseline.positive[sampled], baseline.negative[sampled]
        )
        for train_seed in ordered_candidates:
            candidate = candidates[train_seed]
            candidate_operating = exact_fpr95(
                candidate.positive[sampled], candidate.negative[sampled]
            )
            fpr_deltas[train_seed][iteration] = float(
                candidate_operating["fpr"] - baseline_operating["fpr"]
            )
            threshold_deltas[train_seed][iteration] = float(
                candidate_operating["threshold"]
                - baseline_operating["threshold"]
            )

    baseline_observed = exact_fpr95(baseline.positive, baseline.negative)

    def summarize(
        deltas: np.ndarray, *, observed: float, metric: str
    ) -> Dict[str, Any]:
        summary = _bootstrap_summary(
            deltas,
            observed=observed,
            iterations=iterations,
            confidence=confidence,
            seed=seed,
            unit="image_cluster",
            clusters_n=len(canonical_images),
        )
        summary["metric"] = metric
        return summary

    per_seed: Dict[str, Any] = {}
    observed_fpr = []
    observed_threshold = []
    for train_seed in ordered_candidates:
        candidate_operating = exact_fpr95(
            candidates[train_seed].positive,
            candidates[train_seed].negative,
        )
        fpr_observed = float(
            candidate_operating["fpr"] - baseline_observed["fpr"]
        )
        threshold_observed = float(
            candidate_operating["threshold"] - baseline_observed["threshold"]
        )
        observed_fpr.append(fpr_observed)
        observed_threshold.append(threshold_observed)
        fpr_summary = summarize(
            fpr_deltas[train_seed], observed=fpr_observed, metric="fpr95"
        )
        fpr_summary["observed_candidate_minus_baseline_fpr95"] = fpr_observed
        fpr_summary["recomputes_each_model_q05_per_resample"] = True
        fpr_summary["valid_records_n"] = size
        fpr_summary["ci_supports_lower_candidate_fpr95"] = bool(
            fpr_summary["delta_ci_high"] < 0.0
        )
        per_seed[str(train_seed)] = {
            "fpr95": fpr_summary,
            "positive_q05_threshold": summarize(
                threshold_deltas[train_seed],
                observed=threshold_observed,
                metric="positive_q05_threshold",
            ),
        }

    mean_fpr_deltas = np.mean(
        np.stack(
            [fpr_deltas[train_seed] for train_seed in ordered_candidates],
            axis=0,
        ),
        axis=0,
    )
    mean_threshold_deltas = np.mean(
        np.stack(
            [
                threshold_deltas[train_seed]
                for train_seed in ordered_candidates
            ],
            axis=0,
        ),
        axis=0,
    )
    headline_fpr = summarize(
        mean_fpr_deltas,
        observed=float(np.mean(observed_fpr)),
        metric="fpr95",
    )
    headline_fpr["observed_candidate_minus_baseline_fpr95"] = float(
        np.mean(observed_fpr)
    )
    headline_fpr["recomputes_each_model_q05_per_resample"] = True
    headline_fpr["valid_records_n"] = size
    headline_fpr["ci_supports_lower_candidate_fpr95"] = bool(
        headline_fpr["delta_ci_high"] < 0.0
    )
    return {
        "paired": True,
        "unit": "image_cluster",
        "draw_sharing": (
            "one_image_draw_shared_by_fixed_baseline_and_all_candidate_"
            "train_seeds"
        ),
        "seed_aggregation": (
            "per_seed_delta_then_equal_weight_mean_without_seed_resampling"
        ),
        "candidate_train_seeds": list(ordered_candidates),
        "iterations": int(iterations),
        "confidence": float(confidence),
        "seed": int(seed),
        "valid_records_n": size,
        "image_clusters_n": len(canonical_images),
        "recomputes_each_model_q05_per_resample": True,
        "per_seed": per_seed,
        "mean_across_candidate_train_seeds": {
            "fpr95": headline_fpr,
            "positive_q05_threshold": summarize(
                mean_threshold_deltas,
                observed=float(np.mean(observed_threshold)),
                metric="positive_q05_threshold",
            ),
        },
    }


def _headline_acceptance(
    headline: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> Dict[str, Any]:
    """Evaluate the predeclared headline superiority/noninferiority gates."""

    gates: Dict[str, Any] = {}
    ref = headline.get("ref")
    tn = headline.get("tn")
    if not isinstance(ref, Mapping) or not isinstance(tn, Mapping):
        return {
            "policy": {
                "ref8": "point_delta > 0 and paired_ci_low > 0",
                "ref_split_noninferiority_margin": REF_SPLIT_NONINFERIORITY_MARGIN,
                "strict_fpr95": "point_delta < 0 and paired_ci_high < 0",
                "positive_q05_noninferiority_margin": (
                    POSITIVE_Q05_NONINFERIORITY_MARGIN
                ),
            },
            "gates": {
                "complete_headline_bootstrap": {
                    "passed": False,
                    "reason": "headline seed-first bootstrap is incomplete",
                },
                "provenance": dict(provenance),
            },
            "pass": False,
        }

    ref8 = ref.get("mean8_acc50")
    ref_splits = ref.get("splits")
    if isinstance(ref8, Mapping):
        observed = float(ref8["observed_candidate_minus_baseline"])
        ci_low = float(ref8["delta_ci_low"])
        gates["ref8_superiority"] = {
            "observed_delta": observed,
            "ci_low": ci_low,
            "required_observed_gt": 0.0,
            "required_ci_low_gt": 0.0,
            "passed": bool(observed > 0.0 and ci_low > 0.0),
        }
    else:
        gates["ref8_superiority"] = {"passed": False, "reason": "missing"}

    split_gates: Dict[str, Any] = {}
    if isinstance(ref_splits, Mapping):
        for split in REF_SPLITS:
            values = ref_splits.get(split)
            if not isinstance(values, Mapping):
                split_gates[split] = {"passed": False, "reason": "missing"}
                continue
            observed = float(values["observed_candidate_minus_baseline"])
            ci_low = float(values["delta_ci_low"])
            split_gates[split] = {
                "observed_delta": observed,
                "ci_low": ci_low,
                "noninferiority_margin": REF_SPLIT_NONINFERIORITY_MARGIN,
                "passed": bool(
                    observed > -REF_SPLIT_NONINFERIORITY_MARGIN
                    and ci_low > -REF_SPLIT_NONINFERIORITY_MARGIN
                ),
            }
    else:
        split_gates = {
            split: {"passed": False, "reason": "missing"}
            for split in REF_SPLITS
        }
    gates["ref_split_noninferiority"] = {
        "margin": REF_SPLIT_NONINFERIORITY_MARGIN,
        "splits": split_gates,
        "passed": bool(
            split_gates
            and all(value.get("passed") is True for value in split_gates.values())
        ),
    }

    fpr_gates: Dict[str, Any] = {}
    q05_gates: Dict[str, Any] = {}
    for split in TN_SPLITS:
        values = tn.get(split)
        if not isinstance(values, Mapping):
            fpr_gates[split] = {"passed": False, "reason": "missing"}
            q05_gates[split] = {"passed": False, "reason": "missing"}
            continue
        fpr = values.get("fpr95")
        q05 = values.get("positive_q05_threshold")
        if isinstance(fpr, Mapping):
            observed = float(fpr["observed_candidate_minus_baseline_fpr95"])
            ci_high = float(fpr["delta_ci_high"])
            fpr_gates[split] = {
                "observed_delta": observed,
                "ci_high": ci_high,
                "required_observed_lt": 0.0,
                "required_ci_high_lt": 0.0,
                "passed": bool(observed < 0.0 and ci_high < 0.0),
            }
        else:
            fpr_gates[split] = {"passed": False, "reason": "missing"}
        if isinstance(q05, Mapping):
            observed = float(q05["observed_candidate_minus_baseline"])
            ci_low = float(q05["delta_ci_low"])
            q05_gates[split] = {
                "observed_delta": observed,
                "ci_low": ci_low,
                "noninferiority_margin": POSITIVE_Q05_NONINFERIORITY_MARGIN,
                "passed": bool(
                    observed > -POSITIVE_Q05_NONINFERIORITY_MARGIN
                    and ci_low > -POSITIVE_Q05_NONINFERIORITY_MARGIN
                ),
            }
        else:
            q05_gates[split] = {"passed": False, "reason": "missing"}
    gates["strict_fpr95_superiority"] = {
        "splits": fpr_gates,
        "passed": bool(
            all(value.get("passed") is True for value in fpr_gates.values())
        ),
    }
    gates["positive_q05_noninferiority"] = {
        "margin": POSITIVE_Q05_NONINFERIORITY_MARGIN,
        "splits": q05_gates,
        "passed": bool(
            all(value.get("passed") is True for value in q05_gates.values())
        ),
    }
    gates["provenance"] = dict(provenance)
    return {
        "policy": {
            "ref8": "point_delta > 0 and paired_ci_low > 0",
            "ref_split_noninferiority_margin": REF_SPLIT_NONINFERIORITY_MARGIN,
            "strict_fpr95": "point_delta < 0 and paired_ci_high < 0",
            "positive_q05_noninferiority_margin": (
                POSITIVE_Q05_NONINFERIORITY_MARGIN
            ),
            "strict_inequalities": True,
        },
        "gates": gates,
        "pass": bool(all(value.get("passed") is True for value in gates.values())),
    }


def _derived_seed(seed: int, *parts: Any) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class _Context:
    def __init__(self, allow_incomplete: bool):
        self.allow_incomplete = bool(allow_incomplete)
        self.issues: List[Dict[str, str]] = []

    def issue(self, scope: str, error: Exception | str) -> None:
        message = str(error)
        if not self.allow_incomplete:
            raise PaperAggregationError(f"{scope}: {message}")
        self.issues.append({"scope": scope, "message": message})


def _validate_run_artifacts(
    value: Any,
    *,
    base_dir: Path,
    label: str,
) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PaperAggregationError(f"{label}: artifacts must be an object")
    result: Dict[str, Any] = {}
    for key in ("checkpoint", "config"):
        if key in value:
            result[key] = _artifact_record(
                value[key], base_dir, label=f"{label}.{key}", always_hash=False
            )
    if "data" in value:
        if not isinstance(value["data"], list):
            raise PaperAggregationError(f"{label}.data must be a list")
        result["data"] = [
            _artifact_record(
                artifact,
                base_dir,
                label=f"{label}.data[{index}]",
                always_hash=False,
            )
            for index, artifact in enumerate(value["data"])
        ]
    return result


def _load_run(
    specification: Mapping[str, Any],
    *,
    experiment_id: str,
    base_dir: Path,
    ref_splits: Sequence[str],
    tn_manifests: Mapping[str, ManifestRows],
    context: _Context,
) -> LoadedRun:
    seed = _required_int(
        specification.get("train_seed"),
        label=f"experiment {experiment_id}.train_seed",
    )
    scope = f"experiment {experiment_id} seed {seed}"
    artifacts = _validate_run_artifacts(
        specification.get("artifacts"), base_dir=base_dir, label=f"{scope}.artifacts"
    )
    checkpoint = (
        Path(artifacts["checkpoint"]["path"]) if "checkpoint" in artifacts else None
    )
    results = specification.get("results")
    if not isinstance(results, Mapping):
        raise PaperAggregationError(f"{scope}: results must be an object")
    run = LoadedRun(
        train_seed=seed,
        artifacts=artifacts,
        ref_metrics={},
        ref_records={},
        tn_metrics={},
        tn_metric_details={},
        tn_records={},
        inputs={},
    )

    ref_spec = results.get("ref")
    if not isinstance(ref_spec, Mapping):
        context.issue(f"{scope}.ref", "missing Ref result specification")
    else:
        try:
            summary_record = _artifact_record(
                ref_spec.get("summary"),
                base_dir,
                label=f"{scope}.ref.summary",
                always_hash=True,
            )
            summary_path = Path(summary_record["path"])
            summary = _read_json(summary_path, label=f"{scope}.ref.summary")
            _assert_artifact_unchanged(
                summary_record, label=f"{scope}.ref.summary"
            )
            if not isinstance(summary, Mapping):
                raise PaperAggregationError("Ref summary root must be an object")
            records = ref_spec.get("records")
            if not isinstance(records, Mapping):
                raise PaperAggregationError("Ref records must be a split-to-artifact object")
            run_id = ref_spec.get("run_id")
            if run_id is not None:
                run_id = str(run_id)
            run.inputs["ref_summary"] = summary_record
            for split in ref_splits:
                try:
                    if split not in records:
                        raise PaperAggregationError(f"missing records for required split {split}")
                    row = _select_summary_row(
                        summary.get("refcoco"),
                        label=f"{scope}.ref.{split}",
                        run_id=run_id,
                        checkpoint=checkpoint,
                        dataset=split,
                    )
                    loaded = _load_ref_records(
                        records[split],
                        base_dir=base_dir,
                        label=f"{scope}.ref.{split}",
                        split=split,
                        summary_row=row,
                        summary_path=summary_path,
                    )
                    run.ref_records[split] = loaded
                    run.ref_metrics[split] = float(loaded.correct50.mean())
                except (PaperAggregationError, OSError, ValueError) as error:
                    context.issue(f"{scope}.ref.{split}", error)
        except (PaperAggregationError, OSError, ValueError) as error:
            context.issue(f"{scope}.ref", error)

    tn_spec = results.get("tn")
    if not isinstance(tn_spec, Mapping):
        context.issue(f"{scope}.tn", "missing TN result specification")
    else:
        for split in TN_SPLITS:
            try:
                split_spec = tn_spec.get(split)
                if not isinstance(split_spec, Mapping):
                    raise PaperAggregationError(f"missing result for required TN split {split}")
                if split not in tn_manifests:
                    raise PaperAggregationError(f"protocol manifest for {split} is unavailable")
                summary_record = _artifact_record(
                    split_spec.get("summary"),
                    base_dir,
                    label=f"{scope}.tn.{split}.summary",
                    always_hash=True,
                )
                summary_path = Path(summary_record["path"])
                summary = _read_json(summary_path, label=f"{scope}.tn.{split}.summary")
                _assert_artifact_unchanged(
                    summary_record, label=f"{scope}.tn.{split}.summary"
                )
                if not isinstance(summary, Mapping):
                    raise PaperAggregationError("TN summary root must be an object")
                records_record = _artifact_record(
                    split_spec.get("records"),
                    base_dir,
                    label=f"{scope}.tn.{split}.records",
                    always_hash=True,
                )
                records_path = Path(records_record["path"])
                run_id = split_spec.get("run_id")
                if run_id is not None:
                    run_id = str(run_id)
                row = _select_summary_row(
                    summary.get("tn"),
                    label=f"{scope}.tn.{split}",
                    run_id=run_id,
                    checkpoint=checkpoint,
                )
                if not _summary_record_path_matches(
                    row.get("records_jsonl"),
                    records_path,
                    summary_path=summary_path,
                    manifest_base=base_dir,
                ):
                    raise PaperAggregationError(
                        "explicit record path is not the records_jsonl bound by summary"
                    )
                loaded = load_tn_records(
                    records_path, tn_manifests[split], label=f"{scope}.tn.{split}"
                )
                _assert_artifact_unchanged(
                    records_record, label=f"{scope}.tn.{split}.records"
                )
                if dict(loaded.file_record) != {
                    key: records_record[key]
                    for key in ("path", "size_bytes", "sha256")
                }:
                    raise PaperAggregationError(
                        "TN records changed while being parsed"
                    )
                record_manifest_hashes = {
                    str(record.get("manifest_sha256", "")).lower()
                    for record in loaded.rows
                }
                if len(record_manifest_hashes) != 1:
                    raise PaperAggregationError("TN records contain mixed manifest hashes")
                if str(row.get("manifest_sha256", "")).lower() != next(
                    iter(record_manifest_hashes)
                ):
                    raise PaperAggregationError(
                        "summary manifest_sha256 does not match TN records"
                    )
                if _required_int(
                    row.get("manifest_n"),
                    label=f"{scope}.tn.{split}.summary.manifest_n",
                ) != len(loaded.rows):
                    raise PaperAggregationError(
                        "summary manifest_n does not match TN records"
                    )
                if not bool(loaded.valid.all()):
                    raise PaperAggregationError("formal TN records require every row valid")
                measured = exact_fpr95(loaded.positive, loaded.negative)["fpr"]
                reported = _finite_unit(
                    row.get("fpr95tpr"), label=f"{scope}.tn.{split}.summary.fpr95tpr"
                )
                if not math.isclose(measured, reported, rel_tol=0.0, abs_tol=1e-12):
                    raise PaperAggregationError(
                        f"summary fpr95tpr={reported} != records exact FPR95={measured}"
                    )
                measured_details = _tn_metric_details(
                    loaded, tn_manifests[split]
                )
                global_metrics = measured_details["global"]
                optional_summary_metrics = {
                    "fpr90tpr": ("fpr90", _finite_unit),
                    "threshold_at_95tpr": (
                        "threshold_at_95tpr",
                        _finite_number,
                    ),
                    "actual_tpr_at_95tpr": (
                        "actual_tpr_at_95tpr",
                        _finite_unit,
                    ),
                    "pair_win_rate": ("pair_win_rate", _finite_unit),
                    "pair_tie_rate": ("pair_tie_rate", _finite_unit),
                    "pos_score_mean": ("positive_score_mean", _finite_number),
                    "tn_score_mean": ("negative_score_mean", _finite_number),
                    "score_gap_mean": ("paired_score_gap_mean", _finite_number),
                }
                for summary_key, (metric_key, parser) in optional_summary_metrics.items():
                    if summary_key not in row:
                        continue
                    summary_value = parser(
                        row[summary_key],
                        label=f"{scope}.tn.{split}.summary.{summary_key}",
                    )
                    metric_value = float(global_metrics[metric_key])
                    if not math.isclose(
                        summary_value, metric_value, rel_tol=0.0, abs_tol=1e-7
                    ):
                        raise PaperAggregationError(
                            f"summary {summary_key}={summary_value} != records "
                            f"{metric_key}={metric_value}"
                        )
                if _required_int(row.get("num_pairs"), label=f"{scope}.tn.{split}.num_pairs") != int(loaded.valid.sum()):
                    raise PaperAggregationError("summary num_pairs does not match valid records")
                if _required_int(row.get("max_batches", 0), label=f"{scope}.tn.{split}.max_batches") != 0:
                    raise PaperAggregationError("max_batches must be zero for a formal result")
                if _required_int(row.get("invalid_records", 0), label=f"{scope}.tn.{split}.invalid_records") != 0:
                    raise PaperAggregationError("summary declares invalid TN records")
                run.tn_metrics[split] = float(measured)
                run.tn_metric_details[split] = measured_details
                run.tn_records[split] = loaded
                run.inputs[f"tn_{split}_summary"] = summary_record
                run.inputs[f"tn_{split}_records"] = records_record
            except (PaperAggregationError, RecordComparisonError, OSError, ValueError) as error:
                context.issue(f"{scope}.tn.{split}", error)
    return run


def _aggregate_experiment(
    runs: Mapping[int, LoadedRun], ref_splits: Sequence[str]
) -> Dict[str, Any]:
    per_seed: Dict[str, Any] = {}
    for seed, run in sorted(runs.items()):
        ref8 = (
            float(np.mean([run.ref_metrics[split] for split in ref_splits]))
            if all(split in run.ref_metrics for split in ref_splits)
            else None
        )
        per_seed[str(seed)] = {
            "ref": {
                "splits": dict(run.ref_metrics),
                "mean8_acc50": ref8,
            },
            "tn": dict(run.tn_metrics),
            "tn_metrics": dict(run.tn_metric_details),
            "artifacts": run.artifacts,
            "inputs": run.inputs,
        }
    ref_aggregate: Dict[str, Any] = {"splits": {}}
    for split in ref_splits:
        values = [run.ref_metrics[split] for run in runs.values() if split in run.ref_metrics]
        if values:
            ref_aggregate["splits"][split] = _mean_std(values)
    ref8_values = [
        float(np.mean([run.ref_metrics[split] for split in ref_splits]))
        for run in runs.values()
        if all(split in run.ref_metrics for split in ref_splits)
    ]
    if ref8_values:
        ref_aggregate["mean8_acc50"] = _mean_std(ref8_values)
    tn_aggregate = {}
    for split in TN_SPLITS:
        values = [run.tn_metrics[split] for run in runs.values() if split in run.tn_metrics]
        if values:
            tn_aggregate[split] = _mean_std(values)
    tn_metric_aggregate: Dict[str, Any] = {}
    tn_taxonomy_aggregate: Dict[str, Any] = {}
    tn_taxonomy_raw_aggregate: Dict[str, Any] = {}

    def aggregate_taxonomies(
        details: Sequence[Mapping[str, Any]],
        *,
        split: str,
        key: str,
    ) -> Dict[str, Any]:
        taxonomy_sets = [set(detail[key]) for detail in details]
        if any(values != taxonomy_sets[0] for values in taxonomy_sets[1:]):
            raise PaperAggregationError(
                f"{split}: {key} membership drift across train seeds"
            )
        result: Dict[str, Any] = {}
        for taxonomy in sorted(taxonomy_sets[0]):
            counts = {int(detail[key][taxonomy]["n"]) for detail in details}
            if len(counts) != 1:
                raise PaperAggregationError(
                    f"{split}/{taxonomy}: {key} count drift across train seeds"
                )
            result[taxonomy] = {
                "records_n": next(iter(counts)),
                "metrics": {
                    metric: _mean_std(
                        detail[key][taxonomy][metric] for detail in details
                    )
                    for metric in TN_METRICS
                },
            }
        return result

    for split in TN_SPLITS:
        details = [
            run.tn_metric_details[split]
            for run in runs.values()
            if split in run.tn_metric_details
        ]
        if not details:
            continue
        tn_metric_aggregate[split] = {
            metric: _mean_std(
                detail["global"][metric]
                for detail in details
            )
            for metric in TN_METRICS
        }
        tn_taxonomy_aggregate[split] = aggregate_taxonomies(
            details, split=split, key="by_taxonomy_group"
        )
        tn_taxonomy_raw_aggregate[split] = aggregate_taxonomies(
            details, split=split, key="by_taxonomy"
        )
    return {
        "per_seed": per_seed,
        "aggregate": {
            "ref": ref_aggregate,
            "tn": tn_aggregate,
            "tn_metrics": tn_metric_aggregate,
            "tn_taxonomy": tn_taxonomy_aggregate,
            "tn_taxonomy_raw": tn_taxonomy_raw_aggregate,
        },
    }


def _validate_global_ref_identities(
    experiments: Mapping[str, Mapping[int, LoadedRun]],
    *,
    ref_splits: Sequence[str],
    context: _Context,
) -> None:
    """Require every seed and method to evaluate the same ordered Ref rows."""

    for split in ref_splits:
        canonical: Optional[RefRecords] = None
        canonical_label: Optional[str] = None
        for experiment_id, runs in experiments.items():
            for seed, run in sorted(runs.items()):
                records = run.ref_records.get(split)
                if records is None:
                    continue
                label = f"experiment {experiment_id} seed {seed} ref {split}"
                if canonical is None:
                    canonical = records
                    canonical_label = label
                    continue
                if records.identities != canonical.identities:
                    context.issue(
                        label,
                        "Ref record identity/order mismatch against "
                        f"{canonical_label}",
                    )
                    continue
                if not np.array_equal(records.image_ids, canonical.image_ids):
                    context.issue(
                        label,
                        f"Ref image identity/order mismatch against {canonical_label}",
                    )


def _comparison(
    baseline_runs: Mapping[int, LoadedRun],
    candidate_runs: Mapping[int, LoadedRun],
    *,
    ref_splits: Sequence[str],
    iterations: int,
    confidence: float,
    bootstrap_seed: int,
    candidate_id: str,
    baseline_reference_role: str,
    context: _Context,
) -> Dict[str, Any]:
    per_seed: Dict[str, Any] = {}
    ref_deltas: Dict[str, List[float]] = {split: [] for split in ref_splits}
    ref8_deltas: List[float] = []
    tn_deltas: Dict[str, List[float]] = {split: [] for split in TN_SPLITS}
    tn_metric_deltas: Dict[str, Dict[str, List[float]]] = {
        split: {metric: [] for metric in TN_METRICS}
        for split in TN_SPLITS
    }
    if baseline_reference_role == "fixed_historical_checkpoint":
        if len(baseline_runs) != 1:
            context.issue(
                f"comparison {candidate_id}",
                "fixed historical baseline must contain exactly one real run",
            )
            comparison_pairs: list[tuple[int, int, LoadedRun, LoadedRun]] = []
        else:
            baseline_seed, baseline_run = next(iter(sorted(baseline_runs.items())))
            comparison_pairs = [
                (candidate_seed, baseline_seed, baseline_run, candidate_run)
                for candidate_seed, candidate_run in sorted(candidate_runs.items())
            ]
        comparison_mode = "candidate_seeds_vs_fixed_historical_checkpoint"
    else:
        if set(baseline_runs) != set(candidate_runs):
            context.issue(
                f"comparison {candidate_id}",
                "training-seed distributions require identical seed sets",
            )
        comparison_pairs = [
            (seed, seed, baseline_runs[seed], candidate_runs[seed])
            for seed in sorted(set(baseline_runs) & set(candidate_runs))
        ]
        comparison_mode = "matched_training_seeds"

    shared_ref_bootstrap: Optional[Dict[str, Any]] = None
    shared_tn_bootstrap: Dict[str, Dict[str, Any]] = {}
    if (
        baseline_reference_role == "fixed_historical_checkpoint"
        and len(baseline_runs) == 1
        and candidate_runs
    ):
        fixed_baseline = next(iter(baseline_runs.values()))
        if all(
            split in fixed_baseline.ref_records
            and all(
                split in run.ref_records for run in candidate_runs.values()
            )
            for split in ref_splits
        ):
            try:
                shared_ref_bootstrap = paired_ref_seed_first_bootstrap(
                    fixed_baseline.ref_records,
                    {
                        train_seed: run.ref_records
                        for train_seed, run in candidate_runs.items()
                    },
                    ref_splits,
                    iterations=iterations,
                    confidence=confidence,
                    seed=_derived_seed(
                        bootstrap_seed, candidate_id, "headline", "ref_global"
                    ),
                )
            except PaperAggregationError as error:
                context.issue(
                    f"comparison {candidate_id} headline ref seed-first bootstrap",
                    error,
                )
        for split in TN_SPLITS:
            if split not in fixed_baseline.tn_records or not all(
                split in run.tn_records for run in candidate_runs.values()
            ):
                continue
            try:
                baseline_tn = fixed_baseline.tn_records[split]
                shared_tn_bootstrap[split] = paired_tn_seed_first_bootstrap(
                    baseline_tn,
                    {
                        train_seed: run.tn_records[split]
                        for train_seed, run in candidate_runs.items()
                    },
                    np.asarray(
                        [row.get("image_id") for row in baseline_tn.rows],
                        dtype=np.int64,
                    ),
                    iterations=iterations,
                    confidence=confidence,
                    seed=_derived_seed(
                        bootstrap_seed, candidate_id, "headline", "tn", split
                    ),
                )
            except (PaperAggregationError, RecordComparisonError, ValueError) as error:
                context.issue(
                    f"comparison {candidate_id} headline tn {split} "
                    "seed-first bootstrap",
                    error,
                )

    for seed, baseline_seed, base, cand in comparison_pairs:
        seed_report: Dict[str, Any] = {
            "candidate_train_seed": seed,
            "baseline_reference_seed": baseline_seed,
            "comparison_mode": comparison_mode,
            "ref": {"splits": {}},
            "tn": {},
        }
        for split in ref_splits:
            if split not in base.ref_records or split not in cand.ref_records:
                continue
            try:
                if shared_ref_bootstrap is not None:
                    result = shared_ref_bootstrap["per_seed"][str(seed)][
                        "splits"
                    ][split]
                else:
                    result = paired_accuracy_bootstrap(
                        base.ref_records[split],
                        cand.ref_records[split],
                        iterations=iterations,
                        confidence=confidence,
                        seed=_derived_seed(
                            bootstrap_seed, candidate_id, seed, "ref", split
                        ),
                    )
                seed_report["ref"]["splits"][split] = result
                ref_deltas[split].append(result["observed_candidate_minus_baseline"])
            except PaperAggregationError as error:
                context.issue(f"comparison {candidate_id} seed {seed} ref {split}", error)
        if all(split in base.ref_records and split in cand.ref_records for split in ref_splits):
            try:
                if shared_ref_bootstrap is not None:
                    result = shared_ref_bootstrap["per_seed"][str(seed)][
                        "mean8_acc50"
                    ]
                else:
                    result = paired_ref8_bootstrap(
                        base.ref_records,
                        cand.ref_records,
                        ref_splits,
                        iterations=iterations,
                        confidence=confidence,
                        seed=_derived_seed(
                            bootstrap_seed, candidate_id, seed, "ref8"
                        ),
                    )
                seed_report["ref"]["mean8_acc50"] = result
                ref8_deltas.append(result["observed_candidate_minus_baseline"])
            except PaperAggregationError as error:
                context.issue(f"comparison {candidate_id} seed {seed} ref8", error)
        for split in TN_SPLITS:
            if split not in base.tn_records or split not in cand.tn_records:
                continue
            try:
                baseline_tn = base.tn_records[split]
                candidate_tn = cand.tn_records[split]
                if not np.array_equal(baseline_tn.valid, candidate_tn.valid):
                    raise PaperAggregationError("TN valid-mask mismatch")
                valid = np.flatnonzero(baseline_tn.valid)
                manifest_images = np.asarray(
                    [row.get("image_id") for row in baseline_tn.rows], dtype=np.int64
                )
                if split in shared_tn_bootstrap:
                    shared_seed = shared_tn_bootstrap[split]["per_seed"][
                        str(seed)
                    ]
                    result = dict(shared_seed["fpr95"])
                    result["positive_q05_threshold_bootstrap"] = dict(
                        shared_seed["positive_q05_threshold"]
                    )
                else:
                    result = paired_fpr95_bootstrap(
                        baseline_tn.positive[valid],
                        baseline_tn.negative[valid],
                        candidate_tn.positive[valid],
                        candidate_tn.negative[valid],
                        manifest_images[valid],
                        iterations=iterations,
                        confidence=confidence,
                        seed=_derived_seed(
                            bootstrap_seed, candidate_id, seed, "tn", split
                        ),
                    )
                baseline_metrics = base.tn_metric_details[split]["global"]
                candidate_metrics = cand.tn_metric_details[split]["global"]
                result["observed_metric_deltas"] = {
                    metric: float(
                        candidate_metrics[metric] - baseline_metrics[metric]
                    )
                    for metric in TN_METRICS
                }
                seed_report["tn"][split] = result
                tn_deltas[split].append(
                    result["observed_candidate_minus_baseline_fpr95"]
                )
                for metric, delta in result["observed_metric_deltas"].items():
                    tn_metric_deltas[split][metric].append(delta)
            except (PaperAggregationError, RecordComparisonError, ValueError) as error:
                context.issue(f"comparison {candidate_id} seed {seed} tn {split}", error)
        per_seed[str(seed)] = seed_report

    across = {"ref": {"splits": {}}, "tn": {}, "tn_metrics": {}}
    for split, values in ref_deltas.items():
        if values:
            across["ref"]["splits"][split] = _mean_std(values)
    if ref8_deltas:
        across["ref"]["mean8_acc50"] = _mean_std(ref8_deltas)
    for split, values in tn_deltas.items():
        if values:
            across["tn"][split] = _mean_std(values)
    for split, metric_values in tn_metric_deltas.items():
        available = {
            metric: _mean_std(values)
            for metric, values in metric_values.items()
            if values
        }
        if available:
            across["tn_metrics"][split] = available
    headline_seed_first_bootstrap: Optional[Dict[str, Any]] = None
    if shared_ref_bootstrap is not None and set(shared_tn_bootstrap) == set(
        TN_SPLITS
    ):
        headline_seed_first_bootstrap = {
            "ref": shared_ref_bootstrap[
                "mean_across_candidate_train_seeds"
            ],
            "tn": {
                split: shared_tn_bootstrap[split][
                    "mean_across_candidate_train_seeds"
                ]
                for split in TN_SPLITS
            },
            "draw_contract": {
                "ref": {
                    key: shared_ref_bootstrap[key]
                    for key in (
                        "unit",
                        "draw_sharing",
                        "seed_aggregation",
                        "zero_split_draw_policy",
                        "candidate_train_seeds",
                        "iterations",
                        "confidence",
                        "seed",
                        "image_clusters_n",
                    )
                },
                "tn": {
                    split: {
                        key: shared_tn_bootstrap[split][key]
                        for key in (
                            "unit",
                            "draw_sharing",
                            "seed_aggregation",
                            "candidate_train_seeds",
                            "iterations",
                            "confidence",
                            "seed",
                            "image_clusters_n",
                            "recomputes_each_model_q05_per_resample",
                        )
                    }
                    for split in TN_SPLITS
                },
            },
        }
    acceptance = _headline_acceptance(
        headline_seed_first_bootstrap or {},
        provenance={
            "passed": False,
            "verdict": "not_yet_verified",
            "unmet": ["final_release_provenance_not_connected"],
        },
    )
    return {
        "comparison_mode": comparison_mode,
        "baseline_reference_seeds": sorted(baseline_runs),
        "candidate_train_seeds": sorted(candidate_runs),
        "baseline_is_not_pseudo_replicated": (
            baseline_reference_role == "fixed_historical_checkpoint"
        ),
        "per_seed": per_seed,
        "observed_delta_across_train_seeds": across,
        "headline_seed_first_bootstrap": headline_seed_first_bootstrap,
        "headline_acceptance": acceptance,
    }


def aggregate_manifest(
    manifest_path: str | Path,
    *,
    allow_incomplete: bool = False,
    bootstrap_iterations: Optional[int] = None,
    confidence: Optional[float] = None,
    bootstrap_seed: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    base_dir = path.parent
    manifest_file = _artifact_record(
        str(path), base_dir, label="experiment manifest", always_hash=True
    )
    payload = _read_json(path, label="experiment manifest")
    _assert_artifact_unchanged(manifest_file, label="experiment manifest")
    if not isinstance(payload, Mapping):
        raise PaperAggregationError("experiment manifest root must be an object")
    if payload.get("schema") != SCHEMA:
        raise PaperAggregationError(f"manifest schema must be exactly {SCHEMA!r}")
    context = _Context(allow_incomplete)

    raw_seeds = payload.get("expected_train_seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise PaperAggregationError("expected_train_seeds must be a non-empty list")
    expected_seeds = tuple(
        _required_int(seed, label="expected_train_seeds") for seed in raw_seeds
    )
    if len(set(expected_seeds)) != len(expected_seeds):
        raise PaperAggregationError("expected_train_seeds contains duplicates")

    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise PaperAggregationError("protocol must be an object")
    ref_splits = protocol.get("ref_splits")
    if not isinstance(ref_splits, list) or tuple(ref_splits) != REF_SPLITS:
        raise PaperAggregationError(
            f"protocol.ref_splits must exactly equal the fixed eight-split order {list(REF_SPLITS)}"
        )
    raw_tn = protocol.get("tn_splits")
    if not isinstance(raw_tn, Mapping) or set(raw_tn) != set(TN_SPLITS):
        raise PaperAggregationError(
            f"protocol.tn_splits must contain exactly {list(TN_SPLITS)}"
        )
    tn_manifests: Dict[str, ManifestRows] = {}
    tn_manifest_inputs: Dict[str, Any] = {}
    for split in TN_SPLITS:
        try:
            specification = raw_tn[split]
            if not isinstance(specification, Mapping):
                raise PaperAggregationError("TN protocol entry must be an object")
            record = _artifact_record(
                specification.get("manifest"),
                base_dir,
                label=f"protocol.tn_splits.{split}.manifest",
                always_hash=True,
            )
            loaded = load_manifest(record["path"])
            _assert_artifact_unchanged(
                record, label=f"protocol.tn_splits.{split}.manifest"
            )
            if dict(loaded.file_record) != {
                key: record[key] for key in ("path", "size_bytes", "sha256")
            }:
                raise PaperAggregationError(
                    "TN manifest changed while being parsed"
                )
            expected_n = _required_int(
                specification.get("expected_n"),
                label=f"protocol.tn_splits.{split}.expected_n",
            )
            if expected_n <= 0 or len(loaded.rows) != expected_n:
                raise PaperAggregationError(
                    f"manifest rows={len(loaded.rows)} != expected_n={expected_n}"
                )
            tn_manifests[split] = loaded
            tn_manifest_inputs[split] = {**record, "rows": len(loaded.rows)}
        except (PaperAggregationError, RecordComparisonError, OSError, ValueError) as error:
            context.issue(f"protocol TN {split}", error)

    bootstrap = protocol.get("bootstrap", {})
    if not isinstance(bootstrap, Mapping):
        raise PaperAggregationError("protocol.bootstrap must be an object")
    iterations = int(
        bootstrap_iterations
        if bootstrap_iterations is not None
        else bootstrap.get("iterations", 5000)
    )
    confidence_value = float(
        confidence if confidence is not None else bootstrap.get("confidence", 0.95)
    )
    bootstrap_seed_value = int(
        bootstrap_seed if bootstrap_seed is not None else bootstrap.get("seed", 20260717)
    )
    if iterations <= 0 or not 0.0 < confidence_value < 1.0:
        raise PaperAggregationError("bootstrap iterations/confidence are invalid")

    baseline_id = payload.get("baseline_experiment")
    if not isinstance(baseline_id, str) or not baseline_id:
        raise PaperAggregationError("baseline_experiment must be a non-empty string")
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise PaperAggregationError("experiments must be a non-empty list")

    loaded_experiments: Dict[str, Dict[int, LoadedRun]] = {}
    labels: Dict[str, str] = {}
    experiment_seed_contracts: Dict[str, tuple[int, ...]] = {}
    experiment_reference_roles: Dict[str, str] = {}
    for experiment in raw_experiments:
        if not isinstance(experiment, Mapping):
            context.issue("experiments", "experiment entry must be an object")
            continue
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            context.issue("experiments", "experiment id must be a non-empty string")
            continue
        if experiment_id in loaded_experiments:
            context.issue(f"experiment {experiment_id}", "duplicate experiment id")
            continue
        labels[experiment_id] = str(experiment.get("label") or experiment_id)
        raw_experiment_seeds = experiment.get(
            "expected_train_seeds", list(expected_seeds)
        )
        if not isinstance(raw_experiment_seeds, list) or not raw_experiment_seeds:
            context.issue(
                f"experiment {experiment_id}.expected_train_seeds",
                "must be a non-empty list",
            )
            experiment_seeds = expected_seeds
        else:
            experiment_seeds = tuple(
                _required_int(
                    seed,
                    label=f"experiment {experiment_id}.expected_train_seeds",
                )
                for seed in raw_experiment_seeds
            )
            if len(set(experiment_seeds)) != len(experiment_seeds):
                context.issue(
                    f"experiment {experiment_id}.expected_train_seeds",
                    "contains duplicates",
                )
        experiment_seed_contracts[experiment_id] = tuple(experiment_seeds)
        experiment_seed_set = set(experiment_seeds)
        reference_role = str(
            experiment.get("reference_role") or "training_seed_distribution"
        )
        if reference_role not in {
            "training_seed_distribution",
            "fixed_historical_checkpoint",
        }:
            context.issue(
                f"experiment {experiment_id}.reference_role",
                f"invalid value {reference_role!r}",
            )
            reference_role = "training_seed_distribution"
        if reference_role == "fixed_historical_checkpoint" and (
            experiment_id != baseline_id or len(experiment_seeds) != 1
        ):
            context.issue(
                f"experiment {experiment_id}.reference_role",
                "fixed historical reference requires the declared baseline and one seed",
            )
        experiment_reference_roles[experiment_id] = reference_role
        raw_runs = experiment.get("runs")
        if not isinstance(raw_runs, list):
            context.issue(f"experiment {experiment_id}", "runs must be a list")
            loaded_experiments[experiment_id] = {}
            continue
        by_seed: Dict[int, LoadedRun] = {}
        declared_seeds: List[int] = []
        for index, run_spec in enumerate(raw_runs):
            if not isinstance(run_spec, Mapping):
                context.issue(
                    f"experiment {experiment_id} run {index}", "run must be an object"
                )
                continue
            try:
                seed = _required_int(
                    run_spec.get("train_seed"),
                    label=f"experiment {experiment_id} run {index}.train_seed",
                )
                declared_seeds.append(seed)
                if seed in by_seed:
                    raise PaperAggregationError(f"duplicate train seed {seed}")
                if seed not in experiment_seed_set:
                    raise PaperAggregationError(
                        f"unexpected train seed {seed}; "
                        f"expected {sorted(experiment_seed_set)}"
                    )
                by_seed[seed] = _load_run(
                    run_spec,
                    experiment_id=experiment_id,
                    base_dir=base_dir,
                    ref_splits=REF_SPLITS,
                    tn_manifests=tn_manifests,
                    context=context,
                )
            except (PaperAggregationError, OSError, ValueError) as error:
                context.issue(f"experiment {experiment_id} run {index}", error)
        missing = sorted(experiment_seed_set - set(declared_seeds))
        extras = sorted(set(declared_seeds) - experiment_seed_set)
        if missing or extras:
            context.issue(
                f"experiment {experiment_id}.seeds",
                f"seed contract mismatch: missing={missing}, extra={extras}",
            )
        loaded_experiments[experiment_id] = by_seed

    if baseline_id not in loaded_experiments:
        context.issue("baseline_experiment", f"unknown experiment {baseline_id!r}")

    _validate_global_ref_identities(
        loaded_experiments,
        ref_splits=REF_SPLITS,
        context=context,
    )

    experiment_reports = {
        experiment_id: {
            "label": labels.get(experiment_id, experiment_id),
            "expected_train_seeds": list(
                experiment_seed_contracts.get(experiment_id, ())
            ),
            "reference_role": experiment_reference_roles.get(
                experiment_id, "training_seed_distribution"
            ),
            **_aggregate_experiment(runs, REF_SPLITS),
        }
        for experiment_id, runs in loaded_experiments.items()
    }
    comparisons: Dict[str, Any] = {}
    baseline_runs = loaded_experiments.get(str(baseline_id), {})
    for experiment_id, runs in loaded_experiments.items():
        if experiment_id == baseline_id:
            continue
        comparisons[experiment_id] = _comparison(
            baseline_runs,
            runs,
            ref_splits=REF_SPLITS,
            iterations=iterations,
            confidence=confidence_value,
            bootstrap_seed=bootstrap_seed_value,
            candidate_id=experiment_id,
            baseline_reference_role=experiment_reference_roles.get(
                str(baseline_id), "training_seed_distribution"
            ),
            context=context,
        )

    status = "incomplete" if context.issues else "complete"
    if status == "incomplete" and not allow_incomplete:
        raise PaperAggregationError("aggregation is incomplete")
    from tools import stageb_headline_release_contract as headline_release

    try:
        headline_provenance = headline_release.verify_manifest_release_provenance(
            protocol.get("headline_release_provenance"),
            experiments=raw_experiments,
            baseline_id=str(baseline_id),
            bootstrap={
                "iterations": iterations,
                "confidence": confidence_value,
                "seed": bootstrap_seed_value,
            },
        )
        headline_provenance["data_contract_status"] = status
        if status != "complete":
            raise headline_release.HeadlineReleaseError(
                "metric/data aggregation is incomplete"
            )
    except (
        headline_release.HeadlineReleaseError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
    ) as error:
        headline_provenance = {
            "passed": False,
            "verdict": "unavailable_until_receipt_chain_is_verified",
            "data_contract_status": status,
            "unmet": [
                "one_time_final_evaluation_gate_receipt",
                "validation_only_fixed_m0_eligibility_receipt",
                "fixed_b58_baseline_identity_and_record_consistency_contract",
                "candidate_m0_three_seed_compute_matched_queue_checkpoint_contract",
                "b58_successful_update_batch_slot_derivation_receipt",
                "paper_ablation_completion_receipt",
                "cross_model_evaluation_runtime_code_data_parity_contract",
                "fresh_final_artifact_containment_and_postrun_rehash_contract",
                "formal_5000_draw_bootstrap_contract",
            ],
            "error": f"{type(error).__name__}: {error}",
            "note": (
                "A complete metric manifest is not final-release provenance. "
                "Headline acceptance remains unavailable until the evaluator and "
                "results builder bind and verify every listed receipt."
            ),
        }
    for comparison in comparisons.values():
        comparison["headline_acceptance"] = _headline_acceptance(
            comparison.get("headline_seed_first_bootstrap") or {},
            provenance=headline_provenance,
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "watermark": (
            "INCOMPLETE PROGRESS REPORT - NOT FOR PAPER TABLES"
            if status == "incomplete"
            else None
        ),
        "headline_watermark": (
            None
            if headline_provenance["passed"]
            else "HEADLINE ACCEPTANCE UNAVAILABLE - FINAL RELEASE PROVENANCE NOT VERIFIED"
        ),
        "allow_incomplete": bool(allow_incomplete),
        "validation": {
            "pass": status == "complete",
            "issues": context.issues,
            "expected_train_seeds": list(expected_seeds),
            "experiment_seed_contracts": {
                experiment_id: list(seeds)
                for experiment_id, seeds in experiment_seed_contracts.items()
            },
            "experiment_reference_roles": dict(experiment_reference_roles),
            "required_ref_splits": list(REF_SPLITS),
            "required_tn_splits": list(TN_SPLITS),
        },
        "headline_provenance": headline_provenance,
        "protocol": {
            "ref8_aggregation": "unweighted split mean, then mean/std across train seeds",
            "train_seed_std_ddof": 1,
            "fpr95": "exact positive q05 order statistic; score >= threshold",
            "tn_secondary_metrics": {
                "fpr90": "exact positive 90%-acceptance order statistic; score >= threshold",
                "threshold_at_95tpr": "the exact per-model positive-tail threshold used for FPR95",
                "auroc": "Mann-Whitney probability with half credit for exact ties",
                "pair_win_rate": "mean(pos_score > neg_score) on aligned counterfactual pairs",
                "taxonomy": (
                    "fixed semantic groups derived from replace_category in the "
                    "SHA-bound evaluation manifest; raw groups remain in JSON"
                ),
            },
            "paired_bootstrap": {
                "unit": "canonical image cluster",
                "iterations": iterations,
                "confidence": confidence_value,
                "seed": bootstrap_seed_value,
                "tn_recomputes_each_model_q05_per_resample": True,
                "headline_seed_aggregation": (
                    "per_seed paired delta then equal-weight seed mean; "
                    "training seeds are not resampled"
                ),
                "ref8_global_cluster_scope": (
                    "one canonical COCO image draw shared across all eight "
                    "Ref splits, baseline, and candidate train seeds"
                ),
            },
            "headline_acceptance": {
                "ref_split_noninferiority_margin": (
                    REF_SPLIT_NONINFERIORITY_MARGIN
                ),
                "positive_q05_noninferiority_margin": (
                    POSITIVE_Q05_NONINFERIORITY_MARGIN
                ),
            },
        },
        "input_files": {
            "experiment_manifest": manifest_file,
            "tn_manifests": tn_manifest_inputs,
        },
        "baseline_experiment": baseline_id,
        "experiments": experiment_reports,
        "comparisons_to_baseline": comparisons,
    }


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Stage-B Paper Results", ""]
    if report.get("status") != "complete":
        lines.extend(
            [
                "> **INCOMPLETE PROGRESS REPORT - NOT FOR PAPER TABLES**",
                "",
            ]
        )
    if not report.get("headline_provenance", {}).get("passed"):
        lines.extend(
            [
                "> **HEADLINE ACCEPTANCE UNAVAILABLE - FINAL RELEASE PROVENANCE NOT VERIFIED**",
                "",
            ]
        )
    lines.extend(
        [
            f"Status: **{str(report.get('status', 'unknown')).upper()}**",
            "",
            "| experiment | seeds | Ref8 Acc50 mean +/- std | strict2031 FPR95 | strict1607 FPR95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for experiment_id, experiment in report.get("experiments", {}).items():
        aggregate = experiment.get("aggregate", {})
        ref8 = aggregate.get("ref", {}).get("mean8_acc50", {})
        tn = aggregate.get("tn", {})
        seed_n = ref8.get("n", max((row.get("n", 0) for row in tn.values()), default=0))
        ref_text = f"{_fmt(ref8.get('mean'))} +/- {_fmt(ref8.get('std'))}"
        tn_text = []
        for split in TN_SPLITS:
            values = tn.get(split, {})
            tn_text.append(f"{_fmt(values.get('mean'))} +/- {_fmt(values.get('std'))}")
        lines.append(
            f"| {experiment.get('label', experiment_id)} | {seed_n} | {ref_text} | "
            f"{tn_text[0]} | {tn_text[1]} |"
        )
    lines.extend(
        [
            "",
            "## Headline Seed-First Gate",
            "",
            "| candidate | accepted | Ref8 delta [95% CI] | strict2031 FPR95 delta [95% CI] | strict1607 FPR95 delta [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate, comparison in report.get(
        "comparisons_to_baseline", {}
    ).items():
        headline = comparison.get("headline_seed_first_bootstrap")
        acceptance = comparison.get("headline_acceptance", {})
        if not isinstance(headline, Mapping):
            lines.append(f"| {candidate} | false | NA | NA | NA |")
            continue
        ref8 = headline["ref"]["mean8_acc50"]
        tn = headline["tn"]

        def interval(values: Mapping[str, Any], observed_key: str) -> str:
            return (
                f"{_fmt(values.get(observed_key))} "
                f"[{_fmt(values.get('delta_ci_low'))}, "
                f"{_fmt(values.get('delta_ci_high'))}]"
            )

        lines.append(
            f"| {candidate} | {str(bool(acceptance.get('pass'))).lower()} | "
            f"{interval(ref8, 'observed_candidate_minus_baseline')} | "
            f"{interval(tn['strict2031']['fpr95'], 'observed_candidate_minus_baseline_fpr95')} | "
            f"{interval(tn['strict1607']['fpr95'], 'observed_candidate_minus_baseline_fpr95')} |"
        )
    lines.extend(
        [
            "",
            "## RefCOCO Splits",
            "",
            "| experiment | split | Acc50 mean | std | seeds |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for experiment_id, experiment in report.get("experiments", {}).items():
        for split in REF_SPLITS:
            values = experiment.get("aggregate", {}).get("ref", {}).get("splits", {}).get(split)
            if not values:
                continue
            lines.append(
                f"| {experiment.get('label', experiment_id)} | {split} | "
                f"{_fmt(values.get('mean'))} | {_fmt(values.get('std'))} | {values.get('n')} |"
            )
    lines.extend(
        [
            "",
            "## TN Secondary Metrics",
            "",
            "| experiment | split | FPR90 | q05 threshold | AUROC | pair win | mean pair gap |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment_id, experiment in report.get("experiments", {}).items():
        aggregate = experiment.get("aggregate", {}).get("tn_metrics", {})
        for split in TN_SPLITS:
            values = aggregate.get(split)
            if not values:
                continue
            lines.append(
                f"| {experiment.get('label', experiment_id)} | {split} | "
                f"{_fmt(values['fpr90'].get('mean'))} | "
                f"{_fmt(values['threshold_at_95tpr'].get('mean'))} | "
                f"{_fmt(values['auroc'].get('mean'))} | "
                f"{_fmt(values['pair_win_rate'].get('mean'))} | "
                f"{_fmt(values['paired_score_gap_mean'].get('mean'))} |"
            )
    lines.extend(
        [
            "",
            "## TN Taxonomy",
            "",
            "| experiment | split | taxonomy | N | FPR95 | FPR90 | AUROC | pair win |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment_id, experiment in report.get("experiments", {}).items():
        taxonomy = experiment.get("aggregate", {}).get("tn_taxonomy", {})
        for split in TN_SPLITS:
            for name, values in taxonomy.get(split, {}).items():
                metrics = values["metrics"]
                lines.append(
                    f"| {experiment.get('label', experiment_id)} | {split} | {name} | "
                    f"{values['records_n']} | {_fmt(metrics['fpr95'].get('mean'))} | "
                    f"{_fmt(metrics['fpr90'].get('mean'))} | "
                    f"{_fmt(metrics['auroc'].get('mean'))} | "
                    f"{_fmt(metrics['pair_win_rate'].get('mean'))} |"
                )
    lines.extend(["", "## Paired Deltas", ""])
    baseline_id = report.get("baseline_experiment")
    lines.append(f"Baseline: `{baseline_id}`. Deltas are candidate minus baseline.")
    lines.extend(
        [
            "",
            "| candidate | train seed | metric | observed delta | CI low | CI high |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for candidate, comparison in report.get("comparisons_to_baseline", {}).items():
        for seed, seed_report in comparison.get("per_seed", {}).items():
            ref8 = seed_report.get("ref", {}).get("mean8_acc50")
            if ref8:
                lines.append(
                    f"| {candidate} | {seed} | Ref8 Acc50 | "
                    f"{_fmt(ref8.get('observed_candidate_minus_baseline'))} | "
                    f"{_fmt(ref8.get('delta_ci_low'))} | {_fmt(ref8.get('delta_ci_high'))} |"
                )
            for split in TN_SPLITS:
                values = seed_report.get("tn", {}).get(split)
                if values:
                    lines.append(
                        f"| {candidate} | {seed} | {split} FPR95 | "
                        f"{_fmt(values.get('observed_candidate_minus_baseline_fpr95'))} | "
                        f"{_fmt(values.get('delta_ci_low'))} | {_fmt(values.get('delta_ci_high'))} |"
                    )
    issues = report.get("validation", {}).get("issues", [])
    if issues:
        lines.extend(["", "## Incomplete Contracts", ""])
        for issue in issues:
            lines.append(f"- `{issue.get('scope')}`: {issue.get('message')}")
    return "\n".join(lines).rstrip() + "\n"


CSV_COLUMNS = (
    "report_status",
    "level",
    "experiment",
    "baseline",
    "train_seed",
    "family",
    "split",
    "metric",
    "value",
    "mean",
    "std",
    "n",
    "ci_low",
    "ci_high",
    "bootstrap_iterations",
)


def render_csv(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    status = str(report.get("status"))
    baseline = str(report.get("baseline_experiment"))
    writer.writerow(
        dict(
            report_status=status,
            level="report",
            baseline=baseline,
            metric="status",
            value=status,
        )
    )
    for experiment_id, experiment in report.get("experiments", {}).items():
        for seed, seed_row in experiment.get("per_seed", {}).items():
            for split, value in seed_row.get("ref", {}).get("splits", {}).items():
                writer.writerow(
                    dict(
                        report_status=status,
                        level="seed",
                        experiment=experiment_id,
                        baseline=baseline,
                        train_seed=seed,
                        family="ref",
                        split=split,
                        metric="acc50",
                        value=value,
                    )
                )
            ref8 = seed_row.get("ref", {}).get("mean8_acc50")
            if ref8 is not None:
                writer.writerow(
                    dict(
                        report_status=status,
                        level="seed",
                        experiment=experiment_id,
                        baseline=baseline,
                        train_seed=seed,
                        family="ref",
                        split="mean8",
                        metric="acc50",
                        value=ref8,
                    )
                )
            for split, value in seed_row.get("tn", {}).items():
                writer.writerow(
                    dict(
                        report_status=status,
                        level="seed",
                        experiment=experiment_id,
                        baseline=baseline,
                        train_seed=seed,
                        family="tn",
                        split=split,
                        metric="fpr95",
                        value=value,
                    )
                )
            for split, details in seed_row.get("tn_metrics", {}).items():
                global_metrics = details.get("global", {})
                for metric in TN_METRICS:
                    if metric == "fpr95" or metric not in global_metrics:
                        continue
                    writer.writerow(
                        dict(
                            report_status=status,
                            level="seed",
                            experiment=experiment_id,
                            baseline=baseline,
                            train_seed=seed,
                            family="tn",
                            split=split,
                            metric=metric,
                            value=global_metrics[metric],
                            n=global_metrics.get("n"),
                        )
                    )
        aggregate = experiment.get("aggregate", {})
        aggregate_rows = []
        for split, values in aggregate.get("ref", {}).get("splits", {}).items():
            aggregate_rows.append(("ref", split, "acc50", values))
        if "mean8_acc50" in aggregate.get("ref", {}):
            aggregate_rows.append(("ref", "mean8", "acc50", aggregate["ref"]["mean8_acc50"]))
        for split, values in aggregate.get("tn", {}).items():
            aggregate_rows.append(("tn", split, "fpr95", values))
        for split, metrics in aggregate.get("tn_metrics", {}).items():
            for metric, values in metrics.items():
                if metric != "fpr95":
                    aggregate_rows.append(("tn", split, metric, values))
        for family, split, metric, values in aggregate_rows:
            writer.writerow(
                dict(
                    report_status=status,
                    level="aggregate",
                    experiment=experiment_id,
                    baseline=baseline,
                    family=family,
                    split=split,
                    metric=metric,
                    mean=values.get("mean"),
                    std=values.get("std"),
                    n=values.get("n"),
                )
            )
        for split, groups in aggregate.get("tn_taxonomy", {}).items():
            for taxonomy, group in groups.items():
                for metric, values in group.get("metrics", {}).items():
                    writer.writerow(
                        dict(
                            report_status=status,
                            level="aggregate",
                            experiment=experiment_id,
                            baseline=baseline,
                            family="tn_taxonomy",
                            split=f"{split}:{taxonomy}",
                            metric=metric,
                            mean=values.get("mean"),
                            std=values.get("std"),
                            n=group.get("records_n"),
                        )
                    )
    iterations = report.get("protocol", {}).get("paired_bootstrap", {}).get("iterations")
    for candidate, comparison in report.get("comparisons_to_baseline", {}).items():
        headline = comparison.get("headline_seed_first_bootstrap")
        if isinstance(headline, Mapping):
            headline_rows = [
                (
                    "ref",
                    "mean8",
                    "acc50_delta",
                    headline["ref"]["mean8_acc50"],
                    "observed_candidate_minus_baseline",
                ),
                *[
                    (
                        "ref",
                        split,
                        "acc50_delta",
                        values,
                        "observed_candidate_minus_baseline",
                    )
                    for split, values in headline["ref"]["splits"].items()
                ],
                *[
                    (
                        "tn",
                        split,
                        metric,
                        values[metric_key],
                        value_key,
                    )
                    for split, values in headline["tn"].items()
                    for metric, metric_key, value_key in (
                        (
                            "fpr95_delta",
                            "fpr95",
                            "observed_candidate_minus_baseline_fpr95",
                        ),
                        (
                            "positive_q05_threshold_delta",
                            "positive_q05_threshold",
                            "observed_candidate_minus_baseline",
                        ),
                    )
                ],
            ]
            for family, split, metric, values, value_key in headline_rows:
                writer.writerow(
                    dict(
                        report_status=status,
                        level="headline_seed_first_bootstrap",
                        experiment=candidate,
                        baseline=baseline,
                        train_seed="seed_mean",
                        family=family,
                        split=split,
                        metric=metric,
                        value=values.get(value_key),
                        ci_low=values.get("delta_ci_low"),
                        ci_high=values.get("delta_ci_high"),
                        bootstrap_iterations=iterations,
                    )
                )
        acceptance = comparison.get("headline_acceptance", {})
        writer.writerow(
            dict(
                report_status=status,
                level="headline_acceptance",
                experiment=candidate,
                baseline=baseline,
                family="all",
                split="headline",
                metric="pass",
                value=bool(acceptance.get("pass")),
            )
        )
        for seed, seed_report in comparison.get("per_seed", {}).items():
            bootstrap_rows = []
            for split, values in seed_report.get("ref", {}).get("splits", {}).items():
                bootstrap_rows.append(("ref", split, "acc50_delta", values, "observed_candidate_minus_baseline"))
            if "mean8_acc50" in seed_report.get("ref", {}):
                bootstrap_rows.append(("ref", "mean8", "acc50_delta", seed_report["ref"]["mean8_acc50"], "observed_candidate_minus_baseline"))
            for split, values in seed_report.get("tn", {}).items():
                bootstrap_rows.append(("tn", split, "fpr95_delta", values, "observed_candidate_minus_baseline_fpr95"))
            for family, split, metric, values, value_key in bootstrap_rows:
                writer.writerow(
                    dict(
                        report_status=status,
                        level="paired_bootstrap",
                        experiment=candidate,
                        baseline=baseline,
                        train_seed=seed,
                        family=family,
                        split=split,
                        metric=metric,
                        value=values.get(value_key),
                        ci_low=values.get("delta_ci_low"),
                        ci_high=values.get("delta_ci_high"),
                        bootstrap_iterations=iterations,
                    )
                )
                if family == "tn":
                    for metric, delta in values.get(
                        "observed_metric_deltas", {}
                    ).items():
                        if metric == "fpr95":
                            continue
                        writer.writerow(
                            dict(
                                report_status=status,
                                level="observed_paired_delta",
                                experiment=candidate,
                                baseline=baseline,
                                train_seed=seed,
                                family="tn",
                                split=split,
                                metric=f"{metric}_delta",
                                value=delta,
                            )
                        )
    return buffer.getvalue()


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> Dict[str, str]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    outputs = {
        "json": directory / "paper_results.json",
        "markdown": directory / "paper_results.md",
        "csv": directory / "paper_results.csv",
    }
    _write_atomic(
        outputs["json"],
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_atomic(outputs["markdown"], render_markdown(report))
    _write_atomic(outputs["csv"], render_csv(report))
    return {key: str(path) for key, path in outputs.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_cvpr_v1/aggregate",
        help="Writes paper_results.{json,md,csv}.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a watermarked progress report instead of failing on missing contracts.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = aggregate_manifest(
            args.manifest,
            allow_incomplete=bool(args.allow_incomplete),
            bootstrap_iterations=args.bootstrap_iterations,
            confidence=args.confidence,
            bootstrap_seed=args.bootstrap_seed,
        )
        outputs = write_report(report, args.output_dir)
    except (
        FileExistsError,
        PaperAggregationError,
        RecordComparisonError,
        OSError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "outputs": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
