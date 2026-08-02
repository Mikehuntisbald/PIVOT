#!/usr/bin/env python3
"""Strict paired acceptance gate for Stage-B FPR95 and RefCOCO acc50."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np


DEFAULT_REF_SPLITS = [
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_val",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_val",
    "refcocog_test",
]


class GateValidationError(ValueError):
    pass


def exact_fpr_at_tpr(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    target_tpr: float = 0.95,
) -> Dict[str, float]:
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0 or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise GateValidationError("FPR requires non-empty, finite positive and negative score arrays")
    target = min(1.0, max(0.0, float(target_tpr)))
    if target <= 0.0:
        threshold = float(np.nextafter(pos.max(), np.inf))
    else:
        accepted = max(1, int(math.ceil(target * int(pos.size))))
        ascending_index = int(pos.size) - accepted
        threshold = float(np.partition(pos, ascending_index)[ascending_index])
    return {
        "threshold": threshold,
        "actual_tpr": float(np.mean(pos >= threshold)),
        "fpr": float(np.mean(neg >= threshold)),
    }


def _expand_paths(paths: Sequence[str]) -> List[Path]:
    expanded: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise GateValidationError(f"Record path does not exist: {path}")
        if path.is_dir():
            matches = sorted(path.rglob("*.records.jsonl"))
            if not matches:
                raise GateValidationError(f"Directory contains no *.records.jsonl files: {path}")
            expanded.extend(matches)
        else:
            expanded.append(path)
    if not expanded:
        raise GateValidationError("No record files were provided")
    return expanded


def _records_from_json_payload(payload: Any, path: Path) -> List[Dict[str, Any]]:
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        if _looks_like_record(payload):
            return [dict(payload)]
        for key in ("records", "per_example_records", "eval_records"):
            rows = payload.get(key)
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                return [dict(row) for row in rows]
    raise GateValidationError(
        f"{path} has no per-example records. Expected JSONL objects, a JSON list, or a JSON object "
        "with records/per_example_records/eval_records. Aggregated summaries are insufficient."
    )


def _looks_like_record(row: Mapping[str, Any]) -> bool:
    return "sample_id" in row and (
        "top1_iou" in row
        or "correct50" in row
        or "pos_score" in row
        or "positive_score" in row
    )


def _read_path_with_record(
    path: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        raw = path.read_bytes()
        rendered = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise GateValidationError(f"Could not read records from {path}: {error}") from error
    file_record = {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line_number, line in enumerate(rendered.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateValidationError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise GateValidationError(f"Expected an object at {path}:{line_number}")
            rows.append(dict(row))
        return rows, file_record
    try:
        payload = json.loads(rendered)
    except json.JSONDecodeError as error:
        raise GateValidationError(f"Could not read JSON records from {path}: {error}") from error
    return _records_from_json_payload(payload, path), file_record


def _read_path(path: Path) -> List[Dict[str, Any]]:
    rows, _ = _read_path_with_record(path)
    return rows


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _normalize_record(row: Mapping[str, Any], *, source: Path, source_order: int) -> Dict[str, Any]:
    task = str(row.get("task", "")).lower().strip()
    if not task:
        task = "tn" if _first(row, ("pos_score", "positive_score")) is not None else "ref"
    if task not in {"tn", "ref"}:
        raise GateValidationError(f"Unsupported task {task!r} in {source}")
    split = str(_first(row, ("split", "dataset", "eval_split"), "")).strip()
    if task == "ref" and not split:
        raise GateValidationError(f"Ref record has no split/dataset in {source}")
    manifest_key = str(row.get("manifest_key") or ("tn_global" if task == "tn" else f"ref:{split}"))
    manifest_hash = str(_first(row, ("manifest_sha256", "manifest_hash"), "")).lower()
    sample_id = str(_first(row, ("sample_id", "id"), ""))
    manifest_n_raw = _first(row, ("manifest_n", "manifest_size", "num_manifest_records"))
    image_id_raw = row.get("image_id")
    if not manifest_hash or not sample_id or manifest_n_raw is None or image_id_raw is None:
        raise GateValidationError(
            f"Record in {source} is missing one of manifest_sha256, manifest_n, sample_id, image_id"
        )
    if len(manifest_hash) != 64 or any(char not in "0123456789abcdef" for char in manifest_hash):
        raise GateValidationError(f"Record in {source} has an invalid SHA-256 manifest hash")
    try:
        manifest_n = int(manifest_n_raw)
        image_id = int(image_id_raw)
        manifest_index = int(row.get("manifest_index", source_order))
    except (TypeError, ValueError) as error:
        raise GateValidationError(f"Invalid integer metadata in {source}: {error}") from error

    normalized: Dict[str, Any] = {
        "task": task,
        "split": split,
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_hash,
        "manifest_n": manifest_n,
        "manifest_index": manifest_index,
        "sample_id": sample_id,
        "image_id": image_id,
        "valid": row.get("valid") is True,
        "source": str(source),
    }
    if task == "tn":
        normalized["pos_score"] = _finite_float(_first(row, ("pos_score", "positive_score")))
        normalized["neg_score"] = _finite_float(_first(row, ("neg_score", "tn_score", "negative_score")))
        normalized["valid"] = bool(
            normalized["valid"]
            and normalized["pos_score"] is not None
            and normalized["neg_score"] is not None
        )
    else:
        iou = _finite_float(_first(row, ("top1_iou", "iou", "iou_top1")))
        correct = row.get("correct50")
        if correct is None and iou is not None:
            correct = iou >= 0.5
        if not isinstance(correct, (bool, np.bool_)):
            correct = None
        normalized["top1_iou"] = iou
        normalized["correct50"] = None if correct is None else bool(correct)
        all_query_best_iou = _finite_float(row.get("all_query_best_iou"))
        if all_query_best_iou is not None:
            normalized["all_query_best_iou"] = all_query_best_iou
        normalized["valid"] = bool(normalized["valid"] and normalized["correct50"] is not None)
    return normalized


def load_record_set_with_inputs(
    paths: Sequence[str],
) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    groups: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    group_offsets: Counter[str] = Counter()
    input_files: List[Dict[str, Any]] = []
    for path in _expand_paths(paths):
        rows, file_record = _read_path_with_record(path)
        input_files.append(file_record)
        for row in rows:
            provisional_task = str(row.get("task", "")).lower().strip()
            if not provisional_task:
                provisional_task = (
                    "tn" if _first(row, ("pos_score", "positive_score")) is not None else "ref"
                )
            provisional_split = str(_first(row, ("split", "dataset", "eval_split"), "")).strip()
            provisional_key = str(
                row.get("manifest_key")
                or ("tn_global" if provisional_task == "tn" else f"ref:{provisional_split}")
            )
            normalized = _normalize_record(
                row,
                source=path,
                source_order=group_offsets[provisional_key],
            )
            group_offsets[normalized["manifest_key"]] += 1
            groups[normalized["manifest_key"]].append(normalized)
    return dict(groups), input_files


def load_record_set(paths: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    groups, _ = load_record_set_with_inputs(paths)
    return groups


def _validate_group(label: str, key: str, records: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    if not records:
        return [f"{label}/{key}: empty record group"]
    hashes = {str(row["manifest_sha256"]) for row in records}
    sizes = {int(row["manifest_n"]) for row in records}
    ids = [str(row["sample_id"]) for row in records]
    indices = [int(row["manifest_index"]) for row in records]
    invalid = sum(not bool(row["valid"]) for row in records)
    duplicates = len(ids) - len(set(ids))
    tasks = {str(row["task"]) for row in records}
    splits = {str(row["split"]) for row in records}
    if len(tasks) != 1:
        errors.append(f"{label}/{key}: mixed tasks: {sorted(tasks)}")
    elif key == "tn_global" and tasks != {"tn"}:
        errors.append(f"{label}/{key}: expected TN records, got {sorted(tasks)}")
    elif key.startswith("ref:") and tasks != {"ref"}:
        errors.append(f"{label}/{key}: expected Ref records, got {sorted(tasks)}")
    if key.startswith("ref:") and splits != {key.removeprefix("ref:")}:
        errors.append(f"{label}/{key}: split metadata mismatch: {sorted(splits)}")
    if len(hashes) != 1:
        errors.append(f"{label}/{key}: multiple manifest hashes: {sorted(hashes)}")
    if len(sizes) != 1:
        errors.append(f"{label}/{key}: multiple manifest N values: {sorted(sizes)}")
    elif len(records) != next(iter(sizes)):
        errors.append(f"{label}/{key}: records N={len(records)} != manifest N={next(iter(sizes))}")
    elif next(iter(sizes)) <= 0:
        errors.append(f"{label}/{key}: manifest N must be positive")
    if duplicates:
        errors.append(f"{label}/{key}: duplicates={duplicates}")
    if invalid:
        errors.append(f"{label}/{key}: invalid={invalid}")
    if indices != list(range(len(records))):
        errors.append(f"{label}/{key}: manifest indices are not the exact 0..N-1 order")
    return errors


def _group_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids = [str(row["sample_id"]) for row in records]
    hashes = sorted({str(row["manifest_sha256"]) for row in records})
    declared_sizes = sorted({int(row["manifest_n"]) for row in records})
    return {
        "records_n": len(records),
        "manifest_n": declared_sizes[0] if len(declared_sizes) == 1 else declared_sizes,
        "manifest_sha256": hashes[0] if len(hashes) == 1 else hashes,
        "duplicates": len(ids) - len(set(ids)),
        "invalid": sum(not bool(row["valid"]) for row in records),
        "exact_manifest_index_order": [int(row["manifest_index"]) for row in records]
        == list(range(len(records))),
    }


def validate_paired_records(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_ref_splits: Sequence[str],
) -> Dict[str, Any]:
    errors: List[str] = []
    group_audits: Dict[str, Any] = {}
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    if baseline_keys != candidate_keys:
        errors.append(
            f"manifest groups differ: baseline_only={sorted(baseline_keys - candidate_keys)}, "
            f"candidate_only={sorted(candidate_keys - baseline_keys)}"
        )
    if "tn_global" not in baseline_keys or "tn_global" not in candidate_keys:
        errors.append("missing required tn_global record group")
    for split in required_ref_splits:
        key = f"ref:{split}"
        if key not in baseline_keys or key not in candidate_keys:
            errors.append(f"missing required Ref split: {split}")

    for key in sorted(baseline_keys | candidate_keys):
        group_audits[key] = {}
        if key in baseline:
            errors.extend(_validate_group("baseline", key, baseline[key]))
            group_audits[key]["baseline"] = _group_audit(baseline[key])
        if key in candidate:
            errors.extend(_validate_group("candidate", key, candidate[key]))
            group_audits[key]["candidate"] = _group_audit(candidate[key])
        if key not in baseline or key not in candidate:
            continue
        left = baseline[key]
        right = candidate[key]
        if len(left) != len(right):
            errors.append(f"{key}: baseline/candidate N mismatch: {len(left)} != {len(right)}")
            continue
        left_hashes = {str(row["manifest_sha256"]) for row in left}
        right_hashes = {str(row["manifest_sha256"]) for row in right}
        if left_hashes != right_hashes:
            errors.append(f"{key}: baseline/candidate manifest hash mismatch")
        left_ids = [str(row["sample_id"]) for row in left]
        right_ids = [str(row["sample_id"]) for row in right]
        if left_ids != right_ids:
            errors.append(f"{key}: baseline/candidate sample ID order mismatch")
        left_images = [int(row["image_id"]) for row in left]
        right_images = [int(row["image_id"]) for row in right]
        if left_images != right_images:
            errors.append(f"{key}: baseline/candidate image_id order mismatch")
        oracle_match = None
        if key.startswith("ref:"):
            left_oracle = [row.get("all_query_best_iou") for row in left]
            right_oracle = [row.get("all_query_best_iou") for row in right]
            missing_oracle = [
                index
                for index, (left_value, right_value) in enumerate(
                    zip(left_oracle, right_oracle)
                )
                if left_value is None or right_value is None
            ]
            if missing_oracle:
                errors.append(
                    f"{key}: all_query_best_iou missing at "
                    f"{len(missing_oracle)} paired rows"
                )
                oracle_match = False
            else:
                oracle_match = left_oracle == right_oracle
                if not oracle_match:
                    mismatches = sum(
                        left_value != right_value
                        for left_value, right_value in zip(left_oracle, right_oracle)
                    )
                    errors.append(
                        f"{key}: baseline/candidate all_query_best_iou oracle scalar "
                        f"drift at {mismatches} rows"
                    )
        group_audits[key]["paired"] = {
            "manifest_hash_match": left_hashes == right_hashes,
            "sample_id_order_match": left_ids == right_ids,
            "image_id_order_match": left_images == right_images,
            "n_match": len(left) == len(right),
            "all_query_best_iou_exact_match": oracle_match,
        }
    return {"pass": not errors, "errors": errors, "groups": group_audits}


def _cluster_index_lists(image_ids: Sequence[int]) -> List[np.ndarray]:
    grouped: MutableMapping[int, List[int]] = defaultdict(list)
    order: List[int] = []
    for index, image_id in enumerate(image_ids):
        image_id = int(image_id)
        if image_id not in grouped:
            order.append(image_id)
        grouped[image_id].append(index)
    return [np.asarray(grouped[image_id], dtype=np.int64) for image_id in order]


def _paired_cluster_bootstrap(
    image_ids: Sequence[int],
    metric,
    *,
    iterations: int,
    confidence: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    clusters = _cluster_index_lists(image_ids)
    if not clusters:
        raise GateValidationError("Cannot bootstrap an empty record set")
    deltas = np.empty((int(iterations),), dtype=np.float64)
    for iteration in range(int(iterations)):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[int(cluster)] for cluster in selected])
        deltas[iteration] = float(metric(indices))
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(deltas, [alpha, 1.0 - alpha])
    return {
        "iterations": int(iterations),
        "confidence": float(confidence),
        "num_image_clusters": len(clusters),
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
        "delta_bootstrap_median": float(np.median(deltas)),
    }


def evaluate_dual_gate(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_ref_splits: Sequence[str],
    target_tpr: float = 0.95,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260711,
) -> Dict[str, Any]:
    if bootstrap_iterations <= 0:
        raise GateValidationError("bootstrap_iterations must be positive")
    if not 0.0 < target_tpr <= 1.0:
        raise GateValidationError("target_tpr must be in (0, 1]")
    if not 0.0 < confidence < 1.0:
        raise GateValidationError("confidence must be in (0, 1)")
    validation = validate_paired_records(
        baseline,
        candidate,
        required_ref_splits=required_ref_splits,
    )
    report: Dict[str, Any] = {
        "schema": "stageb-dual-gate-v1",
        "target_tpr": float(target_tpr),
        "required_ref_splits": list(required_ref_splits),
        "validation": validation,
        "gate": {"pass": False},
    }
    if not validation["pass"]:
        return report

    rng = np.random.default_rng(int(seed))
    base_tn = baseline["tn_global"]
    cand_tn = candidate["tn_global"]
    base_pos = np.asarray([row["pos_score"] for row in base_tn], dtype=np.float64)
    base_neg = np.asarray([row["neg_score"] for row in base_tn], dtype=np.float64)
    cand_pos = np.asarray([row["pos_score"] for row in cand_tn], dtype=np.float64)
    cand_neg = np.asarray([row["neg_score"] for row in cand_tn], dtype=np.float64)
    baseline_fpr = exact_fpr_at_tpr(base_pos, base_neg, target_tpr)
    candidate_fpr = exact_fpr_at_tpr(cand_pos, cand_neg, target_tpr)

    def tn_delta(indices: np.ndarray) -> float:
        left = exact_fpr_at_tpr(base_pos[indices], base_neg[indices], target_tpr)["fpr"]
        right = exact_fpr_at_tpr(cand_pos[indices], cand_neg[indices], target_tpr)["fpr"]
        return right - left

    tn_bootstrap = _paired_cluster_bootstrap(
        [int(row["image_id"]) for row in base_tn],
        tn_delta,
        iterations=bootstrap_iterations,
        confidence=confidence,
        rng=rng,
    )
    tn_point_delta = float(candidate_fpr["fpr"] - baseline_fpr["fpr"])
    report["tn_global"] = {
        "n": len(base_tn),
        "baseline": baseline_fpr,
        "candidate": candidate_fpr,
        "candidate_minus_baseline_fpr": tn_point_delta,
        "improved": tn_point_delta < 0.0,
        "bootstrap": tn_bootstrap,
        "ci_supports_improvement": tn_bootstrap["delta_ci_high"] < 0.0,
    }

    evaluated_ref_splits = sorted(
        key.removeprefix("ref:")
        for key in baseline
        if key.startswith("ref:") and key in candidate
    )
    report["evaluated_ref_splits"] = evaluated_ref_splits
    ref_report: Dict[str, Any] = {}
    all_ref_improved = True
    for split in evaluated_ref_splits:
        key = f"ref:{split}"
        left = baseline[key]
        right = candidate[key]
        base_correct = np.asarray([bool(row["correct50"]) for row in left], dtype=np.float64)
        cand_correct = np.asarray([bool(row["correct50"]) for row in right], dtype=np.float64)
        baseline_acc = float(base_correct.mean())
        candidate_acc = float(cand_correct.mean())
        delta = candidate_acc - baseline_acc

        base_correct_mask = base_correct.astype(bool)
        cand_correct_mask = cand_correct.astype(bool)
        both_correct = int(np.sum(base_correct_mask & cand_correct_mask))
        regressions = int(np.sum(base_correct_mask & ~cand_correct_mask))
        fixes = int(np.sum(~base_correct_mask & cand_correct_mask))
        both_wrong = int(np.sum(~base_correct_mask & ~cand_correct_mask))
        net_fixes = fixes - regressions
        transition_total = both_correct + regressions + fixes + both_wrong
        if transition_total != len(left):
            raise GateValidationError(
                f"{split}: Ref transition counts cover {transition_total} rows, expected {len(left)}"
            )
        net_fixes_rate = float(net_fixes / len(left))
        if not math.isclose(delta, net_fixes_rate, rel_tol=0.0, abs_tol=1e-12):
            raise GateValidationError(
                f"{split}: Ref acc delta {delta} != net_fixes/N {net_fixes_rate}"
            )

        def ref_delta(indices: np.ndarray) -> float:
            return float(cand_correct[indices].mean() - base_correct[indices].mean())

        bootstrap = _paired_cluster_bootstrap(
            [int(row["image_id"]) for row in left],
            ref_delta,
            iterations=bootstrap_iterations,
            confidence=confidence,
            rng=rng,
        )
        improved = delta > 0.0
        all_ref_improved = all_ref_improved and improved
        split_report = {
            "n": len(left),
            "baseline_acc50": baseline_acc,
            "candidate_acc50": candidate_acc,
            "candidate_minus_baseline_acc50": delta,
            "transitions": {
                "both_correct": both_correct,
                "baseline_correct_candidate_wrong": regressions,
                "baseline_wrong_candidate_correct": fixes,
                "both_wrong": both_wrong,
                "regressions": regressions,
                "fixes": fixes,
                "net_fixes": net_fixes,
                "net_fixes_rate": net_fixes_rate,
            },
            "improved": improved,
            "bootstrap": bootstrap,
            "ci_supports_improvement": bootstrap["delta_ci_low"] > 0.0,
        }
        if all("all_query_best_iou" in row for row in left) and all(
            "all_query_best_iou" in row for row in right
        ):
            baseline_oracle_recall = float(
                np.mean(
                    [float(row["all_query_best_iou"]) >= 0.5 for row in left]
                )
            )
            candidate_oracle_recall = float(
                np.mean(
                    [float(row["all_query_best_iou"]) >= 0.5 for row in right]
                )
            )
            split_report.update(
                {
                    "baseline_oracle_recall50": baseline_oracle_recall,
                    "candidate_oracle_recall50": candidate_oracle_recall,
                    "baseline_oracle_headroom": (
                        baseline_oracle_recall - baseline_acc
                    ),
                    "candidate_oracle_headroom": (
                        candidate_oracle_recall - candidate_acc
                    ),
                }
            )
        ref_report[split] = split_report
    report["refcoco"] = ref_report
    fpr_improved = tn_point_delta < 0.0
    report["gate"] = {
        "pass": bool(fpr_improved and all_ref_improved),
        "global_fpr95_lower": bool(fpr_improved),
        "every_required_ref_split_acc50_higher": bool(all_ref_improved),
        "bootstrap_ci_is_informational_not_a_gate": True,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and candidate per-example Stage-B records. Exit 0 only when global "
            "FPR@95TPR is lower and acc50 is higher on every required RefCOCO split."
        )
    )
    parser.add_argument("--baseline_records", nargs="+", required=True)
    parser.add_argument("--candidate_records", nargs="+", required=True)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--required_ref_splits", nargs="+", default=DEFAULT_REF_SPLITS)
    parser.add_argument("--target_tpr", type=float, default=0.95)
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        baseline, baseline_inputs = load_record_set_with_inputs(args.baseline_records)
        candidate, candidate_inputs = load_record_set_with_inputs(args.candidate_records)
        report = evaluate_dual_gate(
            baseline,
            candidate,
            required_ref_splits=args.required_ref_splits,
            target_tpr=args.target_tpr,
            bootstrap_iterations=args.bootstrap_iterations,
            confidence=args.confidence,
            seed=args.seed,
        )
        report["input_files"] = {
            "baseline": baseline_inputs,
            "candidate": candidate_inputs,
            "identity_is_from_the_same_bytes_used_for_metrics": True,
        }
    except GateValidationError as error:
        report = {
            "schema": "stageb-dual-gate-v1",
            "validation": {"pass": False, "errors": [str(error)]},
            "gate": {"pass": False},
        }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report.get("validation", {}).get("pass", False):
        return 2
    return 0 if report.get("gate", {}).get("pass", False) else 1


if __name__ == "__main__":
    sys.exit(main())
