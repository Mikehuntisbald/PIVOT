#!/usr/bin/env python3
"""Aggregate ARROW Admission panel, val3, Test5, and confidence parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aggregate_stageb_u2v5_bootstrap import (  # noqa: E402
    _bootstrap_ref_comparison,
    _load_ref,
)


SCHEMA = "arrow.stageb.admission_input_results/v1"
SEEDS = (17, 42, 73)
EVAL = ROOT / "outputs/arrow_admission_input_20260818/evaluations"
ANCHOR_REF = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/ref8_u50/summary.json"
ANCHOR_CAL = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/formal/calibration_u25_u50_u100/summary.json"


class ArrowAggregateError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _panel(summary_path: Path) -> tuple[dict[int, dict[str, bool]], dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_seed, inputs = {}, {}
    for seed in SEEDS:
        path = Path(summary["inputs"][str(seed)]["records"]["path"])
        rows = _jsonl(path)
        grouped = {}
        for row in rows:
            grouped.setdefault(str(row["pair_id"]), []).append(row)
        if len(grouped) != 512 or any(len(pair) != 2 for pair in grouped.values()):
            raise ArrowAggregateError("fresh panel pair alignment drifted")
        by_seed[seed] = {
            pair_id: all(bool(row["active_score_wins"]) for row in pair)
            for pair_id, pair in grouped.items()
        }
        inputs[str(seed)] = _record(path)
    return by_seed, {"summary": _record(summary_path), "records": inputs}


def _panel_contrast(
    candidate: dict[int, dict[str, bool]], reference: dict[int, dict[str, bool]],
    *, rng: np.random.Generator, iterations: int,
) -> dict[str, Any]:
    pair_ids = sorted(candidate[17])
    for seed in SEEDS:
        if set(candidate[seed]) != set(pair_ids) or set(reference[seed]) != set(pair_ids):
            raise ArrowAggregateError("panel sample IDs differ across rows/seeds")
    candidate_values = {seed: np.asarray([candidate[seed][key] for key in pair_ids], dtype=np.float64) for seed in SEEDS}
    reference_values = {seed: np.asarray([reference[seed][key] for key in pair_ids], dtype=np.float64) for seed in SEEDS}
    observed_candidate = {seed: float(candidate_values[seed].mean()) for seed in SEEDS}
    observed_reference = {seed: float(reference_values[seed].mean()) for seed in SEEDS}
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.integers(0, len(pair_ids), size=len(pair_ids))
        draws[index] = np.mean([
            candidate_values[seed][sampled].mean() - reference_values[seed][sampled].mean()
            for seed in SEEDS
        ])
    gain = statistics.mean(observed_candidate.values()) - statistics.mean(observed_reference.values())
    return {
        "metric": "pair_level_bidirectional_category_switch_success_gain",
        "pairs": len(pair_ids),
        "candidate_by_seed": {str(k): v for k, v in observed_candidate.items()},
        "reference_by_seed": {str(k): v for k, v in observed_reference.items()},
        "gain": gain,
        "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        "one_sided_p": float((1 + np.count_nonzero(draws <= 0)) / (iterations + 1)),
    }


def _holm(entries: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(entries, key=lambda item: item[1])
    result, running, count = {}, 0.0, len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        result[name] = running
    return result


def _ref_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped = {seed: [] for seed in SEEDS}
    for row in payload["refcoco"]:
        match = re.search(r"seed(17|42|73)", f"{row.get('run_id','')} {row.get('checkpoint','')}")
        if not match:
            raise ArrowAggregateError("cannot recover Ref training seed")
        grouped[int(match.group(1))].append((float(row["acc50"]), int(row["num_expressions"])))
    values = {}
    for seed in SEEDS:
        total = sum(count for _, count in grouped[seed])
        values[seed] = sum(value * count for value, count in grouped[seed]) / total
    return {"by_seed": {str(k): v for k, v in values.items()}, "mean": statistics.mean(values.values()), "sample_sd": statistics.stdev(values.values()), "source": _record(path)}


def _tn_records(summary_path: Path, *, run_contains: str = "") -> dict[int, dict[str, tuple[float, float]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {}
    for row in summary["tn"]:
        run_id = str(row["run_id"])
        if run_contains and run_contains not in run_id:
            continue
        match = re.search(r"seed(17|42|73)", f"{run_id} {row.get('checkpoint','')}")
        if not match:
            continue
        seed = int(match.group(1))
        records = {}
        for record in _jsonl(Path(row["records_jsonl"])):
            records[str(record["sample_id"])] = (float(record["pos_score"]), float(record["neg_score"]))
        result[seed] = records
    if set(result) != set(SEEDS):
        raise ArrowAggregateError("TN parity source lacks three seeds")
    return result


def _confidence_parity(candidate_path: Path) -> dict[str, Any]:
    reference = _tn_records(ANCHOR_CAL, run_contains="_u50_")
    candidate = _tn_records(candidate_path)
    rows = {}
    for seed in SEEDS:
        if candidate[seed] != reference[seed]:
            differing = sum(candidate[seed].get(key) != reference[seed].get(key) for key in set(candidate[seed]) | set(reference[seed]))
            raise ArrowAggregateError(f"confidence parity failed seed{seed}: {differing}")
        rows[str(seed)] = {"records": len(candidate[seed]), "bitwise_equal_scores": True}
    return {"rows": rows, "candidate": _record(candidate_path), "reference": _record(ANCHOR_CAL)}


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise ArrowAggregateError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    panel_a, input_a = _panel(EVAL / "AR_A_PATCH/fresh_panel/summary.json")
    panel_b, input_b = _panel(EVAL / "AR_B_TEXT/fresh_panel/summary.json")
    panel_c, input_c = _panel(EVAL / "AR_C_NULL/fresh_panel/summary.json")
    seed_sequence = np.random.SeedSequence(20260818)
    rng_ab, rng_bc, rng_test_b, rng_test_c = [np.random.default_rng(child) for child in seed_sequence.spawn(4)]
    panel_contrasts = {
        "visual_over_text": _panel_contrast(panel_a, panel_b, rng=rng_ab, iterations=args.iterations),
        "category_over_null": _panel_contrast(panel_b, panel_c, rng=rng_bc, iterations=args.iterations),
    }
    adjusted = _holm([(name, value["one_sided_p"]) for name, value in panel_contrasts.items()])
    for name, value in adjusted.items():
        panel_contrasts[name]["holm_adjusted_p"] = value
    test_a = _load_ref(ANCHOR_REF)
    test_b = _load_ref(EVAL / "AR_B_TEXT/test5/summary.json")
    test_c = _load_ref(EVAL / "AR_C_NULL/test5/summary.json")
    test5 = {
        "B_minus_A": _bootstrap_ref_comparison(test_b, test_a, iterations=args.iterations, rng=rng_test_b, noninferiority_margin=0.005),
        "C_minus_A": _bootstrap_ref_comparison(test_c, test_a, iterations=args.iterations, rng=rng_test_c, noninferiority_margin=0.005),
    }
    payload = {
        "schema": SCHEMA, "status": "complete",
        "checkpoint_lock": _record(Path(args.checkpoint_lock)),
        "panel": {"inputs": {"A": input_a, "B": input_b, "C": input_c}, "contrasts": panel_contrasts, "bootstrap": {"iterations": args.iterations, "seed": 20260818, "cluster": "image_pair"}},
        "val3": {
            "A": _ref_metrics(ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/formal/ref_val3_admission_u100/summary.json"),
            "B": _ref_metrics(EVAL / "AR_B_TEXT/val3/summary.json"),
            "C": _ref_metrics(EVAL / "AR_C_NULL/val3/summary.json"),
        },
        "test5": test5,
        "confidence_parity": {
            "B": _confidence_parity(EVAL / "AR_B_TEXT/d3_calibration/summary.json"),
            "C": _confidence_parity(EVAL / "AR_C_NULL/d3_calibration/summary.json"),
        },
        "strict_forwarded": False,
    }
    _write(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ArrowAggregateError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
