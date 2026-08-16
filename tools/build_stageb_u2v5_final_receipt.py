#!/usr/bin/env python3
"""Bind the one-time U2-v5 Ref8/strict read to its preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817"
SEEDS = (17, 42, 73)
SCHEMA = "pivot.stageb.u2v5_leakage_clean_final_receipt/v1"


class FinalReceiptError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FinalReceiptError(f"expected JSON object: {path}")
    return value


def ref_micro(summary: dict[str, Any]) -> dict[int, float]:
    grouped: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
    for row in summary.get("refcoco", []):
        run_id = str(row.get("run_id", ""))
        matches = [seed for seed in SEEDS if f"seed{seed}_u50" in run_id]
        if len(matches) != 1:
            raise FinalReceiptError(f"unexpected Ref8 run id: {run_id}")
        grouped[matches[0]].append(row)
    result = {}
    for seed, rows in grouped.items():
        if len(rows) != 8:
            raise FinalReceiptError(f"seed {seed} does not have Ref8")
        count = sum(int(row["num_expressions"]) for row in rows)
        if count != 57457:
            raise FinalReceiptError(f"seed {seed} Ref8 row count drifted")
        result[seed] = sum(
            float(row["acc50"]) * int(row["num_expressions"]) for row in rows
        ) / count
    return result


def tn_scores(summary: dict[str, Any], expected_n: int) -> dict[int, float]:
    result = {}
    for row in summary.get("tn", []):
        run_id = str(row.get("run_id", ""))
        matches = [seed for seed in SEEDS if f"seed{seed}_u50" in run_id]
        if len(matches) != 1 or int(row.get("num_pairs", 0)) != expected_n:
            raise FinalReceiptError("strict summary seed/row contract drifted")
        result[matches[0]] = float(row["fpr95tpr"])
    if set(result) != set(SEEDS):
        raise FinalReceiptError("strict summary does not contain all seeds")
    return result


def mean(values: dict[int, float]) -> float:
    return sum(values.values()) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FinalReceiptError(f"refusing to overwrite {destination}")

    prereg_path = OUT / "preregistration/locked_before_ref8_strict.json"
    prereg = load(prereg_path)
    if prereg.get("schema") != "pivot.stageb.u2v5_leakage_clean_preregistration/v1":
        raise FinalReceiptError("preregistration schema drifted")
    if prereg.get("confidence_selection", {}).get("selected_update") != 50:
        raise FinalReceiptError("preregistered confidence milestone is not U50")

    ref_path = OUT / "final_once/ref8_u50/summary.json"
    s1607_path = OUT / "final_once/strict1607_u50/summary.json"
    s2031_path = OUT / "final_once/strict2031_u50/summary.json"
    ref = load(ref_path)
    s1607 = load(s1607_path)
    s2031 = load(s2031_path)
    micros = ref_micro(ref)
    strict1607 = tn_scores(s1607, 1607)
    strict2031 = tn_scores(s2031, 2031)

    b58_ref = load(
        ROOT / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/summary.json"
    )
    b58_rows = {row["dataset"]: row for row in b58_ref["refcoco"]}
    per_split_nonregression = {}
    strict_improvements = {}
    for seed in SEEDS:
        seed_rows = [
            row for row in ref["refcoco"] if f"seed{seed}_u50" in row["run_id"]
        ]
        per_split_nonregression[str(seed)] = all(
            float(row["acc50"]) >= float(b58_rows[row["dataset"]]["acc50"])
            for row in seed_rows
        )
        strict_improvements[str(seed)] = {
            "strict1607": strict1607[seed] < 0.4984443061605476,
            "strict2031": strict2031[seed] < 0.5120630231413097,
        }

    payload = {
        "schema": SCHEMA,
        "status": "one_time_final_read_complete",
        "preregistration": record(prereg_path),
        "final_artifacts": {
            "ref8": record(ref_path),
            "strict1607": record(s1607_path),
            "strict2031": record(s2031_path),
        },
        "point_estimates": {
            "ref8_micro_by_seed": {str(k): v for k, v in micros.items()},
            "ref8_micro_seed_mean": mean(micros),
            "b58_ref8_micro": 0.7093652644586387,
            "legacy_u2_ref8_micro": 0.7337139077919139,
            "strict1607_fpr95_by_seed": {
                str(k): v for k, v in strict1607.items()
            },
            "strict1607_seed_mean": mean(strict1607),
            "b58_strict1607": 0.4984443061605476,
            "diagnostic_c100_strict1607": 0.454263,
            "strict2031_fpr95_by_seed": {
                str(k): v for k, v in strict2031.items()
            },
            "strict2031_seed_mean": mean(strict2031),
            "b58_strict2031": 0.5120630231413097,
            "diagnostic_c100_strict2031": 0.455933,
        },
        "gates": {
            "every_seed_every_ref_split_nonregresses_vs_b58": all(
                per_split_nonregression.values()
            ),
            "per_seed_ref_nonregression": per_split_nonregression,
            "every_seed_strict_fpr95_improves_vs_b58": all(
                all(value.values()) for value in strict_improvements.values()
            ),
            "per_seed_strict_improvement": strict_improvements,
            "c100_plus_0p01_noninferiority_strict1607": max(
                strict1607.values()
            ) <= 0.454263 + 0.01,
            "c100_plus_0p01_noninferiority_strict2031": max(
                strict2031.values()
            ) <= 0.455933 + 0.01,
            "bootstrap_ci_status": "pending_seed_first_image_cluster_bootstrap",
        },
        "interpretation": (
            "Leakage-clean anchor clears all B58 point-estimate gates and matches "
            "legacy U2 Ref8 on average; it does not clear the diagnostic-C100+0.01 "
            "strict2031 noninferiority margin. CI claims remain pending."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "sealed", "receipt": record(destination)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, FinalReceiptError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
