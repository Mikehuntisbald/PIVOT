#!/usr/bin/env python3
"""Lock the leakage-clean U2-v5 anchor before any test/strict evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817"
SEEDS = (17, 42, 73)
UPDATES = (25, 50, 100)
VAL_DATASETS = ("refcoco_val", "refcocop_val", "refcocog_val")
SCHEMA = "pivot.stageb.u2v5_leakage_clean_preregistration/v1"


class PreregistrationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if rows is not None:
        with path.open("r", encoding="utf-8") as handle:
            observed = sum(1 for line in handle if line.strip())
        if observed != rows:
            raise PreregistrationError(
                f"manifest row count drifted for {path}: {observed} != {rows}"
            )
        result["rows"] = rows
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_contract() -> dict[str, str]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if status.strip():
        raise PreregistrationError("preregistration requires a clean worktree")
    return {"commit": head, "branch": "main", "worktree": "clean"}


def select_confidence(calibration: dict[str, Any]) -> dict[str, Any]:
    rows = calibration.get("tn")
    if not isinstance(rows, list) or len(rows) != len(SEEDS) * len(UPDATES):
        raise PreregistrationError("calibration summary is not the 3x3 panel")
    panel: dict[int, dict[int, float]] = {update: {} for update in UPDATES}
    pattern = re.compile(r"confidence_seed(17|42|73)_u(25|50|100)_checkpoint_iter")
    for row in rows:
        match = pattern.fullmatch(str(row.get("run_id", "")))
        if match is None:
            raise PreregistrationError("unexpected calibration run id")
        seed, update = map(int, match.groups())
        panel[update][seed] = float(row["fpr95tpr"])
    if any(set(values) != set(SEEDS) for values in panel.values()):
        raise PreregistrationError("calibration panel is incomplete")
    candidates = []
    for update, values in panel.items():
        scores = [values[seed] for seed in SEEDS]
        candidates.append(
            {
                "update": update,
                "seed_fpr95": {str(seed): values[seed] for seed in SEEDS},
                "worst_seed_fpr95": max(scores),
                "mean_fpr95": sum(scores) / len(scores),
            }
        )
    candidates.sort(
        key=lambda row: (
            row["worst_seed_fpr95"], row["mean_fpr95"], row["update"]
        )
    )
    if candidates[0]["update"] != 50:
        raise PreregistrationError("locked robust selection no longer chooses U50")
    return {
        "policy": (
            "minimize worst-seed D3 calibration FPR95; then mean FPR95; "
            "then earlier update"
        ),
        "candidates": candidates,
        "selected_update": 50,
    }


def validate_val3(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("refcoco")
    if not isinstance(rows, list) or len(rows) != 9:
        raise PreregistrationError("val3 summary is not the 3-seed panel")
    baseline = load_json(
        ROOT / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/summary.json"
    )
    baseline_val = {
        row["dataset"]: float(row["acc50"])
        for row in baseline["refcoco"]
        if row["dataset"] in VAL_DATASETS
    }
    observed: dict[str, dict[str, float]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        match = re.fullmatch(r"admission_seed(17|42|73)_u100_checkpoint_iter", run_id)
        dataset = str(row.get("dataset", ""))
        if match is None or dataset not in VAL_DATASETS:
            raise PreregistrationError("unexpected val3 row")
        seed = match.group(1)
        score = float(row["acc50"])
        if score <= baseline_val[dataset]:
            raise PreregistrationError(
                f"seed {seed} does not strictly beat B58 on {dataset}"
            )
        observed.setdefault(seed, {})[dataset] = score
    if any(set(values) != set(VAL_DATASETS) for values in observed.values()):
        raise PreregistrationError("val3 seed/split coverage is incomplete")
    return {
        "policy": "fixed admission U100; every seed and val split strictly beats B58",
        "scores": observed,
        "baseline_val_scores": baseline_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if destination.exists():
        raise PreregistrationError(f"refusing to overwrite {destination}")

    calibration_path = OUT / "formal/calibration_u25_u50_u100/summary.json"
    val3_path = OUT / "formal/ref_val3_admission_u100/summary.json"
    confidence_selection = select_confidence(load_json(calibration_path))
    admission_selection = validate_val3(load_json(val3_path))
    admission = {
        str(seed): record(
            OUT / f"formal/admission_seed{seed}_u100/checkpoint_iter.pth"
        )
        for seed in SEEDS
    }
    confidence = {
        str(seed): record(
            OUT / f"formal/confidence_seed{seed}_u50/checkpoint_iter.pth"
        )
        for seed in SEEDS
    }
    strict_root = ROOT / (
        "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
    )
    payload = {
        "schema": SCHEMA,
        "status": "locked_before_ref8_or_strict",
        "git": git_contract(),
        "selection_surfaces": {
            "ref_val3": record(val3_path),
            "d3_calibration": record(calibration_path),
        },
        "admission_selection": admission_selection,
        "confidence_selection": confidence_selection,
        "selected_checkpoints": {
            "admission_u100": admission,
            "confidence_u50": confidence,
        },
        "sealed_future_surfaces": {
            "ref8_baseline": record(
                ROOT / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/summary.json"
            ),
            "strict1607": record(strict_root / "eval_manifest.jsonl", rows=1607),
            "strict2031": record(
                strict_root / "semantic_stageb_union_image_disjoint_manifest.jsonl",
                rows=2031,
            ),
        },
        "execution_contract": {
            "run_once_only": True,
            "runtime": "cuda:0/B16/W4/AMP/seed42/full/per-example-records",
            "ref8_splits": [
                "refcoco_val", "refcoco_testA", "refcoco_testB",
                "refcocop_val", "refcocop_testA", "refcocop_testB",
                "refcocog_val", "refcocog_test",
            ],
            "strict_splits": ["refcocop_val", "refcocog_umd_val"],
            "forbid_additional_checkpoint_selection": True,
        },
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
    print(json.dumps({"status": "locked", "receipt": record(destination)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, PreregistrationError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
