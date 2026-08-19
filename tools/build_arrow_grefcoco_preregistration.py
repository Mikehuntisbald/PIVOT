#!/usr/bin/env python3
"""Seal the ARROW gRefCOCO rejection-transfer contract before GPU forward."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_grefcoco_common import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    PREREG_SCHEMA,
    SEALED_THRESHOLDS,
    SEEDS,
    file_record,
    load_json,
    write_json_atomic,
)


DEFAULT_DATASET = Path("/media/haoyi/T9/data/gRefCOCO/v1/manifests/dataset_manifest.json")
DEFAULT_RELEASE = REPO_ROOT / "outputs/arrow_release_20260818/release_manifest.json"
DEFAULT_FINECOPS_PREREG = REPO_ROOT / "outputs/arrow_finecops_20260819/preregistration.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/arrow_grefcoco_20260820/preregistration.json"
DEFAULT_RESULTS = REPO_ROOT / "outputs/arrow_grefcoco_20260820/evaluations"
DEFAULT_CONFIG = REPO_ROOT / "config/ablations/cfg_arrow_grefcoco_confidence_only.py"
DEFAULT_PREFLIGHT = REPO_ROOT / "outputs/arrow_grefcoco_20260820/preflight.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _checkpoints() -> dict[str, Any]:
    prior = load_json(DEFAULT_FINECOPS_PREREG)
    rows = prior["checkpoints"]
    for route in ("A", "B", "C"):
        for seed in SEEDS:
            expected = rows[route][str(seed)]
            observed = file_record(Path(expected["path"]))
            if any(observed[key] != expected[key] for key in ("sha256", "size_bytes")):
                raise ValueError(f"{route}/seed{seed} checkpoint drifted")
    for seed in SEEDS:
        a_sha = rows["A"][str(seed)]["sha256"]
        for route in ("B", "C"):
            receipt = load_json(Path(rows[route][str(seed)]["overlay_receipt"]["path"]))
            contract = receipt["contract"]
            if (
                contract.get("confidence_source", {}).get("sha256") != a_sha
                or contract.get("confidence_tensor_sha256")
                != load_json(Path(rows["B"][str(seed)]["overlay_receipt"]["path"]))["contract"]["confidence_tensor_sha256"]
                or len(contract.get("confidence_keys", [])) != 12
            ):
                raise ValueError(f"{route}/seed{seed} confidence ownership drifted")
    return rows


def build(dataset_path: Path, output_path: Path, results_root: Path) -> dict[str, Any]:
    if results_root.exists() and any(results_root.iterdir()):
        raise ValueError("gRefCOCO result directory must be absent or empty before preregistration")
    dataset = load_json(dataset_path)
    if dataset.get("schema") != "arrow.grefcoco.dataset_manifest/v1":
        raise ValueError("gRefCOCO dataset contract drifted")
    overlap_path = Path(dataset["manifests"]["overlap_audit"]["path"])
    overlap = load_json(overlap_path)
    required = {
        "full": (1500, 11563, 9121),
        "d3_disjoint": (1288, 9924, 7796),
        "d3_finecops_disjoint": (1274, 9821, 7714),
    }
    for name, expected in required.items():
        row = overlap["surfaces"][name]
        if (row["images"], row["positive"], row["negative"]) != expected:
            raise ValueError(f"overlap surface {name} drifted")
    sources = [
        REPO_ROOT / "tools/arrow_grefcoco_common.py",
        REPO_ROOT / "tools/prepare_arrow_grefcoco.py",
        REPO_ROOT / "tools/build_arrow_grefcoco_preregistration.py",
        REPO_ROOT / "tools/preflight_arrow_grefcoco.py",
        REPO_ROOT / "tools/eval_arrow_grefcoco.py",
        REPO_ROOT / "tools/run_arrow_grefcoco_evaluations.py",
        REPO_ROOT / "tools/aggregate_arrow_grefcoco.py",
        REPO_ROOT / "tools/build_arrow_grefcoco_final_receipt.py",
    ]
    payload = {
        "schema": PREREG_SCHEMA,
        "status": "locked_before_any_grefcoco_model_forward",
        "dataset": file_record(dataset_path),
        "overlap_audit": file_record(overlap_path),
        "release_manifest": file_record(DEFAULT_RELEASE),
        "finecops_preregistration": file_record(DEFAULT_FINECOPS_PREREG),
        "preflight": file_record(DEFAULT_PREFLIGHT),
        "checkpoints": _checkpoints(),
        "formal_checkpoint_route": "A_confidence_only_because_ABC_confidence12_is_seed_matched",
        "thresholds": {
            str(seed): {"raw_threshold": value, "comparison": ">=", "source": "sealed_D3_U50_calibration"}
            for seed, value in SEALED_THRESHOLDS.items()
        },
        "config": file_record(DEFAULT_CONFIG),
        "sources": [file_record(path) for path in sources],
        "execution": {"results_root": str(results_root.resolve()), "seeds": list(SEEDS), "batch_size": 16, "num_workers": 4, "amp": True, "device": "cuda:0", "loader_seed": 20260820},
        "statistics": {"bootstrap": "paired_stratified_image_cluster", "iterations": BOOTSTRAP_ITERATIONS, "pcg64_seed": BOOTSTRAP_SEED, "strata": ["testA", "testB"], "recompute_domain_q05_each_replicate": True, "fixed_threshold_never_reestimated": True},
        "surfaces": {"primary": "d3_disjoint", "co_required_robustness": "full", "sensitivity": "d3_finecops_disjoint", "val": "no_target_fixed_threshold_only"},
        "planned_contrast": {"candidate": "D3", "reference": "B58", "gates": ["auroc_gain_ci_low_gt_0", "fpr95_gain_ci_low_gt_0"], "required_surfaces": ["d3_disjoint", "full"]},
        "prohibitions": ["model_training", "optimizer_creation", "grefcoco_train_forward", "checkpoint_selection", "gap_tuning", "threshold_tuning", "admission_on_no_target", "b58_fixed_threshold_invention", "result_dependent_protocol_change"],
        "claim_boundary": "gRefCOCO annotation/task-zero-shot cross-benchmark transfer on previously exposed COCO imagery",
        "git": {"head": _git("rev-parse", "HEAD"), "status_at_lock": _git("status", "--short")},
    }
    write_json_atomic(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset.resolve(strict=True), args.output.resolve(), args.results_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
