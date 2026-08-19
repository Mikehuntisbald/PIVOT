#!/usr/bin/env python3
"""Seal the input-matched original GroundingDINO-T FineCops replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import file_record, load_json, write_json_atomic
from tools.arrow_original_gdino_common import (
    CHECKPOINT_NUMEL,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_TENSORS,
    EXPECTED_FINECOPS_COUNTS,
    PRIMARY_SCORE,
    PREREG_SCHEMA,
    SENSITIVITY_SCORE,
)


DEFAULT_DATASET = Path(
    "/media/haoyi/T9/data/FineCops-Ref/v1/manifests/dataset_manifest.json"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "weights/groundingdino_swint_ogc.pth"
DEFAULT_CONFIG = (
    REPO_ROOT / "config/ablations/cfg_arrow_original_gdino_swint_ogc.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/arrow_original_gdino_ogc_finecops_20260819/preregistration.json"
)
DEFAULT_RESULTS = (
    REPO_ROOT / "outputs/arrow_original_gdino_ogc_finecops_20260819/evaluation"
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def build(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    results_root: Path,
) -> dict[str, object]:
    if results_root.exists() and any(results_root.iterdir()):
        raise ValueError(
            f"original OGC result directory is non-empty before lock: {results_root}"
        )
    dataset = load_json(dataset_path)
    if dataset.get("schema") != "arrow.finecops.dataset_manifest/v1":
        raise ValueError("FineCops dataset manifest schema drifted")
    if dataset.get("status") != "prepared_before_model_forward":
        raise ValueError("FineCops dataset is not sealed")
    counts = dataset.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("FineCops counts are missing")
    for key, expected in EXPECTED_FINECOPS_COUNTS.items():
        observed = counts.get(key)
        if observed != expected:
            raise ValueError(f"FineCops count {key}={observed!r} != {expected}")
    checkpoint = file_record(checkpoint_path)
    if (
        checkpoint["sha256"] != CHECKPOINT_SHA256
        or checkpoint["size_bytes"] != CHECKPOINT_SIZE
    ):
        raise ValueError("original GroundingDINO-T OGC checkpoint drifted")
    manifest = dataset.get("manifests", {}).get("bc_full")
    if not isinstance(manifest, dict) or manifest.get("rows") != 27926:
        raise ValueError("FineCops full expression manifest drifted")

    source_paths = [
        REPO_ROOT / "tools/arrow_original_gdino_common.py",
        REPO_ROOT / "tools/eval_arrow_original_gdino_finecops.py",
        REPO_ROOT / "tools/aggregate_arrow_original_gdino_finecops.py",
        REPO_ROOT / "tools/run_arrow_original_gdino_finecops_official.py",
        Path(__file__).resolve(),
    ]
    payload: dict[str, object] = {
        "schema": PREREG_SCHEMA,
        "status": "locked_before_original_ogc_forward",
        "evidence_status": "post_hoc_corrective_input_matched_baseline",
        "claim_boundary": (
            "FineCops-specific annotation/task zero-shot; original OGC weights; "
            "no FineCops training, model selection, score selection, or threshold fitting"
        ),
        "checkpoint": {
            **checkpoint,
            "tensor_count": CHECKPOINT_TENSORS,
            "parameter_numel": CHECKPOINT_NUMEL,
            "release": "IDEA-Research GroundingDINO Swin-T OGC",
        },
        "dataset": file_record(dataset_path),
        "manifest": dict(manifest),
        "config": file_record(config_path),
        "sources": [file_record(path) for path in source_paths],
        "primary_score": PRIMARY_SCORE,
        "sensitivity_score": SENSITIVITY_SCORE,
        "score_contract": {
            "expression_mean": (
                "mean sigmoid token probability over the model-generated "
                "full-expression phrase mask; input/scorer matched to ARROW Base"
            ),
            "expression_max": (
                "maximum sigmoid token probability over the same mask; "
                "upstream-native sensitivity only"
            ),
            "selected_after_results": False,
        },
        "prohibitions": {
            "training": True,
            "optimizer": True,
            "finecops_checkpoint_selection": True,
            "finecops_threshold_fitting": True,
            "support_or_canonical_input": True,
            "post_result_score_selection": True,
        },
        "execution": {
            "batch_size": 16,
            "num_workers": 4,
            "amp": True,
            "resize_short": 800,
            "resize_max": 1333,
            "flip": False,
            "loader_seed": 20260821,
            "results_root": str(results_root.resolve()),
        },
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "status_porcelain": _git(
                "status", "--porcelain", "--untracked-files=no"
            ),
            "diff_sha256": hashlib.sha256(
                subprocess.check_output(
                    ["git", "diff", "HEAD", "--binary"], cwd=REPO_ROOT
                )
            ).hexdigest(),
        },
    }
    write_json_atomic(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    payload = build(
        dataset_path=args.dataset.resolve(strict=True),
        checkpoint_path=args.checkpoint.resolve(strict=True),
        config_path=args.config.resolve(strict=True),
        output_path=args.output,
        results_root=args.results_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
