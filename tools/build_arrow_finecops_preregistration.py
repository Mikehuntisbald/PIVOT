#!/usr/bin/env python3
"""Seal the FineCops-Ref external-evaluation contract before model forward."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    OFFICIAL_REPO_COMMIT,
    PREREG_SCHEMA,
    file_record,
    load_json,
    write_json_atomic,
)


DEFAULT_DATASET = Path(
    "/media/haoyi/T9/data/FineCops-Ref/v1/manifests/dataset_manifest.json"
)
DEFAULT_RELEASE = REPO_ROOT / "outputs/arrow_release_20260818/release_manifest.json"
DEFAULT_LOCK = REPO_ROOT / "outputs/arrow_admission_input_20260818/checkpoint_lock_v2.json"
DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "outputs/u2v5_leakage_clean_anchor_20260817/formal/"
    "calibration_u25_u50_u100/summary.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/arrow_finecops_20260819/preregistration.json"
DEFAULT_RESULTS = REPO_ROOT / "outputs/arrow_finecops_20260819/evaluations"
DEFAULT_OFFICIAL_REPO = Path("/media/haoyi/T9/data/FineCops-Ref/v1/official_repo")
DEFAULT_CORRECTION_PARENT = REPO_ROOT / (
    "outputs/arrow_finecops_hf_reencoded_diagnostic_20260819/"
    "diagnostic_relocation.json"
)


def _git(command: list[str]) -> str:
    return subprocess.check_output(
        ["git", *command], cwd=REPO_ROOT, text=True
    ).strip()


def _checkpoint_rows(release: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"A": {}, "B": {}, "C": {}}
    for seed in (17, 42, 73):
        result["A"][str(seed)] = dict(
            release["legacy_evidence"]["main_checkpoints"][str(seed)]
        )
        for row_id, short in (("AR_B_TEXT", "B"), ("AR_C_NULL", "C")):
            receipt_path = (
                REPO_ROOT
                / "outputs/arrow_admission_input_20260818/eval_checkpoints"
                / row_id
                / f"seed{seed}.receipt.json"
            )
            receipt = load_json(receipt_path)
            result[short][str(seed)] = {
                **dict(receipt["checkpoint"]),
                "overlay_receipt": file_record(receipt_path),
            }
    return result


def _validate_checkpoint_files(rows: dict[str, Any]) -> None:
    for route, seeds in rows.items():
        for seed, expected in seeds.items():
            observed = file_record(Path(expected["path"]))
            for key in ("sha256", "size_bytes"):
                if observed[key] != expected[key]:
                    raise ValueError(
                        f"{route}/seed{seed} checkpoint {key} drifted"
                    )


def _thresholds(
    calibration_path: Path, checkpoints: dict[str, Any]
) -> dict[str, Any]:
    summary = load_json(calibration_path)
    rows = summary.get("tn")
    if not isinstance(rows, list):
        raise ValueError("D3 calibration summary has no TN list")
    result: dict[str, Any] = {}
    for seed in (17, 42, 73):
        checkpoint_sha = checkpoints["A"][str(seed)]["sha256"]
        matches = [
            row
            for row in rows
            if str(row.get("checkpoint_sha256")) == checkpoint_sha
            and "_u50/" in str(row.get("checkpoint", ""))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one sealed D3-U50 threshold for seed {seed}, got {len(matches)}"
            )
        result[str(seed)] = {
            "raw_threshold_at_95tpr": float(matches[0]["threshold_at_95tpr"]),
            "calibration_run_id": str(matches[0]["run_id"]),
            "checkpoint_sha256": checkpoint_sha,
        }
    return result


def build(
    *,
    dataset_manifest_path: Path,
    release_manifest_path: Path,
    checkpoint_lock_path: Path,
    calibration_path: Path,
    output_path: Path,
    results_root: Path,
    official_repo: Path,
    correction_parent: Path | None,
) -> dict[str, Any]:
    if results_root.exists() and any(results_root.iterdir()):
        raise ValueError(
            f"FineCops result directory is non-empty before preregistration: {results_root}"
        )
    dataset = load_json(dataset_manifest_path)
    if dataset.get("schema") != "arrow.finecops.dataset_manifest/v1":
        raise ValueError("dataset manifest schema drifted")
    if dataset.get("status") != "prepared_before_model_forward":
        raise ValueError("dataset was not sealed before model forward")
    release = load_json(release_manifest_path)
    if release.get("schema") != "arrow.release_manifest/v1":
        raise ValueError("ARROW release manifest schema drifted")
    lock = load_json(checkpoint_lock_path)
    if lock.get("schema") != "arrow.stageb.admission_input_checkpoint_lock/v1":
        raise ValueError("ARROW admission checkpoint lock schema drifted")
    checkpoints = _checkpoint_rows(release, lock)
    _validate_checkpoint_files(checkpoints)
    thresholds = _thresholds(calibration_path, checkpoints)
    official_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=official_repo, text=True
    ).strip()
    official_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=official_repo,
        text=True,
    ).strip()
    if official_commit != OFFICIAL_REPO_COMMIT or official_status:
        raise ValueError("external FineCops evaluator checkout is not pinned and clean")
    official_source = official_repo / "evaluation" / "eval_metric_mmdet.py"

    config_paths = {
        "A": REPO_ROOT / "config/ablations/cfg_arrow_finecops_a.py",
        "B": REPO_ROOT / "config/ablations/cfg_arrow_finecops_b.py",
        "C": REPO_ROOT / "config/ablations/cfg_arrow_finecops_c.py",
    }
    source_paths = [
        REPO_ROOT / "tools/arrow_finecops_common.py",
        REPO_ROOT / "tools/download_arrow_finecops_gqa.py",
        REPO_ROOT / "tools/download_arrow_finecops_hf_images.py",
        REPO_ROOT / "tools/prepare_arrow_finecops.py",
        REPO_ROOT / "tools/eval_arrow_finecops.py",
        REPO_ROOT / "tools/run_arrow_finecops_evaluations.py",
        REPO_ROOT / "tools/aggregate_arrow_finecops.py",
        REPO_ROOT / "tools/run_arrow_finecops_official.py",
        REPO_ROOT / "tools/render_arrow_finecops_table.py",
        REPO_ROOT / "tools/build_arrow_finecops_final_receipt.py",
        Path(__file__).resolve(),
    ]
    correction = None
    status = "locked_before_any_finecops_model_forward"
    if correction_parent is not None:
        correction_payload = load_json(correction_parent)
        if (
            correction_payload.get("schema")
            != "arrow.finecops.hf_reencoded_diagnostic_relocation/v1"
            or correction_payload.get("model_results_viewed") is not True
        ):
            raise ValueError("FineCops correction parent contract drifted")
        correction = {
            "parent": file_record(correction_parent),
            "reason": "HF GQA mirror reencoded 4313/4313 JPEGs relative to official GQA zip",
            "scope": "official_gqa_image_bytes_only",
            "prior_results_viewed": True,
            "planned_models_contrasts_statistics_unchanged": True,
        }
        status = "locked_before_official_gqa_byte_correction_replay"
    payload = {
        "schema": PREREG_SCHEMA,
        "status": status,
        "claim": "finecops_specific_external_zero_shot_not_image_disjoint",
        "prohibitions": {
            "finecops_train_or_val": True,
            "finecops_checkpoint_selection": True,
            "finecops_threshold_fitting": True,
            "post_result_alias_addition": True,
            "gap_change": True,
        },
        "correction_replay": correction,
        "dataset": file_record(dataset_manifest_path),
        "release_manifest": file_record(release_manifest_path),
        "checkpoint_lock": file_record(checkpoint_lock_path),
        "checkpoints": checkpoints,
        "configs": {key: file_record(path) for key, path in config_paths.items()},
        "sources": [file_record(path) for path in source_paths],
        "calibration": {
            "summary": file_record(calibration_path),
            "thresholds": thresholds,
        },
        "official_exact": {
            "repository": "https://github.com/liujunzhuo/FineCops-Ref",
            "commit": OFFICIAL_REPO_COMMIT,
            "source": file_record(official_source),
            "vendor_code": False,
            "overall_positive_scope": "level_1_historical_behavior",
        },
        "execution": {
            "routes": ["A", "B", "C"],
            "seeds": [17, 42, 73],
            "batch_size": 16,
            "num_workers": 4,
            "amp": True,
            "resize_short": 800,
            "resize_max": 1333,
            "flip": False,
            "gap": 3.0,
            "results_root": str(results_root.resolve()),
        },
        "statistics": {
            "bootstrap": {
                "iterations": BOOTSTRAP_ITERATIONS,
                "seed": BOOTSTRAP_SEED,
                "cluster": "parent_positive_gqa_image",
                "same_draw_across_routes_and_seeds": True,
            },
            "planned_contrasts": ["A_minus_B", "B_minus_C"],
            "contrast_surface": "A_support_covered_matched",
            "holm_family": True,
            "strict_tie_policy": "tie_is_failure",
        },
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "status_porcelain": _git(["status", "--porcelain", "--untracked-files=no"]),
            "diff_sha256": __import__("hashlib").sha256(
                subprocess.check_output(["git", "diff", "--binary"], cwd=REPO_ROOT)
            ).hexdigest(),
        },
        "environment": {
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "pyarrow",
                )
            },
        },
    }
    write_json_atomic(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--checkpoint-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--calibration-summary", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument(
        "--correction-parent", type=Path, default=DEFAULT_CORRECTION_PARENT
    )
    args = parser.parse_args()
    payload = build(
        dataset_manifest_path=args.dataset_manifest.resolve(strict=True),
        release_manifest_path=args.release_manifest.resolve(strict=True),
        checkpoint_lock_path=args.checkpoint_lock.resolve(strict=True),
        calibration_path=args.calibration_summary.resolve(strict=True),
        output_path=args.output,
        results_root=args.results_root,
        official_repo=args.official_repo.resolve(strict=True),
        correction_parent=(
            args.correction_parent.resolve(strict=True)
            if args.correction_parent is not None
            else None
        ),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
