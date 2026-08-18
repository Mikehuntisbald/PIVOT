#!/usr/bin/env python3
"""Run the pinned, external FineCops evaluator without vendoring its code."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import (
    OFFICIAL_REPO_COMMIT,
    file_record,
    load_json,
    load_jsonl,
    write_json_atomic,
)


DEFAULT_RESULTS = REPO_ROOT / "outputs/arrow_finecops_20260819/results.json"
DEFAULT_OFFICIAL_REPO = Path("/media/haoyi/T9/data/FineCops-Ref/v1/official_repo")
DEFAULT_ANNOTATION = Path(
    "/media/haoyi/T9/data/FineCops-Ref/v1/raw/benchmark/"
    "test_expression_all_coco_format.json"
)


def _install_import_shims() -> None:
    from pycocotools.coco import COCO

    class BaseMetric:
        def __init__(self, **_kwargs):
            self.results = []

    @contextlib.contextmanager
    def get_local_path(path):
        yield path

    class Logger:
        @classmethod
        def get_current_instance(cls):
            return cls()

        def info(self, value):
            print(value)

    class Registry:
        def register_module(self):
            return lambda value: value

    def bbox_overlaps(first, second):
        a = np.asarray(first, dtype=np.float64)
        b = np.asarray(second, dtype=np.float64)
        left_top = np.maximum(a[:, None, :2], b[None, :, :2])
        right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
        extent = np.clip(right_bottom - left_top, 0.0, None)
        intersection = extent[..., 0] * extent[..., 1]
        area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(
            a[:, 3] - a[:, 1], 0.0, None
        )
        area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(
            b[:, 3] - b[:, 1], 0.0, None
        )
        return intersection / np.clip(
            area_a[:, None] + area_b[None, :] - intersection, 1e-12, None
        )

    modules = {
        "mmengine": types.ModuleType("mmengine"),
        "mmengine.evaluator": types.ModuleType("mmengine.evaluator"),
        "mmengine.fileio": types.ModuleType("mmengine.fileio"),
        "mmengine.logging": types.ModuleType("mmengine.logging"),
        "mmdet": types.ModuleType("mmdet"),
        "mmdet.datasets": types.ModuleType("mmdet.datasets"),
        "mmdet.datasets.api_wrappers": types.ModuleType("mmdet.datasets.api_wrappers"),
        "mmdet.registry": types.ModuleType("mmdet.registry"),
        "mmdet.evaluation": types.ModuleType("mmdet.evaluation"),
        "seaborn": types.ModuleType("seaborn"),
    }
    modules["mmengine.evaluator"].BaseMetric = BaseMetric
    modules["mmengine.fileio"].get_local_path = get_local_path
    modules["mmengine.logging"].MMLogger = Logger
    modules["mmdet.datasets.api_wrappers"].COCO = COCO
    modules["mmdet.registry"].METRICS = Registry()
    modules["mmdet.evaluation"].bbox_overlaps = bbox_overlaps
    sys.modules.update(modules)


def _load_official(repo: Path):
    import subprocess

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).strip()
    if commit != OFFICIAL_REPO_COMMIT or status:
        raise ValueError("external FineCops repository is not the pinned clean commit")
    path = repo / "evaluation" / "eval_metric_mmdet.py"
    _install_import_shims()
    specification = importlib.util.spec_from_file_location(
        "arrow_external_finecops_metric", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load pinned FineCops evaluator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, path


def _read_table(path: Path) -> list[dict]:
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def run(results_path: Path, repo: Path, annotation: Path) -> dict:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "scikit-learn is required only for the external pinned evaluator"
        ) from error
    results = load_json(results_path)
    if results.get("schema") != "arrow.finecops.results/v1":
        raise ValueError("FineCops results schema drifted")
    prereg_path = Path(results["preregistration"]["path"]).resolve(strict=True)
    prereg = load_json(prereg_path)
    module, official_source = _load_official(repo.resolve(strict=True))
    expected_source = prereg["official_exact"]["source"]
    observed_source = file_record(official_source)
    if any(observed_source[key] != expected_source[key] for key in ("sha256", "size_bytes")):
        raise ValueError("pinned official evaluator source drifted after preregistration")
    official_root = results_path.parent / "official_exact"
    official: dict[str, dict[str, dict]] = {}
    for surface, seeds in results["official_prediction_artifacts"].items():
        official[surface] = {}
        for seed, artifact in seeds.items():
            prediction_path = Path(artifact["path"]).resolve(strict=True)
            if file_record(prediction_path)["sha256"] != artifact["sha256"]:
                raise ValueError(f"official prediction drifted: {surface}/seed{seed}")
            rows = load_jsonl(prediction_path)
            predictions = [
                {
                    "img_id": int(row["image_id"]),
                    "bboxes": np.asarray(row["bboxes"], dtype=np.float32),
                    "scores": np.asarray(row["scores"], dtype=np.float32),
                }
                for row in rows
            ]
            output_dir = official_root / surface / f"seed{seed}"
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(f"official output already exists: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            evaluator = module.RefRecallAUROC(
                ann_file=str(annotation.resolve(strict=True)),
                # The official implementation hard-codes three CSV columns.
                # Each ARROW sample still contains exactly one prediction, so
                # P@5/P@10 duplicate P@1 and are never used in our paper table.
                topk=(1, 5, 10),
                iou_thrs=0.5,
                score_thrs=0.0,
                save_dir=str(output_dir),
            )
            returned = evaluator.compute_metrics(predictions)
            precision = output_dir / "precision.csv"
            recall = output_dir / "recall_['negative_type', 'negative_level', 'negative_cate'].csv"
            auroc = output_dir / "auroc_['negative_type', 'negative_level', 'negative_cate'].csv"
            receipt = {
                "schema": "arrow.finecops.official_exact_receipt/v1",
                "surface": surface,
                "seed": int(seed),
                "official_repo_commit": OFFICIAL_REPO_COMMIT,
                "official_source": file_record(official_source),
                "annotation": file_record(annotation),
                "predictions": dict(artifact),
                "precision_csv": file_record(precision),
                "recall_csv": file_record(recall),
                "auroc_csv": file_record(auroc),
                "precision": _read_table(precision),
                "recall": _read_table(recall),
                "auroc": _read_table(auroc),
                "returned_result_keys": sorted(str(key) for key in returned),
            }
            receipt_path = output_dir / "receipt.json"
            write_json_atomic(receipt_path, receipt)
            official[surface][seed] = file_record(receipt_path)
    results["official_exact"] = {
        "status": "complete_external_pinned_evaluator",
        "repo_commit": OFFICIAL_REPO_COMMIT,
        "source": file_record(official_source),
        "annotation": file_record(annotation),
        "receipts": official,
    }
    results["official_exact_status"] = "complete_external_pinned_evaluator"
    write_json_atomic(results_path, results)
    return results["official_exact"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    args = parser.parse_args()
    payload = run(
        args.results.resolve(strict=True),
        args.official_repo,
        args.annotation,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
