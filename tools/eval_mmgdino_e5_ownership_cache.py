#!/usr/bin/env python3
"""Evaluate Native or learned ownership routes on a frozen e5 cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_MODES,
)
from tools.responsibility_isolation_cache import (
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    file_sha256,
    normalized_cxcywh_iou,
    validate_cached_candidate_row,
)
from tools.extract_mmgdino_e5_eval_cache import EVAL_CACHE_SCHEMA
from tools.train_mmgdino_e5_ownership import CHECKPOINT_SCHEMA, FORMAL_SEEDS


RECORD_SCHEMA = "arrow.mmgdino_e5_ownership.eval_record/v1"
SUMMARY_SCHEMA = "arrow.mmgdino_e5_ownership.eval_summary/v1"
ROUTES = ("native",) + OWNERSHIP_MODES


class OwnershipEvalError(RuntimeError):
    pass


def load_eval_cache(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    expected = {
        "schema",
        "surface",
        "task",
        "source",
        "feature_dim",
        "box_format",
        "rows",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise OwnershipEvalError("evaluation cache envelope drifted")
    if payload["schema"] != EVAL_CACHE_SCHEMA:
        raise OwnershipEvalError("evaluation cache schema drifted")
    rows = tuple(validate_cached_candidate_row(row) for row in payload["rows"])
    if not rows or {row["task"] for row in rows} != {payload["task"]}:
        raise OwnershipEvalError("evaluation cache task rows drifted")
    return {**payload, "rows": rows}


def exact_q05(values: np.ndarray) -> float:
    score = np.asarray(values, dtype=np.float64).reshape(-1)
    if score.size == 0 or not np.isfinite(score).all():
        raise OwnershipEvalError("positive scores must be nonempty and finite")
    accepted = max(1, int(math.ceil(0.95 * score.size)))
    ascending_index = score.size - accepted
    return float(np.partition(score, ascending_index)[ascending_index])


def binary_auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    if not positive.size or not negative.size:
        raise OwnershipEvalError("AUROC requires positive and negative scores")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
        / comparisons.size
    )


def positive_average_precision(positive: np.ndarray, negative: np.ndarray) -> float:
    scores = np.concatenate((positive, negative)).astype(np.float64)
    labels = np.concatenate(
        (np.ones(len(positive), dtype=np.int8), np.zeros(len(negative), dtype=np.int8))
    )
    # Stable pessimistic tie policy: negatives precede positives at equal score.
    order = np.lexsort((labels, -scores))
    labels = labels[order]
    true_positive = np.cumsum(labels)
    precision = true_positive / np.arange(1, len(labels) + 1)
    return float(precision[labels == 1].mean())


def binary_metrics(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    fixed_threshold: float | None = None,
) -> dict[str, float | None]:
    threshold = exact_q05(positive)
    result = {
        "auroc": binary_auroc(positive, negative),
        "aupr": positive_average_precision(positive, negative),
        "fpr95": float(np.mean(negative >= threshold)),
        "domain_q05": threshold,
        "domain_tpr": float(np.mean(positive >= threshold)),
        "positive_mean": float(np.mean(positive)),
        "negative_mean": float(np.mean(negative)),
        "fixed_threshold": fixed_threshold,
        "fixed_tpr": None,
        "fixed_fpr": None,
    }
    if fixed_threshold is not None:
        if not math.isfinite(fixed_threshold):
            raise OwnershipEvalError("fixed threshold must be finite")
        result["fixed_tpr"] = float(np.mean(positive >= fixed_threshold))
        result["fixed_fpr"] = float(np.mean(negative >= fixed_threshold))
    return result


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_module(
    *, route: str, checkpoint_path: Path | None, device: torch.device
) -> tuple[MMGDinoE5ResponsibilityOwners | None, dict[str, Any] | None]:
    if route == "native":
        if checkpoint_path is not None:
            raise OwnershipEvalError("native route must not load an owner checkpoint")
        return None, None
    if route not in OWNERSHIP_MODES or checkpoint_path is None:
        raise OwnershipEvalError("learned route requires matching checkpoint")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise OwnershipEvalError("owner checkpoint schema drifted")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping) or config.get("ownership") != route:
        raise OwnershipEvalError("owner checkpoint route drifted")
    if config.get("seed") not in FORMAL_SEEDS:
        raise OwnershipEvalError("owner checkpoint seed drifted")
    module = MMGDinoE5ResponsibilityOwners(ownership=route).to(device=device)
    module.load_state_dict(checkpoint["model_state_dict"], strict=True)
    module.eval()
    return module, dict(config)


def _batched_output(
    module: MMGDinoE5ResponsibilityOwners,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            output = module(
                torch.stack([row["query_features"] for row in batch]).to(device),
                torch.stack([row["native_score"] for row in batch]).to(device),
                torch.stack([row["candidate_mask"] for row in batch]).to(device),
            )
            for index in range(len(batch)):
                outputs.append(
                    {
                        key: value[index].detach().cpu()
                        for key, value in output.items()
                    }
                )
    return outputs


def evaluate_cache(
    *,
    cache_path: str | Path,
    route: str,
    surface: str,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    fixed_threshold: float | None = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    if route not in ROUTES:
        raise OwnershipEvalError(f"route must be one of {ROUTES}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise OwnershipEvalError("batch size must be positive")
    cache_path = Path(cache_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    if records_path.exists() or summary_path.exists():
        raise OwnershipEvalError("evaluation outputs already exist")
    shard = load_eval_cache(cache_path)
    rows = list(shard["rows"])
    tasks = {row["task"] for row in rows}
    if len(tasks) != 1:
        raise OwnershipEvalError("evaluation cache must contain exactly one task")
    task = next(iter(tasks))
    torch_device = torch.device(device)
    module, checkpoint_config = _load_module(
        route=route,
        checkpoint_path=(
            None if checkpoint_path is None else Path(checkpoint_path).resolve(strict=True)
        ),
        device=torch_device,
    )
    learned_outputs = (
        None
        if module is None
        else _batched_output(
            module, rows, device=torch_device, batch_size=batch_size
        )
    )
    records = []
    if task == CACHE_TASK_RANK:
        correct = 0
        oracle = 0
        for index, row in enumerate(rows):
            rank_score = (
                row["native_score"]
                if learned_outputs is None
                else learned_outputs[index]["rank_score"]
            )
            top_index = int(rank_score.argmax().item())
            iou = normalized_cxcywh_iou(row["boxes"], row["gt_boxes"]).amax(dim=1)
            top_iou = float(iou[top_index])
            row_oracle = bool((iou >= 0.5).any().item())
            row_correct = top_iou >= 0.5
            correct += int(row_correct)
            oracle += int(row_oracle)
            records.append(
                {
                    "schema": RECORD_SCHEMA,
                    "surface": surface,
                    "route": route,
                    "sample_id": row["sample_id"],
                    "image_id": row["image_id"],
                    "top_query": top_index,
                    "top_iou": top_iou,
                    "correct_iou50": row_correct,
                    "oracle_iou50": row_oracle,
                }
            )
        metrics = {
            "p1_iou50": correct / len(rows),
            "correct": correct,
            "rows": len(rows),
            "oracle_iou50": oracle / len(rows),
            "oracle_rows": oracle,
        }
    elif task == CACHE_TASK_CONFIDENCE_PAIR:
        pair_rows: dict[str, dict[str, tuple[Mapping, Mapping | None]]] = {}
        for index, row in enumerate(rows):
            pair_rows.setdefault(row["pair_id"], {})[row["pair_role"]] = (
                row,
                None if learned_outputs is None else learned_outputs[index],
            )
        if any(set(value) != {"positive", "negative"} for value in pair_rows.values()):
            raise OwnershipEvalError("paired cache is incomplete")
        positive_scores = []
        negative_scores = []
        for pair_id in sorted(pair_rows):
            roles = pair_rows[pair_id]
            pair_record = {
                "schema": RECORD_SCHEMA,
                "surface": surface,
                "route": route,
                "pair_id": pair_id,
            }
            for role in ("positive", "negative"):
                row, output = roles[role]
                score = (
                    row["native_score"]
                    if output is None
                    else output["confidence_score"]
                )
                mask = row["candidate_mask"] if output is None else output["candidate_mask"]
                sample_score = float(score.masked_fill(~mask, -torch.inf).max())
                pair_record[f"{role}_sample_id"] = row["sample_id"]
                pair_record[f"{role}_score"] = sample_score
                if role == "positive":
                    pair_record["image_id"] = row["image_id"]
                    positive_scores.append(sample_score)
                else:
                    negative_scores.append(sample_score)
            records.append(pair_record)
        metrics = binary_metrics(
            np.asarray(positive_scores),
            np.asarray(negative_scores),
            fixed_threshold=fixed_threshold,
        )
        metrics["pairs"] = len(pair_rows)
    else:
        raise OwnershipEvalError(f"unsupported cache task {task!r}")
    record_text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_text(records_path, record_text)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "surface": surface,
        "route": route,
        "task": task,
        "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
        "checkpoint": (
            None
            if checkpoint_path is None
            else {
                "path": str(Path(checkpoint_path).resolve(strict=True)),
                "sha256": file_sha256(Path(checkpoint_path)),
                "config": checkpoint_config,
            }
        ),
        "records": {
            "path": str(records_path),
            "sha256": file_sha256(records_path),
            "rows": len(records),
        },
        "metrics": metrics,
        "runtime": {"device": device, "batch_size": batch_size},
    }
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = evaluate_cache(
        cache_path=args.cache,
        route=args.route,
        surface=args.surface,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        fixed_threshold=args.fixed_threshold,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "RECORD_SCHEMA",
    "ROUTES",
    "SUMMARY_SCHEMA",
    "OwnershipEvalError",
    "binary_auroc",
    "binary_metrics",
    "evaluate_cache",
    "exact_q05",
    "positive_average_precision",
]
