#!/usr/bin/env python3
"""Run zero-update cross-dataset rank probes on frozen U150 shared owners."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from tools.build_mmgdino_e5_cross_dataset_probe_manifests import (
    DATASETS,
    PROBE_BATCHES,
    PROBE_BATCH_SIZE,
    RECEIPT_SCHEMA as SELECTION_RECEIPT_SCHEMA,
    SEEDS,
)
from tools.eval_mmgdino_e5_ownership_cache import load_eval_cache
from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
)
from tools.responsibility_isolation_cache import (
    CACHE_TASK_CONFIDENCE_PAIR,
    file_sha256,
    load_cached_candidate_shard,
    normalized_cxcywh_iou,
)
from tools.train_mmgdino_e5_ownership import (
    CHECKPOINT_SCHEMA,
    D3QueueState,
    FORMAL_PROBE_BATCHES,
    _cache_indices,
    _confidence_loss,
    _rank_loss,
    _tensor_mapping_sha256,
)


SCHEMA = "arrow.mmgdino_e5_cross_dataset_probe.results/v1"
ROUTES = (OWNERSHIP_SHARED_128, OWNERSHIP_SHARED_WIDE)


class CrossDatasetProbeError(RuntimeError):
    pass


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise CrossDatasetProbeError("probe summary requires finite values")
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _flat_gradients(
    loss: Tensor, parameters: Sequence[torch.nn.Parameter]
) -> Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=False, allow_unused=False
    )
    flat = torch.cat(
        [gradient.detach().float().reshape(-1) for gradient in gradients]
    )
    if not bool(torch.isfinite(flat).all().item()) or not bool((flat != 0).any().item()):
        raise CrossDatasetProbeError("probe produced nonfinite or zero gradient")
    return flat


def _tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(array).hexdigest()


def _gradient_metrics(rank_gradient: Tensor, confidence_gradient: Tensor) -> dict[str, float]:
    rank_norm = torch.linalg.vector_norm(rank_gradient)
    confidence_norm = torch.linalg.vector_norm(confidence_gradient)
    if not bool((rank_norm > 0).item()) or not bool((confidence_norm > 0).item()):
        raise CrossDatasetProbeError("probe encountered a zero gradient norm")
    cosine = torch.dot(rank_gradient, confidence_gradient) / (
        rank_norm * confidence_norm
    )
    jointly_nonzero = (rank_gradient != 0) & (confidence_gradient != 0)
    if not bool(jointly_nonzero.any().item()):
        raise CrossDatasetProbeError("probe gradients have no jointly nonzero elements")
    conflict = (
        torch.sign(rank_gradient[jointly_nonzero])
        != torch.sign(confidence_gradient[jointly_nonzero])
    ).float().mean()
    return {
        "cosine": float(cosine),
        "sign_conflict_fraction": float(conflict),
        "rank_gradient_l2": float(rank_norm),
        "confidence_gradient_l2": float(confidence_norm),
    }


def _rank_diagnostics(
    module: MMGDinoE5ResponsibilityOwners,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    loss, loss_metrics = _rank_loss(module, rows, device=device)
    native = torch.stack([row["native_score"] for row in rows]).to(device)
    mask = torch.stack([row["candidate_mask"] for row in rows]).to(device)
    masked_native = native.float().masked_fill(~mask, -torch.inf)
    top_values, top_indices = torch.topk(masked_native, k=2, dim=1)
    if not bool(torch.isfinite(top_values).all().item()):
        raise CrossDatasetProbeError("rank probe row has fewer than two valid candidates")
    candidate_iou = torch.stack(
        [
            normalized_cxcywh_iou(row["boxes"], row["gt_boxes"]).amax(dim=1)
            for row in rows
        ]
    ).to(device)
    native_correct = candidate_iou.gather(1, top_indices[:, :1]).squeeze(1) >= 0.5
    positive = candidate_iou >= 0.5
    negative = candidate_iou < 0.5
    valid_gap = positive.any(dim=1) & negative.any(dim=1)
    positive_score = masked_native.masked_fill(~positive, -torch.inf).max(dim=1).values
    negative_score = masked_native.masked_fill(~negative, -torch.inf).max(dim=1).values
    oracle_gap = positive_score[valid_gap] - negative_score[valid_gap]
    diagnostics = {
        "native_p1": float(native_correct.float().mean()),
        "rank_loss": float(loss.detach()),
        "rank_fix_loss": loss_metrics["fix_loss"],
        "rank_preserve_loss": loss_metrics["preserve_loss"],
        "native_top1_runnerup_margin": float((top_values[:, 0] - top_values[:, 1]).mean()),
        "native_oracle_positive_negative_gap": (
            float(oracle_gap.mean()) if int(oracle_gap.numel()) else 0.0
        ),
        "rows_with_iou50_candidate": float(positive.any(dim=1).float().sum()),
        "rows": float(len(rows)),
    }
    return loss, diagnostics


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrossDatasetProbeError(f"JSON object required: {path}")
    return value


def _confidence_probe_batches(
    *, seed: int, training_root: Path
) -> tuple[list[list[Mapping[str, Mapping[str, Any]]]], Mapping[str, Any]]:
    schedule_path = training_root / f"schedules/schedule_seed{seed}.json"
    cache_path = training_root / f"caches/seed{seed}.pt"
    schedule = _load_json(schedule_path)
    shard = load_cached_candidate_shard(cache_path)
    _, pair_index = _cache_indices(shard)
    batches = [
        [pair_index[identity] for identity in update["identities"]]
        for update in schedule["updates"]
        if update["task"] == CACHE_TASK_CONFIDENCE_PAIR
    ][:FORMAL_PROBE_BATCHES]
    if len(batches) != PROBE_BATCHES or any(
        len(batch) != 8 for batch in batches
    ):
        raise CrossDatasetProbeError("fixed confidence probe batches drifted")
    return batches, {
        "schedule": {"path": str(schedule_path), "sha256": file_sha256(schedule_path)},
        "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
    }


def _rank_probe_batches(
    *, dataset: str, seed: int, selection: Mapping[str, Any], cache_root: Path
) -> tuple[list[list[Mapping[str, Any]]], Mapping[str, Any]]:
    binding = selection["datasets"][dataset]
    schedule_path = Path(binding["schedule"]["path"])
    if file_sha256(schedule_path) != binding["schedule"]["sha256"]:
        raise CrossDatasetProbeError("rank probe schedule SHA drifted")
    schedule = _load_json(schedule_path)
    cache_path = cache_root / f"{dataset}.pt"
    shard = load_eval_cache(cache_path)
    index = {row["sample_id"]: row for row in shard["rows"]}
    identities = schedule["batches"][str(seed)]
    batches = [[index[identity] for identity in batch] for batch in identities]
    if len(batches) != PROBE_BATCHES or any(
        len(batch) != PROBE_BATCH_SIZE for batch in batches
    ):
        raise CrossDatasetProbeError("rank probe batch contract drifted")
    return batches, {
        "schedule": binding["schedule"],
        "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
    }


@contextlib.contextmanager
def _deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def run(
    *,
    selection_receipt: Path,
    cache_root: Path,
    training_root: Path,
    output: Path,
    device_name: str,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise CrossDatasetProbeError(f"probe output already exists: {output}")
    selection_receipt = selection_receipt.resolve(strict=True)
    selection = _load_json(selection_receipt)
    if selection.get("schema") != SELECTION_RECEIPT_SCHEMA:
        raise CrossDatasetProbeError("selection receipt schema drifted")
    if selection.get("status") != "complete_before_model_forward":
        raise CrossDatasetProbeError("selection receipt is not sealed")
    cache_root = cache_root.resolve(strict=True)
    training_root = training_root.resolve(strict=True)
    device = torch.device(device_name)
    results: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    with _deterministic_algorithms():
        for route in ROUTES:
            route_results = {}
            for seed in SEEDS:
                checkpoint_path = training_root / f"formal/{route}/seed{seed}/checkpoint_u150.pt"
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
                    raise CrossDatasetProbeError("U150 checkpoint schema drifted")
                if checkpoint.get("config", {}).get("ownership") != route:
                    raise CrossDatasetProbeError("U150 checkpoint route drifted")
                if checkpoint.get("config", {}).get("seed") != seed:
                    raise CrossDatasetProbeError("U150 checkpoint seed drifted")
                module = MMGDinoE5ResponsibilityOwners(ownership=route).to(device)
                module.load_state_dict(checkpoint["model_state_dict"], strict=True)
                module.train()
                state_before = _tensor_mapping_sha256(module.state_dict())
                shared = module.shared_parameters()
                confidence_batches, confidence_binding = _confidence_probe_batches(
                    seed=seed, training_root=training_root
                )
                queue = D3QueueState.empty(size=512)
                queue.load_state_dict(checkpoint["d3_queue_state_dict"])
                confidence_gradients = []
                confidence_gradient_shas = []
                confidence_losses = []
                for pairs in confidence_batches:
                    confidence_loss, _, _ = _confidence_loss(
                        module, pairs, queue, device=device
                    )
                    confidence_losses.append(float(confidence_loss.detach()))
                    gradient = _flat_gradients(confidence_loss, shared)
                    confidence_gradients.append(gradient)
                    confidence_gradient_shas.append(_tensor_sha256(gradient))
                dataset_results = {}
                dataset_bindings = {}
                for dataset in DATASETS:
                    rank_batches, rank_binding = _rank_probe_batches(
                        dataset=dataset,
                        seed=seed,
                        selection=selection,
                        cache_root=cache_root,
                    )
                    batch_results = []
                    for batch_index, (rows, confidence_gradient) in enumerate(
                        zip(rank_batches, confidence_gradients)
                    ):
                        rank_loss, diagnostics = _rank_diagnostics(
                            module, rows, device=device
                        )
                        rank_gradient = _flat_gradients(rank_loss, shared)
                        batch_results.append(
                            {
                                "batch": batch_index,
                                **_gradient_metrics(rank_gradient, confidence_gradient),
                                **diagnostics,
                                "confidence_loss": confidence_losses[batch_index],
                                "confidence_gradient_sha256": confidence_gradient_shas[batch_index],
                            }
                        )
                    summary = {
                        metric: _mean_sd([row[metric] for row in batch_results])
                        for metric in (
                            "cosine",
                            "sign_conflict_fraction",
                            "rank_gradient_l2",
                            "confidence_gradient_l2",
                            "native_p1",
                            "rank_loss",
                            "rank_fix_loss",
                            "rank_preserve_loss",
                            "native_top1_runnerup_margin",
                            "native_oracle_positive_negative_gap",
                        )
                    }
                    summary["rows"] = sum(row["rows"] for row in batch_results)
                    summary["rows_with_iou50_candidate"] = sum(
                        row["rows_with_iou50_candidate"] for row in batch_results
                    )
                    dataset_results[dataset] = {
                        "batches": batch_results,
                        "summary": summary,
                    }
                    dataset_bindings[dataset] = rank_binding
                state_after = _tensor_mapping_sha256(module.state_dict())
                if state_after != state_before:
                    raise CrossDatasetProbeError("zero-update probe changed model state")
                route_results[str(seed)] = {
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": file_sha256(checkpoint_path),
                        "model_state_sha256_before": state_before,
                        "model_state_sha256_after": state_after,
                    },
                    "confidence_probe": {
                        **confidence_binding,
                        "gradient_sha256_by_batch": confidence_gradient_shas,
                    },
                    "datasets": dataset_results,
                }
                bindings[f"{route}:{seed}"] = dataset_bindings
                del module, checkpoint, confidence_gradients
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            results[route] = route_results
    payload = {
        "schema": SCHEMA,
        "status": "complete_zero_update",
        "contract": {
            "routes": list(ROUTES),
            "seeds": list(SEEDS),
            "datasets": list(DATASETS),
            "rank_probe_batches": PROBE_BATCHES,
            "rank_batch_size": PROBE_BATCH_SIZE,
            "confidence_probe": "same first eight formal D3 batches and sealed U150 queue per route/seed",
            "parameters_updated": 0,
            "optimizer_created": False,
            "native_top1_margin_definition": "largest minus second-largest valid native query score",
            "native_oracle_gap_definition": "best IoU>=0.5 native score minus best IoU<0.5 native score",
        },
        "selection_receipt": {
            "path": str(selection_receipt),
            "sha256": file_sha256(selection_receipt),
        },
        "bindings": bindings,
        "results": results,
    }
    _atomic_json(payload, output)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    value = run(
        selection_receipt=args.selection_receipt,
        cache_root=args.cache_root,
        training_root=args.training_root,
        output=args.output,
        device_name=args.device,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
