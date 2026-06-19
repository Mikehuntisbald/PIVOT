#!/usr/bin/env python3
"""Calibrate pure-GDINO allTN tau/weight on the actual Stage-B train mix."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from main import _torch_load_compat, build_model_main, get_args_parser  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402


PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Measure gdino allTN score/loss scales",
        parents=[get_args_parser()],
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint to load. Use the same start checkpoint intended for the ablation.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=120,
        help="Number of weighted-sampler train batches to measure.",
    )
    parser.add_argument(
        "--output_json",
        default="outputs/gdino_alltn_calibration/calibration.json",
        help="Where to write the calibration result.",
    )
    parser.add_argument(
        "--target_ratios",
        nargs="+",
        type=float,
        default=[0.02, 0.05, 0.10, 0.20],
        help="Desired weighted allTN/base-loss ratios used to backsolve weights.",
    )
    parser.add_argument(
        "--tau_candidates",
        nargs="*",
        type=float,
        default=None,
        help="Optional explicit tau_neg candidates. Quantile candidates are always added.",
    )
    parser.add_argument(
        "--calibration_model_mode",
        choices=("eval", "train"),
        default="eval",
        help="Model mode for forward passes. eval is deterministic; train matches dropout.",
    )
    return parser


def _merge_config_into_args(args: argparse.Namespace) -> argparse.Namespace:
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    allow_cfg_override = {"fix_size"}
    for key, value in cfg_dict.items():
        if key not in args_vars:
            setattr(args, key, value)
        elif key in allow_cfg_override:
            setattr(args, key, value)
        else:
            raise ValueError(f"Key {key} can be used by args only")
    if not getattr(args, "debug", None):
        args.debug = False
    return args


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(args: argparse.Namespace, device: torch.device):
    model, criterion, _ = build_model_main(args)
    checkpoint = _torch_load_compat(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    load_result = model.load_state_dict(clean_state_dict(state), strict=False)
    print(f"Loaded {args.checkpoint}: {load_result}", flush=True)
    model.to(device)
    criterion.to(device)
    if args.calibration_model_mode == "train":
        model.train()
        criterion.train()
    else:
        model.eval()
        criterion.eval()
    return model, criterion


def _dataset_sample_weights(dataset_list: Sequence[Any], mix_weights: Sequence[float]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    sample_weights: List[float] = []
    expected: List[Dict[str, Any]] = []
    has_explicit_mix_weights = any(abs(float(w) - 1.0) > 1e-12 for w in mix_weights)
    total_mix = sum(float(w) for w in mix_weights) if has_explicit_mix_weights else sum(len(ds) for ds in dataset_list)
    for idx, (dataset, mix_weight) in enumerate(zip(dataset_list, mix_weights)):
        ds_len = max(1, len(dataset))
        base_weight = float(mix_weight) / float(ds_len) if has_explicit_mix_weights else 1.0
        ds_sample_weights = getattr(dataset, "sample_weights", None)
        if ds_sample_weights is not None and len(ds_sample_weights) == len(dataset):
            sample_weights.extend([base_weight * float(w) for w in ds_sample_weights])
        else:
            sample_weights.extend([base_weight] * len(dataset))
        expected_fraction = (
            float(mix_weight) / float(total_mix)
            if has_explicit_mix_weights and total_mix > 0
            else float(len(dataset)) / float(total_mix) if total_mix > 0 else 0.0
        )
        expected.append(
            {
                "dataset_idx": idx,
                "len": len(dataset),
                "mix_weight": float(mix_weight),
                "expected_fraction": expected_fraction,
                "tn_balance_stats": getattr(dataset, "tn_balance_stats", None),
            }
        )
    return torch.as_tensor(sample_weights, dtype=torch.double), expected


def _build_train_loader(args: argparse.Namespace) -> Tuple[DataLoader, List[Dict[str, Any]]]:
    with open(args.datasets, "r") as f:
        dataset_meta = json.load(f)
    train_infos = dataset_meta["train"]
    if len(train_infos) == 1:
        dataset = build_dataset(image_set="train", args=args, datasetinfo=train_infos[0])
        dataset_list = [dataset]
        mix_weights = [float(train_infos[0].get("mix_weight", 1.0))]
    else:
        dataset_list = [
            build_dataset(image_set="train", args=args, datasetinfo=datasetinfo)
            for datasetinfo in train_infos
        ]
        mix_weights = [float(datasetinfo.get("mix_weight", 1.0)) for datasetinfo in train_infos]
        dataset = ConcatDataset(dataset_list)

    sample_weights, expected_mix = _dataset_sample_weights(dataset_list, mix_weights)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=max(1, int(args.max_batches) * int(args.batch_size)),
        replacement=True,
        generator=generator,
    )
    batch_sampler = torch.utils.data.BatchSampler(sampler, int(args.batch_size), drop_last=True)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=utils.collate_fn,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(args.device).type == "cuda",
    )
    return loader, expected_mix


def _move_targets_like_engine(targets: Sequence[Dict[str, Any]], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    moved = [
        {
            key: value.to(device)
            for key, value in target.items()
            if torch.is_tensor(value) and key not in {"patch", "patch_global"}
        }
        for target in targets
    ]
    mask_kwargs: Dict[str, torch.Tensor] = {}
    for mask_key in (
        "phrase_to_token_mask",
        "canonical_to_token_mask",
        "content_to_token_mask",
        "attr_pos_to_token_mask",
        "attr_neg_to_token_mask",
        "negative_to_token_mask",
        "is_tn",
    ):
        if not all(mask_key in target for target in targets):
            continue
        values = [target[mask_key] for target in targets]
        if not all(torch.is_tensor(value) for value in values):
            continue
        if len({tuple(value.shape) for value in values}) == 1:
            mask_kwargs[mask_key] = torch.stack(values, dim=0).to(device, non_blocking=True)
        elif all(value.dim() == 2 for value in values):
            kmax = max(int(value.shape[0]) for value in values)
            tmax = max(int(value.shape[1]) for value in values)
            padded = values[0].new_zeros((len(values), kmax, tmax))
            for idx, value in enumerate(values):
                padded[idx, : int(value.shape[0]), : int(value.shape[1])] = value
            mask_kwargs[mask_key] = padded.to(device, non_blocking=True)
        elif all(value.dim() == 1 for value in values):
            kmax = max(int(value.shape[0]) for value in values)
            padded = values[0].new_zeros((len(values), kmax))
            for idx, value in enumerate(values):
                padded[idx, : int(value.shape[0])] = value
            mask_kwargs[mask_key] = padded.to(device, non_blocking=True)
    return moved, mask_kwargs


def _target_is_tn(criterion: Any, target: Dict[str, Any], device: torch.device) -> bool:
    if hasattr(criterion, "_target_is_tn"):
        return bool(criterion._target_is_tn(target, device=device))
    value = target.get("is_tn", None)
    if value is None:
        return False
    value = value.to(device=device) if torch.is_tensor(value) else torch.as_tensor(value, device=device)
    return bool(value.numel() > 0 and value.to(torch.bool).all().item())


def _query_scores_from_logits(
    pred_logits_b: torch.Tensor,
    text_mask_b: torch.Tensor,
    text_agg: str,
) -> torch.Tensor:
    probs_b = pred_logits_b.sigmoid()
    if text_agg == "max":
        return probs_b.masked_fill(~text_mask_b[None, :], 0.0).max(dim=-1).values
    denom = text_mask_b.to(dtype=probs_b.dtype).sum().clamp(min=1.0)
    return probs_b.masked_fill(~text_mask_b[None, :], 0.0).sum(dim=-1) / denom


def _tn_layer_aggs(
    outputs_layer: Dict[str, torch.Tensor],
    targets: Sequence[Dict[str, Any]],
    criterion: Any,
    *,
    topk: int,
    lse_tau: float,
    text_agg: str,
) -> Tuple[List[float], List[float]]:
    pred_logits = outputs_layer["pred_logits"]
    text_mask = outputs_layer.get("text_mask", None)
    if text_mask is None:
        text_mask = torch.ones(
            pred_logits.shape[0],
            pred_logits.shape[-1],
            dtype=torch.bool,
            device=pred_logits.device,
        )
    else:
        text_mask = text_mask.to(device=pred_logits.device, dtype=torch.bool)
    aggs: List[float] = []
    maxes: List[float] = []
    tau = max(float(lse_tau), 1e-6)
    for batch_idx, target in enumerate(targets):
        if not _target_is_tn(criterion, target, pred_logits.device):
            continue
        mask = text_mask[batch_idx]
        if not bool(mask.any().item()):
            continue
        query_scores = _query_scores_from_logits(pred_logits[batch_idx], mask, text_agg)
        k = min(max(1, int(topk)), int(query_scores.numel()))
        topk_scores = torch.topk(query_scores, k=k, largest=True).values
        agg = tau * torch.logsumexp(topk_scores / tau, dim=0)
        aggs.append(float(agg.detach().cpu()))
        maxes.append(float(topk_scores.max().detach().cpu()))
    return aggs, maxes


def _all_layers(outputs: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, torch.Tensor]]]:
    text_mask = outputs.get("text_mask", None)
    yield "main", outputs
    for idx, aux in enumerate(outputs.get("aux_outputs", []) or []):
        if "text_mask" not in aux and text_mask is not None:
            aux = dict(aux)
            aux["text_mask"] = text_mask
        yield f"aux_{idx}", aux
    if "interm_outputs" in outputs:
        interm = outputs["interm_outputs"]
        if "text_mask" not in interm and text_mask is not None:
            interm = dict(interm)
            interm["text_mask"] = text_mask
        yield "interm", interm


def _alltn_weight_key(layer_name: str) -> str:
    if layer_name == "main":
        return "loss_tn_alltn"
    if layer_name.startswith("aux_"):
        return "loss_tn_alltn_" + layer_name.split("_", 1)[1]
    if layer_name == "interm":
        return "loss_tn_alltn_interm"
    raise ValueError(f"Unknown layer name: {layer_name}")


def _alltn_weight_coeff(layer_name: str, weight_dict: Dict[str, float], base_weight: float) -> float:
    key = _alltn_weight_key(layer_name)
    if key not in weight_dict:
        return 0.0
    if base_weight <= 0:
        return 1.0
    return float(weight_dict[key]) / float(base_weight)


def _candidate_unweighted_sum(
    outputs: Dict[str, Any],
    targets: Sequence[Dict[str, Any]],
    criterion: Any,
    tau_neg: float,
    *,
    topk: int,
    lse_tau: float,
    text_agg: str,
) -> float:
    total = 0.0
    for _name, layer_outputs in _all_layers(outputs):
        aggs, _ = _tn_layer_aggs(
            layer_outputs,
            targets,
            criterion,
            topk=topk,
            lse_tau=lse_tau,
            text_agg=text_agg,
        )
        if aggs:
            values = torch.as_tensor(aggs, dtype=torch.float32)
            raw = F.softplus(values - float(tau_neg)).mean().item()
            total += float(raw)
    return total


def _matched_positive_scores(outputs: Dict[str, Any], targets: Sequence[Dict[str, Any]], criterion: Any) -> List[float]:
    one_hot = outputs.get("one_hot", None)
    if one_hot is None:
        return []
    pred_logits = outputs["pred_logits"]
    text_mask = outputs.get("text_mask", None)
    if text_mask is None:
        text_mask = torch.ones(
            pred_logits.shape[0],
            pred_logits.shape[-1],
            dtype=torch.bool,
            device=pred_logits.device,
        )
    else:
        text_mask = text_mask.to(device=pred_logits.device, dtype=torch.bool)
    one_hot = one_hot.to(device=pred_logits.device, dtype=torch.bool)
    scores: List[float] = []
    for batch_idx, target in enumerate(targets):
        if _target_is_tn(criterion, target, pred_logits.device):
            continue
        query_has_pos = one_hot[batch_idx].any(dim=-1)
        query_ids = torch.nonzero(query_has_pos, as_tuple=False).flatten()
        for query_id in query_ids:
            mask = one_hot[batch_idx, query_id] & text_mask[batch_idx]
            if not bool(mask.any().item()):
                continue
            score = pred_logits[batch_idx, query_id].sigmoid()[mask].mean()
            scores.append(float(score.detach().cpu()))
    return scores


def _weighted_loss_parts(loss_dict: Dict[str, torch.Tensor], weight_dict: Dict[str, float]) -> Dict[str, float]:
    parts = {"base": 0.0, "tn_alltn": 0.0, "tn_tokens": 0.0, "total": 0.0}
    for key, value in loss_dict.items():
        if key not in weight_dict:
            continue
        weighted = float((value.detach() * float(weight_dict[key])).cpu())
        parts["total"] += weighted
        if key.startswith("loss_tn_alltn"):
            parts["tn_alltn"] += weighted
        elif key.startswith("loss_tn_tokens"):
            parts["tn_tokens"] += weighted
        else:
            parts["base"] += weighted
    return parts


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    out: Dict[str, Any] = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for pct in PERCENTILES:
        out[f"p{pct}"] = float(np.percentile(arr, pct))
    return out


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.asarray(values, dtype=np.float64).mean())


def _unique_sorted(values: Iterable[float], digits: int = 6) -> List[float]:
    uniq = {round(float(value), digits) for value in values if math.isfinite(float(value))}
    return sorted(uniq)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args = _merge_config_into_args(args)
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    args.local_rank = getattr(args, "local_rank", None)
    args.output_dir = args.output_dir or str(Path(args.output_json).resolve().parent)
    _set_reproducible_seed(int(args.seed))

    device = torch.device(args.device)
    model, criterion = _load_model(args, device)
    loader, expected_mix = _build_train_loader(args)

    topk = int(getattr(args, "gdino_tn_alltn_topk", 10))
    lse_tau = float(getattr(args, "gdino_tn_alltn_lse_tau", 0.2))
    current_tau_neg = float(getattr(args, "gdino_tn_alltn_tau_neg", 0.0625))
    current_weight = float(getattr(args, "gdino_tn_alltn_weight", 0.0625))
    text_agg = str(getattr(args, "gdino_tn_alltn_text_agg", "mean")).lower().strip()

    base_parts: List[float] = []
    alltn_parts: List[float] = []
    token_parts: List[float] = []
    total_parts: List[float] = []
    tn_agg_main: List[float] = []
    tn_topk_max_main: List[float] = []
    matched_pos_scores: List[float] = []
    tn_samples_per_batch: List[float] = []
    candidate_batch_layers: List[List[Dict[str, Any]]] = []

    candidate_seed_taus = [current_tau_neg]

    max_batches = int(args.max_batches)
    print(
        f"Measuring {max_batches} batches, batch_size={args.batch_size}, "
        f"topk={topk}, lse_tau={lse_tau}, current_tau_neg={current_tau_neg}, "
        f"current_weight={current_weight}",
        flush=True,
    )

    with torch.no_grad():
        for batch_idx, (samples, raw_targets) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            samples = samples.to(device)
            captions = [target["caption"] for target in raw_targets]
            cap_list = [target["cap_list"] for target in raw_targets]
            patches = None
            if all(("patch" in target) for target in raw_targets):
                patches = torch.stack([target["patch"] for target in raw_targets], dim=0).to(device, non_blocking=True)
            patch_global = None
            if all(("patch_global" in target) for target in raw_targets):
                patch_global = torch.stack([target["patch_global"] for target in raw_targets], dim=0).to(device, non_blocking=True)
            targets, gdino_mask_kwargs = _move_targets_like_engine(raw_targets, device)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                outputs = model(
                    samples,
                    captions=captions,
                    patches=patches,
                    patch_global=patch_global,
                    **gdino_mask_kwargs,
                )
                loss_dict = criterion(outputs, targets, cap_list, captions)
            parts = _weighted_loss_parts(loss_dict, criterion.weight_dict)
            base_parts.append(parts["base"])
            alltn_parts.append(parts["tn_alltn"])
            token_parts.append(parts["tn_tokens"])
            total_parts.append(parts["total"])

            batch_layers: List[Dict[str, Any]] = []
            main_aggs: List[float] = []
            for layer_name, layer_outputs in _all_layers(outputs):
                aggs, maxes = _tn_layer_aggs(
                    layer_outputs,
                    targets,
                    criterion,
                    topk=topk,
                    lse_tau=lse_tau,
                    text_agg=text_agg,
                )
                coeff = _alltn_weight_coeff(layer_name, criterion.weight_dict, current_weight)
                batch_layers.append({"name": layer_name, "aggs": aggs, "coeff": coeff})
                if layer_name == "main":
                    main_aggs = aggs
                    tn_agg_main.extend(aggs)
                    tn_topk_max_main.extend(maxes)
            tn_samples_per_batch.append(float(len(main_aggs)))
            matched_pos_scores.extend(_matched_positive_scores(outputs, targets, criterion))
            candidate_batch_layers.append(batch_layers)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == max_batches:
                print(
                    f"batch {batch_idx + 1}/{max_batches}: "
                    f"base={_mean(base_parts):.4f} alltn={_mean(alltn_parts):.4f} "
                    f"tn_tokens={_mean(token_parts):.4f} tn_samples={int(sum(tn_samples_per_batch))}",
                    flush=True,
                )

    agg_stats = _stats(tn_agg_main)
    max_stats = _stats(tn_topk_max_main)
    pos_stats = _stats(matched_pos_scores)
    for key in ("p10", "p25", "p50", "p75", "p90"):
        if key in agg_stats:
            candidate_seed_taus.append(float(agg_stats[key]))
    if args.tau_candidates:
        candidate_seed_taus.extend(args.tau_candidates)
    tau_candidates = _unique_sorted(candidate_seed_taus)

    candidate_rows: List[Dict[str, Any]] = []
    base_mean = _mean(base_parts)
    for tau_neg in tau_candidates:
        batch_main_raws = []
        batch_effective_raws = []
        for layers in candidate_batch_layers:
            effective = 0.0
            main_raw = 0.0
            saw_main = False
            for layer in layers:
                aggs = layer["aggs"]
                coeff = float(layer["coeff"])
                if not aggs or coeff == 0:
                    if layer["name"] == "main":
                        saw_main = True
                    continue
                values = torch.as_tensor(aggs, dtype=torch.float32)
                raw = float(F.softplus(values - float(tau_neg)).mean().item())
                effective += coeff * raw
                if layer["name"] == "main":
                    main_raw = raw
                    saw_main = True
            batch_effective_raws.append(effective)
            batch_main_raws.append(main_raw if saw_main else 0.0)
        raw_mean_main = _mean(batch_main_raws)
        raw_mean_effective = _mean(batch_effective_raws)
        current_ratio_effective = (
            current_weight * raw_mean_effective / base_mean if base_mean > 0 else float("nan")
        )
        suggestions = {
            f"ratio_{ratio:g}": (ratio * base_mean / raw_mean_effective if raw_mean_effective > 0 else None)
            for ratio in args.target_ratios
        }
        candidate_rows.append(
            {
                "tau_neg": float(tau_neg),
                "main_raw_loss_mean": raw_mean_main,
                "effective_raw_loss_mean_all_weighted_layers": raw_mean_effective,
                "current_weight_effective_ratio_to_base": current_ratio_effective,
                "weight_for_target_effective_ratio_to_base": suggestions,
            }
        )

    result = {
        "config_file": args.config_file,
        "datasets": args.datasets,
        "checkpoint": args.checkpoint,
        "model_mode": args.calibration_model_mode,
        "batch_size": int(args.batch_size),
        "max_batches": max_batches,
        "measured_batches": len(base_parts),
        "alltn_formula": "sigmoid token probs -> valid-token mean per query -> top-k -> lse_tau*logsumexp(topk/lse_tau) -> softplus(agg-tau_neg)",
        "current": {
            "gdino_tn_alltn_tau_neg": current_tau_neg,
            "gdino_tn_alltn_weight": current_weight,
            "gdino_tn_alltn_topk": topk,
            "gdino_tn_alltn_lse_tau": lse_tau,
            "gdino_tn_alltn_text_agg": text_agg,
        },
        "expected_train_mix": expected_mix,
        "loss_weighted_means": {
            "base": base_mean,
            "tn_alltn_current": _mean(alltn_parts),
            "tn_tokens": _mean(token_parts),
            "total": _mean(total_parts),
            "tn_alltn_current_ratio_to_base": (_mean(alltn_parts) / base_mean if base_mean > 0 else None),
            "tn_tokens_ratio_to_base": (_mean(token_parts) / base_mean if base_mean > 0 else None),
        },
        "tn_main_agg_stats": agg_stats,
        "tn_main_topk_max_stats": max_stats,
        "matched_positive_phrase_score_stats": pos_stats,
        "tn_samples_per_batch_stats": _stats(tn_samples_per_batch),
        "candidate_tau_weight_table": candidate_rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output_json}", flush=True)

    print("\nKey calibration:", flush=True)
    print(json.dumps(result["loss_weighted_means"], indent=2), flush=True)
    print("tn_main_agg_stats:", json.dumps(agg_stats, indent=2), flush=True)
    print("candidate_tau_weight_table:", json.dumps(candidate_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
