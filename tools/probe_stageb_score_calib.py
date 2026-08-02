#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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
from engine import _build_stage_b_mask_kwargs, _build_stage_b_rank_subbatch, _stage_b_patch_slot_count  # noqa: E402
from models.GroundingDINO.stage_b_score import compute_stage_b_slot_logits  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from tools.eval_stagea_patch_checkpoints import _prepare_patch_batch, _set_seed, _torch_load_compat  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _expand_dataset_meta(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_dataset_meta(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_dataset_meta(v) for k, v in value.items()}
    return value


def _load_model_and_criterion(cfg, ckpt_path: str, device: torch.device):
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={cfg.modelname}")
    model, criterion, _postprocessors = build_func(cfg)
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    state = clean_state_dict(_extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys={len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] unexpected keys={len(unexpected)}", file=sys.stderr)
    model.to(device).eval()
    criterion.to(device).eval()
    return model, criterion


def _make_train_loader(cfg, datasets_path: str, batch_size: int, num_workers: int, device: torch.device, seed: int):
    with open(datasets_path, "r", encoding="utf-8") as f:
        dataset_meta = _expand_dataset_meta(json.load(f))
    train_infos = list(dataset_meta.get("train", []))
    if not train_infos:
        raise ValueError(f"No train datasets found in {datasets_path}")
    datasets = [build_dataset(image_set="train", args=cfg, datasetinfo=info) for info in train_infos]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    weights: List[float] = []
    if len(datasets) == 1:
        ds_weights = getattr(datasets[0], "sample_weights", None)
        weights = [float(x) for x in ds_weights] if ds_weights is not None else [1.0] * len(datasets[0])
    else:
        explicit_mix = [float(info.get("mix_weight", 1.0)) for info in train_infos]
        total_mix = sum(explicit_mix)
        for ds, mix_weight in zip(datasets, explicit_mix):
            base = float(mix_weight) / max(float(len(ds)), 1.0) if total_mix > 0 else 1.0
            ds_weights = getattr(ds, "sample_weights", None)
            if ds_weights is not None and len(ds_weights) == len(ds):
                weights.extend([base * float(w) for w in ds_weights])
            else:
                weights.extend([base] * len(ds))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        drop_last=True,
        collate_fn=utils.collate_fn,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _nested_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _nested_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_nested_to_device(v, device) for v in value]
    return value


def _weighted_base_loss(loss_dict: Dict[str, torch.Tensor], weight_dict: Dict[str, float]) -> float:
    total = 0.0
    score = 0.0
    for key, value in loss_dict.items():
        if key not in weight_dict:
            continue
        weighted = float(value.detach().float().item()) * float(weight_dict[key])
        total += weighted
        if key.startswith("loss_score_calib"):
            score += weighted
    return max(total - score, 0.0)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def _aggregate_topk(scores: torch.Tensor, topk: int, mode: str, lse_tau: float) -> torch.Tensor:
    k = min(max(1, int(topk)), int(scores.numel()))
    vals = torch.topk(scores.reshape(-1), k=k, largest=True).values
    mode = str(mode).lower().strip()
    if mode in {"logsumexp", "lse"}:
        tau = max(float(lse_tau), 1e-6)
        return tau * torch.logsumexp(vals / tau, dim=0)
    if mode == "max":
        return vals.max()
    return vals.mean()


def _score_kwargs_from_cfg(cfg) -> Dict[str, Any]:
    return {
        "beta": float(getattr(cfg, "stage_b_infer_patch_beta", getattr(cfg, "stage_b_infer_text_beta", 1.0))),
        "canonical_weight": float(getattr(cfg, "stage_b_infer_canonical_weight", 1.0)),
        "text_agg": str(getattr(cfg, "stage_b_infer_text_agg", "mean")),
        "softmin_tau": float(getattr(cfg, "stage_b_infer_softmin_tau", 0.7)),
        "mean_softmin_alpha": float(getattr(cfg, "stage_b_infer_mean_softmin_alpha", 0.5)),
        "detach_patch": bool(getattr(cfg, "stage_b_score_calib_detach_patch", False)),
        "score_mode": str(getattr(cfg, "stage_b_score_mode", "patch_text")),
    }


def _collect_batch_records(cfg, criterion, outputs, targets, match_ctx_neg, rank_outputs, rank_targets, rank_pair_map):
    device = outputs["pred_boxes"].device
    score_neg = compute_stage_b_slot_logits(outputs, **_score_kwargs_from_cfg(cfg))
    topk = int(getattr(cfg, "stage_b_score_calib_topk", 10))
    neg_agg = str(getattr(cfg, "stage_b_score_calib_neg_agg", "logsumexp"))
    lse_tau = float(getattr(cfg, "stage_b_score_calib_neg_lse_tau", 0.2))

    pair_records: List[Tuple[float, float, float, List[float]]] = []
    if rank_outputs is not None and rank_targets and rank_pair_map is not None:
        match_ctx_pos = criterion.compute_matching(rank_outputs, rank_targets)
        score_pos = compute_stage_b_slot_logits(rank_outputs, **_score_kwargs_from_cfg(cfg))
        neg_indices = match_ctx_neg["all_indices"]
        neg_slots = match_ctx_neg["matched_patch_idx_list"]
        pos_indices = match_ctx_pos["all_indices"]
        pos_slots = match_ctx_pos["matched_patch_idx_list"]
        for rank_row, batch_idx_t in enumerate(rank_pair_map.to(device=device, dtype=torch.long).view(-1).tolist()):
            batch_idx = int(batch_idx_t)
            if batch_idx < 0 or batch_idx >= len(targets):
                continue
            rank_source_slot = rank_targets[rank_row].get("rank_source_slot", None)
            source_slot = int(rank_source_slot.view(-1)[0].item()) if torch.is_tensor(rank_source_slot) else 0
            src_neg, tgt_neg = neg_indices[batch_idx]
            slot_neg = neg_slots[batch_idx]
            src_pos, tgt_pos = pos_indices[rank_row]
            slot_pos = pos_slots[rank_row]
            if src_neg.numel() == 0 or src_pos.numel() == 0:
                continue
            neg_by_target = {}
            for query_idx, target_idx, slot_idx in zip(src_neg.tolist(), tgt_neg.tolist(), slot_neg.tolist()):
                if int(slot_idx) == source_slot:
                    neg_by_target[int(target_idx)] = (int(query_idx), int(slot_idx))
            rank_target_ids = rank_targets[rank_row].get("rank_target_ids", None)
            if torch.is_tensor(rank_target_ids):
                rank_target_ids = rank_target_ids.to(device=device, dtype=torch.long).view(-1)
            pos_by_target = {}
            for query_idx, target_idx, slot_idx in zip(src_pos.tolist(), tgt_pos.tolist(), slot_pos.tolist()):
                local = int(target_idx)
                original = int(rank_target_ids[local].item()) if torch.is_tensor(rank_target_ids) and local < int(rank_target_ids.numel()) else local
                pos_by_target[original] = (int(query_idx), int(slot_idx))
            for target_idx in sorted(set(neg_by_target) & set(pos_by_target)):
                q_neg, k_neg = neg_by_target[target_idx]
                q_pos, k_pos = pos_by_target[target_idx]
                if k_neg >= score_neg.shape[2] or k_pos >= score_pos.shape[2]:
                    continue
                s_pos = float(score_pos[rank_row, q_pos, k_pos].detach().float().item())
                s_neg = float(score_neg[batch_idx, q_neg, k_neg].detach().float().item())
                neg_scores = score_neg[batch_idx].reshape(-1)
                agg = float(_aggregate_topk(neg_scores, topk, neg_agg, lse_tau).detach().float().item())
                pos_other_scores: List[float] = []
                pos_slot_scores = score_pos[rank_row, :, k_pos]
                if int(pos_slot_scores.numel()) > 1:
                    masked_pos_scores = pos_slot_scores.clone()
                    masked_pos_scores[q_pos] = torch.finfo(masked_pos_scores.dtype).min
                    pos_k = min(topk, int(masked_pos_scores.numel()) - 1)
                    if pos_k > 0:
                        pos_other = torch.topk(masked_pos_scores, k=pos_k, largest=True).values
                        pos_other_scores = [float(v) for v in pos_other.detach().float().view(-1).tolist()]
                pair_records.append((s_pos, s_neg, agg, pos_other_scores))

    alltn_aggs: List[float] = []
    alltn_maxes: List[float] = []
    for batch_idx, target in enumerate(targets):
        is_tn = target.get("is_tn", None)
        if not torch.is_tensor(is_tn):
            continue
        tn_slots = torch.nonzero(is_tn.to(device=device, dtype=torch.bool).view(-1), as_tuple=False).flatten()
        tn_slots = tn_slots[tn_slots < score_neg.shape[2]]
        if tn_slots.numel() == 0:
            continue
        vals = score_neg[batch_idx].index_select(1, tn_slots).reshape(-1)
        k = min(topk, int(vals.numel()))
        if k <= 0:
            continue
        top_vals = torch.topk(vals, k=k, largest=True).values
        alltn_aggs.append(float(_aggregate_topk(vals, topk, neg_agg, lse_tau).detach().float().item()))
        alltn_maxes.append(float(top_vals.max().detach().float().item()))
    return pair_records, alltn_aggs, alltn_maxes


def _with_parent_score_masks(layer_outputs: Dict[str, Any], parent_outputs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(layer_outputs)
    for key in ("phrase_to_token_mask", "canonical_to_token_mask", "patch_mask", "patch_phrase_mask"):
        if key in parent_outputs and key not in out:
            out[key] = parent_outputs[key]
    return out


def _iter_score_layers(criterion, outputs, rank_outputs):
    yield "main", outputs, rank_outputs
    if not bool(getattr(criterion, "stage_b_score_calib_aux_loss", False)):
        return
    aux_outputs = outputs.get("aux_outputs", []) or []
    rank_aux_outputs = []
    if isinstance(rank_outputs, dict):
        rank_aux_outputs = rank_outputs.get("aux_outputs", []) or []
    start_idx = int(getattr(criterion, "stage_b_aux_loss_start_idx", 0))
    for aux_idx, aux_outputs_i in enumerate(aux_outputs):
        if aux_idx < start_idx:
            continue
        aux_layer_outputs = _with_parent_score_masks(aux_outputs_i, outputs)
        aux_rank_outputs = None
        if isinstance(rank_outputs, dict):
            if aux_idx < len(rank_aux_outputs):
                aux_rank_outputs = _with_parent_score_masks(rank_aux_outputs[aux_idx], rank_outputs)
            else:
                aux_rank_outputs = rank_outputs
        yield f"aux_{aux_idx}", aux_layer_outputs, aux_rank_outputs


def _build_rank_forward(cfg, samples, targets, captions, patches, patch_global, patch_mask, device):
    rank_subbatch = _build_stage_b_rank_subbatch(cfg, samples, targets, captions, patches, patch_global, patch_mask)
    if rank_subbatch is None or not rank_subbatch.get("indices"):
        return None, None, None
    return (
        rank_subbatch,
        _nested_to_device(rank_subbatch["targets"], device),
        torch.as_tensor(rank_subbatch["indices"], dtype=torch.long, device=device),
    )


def _empty_layer_records(name: str) -> Dict[str, Any]:
    return {"name": name, "pair_records": [], "alltn_aggs": [], "alltn_maxes": []}


def _score_component_raw(pair_records, alltn_aggs, margin: float, tau_neg: float) -> Dict[str, float]:
    pos_losses = []
    neg_losses = []
    gap_losses = []
    pos_query_losses = []
    for rec in pair_records:
        s_pos = float(rec[0])
        neg_agg = float(rec[2])
        pos_losses.append(float(F.softplus(torch.tensor(0.1 - s_pos)).item()))
        neg_losses.append(float(F.softplus(torch.tensor(neg_agg - tau_neg)).item()))
        gap_losses.append(float(F.softplus(torch.tensor(float(margin) - s_pos + neg_agg)).item()))
        pos_others = rec[3] if len(rec) > 3 else []
        if pos_others:
            pos_query_losses.append(
                float(np.mean([F.softplus(torch.tensor(float(margin) - s_pos + float(v))).item() for v in pos_others]))
            )
    return {
        "pos_raw": _mean(pos_losses),
        "neg_raw": _mean(neg_losses),
        "gap_raw": _mean(gap_losses),
        "pos_query_raw": _mean(pos_query_losses),
        "alltn_raw": _mean([float(F.softplus(torch.tensor(v - tau_neg)).item()) for v in alltn_aggs]),
    }


def _sweep(records: Dict[str, Any], margins: Sequence[float], weights: Sequence[float], tau_neg: float) -> List[Dict[str, Any]]:
    layer_records = records.get("layer_records") or [
        {
            "name": "main",
            "pair_records": records.get("pair_records", []),
            "alltn_aggs": records.get("alltn_aggs", []),
        }
    ]
    base_mean = max(float(records["base_loss_mean"]), 1e-9)
    neg_weight = float(records["current"]["neg_weight"])
    gap_weight = float(records["current"]["gap_weight"])
    pos_weight = float(records["current"]["pos_weight"])
    pos_query_weight = float(records["current"]["pos_query_weight"])
    rows: List[Dict[str, Any]] = []
    for margin in margins:
        layer_raws = [
            _score_component_raw(layer.get("pair_records", []), layer.get("alltn_aggs", []), float(margin), tau_neg)
            for layer in layer_records
        ]
        pos_raw = sum(x["pos_raw"] for x in layer_raws)
        neg_raw = sum(x["neg_raw"] for x in layer_raws)
        gap_raw = sum(x["gap_raw"] for x in layer_raws)
        pos_query_raw = sum(x["pos_query_raw"] for x in layer_raws)
        alltn_raw = sum(x["alltn_raw"] for x in layer_raws)
        for weight in weights:
            score_loss = (
                pos_weight * pos_raw
                + neg_weight * neg_raw
                + gap_weight * gap_raw
                + pos_query_weight * pos_query_raw
                + float(weight) * alltn_raw
            )
            rows.append(
                {
                    "margin": float(margin),
                    "alltn_weight": float(weight),
                    "score_loss_est": float(score_loss),
                    "score_to_base_ratio": float(score_loss / base_mean),
                    "alltn_weighted": float(float(weight) * alltn_raw),
                    "alltn_to_base_ratio": float(float(weight) * alltn_raw / base_mean),
                    "gap_weighted": float(gap_weight * gap_raw),
                    "gap_to_base_ratio": float(gap_weight * gap_raw / base_mean),
                    "pos_raw": pos_raw,
                    "neg_raw": neg_raw,
                    "gap_raw": gap_raw,
                    "pos_query_raw": pos_query_raw,
                    "alltn_raw": alltn_raw,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Stage-B score calibration margin/allTN weight offline.")
    parser.add_argument("--config", default="config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_alltn_tau05605_m010_w005_tnneg_tokencount.py")
    parser.add_argument("--datasets", default="config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", default="outputs/stageb_score_calib_probe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=40)
    parser.add_argument("--max_rank_pairs_per_batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tau_neg", type=float, default=0.5605)
    parser.add_argument("--margins", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    parser.add_argument("--alltn_weights", nargs="+", type=float, default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20])
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(args.device)
    _set_seed(int(args.seed))
    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.batch_size = int(args.batch_size)
    cfg.patch_only = True
    cfg.patch_only_compute_text_logits = True
    cfg.build_text_token_masks = True
    cfg.use_coco_eval = False
    cfg.stage_b_score_calib_tau_neg = float(args.tau_neg)
    cfg.stage_b_max_rank_pairs_per_batch = int(args.max_rank_pairs_per_batch)

    model, criterion = _load_model_and_criterion(cfg, args.ckpt, device)
    loader = _make_train_loader(cfg, args.datasets, int(args.batch_size), int(args.num_workers), device, int(args.seed))

    base_losses: List[float] = []
    layer_records_by_name: Dict[str, Dict[str, Any]] = {}
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(args.max_batches):
                break
            raw_samples, raw_targets = batch
            raw_targets = list(raw_targets)
            samples, targets, captions, patches, patch_global, patch_mask = _prepare_patch_batch(
                raw_samples, raw_targets, device
            )
            for target, raw_target in zip(targets, raw_targets):
                if "rank_positive_captions" in raw_target:
                    target["rank_positive_captions"] = list(raw_target["rank_positive_captions"])
            stage_b_mask_kwargs = _build_stage_b_mask_kwargs(
                targets,
                (
                    "phrase_to_token_mask",
                    "canonical_to_token_mask",
                    "content_to_token_mask",
                    "attr_pos_to_token_mask",
                    "attr_neg_to_token_mask",
                    "phrase_semantic_token_mask",
                ),
                device,
                min_k=_stage_b_patch_slot_count(patches, patch_global, patch_mask),
            )
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                outputs = model(
                    samples,
                    targets=targets,
                    captions=captions,
                    patches=patches,
                    patch_global=patch_global,
                    patch_mask=patch_mask,
                    patch_only=True,
                    disable_patch_dn=True,
                    patch_only_compute_text_logits=True,
                    **stage_b_mask_kwargs,
                )
            match_ctx_neg = criterion.compute_matching(outputs, targets)

            rank_subbatch, rank_targets, rank_pair_map = _build_rank_forward(
                cfg, samples, targets, captions, patches, patch_global, patch_mask, device
            )
            rank_outputs = None
            if rank_subbatch is not None and rank_targets is not None:
                rank_mask_kwargs = _build_stage_b_mask_kwargs(
                    rank_targets,
                    ("phrase_to_token_mask", "canonical_to_token_mask"),
                    device,
                    min_k=_stage_b_patch_slot_count(
                        rank_subbatch["patches"], rank_subbatch["patch_global"], rank_subbatch["patch_mask"]
                    ),
                )
                with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                    rank_outputs = model(
                        rank_subbatch["samples"].to(device),
                        targets=rank_targets,
                        captions=rank_subbatch["captions"],
                        patches=_nested_to_device(rank_subbatch["patches"], device),
                        patch_global=_nested_to_device(rank_subbatch["patch_global"], device),
                        patch_mask=_nested_to_device(rank_subbatch["patch_mask"], device),
                        patch_only=True,
                        disable_patch_dn=True,
                        patch_only_compute_text_logits=True,
                        **rank_mask_kwargs,
                    )
            outputs_for_loss = dict(outputs)
            if rank_outputs is not None and rank_targets is not None and rank_pair_map is not None:
                outputs_for_loss["rank_pos_outputs"] = rank_outputs
                outputs_for_loss["rank_pos_targets"] = rank_targets
                outputs_for_loss["rank_pair_map"] = rank_pair_map
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                loss_dict = criterion(outputs_for_loss, targets)
            base_losses.append(_weighted_base_loss(loss_dict, criterion.weight_dict))
            for layer_name, layer_outputs, layer_rank_outputs in _iter_score_layers(criterion, outputs, rank_outputs):
                layer_match_ctx = criterion.compute_matching(layer_outputs, targets)
                pairs, aggs, maxes = _collect_batch_records(
                    cfg,
                    criterion,
                    layer_outputs,
                    targets,
                    layer_match_ctx,
                    layer_rank_outputs,
                    rank_targets,
                    rank_pair_map,
                )
                layer_records = layer_records_by_name.setdefault(layer_name, _empty_layer_records(layer_name))
                layer_records["pair_records"].extend(pairs)
                layer_records["alltn_aggs"].extend(aggs)
                layer_records["alltn_maxes"].extend(maxes)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                main_records = layer_records_by_name.get("main", _empty_layer_records("main"))
                total_pairs = sum(len(x["pair_records"]) for x in layer_records_by_name.values())
                total_alltn = sum(len(x["alltn_aggs"]) for x in layer_records_by_name.values())
                print(
                    f"[INFO] batch {batch_idx + 1}/{args.max_batches} "
                    f"base={_mean(base_losses):.4f} "
                    f"main_pairs={len(main_records['pair_records'])} main_alltn={len(main_records['alltn_aggs'])} "
                    f"layer_pairs={total_pairs} layer_alltn={total_alltn} "
                    f"elapsed={(time.time() - start) / 60:.1f}m",
                    flush=True,
                )

    if "main" not in layer_records_by_name:
        layer_records_by_name["main"] = _empty_layer_records("main")
    pair_records = layer_records_by_name["main"]["pair_records"]
    alltn_aggs = layer_records_by_name["main"]["alltn_aggs"]
    alltn_maxes = layer_records_by_name["main"]["alltn_maxes"]
    layer_records_list = list(layer_records_by_name.values())
    layer_stats = {
        layer["name"]: {
            "num_pairs": len(layer["pair_records"]),
            "num_alltn": len(layer["alltn_aggs"]),
            "pos_score": _percentiles([x[0] for x in layer["pair_records"]]),
            "neg_matched_score": _percentiles([x[1] for x in layer["pair_records"]]),
            "neg_agg_score": _percentiles([x[2] for x in layer["pair_records"]]),
            "alltn_agg_score": _percentiles(layer["alltn_aggs"]),
            "alltn_topk_max": _percentiles(layer["alltn_maxes"]),
        }
        for layer in layer_records_list
    }
    records = {
        "config": args.config,
        "ckpt": args.ckpt,
        "tau_neg": float(args.tau_neg),
        "num_batches": min(int(args.max_batches), len(loader)),
        "num_pairs": len(pair_records),
        "num_alltn": len(alltn_aggs),
        "num_layer_pairs": sum(len(x["pair_records"]) for x in layer_records_list),
        "num_layer_alltn": sum(len(x["alltn_aggs"]) for x in layer_records_list),
        "base_loss_mean": _mean(base_losses),
        "pair_records": pair_records,
        "alltn_aggs": alltn_aggs,
        "alltn_maxes": alltn_maxes,
        "layer_records": layer_records_list,
        "layer_stats": layer_stats,
        "stats": {
            "base_loss": _percentiles(base_losses),
            "pos_score": _percentiles([x[0] for x in pair_records]),
            "neg_matched_score": _percentiles([x[1] for x in pair_records]),
            "neg_agg_score": _percentiles([x[2] for x in pair_records]),
            "alltn_agg_score": _percentiles(alltn_aggs),
            "alltn_topk_max": _percentiles(alltn_maxes),
        },
        "current": {
            "pos_weight": float(getattr(cfg, "stage_b_score_calib_pos_weight", 0.05)),
            "neg_weight": float(getattr(cfg, "stage_b_score_calib_neg_weight", 0.125)),
            "gap_weight": float(getattr(cfg, "stage_b_score_calib_gap_weight", 0.125)),
            "pos_query_weight": float(getattr(cfg, "stage_b_score_calib_pos_query_weight", 0.05)),
            "alltn_weight": float(getattr(cfg, "stage_b_score_calib_all_tn_neg_weight", 0.0)),
        },
    }
    rows = _sweep(records, args.margins, args.alltn_weights, float(args.tau_neg))
    records["sweep"] = rows
    candidates = [
        r
        for r in rows
        if 0.05 <= r["score_to_base_ratio"] <= 0.12 and 0.01 <= r["alltn_to_base_ratio"] <= 0.05
    ]
    candidates.sort(key=lambda r: (abs(r["score_to_base_ratio"] - 0.08), abs(r["alltn_to_base_ratio"] - 0.03)))
    records["recommended"] = candidates[:10]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "score_calib_probe.json"
    out_json.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] wrote {out_json}")
    print(json.dumps({"stats": records["stats"], "recommended": records["recommended"][:5]}, indent=2))


if __name__ == "__main__":
    main()
