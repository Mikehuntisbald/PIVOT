#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from models.GroundingDINO.patch_hungarian_criterion import _sigmoid_focal_loss_no_reduce  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from tools.eval_stagea_patch_checkpoints import _prepare_patch_batch, _torch_load_compat  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _load_model_and_criterion(cfg, ckpt_path: str, device: torch.device):
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={cfg.modelname}")
    model, criterion, _postprocessors = build_func(cfg)
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    state = clean_state_dict(_extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] {ckpt_path}: missing keys={len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] {ckpt_path}: unexpected keys={len(unexpected)}", file=sys.stderr)
    model.to(device).eval()
    criterion.to(device).eval()
    return model, criterion


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_train_loader(cfg, dataset_meta: Dict[str, Any], batch_size: int, num_workers: int, device, seed: int):
    if not dataset_meta.get("train"):
        raise ValueError("datasets json has no train split")
    datasets = []
    for datasetinfo in dataset_meta["train"]:
        datasets.append(build_dataset(image_set="train", args=cfg, datasetinfo=datasetinfo))
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        from torch.utils.data import ConcatDataset

        dataset = ConcatDataset(datasets)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = RandomSampler(dataset, generator=generator)
    batch_sampler = BatchSampler(sampler, batch_size=int(batch_size), drop_last=True)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=utils.collate_fn,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
        prefetch_factor=1 if int(num_workers) > 0 else None,
        persistent_workers=bool(int(num_workers) > 0),
    )


def _candidate_key(kind: str, value: float) -> str:
    if kind == "all":
        return "all"
    if kind == "ratio":
        return f"ratio_{value:g}"
    return f"topk_{int(value)}"


def _parse_ratio_candidates(values: List[float]) -> List[float]:
    out = []
    for v in values:
        fv = float(v)
        if fv <= 0:
            continue
        out.append(fv)
    return out


@torch.no_grad()
def calibrate(args) -> Dict[str, Any]:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    _set_seed(int(args.seed))

    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.batch_size = int(args.batch_size)
    cfg.patch_only = True
    cfg.use_coco_eval = False
    # Loss calibration should not be affected by a candidate setting in config.
    cfg.patch_ce_reduction = "legacy"
    cfg.patch_ce_neg_topk = 0
    cfg.patch_ce_neg_topk_ratio = 0.0

    with open(args.datasets, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    model, criterion = _load_model_and_criterion(cfg, args.ckpt, device)
    model.train()
    criterion.eval()
    loader = _build_train_loader(
        cfg,
        dataset_meta,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        seed=int(args.seed),
    )

    fixed_topks = sorted({int(x) for x in args.topk if int(x) > 0})
    ratios = _parse_ratio_candidates(args.ratio)
    candidate_specs = [("all", 0.0)]
    candidate_specs.extend(("topk", float(k)) for k in fixed_topks)
    candidate_specs.extend(("ratio", float(r)) for r in ratios)

    sums = {
        "legacy_dense": 0.0,
        "pos": 0.0,
        "neg_all": 0.0,
        "neg_count": 0.0,
        "pos_count": 0.0,
        "valid_images": 0.0,
        "batches": 0.0,
    }
    cand = {
        _candidate_key(kind, value): {
            "kind": kind,
            "value": value,
            "ce_sum": 0.0,
            "neg_loss_sum": 0.0,
            "topk_count_sum": 0.0,
        }
        for kind, value in candidate_specs
    }

    start = time.time()
    processed = 0
    for batch_i, batch in enumerate(loader):
        if int(args.max_batches) > 0 and batch_i >= int(args.max_batches):
            break
        samples, targets, captions, patches, patch_global, patch_mask = _prepare_patch_batch(*batch, device)
        with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
            outputs = model(
                samples,
                targets=targets,
                captions=captions,
                patches=patches,
                patch_global=patch_global,
                patch_mask=patch_mask,
                patch_only=True,
                patch_only_compute_text_logits=False,
            )
        if "pred_logits_patch" not in outputs or outputs["pred_logits_patch"] is None:
            raise KeyError("model output missing pred_logits_patch")
        matching_outputs = dict(outputs)
        matching_outputs["pred_logits_patch"] = outputs["pred_logits_patch"].float()
        matching_outputs["pred_boxes"] = outputs["pred_boxes"].float()
        match_ctx = criterion.compute_matching(matching_outputs, targets)

        pred_logits_patch = match_ctx["pred_logits_patch"]
        patch_mask_out = match_ctx["patch_mask"]
        all_indices = match_ctx["all_indices"]
        matched_local_patch_idx_list = match_ctx["matched_local_patch_idx_list"]
        num_boxes = float(match_ctx["num_boxes"])
        bsz = int(match_ctx["B"])
        num_slots = int(match_ctx["K"])
        batch_legacy_sum = 0.0
        batch_valid_images = 0
        batch_cand = {
            key: {
                "ce_sum": 0.0,
                "neg_loss_sum": 0.0,
                "topk_count_sum": 0.0,
            }
            for key in cand
        }

        for b in range(bsz):
            support_classes = criterion._get_support_classes(targets[b], K=num_slots, device=device)
            valid_k = support_classes >= 0
            if patch_mask_out is not None:
                valid_k = valid_k & patch_mask_out[b]
            keep = valid_k.nonzero(as_tuple=False).flatten()
            if keep.numel() == 0:
                continue

            logits_b = pred_logits_patch[b][:, keep]
            target_b = torch.zeros_like(logits_b)
            src_idx, _tgt_idx = all_indices[b]
            if src_idx.numel() > 0:
                target_b[src_idx, matched_local_patch_idx_list[b]] = 1.0

            loss_mat = _sigmoid_focal_loss_no_reduce(
                logits_b,
                target_b,
                alpha=criterion.focal_alpha,
                gamma=criterion.focal_gamma,
            )
            batch_legacy_sum += float(loss_mat.sum().detach().item())
            pos = target_b > 0.5
            neg = ~pos
            zero = loss_mat.sum() * 0.0
            pos_values = loss_mat[pos]
            neg_values = loss_mat[neg]
            pos_loss = pos_values.mean() if pos_values.numel() > 0 else zero
            neg_all_loss = neg_values.mean() if neg_values.numel() > 0 else zero

            sums["pos"] += float(pos_loss.detach().item())
            sums["neg_all"] += float(neg_all_loss.detach().item())
            sums["pos_count"] += float(pos_values.numel())
            sums["neg_count"] += float(neg_values.numel())
            sums["valid_images"] += 1.0
            batch_valid_images += 1

            neg_count = int(neg_values.numel())
            for kind, value in candidate_specs:
                key = _candidate_key(kind, value)
                if neg_count <= 0:
                    neg_loss = zero
                    topk_count = 0
                elif kind == "all":
                    neg_loss = neg_all_loss
                    topk_count = neg_count
                elif kind == "ratio":
                    topk_count = min(neg_count, max(1, int(neg_count * float(value) + 0.999999)))
                    neg_loss = neg_values.topk(k=topk_count, largest=True).values.mean()
                else:
                    topk_count = min(neg_count, max(1, int(value)))
                    neg_loss = neg_values.topk(k=topk_count, largest=True).values.mean()
                ce = pos_loss + float(args.patch_lambda_neg) * neg_loss
                batch_cand[key]["ce_sum"] += float(ce.detach().item())
                batch_cand[key]["neg_loss_sum"] += float(neg_loss.detach().item())
                batch_cand[key]["topk_count_sum"] += float(topk_count)

        if batch_valid_images > 0:
            sums["legacy_dense"] += batch_legacy_sum / num_boxes
            sums["batches"] += 1.0
            for key in cand:
                cand[key]["ce_sum"] += batch_cand[key]["ce_sum"] / float(batch_valid_images)
                cand[key]["neg_loss_sum"] += batch_cand[key]["neg_loss_sum"] / float(batch_valid_images)
                cand[key]["topk_count_sum"] += batch_cand[key]["topk_count_sum"] / float(batch_valid_images)

        processed += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if int(args.log_every) > 0 and (processed == 1 or processed % int(args.log_every) == 0):
            elapsed = time.time() - start
            print(f"[INFO] batch {processed}, valid_images={int(sums['valid_images'])}, elapsed={elapsed/60:.1f}m")

    valid = max(1.0, sums["valid_images"])
    batch_den = max(1.0, sums["batches"])
    legacy_mean = sums["legacy_dense"] / batch_den
    result = {
        "checkpoint": str(args.ckpt),
        "config": str(args.config),
        "datasets": str(args.datasets),
        "batch_size": int(args.batch_size),
        "max_batches": int(args.max_batches),
        "processed_batches": int(sums["batches"]),
        "valid_images": int(sums["valid_images"]),
        "patch_lambda_neg": float(args.patch_lambda_neg),
        "legacy_dense_ce": legacy_mean,
        "pos_loss": sums["pos"] / valid,
        "neg_all_loss": sums["neg_all"] / valid,
        "avg_pos_count": sums["pos_count"] / valid,
        "avg_neg_count": sums["neg_count"] / valid,
        "candidates": [],
    }
    for key, row in cand.items():
        ce = row["ce_sum"] / batch_den
        neg_loss = row["neg_loss_sum"] / batch_den
        topk_count = row["topk_count_sum"] / batch_den
        result["candidates"].append(
            {
                "name": key,
                "kind": row["kind"],
                "value": row["value"],
                "ce": ce,
                "neg_loss": neg_loss,
                "avg_topk_count": topk_count,
                "topk_frac_of_neg": topk_count / max(1e-12, result["avg_neg_count"]),
                "ce_over_legacy_dense": ce / max(1e-12, legacy_mean),
            }
        )
    result["candidates"].sort(key=lambda x: (x["kind"] != "all", x["kind"], float(x["value"])))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Stage-A Hungarian CE negative top-k on real train batches.")
    parser.add_argument("--config", default="config/cfg_patch_stage_a_v2_rank.py")
    parser.add_argument("--datasets", default="config/datasets_patch_stage_a_lvis_coco2017_local.json")
    parser.add_argument("--ckpt", default="outputs/stageA_coco_multipatch/checkpoint0004.pth")
    parser.add_argument("--output", default="outputs/stageA_coco_multipatch_v2_rank_posneg_topk_from0004/neg_topk_calibration.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=18)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--ratio", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--patch_lambda_neg", type=float, default=0.25)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    result = calibrate(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
