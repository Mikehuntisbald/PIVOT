#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def load_model_and_criterion(cfg, ckpt_path: str, device: torch.device):
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={cfg.modelname}")
    model, criterion, _postprocessors = build_func(cfg)
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    state = clean_state_dict(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] {ckpt_path}: missing keys={len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] {ckpt_path}: unexpected keys={len(unexpected)}", file=sys.stderr)
    model.to(device).eval()
    criterion.to(device).eval()
    return model, criterion


def _prepare_patch_batch(samples, targets, device: torch.device):
    samples = samples.to(device)
    targets = list(targets)
    captions = [t.get("caption", "object .") for t in targets]
    patch_mask = None
    patch_global = None
    patches = None
    kmax = None

    if all(("patch_global" in t) for t in targets):
        pg0 = targets[0]["patch_global"]
        if pg0.dim() == 1:
            patch_global = torch.stack([t["patch_global"] for t in targets], dim=0).to(device, non_blocking=True)
        else:
            kmax = max(int(t["patch_global"].shape[0]) for t in targets)
            dim = int(pg0.shape[1])
            patch_global = torch.zeros((len(targets), kmax, dim), dtype=pg0.dtype, device=device)
            patch_mask = torch.zeros((len(targets), kmax), dtype=torch.bool, device=device)
            for i, t in enumerate(targets):
                pg = t["patch_global"]
                k = int(pg.shape[0])
                patch_global[i, :k] = pg.to(device, non_blocking=True)
                patch_mask[i, :k] = True
    elif all(("patches" in t) for t in targets):
        p0 = targets[0]["patches"]
        kmax = max(int(t["patches"].shape[0]) for t in targets)
        c, h, w = map(int, p0.shape[1:])
        patches = torch.zeros((len(targets), kmax, c, h, w), dtype=p0.dtype, device=device)
        patch_mask = torch.zeros((len(targets), kmax), dtype=torch.bool, device=device)
        for i, t in enumerate(targets):
            p = t["patches"]
            k = int(p.shape[0])
            patches[i, :k] = p.to(device, non_blocking=True)
            patch_mask[i, :k] = True
    else:
        patches = torch.stack([t["patch"] for t in targets], dim=0).to(device, non_blocking=True)

    filtered_targets = []
    for t in targets:
        out = {
            k: v.to(device)
            for k, v in t.items()
            if torch.is_tensor(v) and k not in {"patch", "patches", "patch_global"}
        }
        if kmax is not None and "support_classes" in out:
            support_classes = out["support_classes"].view(-1)
            if support_classes.numel() < kmax:
                pad = torch.full(
                    (kmax - int(support_classes.numel()),),
                    -1,
                    dtype=support_classes.dtype,
                    device=device,
                )
                out["support_classes"] = torch.cat([support_classes, pad], dim=0)
        filtered_targets.append(out)

    return samples, filtered_targets, captions, patches, patch_global, patch_mask


@torch.no_grad()
def _forward_patch(model, batch, device: torch.device):
    samples, targets, captions, patches, patch_global, patch_mask = _prepare_patch_batch(*batch, device)
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
    return outputs, targets


def _union_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.detach().float()
    return logits if logits.dim() == 2 else logits.max(dim=-1).values


def _compare_logits(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = a.detach().float()
    b = b.detach().float()
    if a.shape != b.shape:
        a = _union_logits(a)
        b = _union_logits(b)
    return a, b


def _matched_query_recall50(outputs, targets, criterion) -> Tuple[float, float]:
    if not hasattr(criterion, "compute_matching"):
        return 0.0, 0.0
    match_ctx = criterion.compute_matching(outputs, targets)
    union_logits = _union_logits(outputs["pred_logits_patch"])
    recalled = 0.0
    matched = 0.0
    for b, (src_idx, _tgt_idx) in enumerate(match_ctx["all_indices"]):
        if src_idx.numel() == 0:
            continue
        topk = torch.topk(union_logits[b], k=min(50, int(union_logits.shape[1])), largest=True).indices
        recalled += float(torch.isin(src_idx.detach().cpu(), topk.detach().cpu()).float().sum().item())
        matched += float(src_idx.numel())
    return recalled, matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--stage_a_ckpt", required=True)
    parser.add_argument("--stage_b_ckpt", required=True)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    with open(args.datasets, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)
    datasetinfo = dataset_meta["val"][0]
    dataset = build_dataset(image_set="val", args=cfg, datasetinfo=datasetinfo)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=int(args.num_workers),
        pin_memory=str(device).startswith("cuda"),
    )

    model_a, criterion = load_model_and_criterion(cfg, args.stage_a_ckpt, device)
    model_b, _criterion_b = load_model_and_criterion(cfg, args.stage_b_ckpt, device)

    abs_logit_sum = 0.0
    abs_logit_count = 0
    cosine_values: List[float] = []
    overlap_values: List[float] = []
    abs_box_sum = 0.0
    abs_box_count = 0
    recall_a = matched_a = 0.0
    recall_b = matched_b = 0.0

    for i, batch in enumerate(loader):
        if i >= int(args.num_batches):
            break
        outputs_a, targets = _forward_patch(model_a, batch, device)
        outputs_b, _targets_b = _forward_patch(model_b, batch, device)

        logits_a, logits_b = _compare_logits(outputs_a["pred_logits_patch"], outputs_b["pred_logits_patch"])
        abs_logit_sum += float((logits_a - logits_b).abs().sum().item())
        abs_logit_count += int(logits_a.numel())
        flat_a = logits_a.reshape(logits_a.shape[0], -1)
        flat_b = logits_b.reshape(logits_b.shape[0], -1)
        cosine_values.extend(F.cosine_similarity(flat_a, flat_b, dim=1).detach().cpu().tolist())

        union_a = _union_logits(outputs_a["pred_logits_patch"])
        union_b = _union_logits(outputs_b["pred_logits_patch"])
        for b in range(union_a.shape[0]):
            k = min(50, int(union_a.shape[1]), int(union_b.shape[1]))
            top_a = torch.topk(union_a[b], k=k, largest=True).indices
            top_b = torch.topk(union_b[b], k=k, largest=True).indices
            overlap_values.append(float(torch.isin(top_a.cpu(), top_b.cpu()).float().mean().item()))

        boxes_a = outputs_a["pred_boxes"].detach().float()
        boxes_b = outputs_b["pred_boxes"].detach().float()
        abs_box_sum += float((boxes_a - boxes_b).abs().sum().item())
        abs_box_count += int(boxes_a.numel())

        r_a, m_a = _matched_query_recall50(outputs_a, targets, criterion)
        r_b, m_b = _matched_query_recall50(outputs_b, targets, criterion)
        recall_a += r_a
        matched_a += m_a
        recall_b += r_b
        matched_b += m_b

    result = {
        "mean_abs_patch_logit_diff": abs_logit_sum / max(1, abs_logit_count),
        "mean_cosine_patch_logit": sum(cosine_values) / max(1, len(cosine_values)),
        "top50_query_overlap": sum(overlap_values) / max(1, len(overlap_values)),
        "matched_query_recall50_stage_a": recall_a / matched_a if matched_a > 0 else 0.0,
        "matched_query_recall50_stage_b": recall_b / matched_b if matched_b > 0 else 0.0,
        "mean_abs_box_diff": abs_box_sum / max(1, abs_box_count),
        "num_batches": min(int(args.num_batches), i + 1 if "i" in locals() else 0),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
