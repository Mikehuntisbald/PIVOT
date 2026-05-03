"""
Visualize patch-only predictions vs GT boxes using the same episode dataset logic as training.

Example (COCO-only):
  /home/haoyi/miniconda/envs/cvpr/bin/python tools/visualize_patch_episode_predictions.py \\
    -c config/cfg_patch_stage_a_emb.py \\
    -p outputs/stageA_emb/checkpoint.pth \\
    --datasets config/datasets_patch_stage_a_coco2017_local.json \\
    --split train --num_images 20 --out_dir outputs/vis_coco_stageA

Example (LVIS+COCO mixed; choose COCO entry):
  /home/haoyi/miniconda/envs/cvpr/bin/python tools/visualize_patch_episode_predictions.py \\
    -c config/cfg_patch_stage_a_emb.py \\
    -p outputs/stageA_emb/checkpoint.pth \\
    --datasets config/datasets_patch_stage_a_lvis_coco2017_local.json \\
    --train_index 1 --split train --num_images 20 --out_dir outputs/vis_mixed_stageA
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from groundingdino.util import box_ops
from groundingdino.util.utils import clean_state_dict
from models.registry import MODULE_BUILD_FUNCS
from util.misc import nested_tensor_from_tensor_list
from util.slconfig import SLConfig


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        try:
            from torch import serialization as _serialization  # type: ignore

            _serialization.add_safe_globals([argparse.Namespace])  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return torch.load(path, map_location=map_location)
        except Exception:
            return torch.load(path, map_location=map_location, weights_only=False)


def _canonical_caption(caption: str) -> str:
    caption = (caption or "").lower().strip()
    if not caption:
        caption = "object ."
    if not caption.endswith("."):
        caption = caption + "."
    return caption


def _id_to_name_map(canonical_classes_json: str | None) -> Dict[int, str]:
    if not canonical_classes_json:
        return {}
    try:
        data = json.load(open(canonical_classes_json, "r", encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    out: Dict[int, str] = {}
    for it in data:
        if not isinstance(it, dict):
            continue
        cid = it.get("id", None)
        name = it.get("raw_name", None) or it.get("name", None)
        if cid is None or name is None:
            continue
        try:
            out[int(cid)] = str(name)
        except Exception:
            continue
    return out


def _tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    """
    img_t: (3,H,W) float, ImageNet normalized.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_t.dtype, device=img_t.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_t.dtype, device=img_t.device)[:, None, None]
    x = (img_t * std + mean).clamp(0, 1)
    x = (x * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def _draw_box(draw: ImageDraw.ImageDraw, box_xyxy: Tuple[float, float, float, float], color: Tuple[int, int, int], w: int = 3):
    x0, y0, x1, y1 = box_xyxy
    draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=color, width=w)


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """
    boxes: (N,4) xyxy in pixels
    scores: (N,)
    Returns indices kept.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64, device=boxes.device)
    try:
        import torchvision.ops as ops

        return ops.nms(boxes, scores, float(iou_thr))
    except Exception:
        # Tiny fallback NMS (CPU), OK for visualization.
        b = boxes.detach().cpu()
        s = scores.detach().cpu()
        order = torch.argsort(s, descending=True)
        keep: List[int] = []
        while order.numel() > 0:
            i = int(order[0].item())
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            iou, _ = box_ops.box_iou(b[i].unsqueeze(0), b[rest])
            rest = rest[iou.view(-1) <= float(iou_thr)]
            order = rest
        return torch.as_tensor(keep, dtype=torch.int64)


def load_model(cfg_path: str, ckpt_path: str, device: str):
    args = SLConfig.fromfile(cfg_path)
    args.device = device
    assert hasattr(args, "modelname"), "Config must define `modelname`."
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={args.modelname}.")
    model, _criterion, postprocessors = build_func(args)
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    return model.to(device), postprocessors


@torch.no_grad()
def run_patch_only(
    model: torch.nn.Module,
    img: torch.Tensor,  # (3,H,W)
    target: Dict[str, Any],
    device: str,
    *,
    topk: int,
    score_thr: float,
    nms_thr: float,
    postprocessor=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    caption = _canonical_caption(str(target.get("caption", "object .")))

    samples = nested_tensor_from_tensor_list([img]).to(device)
    captions = [caption]

    patches = None
    patch_global = None
    if "patch_global" in target:
        patch_global = target["patch_global"].to(device)
        if patch_global.dim() == 1:
            patch_global = patch_global.unsqueeze(0)
        elif patch_global.dim() == 2:
            patch_global = patch_global.unsqueeze(0)
    elif "patches" in target:
        patches = target["patches"].to(device)
        if patches.dim() == 4:
            patches = patches.unsqueeze(0)
    elif "patch" in target:
        patches = target["patch"].to(device)
        if patches.dim() == 3:
            patches = patches.unsqueeze(0)
    else:
        raise KeyError("target must contain 'patch_global', 'patches', or 'patch'.")

    mask_kwargs = {}
    for key in ("phrase_to_token_mask", "canonical_to_token_mask"):
        value = target.get(key, None)
        if torch.is_tensor(value):
            value = value.to(device)
            if value.dim() == 2:
                value = value.unsqueeze(0)
            mask_kwargs[key] = value

    out = model(
        samples,
        captions=captions,
        patches=patches,
        patch_global=patch_global,
        patch_only=True,
        patch_only_compute_text_logits=postprocessor is not None,
        **mask_kwargs,
    )

    H, W = target["size"].tolist()
    if postprocessor is not None and {"pred_logits_text", "phrase_to_token_mask"}.issubset(out.keys()):
        target_sizes = torch.as_tensor([[H, W]], dtype=out["pred_boxes"].dtype, device=out["pred_boxes"].device)
        result = postprocessor(out, target_sizes)[0]
        boxes_xyxy = result["boxes"]
        scores = result["scores"]
    else:
        logits = out["pred_logits_patch"][0]
        if logits.dim() == 2:
            logits = logits.max(dim=-1).values
        boxes = out["pred_boxes"][0]  # (Q,4) cxcywh normalized
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * boxes.new_tensor([W, H, W, H])
        scores = logits.sigmoid()

    keep = scores >= float(score_thr)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy, scores

    order = torch.argsort(scores, descending=True)
    if int(topk) > 0:
        order = order[: int(topk)]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]

    if nms_thr >= 0:
        keep_idx = _nms_xyxy(boxes_xyxy, scores, float(nms_thr))
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]

    return boxes_xyxy, scores


def draw_episode(
    img: torch.Tensor,
    target: Dict[str, Any],
    pred_boxes_xyxy: torch.Tensor,
    pred_scores: torch.Tensor,
    id2name: Dict[int, str],
    *,
    iou_thr: float,
    label_mode: str,
    topk_pos: int,
    topk_iou_thr: float,
    out_path: Path,
):
    pil = _tensor_to_pil(img)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    support_class = int(target["support_class"].item())
    support_name = id2name.get(support_class, str(support_class))

    # GT boxes are in normalized cxcywh after transforms.
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]
    H, W = target["size"].tolist()

    gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes) * gt_boxes.new_tensor([W, H, W, H])

    # draw all GT (blue), support-class GT (green)
    for b, lab in zip(gt_xyxy, gt_labels):
        color = (0, 160, 255)
        if int(lab.item()) == support_class:
            color = (0, 220, 0)
        _draw_box(draw, tuple(b.tolist()), color, w=3)

    # Determine pos/neg queries for focal loss (same labeling logic as PatchOnlyCriterion).
    gt_support = gt_xyxy[gt_labels == support_class]
    if gt_support.numel() > 0 and pred_boxes_xyxy.numel() > 0:
        iou, _ = box_ops.box_iou(pred_boxes_xyxy.float().cpu(), gt_support.float().cpu())
        max_iou = iou.max(dim=1).values
        mode = (label_mode or "iou_thr").lower().strip()
        if mode == "iou_thr":
            is_pos = max_iou > float(iou_thr)
        elif mode == "topk_iou":
            k = int(topk_pos) if int(topk_pos) > 0 else int(max_iou.numel())
            k = min(k, int(max_iou.numel()))
            topk_idx = torch.topk(max_iou, k=k, largest=True).indices
            keep = max_iou > float(topk_iou_thr)
            is_pos = torch.zeros_like(max_iou, dtype=torch.bool)
            if bool(keep.any()):
                is_pos[topk_idx[keep[topk_idx]]] = True
        else:
            is_pos = max_iou > float(iou_thr)
    else:
        is_pos = torch.zeros((pred_boxes_xyxy.shape[0],), dtype=torch.bool)

    # draw predictions: pos queries (yellow) / neg queries (red)
    pos_color = (255, 220, 0)
    neg_color = (255, 0, 0)
    for b, s, p in zip(pred_boxes_xyxy, pred_scores, is_pos):
        color = pos_color if bool(p) else neg_color
        _draw_box(draw, tuple(b.tolist()), color, w=2)
        x0, y0, x1, y1 = b.tolist()
        txt = f"{float(s):.2f}"
        if hasattr(draw, "textbbox"):
            bb = draw.textbbox((int(x0), int(y0)), txt, font=font)
        else:
            tw, th = draw.textsize(txt, font=font)
            bb = (int(x0), int(y0), int(x0) + tw, int(y0) + th)
        draw.rectangle(bb, fill=color)
        draw.text((int(x0), int(y0)), txt, fill=(255, 255, 255), font=font)

    header = (
        f"support_class={support_class} ({support_name})  gt={len(gt_labels)}  "
        f"pred={len(pred_scores)}  posQ={int(is_pos.sum().item())} negQ={int((~is_pos).sum().item())}"
    )
    draw.rectangle([0, 0, pil.size[0], 18], fill=(0, 0, 0))
    draw.text((4, 2), header, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Model config (e.g. config/cfg_patch_stage_a_emb.py)")
    ap.add_argument("-p", "--checkpoint", required=True, help="Checkpoint path (e.g. outputs/.../checkpoint.pth)")
    ap.add_argument("--datasets", required=True, help="Dataset json (same format as main.py --datasets)")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="Which split to sample.")
    ap.add_argument("--train_index", type=int, default=0, help="When datasets.train has multiple entries, pick one.")
    ap.add_argument("--val_index", type=int, default=0, help="When datasets.val has multiple entries, pick one.")
    ap.add_argument("--num_images", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--topk", type=int, default=50, help="Top-k predictions before NMS.")
    ap.add_argument("--score_thr", type=float, default=0.3, help="Sigmoid(score) threshold for visualization.")
    ap.add_argument("--nms_thr", type=float, default=0.6, help="NMS IoU threshold; set <0 to disable.")
    ap.add_argument(
        "--iou_thr",
        type=float,
        default=None,
        help="IoU threshold used to mark focal-loss positive queries; default: patch_iou_thr from config or 0.5.",
    )
    ap.add_argument(
        "--label_mode",
        type=str,
        default=None,
        help="Pos/neg labeling mode for coloring: iou_thr | topk_iou (default: patch_labeling_mode from config).",
    )
    ap.add_argument("--topk_pos", type=int, default=None, help="For topk_iou: top-k positives (default: patch_topk).")
    ap.add_argument(
        "--topk_iou_thr",
        type=float,
        default=None,
        help="For topk_iou: minimum IoU to be eligible positive (default: patch_topk_iou_thr).",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SLConfig.fromfile(args.config)
    iou_thr = float(args.iou_thr) if args.iou_thr is not None else float(getattr(cfg, "patch_iou_thr", 0.5))
    label_mode = str(args.label_mode) if args.label_mode is not None else str(getattr(cfg, "patch_labeling_mode", "iou_thr"))
    topk_pos = int(args.topk_pos) if args.topk_pos is not None else int(getattr(cfg, "patch_topk", 50))
    topk_iou_thr = float(args.topk_iou_thr) if args.topk_iou_thr is not None else float(getattr(cfg, "patch_topk_iou_thr", 0.05))

    with open(args.datasets, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)
    split_key = "train" if args.split == "train" else "val"
    idx = int(args.train_index if split_key == "train" else args.val_index)
    datasetinfo = dataset_meta[split_key][idx]

    from datasets import build_dataset  # local import to avoid module shadowing issues

    ds = build_dataset(image_set=args.split, args=cfg, datasetinfo=datasetinfo)
    id2name = _id_to_name_map(datasetinfo.get("canonical_classes_json", None))

    model, postprocessors = load_model(args.config, args.checkpoint, device=device)
    postprocessor = postprocessors.get("bbox", None) if isinstance(postprocessors, dict) else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(int(args.num_images)):
        j = random.randrange(0, len(ds))
        img, target = ds[j]
        pred_boxes_xyxy, pred_scores = run_patch_only(
            model,
            img=img,
            target=target,
            device=device,
            topk=args.topk,
            score_thr=args.score_thr,
            nms_thr=args.nms_thr,
            postprocessor=postprocessor,
        )
        out_path = out_dir / f"{args.split}_{i:04d}_idx{j}.jpg"
        draw_episode(
            img,
            target,
            pred_boxes_xyxy,
            pred_scores,
            id2name,
            iou_thr=iou_thr,
            label_mode=label_mode,
            topk_pos=topk_pos,
            topk_iou_thr=topk_iou_thr,
            out_path=out_path,
        )
        print(f"[{i+1}/{args.num_images}] wrote {out_path}")


if __name__ == "__main__":
    main()
