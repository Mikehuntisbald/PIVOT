# /home/haoyi/miniconda/envs/cvpr/bin/python tools/inference_on_a_image.py \
#   -c config/cfg_patch_stage_a_emb.py \
#   -p outputs/stageA_emb/checkpoint.pth \
#   -i path/to/query.jpg \
#   --patch_only --patch_emb /path/to/support.npy \
#   -o outputs/patch_only_demo

# /home/haoyi/miniconda/envs/cvpr/bin/python tools/inference_on_a_image.py \
#   -c config/cfg_patch_stage_a_emb.py \
#   -p outputs/stageA_emb/checkpoint.pth \
#   -i path/to/query.jpg \
#   --patch_only --patch_image /path/to/support.jpg \
#   -o outputs/patch_only_demo

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import datasets.transforms as T
from models.registry import MODULE_BUILD_FUNCS
from groundingdino.util import box_ops
from util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


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


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_patch_image(patch_path: str, patch_size: int = 224) -> torch.Tensor:
    patch_pil = Image.open(patch_path).convert("RGB")
    # Match PatchEpisodeJsonlDataset.patch_tfm
    import torchvision.transforms as TV

    tfm = TV.Compose(
        [
            TV.Resize(256),
            TV.CenterCrop(int(patch_size)),
            TV.ToTensor(),
            TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(patch_pil)


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    assert hasattr(args, "modelname"), "Config must define `modelname` (e.g. groundingdino)."
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={args.modelname}. Available: {list(MODULE_BUILD_FUNCS.module_dict.keys())}")
    model, _criterion, _postprocessors = build_func(args)
    checkpoint = _torch_load_compat(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def _canonical_caption(caption: str) -> str:
    caption = (caption or "").lower().strip()
    if not caption:
        caption = "object ."
    if not caption.endswith("."):
        caption = caption + "."
    return caption


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64, device=boxes.device)
    try:
        import torchvision.ops as ops

        return ops.nms(boxes, scores, float(iou_thr))
    except Exception:
        b = boxes.detach().cpu()
        s = scores.detach().cpu()
        order = torch.argsort(s, descending=True)
        keep = []
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


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = _canonical_caption(caption)
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


def get_patch_only_output(
    model,
    image,
    *,
    patch_image: torch.Tensor | None = None,
    patch_global: torch.Tensor | None = None,
    caption: str = "object .",
    box_threshold: float = 0.3,
    topk: int = 100,
    nms_thr: float = 0.6,
    cpu_only: bool = False,
):
    caption = _canonical_caption(caption)
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)

    if (patch_image is None) and (patch_global is None):
        raise ValueError("patch_only requires patch_image or patch_global.")

    patches = None
    patch_global_in = None
    if patch_global is not None:
        patch_global_in = patch_global.to(device)
        if patch_global_in.dim() == 1:
            patch_global_in = patch_global_in.unsqueeze(0)
    if patch_image is not None:
        patches = patch_image.to(device)
        if patches.dim() == 3:
            patches = patches.unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            image[None],
            captions=[caption],
            patches=patches,
            patch_global=patch_global_in,
            patch_only=True,
        )

    scores = outputs["pred_logits_patch"][0].sigmoid().detach().cpu()  # (Q,)
    boxes = outputs["pred_boxes"][0].detach().cpu()  # (Q,4) cxcywh norm

    # filter by score threshold, then keep topk
    keep = scores > float(box_threshold)
    scores_f = scores[keep]
    boxes_f = boxes[keep]
    if scores_f.numel() == 0:
        return boxes_f, []

    order = torch.argsort(scores_f, descending=True)
    if int(topk) > 0:
        order = order[: min(int(topk), int(order.numel()))]
    boxes_f = boxes_f[order]
    scores_f = scores_f[order]

    if float(nms_thr) >= 0:
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_f)
        keep_idx = _nms_xyxy(boxes_xyxy, scores_f, float(nms_thr))
        boxes_f = boxes_f[keep_idx]
        scores_f = scores_f[keep_idx]

    labels = [f"patch({v:.3f})" for v in scores_f.tolist()]
    return boxes_f, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, default=None, help="text prompt (default: object .)")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--patch_only", action="store_true", help="run patch-only inference (Stage A style)")
    parser.add_argument("--patch_image", type=str, default=None, help="support patch image path (optional)")
    parser.add_argument("--patch_emb", type=str, default=None, help="support patch embedding .npy path (optional)")
    parser.add_argument("--patch_size", type=int, default=224, help="support patch crop size (for patch_image)")
    parser.add_argument("--patch_topk", type=int, default=100, help="top-k queries to keep after threshold")
    parser.add_argument("--patch_nms_thr", type=float, default=0.6, help="NMS IoU threshold for patch-only boxes; set <0 to disable")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt if args.text_prompt is not None else "object ."
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")


    if args.patch_only:
        if token_spans is not None:
            raise ValueError("--patch_only does not support --token_spans (no text matching head is used).")
        patch_img_t = None
        patch_global_t = None
        if args.patch_image:
            patch_img_t = load_patch_image(args.patch_image, patch_size=args.patch_size)
            Image.open(args.patch_image).convert("RGB").save(os.path.join(output_dir, "support_patch.jpg"))
        if args.patch_emb:
            arr = np.load(args.patch_emb)
            patch_global_t = torch.from_numpy(arr).to(torch.float32).view(-1)
        if (patch_img_t is None) and (patch_global_t is None):
            raise ValueError("--patch_only requires --patch_image or --patch_emb")
        boxes_filt, pred_phrases = get_patch_only_output(
            model,
            image,
            patch_image=patch_img_t,
            patch_global=patch_global_t,
            caption=text_prompt,
            box_threshold=box_threshold,
            topk=args.patch_topk,
            nms_thr=args.patch_nms_thr,
            cpu_only=args.cpu_only,
        )
    else:
        if args.text_prompt is None:
            raise ValueError("Non-patch inference requires --text_prompt.")
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=token_spans
        )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    save_path = os.path.join(output_dir, "pred.jpg")
    image_with_box.save(save_path)
    print(f"\n======================\n{save_path} saved.\nThe program runs successfully!")
