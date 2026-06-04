#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from groundingdino.util import box_ops, get_tokenlizer  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from models.GroundingDINO.groundingdino import create_positive_map  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402

_WS_RE = re.compile(r"\s+")


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _load_model(cfg, ckpt_path: str, device: torch.device):
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={cfg.modelname}")
    model, _criterion, _postprocessors = build_func(cfg)
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    state = clean_state_dict(_extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] {ckpt_path}: missing keys={len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] {ckpt_path}: unexpected keys={len(unexpected)}", file=sys.stderr)
    model.to(device).eval()
    return model


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _support_classes_from_target(target: Dict[str, torch.Tensor]) -> torch.Tensor:
    support_classes = target.get("support_classes", None)
    if support_classes is None:
        support_classes = target.get("support_class", None)
    if support_classes is None:
        raise KeyError("target must contain support_classes or support_class")
    return support_classes.view(-1).to(torch.long)


def _path_env_defaults() -> Dict[str, str]:
    data_root = os.environ.get("DATA_ROOT", "/media/haoyi/T9/data")
    return {
        "DATA_ROOT": data_root,
        "GDINO_ROOT": str(REPO_ROOT),
    }


def _expand_path(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        out = value
        for key, default_value in _path_env_defaults().items():
            out = out.replace(f"${{{key}}}", default_value)
            out = out.replace(f"${key}", default_value)
        return os.path.expanduser(os.path.expandvars(out))
    if isinstance(value, list):
        return [_expand_path(v) for v in value]
    return value


def _clean_label_text(value: Any) -> str:
    text = str(value or "").replace("_", " ").replace(".", " ").strip()
    text = _WS_RE.sub(" ", text)
    return text or "object"


def _load_cid_to_name(canonical_classes_json: Any) -> Dict[int, str]:
    path_value = _expand_path(canonical_classes_json)
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        print(f"[WARN] canonical_classes_json not found: {path}", file=sys.stderr)
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"canonical_classes_json must contain a list, got {type(data)}")
    out: Dict[int, str] = {}
    for entry in data:
        if not isinstance(entry, dict) or entry.get("id", None) is None:
            continue
        preferred = None
        for key in ("base_name", "raw_name", "norm_name", "synset"):
            value = entry.get(key, None)
            if isinstance(value, str) and value.strip():
                preferred = _clean_label_text(value)
                break
        if preferred is None:
            for value in entry.get("synonyms", []) or []:
                if isinstance(value, str) and value.strip():
                    preferred = _clean_label_text(value)
                    break
        if preferred is not None:
            out[int(entry["id"])] = preferred
    return out


def _class_names_from_target(
    target: Dict[str, Any],
    support_classes: torch.Tensor,
    cid_to_name: Dict[int, str],
) -> List[str]:
    names: List[str] = []
    missing: List[int] = []
    for cid_raw in support_classes.detach().cpu().tolist():
        cid = int(cid_raw)
        if cid in cid_to_name:
            names.append(cid_to_name[cid])
        else:
            missing.append(cid)
            names.append(f"class {cid}")
    if not missing:
        return names
    cap_list = list(target.get("cap_list", []) or [])
    for i, name in enumerate(names):
        if name.startswith("class ") and i < len(cap_list):
            names[i] = _clean_label_text(cap_list[i])
    return names


def _make_prompt(names: List[str]) -> str:
    return " . ".join(str(x).strip() or "object" for x in names) + " ."


def _build_pos_map(tokenizer, caption: str, names: List[str]) -> torch.Tensor:
    tokenized = tokenizer(caption, padding="longest", return_tensors="pt")
    labels = torch.arange(len(names))
    return create_positive_map(tokenized, labels, names, caption)


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


class EvalAccumulator:
    def __init__(self, topks: Iterable[int], *, ap_max_dets_per_image: int) -> None:
        self.topks = sorted({max(1, int(k)) for k in topks})
        self.ap_max_dets_per_image = max(0, int(ap_max_dets_per_image))
        self.num_images = 0
        self.num_targets = 0
        self.supported_targets = 0
        self.unsupported_targets = 0
        self.box_recalled = {k: 0 for k in self.topks}
        self.best_iou_sum = {k: 0.0 for k in self.topks}
        self.detections: List[Tuple[float, int, int, float, float, float, float]] = []
        self.gt_by_key: Dict[Tuple[int, int], List[List[float]]] = {}
        self.ap_targets = 0

    def update(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]], pos_maps: List[torch.Tensor]) -> None:
        token_logits = outputs["pred_logits"].detach().float()
        pred_boxes = outputs["pred_boxes"].detach().float()
        device = pred_boxes.device
        bsz, num_queries, _num_tokens = token_logits.shape
        for b in range(bsz):
            image_id = self.num_images
            self.num_images += 1
            target = targets[b]
            support_classes = _support_classes_from_target(target).to(device)
            valid_slots = support_classes >= 0
            valid_slot_ids = valid_slots.nonzero(as_tuple=False).flatten()
            labels = target["labels"].to(device=device, dtype=torch.long)
            gt_boxes = target["boxes"].to(device=device, dtype=torch.float32)
            if labels.numel() == 0:
                continue
            pos_map = pos_maps[b].to(device=device, dtype=torch.float32)
            pos_map = pos_map[: int(support_classes.numel())]
            denom = pos_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            pos_map = pos_map / denom
            class_scores = token_logits[b].sigmoid() @ pos_map.T
            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[b]).clamp(0.0, 1.0)
            gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes).clamp(0.0, 1.0)
            iou_qt = box_ops.box_iou(pred_xyxy, gt_xyxy)[0]
            valid_label_set = set(int(x) for x in support_classes[valid_slot_ids].detach().cpu().tolist())
            cid_to_slots: Dict[int, List[int]] = {}
            for slot_i, cid in enumerate(support_classes.detach().cpu().tolist()):
                if int(cid) >= 0:
                    cid_to_slots.setdefault(int(cid), []).append(int(slot_i))
            for j, label_value in enumerate(labels.detach().cpu().tolist()):
                label = int(label_value)
                self.num_targets += 1
                if label not in valid_label_set:
                    self.unsupported_targets += 1
                    continue
                self.supported_targets += 1
                gt_box_list = gt_xyxy[j].detach().cpu().tolist()
                self.gt_by_key.setdefault((image_id, label), []).append([float(x) for x in gt_box_list])
                self.ap_targets += 1
                slot_ids = cid_to_slots.get(label, [])
                if not slot_ids:
                    continue
                scores_q = class_scores[:, torch.as_tensor(slot_ids, device=device)].max(dim=-1).values
                for k in self.topks:
                    topq = torch.topk(scores_q, k=min(k, num_queries), largest=True).indices
                    best_iou = float(iou_qt[topq, j].max().item()) if topq.numel() > 0 else 0.0
                    self.best_iou_sum[k] += best_iou
                    if best_iou >= 0.5:
                        self.box_recalled[k] += 1
            self._add_ap_detections(image_id, class_scores, pred_xyxy, support_classes, valid_slot_ids)

    def _add_ap_detections(
        self,
        image_id: int,
        class_scores: torch.Tensor,
        pred_xyxy: torch.Tensor,
        support_classes: torch.Tensor,
        valid_slot_ids: torch.Tensor,
    ) -> None:
        if self.ap_max_dets_per_image <= 0 or valid_slot_ids.numel() == 0:
            return
        slot_scores = class_scores[:, valid_slot_ids]
        flat_scores = slot_scores.reshape(-1)
        keep = min(int(self.ap_max_dets_per_image), int(flat_scores.numel()))
        if keep <= 0:
            return
        top_scores, top_idx = torch.topk(flat_scores, k=keep, largest=True)
        local_slot = top_idx % int(valid_slot_ids.numel())
        query_idx = torch.div(top_idx, int(valid_slot_ids.numel()), rounding_mode="floor")
        full_slot = valid_slot_ids[local_slot]
        det_labels = support_classes[full_slot]
        det_boxes = pred_xyxy[query_idx]
        for score, label, box in zip(top_scores.detach().cpu(), det_labels.detach().cpu(), det_boxes.detach().cpu()):
            x1, y1, x2, y2 = [float(x) for x in box.tolist()]
            self.detections.append((float(score.item()), int(image_id), int(label.item()), x1, y1, x2, y2))

    def _compute_ap50(self) -> float:
        if self.ap_targets <= 0 or not self.detections:
            return 0.0
        gt_arrays = {
            key: np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            for key, boxes in self.gt_by_key.items()
            if boxes
        }
        matched = {key: np.zeros((boxes.shape[0],), dtype=bool) for key, boxes in gt_arrays.items()}
        ordered = sorted(self.detections, key=lambda x: x[0], reverse=True)
        tp = np.zeros((len(ordered),), dtype=np.float32)
        fp = np.zeros((len(ordered),), dtype=np.float32)
        for i, det in enumerate(ordered):
            _score, image_id, label, x1, y1, x2, y2 = det
            key = (int(image_id), int(label))
            boxes = gt_arrays.get(key, None)
            if boxes is None or boxes.shape[0] == 0:
                fp[i] = 1.0
                continue
            ious = _iou_one_to_many(np.asarray([x1, y1, x2, y2], dtype=np.float32), boxes)
            found = False
            for gt_i in np.argsort(-ious):
                if float(ious[gt_i]) < 0.5:
                    break
                if not matched[key][gt_i]:
                    matched[key][gt_i] = True
                    tp[i] = 1.0
                    found = True
                    break
            if not found:
                fp[i] = 1.0
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(1.0, float(self.ap_targets))
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def result(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "num_images": int(self.num_images),
            "num_targets": int(self.num_targets),
            "supported_targets": int(self.supported_targets),
            "unsupported_targets": int(self.unsupported_targets),
            "target_support_coverage": self.supported_targets / max(1, self.num_targets),
            "ap_targets": int(self.ap_targets),
            "ap_detections": int(len(self.detections)),
            "patch_ap50": self._compute_ap50(),
        }
        for k in self.topks:
            out[f"box_recall@{k}"] = self.box_recalled[k] / max(1, self.supported_targets)
            out[f"mean_best_iou@{k}"] = self.best_iou_sum[k] / max(1, self.supported_targets)
        return out


def _build_loader(cfg, datasetinfo, batch_size: int, num_workers: int, device: torch.device, seed: int):
    _set_seed(seed)
    dataset = build_dataset(image_set="val", args=cfg, datasetinfo=datasetinfo)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _dataset_name(datasetinfo: Dict[str, Any], index: int) -> str:
    return str(datasetinfo.get("name") or datasetinfo.get("source") or f"val{index}").replace("/", "_")


@torch.no_grad()
def evaluate_one(
    *,
    cfg,
    model,
    tokenizer,
    datasetinfo,
    dataset_name: str,
    ckpt_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    topks: List[int],
    ap_max_dets_per_image: int,
    amp: bool,
    max_batches: int,
    log_every: int,
) -> Dict[str, Any]:
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    cid_to_name = _load_cid_to_name(datasetinfo.get("canonical_classes_json", None))
    acc = EvalAccumulator(topks, ap_max_dets_per_image=ap_max_dets_per_image)
    start = time.time()
    print(
        f"[INFO] eval ckpt={Path(ckpt_path).name} dataset={dataset_name} "
        f"images={len(loader.dataset)} batches={len(loader)} batch_size={batch_size}",
        flush=True,
    )
    for batch_i, (samples, targets) in enumerate(loader):
        if max_batches > 0 and batch_i >= max_batches:
            break
        samples = samples.to(device)
        targets = list(targets)
        prompts: List[str] = []
        pos_maps: List[torch.Tensor] = []
        for target in targets:
            support_classes = _support_classes_from_target(target)
            names = _class_names_from_target(target, support_classes, cid_to_name)
            prompt = _make_prompt(names)
            prompts.append(prompt)
            pos_maps.append(_build_pos_map(tokenizer, prompt, names))
        with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
            outputs = model(samples, captions=prompts)
        acc.update(outputs, targets, pos_maps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and ((batch_i + 1) % log_every == 0 or batch_i == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            total = min(len(loader), max_batches) if max_batches > 0 else len(loader)
            eta = elapsed / max(1, done) * max(0, total - done)
            print(
                f"[INFO] {dataset_name} {Path(ckpt_path).name}: "
                f"batch {done}/{total}, images={acc.num_images}, "
                f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                flush=True,
            )
    result = acc.result()
    result.update(
        {
            "checkpoint": str(ckpt_path),
            "checkpoint_name": Path(ckpt_path).name,
            "dataset": dataset_name,
            "seconds": time.time() - start,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "seed": int(seed),
            "max_batches": int(max_batches),
            "ap_max_dets_per_image": int(ap_max_dets_per_image),
        }
    )
    return result


def _checkpoint_label(path: str) -> str:
    return Path(path).stem.replace("/", "_")


def _mean_metric(results: List[Dict[str, Any]], ckpt: str, metric: str) -> float:
    vals = [float(r.get(metric, 0.0)) for r in results if r["checkpoint"] == ckpt]
    return sum(vals) / max(1, len(vals))


def _write_outputs(output_dir: Path, results: List[Dict[str, Any]], primary_metric: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpts = []
    seen = set()
    for r in results:
        if r["checkpoint"] not in seen:
            seen.add(r["checkpoint"])
            ckpts.append(r["checkpoint"])
    ranking = [
        {
            "rank": i + 1,
            "checkpoint": ckpt,
            f"mean_{primary_metric}": _mean_metric(results, ckpt, primary_metric),
            "mean_box_recall@50": _mean_metric(results, ckpt, "box_recall@50"),
        }
        for i, ckpt in enumerate(sorted(ckpts, key=lambda c: _mean_metric(results, c, primary_metric), reverse=True))
    ]
    payload = {"primary_metric": primary_metric, "ranking": ranking, "results": results}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    datasets = []
    seen_ds = set()
    for r in results:
        if r["dataset"] not in seen_ds:
            seen_ds.add(r["dataset"])
            datasets.append(r["dataset"])
    by_ckpt_ds = {(r["checkpoint"], r["dataset"]): r for r in results}
    lines = [
        "# Text GroundingDINO Stage-A-caliber eval summary",
        "",
        f"Primary metric: mean `{primary_metric}` across evaluated datasets.",
        "",
        "| rank | checkpoint | mean patch_ap50 | mean box_recall@50 | "
        + " | ".join(f"{ds} patch_ap50" for ds in datasets)
        + " |",
        "|---:|---|---:|---:|" + "|".join("---:" for _ in datasets) + "|",
    ]
    for row in ranking:
        ckpt = row["checkpoint"]
        ds_vals = [f"{float(by_ckpt_ds.get((ckpt, ds), {}).get('patch_ap50', 0.0)):.6f}" for ds in datasets]
        lines.append(
            f"| {row['rank']} | `{Path(ckpt).name}` | "
            f"{float(row[f'mean_{primary_metric}']):.6f} | "
            f"{float(row['mean_box_recall@50']):.6f} | "
            + " | ".join(ds_vals)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text GroundingDINO on Stage-A patch episode caliber.")
    parser.add_argument("--config", default="outputs/ogc_original_finetune_stage_a/cfg_ogc_original_finetune_stage_a.generated.py")
    parser.add_argument("--datasets", default="config/datasets_patch_stage_a_lvis_coco2017_eval_local.json")
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/ogc_original_finetune_stage_a_eval_stagea_caliber")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--ap_max_dets_per_image", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--primary_metric", default="patch_ap50")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.patch_only = False
    cfg.use_coco_eval = False
    cfg.batch_size = int(args.batch_size)
    tokenizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    with open(args.datasets, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)
    val_infos = list(dataset_meta.get("val", []))
    if not val_infos:
        raise ValueError(f"No val datasets in {args.datasets}")
    output_dir = Path(args.output_dir)
    results: List[Dict[str, Any]] = []
    for ckpt_i, ckpt in enumerate(args.ckpts):
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(str(ckpt_path))
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(args.ckpts)}: {ckpt}", flush=True)
        _set_seed(int(args.seed))
        model = _load_model(cfg, str(ckpt_path), device)
        for ds_i, datasetinfo in enumerate(val_infos):
            ds_name = _dataset_name(datasetinfo, ds_i)
            result = evaluate_one(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                datasetinfo=datasetinfo,
                dataset_name=ds_name,
                ckpt_path=str(ckpt_path),
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(args.seed) + ds_i * 100000,
                topks=list(args.topk),
                ap_max_dets_per_image=int(args.ap_max_dets_per_image),
                amp=bool(args.amp),
                max_batches=int(args.max_batches),
                log_every=int(args.log_every),
            )
            results.append(result)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{_checkpoint_label(str(ckpt_path))}__{ds_name}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _write_outputs(output_dir, results, str(args.primary_metric))
            print(
                f"[RESULT] {ckpt_path.name} {ds_name}: "
                f"patch_ap50={result['patch_ap50']:.6f} "
                f"box_recall@50={result.get('box_recall@50', 0.0):.6f}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_outputs(output_dir, results, str(args.primary_metric))
    print(f"[INFO] wrote {output_dir / 'summary.json'}", flush=True)
    print(f"[INFO] wrote {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
