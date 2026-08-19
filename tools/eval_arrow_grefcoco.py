#!/usr/bin/env python3
"""Run one preregistered ARROW gRefCOCO confidence-only seed."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import util.misc as utils
from datasets.coco import make_coco_transforms
from groundingdino.util import box_ops
from tools.arrow_grefcoco_common import (
    PREREG_SCHEMA,
    RECORD_SCHEMA,
    RUN_SCHEMA,
    SEEDS,
    file_record,
    load_json,
    load_jsonl,
    verify_record,
    write_json_atomic,
)
from tools.eval_text_groundingdino_refcoco_tn import _load_model_with_checkpoint_contract
from util.slconfig import SLConfig


DEFAULT_PREREG = REPO_ROOT / "outputs/arrow_grefcoco_20260820/preregistration.json"


def _gdino_caption(expression: str) -> str:
    """Add only GroundingDINO's terminal phrase delimiter."""
    text = str(expression).strip()
    if not text:
        raise ValueError("gRefCOCO expression is empty")
    return text if text.endswith(".") else text + " ."


class GRefConfidenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], cfg: Any) -> None:
        if any(str(row["split"]).lower() == "train" for row in rows):
            raise ValueError("gRefCOCO train rows are forbidden")
        self.rows = rows
        self.transform = make_coco_transforms("val", args=cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        if (width, height) != (int(row["image_width"]), int(row["image_height"])):
            raise ValueError(f"image dimensions drifted for {row['sample_id']}")
        target: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "manifest_index": index,
            "caption": row["expression"],
            "orig_size": torch.as_tensor([height, width], dtype=torch.int64),
            "size": torch.as_tensor([height, width], dtype=torch.int64),
        }
        if row["kind"] == "positive":
            x, y, w, h = [float(value) for value in row["bbox_xywh"]]
            target["boxes"] = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        image, target = self.transform(image, target)
        return image, target


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_max(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tuple(score.shape) != tuple(mask.shape) or bool((~mask.any(dim=1)).any().item()):
        raise ValueError("candidate mask does not align with confidence scores")
    return score.masked_fill(~mask, -torch.inf).max(dim=1).values


def _iou(boxes: torch.Tensor, target: Mapping[str, Any], query_index: int) -> float | None:
    truth = target.get("boxes")
    if not torch.is_tensor(truth) or truth.numel() == 0:
        return None
    prediction = box_ops.box_cxcywh_to_xyxy(boxes[query_index : query_index + 1].float()).clamp(0, 1)
    target_xyxy = box_ops.box_cxcywh_to_xyxy(truth[:1].to(prediction.device).float()).clamp(0, 1)
    result = box_ops.box_iou(prediction, target_xyxy)
    matrix = result[0] if isinstance(result, tuple) else result
    return float(matrix[0, 0].item())


def _pixel_box(box: torch.Tensor, target: Mapping[str, Any]) -> list[float]:
    height, width = [float(value) for value in target["orig_size"].view(-1).tolist()]
    xyxy = box_ops.box_cxcywh_to_xyxy(box.view(1, 4).float())[0].clamp(0, 1)
    return (xyxy * xyxy.new_tensor([width, height, width, height])).cpu().tolist()


def _validate_prereg(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prereg = load_json(path)
    if prereg.get("schema") != PREREG_SCHEMA or prereg.get("status") != "locked_before_any_grefcoco_model_forward":
        raise ValueError("gRefCOCO preregistration is not locked")
    for source in prereg["sources"]:
        verify_record(source, label="preregistered source")
    dataset_path = verify_record(prereg["dataset"], label="dataset manifest")
    dataset = load_json(dataset_path)
    test_path = verify_record(dataset["manifests"]["testab"], label="test manifest")
    val_path = verify_record(dataset["manifests"]["val_no_target"], label="val manifest")
    test_rows = load_jsonl(test_path)
    val_rows = load_jsonl(val_path)
    if len(test_rows) != 20684 or len(val_rows) != 8905:
        raise ValueError("gRefCOCO runtime manifest count drifted")
    return prereg, test_rows, val_rows


def run(preregistration: Path, seed: int, device_name: str, max_batches: int = 0) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("seed is outside the sealed contract")
    prereg, test_rows, val_rows = _validate_prereg(preregistration)
    checkpoint = verify_record(prereg["checkpoints"]["A"][str(seed)], label=f"A/seed{seed}")
    config = verify_record(prereg["config"], label="gRefCOCO config")
    output_dir = Path(prereg["execution"]["results_root"]) / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output_dir.iterdir()} - {"launch.json", "run.log"}
    if unexpected:
        raise ValueError(f"result directory is not clean: {sorted(unexpected)}")
    cfg = SLConfig.fromfile(str(config))
    if not bool(getattr(cfg, "stage_b_arrow_grefcoco_confidence_only", False)):
        raise ValueError("config is not the gRefCOCO confidence-only route")
    device = torch.device(device_name)
    cfg.device = str(device)
    _seed_everything(int(prereg["execution"]["loader_seed"]))
    model, checkpoint_summary = _load_model_with_checkpoint_contract(cfg, checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("gRefCOCO evaluator left trainable parameters")
    all_rows = test_rows + val_rows
    dataset = GRefConfidenceDataset(all_rows, cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(prereg["execution"]["batch_size"]),
        sampler=SequentialSampler(dataset),
        num_workers=int(prereg["execution"]["num_workers"]),
        collate_fn=utils.collate_fn,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_path = output_dir / "records.jsonl"
    temporary = output_path.with_name(output_path.name + ".tmp")
    count = 0
    start = time.time()
    with temporary.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            samples = batch[0].to(device)
            targets = list(batch[1])
            source_rows = all_rows[count : count + len(targets)]
            for target, row in zip(targets, source_rows):
                if target["sample_id"] != row["sample_id"]:
                    raise ValueError("runtime sample alignment drifted")
            captions = [_gdino_caption(str(row["expression"])) for row in source_rows]
            with torch.autocast(device_type=device.type, enabled=bool(prereg["execution"]["amp"]) and device.type == "cuda"):
                outputs = model(samples, captions=captions, stage_b_u2v2_confidence_only=True)
            forbidden = [key for key in outputs if key.startswith("stage_b_u0_") or key.startswith("stage_b_arrow_admission_")]
            if forbidden:
                raise RuntimeError(f"confidence-only bypass emitted admission outputs: {forbidden}")
            boxes = outputs.get("pred_boxes")
            base = outputs.get("stage_b_gdino_base_score")
            rank = outputs.get("stage_b_gdino_rank_score")
            confidence = outputs.get("stage_b_gdino_confidence_score")
            if not all(torch.is_tensor(value) for value in (boxes, base, rank, confidence)):
                raise RuntimeError("confidence-only output contract is incomplete")
            if tuple(base.shape) != tuple(rank.shape) or tuple(base.shape) != tuple(confidence.shape) or tuple(base.shape) != tuple(boxes.shape[:2]):
                raise RuntimeError("confidence-only score shapes do not align")
            candidate_mask = torch.ones_like(base, dtype=torch.bool)
            b58_conf = _masked_max(base.float(), candidate_mask)
            d3_conf = _masked_max(confidence.float(), candidate_mask)
            base_top = base.argmax(dim=1)
            rank_top = rank.argmax(dim=1)
            for local, (target, row) in enumerate(zip(targets, source_rows)):
                bq = int(base_top[local].item())
                rq = int(rank_top[local].item())
                bscore = float(b58_conf[local].item())
                dscore = float(d3_conf[local].item())
                if not math.isfinite(bscore) or not math.isfinite(dscore):
                    raise ValueError("non-finite rejection score")
                record = {
                    "schema": RECORD_SCHEMA,
                    "sample_id": row["sample_id"],
                    "manifest_index": count + local,
                    "split": row["split"],
                    "label": int(row["label"]),
                    "kind": row["kind"],
                    "image_id": int(row["image_id"]),
                    "ref_id": int(row["ref_id"]),
                    "sent_id": int(row["sent_id"]),
                    "ann_id": row["ann_id"],
                    "surface_d3_disjoint": bool(row.get("surface_d3_disjoint", False)),
                    "surface_d3_finecops_disjoint": bool(row.get("surface_d3_finecops_disjoint", False)),
                    "admission_defined": False,
                    "candidate_count": int(base.shape[1]),
                    "scores": {"b58": bscore, "d3": dscore},
                    "positive_localization": None,
                }
                if row["kind"] == "positive":
                    record["positive_localization"] = {
                        "b58_top1_query": bq,
                        "b58_top1_box_xyxy": _pixel_box(boxes[local, bq], target),
                        "b58_top1_iou": _iou(boxes[local], target, bq),
                        "r100_top1_query": rq,
                        "r100_top1_box_xyxy": _pixel_box(boxes[local, rq], target),
                        "r100_top1_iou": _iou(boxes[local], target, rq),
                    }
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
            count += len(targets)
            if batch_index % 100 == 0:
                print(json.dumps({"batch": batch_index, "rows": count}), flush=True)
    os.replace(temporary, output_path)
    expected_rows = len(all_rows) if not max_batches else count
    if count != expected_rows:
        raise ValueError(f"runtime emitted {count} rows, expected {expected_rows}")
    receipt = {
        "schema": RUN_SCHEMA,
        "status": "complete" if not max_batches else "smoke_complete",
        "seed": seed,
        "rows": count,
        "test_rows": len(test_rows) if not max_batches else None,
        "val_rows": len(val_rows) if not max_batches else None,
        "checkpoint": file_record(checkpoint),
        "checkpoint_contract": checkpoint_summary,
        "records": file_record(output_path, rows=count),
        "elapsed_seconds": time.time() - start,
        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "model_training": False,
        "optimizer_created": False,
        "admission_bypassed": True,
    }
    write_json_atomic(output_dir / "run_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.preregistration.resolve(strict=True), args.seed, args.device, args.max_batches), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
