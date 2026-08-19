#!/usr/bin/env python3
"""Evaluate the original GroundingDINO-T OGC checkpoint on FineCops-Ref."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util import box_ops
from tools.arrow_finecops_common import file_record, load_json, write_json_atomic
from tools.arrow_original_gdino_common import (
    CHECKPOINT_NUMEL,
    CHECKPOINT_TENSORS,
    EXPECTED_UNUSED_CHECKPOINT_KEYS,
    PRIMARY_SCORE,
    RECORD_SCHEMA,
    RUN_SCHEMA,
    SENSITIVITY_SCORE,
    load_preregistration,
    verify_file,
)
from tools.eval_arrow_finecops import _datasetinfo
from tools.eval_text_groundingdino_refcoco_tn import (
    _build_loader,
    _forward_ref_batch,
    _load_model_with_checkpoint_contract,
)
from tools.stageb_eval_records import (
    load_eval_manifest,
    validate_eval_manifest_batch_alignment,
)
from util.slconfig import SLConfig
from util.utils import clean_state_dict


DEFAULT_PREREG = (
    REPO_ROOT / "outputs/arrow_original_gdino_ogc_finecops_20260819/preregistration.json"
)


def _source_record(prereg: Mapping[str, Any], suffix: str) -> Mapping[str, Any]:
    matches = [
        row for row in prereg["sources"] if str(row.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one preregistered source ending in {suffix}")
    return matches[0]


def _audit_loaded_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(payload) != {"model"} or not isinstance(payload["model"], dict):
        raise ValueError("original OGC checkpoint payload must contain only model")
    provided = clean_state_dict(payload["model"])
    root = model.module if hasattr(model, "module") else model
    runtime = root.state_dict()
    missing = sorted(set(runtime).difference(provided))
    unexpected = sorted(set(provided).difference(runtime))
    shape_mismatch = sorted(
        key
        for key in set(runtime).intersection(provided)
        if tuple(runtime[key].shape) != tuple(provided[key].shape)
    )
    if missing or shape_mismatch or tuple(unexpected) != EXPECTED_UNUSED_CHECKPOINT_KEYS:
        raise ValueError(
            "original OGC ownership drifted: "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}, "
            f"shape_mismatch={shape_mismatch[:4]}"
        )
    if len(provided) != CHECKPOINT_TENSORS:
        raise ValueError("original OGC tensor count drifted")
    numel = sum(int(value.numel()) for value in provided.values())
    if numel != CHECKPOINT_NUMEL:
        raise ValueError("original OGC parameter count drifted")
    unequal: list[str] = []
    for key in sorted(runtime):
        observed = runtime[key].detach().cpu()
        expected = provided[key].detach().cpu()
        if not torch.equal(observed, expected):
            unequal.append(key)
            if len(unequal) >= 4:
                break
    if unequal:
        raise ValueError(f"original OGC loaded tensor parity failed: {unequal}")
    return {
        "checkpoint_tensors": len(provided),
        "runtime_tensors": len(runtime),
        "parameter_numel": numel,
        "loaded_tensor_parity": "bitwise_equal",
        "unused_legacy_tensors": list(unexpected),
    }


def _query_scores(outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    logits = outputs.get("pred_logits")
    boxes = outputs.get("pred_boxes")
    phrase_rows = outputs.get("phrase_to_token_mask")
    if (
        not torch.is_tensor(logits)
        or not torch.is_tensor(boxes)
        or logits.dim() != 3
        or boxes.dim() != 3
        or tuple(logits.shape[:2]) != tuple(boxes.shape[:2])
    ):
        raise ValueError("original OGC output lacks aligned logits/boxes")
    if (
        not torch.is_tensor(phrase_rows)
        or phrase_rows.dtype != torch.bool
        or phrase_rows.dim() != 3
        or int(phrase_rows.shape[0]) != int(logits.shape[0])
        or int(phrase_rows.shape[2]) != int(logits.shape[2])
    ):
        raise ValueError("original OGC output lacks its generated phrase mask")
    mask = phrase_rows.any(dim=1)
    token_count = mask.sum(dim=1)
    if bool((token_count == 0).any().item()):
        raise ValueError("original OGC produced an empty expression mask")
    probability = logits.detach().float().sigmoid()
    if not bool(torch.isfinite(probability).all().item()):
        raise ValueError("original OGC token probability is non-finite")
    expression_mean = (
        probability.masked_fill(~mask[:, None, :], 0.0).sum(dim=2)
        / token_count[:, None].to(dtype=probability.dtype)
    )
    expression_max = probability.masked_fill(
        ~mask[:, None, :], -torch.inf
    ).max(dim=2).values
    if not bool(
        torch.isfinite(expression_mean).all().item()
        and torch.isfinite(expression_max).all().item()
    ):
        raise ValueError("original OGC query score is non-finite")
    return {
        PRIMARY_SCORE: expression_mean,
        SENSITIVITY_SCORE: expression_max,
    }


def _pixel_box(box_cxcywh: torch.Tensor, target: Mapping[str, Any]) -> list[float]:
    orig_size = target.get("orig_size")
    if not torch.is_tensor(orig_size) or orig_size.numel() != 2:
        raise ValueError("FineCops target lacks orig_size")
    height, width = [float(value) for value in orig_size.view(-1).tolist()]
    box = box_ops.box_cxcywh_to_xyxy(box_cxcywh.view(1, 4).detach().float())[0]
    scale = box.new_tensor([width, height, width, height])
    return (box.clamp(0.0, 1.0) * scale).cpu().tolist()


def _gt_iou(
    boxes_cxcywh: torch.Tensor, target: Mapping[str, Any], query_index: int
) -> float:
    target_boxes = target.get("boxes")
    if not torch.is_tensor(target_boxes) or target_boxes.numel() == 0:
        return float("nan")
    prediction = box_ops.box_cxcywh_to_xyxy(
        boxes_cxcywh[query_index : query_index + 1].detach().float()
    ).clamp(0.0, 1.0)
    truth = box_ops.box_cxcywh_to_xyxy(
        target_boxes[:1].to(prediction.device).detach().float()
    ).clamp(0.0, 1.0)
    iou_output = box_ops.box_iou(prediction, truth)
    iou = iou_output[0] if isinstance(iou_output, tuple) else iou_output
    return float(iou[0, 0].item())


def run(
    *,
    preregistration_path: Path,
    device_name: str,
    max_batches: int,
    log_every: int,
) -> dict[str, Any]:
    prereg = load_preregistration(preregistration_path)
    verify_file(_source_record(prereg, "/tools/eval_arrow_original_gdino_finecops.py"), label="evaluator")
    dataset_path = verify_file(prereg["dataset"], label="dataset manifest")
    dataset = load_json(dataset_path)
    manifest_path = verify_file(prereg["manifest"], label="FineCops manifest")
    eval_manifest = load_eval_manifest(
        manifest_path, task="ref", split="finecops_test"
    )
    config_path = verify_file(prereg["config"], label="original OGC config")
    checkpoint_path = verify_file(prereg["checkpoint"], label="original OGC checkpoint")

    execution = prereg["execution"]
    formal_root = Path(str(execution["results_root"])).resolve()
    output_dir = (
        formal_root
        if max_batches == 0
        else formal_root.parent / "engineering_smoke"
    )
    if output_dir.exists():
        unexpected = {path.name for path in output_dir.iterdir()} - {
            "launch.json",
            "run.log",
        }
        if unexpected:
            raise ValueError(
                f"original OGC output already contains artifacts: {sorted(unexpected)}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SLConfig.fromfile(str(config_path))
    if not bool(getattr(cfg, "stage_b_arrow_original_ogc_eval", False)):
        raise ValueError("config is not the original OGC evaluation config")
    forbidden_flags = (
        "stage_b_gdino_score_adapter",
        "stage_b_u0_patch_rank",
        "stage_b_data_driven_score",
        "stage_b_native_patch_category",
        "enable_patch_branch",
    )
    if any(bool(getattr(cfg, key, False)) for key in forbidden_flags):
        raise ValueError("original OGC config enabled an ARROW/Stage-B branch")
    device = torch.device(device_name)
    cfg.device = str(device)
    model, checkpoint_summary = _load_model_with_checkpoint_contract(
        cfg, checkpoint_path, device
    )
    ownership = _audit_loaded_checkpoint(model, checkpoint_path)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("original OGC evaluation left trainable parameters")

    loader = _build_loader(
        cfg,
        _datasetinfo(dataset, route="B", manifest_path=manifest_path),
        int(execution["batch_size"]),
        int(execution["num_workers"]),
        device,
        int(execution["loader_seed"]),
    )
    records_path = output_dir / "records.jsonl"
    temporary = records_path.with_name(records_path.name + ".tmp")
    count = 0
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with temporary.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            raw_targets = list(batch[1])
            validate_eval_manifest_batch_alignment(raw_targets, eval_manifest, count)
            outputs, targets, _ = _forward_ref_batch(
                cfg, model, batch, device, amp=bool(execution["amp"])
            )
            scores = _query_scores(outputs)
            boxes = outputs["pred_boxes"].detach().float()
            top_indices = {name: score.argmax(dim=1) for name, score in scores.items()}
            source_rows = eval_manifest.rows[count : count + len(targets)]
            for local_index, (target, source_row) in enumerate(zip(targets, source_rows)):
                routes: dict[str, Any] = {}
                for name in (PRIMARY_SCORE, SENSITIVITY_SCORE):
                    query_index = int(top_indices[name][local_index].item())
                    raw_confidence = float(scores[name][local_index, query_index].item())
                    if not math.isfinite(raw_confidence) or not 0.0 <= raw_confidence <= 1.0:
                        raise ValueError("original OGC confidence left [0,1]")
                    routes[name] = {
                        "top1_query_index": query_index,
                        "top1_box_xyxy": _pixel_box(
                            boxes[local_index, query_index], target
                        ),
                        "top1_iou": _gt_iou(boxes[local_index], target, query_index),
                        "raw_confidence": raw_confidence,
                        "official_probability": raw_confidence,
                    }
                record = {
                    "schema": RECORD_SCHEMA,
                    "sample_id": source_row["sample_id"],
                    "finecops_annotation_id": int(source_row["finecops_annotation_id"]),
                    "parent_positive_id": int(source_row["finecops_parent_positive_id"]),
                    "cluster_gqa_image_id": int(source_row["finecops_cluster_gqa_image_id"]),
                    "kind": source_row["finecops_kind"],
                    "level": int(source_row["finecops_level"]),
                    "tuple_type": source_row["finecops_tuple_type"],
                    "negative_type": source_row.get("finecops_negative_type"),
                    "negative_level": source_row.get("finecops_negative_level"),
                    "model_inputs": "image_and_full_expression_only",
                    "routes": routes,
                }
                handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
                count += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if log_every and (batch_index == 0 or (batch_index + 1) % log_every == 0):
                print(
                    f"[original-OGC/FineCops] batch={batch_index + 1}/{len(loader)} "
                    f"rows={count} elapsed={(time.time() - start) / 60:.1f}m",
                    flush=True,
                )
    os.replace(temporary, records_path)
    if not max_batches and count != eval_manifest.size:
        raise ValueError(f"original OGC rows {count} != {eval_manifest.size}")
    receipt = {
        "schema": RUN_SCHEMA,
        "status": "complete" if not max_batches else "engineering_smoke",
        "preregistration": file_record(preregistration_path),
        "checkpoint": file_record(checkpoint_path),
        "config": file_record(config_path),
        "records": file_record(records_path, rows=count),
        "count": count,
        "ownership": ownership,
        "checkpoint_summary": checkpoint_summary,
        "runtime": {
            "device": str(device),
            "amp": bool(execution["amp"]),
            "batch_size": int(execution["batch_size"]),
            "num_workers": int(execution["num_workers"]),
            "elapsed_seconds": time.time() - start,
            "max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "optimizer_created": False,
            "grad_enabled": False,
        },
        "invariants": {
            "full_expression_only": True,
            "support_not_passed_to_model": True,
            "canonical_not_passed_to_model": True,
            "no_training": True,
            "no_threshold_fitting": True,
            "primary_score_predeclared": PRIMARY_SCORE,
            "sensitivity_score_predeclared": SENSITIVITY_SCORE,
        },
    }
    write_json_atomic(output_dir / "run_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.max_batches < 0:
        raise ValueError("--max-batches must be non-negative")
    receipt = run(
        preregistration_path=args.preregistration.resolve(strict=True),
        device_name=args.device,
        max_batches=args.max_batches,
        log_every=args.log_every,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
