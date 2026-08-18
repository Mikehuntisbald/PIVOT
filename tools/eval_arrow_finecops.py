#!/usr/bin/env python3
"""Run one preregistered ARROW FineCops-Ref route/seed evaluation."""

from __future__ import annotations

import argparse
import hashlib
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
from tools.arrow_finecops_common import (
    PREREG_SCHEMA,
    RECORD_SCHEMA,
    digest_file,
    file_record,
    load_json,
    write_json_atomic,
)
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


DEFAULT_PREREG = REPO_ROOT / "outputs/arrow_finecops_20260819/preregistration.json"
DEFAULT_AMENDMENT = REPO_ROOT / "outputs/arrow_finecops_20260819/preregistration_amendment.json"


def _verify_artifact(expected: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(expected.get("path", ""))).expanduser().resolve(strict=True)
    observed = file_record(path)
    for key in ("sha256", "size_bytes"):
        if observed[key] != expected.get(key):
            raise ValueError(f"{label} {key} drifted")
    return path


def _dataset_manifest(prereg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _verify_artifact(prereg["dataset"], label="dataset manifest")
    payload = load_json(path)
    if payload.get("schema") != "arrow.finecops.dataset_manifest/v1":
        raise ValueError("FineCops dataset manifest schema drifted")
    return path, payload


def _datasetinfo(
    dataset_payload: Mapping[str, Any], *, route: str, manifest_path: Path
) -> dict[str, Any]:
    inputs = dataset_payload["inputs"]
    support_tsv = _verify_artifact(
        dataset_payload["manifests"]["support_tsv"], label="selected support TSV"
    )
    canonical = _verify_artifact(
        inputs["canonical_taxonomy"], label="canonical taxonomy"
    )
    return {
        "name": f"finecops_test_{route.lower()}",
        "dataset_mode": "patch_episode",
        "root": "/",
        "anno": str(manifest_path),
        "box_format": "xywh",
        "canonical_classes_json": str(canonical),
        "keep_only_support_gt": True,
        "neg_episode_prob": 0.0,
        "support_min_count": 1,
        "support_patch_size": 224,
        "support_num_patches_min": 1,
        "support_num_patches_max": 1,
        "support_patch_tsv": str(support_tsv),
        "support_patch_bucket": "clean",
        "support_patch_use_embedding": False,
        "support_patch_max_per_class": 1,
        "patch_emb_cache_size": 512,
        "build_text_token_masks": True,
        "text_mask_skip_invalid_canonical": False,
        "text_mask_warn_limit": 0,
        "tn_balance_sampling": False,
        "stage_b_gdino_adapter_ref_eval": True,
    }


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


def _validate_runtime_source(
    preregistration_path: Path, prereg: Mapping[str, Any]
) -> dict[str, Any] | None:
    expected = next(
        row
        for row in prereg["sources"]
        if str(row["path"]).endswith("/tools/eval_arrow_finecops.py")
    )
    observed = file_record(Path(__file__).resolve())
    if all(observed[key] == expected[key] for key in ("sha256", "size_bytes")):
        return None
    if not DEFAULT_AMENDMENT.is_file():
        raise ValueError("FineCops evaluator changed after preregistration without amendment")
    amendment = load_json(DEFAULT_AMENDMENT)
    if amendment.get("schema") != "arrow.finecops.preregistration_amendment/v1":
        raise ValueError("FineCops preregistration amendment schema drifted")
    base = amendment.get("base_preregistration") or {}
    base_observed = file_record(preregistration_path)
    if any(base_observed[key] != base.get(key) for key in ("sha256", "size_bytes")):
        raise ValueError("FineCops amendment does not bind the base preregistration")
    current = amendment.get("current_evaluator") or {}
    if any(observed[key] != current.get(key) for key in ("sha256", "size_bytes")):
        raise ValueError("FineCops evaluator differs from its amendment")
    if amendment.get("change_scope") != "box_iou_tuple_api_compatibility_only":
        raise ValueError("FineCops amendment change scope is not allowlisted")
    return file_record(DEFAULT_AMENDMENT)


def _pixel_box(box_cxcywh: torch.Tensor, target: Mapping[str, Any]) -> list[float]:
    orig_size = target.get("orig_size")
    if not torch.is_tensor(orig_size) or orig_size.numel() != 2:
        raise ValueError("FineCops target lacks orig_size")
    height, width = [float(value) for value in orig_size.view(-1).tolist()]
    box = box_ops.box_cxcywh_to_xyxy(box_cxcywh.view(1, 4).detach().float())[0]
    scale = box.new_tensor([width, height, width, height])
    return (box.clamp(0.0, 1.0) * scale).cpu().tolist()


def _masked_max(score: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return score.max(dim=1).values
    valid = mask.to(device=score.device, dtype=torch.bool)
    if tuple(valid.shape) != tuple(score.shape) or bool((~valid.any(dim=1)).any().item()):
        raise ValueError("candidate mask does not align with scores")
    return score.masked_fill(~valid, -torch.inf).max(dim=1).values


def _route_scores(outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    required = {
        "b58": "stage_b_gdino_base_score",
        "r100_d3": "stage_b_gdino_rank_score",
        "deployed": "stage_b_u0_rank_score",
    }
    result: dict[str, torch.Tensor] = {}
    boxes = outputs.get("pred_boxes")
    if not torch.is_tensor(boxes) or boxes.dim() != 3:
        raise ValueError("FineCops output has no aligned pred_boxes")
    for route, key in required.items():
        value = outputs.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(boxes.shape[:2]):
            raise ValueError(f"FineCops output lacks aligned {key}")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"FineCops output {key} is non-finite")
        result[route] = value.detach().float()
    return result


def _apply_route_admission_inputs(
    route: str,
    targets: list[dict[str, Any]],
    source_rows: list[Mapping[str, Any]],
) -> None:
    if route != "B":
        return
    if len(targets) != len(source_rows):
        raise ValueError("FineCops admission inputs do not align with the batch")
    for target, source_row in zip(targets, source_rows):
        active = str(source_row["finecops_active_category"]).strip()
        if not active:
            raise ValueError("FineCops B row has an empty active category")
        # The legacy patch_episode loader derives stage_a_caption from its
        # support class.  B explicitly owns a category-text admission input,
        # while its geometry/R100 caption remains the full expression.
        target["stage_a_caption"] = active + " ."


def run(
    *,
    preregistration_path: Path,
    route: str,
    seed: int,
    device_name: str,
    max_batches: int,
    log_every: int,
) -> dict[str, Any]:
    route = route.upper()
    if route not in {"A", "B", "C"} or seed not in {17, 42, 73}:
        raise ValueError("FineCops route/seed is outside preregistration")
    prereg = load_json(preregistration_path)
    allowed_statuses = {
        "locked_before_any_finecops_model_forward",
        "locked_before_official_gqa_byte_correction_replay",
    }
    if (
        prereg.get("schema") != PREREG_SCHEMA
        or prereg.get("status") not in allowed_statuses
    ):
        raise ValueError("FineCops preregistration is not locked")
    _verify_artifact(file_record(preregistration_path), label="preregistration")
    amendment_record = _validate_runtime_source(preregistration_path, prereg)
    _, dataset = _dataset_manifest(prereg)
    manifest_record = dataset["manifests"]["a_eval" if route == "A" else "bc_full"]
    manifest_path = _verify_artifact(manifest_record, label=f"route {route} manifest")
    eval_manifest = load_eval_manifest(
        manifest_path, task="ref", split="finecops_test"
    )

    config_path = _verify_artifact(prereg["configs"][route], label=f"route {route} config")
    checkpoint_path = _verify_artifact(
        prereg["checkpoints"][route][str(seed)],
        label=f"route {route}/seed{seed} checkpoint",
    )
    execution = prereg["execution"]
    results_root = Path(str(execution["results_root"])).resolve()
    output_dir = results_root / route / f"seed{seed}"
    if output_dir.exists():
        unexpected = {
            path.name for path in output_dir.iterdir()
        } - {"launch.json", "run.log"}
        if unexpected:
            raise ValueError(
                f"FineCops output already contains result artifacts: {sorted(unexpected)}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SLConfig.fromfile(str(config_path))
    if not bool(getattr(cfg, "stage_b_arrow_finecops_eval", False)):
        raise ValueError("config is not a FineCops evaluation config")
    if str(getattr(cfg, "stage_b_arrow_finecops_row", "")) != route:
        raise ValueError("FineCops config route drifted")
    device = torch.device(device_name)
    cfg.device = str(device)
    model, checkpoint_summary = _load_model_with_checkpoint_contract(
        cfg, checkpoint_path, device
    )
    model.eval()
    loader = _build_loader(
        cfg,
        _datasetinfo(dataset, route=route, manifest_path=manifest_path),
        int(execution["batch_size"]),
        int(execution["num_workers"]),
        device,
        20260819,
    )
    records_path = output_dir / "records.jsonl"
    temporary = records_path.with_name(records_path.name + ".tmp")
    count = 0
    start = time.time()
    with temporary.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            raw_targets = list(batch[1])
            validate_eval_manifest_batch_alignment(raw_targets, eval_manifest, count)
            _apply_route_admission_inputs(
                route,
                raw_targets,
                eval_manifest.rows[count : count + len(raw_targets)],
            )
            outputs, targets, _ = _forward_ref_batch(
                cfg, model, batch, device, amp=bool(execution["amp"])
            )
            scores = _route_scores(outputs)
            boxes = outputs["pred_boxes"].detach().float()
            confidence = outputs.get("stage_b_gdino_confidence_score")
            candidate_mask = outputs.get("stage_b_gdino_candidate_mask")
            if candidate_mask is None:
                candidate_mask = outputs.get("stage_b_u0_candidate_mask")
            if not torch.is_tensor(confidence) or tuple(confidence.shape) != tuple(boxes.shape[:2]):
                raise ValueError("FineCops output lacks D3 confidence")
            d3_confidence = _masked_max(confidence.detach().float(), candidate_mask)
            b58_confidence = _masked_max(scores["b58"], candidate_mask)
            eligible = outputs.get("stage_b_u0_category_gate_eligible_mask")
            if not torch.is_tensor(eligible) or tuple(eligible.shape) != tuple(boxes.shape[:2]):
                raise ValueError("FineCops output lacks admission eligibility")

            top_indices = {
                name: value.argmax(dim=1) for name, value in scores.items()
            }
            for local_index, (target, source_row) in enumerate(
                zip(targets, eval_manifest.rows[count : count + len(targets)])
            ):
                route_payload: dict[str, Any] = {}
                for name in ("b58", "r100_d3", "deployed"):
                    query_index = int(top_indices[name][local_index].item())
                    raw_confidence = (
                        float(b58_confidence[local_index].item())
                        if name == "b58"
                        else float(d3_confidence[local_index].item())
                    )
                    if not math.isfinite(raw_confidence):
                        raise ValueError("FineCops sample confidence is non-finite")
                    route_payload[name] = {
                        "top1_query_index": query_index,
                        "top1_box_xyxy": _pixel_box(
                            boxes[local_index, query_index], target
                        ),
                        "top1_iou": _gt_iou(
                            boxes[local_index], target, query_index
                        ),
                        "raw_confidence": raw_confidence,
                        "official_probability": float(
                            torch.sigmoid(torch.tensor(raw_confidence)).item()
                        ),
                    }
                mask = eligible[local_index].detach().to("cpu", dtype=torch.uint8).contiguous()
                record = {
                    "schema": RECORD_SCHEMA,
                    "sample_id": source_row["sample_id"],
                    "route": route,
                    "seed": seed,
                    "finecops_annotation_id": int(source_row["finecops_annotation_id"]),
                    "parent_positive_id": int(source_row["finecops_parent_positive_id"]),
                    "cluster_gqa_image_id": int(source_row["finecops_cluster_gqa_image_id"]),
                    "kind": source_row["finecops_kind"],
                    "level": int(source_row["finecops_level"]),
                    "tuple_type": source_row["finecops_tuple_type"],
                    "negative_type": source_row.get("finecops_negative_type"),
                    "negative_level": source_row.get("finecops_negative_level"),
                    "active_category": source_row["finecops_active_category"],
                    "admission_caption": (
                        str(source_row["finecops_active_category"]) + " ."
                        if route == "B"
                        else None
                    ),
                    "support_covered": bool(source_row["finecops_support_covered"]),
                    "eligible_queries": int(eligible[local_index].sum().item()),
                    "eligible_mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                    "routes": route_payload,
                }
                handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                    + "\n"
                )
                count += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if log_every and (batch_index == 0 or (batch_index + 1) % log_every == 0):
                elapsed = time.time() - start
                print(
                    f"[FineCops] {route}/seed{seed} batch={batch_index + 1}/{len(loader)} "
                    f"rows={count} elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
    os.replace(temporary, records_path)
    expected_count = eval_manifest.size if not max_batches else count
    if count != expected_count:
        raise ValueError(f"FineCops output count {count} != expected {expected_count}")
    summary = {
        "schema": "arrow.finecops.run_receipt/v1",
        "route": route,
        "seed": seed,
        "status": "complete" if not max_batches else "engineering_smoke",
        "checkpoint": file_record(checkpoint_path),
        "checkpoint_contract_summary": checkpoint_summary,
        "config": file_record(config_path),
        "manifest": file_record(manifest_path, rows=eval_manifest.size),
        "records": file_record(records_path, rows=count),
        "rows": count,
        "seconds": time.time() - start,
        "device": str(device),
        "batch_size": int(execution["batch_size"]),
        "num_workers": int(execution["num_workers"]),
        "amp": bool(execution["amp"]),
        "finecops_threshold_fitted": False,
        "preregistration_amendment": amendment_record,
    }
    write_json_atomic(output_dir / "run_receipt.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--route", choices=("A", "B", "C"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 73), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    result = run(
        preregistration_path=args.preregistration.resolve(strict=True),
        route=args.route,
        seed=args.seed,
        device_name=args.device,
        max_batches=args.max_batches,
        log_every=args.log_every,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
