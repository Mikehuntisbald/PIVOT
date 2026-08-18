#!/usr/bin/env python3
"""Evaluate ARROW A/B/C category-switch control on the fresh 512-pair panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groundingdino.util import box_ops
from groundingdino.util.misc import NestedTensor
from tools.eval_stageb_role_causal import (
    _batch_metadata,
    _metadata_index,
    verify_category_runtime_assets,
)
from tools.eval_text_groundingdino_refcoco_tn import (
    _load_model_with_checkpoint_contract,
    _prepare_stage_b_u0_patch_batch,
)
from tools.eval_refcoco_stageb import (
    _build_loader,
    _make_datasetinfo,
)
from tools.stageb_arrow_admission_contract import SOURCES
from util.slconfig import SLConfig


SCHEMA = "arrow.stageb.admission_panel/v1"
PANEL = ROOT / "data/ablations/stageb_table_a_category_intervention_20260717"


class ArrowPanelError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _boxes(meta: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
    category = meta[key]
    scale = torch.tensor(
        [meta["image_width"], meta["image_height"]] * 2,
        dtype=torch.float32, device=device,
    )
    return torch.as_tensor(category["boxes_xyxy"], device=device).float() / scale


def _query_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return box_ops.box_iou(pred, gt)[0].max(dim=1).values


def _joint_caption(intervention: Mapping[str, Any]) -> str:
    categories = sorted(
        (intervention["class_a"], intervention["class_b"]),
        key=lambda value: int(value["id"]),
    )
    return " ".join(f"{value['name']} ." for value in categories)


def _evaluate_one(
    *, cfg, checkpoint: Path, row_id: str, device: torch.device,
    batch_size: int, num_workers: int, seed: int, max_batches: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = _rows(PANEL / "category_intervention_pairs.jsonl")
    metadata_index = _metadata_index(source_rows)
    support = PANEL / "category_intervention_support.tsv"
    asset_receipt = verify_category_runtime_assets(source_rows, support)
    datasetinfo = _make_datasetinfo(
        Path("/media/haoyi/T9/data"), "arrow_category_panel",
        PANEL / "category_intervention_pairs.jsonl",
    )
    datasetinfo.update({
        "support_min_count": 1,
        "support_patch_tsv": str(support),
        "support_patch_bucket": "clean",
        "support_patch_use_embedding": False,
        "support_patch_max_per_class": 1,
        "support_patch_image_root": "/media/haoyi/T9/data/patches_quality",
        "keep_only_support_gt": True,
    })
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    model, _ = _load_model_with_checkpoint_contract(cfg, checkpoint, device)
    model.eval()
    records = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            samples, targets, _, patches, patch_global, patch_mask = (
                _prepare_stage_b_u0_patch_batch(batch, device)
            )
            metadata = _batch_metadata(targets, metadata_index)
            pair_indices = list(range(0, len(metadata), 2))
            if len(metadata) % 2:
                raise ArrowPanelError("panel batch split a category pair")
            for offset in pair_indices:
                left, right = metadata[offset], metadata[offset + 1]
                li, ri = left["category_intervention"], right["category_intervention"]
                if (
                    li["pair_id"] != ri["pair_id"]
                    or {li["arm"], ri["arm"]} != {"A", "B"}
                    or int(left["image_id"]) != int(right["image_id"])
                ):
                    raise ArrowPanelError("panel loader reordered category arms")
            pair_samples = NestedTensor(
                samples.tensors[pair_indices],
                samples.mask[pair_indices] if samples.mask is not None else None,
            )
            pair_targets = [targets[index] for index in pair_indices]
            pair_captions = [
                _joint_caption(metadata[index]["category_intervention"])
                for index in pair_indices
            ]
            base_arrow_captions = (
                [
                    str(metadata[index]["category_intervention"]["canonical_prompt"])
                    for index in pair_indices
                ]
                if SOURCES[row_id] == "canonical_text" else None
            )
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(
                    pair_samples, targets=pair_targets, captions=pair_captions,
                    patches=(patches[pair_indices] if patches is not None else None),
                    patch_global=(patch_global[pair_indices] if patch_global is not None else None),
                    patch_mask=(patch_mask[pair_indices] if patch_mask is not None else None),
                    patch_only=False,
                    disable_patch_dn=True,
                    stage_b_arrow_admission_captions=base_arrow_captions,
                    stage_b_arrow_panel_diagnostics=True,
                )
            query = outputs["stage_b_arrow_query_projection"].detach()
            if SOURCES[row_id] == "support_patch":
                if patches is not None:
                    provider = model.patch_encoder(
                        patches, text_dict=None, return_tokens=False
                    )["patch_global"]
                elif patch_global is not None:
                    provider = patch_global
                else:
                    raise ArrowPanelError("support route lacks patch input")
            elif SOURCES[row_id] == "canonical_text":
                active_captions = [
                    str(meta["category_intervention"]["canonical_prompt"])
                    for meta in metadata
                ]
                provider = model._project_stage_b_arrow_source(
                    model._stage_b_arrow_canonical_source(active_captions, device)
                )
            else:
                provider = model._project_stage_b_arrow_source(
                    model._stage_b_arrow_null_source(
                        len(metadata), device, query.dtype
                    )
                )
            query = query.repeat_interleave(2, dim=0)
            logit_scale = model.patch_logit_scale.exp().clamp(
                max=model.patch_logit_scale_max
            )
            score = (logit_scale * torch.einsum("bqd,bd->bq", query, provider)).float()
            rank = outputs["stage_b_gdino_rank_score"].detach().repeat_interleave(2, dim=0)
            candidate = outputs["stage_b_u0_candidate_mask"].detach().repeat_interleave(2, dim=0)
            adapter = model.stage_b_u0_patch_rank_adapter
            standardized = adapter._standardize(
                score, candidate, clip=adapter.score_clip
            )
            _, eligible = adapter.apply_category_preserving_gate(
                standardized, rank, candidate
            )
            boxes = box_ops.box_cxcywh_to_xyxy(
                outputs["pred_boxes"].detach().float()
            ).repeat_interleave(2, dim=0)
            for index, meta in enumerate(metadata):
                intervention = meta["category_intervention"]
                arm = str(intervention["arm"])
                active_key, counter_key = (
                    ("class_a", "class_b") if arm == "A" else ("class_b", "class_a")
                )
                active_iou = _query_iou(boxes[index], _boxes(intervention, active_key, device))
                counter_iou = _query_iou(boxes[index], _boxes(intervention, counter_key, device))
                active_queries, counter_queries = active_iou >= 0.5, counter_iou >= 0.5
                has_oracle = bool(active_queries.any() and counter_queries.any())
                active_max = float(score[index][active_queries].max().item()) if active_queries.any() else float("-inf")
                counter_max = float(score[index][counter_queries].max().item()) if counter_queries.any() else float("-inf")
                records.append({
                    "schema": "arrow.stageb.admission_panel_arm/v1",
                    "row_id": row_id, "seed": seed,
                    "sample_id": meta["sample_id"], "pair_id": intervention["pair_id"],
                    "image_id": int(meta["image_id"]), "arm": arm,
                    "active_class_id": int(intervention["active_class_id"]),
                    "counterfactual_class_id": int(intervention["counterfactual_class_id"]),
                    "has_both_oracle_query_sets": has_oracle,
                    "active_max_admission_score": active_max,
                    "counterfactual_max_admission_score": counter_max,
                    "active_score_wins": bool(has_oracle and active_max > counter_max),
                    "active_eligible_recall50": bool((eligible[index] & active_queries).any()),
                    "counterfactual_eligible_leakage": bool((eligible[index] & counter_queries).any()),
                    "eligible_query_count": int(eligible[index].sum().item()),
                    "eligible_indices": eligible[index].nonzero().flatten().cpu().tolist(),
                    "admission_score_sha256": hashlib.sha256(
                        score[index].cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                })
    if max_batches <= 0 and len(records) != 1024:
        raise ArrowPanelError(f"panel emitted {len(records)} arms instead of 1024")
    if row_id == "AR_C_NULL":
        by_pair = {}
        for record in records:
            by_pair.setdefault(record["pair_id"], []).append(record)
        for pair_id, pair in by_pair.items():
            if len(pair) != 2 or pair[0]["admission_score_sha256"] != pair[1]["admission_score_sha256"] or pair[0]["eligible_indices"] != pair[1]["eligible_indices"]:
                raise ArrowPanelError(f"null route changed across pair {pair_id}")
    return records, asset_receipt


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {}
    for record in records:
        pairs.setdefault(str(record["pair_id"]), []).append(record)
    pair_success = []
    for pair in pairs.values():
        if len(pair) != 2:
            raise ArrowPanelError("panel pair is incomplete")
        pair_success.append(all(bool(record["active_score_wins"]) for record in pair))
    total = len(records)
    return {
        "pairs": len(pairs), "arms": total,
        "pair_switch_success": sum(pair_success) / len(pair_success),
        "active_eligible_recall50": sum(bool(r["active_eligible_recall50"]) for r in records) / total,
        "counterfactual_eligible_leakage": sum(bool(r["counterfactual_eligible_leakage"]) for r in records) / total,
        "mean_eligible_queries": sum(int(r["eligible_query_count"]) for r in records) / total,
        "both_oracle_rate": sum(bool(r["has_both_oracle_query_sets"]) for r in records) / total,
    }


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise ArrowPanelError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    if isinstance(value, list):
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in value), encoding="utf-8")
    else:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-id", choices=tuple(SOURCES), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs=3, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ArrowPanelError(f"output must be fresh: {output}")
    cfg = SLConfig.fromfile(args.config)
    cfg.device = args.device
    cfg.use_checkpoint = False
    cfg.use_transformer_ckpt = False
    all_results, inputs = {}, {}
    for seed, checkpoint in zip((17, 42, 73), args.checkpoints):
        records, asset = _evaluate_one(
            cfg=cfg, checkpoint=Path(checkpoint), row_id=args.row_id,
            device=torch.device(args.device), batch_size=args.batch_size,
            num_workers=args.num_workers, seed=seed,
            max_batches=args.max_batches,
        )
        record_path = output / f"seed{seed}.records.jsonl"
        _write(record_path, records)
        all_results[str(seed)] = _summary(records)
        inputs[str(seed)] = {"checkpoint": _record(Path(checkpoint)), "records": _record(record_path), "asset_rehash": asset}
    payload = {
        "schema": SCHEMA, "row_id": args.row_id, "source": SOURCES[args.row_id],
        "results": all_results, "inputs": inputs, "config": _record(Path(args.config)),
    }
    _write(output / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ArrowPanelError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
