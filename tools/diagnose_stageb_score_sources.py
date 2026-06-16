#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util import box_ops  # noqa: E402
from models.GroundingDINO.stage_b_score import aggregate_stage_b_tokens  # noqa: E402
from tools.eval_refcoco_stageb import (  # noqa: E402
    _build_split_jsonl,
    _build_loader,
    _default_splits,
    _forward,
    _load_canonical_name_maps,
    _load_model,
    _load_phrase_maps,
    _make_datasetinfo,
)
from tools.eval_stagea_patch_checkpoints import _set_seed  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _score_components(outputs: Dict[str, torch.Tensor], cfg, beta: float) -> Dict[str, torch.Tensor]:
    patch = outputs["pred_logits_patch"].detach().float()
    if patch.dim() == 2:
        patch = patch.unsqueeze(-1)

    text_logits = outputs["pred_logits_text"].detach().float()
    phrase = outputs["phrase_to_token_mask"].to(device=text_logits.device, dtype=torch.bool)
    B, _Q, K = patch.shape
    phrase = phrase[:, :K, :]
    canonical = outputs.get("canonical_to_token_mask", None)
    if canonical is None:
        canonical = torch.zeros_like(phrase)
    else:
        canonical = canonical.to(device=text_logits.device, dtype=torch.bool)[:, :K, :] & phrase
    attr = phrase & ~canonical

    text_attr = aggregate_stage_b_tokens(
        text_logits,
        attr,
        text_agg=str(getattr(cfg, "stage_b_infer_text_agg", "mean")),
        softmin_tau=float(getattr(cfg, "stage_b_infer_softmin_tau", 0.7)),
        mean_softmin_alpha=float(getattr(cfg, "stage_b_infer_mean_softmin_alpha", 0.5)),
    )
    text_canon = aggregate_stage_b_tokens(
        text_logits,
        canonical,
        text_agg=str(getattr(cfg, "stage_b_infer_text_agg", "mean")),
        softmin_tau=float(getattr(cfg, "stage_b_infer_softmin_tau", 0.7)),
        mean_softmin_alpha=float(getattr(cfg, "stage_b_infer_mean_softmin_alpha", 0.5)),
    )
    text = text_attr + float(getattr(cfg, "stage_b_infer_canonical_weight", 0.15)) * text_canon
    fused = patch.to(text.device) + float(beta) * text

    mask = outputs.get("patch_mask", outputs.get("patch_phrase_mask", None))
    if mask is not None:
        mask = mask.to(device=fused.device, dtype=torch.bool)
        if mask.shape[0] == B and mask.shape[1] >= K:
            invalid = ~mask[:, None, :K]
            patch = patch.masked_fill(invalid, -100.0)
            text = text.masked_fill(invalid, -100.0)
            fused = fused.masked_fill(invalid, -100.0)
    return {"patch": patch, "text": text, "fused": fused}


class Running:
    def __init__(self) -> None:
        self.values: List[float] = []

    def add(self, value: float) -> None:
        if math.isfinite(float(value)):
            self.values.append(float(value))

    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    def q(self, pct: float) -> float:
        return float(np.quantile(self.values, pct)) if self.values else 0.0


class SourceAccumulator:
    def __init__(self, topks: Iterable[int]) -> None:
        self.topks = sorted({max(1, int(k)) for k in topks})
        self.total = 0
        self.paths = ["patch", "text", "fused"]
        self.correct = {p: 0 for p in self.paths}
        self.iou_sum = {p: 0.0 for p in self.paths}
        self.recall = {(p, k): 0 for p in self.paths for k in self.topks}
        self.best_iou_sum = 0.0
        self.best_iou_correct = 0
        self.best_iou_rank = {p: Running() for p in self.paths}
        self.margin_top_minus_best = {(p, c): Running() for p in self.paths for c in self.paths}
        self.top_minus_best_iou = {p: Running() for p in self.paths}
        self.best_minus_top_iou = {p: Running() for p in self.paths}
        self.fused_error_source = {"patch_worse_than_text": 0, "text_worse_than_patch": 0, "both_wrong": 0}
        self.fused_failure_taxonomy = {
            "fused_miss_total": 0,
            "text_top1_correct_patch_suppressed": 0,
            "patch_top1_correct_text_suppressed": 0,
            "selected_near_miss_iou25_50": 0,
            "selected_bad_iou_lt25": 0,
            "oracle_query_available": 0,
            "oracle_query_text_rank_le5": 0,
            "oracle_query_patch_rank_le5": 0,
            "oracle_query_fused_rank_le5": 0,
            "oracle_query_text_rank_le10": 0,
            "oracle_query_patch_rank_le10": 0,
            "oracle_query_fused_rank_le10": 0,
        }
        self.fused_failure_margins = {
            "top_minus_oracle_patch": Running(),
            "top_minus_oracle_text": Running(),
            "top_minus_oracle_fused": Running(),
            "oracle_query_best_iou": Running(),
            "fused_top_iou": Running(),
            "oracle_query_text_rank": Running(),
            "oracle_query_patch_rank": Running(),
            "oracle_query_fused_rank": Running(),
        }

    def update(self, components: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> None:
        pred_boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
        B, Q, K = components["fused"].shape
        flat_scores = {p: components[p].reshape(B, Q * K) for p in self.paths}

        max_topk = min(max(self.topks), Q * K)
        top_idx = {p: torch.topk(flat_scores[p], k=max_topk, dim=1, largest=True).indices for p in self.paths}
        top_query = {p: torch.div(top_idx[p], K, rounding_mode="floor") for p in self.paths}
        top_slot = {p: top_idx[p] % K for p in self.paths}

        for b, target in enumerate(targets):
            gt_boxes = target["boxes"].detach().float()
            if gt_boxes.numel() == 0:
                continue
            self.total += 1
            gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1]).clamp(0.0, 1.0)[0]
            query_ious = box_ops.box_iou(pred_boxes[b], gt.view(1, 4))[0].view(-1)
            best_q = int(query_ious.argmax().item())
            best_iou = float(query_ious[best_q].item())
            self.best_iou_sum += best_iou
            if best_iou >= 0.5:
                self.best_iou_correct += 1

            for p in self.paths:
                q0 = int(top_query[p][b, 0].item())
                s0 = int(top_slot[p][b, 0].item())
                iou0 = float(query_ious[q0].item())
                self.iou_sum[p] += iou0
                if iou0 >= 0.5:
                    self.correct[p] += 1
                for k in self.topks:
                    qs = top_query[p][b, : min(k, max_topk)]
                    if bool((query_ious[qs] >= 0.5).any().item()):
                        self.recall[(p, k)] += 1

                # Use the best slot for each score component at the oracle best-IoU query.
                best_slots = {c: int(components[c][b, best_q].argmax().item()) for c in self.paths}
                score_top = {c: float(components[c][b, q0, s0].item()) for c in self.paths}
                score_best = {c: float(components[c][b, best_q, best_slots[c]].item()) for c in self.paths}
                for c in self.paths:
                    self.margin_top_minus_best[(p, c)].add(score_top[c] - score_best[c])
                rank = int((flat_scores[p][b] > flat_scores[p][b, best_q * K + best_slots[p]]).sum().item()) + 1
                self.best_iou_rank[p].add(rank)
                self.top_minus_best_iou[p].add(iou0 - best_iou)
                self.best_minus_top_iou[p].add(best_iou - iou0)

            fused_q = int(top_query["fused"][b, 0].item())
            fused_iou = float(query_ious[fused_q].item())
            fused_ok = fused_iou >= 0.5
            if not fused_ok:
                patch_q = int(top_query["patch"][b, 0].item())
                text_q = int(top_query["text"][b, 0].item())
                patch_ok = float(query_ious[patch_q].item()) >= 0.5
                text_ok = float(query_ious[text_q].item()) >= 0.5
                if text_ok and not patch_ok:
                    self.fused_error_source["patch_worse_than_text"] += 1
                elif patch_ok and not text_ok:
                    self.fused_error_source["text_worse_than_patch"] += 1
                else:
                    self.fused_error_source["both_wrong"] += 1

                self.fused_failure_taxonomy["fused_miss_total"] += 1
                if text_ok and not patch_ok:
                    self.fused_failure_taxonomy["text_top1_correct_patch_suppressed"] += 1
                if patch_ok and not text_ok:
                    self.fused_failure_taxonomy["patch_top1_correct_text_suppressed"] += 1
                if fused_iou >= 0.25:
                    self.fused_failure_taxonomy["selected_near_miss_iou25_50"] += 1
                else:
                    self.fused_failure_taxonomy["selected_bad_iou_lt25"] += 1
                if best_iou >= 0.5:
                    self.fused_failure_taxonomy["oracle_query_available"] += 1

                best_slots = {c: int(components[c][b, best_q].argmax().item()) for c in self.paths}
                ranks: Dict[str, int] = {}
                for c in self.paths:
                    ranks[c] = int((flat_scores[c][b] > flat_scores[c][b, best_q * K + best_slots[c]]).sum().item()) + 1
                    self.fused_failure_margins[f"oracle_query_{c}_rank"].add(ranks[c])
                    if ranks[c] <= 5:
                        self.fused_failure_taxonomy[f"oracle_query_{c}_rank_le5"] += 1
                    if ranks[c] <= 10:
                        self.fused_failure_taxonomy[f"oracle_query_{c}_rank_le10"] += 1

                fused_s = int(top_slot["fused"][b, 0].item())
                top_scores = {c: float(components[c][b, fused_q, fused_s].item()) for c in self.paths}
                oracle_scores = {c: float(components[c][b, best_q, best_slots[c]].item()) for c in self.paths}
                self.fused_failure_margins["top_minus_oracle_patch"].add(top_scores["patch"] - oracle_scores["patch"])
                self.fused_failure_margins["top_minus_oracle_text"].add(top_scores["text"] - oracle_scores["text"])
                self.fused_failure_margins["top_minus_oracle_fused"].add(top_scores["fused"] - oracle_scores["fused"])
                self.fused_failure_margins["oracle_query_best_iou"].add(best_iou)
                self.fused_failure_margins["fused_top_iou"].add(fused_iou)

    def result(self) -> Dict[str, Any]:
        denom = max(1, int(self.total))
        out: Dict[str, Any] = {
            "num_expressions": int(self.total),
            "oracle_best_query_acc50": float(self.best_iou_correct / denom),
            "oracle_best_query_mean_iou": float(self.best_iou_sum / denom),
            "paths": {},
            "fused_error_source_counts": dict(self.fused_error_source),
            "fused_failure_taxonomy": dict(self.fused_failure_taxonomy),
            "fused_failure_margins_mean": {k: v.mean() for k, v in self.fused_failure_margins.items()},
        }
        for p in self.paths:
            row: Dict[str, Any] = {
                "acc50": float(self.correct[p] / denom),
                "mean_iou_top1": float(self.iou_sum[p] / denom),
                "best_iou_query_rank_mean": self.best_iou_rank[p].mean(),
                "best_iou_query_rank_median": self.best_iou_rank[p].q(0.5),
                "top_iou_deficit_vs_oracle_mean": self.best_minus_top_iou[p].mean(),
            }
            for k in self.topks:
                row[f"recall50@{k}"] = float(self.recall[(p, k)] / denom)
            row["top_minus_oracle_best_score_margin"] = {
                c: self.margin_top_minus_best[(p, c)].mean() for c in self.paths
            }
            out["paths"][p] = row
        return out


def _mean_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(int(r.get("num_expressions", 0)) for r in rows)
    if total <= 0:
        return {}

    def wavg(key: str) -> float:
        return float(sum(_safe_float(r.get(key)) * int(r.get("num_expressions", 0)) for r in rows) / total)

    out: Dict[str, Any] = {
        "num_expressions": total,
        "oracle_best_query_acc50": wavg("oracle_best_query_acc50"),
        "oracle_best_query_mean_iou": wavg("oracle_best_query_mean_iou"),
        "paths": {},
    }
    taxonomy_keys = sorted({k for r in rows for k in (r.get("fused_failure_taxonomy") or {})})
    out["fused_failure_taxonomy"] = {
        k: int(sum(int((r.get("fused_failure_taxonomy") or {}).get(k, 0)) for r in rows)) for k in taxonomy_keys
    }
    margin_keys = sorted({k for r in rows for k in (r.get("fused_failure_margins_mean") or {})})
    miss_total = max(1, int(out["fused_failure_taxonomy"].get("fused_miss_total", 0)))
    out["fused_failure_margins_mean"] = {}
    for key in margin_keys:
        out["fused_failure_margins_mean"][key] = float(
            sum(
                _safe_float((r.get("fused_failure_margins_mean") or {}).get(key))
                * int((r.get("fused_failure_taxonomy") or {}).get("fused_miss_total", 0))
                for r in rows
            )
            / miss_total
        )
    for p in ["patch", "text", "fused"]:
        out["paths"][p] = {}
        keys = sorted({k for r in rows for k in (r.get("paths", {}).get(p, {}) or {}).keys() if isinstance((r.get("paths", {}).get(p, {}) or {}).get(k), (int, float))})
        for key in keys:
            out["paths"][p][key] = float(
                sum(
                    _safe_float((r.get("paths", {}).get(p, {}) or {}).get(key)) * int(r.get("num_expressions", 0))
                    for r in rows
                )
                / total
            )
        out["paths"][p]["top_minus_oracle_best_score_margin"] = {}
        for c in ["patch", "text", "fused"]:
            out["paths"][p]["top_minus_oracle_best_score_margin"][c] = float(
                sum(
                    _safe_float(
                        ((r.get("paths", {}).get(p, {}) or {}).get("top_minus_oracle_best_score_margin", {}) or {}).get(c)
                    )
                    * int(r.get("num_expressions", 0))
                    for r in rows
                )
                / total
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/cfg_patch_stage_b.py")
    ap.add_argument("--datasets", default="config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json")
    ap.add_argument("--checkpoint", default="outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth")
    ap.add_argument("--output_dir", default="outputs/stageb_score_source_diagnostics_v2_ckpt0003")
    ap.add_argument("--splits", nargs="*", default=["refcoco_val", "refcocop_val", "refcocog_val"])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--topks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.data_root is None:
        import os

        data_root = Path(os.environ.get("DATA_ROOT", "/media/haoyi/T9/datasets"))
    else:
        data_root = Path(args.data_root)

    cfg = SLConfig.fromfile(args.config)
    cfg.output_dir = str(output_dir)
    cfg.batch_size = int(args.batch_size)
    cfg.num_workers = int(args.num_workers)
    cfg.use_coco_eval = False
    cfg.patch_only = True
    cfg.build_text_token_masks = True
    cfg.fix_size = True

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    cfg.device = str(device)
    _set_seed(args.seed)
    model = _load_model(cfg, args.checkpoint, device)

    canonical_path = data_root / "canonical_classes_with_aliases.json"
    name_to_id, id_to_name = _load_canonical_name_maps(canonical_path)
    phrase_maps = _load_phrase_maps(
        [
            data_root / "refcoco_text_pairs" / "refcoco_unc_pairs.jsonl",
            data_root / "data_proc" / "refcoco_text_pairs" / "refcoco+_unc_pairs.jsonl",
            data_root / "data_proc" / "refcoco_text_pairs" / "refcocog_google_pairs.jsonl",
            data_root / "patch_episode_prebuilt" / "refcocoplus_stageb_phrase_v1.jsonl",
            data_root / "patch_episode_prebuilt" / "refcocog_stageb_phrase_v1.jsonl",
        ]
    )

    split_defs = {row["name"]: row for row in _default_splits()}
    rows: List[Dict[str, Any]] = []
    for split_i, split_name in enumerate(args.splits):
        split_def = split_defs[split_name]
        jsonl, count = _build_split_jsonl(
            data_root=data_root,
            output_dir=output_dir,
            dataset=split_def["dataset"],
            splitby=split_def["splitby"],
            split=split_def["split"],
            phrase_sources=split_def["sources"],
            phrase_maps=phrase_maps,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
        )
        datasetinfo = _make_datasetinfo(data_root, split_name, jsonl)
        loader = _build_loader(
            cfg,
            datasetinfo=datasetinfo,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            seed=int(args.seed) + split_i * 100000,
        )
        acc = SourceAccumulator(topks=args.topks)
        _set_seed(int(args.seed) + split_i * 100000)
        for batch_i, batch in enumerate(loader):
            if args.max_batches and batch_i >= int(args.max_batches):
                break
            outputs, targets = _forward(model, batch, device, amp=bool(args.amp))
            components = _score_components(outputs, cfg, beta=float(args.beta))
            acc.update(components, outputs, targets)
        row = acc.result()
        row.update({"dataset": split_name, "checkpoint": args.checkpoint, "beta": float(args.beta), "input_rows": int(count)})
        rows.append(row)
        (output_dir / f"{split_name}.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"[RESULT] {split_name}: "
            f"patch={row['paths']['patch']['acc50']:.6f} "
            f"text={row['paths']['text']['acc50']:.6f} "
            f"fused={row['paths']['fused']['acc50']:.6f} "
            f"oracle={row['oracle_best_query_acc50']:.6f}",
            flush=True,
        )

    summary = {"rows": rows, "weighted_mean": _mean_rows(rows)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    mean = summary["weighted_mean"]
    lines = [
        "# Stage-B Score Source Diagnostics",
        "",
        f"checkpoint: `{args.checkpoint}`",
        f"beta: `{float(args.beta)}`",
        "",
        "| split | patch acc50 | text acc50 | fused acc50 | fused recall50@5 | fused recall50@10 | oracle query acc50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | "
            f"{row['paths']['patch']['acc50']:.6f} | "
            f"{row['paths']['text']['acc50']:.6f} | "
            f"{row['paths']['fused']['acc50']:.6f} | "
            f"{row['paths']['fused']['recall50@5']:.6f} | "
            f"{row['paths']['fused']['recall50@10']:.6f} | "
            f"{row['oracle_best_query_acc50']:.6f} |"
        )
    lines.append(
        f"| weighted_mean | "
        f"{mean['paths']['patch']['acc50']:.6f} | "
        f"{mean['paths']['text']['acc50']:.6f} | "
        f"{mean['paths']['fused']['acc50']:.6f} | "
        f"{mean['paths']['fused']['recall50@5']:.6f} | "
        f"{mean['paths']['fused']['recall50@10']:.6f} | "
        f"{mean['oracle_best_query_acc50']:.6f} |"
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
