#!/usr/bin/env python3
"""Freeze the v6 scientific matrix before new head training, not after results.

MDETR's official MD5/revision are fixed here; its downloaded SHA and caches are
bound in separate append-only preparation and per-seed training receipts.
Analysis/evaluator implementation receives an additional lock before scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/arrow_confidence_readout_v6_20260905"
PARENT = ROOT / "outputs/b32a1_finecops_positive_trunk_factorized_20260904"
OLD = ROOT / "outputs/arrow_finecops_fixed_rank_targets_20260905"
CODE = (
    "tools/confidence_readout.py", "tools/train_confidence_readout_heads.py",
    "tools/finecops_fixed_rank_targets.py", "tools/train_finecops_bce_l2_heads.py",
    "tools/b32a1_heads.py", "tools/mmgdino_e5_ownership.py",
    "tools/responsibility_isolation_cache.py", "tools/b32a1_objectives.py",
    "tools/b32a1_metrics.py", "tools/finecops_bce_l2_control.py",
    "tools/lock_confidence_readout_study.py", "tests/test_confidence_readout.py",
)


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"historical SHA drift: {path}")
    return {"path": str(path), "sha256": digest}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build():
    if any((STUDY / name).exists() for name in ("heads", "evaluation", "analysis")):
        raise ValueError("new head/result directories exist before study lock")
    train = bind(PARENT / "formal_cache/train/manifest.json", "23c1e61a33659fddbc153df1f7a13650d110df8c3caa3a17a0f5c2ab238120f0")
    val = bind(PARENT / "formal_cache/val/manifest.json", "c3d30772ed048730b228a35e217e128c17bc3935e2bccb274ec69cbd4ddec085")
    tm, vm = [json.loads(Path(v["path"]).read_text()) for v in (train, val)]
    for manifest in (tm, vm):
        for key in ("checkpoint", "config"):
            bind(**{"path": manifest["model"][key]["path"], "expected": manifest["model"][key]["sha256"]})
    if tm["model"]["checkpoint"] != vm["model"]["checkpoint"]:
        raise ValueError("cache parent mismatch")
    final = bind(OLD / "final_receipt.json", "ec3a1ab3c2fb21462eee8b496f4edf86a59e3e9ab413d07dc5a2bb84b5175e8c")
    previous = json.loads(Path(final["path"]).read_text())
    for seed in ("17", "42", "73"):
        for key in ("checkpoint", "records"):
            record = previous["checkpoints"][seed][key]
            bind(record["path"], record["sha256"])
    design = json.loads((OLD / "design_lock.json").read_text())
    for record in design["parent_test_state"].values():
        bind(record["path"], record["sha256"])
    return {
        "schema": "arrow.confidence_readout.study_protocol/v1",
        "question": "Why is supervising correct output insufficient, and what role does query readout play?",
        "status_at_lock": "mechanism_strengthening_in_progress_no_new_results",
        "study_scope": "posthoc-motivated, prospectively frozen mechanism study; not virgin confirmation",
        "seeds": [17, 42, 73], "new_head_count": 18, "trunk_updates": 0,
        "training": {"epochs": 5, "updates_per_head": 12575, "pair_batch": 32,
            "lr": 1e-4, "wd": 0, "clip": .1, "logit_l2": .001,
            "dtype": "deterministic_fp32", "optimizer": "AdamW foreach=False",
            "trainable_tensors": 8, "trainable_parameters": 50179,
            "architecture": "LayerNorm257 -> 128 -> 128 -> 1; GELU; zero output initialization",
            "initialization": "original make_heads factory including frozen rank initialization; no warm start",
            "event_generator": "original _epoch_schedule; generate rank and confidence then skip rank events",
            "last_batch": {"full_pair_batches": 2514, "last_pairs": 3},
            "source_weights": [.5, .5], "emit_class_rebalancing": False},
        "localizers": {
            "mmgdino_positive": {"train_cache": train, "val_cache": val,
                "checkpoint": tm["model"]["checkpoint"], "runtime": tm["model"],
                "new_arms": ["native_selected__exists", "native_selected__emit"],
                "reused_global_heads": previous["checkpoints"], "reused_final": final,
                "queries": 900, "selector": "native masked argmax; first maximum"},
            "mdetr_r101_refcoco_ema": {"cache_status": "pending_before_head_training",
                "upstream_commit": "ea09acc44ca067072c4b143b726447ee7ff66f5f",
                "checkpoint_url": "https://zenodo.org/records/4721981/files/refcoco_resnet101_checkpoint.pth",
                "checkpoint_md5": "3219e03af7709cd15ab0d0db521b9070", "required_weight_key": "model_ema",
                "raw_weight_fallback": False, "queries": 100,
                "native_score": "1 - softmax(pred_logits)[no_object]",
                "selector": "official descending tuple(native_score, pixel_xyxy); original index for duplicate ties",
                "correctness": "IoU >= 0.5 with own Native box, not upstream GIoU",
                "new_arms": ["global_max__exists", "global_max__emit", "native_selected__exists", "native_selected__emit"]}},
        "readout": {"global_max": "max valid dense head logits",
            "native_selected": "gather fixed Native-selected query after same dense head computation, in train and eval",
            "off_diagonal": "fixed-weight diagnostic only; path dependent, no unique spatial attribution",
            "features": "FP16 cache, FP32 head", "scores_boxes": "FP32", "padding": False,
            "no_target_gt": None, "empty_candidate_mask": "reject"},
        "diagnostics": ["native_selected_index", "confidence_winner_index", "max_logit", "selected_logit",
            "winner_native_box_iou", "winner_gt_iou", "native_gt_iou", "C/W/N winner disagreement"],
        "effects": {"primary": "observed-mixture mixed AUGRC",
            "D_emit": "R(S,emit)-R(G,emit)", "D_exists": "R(S,exists)-R(G,exists)",
            "interaction": "D_emit-D_exists", "report": "four raw cells, each seed, mean, sample SD, effect and paired interval",
            "interpretation": "effect magnitudes and precision; negative interaction alone does not establish emit repair"},
        "three_state": {"pairs": ["C-W", "C-N", "W-N"],
            "existence_identity": "AUC_E=a*U_CN+(1-a)*U_WN",
            "risk_identity": "DeltaAUGRC=-(1-pi)*a*((1-pi)*(1-a)*DeltaU_CW+pi*DeltaU_CN)",
            "crossover": "report internal root with uncertainty and missing-root frequencies; not an AURC formula",
            "conditionals": "FineCops actual edited-negative parent pairs; gRef same-image only; comparable unconditional surface, attenuation, CI, counts",
            "difficulty": "positive or linked parent-positive difficulty, never substitute negative edit level",
            "null_interval_rule": "zero-crossing or wide CI means unresolved, not primarily image/difficulty-level causation"},
        "combinations": {"exists_source": "same-seed global_max__exists", "native_boxes_unchanged": True,
            "product": "log(max(s0,1e-6))+logsigmoid(z)",
            "sirc_style": "-log(max(1-s0,1e-6))-softplus(-(z-(mu-3*sigma))/sigma)",
            "fit": "all 83341 unique TRAIN positives per localizer/seed; population SD",
            "sigma_floor": 1e-12, "degenerate_fallback": "Native, with receipt",
            "contrasts": ["Native", "global_max__exists", "global_max__emit"],
            "interpretation": "success is setting/metric specific; failure of two fixed rules does not imply non-composability"},
        "evaluation": {"surfaces": ["finecops_val", "gref_full", "gref_finecops_train_val_source_disjoint"],
            "finecops_val": {"positive": 9426, "text": 9029},
            "gref_full": {"positive": 11563, "no_target": 9121, "images": 1500},
            "gref_source_disjoint": {"positive": 9848, "no_target": 7716, "images": 1277},
            "gref_data": bind(ROOT / "data/gref_fixed_targets_v1/manifests/records.jsonl", "e7328bdcf0d3c36c5c5d8e010dfcc6bbe0a532108e70d2fc84cc1732a80babbf"),
            "gref_audit": bind(ROOT / "data/gref_fixed_targets_v1/manifests/audit.json", "d1766584db2b8a6436f8c02b068758ae309544132644ee940742c6389da38260"),
            "gref_requires_all_18_heads_sealed": True, "localizer_traversals": "one per localizer, cached for all heads",
            "mm_old_six_heads_and_native_bitwise_parity": True,
            "finecops_test_read": False, "negative_image": False, "multi_target": False,
            "checkpoint_selection": False, "threshold_fitting": False,
            "secondary": ["AURC", "existence AUROC", "FPR95 diagnostic", "fixed-coverage error composition", "Native P@1"]},
        "bootstrap": {"iterations": 5000, "seed": 20260911, "rng": "PCG64",
            "unit": "paired image cluster", "gref_strata": "TestA/TestB",
            "shared_draws": "all localizers, targets, readouts, combinations, and three seeds within surface",
            "aggregation": "recompute each seed metric then equal-weight mean",
            "fpr95_threshold": "recompute positive q05 inside every replicate",
            "interval": "exploratory paired 95 percent"},
        "claims": {"spatial_correspondence": "hypothesis to test, not established cause",
            "G_to_S_intervention": ["scored query", "query competition", "training gradient location"],
            "second_model": "within-model target/readout effect; lineage disclosed, not pure architecture comparison",
            "acceptance": "all heads/seeds/evaluations complete regardless of winners",
            "paper_order": ["question", "target-by-readout", "three-state explanation", "second-model scope", "combination implications"]},
        "finecops_test_state": design["parent_test_state"],
        "code": {name: bind(ROOT / name) for name in CODE},
        "analysis_implementation": "separate append-only hash lock required before any new model scoring",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "protocol.json")
    args = parser.parse_args()
    write_new(args.output, build())
    print(json.dumps(bind(args.output), indent=2))
