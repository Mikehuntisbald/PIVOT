#!/usr/bin/env python3
"""Cache-only, frozen-head v6 scoring; no detector, optimizer or metrics.

Every evaluation model must have a complete locked endpoint. Train-statistics
mode scores only unique training positives with global-exists; evaluation mode
records the four matched cells and their off-diagonal readouts. A separate
record-only analyzer computes metrics only after all requested panels exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools.confidence_readout import (
    CACHE_SCHEMA, GLOBAL, LOCALIZERS, MDETR, MMGDINO, ROW_SCHEMA, SELECTED,
    SHARD_SCHEMA, load_readout_cache, make_readout_heads, native_selected_index,
    parse_arm, readout_scores,
)
from tools.confidence_readout_metrics import CELLS, add_combination_scores, fit_sirc_statistics
from tools.finecops_fixed_rank_targets import make_heads
from tools.responsibility_isolation_cache import normalized_cxcywh_iou
from tools.train_confidence_readout_heads import bind, canonical_sha, verify
from tools.train_finecops_bce_l2_heads import (
    _atomic_json, _deterministic_algorithms, _stack_rows, _tensor_state_sha256,
)

SEEDS = ("17", "42", "73")
CODE_FILES = (
    "tools/evaluate_confidence_readout_cache.py", "tools/confidence_readout.py",
    "tools/confidence_readout_metrics.py", "tools/finecops_fixed_rank_targets.py",
    "tools/train_confidence_readout_heads.py", "tools/train_finecops_bce_l2_heads.py",
    "tools/b32a1_heads.py", "tools/mmgdino_e5_ownership.py",
    "tools/responsibility_isolation_cache.py",
)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_json_binding(record):
    verify(record)
    return json.loads(Path(record["path"]).read_text())


def load_panel(panel_path, protocol_path, localizer, device):
    panel = json.loads(Path(panel_path).read_text())
    study = bind(protocol_path)
    protocol = json.loads(Path(protocol_path).read_text())
    if (panel.get("schema") != "arrow.confidence_readout.checkpoint_panel/v1"
            or panel.get("localizer") != localizer or set(panel.get("seeds", {})) != set(SEEDS)):
        raise ValueError("complete three-seed checkpoint panel required")
    if panel.get("study_protocol") != study:
        raise ValueError("checkpoint panel must bind the exact study protocol")
    models, legacy, checkpoints = {}, {}, {}
    for seed in SEEDS:
        entry = panel["seeds"][seed]
        post = load_json_binding(entry["postflight"])
        if (post.get("schema") != "arrow.confidence_readout.training_postflight/v1"
                or post.get("status") != "complete" or post.get("seed") != int(seed)
                or post.get("localizer") != localizer or post.get("updates_per_head") != 12575
                or post.get("checkpoint") != entry["readout"]):
            raise ValueError("new head endpoint is not sealed and complete")
        design = load_json_binding(post["design"])
        if design.get("study_protocol") != study:
            raise ValueError("head design belongs to another study")
        verify(entry["readout"])
        ck = torch.load(entry["readout"]["path"], map_location="cpu", weights_only=False)
        if (ck.get("schema") != "arrow.confidence_readout.head_checkpoint/v1"
                or ck.get("epoch") != 5 or ck.get("updates") != 12575
                or ck.get("seed") != int(seed) or ck.get("localizer") != localizer
                or ck.get("design") != post["design"]):
            raise ValueError("new head checkpoint identity drift")
        heads = make_readout_heads(int(seed), device, localizer)
        if set(heads) != set(ck.get("models", {})):
            raise ValueError("new checkpoint head slots drift")
        for arm, model in heads.items():
            model.load_state_dict(ck["models"][arm], strict=True)
            frozen_sha = _tensor_state_sha256(dict(model.named_task_parameters("rank")))
            confidence_sha = _tensor_state_sha256(dict(model.named_task_parameters("confidence")))
            audit = post["ownership"][arm]
            if (frozen_sha != audit["frozen_rank_sha256"]
                    or confidence_sha != audit["confidence_sha256"]):
                raise ValueError("endpoint ownership hashes differ from checkpoint")
        if localizer == MMGDINO:
            expected_old = protocol["localizers"][MMGDINO]["reused_global_heads"][seed]["checkpoint"]
            if entry["legacy_global"] != expected_old:
                raise ValueError("legacy global checkpoint differs from sealed study parent")
            verify(entry["legacy_global"])
            old = torch.load(entry["legacy_global"]["path"], map_location="cpu", weights_only=False)
            if (old.get("schema") != "arrow.fixed_rank_targets.checkpoint/v1"
                    or old.get("seed") != int(seed) or old.get("epoch") != 5 or old.get("updates") != 12575):
                raise ValueError("legacy global heads are not the sealed fixed-target endpoint")
            originals = make_heads(int(seed), device)
            for target, model in originals.items():
                model.load_state_dict(old["models"][target], strict=True)
                heads[f"{GLOBAL}__{target}"] = model
            if "legacy_records" in entry:
                raw = load_json_binding(entry["legacy_records"])
                legacy[seed] = {row["sample_id"]: row for row in raw}
                if len(legacy[seed]) != len(raw):
                    raise ValueError("duplicate identity in legacy records")
        if set(heads) != set(CELLS):
            raise ValueError("all four matched heads required")
        frozen = {arm: _tensor_state_sha256(dict(model.named_task_parameters("rank")))
                  for arm, model in heads.items()}
        if len(set(frozen.values())) != 1:
            raise ValueError("head panel uses differing frozen rank owners")
        for model in heads.values():
            model.eval().requires_grad_(False)
        models[seed] = heads
        checkpoints[seed] = {"localizer_checkpoint": design["checkpoint"],
                             "training_caches": design["caches"],
                             "models": {arm: _tensor_state_sha256(model.state_dict())
                                        for arm, model in heads.items()}}
    return panel, models, legacy, checkpoints


def load_evaluation_cache(path, localizer):
    """Training/val reuse the released loader; gRef has an explicit new split."""
    manifest = json.loads(Path(path).read_text())
    split = manifest.get("split")
    if split in ("train", "val"):
        return load_readout_cache(path, split=split, localizer=localizer)
    if (split != "gref_testab" or manifest.get("schema") != CACHE_SCHEMA
            or manifest.get("status") != "complete" or manifest.get("formal") is not True
            or manifest.get("localizer") != localizer):
        raise ValueError("only train/val or explicit single/no-target gRef cache allowed")
    rows = []
    for entry in manifest.get("shards", []):
        shard_path = Path(entry["path"])
        if not shard_path.is_absolute():
            shard_path = Path(path).parent / shard_path
        verify({"path": str(shard_path), "sha256": entry["sha256"]})
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if (shard.get("schema") != SHARD_SCHEMA or shard.get("split") != split
                or shard.get("start") != len(rows) or len(shard.get("rows", [])) != entry["rows"]):
            raise ValueError("gRef shard ordering/count drift")
        for row in shard["rows"]:
            if row.get("schema") != ROW_SCHEMA or row.get("kind") not in ("positive", "no_target"):
                raise ValueError("gRef row outside single/no-target scope")
            q = 100 if localizer == MDETR else 900
            if (row["query_features"].shape != (q, 256) or row["query_features"].dtype != torch.float16
                    or row["native_score"].shape != (q,) or row["native_score"].dtype != torch.float32
                    or row["boxes"].shape != (q, 4) or row["boxes"].dtype != torch.float32):
                raise ValueError("gRef query dtype/shape drift")
            rows.append(row)
    if len(rows) != manifest.get("records"):
        raise ValueError("gRef cache count drift")
    return rows, manifest


def verify_all_heads_sealed(binding, study_binding):
    receipt = load_json_binding(binding)
    if (receipt.get("schema") != "arrow.confidence_readout.all_heads_sealed/v1"
            or receipt.get("status") != "complete" or receipt.get("trajectories") != 18
            or receipt.get("study_protocol") != study_binding
            or set(receipt.get("postflights", {})) != set(LOCALIZERS)):
        raise ValueError("gRef requires the complete two-localizer eighteen-head endpoint gate")
    total = 0
    for localizer in LOCALIZERS:
        seeds = receipt["postflights"][localizer]
        if set(seeds) != set(SEEDS):
            raise ValueError("all-head seal is missing seeds")
        for seed in SEEDS:
            post = load_json_binding(seeds[seed])
            expected_arms = {f"{SELECTED}__{t}" for t in ("exists", "emit")} if localizer == MMGDINO else set(CELLS)
            if (post.get("status") != "complete" or post.get("updates_per_head") != 12575
                    or post.get("localizer") != localizer or post.get("seed") != int(seed)
                    or set(post.get("arms", [])) != expected_arms):
                raise ValueError("gRef endpoint gate has an incomplete arm")
            design = load_json_binding(post["design"])
            if design.get("study_protocol") != study_binding:
                raise ValueError("gRef endpoint gate belongs to a different study")
            verify(post["checkpoint"])
            total += len(expected_arms)
    if total != 18:
        raise ValueError("gRef endpoint gate trajectory count drift")


def evaluation_groups(rows, manifest, mode):
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source sample identity")
    if mode == "train_statistics":
        if manifest["split"] != "train":
            raise ValueError("combination statistics require training positives")
        chosen = [r for r in rows if r["kind"] == "positive"]
        return [chosen[start:start+32] for start in range(0, len(chosen), 32)]
    if manifest["split"] == "val":
        # Preserve old score(model,val) grouping exactly, including the final B23.
        return [rows[start:start+32] for start in range(0, len(rows), 32)]
    if manifest["split"] != "gref_testab":
        raise ValueError("training rows cannot be evaluated")
    declared = manifest.get("evaluation_groups")
    if not isinstance(declared, list) or not declared:
        raise ValueError("gRef cache needs explicit original head-batch grouping")
    expected = [ids[worker::4][start:start+32] for worker in range(4)
                for start in range(0, len(ids[worker::4]), 32)]
    if declared != expected:
        raise ValueError("gRef grouping must preserve original modulo-four/B32 sequence")
    lookup = dict(zip(ids, rows))
    return [[lookup[sid] for sid in group] for group in declared]


def source_metadata(rows, split):
    positive = {}
    if split in ("train", "val"):
        for row in rows:
            if row["kind"] == "positive":
                key = int(row["annotation_id"])
                if key in positive:
                    raise ValueError("positive annotation IDs are not unique")
                positive[key] = row
    result = {}
    for row in rows:
        sid, kind = row["sample_id"], row["kind"]
        if sid in result:
            raise ValueError("duplicate sample ID")
        cluster = str(row.get("cluster_image_id", row.get("image_id", "")))
        if not cluster:
            raise ValueError("missing image cluster")
        parent = None
        level = int(row["level"]) if kind == "positive" and row.get("level") is not None else None
        parent_level = None
        if split in ("train", "val") and kind == "text":
            original = positive.get(int(row["parent_positive_id"]))
            if original is None or str(original["cluster_image_id"]) != cluster:
                raise ValueError("negative lost its original same-image positive parent")
            parent = original["sample_id"]
            parent_level = int(original["level"])
        stratum = ("validation" if split == "val" else "train" if split == "train" else
                   row.get("stratum", row.get("source_split", row.get("split"))))
        if split == "gref_testab" and (stratum not in ("testA", "testB") or kind not in ("positive", "no_target")):
            raise ValueError("gRef stratum/cardinality drift")
        result[sid] = {"sample_id": sid, "cluster_id": cluster, "stratum": stratum,
            "split": split if split != "gref_testab" else stratum, "kind": kind, "level": level,
            "parent_positive_id": parent, "parent_positive_level": parent_level,
            "negative_edit_level": int(row["level"]) if kind == "text" and row.get("level") is not None else None}
        if split == "gref_testab":
            flag = row.get("finecops_train_val_source_disjoint")
            if type(flag) is not bool:
                raise ValueError("gRef source-disjoint identity missing")
            result[sid]["finecops_train_val_source_disjoint"] = flag
    return result


def row_geometry(row, index):
    boxes, mask = row["boxes"], row["candidate_mask"]
    if boxes.ndim != 2 or boxes.shape != (len(mask), 4) or not torch.isfinite(boxes).all():
        raise ValueError("finite Native geometry required")
    ious = None
    if row["kind"] == "no_target" and row.get("gt_boxes") is not None:
        gt = row["gt_boxes"]
        if not torch.is_tensor(gt) or gt.numel():
            raise ValueError("no-target must not contain fabricated GT")
    if row["kind"] == "positive":
        gt = row.get("gt_boxes")
        if not torch.is_tensor(gt) or gt.ndim != 2 or gt.shape[1] != 4 or not len(gt):
            raise ValueError("positive requires real nonempty GT")
        ious = normalized_cxcywh_iou(boxes, gt).amax(1)
        if not torch.isfinite(ious).all():
            raise ValueError("nonfinite ground-truth IoU")
    native_iou = None if ious is None else float(ious[index])
    native_score = float(row["native_score"][index])
    if not 0 <= native_score <= 1:
        raise ValueError("Native score must be a probability")
    return {"native_score": native_score, "native_selected_index": index,
            "native_box": boxes[index].tolist(), "native_gt_iou": native_iou,
            "correct": None if native_iou is None else native_iou >= .5,
            "boxes_sha256": tensor_sha(boxes), "candidate_mask_sha256": tensor_sha(mask)}, ious


def check_legacy_parity(record, reference, seed):
    if record["sample_id"] != reference["sample_id"] or record["correct"] != reference["correct"]:
        raise ValueError("legacy sample/correctness parity failed")
    if "scores" in reference:
        expected = reference["scores"][seed]
        comparisons = {"native_score": "native_score", "native_selected_index": "native_top1_query",
                       "native_box": "native_box", "native_gt_iou": "native_iou",
                       "boxes_sha256": "boxes_sha256", "candidate_mask_sha256": "candidate_mask_sha256"}
        for new, old in comparisons.items():
            if record[new] != reference[old]:
                raise ValueError("legacy gRef Native geometry parity failed: " + new)
    else:
        expected = {"exists": reference["baseline_score"], "emit": reference["candidate_score"]}
    for target in ("exists", "emit"):
        if record["scores"][f"{GLOBAL}__{target}"] != expected[target]:
            raise ValueError("legacy global head bitwise score parity failed: " + target)


def score_groups(models, groups, metadata, *, localizer, mode, device, legacy=None):
    output = {seed: [] for seed in SEEDS}
    with torch.inference_mode():
        for group_index, rows in enumerate(groups):
            features, native, mask = _stack_rows(rows, device)
            indices = [native_selected_index(row, localizer) for row in rows]
            selected = torch.tensor(indices, device=device, dtype=torch.int64)
            geometry = [row_geometry(row, index) for row, index in zip(rows, indices)] if mode == "evaluation" else None
            for seed in SEEDS:
                arms = (f"{GLOBAL}__exists",) if mode == "train_statistics" else CELLS
                values = {arm: readout_scores(models[seed][arm], features, native, mask, selected) for arm in arms}
                values = {arm: {key: tensor.detach().cpu() for key, tensor in value.items()
                                if key in (GLOBAL, SELECTED, "confidence_winner_index")}
                          for arm, value in values.items()}
                for offset, row in enumerate(rows):
                    record = dict(metadata[row["sample_id"]])
                    if mode == "train_statistics":
                        record["scores"] = {f"{GLOBAL}__exists": float(values[f"{GLOBAL}__exists"][GLOBAL][offset])}
                        record["native_score"] = float(native[offset, indices[offset]])
                    else:
                        native_geometry, ious = geometry[offset]
                        record.update(native_geometry)
                        record["scores"], record["readout_diagnostics"] = {}, {}
                        for arm in CELLS:
                            readout, _ = parse_arm(arm)
                            maximum = float(values[arm][GLOBAL][offset])
                            chosen = float(values[arm][SELECTED][offset])
                            winner = int(values[arm]["confidence_winner_index"][offset])
                            winner_native_iou = float(normalized_cxcywh_iou(
                                row["boxes"][winner:winner+1], row["boxes"][indices[offset]:indices[offset]+1])[0, 0])
                            record["scores"][arm] = maximum if readout == GLOBAL else chosen
                            record["readout_diagnostics"][arm] = {
                                "max_logit": maximum, "selected_logit": chosen,
                                "confidence_winner_index": winner, "native_selected_index": indices[offset],
                                "winner_native_box_iou": winner_native_iou,
                                "native_gt_iou": native_geometry["native_gt_iou"],
                                "winner_gt_iou": None if ious is None else float(ious[winner])}
                        if legacy is not None:
                            if row["sample_id"] not in legacy[seed]:
                                raise ValueError("legacy parity reference missing sample")
                            check_legacy_parity(record, legacy[seed][row["sample_id"]], seed)
                    output[seed].append(record)
            if group_index % 100 == 0:
                print(f"[READOUT-CACHE] {mode} batch={group_index+1}/{len(groups)}", flush=True)
    return output


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise ValueError("append-only record output exists")
    with temporary.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)
    return bind(path)


def run(args):
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("new evaluation directory must be empty")
    protocol = json.loads(args.study_protocol.read_text())
    if protocol.get("schema") != "arrow.confidence_readout.study_protocol/v1":
        raise ValueError("versioned study protocol required")
    for item in protocol["code"].values():
        verify(item)
    panel = json.loads(args.checkpoint_panel.read_text())
    cache_manifest = json.loads(args.cache.read_text())
    if args.mode == "evaluation" and args.training_statistics is None:
        raise ValueError("evaluation requires bound train-only combination statistics")
    if args.mode == "train_statistics" and args.training_statistics is not None:
        raise ValueError("training statistics cannot be overridden")
    if cache_manifest.get("split") == "gref_testab":
        verify_all_heads_sealed(cache_manifest["all_heads_sealed"], bind(args.study_protocol))
    bindings = {"study_protocol": bind(args.study_protocol), "cache": bind(args.cache),
                "checkpoint_panel": bind(args.checkpoint_panel)}
    if args.training_statistics is not None:
        bindings["training_statistics"] = bind(args.training_statistics)
    design = {"schema": "arrow.confidence_readout.evaluation_lock/v1", **bindings,
              "localizer": args.localizer, "mode": args.mode, "seeds": list(SEEDS),
              "head_batch": 32, "code": {name: bind(ROOT/name) for name in CODE_FILES},
              "detector_forwards": 0, "optimizer_updates": 0, "threshold_fitting": False}
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "evaluation_lock.json", design)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("deterministic workspace drift")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rows, manifest = load_evaluation_cache(args.cache, args.localizer)
    if manifest != cache_manifest:
        raise ValueError("cache manifest changed while loading")
    expected = {"train": {"positive": 83341, "text": 80451},
                "val": {"positive": 9426, "text": 9029},
                "gref_testab": {"positive": 11563, "no_target": 9121}}
    if dict(Counter(r["kind"] for r in rows)) != expected.get(manifest["split"]):
        raise ValueError("formal cache population drift")
    if manifest["split"] == "gref_testab":
        full_images = {str(r.get("cluster_image_id", r.get("image_id"))) for r in rows}
        disjoint = [r for r in rows if r.get("finecops_train_val_source_disjoint") is True]
        disjoint_images = {str(r.get("cluster_image_id", r.get("image_id"))) for r in disjoint}
        if (len(full_images) != 1500 or len(disjoint_images) != 1277
                or dict(Counter(r["kind"] for r in disjoint)) != {"positive": 9848, "no_target": 7716}):
            raise ValueError("gRef full/source-disjoint population drift")
    models_panel, models, legacy, checkpoints = load_panel(
        args.checkpoint_panel, args.study_protocol, args.localizer, device)
    if models_panel != panel:
        raise ValueError("checkpoint panel changed while loading")
    for seed in SEEDS:
        if manifest["model"]["checkpoint"] != checkpoints[seed]["localizer_checkpoint"]:
            raise ValueError("cache and head localizer checkpoints differ")
        if manifest["split"] in ("train", "val"):
            if checkpoints[seed]["training_caches"][manifest["split"]+"_cache"] != bind(args.cache):
                raise ValueError("scored FineCops cache differs from sealed head training population")
    if args.localizer == MMGDINO and args.mode == "evaluation":
        if set(legacy) != set(SEEDS) or any(set(v) != {r["sample_id"] for r in rows} for v in legacy.values()):
            raise ValueError("MM evaluation needs complete three-seed legacy records parity")
    groups = evaluation_groups(rows, manifest, args.mode)
    metadata = source_metadata(rows, manifest["split"])
    with _deterministic_algorithms():
        scored = score_groups(models, groups, metadata, localizer=args.localizer, mode=args.mode,
                              device=device, legacy=legacy if args.localizer == MMGDINO and args.mode == "evaluation" else None)
    for seed in SEEDS:
        scored[seed].sort(key=lambda row: row["sample_id"])
    record_bindings = {}
    head_identity = {seed: checkpoints[seed]["models"] for seed in SEEDS}
    if args.mode == "train_statistics":
        statistics = {seed: fit_sirc_statistics(scored[seed], expected_count=83341) for seed in SEEDS}
        statistic_bindings = {}
        for seed in SEEDS:
            record_bindings[seed] = write_jsonl(root/f"seed{seed}_train_positive_scores.jsonl", scored[seed])
            statpath = root/"statistics"/f"seed{seed}.json"
            _atomic_json(statpath, statistics[seed])
            statistic_bindings[seed] = bind(statpath)
        _atomic_json(root/"training_statistics.json", {
            "schema": "arrow.confidence_readout.training_statistics_panel/v1", "localizer": args.localizer,
            "study_protocol": bind(args.study_protocol), "checkpoint_panel": bind(args.checkpoint_panel),
            "head_identity": head_identity, "cache": bind(args.cache),
            "statistics": statistic_bindings, "records": record_bindings,
            "unique_train_positive_count": 83341, "evaluation_fitting": False})
        result_binding = bind(root/"training_statistics.json")
    else:
        stats = json.loads(args.training_statistics.read_text())
        if (stats.get("schema") != "arrow.confidence_readout.training_statistics_panel/v1"
                or stats.get("localizer") != args.localizer or stats.get("study_protocol") != bind(args.study_protocol)
                or stats.get("head_identity") != head_identity
                or stats.get("unique_train_positive_count") != 83341
                or set(stats.get("statistics", {})) != set(SEEDS)):
            raise ValueError("combination statistics identity drift")
        verify(stats["cache"])
        for seed in SEEDS:
            if stats["cache"] != checkpoints[seed]["training_caches"]["train_cache"]:
                raise ValueError("combination statistics were not fitted on the sealed training cache")
            verify(stats["records"][seed])
            stat = load_json_binding(stats["statistics"][seed])
            if stat.get("count") != 83341:
                raise ValueError("training statistics count drift")
            scored[seed] = add_combination_scores(scored[seed], stat)
            record_bindings[seed] = write_jsonl(root/f"seed{seed}_records.jsonl", scored[seed])
        result_binding = None
    for seed, heads in models.items():
        current = {arm: _tensor_state_sha256(model.state_dict()) for arm, model in heads.items()}
        if current != checkpoints[seed]["models"]:
            raise ValueError("frozen head tensor drift while evaluating")
    for entry in (*design["code"].values(), *bindings.values()):
        verify(entry)
    _atomic_json(root/"postflight.json", {"schema": "arrow.confidence_readout.cache_evaluation_postflight/v1",
        "status": "complete", "design": bind(root/"evaluation_lock.json"), "mode": args.mode,
        "localizer": args.localizer, "records": record_bindings, "training_statistics": result_binding,
        "rows_per_seed": len(scored[SEEDS[0]]), "head_batch_group_sha256": canonical_sha(
            [[r["sample_id"] for r in batch] for batch in groups]),
        "legacy_global_bitwise_parity": args.localizer == MMGDINO and args.mode == "evaluation",
        "detector_forwards": 0, "optimizer_updates": 0, "metrics_computed": False,
        "threshold_fitting": False, "native_boxes_invariant": True, "checkpoints": checkpoints})
    print("[READOUT-CACHE] COMPLETE no detector forward; no metrics", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--localizer", choices=LOCALIZERS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mode", choices=("train_statistics", "evaluation"), required=True)
    parser.add_argument("--checkpoint-panel", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
