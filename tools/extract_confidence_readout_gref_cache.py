#!/usr/bin/env python3
"""All-query gRefCOCO transfer caches after all eighteen new heads are sealed.

No head is loaded, no metric is aggregated and no optimizer is constructed.
Four independent B1 detector workers own contiguous cache shards.  A separate
evaluation_groups field preserves the OLD four-strided/B32 head-scoring order.
Use the independent MDETR environment for MDETR and the sealed MM environment
for MM-GDINO.  Never import both model runtimes into one process.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MM, MD = "mmgdino_positive", "mdetr_r101_refcoco_ema"
LOCALIZERS, SEEDS = (MM, MD), ("17", "42", "73")
CACHE_SCHEMA = "arrow.confidence_readout.cache_manifest/v1"
ROW_SCHEMA = "arrow.confidence_readout.cache_row/v1"
SHARD_SCHEMA = "arrow.confidence_readout.cache_shard/v1"
WORKER_SCHEMA = "arrow.confidence_readout.gref_cache_worker/v1"
SPLIT = "gref_testab"
EXPECTED = {"records": 20684, "images": 1500, "positive": 11563, "no_target": 9121}
EXPECTED_DISJOINT = {"records": 17564, "images": 1277, "positive": 9848, "no_target": 7716}
EXPECTED_SPLITS = {("testA", "positive"): 5917, ("testA", "no_target"): 4448,
                   ("testB", "positive"): 5646, ("testB", "no_target"): 4673}
LEGACY_SHA = "b0eed5b50e71665929d4c5eed81733f61473e752fafb6b74b500179c243bac0c"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def verify(record, override=None):
    path = Path(record["path"] if override is None else override).resolve(strict=True)
    if digest(path) != record["sha256"]:
        raise ValueError("bound file SHA drift: " + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("append-only artifact exists: " + str(path))
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.rename(path)


def require_all_heads(protocol_path, gate_path):
    """Verify every postflight and its endpoint/initial/design hashes, not a flag."""
    from tools.seal_confidence_readout_heads import check_postflight
    protocol_binding = bind(protocol_path)
    gate = read(gate_path)
    if (gate.get("schema") != "arrow.confidence_readout.all_heads_sealed/v1"
            or gate.get("status") != "complete" or gate.get("trajectories") != 18
            or gate.get("study_protocol") != protocol_binding or gate.get("metric_selection") is not False
            or set(gate.get("postflights", {})) != set(LOCALIZERS)):
        raise ValueError("complete all-eighteen-head seal required before gRef")
    verify(gate["sealer"])
    count = 0
    for localizer in LOCALIZERS:
        records = gate["postflights"][localizer]
        if set(records) != set(SEEDS):
            raise ValueError("all three seed postflights are mandatory")
        for seed, record in records.items():
            post = check_postflight(verify(record), protocol_binding, localizer, seed)
            count += len(post["arms"])
    if count != 18:
        raise ValueError("new-head count differs from eighteen")
    return bind(gate_path)


def population(rows):
    return {"records": len(rows), "images": len({r["image_id"] for r in rows}),
            "positive": sum(r["kind"] == "positive" for r in rows),
            "no_target": sum(r["kind"] == "no_target" for r in rows)}


def validate_source_rows(rows, images):
    if population(rows) != EXPECTED or len(images) != EXPECTED["images"]:
        raise ValueError("gRef Full population drift")
    if dict(Counter((r["split"], r["kind"]) for r in rows)) != EXPECTED_SPLITS:
        raise ValueError("gRef TestA/TestB single/no-target split drift")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("sealed source order and unique sample identities required")
    for row in rows:
        if row["split"] not in ("testA", "testB") or row["kind"] not in ("positive", "no_target"):
            raise ValueError("train/val/multi-target rows are excluded")
        if not row["sample_id"].startswith("grefcoco:" + row["split"] + ":"):
            raise ValueError("sample ID split mismatch")
        if not isinstance(row.get("expression"), str) or not row["expression"].strip():
            raise ValueError("full expression is required")
        if type(row.get("finecops_train_val_source_disjoint")) is not bool:
            raise ValueError("source disjoint label must be explicit bool")
        item = images.get(str(row["image_id"]))
        if item is None or any(row[k] != item[k] for k in ("width", "height")):
            raise ValueError("image identity/dimensions drift")
        if row["image_path"] != item["remote_path"] or row["image_sha256"] != item["sha256"]:
            raise ValueError("transferred image binding drift")
        if (row["finecops_train_val_source_disjoint"]
                != (not (item["overlap"]["train_all"] or item["overlap"]["val_all"]))):
            raise ValueError("source-disjoint membership drift")
        if any(type(row[k]) is not int or row[k] < 1 for k in ("width", "height")):
            raise ValueError("positive integer image dimensions required")
        box = row.get("bbox_xywh")
        if row["kind"] == "no_target":
            if box is not None:
                raise ValueError("no-target cannot carry an invented box")
        else:
            if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
                raise ValueError("single target requires finite xywh box")
            x, y, w, h = box
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > row["width"] + 1e-3 or y + h > row["height"] + 1e-3:
                raise ValueError("GT box outside image")
    if population([r for r in rows if r["finecops_train_val_source_disjoint"]]) != EXPECTED_DISJOINT:
        raise ValueError("source-disjoint population drift")


def source_inputs(protocol, images_path, verify_images=True):
    evaluation = protocol["evaluation"]
    manifest = verify(evaluation["gref_data"])
    audit_path = verify(evaluation["gref_audit"])
    audit = read(audit_path)
    if (audit.get("schema") != "arrow.fixed_targets.gref_preparation/v1"
            or audit.get("excluded_multi_target_expressions") != 14579
            or audit["manifest"]["sha256"] != digest(manifest)):
        raise ValueError("gRef official identity/exclusion audit drift")
    images_path = verify(audit["images"], images_path)
    images = read(images_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    validate_source_rows(rows, images)
    if verify_images:
        from PIL import Image
        for item in images.values():
            path = verify({"path": item["remote_path"], "sha256": item["sha256"]})
            if path.stat().st_size != item["size"]:
                raise ValueError("image byte count drift")
            with Image.open(path) as image:
                if image.size != (item["width"], item["height"]):
                    raise ValueError("image dimensions drift")
                image.verify()
    return rows, {"source": bind(manifest), "audit": bind(audit_path), "images": bind(images_path)}


def legacy_records(path, rows):
    if digest(path) != LEGACY_SHA:
        raise ValueError("sealed old gRef records SHA drift")
    records = read(path)
    if [r["sample_id"] for r in records] != [r["sample_id"] for r in rows]:
        raise ValueError("legacy record/source order mismatch")
    for old, source in zip(records, rows):
        for key in ("sample_id", "split", "image_id", "kind", "finecops_train_val_source_disjoint"):
            if old[key] != source[key]:
                raise ValueError("legacy record metadata differs from source")
    return {r["sample_id"]: r for r in records}


def evaluation_groups(rows):
    """OLD four-worker striding, then B32; deliberately NOT cache shard order."""
    ids = [r["sample_id"] for r in rows]
    groups = []
    for worker in range(4):
        assigned = ids[worker::4]
        groups.extend(assigned[start:start + 32] for start in range(0, len(assigned), 32))
    if len({sid for group in groups for sid in group}) != len(ids) or sum(map(len, groups)) != len(ids):
        raise ValueError("evaluation groups omit or duplicate a source sample")
    return groups


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_row(source, hook, localizer):
    import torch
    if localizer not in LOCALIZERS:
        raise ValueError("unknown frozen localizer")
    q = 900 if localizer == MM else 100
    fields = {k: getattr(hook, k) for k in ("query_features", "native_score", "boxes", "candidate_mask")}
    if (fields["query_features"].shape != (q, 256) or fields["query_features"].dtype != torch.float16
            or fields["native_score"].shape != (q,) or fields["native_score"].dtype != torch.float32
            or fields["boxes"].shape != (q, 4) or fields["boxes"].dtype != torch.float32
            or fields["candidate_mask"].shape != (q,) or fields["candidate_mask"].dtype != torch.bool
            or not fields["candidate_mask"].any()):
        raise ValueError("frozen all-query shape/dtype/mask drift")
    if any(t.device.type != "cpu" or t.requires_grad or (t.is_floating_point() and not torch.isfinite(t).all()) for t in fields.values()):
        raise ValueError("cache tensors must be detached finite CPU tensors")
    if (fields["native_score"] < 0).any() or (fields["native_score"] > 1).any():
        raise ValueError("Native scores must be probabilities")
    index = int(fields["native_score"].masked_fill(~fields["candidate_mask"], -torch.inf).argmax()) if localizer == MM else hook.native_selected_index
    if type(index) is not int or not 0 <= index < q or not fields["candidate_mask"][index]:
        raise ValueError("Native selected query must be valid")
    if source["kind"] == "positive":
        x, y, w, h = source["bbox_xywh"]
        gt = torch.tensor([[(x + w/2)/source["width"], (y + h/2)/source["height"],
                            w/source["width"], h/source["height"]]], dtype=torch.float32)
    elif source["kind"] == "no_target" and source["bbox_xywh"] is None:
        gt = torch.empty((0, 4), dtype=torch.float32)
    else:
        raise ValueError("single/no-target geometry contract violated")
    result = {"schema": ROW_SCHEMA, "localizer": localizer, "split": SPLIT,
        "source_split": source["split"], "stratum": source["split"], "sample_id": source["sample_id"],
        "annotation_id": source["sample_id"], "image_path": source["image_path"],
        "image_sha256": source["image_sha256"], "image_size": [source["height"], source["width"]],
        "source_image_id": str(source["image_id"]), "cluster_image_id": str(source["image_id"]),
        "image_id": source["image_id"], "caption": source["expression"], "kind": source["kind"],
        "parent_positive_id": None, "level": None, "negative_type": None, "negative_level": None,
        "finecops_train_val_source_disjoint": source["finecops_train_val_source_disjoint"],
        "gt_boxes": gt, "native_selected_index": index, **fields}
    if localizer == MD:
        if tuple(hook.image_size) != (source["height"], source["width"]):
            raise ValueError("MDETR hook image dimensions drift")
        pixel = hook.native_boxes_xyxy_abs
        if (pixel.shape != (q, 4) or pixel.dtype != torch.float32 or pixel.device.type != "cpu"
                or pixel.requires_grad or not torch.isfinite(pixel).all()):
            raise ValueError("MDETR official pixel boxes must be detached finite FP32")
        cx, cy, w, h = fields["boxes"].unbind(-1)
        expected_pixel = torch.stack((cx - .5*w, cy - .5*h, cx + .5*w, cy + .5*h), -1) * torch.tensor(
            [source["width"], source["height"], source["width"], source["height"]], dtype=torch.float32)
        if not torch.equal(pixel, expected_pixel) or not fields["candidate_mask"].all():
            raise ValueError("MDETR full-query pixel geometry/mask differs from official output")
        official = max(torch.where(fields["candidate_mask"])[0].tolist(),
                       key=lambda j: (float(fields["native_score"][j]), pixel[j].tolist()))
        if official != index:
            raise ValueError("MDETR Native selector differs from official score/box tie ordering")
        result["native_boxes_xyxy_abs"] = pixel
    return result


def check_mm_legacy(row, legacy):
    from tools.responsibility_isolation_cache import normalized_cxcywh_iou
    import torch
    if row["localizer"] != MM:
        raise ValueError("legacy MM parity cannot certify MDETR")
    index = row["native_selected_index"]
    iou = None if row["kind"] == "no_target" else float(normalized_cxcywh_iou(row["boxes"][index:index+1], row["gt_boxes"])[0, 0])
    checks = {
        "native_top1_query": index, "native_score": float(row["native_score"][index]),
        "native_box": row["boxes"][index].tolist(), "native_iou": iou,
        "correct": None if iou is None else iou >= .5, "boxes_sha256": tensor_sha(row["boxes"]),
        "candidate_mask_sha256": tensor_sha(row["candidate_mask"]),
    }
    if any(legacy.get(key) != value for key, value in checks.items()):
        failed = [k for k, v in checks.items() if legacy.get(k) != v]
        raise ValueError("sealed MM per-record parity failed: " + ",".join(failed))
    if not torch.equal(row["boxes"][index], torch.tensor(legacy["native_box"], dtype=torch.float32)):
        raise ValueError("Native box bitwise parity failed")


def context(args):
    protocol_binding = bind(args.protocol)
    if protocol_binding["sha256"] != args.protocol_sha256:
        raise ValueError("protocol SHA drift")
    protocol = read(args.protocol)
    if protocol.get("schema") != "arrow.confidence_readout.study_protocol/v1" or protocol.get("seeds") != [17, 42, 73]:
        raise ValueError("fixed readout study protocol required")
    gate = require_all_heads(args.protocol, args.all_heads_sealed)
    rows, source = source_inputs(protocol, args.images_manifest)
    binding = {"protocol": protocol_binding, "all_heads_sealed": gate, **source,
               "localizer": args.localizer, "split": SPLIT, "shard_size": 128,
               "worker_count": 4, "trunk_batch": 1, "head_evaluation_grouping": "legacy_four_strided_B32",
               "extractor": bind(__file__), "val_cache": bind(args.val_cache)}
    vm = read(args.val_cache)
    if vm.get("status") != "complete" or vm.get("formal") is not True or vm.get("split") != "val":
        raise ValueError("sealed non-gRef val cache required for runtime preflight")
    old = None
    if args.localizer == MM:
        mm = protocol["localizers"][MM]
        if binding["val_cache"] != mm["val_cache"] or vm["model"] != mm["runtime"]:
            raise ValueError("MM runtime/cache differs from original positive trunk")
        for key in ("checkpoint", "config"):
            verify(mm["runtime"][key])
        mmdet = mm["runtime"]["mmdetection"]
        commit = subprocess.check_output(["git", "-C", mmdet["path"], "rev-parse", "HEAD"], text=True).strip()
        if commit != mmdet["commit"]:
            raise ValueError("MMDetection commit drift")
        old = legacy_records(args.legacy_records, rows)
        binding["legacy_records"] = bind(args.legacy_records)
        binding["runtime_code"] = bind(ROOT / "tools/extract_mmgdino_responsibility_cache.py")
    elif args.localizer == MD:
        preparation = read(args.mdetr_preparation)
        if preparation.get("schema") != "arrow.confidence_readout.mdetr_preparation/v1" or preparation.get("study_protocol") != protocol_binding:
            raise ValueError("MDETR preparation study drift")
        smoke = read(verify(preparation["smoke"]))
        if (smoke.get("status") != "passed" or preparation["runtime"] != smoke.get("runtime")
                or vm.get("runtime") != preparation["runtime"] or vm.get("localizer") != MD):
            raise ValueError("passed official MDETR smoke/val runtime required")
        for flag in ("repeat_bitwise_parity", "raw_postprocess_bitwise_parity", "preprocess_bitwise_parity", "native_tuple_selection_parity"):
            if smoke.get(flag) is not True:
                raise ValueError("missing official MDETR smoke parity")
        for key in ("runtime_code", "extractor_code", "fixture"):
            verify(smoke[key])
        for key in ("checkpoint", "text_assets"):
            verify(preparation["runtime"][key])
        if str(Path(sys.executable).absolute()) != preparation["runtime_python"]:
            raise ValueError("use the independent pinned MDETR Python environment")
        verify(preparation["runtime_environment"])
        binding["mdetr_preparation"] = bind(args.mdetr_preparation)
        binding["runtime_code"] = bind(ROOT / "tools/mdetr_frozen_runtime.py")
    else:
        raise ValueError("unknown localizer")
    return protocol, rows, binding, vm, old


def make_runtime(args, protocol):
    import torch
    if not args.device.startswith("cuda:"):
        raise ValueError("formal extraction requires an explicit CUDA device")
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(2)
    if args.localizer == MM:
        from tools.extract_mmgdino_responsibility_cache import MMDetectionFrozenRuntime
        mm = protocol["localizers"][MM]["runtime"]
        runtime = MMDetectionFrozenRuntime(config_path=Path(mm["config"]["path"]),
            checkpoint_path=Path(mm["checkpoint"]["path"]), device=args.device, feature_dtype=torch.float16)
        receipt = {"localizer": MM, "parent_runtime": mm, "torch": torch.__version__, "all_frozen": True}
    else:
        from tools.mdetr_frozen_runtime import MDETRFrozenRuntime
        prep = read(args.mdetr_preparation)
        identity = prep["runtime"]
        runtime = MDETRFrozenRuntime(upstream_root=prep["upstream"], checkpoint_path=identity["checkpoint"]["path"],
            text_assets=Path(identity["text_assets"]["path"]).parent, device=args.device,
            expected_checkpoint_sha256=identity["checkpoint"]["sha256"])
        if runtime.receipt != identity:
            runtime.close()
            raise ValueError("MDETR runtime/environment drift after official smoke")
        receipt = runtime.receipt
    if runtime.model.training or any(p.requires_grad for p in runtime.model.parameters()):
        runtime.close()
        raise ValueError("model must be eval and fully frozen")
    return runtime, receipt


def runtime_preflight(runtime, vm, gref_ids):
    import torch
    shard = verify(vm["shards"][0])
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    anchor = payload["rows"][0]
    if anchor["sample_id"] in gref_ids or anchor["sample_id"].startswith("gref"):
        raise ValueError("preflight must be a non-gRef validation expression")
    hook = runtime.infer(Path(anchor["image_path"]), anchor["caption"])
    fields = ("query_features", "native_score", "boxes", "candidate_mask")
    parity = {k: torch.equal(getattr(hook, k), anchor[k]) for k in fields}
    if "native_boxes_xyxy_abs" in anchor:
        parity["native_boxes_xyxy_abs"] = torch.equal(hook.native_boxes_xyxy_abs, anchor["native_boxes_xyxy_abs"])
        parity["native_selected_index"] = hook.native_selected_index == anchor["native_selected_index"]
    if not all(parity.values()):
        raise ValueError("non-gRef cache runtime parity failed: " + str(parity))
    return {"non_gref_sample": anchor["sample_id"], "val_shard": bind(shard), "bitwise_parity": parity}


def _existing_shard(path, binding, expected, start, old):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != SHARD_SCHEMA or payload.get("split") != SPLIT
            or payload.get("start") != start or payload.get("binding") != binding
            or [r["sample_id"] for r in payload.get("rows", [])] != [r["sample_id"] for r in expected]):
        raise ValueError("existing shard protocol/order/coverage drift")
    for row, source in zip(payload["rows"], expected):
        attrs = {k: row[k] for k in ("query_features", "native_score", "boxes", "candidate_mask")}
        if binding["localizer"] == MD:
            attrs.update(native_selected_index=row["native_selected_index"], image_size=row["image_size"],
                         native_boxes_xyxy_abs=row["native_boxes_xyxy_abs"])
        reconstructed = build_row(source, SimpleNamespace(**attrs), binding["localizer"])
        for key, value in reconstructed.items():
            if key not in row or (not torch.equal(row[key], value) if torch.is_tensor(value) else row[key] != value):
                raise ValueError("cached row ABI/source-derived field drift: " + key)
        if old is not None:
            check_mm_legacy(row, old[row["sample_id"]])
    return payload


def extract(args):
    import torch
    if args.worker_count != 4 or not 0 <= args.worker_index < 4 or args.shard_size != 128:
        raise ValueError("fixed four-worker/shard128 extraction required")
    protocol, requests, binding, vm, old = context(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    complete = output / f"worker_{args.worker_index:02d}.json"
    if complete.exists():
        receipt = read(complete)
        expected_starts = [start for i, start in enumerate(range(0, len(requests), 128)) if i % 4 == args.worker_index]
        if (receipt.get("schema") != WORKER_SCHEMA or receipt.get("binding") != binding
                or receipt.get("status") != "complete" or receipt.get("worker_index") != args.worker_index
                or receipt.get("worker_count") != 4
                or [r["start"] for r in receipt.get("shards", [])] != expected_starts):
            raise ValueError("existing worker completion binding drift")
        for entry in receipt["shards"]:
            start = entry["start"]
            _existing_shard(verify(entry), binding, requests[start:start+entry["rows"]], start, old)
        print(json.dumps({"status": "already_complete_verified_no_forward", "worker": args.worker_index}), flush=True)
        return
    if (output / "manifest.json").exists():
        raise ValueError("completed cache is immutable")
    runtime, runtime_receipt = make_runtime(args, protocol)
    try:
        preflight = {"schema": "arrow.confidence_readout.gref_cache_preflight/v1", "status": "passed",
            "binding": binding, "runtime": runtime_receipt, "worker_index": args.worker_index,
            **runtime_preflight(runtime, vm, {r["sample_id"] for r in requests}),
            "optimizer_created": False, "gref_forward_before_preflight": False}
        preflight_path = output / f"preflight_{args.worker_index:02d}.json"
        if preflight_path.exists():
            if read(preflight_path) != preflight:
                raise ValueError("resumed preflight identity/parity drift")
        else:
            write_new(preflight_path, preflight)
        deadline = time.monotonic() + args.preflight_timeout
        while True:
            paths = [output / f"preflight_{i:02d}.json" for i in range(4)]
            if all(p.exists() for p in paths):
                for i, path in enumerate(paths):
                    record = read(path)
                    if record.get("binding") != binding or record.get("runtime") != runtime_receipt or record.get("worker_index") != i or record.get("status") != "passed":
                        raise ValueError("all-GPU preflight barrier mismatch")
                break
            if time.monotonic() > deadline:
                raise TimeoutError("other GPUs did not finish non-gRef runtime preflight")
            time.sleep(1)
        shards = []
        for index, start in enumerate(range(0, len(requests), 128)):
            if index % 4 != args.worker_index:
                continue
            expected = requests[start:start+128]
            path = output / f"shard_{index:06d}.pt"
            if path.exists():
                _existing_shard(path, binding, expected, start, old)
            else:
                rows = []
                for source in expected:
                    hook = runtime.infer(Path(source["image_path"]), source["expression"])
                    row = build_row(source, hook, args.localizer)
                    if old is not None:
                        check_mm_legacy(row, old[row["sample_id"]])
                    rows.append(row)
                tmp = path.with_suffix(".pt.partial")
                with tmp.open("xb") as stream:
                    torch.save({"schema": SHARD_SCHEMA, "split": SPLIT, "start": start,
                                "binding": binding, "rows": rows}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                tmp.rename(path)
            shards.append({**bind(path), "rows": len(expected), "start": start})
            print(json.dumps({"localizer": args.localizer, "worker": args.worker_index,
                              "shard": index, "rows_in_shard": len(expected)}), flush=True)
        write_new(complete, {"schema": WORKER_SCHEMA, "status": "complete", "binding": binding,
            "runtime": runtime_receipt, "worker_index": args.worker_index, "worker_count": 4,
            "preflight": bind(preflight_path), "shards": shards, "optimizer_created": False,
            "head_evaluation_performed": False, "frozen": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device)})
    finally:
        runtime.close()


def finalize(args):
    _, requests, binding, vm, old = context(args)
    output = args.output.resolve()
    workers = [read(output / f"worker_{i:02d}.json") for i in range(4)]
    all_shards = []
    for i, worker in enumerate(workers):
        if (worker.get("schema") != WORKER_SCHEMA or worker.get("status") != "complete"
                or worker.get("binding") != binding or worker.get("worker_index") != i
                or worker.get("worker_count") != 4 or worker.get("runtime") != workers[0]["runtime"]):
            raise ValueError("worker completion/runtime/binding drift")
        preflight = read(verify(worker["preflight"]))
        if preflight.get("binding") != binding or preflight.get("status") != "passed":
            raise ValueError("worker lacks passed bound preflight")
        for entry in worker["shards"]:
            if entry["start"] // 128 % 4 != i:
                raise ValueError("worker extracted a foreign shard")
        all_shards.extend(worker["shards"])
    cursor = 0
    for entry in sorted(all_shards, key=lambda r: r["start"]):
        if entry["start"] != cursor or entry["rows"] != min(128, len(requests) - cursor):
            raise ValueError("missing/overlapping/partial extraction shards")
        _existing_shard(verify(entry), binding, requests[cursor:cursor+entry["rows"]], cursor, old)
        cursor += entry["rows"]
    if cursor != len(requests):
        raise ValueError("incomplete all-query cache")
    manifest = {"schema": CACHE_SCHEMA, "status": "complete", "formal": True, "split": SPLIT,
        "localizer": args.localizer, "records": cursor, "shard_size": 128,
        "feature_dtype": "float16", "binding": binding, "all_heads_sealed": binding["all_heads_sealed"],
        "annotation": binding["source"], "index": binding["source"],
        "model": {"checkpoint": vm["model"]["checkpoint"], "frozen": True,
                  "query_count": 900 if args.localizer == MM else 100, "feature_dim": 256},
        "runtime": workers[0]["runtime"], "shards": sorted(all_shards, key=lambda r: r["start"]),
        "evaluation_groups": evaluation_groups(requests), "evaluation_group_contract": "original source[worker::4], B32 per worker, final19 each",
        "population": population(requests), "source_disjoint_population": population([r for r in requests if r["finecops_train_val_source_disjoint"]]),
        "worker_receipts": [bind(output / f"worker_{i:02d}.json") for i in range(4)],
        "no_new_finecops_test": True, "head_evaluation_performed": False,
        "legacy_mm_geometry_parity_every_record": args.localizer == MM}
    write_new(output / "manifest.json", manifest)
    print(json.dumps({"manifest": bind(output / "manifest.json"), "records": cursor}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "finalize"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--all-heads-sealed", type=Path, required=True)
    parser.add_argument("--localizer", choices=LOCALIZERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--images-manifest", type=Path, default=ROOT / "data/gref_fixed_targets_v1/manifests/images.json")
    parser.add_argument("--legacy-records", type=Path, default=ROOT / "outputs/arrow_gref_fixed_targets_20260905/all_records.json")
    parser.add_argument("--mdetr-preparation", type=Path)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-timeout", type=float, default=600.)
    args = parser.parse_args()
    if (args.worker_count != 4 or args.shard_size != 128 or not 0 <= args.worker_index < 4
            or not math.isfinite(args.preflight_timeout) or args.preflight_timeout <= 0):
        raise ValueError("fixed worker4/shard128 topology and positive finite timeout required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    (extract if args.command == "extract" else finalize)(args)


if __name__ == "__main__":
    main()
