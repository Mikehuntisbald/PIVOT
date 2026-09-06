#!/usr/bin/env python3
"""Versioned MDETR query cache, preserving the sealed FineCops train/val order.

Only train/val source manifests are accepted. Formal extraction requires a
protocol SHA and a passed independent runtime smoke receipt. Workers own
disjoint whole shards; finalization checks the exact source identity surface.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.mdetr_frozen_runtime import MDETRFrozenRuntime, LOCALIZER, file_digest, preprocess

CACHE_SCHEMA = "arrow.confidence_readout.cache_manifest/v1"
ROW_SCHEMA = "arrow.confidence_readout.cache_row/v1"
SHARD_SCHEMA = "arrow.confidence_readout.cache_shard/v1"
EXPECTED = {"train": 163792, "val": 18455}


def load_json(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_digest(path)}


def verify_record(value):
    if not isinstance(value, dict) or file_digest(value["path"]) != value["sha256"]:
        raise ValueError("bound file hash drift")
    return Path(value["path"])


def write_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"refuse overwrite: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.rename(path)


def write_json_idempotent(path, value):
    """Keep completed worker/final receipts byte-identical on a valid resume."""
    path = Path(path)
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"completed receipt differs; refuse overwrite: {path}")
        return
    write_json_new(path, value)


def build_row(request, hook, image_sha256):
    if request.kind not in {"positive", "text"}:
        raise ValueError("this study excludes image negatives")
    if hook.query_features.shape != (100, 256) or hook.boxes.shape != (100, 4):
        raise ValueError("MDETR shape drift")
    if request.kind == "text" and request.gt_boxes.numel():
        raise ValueError("no-target must not carry GT")
    if request.kind == "positive" and request.gt_boxes.shape != (1, 4):
        raise ValueError("single-target must have exactly one box")
    return {"schema": ROW_SCHEMA, "localizer": LOCALIZER, "sample_id": request.sample_id,
            "annotation_id": request.annotation_id, "image_path": str(request.image_path),
            "image_sha256": image_sha256, "image_size": list(hook.image_size),
            "source_image_id": request.source_image_id, "cluster_image_id": request.cluster_image_id,
            "caption": request.caption, "kind": request.kind, "parent_positive_id": request.parent_positive_id,
            "level": request.level, "negative_type": request.negative_type, "negative_level": request.negative_level,
            "query_features": hook.query_features, "native_score": hook.native_score, "boxes": hook.boxes,
            "candidate_mask": hook.candidate_mask, "gt_boxes": request.gt_boxes,
            "native_selected_index": hook.native_selected_index,
            "native_boxes_xyxy_abs": hook.native_boxes_xyxy_abs}


def source_requests(source_manifest):
    source = load_json(source_manifest)
    split = source.get("split")
    if split not in EXPECTED or source.get("status") != "complete" or source.get("records") != EXPECTED[split] or source.get("formal") is not True:
        raise ValueError("requires sealed complete train/val source manifest")
    annotation = verify_record(source["annotation"])
    if "test" in annotation.name.lower():
        raise ValueError("FineCops Test annotation forbidden")
    parser_path = verify_record(source["extractor"])
    # Import the sealed data parser only after refusing held-out manifests.
    from tools import extract_b32a1_finecops_cache as source_parser
    if Path(source_parser.__file__).resolve() != parser_path.resolve():
        raise ValueError("imported source parser differs from sealed extractor")
    build_requests = source_parser.build_requests
    index_path = verify_record(source["index"])
    index = [json.loads(line) for line in index_path.read_text().splitlines()]
    requests = build_requests(annotation, split=split, image_root=source["images_root"], require_images=True)
    fields = ("sample_id", "annotation_id", "kind", "parent_positive_id", "cluster_image_id", "level")
    if len(requests) != EXPECTED[split] or len(index) != len(requests):
        raise ValueError("official source count drift")
    for item, request in zip(index, requests):
        if any(item[k] != getattr(request, k) for k in fields):
            raise ValueError("source index identity/order mismatch")
    return source, requests


def runtime(args):
    return MDETRFrozenRuntime(upstream_root=args.upstream, checkpoint_path=args.checkpoint,
                             text_assets=args.text_assets, device=args.device,
                             expected_checkpoint_sha256=args.checkpoint_sha256)


def smoke(args):
    from PIL import Image
    fixture = load_json(args.fixture)
    if fixture.get("scope") not in {"refcoco_val", "finecops_train_nonheldout"}:
        raise ValueError("smoke must use explicitly non-heldout fixture")
    rt = runtime(args)
    fixture_path = verify_record(fixture["image"])
    start = time.monotonic()
    try:
        a = rt.infer(fixture_path, fixture["caption"])
        b = rt.infer(fixture_path, fixture["caption"])
        for key in ("query_features", "native_score", "boxes", "candidate_mask", "native_boxes_xyxy_abs"):
            if not torch.equal(getattr(a, key), getattr(b, key)):
                raise ValueError(f"repeated inference is not bitwise deterministic: {key}")
        # Independently execute official model without our hook and then its
        # PostProcess. This checks that feature collection is observational.
        rt.close()
        with Image.open(fixture_path) as image:
            tensor = preprocess(image).unsqueeze(0).to(rt.device)
        official = rt._forward(tensor, fixture["caption"])
        with torch.inference_mode():
            post = rt.postprocessor(official, torch.tensor([a.image_size], device=rt.device))[0]
        if not torch.equal(a.native_score, post["scores"].cpu()) or not torch.equal(a.boxes, official["pred_boxes"][0].cpu()) or not torch.equal(a.native_boxes_xyxy_abs, post["boxes"].cpu()):
            raise ValueError("hook versus official no-hook raw output parity failed")
        pairs = sorted(zip(post["scores"].tolist(), post["boxes"].tolist()), reverse=True)
        if pairs[0] != (a.native_score[a.native_selected_index].item(), a.native_boxes_xyxy_abs[a.native_selected_index].tolist()):
            raise ValueError("official evaluator tuple selection mismatch")
        # Load only the official transforms module, avoiding dataset registries.
        spec = importlib.util.spec_from_file_location("mdetr_official_transforms_probe", args.upstream / "datasets/transforms.py")
        transforms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transforms)
        with Image.open(fixture_path) as image:
            resized, _ = transforms.resize(image.convert("RGB"), None, 800, 1333)
            original, _ = transforms.ToTensor()(resized, None)
            original, _ = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(original, None)
            if not torch.equal(original, preprocess(image)):
                raise ValueError("official preprocessing parity failed")
        receipt = {"schema": "arrow.confidence_readout.mdetr_smoke/v1", "status": "passed", "runtime": rt.receipt,
                   "fixture": record(args.fixture), "raw_postprocess_bitwise_parity": True,
                   "native_tuple_selection_parity": True, "preprocess_bitwise_parity": True,
                   "repeat_bitwise_parity": True, "confidence_training_performed": False,
                   "FineCops_Test_read": False, "gRef_forward": False, "seconds": time.monotonic()-start,
                   "runtime_code": record(Path(__file__).with_name("mdetr_frozen_runtime.py")),
                   "extractor_code": record(__file__),
                   "cuda_peak_allocated": torch.cuda.max_memory_allocated(rt.device),
                   "cuda_peak_reserved": torch.cuda.max_memory_reserved(rt.device)}
        write_json_new(args.output, receipt)
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        rt.close()


def extract(args):
    if not 0 <= args.worker_index < args.worker_count or args.shard_size <= 0:
        raise ValueError("invalid worker topology")
    if file_digest(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol SHA mismatch")
    smoke_receipt = load_json(args.smoke_receipt)
    if smoke_receipt.get("status") != "passed" or smoke_receipt["runtime"]["checkpoint"]["sha256"] != args.checkpoint_sha256:
        raise ValueError("runtime preflight not passed for checkpoint")
    for key in ("runtime_code", "extractor_code"):
        verify_record(smoke_receipt[key])
    source, requests = source_requests(args.source_manifest)
    if args.shard_size != source["shard_size"]:
        raise ValueError("new cache must preserve source index shard/offset contract")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    split = source["split"]
    binding = {"source_manifest": record(args.source_manifest), "protocol": record(args.protocol),
               "smoke": record(args.smoke_receipt), "checkpoint_sha256": args.checkpoint_sha256,
               "split": split, "localizer": LOCALIZER, "shard_size": args.shard_size}
    if (out / "manifest.json").exists() and load_json(out / "manifest.json").get("binding") != binding:
        raise ValueError("completed cache binding differs")
    rt = runtime(args)
    if rt.receipt != smoke_receipt["runtime"]:
        rt.close()
        raise ValueError("runtime environment or assets drifted since passed smoke")
    hashes = {}
    completed = []
    try:
        for index, start in enumerate(range(0, len(requests), args.shard_size)):
            if index % args.worker_count != args.worker_index:
                continue
            path = out / f"shard_{index:06d}.pt"
            expected = requests[start:start + args.shard_size]
            if path.exists():
                old = torch.load(path, map_location="cpu", weights_only=False)
                if old.get("binding") != binding or old.get("schema") != SHARD_SCHEMA or [r["sample_id"] for r in old["rows"]] != [r.sample_id for r in expected]:
                    raise ValueError("existing shard binding or identity drift")
            else:
                rows = []
                for req in expected:
                    if req.image_path not in hashes:
                        hashes[req.image_path] = file_digest(req.image_path)
                    rows.append(build_row(req, rt.infer(req.image_path, req.caption), hashes[req.image_path]))
                tmp = path.with_suffix(".pt.partial")
                if tmp.exists():
                    raise ValueError("incomplete shard partial preserved; explicit repair required")
                with tmp.open("xb") as f:
                    torch.save({"schema": SHARD_SCHEMA, "split": split, "start": start, "binding": binding, "rows": rows}, f)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.rename(path)
            completed.append({**record(path), "rows": len(expected), "start": start})
            print(json.dumps({"worker": args.worker_index, "shard": index, "rows": start+len(expected)}), flush=True)
        result = {"schema": "arrow.confidence_readout.cache_worker/v1", "status": "complete", "binding": binding,
                  "runtime": rt.receipt, "worker_index": args.worker_index, "worker_count": args.worker_count,
                  "shards": completed}
        write_json_idempotent(out / f"worker_{args.worker_index:02d}.json", result)
    finally:
        rt.close()


def finalize(args):
    source, requests = source_requests(args.source_manifest)
    output = args.output.resolve()
    workers = [load_json(output / f"worker_{i:02d}.json") for i in range(args.worker_count)]
    binding = workers[0]["binding"]
    if any(w["binding"] != binding or w["worker_count"] != args.worker_count or w["worker_index"] != i or w["status"] != "complete" for i, w in enumerate(workers)):
        raise ValueError("worker binding or completion mismatch")
    if any(w["runtime"] != workers[0]["runtime"] for w in workers):
        raise ValueError("workers used different runtime environments")
    if binding["source_manifest"] != record(args.source_manifest):
        raise ValueError("final source manifest drift")
    for key in ("protocol", "smoke", "source_manifest"):
        verify_record(binding[key])
    shards = sorted([s for w in workers for s in w["shards"]], key=lambda x: x["start"])
    cursor = 0
    for shard in shards:
        if shard["start"] != cursor:
            raise ValueError("missing or overlapping shards")
        path = verify_record(shard)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = payload["rows"]
        if len(rows) != shard["rows"] or payload["binding"] != binding or [r["sample_id"] for r in rows] != [r.sample_id for r in requests[cursor:cursor+len(rows)]]:
            raise ValueError("shard source identities drift")
        cursor += len(rows)
    if cursor != len(requests):
        raise ValueError("incomplete total cache")
    manifest = {"schema": CACHE_SCHEMA, "status": "complete", "formal": True, "split": source["split"],
                "records": cursor, "localizer": LOCALIZER, "shard_size": binding["shard_size"],
                "feature_dtype": "float16", "index": source["index"], "annotation": source["annotation"],
                "images_root": source["images_root"], "binding": binding,
                "model": {"checkpoint": workers[0]["runtime"]["checkpoint"], "frozen": True, "query_count": 100, "feature_dim": 256},
                "runtime": workers[0]["runtime"], "shards": shards}
    write_json_idempotent(output / "manifest.json", manifest)
    print(json.dumps({"manifest": record(output / "manifest.json"), "records": cursor}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["smoke", "extract", "finalize"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-sha256")
    p.add_argument("--text-assets", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fixture", type=Path)
    p.add_argument("--source-manifest", type=Path)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--protocol-sha256")
    p.add_argument("--smoke-receipt", type=Path)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--worker-count", type=int, default=1)
    p.add_argument("--shard-size", type=int, default=128)
    args = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "smoke":
        smoke(args)
    elif args.command == "extract":
        extract(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
