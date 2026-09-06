#!/usr/bin/env python3
"""MDETR cache v2: inactive negative edit-reference boxes, unchanged detector.

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
from tools.recover_mdetr_negative_reference_cache import request_for_study, validate_audit, CONTRACT, tensor_hash
from tools import extract_mdetr_readout_cache as legacy

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
    study_request, reference = request_for_study(request)
    row = legacy.build_row(study_request, hook, image_sha256)
    if reference is not None:
        row["annotation_reference_boxes"] = reference
        row["annotation_reference_boxes_active"] = False
        row["annotation_reference_boxes_role"] = "source_parent_edit_reference_not_study_ground_truth"
        row["negative_reference_contract"] = CONTRACT
    return row


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
    validate_audit(args.negative_contract_audit, args.source_manifest)
    source, requests = source_requests(args.source_manifest)
    if source["split"] != "train":
        raise ValueError("row-adapter smoke uses TRAIN only")
    positive = next(r for r in requests if r.kind == "positive")
    negative = next(r for r in requests if r.kind == "text")
    from types import SimpleNamespace
    intermediate = args.output.with_name(args.output.stem + ".runtime_parity.json")
    runtime_args = SimpleNamespace(**vars(args))
    runtime_args.output = intermediate
    legacy.smoke(runtime_args)
    receipt = load_json(intermediate)
    rt = runtime(args)
    results = {}
    try:
        for name, request in (("positive", positive), ("first_actual_negative", negative)):
            hook = rt.infer(request.image_path, request.caption)
            row = build_row(request, hook, file_digest(request.image_path))
            if request.kind == "text":
                if row["gt_boxes"].shape != (0, 4) or row["annotation_reference_boxes_active"] is not False:
                    raise ValueError("negative adapter did not clear study GT")
                if tensor_hash(row["annotation_reference_boxes"]) != tensor_hash(request.gt_boxes):
                    raise ValueError("source negative reference bytes changed")
            elif not torch.equal(row["gt_boxes"], request.gt_boxes):
                raise ValueError("positive study GT changed")
            results[name] = {"sample_id":request.sample_id, "kind":request.kind,
                             "source_reference_shape":list(request.gt_boxes.shape),
                             "study_gt_shape":list(row["gt_boxes"].shape),
                             "reference_preserved_but_inactive":request.kind == "text",
                             "native_query_unchanged": row["native_selected_index"] == hook.native_selected_index}
        if rt.receipt != receipt["runtime"]:
            raise ValueError("row-adapter/runtime parity environment drift")
    finally:
        rt.close()
    receipt.update(extractor_code=record(__file__), parent_extractor_code=record(legacy.__file__),
                   adapter_code=record(Path(__file__).with_name("recover_mdetr_negative_reference_cache.py")),
                   negative_contract_audit=record(args.negative_contract_audit), cache_row_preflight=results,
                   runtime_parity_receipt=record(intermediate), cache_revision=2)
    write_json_new(args.output, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def make_binding(args, source):
    if file_digest(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol SHA mismatch")
    smoke_receipt = load_json(args.smoke_receipt)
    if smoke_receipt.get("cache_revision") != 2:
        raise ValueError("v2 row-adapter smoke required")
    for key in ("extractor_code", "parent_extractor_code", "adapter_code", "negative_contract_audit"):
        verify_record(smoke_receipt[key])
    if smoke_receipt["extractor_code"] != record(__file__):
        raise ValueError("v2 extractor code differs from passed smoke")
    validate_audit(smoke_receipt["negative_contract_audit"]["path"], args.source_manifest)
    if args.shard_size != source["shard_size"]:
        raise ValueError("source shard/offset contract changed")
    return {"source_manifest":record(args.source_manifest), "protocol":record(args.protocol),
            "smoke":record(args.smoke_receipt), "checkpoint_sha256":args.checkpoint_sha256,
            "split":source["split"], "localizer":LOCALIZER, "shard_size":args.shard_size,
            "extractor_revision":2, "negative_reference_contract":CONTRACT,
            "negative_contract_audit":smoke_receipt["negative_contract_audit"]}


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
    binding = make_binding(args, source)
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
    p.add_argument("--negative-contract-audit", type=Path)
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
