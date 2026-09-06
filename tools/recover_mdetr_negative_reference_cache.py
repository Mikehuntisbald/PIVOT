#!/usr/bin/env python3
"""Audit inactive negative reference boxes and losslessly rewrap positive cache.

FineCops edits retain their parent's bbox in COCO annotations. It is never
study ground truth for a no-target request. No labels or model scores change.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.extract_mdetr_readout_cache import load_json, record, verify_record, write_json_new, write_json_idempotent
from tools.mdetr_frozen_runtime import file_digest, LOCALIZER

CONTRACT = "negative_annotation_reference_inactive/v1"
COUNTS = {"train": (83341, 80451), "val": (9426, 9029)}


def request_for_study(request):
    """Separate the source edit reference from the no-target supervision event."""
    if request.kind not in {"positive", "text"}:
        raise ValueError("only positive/text requests belong to this study")
    reference = request.gt_boxes
    if reference.shape != (1, 4) or reference.dtype != torch.float32 or not torch.isfinite(reference).all():
        raise ValueError("sealed source parser must supply one finite reference box")
    if request.kind == "positive":
        return request, None
    value = dict(vars(request))
    value["gt_boxes"] = reference.new_empty((0, 4))
    return SimpleNamespace(**value), reference.detach().clone()


def audit_source(path):
    source = load_json(path)
    split = source.get("split")
    if split not in COUNTS or source.get("formal") is not True or source.get("status") != "complete":
        raise ValueError("negative-reference audit only accepts sealed train/val")
    for field in ("annotation", "index", "extractor"):
        verify_record(source[field])
    if "test" in Path(source["annotation"]["path"]).name.lower():
        raise ValueError("Test forbidden")
    data = load_json(source["annotation"]["path"])
    annotations = {a["id"]: a for a in data["annotations"]}
    images = {i["id"]: i for i in data["images"]}
    if len(annotations) != len(data["annotations"]) or len(images) != len(data["images"]):
        raise ValueError("duplicate source annotation/image identity")
    counts = Counter()
    negative_types = Counter()
    first_negative = None
    indices = [json.loads(line) for line in Path(source["index"]["path"]).read_text().splitlines()]
    if len(indices) != sum(COUNTS[split]) or source["records"] != len(indices):
        raise ValueError("source count drift")
    for index in indices:
        a = annotations[index["annotation_id"]]
        negative = a.get("negative_type") is not None or a.get("negative_cate") is not None
        if negative != (index["kind"] == "text") or index["kind"] not in {"positive", "text"}:
            raise ValueError("source negative semantic/index mismatch")
        if not negative:
            counts["positive"] += 1
            continue
        if a.get("negative_cate") == "image" or not a.get("negative_type"):
            raise ValueError("unexpected negative-image or untyped negative")
        parent = annotations.get(a.get("positive_id"))
        if parent is None or parent.get("negative_type") is not None or parent.get("negative_cate") is not None:
            raise ValueError("negative must reference a positive source")
        ni, pi = images[a["image_id"]], images[parent["image_id"]]
        if a.get("bbox") != parent.get("bbox") or not isinstance(a.get("bbox"), list) or len(a["bbox"]) != 4:
            raise ValueError("negative bbox is not the exact parent edit-reference box")
        if ni["file_name"] != pi["file_name"] or (ni["width"], ni["height"]) != (pi["width"], pi["height"]):
            raise ValueError("negative text must preserve the parent image/geometry")
        counts["negative_text"] += 1
        counts["reference_equals_parent"] += 1
        negative_types[str(a["negative_type"])] += 1
        if first_negative is None:
            first_negative = {"sample_id": index["sample_id"], "annotation_id": a["id"], "parent_positive_id": parent["id"],
                              "source_bbox_xywh_abs": a["bbox"], "gt_boxes_after_adapter_shape": [0, 4]}
    if (counts["positive"], counts["negative_text"]) != COUNTS[split]:
        raise ValueError("positive/text population count drift")
    return {"source_manifest": record(path), "annotation": source["annotation"], "index": source["index"],
            "source_parser": source["extractor"], "counts": dict(counts), "negative_types": dict(negative_types),
            "first_negative_in_source_order": first_negative, "ambiguous_rows": 0}


def validate_audit(path, source_manifest=None):
    audit = load_json(path)
    if audit.get("schema") != "arrow.confidence_readout.negative_reference_audit/v1" or audit.get("status") != "passed" or audit.get("contract") != CONTRACT:
        raise ValueError("negative reference audit missing or failed")
    for name, item in audit["sources"].items():
        if name not in COUNTS:
            raise ValueError("unexpected split in audit")
        for key in ("source_manifest", "annotation", "index", "source_parser"):
            verify_record(item[key])
        if (item["counts"]["positive"], item["counts"]["negative_text"]) != COUNTS[name]:
            raise ValueError("audited population drift")
    if set(audit["sources"]) != set(COUNTS):
        raise ValueError("both train and val audits are required")
    if source_manifest is not None and record(source_manifest) not in [a["source_manifest"] for a in audit["sources"].values()]:
        raise ValueError("source manifest not in sealed negative audit")
    return audit


def tensor_hash(value):
    if not torch.is_tensor(value) or value.device.type != "cpu" or value.requires_grad:
        raise ValueError("only detached CPU tensors may be recovered")
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def rows_fingerprint(rows):
    ordered = hashlib.sha256()
    for row in rows:
        tensor_values = {key: tensor_hash(value) for key, value in row.items() if torch.is_tensor(value)}
        metadata = {key: value for key, value in row.items() if not torch.is_tensor(value)}
        item = {"tensors": tensor_values, "metadata": metadata}
        ordered.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode())
    return ordered.hexdigest()


def rewrap_positive_payload(payload, *, binding, expected_requests):
    rows = payload.get("rows", [])
    if payload.get("schema") != "arrow.confidence_readout.cache_shard/v1" or not rows or len(rows) != len(expected_requests):
        raise ValueError("invalid old positive shard")
    for row, request in zip(rows, expected_requests):
        if row["kind"] != "positive" or request.kind != "positive":
            raise ValueError("only already-complete positive-only shards are reusable")
        for key in ("sample_id", "annotation_id", "kind", "parent_positive_id", "cluster_image_id", "caption", "level", "negative_type", "negative_level"):
            if row[key] != getattr(request, key):
                raise ValueError(f"source metadata drift: {key}")
        if not torch.equal(row["gt_boxes"], request.gt_boxes):
            raise ValueError("positive GT changed")
        if str(request.image_path) != row["image_path"]:
            raise ValueError("source image path changed")
    return {**payload, "binding": binding, "rows": rows}


def recover(args):
    from tools.extract_mdetr_readout_cache_v2 import make_binding, source_requests
    if args.old.resolve() == args.output.resolve():
        raise ValueError("recovery must use a new cache directory")
    if any(args.output.glob("worker_*.json")) or (args.output / "manifest.json").exists():
        raise ValueError("cannot recover into a running or completed cache")
    source, requests = source_requests(args.source_manifest)
    binding = make_binding(args, source)
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for old_path in sorted(args.old.glob("shard_*.pt")):
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        if old["binding"]["checkpoint_sha256"] != args.checkpoint_sha256 or old["binding"]["protocol"] != record(args.protocol) or old["binding"]["source_manifest"] != record(args.source_manifest):
            raise ValueError("v1 shard original model/protocol/source drift")
        start = old["start"]
        if start != int(old_path.stem.split("_")[1])*args.shard_size:
            raise ValueError("old shard start/index mismatch")
        expected = requests[start:start+len(old["rows"])]
        old_fingerprint = rows_fingerprint(old["rows"])
        wrapped = rewrap_positive_payload(old, binding=binding, expected_requests=expected)
        new_path = args.output / old_path.name
        if new_path.exists():
            existing = torch.load(new_path, map_location="cpu", weights_only=False)
            if existing["binding"] != binding or rows_fingerprint(existing["rows"]) != old_fingerprint:
                raise ValueError("existing recovered shard drift")
        else:
            tmp = new_path.with_suffix(".pt.partial")
            with tmp.open("xb") as f:
                torch.save(wrapped, f)
                f.flush(); os.fsync(f.fileno())
            tmp.rename(new_path)
        restored = torch.load(new_path, map_location="cpu", weights_only=False)
        if rows_fingerprint(restored["rows"]) != old_fingerprint:
            raise ValueError("positive row tensor bits or metadata changed during rewrap")
        receipts.append({"old": record(old_path), "new": record(new_path), "rows": len(old["rows"]),
                         "ordered_row_tensor_and_metadata_sha256": old_fingerprint, "each_row_tensor_bitwise_verified": True})
        if len(receipts)%50 == 0:
            print(json.dumps({"recovered_shards":len(receipts),"rows":sum(x["rows"] for x in receipts)}),flush=True)
    result = {"schema":"arrow.confidence_readout.positive_cache_recovery/v1","status":"complete","binding":binding,
              "old_cache_preserved":str(args.old.resolve()),"shards":receipts,"reused_rows":sum(x["rows"] for x in receipts),
              "model_forwards":0,"partial_shards_not_reused":[str(p) for p in args.old.glob("*.partial")],
              "note":"Uncommitted in-memory rows of the failing shard may be re-forwarded; no completed positive record is re-forwarded."}
    write_json_idempotent(args.output / "positive_recovery_receipt.json", result)
    print(json.dumps({"receipt":record(args.output/"positive_recovery_receipt.json"),"reused_rows":result["reused_rows"]}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["audit","recover"])
    for name in ("train-source","val-source","source-manifest","old","output","protocol","smoke-receipt"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--protocol-sha256")
    p.add_argument("--checkpoint-sha256")
    p.add_argument("--shard-size",type=int,default=128)
    args=p.parse_args()
    if args.command=="audit":
        result={"schema":"arrow.confidence_readout.negative_reference_audit/v1","status":"passed","contract":CONTRACT,
                "sources":{s:audit_source(path) for s,path in (("train",args.train_source),("val",args.val_source))},
                "supervision":"kind=text always has E=0 and Y=0; annotation bbox is inactive parent-edit reference", 
                "benchmark_semantics_source":"https://aclanthology.org/2024.emnlp-main.864/", "new_labels_or_model_results_used":False,
                "audit_code":record(__file__)}
        write_json_new(args.output,result)
        print(json.dumps(result,indent=2))
    else: recover(args)


if __name__=="__main__":main()
