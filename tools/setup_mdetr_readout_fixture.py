#!/usr/bin/env python3
"""Bind first train positive and its first paired edit for non-heldout smoke."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.extract_mdetr_readout_cache import record, verify_record, write_json_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.source_manifest.read_text())
    if source.get("split") != "train" or source.get("status") != "complete":
        raise ValueError("fixture must be from sealed TRAIN cache")
    index = [json.loads(line) for line in verify_record(source["index"]).read_text().splitlines()]
    pos = next(r for r in index if r["kind"] == "positive")
    neg = next(r for r in index if r["kind"] == "text" and r["parent_positive_id"] == pos["parent_positive_id"])
    args.output.mkdir(parents=True, exist_ok=True)
    for kind, row in (("positive", pos), ("negative_text", neg)):
        shard = source["shards"][row["shard"]]
        path = verify_record(shard)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        full = payload["rows"][row["offset"]]
        if full["sample_id"] != row["sample_id"]:
            raise ValueError("fixture row identity drift")
        image = record(full["image_path"])
        if image["sha256"] != full["image_sha256"]:
            raise ValueError("fixture image drift")
        result = {"schema": "arrow.confidence_readout.mdetr_fixture/v1", "scope": "finecops_train_nonheldout",
                  "source_manifest": record(args.source_manifest), "source_index": source["index"], "source_shard": shard,
                  "sample_id": full["sample_id"], "parent_positive_id": full["parent_positive_id"],
                  "kind": kind, "image": image, "caption": full["caption"], "benchmark_metrics": False}
        target = args.output / f"fixture_{kind}.json"
        write_json_new(target, result)
        print(json.dumps({"fixture": record(target), "sample_id": full["sample_id"]}), flush=True)


if __name__ == "__main__":
    main()
