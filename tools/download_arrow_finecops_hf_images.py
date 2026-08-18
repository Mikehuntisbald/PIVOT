#!/usr/bin/env python3
"""Fetch only FineCops-required GQA images from the immutable HF mirror."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests
import fsspec
import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import file_record, load_json, write_json_atomic


DATASET = "lmms-lab-encoder/GQA"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
API_URL = "https://huggingface.co/api/datasets/lmms-lab/GQA"
TREE_URL = "https://huggingface.co/api/datasets/lmms-lab/GQA/tree/main"
PAGE_SIZE = 100
SPLITS = (
    ("val_all_images", "val"),
    ("train_all_images", "train"),
    ("submission_all_images", "submission"),
    ("test_all_images", "test"),
    ("testdev_all_images", "testdev"),
    ("challenge_all_images", "challenge"),
)


def _get_page(config: str, split: str, offset: int, length: int = PAGE_SIZE) -> dict:
    for attempt in range(7):
        try:
            response = requests.get(
                ROWS_URL,
                params={
                    "dataset": DATASET,
                    "config": config,
                    "split": split,
                    "offset": offset,
                    "length": length,
                },
                timeout=60,
            )
            response.raise_for_status()
            value = response.json()
            if not isinstance(value.get("rows"), list):
                raise ValueError("HF rows response contract drifted")
            return value
        except (requests.RequestException, ValueError):
            if attempt == 6:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError("unreachable")


def _scan_split(
    config: str, split: str, wanted: set[str], workers: int
) -> dict[str, dict[str, Any]]:
    tree = requests.get(
        TREE_URL,
        params={"recursive": "true", "expand": "false", "limit": 1000},
        timeout=60,
    )
    tree.raise_for_status()
    paths = sorted(
        str(item["path"])
        for item in tree.json()
        if str(item.get("path", "")).startswith(config + "/")
        and str(item.get("path", "")).endswith(".parquet")
    )
    if not paths:
        raise ValueError(f"HF mirror has no parquet shards for {config}")

    def read_ids(path: str) -> tuple[str, list[str]]:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{path}"
        with fsspec.open(url, "rb", block_size=2 * 1024 * 1024) as handle:
            table = pq.read_table(handle, columns=["id"], pre_buffer=False)
        return path, [str(value) for value in table.column("id").to_pylist()]

    shard_results: dict[str, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(4, workers))) as executor:
        for path, values in executor.map(read_ids, paths):
            shard_results[path] = values
            print(
                f"[HF-GQA] {config} id-shards={len(shard_results)}/{len(paths)}",
                flush=True,
            )
    positions: dict[int, str] = {}
    base = 0
    for path in paths:
        values = shard_results[path]
        for local_index, image_id in enumerate(values):
            if image_id in wanted:
                positions[base + local_index] = image_id
        base += len(values)
    if not positions:
        return {}
    pending = sorted(positions)
    offsets: list[int] = []
    while pending:
        start = pending[0]
        offsets.append(start)
        pending = [value for value in pending if value > start + PAGE_SIZE - 1]
    pages = []
    for index, offset in enumerate(offsets, start=1):
        pages.append(_get_page(config, split, offset))
        if index % 25 == 0:
            print(f"[HF-GQA] {config} selected-pages={index}/{len(offsets)}", flush=True)
        # Stay below the public dataset-server rate limit.
        time.sleep(1.1)
    found: dict[str, dict[str, Any]] = {}
    for page in pages:
        for item in page["rows"]:
            row = item.get("row") or {}
            image_id = str(row.get("id", ""))
            if image_id not in wanted or image_id in found:
                continue
            image = row.get("image") or {}
            source_url = image.get("src")
            if not isinstance(source_url, str) or not source_url.startswith("https://"):
                raise ValueError(f"HF image row {image_id} has no signed source URL")
            found[image_id] = {
                "config": config,
                "split": split,
                "row_idx": int(item["row_idx"]),
                "source_url": source_url,
            }
    return found


def download(root: Path, *, workers: int) -> dict[str, Any]:
    root = root.expanduser().resolve()
    annotation_path = root / "raw" / "benchmark" / "test_expression_all_coco_format.json"
    annotation = load_json(annotation_path)
    images = annotation["images"]
    annotations = {int(row["image_id"]): row for row in annotation["annotations"]}
    expected: dict[str, tuple[int, int]] = {}
    for image in images:
        ann = annotations[int(image["id"])]
        if ann.get("negative_cate") == "image":
            continue
        expected[Path(str(image["file_name"])).stem] = (
            int(image["width"]),
            int(image["height"]),
        )
    if len(expected) != 4313:
        raise ValueError("FineCops original-image identity count drifted")
    output_root = root / "images" / "gqa"
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "manifests" / "hf_gqa_selective_download.json"
    previous_records: dict[str, dict[str, Any]] = {}
    if receipt_path.is_file():
        previous = load_json(receipt_path)
        if previous.get("schema") == "arrow.finecops.hf_gqa_selective_download/v1":
            previous_records = {
                str(key): dict(value)
                for key, value in (previous.get("records") or {}).items()
            }

    def valid_existing(image_id: str) -> bool:
        if image_id not in previous_records:
            return False
        path = output_root / f"{image_id}.jpg"
        if not path.is_file():
            return False
        try:
            with Image.open(path) as image:
                return image.size == expected[image_id]
        except OSError:
            return False

    missing = {image_id for image_id in expected if not valid_existing(image_id)}
    initial_missing = set(missing)
    found: dict[str, dict[str, Any]] = {}
    for config, split in SPLITS:
        if not missing:
            break
        print(f"[HF-GQA] scanning {config}/{split} for {len(missing)} IDs", flush=True)
        observed = _scan_split(config, split, missing, workers)
        found.update(observed)
        missing -= set(observed)
        print(f"[HF-GQA] found={len(observed)} remaining={len(missing)}", flush=True)
    if missing:
        raise ValueError(f"HF GQA mirror misses {len(missing)} FineCops images")

    api = requests.get(API_URL, timeout=60)
    api.raise_for_status()
    api_payload = api.json()
    mirror_id = str(api_payload.get("id"))
    mirror_sha = str(api_payload.get("sha"))
    if mirror_id != DATASET or len(mirror_sha) != 40:
        raise ValueError("HF GQA mirror identity drifted")

    records: dict[str, dict[str, Any]] = {
        key: value for key, value in previous_records.items() if key not in initial_missing
    }
    downloaded_records: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    started = time.time()

    def fetch(image_id: str) -> tuple[str, dict[str, Any]]:
        binding = found[image_id]
        response = requests.get(binding["source_url"], timeout=120)
        response.raise_for_status()
        content = response.content
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            observed_size = image.size
        if observed_size != expected[image_id]:
            raise ValueError(
                f"HF GQA image {image_id} size {observed_size} != {expected[image_id]}"
            )
        destination = output_root / f"{image_id}.jpg"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, destination)
        record = {
            "config": binding["config"],
            "split": binding["split"],
            "row_idx": binding["row_idx"],
            "source_url_sha256": hashlib.sha256(
                binding["source_url"].split("?", 1)[0].encode("utf-8")
            ).hexdigest(),
            "artifact": file_record(destination),
        }
        with lock:
            done = len(downloaded_records) + 1
            if done == 1 or done % 100 == 0:
                print(
                    f"[HF-GQA] downloaded={done}/{len(initial_missing)} "
                    f"elapsed={(time.time() - started) / 60:.1f}m",
                    flush=True,
                )
        return image_id, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for image_id, record in executor.map(fetch, sorted(initial_missing)):
            records[image_id] = record
            downloaded_records[image_id] = record
    if len(downloaded_records) != len(initial_missing):
        raise RuntimeError("HF GQA selective download did not complete")
    payload = {
        "schema": "arrow.finecops.hf_gqa_selective_download/v1",
        "mirror": {
            "id": mirror_id,
            "commit": mirror_sha,
            "api": API_URL,
        },
        "annotation": file_record(annotation_path),
        "initial_missing": len(initial_missing),
        "downloaded": len(downloaded_records),
        "mirror_bound_records": len(records),
        "total_required": len(expected),
        "all_required_present": all(
            (output_root / f"{image_id}.jpg").is_file() for image_id in expected
        ),
        "records_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "records": records,
    }
    write_json_atomic(receipt_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/media/haoyi/T9/data/FineCops-Ref/v1")
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be in [1, 32]")
    payload = download(args.root, workers=args.workers)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
