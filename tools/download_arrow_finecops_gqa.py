#!/usr/bin/env python3
"""Resumable parallel Range downloader for the official GQA image archive."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import file_record, write_json_atomic


URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
EXPECTED_SIZE = 21_817_965_542
EXPECTED_ETAG = '"5c55f4cb-51473bbe6"'


def download(output: Path, *, workers: int, chunk_size: int) -> dict:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".parallel.partial")
    state_path = output.with_name(output.name + ".parallel.state.json")
    receipt_path = output.with_name(output.name + ".download_receipt.json")
    head = requests.head(URL, timeout=60)
    head.raise_for_status()
    size = int(head.headers.get("Content-Length", -1))
    etag = str(head.headers.get("ETag", ""))
    if size != EXPECTED_SIZE or etag != EXPECTED_ETAG:
        raise ValueError(f"GQA origin contract drifted: size={size}, etag={etag!r}")
    if "bytes" not in str(head.headers.get("Accept-Ranges", "")).lower():
        raise ValueError("GQA origin no longer advertises byte ranges")
    chunks = [
        (index, start, min(size - 1, start + chunk_size - 1))
        for index, start in enumerate(range(0, size, chunk_size))
    ]
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("url") != URL or state.get("size_bytes") != size or state.get("chunk_size") != chunk_size:
            raise ValueError("parallel download state contract drifted")
        complete = {int(value) for value in state.get("completed_chunks", [])}
    else:
        complete = set()
    descriptor = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(descriptor, size)
        lock = threading.Lock()
        started = time.time()

        def persist() -> None:
            write_json_atomic(
                state_path,
                {
                    "schema": "arrow.finecops.parallel_download/v1",
                    "url": URL,
                    "size_bytes": size,
                    "etag": etag,
                    "chunk_size": chunk_size,
                    "completed_chunks": sorted(complete),
                    "total_chunks": len(chunks),
                },
            )

        persist()

        def fetch(item: tuple[int, int, int]) -> int:
            index, start, end = item
            if index in complete:
                return index
            expected = end - start + 1
            for attempt in range(6):
                try:
                    response = requests.get(
                        URL,
                        headers={"Range": f"bytes={start}-{end}", "If-Range": etag},
                        stream=True,
                        timeout=(30, 180),
                    )
                    if response.status_code != 206:
                        raise RuntimeError(f"range {index} returned HTTP {response.status_code}")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError(f"range {index} Content-Range drifted")
                    offset = start
                    received = 0
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        os.pwrite(descriptor, block, offset)
                        offset += len(block)
                        received += len(block)
                    if received != expected:
                        raise RuntimeError(
                            f"range {index} length {received} != {expected}"
                        )
                    os.fsync(descriptor)
                    with lock:
                        complete.add(index)
                        persist()
                        done_bytes = sum(
                            chunks[value][2] - chunks[value][1] + 1 for value in complete
                        )
                        elapsed = max(1e-6, time.time() - started)
                        print(
                            f"[GQA] chunks={len(complete)}/{len(chunks)} "
                            f"bytes={done_bytes}/{size} rate={done_bytes / elapsed / 1024**2:.1f}MiB/s",
                            flush=True,
                        )
                    return index
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt)
            raise AssertionError("unreachable")

        pending = [item for item in chunks if item[0] not in complete]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(fetch, pending))
        if len(complete) != len(chunks):
            raise RuntimeError("parallel GQA download ended with incomplete ranges")
    finally:
        os.close(descriptor)
    os.replace(partial, output)
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    receipt = {
        "schema": "arrow.finecops.gqa_download_receipt/v1",
        "source_url": URL,
        "content_length": size,
        "etag": etag,
        "accept_ranges": "bytes",
        "sha256": digest.hexdigest(),
        "artifact": file_record(output),
        "workers": workers,
        "chunk_size": chunk_size,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/media/haoyi/T9/data/FineCops-Ref/v1/raw/gqa/images.zip"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size-mib", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be in [1, 32]")
    result = download(
        args.output,
        workers=args.workers,
        chunk_size=args.chunk_size_mib * 1024 * 1024,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
