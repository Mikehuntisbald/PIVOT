#!/usr/bin/env python3
"""Resume a stopped official prefix with bounded, verified HTTP range requests.

Only download artifacts are written. Prefix and completed parts are retained;
the combined output is published only after the official full-file MD5 passes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import urllib.request

URL = "https://zenodo.org/records/4721981/files/refcoco_resnet101_checkpoint.pth?download=1"
SIZE = 2962377847
MD5 = "3219e03af7709cd15ab0d0db521b9070"
CHUNK = 64 << 20


def download_part(item):
    start, end, path = item
    offset = path.stat().st_size if path.exists() else 0
    expected = end - start + 1
    if offset > expected:
        raise ValueError("range part is larger than requested")
    for attempt in range(10):
        offset = path.stat().st_size if path.exists() else 0
        if offset == expected:
            return path
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start+offset}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start+offset}-{end}/{SIZE}":
                    raise ValueError("server did not honor exact byte range")
                with path.open("ab") as target:
                    shutil.copyfileobj(response, target, 1 << 20)
            if path.stat().st_size != expected:
                raise IOError("short range response")
            print(f"completed bytes {start}-{end}", flush=True)
            return path
        except (OSError, TimeoutError) as exc:
            print(f"range {start} retry {attempt+1}: {type(exc).__name__}", flush=True)
    raise RuntimeError(f"range exhausted retries: {start}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parts", type=Path, required=True)
    args = p.parse_args()
    prefix_size = args.prefix.stat().st_size
    if not 0 < prefix_size < SIZE or args.output.exists():
        raise ValueError("requires incomplete stopped prefix and new output")
    args.parts.mkdir(parents=True, exist_ok=True)
    tasks = [(start, min(start+CHUNK, SIZE)-1, args.parts/f"{start:012d}.part") for start in range(prefix_size, SIZE, CHUNK)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        completed = list(pool.map(download_part, tasks))
    if args.prefix.stat().st_size != prefix_size:
        raise ValueError("prefix writer was not stopped")
    assembled = args.output.with_suffix(".assembling")
    digest = hashlib.md5()
    sha = hashlib.sha256()
    with assembled.open("xb") as output:
        for path in [args.prefix, *completed]:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 << 20), b""):
                    output.write(block)
                    digest.update(block)
                    sha.update(block)
    if assembled.stat().st_size != SIZE or digest.hexdigest() != MD5:
        raise ValueError("official assembled MD5 mismatch; artifacts preserved")
    assembled.rename(args.output)
    print(f"verified {args.output} md5={MD5} sha256={sha.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
