#!/usr/bin/env python3
"""Download pinned official MDETR assets, never overwrite an existing seal."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = "ea09acc44ca067072c4b143b726447ee7ff66f5f"
URL = "https://zenodo.org/records/4721981/files/refcoco_resnet101_checkpoint.pth"
MD5 = "3219e03af7709cd15ab0d0db521b9070"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    args = p.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    if not args.upstream.exists():
        subprocess.run(["git", "clone", "https://github.com/ashkamath/mdetr.git", str(args.upstream)], check=True)
        subprocess.run(["git", "-C", str(args.upstream), "checkout", "--detach", COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {actual}")
    status = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
    if status:
        raise RuntimeError("MDETR tracked upstream source is dirty")
    path = args.assets / "refcoco_resnet101_checkpoint.pth"
    if not path.exists():
        partial = args.assets / "refcoco_resnet101_checkpoint.pth.partial"
        subprocess.run(["wget", "-c", "--progress=dot:giga", "--timeout=60", "--tries=20", "-O", str(partial), URL], check=True)
        if digest(partial, "md5") != MD5:
            raise RuntimeError("official checkpoint MD5 mismatch; partial preserved")
        partial.rename(path)
    if digest(path, "md5") != MD5:
        raise RuntimeError("official checkpoint MD5 mismatch")
    receipt = {
        "schema": "arrow.confidence_readout.mdetr_assets/v1",
        "upstream_commit": COMMIT,
        "upstream": str(args.upstream.resolve()),
        "checkpoint": {"path": str(path.resolve()), "bytes": path.stat().st_size,
                       "sha256": digest(path), "md5": MD5, "url": URL},
        "required_state_key": "model_ema", "state_key_verified": False,
        "status": "download_verified_runtime_preflight_required",
    }
    out = args.assets / "download_receipt.json"
    if out.exists():
        if json.loads(out.read_text()) != receipt:
            raise RuntimeError("refuse to replace differing asset receipt")
    else:
        with out.open("x") as f:
            json.dump(receipt, f, indent=2)
            f.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
