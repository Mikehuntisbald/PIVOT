#!/usr/bin/env python3
"""Fetch only pinned RoBERTa tokenizer/config; EMA supplies every model weight."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in FILES:
        dest = args.output / name
        url = f"https://huggingface.co/FacebookAI/roberta-base/resolve/{REVISION}/{name}"
        if not dest.exists():
            tmp = dest.with_suffix(dest.suffix + ".partial")
            subprocess.run(["wget", "-c", "--timeout=60", "--tries=10", "-O", str(tmp), url], check=True)
            tmp.rename(dest)
        rows.append({"name": name, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "bytes": dest.stat().st_size, "url": url})
    receipt = {"schema": "arrow.confidence_readout.mdetr_text_assets/v1", "revision": REVISION, "repo": "FacebookAI/roberta-base", "files": rows, "model_weights_downloaded": False}
    out = args.output / "receipt.json"
    if out.exists() and json.loads(out.read_text()) != receipt:
        raise RuntimeError("text asset receipt drift")
    if not out.exists():
        with out.open("x") as f:
            json.dump(receipt, f, indent=2)
            f.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
