#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.stage_b_dense_duty_audit import (
    audit_checkpoint_payload,
    write_json_atomic,
)


def _load_checkpoint(path: Path):
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    audit = audit_checkpoint_payload(
        _load_checkpoint(checkpoint), checkpoint_path=checkpoint
    )
    if args.output is not None:
        write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
