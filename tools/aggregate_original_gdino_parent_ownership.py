#!/usr/bin/env python3
"""Aggregate the original GroundingDINO pre-Stage-B ownership replay."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import tools.aggregate_mmgdino_pretrain_ownership as mature
from tools.original_gdino_parent_ownership import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    B58_REFERENCE,
    E5_REFERENCE,
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    OWNERS,
    REC_NONINFERIORITY_MARGIN,
    REF_INPUTS,
    TEST5_SURFACES,
    TESTAB_SURFACES,
    TRUNK_ID,
)


SCHEMA = "arrow.original_gdino_parent_ownership.aggregate/v1"


@contextlib.contextmanager
def _context():
    replacements = {
        "BOOTSTRAP_REPLICATES": BOOTSTRAP_REPLICATES,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "B58_REFERENCE": B58_REFERENCE,
        "E5_REFERENCE": E5_REFERENCE,
        "FORMAL_SEEDS": FORMAL_SEEDS,
        "OWNERS": OWNERS,
        "SHARED": OWNERS[0],
        "ISOLATED": OWNERS[1],
        "REC_NONINFERIORITY_MARGIN": REC_NONINFERIORITY_MARGIN,
        "REF_INPUTS": REF_INPUTS,
        "TEST5_SURFACES": TEST5_SURFACES,
        "TESTAB_SURFACES": TESTAB_SURFACES,
        "TRUNK_ID": TRUNK_ID,
        "SCHEMA": SCHEMA,
    }
    previous = {name: getattr(mature, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(mature, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(mature, name, value)


def aggregate(
    *, evaluation_root: Path, formal_root: Path, output: Path
) -> dict[str, Any]:
    with _context():
        return mature.aggregate(
            evaluation_root=evaluation_root,
            formal_root=formal_root,
            output=output,
        )


def main() -> None:
    value = aggregate(
        evaluation_root=EXPERIMENT_ROOT / "evaluation",
        formal_root=EXPERIMENT_ROOT / "formal",
        output=EXPERIMENT_ROOT / "aggregate.json",
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "aggregate"]
