#!/usr/bin/env python3
"""Seal the original GroundingDINO pre-Stage-B ownership result receipt."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/original_gdino_parent_ownership_20260822"
DESTINATION = ROOT / "paper/data/original_gdino_parent_ownership_results.json"
PREREG = ROOT / "paper/data/original_gdino_parent_ownership_preregistration.json"
AMENDMENT = ROOT / "paper/data/original_gdino_parent_ownership_runtime_amendment.json"
TRUNK = "original_parent"
OWNERS = ("shared_wide", "isolated_128")
SEEDS = (17, 42, 73)
SURFACES = (
    "refcoco_testA", "refcoco_testB", "refcocop_testA",
    "refcocop_testB", "refcocog_test", "strict2031",
)


class ReceiptError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReceiptError(f"JSON object required: {path}")
    return value


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def caches(kind: str) -> list[dict[str, Any]]:
    if kind == "training":
        targets = (
            (f"seed{seed}", OUTPUT_ROOT / f"caches/{TRUNK}/seed{seed}.pt")
            for seed in SEEDS
        )
    elif kind == "evaluation":
        targets = (
            (surface, OUTPUT_ROOT / f"evaluation_caches/{TRUNK}/{surface}.pt")
            for surface in SURFACES
        )
    else:
        raise ReceiptError("unknown cache kind")
    rows = []
    for label, path in targets:
        receipt_path = path.with_name(f"{path.stem}_receipt.json")
        receipt = load(receipt_path)
        if receipt.get("status") != "complete":
            raise ReceiptError(f"incomplete cache: {path}")
        if receipt["output"]["file_sha256"] != sha256(path):
            raise ReceiptError(f"cache drift: {path}")
        rows.append({
            "surface_or_seed": label,
            "cache": binding(path),
            "receipt": binding(receipt_path),
            "row_count": receipt["output"]["row_count"],
            "counters": receipt.get("counters"),
            "oracle": receipt.get("oracle"),
        })
    return rows


def trajectories() -> list[dict[str, Any]]:
    rows = []
    for owner in OWNERS:
        for seed in SEEDS:
            directory = OUTPUT_ROOT / f"formal/{TRUNK}/{owner}/seed{seed}"
            receipt_path = directory / "training_receipt.json"
            receipt = load(receipt_path)
            checkpoint = directory / "checkpoint_u150.pt"
            if receipt.get("status") != "complete":
                raise ReceiptError(f"incomplete trajectory: {directory}")
            if receipt["checkpoint"]["sha256"] != sha256(checkpoint):
                raise ReceiptError(f"checkpoint drift: {checkpoint}")
            rows.append({
                "owner": owner,
                "seed": seed,
                "checkpoint": binding(checkpoint),
                "receipt": binding(receipt_path),
                "updates": receipt["updates"],
                "ownership": receipt["ownership"],
                "gradient_probe_u150": receipt["gradient_probes"]["150"],
            })
    return rows


def evaluations() -> list[dict[str, Any]]:
    rows = []
    for surface in SURFACES:
        for route in ("native", *OWNERS):
            seeds = (None,) if route == "native" else SEEDS
            for seed in seeds:
                directory = OUTPUT_ROOT / f"evaluation/{TRUNK}/{surface}/{route}"
                if seed is not None:
                    directory /= f"seed{seed}"
                summary = directory / "summary.json"
                records = directory / "records.jsonl"
                value = load(summary)
                rows.append({
                    "surface": surface,
                    "route": route,
                    "seed": seed,
                    "summary": binding(summary),
                    "records": binding(records),
                    "metrics": value["metrics"],
                })
    if len(rows) != 42:
        raise ReceiptError("expected exactly 42 evaluation routes")
    return rows


def atomic(value: Mapping[str, Any], path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    if DESTINATION.exists():
        raise ReceiptError("result receipt already exists")
    aggregate_path = OUTPUT_ROOT / "aggregate.json"
    aggregate = load(aggregate_path)
    if (
        aggregate.get("schema")
        != "arrow.original_gdino_parent_ownership.aggregate/v1"
        or aggregate.get("status") != "complete"
    ):
        raise ReceiptError("aggregate is incomplete")
    payload = {
        "schema": "arrow.original_gdino_parent_ownership.results/v1",
        "status": "complete_negative_isolation_result",
        "provenance": {
            "git_commit_before_packaging": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "preregistration": binding(PREREG),
            "runtime_amendment": binding(AMENDMENT),
            "aggregate": binding(aggregate_path),
        },
        "matrix": {
            "trunk": TRUNK,
            "owners": list(OWNERS),
            "seeds": list(SEEDS),
            "fixed_endpoint": "U150",
            "shared_128_ran": False,
        },
        "artifacts": {
            "training_caches": caches("training"),
            "formal_trajectories": trajectories(),
            "evaluation_caches": caches("evaluation"),
            "evaluation_routes": evaluations(),
        },
        "statistics": {
            "bootstrap": aggregate["bootstrap"],
            "point_metrics": aggregate["point_metrics"],
            "isolated_minus_shared_wide": aggregate["isolated_minus_shared_wide"],
            "gradient_probes": aggregate["gradient_probes"],
            "claim_gate": aggregate["claim_gate"],
            "references": aggregate["references"],
        },
        "interpretation": {
            "supported": [
                "Shared-Wide improves Test5 and TestAB over Isolated on the direct pre-Stage-B parent",
                "the shared gradient geometry has a negative lower tail despite positive mean cosine",
                "Isolated has no cross-task autograd path",
            ],
            "not_supported": [
                "hard isolation is REC-noninferior on the direct parent",
                "hard isolation improves Strict-TN2031 FPR95",
                "the mature C50 rejection owner improves FPR95 over the native parent score",
                "this result and the existing B58 capacity block alone form a strict same-head causal contrast",
            ],
            "head_mismatch_caveat": (
                "this replay uses the 100k raw-query owner; the existing B58 capacity "
                "block uses an 84k integrated adapter owner"
            ),
        },
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    atomic(payload, DESTINATION)
    print(json.dumps(binding(DESTINATION), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
