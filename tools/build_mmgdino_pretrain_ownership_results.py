#!/usr/bin/env python3
"""Seal the lightweight MM-GDINO-T pretrained ownership result receipt."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/mmgdino_pretrain_ownership_20260822"
DESTINATION = ROOT / "paper/data/mmgdino_pretrain_ownership_results.json"
PREREG = ROOT / "paper/data/mmgdino_pretrain_ownership_preregistration.json"
AMENDMENTS = (ROOT / "paper/data/mmgdino_pretrain_ownership_runtime_amendment.json",)
TRUNK = "pretrained"
OWNERS = ("shared_wide", "isolated_128")
SEEDS = (17, 42, 73)
SURFACES = (
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_test",
    "strict2031",
)


class ResultReceiptError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResultReceiptError(f"JSON object required: {path}")
    return value


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def relative_binding(path: Path) -> dict[str, Any]:
    value = binding(path)
    value["path"] = str(path.relative_to(ROOT))
    return value


def checkpoint_bindings() -> list[dict[str, Any]]:
    rows = []
    for owner in OWNERS:
        for seed in SEEDS:
            directory = OUTPUT_ROOT / f"formal/{TRUNK}/{owner}/seed{seed}"
            receipt_path = directory / "training_receipt.json"
            receipt = load_json(receipt_path)
            if receipt.get("status") != "complete":
                raise ResultReceiptError(f"incomplete training: {receipt_path}")
            checkpoint = directory / "checkpoint_u150.pt"
            if receipt["checkpoint"]["sha256"] != sha256(checkpoint):
                raise ResultReceiptError(f"checkpoint drift: {checkpoint}")
            rows.append(
                {
                    "trunk": TRUNK,
                    "owner": owner,
                    "seed": seed,
                    "checkpoint": relative_binding(checkpoint),
                    "receipt": relative_binding(receipt_path),
                    "updates": receipt["updates"],
                    "ownership": receipt["ownership"],
                    "gradient_probe_u150": receipt["gradient_probes"]["150"],
                }
            )
    return rows


def cache_bindings(kind: str) -> list[dict[str, Any]]:
    rows = []
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
        raise ResultReceiptError(f"unknown cache kind: {kind}")
    for label, cache in targets:
        receipt_path = cache.with_name(f"{cache.stem}_receipt.json")
        receipt = load_json(receipt_path)
        if receipt["output"]["file_sha256"] != sha256(cache):
            raise ResultReceiptError(f"cache drift: {cache}")
        rows.append(
            {
                "trunk": TRUNK,
                "surface_or_seed": label,
                "cache": relative_binding(cache),
                "receipt": relative_binding(receipt_path),
                "row_count": receipt["output"]["row_count"],
                "counters": receipt.get("counters"),
            }
        )
    return rows


def evaluation_bindings() -> list[dict[str, Any]]:
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
                value = load_json(summary)
                rows.append(
                    {
                        "trunk": TRUNK,
                        "surface": surface,
                        "route": route,
                        "seed": seed,
                        "summary": relative_binding(summary),
                        "records": relative_binding(records),
                        "metrics": value["metrics"],
                    }
                )
    if len(rows) != 42:
        raise ResultReceiptError("expected exactly 42 evaluation routes")
    return rows


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise ResultReceiptError(f"destination already exists: {DESTINATION}")
    aggregate_path = OUTPUT_ROOT / "aggregate.json"
    aggregate = load_json(aggregate_path)
    if aggregate.get("schema") != "arrow.mmgdino_pretrain_ownership.aggregate/v1":
        raise ResultReceiptError("aggregate schema drifted")
    if aggregate.get("status") != "complete":
        raise ResultReceiptError("aggregate is incomplete")
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    payload = {
        "schema": "arrow.mmgdino_pretrain_ownership.results/v1",
        "status": "complete_negative_result",
        "provenance": {
            "git_commit_before_result_packaging": commit,
            "preregistration": relative_binding(PREREG),
            "amendments": [relative_binding(path) for path in AMENDMENTS],
            "aggregate": relative_binding(aggregate_path),
        },
        "matrix": {
            "trunk": TRUNK,
            "owners": list(OWNERS),
            "seeds": list(SEEDS),
            "fixed_endpoint": "U150",
            "shared_128_ran": False,
            "rank_updates": 100,
            "confidence_updates": 50,
        },
        "artifacts": {
            "training_caches": cache_bindings("training"),
            "formal_trajectories": checkpoint_bindings(),
            "evaluation_caches": cache_bindings("evaluation"),
            "evaluation_routes": evaluation_bindings(),
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
                "the pretrained shared gradients are near zero on average at U150",
                "Shared-Wide has higher pooled Test5 and TestAB REC than Isolated",
                "both learned owner layouts preserve REC within the preregistered margin",
            ],
            "not_supported": [
                "isolated ownership improves REC on the pretrained trunk",
                "isolated ownership improves Strict-TN2031 FPR95 over Shared-Wide",
                "negative gradient probes alone imply a deployment benefit from hard isolation",
            ],
            "observed_rejection_direction": (
                "Shared-Wide has lower mean Strict-TN2031 FPR95 than Isolated, but "
                "the paired confidence interval includes zero"
            ),
        },
    }
    atomic_json(payload, DESTINATION)
    print(json.dumps(binding(DESTINATION), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
