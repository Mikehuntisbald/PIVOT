#!/usr/bin/env python3
"""Build deterministic, zero-update rank-probe manifests for RefCOCO variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 73)
PROBE_BATCHES = 8
PROBE_BATCH_SIZE = 32
ROWS_PER_SEED = PROBE_BATCHES * PROBE_BATCH_SIZE
SCHEMA = "arrow.mmgdino_e5_cross_dataset_probe.selection/v1"
RECEIPT_SCHEMA = "arrow.mmgdino_e5_cross_dataset_probe.selection_receipt/v1"

DATASETS: dict[str, dict[str, str]] = {
    "refcoco": {
        "path": "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs/refcoco_unc_val.jsonl",
        "sha256": "ac1ab43019a03dcc65ba3530469b6dcb2ac01be836b795ae5a3b1bdb56b6431d",
    },
    "refcocoplus": {
        "path": "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs/refcocoplus_unc_val.jsonl",
        "sha256": "1eef48a64e7c118b736aa6d383d164ff70af3504285a2cb43a34c02631b5f6de",
    },
    "refcocog": {
        "path": "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs/refcocog_umd_val.jsonl",
        "sha256": "6a21fccf3d2330aaf72a3ee16cd1863f29470abc3ebfa64d098c04cf7d10e925",
    },
}


class ProbeSelectionError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(row: Mapping[str, Any]) -> str:
    values = []
    for name in ("source", "image_id", "ann_id", "ref_id", "sent_id"):
        value = row.get(name)
        if name == "source":
            if not isinstance(value, str) or not value.strip():
                raise ProbeSelectionError("source must be a nonempty string")
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProbeSelectionError(f"{name} must be a nonnegative integer")
        values.append(value)
    return f"refcoco:{values[0]}:{values[1]}:{values[2]}:{values[3]}:{values[4]}"


def _read_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if file_sha256(path) != expected_sha256:
        raise ProbeSelectionError(f"source manifest SHA drifted: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith("\n") or not raw.strip():
                raise ProbeSelectionError(f"malformed JSONL line {line_number}: {path}")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ProbeSelectionError(f"row {line_number} is not an object")
            rows.append(row)
    identities = [sample_id(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise ProbeSelectionError(f"duplicate sample identity in {path}")
    if len(rows) < ROWS_PER_SEED:
        raise ProbeSelectionError(f"source manifest is too small: {path}")
    return rows


def select_schedule(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_identity = {sample_id(row): dict(row) for row in rows}
    identities = tuple(by_identity)
    batches: dict[str, list[list[str]]] = {}
    for seed in SEEDS:
        ordered = sorted(
            identities,
            key=lambda identity: hashlib.sha256(
                f"{seed}\0{identity}".encode("utf-8")
            ).hexdigest(),
        )
        selected = ordered[:ROWS_PER_SEED]
        batches[str(seed)] = [
            selected[start : start + PROBE_BATCH_SIZE]
            for start in range(0, ROWS_PER_SEED, PROBE_BATCH_SIZE)
        ]
    union = sorted({identity for seed_batches in batches.values() for batch in seed_batches for identity in batch})
    return {
        "schema": SCHEMA,
        "seeds": list(SEEDS),
        "probe_batches": PROBE_BATCHES,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "batches": batches,
        "union_identities": union,
        "union_rows": [by_identity[identity] for identity in union],
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    receipt_path = output_dir / "selection_receipt.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProbeSelectionError(f"selection output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name, binding in DATASETS.items():
        source_path = (ROOT / binding["path"]).resolve(strict=True)
        rows = _read_rows(source_path, binding["sha256"])
        selection = select_schedule(rows)
        manifest_path = output_dir / f"{name}_selected.jsonl"
        schedule_path = output_dir / f"{name}_schedule.json"
        _atomic_text(
            manifest_path,
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in selection.pop("union_rows")
            ),
        )
        _atomic_text(
            schedule_path,
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
        )
        datasets[name] = {
            "source": {
                "path": str(source_path),
                "sha256": binding["sha256"],
                "rows": len(rows),
            },
            "selected_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "rows": len(selection["union_identities"]),
            },
            "schedule": {
                "path": str(schedule_path),
                "sha256": file_sha256(schedule_path),
            },
        }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete_before_model_forward",
        "selection": {
            "method": "ascending sha256(seed + NUL + stable sample identity)",
            "seeds": list(SEEDS),
            "probe_batches_per_seed": PROBE_BATCHES,
            "rank_batch_size": PROBE_BATCH_SIZE,
            "rows_per_seed": ROWS_PER_SEED,
            "split": "validation mechanism-only",
        },
        "datasets": datasets,
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
