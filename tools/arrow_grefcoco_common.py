#!/usr/bin/env python3
"""Shared fail-closed contracts for the ARROW gRefCOCO transfer audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


DATASET_SCHEMA = "arrow.grefcoco.dataset_manifest/v1"
OVERLAP_SCHEMA = "arrow.grefcoco.overlap_audit/v1"
PREREG_SCHEMA = "arrow.grefcoco.preregistration/v1"
RECORD_SCHEMA = "arrow.grefcoco.eval_record/v1"
RESULTS_SCHEMA = "arrow.grefcoco.results/v1"
FINAL_SCHEMA = "arrow.grefcoco.final_receipt/v1"
RUN_SCHEMA = "arrow.grefcoco.run_receipt/v1"

HF_REVISION = "81eede59b3ac070049f597d023c0ff08d1fb80e9"
EXPECTED_SHA256 = {
    "grefs(unc).json": "cc37c5ff95373c78a6a3f98b4c7bc67fde387ea8514752a1392db64223eb3366",
    "instances.json": "7f86adb7f4c39db19c7f02a3cc604d17f8016e4c4e8312652d0acc4eae3a2cbd",
}
EXPECTED_TEST_COUNTS = {
    "testA": {"positive": 5917, "negative": 4448, "multi": 8835, "images": 750},
    "testB": {"positive": 5646, "negative": 4673, "multi": 5744, "images": 750},
}
EXPECTED_VAL_COUNTS = {"negative": 8905, "multi": 5324}
SEEDS = (17, 42, 73)
SEALED_THRESHOLDS = {
    17: 0.2301006317138672,
    42: 0.31912317872047424,
    73: 0.753686785697937,
}
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_ITERATIONS = 5000


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": digest_file(resolved),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def write_json_atomic(path: Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + "\n"
            )
            count += 1
    os.replace(temporary, output)
    return count


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def require_finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def verify_record(expected: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(expected.get("path", ""))).expanduser().resolve(strict=True)
    observed = file_record(path)
    for key in ("sha256", "size_bytes"):
        if observed[key] != expected.get(key):
            raise ValueError(f"{label} {key} drifted")
    return path


def reject_train_surface(split: str) -> None:
    if str(split).lower() == "train":
        raise ValueError("gRefCOCO train rows are forbidden in evaluation manifests")
