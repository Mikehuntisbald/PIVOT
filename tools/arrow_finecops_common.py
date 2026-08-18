#!/usr/bin/env python3
"""Shared contracts for the ARROW FineCops-Ref zero-shot evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


DATASET_SCHEMA = "arrow.finecops.dataset_manifest/v1"
PREREG_SCHEMA = "arrow.finecops.preregistration/v1"
RECORD_SCHEMA = "arrow.finecops.eval_record/v1"
RESULTS_SCHEMA = "arrow.finecops.results/v1"
FINAL_SCHEMA = "arrow.finecops.final_receipt/v1"
OFFICIAL_REPO_COMMIT = "31d2c8615e65ccef6a4ff516925ef5ae465ec747"
BOOTSTRAP_SEED = 20260819
BOOTSTRAP_ITERATIONS = 5000

EXPECTED_MD5 = {
    "test_expression_all.json": "8c61e48736bece7243ac5884b75f8b3c",
    "test_expression_all_coco_format.json": "6ee0ae6d2cbd5056120b9dfecca2b44f",
    "test_expression_pos.json": "c7cd9cf7755f8a66c0307ae1332e3bf7",
    "test_expression_pos_coco_format.json": "cd2eae0c0d8fe582496128c685bf4f17",
    "neg_images.tgz": "2e18b8b81ca75c795a7c3395cfc7029b",
}
EXPECTED_COUNTS = {
    "positive": 9605,
    "text": 9814,
    "image": 8507,
    "records": 27926,
    "original_images": 4313,
    "negative_images": 8507,
    "all_images": 12820,
}


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


def canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


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


def normalize_name(value: str) -> str:
    text = str(value).strip().lower().replace("_", " ").replace(".", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def require_finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def reject_train_or_val_path(path: Path) -> None:
    lowered = str(Path(path).expanduser().resolve()).lower()
    forbidden = ("train-set", "expression_all_train", "expression_pos_train", "_val_set")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"FineCops train/val inputs are forbidden: {path}")
