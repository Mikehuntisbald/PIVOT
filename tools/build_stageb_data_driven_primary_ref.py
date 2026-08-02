#!/usr/bin/env python3
"""Seal paired DD0 ordinary-primary and DD1 category-complete manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import zip_longest
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot.stageb.data_driven_ordinary_primary/v1"
RECEIPT_SCHEMA = "pivot.stageb.data_driven_ref_pair_receipt/v1"
SOURCE_ROOT = REPO_ROOT / "data/ablations/stageb_refexp_three_train_20260711"
COMPLETE_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_refexp_three_train_category_complete_20260720"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_ref_pair_20260721/dd0_ordinary_primary"
)
MANIFESTS = (
    "refcoco_stageb_phrase_v1.jsonl",
    "refcocoplus_stageb_phrase_v1.jsonl",
    "refcocog_stageb_phrase_v1.jsonl",
)
IDENTITY_KEYS = (
    "source",
    "image_id",
    "ann_id",
    "ref_id",
    "sent_id",
    "split",
    "filename",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_row(raw: str, *, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except Exception as error:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {path}:{line_number}")
    return value


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in IDENTITY_KEYS)


def _source_primary(row: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    instances = row.get("instances")
    if not (
        isinstance(instances, list)
        and len(instances) == 1
        and isinstance(instances[0], dict)
    ):
        raise ValueError(f"{context}: ordinary source must contain one instance")
    return dict(instances[0])


def _complete_primary(row: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    instances = row.get("instances")
    if not (
        isinstance(instances, list)
        and instances
        and all(isinstance(instance, dict) for instance in instances)
        and row.get("primary_support_instance_index") == 0
        and row.get("stage_b_u2_category_complete") is True
        and row.get("stage_b_u2_category_complete_schema")
        == "pivot.stageb.u2_category_complete_ref/v1"
    ):
        raise ValueError(f"{context}: category-complete row contract drifted")
    primary = dict(instances[0])
    if primary.pop("category_complete_primary", None) is not True:
        raise ValueError(f"{context}: complete primary marker drifted")
    primary.pop("coco_ann_id", None)
    if any(
        instance.get("class_id") != instances[0].get("class_id")
        for instance in instances
    ):
        raise ValueError(f"{context}: complete row contains another category")
    return primary


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve(strict=True)
    complete_root = Path(args.complete_root).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    identity_digest = hashlib.sha256()
    primary_digest = hashlib.sha256()
    source_row_digest = hashlib.sha256()
    seen_identities = set()
    manifests: dict[str, Any] = {}
    total_rows = 0
    total_complete_instances = 0

    for name in MANIFESTS:
        source_path = source_root / name
        complete_path = complete_root / name
        output_path = output_root / name
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        rows = 0
        complete_instances = 0
        try:
            with source_path.open("r", encoding="utf-8") as source_handle, complete_path.open(
                "r", encoding="utf-8"
            ) as complete_handle, temporary.open("w", encoding="utf-8") as output_handle:
                for line_number, pair in enumerate(
                    zip_longest(source_handle, complete_handle), start=1
                ):
                    source_raw, complete_raw = pair
                    if source_raw is None or complete_raw is None:
                        raise ValueError(f"{name}: source/complete row counts differ")
                    source = _load_row(
                        source_raw, path=source_path, line_number=line_number
                    )
                    complete = _load_row(
                        complete_raw, path=complete_path, line_number=line_number
                    )
                    context = f"{name}:{line_number}"
                    identity = _identity(source)
                    if None in identity or identity != _identity(complete):
                        raise ValueError(f"{context}: paired identity drifted")
                    if identity in seen_identities:
                        raise ValueError(f"{context}: duplicate global identity")
                    seen_identities.add(identity)
                    source_primary = _source_primary(source, context=context)
                    if source_primary != _complete_primary(complete, context=context):
                        raise ValueError(f"{context}: complete primary differs from source")

                    source_row_sha256 = hashlib.sha256(
                        _canonical_bytes(source)
                    ).hexdigest()
                    ordinary = dict(source)
                    ordinary.update(
                        {
                            "primary_support_instance_index": 0,
                            "stage_b_data_driven_ordinary_primary": True,
                            "stage_b_data_driven_ordinary_primary_schema": SCHEMA,
                            "stage_b_data_driven_source_row_sha256": source_row_sha256,
                        }
                    )
                    output_handle.write(
                        json.dumps(
                            ordinary,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    identity_digest.update(_canonical_bytes(identity) + b"\n")
                    primary_digest.update(_canonical_bytes(source_primary) + b"\n")
                    source_row_digest.update(source_row_sha256.encode("ascii") + b"\n")
                    rows += 1
                    complete_instances += len(complete["instances"])
            os.replace(temporary, output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        manifests[name] = {
            "rows": rows,
            "complete_instances": complete_instances,
            "source": _file_record(source_path),
            "ordinary_primary": _file_record(output_path),
            "category_complete": _file_record(complete_path),
        }
        total_rows += rows
        total_complete_instances += complete_instances

    if total_rows != int(args.expected_rows):
        raise ValueError(
            f"expected {int(args.expected_rows)} rows, observed {total_rows}"
        )
    if total_complete_instances != int(args.expected_complete_instances):
        raise ValueError(
            "category-complete instance count drifted: "
            f"{total_complete_instances}"
        )
    complete_receipt_path = complete_root / "receipt.json"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "ordinary_row_schema": SCHEMA,
        "rows": total_rows,
        "category_complete_instances": total_complete_instances,
        "unique_identities": len(seen_identities),
        "ordered_identity_stream_sha256": identity_digest.hexdigest(),
        "source_primary_stream_sha256": primary_digest.hexdigest(),
        "source_row_sha256_stream_sha256": source_row_digest.hexdigest(),
        "category_complete_receipt": _file_record(complete_receipt_path),
        "manifests": manifests,
        "invariants": {
            "ordinary_instances_are_source_instances_unchanged": True,
            "ordinary_and_complete_rows_share_identity_and_order": True,
            "ordinary_and_complete_primary_instance_matches_source": True,
            "primary_support_instance_index_is_zero": True,
            "ordinary_has_exactly_one_box": True,
            "complete_contains_only_primary_category": True,
        },
    }
    receipt_path = output_root.parent / "receipt.json"
    temporary_receipt = receipt_path.with_suffix(".json.tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_receipt, receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--complete-root", default=str(COMPLETE_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--expected-rows", type=int, default=321327)
    parser.add_argument("--expected-complete-instances", type=int, default=1361554)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
