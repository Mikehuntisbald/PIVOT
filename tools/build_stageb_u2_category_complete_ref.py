#!/usr/bin/env python3
"""Build RefCOCO-family rank rows with complete same-category COCO boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot.stageb.u2_category_complete_ref/v1"
RECEIPT_SCHEMA = "pivot.stageb.u2_category_complete_receipt/v1"
SOURCE_NAMES = (
    "refcoco_stageb_phrase_v1.jsonl",
    "refcocoplus_stageb_phrase_v1.jsonl",
    "refcocog_stageb_phrase_v1.jsonl",
)


class CategoryCompleteBuildError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise CategoryCompleteBuildError(f"not a file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved),
    }


def _load_json(path: Path, *, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CategoryCompleteBuildError(f"could not load {label}: {path}: {error}") from error


def _xywh_iou(left: Iterable[Any], right: Iterable[Any]) -> float:
    try:
        lx, ly, lw, lh = (float(value) for value in left)
        rx, ry, rw, rh = (float(value) for value in right)
    except (TypeError, ValueError) as error:
        raise CategoryCompleteBuildError("bbox must contain four finite numbers") from error
    if min(lw, lh, rw, rh) <= 0.0:
        raise CategoryCompleteBuildError("bbox width and height must be positive")
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0.0 else 0.0


def _coco_split(filename: Any) -> str:
    name = Path(str(filename or "")).name
    for split in ("train2014", "val2014"):
        if f"_{split}_" in name:
            return split
    raise CategoryCompleteBuildError(f"cannot infer COCO split from filename={filename!r}")


def build_coco_index(
    annotation_paths: Mapping[str, Path],
) -> tuple[dict[tuple[str, int, int], list[dict[str, Any]]], dict[str, Any]]:
    index: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    records: dict[str, Any] = {}
    for split, path in annotation_paths.items():
        payload = _load_json(path, label=f"COCO {split}")
        annotations = payload.get("annotations") if isinstance(payload, dict) else None
        if not isinstance(annotations, list):
            raise CategoryCompleteBuildError(f"COCO {split} has no annotations list")
        kept = 0
        for annotation in annotations:
            if not isinstance(annotation, dict) or int(annotation.get("iscrowd", 0)) != 0:
                continue
            try:
                image_id = int(annotation["image_id"])
                category_id = int(annotation["category_id"])
                ann_id = int(annotation["id"])
                bbox = [float(value) for value in annotation["bbox"]]
            except (KeyError, TypeError, ValueError) as error:
                raise CategoryCompleteBuildError(
                    f"malformed COCO {split} annotation"
                ) from error
            if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
                continue
            index[(split, image_id, category_id)].append(
                {"ann_id": ann_id, "bbox": bbox}
            )
            kept += 1
        records[split] = {
            **_file_record(path),
            "noncrowd_annotations": kept,
        }
    for annotations in index.values():
        annotations.sort(key=lambda item: int(item["ann_id"]))
    return dict(index), records


def enrich_row(
    row: Mapping[str, Any],
    coco_index: Mapping[tuple[str, int, int], list[dict[str, Any]]],
    *,
    context: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    instances = row.get("instances")
    if not isinstance(instances, list) or len(instances) != 1 or not isinstance(instances[0], dict):
        raise CategoryCompleteBuildError(f"{context}: expected exactly one source instance")
    primary = dict(instances[0])
    try:
        filename = row["filename"]
        split = _coco_split(filename)
        image_id = int(row["image_id"])
        target_ann_id = int(row["ann_id"])
        class_id = int(primary["class_id"])
        category_id = int(primary["refcoco_category_id"])
        target_bbox = [float(value) for value in primary["bbox"]]
    except (KeyError, TypeError, ValueError) as error:
        raise CategoryCompleteBuildError(f"{context}: malformed Ref row identity") from error
    annotations = coco_index.get((split, image_id, category_id), [])
    if not annotations:
        raise CategoryCompleteBuildError(
            f"{context}: no COCO instances for {(split, image_id, category_id)!r}"
        )
    target = next(
        (annotation for annotation in annotations if int(annotation["ann_id"]) == target_ann_id),
        None,
    )
    if target is None:
        raise CategoryCompleteBuildError(
            f"{context}: target ann_id={target_ann_id} missing from COCO index"
        )
    target_iou = _xywh_iou(target_bbox, target["bbox"])
    if target_iou < 0.99:
        raise CategoryCompleteBuildError(
            f"{context}: source/COCO target bbox drifted (IoU={target_iou:.6f})"
        )

    primary["category_complete_primary"] = True
    primary["coco_ann_id"] = target_ann_id
    enriched_instances = [primary]
    for annotation in annotations:
        ann_id = int(annotation["ann_id"])
        if ann_id == target_ann_id:
            continue
        enriched_instances.append(
            {
                "bbox": list(annotation["bbox"]),
                "class_id": class_id,
                "refcoco_category_id": category_id,
                "coco_ann_id": ann_id,
                "category_complete_auxiliary": True,
            }
        )

    output = dict(row)
    output.update(
        {
            "instances": enriched_instances,
            "primary_support_instance_index": 0,
            "stage_b_u2_category_complete": True,
            "stage_b_u2_category_complete_schema": SCHEMA,
            "category_complete_coco_split": split,
            "category_complete_coco_category_id": category_id,
            "category_complete_instance_count": len(enriched_instances),
        }
    )
    return output, {
        "instances": len(enriched_instances),
        "auxiliary_instances": len(enriched_instances) - 1,
        "multi_instance_row": int(len(enriched_instances) > 1),
    }


def build_manifest(
    source: Path,
    destination: Path,
    coco_index: Mapping[tuple[str, int, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    max_instances = 0
    temp = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    if destination.exists():
        raise CategoryCompleteBuildError(f"refusing to replace existing output: {destination}")
    try:
        with source.open("r", encoding="utf-8") as input_handle, temp.open(
            "x", encoding="utf-8"
        ) as output_handle:
            for line_number, line in enumerate(input_handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CategoryCompleteBuildError(
                        f"invalid JSON at {source}:{line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise CategoryCompleteBuildError(
                        f"expected object at {source}:{line_number}"
                    )
                enriched, row_counts = enrich_row(
                    row, coco_index, context=f"{source}:{line_number}"
                )
                counts.update(row_counts)
                counts["rows"] += 1
                max_instances = max(max_instances, row_counts["instances"])
                output_handle.write(
                    json.dumps(
                        enriched,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()
    if counts["rows"] <= 0:
        raise CategoryCompleteBuildError(f"source is empty: {source}")
    return {
        "source": _file_record(source),
        "output": _file_record(destination),
        "rows": int(counts["rows"]),
        "instances": int(counts["instances"]),
        "auxiliary_instances": int(counts["auxiliary_instances"]),
        "multi_instance_rows": int(counts["multi_instance_row"]),
        "max_instances_per_row": int(max_instances),
    }


def build_all(
    *,
    input_dir: Path,
    output_dir: Path,
    train2014: Path,
    val2014: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipt.json"
    if receipt_path.exists():
        raise CategoryCompleteBuildError(f"refusing to replace existing receipt: {receipt_path}")
    coco_index, coco_records = build_coco_index(
        {"train2014": train2014, "val2014": val2014}
    )
    manifests = {}
    for name in SOURCE_NAMES:
        manifests[name] = build_manifest(
            input_dir / name, output_dir / name, coco_index
        )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "row_schema": SCHEMA,
        "coco_annotations": coco_records,
        "manifests": manifests,
        "invariants": {
            "source_expression_instance_preserved_at_index_zero": True,
            "all_noncrowd_same_category_coco_instances_included": True,
            "primary_support_instance_index_zero": True,
            "target_annotation_matched_by_ann_id_and_iou_at_least_0_99": True,
        },
    }
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    receipt["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    temp = receipt_path.with_name(receipt_path.name + f".tmp-{os.getpid()}")
    try:
        with temp.open("x", encoding="ascii") as handle:
            json.dump(receipt, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, receipt_path)
    finally:
        if temp.exists():
            temp.unlink()
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data/ablations/stageb_refexp_three_train_20260711",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "data/ablations/stageb_refexp_three_train_category_complete_20260720",
    )
    parser.add_argument("--coco-train2014", type=Path, required=True)
    parser.add_argument("--coco-val2014", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train2014=args.coco_train2014,
        val2014=args.coco_val2014,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
