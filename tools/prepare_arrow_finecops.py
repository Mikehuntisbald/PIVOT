#!/usr/bin/env python3
"""Prepare the immutable FineCops-Ref test surface used by ARROW."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
import tarfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import (
    DATASET_SCHEMA,
    EXPECTED_COUNTS,
    EXPECTED_MD5,
    canonical_json_sha256,
    digest_file,
    file_record,
    load_json,
    normalize_name,
    reject_train_or_val_path,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_ROOT = Path("/media/haoyi/T9/data/FineCops-Ref/v1")
DEFAULT_CANONICAL = Path("/media/haoyi/T9/data/canonical_classes_with_aliases.json")
DEFAULT_SUPPORT_CACHE = Path(
    "/media/haoyi/T9/data/patches_quality_emb/"
    "emb_index_from_quality.tsv.bank.clean.img.pkl"
)
DEFAULT_VG_METADATA = Path(
    "/media/haoyi/T9/data/visual_genome_metadata/image_data.json"
)
DEFAULT_COCO_ROOT = Path("/media/haoyi/T9/data/COCO/coco2017")
SUPPORT_SALT = "arrow-finecops-v1-support-20260819"
DUMMY_CLASS_ID = 782


def _raw_paths(root: Path) -> dict[str, Path]:
    benchmark = root / "raw" / "benchmark"
    return {
        "test_expression_all.json": benchmark / "test_expression_all.json",
        "test_expression_all_coco_format.json": benchmark
        / "test_expression_all_coco_format.json",
        "test_expression_pos.json": benchmark / "test_expression_pos.json",
        "test_expression_pos_coco_format.json": benchmark
        / "test_expression_pos_coco_format.json",
        "neg_images.tgz": root / "raw" / "neg_images.tgz",
    }


def verify_raw(root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, path in _raw_paths(root).items():
        reject_train_or_val_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        md5 = digest_file(path, "md5")
        if md5 != EXPECTED_MD5[name]:
            raise ValueError(f"{name}: MD5 {md5} != official {EXPECTED_MD5[name]}")
        records[name] = {**file_record(path), "md5": md5}
    return records


def _positive_rows(root: Path) -> dict[str, dict[str, Any]]:
    value = load_json(_raw_paths(root)["test_expression_pos.json"])
    if not isinstance(value, dict) or len(value) != EXPECTED_COUNTS["positive"]:
        raise ValueError("FineCops positive annotation count drifted")
    return {str(key): dict(row) for key, row in value.items()}


def _all_coco(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    value = load_json(_raw_paths(root)["test_expression_all_coco_format.json"])
    if not isinstance(value, dict):
        raise ValueError("FineCops COCO annotation must be an object")
    images = [dict(row) for row in value.get("images", [])]
    annotations = [dict(row) for row in value.get("annotations", [])]
    if len(images) != EXPECTED_COUNTS["records"] or len(annotations) != len(images):
        raise ValueError("FineCops COCO record count drifted")
    return images, annotations


def referenced_images(root: Path) -> tuple[set[str], set[str]]:
    images, annotations = _all_coco(root)
    image_by_id = {int(row["id"]): row for row in images}
    if len(image_by_id) != len(images):
        raise ValueError("FineCops COCO image IDs are not unique")
    gqa: set[str] = set()
    negative: set[str] = set()
    counts: Counter[str] = Counter()
    for ann in annotations:
        image = image_by_id[int(ann["image_id"])]
        category = ann.get("negative_cate")
        kind = "positive" if category is None else str(category)
        counts[kind] += 1
        filename = str(image.get("file_name", ""))
        if kind == "image":
            if not filename.startswith("neg_"):
                raise ValueError("negative-image row does not use neg_ image")
            negative.add(filename)
        else:
            if filename.startswith("neg_"):
                raise ValueError("positive/text row unexpectedly uses neg_ image")
            gqa.add(filename)
    for key in ("positive", "text", "image"):
        if counts[key] != EXPECTED_COUNTS[key]:
            raise ValueError(f"FineCops {key} count drifted: {counts[key]}")
    if len(gqa) != EXPECTED_COUNTS["original_images"]:
        raise ValueError("FineCops original image count drifted")
    if len(negative) != EXPECTED_COUNTS["negative_images"]:
        raise ValueError("FineCops negative image count drifted")
    return gqa, negative


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    return member


def extract_images(root: Path, gqa_zip: Path) -> dict[str, Any]:
    gqa_names, negative_names = referenced_images(root)
    gqa_output = root / "images" / "gqa"
    negative_output = root / "images" / "negative"
    gqa_output.mkdir(parents=True, exist_ok=True)
    negative_output.mkdir(parents=True, exist_ok=True)

    wanted = {Path(name).name for name in gqa_names}
    with zipfile.ZipFile(Path(gqa_zip).resolve(strict=True)) as archive:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            safe = _safe_member(info.filename)
            basename = safe.name
            if basename in wanted:
                if basename in members:
                    raise ValueError(f"duplicate GQA archive basename: {basename}")
                members[basename] = info
        missing = sorted(wanted - set(members))
        if missing:
            raise ValueError(f"GQA zip misses {len(missing)} required images")
        for basename in sorted(wanted):
            destination = gqa_output / basename
            if destination.is_file() and destination.stat().st_size == members[basename].file_size:
                checksum = 0
                with destination.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        checksum = zlib.crc32(block, checksum)
                if checksum & 0xFFFFFFFF == members[basename].CRC:
                    continue
            temporary = destination.with_name(destination.name + ".tmp")
            with archive.open(members[basename]) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            os.replace(temporary, destination)

    negative_result = extract_negative_images(root, negative_names)

    return {
        "gqa_zip": file_record(gqa_zip),
        "gqa_images": len(wanted),
        **negative_result,
    }


def extract_negative_images(
    root: Path, negative_names: Iterable[str] | None = None
) -> dict[str, Any]:
    if negative_names is None:
        _, names = referenced_images(root)
        negative_names = names
    negative_output = root / "images" / "negative"
    negative_output.mkdir(parents=True, exist_ok=True)
    wanted_negative = {Path(name).name for name in negative_names}
    with tarfile.open(_raw_paths(root)["neg_images.tgz"], "r:gz") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for info in archive:
            safe = _safe_member(info.name)
            if not info.isfile() or safe.name not in wanted_negative:
                continue
            if safe.name in members:
                raise ValueError(f"duplicate negative archive basename: {safe.name}")
            members[safe.name] = info
        missing = sorted(wanted_negative - set(members))
        if missing:
            raise ValueError(f"negative archive misses {len(missing)} required images")
        for basename in sorted(wanted_negative):
            destination = negative_output / basename
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            source = archive.extractfile(members[basename])
            if source is None:
                raise ValueError(f"could not read negative member {basename}")
            temporary = destination.with_name(destination.name + ".tmp")
            with source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            os.replace(temporary, destination)

    return {"negative_images": len(wanted_negative)}


def seed_gqa_from_local_coco(
    root: Path, vg_metadata: Path, coco_root: Path
) -> dict[str, Any]:
    """Preseed official GQA/COCO source images from the existing COCO mirror.

    The completed GQA zip extraction later verifies these bytes against each
    zip member CRC and replaces any mismatch.
    """
    positives = _positive_rows(root)
    wanted = {int(row["image_id"]) for row in positives.values()}
    metadata = load_json(vg_metadata)
    crosswalk = {
        int(row["image_id"]): int(row["coco_id"])
        for row in metadata
        if int(row["image_id"]) in wanted and row.get("coco_id") is not None
    }
    output_root = root / "images" / "gqa"
    output_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    missing = 0
    for gqa_id, coco_id in sorted(crosswalk.items()):
        filename = f"{coco_id:012d}.jpg"
        candidates = (
            coco_root / "train2017" / "train2017" / filename,
            coco_root / "train2017" / filename,
            coco_root / "val2017" / "val2017" / filename,
            coco_root / "val2017" / filename,
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            missing += 1
            continue
        destination = output_root / f"{gqa_id}.jpg"
        if destination.exists():
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        linked += 1
    return {
        "finecops_gqa_with_coco_crosswalk": len(crosswalk),
        "newly_seeded": linked,
        "local_coco_missing": missing,
        "verification": "pending_official_gqa_zip_crc",
    }


def _name_to_canonical(path: Path) -> dict[str, int]:
    value = load_json(path)
    if not isinstance(value, list):
        raise ValueError("canonical taxonomy must be a list")
    result: dict[str, int] = {}
    for entry in value:
        cid = int(entry["id"])
        values: list[Any] = [entry.get(key) for key in (
            "raw_name", "norm_name", "base_name", "synset"
        )]
        values.extend(entry.get("synonyms") or [])
        for alias in entry.get("aliases") or []:
            if isinstance(alias, dict):
                values.extend([alias.get("name"), alias.get("norm_name")])
        for raw in values:
            if isinstance(raw, str) and raw.strip():
                result[normalize_name(raw)] = cid
    return result


def _vg_coco_crosswalk(vg_metadata: Path, test_gqa_ids: set[int]) -> set[int]:
    rows = load_json(vg_metadata)
    if not isinstance(rows, list):
        raise ValueError("Visual Genome metadata must be a list")
    result: set[int] = set()
    observed: set[int] = set()
    for row in rows:
        image_id = int(row["image_id"])
        if image_id not in test_gqa_ids:
            continue
        observed.add(image_id)
        if row.get("coco_id") is not None:
            result.add(int(row["coco_id"]))
    if observed != test_gqa_ids:
        raise ValueError("not every FineCops GQA source has VG metadata")
    return result


def _support_source_id(path: Path) -> tuple[str | None, int | None]:
    parts = path.parts
    source = next((name for name in ("vg_patches", "lvis_patches", "coco_patches") if name in parts), None)
    if source is None:
        return None, None
    class_name = path.parent.name
    prefix = class_name + "_"
    if not path.stem.startswith(prefix):
        return source, None
    token = path.stem[len(prefix):].split("_", 1)[0]
    return source, int(token) if token.isdigit() else None


def _select_supports(
    *,
    support_cache: Path,
    needed_class_ids: set[int],
    test_gqa_ids: set[int],
    test_coco_ids: set[int],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    with support_cache.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("bank"), dict):
        raise ValueError("support cache contract drifted")
    bank = payload["bank"]
    selected: dict[int, dict[str, Any]] = {}
    removed = 0
    for class_id in sorted(needed_class_ids | {DUMMY_CLASS_ID}):
        candidates: list[Path] = []
        for raw in bank.get(int(class_id), []):
            path = Path(str(raw)).expanduser().resolve(strict=True)
            source, source_id = _support_source_id(path)
            if source_id is not None and (
                (source == "vg_patches" and source_id in test_gqa_ids)
                or source_id in test_coco_ids
            ):
                removed += 1
                continue
            candidates.append(path)
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda path: hashlib.sha256(
                f"{SUPPORT_SALT}\0{class_id}\0{path}".encode("utf-8")
            ).hexdigest(),
        )
        selected[class_id] = {
            "class_id": class_id,
            "path": str(chosen),
            "size_bytes": int(chosen.stat().st_size),
            "sha256": digest_file(chosen),
            "selection_key_sha256": hashlib.sha256(
                f"{SUPPORT_SALT}\0{class_id}\0{chosen}".encode("utf-8")
            ).hexdigest(),
            "candidate_count_after_exclusion": len(candidates),
        }
    return selected, {"excluded_candidates": removed, "selected_classes": len(selected)}


def _image_records(root: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    images, annotations = _all_coco(root)
    image_by_id = {int(row["id"]): row for row in images}
    ann_by_id = {int(row["id"]): row for row in annotations}
    if len(image_by_id) != len(images) or len(ann_by_id) != len(annotations):
        raise ValueError("FineCops COCO IDs are not unique")
    if set(image_by_id) != set(ann_by_id):
        raise ValueError("FineCops image/annotation identity drifted")
    return image_by_id, ann_by_id


def _validate_image(path: Path, expected_width: int, expected_height: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.size != (int(expected_width), int(expected_height)):
            raise ValueError(
                f"image size drifted for {path}: {image.size} != "
                f"{(expected_width, expected_height)}"
            )


def build_manifests(
    root: Path,
    canonical_json: Path,
    support_cache: Path,
    vg_metadata: Path,
) -> dict[str, Any]:
    raw_records = verify_raw(root)
    positives = _positive_rows(root)
    image_by_id, ann_by_id = _image_records(root)
    test_gqa_ids = {int(row["image_id"]) for row in positives.values()}
    test_coco_ids = _vg_coco_crosswalk(vg_metadata, test_gqa_ids)
    name_to_id = _name_to_canonical(canonical_json)

    needed: set[int] = set()
    for ann in ann_by_id.values():
        tuple_value = ann.get("tuple")
        if not isinstance(tuple_value, list) or not tuple_value or not tuple_value[0]:
            raise ValueError(f"FineCops annotation {ann['id']} has no active tuple")
        mapped = name_to_id.get(normalize_name(str(tuple_value[0][0])))
        if mapped is not None:
            needed.add(mapped)
    supports, support_stats = _select_supports(
        support_cache=support_cache,
        needed_class_ids=needed,
        test_gqa_ids=test_gqa_ids,
        test_coco_ids=test_coco_ids,
    )
    if DUMMY_CLASS_ID not in supports:
        raise ValueError("dummy support class has no leakage-filtered support")

    all_rows: list[dict[str, Any]] = []
    a_rows: list[dict[str, Any]] = []
    bc_rows: list[dict[str, Any]] = []
    coverage: Counter[str] = Counter()
    image_hash_cache: dict[Path, dict[str, Any]] = {}
    for annotation_id in sorted(ann_by_id):
        ann = ann_by_id[annotation_id]
        image = image_by_id[annotation_id]
        category = ann.get("negative_cate")
        kind = "positive" if category is None else str(category)
        parent_id = annotation_id if kind == "positive" else int(ann["positive_id"])
        if str(parent_id) not in positives:
            raise ValueError(f"annotation {annotation_id} has unknown parent {parent_id}")
        cluster_image_id = int(positives[str(parent_id)]["image_id"])
        active_category = normalize_name(str(ann["tuple"][0][0]))
        canonical_id = name_to_id.get(active_category)
        support = supports.get(canonical_id) if canonical_id is not None else None
        supported = support is not None
        coverage[f"{kind}_total"] += 1
        if supported:
            coverage[f"{kind}_supported"] += 1
        filename = str(image["file_name"])
        image_path = (
            root / "images" / "negative" / filename
            if kind == "image"
            else root / "images" / "gqa" / filename
        )
        _validate_image(image_path, int(image["width"]), int(image["height"]))
        resolved_image = image_path.resolve(strict=True)
        if resolved_image not in image_hash_cache:
            image_hash_cache[resolved_image] = file_record(resolved_image)
        bbox = [float(value) for value in ann["bbox"]]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"annotation {annotation_id} has invalid bbox")
        expression = str(ann.get("expression") or ann.get("caption") or "").strip()
        if not expression:
            raise ValueError(f"annotation {annotation_id} has empty expression")
        base = {
            "sample_id": f"finecops:test:{annotation_id}",
            "image_id": annotation_id,
            "ann_id": annotation_id,
            "ref_id": annotation_id,
            "sent_id": annotation_id,
            "source": "finecops_ref_v1",
            "split": "finecops_test",
            "filename": str(image_path.resolve()),
            "finecops_image_artifact": image_hash_cache[resolved_image],
            "finecops_annotation_id": annotation_id,
            "finecops_parent_positive_id": parent_id,
            "finecops_cluster_gqa_image_id": cluster_image_id,
            "finecops_original_file_name": filename,
            "finecops_kind": kind,
            "finecops_level": int(ann["level"]),
            "finecops_tuple_type": str(ann["tuple_type"]),
            "finecops_negative_type": ann.get("negative_type"),
            "finecops_negative_level": ann.get("negative_level"),
            "finecops_active_category": active_category,
            "finecops_expression": expression,
            "finecops_bbox_xywh": bbox,
            "finecops_support_covered": bool(supported),
            "finecops_canonical_id": canonical_id,
            "finecops_support": support,
        }
        all_rows.append(base)

        def patch_row(class_id: int, head: str) -> dict[str, Any]:
            return {
                **base,
                "primary_support_instance_index": 0,
                "instances": [
                    {
                        "bbox": bbox,
                        "class_id": int(class_id),
                        "raw_phrase": expression,
                        "phrase": expression,
                        "positive_phrase": expression,
                        "head": head,
                        "head_phrase": head,
                        "canonical_name": head,
                        "text_is_negative": False,
                        "category_complete_primary": True,
                    }
                ],
            }

        # Keep the complete row order for cross-route frozen-trunk parity.  An
        # unsupported A row receives the fixed dummy support only for a
        # diagnostic forward and is always excluded from A metrics.
        a_rows.append(
            patch_row(
                int(canonical_id) if supported else DUMMY_CLASS_ID,
                active_category,
            )
        )
        bc_rows.append(patch_row(DUMMY_CLASS_ID, active_category))

    if len(all_rows) != EXPECTED_COUNTS["records"]:
        raise ValueError("derived FineCops manifest row count drifted")
    if coverage["positive_supported"] != 9182:
        raise ValueError(
            f"exact positive coverage drifted: {coverage['positive_supported']} != 9182"
        )

    manifest_dir = root / "manifests"
    all_path = manifest_dir / "finecops_test_all.jsonl"
    a_path = manifest_dir / "finecops_test_a_full_diagnostic.jsonl"
    bc_path = manifest_dir / "finecops_test_bc_full.jsonl"
    write_jsonl_atomic(all_path, all_rows)
    write_jsonl_atomic(a_path, a_rows)
    write_jsonl_atomic(bc_path, bc_rows)

    support_tsv = manifest_dir / "selected_support.tsv"
    temporary = support_tsv.with_name(support_tsv.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class", "bucket"], delimiter="\t")
        writer.writeheader()
        for class_id in sorted(supports):
            writer.writerow({"path": supports[class_id]["path"], "class": class_id, "bucket": "clean"})
    os.replace(temporary, support_tsv)

    support_manifest_path = manifest_dir / "selected_support.json"
    support_payload = {
        "schema": "arrow.finecops.support_manifest/v1",
        "selection_salt": SUPPORT_SALT,
        "source_cache": file_record(support_cache),
        "canonical_taxonomy": file_record(canonical_json),
        "excluded_test_gqa_ids_sha256": canonical_json_sha256(sorted(test_gqa_ids)),
        "excluded_test_coco_ids_sha256": canonical_json_sha256(sorted(test_coco_ids)),
        "stats": support_stats,
        "supports": [supports[key] for key in sorted(supports)],
    }
    write_json_atomic(support_manifest_path, support_payload)

    hf_receipt_path = manifest_dir / "hf_gqa_selective_download.json"
    if not hf_receipt_path.is_file():
        raise ValueError("selective GQA image-source receipt is missing")
    hf_receipt = load_json(hf_receipt_path)
    if (
        hf_receipt.get("schema")
        != "arrow.finecops.hf_gqa_selective_download/v1"
        or hf_receipt.get("all_required_present") is not True
    ):
        raise ValueError("selective GQA image-source receipt is incomplete")
    official_zip_receipt_path = manifest_dir / "official_gqa_zip_verification.json"
    if not official_zip_receipt_path.is_file():
        raise ValueError("official GQA zip verification receipt is missing")
    official_zip_receipt = load_json(official_zip_receipt_path)
    if (
        official_zip_receipt.get("schema")
        != "arrow.finecops.gqa_zip_verification/v1"
        or official_zip_receipt.get("required_image_crc_parity") is not True
        or int(official_zip_receipt.get("required_image_count", -1)) != 4313
    ):
        raise ValueError("official GQA zip byte parity is incomplete")
    payload = {
        "schema": DATASET_SCHEMA,
        "status": "prepared_before_model_forward",
        "official_release": {
            "figshare_doi": "10.6084/m9.figshare.26048050.v1",
            "raw_files": raw_records,
        },
        "counts": dict(sorted(coverage.items())),
        "expected_counts": EXPECTED_COUNTS,
        "image_overlap_disclosure": {
            "claim": "benchmark_specific_zero_shot_not_image_disjoint",
            "finecops_test_gqa_images": len(test_gqa_ids),
            "finecops_test_coco_crosswalk_images": len(test_coco_ids),
        },
        "image_pixel_source": {
            "role": "official_gqa_zip_bytes",
            "verification": file_record(official_zip_receipt_path),
            "hf_mirror_role": "discovery_only_reencoded_bytes_not_used",
        },
        "manifests": {
            "all": file_record(all_path, rows=len(all_rows)),
            "a_eval": file_record(a_path, rows=len(a_rows)),
            "bc_full": file_record(bc_path, rows=len(bc_rows)),
            "support_tsv": file_record(support_tsv, rows=len(supports)),
            "support_manifest": file_record(support_manifest_path),
        },
        "support_contract": {
            "mapping": "existing_taxonomy_exact_last_wins_v1",
            "positive_supported": coverage["positive_supported"],
            "positive_total": coverage["positive_total"],
            "positive_coverage": coverage["positive_supported"] / coverage["positive_total"],
            "unsupported_positive_counted_wrong_for_lower_bound": True,
        },
        "inputs": {
            "canonical_taxonomy": file_record(canonical_json),
            "support_cache": file_record(support_cache),
            "vg_metadata": file_record(vg_metadata),
            "hf_gqa_selective_download": file_record(hf_receipt_path),
            "official_gqa_zip_verification": file_record(
                official_zip_receipt_path
            ),
            "coco_train2017": file_record(
                DEFAULT_COCO_ROOT / "annotations" / "instances_train2017.json"
            ),
            "coco_val2017": file_record(
                DEFAULT_COCO_ROOT / "annotations" / "instances_val2017.json"
            ),
        },
    }
    dataset_manifest_path = manifest_dir / "dataset_manifest.json"
    write_json_atomic(dataset_manifest_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "verify-raw",
            "seed-coco",
            "extract-negative",
            "extract",
            "build",
            "prepare",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gqa-zip", type=Path, default=DEFAULT_ROOT / "raw" / "gqa" / "images.zip")
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--support-cache", type=Path, default=DEFAULT_SUPPORT_CACHE)
    parser.add_argument("--vg-metadata", type=Path, default=DEFAULT_VG_METADATA)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "verify-raw":
        result = verify_raw(root)
    elif args.command == "seed-coco":
        verify_raw(root)
        result = seed_gqa_from_local_coco(root, args.vg_metadata, args.coco_root)
    elif args.command == "extract-negative":
        verify_raw(root)
        result = extract_negative_images(root)
    elif args.command == "extract":
        verify_raw(root)
        result = extract_images(root, args.gqa_zip)
    elif args.command == "build":
        result = build_manifests(root, args.canonical_json, args.support_cache, args.vg_metadata)
    else:
        verify_raw(root)
        extract_images(root, args.gqa_zip)
        result = build_manifests(root, args.canonical_json, args.support_cache, args.vg_metadata)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
