#!/usr/bin/env python3
"""Prepare immutable gRefCOCO rejection manifests and lineage overlap audit."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_grefcoco_common import (
    DATASET_SCHEMA,
    EXPECTED_SHA256,
    EXPECTED_TEST_COUNTS,
    EXPECTED_VAL_COUNTS,
    HF_REVISION,
    OVERLAP_SCHEMA,
    digest_file,
    file_record,
    load_json,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_SOURCE = Path("/home/haoyi/Downloads/grefs")
DEFAULT_ROOT = Path("/media/haoyi/T9/data/gRefCOCO/v1")
DEFAULT_IMAGES = Path("/media/haoyi/T9/data/COCO/coco2014/train2014")
DEFAULT_OUTPUT = REPO_ROOT / "outputs/arrow_grefcoco_20260820"
STAGEA_COCO = Path("/media/haoyi/T9/data/COCO/coco2017/annotations/instances_train2017.json")
STAGEA_IMAGES = Path("/media/haoyi/T9/data/COCO/coco2017/train2017")
SUPPORT_CACHE = Path("/media/haoyi/T9/data/patches_quality_emb/emb_index_from_quality.tsv.bank.clean.img.pkl")
D3_TRAIN = REPO_ROOT / "data/ablations/stageb_tn_table_b_equal_exposure_20260717/d3_proposal_covered_train.jsonl"
D3_CAL = REPO_ROOT / "data/ablations/stageb_tn_table_b_equal_exposure_20260717/d3_proposal_covered_calibration.jsonl"
R100_SOURCES = (
    REPO_ROOT / "data/ablations/stageb_refexp_three_train_category_complete_20260720/refcoco_stageb_phrase_v1.jsonl",
    REPO_ROOT / "data/ablations/stageb_refexp_three_train_category_complete_20260720/refcocoplus_stageb_phrase_v1.jsonl",
    REPO_ROOT / "data/ablations/stageb_refexp_three_train_category_complete_20260720/refcocog_stageb_phrase_v1.jsonl",
)
VG_METADATA = Path("/media/haoyi/T9/data/visual_genome_metadata/image_data.json")
FINECOPS_POS = Path("/media/haoyi/T9/data/FineCops-Ref/v1/raw/benchmark/test_expression_pos.json")
REFCOCO_REFS = Path("/media/haoyi/T9/data/COCO/refcoco/refs(unc).p")


def _normalize_text(value: str) -> str:
    # Preserve punctuation: this is the exact normalization used by the
    # pre-audit, and avoids silently merging distinct benchmark strings.
    return " ".join(str(value).strip().lower().split())


def _jsonl_image_ids(path: Path) -> set[int]:
    result: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(int(json.loads(line)["image_id"]))
    return result


def _safe_copy_annotations(source: Path, root: Path) -> dict[str, Any]:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for name, expected in EXPECTED_SHA256.items():
        source_path = source / name
        if digest_file(source_path) != expected:
            raise ValueError(f"{name} does not match pinned HF bytes")
        destination = raw / name
        if destination.exists() and digest_file(destination) != expected:
            raise ValueError(f"existing {destination} has unexpected bytes")
        if not destination.exists():
            shutil.copy2(source_path, destination)
        records[name] = file_record(destination)
    return records


def _classify_ref(ref: dict[str, Any]) -> str:
    ann_ids = [int(value) for value in ref.get("ann_id", []) if int(value) >= 0]
    no_target = ref.get("no_target")
    if type(no_target) is not bool:
        raise ValueError("gRefCOCO no_target must be boolean")
    if no_target:
        if ref.get("ann_id") != [-1] or ref.get("category_id") != [-1]:
            raise ValueError("no-target row has target annotations")
        return "negative"
    if len(ann_ids) == 1:
        return "positive"
    if len(ann_ids) > 1:
        return "multi"
    raise ValueError("target-present row has no annotation")


def _image_path(images_root: Path, image_id: int, file_name: str) -> Path:
    candidates = (images_root / file_name, images_root / f"COCO_train2014_{image_id:012d}.jpg")
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(candidates[0])
    return path.resolve()


def _build_rows(root: Path, images_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    refs = load_json(root / "raw/grefs(unc).json")
    instances = load_json(root / "raw/instances.json")
    images = {int(row["id"]): row for row in instances["images"]}
    anns = {int(row["id"]): row for row in instances["annotations"]}
    included: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    image_sizes: dict[int, tuple[int, int]] = {}
    for ref in refs:
        split = str(ref["split"])
        if split not in {"val", "testA", "testB"}:
            continue
        kind = _classify_ref(ref)
        image_id = int(ref["image_id"])
        info = images.get(image_id)
        if info is None:
            raise ValueError(f"image {image_id} missing from instances")
        image_path = _image_path(images_root, image_id, str(ref["file_name"]))
        if image_id not in image_sizes:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image_sizes[image_id] = image.size
        if image_sizes[image_id] != (int(info["width"]), int(info["height"])):
            raise ValueError(f"image dimensions drifted for {image_id}")
        ann_id: int | None = None
        bbox: list[float] | None = None
        if kind == "positive":
            ann_id = int(ref["ann_id"][0])
            ann = anns.get(ann_id)
            if ann is None or int(ann["image_id"]) != image_id:
                raise ValueError(f"positive annotation {ann_id} is invalid")
            bbox = [float(value) for value in ann["bbox"]]
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"positive annotation {ann_id} has invalid bbox")
        for sentence in ref["sentences"]:
            sent_id = int(sentence["sent_id"])
            sample_id = f"grefcoco:{split}:{int(ref['ref_id'])}:{sent_id}"
            if sample_id in seen:
                raise ValueError(f"duplicate sample identity {sample_id}")
            seen.add(sample_id)
            counts[split][kind] += 1
            if kind == "multi":
                continue
            row = {
                "schema": "arrow.grefcoco.manifest_row/v1",
                "sample_id": sample_id,
                "split": split,
                "label": 1 if kind == "positive" else 0,
                "kind": kind,
                "image_id": image_id,
                "image_path": str(image_path),
                "image_width": int(info["width"]),
                "image_height": int(info["height"]),
                "ref_id": int(ref["ref_id"]),
                "sent_id": sent_id,
                "expression": str(sentence.get("sent") or sentence.get("raw") or "").strip(),
                "ann_id": ann_id,
                "bbox_xywh": bbox,
                "admission_defined": False,
            }
            if not row["expression"]:
                raise ValueError(f"{sample_id} has an empty expression")
            (val if split == "val" else included).append(row)
    for split, expected in EXPECTED_TEST_COUNTS.items():
        observed = counts[split]
        for key in ("positive", "negative", "multi"):
            if observed[key] != expected[key]:
                raise ValueError(f"{split} {key} count drifted: {observed[key]}")
        if len({row["image_id"] for row in included if row["split"] == split}) != expected["images"]:
            raise ValueError(f"{split} image count drifted")
    if any(counts["val"][key] != value for key, value in EXPECTED_VAL_COUNTS.items()):
        raise ValueError("val count drifted")
    return included, val, {split: dict(value) for split, value in counts.items()}


def _refcoco_expression_overlap(rows: Iterable[dict[str, Any]]) -> int:
    with REFCOCO_REFS.open("rb") as handle:
        refs = pickle.load(handle)
    reference = Counter()
    for ref in refs:
        if str(ref.get("split")) not in {"testA", "testB"}:
            continue
        for sentence in ref["sentences"]:
            reference[(int(ref["image_id"]), int(ref["ann_id"]), _normalize_text(sentence["sent"]))] += 1
    candidate = Counter(
        (int(row["image_id"]), int(row["ann_id"]), _normalize_text(row["expression"]))
        for row in rows if row["kind"] == "positive"
    )
    return int(sum((reference & candidate).values()))


def _finecops_coco_ids() -> set[int]:
    positives = load_json(FINECOPS_POS)
    gqa_ids = {int(row["image_id"]) for row in positives.values()}
    return {
        int(row["coco_id"])
        for row in load_json(VG_METADATA)
        if int(row["image_id"]) in gqa_ids and row.get("coco_id") is not None
    }


def _stagea_pixel_parity(test_ids: set[int], gref_images: dict[int, Path]) -> dict[str, int]:
    matched = missing = mismatch = 0
    for image_id in sorted(test_ids):
        filename = f"{image_id:012d}.jpg"
        candidates = (STAGEA_IMAGES / filename, STAGEA_IMAGES / "train2017" / filename)
        stagea_path = next((path for path in candidates if path.is_file()), None)
        if stagea_path is None:
            missing += 1
        elif digest_file(stagea_path) == digest_file(gref_images[image_id]):
            matched += 1
        else:
            mismatch += 1
    if (matched, missing, mismatch) != (1500, 0, 0):
        raise ValueError("Stage-A pixel-byte parity drifted")
    return {"byte_identical": matched, "missing": missing, "mismatch": mismatch}


def _support_source_overlap(test_ids: set[int]) -> dict[str, Any]:
    with SUPPORT_CACHE.open("rb") as handle:
        bank = pickle.load(handle)["bank"]
    vg_to_coco = {
        int(row["image_id"]): int(row["coco_id"])
        for row in load_json(VG_METADATA)
        if row.get("coco_id") is not None
    }
    source_coco: set[int] = set()
    matched_paths = 0
    for paths in bank.values():
        for raw in paths:
            path = Path(str(raw))
            source = next((name for name in ("vg_patches", "lvis_patches", "coco_patches") if name in path.parts), None)
            if source is None:
                continue
            prefix = path.parent.name + "_"
            token = path.stem[len(prefix):].split("_", 1)[0] if path.stem.startswith(prefix) else ""
            if not token.isdigit():
                continue
            source_id = int(token)
            coco_id = vg_to_coco.get(source_id) if source == "vg_patches" else source_id
            if coco_id is not None:
                source_coco.add(coco_id)
                if coco_id in test_ids:
                    matched_paths += 1
    return {
        "unique_test_source_images": len(test_ids & source_coco),
        "bank_paths_from_test_sources": matched_paths,
        "cache": file_record(SUPPORT_CACHE),
    }


def _overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    test_ids = {int(row["image_id"]) for row in rows}
    gref_images = {int(row["image_id"]): Path(str(row["image_path"])) for row in rows}
    stagea = {int(row["id"]) for row in load_json(STAGEA_COCO)["images"]}
    r100_by_source = {path.name: _jsonl_image_ids(path) for path in R100_SOURCES}
    r100 = set().union(*r100_by_source.values())
    d3_train = _jsonl_image_ids(D3_TRAIN)
    d3_cal = _jsonl_image_ids(D3_CAL)
    d3 = d3_train | d3_cal
    finecops = _finecops_coco_ids()
    for row in rows:
        image_id = int(row["image_id"])
        row["overlap_stagea"] = image_id in stagea
        row["overlap_r100"] = image_id in r100
        row["overlap_d3_train"] = image_id in d3_train
        row["overlap_d3_calibration"] = image_id in d3_cal
        row["surface_d3_disjoint"] = image_id not in d3
        row["surface_d3_finecops_disjoint"] = image_id not in d3 and image_id not in finecops
    surfaces = {}
    for name, predicate in {
        "full": lambda row: True,
        "d3_disjoint": lambda row: bool(row["surface_d3_disjoint"]),
        "d3_finecops_disjoint": lambda row: bool(row["surface_d3_finecops_disjoint"]),
    }.items():
        selected = [row for row in rows if predicate(row)]
        surfaces[name] = {
            "images": len({int(row["image_id"]) for row in selected}),
            "positive": sum(row["kind"] == "positive" for row in selected),
            "negative": sum(row["kind"] == "negative" for row in selected),
            "by_split_images": {
                split: len({int(row["image_id"]) for row in selected if row["split"] == split})
                for split in ("testA", "testB")
            },
        }
    expected_surfaces = {
        "full": (1500, 11563, 9121),
        "d3_disjoint": (1288, 9924, 7796),
        "d3_finecops_disjoint": (1274, 9821, 7714),
    }
    for name, values in expected_surfaces.items():
        observed = surfaces[name]
        if (observed["images"], observed["positive"], observed["negative"]) != values:
            raise ValueError(f"{name} surface count drifted: {observed}")
    return {
        "schema": OVERLAP_SCHEMA,
        "status": "complete_before_gpu_forward",
        "test_images": len(test_ids),
        "stagea": {"images": len(test_ids & stagea), "pixel_byte_audit": _stagea_pixel_parity(test_ids, gref_images)},
        "r100": {"images": len(test_ids & r100), "by_source": {key: len(test_ids & value) for key, value in r100_by_source.items()}},
        "d3": {"train_images": len(test_ids & d3_train), "calibration_images": len(test_ids & d3_cal), "union_images": len(test_ids & d3)},
        "finecops_source_crosswalk_images": len(test_ids & finecops),
        "support_bank": _support_source_overlap(test_ids),
        "refcoco_exact_positive_expression_overlap": _refcoco_expression_overlap(rows),
        "surfaces": surfaces,
        "sources": {"stagea": file_record(STAGEA_COCO), "r100": [file_record(path) for path in R100_SOURCES], "d3_train": file_record(D3_TRAIN), "d3_calibration": file_record(D3_CAL), "vg_metadata": file_record(VG_METADATA), "finecops_positive": file_record(FINECOPS_POS), "refcoco_refs": file_record(REFCOCO_REFS)},
        "claim_boundary": "annotation_task_zero_shot_on_previously_exposed_coco_imagery",
    }


def prepare(source: Path, root: Path, images_root: Path, output_root: Path) -> dict[str, Any]:
    annotations = _safe_copy_annotations(source, root)
    rows, val_rows, counts = _build_rows(root, images_root)
    overlap = _overlap(rows)
    manifests = root / "manifests"
    test_path = manifests / "grefcoco_testab_single_no_target.jsonl"
    val_path = manifests / "grefcoco_val_no_target.jsonl"
    write_jsonl_atomic(test_path, rows)
    write_jsonl_atomic(val_path, val_rows)
    overlap_path = manifests / "overlap_audit.json"
    write_json_atomic(overlap_path, overlap)
    payload = {
        "schema": DATASET_SCHEMA,
        "status": "sealed_before_gpu_forward",
        "official_release": {"hf_revision": HF_REVISION, "license": "CC BY-NC-SA 4.0 non-commercial research"},
        "annotations": annotations,
        "images_root": str(images_root.resolve(strict=True)),
        "counts": counts,
        "manifests": {"testab": file_record(test_path, rows=len(rows)), "val_no_target": file_record(val_path, rows=len(val_rows)), "overlap_audit": file_record(overlap_path)},
        "prohibitions": ["grefcoco_train_forward", "grefcoco_training", "checkpoint_selection", "threshold_fitting", "admission_on_no_target"],
    }
    dataset_path = manifests / "dataset_manifest.json"
    write_json_atomic(dataset_path, payload)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "arrow.grefcoco.dataset_receipt/v1", "dataset": file_record(dataset_path), "overlap": file_record(overlap_path), "status": "complete"}
    write_json_atomic(output_root / "dataset_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.root, args.images_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
