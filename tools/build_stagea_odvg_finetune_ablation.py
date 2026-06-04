#!/usr/bin/env python3
"""
Build ODVG files for the OGC original-training finetune ablation on Stage-A data.

Input is a Stage-A patch_episode datasets JSON. Output is an ODVG datasets JSON
plus jsonl annotations and a canonical label map, suitable for cfg_odvg-style
GroundingDINO training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")
_PUNC_RE = re.compile(r"[^a-z0-9 _-]+")


def _path_env_defaults() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = os.environ.get("DATA_ROOT", "/media/haoyi/T9/data")
    return {
        "DATA_ROOT": data_root,
        "GDINO_ROOT": str(repo_root),
    }


def _expand_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    out = str(value)
    for key, default_value in _path_env_defaults().items():
        out = out.replace(f"${{{key}}}", default_value)
        out = out.replace(f"${key}", default_value)
    out = os.path.expandvars(out)
    out = os.path.expanduser(out)
    return out


def _norm_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = _PUNC_RE.sub(" ", value)
    value = _WS_RE.sub(" ", value).strip()
    return value


def _clean_label_text(value: Any) -> str:
    text = str(value or "").replace("_", " ").replace(".", " ").strip()
    text = _WS_RE.sub(" ", text)
    return text or "object"


def _load_canonical_maps(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"canonical_classes_json must contain a list, got {type(data)}")

    name_to_cid: Dict[str, int] = {}
    cid_to_name: Dict[int, str] = {}

    def add_name(cid: int, name: Any) -> None:
        if not isinstance(name, str) or not name.strip():
            return
        norm = _norm_text(name)
        if norm:
            name_to_cid[norm] = int(cid)

    for entry in data:
        if not isinstance(entry, dict) or entry.get("id", None) is None:
            continue
        cid = int(entry["id"])
        preferred = None
        for key in ("base_name", "raw_name", "norm_name", "synset"):
            value = entry.get(key, None)
            if isinstance(value, str) and value.strip() and preferred is None:
                preferred = _clean_label_text(value)
            add_name(cid, value)
        for value in entry.get("synonyms", []) or []:
            add_name(cid, value)
        for alias in entry.get("aliases", []) or []:
            if isinstance(alias, dict):
                add_name(cid, alias.get("name", None))
                add_name(cid, alias.get("norm_name", None))
            else:
                add_name(cid, alias)
        if preferred is not None:
            cid_to_name[cid] = preferred

    return name_to_cid, cid_to_name


def _xywh_to_xyxy(box: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]


def _resolve_existing_image(image_root: Path, file_name: str) -> Optional[Path]:
    direct = image_root / file_name
    if direct.exists():
        return direct.resolve()
    nested = image_root / image_root.name / file_name
    if nested.exists():
        return nested.resolve()
    return None


def _convert_coco_like(
    datasetinfo: Dict[str, Any],
    *,
    source: str,
    name_to_cid: Dict[str, int],
    cid_to_name: Dict[int, str],
    progress_interval: int = 50000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    anno_path = Path(_expand_path(datasetinfo["anno"]) or "")
    if source == "lvis":
        image_root_value = datasetinfo.get("lvis_image_root", None)
    else:
        image_root_value = datasetinfo.get("coco_image_root", None)
    if not image_root_value:
        raise ValueError(f"source={source!r} requires its image root in {datasetinfo}")
    image_root = Path(_expand_path(image_root_value) or "")

    with anno_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{anno_path} must contain a COCO/LVIS-style dict")

    cat_id_to_cid: Dict[int, int] = {}
    unmapped_categories: List[Dict[str, Any]] = []
    for cat in data.get("categories", []) or []:
        cat_id = int(cat["id"])
        name = str(cat.get("name", ""))
        cid = name_to_cid.get(_norm_text(name), None)
        if cid is None:
            unmapped_categories.append({"category_id": cat_id, "name": name})
            continue
        cat_id_to_cid[cat_id] = int(cid)

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    skipped_iscrowd = 0
    skipped_unmapped = 0
    skipped_bad_box = 0
    annotations = data.get("annotations", []) or []
    for ann_i, ann in enumerate(annotations, start=1):
        if progress_interval > 0 and ann_i % progress_interval == 0:
            print(
                f"[{source}] mapped annotations {ann_i}/{len(annotations)} "
                f"(kept image buckets={len(anns_by_img)})",
                flush=True,
            )
        if int(ann.get("iscrowd", 0)) == 1:
            skipped_iscrowd += 1
            continue
        cat_id = int(ann["category_id"])
        cid = cat_id_to_cid.get(cat_id, None)
        if cid is None:
            skipped_unmapped += 1
            continue
        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            skipped_bad_box += 1
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            skipped_bad_box += 1
            continue
        category = cid_to_name.get(int(cid), str(cid))
        anns_by_img.setdefault(int(ann["image_id"]), []).append(
            {
                "bbox": _xywh_to_xyxy((x, y, w, h)),
                "label": int(cid),
                "category": category,
            }
        )

    rows: List[Dict[str, Any]] = []
    skipped_missing_image = 0
    skipped_empty_image = 0
    images = data.get("images", []) or []
    for img_i, img in enumerate(images, start=1):
        if progress_interval > 0 and img_i % progress_interval == 0:
            print(
                f"[{source}] built rows from images {img_i}/{len(images)} "
                f"(rows={len(rows)})",
                flush=True,
            )
        img_id = int(img["id"])
        instances = anns_by_img.get(img_id, [])
        if not instances:
            skipped_empty_image += 1
            continue

        file_name = None
        if source == "lvis":
            coco_url = img.get("coco_url", None)
            if isinstance(coco_url, str) and coco_url.strip():
                file_name = coco_url.strip().split("/")[-1]
        file_name = file_name or img.get("file_name", None) or f"{img_id:012d}.jpg"
        abs_path = _resolve_existing_image(image_root, str(file_name))
        if abs_path is None:
            skipped_missing_image += 1
            continue

        row = {
            "filename": str(abs_path),
            "height": int(img.get("height", 0) or 0),
            "width": int(img.get("width", 0) or 0),
            "image_id": img_id,
            "detection": {"instances": instances},
        }
        if source == "lvis":
            not_exhaustive = []
            seen_ne = set()
            for cat_id in img.get("not_exhaustive_category_ids", []) or []:
                cid = cat_id_to_cid.get(int(cat_id), None)
                if cid is None or int(cid) in seen_ne:
                    continue
                seen_ne.add(int(cid))
                not_exhaustive.append(int(cid))
            if not_exhaustive:
                row["not_exhaustive_labels"] = not_exhaustive
        rows.append(row)

    stats = {
        "source": source,
        "anno": str(anno_path),
        "image_root": str(image_root),
        "images_out": len(rows),
        "mapped_categories": len(cat_id_to_cid),
        "unmapped_categories": unmapped_categories,
        "skipped_unmapped_annotations": skipped_unmapped,
        "skipped_iscrowd_annotations": skipped_iscrowd,
        "skipped_bad_box_annotations": skipped_bad_box,
        "skipped_empty_images": skipped_empty_image,
        "skipped_missing_images": skipped_missing_image,
    }
    return rows, stats


def _write_jsonl(path: Path, rows: List[Dict[str, Any]], *, progress_interval: int = 50000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {path} rows={len(rows)}", flush=True)
    with path.open("w", encoding="utf-8") as f:
        for row_i, row in enumerate(rows, start=1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if progress_interval > 0 and row_i % progress_interval == 0:
                print(f"[write] {path.name} {row_i}/{len(rows)}", flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _split_entries(entries: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    out = []
    for i, entry in enumerate(entries):
        mode = str(entry.get("dataset_mode", ""))
        if mode != "patch_episode":
            raise ValueError(f"{split_name}[{i}] must be patch_episode, got {mode!r}")
        source = str(entry.get("source", "")).lower().strip()
        if source not in {"lvis", "coco"}:
            raise ValueError(f"{split_name}[{i}] source must be lvis/coco for this ablation, got {source!r}")
        out.append(entry)
    return out


def build(args: argparse.Namespace) -> None:
    stagea_datasets = Path(_expand_path(args.stagea_datasets) or "")
    out_dir = Path(_expand_path(args.out_dir) or "")
    out_name = str(args.out_name)

    with stagea_datasets.open("r", encoding="utf-8") as f:
        stagea_meta = json.load(f)
    canonical_path = _expand_path(args.canonical_classes_json)
    if canonical_path is None:
        train_entries = _split_entries(stagea_meta.get("train", []), "train")
        if not train_entries:
            raise ValueError("stagea datasets JSON has no train entries")
        canonical_path = _expand_path(train_entries[0].get("canonical_classes_json", None))
    if canonical_path is None:
        raise ValueError("--canonical_classes_json is required when it is not present in the Stage-A datasets JSON")

    name_to_cid, cid_to_name = _load_canonical_maps(canonical_path)
    if not name_to_cid or not cid_to_name:
        raise ValueError(f"Failed to load canonical maps from {canonical_path}")

    used_cids = set()
    output_meta = {"train": [], "val": []}
    all_stats: Dict[str, Any] = {
        "stagea_datasets": str(stagea_datasets),
        "canonical_classes_json": str(canonical_path),
        "splits": {},
    }

    label_map_path = out_dir / f"{out_name}_canonical_label_map.json"
    for split in ("train", "val"):
        entries = _split_entries(stagea_meta.get(split, []), split)
        split_stats = []
        for idx, entry in enumerate(entries):
            source = str(entry.get("source", "")).lower().strip()
            rows, stats = _convert_coco_like(
                entry,
                source=source,
                name_to_cid=name_to_cid,
                cid_to_name=cid_to_name,
                progress_interval=int(args.progress_interval),
            )
            for row in rows:
                for inst in row["detection"]["instances"]:
                    used_cids.add(int(inst["label"]))

            anno_path = out_dir / f"{out_name}_{split}_{idx}_{source}.jsonl"
            _write_jsonl(anno_path, rows, progress_interval=int(args.progress_interval))
            output_meta[split].append(
                {
                    "dataset_mode": "odvg",
                    "root": "/",
                    "anno": str(anno_path),
                    "label_map": str(label_map_path),
                }
            )
            stats["output_anno"] = str(anno_path)
            split_stats.append(stats)
        all_stats["splits"][split] = split_stats

    label_map = {str(cid): cid_to_name.get(int(cid), str(cid)) for cid in sorted(used_cids)}
    _write_json(label_map_path, label_map)

    datasets_out = out_dir / f"{out_name}_datasets.json"
    stats_out = out_dir / f"{out_name}_stats.json"
    _write_json(datasets_out, output_meta)
    _write_json(stats_out, all_stats)

    print(json.dumps({
        "datasets": str(datasets_out),
        "label_map": str(label_map_path),
        "stats": str(stats_out),
        "num_labels": len(label_map),
        "train_entries": len(output_meta["train"]),
        "val_entries": len(output_meta["val"]),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stagea_datasets",
        required=True,
        help="Stage-A patch_episode datasets JSON, e.g. config/datasets_patch_stage_a_lvis_coco2017_local.json",
    )
    parser.add_argument(
        "--out_dir",
        default="data/ablations/ogc_original_finetune_stage_a",
        help="Directory for generated ODVG jsonl/datasets files.",
    )
    parser.add_argument(
        "--out_name",
        default="stagea_odvg",
        help="Prefix for generated files.",
    )
    parser.add_argument(
        "--canonical_classes_json",
        default=None,
        help="Override canonical class JSON. Defaults to canonical_classes_json in the Stage-A dataset entry.",
    )
    parser.add_argument(
        "--progress_interval",
        default=50000,
        type=int,
        help="Print progress every N annotations/images/rows; 0 disables progress prints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
