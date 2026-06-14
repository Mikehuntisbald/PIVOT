#!/usr/bin/env python3
"""
Build pure-GroundingDINO ODVG datasets for Stage-B data ablations.

This keeps the model/training path ordinary GroundingDINO: no patch episodes,
no support patches, no patch/rank losses. Stage-B here means the data recipe:
LVIS + COCO + RefCOCO+ phrases + RefCOCOg phrases, optionally with TN phrases.
TN rows are encoded as VG samples with a caption but zero regions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path_env_defaults() -> Dict[str, str]:
    return {
        "DATA_ROOT": os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"),
        "GDINO_ROOT": str(_repo_root()),
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


def _clean_phrase(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    text = text[:-1].strip() if text.endswith(".") else text
    text = _WS_RE.sub(" ", text)
    return text


def _xywh_to_xyxy(box: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]


def _as_xyxy(box: Iterable[float], box_format: str) -> Optional[List[float]]:
    values = [float(v) for v in box]
    if len(values) != 4:
        return None
    if box_format == "xywh":
        x0, y0, x1, y1 = _xywh_to_xyxy(values)
    elif box_format == "xyxy":
        x0, y0, x1, y1 = values
    else:
        raise ValueError(f"Unsupported box_format={box_format!r}")
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _caption_from_phrases(phrases: List[str]) -> str:
    return " . ".join(phrases) + " ."


def _convert_phrase_jsonl(
    datasetinfo: Dict[str, Any],
    out_path: Path,
    *,
    force_empty_negative: bool,
    progress_interval: int,
) -> Dict[str, Any]:
    anno_path = Path(_expand_path(datasetinfo["anno"]) or "")
    box_format = str(datasetinfo.get("box_format", "xywh")).lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = 0
    positive_rows = 0
    negative_rows = 0
    skipped_bad_box = 0
    skipped_empty_positive = 0
    source_counts: Dict[str, int] = {}

    with anno_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line_i, line in enumerate(src, start=1):
            if progress_interval > 0 and line_i % progress_interval == 0:
                print(f"[convert] {anno_path.name} {line_i} rows read, {rows_out} rows written", flush=True)
            if not line.strip():
                continue
            meta = json.loads(line)
            instances = list(meta.get("instances", []) or [])
            source = str(meta.get("source", "refexp"))
            source_counts[source] = source_counts.get(source, 0) + 1
            is_negative = force_empty_negative or any(bool(inst.get("text_is_negative", False)) for inst in instances)

            if is_negative:
                phrases = []
                seen = set()
                for inst in instances:
                    phrase = _clean_phrase(inst.get("raw_phrase") or inst.get("negative_phrase") or inst.get("phrase"))
                    if phrase and phrase not in seen:
                        seen.add(phrase)
                        phrases.append(phrase)
                if not phrases:
                    phrases = ["object"]
                row = {
                    "filename": str(meta["filename"]),
                    "image_id": int(meta.get("image_id", rows_out)),
                    "grounding": {
                        "regions": [],
                        "caption": _caption_from_phrases(phrases),
                        "caption_list": phrases,
                        "is_negative": True,
                    },
                }
                negative_rows += 1
            else:
                regions = []
                for inst in instances:
                    bbox = _as_xyxy(inst.get("bbox", []), box_format)
                    if bbox is None:
                        skipped_bad_box += 1
                        continue
                    phrase = _clean_phrase(
                        inst.get("raw_phrase")
                        or inst.get("positive_phrase")
                        or inst.get("head_phrase")
                        or inst.get("canonical_name")
                    )
                    if not phrase:
                        phrase = "object"
                    regions.append({"bbox": bbox, "phrase": phrase})
                if not regions:
                    skipped_empty_positive += 1
                    continue
                row = {
                    "filename": str(meta["filename"]),
                    "image_id": int(meta.get("image_id", rows_out)),
                    "grounding": {"regions": regions},
                }
                positive_rows += 1

            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_out += 1

    return {
        "source_anno": str(anno_path),
        "output_anno": str(out_path),
        "rows_out": rows_out,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "skipped_bad_box": skipped_bad_box,
        "skipped_empty_positive": skipped_empty_positive,
        "source_counts": source_counts,
    }


def _copy_entry(entry: Dict[str, Any], *, mix_weight: float) -> Dict[str, Any]:
    out = dict(entry)
    out["mix_weight"] = float(mix_weight)
    return out


def _stageb_train_entries(stageb_meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    lvis_entry = None
    coco_entry = None
    phrase_entries: List[Dict[str, Any]] = []
    tn_entry = None
    for entry in stageb_meta.get("train", []) or []:
        source = str(entry.get("source", "")).lower()
        anno = str(entry.get("anno", "")).lower()
        if source == "lvis":
            lvis_entry = entry
        elif source == "coco":
            coco_entry = entry
        elif "tn" in anno:
            tn_entry = entry
        else:
            phrase_entries.append(entry)
    if lvis_entry is None or coco_entry is None:
        raise ValueError("Stage-B datasets JSON must contain LVIS and COCO train entries")
    if len(phrase_entries) != 2:
        raise ValueError(f"Expected two RefCOCO phrase entries, got {len(phrase_entries)}")
    return lvis_entry, coco_entry, phrase_entries, tn_entry


def build(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    stageb_path = Path(_expand_path(args.stageb_datasets) or "")
    stagea_odvg_path = Path(_expand_path(args.stagea_odvg_datasets) or "")
    out_dir = Path(_expand_path(args.out_dir) or "")
    dataset_config_dir = Path(_expand_path(args.dataset_config_dir) or "")

    with stageb_path.open("r", encoding="utf-8") as f:
        stageb_meta = json.load(f)
    with stagea_odvg_path.open("r", encoding="utf-8") as f:
        stagea_odvg = json.load(f)

    lvis_entry, coco_entry, phrase_entries, tn_entry = _stageb_train_entries(stageb_meta)
    if tn_entry is None:
        raise ValueError("Stage-B with-TN datasets JSON must contain a TN train entry")
    if len(stagea_odvg.get("train", []) or []) < 2:
        raise ValueError(f"{stagea_odvg_path} must contain LVIS/COCO ODVG train entries")
    if not stagea_odvg.get("val"):
        raise ValueError(f"{stagea_odvg_path} must contain a val entry")

    lvis_odvg = _copy_entry(stagea_odvg["train"][0], mix_weight=float(lvis_entry.get("mix_weight", 1.0)))
    coco_odvg = _copy_entry(stagea_odvg["train"][1], mix_weight=float(coco_entry.get("mix_weight", 1.0)))

    stats: Dict[str, Any] = {
        "stageb_datasets": str(stageb_path),
        "stagea_odvg_datasets": str(stagea_odvg_path),
        "outputs": {},
    }

    phrase_odvg_entries = []
    for idx, entry in enumerate(phrase_entries):
        stem = Path(_expand_path(entry["anno"]) or "").stem
        anno_out = out_dir / f"{args.out_name}_{idx}_{stem}_vg.jsonl"
        conv_stats = _convert_phrase_jsonl(
            entry,
            anno_out,
            force_empty_negative=False,
            progress_interval=int(args.progress_interval),
        )
        stats["outputs"][stem] = conv_stats
        phrase_odvg_entries.append(
            {
                "dataset_mode": "odvg",
                "root": "/",
                "anno": str(anno_out.resolve()),
                "mix_weight": float(entry.get("mix_weight", 1.0)),
            }
        )

    tn_stem = Path(_expand_path(tn_entry["anno"]) or "").stem
    tn_out = out_dir / f"{args.out_name}_{tn_stem}_vg_empty.jsonl"
    stats["outputs"][tn_stem] = _convert_phrase_jsonl(
        tn_entry,
        tn_out,
        force_empty_negative=True,
        progress_interval=int(args.progress_interval),
    )
    tn_odvg = {
        "dataset_mode": "odvg",
        "root": "/",
        "anno": str(tn_out.resolve()),
        "mix_weight": float(tn_entry.get("mix_weight", 1.0)),
    }

    val_entries = [dict(stagea_odvg["val"][0])]
    with_tn_meta = {"train": [lvis_odvg, coco_odvg, *phrase_odvg_entries, tn_odvg], "val": val_entries}
    no_tn_meta = {"train": [lvis_odvg, coco_odvg, *phrase_odvg_entries], "val": val_entries}

    with_tn_path = dataset_config_dir / "datasets_gdino_ft_stageb_with_tn_local.json"
    no_tn_path = dataset_config_dir / "datasets_gdino_ft_stageb_no_tn_local.json"
    stats_path = out_dir / f"{args.out_name}_stats.json"
    _write_json(with_tn_path, with_tn_meta)
    _write_json(no_tn_path, no_tn_meta)
    _write_json(stats_path, stats)

    print(
        json.dumps(
            {
                "with_tn_datasets": str(with_tn_path.relative_to(repo_root) if with_tn_path.is_relative_to(repo_root) else with_tn_path),
                "no_tn_datasets": str(no_tn_path.relative_to(repo_root) if no_tn_path.is_relative_to(repo_root) else no_tn_path),
                "stats": str(stats_path),
                "with_tn_train_entries": len(with_tn_meta["train"]),
                "no_tn_train_entries": len(no_tn_meta["train"]),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stageb_datasets",
        default="config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json",
        help="Stage-B patch_episode datasets JSON with TN.",
    )
    parser.add_argument(
        "--stagea_odvg_datasets",
        default="data/ablations/ogc_original_finetune_stage_a/stagea_odvg_datasets.json",
        help="Existing pure-GDINO same-data FT ODVG datasets JSON for LVIS/COCO reuse.",
    )
    parser.add_argument(
        "--out_dir",
        default="data/ablations/gdino_ft_stage_b",
        help="Directory for generated RefCOCO/TN VG jsonl files.",
    )
    parser.add_argument(
        "--out_name",
        default="stageb_gdino_ft",
        help="Prefix for generated jsonl/stat files.",
    )
    parser.add_argument(
        "--dataset_config_dir",
        default="config/ablations",
        help="Directory for generated with-TN/no-TN datasets JSON files.",
    )
    parser.add_argument(
        "--progress_interval",
        default=50000,
        type=int,
        help="Print progress every N rows; 0 disables progress prints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
