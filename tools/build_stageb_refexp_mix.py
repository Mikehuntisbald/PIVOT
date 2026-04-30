#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _norm_phrase(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def _load_head_phrase_maps(data_root: Path) -> Tuple[Dict[Tuple, str], Dict[Tuple, str]]:
    exact: Dict[Tuple, str] = {}
    loose: Dict[Tuple, str] = {}
    pair_files = {
        "refcoco_unc": data_root / "refcoco_text_pairs" / "refcoco_unc_pairs.jsonl",
        "refcoco+_unc": data_root / "refcoco_text_pairs" / "refcoco+_unc_pairs.jsonl",
        "refcocog_google": data_root / "refcoco_text_pairs" / "refcocog_google_pairs.jsonl",
    }
    for source_name, path in pair_files.items():
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            head_phrase = row.get("head_phrase")
            if not isinstance(head_phrase, str) or not head_phrase.strip():
                continue
            ref_id = row.get("ref_id")
            ann_id = row.get("ann_id")
            image_id = row.get("image_id")
            raw_phrase = _norm_phrase(row.get("raw_phrase"))
            if ref_id is None or ann_id is None or image_id is None or not raw_phrase:
                continue
            exact[(source_name, int(ref_id), int(ann_id), int(image_id), raw_phrase)] = head_phrase.strip()
            loose[(source_name, int(ref_id), raw_phrase)] = head_phrase.strip()
    return exact, loose


def _lookup_head_phrase(row: Dict, phrase: str, exact_map: Dict[Tuple, str], loose_map: Dict[Tuple, str]) -> Optional[str]:
    source_name = row.get("pair_source") or row.get("source")
    ref_id = row.get("ref_id")
    ann_id = row.get("ann_id")
    image_id = row.get("image_id")
    norm_phrase = _norm_phrase(phrase)
    if not isinstance(source_name, str) or ref_id is None or not norm_phrase:
        return None

    if ann_id is not None and image_id is not None:
        hit = exact_map.get((source_name, int(ref_id), int(ann_id), int(image_id), norm_phrase))
        if hit:
            return hit
    return loose_map.get((source_name, int(ref_id), norm_phrase))


def _head_phrase_with_fallback(
    row: Dict,
    phrase: str,
    exact_map: Dict[Tuple, str],
    loose_map: Dict[Tuple, str],
) -> Optional[str]:
    hit = _lookup_head_phrase(row, phrase, exact_map, loose_map)
    if hit:
        return hit
    for key in ("try_tn_head_phrase", "try_tn_head"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_coco_roots(
    data_root: Path,
    coco_train_root: Optional[Path] = None,
    coco_val_root: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    train_root = coco_train_root if coco_train_root is not None else (data_root / "COCO" / "coco2014" / "train2014")
    val_root = coco_val_root if coco_val_root is not None else (data_root / "COCO" / "coco2014" / "val2014")
    if not train_root.exists():
        raise SystemExit(f"Missing COCO train2014 root: {train_root}")
    return train_root, (val_root if val_root.exists() else None)


def _canonical_coco_name(file_name: Optional[str], image_id: Optional[int]) -> Tuple[Optional[str], str]:
    raw_name = Path(str(file_name)).name if file_name else ""
    split = "train2014"
    if raw_name.startswith("COCO_val2014_"):
        split = "val2014"

    if raw_name:
        if raw_name.startswith("COCO_train2014_") or raw_name.startswith("COCO_val2014_"):
            stem = raw_name[:-4] if raw_name.endswith(".jpg") else raw_name
            parts = stem.split("_")
            if len(parts) >= 3:
                base = "_".join(parts[:3])
                return base + ".jpg", split
            return raw_name, split
        return raw_name, split

    if image_id is None:
        return None, split
    return f"COCO_{split}_{int(image_id):012d}.jpg", split


def _resolve_coco_image(file_name: Optional[str], image_id: Optional[int], train_root: Path, val_root: Optional[Path]) -> Optional[str]:
    if file_name and ("/" in str(file_name)):
        file_path = Path(str(file_name))
        if file_path.is_file():
            return str(file_path)

    name, split = _canonical_coco_name(file_name, image_id)
    if not name:
        return None
    if split == "val2014":
        if val_root is None:
            return None
        path = val_root / name
    else:
        path = train_root / name
    return str(path)


def _pick_bbox(row: Dict) -> Optional[List[float]]:
    for key in ("bbox", "gt_bbox", "sam_bbox"):
        value = row.get(key)
        if not isinstance(value, list) or len(value) != 4:
            continue
        try:
            box = [float(x) for x in value]
        except Exception:
            continue
        if box[2] <= 0 or box[3] <= 0:
            continue
        return box
    return None


def _canonical_name(row: Dict) -> str:
    for key in ("class_norm_name", "class_raw_name", "category_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "object"


def _maybe_add_tn_metadata(instance: Dict, row: Dict) -> None:
    optional_fields = (
        "try_tn_rule",
        "tn_edits",
        "replace_from",
        "replace_to",
        "replace_category",
        "replace_span",
        "try_tn_method",
        "try_tn_head",
        "try_tn_head_phrase",
        "try_tn_retry_count",
        "try_tn_failure_reason",
        "vlm_verdict",
        "vlm_reason",
        "vlm_raw_answer",
        "visual_filter_status",
        "visual_filter_reason",
        "candidate_cache_status",
        "candidate_cache_version",
        "target_bbox_used",
        "target_bbox_source",
        "proposal_num",
    )
    for key in optional_fields:
        value = row.get(key)
        if value is None:
            continue
        instance[key] = copy.deepcopy(value)


def _source_name_from_row(row: Dict, fallback: str, suffix: str) -> str:
    dataset = row.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        return f"{dataset.strip()}_{suffix}"
    pair_source = row.get("pair_source")
    if isinstance(pair_source, str) and pair_source.strip():
        return f"{pair_source.strip()}_{suffix}"
    return fallback


def _default_existing_paths(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if path.exists()]


def _build_meta(
    row: Dict,
    *,
    phrase: str,
    filename: str,
    text_is_negative: bool,
    source_name: str,
    head_phrase: Optional[str] = None,
) -> Dict:
    instance = {
        "bbox": _pick_bbox(row),
        "class_id": int(row["class_id"]),
        "raw_phrase": phrase,
        "head_phrase": head_phrase,
        "canonical_name": _canonical_name(row),
        "text_is_negative": bool(text_is_negative),
        "positive_phrase": row.get("sent"),
        "pair_source": row.get("pair_source"),
    }
    if text_is_negative:
        _maybe_add_tn_metadata(instance, row)
    return {
        "filename": filename,
        "source": source_name,
        "image_id": row.get("image_id"),
        "ann_id": row.get("ann_id"),
        "ref_id": row.get("ref_id"),
        "sent_id": row.get("sent_id"),
        "instances": [instance],
    }


def _write_positive_jsonl(
    out_path: Path,
    src_path: Path,
    source_name: str,
    train_root: Path,
    val_root: Optional[Path],
    exact_head_phrase_map: Dict[Tuple, str],
    loose_head_phrase_map: Dict[Tuple, str],
) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for row in _iter_jsonl(src_path):
            phrase = row.get("sent")
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            if "class_id" not in row:
                continue
            bbox = _pick_bbox(row)
            if bbox is None:
                continue
            filename = _resolve_coco_image(row.get("file_name"), row.get("image_id"), train_root, val_root)
            if filename is None:
                continue
            head_phrase = _head_phrase_with_fallback(row, phrase, exact_head_phrase_map, loose_head_phrase_map)
            meta = _build_meta(
                row,
                phrase=phrase.strip(),
                filename=filename,
                text_is_negative=False,
                source_name=source_name,
                head_phrase=head_phrase,
            )
            if meta["instances"][0]["bbox"] is None:
                continue
            out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_tn_jsonl(
    out_path: Path,
    src_paths: List[Path],
    train_root: Path,
    val_root: Optional[Path],
    exact_head_phrase_map: Dict[Tuple, str],
    loose_head_phrase_map: Dict[Tuple, str],
) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for src_path in src_paths:
            fallback_source_name = src_path.parent.stem if src_path.stem in {"accepted", "rejected", "unknown", "skipped"} else src_path.stem
            for row in _iter_jsonl(src_path):
                visual_filter_status = row.get("visual_filter_status")
                if visual_filter_status is not None:
                    if visual_filter_status != "accept":
                        continue
                    phrase = row.get("try_tn")
                    source_name = _source_name_from_row(row, fallback_source_name, "tn_vlm_filter")
                else:
                    vlm_verdict = row.get("vlm_verdict")
                    if isinstance(vlm_verdict, str) and vlm_verdict.strip() and vlm_verdict != "absent":
                        continue
                    phrase = row.get("vlm_tn") or row.get("try_tn")
                    source_name = _source_name_from_row(row, fallback_source_name, "tn")
                if not isinstance(phrase, str) or not phrase.strip():
                    continue
                if "class_id" not in row:
                    continue
                bbox = _pick_bbox(row)
                if bbox is None:
                    continue
                filename = _resolve_coco_image(row.get("file_name"), row.get("image_id"), train_root, val_root)
                if filename is None:
                    continue
                head_phrase = _head_phrase_with_fallback(
                    row,
                    row.get("sent", phrase),
                    exact_head_phrase_map,
                    loose_head_phrase_map,
                )
                meta = _build_meta(
                    row,
                    phrase=phrase.strip(),
                    filename=filename,
                    text_is_negative=True,
                    source_name=source_name,
                    head_phrase=head_phrase,
                )
                if meta["instances"][0]["bbox"] is None:
                    continue
                out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/media/haoyi/T9/data")
    ap.add_argument("--out-dir", default="/media/haoyi/T9/data/patch_episode_prebuilt")
    ap.add_argument("--coco-train-root", default=None)
    ap.add_argument("--coco-val-root", default=None)
    ap.add_argument("--refcocoplus-src", default=None)
    ap.add_argument("--refcocog-src", default=None)
    ap.add_argument("--tn-srcs", nargs="*", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root, val_root = _resolve_coco_roots(
        data_root,
        coco_train_root=(Path(args.coco_train_root) if args.coco_train_root else None),
        coco_val_root=(Path(args.coco_val_root) if args.coco_val_root else None),
    )
    exact_head_phrase_map, loose_head_phrase_map = _load_head_phrase_maps(data_root)

    refcocoplus_src = (
        Path(args.refcocoplus_src)
        if args.refcocoplus_src
        else (data_root / "SAM3" / "out" / "refcocoplus_sam3_washed_try_tn_llm_head.jsonl")
    )
    refcocog_src = (
        Path(args.refcocog_src)
        if args.refcocog_src
        else (data_root / "SAM3" / "out" / "refcocog_sam3_washed_try_tn_llm_head.jsonl")
    )
    tn_srcs = (
        [Path(p) for p in args.tn_srcs]
        if args.tn_srcs
        else _default_existing_paths(
            [
                data_root / "SAM3" / "output" / "refcoco_sam3_washed_try_tn_llm_head_candidates_vlm_filter" / "accepted.jsonl",
                data_root / "SAM3" / "output" / "refcocoplus_sam3_washed_try_tn_llm_head_candidates_vlm_filter" / "accepted.jsonl",
                data_root / "SAM3" / "output" / "refcocog_sam3_washed_try_tn_llm_head_candidates_vlm_filter" / "accepted.jsonl",
            ]
        )
    )

    if not tn_srcs:
        raise SystemExit("No TN source files found. Pass --tn-srcs explicitly or generate *_vlm_filter/accepted.jsonl files.")

    for path in [refcocoplus_src, refcocog_src] + tn_srcs:
        if not path.exists():
            raise SystemExit(f"Missing source file: {path}")

    out_plus = out_dir / "refcocoplus_stageb_phrase_v1.jsonl"
    out_gog = out_dir / "refcocog_stageb_phrase_v1.jsonl"
    out_tn = out_dir / "refexp_tn_stageb_v1.jsonl"

    n_plus = _write_positive_jsonl(
        out_plus, refcocoplus_src, "refcocoplus_phrase", train_root, val_root, exact_head_phrase_map, loose_head_phrase_map
    )
    n_gog = _write_positive_jsonl(
        out_gog, refcocog_src, "refcocog_phrase", train_root, val_root, exact_head_phrase_map, loose_head_phrase_map
    )
    n_tn = _write_tn_jsonl(
        out_tn, tn_srcs, train_root, val_root, exact_head_phrase_map, loose_head_phrase_map
    )

    summary = {
        "coco_roots": {
            "train2014": str(train_root),
            "val2014": (str(val_root) if val_root is not None else None),
        },
        "outputs": {
            "refcocoplus": {"path": str(out_plus), "count": n_plus},
            "refcocog": {"path": str(out_gog), "count": n_gog},
            "tn": {"path": str(out_tn), "count": n_tn},
        },
        "inputs": {
            "refcocoplus": str(refcocoplus_src),
            "refcocog": str(refcocog_src),
            "tn": [str(path) for path in tn_srcs],
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
