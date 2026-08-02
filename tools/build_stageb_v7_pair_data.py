#!/usr/bin/env python3
"""Build leakage-free, VLM-verified Stage B v7 paired TN training data."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/user/datasets/pivot_data"))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sam3_and_tn" / "stageb_v7_pair_verified_train"

PLUS_INPUT = (
    REPO_ROOT
    / "sam3_and_tn"
    / "refcocoplus_sam3_washed_try_tn_llm_head_candidates_vlm_filter"
    / "accepted.jsonl"
)
G_INPUT = (
    REPO_ROOT
    / "sam3_and_tn"
    / "refcocog_sam3_washed_try_tn_llm_head_candidates_vlm_filter"
    / "accepted.jsonl"
)

PLUS_OUTPUT_NAME = "refcocoplus_verified_train.jsonl"
G_OUTPUT_NAME = "refcocog_verified_train.jsonl"
STATS_NAME = "stats.json"

EXPECTED_PLUS_ROWS = 25_785
EXPECTED_G_ROWS = 21_055
_WS_RE = re.compile(r"\s+")

RefSentenceKey = Tuple[int, int, int]
RefKey = Tuple[int, int, int]
UmdSentenceKey = Tuple[int, int, str]


def _normalize_sentence(value: Any) -> str:
    """Match RefCOCOg sentences across Google and UMD annotations."""
    text = str(value or "").replace("_", " ").replace(".", " ").strip().lower()
    return _WS_RE.sub(" ", text)


def _load_refs(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"RefCOCO annotation file does not exist: {path}")
    with path.open("rb") as handle:
        refs = pickle.load(handle)
    if not isinstance(refs, list):
        raise TypeError(f"Expected a list in {path}, got {type(refs).__name__}")
    return refs


def _add_split(
    mapping: MutableMapping[Tuple[Any, ...], str],
    key: Tuple[Any, ...],
    split: str,
    *,
    source: Path,
) -> bool:
    previous = mapping.get(key)
    if previous is not None and previous != split:
        raise ValueError(
            f"Conflicting splits for key {key!r} in {source}: {previous!r} vs {split!r}"
        )
    mapping[key] = split
    return previous is not None


def _build_plus_split_map(path: Path) -> Tuple[Dict[RefSentenceKey, str], int]:
    split_map: Dict[RefSentenceKey, str] = {}
    duplicate_count = 0
    for ref in _load_refs(path):
        split = str(ref["split"])
        ref_id = int(ref["ref_id"])
        ann_id = int(ref["ann_id"])
        for sentence in ref.get("sentences", []) or []:
            key = (ref_id, ann_id, int(sentence["sent_id"]))
            duplicate_count += int(_add_split(split_map, key, split, source=path))
    return split_map, duplicate_count


def _build_google_split_map(path: Path) -> Tuple[Dict[RefKey, str], int]:
    split_map: Dict[RefKey, str] = {}
    duplicate_count = 0
    for ref in _load_refs(path):
        key = (int(ref["ref_id"]), int(ref["ann_id"]), int(ref["image_id"]))
        duplicate_count += int(
            _add_split(split_map, key, str(ref["split"]), source=path)
        )
    return split_map, duplicate_count


def _build_umd_split_map(path: Path) -> Tuple[Dict[UmdSentenceKey, str], int]:
    split_map: Dict[UmdSentenceKey, str] = {}
    duplicate_count = 0
    for ref in _load_refs(path):
        split = str(ref["split"])
        image_id = int(ref["image_id"])
        ann_id = int(ref["ann_id"])
        for sentence in ref.get("sentences", []) or []:
            normalized = _normalize_sentence(sentence.get("sent") or sentence.get("raw"))
            if not normalized:
                raise ValueError(f"Empty normalized sentence in {path}: {sentence!r}")
            key = (image_id, ann_id, normalized)
            duplicate_count += int(_add_split(split_map, key, split, source=path))
    return split_map, duplicate_count


def _load_json_row(line: str, path: Path, line_number: int) -> Dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(row, dict):
        raise TypeError(f"Expected a JSON object at {path}:{line_number}")
    return row


def _is_verified_negative(row: Mapping[str, Any]) -> bool:
    return (
        row.get("visual_filter_status") == "accept"
        and row.get("visual_filter_reason") == "verified_negative"
    )


def _validate_training_row(row: Mapping[str, Any], context: str) -> None:
    bbox = row.get("target_bbox_used")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Missing target_bbox_used at {context}")
    try:
        bbox_values = [float(value) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid target_bbox_used at {context}: {bbox!r}") from exc
    if bbox_values[2] <= 0 or bbox_values[3] <= 0:
        raise ValueError(f"Non-positive target_bbox_used at {context}: {bbox_values!r}")
    for key in ("sent", "try_tn"):
        if not isinstance(row.get(key), str) or not str(row[key]).strip():
            raise ValueError(f"Missing {key} at {context}")
    _int_key(row, ("image_id", "ann_id", "class_id"), context)


def _write_training_row(target, row: Mapping[str, Any], *, source_splitby: str) -> None:
    output_row = dict(row)
    output_row["split"] = "train"
    output_row["source_splitby"] = source_splitby
    output_row["training_bbox_key"] = "target_bbox_used"
    target.write(json.dumps(output_row, ensure_ascii=True, separators=(",", ":")) + "\n")


def _int_key(row: Mapping[str, Any], fields: Tuple[str, ...], context: str) -> Tuple[int, ...]:
    try:
        return tuple(int(row[field]) for field in fields)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {fields!r} key in {context}") from exc


def _write_plus(
    input_path: Path,
    output_path: Path,
    split_map: Mapping[RefSentenceKey, str],
) -> Tuple[Dict[str, Any], set[RefSentenceKey]]:
    counts: Counter[str] = Counter()
    selected_keys: set[RefSentenceKey] = set()
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with input_path.open("r", encoding="utf-8") as source, temp_path.open(
            "w", encoding="utf-8"
        ) as target:
            for line_number, line in enumerate(source, 1):
                counts["input_rows"] += 1
                row = _load_json_row(line, input_path, line_number)
                counts["status_accept_rows"] += int(row.get("visual_filter_status") == "accept")
                counts["reason_verified_negative_rows"] += int(
                    row.get("visual_filter_reason") == "verified_negative"
                )
                if not _is_verified_negative(row):
                    counts["filtered_visual_rows"] += 1
                    continue
                counts["verified_negative_rows"] += 1
                key = _int_key(
                    row,
                    ("ref_id", "ann_id", "sent_id"),
                    f"{input_path}:{line_number}",
                )
                split = split_map.get(key)
                if split is None:
                    counts["missing_split_rows"] += 1
                    continue
                counts[f"official_split_{split}_rows"] += 1
                if split != "train":
                    counts["filtered_non_train_rows"] += 1
                    continue
                if key in selected_keys:
                    counts["duplicate_selected_keys"] += 1
                    continue
                _validate_training_row(row, f"{input_path}:{line_number}")
                selected_keys.add(key)
                _write_training_row(target, row, source_splitby="unc")
                counts["output_rows"] += 1
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    os.replace(temp_path, output_path)
    return dict(sorted(counts.items())), selected_keys


def _write_g(
    input_path: Path,
    output_path: Path,
    google_map: Mapping[RefKey, str],
    umd_map: Mapping[UmdSentenceKey, str],
) -> Tuple[Dict[str, Any], set[RefKey], set[UmdSentenceKey]]:
    counts: Counter[str] = Counter()
    selected_identity_keys: set[RefSentenceKey] = set()
    selected_google_keys: set[RefKey] = set()
    selected_umd_keys: set[UmdSentenceKey] = set()
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with input_path.open("r", encoding="utf-8") as source, temp_path.open(
            "w", encoding="utf-8"
        ) as target:
            for line_number, line in enumerate(source, 1):
                counts["input_rows"] += 1
                row = _load_json_row(line, input_path, line_number)
                counts["status_accept_rows"] += int(row.get("visual_filter_status") == "accept")
                counts["reason_verified_negative_rows"] += int(
                    row.get("visual_filter_reason") == "verified_negative"
                )
                if not _is_verified_negative(row):
                    counts["filtered_visual_rows"] += 1
                    continue
                counts["verified_negative_rows"] += 1

                identity_key = _int_key(
                    row,
                    ("ref_id", "ann_id", "sent_id"),
                    f"{input_path}:{line_number}",
                )
                google_key = _int_key(
                    row,
                    ("ref_id", "ann_id", "image_id"),
                    f"{input_path}:{line_number}",
                )
                sentence = _normalize_sentence(row.get("sent"))
                umd_key = (int(row["image_id"]), int(row["ann_id"]), sentence)
                google_split = google_map.get(google_key)
                umd_split = umd_map.get(umd_key)
                if google_split is None:
                    counts["missing_google_split_rows"] += 1
                    continue
                if umd_split is None:
                    counts["missing_umd_split_rows"] += 1
                    continue
                counts[f"google_{google_split}_umd_{umd_split}_rows"] += 1
                if google_split != "train" or umd_split != "train":
                    counts["filtered_non_train_rows"] += 1
                    continue
                if identity_key in selected_identity_keys:
                    counts["duplicate_selected_keys"] += 1
                    continue
                _validate_training_row(row, f"{input_path}:{line_number}")
                selected_identity_keys.add(identity_key)
                selected_google_keys.add(google_key)
                selected_umd_keys.add(umd_key)
                _write_training_row(target, row, source_splitby="google+umd")
                counts["output_rows"] += 1
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    os.replace(temp_path, output_path)
    return dict(sorted(counts.items())), selected_google_keys, selected_umd_keys


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--plus-input", type=Path, default=PLUS_INPUT)
    parser.add_argument("--g-input", type=Path, default=G_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-plus-rows", type=int, default=EXPECTED_PLUS_ROWS)
    parser.add_argument("--expected-g-rows", type=int, default=EXPECTED_G_ROWS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plus_refs = data_root / "COCO" / "refcoco+" / "refs(unc).p"
    g_google_refs = data_root / "COCO" / "refcocog" / "refs(google).p"
    g_umd_refs = data_root / "COCO" / "refcocog" / "refs(umd).p"
    plus_map, plus_map_duplicates = _build_plus_split_map(plus_refs)
    google_map, google_map_duplicates = _build_google_split_map(g_google_refs)
    umd_map, umd_map_duplicates = _build_umd_split_map(g_umd_refs)

    plus_output = output_dir / PLUS_OUTPUT_NAME
    g_output = output_dir / G_OUTPUT_NAME
    plus_counts, plus_selected = _write_plus(args.plus_input.resolve(), plus_output, plus_map)
    g_counts, g_google_selected, g_umd_selected = _write_g(
        args.g_input.resolve(), g_output, google_map, umd_map
    )

    plus_eval_keys = {key for key, split in plus_map.items() if split != "train"}
    google_eval_keys = {key for key, split in google_map.items() if split != "train"}
    umd_eval_keys = {key for key, split in umd_map.items() if split != "train"}
    plus_eval_overlap = len(plus_selected & plus_eval_keys)
    google_eval_overlap = len(g_google_selected & google_eval_keys)
    umd_eval_overlap = len(g_umd_selected & umd_eval_keys)

    _require(plus_counts.get("missing_split_rows", 0) == 0, "RefCOCO+ rows missing UNC split")
    _require(g_counts.get("missing_google_split_rows", 0) == 0, "RefCOCOg rows missing Google split")
    _require(g_counts.get("missing_umd_split_rows", 0) == 0, "RefCOCOg rows missing UMD split")
    _require(plus_counts.get("duplicate_selected_keys", 0) == 0, "Duplicate RefCOCO+ output key")
    _require(g_counts.get("duplicate_selected_keys", 0) == 0, "Duplicate RefCOCOg output key")
    _require(
        plus_counts.get("output_rows", 0) == args.expected_plus_rows,
        f"Expected {args.expected_plus_rows} RefCOCO+ rows, got {plus_counts.get('output_rows', 0)}",
    )
    _require(
        g_counts.get("output_rows", 0) == args.expected_g_rows,
        f"Expected {args.expected_g_rows} RefCOCOg rows, got {g_counts.get('output_rows', 0)}",
    )
    _require(plus_eval_overlap == 0, f"RefCOCO+ output overlaps {plus_eval_overlap} UNC eval keys")
    _require(google_eval_overlap == 0, f"RefCOCOg output overlaps {google_eval_overlap} Google eval keys")
    _require(umd_eval_overlap == 0, f"RefCOCOg output overlaps {umd_eval_overlap} UMD eval keys")

    stats = {
        "filter": {
            "visual_filter_status": "accept",
            "visual_filter_reason": "verified_negative",
        },
        "refcocoplus": {
            "input": str(args.plus_input.resolve()),
            "output": str(plus_output),
            "split_source": str(plus_refs),
            "split_key": ["ref_id", "ann_id", "sent_id"],
            "split_map_rows": len(plus_map),
            "split_map_duplicate_rows": plus_map_duplicates,
            "counts": plus_counts,
        },
        "refcocog": {
            "input": str(args.g_input.resolve()),
            "output": str(g_output),
            "google_split_source": str(g_google_refs),
            "google_split_key": ["ref_id", "ann_id", "image_id"],
            "google_split_map_rows": len(google_map),
            "google_split_map_duplicate_rows": google_map_duplicates,
            "umd_split_source": str(g_umd_refs),
            "umd_split_key": ["image_id", "ann_id", "normalized_sentence"],
            "umd_sentence_normalization": "lowercase; replace '_' and '.' with spaces; collapse whitespace",
            "umd_split_map_rows": len(umd_map),
            "umd_split_map_duplicate_rows": umd_map_duplicates,
            "counts": g_counts,
        },
        "validation": {
            "expected_refcocoplus_rows": args.expected_plus_rows,
            "expected_refcocog_rows": args.expected_g_rows,
            "expected_total_rows": args.expected_plus_rows + args.expected_g_rows,
            "actual_total_rows": plus_counts["output_rows"] + g_counts["output_rows"],
            "refcocoplus_unc_eval_overlap": plus_eval_overlap,
            "refcocog_google_eval_overlap": google_eval_overlap,
            "refcocog_umd_eval_overlap": umd_eval_overlap,
            "passed": True,
        },
    }
    stats_path = output_dir / STATS_NAME
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {plus_counts['output_rows']} rows -> {plus_output}")
    print(f"wrote {g_counts['output_rows']} rows -> {g_output}")
    print(f"wrote validation stats -> {stats_path}")


if __name__ == "__main__":
    main()
