#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _changed_words(positive: str, negative: str) -> Optional[Tuple[str, str]]:
    pos_words = _WORD_RE.findall(positive)
    neg_words = _WORD_RE.findall(negative)
    if len(pos_words) != len(neg_words):
        return None
    changed = [(left, right) for left, right in zip(pos_words, neg_words) if left.lower() != right.lower()]
    if len(changed) != 1:
        return None
    return changed[0]


def _convert(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    instances = row.get("instances")
    if not isinstance(instances, list) or len(instances) != 1 or not isinstance(instances[0], dict):
        return None
    instance = instances[0]
    positive = str(instance.get("positive_phrase") or "").strip()
    negative = str(instance.get("raw_phrase") or instance.get("try_tn") or "").strip()
    changed = _changed_words(positive, negative)
    bbox = instance.get("bbox")
    class_id = instance.get("class_id")
    if not positive or not negative or changed is None or not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if class_id is None:
        return None
    replace_from, replace_to = changed
    filename = str(row.get("filename") or row.get("file_name") or "")
    canonical = (
        instance.get("canonical_name")
        or instance.get("head")
        or instance.get("head_phrase")
        or instance.get("category_name")
    )
    return {
        "image_path": filename,
        "file_name": Path(filename).name,
        "image_id": row.get("image_id"),
        "ann_id": row.get("ann_id"),
        "ref_id": row.get("ref_id"),
        "sent_id": row.get("sent_id"),
        "split": row.get("split", "train"),
        "dataset": row.get("source"),
        "pair_source": instance.get("pair_source") or row.get("source"),
        "bbox": bbox,
        "target_bbox_used": bbox,
        "class_id": int(class_id),
        "class_norm_name": canonical,
        "class_raw_name": canonical,
        "category_name": instance.get("category_name"),
        "sent": positive,
        "try_tn_head": instance.get("try_tn_head") or canonical,
        "try_tn_head_phrase": positive,
        "try_tn": negative,
        "try_tn_method": "synthetic_rule",
        "try_tn_rule": instance.get("try_tn_rule", "single_token_attribute_swap"),
        "replace_from": [replace_from],
        "replace_to": [replace_to],
        "replace_category": [instance.get("replace_category", "attribute")],
        "visual_filter_status": "target_local_only",
        "visual_filter_reason": "synthetic_target_counterfactual",
        "tn_scope": "target_local",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert negative-only RefExp rows to paired verifier rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for row in _iter_jsonl(input_path):
            counts["input_rows"] += 1
            converted = _convert(row)
            if converted is None:
                counts["skipped_rows"] += 1
                continue
            handle.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["output_rows"] += 1
            category = str(converted["replace_category"][0])
            counts[f"category_{category}"] += 1

    stats_path = Path(args.stats) if args.stats else output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(dict(counts), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(counts), indent=2, sort_keys=True))
    print(f"wrote {counts['output_rows']} rows -> {output_path}")


if __name__ == "__main__":
    main()
