#!/usr/bin/env python3
"""Recover exact positive/TN pairs for the fixed Stage-B data-FT protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict


_WS = re.compile(r"\s+")


def _clean_text(value: Any) -> str:
    return _WS.sub(
        " ", str(value or "").replace("_", " ").replace(".", " ").strip()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_line(raw: str, *, path: Path, line_number: int) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except Exception as error:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}:{line_number}")
    return value


def _source_fields(row: Dict[str, Any], *, line_number: int):
    instances = row.get("instances", None)
    if not isinstance(instances, list) or len(instances) != 1 or not isinstance(instances[0], dict):
        raise ValueError(f"source row {line_number} must contain exactly one instance")
    instance = instances[0]
    if instance.get("text_is_negative", None) is not True:
        raise ValueError(f"source row {line_number} is not an exact TN row")
    negative = _clean_text(
        instance.get("raw_phrase")
        or instance.get("negative_phrase")
        or instance.get("phrase")
    )
    positive = _clean_text(instance.get("positive_phrase"))
    if not negative or not positive or negative.casefold() == positive.casefold():
        raise ValueError(
            f"source row {line_number} lacks a distinct positive/TN expression"
        )
    bbox = instance.get("bbox", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"source row {line_number} lacks a four-value xywh bbox")
    x, y, width, height = [float(value) for value in bbox]
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"source row {line_number} has an invalid bbox")
    if instance.get("class_id", None) is None:
        raise ValueError(f"source row {line_number} lacks class_id")
    filename = str(row.get("filename", "")).strip()
    if not filename:
        raise ValueError(f"source row {line_number} lacks filename")
    return instance, positive, negative, [x, y, width, height], filename


def _validate_dataft_row(
    row: Dict[str, Any],
    *,
    source_row: Dict[str, Any],
    positive: str,
    negative: str,
    filename: str,
    line_number: int,
) -> None:
    grounding = row.get("grounding", None)
    if not isinstance(grounding, dict):
        raise ValueError(f"data-FT row {line_number} lacks grounding payload")
    if grounding.get("is_negative", None) is not True:
        raise ValueError(f"data-FT row {line_number} is not marked negative")
    if grounding.get("regions", None) != []:
        raise ValueError(f"data-FT row {line_number} must have zero regions")
    caption_list = grounding.get("caption_list", None)
    if not isinstance(caption_list, list) or len(caption_list) != 1:
        raise ValueError(f"data-FT row {line_number} must contain one TN phrase")
    if _clean_text(caption_list[0]).casefold() != negative.casefold():
        raise ValueError(f"TN text drift at paired row {line_number}")
    tn_records = grounding.get("tn_records", None)
    if not isinstance(tn_records, list) or len(tn_records) != 1:
        raise ValueError(f"data-FT row {line_number} lacks its TN audit record")
    if _clean_text(tn_records[0].get("positive_phrase")).casefold() != positive.casefold():
        raise ValueError(f"positive text drift at paired row {line_number}")
    if str(row.get("filename", "")) != filename:
        raise ValueError(f"filename drift at paired row {line_number}")
    if int(row.get("image_id", -1)) != int(source_row.get("image_id", -2)):
        raise ValueError(f"image_id drift at paired row {line_number}")


def _paired_row(
    source_row: Dict[str, Any],
    instance: Dict[str, Any],
    positive: str,
    negative: str,
    bbox,
    filename: str,
    line_number: int,
) -> Dict[str, Any]:
    identity = {
        key: source_row.get(key, None)
        for key in ("image_id", "ann_id", "ref_id", "sent_id", "split")
    }
    output_instance = {
        "bbox": bbox,
        "class_id": int(instance["class_id"]),
        "raw_phrase": positive,
        "phrase": positive,
        "head": instance.get("head", None),
        "head_phrase": instance.get("head_phrase", None),
        "canonical_name": instance.get("canonical_name", None),
        "positive_phrase": positive,
        "negative_phrase": negative,
        "try_tn": negative,
        "try_tn_head": instance.get("try_tn_head", None),
        "try_tn_head_phrase": positive,
        "text_is_negative": False,
        "sam3_tn_pair": True,
        "replace_from": instance.get("replace_from", None),
        "replace_to": instance.get("replace_to", None),
        "replace_category": instance.get("replace_category", None),
        "benchmark_dataft_alltn": True,
        "global_tn_verified": False,
        "proposalset_proxy_verified": False,
        "tn_scope": "benchmark_dataft_alltn",
    }
    return {
        "filename": filename,
        "source": "stage_b_gdino_adapter_benchmark_dataft_alltn",
        "pair_source": source_row.get("source", "refexp_tn_stageb_v1"),
        **identity,
        "sample_id": "dataft-alltn:" + ":".join(
            str(identity.get(key, ""))
            for key in ("image_id", "ann_id", "ref_id", "sent_id")
        ),
        "dataft_row_index": line_number - 1,
        "benchmark_dataft_alltn": True,
        "global_tn_verified": False,
        "proposalset_proxy_verified": False,
        "tn_scope": "benchmark_dataft_alltn",
        "instances": [output_instance],
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    source_path = Path(args.source_pairs).resolve()
    dataft_path = Path(args.dataft_tn).resolve()
    output_path = Path(args.output).resolve()
    audit_path = Path(args.audit).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    seen_ids = set()
    count = 0
    try:
        with source_path.open("r", encoding="utf-8") as source_handle, dataft_path.open(
            "r", encoding="utf-8"
        ) as dataft_handle, temporary.open("w", encoding="utf-8") as output_handle:
            for line_number, pair in enumerate(
                zip_longest(source_handle, dataft_handle), start=1
            ):
                source_raw, dataft_raw = pair
                if source_raw is None or dataft_raw is None:
                    raise ValueError(
                        "source pair and data-FT TN files have different row counts"
                    )
                source_row = _load_line(
                    source_raw, path=source_path, line_number=line_number
                )
                dataft_row = _load_line(
                    dataft_raw, path=dataft_path, line_number=line_number
                )
                instance, positive, negative, bbox, filename = _source_fields(
                    source_row, line_number=line_number
                )
                _validate_dataft_row(
                    dataft_row,
                    source_row=source_row,
                    positive=positive,
                    negative=negative,
                    filename=filename,
                    line_number=line_number,
                )
                identity = tuple(
                    source_row.get(key, None)
                    for key in ("image_id", "ann_id", "ref_id", "sent_id")
                )
                if None in identity or identity in seen_ids:
                    raise ValueError(
                        f"missing or duplicate source identity at row {line_number}: {identity}"
                    )
                seen_ids.add(identity)
                output_handle.write(
                    json.dumps(
                        _paired_row(
                            source_row,
                            instance,
                            positive,
                            negative,
                            bbox,
                            filename,
                            line_number,
                        ),
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )
                count += 1
        if int(args.expected_rows) > 0 and count != int(args.expected_rows):
            raise ValueError(
                f"expected {int(args.expected_rows)} exact pairs, recovered {count}"
            )
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    audit = {
        "schema": "stage-b-gdino-adapter-dataft-pairs-v1",
        "tn_scope": "benchmark_dataft_alltn",
        "rows": count,
        "source_pairs": str(source_path),
        "source_pairs_sha256": _sha256(source_path),
        "dataft_tn": str(dataft_path),
        "dataft_tn_sha256": _sha256(dataft_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "claim": (
            "Fixed Stage-B data-FT all-TN benchmark labels only; not image-global "
            "negative verification."
        ),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pairs",
        default=(
            "/home/user/datasets/pivot_data/patch_episode_prebuilt/"
            "refexp_tn_stageb_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--dataft-tn",
        default=(
            "data/ablations/gdino_ft_stage_b_rebuild_20260711/"
            "stageb_gdino_ft_refexp_tn_stageb_v1_vg_empty.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/ablations/stageb_gdino_adapter_dataft_20260711/"
            "benchmark_dataft_alltn_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--audit",
        default=(
            "data/ablations/stageb_gdino_adapter_dataft_20260711/audit.json"
        ),
    )
    parser.add_argument("--expected-rows", type=int, default=60000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
