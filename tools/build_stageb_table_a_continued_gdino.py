#!/usr/bin/env python3
"""Build the record- and exposure-matched continued-GDINO Table-A control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_stageb_gdino_finetune_ablation import (  # noqa: E402
    _clean_phrase,
    _convert_phrase_jsonl,
)


SCHEMA = "stageb-table-a-continued-gdino-data-audit-v1"


def _expand(value: str) -> str:
    return os.path.expanduser(
        os.path.expandvars(
            str(value)
            .replace("${DATA_ROOT}", os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))
            .replace("/home/user/PIVOT", str(REPO_ROOT))
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _convert_tn(entry: Mapping[str, Any], output: Path) -> Dict[str, Any]:
    source = Path(_expand(str(entry["anno"]))).resolve()
    root = Path(_expand(str(entry.get("sam3_tn_image_root", entry.get("root", "/"))))).resolve()
    rows = 0
    missing_text = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in _iter_jsonl(source):
            negative = _clean_phrase(row.get("try_tn") or row.get("negative_phrase"))
            positive = _clean_phrase(row.get("sent") or row.get("positive_phrase"))
            canonical = _clean_phrase(
                row.get("category_name") or row.get("class_norm_name") or row.get("try_tn_head")
            )
            if not negative:
                missing_text += 1
                continue
            filename = row.get("filename") or row.get("file_name")
            if not filename:
                raise ValueError("continued-GDINO TN row has no image path")
            image_path = Path(str(filename))
            if not image_path.is_absolute():
                image_path = root / image_path
            record = {
                "filename": str(image_path),
                "image_id": int(row.get("image_id", rows)),
                "grounding": {
                    "regions": [],
                    "caption": f"{negative} .",
                    "caption_list": [negative],
                    "is_negative": True,
                    "tn_records": [
                        {
                            "phrase": negative,
                            "head": canonical,
                            "head_phrase": canonical,
                            "canonical_name": canonical,
                            "positive_phrase": positive,
                            "try_tn_head_phrase": positive,
                            "text_is_negative": True,
                            "replace_from": row.get("replace_from", []),
                            "replace_to": row.get("replace_to", []),
                            "replace_category": row.get("replace_category", []),
                        }
                    ],
                },
                "table_a_source_sample_id": row.get("sample_id"),
                "table_a_tn_scope": row.get("tn_scope"),
            }
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            rows += 1
    if missing_text:
        raise ValueError(f"continued-GDINO skipped {missing_text} TN rows with empty text")
    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "rows": rows,
    }


def verify(*, dataset: Path, audit_path: Path) -> Dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != SCHEMA:
        raise ValueError("continued-GDINO audit schema mismatch")
    if _sha256(dataset) != audit["dataset"]["sha256"]:
        raise ValueError("continued-GDINO dataset manifest hash mismatch")
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    if len(manifest.get("train", [])) != 4:
        raise ValueError("continued-GDINO requires exactly three positive sources and one TN source")
    weights = [float(entry.get("mix_weight", 0.0)) for entry in manifest["train"]]
    if weights != [1.0, 1.0, 1.0, 3.0]:
        raise ValueError("continued-GDINO sampler weights must be exactly 1/1/1/3")
    observed = []
    for entry, expected in zip(manifest["train"], audit["sources"]):
        path = Path(_expand(str(entry["anno"]))).resolve()
        if _sha256(path) != expected["output_sha256"]:
            raise ValueError("continued-GDINO converted source hash mismatch")
        count = sum(1 for _ in _iter_jsonl(path))
        if count != int(expected["rows"]):
            raise ValueError("continued-GDINO converted source row count mismatch")
        observed.append(count)
    return {"rows_by_source": observed, "mix_weights": weights}


def build(args: argparse.Namespace) -> Dict[str, Any]:
    source_manifest = Path(_expand(args.source_manifest)).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    value = json.loads(source_manifest.read_text(encoding="utf-8"))
    entries = list(value.get("train", []))
    positives = [entry for entry in entries if str(entry.get("source", "")) != "sam3_tn_pair"]
    negatives = [entry for entry in entries if str(entry.get("source", "")) == "sam3_tn_pair"]
    if len(positives) != 3 or len(negatives) != 1:
        raise ValueError("source manifest must contain exactly three positive and one SAM3 TN source")
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    train = []
    for index, entry in enumerate(positives):
        output = output_dir / f"positive_{index}.jsonl"
        expanded_entry = dict(entry)
        expanded_entry["anno"] = _expand(str(entry["anno"]))
        stats = _convert_phrase_jsonl(
            expanded_entry, output, force_empty_negative=False, progress_interval=0
        )
        record = {
            "source": str(Path(_expand(str(entry["anno"]))).resolve()),
            "source_sha256": _sha256(Path(_expand(str(entry["anno"]))).resolve()),
            "output": str(output),
            "output_sha256": _sha256(output),
            "rows": int(stats["rows_out"]),
        }
        converted.append(record)
        train.append(
            {"dataset_mode": "odvg", "root": "/", "anno": str(output), "mix_weight": 1.0}
        )
    tn_output = output_dir / "tn_d3_proposal_covered.jsonl"
    tn_record = _convert_tn(negatives[0], tn_output)
    converted.append(tn_record)
    train.append(
        {"dataset_mode": "odvg", "root": "/", "anno": str(tn_output), "mix_weight": 3.0}
    )
    _write_json(dataset, {"train": train, "val": []})
    audit = {
        "schema": SCHEMA,
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": _sha256(source_manifest),
        },
        "dataset": {"path": str(dataset), "sha256": _sha256(dataset)},
        "sources": converted,
        "exposure_contract": {
            "source_mix_weights": [1.0, 1.0, 1.0, 3.0],
            "optimizer_updates_must_match_proposed_row": True,
            "global_batch_must_match_proposed_row": True,
            "horizontal_flip": False,
        },
        "architecture_limit": (
            "pure GDINO consumes the same source rows and sampler mass but cannot encode "
            "the proposal model's positive/TN paired scorer slots in one forward"
        ),
        "evidence_status": "runtime_inputs_built_no_training_results",
    }
    _write_json(audit_path, audit)
    verify(dataset=dataset, audit_path=audit_path)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = "data/ablations/stageb_table_a_continued_gdino_20260717"
    parser.add_argument(
        "--source-manifest", default="config/datasets_stageb_table_b_d3_proposal_covered.json"
    )
    parser.add_argument("--output-dir", default=base)
    parser.add_argument(
        "--dataset", default="config/datasets_stageb_table_a_g0c_continued_gdino.json"
    )
    parser.add_argument("--audit", default=f"{base}/audit.json")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        result = verify(
            dataset=Path(args.dataset).expanduser().resolve(),
            audit_path=Path(args.audit).expanduser().resolve(),
        )
        print(json.dumps({"verified": True, **result}, sort_keys=True))
        return
    result = build(args)
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "rows": [source["rows"] for source in result["sources"]],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
