#!/usr/bin/env python3
"""Freeze proposal-verified image-global TN rows for Stage-B v15 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _answer_no(value: Any) -> bool:
    return str(value or "").strip().lower() == "no"


def _row_key(row: Dict[str, Any]) -> Tuple[str, int, int, int, int]:
    return (
        str(row.get("dataset", "")),
        int(row.get("image_id", -1)),
        int(row.get("ann_id", -1)),
        int(row.get("sent_id", -1)),
        int(row.get("ref_id", -1)),
    )


def _proposal_ids(rows: Iterable[Dict[str, Any]]) -> List[int]:
    return sorted(int(row.get("proposal_id", -1)) for row in rows)


def _admit(row: Dict[str, Any]) -> Tuple[bool, str]:
    if str(row.get("split", "")).strip().lower() != "train":
        return False, "not_train"
    if str(row.get("visual_filter_status", "")).strip().lower() != "accept":
        return False, "target_not_accepted"
    target = row.get("visual_local_judgment") or {}
    if not isinstance(target, dict) or not _answer_no(target.get("answer")):
        return False, "target_not_no"

    proposals = row.get("proposal_cache") or []
    judgments = row.get("visual_proposal_judgments") or []
    if not isinstance(proposals, list) or not proposals:
        return False, "no_proposals"
    if not isinstance(judgments, list) or len(judgments) != len(proposals):
        return False, "incomplete_proposal_judgments"
    if _proposal_ids(proposals) != _proposal_ids(judgments):
        return False, "proposal_id_mismatch"
    for judgment_row in judgments:
        judgment = judgment_row.get("judgment") or {}
        if not isinstance(judgment, dict) or not _answer_no(judgment.get("answer")):
            return False, "proposal_not_no"
    return True, "accepted"


def _canonical_json(row: Dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_one(source: Path, destination: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen = set()
    source_count = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_count += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: JSON row is not an object")
            admitted, reason = _admit(row)
            reasons[reason] += 1
            if not admitted:
                continue
            key = _row_key(row)
            if key in seen:
                raise ValueError(f"{source}:{line_number}: duplicate row key {key}")
            seen.add(key)
            frozen = dict(row)
            frozen["tn_scope"] = "image_global_proposal_verified"
            frozen["global_tn_verified"] = True
            rows.append(frozen)

    rows.sort(key=_row_key)
    payload = "".join(_canonical_json(row) + "\n" for row in rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "source_rows": source_count,
        "kept_rows": len(rows),
        "duplicate_keys": 0,
        "filter_counts": dict(sorted(reasons.items())),
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    reports = []
    for input_value in args.inputs:
        source = Path(input_value)
        destination = out_dir / f"{source.stem}.global_verified.jsonl"
        reports.append(build_one(source, destination))

    audit = {
        "schema": "stageb_v15_global_verified_train_v1",
        "definition": (
            "train row; target VLM answer=no; non-empty proposal cache; "
            "one judgment per proposal with matching IDs; every proposal answer=no"
        ),
        "total_source_rows": sum(row["source_rows"] for row in reports),
        "total_kept_rows": sum(row["kept_rows"] for row in reports),
        "files": reports,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
