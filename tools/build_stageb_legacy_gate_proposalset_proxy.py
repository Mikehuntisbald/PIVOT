#!/usr/bin/env python3
"""Relabel cached-proposal VLM rows as an explicit non-global proxy set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _is_no(value: Any) -> bool:
    return str(value or "").strip().lower() == "no"


def _key(row: Dict[str, Any]) -> Tuple[str, int, int, int, int]:
    return (
        str(row.get("dataset", "")),
        int(row.get("image_id", -1)),
        int(row.get("ann_id", -1)),
        int(row.get("sent_id", -1)),
        int(row.get("ref_id", -1)),
    )


def _validate_cached_proposal_proxy(row: Dict[str, Any], source: Path, line_no: int) -> None:
    target = row.get("visual_local_judgment") or {}
    proposals = row.get("proposal_cache") or []
    judgments = row.get("visual_proposal_judgments") or []
    if not isinstance(target, dict) or not _is_no(target.get("answer")):
        raise ValueError(f"{source}:{line_no}: target is not VLM no")
    if not isinstance(proposals, list) or not proposals:
        raise ValueError(f"{source}:{line_no}: cached proposal set is empty")
    if not isinstance(judgments, list) or len(judgments) != len(proposals):
        raise ValueError(f"{source}:{line_no}: cached proposal judgments are incomplete")
    proposal_ids = sorted(int(item.get("proposal_id", -1)) for item in proposals)
    judgment_ids = sorted(int(item.get("proposal_id", -1)) for item in judgments)
    if proposal_ids != judgment_ids:
        raise ValueError(f"{source}:{line_no}: cached proposal IDs do not match judgments")
    if any(
        not isinstance(item.get("judgment"), dict)
        or not _is_no(item["judgment"].get("answer"))
        for item in judgments
    ):
        raise ValueError(f"{source}:{line_no}: cached proposal is not VLM no")


def convert(source: Path, destination: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_no}: JSON row must be an object")
            _validate_cached_proposal_proxy(row, source, line_no)
            row_key = _key(row)
            if row_key in seen:
                raise ValueError(f"{source}:{line_no}: duplicate key {row_key}")
            seen.add(row_key)
            proxy = dict(row)
            proxy["source_tn_scope"] = row.get("tn_scope", None)
            proxy["global_tn_verified"] = False
            proxy["proposalset_proxy_verified"] = True
            proxy["tn_scope"] = "proposal_set_verified"
            rows.append(proxy)

    rows.sort(key=_key)
    payload = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="ascii")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "rows": len(rows),
        "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
        "label_scope": "proposalset_proxy_not_image_global",
        "tn_scope": "proposal_set_verified",
        "global_tn_verified": False,
        "proposalset_proxy_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for source in args.inputs:
        destination = args.out_dir / f"{source.stem}.proposalset_proxy.jsonl"
        reports.append(convert(source, destination))
    audit = {
        "schema": "stageb_legacy_gate_proposalset_proxy_v1",
        "warning": (
            "Rows verify only the annotated target and cached SAM3 proposal set; "
            "they do not verify the actual frozen model candidate set."
        ),
        "total_rows": sum(report["rows"] for report in reports),
        "files": reports,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
