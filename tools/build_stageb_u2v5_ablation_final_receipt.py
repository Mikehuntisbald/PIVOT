#!/usr/bin/env python3
"""Bind confirmatory summaries, bootstrap reports, and paper tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "pivot.stageb.u2v5_ablation_final_receipt/v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--results-manifest", required=True)
    parser.add_argument("--bootstrap", nargs="+", required=True)
    parser.add_argument("--paper-tables", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prereg = Path(args.preregistration)
    results = Path(args.results_manifest)
    tables = Path(args.paper_tables)
    prereg_payload = json.loads(prereg.read_text(encoding="utf-8"))
    results_payload = json.loads(results.read_text(encoding="utf-8"))
    if prereg_payload.get("schema") != "pivot.stageb.u2v5_ablation_preregistration/v1" or results_payload.get("schema") != "pivot.stageb.u2v5_ablation_confirmatory_results/v1":
        raise RuntimeError("final receipt input schema drifted")
    if results_payload.get("preregistration") != _record(prereg):
        raise RuntimeError("confirmatory results do not bind this preregistration")
    bootstrap = [_record(Path(path)) for path in args.bootstrap]
    for record in bootstrap:
        payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        if payload.get("schema") != "pivot.stageb.u2v5_paired_bootstrap/v1":
            raise RuntimeError("bootstrap schema drifted")
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "preregistration": _record(prereg),
        "results_manifest": _record(results),
        "bootstrap": bootstrap,
        "paper_tables": _record(tables),
        "formal_trajectory_count": 42,
        "paper_visible_formal_trajectory_count": 33,
        "paper_excluded_rows": ["D2", "D2m", "D3m"],
        "paper_excluded_contrasts": ["matched_scope"],
        "historical_excluded_artifacts_retained": True,
        "strict1607_derived_from_strict2031": True,
        "c100_excluded_from_formal_hypotheses": True,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": "complete", "receipt": _record(output)}, indent=2))


if __name__ == "__main__":
    main()
