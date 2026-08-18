#!/usr/bin/env python3
"""Bind all completed ARROW Admission-input evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "arrow.stageb.admission_input_final_receipt/v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _schema(path: Path, expected: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != expected:
        raise RuntimeError(f"{path} schema drifted")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--checkpoint-lock", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--paper-table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    release = Path(args.release_manifest).resolve(strict=True)
    prereg = Path(args.preregistration).resolve(strict=True)
    lock = Path(args.checkpoint_lock).resolve(strict=True)
    evaluation = Path(args.evaluation_manifest).resolve(strict=True)
    results = Path(args.results).resolve(strict=True)
    table = Path(args.paper_table).resolve(strict=True)
    _schema(release, "arrow.release_manifest/v1")
    _schema(prereg, "arrow.stageb.admission_input_preregistration/v1")
    _schema(lock, "arrow.stageb.admission_input_checkpoint_lock/v1")
    _schema(evaluation, "arrow.stageb.admission_input_evaluations/v1")
    result_payload = _schema(results, "arrow.stageb.admission_input_results/v1")
    if result_payload.get("strict_forwarded") is not False:
        raise RuntimeError("ARROW Admission block unexpectedly forwarded strict")
    payload = {
        "schema": SCHEMA, "status": "complete",
        "release_manifest": _record(release), "preregistration": _record(prereg),
        "checkpoint_lock": _record(lock), "evaluation_manifest": _record(evaluation),
        "results": _record(results), "paper_table": _record(table),
        "new_training_trajectories": 6, "strict_forwarded": False,
        "sealed_main_model_changed": False,
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
