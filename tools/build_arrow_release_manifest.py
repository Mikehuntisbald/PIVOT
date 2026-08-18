#!/usr/bin/env python3
"""Build the public ARROW release manifest without rewriting legacy artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "arrow.release_manifest/v1"
METHOD_NAME = "ARROW"
METHOD_LONG_NAME = (
    "Admission, Ranking, and Rejection with Ownership-Separated Weights "
    "for Reliable Visual Grounding"
)
DEFAULT_PREREG = ROOT / "outputs/u2v5_cvpr_ablation_20260817/preregistration.json"
DEFAULT_FINAL = ROOT / "outputs/u2v5_cvpr_ablation_20260817/final_receipt_v2.json"
DEFAULT_TABLES = ROOT / "outputs/u2v5_cvpr_ablation_20260817/paper_tables_v2.json"
DEFAULT_SUPPLEMENT = ROOT / "outputs/u2v5_cvpr_ablation_20260817/zero_training_supplement_v3.json"
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/formal"


class ArrowReleaseError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _json(path: Path, *, schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = file_record(path)
    payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ArrowReleaseError(
            f"legacy artifact {path} does not have required schema {schema!r}"
        )
    return dict(payload), record


def _git() -> dict[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    return {"commit": commit, "status": "clean" if not status.strip() else "dirty"}


def build_manifest(
    *, preregistration: Path, final_receipt: Path, paper_tables: Path,
    supplement: Path, checkpoint_root: Path,
) -> dict[str, Any]:
    prereg, prereg_record = _json(
        preregistration, schema="pivot.stageb.u2v5_ablation_preregistration/v1"
    )
    final, final_record = _json(
        final_receipt, schema="pivot.stageb.u2v5_ablation_final_receipt/v1"
    )
    _, table_record = _json(
        paper_tables, schema="pivot.stageb.u2v5_ablation_paper_tables/v1"
    )
    _, supplement_record = _json(
        supplement, schema="pivot.stageb.u2v5_zero_training_supplement/v1"
    )
    if final.get("preregistration") != prereg_record:
        raise ArrowReleaseError("final receipt does not bind the selected preregistration")
    checkpoints = {}
    for seed in (17, 42, 73):
        path = checkpoint_root / f"confidence_seed{seed}_u50/checkpoint_iter.pth"
        checkpoints[str(seed)] = file_record(path)
    return {
        "schema": SCHEMA,
        "method": {"name": METHOD_NAME, "long_name": METHOD_LONG_NAME},
        "implementation": {
            "lineage": "u2v5",
            "legacy_public_name": "PIVOT",
            "legacy_schema_namespace": "pivot.stageb",
            "legacy_artifacts_are_byte_immutable": True,
            "local_checkout_root": str(ROOT),
        },
        "git": _git(),
        "legacy_evidence": {
            "preregistration": prereg_record,
            "final_receipt": final_record,
            "paper_tables": table_record,
            "zero_training_supplement": supplement_record,
            "main_checkpoints": checkpoints,
        },
        "formal_trajectory_count": int(prereg["formal_trajectory_count"]),
        "status": "release_bound",
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise ArrowReleaseError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=str(DEFAULT_PREREG))
    parser.add_argument("--final-receipt", default=str(DEFAULT_FINAL))
    parser.add_argument("--paper-tables", default=str(DEFAULT_TABLES))
    parser.add_argument("--supplement", default=str(DEFAULT_SUPPLEMENT))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_manifest(
        preregistration=Path(args.preregistration),
        final_receipt=Path(args.final_receipt),
        paper_tables=Path(args.paper_tables),
        supplement=Path(args.supplement),
        checkpoint_root=Path(args.checkpoint_root),
    )
    if payload["git"]["status"] != "clean":
        raise ArrowReleaseError("ARROW release manifest requires a clean worktree")
    _write(Path(args.output), payload)
    print(json.dumps({"status": "complete", "manifest": file_record(Path(args.output))}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ArrowReleaseError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
