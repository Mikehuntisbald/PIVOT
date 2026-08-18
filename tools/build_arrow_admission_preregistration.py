#!/usr/bin/env python3
"""Seal the ARROW Admission-input design before any B/C model result exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "arrow.stageb.admission_input_preregistration/v1"
OUTPUT_ROOT = ROOT / "outputs/arrow_admission_input_20260818"
PANEL = ROOT / "data/ablations/stageb_table_a_category_intervention_20260717"


class ArrowPreregistrationError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ArrowPreregistrationError(f"non-object row in {path}")
                yield dict(row)


def _images(paths: Iterable[Path]) -> tuple[set[int], int]:
    images, count = set(), 0
    for path in paths:
        for row in _rows(path):
            if "image_id" not in row:
                raise ArrowPreregistrationError(f"row lacks image_id: {path}")
            images.add(int(row["image_id"]))
            count += 1
    return images, count


def _training_paths() -> list[Path]:
    config = json.loads(
        (ROOT / "config/datasets_stageb_u2_category_complete_three_ref.json")
        .read_text(encoding="utf-8")
    )
    result = []
    for entry in config["train"]:
        value = str(entry["anno"])
        if value.startswith("/home/user/PIVOT/"):
            value = str(ROOT / value.removeprefix("/home/user/PIVOT/"))
        result.append(Path(value).resolve(strict=True))
    return result


def _ref8_record_paths() -> list[Path]:
    summary = json.loads(
        (ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/"
         "ref8_u50/summary.json").read_text(encoding="utf-8")
    )
    paths = []
    for row in summary["refcoco"]:
        path = Path(row["records_jsonl"])
        if path not in paths:
            paths.append(path.resolve(strict=True))
    return paths


def _git() -> dict[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ArrowPreregistrationError("ARROW preregistration requires a clean worktree")
    return {"commit": commit, "status": "clean"}


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise ArrowPreregistrationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (OUTPUT_ROOT / "training").exists():
        raise ArrowPreregistrationError("training root already exists before design lock")
    audit_path = PANEL / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("evidence_status") != "runtime_inputs_built_no_model_results":
        raise ArrowPreregistrationError("fresh panel already contains model evidence")
    panel_path = PANEL / "category_intervention_pairs.jsonl"
    train_paths = _training_paths()
    ref_paths = _ref8_record_paths()
    strict_path = ROOT / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/eval_manifest.jsonl"
    panel_images, panel_rows = _images([panel_path])
    train_images, train_rows = _images(train_paths)
    ref_images, ref_rows = _images(ref_paths)
    strict_images, strict_rows = _images([strict_path])
    if len(panel_images) != 512 or panel_rows != 1024:
        raise ArrowPreregistrationError("fresh panel count drifted")
    overlaps = {
        "admission_train": len(panel_images & train_images),
        "ref8": len(panel_images & ref_images),
        "strict2031": len(panel_images & strict_images),
    }
    if any(overlaps.values()):
        raise ArrowPreregistrationError(f"fresh panel image overlap: {overlaps}")
    inputs = {
        "panel_audit": _record(audit_path),
        "panel_rows": _record(panel_path),
        "panel_support": _record(PANEL / "category_intervention_support.tsv"),
        "strict2031": _record(strict_path),
        "clean_initializer": _record(
            ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/initializer/checkpoint_clean_init.pth"
        ),
        "dataset_config": _record(ROOT / "config/datasets_stageb_u2_category_complete_three_ref.json"),
    }
    for index, path in enumerate(train_paths):
        inputs[f"admission_train_{index}"] = _record(path)
    for relative in (
        "models/GroundingDINO/groundingdino.py", "engine.py", "main.py",
        "tools/stageb_arrow_admission_contract.py",
        "tools/run_arrow_admission_matrix.py",
        "tools/merge_arrow_admission_confidence.py",
        "tools/eval_text_groundingdino_refcoco_tn.py",
        "tools/eval_refcoco_stageb.py",
        "tools/eval_stageb_role_causal.py",
        "tools/eval_arrow_admission_panel.py",
        "tools/build_arrow_admission_checkpoint_lock.py",
        "tools/run_arrow_admission_evaluations.py",
        "tools/aggregate_arrow_admission_results.py",
        "tools/render_arrow_admission_table.py",
        "tools/build_arrow_admission_final_receipt.py",
        "tests/test_stageb_arrow_admission.py",
        "docs/paper_cvpr_arrow_admission_input_protocol_20260818.md",
    ):
        inputs[f"code_{relative}"] = _record(ROOT / relative)
    payload = {
        "schema": SCHEMA,
        "status": "locked_before_training",
        "git": _git(),
        "rows": {
            "AR_A_PATCH": {"source": "support_patch", "training": "reuse_sealed_A5"},
            "AR_B_TEXT": {"source": "canonical_text", "seeds": [17, 42, 73], "updates": 100},
            "AR_C_NULL": {"source": "learned_null", "seeds": [17, 42, 73], "updates": 100},
        },
        "shared_contract": {
            "physical_batch": 56, "gap": 3.0, "trainable_tensors": 16,
            "full_expression_geometry_and_r100": True,
            "confidence_and_strict_reused_by_bitwise_parity": True,
            "milestone_selection": False,
        },
        "fresh_panel": {
            "unique_images": len(panel_images), "rows": panel_rows,
            "overlap": overlaps,
            "comparison_row_counts": {
                "admission_train": train_rows, "ref8": ref_rows,
                "strict2031": strict_rows,
            },
            "primary_metric": "pair_level_bidirectional_category_switch_success",
        },
        "planned_contrasts": {
            "visual_over_text": {"candidate": "AR_A_PATCH", "reference": "AR_B_TEXT", "direction": "greater"},
            "category_over_null": {"candidate": "AR_B_TEXT", "reference": "AR_C_NULL", "direction": "greater"},
        },
        "bootstrap": {"iterations": 5000, "generator": "PCG64", "seed": 20260818, "cluster": "image_pair", "holm_family": ["visual_over_text", "category_over_null"]},
        "test5": {"status": "prospectively_frozen_post_release", "noninferiority_margin": 0.005, "strict_forward": False},
        "inputs": inputs,
    }
    _write(Path(args.output), payload)
    print(json.dumps({"status": "locked", "receipt": _record(Path(args.output))}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ArrowPreregistrationError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
