#!/usr/bin/env python3
"""Add RefCOCO train positives to the rebuilt pure-GDINO Stage-B recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_stageb_gdino_finetune_ablation import _convert_phrase_jsonl


def _repo_root() -> Path:
    return _REPO_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def build(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = _repo_root()
    baseline_path = _resolve(repo_root, args.baseline_datasets)
    source_path = _resolve(repo_root, args.refcoco_source)
    output_path = _resolve(repo_root, args.refcoco_vg_output)
    candidate_path = _resolve(repo_root, args.candidate_datasets)
    stats_path = _resolve(repo_root, args.stats_output)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_train = list(baseline.get("train", []))
    if len(baseline_train) != 5:
        raise ValueError(f"Expected 5 fixed baseline train entries, got {len(baseline_train)}")
    if [entry.get("dataset_mode") for entry in baseline_train] != ["odvg"] * 5:
        raise ValueError("Every fixed baseline train entry must use dataset_mode=odvg")
    if "lvis" not in str(baseline_train[0].get("anno", "")).lower():
        raise ValueError("Fixed baseline entry 0 is not LVIS")
    if "coco" not in str(baseline_train[1].get("anno", "")).lower():
        raise ValueError("Fixed baseline entry 1 is not COCO")
    if "refcocoplus" not in str(baseline_train[2].get("anno", "")).lower():
        raise ValueError("Fixed baseline entry 2 is not RefCOCO+")
    if "refcocog" not in str(baseline_train[3].get("anno", "")).lower():
        raise ValueError("Fixed baseline entry 3 is not RefCOCOg")
    if "tn" not in str(baseline_train[4].get("anno", "")).lower():
        raise ValueError("Fixed baseline entry 4 is not TN")

    conversion = _convert_phrase_jsonl(
        {"anno": str(source_path), "box_format": "xywh"},
        output_path,
        force_empty_negative=False,
        progress_interval=int(args.progress_interval),
    )
    if int(conversion["rows_out"]) != int(conversion["positive_rows"]):
        raise RuntimeError(f"RefCOCO conversion did not produce positive-only rows: {conversion}")
    if int(conversion["negative_rows"]) != 0 or int(conversion["skipped_empty_positive"]) != 0:
        raise RuntimeError(f"Unexpected RefCOCO conversion loss: {conversion}")

    refcoco_entry = {
        "dataset_mode": "odvg",
        "root": "/",
        "anno": str(output_path.resolve()),
        "mix_weight": 2.0,
    }
    candidate_train = [*baseline_train[:2], refcoco_entry, *baseline_train[2:]]
    candidate = {"train": candidate_train, "val": list(baseline.get("val", []))}

    # The candidate is a strict insertion: removing RefCOCO must recover the
    # fixed baseline dataset dictionary exactly, including entry order.
    if candidate_train[:2] + candidate_train[3:] != baseline_train:
        raise RuntimeError("Candidate changed a fixed baseline train entry")
    if candidate["val"] != baseline.get("val", []):
        raise RuntimeError("Candidate changed the fixed baseline validation entries")

    _write_json(candidate_path, candidate)
    stats = {
        "baseline_datasets": str(baseline_path),
        "baseline_datasets_sha256": _sha256(baseline_path),
        "candidate_datasets": str(candidate_path),
        "candidate_datasets_sha256": _sha256(candidate_path),
        "baseline_train_entries": len(baseline_train),
        "candidate_train_entries": len(candidate_train),
        "refcoco_insertion_index": 2,
        "refcoco_mix_weight": 2.0,
        "fixed_baseline_entries_exactly_reused": True,
        "refcoco_source_sha256": _sha256(source_path),
        "refcoco_vg_sha256": _sha256(output_path),
        "conversion": conversion,
    }
    _write_json(stats_path, stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-datasets",
        default="config/ablations/gdino_ft_stage_b_rebuild_20260711/datasets_gdino_ft_stageb_with_tn_local.json",
    )
    parser.add_argument(
        "--refcoco-source",
        default="data/ablations/stageb_refexp_three_train_20260711/refcoco_stageb_phrase_v1.jsonl",
    )
    parser.add_argument(
        "--refcoco-vg-output",
        default="data/ablations/gdino_ft_stage_b_rebuild_three_ref_20260711/stageb_gdino_ft_refcoco_stageb_phrase_v1_vg.jsonl",
    )
    parser.add_argument(
        "--candidate-datasets",
        default="config/ablations/gdino_ft_stage_b_rebuild_20260711/datasets_gdino_ft_stageb_three_ref_with_tn_local.json",
    )
    parser.add_argument(
        "--stats-output",
        default="data/ablations/gdino_ft_stage_b_rebuild_three_ref_20260711/stageb_gdino_ft_three_ref_stats.json",
    )
    parser.add_argument("--progress-interval", type=int, default=50000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
