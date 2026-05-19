#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


COLUMNS = [
    "ablation",
    "matched_query_recall50",
    "patch_recall50",
    "patch_ap50",
    "patch_map50",
    "refcocop_acc50",
    "refcocog_acc50",
    "tn_fpr",
    "fpr95tpr",
]

ALIASES = {
    "matched_query_recall50": [
        "matched_query_recall50",
        "matched_query_recall_50",
        "matched_query_recall@50",
        "matched_query_recall_at_50",
    ],
    "patch_recall50": [
        "patch_recall50",
        "patch_recall_50",
        "patch_recall@50",
        "patch_recall_at_50",
        "patch_match_topk_recall",
    ],
    "patch_ap50": ["patch_ap50", "patch_ap_50", "patch_ap@50", "patch_ap_at_50"],
    "patch_map50": ["patch_map50", "patch_map_50", "patch_map@50", "patch_map_at_50", "patch_mAP50"],
    "refcocop_acc50": [
        "refcocop_acc50",
        "refcoco+_acc50",
        "refcoco_plus_acc50",
        "refcocop_acc_50",
        "refcocop_accuracy50",
    ],
    "refcocog_acc50": ["refcocog_acc50", "refcocog_acc_50", "refcocog_accuracy50"],
    "tn_fpr": ["tn_fpr", "false_positive_rate_tn", "tn_false_positive_rate"],
    "fpr95tpr": ["fpr95tpr", "fpr95", "fpr_at_95_tpr", "fpr@95tpr"],
}


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


NORM_ALIASES = {k: {_norm_key(x) for x in aliases} for k, aliases in ALIASES.items()}


def _iter_json_objects(path: Path) -> Iterable[Any]:
    try:
        if path.name == "log.txt" or path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                yield json.load(f)
    except Exception:
        return


def _flatten_metrics(obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                _flatten_metrics(value, out)
            else:
                out[str(key)] = value
    elif isinstance(obj, list):
        for item in obj:
            _flatten_metrics(item, out)


def _coerce_metric(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _collect_one(run_dir: Path) -> Dict[str, str]:
    found: Dict[str, str] = {}
    paths: List[Path] = []
    log_path = run_dir / "log.txt"
    if log_path.exists():
        paths.append(log_path)
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path == log_path:
            continue
        if path.suffix in {".json", ".jsonl"} and "config_args" not in path.name:
            paths.append(path)

    for path in paths:
        for obj in _iter_json_objects(path):
            flat: Dict[str, Any] = {}
            _flatten_metrics(obj, flat)
            for raw_key, value in flat.items():
                norm = _norm_key(raw_key)
                for metric, aliases in NORM_ALIASES.items():
                    if metric in found:
                        continue
                    if norm in aliases:
                        coerced = _coerce_metric(value)
                        if coerced is not None:
                            found[metric] = coerced
    return {col: found.get(col, "NA") for col in COLUMNS if col != "ablation"}


def _write_markdown(rows: List[Dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(col, "NA")) for col in COLUMNS) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/stageB_ablations")
    args = parser.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, str]] = []
    if root.exists():
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            row = {"ablation": run_dir.name}
            row.update(_collect_one(run_dir))
            rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "ablation_results.csv"
    md_path = root / "ablation_results.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(rows, md_path)
    print(f"[OUT] {csv_path}")
    print(f"[OUT] {md_path}")


if __name__ == "__main__":
    main()
