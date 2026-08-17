#!/usr/bin/env python3
"""Render the sealed U2-v5 mechanism and confirmatory paper tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLOCK = ROOT / "outputs/u2v5_cvpr_ablation_20260817"
DEFAULT_ANCHOR = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817"
SEEDS = (17, 42, 73)
VAL3 = ("refcoco_val", "refcocop_val", "refcocog_val")
TEST5 = (
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_test",
)
SCHEMA = "pivot.stageb.u2v5_ablation_paper_tables/v1"


class TableError(RuntimeError):
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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TableError(f"expected JSON object: {path}")
    return value


def _seed(row: dict[str, Any]) -> int:
    text = f"{row.get('run_id', '')} {row.get('checkpoint', '')}"
    match = re.search(r"seed(17|42|73)(?:\D|$)", text)
    if not match:
        raise TableError(f"cannot recover formal seed from {text!r}")
    return int(match.group(1))


def _mean_sd(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) != 3:
        raise TableError(f"formal table requires exactly three seeds, got {len(values)}")
    return {
        "by_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def _ref_metrics(path: Path, datasets: tuple[str, ...], *, run_contains: str = "") -> dict[str, Any]:
    rows = _load(path).get("refcoco")
    if not isinstance(rows, list):
        raise TableError(f"missing Ref rows: {path}")
    grouped: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in SEEDS}
    for row in rows:
        if run_contains and run_contains not in str(row.get("run_id", "")):
            continue
        dataset = str(row.get("dataset", ""))
        if dataset in datasets:
            grouped[_seed(row)][dataset] = row
    by_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        if set(grouped[seed]) != set(datasets):
            raise TableError(f"{path} seed{seed} does not contain {datasets}")
        split = {name: float(grouped[seed][name]["acc50"]) for name in datasets}
        total = sum(int(grouped[seed][name]["num_expressions"]) for name in datasets)
        micro = sum(
            split[name] * int(grouped[seed][name]["num_expressions"])
            for name in datasets
        ) / total
        by_seed[seed] = {**split, "macro": statistics.mean(split.values()), "micro": micro}
    return {
        "source": _record(path),
        "by_seed": {str(seed): by_seed[seed] for seed in SEEDS},
        "macro": _mean_sd(by_seed[seed]["macro"] for seed in SEEDS),
        "micro": _mean_sd(by_seed[seed]["micro"] for seed in SEEDS),
    }


def _tn_metrics(path: Path, *, run_contains: str = "") -> dict[str, Any]:
    rows = _load(path).get("tn")
    if not isinstance(rows, list):
        raise TableError(f"missing TN rows: {path}")
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        if run_contains and run_contains not in str(row.get("run_id", "")):
            continue
        selected[_seed(row)] = row
    if set(selected) != set(SEEDS):
        raise TableError(f"{path} does not contain one TN result per formal seed")
    return {
        "source": _record(path),
        "fpr95": _mean_sd(selected[seed]["fpr95tpr"] for seed in SEEDS),
        "pair_win": _mean_sd(selected[seed]["pair_win_rate"] for seed in SEEDS),
    }


def _holm(entries: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(entries, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        adjusted[name] = running
    return adjusted


def _bootstrap_tables(paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    reports, families = {}, {"admission": [], "confidence_data": [], "ownership": []}
    family = {
        "admission": "admission",
        "confidence": "confidence_data",
        "matched_scope": "confidence_data",
        "ownership_isolation": "ownership",
        "ownership_schedule": "ownership",
    }
    for path in paths:
        payload = _load(path)
        if payload.get("schema") != "pivot.stageb.u2v5_paired_bootstrap/v1":
            raise TableError(f"bootstrap schema mismatch: {path}")
        contrast = str(payload.get("contrast", path.stem))
        if contrast in reports:
            raise TableError(f"duplicate bootstrap contrast {contrast}")
        reports[contrast] = {"source": _record(path)}
        endpoint_ps = []
        for endpoint in ("test5", "strict2031", "strict1607"):
            if endpoint in payload:
                reports[contrast][endpoint] = payload[endpoint]
                if endpoint != "strict1607":
                    endpoint_ps.append(float(payload[endpoint]["one_sided_p"]))
        if contrast == "ownership_schedule":
            ni = payload.get("test5", {}).get("noninferiority", {}).get("one_sided_p")
            strict = payload.get("strict2031", {}).get("one_sided_p")
            if ni is None or strict is None:
                raise TableError("ownership_schedule requires Test5 NI and strict superiority")
            raw_p = max(float(ni), float(strict))
            reports[contrast]["intersection_union_p"] = raw_p
        elif contrast == "ownership_isolation":
            strict = payload.get("strict2031", {}).get("one_sided_p")
            if strict is None:
                raise TableError("ownership_isolation requires strict superiority")
            raw_p = float(strict)
            reports[contrast]["test5_role"] = (
                "paired route-preservation effect and CI; not a superiority null"
            )
        elif endpoint_ps:
            raw_p = max(endpoint_ps)
        else:
            raise TableError(f"bootstrap {contrast} has no confirmatory endpoint")
        reports[contrast]["family_raw_p"] = raw_p
        if contrast in family:
            families[family[contrast]].append((contrast, raw_p))
    adjusted = {}
    for name, entries in families.items():
        if entries:
            adjusted[name] = _holm(entries)
            for contrast, value in adjusted[name].items():
                reports[contrast]["holm_adjusted_p"] = value
    return reports, adjusted


def _write(path: Path, text: str) -> None:
    path = path.resolve()
    if path.exists():
        raise TableError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# U2-v5 CVPR ablation tables", "", "## Admission (val3 only)", "", "| Row | val3 macro Acc@0.5 | SD |", "|---|---:|---:|"]
    for row, metrics in payload["mechanism"]["admission"].items():
        if "macro" in metrics:
            lines.append(f"| {row} | {_fmt(metrics['macro']['mean'])} | {_fmt(metrics['macro']['sample_sd'])} |")
    lines.extend(["", "## Confidence/data calibration", "", "| Row | FPR95 | SD |", "|---|---:|---:|"])
    for row, metrics in payload["mechanism"]["confidence_data"].items():
        lines.append(f"| {row} | {_fmt(metrics['fpr95']['mean'])} | {_fmt(metrics['fpr95']['sample_sd'])} |")
    lines.extend(["", "## Ownership mechanism", "", "| Row | val3 macro | calibration FPR95 |", "|---|---:|---:|"])
    for row, metrics in payload["mechanism"]["ownership"].items():
        lines.append(f"| {row} | {_fmt(metrics['ref']['macro']['mean'])} | {_fmt(metrics['confidence']['fpr95']['mean'])} |")
    lines.extend(["", "## Preregistered paired contrasts", "", "| Contrast | Test5 gain | strict2031 reduction | Holm p |", "|---|---:|---:|---:|"])
    for name, report in payload["confirmatory"]["bootstrap"].items():
        test5 = report.get("test5", {}).get("gain", math.nan)
        strict = report.get("strict2031", {}).get("gain", math.nan)
        lines.append(f"| {name} | {_fmt(test5)} | {_fmt(strict)} | {_fmt(report.get('holm_adjusted_p', math.nan))} |")
    lines.extend(["", "Test5 is the confirmatory Ref endpoint; val3 and Ref8 aggregates are descriptive. strict1607 is a nested subset derived from the strict2031 forward.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-root", default=str(DEFAULT_BLOCK))
    parser.add_argument("--anchor-root", default=str(DEFAULT_ANCHOR))
    parser.add_argument("--confirmatory-manifest", required=True)
    parser.add_argument("--bootstrap", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    block, anchor = Path(args.block_root), Path(args.anchor_root)
    mechanism = block / "evaluations/mechanism"
    admission = {
        row: _ref_metrics(mechanism / row / "val3/summary.json", VAL3)
        for row in ("A1", "A3", "A4")
    }
    admission["A2"] = {"deployment_parity": _record(mechanism / "A2/deployment_parity.json")}
    admission["A5"] = _ref_metrics(anchor / "formal/ref_val3_admission_u100/summary.json", VAL3)
    confidence_data = {
        row: _tn_metrics(mechanism / row / ("d3_calibration/summary.json" if row.startswith("C") else "calibration/summary.json"))
        for row in ("C1", "C2", "C4", "D1", "D2", "D2m", "D3m")
    }
    confidence_data["C3"] = _tn_metrics(anchor / "formal/calibration_u25_u50_u100/summary.json", run_contains="_u50_")
    ownership = {}
    for row in ("O0", "O1", "O2"):
        ownership[row] = {
            "ref": _ref_metrics(mechanism / row / "val3/summary.json", VAL3),
            "confidence": _tn_metrics(mechanism / row / "d3_calibration/summary.json"),
        }
    ownership["O3"] = {"ref": admission["A5"], "confidence": confidence_data["C3"]}
    bootstraps, holm = _bootstrap_tables([Path(path) for path in args.bootstrap])
    manifest = Path(args.confirmatory_manifest)
    payload = {
        "schema": SCHEMA,
        "mechanism": {"admission": admission, "confidence_data": confidence_data, "ownership": ownership},
        "confirmatory": {
            "manifest": _record(manifest),
            "bootstrap": bootstraps,
            "holm_families": holm,
        },
        "reporting_contract": {
            "ref_primary": "Test5 micro Acc@0.5",
            "strict_primary": "strict2031 FPR95",
            "strict1607": "nested robustness derived from strict2031 records",
            "seed_summary": "per seed, mean, sample SD (ddof=1)",
        },
    }
    _write(Path(args.output_json), json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write(Path(args.output_md), _markdown(payload))
    print(json.dumps({"json": _record(Path(args.output_json)), "markdown": _record(Path(args.output_md))}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TableError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
