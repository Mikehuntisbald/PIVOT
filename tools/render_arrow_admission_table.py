#!/usr/bin/env python3
"""Render the ARROW Admission-input A/B/C paper table."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _fmt(value) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if results.get("schema") != "arrow.stageb.admission_input_results/v1":
        raise RuntimeError("ARROW result schema drifted")
    panel = results["panel"]["contrasts"]
    lines = [
        "# ARROW Admission-input ablation", "",
        "| Version | Admission input | val3 micro Acc@0.5 | Fresh-panel role |",
        "|---|---|---:|---|",
        f"| A | support patch | {_fmt(results['val3']['A']['mean'])} | candidate in A>B |",
        f"| B | canonical category text | {_fmt(results['val3']['B']['mean'])} | A>B reference; B>C candidate |",
        f"| C | learned null/category-agnostic token | {_fmt(results['val3']['C']['mean'])} | B>C reference |",
        "", "## Planned fresh-panel contrasts", "",
        "| Contrast | Gain | 95% CI | Holm p |", "|---|---:|---:|---:|",
    ]
    for name in ("visual_over_text", "category_over_null"):
        row = panel[name]
        lines.append(
            f"| {name} | {_fmt(row['gain'])} | "
            f"[{_fmt(row['ci95'][0])}, {_fmt(row['ci95'][1])}] | "
            f"{_fmt(row['holm_adjusted_p'])} |"
        )
    lines.extend(["", "Test5 is a prospectively frozen post-release no-collapse analysis; strict records are reused only after bitwise confidence parity.", ""])
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
