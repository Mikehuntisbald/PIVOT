#!/usr/bin/env python3
"""Render the compact ARROW FineCops-Ref paper table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import load_json


def _pm(row: dict) -> str:
    return f"{100 * row['mean']:.2f} ± {100 * row['sample_sd']:.2f}"


def render(results_path: Path) -> str:
    payload = load_json(results_path)
    aggregate = payload["seed_aggregates"]
    lines = [
        "# ARROW FineCops-Ref external zero-shot evaluation",
        "",
        "> FineCops-specific benchmark zero-shot; this surface is not image-disjoint from historical model training.",
        "",
        "| Route | Positive P@1 macro | Positive P@1 micro | Neg-text Recall@1 | Neg-image Recall@1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for surface in ("B58", "R100_D3", "A", "B", "C"):
        row = aggregate[surface]
        label = {
            "R100_D3": "R100 + D3 confidence",
            "A": "ARROW-A support patch (95.60% covered)",
            "B": "ARROW-B category text",
            "C": "ARROW-C learned null",
        }.get(surface, surface)
        lines.append(
            f"| {label} | {_pm(row['precision1_macro'])} | "
            f"{_pm(row['precision1_micro'])} | {_pm(row['text_recall1'])} | "
            f"{_pm(row['image_recall1'])} |"
        )
    lines.extend(["", "## Pre-registered admission contrasts", ""])
    lines.append("| Contrast | Gain (pp) | 95% image-cluster CI (pp) | Holm p |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, row in payload["bootstrap"]["contrasts"].items():
        low, high = row["ci95"]
        lines.append(
            f"| {name.replace('_', ' ')} | {100 * row['gain']:.2f} | "
            f"[{100 * low:.2f}, {100 * high:.2f}] | {row['holm_adjusted_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "A metrics exclude unsupported rows and never count unsupported negatives as correct rejection. "
            "B/C and both baselines use the complete 27,926-record test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "outputs/arrow_finecops_20260819/results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/arrow_finecops_20260819/paper_table.md",
    )
    args = parser.parse_args()
    rendered = render(args.results.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
