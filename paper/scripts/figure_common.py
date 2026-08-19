#!/usr/bin/env python3
"""Shared, deterministic helpers for registry-driven ARROW paper figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
REGISTRY_PATH = PAPER / "data" / "paper_numbers.json"
FIGURE_DIR = PAPER / "figures"
SOURCE_DIR = PAPER / "data" / "plot_sources"

# Okabe--Ito palette: distinguishable under the common red/green deficiencies.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#777777",
    "light_gray": "#E5E5E5",
}


def configure_style() -> None:
    """Apply a CVPR-safe vector style with no text below 7 pt."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.titlesize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_registry() -> dict[str, Any]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "arrow.paper.semantic_number_registry/v1":
        raise RuntimeError(f"unexpected registry schema: {payload.get('schema')!r}")
    return payload


def number(registry: Mapping[str, Any], key: str) -> dict[str, Any]:
    try:
        item = registry["numbers"][key]
    except KeyError as exc:
        raise KeyError(f"paper number is not registered: {key}") from exc
    if item.get("value") is None:
        raise ValueError(f"paper number has no point value: {key}")
    return item


def value(registry: Mapping[str, Any], key: str) -> float:
    return float(number(registry, key)["value"])


def ci95(registry: Mapping[str, Any], key: str) -> tuple[float, float]:
    item = number(registry, key)
    if "ci95" not in item:
        raise KeyError(f"paper number has no registered CI: {key}")
    lo, hi = item["ci95"]
    return float(lo), float(hi)


def write_csv(name: str, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_vector_pair(
    fig: plt.Figure, stem: str, *, directory: Path | None = None
) -> tuple[Path, Path]:
    target_dir = FIGURE_DIR if directory is None else directory
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf = target_dir / f"{stem}.pdf"
    svg = target_dir / f"{stem}.svg"
    metadata = {"Creator": f"paper/scripts/{stem}.py", "Subject": "ARROW paper figure"}
    fig.savefig(pdf, metadata=metadata)
    fig.savefig(svg, metadata={"Creator": metadata["Creator"]})
    return pdf, svg
