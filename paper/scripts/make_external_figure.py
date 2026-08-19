#!/usr/bin/env python3
"""Render cross-benchmark rejection ordering and operating-point transfer."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from figure_common import COLORS, configure_style, load_registry, number, save_vector_pair, value, write_csv


SURFACES = (
    ("internal", "Internal", COLORS["blue"]),
    ("finecops", "FineCops", COLORS["orange"]),
    ("grefcoco", "gRefCOCO", COLORS["purple"]),
)


def point_with_optional_ci(ax, y: float, key: str, registry, color: str) -> None:
    item = number(registry, key)
    point = 100 * float(item["value"])
    if "ci95" in item:
        lo, hi = (100 * float(v) for v in item["ci95"])
        ax.errorbar(
            point,
            y,
            xerr=np.asarray([[point - lo], [hi - point]]),
            fmt="o",
            ms=5.2,
            color=color,
            capsize=2.5,
            zorder=3,
        )
    else:
        ax.scatter(point, y, s=30, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.annotate(f"+{point:.2f}", (point, y), xytext=(5, 0), textcoords="offset points", va="center", color=color, weight="bold")


def main() -> None:
    configure_style()
    registry = load_registry()
    # A compact one-column matrix follows the exact diagnostic requested in
    # the gRefCOCO protocol.  It keeps differently scaled gains in separate
    # labeled rows and makes the operating-point failure visually explicit.
    fig, ax = plt.subplots(figsize=(3.35, 2.12))
    ax.set_xlim(0, 4.05)
    ax.set_ylim(-0.15, 4.2)
    ax.axis("off")

    column_x = (1.55, 2.48, 3.41)
    for x, (_, display, color) in zip(column_x, SURFACES):
        ax.text(x, 3.82, display, ha="center", va="center", color=color, weight="bold", fontsize=7.2)

    rows_spec = (
        ("AUROC gain", "auroc_gain", 2.88),
        ("FPR95 gain", "fpr95_reduction", 1.90),
        ("Fixed $\\tau$ $\\to$ 95% TPR", "fixed_tpr", 0.92),
    )
    for row_index, (row_label, metric, y) in enumerate(rows_spec):
        ax.text(0.03, y, row_label, ha="left", va="center", weight="bold", fontsize=6.7)
        for x, (surface, _, color) in zip(column_x, SURFACES):
            key = f"cross_benchmark.{surface}.{metric}"
            point = 100 * float(number(registry, key)["value"])
            is_operating = metric == "fixed_tpr"
            passed = point >= 95.0 if is_operating else point > 0.0
            face = "#E8F5F0" if passed else "#FCEDE8"
            edge = COLORS["green"] if passed else COLORS["vermillion"]
            box = FancyBboxPatch(
                (x - 0.40, y - 0.31), 0.80, 0.62,
                boxstyle="round,pad=0.035,rounding_size=0.045",
                facecolor=face, edgecolor=edge, linewidth=0.85,
            )
            ax.add_patch(box)
            if is_operating:
                glyph = "✓" if passed else "✗"
                label = f"{point:.1f}% {glyph}"
            else:
                label = f"+{point:.2f} pp"
            ax.text(x, y, label, ha="center", va="center", color=edge, weight="bold", fontsize=7.0)

    ax.text(
        2.05, 0.12,
        "FineCops gains are point estimates; no rejection bootstrap CI.\n"
        "gRefCOCO uses the restricted Full single/no-target slice.",
        ha="center", va="center", color=COLORS["gray"], fontsize=6.3,
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.03, top=0.98)

    rows = []
    for surface, display, _ in SURFACES:
        for metric in ("auroc_gain", "fpr95_reduction", "fixed_tpr"):
            key = f"cross_benchmark.{surface}.{metric}"
            item = number(registry, key)
            ci = item.get("ci95", ("", ""))
            rows.append(
                {
                    "surface": display,
                    "metric": metric,
                    "registry_key": key,
                    "value": item["value"],
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "inference": item["notes"],
                }
            )
    write_csv(
        "fig4_external_transfer.csv",
        ["surface", "metric", "registry_key", "value", "ci95_low", "ci95_high", "inference"],
        rows,
    )
    save_vector_pair(fig, "fig4_external_transfer")
    plt.close(fig)


if __name__ == "__main__":
    main()
