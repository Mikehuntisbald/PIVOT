#!/usr/bin/env python3
"""Render Admission controllability and cross-benchmark threshold transfer."""

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


def main() -> None:
    configure_style()
    registry = load_registry()
    fig = plt.figure(figsize=(3.35, 3.62))

    # Top: almost equal standard localization can conceal whether the named
    # Admission input actually controls category routing.
    ax = fig.add_axes([0.24, 0.59, 0.73, 0.32])
    routes = ("arrow_v", "arrow_t", "arrow_n")
    names = ("Visual support", "Category text", "Learned null")
    route_colors = (COLORS["blue"], COLORS["orange"], COLORS["gray"])
    test = np.asarray([100 * value(registry, f"admission_input.{route}.test5") for route in routes])
    switch = np.asarray([100 * value(registry, f"admission_input.{route}.switch_success") for route in routes])
    y = np.arange(3)
    for index, color in enumerate(route_colors):
        ax.plot([switch[index], test[index]], [y[index], y[index]], color=COLORS["light_gray"], lw=2.1)
        ax.scatter(switch[index], y[index], s=30, marker="s", color=color,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(test[index], y[index], s=34, marker="o", facecolor="white",
                   edgecolor=color, linewidth=1.5, zorder=3)
        ax.annotate(f"{switch[index]:.1f}", (switch[index], y[index]),
                    xytext=(0, -10), textcoords="offset points", ha="center", color=color)
        ax.annotate(f"{test[index]:.1f}", (test[index], y[index]),
                    xytext=(0, 5), textcoords="offset points", ha="center", color=color)
    ax.set_yticks(y, names)
    ax.set_xlim(-3, 80)
    ax.set_ylim(2.5, -0.65)
    ax.set_xlabel("success / Acc@0.5 (%)")
    ax.set_title("a  Accuracy does not prove cue control", loc="left", weight="bold", pad=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=30, marker="s", color=COLORS["black"], label="category switch")
    ax.scatter([], [], s=34, marker="o", facecolor="white", edgecolor=COLORS["black"], label="Test5")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              handletextpad=0.35, columnspacing=0.75)

    # Bottom: exact matrix requested by the cross-benchmark protocol.
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.49])
    ax.set_xlim(0, 4.05); ax.set_ylim(-0.15, 4.18); ax.axis("off")
    ax.text(0.03, 4.10, "b  Ordering transfers; operating points shift",
            ha="left", va="top", fontsize=8.5, weight="bold")
    column_x = (1.55, 2.48, 3.41)
    for x, (_, display, color) in zip(column_x, SURFACES):
        ax.text(x, 3.54, display, ha="center", va="center", color=color,
                weight="bold", fontsize=7.2)
    for row_label, metric, y_coord in (
        ("AUROC gain", "auroc_gain", 2.65),
        ("FPR95 gain", "fpr95_reduction", 1.72),
        ("Fixed $\\tau$ → 95% TPR", "fixed_tpr", 0.79),
    ):
        ax.text(0.03, y_coord, row_label, ha="left", va="center", weight="bold", fontsize=6.7)
        for x, (surface, _, _) in zip(column_x, SURFACES):
            point = 100 * float(number(registry, f"cross_benchmark.{surface}.{metric}")["value"])
            operating = metric == "fixed_tpr"
            passed = point >= 95.0 if operating else point > 0.0
            face = "#E8F5F0" if passed else "#FCEDE8"
            edge = COLORS["green"] if passed else COLORS["vermillion"]
            ax.add_patch(FancyBboxPatch(
                (x - 0.40, y_coord - 0.29), 0.80, 0.58,
                boxstyle="round,pad=0.035,rounding_size=0.045",
                facecolor=face, edgecolor=edge, linewidth=0.85,
            ))
            label = f"{point:.1f}% {'✓' if passed else '✗'}" if operating else f"+{point:.2f} pp"
            ax.text(x, y_coord, label, ha="center", va="center", color=edge,
                    weight="bold", fontsize=6.9)
    ax.text(
        2.05, 0.12,
        "FineCops ordering gains are point estimates; gRefCOCO is the\n"
        "restricted single/no-target slice. The source threshold is never refit.",
        ha="center", va="center", color=COLORS["gray"], fontsize=6.45,
    )

    rows = []
    for name, route in zip(names, routes):
        rows.extend([
            {"panel": "admission", "surface": name, "metric": "test5",
             "registry_key": f"admission_input.{route}.test5",
             "value": value(registry, f"admission_input.{route}.test5"), "inference": "standard endpoint"},
            {"panel": "admission", "surface": name, "metric": "category_switch",
             "registry_key": f"admission_input.{route}.switch_success",
             "value": value(registry, f"admission_input.{route}.switch_success"), "inference": "functional control"},
        ])
    for surface, display, _ in SURFACES:
        for metric in ("auroc_gain", "fpr95_reduction", "fixed_tpr"):
            key = f"cross_benchmark.{surface}.{metric}"
            item = number(registry, key)
            rows.append({"panel": "transfer", "surface": display, "metric": metric,
                         "registry_key": key, "value": item["value"], "inference": item["notes"]})
    write_csv(
        "fig4_external_transfer.csv",
        ["panel", "surface", "metric", "registry_key", "value", "inference"],
        rows,
    )
    save_vector_pair(fig, "fig4_external_transfer")
    plt.close(fig)


if __name__ == "__main__":
    main()
