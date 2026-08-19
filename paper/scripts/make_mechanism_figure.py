#!/usr/bin/env python3
"""Render the gradient-conflict, ownership, and controllability evidence."""

from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np

from figure_common import COLORS, ci95, configure_style, load_registry, save_vector_pair, value, write_csv


SEEDS = (17, 42, 73)


def main() -> None:
    configure_style()
    registry = load_registry()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.42), gridspec_kw={"width_ratios": [1.05, 1.05, 1.25]})

    # a) Directly observed conflict on the shared-score owner (O0).
    ax = axes[0]
    x = np.arange(len(SEEDS))
    cosine_keys = [f"ownership.o0.seed{s}.cosine_mean" for s in SEEDS]
    conflict_keys = [f"ownership.o0.seed{s}.sign_conflict_mean" for s in SEEDS]
    cosine = [100 * value(registry, key) for key in cosine_keys]
    conflict = [100 * value(registry, key) for key in conflict_keys]
    width = 0.34
    ax.bar(x - width / 2, cosine, width, color=COLORS["vermillion"], label="gradient cosine")
    ax.bar(x + width / 2, conflict, width, color=COLORS["purple"], label="sign conflict")
    ax.axhline(0, color=COLORS["black"], lw=0.7)
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("training seed")
    ax.set_ylabel("diagnostic (%)")
    ax.set_ylim(-34, 82)
    ax.set_title("a  Shared score conflicts", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.0, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    # b) Isolation changes rejection while preserving localization.
    ax = axes[1]
    owner_keys = [
        "ownership.o2_minus_o0.test5_gain",
        "ownership.o2_minus_o0.strict_fpr95_reduction",
    ]
    labels = ["Test5 localization\ngain", "Strict-TN2031\nFPR95 reduction"]
    colors = [COLORS["orange"], COLORS["green"]]
    y = np.arange(2)
    points = np.asarray([100 * value(registry, key) for key in owner_keys])
    intervals = [ci95(registry, key) for key in owner_keys]
    xerr = np.asarray(
        [
            [points[i] - 100 * intervals[i][0] for i in range(2)],
            [100 * intervals[i][1] - points[i] for i in range(2)],
        ]
    )
    for i in range(2):
        ax.errorbar(points[i], y[i], xerr=xerr[:, i : i + 1], fmt="o", ms=5, color=colors[i], capsize=2.5, zorder=3)
    ax.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("isolated $-$ shared (percentage points)")
    ax.set_xlim(-1.0, 5.3)
    ax.set_title("b  Isolation preserves route", loc="left", weight="bold")
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, point in enumerate(points):
        ax.annotate(f"{point:+.2f}", (point, y[i]), xytext=(4, -10), textcoords="offset points", color=colors[i], weight="bold")

    # c) Capacity-matched Admission inputs expose controllability, not just Acc.
    ax = axes[2]
    routes = ("arrow_v", "arrow_t", "arrow_n")
    names = ("ARROW-V", "ARROW-T", "ARROW-N")
    route_colors = (COLORS["blue"], COLORS["orange"], COLORS["gray"])
    test_keys = [f"admission_input.{route}.test5" for route in routes]
    switch_keys = [f"admission_input.{route}.switch_success" for route in routes]
    test = np.asarray([100 * value(registry, key) for key in test_keys])
    switch = np.asarray([100 * value(registry, key) for key in switch_keys])
    y = np.arange(len(routes))
    for i, color in enumerate(route_colors):
        ax.plot([switch[i], test[i]], [y[i], y[i]], color=COLORS["light_gray"], lw=2, zorder=1)
        ax.scatter(switch[i], y[i], s=28, marker="s", color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(test[i], y[i], s=30, marker="o", facecolor="white", edgecolor=color, linewidth=1.5, zorder=3)
    ax.set_yticks(y, names)
    ax.set_xlim(-3, 81)
    ax.set_ylim(2.5, -0.7)
    ax.set_xlabel("success / Acc@0.5 (%)")
    ax.set_title("c  Accuracy hides control", loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=28, marker="s", color=COLORS["black"], label="category switch")
    ax.scatter([], [], s=30, marker="o", facecolor="white", edgecolor=COLORS["black"], label="Test5")
    ax.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.4, columnspacing=0.8)
    for i in range(3):
        ax.annotate(f"{switch[i]:.1f}", (switch[i], y[i]), xytext=(0, -10), textcoords="offset points", ha="center", color=route_colors[i])
        ax.annotate(f"{test[i]:.1f}", (test[i], y[i]), xytext=(0, 5), textcoords="offset points", ha="center", color=route_colors[i])

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.23, top=0.88, wspace=0.58)

    rows = []
    for seed, cosine_key, conflict_key in zip(SEEDS, cosine_keys, conflict_keys):
        rows.extend(
            [
                {"panel": "a", "series": "gradient_cosine", "label": f"seed{seed}", "registry_key": cosine_key, "value": value(registry, cosine_key), "ci95_low": "", "ci95_high": ""},
                {"panel": "a", "series": "sign_conflict_fraction", "label": f"seed{seed}", "registry_key": conflict_key, "value": value(registry, conflict_key), "ci95_low": "", "ci95_high": ""},
            ]
        )
    for label, key in zip(labels, owner_keys):
        lo, hi = ci95(registry, key)
        rows.append({"panel": "b", "series": "ownership_contrast", "label": label.replace("\n", " "), "registry_key": key, "value": value(registry, key), "ci95_low": lo, "ci95_high": hi})
    for name, test_key, switch_key in zip(names, test_keys, switch_keys):
        rows.extend(
            [
                {"panel": "c", "series": "test5", "label": name, "registry_key": test_key, "value": value(registry, test_key), "ci95_low": "", "ci95_high": ""},
                {"panel": "c", "series": "category_switch", "label": name, "registry_key": switch_key, "value": value(registry, switch_key), "ci95_low": "", "ci95_high": ""},
            ]
        )
    write_csv(
        "fig3_mechanism_controllability.csv",
        ["panel", "series", "label", "registry_key", "value", "ci95_low", "ci95_high"],
        rows,
    )
    save_vector_pair(fig, "fig3_mechanism_controllability")
    plt.close(fig)


if __name__ == "__main__":
    main()
