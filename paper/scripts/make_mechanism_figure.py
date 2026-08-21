#!/usr/bin/env python3
"""Render the two observed ranking/abstention interference regimes."""

from __future__ import annotations

import statistics

from matplotlib import pyplot as plt
import numpy as np

from figure_common import COLORS, configure_style, load_registry, save_vector_pair, value, write_csv


SEEDS = (17, 42, 73)


def mean_pct(registry, prefix: str, metric: str) -> float:
    return 100 * statistics.mean(value(registry, f"{prefix}.seed{seed}.{metric}") for seed in SEEDS)


def main() -> None:
    configure_style()
    registry = load_registry()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.42), gridspec_kw={"width_ratios": [1, 1]})

    # a) On the frozen-base representation, mean cosine can hide recurrent
    # negative events. Show minima and the scheduled negative-event fraction.
    ax = axes[0]
    y = np.arange(len(SEEDS))
    minima = np.asarray([
        value(registry, f"b58_capacity.shared_wide.seed{seed}.cosine_min")
        for seed in SEEDS
    ])
    negative = np.asarray([
        100 * value(registry, f"b58_capacity.shared_wide.seed{seed}.negative_cosine_fraction")
        for seed in SEEDS
    ])
    ax.axvline(0, color=COLORS["black"], lw=0.8)
    for index, (point, fraction) in enumerate(zip(minima, negative)):
        ax.hlines(index, point, 0, color=COLORS["vermillion"], lw=2.2)
        ax.scatter(point, index, s=42, color=COLORS["vermillion"], edgecolor="white", zorder=3)
        ax.annotate(f"min {point:+.3f}\n{fraction:.1f}% neg.", (point, index),
                    xytext=(5, -1), textcoords="offset points", ha="left", va="center",
                    color=COLORS["vermillion"], fontsize=6.8)
    ax.set_yticks(y, [f"seed {seed}" for seed in SEEDS])
    ax.set_xlim(-0.78, 0.10)
    ax.set_ylim(3.45, -0.75)
    ax.set_xlabel("minimum scheduled gradient cosine")
    ax.set_title("a  Frozen base: intermittent conflict", loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    shared_rec = mean_pct(registry, "b58_capacity.shared_wide", "test5")
    isolated_rec = mean_pct(registry, "b58_capacity.isolated", "test5")
    shared_fpr = mean_pct(registry, "b58_capacity.shared_wide", "fpr95")
    isolated_fpr = mean_pct(registry, "b58_capacity.isolated", "fpr95")
    ax.text(
        0.02, 0.02,
        f"REC  {shared_rec:.2f} → {isolated_rec:.2f} (isolated)\n"
        f"FPR95  {shared_fpr:.2f} → {isolated_fpr:.2f} (n.s.)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, weight="bold",
        bbox={"facecolor": "white", "edgecolor": COLORS["light_gray"], "pad": 2.2},
    )

    # b) On strong e5, the same U150 probe is close to orthogonal and changes
    # sign across seeds. Shared-Wide is at least as good at the endpoints.
    ax = axes[1]
    cosine = np.asarray([
        value(registry, f"strong_ownership.shared_wide.seed{seed}.cosine_mean")
        for seed in SEEDS
    ])
    ax.axvline(0, color=COLORS["black"], lw=0.8)
    for index, point in enumerate(cosine):
        color = COLORS["vermillion"] if point < 0 else COLORS["blue"]
        ax.hlines(index, 0, point, color=color, lw=2.2)
        ax.scatter(point, index, s=42, color=color, edgecolor="white", zorder=3)
        ax.annotate(f"{point:+.3f}", (point, index),
                    xytext=(-5 if point < 0 else 5, 0), textcoords="offset points",
                    ha="right" if point < 0 else "left", va="center", color=color,
                    fontsize=7.0, weight="bold")
    ax.set_yticks(y, [f"seed {seed}" for seed in SEEDS])
    ax.set_xlim(-0.050, 0.055)
    ax.set_ylim(3.45, -0.75)
    ax.set_xlabel("U150 mean gradient cosine")
    ax.set_title("b  Strong e5: effective sharing", loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    strong_shared_rec = 100 * value(registry, "strong_ownership.shared_wide.testab")
    strong_isolated_rec = 100 * value(registry, "strong_ownership.isolated_128.testab")
    strong_shared_fpr = 100 * value(registry, "strong_ownership.shared_wide.strict_fpr95")
    strong_isolated_fpr = 100 * value(registry, "strong_ownership.isolated_128.strict_fpr95")
    ax.text(
        0.02, 0.02,
        f"REC  Shared-Wide {strong_shared_rec:.3f} | Isolated {strong_isolated_rec:.3f}\n"
        f"FPR95  Shared-Wide {strong_shared_fpr:.3f} | Isolated {strong_isolated_fpr:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, weight="bold",
        bbox={"facecolor": "white", "edgecolor": COLORS["light_gray"], "pad": 2.2},
    )

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.87, wspace=0.34)

    rows = []
    for seed, minimum, fraction in zip(SEEDS, minima, negative):
        rows.append({
            "panel": "frozen_base", "seed": seed,
            "metric": "minimum_cosine", "value": minimum,
            "secondary_metric": "negative_probe_fraction", "secondary_value": fraction / 100,
        })
    for seed, point in zip(SEEDS, cosine):
        rows.append({
            "panel": "strong_e5", "seed": seed,
            "metric": "u150_mean_cosine", "value": point,
            "secondary_metric": "", "secondary_value": "",
        })
    write_csv(
        "fig3_mechanism_controllability.csv",
        ["panel", "seed", "metric", "value", "secondary_metric", "secondary_value"],
        rows,
    )
    save_vector_pair(fig, "fig3_mechanism_controllability")
    plt.close(fig)


if __name__ == "__main__":
    main()
