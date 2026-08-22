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
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.62), gridspec_kw={"width_ratios": [1, 1]})

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

    # b) On strong continuations, a heavier negative tail does not imply that
    # hard isolation will improve either deployed endpoint.
    ax = axes[1]
    conditions = ("GDINO parent", "Adapted same-head", "MM pretrain", "MM e5", "MM e6 TN10")
    q05 = np.asarray([
        value(registry, "original_parent.shared_wide.u150.q05"),
        value(registry, "b58_raw_query.shared_wide.u150.q05"),
        value(registry, "strong_pretrain.shared_wide.u150.q05"),
        value(registry, "strong_ownership.shared_wide.u150.q05"),
        value(registry, "strong_e6.e6_tn10.shared_wide.u150.q05"),
    ])
    p_negative = np.asarray([
        value(registry, "original_parent.shared_wide.u150.p_negative"),
        value(registry, "b58_raw_query.shared_wide.u150.p_negative"),
        value(registry, "strong_pretrain.shared_wide.u150.p_negative"),
        value(registry, "strong_ownership.shared_wide.u150.p_negative"),
        value(registry, "strong_e6.e6_tn10.shared_wide.u150.p_negative"),
    ])
    ax.axvline(0, color=COLORS["black"], lw=0.8)
    y_strong = np.arange(len(conditions))
    for index, (point, fraction) in enumerate(zip(q05, p_negative)):
        ax.hlines(index, point, 0, color=COLORS["vermillion"], lw=2.2)
        ax.scatter(
            point, index, s=42, color=COLORS["vermillion"],
            edgecolor="white", zorder=3,
        )
        ax.annotate(
            f"q05 {point:+.3f}\n{100 * fraction:.1f}% neg.",
            (point, index), xytext=(5, -1), textcoords="offset points",
            ha="left", va="center", color=COLORS["vermillion"], fontsize=6.8,
        )
    ax.set_yticks(y_strong, conditions)
    ax.set_xlim(-0.59, 0.06)
    ax.set_ylim(4.75, -0.75)
    ax.set_xlabel("U150 gradient-cosine lower tail")
    ax.set_title(r"b  Strong trunk: conflict $\ne$ benefit", loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.21, top=0.88, wspace=0.34)

    rows = []
    for seed, minimum, fraction in zip(SEEDS, minima, negative):
        rows.append({
            "panel": "frozen_base", "seed": seed,
            "metric": "minimum_cosine", "value": minimum,
            "secondary_metric": "negative_probe_fraction", "secondary_value": fraction / 100,
        })
    for condition, point, fraction in zip(conditions, q05, p_negative):
        rows.append({
            "panel": "strong_continuation", "seed": condition,
            "metric": "u150_cosine_q05", "value": point,
            "secondary_metric": "negative_probe_fraction",
            "secondary_value": fraction,
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
