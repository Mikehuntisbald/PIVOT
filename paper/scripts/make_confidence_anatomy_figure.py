#!/usr/bin/env python3
"""Plot committed cross-benchmark score-CDF anatomy for the supplement.

Normal paper builds read only committed paper data.  The empirical score
distributions were reduced to quantile knots by ``build_confidence_anatomy``
on the experiment host; this renderer never opens model outputs or fits a
threshold.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_common import (
    COLORS,
    PAPER,
    SOURCE_DIR,
    configure_style,
    load_registry,
    number,
    save_vector_pair,
    write_csv,
)


RECEIPT = PAPER / "data" / "confidence_anatomy.json"
QUANTILES = SOURCE_DIR / "confidence_quantiles.csv"
OUTPUT_DIR = PAPER / "supplement" / "figures"
SEED = "42"
QUANTILE_COLUMNS = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
    ("q99", 0.99),
)
BENCHMARKS = (
    ("Internal Strict-TN2031", "Internal\nStrict-TN2031", "internal"),
    ("FineCops-Ref", "FineCops-Ref", "finecops"),
    ("gRefCOCO restricted Full", "gRefCOCO\nrestricted Full", "grefcoco"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs() -> tuple[dict[str, Any], list[dict[str, str]]]:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("schema") != "arrow.paper.confidence_anatomy/v1":
        raise RuntimeError(f"unexpected confidence receipt schema: {receipt.get('schema')!r}")
    if receipt.get("status") != "zero_training_derived_from_sealed_records":
        raise RuntimeError(f"confidence receipt is not sealed: {receipt.get('status')!r}")
    with QUANTILES.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("confidence quantile receipt is empty")
    return receipt, rows


def select_row(
    rows: list[dict[str, str]], benchmark: str, route: str, label: str, seed: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["benchmark"] == benchmark
        and row["route"] == route
        and row["label"] == label
        and row["seed"] == seed
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one quantile row for {(benchmark, route, label, seed)}, "
            f"found {len(matches)}"
        )
    return matches[0]


def cdf_knots(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray([float(row[column]) for column, _ in QUANTILE_COLUMNS])
    cumulative = np.asarray([probability for _, probability in QUANTILE_COLUMNS])
    if not np.all(np.diff(scores) >= 0.0):
        raise RuntimeError("quantile scores are not monotone")
    return scores, cumulative


def route_seed(benchmark: str, route: str) -> str:
    if benchmark == "Internal Strict-TN2031" and route == "Frozen base":
        return "fixed"
    return SEED


def negative_labels(benchmark: str) -> tuple[str, ...]:
    if benchmark == "FineCops-Ref":
        return ("negative-text", "negative-image")
    if benchmark == "gRefCOCO restricted Full":
        return ("no-target",)
    return ("negative",)


def main() -> None:
    configure_style()
    receipt, quantile_rows = read_inputs()
    registry = load_registry()
    tau_key = f"cross_benchmark.sealed_source_tau.seed{SEED}"
    tau_item = number(registry, tau_key)
    tau = float(tau_item["value"])
    receipt_tau = float(
        receipt["surfaces"]["internal"]["isolated_rejector_by_seed"][SEED][
            "sealed_d3_threshold"
        ]
    )
    if tau != receipt_tau:
        raise RuntimeError("semantic registry and confidence receipt disagree on sealed source tau")

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.35), sharey=True)
    routes = ("Frozen base", "Isolated rejector")
    curve_rows: list[dict[str, Any]] = []
    quantile_sha = sha256(QUANTILES)
    receipt_sha = sha256(RECEIPT)

    for column, (benchmark, title, surface_key) in enumerate(BENCHMARKS):
        for row_index, route in enumerate(routes):
            ax = axes[row_index, column]
            seed = route_seed(benchmark, route)
            positive = select_row(quantile_rows, benchmark, route, "positive", seed)
            pos_x, pos_y = cdf_knots(positive)
            ax.plot(
                pos_x,
                pos_y,
                marker="o",
                markersize=2.6,
                color=COLORS["blue"],
                label="positive",
                zorder=3,
            )
            for quantile_name, probability in QUANTILE_COLUMNS:
                curve_rows.append(
                    {
                        "panel": f"{row_index + 1}{chr(97 + column)}",
                        "benchmark": benchmark,
                        "route": route,
                        "seed": seed,
                        "series": "positive",
                        "kind": "cdf_knot",
                        "quantile": probability,
                        "score": positive[quantile_name],
                        "registry_key": "",
                        "source_path": str(QUANTILES.relative_to(PAPER)),
                        "source_sha256": quantile_sha,
                        "notes": "empirical distribution summarized at committed quantile knots",
                    }
                )

            negative_styles = (
                (COLORS["vermillion"], "--"),
                (COLORS["purple"], ":"),
            )
            for negative_index, label in enumerate(negative_labels(benchmark)):
                negative = select_row(quantile_rows, benchmark, route, label, seed)
                neg_x, neg_y = cdf_knots(negative)
                color, linestyle = negative_styles[negative_index]
                display = {
                    "negative": "verified negative",
                    "negative-text": "negative text",
                    "negative-image": "negative image",
                    "no-target": "no-target",
                }[label]
                ax.plot(
                    neg_x,
                    neg_y,
                    marker="o",
                    markersize=2.2,
                    color=color,
                    linestyle=linestyle,
                    label=display,
                    zorder=2,
                )
                for quantile_name, probability in QUANTILE_COLUMNS:
                    curve_rows.append(
                        {
                            "panel": f"{row_index + 1}{chr(97 + column)}",
                            "benchmark": benchmark,
                            "route": route,
                            "seed": seed,
                            "series": label,
                            "kind": "cdf_knot",
                            "quantile": probability,
                            "score": negative[quantile_name],
                            "registry_key": "",
                            "source_path": str(QUANTILES.relative_to(PAPER)),
                            "source_sha256": quantile_sha,
                            "notes": "empirical distribution summarized at committed quantile knots",
                        }
                    )

            if route == "Isolated rejector":
                domain_q05 = float(positive["q05"])
                ax.axvline(
                    domain_q05,
                    color=COLORS["black"],
                    linestyle=(0, (1.2, 1.8)),
                    linewidth=1.0,
                    zorder=1,
                )
                ax.axvline(
                    tau,
                    color=COLORS["green"],
                    linestyle="-.",
                    linewidth=1.1,
                    zorder=1,
                )
                fixed_tpr_key = f"cross_benchmark.{surface_key}.fixed_tpr"
                fixed_tpr = float(number(registry, fixed_tpr_key)["value"])
                ax.text(
                    0.98,
                    0.08,
                    f"3-seed mean TPR@τ: {100 * fixed_tpr:.1f}%",
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                    color=COLORS["green"],
                    fontsize=7.0,
                    weight="bold",
                )
                curve_rows.extend(
                    [
                        {
                            "panel": f"{row_index + 1}{chr(97 + column)}",
                            "benchmark": benchmark,
                            "route": route,
                            "seed": seed,
                            "series": "positive",
                            "kind": "domain_positive_q05",
                            "quantile": 0.05,
                            "score": domain_q05,
                            "registry_key": "",
                            "source_path": str(QUANTILES.relative_to(PAPER)),
                            "source_sha256": quantile_sha,
                            "notes": "domain q05 from committed quantile receipt; diagnostic, not deployment calibration",
                        },
                        {
                            "panel": f"{row_index + 1}{chr(97 + column)}",
                            "benchmark": benchmark,
                            "route": route,
                            "seed": seed,
                            "series": "isolated_rejector",
                            "kind": "sealed_source_tau",
                            "quantile": "",
                            "score": tau,
                            "registry_key": tau_key,
                            "source_path": tau_item["source_path"],
                            "source_sha256": tau_item["source_sha256"],
                            "notes": tau_item["notes"],
                        },
                        {
                            "panel": f"{row_index + 1}{chr(97 + column)}",
                            "benchmark": benchmark,
                            "route": route,
                            "seed": "17/42/73 mean",
                            "series": "isolated_rejector",
                            "kind": "fixed_tau_positive_tpr",
                            "quantile": "",
                            "score": fixed_tpr,
                            "registry_key": fixed_tpr_key,
                            "source_path": number(registry, fixed_tpr_key)["source_path"],
                            "source_sha256": number(registry, fixed_tpr_key)["source_sha256"],
                            "notes": number(registry, fixed_tpr_key)["notes"],
                        },
                    ]
                )

            ax.set_ylim(0.0, 1.0)
            ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
            ax.grid(color="#EEEEEE", linewidth=0.55, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("raw score")
            if column == 0:
                ax.set_ylabel(f"{route}\nCDF")
            if row_index == 0:
                ax.set_title(f"{chr(97 + column)}  {title}", loc="left", weight="bold")

    legend_handles = [
        Line2D([0], [0], color=COLORS["blue"], marker="o", markersize=3, label="positive"),
        Line2D([0], [0], color=COLORS["vermillion"], linestyle="--", marker="o", markersize=3, label="negative text / verified / no-target"),
        Line2D([0], [0], color=COLORS["purple"], linestyle=":", marker="o", markersize=3, label="negative image (FineCops)"),
        Line2D([0], [0], color=COLORS["black"], linestyle=(0, (1.2, 1.8)), label="domain positive q05"),
        Line2D([0], [0], color=COLORS["green"], linestyle="-.", label="sealed source τ"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.20, hspace=0.42, wspace=0.34)

    # Bind the entire derived plot to both committed receipts.  The raw
    # host-local paths embedded in the anatomy receipt are intentionally never
    # copied into this public plot CSV.
    curve_rows.append(
        {
            "panel": "all",
            "benchmark": "all",
            "route": "all",
            "seed": "all",
            "series": "receipt",
            "kind": "receipt_binding",
            "quantile": "",
            "score": "",
            "registry_key": "",
            "source_path": str(RECEIPT.relative_to(PAPER)),
            "source_sha256": receipt_sha,
            "notes": "zero-training confidence-anatomy receipt; no model output is read by this renderer",
        }
    )
    for row in curve_rows:
        # The package validator compares the conventional ``value`` column
        # against semantic-number rows.  Distribution knots remain bound by
        # source receipt SHA instead and therefore leave it empty.
        row["value"] = row["score"] if row["registry_key"] else ""
    write_csv(
        "figS1_confidence_anatomy.csv",
        [
            "panel",
            "benchmark",
            "route",
            "seed",
            "series",
            "kind",
            "quantile",
            "score",
            "value",
            "registry_key",
            "source_path",
            "source_sha256",
            "notes",
        ],
        curve_rows,
    )
    save_vector_pair(
        fig,
        "figS1_confidence_anatomy",
        directory=OUTPUT_DIR,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
