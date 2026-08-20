#!/usr/bin/env python3
"""Render ARROW's inference graph and exclusive-ownership contract."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figure_common import COLORS, configure_style, save_vector_pair, write_csv


def box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    face,
    edge,
    linewidth=1.0,
    style="round,pad=0.012",
    fontsize=None,
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=style,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=COLORS["black"],
        linespacing=1.18,
        fontsize=fontsize,
    )
    return patch


def arrow(ax, start, end, *, color=COLORS["black"], dashed=False, bend=0.0, width=1.1):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=width,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=f"arc3,rad={bend}",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def main() -> None:
    configure_style()
    fig = plt.figure(figsize=(7.15, 3.15))
    graph = fig.add_axes([0.015, 0.06, 0.69, 0.89])
    owner = fig.add_axes([0.725, 0.06, 0.265, 0.89])
    for ax in (graph, owner):
        ax.set_axis_off()

    graph.text(0.0, 1.01, "a  Factorized deployment route", weight="bold", transform=graph.transAxes)
    box(graph, (0.01, 0.67), 0.15, 0.18, "image I\nexpression e", face="#F7F7F7", edge=COLORS["gray"])
    box(
        graph,
        (0.20, 0.61),
        0.27,
        0.28,
        "Frozen candidate\ngenerator  Gφ(I, e)\nqueries qᵢ + boxes bᵢ",
        face="#E8E8E8",
        edge=COLORS["gray"],
        linewidth=1.2,
        fontsize=7.1,
    )
    box(graph, (0.01, 0.27), 0.15, 0.18, "category cue u\nvisual / text / null", face="#F7F7F7", edge=COLORS["gray"])
    box(
        graph,
        (0.52, 0.65),
        0.18,
        0.20,
        "Admission owner\nA(q, u)\neligible set E",
        face="#DDEBF7",
        edge=COLORS["blue"],
        linewidth=1.3,
    )
    box(
        graph,
        (0.77, 0.65),
        0.18,
        0.20,
        "Ranking owner\nR(q, e)\nselected box",
        face="#FFF0D3",
        edge=COLORS["orange"],
        linewidth=1.3,
    )
    box(
        graph,
        (0.55, 0.19),
        0.20,
        0.20,
        "Rejection owner\nC(detached z)\naccept / abstain",
        face="#DDF3EA",
        edge=COLORS["green"],
        linewidth=1.3,
    )
    box(
        graph,
        (0.80, 0.21),
        0.15,
        0.16,
        "one box\nor abstain",
        face="#F7F7F7",
        edge=COLORS["black"],
        linewidth=1.2,
    )
    box(
        graph,
        (0.25, 0.16),
        0.20,
        0.16,
        "auxiliary residual\nnot deployed",
        face="#F1E8F5",
        edge=COLORS["purple"],
        linewidth=1.0,
    )

    arrow(graph, (0.16, 0.76), (0.20, 0.76))
    arrow(graph, (0.47, 0.76), (0.52, 0.76))
    arrow(graph, (0.16, 0.36), (0.52, 0.70), color=COLORS["blue"], bend=-0.10)
    arrow(graph, (0.70, 0.76), (0.77, 0.76), color=COLORS["blue"])
    arrow(graph, (0.16, 0.71), (0.77, 0.69), color=COLORS["orange"], bend=0.12)
    arrow(graph, (0.47, 0.66), (0.55, 0.30), color=COLORS["green"], bend=0.18)
    graph.text(0.46, 0.43, "detach", color=COLORS["green"], transform=graph.transAxes, ha="center")
    arrow(graph, (0.86, 0.65), (0.86, 0.37), color=COLORS["orange"])
    arrow(graph, (0.75, 0.29), (0.80, 0.29), color=COLORS["green"])
    arrow(graph, (0.35, 0.32), (0.58, 0.65), color=COLORS["purple"], dashed=True, bend=0.10)
    graph.text(0.31, 0.09, "dashed path carries gradients but never deployment scores", color=COLORS["purple"], transform=graph.transAxes, ha="center")

    owner.text(0.0, 1.01, "b  Exclusive owners", weight="bold", transform=owner.transAxes)
    owner.text(0.03, 0.91, "decision", weight="bold", transform=owner.transAxes)
    owner.text(0.52, 0.91, "parameter owner", weight="bold", transform=owner.transAxes)
    rows = [
        ("Geometry", "φ  frozen", COLORS["gray"], "candidate universe"),
        ("Admission", "θA  only", COLORS["blue"], "category eligibility"),
        ("Ranking", "θR  only", COLORS["orange"], "within-set ordering"),
        ("Rejection", "θC  only", COLORS["green"], "sample acceptance"),
    ]
    y = 0.80
    for label, owner_text, color, _ in rows:
        box(owner, (0.02, y - 0.055), 0.42, 0.12, label, face="#FFFFFF", edge=color, linewidth=1.2)
        box(owner, (0.51, y - 0.055), 0.46, 0.12, owner_text, face=color + "1F", edge=color, linewidth=1.2)
        y -= 0.18
    owner.plot([0.04, 0.96], [0.17, 0.17], color=COLORS["light_gray"], lw=0.8, transform=owner.transAxes)
    owner.text(0.50, 0.095, "cross-duty gradient = 0", ha="center", va="center", transform=owner.transAxes, weight="bold")
    owner.text(0.50, 0.015, "structural isolation, not a small-gradient heuristic", ha="center", va="bottom", transform=owner.transAxes, color=COLORS["gray"])

    source_rows = [
        {
            "panel": "a",
            "component": label,
            "owner": owner_text.replace("$", ""),
            "deployment_role": role,
            "gradient_contract": "exclusive" if label != "Geometry" else "frozen",
        }
        for label, owner_text, _, role in rows
    ]
    source_rows.append(
        {
            "panel": "a",
            "component": "Auxiliary residual",
            "owner": "Admission training",
            "deployment_role": "none",
            "gradient_contract": "supervision-only; runtime output ignored",
        }
    )
    write_csv(
        "fig2_method_ownership.csv",
        ["panel", "component", "owner", "deployment_role", "gradient_contract"],
        source_rows,
    )
    save_vector_pair(fig, "fig2_method_ownership")
    plt.close(fig)


if __name__ == "__main__":
    main()
