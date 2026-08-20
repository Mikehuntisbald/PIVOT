#!/usr/bin/env python3
"""Select and render deterministic FineCops qualitative examples.

This is a zero-training, post-hoc visualization utility.  The selection rules
are fixed in code and the resulting receipt binds every source byte.  The
script never changes checkpoints, routing, margins, or thresholds.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageOps

from figure_common import load_registry, value


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
MANIFEST = Path("/media/haoyi/T9/data/FineCops-Ref/v1/manifests/finecops_test_all.jsonl")
RECORDS = ROOT / "outputs/arrow_finecops_20260819/evaluations/A/seed42/records.jsonl"
SEALED_TAU = value(load_registry(), "cross_benchmark.sealed_source_tau.seed42")
FIGURE_TIMESTAMP = datetime(2026, 8, 19, tzinfo=timezone.utc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_inputs() -> None:
    for path in (MANIFEST, RECORDS):
        if not path.is_file():
            raise FileNotFoundError(path)


def rank_percentiles(rows: list[dict[str, Any]], key, *, reverse: bool) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (key(row), row["sample_id"]), reverse=reverse)
    denominator = max(1, len(ordered) - 1)
    return {row["sample_id"]: index / denominator for index, row in enumerate(ordered)}


def select(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    positives = [row for row in records if row["kind"] == "positive" and row["support_covered"]]

    # Ranking gain: the base fails, the complete-expression ranker succeeds,
    # and Admission preserves that successful query.  Maximize the IoU gain.
    rank_candidates = [
        row for row in positives
        if row["routes"]["b58"]["top1_iou"] < 0.5
        <= row["routes"]["r100_d3"]["top1_iou"]
        and row["routes"]["deployed"]["top1_query_index"]
        == row["routes"]["r100_d3"]["top1_query_index"]
    ]
    ranking = min(
        rank_candidates,
        key=lambda row: (
            -(row["routes"]["r100_d3"]["top1_iou"] - row["routes"]["b58"]["top1_iou"]),
            row["sample_id"],
        ),
    )

    # Admission gain: unrestricted Ranking fails but gating succeeds.  Prefer
    # the smallest admitted set, then the largest IoU recovery.
    admission_candidates = [
        row for row in positives
        if row["sample_id"] != ranking["sample_id"]
        and row["routes"]["r100_d3"]["top1_iou"] < 0.5
        <= row["routes"]["deployed"]["top1_iou"]
    ]
    admission = min(
        admission_candidates,
        key=lambda row: (
            row["eligible_queries"],
            -(row["routes"]["deployed"]["top1_iou"] - row["routes"]["r100_d3"]["top1_iou"]),
            row["sample_id"],
        ),
    )

    # Rejection gain: the frozen source threshold rejects the external
    # negative.  B58 has no fitted absolute threshold, so choose a visually
    # representative row using within-head ranks rather than comparing raw
    # score magnitudes across heads.
    negatives = [
        row for row in records
        if row["kind"] != "positive"
        and row["routes"]["deployed"]["raw_confidence"] < SEALED_TAU
    ]
    base_rank = rank_percentiles(
        negatives, lambda row: row["routes"]["b58"]["official_probability"], reverse=True
    )
    reject_rank = rank_percentiles(
        negatives, lambda row: row["routes"]["deployed"]["raw_confidence"], reverse=False
    )
    rejection = min(
        negatives,
        key=lambda row: (
            base_rank[row["sample_id"]] + reject_rank[row["sample_id"]],
            row["sample_id"],
        ),
    )

    # Honest failure: B58 localizes correctly but Admission removes the useful
    # winner.  Maximize the IoU deterioration; never hide this failure mode.
    failure_candidates = [
        row for row in positives
        if row["routes"]["deployed"]["top1_iou"] < 0.5
        <= row["routes"]["b58"]["top1_iou"]
    ]
    failure = min(
        failure_candidates,
        key=lambda row: (
            -(row["routes"]["b58"]["top1_iou"] - row["routes"]["deployed"]["top1_iou"]),
            row["sample_id"],
        ),
    )

    return [
        ("Ranking rescue", ranking, "base_wrong_ranker_correct_admission_preserves"),
        ("Admission rescue", admission, "ranker_wrong_admission_correct_min_eligible"),
        ("Rejection transfer", rejection, "sealed_source_tau_rejects_external_negative"),
        ("Honest failure", failure, "base_correct_admission_wrong_max_iou_drop"),
    ]


def box(
    ax,
    xyxy: list[float],
    color: str,
    label: str,
    linewidth: float = 2.0,
    *,
    linestyle: str = "-",
    label_corner: str = "top-left",
    zorder: float = 3,
) -> None:
    x1, y1, x2, y2 = xyxy
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                           edgecolor=color, linewidth=linewidth,
                           linestyle=linestyle, zorder=zorder))
    if label_corner == "bottom-left":
        label_x, label_y, va = x1, min(y2 - 1, ax.images[0].get_array().shape[0] - 1), "bottom"
    elif label_corner == "top-right":
        label_x, label_y, va = x2, max(1, y1), "top"
    else:
        label_x, label_y, va = x1, max(1, y1), "top"
    ax.text(
        label_x,
        label_y,
        label,
        color="white",
        fontsize=6.7,
        weight="bold",
        va=va,
        ha="right" if label_corner == "top-right" else "left",
        zorder=zorder + 0.2,
        bbox={"facecolor": color, "edgecolor": "white", "linewidth": 0.35,
              "pad": 1.25, "alpha": 0.94},
    )


def render(selection: list[tuple[str, dict[str, Any], str]], manifests: dict[str, dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "svg.hashsalt": "arrow-paper-fig1-v1",
        }
    )
    colors = {"base": "#D55E00", "arrow": "#0072B2", "gt": "#009E73"}
    # Match the CVPR figure* text width exactly.  Avoid bbox_inches='tight':
    # it lets long labels enlarge the source canvas and silently shrink every
    # font at LaTeX inclusion time.
    figure, axes = plt.subplots(1, 4, figsize=(6.875, 2.92))
    # The lower band is a separate input/caption area.  Cue crops never sit on
    # the detection image, avoiding the false impression that they are scene
    # proposals or ground-truth crops.
    figure.subplots_adjust(left=0.012, right=0.995, bottom=0.28, top=0.82, wspace=0.11)
    legend_handles = [
        Line2D([0], [0], color=colors["base"], lw=2.0, label="Frozen model prediction"),
        Line2D([0], [0], color=colors["arrow"], lw=2.2, label="ARROW decision"),
        Line2D([0], [0], color=colors["gt"], lw=2.4, ls="--", label="Ground truth"),
        Patch(facecolor="#333333", edgecolor="white", label="Visual category cue"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=6.8,
        handlelength=2.2,
        columnspacing=1.25,
    )
    labels = "abcd"
    display_titles = {
        "Ranking rescue": "RANK: fixes the box",
        "Admission rescue": "ADMIT: drops distractor",
        "Rejection transfer": "REJECT: absent target",
        "Honest failure": "FAILURE: target lost",
    }
    for index, ((title, row, _), ax) in enumerate(zip(selection, axes)):
        manifest = manifests[row["sample_id"]]
        image = Image.open(manifest["filename"]).convert("RGB")
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(
            f"({labels[index]}) {display_titles[title]}",
            fontsize=7.5,
            fontweight="bold",
            pad=3,
            color="#111111",
        )

        # Draw predictions first.  Ground truth is a dashed top layer so a
        # coincident ARROW box cannot hide the reference boundary.
        box(
            ax,
            row["routes"]["b58"]["top1_box_xyxy"],
            colors["base"],
            "Frozen",
            1.6,
            label_corner="top-right",
            zorder=3,
        )
        if title != "Rejection transfer":
            box(
                ax,
                row["routes"]["deployed"]["top1_box_xyxy"],
                colors["arrow"],
                "ARROW",
                1.8,
                label_corner="top-left",
                zorder=4,
            )
        if row["kind"] == "positive":
            x, y, w, h = manifest["finecops_bbox_xywh"]
            box(
                ax,
                [x, y, x + w, y + h],
                colors["gt"],
                "GT",
                2.5,
                linestyle="--",
                label_corner="bottom-left",
                zorder=6,
            )

        if title == "Rejection transfer":
            ax.text(
                0.50, 0.03,
                "ARROW ABSTAINS\n"
                rf"$c={row['routes']['deployed']['raw_confidence']:.2f}<$ sealed $\tau={SEALED_TAU:.2f}$",
                transform=ax.transAxes, color="white", fontsize=6.7, ha="center", va="bottom",
                weight="bold", zorder=7,
                bbox={"facecolor": colors["arrow"], "edgecolor": "white",
                      "linewidth": 0.5, "alpha": 0.96, "pad": 2.0},
            )
        else:
            support = manifest.get("finecops_support")
            if support and support.get("path"):
                inset = ax.inset_axes([0.02, -0.40, 0.22, 0.22], zorder=8)
                cue = Image.open(support["path"]).convert("RGB")
                side = max(cue.size)
                cue_square = Image.new("RGB", (side, side), color=(242, 242, 242))
                cue_fit = ImageOps.contain(cue, (side, side))
                cue_square.paste(
                    cue_fit,
                    ((side - cue_fit.width) // 2, (side - cue_fit.height) // 2),
                )
                inset.imshow(cue_square)
                inset.set_xticks([]); inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_edgecolor("white"); spine.set_linewidth(1.5)
                inset.text(
                    0.5,
                    0.98,
                    "CUE",
                    transform=inset.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.8,
                    fontweight="bold",
                    color="white",
                    bbox={"facecolor": "#333333", "edgecolor": "none", "pad": 1.2,
                          "alpha": 0.94},
                )

        support = manifest.get("finecops_support") if title != "Rejection transfer" else None
        has_cue = bool(support and support.get("path"))
        expression = "\n".join(
            textwrap.wrap(
                manifest["finecops_expression"],
                width=25 if has_cue else 34,
                max_lines=3,
                placeholder="…",
            )
        )
        ax.text(
            0.28 if has_cue else 0.5,
            -0.075,
            expression,
            transform=ax.transAxes,
            ha="left" if has_cue else "center",
            va="top",
            fontsize=6.6,
        )

    out = PAPER / "figures" / "fig1_teaser"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        out.with_suffix(".pdf"),
        metadata={
            "Creator": "paper/scripts/select_qualitative_examples.py",
            "Subject": "ARROW sealed qualitative examples",
            "CreationDate": FIGURE_TIMESTAMP,
            "ModDate": FIGURE_TIMESTAMP,
        },
    )
    figure.savefig(
        out.with_suffix(".svg"),
        metadata={
            "Creator": "paper/scripts/select_qualitative_examples.py",
            "Date": FIGURE_TIMESTAMP.date().isoformat(),
        },
    )
    plt.close(figure)


def main() -> None:
    require_inputs()
    manifests_list = jsonl(MANIFEST)
    records = jsonl(RECORDS)
    manifests = {row["sample_id"]: row for row in manifests_list}
    if len(manifests) != len(manifests_list):
        raise RuntimeError("FineCops manifest sample_id is not unique")
    if [row["sample_id"] for row in records] != [row["sample_id"] for row in manifests_list]:
        raise RuntimeError("record/manifest sample order mismatch")
    selection = select(records)

    rows: list[dict[str, Any]] = []
    receipt_selection: list[dict[str, Any]] = []
    for title, record, rule in selection:
        manifest = manifests[record["sample_id"]]
        compact = {
            "panel": title,
            "rule": rule,
            "sample_id": record["sample_id"],
            "kind": record["kind"],
            "expression": manifest["finecops_expression"],
            "image_path": manifest["filename"],
            "image_sha256": manifest["finecops_image_artifact"]["sha256"],
            "eligible_queries": record["eligible_queries"],
            "frozen_base_iou": record["routes"]["b58"]["top1_iou"],
            "ranker_iou": record["routes"]["r100_d3"]["top1_iou"],
            "arrow_iou": record["routes"]["deployed"]["top1_iou"],
            "arrow_raw_confidence": record["routes"]["deployed"]["raw_confidence"],
        }
        receipt_selection.append(compact)
        rows.append(compact)

    receipt = {
        "schema": "arrow.paper.qualitative_selection/v1",
        "status": "post_hoc_zero_training_visualization",
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST)},
        "source_records": {"path": str(RECORDS), "sha256": sha256(RECORDS)},
        "checkpoint_seed": 42,
        "sealed_source_threshold": SEALED_TAU,
        "selection": receipt_selection,
        "claim_boundary": "examples explain mechanisms and include a failure; they are not a metric",
    }
    data_dir = PAPER / "data"
    (data_dir / "qualitative_selection.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_dir = data_dir / "plot_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "fig1_qualitative.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    render(selection, manifests)


if __name__ == "__main__":
    main()
