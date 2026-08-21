#!/usr/bin/env python3
"""Render the deterministic three-decision ARROW teaser.

The figure is a zero-update visualization over sealed records. It uses three
different tests: rank rescue, category-cue intervention, and negative-text
abstention. External cue crops are outside the evaluated image so they cannot
be mistaken for detections or ground truth.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image, ImageOps

from figure_common import load_registry, normalize_svg, value


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FINE_MANIFEST = Path("/media/haoyi/T9/data/FineCops-Ref/v1/manifests/finecops_test_all.jsonl")
FINE_RECORDS = ROOT / "outputs/arrow_finecops_20260819/evaluations/A/seed42/records.jsonl"
PANEL_MANIFEST = ROOT / "data/ablations/stageb_table_a_category_intervention_20260717/category_intervention_pairs.jsonl"
PANEL_VISUAL = ROOT / "outputs/arrow_admission_input_20260818/evaluations/AR_A_PATCH/fresh_panel/seed42.records.jsonl"
PANEL_TEXT = ROOT / "outputs/arrow_admission_input_20260818/evaluations/AR_B_TEXT/fresh_panel/seed42.records.jsonl"
RANK_SAMPLE = "finecops:test:5758"
ABSTAIN_SAMPLE = "finecops:test:30097"
CUE_PAIR = "cat-int:9bc2eb6e5d9a36a81c511e26"
SEALED_TAU = value(load_registry(), "cross_benchmark.sealed_source_tau.seed42")
FIGURE_TIMESTAMP = datetime(2026, 8, 22, tzinfo=timezone.utc)

COLORS = {
    "frozen": "#D55E00",
    "arrow": "#0072B2",
    "gt": "#009E73",
    "dog": "#0072B2",
    "jacket": "#E69F00",
    "ink": "#171717",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows = jsonl(path)
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{path}: duplicate {key}")
    return result


def draw_box(
    ax,
    xyxy: list[float],
    color: str,
    label: str,
    *,
    linestyle: str = "-",
    linewidth: float = 2.1,
    corner: str = "top",
    zorder: int = 4,
) -> None:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor=color, linewidth=linewidth,
        linestyle=linestyle, zorder=zorder,
    ))
    if not label:
        return
    y_label, va = (y2, "bottom") if corner == "bottom" else (y1, "top")
    ax.text(
        max(10.0, x1 + 2), y_label, label, ha="left", va=va, fontsize=6.7,
        color="white", weight="bold", zorder=zorder + 1,
        bbox={"facecolor": color, "edgecolor": "white", "linewidth": 0.35,
              "alpha": 0.96, "pad": 1.2},
    )


def image_axis(fig, rect, path: str | Path):
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#C7C7C7")
    return ax


def title(fig, x: float, text: str, subtitle: str) -> None:
    fig.text(x, 0.955, text, ha="center", va="top", fontsize=8.3,
             weight="bold", color=COLORS["ink"])
    fig.text(x, 0.905, subtitle, ha="center", va="top", fontsize=6.6,
             color="#555555")


def cue_card(fig, rect, path: str, label: str, color: str):
    ax = image_axis(fig, rect, path)
    cue = Image.open(path).convert("RGB")
    side = min(cue.size)
    ax.images[0].set_data(ImageOps.fit(
        cue, (side, side), method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    ))
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_edgecolor(color)
    ax.text(0.5, -0.11, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=6.8, weight="bold", color=color)
    return ax


def render(
    rank: dict[str, Any], rank_manifest: dict[str, Any],
    abstain: dict[str, Any], abstain_manifest: dict[str, Any],
    cue_rows: dict[str, dict[str, Any]],
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "arrow-paper-fig1-v2",
    })
    fig = plt.figure(figsize=(6.875, 2.62), facecolor="white")

    # A: the correct candidate already exists; Ranking changes the winner.
    left = image_axis(fig, [0.015, 0.105, 0.275, 0.74], rank_manifest["filename"])
    title(fig, 0.152, "A  RANK THE RIGHT INSTANCE", "candidate exists; native top-1 is wrong")
    draw_box(left, rank["routes"]["b58"]["top1_box_xyxy"], COLORS["frozen"], "native")
    draw_box(left, rank["routes"]["deployed"]["top1_box_xyxy"], COLORS["arrow"], "")
    x, y, w, h = rank_manifest["finecops_bbox_xywh"]
    draw_box(left, [x, y, x + w, y + h], COLORS["gt"], "target", linestyle="--",
             linewidth=2.4, corner="bottom", zorder=6)
    left.text(
        0.5, 0.025, "ARROW: FULL EXPRESSION → TARGET",
        transform=left.transAxes, ha="center", va="bottom", fontsize=6.2,
        weight="bold", color="white",
        bbox={"facecolor": COLORS["arrow"], "edgecolor": "white", "linewidth": 0.4, "alpha": 0.94, "pad": 2},
        zorder=9,
    )
    left.text(
        0.5, 0.985,
        "“wooden structure right of the car …”",
        transform=left.transAxes, ha="center", va="top", fontsize=6.3,
        color="white", zorder=9,
        bbox={"facecolor": COLORS["ink"], "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
    )

    # B: scene and frozen query universe stay fixed. Large external cue cards
    # visibly control which category is admitted.
    cue_a = cue_rows["A"]["category_intervention"]
    cue_b = cue_rows["B"]["category_intervention"]
    scene = image_axis(fig, [0.330, 0.300, 0.340, 0.545], cue_a["image_path"])
    title(fig, 0.500, "B  ADMIT THE REQUESTED CATEGORY", "same scene + frozen queries; only the cue changes")
    for index, box_xyxy in enumerate(cue_a["class_a"]["boxes_xyxy"]):
        draw_box(scene, box_xyxy, COLORS["dog"], "dog" if index == 0 else "",
                 linewidth=2.0, zorder=4)
    for index, box_xyxy in enumerate(cue_a["class_b"]["boxes_xyxy"]):
        draw_box(scene, box_xyxy, COLORS["jacket"], "jacket" if index == 0 else "",
                 linewidth=2.0, zorder=4)
    dog_ax = cue_card(fig, [0.355, 0.055, 0.100, 0.165], cue_a["class_a"]["support_path"],
                      "DOG CUE", COLORS["dog"])
    jacket_ax = cue_card(fig, [0.545, 0.055, 0.100, 0.165], cue_b["class_b"]["support_path"],
                         "JACKET CUE", COLORS["jacket"])
    dog_center = [sum(cue_a["class_a"]["boxes_xyxy"][0][::2]) / 2,
                  sum(cue_a["class_a"]["boxes_xyxy"][0][1::2]) / 2]
    jacket_center = [sum(cue_a["class_b"]["boxes_xyxy"][0][::2]) / 2,
                     sum(cue_a["class_b"]["boxes_xyxy"][0][1::2]) / 2]
    for source, target, color, bend in (
        (dog_ax, dog_center, COLORS["dog"], -0.12),
        (jacket_ax, jacket_center, COLORS["jacket"], 0.12),
    ):
        fig.add_artist(ConnectionPatch(
            xyA=(0.5, 1.02), coordsA=source.transAxes,
            xyB=target, coordsB=scene.transData,
            arrowstyle="-|>", mutation_scale=10, linewidth=1.8,
            color=color, connectionstyle=f"arc3,rad={bend}", zorder=10,
        ))
    fig.text(0.500, 0.255, "EXTERNAL CUES — DIFFERENT IMAGES",
             ha="center", va="center", fontsize=6.3, weight="bold", color="#555555")

    # C: Ranking always has a winner; Abstention suppresses it.
    right = image_axis(fig, [0.710, 0.105, 0.275, 0.74], abstain_manifest["filename"])
    title(fig, 0.848, "C  ABSTAIN WHEN ABSENT", "a ranking winner still exists")
    draw_box(right, abstain["routes"]["b58"]["top1_box_xyxy"], COLORS["frozen"], "rank winner")
    right.text(
        0.5, 0.19, "✓  ABSTAIN", transform=right.transAxes,
        ha="center", va="center", fontsize=9.6, weight="bold", color="white",
        bbox={"facecolor": COLORS["gt"], "edgecolor": "white", "linewidth": 0.8,
              "alpha": 0.96, "pad": 4.0}, zorder=10,
    )
    right.text(
        0.5, 0.055,
        rf"confidence {abstain['routes']['deployed']['raw_confidence']:.2f} < sealed $\tau$ {SEALED_TAU:.2f}",
        transform=right.transAxes, ha="center", va="bottom", fontsize=6.3,
        weight="bold", color="white",
        bbox={"facecolor": COLORS["ink"], "edgecolor": "none", "alpha": 0.88, "pad": 2},
        zorder=9,
    )
    right.text(
        0.5, 0.985, f'negative: “{abstain_manifest["finecops_expression"]}”',
        transform=right.transAxes, ha="center", va="top", fontsize=6.3,
        color="white", zorder=9,
        bbox={"facecolor": COLORS["ink"], "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
    )

    out = PAPER / "figures" / "fig1_teaser"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), metadata={
        "Creator": "paper/scripts/select_qualitative_examples.py",
        "Subject": "ARROW three-decision teaser",
        "CreationDate": FIGURE_TIMESTAMP, "ModDate": FIGURE_TIMESTAMP,
    })
    fig.savefig(out.with_suffix(".svg"), metadata={
        "Creator": "paper/scripts/select_qualitative_examples.py",
        "Date": FIGURE_TIMESTAMP.date().isoformat(),
    })
    normalize_svg(out.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    for path in (FINE_MANIFEST, FINE_RECORDS, PANEL_MANIFEST, PANEL_VISUAL, PANEL_TEXT):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifests = keyed(FINE_MANIFEST, "sample_id")
    records = keyed(FINE_RECORDS, "sample_id")
    rank, abstain = records[RANK_SAMPLE], records[ABSTAIN_SAMPLE]
    if not (rank["routes"]["b58"]["top1_iou"] < 0.5 <= rank["routes"]["deployed"]["top1_iou"]):
        raise RuntimeError("rank teaser predicate drifted")
    if abstain["kind"] == "positive" or not abstain["routes"]["deployed"]["raw_confidence"] < SEALED_TAU:
        raise RuntimeError("abstention teaser predicate drifted")

    panel_rows = [row for row in jsonl(PANEL_MANIFEST)
                  if row["category_intervention"]["pair_id"] == CUE_PAIR]
    cue_rows = {row["category_intervention"]["arm"]: row for row in panel_rows}
    if set(cue_rows) != {"A", "B"}:
        raise RuntimeError("category-cue teaser pair is incomplete")
    cue_visual_all = keyed(PANEL_VISUAL, "sample_id")
    cue_text_all = keyed(PANEL_TEXT, "sample_id")
    cue_visual = {arm: cue_visual_all[f"{CUE_PAIR}:{arm}"] for arm in ("A", "B")}
    cue_text = {arm: cue_text_all[f"{CUE_PAIR}:{arm}"] for arm in ("A", "B")}
    if not all(row["active_score_wins"] for row in cue_visual.values()):
        raise RuntimeError("visual-cue bidirectional switch predicate drifted")
    if all(row["active_score_wins"] for row in cue_text.values()):
        raise RuntimeError("chosen cue pair no longer distinguishes visual from text Admission")

    render(rank, manifests[RANK_SAMPLE], abstain, manifests[ABSTAIN_SAMPLE], cue_rows)

    selection = [
        {
            "panel": "Ranking", "sample_id": RANK_SAMPLE,
            "rule": "sealed native-wrong complete-expression-rank-correct example",
            "image_path": manifests[RANK_SAMPLE]["filename"],
            "image_sha256": manifests[RANK_SAMPLE]["finecops_image_artifact"]["sha256"],
        },
        {
            "panel": "Admission", "sample_id": CUE_PAIR,
            "rule": "visual bidirectional success, category-text failure, legible external cues",
            "image_path": cue_rows["A"]["category_intervention"]["image_path"],
            "image_sha256": cue_rows["A"]["category_intervention"]["image_sha256"],
            "dog_support_sha256": cue_rows["A"]["category_intervention"]["class_a"]["support_sha256"],
            "jacket_support_sha256": cue_rows["A"]["category_intervention"]["class_b"]["support_sha256"],
        },
        {
            "panel": "Abstention", "sample_id": ABSTAIN_SAMPLE,
            "rule": "sealed source threshold suppresses a negative-expression winner",
            "image_path": manifests[ABSTAIN_SAMPLE]["filename"],
            "image_sha256": manifests[ABSTAIN_SAMPLE]["finecops_image_artifact"]["sha256"],
        },
    ]
    receipt = {
        "schema": "arrow.paper.qualitative_selection/v2",
        "status": "post_hoc_zero_training_visualization",
        "source_files": {
            str(path): sha256(path)
            for path in (FINE_MANIFEST, FINE_RECORDS, PANEL_MANIFEST, PANEL_VISUAL, PANEL_TEXT)
        },
        "checkpoint_seed": 42,
        "sealed_source_threshold": SEALED_TAU,
        "selection": selection,
        "claim_boundary": "illustrative decision diagnosis; no metric or model selection",
    }
    data_dir = PAPER / "data"
    (data_dir / "qualitative_selection.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_dir = data_dir / "plot_sources"
    plot_dir.mkdir(parents=True, exist_ok=True)
    with (plot_dir / "fig1_qualitative.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in selection for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(selection)


if __name__ == "__main__":
    main()
