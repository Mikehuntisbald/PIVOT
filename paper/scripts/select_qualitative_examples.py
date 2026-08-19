#!/usr/bin/env python3
"""Select and render deterministic FineCops qualitative examples.

This is a zero-training, post-hoc visualization utility.  The selection rules
are fixed in code and the resulting receipt binds every source byte.  The
script never changes checkpoints, routing, margins, or thresholds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
MANIFEST = Path("/media/haoyi/T9/data/FineCops-Ref/v1/manifests/finecops_test_all.jsonl")
RECORDS = ROOT / "outputs/arrow_finecops_20260819/evaluations/A/seed42/records.jsonl"
SEALED_TAU = 0.31912317872047424


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


def box(ax, xyxy: list[float], color: str, label: str, linewidth: float = 2.0) -> None:
    x1, y1, x2, y2 = xyxy
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                           edgecolor=color, linewidth=linewidth))
    ax.text(x1, max(1, y1), label, color="white", fontsize=7, va="top", ha="left",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.4, "alpha": 0.9})


def render(selection: list[tuple[str, dict[str, Any], str]], manifests: dict[str, dict[str, Any]]) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    colors = {"base": "#D55E00", "arrow": "#0072B2", "gt": "#009E73"}
    figure, axes = plt.subplots(1, 4, figsize=(7.05, 2.65), constrained_layout=True)
    labels = "abcd"
    for index, ((title, row, _), ax) in enumerate(zip(selection, axes)):
        manifest = manifests[row["sample_id"]]
        image = Image.open(manifest["filename"]).convert("RGB")
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f"({labels[index]}) {title}", fontsize=8.3, fontweight="bold", pad=3)

        if row["kind"] == "positive":
            x, y, w, h = manifest["finecops_bbox_xywh"]
            box(ax, [x, y, x + w, y + h], colors["gt"], "GT", 1.8)
        box(ax, row["routes"]["b58"]["top1_box_xyxy"], colors["base"], "Base", 1.5)

        if title == "Rejection transfer":
            ax.text(
                0.02, 0.03,
                rf"ARROW abstains: $c={row['routes']['deployed']['raw_confidence']:.2f}$; source $\tau={SEALED_TAU:.2f}$",
                transform=ax.transAxes, color="white", fontsize=7, ha="left", va="bottom",
                bbox={"facecolor": colors["arrow"], "edgecolor": "none", "alpha": 0.92, "pad": 2.0},
            )
        else:
            box(ax, row["routes"]["deployed"]["top1_box_xyxy"], colors["arrow"], "ARROW", 1.5)
            support = manifest.get("finecops_support")
            if support and support.get("path"):
                inset = ax.inset_axes([0.72, 0.02, 0.26, 0.26])
                inset.imshow(Image.open(support["path"]).convert("RGB"))
                inset.set_xticks([]); inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_edgecolor("white"); spine.set_linewidth(1.3)
                inset.set_title("support", fontsize=6.4, color="white", pad=1,
                                backgroundcolor="#333333")

        expression = "\n".join(textwrap.wrap(manifest["finecops_expression"], width=34, max_lines=3,
                                              placeholder="…"))
        ax.text(0.5, -0.035, expression, transform=ax.transAxes, ha="center", va="top", fontsize=6.6)

    out = PAPER / "figures" / "fig1_teaser"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(out.with_suffix(".svg"), bbox_inches="tight")
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
