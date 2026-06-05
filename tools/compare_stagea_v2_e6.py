#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ckpt_block(summary, *, ckpt_suffix: str):
    matches = [r["checkpoint"] for r in summary["ranking"] if r["checkpoint"].endswith(ckpt_suffix)]
    if not matches:
        raise SystemExit(f"missing checkpoint suffix {ckpt_suffix}")
    ckpt = matches[0]
    rows = [r for r in summary["results"] if r["checkpoint"] == ckpt]
    by_ds = {r["dataset"]: r for r in rows}

    def mean(metric: str):
        vals = [float(r.get(metric, 0.0)) for r in rows if r.get(metric) is not None]
        return sum(vals) / max(1, len(vals))

    return {
        "checkpoint": ckpt,
        "mean_patch_ap50": mean("patch_ap50"),
        "mean_box_recall@50": mean("box_recall@50"),
        "mean_matched_query_recall@50": (
            mean("matched_query_recall@50") if any("matched_query_recall@50" in r for r in rows) else None
        ),
        "lvis_patch_ap50": by_ds.get("lvis_val", {}).get("patch_ap50"),
        "coco_patch_ap50": by_ds.get("coco_val", {}).get("patch_ap50"),
        "lvis_box_recall@50": by_ds.get("lvis_val", {}).get("box_recall@50"),
        "coco_box_recall@50": by_ds.get("coco_val", {}).get("box_recall@50"),
    }


def deltas(a, b):
    out = {}
    for k, v in a.items():
        if k == "checkpoint" or v is None or b.get(k) is None:
            continue
        out[k] = float(v) - float(b[k])
    return out


def fmt(value):
    return "n/a" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deprecated Stage-A v2 output against Stage-A baselines.")
    parser.add_argument(
        "--v2_summary",
        default="outputs/stageA_coco_multipatch_v2_rank_posneg_eval_0006_fast/summary.json",
        help="Summary JSON for the deprecated Stage-A v2 eval.",
    )
    parser.add_argument(
        "--previous_summary",
        default="outputs/stageA_coco_multipatch_eval_0005_0006_fast/summary.json",
        help="Summary JSON for the previous Stage-A mainline eval.",
    )
    parser.add_argument(
        "--ogc_summary",
        default="outputs/ogc_original_finetune_stage_a_eval_stagea_caliber_e1_e2/summary.json",
        help="Summary JSON for the GroundingDINO same-data finetune baseline.",
    )
    parser.add_argument("--v2_ckpt_suffix", default="checkpoint0006.pth")
    parser.add_argument("--previous_ckpt_suffix", default="checkpoint0006.pth")
    parser.add_argument("--ogc_ckpt_suffix", default="checkpoint0001.pth")
    parser.add_argument(
        "--output_dir",
        default="outputs/stageA_coco_multipatch_v2_rank_posneg_eval_0006_fast",
        help="Directory for comparison_v2_e6_vs_baselines.{json,md}.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v2 = ckpt_block(load_summary(args.v2_summary), ckpt_suffix=args.v2_ckpt_suffix)
    old = ckpt_block(
        load_summary(args.previous_summary),
        ckpt_suffix=args.previous_ckpt_suffix,
    )
    ogc = ckpt_block(
        load_summary(args.ogc_summary),
        ckpt_suffix=args.ogc_ckpt_suffix,
    )
    payload = {
        "deprecated_v2_rank_posneg_e6": v2,
        "previous_stageA_e6": old,
        "ogc_original_finetune_e1": ogc,
        "delta_deprecated_v2_minus_previous_stageA_e6": deltas(v2, old),
        "delta_deprecated_v2_minus_ogc_original_finetune_e1": deltas(v2, ogc),
    }
    (out_dir / "comparison_v2_e6_vs_baselines.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = [
        ("deprecated v2 rank+posneg e6", v2),
        ("previous Stage A e6", old),
        ("OGC original FT e1", ogc),
    ]
    lines = [
        "# Deprecated Stage A v2 e6 comparison",
        "",
        "| run | mean patch_ap50 | LVIS patch_ap50 | COCO patch_ap50 | mean box_recall@50 | mean matched_query_recall@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows:
        lines.append(
            f"| {name} | {fmt(r['mean_patch_ap50'])} | {fmt(r['lvis_patch_ap50'])} | "
            f"{fmt(r['coco_patch_ap50'])} | {fmt(r['mean_box_recall@50'])} | "
            f"{fmt(r['mean_matched_query_recall@50'])} |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| delta | mean patch_ap50 | LVIS patch_ap50 | COCO patch_ap50 | mean box_recall@50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, d in [
        ("deprecated v2 - previous Stage A e6", payload["delta_deprecated_v2_minus_previous_stageA_e6"]),
        ("deprecated v2 - OGC original FT e1", payload["delta_deprecated_v2_minus_ogc_original_finetune_e1"]),
    ]:
        lines.append(
            f"| {name} | {d.get('mean_patch_ap50', 0.0):+.6f} | "
            f"{d.get('lvis_patch_ap50', 0.0):+.6f} | {d.get('coco_patch_ap50', 0.0):+.6f} | "
            f"{d.get('mean_box_recall@50', 0.0):+.6f} |"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "comparison_v2_e6_vs_baselines.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
