#!/usr/bin/env python3
"""Render v7 from sealed v6 results; no training, forward, or new bootstrap.

The L1 statement is an analytic consequence of existing pairwise point
estimates. It has no newly estimated risk interval or simultaneous guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_readout_v6_assets as old

PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / "data/readout_v6"
MM, MD = old.LOCALIZERS
GE, GY, SE, SY = old.CELLS
TARGET = "global_emit_minus_exists"
CROSS = "fixed_weights__global_max__emit__eval_selected_minus_matched"


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": old.sha(path)}


def analytic_l1(a, delta_cw, delta_cn):
    if not 0 < a < 1:
        raise ValueError("nondegenerate fixed correctness fraction required")
    # Delta R(pi) = (1-pi) [(1-pi)*cw + pi*cn].
    cw, cn = -a * (1-a) * delta_cw, -a * delta_cn
    if cw < 0 and cn < 0:
        status = "strictly_lower_mean_risk_for_0_le_pi_lt_1"
    elif cw > 0 and cn > 0:
        status = "strictly_higher_mean_risk_for_0_le_pi_lt_1"
    else:
        status = "no_uniform_strict_direction_from_signs"
    root = None if cw == cn else cw / (cw-cn)
    return {
        "role": "analytic_point_estimate_consequence",
        "formula": "(1-pi)*((1-pi)*cw_coefficient+pi*cn_coefficient)",
        "a": a, "delta_u_cw": delta_cw, "delta_u_cn": delta_cn,
        "cw_coefficient": cw, "cn_coefficient": cn,
        "status": status, "interior_root": root if root is not None and 0 < root < 1 else None,
        "at_pi_one": 0.0, "risk_ci95": None, "new_bootstrap_replicates": 0,
        "simultaneous_curve_guarantee": False,
        "conditioning": "fixed L1-positive and existing no-target class-conditional populations",
    }


def load_sources():
    receipt = json.loads((SOURCE / "experimental_completion.json").read_text())
    assert receipt["status"] == "complete" and receipt["new_heads"] == 18
    data, bindings = {}, {}
    for name, source in receipt["analyses"].items():
        path = SOURCE / (name + ".json")
        assert old.sha(path) == source["sha256"], "sealed analysis drift"
        key = "gref_source_disjoint" if name.endswith("disjoint") else name
        data[key] = json.loads(path.read_text())
        old.validate_analysis(data[key], key)
        bindings[key] = bind(path)
    assets = json.loads((PAPER / "generated/readout_v6/receipt.json").read_text())
    assert old.sha(old.__file__) == assets["generator_sha256"], "v6 renderer drift"
    for name, digest in assets["outputs"].items():
        assert old.sha(PAPER / "generated/readout_v6" / name) == digest
    return data, bindings


def facts(data):
    fc = data["finecops_val"]["localizers"]
    r, e = fc[MM], fc[MM]["effects"][TARGET]
    count = r["conditional_counts"]["difficulty_cw_level1"]
    c, w = count["high"], count["low"]
    n = r["conditional_counts"]["difficulty_cn_level1"]["low"]
    if (c, w, n) != (5506, 591, 9029):
        raise ValueError("L1 conditional population drift")
    per_seed = {}
    for seed in old.SEEDS:
        p = r["per_seed"][seed]
        per_seed[seed] = {key: p[GY][key] - p[GE][key]
                          for key in ("difficulty_cw_level1", "difficulty_cn_level1")}
    for key in per_seed[old.SEEDS[0]]:
        assert abs(sum(v[key] for v in per_seed.values())/3 - e[key]["mean"]) < 1e-14
    derivation = analytic_l1(c/(c+w), e["difficulty_cw_level1"]["mean"],
                            e["difficulty_cn_level1"]["mean"])
    assert abs(e["difficulty_cn_within_level_contribution"]["mean"]
               + e["difficulty_cn_cross_level_contribution"]["mean"]
               - e["correct_vs_no_target_auroc"]["mean"]) < 1e-14
    return {"schema": "arrow.paper.evidence_v7_facts/v1",
            "l1_population": {"C": c, "W": w, "N": n, "positive": c+w},
            "l1_pairwise_effects": {key: e[key] for key in per_seed[old.SEEDS[0]]},
            "l1_per_seed_pairwise_point_effects": per_seed,
            "analytic_l1": derivation, "new_model_forwards": 0, "new_training_updates": 0,
            "new_risk_estimation": False, "new_bootstrap_replicates": 0}


def snippets(data):
    fc = data["finecops_val"]["localizers"]
    mm, md = fc[MM], fc[MD]
    e = mm["effects"][TARGET]
    estimates = {
        "FCMMRisk": mm["effects"][TARGET]["mixed_augrc"],
        "FCMDRisk": md["effects"][TARGET]["mixed_augrc"],
        "MMSelectedTargetRisk": mm["effects"]["selected_emit_minus_exists"]["mixed_augrc"],
        "MMReadoutChange": mm["effects"]["D_emit"]["mixed_augrc"],
        "MDReadoutChange": md["effects"]["D_emit"]["mixed_augrc"],
        "MDInferenceChange": md["effects"][CROSS]["mixed_augrc"],
        "MMInteraction": mm["effects"]["interaction"]["mixed_augrc"],
        "MDInteraction": md["effects"]["interaction"]["mixed_augrc"],
        "LoneCW": e["difficulty_cw_level1"], "LoneCN": e["difficulty_cn_level1"],
        "AllCN": e["correct_vs_no_target_auroc"],
        "WithinCN": e["difficulty_cn_within_level_contribution"],
        "AcrossCN": e["difficulty_cn_cross_level_contribution"],
        "SameImageCN": e["same_image_cn"],
        "ComparableCN": e["same_image_cn_comparable_unconditional"],
        "MDTransferReadout": data["gref_source_disjoint"]["localizers"][MD]["effects"]["D_emit"]["mixed_augrc"],
        "MMTransferProduct": data["gref_source_disjoint"]["localizers"][MM]["effects"]["joint_product_minus_global_max__emit"]["mixed_augrc"],
    }
    macros = "\n".join("\\newcommand{\\" + name + "}{" + old.estimate(value, True) + "}"
                       for name, value in estimates.items()) + "\n"
    coverage = old.table(
        "The MM-GDINO FineCops global target effect changes across difficulty comparisons. "
        "All numbers are emission-minus-existence AUROC points with existing paired image intervals. "
        "L1 uses 6,097 positive requests and the same 9,029 no-target requests. "
        "The final two rows are weighted contributions to the full C/N effect, not conditional AUROCs.",
        "tab:coverage", "@{}lcc@{}", r"Comparison & $\Delta U_{CW}$ & $\Delta U_{CN}$",
        ["Full positive population & " + old.estimate(e["correctness_auroc"], True, True) + " & " + old.estimate(e["correct_vs_no_target_auroc"], True, True) + r"\\",
         "L1 positive population & " + old.estimate(e["difficulty_cw_level1"], True, True) + " & " + old.estimate(e["difficulty_cn_level1"], True, True) + r"\\",
         r"\midrule",
         "Within-level contribution & -- & " + old.estimate(e["difficulty_cn_within_level_contribution"], True, True) + r"\\",
         "Cross-level contribution & -- & " + old.estimate(e["difficulty_cn_cross_level_contribution"], True, True) + r"\\"])
    cross = old.table(
        "Two ways to read the Native-selected query on MDETR FineCops. Both changes reference "
        "the G-trained/G-deployed emission head. Matched S is trained from the same original initialization, "
        "not continued from G. Risk is mixed AUGRC times 100; endpoint cells show seed mean (sample SD), "
        "differences show paired image intervals for the fixed three-head mean.",
        "tab:cross_readout", "@{}llcc@{}", r"Training & Inference & Risk (SD) & Change from G/G",
        ["Global & Global & " + old.point_sd(md["summary"][GY]["mixed_augrc"]) + r" & --\\",
         "Global & Selected & " + old.point_sd(md["summary"]["global_max__emit__eval_selected"]["mixed_augrc"]) + " & " + old.estimate(md["effects"][CROSS]["mixed_augrc"], True, True) + r"\\",
         "Selected & Selected & " + old.point_sd(md["summary"][SY]["mixed_augrc"]) + " & " + old.estimate(md["effects"]["D_emit"]["mixed_augrc"], True, True) + r"\\"])
    return {"numbers.tex": macros, "table_coverage.tex": coverage, "table_cross_readout.tex": cross}


def figure(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "evidence-v7"})
    fc = data["finecops_val"]["localizers"]
    blue, orange = "#236b91", "#c8742f"
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 3.6))
    fig.subplots_adjust(left=.13, right=.985, bottom=.25, top=.72, wspace=.80)
    fig.text(.02, .97, "Supervising the right event is not enough", fontsize=16, weight="bold", va="top")
    panels = [
        ("(a) Same labels, opposite risk", ["MM-GDINO", "MDETR"],
         [fc[loc]["effects"][TARGET]["mixed_augrc"] for loc in (MM, MD)], "Emission − existence\nAUGRC change ×100"),
        ("(b) Re-read versus retrain", ["Inference\nG → S", "Train +\ndeploy S"],
         [fc[MD]["effects"][CROSS]["mixed_augrc"], fc[MD]["effects"]["D_emit"]["mixed_augrc"]], "MDETR; relative to G/G\nAUGRC change ×100"),
        ("(c) Locate the C/N loss", ["All positive\nlevels", "L1 positives"],
         [fc[MM]["effects"][TARGET]["correct_vs_no_target_auroc"], fc[MM]["effects"][TARGET]["difficulty_cn_level1"]], "MM-GDINO C/N AUROC\nChange ×100"),
    ]
    for ax, (title, labels, rows, xlabel) in zip(axes, panels):
        ax.axvline(0, color="#999999", ls="--", lw=.9)
        for y, row, color in zip([1, 0], rows, [blue, orange]):
            lo, hi = [100*x for x in row["ci95"]]; mean = 100*row["mean"]
            ax.plot([lo, hi], [y, y], color=color, lw=2)
            ax.plot(mean, y, "o", color=color, ms=5)
            ax.annotate(f"{mean:+.3f}", (mean, y), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=9, color=color)
        ax.set_yticks([1, 0], labels); ax.set_ylim(-.5, 1.55)
        ax.set_title(title, fontsize=10, pad=15, weight="bold")
        ax.set_xlabel(xlabel, fontsize=8.4); ax.tick_params(axis="both", labelsize=8.5)
        ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.margins(x=.22)
    fig.text(.02, .08, "FineCops val • Fixed output boxes • Three-head means and paired image-cluster 95% intervals", fontsize=8.6)
    fig.text(.02, .025, "L1 positives alone supply head supervision; validation also contains L2/L3 positives.", fontsize=8.6, color="#4e555b")
    fig.savefig(out/"figure1_evidence.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out/"figure1_evidence.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER/"generated/evidence_v7_r2")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data, sources = load_sources()
    sources["legacy_renderer"] = bind(old.__file__)
    sources["legacy_assets"] = bind(PAPER/"generated/readout_v6/receipt.json")
    out = args.output_dir.resolve()
    if args.check:
        receipt = json.loads((out/"receipt.json").read_text())
        assert receipt["sources"] == sources and receipt["generator"] == bind(__file__)
        for name, digest in receipt["outputs"].items(): assert old.sha(out/name) == digest
        assert json.loads((out/"analytic_l1.json").read_text()) == facts(data)
        print("v7 source, analytic-point scope, and asset hashes verified")
        return
    if out.exists(): raise FileExistsError("v7 artifact directory exists; check it or select a new version")
    values, texts = facts(data), snippets(data)
    out.mkdir(parents=True)
    for name, text in texts.items(): (out/name).write_text(text)
    (out/"analytic_l1.json").write_text(json.dumps(values, indent=2, sort_keys=True)+"\n")
    figure(data, out)
    outputs = {p.name: old.sha(p) for p in sorted(out.iterdir())}
    receipt = {"schema": "arrow.paper.evidence_v7_assets/v1", "sources": sources,
               "generator": bind(__file__), "outputs": outputs,
               "new_training": False, "new_evaluation": False, "new_bootstrap": False}
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(out), "new_risk_ci": False, "assets": len(outputs)}))


if __name__ == "__main__":
    main()
