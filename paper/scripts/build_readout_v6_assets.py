#!/usr/bin/env python3
"""Build v6 only from complete three-surface, two-localizer paired analyses.

No records, model inference, optimization or metric recomputation. Missing
results fail before any manuscript asset is written. Existing output directories
are immutable; use --check for a source/output SHA audit, not a regeneration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

PAPER = Path(__file__).resolve().parents[1]
SURFACES = ("finecops_val", "gref_source_disjoint", "gref_full")
LOCALIZERS = ("mmgdino_positive", "mdetr_r101_refcoco_ema")
SEEDS = ("17", "42", "73")
CELLS = ("global_max__exists", "global_max__emit", "native_selected__exists", "native_selected__emit")
EFFECTS = ("D_emit", "D_exists", "interaction")
ARMS = ("native", *CELLS, "joint_product", "joint_sirc")
LOC = {LOCALIZERS[0]: "MM-GDINO-T", LOCALIZERS[1]: "MDETR-R101"}
SUR = {"finecops_val": "FineCops val", "gref_source_disjoint": "gRef source-disjoint", "gref_full": "gRef Full"}
ARM = {"native": "Native", CELLS[0]: "$G/E$", CELLS[1]: "$G/Y$", CELLS[2]: "$S/E$", CELLS[3]: "$S/Y$",
       "joint_product": "Product", "joint_sirc": "SIRC-style"}
PRIMARY = ("mixed_augrc", "correctness_auroc", "correct_vs_no_target_auroc", "wrong_positive_vs_no_target_auroc")
COUNTS = {"finecops_val": (3567, 9426, 9029), "gref_source_disjoint": (1277, 9848, 7716), "gref_full": (1500, 11563, 9121)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_analysis(data, surface):
    if data.get("schema") != "arrow.confidence_readout_metrics/v1":
        raise ValueError("versioned confidence-readout analysis required")
    bootstrap = data.get("bootstrap", {})
    required = {"iterations": 5000, "seed": 20260911, "rng": "PCG64", "unit": "image_cluster",
                "required_seeds": list(SEEDS), "same_draw_all_localizers_heads_seeds": True,
                "q05_recomputed_each_draw": True, "fixed_threshold_fit": False}
    if any(bootstrap.get(k) != value for k, value in required.items()):
        raise ValueError("formal paired-bootstrap contract missing")
    expected_strata = {"validation"} if surface == "finecops_val" else {"testA", "testB"}
    if set(bootstrap.get("strata", {})) != expected_strata:
        raise ValueError("surface stratification differs from the locked image-cluster design")
    if (set(data.get("localizers", {})) != set(LOCALIZERS)
            or data.get("matched_cells") != list(CELLS)
            or data.get("primary_metric") != "mixed_augrc"):
        raise ValueError("complete two-localizer 2x2 matrix required")
    receipt = data.get("receipt", {})
    if (receipt.get("formal_requested_configuration") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("protocol_sha256", ""))):
        raise ValueError("source protocol and full analysis completion receipt required")
    if len(data.get("cross_readout_scores", {})) != 4:
        raise ValueError("all four frozen-weight cross-readouts required")
    images, positives, negatives = COUNTS[surface]
    for localizer, result in data["localizers"].items():
        pop = result["population"]
        if (pop.get("images") != images or pop.get("C", -1) + pop.get("W", -1) != positives
                or pop.get("N") != negatives or pop.get("records") != positives + negatives):
            raise ValueError("complete population count mismatch: " + surface)
        if set(result.get("per_seed", {})) != set(SEEDS) or result.get("conditional_counts") is None:
            raise ValueError("three seeds and complete conditional mechanism analysis required")
        if not finite(result.get("max_state_identity_error")) or result["max_state_identity_error"] > 2e-12:
            raise ValueError("three-state risk identity did not pass")
        for arm in ARMS:
            for metric in (*PRIMARY, "native_p_at_1", "existence_auroc", "mixed_aurc", "diagnostic_fpr95"):
                summary = result["summary"][arm][metric]
                ci = summary.get("ci95")
                if (not finite(summary.get("mean")) or not finite(summary.get("sample_sd"))
                        or not isinstance(ci, list) or len(ci) != 2 or not all(map(finite, ci))
                        or ci[0] > ci[1] or summary.get("undefined_replicates") != 0):
                    raise ValueError("primary score estimate or interval is incomplete")
        for metric in PRIMARY:
            contrasts = result["effects"]
            if abs(contrasts["interaction"][metric]["mean"] - contrasts["D_emit"][metric]["mean"]
                   + contrasts["D_exists"][metric]["mean"]) > 2e-12:
                raise ValueError("interaction is not D_emit minus D_exists")
        for name in EFFECTS:
            summary = result["effects"][name]["mixed_augrc"]
            if summary.get("undefined_replicates") != 0 or summary.get("ci95") is None:
                raise ValueError("main interaction interval missing")
        for seed in SEEDS:
            for arm in CELLS:
                if set(result["winner_geometry"][seed][arm]) != {"C", "W", "N"}:
                    raise ValueError("complete winner geometry required")


def number(value, signed=False):
    if value is None:
        return "--"
    # Readout effects can be only a few hundredths of an AUGRC x100 point.
    # Three decimals preserve, e.g., the +0.003 end of a zero-crossing CI.
    text = format(100*value, "+.3f" if signed else ".2f")
    return "+0.000" if text == "-0.000" and signed else "0.00" if text == "-0.00" else text


def estimate(value, signed=False, stacked=False):
    mean, ci = number(value["mean"], signed), value.get("ci95")
    interval = "undefined" if ci is None else f"[{number(ci[0], signed)}, {number(ci[1], signed)}]"
    return (r"\shortstack{" + mean + r"\\{\scriptsize " + interval + "}}") if stacked else mean + " " + interval


def point_sd(value):
    return r"\shortstack{" + number(value["mean"]) + r"\\{\scriptsize (" + number(value["sample_sd"]) + ")}}"


def table(caption, label, columns, header, lines, wide=True, small=True):
    environment = "table*" if wide else "table"
    return "\n".join([rf"\begin{{{environment}}}[t]", r"\centering" + (r"\small" if small else r"\footnotesize"),
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.15}",
        rf"\caption{{{caption}}}\label{{{label}}}", rf"\begin{{tabular}}{{{columns}}}\toprule",
        header + r"\\\midrule", *lines, r"\bottomrule\end{tabular}", rf"\end{{{environment}}}", ""])


def main_tables(data):
    rows = []
    for localizer in LOCALIZERS:
        for surface in SURFACES:
            result = data[surface]["localizers"][localizer]
            rows.append(" & ".join([LOC[localizer], SUR[surface],
                *[point_sd(result["summary"][arm]["mixed_augrc"]) for arm in CELLS],
                *[estimate(result["effects"][effect]["mixed_augrc"], True, True) for effect in EFFECTS]]) + r"\\")
        rows.append(r"\addlinespace[3pt]")
    target = table("Target by readout with unchanged Native boxes. Mixed AUGRC $\\times100$ (lower is better). "
        "The four cells show seed mean (sample SD); effects show mean [paired 95\\% CI]. "
        "$D_Y=R_{S,Y}-R_{G,Y}$, $D_E=R_{S,E}-R_{G,E}$, and $I=D_Y-D_E$. "
        "The gRef surfaces overlap and are not independent replications.", "tab:readout_matrix", "@{}llccccccc@{}",
        r"Localizer & Population & $G/E$ & $G/Y$ & $S/E$ & $S/Y$ & $D_Y$ & $D_E$ & $I$", rows)
    rows = []
    for localizer in LOCALIZERS:
        for surface in ("finecops_val", "gref_source_disjoint"):
            result = data[surface]["localizers"][localizer]
            for effect, label in (("global_emit_minus_exists", "$G$"),
                                  ("selected_emit_minus_exists", "$S$")):
                rows.append(" & ".join([LOC[localizer], SUR[surface], label,
                    *[estimate(result["effects"][effect][metric], True, True)
                      for metric in (*PRIMARY[1:], "existence_auroc", "mixed_augrc")]]) + r"\\")
    states = table("The same existence-AUROC change need not imply the same output-risk change. "
        "Emission-minus-existence effects $\\times100$ [paired 95\\% CI]; positive denotes larger AUROC, negative denotes lower AUGRC. "
        "$C/W$ tests localized correctness, $C/N$ tests successful outputs against absent targets, and "
        "$W/N$ compares two failures. The first two determine fixed-population AUGRC, whereas existence AUROC also includes the third. "
        "Readout contrasts and gRef Full are retained in Table~\\ref{tab:readout_matrix} and the complete analysis.",
        "tab:readout_states", "@{}lllccccc@{}",
        r"Localizer & Population & Readout & $C/W$ & $C/N$ & $W/N$ & Existence & AUGRC", rows)
    rows = []
    for localizer in LOCALIZERS:
        for surface in (SURFACES[0], SURFACES[1]):
            result = data[surface]["localizers"][localizer]
            for arm in ("joint_product", "joint_sirc"):
                rows.append(" & ".join([LOC[localizer], SUR[surface], ARM[arm],
                    point_sd(result["summary"][arm]["mixed_augrc"]),
                    *[estimate(result["effects"][arm+"_minus_"+baseline]["mixed_augrc"], True, True)
                      for baseline in ("native", CELLS[0], CELLS[1])]]) + r"\\")
    combos = table("Fixed combinations on the same Native boxes. Mixed AUGRC $\\times100$, and paired differences "
        "against all three prespecified references. Lower is better. Product and SIRC-style use Native and global-existence "
        "scores; only SIRC-style uses training-positive score statistics. No evaluation mixture, weight or threshold is fitted.",
        "tab:readout_combinations", "@{}lllcccc@{}", r"Localizer & Population & Score & AUGRC (SD) & $\Delta$ Native & $\Delta G/E$ & $\Delta G/Y$", rows)
    return {"table_target_readout.tex": target, "table_three_states.tex": states, "table_combinations.tex": combos}


def readout_interpretation(effect, interaction):
    de, it = effect["mean"], interaction["mean"]
    if de < 0 and it < 0:
        if effect["ci95"][1] < 0 and interaction["ci95"][1] < 0:
            return ("The selected design reduces emission risk and benefits emission more than existence; "
                    "both paired intervals resolve these directions.")
        return ("Both point estimates favor the emission-specific readout explanation, but their displayed "
                "intervals limit how precisely absolute benefit and interaction can be distinguished.")
    if de >= 0 and it < 0:
        return ("The negative interaction is not an absolute improvement in emission risk: "
                "the two contrasts must not be collapsed into an emission-repair claim.")
    if de < 0:
        return ("Emission risk decreases in the point estimate without a negative target-specific interaction; "
                "a general readout effect is distinct from an emission-specific advantage.")
    return ("This population does not show an absolute emission-risk improvement in the point estimate. "
            "The magnitudes and paired intervals, rather than an interaction sign alone, determine its scope.")


def result_prose(data):
    abstract, target, scope, states, combos = [], [], [], [], []
    for localizer in LOCALIZERS:
        result = data["finecops_val"]["localizers"][localizer]
        e = result["effects"]
        g = e["global_emit_minus_exists"]
        abstract.append(f"For {LOC[localizer]}, emission-minus-existence supervision under global readout "
                        f"changes correct-versus-wrong-box AUROC by {number(g['correctness_auroc']['mean'], True)} "
                        f"points but mixed AUGRC by {estimate(g['mixed_augrc'], True)}.")
        target.append(r"\paragraph{"+LOC[localizer]+" on FineCops.} "
                      + f"Native P@1 is {number(result['summary']['native']['native_p_at_1']['mean'])}\\%, "
                      + "unchanged across every confidence route. The absolute emission change is "
                      + f"$D_Y={estimate(e['D_emit']['mixed_augrc'], True)}$ points; "
                      + f"the existence change is $D_E={estimate(e['D_exists']['mixed_augrc'], True)}$, "
                      + f"and $I={estimate(e['interaction']['mixed_augrc'], True)}$. "
                      + readout_interpretation(e["D_emit"]["mixed_augrc"], e["interaction"]["mixed_augrc"]))
        for surface in ("gref_source_disjoint",):
            r = data[surface]["localizers"][localizer]
            scope.append(r"\paragraph{"+LOC[localizer]+", "+SUR[surface]+".} "
                         + f"Native P@1 is {number(r['summary']['native']['native_p_at_1']['mean'])}\\%. "
                         + f"The frozen-head emission readout effect is {estimate(r['effects']['D_emit']['mixed_augrc'], True)} "
                         + f"AUGRC points, and its target interaction is {estimate(r['effects']['interaction']['mixed_augrc'], True)}. "
                         + readout_interpretation(r["effects"]["D_emit"]["mixed_augrc"], r["effects"]["interaction"]["mixed_augrc"]))
        population = result["population"]
        pc, pw, pn = [population[state]/population["records"] for state in ("C", "W", "N")]
        target_effect = result["effects"]["global_emit_minus_exists"]
        cw_term = -pc*pw*target_effect["correctness_auroc"]["mean"]
        cn_term = -pc*pn*target_effect["correct_vs_no_target_auroc"]["mean"]
        if abs(cw_term+cn_term-target_effect["mixed_augrc"]["mean"]) > 2e-12:
            raise ValueError("printed state contributions do not reconstruct AUGRC")
        states.append(r"\paragraph{"+LOC[localizer]+" on FineCops.} "
            + "Under global readout, emission-minus-existence changes $U_{CW}$ by "
            + estimate(target_effect["correctness_auroc"], True)
            + " AUROC points and $U_{CN}$ by "
            + estimate(target_effect["correct_vs_no_target_auroc"], True)
            + ". Existence AUROC changes by "
            + estimate(target_effect["existence_auroc"], True)
            + ", whereas complete-output AUGRC changes by "
            + estimate(target_effect["mixed_augrc"], True)
            + ". At the point estimate, the $C/W$ and $C/N$ risk contributions are "
            + number(cw_term, True)+" and "+number(cn_term, True)
            + " AUGRC points, respectively. Their sum, not the $W/N$ comparison, reconstructs the risk change.")
        for surface in ("finecops_val", "gref_source_disjoint"):
            r = data[surface]["localizers"][localizer]
            statements=[]
            for arm in ("joint_product", "joint_sirc"):
                deltas=[r["effects"][arm+"_minus_"+b]["mixed_augrc"] for b in ("native", CELLS[0], CELLS[1])]
                improvements=[name for name,v in zip(("Native", "$G/E$", "$G/Y$"),deltas) if v["ci95"][1] < 0]
                costs=[name for name,v in zip(("Native", "$G/E$", "$G/Y$"),deltas) if v["ci95"][0] > 0]
                sentence=ARM[arm]+" has mixed AUGRC "+number(r["summary"][arm]["mixed_augrc"]["mean"])+". "
                sentence+=("Its risk is lower than "+", ".join(improvements)+" with resolved paired intervals. ") if improvements else "No prespecified comparison resolves a lower risk for this rule. "
                if costs:sentence+="It also has higher risk than "+", ".join(costs)+" with resolved intervals. "
                statements.append(sentence)
            combos.append(r"\paragraph{"+LOC[localizer]+", "+SUR[surface]+".} "+" ".join(statements))
    mm=data["finecops_val"]["localizers"][LOCALIZERS[0]]["summary"]
    if all("wrong_box_risk_cov50" in mm[arm] for arm in CELLS[:2]):
        states.append(r"\paragraph{What changes among accepted requests?} "
            + "At approximately 50\\% coverage on MM-GDINO FineCops, global emission supervision "
            + "changes the accepted wrong-box fraction from "
            + number(mm[CELLS[0]]["wrong_box_risk_cov50"]["mean"])+"\\% to "
            + number(mm[CELLS[1]]["wrong_box_risk_cov50"]["mean"])+"\\%, while the accepted no-target fraction changes from "
            + number(mm[CELLS[0]]["no_target_risk_cov50"]["mean"])+"\\% to "
            + number(mm[CELLS[1]]["no_target_risk_cov50"]["mean"])
            + "\\%. Both denominators are accepted requests, not their respective failure classes. "
            + "The fixed-coverage supplement gives achieved coverage and paired intervals.")
    return {"abstract_results.tex":"\n".join(abstract)+"\n", "target_results.tex":"\n\n".join(target)+"\n",
            "transfer_results.tex":"\n\n".join(scope)+"\n", "state_results.tex":"\n\n".join(states)+"\n",
            "combination_results.tex":"\n\n".join(combos)+"\n"}


def supplementary_mechanism(data):
    """Print focused mechanisms without substituting a conditional population."""
    conditional_lines, same_counts, difficulty_counts, cross_lines, parent_lines = [], [], [], [], []
    effect_names = {"D_emit": "$Y:S-G$", "global_emit_minus_exists": "$G:Y-E$"}
    for surface in SURFACES:
        for localizer in LOCALIZERS:
            result = data[surface]["localizers"][localizer]
            counts = result["conditional_counts"]
            for state in ("cw", "cn", "wn"):
                group = counts["same_image_" + state]
                same_counts.append(" & ".join([LOC[localizer], SUR[surface], state.upper(),
                    str(group["eligible_images"]), str(group["eligible_records"]), str(group["within_image_pairs"])]) + r"\\")
                if surface == "finecops_val":
                    for level in (1, 2, 3):
                        group = counts[f"difficulty_{state}_level{level}"]
                        difficulty_counts.append(" & ".join([LOC[localizer], state.upper(), "L"+str(level),
                            str(group["high"]), str(group["low"]), str(group["images"])]) + r"\\")
                for effect, label in effect_names.items():
                    for condition in ("same_image", "difficulty"):
                        if condition == "difficulty" and surface != "finecops_val":
                            continue
                        prefix = f"{condition}_{state}"
                        unconditional = prefix + "_comparable_unconditional"
                        conditional = prefix if condition == "same_image" else prefix + "_same_level_pair_auroc"
                        difference = prefix + ("_unconditional_minus_conditional" if condition == "same_image"
                                               else "_unconditional_minus_same_level")
                        values = result["effects"][effect]
                        conditional_lines.append(" & ".join([LOC[localizer], SUR[surface], label,
                            ("Image " if condition == "same_image" else "Level ") + state.upper(),
                            estimate(values[unconditional], True, True),
                            estimate(values[conditional], True, True),
                            estimate(values[difference], True, True)]) + r"\\")
            for name, desc in sorted(data[surface]["cross_readout_scores"].items()):
                arm = desc["trained_head"]
                expected_eval = "native_selected" if arm.startswith("global_max") else "global_max"
                if desc["eval_readout"] != expected_eval:
                    raise ValueError("alternate readout is not the opposite of the trained readout")
                effect = result["effects"]["fixed_weights__"+name+"_minus_matched"]["mixed_augrc"]
                cross_lines.append(" & ".join([LOC[localizer], SUR[surface], ARM[arm],
                    "$S$" if expected_eval == "native_selected" else "$G$",
                    point_sd(result["summary"][arm]["mixed_augrc"]),
                    point_sd(result["summary"][name]["mixed_augrc"]), estimate(effect, True, True)]) + r"\\")
            if surface == "finecops_val":
                for state in ("all", "C", "W"):
                    metric = "parent_pair_" + state
                    group = counts[metric]
                    parent_lines.append(" & ".join([LOC[localizer], state,
                        r"\shortstack{"+str(group["pairs"])+r"\\("+str(group["images"])+")}",
                        *[number(result["summary"][arm][metric]["mean"]) for arm in (CELLS[0], CELLS[1], CELLS[3])],
                        estimate(result["effects"]["global_emit_minus_exists"][metric], True, True),
                        estimate(result["effects"]["D_emit"][metric], True, True)]) + r"\\")
    outputs = {}
    counts_caption = ("Same-image state comparisons: exact participating images, requests and within-image pairs. "
                      "The conditional and comparable-unconditional estimates use these same participating requests; "
                      "bootstrap image multiplicity applies once to each within-copy pair.")
    outputs["supp_condition_counts.tex"] = table(counts_caption, "tab:supp_v6_image_counts",
        "@{}lllrrr@{}", r"Localizer & Population & State pair & Images & Requests & Within-image pairs", same_counts, small=False)
    outputs["supp_condition_counts.tex"] += table(
        "FineCops original-positive-difficulty counts. High/low states are the first/second state in the pair label. "
        "Zero low-state counts at L2/L3 expose the absence of same-level no-target comparisons; "
        "these cells are not assigned an AUROC. Images overlap across levels/state comparisons.",
        "tab:supp_v6_difficulty_counts", "@{}lllrrr@{}",
        r"Localizer & State pair & Level & High-state requests & Low-state requests & Images", difficulty_counts, small=False)
    outputs["supp_parent_pairs.tex"] = table(
        "True edited-negative/positive-parent pair wins on FineCops, $\\times100$. "
        "Raw columns are equal-seed mean pair wins; effects include paired image-cluster 95\\% intervals. "
        "C/W refers to the parent's Native correctness. Pair counts repeat positive parents; C/W image counts may overlap. "
        "These are diagnostic pair wins, not official Recall@1.",
        "tab:supp_v6_parent_pairs", "@{}llcccccc@{}",
        r"Localizer & Parent & Pairs (images) & $G/E$ & $G/Y$ & $S/Y$ & $G:Y-E$ & $Y:S-G$", parent_lines, small=False)
    for name, lines, columns, header, caption in (
        ("conditionals", conditional_lines, "@{}llllccc@{}",
         r"Localizer & Population & Effect & Condition & Comparable uncond. & Conditional & Uncond.$-$cond.",
         "Conditional effects and their uncertainty, AUROC $\\times100$. Each cell is mean [paired 95\\% CI], "
         "not a raw score AUROC. $Y:S-G$ is the emission readout effect; $G:Y-E$ is the global target effect. "
         "Image conditioning matches participating requests exactly. Level conditioning changes the available "
         "within-level pair composition (including the zero cells in the count table), not a difficulty-matched "
         "target distribution. An undefined interval is not zero effect or proof of attribution."),
        ("cross_readouts", cross_lines, "@{}llllccc@{}",
         r"Localizer & Population & Trained head & Alternate read & Matched (SD) & Alternate (SD) & Alt.$-$matched [CI]",
         "Frozen weights, alternate inference readout: mixed AUGRC $\\times100$. "
         "Matched and alternate values are seed mean (sample SD); differences use paired 95\\% intervals. "
         "Every row keeps Native boxes fixed. These off-diagonal readings diagnose the training/deployment readout "
         "contract; they are not substituted for the matched four-cell matrix."),
    ):
        parts = []
        for start in range(0, len(lines), 18):
            parts.append(table(caption + f" Part {start//18+1}.", f"tab:supp_v6_{name}_{start//18+1}",
                               columns, header, lines[start:start+18], small=False))
        outputs["supp_" + name + ".tex"] = "\n".join(parts)
    return outputs


def supplementary_fixed_coverage(data):
    """Conditional accepted-error composition, not all-request failure mass."""
    lines = []
    for localizer in LOCALIZERS:
        result = data["finecops_val"]["localizers"][localizer]
        for arm in CELLS:
            values = result["summary"][arm]
            for coverage in (50, 90):
                suffix = "_cov" + str(coverage)
                wrong = values["wrong_box_risk" + suffix]
                absent = values["no_target_risk" + suffix]
                mixed = values["mixed_risk" + suffix]
                achieved = values["achieved_coverage" + suffix]["mean"]
                if not all(finite(v["mean"]) for v in (wrong, absent, mixed)) or not finite(achieved):
                    raise ValueError("fixed-coverage composition is incomplete")
                if abs(wrong["mean"] + absent["mean"] - mixed["mean"]) > 2e-12:
                    raise ValueError("accepted wrong/no-target risks do not sum to mixed risk")
                if achieved + 2e-12 < coverage/100 or achieved > 1:
                    raise ValueError("whole-boundary-tie achieved coverage contract failed")
                lines.append(" & ".join([LOC[localizer], ARM[arm], str(coverage),
                    f"{100*achieved:.3f}", estimate(wrong,False,True), estimate(absent,False,True),
                    estimate(mixed,False,True)]) + r"\\")
    return {"supp_fixed_coverage.tex": table(
        "FineCops val error composition at requested coverage 50/90\\%. "
        "Achieved coverage is the equal-seed mean fraction of all requests retained, in percent; "
        "the boundary tie group is accepted in full. W, N and mixed are conditional error probabilities "
        "among accepted requests, $\\times100$, with paired image-cluster 95\\% intervals. "
        "They are not counts or failure rates divided by all requests. W+N=mixed before display rounding. "
        "No deployment threshold is fitted; all other coverages/surfaces remain in the full analysis JSON.",
        "tab:supp_v6_fixed_coverage", "@{}llccccc@{}",
        r"Localizer & Head & Requested (\%) & Achieved (\%) & W$\mid$accept & N$\mid$accept & Failure$\mid$accept",
        lines, small=False)}


def seed_effect_values(result, effect, metric="mixed_augrc"):
    """Subtract frozen per-seed metrics, never infer seed variation from image CI."""
    definitions = {
        "D_emit": {CELLS[3]: 1., CELLS[1]: -1.},
        "D_exists": {CELLS[2]: 1., CELLS[0]: -1.},
        "interaction": {CELLS[3]: 1., CELLS[2]: -1., CELLS[1]: -1., CELLS[0]: 1.},
        "global_emit_minus_exists": {CELLS[1]: 1., CELLS[0]: -1.},
    }
    coefficients = definitions[effect]
    points = {seed: sum(coefficient * result["per_seed"][seed][arm][metric]
                        for arm, coefficient in coefficients.items()) for seed in SEEDS}
    if "per_seed_effects" in result:
        provided = {seed: result["per_seed_effects"][seed][effect][metric] for seed in SEEDS}
        if any(not finite(provided[s]) or abs(provided[s]-points[s]) > 2e-12 for s in SEEDS):
            raise ValueError("provided per-seed effect differs from frozen cell subtraction")
    if not all(map(finite, points.values())):
        raise ValueError("nonfinite per-seed primary effect")
    mean = sum(points.values()) / len(points)
    sd = math.sqrt(sum((value-mean)**2 for value in points.values()) / (len(points)-1))
    reported = result["effects"][effect][metric]
    if abs(mean-reported["mean"]) > 2e-12 or abs(sd-reported["sample_sd"]) > 2e-12:
        raise ValueError("per-seed effect mean/sample SD differs from the frozen analysis")
    return points


def supplementary_seed_effects(data):
    lines = []
    labels = {"D_emit": "$D_Y$", "D_exists": "$D_E$", "interaction": "$I$",
              "global_emit_minus_exists": "$G:Y-E$"}
    for localizer in LOCALIZERS:
        for surface in SURFACES:
            result = data[surface]["localizers"][localizer]
            for effect, label in labels.items():
                points = seed_effect_values(result, effect)
                value = result["effects"][effect]["mixed_augrc"]
                mean_sd = (r"\shortstack{" + number(value["mean"], True)
                           + r"\\(" + f"{100*value['sample_sd']:.3f}" + ")}")
                ci = value.get("ci95")
                interval = "undefined" if ci is None else "[" + ", ".join(number(v, True) for v in ci) + "]"
                lines.append(" & ".join([LOC[localizer], SUR[surface], label,
                    *[number(points[seed], True) for seed in SEEDS], mean_sd, interval]) + r"\\")
    parts = []
    for start in range(0, len(lines), 12):
        parts.append(table(
            "Per-seed mixed-AUGRC effects, $\\times100$: fixed seeds 17/42/73, their arithmetic mean "
            "and sample SD ($n-1$ denominator), and the paired image-cluster 95\\% interval of the "
            "fixed three-seed mean effect. The image interval is not a training-seed interval: it can exclude "
            "zero even when the listed seed effects have different signs. No $n=3$ t-test is used. "
            f"Part {start//12+1}.",
            f"tab:supp_v6_seed_effects_{start//12+1}", "@{}lllccccc@{}",
            r"Localizer & Population & Effect & Seed 17 & Seed 42 & Seed 73 & Mean (SD) & Image CI",
            lines[start:start+12], small=False))
    return {"supp_seed_effects.tex": "\n".join(parts)}


def supplementary(data):
    seed_lines, endpoint_lines, winner_lines, crossover_lines = [], [], [], []
    for surface in SURFACES:
        for localizer in LOCALIZERS:
            r = data[surface]["localizers"][localizer]
            for arm in ARMS:
                endpoint_lines.append(" & ".join([LOC[localizer],SUR[surface],ARM[arm],
                    *[estimate(r["summary"][arm][m],False,True) for m in
                      ("native_p_at_1","existence_auroc","correctness_auroc","diagnostic_fpr95","mixed_aurc","mixed_augrc")]])+r"\\")
                seed_lines.append(" & ".join([LOC[localizer],SUR[surface],ARM[arm],
                    *[number(r["per_seed"][seed][arm]["mixed_augrc"]) for seed in SEEDS]])+r"\\")
            for arm in CELLS:
                for state in ("C","W","N"):
                    diagnostic=[r["winner_geometry"][seed][arm][state] for seed in SEEDS]
                    def mean_field(field):
                        values=[d.get(field) for d in diagnostic]
                        return None if any(v is None for v in values) else sum(values)/len(values)
                    winner_lines.append(" & ".join([LOC[localizer],SUR[surface],ARM[arm],state,
                        str(diagnostic[0]["records"]),number(mean_field("winner_differs_mean")),
                        number(mean_field("winner_native_box_iou_mean")),number(mean_field("confidence_winner_correct_fraction"))])+r"\\")
            for effect in ("global_emit_minus_exists","selected_emit_minus_exists","D_emit"):
                cross=r["augrc_crossovers"][effect]
                point=cross["point"]
                ci=cross["ci95"]
                conditional=cross["conditional_on_interior_ci95"]
                crossover_lines.append(" & ".join([LOC[localizer],SUR[surface],effect.replace("_",r"\_"),
                    number(point.get("prior")),point["status"].replace("_"," "),
                    "--" if ci is None else "["+", ".join(number(v) for v in ci)+"]",
                    "--" if conditional is None else "["+", ".join(number(v) for v in conditional)+"]",
                    str(cross["bootstrap_status_counts"].get("interior",0))])+r"\\")
    # Split lengthy supplementary tables into bounded pages; no numeric row is dropped.
    specs=("seed_endpoints",seed_lines,"@{}lllccc@{}",r"Localizer & Population & Score & Seed 17 & Seed 42 & Seed 73",
           "Mixed AUGRC by seed, $\\times100$. Native repeats one frozen prediction route, not three localizer training seeds.")
    outputs={}
    all_specs=[specs,
        ("complete_endpoints",endpoint_lines,"@{}lllcccccc@{}",r"Localizer & Population & Score & P@1 & Existence & $C/W$ & FPR95 & AURC & AUGRC",
         "All score endpoints, $\\times100$: seed means with paired image-cluster intervals. P@1 is identical across confidence routes."),
        ("winner_geometry",winner_lines,"@{}llllcccc@{}",r"Localizer & Population & Head & State & Requests & Query differs & Box IoU & Winner correct",
         "Dense-logit global-winner geometry, $\\times100$; seed-mean rates/IoU, not causal spatial contributions. "
         "For G-trained heads this winner supplies matched confidence. For S-trained heads it is only a "
         "counterfactual diagnostic: matched S deployment always reads the Native query and has zero "
         "readout-index disagreement. Winner-correct is shown only for Native-wrong positives."),
        ("crossovers",crossover_lines,"@{}lllllccc@{}",r"Localizer & Population & Contrast & Prior & Root status & All-draw CI & Interior-only CI & Interior draws",
         "AUGRC crossover prior (percent). Interior-only intervals are conditional summaries; non-interior replicates remain counted out of 5,000.")]
    for name,lines,columns,header,caption in all_specs:
        pieces=[]
        per_page=18 if name in ("complete_endpoints","conditionals") else 24
        for start in range(0,len(lines),per_page):
            pieces.append(table(caption+f" Part {start//per_page+1}.",f"tab:supp_v6_{name}_{start//per_page+1}",
                                columns,header,lines[start:start+per_page],small=False))
        outputs["supp_"+name+".tex"]="\n".join(pieces)
    outputs.update(supplementary_mechanism(data))
    outputs.update(supplementary_fixed_coverage(data))
    outputs.update(supplementary_seed_effects(data))
    return outputs


def figure_one(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"ps.fonttype":42,
                         "svg.fonttype":"none","svg.hashsalt":"readout-v6-target-interaction"})
    fig=plt.figure(figsize=(8.4,4.0),facecolor="white")
    fig.text(.025,.975,"Same output box. Which query judges it?",fontsize=15,weight="bold",va="top")
    fig.text(.025,.885,"(a) Change the readout, keep the prediction",fontsize=9.4,weight="bold")
    ax=fig.add_axes([.025,.49,.46,.345]);ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    def box(x,y,w,h,text,color="#eff3f6"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.015,rounding_size=.02",facecolor=color,edgecolor="#94a0aa",lw=.8))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8.4)
    box(.01,.64,.28,.27,"Image + expression\nFrozen localizer")
    box(.40,.64,.19,.27,"Native query\n"+r"$j^*$","#e0edf3")
    box(.72,.64,.25,.27,"One output box\n"+r"$b^*$","#e0edf3")
    ax.annotate("",(.4,.78),(.29,.78),arrowprops={"arrowstyle":"->","lw":1.1})
    ax.annotate("",(.72,.78),(.59,.78),arrowprops={"arrowstyle":"->","lw":1.1})
    box(.06,.08,.40,.32,"Global: "+r"$\max_j h(q_j,s_j^0)$"+"\nAny valid query may win","#fff0d9")
    box(.57,.08,.40,.32,"Selected: "+r"$h(q_{j^*},s_{j^*}^0)$"+"\nRead the output query","#e5f2eb")
    ax.text(.51,.46,"Dense logits in both arms",ha="center",va="center",fontsize=8,color="#5d6570")
    fig.text(.545,.885,"(b) Change the target only on wrong boxes",fontsize=9.4,weight="bold")
    table_ax=fig.add_axes([.545,.535,.425,.285]);table_ax.axis("off")
    t=table_ax.table(cellText=[["Absent (N)","0","0"],["Present, wrong box (W)","1","0"],
                              ["Present, correct box (C)","1","1"]],
                    colLabels=["Fixed request state","Exists E","Emit Y"],cellLoc="center",colWidths=[.59,.19,.22],bbox=[0,0,1,1])
    t.auto_set_font_size(False);t.set_fontsize(8)
    for (i,j),cell in t.get_celld().items():
        cell.set_linewidth(.5);cell.set_edgecolor("#c9d0d5")
        cell.set_facecolor("#fff0d9" if i==2 else "#f0f3f6" if i==0 else "white")
    fig.text(.545,.485,"Two labels × two readouts; Native boxes never change",fontsize=8.2,color="#4d5660")
    colors={"finecops_val":"#276d99","gref_source_disjoint":"#c57932"}
    for left,effect,title in ((.20,"D_emit",r"Absolute emission change $D_Y$"),(.64,"interaction",r"Target × readout interaction $I$")):
        a=fig.add_axes([left,.135,.30,.235])
        a.axvline(0,color="#89929c",lw=.8,ls="--")
        for loc_index,localizer in enumerate(LOCALIZERS):
            for surface,offset in (("finecops_val",.13),("gref_source_disjoint",-.13)):
                e=data[surface]["localizers"][localizer]["effects"][effect]["mixed_augrc"]
                lo,hi=[100*v for v in e["ci95"]];mean=100*e["mean"];y=1-loc_index+offset
                a.plot([lo,hi],[y,y],color=colors[surface],lw=1.5)
                a.plot(mean,y,"o",color=colors[surface],ms=4)
        a.set_yticks([1,0],[LOC[l] for l in LOCALIZERS] if effect=="D_emit" else [])
        a.set_ylim(-.45,1.45);a.set_title(title,fontsize=9,pad=8)
        a.set_xlabel("AUGRC change ×100 (lower is better)",fontsize=8)
        a.tick_params(axis="y",length=0);a.spines[["top","right","left"]].set_visible(False)
    fig.text(.025,.37,"(c) Measured effects",fontsize=9.4,weight="bold")
    for x,surface in ((.30,"finecops_val"),(.52,"gref_source_disjoint")):
        fig.text(x,.017,"● "+SUR[surface],color=colors[surface],fontsize=8)
    fig.text(.025,.017,"Three-seed means; paired 95% CI",fontsize=7.7,color="#4d5660")
    fig.savefig(output/"figure1_readout.pdf",metadata={"CreationDate":None,"ModDate":None,"Title":"Target by readout with fixed Native boxes"})
    fig.savefig(output/"figure1_readout.svg",metadata={"Date":None})
    plt.close(fig)


def self_test():
    assert number(-1e-9,True)=="+0.000"
    assert number(0.000025686774,True)=="+0.003"
    e={"mean":-.01,"ci95":[-.02,-.001]};i={"mean":-.005,"ci95":[-.01,-.001]}
    assert "reduces emission risk" in readout_interpretation(e,i)
    assert "not an absolute improvement" in readout_interpretation({"mean":.01,"ci95":[.001,.02]},i)
    assert "general readout effect" in readout_interpretation(e,{"mean":.005,"ci95":[-.001,.01]})
    for bad in ({},{"schema":"arrow.confidence_readout_metrics/v1"}):
        try:validate_analysis(bad,"finecops_val")
        except (ValueError,KeyError):pass
        else:raise AssertionError("incomplete analyses accepted")
    print("v6 asset synthetic formatter/interpretation/fail-closed checks passed; no assets written")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finecops",type=Path);parser.add_argument("--gref-full",type=Path)
    parser.add_argument("--gref-disjoint",type=Path)
    parser.add_argument("--output-dir",type=Path,default=PAPER/"generated/readout_v6")
    parser.add_argument("--check",action="store_true");parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:self_test();return
    paths={"finecops_val":args.finecops,"gref_full":args.gref_full,"gref_source_disjoint":args.gref_disjoint}
    if any(path is None for path in paths.values()):parser.error("all three completed analysis sources are required")
    data={surface:json.loads(path.read_text()) for surface,path in paths.items()}
    for surface,result in data.items():validate_analysis(result,surface)
    protocols={result["receipt"]["protocol_sha256"] for result in data.values()}
    if len(protocols)!=1:raise ValueError("surface analyses do not belong to one study protocol")
    sources={surface:{"path":str(path.resolve()),"sha256":sha(path)} for surface,path in paths.items()}
    root=args.output_dir.resolve()
    if args.check:
        receipt=json.loads((root/"receipt.json").read_text())
        if receipt["sources"]!=sources or receipt["generator_sha256"]!=sha(__file__):raise ValueError("asset provenance drift")
        for name,digest in receipt["outputs"].items():
            if sha(root/name)!=digest:raise ValueError("generated asset drift: "+name)
        print("v6 complete-source and generated-asset hashes verified");return
    if root.exists():raise FileExistsError("append-only asset directory already exists")
    text={**main_tables(data),**result_prose(data),**supplementary(data)}
    ready="% Complete analyses validated; this is not a submission-readiness certificate.\n"+r"\def\ReadoutV6AssetsComplete{1}"+"\n"
    root.mkdir(parents=True)
    for name,content in text.items():(root/name).write_text(content)
    figure_one(data,root)
    outputs={path.name:sha(path) for path in root.iterdir() if path.is_file()}
    outputs["publication_ready.tex"]=hashlib.sha256(ready.encode()).hexdigest()
    receipt={"schema":"arrow.paper.readout_v6_assets/v1","status":"complete_analysis_assets",
        "sources":sources,"protocol_sha256":next(iter(protocols)),"generator_sha256":sha(__file__),"outputs":outputs,
        "all_surfaces":list(SURFACES),"all_localizers":list(LOCALIZERS),"training_seeds":list(SEEDS),
        "bootstrap_iterations":5000,"spatial_cause_inferred":False,"model_forward":False,"metric_recompute":False,
        "manuscript_review_required":True}
    (root/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    # The manuscript's required gate appears only after all assets and receipt.
    (root/"publication_ready.tex").write_text(ready)
    print(json.dumps({"output":str(root),"assets":len(outputs),"manuscript_review_required":True}))


if __name__=="__main__":main()
