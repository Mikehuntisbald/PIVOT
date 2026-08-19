#!/usr/bin/env python3
"""Render ARROW's four main LaTeX tables from the semantic registry."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
REGISTRY_PATH = PAPER / "data" / "paper_numbers.json"


class Registry:
    def __init__(self) -> None:
        self.payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.numbers = self.payload["numbers"]
        self.used: dict[str, dict[str, Any]] = {}

    def item(self, key: str) -> dict[str, Any]:
        item = self.numbers[key]
        self.used[key] = item
        return item

    def value(self, key: str) -> float:
        return float(self.item(key)["value"])

    def sd_from_seeds(self, key: str) -> float:
        values = list(self.item(key)["by_seed"].values())
        return statistics.stdev(float(value) for value in values)

    def reset_used(self) -> None:
        self.used = {}


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}"


def frac(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def signed_pp(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:+.{digits}f}"


def kparams(value: float) -> str:
    return "--" if value < 0 else ("0" if value == 0 else f"{value / 1000.0:.1f}")


def write_table(name: str, body: str, registry: Registry) -> None:
    table_path = PAPER / "tables" / f"{name}.tex"
    table_path.write_text(body.rstrip() + "\n", encoding="utf-8")
    csv_path = PAPER / "data" / "table_sources" / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "key", "value", "unit", "surface", "status", "source_path",
        "source_sha256", "source_json_path", "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key in sorted(registry.used):
            item = registry.used[key]
            writer.writerow({column: item.get(column) for column in columns})


def write_supp_table(name: str, body: str, registry: Registry) -> None:
    table_path = PAPER / "supplement" / "tables" / f"{name}.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(body.rstrip() + "\n", encoding="utf-8")
    csv_path = PAPER / "data" / "table_sources" / f"{name}.csv"
    columns = [
        "key", "value", "unit", "surface", "status", "source_path",
        "source_sha256", "source_json_path", "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key in sorted(registry.used):
            item = registry.used[key]
            writer.writerow({column: item.get(column) for column in columns})


def table1(reg: Registry) -> None:
    reg.reset_used()
    rows = [
        (
            "Frozen base",
            pct(reg.value("main.test5.frozen_base")),
            pct(reg.value("main.strict2031.frozen_base")),
            "--", "--",
            kparams(reg.value("efficiency.route.frozen_base.active_params")),
            kparams(reg.value("efficiency.route.frozen_base.cumulative_params")),
            kparams(reg.value("efficiency.route.frozen_base.inference_added_params")),
        ),
        (
            "+ complete-expression ranker",
            "--", "--", "--", "--",
            kparams(reg.value("efficiency.route.ranker.active_params")),
            kparams(reg.value("efficiency.route.ranker.cumulative_params")),
            kparams(reg.value("efficiency.route.ranker.inference_added_params")),
        ),
        (
            "+ static Admission$^{\\dagger}$",
            "--", "--", "--", "--", "0",
            kparams(reg.value("efficiency.route.static_admission.cumulative_params")), "--",
        ),
        (
            "+ learned Admission",
            pct(reg.value("main.test5.arrow")),
            "--",
            pct(reg.value("anatomy.eligible_gt_recall50")),
            f"{reg.value('anatomy.mean_eligible_queries'):.1f}",
            kparams(reg.value("efficiency.route.learned_admission.active_params")),
            kparams(reg.value("efficiency.route.learned_admission.cumulative_params")),
            kparams(reg.value("efficiency.route.learned_admission.inference_added_params")),
        ),
        (
            "+ isolated Rejection (ARROW-V)",
            pct(reg.value("main.test5.arrow")),
            pct(reg.value("main.strict2031.arrow")),
            pct(reg.value("anatomy.eligible_gt_recall50")),
            f"{reg.value('anatomy.mean_eligible_queries'):.1f}",
            kparams(reg.value("efficiency.route.full_arrow.active_params")),
            kparams(reg.value("efficiency.route.full_arrow.cumulative_params")),
            kparams(reg.value("efficiency.route.full_arrow.inference_added_params")),
        ),
        (
            "ARROW-T",
            pct(reg.value("admission_input.arrow_t.test5")),
            pct(reg.value("main.strict2031.arrow")),
            "--", "--",
            kparams(reg.value("efficiency.route.full_arrow.active_params")),
            kparams(reg.value("efficiency.route.full_arrow.cumulative_params")),
            kparams(reg.value("efficiency.route.full_arrow.inference_added_params")),
        ),
    ]
    body_rows = "\n".join(" & ".join(row) + r" \\" for row in rows)
    body = rf"""% Generated by paper/scripts/make_main_tables.py.
\begin{{table*}}[t]
\centering
\caption{{Cumulative decision factorization. Intermediate decomposition rows are validation-only mechanism evidence; held-out Test5 and Strict-TN2031 cells are shown only where sealed records exist. Rejection does not alter the localization route.}}
\label{{tab:main}}
\scriptsize
\setlength{{\tabcolsep}}{{3.4pt}}
\begin{{tabular}}{{lccccccc}}
\toprule
Route & Test5 Acc$\uparrow$ & Strict FPR95$\downarrow$ & Elig. recall$\uparrow$ & Elig. queries$\downarrow$ & Active & Cum. & Runtime-loaded $\Delta$ \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\vspace{{2pt}}
\parbox{{0.98\textwidth}}{{\footnotesize Accuracy, FPR95, and recall are percentages; parameter columns report thousands. Runtime-loaded $\Delta$ includes the serialized auxiliary and a frozen scalar; it is not a count of deployed score owners. Learned Admission and full ARROW have bitwise-identical localization records. $^\dagger$Static Admission is a validation-only mechanism route, so no held-out result or exact runtime-loaded count is assigned. ARROW-T reuses the rejector through bitwise confidence parity.}}
\end{{table*}}
"""
    write_table("table1_main", body, reg)


def table2(reg: Registry) -> None:
    reg.reset_used()
    cosines = [reg.value(f"ownership.o0.seed{seed}.cosine_mean") for seed in (17, 42, 73)]
    sign_conflicts = [reg.value(f"ownership.o0.seed{seed}.sign_conflict_mean") for seed in (17, 42, 73)]
    rows = [
        (
            "Shared scalar", "No", "Yes",
            "/".join(f"{value:.2f}" for value in cosines),
            f"{statistics.fmean(sign_conflicts):.2f}",
            pct(reg.value("ownership.o0.val3_macro")),
            pct(reg.value("ownership.o0.test5")),
            pct(reg.value("ownership.o0.calibration_fpr95")),
            pct(reg.value("ownership.o0.strict2031_fpr95")),
        ),
        (
            "Separate heads/shared feature$^{\\dagger}$", "Yes", "Yes", "shared path", "--",
            pct(reg.value("ownership.o1.val3_macro")), "--",
            pct(reg.value("ownership.o1.calibration_fpr95")), "--",
        ),
        (
            "Isolated/interleaved", "Yes", "No", "zero by design", "--",
            pct(reg.value("ownership.o2.val3_macro")),
            pct(reg.value("ownership.o2.test5")),
            pct(reg.value("ownership.o2.calibration_fpr95")),
            pct(reg.value("ownership.o2.strict2031_fpr95")),
        ),
        (
            "Isolated/phased", "Yes", "No", "zero by design", "--",
            pct(reg.value("ownership.o3.val3_macro")),
            pct(reg.value("main.test5.arrow")),
            pct(reg.value("ownership.o3.calibration_fpr95")),
            pct(reg.value("main.strict2031.arrow")),
        ),
    ]
    body_rows = "\n".join(" & ".join(row) + r" \\" for row in rows)
    strict_gain = pct(reg.value("ownership.o2_minus_o0.strict_fpr95_reduction"))
    route_gain = signed_pp(reg.value("ownership.o2_minus_o0.test5_gain"), 3)
    body = rf"""% Generated by paper/scripts/make_main_tables.py.
\begin{{table*}}[t]
\centering
\caption{{Separate outputs are insufficient without exclusive trainable ownership. Isolation changes Test5 by {route_gain} points while reducing held-out Strict-TN2031 FPR95 by {strict_gain} points.}}
\label{{tab:ownership}}
\scriptsize
\setlength{{\tabcolsep}}{{2.4pt}}
\begin{{tabular}}{{lcccccccc}}
\toprule
Design & Sep. heads? & Shared feature? & Grad. cosine & Sign-conf. & Val3$\uparrow$ & Test5$\uparrow$ & Cal.$\downarrow$ & Strict$\downarrow$ \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\vspace{{2pt}}
\parbox{{0.98\textwidth}}{{\footnotesize $^\dagger$The separate-head/shared-feature control was evaluated on validation and calibration only; no held-out Test5 or Strict result is claimed. ``Shared path'' records the non-isolated autograd topology; no unregistered gradient summary is inserted.}}
\end{{table*}}
"""
    write_table("table2_ownership", body, reg)


def table3(reg: Registry) -> None:
    reg.reset_used()
    rows = []
    for route, label in (("arrow_v", "V"), ("arrow_t", "T"), ("arrow_n", "N")):
        test_key = f"admission_input.{route}.test5"
        switch_key = f"admission_input.{route}.switch_success"
        sd = reg.sd_from_seeds(test_key) if "by_seed" in reg.item(test_key) else math.nan
        test_cell = pct(reg.value(test_key))
        if not math.isnan(sd):
            test_cell += rf" $\pm$ {pct(sd)}"
        rows.append((
            label,
            test_cell,
            pct(reg.value(switch_key)),
            pct(reg.value(f"finecops.matched.{route}.positive_p1_macro")),
            pct(reg.value(f"finecops.matched.{route}.negative_text_recall1")),
            pct(reg.value(f"finecops.matched.{route}.negative_image_recall1")),
        ))
    body_rows = "\n".join(" & ".join(row) + r" \\" for row in rows)
    coverage = pct(reg.value("finecops.arrow_v.support_coverage"))
    body = rf"""% Generated by paper/scripts/make_main_tables.py.
\begin{{table}}[t]
\centering
\caption{{Admission cues. FineCops columns share one exact-support matched surface.}}
\label{{tab:admission}}
\scriptsize
\setlength{{\tabcolsep}}{{1.7pt}}
\begin{{tabular}}{{lccccc}}
\toprule
Cue & Test5$\uparrow$ & Switch$\uparrow$ & P@1$\uparrow$ & Text R@1$\uparrow$ & Image R@1$\uparrow$ \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\vspace{{2pt}}
\parbox{{0.98\columnwidth}}{{\scriptsize V is the sealed anchor; T/N Test5 is prospective post-release evidence. Percentages; mean $\pm$ SD. Visual-support coverage is {coverage}\%; unsupported rows are not counted as correct rejection.}}
\end{{table}}
"""
    write_table("table3_admission", body, reg)


def table4(reg: Registry) -> None:
    reg.reset_used()
    fine_rows = []
    for route, label in (
        ("frozen_base", "Base"),
        ("ranker_rejector", "Ranker+rej."),
        ("arrow_t", "ARROW-T"),
    ):
        fine_rows.append((
            label,
            pct(reg.value(f"finecops.{route}.positive_p1_macro")),
            pct(reg.value(f"finecops.{route}.negative_text_recall1")),
            pct(reg.value(f"finecops.{route}.negative_image_recall1")),
            pct(reg.value(
                "finecops.official_exact.frozen_base.negative_text_auroc_type_macro"
                if route == "frozen_base"
                else "finecops.official_exact.isolated_rejector.negative_text_auroc_type_macro"
            )),
            pct(reg.value(
                "finecops.official_exact.frozen_base.negative_image_auroc_type_macro"
                if route == "frozen_base"
                else "finecops.official_exact.isolated_rejector.negative_image_auroc_type_macro"
            )),
        ))
    fine_rows.append((
        "Original GDINO-T",
        pct(reg.value("finecops.original_ogc.native_max.positive_p1_macro")),
        pct(reg.value("finecops.original_ogc.native_max.negative_text_recall1")),
        pct(reg.value("finecops.original_ogc.native_max.negative_image_recall1")),
        pct(reg.value("finecops.original_ogc.native_max.negative_text_auroc_type_macro")),
        pct(reg.value("finecops.original_ogc.native_max.negative_image_auroc_type_macro")),
    ))
    fine_body = "\n".join(" & ".join(row) + r" \\" for row in fine_rows)

    gref_rows = []
    for surface, surface_label in (("full", "Full restricted"), ("rejector_supervision_disjoint", "Rejector-supervision-disjoint")):
        for route, route_label in (("frozen_base", "Frozen base"), ("isolated_rejector", "Isolated rejector")):
            fixed = "--" if route == "frozen_base" else pct(reg.value(f"gref.{surface}.{route}.fixed_tpr"))
            gref_rows.append((
                ({"Full restricted": "Full", "Rejector-supervision-disjoint": "Rej.-sup.-disj."}[surface_label]) if route == "frozen_base" else "",
                {"Frozen base": "Base", "Isolated rejector": "Rejector"}[route_label],
                pct(reg.value(f"gref.{surface}.{route}.auroc")),
                pct(reg.value(f"gref.{surface}.{route}.aupr")),
                pct(reg.value(f"gref.{surface}.{route}.fpr95")),
                fixed,
            ))
    gref_body = "\n".join(" & ".join(row) + r" \\" for row in gref_rows)
    body = rf"""% Generated by paper/scripts/make_main_tables.py.
\begin{{table}}[t]
\centering
\caption{{External evaluation. FineCops uses full-test text-cue routes; gRefCOCO is the restricted single/no-target slice.}}
\label{{tab:external}}
\textbf{{(a) FineCops-Ref full test}}\\[2pt]
\scriptsize
\setlength{{\tabcolsep}}{{1.0pt}}
\begin{{tabular}}{{lccccc}}
\toprule
Route & P@1$\uparrow$ & T R@1$\uparrow$ & I R@1$\uparrow$ & T AUC$\uparrow$ & I AUC$\uparrow$ \\
\midrule
{fine_body}
\bottomrule
\end{{tabular}}
\\[5pt]
\textbf{{(b) gRefCOCO rejection transfer}}\\[2pt]
\scriptsize
\begin{{tabular}}{{llcccc}}
\toprule
Slice & Route & AUROC$\uparrow$ & AUPR$\uparrow$ & FPR95$\downarrow$ & Fixed TPR \\
\midrule
{gref_body}
\bottomrule
\end{{tabular}}
\\[-1pt]
\parbox{{0.98\columnwidth}}{{\scriptsize T/I denote negative text/image. Original GDINO-T is our local replay of the unmodified Swin-T OGC checkpoint with its native max-token score; the preregistered expression-mean sensitivity is reported in the supplement. Table AUC uses the pinned official evaluator's historical level-1-positive scope for every row. Fig.~\ref{{fig:external}} separately labels the audited all-positive FineCops gain as a point estimate. Fixed TPR uses the sealed source threshold. Multi-target gRefCOCO is excluded.}}
\end{{table}}
"""
    write_table("table4_external", body, reg)


def number_macros(reg: Registry) -> None:
    reg.reset_used()
    values = {
        "ArrowTestGain": signed_pp(reg.value("main.test5.gain")),
        "ArrowTestGainLo": pct(reg.item("main.test5.gain")["ci95"][0]),
        "ArrowTestGainHi": pct(reg.item("main.test5.gain")["ci95"][1]),
        "ArrowStrictGain": pct(reg.value("main.strict2031.gain")),
        "ArrowStrictGainLo": pct(reg.item("main.strict2031.gain")["ci95"][0]),
        "ArrowStrictGainHi": pct(reg.item("main.strict2031.gain")["ci95"][1]),
        "ArrowOwnershipStrictGain": pct(reg.value("ownership.o2_minus_o0.strict_fpr95_reduction")),
        "ArrowVisualSwitch": pct(reg.value("admission_input.arrow_v.switch_success"), 1),
        "ArrowTextSwitch": pct(reg.value("admission_input.arrow_t.switch_success"), 1),
        "ArrowNullSwitch": pct(reg.value("admission_input.arrow_n.switch_success"), 1),
        "ArrowFineCopsCoverage": pct(reg.value("finecops.arrow_v.support_coverage"), 2),
        "ArrowFineCopsFixedTPR": pct(reg.value("finecops.arrow_t.fixed_tpr"), 2),
        "ArrowGrefFixedTPR": pct(reg.value("gref.full.isolated_rejector.fixed_tpr"), 2),
        "OriginalOgcNativeP": pct(reg.value("finecops.original_ogc.native_max.positive_p1_macro")),
        "OriginalOgcNativeTextR": pct(reg.value("finecops.original_ogc.native_max.negative_text_recall1")),
        "OriginalOgcNativeImageR": pct(reg.value("finecops.original_ogc.native_max.negative_image_recall1")),
        "OriginalOgcNativeTextAuc": pct(reg.value("finecops.original_ogc.native_max.negative_text_auroc_type_macro")),
        "OriginalOgcNativeImageAuc": pct(reg.value("finecops.original_ogc.native_max.negative_image_auroc_type_macro")),
        "OriginalOgcMeanP": pct(reg.value("finecops.original_ogc.matched_mean.positive_p1_macro")),
        "OriginalOgcMeanTextR": pct(reg.value("finecops.original_ogc.matched_mean.negative_text_recall1")),
        "OriginalOgcMeanImageR": pct(reg.value("finecops.original_ogc.matched_mean.negative_image_recall1")),
        "OriginalOgcMeanTextAuc": pct(reg.value("finecops.original_ogc.matched_mean.negative_text_auroc_type_macro")),
        "OriginalOgcMeanImageAuc": pct(reg.value("finecops.original_ogc.matched_mean.negative_image_auroc_type_macro")),
    }
    lines = ["% Generated by paper/scripts/make_main_tables.py."]
    lines.extend(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items())
    (PAPER / "data" / "generated_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def supplement_training_table(reg: Registry) -> None:
    reg.reset_used()
    def mean_sd(key: str) -> str:
        return f"{pct(reg.value(key))} $\\pm$ {pct(float(reg.item(key)['sample_sd']))}"

    static_seed17 = pct(reg.value("admission_training.a0_a2.seed17_val3_micro"))
    admission_rows = [
        r"A0 & Frozen & Frozen & Off & Off & -- & -- \\",
        "A1 & Train & Frozen & On & On & "
        + mean_sd("admission_training.a1.val3_macro")
        + " & " + pct(reg.value("admission_training.a1.test5")) + r" \\",
        r"A2 & Frozen & Train & On & On & -- & -- \\",
        "A3 & Train & Train & On & Off & "
        + mean_sd("admission_training.a3.val3_macro") + r" & -- \\",
        "A4 & Train & Train & Off & On & "
        + mean_sd("admission_training.a4.val3_macro") + r" & -- \\",
        "A5 & Train & Train & On & On & "
        + mean_sd("admission_training.a5.val3_macro")
        + " & " + pct(reg.value("admission_training.a5.test5")) + r" \\",
    ]

    rejection_rows = [r"C0 & Identity & No & No & No & -- & -- & -- \\"]
    for row, objective, queue, trust, margin in (
        ("c1", "Batch negative", "No", "No", "No"),
        ("c2", "Negative FPR", "Yes", "No", "No"),
        ("c3", "Negative FPR", "Yes", "Yes", "No"),
        ("c4", "Negative FPR", "Yes", "Yes", "Yes"),
    ):
        fpr = f"rejection_training.{row}.fpr95"
        win = f"rejection_training.{row}.pair_win"
        strict = (
            pct(reg.value(f"rejection_training.{row}.strict2031_fpr95"))
            if row in {"c2", "c3"} else "--"
        )
        rejection_rows.append(
            f"{row.upper()} & {objective} & {queue} & {trust} & {margin} & "
            f"{mean_sd(fpr)} & {mean_sd(win)} & {strict} " + r"\\"
        )

    data_rows = [
        r"D0 & None (positive-only) & None & -- & -- & No-TN control \\",
        "D1 & Edited text & Unverified & "
        + mean_sd("rejection_training.d1.fpr95") + " & "
        + mean_sd("rejection_training.d1.pair_win") + r" & Stress control \\",
        "D3 & Proposal-covered pairs & Proposal-covered & "
        + mean_sd("rejection_training.c3.fpr95") + " & "
        + mean_sd("rejection_training.c3.pair_win") + r" & Main clean data \\",
    ]

    ownership_rows = []
    for row, owner, schedule, test_key, strict_key in (
        ("o0", "Shared scalar", "Interleaved", "ownership.o0.test5", "ownership.o0.strict2031_fpr95"),
        ("o1", "Separate heads/shared feature", "Interleaved", None, None),
        ("o2", "Exclusive owners", "Interleaved", "ownership.o2.test5", "ownership.o2.strict2031_fpr95"),
        ("o3", "Exclusive owners", "Phased", "ownership.o3.test5", "ownership.o3.strict2031_fpr95"),
    ):
        test = pct(reg.value(test_key)) if test_key else "--"
        strict = pct(reg.value(strict_key)) if strict_key else "--"
        ownership_rows.append(
            f"{row.upper()} & {owner} & {schedule} & "
            f"{mean_sd(f'ownership.{row}.val3_macro')} & "
            f"{mean_sd(f'ownership.{row}.calibration_fpr95')} & "
            f"{mean_sd(f'ownership.{row}.calibration_pair_win')} & "
            f"{test} & {strict} " + r"\\"
        )

    body = r"""% Generated by paper/scripts/make_main_tables.py.
\begin{table*}[t]
\centering
\caption{Complete Admission training grid. Val3 is validation-only mechanism evidence; Test5 appears only for the preregistered A5--A1 contrast.}
\label{tab:supp-admission-grid}
\label{tab:supp-training}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}\toprule
ID & Surface & Aux. & Category-complete & Preserve & Val3 macro$\uparrow$ & Test5$\uparrow$\\\midrule
""" + "\n".join(admission_rows) + rf"""
\bottomrule\end{{tabular}}
\parbox{{0.97\textwidth}}{{\scriptsize Values are three-seed mean $\pm$ sample SD unless only a held-out mean is shown. A2's training-only auxiliary is never deployed and its three deployable checkpoints are bitwise equal to A0. The only sealed static-route score is a seed-17 Val3 micro value of {static_seed17}\%; it is not inserted into the three-seed macro column.}}
""" + r"""
\end{table*}

\begin{table*}[t]
\centering
\caption{Complete clean Rejection-objective grid and allowed data-provenance controls. Calibration rows are mechanism evidence. Strict-TN2031 was opened only for preregistered C3--C2; no row-specific checkpoint or milestone was selected.}
\label{tab:supp-rejection-grid}
\scriptsize
\setlength{\tabcolsep}{2.3pt}
\begin{tabular}{llcccccc}\toprule
ID & Objective & Queue? & Trust? & Margin? & Cal. FPR95$\downarrow$ & Pair win$\uparrow$ & Strict$\downarrow$\\\midrule
""" + "\n".join(rejection_rows) + r"""
\bottomrule\end{tabular}
\vspace{4pt}

\textbf{Data provenance controls}\par\smallskip
\begin{tabular}{llllll}\toprule
ID & Negative source & Verification scope & Cal. FPR95$\downarrow$ & Pair win$\uparrow$ & Role\\\midrule
""" + "\n".join(data_rows) + r"""
\bottomrule\end{tabular}
\parbox{0.97\textwidth}{\scriptsize C0/D0 have no independently trained absolute rejector, hence ``--''. D1 is an unverified weak-data stress control; D3 is the proposal-covered clean source used by C3. Historical rule-swap and approximate matched rows are excluded from manuscript tables.}
\end{table*}

\begin{table*}[t]
\centering
\caption{Complete ownership and scheduling block. Val3/calibration columns are mechanism evidence; held-out cells occur only for the preregistered O2--O0 and O3--O2 contrasts.}
\label{tab:supp-ownership-grid}
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lllccccc}\toprule
ID & Trainable topology & Schedule & Val3$\uparrow$ & Cal. FPR95$\downarrow$ & Pair win$\uparrow$ & Test5$\uparrow$ & Strict$\downarrow$\\\midrule
""" + "\n".join(ownership_rows) + r"""
\bottomrule\end{tabular}
\parbox{0.97\textwidth}{\scriptsize O0 has one shared deployed scalar; O1 separates outputs but retains a shared trainable feature; O2/O3 use exclusive owners. O2 establishes the ownership effect. The O3--O2 strict contrast is not significant, so phasing is not claimed as independently beneficial.}
\end{table*}
"""
    write_supp_table("supp_training_controls", body, reg)


def supplement_ref_breakdown_table(reg: Registry) -> None:
    reg.reset_used()
    rows = []
    for family, labels in (
        ("Admission", (("A1", "a1"), ("A3", "a3"), ("A4", "a4"), ("A5", "a5"))),
        ("Ownership", (("O0", "o0"), ("O1", "o1"), ("O2", "o2"), ("O3", "o3"))),
    ):
        prefix = "admission_training" if family == "Admission" else "ownership"
        for label, row in labels:
            values = [
                pct(reg.value(f"{prefix}.{row}.refcoco_val.mean")),
                pct(reg.value(f"{prefix}.{row}.refcocop_val.mean")),
                pct(reg.value(f"{prefix}.{row}.refcocog_val.mean")),
                *[
                    pct(reg.value(f"{prefix}.{row}.seed{seed}.val3_micro"))
                    for seed in (17, 42, 73)
                ],
            ]
            rows.append(" & ".join((family, label, *values)) + r" \\")
    body = r"""% Generated by paper/scripts/make_main_tables.py.
\begin{table*}[t]
\centering
\caption{Per-split and per-seed Ref validation results for every trajectory with a sealed three-seed forward. Split columns average the three seeds; seed columns are pooled Val3 micro Acc@0.5. These are validation-only mechanism results.}
\label{tab:supp-ref-breakdown}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{llcccccc}\toprule
Block & ID & RefCOCO & RefCOCO+ & RefCOCOg & s17 & s42 & s73\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\parbox{0.97\textwidth}{\scriptsize A0/A2 are absent here because only deployment parity and a single-seed static route were sealed; reporting copied three-seed values would overstate the available evidence.}
\end{table*}
"""
    write_supp_table("supp_ref_breakdown", body, reg)


def supplement_gap_table(reg: Registry) -> None:
    reg.reset_used()
    rows = []
    total_n = int(reg.value("gap_sensitivity.gap_3.n"))
    for label, display in (
        ("0", "0"), ("0p5", "0.5"), ("1", "1"), ("2", "2"),
        ("3", "3"), ("5", "5"), ("10", "10"), ("infinity", r"$\infty$"),
    ):
        prefix = f"gap_sensitivity.gap_{label}"
        reg.value(f"{prefix}.gap")
        rows.append(
            f"{display} & {pct(reg.value(f'{prefix}.acc50'))} & "
            f"{pct(reg.value(f'{prefix}.eligible_recall50'))} & "
            f"{reg.value(f'{prefix}.mean_eligible_queries'):.1f} " + r"\\"
        )
    body = r"""% Generated by paper/scripts/make_main_tables.py.
\begin{table}[t]
\centering
\caption{Relative Admission-gap sensitivity on the fixed seed-42 Val3 single forward. This exploratory sweep cannot change the sealed model.}
\label{tab:supp-gap}
\scriptsize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{rrrr}\toprule
Gap & Acc$\uparrow$ & Elig. recall$\uparrow$ & Elig. candidates$\downarrow$\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
""" + rf"""
\parbox{{0.96\columnwidth}}{{\scriptsize Accuracy and recall are percentages. Every value is pooled over {total_n:,} expressions. Candidate counts are recomputed from the immutable records referenced by the sealed sweep summary; their byte hashes are stored in the derived receipt.}}
\end{{table}}
"""
    write_supp_table("supp_gap_sensitivity", body, reg)


def supplement_finecops_subgroup_table(reg: Registry) -> None:
    reg.reset_used()
    positive_rows = []
    for level in (1, 2, 3):
        positive_rows.append(
            f"L{level} & {pct(reg.value(f'finecops.subgroup.frozen_base.positive_l{level}_p1'))} & "
            f"{pct(reg.value(f'finecops.subgroup.arrow_t.positive_l{level}_p1'))} " + r"\\"
        )
    subgroup_names = {
        "text": (
            ("Attribute", "L1"), ("Attribute", "L2"), ("Object", "L1"),
            ("Object", "L2"), ("Order", "L1"), ("Order", "L2"),
            ("Relation", "L1"), ("Relation", "L2"),
            ("Swap-attr.", "L1"), ("Swap-attr.", "L2"),
        ),
        "image": (
            ("Attribute", "L1"), ("Attribute", "L2"), ("Flip", "L1"),
            ("Flip", "L2"), ("Object", "L1"), ("Object", "L2"),
            ("Order", "L1"), ("Swap-attr.", "L1"), ("Swap-attr.", "L2"),
        ),
    }
    kind_rows: dict[str, list[str]] = {"text": [], "image": []}
    for kind, groups in subgroup_names.items():
        for name, level in groups:
            raw_name = name.lower().replace("-attr.", "_attr").replace(".", "")
            safe = f"{raw_name}_{level.lower()}"
            count = int(reg.value(f"finecops.subgroup.count.{kind}.{safe}"))
            base = pct(reg.value(f"finecops.subgroup.frozen_base.{kind}.{safe}.recall1"))
            arrow = pct(reg.value(f"finecops.subgroup.arrow_t.{kind}.{safe}.recall1"))
            kind_rows[kind].append(f"{name} & {level} & {count} & {base} & {arrow} " + r"\\")
    body = r"""% Generated by paper/scripts/make_main_tables.py.
\begin{table*}[t]
\centering
\caption{FineCops-Ref descriptive subgroup results (full-test routes; three-seed means). Only subgroups explicitly present in the sealed result object are shown. No subgroup threshold is fit.}
\label{tab:supp-finecops-subgroups}
\scriptsize
\centering\textbf{Positive difficulty}\par\smallskip
\begin{tabular}{lrr}\toprule
Level & Base P@1 & ARROW-T P@1\\\midrule
""" + "\n".join(positive_rows) + r"""
\bottomrule\end{tabular}
\par\medskip
\begin{minipage}[t]{0.48\textwidth}
\centering\textbf{Negative text}\par\smallskip
\begin{tabular}{llrrr}\toprule
Type & Level & $n$ & Base R@1 & ARROW-T R@1\\\midrule
""" + "\n".join(kind_rows["text"]) + r"""
\bottomrule\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering\textbf{Negative image}\par\smallskip
\begin{tabular}{llrrr}\toprule
Type & Level & $n$ & Base R@1 & ARROW-T R@1\\\midrule
""" + "\n".join(kind_rows["image"]) + r"""
\bottomrule\end{tabular}
\end{minipage}
\parbox{0.97\textwidth}{\scriptsize ARROW-T uses FineCops' structured target noun as its explicit canonical cue. These full-test descriptive rows are not input-matched comparisons to expression-only systems and are not additional confirmatory families.}
\end{table*}
"""
    write_supp_table("supp_finecops_subgroups", body, reg)


def supplement_intervention_table(reg: Registry) -> None:
    reg.reset_used()
    labels = (
        ("Full geometry + full rank + bound support", "p0_s0"),
        ("Canonical geometry", "p1"),
        ("Generic-object geometry", "p2"),
        ("Canonical ranking text", "p3"),
        ("Bound support, different loader draw", "s1"),
        ("Same-category support shuffle", "s2"),
        ("Wrong-category support", "s3"),
        ("Zero support", "s4"),
    )
    rows = []
    for label, row in labels:
        prefix = f"intervention.{row}"
        values = [
            pct(reg.value(f"{prefix}.acc50")),
            pct(reg.value(f"{prefix}.all_query_oracle_recall50")),
            pct(reg.value(f"{prefix}.eligible_recall50")),
            f"{reg.value(f'{prefix}.eligible_query_count_mean'):.1f}",
            f"{reg.value(f'{prefix}.eligible_mask_hamming_mean'):.1f}",
            pct(reg.value(f"{prefix}.top1_query_churn")),
        ]
        rows.append(" & ".join((label, *values)) + r" \\")
    body = r"""% Generated by paper/scripts/make_main_tables.py.
\begin{table*}[t]
\centering
\caption{Zero-training prompt and support interventions on the fixed seed-42 Val3 surface. Geometry prompts alter the candidate universe; ranking-text and support interventions keep the full-expression geometry path unless named otherwise.}
\label{tab:supp-interventions}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}\toprule
Intervention & Acc$\uparrow$ & All-oracle$\uparrow$ & Elig.-recall$\uparrow$ & Elig. $n$ & Mask $\Delta$ & Top-1 churn\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\end{table*}
"""
    write_supp_table("supp_input_interventions", body, reg)


def main() -> None:
    reg = Registry()
    table1(reg)
    table2(reg)
    table3(reg)
    table4(reg)
    supplement_training_table(reg)
    supplement_ref_breakdown_table(reg)
    supplement_gap_table(reg)
    supplement_finecops_subgroup_table(reg)
    supplement_intervention_table(reg)
    number_macros(reg)


if __name__ == "__main__":
    main()
