# ARROW CVPR Codex Handoff v1

## Mission

Turn the current sealed repository evidence into a submission-ready CVPR paper package **without retraining, test-set tuning, or changing any sealed model/metric decision**. The work is presentation, record replay, figure/table generation, efficiency measurement, citation organization, and README cleanup.

## Non-negotiable constraints

- Do not train a new model.
- Do not select a new checkpoint, Gap, queue size, threshold, or Admission input.
- Do not fit calibration on FineCops or gRefCOCO.
- Do not add a new alias to improve FineCops support coverage.
- Do not rerun held-out Test5 or Strict-TN merely to obtain prettier numbers if immutable records already exist.
- Generate numerical tables and plots from sealed receipts/per-example records; do not manually type values into plotting code.
- Preserve all preregistration amendments and failed-attempt ledgers.
- Never call gRefCOCO `D3-disjoint` globally image-disjoint. Rename it **Rejector-supervision-disjoint** in paper-facing outputs.
- Never call D3 labels image-global or all-query verified.
- Never compare ARROW-V’s 95.60%-covered FineCops score directly with full-test ARROW-T/MM-GDINO-T as if the evaluation universes were identical.

## Source-of-truth documents

Read these first and create a source registry containing path, git SHA, SHA256 if available, and the claims/numbers owned by each file:

- `docs/paper_cvpr_u2v5_complete_ablation_design_20260817.md`
- `docs/paper_cvpr_u2v5_ablation_results_20260818.md`
- `docs/paper_cvpr_arrow_admission_input_results_20260818.md`
- `docs/paper_cvpr_arrow_finecops_protocol_20260819.md`
- `docs/paper_cvpr_arrow_finecops_results_20260819.md`
- `docs/arrow_grefcoco_rejection_transfer_20260820.md`
- `docs/stageb_u2v5_leakage_clean_anchor_20260817.md`
- `outputs/u2v5_cvpr_ablation_20260817/`
- `outputs/arrow_admission_input_20260818/`
- `outputs/arrow_finecops_20260819/`
- `outputs/arrow_grefcoco_20260820/`

If a named output directory differs locally, locate it from committed receipts rather than guessing.

## Deliverable directory

Create:

```text
paper/
  main.tex
  references.bib
  sections/
    01_intro.tex
    02_related.tex
    03_problem.tex
    04_method.tex
    05_experiments.tex
    06_results.tex
    07_limitations.tex
    08_conclusion.tex
  figures/
    fig1_teaser.pdf
    fig2_method_ownership.pdf
    fig3_mechanism_controllability.pdf
    fig4_external_transfer.pdf
    qualitative/
  tables/
    table1_main.tex
    table2_ownership.tex
    table3_admission.tex
    table4_external.tex
  data/
    source_registry.json
    paper_numbers.json
    table_sources/*.csv
    plot_sources/*.csv
  scripts/
    build_number_registry.py
    make_main_tables.py
    make_mechanism_figure.py
    make_admission_figure.py
    make_external_figure.py
    select_qualitative_examples.py
    benchmark_efficiency.py
    validate_paper_package.py
  supplement/
    supplement.tex
    tables/
    figures/
  README.md
```

## Phase 1 — Build a semantic number registry

Create `paper/data/paper_numbers.json`. Every number must include:

```json
{
  "key": "ownership.o2_minus_o0.strict_fpr95_reduction",
  "value": 0.028065,
  "unit": "absolute",
  "surface": "Strict-TN2031",
  "ci95": [0.009268, 0.046384],
  "direction": "higher_is_better_reduction",
  "source_path": "docs/paper_cvpr_u2v5_ablation_results_20260818.md",
  "source_sha256": "...",
  "status": "confirmatory",
  "notes": "paired image-cluster bootstrap"
}
```

Required registry groups:

- baseline-to-ARROW Test5 and Strict-TN gains;
- M0--M4 cumulative routes;
- A0/A1/A5 main Admission mechanism rows;
- O0/O1/O2/O3 ownership metrics;
- A/B/C Admission-input results and category-switch CIs;
- FineCops full and matched results, coverage, and official reference;
- gRefCOCO Full, Rejector-supervision-disjoint, and fixed-threshold results;
- parameter/memory/latency data;
- all claim-boundary caveats.

Add a validator that fails when a LaTeX table contains a numeric literal not present in the registry, except formatting constants.

## Phase 2 — Normalize paper-facing names

Use this mapping in main paper output:

| Legacy | Paper-facing |
|---|---|
| B58 | Frozen base |
| R100 | Complete-expression ranker |
| D3 U50 / confidence12 | Isolated rejector / verified rejector |
| U2-v5 A | ARROW-V |
| Admission B | ARROW-T |
| Admission C | ARROW-N |
| Gap3 | relative admission margin \(\gamma=3\) |
| strict2031 | Strict-TN2031 |
| D3-disjoint | Rejector-supervision-disjoint |

Legacy names may appear in appendix parentheses once for reproducibility.

## Phase 3 — Generate the four main figures

### Figure 1: teaser/failure decomposition

Use real, deterministic examples selected from sealed records. Select by predeclared rules:

1. a base wrong-top1 case where an IoU>=0.5 candidate exists and ARROW fixes it;
2. an admission rescue with large eligible-set reduction and retained GT;
3. a no-target case where D3 lowers confidence relative to B58;
4. optionally one honest failure.

Do not select by subjective visual attractiveness. Save the selection rule, sample IDs, and source hashes in `paper/data/qualitative_selection.json`.

Overlay:

- expression;
- base top-1;
- GT-compatible candidate;
- eligible set summary;
- ARROW output or abstention;
- scores only if their meaning is defined.

### Figure 2: method/ownership diagram

Produce as vector SVG/PDF. Required structure:

```text
Image + complete expression
        -> Frozen candidate generator
        -> candidates {q_i, b_i}

Category cue (visual exemplar or category text)
        -> Admission owner
        -> relative-gap eligible set

Complete expression + eligible candidates
        -> frozen complete-expression ranker
        -> provisional top-1

Detached base statistics
        -> isolated rejection owner
        -> accept / abstain
```

Mark:

- frozen blocks;
- trainable owner sets;
- stop-gradient;
- training-only auxiliary residual with dashed line and “not deployed”;
- no shared trainable path between Admission/Ranking and Rejection;
- ARROW-V and ARROW-T as alternate Admission inputs.

### Figure 3: mechanism + controllability

Panel A: O0 per-seed gradient cosine and O2 structural isolation.

Panel B: paired CI for O2−O0 Test5 and Strict-TN FPR95 reduction. Use a zero reference line.

Panel C: A/B/C standard Test5 versus category-switch success. The intended message is “endpoint accuracy is insensitive; responsibility control is not.”

All plotted data must come from `plot_sources/*.csv` generated from the registry.

### Figure 4: external rejection transfer

Show:

- internal Strict-TN FPR95 reduction;
- FineCops AUROC improvements for negative expression/image;
- gRefCOCO Full and Rejector-supervision-disjoint AUROC/FPR95 improvements;
- fixed-threshold TPR separately, with a 0.95 source reference line.

Do not combine differently scaled metrics into one unlabeled bar axis. Prefer aligned small multiples.

## Phase 4 — Generate main tables

### Table 1: cumulative route

Rows:

- Frozen base
- + complete-expression ranker
- + static Admission
- + learned Admission
- + isolated Rejection (ARROW-V)
- ARROW-T optional adjacent row

Columns:

- Test5 Acc@0.5
- Strict-TN2031 FPR95
- eligible GT recall
- mean eligible candidates
- active/cumulative trained params
- inference-added params

Add a note that learned Admission and full ARROW have identical localization records.

### Table 2: ownership

Rows: shared scalar, separate outputs/shared feature, isolated owners. Put phased isolated in supplement unless layout allows.

Columns:

- output separation
- feature-owner sharing
- gradient cosine
- sign conflict
- val/Test5 localization
- calibration/strict FPR95

### Table 3: Admission cue

Rows: ARROW-V/T/N.

Columns:

- Test5
- fresh category-switch success
- FineCops matched positive P@1
- matched neg-expression Recall@1
- matched neg-image Recall@1
- seed SD

Footnote exact-support coverage and equal parameter count.

### Table 4: external transfer

Subtable A, FineCops full-test standard-input rows:

- MM-GDINO-T official reference
- Frozen base
- complete-expression ranker + isolated rejector
- ARROW-T

Subtable B, gRefCOCO:

- Full: Frozen base vs isolated rejector
- Rejector-supervision-disjoint: Frozen base vs isolated rejector

Report fixed-threshold TPR in its own column and state it is not tuned externally.

## Phase 5 — Efficiency receipt

Implement `benchmark_efficiency.py` without changing model weights.

Measure at the paper preprocessing resolution:

- Frozen base;
- + ranker;
- ARROW-T;
- ARROW-V with support embedding cached;
- ARROW-V with support encoding uncached.

Protocol:

- batch size 1;
- fixed manifest of at least 200 representative images/expressions;
- exclude file I/O and visualization;
- CUDA event timing with synchronization;
- warmup >= 50 iterations;
- report median, mean, p90, and IQR;
- reset/report peak allocated and reserved VRAM;
- record GPU, driver, CUDA, PyTorch, precision, input resolution;
- separately count total params, frozen params, cumulative-ever-trained params, active trainable params per phase, and inference-added params.

Do not report the existing approximate parameter or memory values as measured final numbers without this receipt.

## Phase 6 — Confidence anatomy, zero training

From immutable records only, create supplementary diagnostics:

- internal/FineCops/gRef positive and negative CDFs;
- q05/q10/q25/q50/q75/q90/q95;
- FineCops object/attribute/relation/swap/negative-image subgroups;
- where possible, noun-absent vs noun-present-but-expression-invalid TNs;
- source fixed threshold drawn on each domain.

Interpretation must distinguish:

- ranking/discrimination transfer;
- operating-point calibration transfer.

## Phase 7 — Qualitative appendix

Select 8--12 examples under deterministic categories:

- base wrong top-1, ARROW correct;
- static Admission failure, learned Admission correct;
- ARROW-V category switch succeeds, ARROW-T fails;
- ARROW-T succeeds, ARROW-N fails;
- external TN correctly down-ranked;
- fixed-threshold false rejection under domain shift;
- hard compositional false positive;
- two representative ARROW failures.

Each example must record its selection predicate and sample ID.

## Phase 8 — Paper draft

Use `/mnt/data/ARROW_CVPR_Paper_Blueprint_v1.md` as the prose seed. Preserve these framing rules:

- first contribution is decision factorization, not patch-text fusion;
- second contribution is exclusive parameter ownership;
- visual support is the strongest Admission interface, not the entire method;
- separate outputs/shared trunk is the critical closest internal control;
- FineCops/gRef show discrimination transfer, not universal calibration;
- do not present development history as the method section;
- Stage-A/Stage-B failures belong in motivation or supplement, compressed to one paragraph.

Target one main thesis per section. Avoid internal version identifiers in the narrative.

## Phase 9 — Related-work matrix and BibTeX

Collect primary BibTeX for at least:

- Grounding DINO
- RefCOCO / RefCOCO+ / RefCOCOg
- GREC / gRefCOCO / GREx
- FineCops-Ref
- T-Rex2
- PET-DINO
- DINO-X or DINOv
- RECANTFormer
- HieA2G
- InstAlign / InstanceVG
- RC-GRPO
- PCGrad
- GradNorm or CAGrad
- relevant adapter/PEFT citation if parameter efficiency is emphasized

Create `paper/data/closest_work_matrix.csv` with columns:

```text
paper, task, specialist_or_mllm, text_prompt, visual_prompt,
no_target, multi_target, dedicated_rejector,
shared_feature_owner_tested, structural_zero_cross_gradient,
main_distinction, citation_key
```

Do not use “first” language unless the matrix and primary papers support it.

## Phase 10 — Rewrite the root README

The current README is dominated by the historical third-party GroundingDINO and Stage-A/Stage-B workflow. Replace the top-level presentation with:

1. ARROW title and one-sentence thesis;
2. method figure;
3. key internal and external results;
4. ARROW-V versus ARROW-T interfaces;
5. quick inference example;
6. reproduction commands for sealed evaluations;
7. model/record availability and hashes;
8. citation;
9. a short “Historical lineage” link to the old Stage-A/Stage-B documents.

Move or collapse the old setup/training history below the ARROW release section. Keep attribution to the GroundingDINO implementation.

## Phase 11 — Validation gates

`validate_paper_package.py` must fail if:

- a paper-facing table contains `U2-v5`, `B58`, `R100`, `D3-disjoint`, `surface8`, `confidence12`, or `Gap3` outside an approved appendix mapping;
- gRef text contains “image-disjoint” without the explicit Stage-A exposure caveat;
- FineCops ARROW-V covered results share a column labeled full test without a coverage note;
- any external threshold is described as tuned or calibrated on the target test;
- D2/D2m/D3m appear in manuscript tables;
- positive-trust or phased scheduling is described as a proven independent gain;
- support patches are described as necessary for every correct localization;
- a plot/table number lacks a registry source;
- LaTeX does not compile;
- figure fonts are unreadable at one-column/two-column print size.

## Definition of done

- CVPR LaTeX compiles with no missing references or overflowing tables.
- Four main figures and four main tables are generated by scripts.
- All paper numbers are registry-bound and reproducible.
- Efficiency receipt is measured and committed.
- Supplement contains complete controlled ablations and caveats.
- README presents ARROW rather than the historical Stage-A/B development process.
- No new model or held-out selection decision has been introduced.
