# Grounding confidence empirical study — current paper package

**Current v8: Confidence for Which Prediction? Supervision Coverage Shapes Grounding Reliability**

[Main PDF](empirical_study_v8.pdf) · [Supplement PDF](empirical_supplement_v8.pdf) ·
[Main source](empirical_study_v8.tex) · [Supplement source](empirical_supplement_v8.tex) ·
[Results and argument](../docs/coverage_v8_results_and_story_20260906.md).

Compiled and checked: main 6 pages including references; supplement 22 pages.
[Build receipt](data/coverage_v1/v8_build_receipt.json) ·
[PDF checks](data/coverage_v1/v8_pdf_validation.json).

All twelve coverage heads, cache evaluations and three 5,000-draw analyses
are complete. The paper now centers the controlled coverage finding:
Full-positive supervision improves the correct-output head but worsens the
existence head, with different C/W and C/N consequences. Frozen transfer
tests the response; readout controls and fixed combinations support the same
question instead of forming a chronological experiment report.

Figure 1(b) retains the explicit three-seed display. Figure 1(c) and Table 2
present the coverage intervention; Table 3 and Figure 2 explain the error
tradeoff and its population boundary. Old papers and artifacts remain unchanged.

```bash
make -C paper current TECTONIC=/path/to/tectonic
make -C paper empirical-v8-audit
```

Builds write to `paper/.build/empirical_v8`, not over versioned PDFs.
The asset builder checks completed source analyses and fixed runtime code;
it does not train, score models, fit thresholds or repeat the bootstrap.
The public `empirical-v8-audit` gate rebases file locations to this checkout
and checks the original SHA bindings. Sealed absolute-path receipts and
renderer bytes are never rewritten. Earlier host-specific audit commands
remain preserved separately.

## Preserved v7.1 manuscript snapshot

**Confidence for Which Prediction? Supervision, Readout, and Coverage in Visual Grounding**

Preserved v7.1: [main PDF](empirical_study_v7_1.pdf), [supplement PDF](empirical_supplement_v7.pdf),
[main source](empirical_study_v7_1.tex), [supplement source](empirical_supplement_v7.tex).
Figure 1(b) now shows paired effects for seeds17/42/73 and distinguishes
the seed17-driven inference-only mean from the smaller matched-retraining effects.
The [v7 main PDF](empirical_study_v7.pdf) and [source](empirical_study_v7.tex) remain preserved.
The new [12-head coverage intervention](../docs/confidence_coverage_v1_20260906.md)
had completed all training and scoring at this snapshot; no coverage result
was yet included in this draft. See the historical
[execution snapshot](../docs/confidence_coverage_v1_execution_20260906.md).
This is an editorial/analytic revision using the unchanged v6 results, not a
new experiment. The main paper is seven pages including references; the
supplement is 19 pages.

The argument is **opposite risk consequences → output-query readout test →
three-state decomposition → supervision coverage and transfer → fixed combinations**.
Interaction remains a reported diagnostic, not the headline contribution.
New Figure 1 and main-text tables elevate inference-only versus retrained
readout and L1 versus cross-difficulty comparisons. The no-crossover statement
for the L1 mean curve is an analytic consequence of existing pairwise means;
it carries no new risk CI or simultaneous curve guarantee.

See [the revision and numerical interpretation](../docs/evidence_v7_revision_20260906.md)
and [v7 build receipt](data/evidence_v7/paper_build_receipt.json).
The [v7.1 build/seed-display receipt](data/coverage_v1/v7_1_build_receipt.json)
binds the updated PDF and the separate coverage launch protocol.
The [v6 result document](../docs/readout_v6_final_results_20260906.md) and
[record-only reproduction guide](../docs/readout_v6_reproduction.md) retain the
complete experimental details and statistics.

```bash
make -C paper empirical-v7-1 empirical-supplement-v7
make -C paper empirical-v7-seeds-audit
```

Builds write the main to `paper/.build/empirical_v7_1` and the preserved supplement
to `paper/.build/empirical_v7`, not over the versioned PDFs.
Set `TECTONIC=/path/to/tectonic` if needed. No weights, images, raw records,
optimizer, detector traversal or new bootstrap are needed for this revision.

## Preserved v6 study

**Confidence for Which Prediction? Supervision Targets and Query Readout in Visual Grounding**

Preserved v6: [main PDF](empirical_study_v6.pdf), [supplement PDF](empirical_supplement_v6.pdf),
[main source](empirical_study_v6.tex), [supplement source](empirical_supplement_v6.tex).
The main paper is eight pages including references; the complete supplement is
18 pages. Both have been compiled and visually checked. Numeric assets bind
the completed two-localizer, three-seed, three-surface analyses; all 18 new
heads are complete, without trunk updates or reopening FineCops Test.

The argument is **question → target/readout control → three-state explanation →
second-model scope → fixed-combination implications**. Figure 1 separates
absolute emission changes from their interaction. Table 1 gives all four risks
and paired effects; Table 2 relates C/W, C/N and W/N changes to existence AUROC
and output risk; Table 3 compares combinations with all three references.
The supplement retains seed heterogeneity, cross-readouts, conditional populations,
fixed-coverage failure composition and absent crossover roots.

See [the completed results](../docs/readout_v6_final_results_20260906.md),
[execution ledger](../docs/confidence_readout_v6_20260906.md),
[CPU reproduction guide](../docs/readout_v6_reproduction.md), and
[build receipt](data/readout_v6/paper_build_receipt.json).
This is a post-hoc-motivated mechanism study with a prospectively fixed recipe,
not a virgin confirmation or an assertion of image-disjoint pretraining.
Head supervision covers L1 positive parents; seed sample SD and image-bootstrap
intervals quantify different uncertainty. Current venue compliance and the author
submission decision remain separate from completing this experiment and draft.

## Build v6

```bash
make -C paper empirical-v6 empirical-supplement-v6
make -C paper empirical-v6-audit
```

Builds go to `paper/.build/empirical_v6`, never over the versioned PDFs in this
directory. Set `TECTONIC=/path/to/tectonic` if needed. The SHA audit uses the
committed lightweight analysis files and generated assets, not model weights
or private per-example records. `empirical-v6-assets` is for a fresh artifact
directory; existing generated assets are checked, not silently replaced.

## Other historical work

This staged package includes v6, v7, v7.1 and current v8 sources, reviewed PDFs,
and the immutable evidence required by their build gates. Earlier empirical
drafts and unfinished legacy ownership edits remain in the original working
tree outside this staged release; none were deleted. The pre-existing method
paper remains in Git history and the historical `main.tex` workflow.

## Anonymous review packaging

The manuscript PDFs are review artifacts, not an anonymized repository release.
Do not upload this repository or its history as an anonymous code supplement:
sealed receipts intentionally retain experiment-host paths. Prepare a separate,
history-free and identity-scrubbed review snapshot if required.

## Prerequisites

Python 3.11 or newer and a TeX installation (latexmk or Tectonic) suffice for
the current SHA-verified paper build. Tests additionally require pytest; the
experiment kernels require NumPy and, for head/runtime tests, PyTorch.
No model weights, images, per-example predictions or detector traversal are
needed to compile the current paper.

The package retains the existing provisionally pinned CVPR author kit.
Current venue compliance and author submission decisions are separate from
a successful build.
