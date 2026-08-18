# ARROW on FineCops-Ref: external zero-shot results

## Status and claim boundary

The complete FineCops-Ref test was evaluated with the sealed ARROW A/B/C
checkpoints for seeds 17, 42, and 73.  No FineCops train/val row, checkpoint
selection, Gap tuning, alias addition, or confidence-threshold fitting was
used.  This is a FineCops-specific external benchmark zero-shot result, not an
image-disjoint result; the exact exposure caveat is recorded in
`paper_cvpr_arrow_finecops_protocol_20260819.md`.

The final numbers use byte-exact images from the official GQA zip (archive SHA
`02ce5c49c793accd5305356de9c39a50f80a7aaac193b0203de30dbbc65bde62`).
An earlier complete diagnostic used a Hugging Face GQA mirror whose 4,313 JPEGs
had the same dimensions but different encodings.  Those diagnostic results
were viewed before the official-byte correction replay.  The corrected replay
is therefore prospectively frozen relative to its data correction, not a
second virgin holdout.

All 27,926 sample identities passed the frozen-route parity audit:

- B58 boxes and scores are bitwise identical across all nine runs;
- raw-R100 geometry/ranking is bitwise identical across all nine runs;
- D3 confidence is bitwise identical across A/B/C within each seed.

This establishes that the A/B/C differences below are owned by Admission and
not by query geometry, R100, or absolute confidence.

## Main results

| Route | Positive P@1 macro | Positive P@1 micro | Neg-text Recall@1 | Neg-image Recall@1 |
| --- | ---: | ---: | ---: | ---: |
| B58 | 43.11 ± 0.00 | 53.14 ± 0.00 | 42.66 ± 0.00 | 43.98 ± 0.00 |
| R100 + D3 | 43.02 ± 0.00 | 53.09 ± 0.00 | 43.85 ± 0.16 | 44.76 ± 0.17 |
| ARROW-A visual support | 44.16 ± 0.02 | 54.28 ± 0.03 | 45.05 ± 0.11 | 46.14 ± 0.15 |
| ARROW-B category text | 43.15 ± 0.22 | 53.06 ± 0.30 | 43.68 ± 0.24 | 44.51 ± 0.15 |
| ARROW-C learned null | 41.50 ± 0.86 | 51.12 ± 1.24 | 42.16 ± 0.90 | 42.94 ± 1.24 |

The A row is evaluated on the preregistered exact-support surface: 9,182/9,605
positive expressions (95.60%) and only negative pairs for which both arms have
support.  B/C and the baselines use the complete test.  The A full-test micro
lower bound, counting all 423 unsupported positives as wrong, is 51.89%; it
must not be compared to the covered A micro as if both were full-test scores.

## Preregistered matched contrasts

All contrasts use the same A-covered image/pair surface and 5,000 paired parent
GQA-image bootstrap replicates.

| Endpoint | Contrast | Gain | 95% CI | Holm p |
| --- | --- | ---: | ---: | ---: |
| Positive macro P@1 | A−B | +0.462 pp | [+0.115, +0.800] | 0.0036 |
| Positive macro P@1 | B−C | +1.589 pp | [+1.166, +2.054] | 0.0004 |
| Negative-text Recall@1 | A−B | +0.555 pp | [+0.294, +0.822] | 0.0004 |
| Negative-text Recall@1 | B−C | +1.532 pp | [+1.135, +1.945] | 0.0004 |
| Negative-image Recall@1 | A−B | +0.712 pp | [+0.391, +1.031] | 0.0004 |
| Negative-image Recall@1 | B−C | +1.553 pp | [+1.137, +1.979] | 0.0004 |

The external endpoint therefore supports both Admission-input claims:

1. a visual exemplar supplies useful category evidence beyond the category
   name; and
2. an explicit category condition is materially better than a generic learned
   gate.

This is stronger evidence than Ref accuracy alone: C loses category-switch
semantics on the fresh intervention panel and also loses FineCops positive and
negative performance with much larger seed variance.

## Official FineCops comparison

The pinned official evaluator at commit
`31d2c8615e65ccef6a4ff516925ef5ae465ec747` was executed externally; its code
is not vendored.  Against the FineCops paper's zero-shot MM-GDINO-T reference:

| Metric | MM-GDINO-T | ARROW-B full test |
| --- | ---: | ---: |
| Positive P@1 macro | 48.45 | 43.15 |
| Negative-expression Recall@1 | 38.69 | 43.68 |
| Negative-image Recall@1 | 43.14 | 44.51 |
| Negative-expression AUROC type macro | 53.98 | 60.88 |
| Negative-image AUROC type macro | 56.52 | 60.69 |

ARROW does not establish a FineCops positive-grounding SOTA: its full-test
positive macro is 5.30 pp below MM-GDINO-T.  Its advantage is instead in hard
negative ranking and confidence separation.  The support-conditioned A result
cannot be substituted for the full-test B row in this comparison.

## Rejection calibration transfer

The confidence owner behaves exactly as intended structurally: official
AUROC/FPR outputs for R100+D3, B, and C are identical because their D3
confidence tensors and per-record confidence scores are identical.  Admission
changes Recall@1 only through the selected box/localization outcome.  On the
full test, D3 improves the official type-macro AUROC from B58's 58.87/59.55
(text/image) to 60.88/60.69.

The stronger deployment-calibration claim does **not** transfer.  Applying the
sealed D3 thresholds without FineCops tuning yields, averaged over seeds:

- positive TPR: 80.05% for full-test B/C/R100+D3 (80.20% on A-covered rows);
- negative-text FPR: 68.68% (68.90% on A-covered rows);
- negative-image FPR: 64.61% (64.69% on A-covered rows).

Thus the external benchmark validates ownership separation and relative
rejection ranking, but exposes a large absolute calibration/domain-shift gap.
The paper must not describe D3 as a universally calibrated rejector.

## Audit artifacts

- dataset manifest:
  `/media/haoyi/T9/data/FineCops-Ref/v1/manifests/dataset_manifest.json`
  (`8711fb409f2c7bdcb741cc04f0d307642d66f2f781602560c0dd0fb0e3d6e8c5`);
- official GQA zip verification:
  `/media/haoyi/T9/data/FineCops-Ref/v1/manifests/official_gqa_zip_verification.json`
  (4,313/4,313 required members CRC-equal, zero mismatches);
- preregistration:
  `outputs/arrow_finecops_20260819/preregistration.json`
  (official-byte correction replay; see its `correction_replay` binding);
- audited results:
  `outputs/arrow_finecops_20260819/results.json`;
- terminal receipt:
  `outputs/arrow_finecops_20260819/final_receipt.json`.
- relocated HF diagnostic:
  `outputs/arrow_finecops_hf_reencoded_diagnostic_20260819/diagnostic_relocation.json`.

The relocated HF diagnostic preserves the initial engineering amendment, one
first-batch `box_iou` API failure with a zero-byte record file, and the complete
reencoded-image results.  Relative to that diagnostic, official JPEG bytes
change absolute metrics by as much as 0.89 pp and planned contrasts by at most
0.17 pp.  All six contrast directions and significance decisions remain
unchanged, but only the official-byte numbers above are paper results.
