# ARROW on FineCops-Ref: external zero-shot results

## Status and claim boundary

The complete FineCops-Ref test was evaluated with the sealed ARROW A/B/C
checkpoints for seeds 17, 42, and 73.  No FineCops train/val row, checkpoint
selection, Gap tuning, alias addition, or confidence-threshold fitting was
used.  This is a FineCops-specific external benchmark zero-shot result, not an
image-disjoint result; the exact exposure caveat is recorded in
`paper_cvpr_arrow_finecops_protocol_20260819.md`.

All 27,926 sample identities passed the frozen-route parity audit:

- B58 boxes and scores are bitwise identical across all nine runs;
- raw-R100 geometry/ranking is bitwise identical across all nine runs;
- D3 confidence is bitwise identical across A/B/C within each seed.

This establishes that the A/B/C differences below are owned by Admission and
not by query geometry, R100, or absolute confidence.

## Main results

| Route | Positive P@1 macro | Positive P@1 micro | Neg-text Recall@1 | Neg-image Recall@1 |
| --- | ---: | ---: | ---: | ---: |
| B58 | 43.35 ± 0.00 | 52.91 ± 0.00 | 42.12 ± 0.00 | 43.29 ± 0.00 |
| R100 + D3 | 43.23 ± 0.00 | 52.74 ± 0.00 | 43.13 ± 0.20 | 44.71 ± 0.16 |
| ARROW-A visual support | 44.48 ± 0.08 | 53.84 ± 0.03 | 44.17 ± 0.16 | 46.00 ± 0.18 |
| ARROW-B category text | 43.31 ± 0.26 | 52.58 ± 0.29 | 42.89 ± 0.31 | 44.30 ± 0.43 |
| ARROW-C learned null | 41.67 ± 0.98 | 50.63 ± 1.29 | 41.27 ± 0.92 | 42.60 ± 1.48 |

The A row is evaluated on the preregistered exact-support surface: 9,182/9,605
positive expressions (95.60%) and only negative pairs for which both arms have
support.  B/C and the baselines use the complete test.  The A full-test micro
lower bound, counting all 423 unsupported positives as wrong, is 51.47%; it
must not be compared to the covered A micro as if both were full-test scores.

## Preregistered matched contrasts

All contrasts use the same A-covered image/pair surface and 5,000 paired parent
GQA-image bootstrap replicates.

| Endpoint | Contrast | Gain | 95% CI | Holm p |
| --- | --- | ---: | ---: | ---: |
| Positive macro P@1 | A−B | +0.563 pp | [+0.199, +0.930] | 0.0020 |
| Positive macro P@1 | B−C | +1.608 pp | [+1.176, +2.071] | 0.0004 |
| Negative-text Recall@1 | A−B | +0.573 pp | [+0.297, +0.861] | 0.0004 |
| Negative-text Recall@1 | B−C | +1.630 pp | [+1.236, +2.042] | 0.0004 |
| Negative-image Recall@1 | A−B | +0.774 pp | [+0.449, +1.113] | 0.0004 |
| Negative-image Recall@1 | B−C | +1.726 pp | [+1.316, +2.152] | 0.0004 |

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
| Positive P@1 macro | 48.45 | 43.31 |
| Negative-expression Recall@1 | 38.69 | 42.89 |
| Negative-image Recall@1 | 43.14 | 44.30 |
| Negative-expression AUROC type macro | 53.98 | 59.77 |
| Negative-image AUROC type macro | 56.52 | 59.95 |

ARROW does not establish a FineCops positive-grounding SOTA: its full-test
positive macro is 5.14 pp below MM-GDINO-T.  Its advantage is instead in hard
negative ranking and confidence separation.  The support-conditioned A result
cannot be substituted for the full-test B row in this comparison.

## Rejection calibration transfer

The confidence owner behaves exactly as intended structurally: official
AUROC/FPR outputs for R100+D3, B, and C are identical because their D3
confidence tensors and per-record confidence scores are identical.  Admission
changes Recall@1 only through the selected box/localization outcome.  On the
full test, D3 improves the official type-macro AUROC from B58's 57.92/58.56
(text/image) to 59.77/59.95.

The stronger deployment-calibration claim does **not** transfer.  Applying the
sealed D3 thresholds without FineCops tuning yields, averaged over seeds:

- positive TPR: 79.63% for full-test B/C/R100+D3 (79.87% on A-covered rows);
- negative-text FPR: 69.21% (69.53% on A-covered rows);
- negative-image FPR: 64.61% (64.69% on A-covered rows).

Thus the external benchmark validates ownership separation and relative
rejection ranking, but exposes a large absolute calibration/domain-shift gap.
The paper must not describe D3 as a universally calibrated rejector.

## Audit artifacts

- dataset manifest:
  `/media/haoyi/T9/data/FineCops-Ref/v1/manifests/dataset_manifest.json`
  (`9b0aaaa22321a1f1361e2e295ab52f4991a3fa8c808bdde927b1cc070b04e605`);
- preregistration:
  `outputs/arrow_finecops_20260819/preregistration.json`
  (`668dc6813e33abedfada50bf420d90d3543075c2e17d56b673814eb4bd9ee631`);
- engineering amendment:
  `outputs/arrow_finecops_20260819/preregistration_amendment.json`
  (`0f93701104dc1799f617c05848bbb35a4251cb403241416f0e59c9dc83c00573`);
- audited results:
  `outputs/arrow_finecops_20260819/results.json`;
- terminal receipt:
  `outputs/arrow_finecops_20260819/final_receipt.json`.

The amendment records one first-batch engineering failure caused solely by the
repository `box_iou` tuple API.  The failed temporary record file is zero bytes;
no metric or prediction was viewed before the compatibility-only fix.
