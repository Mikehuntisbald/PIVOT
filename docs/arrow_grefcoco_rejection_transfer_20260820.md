# ARROW × gRefCOCO rejection transfer

## Outcome

The preregistered result supports the intended conclusion:

> Parameter-isolated confidence consistently improves cross-benchmark rejection ordering, while absolute operating-point calibration remains domain dependent.

This is an annotation/task-zero-shot cross-benchmark evaluation on previously exposed COCO imagery. It is not image-disjoint zero-shot: all 1,500 TestA/B images are byte-identical to Stage-A COCO train2017 inputs.

## Evaluation scope: restricted single/no-target GREC slice

gRefCOCO itself contains **single-target, multi-target, and no-target**
expressions. This experiment does **not** evaluate the full 0-to-N-box GREC
task. It evaluates only the preregistered binary rejection slice:

| TestA + TestB scope | Images | Single-target positive | No-target negative | Total included expressions |
|---|---:|---:|---:|---:|
| Full restricted slice | 1,500 | 11,563 | 9,121 | 20,684 |
| Rejector-supervision-disjoint restricted slice | 1,288 | 9,924 | 7,796 | 17,720 |

All **14,579 multi-target expressions are explicitly excluded**. A
single-target row is positive and a no-target row is negative. Accordingly,
the reported AUROC/AUPR/FPR95 and fixed-threshold rejection results are for the
restricted single/no-target slice, not full gRefCOCO/GREC precision over
arbitrary target-set cardinality. No multi-target result is claimed.

## Frozen protocol

- Official gRefCOCO HF revision: `81eede59b3ac070049f597d023c0ff08d1fb80e9`.
- Included: 11,563 single-target positives and 9,121 no-target negatives from TestA/B.
- Excluded: all 14,579 multi-target expressions; they are never forwarded or scored.
- Models: sealed D3-U50 seeds 17/42/73. There is no gRefCOCO training and no checkpoint, milestone, Gap, or threshold selection using gRefCOCO.
- Forward: admission bypassed; each pass emits B58 base confidence and D3 absolute confidence from the same query universe.
- Fixed thresholds remain the sealed D3 values `0.2301006 / 0.3191232 / 0.7536868` and use raw-score `>=`. Each τ comes strictly from the corresponding seed's sealed 1,570-row D3 calibration result; no gRefCOCO score contributes to τ.
- Bootstrap: 5,000 paired TestA/TestB-stratified **image-cluster** replicates, PCG64 seed `20260820`. Expressions are never treated as IID samples.

The A/B/C preflight verified seed-matched trunk938, rank8, and confidence12 bitwise equality, plus synthetic confidence-only runtime parity. Formal gRef inference therefore loads only the A checkpoint for each seed; B/C are not redundantly evaluated.

## Overlap audit

| Source | TestA/B image overlap |
|---|---:|
| Stage-A COCO train2017 | 1,500 / 1,500 |
| R100 train | 942 / 1,500 |
| D3 train | 182 / 1,500 |
| D3 calibration | 30 / 1,500 |
| D3 train ∪ calibration | 212 / 1,500 |
| FineCops source-image crosswalk | 14 / 1,500 |

“Rejector-supervision-disjoint” has a narrow and explicit meaning: remove
every gRef TestA/B image whose COCO `image_id` appears in either
`d3_proposal_covered_train.jsonl` or
`d3_proposal_covered_calibration.jsonl`. This excludes 212 unique test images:
182 overlapping D3-training images and 30 overlapping D3-calibration images.
The resulting primary surface contains 1,288 images, 9,924 positives, and
7,796 negatives. It is disjoint from **both confidence-owner train and
calibration image sets**.

It is not disjoint from all model development data: Stage-A still overlaps
1,500/1,500 images, and R100 training overlaps 942/1,500. Therefore the
correct claim is rejector-supervision-disjoint sensitivity, not full-model
image-disjoint zero-shot. The Full surface is a co-required robustness gate.
10,479/11,563 positive expressions overlap the sealed RefCOCO TestA/B input
multiset, so the new evidence is primarily no-target rejection rather than
localization.

## Bootstrap and threshold contracts

- The resampling unit is COCO `image_id`, stratified by TestA and TestB. Each
  sampled image carries all of its single-target and no-target expressions.
  The same cluster draw is applied to B58 and all three D3 seeds. There is no
  expression-IID bootstrap.
- In **every bootstrap replicate**, FPR95 is recomputed independently for each
  model/seed: derive that replicate's threshold from its resampled positive
  q05, then evaluate the resampled no-target FPR at that threshold. The
  full-data q05 is never held fixed inside FPR95 bootstrap.
- This replicate-specific q05 is only the definition of the diagnostic FPR95
  metric. It does not replace the deployment threshold τ.
- Fixed-threshold metrics always use the preregistered per-seed τ copied from
  sealed D3 calibration. τ is never re-estimated in a bootstrap replicate and
  was never fit, shifted, or selected on gRefCOCO.
- All three checkpoints, Gap3, candidate routing, and τ values were locked
  before the first gRefCOCO model output. No gRef result was used to change a
  checkpoint, Gap, confidence threshold, or evaluation subset.

## Results

| Surface | Model | AUROC | AUPR | FPR95 | Fixed TPR | Fixed no-target N-acc |
|---|---|---:|---:|---:|---:|---:|
| Full TestAB | B58 | 0.6895 | 0.7201 | 0.7410 | — | — |
| Full TestAB | D3 mean | 0.7175 | 0.7381 | 0.7083 | 0.9010 | 0.3905 |
| Rejector-supervision-disjoint | B58 | 0.6869 | 0.7192 | 0.7497 | — | — |
| Rejector-supervision-disjoint | Isolated rejector mean | 0.7150 | 0.7373 | 0.7116 | 0.8972 | 0.3893 |
| Rejector+FineCops-source-disjoint sensitivity | B58 | 0.6871 | 0.7192 | 0.7488 | — | — |
| Rejector+FineCops-source-disjoint sensitivity | Isolated rejector mean | 0.7151 | 0.7374 | 0.7112 | 0.8964 | 0.3908 |

Paired bootstrap gates:

- Full AUROC gain CI: `[+0.02473, +0.03124]`; FPR95 gain CI: `[+0.02151, +0.04997]`.
- Rejector-supervision-disjoint AUROC gain CI: `[+0.02456, +0.03164]`; FPR95 gain CI: `[+0.02290, +0.05274]`.
- Each of the four one-sided bootstrap p-values: `0.00019996`.
- Rejector-supervision-disjoint fixed-threshold TPR CI: `[0.88390, 0.90982]`, excluding the source operating point `0.95`.

Thus D3 improves threshold-free ordering and domain-normalized FPR95 on both required surfaces, while its frozen source threshold under-accepts gRef positives. The val no-target-only N-acc is `0.4621 / 0.4889 / 0.4832` for seeds 17/42/73; it is descriptive and was never used for selection.

## Zero-training cross-benchmark summary

The following compact figure is derived only from already sealed records; it
adds no training, model selection, threshold fitting, or new model forward.

| Transfer diagnostic | Internal | FineCops-Ref | gRefCOCO |
|---|:---:|:---:|:---:|
| AUROC gain over B58 | + | + | + |
| FPR95 gain over B58 | + | + | + |
| Fixed sealed τ retains ≈95% positive TPR | ✓ | ✗ | ✗ |

Here, Internal ordering uses strict2031: B58→D3 mean AUROC is
`0.8252→0.8478`, and FPR95 is `0.5121→0.4702`. Applying the sealed τ to
strict2031 gives mean positive TPR ≈`95.7%`. FineCops improves AUROC and
FPR95 under its diagnostic per-domain q05, but fixed-τ positive TPR is only
`80.05%`. On gRefCOCO Full, AUROC is `0.6895→0.7175`, FPR95 is
`0.7410→0.7083`, and fixed-τ positive TPR is `90.10%`.

The figure therefore separates two claims: rejection **ordering** transfers
consistently, while the absolute operating point selected by the source-domain
τ does not remain at 95% TPR on either external benchmark.

## Receipts and failure ledger

- Dataset and overlap audit: `/media/haoyi/T9/data/gRefCOCO/v1/manifests/`.
- Preregistration: `outputs/arrow_grefcoco_20260820/preregistration_v2.json`.
- Results/table/final receipt: `outputs/arrow_grefcoco_20260820/`.
- Per-seed records: ignored `evaluations_v2/`; each contains 29,589 rows and peaked at about 10.07 GiB allocated CUDA memory.

The first preregistered launch failed before emitting any record because raw expressions lacked GroundingDINO's terminal phrase delimiter. Its log and receipt are preserved. Version 2 changed only the input wrapper to append ` .`; checkpoints, manifests, metrics, thresholds, bootstrap, and claim gates were unchanged. A separate final-packaging amendment records that the v1 receipt builder assumed canonical filenames and could not bind the preserved v1 and v2 directories simultaneously.
