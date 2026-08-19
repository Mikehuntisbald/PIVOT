# ARROW × gRefCOCO rejection transfer

## Outcome

The preregistered result supports the intended conclusion:

> Parameter-isolated confidence consistently improves cross-benchmark rejection ordering, while absolute operating-point calibration remains domain dependent.

This is an annotation/task-zero-shot cross-benchmark evaluation on previously exposed COCO imagery. It is not image-disjoint zero-shot: all 1,500 TestA/B images are byte-identical to Stage-A COCO train2017 inputs.

## Frozen protocol

- Official gRefCOCO HF revision: `81eede59b3ac070049f597d023c0ff08d1fb80e9`.
- Included: 11,563 single-target positives and 9,121 no-target negatives from TestA/B.
- Excluded: 14,579 multi-target expressions.
- Models: sealed D3-U50 seeds 17/42/73; no training, checkpoint selection, Gap tuning, or gRef threshold fitting.
- Forward: admission bypassed; each pass emits B58 base confidence and D3 absolute confidence from the same query universe.
- Fixed thresholds remain the sealed D3 values `0.2301006 / 0.3191232 / 0.7536868` and use raw-score `>=`.
- Bootstrap: 5,000 paired TestA/TestB-stratified image-cluster replicates, PCG64 seed `20260820`; q05/FPR95 is recomputed inside every replicate.

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

The primary D3-disjoint surface contains 1,288 images, 9,924 positives, and 7,796 negatives. The full surface is a co-required robustness gate. 10,479/11,563 positive expressions overlap the sealed RefCOCO TestA/B input multiset, so the new evidence is primarily no-target rejection rather than localization.

## Results

| Surface | Model | AUROC | AUPR | FPR95 | Fixed TPR | Fixed no-target N-acc |
|---|---|---:|---:|---:|---:|---:|
| Full TestAB | B58 | 0.6895 | 0.7201 | 0.7410 | — | — |
| Full TestAB | D3 mean | 0.7175 | 0.7381 | 0.7083 | 0.9010 | 0.3905 |
| D3-disjoint | B58 | 0.6869 | 0.7192 | 0.7497 | — | — |
| D3-disjoint | D3 mean | 0.7150 | 0.7373 | 0.7116 | 0.8972 | 0.3893 |
| D3+FineCops-disjoint | B58 | 0.6871 | 0.7192 | 0.7488 | — | — |
| D3+FineCops-disjoint | D3 mean | 0.7151 | 0.7374 | 0.7112 | 0.8964 | 0.3908 |

Paired bootstrap gates:

- Full AUROC gain CI: `[+0.02473, +0.03124]`; FPR95 gain CI: `[+0.02151, +0.04997]`.
- D3-disjoint AUROC gain CI: `[+0.02456, +0.03164]`; FPR95 gain CI: `[+0.02290, +0.05274]`.
- One-sided bootstrap p-value for all four gains: `0.00019996`.
- D3-disjoint fixed-threshold TPR CI: `[0.88390, 0.90982]`, excluding the source operating point `0.95`.

Thus D3 improves threshold-free ordering and domain-normalized FPR95 on both required surfaces, while its frozen source threshold under-accepts gRef positives. The val no-target-only N-acc is `0.4621 / 0.4889 / 0.4832` for seeds 17/42/73; it is descriptive and was never used for selection.

## Receipts and failure ledger

- Dataset and overlap audit: `/media/haoyi/T9/data/gRefCOCO/v1/manifests/`.
- Preregistration: `outputs/arrow_grefcoco_20260820/preregistration_v2.json`.
- Results/table/final receipt: `outputs/arrow_grefcoco_20260820/`.
- Per-seed records: ignored `evaluations_v2/`; each contains 29,589 rows and peaked at about 10.07 GiB allocated CUDA memory.

The first preregistered launch failed before emitting any record because raw expressions lacked GroundingDINO's terminal phrase delimiter. Its log and receipt are preserved. Version 2 changed only the input wrapper to append ` .`; checkpoints, manifests, metrics, thresholds, bootstrap, and claim gates were unchanged. A separate final-packaging amendment records that the v1 receipt builder assumed canonical filenames and could not bind the preserved v1 and v2 directories simultaneously.
