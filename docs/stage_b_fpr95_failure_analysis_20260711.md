# Stage-B FPR@95TPR Failure Analysis

Date: 2026-07-11

## Decision

The earlier separated scorers did not fail because their numeric scores were
too large. They failed because the negative global maxima remained ahead of the
lowest 5% of positive global maxima. A common threshold, temperature, affine
shift, or any other shared strictly increasing calibration cannot change that
ordering and therefore cannot change FPR@95TPR.

The new pure-GDINO adapter should retain the following boundary:

```text
rank_score       = frozen_base_score + candidate_specific_rank_residual
confidence_score = frozen_base_score + image_expression_gate
```

The rank and confidence branches must remain functionally independent. The
confidence loss must operate on the maximum over all 900 deployed queries, not
on the target query or a proposal subset. The remaining high-risk mismatch is
the label/data scope: the recovered 60k pairs reproduce the fixed data-FT
`allTN` labels, but they are not image-global verified negatives.

The historical pure-GDINO number, global FPR95 `0.528302`, was measured on the
historical TN set. It has no surviving per-example records and is not a
same-manifest comparison with the strict2031 results below. The rebuilt fixed
baseline must be evaluated on the exact same strict2031 and strict1607 records
before it can be used as the acceptance threshold.

## What The Existing Strict Records Show

All rows below use the same 2,026 valid pairs from the 2,031-row strict
annotation-holdout manifest. The old v5.2 row is not the pure-GDINO data-FT
baseline; it is only the common initialization for these legacy gate probes.

| model | global FPR95 | global pair win | global AUC | target-query FPR95 | positive q05 threshold | TN median |
|---|---:|---:|---:|---:|---:|---:|
| old v5.2 base | 0.782823 | 0.764561 | 0.689300 | 0.977295 | 0.330808 | 0.449701 |
| legacy CVar, 500 iter | **0.731984** | 0.765548 | **0.722671** | 0.962488 | -0.067275 | -0.019895 |
| proposal-proxy FPR95, 500 iter | 0.755676 | **0.772458** | 0.716711 | **0.851431** | -0.275347 | -0.057870 |

This gives four concrete failure signatures:

1. Score shrinkage is not rejection. CVar moves both distributions close to
   zero, but FPR95 remains `0.731984`.
2. Pair ordering is not the operating metric. The proxy probe has the best
   pair win, but worse FPR95 than CVar.
3. The positive lower tail controls the result. The proxy probe lowers the TN
   median relative to CVar, but its positive q05 collapses by another `0.2081`;
   the operating threshold follows that collapse and FPR gets worse.
4. Target/proposal improvement does not imply deployed improvement. The proxy
   objective improves target-query FPR95 by `0.1259` absolute versus the base,
   while global FPR95 improves by only `0.0271`.

The legacy gate is a uniform query offset, so it cannot change within-image
query order. Consistent with that, all three runs have identical positive and
TN top-1 IoU50 (`0.787759` and `0.511846`). Their global-minus-target score-gap
quantiles are also identical; for TN rows the median gap is `0.150814` and the
q90 gap is `0.600383`. This is direct evidence that a target-local loss can be
overwhelmed by a different high-scoring query at deployment.

## Training-Scope Shift

The recovered fixed data-FT pair file contains 60,000 exact positive/TN pairs,
but its support is narrow:

| property | 60k data-FT pairs | strict2031 manifest |
|---|---:|---:|
| source | RefCOCO UNC train only | RefCOCO+ UNC val + RefCOCOg UMD val |
| spatial | 42,581 (71.0%) | 50 rows (2.5%) |
| color | 16,048 (26.7%) | 1,056 rows (52.0%) |
| size | 1,371 (2.3%) | 92 rows (4.5%) |
| other edit taxonomies | 0 | 855/2,026 valid rows (42.2%) |
| mean positive words | 3.84 | 5.15 |
| unique positive/TN text pairs | 34,805/60,000 | not applicable |
| image overlap with strict2031 | 88 rows on 67 images | 2,031 rows |
| exact annotation overlap | 0 | 2,031 rows |

The 60k labels explicitly carry `global_tn_verified=false` and
`tn_scope=benchmark_dataft_alltn`. They are fair reproductions of what the
fixed baseline was trained to treat as all-negative, but they do not prove that
the edited phrase is false everywhere in the image.

Two independent input-pipeline hazards are now fail-closed in the adapter
recipes. First, adapter training sets `data_aug_hflip_prob=0`. The existing
horizontal-flip transform changes the image and boxes but does not rewrite the
caption, so applying it to phrases containing left/right would corrupt their
semantics. Second, confidence pair datasets set `neg_episode_prob=0`
explicitly. This prevents `PatchEpisodeDataset` from internally replacing an
audited image/text pair with a different negative episode. These safeguards
preserve the pair labels; they do not resolve the remaining difference between
`benchmark_dataft_alltn` and image-global verified negatives.

The legacy strict results already show the domain effect. At each model's
single global operating threshold:

| model | FPR on exact color/spatial/size rows | FPR on other taxonomies |
|---|---:|---:|
| old v5.2 base | 0.736977 | 0.845614 |
| legacy CVar | **0.663535** | **0.825731** |
| proposal-proxy FPR95 | 0.674637 | 0.866667 |

The proxy objective improves its familiar edit families and regresses the
other 42.2% of strict rows. This is the most likely reason a pair-trained gate
can look healthy on its training scope but fail the final global FPR gate.

## Objective Alignment Audit

The current adapter's confidence loss has the right deployed score shape:

```text
positive_global = max over all 900 confidence scores
negative_global = max over all 900 confidence scores
metric threshold = exact positive q05 order statistic, score >= threshold
```

Therefore the old proposal-set-versus-900-query mismatch must not be attributed
to the new loss unless runtime assertions show `Q != 900` or a candidate mask
restricts the max. The new mismatch is primarily label scope and distribution.

The superseded P2 objective had an objective-estimation risk. With two DDP
ranks and four positives per rank, the exact 95%-TPR threshold on the global
batch of eight is the global batch minimum:

```text
k = B - ceil(0.95 * B) + 1 = 1, for B = 8
```

P2's larger queue made the forward threshold closer to a dataset q05, but its
straight-through gradient still came from one selected global-batch minimum. A
sample minimum has expected population quantile `1/(B+1)`, about q11 for
`B=8`, not q05. Its 8,192-entry queue also could not roll over during a
500-step, global-batch-8 probe, which supplies only 4,000 positive scores; early
history therefore remained in the bank for the whole probe. The proxy probe's
very low positive tail is a concrete warning against that estimator.

P3 replaces those P2 mechanics with `detached_recent_q05_trust`:

```text
t = q05(recent positive queue of capacity 512).detach()
warmup = 256 valid positive scores
t_loss = t + mean(positive_gate) - detach(mean(positive_gate))
Lneg   = mean softplus((negative_global - t_loss + margin) / temperature)
Lpair  = mean softplus((negative_global - positive_global + pair_margin) / temperature)
Ltrust = mean(relu(-0.02 - positive_gate))
Ltotal = Lneg + 0.25 * Lpair + 1.0 * Ltrust
```

The 512-entry queue retains the most recent 64 global-batch-8 steps and becomes
the forward threshold source at 256 scores; the pre-warmup current q05 is also
detached. The added term in `t_loss` is exactly zero in value. Its gradient
makes the threshold follow a common positive-gate translation, cancelling the
otherwise attractive shared downward shift of both positive and negative
scores in `Lneg`. `Ltrust` is a linear hinge, not a squared penalty, and guards
positive gates below `-0.02`. P3 therefore keeps every current TN in the loss
without sending a noisy batch-min gradient through the threshold or treating a
common score translation as useful separation.

These are implemented and audited safeguards replacing the old P2 risks. They
do not establish that the rebuilt fixed-baseline FPR or all RefCOCO-family AP
requirements have been beaten; only the unchanged strict2031, strict1607, and
eight-split formal evaluations can establish that.

## Minimum Probe Matrix

Use one fixed rank checkpoint, identical sample order, and checkpoints at 0,
50, 100, 250, and 500 optimizer steps. Confidence probes must never update the
rank branch or frozen GDINO.

| probe | confidence objective | question answered | early acceptance |
|---|---|---|---|
| P0 identity | no training | Is adapter/evaluator parity exact? | base, rank, confidence scores bitwise equal; identical FPR/AP records |
| P1 pair-only | aligned 900-query pair margin | Does pair ordering alone transfer? | held-out pair win and FPR improve; strict FPR must also move |
| P2 superseded control | 8,192 queue q05 ST + every-TN + pair margin | Did global-batch-min ST gradient and stale history cause false progress? | diagnostic only; do not promote over P3 |
| P3 current | detached recent q05 (512/256) + every-TN + pair margin + translation proxy + linear positive trust | Do the corrected gradients transfer beyond train scope? | require strict2031 improvement and the same direction on strict1607 without positive-q05 collapse |

For speed, first evaluate a fixed diagnostic subset, but promote a probe only
after exact full strict2031 evaluation. Also report the strict1607 semantic
image-disjoint subset in the same direction. Do not choose a checkpoint by
pair win, BCE, score magnitude, or a tuned fixed threshold.

Every probe report must include:

- exact global FPR95 and its positive q05 threshold;
- positive/TN global-score quantiles q01/q05/q50/q95/q99;
- gate quantiles separately for positives and TNs;
- pair win and AUC as diagnostics, not selection metrics;
- RefCOCO+ and RefCOCOg split FPR;
- color, spatial, size, and other-taxonomy FPR at the one global threshold;
- `max over 900`, target-query, and proposal-subset FPR side by side;
- confidence/rank branch gradient norms and bitwise branch-isolation checks.

For a 250/500-step probe, require at least `0.02` absolute strict2031 FPR
improvement and a paired-bootstrap 95% upper bound below zero before spending a
full run. This is only a resource gate. The final acceptance remains strict:
global FPR95 lower than the rebuilt fixed data-FT baseline and RefCOCO-family
AP/acc50 higher under identical records and manifests.

## Evidence

```text
outputs/stageB_v5_legacy_gate_cvar500_strict2031_correctsplit_tn_20260711/summary.md
outputs/stageB_v5_legacy_gate_cvar500_strict2031_correctsplit_tn_20260711/per_example_records/
outputs/stageB_v5_legacy_gate_fpr95_proxy500_strict2031_tn_20260711/summary.md
outputs/stageB_v5_legacy_gate_fpr95_proxy500_strict2031_tn_20260711/per_example_records/
data/ablations/stageb_gdino_adapter_dataft_20260711/audit.json
data/ablations/stageb_gdino_adapter_dataft_20260711/benchmark_dataft_alltn_pairs.jsonl
data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/audit.json
data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/eval_manifest.jsonl
docs/stage_b_fixed_text_scorer_v11_v13.md
```
