# Stage B Fixed-Box Full-Text Scorer (v11-v13)

Date: 2026-07-11

## Decision

The selected model is Stage B v12 at iteration 4000:

```text
config:     config/ablations/cfg_stageb_v12_predicate_token_rank.py
dataset:    config/datasets_stageb_v10_aliasfix_synthetic_local_pairs.json
checkpoint: outputs/stageB_v12_predicate_from_v11i3000_to4000_ddp_bs4_20260711/checkpoint_iter.pth
inference:  candidate_topk=50, beta=0
```

This is the best balanced checkpoint in this experiment. It has the best
three-split RefCOCO subset mean, beats v13 on strict annotation-holdout RefCOCO,
and keeps a higher strict target-conditional TN pair win rate than v13. Longer
v12 training and the v13 calibration probe move individual TN metrics, but lose
grounding accuracy or target pair ordering.

This result does **not** exceed the pure GroundingDINO Stage-B data fine-tuning
baseline. On the conventional full validation splits, v12 reaches mean acc50
`0.569251`, versus `0.714429` for the recorded pure GDINO baseline. Its full
historical TN FPR@95 is also much worse (`0.907556` versus `0.528302`; lower is
better).

## Diagnosis of the Earlier Separation Attempts

The Stage-A observation is important but narrower than it first appears. High
recall under an `object` or wrong-text prompt shows that the patch/entity branch
can retain the target in a candidate pool. It does not show that the current
top-1 score can identify the correct candidate or reject an edited expression.
The remaining task is candidate-conditional text ranking.

The v7-v10 post-candidate verifier underperformed for several concrete reasons:

1. The verifier was not a full multimodal re-scoring path. It combined a
   detached decoder query, 1x1 ROIAlign features for the box and enlarged
   context, box coordinates, and pooled text in shallow residual/MLP heads.
   The 1x1 pooling discarded the spatial layout needed for relations such as
   left/right, front/back, and above/below. It also lacked iterative token-image
   cross-attention on the candidate.
2. Phrase scoring averaged token probabilities. When a positive and TN phrase
   differed by one token, shared object words dominated the average and diluted
   the only discriminative token.
3. The frozen Stage-A tower and copied frozen BERT were previously left in
   training mode. Swin DropPath, fusion DropPath, and BERT dropout therefore
   changed candidates or text features across forwards. A positive/TN rank loss
   could compare two different stochastic candidate computations. v10 fixed the
   module modes and made this comparison deterministic.
4. Data plumbing weakened pair supervision. The one-patch paired loader did not
   always expose a pair stride of two; a missing canonical word could invalidate
   an otherwise usable positive phrase; canonical aliases could take priority
   over another class's exact canonical name; and supervision could miss a valid
   high-IoU candidate by relying too narrowly on one match. These were corrected
   before v11 with explicit pair slots, zero canonical masks when appropriate,
   exact canonical/synonym priority, and multi-positive IoU supervision.
5. The old global TN interpretation was not label-safe. An edited local phrase
   can legitimately describe a different same-class object in the image. Such a
   row is false only for the original target, not necessarily for the full image.
   Target-conditional scores on the same best-IoU candidate are therefore the
   primary TN metric; global FPR is retained only for compatibility.
6. Earlier AMP runs could silently make non-finite gradients or skip optimizer
   steps. The current path checks finite gradients, monitors the scaler, and
   aborts after repeated skips.

These fixes helped v10, but its 1x1-ROI and phrase-mean architecture remained a
capacity bottleneck. v11 replaces that architecture rather than adding another
calibration term to it.

## v11: Immutable Candidates, Full-Text Scoring

The v11 boundary is explicit:

```text
frozen Stage A (canonical class + one support patch)
    -> patch-only Top-50 query/box candidates
    -> detach query states and boxes
    -> encode each full expression independently
    -> expression-conditioned fusion encoder in no-grad mode
    -> trainable three-layer fixed-box scoring decoder
    -> full-phrase text score for each unchanged candidate
```

Stage A still produces 900 queries and boxes. Patch score alone admits the
Top-50 candidates. Candidate indices, query states, and boxes are detached, and
the scorer has no bbox head, patch head, private BERT, or box-update path. The
same candidate tensor is used for the positive expression and its paired TN.
Assertions and tests cover candidate/box identity across the two slots.

The scorer copies the final three GroundingDINO decoder layers. Those layers
retain query self-attention, full-text cross-attention, deformable image
cross-attention, and FFNs, but `bbox_embed` and `class_embed` are removed from
the decoder. Each expression reuses the frozen base BERT and reruns the frozen
text enhancer/fusion encoder; only the fixed-box scoring decoder is trainable.
The trainable parameter count is 5,626,240.

The v11 loss is candidate-conditional:

```text
positive candidate: IoU >= 0.5
negative candidate: IoU <= 0.3

0.2 * multi-positive listwise loss, temperature 0.2
1.0 * softplus(score_tn - score_pos + 0.3) on the same IoU-positive query
0.5 * absolute anchor loss, with positive logit +0.5 and TN logit -0.5
0.25 * batch tail separation, positive q05 versus TN q95, margin 0.3
```

Image-global TN losses are disabled because the training edits are local
counterfactuals and do not carry an image-global-negative guarantee. No patch,
bbox, or GIoU loss can move localization. The scorer uses `lr=2e-5`; sampling
offsets and reference-point projections use the repository's absolute
`2e-6` projection learning rate. A controlled `1e-5` geometry-LR probe reduced
Ref accuracy and did not improve spatial TN separation, so `2e-6` was retained.

Training used two GPUs, per-rank batch size 4, expression microbatch size 8,
AMP initial scale 512, and a single support patch. v11 was initialized from:

```text
outputs/stageA_v3_all_gt_lvis_neg02_lam02_from0006_ddp_bs16/checkpoint0004.pth
```

The Stage-A checkpoint must be loaded with `--pretrain_model_path` so the new
scorer starts with a fresh optimizer. Later v11/v12/v13 continuations use
`--resume` because their scorer state and optimizer groups are compatible.

## v12: Changed-Token Auxiliary Rank Loss

v11 still deployed a full-phrase probability mean. That is desirable at
inference, but it lets one edited word be overwhelmed during training. v12 adds
a training-only changed-token auxiliary objective without changing inference or
adding parameters.

Positive and TN captions are tokenized independently through the same right-
padded BERT tokenizer. `SequenceMatcher` on token IDs identifies replacement,
insertion, and deletion spans. Special tokens, punctuation-only tokens, and
padding are excluded. Shared words, including the object head, receive no
predicate-mask supervision. A pair is admitted to the symmetric auxiliary loss
only when both sides contain an eligible changed token; insertion/deletion-only
pairs retain the full-phrase v11 losses.

For every IoU-positive fixed candidate, v12 adds:

```text
1.0 * softplus(changed_token_score_tn - changed_token_score_pos + 0.3)
```

The deployed score is still the independently computed full-expression score;
no positive/TN pair or edit mask is required at inference. The mask coverage
audit found valid symmetric changed-token masks for 100% of synthetic pairs,
99.67% of verified RefCOCOg pairs, and 99.58% of verified RefCOCO+ pairs.

## v13: Calibration Probe

v13 resumes v12@4000 for 500 steps and changes only two loss aggregation
details:

```text
stage_b_v11_balance_local_anchor_classes = True
stage_b_v11_batch_tail_ddp_global = True
```

Positive and TN absolute anchors are averaged as two equally weighted classes,
independent of the clean/pair sampling ratio. The q05/q95 tail loss gathers a
fixed score/mask payload across DDP ranks, so it operates on the global batch of
eight rather than each rank's batch of four.

This calibration improves several tail and spatial metrics. On the UMD strict
TN set, global FPR@95 improves from `0.908323` to `0.900621`, and spatial target
pair win improves from `0.526718` to `0.543893`. However, overall target pair
win falls from `0.712298` to `0.702857`, and strict Ref mean acc50 falls from
`0.554802` to `0.548925`. v13 is therefore an informative calibration ablation,
not the selected model.

## Training Data and Leakage Audit

The final training mixture is:

| source | rows | mix weight | effective mixture |
|---|---:|---:|---:|
| RefCOCO+ clean train expressions | 120,191 | 1 | 14.3% |
| RefCOCOg clean train expressions | 80,512 | 1 | 14.3% |
| RefCOCO+ verified positive/TN pairs | 25,785 | 2 | 28.6% |
| RefCOCOg verified positive/TN pairs | 21,055 | 2 | 28.6% |
| rule-generated local pairs | 179,347 | 1 | 14.3% |

The rule-generated set contains 95,224 color, 8,997 size, and 75,126 spatial
edits. Verified pair datasets use capped inverse-square-root category balancing.
The positive prebuilt files were regenerated after fixing canonical lookup:
exact canonical names and synonyms now have priority, while broad aliases only
fill names not already claimed. This avoids assigning a COCO category through
an alias collision.

The five exact training JSONLs yield 64,549 unique `(image_id, ann_id)` keys
over 28,158 images. Strict evaluation uses `--holdout_level ann` and excludes
every matching target annotation from evaluation. The resulting evaluation
sizes are:

| protocol/split | expressions |
|---|---:|
| conventional RefCOCO val | 10,834 |
| strict ann-holdout RefCOCO val | 5,947 |
| conventional RefCOCO+ val | 10,758 |
| strict ann-holdout RefCOCO+ val | 5,857 |
| conventional RefCOCOg UMD val | 4,896 |
| strict ann-holdout RefCOCOg UMD val | 2,603 |

The conventional and strict RefCOCOg rows use the same UMD validation split;
the strict annotation filter reduces it from 4,896 to 2,603 expressions.

This is an annotation-level, not image-level, holdout. A strict row cannot use
the same `(image_id, ann_id)` target seen in training, but another annotation
from the same image can remain. Consequently, the strict tables support a claim
of no exact target-annotation overlap, not a claim of image-disjoint evaluation.
The 1,008-expression screening subsets are conventional, non-holdout subsets and
may contain training target annotations; they are controlled checkpoint A/B
signals, not clean generalization estimates. The conventional full table is
reported for compatibility with the historical pure-GDINO benchmark. The final
v12-versus-v13 choice is also checked on the strict sets below.

Two TN strict variants are reported:

```text
historical strict: RefCOCO+ val 2504 + RefCOCOg Google val 440 = 2944 pairs
UMD-val strict:    RefCOCO+ val 2504 + RefCOCOg UMD val 1521 = 4025 pairs
```

The non-holdout historical full TN set contains 9,833 pairs and is also reported
only for benchmark compatibility.

## Metric Interpretation

Ref `acc50` is top-1 IoU >= 0.5. The subset tables use 1,008 expressions per
split and identical seeds/checkpoints. The conventional and strict tables use
all available expressions.

For TN evaluation, lower FPR@95 is better; higher pair win and score gap are
better. `global` lets each expression choose its own top-scoring candidate.
`target` evaluates both expressions on the same best-IoU target candidate. The
target metrics are primary for local edits because the edited phrase may be
valid for another object in the image.

## Controlled Subset Results

### RefCOCO-family acc50

| model | iter | RefCOCO | RefCOCO+ | RefCOCOg | mean |
|---|---:|---:|---:|---:|---:|
| v11 fixed full text | 3000 | 0.559524 | 0.559524 | 0.629960 | 0.583003 |
| v12 changed token | 3500 | 0.536706 | 0.566468 | 0.638889 | 0.580688 |
| **v12 changed token** | **4000** | **0.560516** | **0.570437** | **0.643849** | **0.591601** |
| v12 changed token | 4500 | 0.562500 | 0.551587 | 0.637897 | 0.583995 |
| v12 changed token | 5000 | 0.552579 | 0.555556 | 0.641865 | 0.583333 |
| v13 calibrated tail | 4500 | 0.562500 | 0.547619 | 0.626984 | 0.579034 |

### Historical TN subset, 1,200 pairs

| model | iter | global FPR95 | target FPR95 | target win | target gap | spatial target win | spatial target gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| v11 | 3000 | 0.913333 | 0.917500 | 0.673333 | 0.022999 | 0.554286 | 0.000269 |
| v12 | 3500 | 0.920000 | 0.905833 | 0.683333 | 0.040372 | 0.554286 | 0.003726 |
| **v12** | **4000** | **0.912500** | **0.920833** | **0.687500** | **0.039345** | **0.560000** | **0.002539** |
| v12 | 4500 | 0.919167 | 0.909167 | 0.675833 | 0.043321 | 0.542857 | 0.002844 |
| v12 | 5000 | 0.907500 | 0.904167 | 0.695833 | 0.043480 | 0.514286 | 0.001598 |
| v13 | 4500 | 0.920833 | 0.911667 | 0.682500 | 0.042280 | 0.577143 | 0.005124 |

The later v12 checkpoints improve some aggregate TN values but lose about 0.8
points of mean Ref acc50 and sharply reduce spatial pair ordering at iteration
5000. The v13 calibration restores some spatial separation but does not recover
Ref or overall target pair win. This is why the selection is v12@4000 rather
than the numerically best single TN column.

## Strict Annotation-Holdout Results

### Full RefCOCO-family acc50

| model | RefCOCO (5947) | RefCOCO+ (5857) | RefCOCOg UMD (2603) | mean |
|---|---:|---:|---:|---:|
| **v12@4000** | **0.491508** | **0.511354** | **0.661544** | **0.554802** |
| v13@4500 | 0.486127 | 0.504866 | 0.655782 | 0.548925 |

### TN strict sets

| set/model | pairs | global FPR95 | target FPR95 | target win | target gap | spatial target win | spatial target gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| historical strict, **v12@4000** | 2944 | 0.907948 | **0.892663** | **0.719769** | 0.052374 | 0.550000 | -0.000371 |
| historical strict, v13@4500 | 2944 | **0.906250** | 0.894361 | 0.705163 | **0.055937** | **0.561538** | **0.003661** |
| UMD strict, **v12@4000** | 4025 | 0.908323 | 0.894658 | **0.712298** | 0.050173 | 0.526718 | 0.004181 |
| UMD strict, v13@4500 | 4025 | **0.900621** | **0.889689** | 0.702857 | **0.053502** | **0.543893** | **0.006209** |

v13 shifts calibration in a real but incomplete way: FPR and spatial gap often
improve, while target pair win and Ref accuracy regress. Spatial language remains
the weakest category, so the result does not justify replacing v12@4000.

## Final Conventional Full Results

### RefCOCO-family

| model | RefCOCO | RefCOCO+ | RefCOCOg UMD | mean |
|---|---:|---:|---:|---:|
| **v12@4000** | **0.526491** | **0.545640** | **0.635621** | **0.569251** |
| pure GDINO Stage-B FT `ckpt0001` | 0.645468 | 0.690835 | 0.806985 | 0.714429 |
| v12 minus pure GDINO | -0.118977 | -0.145195 | -0.171364 | -0.145178 |

The v12 full evaluation covers 26,488 expressions. Recall50@5 is `0.890991`,
`0.887897`, and `0.905025`; Recall50@10 is `0.955326`, `0.956126`, and
`0.957925` on the three splits, respectively. The remaining top-1 gap is
consistent with a ranking problem, but the pure GDINO comparison shows that
this scorer has not solved it.

### Historical TN, no holdout

| model/score | pairs | FPR95 | FPR90 | pair win | score gap |
|---|---:|---:|---:|---:|---:|
| v12@4000 global | 9833 | 0.907556 | 0.829147 | 0.649751 | 0.031153 |
| v12@4000 target | 9833 | 0.905319 | 0.823248 | 0.706397 | 0.045860 |
| pure GDINO Stage-B FT global | historical set | 0.528302 | 0.450905 | 0.825760 | not recorded |

The v12 target breakdown confirms the residual weakness:

| category | pairs | target FPR95 | target win | target gap |
|---|---:|---:|---:|---:|
| color | 7495 | 0.892595 | 0.746364 | 0.056462 |
| size | 758 | 0.926121 | 0.620053 | 0.030198 |
| spatial | 1580 | 0.950633 | 0.558228 | 0.003082 |

The fixed-box separation is therefore validated as an engineering boundary and
v12 improves candidate-conditional text learning, but the absolute TN score
overlap and spatial-relation scoring remain far behind the pure-GDINO result.

## Reproduction and Evidence Paths

Implementation and configuration:

```text
models/GroundingDINO/stage_b_fixed_text_scorer.py
models/GroundingDINO/stage_b_fixed_text_criterion.py
models/GroundingDINO/transformer.py
models/GroundingDINO/groundingdino.py
config/ablations/cfg_stageb_v11_fixed_text_scorer.py
config/ablations/cfg_stageb_v12_predicate_token_rank.py
config/ablations/cfg_stageb_v13_calibrated_tail.py
config/datasets_stageb_v10_aliasfix_synthetic_local_pairs.json
```

Training lineage:

```text
outputs/stageB_v11_fixed_text_from_stageA0004_ddp_bs4_probe250_20260711
outputs/stageB_v11_fixed_text_from250_to1000_ddp_bs4_20260711
outputs/stageB_v11_fixed_text_from1000_to3000_ddp_bs4_20260711
outputs/stageB_v12_predicate_from_v11i3000_to4000_ddp_bs4_20260711
outputs/stageB_v12_predicate_from_i4000_to5000_ddp_bs4_20260711
outputs/stageB_v13_calibrated_from_v12i4000_to4500_ddp_bs4_20260711
```

Subset summaries:

```text
outputs/stageB_v11_i3000_k50_ref3000_20260711/summary.json
outputs/stageB_v11_i3000_k50_tn1200_20260711/summary.json
outputs/stageB_v12_i3500_i4000_k50_ref3000_20260711/summary.json
outputs/stageB_v12_i3500_i4000_k50_tn1200_20260711/summary.json
outputs/stageB_v12_i4500_i5000_k50_ref3000_20260711/summary.json
outputs/stageB_v12_i4500_i5000_k50_tn1200_20260711/summary.json
outputs/stageB_v13_i4500_k50_ref3000_20260711/summary.json
outputs/stageB_v13_i4500_k50_tn1200_20260711/summary.json
```

Strict holdout summaries:

```text
outputs/stageB_v12_i4000_k50_ref_strict_ann_full_20260711/summary.json
outputs/stageB_v13_i4500_k50_ref_strict_ann_full_20260711/summary.json
outputs/stageB_v12_i4000_k50_tn_strict_ann_full_20260711/summary.json
outputs/stageB_v13_i4500_k50_tn_strict_ann_full_20260711/summary.json
outputs/stageB_v12_i4000_k50_tn_umdval_strict_ann_20260711/summary.json
outputs/stageB_v13_i4500_k50_tn_umdval_strict_ann_20260711/summary.json
```

Final conventional summaries:

```text
outputs/stageB_v12_i4000_k50_ref_full_conventional_20260711/summary.json
outputs/stageB_v12_i4000_k50_tn_full_historical_20260711/summary.json
```

Recorded pure-GDINO baseline evidence:

```text
outputs/text_gdino_ft_stageb_with_tn_ckpt0001_tn_eval/summary.md
outputs/text_gdino_ft_stageb_with_tn_ckpt0001_refcoco_series_eval/summary.md
```

For strict reproduction, pass all five training JSONLs from the dataset config
to `--exclude_train_jsonl` and set `--holdout_level ann`. All v11-v13 evaluations
must also use `--stage_b_v11_candidate_topk 50 --betas 0`; changing either value
is a different scoring protocol.
