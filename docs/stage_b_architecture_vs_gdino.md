# Stage-B Architecture Relative to GroundingDINO

Date: 2026-07-17

> Historical record: this document describes the 2026-07-16 multi-checkpoint
> P750/R100/P50 winner. The current single-network dense-duty development line,
> including the V55 negative result and V56 preregistration, is documented in
> [`paper_cvpr_stage_b_development_20260802.md`](paper_cvpr_stage_b_development_20260802.md).

## Status

This document is the current architecture and experiment record for the final
Stage-B composite evaluated on 2026-07-16. It supersedes the implementation
recommendations in the paused
[`stage_b_decoupled_scoring_handoff_20260716.md`](stage_b_decoupled_scoring_handoff_20260716.md),
but does not rewrite that historical handoff.

The recorded run met the point-estimate acceptance rule against the same
GroundingDINO Stage-B data-FT checkpoint:

- lower exact FPR@95TPR on both strict2031 and strict1607; and
- higher top-1 IoU >= 0.5 accuracy on all eight RefCOCO-family splits.

In this repository, the Ref metric is `acc50`: whether the single selected box
has IoU >= 0.5. It is not COCO-style AP. References to "Ref AP" in older notes
must be read as this top-1 `acc50` protocol unless an evaluator explicitly says
otherwise.

There is an important durability qualification. The final checkpoints,
per-example records, route artifacts, and gate reports were written under
`/tmp` during the 2026-07-16 run. They are no longer present on 2026-07-17, and
`outputs/` is empty. The numbers below are the recorded result of that run, not
a currently replayable release artifact. The model must be re-materialized and
the sealed records promoted to durable storage before release.

## Comparison Boundary

The baseline is the pure GroundingDINO Stage-B data-FT checkpoint:

```text
/media/haoyi/T9/gdino/outputs/
  gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/
  checkpoint0001.pth

SHA-256:
b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157
```

The baseline continues ordinary GroundingDINO on the Stage-B data recipe. It
has one trainable phrase-score surface. The same query scores determine both:

```text
Ref box = box(argmax_q score(image, expression, q))
TN score = max_q score(image, expression, q)
```

Its ordinary detector loss is dense token sigmoid focal. For an empty-box TN
row, the `(queries, tokens)` target tensor is all zero, so every valid query and
token is trained as a negative. The historical baseline registration is
[`config/eval_baselines/gdino_stageb_data_ft_checkpoint0001_diagnostic.json`](../config/eval_baselines/gdino_stageb_data_ft_checkpoint0001_diagnostic.json).
Its legacy Ref/TN summaries are diagnostic only; the final comparison reran the
same checkpoint with paired manifests and records.

The final candidate is not one monolithic fine-tuned detector. It is a sealed
composition of:

1. a v20/P750 patch runtime supplying frozen Stage-A Top-50 boxes, patch
   logits, and canonical-caption provenance;
2. a frozen pure-GDINO data-FT base with an R100 query-rank adapter;
3. an independently trained P50 image-expression confidence adapter;
4. an eval-only R100/P50 merge; and
5. a validation-frozen routed-v3 Ref policy.

This is a same-record evaluation, not an equal-input architecture ablation.
The candidate's intended patch-episode contract includes a support patch and a
canonical category prompt, while pure GDINO receives only the referring text
and image. The supported claim is that the final pivot system beats the GDINO
checkpoint under their stated inference contracts on identical evaluation
examples. It does not isolate the gain from extra input, architecture, training
data, or routing policy.

## Deployed Data Flow

```text
support patch + canonical category
        |
        v
frozen Stage-A patch tower
        |
        +--> 900 queries/boxes --> patch Top-50 admission
        |                            boxes and query states are fixed
        |
        +--> explicit patch category score/prior

complete referring expression
        |
        +--> frozen pure GDINO + R100 rank adapter on all 900 queries
        |      frozen external boxes, base_score, and query-specific rank_score
        |
        +--> frozen pure GDINO + P50 confidence adapter on all 900 queries
               base_score + one image-expression gate broadcast to every query

Ref consumer: routed-v3 base/R100 rank path -> one selected box
FPR consumer: max over the 900 P50 confidence scores -> one absolute score
```

The phrase "text handles localization" has a narrow meaning here. Complete
text decides which already generated candidate box refers to the expression.
It does not regress or move box geometry. The patch-assisted route selects from
fixed Stage-A Top-50 boxes; the default route selects from frozen external
GDINO boxes. Neither Stage-B scoring path updates either box source.

## What Changed

| Surface | Pure GDINO Stage-B data-FT | Final pivot Stage-B |
| --- | --- | --- |
| Detector update | Broad ordinary GDINO fine-tuning, with BERT frozen | Base detectors and boxes are frozen at adapter/scorer boundaries |
| Category signal | Entangled with phrase token logits | Patch score is an explicit category prior and Top-50 admission signal; the complete-text scorer still retains the head noun |
| Canonical text | Part of the same phrase surface | The dedicated canonical-only input is used for the patch prompt, provenance, and route selection; the same head noun remains inside the complete expression |
| Complete text | One GDINO token-logit path | Scored by the external GDINO full-expression path; the v20 internal scorer exists but is not a routed-v3 score consumer |
| Box geometry | Can move during detector fine-tuning | Frozen during Stage-B ranking; text only selects among boxes |
| Ref ranking | Argmax of the shared phrase score | Direct full-text base by default; selected groups use R100 transfer plus a small patch prior |
| TN confidence | Max of the same shared phrase score | Separate P50 image-expression confidence score over all 900 queries |
| TN supervision | Empty-target dense query x token focal | One sample-global FPR surrogate per verified positive/TN pair |
| Optimization | Ref ranking and TN rejection share trainable scores | Rank and confidence have disjoint parameters, optimizers, phases, and checkpoints |
| Evaluation | A single score is reused for both jobs | Ref reads rank; FPR reads confidence; cross-consumption fails closed |

### Patch candidate and category path

`stage_a_caption` is the canonical class caption used by the original patch
tower. The support patch and canonical caption produce the category-conditioned
queries, boxes, and patch scores. Candidate admission is fixed at patch Top-50;
candidate indices, query states, and boxes are detached before the full-text
scorer.

This assigns the patch path the job supported by Stage-A evidence: keep the
right object category and a useful candidate set. The current same-caliber
evidence supports only the narrow statement that aggregate candidate recall is
slightly higher and already sufficient for downstream ranking. It does not
support an AP50 advantage, and it must not be generalized into overall
Stage-A dominance over the GroundingDINO same-data FT baseline.

### Complete-text ranking path

The complete referring expression, including the head noun, attributes,
relations, and spatial words, is sent to the independent external GDINO
scoring path. The head noun is retained as useful evidence; therefore
"category only comes from patch" would also be too strong. The actual
responsibility boundary is:

- patch controls candidate admission and supplies an explicit category prior;
- full text performs attribute/relation understanding and referring
  localization among the frozen external queries or transferred patch
  candidates.

The patch checkpoint also contains the v20 fixed-box full-text scorer. Its
internal training score is:

```text
patch_fulltext_rank(q, e)
    = fulltext_rank_logit(q, e) + 1.0 * patch_logit(q)
```

and its negative threshold is aligned to `acc50` at `0.499`. However, the
final routed-v3 evaluator does not consume this internal
`stage_b_v11_rank_score`. From the P750 runtime it consumes only the exact
Top-50 candidate indices/boxes, patch logits, and canonical caption. Routed-v3
combines those patch surfaces with the external R100 full-text score:

```text
transferred_score(candidate, expression)
    = 1.0 * transfer(R100_rank_score, external_boxes, patch_candidate)
      + patch_weight * patch_logit(candidate)
```

The final composite also never consumes the v20 patch checkpoint's confidence
output. That checkpoint inherits a gate-only confidence implementation, but it
is not the deployed TN score.

### R100 query-rank adapter

The external pure-GDINO adapter starts from the frozen data-FT checkpoint and
adds a query-specific residual:

```text
rank_score(q, e) = frozen_base_score(q, e) + rank_residual(q, e)
```

R100 is trained only on positive RefCOCO, RefCOCO+, and RefCOCOg expressions.
Its baseline-preserving objective separates two row classes:

- repair a row when the frozen top-1 is wrong but an IoU >= 0.5 query exists;
- preserve the frozen positive-negative margin when top-1 is already correct.

The two classes are normalized separately, and residual L2 limits unnecessary
score movement. No TN loss or confidence parameter is active in this phase.

### P50 absolute-confidence adapter

The confidence phase consumes the complete positive or TN expression. Frozen
GDINO token logits are sigmoid-averaged over the generated full-expression
token mask to produce `(B, 900)` base scores. A separate network pools detached
query features and score-distribution statistics, then emits one scalar gate
per image-expression pair:

```text
confidence_score(q, e) = frozen_base_score(q, e) + gate(image, e)
deployed_confidence(e)  = max over all 900 confidence_score(q, e)
```

P50 uses 17,829 semantic-verified positive/TN pairs and the
`detached_recent_q05_trust` objective:

```text
t = detach(exact q05 of recent positive global-max scores)

Lneg   = tau * softplus((TN_global - t_proxy) / tau)
Ltrust = mean relu(-0.02 - positive_gate)
Lpair  = tau * softplus((TN_global - positive_global + 0.05) / tau)

Lconfidence = Lneg + 1.0 * Ltrust + 0.25 * Lpair
```

The zero-valued translation proxy in `t_proxy` cancels the shortcut of moving
positive and TN scores down together. The trust hinge protects the positive
low tail that defines the 95%-TPR operating threshold.

This is sample-global supervision, not token-level TN supervision and not a
continuation of the baseline's all-negative focal loss. The inherited dense
focal behavior remains only in the frozen data-FT weights.

### Routed-v3 Ref policy

The final Ref policy is deliberately selective. Its default is the frozen
external GDINO full-expression base over all 900 queries. Only validation-
selected canonical groups use R100-to-patch transfer:

| Canonical route | Final behavior |
| --- | --- |
| `bowl` | unconditional `max_external_p05_w0046875_v1` transfer |
| `doughnut` | unconditional `max_external_p05_w0046875_v1` transfer |
| `person` and full expression has <= 7 ASCII lexical tokens | `max_patch_p025_w003125_v1` transfer to a fixed patch candidate |
| long `person` expression or any other category | direct frozen external base over 900 queries |

For the conditional route, lexical tokens are defined by lowercase
`[a-z0-9]+`. The route consumes the canonical caption and complete expression;
it does not inspect the target category field, target box, or test correctness
at runtime.

The full-text gate was necessary because routed-v2 regressed RefCOCOg test by
13 correct examples. The failure concentrated in long `person` expressions:
the category was correct, but the patch-assisted route often selected the wrong
person instance. Falling back to the external full-expression base on long
descriptions restored the fine-grained disambiguation signal.

## Why The Final System Beat The Baseline

The successful mechanism is a division of score responsibilities, not simply
more capacity.

### 1. It optimizes the deployed surfaces

Ref accuracy depends on relative query order within one image. R100 and the
selective transfer policy change that order directly. FPR@95TPR depends on absolute
ordering across image-expression samples after an all-query maximum. P50 trains
that exact global maximum against the positive q05 tail.

The baseline's dense token focal is only indirectly related to both deployed
decisions. The new losses match the actual inference reductions.

### 2. It removes the rank-confidence gradient conflict

A rank loss may need to raise a correct query above a hard competitor. If the
same score feeds FPR, that increase can also raise a TN global maximum. A TN
loss may need to lower a global maximum, but lowering a correct positive query
can collapse the positive q05 threshold.

The final system makes those updates mathematically independent:

```text
d(confidence_score) / d(rank parameters) = 0
d(rank_score)       / d(confidence parameters) = 0
```

R100 and P50 were trained separately and merged only for evaluation. Tensor
audits showed the 938 base tensors bitwise identical to the baseline; R50 to
R100 changed only the eight rank tensors, and P1 to P50 changed only the twelve
confidence tensors.

### 3. It preserves localization while repairing selection

The useful Stage-A/GDINO observation is that the correct box is usually already
in the candidate pool. An interrupted diagnostic saw best-of-900 recall near
1.0 on three Ref splits, while top-1 was substantially lower. That observation
is non-authoritative as a score result, but it identifies query selection as a
high-leverage bottleneck.

Freezing boxes prevents Stage-B text/TN training from destroying that candidate
recall. On selected routes, patch Top-50 narrows the category-conditioned search
space while complete text chooses the referred instance. The default route
retains the frozen external GDINO query pool.

### 4. It keeps complete text at the semantic decision point

Canonical-only scoring loses the attributes and relations needed to distinguish
same-class objects. The final design uses canonical text only to establish the
patch category/candidate context. The full expression remains available to the
fixed-box and external GDINO scorers, and the long-person fallback explicitly
prefers that full-expression evidence when the patch prior is too coarse.

### 5. It uses stronger TN semantics at the right granularity

P50 sees verified positive/TN expression pairs and gives every TN sample one
global-max training vote. It does not dilute the decision across hundreds of
token focal terms. The detached recent q05, positive trust hinge, and paired
margin jointly suppress TN maxima without buying a lower FPR by collapsing the
positive threshold.

The label contract is still imperfect: verification covered the target plus
1-8 cached proposals, not all 900 GDINO queries. Applying the semantic label to
the all-query maximum is a generalization assumption. strict1607 had zero image
overlap with this semantic training set; strict2031 had 59 overlapping image
IDs.

### 6. It applies patch transfer only where it helps

A universal patch fusion was not robust. The final route keeps direct GDINO as
the default and enables patch/R100 transfer only for validation-supported
canonical groups. The full-text length gate then removes the known long-person
failure mode. This selective fallback is why the final eight-split point gate
passed when routed-v2 did not.

This is also the least clean part of the causal claim. The route hypothesis was
introduced after inspecting the routed-v2 RefCOCOg-test failure. Although its
threshold was selected only from the three validation splits, the reused test
splits are not an independent blind holdout.

### 7. Training remained numerically stable

The P750 patch/full-text training recipe uses `batch_size = 56` and expression
microbatch 16. The recorded run held approximately 29.3 GiB steadily and had no
AMP optimizer-step skip. This is an engineering success for that training path,
but routed-v3 does not consume its internal rank or confidence outputs. High
memory use is therefore not evidence for the final metric gain.

### Evidence and attribution boundary

The following are directly checked observations:

- branch hashes prove base, rank, and confidence parameter isolation;
- routed-v2 lost 13 RefCOCOg-test examples, concentrated in the `person`
  override, while routed-v3 passed the eight point gates;
- P50 lowered paired FPR on both strict manifests; and
- the final candidate was higher on every recorded Ref split.

The explanation above is the mechanism most consistent with those observations,
but it is not a complete factorial ablation. In particular, the final Ref gain
combines the frozen patch candidate/logit surface, R100, selective transfer, and
the adaptive full-text route. The result does not identify one component as the
sole cause of every split improvement.

## Recorded Same-Protocol Results

### RefCOCO-family top-1 acc50

The formal Ref run used 57,457 paired records with batch 32, four workers, AMP,
identical manifests, stable split seeds, and an unchanged all-query oracle per
example.

| Split | Expressions | GDINO data-FT | Final candidate | Delta | Correct delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| RefCOCO val | 10,834 | 0.643068119 | 0.658205649 | +0.015137530 | +164 |
| RefCOCO testA | 5,657 | 0.723705144 | 0.743503624 | +0.019798480 | +112 |
| RefCOCO testB | 5,095 | 0.580765456 | 0.581354269 | +0.000588813 | +3 |
| RefCOCO+ val | 10,758 | 0.695296524 | 0.729596579 | +0.034300055 | +369 |
| RefCOCO+ testA | 5,726 | 0.751135173 | 0.804750262 | +0.053615089 | +307 |
| RefCOCO+ testB | 4,889 | 0.628553897 | 0.631008386 | +0.002454489 | +12 |
| RefCOCOg val | 4,896 | 0.806372549 | 0.807802288 | +0.001429739 | +7 |
| RefCOCOg test | 9,602 | 0.817538013 | 0.817642158 | +0.000104145 | +1 |

The point-estimate rule passed 8/8 splits. The smallest gains are fragile:
RefCOCO testB, RefCOCOg val, and RefCOCOg test had paired confidence intervals
that crossed zero, and RefCOCOg test improved by only one example. These are
not yet strong independent-generalization claims.

### Exact FPR@95TPR

| Manifest | Pairs | GDINO data-FT | Final P50 | Delta | False-positive delta | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict2031 | 2,031 | 0.508124077 | 0.469719350 | -0.038404727 | -78 | [-0.061286, -0.021066] |
| strict1607 | 1,607 | 0.497199751 | 0.452395769 | -0.044803983 | -72 | [-0.065138, -0.024136] |

Both paired dual-gate invocations exited zero. Unlike the narrow Ref gains, the
paired FPR intervals were below zero on both manifests.

## Artifact Lineage Recorded On 2026-07-16

```text
GDINO Stage-B data-FT checkpoint0001
  base tensor SHA-256:
  889ea86458e5c2a3a92a2cf4cbfc006662e9dd268f0d9e22bd9abbdafaa2ec14

  + R100 rank branch
    rank tensor SHA-256:
    90f8d970ffeb2b906acedc6b908ecc96802acc88362ee1a51d04b5c93e0fe335

  + P50 semantic confidence branch
    confidence tensor SHA-256:
    e30e5e07cd5b0d83336547e3b038deb67f5acc94ae6c28d2ccc6b681072b8c1d

  -> eval-only merged checkpoint
     /tmp/pivot_gdino_adapter_merged_r100_p50_eval_20260716/
       checkpoint_eval_only.pth
     file SHA-256:
     face382af1d94e09d5c5cf3d729215253530d437bda4565da3ea1875bd171918
     full tensor SHA-256:
     7c5c25a4c52b4fc469b9ecb3301713bb1383b0af6ab03fe9cbd46fe5af221555
```

The merge is evaluation-only and intentionally contains no optimizer,
scheduler, scaler, criterion queue, RNG, or resumable training state.

The recorded final routing artifacts were:

```text
full-text gate:
  /tmp/pivot_stageb_fulltext_person_gate_fullval_v2_20260716.json
  logical SHA-256:
  f9525ab1d98ad233c35a3e8554c6c029bef53c616b4e929635036d8085a23f1f
  file SHA-256:
  44ed5362b49dabbcafa5ebf0cf4a1572495f1f06cc7fc677b99caa02913bf30d

routed-v3 formal artifact:
  /tmp/pivot_stageb_external_rank_route_r100p50_fulltext_gate_v3_20260716.json
  logical SHA-256:
  8c996481287d17acf02398dd97ad6d7edbc14d21ba96807c5867002375cd1683
  file SHA-256:
  fd90f1df5ce33e65d926e1acfae76a9bc5049299f85ac8e09e1b84cc1a782c1c

Ref8 summary:
  /tmp/pivot_stageb_fulltext_gated_v3_formal_ref8_b32_amp_20260716/summary.json
  SHA-256:
  893838c005d86eb9367b0a3330db9e0efad01f0724eaa091a871be4f22237643

paired gate reports:
  /tmp/pivot_stageb_fulltext_gated_v3_dual_gate_strict2031_b32_amp_20260716.json
  /tmp/pivot_stageb_fulltext_gated_v3_dual_gate_strict1607_b32_amp_20260716.json
```

These paths document lineage only; the files are currently absent.

## Non-Negotiable Invariants

Future iterations must preserve these contracts unless a new experiment
explicitly replaces and revalidates them:

1. Ref top-1 consumes a rank score; FPR consumes a confidence score.
2. Rank and confidence may not share a trainable score trunk or optimizer
   parameter.
3. Confidence remains an image-expression scalar broadcast over queries; it
   may not change Ref query order.
4. Rank may change query order but may not feed the FPR global maximum.
5. Canonical text may prompt/category-route the patch branch, but it may not
   replace the complete expression at the semantic scoring boundary.
6. Complete-text localization means candidate selection, not box regression.
7. Patch Top-50 admission, box geometry, and query identity must remain fixed
   across positive/TN expression slots.
8. A TN label must state whether it is target-local, proposal-set verified, or
   image-global. The loss may not silently supervise a broader surface.
9. Formal comparisons require identical manifests, record order, score
   definitions, batch size, workers, AMP, and stable split seeds.
10. Route selection must be redone on validation-only evidence for any changed
    checkpoint; no current test split may be presented as a fresh holdout.

## Key Implementation Files

- Adapter and rank/confidence losses:
  [`models/GroundingDINO/stage_b_gdino_score_adapter.py`](../models/GroundingDINO/stage_b_gdino_score_adapter.py)
- Full-expression aggregation and adapter outputs:
  [`models/GroundingDINO/groundingdino.py`](../models/GroundingDINO/groundingdino.py)
- Fixed-box complete-text scorer retained inside the patch checkpoint but not
  consumed by routed-v3:
  [`models/GroundingDINO/stage_b_fixed_text_scorer.py`](../models/GroundingDINO/stage_b_fixed_text_scorer.py)
- v20 acc50-aligned patch rank recipe:
  [`config/ablations/cfg_stageb_v20_acc50_aligned_hard_negatives.py`](../config/ablations/cfg_stageb_v20_acc50_aligned_hard_negatives.py)
- R100 rank recipe:
  [`config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py`](../config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py)
- P50 semantic confidence recipe:
  [`config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py`](../config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py)
- Eval-only R100/P50 merger:
  [`tools/merge_stageb_gdino_adapter_eval.py`](../tools/merge_stageb_gdino_adapter_eval.py)
- Composite role validator:
  [`tools/stageb_composite_artifact.py`](../tools/stageb_composite_artifact.py)
- Canonical route selection:
  [`tools/stageb_canonical_caption_route_artifact.py`](../tools/stageb_canonical_caption_route_artifact.py)
- Full-expression route gate:
  [`tools/stageb_fulltext_route_gate_artifact.py`](../tools/stageb_fulltext_route_gate_artifact.py)
- Routed-v3 artifact:
  [`tools/stageb_external_rank_transfer_artifact.py`](../tools/stageb_external_rank_transfer_artifact.py)
- Ref evaluator and record verifier:
  [`tools/eval_refcoco_stageb.py`](../tools/eval_refcoco_stageb.py)
- Final paired gate:
  [`tools/verify_stageb_dual_gate.py`](../tools/verify_stageb_dual_gate.py)

## Required Follow-Up

1. Re-materialize the exact R100, P50, merged checkpoint, P750 patch checkpoint,
   route artifacts, Ref8 records, and both strict TN record sets.
2. Store them outside `/tmp` with file SHA-256 sidecars and immutable lineage.
3. Replay both paired dual gates from the durable copies.
4. Evaluate the frozen architecture once on a genuinely unseen holdout. This is
   required to resolve the adaptive route-selection caveat and the one-example
   RefCOCOg-test margin.
5. Treat any routing, score-consumer, or candidate-admission change as a new
   model and rerun all eight Ref splits plus both strict FPR manifests.
