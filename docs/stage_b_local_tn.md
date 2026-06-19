# Stage B Local TN Supervision

Stage B treats a true-negative (TN) RefCOCO expression as a local text conflict. The matched query still points to the real object; only the changed content tokens are supervised as false.

Example:

```text
positive: blue shirt man
TN:       white shirt man
```

The current v2 supervision is:

```text
white -> target 0
shirt -> target 1 if it is non-canonical content in the slot
man   -> target 1 with canonical/head weight
```

Stage B v2 does not train phrase ranking by default. The rank-enabled v3 /
ablation path can additionally train an independent phrase-ranking loss for TN
rows that have a real `positive_phrase`:

```text
S(q, t+, p) > S(q, t-, p)
```

The ranking score is the same phrase score used by Stage-B inference. `try_tn_head_phrase` is not used as a fallback for ranking positives.

## Dataset Masks

`datasets/patch_episode.py` builds these text masks per support slot:

```text
phrase_to_token_mask
canonical_to_token_mask
content_to_token_mask
attr_pos_to_token_mask        # compatibility alias for positive content tokens
attr_neg_to_token_mask        # compatibility alias for TN negative tokens
negative_to_token_mask        # same TN negative mask for older callers
relation_to_token_mask        # audit/debug only
phrase_semantic_token_mask    # compatibility alias for content_to_token_mask
is_tn
attr_neg_weight_mask
tn_group_ids
rank_positive_phrase_to_token_mask
rank_positive_canonical_to_token_mask
has_rank_positive
```

`content_to_token_mask` is training-only. It marks non-canonical meaningful content tokens inside the phrase slot. It excludes canonical/head tokens, articles `a/an/the`, punctuation, special tokens, and padding.

It intentionally keeps relation/spatial prepositions such as:

```text
to of in on with from near behind under over above below
```

This preserves phrases such as `next to`, `left of`, `on top of`, and `with a hat`.

## TN Category Groups

Raw `replace_category` values are normalized and mapped to coarse groups:

```text
color_like             category contains "color"
attr_like              attribute, size, clothing, age, state, material, accessory, pattern, etc.
spatial_like           spatial, position, spatial relation, location
relation_action_like   action, posture, pose
other                  everything else
```

All valid groups currently use TN negative token weight `1.0`. `default_tn_category_weight` is also `1.0`; spatial/relation/action TN tokens are not filtered out.

Dataset loading logs:

```text
raw_category_counts
normalized_group_counts
rows_with_category
total_edits
```

The invalid text-mask audit rows include TN group/category when available.

## Balanced TN Sampling

For prebuilt TN datasets, one primary group is taken from the first normalized `replace_category` in the row. Stage B uses capped inverse-sqrt group weights:

```python
weight = min(tn_balance_cap, sqrt(max_group_count / group_count))
```

The per-sample weights are normalized inside each dataset, so dataset-level `mix_weight` stays unchanged. The current default cap is `5.0`.

## Text Loss

`models/GroundingDINO/stage_b_criterion.py` uses token-level BCE only:

```text
canonical_to_token_mask        target 1, weight canonical_pos_weight
content_to_token_mask          target 1, weight 1.0
attr_neg_to_token_mask/TN neg  target 0, weight 1.0
```

Mask priority is:

```text
canonical > TN negative > content positive > ignore
```

So changed TN tokens are removed from the effective positive content mask before loss is computed.

The total Stage-B text loss is:

```python
loss_text = content_pos_loss + canonical_loss + tn_neg_loss
```

Phrase ranking, when an ablation explicitly enables it, is a separate loss:

```python
loss_phrase_rank = mean(max(0, stage_b_rank_margin - S_pos + S_neg))
```

`loss_phrase_rank` is weighted by `stage_b_rank_loss_coef`; it is not part of
`loss_text` and is not multiplied by `lambda_text`. The current v2 default rank
loss coefficient is `0.0`; rank-enabled v3 / ablation configs set it to `1.0`.

Ranking uses match-by-target alignment. The negative forward and positive forward each run Hungarian matching; scores are compared only when both forwards match the same GT target id and the same support slot. The patch part of the rank score is detached by default (`stage_b_rank_detach_patch=True`).

There is no softmin phrase rejection loss in v2. The old knobs `attr_pos_weight`, `tn_shared_attr_pos_weight`, `attr_neg_weight`, `use_phrase_tn_loss`, `phrase_score_type`, `softmin_tau`, and `lambda_phrase` are deprecated and ignored by the content-token loss.

## Stage-B v4 Score Calibration

Stage-B v4 is an ablation that starts from the v2 `checkpoint0003.pth`, replaces
the v2 matched-query token BCE with a GroundingDINO-like all-query sigmoid focal
text loss, and adds inference-score calibration. It does not enable the
historical v3 phrase-rank loss.

The v4 text loss builds a dense `(query, token)` target map from the Hungarian
matched query/slot pairs: matched positive content and canonical tokens are
positive, and all other valid query-token cells are focal negatives. This keeps
the text head closer to original GroundingDINO dense token classification than
the earlier matched-only BCE. Because dense focal has many query-token
negatives, v4 uses `lambda_text = 0.05` instead of the v2 `0.25`.

The v4 score calibration runs on TN rows with a valid positive prompt:

```text
S = patch_score + beta * text_score
```

The scorer is the same Stage-B slot scorer used by RefCOCO/TN-val evaluation.
The current v4 loss uses the matched positive query and the TN prompt's global
top-10 over all `(query, slot)` scores, matching the inference flattening of
`slot_logits.reshape(B, -1)`:

```text
loss_pos       = softplus(tau_pos - S_pos)
loss_neg       = mean_global_top10 softplus(S_tn_topk - tau_neg)
loss_gap       = mean_global_top10 softplus(margin - S_pos + S_tn_topk)
loss_pos_query = mean_top10 softplus(margin - S_pos + S_pos_other_topk)
```

`score_calib_neg_matched_score` remains logged only as a diagnostic; it is no
longer the negative score used by `loss_neg` or `loss_gap`.

Default v4 constants come from the recorded v2 `checkpoint0003.pth` TN-val score
quantiles at `beta=1`:

```text
outputs/stageb_tn_val_compare_quantiles_v2/v2_ckpt0003_beta1_score_quantiles.json

tau_pos = 0.10   # v2 positive score q10 ~= 0.096
tau_neg = 1.40   # v2 TN score q75 ~= 1.425
margin  = 0.30   # conservative gap below v2 gap median ~= 0.641
topk    = 10
```

The weighted v4 objective is:

```text
loss_score_calib =
  0.5 * loss_neg
+ 0.1 * loss_pos
+ 0.1 * loss_gap
+ 0.1 * loss_pos_query
```

This objective is intended to lower high-scoring TN tails and improve top-1
query ordering without globally raising all positive/TN scores as the v3
pairwise rank probe did.

## Stage-B v5 all-TN Score Calibration

Stage-B v5 is the current TN/FPR research branch, not the v2 mainline. It starts
from the recorded v2 epoch-3 checkpoint and changes both the optimization scope
and the loss:

```text
base checkpoint:
outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth

selected long-run checkpoint:
outputs/stageB_v5_tnneg10_lse_top10_alltn00625_from_v2e3_bs4_long/checkpoint_iter0020000.pth
```

The selected config is:

```text
config/ablations/cfg_stageb_v5_tnneg10_lse_top10_alltn00625_from_w0125.py
```

Compared with v2, `alltn00625` is not just a same-freeze score-head probe:

| item | v2 mainline | v5 alltn00625 |
|---|---|---|
| text loss | `matched_bce` | `allquery_focal_tn_matched_bce` |
| trainable scope | `feat_map`, `class_embed` | `feat_map`, `class_embed`, `bbox_embed`, `transformer.decoder.layers` |
| box losses | off (`bbox_loss_coef=0`, `giou_loss_coef=0`) | on (`bbox_loss_coef=5`, `giou_loss_coef=2`) |
| TN token weights | standard | `lambda_tn_neg=10`, `lambda_tn_content=0`, `lambda_tn_canonical=0` |
| score calibration | off | on |

The v5 text loss keeps TN rows on the matched-query BCE path, but positive
RefCOCO rows use all-query focal text supervision. Unmatched queries with
`IoU > 0.5` to a ground-truth box are also supervised as positive queries for
the corresponding slot.

Stage-B 5.x lineage:

| version | base | intended delta |
|---|---|---|
| v5 `alltn00625` | v2 `checkpoint0003.pth` | all-query focal for positive text rows, matched-query TN BCE, decoder/box-head unfreeze, bbox/GIoU losses, top-10 score calibration with all-TN penalty |
| v5.1 | v5 `alltn00625` | RefCOCO-family patch CE positive-only; LVIS/COCO patch CE unchanged |
| v5.2 | v5.1 | enable decoder aux losses for patch/text/bbox/GIoU |
| v5.2 text sweep | v5.2 | only sweep `lambda_text` (`0.30` to `0.75`) |
| v5.3 | v5.2 | `lambda_text=1.0` |
| v5.4 | v5.2 | additionally unfreeze `backbone.0`, `input_proj`, and `transformer.encoder`; patch branch and BERT remain frozen |
| v5.5 | v5.2 | restrict decoder trainable scope to layers 3/4/5 and aux losses to layers 3/4 only; final layer 5 keeps main loss |

The selected score calibration uses the final Stage-B inference score:

```text
S = patch_score + beta * text_score
```

and applies a hard-tail global top-10 constraint:

```text
stage_b_score_calib_loss_coef = 1.0
stage_b_score_calib_topk = 10
stage_b_score_calib_neg_agg = "logsumexp"
stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_tau_pos = 0.1
stage_b_score_calib_tau_neg = -2.4
stage_b_score_calib_margin = 0.8
stage_b_score_calib_pos_weight = 0.05
stage_b_score_calib_neg_weight = 0.125
stage_b_score_calib_gap_weight = 0.125
stage_b_score_calib_pos_query_weight = 0.05
stage_b_score_calib_all_tn_neg_weight = 0.0625
stage_b_score_calib_detach_patch = False
```

`stage_b_score_calib_all_tn_neg_weight=0.0625` is the final addition in the
selected probe. It adds a light penalty on every TN slot's actual global
top-10 final score, rather than only the sparse positive/TN rank pairs. This
targets the same score distribution that TN-val and RefCOCO inference use.

### alltn00625 Result

Primary comparison points:

| checkpoint | best beta | TN-val FPR@95 | TN-val pair win | RefCOCO val acc50 | RefCOCO+ val acc50 |
|---|---:|---:|---:|---:|---:|
| v2 `checkpoint0003.pth` | 1.0 | 0.780817 | 0.744607 | 0.533229 | 0.564045 |
| v2 `checkpoint0003.pth` Ref-best | 0.5 | 0.797381 | 0.731125 | 0.535536 | 0.568600 |
| alltn00625 1k | 2.0 | 0.780046 | 0.751541 | 0.544213 | 0.565533 |
| alltn00625 1k Ref-best | 0.5 | 0.810285 | 0.732473 | 0.567196 | 0.596672 |
| alltn00625 long20k | 0.5 | 0.752119 | 0.789869 | 0.578826 | 0.595464 |

Evidence files:

```text
outputs/stageb_tn_val_v4_allquery_focal_vs_v2/summary.md
outputs/refcoco_stageb_v2_v3_eval_val/summary.md
outputs/stageb_tn_val_v5_tnneg10_lse_top10_alltn00625_1k/summary.md
outputs/refcoco_stageb_v5_tnneg10_lse_top10_alltn00625_1k_ref_refp/summary.md
outputs/stageb_tn_val_v5_tnneg10_lse_top10_alltn00625_long20k/summary.md
outputs/refcoco_stageb_v5_tnneg10_lse_top10_alltn00625_long20k_ref_refp/summary.md
```

The 1k alltn00625 probe was already competitive with v2 on TN FPR, but its
TN-best beta did not align with the RefCOCO-best beta. The long20k checkpoint is
the first clean candidate in this branch: beta `0.5` is best for both TN-val
FPR@95 and RefCOCO/RefCOCO+ val acc50.

### Stage-B v5.1 RefCOCO Patch-Positive CE Probe

v5.1 keeps the selected v5 alltn00625 recipe and changes only the patch CE
semantics for RefCOCO-family phrase rows:

```text
config/ablations/cfg_stageb_v5_1_refcoco_patchpos_from_v5_alltn00625.py
```

Motivation: RefCOCO/RefCOCO+/RefCOCOg phrase data annotates the referred
expression object, not every same-class object in the image. Dense patch CE
negatives on those rows can therefore treat unannotated valid referents as
false negatives. v5.1 keeps LVIS/COCO patch CE unchanged, but for targets whose
`dataset_name/source` matches `refcoco`, `refcocoplus`, `refcocog`, or `refexp`,
`loss_patch_ce` is computed from matched positive patch cells only.

The training log should show nonzero `patch_ce_positive_only_batch_frac` when
RefCOCO-family rows are sampled. `patch_ce_neg_count` should remain nonzero on
LVIS/COCO batches.

For a fair 1k ablation, initialize all rows below from the same v2 checkpoint:

```text
outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth
```

v5.2 adds auxiliary decoder losses on top of v5.1:

```text
config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625.py
```

`aux_loss=True` makes the patch-only forward path expose intermediate decoder
patch scores. `StageBCriterion` then applies aux `loss_patch_ce`, `loss_text`,
`loss_bbox`, and `loss_giou` for intermediate decoder layers. Score calibration
and phrase-rank losses remain final-layer only.

The text-weight sweep probes whether stronger text supervision can recover more
RefCOCO localization without giving back the TN-rejection gain:

```text
config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_text030_from_v5_alltn00625.py
lambda_text = 0.30

config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_text035_from_v5_alltn00625.py
lambda_text = 0.35

config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_text040_from_v5_alltn00625.py
lambda_text = 0.40

config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_text045_from_v5_alltn00625.py
lambda_text = 0.45

config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_text050_from_v5_alltn00625.py
lambda_text = 0.5

config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_text075_from_v5_alltn00625.py
lambda_text = 0.75

config/ablations/cfg_stageb_v5_3_refcoco_patchpos_aux_text1_from_v5_alltn00625.py
lambda_text = 1.0
```

These configs keep the v5.2 RefCOCO patch-positive CE, aux losses, TN
calibration, and freeze/unfreeze settings unchanged, and only sweep
`lambda_text` from the v5 family default `0.25`.

Fair 1k result:

| probe | TN-best beta | TN FPR@95 | TN pair win | pos top1 IoU50 | Ref-best beta | mean Ref acc50 | RefCOCO val acc50 | RefCOCO+ val acc50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v5 alltn00625 1k | 2.0 | 0.780046 | 0.751541 | 0.725539 | 0.5 | 0.581934 | 0.567196 | 0.596672 |
| v5.1 patch-positive 1k | 1.0 | 0.759823 | 0.779083 | 0.746341 | 0.5 | 0.574426 | 0.561104 | 0.587749 |
| v5.2 patch-positive + aux 1k | 1.0 | 0.757126 | 0.778505 | 0.750193 | 0.5 | 0.586751 | 0.571811 | 0.601692 |
| v5.2 + `lambda_text=0.30` 1k | 2.0 | 0.754237 | 0.781202 | 0.741718 | 0.5 | 0.586394 | 0.567750 | 0.605038 |
| v5.2 + `lambda_text=0.35` 1k | 1.0 | 0.751733 | 0.772149 | 0.745955 | 1.0 | 0.583279 | 0.568119 | 0.598438 |
| v5.2 + `lambda_text=0.40` 1k | 1.0 | 0.747304 | 0.779083 | 0.753467 | 1.0 | 0.581381 | 0.565811 | 0.596951 |
| v5.2 + `lambda_text=0.45` 1k | 1.0 | 0.762519 | 0.781587 | 0.755586 | 0.5 | 0.587875 | 0.569596 | 0.606154 |
| v5.2 + `lambda_text=0.5` 1k | 1.0 | 0.761749 | 0.774846 | 0.748074 | 1.0 | 0.584947 | 0.569411 | 0.600483 |
| v5.2 + `lambda_text=0.75` 1k | 1.0 | 0.772920 | 0.782357 | 0.756549 | 1.0 | 0.588189 | 0.572826 | 0.603551 |
| v5.3 v5.2 + `lambda_text=1` 1k | 1.0 | 0.785054 | 0.763867 | 0.754045 | 1.0 | 0.590285 | 0.571534 | 0.609035 |

At a fixed beta `1.0`:

| probe | TN FPR@95 | TN pair win | mean Ref acc50 | RefCOCO val acc50 | RefCOCO+ val acc50 |
|---|---:|---:|---:|---:|---:|
| v5 alltn00625 1k | 0.790254 | 0.742488 | 0.573496 | 0.561196 | 0.585797 |
| v5.1 patch-positive 1k | 0.759823 | 0.779083 | 0.569188 | 0.557227 | 0.581149 |
| v5.2 patch-positive + aux 1k | 0.757126 | 0.778505 | 0.583785 | 0.569596 | 0.597974 |
| v5.2 + `lambda_text=0.30` 1k | 0.758475 | 0.780046 | 0.584346 | 0.568673 | 0.600019 |
| v5.2 + `lambda_text=0.35` 1k | 0.751733 | 0.772149 | 0.583279 | 0.568119 | 0.598438 |
| v5.2 + `lambda_text=0.40` 1k | 0.747304 | 0.779083 | 0.581381 | 0.565811 | 0.596951 |
| v5.2 + `lambda_text=0.45` 1k | 0.762519 | 0.781587 | 0.586567 | 0.571257 | 0.601878 |
| v5.2 + `lambda_text=0.5` 1k | 0.761749 | 0.774846 | 0.584947 | 0.569411 | 0.600483 |
| v5.2 + `lambda_text=0.75` 1k | 0.772920 | 0.782357 | 0.588189 | 0.572826 | 0.603551 |
| v5.3 v5.2 + `lambda_text=1` 1k | 0.785054 | 0.763867 | 0.590285 | 0.571534 | 0.609035 |

Evidence files:

```text
outputs/stageB_v5_1_refcoco_patchpos_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageB_v5_2_refcoco_patchpos_aux_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageB_v5_2_refcoco_patchpos_aux_text030_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageB_v5_2_refcoco_patchpos_aux_text035_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageB_v5_2_refcoco_patchpos_aux_text040_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageB_v5_2_refcoco_patchpos_aux_text045_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageB_v5_2_refcoco_patchpos_aux_text050_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageB_v5_2_refcoco_patchpos_aux_text075_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageB_v5_3_refcoco_patchpos_aux_text1_from_v2e3_probe1k/checkpoint_iter.pth
outputs/stageb_tn_val_v5_1_refcoco_patchpos_from_v2e3_probe1k/summary.md
outputs/stageb_tn_val_v5_2_refcoco_patchpos_aux_from_v2e3_probe1k/summary.md
outputs/stageb_tn_val_v5_2_lambda_text_sweep_030_045_probe1k/summary.md
outputs/stageb_tn_val_v5_2_lambda_text_sweep_050_075_probe1k/summary.md
outputs/stageb_tn_val_v5_3_refcoco_patchpos_aux_text1_from_v2e3_probe1k/summary.md
outputs/refcoco_stageb_v5_1_refcoco_patchpos_from_v2e3_probe1k_ref_refp/summary.md
outputs/refcoco_stageb_v5_2_refcoco_patchpos_aux_from_v2e3_probe1k_ref_refp/summary.md
outputs/refcoco_stageb_v5_2_lambda_text_sweep_030_045_probe1k_ref_refp/summary.md
outputs/refcoco_stageb_v5_2_lambda_text_sweep_050_075_probe1k_ref_refp/summary.md
outputs/refcoco_stageb_v5_3_refcoco_patchpos_aux_text1_from_v2e3_probe1k_ref_refp/summary.md
```

Conclusion: v5.1 confirms that removing RefCOCO-family patch CE negatives helps
TN separation, but by itself it hurts RefCOCO localization accuracy relative to
v5 alltn00625 1k. v5.2 recovers and improves the positive grounding side while
keeping the lower TN FPR. The finer `0.30/0.35/0.40/0.45` sweep shows the
tradeoff is not monotonic near the default: `lambda_text=0.40` is the best
TN/FPR point (`0.747304` FPR@95 at beta `1.0`) but gives back RefCOCO accuracy
(`0.581381` mean acc50). `lambda_text=0.45` is the best Ref point in the fine
sweep (`0.587875`) but already regresses FPR to `0.762519`. `lambda_text=0.30`
is the best balanced fine-sweep point: it keeps Ref accuracy near the default
(`0.586394` vs `0.586751`) while modestly improving TN-best FPR (`0.754237` vs
`0.757126`). For the current TN-rejection objective, keep v5.2
`lambda_text=0.25` as the established default unless the next longer run chooses
the `0.30` balanced candidate or the `0.40` FPR-focused candidate explicitly.

The RefCOCO gain should be interpreted as ranking/calibration improvement, not
as proof that the frozen proposal pool changed by itself. v5 alltn00625 does
unfreeze decoder and box heads, but the measured improvement is still consistent
with better top-1 query/slot selection: TN-val `pos_top1_iou50` improves from
v2 beta=1 `0.702234` to alltn00625 long20k beta=0.5 `0.782550`, while the
selected TN top-1 IoU50 is still high (`0.479391`), showing that remaining FPR
is a score-tail separation problem rather than a pure localization failure.

Near-v2-FPR probes that were checked on RefCOCO val did not beat the selected
long20k branch:

| probe | TN-best beta | TN FPR@95 | RefCOCO val best | RefCOCO val at TN-best beta |
|---|---:|---:|---:|---:|
| `w0125_tau02` | 2.0 | 0.783513 | 0.566273 | 0.555658 |
| `w025` | 1.0 | 0.780046 | 0.560181 | 0.543843 |
| `neg025_gap0125_tau02` | 2.0 | 0.784861 | 0.563227 | 0.535352 |
| `w025_detach` | 2.0 | 0.790062 | 0.565904 | 0.534706 |
| `w025_tau02_detach` | 1.0 | 0.795262 | 0.562304 | 0.542459 |

## Stage-B v6 GDINO-like Text + v5.2 Patch CE Probe

Stage-B v6 tests whether matching the GDINO Stage-B data-FT text/detection loss
semantics can improve positive RefCOCO grounding while preserving patch support
learning. The current v6 definition keeps v5.2's patch CE path, including
RefCOCO-family positive-only patch CE and aux losses, but makes the text branch
as GDINO-like as possible: all-query token sigmoid focal, uniform token weights,
`lambda_text=2.0` to mirror GDINO `cls_loss_coef=2.0`, TN rows as no-positive
all-negative text rows, no extra-IoU positive query expansion, and no TN-specific
BCE/calibration.

```text
config:
config/ablations/cfg_stageb_v6_gdino_like_tn_empty_det_patchpos_aux.py

initial checkpoint:
outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth

1k probe checkpoint:
outputs/stageB_v6_gdino_like_text_v52patch_from_v2e3_probe1k/checkpoint_iter.pth
```

The key semantic split is:

| branch | positive samples | TN samples |
|---|---|---|
| patch branch | matched support/box positive | still matched support/box positive |
| text branch | all-query sigmoid focal positives on matched content/canonical tokens, uniform token weights | empty-det negative prompt: no positive text tokens |
| detection box branch | bbox/GIoU on matched non-TN boxes | empty-det: no bbox/GIoU target |

Important config deltas:

```text
stage_b_text_loss_type = "allquery_focal_tn_empty_det"
lambda_patch = 1.0
lambda_text = 2.0
cls_loss_coef = 2.0
canonical_pos_weight = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
aux_loss = True
patch_ce_positive_only_for_datasets = ("refcoco", "refcocoplus", "refcocog", "refexp")
stage_b_extra_iou_match_thr = 0.0
stage_b_enable_phrase_rank = False
stage_b_score_calib_loss_coef = 0.0
```

The smoke/probe logs should verify the intended TN semantics:

```text
stageb_tn_box_filtered_count > 0 when TN appears
text_v6_tn_empty_det_sample_count > 0 when TN appears
text_v6_tn_patch_matched_query_count > 0 when TN appears
text_v6_tn_text_positive_token_count = 0
```

Separate Stage-A-initialized probe: the full-val table below is from a v6-style
run initialized directly from Stage A `checkpoint0006.pth`
(`stageA0006_probe1k`). It is not the current Stage-B V6 lineage, which
continues from Stage-B v2 `checkpoint0003.pth`; do not compare it as an older
checkpoint of the same run.

| checkpoint | best beta | TN-val FPR@95 | TN pair win | RefCOCO val acc50 | RefCOCO+ val acc50 | RefCOCOg val acc50 |
|---|---:|---:|---:|---:|---:|---:|
| Stage-A-init v6-style 1k TN-best | 2.0 | 0.890216 | 0.634630 | 0.572734 | 0.606061 | 0.711601 |
| Stage-A-init v6-style 1k Ref-best | 1.0 | 0.896572 | 0.628082 | 0.572365 | 0.607269 | 0.712418 |

Evidence files:

```text
outputs/stageb_tn_val_v6_gdino_like_tn_empty_det_patchpos_probe1k/summary.md
outputs/refcoco_stageb_v6_gdino_like_tn_empty_det_patchpos_probe1k_ref_refp_refg/summary.md
```

Important: v6 aligns the text-loss formula with original GroundingDINO, but it
is not criterion-identical. The Stage-B wrapper still gets its Hungarian
assignment from the patch branch (`pred_logits_patch` via
`PatchHungarianCriterion.compute_matching`). Original GroundingDINO matching
uses text-token classification cost from the text logits and `positive_map`.
The aux text losses in v6 also inherit aux patch matching. If the two matchers
choose different queries, the same all-query focal formula supervises different
query rows.

The target-token map should be semantically equivalent when content/canonical
masks are built from the same phrase spans, but this is an audit point rather
than the main known mismatch. The larger known differences are:

| item | pure GroundingDINO Stage-B data FT | Stage-B v6 wrapper |
|---|---|---|
| text loss formula | all-query sigmoid focal | all-query sigmoid focal |
| text loss weight | `cls_loss_coef=2.0` | `lambda_text=2.0`, `cls_loss_coef=2.0` for provenance |
| TN rows | no positive boxes/text tokens | no positive text tokens; patch branch still has TN support positives |
| Hungarian matching | text-logit class cost plus box costs | patch-logit matching, box losses filtered for TN |
| aux text supervision | original aux matching path | aux patch matching path |
| inference score | text phrase score | `patch_score + beta * text_score` slot fusion |

Completed Stage-A0006 epoch-1 v6 comparison:

```text
checkpoint:
outputs/stageB_v6_gdino_like_tn_empty_det_patchpos_aux_from_stageA0006_bs5_resume35000_epoch1/checkpoint0000.pth
```

Shared TN/RefCOCO-val readout versus pure GroundingDINO Stage-B data FT
`checkpoint0001.pth`:

| checkpoint | TN FPR@95 | TN FPR@90 | TN pair win | RefCOCO val acc50 | RefCOCO+ val acc50 | RefCOCOg val acc50 | shared mean acc50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure GDINO Stage-B FT `ckpt0001` | 0.528302 | 0.450905 | 0.825760 | 0.645468 | 0.690835 | 0.806985 | 0.714429 |
| Stage-B v6 Stage-A0006 `ckpt0000` beta=2 | 0.790832 | 0.661980 | 0.767720 | 0.600332 | 0.652352 | 0.721405 | 0.658030 |

Pure GDINO `checkpoint0000.pth` was also evaluated on the full RefCOCO series,
but that comparison did not rerun TN metrics. It is effectively tied with
`checkpoint0001.pth` on RefCOCO:

| pure GDINO checkpoint | val mean acc50 | full-series mean acc50 | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|---|---:|---:|---:|---:|---:|
| `ckpt0000` | 0.713925 | 0.704004 | 0.645099 | 0.697249 | 0.799428 |
| `ckpt0001` | 0.714429 | 0.703895 | 0.645468 | 0.690835 | 0.806985 |

Evidence files:

```text
outputs/text_gdino_ft_stageb_with_tn_ckpt0000_refcoco_series_eval/compare_with_ckpt0001.md
outputs/text_gdino_ft_stageb_with_tn_ckpt0001_tn_eval/summary.md
outputs/text_gdino_ft_stageb_with_tn_ckpt0001_refcoco_series_eval/summary.md
outputs/stageb_tn_val_v6_gdino_like_tn_empty_det_patchpos_from_stageA0006_epoch1_bs5/summary.md
outputs/refcoco_stageb_v6_gdino_like_tn_empty_det_patchpos_from_stageA0006_epoch1_bs5_ref_refp_refg/summary.md
```

This comparison is not a clean "patch loss only" ablation: v6 still uses the
Stage-B wrapper, patch-based matching, and beta-fused Stage-B inference scorer.
Current evidence says v6 does not yet close the gap to pure GDINO Stage-B data
FT on either TN rejection or RefCOCO localization.

Decision status for the v2-initialized fair v6 probe remains pending; keep that
separate from the completed Stage-A0006 epoch-1 run above.

## Pure GroundingDINO allTN Calibration

The pure GroundingDINO Stage-B data-FT allTN ablation is separate from the
Stage-B wrapper v5/v6 lineage. It uses:

```text
patch_only = False
stage_b = False
enable_patch_branch = False
config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py
```

It keeps original GroundingDINO all-query sigmoid focal text loss and original
Hungarian matching. TN rows are zero-region/all-negative rows for the dense
focal objective, with two added TN-only terms:

```text
loss_tn_tokens: TN negative/content/canonical token BCE
loss_tn_alltn: top-10 sigmoid-mean query score suppression
```

The selected allTN aggregate threshold is:

```text
gdino_tn_alltn_topk = 10
gdino_tn_alltn_lse_tau = 0.2
gdino_tn_alltn_tau_neg = 0.5605
```

`0.5605` is not a raw query threshold. It is the logsumexp aggregate
corresponding to ten top queries all scoring about `0.1` under sigmoid-mean
token scoring:

```text
0.1 + 0.2 * log(10) ~= 0.5605
```

The measured train-mix calibration selected:

```text
gdino_tn_alltn_weight = 0.36
```

Evidence:

```text
outputs/gdino_alltn_calibration_stagea0001_tau05605_mix120/calibration.json
outputs/text_gdino_alltn_tau05605_weight_probe303_eval/summary.md
```

303-iter probe summary:

| config | mean RefCOCO acc50 | TN FPR@95 | TN FPR@90 | TN pair win |
|---|---:|---:|---:|---:|
| baseline `tau=0.0625,w=0.0625` | 0.622222 | 0.888378 | 0.810619 | 0.636706 |
| `tau=0.5605,w=0.1809` | 0.624306 | 0.913880 | 0.836120 | 0.628763 |
| selected `tau=0.5605,w=0.36` | 0.625278 | 0.883779 | 0.806856 | 0.631689 |

Decision: `tau=0.5605,w=0.36` is the current selected pure-GDINO allTN setting.
The weaker `w=0.1809` raises RefCOCO but lets TN FPR degrade, so it should not
be used for the full data-FT run.

## Inference

`PostProcessStageB.compute_slot_logits` does not use `content_to_token_mask` or `phrase_semantic_token_mask`. Inference scoring uses phrase-level token spans from `phrase_to_token_mask`, with the canonical mask only for the configured canonical contribution.

Training-only content masks therefore cannot change demo/postprocess scoring.

The phrase scorer is configured by `stage_b_infer_text_agg`. The default remains `mean`. The optional mixed scorer is:

```python
stage_b_infer_text_agg = "mean_norm_softmin"
score = alpha * mean_score + (1 - alpha) * normalized_softmin_score
```

where:

```python
normalized_softmin_score = softmin_score + tau * log(num_tokens)
```

This normalization keeps the score equal to the common token logit when all selected token logits are equal. The rank-enabled ablation path uses the same scorer config as inference.

## Default Config

```python
tn_loss_profile = "standard"
canonical_pos_weight = 0.15
stage_b_infer_text_agg = "mean"
stage_b_infer_softmin_tau = 0.7
stage_b_infer_mean_softmin_alpha = 0.5
stage_b_enable_phrase_rank = False
stage_b_rank_margin = 0.3
stage_b_rank_loss_coef = 0.0
stage_b_rank_detach_patch = True
stage_b_score_calib_loss_coef = 0.0

use_tn_category_weights = True
default_tn_category_weight = 1.0
tn_balance_sampling = True
tn_balance_cap = 5.0

skip_tn_if_neg_overlaps_canonical = True
skip_ambiguous_tn = True
skip_tn_if_changed_span_not_found = True
skip_tn_if_changed_span_empty_after_filter = True
skip_relation_like_tn_in_v1 = False
```

RefCOCO/RefCOCO+/RefCOCOg positive and TN samples remain one annotation/sample. Positive and TN prompts are not merged into one dot-separated caption.
