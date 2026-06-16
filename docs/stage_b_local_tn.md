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
