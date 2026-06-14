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
The v4 loss uses the matched positive/TN queries and the top-10 competing query
scores:

```text
loss_pos       = softplus(tau_pos - S_pos)
loss_neg       = mean_top10 softplus(S_tn_topk - tau_neg)
loss_gap       = softplus(margin - S_pos + S_tn_matched)
loss_pos_query = mean_top10 softplus(margin - S_pos + S_pos_other_topk)
```

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
