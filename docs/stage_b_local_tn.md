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

There is no softmin phrase rejection loss in v2. The old knobs `attr_pos_weight`, `tn_shared_attr_pos_weight`, `attr_neg_weight`, `use_phrase_tn_loss`, `phrase_score_type`, `softmin_tau`, and `lambda_phrase` are deprecated and ignored by the content-token loss.

## Inference

`PostProcessStageB.compute_slot_logits` does not use `content_to_token_mask` or `phrase_semantic_token_mask`. Inference scoring uses phrase-level token spans from `phrase_to_token_mask`, with the canonical mask only for the configured canonical contribution.

Training-only content masks therefore cannot change demo/postprocess scoring.

## Default Config

```python
tn_loss_profile = "standard"
canonical_pos_weight = 0.15

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
