# Stage B Local TN Supervision

Stage B treats a true-negative (TN) refexp as a local semantic conflict, not as a fully absent object. The matched query still corresponds to a real object in the image; only the visual attribute that was changed by the TN construction is supervised as false.

Example:

```text
positive: blue shirt man
TN:       white shirt man
```

The intended supervision is:

```text
white -> target 0
shirt -> target 1 if it is a verified shared visual token
man   -> target 1 with weak canonical/head weight
```

## Dataset Masks

`datasets/patch_episode.py` builds these text masks per support slot:

```text
phrase_to_token_mask
canonical_to_token_mask
attr_pos_to_token_mask
attr_neg_to_token_mask
relation_to_token_mask
phrase_semantic_token_mask
is_tn
attr_neg_weight_mask
```

`phrase_to_token_mask` is retained for alignment/debugging. It is not used directly as the text loss mask.

`attr_neg_to_token_mask` is built only from changed visual attribute tokens in `replace_to`. It is not the full `replace_to` span and it is not `phrase_to_token_mask & ~canonical_to_token_mask`.

`relation_to_token_mask` is constructed for audit/v2, but v1 does not use it for token loss or phrase loss.

## Changed Attribute Alignment

TN replacement metadata is aligned with token sequences, preserving character offsets in the normalized `replace_to` text:

```python
opcodes = SequenceMatcher(None, from_tokens, to_tokens).get_opcodes()
```

The dataset uses `replace` and `insert` opcodes from the `to_tokens` side as candidate negative tokens. Candidate tokens keep their local char offsets, then map into the TN phrase and finally into tokenizer positions.

This avoids set-difference bugs such as:

```text
positive: blue shirt with blue logo
TN:       white shirt with blue logo
```

Only `white` should become negative. The later `blue` in `blue logo` remains untouched.

After alignment, candidates are filtered:

```text
drop stopwords
drop punctuation-only tokens
drop relation/action/spatial words
drop tokens overlapping canonical/head
```

If a configured visual TN cannot find a changed span, or the changed span becomes empty after filtering, the slot is treated as invalid and can be resampled/audited.

## Category Policy

`replace_category` is normalized to one format:

```text
lowercase
strip
replace _, /, comma, and hyphen with spaces
collapse whitespace
```

Examples:

```text
hair_color    -> hair color
color/pattern -> color pattern
```

Known visual categories receive final negative weights in `attr_neg_weight_mask`:

```text
color family              1.5
size/material/clothing    1.3
shape/texture/height      1.2
state/condition           1.0
```

Unknown categories default to `0.0` and are skipped. Relation-like categories such as spatial, position, location, distance, action, posture, and pose are skipped in v1 and do not receive phrase rejection.

## Text Loss

`models/GroundingDINO/stage_b_criterion.py` uses slot-wise group means:

```python
L_attr_pos = mean_over_slots(mean_bce(pos_attr_tokens, 1))
L_attr_neg = mean_over_slots(mean_bce(neg_attr_tokens, 0, attr_neg_weight_mask))
L_canonical = mean_over_slots(mean_bce(canonical_tokens, 1))
```

The negative weight mask is the final per-token negative weight. It is not multiplied by another global negative scalar.

For TN slots, shared positive attributes use `tn_shared_attr_pos_weight`; canonical/head tokens use `canonical_pos_weight`.

## TN Phrase Rejection

Phrase rejection is TN-only. There is no positive phrase soft-min loss in v1.

A TN slot receives phrase rejection only when:

```text
is_tn == True
attr_neg_to_token_mask has at least one token
attr_neg_weight_mask.max() > 0
category is not relation-like/skipped
```

The phrase score is a soft-min over semantic tokens only:

```python
s = -tau * torch.logsumexp(-logits / tau, dim=-1)
L_phrase_tn = BCEWithLogits(s, 0)
```

The v1 semantic mask is:

```text
canonical_to_token_mask | attr_pos_to_token_mask | attr_neg_to_token_mask
```

It does not include `relation_to_token_mask`.

## Default Config

```python
tn_loss_profile = "standard"

attr_pos_weight = 1.0
tn_shared_attr_pos_weight = 0.75
canonical_pos_weight = 0.15

use_tn_category_weights = True
default_tn_category_weight = 0.0

use_phrase_tn_loss = True
phrase_score_type = "softmin"
softmin_tau = 0.7
lambda_phrase = 0.3

skip_tn_if_neg_overlaps_canonical = True
skip_ambiguous_tn = True
skip_tn_if_changed_span_not_found = True
skip_tn_if_changed_span_empty_after_filter = True
skip_relation_like_tn_in_v1 = True
```

## V1 Exclusions

These are intentionally not part of v1 local TN supervision:

```text
canonical/head replacement TN
relation/action/spatial TN token negatives
relation/action/spatial TN phrase rejection
paired positive-vs-TN ranking loss
```
