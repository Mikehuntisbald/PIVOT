
- [ODVG Dataset Format](#odvg-dataset-format)
  - [Label map](#label-map)
- [Config file](#config-file)

# ODVG Dataset Format

The files are in jsonl format, with one json object per line, as follows:
- Object Detection datasets utilize the ``detection`` field. If dealing with an Object Detection dataset, an additional ``label_map`` is required in the Dataset settings.
- Visual Grounding datasets employ the ``grounding`` field.  

You can refer to the [tools](./tools/) to convert other anno formats to ovdg format.
```json
{
  "filename": "image.jpg",
  "height": 693,
  "width": 1024,
  "detection": {
      "instances": [     
        {
          "bbox": [262,210,323,338],   # [x1,y1,x2,y2]
          "label": 0,
          "category": "dog"
        },
        {
          "bbox": [164,263,252,371],
          "label": 1,
          "category": "cat"
        },
        {
          "bbox": [4,243,66,373],
          "label": 2,
          "category": "apple"
        }
      ]
  },
  "grounding": { 
      "caption": "a wire hanger with a paper cover that reads we heart our customers", 
      "regions": [
        {
          "bbox": [20,215,985,665],   # [x1,y1,x2,y2]
          "phrase": "a paper cover that reads we heart our customers"
        },
        { 
          "bbox": [19,19,982,671],
          "phrase": "a wire hanger"
        }
      ]
    }
}
```

## Label map

- In order to align with VG data, we need to provide an additional mapping table for OD data.
- In dictionary form, indices start from "0" (it is essential to start from 0 to accommodate caption/grounding data). [Here](./config/coco2017_label_map.json) is an example for dataset:

```json
{"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}
```

# Config file

- config spec:
  - The ``train`` supports multiple datasets for simultaneous training, and ``dataset_model`` needs to be set to ``odvg``. 
  - The ``val``  only supports datasets in the COCO format now, so ``dataset_model`` should be set to ``coco``, and ``label_map`` should be set to null.
- config example:
  - [datasets_mixed_odvg.json](config/datasets_mixed_odvg.json)
  - [datasets_od_example.json](config/datasets_od_example.json)
  - [datasets_vg_example.json](config/datasets_vg_example.json)

```json
{
  "train": [
    {
      "root": "path/coco_2017/train2017/",
      "anno": "path/coco_2017/annotations/coco2017_train_odvg.jsonl",
      "label_map": "path/coco_2017/annotations/coco2017_label_map.json",
      "dataset_mode": "odvg"
    }, 
    {
      "root": "path/GRIT-20M/data/",
      "anno": "path/GRIT-20M/anno/grit_odvg_10k.jsonl",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "path/coco_2017/val2017",
      "anno": "config/instances_val2017.json",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
```

# Development record: patch episode / Stage-B TN

This section records the local data lineage and design evolution for the
patch-episode and Stage-B RefCOCO/TN training pipeline. It is based on the
current scripts, configs, and git history in `/media/haoyi/T9/gdino`,
`/media/haoyi/T9/data/data_proc`, and `/media/haoyi/T9/data/SAM3`. It is a
development record, not a claim that every command below was rerun end to end in
the current session.

## Data lineage overview

The Stage-B local TN data is produced in several layers:

1. Canonical vocabulary cleaning under `/media/haoyi/T9/data/data_proc`.
2. VG / RefCOCO phrase-to-head / phrase-to-class extraction under
   `/media/haoyi/T9/data/data_proc`.
3. RefCOCO box washing and TN text generation under `/media/haoyi/T9/data/SAM3`.
4. SAM3 / VLM filtering of TN candidates under `/media/haoyi/T9/data/SAM3`.
5. Conversion into patch-episode prebuilt JSONL under
   `/media/haoyi/T9/data/patch_episode_prebuilt`.
6. Stage-B training with LVIS, COCO, positive RefCOCO+, positive RefCOCOg, and
   RefCOCO TN data mixed by `config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json`.

The important persistent artifacts are:

- `/media/haoyi/T9/data/canonical_classes_with_aliases.json`
- `/media/haoyi/T9/data/data_proc/vg_text_pairs.jsonl`
- `/media/haoyi/T9/data/data_proc/vg_text_pairs_clean.jsonl`
- `/media/haoyi/T9/data/data_proc/refcoco_text_pairs/*_pairs.jsonl`
- `/media/haoyi/T9/data/SAM3/out/*_sam3_washed.jsonl`
- `/media/haoyi/T9/data/SAM3/out/*_sam3_washed_try_tn_llm_head.jsonl`
- `/media/haoyi/T9/data/SAM3/output/*_candidates_vlm_filter/accepted.jsonl`
- `/media/haoyi/T9/data/patch_episode_prebuilt/refcocoplus_stageb_phrase_v1.jsonl`
- `/media/haoyi/T9/data/patch_episode_prebuilt/refcocog_stageb_phrase_v1.jsonl`
- `/media/haoyi/T9/data/patch_episode_prebuilt/refexp_tn_stageb_v1.jsonl`

## Canonical vocabulary cleaning

The canonical class vocabulary is generated by
`/media/haoyi/T9/data/data_proc/wash.py`.

Its job is to merge LVIS, COCO, TAO, and VAW names into one canonical class
table:

- LVIS is used as the base class space.
- COCO / TAO / VAW aliases are mapped into the same class space by exact match,
  normalized-name match, synonym / lemma match, semantic similarity, and optional
  LLM judgment.
- Unmatched aliases can become new canonical object classes when the LLM judges
  them to be valid standalone object categories.
- The main output is `canonical_classes_with_aliases.json`; unmatched or
  rejected aliases are written separately for audit.

`/media/haoyi/T9/data/data_proc/after_wash.py` adds part / base-class metadata:

- It asks an LLM whether a canonical class is a part concept.
- For part concepts, it records `base_class`.
- For standalone concepts, `base_class` defaults to itself.

Design reason:

- Stage A/B both need a stable class id space shared by OD data, patch support
  patches, RefCOCO phrases, and TN examples.
- The `base_class` metadata gives later code a conservative way to reason about
  parts without exploding the training label space.

## VG and RefCOCO text-pair extraction

`/media/haoyi/T9/data/data_proc/build_text_pairs_from_vg_refcoco_batch.py` is
the main phrase cleaning script.

It uses:

- `canonical_classes_with_aliases.json`
- VG `region_descriptions.json`
- RefCOCO / RefCOCO+ / RefCOCOg `instances.json`
- RefCOCO `refs(...).p`
- a local OpenAI-compatible Qwen endpoint by default

For VG:

- It rough-matches region phrases to canonical classes.
- It balances candidate regions per canonical class with `VG_PER_CLASS_MAX`.
- It asks the LLM to extract a continuous `head_phrase` and `head` from each
  phrase.
- It maps the extracted head back to a canonical `class_id`.
- Invalid rows go to unmatched files.

For RefCOCO:

- It reads each referring expression.
- It extracts a continuous `head_phrase` and `head`.
- It keeps the COCO annotation identity and maps the category name into the
  canonical vocabulary.
- Outputs are written under `refcoco_text_pairs/*.jsonl`.

Important validation:

- `head_phrase` and `head` are required to be spans in the original phrase.
- `revalidate_existing_refcoco_text_pairs.py` can re-check existing RefCOCO pair
  files with stricter letter-bounded span checks and move invalid rows into
  unmatched outputs.

Additional cleanup:

- `clean_vg_text_pairs_headphrase.py` fixes common VG errors such as
  "person wearing / holding / riding object", where the extracted head can
  incorrectly become the interacted object instead of the person.
- `data_aug.py` adds simple template augmentation for low-frequency VG classes,
  with the local default target of at least 50 rows per class.
- `build_flickr30k_fused_clean_v5.py` is an optional Flickr30k Entities path for
  producing similar phrase/head/class rows.

Design reason:

- Stage-B text supervision needs token-level alignment to meaningful phrase
  spans. Requiring continuous spans makes token masks and audits tractable.
- VG supplies broad phrase diversity; RefCOCO supplies target boxes and
  referring-expression style language.
- The head cleanup is necessary because a wrong head corrupts both canonical
  class supervision and later TN construction.

## SAM3 RefCOCO washing

`/media/haoyi/T9/data/SAM3/wash_refcoco_with_sam3.py` cleans RefCOCO boxes with
SAM3.

For each expression:

- SAM3 is queried with the phrase.
- If SAM3 returns zero boxes, the original GT box is kept and marked as a zero
  result.
- If SAM3 returns one box, the script keeps the original GT when IoU is high
  enough; otherwise it can replace the box with the SAM3 box.
- If SAM3 returns multiple boxes, single-word phrases are dropped more
  conservatively; multi-word phrases are filtered by confidence and IoU.

Typical outputs:

- `out/refcoco_sam3_washed.jsonl`
- `out/refcocoplus_sam3_washed.jsonl`
- `out/refcocog_sam3_washed.jsonl`
- dropped-row and visualization directories

Design reason:

- RefCOCO annotations are useful, but box-language alignment is not always
  reliable for fine-grained text negatives.
- SAM3 washing removes or repairs cases where the expression and box are likely
  inconsistent before TN construction.

## TN text generation

There are two TN-generation paths under `/media/haoyi/T9/data/SAM3`.

`build_try_tn_from_washed.py` is the vocab/rule path:

- It reads SAM3-washed RefCOCO rows.
- It joins category and head information from
  `/media/haoyi/T9/data/data_proc/refcoco_text_pairs`.
- It detects replaceable color / clothing / accessory / attribute words with a
  curated vocabulary.
- It skips ambiguous replacements such as words that are common object heads or
  ambiguous modifiers.
- It can ask the local Qwen endpoint to polish the generated negative phrase.

`build_try_tn_llm_head_from_washed.py` is the LLM-first path:

- It reads `head_phrase` and `head` from the data-proc pair files.
- It asks the LLM to replace non-head attributes while preserving the head.
- It retries once when the proposed edit is invalid.
- It falls back to the vocab/rule replacement path when needed.
- It writes `*_sam3_washed_try_tn_llm_head.jsonl`.

The generated rows keep fields such as:

- original positive phrase
- `positive_phrase`
- `try_tn` / TN phrase
- `try_tn_head_phrase`
- `head` / `head_phrase`
- `replace_category`
- edit metadata

Design reason:

- The TN should usually preserve the object identity but change content that
  matters for referring expression grounding.
- Preserving the head prevents degenerate negatives that turn into a different
  class and become an OD-class problem instead of a phrase-composition problem.
- LLM-first generation improves naturalness; the rule fallback keeps coverage.

## SAM3 and VLM TN filtering

The TN candidates are filtered in two stages.

`sam3_try_tn.py` queries SAM3 with the TN phrase:

- If SAM3 confidently finds the TN object overlapping the original target, the
  row is not a useful true negative.
- Low-overlap TN detections can be kept as candidate negatives.

`sam3_prepare_candidates.py` prepares candidate regions:

- It reads `*_sam3_washed_try_tn_llm_head.jsonl`.
- It uses the SAM3 TensorRT bridge when configured.
- It chooses a target box conservatively, favoring the original GT fallback when
  SAM drift is detected.
- It prepares red-box crop candidates for later VLM verification.

`vlm_filter_try_tn.py` verifies candidates with a VLM:

- It asks whether the red-boxed target fully satisfies the TN phrase.
- If the head matches but any important modifier, relation, number, color,
  posture, pattern, or attached description is wrong, the answer should be
  `no`.
- Rows are split into `accepted.jsonl`, `rejected.jsonl`, `unknown.jsonl`, and
  `skipped.jsonl`.
- Stage-B TN training uses accepted negatives from
  `/media/haoyi/T9/data/SAM3/output/*_candidates_vlm_filter/accepted.jsonl`.

Design reason:

- SAM3 gives a geometry-based check for whether the negative phrase has an
  obvious matching object.
- The VLM filter catches semantic false negatives that geometry alone misses,
  especially relation, action, color, and attached-clause errors.
- Unknown rows are separated instead of forced into training labels.

## Stage-B prebuilt JSONL construction

`/media/haoyi/T9/gdino/tools/build_stageb_refexp_mix.py` converts the cleaned
RefCOCO/SAM3/TN artifacts into patch-episode JSONL files.

Default positive sources include:

- `/media/haoyi/T9/data/SAM3/out/refcocoplus_sam3_washed_try_tn_llm_head.jsonl`
- `/media/haoyi/T9/data/SAM3/out/refcocog_sam3_washed_try_tn_llm_head.jsonl`

Default TN sources include:

- `/media/haoyi/T9/data/SAM3/output/refcoco_sam3_washed_try_tn_llm_head_candidates_vlm_filter/accepted.jsonl`
- `/media/haoyi/T9/data/SAM3/output/refcocoplus_sam3_washed_try_tn_llm_head_candidates_vlm_filter/accepted.jsonl`
- `/media/haoyi/T9/data/SAM3/output/refcocog_sam3_washed_try_tn_llm_head_candidates_vlm_filter/accepted.jsonl`

Default outputs:

- `refcocoplus_stageb_phrase_v1.jsonl`
- `refcocog_stageb_phrase_v1.jsonl`
- `refexp_tn_stageb_v1.jsonl`

The builder can use the head classifier checkpoint
`/media/haoyi/T9/gdino/exp_vg_multiclass_clean/best.pt` to override RefCOCO
annotation categories from extracted heads.

Design reason:

- RefCOCO category labels can be too coarse or inconsistent with the extracted
  phrase head.
- The classifier-based override aligns support-class selection with the actual
  phrase head used by Stage-B.

## Head/classifier training

The small classifier is trained by
`/media/haoyi/T9/gdino/models/GroundingDINO/train_classifier_clean.py`.

The recorded command in the script is:

```bash
python train_classifier_clean.py \
  --train_jsonl /media/haoyi/T9/data/vg_text_pairs_clean.jsonl \
  --canonical_json /media/haoyi/T9/data/canonical_classes_with_aliases.json \
  --bert_model_name bert-base-uncased \
  --batch_size 256 \
  --epochs 8 \
  --max_len 24 \
  --val_ratio 0.05 \
  --output_dir ../../exp_vg_multiclass_clean \
  --use_head_phrase \
  --focal_gamma 2.0 \
  --lr 5e-5 \
  --lr_milestones 3 5 \
  --lr_gamma 0.1 \
  --early_stop_patience 3
```

Implementation details:

- Input rows come from `vg_text_pairs_clean.jsonl`.
- The model uses `bert-base-uncased`, `BertModelWarper`, dropout, and a linear
  classifier over canonical class ids.
- `--use_head_phrase` means the classifier prefers `head_phrase` over the full
  raw phrase.
- The loss is multiclass focal loss with `gamma=2.0`.
- The checkpoint is saved under `exp_vg_multiclass_clean`, with `best.pt` used
  by later data builders.

The classifier is used in two places:

- `_PhraseClassifierLabeler` in `datasets/patch_episode.py` and
  `offline_label_vg_regions.py` for labeling raw region phrases.
- `_HeadClassifierResolver` in `tools/build_stageb_refexp_mix.py` for RefCOCO
  head-based category override.

Design reason:

- LLM extraction gives a textual head; training still needs a canonical class id
  to choose support patches and labels.
- A local classifier is cheaper and more reproducible than asking an LLM for
  every downstream category resolution.
- Focal loss helps because VG-derived class frequencies are long-tailed.

## Stage-B training data mix

The current local dataset config is
`config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json`.

Training mix weights:

| source | artifact | mix weight |
| --- | --- | ---: |
| LVIS train | `LVIS/lvis_v1_train.json` | 2.0 |
| COCO train | `COCO/coco2017/annotations/instances_train2017.json` | 2.0 |
| RefCOCO+ positive | `patch_episode_prebuilt/refcocoplus_stageb_phrase_v1.jsonl` | 2.0 |
| RefCOCOg positive | `patch_episode_prebuilt/refcocog_stageb_phrase_v1.jsonl` | 2.0 |
| RefCOCO TN | `patch_episode_prebuilt/refexp_tn_stageb_v1.jsonl` | 1.0 |

Within TN prebuilt data, normalized group balancing is applied so that training
does not follow the raw `replace_category` distribution only. The current
coarse groups are:

- `color_like`
- `attr_like`
- `spatial_like`
- `relation_action_like`
- `other`

The TN sampler uses capped inverse-sqrt group weights:

```text
weight = min(cap, sqrt(max_group_count / group_count))
```

The per-row group weights are normalized within the dataset so the dataset-level
`mix_weight` remains the intended global mixture weight.

Design reason:

- Color-like edits are overrepresented in raw TN generation.
- Balanced TN sampling keeps spatial, relation, action, and less frequent
  attribute failures visible during training.
- Dataset-level mix weights are kept separate from within-TN balancing so
  ablations remain interpretable.

## Stage-B mask and loss design

The current Stage-B content-token design is:

- `phrase_to_token_mask`: full phrase span used for inference-style phrase
  scoring.
- `canonical_to_token_mask`: canonical/head noun tokens, trained positive.
- `content_to_token_mask`: non-canonical meaningful content tokens, trained
  positive.
- `changed_negative_token_mask`: TN-changed tokens, trained negative.
- `phrase_semantic_token_mask`: temporary compatibility alias for
  `content_to_token_mask`.
- `attr_pos_to_token_mask`: compatibility alias for content-positive tokens.
- `attr_neg_to_token_mask`: compatibility alias for TN-negative tokens.

Mask priority during token BCE:

```text
canonical > tn_neg > content_pos > ignore
```

`loss_text` is token BCE only:

```text
loss_text = canonical_loss + content_pos_loss + tn_neg_loss
```

The old softmin phrase TN rejection loss is disabled:

- `use_phrase_tn_loss = False`
- `lambda_phrase = 0.0`

Design reason:

- Canonical tokens should keep the object identity anchored.
- Content tokens should teach the model to use non-head words such as color,
  attributes, size, clothing, material, spatial words, relation/action words,
  posture, state, pattern, accessory, count, anchors, and context objects.
- TN changed tokens are local negative supervision; they should not define
  inference behavior by themselves.
- Spatial and relation prepositions such as `to`, `of`, `in`, `on`, `with`,
  `near`, `behind`, and `under` are kept because they are often essential to
  relation expressions.

## Phrase-ranking design

The current working-tree design adds an independent Stage-B phrase-ranking loss.

The score is aligned with inference:

```text
S(q, t, p) = S_patch(q, p) + beta * S_text(q, t)
```

For a TN pair:

```text
t+ = positive_phrase
t- = TN phrase
loss_phrase_rank = mean(max(0, margin - S_pos + S_neg))
```

Important constraints:

- `t+` comes only from `positive_phrase`.
- `try_tn_head_phrase` is not used as a fallback for phrase ranking.
- `loss_phrase_rank` is independent and is not added into `loss_text`.
- `stage_b_rank_loss_coef` controls the rank loss directly.
- Ranking uses match-by-target alignment:
  - compute matching for the negative forward;
  - compute matching again for the positive forward;
  - compare only the same original GT `target_id` and slot;
  - do not assume the same query index across the two forwards.
- The positive ranking forward disables patch DN randomness.
- `stage_b_rank_detach_patch=True` detaches the patch score in the rank loss by
  default.

The shared Stage-B scorer supports:

- `mean`
- `max`
- `softmin`
- `mean_norm_softmin`

The mixed scorer form is:

```text
alpha * mean_score + (1 - alpha) * normalized_softmin_score
```

Design reason:

- The main ranking signal should optimize the same phrase score used at
  inference.
- The changed-token mask remains useful as a local auxiliary loss, but inference
  must not depend on training-only masks.
- Match-by-target avoids the invalid assumption that positive and negative
  forwards choose the same query id.
- Detaching patch score prevents phrase ranking from destabilizing the support
  patch branch.
- Disabling patch DN in the ranking forward removes random DN-query mismatch
  between the negative and positive passes.

## Inference behavior

Stage-B postprocess should use inference-style phrase scoring only.

`PostProcessStageB.compute_slot_logits` uses `phrase_to_token_mask` and the
configured scorer. It folds canonical tokens into the same phrase-level
weighted sigmoid mean instead of adding a separate canonical score:

```text
text_score = weighted_mean(sigmoid(token_logits), phrase_tokens, canonical_weight)
slot_score = (sigmoid(patch_logit) + beta * text_score) / (1 + beta)
```

The default `canonical_weight` is `1.0`. Stage-B normalizes the fused score by
`1 + beta` by default; allTN thresholds for Stage-B should be calibrated on this
normalized score rather than inherited from text-only GDINO calibration.

The scorer must not read:

- `content_to_token_mask`
- `phrase_semantic_token_mask`
- `changed_negative_token_mask`

Design reason:

- Training-only masks are allowed to shape representations, but final scoring
  should stay consistent with demo/eval behavior.
- This avoids a train/eval mismatch where a mask available only in curated
  training rows changes the semantics of inference.

## Stage A, Stage B, and Stage AB training roles

Stage A is patch-centric:

- It trains support-patch matching and localization behavior.
- In the current ablation setup, later decoder layers can be unfrozen.
- The text is usually a generic object/canonical prompt rather than the
  fine-grained RefCOCO phrase.

Current local Stage-A foundation path:

```text
/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch
```

The checkpoint sequence in that directory is:

```text
checkpoint0000.pth  epoch 0
checkpoint0001.pth  epoch 1
checkpoint0002.pth  epoch 2
checkpoint0003.pth  epoch 3
checkpoint0004.pth  epoch 4
checkpoint0005.pth  epoch 5
checkpoint0006.pth  epoch 6
checkpoint.pth      latest, epoch 6
```

This sequence uses `config/cfg_patch_stage_a.py` with
`config/datasets_patch_stage_a_lvis_coco2017_local.json` for the current
LVIS+COCO Stage-A run. It should not be confused with older warmup / COCO-only
startup records that also exist in the same `info.txt` because the output
directory was reused.

Important checkpoint metadata:

- `checkpoint0000.pth` through `checkpoint0002.pth` used
  `patch_dn_num_queries=50` and `patch_dn_box_noise_scale=0.4`.
- `checkpoint0003.pth` through `checkpoint0006.pth` were resumed from
  `outputs/stageA_coco_multipatch/checkpoint.pth` and used
  `patch_dn_num_queries=1` and `patch_dn_box_noise_scale=1.0`.
- The core Stage-A setup stayed the same: `patch_only=True`,
  `patch_matching="hungarian"`, `support_num_patches_max=80`,
  `patch_labeling_mode="topk_iou"`, `patch_topk=50`,
  `patch_topk_iou_thr=0.04`, `patch_lambda_neg=0.25`,
  `unfreeze_decoder_last_n_layers=3`, `bbox_loss_coef=5.0`,
  `giou_loss_coef=2.0`, batch size `18`, and LR `1e-4`.

Stage B is phrase/TN-centric:

- It starts from a Stage-A checkpoint in the two-stage path.
- It keeps localization stable and adds phrase/content/TN supervision.
- Decoder and bbox behavior are kept conservative unless the config explicitly
  opens them.

Stage AB from OGC is the joint ablation path:

- It starts directly from `groundingdino_swint_ogc`.
- It uses Stage-A-style patch losses and Stage-B-style text/TN losses in one
  run.
- The config opens the last decoder layers and keeps bbox / GIoU coefficients
  aligned with Stage A.

OGC original-training finetune on Stage-A data is a separate baseline:

- It starts directly from `groundingdino_swint_ogc`.
- It converts the Stage-A LVIS/COCO patch-episode annotations into ODVG object
  detection records.
- It trains with the normal GroundingDINO ODVG objective: text-token focal
  classification plus Hungarian box/GIoU/L1 losses, with aux/interm losses from
  `cfg_odvg.py`.
- It does not use support patches, patch logits, Stage-B text/TN masks, or
  Stage-A patch losses.
- Its training exposure is matched by using the same Stage-A train split and
  the same number of completed epochs / image samples. Per-epoch eval is skipped
  during the exposure-matched run, and evaluation should be run separately for
  all checkpoints under the same protocol. GPU time is recorded as a resource
  metric, not used as the primary matching variable.

Design reason:

- Two-stage training isolates localization learning from fine-grained phrase
  rejection.
- Joint AB training tests whether the same behavior can emerge without a
  separate Stage-A pretraining phase.
- Keeping bbox coefficients aligned makes the ablation compare schedule and data
  structure rather than silently changing localization supervision.
- The OGC original-training finetune baseline tests whether ordinary
  GroundingDINO finetuning on the same Stage-A images/classes can explain the
  gains without exemplar-patch supervision.

## Ablation preparation

`tools/run_stageb_ablations.sh` prints the Stage-B ablation commands. It expects:

```bash
export STAGE_A_CKPT=/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch/checkpoint0006.pth
```

The prepared Stage-B ablation panel, separate from the GroundingDINO same-data
FT baseline, includes:

- `rank_loss_only`: keep TN rows and phrase-ranking supervision; disable
  token-level text BCE.
- `tn_token_only`: keep TN rows and token-level content/TN BCE; disable
  phrase-ranking supervision.
- `no_tn`: remove TN rows and disable phrase-ranking supervision; keep positive
  RefCOCO token supervision.

`tools/collect_stageb_ablation_results.py` collects metrics such as:

- `matched_query_recall50`
- `patch_recall50`
- `patch_ap50`
- `patch_map50`
- `refcocop_acc50`
- `refcocog_acc50`
- `tn_fpr`
- `fpr95tpr`

The Stage AB ablation config is
`config/ablations/cfg_stageab_from_ogc.py`.

The OGC original-training finetune baseline uses:

```text
config/ablations/cfg_ogc_original_finetune_stage_a.py
tools/build_stagea_odvg_finetune_ablation.py
tools/run_ogc_original_finetune_stage_a.sh
```

Typical run:

```bash
cd /media/haoyi/T9/gdino

STAGEA_DATASETS=/media/haoyi/T9/gdino/config/datasets_patch_stage_a_lvis_coco2017_local.json \
STAGE_A_LOG=/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch/log.txt \
PRETRAIN_MODEL_PATH=/media/haoyi/T9/gdino/weights/groundingdino_swint_ogc.pth \
CUDA_VISIBLE_DEVICES=0 \
tools/run_ogc_original_finetune_stage_a.sh
```

If the completed Stage-A epoch count is known explicitly, set
`MATCH_EPOCHS=<epochs>` instead of `STAGE_A_LOG`.

Stage-A-caliber evaluation uses the shared LVIS/COCO patch-episode val config:

```text
config/datasets_patch_stage_a_lvis_coco2017_eval_local.json
```

Evaluator roles:

- `tools/eval_stagea_patch_checkpoints.py` evaluates patch-only Stage-A
  checkpoints with support-patch logits.
- `tools/eval_text_stagea_caliber_checkpoints.py` evaluates ordinary
  GroundingDINO text checkpoints on the same target episodes with canonical text
  prompts built from `support_classes`.

The local recorded comparison is documented in
`docs/stage_a_caliber_eval.md`. In the current run:

- Stage-A patch-only `checkpoint0006.pth` is the best local checkpoint so far
  under mean `patch_ap50`.
- OGC same-data FT `checkpoint0001.pth` is slightly better than
  `checkpoint0002.pth` under the text Stage-A-caliber AP50 computation.

Design reason:

- `rank_loss_only` tests whether inference-aligned phrase ranking alone can
  explain TN gains.
- `tn_token_only` tests whether local changed-token supervision is sufficient
  without ranking.
- `no_tn` tests the positive RefCOCO Stage-B path without true-negative rows.
- Stage AB from OGC tests whether the staged recipe is necessary or whether a
  single joint run is sufficient.
- OGC original-training finetune tests whether standard GroundingDINO training
  on the same Stage-A training set is enough under the same sample exposure.

## Training reliability notes

The training loop has been adjusted for long Stage-B runs:

- `checkpoint_iter.pth` can save the current iteration state.
- Resume restores epoch, iteration, optimizer, LR scheduler, and AMP GradScaler.
- GradScaler is created once per run and restored on resume instead of being
  recreated per epoch.
- Mid-epoch resume may need to skip already-finished dataloader batches before
  loss logging resumes; this is expected but can be slow because the loader must
  advance through dataset transforms.

Design reason:

- Stage-B runs are long enough that interruption should not waste thousands of
  iterations.
- Keeping GradScaler continuous avoids a silent AMP behavior change after
  resume.

## Git-history evolution

Current tracked history:

| commit | summary | development meaning |
| --- | --- | --- |
| `515dda7` | Initial commit | Base GroundingDINO / patch-episode code and configs. |
| `6352313` | `feat(stage-b): add local TN supervision` | Introduced Stage-B local TN supervision, criterion, config, docs, and visualization support. |
| `c66ec6e` | `feat(refexp): override RefCOCO classes from heads` | Added RefCOCO head-based class override for Stage-B data building. |
| `50ab42d` | `chore(train): make resume explicit and stabilize logs` | Clarified resume behavior and stabilized train-time logging. |
| `f040cea` | `Add robust Stage B training resume utilities` | Added iteration checkpoint/resume utilities and Stage-B trainable/drift checks. |
| `9ad8cb2` | `Add Stage B ablation configs and helpers` | Added ablation configs, datasets, run script, and result collection. |
| `f0e5472` | `Implement Stage B content-token TN loss` | Replaced phrase-semantic TN semantics with content-token masks and token BCE TN supervision. |
| `49895f0` | `Update Stage B local TN documentation` | Updated the Stage-B TN documentation for the content-token loss design. |
| `4d2f922` | `Clean ablation config formatting` | Formatting cleanup for ablation configs. |

Current uncommitted working-tree development after `4d2f922` includes:

- shared Stage-B phrase scorer;
- independent `loss_phrase_rank`;
- positive phrase rank subbatch construction in `engine.py`;
- match-by-target ranking alignment;
- ranking positive forward with patch DN disabled;
- optional `mean_norm_softmin` phrase aggregation.

## Reproduction checklist

Recommended high-level order:

1. Build / verify `canonical_classes_with_aliases.json` with `data_proc/wash.py`.
2. Optionally annotate parts with `data_proc/after_wash.py`.
3. Build VG and RefCOCO text pairs with
   `data_proc/build_text_pairs_from_vg_refcoco_batch.py`.
4. Revalidate RefCOCO pair spans with
   `data_proc/revalidate_existing_refcoco_text_pairs.py`.
5. Clean VG head phrases with `data_proc/clean_vg_text_pairs_headphrase.py`.
6. Train the head classifier with
   `models/GroundingDINO/train_classifier_clean.py`.
7. Wash RefCOCO boxes with `SAM3/wash_refcoco_with_sam3.py`.
8. Generate TN text with `SAM3/build_try_tn_llm_head_from_washed.py`.
9. Prepare SAM3 candidates with `SAM3/sam3_prepare_candidates.py`.
10. Filter candidates with `SAM3/vlm_filter_try_tn.py`.
11. Build Stage-B prebuilt JSONL with `tools/build_stageb_refexp_mix.py`.
12. Train Stage A, Stage B, or Stage AB with the intended config.
13. Run sanity checks:

```bash
python -m py_compile \
  datasets/patch_episode.py \
  models/GroundingDINO/stage_b_criterion.py \
  models/GroundingDINO/groundingdino.py \
  engine.py \
  main.py

python tools/check_stageb_content_masks.py
```

## Caveats

- Data-generation scripts depend on local model services, SAM3 assets, TensorRT
  bridge paths, and large local datasets; exact reproduction requires the same
  local environment.
- JSONL artifacts under `/media/haoyi/T9/data` are data products, not normal git
  source files.
- Counts for TN categories and accepted/rejected rows should be taken from the
  current data files or training logs when reporting a specific experiment.
- Old names such as `attr_pos_to_token_mask`, `attr_neg_to_token_mask`, and
  `phrase_semantic_token_mask` may still exist as compatibility aliases. Their
  current semantics are content-positive, TN-negative, and content-token mask,
  respectively.
