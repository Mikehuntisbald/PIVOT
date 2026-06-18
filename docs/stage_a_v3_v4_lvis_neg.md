# Stage-A v3/v4 LVIS Negative-Category Probes

This note records the Stage-A v3/v4 recipes that test whether Stage A needs
stronger score calibration from verified negatives, without changing the
same-caliber validation protocol.

## Recipe Summary

Stage A v3 changes support-slot assignment:

```text
config/cfg_patch_stage_a_v3_all_gt_classes.py
support_use_all_gt_classes = True
support_min_count = 1
support_num_patches_min = 1
```

For multi-patch episodes, v3 removes random K support-class sampling. If an
image has N eligible annotated GT classes, it uses those N classes as support
slots, excluding LVIS `not_exhaustive_category_ids`.

Stage A v4 is v3 plus decoder auxiliary patch losses:

```text
config/cfg_patch_stage_a_v4_all_gt_aux.py
aux_loss = True
```

v4 is Stage-A-only. Stage-B configs keep their own aux settings and ablations.

## LVIS Negative Subset

The v3/v4 dataset recipe is:

```text
config/datasets_patch_stage_a_v3_v4_lvis_neg025_local.json
```

It contains three train sources:

```text
LVIS normal:        mix_weight = 1.0
COCO normal:        mix_weight = 1.0
LVIS neg_category:  mix_weight = 0.6666666667
```

With the existing non-distributed `WeightedRandomSampler`, the expected LVIS
negative subset fraction is:

```text
0.6666666667 / (1.0 + 1.0 + 0.6666666667) = 0.25
```

The negative source sets `lvis_neg_category_only = true`. During dataset
construction it is filtered into a true subset: an image is kept only if it has
at least one LVIS `neg_category_ids` class that is absent from the current
annotations, not marked non-exhaustive, and covered by the support patch bank.

## Loss Semantics

For an LVIS negative-category episode:

- support slots come from eligible `neg_category_ids`;
- each slot gets a patch from the support patch bank;
- the query target boxes and labels are set to empty tensors;
- Hungarian matching returns no positive assignments;
- patch focal CE sees an all-zero query-slot target matrix and pushes all valid
  query-slot scores negative.

This is intentionally different from multi-patch wrong-slot negatives. Wrong
slots still coexist with positive slots in a normal image. LVIS negative-category
episodes are pure no-object-for-these-slots supervision from LVIS verified
negative metadata.

## Commands

Stage A v3:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/cfg_patch_stage_a_v3_all_gt_classes.py \
  --datasets config/datasets_patch_stage_a_v3_v4_lvis_neg025_local.json \
  --output_dir outputs/stageA_v3_all_gt_lvis_neg025 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --num_workers 8 \
  --amp
```

Stage A v4:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/cfg_patch_stage_a_v4_all_gt_aux.py \
  --datasets config/datasets_patch_stage_a_v3_v4_lvis_neg025_local.json \
  --output_dir outputs/stageA_v4_all_gt_aux_lvis_neg025 \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --num_workers 8 \
  --amp
```

## Evaluation

Evaluate v3/v4 with the existing Stage-A-caliber validation dataset:

```text
config/datasets_patch_stage_a_lvis_coco2017_eval_local.json
```

Do not evaluate on the negative training subset. The question is whether the
extra verified-negative training signal improves the usual LVIS/COCO patch AP50,
box recall, matched-query recall, and score ranking under the shared validation
episodes.

## Caveats

`mix_weight` sampling currently works only in non-distributed training. In
distributed training, `main.py` raises when explicit `mix_weight` values are
present.

The LVIS annotation meta cache version was bumped so cached metadata is rebuilt
with `neg_category_ids`. If a run does not log the negative subset size, check
that the expected LVIS annotation file and support patch bank are being used.
