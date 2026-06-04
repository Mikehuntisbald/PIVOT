# Stage-A-Caliber Evaluation

This note records the local Stage-A-caliber evaluation protocol used for both
patch-only Stage A checkpoints and the original GroundingDINO same-data
finetune baseline.

## Scope

The shared validation set is:

```text
config/datasets_patch_stage_a_lvis_coco2017_eval_local.json
```

It contains `lvis_val` and `coco_val` patch-episode entries with:

- `keep_only_patchset_gt = true`
- `support_num_patches_max = 80`
- `neg_episode_prob = 0.0`
- `patch_text_augment = false`
- `build_text_token_masks = false`

The primary metric is mean `patch_ap50` over `lvis_val` and `coco_val`. For the
text GroundingDINO baseline this is the same AP50 computation, but detections are
ranked by text class scores instead of support-patch scores.

## Patch-Only Stage A

Evaluator:

```text
tools/eval_stagea_patch_checkpoints.py
```

Command used for the recorded `checkpoint0002.pth` and `checkpoint0004.pth`
comparison:

```bash
DATA_ROOT=/media/haoyi/T9/data \
/home/haoyi/miniconda/envs/cvpr/bin/python tools/eval_stagea_patch_checkpoints.py \
  --config config/cfg_patch_stage_a.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_eval_local.json \
  --ckpts \
    outputs/stageA_coco_multipatch/checkpoint0002.pth \
    outputs/stageA_coco_multipatch/checkpoint0004.pth \
  --output_dir outputs/stageA_coco_multipatch_eval_0002_0004_fast \
  --batch_size 28 \
  --num_workers 8 \
  --log_every 25 \
  --amp
```

Recorded result:

| rank | checkpoint | mean patch_ap50 | mean box_recall@50 | mean matched_query_recall@50 | lvis_val patch_ap50 | coco_val patch_ap50 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `checkpoint0004.pth` | 0.590184 | 0.869767 | 0.830631 | 0.478985 | 0.701384 |
| 2 | `checkpoint0002.pth` | 0.247510 | 0.737447 | 0.662394 | 0.180233 | 0.314787 |

Under this Stage-A patch-only protocol, `checkpoint0004.pth` is the stronger
Stage-A checkpoint among the evaluated pair.

## Original GroundingDINO Same-Data FT

Training helper:

```text
tools/build_stagea_odvg_finetune_ablation.py
tools/run_ogc_original_finetune_stage_a.sh
config/ablations/cfg_ogc_original_finetune_stage_a.py
```

The helper converts the Stage-A LVIS/COCO patch-episode train split to ODVG
object detection records and trains from `groundingdino_swint_ogc` with the
normal GroundingDINO ODVG objective. It does not use support patches, patch
logits, Stage-B TN masks, or Stage-A patch losses.

The generated ODVG files are local experiment artifacts under:

```text
data/ablations/ogc_original_finetune_stage_a/
```

These files are intentionally ignored by git.

Text evaluator:

```text
tools/eval_text_stagea_caliber_checkpoints.py
```

This evaluator builds each prompt from `support_classes` and
`canonical_classes_json`, not from patch-episode `cap_list`. That keeps the text
baseline prompt aligned to canonical class ids even when patch episode text is
generic or sampled for patch-only training.

Command used for `checkpoint0001.pth` and `checkpoint0002.pth`:

```bash
TOKENIZERS_PARALLELISM=false \
DATA_ROOT=/media/haoyi/T9/data \
/home/haoyi/miniconda/envs/cvpr/bin/python tools/eval_text_stagea_caliber_checkpoints.py \
  --config outputs/ogc_original_finetune_stage_a/cfg_ogc_original_finetune_stage_a.generated.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_eval_local.json \
  --ckpts \
    outputs/ogc_original_finetune_stage_a/checkpoint0001.pth \
    outputs/ogc_original_finetune_stage_a/checkpoint0002.pth \
  --output_dir outputs/ogc_original_finetune_stage_a_eval_stagea_caliber_e1_e2 \
  --batch_size 32 \
  --num_workers 8 \
  --log_every 25 \
  --amp
```

`batch_size=64` was unstable on the local RTX 5090 run. `batch_size=32` used
about 28-32 GB and completed the evaluation.

Recorded result:

| rank | checkpoint | mean patch_ap50 | mean box_recall@50 | coco_val patch_ap50 | lvis_val patch_ap50 |
|---:|---|---:|---:|---:|---:|
| 1 | `checkpoint0001.pth` | 0.681998 | 0.873199 | 0.723879 | 0.640117 |
| 2 | `checkpoint0002.pth` | 0.678295 | 0.869198 | 0.721198 | 0.635392 |

Under this text same-data FT protocol, `checkpoint0001.pth` is slightly better
than `checkpoint0002.pth`.

## Interpretation Notes

- Stage-A patch-only and text GroundingDINO FT share the same patch-episode
  validation images and target support-class episodes.
- The `patch_ap50` name is kept for table compatibility. For Stage A it is AP50
  from patch logits; for text FT it is AP50 from text class scores.
- Stage-A patch-only also reports `matched_query_recall@K` because the patch
  criterion can expose matched-query behavior. The text FT evaluator does not
  report this metric.
- The local `outputs/.../summary.json` files remain the source for full per-dataset
  metrics such as `mean_best_iou@K`, target counts, runtime, and support coverage.
