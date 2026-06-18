# Stage-A-Caliber Evaluation

This note records the local Stage-A-caliber evaluation protocol used for both
patch-only Stage A checkpoints and the original GroundingDINO same-data
finetune baseline.

The reproducible Stage A/B training and evaluation runbook is
`docs/stage_ab_runbook.md`.

Stage A commands have separate model and dataset inputs:

```text
model/training config: config/cfg_patch_stage_a.py
train dataset config:  config/datasets_patch_stage_a_lvis_coco2017_local.json
eval dataset config:   config/datasets_patch_stage_a_lvis_coco2017_eval_local.json
```

Stage-A v3/v4 experimental recipes are documented in
`docs/stage_a_v3_v4_lvis_neg.md`. They are not replacements for the recorded
`checkpoint0006.pth` foundation until they beat it under this same-caliber
evaluation.

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

Local Stage-A foundation directory:

```text
outputs/stageA_coco_multipatch/
```

Recorded checkpoints:

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

The recorded checkpoint sequence is staged. The checkpoint `args` field is the
source of truth, because the same `config/cfg_patch_stage_a.py` path was used
with different effective knobs across time.

| checkpoint range | saved config/dataset path | init/resume evidence | key saved-args difference |
|---|---|---|---|
| `checkpoint0000.pth` - `checkpoint0002.pth` | `config/cfg_patch_stage_a.py`, `config/datasets_patch_stage_a_lvis_coco2017_local.json` | `resume=""`, `pretrain_model_path=weights/groundingdino_swint_ogc.pth` | phase-1 DN: `patch_dn_num_queries=50`, `patch_dn_box_noise_scale=0.4`, `patch_sanity_interval=250` |
| `checkpoint0003.pth` - `checkpoint0004.pth` | same paths | `resume=outputs/stageA_coco_multipatch/checkpoint.pth`, `start_epoch=3` | phase-2 DN: `patch_dn_num_queries=1`, `patch_dn_box_noise_scale=1.0`, `patch_sanity_interval=500` |
| `checkpoint0005.pth` - `checkpoint0006.pth` | same paths plus `--options epochs=7` | resumed from `checkpoint.pth` / `checkpoint_iter.pth` on 2026-06-05 | phase-2 continuation: same `patch_dn_num_queries=1`, `patch_dn_box_noise_scale=1.0` |

This means `checkpoint0002.pth` versus `checkpoint0004.pth` is not a pure epoch
comparison; it also crosses a Stage-A config boundary. The same-caliber eval
table below is still valid for model selection, but this lineage matters when
reproducing training.

The current phase-2/mainline core setup is:

```text
datasets: config/datasets_patch_stage_a_lvis_coco2017_local.json
patch_only: true
patch_matching: hungarian
support_num_patches_max: 80
patch_ce_reduction: legacy
patch_rank_loss_coef: 0.0
patch_labeling_mode: topk_iou
patch_topk: 50
patch_topk_iou_thr: 0.04
patch_lambda_neg: 0.25
unfreeze_decoder_last_n_layers: 3
batch_size: 18
lr: 1e-4
```

Current Stage-A v3/v4 probe knobs:

```text
v3 config: config/cfg_patch_stage_a_v3_all_gt_classes.py
v4 config: config/cfg_patch_stage_a_v4_all_gt_aux.py
dataset:   config/datasets_patch_stage_a_v3_v4_lvis_neg025_local.json
```

v3 removes random multi-patch support-class K sampling: each query image uses
all eligible annotated GT classes as support slots. v4 is v3 plus patch-only
decoder auxiliary losses. The shared v3/v4 dataset adds an LVIS
`neg_category_ids` negative subset with expected training sample fraction 0.25.

Evaluator:

```text
tools/eval_stagea_patch_checkpoints.py
```

Command used for the recorded `checkpoint0005.pth` and `checkpoint0006.pth`
comparison:

```bash
DATA_ROOT=/media/haoyi/T9/data \
/home/haoyi/miniconda/envs/cvpr/bin/python tools/eval_stagea_patch_checkpoints.py \
  --config config/cfg_patch_stage_a.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_eval_local.json \
  --ckpts \
    outputs/stageA_coco_multipatch/checkpoint0005.pth \
    outputs/stageA_coco_multipatch/checkpoint0006.pth \
  --output_dir outputs/stageA_coco_multipatch_eval_0005_0006_fast \
  --batch_size 28 \
  --num_workers 8 \
  --log_every 25 \
  --amp
```

Recorded result:

| rank | checkpoint | mean patch_ap50 | mean box_recall@50 | mean matched_query_recall@50 | lvis_val patch_ap50 | coco_val patch_ap50 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `checkpoint0006.pth` | 0.596751 | 0.871689 | 0.833839 | 0.488652 | 0.704849 |
| 2 | `checkpoint0005.pth` | 0.594680 | 0.870034 | 0.831887 | 0.485812 | 0.703548 |
| 3 | `checkpoint0004.pth` | 0.590184 | 0.869767 | 0.830631 | 0.478985 | 0.701384 |
| 4 | `checkpoint0002.pth` | 0.247510 | 0.737447 | 0.662394 | 0.180233 | 0.314787 |

Under this Stage-A patch-only protocol, `checkpoint0006.pth` is the strongest
Stage-A checkpoint among the evaluated local sequence so far.

## Stage-A Decoder-Unfreeze Ablation

Status: negative result, not Stage-A mainline. This test asked whether starting
from `checkpoint0002.pth`, unfreezing all decoder layers, and keeping the phase-2
recipe without an epoch-4 LR drop could beat the original Stage A
`checkpoint0004.pth`.

Config and output:

```text
config/ablations/cfg_stagea_phase2_decoder_all_nolrdrop.py
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_from0002_v2/
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_eval_0003_vs_orig/
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_eval_0004_vs_orig/
```

Effective ablation knobs:

```text
source checkpoint = outputs/stageA_coco_multipatch/checkpoint0002.pth
phase-2 DN settings = same as checkpoint0003/checkpoint0004
unfreeze_decoder_last_n_layers = 6
lr_drop = 100
```

Training evidence confirms that the run did not drop LR at epoch 4:

```text
Epoch 4 start lr = 0.000100
Epoch 4 averaged lr = 0.000100
checkpoint0004.pth = outputs/stageA_coco_multipatch_decoder_all_nolrdrop_from0002_v2/checkpoint0004.pth
```

Same-caliber result:

| checkpoint | branch | mean patch_ap50 | mean box_recall@50 | mean matched_query_recall@50 | lvis_val patch_ap50 | coco_val patch_ap50 |
|---|---|---:|---:|---:|---:|---:|
| `checkpoint0003.pth` | original | 0.581870 | 0.867450 | 0.829550 | 0.468977 | 0.694764 |
| `checkpoint0003.pth` | decoder-all no-drop | 0.576159 | 0.859027 | 0.816437 | 0.461330 | 0.690988 |
| `checkpoint0004.pth` | original | 0.590184 | 0.869767 | 0.830631 | 0.478985 | 0.701384 |
| `checkpoint0004.pth` | decoder-all no-drop | 0.577569 | 0.861146 | 0.818542 | 0.462768 | 0.692369 |

Decision: do not continue this branch as the Stage-A foundation. Unfreezing all
decoder layers from `checkpoint0002.pth` with no LR drop underperformed the
original checkpoint at both `checkpoint0003.pth` and `checkpoint0004.pth` on
both LVIS and COCO.

## Deprecated Stage-A v2 Loss Ablations

Status: deprecated for the Stage-A mainline. The Stage-A v2 loss code remains
available only to reproduce negative ablations; it is not enabled by the default
Stage-A config and should not be used as the Stage-A foundation.

The default `config/cfg_patch_stage_a.py` keeps:

```text
patch_ce_reduction = "legacy"
patch_rank_loss_coef = 0.0
patch_ce_neg_topk = 0
patch_ce_neg_topk_ratio = 0.0
```

The additional Stage-A losses are only activated by explicit ablation configs:

```text
config/cfg_patch_stage_a_v2_rank.py
config/cfg_patch_stage_a_v2_posneg_topk32_lam4.py
config/cfg_patch_stage_a_v2_rank_posneg_topk.py
```

Probe comparisons from `checkpoint0004.pth` showed no support for promoting
these losses into the Stage-A mainline:

| ablation | output | mean patch_ap50 | delta vs probe control | mean box_recall@50 | mean matched_query_recall@50 |
|---|---|---:|---:|---:|---:|
| probe control `checkpoint0004.pth` | `outputs/stageA_coco_multipatch_v2_rank_only_legacy_probe_iter0802_vs_0004` | 0.600179 | 0.000000 | 0.878200 | 0.833481 |
| legacy CE + rank only | `outputs/stageA_coco_multipatch_v2_rank_only_legacy_probe_iter0802_vs_0004` | 0.600012 | -0.000166 | 0.874776 | 0.824930 |
| pos/neg top-k32 CE, lambda_neg=4, no rank | `outputs/stageA_coco_multipatch_v2_posneg_topk32_lam4_norank_probe_iter0802_vs_0004` | 0.584910 | -0.015269 | 0.873120 | 0.807626 |
| rank + pos/neg top-k16 CE, lambda_neg=8 | `outputs/stageA_coco_multipatch_v2_rank_posneg_topk16_lam8_probe_iter0802_vs_0004` | 0.563079 | -0.037100 | 0.861275 | 0.782602 |

The longer `rank + pos/neg` continuation also underperformed:

```text
outputs/stageA_coco_multipatch_v2_rank_posneg_eval_0006_fast
mean patch_ap50 = 0.506181
mean box_recall@50 = 0.843147
mean matched_query_recall@50 = 0.763575
```

Decision: Stage-A v2 is deprecated for mainline training. Keep Stage A as the
legacy patch foundation and treat Stage-A rank / pos-neg CE as ablation code
only. Ranking and calibration improvements should be evaluated in Stage B or
inference-time fusion unless a future Stage-A ablation beats the legacy control
under the same checkpoint and dataset caliber.

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
