# Stage A/B Reproducible Runbook

This runbook records the local two-stage patch-episode workflow. It is intended
to reproduce the current Stage A and Stage B experiments with explicit paths,
resume policy, and evaluation caliber.

## Environment

Use the repo root as the working directory:

```bash
cd /media/haoyi/T9/gdino

export GDINO_ROOT=/media/haoyi/T9/gdino
export DATA_ROOT=/media/haoyi/T9/data
export PY=/home/haoyi/miniconda/envs/cvpr/bin/python
export TOKENIZERS_PARALLELISM=false
```

Required local inputs:

```text
weights/groundingdino_swint_ogc.pth
${DATA_ROOT}/LVIS/lvis_v1_train.json
${DATA_ROOT}/LVIS/lvis_v1_val.json
${DATA_ROOT}/COCO/coco2017/annotations/instances_train2017.json
${DATA_ROOT}/COCO/coco2017/annotations/instances_val2017.json
${DATA_ROOT}/patches_quality_emb/emb_index_from_quality.tsv
${DATA_ROOT}/patches_quality/
${DATA_ROOT}/canonical_classes_with_aliases.json
```

Stage B additionally requires:

```text
${DATA_ROOT}/patch_episode_prebuilt/refcocoplus_stageb_phrase_v1.jsonl
${DATA_ROOT}/patch_episode_prebuilt/refcocog_stageb_phrase_v1.jsonl
${DATA_ROOT}/patch_episode_prebuilt/refexp_tn_stageb_v1.jsonl
```

## Checkpoint Policy

There are two different checkpoint modes:

- Use `--pretrain_model_path` when starting a new stage from an upstream model
  or from a completed Stage A checkpoint. This loads model weights only.
- Use `--resume` when continuing the same run. This restores model, optimizer,
  scheduler, AMP scaler, epoch, iteration, and RNG state when present.

Do not use `--resume` to initialize a new Stage B run from Stage A. Use
`--pretrain_model_path` for that.

If an output directory already contains `checkpoint.pth` and `--resume` is not
passed, `main.py` does not auto-resume. Use a fresh output directory for clean
reproduction, or pass `--resume` explicitly.

Use `checkpoint_iter.pth` for interrupted mid-epoch continuation. Use
`checkpoint.pth` or `checkpointXXXX.pth` for completed epoch checkpoints.

## Stage A Mainline

Stage A trains the patch foundation. The current mainline keeps legacy dense
patch CE and disables the deprecated Stage-A v2 rank / pos-neg CE losses.

Key files:

```text
config/cfg_patch_stage_a.py
config/datasets_patch_stage_a_lvis_coco2017_local.json
config/datasets_patch_stage_a_lvis_coco2017_eval_local.json
outputs/stageA_coco_multipatch/
```

Stage A commands have separate model and dataset inputs:

```text
model/training config: config/cfg_patch_stage_a.py
train dataset config:  config/datasets_patch_stage_a_lvis_coco2017_local.json
eval dataset config:   config/datasets_patch_stage_a_lvis_coco2017_eval_local.json
```

`cfg_patch_stage_a.py` controls model/loss/freezing/training knobs. The dataset
JSON controls which LVIS/COCO patch-episode split is loaded by `--datasets`.
Training uses the `*_local.json` file; same-caliber validation uses the
`*_eval_local.json` file. This input split is separate from the checkpoint
lineage boundary below. Do not treat deprecated Stage-A v2 configs as mainline
Stage-A configs.

Mainline loss switches:

```text
patch_ce_reduction = "legacy"
patch_rank_loss_coef = 0.0
patch_ce_neg_topk = 0
patch_ce_neg_topk_ratio = 0.0
```

### Recorded Stage A Checkpoint Lineage

The recorded `outputs/stageA_coco_multipatch` checkpoints were not produced by
one immutable config from epoch 0 through epoch 6. The checkpoint `args` field is
the source of truth; the same `config/cfg_patch_stage_a.py` path was used after
the file's effective knobs had changed.

| checkpoint range | saved config/dataset path | init/resume evidence | key saved-args difference |
|---|---|---|---|
| `checkpoint0000.pth` - `checkpoint0002.pth` | `config/cfg_patch_stage_a.py`, `config/datasets_patch_stage_a_lvis_coco2017_local.json` | `resume=""`, `pretrain_model_path=weights/groundingdino_swint_ogc.pth` | phase-1 DN: `patch_dn_num_queries=50`, `patch_dn_box_noise_scale=0.4`, `patch_sanity_interval=250` |
| `checkpoint0003.pth` - `checkpoint0004.pth` | same paths | `resume=outputs/stageA_coco_multipatch/checkpoint.pth`, `start_epoch=3` | phase-2 DN: `patch_dn_num_queries=1`, `patch_dn_box_noise_scale=1.0`, `patch_sanity_interval=500` |
| `checkpoint0005.pth` - `checkpoint0006.pth` | same paths plus `--options epochs=7` | resumed from `checkpoint.pth` / `checkpoint_iter.pth` on 2026-06-05 | phase-2 continuation: same `patch_dn_num_queries=1`, `patch_dn_box_noise_scale=1.0` |

So `checkpoint0002.pth` versus `checkpoint0004.pth` crosses a training-config
boundary, while `checkpoint0004.pth` through `checkpoint0006.pth` are the same
phase-2 core recipe with later continuation.

Recorded best local Stage A foundation:

```text
outputs/stageA_coco_multipatch/checkpoint0006.pth
outputs/stageA_coco_multipatch/checkpoint.pth
```

`checkpoint0006.pth` is the current best Stage A patch-only checkpoint under
the LVIS/COCO Stage-A-caliber evaluation.

### Start Stage A From OGC

For a clean reproduction, prefer a new output directory. The command below is
the current phase-2/mainline recipe; it does not claim that the historical
`checkpoint0000.pth` through `checkpoint0006.pth` lineage was a single unchanged
from-scratch config.

```bash
export STAGE_A_OUT=outputs/stageA_coco_multipatch

CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/cfg_patch_stage_a.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_local.json \
  --output_dir "${STAGE_A_OUT}" \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --num_workers 8 \
  --amp \
  --options epochs=7
```

Current phase-2/mainline Stage A core settings:

```text
batch_size = 18
lr = 1e-4
patch_matching = "hungarian"
support_num_patches_max = 80
patch_labeling_mode = "topk_iou"
patch_topk = 50
patch_topk_iou_thr = 0.04
patch_lambda_neg = 0.25
patch_dn_num_queries = 1
patch_dn_box_noise_scale = 1.0
unfreeze_decoder_last_n_layers = 3
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
```

The current output directory also contains older warmup / COCO-only startup
records in `info.txt`. Treat `config_args_all.json`, `log.txt`, and checkpoint
metadata from the target run as the source of truth.

### Resume Stage A

Resume a mid-epoch interruption:

```bash
export STAGE_A_OUT=outputs/stageA_coco_multipatch

CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/cfg_patch_stage_a.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_local.json \
  --output_dir "${STAGE_A_OUT}" \
  --resume "${STAGE_A_OUT}/checkpoint_iter.pth" \
  --num_workers 8 \
  --amp \
  --options epochs=7
```

Resume from a completed epoch checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/cfg_patch_stage_a.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_local.json \
  --output_dir "${STAGE_A_OUT}" \
  --resume "${STAGE_A_OUT}/checkpoint.pth" \
  --num_workers 8 \
  --amp \
  --options epochs=7
```

### Evaluate Stage A Patch-Only

Use the shared Stage-A-caliber validation config:

```text
config/datasets_patch_stage_a_lvis_coco2017_eval_local.json
```

Run the patch-only evaluator:

```bash
export STAGE_A_EVAL_OUT=outputs/stageA_coco_multipatch_eval_0005_0006_fast

DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u tools/eval_stagea_patch_checkpoints.py \
  --config config/cfg_patch_stage_a.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_eval_local.json \
  --ckpts \
    outputs/stageA_coco_multipatch/checkpoint0005.pth \
    outputs/stageA_coco_multipatch/checkpoint0006.pth \
  --output_dir "${STAGE_A_EVAL_OUT}" \
  --batch_size 28 \
  --num_workers 8 \
  --log_every 25 \
  --amp
```

Expected current best:

```text
checkpoint0006.pth
mean patch_ap50 = 0.596751
lvis_val patch_ap50 = 0.488652
coco_val patch_ap50 = 0.704849
```

### Evaluate Stage A Patch + Canonical Text Fusion

This evaluation is not a Stage A training loss. It tests inference-time fusion
between support-patch logits and canonical text logits on the same LVIS/COCO
validation episodes.

```bash
export STAGE_A_FUSION_OUT=outputs/stageA_coco_multipatch_patch_text_fusion_eval_0006_betas_full

DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false \
"${PY}" -u tools/eval_stagea_patch_text_fusion_probe.py \
  --config config/cfg_patch_stage_a.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_eval_local.json \
  --ckpts outputs/stageA_coco_multipatch/checkpoint0006.pth \
  --output_dir "${STAGE_A_FUSION_OUT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --log_every 25 \
  --betas 0 0.5 1.0 2.0
```

Expected current best:

```text
checkpoint0006:patch_plus_text_b1
mean patch_ap50 = 0.665160
lvis_val patch_ap50 = 0.604870
coco_val patch_ap50 = 0.725449
```

## Stage A Decoder-Unfreeze Ablation

Status: negative result, not Stage-A mainline.

This ablation starts from the historical Stage A `checkpoint0002.pth`, keeps the
phase-2/mainline recipe used for `checkpoint0003.pth` and `checkpoint0004.pth`,
but unfreezes all 6 decoder layers instead of the mainline last 3 layers. The
run also disables the epoch-4 LR drop, so both epoch 3 and epoch 4 train at
base LR.

Config and artifacts:

```text
config/ablations/cfg_stagea_phase2_decoder_all_nolrdrop.py
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_from0002_v2/
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_from0002_v2/bridge_resume_from0002_all_decoder_nolrdrop_noscaler.pth
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_from0002_v2/checkpoint0003.pth
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_from0002_v2/checkpoint0004.pth
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_eval_0003_vs_orig/summary.md
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_eval_0004_vs_orig/summary.md
```

Important settings:

```text
source checkpoint = outputs/stageA_coco_multipatch/checkpoint0002.pth
unfreeze_decoder_last_n_layers = 6
lr_drop = 100
command override = --options epochs=5 lr_drop=100
```

Because changing the number of trainable decoder layers changes optimizer
parameter groups, this run used a bridge resume checkpoint. The bridge reused
Adam state for matching parameters and initialized optimizer state for newly
trainable decoder parameters. The AMP scaler was intentionally omitted from the
bridge so `main.py` could restore optimizer and scheduler state cleanly.

Training evidence:

```text
Epoch 4 start: lr = 0.000100
Epoch 4 averaged stats: lr = 0.000100, loss = 0.7364
checkpoint0004.pth written at 2026-06-06 11:04 local time
```

Same-caliber eval versus the original Stage A `checkpoint0004.pth`:

| checkpoint | branch | mean patch_ap50 | mean box_recall@50 | mean matched_query_recall@50 | lvis_val patch_ap50 | coco_val patch_ap50 |
|---|---|---:|---:|---:|---:|---:|
| `checkpoint0003.pth` | original | 0.581870 | 0.867450 | 0.829550 | 0.468977 | 0.694764 |
| `checkpoint0003.pth` | decoder-all no-drop | 0.576159 | 0.859027 | 0.816437 | 0.461330 | 0.690988 |
| `checkpoint0004.pth` | original | 0.590184 | 0.869767 | 0.830631 | 0.478985 | 0.701384 |
| `checkpoint0004.pth` | decoder-all no-drop | 0.577569 | 0.861146 | 0.818542 | 0.462768 | 0.692369 |

Decision: do not promote decoder-all/no-drop into the Stage-A foundation. It
reduced patch AP50, box recall, and matched-query recall on both LVIS and COCO
relative to the original checkpoint at both `checkpoint0003.pth` and
`checkpoint0004.pth`.

## Deprecated Stage A v2 Loss Ablations

Status: deprecated for the Stage-A mainline. Stage-A rank loss and pos/neg
top-k CE are retained only as ablation code to reproduce negative probes. They
are not enabled by `config/cfg_patch_stage_a.py` and should not be used as the
Stage-A foundation.

Explicit ablation configs:

```text
config/cfg_patch_stage_a_v2_rank.py
config/cfg_patch_stage_a_v2_posneg_topk32_lam4.py
config/cfg_patch_stage_a_v2_rank_posneg_topk.py
```

Use a separate output directory for each ablation, and compare from the same
starting checkpoint under the same Stage-A-caliber eval. Current probes did not
beat the `checkpoint0004.pth` control, so these deprecated configs should not
be used for the Stage A foundation unless a future same-caliber probe reverses
that result.

## Stage B Local TN

Stage B starts from a Stage A checkpoint, freezes the localization-heavy path,
and trains phrase/content/TN text behavior while keeping patch matching active.

Key files:

```text
config/cfg_patch_stage_b.py
config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json
docs/stage_b_local_tn.md
outputs/stageB_local_tn_v2_no_phrase_loss/
outputs/stageB_local_tn_v3/
```

Stage B v2 mainline default training behavior:

```text
stage_b = True
patch_only = True
patch_matching = "hungarian"
patch_only_compute_text_logits = True
build_text_token_masks = True
lambda_patch = 1.0
lambda_text = 0.25
stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0
canonical_pos_weight = 0.15
only_train_keywords = ["feat_map", "class_embed"]
```

The current Stage-B mainline is v2: token/content/canonical/TN BCE only. Phrase
ranking code is retained for ablations, but it is not part of mainline Stage B.
Use `config/ablations/cfg_stageb_full.py` only for the historical rank-enabled
v3 / rank probe.

Stage-B v4 is a separate ablation that starts from the v2 epoch-3 checkpoint,
replaces v2 matched-only token BCE with GroundingDINO-like all-query sigmoid
focal text loss, and adds inference-score calibration/listwise top-10
constraints. It does not use the historical v3 phrase-rank loss.

### Reproduce Current Stage B v2

The recorded `outputs/stageB_local_tn_v2_no_phrase_loss` run was initialized
from Stage A `checkpoint0004.pth` and uses the no-ranking v2 objective.

Initial command recorded in `outputs/stageB_local_tn_v2_no_phrase_loss/info.txt`:

```bash
export STAGE_B_OUT=outputs/stageB_local_tn_v2_no_phrase_loss

CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/cfg_patch_stage_b.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --output_dir "${STAGE_B_OUT}" \
  --pretrain_model_path outputs/stageA_coco_multipatch/checkpoint0004.pth \
  --num_workers 8 \
  --amp \
  --options batch_size=22
```

Recorded latest v2 epoch checkpoint:

```text
outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth
outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint.pth
```

### Historical Stage B v3 Rank Probe

The recorded `outputs/stageB_local_tn_v3` run was initialized from Stage A
`checkpoint0004.pth`, then later resumed from `checkpoint_iter.pth`. Its saved
args have `stage_b_enable_phrase_rank=True` and `stage_b_rank_loss_coef=1.0`,
so treat it as the historical rank-enabled v3 / rank probe, not the current
Stage-B v2 mainline.

Initial command recorded in `outputs/stageB_local_tn_v3/info.txt`:

```bash
export STAGE_B_OUT=outputs/stageB_local_tn_v3

CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/ablations/cfg_stageb_full.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --output_dir "${STAGE_B_OUT}" \
  --pretrain_model_path outputs/stageA_coco_multipatch/checkpoint0004.pth \
  --num_workers 8 \
  --amp \
  --options batch_size=21
```

Recorded later resume used the same output directory and `checkpoint_iter.pth`:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/ablations/cfg_stageb_full.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --output_dir "${STAGE_B_OUT}" \
  --resume "${STAGE_B_OUT}/checkpoint_iter.pth" \
  --num_workers 8 \
  --amp \
  --options batch_size=19
```

Recorded latest v3 epoch checkpoint:

```text
outputs/stageB_local_tn_v3/checkpoint0003.pth
outputs/stageB_local_tn_v3/checkpoint.pth
```

`checkpoint_iter.pth` is newer than `checkpoint0003.pth`, but it is a mid-epoch
state. Use `checkpoint0003.pth` when the requested caliber is "latest epoch".

### Stage B v4 Score-Calibration Ablation

Stage-B v4 initializes from the recorded v2 `checkpoint0003.pth`, uses
all-query sigmoid focal text loss, and adds `loss_score_calib`:

```text
config/ablations/cfg_stageb_v4_score_calib.py
outputs/stageB_local_tn_v4_allquery_focal_score_calib_from_v2e3/
```

v4 constants were chosen from the v2 checkpoint0003 TN-val score distribution
at `beta=1`:

```text
score quantiles:
outputs/stageb_tn_val_compare_quantiles_v2/v2_ckpt0003_beta1_score_quantiles.json

tau_pos = 0.10
tau_neg = 1.40
margin = 0.30
topk_query = 10
```

The v4 loss terms are:

```text
text: original GroundingDINO-like all-query sigmoid focal over valid tokens
text weight: lambda_text = 0.05, because all-query focal has many query-token negatives
positive prompt: matched query > other top-10 queries
TN prompt: max/top-10 query score should be low
plus a light positive score floor and matched positive-vs-TN gap
```

Run v4 from v2 epoch 3:

```bash
export STAGE_B_V4_OUT=outputs/stageB_local_tn_v4_allquery_focal_score_calib_from_v2e3

CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/ablations/cfg_stageb_v4_score_calib.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --output_dir "${STAGE_B_V4_OUT}" \
  --pretrain_model_path outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth \
  --num_workers 8 \
  --amp \
  --options batch_size=19 epochs=1 lr_drop=100
```

The earlier directory `outputs/stageB_local_tn_v4_score_calib_from_v2e3/` was
an aborted pre-v4-final run with matched-only BCE and should not be used for
results.

Primary validation for v4 remains paired:

```text
RefCOCO val acc50 should not drop versus v2 checkpoint0003.
TN-val fpr@95tpr should improve versus v2 checkpoint0003.
```

### Recommended New Stage B From Current Stage A

For a new v2 mainline run using the current best Stage A foundation, initialize
from `checkpoint0006.pth` and use a fresh output directory:

```bash
export STAGE_A_CKPT=outputs/stageA_coco_multipatch/checkpoint0006.pth
export STAGE_B_OUT=outputs/stageB_local_tn_from_stageA0006

CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/cfg_patch_stage_b.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --output_dir "${STAGE_B_OUT}" \
  --pretrain_model_path "${STAGE_A_CKPT}" \
  --num_workers 8 \
  --amp \
  --options batch_size=19
```

Use this resume command for that same run:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/cfg_patch_stage_b.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --output_dir "${STAGE_B_OUT}" \
  --resume "${STAGE_B_OUT}/checkpoint_iter.pth" \
  --num_workers 8 \
  --amp \
  --options batch_size=19
```

## Stage B Same-Caliber LVIS/COCO Evaluation

For the current FT comparison, do not use RefCOCO/TN text-val. Use the same
LVIS/COCO Stage-A-caliber validation episodes:

```text
config/datasets_patch_stage_a_lvis_coco2017_eval_local.json
```

Evaluate the latest recorded Stage B v2 epoch with patch/text fusion:

```bash
export STAGE_B_EVAL_OUT=outputs/stageB_local_tn_v2_no_phrase_loss_patch_text_fusion_eval_0003_stagea_caliber

DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false \
"${PY}" -u tools/eval_stagea_patch_text_fusion_probe.py \
  --config config/cfg_patch_stage_b.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_eval_local.json \
  --ckpts outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth \
  --output_dir "${STAGE_B_EVAL_OUT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --log_every 25 \
  --betas 0 0.5 1.0 2.0
```

Expected current best:

```text
checkpoint0003:patch_plus_text_b0.5
mean patch_ap50 = 0.698039
lvis_val patch_ap50 = 0.651198
coco_val patch_ap50 = 0.744881
```

This beats the recorded GroundingDINO same-data FT `checkpoint0001.pth` under
the same LVIS/COCO Stage-A-caliber comparison:

```text
GDINO FT e1 mean patch_ap50 = 0.681998
Stage B v2 e3 beta=0.5 delta = +0.016041
```

## GroundingDINO Same-Data FT Baseline

The baseline run converts the Stage A LVIS/COCO patch-episode train split into
ODVG records, then trains ordinary GroundingDINO from OGC weights.

Run helper:

```bash
CUDA_VISIBLE_DEVICES=0 \
STAGEA_DATASETS="${GDINO_ROOT}/config/datasets_patch_stage_a_lvis_coco2017_local.json" \
STAGE_A_LOG="${GDINO_ROOT}/outputs/stageA_coco_multipatch/log.txt" \
PRETRAIN_MODEL_PATH="${GDINO_ROOT}/weights/groundingdino_swint_ogc.pth" \
OUTPUT_DIR="${GDINO_ROOT}/outputs/ogc_original_finetune_stage_a" \
tools/run_ogc_original_finetune_stage_a.sh
```

Evaluate with canonical text prompts on the shared LVIS/COCO episodes:

```bash
DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u tools/eval_text_stagea_caliber_checkpoints.py \
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

Recorded best:

```text
checkpoint0001.pth
mean patch_ap50 = 0.681998
lvis_val patch_ap50 = 0.640117
coco_val patch_ap50 = 0.723879
```

## Pure GroundingDINO Stage-B Data Ablations

These ablations continue the GroundingDINO same-data FT checkpoint with the
Stage-B data recipe, but keep the model path pure GroundingDINO. They must not
use `config/cfg_patch_stage_b.py`, `patch_episode` datasets, support patches,
patch losses, Stage-B criterion, or phrase-rank loss.

Key files:

```text
tools/build_stageb_gdino_finetune_ablation.py
config/ablations/cfg_stageb_from_gdino_ft_with_tn.py
config/ablations/cfg_stageb_from_gdino_ft_no_tn.py
config/ablations/datasets_gdino_ft_stageb_with_tn_local.json
config/ablations/datasets_gdino_ft_stageb_no_tn_local.json
data/ablations/gdino_ft_stage_b/
```

Pure-structure boundary:

```text
patch_only = False
stage_b = False
enable_patch_branch = False
batch_size = 19
```

`enable_patch_branch = False` prevents this fork's optional patch encoder,
query-to-patch projection, and patch logit scale from being instantiated at all.
When loading a checkpoint that contains historical patch-extension keys, those
keys are expected `unexpected_keys` and are ignored. The ordinary GroundingDINO
SetCriterion remains the only training loss.

Build the pure-GDINO Stage-B ODVG/VG datasets:

```bash
DATA_ROOT="${DATA_ROOT}" "${PY}" tools/build_stageb_gdino_finetune_ablation.py \
  --stageb_datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --stagea_odvg_datasets data/ablations/ogc_original_finetune_stage_a/stagea_odvg_datasets.json \
  --out_dir data/ablations/gdino_ft_stage_b \
  --dataset_config_dir config/ablations
```

Generated data semantics:

```text
LVIS train:      ODVG detection, mix_weight = 2
COCO train:      ODVG detection, mix_weight = 2
RefCOCO+ train:  VG positive phrase + box, mix_weight = 2
RefCOCOg train:  VG positive phrase + box, mix_weight = 2
TN train:        VG negative phrase + zero regions, mix_weight = 1
```

The with-TN dataset includes the TN row above. The no-TN dataset removes only
that entry, keeping the other four entries and ordinary GroundingDINO training
config identical.

Run the with-TN ablation from the same-data FT best checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/ablations/cfg_stageb_from_gdino_ft_with_tn.py \
  --datasets config/ablations/datasets_gdino_ft_stageb_with_tn_local.json \
  --output_dir outputs/gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch \
  --pretrain_model_path outputs/ogc_original_finetune_stage_a/checkpoint0001.pth \
  --num_workers 8 \
  --amp \
  --options batch_size=19
```

Run the no-TN companion:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/ablations/cfg_stageb_from_gdino_ft_no_tn.py \
  --datasets config/ablations/datasets_gdino_ft_stageb_no_tn_local.json \
  --output_dir outputs/gdino_ft_stageb_from_gdino_ft_e1_no_tn_bs19_nopatchbranch \
  --pretrain_model_path outputs/ogc_original_finetune_stage_a/checkpoint0001.pth \
  --num_workers 8 \
  --amp \
  --options batch_size=19
```

## Result Files To Check

Stage A patch-only:

```text
outputs/stageA_coco_multipatch_eval_0005_0006_fast/summary.json
outputs/stageA_coco_multipatch_eval_0005_0006_fast/summary.md
```

Stage A patch/text fusion:

```text
outputs/stageA_coco_multipatch_patch_text_fusion_eval_0006_betas_full/summary.json
outputs/stageA_coco_multipatch_patch_text_fusion_eval_0006_betas_full/summary.md
```

Stage A decoder-all/no-drop ablation:

```text
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_eval_0003_vs_orig/summary.json
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_eval_0003_vs_orig/summary.md
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_eval_0004_vs_orig/summary.json
outputs/stageA_coco_multipatch_decoder_all_nolrdrop_eval_0004_vs_orig/summary.md
```

Stage B same-caliber patch/text fusion:

```text
outputs/stageB_local_tn_v2_no_phrase_loss_patch_text_fusion_eval_0003_stagea_caliber/summary.json
outputs/stageB_local_tn_v2_no_phrase_loss_patch_text_fusion_eval_0003_stagea_caliber/summary.md
```

GroundingDINO same-data FT:

```text
outputs/ogc_original_finetune_stage_a_eval_stagea_caliber_e1_e2/summary.json
outputs/ogc_original_finetune_stage_a_eval_stagea_caliber_e1_e2/summary.md
```

The detailed metric interpretation and Stage-A v2 loss-ablation decision are
recorded in `docs/stage_a_caliber_eval.md`.
