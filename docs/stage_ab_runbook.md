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
TN prompt: global top-10 over all query/slot scores should be low
plus a light positive score floor and matched positive-vs-TN gap
```

The TN score calibration uses the same global flattening as inference
(`slot_logits.reshape(B, -1)`), not only the matched/source slot. The matched TN
score is still logged for diagnosis.

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

### Stage B v5 alltn00625 Candidate

The current best TN/FPR branch is v5 `alltn00625`, initialized from the recorded
v2 epoch-3 checkpoint. Unlike v2, this probe unfreezes decoder/box heads and
enables bbox losses while keeping the patch branch, backbone/input projection,
and BERT frozen:

```text
config/ablations/cfg_stageb_v5_tnneg10_lse_top10_alltn00625_from_w0125.py
outputs/stageB_v5_tnneg10_lse_top10_alltn00625_from_v2e3_bs4_long/
```

Important config deltas from v2:

```text
stage_b_text_loss_type = "allquery_focal_tn_matched_bce"
only_train_keywords = ["feat_map", "class_embed", "bbox_embed", "transformer.decoder.layers"]
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
lambda_tn_neg = 10.0
lambda_tn_content = 0.0
lambda_tn_canonical = 0.0
stage_b_score_calib_neg_agg = "logsumexp"
stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_neg_weight = 0.125
stage_b_score_calib_gap_weight = 0.125
stage_b_score_calib_all_tn_neg_weight = 0.0625
stage_b_score_calib_detach_patch = False
```

Run from v2 checkpoint0003:

```bash
export STAGE_B_V5_OUT=outputs/stageB_v5_tnneg10_lse_top10_alltn00625_from_v2e3_bs4_long

DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u main.py \
  --config_file config/ablations/cfg_stageb_v5_tnneg10_lse_top10_alltn00625_from_w0125.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --pretrain_model_path outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth \
  --output_dir "${STAGE_B_V5_OUT}" \
  --num_workers 8 \
  --amp \
  --iter_checkpoint_interval 5000 \
  --options batch_size=4
```

The selected checkpoint is the explicit 20k backup:

```text
outputs/stageB_v5_tnneg10_lse_top10_alltn00625_from_v2e3_bs4_long/checkpoint_iter0020000.pth
```

Use the explicit backup for comparisons; `checkpoint_iter.pth` may be a later
signal-save checkpoint and is not the same comparison point.

TN-val evaluation:

```bash
DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u tools/eval_stageb_tn_val.py \
  --config config/ablations/cfg_stageb_v5_tnneg10_lse_top10_alltn00625_from_w0125.py \
  --ckpts outputs/stageB_v5_tnneg10_lse_top10_alltn00625_from_v2e3_bs4_long/checkpoint_iter0020000.pth \
  --output_dir outputs/stageb_tn_val_v5_tnneg10_lse_top10_alltn00625_long20k \
  --data_root "${DATA_ROOT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --betas 0 0.5 1 2 \
  --splits refcocop_val refcocog_val \
  --log_every 50
```

RefCOCO/RefCOCO+ val evaluation:

```bash
DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u tools/eval_refcoco_stageb.py \
  --config config/ablations/cfg_stageb_v5_tnneg10_lse_top10_alltn00625_from_w0125.py \
  --ckpts outputs/stageB_v5_tnneg10_lse_top10_alltn00625_from_v2e3_bs4_long/checkpoint_iter0020000.pth \
  --output_dir outputs/refcoco_stageb_v5_tnneg10_lse_top10_alltn00625_long20k_ref_refp \
  --data_root "${DATA_ROOT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --betas 0 0.5 1 2 \
  --splits refcoco_val refcocop_val \
  --log_every 50
```

Recorded result:

| checkpoint | beta | TN FPR@95 | TN pair win | RefCOCO val acc50 | RefCOCO+ val acc50 |
|---|---:|---:|---:|---:|---:|
| v2 `checkpoint0003.pth` | 1.0 | 0.780817 | 0.744607 | 0.533229 | 0.564045 |
| alltn00625 long20k | 0.5 | 0.752119 | 0.789869 | 0.578826 | 0.595464 |

Decision: for TN rejection plus local RefCOCO val, `alltn00625` long20k is the
current strongest candidate. Its gain should be described as a combined v5
optimization-scope plus inference-score calibration result, not as a pure
same-freeze v2 score-head change.

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
| v5.2 calibrated allTN | v5.2 | use the calibrated allTN tail threshold/weight `tau=0.5605,w=0.36` and current TN token weights `10/1/1` |

### Stage B v5.1 RefCOCO Patch-Positive CE Probe

v5.1 keeps the v5 alltn00625 text/TN/score-calibration recipe and changes only
the patch CE labels for RefCOCO-family phrase rows:

```text
config/ablations/cfg_stageb_v5_1_refcoco_patchpos_from_v5_alltn00625.py
patch_ce_positive_only_for_datasets = ("refcoco", "refcocoplus", "refcocog", "refexp")
```

Reason: RefCOCO-style phrase rows only label the referred phrase object. Dense
patch CE negatives on those rows can create false negatives for unannotated
same-class/person objects. v5.1 computes patch CE from matched positive cells
only on RefCOCO-family rows; LVIS/COCO patch CE remains the v5 dense objective.

Fair 1k probe from the shared v2 Stage-B checkpoint used by the alltn00625 1k
baseline:

```bash
export STAGE_B_V51_OUT=outputs/stageB_v5_1_refcoco_patchpos_from_v2e3_probe1k

DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u main.py \
  --config_file config/ablations/cfg_stageb_v5_1_refcoco_patchpos_from_v5_alltn00625.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --pretrain_model_path outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth \
  --output_dir "${STAGE_B_V51_OUT}" \
  --num_workers 8 \
  --amp \
  --max_train_iters 1000 \
  --iter_checkpoint_interval 500 \
  --options batch_size=4
```

Check the train log for `patch_ce_positive_only_batch_frac > 0` on mixed
RefCOCO batches and nonzero `patch_ce_neg_count` on LVIS/COCO batches.

### Stage B v5.2 RefCOCO Patch-Positive CE + Aux Probe

v5.2 is v5.1 plus decoder auxiliary losses:

```text
config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625.py
aux_loss = True
use_checkpoint = False
```

The patch-only forward path emits intermediate decoder patch scores, and
`StageBCriterion` applies aux `loss_patch_ce`, `loss_text`, `loss_bbox`, and
`loss_giou` for intermediate layers. Score calibration and phrase-rank losses
remain final-layer only.

Run the fair 1k probe from the same v2 checkpoint:

```bash
export STAGE_B_V52_OUT=outputs/stageB_v5_2_refcoco_patchpos_aux_from_v2e3_probe1k

DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u main.py \
  --config_file config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --pretrain_model_path outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth \
  --output_dir "${STAGE_B_V52_OUT}" \
  --num_workers 8 \
  --amp \
  --max_train_iters 1000 \
  --iter_checkpoint_interval 500 \
  --options batch_size=4
```

Text-weight sweep probes on top of v5.2 raise only `lambda_text` while keeping
the RefCOCO patch-positive CE, aux losses, TN calibration, and freeze/unfreeze
settings unchanged:

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

Run each point from the same v2 checkpoint and the same 1k budget. Example for
`lambda_text=1.0`:

```bash
export STAGE_B_V53_OUT=outputs/stageB_v5_3_refcoco_patchpos_aux_text1_from_v2e3_probe1k

DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u main.py \
  --config_file config/ablations/cfg_stageb_v5_3_refcoco_patchpos_aux_text1_from_v5_alltn00625.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --pretrain_model_path outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth \
  --output_dir "${STAGE_B_V53_OUT}" \
  --num_workers 8 \
  --amp \
  --max_train_iters 1000 \
  --iter_checkpoint_interval 500 \
  --options batch_size=4
```

Evaluate v5.1/v5.2/v5.3 and the text-weight sweep with the same TN and RefCOCO
val caliber. Replace the config, checkpoint, and output directory for one-off
probes. For the `0.5/0.75` sweep, both checkpoints can share the `0.5` config
at eval time because `lambda_text` affects only training loss, not model
structure or inference:

```bash
DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u tools/eval_stageb_tn_val.py \
  --config config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625.py \
  --ckpts outputs/stageB_v5_2_refcoco_patchpos_aux_from_v2e3_probe1k/checkpoint_iter.pth \
  --output_dir outputs/stageb_tn_val_v5_2_refcoco_patchpos_aux_from_v2e3_probe1k \
  --data_root "${DATA_ROOT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --betas 0 0.5 1 2 \
  --splits refcocop_val refcocog_val \
  --log_every 50
```

```bash
DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u tools/eval_refcoco_stageb.py \
  --config config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625.py \
  --ckpts outputs/stageB_v5_2_refcoco_patchpos_aux_from_v2e3_probe1k/checkpoint_iter.pth \
  --output_dir outputs/refcoco_stageb_v5_2_refcoco_patchpos_aux_from_v2e3_probe1k_ref_refp \
  --data_root "${DATA_ROOT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --betas 0 0.5 1 2 \
  --splits refcoco_val refcocop_val \
  --log_every 50
```

Additional fine-sweep evidence:

```text
outputs/stageb_tn_val_v5_2_lambda_text_sweep_030_045_probe1k/summary.md
outputs/refcoco_stageb_v5_2_lambda_text_sweep_030_045_probe1k_ref_refp/summary.md
```

Recorded fair 1k comparison:

| checkpoint | TN-best beta | TN FPR@95 | TN pair win | pos top1 IoU50 | Ref-best beta | mean Ref acc50 | RefCOCO val acc50 | RefCOCO+ val acc50 |
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

Decision: v5.2 is the best 1k direction. v5.1 improves TN rejection but loses
RefCOCO val accuracy; aux losses recover the positive grounding side while
preserving the lower FPR. The fine `0.30/0.35/0.40/0.45` sweep shows the local
tradeoff is not monotonic. `lambda_text=0.40` is the best TN/FPR point
(`0.747304` FPR@95 at beta `1.0`) but drops mean Ref acc50 to `0.581381`;
`lambda_text=0.45` gives the best fine-sweep Ref mean (`0.587875`) but regresses
FPR to `0.762519`; `lambda_text=0.30` is the balanced candidate, keeping Ref
mean near the default (`0.586394` vs `0.586751`) while improving TN-best FPR
(`0.754237` vs `0.757126`). Keep v5.2 `lambda_text=0.25` as the established
default unless the next longer run explicitly targets the `0.30` balanced point
or the `0.40` FPR-focused point.

### Stage B v5.2 Calibrated allTN

This is the v5.x counterpart of the calibrated pure-GDINO allTN run. It keeps
the established v5.2 wrapper recipe and changes only the allTN tail-suppression
setting plus the current TN-token contract:

```text
config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_alltn_tau05605_w036.py
```

Compared with historical v5 `alltn00625`, this file updates:

```text
stage_b_score_calib_tau_neg = 0.5605
stage_b_score_calib_all_tn_neg_weight = 0.36
lambda_tn_neg = 10.0
lambda_tn_content = 1.0
lambda_tn_canonical = 1.0
tn_content_target = 0.0
tn_canonical_target = 0.0
```

The parameter names differ from the pure-GDINO config because v5.x implements
allTN inside `loss_score_calib`:

```text
pure GDINO: loss_tn_alltn, gdino_tn_alltn_tau_neg, gdino_tn_alltn_weight
Stage-B v5.x: loss_score_calib, stage_b_score_calib_tau_neg,
              stage_b_score_calib_all_tn_neg_weight
```

The score space is also not identical: pure GDINO suppresses a text-only
sigmoid-mean query score, while Stage-B v5.x suppresses the final Stage-B slot
score that combines patch and text. The shared `topk=10`, `lse_tau=0.2`,
`tau=0.5605`, and `w=0.36` are therefore a calibrated run setting, not a claim
that the two losses are numerically interchangeable.

Run from the selected v5.2 lineage checkpoint:

```bash
export STAGE_B_V52_CAL_OUT=outputs/stageB_v5_2_refcoco_patchpos_aux_alltn_tau05605_w036

DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u main.py \
  --config_file config/ablations/cfg_stageb_v5_2_refcoco_patchpos_aux_alltn_tau05605_w036.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --pretrain_model_path outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth \
  --output_dir "${STAGE_B_V52_CAL_OUT}" \
  --num_workers 8 \
  --amp
```

### Stage B v6 GDINO-like Text + v5.2 Patch CE Probe

v6 is a probe, not the selected Stage-B branch. It keeps TN examples positive
for the patch branch, but makes the text/detection view match GDINO Stage-B
data-FT as closely as this Stage-B wrapper allows: all-query token sigmoid
focal, uniform token weights, GDINO `cls_loss_coef=2.0` mirrored by
`lambda_text=2.0`, TN prompts as no-positive/all-negative text rows, no
extra-IoU positive query expansion, and no TN-specific BCE/calibration. The
only intended addition over the GDINO-like text/detection branch is v5.2's
patch CE path, including RefCOCO-family positive-only patch CE and aux losses.

```text
config/ablations/cfg_stageb_v6_gdino_like_tn_empty_det_patchpos_aux.py
outputs/stageB_v6_gdino_like_text_v52patch_from_v2e3_probe1k/checkpoint_iter.pth
```

Main deltas:

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
stage_b_score_calib_loss_coef = 0.0
```

Run the 1k probe from the same Stage-B v2 checkpoint used by the v5.x fair
probes:

```bash
DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u main.py \
  -c config/ablations/cfg_stageb_v6_gdino_like_tn_empty_det_patchpos_aux.py \
  --datasets config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json \
  --pretrain_model_path outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth \
  --output_dir outputs/stageB_v6_gdino_like_text_v52patch_from_v2e3_probe1k \
  --num_workers 4 \
  --amp \
  --max_train_iters 1000 \
  --iter_checkpoint_interval 500 \
  --options batch_size=4
```

Evaluation uses the same TN-val and RefCOCO scripts as v5:

```bash
DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u tools/eval_stageb_tn_val.py \
  --config config/ablations/cfg_stageb_v6_gdino_like_tn_empty_det_patchpos_aux.py \
  --ckpts outputs/stageB_v6_gdino_like_text_v52patch_from_v2e3_probe1k/checkpoint_iter.pth \
  --output_dir outputs/stageb_tn_val_v6_gdino_like_text_v52patch_probe1k \
  --data_root "${DATA_ROOT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --betas 0 0.5 1 2 \
  --splits refcocop_val refcocog_val \
  --log_every 50
```

```bash
DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u tools/eval_refcoco_stageb.py \
  --config config/ablations/cfg_stageb_v6_gdino_like_tn_empty_det_patchpos_aux.py \
  --ckpts outputs/stageB_v6_gdino_like_text_v52patch_from_v2e3_probe1k/checkpoint_iter.pth \
  --output_dir outputs/refcoco_stageb_v6_gdino_like_text_v52patch_probe1k_ref_refp_refg \
  --data_root "${DATA_ROOT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --betas 0 0.5 1 2 \
  --splits refcoco_val refcocop_val refcocog_val \
  --log_every 50
```

Separate Stage-A-initialized probe: the table below is from a v6-style run
initialized directly from Stage A `checkpoint0006.pth`
(`stageA0006_probe1k`). It is not the current Stage-B V6 lineage, which
continues from Stage-B v2 `checkpoint0003.pth`; do not compare it as an older
checkpoint of the same run.

| checkpoint | beta | TN FPR@95 | TN pair win | RefCOCO val acc50 | RefCOCO+ val acc50 | RefCOCOg val acc50 |
|---|---:|---:|---:|---:|---:|---:|
| Stage-A-init v6-style 1k TN-best | 2.0 | 0.890216 | 0.634630 | 0.572734 | 0.606061 | 0.711601 |
| Stage-A-init v6-style 1k Ref-best | 1.0 | 0.896572 | 0.628082 | 0.572365 | 0.607269 | 0.712418 |

Wrapper-vs-original criterion audit:

| item | pure GroundingDINO Stage-B data FT | Stage-B v6 wrapper |
|---|---|---|
| text loss formula | all-query sigmoid focal | all-query sigmoid focal |
| text loss weight | `cls_loss_coef=2.0` | `lambda_text=2.0`, `cls_loss_coef=2.0` for provenance |
| TN rows | no positive boxes/text tokens | no positive text tokens; patch branch still has TN support positives |
| Hungarian matching | text-logit class cost plus box costs | patch-logit matching, box losses filtered for TN |
| aux text supervision | original aux matching path | aux patch matching path |
| inference score | text phrase score | `patch_score + beta * text_score` slot fusion |

So v6 aligns the text-loss formula and weighting with original GDINO, but it is
not criterion-identical. The main remaining mismatch is Hungarian assignment:
Stage-B still matches through `PatchHungarianCriterion.compute_matching` on
`pred_logits_patch`, while original GDINO uses text logits and `positive_map`
for class cost. The content/canonical target-token map should be equivalent
when built from the same phrase spans, but the matcher and inference scorer are
still different.

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

This comparison is not a clean "patch loss only" ablation because the Stage-B
wrapper, patch-based matching, and beta-fused scorer remain active. Current
evidence says v6 does not yet close the gap to pure GDINO Stage-B data FT on
either TN rejection or RefCOCO localization.

Decision status for the v2-initialized fair v6 probe remains pending; keep that
separate from the completed Stage-A0006 epoch-1 run above.

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

### Calibrated Pure-GDINO allTN Run

The selected pure-GDINO allTN variant keeps the same no-patch GroundingDINO
Stage-B data-FT path as `cfg_stageb_from_gdino_ft_with_tn.py`. It is not a
Stage-B wrapper/v5/v6 run. The only intended deltas over the with-TN pure-GDINO
baseline are TN token BCE plus top-k allTN suppression:

```text
config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py
patch_only = False
stage_b = False
enable_patch_branch = False
gdino_tn_loss_type = "alltn00625"
gdino_tn_alltn_topk = 10
gdino_tn_alltn_lse_tau = 0.2
gdino_tn_alltn_tau_neg = 0.5605
gdino_tn_alltn_weight = 0.36
lambda_tn_neg = 10.0
lambda_tn_content = 1.0
lambda_tn_canonical = 1.0
```

`tau_neg=0.5605` is in the allTN aggregate space, not the single-query score
space. With `topk=10` and `lse_tau=0.2`, ten queries at sigmoid-mean score
`0.1` map to:

```text
0.1 + 0.2 * log(10) ~= 0.5605
```

The calibrated weight comes from the actual Stage-B train mix:

```text
calibration:
outputs/gdino_alltn_calibration_stagea0001_tau05605_mix120/calibration.json

base weighted loss = 15.2554
effective raw allTN at tau=0.5605 = 4.2167
target allTN/base ratio 10% -> weight = 0.3618
```

303-iter capped-probe evidence, all initialized from
`outputs/ogc_original_finetune_stage_a/checkpoint0001.pth` and evaluated on
100 batches per RefCOCO split plus 100 TN batches:

```text
outputs/text_gdino_alltn_tau05605_weight_probe303_eval/summary.md
```

| config | mean RefCOCO acc50 | TN FPR@95 | TN FPR@90 | TN pair win |
|---|---:|---:|---:|---:|
| baseline `tau=0.0625,w=0.0625` | 0.622222 | 0.888378 | 0.810619 | 0.636706 |
| `tau=0.5605,w=0.1809` | 0.624306 | 0.913880 | 0.836120 | 0.628763 |
| selected `tau=0.5605,w=0.36` | 0.625278 | 0.883779 | 0.806856 | 0.631689 |

Decision: `w=0.1809` is too weak for TN rejection despite better RefCOCO.
Use `tau=0.5605,w=0.36` for the full pure-GDINO Stage-B data-FT allTN run.

Run one full Stage-B data epoch:

```bash
export GDINO_ALLTN_OUT=outputs/gdino_ft_stageb_from_stagea0001_with_tn_alltn_tau05605_w036

CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py \
  --datasets config/ablations/datasets_gdino_ft_stageb_with_tn_local.json \
  --output_dir "${GDINO_ALLTN_OUT}" \
  --pretrain_model_path outputs/ogc_original_finetune_stage_a/checkpoint0001.pth \
  --num_workers 8 \
  --prefetch_factor 1 \
  --amp
```

Resume the same run from an interrupted iteration checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT="${DATA_ROOT}" TOKENIZERS_PARALLELISM=false \
"${PY}" -u main.py \
  -c config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py \
  --datasets config/ablations/datasets_gdino_ft_stageb_with_tn_local.json \
  --output_dir "${GDINO_ALLTN_OUT}" \
  --resume "${GDINO_ALLTN_OUT}/checkpoint_iter.pth" \
  --num_workers 8 \
  --prefetch_factor 1 \
  --amp
```

Evaluate RefCOCO/TN on the pure text GroundingDINO scorer:

```bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
"${PY}" -u tools/eval_text_groundingdino_refcoco_tn.py \
  --config config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py \
  --ckpts "${GDINO_ALLTN_OUT}/checkpoint0000.pth" \
  --output_dir outputs/text_gdino_alltn_tau05605_w036_epoch0_eval \
  --data_root "${DATA_ROOT}" \
  --batch_size 24 \
  --num_workers 8 \
  --amp \
  --ref_splits refcoco_val refcocop_val refcocog_val \
  --tn_splits refcocop_val refcocog_val \
  --score_thresholds 0.5 \
  --log_every 25
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
