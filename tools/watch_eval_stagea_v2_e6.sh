#!/usr/bin/env bash
set -euo pipefail

cd /media/haoyi/T9/gdino

OUT="outputs/stageA_coco_multipatch_v2_rank_posneg_from0004"
EVAL_OUT="outputs/stageA_coco_multipatch_v2_rank_posneg_eval_0006_fast"
CKPT="$OUT/checkpoint0006.pth"
TRAIN_PID="$(cat "$OUT/train.pid")"
PY="/home/haoyi/miniconda/envs/cvpr/bin/python"

export CUDA_VISIBLE_DEVICES=0
export DATA_ROOT=/media/haoyi/T9/data
export PYTHONPATH=/media/haoyi/T9/gdino:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$EVAL_OUT"
echo "$(date --iso-8601=seconds) watcher_start train_pid=$TRAIN_PID ckpt=$CKPT eval_out=$EVAL_OUT"

while true; do
  if [ -f "$CKPT" ] && ! ps -p "$TRAIN_PID" >/dev/null 2>&1; then
    size1="$(stat -c %s "$CKPT")"
    sleep 20
    size2="$(stat -c %s "$CKPT")"
    if [ "$size1" = "$size2" ]; then
      echo "$(date --iso-8601=seconds) checkpoint_ready size=$size2"
      break
    fi
  fi

  if ps -p "$TRAIN_PID" >/dev/null 2>&1; then
    train_log="$(cat "$OUT/train.logpath")"
    line="$(grep -E "Epoch: \[[56]\]" "$train_log" | tail -n 1 || true)"
    echo "$(date --iso-8601=seconds) waiting_train $line"
  else
    if [ -f "$CKPT" ]; then
      exists=yes
    else
      exists=no
    fi
    echo "$(date --iso-8601=seconds) train_exited_waiting_ckpt ckpt_exists=$exists"
  fi
  sleep 300
done

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
"$PY" -u tools/eval_stagea_patch_checkpoints.py \
  --config config/cfg_patch_stage_a_v2_rank.py \
  --datasets config/datasets_patch_stage_a_lvis_coco2017_eval_local.json \
  --ckpts "$CKPT" \
  --output_dir "$EVAL_OUT" \
  --batch_size 28 \
  --num_workers 8 \
  --amp \
  --log_every 25

"$PY" -u tools/compare_stagea_v2_e6.py
echo "$(date --iso-8601=seconds) watcher_done"
