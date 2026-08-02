#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${STAGE_A_CKPT:?Set STAGE_A_CKPT to the rebuilt Stage-A checkpoint.}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/gdino_stageb_three_ref_probe303_20260711}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRINT_ONLY="${PRINT_ONLY:-1}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/main.py"
  -c "${REPO_ROOT}/config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036_three_ref.py"
  --datasets "${REPO_ROOT}/config/ablations/gdino_ft_stage_b_rebuild_20260711/datasets_gdino_ft_stageb_three_ref_with_tn_local.json"
  --output_dir "${OUTPUT_DIR}"
  --pretrain_model_path "${STAGE_A_CKPT}"
  --max_train_iters 303
  --iter_checkpoint_interval 303
  --num_workers "${NUM_WORKERS}"
  --amp
  --save_log
)

if [[ "${PRINT_ONLY}" == "1" ]]; then
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' \
    "${REPO_ROOT}" "${CUDA_VISIBLE_DEVICES}" "${REPO_ROOT}:${PYTHONPATH:-}"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${CMD[@]}"
