#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/tools/stageb_python.sh"

PYTHON_BIN="$(stageb_resolve_python "${PYTHON_BIN:-}")"
DATA_ROOT="${DATA_ROOT:-/home/user/datasets/pivot_data}"
STAGEA_CKPT="${STAGEA_CKPT:-outputs/ogc_original_finetune_stage_a_rebuild_20260711/checkpoint0001.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gdino_ft_stage_b_fixed_baseline_20260711}"
MASTER_PORT="${MASTER_PORT:-29519}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
DRY_RUN="${DRY_RUN:-0}"

CONFIG="config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py"
DATASETS="config/ablations/gdino_ft_stage_b_rebuild_20260711/datasets_gdino_ft_stageb_with_tn_local.json"
WORLD_SIZE=2
PER_GPU_BATCH=9

if [[ -e "${OUTPUT_DIR}/checkpoint.pth" || -e "${OUTPUT_DIR}/checkpoint0000.pth" || -e "${OUTPUT_DIR}/checkpoint_iter.pth" ]]; then
    echo "[FAIL] output directory already contains a checkpoint: ${OUTPUT_DIR}" >&2
    echo "Use a fresh OUTPUT_DIR; this script never guesses resume semantics." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py train-preflight \
    --stagea_checkpoint "${STAGEA_CKPT}" \
    --data_root "${DATA_ROOT}" \
    --world_size "${WORLD_SIZE}" \
    --per_gpu_batch "${PER_GPU_BATCH}" \
    --output "${OUTPUT_DIR}/protocol_train_preflight.json"

command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node="${WORLD_SIZE}"
    --master_port="${MASTER_PORT}"
    main.py
    -c "${CONFIG}"
    --datasets "${DATASETS}"
    --output_dir "${OUTPUT_DIR}"
    --pretrain_model_path "${STAGEA_CKPT}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --amp
    --save_log
    --iter_checkpoint_interval 1000
    --options batch_size="${PER_GPU_BATCH}"
)

printf '%q ' "${command[@]}" > "${OUTPUT_DIR}/launch_command.txt"
printf '\n' >> "${OUTPUT_DIR}/launch_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY RUN] fixed baseline command:"
    cat "${OUTPUT_DIR}/launch_command.txt"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DATA_ROOT
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_console.log"

"${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py train-postflight \
    --output_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/protocol_train_complete.json"

echo "[OK] authoritative fixed baseline checkpoint: ${OUTPUT_DIR}/checkpoint0000.pth"
