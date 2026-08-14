#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/tools/stageb_python.sh"

PYTHON_BIN="$(stageb_resolve_python "${PYTHON_BIN:-}")"
STAGEA_CHECKPOINT="${STAGEA_CHECKPOINT:-/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_bs38_formal_20260814/checkpoint0007.pth}"
INITIALIZER="${INITIALIZER:-/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_20260814_initializer.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stagea_b58_patch0006_realign_r100_c100_20260815}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
RANK_DIR="${OUTPUT_ROOT}/rank"
RANK_CHECKPOINT="${RANK_DIR}/milestones/checkpoint_iter_000100.pth"
RANK_RECEIPT="${OUTPUT_ROOT}/stagea_r100_receipt.json"

if [[ ! -f "${STAGEA_CHECKPOINT}" ]]; then
    echo "[FAIL] completed Stage A checkpoint is missing: ${STAGEA_CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${INITIALIZER}" ]]; then
    echo "[FAIL] Stage A initializer is missing: ${INITIALIZER}" >&2
    exit 2
fi
if [[ -e "${RANK_DIR}/checkpoint_iter.pth" || -e "${RANK_CHECKPOINT}" || -e "${RANK_RECEIPT}" ]]; then
    echo "[FAIL] R100 output already exists; inspect it instead of overwriting: ${OUTPUT_ROOT}" >&2
    exit 2
fi

mkdir -p "${RANK_DIR}/milestones"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"${PYTHON_BIN}" main.py \
    -c config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py \
    --datasets config/datasets_stageb_gdino_adapter_rank_three_ref.json \
    --output_dir "${RANK_DIR}" \
    --pretrain_model_path "${STAGEA_CHECKPOINT}" \
    --max_train_iters 100 \
    --iter_checkpoint_interval 50 \
    --num_workers 8 \
    --prefetch_factor 1 \
    --amp \
    --save_log \
    --options batch_size=32 \
    2>&1 | tee "${RANK_DIR}/train_console.log"

if [[ ! -f "${RANK_DIR}/checkpoint_iter.pth" ]]; then
    echo "[FAIL] R100 produced no checkpoint_iter.pth" >&2
    exit 2
fi
cp --reflink=auto --preserve=timestamps \
    "${RANK_DIR}/checkpoint_iter.pth" "${RANK_CHECKPOINT}"

"${PYTHON_BIN}" tools/build_stagea_b58_r100_receipt.py build \
    --stagea-checkpoint "${STAGEA_CHECKPOINT}" \
    --initializer "${INITIALIZER}" \
    --rank-checkpoint "${RANK_CHECKPOINT}" \
    --output "${RANK_RECEIPT}"

tools/run_stageb_gdino_adapter_total_trust_probe.sh \
    --rank-checkpoint "${RANK_CHECKPOINT}" \
    --rank-audit "${RANK_RECEIPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --confidence-max-target 100

echo "[OK] completed sealed R100/C100: ${OUTPUT_ROOT}"
