#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/tools/stageb_python.sh"
PYTHON_BIN="$(stageb_resolve_python "${PYTHON_BIN:-}")"

CONFIG="${REPO_ROOT}/config/ablations/cfg_stagea_b58_trunk_patch0006_realign_20260814.py"
DATASETS="${REPO_ROOT}/config/datasets_patch_stage_a_lvis_coco2017_local.json"
INITIALIZER="/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_20260814_initializer.pth"
OUTPUT_DIR="${OUTPUT_DIR:-/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_bs40_formal_20260814}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" tools/build_stagea_b58_patch0006_initializer.py verify \
    --checkpoint "${INITIALIZER}" >/dev/null

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" main.py \
    -c "${CONFIG}" \
    --datasets "${DATASETS}" \
    --output_dir "${OUTPUT_DIR}" \
    --pretrain_model_path "${INITIALIZER}" \
    --num_workers 8 \
    --amp \
    "$@"
