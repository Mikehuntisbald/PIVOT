#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/media/${MEDIA_USER:-haoyi}/T9/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/stageB_ablations}"
: "${STAGE_A_CKPT:?Please set STAGE_A_CKPT=/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch/checkpoint0004.pth or another intended Stage A foundation checkpoint.}"

FULL_DATASET="${REPO_ROOT}/config/datasets_patch_stage_b_lvis_coco_refexp_tn_local.json"
NO_TN_DATASET="${REPO_ROOT}/config/ablations/datasets_patch_stage_b_lvis_coco_refexp_no_tn_local.json"

print_cmd() {
  local name="$1"
  local cfg="$2"
  local dataset="$3"
  cat <<EOF
# ${name}
cd "${REPO_ROOT}" && DATA_ROOT="${DATA_ROOT}" python main.py \\
  -c "${cfg}" \\
  --datasets "${dataset}" \\
  --pretrain_model_path "${STAGE_A_CKPT}" \\
  --output_dir "${OUTPUT_ROOT}/${name}" \\
  --amp \\
  --num_workers 8

EOF
}

print_cmd "full" "${REPO_ROOT}/config/ablations/cfg_stageb_full.py" "${FULL_DATASET}"
print_cmd "no_tn" "${REPO_ROOT}/config/ablations/cfg_stageb_no_tn.py" "${NO_TN_DATASET}"
print_cmd "no_canonical" "${REPO_ROOT}/config/ablations/cfg_stageb_no_canonical.py" "${FULL_DATASET}"
print_cmd "full_canonical" "${REPO_ROOT}/config/ablations/cfg_stageb_full_canonical.py" "${FULL_DATASET}"
# print_cmd "no_text" "${REPO_ROOT}/config/ablations/cfg_stageb_no_text.py" "${FULL_DATASET}"
