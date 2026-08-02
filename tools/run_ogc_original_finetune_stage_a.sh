#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/media/${MEDIA_USER:-haoyi}/T9/data}"
source "${REPO_ROOT}/tools/stageb_python.sh"

STAGEA_DATASETS="${STAGEA_DATASETS:-${REPO_ROOT}/config/datasets_patch_stage_a_lvis_coco2017_local.json}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-${REPO_ROOT}/weights/groundingdino_swint_ogc.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/ogc_original_finetune_stage_a}"
ODVG_OUT_DIR="${ODVG_OUT_DIR:-${REPO_ROOT}/data/ablations/ogc_original_finetune_stage_a}"
ODVG_OUT_NAME="${ODVG_OUT_NAME:-stagea_odvg}"
BASE_CFG="${BASE_CFG:-${REPO_ROOT}/config/ablations/cfg_ogc_original_finetune_stage_a.py}"
RUN_CFG="${RUN_CFG:-${OUTPUT_DIR}/cfg_ogc_original_finetune_stage_a.generated.py}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="$(stageb_resolve_python "${PYTHON_BIN:-}")"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-12}"
AMP="${AMP:-1}"
SAVE_LOG="${SAVE_LOG:-1}"
PRINT_ONLY="${PRINT_ONLY:-0}"
REUSE_ODVG="${REUSE_ODVG:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-50000}"
STAGE_A_LOG="${STAGE_A_LOG:-}"
MATCH_EPOCHS="${MATCH_EPOCHS:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME="${RESUME:-}"

mkdir -p "${OUTPUT_DIR}" "${ODVG_OUT_DIR}"

DATASETS_JSON="${ODVG_OUT_DIR}/${ODVG_OUT_NAME}_datasets.json"
LABEL_MAP_JSON="${ODVG_OUT_DIR}/${ODVG_OUT_NAME}_canonical_label_map.json"

STATS_JSON="${ODVG_OUT_DIR}/${ODVG_OUT_NAME}_stats.json"
if [[ "${REUSE_ODVG}" == "1" && -s "${DATASETS_JSON}" && -s "${LABEL_MAP_JSON}" && -s "${STATS_JSON}" ]]; then
  echo "Reusing existing ODVG files under ${ODVG_OUT_DIR}"
else
  DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${REPO_ROOT}/tools/build_stagea_odvg_finetune_ablation.py" \
    --stagea_datasets "${STAGEA_DATASETS}" \
    --out_dir "${ODVG_OUT_DIR}" \
    --out_name "${ODVG_OUT_NAME}" \
    --progress_interval "${PROGRESS_INTERVAL}"
fi

infer_epochs_from_log() {
  local log_path="$1"
  "${PYTHON_BIN}" - "$log_path" <<'PY'
import json
import sys

rows = 0
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except Exception:
            continue
        rows += 1
print(rows)
PY
}

if [[ -z "${MATCH_EPOCHS}" ]]; then
  : "${STAGE_A_LOG:?Set STAGE_A_LOG=/path/to/stageA/log.txt or MATCH_EPOCHS=<completed_stage_a_epochs> to match sample exposure.}"
  if [[ ! -f "${STAGE_A_LOG}" ]]; then
    echo "Missing STAGE_A_LOG=${STAGE_A_LOG}" >&2
    exit 1
  fi
  MATCH_EPOCHS="$(infer_epochs_from_log "${STAGE_A_LOG}")"
fi
MATCH_EPOCHS="$(printf '%s' "${MATCH_EPOCHS}" | tr -d '[:space:]')"
if [[ "${MATCH_EPOCHS}" -le 0 ]]; then
  echo "MATCH_EPOCHS must be > 0, got ${MATCH_EPOCHS}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${BASE_CFG}" "${LABEL_MAP_JSON}" "${RUN_CFG}" "${MATCH_EPOCHS}" "${BATCH_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

base_cfg = Path(sys.argv[1]).resolve()
label_map_path = Path(sys.argv[2]).resolve()
run_cfg = Path(sys.argv[3]).resolve()
match_epochs = int(sys.argv[4])
batch_size = int(sys.argv[5])

with label_map_path.open("r", encoding="utf-8") as f:
    label_map = json.load(f)
labels = [label_map[k] for k in sorted(label_map, key=lambda x: int(x))]

run_cfg.parent.mkdir(parents=True, exist_ok=True)
base_text = base_cfg.read_text(encoding="utf-8")
run_cfg.write_text(
    base_text.rstrip()
    + "\n\n"
    f"epochs = {match_epochs}\n"
    f"batch_size = {batch_size}\n"
    f"label_list = {labels!r}\n",
    encoding="utf-8",
)
print(json.dumps({"run_cfg": str(run_cfg), "num_labels": len(labels), "epochs": match_epochs, "batch_size": batch_size}, indent=2))
PY

if [[ ! -f "${PRETRAIN_MODEL_PATH}" ]]; then
  echo "Missing PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH}" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/main.py"
  -c "${RUN_CFG}"
  --datasets "${DATASETS_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --pretrain_model_path "${PRETRAIN_MODEL_PATH}"
  --num_workers "${NUM_WORKERS}"
)

if [[ -z "${RESUME}" && "${AUTO_RESUME}" == "1" ]]; then
  if [[ -f "${OUTPUT_DIR}/checkpoint_iter.pth" ]]; then
    RESUME="${OUTPUT_DIR}/checkpoint_iter.pth"
  elif [[ -f "${OUTPUT_DIR}/checkpoint.pth" ]]; then
    RESUME="${OUTPUT_DIR}/checkpoint.pth"
  fi
fi
if [[ -n "${RESUME}" ]]; then
  if [[ ! -f "${RESUME}" ]]; then
    echo "Missing RESUME=${RESUME}" >&2
    exit 1
  fi
  CMD+=(--resume "${RESUME}")
fi

if [[ "${AMP}" == "1" ]]; then
  CMD+=(--amp)
fi
if [[ "${SAVE_LOG}" == "1" ]]; then
  CMD+=(--save_log)
fi

echo "Running OGC original-training finetune ablation"
echo "  repo: ${REPO_ROOT}"
echo "  datasets: ${DATASETS_JSON}"
echo "  cfg: ${RUN_CFG}"
echo "  output: ${OUTPUT_DIR}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  PYTHON_BIN: ${PYTHON_BIN}"
echo "  MATCH_EPOCHS: ${MATCH_EPOCHS}"
echo "  BATCH_SIZE: ${BATCH_SIZE}"
echo "  RESUME: ${RESUME:-<none>}"
echo

cd "${REPO_ROOT}"
if [[ "${PRINT_ONLY}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q DATA_ROOT=%q PYTHONPATH=%q' \
    "${CUDA_VISIBLE_DEVICES}" "${DATA_ROOT}" "${REPO_ROOT}:${PYTHONPATH:-}"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DATA_ROOT="${DATA_ROOT}" PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  "${CMD[@]}"
