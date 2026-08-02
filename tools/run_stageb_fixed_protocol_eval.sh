#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/tools/stageb_python.sh"

usage() {
    cat <<'EOF'
Usage: tools/run_stageb_fixed_protocol_eval.sh \
  --config CONFIG.py --checkpoint CHECKPOINT.pth --output-dir OUTPUT_DIR [options]

Options:
  --data-root DIR       Dataset root (default: $DATA_ROOT or /home/user/datasets/pivot_data)
  --device DEVICE       Evaluation device (default: cuda:0)
  --batch-size N        Fixed inference batch size (default: 16)
  --num-workers N       DataLoader workers (default: 4)
  --seed N              Evaluation seed (default: 42)
  --no-amp              Disable AMP; use the same setting for baseline and candidate
  --python PATH         Python executable (default: /usr/bin/python3)
  --dry-run             Audit inputs and print commands without evaluating
EOF
}

CONFIG=""
CHECKPOINT=""
OUTPUT_DIR=""
DATA_ROOT_VALUE="${DATA_ROOT:-/home/user/datasets/pivot_data}"
DEVICE="cuda:0"
BATCH_SIZE=16
NUM_WORKERS=4
SEED=42
AMP=1
PYTHON_BIN="$(stageb_resolve_python "${PYTHON_BIN:-}")"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT_VALUE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-amp)
            AMP=0
            shift
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[FAIL] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${CONFIG}" || -z "${CHECKPOINT}" || -z "${OUTPUT_DIR}" ]]; then
    usage >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}/ref8/summary.json" || -e "${OUTPUT_DIR}/strict2031/summary.json" || -e "${OUTPUT_DIR}/strict1607/summary.json" ]]; then
    echo "[FAIL] output directory already contains protocol results: ${OUTPUT_DIR}" >&2
    echo "Use a fresh output directory to prevent record mixing." >&2
    exit 2
fi

PRIMARY_MANIFEST="data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/eval_manifest.jsonl"
SUPPLEMENTAL_MANIFEST="data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/semantic_stageb_union_image_disjoint_manifest.jsonl"
EVALUATOR="tools/eval_text_groundingdino_refcoco_tn.py"

mkdir -p "${OUTPUT_DIR}"

audit_command=(
    "${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py eval-preflight
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --data_root "${DATA_ROOT_VALUE}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --output "${OUTPUT_DIR}/protocol_eval_preflight.json"
)
if [[ "${AMP}" == "1" ]]; then
    audit_command+=(--amp)
fi
"${audit_command[@]}"

common=(
    "${PYTHON_BIN}" "${EVALUATOR}"
    --config "${CONFIG}"
    --ckpts "${CHECKPOINT}"
    --data_root "${DATA_ROOT_VALUE}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --topk 1
    --threshold_tprs 0.75 0.9 0.95
    --score_thresholds 0.5
    --log_every 50
)
if [[ "${AMP}" == "1" ]]; then
    common+=(--amp)
fi

ref_command=(
    "${common[@]}"
    --output_dir "${OUTPUT_DIR}/ref8"
    --skip_tn
    --ref_splits all
)
primary_command=(
    "${common[@]}"
    --output_dir "${OUTPUT_DIR}/strict2031"
    --skip_ref
    --tn_jsonl "${PRIMARY_MANIFEST}"
    --tn_splits refcocop_val refcocog_umd_val
)
supplemental_command=(
    "${common[@]}"
    --output_dir "${OUTPUT_DIR}/strict1607"
    --skip_ref
    --tn_jsonl "${SUPPLEMENTAL_MANIFEST}"
    --tn_splits refcocop_val refcocog_umd_val
)

{
    printf '%q ' "${ref_command[@]}"
    printf '\n'
    printf '%q ' "${primary_command[@]}"
    printf '\n'
    printf '%q ' "${supplemental_command[@]}"
    printf '\n'
} > "${OUTPUT_DIR}/launch_commands.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY RUN] fixed evaluation commands:"
    cat "${OUTPUT_DIR}/launch_commands.txt"
    exit 0
fi

export DATA_ROOT="${DATA_ROOT_VALUE}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"${ref_command[@]}" 2>&1 | tee "${OUTPUT_DIR}/ref8_console.log"
"${primary_command[@]}" 2>&1 | tee "${OUTPUT_DIR}/strict2031_console.log"
"${supplemental_command[@]}" 2>&1 | tee "${OUTPUT_DIR}/strict1607_console.log"

"${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py eval-postflight \
    --output_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/protocol_eval_complete.json"

echo "[OK] fixed protocol evaluation completed: ${OUTPUT_DIR}"
