#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    cat <<'EOF'
Usage: tools/run_stageb_gdino_fixed_top1_calibration.sh [options]

Options:
  --probe-root DIR              Completed fixed-top1 training root
  --partition-audit PATH        Sealed train/calibration partition audit
  --p0-checkpoint PATH          Exact-identity P0 checkpoint
  --p0-audit PATH               P0 sidecar (default: CHECKPOINT.audit.json)
  --baseline-checkpoint PATH    Authoritative fixed pure Stage-B baseline
  --output-root DIR             Must equal PROBE_ROOT/calibration_selection
  --data-root DIR               Dataset root
  --device DEVICE               Evaluation device (default: cuda:0)
  --num-workers N               DataLoader workers (default: 4)
  --dry-run                     Verify CPU inputs and print commands only
  --python PATH                 Python executable (default: /usr/bin/python3)
EOF
}

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
EVALUATOR="tools/eval_stageb_gdino_fixed_top1_calibration.py"
SELECTOR="tools/stageb_gdino_fixed_top1_selection.py"
AUDITOR="tools/stageb_gdino_fixed_top1_probe_audit.py"
PROBE_ROOT="outputs/stageb_gdino_adapter_fixed_top1_confidence_probe_20260712"
PARTITION_AUDIT="data/ablations/stageb_gdino_adapter_fixed_top1_verified_20260712/partition_audit.json"
P0_CHECKPOINT="outputs/stageb_gdino_adapter_p0_20260711/checkpoint_p0.pth"
P0_AUDIT=""
BASELINE_CHECKPOINT="outputs/gdino_ft_stage_b_fixed_baseline_20260711/checkpoint0000.pth"
OUTPUT_ROOT=""
DATA_ROOT_VALUE="${DATA_ROOT:-/home/user/datasets/pivot_data}"
DEVICE="cuda:0"
NUM_WORKERS=4
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --probe-root) PROBE_ROOT="$2"; shift 2 ;;
        --partition-audit) PARTITION_AUDIT="$2"; shift 2 ;;
        --p0-checkpoint) P0_CHECKPOINT="$2"; shift 2 ;;
        --p0-audit) P0_AUDIT="$2"; shift 2 ;;
        --baseline-checkpoint) BASELINE_CHECKPOINT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --data-root) DATA_ROOT_VALUE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[FAIL] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

absolute_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

PROBE_ROOT="$(absolute_path "${PROBE_ROOT}")"
PARTITION_AUDIT="$(absolute_path "${PARTITION_AUDIT}")"
P0_CHECKPOINT="$(absolute_path "${P0_CHECKPOINT}")"
BASELINE_CHECKPOINT="$(absolute_path "${BASELINE_CHECKPOINT}")"
DATA_ROOT_VALUE="$(absolute_path "${DATA_ROOT_VALUE}")"
if [[ "${DATA_ROOT_VALUE}" != "/home/user/datasets/pivot_data" ]]; then
    echo "[FAIL] calibration data root is locked to the partition image-identity root" >&2
    exit 2
fi
PREFLIGHT="${PROBE_ROOT}/probe_preflight.json"
if [[ -z "${P0_AUDIT}" ]]; then
    P0_AUDIT="${P0_CHECKPOINT}.audit.json"
else
    P0_AUDIT="$(absolute_path "${P0_AUDIT}")"
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="${PROBE_ROOT}/calibration_selection"
else
    OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT}")"
fi
if [[ "${OUTPUT_ROOT}" != "${PROBE_ROOT}/calibration_selection" ]]; then
    echo "[FAIL] calibration output root is fixed under the sealed probe root" >&2
    exit 2
fi
SELECTION_AUDIT="${PROBE_ROOT}/selection/selected_milestone.json"

if [[ ! -f "${PREFLIGHT}" || ! -f "${PARTITION_AUDIT}" ]]; then
    echo "[FAIL] probe preflight or partition audit is missing" >&2
    exit 2
fi
if [[ ! -f "${P0_CHECKPOINT}" || ! -f "${P0_AUDIT}" || ! -f "${BASELINE_CHECKPOINT}" ]]; then
    echo "[FAIL] P0 checkpoint/audit or baseline checkpoint is missing" >&2
    exit 2
fi
if [[ "${NUM_WORKERS}" -lt 0 ]]; then
    echo "[FAIL] --num-workers must be non-negative" >&2
    exit 2
fi

"${PYTHON_BIN}" "${SELECTOR}" verify-partition --audit "${PARTITION_AUDIT}" >/dev/null
mapfile -t MILESTONES < <(
    "${PYTHON_BIN}" -c \
        'import json,sys; print("\n".join(str(x) for x in json.load(open(sys.argv[1], encoding="utf-8"))["launch"]["milestones"]))' \
        "${PREFLIGHT}"
)
if [[ "${#MILESTONES[@]}" -lt 1 ]]; then
    echo "[FAIL] probe preflight has no milestones" >&2
    exit 2
fi

build_eval_command() {
    local role="$1"
    local checkpoint="$2"
    local checkpoint_audit="$3"
    local output_dir="$4"
    EVAL_COMMAND=(
        "${PYTHON_BIN}" "${EVALUATOR}"
        --checkpoint "${checkpoint}"
        --checkpoint-audit "${checkpoint_audit}"
        --role "${role}"
        --probe-preflight "${PREFLIGHT}"
        --partition-audit "${PARTITION_AUDIT}"
        --output-dir "${output_dir}"
        --data-root "${DATA_ROOT_VALUE}"
        --device "${DEVICE}"
        --num-workers "${NUM_WORKERS}"
    )
}

run_or_verify() {
    local output_dir="$1"
    shift
    local command=("$@")
    if [[ -f "${output_dir}/calibration_eval_complete.json" ]]; then
        "${command[@]}" --verify-only >/dev/null
        echo "[OK] reused audited calibration evaluation: ${output_dir}"
    elif [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "[FAIL] partial calibration output is not reusable: ${output_dir}" >&2
        exit 2
    else
        "${command[@]}"
    fi
}

P0_OUTPUT="${OUTPUT_ROOT}/p0"
build_eval_command p0 "${P0_CHECKPOINT}" "${P0_AUDIT}" "${P0_OUTPUT}"
EVAL_COMMAND+=(--baseline-checkpoint "${BASELINE_CHECKPOINT}")
if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "${EVAL_COMMAND[@]}"
else
    run_or_verify "${P0_OUTPUT}" "${EVAL_COMMAND[@]}"
fi

for iteration in "${MILESTONES[@]}"; do
    printf -v padded '%06d' "${iteration}"
    checkpoint="${PROBE_ROOT}/milestones/checkpoint_iter_${padded}.pth"
    checkpoint_audit="${PROBE_ROOT}/milestones/checkpoint_iter_${padded}.audit.json"
    output_dir="${OUTPUT_ROOT}/s${padded}"
    if [[ ! -f "${checkpoint}" || ! -f "${checkpoint_audit}" ]]; then
        echo "[FAIL] missing complete fixed-top1 milestone ${iteration}" >&2
        exit 2
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        "${PYTHON_BIN}" "${AUDITOR}" verify-calibration \
            --checkpoint "${checkpoint}" \
            --audit "${checkpoint_audit}" \
            --expected-iteration "${iteration}" >/dev/null
    fi
    build_eval_command milestone "${checkpoint}" "${checkpoint_audit}" "${output_dir}"
    EVAL_COMMAND+=(--iteration "${iteration}")
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command "${EVAL_COMMAND[@]}"
    else
        run_or_verify "${output_dir}" "${EVAL_COMMAND[@]}"
    fi
done

SELECT_COMMAND=(
    "${PYTHON_BIN}" "${SELECTOR}" select
    --probe-preflight "${PREFLIGHT}"
    --calibration-root "${OUTPUT_ROOT}"
    --output "${SELECTION_AUDIT}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "${SELECT_COMMAND[@]}"
    echo "[DRY RUN] no calibration or selection output was created"
elif [[ -f "${SELECTION_AUDIT}" ]]; then
    "${PYTHON_BIN}" "${SELECTOR}" verify-selection \
        --audit "${SELECTION_AUDIT}" \
        --calibration-root "${OUTPUT_ROOT}" >/dev/null
    echo "[OK] reused audited unique milestone selection: ${SELECTION_AUDIT}"
else
    "${SELECT_COMMAND[@]}"
    echo "[OK] wrote unique held-out milestone selection: ${SELECTION_AUDIT}"
fi
