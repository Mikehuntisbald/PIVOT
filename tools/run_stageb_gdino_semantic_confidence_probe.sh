#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/run_stageb_gdino_semantic_confidence_probe.sh [options]

Options:
  --source-kind KIND       rank or dataft-confidence (default: rank)
  --source-checkpoint PATH Audited R/C milestone used only with pretrain_model_path
  --source-audit PATH      Audit sidecar for the selected R/C milestone
  --output-root PATH       Semantic probe output root
  --continue               Continue an interrupted semantic-scope run
  --dry-run                Run static audits and print launch commands only
  --num-workers N          DataLoader workers per rank (default: 4)
  --prefetch-factor N      DataLoader prefetch factor (default: 1)
  --master-port PORT       DDP master port (default: 29529)
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
AUDITOR="tools/stageb_gdino_semantic_probe_audit.py"
CONFIG="config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py"
DATASETS="config/datasets_stageb_gdino_adapter_semantic_verified_pairs.json"
SOURCE_KIND="rank"
SOURCE_CHECKPOINT=""
SOURCE_AUDIT=""
OUTPUT_ROOT="outputs/stageb_gdino_adapter_semantic_confidence_probe_20260711"
CONTINUE_RUN=0
DRY_RUN=0
NUM_WORKERS=4
PREFETCH_FACTOR=1
MASTER_PORT=29529
WORLD_SIZE=2
PER_GPU_BATCH=4
ITER_CHECKPOINT_INTERVAL=50
MILESTONES=(50 100 250 500)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-kind) SOURCE_KIND="$2"; shift 2 ;;
        --source-checkpoint) SOURCE_CHECKPOINT="$2"; shift 2 ;;
        --source-audit) SOURCE_AUDIT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --continue) CONTINUE_RUN=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[FAIL] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${SOURCE_KIND}" in
    rank|dataft-confidence) ;;
    *) echo "[FAIL] --source-kind must be rank or dataft-confidence" >&2; exit 2 ;;
esac
if [[ "${NUM_WORKERS}" -lt 0 || "${PREFETCH_FACTOR}" -lt 1 ]]; then
    echo "[FAIL] invalid DataLoader worker/prefetch settings" >&2
    exit 2
fi

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    if [[ "${SOURCE_KIND}" == "rank" ]]; then
        SOURCE_CHECKPOINT="outputs/stageb_gdino_adapter_two_phase_probe_20260711/rank/milestones/checkpoint_iter_000250.pth"
    else
        SOURCE_CHECKPOINT="outputs/stageb_gdino_adapter_two_phase_probe_20260711/confidence/milestones/checkpoint_iter_000250.pth"
    fi
fi
if [[ -z "${SOURCE_AUDIT}" ]]; then
    SOURCE_AUDIT="${SOURCE_CHECKPOINT%.pth}.audit.json"
fi

absolute_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

milestone_path() {
    printf '%s/milestones/checkpoint_iter_%06d.pth\n' "$1" "$2"
}

milestone_audit_path() {
    printf '%s/milestones/checkpoint_iter_%06d.audit.json\n' "$1" "$2"
}

audit_segment_path() {
    "${PYTHON_BIN}" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["segment_lineage"]["path"])' \
        "$1"
}

build_train_command() {
    local target="$1"
    local mode="$2"
    local source="$3"
    TRAIN_COMMAND=(
        "${PYTHON_BIN}" -m torch.distributed.run
        --nproc_per_node="${WORLD_SIZE}"
        --master_port="${MASTER_PORT}"
        main.py
        -c "${CONFIG}"
        --datasets "${DATASETS}"
        --output_dir "${OUTPUT_ROOT}"
        --max_train_iters "${target}"
        --iter_checkpoint_interval "${ITER_CHECKPOINT_INTERVAL}"
        --num_workers "${NUM_WORKERS}"
        --prefetch_factor "${PREFETCH_FACTOR}"
        --amp
        --save_log
        --options batch_size="${PER_GPU_BATCH}"
    )
    if [[ "${mode}" == "pretrain" ]]; then
        TRAIN_COMMAND+=(--pretrain_model_path "${source}")
    elif [[ "${mode}" == "resume" ]]; then
        TRAIN_COMMAND+=(--resume "${source}")
    else
        echo "[FAIL] invalid internal initialization mode: ${mode}" >&2
        exit 2
    fi
}

if [[ "${DRY_RUN}" == "1" ]]; then
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" "${AUDITOR}" static >/dev/null
    echo "[OK] semantic data/config/code static audit passed"
    dry_source="${SOURCE_CHECKPOINT}"
    dry_mode="pretrain"
    for target in "${MILESTONES[@]}"; do
        build_train_command "${target}" "${dry_mode}" "${dry_source}"
        print_command "${TRAIN_COMMAND[@]}"
        dry_source="$(milestone_path "${OUTPUT_ROOT}" "${target}")"
        dry_mode="resume"
    done
    exit 0
fi

SOURCE_CHECKPOINT="$(absolute_path "${SOURCE_CHECKPOINT}")"
SOURCE_AUDIT="$(absolute_path "${SOURCE_AUDIT}")"
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT}")"
PREFLIGHT="${OUTPUT_ROOT}/probe_preflight.json"
LIVE="${OUTPUT_ROOT}/checkpoint_iter.pth"

preflight_command=(
    "${PYTHON_BIN}" "${AUDITOR}" preflight
    --source-kind "${SOURCE_KIND}"
    --source-checkpoint "${SOURCE_CHECKPOINT}"
    --source-audit "${SOURCE_AUDIT}"
    --world-size "${WORLD_SIZE}"
    --per-gpu-batch "${PER_GPU_BATCH}"
    --output "${PREFLIGHT}"
)
if [[ -f "${PREFLIGHT}" ]]; then
    if [[ "${CONTINUE_RUN}" != "1" ]]; then
        echo "[FAIL] output already has a preflight; pass --continue: ${OUTPUT_ROOT}" >&2
        exit 2
    fi
    preflight_command+=(--continue-run)
elif [[ -d "${OUTPUT_ROOT}" && -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[FAIL] fresh semantic output directory is not empty: ${OUTPUT_ROOT}" >&2
    exit 2
fi
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${preflight_command[@]}"
mkdir -p "${OUTPUT_ROOT}/milestones" "${OUTPUT_ROOT}/recovery" "${OUTPUT_ROOT}/launches"

previous_snapshot=""
previous_audit=""
previous_iteration=0

for target in "${MILESTONES[@]}"; do
    snapshot="$(milestone_path "${OUTPUT_ROOT}" "${target}")"
    audit="$(milestone_audit_path "${OUTPUT_ROOT}" "${target}")"
    if [[ -f "${snapshot}" || -f "${audit}" ]]; then
        if [[ ! -f "${snapshot}" || ! -f "${audit}" ]]; then
            echo "[FAIL] incomplete preserved semantic milestone ${target}" >&2
            exit 2
        fi
        audited_segment="$(audit_segment_path "${audit}")"
        verify=(
            "${PYTHON_BIN}" "${AUDITOR}" milestone
            --checkpoint "${snapshot}"
            --preflight "${PREFLIGHT}"
            --expected-iteration "${target}"
            --segment-lineage "${audited_segment}"
            --output "${audit}"
            --verify-only
        )
        if [[ -n "${previous_audit}" ]]; then
            verify+=(--previous-audit "${previous_audit}")
        fi
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${verify[@]}"
        previous_snapshot="${snapshot}"
        previous_audit="${audit}"
        previous_iteration="${target}"
        continue
    fi

    initialization_mode=""
    source_checkpoint=""
    live_iteration=0
    recovery_inspection=""
    if [[ -f "${LIVE}" ]]; then
        live_iteration="$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PYTHON_BIN}" "${AUDITOR}" metadata --checkpoint "${LIVE}" --field iteration)"
    fi
    if [[ "${live_iteration}" -gt "${previous_iteration}" ]]; then
        if [[ "${CONTINUE_RUN}" != "1" ]]; then
            echo "[FAIL] found unpreserved semantic live state; inspect and pass --continue" >&2
            exit 2
        fi
        mapfile -t lineage_files < <(
            find "${OUTPUT_ROOT}/launches" -maxdepth 1 -type f \
                -name "target_$(printf '%06d' "${target}")_attempt_*.lineage.json" | sort
        )
        if [[ "${#lineage_files[@]}" -lt 1 ]]; then
            echo "[FAIL] live semantic checkpoint has no recorded segment ancestry" >&2
            exit 2
        fi
        segment_lineage="${lineage_files[$((${#lineage_files[@]} - 1))]}"
        live_sha="$(sha256sum "${LIVE}" | awk '{print $1}')"
        lineage_source_sha="$("${PYTHON_BIN}" -c \
            'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source_checkpoint"]["sha256"])' \
            "${segment_lineage}")"
        if [[ "${live_sha}" == "${lineage_source_sha}" ]]; then
            if [[ "${live_iteration}" -eq "${target}" ]]; then
                echo "[FAIL] exact-target LIVE unexpectedly equals its segment source" >&2
                exit 2
            fi
            source_checkpoint="$("${PYTHON_BIN}" -c \
                'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source_checkpoint"]["path"])' \
                "${segment_lineage}")"
            initialization_mode="$("${PYTHON_BIN}" -c \
                'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["initialization_mode"])' \
                "${segment_lineage}")"
            recovery_inspection="$("${PYTHON_BIN}" -c \
                'import json,sys; v=json.load(open(sys.argv[1], encoding="utf-8"))["recovery_inspection"]; print("" if v is None else v["path"])' \
                "${segment_lineage}")"
        else
            inspect_output="${OUTPUT_ROOT}/recovery/live_target_${target}_iter_${live_iteration}_${live_sha:0:12}.audit.json"
            inspect=(
                "${PYTHON_BIN}" "${AUDITOR}" inspect
                --checkpoint "${LIVE}"
                --preflight "${PREFLIGHT}"
                --expected-target "${target}"
                --segment-lineage "${segment_lineage}"
                --output "${inspect_output}"
            )
            if [[ -n "${previous_audit}" ]]; then
                inspect+=(--previous-audit "${previous_audit}")
            fi
            PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${inspect[@]}" >/dev/null
            if [[ "${live_iteration}" -eq "${target}" ]]; then
                cp --reflink=auto --preserve=timestamps "${LIVE}" "${snapshot}"
                milestone_command=(
                    "${PYTHON_BIN}" "${AUDITOR}" milestone
                    --checkpoint "${snapshot}"
                    --preflight "${PREFLIGHT}"
                    --expected-iteration "${target}"
                    --segment-lineage "${segment_lineage}"
                    --output "${audit}"
                )
                if [[ -n "${previous_audit}" ]]; then
                    milestone_command+=(--previous-audit "${previous_audit}")
                fi
                PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${milestone_command[@]}"
                previous_snapshot="${snapshot}"
                previous_audit="${audit}"
                previous_iteration="${target}"
                continue
            fi
            recovery="${OUTPUT_ROOT}/recovery/checkpoint_target_${target}_iter_${live_iteration}_${live_sha:0:12}.pth"
            if [[ ! -f "${recovery}" ]]; then
                cp --reflink=auto --preserve=timestamps "${LIVE}" "${recovery}"
            elif [[ "$(sha256sum "${recovery}" | awk '{print $1}')" != "${live_sha}" ]]; then
                echo "[FAIL] recovery checkpoint hash collision: ${recovery}" >&2
                exit 2
            fi
            initialization_mode="resume"
            source_checkpoint="${recovery}"
            recovery_inspection="${inspect_output}"
        fi
    elif [[ -n "${previous_snapshot}" ]]; then
        initialization_mode="resume"
        source_checkpoint="${previous_snapshot}"
    else
        initialization_mode="pretrain"
        source_checkpoint="${SOURCE_CHECKPOINT}"
    fi

    build_train_command "${target}" "${initialization_mode}" "${source_checkpoint}"
    attempt=1
    while [[ -e "${OUTPUT_ROOT}/launches/target_$(printf '%06d' "${target}")_attempt_$(printf '%02d' "${attempt}").sh" ]]; do
        attempt=$((attempt + 1))
    done
    launch="${OUTPUT_ROOT}/launches/target_$(printf '%06d' "${target}")_attempt_$(printf '%02d' "${attempt}").sh"
    segment_log="${launch%.sh}.log"
    segment_lineage="${launch%.sh}.lineage.json"
    lineage_command=(
        "${PYTHON_BIN}" "${AUDITOR}" segment-lineage
        --preflight "${PREFLIGHT}"
        --expected-target "${target}"
        --source-checkpoint "${source_checkpoint}"
        --initialization-mode "${initialization_mode}"
        --output "${segment_lineage}"
    )
    if [[ -n "${previous_audit}" ]]; then
        lineage_command+=(--previous-audit "${previous_audit}")
    fi
    if [[ -n "${recovery_inspection}" ]]; then
        lineage_command+=(--recovery-inspection "${recovery_inspection}")
    fi
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${lineage_command[@]}"
    print_command "${TRAIN_COMMAND[@]}" > "${launch}"
    echo "[RUN] semantic target=${target} mode=${initialization_mode} source=${source_checkpoint}"
    print_command "${TRAIN_COMMAND[@]}"

    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    export DATA_ROOT="${DATA_ROOT:-/home/user/datasets/pivot_data}"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/train_console.log" "${segment_log}"

    if [[ ! -f "${LIVE}" ]]; then
        echo "[FAIL] semantic target ${target} produced no checkpoint_iter.pth" >&2
        exit 2
    fi
    live_iteration="$("${PYTHON_BIN}" "${AUDITOR}" metadata --checkpoint "${LIVE}" --field iteration)"
    if [[ "${live_iteration}" -ne "${target}" ]]; then
        echo "[INCOMPLETE] semantic run stopped at ${live_iteration}; rerun with --continue" >&2
        exit 3
    fi
    cp --reflink=auto --preserve=timestamps "${LIVE}" "${snapshot}"
    milestone_command=(
        "${PYTHON_BIN}" "${AUDITOR}" milestone
        --checkpoint "${snapshot}"
        --preflight "${PREFLIGHT}"
        --expected-iteration "${target}"
        --segment-lineage "${segment_lineage}"
        --output "${audit}"
    )
    if [[ -n "${previous_audit}" ]]; then
        milestone_command+=(--previous-audit "${previous_audit}")
    fi
    "${milestone_command[@]}"
    previous_snapshot="${snapshot}"
    previous_audit="${audit}"
    previous_iteration="${target}"
done

echo "[OK] completed semantic confidence milestones: ${OUTPUT_ROOT}/milestones"
