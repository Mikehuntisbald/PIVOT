#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/tools/stageb_python.sh"

usage() {
    cat <<'EOF'
Usage: tools/run_stageb_gdino_adapter_two_phase_probe.sh [options]

Options:
  --phase rank|confidence|both   Phase(s) to run (default: both)
  --baseline-checkpoint PATH    Fixed pure Stage-B data-FT checkpoint
  --rank-checkpoint PATH        Audited R checkpoint used to initialize C
  --rank-audit PATH             Milestone audit for --rank-checkpoint
  --rank-selection N            Audited R milestone selected for C (default: 500)
  --rank-max-target N           Last R milestone to run (default: 500)
  --confidence-max-target N     Last C milestone to run (default: 500)
  --output-root DIR             Probe output root
  --continue                    Continue an existing, audited output directory
  --dry-run                     Run static input audits and print commands only
  --python PATH                 Python executable (default: /usr/bin/python3)
  --num-workers N               Workers per rank (default: 4)
  --prefetch-factor N           DataLoader prefetch factor (default: 1)

Environment:
  CUDA_VISIBLE_DEVICES          Visible GPU(s) (default: 0 for world 1; 0,1 otherwise)
  STAGEB_ADAPTER_WORLD_SIZE     Number of training processes (default: 2)
  STAGEB_ADAPTER_PER_GPU_BATCH  Batch size per process (default: 4)
  RANK_MASTER_PORT              Rank phase DDP port (default: 29521)
  CONFIDENCE_MASTER_PORT        Confidence phase DDP port (default: 29522)
EOF
}

PHASE="both"
BASELINE_CHECKPOINT="outputs/gdino_ft_stage_b_fixed_baseline_20260711/checkpoint0000.pth"
RANK_CHECKPOINT=""
RANK_AUDIT=""
RANK_SELECTION=500
RANK_MAX_TARGET=500
OUTPUT_ROOT="outputs/stageb_gdino_adapter_two_phase_probe_20260711"
CONTINUE_RUN=0
DRY_RUN=0
PYTHON_BIN="$(stageb_resolve_python "${PYTHON_BIN:-}")"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
RANK_MASTER_PORT="${RANK_MASTER_PORT:-29521}"
CONFIDENCE_MASTER_PORT="${CONFIDENCE_MASTER_PORT:-29522}"
WORLD_SIZE="${STAGEB_ADAPTER_WORLD_SIZE:-2}"
PER_GPU_BATCH="${STAGEB_ADAPTER_PER_GPU_BATCH:-4}"
ITER_CHECKPOINT_INTERVAL=50
RANK_MILESTONES=(50 100 250 500 1000 2000 5000)
CONFIDENCE_MILESTONES=(50 100 250 500)
CONFIDENCE_MAX_TARGET=500
AUDITOR="${STAGEB_ADAPTER_AUDITOR:-tools/stageb_gdino_adapter_probe_audit.py}"
CONFIDENCE_CONFIG="${STAGEB_CONFIDENCE_CONFIG:-config/ablations/cfg_stageb_gdino_score_adapter_dataft.py}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --baseline-checkpoint) BASELINE_CHECKPOINT="$2"; shift 2 ;;
        --rank-checkpoint) RANK_CHECKPOINT="$2"; shift 2 ;;
        --rank-audit) RANK_AUDIT="$2"; shift 2 ;;
        --rank-selection) RANK_SELECTION="$2"; shift 2 ;;
        --rank-max-target) RANK_MAX_TARGET="$2"; shift 2 ;;
        --confidence-max-target) CONFIDENCE_MAX_TARGET="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --continue) CONTINUE_RUN=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[FAIL] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${PHASE}" != "rank" && "${PHASE}" != "confidence" && "${PHASE}" != "both" ]]; then
    echo "[FAIL] --phase must be rank, confidence, or both" >&2
    exit 2
fi
case "${RANK_SELECTION}" in
    50|100|250|500|1000|2000|5000) ;;
    *) echo "[FAIL] --rank-selection must be 50, 100, 250, 500, 1000, 2000, or 5000" >&2; exit 2 ;;
esac
case "${RANK_MAX_TARGET}" in
    50|100|250|500|1000|2000|5000) ;;
    *) echo "[FAIL] --rank-max-target must be 50, 100, 250, 500, 1000, 2000, or 5000" >&2; exit 2 ;;
esac
case "${CONFIDENCE_MAX_TARGET}" in
    50|100|250|500) ;;
    *) echo "[FAIL] --confidence-max-target must be 50, 100, 250, or 500" >&2; exit 2 ;;
esac
if [[ "${WORLD_SIZE}" -lt 1 || "${PER_GPU_BATCH}" -lt 1 ]]; then
    echo "[FAIL] invalid adapter world-size/per-GPU batch settings" >&2
    exit 2
fi
if [[ "${PHASE}" == "both" && -z "${RANK_CHECKPOINT}" && "${RANK_SELECTION}" -gt "${RANK_MAX_TARGET}" ]]; then
    echo "[FAIL] --phase both cannot select R${RANK_SELECTION} when --rank-max-target is ${RANK_MAX_TARGET}" >&2
    exit 2
fi
if [[ "${NUM_WORKERS}" -lt 0 || "${PREFETCH_FACTOR}" -lt 1 ]]; then
    echo "[FAIL] invalid DataLoader worker/prefetch settings" >&2
    exit 2
fi

absolute_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

phase_config() {
    if [[ "$1" == "rank" ]]; then
        printf '%s\n' "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py"
    else
        printf '%s\n' "${CONFIDENCE_CONFIG}"
    fi
}

phase_datasets() {
    if [[ "$1" == "rank" ]]; then
        printf '%s\n' "config/datasets_stageb_gdino_adapter_rank_three_ref.json"
    else
        printf '%s\n' "config/datasets_stageb_gdino_adapter_dataft_pairs.json"
    fi
}

phase_port() {
    if [[ "$1" == "rank" ]]; then
        printf '%s\n' "${RANK_MASTER_PORT}"
    else
        printf '%s\n' "${CONFIDENCE_MASTER_PORT}"
    fi
}

phase_milestones() {
    local phase="$1"
    local target
    if [[ "${phase}" == "rank" ]]; then
        for target in "${RANK_MILESTONES[@]}"; do
            if [[ "${target}" -le "${RANK_MAX_TARGET}" ]]; then
                printf '%s\n' "${target}"
            fi
        done
    else
        for target in "${CONFIDENCE_MILESTONES[@]}"; do
            if [[ "${target}" -le "${CONFIDENCE_MAX_TARGET}" ]]; then
                printf '%s\n' "${target}"
            fi
        done
    fi
}

milestone_path() {
    local phase_dir="$1"
    local target="$2"
    printf '%s/milestones/checkpoint_iter_%06d.pth\n' "${phase_dir}" "${target}"
}

milestone_audit_path() {
    local phase_dir="$1"
    local target="$2"
    printf '%s/milestones/checkpoint_iter_%06d.audit.json\n' "${phase_dir}" "${target}"
}

json_source_path() {
    "${PYTHON_BIN}" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source_checkpoint"]["path"])' \
        "$1"
}

json_record_path() {
    "${PYTHON_BIN}" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]["path"])' \
        "$1" "$2"
}

run_preflight() {
    local phase="$1"
    local phase_dir="$2"
    local initial_checkpoint="$3"
    local initial_audit="$4"
    local preflight="${phase_dir}/probe_preflight.json"
    local command=(
        "${PYTHON_BIN}" "${AUDITOR}" phase-preflight
        --phase "${phase}"
        --initial-checkpoint "${initial_checkpoint}"
        --output-dir "${phase_dir}"
        --world-size "${WORLD_SIZE}"
        --per-gpu-batch "${PER_GPU_BATCH}"
        --output "${preflight}"
    )
    if [[ -n "${initial_audit}" ]]; then
        command+=(--initial-audit "${initial_audit}")
    fi
    if [[ -f "${preflight}" ]]; then
        if [[ "${CONTINUE_RUN}" != "1" ]]; then
            echo "[FAIL] ${phase} output already has a preflight; pass --continue: ${phase_dir}" >&2
            exit 2
        fi
        command+=(--continue-run)
    elif [[ -d "${phase_dir}" && -n "$(find "${phase_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "[FAIL] fresh ${phase} output directory is not empty: ${phase_dir}" >&2
        exit 2
    fi
    "${command[@]}"
}

build_train_command() {
    local phase="$1"
    local phase_dir="$2"
    local target="$3"
    local initialization_mode="$4"
    local source_checkpoint="$5"
    local config datasets port
    config="$(phase_config "${phase}")"
    datasets="$(phase_datasets "${phase}")"
    port="$(phase_port "${phase}")"
    local -a train_entry
    if [[ "${WORLD_SIZE}" -eq 1 ]]; then
        train_entry=("${PYTHON_BIN}" main.py)
    else
        train_entry=(
            "${PYTHON_BIN}" -m torch.distributed.run
            --nproc_per_node="${WORLD_SIZE}"
            --master_port="${port}"
            main.py
        )
    fi
    TRAIN_COMMAND=(
        "${train_entry[@]}"
        -c "${config}"
        --datasets "${datasets}"
        --output_dir "${phase_dir}"
        --max_train_iters "${target}"
        --iter_checkpoint_interval "${ITER_CHECKPOINT_INTERVAL}"
        --num_workers "${NUM_WORKERS}"
        --prefetch_factor "${PREFETCH_FACTOR}"
        --amp
        --save_log
        --options batch_size="${PER_GPU_BATCH}"
    )
    if [[ "${initialization_mode}" == "pretrain" ]]; then
        TRAIN_COMMAND+=(--pretrain_model_path "${source_checkpoint}")
    elif [[ "${initialization_mode}" == "resume" ]]; then
        TRAIN_COMMAND+=(--resume "${source_checkpoint}")
    else
        echo "[FAIL] internal invalid initialization mode: ${initialization_mode}" >&2
        exit 2
    fi
}

audit_milestone() {
    local phase="$1"
    local phase_dir="$2"
    local target="$3"
    local snapshot="$4"
    local source_checkpoint="$5"
    local previous_audit="$6"
    local segment_lineage="$7"
    local audit
    audit="$(milestone_audit_path "${phase_dir}" "${target}")"
    local command=(
        "${PYTHON_BIN}" "${AUDITOR}" milestone
        --phase "${phase}"
        --checkpoint "${snapshot}"
        --preflight "${phase_dir}/probe_preflight.json"
        --expected-iteration "${target}"
        --source-checkpoint "${source_checkpoint}"
        --segment-lineage "${segment_lineage}"
        --output "${audit}"
    )
    if [[ -n "${previous_audit}" ]]; then
        command+=(--previous-audit "${previous_audit}")
    fi
    "${command[@]}"
}

run_phase() {
    local phase="$1"
    local initial_checkpoint="$2"
    local initial_audit="$3"
    local phase_dir="${OUTPUT_ROOT}/${phase}"
    local live="${phase_dir}/checkpoint_iter.pth"
    local previous_snapshot=""
    local previous_audit=""
    local previous_iteration=0
    local -a milestones=()
    mapfile -t milestones < <(phase_milestones "${phase}")

    run_preflight "${phase}" "${phase_dir}" "${initial_checkpoint}" "${initial_audit}"
    mkdir -p "${phase_dir}/milestones" "${phase_dir}/recovery" "${phase_dir}/launches"

    for target in "${milestones[@]}"; do
        local snapshot audit
        snapshot="$(milestone_path "${phase_dir}" "${target}")"
        audit="$(milestone_audit_path "${phase_dir}" "${target}")"
        if [[ -f "${snapshot}" || -f "${audit}" ]]; then
            if [[ ! -f "${snapshot}" || ! -f "${audit}" ]]; then
                echo "[FAIL] incomplete preserved ${phase} milestone ${target}" >&2
                exit 2
            fi
            local audited_source audited_lineage
            audited_source="$(json_source_path "${audit}")"
            audited_lineage="$(json_record_path "${audit}" segment_lineage)"
            local verify=(
                "${PYTHON_BIN}" "${AUDITOR}" milestone
                --phase "${phase}"
                --checkpoint "${snapshot}"
                --preflight "${phase_dir}/probe_preflight.json"
                --expected-iteration "${target}"
                --source-checkpoint "${audited_source}"
                --segment-lineage "${audited_lineage}"
                --output "${audit}"
                --verify-only
            )
            if [[ -n "${previous_audit}" ]]; then
                verify+=(--previous-audit "${previous_audit}")
            fi
            "${verify[@]}"
            previous_snapshot="${snapshot}"
            previous_audit="${audit}"
            previous_iteration="${target}"
            continue
        fi

        local initialization_mode source_checkpoint live_iteration recovery_inspection
        initialization_mode=""
        source_checkpoint=""
        live_iteration=0
        recovery_inspection=""
        if [[ -f "${live}" ]]; then
            live_iteration="$("${PYTHON_BIN}" "${AUDITOR}" metadata --checkpoint "${live}" --field iteration)"
        fi

        if [[ "${live_iteration}" -gt "${previous_iteration}" ]]; then
            if [[ "${CONTINUE_RUN}" != "1" ]]; then
                echo "[FAIL] found an unpreserved live ${phase} checkpoint; pass --continue after inspection" >&2
                exit 2
            fi
            local lineage_files=()
            mapfile -t lineage_files < <(
                find "${phase_dir}/launches" -maxdepth 1 -type f \
                    -name "target_$(printf '%06d' "${target}")_attempt_*.lineage.json" | sort
            )
            if [[ "${#lineage_files[@]}" -lt 1 ]]; then
                echo "[FAIL] live ${phase} checkpoint has no recorded current-segment ancestry" >&2
                exit 2
            fi
            local segment_lineage inspect_output live_sha
            segment_lineage="${lineage_files[$((${#lineage_files[@]} - 1))]}"
            live_sha="$(sha256sum "${live}" | awk '{print $1}')"
            inspect_output="${phase_dir}/recovery/live_target_${target}_iter_${live_iteration}_${live_sha:0:12}.audit.json"
            local inspect_command=(
                "${PYTHON_BIN}" "${AUDITOR}" inspect
                --phase "${phase}"
                --checkpoint "${live}"
                --preflight "${phase_dir}/probe_preflight.json"
                --expected-target "${target}"
                --segment-lineage "${segment_lineage}"
                --output "${inspect_output}"
            )
            if [[ -n "${previous_audit}" ]]; then
                inspect_command+=(--previous-audit "${previous_audit}")
            fi
            "${inspect_command[@]}"
            if [[ "${live_iteration}" -eq "${target}" ]]; then
                source_checkpoint="$(json_source_path "${segment_lineage}")"
                cp --reflink=auto --preserve=timestamps "${live}" "${snapshot}"
                audit_milestone \
                    "${phase}" "${phase_dir}" "${target}" "${snapshot}" \
                    "${source_checkpoint}" "${previous_audit}" "${segment_lineage}"
                previous_snapshot="${snapshot}"
                previous_audit="${audit}"
                previous_iteration="${target}"
                continue
            fi
            local recovery_checkpoint
            recovery_checkpoint="${phase_dir}/recovery/checkpoint_target_${target}_iter_${live_iteration}_${live_sha:0:12}.pth"
            if [[ -f "${recovery_checkpoint}" ]]; then
                if [[ "$(sha256sum "${recovery_checkpoint}" | awk '{print $1}')" != "${live_sha}" ]]; then
                    echo "[FAIL] recovery checkpoint name collision: ${recovery_checkpoint}" >&2
                    exit 2
                fi
            else
                cp --reflink=auto --preserve=timestamps "${live}" "${recovery_checkpoint}"
            fi
            initialization_mode="resume"
            source_checkpoint="${recovery_checkpoint}"
            recovery_inspection="${inspect_output}"
        elif [[ -n "${previous_snapshot}" ]]; then
            initialization_mode="resume"
            source_checkpoint="${previous_snapshot}"
        else
            initialization_mode="pretrain"
            source_checkpoint="${initial_checkpoint}"
        fi

        build_train_command \
            "${phase}" "${phase_dir}" "${target}" \
            "${initialization_mode}" "${source_checkpoint}"
        local attempt launch_path segment_log lineage_path lineage_command
        attempt=1
        while [[ -e "${phase_dir}/launches/target_$(printf '%06d' "${target}")_attempt_$(printf '%02d' "${attempt}").sh" ]]; do
            attempt=$((attempt + 1))
        done
        launch_path="${phase_dir}/launches/target_$(printf '%06d' "${target}")_attempt_$(printf '%02d' "${attempt}").sh"
        segment_log="${phase_dir}/launches/target_$(printf '%06d' "${target}")_attempt_$(printf '%02d' "${attempt}").log"
        lineage_path="${phase_dir}/launches/target_$(printf '%06d' "${target}")_attempt_$(printf '%02d' "${attempt}").lineage.json"
        lineage_command=(
            "${PYTHON_BIN}" "${AUDITOR}" segment-lineage
            --phase "${phase}"
            --preflight "${phase_dir}/probe_preflight.json"
            --expected-target "${target}"
            --source-checkpoint "${source_checkpoint}"
            --initialization-mode "${initialization_mode}"
            --output "${lineage_path}"
        )
        if [[ -n "${previous_audit}" ]]; then
            lineage_command+=(--previous-audit "${previous_audit}")
        fi
        if [[ -n "${recovery_inspection}" ]]; then
            lineage_command+=(--recovery-inspection "${recovery_inspection}")
        fi
        "${lineage_command[@]}"
        print_command "${TRAIN_COMMAND[@]}" > "${launch_path}"
        echo "[RUN] ${phase} target=${target} mode=${initialization_mode} source=${source_checkpoint}"
        print_command "${TRAIN_COMMAND[@]}"

        local default_visible_devices="0,1"
        if [[ "${WORLD_SIZE}" -eq 1 ]]; then
            default_visible_devices="0"
        fi
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${default_visible_devices}}"
        export DATA_ROOT="${DATA_ROOT:-/home/user/datasets/pivot_data}"
        export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${phase_dir}/train_console.log" "${segment_log}"

        if [[ ! -f "${live}" ]]; then
            echo "[FAIL] ${phase} target ${target} produced no checkpoint_iter.pth" >&2
            exit 2
        fi
        live_iteration="$("${PYTHON_BIN}" "${AUDITOR}" metadata --checkpoint "${live}" --field iteration)"
        if [[ "${live_iteration}" -ne "${target}" ]]; then
            echo "[INCOMPLETE] ${phase} stopped at iteration ${live_iteration}; rerun with --continue" >&2
            exit 3
        fi
        cp --reflink=auto --preserve=timestamps "${live}" "${snapshot}"
        audit_milestone \
            "${phase}" "${phase_dir}" "${target}" "${snapshot}" \
            "${source_checkpoint}" "${previous_audit}" "${lineage_path}"
        previous_snapshot="${snapshot}"
        previous_audit="${audit}"
        previous_iteration="${target}"
    done
    echo "[OK] completed ${phase} milestones: ${phase_dir}/milestones"
}

if [[ "${DRY_RUN}" == "1" ]]; then
    static_phase="${PHASE}"
    if [[ "${PHASE}" == "both" ]]; then
        static_phase="all"
    fi
    "${PYTHON_BIN}" "${AUDITOR}" static --phase "${static_phase}" >/dev/null
    echo "[OK] static adapter probe inputs passed"
    for requested_phase in rank confidence; do
        if [[ "${PHASE}" != "both" && "${PHASE}" != "${requested_phase}" ]]; then
            continue
        fi
        if [[ "${requested_phase}" == "rank" ]]; then
            dry_initial="${BASELINE_CHECKPOINT}"
            dry_audit=""
        else
            dry_initial="${RANK_CHECKPOINT:-${OUTPUT_ROOT}/rank/milestones/checkpoint_iter_$(printf '%06d' "${RANK_SELECTION}").pth}"
            dry_audit="${RANK_AUDIT}"
        fi
        if [[ -f "${dry_initial}" && ( "${requested_phase}" == "rank" || -f "${dry_audit}" ) ]]; then
            validate_initial=(
                "${PYTHON_BIN}" "${AUDITOR}" validate-initial
                --phase "${requested_phase}"
                --initial-checkpoint "${dry_initial}"
                --world-size "${WORLD_SIZE}"
                --per-gpu-batch "${PER_GPU_BATCH}"
            )
            if [[ -n "${dry_audit}" ]]; then
                validate_initial+=(--initial-audit "${dry_audit}")
            fi
            "${validate_initial[@]}"
        elif [[ "${requested_phase}" == "confidence" && -n "${RANK_CHECKPOINT}" ]]; then
            echo "[FAIL] dry-run cannot validate explicit confidence initializer/audit" >&2
            exit 2
        fi
        dry_phase_dir="${OUTPUT_ROOT}/${requested_phase}"
        dry_source="${dry_initial}"
        dry_mode="pretrain"
        mapfile -t dry_milestones < <(phase_milestones "${requested_phase}")
        for target in "${dry_milestones[@]}"; do
            build_train_command \
                "${requested_phase}" "${dry_phase_dir}" "${target}" \
                "${dry_mode}" "${dry_source}"
            print_command "${TRAIN_COMMAND[@]}"
            dry_source="$(milestone_path "${dry_phase_dir}" "${target}")"
            dry_mode="resume"
        done
    done
    exit 0
fi

BASELINE_CHECKPOINT="$(absolute_path "${BASELINE_CHECKPOINT}")"
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT}")"

if [[ "${PHASE}" == "rank" || "${PHASE}" == "both" ]]; then
    run_phase "rank" "${BASELINE_CHECKPOINT}" ""
fi

if [[ "${PHASE}" == "confidence" || "${PHASE}" == "both" ]]; then
    if [[ -z "${RANK_CHECKPOINT}" ]]; then
        RANK_CHECKPOINT="$(milestone_path "${OUTPUT_ROOT}/rank" "${RANK_SELECTION}")"
    fi
    if [[ -z "${RANK_AUDIT}" ]]; then
        RANK_AUDIT="$(milestone_audit_path "${OUTPUT_ROOT}/rank" "${RANK_SELECTION}")"
    fi
    RANK_CHECKPOINT="$(absolute_path "${RANK_CHECKPOINT}")"
    RANK_AUDIT="$(absolute_path "${RANK_AUDIT}")"
    run_phase "confidence" "${RANK_CHECKPOINT}" "${RANK_AUDIT}"
fi
