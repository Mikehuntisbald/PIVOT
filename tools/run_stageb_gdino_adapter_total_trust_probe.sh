#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export STAGEB_ADAPTER_AUDITOR="tools/stageb_gdino_adapter_total_trust_probe_audit.py"
export STAGEB_CONFIDENCE_CONFIG="${STAGEB_CONFIDENCE_CONFIG:-config/ablations/cfg_stageb_gdino_score_adapter_dataft_total_trust.py}"
export STAGEB_ADAPTER_WORLD_SIZE=1
export STAGEB_ADAPTER_PER_GPU_BATCH=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA_ROOT="${DATA_ROOT:-/media/haoyi/T9/data}"

DEFAULT_RANK_CHECKPOINT="outputs/paper_cvpr_v1/legacy_replay/rank_r100_seed42_b32_v3/milestones/checkpoint_iter_000100.pth"
DEFAULT_RANK_RECEIPT="outputs/paper_cvpr_v1/legacy_replay/legacy_r100_p50_exact_replay_receipt.json"
DEFAULT_OUTPUT_ROOT="outputs/stageb_gdino_adapter_total_trust_from_legacy_b58_r100_20260813"

effective_phase="confidence"
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    if [[ "${arguments[${index}]}" == "--phase" ]]; then
        if (( index + 1 >= ${#arguments[@]} )); then
            echo "[FAIL] --phase requires a value" >&2
            exit 2
        fi
        effective_phase="${arguments[$((index + 1))]}"
    fi
done
if [[ "${effective_phase}" != "confidence" ]]; then
    echo "[FAIL] total-trust profile is confidence-only; use the generic two-phase launcher for rank training" >&2
    exit 2
fi

# Defaults are prepended so explicit user paths later in argv still override
# them in the generic parser.
exec tools/run_stageb_gdino_adapter_two_phase_probe.sh \
    --phase confidence \
    --rank-checkpoint "${DEFAULT_RANK_CHECKPOINT}" \
    --rank-audit "${DEFAULT_RANK_RECEIPT}" \
    --output-root "${DEFAULT_OUTPUT_ROOT}" \
    "$@"
