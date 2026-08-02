#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 BASELINE_EVAL_DIR CANDIDATE_EVAL_DIR OUTPUT_DIR" >&2
    exit 2
fi

BASELINE_DIR="$1"
CANDIDATE_DIR="$2"
OUTPUT_DIR="$3"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

for directory in "${BASELINE_DIR}" "${CANDIDATE_DIR}"; do
    if [[ ! -f "${directory}/protocol_eval_complete.json" ]]; then
        echo "[FAIL] missing completed fixed-protocol audit: ${directory}/protocol_eval_complete.json" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py compare-evals \
    --baseline_dir "${BASELINE_DIR}" \
    --candidate_dir "${CANDIDATE_DIR}" \
    --output "${OUTPUT_DIR}/paired_protocol_audit.json"

run_gate() {
    local tn_dir="$1"
    local output="$2"
    "${PYTHON_BIN}" tools/verify_stageb_dual_gate.py \
        --baseline_records \
            "${BASELINE_DIR}/ref8/per_example_records" \
            "${BASELINE_DIR}/${tn_dir}/per_example_records" \
        --candidate_records \
            "${CANDIDATE_DIR}/ref8/per_example_records" \
            "${CANDIDATE_DIR}/${tn_dir}/per_example_records" \
        --output "${output}"
}

set +e
run_gate strict2031 "${OUTPUT_DIR}/primary_strict2031.json"
primary_status=$?
run_gate strict1607 "${OUTPUT_DIR}/supplemental_strict1607.json"
supplemental_status=$?
set -e

if [[ ${primary_status} -ne 0 || ${supplemental_status} -ne 0 ]]; then
    echo "[FAIL] fixed dual gate failed: primary=${primary_status}, supplemental=${supplemental_status}" >&2
    exit 1
fi

echo "[OK] fixed dual gate passed on strict2031 and strict1607"
