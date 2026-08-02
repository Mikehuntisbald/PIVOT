#!/usr/bin/env bash

stageb_python_has_runtime() {
    local candidate="$1"
    [[ -x "${candidate}" ]] || return 1
    "${candidate}" -c 'import addict, numpy, torch' >/dev/null 2>&1
}

stageb_resolve_python() {
    local explicit="${1:-}"
    if [[ -n "${explicit}" ]]; then
        if ! stageb_python_has_runtime "${explicit}"; then
            echo "[FAIL] Python runtime lacks torch/numpy/addict: ${explicit}" >&2
            return 1
        fi
        printf '%s\n' "${explicit}"
        return 0
    fi

    local parent_python=""
    if [[ -r "/proc/${PPID}/exe" ]]; then
        parent_python="$(readlink -f "/proc/${PPID}/exe" || true)"
    fi
    local candidates=(
        "${PIVOT_PYTHON_BIN:-}"
        "${parent_python}"
        "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
        "${HOME}/miniconda/envs/gdino5090/bin/python"
        "/usr/bin/python3"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        [[ -n "${candidate}" ]] || continue
        if stageb_python_has_runtime "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    echo "[FAIL] no Python runtime with torch/numpy/addict was found; set PYTHON_BIN" >&2
    return 1
}
