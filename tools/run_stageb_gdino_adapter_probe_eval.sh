#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/tools/stageb_python.sh"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    cat <<'EOF'
Usage: tools/run_stageb_gdino_adapter_probe_eval.sh \
  --checkpoint CANDIDATE.pth --label LABEL [options]

Options:
  --baseline-checkpoint PATH  Fixed pure Stage-B data-FT checkpoint
  --checkpoint-audit PATH     Required trusted milestone audit for non-P0 candidates
  --checkpoint-audit-kind K   auto|two-phase|semantic-confidence|fixed-top1-confidence
                              (default: auto)
  --output-root DIR           Evaluation output root
  --data-root DIR             Dataset root
  --device DEVICE             Evaluation device (default: cuda:0)
  --p0-parity                 Require exact same-record P0 parity, not improvement
  --parity-atol FLOAT         P0 score/IoU tolerance (default: 0, exact)
  --diagnostic                Keep reports but do not fail only because metric gate misses
  --allow-gate-fail           Alias for --diagnostic
  --dry-run                   Static checks and command printing; no GPU work
  --python PATH               Python executable (default: /usr/bin/python3)

Completed baseline/candidate evaluations are reused only when their preflight
checkpoint and config hashes still match. Partial evaluation directories fail.
EOF
}

CHECKPOINT=""
CHECKPOINT_AUDIT=""
CHECKPOINT_AUDIT_KIND="auto"
LABEL=""
BASELINE_CHECKPOINT="outputs/gdino_ft_stage_b_fixed_baseline_20260711/checkpoint0000.pth"
OUTPUT_ROOT="outputs/stageb_gdino_adapter_fixed_protocol_eval_20260711"
DATA_ROOT_VALUE="${DATA_ROOT:-/home/user/datasets/pivot_data}"
DEVICE="cuda:0"
P0_PARITY=0
PARITY_ATOL=0
DIAGNOSTIC=0
DRY_RUN=0
PYTHON_BIN="$(stageb_resolve_python "${PYTHON_BIN:-}")"
BASELINE_CONFIG="config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py"
P0_CONFIG="config/ablations/cfg_stageb_gdino_score_adapter_dataft.py"
CANDIDATE_CONFIG=""
CANDIDATE_REUSE_ALLOWED=1
FIXED_TOP1_AUDITOR="tools/stageb_gdino_fixed_top1_probe_audit.py"
FIXED_TOP1_AUDITOR_SHA256="b881a61004747c31f3cee03ac8d107a2506635affc6ad42f39ba57b7ee3f65d7"
SEMANTIC_AUDITOR="tools/stageb_gdino_semantic_probe_audit.py"
SEMANTIC_AUDITOR_SHA256="7a35c816ed61f088b426791afd99a3d7b9e26a60102d9f25e467f6907c69e87d"
EVAL_SUMMARY_AUDITOR="tools/verify_stageb_fixed_eval_summary_binding.py"
EVAL_SUMMARY_AUDITOR_SHA256="b59f8e4585cd30159fab07b9911c3b00e3c1a2678a7e53e31246cccd93bf8157"
FIXED_TOP1_SELECTOR="tools/stageb_gdino_fixed_top1_selection.py"
FIXED_TOP1_SELECTOR_SHA256="07d456e571fb8931c3fb62bf2a0003d918e43aceb194129cd60b5307c935f9d9"
FPR_COMPARATOR="tools/compare_stageb_fpr95_records.py"
FPR_COMPARATOR_SHA256="3796f96df57ce7ad97d433c9e610efe324cc501d0bea7a922f92749b55553449"
FIXED_TOP1_CALIBRATION_EVALUATOR="tools/eval_stageb_gdino_fixed_top1_calibration.py"
FIXED_TOP1_CALIBRATION_EVALUATOR_SHA256="61d9f82a952edadbaf3211d3986139e0e53724553b7dd45fdfdea1c6ffdedcd3"
FIXED_TOP1_CALIBRATION_LAUNCHER="tools/run_stageb_gdino_fixed_top1_calibration.sh"
FIXED_TOP1_CALIBRATION_LAUNCHER_SHA256="fe04c1c2e4b86f0784b014c64236a338f5c1d61decd19354b4f1f2198f6137c3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --checkpoint-audit) CHECKPOINT_AUDIT="$2"; shift 2 ;;
        --checkpoint-audit-kind) CHECKPOINT_AUDIT_KIND="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --baseline-checkpoint) BASELINE_CHECKPOINT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --data-root) DATA_ROOT_VALUE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --p0-parity) P0_PARITY=1; shift ;;
        --parity-atol) PARITY_ATOL="$2"; shift 2 ;;
        --diagnostic|--allow-gate-fail) DIAGNOSTIC=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[FAIL] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${CHECKPOINT}" || -z "${LABEL}" ]]; then
    usage >&2
    exit 2
fi
case "${CHECKPOINT_AUDIT_KIND}" in
    auto|two-phase|semantic-confidence|fixed-top1-confidence) ;;
    *) echo "[FAIL] invalid --checkpoint-audit-kind: ${CHECKPOINT_AUDIT_KIND}" >&2; exit 2 ;;
esac
if [[ "${P0_PARITY}" != "1" && "${DRY_RUN}" != "1" && -z "${CHECKPOINT_AUDIT}" ]]; then
    echo "[FAIL] formal non-P0 evaluation requires --checkpoint-audit" >&2
    exit 2
fi
if [[ "${P0_PARITY}" == "1" && -n "${CHECKPOINT_AUDIT}" ]]; then
    echo "[FAIL] P0 uses its adjacent P0 sidecar, not --checkpoint-audit" >&2
    exit 2
fi
if [[ ! "${LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "[FAIL] --label may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
fi
"${PYTHON_BIN}" -c \
    'import math,sys; x=float(sys.argv[1]); assert math.isfinite(x) and x >= 0' \
    "${PARITY_ATOL}" || {
        echo "[FAIL] --parity-atol must be finite and non-negative" >&2
        exit 2
    }

BASELINE_DIR="${OUTPUT_ROOT}/baseline"
CANDIDATE_DIR="${OUTPUT_ROOT}/${LABEL}"
COMPARISON_DIR="${OUTPUT_ROOT}/${LABEL}_vs_baseline"
LINEAGE_DIR="${OUTPUT_ROOT}/checkpoint_lineage"
LINEAGE_OUTPUT="${LINEAGE_DIR}/${LABEL}.verified.json"
LINEAGE_POST_OUTPUT="${LINEAGE_DIR}/${LABEL}.post_eval.verified.json"

build_eval_commands() {
    baseline_command=(
        tools/run_stageb_fixed_protocol_eval.sh
        --config "${BASELINE_CONFIG}"
        --checkpoint "${BASELINE_CHECKPOINT}"
        --output-dir "${BASELINE_DIR}"
        --data-root "${DATA_ROOT_VALUE}"
        --device "${DEVICE}"
        --python "${PYTHON_BIN}"
    )
    candidate_command=(
        tools/run_stageb_fixed_protocol_eval.sh
        --config "${CANDIDATE_CONFIG}"
        --checkpoint "${CHECKPOINT}"
        --output-dir "${CANDIDATE_DIR}"
        --data-root "${DATA_ROOT_VALUE}"
        --device "${DEVICE}"
        --python "${PYTHON_BIN}"
    )
}
PRIMARY_MANIFEST="data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/eval_manifest.jsonl"
SUPPLEMENTAL_MANIFEST="data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/semantic_stageb_union_image_disjoint_manifest.jsonl"

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

verify_fixed_top1_auditor() {
    local observed
    observed="$("${PYTHON_BIN}" -c \
        'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
        "${FIXED_TOP1_AUDITOR}")"
    if [[ "${observed}" != "${FIXED_TOP1_AUDITOR_SHA256}" ]]; then
        echo "[FAIL] fixed-top1 checkpoint auditor hash drifted" >&2
        exit 2
    fi
}

verify_semantic_auditor() {
    local observed
    observed="$("${PYTHON_BIN}" -c \
        'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
        "${SEMANTIC_AUDITOR}")"
    if [[ "${observed}" != "${SEMANTIC_AUDITOR_SHA256}" ]]; then
        echo "[FAIL] semantic checkpoint auditor hash drifted" >&2
        exit 2
    fi
}

verify_eval_summary_auditor() {
    local observed
    observed="$("${PYTHON_BIN}" -c \
        'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
        "${EVAL_SUMMARY_AUDITOR}")"
    if [[ "${observed}" != "${EVAL_SUMMARY_AUDITOR_SHA256}" ]]; then
        echo "[FAIL] fixed evaluation summary auditor hash drifted" >&2
        exit 2
    fi
}

verify_fixed_top1_acceptance_closure() {
    local paths=(
        "${FIXED_TOP1_SELECTOR}"
        "${FPR_COMPARATOR}"
        "${FIXED_TOP1_CALIBRATION_EVALUATOR}"
        "${FIXED_TOP1_CALIBRATION_LAUNCHER}"
    )
    local expected=(
        "${FIXED_TOP1_SELECTOR_SHA256}"
        "${FPR_COMPARATOR_SHA256}"
        "${FIXED_TOP1_CALIBRATION_EVALUATOR_SHA256}"
        "${FIXED_TOP1_CALIBRATION_LAUNCHER_SHA256}"
    )
    local index observed
    for index in "${!paths[@]}"; do
        observed="$("${PYTHON_BIN}" -c \
            'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
            "${paths[${index}]}")"
        if [[ "${observed}" != "${expected[${index}]}" ]]; then
            echo "[FAIL] fixed-top1 acceptance closure hash drifted: ${paths[${index}]}" >&2
            exit 2
        fi
    done
}

validate_reused_eval() {
    local eval_dir="$1"
    local checkpoint="$2"
    local config="$3"
    "${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py verify-eval \
        --output_dir "${eval_dir}" \
        --checkpoint "${checkpoint}" \
        --config "${config}"
}

validate_candidate_lineage() {
    local lineage_output="$1"
    mkdir -p "${LINEAGE_DIR}"
    if [[ "${P0_PARITY}" == "1" ]]; then
        "${PYTHON_BIN}" tools/make_stageb_gdino_adapter_p0.py verify \
            --baseline-checkpoint "${BASELINE_CHECKPOINT}" \
            --checkpoint "${CHECKPOINT}" \
            --output "${lineage_output}"
        CANDIDATE_CONFIG="${P0_CONFIG}"
        return
    fi
    if [[ ! -f "${CHECKPOINT_AUDIT}" ]]; then
        echo "[FAIL] candidate checkpoint audit is missing: ${CHECKPOINT_AUDIT}" >&2
        exit 2
    fi
    local schema selected_kind
    schema="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("schema", ""))' "${CHECKPOINT_AUDIT}")"
    selected_kind="${CHECKPOINT_AUDIT_KIND}"
    if [[ "${selected_kind}" == "auto" ]]; then
        case "${schema}" in
            stageb-gdino-adapter-two-phase-probe-v1) selected_kind="two-phase" ;;
            stageb-gdino-adapter-semantic-confidence-probe-v1) selected_kind="semantic-confidence" ;;
            stageb-gdino-adapter-fixed-top1-confidence-probe-v1) selected_kind="fixed-top1-confidence" ;;
            *) echo "[FAIL] unknown candidate checkpoint audit schema: ${schema}" >&2; exit 2 ;;
        esac
    fi
    if [[ "${selected_kind}" == "two-phase" ]]; then
        if [[ "${schema}" != "stageb-gdino-adapter-two-phase-probe-v1" ]]; then
            echo "[FAIL] checkpoint audit kind/schema mismatch" >&2
            exit 2
        fi
        "${PYTHON_BIN}" tools/stageb_gdino_adapter_probe_audit.py verify-evaluation \
            --checkpoint "${CHECKPOINT}" \
            --audit "${CHECKPOINT_AUDIT}" \
            --output "${lineage_output}"
    elif [[ "${selected_kind}" == "semantic-confidence" ]]; then
        if [[ "${schema}" != "stageb-gdino-adapter-semantic-confidence-probe-v1" ]]; then
            echo "[FAIL] checkpoint audit kind/schema mismatch" >&2
            exit 2
        fi
        "${PYTHON_BIN}" tools/stageb_gdino_semantic_probe_audit.py verify-evaluation \
            --checkpoint "${CHECKPOINT}" \
            --audit "${CHECKPOINT_AUDIT}" \
            --output "${lineage_output}"
    else
        if [[ "${schema}" != "stageb-gdino-adapter-fixed-top1-confidence-probe-v1" ]]; then
            echo "[FAIL] checkpoint audit kind/schema mismatch" >&2
            exit 2
        fi
        verify_fixed_top1_auditor
        CANDIDATE_REUSE_ALLOWED=0
        "${PYTHON_BIN}" tools/stageb_gdino_fixed_top1_probe_audit.py verify-evaluation \
            --checkpoint "${CHECKPOINT}" \
            --audit "${CHECKPOINT_AUDIT}" \
            --output "${lineage_output}"
    fi
    CANDIDATE_CONFIG="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["config"]["path"])' "${lineage_output}")"
    if [[ ! -f "${CANDIDATE_CONFIG}" ]]; then
        echo "[FAIL] verified checkpoint audit resolved a missing training config: ${CANDIDATE_CONFIG}" >&2
        exit 2
    fi
}

run_or_reuse_eval() {
    local eval_dir="$1"
    local checkpoint="$2"
    local config="$3"
    local reuse_allowed="$4"
    shift 4
    local command=("$@")
    if [[ -f "${eval_dir}/protocol_eval_complete.json" ]]; then
        if [[ "${reuse_allowed}" != "1" ]]; then
            echo "[FAIL] fixed-top1 candidate evaluation must be fresh after the pre-eval lineage seal" >&2
            exit 2
        fi
        validate_reused_eval "${eval_dir}" "${checkpoint}" "${config}"
        echo "[OK] reusing completed fixed-protocol evaluation: ${eval_dir}"
        return
    fi
    if [[ -d "${eval_dir}" && -n "$(find "${eval_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "[FAIL] partial evaluation directory is not reusable: ${eval_dir}" >&2
        exit 2
    fi
    "${command[@]}"
}

single_tn_record() {
    local directory="$1"
    local records=()
    mapfile -t records < <(find "${directory}" -maxdepth 1 -type f -name '*.records.jsonl' | sort)
    if [[ "${#records[@]}" -ne 1 ]]; then
        echo "[FAIL] expected exactly one TN record file in ${directory}, found ${#records[@]}" >&2
        exit 2
    fi
    printf '%s\n' "${records[0]}"
}

run_fpr_comparator() {
    local section="$1"
    local manifest="$2"
    local baseline_record candidate_record
    baseline_record="$(single_tn_record "${BASELINE_DIR}/${section}/per_example_records")"
    candidate_record="$(single_tn_record "${CANDIDATE_DIR}/${section}/per_example_records")"
    "${PYTHON_BIN}" tools/compare_stageb_fpr95_records.py \
        --baseline-records "${baseline_record}" \
        --candidate-records "${candidate_record}" \
        --manifest "${manifest}" \
        --output-json "${COMPARISON_DIR}/${section}_fpr95_comparison.json" \
        --output-markdown "${COMPARISON_DIR}/${section}_fpr95_comparison.md" \
        > "${COMPARISON_DIR}/${section}_fpr95_comparison.console.json"
}

write_nonacceptance_status() {
    local mode="$1"
    local reason="$2"
    local gate_status="${3:-}"
    "${PYTHON_BIN}" - \
        "${COMPARISON_DIR}/final_acceptance_status.json" \
        "${mode}" "${reason}" "${gate_status}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "stageb-gdino-adapter-final-acceptance-v1",
    "kind": "final_acceptance_status",
    "mode": sys.argv[2],
    "reason": sys.argv[3],
    "final_acceptance_claimed": False,
    "all_required_evidence_verified": False,
}
if sys.argv[4]:
    payload["dual_gate_exit_status"] = int(sys.argv[4])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

write_final_acceptance_status() {
    "${PYTHON_BIN}" - \
        "${COMPARISON_DIR}/final_acceptance_status.json" \
        "${COMPARISON_DIR}/baseline_summary_binding.json" \
        "${COMPARISON_DIR}/candidate_summary_binding.json" \
        "${COMPARISON_DIR}/lineage_replay_equality.json" \
        "${COMPARISON_DIR}/paired_protocol_audit.json" \
        "${COMPARISON_DIR}/paired_record_identity.json" \
        "${COMPARISON_DIR}/strict2031_fpr95_comparison.json" \
        "${COMPARISON_DIR}/strict1607_fpr95_comparison.json" \
        "${COMPARISON_DIR}/primary_strict2031.json" \
        "${COMPARISON_DIR}/supplemental_strict1607.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from tools.verify_stageb_fixed_eval_summary_binding import (
    SummaryBindingError,
    validate_final_metric_input_binding,
)

output = Path(sys.argv[1])
evidence_paths = [Path(value) for value in sys.argv[2:]]

def read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"final acceptance evidence is not an object: {path}")
    return value

def record(path):
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

values = [read(path) for path in evidence_paths]
baseline_binding, candidate_binding, lineage_equality, paired, identity, fpr2031, fpr1607, gate2031, gate1607 = values
try:
    metric_input_binding = validate_final_metric_input_binding(
        baseline_binding=baseline_binding,
        candidate_binding=candidate_binding,
        fpr_reports={"strict2031": fpr2031, "strict1607": fpr1607},
        dual_gate_reports={"strict2031": gate2031, "strict1607": gate1607},
    )
except SummaryBindingError as error:
    raise SystemExit(f"final metric input binding failed: {error}") from error
expected_ref_splits = [
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_val",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_val",
    "refcocog_test",
]
for label, binding in (("baseline", baseline_binding), ("candidate", candidate_binding)):
    if binding.get("pass") is not True:
        raise SystemExit(f"{label} summary binding did not pass")
    official = binding.get("official_ref_contract", {})
    if official.get("all_eight_exact_rows_and_manifest_sha256") is not True:
        raise SystemExit(f"{label} official Ref8 contract is not sealed")
lineage = candidate_binding.get("lineage_binding")
if not isinstance(lineage, dict) or lineage.get("pass") is not True:
    raise SystemExit("candidate checkpoint/lineage/baseline binding did not pass")
if (
    lineage_equality.get("schema") != "stageb-lineage-pre-post-equality-v1"
    or lineage_equality.get("kind") != "completed_lineage_pre_post_equality"
    or lineage_equality.get("pass") is not True
    or lineage_equality.get("same_bytes") is not True
    or lineage_equality.get("same_json") is not True
    or lineage_equality.get("pre_eval_lineage")
    != lineage.get("trusted_lineage_output")
):
    raise SystemExit("candidate summary is not bound to an unchanged pre/post lineage seal")
if lineage.get("schema") == "stageb-gdino-adapter-fixed-top1-confidence-probe-v1":
    selection = lineage.get("selection_binding")
    if (
        not isinstance(selection, dict)
        or selection.get("pass") is not True
        or selection.get("input_scope") != "calibration_only"
        or selection.get("strict_paths_consumed_for_scoring") is not False
        or selection.get("selected_checkpoint") != lineage.get("candidate_checkpoint")
    ):
        raise SystemExit("fixed-top1 final evidence is not bound to held-out selection")
if baseline_binding.get("lineage_binding") is not None:
    raise SystemExit("baseline summary unexpectedly carries candidate lineage")
if lineage.get("root_authoritative_baseline_checkpoint") != baseline_binding.get("checkpoint"):
    raise SystemExit("candidate lineage root differs from the evaluated baseline checkpoint")
if paired.get("kind") != "fixed_stageb_paired_eval_protocol":
    raise SystemExit("paired protocol audit is invalid")
if identity.get("pass") is not True:
    raise SystemExit("paired record identity audit did not pass")
for label, fpr in (("strict2031", fpr2031), ("strict1607", fpr1607)):
    if fpr.get("validation", {}).get("pass") is not True:
        raise SystemExit(f"{label} FPR record validation did not pass")
    if not float(fpr.get("global", {}).get("candidate_minus_baseline_fpr95", 0.0)) < 0.0:
        raise SystemExit(f"{label} FPR95 is not strictly lower")
for label, gate in (("strict2031", gate2031), ("strict1607", gate1607)):
    if gate.get("validation", {}).get("pass") is not True or gate.get("gate", {}).get("pass") is not True:
        raise SystemExit(f"{label} dual gate did not pass")
    if gate.get("gate", {}).get("global_fpr95_lower") is not True:
        raise SystemExit(f"{label} global FPR95 gate is not strict")
    if gate.get("gate", {}).get("every_required_ref_split_acc50_higher") is not True:
        raise SystemExit(f"{label} Ref8 gate is not strict")
    if gate.get("required_ref_splits") != expected_ref_splits:
        raise SystemExit(f"{label} does not contain the exact official Ref8 split contract")
    ref_rows = gate.get("refcoco")
    if not isinstance(ref_rows, dict) or set(ref_rows) != set(expected_ref_splits):
        raise SystemExit(f"{label} does not report exactly all official Ref8 splits")
    if any(
        row.get("improved") is not True
        or not float(row.get("candidate_minus_baseline_acc50", 0.0)) > 0.0
        for row in ref_rows.values()
    ):
        raise SystemExit(f"{label} has a Ref split without strict acc50 improvement")

payload = {
    "schema": "stageb-gdino-adapter-final-acceptance-v1",
    "kind": "final_acceptance_status",
    "mode": "strict_final_acceptance",
    "reason": "all checkpoint, record, Ref8, strict2031, and strict1607 gates passed",
    "final_acceptance_claimed": True,
    "all_required_evidence_verified": True,
    "candidate_checkpoint": candidate_binding["checkpoint"],
    "root_authoritative_baseline_checkpoint": lineage[
        "root_authoritative_baseline_checkpoint"
    ],
    "required_ref_splits": expected_ref_splits,
    "metric_input_binding": metric_input_binding,
    "evidence": {path.stem: record(path) for path in evidence_paths},
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
}

verify_fixed_top1_auditor
verify_fixed_top1_acceptance_closure
verify_eval_summary_auditor
verify_semantic_auditor

if [[ "${DRY_RUN}" == "1" ]]; then
    "${PYTHON_BIN}" tools/stageb_gdino_adapter_probe_audit.py static \
        --phase all >/dev/null
    "${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py static \
        --data_root "${DATA_ROOT_VALUE}" >/dev/null
    if [[ "${P0_PARITY}" == "1" ]]; then
        CANDIDATE_CONFIG="${P0_CONFIG}"
    elif [[ -f "${CHECKPOINT_AUDIT}" ]]; then
        CANDIDATE_CONFIG="$("${PYTHON_BIN}" - "${CHECKPOINT_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

audit = json.load(open(sys.argv[1], encoding="utf-8"))
if audit.get("schema") == "stageb-gdino-adapter-two-phase-probe-v1":
    preflight = json.load(open(audit["preflight"]["path"], encoding="utf-8"))
    print(preflight["static"]["config"]["path"])
elif audit.get("schema") == "stageb-gdino-adapter-semantic-confidence-probe-v1":
    print(audit["config"]["path"])
elif audit.get("schema") == "stageb-gdino-adapter-fixed-top1-confidence-probe-v1":
    print(audit["config"]["path"])
else:
    raise SystemExit("unknown checkpoint audit schema")
PY
)"
    else
        CANDIDATE_CONFIG="<FROM_TRUSTED_CHECKPOINT_AUDIT>"
    fi
    build_eval_commands
    echo "[OK] data/config/code and fixed-protocol static audits passed; dry-run does not deep-verify candidate checkpoint lineage"
    print_command "${baseline_command[@]}"
    print_command "${candidate_command[@]}"
    print_command \
        "${PYTHON_BIN}" "${EVAL_SUMMARY_AUDITOR}" \
        --eval-dir "${BASELINE_DIR}" \
        --expected-checkpoint "${BASELINE_CHECKPOINT}" \
        --output "${COMPARISON_DIR}/baseline_summary_binding.json"
    print_command \
        "${PYTHON_BIN}" "${EVAL_SUMMARY_AUDITOR}" \
        --eval-dir "${CANDIDATE_DIR}" \
        --expected-checkpoint "${CHECKPOINT}" \
        --trusted-lineage "${LINEAGE_OUTPUT}" \
        --expected-baseline-checkpoint "${BASELINE_CHECKPOINT}" \
        --output "${COMPARISON_DIR}/candidate_summary_binding.json"
    if [[ "${P0_PARITY}" == "1" ]]; then
        print_command \
            "${PYTHON_BIN}" tools/make_stageb_gdino_adapter_p0.py verify \
            --baseline-checkpoint "${BASELINE_CHECKPOINT}" \
            --checkpoint "${CHECKPOINT}" \
            --output "${LINEAGE_OUTPUT}"
        print_command \
            "${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py compare-evals \
            --baseline_dir "${BASELINE_DIR}" \
            --candidate_dir "${CANDIDATE_DIR}" \
            --output "${COMPARISON_DIR}/paired_protocol_audit.json"
        print_command \
            "${PYTHON_BIN}" tools/verify_stageb_p0_record_parity.py \
            --baseline-eval-dir "${BASELINE_DIR}" \
            --p0-eval-dir "${CANDIDATE_DIR}" \
            --atol "${PARITY_ATOL}" \
            --output "${COMPARISON_DIR}/p0_record_parity.json"
    else
        echo "[DRY RUN] formal execution requires a trusted --checkpoint-audit; auto dispatches its schema"
        print_command \
            "${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py compare-evals \
            --baseline_dir "${BASELINE_DIR}" \
            --candidate_dir "${CANDIDATE_DIR}" \
            --output "${COMPARISON_DIR}/paired_protocol_audit.json"
        print_command \
            "${PYTHON_BIN}" tools/verify_stageb_p0_record_parity.py \
            --baseline-eval-dir "${BASELINE_DIR}" \
            --candidate-eval-dir "${CANDIDATE_DIR}" \
            --identity-only \
            --output "${COMPARISON_DIR}/paired_record_identity.json"
        print_command \
            "${PYTHON_BIN}" tools/compare_stageb_fpr95_records.py \
            --baseline-records "${BASELINE_DIR}/strict2031/per_example_records/ONE.records.jsonl" \
            --candidate-records "${CANDIDATE_DIR}/strict2031/per_example_records/ONE.records.jsonl" \
            --manifest "${PRIMARY_MANIFEST}" \
            --output-json "${COMPARISON_DIR}/strict2031_fpr95_comparison.json" \
            --output-markdown "${COMPARISON_DIR}/strict2031_fpr95_comparison.md"
        print_command \
            "${PYTHON_BIN}" tools/compare_stageb_fpr95_records.py \
            --baseline-records "${BASELINE_DIR}/strict1607/per_example_records/ONE.records.jsonl" \
            --candidate-records "${CANDIDATE_DIR}/strict1607/per_example_records/ONE.records.jsonl" \
            --manifest "${SUPPLEMENTAL_MANIFEST}" \
            --output-json "${COMPARISON_DIR}/strict1607_fpr95_comparison.json" \
            --output-markdown "${COMPARISON_DIR}/strict1607_fpr95_comparison.md"
        print_command \
            tools/run_stageb_fixed_dual_gate.sh \
            "${BASELINE_DIR}" "${CANDIDATE_DIR}" "${COMPARISON_DIR}"
    fi
    exit 0
fi

if [[ ! -f "${BASELINE_CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "[FAIL] baseline or candidate checkpoint is missing" >&2
    exit 2
fi

if [[ -e "${LINEAGE_OUTPUT}" || -e "${LINEAGE_POST_OUTPUT}" ]]; then
    echo "[FAIL] lineage seal already exists; refusing to overwrite" >&2
    exit 2
fi
validate_candidate_lineage "${LINEAGE_OUTPUT}"
build_eval_commands

run_or_reuse_eval \
    "${BASELINE_DIR}" "${BASELINE_CHECKPOINT}" "${BASELINE_CONFIG}" "1" \
    "${baseline_command[@]}"
run_or_reuse_eval \
    "${CANDIDATE_DIR}" "${CHECKPOINT}" "${CANDIDATE_CONFIG}" "${CANDIDATE_REUSE_ALLOWED}" \
    "${candidate_command[@]}"

if [[ -e "${COMPARISON_DIR}" ]]; then
    echo "[FAIL] comparison output already exists: ${COMPARISON_DIR}" >&2
    exit 2
fi
mkdir -p "${COMPARISON_DIR}"
write_nonacceptance_status "pending" "final acceptance has not completed"

"${PYTHON_BIN}" "${EVAL_SUMMARY_AUDITOR}" \
    --eval-dir "${BASELINE_DIR}" \
    --expected-checkpoint "${BASELINE_CHECKPOINT}" \
    --output "${COMPARISON_DIR}/baseline_summary_binding.json"
# Close the checkpoint-lineage TOCTOU window without overwriting the pre-eval seal.
validate_candidate_lineage "${LINEAGE_POST_OUTPUT}"
"${PYTHON_BIN}" - \
    "${LINEAGE_OUTPUT}" \
    "${LINEAGE_POST_OUTPUT}" \
    "${COMPARISON_DIR}/lineage_replay_equality.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

pre_path, post_path, output = map(Path, sys.argv[1:])
pre_raw = pre_path.read_bytes()
post_raw = post_path.read_bytes()
try:
    pre_json = json.loads(pre_raw.decode("utf-8"))
    post_json = json.loads(post_raw.decode("utf-8"))
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"lineage replay is not valid UTF-8 JSON: {error}") from error
if pre_raw != post_raw or pre_json != post_json:
    raise SystemExit("pre/post evaluation lineage replay differs")

def record(path, raw):
    return {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

payload = {
    "schema": "stageb-lineage-pre-post-equality-v1",
    "kind": "completed_lineage_pre_post_equality",
    "pre_eval_lineage": record(pre_path, pre_raw),
    "post_eval_lineage": record(post_path, post_raw),
    "same_bytes": True,
    "same_json": True,
    "pass": True,
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
"${PYTHON_BIN}" "${EVAL_SUMMARY_AUDITOR}" \
    --eval-dir "${CANDIDATE_DIR}" \
    --expected-checkpoint "${CHECKPOINT}" \
    --trusted-lineage "${LINEAGE_OUTPUT}" \
    --expected-baseline-checkpoint "${BASELINE_CHECKPOINT}" \
    --output "${COMPARISON_DIR}/candidate_summary_binding.json"

if [[ "${P0_PARITY}" == "1" ]]; then
    "${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py compare-evals \
        --baseline_dir "${BASELINE_DIR}" \
        --candidate_dir "${CANDIDATE_DIR}" \
        --output "${COMPARISON_DIR}/paired_protocol_audit.json"
    "${PYTHON_BIN}" tools/verify_stageb_p0_record_parity.py \
        --baseline-eval-dir "${BASELINE_DIR}" \
        --p0-eval-dir "${CANDIDATE_DIR}" \
        --atol "${PARITY_ATOL}" \
        --output "${COMPARISON_DIR}/p0_record_parity.json"
    write_nonacceptance_status \
        "p0_control" \
        "P0 parity is a control and can never claim final metric acceptance"
    echo "[OK] P0 parity passed with strict2031 valid=2031/2031 and identity alignment"
else
    "${PYTHON_BIN}" tools/stageb_fixed_protocol_audit.py compare-evals \
        --baseline_dir "${BASELINE_DIR}" \
        --candidate_dir "${CANDIDATE_DIR}" \
        --output "${COMPARISON_DIR}/paired_protocol_audit.json"
    "${PYTHON_BIN}" tools/verify_stageb_p0_record_parity.py \
        --baseline-eval-dir "${BASELINE_DIR}" \
        --candidate-eval-dir "${CANDIDATE_DIR}" \
        --identity-only \
        --output "${COMPARISON_DIR}/paired_record_identity.json"
    run_fpr_comparator strict2031 "${PRIMARY_MANIFEST}"
    run_fpr_comparator strict1607 "${SUPPLEMENTAL_MANIFEST}"

    set +e
    tools/run_stageb_fixed_dual_gate.sh \
        "${BASELINE_DIR}" "${CANDIDATE_DIR}" "${COMPARISON_DIR}"
    gate_status=$?
    set -e
    if [[ "${gate_status}" -ne 0 ]]; then
        # Diagnostic mode may relax only the metric decision. Missing/malformed
        # paired records remain a protocol failure.
        "${PYTHON_BIN}" - \
            "${COMPARISON_DIR}/primary_strict2031.json" \
            "${COMPARISON_DIR}/supplemental_strict1607.json" <<'PY'
import json
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"missing dual-gate report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("validation", {}).get("pass") is not True:
        raise SystemExit(f"paired-record protocol validation failed: {path}")
PY
        if [[ "${DIAGNOSTIC}" != "1" ]]; then
            write_nonacceptance_status \
                "strict_final_acceptance" \
                "one or more strict metric gates failed" \
                "${gate_status}"
            echo "[FAIL] final metric gate failed; reports preserved in ${COMPARISON_DIR}" >&2
            exit "${gate_status}"
        fi
        write_nonacceptance_status \
            "diagnostic" \
            "diagnostic mode cannot claim final acceptance and a metric gate failed" \
            "${gate_status}"
        "${PYTHON_BIN}" - "${COMPARISON_DIR}/diagnostic_status.json" "${gate_status}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "stageb-gdino-adapter-diagnostic-gate-v1",
    "mode": "diagnostic_allow_metric_gate_fail",
    "dual_gate_exit_status": int(sys.argv[2]),
    "protocol_validation_pass": True,
    "final_acceptance_claimed": False,
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
        echo "[DIAGNOSTIC] metric gate did not pass; protocol and reports are valid: ${COMPARISON_DIR}"
    elif [[ "${DIAGNOSTIC}" == "1" ]]; then
        write_nonacceptance_status \
            "diagnostic" \
            "diagnostic mode cannot claim final acceptance even when metrics pass" \
            "0"
        echo "[DIAGNOSTIC] metrics passed, but diagnostic mode does not claim final acceptance: ${COMPARISON_DIR}"
    else
        write_final_acceptance_status
        echo "[OK] strict final acceptance evidence sealed: ${COMPARISON_DIR}/final_acceptance_status.json"
    fi
fi
