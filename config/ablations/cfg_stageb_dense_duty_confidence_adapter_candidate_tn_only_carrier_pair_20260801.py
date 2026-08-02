from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_gate_zero_offset_20260801 import *  # noqa: F401,F403

# v42 keeps v39's model, deployed score, token supervision, and loss values.
# The carrier-pair hinge treats the positive carrier as a detached anchor so
# its backward path spends the paired gradient only on opening the TN veto.
stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract = (
    "tn_only_positive_detached_v2"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_tn_only_carrier_pair_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "9a3cdcb548966c8c714e66d3423c09d07fa0b1d5df1f6c02981542288c4baa4f"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_tn_only_carrier_pair_confidence_strict1607_v42"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_tn_only_carrier_pair_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
