from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_gate_zero_offset_20260801 import *  # noqa: F401,F403

# v40 keeps the deployed v39 scorer unchanged. For verified direct-trace TN
# rows only, changed-token BCE also supervises the detached candidate whose
# confidence base logit supplies the deployed global maximum.
stage_b_v21_token_edit_query_scope = (
    "target_iou_union_detached_final_confidence_base_argmax_v2"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_hardest_edit_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "9c1b3e3ee568a829ce78324b8b32bfef7c31a97860abe530c8232e53325b120c"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_hardest_edit_confidence_strict1607_v40"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_hardest_edit_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
