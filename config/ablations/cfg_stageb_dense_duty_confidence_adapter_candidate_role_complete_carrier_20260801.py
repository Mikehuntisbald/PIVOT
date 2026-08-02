from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_hardest_edit_20260801 import *  # noqa: F401,F403

# v41 keeps the v40 model, scorer, and inference surface unchanged. On verified
# direct-trace rows, the detached positive/TN deployed carriers receive the
# complete token roles: positive=1, TN shared=1, and TN changed=0.
stage_b_v21_token_edit_query_scope = (
    "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_role_complete_carrier_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "6f4d1cdfbd43d0dd3b1256e609f5399b221c28b20c44ef1dc1f90efaf10554cf"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_role_complete_carrier_confidence_strict1607_v41"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_role_complete_carrier_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
