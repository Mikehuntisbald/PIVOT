from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_asymmetric_20260731 import *  # noqa: F401,F403

# v34 keeps the stable v32 scorer surface and every loss weight/data/update
# contract.  Only the FPR95 positive-threshold straight-through carrier changes:
# it follows the current batch's exact lower-tail order statistic instead of
# its mean, so difficult positives cannot be sacrificed while suppressing TNs.
stage_b_v15_tail_queue_positive_gradient_contract = (
    "exact_batch_lower_tail_st_v2"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_q05_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "a084497abb7de5149f028875ec385bec6975bd845d15f411aae774a6d1801733"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_q05_confidence_strict1607_v34"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_q05_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
