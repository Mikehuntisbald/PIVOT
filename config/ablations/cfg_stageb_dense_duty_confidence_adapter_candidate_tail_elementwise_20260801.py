from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_asymmetric_20260731 import *  # noqa: F401,F403

# v38 keeps the stable v32 scorer and bounds each positive gate before taking
# the mean carrier.  This prevents high-logit outliers from restoring a broad
# batch gradient while preserving the historical q05 forward value.
stage_b_v15_tail_queue_positive_gradient_contract = (
    "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_tail_elementwise_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "8de545b20756f512d5f68a7c278954d5699498431bba68d965348fd6e3e5ecc8"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_tail_elementwise_confidence_strict1607_v38"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_tail_elementwise_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
