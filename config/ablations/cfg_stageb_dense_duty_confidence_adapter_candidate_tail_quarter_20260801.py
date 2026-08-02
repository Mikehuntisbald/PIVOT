from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_asymmetric_20260731 import *  # noqa: F401,F403

# v36 keeps the stable v32 scorer and uses a quarter-strength exact positive
# lower-tail straight-through path alongside the mean carrier.  The reduced
# coefficient is intentional: v35's unscaled sum destabilized the global
# confidence head after the first epoch.
stage_b_v15_tail_queue_positive_gradient_contract = (
    "mean_plus_quarter_exact_lower_tail_st_v4"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_tail_quarter_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "641d8dccd7f0bb7a9d2f91594bdb400c2cf8d37a8f5cb813c6dc69626bc8d2ba"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_tail_quarter_confidence_strict1607_v36"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_tail_quarter_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
