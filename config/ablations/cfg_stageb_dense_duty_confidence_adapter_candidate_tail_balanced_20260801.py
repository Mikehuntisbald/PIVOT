from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_asymmetric_20260731 import *  # noqa: F401,F403

# v35 keeps v32's scorer, data, loss weights, and update contract.  The FPR95
# threshold gradient retains the mean carrier that preserved absolute logit
# scale, while adding an exact lower-tail path for the positive operating tail.
stage_b_v15_tail_queue_positive_gradient_contract = (
    "mean_plus_exact_lower_tail_st_v3"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_tail_balanced_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "9b1b750928f92c798e453d25c909b6417f8ab012be79b3e6054cfd0da9c2be26"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_tail_balanced_confidence_strict1607_v35"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_tail_balanced_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
