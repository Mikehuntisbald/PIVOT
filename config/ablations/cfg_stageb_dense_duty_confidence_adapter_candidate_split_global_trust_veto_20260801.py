from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_boundary_routing_20260801 import *  # noqa: F401,F403

# V49 is a fresh V47 structural ablation. It preserves V47's all-TN mean,
# boundary routing, positive-tail trust, data, and update budget while splitting
# the global absolute path into independently routed trust and veto heads.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_split_global_trust_veto_v49"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_global_trust_veto_v4"
)
# V49 restores V47's all-TN objective and seals the default in the v31 contract.
stage_b_v15_tail_queue_negative_reduction_contract = "all_mean_v1"
stage_b_v11_trainable_params_min = 669_322
stage_b_v11_trainable_params_max = 669_322

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_split_global_trust_veto_"
    "trace_audit_20260801/receipt.json"
)
# Seals 13,890 exact direct-trace rows and the V49 code source closure.
stage_b_dense_duty_trace_audit_sha256 = (
    "31b7e41ff56c16d5eaee3bbc490c25803a0e2744c68ff8566dc838af6aa9ab7c"
)

# The controller can verify an exact future main.py admission branch, but the
# probe config below disables promotion until strict1607 evidence exists.
stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_split_global_trust_veto_confidence_strict1607_v49"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_global_trust_veto_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
