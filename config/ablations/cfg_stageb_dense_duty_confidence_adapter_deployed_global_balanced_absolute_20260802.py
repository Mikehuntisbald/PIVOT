from config.ablations.cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_20260802 import *  # noqa: F401,F403

# V57 preserves V56's deployment-owned representation and adds the missing
# balanced absolute objective directly on the two deployed sample-global
# logits. Candidate-local supervision stays disabled and its serialized head
# remains a frozen diagnostic.
stage_b_dense_duty_confidence_revision = (
    "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57"
)
stage_b_dense_duty_deployed_global_absolute_weight = 1.0
stage_b_dense_duty_deployed_global_absolute_gamma = 1.0

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_deployed_global_balanced_absolute_"
    "trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "8fa36d8f32d509cd36d944fe128675e6d401f02f99d1d8686668f1e5632bf84c"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_rank_full_expression_deployed_global_balanced_absolute_"
    "confidence_strict1607_v57"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_deployed_global_balanced_absolute_highmem_20260802/"
    "probe_evaluation/u000400_strict1607_report.json"
)
