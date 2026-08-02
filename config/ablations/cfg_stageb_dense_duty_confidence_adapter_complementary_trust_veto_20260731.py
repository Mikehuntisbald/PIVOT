from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_token_conditioned_20260731 import *  # noqa: F401,F403

# v24 retains v23's token-conditioned global evidence, but replaces the
# monotone-only readout with two mutually gated one-sided depths:
# mismatch evidence may only lower confidence, while complementary token trust
# may only restore the positive low tail. U0 remains exactly rank-identical.
stage_b_dense_duty_confidence_revision = (
    "word_veto_complementary_trust_veto_v24"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "continuous_sigmoid_complementary_trust_veto_v5"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_complementary_trust_veto_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "3d80cb1220a4a6cf290590eae62f0ac6ef291bedbe5f74a7206d5ed3cc816e66"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_complementary_trust_veto_strict1607_v24"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_complementary_trust_veto_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
