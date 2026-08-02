from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_20260731 import *  # noqa: F401,F403

# v19 preserves v18's detached word-veto, absolute pool, training data, and
# tail-paired objective. It adds a zero-initialized verifier over detached
# query/text rank channels, evaluated only for inference-available modifier
# tokens. No edit label enters the forward path.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_tail_paired_rank_channel_v19"
)
stage_b_dense_duty_confidence_rank_evidence_contract = (
    "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
)
stage_b_v11_trainable_params_min = 203_142
stage_b_v11_trainable_params_max = 203_142

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_"
    "rank_channel_trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "af556ae7b695935251e4b99b7e9db515d69dba4a8d0339f9715d5cbb66710b66"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_tail_paired_rank_channel_strict1607_v19"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_tail_paired_rank_channel_"
    "highmem_20260731/probe_evaluation/u000300_strict1607_report.json"
)
