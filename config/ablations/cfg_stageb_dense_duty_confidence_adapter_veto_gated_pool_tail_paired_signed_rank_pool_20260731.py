from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_rank_channel_20260731 import *  # noqa: F401,F403

# v20 restores the proven P50-style signed sample calibration path without
# restoring a second text tower. The pool consumes detached, fully text-
# conditioned rank queries plus patch statistics; token logits and the exact
# word veto remain independently supervised.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_tail_paired_signed_rank_pool_v20"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_rank_query_plus_patch_statistics_signed_residual_v2"
)
stage_b_v11_trainable_params_min = 236_806
stage_b_v11_trainable_params_max = 236_806

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_"
    "signed_rank_pool_trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "816aa1b4a3b91e61d3d1a384979e7196e2cb46f4c29d67406ab3223303b9ba01"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_tail_paired_signed_rank_pool_strict1607_v20"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_tail_paired_signed_rank_pool_"
    "highmem_20260731/probe_evaluation/u000300_strict1607_report.json"
)
