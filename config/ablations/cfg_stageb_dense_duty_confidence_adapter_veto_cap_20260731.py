from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gate_20260731 import *  # noqa: F401,F403

# CVPR confidence revision v4. Exact changed-word residuals open a detached
# candidate gate; frozen reference scores turn candidate gates into expression
# coverage; a smooth absolute cap is applied after the shared confidence pool.
stage_b_dense_duty_confidence_revision = "word_veto_coverage_absolute_cap_v4"
stage_b_dense_duty_confidence_phrase_aggregation = (
    "trace_activated_word_veto_absolute_cap_v4"
)
stage_b_dense_duty_confidence_veto_gate_offset = 0.05
stage_b_dense_duty_confidence_veto_gate_scale = 0.10
stage_b_dense_duty_confidence_veto_coverage_offset = 0.10
stage_b_dense_duty_confidence_veto_coverage_ramp = 0.80
stage_b_dense_duty_confidence_veto_cap_temperature = 0.10
stage_b_dense_duty_confidence_veto_cap_initial_ceiling = -0.10

# The positive IoU queries plus the frozen reference-score carrier are closed.
# The exact changed word is opened on every admitted TN candidate. Each side is
# reduced per sample, so examples with more candidates do not receive more loss.
stage_b_dense_duty_raw_veto_query_scope = (
    "tn_all_admitted_positive_carrier_v2"
)
stage_b_dense_duty_raw_veto_positive_margin = 0.10
stage_b_dense_duty_raw_veto_tn_margin = 0.15

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_absolute_cap_strict1607_v4"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_cap_highmem_20260731/probe_evaluation/"
    "u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_cap_trace_audit_20260731/"
    "receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "40d0f8e7f519a815fa21ef2f0c6a7eb8fba583af6fa19d64d9687f8c8c2ae70e"
)

# v3 had 185,924 active adapter/pool parameters. v4 adds one constrained cap
# ceiling scalar and leaves the frozen U6551 rank tower bitwise unchanged.
stage_b_v11_trainable_params_min = 185_925
stage_b_v11_trainable_params_max = 185_925
