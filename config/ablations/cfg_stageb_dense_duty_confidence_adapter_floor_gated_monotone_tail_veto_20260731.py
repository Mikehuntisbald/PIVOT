from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_token_conditioned_20260731 import *  # noqa: F401,F403

# v26 keeps the selective v23 lexical veto but adds a fixed 0.25 fallback
# route. Open edit gates retain full veto strength; closed gates still expose
# high-score TNs to one quarter of the learned monotone depth. A stronger
# positive-delta trust penalty prevents the fallback from eroding the inherited
# positive low tail. No parameter or frozen-rank ownership surface changes.
stage_b_dense_duty_confidence_revision = (
    "word_veto_floor_gated_monotone_tail_veto_v26"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "token_conditioned_floor_gated_monotone_depth_v7"
)
stage_b_v15_tail_queue_positive_trust_weight = 5.0

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_floor_gated_monotone_tail_veto_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "17800ad62a71cd99b79887395f86cb4a42ae3310f8da4f24d988f13a53b13569"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_floor_gated_monotone_tail_veto_strict1607_v26"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_floor_gated_monotone_tail_veto_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
