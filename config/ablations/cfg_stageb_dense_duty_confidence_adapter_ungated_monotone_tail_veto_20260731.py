from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_token_conditioned_20260731 import *  # noqa: F401,F403

# v25 keeps v23's frozen rank tower, inherited token semantics, independent
# edit-token residual BCE, and token-conditioned pool. The lexical mismatch
# gate remains an input/supervision signal but no longer multiplies the final
# global correction. An independent nonnegative depth can therefore suppress
# a hard high-score TN even when the carrier-token gate is still closed. The
# final confidence can only stay unchanged or move down, and U0 is exactly the
# inherited rank confidence.
stage_b_dense_duty_confidence_revision = (
    "word_veto_ungated_monotone_tail_veto_v25"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "token_conditioned_ungated_monotone_depth_v6"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_ungated_monotone_tail_veto_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "171fdee1b037eedeee4926d2e6cfdd75b8dd9cde7928b58e8d764942e776183f"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_ungated_monotone_tail_veto_strict1607_v25"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_ungated_monotone_tail_veto_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
