from config.ablations.cfg_stageb_dense_duty_confidence_candidate_complete_trace_c1_20260803 import *  # noqa: F401,F403

# C2 removes the free-head bypass. Deployed depth is a fixed monotone function
# of the verifier's non-canonical semantic-word entailment.
stage_b_dense_duty_confidence_candidate_trace_contract = (
    "candidate_complete_monotone_token_entailment_v2"
)
stage_b_dense_duty_confidence_capacity_contract = (
    "rank_cloned_full_decoder_candidate_complete_monotone_v4"
)
stage_b_dense_duty_confidence_variant = (
    "candidate_complete_trace_monotone_token_entailment_c2"
)

# The serialized free head remains for V27 checkpoint compatibility but is
# frozen and absent from the optimizer.
stage_b_v11_trainable_params_min = 25_464_320
stage_b_v11_trainable_params_max = 25_464_320

# C2's only all-candidate lexical gradient is the provenance-safe monotone
# expression-depth objective. The inherited V62 raw-residual losses broadcast
# changed-word labels without candidate provenance and must remain disabled.
stage_b_dense_duty_raw_veto_gate_weight = 0.0
stage_b_dense_duty_raw_veto_carrier_pair_weight = 0.0

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_candidate_complete_trace_c2_audit_20260803/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "9266e285466190facce59fdd69cc584a51ce0e1e34ce115f7e3236fd48353a25"
)
