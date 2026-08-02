from config.ablations.cfg_stageb_v22_s2_independent_joint import *  # noqa: F401,F403

# Throwaway one-batch preflight for the S3 graph before splitting optimizer
# ownership. Both trainable branches are visible only so autograd can prove
# bidirectional isolation; this config is not a Table-D training row.
stage_b_v22_table_id = "S3-isolation-probe"
stage_b_v22_score_ownership = "independent_decoders_two_phase"
stage_b_v22_train_phase = "isolation_probe"
stage_b_v22_objective_fidelity = (
    "common_objective_ownership_ablation_probe_only"
)
stage_b_v22_probe_only = True
stage_b_v22_gradient_diagnostic_interval = 1
skip_eval = True
