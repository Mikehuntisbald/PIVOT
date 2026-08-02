from config.ablations.cfg_stageb_v18_strong_fpr_tail import *  # noqa: F401,F403

# Retain the immutable warm-started full-text confidence base and train only a
# broadcast residual gate for absolute calibration. The explicit contract keeps
# this phase incompatible with legacy v15 and gate-only v16-v18 checkpoints.
stage_b_v16_confidence_output_mode = "base_plus_gate"
stage_b_v19_explicit_confidence_output_contract = True

# Weak-scope negatives are never accepted by default. Paper Table-B leaf
# configs must bind an exact audited source before enabling this switch.
stage_b_v19_allow_scope_labeled_tn_ablation = False
stage_b_v19_table_b_id = ""
stage_b_v19_table_b_scope_allowlist = []
stage_b_v19_table_b_audit = ""
stage_b_v19_table_b_audit_sha256 = ""
stage_b_v19_table_b_allow_single_edit_token_provenance = False
