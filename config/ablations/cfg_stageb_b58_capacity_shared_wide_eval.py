"""Gap3/Strict evaluation for the locked B58 Shared-Wide capacity arm."""

from config.ablations.cfg_stageb_b58_capacity_shared_wide import *  # noqa: F401,F403

stage_b_u2v5_ownership = True
stage_b_u2v5_ownership_eval = False
stage_b_b58_capacity_control_eval = True
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 3.0
stage_b_u2v2_category_gate_max_gap = 3.0
stage_b_gdino_ref_top1_guard = False
stage_b_u2v5_emit_causal_ref_routes = True
stage_b_gdino_tn_scope = "proposal_covered_verified"
stage_b_v19_table_b_id = "D3"
stage_b_v19_table_b_audit_sha256 = "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
stage_b_u2v4_checkpoint_eval = False
stage_b_u2v4_legacy_training_replay = False
