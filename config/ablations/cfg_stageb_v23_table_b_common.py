from config.ablations.cfg_stageb_v21_token_l4_edit_bce_pair_rank import *  # noqa: F401,F403

# Table B fixes the selected v19 base-plus-gate + Acc50 hard-negative + L4
# architecture/objective. TN edit-token provenance is disabled uniformly by
# the D1-D3 dataset manifests, so this table isolates paired TN/confidence data
# quality; Table C owns the edit-token-label ablation.
stage_b_v23_ablation_table = "B"
stage_b_v23_objective_contract = "v19_base_plus_gate_acc50_hardneg_v21_l4"
stage_b_v23_tn_token_provenance_contract = "disabled_uniformly_D1_D3"
stage_b_v19_table_b_allow_single_edit_token_provenance = False

stage_b_v19_table_b_audit = (
    "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
)
stage_b_v19_table_b_audit_sha256 = (
    "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
)
