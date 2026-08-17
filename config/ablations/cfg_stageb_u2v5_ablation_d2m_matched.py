from config.ablations.cfg_stageb_u2v5_ablation_confidence_base import *  # noqa: F401,F403

stage_b_u2v5_ablation_row_id = "D2m"
stage_b_u2v5_matched_data = True
stage_b_gdino_tn_scope = "traceable_counterfactual_edit"
stage_b_gdino_confidence_objective = "detached_recent_q05_scope_labeled"
stage_b_v19_table_b_id = "D2m"
stage_b_v19_table_b_scope_allowlist = ["traceable_counterfactual_edit"]
stage_b_v19_table_b_audit = "data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/audit.json"
stage_b_v19_table_b_audit_sha256 = "5ff62a838a5123d580a72e353147b97bb69e9d7967348b55cba4ccb9ca36cb96"
