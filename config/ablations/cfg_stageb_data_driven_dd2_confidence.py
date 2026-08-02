from config.ablations.cfg_stageb_data_driven_confidence_template import *  # noqa: F401,F403

# DD2 starts from the sealed DD1 phase checkpoint and trains only the disjoint
# absolute-confidence branch on aligned positive/proposal-covered-TN pairs.
stage_b_data_driven_experiment_id = "DD2"
# Filled only after the sealed DD1-U1000 checkpoint is converted. An empty SHA
# deliberately makes formal DD2/DD3 launches fail closed in the meantime.
stage_b_data_driven_confidence_initializer_scope = "formal"
stage_b_data_driven_confidence_initializer_min_dd1_updates = 1000
stage_b_data_driven_confidence_initializer_sha256 = (
    "0ca957ff1dfda4d49f3e2138f416223ad7d8e37e19016004a237b42944333ed1"
)
stage_b_data_driven_confidence_dataset_config_path = (
    "/media/haoyi/T9/pivot/"
    "config/datasets_stageb_data_driven_dd2_dd3_proposal_covered_pairs.json"
)
stage_b_data_driven_confidence_dataset_config_sha256 = (
    "7b80355e302dd31ca83e6fc9296b4855b4f2a21a8ba715772f93310dd1c8dcca"
)

# These v19-named fields are only the existing fail-closed Table-B data
# binding (audit path, SHA, row count, scope, and per-row provenance). No
# legacy scorer, confidence head, or loss is enabled in the data-driven route.
stage_b_v19_allow_scope_labeled_tn_ablation = True
stage_b_v19_table_b_id = "D3"
stage_b_v19_table_b_scope_allowlist = ["proposal_covered_verified"]
stage_b_v19_table_b_audit = (
    "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
)
stage_b_v19_table_b_audit_sha256 = (
    "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
)
stage_b_v19_table_b_allow_single_edit_token_provenance = True

# DD2 and DD3 use the same natural two-epoch exposure budget. With 14,196
# paired rows and drop_last=True this is exactly 2 * floor(14196 / 64) = U442.
batch_size = 64
epochs = 2
