from config.ablations.cfg_stageb_v23_table_b_common import *  # noqa: F401,F403

# C2 causal panel: parent expression, dataset, image, sentence and edit
# category are paired one-to-one. This is a distinct 7,074-row audit boundary;
# it never aliases the D1-D3 equal-exposure audit.
stage_b_v24_matched_causal_panel = True
stage_b_v23_tn_token_provenance_contract = "disabled_uniformly_D2m_D3m"
stage_b_v19_table_b_audit = (
    "data/ablations/stageb_tn_c2_parent_matched_20260717/audit.json"
)
stage_b_v19_table_b_audit_sha256 = (
    "ca1c9c581fd78f1fe026397cc127d9b7448c60227b31c5e83148c91e9c61861e"
)
