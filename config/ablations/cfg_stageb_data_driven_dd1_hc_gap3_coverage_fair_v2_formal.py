from config.ablations.cfg_stageb_data_driven_dd1_hc_gap3_coverage_fair_v2 import *  # noqa: F401,F403

# Sole authorized DD1-HC U5020 entry point. Runtime validation rejects resume,
# code-closure drift from the sealed U50 probe, a different initializer,
# dataset, output directory, update budget, loader geometry, or AMP mode.
stage_b_data_driven_execution_scope = "formal_fresh_a1_u5020_v1"
stage_b_data_driven_formal_fresh_start = True
stage_b_data_driven_formal_expected_optimizer_updates = 5020
stage_b_data_driven_formal_config_path = (
    "/media/haoyi/T9/pivot/config/ablations/"
    "cfg_stageb_data_driven_dd1_hc_gap3_coverage_fair_v2_formal.py"
)
stage_b_data_driven_formal_output_dir = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_hc_gap3_coverage_fair_v2_seed42_b64_u5020_v1"
)
stage_b_data_driven_formal_preflight_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_hc_gap3_coverage_fair_v2_preflight/"
    "metadata_causal_and_ledger_audit.json"
)
stage_b_data_driven_formal_preflight_sha256 = (
    "c09ce8ff8fcfc899908badca19fa7d13bd1bff34bb157296a727bebf1ee0804b"
)
stage_b_data_driven_formal_probe_receipt_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_hc_gap3_coverage_fair_v2_seed42_b64_memprobe_seal_u50_v2/"
    "probe_receipt.json"
)
stage_b_data_driven_formal_probe_receipt_sha256 = (
    "a8d67aa33ffdfa8c355f115fcd27f29275682ccef7df901afb04773054f618fa"
)
stage_b_data_driven_formal_gate_contract_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_hc_gap3_coverage_fair_v2_preflight/"
    "formal_gate_contract.json"
)
stage_b_data_driven_formal_gate_contract_sha256 = (
    "e051c52dd45429df4d89afe1012294b0d1545df13bca10f058be8d2a38267117"
)
