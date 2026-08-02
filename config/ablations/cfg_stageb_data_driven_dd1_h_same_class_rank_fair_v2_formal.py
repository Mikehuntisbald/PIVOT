from config.ablations.cfg_stageb_data_driven_dd1_h_same_class_rank_fair_v2 import *  # noqa: F401,F403

# This wrapper is the only authorized DD1-H U5020 training entry point. The
# runtime contract rejects resume, a different initializer, output directory,
# update count, loader geometry, AMP mode, or distributed execution.
stage_b_data_driven_execution_scope = "formal_fresh_a1_u5020_v1"
stage_b_data_driven_formal_fresh_start = True
stage_b_data_driven_formal_expected_optimizer_updates = 5020
stage_b_data_driven_formal_config_path = (
    "/media/haoyi/T9/pivot/config/ablations/"
    "cfg_stageb_data_driven_dd1_h_same_class_rank_fair_v2_formal.py"
)
stage_b_data_driven_formal_output_dir = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_h_same_class_rank_fair_v2_seed42_b64_u5020_v1"
)
stage_b_data_driven_formal_preflight_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_h_same_class_rank_fair_v2_preflight/"
    "metadata_and_ledger_audit.json"
)
stage_b_data_driven_formal_preflight_sha256 = (
    "af18debb609bffdceab56b59ebac18cbd2c6dcb42817d6677c95286830785ed5"
)
stage_b_data_driven_formal_probe_receipt_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_h_same_class_rank_fair_v2_seed42_b64_memprobe_strict_v2_u50/"
    "probe_receipt.json"
)
stage_b_data_driven_formal_probe_receipt_sha256 = (
    "00e0bdcc5afc3fb8c9aa625e862fcf8e5eeb5857c252dd7f863bfc8c6d58c2e8"
)
stage_b_data_driven_formal_gate_contract_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_h_same_class_rank_fair_v2_preflight/"
    "formal_gate_contract.json"
)
stage_b_data_driven_formal_gate_contract_sha256 = (
    "66f7196e6112a630f466547b5ec069d54c75dc7eefdabc76e505f27448bd5b60"
)
