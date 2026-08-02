from config.ablations.cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2 import *  # noqa: F401,F403

# Sole authorized full-data U5020 entry point. The runtime validator requires a
# fresh A1 start and exact content hashes for the paired U50 causal receipt,
# preflight, fixed headline gate, training code closure, data, and initializer.
stage_b_data_driven_execution_scope = "formal_fresh_a1_u5020_v1"
stage_b_data_driven_formal_fresh_start = True
stage_b_data_driven_formal_expected_optimizer_updates = 5020
stage_b_data_driven_formal_config_path = (
    "/media/haoyi/T9/pivot/config/ablations/"
    "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2_formal.py"
)
stage_b_data_driven_formal_output_dir = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_pairtop1_hardgap3_fair_v2_seed42_b64_u5020_v1"
)
stage_b_data_driven_formal_preflight_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_pairtop1_hardgap3_fair_v2_preflight/"
    "metadata_causal_and_ledger_audit.json"
)
stage_b_data_driven_formal_preflight_sha256 = (
    "8ffc0ac5b174949d8799e66b69cfa3c3474c63ea05be33a5a9891d5d93c7de38"
)
stage_b_data_driven_formal_probe_receipt_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_pairtop1_hardgap3_fair_v2_seed42_b64_fullprobe_u50_v2/"
    "probe_receipt.json"
)
stage_b_data_driven_formal_probe_receipt_sha256 = (
    "d530a2d97be56fe08fa0aa27468c7791965b43bb148395f1e42ddf110f66bf6f"
)
stage_b_data_driven_formal_gate_contract_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_pairtop1_hardgap3_fair_v2_preflight/"
    "formal_gate_contract.json"
)
stage_b_data_driven_formal_gate_contract_sha256 = (
    "067abfe62838adee0de343d6be5c396132322db21bfdfc9c195490b4282adea0"
)

epochs = 1
batch_size = 64
persistent_workers = False
