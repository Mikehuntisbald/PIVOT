"""B58 capacity control: one wider owner feeding separate rank/rejection outputs."""

from config.ablations.cfg_stageb_u2v5_ablation_ownership_base import *  # noqa: F401,F403

stage_b_u2v5_ablation_row_id = "B58_SHARED_WIDE"
stage_b_u2v5_score_ownership = "shared_wide_two_heads"
stage_b_gdino_adapter_dim = 163
stage_b_gdino_gate_hidden_dim = 62
