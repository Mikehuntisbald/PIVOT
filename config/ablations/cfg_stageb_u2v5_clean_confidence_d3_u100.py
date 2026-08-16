"""Fresh, image-disjoint D3 confidence phase after clean U2-v4 admission."""

from config.ablations.cfg_stageb_u2v5_clean_admission_eval_gap3 import *  # noqa: F401,F403

# Switch ownership from the complete admission subsystem to confidence12.
stage_b_u2v4_legacy_training_replay = False
stage_b_u2v4_checkpoint_eval = False
stage_b_u2v5_clean_confidence = True
stage_b_gdino_adapter_train_mode = "confidence_only"
stage_b_gdino_rank_weight = 0.0
stage_b_gdino_confidence_weight = 1.0

# D3 is image-disjoint from both strict manifests, but verifies only the
# annotated target plus cached proposals.  Keep this weaker scope explicit;
# never relabel it as image-global/all-query supervision.
stage_b_gdino_tn_scope = "proposal_covered_verified"
stage_b_gdino_confidence_objective = "detached_recent_q05_proposal_covered"
stage_b_gdino_paired_margin_weight = 0.0
stage_b_gdino_positive_trust_margin = 0.02
stage_b_gdino_positive_trust_weight = 1.0
stage_b_gdino_queue_size = 512
stage_b_gdino_queue_min_count = 256
stage_b_gdino_fpr_temperature = 0.1
stage_b_gdino_fpr_margin = 0.0

# Reuse the sealed Table-B loader audit.  These fields authorize only the D3
# dataset binding; they do not upgrade D3 into an image-global label source.
stage_b_v19_allow_scope_labeled_tn_ablation = True
stage_b_v19_table_b_id = "D3"
stage_b_v19_table_b_scope_allowlist = ["proposal_covered_verified"]
stage_b_v19_table_b_allow_single_edit_token_provenance = False
stage_b_v19_table_b_audit = (
    "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
)
stage_b_v19_table_b_audit_sha256 = (
    "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
)
stage_b_v14_global_tn_all_candidates = True
stage_b_v15_decoupled_confidence = True
stage_b_v19_explicit_confidence_output_contract = True
stage_b_v16_confidence_output_mode = "base_plus_gate"

lr = 1e-4
stage_b_gdino_rank_lr = 3e-5
stage_b_gdino_gate_lr = 3e-4
weight_decay = 1e-4
clip_max_norm = 0.1
amp_init_scale = 8192.0

batch_size = 8
epochs = 1
lr_drop = 100
save_checkpoint_interval = 25
skip_eval = True
use_coco_eval = False
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
freeze_keywords = None
only_train_keywords = None
unfreeze_decoder_last_n_layers = 0
