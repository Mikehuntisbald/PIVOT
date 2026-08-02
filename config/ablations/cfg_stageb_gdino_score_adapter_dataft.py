from config.ablations.cfg_stageb_from_gdino_ft_with_tn import *  # noqa: F401,F403

# Confidence-only phase C. Initialize this config from the completed rank-only
# phase R checkpoint with --pretrain_model_path (not --resume). This recipe is
# trained on the fixed data-FT all-TN protocol; final FPR must still be measured
# on the separately audited strict-v2 image-global manifests.
stage_b_gdino_score_adapter = True
stage_b_gdino_adapter_train_mode = "confidence_only"
stage_b_gdino_tn_scope = "benchmark_dataft_alltn"

patch_only = False
stage_b = False
stage_b_v7 = False
stage_b_v11_fixed_text = False
stage_b_legacy_global_gate = False
enable_patch_branch = False

stage_b_gdino_adapter_dim = 128
stage_b_gdino_gate_hidden_dim = 128
# The deployed confidence is the maximum over all 900 frozen queries. A
# temperature of 0.1 lets the aggregate mass of hundreds of background queries
# dominate the pooled feature; 0.01 keeps the gate focused on the queries that
# can actually determine that maximum. Top-3 retains a small response-density
# cue without duplicating the separate score_max feature.
stage_b_gdino_gate_pool_temperature = 0.01
stage_b_gdino_gate_topk = 3

# Referring expressions are not rewritten when an image is mirrored.  In
# particular, left/right pairs would receive reversed semantics under the
# default ODVG/COCO horizontal flip, so every adapter phase disables it.
data_aug_hflip_prob = 0.0

stage_b_gdino_positive_iou_threshold = 0.5
stage_b_gdino_negative_iou_threshold = 0.5
stage_b_gdino_listwise_temperature = 0.1
stage_b_gdino_rank_fix_margin = 0.05
# Maximum amount by which an already-correct frozen positive-vs-negative gap
# may shrink. Fix and preserve rows are normalized separately across DDP.
stage_b_gdino_rank_preserve_margin = 0.02
stage_b_gdino_rank_residual_weight = 1e-3
stage_b_gdino_rank_weight = 0.0
stage_b_gdino_confidence_weight = 1.0
stage_b_gdino_confidence_objective = "detached_recent_q05_trust"
stage_b_gdino_fpr_temperature = 0.1
stage_b_gdino_fpr_margin = 0.0
stage_b_gdino_paired_margin_weight = 0.25
stage_b_gdino_paired_margin = 0.05
# The linear one-sided hinge is a numerical guard.  The zero-valued positive-
# gate translation proxy already removes shared-score drift analytically.
stage_b_gdino_positive_trust_margin = 0.02
stage_b_gdino_positive_trust_weight = 1.0
# At global batch 8, the q05 bank warms after 32 steps and then retains only
# the most recent 64 successful optimizer steps.  Its threshold is detached.
stage_b_gdino_queue_size = 512
stage_b_gdino_queue_min_count = 256

# The two branches have independent optimizer groups and gradient clipping.
lr = 1e-4
stage_b_gdino_rank_lr = 3e-5
stage_b_gdino_gate_lr = 3e-4
clip_max_norm = 0.1
weight_decay = 1e-4

# The base stays in eval mode. Decoder/BERT checkpointing and auxiliary detector
# outputs only add work because every base parameter is frozen.
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
freeze_keywords = None
only_train_keywords = None
unfreeze_decoder_last_n_layers = 0

batch_size = 4
epochs = 1
lr_drop = 100
save_checkpoint_interval = 1
skip_eval = True
use_coco_eval = False
