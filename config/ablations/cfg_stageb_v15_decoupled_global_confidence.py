from config.ablations.cfg_stageb_v14_phrase_validity_cvar import *  # noqa: F401,F403

# Candidate ranking and image-text rejection are separate optimization paths.
# The confidence head sees detached decoder features and broadcasts one scalar
# gate to every candidate, so confidence training cannot change box ordering.
stage_b_v15_decoupled_confidence = True
stage_b_v15_validity_pool_temperature = 0.2
stage_b_v15_patch_rank_fusion = True
stage_b_v15_patch_rank_weight = 1.0
stage_b_v15_exclude_canonical_from_score = True
stage_b_v15_tail_queue_global_scores = True
stage_b_v15_tail_queue_objective = "fpr95"
stage_b_v15_separate_grad_clip = True

# Optional scorer-only warm-start. Set this through --options to a mature
# full-text GDINO checkpoint. The Stage-A candidate path still comes only from
# --pretrain_model_path; default empty preserves the legacy initialization.
stage_b_v15_scorer_init_checkpoint = ""

# The detached confidence head needs a materially larger step than the rank
# decoder. main.py isolates this prefix into its own optimizer group, while the
# decoder remains at lr / lr_linear_proj_mult from the inherited v12 config.
stage_b_v15_validity_lr = 5e-4

# The confidence decoder is an immutable Stage-A/initial-text snapshot. The
# trainability filter is name based, so exclude it explicitly as well as
# enforcing requires_grad=False inside the scorer.
only_train_exclude_keywords = ["stage_b_fixed_text_scorer.confidence_decoder"]

# Spatial words in full expressions are labels, not image-only annotations.
# Horizontal flipping without rewriting left/right corrupts those labels.
data_aug_hflip_prob = 0.0

# Rank/localization branch. These losses update the fixed-box text decoder but
# never the confidence inputs supplied to the absolute/global objectives.
stage_b_v11_listwise_weight = 0.5
stage_b_v11_local_tn_rank_weight = 0.5
stage_b_v11_predicate_tn_rank_weight = 1.0

# Confidence branch. This config must be used only with proposal-verified
# global TN rows; the companion data manifest enforces that contract.
stage_b_v14_global_tn_all_candidates = True
stage_b_v14_local_absolute_weight = 1.0
stage_b_v14_predicate_absolute_weight = 0.0
stage_b_v11_global_tn_negative_weight = 0.25
stage_b_v11_global_tn_tail_weight = 0.5

# Directly optimize the same per-image global maxima used by FPR@95TPR. A
# modest queue weight avoids the 5%/95% CVaR term overwhelming rank learning.
stage_b_v14_tail_queue_weight = 0.05
stage_b_v14_tail_queue_size = 4096
stage_b_v14_tail_queue_min_count = 256
stage_b_v14_tail_queue_positive_quantile = 0.05
stage_b_v14_tail_queue_negative_quantile = 0.95
stage_b_v14_tail_queue_temperature = 0.1
stage_b_v14_tail_queue_margin = 0.3
stage_b_v15_tail_queue_pair_weight = 0.25
stage_b_v15_tail_queue_pair_margin = 0.05
stage_b_v15_tail_queue_positive_trust_weight = 1.0
stage_b_v15_tail_queue_positive_trust_margin = 0.02

batch_size = 56
stage_b_v11_expression_microbatch = 16
