from config.ablations.cfg_stageb_v12_predicate_token_rank import *  # noqa: F401,F403

# Controlled continuation from the selected v12@4000 checkpoint. The 500-step
# limit is an argparse runtime guard and must be supplied as
# `--max_train_iters 500`; putting that key here would collide with argparse.
batch_size = 4
epochs = 1

# Rank-only objective: locate the positive expression among fixed Stage-A
# candidates and compare paired positive/TN text on those same candidates.
stage_b_v11_assert_fixed_candidates = True
stage_b_v11_listwise_weight = 0.2
stage_b_v11_local_tn_rank_weight = 1.0
stage_b_v11_predicate_tn_rank_weight = 1.0

# Do not optimize absolute score calibration or image-global rejection in this
# probe. It isolates whether adding RefCOCO positive data repairs box ranking.
stage_b_v11_local_anchor_weight = 0.0
stage_b_v11_batch_tail_separation_weight = 0.0
stage_b_v11_global_tn_negative_weight = 0.0
stage_b_v11_global_tn_tail_weight = 0.0

# Keep the scorer exactly on the v12 architecture and loss surface. These
# explicit guards prevent later v14/v15 defaults from changing this ablation.
stage_b_v14_validity_head = False
stage_b_v15_decoupled_confidence = False
stage_b_v14_local_absolute_weight = 0.0
stage_b_v14_predicate_absolute_weight = 0.0
stage_b_v14_tail_queue_weight = 0.0
stage_b_v14_tail_queue_size = 0
stage_b_v14_global_tn_all_candidates = False
