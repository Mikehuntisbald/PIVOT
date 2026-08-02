from config.ablations.cfg_stageb_v12_predicate_token_rank import *  # noqa: F401,F403

# Keep positive and paired-TN absolute anchors equally weighted even when the
# clean/pair sampling ratio changes, and calibrate score tails over the full DDP
# batch rather than each GPU's four samples independently.
stage_b_v11_balance_local_anchor_classes = True
stage_b_v11_batch_tail_ddp_global = True
