from config.ablations.cfg_stageb_from_gdino_ft_with_tn import *  # noqa: F401,F403

# Table A G0c: continue the historical pure-GDINO Stage-B data-FT checkpoint
# for exactly the same optimizer-update count and effective global batch as the
# proposed row. The launcher uses physical batch 10 with four-way gradient
# accumulation (effective batch 40) and supplies checkpoint b58 as pretraining.
paper_table_a_id = "G0c"
paper_table_a_control = "continued_gdino_same_sources_same_updates"
paper_table_a_data_audit = (
    "data/ablations/stageb_table_a_continued_gdino_20260717/audit.json"
)

# Referring expressions containing left/right are not rewritten under a flip.
data_aug_hflip_prob = 0.0

# The launcher terminates on max_train_iters. Keep the epoch ceiling out of the
# way and never select a checkpoint by validation performance.
epochs = 100
lr_drop = 100
skip_eval = True
use_coco_eval = False

# Match the paper rows' stable AMP contract. Postflight still requires zero
# skipped steps, while this bound prevents an unproductive infinite retry run.
amp_init_scale = 512.0
amp_max_consecutive_skips = 8
