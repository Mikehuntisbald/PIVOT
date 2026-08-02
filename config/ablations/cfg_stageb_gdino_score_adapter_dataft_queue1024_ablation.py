from config.ablations.cfg_stageb_gdino_score_adapter_dataft import *  # noqa: F401,F403

# Delayed/recent-queue ablation. At global batch 8 this does not activate until
# step 128 and then retains the latest 128 steps.
stage_b_gdino_queue_size = 1024
stage_b_gdino_queue_min_count = 1024
