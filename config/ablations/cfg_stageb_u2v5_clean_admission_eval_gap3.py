"""Val-only Gap3 evaluation for a clean U2-v5 admission milestone."""

from config.ablations.cfg_stageb_u2v4_legacy_training_replay_eval_gap3 import *  # noqa: F401,F403

stage_b_u2v5_clean_anchor = True
stage_b_u2v2_initializer_path = (
    "outputs/u2v5_leakage_clean_anchor_20260817/initializer/"
    "checkpoint_clean_init.pth"
)
stage_b_u2v2_initializer_sha256 = (
    "ad7b3a563ef84356c6d952167ee6a48f615f8db887eba31bed92a81b0ba756a7"
)
