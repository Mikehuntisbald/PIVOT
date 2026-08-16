"""Leakage-clean U2-v4 admission phase on Stage-A + positive-only R100."""

from config.ablations.cfg_stageb_u2v4_legacy_training_replay_u100 import *  # noqa: F401,F403

stage_b_u2v5_clean_anchor = True
# Erase inherited diagnostic-only C100 provenance.  The clean builder and
# runtime contract reject any confidence source other than R100 identity.
stage_b_u2v2_c100_checkpoint = None
stage_b_u2v2_c100_sha256 = None
stage_b_u2v2_initializer_path = (
    "outputs/u2v5_leakage_clean_anchor_20260817/initializer/"
    "checkpoint_clean_init.pth"
)
stage_b_u2v2_initializer_sha256 = (
    "ad7b3a563ef84356c6d952167ee6a48f615f8db887eba31bed92a81b0ba756a7"
)

# The inherited U2-v4 mechanism is intentional: surface8 and its auxiliary
# residual8 form one admission subsystem.  R100 and identity confidence12 are
# frozen and have no autograd connection to its loss.
stage_b_gdino_tn_scope = ""
