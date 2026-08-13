"""Data-FT all-TN confidence phase with the total-trust objective.

This is a confidence-only continuation of the data-FT adapter protocol.  The
rank branch is frozen by ``stage_b_gdino_adapter_train_mode``; only the
confidence gate/adapter owner is allowed to update.  Start from a completed
rank checkpoint with ``--pretrain_model_path`` and keep the data-FT all-TN
scope unchanged for a fair comparison.

The objective name is intentionally explicit.  Its implementation and
checkpoint compatibility are owned by the adapter runtime, not this config
leaf.
"""

from config.ablations.cfg_stageb_gdino_score_adapter_dataft import *  # noqa: F401,F403


# Keep the benchmark data-FT all-TN exposure fixed while changing only the
# confidence objective.  ``confidence_only`` freezes rank parameters and
# leaves the gate/adapter as the sole trainable owner.
stage_b_gdino_adapter_train_mode = "confidence_only"
stage_b_gdino_tn_scope = "benchmark_dataft_alltn"
stage_b_gdino_rank_weight = 0.0
stage_b_gdino_confidence_weight = 1.0
stage_b_gdino_confidence_objective = "detached_recent_q05_total_trust"
# Keep the default objective focused on the deployed-score tail and TN
# suppression.  The always-active pair term is retained only as an ablation.
stage_b_gdino_paired_margin_weight = 0.0

# The available probe host has one 32 GiB GPU. Keep the historical global
# batch of eight without entering torch.distributed for a one-process launch.
batch_size = 8
