from config.ablations.cfg_stageb_v22_s2_independent_joint_full import *  # noqa: F401,F403

# Headline main row: the finalized S2F architecture/objective trained under a
# successful-optimizer-update global batch-slot budget matched to b58.
stage_b_v25_main_id = "M0"
stage_b_v25_compute_contract = "b58_successful_update_batch_slot_matched"
stage_b_v25_budget_unit = "successful_optimizer_update_global_batch_slots"
stage_b_v25_successful_update_batch_slots = 941_280
stage_b_v25_initializer_contract = "same_stage_a_model_and_scorer_no_b58"
stage_b_v25_strict_resume = True

# Both the frozen Stage-A candidate generator and the independent text scorer
# originate from the same scorer-free Stage-A checkpoint. The runner repeats
# this path explicitly and records both roles in its immutable input closure.
stage_b_v15_scorer_init_checkpoint = "/media/haoyi/T9/gdino/checkpoint0004.pth"

# Keep the finalized S2F optimizer semantics explicit in the headline leaf.
batch_size = 40
epochs = 8
lr_drop = 4
save_checkpoint_interval = 1
onecyclelr = False
multi_step_lr = False
lr = 2e-5
stage_b_v15_validity_lr = 5e-4
weight_decay = 1e-4
clip_max_norm = 0.1
stage_b_v15_separate_grad_clip = True
stage_b_v22_gradient_diagnostic_interval = 100
