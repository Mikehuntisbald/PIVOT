from config.cfg_patch_stage_a import *  # noqa: F401,F403

# Stage A ablation from outputs/stageA_coco_multipatch/checkpoint0002.pth.
# Keep the phase-2/mainline Stage-A recipe used for checkpoint0003/0004, but:
# - unfreeze all 6 decoder layers instead of the mainline last 3 layers
# - disable the epoch-4 LR drop so epoch 3 and epoch 4 both train at base LR
unfreeze_decoder_last_n_layers = 6
lr_drop = 100
