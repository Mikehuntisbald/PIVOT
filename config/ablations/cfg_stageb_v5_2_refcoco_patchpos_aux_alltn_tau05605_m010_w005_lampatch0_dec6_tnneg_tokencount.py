from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_alltn_tau05605_m010_w005_tnneg_tokencount import *  # noqa: F401,F403

# Stage-B v5.2 ablation:
# - remove patch CE from the optimized loss while keeping box/GIoU, text, and
#   score calibration losses;
# - train all six decoder layers instead of only the last three.
lambda_patch = 0.0

only_train_keywords = [
    "feat_map",
    "class_embed",
    "bbox_embed",
    "transformer.decoder.layers",
]
unfreeze_decoder_last_n_layers = 6
