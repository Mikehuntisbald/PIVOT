from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v5.5:
# Keep the v5.2 loss/data recipe, but restrict decoder adaptation and aux losses
# to the last three decoder layers, matching the current Stage-A decoder scope.
# Aux output indices map to decoder layers 0..4; the final decoder layer 5 is
# supervised by the main losses. With start_idx=3, aux losses supervise layers
# 3/4 and the main losses supervise layer 5.
# Class/box heads and feat_map remain trainable; patch branch, encoder/backbone/
# input projection, and BERT remain frozen through the inherited freeze list.

only_train_keywords = [
    "feat_map",
    "class_embed",
    "bbox_embed",
    "transformer.decoder.layers.3.",
    "transformer.decoder.layers.4.",
    "transformer.decoder.layers.5.",
]

unfreeze_decoder_last_n_layers = 0
stage_b_aux_loss_start_idx = 3
