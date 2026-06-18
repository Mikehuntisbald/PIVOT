from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v5.4:
# Keep the v5.2 loss/data recipe, but expand the trainable detector scope beyond
# decoder+box heads. Patch branch and BERT stay frozen.

only_train_keywords = [
    "feat_map",
    "class_embed",
    "bbox_embed",
    "transformer.decoder.layers",
    "backbone.0",
    "input_proj",
    "transformer.encoder",
]

only_train_exclude_keywords = [
    "patch_encoder",
]
