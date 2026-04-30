from config.cfg_patch_stage_a import *  # noqa: F401,F403

# Stage A variant: use precomputed patch_global embeddings from emb_index TSV.
# Note: this bypasses PatchEncoder, so PatchEncoder will NOT be trained in Stage A.
freeze_keywords = list(freeze_keywords) + ["patch_encoder"]

# Recommended: if you trust pretrained box head, disable stabilization first.
bbox_loss_coef = 0.0
giou_loss_coef = 0.0

