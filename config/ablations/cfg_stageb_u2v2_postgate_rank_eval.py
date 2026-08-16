"""U2-v2 post-gate residual evaluation without the D2 training binding."""

from config.ablations.cfg_stageb_u2v2_postgate_rank_u100 import *  # noqa: F401,F403

# The model route and checkpoint contract stay enabled; only the row-locked D2
# loader guard is disabled so evaluator-built Ref/TN manifests can be consumed.
stage_b_u2v2_training_dataset_binding = False
stage_b_u2v2_emit_causal_ref_routes = True

# Restore the sealed B58/C100 evaluation geometry.  The training leaf uses
# aspect-preserving 800/max1333 augmentation, but it must never leak into the
# fixed-size Ref/TN comparison protocol.
fix_size = True
data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_train_deterministic_aspect_resize = False
