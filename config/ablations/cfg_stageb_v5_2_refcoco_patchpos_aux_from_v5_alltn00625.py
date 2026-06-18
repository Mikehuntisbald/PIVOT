from config.ablations.cfg_stageb_v5_1_refcoco_patchpos_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v5.2:
# v5.1 plus auxiliary decoder losses. StageBCriterion applies aux patch CE,
# text, bbox, and GIoU losses on intermediate decoder layers; score calibration
# remains final-layer only.

aux_loss = True
use_checkpoint = False
