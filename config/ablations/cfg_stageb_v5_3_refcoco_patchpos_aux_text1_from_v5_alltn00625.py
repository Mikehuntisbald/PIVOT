from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v5.3:
# v5.2 plus a stronger text loss. This is a single-variable probe against v5.2:
# patch-positive RefCOCO CE, aux losses, TN calibration, and freeze/unfreeze
# settings remain inherited.

lambda_text = 1.0
