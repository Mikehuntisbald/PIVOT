from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v5.2 text-weight sweep:
# v5.2 plus lambda_text=0.50. This keeps patch-positive RefCOCO CE, aux losses,
# TN calibration, and freeze/unfreeze settings unchanged.

lambda_text = 0.5
