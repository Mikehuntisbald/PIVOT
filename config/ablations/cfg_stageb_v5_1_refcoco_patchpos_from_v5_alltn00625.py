from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_alltn00625_from_w0125 import *  # noqa: F401,F403

# Stage-B v5.1:
# Keep the selected v5 alltn00625 recipe, but do not train patch CE negatives
# on RefCOCO-family phrase rows. RefCOCO annotations mark only the referred
# phrase object, so unmatched same-class/person queries can be false negatives
# for patch-only classification. LVIS/COCO patch CE remains unchanged.

patch_ce_positive_only_for_datasets = (
    "refcoco",
    "refcocoplus",
    "refcocog",
    "refexp",
)
