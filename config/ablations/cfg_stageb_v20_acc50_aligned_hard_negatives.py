from config.ablations.cfg_stageb_v18_strong_fpr_tail import *  # noqa: F401,F403

# RefCOCO Acc@0.5 accepts candidates at IoU >= 0.5.  Keep the rank-path
# negative pool immediately below that operating point so this ablation tests
# acc50-aligned hard negatives without changing the v18 score contract.
stage_b_v11_negative_iou_threshold = 0.499
stage_b_v20_acc50_aligned_hard_negatives = True
