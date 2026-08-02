from config.ablations.cfg_stageb_v14_phrase_validity_cvar import *  # noqa: F401,F403

# Historical Stage-B data-FT treats every edited TN caption as image-global
# negative. Keep this as an explicit benchmark mode: local edits are not always
# semantically global negatives, so the label-safe config remains separate.
stage_b_v14_global_tn_all_candidates = True
stage_b_v11_global_tn_negative_weight = 0.25
stage_b_v11_global_tn_tail_weight = 0.5
stage_b_v11_global_tn_tail_topk = 10
stage_b_v11_global_tn_tail_temperature = 0.2
stage_b_v11_global_tn_tail_target_logit = 0.0
