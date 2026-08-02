from config.ablations.cfg_stageb_data_driven_dd1_category_complete import *  # noqa: F401,F403

# SHA-independent DD1 -> confidence-phase architecture/loss template. Artifact
# and dataset bindings belong to the DD2/DD3 consumer configs, not this file.
stage_b_data_driven_train_mode = "confidence_pair"
stage_b_data_driven_category_complete = True
stage_b_data_driven_confidence_trained = True

stage_b_data_driven_confidence_weight = 1.0
stage_b_data_driven_token_weight = 0.0
stage_b_data_driven_shared_token_weight = 0.25
stage_b_data_driven_fpr_temperature = 0.1
stage_b_data_driven_fpr_margin = 0.0
stage_b_data_driven_target_tpr = 0.95
stage_b_data_driven_positive_queue_size = 4096
