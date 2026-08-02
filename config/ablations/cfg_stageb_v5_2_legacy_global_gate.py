from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_alltn_tau05605_m010_w005_tnneg_tokencount import *  # noqa: F401,F403

# Freeze the selected v5.2 localization/text model and train only one
# image-expression confidence offset. The offset is broadcast across queries,
# so RefCOCO box ranking remains exactly the legacy v5.2 ranking.
#
# DATA SEMANTICS: the current 17,829 TN rows verify only the annotated target
# plus their cached SAM3 proposal set (often one proposal). They do not cover
# the actual v5 candidate set and therefore are a benchmark/proposal-set proxy,
# not authoritative image-global negatives. A final FPR run must rebuild labels
# over the actual frozen v5 candidates or use the exact data-FT protocol.
stage_b_legacy_global_gate = True
stage_b_legacy_global_gate_hidden_dim = 128
stage_b_legacy_global_gate_pool_temperature = 0.1
stage_b_legacy_global_gate_score_topk = 10

# Losses operate on max over all deployed queries. The FPR95 objective uses the
# exact positive q05 order statistic and penalizes every negative global max.
stage_b_legacy_global_gate_absolute_weight = 1.0
stage_b_legacy_global_gate_pair_weight = 1.0
stage_b_legacy_global_gate_tail_weight = 1.0
stage_b_legacy_global_gate_pair_margin = 0.30
stage_b_legacy_global_gate_tail_margin = 0.30
stage_b_legacy_global_gate_loss_temperature = 0.10
stage_b_legacy_global_gate_tail_fraction = 0.05
stage_b_legacy_global_gate_tail_objective = "fpr95"
stage_b_legacy_global_gate_require_proposalset_proxy_verified = True
stage_b_legacy_global_gate_label_scope = "proposalset_proxy_not_image_global"

only_train_keywords = ["stage_b_legacy_global_gate"]
unfreeze_decoder_last_n_layers = 0
stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0

# Auxiliary detector outputs are irrelevant because the detector is frozen.
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
batch_size = 8
lr = 3e-4
weight_decay = 1e-4
amp_init_scale = 512.0
amp_max_consecutive_skips = 8
epochs = 1
lr_drop = 100
skip_eval = True
use_coco_eval = False
