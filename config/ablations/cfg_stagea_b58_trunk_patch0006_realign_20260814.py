_base_ = "../cfg_patch_stage_a.py"

# Stage-A B58 realignment: B58 owns every query-producing and box-producing
# tensor.  checkpoint0006 contributes only the independent patch score path.
stage_a_b58_patch_realign = True
stage_a_b58_checkpoint_path = (
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)
stage_a_b58_checkpoint_sha256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
stage_a_patch0006_checkpoint_path = (
    "/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch/checkpoint0006.pth"
)
stage_a_patch0006_checkpoint_sha256 = (
    "a4f153c8cbd9b408b9479901e27ec486a10f393013193d44b0da1dcd1888cb91"
)
stage_a_b58_patch_initializer_path = (
    "/media/haoyi/T9/gdino/outputs/"
    "stageA_b58_trunk_patch0006_realign_20260814_initializer.pth"
)
stage_a_b58_patch_initializer_sha256 = (
    "25ca02e5ec7b127f1d90f5642f7d36035c0eb71669ad9aa85cd158f12eedf3b8"
)

# No decoder layer, encoder, image backbone, BERT/text fusion, bbox/class head,
# or DN query may move.  The trainable patch projection learns to read the fixed
# B58 query coordinate system without sending gradients into it.
unfreeze_decoder_last_n_layers = 0
patch_dn_num_queries = 0
only_train_keywords = [
    "patch_encoder.input_proj",
    "patch_encoder.norm",
    "query_proj_for_patch",
    "patch_logit_scale",
]
only_train_exclude_keywords = ["patch_encoder.backbone"]

# The B58 box/query surface is already mature.  Use a smaller head-only rate
# than historical Stage A and keep the original patch objective/data contract.
batch_size = 24
lr = 2e-5
lr_drop = 4
epochs = 8
