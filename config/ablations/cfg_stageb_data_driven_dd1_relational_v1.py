from config.ablations.cfg_stageb_data_driven_relational_v1_initializer import *  # noqa: F401,F403

# A1 fair comparison against DD1 absolute-token A0: same b58, DD1 data, loss,
# LR, seed and update budget; only the rank architecture differs.
stage_b_data_driven_base_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_initializers/seed42/checkpoint_dd_relational_v1_init.pth"
)
stage_b_data_driven_base_initializer_sha256 = (
    "7caede9e52a6015d554e098991e54314662bc9fb5003ab92c3bdbfe26f0f15ab"
)
