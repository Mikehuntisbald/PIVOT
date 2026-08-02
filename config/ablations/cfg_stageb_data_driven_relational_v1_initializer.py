from config.ablations.cfg_stageb_data_driven_dd1_category_complete import *  # noqa: F401,F403
from config.ablations.cfg_stageb_data_driven_relational_v1_spec import *  # noqa: F401,F403

# The initializer template is intentionally SHA-independent. The consuming
# training config binds the published relational initializer after generation.
stage_b_data_driven_base_initializer_path = ""
stage_b_data_driven_base_initializer_sha256 = ""
