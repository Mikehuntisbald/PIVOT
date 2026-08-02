from config.ablations.cfg_stageb_gdino_score_adapter_rank_o64_direct_u500 import *  # noqa: F401,F403

# Resource-only amendment after the preregistered B64 identity forward OOMed
# before producing metrics.  Gradient accumulation preserves effective B64.
batch_size = 32
