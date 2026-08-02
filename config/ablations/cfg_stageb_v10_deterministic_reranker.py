from config.ablations.cfg_stageb_v9_explicit_tn_pair_reranker import *  # noqa: F401,F403

# The architecture/loss intentionally match v9. engine.py keeps the frozen
# Stage-A proposal tower and frozen verifier BERT in eval mode, so this run
# isolates the train/eval stochasticity fix before adding scorer capacity.
