from config.ablations.cfg_stageb_v17_full_text_gate_only import *  # noqa: F401,F403

# v17 made the positive-gate trust hinge effective, but left the actual
# FPR@95 tail and paired separation terms behind the inherited 0.05 outer
# weight.  Confidence is an independent scalar branch, so give the operating
# point objective unit weight without changing the rank scorer or its contract.
stage_b_v14_tail_queue_weight = 1.0
stage_b_v15_tail_queue_pair_weight = 1.0
stage_b_v15_tail_queue_positive_trust_weight = 1.0
