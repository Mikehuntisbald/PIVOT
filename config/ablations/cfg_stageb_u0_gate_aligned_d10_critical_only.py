"""D10-C: change only deployment-critical blockers, not broad coverage."""

from config.ablations.cfg_stageb_u0_gate_aligned_d10 import *  # noqa: F401,F403

# D9 has already learned category-complete geometry.  The U25 D10 pilot showed
# that repeating an every-instance inward barrier dominates the sparse
# deployment errors and changes too many otherwise-correct gate decisions.
# D10-C keeps the exact blocker/critical-positive/positive-winner terms and
# removes only that redundant broad coverage gradient.
stage_b_u0_d10_variant = "critical_only"
stage_b_u0_d10_instance_coverage_weight = 0.0
