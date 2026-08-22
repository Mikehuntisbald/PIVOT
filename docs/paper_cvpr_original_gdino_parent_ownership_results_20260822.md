# Original GroundingDINO pre-Stage-B ownership results

## Outcome

The fixed Shared-Wide/Isolated replay is complete on the direct parent of B58,
seeds 17/42/73, U150.  Hard isolation is substantially worse for Ranking and
does not improve rejection.

| Route | Test5 P@1 | TestAB P@1 | Strict2031 FPR95 |
|---|---:|---:|---:|
| Native parent | 0.395331 | 0.405878 | **0.896603** |
| Shared-Wide | **0.431238 ± 0.005686** | **0.430277 ± 0.003308** | 0.919908 ± 0.016392 |
| Isolated | 0.411228 ± 0.004863 | 0.419116 ± 0.003948 | 0.926801 ± 0.018982 |

Learned routes are three-seed mean ± sample SD.  Lower FPR95 is better.
Relative to Isolated, Shared-Wide gains 2.001 Test5 points and 1.116 TestAB
points.  The confidence heads do not beat the native full-expression score on
Strict2031.

## Gradient geometry

Shared-Wide fixed probes:

| Horizon | mean cosine | P(cos<0) | q05 | minimum |
|---|---:|---:|---:|---:|
| U150 | +0.0118 | 0.500 | -0.2288 | -0.3240 |
| U25/U50/U100/U150 | +0.0541 | 0.354 | -0.1005 | -0.3240 |

The three U150 seed means are -0.0538, -0.0061, and +0.0955.  A negative
tail therefore coexists with a large deployment advantage for sharing.
Isolated has no cross-task autograd path at every milestone.

## Paired contrasts

The preregistered candidate is Isolated and the reference is Shared-Wide.
FPR95 gain is Shared-Wide minus Isolated, so positive values favor isolation.

| Endpoint | Isolated−Shared gain | Paired 95% CI | Result |
|---|---:|---:|---|
| Test5 | -0.020009 | [-0.022694, -0.017354] | Isolated fails non-inferiority |
| TestAB | -0.011161 | [-0.014823, -0.007411] | Isolated fails non-inferiority |
| FPR95 reduction | -0.006893 | [-0.020457, +0.001823] | no superiority; point favors Shared-Wide |

Every Ref split favors Shared-Wide in point estimate.  RefCOCO TestA,
RefCOCO+ TestA/B, and RefCOCOg intervals exclude zero; RefCOCO TestB crosses
zero.

## Interpretation boundary

This is a strong negative result for universal hard isolation.  On the direct
pre-Stage-B parent, the same mature raw-query head benefits substantially from
sharing even though half of the final probes have negative cosine.  Gradient
sign is therefore not a topology-selection rule.

The comparison to B58 is a useful representation-stage trend but not yet a
strict same-head causal contrast.  This replay uses the 100,362/100,358
parameter raw-query owner pair.  The existing B58 capacity block uses an
83,971/83,969 parameter integrated adapter owner pair.  Both are
capacity-matched internally, but their features and head structures differ.
The paper may state that the observed ownership regime changes across the
parent/B58 representations; it may not attribute that change solely to B58's
mixed Stage-B adaptation without a same-head B58 replay.

## Native pretrained MM-GDINO number

The already sealed MM-GDINO-T pretrained cache gives the previously implicit
native row:

| Route | TestAB | Strict2031 FPR95 |
|---|---:|---:|
| Native MM-GDINO-T pretrained | 0.532738 | 0.922698 |
| Shared-Wide | 0.533823 ± 0.000459 | 0.910061 ± 0.014953 |
| Isolated | 0.533265 ± 0.000698 | 0.916954 ± 0.007649 |

No new MM-GDINO forward was run; these values come from the sealed pretrained
ownership receipt.

## Audit

- Training caches: 3/3 complete; 4,000 rows each.
- Formal trajectories: 6/6 complete.
- Evaluation caches: 6/6 complete; evaluation routes: 42/42 complete.
- Bootstrap: 5,000 paired image-cluster draws, PCG64 seed `20260824`; every
  FPR95 replicate recomputes each owner/seed positive q05.
- Aggregate SHA-256:
  `a9bb3b58f13a813e42870f1ee22f985f1d9aae457d07c08fd663476d8f6a4a80`.
- Result receipt SHA-256:
  `be371aee477efb42ab187f0213ddd545e4d31cac5d7bc4eaa38dbd1066087b9b`.
- No checkpoint, milestone, score reduction, Gap, threshold, sample alias, or
  evaluation surface was selected from the result.
