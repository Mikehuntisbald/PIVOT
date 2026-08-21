# ARROW-U2 / B58 Shared-Wide capacity control (2026-08-21)

## Question and evidence status

This post-release experiment asks whether the original ARROW-U2 ownership
result can be explained by extra isolated-head capacity, shared Adam moments,
or repeated AdamW decay.  It is prospectively frozen relative to this new
control, but Test5 and Strict-TN2031 were already observed by the project; it is
not a virgin held-out experiment.

The design and endpoint locks are tracked in:

- `paper/data/b58_capacity_control_preregistration.json` (`3deb250`);
- `paper/data/b58_capacity_control_evaluation_preregistration.json`
  (`9517788`);
- the pre-forward loader amendment
  `paper/data/b58_capacity_control_evaluation_amendment_v2.json` (`f972675`).

## Matched arms

| arm | score-owner parameters | MAC/query, both outputs | representation |
|---|---:|---:|---:|
| Shared-Wide | 83,971 | 83,007 | shared 163-d; gate hidden 62 |
| Isolated replay | 83,969 | 82,944 | separate 128-d owners |

Shared-Wide has two more parameters and 0.076% more MACs, while either duty can
access a wider representation.  Both arms use two independent task-specific
Adam states, zero weight decay, the same clean initializer/data/order, seeds
17/42/73, 100 Admission plus 50 Rejection updates, fixed U150, and no
checkpoint selection.  The six runs completed 150/150 updates with zero AMP
skips/non-finite boundaries and bitwise-stable frozen hashes.

Shared-Wide showed negative shared-gradient cosine pairs in every seed:

| seed | mean cosine | minimum cosine | negative fraction | sign conflict |
|---:|---:|---:|---:|---:|
| 17 | +0.2005 | -0.3931 | 0.1554 | 0.3508 |
| 42 | +0.1454 | -0.6848 | 0.2500 | 0.3572 |
| 73 | +0.2031 | -0.6443 | 0.1892 | 0.3485 |

The isolated replay passed 450/450 structural checks with zero cross-task
autograd paths.

## Formal results

All point estimates, source-record hashes, capacity/runtime audits, and 5,000
bootstrap replicates are bound by
`paper/data/b58_capacity_control_results.json`, SHA-256
`c916a087ca731f218bdf1ef2b5ad905ce4fb6d1a37753fc4ce6842634070d829`.

| arm | seed | Test5 micro | Strict2031 FPR95 | AUROC | AUPR |
|---|---:|---:|---:|---:|---:|
| Shared-Wide | 17 | 0.687203 | 0.480059 | 0.845511 | 0.829106 |
| Shared-Wide | 42 | 0.698182 | 0.481536 | 0.844207 | 0.828161 |
| Shared-Wide | 73 | 0.684620 | 0.477597 | 0.842677 | 0.827149 |
| Isolated replay | 17 | 0.742904 | 0.469719 | 0.850586 | 0.834191 |
| Isolated replay | 42 | 0.743485 | 0.474151 | 0.847002 | 0.829931 |
| Isolated replay | 73 | 0.742936 | 0.474643 | 0.844693 | 0.828375 |

The paired image-cluster bootstrap uses PCG64 seed 20260821 and the same image
draw across both arms and all three seeds.  Training seeds are equal-weighted,
not resampled.  Every Strict replicate recomputes each arm/seed's positive-q05
threshold.

- Test5 gain, Isolated minus Shared-Wide: **+0.053107**, 95% CI
  **[+0.048754, +0.057675]**.  Every split is positive; the smallest split CI
  lower bound is +0.039649.  REC non-inferiority passes decisively.
- Strict2031 FPR95 reduction, Shared-Wide minus Isolated: **+0.006893**,
  95% CI **[-0.003817, +0.013373]**, one-sided `p=0.13917`.  Rejection
  superiority does not pass.
- The preregistered intersection gate therefore has `joint_pass=false`.

## Interpretation

The control rules out score-owner capacity, accessible representation width,
task-specific Adam moments, and weight decay as explanations for the large REC
gap.  Sharing rank/rejection optimization can damage Ranking even when the
shared arm is slightly larger and wider.  The same experiment does **not**
establish a statistically reliable Strict-TN2031 rejection gain for isolation;
the FPR95 point estimates favor isolation but their paired interval crosses
zero.  Paper wording is therefore restricted to ranking protection, while the
original ARROW-U2 and strong-e5 blocks retain their separately supported claim
boundaries.

## Failed-attempt ledger

The first Shared-Wide Test5 launch failed before any model forward because the
generic loader applied a legacy U2-v2 provenance validator before the new
capacity validator.  The log is retained at
`outputs/b58_capacity_control_20260821/logs/eval_shared_wide_test5.log`.  The
only v2 change routes model construction past that legacy check; a CPU full
checkpoint preflight passed, and all checkpoints, data, Gap, metrics, and
statistics remained unchanged.  During Shared-Wide Strict seed73 loading, an
unnecessary HuggingFace HEAD request briefly retried and then used the local
cache; later commands explicitly set offline mode.
