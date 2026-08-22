# MM-GDINO e6 ownership 2×2 results

## Outcome

The requested 2×2 is complete at the fixed U150 endpoint. Shared-128 was not
run. The original e5 result is reused as a reference without a new forward.

| Frozen trunk | Owner | TestAB P@1 | Strict2031 FPR95 |
|---|---|---:|---:|
| e5 reference | Shared-Wide | 0.889788 ± 0.000093 | 0.836862 ± 0.004440 |
| e5 reference | Isolated | 0.889757 ± 0.000142 | 0.842114 ± 0.019832 |
| e6 PosCtrl | Shared-Wide | 0.889137 ± 0.000093 | 0.830954 ± 0.002327 |
| e6 PosCtrl | Isolated | 0.889106 ± 0.000107 | 0.832759 ± 0.009372 |
| e6 TN10 | Shared-Wide | 0.891307 ± 0.000054 | **0.670934 ± 0.001729** |
| e6 TN10 | Isolated | 0.891276 ± 0.000093 | 0.686526 ± 0.017686 |

Values are three-seed mean ± sample SD for seeds 17/42/73. TestAB is pooled
RefCOCO TestA+TestB micro P@1. Lower FPR95 is better.

The gradient stress worked, but the ownership prediction did not:

| Trunk | Probe horizon | mean cosine | P(cos<0) | q05 | minimum |
|---|---|---:|---:|---:|---:|
| e5 reference | U150, 24 probes | +0.0088 | 0.333 | -0.156 | -0.162 |
| e6 PosCtrl | U150, 24 probes | -0.0453 | 0.583 | -0.182 | -0.242 |
| e6 TN10 | U150, 24 probes | -0.0763 | 0.667 | -0.516 | -0.560 |
| e6 PosCtrl | U25/50/100/150, 96 probes | +0.0103 | 0.469 | -0.248 | -0.354 |
| e6 TN10 | U25/50/100/150, 96 probes | +0.0543 | 0.531 | -0.470 | -0.673 |

TN10 therefore raises the all-milestone negative-cosine probability by 0.0625
and shifts q05 by -0.222. At U150 it raises P(cos<0) by 0.0833 and produces a
much heavier negative tail. This is a real mechanism change, not an endpoint
improvement.

## Planned contrasts

The paired candidate is Isolated and the reference is Shared-Wide. FPR95 gain
is defined as Shared-Wide minus Isolated, so positive values favor Isolated.

| Trunk | Isolated−Shared REC [95% CI] | FPR95 gain [95% CI] | Holm-IUT | Gate |
|---|---:|---:|---:|---:|
| e6 PosCtrl | -0.000031 [-0.000095, 0] | -0.001805 [-0.019228, 0.011178] | 1.0 | fail |
| e6 TN10 | -0.000031 [-0.000095, 0] | **-0.015592 [-0.029252, -0.003038]** | 1.0 | fail |

Both Isolated rows satisfy the 0.005 REC non-inferiority margin, but neither is
REC-superior. TestA is exactly tied; the pooled difference comes from at most
one TestB prediction per seed. On TN10, the rejection interval excludes zero
in the opposite direction: Shared-Wide has lower FPR95 and substantially lower
seed variance.

The cross-trunk difference in the isolation gap is zero for REC. Its FPR95
difference-in-differences is -0.013786 with 95% CI [-0.031543, 0.006962], so it
does not establish a cross-trunk superiority interaction.

## Interpretation boundary

Supported:

- PosCtrl is near zero on average at U150.
- TN10 increases the frequency and lower-tail severity of negative shared
  gradients on fixed probes.
- Both learned owner layouts preserve REC within the preregistered margin.

Not supported:

- Isolated ownership improves REC on TN10.
- A heavier negative cosine tail is sufficient to predict deployment benefit.
- Isolated ownership improves Strict-TN2031 over the capacity-matched shared
  owner on this strong-trunk family.

The correct paper conclusion is stronger than a failed positive result:
gradient conflict is a diagnostic, not a sufficient selection rule. Under the
strong e6 representations, task-specific optimizer states plus a wide shared
representation absorb the observed conflict without REC collapse, while hard
isolation increases rejection variance. This block must not be presented as
evidence that parameter isolation universally improves strong grounders.

## Audit and reproducibility

- Frozen checkpoints:
  - `weights/epoch_6_postctrl.pth`, SHA-256
    `08177fac668d62de99100b292ee5ff157366c33c48eb56b742006263a42022c3`.
  - `weights/epoch_6_tn10.pth`, SHA-256
    `a7078f1139c847d99e85221c8228f7cfd00e5be5ca0b85820f5d4d6a02cfa66c`.
- Formal trajectories: 12/12 complete; U150 only, 100 rank + 50 confidence
  updates, two task-specific Adam states, weight decay zero.
- Evaluation routes: 42/42 complete from six shared frozen-candidate caches.
- Bootstrap: 5,000 paired image-cluster draws, PCG64 seed `20260822`; every
  FPR95 replicate recomputes each model/seed positive q05.
- Aggregate:
  `outputs/mmgdino_e6_ownership_2x2_20260822/aggregate.json`, SHA-256
  `1b8927b3368fe40af055676d3023f4bd64c2d9e56d45999541fb11e129444ed5`.
- Lightweight result receipt:
  `paper/data/mmgdino_e6_ownership_2x2_results.json`, SHA-256
  `7ebaf3bc04e386c7cb82a44974a627dc1a7c7939cc9eec93bde8191eaae954a1`.

One TN10/seed73 scheduled rank row had no eligible IoU≥0.5 candidate. It was
retained without replacement and contributes zero rank-margin loss through the
pre-existing valid-row mask; all other scheduled identities remain unchanged.
The immutable amendment is
`paper/data/mmgdino_e6_ownership_2x2_candidate_availability_amendment.json`.
The two earlier no-metric runtime failures and their narrow amendments are also
bound by the result receipt.
