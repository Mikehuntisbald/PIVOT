# MM-GDINO-T pretrained ownership results

## Outcome

The preregistered pretrained-trunk replay is complete at U150 for
Shared-Wide and Isolated, seeds 17/42/73. The frozen checkpoint has broad
grounding pretraining but no RefCOCO task-specific trunk fine-tuning. The rank
and confidence heads still use the fixed RefCOCO R100 and leakage-clean D3 C50
schedules.

Hard isolation does not win this comparison:

| Route | Test5 P@1 | TestAB P@1 | Strict2031 FPR95 |
|---|---:|---:|---:|
| Native pretrained | 0.564855 | 0.532738 | 0.922698 |
| Shared-Wide | **0.566814 ± 0.000970** | **0.533823 ± 0.000459** | **0.910061 ± 0.014953** |
| Isolated | 0.565867 ± 0.001221 | 0.533265 ± 0.000698 | 0.916954 ± 0.007649 |

Learned-route values are three-seed mean ± sample SD. Test5 contains 30,969
expressions from RefCOCO TestA/B, RefCOCO+ TestA/B, and RefCOCOg UMD test.
TestAB is pooled RefCOCO TestA+TestB micro P@1. Lower FPR95 is better.

## Gradient geometry

For Shared-Wide, the U150 fixed probes give:

- mean cosine: -0.00151;
- `P(cos<0)`: 0.4583 (11/24 probes);
- q05: -0.1691;
- minimum: -0.1792.

Across U25/U50/U100/U150, 96 probes give mean +0.0510,
`P(cos<0)=0.3854`, q05 -0.2064, and minimum -0.2743. The representation is
therefore near-orthogonal on average but has a moderate negative tail.
Isolated has no cross-task autograd path at every probe.

## Planned contrast

The candidate is Isolated and the reference is Shared-Wide. REC gain is
Isolated minus Shared-Wide. FPR95 gain is Shared-Wide minus Isolated, so a
positive value favors isolation.

| Endpoint | Gain | Paired 95% CI | Result |
|---|---:|---:|---|
| Test5 P@1 | -0.000947 | [-0.001288, -0.000615] | Isolated non-inferior, but Shared-Wide is better |
| TestAB P@1 | -0.000558 | [-0.001037, -0.000126] | Isolated non-inferior, but Shared-Wide is better |
| Strict2031 FPR95 | -0.006893 | [-0.023398, +0.003654] | no superiority; point favors Shared-Wide |

The joint gate fails because Isolated does not improve FPR95. Shared-Wide also
has a statistically resolved, though small, pooled REC advantage. The
Strict2031 interval crosses zero, so its rejection advantage remains a point
estimate rather than a confirmed difference.

## Interpretation boundary

Supported:

- before RefCOCO-specific trunk fine-tuning, Shared-Wide gradients are close to
  orthogonal on average but are not conflict-free;
- both owner layouts preserve REC within the preregistered 0.005 margin;
- Shared-Wide gives slightly higher pooled REC than Isolated on this trunk.

Not supported:

- hard isolation improves REC for the pretrained representation;
- hard isolation improves Strict2031 rejection over the capacity-matched
  Shared-Wide owner;
- a negative cosine tail alone predicts that hard isolation will improve
  deployment.

Together with the frozen B58 and strong e5/e6 controls, this favors a
representation-dependent conclusion. Isolation helps the weaker B58 route,
whereas the broad pretrained and stronger task-adapted MM-GDINO
representations tolerate or benefit from wide sharing with task-specific Adam
states. This block should not be used to claim universal superiority of
ownership isolation.

The native pretrained trunk reaches only 53.27% TestAB despite high candidate
oracle coverage; task-specific MM-GDINO trunk fine-tuning is therefore the
main source of the approximately 89% e5 REC level. The R100/C50 heads do not
substitute for that trunk adaptation.

## Audit and reproducibility

- Frozen checkpoint SHA-256:
  `b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a`.
- Formal trajectories: 6/6 complete; U150, R100+C50, two task-specific Adam
  states, weight decay zero.
- Evaluation routes: 42/42 complete from six shared frozen-candidate caches.
- Bootstrap: 5,000 paired image-cluster draws, PCG64 seed `20260823`; each
  Strict2031 replicate recomputes each route/seed positive q05.
- Aggregate:
  `outputs/mmgdino_pretrain_ownership_20260822/aggregate.json`, SHA-256
  `fd4bb5559e69ed7214ccba18bfa0f56a4d982161750bd555efbe9c2163c43a83`.
- Lightweight result receipt:
  `paper/data/mmgdino_pretrain_ownership_results.json`, SHA-256
  `4ff72c188b9df21f8ff22604cf6d5902a165a2cdb1082785a18432f3ee0db514`.

Five of 9,600 scheduled rank rows had no eligible IoU≥0.5 candidate. They
were retained without replacement and contribute zero rank-margin loss through
the pre-existing valid-row mask. No checkpoint, milestone, score threshold,
Gap, loss, sample alias, or evaluation surface was selected from these results.
