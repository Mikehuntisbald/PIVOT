# U2-v5 CVPR ablation results (2026-08-18)

## Paper-level conclusion

The clean U2-v5 anchor remains the paper main model. The completed block
supports two proposed mechanism claims and narrows the third:

1. **Supported:** the non-deployed admission auxiliary residual is an effective
   training carrier. Removing it lowers Test5 micro Acc@0.5 by 0.006964.
2. **Supported:** isolating admission/rank from absolute confidence removes a
   real gradient-conflict failure. O2 retains O0's Test5 result while reducing
   strict2031 FPR95 by 0.028065.
3. **Narrowed:** parameter isolation is necessary; phased scheduling is not
   proven better than already-isolated interleaving. O3 passes Ref
   non-inferiority against O2, but its strict improvement is not significant.

Positive trust and matched verification scope are useful calibration-design
diagnostics, but neither receives a standalone strict held-out causal claim.

## Confirmatory endpoints

All new confirmatory forwards were launched exactly once after preregistration.
The sealed U2-v5 anchor was reused without another forward.

### Main anchor context

Against B58, the leakage-clean anchor has already sealed:

| Endpoint | Gain | 95% paired image-cluster CI |
|---|---:|---:|
| Test5 micro Acc@0.5 | +0.020773 | [0.018267, 0.023470] |
| strict2031 FPR95 reduction | +0.041851 | [0.024571, 0.061989] |

The anchor Test5 mean is 0.742398 across seeds 17/42/73; strict2031 FPR95 mean
is 0.470212. Ref8 remains descriptive because its val3 portion was used during
selection.

### Preregistered contrasts

All intervals use 5,000 paired image-cluster bootstrap replicates with PCG64
seed 20260719. The same image draw is applied to candidate, reference and all
three training seeds. FPR95 recomputes each side's positive q05 threshold in
every replicate.

| Contrast | Endpoint | Gain (positive is better) | 95% CI | raw one-sided p | Result |
|---|---|---:|---:|---:|---|
| A5−A1 | Test5 | +0.006964 | [0.005851, 0.008199] | 0.000200 | passes |
| C3−C2 | strict2031 | −0.000492 | [−0.009500, 0.005796] | 0.596481 | does not pass |
| O2−O0 | Test5 | −0.000022 | [−0.000716, 0.000664] | descriptive | route preserved |
| O2−O0 | strict2031 | +0.028065 | [0.009268, 0.046384] | 0.001000 | passes |
| O3−O2 | Test5 | −0.000592 | [−0.000954, −0.000216] | NI p=0.000200 at margin 0.005 | NI passes |
| O3−O2 | strict2031 | +0.002790 | [−0.003178, 0.010905] | 0.135973 | superiority does not pass |
| D3m−D2m | strict2031 | −0.000985 | [−0.007429, 0.006826] | 0.543091 | does not pass |

The ownership-schedule IUT therefore does not pass. Family-wise Holm correction
is applied within admission, confidence/data and ownership. For O2−O0, Test5
is a route-preservation effect/CI; its family hypothesis is strict superiority,
not Test5 superiority.

strict1607 is derived from the same strict2031 records. It is nested robustness,
not an independent replication. O2−O0 remains positive there (+0.027588,
95% CI [0.005664, 0.046712]).

## Mechanism interpretation

### Admission residual

A1 trains only the deployed surface. A5 trains the surface plus the auxiliary
residual, but the residual is absent from deployment scoring. The significant
A5−A1 Test5 gain therefore identifies a training-gradient effect rather than an
extra inference score. A2 independently verifies that auxiliary-only training
leaves all 1,157 deployed tensors bitwise equal to the initializer.

Removing preserve loss (A3) does not reduce val3 accuracy. It should be
described as a stability regularizer, not an accuracy contributor. Removing
category-complete supervision (A4) particularly weakens RefCOCOg/coverage, but
the A3/A4 mechanism grid remains val-only.

### Gradient ownership

O0 and O1 retain Ref accuracy, so gradient conflict is not diagnosed by Ref
collapse. It appears in absolute confidence:

| Row | Trainable ownership | val3 macro | calibration FPR95 |
|---|---|---:|---:|
| O0 | shared score | 0.739437 | 0.573248 |
| O1 | shared feature, separate outputs | 0.738681 | 0.572187 |
| O2 | isolated parameters, interleaved | 0.738932 | 0.533758 |
| O3 | isolated parameters, phased | 0.738851 | 0.536518 |

O0's admission/confidence gradient cosine is negative for every seed
(−0.1900, −0.2523, −0.2810), with sign-conflict fractions around 0.53–0.56.
O2 has zero cross-task autograd connections. The held-out O2−O0 result turns
that structural observation into endpoint evidence.

### Confidence objective and data scope

C2 is not worse than C3 on strict2031, so the paper must not claim that the
positive-trust term independently improves FPR95. C1 shows that current-batch
negatives are insufficient, while C4 gives no paired-margin benefit.

D3m strongly improves its matched calibration surface over D2m (0.344589 vs
0.443723), but the benefit does not transfer significantly to the full strict
universe. This is targeted calibration behavior, not a general held-out causal
claim. Broad D1/D2/D3 numbers use different universes and must not be compared
causally.

## Cumulative routes and error attribution

The zero-training supplement binds these val3 micro routes:

| Route | Acc@0.5 |
|---|---:|
| M0 B58 | 0.695032 |
| M1 positive-only R100 | 0.695107 |
| M2 static admission + R100 | 0.702507 (sealed seed17 parity route) |
| M3 trained admission + identity confidence | 0.723611 ± 0.001152 |
| M4 full U2-v5 | 0.723611 ± 0.001152 |

M3 and M4 have the same Ref route by construction: confidence is an independent
output and cannot alter boxes, eligibility or Ref top-1.

On 26,488 val3 expressions per seed, all-query geometry supplies an IoU≥0.5
candidate for every expression. Admission removes the last correct candidate
in only 0.31–0.38% of expressions; 27.23–27.39% remain rank errors inside the
eligible set. The next model-improvement target is eligible-set ranking, not a
wider admission gate.

## Artifacts and provenance

- Preregistration SHA256:
  `6eaa844c87b4a116dfaf13dc2c9fa1cbced030f2b9438197a8579f5c410916a4`
- Confirmatory results manifest SHA256:
  `ffb99ad3d77a0ec02b2e8cf5710a577b18a2f05f4fc354d2cd2f9093da90bce7`
- Corrected paper table v2 SHA256:
  `9db98af1f0fff1c91372a81af43b12627974105863302b42abe59b4675d9bb0f`
- Final receipt v2 SHA256:
  `98173c28258a25c450530f6929c0a719228c52e401851d37307a33a4d01e8c3d`
- Zero-training M/G/H supplement SHA256:
  `09cd4c1c4f7430f51c328f53876669973c791315ecf6ba50cbed0896d6ccdda3`

All paths are rooted at `outputs/u2v5_cvpr_ablation_20260817/`. Weight files
remain outside Git. `paper_tables.json` and `final_receipt.json` are superseded
by `_v2` because the first renderer incorrectly included Test5 superiority in
the O2−O0 family p-value. No bootstrap draw or model result changed.

C100 and legacy U2 remain gray diagnostic references and are excluded from all
formal hypotheses.

## Remaining supplemental forwards

P (full/canonical/object geometry × ranking-text swap) and S (support
counterfactuals) require additional zero-training GPU forwards. They were not
used for selection or any claim above and are explicitly marked pending in
`zero_training_supplement.json`. Running them later is safe as exploratory
supplementary analysis, but their results must not modify the preregistered
model or hypotheses.
