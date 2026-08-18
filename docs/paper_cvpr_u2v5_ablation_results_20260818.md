# ARROW CVPR ablation results (U2-v5 lineage, 2026-08-18)

> The public ARROW Admission-input A/B/C follow-up is reported in
> `paper_cvpr_arrow_admission_input_results_20260818.md`.

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

Positive trust does not receive a standalone strict held-out causal claim.
D2/D2m/D3m are excluded from the manuscript because D2 is not D3's direct
pre-VLM source population.

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

D1 is retained only as a synthetic/unverified weak-data stress control. D3 is
the real SAM3+VLM verified main-data source. The historical rule-swap D2 and
the approximate D2m/D3m parent match are not shown or interpreted.

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
- Manuscript-facing paper table v3 SHA256:
  `b5ca62d73075feb7760c79eac8d1d86fd3b4b487a2092424df7ceff2fc2be14f`
- Final receipt v3 SHA256:
  `329bf8eae4f2505d73ac5164d3f89a592a7ec8af38a9c69044828a215278ffc8`
- Complete zero-training M/P/S/G/H-core supplement v3 SHA256:
  `a4a6d43fd705bbabaf578595fed46a8af632676e1c41cd0a92981bb009a43b1c`
- P/S per-record receipt SHA256:
  `c459bd3323fff5b263bb2210d7c6af20e64491b18be8a54d642e25c6ea59560c`

All paths are rooted at `outputs/u2v5_cvpr_ablation_20260817/`. Weight files
remain outside Git. Earlier paper-table/final-receipt versions are superseded
by `_v3`: v2 fixed the O2−O0 family p-value, while v3 additionally excludes
D2/D2m/D3m from the manuscript view. No bootstrap draw or model result changed.

C100 and legacy U2 remain gray diagnostic references and are excluded from all
formal hypotheses.

## P/S zero-training supplement

P and S were run on val3 only with the sealed seed42 checkpoint after the
confirmatory block. They are exploratory and cannot modify the preregistered
model or hypotheses.

| Row | val3 micro Acc@0.5 | Oracle R@0.5 | Eligible R@0.5 | Mean eligible | Mask Hamming | Top-1 churn |
|---|---:|---:|---:|---:|---:|---:|
| P0/S0 full/full/bound | 0.724555 | 1.000000 | 0.996904 | 69.52 | 0.00 | 0.0000 |
| P1 canonical query, full rank | 0.401691 | 1.000000 | 0.994148 | 42.03 | 82.15 | 0.1043 |
| P2 object query, full rank | 0.305384 | 0.999434 | 0.994564 | 91.12 | 117.46 | 0.3044 |
| P3 full query, canonical rank | 0.617223 | 1.000000 | 0.996904 | 69.52 | 0.00 | 0.3785 |
| S1 alternate same-category | 0.724404 | 1.000000 | 0.997055 | 69.55 | 44.52 | 0.0064 |
| S2 same-category shuffle | 0.724441 | 1.000000 | 0.997131 | 69.97 | 43.15 | 0.0058 |
| S3 wrong-category | 0.713002 | 1.000000 | 0.988221 | 97.06 | 87.24 | 0.0227 |
| S4 zero patch | 0.722214 | 1.000000 | 0.998188 | 249.79 | 244.74 | 0.0127 |

P1/P2 show that full-expression query semantics cannot be replaced by a noun
or `object`, even though an adequate all-query box usually still exists. P3
keeps boxes and eligibility fixed but changes top-1 for 37.85% of expressions,
directly establishing the role of full-expression R100 ranking.

S1 changes the support tensor for 99.21% of expressions and S2 for 97.05% while
leaving accuracy effectively unchanged, showing same-category instance
robustness. S3 has valid wrong-category substitutions for 88.73% of expressions
and lowers accuracy by 1.16 points while doubling mask churn. S4 is especially
informative: a zero patch expands the eligible set from 69.5 to 249.8 queries,
yet R100 recovers most final accuracy. Therefore the evidence supports
"support patch controls category admission/candidate compression," not the
stronger claim that patch pixels are strictly necessary for every correct
top-1 prediction.

## H sensitivity

The single-forward seed42 val3 gap sweep gives:

| Gap | val3 micro Acc@0.5 |
|---:|---:|
| 0 | 0.649539 |
| 0.5 | 0.693257 |
| 1 | 0.715154 |
| 2 | 0.725649 |
| 3 | 0.724555 |
| 5 | 0.709831 |
| 10 / ∞ | 0.695107 |

Gap2 is 0.001095 above the frozen Gap3 anchor on this exploratory seed42 val
surface. The held-out block has already been opened, so this cannot trigger a
main-model change or a second Test5/strict evaluation. It is reported only as
sensitivity and a future-version hypothesis. The curve nevertheless confirms
that both an overly narrow gate and no effective gate are materially worse.

Confidence milestone sensitivity is sealed for U25/U50/U100; U50 was selected
by the preregistered worst-seed/mean/earlier-update rule. Queue128/1024,
candidate-count and support-bank-size sweeps were not added after held-out
evaluation because they would require new trained checkpoints or new design
choices outside the frozen 42-trajectory registry. They remain optional future
exploration rather than missing evidence for the claims tested here.
