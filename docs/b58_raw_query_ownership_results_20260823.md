# B58 100k raw-query ownership replay: sealed results

## Outcome

The strict same-head replay is complete: Shared-Wide and Isolated use the same
100k raw-query owner implementations on the direct parent and B58, with
identical initialization, schedules, losses, seeds, optimizers, native score,
and evaluator.  Only the frozen 938-tensor trunk weights change (727 changed,
211 bitwise unchanged).

B58 removes the large localization penalty of hard isolation seen on the
direct parent, but it does not make isolation superior.  Its FPR95 point
estimate is lower, while the paired interval crosses zero.

## B58 fixed-U150 endpoints

Percentages below are three-seed mean ± sample SD for learned routes.

| Route | Test5 P@1 | TestAB P@1 | Strict2031 FPR95 ↓ | AUROC ↑ |
|---|---:|---:|---:|---:|
| Native B58 | 72.046 | 65.690 | 50.222 | 83.067 |
| Shared-Wide | **72.169 ± 0.031** | **65.823 ± 0.046** | 46.939 ± 0.271 | **84.050 ± 0.680** |
| Isolated | 72.116 ± 0.021 | 65.802 ± 0.009 | **46.414 ± 0.571** | 83.449 ± 1.948 |

Paired Isolated minus Shared-Wide:

- Test5: −0.0527 pp, 95% CI [−0.0951, −0.0109]; isolation is not superior but
  passes the preregistered −0.5 pp non-inferiority margin.
- TestAB: −0.0217 pp, 95% CI [−0.1037, +0.0536]; non-inferior.
- FPR95 reduction (`Shared − Isolated`): +0.525 pp, 95% CI
  [−0.444, +1.100], one-sided p=0.175; not superior.
- The intersection-union gate fails because FPR95 superiority is unresolved.

Per-split no-collapse holds.  Isolation is significantly lower on RefCOCO+
TestA and TestB, but both gaps are below 0.16 pp and remain well inside the
registered non-inferiority guard.

## Strict same-head parent → B58 axis

The primary causal statistic is
`(Isolated − Shared)_B58 − (Isolated − Shared)_parent`, using the same image
draw across both trunks, both owners, and all seeds.

| Endpoint | Parent ownership effect | B58 ownership effect | Difference-in-differences [95% CI] |
|---|---:|---:|---:|
| Test5 | −2.001 pp | −0.0527 pp | **+1.948 pp [1.686, 2.208]** |
| TestAB | −1.116 pp | −0.0217 pp | **+1.094 pp [0.718, 1.462]** |
| FPR95 reduction | −0.689 pp | +0.525 pp | +1.215 pp [−0.113, 2.631] |

Thus mixed Stage-B adaptation significantly changes topology sensitivity for
localization: the same isolated owner moves from a large REC penalty to
practical parity.  The analogous rejection shift is directionally favorable
but not statistically resolved.

Native parent → B58 gains under the same evaluator are +32.513 Test5 points,
+25.102 TestAB points, and a 39.439-point FPR95 reduction.  Candidate
availability also changes: the three fixed training schedules contain 40
no-positive rank rows on the parent versus one on B58; all five B58 Test5
surfaces have 100% all-query oracle recall.  This is part of the frozen-trunk
treatment, not a head-protocol difference.

## Gradient geometry

For B58 Shared-Wide:

- U150: mean cosine −0.0335, `P(cos<0)=58.3%`, q05 −0.1745, minimum −0.3747;
- all U25/U50/U100/U150 probes: mean +0.0306,
  `P(cos<0)=47.9%`, q05 −0.4471, minimum −0.6234;
- seed73 U150 has 75% negative probes and mean −0.1079.

Isolated has no cross-task autograd path.  Despite the recurrent and sometimes
deep negative tail, hard isolation does not improve REC and its FPR95 advantage
does not exclude zero.  Gradient conflict is therefore a representation-state
diagnostic, not a sufficient topology-selection rule.

## Evaluation boundary

- Fixed U150 only; seeds 17/42/73 all completed.
- Test5 has 30,969 expressions; TestAB has 10,752.
- Strict2031 uses the sealed 2,031 pairs.
- 5,000 paired global image-cluster bootstrap replicates, PCG64 seed 20260825.
- Every FPR95 replicate recomputes each trunk/owner/seed positive q05.
- No checkpoint, Gap, score reduction, or threshold was selected from results.
- The matched-block Native B58 FPR95 is 50.222 rather than the historical
  51.206 figure.  The causal axis uses the newly re-extracted parent and B58
  records under one exact-q05 evaluator; the historical number is not mixed
  into this contrast.

## Artifact bindings

- Preregistration: `paper/data/b58_raw_query_ownership_preregistration.json`,
  SHA-256 `2c89d7f9a8909a1dc81eec4f6e75315f282bf4f3538a22595120805f500e8a2a`.
- Aggregate: `outputs/b58_raw_query_ownership_20260822/aggregate.json`,
  SHA-256 `46958630937a123ca6363c07f4cc7f49dfdd761c5bb2390dd034e52f01cbd768`.
- Result receipt: `paper/data/b58_raw_query_ownership_results.json`,
  SHA-256 `fc7b03d66765b43782ab11e9f4aa3baec0f74ae5963da7fe699ddb1255e93105`.
