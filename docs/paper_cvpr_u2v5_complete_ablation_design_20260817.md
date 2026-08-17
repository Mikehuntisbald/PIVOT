# U2-v5 CVPR complete ablation design

Date: 2026-08-17

Status: design contract; no new ablation result is authorized until the
configs, datasets, runner, selection rules, and final-evaluation manifest are
implemented and sealed in one clean commit.

For U2-v5, this document supersedes `paper_cvpr_ablation_protocol.md` as the
design authority. The older document remains an immutable record of the
checkpoint0004/Top-50/v19-v24 experiment family.

## 1. Paper decision and claim boundary

The paper main model is the already sealed leakage-clean U2-v5 anchor:

```text
full expression -> frozen B58 trunk -> frozen positive-only R100 rank
support patch   -> Stage-A patch surface -> trained Gap3 category admission
detached B58 statistics -> fresh image-disjoint D3 confidence
```

The ablation must support three narrow claims:

1. **Duty factorization.** The support patch supplies category admission,
   full-expression R100 supplies within-admission Ref ordering, and the
   confidence head supplies an absolute rejection score.
2. **Isolated optimization.** The admission surface and absolute confidence
   head do not share trainable parameters. The auxiliary admission residual is
   a training-only gradient carrier, not a deployed fourth score.
3. **Leakage-clean rejection.** Fresh proposal-covered D3 supervision improves
   both strict FPR95 surfaces over B58 without importing C100 confidence12.

The paper must not claim that D3 is image-global or all-900-query verified.
It must not claim that U2-v5 beats diagnostic C100: the sealed U2-v5 result
fails the predeclared C100 `+0.01` non-inferiority margin on strict2031. The
old edit-token v19/v21 system is not part of U2-v5 and therefore cannot be
presented as an ablation of the main model.

Because val3 selected the admission checkpoint, **Test5 micro** (the five
official Ref test splits, 30,969 expressions from 3,982 unique images) is the
confirmatory Ref endpoint.
Ref8 contains the three selection splits and is reported descriptively. The
sealed Test5 micro values are `0.741419/0.743195/0.742581` for seeds
17/42/73, versus B58 `0.721625`. Their mean is approximately `0.742398`;
it should not be described as better than legacy U2 without a predeclared
paired analysis.

## 2. Immutable U2-v5 comparison contract

All controlled rows use:

- Stage-A checkpoint0007, SHA-256
  `fe20fe91f3c46b6d143db13c74817ff3aa810cc51d1579104913c3d23fec9a8b`;
- positive-only R100 U100, SHA-256
  `346e847228f7a14a70ee772233c8d5fb2b090aebab76d7deda981901e74cc2b7`;
- clean initializer, SHA-256
  `ad7b3a563ef84356c6d952167ee6a48f615f8db887eba31bed92a81b0ba756a7`;
- the full referring expression in the original GDINO text branch unless the
  row is explicitly in the prompt-routing block;
- no B58 top-1 guard, no horizontal flip, deterministic 800/max1333;
- admission data source weights `2:2:1`, physical B56, U100, LR `3e-4`,
  clip `0.1`, AMP scale 8192;
- confidence D3 train/calibration audit SHA-256
  `7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d`;
- confidence physical B8, milestones U25/U50/U100, LR `3e-4`, clip `0.1`,
  queue size 512 and positive-trust weight 1.0 except when directly ablated;
- training seeds 17/42/73 and evaluation seed 42;
- B16/W4/AMP evaluation with complete per-example records;
- strict1607 manifest SHA-256
  `f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25`;
- strict2031 manifest SHA-256
  `0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918`.

The current gate operates on the exact valid-query universe emitted by the
model, not the old paper protocol's fixed Stage-A Top-50 surface. Every new
receipt must record the per-example valid and eligible query counts. A Top-50
row is a separate candidate-count diagnostic, not a U2-v5 control.

No row may name or load C100 confidence12, except the explicitly shaded
historical diagnostic row. A checkpoint is invalid if its serialized config
contains a non-null C100 path/SHA, if a frozen tensor changes, or if a loss has
an autograd connection to an undeclared owner.

## 3. What can and cannot be reused

| Evidence | Status | Paper use |
|---|---|---|
| B58 Ref8/strict records | reusable sealed baseline | Main comparison |
| U2-v5 seeds 17/42/73 | reusable sealed main row | Main comparison |
| legacy U2/P50 | historical, non-clean | Context row, never clean control |
| C100 | overlaps strict2031 by 67 images | Shaded diagnostic upper bound only |
| U2-v2 post-gate residual | seed42/C100 diagnostic | Negative mechanism appendix |
| U2-v4 seed42 admission | Ref route equals clean seed42 | Provenance cross-check only |
| old Table A-D and L0-L10 | different initializer, Top-50, heads/objectives | Historical appendix only |

The old L0-L10 checkpoints are useful evidence about an earlier edit-token
architecture, but token supervision is absent from U2-v5. Mixing their numbers
into a U2-v5 component table would be an invalid comparison.

Reusable clean inputs include:

- admission positives and category-complete boxes in
  `data/ablations/stageb_refexp_three_train_category_complete_20260720/receipt.json`
  (120,624/120,191/80,512 RefCOCO/+/g rows);
- D1/D2/D3 source audit in
  `data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json`;
  all three source blocks are image-disjoint from the strict union;
- D3 train/calibration with 14,196/1,570 rows;
- synchronized category interventions in
  `data/ablations/stageb_table_a_category_intervention_20260717/audit.json`
  (512 pairs, 1,024 arms, 318 categories). These are built inputs, not existing
  model results.

## 4. Main-paper tables

### Table M: cumulative main-model decomposition

This is the first and most important ablation table. All routes should be
emitted from the same forward wherever possible.

| ID | Category admission | Ref rank | Confidence | Training | Purpose |
|---|---|---|---|---|---|
| M0 | none | B58 base | B58 base | existing | Data-FT baseline |
| M1 | none | positive-only R100 | identity confidence | zero | Rank contribution |
| M2 | frozen Stage-A Gap3 | positive-only R100 | identity confidence | zero | Static patch prior |
| M3 | trained U2-v5 Gap3 | positive-only R100 | identity confidence | admission only | Admission contribution |
| M4 | trained U2-v5 Gap3 | positive-only R100 | fresh D3 U50 | existing main | Full model |
| M5 | legacy U2 Gap3 | legacy R100 | P50 | historical | Non-clean historical context |

Report Test5 micro as primary, Ref8 descriptively, per-split Acc@0.5,
strict1607/2031 FPR95, positive q05,
eligible GT Recall@0.5, mean eligible-query count, and trainable parameters.
M1-M4 share the same clean initializer lineage. M3 and M4 must have bitwise
identical Ref records; M1-M3 must have bitwise identical confidence records.
Those parity checks replace unnecessary repeated evaluation.

Primary Ref contrasts use Test5 micro. They are M3-M1 for category admission,
M3-M2 for admission
training, and M4-M3 for fresh confidence. M5 is not included in clean-block
hypothesis tests.

### Table A: admission training mechanism

All rows start from the clean initializer, train for fixed U100, and evaluate
Ref only. Confidence12 remains identity and must be bitwise unchanged.

| ID | Surface8 | Auxiliary residual8 | Category-complete loss | Preserve loss | Deployment |
|---|---|---|---|---|---|
| A0 | frozen | frozen | off | off | static Gap3 + R100 |
| A1 | train | frozen | on | on | trained Gap3 + R100 |
| A2 | frozen | train | on | on | static Gap3 + R100 |
| A3 | train | train | on | off | trained Gap3 + R100 |
| A4 | train | train | off | on | trained Gap3 + R100 |
| A5 | train | train | on | on | main admission |

A5-A1 isolates the training-only auxiliary residual. A2 must be deployment
bitwise-equal to A0 because the auxiliary residual is not deployed; it is a
negative control proving that any A5 gain comes through the trained surface.
A5-A3 measures target preservation and A5-A4 measures category-complete
supervision. The mechanism table reports val3, eligible recall/size,
wrong-to-correct and correct-to-wrong transitions versus A0, gradient norms
for both owners, and the frozen-state hash. Only the preregistered A5-A1
contrast receives a confirmatory Test5 pass.

The main paper may show A0/A1/A5; the full A0-A5 table belongs in the
supplement.

### Table C: leakage-clean confidence objective

Every trained row starts from its seed-matched A5 U100 admission checkpoint.
Only confidence12 is trainable; all 1,153 other tensors must remain bitwise
unchanged. These rows use D3 calibration for the mechanism table because their
Ref route must match A5 bitwise. Only preregistered confidence contrasts receive
one strict2031 pass, from which strict1607 is derived.

| ID | Negative FPR surrogate | Recent-q05 queue | Positive trust | Paired margin | Status |
|---|---:|---:|---:|---:|---|
| C0 | 0 | 0 | 0 | 0 | identity/no confidence training |
| C1 | 1 | 0 | 0 | 0 | current-batch negative term |
| C2 | 1 | 512 | 0 | 0 | queue contribution |
| C3 | 1 | 512 | 1 | 0 | main clean confidence |
| C4 | 1 | 512 | 1 | 1 | paired-margin control |
| C5 | different legacy objective | legacy | legacy | legacy | shaded C100 diagnostic |

C2-C1 isolates the queue, C3-C2 isolates positive trust, and C4-C3 tests the
omitted paired term. All confidence ablations use the anchor-selected U50;
they do not select a separate best milestone per row. U25/U50/U100 remains a
descriptive sensitivity plot. Report strict2031 FPR95 as primary, nested
strict1607 FPR95 as robustness, positive q01/q05, AUROC,
pair-win rate, threshold, TN/positive score means, and score-scale quantiles.
C5 is never pooled into a clean confidence comparison.

### Table O: ownership and gradient conflict

This block uses exactly 100 admission exposures and 50 confidence exposures
per seed. For joint schedules, admission is active on every step and
confidence on every second step; for the main sequential row, all 100
admission steps precede all 50 confidence steps. Thus data exposure, not only
the nominal step counter, is matched.

| ID | Ref/confidence parameters | Schedule | Required diagnostic |
|---|---|---|---|
| O0 | one shared R100-derived scalar | interleaved | gradient cosine/conflict on shared rank8 |
| O1 | separate outputs, shared trainable R100 feature trunk | interleaved | gradient cosine/conflict on shared trunk |
| O2 | frozen R100 + independent surface/confidence12 | interleaved | structural zero cross-gradient |
| O3 | frozen R100 + independent surface/confidence12 | sequential | main ownership |

O1-O0 tests whether separate outputs are sufficient. O2-O1 tests parameter
isolation. O3-O2 tests scheduling after isolation. O0/O1 must report
rank/admission-versus-confidence cosine, negative-cosine step fraction,
elementwise sign-conflict fraction, and each task's regression immediately
after the other task's update. O2/O3 must fail if an inactive owner has even a
structural autograd connection, not merely a numerically zero gradient.

All four rows report val3, D3 calibration, and gradient diagnostics. O0/O2/O3
are the compact main-paper rows; the preregistered O2-O0 and O3-O2 contrasts
receive confirmatory Test5/strict2031 evaluation. This is the only new block
allowed to evaluate both duties, because ownership can change both.

## 5. Supplementary controlled blocks

### Block D: confidence data provenance

Keep the C3 architecture/objective fixed.

| ID | TN source | Verification scope | Comparison role |
|---|---|---|---|
| D0 | none | positive-only | no-TN control |
| D1 | unverified edited negatives | edit text only | weak-data control |
| D2 | traceable counterfactual edits | traceable edit | broad equal-exposure row |
| D3 | clean proposal-covered pairs | proposal-covered | main data |
| D2m | matched traceable parents | traceable edit | causal matched block |
| D3m | same matched parents | proposal-covered | causal matched block |

D1-D3 use identical TN exposure and total updates. D0 is a separate no-TN
control. D2m/D3m use the existing 7,074-row matched audit as their own block
and must not be pooled with the broad rows. Report identical-TN and changed-TN
strata separately. No row may upgrade proposal-covered labels to image-global.
The broad rows require new paired-only U2-v5 confidence manifests; the old
Table-B mixture of three positive sources plus TN mass is not the final
confidence phase and cannot be reused as a config.

### Block P: text-route semantics, zero training

| ID | Query/box text | R100 ranking text | Purpose |
|---|---|---|---|
| P0 | full expression | full expression | main route |
| P1 | canonical noun | full expression | canonical query-universe cost |
| P2 | `object` | full expression | generic query-universe cost |
| P3 | full expression | canonical noun | remove attribute/relation ranking |

P1/P2 intentionally change query geometry; report candidate oracle recall and
box churn so the result is not misdescribed as a pure score ablation. P3 keeps
geometry and removes full-expression rank semantics. Break down Ref accuracy
by color, size, action, spatial, and relation language.

### Block S: support-patch causal checks, zero training

| ID | Support patch | Expected interpretation |
|---|---|---|
| S0 | bound support patch | main route |
| S1 | alternate same-category support | robustness to instance choice |
| S2 | batch-shuffled same-category support | category versus instance evidence |
| S3 | wrong-category support | causal category sensitivity |
| S4 | zero/no patch | necessity control |

Keep image and text fixed. Report eligibility-mask Hamming distance, eligible
recall/size, top-1 churn, and Ref accuracy. S3 is not an adversarial robustness
claim; it is a causal responsibility check.

### Block H: deployment and hyperparameter sensitivity

Use val/calibration only; never rerun Ref test or strict for every setting.

- category gate gap: `0, .5, 1, 2, 3, 5, 10, infinity`;
- admission milestones: U25/U50/U100;
- confidence milestones: U25/U50/U100;
- confidence queue: `0, 128, 512, 1024`;
- candidate-count diagnostic: Top-10/25/50/100/all-valid queries;
- support-bank size and alternate-support variance;
- latency, peak VRAM, total/trainable parameters, and throughput.

The efficiency receipt should verify, rather than merely copy, the current
reference counts: about 172.651M total parameters, 268,167 active admission
parameters, and 83,969 active confidence parameters. Report admission and
confidence memory separately; the existing observed peak reserved memory is
approximately 30.39/12.26 GB.

Gap plots must show both accuracy and eligible GT recall/mean set size. A gap
is not selected from test. Candidate-count rows are diagnostics because they
change the deployed candidate domain.

### Block G: geometry and error ownership, zero training

Re-evaluate Stage-A checkpoint0007 under the current query universe. Report
all-query oracle Recall@1/5/10/50/all, mean best IoU, patch category AP/Acc,
eligible recall, and bitwise box parity across B58/R100/U2-v5 routes. Partition
every Ref failure into: no adequate query geometry, adequate query rejected by
admission, adequate eligible query mis-ranked, or correct Ref with low absolute
confidence. The older checkpoint0004 caliber report cannot fill this row.

### Historical mechanism appendix

The following may be summarized as historical negative evidence but never
merged numerically with U2-v5 controlled rows:

- U2-v2 learned post-gate residual: consistent but much weaker Ref gain;
- V55-V62/C1-C2 confidence mechanisms: capacity or optimization interventions
  that did not beat their registered confidence gates;
- old L0-L10 edit-token matrix: a different v19/v21 model family;
- legacy U2/P50 and C100: non-clean confidence provenance.

## 6. Evaluation factorization

To reduce cost, avoid post-anchor test fishing, and strengthen causal
attribution, every block has a mechanism profile and an optional confirmatory
profile:

- Ref-only rows (A, P, S) report val3 in the mechanism table; their confidence
  records must match their declared parent checkpoint bitwise.
- confidence-only rows (C, D) report D3 calibration in the mechanism table;
  their Ref route, boxes, eligibility, and selected query must match A5
  bitwise.
- route-decomposition rows in M are emitted from one forward whenever their
  tensors already coexist.
- ownership rows O report val3 plus D3 calibration and gradient diagnostics.
- only contrasts explicitly named in the block preregistration receive a
  confirmatory Test5 and/or strict2031 pass. The rest remain validation
  ablations and are not promoted after seeing held-out results.

The 1,607-row strict surface is an exact sample-identity subset of strict2031,
not an independent second dataset. Future rows must not run a second model
forward for strict1607. The evaluator filters the 2,031 aligned records using
the preregistered subset IDs and replays both aggregate metrics.

A parity failure is a failed experiment, not an extra metric. It must be fixed
and rerun under a fresh output root before any held-out evaluation.

## 7. Selection and one-time held-out protocol

1. Implement every core config and unit test before starting formal training.
2. Seal config/data/code/initializer hashes and all trainable/frozen key sets.
3. Run U1 gradient/ownership checks and a worst-memory U50 soak.
4. Use seed17 val/calibration only as a numerical-health screen. Core negative
   rows are not dropped for poor performance; only invalid runs are stopped.
5. Run seeds 17/42/73 for every core row.
6. Admission rows use fixed U100 and confidence rows use fixed U50, transferred
   from the anchor selection. Per-row milestone selection is forbidden.
7. Seal one block-level preregistration receipt containing selected checkpoint
   SHA values and exact future surfaces.
8. Evaluate each required held-out surface once (one Ref pass and/or one
   strict2031 pass, with strict1607 derived). No new checkpoint, gap,
   threshold, or row selection is allowed afterward.

The sealed U2-v5 main final records are never rerun. Bootstrap and new route
derivations must consume the immutable existing records when possible. New
rows are described as "prospectively frozen ablations after anchor release",
not as a globally virgin held-out study.

## 8. Statistics

Report every seed plus the cross-seed mean and sample standard deviation
(`ddof=1`); do not run a seed-level t-test with n=3. Primary differences use
5,000 paired image-cluster bootstrap replicates with PCG64 base seed
`20260719`:

- sample image IDs with replacement;
- apply the same image draw to candidate/reference and all three seeds;
- keep every expression or positive/TN pair from an image together;
- recompute each seed's metric, then average the three seed metrics with equal
  weight rather than pooling their rows;
- recompute Test5 micro Acc@0.5 or the complete FPR95 threshold in every
  replicate;
- for FPR95, recompute each model's positive q05 threshold inside every
  replicate rather than holding the summary threshold fixed;
- align candidate and baseline records by immutable sample ID.

Use 95% percentile intervals and the one-sided bootstrap p-value
`(1 + #gain<=0) / 5001`, with gain oriented positive for both Acc and FPR
reduction. The primary planned contrasts are:

1. M3-M1: trained category admission;
2. M4-M3: fresh confidence on primary strict2031, with nested strict1607
   robustness;
3. A5-A1: training-only auxiliary residual;
4. O2-O0: isolated versus shared parameters;
5. O3-O2: sequential versus interleaved isolated optimization;
6. D3m-D2m: verification scope on matched parents.

Apply Holm-Bonferroni within each admission, confidence/data, and ownership
contrast family rather than across exploratory taxonomy rows. The nested
strict2031/1607 metrics are co-primary for a confidence contrast: both
directions must pass, but they form one contrast family and one model forward,
not two independent pieces of evidence.
For phased-versus-joint ownership, use an intersection-union gate: Test5
non-inferiority at margin 0.005 and strict2031 FPR superiority. All other rows
receive CIs and effect sizes but are labelled secondary or
exploratory. Also report exact paired net-example counts, positive q05 change,
and the worst individual Ref-split change; a mean cannot hide a split collapse.

## 9. Acceptance and failure rules

A formal row requires:

- exact trainable tensor allowlist and frozen-state SHA parity;
- finite/nonzero owner gradients, zero AMP skips and zero nonfinite steps;
- no cross-owner autograd connection where isolation is declared;
- complete seed/split/manifest/record identity coverage;
- no overlap between clean confidence train/calibration images and either
  strict manifest.

The existing B56 anchor does **not** satisfy the older 1 GiB headroom rule:
seed42 reached approximately 0.44 GiB minimum free memory. New exact-comparison
rows retain B56 and disclose this fact. If a 1 GiB safety gate is required, a
new lower-batch comparison block must retrain the full anchor and every row;
only lowering the ablations would break fairness.

The main clean claim is accepted only if:

- Test5 mean exceeds B58 and every split is non-regressing within the declared
  0.005 materiality margin, with Holm correction across the five test splits;
- strict2031 FPR95 reduction versus B58 has CI lower bound above zero and the
  nested strict1607 effect has the same direction;
- positive q05 does not regress by more than 0.02;
- ownership and leakage audits pass.

C100 `+0.01` non-inferiority is a secondary diagnostic gate, not a condition
for the B58 clean claim. Its strict2031 failure must remain visible.

## 10. Main paper versus supplement layout

Main paper, in priority order:

1. headline U2-v5 versus B58 result plus Table M decomposition;
2. compact A0/A1/A5 admission table on val3;
3. O0/O2/O3 ownership table plus gradient-conflict figure on val/calibration;
4. compact C0/C2/C3 confidence table if space permits.

The D table enters the main paper only if the matched D3m-D2m contrast supports
the verification-scope claim. Otherwise D and full C remain supplementary.
Legacy U2 and C100 appear only in a visually separated diagnostic panel.

Supplement:

- full A0-A5, C0-C5 and D0-D3/D2m-D3m tables;
- P and S causal routes;
- gap/milestone/queue/candidate-count sensitivity;
- edit-category and Ref-expression-type breakdowns;
- seed-level values, bootstrap distributions, latency/VRAM/parameter counts;
- historical negative-mechanism appendix.

## 11. Execution priority and budget

### P0: no-training evidence

- run seed-first bootstrap on the already sealed main records;
- add the multi-route M evaluator;
- run P/S and Gap3 sensitivity on val/calibration first.

### P1: minimum publishable ablation

- A1 for three seeds; A0 and A5 already exist;
- C2 for three seeds; C0 and C3 already exist;
- O0/O2 for three seeds; O3 is the main architecture.

This minimum requires 12 new training trajectories (A1: 3, C2: 3,
O0/O2: 6). It directly supports the three paper claims without an
unnecessary intermediate architecture.

### P2: complete supplement

- A2/A3/A4: 9 trajectories;
- C1/C4: 6 trajectories;
- O1 shared-trunk intermediate: 3 trajectories;
- D0 reuses C0 identity; D1/D2 add 6 broad-panel trajectories, while D3 is
  the existing main row;
- D2m/D3m: 6 new trajectories;
- all zero-training prompt/support/sensitivity diagnostics.

The complete design therefore adds 42 formal training trajectories. Each
confidence/data trajectory may save U25/U50/U100 for sensitivity, but every
formal comparison consumes the fixed U50. Training is short; evaluation is the
dominant cost, which is why factorized parity and val/calibration mechanism
profiles are mandatory.

## 12. Required implementation before launch

Add, test, and seal:

1. `cfg_stageb_u2v5_ablation_*` leaves that inherit only the clean U2-v5
   configs and expose an explicit one-variable allowlist;
2. a U2-v5 ablation runner with U1/postflight/frozen-hash/AMP/VRAM receipts;
3. a multi-route evaluator for M/P/S with one-forward parity checks;
4. a block preregistration builder that refuses existing held-out outputs;
5. a seed-first image-cluster bootstrap manifest and aggregator;
6. paper-table rendering from sealed receipts only.

The current causal-route evaluator writes route aggregates but not route-level
per-example records. Formal paired bootstrap is blocked until those records
are emitted. The bootstrap implementation must share each image draw across
all three seeds; independently resampling images per seed is not the declared
statistic.

Do not launch old Table B/D runners by renaming their output roots. Their
fixed Top-50 and v19/v22 contracts are not the U2-v5 comparison block.
