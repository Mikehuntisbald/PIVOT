# ARROW CVPR Paper Blueprint v1

> Recommended title: **ARROW: Responsibility-Isolated Admission, Ranking, and Rejection for Selective Visual Grounding**
>
> Alternative title: **ARROW: Responsibility-Isolated Decision Surfaces for Selective Visual Grounding**
>
> One-sentence thesis: **When a foundation grounding model already contains the correct candidate, reliable deployment is primarily a decision-learning problem: who may enter, who should rank first, and whether any output should be emitted must be learned by separate owners.**

## 0. Paper identity and scope

### Public method identity

- **ARROW-V**: visual-support-conditioned Admission; this is the strongest and currently sealed main route.
- **ARROW-T**: canonical-category-text Admission; this preserves the standard image-plus-expression interface and must appear beside ARROW-V in the main paper.
- **ARROW-N**: learned-null/category-agnostic Admission; mechanism control, not a recommended deployment route.
- Historical implementation names such as `B58`, `R100`, `D3`, `U2-v5`, `surface8`, `auxiliary8`, `confidence12`, and `Gap3` should appear only in a reproducibility mapping table or appendix.

### Task scope

ARROW addresses **single-target selective visual grounding**:

\[
(I,e,u) \longrightarrow \hat y \in \{\varnothing, b\},
\]

where \(I\) is an image, \(e\) is a complete referring expression, \(u\) is an optional category cue, and the model either emits one box \(b\) or abstains. ARROW does not claim to solve variable-cardinality multi-target GREC.

### Claim boundary

The paper may claim:

1. Deployed grounding should be factorized into **Admission**, **Ranking**, and **Rejection**.
2. Separate output heads are insufficient when their trainable feature owner is shared; **exclusive parameter ownership** is the supported intervention.
3. Visual support provides category-control evidence beyond category text, while category text provides control beyond a generic learned gate.
4. The isolated rejection score transfers in **relative discrimination** across internal TNs, FineCops-Ref hard negatives, and gRefCOCO no-target cases.
5. Absolute operating-point calibration remains domain-dependent.

The paper must not claim:

- first use of text and visual prompts;
- first no-target/rejection head;
- first observation of a localization–rejection trade-off;
- positive-grounding SOTA on FineCops-Ref;
- universal calibration or one transferable threshold;
- globally image-disjoint gRefCOCO evaluation;
- full multi-target GREC capability;
- image-global or all-query verification for D3 TN labels.

---

# Abstract

Modern grounding models often already generate a geometrically valid candidate, yet deployment still requires three distinct decisions: which candidates are category-eligible, which admitted candidate best matches the complete expression, and whether any valid referent exists. These decisions are commonly coupled through shared scores or trainable features. We show that this coupling is harmful. In our validation analysis, a frozen Grounding DINO baseline contains an IoU\(\geq0.5\) candidate for every expression, while over 27% of examples remain mis-ranked inside the admitted set. Joint admission/rejection optimization exhibits negative gradient alignment across all seeds, and merely separating output heads does not recover rejection. We introduce **ARROW**, which factorizes selective grounding into **Admission**, **Ranking**, and **Rejection** over a frozen candidate representation and assigns every deployed surface an exclusive parameter owner. Visual support controls category admission, the complete expression ranks admitted candidates, and an independent sample-level score determines abstention. ARROW improves Test5 Acc@0.5 by **2.08 points** and reduces FPR95 by **4.19 points** on a leakage-controlled hard-negative evaluation. A controlled ownership intervention preserves localization while reducing FPR95 by **2.81 points**. A fresh category-switch test separates generic gating, category-text control, and visual support (**0.0%, 32.3%, and 48.2%**). Rejection gains transfer to FineCops-Ref and gRefCOCO no-target cases, although fixed operating thresholds remain domain-dependent.

---

# 1. Introduction

## Paragraph 1: practical grounding requires abstention

Referring expression comprehension is usually evaluated under the assumption that every expression denotes exactly one object in the image. A deployed system cannot rely on this assumption: an expression may be invalid, stale, or unsupported by the current scene, and the system must decide whether to localize or abstain. Generalized referring-expression benchmarks make this requirement explicit by including no-target cases. The resulting problem is not only to produce a good box, but to produce a box **selectively**—only when the image-expression pair provides sufficient evidence.

## Paragraph 2: one score is asked to do three incompatible jobs

Most detector-style grounding pipelines expose a collection of text-conditioned queries and attach one or more matching scores to them. In practice, these scores are often asked to serve three different duties: filtering category-ineligible candidates, ordering the remaining instances by the complete expression, and determining whether any candidate should be accepted at all. These duties have different semantics. Admission is a set-construction decision, ranking is a relative comparison, and rejection is a sample-level decision. A relative ranker must always choose a winner, even when every candidate is invalid; a rejector must instead suppress the entire sample. Coupling them through a shared scalar or trainable representation therefore creates an optimization problem, not merely an architectural inconvenience.

## Paragraph 3: empirical diagnosis

We begin from a frozen, data-finetuned Grounding DINO representation. On 26,488 validation expressions per seed, its all-query geometry contains an IoU\(\geq0.5\) candidate for every expression. Admission removes the last correct candidate in only 0.31--0.38% of cases, yet 27.23--27.39% remain ranking errors inside the eligible set. Thus many failures are not caused by missing geometry; the correct candidate is present but loses on the deployed decision surface. Attempts to jointly adapt shared query features for ranking and rejection reveal a second failure: their training gradients are negatively aligned for every seed. Splitting the outputs while retaining a shared trainable feature trunk leaves rejection essentially unchanged, whereas isolating the trainable owners improves rejection without changing the localization route.

## Paragraph 4: method

We therefore introduce **ARROW**, a factorized policy over a frozen candidate representation. **Admission** constructs a category-eligible candidate set from an optional category cue. **Ranking** uses the complete referring expression to select an instance within that set. **Rejection** produces a non-relative sample-level score from detached base statistics and decides whether the selected box should be emitted. Each deployed decision surface has an exclusive trainable owner. A training-only auxiliary route can carry admission supervision, but it is absent from inference and cannot become a hidden fourth score. ARROW-V uses a visual support exemplar for Admission; ARROW-T replaces it with canonical category text while preserving the same decision factorization.

## Paragraph 5: evidence and scope

ARROW improves Test5 Acc@0.5 by 2.08 points and reduces strict hard-negative FPR95 by 4.19 points over the frozen baseline. In the ownership intervention, the isolated model retains the shared model's Test5 result while reducing FPR95 by 2.81 points. A fresh image-disjoint category-switch panel shows that endpoint accuracy alone conceals functional responsibility: a learned null cue retains much standard accuracy but has zero category-switch ability; category text reaches 32.3%, and visual support reaches 48.2%. On FineCops-Ref, the same ordering is recovered on matched positive, negative-expression, and negative-image surfaces. The isolated rejection score also improves AUROC and FPR95 on gRefCOCO no-target cases, including a subset disjoint from its own rejection supervision. Fixed thresholds, however, do not preserve the source operating point across datasets; ARROW improves discrimination rather than universal calibration.

## Contributions

1. **Deployment-duty factorization.** We formulate single-target selective grounding as three deployed decisions—Admission, Ranking, and Rejection—and show that many errors arise after a valid candidate already exists.
2. **Responsibility-isolated optimization.** We assign each deployed surface an exclusive parameter owner. Controlled comparisons show that separate outputs with a shared trainable trunk are insufficient, while structural cross-duty isolation preserves localization and improves rejection.
3. **Functional and external validation.** Category-switch interventions distinguish generic gating, text-conditioned control, and visual-support control; rejection improvements transfer from internally verified TNs to FineCops-Ref counterfactual negatives and gRefCOCO no-target cases, with an explicit boundary between transferable discrimination and domain-dependent calibration.

---

# 2. Related Work

## 2.1 Text and visual prompts for grounding and detection

Open-vocabulary detectors and visual-prompt models have established that text and visual examples provide complementary category evidence. T-Rex2 aligns text and visual prompts in a unified detector, and PET-DINO similarly supports multiple prompt routes in Grounding DINO. ARROW does **not** claim novelty in accepting both modalities. Its focus is decision-level responsibility: the category cue controls Admission, whereas the complete expression owns within-set Ranking. The modalities are not presented as two interchangeable scores to be fused into a single matching surface. A fresh category-switch intervention directly tests whether the Admission route is controlled by its assigned cue.

**Required citations:** Grounding DINO; T-Rex2; PET-DINO; optionally DINO-X and DINOv.

## 2.2 Generalized referring-expression grounding and no-target prediction

GREC extends classical REC to zero, one, or multiple targets. RECANTFormer introduces validity classification for varying target counts; HieA2G predicts output cardinality with an adaptive grounding counter; related GRES methods use dedicated no-target predictors. Recent MLLM work such as RC-GRPO explicitly balances localization and refusal during reinforcement learning. ARROW therefore does not claim the first absent-target head or the first localization–refusal trade-off. It studies a narrower but distinct question in a specialist detector: **which trainable parameters are allowed to own each deployed decision?** Its central comparison shows that separate outputs still fail when they share a trainable feature owner, while disjoint owners remove the cross-duty autograd path.

**Required citations:** GREC/gRefCOCO; RECANTFormer; HieA2G; InstAlign or InstanceVG; RC-GRPO.

## 2.3 Multi-task interference and parameter-efficient adaptation

Multi-task optimization methods diagnose or modify conflicting gradients, while adapters and parameter-efficient finetuning allocate lightweight task-specific parameters. ARROW is not proposed as a general gradient-surgery algorithm. It uses the grounding deployment policy itself to define ownership: a loss may update only the owner of the surface that will execute that duty. This produces a structural zero cross-gradient rather than a numerical correction to a shared gradient. The frozen candidate representation is not merely a compute-saving choice; it preserves a representation already shown to contain adequate geometry and prevents downstream duties from overwriting it.

**Required citations:** PCGrad; GradNorm or CAGrad; adapters/LoRA only if discussed in experiments.

---

# 3. Problem Formulation and Diagnosis

## 3.1 Selective single-target grounding

Given image \(I\), complete expression \(e\), and an optional category cue \(u\), the model predicts either one bounding box or abstention:

\[
\hat y = \pi(I,e,u) \in \mathcal B \cup \{\varnothing\}.
\]

A frozen grounding model produces \(N\) candidate queries:

\[
\mathcal Q = G_{\phi}(I,e)=\{(q_i,b_i,s_i^{0})\}_{i=1}^{N},
\qquad \phi\ \text{frozen}.
\]

The deployed policy must answer three questions:

1. **Admission:** which candidates are eligible for comparison?
2. **Ranking:** which eligible candidate best satisfies the complete expression?
3. **Rejection:** is the selected result sufficiently supported to emit?

## 3.2 Why the duties should not share one score

A rank score is relative:

\[
r_{i^+} > r_j, \quad j\neq i^+.
\]

It need not be low on a no-target image because a maximum always exists. A rejection score is sample-level:

\[
c(I,e) \text{ high for valid referents}, \qquad c(I,e) \text{ low for no-target inputs}.
\]

Admission defines the domain over which ranking is meaningful. A single score trained for all three duties can therefore receive incompatible updates.

## 3.3 Candidate-availability diagnostic

Report the four mutually exclusive error classes:

1. no IoU\(\geq0.5\) candidate exists;
2. a valid candidate exists but Admission removes the final valid candidate;
3. a valid admitted candidate exists but Ranking selects another candidate;
4. localization is correct but Rejection falls below the operating threshold.

The current validation result should be presented as motivation, not as a universal property: all-query geometry supplies a valid candidate for every analyzed expression; Admission loses the last valid candidate in only 0.31--0.38%; roughly 27.3% remain within-set ranking errors.

---

# 4. ARROW

## 4.1 Frozen candidate representation

ARROW retains the frozen data-finetuned Grounding DINO candidate generator. This design follows the diagnosis: the representation already supplies sufficient candidate geometry, while broad joint adaptation previously mixed representation learning with deployment-score learning. The frozen generator also ensures that Admission-input and ownership controls can be compared on identical boxes and base query statistics.

## 4.2 Category-conditioned Admission

For each query, Admission produces an eligibility score:

\[
a_i = A_{\theta_A}(q_i,u).
\]

A relative-gap gate constructs the eligible set:

\[
\mathcal E_{\gamma}
=\left\{i: a_i \geq \max_j a_j - \gamma\right\}.
\]

**Implementation note:** Codex must verify the exact inequality, score domain, and fallback behavior against the deployed evaluator before this equation enters the paper.

ARROW instantiates \(u\) in three capacity-matched ways:

- **ARROW-V:** a visual support exemplar;
- **ARROW-T:** canonical category text;
- **ARROW-N:** a learned category-agnostic null token.

ARROW-V is the strongest route. ARROW-T is the standard-input alternative. ARROW-N is a mechanism control that isolates generic candidate-quality gating from category control.

### Training-only auxiliary carrier

Admission training includes an auxiliary residual that conveys category-complete supervision. This residual is absent from deployed scoring. The paper should distinguish **gradient carriers** from **deployed decision variables**. An auxiliary-only negative control leaves all deployed tensors bitwise equal to the initializer, while adding the residual during surface training improves Test5 by 0.696 points.

## 4.3 Complete-expression Ranking

Within the eligible set, the positive-only ranker scores each candidate using the complete expression:

\[
r_i=R_{\theta_R}(q_i,e), \qquad
 i^*=\arg\max_{i\in\mathcal E_{\gamma}} r_i.
\]

The ranker is trained separately and frozen before final Admission/Rejection optimization. Canonicalizing the ranking text while keeping geometry and eligibility fixed causes substantial top-1 churn and a large localization drop, establishing that attributes and relations belong to the Ranking duty.

A key presentation point is that Ranking alone over the unrestricted query universe gives little gain. This does not make Ranking redundant: Admission constructs a meaningful comparison domain, after which the complete expression resolves instance ambiguity. The two duties are complementary rather than independently additive.

## 4.4 Sample-level Rejection

The rejector receives detached base statistics:

\[
c=C_{\theta_C}(\operatorname{sg}(z_{\phi}(I,e))),
\]

where \(\operatorname{sg}\) denotes stop-gradient. Deployment is

\[
\hat y=
\begin{cases}
 b_{i^*}, & c\geq\tau,\\
 \varnothing, & c<\tau.
\end{cases}
\]

Use the term **sample-level rejection score** or **non-relative rejection score**. If the paper uses “absolute,” define it as absolute with respect to candidate ranking—not as a calibrated probability with invariant semantics across domains.

## 4.5 Exclusive parameter ownership

Let \(\Theta_A,\Theta_R,\Theta_C\) denote the trainable owners of the three surfaces during their respective training phases. ARROW enforces

\[
\Theta_A\cap\Theta_R
=\Theta_A\cap\Theta_C
=\Theta_R\cap\Theta_C
=\varnothing,
\]

and, for duties \(d\neq d'\),

\[
\nabla_{\Theta_{d'}}\mathcal L_d=0.
\]

The second equation is an architectural contract, verified by autograd ownership audits. It is stronger than observing a small numerical gradient. The ownership ablation compares:

- one shared deployed scalar;
- separate outputs over a shared trainable feature trunk;
- independent trainable owners with interleaved exposure;
- independent owners with phased exposure.

The evidence supports **ownership isolation**, not superiority of phased scheduling.

---

# 5. Experiments

## 5.1 Research questions

- **RQ1:** Does factorizing the deployed policy improve localization and rejection over the frozen baseline?
- **RQ2:** Are separate output heads sufficient, or is exclusive trainable ownership required?
- **RQ3:** What information is carried by generic, text-conditioned, and visual-support Admission?
- **RQ4:** Does rejection discrimination transfer to independently constructed TN distributions?
- **RQ5:** Does the source-domain operating threshold transfer unchanged?

## 5.2 Datasets and surfaces

### Internal localization

Define `Test5` semantically in the paper and list its five RefCOCO/RefCOCO+/RefCOCOg test splits. The exact aggregation rule must be stated. Use Acc@0.5 as the primary localization metric.

### Internal rejection

Rename `strict2031` in prose to a descriptive name such as **Strict-TN2031**, then define the 2,031 proposal-covered verified TN pairs, positive set, image-disjointness relative to rejection supervision, and FPR95 computation. Do not imply image-global verification.

### FineCops-Ref

Use the official positive P@1, negative-expression Recall@1, negative-image Recall@1, and official AUROC. State that no FineCops train/val data, checkpoint selection, Gap tuning, alias addition, or threshold fitting was used. ARROW-V is evaluated on the preregistered exact-support surface with 95.60% positive coverage; all A/B/N contrasts use the same covered parent-image/pair surface. The full-test ARROW-T row is the fair line for official comparison.

### gRefCOCO

Evaluate only single-target positives and no-target negatives; exclude multi-target expressions and state the scope. Rename the current `D3-disjoint` surface to **Rejector-supervision-disjoint** or **D3-supervision-disjoint**. All images were exposed during Stage-A, so never call this globally image-disjoint. The subset only excludes images used by D3 rejection train/calibration.

## 5.3 Metrics and statistics

- Localization: Acc@0.5 / P@1 according to the benchmark.
- Rejection: AUROC, AUPR, FPR95.
- Fixed-threshold transfer: TPR and no-target accuracy under the sealed source threshold.
- Mechanism: eligible GT recall, mean eligible count, top-1 churn, mask Hamming distance, gradient cosine, negative-cosine fraction, sign-conflict fraction.
- Confidence intervals: paired image-cluster or parent-image-cluster bootstrap; recompute the positive q05 threshold inside every FPR95 replicate.
- Report exact bootstrap replicate count, RNG, seed, and family-wise correction.

## 5.4 Implementation details

Replace internal shorthand with semantic descriptions in the main text. Put a mapping table in the appendix:

| Paper name | Legacy repository name |
|---|---|
| Frozen base | B58 |
| Complete-expression ranker | R100 |
| Verified rejector | D3 U50 / confidence12 |
| ARROW-V | U2-v5 A / A5+C3/O2 lineage |
| Relative admission margin | Gap3 |

Report measured total, frozen, cumulative-trained, phase-active, and inference-added parameters. Report latency with visual-support encoding both cached and uncached.

---

# 6. Results

## 6.1 Main result: better localization and rejection

**Suggested text:**

ARROW improves the frozen baseline on both deployed endpoints. Test5 Acc@0.5 increases by 2.077 points, with a paired image-cluster 95% CI of [1.827, 2.347] points. On Strict-TN2031, FPR95 decreases by 4.185 points, with a 95% CI of [2.457, 6.199]. These gains do not arise from confidence changing the selected box: the trained Admission-plus-Ranking route and the full model emit bitwise-identical localization records by construction, while Rejection is an independent output.

## 6.2 Where the gain comes from

The unrestricted positive-only ranker is nearly equal to the frozen baseline. A static Admission prior yields a small gain, while learned Admission raises validation Acc@0.5 from 0.6951 to 0.7236. This does not imply that Ranking is unimportant. With geometry and eligibility fixed, replacing the complete-expression ranking text with a canonical noun reduces Acc@0.5 from 0.7246 to 0.6172 and changes the top-1 query for 37.85% of expressions. Admission creates the comparison set; complete-expression Ranking resolves the remaining instance ambiguity.

## 6.3 Parameter ownership, not output count, resolves the conflict

A shared scalar and separate outputs over a shared trainable feature trunk obtain similar calibration FPR95 (0.5732 and 0.5722). Independent owners reduce it to 0.5338. The shared row has negative admission/rejection gradient cosine for every seed (−0.190, −0.252, and −0.281), with sign conflict around 0.53--0.56. The isolated row has no cross-task autograd connection. On confirmatory held-out endpoints, isolation changes Test5 by only −0.002 points while reducing FPR95 by 2.807 points. A phased schedule is not significantly better after ownership has already been isolated.

## 6.4 Endpoint accuracy conceals Admission controllability

The three capacity-matched Admission cues are close on Test5: ARROW-V 0.7424, ARROW-T 0.7412, and ARROW-N 0.7383. Their behavior under a fresh category-switch intervention is radically different: 48.24%, 32.29%, and 0.00%. Thus a generic gate can preserve much standard accuracy without implementing category control. Category text supplies explicit control, and visual support provides additional category evidence beyond the name alone. Avoid the stronger claim that support pixels are required for every correct top-1 prediction.

## 6.5 External hard-negative transfer

On the matched ARROW-V support-covered FineCops surface, visual support exceeds category text on positive P@1 (+0.462 points), negative-expression Recall@1 (+0.555), and negative-image Recall@1 (+0.712), with all paired CIs above zero. Category text also significantly exceeds the learned null route. On the complete FineCops test, ARROW-T is below MM-GDINO-T on positive P@1 but stronger on negative-expression/image Recall@1 and AUROC; present this as a hard-negative advantage, not a positive-grounding SOTA claim.

The rejection owner also transfers to gRefCOCO no-target cases. On Full TestAB, AUROC rises from 0.6895 to 0.7175 and FPR95 falls from 0.7410 to 0.7083. On the Rejector-supervision-disjoint subset, AUROC rises from 0.6869 to 0.7150 and FPR95 falls from 0.7497 to 0.7116; all four gain CIs exclude zero. Because all gRefCOCO images were exposed during Stage-A, this isolates rejection-supervision overlap rather than global image exposure.

## 6.6 Discrimination transfers; calibration does not

The sealed source threshold does not preserve the target operating point externally. FineCops positive TPR is about 80%, and gRefCOCO Rejector-supervision-disjoint TPR has a 95% CI of [0.8839, 0.9098], excluding 0.95. This does not contradict improved AUROC or dataset-normalized FPR95. ARROW learns a more transferable positive-versus-negative ordering, while its score offset, scale, and positive tail remain domain-dependent. Cross-domain operating-point calibration is outside the present scope.

---

# 7. Limitations

1. **Single-target scope.** ARROW emits at most one box or abstention and does not solve variable-cardinality multi-target GREC.
2. **Domain-dependent calibration.** Relative rejection discrimination transfers, but a threshold calibrated internally does not preserve 95% TPR on FineCops or gRefCOCO.
3. **Visual-support coverage.** ARROW-V relies on a frozen support taxonomy; FineCops exact support covers 95.60% of positives. ARROW-T provides a full-coverage standard-input alternative.
4. **Exposure caveats.** FineCops is annotation/task-zero-shot rather than fully image-disjoint. gRefCOCO’s supervision-disjoint subset excludes D3 rejection supervision only; all images were seen in Stage-A.
5. **Positive-grounding ceiling.** ARROW targets selective reliability and does not establish positive P@1 SOTA on FineCops.
6. **Verification scope.** D3 TN labels are proposal-covered SAM3+VLM verified, not image-global or all-query verified.

---

# 8. Conclusion

ARROW treats selective visual grounding as a deployed decision problem rather than a monolithic matching problem. A frozen foundation model supplies candidate geometry; Admission constructs a category-eligible set, the complete expression ranks within that set, and an independent rejector decides whether to emit the selected box. Assigning these surfaces exclusive trainable owners removes a measurable optimization conflict that separate output heads alone do not resolve. The resulting model improves both localization and rejection, exposes the distinct controllability of generic, textual, and visual Admission cues, and transfers rejection discrimination to two external negative distributions. The remaining calibration shift identifies a separate problem: learning a domain-invariant operating point, rather than learning to order valid and invalid samples.

---

# 9. Main-paper figure plan

## Figure 1 — Teaser: the bottleneck is the deployed decision surface

**Purpose:** communicate the scientific discovery, not merely the network diagram.

Three horizontally arranged examples/panels:

1. **Candidate present, wrong top-1:** show the GT-compatible query already among frozen candidates, but the base score chooses another instance.
2. **Shared-duty conflict:** show Admission/Ranking and Rejection gradients entering the same trainable block with negative cosine.
3. **ARROW:** the same frozen candidates flow through separate Admission, Ranking, and Rejection surfaces; output is a correct box or abstention.

Caption headline: “A valid candidate may already exist; reliable grounding requires learning who enters, who wins, and whether any winner should be emitted—without allowing these duties to overwrite one another.”

## Figure 2 — Method and ownership graph

Required visual elements:

- image + complete expression into frozen base candidate generator;
- optional visual support/category text into Admission;
- relative-gap eligible-set construction;
- complete-expression Ranking and selected top-1;
- detached base statistics into sample-level Rejection;
- accept/reject decision;
- stop-gradient symbols;
- deployed routes in solid lines;
- training-only auxiliary residual in dashed lines;
- three owner boxes with no shared trainable path;
- cached support encoding shown as an implementation option.

Do not draw three generic MLP heads on a shared trunk; that would visually erase the novelty.

## Figure 3 — Mechanism evidence and responsibility intervention

Two or three compact panels:

- **(a) Gradient ownership:** O0 per-seed gradient cosine below zero; O2 marked “structural no shared path.”
- **(b) Endpoint effect:** O2−O0 Test5 near zero with CI; strict FPR95 reduction positive with CI.
- **(c) Admission cue:** Test5 bars nearly flat for V/T/N, category-switch bars at 48.24/32.29/0.00.

## Figure 4 — External transfer and calibration boundary

Show three domains/surfaces:

- internal Strict-TN2031;
- FineCops negative expression/image;
- gRefCOCO no-target.

Separate two visual messages:

- discrimination improvement (AUROC up/FPR95 down);
- fixed-threshold operating point (source reaches 95% TPR, external domains do not).

Recommended caption: “Rejection ordering transfers across negative constructions; the source operating threshold does not.”

---

# 10. Main-paper table plan

## Table 1 — Main cumulative decomposition

Rows should use semantic names:

1. Frozen base
2. + complete-expression ranker
3. + static Admission
4. + learned Admission
5. + isolated Rejection (ARROW-V)
6. ARROW-T, optionally adjacent

Columns:

- Test5 Acc@0.5
- Strict-TN2031 FPR95
- eligible GT recall
- mean eligible candidates
- cumulative trained parameters
- inference-added parameters

Use em dashes where a row does not define an independent rejection route. State that learned Admission and full ARROW have bitwise-identical localization outputs.

## Table 2 — Ownership intervention

Main rows: shared scalar, separate outputs/shared trunk, isolated owners. Move phased isolated row to supplement unless space permits.

Columns:

- deployed outputs separate?
- trainable feature owner shared?
- gradient cosine
- sign-conflict fraction
- Test5
- calibration FPR95
- confirmatory strict FPR95

The table title should state the conclusion: **separate outputs are insufficient without separate trainable ownership**.

## Table 3 — Admission cue and controllability

Rows: ARROW-V, ARROW-T, ARROW-N.

Columns:

- Test5
- category-switch success
- FineCops matched positive P@1
- matched negative-expression Recall@1
- matched negative-image Recall@1
- seed SD

Footnotes:

- all routes use 268,167 trainable parameters;
- FineCops comparisons use the identical ARROW-V exact-support surface;
- ARROW-V exact positive support coverage is 95.60%.

## Table 4 — External rejection transfer

Two subtables:

**FineCops:** B58, R100+D3, ARROW-T, MM-GDINO-T reference; report positive P@1, negative-expression/image Recall@1 and AUROC. Keep ARROW-V matched-support results in Table 3.

**gRefCOCO:** B58 vs isolated rejector on Full and Rejector-supervision-disjoint surfaces; report AUROC, AUPR, FPR95, fixed TPR. Do not call the second surface image-disjoint.

---

# 11. Supplement plan

- full A0--A5 admission mechanism grid;
- full C0--C4 confidence objective grid;
- D0/D1/D3 provenance controls; never restore D2/D2m/D3m to manuscript tables;
- O0--O3 complete ownership/schedule results;
- P/S prompt-route and support-patch counterfactuals;
- relative-gap sensitivity with accuracy, eligible recall, and candidate count;
- per-split Ref results and per-seed values;
- FineCops per-negative-type and per-level breakdown;
- confidence CDFs/quantiles across internal, FineCops, and gRefCOCO;
- overlap audits and support coverage;
- qualitative failure taxonomy;
- exact parameter, latency, throughput, and memory receipts;
- preregistration, amendment, failure-ledger, and hash provenance summary;
- legacy-to-paper naming map.

---

# 12. Closest-work comparison matrix for the paper/rebuttal

| Work family | Main focus | Visual/text prompts | No-target support | Shared-vs-isolated owner tested? | ARROW distinction |
|---|---|---:|---:|---:|---|
| T-Rex2 / PET-DINO / DINO-X | prompt unification and generic detection | yes | not the central question | no controlled ownership claim | assigns cues to different deployed duties and tests controllability |
| GREC / RECANTFormer / HieA2G | zero/one/multi-target prediction | text-centric | yes | not their primary controlled variable | single-target selective policy with explicit owner topology |
| InstAlign / InstanceVG | instance-aware generalized grounding | text-centric | yes | not their central mechanism | frozen sufficient candidates + decision-surface ownership |
| RC-GRPO | MLLM refusal/localization balance through RL | language/MLLM | yes | objective-level balance | structural zero cross-duty autograd in a specialist detector |
| Gradient-surgery methods | general multi-task conflict | task-agnostic | n/a | modify shared gradients | removes the shared trainable path based on deployment responsibilities |

Use cautious wording: “not their primary controlled variable” is safer than claiming a work never uses task-specific parameters.
