# PIVOT Stage-B CVPR Development Ledger

Date: 2026-08-02

## Purpose and status

This is the current development ledger for the single-network PIVOT Stage-B
paper system. It connects the paper claims to the implemented score consumers,
training contracts, checkpoint lineage, strict evaluation evidence, negative
results, and the next preregistered experiment.

The current headline status is:

- the historical multi-checkpoint P750/R100/P50 composition beat the recorded
  GroundingDINO Stage-B data-FT point estimates, but its final artifacts were
  temporary and its components were trained as separate models;
- the current dense-duty line has consolidated patch admission, full-text
  ranking, and absolute confidence into one checkpoint while preserving
  separate score consumers;
- the frozen full-text rank tower is complete at U6551 and remains unchanged
  during confidence training;
- V55 completed a healthy fresh U400 confidence probe, but produced 852 false
  accepts on strict1607 against the controller's preregistered reference count
  of 801 and strict-admission limit of 800;
- V56 then removed representation-level candidate-loss ownership, completed a
  healthy fresh U400 probe, and produced 844 false accepts on the same
  strict1607 evaluation;
- V56 is therefore also a valid negative result and did not enter formal U4412
  training;
- V57 added balanced focal BCE directly to the true deployed global logit and
  completed a healthy fresh U400 probe, but regressed to 849 false accepts; and
- V58 implemented the requested FPR95 active set literally, but regressed to
  914 false accepts despite a healthy U400 run;
- V59 implemented query-structured evidence inside the deployed global score
  while keeping its complete representation owned only by deployed losses;
  its healthy fresh U400 probe produced 876 false accepts on strict1607
  (FPR95 0.545115), so it is a valid negative result and did not enter formal
  U4412 training; and
- paired diagnostics show that V59's query head wins all 1,607 positive/TN
  pairs, but its unbounded additive coordinate raises TN scores more than
  positive scores across samples. The next experiment must preserve V56's
  independent absolute pool as the cross-sample baseline and admit structured
  evidence only through a bounded, one-sided veto; and
- V60 now implements that preregistered intervention with exact V56 U0
  inheritance, deployment-only ownership, detached token routing, and a
  bounded query-wise veto. Its model, migration, training-contract, evaluator,
  controller, and focused regression contracts are implemented; it has no
  empirical result yet.

The repository contains source, configuration, audit, controller, and test
contracts. Model weights and `outputs/` are intentionally not committed. The
absolute artifact paths below are evidence locations on the development
machine, not portable download URLs.

Evidence in this ledger is tagged by use:

| Tier | Meaning | Current examples |
| --- | --- | --- |
| sealed formal | checkpoint, source, manifest, and per-example records are durably bound | required for a final paper comparison; none claimed for V55 |
| diagnostic only | useful controlled screen, explicitly ineligible as a headline result | V45-V59 U400 strict1607 reports |
| historical non-durable | recorded measurements whose complete final artifact closure no longer exists | July P750/R100/P50 composition |
| hypothesis | interpretation that still requires an isolating intervention | shared-trunk local-auxiliary conflict |

Only sealed-formal evidence may support the final superiority claim. Diagnostic
and historical evidence guide architecture decisions but cannot be silently
promoted into the headline table.

## Paper objective

The target comparison is the fixed GroundingDINO Stage-B data-FT system under
the same strict evaluation manifests:

1. lower FPR at the threshold that accepts at least 95 percent of positives;
2. higher RefCOCO-family top-1 `Acc@0.5` on the fixed candidate protocol; and
3. no hidden score sharing that lets one objective improve by corrupting the
   score consumed by the other task.

Older notes sometimes call the Ref metric "AP". In this codebase the primary
Ref metric is top-1 `Acc@0.5`: the selected query is correct when its box has
IoU at least 0.5. It is not COCO-style average precision.

## Three paper contributions

### C1: query localization is sufficient for downstream scoring

The defensible statement is:

> Frozen detector queries already provide a high-recall box pool. PIVOT uses
> patch evidence to admit canonical-category candidates and full referring
> text to order those fixed candidates by attributes, relations, actions, and
> spatial conditions.

This is a candidate-recall and responsibility-separation claim. It is not a
claim that the patch branch has higher Ref top-1 accuracy than GroundingDINO,
that query geometry is universally sufficient on every dataset, or that text
regresses new boxes. Complete-text localization means selecting among fixed
query boxes.

The deployed responsibility split is:

```text
support patch + canonical category
        |
        v
frozen patch branch
        |
        +--> category evidence
        +--> patch Top-50 query IDs and boxes
                         |
complete expression ----+--> full-text rank tower --> Ref query order
                         |
                         +--> confidence adapter --> absolute sample score
```

### C2: traceable counterfactual TN construction

PIVOT retains the identity of the edited word and the source expression for a
counterfactual TN. The strongest label must still match the performed review:

| Verified surface | Allowed wording |
| --- | --- |
| edit provenance only | traceable counterfactual edit |
| target box | target-local verified TN |
| cached proposal set | proposal-covered verified TN |
| every deployed fixed Top-K candidate | deployed-candidate-exact TN |
| independently reviewed full image | image-global verified TN |

The 14,196-row Stage-B training view must not be described as universally
image-global merely because the modified phrase is intended to be absent. Its
sealed metadata states `cached_proposal_coverage_only=true`,
`all_900_gdino_queries_verified=false`, and
`global_max_label_is_semantic_extrapolation=true`. The sample-global Top-50 and
global-pool objectives therefore extrapolate proposal-covered labels beyond the
individually reviewed surface; they are neither image-global nor
deployed-candidate-exact supervision. The current V55 trace receipt contains
13,890 direct-token-valid rows. The remaining 306 rows are masked out of the
confidence training path at runtime and cannot silently receive edit-token BCE
labels.

### C3: edit-role-aware token supervision

The paper contrast is with GroundingDINO's empty-target behavior. For an
empty-box TN, ordinary dense sigmoid focal creates an all-zero query-token
target and treats every valid token on every query as negative. PIVOT instead
uses the traceable edit roles:

| Token role | Positive expression | Counterfactual TN |
| --- | ---: | ---: |
| canonical noun/category | auxiliary positive | shared positive context |
| unchanged modifier/context | positive | positive |
| traceably changed word | positive source word | negative edited word |
| padding/punctuation/non-word | ignored | ignored |

The supervision is also target-query scoped. It is not an all-query blanket
negative objective. This lets the token-veto head learn which textual
condition failed without teaching all unchanged words to be absent.

## Why rank and confidence must be different scores

Ref ranking and FPR95 optimize different order relations:

- Ref ranking needs relative order among queries from the same image;
- FPR95 needs absolute order across different image-expression samples and is
  controlled by the low tail of positive scores and high tail of TN scores.

If one trainable score is shared, their gradients can directly conflict:

- a rank loss can raise a correct query and simultaneously raise the wrong
  text's sample-global maximum;
- a confidence loss can suppress a high TN query and simultaneously suppress
  the correct positive query that determines the positive q05 threshold; and
- pairwise rank improvements do not constrain a common sample-wise translation
  or cross-sample score scale.

PIVOT therefore enforces two score consumers:

```text
rank_score(image, text, query)       -> argmax query for Ref
confidence_score(image, text)        -> one scalar for FPR95
```

The confidence phase consumes detached rank query/text features. Its gradients
must not update the rank tower. Conversely, the rank score must never be added
to the deployed absolute-confidence scalar merely to inherit a convenient
initial scale.

## Historical system versus current single checkpoint

The 2026-07-16 winner was a composition, not one jointly trained network:

| Component | Role | Training state |
| --- | --- | --- |
| P750 | patch candidates/category evidence | separate patch checkpoint |
| R100 | complete-text query ranking | separate rank adapter checkpoint |
| P50 | image-expression confidence | separate confidence adapter checkpoint |
| routed-v3 | validation-selected Ref policy | evaluation-only artifact |

That composition recorded higher top-1 `Acc@0.5` on all eight Ref splits and
727 false accepts on strict1607 versus 799 in its paired July baseline replay.
Those results remain evidence for role separation, but they are not evidence
that the current single-network checkpoint passes. The final merged checkpoint
and route artifacts were stored under `/tmp` and are no longer durable.

The current paper architecture instead uses one serialized model state:

```text
frozen patch candidate path
        |
full-text rank tower, trained to U6551
        |
        +-- detach query/text/rank-token features
                     |
token-aware confidence adapter + absolute confidence pool
```

Rank and confidence remain different outputs inside that one network. "One
network" means shared serialization and one inference graph, not one score.

## Dense-duty training phases

### Comparison-block boundary

The dense-duty V55/V56 development block is not the fixed block defined by
`paper_cvpr_ablation_protocol.md`. That earlier protocol fixes Stage-A
`checkpoint0004` and a b58 Stage-B scorer warm-start. Dense-duty instead fixes
Stage-A `checkpoint0006`, initializes text from GroundingDINO OGC, sets
`no_stage_b_teacher=true`, and explicitly forbids the b58 warm-start. Results
from the two blocks must not be mixed in one controlled ablation table unless
the paper protocol is formally amended and the affected baselines are rerun.

### Rank phase

The rank phase trains the complete-text tower while keeping patch admission
fixed. Its current handoff checkpoint is:

```text
outputs/paper_cvpr_v1/dense_duty_20260728/formal/rank/checkpoint_iter.pth
source optimizer updates: 6551
file SHA-256 used by the V55 migration:
50e60a1314f7f2908bee5eea84ede5549b908177b367609efdec1682caa67ed3
```

Rank selects a query but is not an absolute confidence score. The completed
rank checkpoint is reused as immutable input for every fresh confidence
experiment.

### Confidence phase

The intended confidence phase contract is:

- freeze the patch path and the complete rank tower;
- initialize a fresh confidence adapter from the U6551 rank handoff;
- inherit token semantics with
  `stopgrad(rank_token_logits) - zero_init_token_residual`;
- train token-veto and absolute-confidence parameters only;
- preserve the same TN data, positive data, update budget, tail queue, positive
  q05 trust, pair objective, batch construction, and evaluation manifest across
  controlled architecture comparisons; and
- start formal U4412 only after a fresh U400 probe passes strict1607.

The U400 probe is a screening experiment. It is not a shortened formal result.

## Development trajectory

The table below records complete U400 strict1607 diagnostics available in the
current output tree. All rows use 1,607 pairs. The controller contains a
preregistered reference count of 801 false accepts, so probe admission requires
at most 800. The V55 report does not itself bind a baseline checkpoint and
baseline per-example record artifact; `801` must therefore not be presented as
a new paired comparison produced by that report.

| Revision | Primary change | False accepts | FPR95 | Gate |
| --- | --- | ---: | ---: | --- |
| V45 | split tail-aligned heads | 832 | 0.517735 | fail |
| V46 | positive-tail objective | 844 | 0.525202 | fail |
| V47 | boundary routing | 826 | 0.514001 | fail |
| V48 | FPR active-set reduction | 865 | 0.538270 | fail |
| V49 | global trust/veto routing | 1,497 | 0.931549 | fail |
| V50 | stronger boundary routing | 832 | 0.517735 | fail |
| V51 | independent deployed router | 825 | 0.513379 | fail |
| V52 | candidate/sample calibrator | 891 | 0.554449 | fail |
| V53 | full-expression global absolute pool with inherited carrier | 874 | 0.543871 | fail |
| V54 | exact frozen-rank residual reference | 884 | 0.550093 | fail |
| V55 | independent pool-only absolute confidence | 852 | 0.530180 | fail |

V55 improves over V53 and V54 and is 26 false accepts behind the best dense-duty
U400 row, V51. None of these rows beats the controller's preregistered reference
count. Selecting a row because it is best within this failed set would not
satisfy the paper gate.

## V55 architecture contract

V55 removed every additive frozen-rank carrier from deployed confidence:

```text
token logits = stopgrad(rank token logits) - learned token residual

local candidate logit = candidate_absolute_head(shared full-text feature)

deployed global logit = AbsoluteConfidencePool(shared full-text feature,
                                                detached rank reference).residual
```

The relevant sealed labels are:

```text
revision:
  word_veto_rank_full_expression_global_independent_absolute_v55
head contract:
  split_token_veto_local_candidate_global_absolute_v8
pool contract:
  detached_rank_full_expression_local_candidate_frozen_rank_global_pool_v12
positive trust:
  absolute_global_pool_logit_v4
training schema:
  pivot.stageb.dense_duty_training_contract/v37
migration schema:
  pivot.stageb.rank_to_token_confidence_adapter_fulltext_global_independent_absolute/v22
```

V55 has two optimizer/clip owners:

- token-veto: 21 tensors, 51,267 elements; and
- global owner: 44 tensors, 483,458 elements.

The global owner unfortunately also contains the six-tensor local candidate
head and the full-text feature trunk shared with the global pool. Final affine
separation therefore did not provide representation-level separation.

## V55 execution evidence

### Fresh initialization and trace binding

The V55 confidence probe was freshly migrated from U6551; it did not load a
retired Stage-B confidence tower. Its trace receipt is:

```text
data/ablations/
  stageb_dense_duty_confidence_adapter_fulltext_global_independent_absolute_
  trace_audit_20260802/receipt.json
receipt SHA-256:
8f835b6c39deba306483e6562a6b78683aad8cea3e851c96775841171271933f
direct-token-valid rows: 13,890 / 14,196
```

The first V55 launch failed before its first optimizer update because the v8
head contract was absent from the gradient-clip allowlist. That run is retained
only as a pre-update implementation diagnostic. After adding the v8 contract,
testing it, and regenerating the source receipt, a new fresh U400 run was
started. No failed-run weights or optimizer state were reused.

### Terminal checkpoint

```text
outputs/paper_cvpr_v1/
  dense_duty_adapter_fulltext_global_independent_absolute_highmem_20260802/
  probe/u000400_fresh/checkpoint_iter.pth
file SHA-256:
3b0ba0d55e0423ef9c665fd9381802209470f470b9c2a2e2f518326017e22323
checkpoint reason: max_train_iters
optimizer updates: 400
```

The fail-closed health audit passed:

| Runtime check | Result |
| --- | ---: |
| optimizer boundaries / successful steps | 400 / 400 |
| AMP-skipped steps | 0 |
| nonfinite gradient boundaries | 0 |
| zero-gradient successful steps | 0 |
| token/global owner clip violations | 0 |
| frozen parameter state changed | no |
| active tensors | 65 |
| peak reserved GPU memory | 31,459,377,152 bytes |

Both owners had nonzero gradients at every successful step and were clipped
independently to 0.1. The failure is therefore not explained by a dead branch,
AMP skipping, missing optimizer ownership, or low memory use.

### Tail health and strict evaluation

The terminal training queue already exposed the failure mode:

| Statistic | V55 U400 |
| --- | ---: |
| positive q05 | -0.1137085 |
| TN q95 | 1.0039063 |
| operating gap | -1.1176147 |

The fixed strict1607 diagnostic then reported:

| Metric | Value |
| --- | ---: |
| valid pairs | 1,607 |
| positive accepts | 1,527 |
| V55 false accepts | 852 |
| V55 FPR95 | 0.53018046 |
| baseline false accepts | 801 |
| maximum admitted false accepts | 800 |
| decision | `valid_nonwin_do_not_enter_formal` |

The report is stored locally at:

```text
outputs/paper_cvpr_v1/
  dense_duty_adapter_fulltext_global_independent_absolute_highmem_20260802/
  probe_evaluation/u000400_strict1607_report.json
```

The controller correctly returned a nonzero non-win status and did not start
formal U4412.

This report is explicitly `diagnostic_only=true` and
`formal_gate_eligible=false`. It is a development-screen result, not a paper
headline result.

## V55 root-cause audit

### What did learn

V55 was not an underpowered zero-output head:

| Update | candidate final-output L2 | pool final-output L2 |
| ---: | ---: | ---: |
| U100 | 0.01853 | 0.01482 |
| U200 | 0.04215 | 0.04184 |
| U300 | 0.06119 | 0.06674 |
| U400 | 0.07038 | 0.07230 |

The terminal global pre-clip gradient norm was 13.20 and its maximum was
39.37. The optimizer had a live signal throughout training.

Same-pair separation is present. On strict1607, the raw deployed global logit
put the positive above its paired TN 86.80772 percent of the time with a mean
pair gap of +0.7091383. This rules out a wholesale sign inversion and shows
that the data contains learnable semantic signal. Without a matched U0
evaluation, it does not establish that U400 improved either statistic.

### What failed

Cross-sample absolute ordering failed. Keep the three diagnostic scales
separate:

| Surface | positive q05 | TN q95 | pair win | mean pair gap |
| --- | ---: | ---: | ---: | ---: |
| terminal training queue, raw logit | -0.1137085 | 1.0039063 | not sealed | not sealed |
| strict1607, raw deployed global logit | -0.2121582 | 1.0117188 | 0.8680772 | 0.7091383 |
| strict1607, sigmoid score | threshold 0.447265625 | not separately reported | 0.8674549 | 0.1645765 |

The raw strict operating gap is approximately -1.223877. Pairwise loss cannot
fix this by itself because it is insensitive to sample-wise common translation
and does not directly constrain the global positive-low/TN-high tails. Values
from the raw-logit, sigmoid, and training-queue surfaces must not be combined in
one claimed statistic.

The leading structural explanation is:

1. one full-text cross-attention/global-query trunk creates `dense_feature`;
2. `candidate_absolute_head(dense_feature)` receives local candidate focal;
3. `AbsoluteConfidencePool(dense_feature, ...)` receives sample-global losses;
4. both paths are grouped into the same 44-tensor owner and one 0.1 clip; and
5. only the pool scalar is deployed for FPR95.

The non-deployed local objective became easier while the deployed tail became
worse. Local focal fell from roughly 0.332 at U100 to 0.236 at U400, whereas the
global/TN tail did not improve monotonically. The strict records are consistent
with, but do not prove, the coupling hypothesis:

- deployed global and candidate-local correlation: 0.937 on positives;
- deployed global and candidate-local correlation: 0.972 on TNs;
- candidate-local false accepts: 861;
- deployed global false accepts: 852; and
- overlap: 832 of the 852 deployed false accepts were also candidate false
  accepts.

The candidate-local and deployed-global false-accept sets above each use their
own positive-q05 threshold. Their high overlap and correlation are association,
not causal identification.

V55 therefore separated final affine outputs, but did not separate the learned
representation or objective ownership. Live gradients and nonzero parameter
movement exclude a dead or zero-output head; they do not prove that capacity is
sufficient. The current evidence neither establishes a capacity bottleneck nor
rules one out. V56 isolates the potentially conflicting local auxiliary before
larger-capacity alternatives are interpreted.

## V56 ownership intervention and result

V56 was preregistered as an ownership intervention for the hypothesized
non-deployed auxiliary conflict. It must be fresh from U6551 and must not
continue V55. Its deployed V55 score definition is held fixed; the treatment
is which losses are allowed to own the representation that produces that
score.

### Required change

```text
stage_b_v14_local_absolute_weight = 0
candidate_absolute_head input = dense_global.detach()
candidate_absolute_head = fully frozen, diagnostic only
global trunk + AbsoluteConfidencePool = deployed-global owner only
```

The existing candidate head remains serialized for exact U6551 migration and
diagnostics, but all six of its tensors are frozen, its input is detached, and
its local loss has exactly zero weight. Full freezing is necessary because a
zero loss alone would not prevent AdamW from moving a parameter that remained
in the optimizer. The audit must prove that the six tensors have no optimizer
state and are bitwise unchanged. Removing the head entirely is reserved for a
later mechanical cleanup only after U0 and deployed-output parity are proved.
The intended deployed system has exactly two trainable owners:

```text
token-veto head
global absolute-confidence head
```

The 21-tensor token-veto owner contains the independent token residual scorer.
The 38-tensor deployed-global owner contains `patch_feature`, the full-text
cross-attention path, `global_query_trunk`, and `AbsoluteConfidencePool`.
Together they form the 59-tensor, 468,164-parameter active surface. Frozen rank
token evidence is an input, and the token residual consumed by the global path
is detached, so neither owner's losses rewrite the other owner.

The full-text global trunk remains trainable and is not detached from the
global pool. FPR95 active-set loss, global TN softplus, same-pair loss, and
positive-q05 protection all consume the true sample-global deployed logit.
The q05 route returns that deployed tensor itself rather than a merely
value-equal pool diagnostic alias.

### Variables that must remain fixed

- U6551 rank checkpoint with SHA-256
  `50e60a1314f7f2908bee5eea84ede5549b908177b367609efdec1682caa67ed3`;
- patch Top-50 candidates and boxes;
- positive and TN training manifests;
- 13,890 direct-token-valid trace subset;
- token-veto BCE and role masks;
- global TN softplus objective;
- FPR95 tail queue and reduction;
- positive q05 trust objective;
- same-pair loss;
- batch size 16, gradient accumulation 2, AMP, learning rate `2e-5`, seed 42,
  and the existing clipping policy;
- tail queue capacity 4096 with minimum 256 entries;
- pair-loss weight 0.25 and margin 0.05;
- unchanged q05/q95 reductions and positive-trust implementation;
- U400 health audit; and
- strict1607 manifest, evaluator, and integer false-accept gate.

This test rejects the alternative split that detaches the global-pool input and
lets the local candidate branch own the trunk. That alternative would leave the
deployed scalar consuming a representation optimized by a non-deployed
objective. V56 instead gives the deployed loss complete ownership of its input
representation. It therefore addresses representation-level leakage while V55
addressed only score-level leakage.

### Decision rule

1. Train a fresh V56 U400 probe.
2. Require complete optimizer/AMP/gradient/ownership health.
3. Run the complete fixed strict1607 diagnostic.
4. Treat at most 800 false accepts only as probe admission, not a paper result.
5. After admission, run formal U4412, strict2031, strict1607, all eight fixed Ref
   splits, and the preregistered seed contract before making a headline claim.
6. If V56 fails, preserve it as a controlled negative result and do not extend it
   merely because one training loss is still decreasing.

### Executed evidence

The fresh V56 U400 checkpoint is:

```text
outputs/paper_cvpr_v1/
  dense_duty_adapter_deployment_owned_global_highmem_20260802/
  probe/u000400_fresh/checkpoint_iter.pth
SHA-256 = 4b0d001048b51da16231d5ac66297ca9d9b390f123722e4f3688932f46615026
```

The health audit passed all executable checks:

- 400 requested and 400 successful optimizer steps;
- zero AMP skips and zero non-finite gradient boundaries;
- 59 active tensors and 468,164 active parameters;
- token owner: 21 tensors and 51,267 parameters;
- deployed-global owner: 38 tensors and 416,897 parameters;
- candidate diagnostic: six tensors and 66,561 parameters, all frozen, with no
  optimizer state and an unchanged fingerprint;
- nonzero token and global gradients on every successful step; and
- peak reserved CUDA memory of 31,314,673,664 bytes.

The bound strict1607 report is:

```text
outputs/paper_cvpr_v1/
  dense_duty_adapter_deployment_owned_global_highmem_20260802/
  probe_evaluation/u000400_strict1607_report.json
decision = valid_nonwin_do_not_enter_formal
```

Its complete 1,607-pair result was 844 false accepts, FPR95 0.525202,
pair-win 0.868699, mean positive score 0.657434, mean TN score 0.494454,
and mean paired score gap 0.162979. Relative to V55, V56 removed eight false
accepts and slightly improved paired discrimination, but remained 44 false
accepts above admission. The result is diagnostic only and no formal U4412
training was launched.

The training-tail trajectory rules out blind continuation as the next action.
Between U222 and U400, the training queue positive q05 fell from -0.07094 to
-0.13251 while TN q95 rose from 0.87036 to 1.01758. At the same time, positive
mean rose from 0.54156 to 0.65938 and TN mean fell from 0.12326 to 0.05371.
Thus mean and paired discrimination improved while exactly the two tails used
by FPR95 became less separated. More updates to the identical objective are not
preregistered as a remedy.

### Interpretation

V56 proves the ownership intervention is executable and auditable, but it does
not prove that the removed candidate loss was harmful. It exposes a more direct
objective hole: after setting local candidate absolute loss to zero, the true
deployed global logit has a negative softplus, paired/tail terms, and a
low-positive hinge, but no balanced per-sample absolute positive/TN objective.
The weak hinge is active only below its margin and did not prevent unseen
positive low-tail collapse. This is an objective-routing deficiency, not
evidence that the 416,897-parameter global owner is too small.

## V57 preregistration: deployed-global balanced absolute supervision

V57 keeps the complete V56 architecture, U0 migration, parameter ownership,
candidate freeze, inference score, data, optimizer, and U400 budget. Its only
treatment is a new balanced focal-BCE term evaluated on the two real deployed
sample-global logits:

```text
positive term = focal_bce(deployed_global_positive_logit, target=1)
TN term       = focal_bce(deployed_global_TN_logit, target=0)
loss          = 0.5 * positive term + 0.5 * TN term
gamma         = 1
weight        = 1
```

This loss is not candidate-local supervision. It must consume exactly the same
sample-global tensor used by inference, FPR95 active-set training, and q05
protection. Candidate loss remains exactly zero; the candidate head remains
frozen and reads `dense_global.detach()`; token logits remain stop-gradient at
the global boundary. Consequently every trainable global-trunk and pool
parameter is still owned only by losses on the deployed global output.

The implementation must add a separately named loss/metric rather than
silently reusing `loss_fixed_text_local_absolute`. Tests must prove exact value
and gradient routing, disjoint token/global owners, frozen-candidate stability,
and no change to U0 deployed scores. V57 receives the same health audit and
strict1607 gate: at most 800 false accepts admits formal U4412; any larger count
is a preserved negative result.

### Executed V57 result

The fresh V57 checkpoint completed exactly 400/400 optimizer updates with zero
AMP skips, zero nonfinite-gradient boundaries, and zero successful steps with
an inactive owner. Its 59 trainable tensors remained exactly partitioned into
21 token-veto tensors (51,267 parameters) and 38 deployment-global tensors
(416,897 parameters). The six serialized candidate-head tensors remained
frozen and the complete frozen-parameter hash was unchanged.

```text
checkpoint = outputs/paper_cvpr_v1/
  dense_duty_adapter_deployed_global_balanced_absolute_highmem_20260802/
  probe/u000400_fresh/checkpoint_iter.pth
sha256 = 72e6c0630a56e181f5a93faa1a0de197df49eef851b783a544dede7a44763fdd
```

The fixed strict1607 evaluation produced 849 false accepts, FPR95 0.528314,
pair-win 0.869944, mean paired gap 0.165416, mean positive score 0.644419, and
mean TN score 0.479003. V57 improved pair-win and paired gap slightly relative
to V56, but added five false accepts and remained 49 above admission. It is a
valid negative result; formal U4412 was not launched.

The operating-point decomposition is diagnostic. Relative to V56, V57 shifted
the positive mean down by 0.013015, the TN mean down by 0.015451, and the
positive-q05 threshold down by 0.013916. FPR75 and FPR90 improved slightly, but
FPR95 worsened. Ordinary balanced BCE therefore improved broad calibration and
same-pair discrimination without protecting the exact positive low tail that
defines the 95%-TPR threshold. This rules out merely increasing the same BCE or
continuing V57 as the next controlled action.

The failed strict launch before evaluation was a controller-only defect: the
V57 wrapper returned the health-audit result where the shared core expected an
audit callable. It failed before model evaluation and created no result. The
wrapper now returns `health.audit`, a regression assertion covers the callable
contract, and the unchanged checkpoint/config/manifest completed the valid
strict1607 run above.

## V58 preregistration: deployment-owned stable FPR95 active set

Reviewing the executed contract exposed a semantic mismatch with the proposed
deployment-owned design. V56 and V57 computed the exact historical positive
q05 and recorded the exact `TN >= q05` active set, but their configured
`all_mean_v1` reduction still backpropagated through every TN. With margin 0.3
and temperature 0.1, already-rejected TNs immediately below q05 can retain a
large gradient. Thus the training route used the true deployed logit, but the
negative reduction was not literally active-set-only.

V58 returns to V56 (no deployed-global BCE) and changes one loss geometry:

```text
active_i = stopgrad(TN_i >= historical_positive_q05)
loss_i   = tau * softplus((TN_i - surrogate_q05 + margin) / tau)
loss     = sum(active_i * loss_i) / number_of_all_valid_TNs
```

The denominator deliberately remains the complete valid-TN count. The earlier
V48 `exact_fpr95_active_set_mean_v1` divided by the number of active TNs and
regressed to 865 false accepts; as the active set shrank, each remaining TN
received a larger gradient. V58 removes gradients from inactive TNs without
introducing this inverse-active-fraction gain. Positive-q05 protection, pair
term, queue size, margin, temperature, data, U0, optimizer, U400 budget,
candidate freeze, and all 59 owner tensors remain identical to V56.

V58 is admitted to formal U4412 only at 800 or fewer false accepts on the same
strict1607 manifest. A larger count is another controlled negative result and
must not trigger continuation of the identical objective.

### Executed V58 result

V58 completed exactly 400/400 optimizer updates. Its checkpoint is:

```text
outputs/paper_cvpr_v1/
  dense_duty_adapter_deployment_owned_global_stable_fpr95_active_set_
  highmem_20260802/probe/u000400_fresh/checkpoint_iter.pth
sha256 = 1b921bd8e553be558ec874c9d3cac266582a1c617c96f06b1ea7c6c1ef0b652e
```

The health audit reported zero AMP skips, nonfinite-gradient boundaries, and
zero-gradient owner steps. The frozen hash was unchanged. The 59 active tensors
remained 21 token-veto plus 38 global-owner tensors. Across the terminal logged
window, the mean valid-TN count was 15.301, active and selected counts were both
9.773, and the active fraction was 0.6249. Thus V58 genuinely removed about
37.5% of TNs from the FPR95 term; it was not an alias of `all_mean_v1`.

The corrected fixed strict1607 report reproduced 914 false accepts, FPR95
0.568762, pair-win 0.861854, mean paired gap 0.102963, positive mean 0.622630,
TN mean 0.519667, and positive-q05 probability threshold 0.492432. Both splits
regressed: RefCOCO+ val FPR95 was 0.561658 and RefCOCOg UMD val was 0.588785.
No formal training was launched.

V58 disproves the pure-active-set remedy. Removing inactive-TN gradients kept
the training positive q05 much healthier than V56, but unseen TN scores
collapsed toward an approximately 0.5 absolute-confidence plateau and their
high tail was not suppressed. The all-TN term was therefore providing useful
cross-sample scale regularization even though it was not an exact FPR95
objective. V57 conversely showed that ordinary sample BCE improves broad
calibration but not q05. The next change must add query-structured deployed
evidence, not select another scalar loss weight.

## V59 implementation: query-structured deployed global evidence

V55's local candidate loss improved representation learning but violated the
desired ownership rule because a non-deployed objective owned the trunk. V56
through V58 removed that leakage, yet left every global loss supervising only
one scalar per expression. A 416,897-parameter cross-attention trunk is then
weakly identified: many query representations can produce the same pooled
scalar, and scalar positive/TN gradients fight in the shared representation.

V59 preserves deployment ownership while changing the deployed score's
granularity:

```text
dense_global(query, text, patch)
        -> deployed_query_absolute_logits       # trainable, not diagnostic
        -> rank-conditioned monotone aggregate
        -> deployed_sample_global_logit
```

For eligible queries `q`, frozen rank logits `r_q`, deployed absolute query
logits `a_q`, temperature `tau`, and the independent pool residual `p`, the
sample-global deployment logit is

```text
g = tau * (logsumexp_q((stopgrad(r_q) + a_q) / tau)
           - logsumexp_q(stopgrad(r_q) / tau)) + p
```

This normalized log-sum-exp is monotone in every `a_q`, equivariant to a common
shift of all query logits, and exactly zero at the zero-initialized query head.
Consequently V59 inherits V56's deployed global output at U0 rather than
changing the checkpoint's initial decision surface. Rank affects which query
evidence matters but is detached and is never added as an absolute confidence
coordinate.

The existing six-tensor, 66,561-parameter query head moves from
`diagnostic/frozen` into the deployed global owner. Together with the
416,897-parameter global trunk and pool, the global owner contains 44 tensors
and 483,458 parameters. The token-veto owner remains 21 tensors and 51,267
parameters, for 65 active tensors and 534,725 confidence parameters total.
Candidate local loss is exactly zero; candidate head input is not detached in
V59 because the head and its input trunk are both consumed by and owned by the
actual deployed score. No IoU/candidate auxiliary is permitted to update them.

All global TN, FPR95 tail-queue, and positive-q05 protection losses consume
`absolute_global_confidence_logit_v2`, the actual `g` above. V59 returns to
V56's all-TN reduction, omits V57's added balanced BCE and V58's active-only TN
reduction, and retains the same U400/strict1607 gate. Migration schema v24,
fresh-training contract v22, training audit schema v41, runtime owner counts,
formal admission binding, and U0 equality/gradient-routing tests fail closed.
The direct-trace audit accepts 13,890 of 14,196 rows and is bound by receipt SHA
`197c1fb2d6680b9f1785c0f2c36eb053bbf13922712ed438ce88267f33c13396`.

V59 completed exactly 400 successful optimizer updates with zero AMP skips,
zero nonfinite gradients, an unchanged frozen-state fingerprint, and 65/65
active tensors live at every audited boundary. The checkpoint is
`outputs/paper_cvpr_v1/dense_duty_adapter_deployment_owned_query_global_highmem_20260802/probe/u000400_fresh/checkpoint_iter.pth`,
with SHA256
`6f49bce4a8bde3c9c6af2b564c29f44fb3bd07260668f08c3cd647c577260215`.
The health audit classified it as `healthy_for_strict1607_diagnostic`.

The sealed strict1607 replay then produced 876 false accepts, FPR95 0.545115,
1,527 positive accepts, pair-win 0.870566, positive mean probability 0.666412,
TN mean probability 0.534597, and a 95%-TPR threshold of 0.483545. Its
per-example record SHA256 is
`520e41a2bb47a8414f0f31de8c79c1dc28ae470fdc7e3defd4a8604dfb6208e8`.
The controller returned `valid_nonwin_do_not_enter_formal`; no formal training
was launched.

The failure is not evidence that the adapter lacks cross-modal capacity. The
query-head maximum separates the positive from its paired TN on all 1,607
pairs, with a mean paired query-logit gap of 0.532020. Relative to V56, however,
V59 raises the TN deployed score by 0.040140 on average while raising the
positive score by only 0.008980; 80 examples become V59-only false accepts,
whereas only 48 V56 false accepts are repaired. Query-relative evidence is
therefore useful, but its free additive offset is not a calibrated cross-image
confidence coordinate. The next structure must keep the V56 independent pool
as the absolute baseline and use query/token evidence only to lower confidence
under detected mismatch, never to raise it freely.

## V60 implementation: V56 absolute baseline plus bounded query veto

V60 changes the semantics of the same six-tensor query head without adding a
new parameter surface. The independent V56 pool residual `p` remains the sole
cross-sample absolute-confidence coordinate. For frozen-rank weights `w_q`,
detached token mismatch gate `m_q`, raw query-head output `z_q`, gate floor
`f=0.25`, and maximum depth `D=8`, the deployed score is

```text
d_q = D * tanh(relu(z_q) / D)
e_q = f + (1 - f) * stopgrad(clamp(m_q, 0, 1))
v   = sum_q w_q * e_q * d_q
g   = p - v
```

Thus `0 <= v <= 8` and `g <= p` for every deployed sample. V60 cannot repeat
V59's TN-raising failure mode. At U0 the zero-initialized query head makes
`v=0` exactly, so the deployed score is bitwise equal to V56. A centered
softplus surrogate backward is used behind the exact ReLU/tanh forward; the
0.25 gate floor therefore gives the query head a nonzero first-step learning
path even when the token gate has not opened.

Token mismatch routing is detached before it enters the global score. Token
edit BCE owns the token-veto parameters; deployed global TN, FPR95 tail queue,
and positive-q05 protection own the complete cross-attention/global trunk,
query veto head, patch feature path, and independent pool. Candidate-local loss
and the extra deployed focal BCE are both exactly zero. The real deployed `g`,
not `p` or a diagnostic candidate logit, is the positive trust and FPR95 loss
consumer.

V60 retains the 65-tensor, 534,725-parameter active surface: 21 token tensors
(51,267 parameters) and 44 deployed-global tensors (483,458 parameters).
Migration schema v25, fresh-confidence contract v23, training-contract schema
v42, strict evaluator registration, formal admission binding, and independent
owner clipping fail closed. Focused tests cover V56/V60 state equality, exact
U0 deployed equality, one-sidedness, the depth bound, gradient ownership,
migration identity, combined evaluator registration, and controller wiring.

## Evaluation and claim gates

Every paper candidate must satisfy all of the following:

1. exact checkpoint/config/source-closure binding;
2. frozen patch and rank state during confidence training;
3. complete optimizer state and exact successful-update count;
4. zero AMP-skipped and nonfinite-gradient steps;
5. per-example strict records with exact manifest identity/order;
6. integer-replayed FPR95 false accepts;
7. strict improvement rather than equality with the baseline;
8. Ref evaluation where the rank output, not confidence, selects the box; and
9. no claim based solely on training queue q05/q95 or pair win rate.

For the current strict1607 gate:

```text
controller reference = 801 false accepts
probe admission = candidate false accepts <= 800
```

The one-count margin is intentional: equality with the preregistered reference
is not probe admission. A formal paper comparison additionally requires a
checkpoint- and manifest-bound paired baseline artifact.

## Reproduction entrypoints

V55 terminal training status:

```bash
/home/haoyi/miniconda/envs/gdino5090/bin/python3.11 \
  tools/run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_u0400.py \
  --status
```

V55 training health:

```bash
/home/haoyi/miniconda/envs/gdino5090/bin/python3.11 \
  tools/audit_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_health.py
```

V55 strict1607 status or replay:

```bash
/home/haoyi/miniconda/envs/gdino5090/bin/python3.11 \
  tools/run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_evaluation.py \
  --status

/home/haoyi/miniconda/envs/gdino5090/bin/python3.11 \
  tools/run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_evaluation.py \
  --run
```

The formal controller exists but is fail-closed on the U400 admission report.
It was not run for V55.

## Current verification record

Before the V55 run, the focused contract suites reported:

- 369 pytest tests passed;
- 31 unittest subtests passed;
- 7 V55 controller tests passed; and
- 62 evaluator tests passed.

After the pre-update clip-allowlist correction, the checkpoint-audit suite
reported 41 tests and 18 subtests passed. The source receipt was regenerated
after that code change, so the successful U400 checkpoint does not rely on the
pre-fix source closure.

These are focused development suites, not a claim that every historical test
in the repository was rerun after every experiment.

### Repository-wide release audit on 2026-08-02

After reconstructing the Git baseline and staging the full research tree, the
entire repository suite was run twice. The first pass exposed 33 failures. Four
implementation/test-isolation repairs were made:

- the GDINO adapter and semantic auditors now inherit the argparse-owned
  `find_unused_params=False` default instead of requiring an illegal config
  redeclaration;
- the evaluation wrapper's content hashes were rebound after auditing the
  referenced files;
- serial queue marker proof now ignores same-UID processes that provably
  predate the sealed launch epoch, while remaining fail-closed for unreadable
  processes created inside that epoch; and
- the Table-C outer-window race unit test now isolates its unrelated validation
  recovery dependency and expects the implemented v5 adapter.

The post-fix full command was:

```bash
/home/haoyi/miniconda/envs/gdino5090/bin/python3.11 -m pytest -q
```

Its exact result was:

```text
2637 passed, 3 skipped, 12 failed, 1141 subtests passed
runtime: 792.61 seconds
```

The 12 remaining failures are classified, not unexplained:

| Count | Fail-closed surface | Required resolution |
| ---: | --- | --- |
| 4 | Table-C execution-snapshot tests cascade before inventory publication | version the evaluation source profile, then rebuild/reseal the snapshot |
| 3 | legacy matrix/evaluation contracts require the sealed 72 common files | split legacy Table-C/M0 v1 from the current dense-duty profile, or upgrade every schema and consumer together |
| 5 | V45/V46/V47/V48/V50 tests inspect host-local historical checkpoints whose source closure has since changed | make controller unit tests hermetic; retain the real checkpoints as correctly invalid historical evidence |

The current evaluation common closure contains 178 files rather than the
sealed legacy 72; token provenance adds 3 files, and the controller/evaluator
union is 183 rather than 77. This is an unaudited profile expansion, not a
number to patch from 72 to 178. Formal Table-C/M0 replay must remain closed
until a legacy/current profile split or a complete schema upgrade and reseal is
performed.

There is also an independent Table-C evidence-lifecycle issue not hidden by the
unit-test isolation fix. The validation-recovery receipt bound a live GPU lease
that the retirement controller later deleted. Durable replay therefore needs
either an archived lease record or retirement-aware lineage with a new
receipt/adapter schema. Until then, that Table-C completion evidence is not a
replayable paper artifact.

The targeted suites for the repaired implementation are green: 38 GDINO
adapter/semantic orchestration tests, 22 serial queue tests, and 2 Table-C
receipt tests passed. The repository-wide suite is deliberately not described
as green while the 12 evidence canaries remain active.

## Paper wording boundary

Supported now:

- patch evidence and full text are assigned different scoring responsibilities;
- rank and absolute confidence require separate trainable outputs;
- traceable edits permit token-role supervision that all-negative focal cannot
  express;
- one checkpoint can contain frozen patch/rank paths plus an independently
  trained confidence adapter; and
- V55 is a useful negative result consistent with final-head separation being
  insufficient when a non-deployed local loss still owns the deployed trunk.

Not yet supported by the current single-network result:

- "the final single-network PIVOT system beats GroundingDINO Stage-B data-FT";
- "all TN expressions are absolutely absent from the image";
- "query localization is always sufficient";
- "the lightweight adapter lacks capacity"; or
- "ordinary balanced BCE solves deployed confidence calibration".

The historical composite winner may be reported as prior system evidence only
with its multi-checkpoint and adaptive-route limitations stated explicitly.

## Repository commit phases

The development tree is committed in reviewable layers:

| Phase | Scope |
| --- | --- |
| core architecture | datasets, model branches, losses, engine, main entrypoint, migration/audit utilities |
| experiment protocol | configs, evaluators, controllers, manifests, result aggregators |
| verification | unit, contract, source-closure, controller, and evaluator tests |
| paper documentation | this ledger, claim contracts, historical handoffs, and README index |

Generated checkpoints, output records, telemetry, caches, build products, and
the multi-gigabyte `data/ablations/` workspace remain excluded from Git. Their
hashes and expected locations are recorded by the audit/controller layer.

The repository is currently a sealed-host research release rather than a
portable package: many launchers retain `/home/haoyi` interpreter paths and
`/media/haoyi` dataset/artifact paths. These paths document the executed
environment but require explicit remapping on another host. Portability work is
separate from the result-preserving commits in this ledger.

## Immediate next action

Run the fresh V60 U400 probe, health audit, and strict1607 replay. Do not start
formal training unless the controller records at most 800 false accepts. If it
passes, launch the separately gated fresh U4412 formal confidence run; if it
does not pass, preserve the result and diagnose veto coverage/depth rather than
reintroducing an unconstrained additive absolute coordinate.
Independently, the evaluation source profile and Table-C recovery evidence must
be versioned and resealed before any affected result is promoted into a paper
table.
