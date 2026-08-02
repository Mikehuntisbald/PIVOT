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
- V55 is therefore a valid negative result and did not enter formal U4412
  training; and
- V56 is preregistered as a no-local-candidate-auxiliary controlled isolation
  test. It has not been implemented or trained at the time of this ledger.

The repository contains source, configuration, audit, controller, and test
contracts. Model weights and `outputs/` are intentionally not committed. The
absolute artifact paths below are evidence locations on the development
machine, not portable download URLs.

Evidence in this ledger is tagged by use:

| Tier | Meaning | Current examples |
| --- | --- | --- |
| sealed formal | checkpoint, source, manifest, and per-example records are durably bound | required for a final paper comparison; none claimed for V55 |
| diagnostic only | useful controlled screen, explicitly ineligible as a headline result | V45-V55 U400 strict1607 reports |
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

## V56 preregistration

V56 is a preregistered single-variable test of the hypothesized non-deployed
auxiliary conflict. It must be fresh from U6551 and must not continue V55.

### Required change

```text
stage_b_v14_local_absolute_weight = 0
candidate_absolute_head final affine = frozen and bitwise unchanged
```

The primary V56 treatment changes only the local auxiliary weight from 1 to 0.
The existing candidate head remains serialized, its final affine is frozen, and
the audit must prove it is bitwise unchanged. Removing that head would alter the
architecture and parameter surface, so it is reserved for a later mechanical
cleanup only after U0 and deployed-output parity are proved. The intended
deployed system still has exactly two meaningful trainable consumers:

```text
token-veto head
global absolute-confidence head
```

The full-text feature trunk remains trainable. It is not detached from the
global pool; it is optimized exclusively by deployed sample-global objectives.

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

This test is stronger than merely detaching the pool input. Detaching the pool
would force the deployed global head to consume a representation still trained
by a potentially conflicting local auxiliary. V56 instead lets the shared
full-text trunk learn only the deployed global task.

### Decision rule

1. Train a fresh V56 U400 probe.
2. Require complete optimizer/AMP/gradient/ownership health.
3. Run the complete fixed strict1607 diagnostic.
4. Treat at most 800 false accepts only as probe admission, not a paper result.
5. After admission, run formal U4412, strict2031, strict1607, all eight fixed Ref
   splits, and the preregistered seed contract before making a headline claim.
6. If V56 fails, preserve it as a controlled negative result and do not extend it
   merely because one training loss is still decreasing.

### Next branch only if V56 fails

V57 may retain a local candidate auxiliary only by giving candidate and global
separate representation trunks and separate optimizer/clip owners. That is a
larger architecture ablation. It should not be mixed into V56 because doing so
would make the source of any gain ambiguous.

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
- "V56 will pass" before its fresh strict evaluation exists.

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

The next model change is V56 only. No formal training should start until the
fresh U400 result is healthy and strict1607 records at most 800 false accepts.
If that gate fails, the next architectural question is representation-level
trunk separation, not additional unregistered training of V55. Independently,
the evaluation source profile and Table-C recovery evidence must be versioned
and resealed before any affected result is promoted into a paper table.
