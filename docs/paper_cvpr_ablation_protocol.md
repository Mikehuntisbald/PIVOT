> **Historical pre-ARROW artifact.** PIVOT names and schemas below identify the
> sealed implementation lineage and are intentionally preserved.

# CVPR Stage-B Ablation Protocol

> Historical protocol for the checkpoint0004/Top-50/v19-v24 model family.
> It is not the execution authority for the leakage-clean U2-v5 main model.
> See `paper_cvpr_u2v5_complete_ablation_design_20260817.md`; do not port this
> matrix by changing only checkpoint or output paths.

This document is the experiment contract for the three paper claims. It is
deliberately stricter than the historical development comparisons: a row may
enter a paper table only when its initialization, data split, update budget,
candidate surface, evaluation records, and training seeds are sealed.

## Claim wording

### C1: localization and scoring roles

Use the following claim:

> Frozen detector queries already provide high box-recall. PIVOT therefore
> uses the patch branch as a canonical-category candidate prior and the
> full-text branch to rank those fixed candidates using attributes, relations,
> and spatial language.

The recovered same-caliber evidence supports only a narrow candidate-pool
statement: the Stage-A patch route has slightly higher aggregate candidate
recall and already supplies sufficient localization coverage for downstream
ranking. It does not support an AP50 advantage; patch AP50 is lower than the
same-data GroundingDINO FT control. Do not generalize the recall observation to
every split, query source, or operating point. The publishable point is that
localization is not the dominant bottleneck and that the two score sources have
measurably different causal roles.

### C2: traceable TN data

Every TN claim must name its verification scope:

| Scope | Allowed wording | Forbidden wording |
| --- | --- | --- |
| edited text only | traceable counterfactual edit | true negative |
| target box reviewed | target-local verified TN | image-global TN |
| target plus cached proposals reviewed | proposal-covered verified TN | absent from the whole image |
| every deployed fixed Stage-A Top-K candidate reviewed | deployed-candidate-exact TN | all 900 queries or whole-image absence |
| whole image independently reviewed | image-global verified TN | none, if the audit is complete |

The current 17,829-row semantic set is only proposal-covered. Its own audit
sets `cached_proposal_coverage_only=true`,
`all_900_gdino_queries_verified=false`, and
`global_max_label_is_semantic_extrapolation=true`. It cannot support the phrase
"absolutely absent from the image". The strongest currently implementable
model claim is deployed-candidate-exact after the fixed Top-50 extraction and
review protocol in `docs/stage_b_v15_exact_topk_verification.md` is completed.

### C3: edit-aware token supervision

Use the following claim:

> Unlike all-query all-token negative focal supervision, PIVOT assigns
> target-local labels by edit role: positive-expression tokens are positive,
> unchanged TN context remains positive, and only traceably edited TN tokens
> are negative.

The main token result must use provenance-certified single-edit rows. A token
diff computed after independent tokenization is an alignment mechanism, not
proof that the edit itself was verified.

## Fixed experiment contract

The following fields are immutable across a controlled ablation block:

- Stage-A initializer:
  `/media/haoyi/T9/gdino/checkpoint0004.pth`, SHA-256
  `7f4cdd0ab94fc74d46fc7658b2014588a06d7de44be2c1d482ed073bbd7ca1b1`.
- Full-text scorer warm-start:
  `/media/haoyi/T9/gdino/outputs/gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/checkpoint0001.pth`,
  SHA-256
  `b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157`.
  The scorer-only loader copies the last three text-capable decoder layers,
  decoder normalization, and reference-point head; it must not modify the
  Stage-A patch candidate tower.
- Candidate selection: frozen Stage-A patch-logit Top-50, canonical category
  prompt, descending order, normalized `cxcywh` boxes.
- Positive training sources: the official train rows from RefCOCO,
  RefCOCO+, and RefCOCOg.
- Strict evaluation manifests: strict2031 SHA-256
  `0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918`
  and strict1607 SHA-256
  `f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25`.
- Ref evaluation: all eight official RefCOCO-family splits, top-1 IoU >= 0.5,
  plus candidate and all-query oracle recall at 1, 5, 10, 50, and all.
- Evaluation seed 42, AMP enabled, batch size 16, and per-example records.
- Training seeds: 17, 42, and 73. Five seeds may be added for the final two
  headline rows, but may not replace or remove these three.
- No horizontal flip, because left/right text is not rewritten.
- Rank scores select the Ref box. Confidence scores alone produce the
  cross-image global maximum used for FPR95.

Before the first controlled training run, a one-step CUDA memory probe fixes
the largest batch size that finishes with at least 1 GiB headroom. That batch
size and the update count are then identical for every row in the block.
Changing batch size, update budget, or TN exposure creates a new block and
cannot be presented as a single-variable ablation.

## Required tables

### Table A: geometry and role decomposition

| ID | Candidate source | Admission/rank score | Purpose |
| --- | --- | --- | --- |
| G0 | GDINO all queries | full text | historical data-FT baseline |
| G1 | Stage-A all queries | patch canonical | patch geometry/oracle |
| G2 | patch Top-50 | patch canonical only | category-only route |
| G3 | patch Top-50 | full text only | text-only scoring on fixed geometry |
| G4 | patch Top-50 | patch admission + full-text rank | proposed route |
| G5 | text-canonical Top-50 | patch/full-expression rank | true role swap |

Report Ref8 Acc@0.5, Recall@1/5/10/50/all, mean best IoU, and top-query churn.
G2-G5 must consume the same frozen query/box tensor. In addition, report score
changes after controlled color, size, action, spatial, and relation edits while
holding the canonical prompt/support fixed. Those pairs can test patch
invariance and full-text sensitivity, but they cannot test category response.
Category responsibility must instead be supported by Stage-A category AP and
candidate admission, or by a separate rerun that changes the canonical prompt
and support patch consistently with a noun edit. Fixed-canonical noun pairs
must be labelled unsupported for that causal claim.

`tools/eval_stageb_role_causal.py --true_role_swap` implements G5 without
changing the model: a first deterministic forward selects canonical-text
Top-50 from all 900 queries, then a second forward replays those exact query
IDs through the patch/full-expression scorer. The evaluator requires the two
forwards' all-query hidden states, boxes, and patch logits to be bitwise equal
and rejects any candidate-order drift. The synchronized noun-category causal
input is sealed by
`data/ablations/stageb_table_a_category_intervention_20260717/audit.json`;
each A/B arm keeps the image fixed while changing both the canonical category
prompt and its uniquely bound support patch. It is a separate category
intervention, not evidence from fixed-canonical attribute edits.

G0c is the record/exposure-matched continued-GDINO control built by
`tools/build_stageb_table_a_continued_gdino.py` and launched by
`tools/run_stageb_table_a_controls.py`. It uses the same three positive source
row sets, D3 TN row set, `1/1/1/3` sampler mass, effective batch size, update
budget, and no-flip policy, but remains a pure-GDINO training path. G0c uses a
physical per-rank batch of 10 and four micro-batches per optimizer update on
one GPU, so every registered update has effective global batch 40 while its
physical batch remains below the historical pure-GDINO batch-19 setting. The launch
plan records physical batch, accumulation factor, world size, effective batch,
optimizer updates, planned micro-batches, and the zero-AMP-skip requirement
separately, and seals the training entrypoint, engine, and launcher hashes.
The leaf config and its imported pure-GDINO config chain are sealed as separate
inputs as well. The four materialized JSONL files are individually hashed, and
a deterministic source-tree digest covers the GroundingDINO model, loss,
dataset, and utility implementation used at runtime. The launcher removes
inherited torchrun/SLURM rank variables and postflight requires
`world_size=1`, `rank=0`, and `distributed=False`. `max_train_iters`,
periodic checkpoints, AMP clipping/stepping, and OneCycleLR all use successful
optimizer-update boundaries; checkpoints retain both the consumed DataLoader
iteration and cumulative optimizer-update count for exact resume, including
across epoch boundaries. The runner's postflight requires consumed `iteration`
to equal `optimizer_updates * gradient_accumulation_steps`; because skipped
AMP steps consume micro-batches without advancing the update counter, this equality also seals
the zero-skip condition. This is the conventional optimizer effective-batch
contract, not a claim that four sequential micro-batch losses are bitwise
identical to one physical batch-40 forward under every batch-coupled
normalizer; the original GDINO loss is otherwise unchanged. The registered
budget must terminate inside epoch 0; postflight binds both the
expected epoch and micro-batch iteration and rejects empty or structurally
invalid model, optimizer, scheduler, or AMP scaler state. Because pure GDINO has
no proposal scorer, matching source records and expected exposure cannot make
its per-forward training layout identical to the proposed paired-slot loss;
that structural limitation must be reported. The evaluator's deterministic
900-to-50 score-independent query subsets are a global-max multiplicity
diagnostic only, not a replacement architecture or formal G0/G0c result.
These rows enter the paper only after their registered GPU runs complete; a
dry-run plan alone is not experimental evidence.

`tools/run_stageb_table_a_evaluations.py` is the only registered Table-A
evaluation launcher. For a completed candidate training sequence, use:

```bash
python tools/run_stageb_table_a_evaluations.py run \
  --kind candidate \
  --training-run-root outputs/paper_cvpr_v1/token_ablation/L4/seed17 \
  --output-dir outputs/paper_cvpr_v1/table_a/candidate/L4_seed17
```

The candidate command evaluates all eight Ref splits, locked strict2031 edit
pairs, and all 512 synchronized category pairs in one model load with seed 42,
AMP, no batch limit, and `--true_role_swap`. Each Ref split exposes explicit
G1-G5 rows with Acc@0.5, mean selected IoU, route-ranked Recall and mean-best
IoU at 1/5/10/50/all, and top-query churn. G1's `all` domain is all 900
queries; G2-G5 use their registered 50-candidate domains. Postflight requires
the exact Ref split counts/hashes, every required color/size/action/spatial/
relation edit stratum, bitwise fixed-canonical patch invariance, 2,031 aligned
TN records, 1,024 category arms, and rehashes all 512 intervention images and
318 support assets. G5 accepts the canonical-text candidate ordering but still
requires the rerun's all-query states, boxes, and patch logits to be bitwise
equal. The direct G5 forwards run under `no_grad`.

For a completed G0c training plan, use:

The dedicated controller owns the exact `G0c:17/42/73` U1000 training queue
and the exact six-item validation queue (`candidate:17/42/73`, then
`g0c:17/42/73`) under the repository-wide durable GPU lease. Before any formal
creation or launch, the CPU-only readiness checks are:

```bash
/home/haoyi/miniconda/envs/gdino5090/bin/python \
  tools/run_stageb_table_a_g0c_queues.py audit
/home/haoyi/miniconda/envs/gdino5090/bin/python \
  tools/run_stageb_table_a_g0c_queues.py dry-run g0c_training
/home/haoyi/miniconda/envs/gdino5090/bin/python \
  tools/run_stageb_table_a_g0c_queues.py dry-run table_a_validation
```

Training readiness stays blocked until the canonical telemetry-sealed U50 soak
passes semantic replay and all three canonical U1000 roots are fresh. Validation
readiness stays blocked until the dedicated three-item training queue is exactly
completed and replayed. Legacy U50/U1000 artifacts without the current source,
runtime, input, active-item, and job identity closure are audit-only and are not
adopted.

```bash
/home/haoyi/miniconda/envs/gdino5090/bin/python \
  tools/run_stageb_table_a_evaluations.py run \
  --kind g0c \
  --profile validation \
  --g0c-training-plan \
    outputs/paper_cvpr_v1/plans/table_a_g0c_seed17_b10a4_effective40_u1000_v3.json \
  --training-queue-dir \
    outputs/paper_cvpr_v1/queues/table_a_g0c_training_u1000_v1 \
  --output-dir \
    outputs/paper_cvpr_v1/table_a/evaluations/validation/g0c/G0c/seed17
```

G0c evaluation accepts only a plan with a passed G0c training postflight and
an unchanged checkpoint/input hash chain. The validation profile evaluates the
three registered validation Ref splits and sealed calibration manifest without
accessing Ref test or strict TN surfaces. After all six Table-A validation
instances pass and the final gate is sealed, the corresponding final command
adds `--profile final`,
`--final-gate outputs/paper_cvpr_v1/gates/table_a_final_evaluation_gate.json`,
and uses the canonical
`outputs/paper_cvpr_v1/table_a/evaluations/final/g0c/G0c/seed17` output root.
The final profile runs Ref8+strict2031 followed by strict1607 with Ref skipped
and reports top-k 1/5/10/50 plus the all-query oracle. Both profiles write
`launch_manifest.json` and finish only after `postflight.json` binds full
per-example records and replays every aggregate-consumed metric. Run the
launcher itself with the same Python selected for the evaluation child so the
native extension and interpreter bytes are sealed correctly. Use `dry-run`
with the same arguments to validate and print the exact commands without
creating an evaluation output directory or starting a GPU process.

The shared training loop treats an epoch-tail stop as an epoch-finished
checkpoint: criterion epoch hooks and epoch-based LR schedulers advance before
serialization. Stop requests and AMP step-success decisions are synchronized
across distributed ranks at optimizer boundaries, preventing rank-local exit
or scheduler-counter divergence. The default accumulation factor remains one.

### Table B: TN construction

Keep the final architecture and loss fixed. Within each TN comparison block,
equalize sampled row count, sampler mass, and total TN optimizer exposures.
Do not claim full taxonomy matching for the broad D1-D3 block: D1/D2 match the
available color/size/spatial counts, while D3 retains a broader semantic
taxonomy. D2m/D3m instead match edit category through the parent key.

| ID | TN source | Verification scope |
| --- | --- | --- |
| D0 | none | no TN supervision |
| D1 | recovered all-negative data-FT TN | unverified all-negative |
| D2 | traceable edits before visual filtering | traceable counterfactual edit only |
| D3 | semantic 17,829 set after leakage removal | proposal-covered verified |
| D4 | fixed Stage-A Top-50 reviewed set | deployed-candidate exact |

D0-D3 are the broad equal-exposure comparison. They do not by themselves
identify the effect of visual verification: D2 and D3 have different source
populations and D3 has a broader taxonomy. Use the secondary parent-matched
panel for that causal question:

| ID | TN source | Matched unit |
| --- | --- | --- |
| D2m | traceable edit before visual filtering | same dataset, image, sentence, edit category, and positive expression as D3m |
| D3m | proposal-covered verified edit | same dataset, image, sentence, edit category, and positive expression as D2m |

The sealed D2m/D3m train panel contains 7,074 unique parent pairs from 4,431
images. The positive text is exactly identical in every pair. TN text is also
exactly identical in 3,242 pairs and differs in 3,832 pairs. However, 39 of the
text-identical train pairs retain different canonical class IDs, leaving 3,203
`class_aligned_identical_complete_input` pairs as the primary causal stratum.
The 3,832 different-TN pairs still control the parent and edit category, but
the chosen edit realization remains a confound; all 91 train class-ID mismatch
pairs are likewise excluded from the clean causal denominator. Calibration
contains another 770 disjoint pairs: 378 have identical TN text, 3 of those
have a class-ID mismatch, and the clean complete-input denominator is 375.

Report strict2031 and strict1607 exact global-max FPR95, q05 positive
threshold, pair-win rate, ROC AUC, and FPR by edit taxonomy. Report TN data
yield, unique images, edit count, and a human audit with two annotators,
adjudication, and Cohen's kappa. D4 must be omitted rather than approximated if
the exact Top-50 review is incomplete.

The executable D0-D3 data block is sealed by
`data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json` and the
four `config/datasets_stageb_table_b_*.json` manifests. D1-D3 each contain
14,196 TN rows, use TN sampler mass `3.0` against three positive sources of
mass `1.0`, and therefore have 50% expected TN draws. Their runtime-facing
`global_tn_verified` remains false. The v23 config leaves bind the exact table
ID, singleton scope, audit path, and audit SHA; the ablation-only engine path
creates a separate `confidence_ablation_eligible` mask and never upgrades that
label to image-global verification.

Table B keeps the selected v19 base-plus-gate + Acc50-hard-negative + L4
architecture/objective. To prevent D3 from receiving an edit-token advantage,
all D1-D3 dataset manifests set
`require_single_edit_token_provenance=false`. Thus positive token BCE and the
predicate-pair rank term remain fixed, while TN shared/edit-token BCE is
uniformly ineligible. Table C, not Table B, measures certified edit-token
supervision. D0 has no TN source; equal TN exposure applies to D1-D3, not D0.

The formal matched panel is sealed and runtime-enabled separately at
`data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/audit.json`,
with aligned D2m/D3m JSONL rows and a one-row-per-pair ledger. Its generated
dataset manifests preserve the same positive sources and 50% TN sampler mass.
The v24 leaves `cfg_stageb_v24_table_b_d2m_matched.py` and
`cfg_stageb_v24_table_b_d3m_matched.py` bind the distinct 7,074-row v2 audit,
whose SHA-256 is
`5ff62a838a5123d580a72e353147b97bb69e9d7967348b55cba4ccb9ca36cb96`.
The fail-closed runtime recognizes D2m/D3m without aliasing them to D2/D3 and
still requires `global_tn_verified=false`. Because their row count and audit
boundary differ from D1-D3, D2m/D3m form a separate matched comparison block;
they must not be pooled with the broad equal-exposure block as if all five TN
rows differed by one variable.

Use the shared audited launcher for Table B:

```bash
python tools/run_stageb_paper_ablation_matrices.py list --table B
python tools/run_stageb_paper_ablation_matrices.py dry-run --run-id D3:17
python tools/run_stageb_table_b_v2.py list
python tools/run_stageb_table_b_v2.py dry-run --run-id D2m:17
PIVOT_BATCH_SIZE=16 PIVOT_MAX_TRAIN_ITERS=1000 \
  python tools/run_stageb_paper_ablation_matrices.py run --run-id D3:17

python tools/build_stageb_tn_matched_causal_panel.py
python tools/build_stageb_tn_matched_causal_panel.py --verify
python -m unittest tests.test_stageb_tn_matched_causal_panel
```

For a run that must outlive the invoking terminal or agent session, use the
built-in detached orchestrator:

```bash
PIVOT_BATCH_SIZE=16 PIVOT_MAX_TRAIN_ITERS=1000 \
  python tools/run_stageb_paper_ablation_matrices.py detach --run-id D3:17
```

It completes preflight before spawning anything, then launches the exact
`run` command in a new OS session with `stdin=DEVNULL` and stdout/stderr in a
persistent `orchestrator.log`. The returned JSON identifies a job directory
under `outputs/paper_cvpr_v1/orchestration/`: `launch.json` records the spawn,
`plans/` retains preflight manifests, and `status.json` is atomically updated
with the current run/phase through failure or completion. Closing the parent
shell therefore does not own the training process or its output pipe. A manual
`setsid` invocation is equivalent only when stdout and stderr are redirected
to a persistent file.

### Table C: token objective

Keep v19 model, initialization, train rows, sampler, optimizer, and updates
fixed. L0-L3 and L7-L9 disable predicate-pair rank. L4 tests whether pair-level
rank complements the L3 token objective; L5, L6, and L10 are controls around
L4 and keep predicate-pair rank weight `1.0`.

| ID | Token objective | Query surface | TN token targets | Reduction | Pair rank |
| --- | --- | --- | --- | --- | ---: |
| L0 | off | none | none | none | 0 |
| L1 | all-negative focal | all scorer-visible Top-50 queries | every TN token negative | flat mean | 0 |
| L2 | all-negative focal | target-local, IoU >= 0.5 | every TN token negative | flat mean | 0 |
| L3 | edit-aware BCE | target-local, IoU >= 0.5 | shared=positive, edit=negative | flat mean | 0 |
| L4 | edit-aware BCE | target-local, IoU >= 0.5 | shared=positive, edit=negative | flat mean | 1 |
| L5 | edit-aware BCE, uniform roles | target-local, IoU >= 0.5 | shared=positive, edit=negative | flat mean | 1 |
| L6 | edit-aware BCE, no shared-positive term | target-local, IoU >= 0.5 | edit=negative only | flat mean | 1 |
| L7 | all-negative BCE | target-local, IoU >= 0.5 | every TN token negative | flat mean | 0 |
| L8 | edit-aware focal | target-local, IoU >= 0.5 | shared=positive, edit=negative | flat mean | 0 |
| L9 | GDINO loss-form all-negative focal | all scorer-visible Top-50 queries | every TN token negative | dense sum / positive target-query count | 0 |
| L10 | edit-aware BCE | target-local, IoU >= 0.5 | shared=positive, edit=negative | group-balanced role means | 1 |

The default edit-role weights are positive `1.0`, shared TN `0.25`, and edit
TN `1.0`; L5 changes them to `1.0/1.0/1.0`, while L6 sets the shared
coefficient to `0.0`. In the flat reduction, role coefficients are applied to
individual supervised token losses before one common denominator. L10 instead
averages each non-empty role independently before applying its coefficient, so
it is the reduction control for the L4 main row. Hyperparameter selection uses
only the sealed calibration split.

Every focal row uses alpha `0.25` and gamma `2.0`. L1 versus L2 isolates query
scope under the same flat focal scale; L2 versus L7 and L3 versus L8 isolate
the loss family; L2 versus L3 changes both the loss family and token labels and
must not be presented as a single-variable comparison. L9 is only a
fixed-Top-50 analogue of GroundingDINO's dense focal loss form. It preserves
the sum/positive-query normalization but does not reproduce GroundingDINO's
900-query architecture, so it must not be labelled "exact GDINO Stage-B".

The executable Table-C configs are the eleven
`config/ablations/cfg_stageb_v21_token_l*.py` leaves registered as L0-L10 in
`tools/run_stageb_token_ablation_matrix.py`. They all
inherit `cfg_stageb_v21_token_matrix_base.py`, which combines v19's explicit
base-plus-gate confidence contract with v20's Acc@0.5-aligned hard-negative
boundary and fixes every non-allowlisted training knob. The
single shared dataset manifest is
`config/datasets_stageb_v21_single_edit_train.json`; its three positive-source
weights are `1.0` and its TN weight is `3.0`, giving 50% expected TN exposure.
Those TN rows retain the D3 `proposal_covered_verified` scope with
`global_tn_verified=false`; their separately authorized, certified single-edit
provenance is used for token labels and does not upgrade confidence scope.

Use the audited launcher rather than invoking `main.py` directly:

```bash
python tools/run_stageb_token_ablation_matrix.py list
python tools/run_stageb_token_ablation_matrix.py dry-run --run-id L4:17
PIVOT_BATCH_SIZE=16 PIVOT_MAX_TRAIN_ITERS=1000 \
  python tools/run_stageb_token_ablation_matrix.py run --run-id L4:17
PIVOT_BATCH_SIZE=16 PIVOT_MAX_TRAIN_ITERS=50 \
  python tools/run_stageb_token_ablation_matrix.py detach --run-id L1:17
python tools/run_stageb_token_ablation_matrix.py status JOB_DIR
python tools/run_stageb_token_ablation_matrix.py reconcile JOB_DIR
```

`PIVOT_STAGE_A_INIT` and `PIVOT_SCORER_WARMSTART` override the protocol
defaults only when intentionally creating a new comparison block. The launcher
refuses non-fresh run directories under
`outputs/paper_cvpr_v1/token_ablation/<row>/seed<seed>`, records the exact
command and all input hashes in `launch_manifest.json`, streams output to
`train_console.log`, and marks a run complete only after a weights-only
checkpoint metadata audit passes. That audit is also stored in
`postflight.json`. Table-C runs use the same GPU evidence implementation as the
Table-B/D paper runner: pre-run GPU identity, one-second telemetry, finite-loss
and AMP zero-skip checks, plus a fresh SHA-256 rehash of every input persisted
as `input_rehash.json`. A completed single-stage run also writes
`sequence_manifest.json` with its exact batch size and optimizer-update budget,
so L1/L9 two-step ladders and 50-update soaks can be passed directly to
`tools/seal_stageb_memory_probe.py`. Detached jobs retain preflight plans,
orchestrator log/status, PID start-time/boot identity, and support read-only
`status` plus atomic `reconcile`; missing processes or log text never establish
OOM. A `max_train_iters` stop occurs before native epoch aggregation, so
`log.txt` may correctly be absent; `info.txt`, the console log, scorer-init
audit, GPU artifacts, `input_rehash.json`, and `checkpoint_iter.pth` are
mandatory.

Each ladder point must use a distinct `PIVOT_TOKEN_OUTPUT_ROOT`; otherwise the
fresh-output guard correctly rejects the second probe. For either `ROW=L1` or
`ROW=L9`, use two-update lower/upper candidates, then rerun the selected batch
for 50 updates under a third root. Seal only completed evidence, for example:

```bash
python tools/seal_stageb_memory_probe.py \
  --probe lower=LOWER_ROOT/L1/seed17 \
  --probe upper=UPPER_ROOT/L1/seed17 \
  --probe soak=SOAK_ROOT/L1/seed17 \
  --selected soak --expected-row-id L1 \
  --output outputs/paper_cvpr_v1/memory_probes/table_c_l1_batch_contract.json
```

Replace `L1` with `L9` consistently for the full-query GDINO-loss-form family.
The selected soak must contain at least 50 optimizer updates, finite losses,
zero AMP-skipped steps, and at least 1024 MiB sampled free-memory headroom.

### Table D: score decoupling

Keep data and objectives fixed while changing parameter ownership:

| ID | Rank/confidence ownership |
| --- | --- |
| S0 | one shared score |
| S1 | shared decoder trunk, separate output heads |
| S2 | trainable rank decoder plus a frozen independent confidence decoder and trainable validity head, optimized jointly |
| S3 | the same isolated ownership, rank phase followed by confidence phase |

Report Ref8, both FPR95 metrics, rank-loss/confidence-loss gradient cosine on
shared parameters for S0/S1, and cross-task regression after each isolated
update. Do not describe S2/S3 as two trainable trunks: the confidence decoder
is an immutable full-text snapshot and only its validity head trains. S2/S3
must pass the branch-isolation test: rank backward produces no validity-head
gradient and confidence backward produces no rank-decoder gradient.

The clean ownership block is S0-S3. Every one of those rows uses the same data,
fixed Top-50 surface, flat L4 edit-token/predicate-pair objective, and
`stage_b_v15_tail_queue_positive_trust_weight=0.0`. The executable ownership
contracts are the v22 config family:

| ID | Config | Trainable ownership | Confidence base |
| --- | --- | --- | --- |
| S0 | `cfg_stageb_v22_s0_shared_score.py` | rank decoder; phrase score serves both tasks | none |
| S1 | `cfg_stageb_v22_s1_shared_trunk_two_heads.py` | shared rank decoder plus validity head | detached phrase score |
| S2 | `cfg_stageb_v22_s2_independent_joint.py` | rank decoder plus validity head, joint step | frozen independent decoder |
| S3a | `cfg_stageb_v22_s3_rank_phase.py` | rank decoder only | frozen independent decoder |
| S3b | `cfg_stageb_v22_s3_confidence_phase.py` | validity head only | frozen independent decoder |
| S3 probe | `cfg_stageb_v22_s3_isolation_probe.py` | rank decoder plus validity head; throwaway preflight only | frozen independent decoder |
| S2F | `cfg_stageb_v22_s2_independent_joint_full.py` | rank decoder plus validity head, joint step | frozen independent decoder |

S0-S3 retain the global-max q05 negative term, paired positive/TN separation,
all rank/token losses, the `0.499` negative-IoU boundary, and predicate-pair
rank weight `1.0`. S0/S1 have no candidate-invariant broadcast gate, so v19's
positive-residual trust/translation term has no corresponding parameter; the
clean comparison therefore disables that gate-specific term for S2/S3 as well.
S2F restores trust weight `1.0` on the S2 architecture and is the separately
labelled full-v19-objective row. S2F is not part of the single-variable
ownership contrast and must not be pooled with S0-S3 as a fifth ownership row.
Historical v20 alone is gate-only and is not the provenance label for S2. S3
uses the same isolated architecture as S2 and splits the fixed optimizer-update
budget exactly in half: with the declared 1,000-update paper budget this is 500
rank updates followed by 500 confidence updates. These are deliberate
mid-epoch stops, not four-plus-four epoch training. Phase 2 must load the
completed phase-1 model state, not reapply the scorer warm-start.

Set `stage_b_v22_gradient_diagnostic_interval` to a positive probe interval.
S0/S1 then log rank/confidence gradient cosine, both norms, elementwise and
per-tensor conflict fractions, and the number of shared tensors/elements. The
diagnostic fails if no shared parameter exists. S2 logs a bidirectional
branch-isolation result and fails on any structural cross-gradient, including
a numerically zero gradient tensor that remains connected by autograd. S3
uses the same structural graph as S2, and its two phase configs additionally
exclude the inactive branch from the optimizer. Run the S3 isolation-probe
config for one throwaway batch before phase 1; it exposes both trainable
branches to autograd and must not be used as an experiment checkpoint.

Historical names are not ownership labels: v12 has only a rank score and no
Table-D confidence objective; v14 routes its validity score as both rank and
confidence and is closest to S0, not S1; v15/v19 freeze the copied confidence
decoder and train only its validity gate.

Before Table-D launch, the runner verifies that S0-S3, including both S3 phases
and its isolation probe, expose the same trust weight `0.0` and common-objective
contract. `dry-run`, `run`, and `detach` fail before creating training output
if that equality drifts. S2F is validated separately with trust weight `1.0`
and objective fidelity `full_v19_base_plus_gate_objective`; its different
objective is intentional and explicit rather than an ownership confound.

Seal Table-D runtime readiness on S2, the sustained worst-memory ownership row:
each optimizer update retains the trainable rank-decoder graph and validity-head
graph while also executing the frozen confidence decoder. S0/S1 have no frozen
independent decoder and the two S3 training phases optimize only one branch at a
time. The S2 two-update ladder must set the gradient-diagnostic interval to one,
so its measured peak also includes the bidirectional branch-isolation probe used
by the throwaway S3 preflight. Probe batch 32 before batch 40 on seed 17; only
the largest candidate that completes both updates with finite losses, zero AMP
skips, passed postflight, and at least 1 GiB free GPU memory may proceed to a
50-update soak. Run the soak with diagnostic interval 10 and seal the completed
sequence with `tools/seal_stageb_memory_probe.py --expected-row-id S2`. S2F has
the same scorer graph plus the active trust term; confirm it for two updates at
the selected batch before launching the full objective-completion row. Any
failure lowers the batch for the affected comparison block rather than silently
changing update count or gradient accumulation.

The same launcher executes Table D and treats `S3:<seed>` as one atomic
sequence: a one-update isolation probe (excluded from the paper update
budget), rank training for half of `PIVOT_MAX_TRAIN_ITERS`, then confidence
training for the other half. The confidence phase consumes
`rank/checkpoint_iter.pth` through `--pretrain_model_path`, does not pass
`--resume`, starts a fresh optimizer/scheduler/scaler, and must not reapply the
scorer warm-start. An odd total update count is rejected.

```bash
python tools/run_stageb_paper_ablation_matrices.py list --table D
python tools/run_stageb_paper_ablation_matrices.py dry-run --run-id S3:17
PIVOT_BATCH_SIZE=16 PIVOT_MAX_TRAIN_ITERS=1000 \
  python tools/run_stageb_paper_ablation_matrices.py run --run-id S3:17
```

The Table-D memory plans are themselves dry-run artifacts. Use a fresh output
root for every rung and do not launch the 50-update command until the two-update
ladder has selected its batch:

```bash
PIVOT_BATCH_SIZE=32 PIVOT_MAX_TRAIN_ITERS=2 \
PIVOT_ITER_CHECKPOINT_INTERVAL=2 PIVOT_NUM_WORKERS=2 \
PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL=1 \
PIVOT_SCORE_OUTPUT_ROOT=outputs/paper_cvpr_v1/memory_probes/table_d_s2_b32_w2_iter2 \
  python tools/run_stageb_paper_ablation_matrices.py dry-run --run-id S2:17 \
  --manifest outputs/paper_cvpr_v1/plans/table_d_s2_b32_w2_iter2_20260717.json

PIVOT_BATCH_SIZE=40 PIVOT_MAX_TRAIN_ITERS=2 \
PIVOT_ITER_CHECKPOINT_INTERVAL=2 PIVOT_NUM_WORKERS=2 \
PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL=1 \
PIVOT_SCORE_OUTPUT_ROOT=outputs/paper_cvpr_v1/memory_probes/table_d_s2_b40_w2_iter2 \
  python tools/run_stageb_paper_ablation_matrices.py dry-run --run-id S2:17 \
  --manifest outputs/paper_cvpr_v1/plans/table_d_s2_b40_w2_iter2_20260717.json

# Replace 40 with the largest passing ladder batch if needed.
PIVOT_BATCH_SIZE=40 PIVOT_MAX_TRAIN_ITERS=50 \
PIVOT_ITER_CHECKPOINT_INTERVAL=50 PIVOT_NUM_WORKERS=2 \
PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL=10 \
PIVOT_SCORE_OUTPUT_ROOT=outputs/paper_cvpr_v1/memory_soaks/table_d_s2_b40_w2_u50_20260717 \
  python tools/run_stageb_paper_ablation_matrices.py dry-run --run-id S2:17 \
  --manifest outputs/paper_cvpr_v1/plans/table_d_s2_b40_w2_u50_20260717.json
```

For both tables, each phase gets a launch manifest with config/data/code and
initializer hashes, a scorer-init audit where applicable, a weights-only
checkpoint postflight, and a full post-run SHA-256 rehash of every input.
The rehash is independently persisted as `input_rehash.json` before later
derived-output checks. For S3 confidence training, its input set contains the
exact rank-checkpoint digest; checkpoint metadata must name that file as
`pretrain_model_path`, record an empty `resume`, and contain no new scorer-init
request or audit.
Paper runs also retain `gpu_environment.json`, one-second
`gpu_telemetry.csv`, and `gpu_telemetry_summary.json` with GPU identity,
driver/PyTorch/CUDA versions, peak used memory, and minimum free memory. The
postflight rejects mismatched pre-run and sampled GPU UUID/name/driver/memory.
It requires finite losses, positive finite AMP scales, and zero AMP-skipped
steps in both the displayed meter value and its global average, rather than
checking only the sliding-window median.
Torch max-allocated memory is recovered from the native training log; max
reserved memory is marked unavailable until the checkpoint/log schema emits
it, rather than inferred from device-level telemetry.

## Selection and statistics

Development selection uses RefCOCO-family validation splits and the sealed TN
calibration partition only. strict2031, strict1607, and Ref test splits are run
once for a predeclared checkpoint selection rule.

For each metric report mean and sample standard deviation over training seeds.
For the headline comparison also report a paired image-cluster bootstrap 95%
confidence interval from aligned per-example records. Training-seed variance
and bootstrap uncertainty are different and both are required. A result is
incomplete if any expected seed, split, manifest binding, or record identity is
missing.

The primary acceptance gate is stronger than a mean-only result:

1. lower FPR95 on both strict2031 and strict1607;
2. higher mean Ref8 Acc@0.5;
3. no statistically material collapse on any individual Ref split;
4. no positive-q05 collapse hidden by lower raw negative scores; and
5. all provenance, leakage, and branch-isolation audits pass.

## Execution order

1. Seal leakage-free train/calibration TN files and strict single-edit token
   subsets.
2. Complete baseline Ref8 and strict evaluations with per-example records.
3. Run one-step memory and gradient-isolation probes for every architecture
   family.
4. Run one seed of L0-L4 as a screen; reject numerically invalid objectives.
5. Run the complete three-seed L0-L10 matrix and the broad D0-D3 block.
6. Run D2m/D3m as its separate matched block and report identical-TN and
   different-TN strata separately.
7. Complete role/causal diagnostics and the D4 extraction/review if available.
8. Train S0-S3 for the clean ownership block and S2F as the separately labelled
   full-objective control for all seeds.
9. Aggregate only sealed runs and render the final paper tables.
