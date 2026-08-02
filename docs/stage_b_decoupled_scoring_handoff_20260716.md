# Stage-B Decoupled Scoring Development Handoff

Date: 2026-07-16

> Historical pause record: the missing-artifact status below describes the
> earlier 2026-07-16 pause point. The later composite architecture and recorded
> same-protocol gate result are documented in
> [`stage_b_architecture_vs_gdino.md`](stage_b_architecture_vs_gdino.md).

## Status

Work is paused before retraining at the user's request. The target has not
passed its acceptance gate:

1. Global FPR@95TPR must be lower than the completed fixed pure Stage-B
   data-FT baseline on both strict2031 and strict1607.
2. Every one of the eight official RefCOCO-family splits must be higher than
   the same baseline under the same checkpoint-independent evaluator contract.
3. The final comparison must use identical manifests, sample order, score
   definition, AMP mode, batch size, seed, and per-example record schema.

The Ref evaluator currently reports top-1 IoU >= 0.5 accuracy (`acc50`).
This is the concrete metric used by the protocol when this document says
"Ref AP/acc50"; it is not COCO-style area-under-curve AP.

No formal baseline or candidate result currently exists. All files under
`outputs/` were removed while this work was in progress, so a checkpoint or
metric mentioned as a diagnostic observation below is not a reusable artifact.

## Main Conclusion

The evidence supports the user's hypothesis: localization is not the primary
limitation. Query selection and expression confidence are.

Stage A already showed high recall even with an object plus incorrect text
prompt. Before its files were removed, an incomplete fixed Stage-B diagnostic
checkpoint was also inspected on three Ref splits. The observed top-1 and
all-query oracle ceilings were:

| Split | Observed top-1 acc50 | Best-of-900 oracle acc50 | Recoverable top-1 errors |
| --- | ---: | ---: | ---: |
| RefCOCO val | 0.665682 | 1.000000 | 3,622 / 3,622 |
| RefCOCO testA | 0.737317 | 1.000000 | 1,486 / 1,486 |
| RefCOCO testB | 0.575859 | 0.999607 | 2,159 / 2,161 |

The same diagnostic run observed these eight top-1 values:

| Split | acc50 |
| --- | ---: |
| RefCOCO val | 0.665682 |
| RefCOCO testA | 0.737317 |
| RefCOCO testB | 0.575859 |
| RefCOCO+ val | 0.725414 |
| RefCOCO+ testA | 0.816102 |
| RefCOCO+ testB | 0.637554 |
| RefCOCOg val | 0.773897 |
| RefCOCOg test | 0.775359 |

These numbers are non-authoritative. They came from an interrupted
`next_iter=26000` checkpoint rather than the required completed 26,519-step
baseline, and their record files no longer exist. Their useful conclusion is
only that the correct box is almost always already among the 900 queries.

The model boundary should therefore remain:

```text
boxes(q)          = frozen localization output
base_score(q, e)  = frozen GDINO full-expression score
rank_score(q, e)  = base_score(q, e) + query_specific_rank_residual(q, e)
confidence(q, e)  = base_score(q, e) + image_expression_gate(image, e)

Ref prediction    = boxes(argmax_q rank_score(q, e))
FPR score         = max_q confidence(q, e)
```

The patch branch, if reintroduced later, should provide proposals or frozen
features only. It must not multiply or overwrite either deployed score without
a new, separately audited contract.

## Why Earlier Separation Underperformed

### 1. FPR@95TPR is a tail-ordering metric

For each positive expression and TN expression, deployment first takes the
maximum over all 900 queries:

```text
p_i = max_q score(image_i, positive_expression_i, q)
n_i = max_q score(image_i, negative_expression_i, q)
t95 = exact lower-tail threshold that retains 95% of p_i
FPR95 = mean(n_i >= t95)
```

A shared threshold, temperature, affine calibration, or any common strictly
increasing transform cannot change this ordering. Earlier branches often
reduced score magnitude while leaving negatives above the positive q05 tail.
That can improve BCE, mean separation, pair win, or AUC while FPR95 stays high.

### 2. Training scored the wrong query surface

Several earlier objectives supervised the target query or a cached proposal
subset. Deployment uses the largest score from any of 900 queries. The strict
diagnostics showed a large global-minus-target gap, so a different query could
dominate the deployed negative score even when the supervised target improved.

The current confidence loss fixes the numerical surface by taking the maximum
over all 900 confidence scores. Data labels still have to match that surface.

### 3. The negative labels did not match the deployed maximum

The recovered 60k allTN pairs reproduce the fixed data-FT recipe, but they are
not image-global verified negatives. They are concentrated in RefCOCO train
and in color/spatial edits, while the strict sets contain broader taxonomies.

The later semantic-verified set is stronger, but its own audit deliberately
states:

```text
cached_proposal_coverage_only = true
all_900_gdino_queries_verified = false
global_max_label_is_semantic_extrapolation = true
```

It has 17,829 rows on 9,317 images and is image-disjoint from strict1607, but
it only proves target-plus-cached-proposal coverage. Treating it as supervision
for the 900-query maximum still requires generalization beyond the verified
regions. That mismatch can directly create false positives.

### 4. The positive q05 tail collapsed

The old queue-q05 straight-through objective drew its gradient from a very
small current positive batch and retained stale early history. It could lower
negative scores by shifting positive and negative gates down together. The
operating threshold then moved down with the positives, leaving FPR high.

The replacement objective detaches the recent q05, cancels the common gate
translation gradient, and adds a per-positive trust hinge. This repairs the
loss estimator, but it cannot repair mislabeled or out-of-domain TN data.

### 5. Ranking and confidence were coupled by optimization and evaluation

Ref accuracy needs a query-specific change: the correct existing box must move
above competing boxes. FPR needs an image-expression decision: the negative
expression's global maximum must fall below the positive lower tail.

A uniform confidence offset cannot repair Ref query order. A query-specific
rank residual can change which query wins and therefore must not feed the FPR
score. Joint objectives, shared trainable trunks, or evaluators that consume
the wrong output make one metric regress while optimizing the other.

### 6. Interrupted DDP resume was not an authoritative continuation

The interrupted baseline saved one checkpoint payload through rank 0. Its RNG
payload contains only the process-local Python, NumPy, Torch, and CUDA state
that rank 0 evaluated before `save_on_master`. Resuming both ranks from that
single payload cannot reproduce both pre-interruption rank-local RNG streams.
The old partial checkpoint therefore could not be promoted as the fixed
baseline even before it was deleted.

## Implemented Work

### Decoupled adapter

`models/GroundingDINO/stage_b_gdino_score_adapter.py` now provides an
identity-initialized adapter with two independent outputs:

- `rank_score`: frozen base score plus a per-query rank residual.
- `confidence_score`: frozen base score plus one uniform
  image-expression gate.

Inputs are detached at the adapter boundary. Rank-only training owns only rank
parameters; confidence-only training owns only confidence parameters. The
evaluator uses rank scores for Ref top-1 and confidence scores for TN FPR.
Checkpoint loading fails closed on missing, unexpected, or shape-incompatible
adapter state.

### Rank objective

The rank-only path uses the three positive RefCOCO, RefCOCO+, and RefCOCOg
training sources. Its baseline-preserving loss separates:

- repair rows, where the frozen top-1 query is wrong but a positive query
  exists; and
- preservation rows, where the frozen top-1 query is already correct.

The two row classes are normalized independently across DDP ranks so rare
repair rows are not diluted by easy preservation rows. Correct rows may lose
at most 0.02 of their frozen positive-negative margin, and a residual L2 term
limits unnecessary score movement.

### Confidence objective

The current P3 objective is `detached_recent_q05_trust`:

```text
t = detach(q05(recent positive max-score queue))
t_loss = t + mean(positive_gate) - detach(mean(positive_gate))
Lneg = mean softplus((negative_global - t_loss + margin) / temperature)
Lpair = mean softplus((negative_global - positive_global + pair_margin) / temperature)
Ltrust = mean relu(-0.02 - positive_gate)
L = Lneg + 0.25 * Lpair + 1.0 * Ltrust
```

The queue holds 512 recent positive scores and activates at 256. Every current
negative contributes to the loss. The forward threshold is detached, the
zero-valued proxy removes the shared-translation shortcut, and the linear
trust hinge protects the positive lower tail.

### Input-label safeguards

Adapter pair recipes now set:

- `data_aug_hflip_prob=0.0`, because the existing horizontal flip does not
  rewrite left/right referring expressions; and
- `neg_episode_prob=0.0`, so the dataset cannot replace a sealed pair with
  another internally sampled negative episode.

Scope codes and required verification flags are checked by the criterion
instead of silently accepting mixed TN semantics.

### Reproducible baseline and final gate

The following protocol surfaces were added:

- `tools/run_stageb_fixed_baseline.sh`
- `tools/stageb_fixed_protocol_audit.py`
- `tools/run_stageb_fixed_protocol_eval.sh`
- `tools/run_stageb_fixed_dual_gate.sh`
- `tools/verify_stageb_dual_gate.py`

They lock the Stage-A initializer, fixed Stage-B config and five data sources,
code/config dependency closures, world size 2, per-GPU batch 9, AMP, manifests,
record order, and evaluator contract. The dual gate requires lower FPR on both
strict sets and higher acc50 on all eight Ref splits. It does not accept a
historical headline metric or a comparison from different records.

### Two-phase probes and recovery audits

Rank and confidence phases have independent launchers, milestone checkpoints,
optimizer ownership checks, recursive lineage audits, and exact same-phase
resume rules. P0 constructs an identity adapter from the fixed baseline and
requires bitwise score and record parity before any trained adapter may be
interpreted.

Relevant entry points include:

- `tools/run_stageb_gdino_adapter_two_phase_probe.sh`
- `tools/run_stageb_gdino_semantic_confidence_probe.sh`
- `tools/run_stageb_gdino_fixed_top1_confidence_probe.sh`
- `tools/make_stageb_gdino_adapter_p0.py`
- `tools/run_stageb_gdino_adapter_probe_eval.sh`

### Fixed-baseline top-1 verification path

A checkpoint-specific data path was implemented to remove the remaining
confidence-label mismatch:

1. Replay the exact frozen fixed baseline under the train primary, train
   shadow, and formal deploy transforms.
2. Extract every top-1, disagreement, and near-tie region that can realize the
   frozen global maximum.
3. Render deterministic crops and boxed context images.
4. Judge the union with a pinned Qwen contract.
5. Accept only rows whose full region union is negative.
6. Bind accepted rows to the exact checkpoint hash and both transform
   contracts; they are not portable to another detector.
7. Remove all strict2031 and strict1607 image overlap.
8. Create a sealed image-disjoint train/calibration partition and select the
   confidence milestone on calibration records, not on either strict set.

This is valid for the current uniform confidence gate because adding the same
gate to every query cannot change the frozen base-score argmax. It would not
remain valid if confidence became query-specific.

The extraction, verification, partition, calibration, and training tools are
implemented, but no production fixed-top1 data was generated because the
completed baseline does not exist.

### Runtime smoke

`tools/smoke_stageb_fixed_gdino_top1_runtime.py` replays the exact production
surface before expensive extraction:

- train batch 0 at B=4 with paired and separate calls;
- deployment batches 0 through 48, including exact rows 768-783 at
  `(16, 3, 1333, 1333)`;
- CUDA AMP at the actual model-call boundary;
- exactly 900 finite query scores and boxes;
- at least 1 GiB total and observed system memory headroom;
- private checkpoint/config copies and pre/post input seals;
- canonical-class and selected-image seals;
- direct-child, non-symlink output confinement and one-winner atomic report
  publication.

The current targeted smoke suite passes 14/14 tests and both files pass
`py_compile`. The private snapshot now synthesizes and records an empty
`config/__init__.py` marker, requires that marker before parsing, and verifies
that every loaded `config.*` module originates inside the snapshot. The test
starts with a real markerless namespace tree plus a malicious later regular
`config` package, requires rejection, then proves the marked snapshot wins.
The CUDA production smoke itself remains unrun because its completed baseline,
dataset paths, and free GPU are unavailable.

## Current Artifact State

| Artifact | Current state |
| --- | --- |
| OGC initializer | Present: `weights/groundingdino_swint_ogc.pth`, SHA-256 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` |
| Stage-A ODVG label map | Present, SHA-256 `56bc4800ead2ad97dc7d27fee9ac515f89748ce17acaf5acb4c6664d6dffedf4` |
| Stage-A LVIS JSONL | Present, 99,388 rows, SHA-256 `3177e55d55cda0ebde3b2905b89ddc9d26d7fd2990fbafd8229c181f22c0f136` |
| Stage-A COCO JSONL | Present, 117,266 rows, SHA-256 `8c8d84b32e64e57e629d0899dfaa65ed4cda4341c7fdb25c8d74cf0aa7cfd5b5` |
| strict2031 | Present, 2,031 rows, SHA-256 `0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918` |
| strict1607 | Present, 1,607 rows, SHA-256 `f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25` |
| Semantic verified pairs | Present, 17,829 rows, SHA-256 `bea2aca85d207d883da85cb219420f748a65a840516218731811e8e46449b645` |
| Stage-A checkpoint | Missing; `outputs/` was cleared |
| Completed fixed Stage-B baseline | Missing; `outputs/` was cleared |
| Formal baseline Ref8/strict records | Missing |
| Adapter rank/confidence checkpoints | Missing |
| Fixed-top1 extraction and verified partition | Not generated |
| `/home/user/datasets/pivot_data` | Missing |
| `/home/user/datasets/vision_benchmarks` | Missing; retained Stage-A JSONLs contain absolute image paths under this tree |

At the pause point both 24 GiB GPUs were occupied by unrelated processes. No
process was stopped, and no new training was launched.

## Next Development Order

### P0. Repair prerequisites and reproducibility

1. Restore the exact `/home/user/datasets` mount or path mapping. Restoring
   only `DATA_ROOT` is insufficient because the retained Stage-A ODVG JSONLs
   contain absolute paths under
   `/home/user/datasets/vision_benchmarks/COCO_2017/train2017`.
2. Run the completed runtime smoke on CUDA after the baseline and dataset
   paths exist. The namespace-package isolation fix and targeted CPU tests are
   already complete.
3. Either make mid-epoch DDP checkpoints store and restore RNG state per rank,
   or declare the fixed baseline non-resumable and always restart it cleanly.
   A per-rank RNG payload is preferable for the later milestone probes.
4. Use durable detached launch supervision and preserve completed checkpoints
   outside the disposable experiment-output directory with a hash sidecar.

### P1. Rebuild the authoritative fixed baseline

After the data paths and one free GPU are restored, rebuild the exposure-matched
two-epoch Stage A from the retained ODVG data:

```bash
DATA_ROOT=/home/user/datasets/pivot_data \
STAGEA_DATASETS=/home/user/PIVOT/config/datasets_patch_stage_a_lvis_coco2017_local.json \
PRETRAIN_MODEL_PATH=/home/user/PIVOT/weights/groundingdino_swint_ogc.pth \
OUTPUT_DIR=/home/user/PIVOT/outputs/ogc_original_finetune_stage_a_rebuild_20260711 \
ODVG_OUT_DIR=/home/user/PIVOT/data/ablations/ogc_original_finetune_stage_a_20260711 \
MATCH_EPOCHS=2 BATCH_SIZE=12 PYTHON_BIN=/usr/bin/python3 \
CUDA_VISIBLE_DEVICES=<free_gpu> \
tools/run_ogc_original_finetune_stage_a.sh
```

Verify `checkpoint0001.pth` has `epoch=1`,
`epoch_finished=true`, and `iteration=0`. Then, with both GPUs free, run
the fixed one-epoch Stage-B baseline from a fresh output directory:

```bash
DATA_ROOT=/home/user/datasets/pivot_data \
CUDA_VISIBLE_DEVICES=0,1 \
tools/run_stageb_fixed_baseline.sh
```

Do not resume the old interrupted run. Require
`protocol_train_complete.json` before any baseline evaluation.

### P2. Establish baseline and identity parity

1. Run the fixed Ref8, strict2031, and strict1607 evaluation once.
2. Preserve all per-example records and completion seals.
3. Create P0 from the authoritative baseline.
4. Require exact baseline/P0 parity for rank score, confidence score, Ref
   records, and TN records.

This determines the real thresholds to beat. The historical FPR 0.528302 and
the deleted R26000 observations are not acceptance baselines.

### P3. Improve Ref query ranking only

1. Train R with frozen boxes, frozen base GDINO, and confidence disabled.
2. Keep the repair/preservation class normalization.
3. Report separately:
   - base-wrong repair rate;
   - base-correct regression rate;
   - all-query oracle ceiling;
   - top-1 acc50 on every Ref split;
   - residual magnitude and top-query churn.
4. Select a rank milestone only if every Ref split improves over P0. A mean
   gain is not sufficient.
5. Confirm confidence scores and strict FPR records remain bitwise identical
   to the rank source.

If R still fails despite the near-1.0 oracle ceiling, the next change should be
hard-negative listwise ranking over the frozen query set, with extra weight on
base-wrong rows and explicit distillation of the complete base ordering on
base-correct rows. Do not unfreeze box regression first.

### P4. Build checkpoint-exact confidence data

Start the fixed-top1 pipeline after P2 and run it in parallel with rank
training. Extract regions from the exact frozen base checkpoint used by the
confidence score, never from an R checkpoint. The current confidence output
ignores R, so the fixed pure baseline is the correct region source and the
verified data does not need to wait for rank-milestone selection.

Use the implemented sequence:

1. `tools/smoke_stageb_fixed_gdino_top1_runtime.py`
2. `tools/extract_stageb_fixed_gdino_top1_vlm_manifest.py`
3. `tools/judge_stageb_fixed_gdino_top1_qwen.py`
4. `tools/verify_stageb_fixed_gdino_top1_vlm_results.py`
5. `tools/stageb_gdino_fixed_top1_selection.py`

Reject unstable, ambiguous, strict-overlapping, or partially judged rows.
Retain the 17,829 semantic pairs only as an ablation; do not promote a model
trained solely on extrapolated cached-proposal labels.

### P5. Train and select confidence only

1. Initialize S from the selected audited R checkpoint using
   `--pretrain_model_path`, creating a fresh optimizer and q05 queue.
2. Train only the uniform confidence gate with the P3 objective.
3. Use the sealed image-disjoint calibration partition for milestone selection.
4. Log positive/TN q01, q05, q50, q95, q99, exact FPR95, threshold, taxonomy
   FPR, gate quantiles, pair win, and AUC. Select by exact calibration FPR95,
   not pair win or BCE.
5. Require the positive q05 not to collapse relative to P0.
6. Confirm rank tensors and all Ref records remain bitwise unchanged across S
   milestones.

If P3 still has high calibration FPR, change data weighting before adding model
capacity: balance edit taxonomies, mine high-scoring fixed-top1 negatives on
official train images, and weight negatives near the detached q05 boundary.
Only after those checks should the uniform gate be replaced by a richer
expression-existence head. A query-specific confidence head would invalidate
the fixed-top1 verification proof and require a new label contract.

### P6. Run the final dual gate

Evaluate only the preselected candidate on the unchanged Ref8, strict2031, and
strict1607 protocol, then run:

```bash
tools/run_stageb_fixed_dual_gate.sh \
  outputs/gdino_ft_stage_b_fixed_baseline_20260711_eval_fixed \
  outputs/<candidate>_eval_fixed \
  outputs/<candidate>_vs_fixed_baseline_dual_gate
```

Success requires:

- candidate FPR95 strictly lower on strict2031;
- candidate FPR95 strictly lower on strict1607;
- candidate acc50 strictly higher on all eight Ref splits; and
- complete protocol, checkpoint, manifest, runtime, and record parity.

Until that command passes, the goal remains open.

## Related Documents

- `docs/stage_b_architecture_vs_gdino.md`
- `docs/stage_b_fpr95_failure_analysis_20260711.md`
- `docs/stage_b_fixed_baseline_protocol.md`
- `docs/stage_b_gdino_adapter_two_phase_probe.md`
- `docs/stage_b_gdino_semantic_confidence_probe.md`
- `docs/stage_b_final_dual_gate.md`
