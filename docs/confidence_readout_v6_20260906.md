# Confidence readout v6 — completed mechanism study

Status: COMPLETE. All 18 new heads, both localizers' val and gRef records,
and all three full 5,000-replicate analyses are complete. The completion
supervisor exited with code zero. GPU work has ended; no detector or head
is being retrained. The v6 main/supplement PDFs are compiled and visually
inspected; v5 remains unchanged.

Start with [the final results and interpretation](readout_v6_final_results_20260906.md),
[main PDF](../paper/empirical_study_v6.pdf),
[supplement PDF](../paper/empirical_supplement_v6.pdf), and
[record-only reproduction](readout_v6_reproduction.md).
The [experimental completion receipt](../paper/data/readout_v6/experimental_completion.json),
[successful terminal](../paper/data/readout_v6/completion_terminal.json), and
[paper build receipt](../paper/data/readout_v6/paper_build_receipt.json) are
separate, explicit completion evidence. Completing the study does not assert
venue compliance or guarantee publication.
The completed first-localizer validation study is described in
[the staged result report](readout_v6_mm_stage_results_20260906.md); it is not a
substitute for the final two-localizer study. The earlier v5 scientific
readiness judgment is superseded by this controlled readout study. Published
v5 sources, PDFs, results and receipts remain unchanged.

## Execution receipts

The scientific/training protocol is sealed at:

`/mnt/why/PIVOT/outputs/arrow_confidence_readout_v6_20260905/protocol.json`

SHA-256: `bc39843bd4694d80dee3e623bcb40f05f30f9df92083d5a3937dfcf1b4093a1e`.
The lightweight local copy is [confidence_readout_v6_protocol.json](../paper/data/confidence_readout_v6_protocol.json).
Metric implementation is separately locked before scoring in
`analysis_code_lock.json`, SHA-256
`a3726b54c81cb1dd46c11d1df546b72dc565081778994a6781cfaa6ba5944f5d`.

The final three analysis SHA-256 values are:

| Surface | SHA-256 |
| --- | --- |
| FineCops val | `91a088208d0a2786f9a07de2d9248ffb0a273c0e78c32f5de7990e4acc03196d` |
| gRef Full | `7d8c2b9db2fdf90b9e9b9218383afef4a51b17089763ca85f87742963b33c5f9` |
| gRef source-disjoint | `33cbd5765b6cf06dcc11fb21545aeb0456f839d0c5c586baf47122baccb674fd` |

The original three-process formal bootstrap is authoritative. The optional
new per-localizer parallel wrapper was tested on synthetic data but never
replaced a formal run. The MM FineCops localizer block is exactly identical
to its earlier completed staged block, including all paired intervals.

MM-GDINO seeds 17, 42 and 73 have completed all six selected-query heads,
with 12,575 updates each, successful terminal exits and ownership postflights.
Their fixed checkpoint panel SHA-256 is
`4e54484bf99da066515d097f3e2baff0c3909b801dd75d899da202e89a259fac`.
Seed 73 completed separately under the identical recipe. Its first formal
attempt was killed while loading cache, before any parameter update: the
container's actual cgroup-v1 memory limit is 200 GiB, despite host-level
memory reporting about 1.3 TiB. Three independent copies of the 900-query
cache exceeded that limit. This was a host-memory failure, not a CUDA OOM
or a metric-dependent stopping decision. The failed attempt and the two
completed seeds are retained; no batch, optimizer or head changes were made.

An earlier launch was intentionally stopped before training because the
launcher resolved the virtual-environment Python symlink to the base Python.
The launcher now preserves the virtual-environment path. Both launch logs
and terminal exit records are retained under `launches/`.

Completion is determined by `postflight.json` plus a successful terminal exit,
not by a checkpoint's existence or a progress line. `seal_confidence_readout_heads.py`
requires all six new MM heads and all twelve new MDETR heads before gRef access.
All eighteen have now passed this gate; the lightweight
[all-head seal](../paper/data/readout_v6/all_heads_sealed.json) binds every
postflight and successful execution. Both localizers used the unchanged
five-epoch / 12,575-update recipe. No target/readout arm was dropped or
selected based on results.

An independent endpoint audit loaded the small initial and final head
checkpoints and verified all 18 heads: 12,575 Adam steps, eight optimizer
states, finite moments, LR `1e-4`, WD zero, and unchanged frozen rank tensors.
Same-seed initialization and epoch permutations match across localizers.
All six training-statistics streams contain exactly 83,341 unique positives;
independent population-mean/SD recomputation matches the sealed SIRC statistics
within `1e-12`. Full record streams pass Native/geometry, readout and seed
parity checks. The 97-test runtime regression is saved in
[runtime_regression.xml](../paper/data/readout_v6/runtime_regression.xml).

### Data-adapter corrections before MDETR head training

The initial MDETR cache adapter stopped at negative-text rows because the
historical COCO-format parser retains a bbox for every row. A full train/val
annotation audit established that every negative-text bbox equals its positive
parent's reference bbox: 80,451/80,451 train and 9,029/9,029 val. This reference
is not valid GT for the edited no-target expression. The v2 adapter preserves
it as inactive annotation-reference metadata and emits an empty study-GT
tensor for negatives. Targets, requests, weights, scores and the head recipe
are unchanged. Completed positive-only query shards are preserved and may be
re-enveloped only with per-tensor bitwise reuse verification. No MDETR head
or gRef transfer ran before this correction; the failed v1 cache remains.

A separate, unused export field initially named annotation `level` as
`negative_edit_level`. The append-only record-v2 export restores official
`negative_level` and separately retains `raw_annotation_level`. Per seed,
7,959 val rows change that field from 1 to null, 112 from 1 to 2, and 958
remain 1. Scores, Native labels/geometry, winner diagnostics, sample identities,
positive difficulty and linked-parent difficulty are unchanged. The analysis
does not consume either edit-level export field, so this metadata correction
does not change any fitted head, score or statistical input. Original records
and analysis inputs remain immutable.

### gRef loader compatibility

The first new gRef worker launch stopped during MMEngine checkpoint loading,
before any detector forward: PyTorch 2.8 defaults `torch.load` to
`weights_only=True`, whereas this project-owned, SHA-verified MM checkpoint
also carries trusted MMEngine `HistoryBuffer` metadata. The historical gRef
run already used the same bounded compatibility setting, recorded in
[its loading recovery receipt](../paper/data/gref_fixed_targets_loading_recovery.json).
The new completion subprocess therefore inherits
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, with the exact `env` command captured by
`launches/completion_loading_compat/launch.json`. Code, weights, scores and
thresholds are unchanged; no persistent system environment or package was
modified. The ordinary all-GPU non-gRef parity barrier remains mandatory.

## Scientific question

Why can correct-emission supervision be insufficient, and what role does
query readout play? The study targets supervision-by-readout interaction,
three-state risk decomposition and their range of validity. Spatial alignment
is an explanation to test, not an established cause.

## Locked matrix and optimization

MM-GDINO FineCops positive-source Native: retain old global-max exists/emit
heads for seeds 17/42/73; add selected-query exists/emit for each seed.
MDETR R101 RefCOCO EMA: train both readouts and both targets for all seeds.
There are 18 new heads, no trunk update. Remote study identifier:
arrow_confidence_readout_v6_20260905 (identifier retained from the planning date).

All heads: original LayerNorm257→128→128→1, 50,179 parameters/eight trainable
tensors; original initialization and complete event generator; five epochs,
12,575 updates, B32 pairs, deterministic FP32, AdamW LR 1e-4, WD 0,
clip 0.1, logit L2=0.001. Both sources retain weight 0.5; incorrect positives
change label only for emit.
Selected readout is used in BOTH optimization and matched inference, with
the full dense head computation retained. No warm start from trained max heads.

## Fixed score and output consumers

Native alone selects the box. MM keeps its original argmax/tie semantics.
MDETR uses official EMA weights and score/box lexicographic native ordering;
the study recomputes IoU>=.5 correctness from its own Native output.
No-target has no fabricated GT. Dynamic Q=900/100 is supported without padding.

Save all per-query-derived max/selected readings, winner indices, winner-vs-
native box IoU and both GT IoUs on positives. Cross-readout inference is a
diagnostic, not another matched-training cell. Its decomposition is path
dependent and is not a unique spatial-causal attribution.

## Effects and interpretation

For R=mixed AUGRC, report all four cells and:
D_emit=R(S,emit)-R(G,emit);D_exists=R(S,exists)-R(G,exists);
I=D_emit-D_exists. Both effect magnitudes and paired intervals matter.
A negative I without an emit improvement can be caused by harming exists.
A negative D_emit without a target-specific interaction can be a general
readout gain. Neither alone certifies spatial correspondence as the cause.

C=correct positive,W=wrong positive,N=no target. Report C-W,C-N,W-N.
Existence AUROC=a*U_CN+(1-a)*U_WN, where a=P(C|positive).
DeltaAUGRC=-(1-pi)*a*((1-pi)*(1-a)*DeltaU_CW+pi*DeltaU_CN).
Report crossover validity and uncertainty; no fabricated interior root.
Same-image and difficulty-conditioned effects retain eligible counts and
comparable unconditioned effects. Wide/zero-crossing CIs do not establish
absence or primarily image/difficulty-level causation.

## Fixed joint scores

Use the same-seed global-max existence score z and Native score s0.
Product: log(max(s0,1e-6))+log_sigmoid(z).
SIRC-style: -log(max(1-s0,1e-6))-softplus(-(z-(mu-3*sigma))/sigma).
mu/sigma use all 83,341 unique TRAIN positive expressions, population SD.
sigma<=1e-12 falls back to Native, explicitly recorded. No evaluation-set
fit, weight search or deployment-threshold calibration.
Compare both combinations with Native,global-exists,global-emit.
Failure of these two rules does not imply that the abilities cannot combine.

## Data and statistics

FineCops val: 9,426 positives/9,029 text negatives; no FineCops Test reopening.
gRef Full and existing FineCops-source-disjoint slice; no multi-target or
negative-image matrix expansion. All new heads seal before new gRef traversal.
One frozen detector traversal per localizer serves every head through caches.
MM old six-head and Native parity is mandatory.

5,000 paired image-cluster draws, PCG64 seed 20260911. Every localizer/score/seed
shares draws within each surface; gRef stratifies TestA/TestB. Recompute
each positive q05 every replicate for diagnostic FPR95. Three head-seed
means and sample SD are separate from image-bootstrap uncertainty.
These are post-hoc-motivated,prospectively locked mechanism analyses, not
virgin held-out confirmation. Different localizer lineages and Native
accuracies are disclosed, not treated as a pure architecture intervention.

## Completion requirements

All 18 heads and all seeds complete; parameter/update/resume and Native
invariance pass; full requested evaluation and statistical analyses finish.
Only then produce v6 figures, tables and prose based on actual effects.
Main order: question→target/readout control→three-state explanation→
second-model scope→combination implications. Tools are research deliverables.
Unrelated historical material remains in the repository, not the new main text.
