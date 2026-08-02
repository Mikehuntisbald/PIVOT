# Semantic Confidence Probe for the Pure-GDINO Adapter

This is an independent confidence-only probe for the frozen pure-GDINO score
adapter.  It addresses the main label mismatch in phase C: the fixed data-FT TN
rows reproduce the baseline's training protocol, but they are not verified
image-expression negatives.  This probe uses only official training rows whose
target and cached candidate regions were all judged negative by the VLM.

It does not use either strict evaluation manifest for training.

## Data contract

The frozen inputs are the two files under
`data/ablations/stageb_v15_global_verified_train_20260711`.  The builder checks
their upstream audit and file hashes, then independently requires every row to
have:

- official train split membership: RefCOCO+ UNC train, or both RefCOCOg Google
  and UMD train;
- exact target VLM answer `no`;
- a non-empty cached proposal set;
- matching, unique proposal IDs and one VLM answer `no` for every proposal;
- a valid, distinct positive/TN expression pair and target box;
- a unique `(dataset,image_id,ann_id,ref_id,sent_id)` identity.

The resulting fixed data has 17,829 rows and SHA-256
`bea2aca85d207d883da85cb219420f748a65a840516218731811e8e46449b645`:

| Source | Rows | Unique images | Official split proof | Proposal count |
| --- | ---: | ---: | --- | --- |
| RefCOCO+ | 10,855 | 6,009 | UNC train 10,855/10,855 | 1-8 |
| RefCOCOg | 6,974 | 4,855 | Google+UMD train 6,974/6,974 | 1-8 |

There are 9,317 unique training images and 1,547 image IDs shared by the two
datasets.  The authoritative strict2031 set overlaps by 59 image IDs; this is
disclosed and strict2031 remains the final full gate.  The separately fixed
strict1607 manifest has exactly zero image-ID overlap with this probe source.

The emitted scope is `image_global_topk_verified`.  This is intentionally not a
claim that all 900 GDINO decoder queries were individually labeled.  The label
is a semantic image-expression negative supported by target plus cached top-k
coverage; the confidence objective then applies it to the all-query maximum as
a generalization objective.  `proposalset_proxy_verified=false` distinguishes
this semantic label from the old proposal-index score-target protocol.  Every
pair also carries `cached_proposal_coverage_only=true`,
`all_900_gdino_queries_verified=false`, and
`global_max_label_is_semantic_extrapolation=true`; these prevent the shorter
`global_tn_verified` compatibility flag from being read as a 900-query claim.

Rebuild or verify the frozen pairs with:

```bash
PYTHONPATH=/home/user/PIVOT DATA_ROOT=/home/user/datasets/pivot_data \
  /usr/bin/python3 tools/build_stageb_gdino_adapter_semantic_verified_pairs.py

PYTHONPATH=/home/user/PIVOT \
  /usr/bin/python3 tools/build_stageb_gdino_adapter_semantic_verified_pairs.py \
  --verify-only
```

## Training contract

The phase uses `confidence_only`, leaves the base and rank branch frozen, and
inherits the repaired confidence objective from phase C:

- exact all-query maximum for the positive and TN expressions;
- `detached_recent_q05_trust` objective with queue size 512 and activation
  count 256;
- global batch 8, so the detached recent q05 bank is active by step 32 and
  retains the latest 64 successful steps;
- every current negative contributes; negative history does not dilute it;
- the detached bank threshold receives a zero-valued global mean positive-gate
  translation proxy, so a shared score/bias shift has zero net gradient;
- positive gate trust is the linear hinge
  `mean(relu(-0.02 - positive_gate))` with weight 1.0, protecting the
  low-positive tail while suppressing TN maxima;
- paired positive/TN margin weight 0.25 and margin 0.05;
- confidence gate LR `3e-4`;
- horizontal flip probability exactly `0.0`, because flipping a spatial
  referring expression without rewriting its caption corrupts supervision;
- internal negative-episode resampling exactly `0.0`; adapter no-support mode
  fails closed otherwise because resampling would break the fixed pair caption
  and sample identity.

The phase may start from an audited R milestone or an audited data-FT C
milestone.  The first semantic segment must use `--pretrain_model_path`, which
creates fresh optimizer, scheduler, criterion, and queue state for this scope.
Using `--resume` across the R/C-to-semantic scope boundary fails audit.  Only
later segments within this semantic scope may use `--resume`.

Static audit and command preview:

```bash
tools/run_stageb_gdino_semantic_confidence_probe.sh --dry-run
```

Start from an audited R milestone:

```bash
tools/run_stageb_gdino_semantic_confidence_probe.sh \
  --source-kind rank \
  --source-checkpoint /path/to/rank/checkpoint_iter_000250.pth \
  --source-audit /path/to/rank/checkpoint_iter_000250.audit.json
```

Start a fresh semantic phase from an audited data-FT C milestone:

```bash
tools/run_stageb_gdino_semantic_confidence_probe.sh \
  --source-kind dataft-confidence \
  --source-checkpoint /path/to/confidence/checkpoint_iter_000250.pth \
  --source-audit /path/to/confidence/checkpoint_iter_000250.audit.json
```

The launcher preserves semantic-confidence steps 50, 100, 250, and 500. Its
audited rank source may be any R milestone through R5000; accepting an extended
R source does not extend the semantic-confidence schedule. Each milestone audit
records checkpoint/file/base/rank/confidence hashes, the semantic scope and
protocol, config/dataset/data/code hashes, queue state, initial checkpoint, and
same-scope resume lineage.  Before every launch it also writes an immutable
`segment_lineage` sidecar.  Recovery validates the live checkpoint against the
sidecar's externally recorded source, then binds the recovery copy to that live
inspection.  It never accepts the live checkpoint's self-reported source as the
expected ancestry, so a forked resume checkpoint is rejected.

The static record is a dependency closure, not a hand-written leaf list.  It
hashes the recursive config import chain from the semantic config through its
data-FT and baseline parents, plus the recursive repo-local Python imports of
the training entry points.  Dataset metadata and the launcher are locked
separately.  A changed parent config or any discovered local Python dependency
invalidates preflight and formal evaluation.

## Evaluation contract

Formal evaluation must receive the semantic milestone audit.  The evaluator
dispatches this schema to:

```bash
/usr/bin/python3 tools/stageb_gdino_semantic_probe_audit.py verify-evaluation \
  --checkpoint /path/to/semantic/checkpoint_iter_000250.pth \
  --audit /path/to/semantic/checkpoint_iter_000250.audit.json
```

That command recomputes the checkpoint path, file hash, and base/rank/confidence
tensor hashes, then revalidates the current config, dataset, 17,829-row data,
data audit, code, preflight, and source/previous milestone lineage.  Unknown or
drifted audit schemas fail closed.  It also recursively replays every semantic
milestone and segment back to the deeply revalidated two-phase R/C initial
milestone.  Recovery inspection contents are replayed against the immutable
recovery checkpoint copy and the segment that produced it; validating only the
sidecar file hash is insufficient.

This remains a probe, not an acceptance result.  A selected milestone must beat
the fixed pure Stage-B data-FT baseline on global FPR@95TPR on strict2031, move
strict1607 in the same direction, and improve the fixed eight-split RefCOCO
acc50/AP gate under the unchanged evaluator and record protocol.
