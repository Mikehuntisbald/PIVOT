# Original GroundingDINO pre-Stage-B ownership protocol

## Causal identity

The formal frozen trunk is the direct parent of B58:

- release OGC initializer: `weights/groundingdino_swint_ogc.pth`, SHA-256
  `3b3ca256...799`;
- formal pre-Stage-B parent:
  `/media/haoyi/T9/gdino/outputs/ogc_original_finetune_stage_a/checkpoint0001.pth`,
  SHA-256 `2aa2b20b...45de`;
- mixed Stage-B descendant B58:
  `/media/haoyi/T9/gdino/outputs/gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/checkpoint0001.pth`,
  SHA-256 `b58e5209...1157`.

All three expose the same effective 938-tensor, 174,327,226-value pure
GroundingDINO-T architecture.  The parent checkpoint also stores 200 historical
patch tensors; they are rejected from the runtime and audited as unused.  OGC
to parent and parent to B58 each change 727 of the 938 effective tensors.

Using the direct parent, rather than the earlier release initializer, isolates
the representation change introduced by B58's mixed Stage-B continuation.  It
does not remove the parent's same-data positive fine-tuning.

## Fixed matrix

Only two mature cache heads are run:

| Owner | Parameters | MAC/query | Representation |
|---|---:|---:|---:|
| Shared-Wide | 100,362 | 99,424 | shared 210-d |
| Isolated | 100,358 | 98,816 | disjoint 128-d + 128-d |

Both consume the detached 256-d final decoder query plus one native score.  For
this pure-GDINO family the native score is fixed before evaluation as the mean
sigmoid probability over the generated full-expression phrase tokens, matching
the authoritative B58 base-score reduction.

Training reuses the exact e5/e6/pretrained schedule bytes: seeds 17/42/73,
U150, 100 Ranking and 50 Abstention updates, batches 32/8, task-specific Adam
states, zero weight decay, rank/confidence learning rates `3e-5`/`1e-4`, and
fixed probes at U25/U50/U100/U150.  Shared-128 is excluded.

## Evaluation and statistics

- Test5: RefCOCO TestA/B, RefCOCO+ TestA/B, RefCOCOg UMD test, 30,969 rows.
- TestAB: pooled RefCOCO TestA/B, derived from the same Test5 forward.
- Strict-TN2031: paired FPR95, AUROC, and AUPR.
- 5,000 paired image-cluster bootstraps, PCG64 seed `20260824`; every FPR95
  replicate recomputes each owner/seed positive q05.
- Same frozen cache for Native, Shared-Wide, and Isolated.

No checkpoint, milestone, score reduction, Gap, threshold, sample alias, loss,
or evaluation surface may be selected after the first formal forward.  The
experiment answers ownership geometry for the direct pre-Stage-B parent; it
does not claim that the OGC release checkpoint itself was formally trained.
