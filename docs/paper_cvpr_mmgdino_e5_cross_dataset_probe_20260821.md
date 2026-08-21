# MM-GDINO e5 cross-dataset rank-gradient probe

## Question and frozen contract

This zero-update mechanism probe keeps fixed:

- the RefCOCO-e5 frozen MM-GDINO query representation;
- each seed's U150 Shared-128 or Shared-Wide checkpoint;
- the first eight formal D3 confidence batches;
- the sealed U150 confidence queue;
- batch size, losses, and all parameters.

Only the rank probe changes among deterministic 8-by-32 samples from
RefCOCO val, RefCOCO+ val, and RefCOCOg UMD val.  The sampling rule and all
sample identities were sealed before selected-candidate extraction.  No
optimizer is created.  Each checkpoint's tensor hash is identical before and
after probing.

Cosine and sign conflict use only parameters in the shared representation.
The native top-1 margin is the largest minus second-largest valid native query
score.  We additionally report the best IoU-positive minus best IoU-negative
native-score gap to disambiguate score certainty from oracle separation.

## Three-seed results

Each cell is the equal-weight mean of the seed-specific eight-batch means.
Norms are L2 norms on the shared parameters.

| Owner | Rank probe | Cosine | Sign conflict | $\|g_R\|$ | $\|g_C\|$ | Native P@1 | Rank loss | Top1 margin | Oracle gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Shared-128 | RefCOCO | +0.0714 | 0.4857 | 0.0372 | 0.0206 | 0.8945 | 0.4380 | 0.6899 | 0.6273 |
| Shared-128 | RefCOCO+ | +0.0739 | 0.4851 | 0.0344 | 0.0206 | 0.7448 | 0.4133 | 0.5551 | 0.3949 |
| Shared-128 | RefCOCOg | +0.0792 | 0.4851 | 0.0402 | 0.0206 | 0.8268 | 0.4925 | 0.6500 | 0.5215 |
| Shared-Wide | RefCOCO | +0.0260 | 0.4916 | 0.0384 | 0.0501 | 0.8945 | 0.4376 | 0.6899 | 0.6273 |
| Shared-Wide | RefCOCO+ | +0.0235 | 0.4929 | 0.0342 | 0.0501 | 0.7448 | 0.4127 | 0.5551 | 0.3949 |
| Shared-Wide | RefCOCOg | +0.0282 | 0.4902 | 0.0412 | 0.0501 | 0.8268 | 0.4919 | 0.6500 | 0.5215 |

The confidence gradient norm is exactly invariant to rank-dataset choice for a
fixed owner/seed, as required.  Its per-batch gradient SHA-256 also matches
across all three rank datasets.

## Seed-level cosine

| Owner | Rank probe | seed17 | seed42 | seed73 |
|---|---|---:|---:|---:|
| Shared-128 | RefCOCO | +0.1100 | +0.0993 | +0.0051 |
| Shared-128 | RefCOCO+ | +0.0752 | +0.1395 | +0.0069 |
| Shared-128 | RefCOCOg | +0.0756 | +0.1535 | +0.0086 |
| Shared-Wide | RefCOCO | +0.0087 | +0.0739 | -0.0045 |
| Shared-Wide | RefCOCO+ | +0.0075 | +0.0946 | -0.0316 |
| Shared-Wide | RefCOCOg | +0.0036 | +0.0867 | -0.0057 |

Only 3 of 18 route/dataset/seed cosine cells are negative, all from the same
Shared-Wide seed73 checkpoint.  Changing RefCOCO to RefCOCO+ or RefCOCOg does
not reveal a hidden dataset-independent conflict.  RefCOCO+ is clearly the
harder rank surface for this e5 trunk (lower native P@1 and smaller native
margins), but that difficulty does not turn the mean shared gradient cosine
negative.

## Paper claim boundary

Allowed:

> Changing the rank probe among RefCOCO, RefCOCO+, and RefCOCOg changes rank
> difficulty and gradient magnitude, but does not reveal consistent negative
> rank/rejection alignment on the strong e5 shared owners.

Prohibited:

> The strong e5 shared owner exhibits dataset-independent harmful
> rank/rejection gradient conflict.

Machine-readable authority:
`paper/data/mmgdino_e5_cross_dataset_probe_results.json`.  The selected
candidate caches and per-batch result remain in ignored `outputs/`.
