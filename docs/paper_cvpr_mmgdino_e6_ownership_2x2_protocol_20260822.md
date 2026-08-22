# MM-GDINO e6 ownership 2×2 protocol

> Status: complete. The preregistered full pattern did not pass. See
> `docs/paper_cvpr_mmgdino_e6_ownership_2x2_results_20260822.md` and the sealed
> receipt `paper/data/mmgdino_e6_ownership_2x2_results.json`.

## Question

This block tests whether the usefulness of parameter isolation depends on how
the frozen query representation was trained.  It changes neither the owner
architecture nor the owner supervision.  It changes only the frozen trunk:

| Frozen trunk | Shared-Wide | Isolated |
|---|---:|---:|
| e5→e6 PosCtrl | seeds 17/42/73 | seeds 17/42/73 |
| e5→e6 TN10 | seeds 17/42/73 | seeds 17/42/73 |

The existing e5 Native/Shared-Wide/Isolated result is reused as a reference;
it is not forwarded or trained again. Shared-128 is intentionally excluded.

## Frozen trunks

- `weights/epoch_6_postctrl.pth`, SHA-256
  `08177fac668d62de99100b292ee5ff157366c33c48eb56b742006263a42022c3`;
  epoch 6, iteration 45,234.
- `weights/epoch_6_tn10.pth`, SHA-256
  `a7078f1139c847d99e85221c8228f7cfd00e5be5ca0b85820f5d4d6a02cfa66c`;
  epoch 6, iteration 45,988.

Both checkpoints have the same 908-tensor schema and 173,006,505 parameters
as the e5 trunk. PosCtrl is the positive RefCOCO continuation control. TN10 is
the corresponding continuation with 10% TN exposure. The supplied trunk
training is retrospective because its loop evaluated RefCOCO benchmark splits;
the new owner block is prospectively frozen.

## Matched owner training

Every trunk/owner/seed reuses the exact e5 schedule bytes: the same selected
sample identities, batch order, 100 rank updates, 50 confidence updates, and
`rank, confidence, rank` interleave. Shared-Wide and Isolated retain the prior
capacity control:

| Owner | Parameters | MAC/query, both outputs | Representation |
|---|---:|---:|---:|
| Shared-Wide | 100,362 | 99,424 | shared 210-d |
| Isolated | 100,358 | 98,816 | disjoint 128-d + 128-d |

Both duties use independent AdamW optimizer states. Weight decay is zero;
rank/confidence learning rates are `3e-5`/`1e-4`; clip norm is `0.1`; training
is deterministic FP32. U25/U50/U100 are audit milestones only and U150 is the
fixed endpoint.

## Evaluation and statistics

- REC: RefCOCO TestA+TestB pooled micro P@1, with splits reported.
- Rejection: Strict-TN2031 FPR95; every bootstrap replicate recomputes each
  trunk/owner/seed positive q05.
- Uncertainty: 5,000 paired image-cluster bootstrap replicates, PCG64 seed
  `20260822`; the same image draw is applied across both trunks, both owners,
  and all seeds.
- Planned within-trunk contrasts: Isolated − Shared-Wide for PosCtrl and TN10;
  Holm correction is applied across these two contrasts.
- Planned cross-trunk contrast: difference in the isolation gap,
  `(Isolated−Shared-Wide)_TN10 − (Isolated−Shared-Wide)_PosCtrl`.

Gradient probes are the fixed eight rank/confidence batch pairs at
U25/U50/U100/U150. The paper reports cosine mean, `P(cos<0)`, q05, and minimum.
These probes are paired mechanism diagnostics, not expression-IID samples.

## Claim boundary

The desired pattern is preregistered but not assumed:

1. PosCtrl remains near-orthogonal and Shared-Wide approximately matches
   Isolated.
2. TN10 develops a larger negative cosine tail.
3. TN10 Isolated improves REC over TN10 Shared-Wide.

If any component fails, the paper states the observed subset. No checkpoint,
milestone, threshold, loss, margin, seed, or sample alias may be changed after
seeing a result.
