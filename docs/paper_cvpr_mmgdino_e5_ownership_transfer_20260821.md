# Strong MM-GDINO e5 ownership transfer

## Scope and evidence status

This experiment asks whether responsibility isolation remains beneficial when
the frozen query representation is already strong.  The candidate generator is
the locally supplied MM-GDINO-T RefCOCO five-epoch checkpoint
`weights/epoch_5.pth` (SHA-256
`2ec6fbc01ee70e8c18f96e22614053c95f54932fee7fa14b488c404191c05d7b`).
It is a retrospective strong-trunk replay: the embedded training loop evaluated
RefCOCO val/testA/testB after every epoch.  The ownership heads and their
evaluation were frozen prospectively, but the experiment is not a virgin
held-out evaluation of the trunk.

The trunk, its 900 queries, and its boxes are frozen.  Confidence never enters
query or box selection.  All learned routes start with the native ranking,
then receive exactly 100 rank and 50 rejection updates in the fixed
rank--confidence--rank schedule.  Seeds are 17/42/73.  AdamW weight decay is
zero and the rank and rejection duties always have separate optimizer states,
including when their parameter references overlap.

## Capacity-controlled matrix

| Route | Trainable parameters | MAC/query, both outputs | Per-task representation |
|---|---:|---:|---:|
| Native | 0 | 0 | -- |
| Shared-128 | 50,308 | 49,536 | 128 |
| Shared-Wide | 100,362 | 99,424 | 210 |
| Isolated | 100,358 | 98,816 | 128 + 128 disjoint |

Shared-Wide has four more parameters and 0.62% more MACs than Isolated, and
each task can access a wider representation.  It is therefore the primary
capacity/FLOP control rather than Shared-128 alone.

## Primary results

Values below are three-seed means for learned routes.  RefCOCO is pooled
TestA+TestB micro P@1.  Strict is FPR95 on Strict-TN2031.

| Route | RefCOCO TestAB | TestA | TestB | Strict FPR95 |
|---|---:|---:|---:|---:|
| Native | 88.969 | 91.267 | 86.418 | 88.282 |
| Shared-128 | 88.973 | 91.273 | 86.418 | 85.442 |
| Shared-Wide | **88.979** | 91.273 | **86.431** | **83.686** |
| Isolated | 88.976 | 91.273 | 86.425 | 84.211 |

Isolated versus Native reduces FPR95 by 4.070 points (95% paired
image-cluster CI [2.092, 5.907]) while remaining non-inferior on RefCOCO.
That preregistered intersection-union contrast passes after Holm correction.

Isolated does **not** significantly beat Shared-128: the FPR95 reduction is
1.231 points with CI [-0.115, 2.390].  More importantly, it does not beat the
capacity-matched Shared-Wide control; Shared-Wide is better by 0.525 FPR95
points at the point estimate, and the isolated-minus-shared improvement CI is
[-1.812, 0.465].  The full three-contrast claim gate therefore fails.

## Gradient result

At U150, mean rank/rejection gradient cosines on the shared representation are:

- Shared-128: -0.0633, +0.0082, -0.0111 for seeds 17/42/73.
- Shared-Wide: -0.0178, +0.0324, +0.0118 for seeds 17/42/73.
- Isolated: no cross-task autograd path by construction.

The shared controls do not exhibit negative mean alignment in every seed.
Consequently, this transfer does not support a universal claim that strong
query representations still suffer systematic rank/rejection gradient
conflict.

## Allowed paper conclusion

On a strong RefCOCO-finetuned candidate representation, a learned absolute
rejector improves verified-negative rejection without changing REC ranking.
Exclusive ownership is sufficient to preserve the localization route, but it
is not necessary for the observed gain and is not superior to the
capacity-matched Shared-Wide control.  The original ARROW-U2 ownership result
remains evidence for that model and training surface; its strongest causal
interpretation does not automatically transfer to the strong e5 trunk.

The following sentence is prohibited by the sealed result:

> Responsibility isolation universally outperforms shared optimization on
> strong query representations.

## Statistical and provenance contract

- RefCOCO TestA and TestB use a pooled micro primary and per-split 0.005
  no-collapse guards.
- Strict-TN2031 is primary; Strict1607 is derived from the same forward.
- The 5,000 paired bootstrap replicates sample images, not expressions, and
  use the same draw for all routes and seeds.
- Every FPR95 bootstrap replicate recomputes each route's positive q05.
- No route, milestone, margin, queue, or threshold was selected from these
  results.
- Two fail-closed attempts are retained: one stopped before U1 because the
  deterministic CUBLAS environment was absent; one stopped after one rank
  update and before any confidence update because CUDA `kthvalue` violated
  determinism.  All nine formal trajectories restarted from zero after the
  order-statistic amendment.
- The mechanism-only Val cache first failed after forward because an eval row
  without an IoU-positive candidate reached the stricter training-cache
  validator; a sealed eval-only schema amendment counts such rows as failures
  without weakening the training contract.  A second retry stopped before
  forward because PyTorch's `weights_only` default rejected trusted MMEngine
  metadata.  Its runtime-only amendment restored the already pinned offline
  checkpoint-loading environment.  Neither failure produced result rows or
  changed any primary output.
- A final pre-evaluation runner check then rejected the newly sealed Val-only
  hashes because it knew only the earlier Strict-path amendment.  The runner
  amendment allowlists exactly the Val extractor/evaluator hashes for status
  and `--include-val`; non-Val reruns remain prohibited.  This attempt also
  wrote zero result rows.

The machine-readable authority is
`paper/data/mmgdino_e5_ownership_results.json`; large checkpoints, caches, and
per-example records remain under ignored `outputs/`.
