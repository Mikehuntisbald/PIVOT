# U2-v2 frozen Stage-A/C100 post-gate rank residual diagnostic

Date: 2026-08-16

## Decision

The diagnostic is **rejected for CVPR promotion**.  U2-v2 achieved the two
isolated goals it was designed to test:

- Ref8 is strictly above B58 on all 8 splits, with micro Acc@0.5 improving from
  `0.709365` to `0.711280` (`+0.001914`);
- confidence is bitwise identical to sealed C100 on all 1,607 and 2,031 strict
  records, preserving FPR95 `0.454263 / 0.455933`.

It nevertheless misses the preregistered main-model gate: legacy U2 Ref8 micro
is `0.733714`, so U2-v2 is lower by `0.022434`.  Therefore the leakage-clean
D3 confidence retrain and seeds `17/42/73` were **not started**.  This result is
useful as a clean ablation: a small post-gate residual can recover a consistent
8/8 win over B58, but Gap5's recall-preserving patch admission is much weaker
than legacy U2's category gate.

The machine-readable decision is
`outputs/u2v2_diagnostic_20260816/final_u100_sealed/diagnostic_final_receipt.json`,
SHA256 `1f964592da8d246d8011ab8aa17ecf7c051b1053fe42dd4664a7326bb6a99ccc`.

## Model and ownership contract

The implementation keeps one full-expression B58 forward and one checkpoint:

```text
full expression -> frozen B58 -> frozen raw R100 feature/score
support patch   -> frozen Stage-A patch branch -> eligible mask
eligible set    -> raw R100 + bounded learned residual -> Ref top-1
detached B58 statistics -> frozen C100 -> absolute confidence
```

There is no B58 top-1 guard.  The new residual is
`LayerNorm(128) -> MLP(129,64,64) -> bias-free output`, with
`0.1*tanh(raw/0.1)`, and is applied only inside the frozen eligible set.
Ineligible queries retain the existing lexicographic demotion.  Exactly seven
parameter tensors (12,800 scalars) are trainable; Stage-A patch, 938 B58 trunk
tensors, R100 rank8, C100 confidence12, and the identity/zero U0 shell remain
frozen and are checked by role hashes at load, resume, and every milestone.

The residual loss uses fix margin `0.05`, preserve tolerance `0.01`, floor
`0.005`, temperature `0.05`, and weights `1/2/0.05`.  Its differentiable
surface is the pre-demotion score restricted to the exact eligible mask;
deployment still uses the fail-close `nextafter` demotion.

## Initializer C0

Sources:

- Stage-A `checkpoint0007.pth`: `fe20fe91...a8b`;
- sealed R100 U100: `346e8472...2b7`;
- sealed C100 U100: `c9737d6b...000e`.

C0 is
`outputs/u2v2_diagnostic_20260816/c0/checkpoint_u2v2_c0.pth`, SHA256
`2578c62a187948e7e459afba0f2c72d3de6901912abc3ac7770a19b86f177309`.
The builder verified 1,165 tensors: C100's 938 trunk + rank8 + confidence12,
Stage-A patch196, and U0 shell11.  Stage-A and C100 trunk tensors are bitwise
equal; C100 differs from R100 only in confidence12; patch backbone187 equals
the main backbone.  Runtime construction adds seven residual parameters and
two contract buffers, for 1,174 model-state tensors.

The single-forward gap sweep rejected the more accurate Gap2 because pooled
eligible Recall@0.5 fell by `0.013516`.  Gap3 also missed the pooled `0.002`
limit (`-0.002643`).  Gap5 was the only admitted non-trivial choice:
pooled drop `-0.000113`, worst split drop `-0.000185`, zero raw
correct-to-wrong transitions, and three raw wrong-to-correct transitions.

## Training and runtime audit

Training used seed 42, D2 source weights `2:2:1`, physical batch 38,
accumulation 2, LR/weight decay `1e-4`, clip `0.1`, and AMP scale 8192.  A
literal B38 forward OOMed on a later aspect batch, so the physical DataLoader
batch remains 38 while the frozen trunk forward is split exactly into `19+19`;
losses are sample-count weighted before one backward.  Effective batch remains
76 and the objective is unchanged.

| milestone | checkpoint SHA256 | AMP skips | optimizer tensors |
|---|---|---:|---:|
| U25 | `44c330ba...d30cd` | 0 | 7 |
| U50 | `64bfa294...6ed1b` | 0 | 7 |
| U100 | `bd16b73d...61c04` | 0 | 7 |

At U100 the cumulative audit records 100 successful finite/nonzero updates,
maximum pre-clip grad norm `0.570917`, peak allocated/reserved VRAM
`13.19/28.10 GiB`, and minimum observed free VRAM `4.38 GiB`.  Resume restores
the initializer provenance, scaler, optimizer, update count, and accumulated
runtime audit before continuing.

## Val-only selection

All 28,488 eligible masks for every milestone were SHA-bound per example and
bitwise equal to C0.

| update | RefCOCO val | RefCOCO+ val | RefCOCOg val | micro | admitted |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.645376 | 0.697063 | 0.804330 | 0.695749 | no |
| 50 | 0.645837 | 0.697249 | 0.804739 | 0.696089 | no |
| 100 | **0.646945** | **0.697899** | **0.805556** | **0.696957** | yes |

U25 missed B58 on RefCOCO by one correct example and regressed C0 on
RefCOCOg.  U50 matched C0 on RefCOCOg but remained two examples below B58.
U100 is strictly above B58/C100 Ref-safe and C0 on all three splits, and above
raw R100 (`0.695107`) and C0 (`0.695220`) in aggregate.  Relative to C0 it has
32 correct-to-wrong and 78 wrong-to-correct transitions, net `+46`.

Selection receipt:
`outputs/u2v2_diagnostic_20260816/formal_seed42_b38a2_gap5_micro19/milestone_selection_receipt_v2_fixedsize.json`,
SHA256 `a9d7b02656da727202e87059ddf1e47be005eeaaaa96af36e764926c983fd360`.

## Sealed diagnostic results

The evaluator aggregates all four routes from the same model forward.  No
additional model forward or training is used for these component controls.

| split | B58 base | raw R100 | patch+R100 | patch+residual |
|---|---:|---:|---:|---:|
| RefCOCO val | 0.645468 | 0.644822 | 0.644822 | **0.646945** |
| RefCOCO testA | 0.725119 | 0.725296 | 0.725473 | **0.726887** |
| RefCOCO testB | 0.580373 | 0.579588 | 0.579588 | **0.580765** |
| RefCOCO+ val | 0.694832 | 0.695854 | 0.696133 | **0.697899** |
| RefCOCO+ testA | 0.751834 | 0.752008 | 0.752183 | **0.753929** |
| RefCOCO+ testB | 0.635713 | 0.636531 | 0.636531 | **0.639190** |
| RefCOCOg val | 0.805147 | 0.804739 | 0.804739 | **0.805556** |
| RefCOCOg test | 0.820246 | 0.820975 | 0.820975 | **0.822120** |

Thus the residual contributes the consistent 8/8 B58 promotion; Gap5 patch
admission alone changes only three splits and is nearly identity.  Ref8 summary
SHA256 is `dc94782ad4f93604edb5424a9df5a44653a757b9501a9674b35d83d475cbe628`.

For confidence, U2-v2 uses the standalone C100 execution contract: separate
negative and positive ordinary forwards plus an explicit confidence-only flag
that bypasses the patch gate.  The legacy U0 evaluator's packed 2B forward was
rejected because cross-example image padding changed scores despite identical
weights.  Final parity checks cover sample universe, positive/negative score,
and positive/negative top-query IoU:

| manifest | U2-v2 FPR95 | C100 FPR95 | bitwise record parity |
|---|---:|---:|---:|
| strict1607 | **0.454263** | 0.454263 | 1,607 / 1,607 |
| strict2031 | **0.455933** | 0.455933 | 2,031 / 2,031 |

The strict summaries have SHA256 `93281381...7ac` and `dc94782a...628`
(the latter is the combined Ref8/strict2031 summary).  These are diagnostic
only: sealed C100's training images overlap strict2031 by 67 images.

## Implementation and compatibility

New code includes the strict U2-v2 initializer, bounded residual and criterion,
seven-tensor freeze/optimizer ownership, B38 forward microbatching, cumulative
AMP/VRAM audit, fixed-size eval leaf, per-example eligibility hashes, gap and
milestone selectors, causal route aggregation, C100 confidence-only parity
route, and diagnostic final receipt builder.  Historical U2, D9, D12, and D13
schemas remain unchanged.

The first two staged pushes were:

1. `e470154` — retire the failed V51 decoder adaptation;
2. `aba8ae4` — introduce U2-v2 code, configs, builder, and core tests.

Weights and generated `outputs/` receipts are intentionally not committed.
