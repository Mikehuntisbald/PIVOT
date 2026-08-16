# Stage-B U2-v4: legacy admission replay on the C100 skeleton

Date: 2026-08-16

## Conclusion

The legacy U2 gain is recoverable on the new Stage-A/C100 skeleton. The decisive
component is the complete admission subsystem, not the nine deployed patch
surface tensors alone:

- transplanting the legacy surface9 without training reproduces legacy U2 Gap3
  exactly on all 26,488 val records;
- replay training must restore the legacy U0 auxiliary initialization (nonzero
  LayerNorm/MLP trunk and zero output), otherwise the auxiliary rank gradient is
  exactly zero;
- with the correct initializer, U100 improves the zero-training transplant by
  29 net val examples and improves Ref8 micro over legacy U2;
- B58 trunk, R100, C100 confidence, patch backbone and patch scale remain frozen
  and bitwise unchanged throughout training.

This supports the intended ownership rule: isolate gradients between
trunk/R100/confidence, while keeping the admission surface and its auxiliary
residual trainable as one subsystem.

## Bound artifacts

Sources:

- C100 C0: `outputs/u2v2_diagnostic_20260816/c0/checkpoint_u2v2_c0.pth`,
  SHA256 `2578c62a187948e7e459afba0f2c72d3de6901912abc3ac7770a19b86f177309`.
- legacy U0 initializer:
  `outputs/paper_cvpr_v1/u0_single_network_seed42_b56_v1/initializer/checkpoint_u0_init.pth`,
  SHA256 `c89e5dfba795fd8074a044f0c09d81c871705c20a1dbf819b9f16c770a2cba43`.
- legacy U2 checkpoint:
  `outputs/paper_cvpr_v1/u2_category_complete_seed42_b56_scale8192_v2/checkpoint_iter.pth`,
  SHA256 `44e3d70b164eff2bcefacc37081b7cbab184a9373720ef69713d47949d449b90`.

Zero-training surface9 transplant:

- checkpoint:
  `outputs/u2v4_legacy_admission_replay_20260816/c0_legacy_surface9/checkpoint_eval_only.pth`
- SHA256 `7729d7cc98a40028ffb63c86b528073fb58290ae3ab3f132c06e0ba19ca99cc0`
- val summary:
  `outputs/u2v4_legacy_admission_replay_20260816/c0_legacy_surface9/val3_gap3_b16/summary.json`

Corrected training initializer:

- checkpoint:
  `outputs/u2v4_legacy_admission_replay_20260816/training_initializer/checkpoint_u2v4_init.pth`
- SHA256 `97edb922caf2461c147c83d58bcbae3cd759457c2f7f58305293c43422c5ce02`
- 1,154 non-U0 tensors are bitwise C100 C0;
- 11 U0 tensors are bitwise legacy U0;
- U0 output weight/bias are zero while all trunk weight tensors are nonzero.

Training milestones:

| update | SHA256 |
| ---: | --- |
| 25 | `396c4f48446ceba3d4cfcf0f00ff2f49469c427ce9dca26e71b58980d6ace998` |
| 50 | `a5cf3b1338a15dfa02465248b2d6459688da705edccf21ef52d6772513a0cc9e` |
| 100 | `134f1ccf8ea927ceaa1d06ab315d43a8bd358388b7f39f82c89074dc10ce7c3f` |

The files are under
`outputs/u2v4_legacy_admission_replay_20260816/training_replay_formal_seed42_b56/milestones/`.
Weights under `outputs/` are intentionally not committed.

## Zero-training replay result

Loading only the nine legacy U2 patch surface tensors into the C100 skeleton
reproduced the legacy U2 Gap3 val result exactly:

| split | accuracy@0.5 |
| --- | ---: |
| RefCOCO val | 0.666235924 |
| RefCOCO+ val | 0.743074921 |
| RefCOCOg val | 0.806985294 |

For all 26,488 records, `sample_id`, `correct50`, `top1_iou`, and eligible-query
count matched legacy U2 with zero differences. This isolates the deployed
legacy admission behavior to surface9 when the trunk/R100 query universe is
held fixed.

## Why the first replay attempt had zero residual gradient

The original C100 C0 builder initialized all eight U0 auxiliary parameters to
zero. Legacy U0 instead preserves a nonzero LayerNorm/MLP trunk and zeroes only
the final output. With the all-zero trunk the auxiliary output is query-constant,
so its pairwise ranking gradient cancels.

Observed first-step gradient norms at identical seed/B56:

| initializer | auxiliary residual | admission surface |
| --- | ---: | ---: |
| all-zero C0 shell | 0.000000 | 1.373753 |
| restored legacy U0 shell | 0.292730 | 1.373760 |

The corrected formal run had nonzero gradients for both branches on 100/100
successful optimizer steps, zero AMP skips, zero nonfinite boundaries, and AMP
scale 8192 throughout.

Training ownership audit:

- trainable: surface8 plus U0 auxiliary residual8 (16 tensors, 268,167 params);
- frozen: 1,149 tensors, with identical frozen SHA256
  `9a87220026eb7e5f39a3b84b298340bbc4819f2df5b7cbd48a6a574060a20ce4`
  at U25/U50/U100;
- peak allocated 23,027,982,336 bytes, peak reserved 27,476,885,504 bytes;
- minimum device-free memory 3,458,465,792 bytes.

## Val-only milestone selection

Gap3 results:

| checkpoint | RefCOCO val | RefCOCO+ val | RefCOCOg val | micro |
| --- | ---: | ---: | ---: | ---: |
| surface9 C0 | 0.666236 | 0.743075 | 0.806985 | 0.723460 |
| U25 | 0.659313 | 0.729875 | 0.805147 | 0.714928 |
| U50 | 0.666051 | 0.743633 | 0.805964 | 0.723422 |
| **U100** | **0.667344** | **0.744655** | **0.806985** | **0.724555** |

U100 changes C0 outcomes by:

- RefCOCO: 19 wrong-to-correct, 7 correct-to-wrong, net +12;
- RefCOCO+: 27 wrong-to-correct, 10 correct-to-wrong, net +17;
- RefCOCOg: 3 wrong-to-correct, 3 correct-to-wrong, net 0.

Thus U100 is selected: no val split loses accuracy to C0, at least one improves,
and aggregate accuracy improves. RefCOCOg eligible Recall@0.5 decreases by
0.001225, within the registered 0.005 per-split tolerance.

## One-time diagnostic Ref8 and strict results

U100 Gap3 versus B58 and legacy U2:

| split | U2-v4 | B58 | legacy U2 |
| --- | ---: | ---: | ---: |
| RefCOCO val | 0.667344 | 0.645468 | 0.666236 |
| RefCOCO testA | 0.758529 | 0.725119 | 0.757469 |
| RefCOCO testB | 0.581354 | 0.580373 | 0.581747 |
| RefCOCO+ val | 0.744655 | 0.694832 | 0.743075 |
| RefCOCO+ testA | 0.826231 | 0.751834 | 0.824834 |
| RefCOCO+ testB | 0.641849 | 0.635713 | 0.640622 |
| RefCOCOg val | 0.806985 | 0.805147 | 0.806985 |
| RefCOCOg test | 0.822120 | 0.820246 | 0.821704 |
| **Ref8 micro** | **0.734602** | **0.709365** | **0.733714** |

U2-v4 is strictly above B58 on 8/8 splits and improves Ref8 micro over legacy
U2 by 0.000888 (51 net examples). It is not splitwise dominant over legacy U2:
RefCOCO testB is two examples lower, so the paper claim must be micro improvement
plus 8/8 B58 wins, not 8/8 legacy-U2 wins.

Confidence/FPR remains the frozen C100 result:

| manifest | U2-v4/C100 FPR95 | B58 | legacy U2/P50 |
| --- | ---: | ---: | ---: |
| strict1607 | 0.454263 | 0.498444 | 0.455507 |
| strict2031 | 0.455933 | 0.512063 | 0.462334 |

Current-code standalone C100 and U2-v4 records are bitwise identical for all
1,607 and 2,031 rows across `sample_id`, positive/negative confidence score,
positive/negative IoU, and validity. Historical C100 records differ in a small
number of scores by at most one-to-two float32 ULP (`2.38e-7`) while producing
identical metrics; this is cross-run numeric jitter, not an ownership change.

The final per-record outputs are under
`outputs/u2v4_legacy_admission_replay_20260816/final_u100_once/`. The combined
Ref8+strict2031 command completed all Ref8 records, then exposed and stopped on
a U2-v4 confidence-only guard bug. Commit `de15be0` fixes that fail-close route;
strict1607/2031 were then rerun independently and completed.

## Interpretation and next experiment

The admission auxiliary residual is useful even though it is not deployed: it
provides structured category-complete ranking supervision to the deployed patch
surface. Removing it made the intended replay degenerate. At the same time,
U25's large regression shows that admission training has a transient destructive
phase; the benefit appears only near U100.

For the CVPR-clean version, retain this exact ownership and mechanism, but replace
the diagnostic C100 confidence tensors because its training data overlaps 67
strict2031 images. Build from Stage-A plus positive-only R100, train the admission
subsystem and fresh image-disjoint D3 confidence in separate phases, then run
seeds 17/42/73. The diagnostic result validates the architecture and replay
mechanism; it is not leakage-clean headline evidence.
