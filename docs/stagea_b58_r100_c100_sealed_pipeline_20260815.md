# Stage-A B58 -> R100 -> C100 sealed pipeline

## Goal

Train the B58-trunk/checkpoint0006-patch Stage A to completion, then train an
R100 rank adapter and a C100 total-trust confidence adapter. The terminal
checkpoint is accepted only when the sealed replays prove all of the following:

- strict2031 FPR95 is strictly lower than the fixed historical B58 result;
- strict1607 FPR95 is strictly lower than the fixed historical B58 result;
- every canonical Ref8 split has no Acc@0.5 regression against B58;
- the C100 rank tensors are bitwise identical to the sealed R100 tensors;
- the R100 ordinary-GDINO trunk is bitwise identical to the completed Stage A
  non-patch trunk.

The historical unguarded R100/C100 replay already proves the confidence side:
it reduced strict1607 false accepts from 801 to 730 and strict2031 from 1,040
to 926. It did not satisfy the stronger Stage-A goal because its raw R100
residual regressed three Ref splits (RefCOCO val by 7 correct rows, RefCOCO
testB by 4, and RefCOCOg val by 2). The Stage-A lineage therefore uses a new
Ref-only deployment contract rather than silently reusing that failed route.

## Stage A

The formal run is:

```text
/media/haoyi/T9/gdino/outputs/
  stageA_b58_trunk_patch0006_realign_bs38_formal_20260814/
```

It starts from the sealed B58/checkpoint0006 initializer and trains exactly the
nine patch-owned tensors. The first batch-38 attempt ended at update 160 only
because the host rebooted at 2026-08-15 00:52; it had no Python traceback, AMP
skip, or CUDA OOM. That interrupted directory is retained as
`stageA_b58_trunk_patch0006_realign_bs38_formal_20260814_host_reboot_u0160`.
The fresh run uses an iteration-checkpoint interval of 100 so a later host
restart cannot discard 500 updates.

That run exposed one AMP overflow when the default GradScaler growth probe
raised the scale from 131,072 to 262,144: at DataLoader iteration 4,101 the
checkpoint recorded only 4,100 successful optimizer updates. Because the
sealed Stage A contract requires one successful update for every one of the
45,608 forwards, the run was stopped and retained with suffix
`_amp_growth_skip_u4060`; its weights are not resumed. The replacement run
starts from the initializer with scale 65,536 and growth interval 1,000,000,
so no automatic scale probe occurs within the formal run.

The Stage A completion contract expects `checkpoint0007.pth`, epoch 7 complete,
45,608 successful optimizer updates, zero skipped AMP steps, unchanged frozen
tensors, and changes in all nine trainable patch tensors.

## R100/C100 ownership

The completed Stage A contains 1,134 tensors. Its 938 non-patch tensors are the
ordinary GroundingDINO trunk and are bitwise B58. The remaining 196 tensors
belong to the patch branch. The ordinary-GDINO score-adapter architecture is
deliberately patch-free, so R100 consists of exactly those 938 Stage A trunk
tensors plus 20 adapter tensors:

- eight R100 rank tensors;
- twelve identity-initialized confidence tensors.

R100 runs for exactly 100 optimizer updates at physical batch 32. Its raw rank
residual remains the optimized signal, but the Stage-A-lineage deployment route
anchors B58's top-1 query and allows the learned tower to reorder only the
remaining rank tail. This is a structural Ref8 no-regression boundary: unlike
the historical unguarded R100, a residual cannot flip a B58-correct top-1 into
an error. C100 starts from the receipt-bound R100, runs the total-trust objective
for exactly 100 updates at physical batch 8, and is forbidden from changing
either the trunk or rank tensors. Ranking and confidence therefore retain
disjoint parameter and deployment ownership over one shared frozen trunk.
The guard does not use Ref labels, split identity, or an evaluation-time
oracle. It deterministically preserves the argmax of the frozen base score for
every expression and uses the trained residual only to order the remaining 899
queries. Since the Stage-A receipt separately proves the ordinary trunk is
bitwise B58, Ref8 equality is an architectural consequence that the sealed
replay must still verify from full per-example records.

The launcher is:

```bash
bash tools/run_stagea_b58_r100_c100.sh
```

It emits `pivot.stagea_b58_r100_receipt/v2` between R100 and C100. The receipt
rehashes the Stage A checkpoint, its initializer and B58 source, the R100
checkpoint, configs, datasets, orchestration, and training code. It also
requires the Stage A launch-source manifest. That manifest binds the logged
launch commit to the exact tracked dirty-worktree patch, the automatically
snapshotted resolved config, and SHA-256 records for the Stage A runtime
sources; every relevant mtime was verified not to exceed the logged launch
time. This prevents a later source edit from being misrepresented as the code
that produced Stage A. The receipt fails closed if an incomplete Stage A, a
missing/drifted launch seal, or a non-R100 checkpoint is supplied. After C100,
the same launcher runs the two-process sealed evaluation and exits successfully
only if `stagea_r100_c100_all_sealed_gates_passed=true`.

## Sealed replay

The existing total-trust evaluator now accepts either the historical legacy
R100 receipt or the new Stage-A/R100 receipt. For the new receipt it verifies
the Stage A -> B58 root and makes Ref8 no-regression a required goal gate in
addition to both strict FPR95 wins. A protocol execution may still complete and
publish a negative result, but the full goal is achieved only when
`stagea_r100_c100_all_sealed_gates_passed=true` in `postflight.json`.
