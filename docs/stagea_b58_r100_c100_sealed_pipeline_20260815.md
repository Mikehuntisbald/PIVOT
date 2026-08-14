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

The Stage A completion contract expects `checkpoint0007.pth`, epoch 7 complete,
45,608 successful optimizer updates, unchanged frozen tensors, and changes in
all nine trainable patch tensors.

## R100/C100 ownership

The completed Stage A contains 1,134 tensors. Its 938 non-patch tensors are the
ordinary GroundingDINO trunk and are bitwise B58. The remaining 196 tensors
belong to the patch branch. The ordinary-GDINO score-adapter architecture is
deliberately patch-free, so R100 consists of exactly those 938 Stage A trunk
tensors plus 20 adapter tensors:

- eight R100 rank tensors;
- twelve identity-initialized confidence tensors.

R100 runs for exactly 100 optimizer updates at physical batch 32. C100 starts
from the receipt-bound R100, runs the total-trust objective for exactly 100
updates at physical batch 8, and is forbidden from changing either the trunk or
rank tensors. This preserves the existing gradient-conflict boundary: ranking
and confidence own disjoint adapter parameters over one shared frozen trunk.

The launcher is:

```bash
bash tools/run_stagea_b58_r100_c100.sh
```

It emits `pivot.stagea_b58_r100_receipt/v1` between R100 and C100. The receipt
rehashes the Stage A checkpoint, its initializer and B58 source, the R100
checkpoint, configs, datasets, orchestration, and training code. It fails
closed if an incomplete Stage A or a non-R100 checkpoint is supplied. After
C100, the same launcher runs the two-process sealed evaluation and exits
successfully only if `stagea_r100_c100_all_sealed_gates_passed=true`.

## Sealed replay

The existing total-trust evaluator now accepts either the historical legacy
R100 receipt or the new Stage-A/R100 receipt. For the new receipt it verifies
the Stage A -> B58 root and makes Ref8 no-regression a required goal gate in
addition to both strict FPR95 wins. A protocol execution may still complete and
publish a negative result, but the full goal is achieved only when
`stagea_r100_c100_all_sealed_gates_passed=true` in `postflight.json`.
