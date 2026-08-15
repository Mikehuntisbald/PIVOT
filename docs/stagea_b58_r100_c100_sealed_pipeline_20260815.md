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

### Live u10000 ownership audit

The replacement batch-38 run reached and persisted 10,000 successful optimizer
updates on 2026-08-15. A CPU-only audit loaded the completed iteration
checkpoint only after its save record appeared and compared every model tensor
against the fixed initializer. All assertions passed:

- the checkpoint contains 1,134 model tensors; exactly the nine allowlisted
  patch tensors changed and the other 1,125 tensors remained bitwise equal;
- all model tensors are finite;
- the optimizer contains exactly nine parameter states and nine parameter-group
  entries; all 27 AdamW state tensors are finite;
- `optimizer_updates=10000`, AMP scale is 65,536, and the scaler growth tracker
  is exactly 10,000 with growth interval 1,000,000.

The changed set is `patch_logit_scale`, the six
`patch_encoder.{input_proj,norm}` tensors, and the two
`query_proj_for_patch` tensors. Therefore the decoder, query embeddings,
ordinary GroundingDINO trunk, and every other query-semantic alignment tensor
remain frozen at this milestone. This is an intermediate ownership proof; the
same assertions must be repeated against the final `checkpoint0007.pth` before
R100 handoff.

### Epoch-1 checkpoint audit

The immutable `checkpoint0001.pth` was audited after its size and mtime were
stable and the epoch-1 averaged statistics had been written. It records
`epoch=1`, `epoch_finished=true`, and exactly 11,402 successful optimizer
updates: two of eight epochs, or 25% of the Stage A update contract. The same
full assertions passed against this formal epoch checkpoint:

- exactly the nine allowlisted patch tensors changed and all 1,125 frozen
  tensors remained bitwise equal to the initializer;
- every model tensor and all 27 tensors across the nine AdamW states remained
  finite;
- AMP scale remained 65,536, with growth tracker exactly 11,402 and growth
  interval 1,000,000.

Thus the u10000 ownership result persists at the first subsequent immutable
epoch boundary. The remaining six epochs must complete before the final
`checkpoint0007.pth` handoff to R100.

### Epoch-2 checkpoint audit

The next immutable boundary, `checkpoint0002.pth`, was audited after the file
reached its stable 695,205,625-byte size. It records `epoch=2`,
`epoch_finished=true`, `iteration=0`, and exactly 17,103 successful optimizer
updates: three times the fixed 5,701-update epoch length, or 37.5% of the
45,608-update Stage A contract. The full ownership and numerical assertions
again passed:

- exactly the same nine allowlisted patch tensors changed and all 1,125 frozen
  tensors remained bitwise equal to the initializer;
- all 1,134 model tensors and all 27 tensors across the nine AdamW states were
  finite;
- the optimizer still contained exactly nine states and nine parameter-group
  entries;
- AMP scale remained 65,536 and its growth tracker was exactly 17,103, equal
  to the successful optimizer-update count.

The formal process and automatic R100/C100 controller remained alive after the
checkpoint was written, and the GPU returned to full training utilization.
This proves that the Stage A ownership boundary persisted through three of the
eight epochs; five epochs remain before the final R100 handoff.

### 20,000-update ownership audit

The live interval checkpoint at exactly 20,000 successful optimizer updates
was compared tensor-by-tensor with the sealed Stage A initializer. The audit
passed the same ownership and numerical contract at 43.852% of the full
45,608-update run:

- the changed set was exactly the nine allowlisted patch tensors, with no
  missing expected change and no unexpected change;
- all other 1,125 model tensors remained bitwise equal to the initializer;
- all 1,134 model tensors were finite;
- the optimizer still contained exactly nine states, nine parameter-group
  entries, and 27 finite state tensors;
- AMP scale remained 65,536 and its growth tracker was exactly 20,000, equal
  to the successful optimizer-update count.

The checkpoint records `epoch=3` and `iteration=2897`. The formal Stage A
process and downstream controller remained alive, and GPU training resumed at
normal utilization after checkpoint serialization. This is an intermediate
ownership proof; it does not replace the immutable epoch-3 audit or the final
`checkpoint0007.pth` lineage gate.

### Epoch-3 checkpoint audit

The immutable `checkpoint0003.pth` reached a stable 695,205,625-byte size and
was then audited tensor-by-tensor against the sealed initializer. It records
`epoch=3`, `epoch_finished=true`, `iteration=0`, and exactly 22,804 successful
optimizer updates: four times the fixed 5,701-update epoch length, or exactly
50% of the 45,608-update Stage A contract. Every assertion passed:

- the changed set remained exactly the nine allowlisted patch tensors, while
  all 1,125 frozen tensors remained bitwise equal to the initializer;
- all 1,134 model tensors and all 27 tensors across the nine AdamW states were
  finite;
- the optimizer contained exactly nine states and nine parameter-group
  entries;
- AMP scale remained 65,536 and its growth tracker was exactly 22,804, equal
  to the successful optimizer-update count.

The Stage A process and automatic R100/C100 controller remained alive, and GPU
utilization returned to 95% after checkpoint serialization. The first half of
Stage A therefore completed without an AMP skip, numerical failure, or breach
of the frozen query-semantic ownership boundary; four epochs remain before the
final R100 handoff.

### Epoch-4 checkpoint audit

The immutable `checkpoint0004.pth` reached a stable 695,205,625-byte size and
was audited tensor-by-tensor against the sealed initializer. It records
`epoch=4`, `epoch_finished=true`, `iteration=0`, and exactly 28,505 successful
optimizer updates: five times the fixed 5,701-update epoch length, or 62.5% of
the 45,608-update Stage A contract. The full contract passed again:

- the changed set was exactly the nine allowlisted patch tensors and all 1,125
  frozen tensors remained bitwise equal to the initializer;
- all 1,134 model tensors and all 27 tensors across the nine AdamW states were
  finite;
- the optimizer contained exactly nine states and nine parameter-group
  entries;
- AMP scale remained 65,536 and its growth tracker was exactly 28,505, equal
  to the successful optimizer-update count.

The Stage A process and automatic R100/C100 controller remained alive, and GPU
training resumed after checkpoint serialization. Five of eight epochs have
therefore completed with no AMP skip, numerical failure, or frozen-boundary
violation; three epochs remain before the final R100 handoff.

### 30,000-update ownership audit

The interval checkpoint at exactly 30,000 successful optimizer updates was
audited tensor-by-tensor against the sealed initializer. It records `epoch=5`,
`iteration=1495`, and `optimizer_updates=30000`, consistent with the fixed
5,701-update epoch length. The complete ownership and numerical contract
passed:

- the changed set was exactly the nine allowlisted patch tensors, while all
  1,125 frozen tensors remained bitwise equal to the initializer;
- all 1,134 model tensors and all 27 tensors across the nine AdamW states were
  finite;
- the optimizer contained exactly nine states and 27 state entries;
- AMP scale remained 65,536 and its growth tracker was exactly 30,000.

The immediately preceding u29,700, u29,800, and u29,900 interval checkpoints
also passed metadata, finiteness, optimizer-ownership, and AMP-state audits.
The formal process and automatic downstream controller remained alive with
GPU memory near the requested 30 GB envelope. Stage A is therefore at 65.8% of
its 45,608-update contract with no observed AMP skip, numerical failure, or
frozen-boundary violation.

### Epoch-5 checkpoint audit

The immutable `checkpoint0005.pth` reached a stable 695,205,625-byte size and
was audited tensor-by-tensor against the sealed initializer. It records
`epoch=5`, `epoch_finished=true`, `iteration=0`, and exactly 34,206 successful
optimizer updates: six times the fixed 5,701-update epoch length, or 75% of
the 45,608-update Stage A contract. The complete contract passed:

- the changed set was exactly the nine allowlisted patch tensors and all 1,125
  frozen tensors remained bitwise equal to the initializer;
- all 1,134 model tensors and all 27 tensors across the nine AdamW states were
  finite;
- the optimizer contained exactly nine states and 27 state entries;
- AMP scale remained 65,536 and its growth tracker was exactly 34,206.

The Stage A process and automatic R100/C100 controller remained alive, and the
training log entered epoch 6 with zero AMP skips. Six of eight epochs have
therefore completed without a numerical failure or frozen-boundary violation;
two epochs remain before the final R100 handoff.

### Epoch-6 checkpoint audit

The immutable `checkpoint0006.pth` reached a stable 695,205,625-byte size and
was audited tensor-by-tensor against the sealed initializer. It records
`epoch=6`, `epoch_finished=true`, `iteration=0`, and exactly 39,907 successful
optimizer updates: seven times the fixed 5,701-update epoch length, or 87.5%
of the 45,608-update Stage A contract. The complete contract passed:

- the changed set was exactly the nine allowlisted patch tensors and all 1,125
  frozen tensors remained bitwise equal to the initializer;
- all 1,134 model tensors and all 27 tensors across the nine AdamW states were
  finite;
- the optimizer contained exactly nine states and 27 state entries;
- AMP scale remained 65,536 and its growth tracker was exactly 39,907.

The Stage A process and automatic R100/C100 controller remained alive, and the
training log entered the final epoch 7 with zero AMP skips. Seven of eight
epochs have therefore completed without a numerical failure or frozen-boundary
violation; one epoch remains before the final R100 handoff.

### 42,000-update ownership audit

The final-epoch interval checkpoint at exactly 42,000 successful optimizer
updates was audited tensor-by-tensor against the sealed initializer. It records
`epoch=7`, `iteration=2093`, `epoch_finished=false`, and
`optimizer_updates=42000`, leaving 3,608 updates in the fixed 45,608-update
Stage A contract. The complete ownership and numerical contract passed:

- the changed set was exactly the nine allowlisted patch tensors and all 1,125
  frozen tensors remained bitwise equal to the initializer;
- all 1,134 model tensors and all 27 tensors across the nine AdamW states were
  finite;
- the optimizer contained exactly nine states and 27 state tensors;
- AMP scale remained 65,536, with growth tracker exactly 42,000 and growth
  interval 1,000,000.

The preceding `u41200` through `u41900` interval checkpoints also passed
metadata, finiteness, optimizer-ownership, and AMP-state audits. The Stage A
process and automatic R100/C100 controller remained alive after serialization.
This is the last scheduled full live ownership audit; `checkpoint0007.pth`
must still pass the same complete audit before the controller's R100 output can
be accepted.

### Final Stage-A completion audit

The immutable `checkpoint0007.pth` completed the fixed eight-epoch schedule
and reached a stable size of 695,205,625 bytes. Its SHA-256 is
`fe20fe91f3c46b6d143db13c74817ff3aa810cc51d1579104913c3d23fec9a8b`.
The checkpoint records `epoch=7`, `iteration=0`, `epoch_finished=true`, and
exactly 45,608 successful optimizer updates. The full terminal audit passed:

- the changed set was exactly the nine allowlisted patch tensors, with all
  1,125 frozen tensors bitwise equal to the initializer;
- all 1,134 model tensors were finite;
- the optimizer contained exactly nine states and 27 finite state tensors;
- AMP scale remained 65,536, its growth tracker was exactly 45,608, and its
  growth interval remained 1,000,000.

The formal log spans 2026-08-15 02:09:24 through 13:09:24 and contains no AMP
skip, CUDA OOM, or non-finite failure. This closes the Stage-A contract and
authorizes the sealed checkpoint, rather than an interval or interrupted
checkpoint, as the sole source for R100.

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

### CPU load preflight

A CPU-only construction/load replay was run on 2026-08-15 with the exact
Ref-safe R100 config and the completed epoch-0 Stage-A checkpoint (the final
checkpoint has the same state-dict layout). It established the handoff before
the automatic controller reaches R100:

- all 938 shared ordinary-GDINO tensors had matching shapes and were bitwise
  equal after loading;
- the only 196 unexpected source tensors were the deliberately excluded
  Stage-A patch branch;
- the only 20 missing target tensors were the newly initialized score adapter;
- the real `main.py` freeze audit left exactly the eight rank tensors trainable
  (50,177 parameters), with the confidence tower and complete base frozen.

The preflight ended with `R100_STAGEA_CPU_LOAD_PREFLIGHT=PASS`. This is an
early compatibility proof, not a substitute for the final receipt: the R100
launcher must still authenticate `checkpoint0007.pth` and repeat the complete
Stage-A/trunk/update-count lineage audit.

### Downstream data preflight

The fixed R100/C100 inputs were re-audited before automatic handoff. Their row
counts and SHA-256 identities match the two-phase audit contract: 120,624
RefCOCO rows (`9578a59c...`), 120,191 RefCOCO+ rows (`015e6821...`), 80,512
RefCOCOg rows (`cd4eda88...`), and 60,000 confidence pairs (`90bb0702...`). A
full 381,327-row JSON/path replay found no missing image after legacy-path
remapping. The confidence source also has the exact required row semantics on
all 60,000 rows: `benchmark_dataft_alltn=true`, scope
`benchmark_dataft_alltn`, `proposalset_proxy_verified=false`, and
`global_tn_verified=false`. Its sidecar audit matches the fixed schema, row
count, and output hash. The canonical-class file resolves under the launcher's
default `/media/haoyi/T9/data` root and contains 2,048 entries.

Operationally, do not run broad `find /media/haoyi/T9` or `rg --files` scans
while this formal job is active. Such scans left reader processes traversing
the exFAT volume and temporarily held Stage A in `exfat_get_block` after the
u8400 checkpoint. Terminating only those scanner PIDs restored GPU execution;
u8500 then completed with no skipped optimizer update. Subsequent monitoring
uses explicit paths only.

The launcher is:

```bash
bash tools/run_stagea_b58_r100_c100.sh
```

It emits `pivot.stagea_b58_r100_receipt/v3` between R100 and C100. The receipt
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

### Accepted v3 lineage

The accepted downstream run is:

```text
outputs/stagea_b58_patch0006_realign_r100_c100_sealed_evaluator_v3_20260815/
```

Its R100 checkpoint is
`rank/milestones/checkpoint_iter_000100.pth`, SHA-256
`346e847228f7a14a70ee772233c8d5fb2b090aebab76d7deda981901e74cc2b7`.
The Stage-A/R100 receipt authenticates exactly 100 optimizer updates, bitwise
equality of all 938 ordinary-GDINO tensors to the completed Stage-A non-patch
trunk, eight rank tensors, and twelve still-initial confidence tensors. A
terminal CPU audit found exactly eight R100 optimizer states and 24 finite
AdamW state tensors.

Its C100 checkpoint is
`confidence/milestones/checkpoint_iter_000100.pth`, SHA-256
`c9737d6bcabec4325bd53b146782b82a4d1119237d01d87de9f8d2987e03000e`.
It records exactly 100 optimizer updates. Relative to R100, the changed set is
exactly all twelve confidence tensors; all eight rank tensors and all 938 trunk
tensors remain bitwise equal (946/946). Its optimizer contains exactly twelve
states, twelve parameter-group entries, and 36 finite AdamW state tensors.
Thus the two training phases have disjoint gradient ownership in the produced
checkpoints, rather than merely disjoint loss labels.

Two superseded diagnostic launches attempted to replay the historical Stage-A
launch worktree as the live R100/C100 source. They correctly failed before any
result could be accepted: first because the historical evaluator file also
drifted, then because that historical source predates the Ref-safe score route.
The evaluator was corrected to verify the Stage-A launch-source manifest as
forensic history (git commit plus preserved worktree patch), while still using
and sealing the current R100/C100/evaluation implementation. The focused
regression suite passed 9/9, and this correction was committed as `e864367`.

### Final sealed replay result

The accepted formal output is:

```text
outputs/stagea_b58_patch0006_realign_r100_c100_sealed_evaluator_v3_20260815/
  formal/c100_stagea_b58_full_20260815/
```

Both strict manifests beat the fixed historical B58 checkpoint:

| Manifest | B58 FPR95 | C100 FPR95 | Delta | False accepts | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict1607 | 0.498444 | **0.454263** | -0.044182 | 801 -> 730 | [-0.071109, -0.017509] |
| strict2031 | 0.512063 | **0.455933** | -0.056130 | 1040 -> 926 | [-0.079326, -0.032752] |

The 5,000-resample paired image-cluster bootstrap supports a lower candidate
FPR95 for both manifests (`P(delta < 0)=0.999` for strict1607 and `1.0` for
strict2031). Ref-safe deployment exactly preserved B58's Acc@0.5 correct count
on every Ref8 split:

| Split | Expressions | B58 correct | C100 correct | Acc@0.5 delta |
| --- | ---: | ---: | ---: | ---: |
| refcoco_val | 10,834 | 6,993 | 6,993 | 0 |
| refcoco_testA | 5,657 | 4,102 | 4,102 | 0 |
| refcoco_testB | 5,095 | 2,957 | 2,957 | 0 |
| refcocop_val | 10,758 | 7,475 | 7,475 | 0 |
| refcocop_testA | 5,726 | 4,305 | 4,305 | 0 |
| refcocop_testB | 4,889 | 3,108 | 3,108 | 0 |
| refcocog_val | 4,896 | 3,942 | 3,942 | 0 |
| refcocog_test | 9,602 | 7,876 | 7,876 | 0 |

The pre- and post-evaluation lineage receipts are byte-identical, both with
SHA-256 `b7a27f6309286681bf8d34fa84317c6966c8973a7ce4acfc5e8de842501b40a0`.
The final comparison SHA-256 is
`c06320d52c858d6045807b33b093e32890f7c485cf868d4f1e1f9f70d00f0bf5`,
and the postflight SHA-256 is
`1058b0cdfd488d479381f3c5f7d9d7446a9a4b367997247d20f9465ff490a9c8`.
`postflight.json` reports `status=passed`,
`both_strict_fpr95_strictly_lower=true`,
`all_ref8_splits_no_regression=true`, `rank_branch_unchanged_from_r100=true`,
and `stagea_r100_c100_all_sealed_gates_passed=true`.
