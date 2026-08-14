# Stage A B58 trunk + checkpoint0006 patch realignment

## Decision

This Stage-A revision uses the sealed GDINO Stage-B data-FT B58 checkpoint as
the query, box, image, and text trunk. It transfers only the independent patch
scoring tensors from the historical Stage-A `checkpoint0006`, then trains that
small patch surface against fixed B58 queries.

This is deliberately different from the historical lineage in which Stage B
rank/confidence training started from Stage A. Here B58 is moved upstream only
for a new Stage-A experiment: it owns the mature query semantics and geometry;
the patch branch is realigned to that fixed coordinate system. The result does
not by itself establish a Stage-B FPR95 improvement. That claim requires a
downstream rank/confidence replay using this new Stage-A checkpoint.

## Exact sources

| Role | Checkpoint | SHA-256 |
| --- | --- | --- |
| Fixed query/box/text/image trunk | `/media/haoyi/T9/gdino/outputs/gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/checkpoint0001.pth` | `b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157` |
| Patch projection/normalization/temperature | `/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch/checkpoint0006.pth` | `a4f153c8cbd9b408b9479901e27ec486a10f393013193d44b0da1dcd1888cb91` |
| Composed initializer | `/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_20260814_initializer.pth` | `25ca02e5ec7b127f1d90f5642f7d36035c0eb71669ad9aa85cd158f12eedf3b8` |

The initializer contains 1,134 tensors:

- 938 direct B58 trunk tensors;
- 187 `patch_encoder.backbone.*` aliases mirrored from the B58 main backbone;
- 9 independent patch tensors copied bitwise from checkpoint0006;
- no `patch_dn_tgt` tensor.

The full ordered model-tensor digest is
`8aa54cd6510ed06a4b4533e1e67797cc855493ad9974dcc4fe18cea62efe2f5c`.

## Why the patch backbone is not copied from checkpoint0006

`patch_encoder.backbone` and the main `backbone` are the same Python module.
Their state appears under two prefixes in a checkpoint, but both names target
the same parameters. Copying all `patch_encoder.*` keys from checkpoint0006
would therefore overwrite B58's image trunk late in `load_state_dict` and break
the intended lineage. The initializer mirrors B58 under both prefixes and uses
checkpoint0006 only for:

- `patch_encoder.input_proj.*`;
- `patch_encoder.norm.*`;
- `query_proj_for_patch.*`;
- `patch_logit_scale`.

## Gradient ownership

Exactly 9 tensors / 263,681 parameters are trainable. The decoder, encoder,
main/shared image backbone, BERT/text fusion, bbox head, class head, and query
state are frozen. Patch DN queries are disabled. During training, all frozen
query-producing modules remain in evaluation mode so DropPath or dropout cannot
shift the query distribution seen by the patch projection.

The patch head is allowed to learn a new readout of fixed B58 queries. No patch
loss gradient can update the B58 query representation or box geometry. This is
the mechanism that removes the historical patch-versus-text/box gradient
conflict; it does not merely reduce the learning rate of a shared decoder.

## Reproduce

Rebuild the initializer only if the recorded output is unavailable:

```bash
PYTHONPATH=/media/haoyi/T9/pivot \
/home/haoyi/miniconda/envs/gdino5090/bin/python \
  tools/build_stagea_b58_patch0006_initializer.py build \
  --output /media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_20260814_initializer.pth
```

The formal single-GPU launch is:

```bash
bash tools/run_stagea_b58_patch0006_realign.sh
```

Extra `main.py` arguments can be appended to the launcher. Resume a produced
training checkpoint with `--resume`; do not pass the initializer as `--resume`,
because it intentionally has no optimizer/scheduler state.

## Verified preflight

- initializer schema and source hashes verified;
- strict model load: 0 missing and 0 unexpected keys;
- model state: 1,134 tensors;
- trainable surface: 9 tensors / 263,681 parameters;
- model/backbone/decoder training flags after ownership setup: all frozen query
  producers are in evaluation mode;
- focused tests: `5 passed`.

A one-update runtime smoke also completed with `batch_size=1`, at most two
support patches, and sanity rendering disabled. It exercised the real
LVIS/COCO loaders, strict initializer validation, CUDA forward/backward, AMP,
optimizer step, and interrupt-checkpoint write:

`/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_20260814_smoke_u1/checkpoint_iter.pth`

Its SHA-256 is
`fbdb35e2f0cbf91b64929c37e8e3db42e5336b757b8c312fa794ba5ed1ddbe23`.
The checkpoint records `optimizer_updates=1` and
`checkpoint_reason=max_train_iters`. A tensor-by-tensor pre/post audit found
zero changes among all frozen tensors and changes in all 9 trainable patch
tensors. This smoke is an execution check, not a quality measurement or a
substitute for the then-current batch-18 run.

## VRAM push

The first formal batch-18 launch reached step 500 on the full support-patch
surface, demonstrating batch 18 through that milestone. Because this
revision backpropagates through only the 9 patch-owned tensors and does not keep
decoder/encoder/backbone backward graphs, the physical batch was first raised
to 24, probed at 40, and then set to 38 while keeping `lr=2e-5` unchanged. This
isolates the memory/throughput change from learning-rate tuning.

The batch-24 run was stopped cleanly after update 673 and retains its exact
iteration checkpoint at
`/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_bs24_formal_20260814/checkpoint_iter.pth`
(`epoch=0`, `next_iter=673`, `optimizer_updates=673`, `reason=signal`). It is
preserved for audit but is not resumed after changing the physical batch,
because doing so would change sample exposure at the iteration boundary.

Batch 24 completed 673 CUDA updates on the target RTX 5090 with no AMP skip or
OOM. Peak PyTorch allocation reached 10.0 GiB; the training process occupied
about 17.7 GiB and whole-card use was about 19.7/32.6 GiB including resident
desktop/ComfyUI processes and CUDA caching.

To move closer to the requested 30 GiB operating point, a full-data batch-40
probe completed one optimizer update and saved
`/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_vram_probe_bs40_u1_20260814/checkpoint_iter.pth`
with `optimizer_updates=1` and `reason=max_train_iters`. This established that
batch 40 could execute, but did not establish long-run stability. The full
support-patch cap remains unchanged, because reducing it would change the
Stage-A data contract.

The fresh batch-40 formal launch reached update 1191 with no AMP-skipped steps.
Its observed whole-card use was about 30,452/32,607 MiB, leaving only about
1,657 MiB free. It then exited while the NVIDIA driver was temporarily
unavailable; the log contains no Python traceback or explicit CUDA OOM. The
last exact interval checkpoint is preserved at update 1000 in
`/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_bs40_formal_20260814/checkpoint_iter.pth`.

Because one-update success did not survive the longer run, batch 40 is rejected
as the formal operating point. Batch 38 is the selected fallback and starts
fresh from the sealed initializer in
`/media/haoyi/T9/gdino/outputs/stageA_b58_trunk_patch0006_realign_bs38_formal_20260814`;
the batch-40 checkpoint is retained for audit but is not resumed across the
physical-batch change. Iteration checkpoints remain every 500 optimizer
updates.

The first batch-38 process was interrupted at update 160 by a full host reboot
at 2026-08-15 00:52, not by a Python exception, AMP skip, or CUDA OOM. Its
directory was preserved with suffix `_host_reboot_u0160`. The fresh formal run
keeps batch 38 but writes an exact iteration checkpoint every 100 updates to
limit recovery loss if the host restarts again.
