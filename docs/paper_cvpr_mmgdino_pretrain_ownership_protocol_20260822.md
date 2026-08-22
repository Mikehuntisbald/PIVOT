# MM-GDINO-T pretrained ownership protocol

## Question

This replay freezes the official MM-GDINO-T
Objects365+GoldG+GRIT-9M+V3Det pretrained checkpoint before any RefCOCO
task-specific full-model fine-tuning. It asks whether the pretrained 256-d
decoder-query representation makes Ranking and Rejection gradients compatible,
or whether capacity-matched hard isolation still improves deployment.

The rank and rejection heads are still trained on the same RefCOCO R100 and
leakage-clean D3 C50 schedules used by the e5/e6 controls. Therefore
"pretrained" describes the frozen trunk, not an absence of downstream head
supervision.

## Frozen trunk

- Checkpoint:
  `/media/haoyi/T9/external/mmgdino_l_baseline/weights/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`.
- SHA-256:
  `b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a`.
- Embedded endpoint: epoch 30, iteration 483,060; experiment name
  `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047`.
- The checkpoint contains the same 908 effective runtime tensors as e5/e6 plus
  one known non-persistent BERT `position_ids` buffer. No weight is updated.

This is not claimed to be image-disjoint from RefCOCO: broad grounding
pretraining may include overlapping source imagery. The clean statement is
"no RefCOCO task-specific trunk fine-tuning."

## Fixed owner matrix

Only two learned routes are run:

| Owner | Parameters | MAC/query, both outputs | Representation |
|---|---:|---:|---:|
| Shared-Wide | 100,362 | 99,424 | shared 210-d |
| Isolated | 100,358 | 98,816 | disjoint 128-d + 128-d |

Both consume `[query feature 256d; native score]`. They use two task-specific
AdamW states, zero weight decay, rank/rejection learning rates
`3e-5`/`1e-4`, clip 0.1, deterministic FP32, seeds 17/42/73, and the identical
`rank, confidence, rank` U150 schedule: R100 at batch 32 and C50 at batch 8.
Shared-128 is excluded.

Scheduled rank rows without an eligible IoU≥0.5 candidate are retained without
replacement. The existing valid-row mask gives them zero rank-margin loss;
residual regularization remains active. Rows without a hard negative still
fail closed.

## Endpoints

- RefCOCO TestAB micro P@1 is the direct e5 comparison.
- Test5 micro P@1 adds RefCOCO+ TestA/TestB and RefCOCOg UMD test, for 30,969
  expressions total.
- Strict-TN2031 FPR95 is the rejection endpoint; every bootstrap replicate
  recomputes each route/seed positive q05.
- Gradient probes use the same fixed eight rank/confidence batch pairs at
  U25/U50/U100/U150. Report mean, `P(cos<0)`, q05, minimum, and sign-conflict
  fraction for Shared-Wide. Isolated must have no cross-task autograd path.

Statistics use 5,000 paired image-cluster bootstrap replicates with PCG64 seed
`20260823`. A global image draw covers all five Test5 splits; Strict2031 draws
carry complete positive/negative pairs. The planned contrast is
Isolated−Shared-Wide for REC and Shared-Wide−Isolated for FPR95 reduction.

No checkpoint, milestone, score threshold, loss, margin, seed, sample alias,
or evaluation surface may be changed after any pretrained-trunk GPU forward.
