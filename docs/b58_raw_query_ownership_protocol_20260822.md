# B58 100k raw-query ownership replay

This replay closes the head mismatch in the direct-parent versus B58 analysis.
The only formal frozen-model change is the 938-tensor trunk checkpoint:

- parent: `ogc_original_finetune_stage_a/checkpoint0001.pth`;
- descendant: `gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/checkpoint0001.pth` (B58).

The checkpoints have the same effective schema and differ in 727 tensors; 211
tensors are bitwise unchanged.  Both stages use the mature capacity-matched
owners:

| Owner | Parameters | MAC/query | Representation |
|---|---:|---:|---:|
| Shared-Wide | 100,362 | 99,424 | shared 210-d |
| Isolated | 100,358 | 98,816 | disjoint 128-d + 128-d |

Everything else is copied bitwise from the parent replay: seeds 17/42/73,
rank/D3 schedules and order, U150 (`rank, confidence, rank`), R100+C50,
batches 32/8, task-specific Adam states, zero weight decay, objectives,
full-expression mean native score, gradient probes, Test5, TestAB, and
Strict2031.

The primary cross-stage statistic is a paired image-cluster
difference-in-differences:

`(Isolated - Shared-Wide)_B58 - (Isolated - Shared-Wide)_parent`.

For FPR95 the within-stage effect is `Shared-Wide - Isolated`, so positive
means isolation reduces FPR95.  Every one of 5,000 PCG64 bootstrap replicates
uses the same image draw across both trunks, both owners, and all seeds, and
recomputes each route/seed/trunk positive q05.

No checkpoint, score reduction, threshold, milestone, or Gap is selected from
the replay results.  The preregistration must exist before the first B58 GPU
forward.
