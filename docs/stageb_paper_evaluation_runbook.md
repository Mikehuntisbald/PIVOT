> **Historical pre-ARROW artifact.** PIVOT names and schemas below identify the
> sealed implementation lineage and are intentionally preserved.

# Stage-B Paper Evaluation Runbook

This runbook covers the sealed evaluation path after a training runner has
written a completed `sequence_manifest.json`. It does not replace the paper
protocol. In particular, `screen_validation` is development-only and cannot be
consumed by the final results-manifest builder.

## L0-L4 Screen

Run the screen only for seed 17 after each corresponding 1,000-update training
sequence has completed. Each output root must be new.

```bash
PY=/home/haoyi/miniconda/envs/gdino5090/bin/python

$PY tools/run_stageb_paper_evaluations.py dry-run \
  --profile screen_validation \
  --training-run-root outputs/paper_cvpr_v1/token_ablation/L0/seed17 \
  --output-dir outputs/paper_cvpr_v1/evaluations/screen/L0/seed17

$PY tools/run_stageb_paper_evaluations.py run \
  --profile screen_validation \
  --training-run-root outputs/paper_cvpr_v1/token_ablation/L0/seed17 \
  --output-dir outputs/paper_cvpr_v1/evaluations/screen/L0/seed17
```

Repeat with `L1`, `L2`, `L3`, and `L4`. The profile runs exactly:

- `refcoco_val`, `refcocop_val`, and `refcocog_val`;
- all 1,570 rows of the sealed, image-disjoint, single-edit D3 calibration
  manifest; and
- no Ref test split, strict2031 row, or strict1607 row.

Completion requires `postflight.json` with `profile=screen_validation`, full
per-example records, zero invalid rows, and a replayable calibration
source-to-derived binding. Screen results are used only to reject numerically
invalid objectives. They do not authorize final-test checkpoint selection.

If a completed formal run used a non-default output root, it must be attested
by the completed serial queue that launched it. Pass that queue explicitly;
the evaluator verifies its immutable plan, runner SHA, completed item, resolved
output root, and detached launch/status evidence, then includes those artifacts
in the evaluation input hash:

```bash
$PY tools/run_stageb_paper_evaluations.py dry-run \
  --profile screen_validation \
  --training-run-root outputs/paper_cvpr_v1/token_ablation_frozen_v2/L0/seed17 \
  --training-queue-dir outputs/paper_cvpr_v1/queues/table_c_screen_l0_l4_seed17_b40_u1000_frozen_v2 \
  --output-dir outputs/paper_cvpr_v1/evaluations/screen/L0/seed17
```

An alternate training root without this queue attestation is rejected. Do not
move, copy, symlink, or overwrite a failed run to satisfy the default-root
check.

## Final Evaluation

Run the final profile only for checkpoints admitted by the predeclared
selection rule. The default profile is `final`, but keep it explicit in paper
commands.

```bash
$PY tools/run_stageb_paper_evaluations.py dry-run \
  --profile final \
  --training-run-root outputs/paper_cvpr_v1/token_ablation/L4/seed17 \
  --output-dir outputs/paper_cvpr_v1/evaluations/final/L4/seed17

$PY tools/run_stageb_paper_evaluations.py run \
  --profile final \
  --training-run-root outputs/paper_cvpr_v1/token_ablation/L4/seed17 \
  --output-dir outputs/paper_cvpr_v1/evaluations/final/L4/seed17
```

The same command accepts completed Table-B and Table-D run roots. For S3 it
selects only the final `confidence/checkpoint_iter.pth`; for every other row it
selects the canonical joint checkpoint. The final profile runs Ref8 plus
strict2031 in the first process and strict1607 with `--skip_ref` in the second.

## Results Manifest

The results builder can derive checkpoint, config, training data, Ref root,
and both strict roots directly from a completed final evaluation. A build-spec
run therefore needs only the training seed and evaluation root:

```json
{
  "train_seed": 17,
  "evaluation_root": "../../outputs/paper_cvpr_v1/evaluations/final/L4/seed17",
  "expected_training_run_id": "L4:17"
}
```

Use one such run for each declared seed in each experiment, then build and
validate the immutable final manifest:

```bash
$PY tools/build_stageb_paper_results_manifest.py \
  --spec config/paper_results/stageb_final_build_spec.json \
  --output outputs/paper_cvpr_v1/manifests/stageb_final.json \
  --validate --validation-bootstrap-iterations 8

$PY tools/aggregate_stageb_paper_results.py \
  --manifest outputs/paper_cvpr_v1/manifests/stageb_final.json \
  --output-dir outputs/paper_cvpr_v1/aggregate/stageb_final
```

`evaluation_root` mode rejects screen outputs, historical unsealed sources,
mixed explicit/derived artifact declarations, incomplete evaluation phases,
input-rehash drift, and missing final postflight contracts. The historical
GDINO baseline continues to use the explicit build-spec form because it has no
PIVOT training sequence manifest. If that baseline has one real checkpoint,
declare its experiment with `"expected_train_seeds": [42]` and
`"reference_role": "fixed_historical_checkpoint"`; declare each trained PIVOT
experiment with its actual seed set (for example `[17, 42, 73]`) and
`"reference_role": "training_seed_distribution"`. The aggregator compares
every candidate seed with the one fixed reference but keeps the baseline
sample size at one; duplicating the baseline checkpoint across seed labels is
not permitted.
