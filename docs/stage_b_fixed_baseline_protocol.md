# Fixed Pure Stage-B Data-FT Protocol

This protocol rebuilds the comparison baseline as ordinary GroundingDINO. It
does not enable the patch branch, a Stage-B wrapper, or the GDINO score adapter.
The immutable baseline inputs are checked by SHA-256 before training.

## Training

Wait for the exposure-matched Stage-A rebuild to finish both epochs. Its final
checkpoint is:

```text
outputs/ogc_original_finetune_stage_a_rebuild_20260711/checkpoint0001.pth
```

Launch the one-epoch fixed Stage-B data-FT baseline with:

```bash
DATA_ROOT=/home/user/datasets/pivot_data \
tools/run_stageb_fixed_baseline.sh
```

The launch is fixed to two DDP ranks, batch 9 per rank, and global batch 18.
This preserves the original config's global batch and therefore keeps the
learning rate at `1e-4`. It also gives the same `floor(477357 / 18) = 26519`
optimizer steps as the single-process batch-18 recipe. Initialization uses
`--pretrain_model_path`, so Stage-B starts with a fresh optimizer and scheduler.

The five train sources, in order, are LVIS, COCO, RefCOCO+, RefCOCOg, and the
60k original allTN rows, with sampling weights `2, 2, 2, 2, 1`. The three-ref
candidate dataset is intentionally excluded from this baseline.

The authoritative output is:

```text
outputs/gdino_ft_stage_b_fixed_baseline_20260711/checkpoint0000.pth
```

`protocol_train_preflight.json` records the Stage-A checkpoint hash, all data
hashes, global batch, the full imported-config chain, and the recursive
repo-local training dependency closure. Before `protocol_train_complete.json`
can be written, postflight reconstructs the entire preflight from its recorded
inputs and requires exact equality, including the launch-wrapper hash. A config,
parent config, source datum,
checkpoint, launch argument, or code dependency changed during the run therefore
invalidates completion. Existing checkpoint outputs are rejected to avoid
accidentally mixing fresh initialization and resume semantics.

## Evaluation

Use the same wrapper for the baseline and every candidate. For example:

```bash
tools/run_stageb_fixed_protocol_eval.sh \
  --config config/ablations/cfg_stageb_from_gdino_ft_with_tn_alltn_tau05605_w036.py \
  --checkpoint outputs/gdino_ft_stage_b_fixed_baseline_20260711/checkpoint0000.pth \
  --output-dir outputs/gdino_ft_stage_b_fixed_baseline_20260711_eval_fixed
```

For an adapter candidate, pass its own config and checkpoint:

```bash
tools/run_stageb_fixed_protocol_eval.sh \
  --config CANDIDATE_CONFIG.py \
  --checkpoint CANDIDATE_CHECKPOINT.pth \
  --output-dir outputs/CANDIDATE_eval_fixed
```

The wrapper produces three isolated evaluations:

- `ref8`: all eight official RefCOCO, RefCOCO+, and RefCOCOg splits, top-1
  `acc50`, with one canonical record file per split.
- `strict2031`: the 2,031-row annotation-holdout verified TN manifest,
  restricted to `refcocop_val` and `refcocog_umd_val`.
- `strict1607`: the 1,607-row semantic Stage-B-union image-disjoint subset.

The evaluator uses exact FPR@95TPR (`>=` at the positive 5th-percentile
threshold), AMP, seed 42, and batch 16 by default. Baseline and candidate must
use the same runtime arguments. The preflight locks both TN input hashes and
the checkpoint hash, complete config import chain, resolved model architecture,
adapter scoring contract, and evaluator dependency closure. Postflight rejects
missing splits, reordered or duplicate samples, partial manifests, and any
invalid record.

`protocol_eval_complete.json` is a seal, not only a marker. Before an evaluation
directory is reused or paired, `stageb_fixed_protocol_audit.py verify-eval`
re-parses all three summaries and every record, rechecks finite metrics,
validity, manifest hash/N/order/identity, and requires the recomputed file hashes
to equal the sealed `outputs` payload exactly. It also recomputes the current
checkpoint/config records, imported-config chain, resolved model contract, and
evaluator code closure. Evaluator code lineage includes the adapter, dataset,
and transformer dependency closure, plus the fixed evaluation wrapper hash, for
both baseline and candidate preflights.

## Final Gate

After both complete evaluations exist, run:

```bash
tools/run_stageb_fixed_dual_gate.sh \
  outputs/gdino_ft_stage_b_fixed_baseline_20260711_eval_fixed \
  outputs/CANDIDATE_eval_fixed \
  outputs/CANDIDATE_vs_fixed_baseline_dual_gate
```

The command compares paired per-example records on both strict TN manifests.
It exits successfully only when the candidate has lower global FPR@95TPR and
higher `acc50` on every required Ref split, with identical manifest hashes,
orders, and denominators. Before scoring, it also requires identical evaluator
code hashes, AMP mode, batch size, seed, top-k, and TN threshold settings. The
resolved detector architecture must match, and every config path imported by
both recipes must have the same hash. The baseline checkpoint hash must trace
back to a replay-verified `protocol_train_complete.json`.
