# GDINO Adapter Two-Phase Probe Protocol

This protocol trains localization/ranking and image-expression confidence in
separate optimizer phases on top of the same frozen pure GroundingDINO Stage-B
data-FT checkpoint. It is a probe protocol; acceptance still requires the fixed
eight-split RefCOCO and strict TN dual gate.

## Fixed phase contract

Phase R uses:

- `cfg_stageb_gdino_score_adapter_rank_three_ref.py`
- the three positive RefCOCO, RefCOCO+, and RefCOCOg ODVG sources, sampled with
  equal source weights
- `rank_only`, no TN scope, rank LR `3e-5`
- base-wrong repair rows and base-correct preservation rows are normalized as
  two separate global DDP classes. A rare repair row therefore cannot be
  diluted by easy, zero-loss preservation rows on either rank
- an already-correct row may shrink its frozen positive-negative score gap by
  at most `0.02`; the protection target is `base_gap - 0.02`, not a cap that
  permits a large frozen gap to collapse to `0.1`
- `data_aug_hflip_prob=0`: captions are not rewritten by the image/box
  horizontal-flip transform, so disabling the flip prevents left/right text
  from becoming inconsistent with the transformed image
- two DDP ranks, batch 4 per rank, global batch 8

Phase C uses:

- `cfg_stageb_gdino_score_adapter_dataft.py`
- exactly the audited 60,000 `benchmark_dataft_alltn` pairs
- `data_aug_hflip_prob=0` and an explicit `neg_episode_prob=0`; the latter
  prevents `PatchEpisodeDataset` from replacing the sealed image/text pair by
  internally sampling another negative episode
- `confidence_only`, gate LR `3e-4`, recent-history queue size 512 and warmup
  count 256
- `detached_recent_q05_trust`: a detached recent-positive q05 threshold plus
  positive trust margin `0.02` at weight `1.0`
- two DDP ranks, batch 4 per rank, global batch 8

The current confidence recipe is P3. Its recent-positive queue holds 512
scores and becomes the threshold source after at least 256 valid scores. The
queue q05 used in the forward value is detached; before warmup, the current
positive q05 is also detached. To avoid rewarding a meaningless common
downward translation of both positive and negative gates, the loss uses the
zero-valued gradient proxy

```text
threshold_for_loss = detached_q05
                   + mean(positive_gate)
                   - detach(mean(positive_gate))
```

This is numerically identical to the detached q05, but gives the threshold the
same shared-translation gradient as the current gates, cancelling that degree
of freedom in the negative loss. Positive support is guarded separately by the
linear hinge

```text
Ltrust = mean(relu(-0.02 - positive_gate))
```

It is deliberately not a squared trust penalty. These definitions, the queue
capacity/warmup, and the trust hyperparameters are sealed into each C
milestone audit. They are implementation safeguards, not evidence that the
formal FPR/AP gates have passed.

R preserves iterations 50, 100, 250, 500, 1000, 2000, and 5000. C remains
bounded to iterations 50, 100, 250, and 500. The first R invocation uses
`--pretrain_model_path` from the completed fixed pure Stage-B data-FT
checkpoint. Later R invocations use `--resume`. The first C invocation uses
`--pretrain_model_path` from any selected, audited R milestone, which creates a
fresh C optimizer, scheduler, criterion, and tail queue. Later C invocations use
`--resume`. `--resume` must never cross the R/C boundary.

Run a CPU-only static audit and print every launch command:

```bash
tools/run_stageb_gdino_adapter_two_phase_probe.sh --dry-run
```

After the fixed baseline exists, train R:

```bash
tools/run_stageb_gdino_adapter_two_phase_probe.sh \
  --phase rank \
  --baseline-checkpoint \
    outputs/gdino_ft_stage_b_fixed_baseline_20260711/checkpoint0000.pth
```

The default R prefix stops at R500. Extend the same output directory without
changing its preflight or skipping an audit-chain node:

```bash
tools/run_stageb_gdino_adapter_two_phase_probe.sh \
  --phase rank --rank-max-target 1000 --continue
tools/run_stageb_gdino_adapter_two_phase_probe.sh \
  --phase rank --rank-max-target 2000 --continue
tools/run_stageb_gdino_adapter_two_phase_probe.sh \
  --phase rank --rank-max-target 5000 --continue
```

`--rank-max-target` must be one of the preserved R milestones. It only bounds
the current launcher invocation; it is deliberately not part of the immutable
phase preflight, so an audited prefix can be extended in place. Every extension
replays the complete existing chain and resumes from the immediately adjacent
milestone.

Select an R milestone using localization diagnostics, then initialize C. This
example selects R250:

```bash
tools/run_stageb_gdino_adapter_two_phase_probe.sh \
  --phase confidence \
  --rank-selection 250
```

`--rank-selection` accepts every R milestone through R5000.
`--rank-checkpoint` and `--rank-audit` can instead identify an R milestone from
another output root. The audit file is mandatory because C preflight checks the
selected R milestone by recursively replaying its phase preflight, segment
ancestry, previous/recovery chain, checkpoint state, and file hashes.

## Recovery and fail-closed behavior

A fresh run rejects a non-empty phase directory. Continue an interrupted run
only with:

```bash
tools/run_stageb_gdino_adapter_two_phase_probe.sh --phase rank --continue
```

The trainer writes a live `checkpoint_iter.pth` every 50 iterations and on a
signal. The wrapper validates a live checkpoint before continuation, preserves
it under `recovery/`, and resumes from that immutable copy. An exact milestone
is copied under `milestones/` before the live path can be overwritten.
Every invocation also has an immutable `*.lineage.json`. Recovery must match
the source recorded for the current segment; a checkpoint forked from another
initial checkpoint or milestone is rejected even when its iteration matches.

Each milestone audit records and verifies:

- checkpoint file SHA-256 and tensor-state hashes for base, rank, and confidence
- epoch, absolute iteration, `max_train_iters`, and checkpoint reason
- config, dataset, source-data, and relevant code hashes from phase preflight
- the recursive imported-config chain and recursive repo-local Python dependency
  closure, including dataset, transformer, and adapter implementations
- complete criterion, optimizer, scheduler, scaler, and RNG resume state
- exact phase criterion code, TN scope code, one-branch optimizer ownership, and
  phase LR
- exact confidence-objective code, trust hyperparameters, queue capacity/warmup,
  finite queue tensors, and valid queue count/pointer state; every C milestone
  must have reached the 256-sample warmup
- zero-valued positive-gate translation proxy and linear
  `mean(relu(-0.02 - positive_gate))` trust-loss semantics
- disabled caption-unsafe horizontal flipping and explicit zero negative-episode
  resampling for the sealed confidence pairs
- R: frozen base and unchanged confidence branch across milestones
- C: frozen base and bitwise unchanged selected-R rank branch at every milestone
- global batch 8 and explicit `pretrain_model_path`/`resume` lineage

Changing config, data, code, the initial checkpoint, or a preserved milestone
after preflight makes `--continue` fail. Use a new output root for a changed
experiment.

## P0 zero-init parity

P0 is an evaluation-only checkpoint made by adding the identity-initialized
adapter to the fixed pure baseline model state. It deliberately omits optimizer,
scheduler, and criterion state so it cannot be mistaken for a resumable phase.

```bash
/usr/bin/python3 tools/make_stageb_gdino_adapter_p0.py create \
  --baseline-checkpoint \
    outputs/gdino_ft_stage_b_fixed_baseline_20260711/checkpoint0000.pth \
  --checkpoint \
    outputs/stageb_gdino_adapter_p0_20260711/checkpoint_p0.pth
```

The sidecar audit proves that all base tensors are unchanged, both final adapter
outputs are exactly zero, and random-query rank/confidence scores are bitwise
equal to the base score.

Evaluate P0 on exactly the fixed baseline records and require exact record
parity:

```bash
tools/run_stageb_gdino_adapter_probe_eval.sh \
  --checkpoint outputs/stageb_gdino_adapter_p0_20260711/checkpoint_p0.pth \
  --label p0 \
  --p0-parity
```

Before reading records, the wrapper reruns the P0 checkpoint audit and requires
it to match `checkpoint_p0.pth.audit.json` exactly. This checkpoint-level proof
is necessary because Ref records expose the deployed top-1 IoU, not every query
score: it proves that both adapter outputs are still exact identity functions.
The record parity check then compares manifest hash/order/N plus `sample_id`,
`image_id`, `ann_id`, `ref_id`, and `sent_id`, and every exposed deployed score
and IoU. Its default tolerance is zero. It also requires all eight Ref splits,
strict2031 with exactly 2031/2031 valid aligned records, and strict1607 with
1607/1607. Records alone are not accepted as P0 identity proof.

## Candidate evaluation

Use the same wrapper for any preserved R or C checkpoint. The wrapper does not
guess or hard-code a candidate config: after replaying the supplied checkpoint
audit, it evaluates with the exact config path sealed by that audit. Thus an R
checkpoint uses the rank-only recipe, a C checkpoint uses its confidence recipe,
and a semantic-confidence checkpoint uses its separately audited semantic
recipe:

```bash
tools/run_stageb_gdino_adapter_probe_eval.sh \
  --checkpoint \
    outputs/stageb_gdino_adapter_two_phase_probe_20260711/confidence/milestones/checkpoint_iter_000250.pth \
  --checkpoint-audit \
    outputs/stageb_gdino_adapter_two_phase_probe_20260711/confidence/milestones/checkpoint_iter_000250.audit.json \
  --label c250
```

For milestone selection, use diagnostic mode:

```bash
tools/run_stageb_gdino_adapter_probe_eval.sh \
  --checkpoint \
    outputs/stageb_gdino_adapter_two_phase_probe_20260711/rank/milestones/checkpoint_iter_000100.pth \
  --checkpoint-audit \
    outputs/stageb_gdino_adapter_two_phase_probe_20260711/rank/milestones/checkpoint_iter_000100.audit.json \
  --label r100 \
  --diagnostic
```

Before training R, inspect each fixed-baseline Ref summary's
`recall50@all_queries`. The evaluator computes this from all frozen query boxes
in the same forward as top-1 accuracy. If a split has no strict headroom above
the baseline `acc50`, a score-only R branch cannot improve that split and must
not be extended blindly. Evaluate R50/R100/R250/R500 on complete Ref splits
first; continue to R1000/R2000/R5000 only while repair gains and held-out
all-eight-split accuracy remain aligned.

R has an identity confidence branch by construction, so its FPR is expected to
match P0/baseline and cannot pass the final strict-improvement gate. Early C
milestones may also miss one metric. `--diagnostic` (or `--allow-gate-fail`)
still writes the paired-protocol audit, strict2031 and strict1607 FPR comparison
JSON/Markdown, and both dual-gate JSON reports. It returns success only when the
record protocol itself is valid; it records the missed metric gate in
`diagnostic_status.json` and makes no final acceptance claim. Without this flag,
the same missed metric gate remains a non-zero, fail-closed result.

The wrapper runs the fixed protocol: all eight RefCOCO-family splits,
strict2031, and strict1607. The postflight requires complete valid manifests and
identity alignment. It then invokes the fixed paired dual gate. A candidate is
accepted only if exact global FPR@95TPR is lower than the fixed pure data-FT
baseline and Ref acc50 is higher under the unchanged paired-record protocol.
Formal non-P0 evaluation always requires `--checkpoint-audit`. Two-phase audits
are fully replayed by the two-phase auditor. The verified semantic-confidence
schema is automatically dispatched to its own `verify-evaluation` command;
unknown audit schemas fail. Completed evaluation reuse is also re-audited from
the current summaries and records and must exactly match its completion seal.

Run `--dry-run` first to perform CPU-only static audits and print the evaluation
commands without occupying a GPU.
