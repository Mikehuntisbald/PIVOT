# Stage-B GDINO Total-Trust Confidence Branch

This branch is a confidence-only continuation of the frozen GDINO Stage-B
data-FT rank checkpoint.  The rank adapter is frozen and the confidence
adapter is the only trainable owner, so rank and calibration gradients do not
share parameters.

## Architecture and score ownership

This is one model checkpoint with one shared frozen GDINO image/text backbone,
not two complete GDINO copies. The backbone produces one set of detached query
features and base phrase scores. Two parameter-disjoint lightweight adapter
heads consume those same frozen inputs:

```text
                  one frozen GDINO backbone
                 query_hs + base phrase score
                             |
                       detach boundary
                             |
               +-------------+-------------+
               |                           |
       frozen R100 rank head        trainable confidence head
       per-query residual           image-expression scalar gate
               |                           |
  rank_score(q) = base(q)+r(q)  confidence_score(q) = base(q)+g(image,text)
               |                           |
       Ref localization/ranking       TN rejection / FPR95
```

The rank head owns eight adapter tensors and the confidence head owns twelve.
The selected confidence run never updates the shared backbone or the rank
head. Evaluation records this explicitly as
`shared_frozen_gdino_trunk_independent_rank_confidence_adapters`, with
`stage_b_gdino_rank_score` as the Ref score and
`stage_b_gdino_confidence_score` as the TN score. Thus "rank/conf dual tower"
is a useful shorthand only for the two independent heads; the expensive
backbone is shared and executed once.

For an image/phrase pair, inference uses the deployed score directly:

```text
s_conf(b, q) = s_base(b, q) + g(image, expression)
S+ = max_q s_conf(b+, q)
S- = max_q s_conf(b-, q)
```

The positive protection term is attached to `S+`, rather than to the hidden
gate value.  The recent detached positive queue supplies a q05 threshold. A
zero-valued straight-through term gives that threshold the current positive
mean's backward derivative:

```text
tau_st = stop(tau_queue_q05) + mean(S+) - stop(mean(S+))
L = L_TN(S-, tau_st)
  + lambda_score * relu(tau_queue_q05 - margin - S+)
  + lambda_pair * softplus(S- - S+ + pair_margin)
```

Only the threshold statistics and queue contents are detached. The deployed
positive and negative scores remain live, which removes the old gate-vs-score
gradient conflict while preserving TN tail pressure. The default
`lambda_pair` is zero so the confidence branch has one primary tail direction;
the pair term remains available as an explicit ablation. At the parameter
level, `confidence_only` freezes the rank owner, and the adapter detaches both
query features and base scores before the confidence path. Rank gradients were
zero throughout C0-C100 and the selected checkpoint's rank tensor is bitwise
identical to R100.

## Formal result

The authenticated C100 checkpoint beats the fixed historical GDINO Stage-B
data-FT B58 baseline on both strict manifests. The paired intervals use 5,000
image-cluster bootstrap resamples and recompute each model's q05 threshold in
every resample.

| Manifest | Historical B58 FPR95 | Total-trust C100 FPR95 | Delta | False accepts | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict1607 | 0.498444 | **0.454263** | -0.044182 | 801 -> 730 (-71) | [-0.071109, -0.017509] |
| strict2031 | 0.512063 | **0.455933** | -0.056130 | 1040 -> 926 (-114) | [-0.079326, -0.032752] |

The postflight decision is `fpr95_goal_met`. Against the earlier P50 replay,
C100 is also numerically lower on strict1607 (`0.455507 -> 0.454263`) and
strict2031 (`0.462334 -> 0.455933`), but those smaller paired differences are
not statistically resolved: their 95% intervals are respectively
`[-0.019036, 0.014572]` and `[-0.028500, 0.008257]`.

The evidence therefore supports two different claims:

- C100 is significantly better than historical GDINO Stage-B data-FT B58 on
  both strict manifests.
- C100 has a better point estimate than the older P50/V50+ confidence result,
  but it is statistically tied under the paired bootstrap and must not be
  described as a significant win over P50.

## Why FPR95 improved

The improvement comes from aligning ownership, training signal, and deployed
score rather than adding another backbone:

1. `confidence_only` freezes the authenticated R100 rank owner. Confidence
   gradients cannot trade Ref ranking quality for TN calibration, and the
   shared GDINO inputs are detached at the adapter boundary.
2. P50's trust constraint protected the auxiliary hidden gate. Total-trust
   instead protects the final positive score `max_q(base_score + gate)`, which
   is exactly the quantity that establishes the deployed 95%-TPR threshold.
3. P50 retained an always-active pair margin with weight `0.25`. Total-trust
   sets it to zero and concentrates its capacity on the distribution-level
   positive-q05/TN-tail operating point used by FPR95.
4. The C100 run uses 60,000 `benchmark_dataft_alltn` pairs, whereas the older
   P50 recipe used 17,829 semantic-verified pairs. This broader TN exposure is
   part of the result, so the small C100-versus-P50 difference cannot be
   attributed to the objective alone without a matched-data ablation.

Consistent with that interpretation, C100's AUROC rises slightly relative to
P50, while its per-pair win rate falls slightly. The gain is therefore a
better global tail operating point, not a universal improvement on every
positive/TN pair or every Stage-B metric.

The full Ref-8 report is retained as a diagnostic, not an FPR acceptance gate:

| Dataset | val | testA | testB/test |
| --- | ---: | ---: | ---: |
| RefCOCO | 0.644822 | 0.725296 | 0.579588 |
| RefCOCO+ | 0.695854 | 0.752008 | 0.636531 |
| RefCOCOg | 0.804739 | - | 0.820975 |

Some Ref-8 counts differ slightly from B58 because the frozen rank input is the
authenticated R100 continuation. Confidence training did not change that rank
route: its tensor SHA-256 remains
`90f8d970ffeb2b906acedc6b908ecc96802acc88362ee1a51d04b5c93e0fe335`.

## Selected checkpoint

```text
outputs/stageb_gdino_adapter_total_trust_from_legacy_b58_r100_20260813/
  confidence/milestones/checkpoint_iter_000100.pth
  SHA-256 cb60942536dacf6919210118d9e7fe25695b0f05c10ccdf2a067f1eeef3897a0
  confidence/milestones/checkpoint_iter_000100.audit.json
  SHA-256 49611a5dc5d13e26d392fda32100eee7074db9b1cb42143219951a146eba1e2d
  confidence/milestones/checkpoint_iter_000100.evaluation.json
  SHA-256 8d7eeb86b2d66156be10d4e60c7c5032ff4a8c5c3d98729cd4606b7ba90fcd4c
```

The formal evaluation is sealed under
`formal/c100_historical_b58_full_20260813/`. Its pre/post lineage receipts are
byte-identical. The postflight and historical comparison hashes are:

```text
postflight.json
SHA-256 547a8efe53ce362aa12b84f37ffe4e534d3c0443d27c4aa265ca10f284bc4250
comparisons/historical_b58_comparison.json
SHA-256 0311aa9bd98b383d459e414950c3dc2f89b94ad9703ce97a57cc6e528b9447df
```

## Lineage and reproduction

The authenticated local lineage is:

```text
/media/haoyi/T9/gdino/outputs/
  gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/checkpoint0001.pth
  SHA-256 b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157
    -> outputs/paper_cvpr_v1/legacy_replay/rank_r100_seed42_b32_v3/
       milestones/checkpoint_iter_000100.pth
       SHA-256 a933725aab2226d35aa1cd94992c154b332bbbbc3406060bbda061cef4959dd5
```

The R100 transition is sealed by
`outputs/paper_cvpr_v1/legacy_replay/legacy_r100_p50_exact_replay_receipt.json`.
The profile launcher validates the receipt and both checkpoints before it
prints a dry-run or starts training. It runs confidence-only at global batch
eight on the available single GPU (`world_size=1`, `batch_size=8`,
`distributed=false`):

```bash
tools/run_stageb_gdino_adapter_total_trust_probe.sh \
  --confidence-max-target 100 --dry-run

tools/run_stageb_gdino_adapter_total_trust_probe.sh \
  --confidence-max-target 100
```

An explicit `--rank-checkpoint R.pth --rank-audit RECEIPT.json` can replace
those defaults. The compatibility shim also accepts a historical P3 rank
milestone, while writing new confidence records under the total-trust schema.
The final summary binder permits this historical schema only at the root rank
node; every new confidence node must remain total-trust tagged.
Historical rank code-closure records remain sealed inputs rather than being
rewritten to the current total-trust hashes; the rank checkpoint, config, data,
and lineage identities are still revalidated.

The resulting audit schema is
`stageb-gdino-adapter-total-trust-probe-v1`; it is intentionally distinct from
the historical P3 schema.

The full fail-closed evaluator binds the B58 identity, C100 audit and R100 rank
hash, evaluation code and data hashes, score routes, ownership fields, per-row
provenance, and pre/post checkpoint lineage:

```bash
/home/haoyi/miniconda/envs/gdino5090/bin/python \
  tools/run_stageb_gdino_adapter_total_trust_evaluation.py dry-run \
  --checkpoint outputs/stageb_gdino_adapter_total_trust_from_legacy_b58_r100_20260813/confidence/milestones/checkpoint_iter_000100.pth \
  --checkpoint-audit outputs/stageb_gdino_adapter_total_trust_from_legacy_b58_r100_20260813/confidence/milestones/checkpoint_iter_000100.audit.json \
  --output-dir OUTPUT_DIR
```

## Stability boundary

C100 is an intentional early-stop selection. Its training segments report no
AMP-skipped updates and zero rank pre/post-clip gradient. An exploratory
continuation beyond C100 was stopped after confidence gradients and scores ran
away; the live `confidence/checkpoint_iter.pth` around C200 has no milestone
audit and must not be used.

The failure is internal to the code-3 confidence surrogate, not cross-branch
rank/confidence interference. With batch eight, the 512-entry positive queue
lags the current model by roughly 64 updates. The negative loss is evaluated
in the forward pass against that stale q05, while the straight-through term
uses the current positive mean in the backward pass. Once the current positive
tail moves far ahead of the bank, this forward/backward mismatch permits an
unbounded common gate shift and eventually produces AMP overflows. Therefore
this result authenticates C100 only; extending the training horizon requires a
new objective schema with a bounded, forward-consistent positive anchor.
