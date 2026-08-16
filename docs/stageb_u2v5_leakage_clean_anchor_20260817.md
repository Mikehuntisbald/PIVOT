# U2-v5 leakage-clean main-model anchor

Date: 2026-08-17

## Decision

The CVPR main-model candidate is rebuilt without diagnostic C100 confidence:

```text
full expression -> frozen B58 + positive-only R100
support patch   -> Stage-A patch196 -> U2-v4 category admission
detached score statistics -> fresh D3 confidence12
```

Admission and confidence are separate optimizer phases. Admission trains only
surface8 plus the U2 auxiliary residual8. Confidence trains only confidence12.
The B58 trunk, R100 rank8, patch backbone, and the other phase's parameters are
frozen. The Ref route has no B58 top-1 guard.

## Clean initializer

Builder: `tools/build_stageb_u2v5_clean_initializer.py`

Output (not committed):

- `/media/haoyi/T9/pivot/outputs/u2v5_leakage_clean_anchor_20260817/initializer/checkpoint_clean_init.pth`
- SHA-256: `ad7b3a563ef84356c6d952167ee6a48f615f8db887eba31bed92a81b0ba756a7`
- schema: `pivot.stageb.u2v5_clean_initializer/v1`
- model state: 1,165 tensors

Ownership is exactly trunk938 + positive-only R100 rank8 + R100 identity
confidence12 + Stage-A patch196 + legacy U0 admission shell11. The builder has
no C100 command-line input and rejects any C100 source in the serialized
contract.

Bound sources:

- Stage-A checkpoint0007: `fe20fe91f3c46b6d143db13c74817ff3aa810cc51d1579104913c3d23fec9a8b`
- positive-only R100 U100: `346e847228f7a14a70ee772233c8d5fb2b090aebab76d7deda981901e74cc2b7`
- Stage-A/R100 receipt: `f659504805f4b62cc5ea3afda5ab629e6e5391ca230da737583e3e9b8da7d659`
- legacy U0 shell: `c89e5dfba795fd8074a044f0c09d81c871705c20a1dbf819b9f16c770a2cba43`

The identity confidence12 state hash is
`521c33d6ccbd408b9935f41a3ef986f23676803e6931ccbf1b50fbc9650e5be0`.

## D3 confidence scope

D3 has 14,196 training rows and 1,570 calibration rows. Its sealed audit SHA
is `7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d`.
The audit records zero image overlap between D3 train/calibration and the union
of strict1607/strict2031, plus zero train/calibration image overlap.

D3 is deliberately **not** called global all-query supervision. Every row
retains:

- `tn_scope=proposal_covered_verified`;
- `global_tn_verified=false`;
- `all_900_gdino_queries_verified=false`;
- `global_max_label_is_semantic_extrapolation=true`.

The adapter therefore uses a separate scope code (3) and objective code (4),
`detached_recent_q05_proposal_covered`. The runtime requires the sealed D3
dataset audit and refuses a global/all-TN upgrade.

## Smoke evidence

Admission seed17 U1 changed only the declared admission16. Both auxiliary and
surface gradients were finite and nonzero; AMP skipped zero steps. Peak
reserved memory was 30.39 GB and minimum device-free memory was 3.99 GB.

Confidence seed17 U1 changed only confidence tensors; the other 1,153 tensors
were bitwise unchanged. Its confidence gradient was finite/nonzero, AMP
skipped zero steps, and peak reserved memory was 12.26 GB.

All outputs produced before the clean source commit are engineering probes,
not formal paper checkpoints.

## Formal protocol

Seeds are fixed to 17, 42, and 73. Admission uses the full U2-v4 replay
mechanism for 100 updates. Fresh confidence starts from each seed's selected
admission checkpoint and is trained on D3 only. Checkpoint selection may read
only the three Ref val splits and the 1,570-row D3 screen calibration.

Before any held-out read, bind the selected checkpoint SHA values, code commit,
configs, seeds, selection summaries, Ref8 split list, and both strict manifests
in a preregistration receipt. Ref8 and strict1607/strict2031 are then evaluated
once. Until that receipt exists, test and strict results are prohibited and no
headline claim is authorized.

## Commands

Admission config:
`config/ablations/cfg_stageb_u2v5_clean_admission_u100.py`

Confidence config:
`config/ablations/cfg_stageb_u2v5_clean_confidence_d3_u100.py`

Confidence dataset:
`config/datasets_stageb_u2v5_clean_confidence_d3.json`

Tests:

```bash
/home/haoyi/miniconda/envs/gdino5090/bin/python -m pytest -q \
  tests/test_stageb_u2v5_clean_anchor.py \
  tests/test_build_stageb_u2v4_training_initializer.py \
  tests/test_stageb_u2v4_legacy_training_contract.py \
  tests/test_stage_b_gdino_score_adapter.py \
  tests/test_stageb_u2v4_confidence_only_route.py
```

Current result: 51 passed.

## Locked val/calibration selection

All three formal admission runs completed 100 successful optimizer updates
with zero AMP skips. Their checkpoint SHA-256 values are:

- seed17: `57e614c8244a32409feb01aa7630230a1c7935f0945a045e20f1dca978ee4f38`;
- seed42: `0127d15ef14e772884681accf148bed25f368d1fa05980c66654fa2b706048d5`;
- seed73: `8f5f1c15f67c5cd59f21a15f0d83765f7fab71f6174cf24170a4ca46e705d3c0`.

The B16/W4/AMP val-only results are:

| seed | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|---:|---:|---:|---:|
| 17 | 0.665682 | 0.741123 | 0.806373 |
| 42 | 0.667344 | 0.744655 | 0.806985 |
| 73 | 0.666697 | 0.743819 | 0.806985 |

Every seed strictly exceeds B58 on every val split. Seed42 exactly reproduces
the clean U2-v4 replay result and is `+0.001108/+0.001580/+0.000000` versus
legacy U2 Gap3 on the three val splits.

Fresh D3 confidence was evaluated at U25/U50/U100 on the sealed 1,570-row
calibration surface. The cross-seed mean/worst-seed FPR95 values were:

| update | mean | worst seed |
|---:|---:|---:|
| 25 | 0.569639 | 0.577707 |
| 50 | 0.536518 | 0.542675 |
| 100 | 0.535456 | 0.544586 |

The locked robust rule minimizes worst-seed FPR95, then mean FPR95, then the
earlier update. It selects U50 for all seeds. U100 is not selected: its mean is
only 0.001062 lower, its worst seed is worse, and its score scale is much less
stable.

Preregistration is built by
`tools/build_stageb_u2v5_preregistration.py`. The receipt binds all six
selected checkpoints, val/calibration summaries, the Ref8 baseline, and both
strict manifests. No Ref test or strict result was read during this stage.
