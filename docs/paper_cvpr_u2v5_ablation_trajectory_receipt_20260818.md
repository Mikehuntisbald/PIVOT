# U2-v5 CVPR ablation trajectory receipt (2026-08-18)

## Scope

This receipt closes the training and mechanism-selection stage of the clean
U2-v5 ablation block.  It precedes and is independent of confirmatory Test5 and
strict2031 interpretation.

## Formal matrix

- Registry: 14 trainable rows × seeds 17/42/73 = 42 trajectories.
- Admission: A1/A2/A3/A4, U100, physical B56.
- Confidence/data: C1/C2/C4 and D1/D2/D2m/D3m, U50, physical B8.
- Ownership: O0/O1/O2, 100 admission + 50 confidence updates.
- Every accepted trajectory has the intended update count, zero non-finite
  steps, zero AMP skips and a passed frozen-ownership postflight.
- O2 uses AMP initial scale 4096 for all three seeds. O0 seed73 uses the sealed
  expandable CUDA allocator; no microbatch or exposure-count exception was
  introduced.
- Failed attempts remain under
  `outputs/u2v5_cvpr_ablation_20260817/failed_attempts/` and are not silently
  replaced: A4 configuration, C1 objective contract, D1 scope, D2m binding,
  O0 seed73 OOM and O2 seed42 AMP-skip attempts.

## Ownership audit

The shared-score O0 gradients conflict consistently:

| Seed | gradient cosine | negative-cosine fraction | sign-conflict fraction |
|---:|---:|---:|---:|
| 17 | -0.1900 | 0.6554 | 0.5339 |
| 42 | -0.2523 | 0.7027 | 0.5563 |
| 73 | -0.2810 | 0.7838 | 0.5567 |

O1 reduces but does not eliminate conflict. O2 completed 150 structural checks
per seed with zero cross-gradient connections.

## Mechanism surfaces

Admission val3 macro Acc@0.5 (mean ± sample SD):

| Row | Result |
|---|---:|
| A1 surface only | 0.730670 ± 0.002069 |
| A3 no preserve | 0.739263 ± 0.001344 |
| A4 no category-complete | 0.738000 ± 0.000421 |
| A5 full | 0.738851 ± 0.001006 |

A2 changes only the non-deployed auxiliary tensors. Its 1,157 deployed tensors
are bitwise equal to the initializer for all three seeds; receipt SHA256 is
`f2b916c2e371a5b10ad754b39bdfd6e0789f59540336bc7a828dc9df384e178f`.

Calibration FPR95 (mean over three seeds):

| Row | FPR95 |
|---|---:|
| C1 current batch | 0.588110 |
| C2 queue512, no positive trust | 0.536093 |
| C3 clean anchor | 0.536518 |
| C4 paired margin | 0.538641 |
| O0 shared score | 0.573248 |
| O1 shared trainable feature | 0.572187 |
| O2 isolated, interleaved | 0.533758 |
| O3 isolated, phased | 0.536518 |

Matched calibration strongly favors D3m over D2m (0.344589 vs 0.443723), but
this is a selection-surface observation, not a strict held-out claim.

## Preregistration

- Receipt:
  `outputs/u2v5_cvpr_ablation_20260817/preregistration.json`
- SHA256:
  `6eaa844c87b4a116dfaf13dc2c9fa1cbced030f2b9438197a8579f5c410916a4`
- Git commit: `d8afbcf2f0b4dc7b07212efe1cb8981aabac4a37`
- Bound trajectories: 42.
- Bound contrasts: A5−A1, C3−C2, O2−O0, O3−O2 and D3m−D2m.
- Test5 and strict2031 outputs did not exist when this receipt was written.
- strict1607 is registered as an exact identity subset of strict2031 and is
  never forwarded separately in this block.

The B56 admission block retains the disclosed seed42 minimum-free-memory value
of approximately 0.44 GiB. It does not claim the deprecated 1 GiB margin.
