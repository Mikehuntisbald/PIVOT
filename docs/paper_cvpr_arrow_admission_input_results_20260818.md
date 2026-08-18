# ARROW Admission-input results (2026-08-18)

## Conclusion

The three-way Admission input block supports both preregistered mechanism
claims while retaining the sealed support-patch model as ARROW's main route:

1. A support patch provides category-switch information beyond an explicit
   canonical category name.
2. Explicit category text provides controllable category conditioning that a
   learned category-agnostic null route cannot provide.
3. Final Ref accuracy is less sensitive than category controllability: B and C
   both satisfy the preregistered Test5 `−0.005` no-collapse margin.

## Capacity-matched routes

| Version | Admission input | val3 micro Acc@0.5, mean ± SD | Test5 micro mean | Test5 delta vs A |
|---|---|---:|---:|---:|
| A | support patch | 0.723611 ± 0.001152 | 0.742398 | — |
| B | canonical category text | 0.722780 ± 0.002209 | 0.741193 | −0.001206 |
| C | learned null/category-agnostic token | 0.718137 ± 0.009177 | 0.738308 | −0.004090 |

All routes use the same 268,167 trainable parameters: legacy surface8 plus the
training-only auxiliary8. Full referring expressions continue to own frozen
B58 geometry and R100 ranking. The C route shows higher seed variability,
especially seed42, and should not be described as equivalent to A despite
passing the aggregate no-collapse margin.

Test5 paired image-cluster results:

| Contrast | Gain | 95% CI | NI p at margin 0.005 |
|---|---:|---:|---:|
| B−A | −0.001206 | [−0.002133, −0.000310] | 0.000200 |
| C−A | −0.004090 | [−0.004980, −0.003210] | 0.023995 |

Test5 is a prospectively frozen post-release ablation, not a new globally
virgin benchmark. A records were reused without another forward.

## Fresh image-disjoint category-switch panel

The primary endpoint contains 512 LVIS image pairs/1,024 active-category arms.
It is image-disjoint from 321,327 admission-training rows, Ref8 and strict2031,
and had no model outputs before preregistration. Missing either category's
IoU≥0.5 oracle queries is counted as failure.

Pair-level bidirectional switch success:

| Route | seed17 | seed42 | seed73 | Mean |
|---|---:|---:|---:|---:|
| A support patch | 0.498047 | 0.472656 | 0.476562 | 0.482422 |
| B canonical text | 0.326172 | 0.324219 | 0.318359 | 0.322917 |
| C learned null | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Preregistered paired pair-cluster bootstrap:

| Contrast | Gain | 95% CI | raw p | Holm p |
|---|---:|---:|---:|---:|
| A−B: visual over text | +0.159505 | [0.108073, 0.208333] | 0.000200 | 0.000400 |
| B−C: category over null | +0.322917 | [0.286458, 0.360026] | 0.000200 | 0.000400 |

The correct paper claim is therefore not that support pixels are required for
every correct Ref top-1. Rather, generic learned admission can preserve much of
the final accuracy, canonical category text supplies explicit controllability,
and visual support supplies additional category-switch evidence beyond the
name alone.

## Ownership and rejection parity

- Six new U100/B56 trajectories completed at AMP scale8192 with 0 skipped
  steps and 0 non-finite boundaries.
- B/C each trained exactly surface8+auxiliary8; trunk, patch backbone, R100,
  confidence12 and patch temperature stayed frozen.
- On all 1,570 D3 calibration records for every seed, B and C confidence scores
  are bitwise equal to the sealed clean A/D3 confidence records.
- No strict1607/2031 forward was run. Existing FPR claims are reused only via
  the verified confidence parity.
- The sealed A main model is unchanged.

## Failure and amendment lineage

The first C seed17 attempt produced one AMP overflow at the original fp16
provider surface. Before any model evaluation, all initial B/C trajectories
were superseded. The global AMP scale remained 8192; the shared B/C provider
projection/query cosine was evaluated in fp32, and all six rows were retrained.

The first evaluation queue completed B val3 and then stopped before fresh panel
or Test5 because confidence-only evaluation incorrectly entered the Admission
provider. Evaluation lock v2 makes confidence-only an explicit Admission
bypass and preserves clean-confidence provenance. Training checkpoint hashes
did not change.

## Sealed artifacts

- ARROW release manifest SHA256:
  `ebe587bee63ed4288f464d1a4872184735a2d0ba0b3ce8cba121973cf1bc49a7`
- Amended preregistration SHA256:
  `1cc68fbb95ef524be66bf41cbe569fe3d611f21aeb48eeb45df8ff98d82144d8`
- Evaluation checkpoint lock v2 SHA256:
  `44c60285aa9ed462f907e770a44feacc2117cf1e40a48867c2955ba9f527752e`
- Final receipt SHA256:
  `427d1daf9288ab9cc9c343b3e0a0e352826f578a96adb1922e8869e2edd39458`

Weights and per-example records remain under
`outputs/arrow_admission_input_20260818/` and are not committed to Git.
