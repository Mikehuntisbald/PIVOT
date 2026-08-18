# ARROW FineCops-Ref external zero-shot evaluation

> FineCops-specific benchmark zero-shot; this surface is not image-disjoint from historical model training.

| Route | Positive P@1 macro | Positive P@1 micro | Neg-text Recall@1 | Neg-image Recall@1 |
| --- | ---: | ---: | ---: | ---: |
| B58 | 43.35 ± 0.00 | 52.91 ± 0.00 | 42.12 ± 0.00 | 43.29 ± 0.00 |
| R100 + D3 confidence | 43.23 ± 0.00 | 52.74 ± 0.00 | 43.13 ± 0.20 | 44.71 ± 0.16 |
| ARROW-A support patch (95.60% covered) | 44.48 ± 0.08 | 53.84 ± 0.03 | 44.17 ± 0.16 | 46.00 ± 0.18 |
| ARROW-B category text | 43.31 ± 0.26 | 52.58 ± 0.29 | 42.89 ± 0.31 | 44.30 ± 0.43 |
| ARROW-C learned null | 41.67 ± 0.98 | 50.63 ± 1.29 | 41.27 ± 0.92 | 42.60 ± 1.48 |

## Pre-registered admission contrasts

| Contrast | Gain (pp) | 95% image-cluster CI (pp) | Holm p |
| --- | ---: | ---: | ---: |
| image A minus B | 0.77 | [0.45, 1.11] | 0.0003999 |
| image B minus C | 1.73 | [1.32, 2.15] | 0.0003999 |
| positive A minus B | 0.56 | [0.20, 0.93] | 0.002 |
| positive B minus C | 1.61 | [1.18, 2.07] | 0.0003999 |
| text A minus B | 0.57 | [0.30, 0.86] | 0.0003999 |
| text B minus C | 1.63 | [1.24, 2.04] | 0.0003999 |

A metrics exclude unsupported rows and never count unsupported negatives as correct rejection. B/C and both baselines use the complete 27,926-record test.
