# ARROW FineCops-Ref external zero-shot evaluation

> FineCops-specific benchmark zero-shot; this surface is not image-disjoint from historical model training.

| Route | Positive P@1 macro | Positive P@1 micro | Neg-text Recall@1 | Neg-image Recall@1 |
| --- | ---: | ---: | ---: | ---: |
| B58 | 43.11 ± 0.00 | 53.14 ± 0.00 | 42.66 ± 0.00 | 43.98 ± 0.00 |
| R100 + D3 confidence | 43.02 ± 0.00 | 53.09 ± 0.00 | 43.85 ± 0.16 | 44.76 ± 0.17 |
| ARROW-A support patch (95.60% covered) | 44.16 ± 0.02 | 54.28 ± 0.03 | 45.05 ± 0.11 | 46.14 ± 0.15 |
| ARROW-B category text | 43.15 ± 0.22 | 53.06 ± 0.30 | 43.68 ± 0.24 | 44.51 ± 0.15 |
| ARROW-C learned null | 41.50 ± 0.86 | 51.12 ± 1.24 | 42.16 ± 0.90 | 42.94 ± 1.24 |

## Pre-registered admission contrasts

| Contrast | Gain (pp) | 95% image-cluster CI (pp) | Holm p |
| --- | ---: | ---: | ---: |
| image A minus B | 0.71 | [0.39, 1.03] | 0.0003999 |
| image B minus C | 1.55 | [1.14, 1.98] | 0.0003999 |
| positive A minus B | 0.46 | [0.12, 0.80] | 0.003599 |
| positive B minus C | 1.59 | [1.17, 2.05] | 0.0003999 |
| text A minus B | 0.55 | [0.29, 0.82] | 0.0003999 |
| text B minus C | 1.53 | [1.14, 1.95] | 0.0003999 |

A metrics exclude unsupported rows and never count unsupported negatives as correct rejection. B/C and both baselines use the complete 27,926-record test.
