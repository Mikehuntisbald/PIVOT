# ARROW × gRefCOCO rejection transfer

| Surface | Model | AUROC | AUPR | FPR95 | Fixed TPR | Fixed N-acc |
|---|---|---:|---:|---:|---:|---:|
| testA | B58 | 0.7449 | 0.7728 | 0.6362 | — | — |
| testA | D3 mean | 0.7763 | 0.7918 | 0.5917 | 0.9338 | 0.4446 |
| testB | B58 | 0.6229 | 0.6445 | 0.8155 | — | — |
| testB | D3 mean | 0.6450 | 0.6603 | 0.7988 | 0.8667 | 0.3389 |
| full | B58 | 0.6895 | 0.7201 | 0.7410 | — | — |
| full | D3 mean | 0.7175 | 0.7381 | 0.7083 | 0.9010 | 0.3905 |
| d3_disjoint | B58 | 0.6869 | 0.7192 | 0.7497 | — | — |
| d3_disjoint | D3 mean | 0.7150 | 0.7373 | 0.7116 | 0.8972 | 0.3893 |
| d3_finecops_disjoint | B58 | 0.6871 | 0.7192 | 0.7488 | — | — |
| d3_finecops_disjoint | D3 mean | 0.7151 | 0.7374 | 0.7112 | 0.8964 | 0.3908 |

Decision: `parameter-isolated confidence consistently improves cross-benchmark rejection ordering, while absolute operating-point calibration remains domain dependent`

This is annotation/task-zero-shot transfer on previously exposed COCO imagery, not image-disjoint zero-shot.
