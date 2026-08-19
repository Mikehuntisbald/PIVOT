# Original GroundingDINO-T OGC FineCops replay

This is a post-hoc corrective, input-matched baseline replay. It replaces the
published MM-GDINO-T row in the ARROW FineCops table; it is not part of the
original preregistered ARROW contrast family.

## Bound model and protocol

- Checkpoint: `/media/haoyi/T9/pivot/weights/groundingdino_swint_ogc.pth`
- SHA-256: `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`
- Architecture: upstream GroundingDINO Swin-T OGC, 940 checkpoint tensors.
- Runtime parity: 938 model tensors load bitwise; only the upstream legacy
  `bert.embeddings.position_ids` and `label_enc.weight` payload entries are not
  runtime parameters.
- FineCops surface: all 27,926 official test rows, no training, checkpoint
  selection, taxonomy change, or FineCops threshold fitting.
- Preregistration:
  `outputs/arrow_original_gdino_ogc_finecops_20260819/preregistration.json`.
- Per-example records: 27,926 rows, SHA-256
  `e4c335e659629f523e7de476d0dd52b99eb42a1ac19f0af37182d74bd4b59668`.
- Pinned official evaluator commit:
  `31d2c8615e65ccef6a4ff516925ef5ae465ec747`.

The run locked two score reductions before looking at results. Both use the
same boxes and full referring expression. `expression_max` is the upstream
native maximum expression-token probability and is the public main-table row.
`expression_mean` averages probabilities across the model-generated
full-expression phrase mask and is retained as a scorer sensitivity.

## Results

| Score route | Positive P@1 macro | Positive P@1 micro | Text R@1 micro | Image R@1 micro | Text AUROC macro | Image AUROC macro |
|---|---:|---:|---:|---:|---:|---:|
| Native max | 44.82 | 55.19 | 41.99 | 41.98 | 54.23 | 53.92 |
| Expression mean | 40.34 | 51.36 | 29.96 | 39.85 | 48.43 | 53.56 |

P@1 macro is the mean across FineCops positive L1/L2/L3. R@1 is the audited
strict-tie micro metric. AUROC uses the pinned official evaluator's historical
L1-positive scope and is macro-averaged over its reported type-by-level rows.
These scopes are intentionally kept separate.

## Interpretation boundary

The original OGC replay is not the existing paper `Base` row. `Base` is the
Stage-B data-finetuned B58 checkpoint: among their 938 same-named runtime
tensors, only 211 are bitwise equal and 727 differ. Therefore the paper keeps
`Original GDINO-T` and `Base` as distinct baselines and does not rename B58.

The large max-versus-mean spread is disclosed rather than choosing a scorer
silently after evaluation. No confidence threshold is invented for OGC; only
ranking, Recall@1, and domain-derived rejection diagnostics are reported.
