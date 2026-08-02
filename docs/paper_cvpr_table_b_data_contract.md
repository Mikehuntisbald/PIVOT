# CVPR Table-B TN Data Contract

This contract instantiates D0-D3 and the secondary D2m/D3m causal panel from
`docs/paper_cvpr_ablation_protocol.md`. The machine-readable authority is
`data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json`; the
formal matched-panel authority is the class-aligned v2 audit
`data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/audit.json`.

## Sealed rows

| ID | Runtime-facing scope | Train rows | Dataset config |
| --- | --- | ---: | --- |
| D0 | no TN | 0 | `config/datasets_stageb_table_b_d0_no_tn.json` |
| D1 | unverified all-negative | 14,196 | `config/datasets_stageb_table_b_d1_unverified_allneg.json` |
| D2 | traceable counterfactual edit only | 14,196 | `config/datasets_stageb_table_b_d2_traceable_edits.json` |
| D3 | target plus cached proposals, proposal-covered | 14,196 | `config/datasets_stageb_table_b_d3_proposal_covered.json` |

Every TN source is image-disjoint from the union of strict2031 and strict1607.
D1 and D2 use the declared seed `20260717-table-b` for a SHA-256 image-level
90/10 train/calibration split. D3 reuses and verifies the upstream sealed
single-edit split. All three train files contain exactly 14,196 rows.

The three positive sources are identical in all four dataset configs, each
with `mix_weight=1.0`. D1-D3 add one TN source with `mix_weight=3.0` and
`tn_balance_sampling=false`, yielding the same 50% expected TN sampler mass.
Training comparisons must still use the same fixed optimizer-update count.
D1 and D2 are additionally matched to identical color/size/spatial edit counts;
D2 retains its native dataset proportions inside each edit category. D3 has a
substantially broader semantic taxonomy, which cannot be reproduced by D1/D2
without fabricating labels, so that limitation is explicitly reported rather
than hidden.

## Parent-matched causal panel

The broad D2/D3 comparison changes both verification scope and source
population. D2m/D3m therefore retain only parent expressions present in both
sources under this exact key:

```text
normalized dataset + image_id + sent_id + normalized edit category
+ normalized positive expression
```

The negative expression is deliberately not a key, because visual filtering
may select a different edit. Each D3 parent selects at most one D2 row using a
seeded SHA-256 priority, and every decision records both source row hashes and
line numbers. The train panel has 7,074 unique pair IDs and unique parent keys,
4,431 images, and no overlap with strict2031, strict1607, or its calibration
split. Calibration has 770 pairs from 496 images.

| Split | Pairs | TN text identical | TN text different | Class-aligned identical complete input |
| --- | ---: | ---: | ---: | ---: |
| train | 7,074 | 3,242 | 3,832 | 3,203 |
| calibration | 770 | 378 | 392 | 375 |

"Identical" here is exact string identity; it is also identical after
lowercase/trim/whitespace normalization. All 7,074 train pairs have exactly
identical positive expressions. Exact TN-text identity alone is not a complete
model-input match: 39 train pairs and 3 calibration pairs retain different
canonical class IDs. Only the 3,203 train and 375 calibration pairs marked
`class_aligned_identical_complete_input` hold every model-consumed identity
component fixed. Those are the primary causal strata. The remaining rows are
matched-parent diagnostics: different-TN rows remain confounded by edit
realization, and every class-ID mismatch is excluded from the clean causal
denominator.

The formal paired data files and pair ledger live under
`data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/`. The
generated manifests are
`config/datasets_stageb_table_b_d2m_matched_class_aligned_v2_traceable.json`
and
`config/datasets_stageb_table_b_d3m_matched_class_aligned_v2_proposal_covered.json`.
Both use the same three positive sources, TN `mix_weight=3.0`, and 50% expected
TN draw fraction.

D1 and D2 are converted to the fixed scorer's explicit `[positive, TN]` pair
surface. The conversion retains source row SHA-256, line number, edit fields,
and source rule where available. It does not grant visual verification. D3's
legacy source calls its labels global, but its own audit says cached-proposal
coverage only, not all 900 queries. The Table-B view therefore retains the
legacy flag only inside `source_provenance` and exposes
`global_tn_verified=false`, `tn_scope=proposal_covered_verified` at runtime.

## Runtime gate

The sealed data audit records the base v19 behavior: without an explicit
Table-B contract, a paired TN with `global_tn_verified != True` raises:

```text
Stage B score-confidence training received paired TN rows without global_tn_verified=True
```

Setting that flag to true is not an acceptable workaround. The implemented
ablation-only switch is
`stage_b_v19_allow_scope_labeled_tn_ablation`, default false. D1-D3 leaf
configs bind the exact Table-B ID, audit SHA-256, singleton scope allowlist,
and audited train-file hash. Dataset construction verifies those bindings;
the engine then validates every paired row and creates the distinct
`confidence_ablation_eligible` mask. The criterion uses that mask for
confidence/global-max negative objectives without modifying or aliasing
`global_tn_verified`. Unknown scope, mixed IDs, global-flag upgrades, or hash
drift fail closed.

The runtime has two non-interchangeable audit schemas. D1-D3 bind the broad
14,196-row equal-exposure audit. Formal D2m/D3m bind the separate 7,074-row
class-aligned v2 parent-matched audit, SHA-256
`5ff62a838a5123d580a72e353147b97bb69e9d7967348b55cba4ccb9ca36cb96`,
and require its `D2m_D3m_supported_by_current_v24=true` runtime seal. The
matched manifests therefore set `paper_runtime_supported=true`, and v24 model
leaves make them runnable without relabelling them to D2/D3. Cross-binding a
matched row to the broad or legacy matched audit, changing its ID, or upgrading
its global flag fails closed.

The broad-block model configs are:

- `config/ablations/cfg_stageb_v23_table_b_d0_no_tn.py`
- `config/ablations/cfg_stageb_v23_table_b_d1_unverified_allneg.py`
- `config/ablations/cfg_stageb_v23_table_b_d2_traceable_edits.py`
- `config/ablations/cfg_stageb_v23_table_b_d3_proposal_covered.py`

The matched-block model configs are:

- `config/ablations/cfg_stageb_v24_table_b_d2m_matched.py`
- `config/ablations/cfg_stageb_v24_table_b_d3m_matched.py`

All inherit the selected v19 base-plus-gate, Acc50 hard-negative, and L4
objective. D1-D3 uniformly keep `require_single_edit_token_provenance=false`,
and D2m/D3m enforce the separate
`disabled_uniformly_D2m_D3m` token-provenance contract. Table B therefore fixes
positive-token supervision and predicate-pair rank but does not grant any TN
source edit-token BCE labels. Certified edit-token supervision is isolated in
Table C.

## Rebuild and verify

```bash
python tools/build_stageb_tn_data_ablation_matrix.py
python tools/build_stageb_tn_data_ablation_matrix.py --verify
python -m unittest tests.test_stageb_tn_data_ablation_matrix

python tools/build_stageb_tn_matched_causal_panel.py
python tools/build_stageb_tn_matched_causal_panel.py --verify
python -m unittest tests.test_stageb_tn_matched_causal_panel
python -m unittest tests.test_stage_b_v23_table_b_scope_contract
```
