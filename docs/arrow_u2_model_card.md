# ARROW-U2 model card

## Identity

**ARROW-U2** is the complete sealed model reported in the ARROW paper. It is
the visual-support `ARROW-V` interface with the isolated D3 abstention owner.

The implementation was developed and sealed under the internal `U2-v5`
lineage. That identifier remains in configs, checkpoint payloads, receipts, and
filenames for byte compatibility; it is not a second model name.

| Layer | Public name | Immutable implementation identifier |
| --- | --- | --- |
| Project/method | ARROW | release name |
| Complete paper model | ARROW-U2 | `U2-v5 A / A5+C3` |
| Visual interface | ARROW-V | A5 support-patch route |
| Text control | ARROW-T | canonical-text overlay |
| Null control | ARROW-N | learned-null overlay |
| Repository lineage | PIVOT | `pivot.*`, `stage_b_u2v5_*` |

## Architecture

```text
image I + full expression e
        │
        ▼
Gφ: frozen B58 Grounding DINO trunk
        └── 900 query features and boxes
                │
visual support u ──► AθA: Admission ──► eligible set E
                │
full expression e ─► R100: Ranking within E ─► selected query i*
                │
detached B58 statistics ─► D3: absolute abstention ─► emit box / abstain
```

- **Candidate owner:** frozen B58 full-expression trunk. Canonical text or
  `object` never replaces the expression in the geometry path.
- **Admission owner:** category-conditioned surface trained with the
  supervision-only auxiliary residual. Deployment uses the Gap3 eligible mask;
  the auxiliary residual never affects the deployed decision.
- **Ranking owner:** frozen positive-only R100 complete-expression ranker.
  Eligible-query scores remain unchanged; ineligible queries are
  lexicographically demoted.
- **Abstention owner:** independently trained D3 confidence12 over detached
  frozen statistics. It decides emit/abstain and cannot reorder queries.

The three deployed duties do not share trainable score parameters. Admission
and abstention are trained in separate phases; candidate geometry and R100 are
frozen throughout.

## Interfaces

`ARROW-U2` names the complete model, not every input control:

- `ARROW-V` is its primary visual-support interface.
- `ARROW-T` substitutes an explicit canonical phrase only at the Admission
  input. It is a controlled text-cue variant.
- `ARROW-N` substitutes a category-agnostic learned null cue and is a mechanism
  ablation.

All interfaces keep the complete expression in the frozen candidate generator
and Ranking path.

## Sealed implementation

- Seeds: 17, 42, 73.
- Internal composition: admission A5 at U100 followed by D3 confidence at U50.
- Admission gate: Gap3, selected without held-out Test5 access.
- Confidence data: image-disjoint D3 training/calibration; no C100 confidence
  tensor is imported.
- Release manifest: `paper/data/arrow_release_manifest.json`.

The manifest and receipts deliberately retain `U2-v5`, `PIVOT`, and `pivot.*`
identifiers. Renaming those serialized fields would invalidate hashes and break
historical checkpoint loading.

## Evidence scope

The primary localization endpoint is Test5 micro Acc@0.5. Ref8 additionally
contains the three validation splits used by the development protocol and is
descriptive. The primary rejection endpoint is Strict-TN2031 FPR95;
Strict-TN1607 is its identity-matched subset, not independent evidence.

FineCops-Ref and gRefCOCO are cross-benchmark task-transfer evaluations. They
are not image-disjoint from every stage of the detector lineage, and the paper
reports those overlap boundaries explicitly.

## Historical boundary

The following are not public model names:

- Stage-A / Stage-B patch exploration;
- V55, V56, and other dense-duty revisions;
- routed-v3 multi-checkpoint evaluation;
- U2-v2, U2-v3, and U2-v4 diagnostics.

They are indexed under [`docs/historical/`](historical/README.md) and retained
only for provenance, negative-result tracing, and ABI compatibility.
