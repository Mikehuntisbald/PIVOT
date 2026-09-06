# v6 record-only reproduction

This guide reproduces the **analysis of fixed predictions**, not detector fine-tuning or head training. It needs numeric per-example records, training-score statistics and their bound protocol; **no images, model weights, detector packages, optimizer or GPU** are needed. The tool is a reusable research deliverable, not a claim of a new risk metric or an additional model result.

For the experiment and its scope, see [the v6 study ledger](confidence_readout_v6_20260906.md) and [the completed numerical results](readout_v6_final_results_20260906.md). The [first-localizer report](readout_v6_mm_stage_results_20260906.md) remains a historical staged analysis, not a substitute for the completed two-localizer study.

## 1. CPU environment and files

Run commands from the repository root. The reference record-analysis environment uses Python **3.12.11** and NumPy **1.26.4**. A separate environment avoids modifying a detector environment:

```bash
python3.12 -m venv .venv-readout-cpu
.venv-readout-cpu/bin/python -m pip install numpy==1.26.4
```

The analysis depends only on NumPy and the standard library, plus these repository modules:

- `tools/confidence_readout_metrics.py` and `tools/analyze_confidence_readout.py`;
- `tools/grounding_confidence_ordering.py`, `tools/grounding_emission_audit.py`, `tools/grounding_generalized_risk_audit.py`, `tools/grounding_prevalence_audit.py`.

These six files are hash-locked. Do not upgrade or edit them to make an old receipt pass. Record the interpreter and NumPy version when reproducing elsewhere; cross-version last-bit equality is not implied by using the same formulas.

A portable numeric artifact directory can have this layout:

```text
readout-records/
  protocol.json                         # unchanged sealed bytes
  analysis_code_lock.json               # unchanged sealed bytes
  inputs/
    finecops_val.portable.json           # new path-binding wrapper
    gref_full.portable.json
    gref_disjoint.portable.json
  records/
    mmgdino_positive/seed17_val.jsonl    # and seeds 42/73, other surfaces
    mdetr_r101_refcoco_ema/...
  statistics/
    mmgdino_positive/seed17.json         # and seeds 42/73
    mdetr_r101_refcoco_ema/...
```

The exact files and SHA-256 values come from the sealed analysis input, not from the illustrative filenames above. The repository's ignored `outputs/` directory is not an automatic public download: obtain the corresponding numeric artifact pack before attempting formal reproduction. Images being absent does not by itself certify a pack's anonymity or redistribution permissions.

## 2. Record interface: state and score are separate

`analyze_readout()` takes `runs[localizer][seed] = rows`. For each row:

| Field | Meaning / requirement |
|---|---|
| `sample_id` | Unique expression/request identity, identical across compared runs. |
| `cluster_id` | Source-image cluster, shared by every expression from that image. |
| `stratum` | `validation` for the formal FineCops panel; `testA`/`testB` for formal gRef. |
| `kind`, `correct` | C: `positive,true`; W: `positive,false`; N: `text,null` or `no_target,null`. Never invent a no-target correctness label. |
| `level` | Official positive difficulty, or null. Negative edit difficulty is not a substitute. |
| `parent_positive_id` | Exact same-image positive sample ID for a genuine edited-text pair; otherwise null. No synthetic parent links for gRef. |
| `native_score` | Fixed Native request score in [0,1]. This field supplies the `native` reference slot automatically. |
| `scores` | Four matched head scores below; formal records also contain `joint_product` and `joint_sirc`. |
| `readout_diagnostics` | Four trained-head diagnostic objects, required for the complete formal analysis. |

The matched score keys are:

```text
global_max__exists
global_max__emit
native_selected__exists
native_selected__emit
```

All rows must have the same score slots. Additional score slots can be analyzed through the import API but do not replace these four matched cells. Scores are finite raw head logits/ranking values, not claimed calibrated probabilities. Native boxes remain fixed; `correct` is localizer-specific, and must be identical across that localizer's three seeds, but may differ between localizers.

Each `readout_diagnostics[matched_key]` contains:

```json
{
  "max_logit": 0.7,
  "selected_logit": 0.5,
  "confidence_winner_index": 9,
  "native_selected_index": 2,
  "winner_native_box_iou": 0.6,
  "native_gt_iou": 0.8,
  "winner_gt_iou": 0.7
}
```

These numbers illustrate shape only. The matched score must exactly equal `max_logit` for G-trained/G-deployed cells or `selected_logit` for S-trained/S-deployed cells. All four heads retain the same Native index and Native GT IoU; for N, both GT IoUs are null. Index and IoU are different notions of agreement. In particular, the dense global winner of an S-trained head is a **counterfactual diagnostic**, not its actual selected-query deployment.

The analyzer derives four off-diagonal scores from these objects. Supply diagnostics for every row or omit the entire diagnostic panel in a non-formal API example; incomplete diagnostic populations fail closed. Full-query features, full boxes and images are not needed for this numerical step: upstream parity receipts establish their provenance.

## 3. Minimal import example on synthetic data

The import API permits arbitrary localizer names, a declared set of fixed seeds, smaller populations and fewer bootstrap replicates. This is useful for unit tests or applying the protocol to another fixed model; it does **not** produce a formal v6 completion receipt.

```python
from tools.confidence_readout_metrics import CELLS, analyze_readout

rows = []
for image in range(2):
    for state, kind, correct, native, existence, emission in (
        ("C", "positive", True, 0.9, 0.6, 0.9),
        ("W", "positive", False, 0.4, 0.7, 0.1),
        ("N", "no_target", None, 0.3, 0.2, 0.2),
    ):
        rows.append({
            "sample_id": f"image{image}:{state}",
            "cluster_id": f"image{image}",
            "stratum": "synthetic",
            "kind": kind,
            "correct": correct,
            "level": 1 if kind == "positive" else None,
            "parent_positive_id": None,
            "native_score": native,
            "scores": dict(zip(CELLS, (
                existence, emission, existence + 0.01, emission + 0.01,
            ))),
        })

result = analyze_readout(
    {"toy_grounder": {"toy_seed": rows}},
    iterations=16,
    seed=20260911,
    required_seeds=("toy_seed",),
    conditionals=False,
)
assert result["primary_metric"] == "mixed_augrc"
assert result["bootstrap"]["unit"] == "image_cluster"
```

This example intentionally has no trained model, formal population, geometry diagnostics or combination scores. It must never enter a paper result table.

## 4. Fixed combinations and training-statistics provenance

Each formal localizer/seed has a separate JSON returned by `fit_sirc_statistics()`. Its input consists of **83,341 unique TRAIN-positive score rows**, explicitly marked `split="train"`, `kind="positive"`, with `scores["global_max__exists"]`. Duplicated pair-expanded positives and validation rows are rejected. The statistics use float64 population standard deviation (`ddof=0`) and bind sorted sample IDs plus scores by SHA-256.

With Native score s0, G-exists logit z, training mean μ and standard deviation σ:

```text
product = log(max(s0, 1e-6)) - softplus(-z)
a = μ - 3σ
SIRC-style = -log(max(1-s0, 1e-6)) - softplus(-(z-a)/σ)
```

At σ≤1e−12, SIRC-style falls back to the Native score and records the degeneracy. There is no validation weight search or confidence-threshold fitting. The formal CLI consumes the already sealed statistics and **recomputes both combination values to verify record parity**, not to fit them again. Use `combine_scores()` or `add_combination_scores()` for a new, explicitly separate analysis; the latter returns copied rows and refuses to overwrite existing combination fields.

Do not confuse the statistics population with the head-loss population. The confidence optimizer used 80,451 pairs drawn from **43,979 unique L1 parents**, not all 83,341 unique positives; no L2/L3 positive targets entered head loss. Full positive caching, label audits, and the later unsupervised SIRC-statistics pass do include those rows. The MM positive-source trunk's earlier training is broader. Exact counts and the resulting scope restriction are in [the metadata follow-up](readout_v6_mm_stage_results_20260906.md#8-后续-metadata-核查confidence-的直接监督只覆盖-l1-parent-正例).

## 5. Portable rebasing without rewriting sealed provenance

The formal CLI follows these paths from its **analysis-input JSON**:

- `protocol.path`;
- optional `analysis_code_lock.path`;
- `runs[localizer][seed].path`;
- `sirc_statistics[localizer][seed].path`.

They may be absolute or relative to the input JSON's parent directory. To move an artifact pack:

1. Copy the referenced protocol, code lock, record and statistics files **byte for byte**; verify every SHA against the original input before using the copies.
2. Create a **new** analysis-input JSON, retaining the original input separately. Change only those four classes of file-binding paths to the copied artifacts. Keep all their SHA fields, `protocol_sha256`, population, surface and source metadata unchanged.
3. Do not rewrite the protocol's old internal absolute paths or the code lock's paths. The record-only CLI reads the copied protocol bytes, and checks the code lock's six code hashes against the current checkout; it does not need to open old checkpoint/image paths inside the protocol.
4. Run the new input into a fresh output filename. Its `receipt.input_sha256` correctly differs from the original input because the path-binding wrapper changed. Record/statistics/code SHAs and numerical results should match; receipt timestamps and input SHA are not numerical-reproduction failures.

For example, a copied input in `readout-records/inputs/` can bind:

```json
{
  "protocol": {"path": "../protocol.json", "sha256": "UNCHANGED_ORIGINAL_SHA256"},
  "analysis_code_lock": {"path": "../analysis_code_lock.json", "sha256": "UNCHANGED_ORIGINAL_SHA256"}
}
```

This is a **path-edit fragment**, not a complete runnable manifest; do not replace its SHA placeholders with new hashes of modified sealed files. The real manifest must retain schema `arrow.confidence_readout_analysis_input/v1`, the surface and expected population, both complete localizer mappings, and all six statistics/record bindings.

`source_postflights` and historical absolute paths remain provenance, not a request to resolve private image/weight paths during this step. Do not use `prepare_confidence_readout_analysis.py inputs` as the portable entry point: that producer deliberately follows the full in-repository experiment lineage and is for the original experiment workspace. Portable replay starts from its already-created, sealed input and a new path-only wrapper.

Retain the original scientific protocol and raw artifact bytes. Sanitizing an artifact for a separate public release is a different, reviewed export operation; it is not silently equivalent to rebasing paths.

## 6. Formal commands and acceptance boundaries

Unlike the flexible import API, the formal CLI requires the exact localizers `mmgdino_positive` and `mdetr_r101_refcoco_ema`, seeds `17/42/73`, all four matched and four diagnostic readouts, both fixed combinations, and these complete populations:

| Surface name | Images | Positive | No target |
|---|---:|---:|---:|
| `finecops_val` | 3,567 | 9,426 | 9,029 text negatives |
| `gref_full` | 1,500 | 11,563 | 9,121 |
| `gref_finecops_train_val_source_disjoint` | 1,277 | 9,848 | 7,716 |

Example, after copying and verifying the artifacts:

```bash
.venv-readout-cpu/bin/python -m tools.analyze_confidence_readout \
  --input readout-records/inputs/finecops_val.portable.json \
  --output readout-records/reproduced/finecops_val.json
```

Repeat for the other two input files with distinct output filenames. Defaults are 5,000 replicates and seed20260911. Outputs are append-only: never delete or overwrite a sealed analysis merely to rerun it. Use a new destination and preserve a failed/incomplete file for audit.

`--stage-mm-only` admits the first localizer alone and explicitly cannot complete the two-model study. `--no-conditionals`, fewer iterations or another bootstrap seed are diagnostic executions, not paper-ready formal outputs. Do not rename a stage output into the full-result filename and infer completeness from its existence.

The output contains:

- `localizers[loc].per_seed` and `localizers[loc].summary[arm][metric]`;
- `localizers[loc].effects.D_emit`, `.D_exists`, `.interaction`, plus target and combination contrasts;
- `localizers[loc].augrc_crossovers`, `.conditional_counts`, and descriptive `.winner_geometry`;
- input/record/statistics/code bindings in `receipt`.

The metrics are raw fractions: AUROC lies in [0,1], AUGRC in [0,0.5]. Paper assets multiply values by100. `formal_requested_configuration=true` means the requested numerical configuration was supplied; `study_final_receipt` remains false because this CLI alone does not certify all training, transfer, paper or submission requirements.

For a new CPU replay, `tools/analyze_confidence_readout_parallel.py` is an optional two-process scheduler around the **unchanged** numerical function:

```bash
.venv-readout-cpu/bin/python -m tools.analyze_confidence_readout_parallel \
  --input readout-records/inputs/finecops_val.portable.json \
  --output readout-records/reproduced/finecops_val.parallel.json \
  --workers 2
```

The full two-localizer loader validates sample identities before splitting work. Each worker processes all three seeds with the same complete image universe and PCG64 draw sequence; the merge requires matching draw SHA and exact equality of every non-localizer result field. This separation is valid because this study estimates only within-localizer contrasts. It does not implement paired cross-localizer contrasts or split the bootstrap into independently seeded batches. A synthetic 5,000-draw check verified exact equality of the joined result and the original joint call, including all conditionals; performance depends on the CPU and population.

The wrapper binds its own code and Python/NumPy versions in `receipt.parallel_execution` in addition to the original six numerical-module hashes. Optional `--reuse-mm-stage PATH --reuse-mm-stage-sha256 SHA` accepts only a **completed** FineCops MM stage whose protocol, record/statistics hashes, six core hashes, image universe and complete bootstrap match the new full input. Changed metadata bytes also invalidate such reuse even if model scores are unchanged. Partial draws cannot be resumed. Do not interrupt an existing formal run merely to adopt this scheduler; it is a separate replay destination, not permission to replace sealed receipts.

## 7. Resampling, interpretation and compatibility notes

Bootstrap units are **images, not expression-IID rows**. Within a surface, the same stratified image draw is applied to all localizers, all scores and all three fixed seeds. Each selected image carries all its expressions; metrics are computed per seed and then equally averaged. gRef uses TestA/TestB strata. Full and source-disjoint gRef are overlapping sensitivity surfaces, not independent replications.

Observed-mixture risk retains each draw's changing class proportions. The separate fixed-prior grid `0,.1,.25,.5,.75,.9,1` renormalizes positive/no-target masses within every draw. Every diagnostic FPR95 replicate recomputes its model's positive q05; no sealed deployment threshold is estimated by this analysis. Undefined replicates are counted, not silently discarded, and null the affected ordinary interval. Crossover reports separately count absent/non-interior roots; a conditional-on-interior interval is labelled as such.

The primary questions require both absolute emit change and interaction:

```text
D_emit   = R(S,emit)  - R(G,emit)
D_exists = R(S,exists)- R(G,exists)
I        = D_emit - D_exists
```

Negative I alone can arise by harming exists. A conditional interval crossing zero is not proof of absence or image/difficulty causation. C–W/C–N/W–N decomposition explains which rankings changed; it does not prove that a head explicitly represents difficulty or that spatial correspondence is the unique mechanism.

Two compatibility corrections are documented in [the adapter/metadata ledger](confidence_readout_v6_20260906.md#data-adapter-corrections-before-mdetr-head-training):

- The COCO-format negative annotation retains its parent's bbox as an **inactive reference**, not GT for the edited no-target expression. MDETR cache-v2 preserves that reference separately and supplies empty negative study-GT; it does not create a localization label for N.
- The old unused `negative_edit_level` export copied annotation `level`. The append-only record-v2 export restores official `negative_level` and retains `raw_annotation_level`. Scores, Native labels/geometry, parent identity/difficulty and winner diagnostics are preserved. The numerical analyzer consumes neither edit-level field.

For byte-identical reproduction of an original analysis, use the exact record files it binds, and retain record-v2 as a companion erratum. If using corrected record-v2 files in a new replay, create a new input with their actual SHAs, retain the erratum receipt/protected-field parity, and label it a metadata-corrected replay; do not substitute its bytes under an old hash. No v5 manuscript, old checkpoint, original record or experimental lineage should be deleted.

## 8. Build paper assets only after all three analyses exist

Asset generation does not run a model or recompute metrics. It needs Matplotlib in addition to NumPy; keep plotting dependencies separate from a hash-matched numerical environment if necessary. The repository's paper environment pins them in `paper/requirements.txt`.

```bash
python paper/scripts/build_readout_v6_assets.py \
  --finecops readout-records/reproduced/finecops_val.json \
  --gref-full readout-records/reproduced/gref_full.json \
  --gref-disjoint readout-records/reproduced/gref_disjoint.json \
  --output-dir paper/generated/readout_v6_reproduced
```

Run the same command with `--check` to verify already-generated asset/source hashes. An existing output directory is immutable. `--self-test` checks formatter/interpretation guards using synthetic values without producing assets. The generator refuses incomplete staged data and requires all three surfaces, both models and three seeds; it does not require any arm to win. Its output still requires manuscript review and is not a submission-readiness certificate.

For focused CPU tests, install pytest in the analysis environment and run:

```bash
.venv-readout-cpu/bin/python -m pytest -q \
  tests/test_confidence_readout_metrics.py \
  tests/test_grounding_generalized_risk_audit.py \
  tests/test_grounding_emission_audit.py \
  tests/test_grounding_prevalence_audit.py
```

This document supplies reproduction instructions only. Experiment-completion claims require the actual receipts linked in the results report; the guide itself adds no model-performance claims or submission-readiness certificate.
