# U2-v5 CVPR ablation execution runbook

> Execution status (2026-08-18): 42 historical trajectories were executed;
> 33 remain manuscript-visible after archiving D2/D2m/D3m. Mechanism
> evaluation, preregistered confirmatory block, bootstrap and final receipt are
> complete. See `paper_cvpr_u2v5_ablation_results_20260818.md`. The exploratory
> P/S counterfactual forwards and M/G/H analysis are sealed in
> `zero_training_supplement_v3.json`. The registry-external queue128/1024,
> candidate-count and support-bank-size expansions remain optional future work.

The manuscript-facing block contains 11 new rows and 33 trajectories:

```text
A1 A2 A3 A4
C1 C2 C4
D1
O0 O1 O2
times seeds 17, 42, 73
```

Historical rows D2/D2m/D3m were executed before their source-lineage mismatch
was identified. They remain under `outputs/` for audit but are excluded from
all paper tables and hypotheses.

Sealed controls A0/A5, C0/C3, D0/D3, and O3 are reused and are not retrained.

## CPU preflight

```bash
python tools/run_stageb_u2v5_ablation_matrix.py list
python tools/run_stageb_u2v5_ablation_matrix.py dry-run --run-id A1:17
python tools/run_stageb_u2v5_ablation_matrix.py dry-run --run-id O0:17
```

The launcher binds config, dataset, every source JSONL/audit, initializer or
parent checkpoint, runtime Python, and the U2-v5 code closure. Formal `run`
requires a clean worktree and a fresh output root.

## Formal training

```bash
DATA_ROOT=/media/haoyi/T9/data CUDA_VISIBLE_DEVICES=0 \
python tools/run_stageb_u2v5_ablation_matrix.py run --run-id A1:17

python tools/run_stageb_u2v5_ablation_matrix.py detach --run-id O2:73
python tools/run_stageb_u2v5_ablation_matrix.py status JOB_DIR
python tools/run_stageb_u2v5_ablation_matrix.py reconcile JOB_DIR
```

Admission rows run B56/U100. Confidence/data rows run B8/U50. Ownership rows
run an exact 100-admission/50-confidence interleaved schedule. O0 and O1 log
shared-parameter gradient cosine and sign-conflict fraction; O2 fails on any
structural cross-gradient.

## Mechanism evaluation

```bash
python tools/run_stageb_u2v5_ablation_evaluations.py dry-run --row-id A1
python tools/run_stageb_u2v5_ablation_evaluations.py run --row-id A1
```

A and O write val3 results. C/D/O write audit-bound calibration results. The
multi-route Ref evaluator writes complete B58/raw-R100/admission-R100 records
from one forward. Ref-only and confidence-only parity is mandatory.

## Preregistration and confirmatory evaluation

```bash
python tools/build_stageb_u2v5_ablation_preregistration.py \
  --output outputs/u2v5_cvpr_ablation_20260817/preregistration.json

python tools/run_stageb_u2v5_ablation_confirmatory.py \
  --preregistration outputs/u2v5_cvpr_ablation_20260817/preregistration.json
```

Only A5-A1, C3-C2, O2-O0, and O3-O2 are manuscript-facing confirmatory contrasts.
The historically executed D3m-D2m output is archived and excluded.
The anchor records are reused without a new forward. strict2031 is forwarded
once; strict1607 is derived from its sealed identity subset.

## Bootstrap and final receipt

```bash
python tools/aggregate_stageb_u2v5_bootstrap.py \
  --candidate-ref-summary CANDIDATE_REF \
  --reference-ref-summary REFERENCE_REF \
  --candidate-strict2031-summary CANDIDATE_TN \
  --reference-strict2031-summary REFERENCE_TN \
  --iterations 5000 --seed 20260719 --output BOOTSTRAP.json

python tools/build_stageb_u2v5_ablation_final_receipt.py \
  --preregistration PREREG.json \
  --results-manifest RESULTS.json \
  --bootstrap BOOTSTRAP_1.json BOOTSTRAP_2.json \
  --paper-tables PAPER_TABLES.json \
  --output FINAL_RECEIPT.json
```

The bootstrap samples image IDs, applies the same draw to both models and all
three seeds, recomputes per-seed metrics, and recomputes both positive-q05
thresholds on every FPR replicate.

## Completed engineering evidence

- A1 and A2 U1: declared owner only, finite/nonzero, zero AMP skips.
- C2 and D1 U1: exactly confidence12, finite/nonzero, zero AMP skips.
- O0 2A+1C: gradient cosine `-0.7643`, sign conflict `0.7225`.
- O1/O2 2A+1C: executable; O2 structural isolation passed.
- Multi-route one-batch smoke emitted all route-level records.
- Focused compatibility suite: 88 tests passed.

All of the above are dirty-source engineering probes, not paper results.
