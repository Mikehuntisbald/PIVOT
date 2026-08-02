# Stage-B Final Dual Gate

The final comparison consumes per-example `*.records.jsonl` files. The Stage-B
Ref/TN evaluators and the pure text GroundingDINO Ref/TN evaluator write these
files under `OUTPUT_DIR/per_example_records/` by default.

Each line must contain the following common fields:

- `task`: `tn` or `ref`
- `manifest_key`: `tn_global` or `ref:<official_split_name>`
- `manifest_sha256`, `manifest_n`, `manifest_index`
- `sample_id`, `image_id`, `split`, `valid`

TN records additionally require finite `pos_score` and `neg_score`. Ref records
require `correct50`, or a finite `top1_iou` from which `correct50` is derived.
Invalid examples must remain in the file with `valid=false`; they must not be
dropped from the denominator.

Run the gate with exactly one selected scorer/beta for each model. Passing a
directory containing several beta variants intentionally fails duplicate and N
validation.

```bash
python3 tools/verify_stageb_dual_gate.py \
  --baseline_records BASELINE_TN.records.jsonl BASELINE_REF_*.records.jsonl \
  --candidate_records CANDIDATE_TN.records.jsonl CANDIDATE_REF_*.records.jsonl \
  --output outputs/final_stageb_dual_gate.json
```

By default all eight official RefCOCO/+/g splits are required. Use
`--required_ref_splits ...` only for an explicitly named development protocol.
The command exits `0` only when candidate global exact FPR@95TPR is strictly
lower and candidate acc50 is strictly higher on every observed Ref split.
Manifest hash, order, N, duplicate, and invalid checks must all pass first.

The report also contains paired percentile confidence intervals from bootstrap
resampling clustered by `image_id`. Confidence intervals are diagnostic and do
not replace the point-estimate acceptance rule.
