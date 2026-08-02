# Stage-B v15 fixed Stage-A Top-K verification

## Why the older TN assets are not exact

Three existing protocols have different, narrower evidence:

| Asset | What was actually reviewed | Why it is not v15 exact Top-K |
| --- | --- | --- |
| `stageb_v15_global_verified_train_20260711` | The target and 1-8 cached proposals in each source row | No Stage-A checkpoint, query transform, support patch, or current Top-50 candidate binding |
| `stageb_gdino_adapter_semantic_verified_20260711` | The same cached proposal coverage, promoted as a semantic training objective | Its audit explicitly says `global_max_label_is_semantic_extrapolation=true` |
| fixed-GDINO top1 | A checkpoint/train/deploy-transform-specific union around one pure-GDINO maximum | It is a top1 pure-GDINO protocol, not the patch-selected v15 Top-50 set |

Do not rename any of these rows to the exact scope. The exact loader rejects
their schemas and scopes.

## Exact claim

The new scope is:

```text
image_global_fixed_stagea_topk_exact_verified
```

It means only this:

> For one frozen Stage-A checkpoint, model/data/canonical-class config, one
> deterministic query transform, one row-locked support patch, and one ordered
> patch-logit Top-K selection contract, every candidate consumed by Stage B was
> reviewed and judged not to satisfy the TN expression.

It does not claim semantic absence everywhere in the image and does not claim
that all 900 decoder queries were reviewed (`all_stagea_queries_verified=false`).
It is nevertheless an exact label
for the deployed Stage-B global maximum because Stage B consumes only that
verified candidate set.

## Required extraction closure

The GPU extraction step must emit one
`stage-b-v15-fixed-stagea-topk-extraction-v1` JSONL row per source pair and a
completed `stage-b-v15-fixed-stagea-topk-extraction-audit-v1` sidecar. The audit
binds:

- the Stage-A checkpoint file SHA-256;
- the Stage-A model config, data config, and canonical-class file SHA-256;
- canonical query-transform and support-transform payload hashes;
- the ordered candidate-selection payload hash;
- the extraction JSONL path, size, row count, and SHA-256.

Each extraction row binds the source pair and source image, the exact support
patch path/SHA/class, the per-row query transform trace, and exactly K ordered
candidates. Every candidate contains rank, decoder query index, normalized
`cxcywh` box, patch logit, and a canonical candidate hash. The ordered list has
its own `candidate_set_sha256`.

For v15 the candidate-selection contract must include:

```json
{
  "schema": "stage-b-v15-fixed-stagea-topk-candidate-selection-contract-v1",
  "candidate_topk": 50,
  "score_source": "score_patch_logits",
  "selection": "torch.topk(largest=true,sorted=true)",
  "candidate_order": "descending_patch_logit",
  "candidate_box_space": "normalized_cxcywh",
  "fixed_support_patch_per_row": true,
  "deterministic_query_transform": true,
  "dynamic_candidate_replay_must_match": true,
  "candidate_box_atol": 0.00001
}
```

The current training transform is suitable only while `fix_size=true` and
`data_aug_hflip_prob=0.0`. The support patch must not be sampled randomly from
the patch bank: its path and SHA are part of every extraction row.

## Required judgment closure

The external review step must emit exactly `rows * K` completed judgment rows
plus a completed judgment audit. A row is keyed by `(sample_id, candidate_rank)`
and also binds query index, candidate hash, candidate-set hash, extraction-row
hash, evidence asset hash, judge-contract hash, answer, and confidence.

The judge contract records whether review was human, model, or hybrid, along
with prompt and evidence-asset policy hashes and the minimum confidence for a
`no` answer. Missing, duplicate, orphan, or mixed-provenance judgments abort the
build. A `yes` candidate rejects the source row; uncertain or low-confidence
answers quarantine it; only K complete confident `no` answers are admitted.

The repository separates evidence generation from judgment. The renderer emits
only `status=pending` rows with null `answer`, `confidence`, and
`judgment_sha256`; it never calls a judge and cannot produce an admissible
judgment file. The sealer accepts a different, explicitly supplied external
decision file and refuses missing or orphan keys. Before packaging decisions it
revalidates every extraction/candidate binding and re-hashes every evidence
asset. It never fills a default answer.

## End-to-end extraction and review workflow

First inspect and lock the extraction plan without loading the model or GPU:

```bash
python tools/extract_stageb_v15_fixed_stagea_topk.py \
  --checkpoint /path/to/frozen_stagea.pth \
  --model-config /path/to/stagea_model.py \
  --query-transform-config /path/to/stageb_query_config.py \
  --data-config /path/to/d4_dataset.json \
  --source-pairs /path/to/d4_source_pairs.jsonl \
  --output /path/to/d4/extractions.jsonl \
  --audit /path/to/d4/extractions.audit.json \
  --work-dir /path/to/d4/extraction_work \
  --dry-run
```

Remove `--dry-run` to perform the frozen CUDA extraction. Formal D4 does not
permit a nonstandard K: `candidate_topk=50` and
`candidate_box_atol=0.00001` are enforced. A stopped run resumes from validated
per-row shards; finalized output and audit files are never overwritten.

Render inspectable evidence and an unanswered review manifest:

```bash
python tools/render_stageb_v15_fixed_stagea_topk_evidence.py \
  --extractions /path/to/d4/extractions.jsonl \
  --extraction-audit /path/to/d4/extractions.audit.json \
  --prompt-template docs/stage_b_v15_exact_topk_review_prompt.txt \
  --judge-type human \
  --min-no-confidence 0.90 \
  --review-manifest /path/to/d4/pending_reviews.jsonl \
  --audit /path/to/d4/pending_reviews.audit.json \
  --assets-dir /path/to/d4/evidence \
  --work-dir /path/to/d4/evidence_work
```

The external reviewer must return exactly one row for every pending
`(sample_id, candidate_rank)` key. This is the only accepted decision schema:

```json
{
  "schema": "stage-b-v15-fixed-stagea-topk-external-review-decision-v1",
  "sample_id": "the unchanged pending sample_id",
  "candidate_rank": 0,
  "answer": "no",
  "confidence": 0.97,
  "reviewer_note": "optional free-form note"
}
```

Allowed answers are `yes`, `no`, and `uncertain`. Do not edit the pending
manifest. Store the explicit rows in a separate `completed_reviews.jsonl`, then
package them into the exact judgment schema expected by the CPU builder:

```bash
python tools/seal_stageb_v15_fixed_stagea_topk_reviews.py \
  --extractions /path/to/d4/extractions.jsonl \
  --extraction-audit /path/to/d4/extractions.audit.json \
  --review-manifest /path/to/d4/pending_reviews.jsonl \
  --review-audit /path/to/d4/pending_reviews.audit.json \
  --completed-reviews /path/to/d4/completed_reviews.jsonl \
  --judgments /path/to/d4/judgments.jsonl \
  --judgment-audit /path/to/d4/judgments.audit.json
```

Both commands support `--verify-only`; the sealer also supports `--dry-run`.
The dry run verifies exact key coverage and computes the packaged rows without
writing judgments. For the paper-level human audit, retain the two independent
annotator files, the adjudicated decision file supplied above, reviewer counts,
disagreement counts, and Cohen's kappa. The sealer packages only the explicit
final decision; it does not simulate a second reviewer or compute agreement.

## CPU verification and pair build

```bash
python tools/build_stageb_v15_fixed_stagea_topk_exact_pairs.py \
  --extractions /path/to/extractions.jsonl \
  --extraction-audit /path/to/extraction_audit.json \
  --judgments /path/to/judgments.jsonl \
  --judgment-audit /path/to/judgment_audit.json \
  --output /path/to/train_pairs.jsonl \
  --decisions /path/to/decisions.jsonl \
  --audit /path/to/audit.json
```

The output sidecar closes over all four inputs, the decisions, and the accepted
annotation by path, size, rows, and SHA-256.

## Dataset fail-closed configuration

An exact data source must set the strict switch, sidecar path, and every expected
singleton from the extraction audit. Placeholder hashes are not valid:

```json
{
  "require_fixed_stagea_topk_exact_verified": true,
  "fixed_stagea_topk_exact_audit": "/path/to/audit.json",
  "fixed_stagea_topk_expected_contract": {
    "checkpoint_sha256": "<64 lowercase hex>",
    "model_config_sha256": "<64 lowercase hex>",
    "data_config_sha256": "<64 lowercase hex>",
    "canonical_classes_sha256": "<64 lowercase hex>",
    "query_transform_contract_sha256": "<64 lowercase hex>",
    "support_transform_contract_sha256": "<64 lowercase hex>",
    "candidate_selection_contract_sha256": "<64 lowercase hex>",
    "candidate_topk": 50,
    "candidate_box_atol": 0.00001
  }
}
```

At dataset construction the loader re-hashes the annotation, extraction,
judgment, and sidecar closure and validates every candidate rank `0..K-1`. At
sample loading it re-hashes the fixed support patch. During training the model
compares runtime ordered Top-K query IDs and candidate boxes against the reviewed
record; any support, transform, checkpoint, numerical-order, or box drift fails
before the confidence loss is evaluated.

## Work still requiring GPU or review

This repository now contains the extractor, resumable evidence renderer,
external-decision sealer, CPU verifier/builder, runtime enforcement, and
synthetic CPU-chain tests. It does not contain a completed formal v15 Top-50
extraction or any externally supplied visual judgments. The remaining external
work is:

1. Run the frozen Stage-A checkpoint over the chosen source rows with the exact
   deterministic query/support transforms and save all 50 candidate regions.
2. Render evidence assets for every candidate and complete external human/VLM
   review. Do not treat the pending manifest as completed judgments.
3. Seal the externally supplied judgments, run the CPU builder, then insert the
   resulting literal hashes into the dataset config.
4. Run a small training preflight. Runtime candidate replay must pass before a
   full Stage-B run is allowed.
