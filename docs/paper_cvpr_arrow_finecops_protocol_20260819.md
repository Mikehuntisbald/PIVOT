# ARROW FineCops-Ref external zero-shot protocol

## Claim boundary

This block evaluates the sealed ARROW release on FineCops-Ref without using
FineCops train/val for optimization, checkpoint selection, threshold fitting,
or alias construction.  It is a **FineCops-specific external benchmark
zero-shot** evaluation, not an image-disjoint claim.

The distinction is material.  FineCops is built on GQA/Visual Genome images.
Of its 4,313 original test images, 2,107 have a COCO crosswalk and 2,007 occur
in the Stage-A COCO-train surface.  The historical clean support bank also
contains crops sourced from 1,641 FineCops test images.  Runtime support is
therefore filtered against every FineCops GQA source ID and its COCO crosswalk,
but this cannot erase historical image exposure from the frozen model.

## Immutable surface

- Official release: Figshare `10.6084/m9.figshare.26048050.v1`.
- Test: 9,605 positive expressions, 9,814 negative expressions, and 8,507
  negative images.
- Positive difficulty: 5,730 level-1, 3,404 level-2, and 471 level-3 rows.
- Official evaluator source: external checkout at commit
  `31d2c8615e65ccef6a4ff516925ef5ae465ec747`; its unlicensed source is not
  vendored into this repository.
- Checkpoints: sealed ARROW A/B/C routes for seeds 17, 42, and 73.  No
  milestone is selected on FineCops.

The A route uses only the existing canonical taxonomy's exact last-wins name
mapping.  It covers 9,182/9,605 positives (95.60%).  No FineCops-specific alias
is introduced.  Unsupported A positives count as failures only in the
coverage-penalized lower bound; unsupported negatives never count as correct
rejection.  B uses the annotation's active `tuple[0][0]` noun as category text,
and C reads neither category nor support.  All three keep the full expression
in the frozen GroundingDINO geometry and R100 ranking path.

## Routing and scores

Each record exports exactly one box per route:

- B58: native B58 ranking and native absolute score;
- R100+D3: raw R100 ranking and the sealed D3 confidence;
- ARROW A/B/C: Gap3 admission followed by raw R100 ranking and the same sealed
  D3 confidence.

The raw score remains in the audit record and is used with the pre-existing D3
calibration threshold.  A monotonic sigmoid copy is emitted solely for the
official FineCops evaluator, whose implementation discards non-positive
scores.  FineCops-derived FPR95 is a domain-normalized diagnostic and never a
deployment threshold.

## Statistics

The official historical outputs and an independent audited implementation are
reported separately.  The audited version uses all positives and treats score
ties as rejection failures; the official overall AUROC/FPR95 behavior uses
only level-1 positives.

Paired uncertainty uses 5,000 PCG64 replicates with seed `20260819`.  Sampling
clusters by the parent GQA image carries all of its expressions and both kinds
of negatives together.  The same draw is applied to all routes and all three
training seeds.  Planned admission contrasts are A−B and B−C on the exact
A-covered matched surface, with Holm correction.

## Fail-closed execution

Before any model forward, the preregistration binds the test annotations,
extracted images, support selection, canonical taxonomy, nine checkpoints,
three configs, source files, D3 thresholds, commands, and bootstrap contract.
Any checksum, sample identity, support provenance, frozen-route parity, or
checkpoint drift aborts the block.  Results cannot trigger a new alias, Gap,
checkpoint, or threshold under this version.
