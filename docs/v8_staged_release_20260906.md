# v8 staged release

This release separates experiment implementation, immutable evidence, and paper
integration. It does not launch training, reopen FineCops Test, or change any
reported estimate.

| Stage | Scope | Commit |
| --- | --- | --- |
| 1 | Matched confidence readout/coverage implementations and tests | `cfffbad` |
| 2 | Completed aggregate analyses, protocols, and lightweight receipts | `1d4cbcd` |
| 3 | v6–v8 manuscript snapshots, current v8 presentation, and portable build gate | This document's commit |

## Release validation

Validation used independent directories exported from the Git index, so
uncommitted files in the research working tree could not satisfy dependencies.

- Stage 1 synthetic/runtime contract suite: **160 passed, 1 skipped**. CUDA was
  disabled; the skipped CUDA smoke is not a fresh GPU validation claim.
- Stage 3 asset, portable-gate, and PDF-validator suite: **85 passed**.
- `make -C paper current TECTONIC=/path/to/tectonic` built both current PDFs.
- PDF/log validation passed with no required-check errors: main **6 pages**
  including references, supplement **22 pages**.
- Rebuilt PDFs have exactly the same `pdftotext -layout` output as the reviewed
  versioned PDFs. Binary PDF equality is not claimed.
- The TeX build retains the upstream `lineno.sty:296` UTF-8 warning.

The portable gate verifies existing source/renderer/output SHA bindings using
paths relative to the checkout. It does not rewrite sealed receipts, rerun a
detector, or repeat a bootstrap. Hash-bound generated whitespace is retained.

## Scope boundaries

Weights, images, per-example records, and build scratch files are excluded.
Aggregate evidence and reviewed manuscript/figure PDFs are included. Unrelated
legacy ownership edits, earlier empirical drafts, and unfinished artifacts
remain in the original working tree; none were deleted by this release.

The repository is not an anonymous review bundle: historical receipts retain
host paths. A venue submission and its anonymity/format checks are separate
from this source release.
