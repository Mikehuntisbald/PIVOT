# ARROW paper package

This directory is the reproducible manuscript package for ARROW. A normal
build is deliberately independent of model checkpoints, datasets, and the
host-local `outputs/` tree. It consumes the committed semantic registry,
table/plot provenance CSVs, TeX sources, and vector figure assets.

The package provisionally vendors the official
`CVPR2026-v1(latex)` author-kit tag for a CVPR 2027 submission. The validator
binds the style and bibliography files to that tag's SHA-256 values. Replace
them and the pinned hashes with the official 2027 release when it becomes
available, then rerun every validation gate.

## Prerequisites

- Python 3.11 or newer;
- TeX Live (`latexmk`, `pdflatex`, and `bibtex`) for the final CVPR build;
- the Python packages pinned by `requirements.txt`.

Tectonic may be used for a source preview when its bundle provides the pinned
T1 Times-compatible fonts.  Regardless of engine, the PDF is submission-ready
only if the package validator passes the embedded-font and page-limit gates.

For a fresh checkout:

```bash
python3 -m venv .venv-paper
. .venv-paper/bin/activate
python -m pip install -r paper/requirements.txt
make -C paper all
```

The generated review PDFs are `paper/main.pdf` and
`paper/supplement/supplement.pdf`. They and LaTeX intermediates are ignored by
Git; source figures and provenance CSVs remain tracked.

## Anonymous review packaging

The generated main and supplement PDFs are anonymous. Do **not** upload this
repository, its Git history, or the host-bound provenance JSON files as an
anonymous-review code supplement: immutable audit records intentionally retain
experiment-host paths, and the repository history is not anonymized. If code
is supplied during double-blind review, first create a separate history-free,
identity-scrubbed snapshot and audit that snapshot independently. The tracked
provenance remains appropriate for post-review artifact release.

## Build targets

```bash
make -C paper generated       # regenerate tables/plots from committed data
make -C paper main            # compile the eight-page review manuscript
make -C paper supplement      # compile the separate supplement
make -C paper validate        # source, log, PDF, page, and font gates
make -C paper validate-static # gates that do not require TeX Live
make -C paper clean
```

`make all` never refreshes experimental evidence. `refresh-sources` is an
explicit experiment-host operation that verifies sealed local artifacts and
rewrites the lightweight registries:

```bash
# Experiment host only; never a CI or clean-checkout prerequisite.
make -C paper verify-sources
make -C paper refresh-sources
git diff -- paper/data paper/tables
```

Review the diff before committing. Refreshing sources must not train a model,
rerun a held-out evaluation, change a checkpoint, or fit an external
threshold.

## Validation contract

The package validator fails on missing dependencies, unknown citations or
references, duplicate labels, overfull boxes, an over-limit main body,
unreadable vector-figure fonts, missing provenance, plot/table values that do
not match the semantic registry, normal-build scripts that reach host-local
experiment data, or internal implementation names in public manuscript
artifacts. Historical identifiers remain confined to the provenance
registries and are not scanned as public prose.
