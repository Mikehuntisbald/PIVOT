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
- either TeX Live (`latexmk`, `pdflatex`, and `bibtex`) or Tectonic 0.17+;
- the Python packages pinned by `requirements.txt`.

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
