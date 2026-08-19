PYTHON ?= python3
LATEXMK ?= latexmk
TECTONIC ?= tectonic

LATEX_FLAGS := -pdf -halt-on-error -interaction=nonstopmode -file-line-error
REGISTRY := data/paper_numbers.json
SOURCE_REGISTRY := data/source_registry.json
TABLE_SOURCES := $(wildcard data/table_sources/*.csv)
PLOT_SOURCES := $(wildcard data/plot_sources/*.csv)
FIGURE_SCRIPTS := $(sort $(wildcard scripts/make_*figure.py scripts/make_*figures.py))

MAIN_FIGURES := \
	figures/fig1_teaser.pdf \
	figures/fig2_method_ownership.pdf \
	figures/fig3_mechanism_controllability.pdf \
	figures/fig4_external_transfer.pdf

.PHONY: all pdf generated tables figures main supplement validate validate-static \
	verify-sources refresh-sources clean distclean help

# A normal checkout build starts from committed registries and CSVs.  It never
# refreshes numbers from host-local experiment outputs.
all: generated pdf validate

generated: tables figures

tables: $(REGISTRY) $(SOURCE_REGISTRY) scripts/make_main_tables.py $(TABLE_SOURCES)
	$(PYTHON) scripts/make_main_tables.py

figures: $(REGISTRY) $(SOURCE_REGISTRY)
	@test -n "$(FIGURE_SCRIPTS)" || { \
		echo "No figure generator found under paper/scripts/." >&2; \
		exit 1; \
	}
	@for script in $(FIGURE_SCRIPTS); do \
		echo "[paper] $$script"; \
		$(PYTHON) "$$script" || exit $$?; \
	done
	@for figure in $(MAIN_FIGURES); do \
		test -s "$$figure" || { echo "Missing generated figure: $$figure" >&2; exit 1; }; \
	done

pdf: main supplement

main: generated
	@if command -v "$(LATEXMK)" >/dev/null 2>&1; then \
		$(LATEXMK) $(LATEX_FLAGS) main.tex; \
	elif command -v "$(TECTONIC)" >/dev/null 2>&1; then \
		$(TECTONIC) main.tex --keep-logs --keep-intermediates; \
	else \
		echo "Neither latexmk nor tectonic is available." >&2; exit 1; \
	fi

supplement: generated
	@if command -v "$(LATEXMK)" >/dev/null 2>&1; then \
		cd supplement && TEXINPUTS=..:$${TEXINPUTS:-} \
		$(LATEXMK) $(LATEX_FLAGS) supplement.tex; \
	elif command -v "$(TECTONIC)" >/dev/null 2>&1; then \
		cd supplement && $(TECTONIC) -Z search-path=.. supplement.tex \
			--keep-logs --keep-intermediates; \
	else \
		echo "Neither latexmk nor tectonic is available." >&2; exit 1; \
	fi

validate:
	$(PYTHON) scripts/validate_paper_package.py \
		--paper-root . \
		--main-pdf main.pdf \
		--supplement-pdf supplement/supplement.pdf \
		--json-report validation_report.json

# Source-only gate for hosts without TeX Live.  It still checks the committed
# registry/CSV contract, public names, citations, references, and figure files.
validate-static:
	$(PYTHON) scripts/validate_paper_package.py --paper-root . --skip-latex \
		--json-report validation_static_report.json

# Explicit opt-in for the experiment host.  This target verifies the sealed
# artifacts and rewrites the committed registries; it is deliberately absent
# from every normal build dependency.
verify-sources:
	$(PYTHON) scripts/build_gap_sensitivity.py --check
	$(PYTHON) scripts/build_number_registry.py --check

refresh-sources:
	$(PYTHON) scripts/build_gap_sensitivity.py
	$(PYTHON) scripts/build_number_registry.py
	$(PYTHON) scripts/make_main_tables.py

clean:
	@if command -v "$(LATEXMK)" >/dev/null 2>&1; then $(LATEXMK) -C main.tex; fi
	@if command -v "$(LATEXMK)" >/dev/null 2>&1; then \
		cd supplement && TEXINPUTS=..:$${TEXINPUTS:-} $(LATEXMK) -C supplement.tex; \
	fi
	@rm -f main.bbl main.blg supplement/supplement.bbl supplement/supplement.blg

distclean: clean
	@rm -f main.pdf supplement/supplement.pdf validation_report.json \
		validation_static_report.json

help:
	@echo "make all             regenerate from committed registries, compile, validate"
	@echo "make validate-static run all non-LaTeX package checks"
	@echo "make verify-sources  EXPERIMENT HOST ONLY: verify sealed source hashes"
	@echo "make refresh-sources EXPERIMENT HOST ONLY: replay sealed local artifacts"
	@echo "make clean           remove LaTeX intermediates"
