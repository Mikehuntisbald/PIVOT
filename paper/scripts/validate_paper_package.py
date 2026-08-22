#!/usr/bin/env python3
"""Fail-closed validation for the committed ARROW paper package.

The source-only mode is suitable for a clean checkout without TeX Live.  The
full mode additionally inspects LaTeX logs, rendered PDFs, body-page count, and
the effective font sizes of vector figures.  Neither mode reads ``outputs/``
or regenerates the semantic registry.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_MAIN_SECTIONS = tuple(
    f"sections/{index:02d}_{name}.tex"
    for index, name in enumerate(
        (
            "intro",
            "related",
            "problem",
            "method",
            "experiments",
            "results",
            "limitations",
            "conclusion",
        ),
        start=1,
    )
)
REQUIRED_TABLES = tuple(f"tables/table{index}_{name}.tex" for index, name in (
    (1, "main"),
    (2, "ownership"),
    (3, "admission"),
    (4, "external"),
))
REQUIRED_SUPPLEMENT_TABLES = (
    "supplement/tables/supp_training_controls.tex",
    "supplement/tables/supp_ref_breakdown.tex",
    "supplement/tables/supp_gap_sensitivity.tex",
    "supplement/tables/supp_finecops_subgroups.tex",
    "supplement/tables/supp_input_interventions.tex",
    "supplement/tables/supp_strong_e6_ownership.tex",
    "supplement/tables/supp_pretrain_ownership.tex",
    "supplement/tables/supp_original_parent_ownership.tex",
)
REQUIRED_FIGURES = (
    "figures/fig1_teaser.pdf",
    "figures/fig2_method_ownership.pdf",
    "figures/fig3_mechanism_controllability.pdf",
    "figures/fig4_external_transfer.pdf",
)
REQUIRED_TABLE_CSVS = tuple(
    f"data/table_sources/table{index}_{name}.csv" for index, name in (
        (1, "main"),
        (2, "ownership"),
        (3, "admission"),
        (4, "external"),
    )
)
REQUIRED_SUPPLEMENT_TABLE_CSVS = (
    "data/table_sources/supp_training_controls.csv",
    "data/table_sources/supp_ref_breakdown.csv",
    "data/table_sources/supp_gap_sensitivity.csv",
    "data/table_sources/supp_finecops_subgroups.csv",
    "data/table_sources/supp_input_interventions.csv",
    "data/table_sources/supp_strong_e6_ownership.csv",
    "data/table_sources/supp_pretrain_ownership.csv",
    "data/table_sources/supp_original_parent_ownership.csv",
)
REQUIRED_PLOT_CSVS = (
    "data/plot_sources/fig1_qualitative.csv",
    "data/plot_sources/fig2_method_ownership.csv",
    "data/plot_sources/fig3_mechanism_controllability.csv",
    "data/plot_sources/fig4_external_transfer.csv",
    "data/plot_sources/figS1_confidence_anatomy.csv",
    "data/plot_sources/qualitative_appendix.csv",
)
REQUIRED_SUPPLEMENT_SECTIONS = (
    "supplement/sections/a_scope_and_provenance.tex",
    "supplement/sections/b_training_and_ownership.tex",
    "supplement/sections/c_statistics.tex",
    "supplement/sections/d_external_protocols.tex",
    "supplement/sections/e_additional_analyses.tex",
    "supplement/sections/f_reproducibility.tex",
)
OFFICIAL_TEMPLATE_SHA256 = {
    # Official cvpr-org/author-kit tag CVPR2026-v1(latex).  Update these
    # together with the vendored files when the CVPR 2027 kit is released.
    "cvpr.sty": "2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153",
    "ieeenat_fullname.bst": "e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc",
}

FORBIDDEN_PUBLIC_NAMES = {
    "PIVOT": re.compile(r"(?<![A-Za-z0-9])PIVOT(?![A-Za-z0-9])", re.IGNORECASE),
    "U2-v5": re.compile(r"(?<![A-Za-z0-9])U2[-‑–—]v5(?![A-Za-z0-9])", re.IGNORECASE),
    "B58": re.compile(r"(?<![A-Za-z0-9])B58(?![A-Za-z0-9])", re.IGNORECASE),
    "R100": re.compile(r"(?<![A-Za-z0-9])R100(?![A-Za-z0-9])", re.IGNORECASE),
    "D3-disjoint": re.compile(
        r"(?<![A-Za-z0-9])D3[-‑–—_]disjoint(?![A-Za-z0-9])", re.IGNORECASE
    ),
    "surface8": re.compile(r"(?<![A-Za-z0-9])surface8(?![A-Za-z0-9])", re.IGNORECASE),
    "confidence12": re.compile(
        r"(?<![A-Za-z0-9])confidence12(?![A-Za-z0-9])", re.IGNORECASE
    ),
    "Gap3": re.compile(r"(?<![A-Za-z0-9])Gap3(?![A-Za-z0-9])", re.IGNORECASE),
}

APPROVED_LEGACY_MAP = Path("supplement/sections/a_scope_and_provenance.tex")
LEGACY_MAP_BEGIN = "% ARROW_APPROVED_LEGACY_MAP_BEGIN"
LEGACY_MAP_END = "% ARROW_APPROVED_LEGACY_MAP_END"
QUALITATIVE_APPENDIX_CSV = Path("data/plot_sources/qualitative_appendix.csv")
QUALITATIVE_PUBLIC_FIELDS = (
    "label",
    "surface",
    "predicate",
    "selection_order",
    "kind",
    "expression",
    "metrics_json",
    "display_note",
)

INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
GRAPHICS_RE = re.compile(
    r"\\includegraphics(?:\[([^]]*)\])?\s*\{([^}]+)\}", re.DOTALL
)
CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear)"
    r"(?:\s*\[[^]]*\]){0,2}\s*\{([^}]+)\}"
)
REF_RE = re.compile(
    r"\\(?:ref|pageref|autoref|cref|Cref|eqref)\*?\s*\{([^}]+)\}"
)
LABEL_RE = re.compile(r"\\label\s*\{([^}]+)\}")
BIB_RE = re.compile(r"\\bibliography\s*\{([^}]+)\}")
BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
OVERFULL_RE = re.compile(
    r"Overfull \\(?:hbox|vbox)(?: \(([0-9.]+)pt too (?:wide|high)\))?"
)
FONT_SIZE_RE = re.compile(rb"/[A-Za-z0-9_.+-]+\s+([0-9]+(?:\.[0-9]+)?)\s+Tf\b")
TABLE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9_])"
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def strip_comments(text: str) -> str:
    """Remove TeX comments while preserving escaped percent signs."""
    cleaned: list[str] = []
    for line in text.splitlines():
        cursor = 0
        while True:
            index = line.find("%", cursor)
            if index < 0:
                cleaned.append(line)
                break
            slash_count = 0
            probe = index - 1
            while probe >= 0 and line[probe] == "\\":
                slash_count += 1
                probe -= 1
            if slash_count % 2 == 0:
                cleaned.append(line[:index])
                break
            cursor = index + 1
    return "\n".join(cleaned)


def split_keys(value: str) -> set[str]:
    return {token.strip() for token in value.split(",") if token.strip()}


def resolve_tex_dependency(token: str, current: Path, paper_root: Path) -> Path | None:
    value = Path(token.strip())
    candidates: list[Path] = []
    for base in (current.parent, paper_root):
        candidate = base / value
        candidates.append(candidate)
        if not candidate.suffix:
            candidates.append(candidate.with_suffix(".tex"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def collect_tex_graph(entrypoint: Path, paper_root: Path, report: Report) -> list[Path]:
    pending = [entrypoint.resolve()]
    visited: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        if not current.is_file():
            report.error(f"missing TeX entry/dependency: {current}")
            continue
        visited.add(current)
        text = strip_comments(current.read_text(encoding="utf-8"))
        for token in INPUT_RE.findall(text):
            dependency = resolve_tex_dependency(token, current, paper_root)
            if dependency is None:
                report.error(f"{current.relative_to(paper_root)}: missing input {{{token}}}")
            else:
                pending.append(dependency)
    return sorted(visited)


def check_required_files(paper_root: Path, report: Report) -> None:
    required = (
        "Makefile",
        "README.md",
        "requirements.txt",
        "main.tex",
        "preamble.tex",
        "references.bib",
        "cvpr.sty",
        "ieeenat_fullname.bst",
        "data/paper_numbers.json",
        "data/source_registry.json",
        "data/arrow_release_manifest.json",
        "data/generated_numbers.tex",
        "data/gap_sensitivity.json",
        "data/closest_work_matrix.csv",
        "data/qualitative_selection.json",
        "data/qualitative_appendix.json",
        "scripts/build_number_registry.py",
        "scripts/build_gap_sensitivity.py",
        "scripts/figure_common.py",
        "scripts/make_external_figure.py",
        "scripts/make_main_tables.py",
        "scripts/make_mechanism_figure.py",
        "scripts/make_method_figure.py",
        "scripts/select_qualitative_examples.py",
        "scripts/select_qualitative_appendix.py",
        "scripts/validate_paper_package.py",
        "supplement/supplement.tex",
        "supplement/figures/figS1_confidence_anatomy.pdf",
        "qualitative/qualitative_appendix_grid.pdf",
        *REQUIRED_MAIN_SECTIONS,
        *REQUIRED_TABLES,
        *REQUIRED_SUPPLEMENT_TABLES,
        *REQUIRED_TABLE_CSVS,
        *REQUIRED_PLOT_CSVS,
        *REQUIRED_FIGURES,
        *REQUIRED_SUPPLEMENT_SECTIONS,
    )
    for relative in required:
        path = paper_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            report.error(f"missing or empty required paper artifact: {relative}")

    plot_csvs = sorted((paper_root / "data/plot_sources").glob("*.csv"))
    if any(path.stat().st_size == 0 for path in plot_csvs):
        report.error("one or more plot provenance CSVs are empty")
    report.facts["plot_source_csv_count"] = len(plot_csvs)


def load_object(path: Path, report: Report) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        report.error(f"cannot read JSON {path}: {error}")
        return None
    if not isinstance(value, dict):
        report.error(f"expected JSON object: {path}")
        return None
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_vendored_cvpr_template(paper_root: Path, report: Report) -> None:
    hashes: dict[str, str] = {}
    for relative, expected in OFFICIAL_TEMPLATE_SHA256.items():
        path = paper_root / relative
        if not path.is_file():
            continue
        actual = file_sha256(path)
        hashes[relative] = actual
        if actual != expected:
            report.error(
                f"{relative}: vendored CVPR author-kit hash drift; "
                f"expected {expected}, got {actual}"
            )
    report.facts["cvpr2026_author_kit_sha256"] = hashes


def check_registry_and_csvs(paper_root: Path, report: Report) -> None:
    number_path = paper_root / "data/paper_numbers.json"
    source_path = paper_root / "data/source_registry.json"
    if not number_path.is_file() or not source_path.is_file():
        return
    number_registry = load_object(number_path, report)
    source_registry = load_object(source_path, report)
    if number_registry is None or source_registry is None:
        return
    if number_registry.get("schema") != "arrow.paper.semantic_number_registry/v1":
        report.error("paper_numbers.json schema drift")
    if source_registry.get("schema") != "arrow.paper.source_registry/v1":
        report.error("source_registry.json schema drift")
    numbers = number_registry.get("numbers")
    if not isinstance(numbers, dict) or not numbers:
        report.error("paper_numbers.json has no semantic numbers")
        return

    used_keys: set[str] = set()
    for relative in (*REQUIRED_TABLE_CSVS, *REQUIRED_SUPPLEMENT_TABLE_CSVS):
        csv_path = paper_root / relative
        if not csv_path.is_file():
            continue
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as error:
            report.error(f"cannot parse {relative}: {error}")
            continue
        if not rows:
            report.error(f"empty table provenance CSV: {relative}")
            continue
        for row_number, row in enumerate(rows, start=2):
            key = (row.get("key") or "").strip()
            if key not in numbers:
                report.error(f"{relative}:{row_number}: unknown registry key {key!r}")
                continue
            used_keys.add(key)
            item = numbers[key]
            if not isinstance(item, dict):
                report.error(f"registry item is not an object: {key}")
                continue
            for column in ("source_path", "source_sha256", "source_json_path"):
                csv_value = (row.get(column) or "").strip()
                registry_value = "" if item.get(column) is None else str(item.get(column))
                if csv_value != registry_value:
                    report.error(
                        f"{relative}:{row_number}: {column} differs from registry for {key}"
                    )
            try:
                csv_value = float(row["value"])
                registry_value = float(item["value"])
                if not math.isclose(csv_value, registry_value, rel_tol=0.0, abs_tol=1e-12):
                    report.error(f"{relative}:{row_number}: value differs for {key}")
            except (KeyError, TypeError, ValueError):
                report.error(f"{relative}:{row_number}: non-numeric value for {key}")
    report.facts["table_registry_key_count"] = len(used_keys)
    report.facts["semantic_number_count"] = len(numbers)


def check_public_release_manifest(paper_root: Path, report: Report) -> None:
    """Bind the tracked release copy to the immutable source-registry bytes."""

    manifest_path = paper_root / "data/arrow_release_manifest.json"
    registry_path = paper_root / "data/source_registry.json"
    if not manifest_path.is_file() or not registry_path.is_file():
        return
    manifest = load_object(manifest_path, report)
    registry = load_object(registry_path, report)
    if manifest is None or registry is None:
        return
    if manifest.get("schema") != "arrow.release_manifest/v1":
        report.error("arrow_release_manifest.json schema drift")
    if manifest.get("status") != "release_bound":
        report.error("arrow_release_manifest.json is not release_bound")
    method = manifest.get("method")
    if not isinstance(method, dict) or method.get("name") != "ARROW":
        report.error("arrow_release_manifest.json public method identity drift")
    source = registry.get("sources")
    source_item = source.get("arrow_release") if isinstance(source, dict) else None
    expected = source_item.get("sha256") if isinstance(source_item, dict) else None
    actual = file_sha256(manifest_path)
    if not isinstance(expected, str) or actual != expected:
        report.error(
            "tracked ARROW release manifest does not match immutable source registry"
        )
    checkpoints = (
        manifest.get("legacy_evidence", {}).get("main_checkpoints", {})
        if isinstance(manifest.get("legacy_evidence"), dict)
        else {}
    )
    if set(checkpoints) != {"17", "42", "73"}:
        report.error("ARROW release manifest must bind seeds 17/42/73 exactly")
    report.facts["arrow_release_manifest_sha256"] = actual


def numeric_atoms(value: object) -> list[float]:
    result: list[float] = []
    if isinstance(value, bool) or value is None:
        return result
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            result.append(float(value))
        return result
    if isinstance(value, dict):
        for child in value.values():
            result.extend(numeric_atoms(child))
        by_seed = value.get("by_seed")
        if isinstance(by_seed, dict):
            seed_values = [
                float(child)
                for child in by_seed.values()
                if isinstance(child, (int, float)) and not isinstance(child, bool)
            ]
            if seed_values:
                result.append(statistics.fmean(seed_values))
            if len(seed_values) >= 2:
                result.append(statistics.stdev(seed_values))
        return result
    if isinstance(value, list):
        for child in value:
            result.extend(numeric_atoms(child))
    return result


def check_table_numeric_literals(paper_root: Path, report: Report) -> None:
    """Bind rendered table numbers to registry values (including formatting)."""
    registry_path = paper_root / "data/paper_numbers.json"
    if not registry_path.is_file():
        return
    payload = load_object(registry_path, report)
    if payload is None or not isinstance(payload.get("numbers"), dict):
        return
    allowed: set[str] = {"0.5", "1", "5", "95"}  # metric-name constants
    for value in numeric_atoms(payload["numbers"]):
        # Tables may report fractions as percentages and parameter counts in
        # thousands.  All three views remain bound to the same registry atom.
        for transformed in (value, 100.0 * value, value / 1000.0):
            for digits in range(7):
                allowed.add(f"{transformed:.{digits}f}")
                allowed.add(f"{transformed:+.{digits}f}")
            if transformed.is_integer():
                allowed.add(f"{int(transformed):,}")

    checked = 0
    for relative in (*REQUIRED_TABLES, *REQUIRED_SUPPLEMENT_TABLES):
        path = paper_root / relative
        if not path.is_file():
            continue
        checked += 1
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in TABLE_NUMBER_RE.finditer(line):
                token = match.group(0)
                before = line[max(0, match.start() - 16) : match.start()]
                after = line[match.end() : match.end() + 20]
                # TeX geometry and a documented seed identifier are formatting
                # constants rather than reported values.
                if re.match(r"\\(?:textwidth|columnwidth|linewidth)", after):
                    continue
                if before.endswith("seed-"):
                    continue
                normalized = token.replace(",", "")
                if normalized not in allowed:
                    report.error(
                        f"{relative}:{line_number}: numeric literal {token!r} "
                        "is not reproducible from paper_numbers.json"
                    )
    report.facts["registry_checked_table_count"] = checked


def check_plot_registry_bindings(paper_root: Path, report: Report) -> None:
    registry_path = paper_root / "data/paper_numbers.json"
    if not registry_path.is_file():
        return
    payload = load_object(registry_path, report)
    if payload is None or not isinstance(payload.get("numbers"), dict):
        return
    numbers = payload["numbers"]
    bound_rows = 0
    for relative in REQUIRED_PLOT_CSVS:
        path = paper_root / relative
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as error:
            report.error(f"cannot parse {relative}: {error}")
            continue
        if not rows:
            report.error(f"empty plot provenance CSV: {relative}")
            continue
        if "registry_key" not in rows[0]:
            # Qualitative/method diagrams have no plotted metric; their CSVs
            # bind selection or topology instead of semantic numbers.
            continue
        for row_number, row in enumerate(rows, start=2):
            key = (row.get("registry_key") or "").strip()
            if not key:
                # Distribution knots and qualitative rows need not be scalar
                # semantic numbers, but must bind byte-for-byte to a committed
                # source receipt instead.
                source_text = (row.get("source_path") or "").strip()
                digest_text = (row.get("source_sha256") or "").strip()
                source = paper_root / source_text
                if not source_text or not digest_text or not source.is_file():
                    report.error(
                        f"{relative}:{row_number}: missing registry_key and valid source receipt"
                    )
                else:
                    actual = hashlib.sha256(source.read_bytes()).hexdigest()
                    if actual != digest_text:
                        report.error(
                            f"{relative}:{row_number}: source receipt hash differs"
                        )
                continue
            item = numbers.get(key)
            if not isinstance(item, dict):
                report.error(f"{relative}:{row_number}: unknown registry key {key!r}")
                continue
            try:
                plotted = float(row["value"])
                registered = float(item["value"])
            except (KeyError, TypeError, ValueError):
                report.error(f"{relative}:{row_number}: invalid plotted value for {key}")
                continue
            if not math.isclose(plotted, registered, rel_tol=0.0, abs_tol=1e-12):
                report.error(f"{relative}:{row_number}: plotted value differs for {key}")
            if "ci95_low" in row or "ci95_high" in row:
                low_text = (row.get("ci95_low") or "").strip()
                high_text = (row.get("ci95_high") or "").strip()
                registered_ci = item.get("ci95")
                if registered_ci is None:
                    if low_text or high_text:
                        report.error(
                            f"{relative}:{row_number}: plotted CI is not registered for {key}"
                        )
                else:
                    try:
                        low, high = (float(registered_ci[0]), float(registered_ci[1]))
                        plotted_low, plotted_high = float(low_text), float(high_text)
                    except (IndexError, TypeError, ValueError):
                        report.error(f"{relative}:{row_number}: invalid CI for {key}")
                    else:
                        if not math.isclose(
                            plotted_low, low, rel_tol=0.0, abs_tol=1e-12
                        ) or not math.isclose(
                            plotted_high, high, rel_tol=0.0, abs_tol=1e-12
                        ):
                            report.error(
                                f"{relative}:{row_number}: plotted CI differs for {key}"
                            )
            bound_rows += 1
    report.facts["plot_registry_bound_rows"] = bound_rows


def check_clean_clone_generators(paper_root: Path, report: Report) -> None:
    """Ensure default-build generators cannot reach host-local experiment data."""
    scripts = [
        paper_root / "scripts/make_main_tables.py",
        paper_root / "scripts/figure_common.py",
    ]
    scripts.extend(sorted((paper_root / "scripts").glob("make_*figure.py")))
    scripts.extend(sorted((paper_root / "scripts").glob("make_*figures.py")))
    checked: list[str] = []
    for path in scripts:
        if not path.is_file():
            continue
        checked.append(str(path.relative_to(paper_root)))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            report.error(f"cannot audit normal-build generator {path}: {error}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            literal = node.value.replace("\\", "/")
            path_parts = tuple(part for part in literal.split("/") if part)
            if "outputs" in path_parts or literal.startswith(("/media/", "/home/")):
                report.error(
                    f"{path.relative_to(paper_root)}:{getattr(node, 'lineno', '?')}: "
                    "normal-build generator references host-local/outputs data"
                )
            if "build_number_registry.py" in literal:
                report.error(
                    f"{path.relative_to(paper_root)}:{getattr(node, 'lineno', '?')}: "
                    "normal-build generator invokes source refresh"
                )
    report.facts["clean_clone_generators"] = checked


def public_text_files(paper_root: Path, tex_graphs: Sequence[Path]) -> list[Path]:
    files = set(tex_graphs)
    bibliography = paper_root / "references.bib"
    if bibliography.is_file():
        files.add(bibliography.resolve())
    readme = paper_root / "README.md"
    if readme.is_file():
        files.add(readme.resolve())
    closest_work = paper_root / "data/closest_work_matrix.csv"
    if closest_work.is_file():
        files.add(closest_work.resolve())
    files.update(path.resolve() for path in (paper_root / "data/plot_sources").glob("*.csv"))
    return sorted(files)


def check_public_names(paths: Iterable[Path], paper_root: Path, report: Report) -> None:
    for path in paths:
        relative = path.relative_to(paper_root)
        if relative == QUALITATIVE_APPENDIX_CSV:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row_number, row in enumerate(csv.DictReader(handle), start=2):
                    line = " ".join(row.get(field, "") for field in QUALITATIVE_PUBLIC_FIELDS)
                    for name, pattern in FORBIDDEN_PUBLIC_NAMES.items():
                        if pattern.search(line):
                            report.error(
                                f"{relative}:{row_number}: forbidden public name {name}"
                            )
            # Machine IDs and source-artifact keys intentionally retain their
            # sealed names and are audited separately against the JSON receipt.
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if relative == APPROVED_LEGACY_MAP:
            begin_count = text.count(LEGACY_MAP_BEGIN)
            end_count = text.count(LEGACY_MAP_END)
            if begin_count != 1 or end_count != 1:
                report.error(
                    f"{relative}: approved legacy mapping must have exactly one marker pair"
                )
            else:
                before, remainder = text.split(LEGACY_MAP_BEGIN, 1)
                _, after = remainder.split(LEGACY_MAP_END, 1)
                text = before + after
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in FORBIDDEN_PUBLIC_NAMES.items():
                if pattern.search(line):
                    report.error(
                        f"{relative}:{line_number}: forbidden public name {name}"
                    )


def check_qualitative_appendix_artifact_links(
    paper_root: Path, report: Report
) -> None:
    """Require every CSV provenance ID to resolve to the sealed JSON receipt."""

    receipt_path = paper_root / "data/qualitative_appendix.json"
    csv_path = paper_root / QUALITATIVE_APPENDIX_CSV
    if not receipt_path.is_file() or not csv_path.is_file():
        return
    receipt = load_object(receipt_path, report)
    if receipt is None:
        return
    artifacts = receipt.get("source_artifacts")
    selection = receipt.get("selection")
    if not isinstance(artifacts, dict) or not isinstance(selection, list):
        report.error("qualitative appendix receipt lacks artifacts/selection")
        return
    selected_by_order: dict[str, dict[str, object]] = {}
    for item in selection:
        if not isinstance(item, dict):
            report.error("qualitative appendix selection contains a non-object")
            continue
        order = str(item.get("order"))
        if order in selected_by_order:
            report.error(f"qualitative appendix duplicate selection order {order}")
        selected_by_order[order] = item

    seen_orders: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            order = row.get("order", "")
            if order in seen_orders:
                report.error(f"{QUALITATIVE_APPENDIX_CSV}:{row_number}: duplicate order")
            seen_orders.add(order)
            selected = selected_by_order.get(order)
            if selected is None:
                report.error(
                    f"{QUALITATIVE_APPENDIX_CSV}:{row_number}: order absent from receipt"
                )
                continue
            for field in ("sample_id", "predicate_id"):
                if row.get(field) != str(selected.get(field, "")):
                    report.error(
                        f"{QUALITATIVE_APPENDIX_CSV}:{row_number}: {field} differs from receipt"
                    )
            artifact_ids = [
                token for token in row.get("source_artifact_ids", "").split(";") if token
            ]
            if artifact_ids != selected.get("source_artifact_ids"):
                report.error(
                    f"{QUALITATIVE_APPENDIX_CSV}:{row_number}: artifact ID list differs from receipt"
                )
            for artifact_id in artifact_ids:
                if artifact_id not in artifacts:
                    report.error(
                        f"{QUALITATIVE_APPENDIX_CSV}:{row_number}: unknown artifact ID {artifact_id}"
                    )
    if seen_orders != set(selected_by_order):
        report.error("qualitative appendix CSV/receipt selection orders are incomplete")


def check_cvpr_review_contract(paper_root: Path, report: Report) -> None:
    for relative in ("main.tex", "supplement/supplement.tex"):
        path = paper_root / relative
        if not path.is_file():
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        if not re.search(
            r"\\documentclass\[10pt,twocolumn,letterpaper\]\{article\}", text
        ):
            report.error(f"{relative}: must use the CVPR letterpaper two-column class")
        if not re.search(r"\\usepackage\[review\]\{cvpr\}", text):
            report.error(f"{relative}: CVPR review mode is not enabled")
        author_match = re.search(r"\\author\s*\{([^}]*)\}", text, re.DOTALL)
        if author_match is None or "anonymous" not in author_match.group(1).lower():
            report.error(f"{relative}: review author block is not anonymous")
        if re.search(r"\\def\\paperID\s*\{\*+\}", text):
            report.warn(f"{relative}: paper ID remains a submission placeholder")


def bib_files_for_graph(graph: Sequence[Path], paper_root: Path, report: Report) -> list[Path]:
    result: set[Path] = set()
    for path in graph:
        text = strip_comments(path.read_text(encoding="utf-8"))
        for group in BIB_RE.findall(text):
            for token in split_keys(group):
                candidate = paper_root / token
                if not candidate.suffix:
                    candidate = candidate.with_suffix(".bib")
                if candidate.is_file():
                    result.add(candidate.resolve())
                else:
                    report.error(f"{path.relative_to(paper_root)}: missing bibliography {{{token}}}")
    return sorted(result)


def check_citations_and_references(
    graph: Sequence[Path], paper_root: Path, document_name: str, report: Report
) -> None:
    labels: dict[str, Path] = {}
    refs: dict[str, Path] = {}
    cites: dict[str, Path] = {}
    for path in graph:
        text = strip_comments(path.read_text(encoding="utf-8"))
        for label in LABEL_RE.findall(text):
            if label in labels:
                report.error(
                    f"{document_name}: duplicate label {label!r} in "
                    f"{labels[label].relative_to(paper_root)} and {path.relative_to(paper_root)}"
                )
            labels[label] = path
        for group in REF_RE.findall(text):
            for key in split_keys(group):
                refs.setdefault(key, path)
        for group in CITE_RE.findall(text):
            for key in split_keys(group):
                cites.setdefault(key, path)

    for key in sorted(set(refs) - set(labels)):
        report.error(
            f"{document_name}: undefined reference {key!r} used in "
            f"{refs[key].relative_to(paper_root)}"
        )

    bib_keys: set[str] = set()
    for path in bib_files_for_graph(graph, paper_root, report):
        parsed = BIB_KEY_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
        duplicates = sorted({key for key in parsed if parsed.count(key) > 1})
        for key in duplicates:
            report.error(
                f"{document_name}: duplicate bibliography key {key!r} in "
                f"{path.relative_to(paper_root)}"
            )
        bib_keys.update(parsed)
    for key in sorted(set(cites) - bib_keys):
        report.error(
            f"{document_name}: undefined citation {key!r} used in "
            f"{cites[key].relative_to(paper_root)}"
        )
    report.facts[f"{document_name}_citation_count"] = len(cites)
    report.facts[f"{document_name}_label_count"] = len(labels)


def resolve_graphic(token: str, tex_path: Path, paper_root: Path) -> Path | None:
    value = Path(token.strip())
    extensions = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")
    bases = [tex_path.parent]
    supplement_root = paper_root / "supplement"
    if supplement_root == tex_path.parent or supplement_root in tex_path.parents:
        # The supplement is compiled with its own directory as the TeX
        # working directory; mirror that lookup in the source validator.
        bases.append(supplement_root)
    bases.append(paper_root)
    for base in dict.fromkeys(bases):
        for extension in extensions:
            candidate = base / (str(value) + extension if extension and not value.suffix else value)
            if candidate.is_file():
                return candidate.resolve()
    return None


def graphic_target_width(options: str | None) -> float | None:
    if not options:
        return None
    match = re.search(
        r"width\s*=\s*(?:(0(?:\.\d+)?|1(?:\.0+)?)\s*)?\\(textwidth|columnwidth|linewidth)",
        options,
    )
    if not match:
        return None
    scale = float(match.group(1)) if match.group(1) else 1.0
    # CVPR2026-v1: textwidth=6.875in and columnsep=0.3125in.
    width = 495.0 if match.group(2) == "textwidth" else 236.25
    return scale * width


def check_graphics(
    graph: Sequence[Path], paper_root: Path, min_font_pt: float, report: Report
) -> None:
    uses: dict[Path, list[float | None]] = {}
    for tex_path in graph:
        text = strip_comments(tex_path.read_text(encoding="utf-8"))
        for options, token in GRAPHICS_RE.findall(text):
            graphic = resolve_graphic(token, tex_path, paper_root)
            if graphic is None:
                report.error(f"{tex_path.relative_to(paper_root)}: missing graphic {{{token}}}")
                continue
            uses.setdefault(graphic, []).append(graphic_target_width(options))

    try:
        from pypdf import PdfReader
    except ImportError:
        if any(path.suffix.lower() == ".pdf" for path in uses):
            report.error("pypdf is required to validate figure fonts; install paper/requirements.txt")
        return

    font_facts: dict[str, float | None] = {}
    for path, target_widths in uses.items():
        if path.suffix.lower() != ".pdf":
            report.warn(f"font-size audit unavailable for raster figure {path.relative_to(paper_root)}")
            continue
        try:
            reader = PdfReader(str(path))
            sizes: list[float] = []
            source_widths: list[float] = []
            for page in reader.pages:
                source_widths.append(float(page.mediabox.width))
                try:
                    page.extract_text(
                        visitor_text=lambda _text, _cm, _tm, _font, size: (
                            sizes.append(float(size)) if float(size) > 0.0 else None
                        )
                    )
                except TypeError:
                    # Older pypdf releases do not expose visitor_text.  Their
                    # raw page stream still catches ordinary PDF text.
                    contents = page.get_contents()
                    if contents is not None:
                        sizes.extend(
                            float(value) for value in FONT_SIZE_RE.findall(contents.get_data())
                        )
        except Exception as error:  # pypdf exposes backend-specific parse errors
            report.error(f"cannot inspect figure PDF {path.relative_to(paper_root)}: {error}")
            continue
        if not sizes:
            report.error(
                f"no parseable text operators in {path.relative_to(paper_root)}; "
                "cannot prove figure-font readability"
            )
            font_facts[str(path.relative_to(paper_root))] = None
            continue
        source_width = max(source_widths) if source_widths else 0.0
        scale_candidates = [
            target / source_width
            for target in target_widths
            if target is not None and source_width > 0.0
        ]
        effective_min = min(sizes) * (min(scale_candidates) if scale_candidates else 1.0)
        font_facts[str(path.relative_to(paper_root))] = effective_min
        if effective_min + 1e-9 < min_font_pt:
            report.error(
                f"{path.relative_to(paper_root)}: effective minimum font "
                f"{effective_min:.2f}pt is below {min_font_pt:.2f}pt"
            )
    report.facts["figure_effective_min_font_pt"] = font_facts


def check_log(path: Path, label: str, max_overfull_pt: float, report: Report) -> None:
    if not path.is_file():
        report.error(f"missing LaTeX log: {path}")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    undefined_patterns = (
        r"LaTeX Warning: Citation .* undefined",
        r"LaTeX Warning: Reference .* undefined",
        r"There were undefined references",
        r"There were undefined citations",
        r"Package natbib Warning: Citation .* undefined",
    )
    for pattern in undefined_patterns:
        if re.search(pattern, text):
            report.error(f"{label}: LaTeX log contains undefined citations/references")
            break
    for match in OVERFULL_RE.finditer(text):
        amount = float(match.group(1)) if match.group(1) is not None else math.inf
        if amount > max_overfull_pt:
            amount_text = "unknown" if not math.isfinite(amount) else f"{amount:.3f}pt"
            report.error(f"{label}: overfull box ({amount_text})")


def reference_start_page(aux_path: Path) -> int | None:
    if not aux_path.is_file():
        return None
    text = aux_path.read_text(encoding="utf-8", errors="replace")
    for label in ("paper:references-start", "sec:references-start"):
        match = re.search(
            rf"\\newlabel\{{{re.escape(label)}\}}\{{\{{[^}}]*\}}\{{([0-9]+)\}}",
            text,
        )
        if match:
            return int(match.group(1))
    return None


def direct_pdf_font_names(reader: object) -> set[str]:
    """Return fonts attached directly to PDF pages, excluding form internals."""

    result: set[str] = set()
    for page in reader.pages:  # type: ignore[attr-defined]
        resources = page.get("/Resources") or {}
        fonts = resources.get("/Font") or {}
        for value in fonts.values():
            try:
                font = value.get_object()
            except AttributeError:
                font = value
            base_font = font.get("/BaseFont") if hasattr(font, "get") else None
            if base_font:
                result.add(str(base_font).lstrip("/"))
    return result


def has_times_compatible_regular(font_names: Sequence[str]) -> bool:
    """Recognize regular Times-compatible fonts used by supported CVPR builds."""

    allowed = {
        "NimbusRomNo9L-Regu",
        "NimbusRoman-Regular",
        "TeXGyreTermes-Regular",
        "Times-Roman",
        "TimesNewRomanPSMT",
    }
    return any(name.split("+", 1)[-1] in allowed for name in font_names)


def check_pdfs(
    main_pdf: Path,
    supplement_pdf: Path,
    main_aux: Path,
    max_main_pages: int,
    report: Report,
) -> None:
    try:
        from pypdf import PdfReader
    except ImportError:
        report.error("pypdf is required for PDF validation; install paper/requirements.txt")
        return
    for label, path in (("main", main_pdf), ("supplement", supplement_pdf)):
        if not path.is_file() or path.stat().st_size == 0:
            report.error(f"missing or empty {label} PDF: {path}")
            return
    try:
        main_reader = PdfReader(str(main_pdf))
        supplement_reader = PdfReader(str(supplement_pdf))
    except Exception as error:
        report.error(f"cannot parse compiled PDF: {error}")
        return
    for label, reader in (("main", main_reader), ("supplement", supplement_reader)):
        font_names = sorted(direct_pdf_font_names(reader))
        report.facts[f"{label}_direct_pdf_fonts"] = font_names
        if not has_times_compatible_regular(font_names):
            report.error(
                f"{label}: no embedded regular Times-compatible body font; "
                "compile the CVPR package with pdfLaTeX/latexmk rather than a "
                "font-fallback preview engine"
            )
    total_pages = len(main_reader.pages)
    start_page = None
    heading_extracted = False
    for index, page in enumerate(main_reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if re.search(r"(?m)^\s*References\s*$", page_text):
            start_page = index
            heading_extracted = True
            break
    if start_page is None:
        start_page = reference_start_page(main_aux)
        if start_page is not None:
            report.warn(
                "References heading could not be extracted; body-page count uses the TeX label"
            )
    body_pages = total_pages
    if start_page is not None:
        # References beginning on page k imply at most k-1 pages of body when
        # that page contains references only.  If extracted text before the
        # heading contains substantive prose, page k is conservatively body.
        body_pages = max(0, start_page - 1)
        # A TeX label records the page where references begin, but cannot tell
        # whether body prose shares that page.  If text extraction cannot
        # locate the heading, count the page conservatively as body rather
        # than silently subtracting it.
        if not heading_extracted:
            body_pages = start_page
        try:
            first_reference_text = main_reader.pages[start_page - 1].extract_text() or ""
        except Exception:
            first_reference_text = ""
        heading = re.search(r"(?mi)^\s*References\s*$", first_reference_text)
        if heading:
            prefix = first_reference_text[: heading.start()]
            prefix = re.sub(r"(?mi)^\s*(?:CVPR|Anonymous).*?$", "", prefix)
            prefix = re.sub(r"[\W\d_]+", "", prefix)
            if len(prefix) >= 40:
                body_pages = start_page
    if body_pages > max_main_pages:
        report.error(
            f"main body occupies at least {body_pages} pages; limit is {max_main_pages}"
        )
    report.facts.update(
        {
            "main_total_pages": total_pages,
            "main_body_pages_conservative": body_pages,
            "references_start_page": start_page,
            "supplement_pages": len(supplement_reader.pages),
        }
    )


def build_report(args: argparse.Namespace) -> Report:
    paper_root = args.paper_root.resolve()
    report = Report()
    check_required_files(paper_root, report)
    check_vendored_cvpr_template(paper_root, report)
    check_registry_and_csvs(paper_root, report)
    check_public_release_manifest(paper_root, report)
    check_qualitative_appendix_artifact_links(paper_root, report)
    check_table_numeric_literals(paper_root, report)
    check_plot_registry_bindings(paper_root, report)
    check_clean_clone_generators(paper_root, report)
    check_cvpr_review_contract(paper_root, report)

    main_graph = collect_tex_graph(paper_root / args.main_tex, paper_root, report)
    supplement_graph = collect_tex_graph(
        paper_root / args.supplement_tex, paper_root, report
    )
    all_tex = sorted(set(main_graph) | set(supplement_graph))
    check_public_names(public_text_files(paper_root, all_tex), paper_root, report)
    check_citations_and_references(main_graph, paper_root, "main", report)
    check_citations_and_references(supplement_graph, paper_root, "supplement", report)
    check_graphics(all_tex, paper_root, args.min_figure_font_pt, report)

    if not args.skip_latex:
        check_log(paper_root / args.main_log, "main", args.max_overfull_pt, report)
        check_log(
            paper_root / args.supplement_log,
            "supplement",
            args.max_overfull_pt,
            report,
        )
        check_pdfs(
            paper_root / args.main_pdf,
            paper_root / args.supplement_pdf,
            paper_root / args.main_aux,
            args.max_main_pages,
            report,
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--main-tex", default="main.tex")
    parser.add_argument("--supplement-tex", default="supplement/supplement.tex")
    parser.add_argument("--main-log", default="main.log")
    parser.add_argument("--supplement-log", default="supplement/supplement.log")
    parser.add_argument("--main-aux", default="main.aux")
    parser.add_argument("--main-pdf", default="main.pdf")
    parser.add_argument("--supplement-pdf", default="supplement/supplement.pdf")
    parser.add_argument("--max-main-pages", type=int, default=8)
    parser.add_argument("--max-overfull-pt", type=float, default=0.0)
    parser.add_argument(
        "--min-figure-font-pt",
        type=float,
        default=6.0,
        help="minimum effective vector-figure font after TeX scaling (default: 6pt)",
    )
    parser.add_argument("--skip-latex", action="store_true")
    parser.add_argument("--json-report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    payload = {
        "schema": "arrow.paper.validation_report/v1",
        "ok": not report.errors,
        "errors": report.errors,
        "warnings": report.warnings,
        "facts": report.facts,
        "latex_checks_skipped": args.skip_latex,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_report:
        destination = args.json_report
        if not destination.is_absolute():
            destination = args.paper_root / destination
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
