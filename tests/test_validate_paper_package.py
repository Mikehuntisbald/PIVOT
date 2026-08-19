from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "paper/scripts/validate_paper_package.py"
SPEC = importlib.util.spec_from_file_location("validate_paper_package", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_strip_comments_preserves_escaped_percent() -> None:
    text = "score is 95\\% % remove this\nnext % remove that"
    assert validator.strip_comments(text) == "score is 95\\% \nnext "


def test_tex_graph_resolves_citations_and_references(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_text(
        "\\input{section}\\bibliography{references}\n", encoding="utf-8"
    )
    (tmp_path / "section.tex").write_text(
        "\\section{A}\\label{sec:a} See \\ref{sec:a} and \\cite{paper}.\n",
        encoding="utf-8",
    )
    (tmp_path / "references.bib").write_text(
        "@article{paper, title={Evidence}, author={Anonymous}, year={2026}}\n",
        encoding="utf-8",
    )
    report = validator.Report()
    graph = validator.collect_tex_graph(tmp_path / "main.tex", tmp_path, report)
    validator.check_citations_and_references(graph, tmp_path, "main", report)
    assert report.errors == []


def test_tex_graph_rejects_undefined_citation_and_reference(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_text(
        "See \\ref{sec:missing} and \\cite{missing}.\\bibliography{references}\n",
        encoding="utf-8",
    )
    (tmp_path / "references.bib").write_text("", encoding="utf-8")
    report = validator.Report()
    graph = validator.collect_tex_graph(tmp_path / "main.tex", tmp_path, report)
    validator.check_citations_and_references(graph, tmp_path, "main", report)
    assert any("undefined reference 'sec:missing'" in error for error in report.errors)
    assert any("undefined citation 'missing'" in error for error in report.errors)


def test_log_gate_rejects_undefined_and_overfull(tmp_path: Path) -> None:
    log = tmp_path / "main.log"
    log.write_text(
        "LaTeX Warning: Citation `missing' on page 1 undefined.\n"
        "Overfull \\hbox (2.75pt too wide) in paragraph at lines 1--2\n",
        encoding="utf-8",
    )
    report = validator.Report()
    validator.check_log(log, "main", 0.0, report)
    assert any("undefined citations/references" in error for error in report.errors)
    assert any("2.750pt" in error for error in report.errors)


def test_public_name_gate_does_not_need_to_scan_registry(tmp_path: Path) -> None:
    manuscript = tmp_path / "main.tex"
    manuscript.write_text("The U2-v5 route exposes b58_iou.\n", encoding="utf-8")
    registry = tmp_path / "source_registry.json"
    registry.write_text('{"legacy": "U2-v5"}\n', encoding="utf-8")
    report = validator.Report()
    validator.check_public_names([manuscript], tmp_path, report)
    assert len(report.errors) == 2
    assert any("forbidden public name U2-v5" in error for error in report.errors)
    assert any("forbidden public name B58" in error for error in report.errors)
    assert all("source_registry" not in error for error in report.errors)


def test_public_name_gate_allows_only_marked_appendix_mapping(tmp_path: Path) -> None:
    mapping = tmp_path / validator.APPROVED_LEGACY_MAP
    mapping.parent.mkdir(parents=True)
    mapping.write_text(
        "Public prose.\n"
        f"{validator.LEGACY_MAP_BEGIN}\n"
        "Frozen base = B58; ranker = R100; lineage = U2-v5.\n"
        f"{validator.LEGACY_MAP_END}\n",
        encoding="utf-8",
    )
    report = validator.Report()
    validator.check_public_names([mapping], tmp_path, report)
    assert report.errors == []

    mapping.write_text(
        mapping.read_text(encoding="utf-8") + "Leaked B58 outside the map.\n",
        encoding="utf-8",
    )
    report = validator.Report()
    validator.check_public_names([mapping], tmp_path, report)
    assert any("forbidden public name B58" in error for error in report.errors)


def test_default_make_build_never_depends_on_refresh_sources() -> None:
    makefile = (ROOT / "paper/Makefile").read_text(encoding="utf-8")
    all_line = next(line for line in makefile.splitlines() if line.startswith("all:"))
    generated_line = next(
        line for line in makefile.splitlines() if line.startswith("generated:")
    )
    assert "refresh-sources" not in all_line
    assert "refresh-sources" not in generated_line
    assert "refresh-sources:" in makefile


def test_clean_clone_generator_gate_rejects_outputs_dependency(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "make_main_tables.py").write_text(
        'SOURCE = "outputs/local/result.json"\n', encoding="utf-8"
    )
    report = validator.Report()
    validator.check_clean_clone_generators(tmp_path, report)
    assert any("host-local/outputs data" in error for error in report.errors)


def test_table_number_gate_accepts_registry_format_and_rejects_literal(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / "data/paper_numbers.json").write_text(
        '{"numbers":{"metric":{"value":0.742398,"by_seed":'
        '{"17":0.741,"42":0.743,"73":0.742}}}}\n',
        encoding="utf-8",
    )
    for relative in validator.REQUIRED_TABLES:
        path = tmp_path / relative
        path.write_text("74.24 \\\\\n", encoding="utf-8")
    report = validator.Report()
    validator.check_table_numeric_literals(tmp_path, report)
    assert report.errors == []

    (tmp_path / validator.REQUIRED_TABLES[0]).write_text(
        "74.24 & 88.88 \\\\\n", encoding="utf-8"
    )
    report = validator.Report()
    validator.check_table_numeric_literals(tmp_path, report)
    assert any("88.88" in error for error in report.errors)


def test_plot_csv_value_is_bound_to_registry(tmp_path: Path) -> None:
    (tmp_path / "data/plot_sources").mkdir(parents=True)
    (tmp_path / "data/paper_numbers.json").write_text(
        '{"numbers":{"metric":{"value":0.25}}}\n', encoding="utf-8"
    )
    for index, relative in enumerate(validator.REQUIRED_PLOT_CSVS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if index < 2:
            path.write_text("panel,role\na,diagram\n", encoding="utf-8")
        else:
            path.write_text(
                "registry_key,value\nmetric,0.25\n", encoding="utf-8"
            )
    report = validator.Report()
    validator.check_plot_registry_bindings(tmp_path, report)
    assert report.errors == []

    (tmp_path / validator.REQUIRED_PLOT_CSVS[-1]).write_text(
        "registry_key,value\nmetric,0.5\n", encoding="utf-8"
    )
    report = validator.Report()
    validator.check_plot_registry_bindings(tmp_path, report)
    assert any("plotted value differs" in error for error in report.errors)


def test_cvpr_review_contract_requires_anonymity_and_review_mode(tmp_path: Path) -> None:
    (tmp_path / "supplement").mkdir()
    valid = (
        "\\documentclass[10pt,twocolumn,letterpaper]{article}\n"
        "\\usepackage[review]{cvpr}\n"
        "\\def\\paperID{*****}\n"
        "\\author{Anonymous CVPR submission}\n"
    )
    (tmp_path / "main.tex").write_text(valid, encoding="utf-8")
    (tmp_path / "supplement/supplement.tex").write_text(valid, encoding="utf-8")
    report = validator.Report()
    validator.check_cvpr_review_contract(tmp_path, report)
    assert report.errors == []
    assert len(report.warnings) == 2

    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\usepackage{cvpr}\n\\author{Named Author}\n",
        encoding="utf-8",
    )
    report = validator.Report()
    validator.check_cvpr_review_contract(tmp_path, report)
    assert len(report.errors) == 3


def test_vendored_cvpr_kit_matches_pinned_official_release() -> None:
    report = validator.Report()
    validator.check_vendored_cvpr_template(ROOT / "paper", report)
    assert report.errors == []
    assert set(report.facts["cvpr2026_author_kit_sha256"]) == {
        "cvpr.sty",
        "ieeenat_fullname.bst",
    }


def test_tracked_arrow_release_manifest_matches_source_registry() -> None:
    report = validator.Report()
    validator.check_public_release_manifest(ROOT / "paper", report)
    assert report.errors == []
    assert (
        report.facts["arrow_release_manifest_sha256"]
        == "ebe587bee63ed4288f464d1a4872184735a2d0ba0b3ce8cba121973cf1bc49a7"
    )
