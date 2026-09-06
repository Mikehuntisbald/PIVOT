import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "paper/scripts/validate_empirical_pdf.py"
SPEC = importlib.util.spec_from_file_location("validate_empirical_pdf", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

FONTS = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
AAAAAA+NimbusRomNo9L-Regu              Type 1C           Custom           yes yes yes      7  0
BBBBBB+DejaVuSans                     CID TrueType      Identity-H       yes yes yes     79  0
"""
GOOD_TEX = r"""This is XeTeX
Package: silence 2012/07/02 v1.5b Selective filtering of warnings and error messages
\sl@WarningCount=\count276
\sl@ErrorCount=\count279
LaTeX Info: Redefining \GenericError on input line 601.
Package rerunfilecheck Info: File 'paper.out' has not changed.
Output written on paper.xdv (2 pages, 3120 bytes).
"""
GOOD_BIB = """This is BibTeX, Version 0.99d
The top-level auxiliary file: paper.aux
The style file: ieeenat_fullname.bst
Database file #1: references.bib
"""


def test_good_fonts_and_real_log_not_package_info():
    assert len(validator.parse_fonts(FONTS)) == 2
    assert not validator.inspect_log(GOOD_TEX, "tex")["errors"]
    assert validator.inspect_log(GOOD_TEX, "tex")["output_pages"] == 2
    assert not validator.inspect_log(GOOD_BIB, "bibtex")["errors"]


@pytest.mark.parametrize("text,reason", [
    (FONTS.replace("Type 1C", "Type 3"), "Type 3"),
    (FONTS.replace("yes yes yes", "no  yes yes", 1), "unembedded"),
    ("name   type  encoding emb sub uni object ID\n------ ----\n", "no PDF fonts"),
    (FONTS + "unparseable row\n", "unrecognized"),
])
def test_bad_font_reports_fail_closed(text, reason):
    with pytest.raises(ValueError, match=reason):
        validator.parse_fonts(text)


@pytest.mark.parametrize("bad", [
    "! Undefined control sequence.",
    "./paper.tex:17: Undefined control sequence.",
    "LaTeX Error: File 'missing.sty' not found.",
    "LaTeX Warning: Reference 'sec:missing' on page 1 undefined on input line 9.",
    "Package natbib Warning: Citation 'missing'\n(natbib)                undefined on input line 7.",
    "LaTeX Warning: There were undefined references.",
    "LaTeX Warning: Label 'duplicated' multiply defined.",
    "LaTeX Warning: There were multiply-defined labels.",
    "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.",
    "Package rerunfilecheck Warning: File 'paper.out' has changed.",
    "LaTeX Font Warning: Font shape 'T1/foo/m/n' undefined",
    "Missing character: There is no x in font nullfont!",
    r"Overfull \hbox (0.15pt too wide) in paragraph at lines 6--8",
    "Emergency stop.",
])
def test_actual_tex_diagnostics_fail(bad):
    assert validator.inspect_log(GOOD_TEX + bad + "\n", "tex")["errors"]


@pytest.mark.parametrize("bad", [
    "Repeated entry---line 24 of file references.bib",
    'Warning--I didn\'t find a database entry for "unknown"',
    "I couldn't open style file missing.bst",
    "Illegal, another \\bibstyle command---line 3 of file paper.aux",
    "(There were 2 error messages)",
])
def test_bibliography_diagnostics_fail(bad):
    assert validator.inspect_log(GOOD_BIB + bad + "\n", "bibtex")["errors"]


def test_incomplete_logs_and_underfull_warning_scope():
    assert validator.inspect_log("This is XeTeX\n", "tex")["errors"]
    assert validator.inspect_log("This is BibTeX\n", "bibtex")["errors"]
    result = validator.inspect_log(GOOD_TEX + "Underfull \\hbox (badness 1200)\n", "tex")
    assert not result["errors"]
    assert result["warnings"]


def test_reference_only_page_and_mixed_page_count():
    body = ["Substantive body"] * 8
    references = ("CVPR\n#*****\nCVPR\n#*****\n"
                  "CVPR 2027 Submission #*****. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.\n"
                  "References\n394\n[1] A bibliographic entry.\n")
    only = validator.body_page_contract(body + [references])
    assert only["total_pages"] == 9
    assert only["body_pages_conservative"] == 8
    assert only["within_provisional_body_limit"]
    assert only["reference_start_shares_body"] is False
    mixed = validator.body_page_contract(body + ["Final conclusion.\n" + references])
    assert mixed["body_pages_conservative"] == 9
    assert not mixed["within_provisional_body_limit"]
    assert validator.body_page_contract(body + ["No bibliography"])["body_pages_conservative"] == 9
    assert validator.body_page_contract(body + ["References\nA prose heading."])["references_start_page"] is None
    assert validator.body_page_contract(body + [references], limit=7)["within_provisional_body_limit"] is False


def test_page_splitting_keeps_blank_internal_page_and_requires_content():
    assert validator.split_raw_pages("Text\f\fReferences\f") == ["Text", "", "References"]
    with pytest.raises(ValueError, match="empty"):
        validator.split_raw_pages("\f")


def test_metadata_and_anonymity_markers():
    assert validator.parse_pdfinfo("Pages: 2\nEncrypted: no\nJavaScript: no\n")["Pages"] == 2
    for bad in ("Pages: 0\nEncrypted: no\n", "Pages: 2\n", "Pages: 2\nEncrypted: no\nJavaScript: yes\n"):
        with pytest.raises(ValueError):
            validator.parse_pdfinfo(bad)
    assert validator.author_metadata_error("A. Scientist")
    assert validator.author_metadata_error("Anonymous submission by A. Scientist")
    assert not validator.author_metadata_error("")
    assert not validator.author_metadata_error("Anonymous CVPR submission")
    assert not validator.anonymity_markers("https://arxiv.org/abs/2308.16182")
    assert validator.anonymity_markers("/home/researcher/project/data.json")
    assert validator.anonymity_markers("ssh -p 2222 root@example.org")
    assert validator.anonymity_markers("Public Project Name", ["project name"])
    with pytest.raises(ValueError):
        validator.anonymity_markers("anything", [""])


def test_source_binding_is_recursive_and_missing_file_is_not_ignored(tmp_path):
    entry = tmp_path / "main.tex"
    entry.write_text("\\input{piece}\n\\bibliography{refs}\n\\usepackage{cvpr}\n")
    (tmp_path / "piece.tex").write_text("Some content.")
    (tmp_path / "refs.bib").write_text("@article{x,title={X}}")
    (tmp_path / "cvpr.sty").write_text("local style")
    bindings = validator.collect_sources(entry, tmp_path)
    assert {Path(item["path"]).name for item in bindings} == {"main.tex", "piece.tex", "refs.bib", "cvpr.sty"}
    entry.write_text("\\input{missing}")
    with pytest.raises(ValueError, match="unresolved"):
        validator.collect_sources(entry, tmp_path)


def test_missing_external_tool_fails_and_report_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(validator.shutil, "which", lambda name: None)
    args = ["--main", str(tmp_path / "main.pdf"), "--supplement", str(tmp_path / "supp.pdf"),
            "--report", str(tmp_path / "report.json")]
    report = validator.main(args)
    assert report["status"] == "failed"
    assert "missing required external commands" in report["errors"][0]
    assert report["submission_ready"] is False
    assert report["venue_2027_rules_verified"] is False
    assert json.loads((tmp_path / "report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        validator.main(args)


def test_stubbed_end_to_end_pdf_counts_and_log_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(validator.shutil, "which", lambda name: "/synthetic/" + name)
    for stem in ("main", "supp"):
        (tmp_path / f"{stem}.pdf").write_bytes(b"synthetic fixture, external parser is stubbed")
        (tmp_path / f"{stem}.tex").write_text("Synthetic source only.")
        (tmp_path / f"{stem}.log").write_text(GOOD_TEX)
        (tmp_path / f"{stem}.blg").write_text(GOOD_BIB)
    def run(executable, arguments):
        tool = Path(executable).name
        value = {"pdfinfo": "Pages: 2\nEncrypted: no\nJavaScript: no\n",
                 "pdffonts": FONTS,
                 "pdftotext": "Body text\fReferences\n[1] A reference.\f"}[tool]
        return {"stdout": value, "stderr": ""}
    monkeypatch.setattr(validator, "run_tool", run)
    monkeypatch.setattr(validator, "inspect_pdf_objects", lambda path: {
        "available": True, "checked": True, "external_uris": []})
    args = ["--main", str(tmp_path / "main.pdf"), "--supplement", str(tmp_path / "supp.pdf"),
            "--source-root", str(tmp_path), "--build-dir", str(tmp_path),
            "--report", str(tmp_path / "pass.json")]
    result = validator.main(args)
    assert result["status"] == "passed_required_artifact_checks"
    assert result["artifacts"]["main"]["page_contract"]["body_pages_conservative"] == 1
    assert not result["submission_ready"]
    (tmp_path / "main.log").write_text(GOOD_TEX.replace("(2 pages", "(3 pages"))
    result = validator.main(args[:-1] + [str(tmp_path / "mismatch.json")])
    assert result["status"] == "failed"
    assert any("page count differs" in message for message in result["errors"])


@pytest.mark.parametrize("mode", ["attachment", "javascript"])
def test_optional_object_inspection_rejects_active_or_embedded_content(tmp_path, mode):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    if mode == "attachment":
        writer.add_attachment("fixture.txt", b"not extracted")
    else:
        writer.add_js("/* synthetic; never executed */")
    path = tmp_path / "unsafe.pdf"
    with path.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(ValueError, match="embedded|active"):
        validator.inspect_pdf_objects(path)
