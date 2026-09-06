#!/usr/bin/env python3
"""Inspect explicit empirical PDFs and final build logs without changing them.

Not a submission-readiness certificate. The default eight-page body limit is
provisional, not verified CVPR 2027 policy. Source hashes bind files inspected
now; they do not prove the PDF was built from them. Keep the build receipt too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

PAPER = Path(__file__).resolve().parents[1]
REQUIRED_COMMANDS = ("pdfinfo", "pdffonts", "pdftotext")
FONT_ROW = re.compile(
    r"^(?P<name>\S+)\s+(?P<type>.+?)\s+(?P<encoding>\S+)\s+"
    r"(?P<embedded>yes|no)\s+(?P<subset>yes|no)\s+(?P<unicode>yes|no)\s+"
    r"(?P<object>\d+)\s+(?P<generation>\d+)\s*$"
)
MARKERS = {
    "host_home_path": re.compile(r"/(?:home|Users)/[^\s/]+", re.I),
    "host_mount_path": re.compile(r"/(?:media|mnt)/[^\s]+", re.I),
    "windows_user_path": re.compile(r"[A-Z]:[\\/]Users[\\/]", re.I),
    "file_uri": re.compile(r"\bfile://", re.I),
    "ssh_uri_or_command": re.compile(r"\bssh://|\bssh\s+-[pPiIlL]\s+", re.I),
    "ssh_account": re.compile(r"\b(?:root|ubuntu|haoyi)@[A-Za-z0-9]", re.I),
}


def bound(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty required file: {path}")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def parse_fonts(text: str) -> list[dict]:
    """Parse Poppler rows, not incidental header text."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("name ") or re.fullmatch(r"[-\s]+", line):
            continue
        match = FONT_ROW.fullmatch(line.strip())
        if match is None:
            raise ValueError("unrecognized pdffonts output row")
        row = match.groupdict()
        if row["embedded"] != "yes":
            raise ValueError(f"unembedded PDF font: {row['name']}")
        if re.search(r"\bType\s*3\b", row["type"], re.I):
            raise ValueError(f"Type 3 PDF font: {row['name']}")
        rows.append(row)
    if not rows:
        raise ValueError("no PDF fonts reported")
    return rows


def parse_pdfinfo(text: str) -> dict:
    info = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    if not re.fullmatch(r"[1-9]\d*", info.get("Pages", "")):
        raise ValueError("pdfinfo did not report a positive page count")
    info["Pages"] = int(info["Pages"])
    if info.get("Encrypted", "").lower() != "no":
        raise ValueError("unencrypted PDF status not established")
    if info.get("JavaScript", "").lower() == "yes":
        raise ValueError("PDF contains JavaScript")
    return info


def inspect_log(text: str, kind: str) -> dict:
    """Flag actual diagnostics, not package Info about warning machinery."""
    if kind not in {"tex", "bibtex"}:
        raise ValueError("unknown log kind")
    if not text.strip():
        raise ValueError("empty build log")
    errors, warnings = [], []
    logical = re.sub(r"\n(?:\([^)\n]+\)[ \t]+|[ \t]+)", " ", text)
    if kind == "tex":
        patterns = (
            r"(?m)^!\s*.+",
            r"(?mi)^(?:LaTeX(?: Font)?|Package [^\n]+?) Error:.*",
            r"(?mi)^.+\.(?:tex|sty|cls):\d+:\s*(?:Undefined control sequence|LaTeX Error|Package .+ Error|Missing .*|Extra .*|Emergency stop).*",
            r"(?mi)^(?:Emergency stop|Fatal error|.*==> Fatal error).*",
            r"(?mi)^Overfull \\[hv]box.*",
            r"(?mi)^Missing character:.*",
            r"(?mi)^(?:LaTeX(?: Font)?|Package [^\n]+?) Warning:[^\n]*(?:undefined|multiply[- ]defined|multiply defined|Rerun to get cross-references|has changed|Label\(s\) may have changed)[^\n]*",
        )
        for pattern in patterns:
            errors.extend(m.group(0) for m in re.finditer(pattern, logical))
        for line in text.splitlines():
            if re.match(r"^(?:LaTeX(?: Font)?|Package .+?) Warning:", line) or "Invalid UTF-8" in line or line.startswith("Underfull "):
                if not any(line in error for error in errors):
                    warnings.append(line)
        completion = re.search(r"Output written on .+?\((\d+) pages?[,)]", text)
        if completion is None:
            errors.append("final TeX output-completion/page-count marker missing")
        pages = int(completion.group(1)) if completion else None
    else:
        for line in text.splitlines():
            if re.match(
                r"^(?:Repeated entry|Warning--|I couldn't open |I found no |Illegal,|"
                r"You're missing |You can't pop |[A-Za-z_]+ is an unknown function|"
                r"The style file.*not found|\(There (?:was|were) \d+ (?:error|warning) messages?\))",
                line,
            ):
                errors.append(line)
        if not text.startswith("This is BibTeX") or "The top-level auxiliary file:" not in text or "Database file #" not in text:
            errors.append("expected completed BibTeX log structure missing")
        pages = None
    return {"errors": sorted(set(errors)), "warnings": sorted(set(warnings)), "output_pages": pages}


def split_raw_pages(text: str) -> list[str]:
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not pages or not any(page.strip() for page in pages):
        raise ValueError("PDF text extraction is empty")
    return pages


def body_page_contract(pages: list[str], limit: int = 8) -> dict:
    """Use raw text order; count mixed body/reference pages conservatively.

    If no isolated bibliography heading with reference entries is established,
    count all pages as body. This heuristic still requires visual confirmation.
    """
    if not pages or limit < 1:
        raise ValueError("nonempty pages and a positive body limit required")
    start, shares_body, count = None, None, len(pages)
    for index, text in enumerate(pages, 1):
        match = re.search(r"(?mi)^\s*References\s*$", text)
        if match is None or not re.search(r"(?m)^\s*\[\d+\]\s+\S", text[match.end():]):
            continue
        prefix = []
        for line in text[:match.start()].splitlines():
            line = line.strip()
            if not line or re.fullmatch(r"[\d*#\s]+", line):
                continue
            if re.fullmatch(r"(?:CVPR|ICCV|ECCV)(?:\s+\d{4}\s+Submission\s+.*)?", line, re.I):
                continue
            if re.fullmatch(r"Anonymous(?: (?:CVPR|ICCV|ECCV))? submission", line, re.I):
                continue
            prefix.append(line)
        start, shares_body = index, bool(prefix)
        count = max(1, index if shares_body else index - 1)
        break
    return {"total_pages": len(pages), "body_pages_conservative": count,
            "references_start_page": start, "reference_start_shares_body": shares_body,
            "body_limit": limit, "within_provisional_body_limit": count <= limit,
            "method": "isolated References heading and bibliographic rows; mixed page counted; otherwise all pages",
            "requires_visual_confirmation": True}


def anonymity_markers(text: str, extra=()) -> list[str]:
    found = [name for name, pattern in MARKERS.items() if pattern.search(text)]
    for index, marker in enumerate(extra):
        if not marker:
            raise ValueError("empty forbidden marker")
        if marker.casefold() in text.casefold():
            found.append(f"explicit_forbidden_marker_{index + 1}")
    return sorted(set(found))


def author_metadata_error(author: str) -> bool:
    return bool(author.strip()) and re.fullmatch(
        r"Anonymous(?: (?:(?:CVPR|ICCV|ECCV) )?submission)?", author.strip(), re.I
    ) is None


def inspect_pdf_objects(path: Path) -> dict:
    """Optional deeper check; never execute an action or extract attachments."""
    try:
        from pypdf import PdfReader
        from pypdf.generic import IndirectObject
    except ImportError:
        return {"available": False, "checked": False, "reason": "pypdf not installed"}
    reader = PdfReader(path, strict=True)
    if reader.is_encrypted:
        raise ValueError("encrypted PDF")
    attachments = list(reader.attachments)
    errors = ["embedded PDF attachments"] if attachments else []
    uris, visited, pending, count = set(), set(), [reader.trailer], 0
    forbidden_actions = {"/JavaScript", "/Launch", "/SubmitForm", "/ImportData", "/Rendition", "/GoToR"}
    while pending:
        obj = pending.pop()
        if isinstance(obj, IndirectObject):
            key = (obj.idnum, obj.generation)
            if key in visited:
                continue
            visited.add(key)
            obj = obj.get_object()
        count += 1
        if count > 200000:
            raise ValueError("PDF object inspection bound exceeded")
        if isinstance(obj, dict):
            if any(key in obj for key in ("/JS", "/JavaScript", "/EmbeddedFiles", "/RichMediaContent")) or obj.get("/Type") == "/EmbeddedFile":
                errors.append("active or embedded PDF content")
            if obj.get("/S") in forbidden_actions:
                errors.append("active or remote-file PDF action")
            if obj.get("/S") == "/URI":
                uris.add(str(obj.get("/URI", "")))
            pending.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            pending.extend(obj)
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))
    return {"available": True, "checked": True, "attachments": 0,
            "forbidden_active_actions": 0, "external_uris": sorted(uris),
            "object_visits": count, "scope": "reachable objects, not a general PDF security certificate"}


def collect_sources(entry: Path, root: Path) -> list[dict]:
    """Bind static TeX inputs, figures, bib files and locally vendored styles."""
    pending, seen = [entry], set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        bound(path)
        seen.add(path)
        if path.suffix != ".tex":
            continue
        text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
        dependencies = [(name, ".tex", True) for name in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", text)]
        dependencies += [(name, ".pdf", True) for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}", text)]
        for command, suffix, required in (("bibliography", ".bib", True), ("bibliographystyle", ".bst", False), ("usepackage", ".sty", False)):
            for group in re.findall(r"\\" + command + r"(?:\[[^]]*\])?\s*\{([^}]+)\}", text):
                dependencies += [(name.strip(), suffix, required) for name in group.split(",")]
        for name, suffix, required in dependencies:
            rel = Path(name)
            if not rel.suffix:
                rel = rel.with_suffix(suffix)
            candidates = [path.parent / rel, root / rel]
            existing = next((p for p in candidates if p.is_file()), None)
            if existing is not None:
                pending.append(existing)
            elif required:
                raise ValueError(f"unresolved static source dependency: {name}")
    return [bound(path) for path in sorted(seen)]


def run_tool(executable: str, arguments: list[str]) -> dict:
    result = subprocess.run([executable, *arguments], text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise ValueError(f"{Path(executable).name} failed ({result.returncode}): {result.stderr.strip()[:500]}")
    return {"stdout": result.stdout, "stderr": result.stderr}


def validate(args: argparse.Namespace) -> dict:
    report = {"schema": "arrow.paper.empirical_pdf_validation/v1", "status": "failed",
              "submission_ready": False, "venue_2027_rules_verified": False,
              "distribution_status": "local_artifact_audit_not_anonymous_release",
              "code": bound(Path(__file__)), "errors": [], "warnings": [], "artifacts": {},
              "scope": "explicit PDF/log/source checks, provisional page bound, embedding and bounded anonymity scan",
              "not_verified": ["visual legibility and figure semantics", "scientific claims and statistics",
                               "source-to-PDF compilation provenance", "official target-venue rules",
                               "absence of every identifying cue", "dataset distribution rights"]}
    commands = {name: shutil.which(name) for name in REQUIRED_COMMANDS}
    report["commands"] = commands
    missing = [name for name, path in commands.items() if path is None]
    if missing:
        report["errors"].append("missing required external commands: " + ", ".join(missing))
        return report
    for label in ("main", "supplement"):
        artifact = report["artifacts"][label] = {}
        try:
            pdf = getattr(args, label).resolve()
            build_dir = args.build_dir.resolve() if args.build_dir else pdf.parent
            stem = getattr(args, label + "_stem") or pdf.stem
            log = getattr(args, label + "_log") or build_dir / (stem + ".log")
            blg = getattr(args, label + "_blg") or build_dir / (stem + ".blg")
            entry = getattr(args, label + "_source") or args.source_root / (stem + ".tex")
            artifact["pdf"], artifact["tex_log"], artifact["bibtex_log"] = bound(pdf), bound(log), bound(blg)
            artifact["sources"] = collect_sources(entry, args.source_root)
            for kind, path in (("tex", log), ("bibtex", blg)):
                parsed = inspect_log(path.read_text(errors="replace"), kind)
                artifact[kind + "_diagnostics"] = parsed
                report["errors"].extend(label + ": " + error for error in parsed["errors"])
                report["warnings"].extend(label + ": " + warning for warning in parsed["warnings"])
            info_run = run_tool(commands["pdfinfo"], [str(pdf)])
            info = artifact["pdfinfo"] = parse_pdfinfo(info_run["stdout"])
            fonts_run = run_tool(commands["pdffonts"], [str(pdf)])
            artifact["fonts"] = parse_fonts(fonts_run["stdout"])
            text_run = run_tool(commands["pdftotext"], ["-raw", str(pdf), "-"])
            pages = split_raw_pages(text_run["stdout"])
            if len(pages) != info["Pages"]:
                raise ValueError("extracted page count differs from pdfinfo")
            logged = artifact["tex_diagnostics"]["output_pages"]
            if logged is not None and logged != info["Pages"]:
                raise ValueError("TeX final output page count differs from PDF")
            if author_metadata_error(info.get("Author", "")):
                raise ValueError("non-anonymous PDF Author metadata")
            marker_hits = anonymity_markers(text_run["stdout"] + "\n" + info_run["stdout"], args.forbid_marker)
            if marker_hits:
                raise ValueError("anonymity markers found: " + ", ".join(marker_hits))
            artifact["text_anonymity_markers"] = []
            artifact["pdf_objects"] = inspect_pdf_objects(pdf)
            if not artifact["pdf_objects"]["available"]:
                if args.require_pypdf:
                    raise ValueError("pypdf is required but unavailable")
                report["warnings"].append(label + ": optional PDF object/attachment check unavailable")
            elif anonymity_markers("\n".join(artifact["pdf_objects"]["external_uris"]), args.forbid_marker):
                raise ValueError("anonymity markers in external PDF URI")
            if label == "main":
                contract = artifact["page_contract"] = body_page_contract(pages, args.max_main_body_pages)
                if not contract["within_provisional_body_limit"]:
                    raise ValueError("main exceeds provisional body-page bound")
            elif args.max_supplement_pages and len(pages) > args.max_supplement_pages:
                raise ValueError("supplement exceeds configured page bound")
            for command, result in (("pdfinfo", info_run), ("pdffonts", fonts_run), ("pdftotext", text_run)):
                if result["stderr"].strip():
                    report["warnings"].append(label + " " + command + ": " + result["stderr"].strip())
        except Exception as error:
            report["errors"].append(label + ": " + str(error))
    if not report["errors"]:
        report["status"] = "passed_required_artifact_checks"
    return report


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("main", "supplement"):
        parser.add_argument("--" + label, type=Path, required=True)
        for suffix in ("log", "blg", "source"):
            parser.add_argument("--" + label + "-" + suffix, type=Path)
        parser.add_argument("--" + label + "-stem")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=PAPER)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-main-body-pages", type=int, default=8)
    parser.add_argument("--max-supplement-pages", type=int)
    parser.add_argument("--forbid-marker", action="append", default=[])
    parser.add_argument("--require-pypdf", action="store_true")
    args = parser.parse_args(argv)
    if args.max_main_body_pages < 1 or (args.max_supplement_pages is not None and args.max_supplement_pages < 1):
        raise ValueError("page limits must be positive")
    if args.report.exists():
        raise FileExistsError("report exists; choose a new report path")
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "errors": report["errors"], "report": bound(args.report)}))
    return report


if __name__ == "__main__":
    sys.exit(0 if main()["status"] == "passed_required_artifact_checks" else 1)
