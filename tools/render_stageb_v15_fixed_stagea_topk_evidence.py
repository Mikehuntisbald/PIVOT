#!/usr/bin/env python3
"""Render inspectable evidence for every exact Stage-A Top-K candidate.

The output review manifest deliberately contains pending judgments.  It binds
the exact fields consumed by the CPU pair builder, but cannot be mistaken for a
completed judgment JSONL because answer/confidence/hash remain null and the
audit states ``review_complete=false``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_stageb_v15_fixed_stagea_topk_exact_pairs import (  # noqa: E402
    JUDGE_CONTRACT_SCHEMA,
    _validate_extraction_audit,
    _validate_extraction_row,
)
from util.stageb_exact_topk_contract import (  # noqa: E402
    EXACT_TOPK_JUDGMENT_SCHEMA,
    EXACT_TOPK_PROTOCOL,
    ExactTopKContractError,
    canonical_sha256,
    file_record,
    is_sha256,
    sha256_file,
)


REVIEW_AUDIT_SCHEMA = "stage-b-v15-fixed-stagea-topk-review-manifest-audit-v1"
EVIDENCE_SHARD_SCHEMA = "stage-b-v15-fixed-stagea-topk-evidence-shard-v1"
EVIDENCE_POLICY = {
    "schema": "stage-b-v15-fixed-stagea-topk-evidence-asset-policy-v1",
    "box_color_rgb": [232, 52, 52],
    "box_width_px": 4,
    "background_rgb": [245, 245, 242],
    "panel_background_rgb": [255, 255, 255],
    "text_rgb": [20, 20, 20],
    "full_panel_max_wh": [760, 540],
    "detail_panel_wh": [360, 176],
    "canvas_wh": [1200, 760],
    "context_scale": 2.0,
    "rasterization": "floor_xy_min_ceil_xy_max_clamp_min_one_pixel",
    "resampling": "PIL.Image.Resampling.LANCZOS",
    "png": {"format": "PNG", "compress_level": 9, "optimize": False},
    "text_encoding": "ascii_backslashreplace_for_render_only",
}


class EvidenceError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                dict(value),
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise EvidenceError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not an object: {path}")
    return value


def _iter_jsonl(path: Path, *, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise EvidenceError(f"missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise EvidenceError(f"blank row at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise EvidenceError(f"non-object row at {path}:{line_number}")
            yield line_number, value


def _hashed_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def make_judge_contract(
    *, judge_type: str, prompt_template: Path, min_no_confidence: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_record = file_record(prompt_template)
    if judge_type not in {"human", "model", "hybrid"}:
        raise EvidenceError("judge type must be human, model, or hybrid")
    confidence = float(min_no_confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise EvidenceError("minimum no confidence must be in [0,1]")
    contract = _hashed_contract(
        {
            "schema": JUDGE_CONTRACT_SCHEMA,
            "judge_type": judge_type,
            "prompt_template_sha256": prompt_record["sha256"],
            "evidence_asset_policy_sha256": canonical_sha256(EVIDENCE_POLICY),
            "min_no_confidence": confidence,
        }
    )
    return contract, prompt_record


def _cxcywh_to_original(
    bbox: Sequence[float], trace: Mapping[str, Any]
) -> list[float]:
    try:
        cx, cy, width, height = [float(value) for value in bbox]
        original_h, original_w = [int(value) for value in trace["original_hw"]]
        output_h, output_w = [int(value) for value in trace["output_hw"]]
        scale_x, scale_y = [float(value) for value in trace["scale_xy"]]
        offset_x, offset_y = [float(value) for value in trace["offset_xy"]]
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError("malformed candidate box/transform trace") from error
    if min(original_h, original_w, output_h, output_w) <= 0 or min(scale_x, scale_y) <= 0:
        raise EvidenceError("transform trace has non-positive geometry")
    transformed = [
        (cx - width / 2.0) * output_w,
        (cy - height / 2.0) * output_h,
        (cx + width / 2.0) * output_w,
        (cy + height / 2.0) * output_h,
    ]
    original = [
        (transformed[0] - offset_x) / scale_x,
        (transformed[1] - offset_y) / scale_y,
        (transformed[2] - offset_x) / scale_x,
        (transformed[3] - offset_y) / scale_y,
    ]
    return [
        min(float(original_w), max(0.0, original[0])),
        min(float(original_h), max(0.0, original[1])),
        min(float(original_w), max(0.0, original[2])),
        min(float(original_h), max(0.0, original[3])),
    ]


def _raster_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in box]
    left = min(max(0, math.floor(x0)), max(0, width - 1))
    top = min(max(0, math.floor(y0)), max(0, height - 1))
    right = min(width, max(left + 1, math.ceil(x1)))
    bottom = min(height, max(top + 1, math.ceil(y1)))
    return int(left), int(top), int(right), int(bottom)


def _context_box(
    box: Sequence[int], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = [int(value) for value in box]
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    context_w = max(1.0, (right - left) * float(EVIDENCE_POLICY["context_scale"]))
    context_h = max(1.0, (bottom - top) * float(EVIDENCE_POLICY["context_scale"]))
    return _raster_box(
        [
            cx - context_w / 2.0,
            cy - context_h / 2.0,
            cx + context_w / 2.0,
            cy + context_h / 2.0,
        ],
        width,
        height,
    )


def _draw_box(image: Image.Image, box: Sequence[int]) -> None:
    left, top, right, bottom = [int(value) for value in box]
    ImageDraw.Draw(image).rectangle(
        [left, top, max(left, right - 1), max(top, bottom - 1)],
        outline=tuple(EVIDENCE_POLICY["box_color_rgb"]),
        width=int(EVIDENCE_POLICY["box_width_px"]),
    )


def _fit(image: Image.Image, max_wh: Sequence[int]) -> Image.Image:
    max_w, max_h = [int(value) for value in max_wh]
    ratio = min(max_w / image.width, max_h / image.height)
    size = (
        max(1, int(round(image.width * ratio))),
        max(1, int(round(image.height * ratio))),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _ascii_text(value: Any) -> str:
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _save_png_atomic(image: Image.Image, path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        image.save(temporary, **EVIDENCE_POLICY["png"])
        record = {
            "path": str(path),
            "size_bytes": int(temporary.stat().st_size),
            "sha256": sha256_file(temporary),
            "width": int(image.width),
            "height": int(image.height),
        }
        if path.exists():
            existing = file_record(path)
            if existing["sha256"] != record["sha256"]:
                raise EvidenceError(f"existing evidence asset drifted: {path}")
            return {**record, "size_bytes": existing["size_bytes"]}
        os.replace(temporary, path)
        return record
    finally:
        temporary.unlink(missing_ok=True)


def render_candidate_assets(
    extraction: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    assets_dir: Path,
) -> dict[str, Any]:
    sample_id = str(extraction["sample_id"])
    rank = int(candidate["rank"])
    sample_key = canonical_sha256(sample_id)[:20]
    candidate_key = str(candidate["candidate_sha256"])[:20]
    root = assets_dir.expanduser().resolve() / sample_key / f"rank-{rank:02d}-{candidate_key}"
    with Image.open(extraction["image"]["path"]) as handle:
        image = handle.convert("RGB")
    with Image.open(extraction["support"]["path"]) as handle:
        support = handle.convert("RGB")
    width, height = image.size
    original_box = _cxcywh_to_original(
        candidate["bbox_cxcywh_normalized"], extraction["transform_trace"]
    )
    raster = _raster_box(original_box, width, height)
    context = _context_box(raster, width, height)

    full = image.copy()
    _draw_box(full, raster)
    tight = image.crop(raster)
    context_image = image.crop(context)
    _draw_box(
        context_image,
        [
            raster[0] - context[0],
            raster[1] - context[1],
            raster[2] - context[0],
            raster[3] - context[1],
        ],
    )
    components = {
        "full_boxed": _save_png_atomic(full, root / "full_boxed.png"),
        "context_boxed": _save_png_atomic(context_image, root / "context_boxed.png"),
        "tight": _save_png_atomic(tight, root / "tight.png"),
        "support": _save_png_atomic(support, root / "support.png"),
    }

    canvas_w, canvas_h = [int(value) for value in EVIDENCE_POLICY["canvas_wh"]]
    canvas = Image.new("RGB", (canvas_w, canvas_h), tuple(EVIDENCE_POLICY["background_rgb"]))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text_lines = [
        f"sample={_ascii_text(sample_id)} rank={rank} query={int(candidate['query_index'])}",
        f"patch_logit={float(candidate['patch_logit']):.7f} class={int(extraction['source_pair']['class_id'])}",
        f"positive: {_ascii_text(extraction['source_pair']['sent'])}",
        f"TN: {_ascii_text(extraction['source_pair']['try_tn'])}",
        f"canonical: {_ascii_text(extraction.get('canonical_caption', ''))}",
    ]
    for index, line in enumerate(text_lines):
        draw.text((18, 12 + index * 17), line[:185], fill=tuple(EVIDENCE_POLICY["text_rgb"]), font=font)

    full_panel = _fit(full, EVIDENCE_POLICY["full_panel_max_wh"])
    full_x, full_y = 18, 112
    canvas.paste(full_panel, (full_x, full_y))
    draw.text((full_x, full_y - 16), "full image / candidate boxed", fill=(20, 20, 20), font=font)
    detail_x = 816
    detail_y = 112
    for label, panel in (
        ("context 2x", context_image),
        ("tight candidate", tight),
        ("fixed support patch", support),
    ):
        fitted = _fit(panel, EVIDENCE_POLICY["detail_panel_wh"])
        canvas.paste(fitted, (detail_x, detail_y))
        draw.text((detail_x, detail_y - 15), label, fill=(20, 20, 20), font=font)
        detail_y += 210
    evidence = _save_png_atomic(canvas, root / "evidence.png")
    bundle_payload = {
        "policy_sha256": canonical_sha256(EVIDENCE_POLICY),
        "raster_bbox_xyxy_original": list(raster),
        "context_bbox_xyxy_original": list(context),
        "components": components,
        "evidence": evidence,
        "pillow_version": PILLOW_VERSION,
    }
    return {
        **bundle_payload,
        "asset_bundle_sha256": canonical_sha256(bundle_payload),
    }


def _load_extractions(
    extraction_path: Path, extraction_audit_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    audit, contract = _validate_extraction_audit(
        extraction_audit_path.expanduser().resolve(),
        extraction_path.expanduser().resolve(),
    )
    rows: list[dict[str, Any]] = []
    support_cache: dict[Path, str] = {}
    seen: set[str] = set()
    for line_number, raw in _iter_jsonl(extraction_path, label="extractions"):
        parsed = _validate_extraction_row(
            raw,
            line_number=line_number,
            exact_contract=contract,
            support_hash_cache=support_cache,
        )
        if parsed["sample_id"] in seen:
            raise EvidenceError(f"duplicate extraction sample: {parsed['sample_id']}")
        seen.add(parsed["sample_id"])
        rows.append(
            {
                **parsed,
                "raw": raw,
                "transform_trace": dict(raw["query_transform_trace"]),
                "canonical_caption": str(raw.get("canonical_stage_a_caption", "")),
            }
        )
    if len(rows) != int(audit["rows"]):
        raise EvidenceError("extraction audit row count drifted")
    return audit, contract, rows


def _pending_review_row(
    extraction: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    assets: Mapping[str, Any],
    judge_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": EXACT_TOPK_JUDGMENT_SCHEMA,
        "protocol": EXACT_TOPK_PROTOCOL,
        "status": "pending",
        "sample_id": extraction["sample_id"],
        "candidate_rank": int(candidate["rank"]),
        "query_index": int(candidate["query_index"]),
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_set_sha256": extraction["candidate_set_sha256"],
        "extraction_row_sha256": extraction["extraction_row_sha256"],
        "evidence_sha256": assets["evidence"]["sha256"],
        "judge_contract_sha256": judge_contract_sha256,
        "answer": None,
        "confidence": None,
        "judgment_sha256": None,
        "positive_expression": extraction["source_pair"]["sent"],
        "negative_expression": extraction["source_pair"]["try_tn"],
        "canonical_stage_a_caption": extraction["canonical_caption"],
        "candidate_bbox_cxcywh_normalized": candidate["bbox_cxcywh_normalized"],
        "candidate_patch_logit": candidate["patch_logit"],
        "evidence_path": assets["evidence"]["path"],
        "evidence_asset_bundle": dict(assets),
    }


def _shard_path(work_dir: Path, sample_id: str, rank: int) -> Path:
    sample_key = canonical_sha256(sample_id)[:20]
    return work_dir / "rows" / sample_key / f"rank-{rank:02d}.json"


def _validate_pending_row(
    row: Mapping[str, Any],
    extraction: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    judge_contract_sha256: str,
) -> None:
    expected = {
        "schema": EXACT_TOPK_JUDGMENT_SCHEMA,
        "protocol": EXACT_TOPK_PROTOCOL,
        "status": "pending",
        "sample_id": extraction["sample_id"],
        "candidate_rank": int(candidate["rank"]),
        "query_index": int(candidate["query_index"]),
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_set_sha256": extraction["candidate_set_sha256"],
        "extraction_row_sha256": extraction["extraction_row_sha256"],
        "judge_contract_sha256": judge_contract_sha256,
        "answer": None,
        "confidence": None,
        "judgment_sha256": None,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise EvidenceError(f"pending review binding {key} drifted")
    convenience_bindings = {
        "positive_expression": extraction["source_pair"]["sent"],
        "negative_expression": extraction["source_pair"]["try_tn"],
        "canonical_stage_a_caption": extraction["canonical_caption"],
        "candidate_bbox_cxcywh_normalized": candidate[
            "bbox_cxcywh_normalized"
        ],
        "candidate_patch_logit": candidate["patch_logit"],
    }
    for key, value in convenience_bindings.items():
        if row.get(key) != value:
            raise EvidenceError(f"pending review display binding {key} drifted")
    evidence = row.get("evidence_asset_bundle")
    if not isinstance(evidence, Mapping):
        raise EvidenceError("pending review has no evidence asset bundle")
    if evidence.get("policy_sha256") != canonical_sha256(EVIDENCE_POLICY):
        raise EvidenceError("pending review evidence policy drifted")
    bundle_payload = {
        key: value
        for key, value in evidence.items()
        if key != "asset_bundle_sha256"
    }
    if evidence.get("asset_bundle_sha256") != canonical_sha256(bundle_payload):
        raise EvidenceError("pending review asset bundle hash drifted")
    for name in ("full_boxed", "context_boxed", "tight", "support"):
        record = evidence.get("components", {}).get(name)
        if not isinstance(record, Mapping):
            raise EvidenceError(f"pending review {name} asset record is missing")
        current = file_record(Path(record.get("path", "")))
        if any(
            current.get(key) != record.get(key)
            for key in ("path", "size_bytes", "sha256")
        ):
            raise EvidenceError(f"pending review {name} asset drifted")
        try:
            with Image.open(current["path"]) as handle:
                observed_size = [int(handle.width), int(handle.height)]
                handle.verify()
        except (OSError, ValueError) as error:
            raise EvidenceError(
                f"pending review {name} asset is not a valid image"
            ) from error
        if observed_size != [record.get("width"), record.get("height")]:
            raise EvidenceError(f"pending review {name} dimensions drifted")
    evidence_record = evidence.get("evidence")
    if not isinstance(evidence_record, Mapping):
        raise EvidenceError("pending review composite evidence record is missing")
    current_evidence = file_record(Path(evidence_record.get("path", "")))
    if any(
        current_evidence.get(key) != evidence_record.get(key)
        for key in ("path", "size_bytes", "sha256")
    ):
        raise EvidenceError("pending review composite evidence drifted")
    try:
        with Image.open(current_evidence["path"]) as handle:
            observed_size = [int(handle.width), int(handle.height)]
            handle.verify()
    except (OSError, ValueError) as error:
        raise EvidenceError(
            "pending review composite evidence is not a valid image"
        ) from error
    if observed_size != [
        evidence_record.get("width"),
        evidence_record.get("height"),
    ]:
        raise EvidenceError("pending review composite evidence dimensions drifted")
    if row.get("evidence_sha256") != evidence_record["sha256"]:
        raise EvidenceError("pending review evidence_sha256 drifted")
    if row.get("evidence_path") != evidence_record["path"]:
        raise EvidenceError("pending review evidence_path drifted")


def _validate_hashed_mapping(
    value: Any,
    *,
    label: str,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} is missing")
    result = dict(value)
    observed = result.pop("sha256", None)
    if observed != canonical_sha256(result):
        raise EvidenceError(f"{label} canonical hash drifted")
    if expected_schema is not None and result.get("schema") != expected_schema:
        raise EvidenceError(f"{label} schema drifted")
    return {**result, "sha256": observed}


def validate_review_bundle(
    *,
    extraction_path: Path,
    extraction_audit_path: Path,
    review_path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int]:
    """Validate the complete pending evidence bundle without trusting CLI values."""

    extraction_path = extraction_path.expanduser().resolve()
    extraction_audit_path = extraction_audit_path.expanduser().resolve()
    review_path = review_path.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    extraction_audit, contract, extractions = _load_extractions(
        extraction_path, extraction_audit_path
    )
    audit = _read_json(audit_path, label="review audit")
    if (
        audit.get("schema") != REVIEW_AUDIT_SCHEMA
        or audit.get("protocol") != EXACT_TOPK_PROTOCOL
        or audit.get("complete") is not True
        or audit.get("evidence_complete") is not True
        or audit.get("review_complete") is not False
        or audit.get("answers_present") != 0
    ):
        raise EvidenceError("review audit is not a completed pending-evidence run")
    if audit.get("exact_contract") != contract:
        raise EvidenceError("review audit exact contract drifted")
    expected = len(extractions) * int(contract["candidate_topk"])
    scalar_bindings = {
        "rows": expected,
        "source_rows": len(extractions),
        "candidate_topk": int(contract["candidate_topk"]),
        "extraction_plan_sha256": extraction_audit.get("plan_sha256"),
    }
    for key, value in scalar_bindings.items():
        if audit.get(key) != value:
            raise EvidenceError(f"review audit {key} drifted")
    if audit.get("extractions") != file_record(
        extraction_path, rows=len(extractions)
    ):
        raise EvidenceError("review audit extraction record drifted")
    if audit.get("extraction_audit") != file_record(extraction_audit_path):
        raise EvidenceError("review audit extraction-audit record drifted")
    current_review = file_record(review_path, rows=expected)
    if audit.get("review_manifest") != current_review:
        raise EvidenceError("review manifest file record drifted")
    judge_contract = _validate_hashed_mapping(
        audit.get("judge_contract"),
        label="judge contract",
        expected_schema=JUDGE_CONTRACT_SCHEMA,
    )
    if judge_contract.get("judge_type") not in {"human", "model", "hybrid"}:
        raise EvidenceError("judge contract judge_type drifted")
    for key in ("prompt_template_sha256", "evidence_asset_policy_sha256"):
        value = judge_contract.get(key)
        if not is_sha256(value):
            raise EvidenceError(f"judge contract {key} is invalid")
    try:
        minimum = float(judge_contract.get("min_no_confidence"))
    except (TypeError, ValueError) as error:
        raise EvidenceError("judge contract min_no_confidence is invalid") from error
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise EvidenceError("judge contract min_no_confidence is outside [0,1]")
    if judge_contract["evidence_asset_policy_sha256"] != canonical_sha256(
        EVIDENCE_POLICY
    ):
        raise EvidenceError("judge contract evidence policy drifted")
    prompt = audit.get("prompt_template")
    if not isinstance(prompt, Mapping) or not isinstance(prompt.get("path"), str):
        raise EvidenceError("review audit prompt template record is missing")
    if dict(prompt) != file_record(Path(prompt["path"])):
        raise EvidenceError("review audit prompt template record drifted")
    if prompt.get("sha256") != judge_contract["prompt_template_sha256"]:
        raise EvidenceError("judge contract prompt hash drifted")
    evidence_policy = _validate_hashed_mapping(
        audit.get("evidence_asset_policy"), label="evidence asset policy"
    )
    if evidence_policy != _hashed_contract(EVIDENCE_POLICY):
        raise EvidenceError("review audit evidence asset policy drifted")
    expected_claims = {
        "evidence_assets_rendered": True,
        "candidate_judgments_completed": False,
        "all_stagea_topk_candidates_reviewed": False,
        "image_global_semantic_absence_proven": False,
    }
    if audit.get("claims") != expected_claims:
        raise EvidenceError("review audit claims drifted")

    extraction_by_sample = {row["sample_id"]: row for row in extractions}
    seen: set[tuple[str, int]] = set()
    pending: list[dict[str, Any]] = []
    for line_number, row in _iter_jsonl(review_path, label="review manifest"):
        sample_id = row.get("sample_id")
        try:
            rank = int(row.get("candidate_rank"))
        except (TypeError, ValueError) as error:
            raise EvidenceError(f"invalid review key at line {line_number}") from error
        extraction = extraction_by_sample.get(sample_id)
        if extraction is None or rank < 0 or rank >= len(extraction["candidates"]):
            raise EvidenceError(f"orphan review row at line {line_number}")
        key = (str(sample_id), rank)
        if key in seen:
            raise EvidenceError(f"duplicate review row: {key}")
        seen.add(key)
        _validate_pending_row(
            row,
            extraction,
            extraction["candidates"][rank],
            judge_contract_sha256=judge_contract["sha256"],
        )
        pending.append(dict(row))
    expected_keys = {
        (str(extraction["sample_id"]), int(candidate["rank"]))
        for extraction in extractions
        for candidate in extraction["candidates"]
    }
    if seen != expected_keys:
        raise EvidenceError("pending review coverage is incomplete")
    return audit, judge_contract, pending, len(extractions)


def render(args: argparse.Namespace) -> dict[str, Any]:
    extraction_path = Path(args.extractions).expanduser().resolve()
    extraction_audit_path = Path(args.extraction_audit).expanduser().resolve()
    review_path = Path(args.review_manifest).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if review_path.exists() or audit_path.exists():
        if review_path.is_file() and audit_path.is_file():
            return verify(args)
        raise EvidenceError("partial finalized review manifest/audit exists; refuse overwrite")
    extraction_audit, contract, extractions = _load_extractions(
        extraction_path, extraction_audit_path
    )
    judge_contract, prompt_record = make_judge_contract(
        judge_type=args.judge_type,
        prompt_template=Path(args.prompt_template).expanduser().resolve(),
        min_no_confidence=float(args.min_no_confidence),
    )
    expected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for extraction in extractions:
        for candidate in extraction["candidates"]:
            expected.append((extraction, candidate))
    expected_shards = {
        _shard_path(work_dir, str(extraction["sample_id"]), int(candidate["rank"]))
        for extraction, candidate in expected
    }
    rows_root = work_dir / "rows"
    orphan = (
        set(rows_root.glob("**/*.json")).difference(expected_shards)
        if rows_root.exists()
        else set()
    )
    if orphan:
        raise EvidenceError(f"found {len(orphan)} orphan evidence shards")
    completed: list[dict[str, Any]] = []
    for index, (extraction, candidate) in enumerate(expected, 1):
        shard_path = _shard_path(
            work_dir, str(extraction["sample_id"]), int(candidate["rank"])
        )
        if shard_path.is_file():
            shard = _read_json(shard_path, label="evidence shard")
            if shard.get("schema") != EVIDENCE_SHARD_SCHEMA:
                raise EvidenceError("evidence shard schema drifted")
            row = shard.get("review_row")
            if not isinstance(row, Mapping):
                raise EvidenceError("evidence shard has no review row")
            if shard.get("review_row_sha256") != canonical_sha256(row):
                raise EvidenceError("evidence shard review-row hash drifted")
            _validate_pending_row(
                row,
                extraction,
                candidate,
                judge_contract_sha256=judge_contract["sha256"],
            )
            completed.append(dict(row))
            continue
        assets = render_candidate_assets(
            extraction,
            candidate,
            assets_dir=assets_dir,
        )
        row = _pending_review_row(
            extraction,
            candidate,
            assets=assets,
            judge_contract_sha256=judge_contract["sha256"],
        )
        _validate_pending_row(
            row,
            extraction,
            candidate,
            judge_contract_sha256=judge_contract["sha256"],
        )
        _atomic_json(
            shard_path,
            {
                "schema": EVIDENCE_SHARD_SCHEMA,
                "review_row": row,
                "review_row_sha256": canonical_sha256(row),
            },
        )
        completed.append(row)
        if int(args.log_every) > 0 and index % int(args.log_every) == 0:
            print(f"[INFO] rendered {index}/{len(expected)} candidates", flush=True)
    _atomic_jsonl(review_path, completed)
    audit = {
        "schema": REVIEW_AUDIT_SCHEMA,
        "protocol": EXACT_TOPK_PROTOCOL,
        "complete": True,
        "evidence_complete": True,
        "review_complete": False,
        "answers_present": 0,
        "rows": len(completed),
        "source_rows": len(extractions),
        "candidate_topk": int(contract["candidate_topk"]),
        "exact_contract": contract,
        "extractions": file_record(extraction_path, rows=len(extractions)),
        "extraction_audit": file_record(extraction_audit_path),
        "review_manifest": file_record(review_path, rows=len(completed)),
        "judge_contract": judge_contract,
        "prompt_template": prompt_record,
        "evidence_asset_policy": _hashed_contract(EVIDENCE_POLICY),
        "extraction_plan_sha256": extraction_audit.get("plan_sha256"),
        "claims": {
            "evidence_assets_rendered": True,
            "candidate_judgments_completed": False,
            "all_stagea_topk_candidates_reviewed": False,
            "image_global_semantic_absence_proven": False,
        },
    }
    _atomic_json(audit_path, audit)
    return audit


def verify(args: argparse.Namespace) -> dict[str, Any]:
    extraction_path = Path(args.extractions).expanduser().resolve()
    extraction_audit_path = Path(args.extraction_audit).expanduser().resolve()
    review_path = Path(args.review_manifest).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    judge_contract, prompt_record = make_judge_contract(
        judge_type=args.judge_type,
        prompt_template=Path(args.prompt_template).expanduser().resolve(),
        min_no_confidence=float(args.min_no_confidence),
    )
    audit, audited_judge_contract, pending, _source_rows = validate_review_bundle(
        extraction_path=extraction_path,
        extraction_audit_path=extraction_audit_path,
        review_path=review_path,
        audit_path=audit_path,
    )
    if audited_judge_contract != judge_contract or audit.get("prompt_template") != prompt_record:
        raise EvidenceError("review judge/prompt contract drifted")
    current_review = file_record(review_path, rows=len(pending))
    return {
        "schema": REVIEW_AUDIT_SCHEMA,
        "verified": True,
        "review_complete": False,
        "rows": len(pending),
        "review_manifest": current_review,
        "audit": file_record(audit_path),
    }


def dry_run(args: argparse.Namespace, *, list_rows: bool) -> dict[str, Any]:
    extraction_path = Path(args.extractions).expanduser().resolve()
    extraction_audit_path = Path(args.extraction_audit).expanduser().resolve()
    _audit, contract, extractions = _load_extractions(
        extraction_path, extraction_audit_path
    )
    judge_contract, prompt_record = make_judge_contract(
        judge_type=args.judge_type,
        prompt_template=Path(args.prompt_template).expanduser().resolve(),
        min_no_confidence=float(args.min_no_confidence),
    )
    result = {
        "schema": REVIEW_AUDIT_SCHEMA,
        "kind": "list_no_assets_written" if list_rows else "dry_run_no_assets_written",
        "source_rows": len(extractions),
        "candidate_topk": int(contract["candidate_topk"]),
        "planned_review_rows": len(extractions) * int(contract["candidate_topk"]),
        "exact_contract": contract,
        "judge_contract": judge_contract,
        "prompt_template": prompt_record,
        "evidence_asset_policy": _hashed_contract(EVIDENCE_POLICY),
        "review_complete": False,
    }
    if list_rows:
        result["planned_keys"] = [
            {
                "sample_id": extraction["sample_id"],
                "candidate_rank": int(candidate["rank"]),
                "query_index": int(candidate["query_index"]),
                "candidate_sha256": candidate["candidate_sha256"],
            }
            for extraction in extractions
            for candidate in extraction["candidates"]
        ]
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--extraction-audit", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--judge-type", choices=("human", "model", "hybrid"), required=True)
    parser.add_argument("--min-no-confidence", type=float, default=0.90)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=500)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.verify_only:
            result = verify(args)
        elif args.dry_run or args.list:
            result = dry_run(args, list_rows=bool(args.list))
        else:
            result = render(args)
    except (EvidenceError, ExactTopKContractError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
