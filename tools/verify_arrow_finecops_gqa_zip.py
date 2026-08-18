#!/usr/bin/env python3
"""Bind the official GQA zip and prove required-image CRC parity."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import file_record, load_json, write_json_atomic


EXPECTED_SIZE = 21_817_965_542
EXPECTED_ETAG = '"5c55f4cb-51473bbe6"'
SOURCE_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"


def _crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def verify(root: Path, archive_path: Path, output_path: Path) -> dict:
    root = root.resolve(strict=True)
    archive = archive_path.resolve(strict=True)
    if archive.stat().st_size != EXPECTED_SIZE:
        raise ValueError(
            f"GQA zip is incomplete: {archive.stat().st_size} != {EXPECTED_SIZE}"
        )
    annotation = load_json(
        root / "raw" / "benchmark" / "test_expression_all_coco_format.json"
    )
    anns = {int(row["image_id"]): row for row in annotation["annotations"]}
    wanted = {
        str(image["file_name"])
        for image in annotation["images"]
        if anns[int(image["id"])].get("negative_cate") != "image"
    }
    if len(wanted) != 4313:
        raise ValueError("FineCops required GQA image count drifted")
    with zipfile.ZipFile(archive) as handle:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in handle.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe GQA zip member: {info.filename}")
            if member.name not in wanted:
                continue
            if member.name in members:
                raise ValueError(f"duplicate GQA zip basename: {member.name}")
            members[member.name] = info
        if set(members) != wanted:
            raise ValueError(f"GQA zip misses {len(wanted - set(members))} required images")
        mismatches = []
        for name in sorted(wanted):
            path = root / "images" / "gqa" / name
            info = members[name]
            if (
                not path.is_file()
                or path.stat().st_size != info.file_size
                or _crc32(path) != info.CRC
            ):
                mismatches.append(name)
        member_count = len(handle.infolist())
    if mismatches:
        raise ValueError(f"{len(mismatches)} extracted GQA images differ from official zip")
    payload = {
        "schema": "arrow.finecops.gqa_zip_verification/v1",
        "source_url": SOURCE_URL,
        "origin_contract": {
            "content_length": EXPECTED_SIZE,
            "etag": EXPECTED_ETAG,
            "accept_ranges": "bytes",
        },
        "archive": file_record(archive),
        "zip_member_count": member_count,
        "required_image_count": len(wanted),
        "required_image_crc_parity": True,
        "mismatch_count": 0,
    }
    write_json_atomic(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/media/haoyi/T9/data/FineCops-Ref/v1")
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("/media/haoyi/T9/data/FineCops-Ref/v1/raw/gqa/images.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/media/haoyi/T9/data/FineCops-Ref/v1/manifests/"
            "official_gqa_zip_verification.json"
        ),
    )
    args = parser.parse_args()
    payload = verify(args.root, args.archive, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
