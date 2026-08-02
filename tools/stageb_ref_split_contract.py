#!/usr/bin/env python3
"""Single source of truth for deterministic official RefCOCO manifests.

The SHA-256 values name the raw JSONL files emitted by
``tools.eval_refcoco_stageb._build_split_jsonl`` with the repository's locked
phrase sources and ``/media/haoyi/T9/data``.  They are also the hashes embedded
in the completed b58 Ref8 summary and per-example records.
"""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA = "pivot.stageb.official_ref_split_contract/v1"

REF_SPLITS = (
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_val",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_val",
    "refcocog_test",
)

REF_SPLIT_CONTRACT: Mapping[str, Mapping[str, Any]] = {
    "refcoco_val": {
        "rows": 10834,
        "sha256": "ac1ab43019a03dcc65ba3530469b6dcb2ac01be836b795ae5a3b1bdb56b6431d",
    },
    "refcoco_testA": {
        "rows": 5657,
        "sha256": "47278ef1043382235a151cd90d1e6c18c79d30bb71cb4eb7df1932abc622946e",
    },
    "refcoco_testB": {
        "rows": 5095,
        "sha256": "41687648194225a693da5c42c5448eb1a9f4d2f59ca4cd138d4063d818116c8f",
    },
    "refcocop_val": {
        "rows": 10758,
        "sha256": "1eef48a64e7c118b736aa6d383d164ff70af3504285a2cb43a34c02631b5f6de",
    },
    "refcocop_testA": {
        "rows": 5726,
        "sha256": "57a0fb2342f120d49a1174084a7748cb18ff75a7b789bf2ddaf6c8555dce1105",
    },
    "refcocop_testB": {
        "rows": 4889,
        "sha256": "49fe753d28a45cfb47f3d33cf5fbe34a1fda0ae111c7dcd24063c68e2b411d36",
    },
    "refcocog_val": {
        "rows": 4896,
        "sha256": "6a21fccf3d2330aaf72a3ee16cd1863f29470abc3ebfa64d098c04cf7d10e925",
    },
    "refcocog_test": {
        "rows": 9602,
        "sha256": "6c1c9bf2006344167bdce1859578faf83ca594383cc1acac62792c3e6a0f0a1d",
    },
}

REF_SPLIT_MANIFEST_FILES = {
    "refcoco_val": "refcoco_unc_val.jsonl",
    "refcoco_testA": "refcoco_unc_testA.jsonl",
    "refcoco_testB": "refcoco_unc_testB.jsonl",
    "refcocop_val": "refcocoplus_unc_val.jsonl",
    "refcocop_testA": "refcocoplus_unc_testA.jsonl",
    "refcocop_testB": "refcocoplus_unc_testB.jsonl",
    "refcocog_val": "refcocog_umd_val.jsonl",
    "refcocog_test": "refcocog_umd_test.jsonl",
}

CONTRACT = {
    "schema": SCHEMA,
    "generator": "tools.eval_refcoco_stageb._build_split_jsonl",
    "split_order": list(REF_SPLITS),
    "splits": REF_SPLIT_CONTRACT,
    "manifest_files": REF_SPLIT_MANIFEST_FILES,
}


def validate_contract() -> None:
    if tuple(REF_SPLIT_CONTRACT) != REF_SPLITS:
        raise RuntimeError("official Ref split contract order drifted")
    if set(REF_SPLIT_MANIFEST_FILES) != set(REF_SPLITS):
        raise RuntimeError("official Ref split manifest filenames are incomplete")
    filenames = list(REF_SPLIT_MANIFEST_FILES.values())
    if len(set(filenames)) != len(filenames):
        raise RuntimeError("official Ref split manifest filenames are duplicated")
    for split, record in REF_SPLIT_CONTRACT.items():
        rows = record.get("rows")
        sha256 = record.get("sha256")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise RuntimeError(f"official Ref split {split} has invalid rows")
        if not isinstance(sha256, str) or len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise RuntimeError(f"official Ref split {split} has invalid SHA-256")


validate_contract()
