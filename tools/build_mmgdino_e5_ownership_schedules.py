#!/usr/bin/env python3
"""Build immutable per-seed R100+C50 schedules for the e5 transfer.

Selection is performed before any owner training or held-out evaluation.  Rows
are ordered by a SHA-256 rank over their stable identities, then written to
small seed-specific JSONL files consumed by the frozen detector extractor.
Shared-128, Shared-Wide, and Isolated always reuse the same selected bytes and
explicit batch order for a given seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.responsibility_isolation_cache import file_sha256
from tools.train_mmgdino_e5_ownership import (
    FORMAL_CONFIDENCE_BATCH_SIZE,
    FORMAL_CONFIDENCE_UPDATES,
    FORMAL_RANK_BATCH_SIZE,
    FORMAL_RANK_UPDATES,
    FORMAL_SEEDS,
    SCHEDULE_SCHEMA,
    validate_schedule,
)


SELECTION_SCHEMA = "arrow.mmgdino_e5_ownership.schedule_selection/v1"
RECEIPT_SCHEMA = "arrow.mmgdino_e5_ownership.schedule_receipt/v1"


class ScheduleBuildError(RuntimeError):
    """Raised when source rows or schedule outputs drift."""


def _strict_load(line: str, *, location: str) -> Mapping[str, Any]:
    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ScheduleBuildError(
                    f"duplicate JSON key {key!r} at {location}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(line, object_pairs_hook=pairs_hook)
    except ScheduleBuildError:
        raise
    except Exception as exc:
        raise ScheduleBuildError(f"invalid JSON at {location}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ScheduleBuildError(f"row at {location} must be an object")
    return value


def _read_jsonl(path: Path) -> list[tuple[str, Mapping[str, Any]]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith("\n") or not raw.strip():
                raise ScheduleBuildError(
                    f"{path}:{line_number} must be nonempty and newline terminated"
                )
            rows.append(
                (
                    raw,
                    _strict_load(raw, location=f"{path}:{line_number}"),
                )
            )
    if not rows:
        raise ScheduleBuildError(f"source is empty: {path}")
    return rows


def _integer(row: Mapping[str, Any], field: str, *, location: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScheduleBuildError(f"{location}.{field} must be nonnegative int")
    return value


def _string(row: Mapping[str, Any], field: str, *, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ScheduleBuildError(f"{location}.{field} must be trimmed string")
    return value


def rank_identity(row: Mapping[str, Any], *, location: str) -> str:
    source = _string(row, "source", location=location)
    image_id = _integer(row, "image_id", location=location)
    ann_id = _integer(row, "ann_id", location=location)
    ref_id = _integer(row, "ref_id", location=location)
    sent_id = _integer(row, "sent_id", location=location)
    instances = row.get("instances")
    if (
        not isinstance(instances, Sequence)
        or isinstance(instances, (str, bytes))
        or len(instances) != 1
        or not isinstance(instances[0], Mapping)
        or instances[0].get("text_is_negative") is not False
    ):
        raise ScheduleBuildError(f"{location} must contain one positive instance")
    _string(instances[0], "positive_phrase", location=f"{location}.instances[0]")
    return f"refcoco:{source}:{image_id}:{ann_id}:{ref_id}:{sent_id}"


def d3_identity(row: Mapping[str, Any], *, location: str) -> str:
    for field in (
        "proposal_covered_verified",
        "visual_verified_negative",
        "traceable_counterfactual_edit",
    ):
        if row.get(field) is not True:
            raise ScheduleBuildError(f"{location}.{field} must be true")
    if row.get("table_b_id") != "D3":
        raise ScheduleBuildError(f"{location}.table_b_id must be D3")
    if row.get("tn_scope") != "proposal_covered_verified":
        raise ScheduleBuildError(f"{location}.tn_scope drifted")
    if row.get("split") != "train":
        raise ScheduleBuildError(f"{location}.split must be train")
    sample_id = _string(row, "sample_id", location=location)
    _string(row, "sent", location=location)
    _string(row, "try_tn", location=location)
    return f"d3:{sample_id}"


def _rank_key(*, seed: int, task: str, identity: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SCHEMA}:{seed}:{task}:{identity}".encode("utf-8")
    ).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _schedule(
    *,
    seed: int,
    rank_ids: Sequence[str],
    pair_ids: Sequence[str],
    rank_sha: str,
    d3_sha: str,
) -> dict[str, Any]:
    rank_cursor = 0
    pair_cursor = 0
    updates = []
    for update_index in range(1, 151):
        if (update_index - 1) % 3 == 1:
            identities = list(
                pair_ids[
                    pair_cursor : pair_cursor + FORMAL_CONFIDENCE_BATCH_SIZE
                ]
            )
            pair_cursor += FORMAL_CONFIDENCE_BATCH_SIZE
            task = "confidence_pair"
        else:
            identities = list(
                rank_ids[rank_cursor : rank_cursor + FORMAL_RANK_BATCH_SIZE]
            )
            rank_cursor += FORMAL_RANK_BATCH_SIZE
            task = "rank"
        updates.append(
            {"update": update_index, "task": task, "identities": identities}
        )
    value = {
        "schema": SCHEDULE_SCHEMA,
        "seed": seed,
        "source": {
            "rank_jsonl_sha256": rank_sha,
            "d3_jsonl_sha256": d3_sha,
        },
        "rank_batch_size": FORMAL_RANK_BATCH_SIZE,
        "confidence_batch_size": FORMAL_CONFIDENCE_BATCH_SIZE,
        "updates": updates,
    }
    return validate_schedule(value)


def build_schedules(
    *,
    rank_jsonl: str | Path,
    rank_jsonl_sha256: str,
    d3_jsonl: str | Path,
    d3_jsonl_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    rank_path = Path(rank_jsonl).resolve(strict=True)
    d3_path = Path(d3_jsonl).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists():
        raise ScheduleBuildError("schedule output directory must not exist")
    for path, expected, name in (
        (rank_path, rank_jsonl_sha256, "rank_jsonl"),
        (d3_path, d3_jsonl_sha256, "d3_jsonl"),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise ScheduleBuildError(
                f"{name} SHA mismatch: expected {expected}, got {actual}"
            )
    rank_rows = _read_jsonl(rank_path)
    d3_rows = _read_jsonl(d3_path)
    rank_by_id: dict[str, str] = {}
    for index, (raw, row) in enumerate(rank_rows):
        identity = rank_identity(row, location=f"rank[{index}]")
        if identity in rank_by_id:
            raise ScheduleBuildError(f"duplicate rank identity {identity!r}")
        rank_by_id[identity] = raw
    d3_by_id: dict[str, str] = {}
    for index, (raw, row) in enumerate(d3_rows):
        identity = d3_identity(row, location=f"d3[{index}]")
        if identity in d3_by_id:
            raise ScheduleBuildError(f"duplicate D3 pair identity {identity!r}")
        d3_by_id[identity] = raw
    required_rank = FORMAL_RANK_UPDATES * FORMAL_RANK_BATCH_SIZE
    required_pairs = FORMAL_CONFIDENCE_UPDATES * FORMAL_CONFIDENCE_BATCH_SIZE
    if len(rank_by_id) < required_rank or len(d3_by_id) < required_pairs:
        raise ScheduleBuildError("source populations are smaller than formal exposure")
    output.mkdir(parents=True)
    seed_outputs = {}
    for seed in FORMAL_SEEDS:
        rank_ids = sorted(
            rank_by_id,
            key=lambda identity: _rank_key(
                seed=seed, task="rank", identity=identity
            ),
        )[:required_rank]
        pair_ids = sorted(
            d3_by_id,
            key=lambda identity: _rank_key(
                seed=seed, task="confidence", identity=identity
            ),
        )[:required_pairs]
        rank_selected = output / f"rank_seed{seed}.jsonl"
        d3_selected = output / f"d3_seed{seed}.jsonl"
        _atomic_text(rank_selected, "".join(rank_by_id[value] for value in rank_ids))
        _atomic_text(d3_selected, "".join(d3_by_id[value] for value in pair_ids))
        schedule = _schedule(
            seed=seed,
            rank_ids=rank_ids,
            pair_ids=pair_ids,
            rank_sha=file_sha256(rank_selected),
            d3_sha=file_sha256(d3_selected),
        )
        schedule_path = output / f"schedule_seed{seed}.json"
        _atomic_text(
            schedule_path,
            json.dumps(schedule, indent=2, sort_keys=True) + "\n",
        )
        seed_outputs[str(seed)] = {
            "rank": _record(rank_selected),
            "d3": _record(d3_selected),
            "schedule": _record(schedule_path),
            "rank_rows": len(rank_ids),
            "confidence_pairs": len(pair_ids),
        }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete_before_owner_training",
        "selection_schema": SELECTION_SCHEMA,
        "sources": {
            "rank": {**_record(rank_path), "rows": len(rank_rows)},
            "d3": {**_record(d3_path), "rows": len(d3_rows)},
        },
        "contract": {
            "seeds": list(FORMAL_SEEDS),
            "rank_updates": FORMAL_RANK_UPDATES,
            "rank_batch_size": FORMAL_RANK_BATCH_SIZE,
            "confidence_updates": FORMAL_CONFIDENCE_UPDATES,
            "confidence_batch_size": FORMAL_CONFIDENCE_BATCH_SIZE,
            "shared_isolated_same_schedule_per_seed": True,
            "checkpoint_or_metric_dependent_selection": False,
        },
        "outputs": seed_outputs,
    }
    receipt_path = output / "schedule_receipt.json"
    _atomic_text(
        receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-jsonl", type=Path, required=True)
    parser.add_argument("--rank-jsonl-sha256", required=True)
    parser.add_argument("--d3-jsonl", type=Path, required=True)
    parser.add_argument("--d3-jsonl-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    receipt = build_schedules(
        rank_jsonl=args.rank_jsonl,
        rank_jsonl_sha256=args.rank_jsonl_sha256,
        d3_jsonl=args.d3_jsonl,
        d3_jsonl_sha256=args.d3_jsonl_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "RECEIPT_SCHEMA",
    "SELECTION_SCHEMA",
    "ScheduleBuildError",
    "build_schedules",
    "d3_identity",
    "rank_identity",
]
