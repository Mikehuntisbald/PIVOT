#!/usr/bin/env python3
"""Guard the frozen paper launcher behind the dedicated formal Table-D queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_paper_ablation_matrices as paper_launcher  # noqa: E402


ROWS = ("S0", "S1", "S2", "S3", "S2F")
SEEDS = (17, 42, 73)
RUN_IDS = tuple(f"{row}:{seed}" for row in ROWS for seed in SEEDS)


class TableDFormalTrainingError(RuntimeError):
    """The wrapper was invoked outside its exact dedicated queue item."""


def _queue_from_orchestration_root(root: Path) -> Path:
    root = root.expanduser().resolve(strict=False)
    if root.parent.name != "jobs":
        raise TableDFormalTrainingError(
            "formal item orchestration root must be QUEUE_DIR/jobs/INDEX-RUN"
        )
    return root.parent.parent


def _queue_from_job_dir(job_dir: Path) -> Path:
    job_dir = job_dir.expanduser().resolve(strict=True)
    item_root = job_dir.parent
    if item_root.parent.name != "jobs":
        raise TableDFormalTrainingError(
            "formal detached job must remain under QUEUE_DIR/jobs/INDEX-RUN"
        )
    return item_root.parent.parent


def _authorize(
    queue_dir: Path,
    *,
    run_id: str | None = None,
    orchestration_root: Path | None = None,
    job_dir: Path | None = None,
) -> None:
    from tools import run_stageb_table_d_formal_queue as formal_queue

    try:
        formal_queue.authorize_wrapper_operation(
            queue_dir,
            run_id=run_id,
            orchestration_root=orchestration_root,
            job_dir=job_dir,
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        formal_queue.TableDFormalQueueError,
    ) as exc:
        raise TableDFormalTrainingError(str(exc)) from exc


def _list_payload() -> dict[str, object]:
    return {
        "schema": "pivot.stageb.table_d_formal_training_catalog/v1",
        "rows": list(ROWS),
        "seeds": list(SEEDS),
        "run_ids": list(RUN_IDS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("--json", action="store_true")
    detach = subparsers.add_parser("detach")
    detach.add_argument("--orchestration-root", type=Path, required=True)
    detach.add_argument("--run-id", required=True, choices=RUN_IDS)
    for mode in ("status", "reconcile"):
        child = subparsers.add_parser(mode)
        child.add_argument("job_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "list":
            payload = _list_payload()
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("\n".join(RUN_IDS))
            return 0
        if args.mode == "detach":
            queue_dir = _queue_from_orchestration_root(args.orchestration_root)
            _authorize(
                queue_dir,
                run_id=args.run_id,
                orchestration_root=args.orchestration_root,
            )
            return paper_launcher.main(
                [
                    "detach",
                    "--orchestration-root",
                    str(args.orchestration_root),
                    "--run-id",
                    args.run_id,
                ]
            )
        queue_dir = _queue_from_job_dir(args.job_dir)
        _authorize(queue_dir, job_dir=args.job_dir)
        return paper_launcher.main([args.mode, str(args.job_dir)])
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        TableDFormalTrainingError,
    ) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
