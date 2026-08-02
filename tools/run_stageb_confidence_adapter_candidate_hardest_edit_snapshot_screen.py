#!/home/haoyi/miniconda/envs/gdino5090/bin/python
"""Run diagnostic-only strict1607 screens for the archived v40 snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import eval_refcoco_stageb as ref_eval  # noqa: E402
from tools import eval_text_groundingdino_refcoco_tn as combined_eval  # noqa: E402


FIXED_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
EVALUATOR = REPO_ROOT / "tools/eval_text_groundingdino_refcoco_tn.py"
CONFIG = combined_eval._CANDIDATE_HARDEST_EDIT_CONFIDENCE_U0400_CONFIG
DATA_ROOT = Path("/media/haoyi/T9/data")
TN_MANIFEST = Path(
    combined_eval._PARTIAL_CONFIDENCE_TN_SPECS["strict1607"]["path"]
)
SNAPSHOTS = dict(combined_eval._V40_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS)
OUTPUTS = dict(combined_eval._V40_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS)
LOGS = {
    update: combined_eval._V40_IMMUTABLE_ARCHIVED_SCREEN_ROOT
    / f"u{update:06d}_strict1607_console.log"
    for update in SNAPSHOTS
}


class SnapshotScreenError(RuntimeError):
    pass


def build_command(update: int) -> list[str]:
    if update not in SNAPSHOTS:
        raise SnapshotScreenError(f"unsupported archived update: {update}")
    return [
        str(FIXED_PYTHON),
        str(EVALUATOR),
        "--config",
        str(CONFIG),
        "--ckpts",
        str(SNAPSHOTS[update]),
        "--output_dir",
        str(OUTPUTS[update]),
        "--data_root",
        str(DATA_ROOT),
        "--device",
        "cuda:0",
        "--batch_size",
        "16",
        "--num_workers",
        "4",
        "--seed",
        "42",
        "--amp",
        "--skip_ref",
        "--tn_jsonl",
        str(TN_MANIFEST),
        "--tn_splits",
        "refcocop_val",
        "refcocog_umd_val",
        "--partial_dense_duty_confidence_diagnostic",
        "--immutable_v40_archived_snapshot_diagnostic",
        "--topk",
        "1",
        "--threshold_tprs",
        "0.75",
        "0.9",
        "0.95",
        "--score_thresholds",
        "0.5",
        "--max_ref_batches",
        "0",
        "--max_tn_batches",
        "0",
        "--log_every",
        "50",
    ]


def _expected_sha256(update: int) -> str:
    return str(ref_eval._V40_IMMUTABLE_ARCHIVED_SNAPSHOTS[update]["sha256"])


def inspect(update: int) -> Dict[str, Any]:
    records = ref_eval._verify_v40_immutable_archived_diagnostic_files(
        SNAPSHOTS[update]
    )
    return {
        "schema": "pivot.stageb.v40_immutable_snapshot_screen/v1",
        "diagnostic_only": True,
        "formal_gate_eligible": False,
        "confidence_evaluated": True,
        "terminal_checkpoint": False,
        "optimizer_updates": update,
        "snapshot": records["snapshot"],
        "terminal": records["terminal"],
        "output": str(OUTPUTS[update]),
        "command": build_command(update),
    }


def validate_inputs(update: int) -> Dict[str, Any]:
    state = inspect(update)
    output = OUTPUTS[update]
    if output.exists():
        raise SnapshotScreenError(f"fixed output directory must be fresh: {output}")
    for path in (FIXED_PYTHON, EVALUATOR, CONFIG, DATA_ROOT, TN_MANIFEST):
        if not path.exists():
            raise SnapshotScreenError(f"required input is missing: {path}")
    if state["snapshot"]["sha256"] != _expected_sha256(update):
        raise SnapshotScreenError("archived snapshot SHA256 mismatch")
    return state


def _require_record(
    value: Any, *, expected_path: Path, expected_sha256: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("path") != str(expected_path.resolve(strict=True))
        or value.get("sha256") != expected_sha256
        or not isinstance(value.get("size_bytes"), int)
        or value.get("size_bytes", 0) <= 0
    ):
        raise SnapshotScreenError("summary contains invalid immutable file provenance")
    return value


def verify_summary(update: int) -> Dict[str, Any]:
    summary_path = OUTPUTS[update] / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotScreenError(f"cannot read snapshot summary: {exc}") from exc
    ref_rows = payload.get("refcoco") if isinstance(payload, Mapping) else None
    tn_rows = payload.get("tn") if isinstance(payload, Mapping) else None
    if ref_rows != [] or not isinstance(tn_rows, list) or len(tn_rows) != 1:
        raise SnapshotScreenError("snapshot screen must contain one TN-only result")
    row = tn_rows[0]
    if (
        not isinstance(row, Mapping)
        or row.get("diagnostic_only") is not True
        or row.get("formal_gate_eligible") is not False
        or row.get("confidence_evaluated") is not True
        or row.get("terminal_checkpoint") is not False
        or row.get("immutable_v40_archived_snapshot_diagnostic") is not True
        or row.get("immutable_v39_archived_snapshot_diagnostic") is not None
        or row.get("optimizer_updates") != update
        or row.get("expected_optimizer_updates") != 400
        or row.get("remaining_optimizer_updates") != 400 - update
        or row.get("checkpoint_reason") != "interval"
        or row.get("num_pairs") != 1607
    ):
        raise SnapshotScreenError("snapshot summary violates diagnostic-only contract")
    provenance = row.get("immutable_archived_snapshot_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema")
        != "pivot.stageb.v40_immutable_archived_diagnostic/v1"
        or provenance.get("optimizer_updates") != update
        or provenance.get("checkpoint_reason") != "interval"
    ):
        raise SnapshotScreenError("snapshot summary lacks immutable provenance")
    snapshot_sha256 = _expected_sha256(update)
    terminal_path = Path(ref_eval._V40_IMMUTABLE_ARCHIVED_TERMINAL["path"])
    terminal_sha256 = str(
        ref_eval._V40_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]
    )
    for key in (
        "snapshot_before_validation",
        "snapshot_after_validation",
        "snapshot_after_model_load",
        "snapshot_after_evaluation",
    ):
        _require_record(
            provenance.get(key),
            expected_path=SNAPSHOTS[update],
            expected_sha256=snapshot_sha256,
        )
    for key in (
        "terminal_before_validation",
        "terminal_after_validation",
        "terminal_after_model_load",
        "terminal_after_evaluation",
    ):
        _require_record(
            provenance.get(key),
            expected_path=terminal_path,
            expected_sha256=terminal_sha256,
        )
    return dict(row)


def run(update: int) -> Dict[str, Any]:
    before = validate_inputs(update)
    log = LOGS[update]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            build_command(update),
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise SnapshotScreenError(
            f"U{update} evaluator failed with exit code {completed.returncode}; "
            f"see {log}"
        )
    after = inspect(update)
    if before["snapshot"] != after["snapshot"] or before["terminal"] != after[
        "terminal"
    ]:
        raise SnapshotScreenError("checkpoint files changed across evaluator process")
    return verify_summary(update)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--updates",
        nargs="+",
        type=int,
        choices=sorted(SNAPSHOTS),
        default=sorted(SNAPSHOTS),
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    updates = list(dict.fromkeys(args.updates))
    results = []
    for update in updates:
        results.append(run(update) if args.run else inspect(update))
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
