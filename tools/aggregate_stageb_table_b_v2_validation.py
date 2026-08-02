#!/usr/bin/env python3
"""Aggregate the queue-bound, three-seed formal Table-B v2 validation panel."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import aggregate_stageb_table_b_matched_panel as matched_aggregator  # noqa: E402
from tools import run_stageb_table_b_matched_evaluations as evaluator  # noqa: E402
from tools import run_stageb_table_b_v2_validation_queue as queue_runner  # noqa: E402
from tools import stageb_table_b_matched_eval_surface as surface  # noqa: E402


REPORT_SCHEMA = "pivot.stageb.table_b_v2_validation_aggregate/v1"


class TableBV2ValidationAggregateError(RuntimeError):
    """The formal Table-B v2 validation aggregate could not be replayed."""


def _write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"aggregate output must be fresh: {path}")
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _evaluation_outputs(
    queue: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[int, Path]:
    raw = spec.get("evaluation_outputs")
    if not isinstance(raw, Mapping) or set(raw) != {
        str(seed) for seed in queue_runner.SEEDS
    }:
        raise TableBV2ValidationAggregateError(
            "validation spec does not name the exact three seed outputs"
        )
    outputs: dict[int, Path] = {}
    planned = {
        int(item["seed"]): Path(str(item["evaluation_root"])).resolve(strict=True)
        for item in queue["plan"]["items"]
    }
    for seed in queue_runner.SEEDS:
        output = Path(str(raw[str(seed)])).resolve(strict=True)
        if output != planned[seed]:
            raise TableBV2ValidationAggregateError(
                f"seed {seed} output differs from the immutable queue"
            )
        launch = queue_runner._read_json(
            output / "launch.json", label=f"seed {seed} matched launch"
        )
        contract = launch.get("contract")
        if not (
            isinstance(contract, Mapping)
            and contract.get("seed") == seed
            and contract.get("training_source_contract")
            == evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT
            and contract.get("conditions") == list(queue_runner.PHASE_ORDER)
        ):
            raise TableBV2ValidationAggregateError(
                f"seed {seed} is not a formal-v2 matched evaluation"
            )
        outputs[seed] = output
    return outputs


def aggregate(queue_dir: Path) -> dict[str, Any]:
    queue_dir = Path(queue_dir).expanduser().resolve(strict=True)
    queue = queue_runner.load_queue(queue_dir)
    verification = queue_runner.verify_queue(queue_dir)
    if queue.get("status") != "completed" or verification.get("status") != "passed":
        raise TableBV2ValidationAggregateError(
            "Table-B v2 validation queue is not completed and verified"
        )
    spec_path = (queue_dir / queue_runner.VALIDATION_SPEC_NAME).resolve(strict=True)
    spec = queue_runner._read_json(spec_path, label="Table-B v2 validation input")
    expected_spec = queue_runner._spec_payload(
        queue["plan"], str(queue["plan_sha256"])
    )
    if spec != expected_spec:
        raise TableBV2ValidationAggregateError(
            "validation input differs from the immutable queue"
        )
    outputs = _evaluation_outputs(queue, spec)
    try:
        audit = matched_aggregator.verify_panel(surface.DEFAULT_AUDIT)
        report = matched_aggregator.aggregate_formal_matched_panel(
            audit_path=surface.DEFAULT_AUDIT,
            pair_ledger_path=surface.DEFAULT_LEDGER,
            d2m_source_path=Path(audit["outputs"]["d2m_calibration"]["path"]),
            d3m_source_path=surface.DEFAULT_D3M_SOURCE,
            evaluation_manifest_path=surface.DEFAULT_D3M_SOURCE,
            evaluation_outputs=outputs,
            declared_surface=surface.DECLARED_SCOPE,
            require_training_queue=True,
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        matched_aggregator.MatchedPanelError,
        matched_aggregator.MatchedPanelReportError,
    ) as exc:
        raise TableBV2ValidationAggregateError(
            f"formal Table-B v2 matched aggregation failed: {exc}"
        ) from exc
    protocol = report.get("formal_evaluation_protocol")
    if not (
        isinstance(protocol, Mapping)
        and protocol.get("training_source_contract")
        == evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT
    ):
        raise TableBV2ValidationAggregateError(
            "aggregate did not replay the formal-v2 training resolver"
        )
    report["schema"] = REPORT_SCHEMA
    report["status"] = "validated_formal_v2_supplemental_diagnostic"
    report.setdefault("validation", {}).update(
        {
            "formal_v2_training_resolver_replayed": True,
            "exact_three_seed_six_phase_queue_replayed": True,
            "validation_queue_spec_replayed": True,
            "shared_gpu_lease_queue_verified": True,
        }
    )
    report.setdefault("inputs", {})["formal_v2_validation_queue"] = {
        "queue_id": queue["plan"]["queue_id"],
        "plan_sha256": queue["plan_sha256"],
        "training_queue_id": queue["plan"]["training_queue"]["queue_id"],
        "training_queue_plan_sha256": queue["plan"]["training_queue"][
            "plan_sha256"
        ],
        "validation_input_spec": queue_runner._file_record(spec_path),
        "verification_schema": verification["schema"],
        "ordered_seeds": list(queue_runner.SEEDS),
        "phase_order_per_seed": list(queue_runner.PHASE_ORDER),
        "total_phase_count": len(queue_runner.SEEDS) * len(queue_runner.PHASE_ORDER),
        "aggregation_source_closure": list(queue["plan"]["aggregation_sources"]),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = aggregate(args.queue_dir)
        if args.output is not None:
            output = Path(args.output).expanduser().resolve(strict=False)
            queue = queue_runner.load_queue(args.queue_dir)
            evaluation_root = Path(queue["plan"]["output_root"]).resolve(strict=False)
            queue_dir = Path(queue["plan"]["queue_dir"]).resolve(strict=True)
            if (
                output == evaluation_root
                or evaluation_root in output.parents
                or output == queue_dir
                or queue_dir in output.parents
            ):
                raise TableBV2ValidationAggregateError(
                    "aggregate output cannot be written inside queue/evaluation evidence"
                )
            _write_json_no_replace(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
        TableBV2ValidationAggregateError,
        queue_runner.TableBV2ValidationQueueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
