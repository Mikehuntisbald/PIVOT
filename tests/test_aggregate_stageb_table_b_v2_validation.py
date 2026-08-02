import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import aggregate_stageb_table_b_matched_panel as matched_aggregator
from tools import aggregate_stageb_table_b_v2_validation as aggregator


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


class TableBV2ValidationAggregateTest(unittest.TestCase):
    def test_formal_aggregate_reresolves_every_seed_with_formal_v2_true(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_queue = root / "training-queue"
            training_queue.mkdir()
            _write_json(training_queue / "queue.json", {"status": "completed"})
            _write_json(root / "audit.json", {"fixture": True})
            outputs = {}
            postflights = {}
            for seed in matched_aggregator.FORMAL_SEEDS:
                output = root / f"seed{seed}"
                output.mkdir()
                training = {}
                conditions = {}
                for condition_index, condition in enumerate(("D2m", "D3m")):
                    checkpoint_sha = (
                        f"{seed:02x}{condition_index:02x}".ljust(64, "a")[:64]
                    )
                    training[condition] = {
                        "condition": condition,
                        "training_run_id": f"{condition}:{seed}",
                        "training_run_root": str(root / "training" / condition / f"seed{seed}"),
                        "checkpoint": {
                            "path": str(root / f"{condition}-{seed}.pth"),
                            "sha256": checkpoint_sha,
                        },
                        "training_source": {
                            "path": str(root / f"{condition}-train.jsonl"),
                            "sha256": "b" * 64,
                            "rows": 1,
                        },
                        "training_queue": {
                            "manifest": {
                                "path": str((training_queue / "queue.json").resolve())
                            }
                        },
                    }
                    (output / f"{condition}.records.jsonl").write_text(
                        "{}\n", encoding="ascii"
                    )
                    conditions[condition] = {
                        "final_records": {
                            "path": str(output / f"{condition}.records.jsonl")
                        }
                    }
                contract = {
                    "seed": seed,
                    "evaluation_seed": 42,
                    "conditions": ["D2m", "D3m"],
                    "formal_global_fpr_eligible": False,
                    "training_source_contract": "table_b_v2_formal",
                    "training": training,
                }
                launch = {"contract": contract, "contract_sha256": f"contract-{seed}"}
                postflight = {"surface": {}, "conditions": conditions}
                _write_json(output / "launch.json", launch)
                _write_json(output / "postflight.json", postflight)
                outputs[seed] = output
                postflights[str(output.resolve())] = postflight

            protocol = {
                "schema": "pivot.stageb.table_b_matched_formal_protocol/v1",
                "common_runtime": {"runtime": "fixed"},
                "phase_command_templates": {"D2m": ["cmd"], "D3m": ["cmd"]},
                "command_template_sha256": "c" * 64,
                "evaluation_code_closure": [{"path": "code.py"}],
                "evaluation_code_closure_sha256": "d" * 64,
            }

            def completed(output):
                return postflights[str(Path(output).resolve())]

            with (
                mock.patch.object(
                    matched_aggregator, "verify_panel", return_value={"outputs": {}}
                ),
                mock.patch(
                    "tools.run_stageb_table_b_matched_evaluations.verify_completed_output",
                    side_effect=completed,
                ),
                mock.patch(
                    "tools.run_stageb_table_b_matched_evaluations.formal_protocol_identity",
                    return_value=protocol,
                ),
                mock.patch(
                    "tools.run_stageb_table_b_matched_evaluations._resolve_sources",
                    return_value=(
                        {"D2m": SimpleNamespace(), "D3m": SimpleNamespace()},
                        {},
                    ),
                ) as resolve,
                mock.patch(
                    "tools.run_stageb_table_b_matched_evaluations._validate_runtime_sources"
                ),
                mock.patch.object(
                    matched_aggregator,
                    "aggregate_matched_panel",
                    return_value={"validation": {}, "inputs": {}},
                ),
            ):
                report = matched_aggregator.aggregate_formal_matched_panel(
                    audit_path=root / "audit.json",
                    pair_ledger_path=root / "ledger.jsonl",
                    d2m_source_path=root / "d2m.jsonl",
                    d3m_source_path=root / "d3m.jsonl",
                    evaluation_manifest_path=root / "d3m.jsonl",
                    evaluation_outputs=outputs,
                )
            self.assertEqual(resolve.call_count, 3)
            self.assertTrue(
                all(call.kwargs["formal_v2"] is True for call in resolve.call_args_list)
            )
            self.assertEqual(
                report["formal_evaluation_protocol"]["training_source_contract"],
                "table_b_v2_formal",
            )

    def test_queue_aggregate_accepts_only_exact_formal_v2_seed_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            queue_dir.mkdir()
            items = []
            outputs = {}
            for seed in aggregator.queue_runner.SEEDS:
                output = root / "evaluations" / f"seed{seed}"
                _write_json(
                    output / "launch.json",
                    {
                        "contract": {
                            "seed": seed,
                            "training_source_contract": "table_b_v2_formal",
                            "conditions": ["D2m", "D3m"],
                        }
                    },
                )
                items.append({"seed": seed, "evaluation_root": str(output.resolve())})
                outputs[str(seed)] = str(output.resolve())
            spec_path = queue_dir / aggregator.queue_runner.VALIDATION_SPEC_NAME
            spec = {"evaluation_outputs": outputs}
            _write_json(spec_path, spec)
            queue = {
                "status": "completed",
                "plan_sha256": "a" * 64,
                "plan": {
                    "queue_id": "validation-queue",
                    "queue_dir": str(queue_dir.resolve()),
                    "output_root": str((root / "evaluations").resolve()),
                    "training_queue": {
                        "queue_id": "training-queue",
                        "plan_sha256": "b" * 64,
                    },
                    "aggregation_sources": [{"path": "aggregate.py"}],
                    "items": items,
                },
            }
            base_report = {
                "schema": "base",
                "status": "base",
                "validation": {},
                "inputs": {},
                "formal_evaluation_protocol": {
                    "training_source_contract": "table_b_v2_formal"
                },
            }
            with (
                mock.patch.object(
                    aggregator.queue_runner, "load_queue", return_value=queue
                ),
                mock.patch.object(
                    aggregator.queue_runner,
                    "verify_queue",
                    return_value={"schema": "verified/v1", "status": "passed"},
                ),
                mock.patch.object(
                    aggregator.queue_runner, "_spec_payload", return_value=spec
                ),
                mock.patch.object(
                    aggregator.matched_aggregator,
                    "verify_panel",
                    return_value={
                        "outputs": {
                            "d2m_calibration": {"path": str(root / "d2m.jsonl")}
                        }
                    },
                ),
                mock.patch.object(
                    aggregator.matched_aggregator,
                    "aggregate_formal_matched_panel",
                    return_value=base_report,
                ) as aggregate_formal,
            ):
                report = aggregator.aggregate(queue_dir)
            self.assertEqual(
                report["schema"], "pivot.stageb.table_b_v2_validation_aggregate/v1"
            )
            self.assertTrue(
                report["validation"]["exact_three_seed_six_phase_queue_replayed"]
            )
            self.assertEqual(
                set(aggregate_formal.call_args.kwargs["evaluation_outputs"]),
                {17, 42, 73},
            )


if __name__ == "__main__":
    unittest.main()
