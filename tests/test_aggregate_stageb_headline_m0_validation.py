import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import aggregate_stageb_headline_m0_validation as aggregator


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


def _sealed_report(*, created_at_utc: str = "2026-07-19T09:00:00+00:00") -> dict:
    report = {
        "schema": aggregator.REPORT_SCHEMA,
        "status": aggregator.REPORT_STATUS,
        "created_at_utc": created_at_utc,
        "payload": {"metric": 0.25},
    }
    report["report_sha256"] = aggregator._report_sha256(report)
    return report


def _aggregate_fixture(root: Path) -> dict:
    queue_dir = root / "queue"
    queue_dir.mkdir()

    def bound_file(name: str) -> tuple[Path, dict]:
        path = queue_dir / "bound" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="ascii")
        return path, aggregator.queue_runner._file_record(path)

    _, runner_python = bound_file("python")
    _, evaluation_runner = bound_file("evaluation_runner.py")
    _, evaluation_source = bound_file("evaluation_source.py")
    _, controller_source = bound_file("controller_source.py")
    _, aggregation_source = bound_file("aggregation_source.py")
    training_queues = []
    for contract_id in aggregator.queue_runner.CONTRACT_IDS:
        path, record = bound_file(f"{contract_id}_training_queue.json")
        training_queues.append(
            {
                "contract_id": contract_id,
                "queue_dir": str(path.parent),
                "manifest_at_creation": record,
            }
        )
    items = [
        {
            "run_id": run_id,
            "evaluation_root": str(
                queue_dir / "evaluations" / run_id.replace(":", "_")
            ),
        }
        for run_id in aggregator.queue_runner.RUN_IDS
    ]
    plan = {
        "queue_id": "validation-queue",
        "queue_dir": str(queue_dir),
        "output_root": str(queue_dir.parent / "evaluation-root"),
        "items": items,
        "runner_python": runner_python,
        "evaluation_runner": evaluation_runner,
        "evaluation_sources": [evaluation_source],
        "controller_sources": [controller_source],
        "aggregation_sources": [aggregation_source],
        "training_queues": training_queues,
    }
    plan_sha = "a" * 64
    queue = {"status": "completed", "plan": plan, "plan_sha256": plan_sha}
    _write_json(queue_dir / "queue.json", queue)
    spec = aggregator.queue_runner._aggregation_spec_payload(plan, plan_sha)
    _write_json(queue_dir / aggregator.queue_runner.AGGREGATION_SPEC_NAME, spec)

    evidence_paths: dict[str, Path] = {}
    loaded = {}
    for contract_id in aggregator.queue_runner.CONTRACT_IDS:
        runs = {}
        for seed in aggregator.queue_runner.SEEDS:
            run_id = f"{contract_id}:{seed}"
            evidence_path, _ = bound_file(
                f"{contract_id}_seed{seed}_evaluation_evidence.jsonl"
            )
            evidence_paths[run_id] = evidence_path
            runs[seed] = SimpleNamespace(
                checkpoint_sha256=(
                    f"{contract_id}-{seed}".encode("ascii").hex().ljust(64, "0")
                )[:64],
                sealed_files=aggregator.matrix._snapshot_files(
                    (evidence_path,), aggregator.queue_runner.evaluator.HashCache()
                ),
            )
        loaded[contract_id] = runs
    return {
        "queue_dir": queue_dir,
        "queue": queue,
        "spec": spec,
        "loaded": loaded,
        "evidence_paths": evidence_paths,
        "aggregation_sources": [aggregation_source],
        "comparison": {"candidate_minus_reference": {"metric": 0.1}},
    }


class HeadlineM0ValidationAggregationTest(unittest.TestCase):
    def test_report_sha_excludes_only_top_level_timestamp_and_digest(self):
        first = _sealed_report(created_at_utc="2026-07-19T09:00:00+00:00")
        second = copy.deepcopy(first)
        second["created_at_utc"] = "2026-07-20T10:00:00+00:00"
        second["report_sha256"] = "f" * 64
        self.assertEqual(
            aggregator._report_sha256(first), aggregator._report_sha256(second)
        )

        nested_timestamp = copy.deepcopy(first)
        nested_timestamp["payload"]["created_at_utc"] = (
            "2026-07-20T10:00:00+00:00"
        )
        self.assertNotEqual(
            aggregator._report_sha256(first),
            aggregator._report_sha256(nested_timestamp),
        )

    def test_formal_bootstrap_contract_is_not_configurable(self):
        with self.assertRaisesRegex(
            aggregator.HeadlineM0AggregationError, "exactly 5000/0.95/20260719"
        ):
            aggregator.aggregate(Path("unused"), bootstrap_iterations=4_999)
        with self.assertRaisesRegex(
            aggregator.HeadlineM0AggregationError, "exactly 5000/0.95/20260719"
        ):
            aggregator.aggregate(Path("unused"), confidence=0.90)
        with self.assertRaisesRegex(
            aggregator.HeadlineM0AggregationError, "exactly 5000/0.95/20260719"
        ):
            aggregator.aggregate(Path("unused"), bootstrap_seed=1)

    def test_report_writer_is_atomic_no_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            aggregator._write_json_no_replace(output, {"value": 1})
            original = output.read_bytes()
            with self.assertRaisesRegex(
                aggregator.HeadlineM0AggregationError, "refusing to overwrite"
            ):
                aggregator._write_json_no_replace(output, {"value": 2})
            self.assertEqual(output.read_bytes(), original)

    def test_dedicated_launch_validator_accepts_only_formal_paper_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_root = root / "evaluation"
            evaluation_root.mkdir()
            training_queue_dir = root / "training-queue"
            training_queue_dir.mkdir()
            _write_json(training_queue_dir / "queue.json", {"status": "completed"})
            spec_path = root / "spec.json"
            _write_json(spec_path, {"schema": "fixture"})
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            contract = aggregator.queue_runner.training_runner.CONTRACTS["M0"]
            training_root = contract.canonical_training_root(17)
            source = SimpleNamespace(
                training_run_root=training_root,
                training_queue_manifest=(training_queue_dir / "queue.json").resolve(),
                checkpoint=checkpoint.resolve(),
            )
            raw_source = {
                "kind": "pivot_paper_training_run",
                "formal_contract_id": "M0",
                "evaluation_id": "M0_seed17",
                "training_run_id": "M0:17",
                "training_seed": 17,
                "training_phase": "final",
                "diagnostic_only": False,
                "matrix_validation_only": contract.matrix_validation_only,
                "training_queue_id": "m0-queue",
                "training_queue_plan_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
            }
            launch = {
                "schema": aggregator.queue_runner.evaluator.SCHEMA,
                "status": "completed",
                "output_dir": str(evaluation_root),
                "evaluation_id": "M0_seed17",
                "protocol": {
                    "profile": aggregator.queue_runner.PROFILE,
                    "ref_splits": list(aggregator.matrix.REF_VALIDATION_SPLITS),
                    "strict_manifests": {},
                    "processes": ["validation_calibration"],
                    "strict1607_skip_ref": False,
                    "per_example_records": True,
                    "release_policy": (
                        "ablation_matrix_validation_only_no_ref_test_or_strict_access"
                    ),
                    "screen_calibration": {"sealed": True},
                },
                "completed_phases": [
                    {
                        "phase_id": "validation_calibration",
                        "status": "completed",
                        "returncode": 0,
                    }
                ],
                "source": raw_source,
                "runtime": {},
            }
            runtime = SimpleNamespace()
            with (
                mock.patch.object(
                    aggregator.matrix,
                    "_evaluation_source_from_launch",
                    return_value=source,
                ),
                mock.patch.object(
                    aggregator.queue_runner.evaluator, "_revalidate_matrix_source"
                ),
                mock.patch.object(
                    aggregator.matrix, "_runtime_from_launch", return_value=runtime
                ),
                mock.patch.object(
                    aggregator,
                    "_replay_headline_plan_contract",
                    return_value={"runtime": "sealed"},
                ),
            ):
                observed = aggregator._validate_headline_launch(
                    launch,
                    contract_id="M0",
                    seed=17,
                    root=evaluation_root.resolve(),
                    training_queue={
                        "queue_dir": str(training_queue_dir),
                        "queue_id": "m0-queue",
                        "plan_sha256": "a" * 64,
                    },
                    spec_path=spec_path,
                    cache=aggregator.queue_runner.evaluator.HashCache(),
                )
            self.assertEqual(observed[0:2], ("M0_seed17", "M0:17"))
            self.assertEqual(observed[-1], {"runtime": "sealed"})

            launch["source"] = {**raw_source, "kind": "pivot_token_ablation_training_run"}
            with self.assertRaisesRegex(
                aggregator.HeadlineM0AggregationError, "queue-attested"
            ):
                aggregator._validate_headline_launch(
                    launch,
                    contract_id="M0",
                    seed=17,
                    root=evaluation_root.resolve(),
                    training_queue={
                        "queue_dir": str(training_queue_dir),
                        "queue_id": "m0-queue",
                        "plan_sha256": "a" * 64,
                    },
                    spec_path=spec_path,
                    cache=aggregator.queue_runner.evaluator.HashCache(),
                )

    def test_aggregate_consumes_only_verified_exact_six_item_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _aggregate_fixture(Path(temporary))
            with (
                mock.patch.object(
                    aggregator.queue_runner,
                    "load_queue",
                    return_value=fixture["queue"],
                ),
                mock.patch.object(
                    aggregator.queue_runner,
                    "verify_queue",
                    return_value={
                        "schema": aggregator.queue_runner.VERIFICATION_SCHEMA,
                        "status": "passed",
                    },
                ),
                mock.patch.object(
                    aggregator,
                    "_verify_predeclared_sources",
                    return_value=fixture["aggregation_sources"],
                ),
                mock.patch.object(
                    aggregator, "_load_all_runs", return_value=fixture["loaded"]
                ),
                mock.patch.object(
                    aggregator.matrix,
                    "_comparison",
                    return_value=fixture["comparison"],
                ) as compare,
                mock.patch.object(
                    aggregator.matrix,
                    "_aggregate_experiment",
                    side_effect=lambda runs: {"seed_count": len(runs)},
                ),
            ):
                report = aggregator.aggregate(fixture["queue_dir"])

            self.assertEqual(report["status"], "validated_matrix_validation_only")
            self.assertEqual(report["direction"], "M0N_minus_M0")
            self.assertEqual(report["comparison"], fixture["comparison"])
            self.assertEqual(report["experiments"]["M0"]["seed_count"], 3)
            self.assertEqual(report["experiments"]["M0N"]["seed_count"], 3)
            self.assertEqual(report["report_sha256"], aggregator._report_sha256(report))
            self.assertEqual(compare.call_args.kwargs["iterations"], 5_000)
            self.assertEqual(compare.call_args.kwargs["confidence"], 0.95)
            self.assertEqual(compare.call_args.kwargs["bootstrap_seed"], 20260719)

            drifted = dict(fixture["spec"])
            drifted["expected_train_seeds"] = [17, 42]
            _write_json(
                fixture["queue_dir"]
                / aggregator.queue_runner.AGGREGATION_SPEC_NAME,
                drifted,
            )
            with (
                mock.patch.object(
                    aggregator.queue_runner,
                    "load_queue",
                    return_value=fixture["queue"],
                ),
                mock.patch.object(
                    aggregator.queue_runner,
                    "verify_queue",
                    return_value={"schema": "verification", "status": "passed"},
                ),
                mock.patch.object(
                    aggregator,
                    "_verify_predeclared_sources",
                    return_value=fixture["aggregation_sources"],
                ),
                self.assertRaisesRegex(
                    aggregator.HeadlineM0AggregationError, "immutable queue"
                ),
                ):
                aggregator.aggregate(fixture["queue_dir"])

    def test_aggregate_rejects_bound_files_mutated_during_statistics(self):
        targets = {
            "aggregation_spec": lambda fixture: (
                fixture["queue_dir"] / aggregator.queue_runner.AGGREGATION_SPEC_NAME
            ),
            "queue_manifest": lambda fixture: fixture["queue_dir"] / "queue.json",
            "queue_source": lambda fixture: Path(
                fixture["aggregation_sources"][0]["path"]
            ),
            "evaluation_evidence": lambda fixture: fixture["evidence_paths"][
                "M0:17"
            ],
        }
        for surface, select_target in targets.items():
            with self.subTest(surface=surface), tempfile.TemporaryDirectory() as temporary:
                fixture = _aggregate_fixture(Path(temporary))
                target = select_target(fixture)
                mutated = False

                def aggregate_experiment(runs):
                    nonlocal mutated
                    if not mutated:
                        target.write_text(
                            f"mutated {surface} during paired aggregation\n",
                            encoding="ascii",
                        )
                        mutated = True
                    return {"seed_count": len(runs)}

                with (
                    mock.patch.object(
                        aggregator.queue_runner,
                        "load_queue",
                        return_value=fixture["queue"],
                    ),
                    mock.patch.object(
                        aggregator.queue_runner,
                        "verify_queue",
                        return_value={
                            "schema": aggregator.queue_runner.VERIFICATION_SCHEMA,
                            "status": "passed",
                        },
                    ),
                    mock.patch.object(
                        aggregator,
                        "_verify_predeclared_sources",
                        return_value=fixture["aggregation_sources"],
                    ),
                    mock.patch.object(
                        aggregator, "_load_all_runs", return_value=fixture["loaded"]
                    ),
                    mock.patch.object(
                        aggregator.matrix,
                        "_comparison",
                        return_value=fixture["comparison"],
                    ),
                    mock.patch.object(
                        aggregator.matrix,
                        "_aggregate_experiment",
                        side_effect=aggregate_experiment,
                    ),
                    self.assertRaisesRegex(
                        aggregator.HeadlineM0AggregationError,
                        "evidence changed during aggregation",
                    ),
                ):
                    aggregator.aggregate(fixture["queue_dir"])

    def test_verify_report_accepts_timestamp_volatility_and_replays_default_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "headline_m0_m0n_validation_report.json"
            queue_dir = root / "canonical-queue"
            persisted = _sealed_report(
                created_at_utc="2026-07-19T09:00:00+00:00"
            )
            replayed = _sealed_report(
                created_at_utc="2026-07-19T09:30:00+00:00"
            )
            _write_json(report_path, persisted)
            with (
                mock.patch.object(aggregator, "DEFAULT_REPORT_PATH", report_path),
                mock.patch.object(aggregator, "DEFAULT_QUEUE_DIR", queue_dir),
                mock.patch.object(
                    aggregator, "aggregate", return_value=replayed
                ) as replay,
            ):
                observed = aggregator.verify_report(report_path)
            self.assertEqual(observed, persisted)
            replay.assert_called_once_with(queue_dir)

    def test_verify_report_rejects_noncanonical_path_before_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.json"
            alternate = root / "alternate.json"
            _write_json(canonical, _sealed_report())
            with (
                mock.patch.object(aggregator, "DEFAULT_REPORT_PATH", canonical),
                mock.patch.object(aggregator, "aggregate") as replay,
                self.assertRaisesRegex(
                    aggregator.HeadlineM0AggregationError, "path is not canonical"
                ),
            ):
                aggregator.verify_report(alternate)
            replay.assert_not_called()

    def test_verify_report_rejects_direct_and_self_consistent_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "canonical.json"
            queue_dir = root / "canonical-queue"
            expected = _sealed_report()

            direct = copy.deepcopy(expected)
            direct["payload"]["metric"] = 0.5
            _write_json(report_path, direct)
            with (
                mock.patch.object(aggregator, "DEFAULT_REPORT_PATH", report_path),
                mock.patch.object(aggregator, "DEFAULT_QUEUE_DIR", queue_dir),
                mock.patch.object(aggregator, "aggregate") as replay,
                self.assertRaisesRegex(
                    aggregator.HeadlineM0AggregationError, "self SHA-256 mismatch"
                ),
            ):
                aggregator.verify_report(report_path)
            replay.assert_not_called()

            forged = copy.deepcopy(expected)
            forged["payload"]["metric"] = 0.75
            forged["report_sha256"] = aggregator._report_sha256(forged)
            _write_json(report_path, forged)
            with (
                mock.patch.object(aggregator, "DEFAULT_REPORT_PATH", report_path),
                mock.patch.object(aggregator, "DEFAULT_QUEUE_DIR", queue_dir),
                mock.patch.object(aggregator, "aggregate", return_value=expected),
                self.assertRaisesRegex(
                    aggregator.HeadlineM0AggregationError,
                    "differs from full canonical replay",
                ),
            ):
                aggregator.verify_report(report_path)

    def test_verify_report_rejects_wrong_schema_and_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "canonical.json"
            for key, value in (
                ("schema", "pivot.stageb.other/v1"),
                ("status", "passed"),
            ):
                with self.subTest(key=key):
                    report = _sealed_report()
                    report[key] = value
                    report["report_sha256"] = aggregator._report_sha256(report)
                    _write_json(report_path, report)
                    with (
                        mock.patch.object(
                            aggregator, "DEFAULT_REPORT_PATH", report_path
                        ),
                        mock.patch.object(aggregator, "aggregate") as replay,
                        self.assertRaisesRegex(
                            aggregator.HeadlineM0AggregationError, "schema/status"
                        ),
                    ):
                        aggregator.verify_report(report_path)
                    replay.assert_not_called()

    def test_verify_report_detects_mutation_during_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "canonical.json"
            queue_dir = root / "canonical-queue"
            expected = _sealed_report()
            _write_json(report_path, expected)

            def mutate_during_replay(_queue_dir):
                report_path.write_text(
                    json.dumps(expected, indent=2, sort_keys=True) + "\n",
                    encoding="ascii",
                )
                return expected

            with (
                mock.patch.object(aggregator, "DEFAULT_REPORT_PATH", report_path),
                mock.patch.object(aggregator, "DEFAULT_QUEUE_DIR", queue_dir),
                mock.patch.object(
                    aggregator, "aggregate", side_effect=mutate_during_replay
                ),
                self.assertRaisesRegex(
                    aggregator.HeadlineM0AggregationError,
                    "changed during full replay",
                ),
            ):
                aggregator.verify_report(report_path)

    def test_cli_defaults_to_canonical_fresh_no_replace_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "canonical-queue"
            report_path = root / "headline_m0_m0n_validation_report.json"
            queue_dir.mkdir()
            report = _sealed_report()
            with (
                mock.patch.object(aggregator, "DEFAULT_QUEUE_DIR", queue_dir),
                mock.patch.object(aggregator, "DEFAULT_REPORT_PATH", report_path),
                mock.patch.object(aggregator, "aggregate", return_value=report) as build,
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(aggregator.main([]), 0)
                original = report_path.read_bytes()
                with (
                    mock.patch("sys.stderr", new=io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    aggregator.main([])
            self.assertEqual(report_path.read_bytes(), original)
            self.assertEqual(build.call_count, 2)
            self.assertEqual(build.call_args.args, (queue_dir.resolve(),))

    def test_cli_rejects_noncanonical_output_and_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "canonical-queue"
            queue_dir.mkdir()
            report_path = root / "canonical-report.json"
            with (
                mock.patch.object(aggregator, "DEFAULT_QUEUE_DIR", queue_dir),
                mock.patch.object(aggregator, "DEFAULT_REPORT_PATH", report_path),
                mock.patch.object(aggregator, "aggregate") as build,
            ):
                for argv in (
                    ["--output", str(root / "alternate.json")],
                    [str(root / "alternate-queue")],
                ):
                    with (
                        self.subTest(argv=argv),
                        mock.patch("sys.stderr", new=io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        aggregator.main(argv)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
