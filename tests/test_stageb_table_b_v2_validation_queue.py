import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from tools import run_stageb_table_b_v2_validation_queue as queue_runner


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


class TableBV2ValidationQueueTest(unittest.TestCase):
    def _training_record(self, root: Path) -> dict:
        queue_dir = root / "training-queue"
        queue_dir.mkdir()
        records = {}
        for name in (
            "queue.json",
            queue_runner.training_queue.SOURCE_PLAN_NAME,
            queue_runner.training_queue.SCOPE_PLAN_NAME,
            queue_runner.training_queue.COMPLETION_NAME,
        ):
            path = queue_dir / name
            _write_json(path, {"fixture": name})
            records[name] = queue_runner._file_record(path)
        runs = {}
        for run_id in queue_runner.training_contract.FORMAL_RUN_IDS:
            condition, raw_seed = run_id.split(":", 1)
            run_root = root / "training-runs" / condition / f"seed{raw_seed}"
            run_root.mkdir(parents=True)
            runs[run_id] = str(run_root.resolve())
        return {
            "queue_dir": str(queue_dir.resolve()),
            "queue_id": "training-queue-id",
            "plan_sha256": "a" * 64,
            "profile": queue_runner.training_contract.FORMAL_PROFILE,
            "ordered_run_ids": list(queue_runner.training_contract.FORMAL_RUN_IDS),
            "completion_semantic_sha256": "b" * 64,
            "manifest": records["queue.json"],
            "source_plan": records[queue_runner.training_queue.SOURCE_PLAN_NAME],
            "scope_plan": records[queue_runner.training_queue.SCOPE_PLAN_NAME],
            "completion_attestation": records[
                queue_runner.training_queue.COMPLETION_NAME
            ],
            "runs": runs,
        }

    def _create(self, root: Path):
        training = self._training_record(root)
        source = root / "source.py"
        source.write_text("VALUE = 1\n", encoding="ascii")
        matched = root / "matched.json"
        _write_json(matched, {"matched": True})
        matched_records = [
            {"role": "audit", **queue_runner._file_record(matched)}
        ]
        queue_dir = root / "validation-queue"
        output_root = root / "evaluations"
        patches = (
            mock.patch.object(
                queue_runner,
                "_training_queue_record",
                return_value=copy.deepcopy(training),
            ),
            mock.patch.object(
                queue_runner, "_evaluation_source_paths", return_value=[source]
            ),
            mock.patch.object(
                queue_runner, "_recursive_sources", return_value=[source]
            ),
            mock.patch.object(
                queue_runner,
                "_matched_input_records",
                return_value=matched_records,
            ),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            queue = queue_runner.create_queue(
                queue_dir,
                training_queue_dir=Path(training["queue_dir"]),
                output_root=output_root,
                lease_root=root / "leases",
                gpu_key="0",
            )
        return queue_dir, output_root, training, queue

    def test_catalog_and_plan_are_exact_three_seed_six_phase_contract(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(queue_runner.main(["list", "--json"]), 0)
        catalog = json.loads(stdout.getvalue())
        self.assertEqual(catalog["ordered_seeds"], [17, 42, 73])
        self.assertEqual(catalog["phase_order_per_seed"], ["D2m", "D3m"])
        self.assertEqual(catalog["total_phase_count"], 6)

        with tempfile.TemporaryDirectory() as temporary:
            queue_dir, _output_root, training, queue = self._create(Path(temporary))
            plan = queue["plan"]
            self.assertEqual(plan["ordered_run_ids"], ["seed17", "seed42", "seed73"])
            self.assertEqual(len(plan["items"]), 3)
            self.assertEqual(plan["runtime"], queue_runner.RUNTIME)
            spec = queue_dir / queue_runner.VALIDATION_SPEC_NAME
            for seed, item in zip(queue_runner.SEEDS, plan["items"]):
                self.assertEqual(item["phase_order"], ["D2m", "D3m"])
                command = item["command"]
                self.assertEqual(command[0], str(queue_runner.DEFAULT_PYTHON.resolve()))
                self.assertEqual(command[command.index("--seed") + 1], str(seed))
                self.assertEqual(
                    command[command.index("--training-source-contract") + 1],
                    queue_runner.evaluator.FORMAL_V2_TRAINING_SOURCE_CONTRACT,
                )
                self.assertEqual(
                    command[command.index("--validation-queue-spec") + 1], str(spec)
                )
                self.assertEqual(
                    command[command.index("--d2m-training-run-root") + 1],
                    training["runs"][f"D2m:{seed}"],
                )
                self.assertLess(
                    command.index("--d2m-training-run-root"),
                    command.index("--d3m-training-run-root"),
                )

    def test_self_rehashed_command_tamper_and_preexisting_outputs_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, output_root, _training, queue = self._create(root)
            tampered = copy.deepcopy(queue)
            tampered["plan"]["items"][0]["command"].remove(
                "--validation-queue-spec"
            )
            tampered["plan_sha256"] = queue_runner._canonical_sha(tampered["plan"])
            queue_runner._write_json_atomic(queue_dir / "queue.json", tampered)
            queue_runner._write_json_atomic(
                queue_dir / queue_runner.VALIDATION_SPEC_NAME,
                queue_runner._spec_payload(
                    tampered["plan"], tampered["plan_sha256"]
                ),
            )
            with self.assertRaisesRegex(
                queue_runner.TableBV2ValidationQueueError, "item 0 drifted"
            ):
                queue_runner.load_queue(queue_dir)

            second_root = root / "second"
            second_root.mkdir()
            training = self._training_record(second_root)
            occupied = second_root / "evaluations"
            occupied.mkdir()
            with self.assertRaisesRegex(FileExistsError, "output root must be fresh"):
                queue_runner.build_plan(
                    second_root / "queue",
                    training_queue_dir=Path(training["queue_dir"]),
                    output_root=occupied,
                    lease_root=second_root / "leases",
                    gpu_key="0",
                )
            self.assertFalse(output_root.exists())

    def test_reserved_crash_recovers_only_completed_orphan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _output_root, _training, queue = self._create(root)
            queue["status"] = "running"
            queue["items"][0].update(
                {
                    "status": "reserved",
                    "console_log": str(queue_dir / "logs/000-seed17.log"),
                }
            )
            queue_runner._write_json_atomic(queue_dir / "queue.json", queue)
            loaded = queue_runner.load_queue(queue_dir)
            output = Path(loaded["plan"]["items"][0]["evaluation_root"])
            output.mkdir(parents=True)
            _write_json(output / "launch.json", {"status": "running"})
            with (
                mock.patch.object(queue_runner, "_verify_sources"),
                mock.patch.object(queue_runner, "_matching_processes", return_value=[]),
                mock.patch.object(queue_runner.serial_queue, "_ensure_lease"),
                self.assertRaisesRegex(
                    queue_runner.TableBV2ValidationQueueError,
                    "without a recoverable completed launch",
                ),
            ):
                queue_runner._launch_reserved(loaded, 0)

            _write_json(output / "launch.json", {"status": "completed"})
            with (
                mock.patch.object(queue_runner, "_verify_sources"),
                mock.patch.object(queue_runner, "_matching_processes", return_value=[]),
                mock.patch.object(queue_runner.serial_queue, "_ensure_lease"),
            ):
                queue_runner._launch_reserved(loaded, 0)
            recovered = queue_runner.load_queue(queue_dir)
            self.assertEqual(recovered["items"][0]["status"], "launched")
            self.assertEqual(recovered["items"][0]["child_pid"], 0)

    def test_final_receipt_is_reloaded_before_lease_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _output_root, _training, queue = self._create(root)
            queue["status"] = "running"
            evidence_by_index = {
                0: {"run_id": "seed17", "durable": True},
                1: {"run_id": "seed42", "durable": True},
                2: {"run_id": "seed73", "durable": True},
            }
            for index in (0, 1):
                queue["items"][index].update(
                    {
                        "status": "completed",
                        "completion_evidence": evidence_by_index[index],
                    }
                )
            queue["items"][2].update(
                {
                    "status": "launched",
                    "child_pid": 123,
                    "child_process_identity": {"available": True},
                }
            )
            queue_runner._write_json_atomic(queue_dir / "queue.json", queue)
            loaded = queue_runner.load_queue(queue_dir)
            evidence = evidence_by_index[2]
            with (
                mock.patch.object(queue_runner.serial_queue, "_ensure_lease"),
                mock.patch.object(
                    queue_runner.serial_queue, "_process_running", return_value=False
                ),
                mock.patch.object(
                    queue_runner,
                    "_verify_completed",
                    side_effect=lambda _queue, index: evidence_by_index[index],
                ) as verify,
                mock.patch.object(
                    queue_runner.serial_queue, "_clear_owned_lease"
                ) as clear,
            ):
                queue_runner._advance_launched(loaded, 2)
            persisted = queue_runner.load_queue(queue_dir)
            self.assertEqual(persisted["status"], "completed")
            self.assertEqual(persisted["items"][2]["completion_evidence"], evidence)
            self.assertEqual(verify.call_count, 4)
            clear.assert_called_once()

    def test_final_replay_drift_marks_queue_failed_and_keeps_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _output_root, _training, queue = self._create(root)
            queue["status"] = "running"
            for index in (0, 1):
                queue["items"][index].update(
                    {
                        "status": "completed",
                        "completion_evidence": {"run_id": f"seed{queue_runner.SEEDS[index]}"},
                    }
                )
            queue["items"][2].update(
                {
                    "status": "launched",
                    "child_pid": 123,
                    "child_process_identity": {"available": True},
                }
            )
            queue_runner._write_json_atomic(queue_dir / "queue.json", queue)
            final = {"run_id": "seed73"}
            with (
                mock.patch.object(queue_runner.serial_queue, "_ensure_lease"),
                mock.patch.object(
                    queue_runner.serial_queue, "_process_running", return_value=False
                ),
                mock.patch.object(queue_runner, "_matching_processes", return_value=[]),
                mock.patch.object(
                    queue_runner,
                    "_verify_completed",
                    side_effect=[final, {"run_id": "drifted-seed17"}],
                ),
                mock.patch.object(
                    queue_runner.serial_queue, "_clear_owned_lease"
                ) as clear,
            ):
                failed = queue_runner.advance_once(queue_dir)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["items"][2]["status"], "failed")
            self.assertIn("completion evidence drifted", failed["failure"]["error"])
            clear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
