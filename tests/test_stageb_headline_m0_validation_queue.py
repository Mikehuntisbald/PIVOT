import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from tools import run_stageb_headline_m0_validation_queue as queue_runner


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


class HeadlineM0ValidationQueueTest(unittest.TestCase):
    def test_catalog_is_exact_ordered_six_item_objective_control(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(queue_runner.main(["list", "--json"]), 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["ordered_run_ids"], list(queue_runner.RUN_IDS))
        self.assertEqual(
            payload["ordered_run_ids"],
            ["M0:17", "M0:42", "M0:73", "M0N:17", "M0N:42", "M0N:73"],
        )
        self.assertTrue(payload["training_queues_separate"])
        self.assertEqual(payload["reference_experiment"], "M0")
        self.assertEqual(payload["candidate_experiment"], "M0N")

    def _training_record(self, root: Path, contract_id: str) -> dict:
        queue_dir = root / f"{contract_id}-training"
        queue_dir.mkdir()
        manifest = queue_dir / "queue.json"
        _write_json(manifest, {"status": "completed", "contract_id": contract_id})
        digest = ("a" if contract_id == "M0" else "b") * 64
        return {
            "contract_id": contract_id,
            "queue_dir": str(queue_dir.resolve()),
            "queue_id": f"{contract_id}-queue",
            "plan_sha256": digest,
            "queue_contract_sha256": digest,
            "stable_input_closure_digest": digest,
            "ordered_run_ids": list(
                queue_runner.training_runner.CONTRACTS[
                    contract_id
                ].dedicated_queue_run_ids
            ),
            "manifest_at_creation": queue_runner._file_record(manifest),
        }

    def test_create_freezes_exact_commands_and_rejects_self_rehashed_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "validation-queue"
            output_root = root / "evaluations"
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="ascii")
            records = {
                contract_id: self._training_record(root, contract_id)
                for contract_id in queue_runner.CONTRACT_IDS
            }

            with (
                mock.patch.object(
                    queue_runner,
                    "_training_queue_record",
                    side_effect=lambda _path, contract_id: copy.deepcopy(
                        records[contract_id]
                    ),
                ),
                mock.patch.object(
                    queue_runner, "_evaluation_source_paths", return_value=[source]
                ),
                mock.patch.object(
                    queue_runner, "_recursive_sources", return_value=[source]
                ),
            ):
                queue = queue_runner.create_queue(
                    queue_dir,
                    m0_training_queue=Path(records["M0"]["queue_dir"]),
                    m0n_training_queue=Path(records["M0N"]["queue_dir"]),
                    output_root=output_root,
                    lease_root=root / "leases",
                    gpu_key="0",
                )

            self.assertEqual(queue["plan"]["ordered_run_ids"], list(queue_runner.RUN_IDS))
            self.assertEqual(len(queue["plan"]["items"]), 6)
            spec = queue_dir / queue_runner.AGGREGATION_SPEC_NAME
            for item in queue["plan"]["items"]:
                command = item["command"]
                self.assertEqual(
                    command[0], str(queue_runner.evaluator.DEFAULT_PYTHON.resolve())
                )
                self.assertEqual(command[command.index("--profile") + 1], queue_runner.PROFILE)
                self.assertEqual(command[command.index("--matrix-queue-spec") + 1], str(spec))
                self.assertEqual(command[command.index("--device") + 1], "cuda:0")

            tampered = copy.deepcopy(queue)
            tampered["plan"]["items"][0]["command"].remove("--matrix-queue-spec")
            tampered["plan_sha256"] = queue_runner._canonical_sha(tampered["plan"])
            _write_json(queue_dir / "queue.json", tampered)
            with self.assertRaisesRegex(
                queue_runner.HeadlineValidationQueueError, "item 0 drifted"
            ):
                queue_runner.load_queue(queue_dir)

    def _completed_fixture(self, root: Path) -> tuple[dict, int]:
        queue_dir = root / "queue"
        output = root / "evaluation"
        queue_dir.mkdir()
        output.mkdir()
        spec_path = queue_dir / queue_runner.AGGREGATION_SPEC_NAME
        _write_json(spec_path, {"schema": queue_runner.SPEC_SCHEMA})
        spec_record = {
            **queue_runner._file_record(spec_path),
            "roles": ["matrix_validation_queue_spec"],
        }
        input_rehash = {"schema": "rehash/v1", "status": "passed", "records": []}
        postflight = {
            "schema": queue_runner.evaluator.POSTFLIGHT_SCHEMA,
            "status": "passed",
            "validated_at_utc": "2026-07-19T00:00:00+00:00",
            "profile": queue_runner.PROFILE,
            "evaluation_id": "M0_seed17",
            "input_rehash": input_rehash,
        }
        _write_json(output / "input_rehash.json", input_rehash)
        _write_json(output / "postflight.json", postflight)
        planned = {
            "run_id": "M0:17",
            "contract_id": "M0",
            "train_seed": 17,
            "training_root": str(
                queue_runner.training_runner.CONTRACTS[
                    "M0"
                ].canonical_training_root(17)
            ),
            "evaluation_id": "M0_seed17",
            "evaluation_root": str(output),
        }
        training_queues = [
            {"queue_id": "m0-q", "plan_sha256": "a" * 64},
            {"queue_id": "m0n-q", "plan_sha256": "b" * 64},
        ]
        launch = {
            "schema": queue_runner.evaluator.SCHEMA,
            "status": "completed",
            "evaluation_id": planned["evaluation_id"],
            "output_dir": str(output),
            "matrix_validation_queue_spec": spec_record,
            "protocol": {"profile": queue_runner.PROFILE},
            "inputs": {"records": [spec_record]},
            "source": {
                "training_run_id": planned["run_id"],
                "training_seed": planned["train_seed"],
                "training_run_root": planned["training_root"],
                "training_queue_id": training_queues[0]["queue_id"],
                "training_queue_plan_sha256": training_queues[0]["plan_sha256"],
            },
            "completed_phases": [
                {
                    "phase_id": "validation_calibration",
                    "status": "completed",
                    "returncode": 0,
                }
            ],
            "input_rehash_artifact": {"fixture": "rehash"},
            "postflight_artifact": {"fixture": "postflight"},
            "postflight": postflight,
        }
        _write_json(output / "launch_manifest.json", launch)
        queue = {
            "plan": {
                "queue_dir": str(queue_dir),
                "items": [planned],
                "training_queues": training_queues,
                "evaluation_sources": [],
            }
        }
        return queue, 0

    def test_completed_item_replays_queue_spec_and_exact_artifact_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, index = self._completed_fixture(root)
            output = Path(queue["plan"]["items"][0]["evaluation_root"])
            input_rehash = json.loads(
                (output / "input_rehash.json").read_text(encoding="ascii")
            )
            postflight = json.loads(
                (output / "postflight.json").read_text(encoding="ascii")
            )
            with (
                mock.patch.object(queue_runner, "_launch_source_binding"),
                mock.patch.object(
                    queue_runner.evaluator,
                    "_verify_declared_file",
                    side_effect=[
                        (output / "input_rehash.json").resolve(),
                        (output / "postflight.json").resolve(),
                    ],
                ),
                mock.patch.object(
                    queue_runner.evaluator,
                    "_rehash_inputs",
                    return_value=input_rehash,
                ),
                mock.patch.object(
                    queue_runner.evaluator,
                    "_postflight_screen",
                    return_value=postflight,
                ),
            ):
                evidence = queue_runner._verify_completed(queue, index)
            self.assertEqual(evidence["run_id"], "M0:17")

            launch_path = output / "launch_manifest.json"
            launch = json.loads(launch_path.read_text(encoding="ascii"))
            launch["matrix_validation_queue_spec"]["sha256"] = "0" * 64
            _write_json(launch_path, launch)
            with self.assertRaisesRegex(
                queue_runner.HeadlineValidationQueueError, "hash-bind"
            ):
                queue_runner._verify_completed(queue, index)

    def test_completed_queue_ignores_a_later_queues_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / queue_runner.AGGREGATION_SPEC_NAME
            _write_json(spec, {"schema": queue_runner.SPEC_SCHEMA})
            lease = root / "gpu.json"
            _write_json(lease, {"queue_id": "later-queue"})
            evidence = {"run_id": "M0:17"}
            queue = {
                "status": "completed",
                "plan": {
                    "queue_id": "completed-queue",
                    "queue_dir": str(root),
                    "lease_path": str(lease),
                },
                "plan_sha256": "a" * 64,
                "items": [
                    {
                        "run_id": "M0:17",
                        "status": "completed",
                        "completion_evidence": evidence,
                    }
                ],
            }
            with (
                mock.patch.object(queue_runner, "load_queue", return_value=queue),
                mock.patch.object(queue_runner, "_verify_sources"),
                mock.patch.object(
                    queue_runner, "_verify_completed", return_value=evidence
                ),
                mock.patch.object(queue_runner, "RUN_IDS", ("M0:17",)),
            ):
                result = queue_runner.verify_queue(root)
            self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
