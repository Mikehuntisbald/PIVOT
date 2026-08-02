import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_stageb_table_a_g0c_soak_queue as soak_queue


def _native_plan(source: Path, output_root: Path) -> dict:
    source_record = {
        "path": str(source.resolve()),
        "sha256": soak_queue._sha256_file(source),
    }
    plan = {
        "schema": soak_queue.training_runner.PLAN_SCHEMA,
        "row_id": "G0c",
        "purpose": "soak",
        "matched_contract": {
            "seed": soak_queue.SEED,
            "micro_batch_size_per_rank": soak_queue.MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": (
                soak_queue.GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_global_batch": soak_queue.EFFECTIVE_GLOBAL_BATCH,
            "optimizer_updates": soak_queue.OPTIMIZER_UPDATES,
        },
        "inputs": {"fixture": source_record},
        "source_dependency_tree": {"records": [source_record]},
        "output_dir": str(output_root.resolve()),
        "command": ["fixture"],
    }
    plan["plan_sha256"] = soak_queue.training_runner._plan_sha256(plan)
    return plan


def _fixture_plan(
    root: Path,
    *,
    name: str = "soak",
    lease_root: Path | None = None,
    gpu_key: str = "fixture-gpu",
) -> dict:
    queue_dir = root / f"{name}-queue"
    artifact_root = root / f"{name}-artifacts"
    output_root = artifact_root / "output"
    plan_path = artifact_root / "plan.json"
    seal_path = artifact_root / "seal.json"
    source = root / f"{name}-source.txt"
    source.write_text(f"{name} source\n", encoding="ascii")
    python = Path(sys.executable).resolve()
    runtime = {
        "python": str(python),
        "python_record": soak_queue._file_record(
            python, "queue_python_runtime"
        ),
        "data_root": str(root.resolve()),
        "gpu_key": gpu_key,
        "cuda_visible_devices": gpu_key,
        "num_workers": 0,
        "seed": soak_queue.SEED,
        "micro_batch_size": soak_queue.MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": (
            soak_queue.GRADIENT_ACCUMULATION_STEPS
        ),
        "effective_global_batch": soak_queue.EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": soak_queue.OPTIMIZER_UPDATES,
    }
    native = _native_plan(source, output_root)
    selected_lease_root = lease_root or (root / "leases")
    return {
        "schema": soak_queue.PLAN_SCHEMA,
        "queue_id": f"fixture-{name}",
        "created_at_utc": "2026-07-19T00:00:00+00:00",
        "queue_dir": str(queue_dir.resolve()),
        "repository_root": str(soak_queue.REPO_ROOT),
        "runtime": runtime,
        "runtime_environment": soak_queue._runtime_environment(runtime),
        "gpu_key": gpu_key,
        "lease_root": str(selected_lease_root.resolve()),
        "lease_path": str(
            soak_queue.shared_queue._lease_path(selected_lease_root, gpu_key)
        ),
        "controller_sources": [
            soak_queue._file_record(source, "fixture_controller")
        ],
        "items": [
            {
                "index": 0,
                "run_id": soak_queue.RUN_ID,
                "item_kind": "g0c_soak",
                "seed": soak_queue.SEED,
                "micro_batch_size": soak_queue.MICRO_BATCH_SIZE,
                "gradient_accumulation_steps": (
                    soak_queue.GRADIENT_ACCUMULATION_STEPS
                ),
                "effective_global_batch": soak_queue.EFFECTIVE_GLOBAL_BATCH,
                "optimizer_updates": soak_queue.OPTIMIZER_UPDATES,
                "output_root": str(output_root.resolve()),
                "soak_plan_path": str(plan_path.resolve()),
                "soak_seal_path": str(seal_path.resolve()),
                "expected_plan": native,
                "expected_plan_sha256": native["plan_sha256"],
                "training_command": [
                    str(python),
                    str(Path(soak_queue.training_runner.__file__).resolve()),
                    "run",
                    "--purpose",
                    "soak",
                ],
                "input_records": [
                    soak_queue._file_record(source, "fixture_input")
                ],
            }
        ],
    }


def _create_fixture(plan: dict) -> dict:
    with mock.patch.object(soak_queue, "_validate_creation_attestation"):
        return soak_queue.create_queue_from_plan(plan)


def _dead_identity(pid: int = 987654) -> dict:
    return {
        "pid": pid,
        "available": True,
        "state": "R",
        "start_time_ticks": 123456,
        "boot_id": "fixture-boot-id",
    }


def _publish_child_record(
    queue: dict, *, status: str = "completed", returncode: int = 0
) -> dict:
    item = queue["items"][0]
    pid = 987654
    identity = _dead_identity(pid)
    child = {
        "schema": soak_queue.CHILD_STATUS_SCHEMA,
        "status": status,
        **soak_queue._job_identity(queue, item),
        "child_pid": pid,
        "process_identity": identity,
        "process_group_id": pid,
        "session_id": pid,
        "training_command": copy.deepcopy(
            soak_queue._planned_item(queue)["training_command"]
        ),
        "started_at_utc": "2026-07-19T00:00:01+00:00",
        "returncode": returncode,
        "finished_at_utc": "2026-07-19T00:00:02+00:00",
    }
    soak_queue._write_json(
        soak_queue._job_dir(queue, item) / "child.json", child
    )
    return child


def _advance_to_launching(plan: dict) -> dict:
    _create_fixture(plan)
    soak_queue.advance_once(Path(plan["queue_dir"]))
    return soak_queue.advance_once(Path(plan["queue_dir"]))


def _bind_dead_completed_child(plan: dict) -> dict:
    launching = _advance_to_launching(plan)
    _publish_child_record(launching)
    with mock.patch.object(
        soak_queue.shared_queue, "_process_running", return_value=False
    ):
        return soak_queue.advance_once(Path(plan["queue_dir"]))


def _write_native_outputs(plan: dict) -> dict:
    item = plan["items"][0]
    output = Path(item["output_root"])
    output.mkdir(parents=True)
    Path(item["soak_plan_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(item["soak_plan_path"]).write_text(
        json.dumps(item["expected_plan"]), encoding="utf-8"
    )
    (output / "checkpoint_iter.pth").write_bytes(b"fixture checkpoint")
    replay = {
        "schema": soak_queue.training_runner.POSTFLIGHT_SCHEMA,
        "status": "PASS",
        "purpose": "soak",
        "plan_sha256": item["expected_plan_sha256"],
        "validated_at_utc": "replay",
    }
    (output / "postflight.json").write_text(
        json.dumps({**replay, "validated_at_utc": "persisted"}),
        encoding="utf-8",
    )
    return replay


class G0cSoakQueueTest(unittest.TestCase):
    def test_plan_contract_is_exact_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            soak_queue._validate_plan(plan, Path(plan["queue_dir"]))
            for key, value in (
                ("seed", 42),
                ("micro_batch_size", 11),
                ("gradient_accumulation_steps", 5),
                ("effective_global_batch", 44),
                ("optimizer_updates", 49),
            ):
                changed = copy.deepcopy(plan)
                changed["items"][0][key] = value
                with self.assertRaisesRegex(
                    soak_queue.G0cSoakQueueError, "item contract"
                ):
                    soak_queue._validate_plan(
                        changed, Path(changed["queue_dir"])
                    )

    def test_creation_is_fresh_only_and_never_acquires_the_gpu_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for artifact in ("soak_plan_path", "output_root", "soak_seal_path"):
                plan = _fixture_plan(root, name=artifact)
                path = Path(plan["items"][0][artifact])
                if artifact == "output_root":
                    path.mkdir(parents=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("old artifact\n", encoding="ascii")
                with (
                    mock.patch.object(
                        soak_queue, "_validate_creation_attestation"
                    ),
                    self.assertRaisesRegex(
                        soak_queue.G0cSoakQueueError, "no adoption"
                    ),
                ):
                    soak_queue.create_queue_from_plan(plan)

            plan = _fixture_plan(root, name="fresh")
            queue = _create_fixture(plan)
            self.assertEqual(queue["status"], "planned")
            self.assertFalse(Path(plan["lease_path"]).exists())

    def test_shared_lease_blocks_another_queue_before_any_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lease_root = root / "shared-leases"
            first = _fixture_plan(root, name="first", lease_root=lease_root)
            second = _fixture_plan(root, name="second", lease_root=lease_root)
            _create_fixture(first)
            _create_fixture(second)
            with mock.patch.object(soak_queue.subprocess, "Popen") as spawn:
                reserved = soak_queue.advance_once(Path(first["queue_dir"]))
                with self.assertRaises(soak_queue.G0cSoakQueueBusy):
                    soak_queue.advance_once(Path(second["queue_dir"]))
            spawn.assert_not_called()
            self.assertEqual(reserved["items"][0]["status"], "reserved")
            self.assertEqual(
                soak_queue.load_queue(Path(second["queue_dir"]))["status"],
                "planned",
            )
            lease = json.loads(
                Path(first["lease_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(lease["queue_id"], first["queue_id"])
            self.assertEqual(lease["first_run_id"], soak_queue.RUN_ID)

    def test_launching_state_recovers_exact_dead_child_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            launching = _advance_to_launching(plan)
            child = _publish_child_record(launching)
            with (
                mock.patch.object(
                    soak_queue.shared_queue, "_process_running", return_value=False
                ),
                mock.patch.object(soak_queue.subprocess, "Popen") as spawn,
            ):
                launched = soak_queue.advance_once(Path(plan["queue_dir"]))
            spawn.assert_not_called()
            self.assertEqual(launched["items"][0]["status"], "launched")
            self.assertEqual(launched["items"][0]["child_pid"], child["child_pid"])
            self.assertEqual(
                launched["items"][0]["child_process_identity"]["boot_id"],
                "fixture-boot-id",
            )

    def test_prepare_is_idempotent_after_state_commit_interruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            _create_fixture(plan)
            reserved = soak_queue.advance_once(Path(plan["queue_dir"]))
            original_save = soak_queue._save_queue
            with (
                mock.patch.object(
                    soak_queue,
                    "_save_queue",
                    side_effect=KeyboardInterrupt("simulated supervisor stop"),
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                soak_queue.advance_once(Path(plan["queue_dir"]))
            persisted = soak_queue.load_queue(Path(plan["queue_dir"]))
            self.assertEqual(persisted["items"][0]["status"], "reserved")
            job_dir = Path(reserved["items"][0]["job_dir"])
            self.assertEqual(
                {path.name for path in job_dir.iterdir()},
                {"launch.json", "status.json", "seal_intent.json"},
            )
            with mock.patch.object(
                soak_queue, "_save_queue", side_effect=original_save
            ):
                recovered = soak_queue.advance_once(Path(plan["queue_dir"]))
            self.assertEqual(recovered["items"][0]["status"], "launching")

    def test_running_child_waits_for_identity_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            launching = _advance_to_launching(plan)
            pid = 987654
            with (
                mock.patch.object(
                    soak_queue.shared_queue, "_process_running", return_value=True
                ),
                mock.patch.object(
                    soak_queue.shared_queue,
                    "_process_group_exists",
                    return_value=True,
                ),
                mock.patch.object(os, "getpgid", return_value=pid),
                mock.patch.object(os, "getsid", return_value=pid),
            ):
                soak_queue._bind_child(launching, pid, _dead_identity(pid))
                observed = soak_queue.advance_once(Path(plan["queue_dir"]))
            self.assertEqual(observed["status"], "running")
            self.assertEqual(observed["items"][0]["status"], "launched")
            self.assertEqual(
                observed["items"][0]["last_observation"]["child_status"],
                "publication_pending",
            )

    def test_completion_publishes_fresh_seal_then_reloads_before_lease_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            _bind_dead_completed_child(plan)
            replay = _write_native_outputs(plan)
            seal_path = Path(plan["items"][0]["soak_seal_path"])
            seal = {"schema": "fixture-seal", "status": "sealed"}
            release_observations = []
            original_clear = soak_queue._clear_lease

            def clear_after_durable_reload(queue):
                durable = json.loads(
                    (Path(plan["queue_dir"]) / "queue.json").read_text(
                        encoding="utf-8"
                    )
                )
                release_observations.append(
                    (
                        durable["status"],
                        durable["items"][0]["status"],
                        "completion_evidence" in durable["items"][0],
                    )
                )
                return original_clear(queue)

            def validate_seal(path):
                self.assertEqual(path.resolve(), seal_path.resolve())
                self.assertTrue(path.is_file())
                return {
                    "path": str(path.resolve()),
                    "plan": plan["items"][0]["expected_plan"],
                }

            with (
                mock.patch.object(
                    soak_queue.shared_queue, "_process_running", return_value=False
                ),
                mock.patch.object(
                    soak_queue.shared_queue,
                    "_process_group_exists",
                    return_value=False,
                ),
                mock.patch.object(
                    soak_queue.training_runner,
                    "verify_checkpoint",
                    return_value=replay,
                ),
                mock.patch.object(
                    soak_queue.training_runner,
                    "build_soak_seal",
                    return_value=seal,
                ) as build_seal,
                mock.patch.object(
                    soak_queue.training_runner,
                    "_validate_soak_seal",
                    side_effect=validate_seal,
                ),
                mock.patch.object(
                    soak_queue, "_clear_lease", side_effect=clear_after_durable_reload
                ),
            ):
                completed = soak_queue.advance_once(Path(plan["queue_dir"]))
                verification = soak_queue.verify_queue(Path(plan["queue_dir"]))
                seal_path.write_text(
                    '{"status":"tampered-after-completion"}\n', encoding="ascii"
                )
                tampered = soak_queue.verify_queue(Path(plan["queue_dir"]))

            build_seal.assert_called_once()
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(verification["status"], "passed")
            self.assertEqual(tampered["status"], "failed")
            self.assertIn("differs from replay", tampered["errors"][0])
            self.assertTrue(seal_path.is_file())
            self.assertFalse(Path(plan["lease_path"]).exists())
            self.assertEqual(release_observations, [("completed", "completed", True)])

    def test_crash_recovery_accepts_only_queue_authorized_existing_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            _bind_dead_completed_child(plan)
            replay = _write_native_outputs(plan)
            seal_path = Path(plan["items"][0]["soak_seal_path"])
            seal_path.parent.mkdir(parents=True, exist_ok=True)
            seal_path.write_text('{"status":"sealed"}\n', encoding="ascii")

            with (
                mock.patch.object(
                    soak_queue.shared_queue, "_process_running", return_value=False
                ),
                mock.patch.object(
                    soak_queue.shared_queue,
                    "_process_group_exists",
                    return_value=False,
                ),
                mock.patch.object(
                    soak_queue.training_runner,
                    "verify_checkpoint",
                    return_value=replay,
                ),
                mock.patch.object(
                    soak_queue.training_runner, "build_soak_seal"
                ) as build_seal,
                mock.patch.object(
                    soak_queue.training_runner,
                    "_validate_soak_seal",
                    return_value={
                        "path": str(seal_path.resolve()),
                        "plan": plan["items"][0]["expected_plan"],
                    },
                ),
            ):
                completed = soak_queue.advance_once(Path(plan["queue_dir"]))
            build_seal.assert_not_called()
            self.assertEqual(completed["status"], "completed")
            evidence = completed["items"][0]["completion_evidence"]
            self.assertEqual(
                evidence["seal_intent"]["path"],
                str(
                    Path(completed["items"][0]["job_dir"])
                    / "seal_intent.json"
                ),
            )

    def test_partial_terminal_job_commit_recovers_from_completion_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            _bind_dead_completed_child(plan)
            replay = _write_native_outputs(plan)
            seal_path = Path(plan["items"][0]["soak_seal_path"])
            seal = {"schema": "fixture-seal", "status": "sealed"}
            original_write = soak_queue._write_json

            def interrupt_status_commit(path, value):
                if path.name == "status.json" and value.get("status") == "completed":
                    raise KeyboardInterrupt("simulated terminal-record interruption")
                return original_write(path, value)

            def validate_seal(path):
                return {
                    "path": str(path.resolve()),
                    "plan": plan["items"][0]["expected_plan"],
                }

            with (
                mock.patch.object(
                    soak_queue.shared_queue, "_process_running", return_value=False
                ),
                mock.patch.object(
                    soak_queue.shared_queue,
                    "_process_group_exists",
                    return_value=False,
                ),
                mock.patch.object(
                    soak_queue.training_runner,
                    "verify_checkpoint",
                    return_value=replay,
                ),
                mock.patch.object(
                    soak_queue.training_runner,
                    "build_soak_seal",
                    return_value=seal,
                ),
                mock.patch.object(
                    soak_queue.training_runner,
                    "_validate_soak_seal",
                    side_effect=validate_seal,
                ),
            ):
                with (
                    mock.patch.object(
                        soak_queue,
                        "_write_json",
                        side_effect=interrupt_status_commit,
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    soak_queue.advance_once(Path(plan["queue_dir"]))

                interrupted = soak_queue.load_queue(Path(plan["queue_dir"]))
                job_dir = Path(interrupted["items"][0]["job_dir"])
                launch = json.loads(
                    (job_dir / "launch.json").read_text(encoding="utf-8")
                )
                status = json.loads(
                    (job_dir / "status.json").read_text(encoding="utf-8")
                )
                self.assertEqual(interrupted["items"][0]["status"], "launched")
                self.assertEqual(launch["status"], "completed")
                self.assertEqual(status["status"], "running")
                self.assertTrue((job_dir / "completion_candidate.json").is_file())
                self.assertTrue(seal_path.is_file())

                recovered = soak_queue.advance_once(Path(plan["queue_dir"]))
            self.assertEqual(recovered["status"], "completed")
            self.assertFalse(Path(plan["lease_path"]).exists())

    def test_source_drift_fails_closed_and_retains_owned_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            _create_fixture(plan)
            reserved = soak_queue.advance_once(Path(plan["queue_dir"]))
            Path(plan["controller_sources"][0]["path"]).write_text(
                "drifted source\n", encoding="ascii"
            )
            failed = soak_queue.advance_once(Path(plan["queue_dir"]))
            self.assertEqual(reserved["items"][0]["status"], "reserved")
            self.assertEqual(failed["status"], "failed")
            self.assertIn("changed after queue planning", failed["failure"]["error"])
            self.assertTrue(failed["failure"]["lease_retained_fail_closed"])
            self.assertTrue(Path(plan["lease_path"]).is_file())

    def test_active_lease_loss_fences_exact_child_before_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            launching = _advance_to_launching(plan)
            child = _publish_child_record(
                launching, status="running", returncode=0
            )
            with (
                mock.patch.object(
                    soak_queue.shared_queue, "_process_running", return_value=True
                ),
                mock.patch.object(os, "getpgid", return_value=child["child_pid"]),
                mock.patch.object(os, "getsid", return_value=child["child_pid"]),
            ):
                launched = soak_queue.advance_once(Path(plan["queue_dir"]))
            self.assertEqual(launched["items"][0]["status"], "launched")

            Path(plan["lease_path"]).write_text(
                json.dumps({"schema": "foreign", "queue_id": "foreign"}),
                encoding="utf-8",
            )
            termination = {"status": "terminated", "pid": child["child_pid"]}
            with mock.patch.object(
                soak_queue.shared_queue,
                "_terminate_exact_process_group",
                return_value=termination,
            ) as terminate:
                failed = soak_queue.advance_once(Path(plan["queue_dir"]))
            terminate.assert_called_once()
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(
                failed["items"][0]["child_termination"], termination
            )
            self.assertEqual(failed["failure"]["phase"], "lease_ownership_loss")
            self.assertFalse(failed["failure"]["lease_retained_fail_closed"])

    def test_execute_child_publishes_identity_before_mocked_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = _fixture_plan(Path(temporary))
            launching = _advance_to_launching(plan)
            pid = os.getpid()
            identity = {
                "pid": pid,
                "available": True,
                "state": "R",
                "start_time_ticks": 999,
                "boot_id": "fixture-boot",
            }
            with (
                mock.patch.object(
                    soak_queue.shared_queue,
                    "_read_process_identity",
                    return_value=identity,
                ),
                mock.patch.object(os, "getpgid", return_value=pid),
                mock.patch.object(os, "getsid", return_value=pid),
                mock.patch.object(
                    soak_queue.subprocess,
                    "run",
                    return_value=mock.Mock(returncode=0),
                ) as run,
            ):
                code = soak_queue.execute_child(
                    Path(plan["queue_dir"]), launching["items"][0]["job_id"]
                )
            self.assertEqual(code, 0)
            run.assert_called_once()
            child = json.loads(
                (
                    Path(launching["items"][0]["job_dir"]) / "child.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(child["status"], "completed")
            self.assertEqual(child["returncode"], 0)
            self.assertEqual(child["process_identity"]["boot_id"], "fixture-boot")

    def test_audit_labels_legacy_plans_non_adoptable_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plans = root / "outputs/paper_cvpr_v1/plans"
            plans.mkdir(parents=True)
            legacy = plans / "table_a_g0c_b10a4_u50_old.json"
            legacy.write_text("{}\n", encoding="ascii")
            canonical_plan = plans / "table_a_g0c_u50_v2.json"
            canonical_root = root / "soak"
            canonical_seal = root / "seal.json"
            canonical_queue = root / "queue"
            with (
                mock.patch.object(soak_queue, "REPO_ROOT", root),
                mock.patch.object(soak_queue, "DEFAULT_QUEUE_DIR", canonical_queue),
                mock.patch.object(
                    soak_queue.training_runner,
                    "DEFAULT_SOAK_PLAN",
                    canonical_plan,
                ),
                mock.patch.object(
                    soak_queue.training_runner,
                    "DEFAULT_SOAK_ROOT",
                    canonical_root,
                ),
                mock.patch.object(
                    soak_queue.training_runner,
                    "DEFAULT_SOAK_SEAL",
                    canonical_seal,
                ),
            ):
                audit = soak_queue.audit_existing_artifacts()
            self.assertTrue(audit["canonical_artifacts_fresh"])
            self.assertFalse(audit["legacy_adoption_allowed"])
            self.assertEqual(audit["legacy_artifacts"][0]["path"], str(legacy))
            self.assertEqual(audit["legacy_artifacts"][0]["status"], "non_adoptable")
            self.assertEqual(legacy.read_text(encoding="ascii"), "{}\n")


if __name__ == "__main__":
    unittest.main()
