import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_stageb_serial_matrix_queue as queue_runner


FAKE_RUNNER_TEMPLATE = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

RUNNER = __RUNNER__
RUN_IDS = __RUN_IDS__
BEHAVIORS = __BEHAVIORS__


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_after(name):
    return sys.argv[sys.argv.index(name) + 1]


mode = sys.argv[1]
if mode == "list":
    print(json.dumps({"run_ids": RUN_IDS, "rows": [], "seeds": [17, 42, 73]}))
    raise SystemExit(0)

if mode == "run":
    run_id = value_after("--run-id")
    if BEHAVIORS.get(run_id) == "spawn_window_descendant":
        status_path = Path(os.environ["PIVOT_ORCHESTRATION_STATUS"])
        descendant_path = status_path.parent.parent / (
            "spawn_window_descendant.json"
        )
        helper = (
            "import json,subprocess,sys; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(60)'],start_new_session=True); "
            "Path(sys.argv[1]).write_text(json.dumps({'descendant_pid':p.pid}))"
        )
        subprocess.run(
            [sys.executable, "-c", helper, str(descendant_path)],
            check=True,
        )
    time.sleep(60)
    raise SystemExit(0)

if mode == "detach":
    root = Path(value_after("--orchestration-root")).resolve()
    run_id = value_after("--run-id")
    behavior = BEHAVIORS.get(run_id, "success")
    job = root / "job"
    output = root.parent.parent / "fake_outputs" / re.sub(r"[^A-Za-z0-9]+", "_", run_id)
    job.mkdir(parents=True)
    if behavior == "stale_plan":
        output.mkdir(parents=True)
    plan = {
        "status": "planned",
        "run_id": run_id,
        "output_dir": str(output),
        "output_dir_fresh_at_plan": not output.exists(),
    }
    write(job / "plans" / "row" / "seed.json", plan)
    if behavior in {"spawn_window_running", "spawn_window_descendant"}:
        child_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--run-id",
            run_id,
        ]
        status_path = (job / "status.json").resolve()
        launch = {
            "status": "prepared",
            "job_dir": str(job.resolve()),
            "run_ids": [run_id],
            "expected_run_roots": [str(output)],
            "command": child_command,
            "command_shell": shlex.join(child_command),
            "orchestrator_status": str(status_path),
            "runtime": {
                "cuda_visible_devices": os.environ.get(
                    "PIVOT_CUDA_VISIBLE_DEVICES",
                    os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                )
            },
        }
        write(job / "launch.json", launch)
        write(
            status_path,
            {
                "status": "prepared",
                "run_ids": [run_id],
                "expected_run_roots": [str(output)],
                "pid": os.getpid(),
            },
        )
        environment = dict(os.environ)
        environment["PIVOT_ORCHESTRATION_STATUS"] = str(status_path)
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        child = subprocess.Popen(
            child_command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        write(
            root / "spawn_window_child.json",
            {"child_pid": child.pid, "detach_pid": os.getpid()},
        )
        time.sleep(60)
        raise SystemExit(0)
    launch = {
        "status": "launched",
        "run_ids": [run_id],
        "expected_run_roots": [str(output)],
        "child_pid": os.getpid(),
        "runtime": {
            "cuda_visible_devices": os.environ.get(
                "PIVOT_CUDA_VISIBLE_DEVICES",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            )
        },
    }
    write(job / "launch.json", launch)
    phases = (
        ["isolation_probe", "rank", "confidence"]
        if RUNNER.startswith("paper") and run_id.startswith("S3:")
        else ["joint"]
    )
    planned = []
    completed = []
    output.mkdir(parents=True, exist_ok=True)
    for phase_id in phases:
        phase_dir = output / phase_id if len(phases) > 1 else output
        planned.append({"phase_id": phase_id, "output_dir": str(phase_dir)})
        phase_dir.mkdir(parents=True, exist_ok=True)
        if behavior not in {"failed", "running"}:
            postflight = {"status": "passed", "run_id": run_id}
            if RUNNER.startswith("paper"):
                postflight["phase_id"] = phase_id
            postflight_path = phase_dir / "postflight.json"
            if not (behavior == "missing_postflight" and phase_id == phases[-1]):
                write(postflight_path, postflight)
            launch_manifest = {"status": "completed", "run_id": run_id}
            if postflight_path.is_file():
                launch_manifest["postflight_artifact"] = {
                    "path": str(postflight_path),
                    "sha256": sha(postflight_path),
                }
            write(phase_dir / "launch_manifest.json", launch_manifest)
            completed_entry = {
                "phase_id": phase_id,
                "status": "completed",
                "output_dir": str(phase_dir),
            }
            if RUNNER == "token" and postflight_path.is_file():
                completed_entry["postflight"] = {
                    "path": str(postflight_path),
                    "sha256": sha(postflight_path),
                }
            completed.append(completed_entry)
    sequence_status = (
        "failed"
        if behavior == "failed"
        else ("running" if behavior == "running" else "completed")
    )
    sequence = {
        "status": sequence_status,
        "run_id": run_id,
        "output_dir": str(output),
        "phases": planned,
        "completed_phases": [] if behavior in {"failed", "running"} else completed,
    }
    write(output / "sequence_manifest.json", sequence)
    status = {
        "status": sequence_status,
        "run_ids": [run_id],
        "expected_run_roots": [str(output)],
        "completed_run_ids": [] if behavior in {"failed", "running"} else [run_id],
        "pid": os.getpid(),
    }
    write(job / "status.json", status)
    print(json.dumps({"status": "launched", "job_dir": str(job), "pid": os.getpid()}))
    raise SystemExit(0)

if mode in {"status", "reconcile"}:
    job = Path(sys.argv[2]).resolve()
    status = json.loads((job / "status.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "observed_at_utc": "test",
        "job_dir": str(job),
        "persisted_status": status["status"],
        "observed_status": status["status"],
        "reason": "fake_explicit_status",
        "reconciliation_required": False,
        "pid_liveness": {
            "running": status["status"] == "running",
            "state": "alive" if status["status"] == "running" else "exited",
        },
    }))
    raise SystemExit(0)

raise SystemExit(2)
'''


class StageBSerialMatrixQueueTest(unittest.TestCase):
    def _fake_runner(
        self,
        root: Path,
        name: str,
        run_ids: list[str],
        behaviors: dict[str, str] | None = None,
    ) -> Path:
        path = root / f"{name}_runner.py"
        source = (
            FAKE_RUNNER_TEMPLATE.replace("__RUNNER__", repr(name))
            .replace("__RUN_IDS__", repr(run_ids))
            .replace("__BEHAVIORS__", repr(behaviors or {}))
        )
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    def _create(
        self,
        root: Path,
        run_ids: list[str],
        *,
        token_behaviors: dict[str, str] | None = None,
        paper_behaviors: dict[str, str] | None = None,
        queue_name: str = "queue",
    ) -> tuple[Path, Path, Path]:
        token = self._fake_runner(
            root,
            f"token_{queue_name}",
            ["L0:17", "L1:17", "L0:42"],
            token_behaviors,
        )
        paper = self._fake_runner(
            root,
            f"paper_{queue_name}",
            ["D0:17", "S3:17"],
            paper_behaviors,
        )
        queue_dir = root / queue_name
        queue_runner.create_queue(
            queue_dir,
            run_ids=run_ids,
            runner_python=Path(sys.executable),
            token_runner=token,
            paper_runner=paper,
            lease_root=root / "leases",
            gpu_key="0",
        )
        return queue_dir, token, paper

    def _run(self, queue_dir: Path, *, once: bool = False):
        return queue_runner.run_queue(queue_dir, poll_seconds=0.05, once=once)

    def _wait_for_spawn_window(self, launching):
        root = Path(launching["items"][0]["orchestration_root"])
        record_path = root / "spawn_window_child.json"
        deadline = time.monotonic() + 5
        while not record_path.is_file():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        return json.loads(record_path.read_text())

    def _foreign_lease(self, queue, root):
        lease_path = Path(queue["plan"]["lease_path"])
        foreign = json.loads(lease_path.read_text())
        foreign.update(
            queue_id="foreign-queue",
            queue_dir=str(root / "foreign"),
            plan_sha256="f" * 64,
        )
        queue_runner._write_json_atomic(lease_path, foreign)
        return lease_path, foreign

    def _cleanup_spawn_window(self, metadata):
        process_group_ids = [int(metadata["child_pid"])]
        if "descendant_pid" in metadata:
            process_group_ids.insert(0, int(metadata["descendant_pid"]))
        for process_group_id in process_group_ids:
            if queue_runner._process_group_exists(process_group_id) is True:
                try:
                    os.killpg(process_group_id, queue_runner.signal.SIGKILL)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 5
            while queue_runner._process_group_exists(process_group_id) is True:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
        detach_pid = int(metadata["detach_pid"])
        launcher = queue_runner._LOCAL_DETACH_LAUNCHERS.pop(detach_pid, None)
        if launcher is not None and launcher.poll() is None:
            try:
                os.killpg(detach_pid, queue_runner.signal.SIGKILL)
            except ProcessLookupError:
                pass
            launcher.wait(timeout=5)

    def test_real_runner_catalog_covers_all_table_c_b_and_d_run_ids_without_gpu(self):
        snapshot = queue_runner._snapshot_environment()
        environment = queue_runner._runner_environment(snapshot)
        python = Path(sys.executable).resolve()
        token = queue_runner._runner_inventory(
            python, queue_runner.DEFAULT_TOKEN_RUNNER, environment
        )
        paper = queue_runner._runner_inventory(
            python, queue_runner.DEFAULT_PAPER_RUNNER, environment
        )
        self.assertEqual(len(token), 33)
        self.assertEqual(len(paper), 33)
        self.assertIn("L10:73", token)
        for run_id in ("D0:17", "D3m:73", "S0:42", "S3:17", "S2F:73"):
            self.assertIn(run_id, paper)
        self.assertFalse(set(token) & set(paper))

    def test_optional_plan_extension_is_canonical_and_plan_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            token = self._fake_runner(root, "token_extension", ["L0:17"])
            paper = self._fake_runner(root, "paper_extension", ["D0:17"])
            queue_dir = root / "queue"
            queue = queue_runner.create_queue(
                queue_dir,
                run_ids=["L0:17"],
                runner_python=Path(sys.executable),
                token_runner=token,
                paper_runner=paper,
                lease_root=root / "leases",
                gpu_key="0",
                plan_extensions={"fixture": {"version": 1, "values": [3, 2, 1]}},
            )
            self.assertEqual(
                queue["plan"]["extensions"],
                {"fixture": {"values": [3, 2, 1], "version": 1}},
            )
            original_sha = queue["plan_sha256"]
            queue["plan"]["extensions"]["fixture"]["version"] = 2
            (queue_dir / "queue.json").write_text(
                json.dumps(queue, sort_keys=True) + "\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                queue_runner.QueueContractError, "plan SHA-256"
            ):
                queue_runner.load_queue(queue_dir)
            self.assertNotEqual(
                original_sha,
                queue_runner._sha256_bytes(
                    queue_runner._canonical_json_bytes(queue["plan"])
                ),
            )

    def test_success_queue_runs_strictly_in_order_and_verifies_multiphase(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root, ["L0:17", "S3:17", "L1:17"]
            )
            completed = self._run(queue_dir)
            self.assertEqual(completed["status"], "completed", completed.get("failure"))
            self.assertEqual(
                [item["status"] for item in completed["items"]],
                ["completed", "completed", "completed"],
            )
            self.assertEqual(
                [
                    phase["phase_id"]
                    for phase in completed["items"][1]["completion_evidence"][
                        "phases"
                    ]
                ],
                ["isolation_probe", "rank", "confidence"],
            )
            self.assertFalse(Path(completed["plan"]["lease_path"]).exists())
            verification = queue_runner.verify_queue(queue_dir)
            self.assertEqual(verification["status"], "passed")

    def test_restart_recovers_detached_job_without_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(root, ["L0:17"])
            first = self._run(queue_dir, once=True)
            self.assertEqual(first["items"][0]["status"], "reserved")
            second = self._run(queue_dir, once=True)
            self.assertEqual(second["items"][0]["status"], "launching")
            orchestration_root = Path(second["items"][0]["orchestration_root"])
            deadline = time.monotonic() + 5
            while not (orchestration_root / "job" / "status.json").is_file():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
            resumed = self._run(queue_dir)
            self.assertEqual(resumed["status"], "completed", resumed.get("failure"))
            self.assertEqual(
                len(queue_runner._detached_job_dirs(orchestration_root)), 1
            )
            spawned = [
                event
                for event in resumed["events"]
                if event["event"] == "detach_launcher_spawned"
            ]
            self.assertEqual(len(spawned), 1)

    def test_global_gpu_lease_blocks_second_queue_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_one, _, _ = self._create(root, ["L0:17"], queue_name="queue_one")
            queue_two, _, _ = self._create(root, ["L0:17"], queue_name="queue_two")
            first = self._run(queue_one, once=True)
            self.assertEqual(first["items"][0]["status"], "reserved")
            with self.assertRaises(queue_runner.QueueLeaseOwnershipError):
                self._run(queue_two, once=True)
            untouched = queue_runner.load_queue(queue_two)
            self.assertEqual(untouched["status"], "planned")
            self.assertEqual(untouched["items"][0]["status"], "pending")
            self.assertFalse((queue_two / "jobs").exists())

    def test_gpu_lease_identity_rejects_field_and_timestamp_drift(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(root, ["L0:17"])
            reserved = self._run(queue_dir, once=True)
            lease_path = Path(reserved["plan"]["lease_path"])
            original = json.loads(lease_path.read_text())
            mutations = {
                "status": lambda payload: payload.update(status="released"),
                "first_run_id": lambda payload: payload.update(first_run_id="L1:17"),
                "policy": lambda payload: payload.update(policy="release_per_item"),
                "created_at_utc": lambda payload: payload.update(
                    created_at_utc="not-a-timestamp"
                ),
                "extra_key": lambda payload: payload.update(extra=True),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    payload = dict(original)
                    mutate(payload)
                    lease_path.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="ascii",
                    )
                    with self.assertRaises(
                        queue_runner.QueueLeaseOwnershipError
                    ):
                        queue_runner._ensure_lease(
                            reserved, reserved["items"][0], create=False
                        )
            lease_path.write_text(
                json.dumps(original, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )

    def test_active_lease_ownership_loss_terminates_bound_child_group(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(root, ["L0:17"])
            reserved = self._run(queue_dir, once=True)
            sleeper = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            pid = sleeper.pid
            try:
                deadline = time.monotonic() + 5
                identity = queue_runner._read_process_identity(pid)
                while not (
                    identity.get("available") is True
                    and isinstance(identity.get("start_time_ticks"), int)
                    and identity["start_time_ticks"] > 0
                    and identity.get("boot_id")
                    and os.getpgid(pid) == pid
                    and os.getsid(pid) == pid
                ):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                    identity = queue_runner._read_process_identity(pid)
                queue_runner._LOCAL_DETACH_LAUNCHERS[pid] = sleeper
                active = queue_runner.load_queue(queue_dir)
                item = active["items"][0]
                item.update(
                    status="launched",
                    child_pid=pid,
                    child_process_identity=identity,
                    job_dir=str(queue_dir / "synthetic-job"),
                    output_root=str(queue_dir / "synthetic-output"),
                )
                queue_runner._save_queue(active)
                lease_path = Path(reserved["plan"]["lease_path"])
                foreign = json.loads(lease_path.read_text())
                foreign.update(
                    queue_id="foreign-queue",
                    queue_dir=str(root / "foreign"),
                    plan_sha256="f" * 64,
                )
                queue_runner._write_json_atomic(lease_path, foreign)
                failed = self._run(queue_dir, once=True)
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(
                    failed["failure"]["phase"], "lease_ownership_loss"
                )
                self.assertFalse(
                    failed["failure"]["lease_retained_fail_closed"]
                )
                self.assertEqual(
                    failed["items"][0]["child_termination"]["status"],
                    "terminated",
                )
                self.assertEqual(json.loads(lease_path.read_text()), foreign)
                self.assertIsNotNone(sleeper.poll())
                self.assertFalse(queue_runner._process_group_exists(pid))
            finally:
                queue_runner._LOCAL_DETACH_LAUNCHERS.pop(pid, None)
                if sleeper.poll() is None:
                    os.killpg(pid, queue_runner.signal.SIGKILL)
                    sleeper.wait(timeout=5)

    def test_spawn_window_ownership_loss_binds_and_terminates_exact_child(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17"],
                token_behaviors={"L0:17": "spawn_window_running"},
            )
            self._run(queue_dir, once=True)
            launching = self._run(queue_dir, once=True)
            metadata = self._wait_for_spawn_window(launching)
            lease_path, foreign = self._foreign_lease(launching, root)
            try:
                failed = self._run(queue_dir, once=True)
                child_pid = int(metadata["child_pid"])
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(
                    failed["failure"]["phase"], "lease_ownership_loss"
                )
                self.assertEqual(
                    failed["items"][0]["child_pid"], child_pid
                )
                self.assertEqual(
                    failed["items"][0]["child_process_group_id"],
                    child_pid,
                )
                self.assertEqual(
                    failed["items"][0]["child_session_id"], child_pid
                )
                self.assertEqual(
                    failed["items"][0]["spawn_window_launch_status"],
                    "prepared",
                )
                self.assertEqual(
                    failed["items"][0]["child_termination"]["status"],
                    "terminated",
                )
                self.assertFalse(
                    queue_runner._process_group_exists(child_pid)
                )
                self.assertFalse(
                    queue_runner._process_group_exists(
                        int(metadata["detach_pid"])
                    )
                )
                self.assertIn(
                    "detach_launcher_termination", failed["items"][0]
                )
                self.assertEqual(json.loads(lease_path.read_text()), foreign)
            finally:
                self._cleanup_spawn_window(metadata)

    def test_preexisting_unreadable_process_does_not_block_marker_proof(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17"],
                token_behaviors={"L0:17": "spawn_window_running"},
            )
            self._run(queue_dir, once=True)
            launching = self._run(queue_dir, once=True)
            metadata = self._wait_for_spawn_window(launching)
            lease_path, foreign = self._foreign_lease(launching, root)
            real_process_environment = queue_runner._process_environment

            def selectively_unreadable(pid):
                if pid == os.getpid():
                    return None
                return real_process_environment(pid)

            try:
                with patch.object(
                    queue_runner,
                    "_process_environment",
                    side_effect=selectively_unreadable,
                ):
                    failed = self._run(queue_dir, once=True)
                child_pid = int(metadata["child_pid"])
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(
                    failed["failure"]["phase"], "lease_ownership_loss"
                )
                self.assertEqual(
                    failed["items"][0]["child_termination"]["status"],
                    "terminated",
                )
                self.assertFalse(queue_runner._process_group_exists(child_pid))
                self.assertEqual(json.loads(lease_path.read_text()), foreign)
            finally:
                self._cleanup_spawn_window(metadata)

    def test_spawn_window_ownership_loss_preserves_unprovable_identity(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17"],
                token_behaviors={"L0:17": "spawn_window_running"},
            )
            self._run(queue_dir, once=True)
            launching = self._run(queue_dir, once=True)
            metadata = self._wait_for_spawn_window(launching)
            lease_path, foreign = self._foreign_lease(launching, root)
            try:
                with patch.object(
                    queue_runner, "_process_environment", return_value=None
                ), self.assertRaisesRegex(
                    queue_runner.QueueContractError, "remains launching"
                ):
                    self._run(queue_dir, once=True)
                preserved = queue_runner.load_queue(queue_dir)
                child_pid = int(metadata["child_pid"])
                self.assertEqual(preserved["status"], "running")
                self.assertEqual(
                    preserved["items"][0]["status"], "launching"
                )
                self.assertIn(
                    "spawn_window_reconciliation_blocked",
                    preserved["items"][0],
                )
                self.assertTrue(
                    queue_runner._process_group_exists(child_pid)
                )
                self.assertFalse(
                    queue_runner._process_group_exists(
                        int(metadata["detach_pid"])
                    )
                )
                self.assertEqual(json.loads(lease_path.read_text()), foreign)
            finally:
                self._cleanup_spawn_window(metadata)

    def test_spawn_window_ownership_loss_preserves_unproven_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17"],
                token_behaviors={"L0:17": "spawn_window_running"},
            )
            self._run(queue_dir, once=True)
            launching = self._run(queue_dir, once=True)
            metadata = self._wait_for_spawn_window(launching)
            lease_path, foreign = self._foreign_lease(launching, root)
            try:
                with patch.object(
                    queue_runner,
                    "_terminate_bound_child_process_group",
                    side_effect=queue_runner.QueueContractError(
                        "injected unprovable shutdown"
                    ),
                ), self.assertRaisesRegex(
                    queue_runner.QueueContractError, "remains launched"
                ):
                    self._run(queue_dir, once=True)
                preserved = queue_runner.load_queue(queue_dir)
                child_pid = int(metadata["child_pid"])
                self.assertEqual(preserved["status"], "running")
                self.assertEqual(
                    preserved["items"][0]["status"], "launched"
                )
                self.assertIn(
                    "child_termination_blocked", preserved["items"][0]
                )
                self.assertTrue(
                    queue_runner._process_group_exists(child_pid)
                )
                self.assertFalse(
                    queue_runner._process_group_exists(
                        int(metadata["detach_pid"])
                    )
                )
                self.assertEqual(json.loads(lease_path.read_text()), foreign)
            finally:
                self._cleanup_spawn_window(metadata)

    def test_spawn_window_escaped_descendant_preserves_nonterminal_state(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17"],
                token_behaviors={"L0:17": "spawn_window_descendant"},
            )
            self._run(queue_dir, once=True)
            launching = self._run(queue_dir, once=True)
            metadata = self._wait_for_spawn_window(launching)
            descendant_path = (
                Path(launching["items"][0]["orchestration_root"])
                / "spawn_window_descendant.json"
            )
            deadline = time.monotonic() + 5
            while not descendant_path.is_file():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            metadata.update(json.loads(descendant_path.read_text()))
            lease_path, foreign = self._foreign_lease(launching, root)
            try:
                with self.assertRaisesRegex(
                    queue_runner.QueueContractError, "remains launched"
                ):
                    self._run(queue_dir, once=True)
                preserved = queue_runner.load_queue(queue_dir)
                child_pid = int(metadata["child_pid"])
                descendant_pid = int(metadata["descendant_pid"])
                self.assertEqual(preserved["status"], "running")
                self.assertEqual(
                    preserved["items"][0]["status"], "launched"
                )
                self.assertIn(
                    "escaped the sealed child group",
                    preserved["items"][0]["child_termination_blocked"][
                        "termination_error"
                    ],
                )
                self.assertTrue(
                    queue_runner._process_group_exists(child_pid)
                )
                self.assertTrue(
                    queue_runner._process_group_exists(descendant_pid)
                )
                self.assertFalse(
                    queue_runner._process_group_exists(
                        int(metadata["detach_pid"])
                    )
                )
                self.assertEqual(json.loads(lease_path.read_text()), foreign)
            finally:
                self._cleanup_spawn_window(metadata)

    def test_concurrent_launch_publication_is_sealed_before_ownership_failure(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17"],
                token_behaviors={"L0:17": "spawn_window_descendant"},
            )
            self._run(queue_dir, once=True)
            launching = self._run(queue_dir, once=True)
            metadata = self._wait_for_spawn_window(launching)
            orchestration_root = Path(
                launching["items"][0]["orchestration_root"]
            )
            descendant_path = (
                orchestration_root / "spawn_window_descendant.json"
            )
            deadline = time.monotonic() + 5
            while not descendant_path.is_file():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            metadata.update(json.loads(descendant_path.read_text()))
            launch_path = orchestration_root / "job" / "launch.json"
            launch = json.loads(launch_path.read_text())
            child_pid = int(metadata["child_pid"])
            launch.update(
                status="launched",
                child_pid=child_pid,
                child_process_identity=queue_runner._read_process_identity(
                    child_pid
                ),
                child_start_new_session=True,
            )
            queue_runner._write_json_atomic(launch_path, launch)
            lease_path, foreign = self._foreign_lease(launching, root)
            try:
                with self.assertRaisesRegex(
                    queue_runner.QueueContractError, "remains launched"
                ):
                    self._run(queue_dir, once=True)
                preserved = queue_runner.load_queue(queue_dir)
                descendant_pid = int(metadata["descendant_pid"])
                self.assertEqual(preserved["status"], "running")
                self.assertEqual(
                    preserved["items"][0]["status"], "launched"
                )
                self.assertEqual(
                    preserved["items"][0]["child_process_group_id"],
                    child_pid,
                )
                self.assertEqual(
                    preserved["items"][0]["child_session_id"], child_pid
                )
                self.assertIn(
                    "escaped the sealed child group",
                    preserved["items"][0]["child_termination_blocked"][
                        "termination_error"
                    ],
                )
                self.assertTrue(
                    queue_runner._process_group_exists(child_pid)
                )
                self.assertTrue(
                    queue_runner._process_group_exists(descendant_pid)
                )
                self.assertFalse(
                    queue_runner._process_group_exists(
                        int(metadata["detach_pid"])
                    )
                )
                self.assertEqual(json.loads(lease_path.read_text()), foreign)
            finally:
                self._cleanup_spawn_window(metadata)

    def test_unreadable_same_user_marker_process_blocks_terminal_failure(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17"],
                token_behaviors={"L0:17": "spawn_window_descendant"},
            )
            self._run(queue_dir, once=True)
            launching = self._run(queue_dir, once=True)
            metadata = self._wait_for_spawn_window(launching)
            descendant_path = (
                Path(launching["items"][0]["orchestration_root"])
                / "spawn_window_descendant.json"
            )
            deadline = time.monotonic() + 5
            while not descendant_path.is_file():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            metadata.update(json.loads(descendant_path.read_text()))
            descendant_pid = int(metadata["descendant_pid"])
            lease_path, foreign = self._foreign_lease(launching, root)
            real_process_environment = queue_runner._process_environment

            def selectively_unreadable(pid):
                if pid == descendant_pid:
                    return None
                return real_process_environment(pid)

            try:
                with patch.object(
                    queue_runner,
                    "_process_environment",
                    side_effect=selectively_unreadable,
                ), self.assertRaisesRegex(
                    queue_runner.QueueContractError, "remains launched"
                ):
                    self._run(queue_dir, once=True)
                preserved = queue_runner.load_queue(queue_dir)
                self.assertEqual(preserved["status"], "running")
                self.assertEqual(
                    preserved["items"][0]["status"], "launched"
                )
                self.assertIn(
                    "same-user process environment is unreadable",
                    preserved["items"][0]["child_termination_blocked"][
                        "termination_error"
                    ],
                )
                self.assertTrue(
                    queue_runner._process_group_exists(descendant_pid)
                )
                self.assertEqual(json.loads(lease_path.read_text()), foreign)
            finally:
                self._cleanup_spawn_window(metadata)

    def test_running_child_never_advances_or_launches_next_item(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17", "L1:17"],
                token_behaviors={"L0:17": "running"},
            )
            for _ in range(5):
                state = self._run(queue_dir, once=True)
                if state["items"][0]["status"] == "launched":
                    break
                time.sleep(0.05)
            self.assertEqual(state["items"][0]["status"], "launched")
            for _ in range(3):
                state = self._run(queue_dir, once=True)
            self.assertEqual(state["status"], "running")
            self.assertEqual(
                [item["status"] for item in state["items"]],
                ["launched", "pending"],
            )
            self.assertFalse((queue_dir / "jobs" / "001-L1_17").exists())

    def test_explicit_child_failure_stops_before_second_item_and_retains_lease(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17", "L1:17"],
                token_behaviors={"L0:17": "failed"},
            )
            failed = self._run(queue_dir)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(
                [item["status"] for item in failed["items"]], ["failed", "pending"]
            )
            self.assertTrue(Path(failed["plan"]["lease_path"]).is_file())
            self.assertFalse(Path(failed["plan"]["queue_dir"], "jobs", "001-L1_17").exists())

    def test_completed_status_without_postflight_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17", "L1:17"],
                token_behaviors={"L0:17": "missing_postflight"},
            )
            failed = self._run(queue_dir)
            self.assertEqual(failed["status"], "failed")
            self.assertIn("postflight", failed["items"][0]["failure_error"])
            self.assertEqual(failed["items"][1]["status"], "pending")

    def test_preflight_without_fresh_output_proof_fails_before_binding(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17", "L1:17"],
                token_behaviors={"L0:17": "stale_plan"},
            )
            failed = self._run(queue_dir)
            self.assertEqual(failed["status"], "failed")
            self.assertIn(
                "fresh output", failed["items"][0]["failure_error"]
            )
            self.assertEqual(failed["items"][1]["status"], "pending")

    def test_status_is_byte_for_byte_read_only(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(
                root,
                ["L0:17", "L1:17"],
                token_behaviors={"L0:17": "running"},
            )
            for _ in range(6):
                state = self._run(queue_dir, once=True)
                if state["items"][0]["status"] == "launched":
                    break
                time.sleep(0.05)
            self.assertEqual(state["items"][0]["status"], "launched")
            queue_path = queue_dir / "queue.json"
            child_status = Path(state["items"][0]["job_dir"]) / "status.json"
            before_queue = queue_path.read_bytes()
            before_child = child_status.read_bytes()
            report = queue_runner.queue_status(queue_dir)
            self.assertEqual(report["detached_job_observation"]["observed_status"], "running")
            self.assertEqual(queue_path.read_bytes(), before_queue)
            self.assertEqual(child_status.read_bytes(), before_child)

    def test_runner_source_drift_fails_before_detach(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, token, _ = self._create(root, ["L0:17"])
            token.write_text(token.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            failed = self._run(queue_dir, once=True)
            self.assertEqual(failed["status"], "failed")
            self.assertIn("runner source changed", failed["failure"]["error"])
            self.assertFalse((queue_dir / "jobs").exists())

    def test_immutable_plan_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            queue_dir, _, _ = self._create(root, ["L0:17"])
            queue_path = queue_dir / "queue.json"
            payload = json.loads(queue_path.read_text(encoding="utf-8"))
            payload["plan"]["gpu_key"] = "1"
            queue_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                queue_runner.QueueContractError, "plan SHA-256 mismatch"
            ):
                queue_runner.load_queue(queue_dir)

    def test_create_rejects_duplicate_unknown_and_multi_gpu_selection(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0"}
        ):
            root = Path(temporary)
            token = self._fake_runner(root, "token", ["L0:17"])
            paper = self._fake_runner(root, "paper", ["D0:17"])
            common = dict(
                runner_python=Path(sys.executable),
                token_runner=token,
                paper_runner=paper,
                lease_root=root / "leases",
                gpu_key="0",
            )
            with self.assertRaisesRegex(queue_runner.QueueContractError, "duplicate"):
                queue_runner.create_queue(
                    root / "duplicate", run_ids=["L0:17", "l0:17"], **common
                )
            with self.assertRaisesRegex(queue_runner.QueueContractError, "not registered"):
                queue_runner.create_queue(
                    root / "unknown", run_ids=["L9:17"], **common
                )
        with patch.dict(os.environ, {"PIVOT_CUDA_VISIBLE_DEVICES": "0,1"}):
            snapshot = queue_runner._snapshot_environment()
            with self.assertRaisesRegex(queue_runner.QueueContractError, "exactly one"):
                queue_runner._gpu_key_from_environment(snapshot, None)


if __name__ == "__main__":
    unittest.main()
