import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


from tools import run_stageb_table_b_v2 as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeProcess:
    def __init__(self, pid):
        self.pid = pid

    def terminate(self):
        return None


class TableBV2RunnerTest(unittest.TestCase):
    def _runtime(self, root):
        launcher = runner._launcher()
        stage_a = root / "stage_a.pth"
        scorer = root / "scorer.pth"
        stage_a.write_bytes(b"stage-a")
        scorer.write_bytes(b"scorer")
        return launcher.Runtime(
            python=Path(sys.executable).resolve(),
            stage_a_init=stage_a,
            scorer_warmstart=scorer,
            tn_output_root=root / "runs",
            score_output_root=root / "score",
            data_root=Path(os.environ.get("DATA_ROOT", "/media/haoyi/T9/data")),
            batch_size=40,
            total_train_iters=1000,
            iter_checkpoint_interval=1000,
            num_workers=0,
            prefetch_factor=1,
            omp_num_threads=1,
            min_nofile=0,
            cuda_visible_devices="0",
            mp_sharing_strategy="none",
            gradient_diagnostic_interval=100,
        )

    def _manifest(self, root, table_b_id="D2m", seed=17):
        launcher = runner._launcher()
        runtime = self._runtime(root)
        row = launcher.ROW_BY_ID[table_b_id]
        return runner.build_manifest(
            runtime, row, seed, launcher.token_launcher.HashCache()
        ), runtime, row

    def test_import_does_not_load_training_stack(self):
        script = (
            "import sys; import tools.run_stageb_table_b_v2; "
            "assert 'main' not in sys.modules; "
            "assert 'engine' not in sys.modules; "
            "assert 'datasets' not in sys.modules; "
            "assert 'datasets.patch_episode' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_inventory_is_only_six_v2_runs(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(runner.__file__)), "list", "--json"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(
            payload["run_ids"],
            [
                "D2m:17",
                "D2m:42",
                "D2m:73",
                "D3m:17",
                "D3m:42",
                "D3m:73",
            ],
        )
        self.assertEqual(payload["phases"], ["joint"])

    def test_private_runtime_does_not_mutate_canonical_legacy_launcher(self):
        from tools import run_stageb_paper_ablation_matrices as legacy

        rows_before = legacy.ROWS
        audit_before = legacy.MATCHED_TABLE_B_AUDIT
        runner._launcher()
        self.assertIs(legacy.ROWS, rows_before)
        self.assertEqual(legacy.MATCHED_TABLE_B_AUDIT, audit_before)
        self.assertNotEqual(legacy.ROWS, runner._launcher().ROWS)

    def test_manifest_bootstraps_through_this_runner_and_binds_every_phase_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _ = self._manifest(Path(temporary))
            phase = manifest["phases"][0]
            scope_sha = manifest["table_b_v2_scope_sha256"]
            self.assertEqual(manifest["phase_id"], "joint")
            self.assertEqual(manifest["v2_provenance"]["phase_id"], "joint")
            self.assertEqual(phase["phase_id"], "joint")
            self.assertEqual(phase["phase"]["phase_id"], "joint")
            self.assertEqual(phase["table_b_v2_scope"]["phase_id"], "joint")
            self.assertEqual(phase["v2_provenance"]["scope_sha256"], scope_sha)
            command = phase["command"]
            self.assertEqual(Path(command[1]).resolve(), Path(runner.__file__).resolve())
            self.assertEqual(command[2], "_bootstrap-main")
            self.assertNotIn(str(REPO_ROOT / "main.py"), command[: command.index("--")])
            self.assertIn(
                f"stage_b_v2_scope_contract_sha256={scope_sha}", command
            )
            self.assertIn("stage_b_v2_phase_id=joint", command)
            self.assertIn(
                f"stage_b_v19_table_b_audit_sha256={runner.V2_AUDIT_SHA256}",
                command,
            )

    def test_completed_sequence_rejects_nested_phase_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _ = self._manifest(Path(temporary))
            manifest["status"] = "completed"
            manifest["completed_phases"] = [
                {
                    "phase_id": "joint",
                    "status": "completed",
                    "profile": manifest["profile"],
                    "formal_queue": manifest["formal_queue"],
                    "table_b_v2_scope_sha256": manifest[
                        "table_b_v2_scope_sha256"
                    ],
                    "v2_provenance": {
                        "phase_id": "joint",
                        "scope_sha256": manifest["table_b_v2_scope_sha256"],
                        "profile": manifest["profile"],
                        "queue": manifest["formal_queue"],
                        "source_plan_semantic_sha256": manifest[
                            "table_b_v2_scope"
                        ]["source_plan_semantic_sha256"],
                    },
                }
            ]
            runner._validate_completed_sequence(manifest)
            bad = copy.deepcopy(manifest)
            bad["completed_phases"][0]["v2_provenance"]["phase_id"] = "final"
            with self.assertRaisesRegex(runner.TableBV2RunnerError, "joint v2 provenance"):
                runner._validate_completed_sequence(bad)

    def test_bootstrap_rejects_unowned_child_before_importing_main(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self._manifest(root)
            output = Path(manifest["output_dir"])
            output.mkdir(parents=True)
            launch = manifest["phases"][0]
            launch["status"] = "running"
            launch["runner_owner"]["process_identity"] = {
                "available": True,
                "pid": 1,
                "start_time_ticks": 1,
                "boot_id": "wrong",
            }
            path = output / "launch_manifest.json"
            runner._write_json_atomic(path, launch)
            command = [
                sys.executable,
                "-B",
                str(Path(runner.__file__)),
                "_bootstrap-main",
                "--launch-manifest",
                str(path),
                "--scope-sha256",
                manifest["table_b_v2_scope_sha256"],
                "--",
                "--help",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not owned by the bound v2 runner", completed.stderr)

    def test_bootstrap_installs_v2_contract_before_importing_training_entry(self):
        from util import stage_b_table_b_v2_contract as v2_contract

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self._manifest(root)
            output = Path(manifest["output_dir"])
            output.mkdir(parents=True)
            launch = manifest["phases"][0]
            launch["status"] = "running"
            launch["runner_owner"]["process_identity"] = runner._read_process_identity(
                os.getppid()
            )
            path = output / "launch_manifest.json"
            runner._write_json_atomic(path, launch)
            observed = {}
            fake_main = types.SimpleNamespace()

            def get_args_parser():
                parser = argparse.ArgumentParser(add_help=False)
                parser.add_argument("--output_dir")
                return parser

            def train(args):
                observed["contract"] = sys.modules.get(
                    "util.stage_b_table_b_contract"
                )
                observed["engine_loaded"] = "engine" in sys.modules
                observed["output_dir"] = args.output_dir

            fake_main.get_args_parser = get_args_parser
            fake_main.main = train
            real_import = runner.importlib.import_module

            def import_module(name, *args, **kwargs):
                if name == "main":
                    observed["contract_at_main_import"] = sys.modules.get(
                        "util.stage_b_table_b_contract"
                    )
                    return fake_main
                return real_import(name, *args, **kwargs)

            module_names = (
                "main",
                "engine",
                "datasets",
                "datasets.patch_episode",
                "util.stage_b_table_b_contract",
            )
            saved_modules = {name: sys.modules.pop(name, None) for name in module_names}
            saved_scope = v2_contract._PROCESS_SCOPE
            saved_scope_sha = v2_contract._PROCESS_SCOPE_SHA256
            saved_environment = os.environ.pop(v2_contract.SCOPE_SHA_ENV, None)
            try:
                with patch.object(
                    runner.importlib, "import_module", side_effect=import_module
                ):
                    result = runner._bootstrap_main(
                        [
                            "--launch-manifest",
                            str(path),
                            "--scope-sha256",
                            manifest["table_b_v2_scope_sha256"],
                            "--",
                            "--output_dir",
                            str(output),
                        ]
                    )
                self.assertEqual(result, 0)
                self.assertIs(observed["contract_at_main_import"], v2_contract)
                self.assertIs(observed["contract"], v2_contract)
                self.assertFalse(observed["engine_loaded"])
                receipt = json.loads((output / "scope_bootstrap.json").read_text())
                self.assertEqual(
                    receipt["status"], "scope_established_before_training_imports"
                )
                self.assertEqual(receipt["evidence"]["phase_id"], "joint")
            finally:
                for name in module_names:
                    sys.modules.pop(name, None)
                for name, module in saved_modules.items():
                    if module is not None:
                        sys.modules[name] = module
                v2_contract._PROCESS_SCOPE = saved_scope
                v2_contract._PROCESS_SCOPE_SHA256 = saved_scope_sha
                if saved_environment is not None:
                    os.environ[v2_contract.SCOPE_SHA_ENV] = saved_environment
                else:
                    os.environ.pop(v2_contract.SCOPE_SHA_ENV, None)

    def test_direct_detach_rejects_missing_dedicated_formal_plans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            args = runner.build_parser().parse_args(
                [
                    "detach",
                    "--orchestration-root",
                    str(root / "jobs"),
                    "--run-id",
                    "D2m:17",
                ]
            )
            with patch.object(
                runner._launcher(), "runtime_from_environment", return_value=runtime
            ), patch.object(
                runner.subprocess,
                "Popen",
                return_value=_FakeProcess(os.getpid()),
            ):
                with self.assertRaisesRegex(
                    runner.TableBV2RunnerError, "dedicated formal source/scope plans"
                ):
                    runner._detach(args)

    def test_generic_serial_queue_cannot_bypass_dedicated_formal_plans(self):
        from tools import run_stageb_serial_matrix_queue as queue_runner

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            environment = {
                "CUDA_VISIBLE_DEVICES": "0",
                "PIVOT_TN_OUTPUT_ROOT": str(runtime.tn_output_root),
                "PIVOT_BATCH_SIZE": "40",
                "PIVOT_MAX_TRAIN_ITERS": "1000",
                "PIVOT_ITER_CHECKPOINT_INTERVAL": "1000",
            }
            with patch.dict(os.environ, environment, clear=False):
                queue = queue_runner.create_queue(
                    root / "queue",
                    run_ids=["D2m:17"],
                    runner_python=Path(sys.executable),
                    token_runner=REPO_ROOT / "tools/run_stageb_token_ablation_matrix.py",
                    paper_runner=Path(runner.__file__),
                    lease_root=root / "leases",
                    gpu_key="0",
                )
            item = queue["items"][0]
            item_root = queue_runner._item_orchestration_root(queue, item)
            item_root.mkdir(parents=True)
            item["orchestration_root"] = str(item_root)
            item["status"] = "reserved"
            queue["status"] = "running"
            args = runner.build_parser().parse_args(
                [
                    "detach",
                    "--orchestration-root",
                    str(item_root),
                    "--run-id",
                    "D2m:17",
                ]
            )
            with patch.object(
                runner._launcher(), "runtime_from_environment", return_value=runtime
            ), patch.object(
                runner.subprocess,
                "Popen",
                return_value=_FakeProcess(os.getpid()),
            ):
                with self.assertRaisesRegex(
                    runner.TableBV2RunnerError, "dedicated formal source/scope plans"
                ):
                    runner._detach(args)

    def test_formal_resolver_requires_explicit_queue_argument(self):
        with self.assertRaises(TypeError):
            runner.resolve_for_matched_evaluation(
                Path("/missing"), object(), training_phase="final"
            )

    def test_input_rehash_replay_detects_record_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.txt"
            path.write_text("sealed\n", encoding="utf-8")
            record = runner._file_record(path, role="fixture_input")
            phase = {"inputs": {"records": [record]}}
            rehash_record = {
                "path": str(path.resolve()),
                "roles": ["fixture_input"],
                "expected_sha256": record["sha256"],
                "observed_sha256": record["sha256"],
                "observed_size_bytes": record["size_bytes"],
                "observed_mtime_ns": record["mtime_ns"],
                "passed": True,
            }
            postflight = {
                "input_rehash": {
                    "status": "passed",
                    "algorithm": "sha256",
                    "unique_input_count": 1,
                    "records": [rehash_record],
                }
            }
            identity = runner._verify_input_rehash(phase, postflight)
            self.assertEqual(set(identity), {str(path.resolve())})
            bad = copy.deepcopy(postflight)
            bad["input_rehash"]["records"][0]["observed_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                runner.TableBV2RunnerError, "input rehash drifted"
            ):
                runner._verify_input_rehash(phase, bad)


if __name__ == "__main__":
    unittest.main()
