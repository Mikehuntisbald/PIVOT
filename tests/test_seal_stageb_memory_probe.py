import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from tools import run_stageb_paper_ablation_matrices as paper_launcher
from tools import seal_stageb_memory_probe as memory_seal
from tools.seal_stageb_memory_probe import MemoryProbeError, build_seal, inspect_probe


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class SealStageBMemoryProbeTest(unittest.TestCase):
    def _probe(
        self,
        root: Path,
        *,
        batch: int,
        updates: int,
        free_mib: float,
        status: str = "completed",
        postflight: str = "passed",
        row_id: str | None = None,
    ) -> Path:
        sequence = {
            "status": status,
            "equal_budget_contract": {
                "batch_size": batch,
                "optimizer_updates": updates,
            },
        }
        if row_id is not None:
            sequence.update(
                {
                    "run_id": f"{row_id}:17",
                    "row": {
                        "row_id": row_id,
                        "table": "D",
                        "score_ownership": "independent_decoders_joint",
                        "objective_fidelity": (
                            "common_objective_ownership_ablation"
                        ),
                    },
                }
            )
        _write(
            root / "sequence_manifest.json",
            sequence,
        )
        _write(
            root / "postflight.json",
            {
                "status": postflight,
                "numerical_status": {
                    "status": "passed",
                    "loss_values_all_finite": True,
                    "max_amp_step_skipped": 0.0,
                },
            },
        )
        _write(
            root / "gpu_telemetry_summary.json",
            {
                "devices": [
                    {
                        "uuid": "GPU-test",
                        "name": "RTX test",
                        "driver_version": "1",
                        "total_memory_mib": 32000,
                        "peak_used_memory_mib": 32000 - free_mib,
                        "min_free_memory_mib": free_mib,
                        "sample_count": 10,
                    }
                ]
            },
        )
        return root

    def _strict_paper_probe(
        self,
        family_root: Path,
        *,
        updates: int,
        diagnostic_interval: int,
        row_id: str = "S2",
        batch_size: int = 40,
    ) -> tuple[Path, Path, dict]:
        row = paper_launcher.ROW_BY_ID[row_id]
        root = family_root / row_id / "seed17"
        root.mkdir(parents=True, exist_ok=True)
        run_id = f"{row_id}:17"
        config = (paper_launcher.REPO_ROOT / row.config).resolve(strict=True)
        stage_a = family_root / "stage_a.pth"
        scorer = family_root / "scorer.pth"
        stage_a.write_bytes(b"stage-a")
        scorer.write_bytes(b"scorer")
        runtime = paper_launcher.Runtime(
            python=paper_launcher.DEFAULT_PYTHON.resolve(strict=True),
            stage_a_init=stage_a.resolve(strict=True),
            scorer_warmstart=scorer.resolve(strict=True),
            tn_output_root=(
                family_root.resolve()
                if row.table == "B"
                else (family_root / "unused_tn").resolve()
            ),
            score_output_root=(
                family_root.resolve()
                if row.table == "D"
                else (family_root / "unused_score").resolve()
            ),
            data_root=Path("/media/haoyi/T9/data").resolve(strict=True),
            batch_size=batch_size,
            total_train_iters=updates,
            iter_checkpoint_interval=updates,
            num_workers=2,
            prefetch_factor=1,
            omp_num_threads=8,
            min_nofile=65536,
            cuda_visible_devices="0",
            mp_sharing_strategy="file_system",
            gradient_diagnostic_interval=diagnostic_interval,
        )
        phase_contract = paper_launcher._phases(runtime, row)[0]
        planned = paper_launcher._phase_manifest(
            runtime,
            row,
            17,
            phase_contract,
            root,
            paper_launcher.token_launcher.HashCache(),
            rank_checkpoint=None,
        )
        checkpoint = root / "checkpoint_iter.pth"
        checkpoint.write_bytes(b"checkpoint")
        gpu_environment = {
            "schema": "pivot.gpu_environment/v1",
            "cuda_visible_devices": "0",
            "torch_runtime": {"cuda_available": True},
            "nvidia_devices": [
                {
                    "physical_index": 0,
                    "uuid": "GPU-test",
                    "name": "RTX test",
                    "driver_version": "1",
                    "total_memory_mib": 32000.0,
                }
            ],
            "amp_requested": True,
        }
        _write(root / "gpu_environment.json", gpu_environment)
        (root / "gpu_telemetry.csv").write_text(
            "timestamp,index,uuid,name,driver_version,total_memory_mib,"
            "used_memory_mib,free_memory_mib,utilization_percent\n"
            "2026-07-18T00:00:00,0,GPU-test,RTX test,1,32000,30000,2000,99\n",
            encoding="utf-8",
        )
        telemetry = paper_launcher._summarize_nvidia_csv(
            root / "gpu_telemetry.csv"
        )
        telemetry.update(
            {"captured_at_utc": "2026-07-18T00:00:01+00:00", "sampling_interval_ms": 1000}
        )
        _write(root / "gpu_telemetry_summary.json", telemetry)
        logs = (
            "loss: 1.0 amp_step_skipped: 0.0 (0.0) "
            "amp_scale: 512.0 (512.0) max mem: 1000 "
            "stage_b_v22_branch_isolation_pass\n"
        )
        (root / "info.txt").write_text(logs, encoding="utf-8")
        (root / "train_console.log").write_text(logs, encoding="utf-8")
        numerical = paper_launcher._training_numerical_status(
            root / "info.txt", root / "train_console.log"
        )
        scorer_record = next(
            record
            for record in planned["inputs"]["records"]
            if record["role"] == "scorer_warmstart"
        )
        stage_a_record = next(
            record
            for record in planned["inputs"]["records"]
            if record["role"] == "stage_a_initializer"
        )
        scorer_audit = {
            "schema": "stage_b_v15_scorer_init/v1",
            "status": "applied",
            "source_sha256": scorer_record["sha256"],
            "loaded_num_layers": 3,
            "resolved_source_path": str(scorer.resolve(strict=True)),
        }
        _write(root / "stage_b_v15_scorer_init_audit.json", scorer_audit)
        input_rehash = paper_launcher._rehash_inputs(planned)
        _write(root / "input_rehash.json", input_rehash)
        metadata = {
            "has_complete_training_state": True,
            "iteration": updates,
            "checkpoint_reason": "max_train_iters",
            "epoch_finished": False,
            "args": {},
        }
        phase = asdict(phase_contract)
        postflight = {
            "schema": memory_seal.PAPER_POSTFLIGHT_SCHEMA,
            "status": "passed",
            "run_id": run_id,
            "phase_id": "joint",
            "gpu_environment": gpu_environment,
            "gpu_telemetry_summary": telemetry,
            "input_rehash": input_rehash,
            "numerical_status": numerical,
            "checkpoint_metadata": metadata,
            "model_state_ancestry": {
                "pretrain_path": str(stage_a.resolve(strict=True)),
                "pretrain_sha256": stage_a_record["sha256"],
                "pretrain_manifest_role": "stage_a_initializer",
                "pretrain_mode": "model_state_only_no_optimizer_resume",
                "checkpoint_resume_argument": None,
                "scorer_warmstart_applied": True,
                "generated_dependency": None,
            },
            "artifacts": {
                "checkpoint": memory_seal._file_record(checkpoint),
                "gpu_environment": memory_seal._file_record(
                    root / "gpu_environment.json"
                ),
                "gpu_telemetry": memory_seal._file_record(
                    root / "gpu_telemetry.csv"
                ),
                "gpu_telemetry_summary": memory_seal._file_record(
                    root / "gpu_telemetry_summary.json"
                ),
                "native_info_log": memory_seal._file_record(root / "info.txt"),
                "train_console_log": memory_seal._file_record(
                    root / "train_console.log"
                ),
                "scorer_init_audit": memory_seal._file_record(
                    root / "stage_b_v15_scorer_init_audit.json"
                ),
                "input_rehash": memory_seal._file_record(root / "input_rehash.json"),
            },
        }
        _write(root / "postflight.json", postflight)
        launch = deepcopy(planned)
        launch.update(
            {
                "status": "completed",
                "returncode": 0,
                "gpu_environment": gpu_environment,
                "gpu_telemetry_summary": telemetry,
                "postflight": postflight,
                "postflight_artifact": memory_seal._file_record(
                    root / "postflight.json"
                ),
            }
        )
        _write(root / "launch_manifest.json", launch)
        sequence = {
            "schema": memory_seal.PAPER_SEQUENCE_SCHEMA,
            "status": "completed",
            "repository_root": str(paper_launcher.REPO_ROOT),
            "run_id": run_id,
            "seed": 17,
            "training_seeds_contract": list(paper_launcher.SEEDS),
            "row": asdict(row),
            "output_dir": str(root.resolve()),
            "output_dir_fresh_at_plan": True,
            "equal_budget_contract": {
                "batch_size": batch_size,
                "optimizer_updates": updates,
                "s3_probe_updates_excluded": 0,
                "contributing_phase_updates": {"joint": updates},
            },
            "phases": [planned],
            "completed_phases": [
                {
                    "phase_id": "joint",
                    "status": "completed",
                    "output_dir": str(root.resolve()),
                    "checkpoint": memory_seal._file_record(checkpoint),
                }
            ],
        }
        _write(root / "sequence_manifest.json", sequence)
        return root, config, metadata

    def test_seals_only_completed_headroom_soak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = self._probe(
                root / "short", batch=40, updates=2, free_mib=1600
            )
            soak = self._probe(
                root / "soak", batch=40, updates=50, free_mib=1500
            )
            seal = build_seal(
                {"b40-short": short, "b40-soak": soak},
                selected="b40-soak",
            )

        self.assertEqual(seal["status"], "sealed")
        self.assertEqual(seal["selection"]["batch_size"], 40)
        self.assertFalse(seal["probes"]["b40-short"]["soak_pass"])
        self.assertTrue(seal["probes"]["b40-soak"]["soak_pass"])

    def test_rejects_low_headroom_short_or_nonterminal_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low = self._probe(root / "low", batch=42, updates=50, free_mib=900)
            with self.assertRaisesRegex(MemoryProbeError, "headroom"):
                build_seal({"low": low}, selected="low")

            short = self._probe(
                root / "short", batch=40, updates=49, free_mib=1500
            )
            with self.assertRaisesRegex(MemoryProbeError, "requires at least 50"):
                build_seal({"short": short}, selected="short")

            interrupted = self._probe(
                root / "interrupted",
                batch=40,
                updates=50,
                free_mib=1500,
                status="hard_terminated_unknown",
            )
            with self.assertRaisesRegex(MemoryProbeError, "not completed"):
                inspect_probe(interrupted)

    def test_expected_row_contract_rejects_cross_family_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            s2 = self._probe(
                root / "s2", batch=32, updates=50, free_mib=1800, row_id="S2"
            )
            seal = build_seal(
                {"s2": s2}, selected="s2", expected_row_id="S2"
            )
            self.assertEqual(
                seal["experiment_contract"],
                {"expected_row_id": "S2", "all_probes_match": True},
            )
            self.assertEqual(seal["probes"]["s2"]["experiment"]["row_id"], "S2")

            wrong = self._probe(
                root / "s1", batch=32, updates=50, free_mib=1800, row_id="S1"
            )
            with self.assertRaisesRegex(MemoryProbeError, "expected row 'S2'"):
                build_seal(
                    {"wrong": wrong},
                    selected="wrong",
                    expected_row_id="S2",
                )

    def test_current_closure_rejects_stale_config_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            protocol = root / "protocol.md"
            config.write_text("value = 1\n", encoding="utf-8")
            protocol.write_text("contract v1\n", encoding="utf-8")

            def launch():
                return {
                    "inputs": {
                        "records": [
                            {
                                **memory_seal._file_record(config),
                                "role": "config_dependency",
                            },
                            {
                                **memory_seal._file_record(protocol),
                                "role": "repository_source",
                            },
                        ]
                    }
                }

            config_launch = launch()
            config.write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(MemoryProbeError, "config.py.*sha256 drifted"):
                memory_seal._verify_current_input_closure(
                    config_launch, (config, protocol)
                )

            config.write_text("value = 1\n", encoding="utf-8")
            protocol_launch = launch()
            protocol.write_text("contract v2\n", encoding="utf-8")
            with self.assertRaisesRegex(MemoryProbeError, "protocol.md.*sha256 drifted"):
                memory_seal._verify_current_input_closure(
                    protocol_launch, (config, protocol)
                )

    def test_current_closure_rejects_unbound_recursive_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            imported = root / "imported.py"
            config.write_text("value = 1\n", encoding="utf-8")
            imported.write_text("value = 2\n", encoding="utf-8")
            launch = {
                "inputs": {
                    "records": [
                        {
                            **memory_seal._file_record(config),
                            "role": "config_dependency",
                        }
                    ]
                }
            }
            with self.assertRaisesRegex(MemoryProbeError, "omitted.*closure"):
                memory_seal._verify_current_input_closure(
                    launch, (config, imported)
                )

    def test_current_d3m_probe_replays_canonical_runner_and_can_be_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, config, metadata = self._strict_paper_probe(
                Path(temporary) / "table_b_memory",
                updates=50,
                diagnostic_interval=0,
                row_id="D3m",
                batch_size=40,
            )
            with (
                patch.object(
                    memory_seal,
                    "_current_paper_training_closure",
                    return_value=(config,),
                ),
                patch.object(
                    paper_launcher,
                    "_inspect_checkpoint_safely",
                    return_value=metadata,
                ) as inspect_checkpoint,
                patch.object(
                    paper_launcher, "_validate_checkpoint_metadata"
                ) as validate_checkpoint,
            ):
                seal = build_seal(
                    {"soak": root},
                    selected="soak",
                    expected_row_id="D3m",
                )
            self.assertEqual(seal["selection"]["batch_size"], 40)
            self.assertEqual(
                seal["probes"]["soak"]["optimizer_updates"], 50
            )
            inspect_checkpoint.assert_called_once()
            validate_checkpoint.assert_called_once()

    def test_d3m_current_runner_audit_drift_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = self._strict_paper_probe(
                Path(temporary) / "audit_drift",
                updates=50,
                diagnostic_interval=0,
                row_id="D3m",
            )
            with patch.object(
                paper_launcher,
                "_phase_manifest",
                side_effect=RuntimeError("current Table-B audit SHA-256 drifted"),
            ):
                with self.assertRaisesRegex(
                    MemoryProbeError,
                    "current canonical phase reconstruction failed.*audit SHA-256 drifted",
                ):
                    build_seal(
                        {"soak": root},
                        selected="soak",
                        expected_row_id="D3m",
                    )

    def test_current_training_closure_includes_native_runtime_lineage(self):
        paths = memory_seal._native_training_dependency_paths()
        suffixes = {path.suffix for path in paths}
        self.assertIn(".so", suffixes)
        self.assertTrue({".cpp", ".cu", ".h"}.issubset(suffixes))
        self.assertTrue(any(path.name == "build.ninja" for path in paths))
        spec = memory_seal.importlib.util.find_spec("MultiScaleDeformableAttention")
        self.assertIsNotNone(spec)
        actual = Path(spec.origin).resolve(strict=True)
        self.assertIn(actual, paths)

    def test_d3m_creation_rejects_runtime_and_current_input_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runtime_root, _, _ = self._strict_paper_probe(
                base / "runtime",
                updates=50,
                diagnostic_interval=0,
                row_id="D3m",
            )
            launch_path = runtime_root / "launch_manifest.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["runtime"]["prefetch_factor"] = 2
            _write(launch_path, launch)
            with self.assertRaisesRegex(
                MemoryProbeError, "launch (?:command|runtime).*canonical"
            ):
                build_seal(
                    {"soak": runtime_root},
                    selected="soak",
                    expected_row_id="D3m",
                )

            stale_root, _, _ = self._strict_paper_probe(
                base / "stale_input",
                updates=50,
                diagnostic_interval=0,
                row_id="D3m",
            )
            launch_path = stale_root / "launch_manifest.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            dataset_record = next(
                record
                for record in launch["inputs"]["records"]
                if record["role"] == "dataset_manifest"
            )
            dataset_record["sha256"] = "0" * 64
            _write(launch_path, launch)
            sequence_path = stale_root / "sequence_manifest.json"
            sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
            sequence["phases"][0]["inputs"] = deepcopy(launch["inputs"])
            _write(sequence_path, sequence)
            with self.assertRaisesRegex(
                MemoryProbeError, "postflight evidence replay failed.*SHA-256 drift"
            ):
                build_seal(
                    {"soak": stale_root},
                    selected="soak",
                    expected_row_id="D3m",
                )

    def test_d3m_deep_replay_rejects_consistently_rewritten_telemetry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = self._strict_paper_probe(
                Path(temporary) / "telemetry",
                updates=50,
                diagnostic_interval=0,
                row_id="D3m",
            )
            telemetry_path = root / "gpu_telemetry_summary.json"
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            telemetry["devices"][0]["min_free_memory_mib"] = 1900.0
            _write(telemetry_path, telemetry)
            postflight_path = root / "postflight.json"
            postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
            postflight["gpu_telemetry_summary"] = telemetry
            postflight["artifacts"]["gpu_telemetry_summary"] = (
                memory_seal._file_record(telemetry_path)
            )
            _write(postflight_path, postflight)
            launch_path = root / "launch_manifest.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["gpu_telemetry_summary"] = telemetry
            launch["postflight"] = postflight
            launch["postflight_artifact"] = memory_seal._file_record(
                postflight_path
            )
            _write(launch_path, launch)
            with self.assertRaisesRegex(
                MemoryProbeError, "telemetry summary differs from CSV replay"
            ):
                build_seal(
                    {"soak": root},
                    selected="soak",
                    expected_row_id="D3m",
                )

    def test_d3m_rejects_missing_current_recursive_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            family = Path(temporary) / "closure"
            root, config, metadata = self._strict_paper_probe(
                family,
                updates=50,
                diagnostic_interval=0,
                row_id="D3m",
            )
            newly_required = family / "new_training_dependency.py"
            newly_required.write_text("value = 1\n", encoding="utf-8")
            with (
                patch.object(
                    memory_seal,
                    "_current_paper_training_closure",
                    return_value=(config, newly_required),
                ),
                patch.object(
                    paper_launcher,
                    "_inspect_checkpoint_safely",
                    return_value=metadata,
                ),
                patch.object(
                    paper_launcher, "_validate_checkpoint_metadata"
                ),
            ):
                with self.assertRaisesRegex(
                    MemoryProbeError, "omitted current recursive.*closure"
                ):
                    build_seal(
                        {"soak": root},
                        selected="soak",
                        expected_row_id="D3m",
                    )

    def test_matched_table_b_rejects_d2m_and_missing_expected_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            d2m, _, _ = self._strict_paper_probe(
                base / "wrong_row",
                updates=50,
                diagnostic_interval=0,
                row_id="D2m",
            )
            with self.assertRaisesRegex(MemoryProbeError, "expected row 'D3m'"):
                build_seal(
                    {"soak": d2m},
                    selected="soak",
                    expected_row_id="D3m",
                )

            d3m, config, metadata = self._strict_paper_probe(
                base / "implicit_row",
                updates=50,
                diagnostic_interval=0,
                row_id="D3m",
            )
            with (
                patch.object(
                    memory_seal,
                    "_current_paper_training_closure",
                    return_value=(config,),
                ),
                patch.object(
                    paper_launcher,
                    "_inspect_checkpoint_safely",
                    return_value=metadata,
                ),
                patch.object(
                    paper_launcher, "_validate_checkpoint_metadata"
                ),
            ):
                with self.assertRaisesRegex(
                    MemoryProbeError, "require --expected-row-id D3m"
                ):
                    build_seal({"soak": d3m}, selected="soak")

    def test_d3m_creation_never_bypasses_deep_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            probe = self._probe(
                Path(temporary) / "d3m",
                batch=40,
                updates=50,
                free_mib=1800,
                row_id="D3m",
            )
            with patch.object(
                memory_seal,
                "_verify_paper_probe_current",
                side_effect=MemoryProbeError("deep replay required"),
            ) as replay:
                with self.assertRaisesRegex(MemoryProbeError, "deep replay required"):
                    build_seal(
                        {"soak": probe},
                        selected="soak",
                        expected_row_id="D3m",
                    )
            replay.assert_called_once()

    def test_d3m_formal_soak_profile_is_not_relaxable(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cases = (
                ("batch", 39, 50, 1024.0, "batch size 40"),
                ("updates", 40, 51, 1024.0, "exactly 50 optimizer updates"),
                ("headroom", 40, 50, 1000.0, "cannot relax minimum headroom"),
            )
            with patch.object(memory_seal, "_verify_paper_probe_current"):
                for name, batch, updates, minimum_headroom, message in cases:
                    with self.subTest(case=name):
                        probe = self._probe(
                            base / name,
                            batch=batch,
                            updates=updates,
                            free_mib=1800,
                            row_id="D3m",
                        )
                        with self.assertRaisesRegex(MemoryProbeError, message):
                            build_seal(
                                {"soak": probe},
                                selected="soak",
                                expected_row_id="D3m",
                                minimum_headroom_mib=minimum_headroom,
                            )

    def test_table_d_soak_requires_declared_diagnostic_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, config, _ = self._strict_paper_probe(
                Path(temporary) / "probe", updates=50, diagnostic_interval=1
            )
            with patch.object(
                memory_seal,
                "_current_paper_training_closure",
                return_value=(config,),
            ):
                with self.assertRaisesRegex(
                    MemoryProbeError, "50-update probe requires diagnostic interval 10"
                ):
                    memory_seal._verify_paper_probe_current(
                        root,
                        expected_row_id="S2",
                        minimum_soak_updates=50,
                    )

    def test_table_d_probe_rejects_postflight_and_telemetry_hash_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, config, _ = self._strict_paper_probe(
                Path(temporary) / "probe", updates=2, diagnostic_interval=1
            )
            (root / "gpu_telemetry_summary.json").write_text(
                json.dumps({"devices": []}), encoding="utf-8"
            )
            with patch.object(
                memory_seal,
                "_current_paper_training_closure",
                return_value=(config,),
            ):
                with self.assertRaisesRegex(
                    MemoryProbeError,
                    "gpu_telemetry_summary.*(?:size_bytes|sha256) drifted",
                ):
                    memory_seal._verify_paper_probe_current(
                        root,
                        expected_row_id="S2",
                        minimum_soak_updates=50,
                    )

    def test_verify_only_runs_current_table_d_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = self._probe(
                root / "s2", batch=40, updates=50, free_mib=1800, row_id="S2"
            )
            output = root / "seal.json"
            _write(
                output,
                build_seal(
                    {"s2": probe}, selected="s2", expected_row_id="S2"
                ),
            )
            with patch.object(
                memory_seal,
                "_verify_paper_probe_current",
                side_effect=MemoryProbeError("stale current closure"),
            ) as replay:
                with contextlib.redirect_stderr(io.StringIO()):
                    result = memory_seal.main(
                        [
                            "--probe",
                            f"s2={probe}",
                            "--selected",
                            "s2",
                            "--expected-row-id",
                            "S2",
                            "--output",
                            str(output),
                            "--verify-only",
                        ]
                    )
            self.assertEqual(result, 2)
            replay.assert_called_once()


if __name__ == "__main__":
    unittest.main()
