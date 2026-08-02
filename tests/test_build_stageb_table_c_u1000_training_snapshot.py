import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_stageb_table_c_u1000_training_snapshot as snapshot


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _record(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    data = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "mtime_ns": metadata.st_mtime_ns,
    }


class TableCU1000TrainingSnapshotTest(unittest.TestCase):
    def test_current_attestation_declares_exact_audited_source_union(self):
        attestation = json.loads(
            snapshot.DEFAULT_DEPENDENCY_ATTESTATION.read_text(encoding="utf-8")
        )
        closure = attestation["dependency_closure"]["file_records"]
        static = attestation["training_evidence"]["static_repository_sources"]
        auditors = attestation["auditor_sources"]
        self.assertEqual(len(closure), 85)
        self.assertEqual(len(static), 9)
        self.assertEqual(len(auditors), 3)
        records = {}
        for record in [*closure, *static, *auditors]:
            identity = {
                key: record[key]
                for key in ("path", "sha256", "size_bytes", "mtime_ns")
            }
            if record["path"] in records:
                self.assertEqual(records[record["path"]], identity)
            records[record["path"]] = identity
        self.assertEqual(len(records), 89)
        self.assertEqual(
            len({record["sha256"] for record in records.values()}), 89
        )
        self.assertEqual(
            sum(record["size_bytes"] for record in records.values()), 1_790_057
        )

    def test_source_union_deduplicates_by_path_and_preserves_memberships(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = []
            for index in range(89):
                path = root / f"source_{index:03d}.py"
                _write(path, f"value = {index}\n".encode("ascii"))
                paths.append(path)
            closure = [
                {**_record(path), "relative_path": path.relative_to(root).as_posix()}
                for path in paths[:85]
            ]
            static = [
                {**_record(path), "relative_path": path.relative_to(root).as_posix()}
                for path in [*paths[:8], paths[85]]
            ]
            auditors = [_record(path) for path in paths[86:89]]
            attestation = {
                "schema": snapshot.ATTESTATION_SCHEMA,
                "repository_root": str(root),
                "dependency_closure": {"file_records": closure},
                "training_evidence": {"static_repository_sources": static},
                "auditor_sources": auditors,
            }
            attestation["attestation_sha256"] = snapshot._attestation_digest(
                attestation
            )
            expected_size = sum(path.stat().st_size for path in paths)
            with patch.object(
                snapshot, "SOURCE_UNION_SIZE_BYTES", expected_size
            ):
                result = snapshot._source_union_from_attestation(
                    attestation, snapshot._Binder()
                )
            self.assertEqual(len(result), 89)
            by_path = {record["path"]: record for record in result}
            self.assertEqual(
                by_path[str(paths[0])]["memberships"],
                ["dependency_closure", "static_repository_source"],
            )
            self.assertEqual(
                by_path[str(paths[85])]["memberships"],
                ["static_repository_source"],
            )
            self.assertEqual(
                by_path[str(paths[88])]["memberships"], ["auditor_source"]
            )

    def test_source_union_rejects_declared_identity_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = []
            for index in range(89):
                path = root / f"s{index}.py"
                _write(path, bytes([index]))
                paths.append(path)
            closure = [
                {**_record(path), "relative_path": path.name}
                for path in paths[:85]
            ]
            static_paths = [*paths[:8], paths[85]]
            static = [
                {**_record(path), "relative_path": path.name}
                for path in static_paths
            ]
            static[0]["sha256"] = "0" * 64
            attestation = {
                "schema": snapshot.ATTESTATION_SCHEMA,
                "repository_root": str(root),
                "dependency_closure": {"file_records": closure},
                "training_evidence": {"static_repository_sources": static},
                "auditor_sources": [_record(path) for path in paths[86:89]],
            }
            attestation["attestation_sha256"] = snapshot._attestation_digest(
                attestation
            )
            with self.assertRaisesRegex(
                snapshot.TrainingSnapshotError, "identity drifted"
            ):
                snapshot._source_union_from_attestation(
                    attestation, snapshot._Binder()
                )

    def _budget_fixture(self):
        run_id = "L4:42"
        root = "/archive/training/L4/seed42"
        checkpoint = {
            "path": f"{root}/checkpoint_iter.pth",
            "sha256": "1" * 64,
            "size_bytes": 10,
            "mtime_ns": 20,
        }
        postflight_record = {
            "path": f"{root}/postflight.json",
            "sha256": "2" * 64,
            "size_bytes": 11,
            "mtime_ns": 21,
        }
        launch = {
            "schema": "pivot.stageb.token_ablation_launch/v2",
            "status": "completed",
            "returncode": 0,
            "run_id": run_id,
            "seed": 42,
            "row": {
                "row_id": "L4",
                "config": "config/L4.py",
                "token_objective": "edit_bce",
            },
            "repository_root": "/archive/repository",
            "output_dir": root,
            "output_dir_fresh_at_plan": True,
            "runtime": {
                "batch_size": 40,
                "max_train_iters": 1000,
                "iter_checkpoint_interval": 1000,
                "amp": True,
            },
            "inputs": {
                "dataset_manifest": {"path": "/archive/data.json"},
                "stage_a_initializer": {"path": "/archive/stage_a.pth"},
                "scorer_warmstart": {"path": "/archive/scorer.pth"},
            },
        }
        sequence = {
            "schema": "pivot.stageb.token_ablation_sequence/v1",
            "status": "completed",
            "run_id": run_id,
            "seed": 42,
            "row": dict(launch["row"]),
            "training_seeds_contract": [17, 42, 73],
            "output_dir": root,
            "equal_budget_contract": {
                "batch_size": 40,
                "optimizer_updates": 1000,
                "contributing_phase_updates": {"joint": 1000},
            },
            "phases": [
                {
                    "phase_id": "joint",
                    "optimizer_updates": 1000,
                    "contributes_to_budget": True,
                    "output_dir": root,
                }
            ],
            "completed_phases": [
                {
                    "phase_id": "joint",
                    "status": "completed",
                    "output_dir": root,
                    "checkpoint": checkpoint,
                    "postflight": postflight_record,
                }
            ],
        }
        postflight = {
            "schema": "pivot.stageb.token_ablation_postflight/v2",
            "status": "passed",
            "run_id": run_id,
            "checkpoint_metadata": {
                "iteration": 1000,
                "epoch": 0,
                "checkpoint_reason": "max_train_iters",
                "has_complete_training_state": True,
                "epoch_finished": False,
                "args": {
                    "batch_size": 40,
                    "max_train_iters": 1000,
                    "iter_checkpoint_interval": 1000,
                    "seed": 42,
                    "config_file": "/archive/repository/config/L4.py",
                    "datasets": "/archive/data.json",
                    "output_dir": root,
                    "pretrain_model_path": "/archive/stage_a.pth",
                    "stage_b_v15_scorer_init_checkpoint": "/archive/scorer.pth",
                    "stage_b_v21_token_objective": "edit_bce",
                },
            },
            "numerical_status": {
                "status": "passed",
                "amp_enabled": True,
                "loss_values_all_finite": True,
                "finite_loss_observations": 200,
                "amp_skip_observations": 200,
                "max_amp_step_skipped": 0.0,
                "min_amp_scale": 512.0,
                "max_amp_scale": 512.0,
            },
        }
        return run_id, root, launch, sequence, postflight

    def test_budget_gate_requires_fixed_positive_amp_scale_and_zero_skips(self):
        run_id, root, launch, sequence, postflight = self._budget_fixture()
        snapshot._validate_budget_and_numerics(
            run_id=run_id,
            run_root=root,
            launch=launch,
            sequence=sequence,
            postflight=postflight,
        )
        postflight["numerical_status"]["max_amp_scale"] = 1024.0
        with self.assertRaisesRegex(
            snapshot.TrainingSnapshotError, "fixed positive AMP scale"
        ):
            snapshot._validate_budget_and_numerics(
                run_id=run_id,
                run_root=root,
                launch=launch,
                sequence=sequence,
                postflight=postflight,
            )

    def test_input_rehash_requires_exact_36_record_launch_union(self):
        source_records = [
            {
                "path": f"/source/{index}",
                "sha256": f"{index:064x}",
                "size_bytes": index + 1,
                "mtime_ns": index + 10,
            }
            for index in range(26)
        ]
        non_source_records = [
            {
                "path": f"/input/{index}",
                "sha256": f"{index + 100:064x}",
                "size_bytes": index + 100,
                "mtime_ns": index + 200,
            }
            for index in range(10)
        ]
        records = []
        for record in [*source_records, *non_source_records]:
            records.append(
                {
                    "path": record["path"],
                    "expected_sha256": record["sha256"],
                    "observed_sha256": record["sha256"],
                    "observed_size_bytes": record["size_bytes"],
                    "observed_mtime_ns": record["mtime_ns"],
                    "passed": True,
                }
            )
        rehash = {
            "algorithm": "sha256",
            "status": "passed",
            "unique_input_count": 36,
            "records": records,
        }
        snapshot._validate_input_rehash(
            run_id="L0:17",
            launch_sources=source_records,
            non_sources=non_source_records,
            postflight={"input_rehash": rehash},
            input_rehash=rehash,
        )
        rehash["records"] = rehash["records"][:-1]
        with self.assertRaisesRegex(
            snapshot.TrainingSnapshotError, "exact 36-record gate"
        ):
            snapshot._validate_input_rehash(
                run_id="L0:17",
                launch_sources=source_records,
                non_sources=non_source_records,
                postflight={"input_rehash": rehash},
                input_rehash=rehash,
            )

    def test_queue_identity_is_nested_under_plan_and_revision_is_bound(self):
        plan = {
            "schema": "pivot.stageb.serial_matrix_queue_plan/v1",
            "queue_id": "nested-id",
            "items": [{"run_id": "L0:17", "runner": "token"}],
        }
        digest = snapshot._canonical_sha256(plan)
        locked = {
            "fixture": {
                "queue_id": "nested-id",
                "plan_sha256": digest,
                "run_ids": ("L0:17",),
            }
        }
        queue = {
            "schema": "pivot.stageb.serial_matrix_queue/v1",
            "status": "completed",
            "revision": 7,
            "plan": plan,
            "plan_sha256": digest,
            "items": [
                {
                    "index": 0,
                    "run_id": "L0:17",
                    "runner": "token",
                    "status": "completed",
                }
            ],
            "queue_id": "misleading-top-level-id",
        }
        with patch.object(snapshot, "LOCKED_QUEUES", locked):
            snapshot._validate_queue_payload(queue, role="fixture")
            queue["plan"] = {**plan, "queue_id": "drifted"}
            queue["plan_sha256"] = snapshot._canonical_sha256(queue["plan"])
            locked["fixture"]["plan_sha256"] = queue["plan_sha256"]
            with self.assertRaisesRegex(
                snapshot.TrainingSnapshotError, "final identity drifted"
            ):
                snapshot._validate_queue_payload(queue, role="fixture")

    def test_existing_live_final_gates_are_all_replayed(self):
        from tools import audit_stageb_table_c_dependency_closure as dependency
        from tools import build_stageb_paper_ablation_completion_receipt as completion
        from tools import recover_stageb_serial_matrix_pretraining_failure as recovery
        from tools import run_stageb_serial_matrix_queue as queue_runner

        queue_results = [
            {
                "status": "passed",
                "queue_id": "one",
                "plan_sha256": "a" * 64,
                "verified_items": [{}] * 5,
            },
            {
                "status": "passed",
                "queue_id": "two",
                "plan_sha256": "b" * 64,
                "verified_items": [{}] * 28,
            },
        ]
        recovery_result = {
            "status": "passed",
            "run_id": "L2:42",
            "current_item_status": "completed",
            "archived_evidence_verified": True,
            "semantic_replay": recovery.SEMANTIC_REPLAY_PROOF,
            "receipt_sha256": "c" * 64,
        }
        with patch.object(
            dependency,
            "verify_attestation",
            return_value={"status": "passed", "canonical_closure_sha256": "d" * 64},
        ) as dependency_gate, patch.object(
            completion,
            "_validate_table_c_sequences",
            return_value=(list(snapshot.EXPECTED_RUN_IDS), [{}] * 33),
        ) as sequence_gate, patch.object(
            queue_runner, "verify_queue", side_effect=queue_results
        ) as queue_gate, patch.object(
            recovery, "verify_recovery", return_value=recovery_result
        ) as recovery_gate:
            result = snapshot._run_live_final_gates(
                dependency_attestation=Path("attestation.json"),
                queue_dirs=(Path("q1"), Path("q2")),
                recovery_receipt=Path("recovery.json"),
            )
        dependency_gate.assert_called_once_with(Path("attestation.json"), policy="final")
        sequence_gate.assert_called_once_with()
        self.assertEqual(queue_gate.call_count, 2)
        recovery_gate.assert_called_once_with(Path("q2"), Path("recovery.json"))
        self.assertEqual(result["single_pretraining_recovery"]["failed_revision"], 590)

    def _minimal_offline_snapshot(self, root: Path) -> Path:
        live = root / "live-source.py"
        _write(live, b"archived bytes\n")
        live_record = _record(live)
        object_relative = snapshot._object_relative_path(live_record["sha256"])
        _write(root / "bundle" / object_relative, live.read_bytes())
        live.unlink()
        archived = {**live_record, "archive_object": object_relative}
        source_snapshot = {
            "schema": snapshot.SOURCE_SNAPSHOT_SCHEMA,
            "status": "retrospective_training_source_snapshot",
            "claim_scope": {
                "retroactively_launch_binds_omitted_files": False,
                "snapshot_builder_is_a_training_source": False,
            },
            "sources": [archived],
            "object_store": {
                "algorithm": "sha256",
                "layout": "objects/sha256/HH/SHA256",
                "object_count": 1,
                "unique_source_path_count": 1,
                "unique_source_digest_count": 1,
            },
        }
        source_snapshot["source_snapshot_sha256"] = snapshot._source_snapshot_digest(
            source_snapshot
        )
        source_path = root / "bundle/source_snapshot.json"
        rendered_source = snapshot._canonical_json_bytes(source_snapshot) + b"\n"
        _write(source_path, rendered_source)
        attestation_path = root / "completion-attestation.json"
        _write(attestation_path, b"{}")
        completion = {
            "schema": snapshot.COMPLETION_SUBRECEIPT_SCHEMA,
            "status": "complete_retrospective_training_completion_subreceipt",
            "expected_run_ids": list(snapshot.EXPECTED_RUN_IDS),
            "claim_scope": {
                "outside_historical_dependency_closure": True,
                "retroactively_launch_binds_omitted_files": False,
            },
            "source_snapshot": {
                "relative_path": "source_snapshot.json",
                "sha256": hashlib.sha256(rendered_source).hexdigest(),
                "size_bytes": len(rendered_source),
                "source_snapshot_sha256": source_snapshot[
                    "source_snapshot_sha256"
                ],
            },
            "dependency_attestation": _record(attestation_path),
            "non_source_launch_inputs": [],
            "launch_source_union": [],
            "training_queues": [],
            "pretraining_recovery": {},
            "runs": [],
        }
        completion["completion_subreceipt_sha256"] = (
            snapshot._completion_subreceipt_digest(completion)
        )
        _write(
            root / "bundle/completion_subreceipt.json",
            snapshot._canonical_json_bytes(completion) + b"\n",
        )
        return root / "bundle"

    def test_verify_uses_archived_source_unless_live_parity_is_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._minimal_offline_snapshot(Path(temporary))

            def source_gate(source_snapshot, _attestation, reader):
                reader.read(source_snapshot["sources"][0], label="fixture object")

            with patch.object(
                snapshot, "_verify_source_snapshot", side_effect=source_gate
            ), patch.object(
                snapshot, "_verify_completion_semantics", return_value=None
            ), patch.object(
                snapshot, "SOURCE_UNION_COUNT", 1
            ), patch.object(
                snapshot, "LAUNCH_SOURCE_UNION_COUNT", 0
            ):
                verified = snapshot.verify_snapshot(bundle)
                self.assertEqual(verified["status"], "passed")
                with self.assertRaisesRegex(
                    snapshot.TrainingSnapshotError, "cannot stat live source parity"
                ):
                    snapshot.verify_snapshot(
                        bundle, require_live_source_parity=True
                    )

    def test_archived_object_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._minimal_offline_snapshot(Path(temporary))
            source_snapshot = json.loads(
                (bundle / "source_snapshot.json").read_text(encoding="utf-8")
            )
            object_path = bundle / source_snapshot["sources"][0]["archive_object"]
            object_path.write_bytes(b"tampered")

            def source_gate(source_snapshot, _attestation, reader):
                reader.read(source_snapshot["sources"][0], label="fixture object")

            with patch.object(
                snapshot, "_verify_source_snapshot", side_effect=source_gate
            ), patch.object(
                snapshot, "_verify_completion_semantics", return_value=None
            ), patch.object(
                snapshot, "SOURCE_UNION_COUNT", 1
            ), patch.object(
                snapshot, "LAUNCH_SOURCE_UNION_COUNT", 0
            ), self.assertRaisesRegex(
                snapshot.TrainingSnapshotError, "archived object bytes drifted"
            ):
                snapshot.verify_snapshot(bundle)

    def test_default_verify_rejects_live_completion_evidence_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._minimal_offline_snapshot(root)
            (root / "completion-attestation.json").write_bytes(b'{"drift":true}')
            with patch.object(
                snapshot, "SOURCE_UNION_COUNT", 1
            ), patch.object(
                snapshot, "LAUNCH_SOURCE_UNION_COUNT", 0
            ), self.assertRaisesRegex(
                snapshot.TrainingSnapshotError, "live identity drifted"
            ):
                snapshot.verify_snapshot(bundle)

    def test_build_refuses_existing_canonical_output_before_live_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "snapshot"
            output.mkdir()
            with patch.object(snapshot, "CANONICAL_OUTPUT_ROOT", output), patch.object(
                snapshot, "_collect_live_evidence"
            ) as collect, self.assertRaisesRegex(
                snapshot.TrainingSnapshotError, "must be fresh"
            ):
                snapshot.build_snapshot()
            collect.assert_not_called()

    def test_atomic_publication_does_not_replace_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            _write(source / "new", b"new")
            _write(destination / "old", b"old")
            with self.assertRaisesRegex(
                snapshot.TrainingSnapshotError, "not replaced"
            ):
                snapshot._rename_noreplace(source, destination)
            self.assertTrue((source / "new").is_file())
            self.assertEqual((destination / "old").read_bytes(), b"old")

    def test_only_the_source_union_is_selected_for_object_archival(self):
        source = {
            "path": "/source.py",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "mtime_ns": 1,
        }
        completion = {
            "path": "/checkpoint.pth",
            "sha256": "b" * 64,
            "size_bytes": 2,
            "mtime_ns": 2,
        }
        evidence = {
            "source_union": [source],
            "dependency_attestation": completion,
            "training_queues": [{"queue_json": completion}],
            "pretraining_recovery": {"receipt": completion},
            "runs": [
                {
                    "artifacts": {"checkpoint": completion},
                    "postflight_artifacts": {"checkpoint": completion},
                }
            ],
        }
        self.assertEqual(list(snapshot._iter_archivable_records(evidence)), [source])


if __name__ == "__main__":
    unittest.main()
