import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from tools import build_stageb_paper_ablation_completion_receipt as completion
from tools import recover_stageb_serial_matrix_pretraining_failure as recovery
from tools import run_stageb_matrix_validation_queue as matrix_queue
from tools import stageb_headline_release_contract as release


def _write_json(path: Path, value: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({} if value is None else value, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _fake_registry(artifact: Path):
    record = completion.file_record(artifact)

    def adapter(block: str, adapter_id: str):
        def verify():
            return {
                "status": "completed",
                "adapter_id": adapter_id,
                "contract": {
                    "block": block,
                    "run_ids": [f"{block}:17", f"{block}:42", f"{block}:73"],
                    "seeds": [17, 42, 73],
                    "batch_size": 40,
                    "optimizer_updates": 1000,
                },
                "artifacts": {"fixture": record},
                "semantic_replay": {
                    "training_queue_verified": True,
                    "validation_queue_verified": True,
                    "aggregate_recomputed": True,
                },
            }

        return completion.BlockAdapter(adapter_id=adapter_id, verifier=verify)

    return {
        block: adapter(block, f"fixture_{block}/v1")
        for block in completion.BLOCKS
    }


def _headline_snapshot(training_runner, path: Path, *, salt: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(salt + "\n", encoding="ascii")
    return training_runner._completed_evidence_snapshot((path,))


def _headline_completed_run(
    training_runner,
    contract_id: str,
    seed: int,
    *,
    queue_id: str,
    plan_sha256: str,
    queue_contract_sha256: str,
    closure_digest: str,
    snapshot_root: Path,
) -> dict:
    contract = training_runner.CONTRACTS[contract_id]
    run_id = f"{contract_id}:{seed}"
    run_root = contract.canonical_training_root(seed).resolve(strict=False)

    def digest(label: str) -> str:
        return hashlib.sha256(f"{run_id}:{label}".encode("ascii")).hexdigest()

    attempt_artifacts = {}
    attempt_artifact_paths = []
    for name in ("gpu_environment", "gpu_telemetry", "gpu_telemetry_summary"):
        path = snapshot_root / "attempt_000" / f"{name}.fixture"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{run_id}:{name}\n", encoding="ascii")
        attempt_artifacts[name] = completion.file_record(path)
        attempt_artifact_paths.append(path)
    telemetry_devices = [{"physical_index": 0, "name": "fixture-gpu"}]
    attempt_telemetry = {
        "schema": training_runner.ATTEMPT_TELEMETRY_SCHEMA,
        "status": "sealed",
        "attempt_ordinal": 0,
        "sampling_interval_ms": 1000,
        "sample_rows": 10,
        "devices": telemetry_devices,
        "artifacts": attempt_artifacts,
    }
    projected_attempt = {
        "attempt_ordinal": 0,
        "sample_rows": 10,
        "devices": telemetry_devices,
        "artifacts": attempt_artifacts,
        "evidence_sha256": completion.canonical_json_sha256(attempt_telemetry),
    }
    full_run_telemetry = {
        "schema": training_runner.FULL_RUN_TELEMETRY_SCHEMA,
        "status": "passed",
        "attempt_count": 1,
        "sampling_interval_ms": 1000,
        "sample_rows": 10,
        "devices": telemetry_devices,
        "all_attempts_same_devices": True,
        "attempts": [projected_attempt],
    }
    full_run_telemetry["semantic_sha256"] = completion.canonical_json_sha256(
        full_run_telemetry
    )
    completed_evidence_path = snapshot_root / "completed-evidence.fixture"
    completed_evidence_path.write_text(
        f"{run_id}:completed-evidence\n", encoding="ascii"
    )
    evidence_snapshot = training_runner._completed_evidence_snapshot(
        (completed_evidence_path, *attempt_artifact_paths)
    )

    run = {
        "schema": training_runner.COMPLETED_TRAINING_VERIFICATION_SCHEMA,
        "status": "passed",
        "run_id": run_id,
        "contract_id": contract_id,
        "seed": seed,
        "run_root": str(run_root),
        "contract": {
            "row": contract.expected_row(),
            "headline": bool(contract.headline),
            "matrix_validation_only": bool(contract.matrix_validation_only),
            "token_objective": (
                "edit_bce" if contract_id == "M0" else contract.token_objective
            ),
            "token_objective_scope": contract.token_objective_scope,
        },
        "final_checkpoint": {
            "path": str(run_root / "checkpoint_iter.pth"),
            "sha256": digest("checkpoint"),
            "size_bytes": 100,
        },
        "training_queue_binding": {
            "contract_id": contract_id,
            "queue_id": queue_id,
            "plan_sha256": plan_sha256,
            "queue_contract_sha256": queue_contract_sha256,
            "stable_input_closure_digest": closure_digest,
            "ordered_run_ids": list(contract.dedicated_queue_run_ids),
            "active_item": {
                "item_index": list(contract.seeds).index(seed),
                "run_id": run_id,
                "item_status": "launched",
                "orchestration_root": str(run_root / "orchestration"),
                "gpu_key": "0",
                "lease_path": str(run_root / "gpu.lease.json"),
            },
        },
        "budget": {
            **contract.expected_budget(),
            "gradient_accumulation_steps": 1,
            "amp": True,
            "final_epoch": training_runner.FORMAL_FINAL_EPOCH,
            "final_iteration": training_runner.FORMAL_FINAL_ITERATION,
            "optimizer_state_count": training_runner.FORMAL_OPTIMIZER_STATE_COUNT,
        },
        "ancestry": {
            "status": "passed",
            "stage_a_initializer": {
                "path": str(training_runner.DEFAULT_STAGE_A_INIT),
                "sha256": training_runner.DEFAULT_STAGE_A_SHA256,
                "size_bytes": 200,
            },
            "stage_a_and_scorer_same_source": True,
            "b58_ancestry_count": 0,
            "resume_chain_contiguous": True,
            "attempt_count": 1,
            "resume_count": 0,
            "attempt_manifests": [
                {
                    "path": str(run_root / "attempts/attempt_000.json"),
                    "sha256": digest("attempt"),
                    "size_bytes": 10,
                }
            ],
        },
        "numerical": {
            "status": "passed",
            "amp_enabled": True,
            "finite_loss_observations": 10,
            "loss_values_all_finite": True,
            "amp_skip_observations": 10,
            "max_amp_step_skipped": 0.0,
            "evidence_sha256": digest("numerical"),
        },
        "telemetry": {
            "status": "passed",
            "sampling_interval_ms": 1000,
            "sample_rows": 10,
            "devices": telemetry_devices,
            "evidence_sha256": digest("telemetry"),
            "full_run": full_run_telemetry,
        },
        "input_closure": {
            "status": "passed",
            "digest": closure_digest,
            "record_count": 10,
            "verifier_source_digest": hashlib.sha256(
                f"{contract_id}:verifier".encode("ascii")
            ).hexdigest(),
            "verifier_source_count": 5,
            "identity_snapshot": _headline_snapshot(
                training_runner,
                snapshot_root / "stable-input.fixture",
                salt=f"{run_id}:stable-input",
            ),
        },
        "artifacts": {
            name: {
                "path": str(run_root / filename),
                "sha256": digest(name),
                "size_bytes": 10,
            }
            for name, filename in (
                ("sequence_manifest", "sequence_manifest.json"),
                ("launch_manifest", "launch_manifest.json"),
                ("postflight", "postflight.json"),
            )
        },
        "evidence_snapshot": evidence_snapshot,
    }
    run["semantic_sha256"] = completion.canonical_json_sha256(run)
    return run


def _headline_training_verification(
    training_runner,
    contract_id: str,
    *,
    queue_path: Path,
    queue_id: str,
    salt: str,
) -> dict:
    contract = training_runner.CONTRACTS[contract_id]
    plan_sha256 = hashlib.sha256(f"{salt}:plan".encode()).hexdigest()
    queue_contract_sha256 = hashlib.sha256(
        f"{salt}:contract".encode()
    ).hexdigest()
    closure_digest = hashlib.sha256(f"{salt}:closure".encode()).hexdigest()
    runs = [
        _headline_completed_run(
            training_runner,
            contract_id,
            seed,
            queue_id=queue_id,
            plan_sha256=plan_sha256,
            queue_contract_sha256=queue_contract_sha256,
            closure_digest=closure_digest,
            snapshot_root=queue_path.parent
            / "training-evidence"
            / contract_id
            / f"seed{seed}",
        )
        for seed in contract.seeds
    ]
    queue_manifest = training_runner._stable_completed_file_record(queue_path)
    stable_input_snapshot = _headline_snapshot(
        training_runner,
        queue_path.parent / f"{contract_id}.stable-input.fixture",
        salt=f"{contract_id}:queue-stable-input",
    )
    serial_completion = {
        "schema": "pivot.stageb.serial_matrix_queue_verification/v1",
        "status": "passed",
        "queue_status": "completed",
        "queue_id": queue_id,
        "plan_sha256": plan_sha256,
        "verified_items": [
            {"run_id": run_id} for run_id in contract.dedicated_queue_run_ids
        ],
        "errors": [],
    }
    completion_sha = completion.canonical_json_sha256(
        {
            "contract_id": contract_id,
            "queue_id": queue_id,
            "plan_sha256": plan_sha256,
            "queue_contract_sha256": queue_contract_sha256,
            "queue_manifest": queue_manifest,
            "ordered_run_ids": list(contract.dedicated_queue_run_ids),
            "run_semantic_sha256s": [run["semantic_sha256"] for run in runs],
            "stable_input_snapshot": stable_input_snapshot,
            "serial_completion_evidence": serial_completion,
            "verifier_source_digest": runs[0]["input_closure"][
                "verifier_source_digest"
            ],
        }
    )
    return {
        "schema": "pivot.stageb.headline_m0_queue_verification/v1",
        "status": "passed",
        "contract_id": contract_id,
        "queue_status": "completed",
        "queue_id": queue_id,
        "plan_sha256": plan_sha256,
        "ordered_run_ids": list(contract.dedicated_queue_run_ids),
        "queue_contract_sha256": queue_contract_sha256,
        "queue_manifest": queue_manifest,
        "stable_input_closure_digest": closure_digest,
        "active_item": None,
        "completion_verification": {
            **serial_completion,
            "verified_at_utc": "2026-07-19T09:00:00+00:00",
        },
        "serial_completion_evidence": serial_completion,
        "completed_stable_input_snapshot": stable_input_snapshot,
        "completed_training_runs": runs,
        "completion_semantic_sha256": completion_sha,
    }


class PaperAblationCompletionReceiptTest(unittest.TestCase):
    def _headline_fixture(self, root: Path) -> dict:
        from tools import aggregate_stageb_headline_m0_validation as aggregate
        from tools import run_stageb_headline_m0 as training_runner
        from tools import run_stageb_headline_m0_validation_queue as validation_queue

        m0_queue = root / "m0-training"
        m0n_queue = root / "m0n-training"
        validation_dir = root / "validation"
        output_root = root / "evaluations"
        report_path = root / "headline_m0_m0n_validation_report.json"
        _write_json(m0_queue / "queue.json", {"contract_id": "M0"})
        _write_json(m0n_queue / "queue.json", {"contract_id": "M0N"})
        _write_json(validation_dir / "queue.json", {"status": "completed"})

        training = {
            "M0": _headline_training_verification(
                training_runner,
                "M0",
                queue_path=m0_queue / "queue.json",
                queue_id="00000000-0000-0000-0000-000000000001",
                salt="m0",
            ),
            "M0N": _headline_training_verification(
                training_runner,
                "M0N",
                queue_path=m0n_queue / "queue.json",
                queue_id="00000000-0000-0000-0000-000000000002",
                salt="m0n",
            ),
        }
        queue_dirs = {"M0": m0_queue, "M0N": m0n_queue}
        validation_queue_id = "00000000-0000-0000-0000-000000000003"
        validation_plan_sha = hashlib.sha256(b"validation-plan").hexdigest()
        plan_training = []
        for contract_id in validation_queue.CONTRACT_IDS:
            verified = training[contract_id]
            plan_training.append(
                {
                    "contract_id": contract_id,
                    "queue_dir": str(queue_dirs[contract_id].resolve()),
                    "queue_id": verified["queue_id"],
                    "plan_sha256": verified["plan_sha256"],
                    "queue_contract_sha256": verified["queue_contract_sha256"],
                    "stable_input_closure_digest": verified[
                        "stable_input_closure_digest"
                    ],
                    "ordered_run_ids": verified["ordered_run_ids"],
                    "manifest_at_creation": validation_queue._file_record(
                        queue_dirs[contract_id] / "queue.json"
                    ),
                }
            )
        plan_items = []
        for index, run_id in enumerate(validation_queue.RUN_IDS):
            contract_id, raw_seed = run_id.split(":", 1)
            seed = int(raw_seed)
            verified = training[contract_id]
            plan_items.append(
                {
                    "index": index,
                    "run_id": run_id,
                    "contract_id": contract_id,
                    "train_seed": seed,
                    "training_root": str(
                        training_runner.CONTRACTS[
                            contract_id
                        ].canonical_training_root(seed)
                    ),
                    "training_queue_dir": str(queue_dirs[contract_id].resolve()),
                    "training_queue_id": verified["queue_id"],
                    "training_queue_plan_sha256": verified["plan_sha256"],
                    "evaluation_root": str(
                        validation_queue._evaluation_root(output_root, run_id)
                    ),
                }
            )
        spec_path = validation_dir / validation_queue.AGGREGATION_SPEC_NAME
        plan = {
            "schema": validation_queue.PLAN_SCHEMA,
            "queue_id": validation_queue_id,
            "queue_dir": str(validation_dir.resolve()),
            "output_root": str(output_root.resolve()),
            "profile": validation_queue.PROFILE,
            "ordered_run_ids": list(validation_queue.RUN_IDS),
            "training_queues": plan_training,
            "aggregation_input_spec": {
                "schema": validation_queue.SPEC_SCHEMA,
                "path": str(spec_path.resolve()),
            },
            "items": plan_items,
        }
        validation_manifest = {
            "schema": validation_queue.QUEUE_SCHEMA,
            "status": "completed",
            "plan": plan,
            "plan_sha256": validation_plan_sha,
        }
        spec = validation_queue._aggregation_spec_payload(plan, validation_plan_sha)
        _write_json(spec_path, spec)
        spec_record = validation_queue._file_record(spec_path)
        validation = {
            "schema": validation_queue.VERIFICATION_SCHEMA,
            "status": "passed",
            "queue_status": "completed",
            "queue_id": validation_queue_id,
            "plan_sha256": validation_plan_sha,
            "ordered_run_ids": list(validation_queue.RUN_IDS),
            "aggregation_input_spec": spec_record,
            "verified_items": [
                {"run_id": run_id} for run_id in validation_queue.RUN_IDS
            ],
            "errors": [],
        }
        checkpoint_shas = {
            contract_id: {
                str(seed): hashlib.sha256(
                    f"{contract_id}:{seed}:checkpoint".encode("ascii")
                ).hexdigest()
                for seed in validation_queue.SEEDS
            }
            for contract_id in validation_queue.CONTRACT_IDS
        }
        report = {
            "schema": aggregate.REPORT_SCHEMA,
            "status": aggregate.REPORT_STATUS,
            "created_at_utc": "2026-07-19T09:00:00+00:00",
            "formal_test_or_strict_result": False,
            "comparison_claim": "full_token_objective_control_not_labels_only",
            "reference_experiment": "M0",
            "candidate_experiment": "M0N",
            "direction": "M0N_minus_M0",
            "protocol": {
                "profile": validation_queue.PROFILE,
                "train_seeds": list(validation_queue.SEEDS),
                "seed_estimator": "equal-seed mean and sample standard deviation",
                "paired_bootstrap": {
                    "iterations": aggregate.FORMAL_BOOTSTRAP_ITERATIONS,
                    "confidence": aggregate.FORMAL_BOOTSTRAP_CONFIDENCE,
                    "seed": aggregate.FORMAL_BOOTSTRAP_SEED,
                    "unit": "image cluster within training seed",
                    "seed_first": True,
                },
                "ref_test_access": False,
                "strict_tn_access": False,
            },
            "validation": {
                "pass": True,
                "training_queues_separate": True,
                "exact_six_evaluations": True,
                "record_identities_aligned": True,
                "runtime_code_data_surface_equal": True,
                "input_rehash_and_postflight_replayed": True,
            },
            "inputs": {
                "aggregation_spec": spec_record,
                "evaluation_queue": {
                    "queue_id": validation_queue_id,
                    "plan_sha256": validation_plan_sha,
                    "verification_schema": validation_queue.VERIFICATION_SCHEMA,
                },
                "aggregation_source_closure": [{"fixture": "sealed"}],
                "checkpoint_sha256s": checkpoint_shas,
            },
            "experiments": {"M0": {"seed_count": 3}, "M0N": {"seed_count": 3}},
            "comparison": {"candidate_minus_reference": {"metric": 0.1}},
        }
        report["report_sha256"] = aggregate._report_sha256(report)
        _write_json(report_path, report)
        return {
            "aggregate": aggregate,
            "training_runner": training_runner,
            "validation_queue": validation_queue,
            "m0_queue": m0_queue,
            "m0n_queue": m0n_queue,
            "validation_dir": validation_dir,
            "output_root": output_root,
            "report_path": report_path,
            "training": training,
            "validation_manifest": validation_manifest,
            "validation": validation,
            "report": report,
        }

    def _headline_patch_stack(
        self,
        fixture: dict,
        *,
        training_side_effect=None,
        validation_manifest=None,
        validation=None,
        report=None,
        aggregate_source: Path | None = None,
        report_side_effect=None,
    ) -> ExitStack:
        aggregate = fixture["aggregate"]
        training_runner = fixture["training_runner"]
        validation_queue = fixture["validation_queue"]
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                completion, "HEADLINE_M0_TRAINING_QUEUE", fixture["m0_queue"]
            )
        )
        stack.enter_context(
            mock.patch.object(
                completion, "HEADLINE_M0N_TRAINING_QUEUE", fixture["m0n_queue"]
            )
        )
        stack.enter_context(
            mock.patch.object(
                completion,
                "HEADLINE_M0_VALIDATION_QUEUE",
                fixture["validation_dir"],
            )
        )
        stack.enter_context(
            mock.patch.object(
                completion,
                "HEADLINE_M0_VALIDATION_AGGREGATE",
                fixture["report_path"],
            )
        )
        stack.enter_context(
            mock.patch.object(
                validation_queue, "DEFAULT_QUEUE_DIR", fixture["validation_dir"]
            )
        )
        stack.enter_context(
            mock.patch.object(
                validation_queue, "DEFAULT_OUTPUT_ROOT", fixture["output_root"]
            )
        )
        stack.enter_context(
            mock.patch.object(
                aggregate, "DEFAULT_QUEUE_DIR", fixture["validation_dir"]
            )
        )
        stack.enter_context(
            mock.patch.object(
                aggregate, "DEFAULT_REPORT_PATH", fixture["report_path"]
            )
        )
        if aggregate_source is not None:
            stack.enter_context(
                mock.patch.object(aggregate, "__file__", str(aggregate_source))
            )
        if training_side_effect is None:
            training_side_effect = lambda _queue, contract_id, **_kwargs: fixture[
                "training"
            ][contract_id]
        fixture["training_mock"] = stack.enter_context(
            mock.patch.object(
                training_runner,
                "verify_training_queue",
                side_effect=training_side_effect,
            )
        )
        fixture["validation_load_mock"] = stack.enter_context(
            mock.patch.object(
                validation_queue,
                "load_queue",
                return_value=(
                    fixture["validation_manifest"]
                    if validation_manifest is None
                    else validation_manifest
                ),
            )
        )
        fixture["validation_verify_mock"] = stack.enter_context(
            mock.patch.object(
                validation_queue,
                "verify_queue",
                return_value=(
                    fixture["validation"] if validation is None else validation
                ),
            )
        )
        if report_side_effect is None:
            fixture["report_mock"] = stack.enter_context(
                mock.patch.object(
                    aggregate,
                    "verify_report",
                    return_value=(fixture["report"] if report is None else report),
                )
            )
        else:
            fixture["report_mock"] = stack.enter_context(
                mock.patch.object(
                    aggregate, "verify_report", side_effect=report_side_effect
                )
            )
        return stack

    def _recovery_fixture(self, root: Path):
        receipt = root / "recovery_receipt.json"
        receipt.write_text('{"status":"archived"}\n', encoding="ascii")
        receipt_record = completion._table_c_mtime_file_record(receipt)
        specification = {
            "queue_id": "queue-id",
            "plan_sha256": "a" * 64,
        }
        queue = {
            "events": [
                {
                    "event": recovery.RECOVERY_EVENT,
                    "run_id": "L2:42",
                    "failed_revision": 590,
                    "receipt": receipt_record,
                }
            ],
            "items": [
                {
                    "run_id": "L2:42",
                    "status": "completed",
                    "pretraining_recovery_receipts": [receipt_record],
                }
            ],
        }
        verification = {
            "status": "passed",
            "queue_id": "queue-id",
            "plan_sha256": "a" * 64,
            "run_id": "L2:42",
            "current_item_status": "completed",
            "archived_evidence_verified": True,
            "semantic_replay": recovery.SEMANTIC_REPLAY_PROOF,
            "verifier_source": completion._table_c_mtime_file_record(
                Path(recovery.__file__)
            ),
        }
        return receipt, receipt_record, specification, queue, verification

    def test_table_c_adapter_replays_single_pretraining_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt, receipt_record, specification, queue, verification = (
                self._recovery_fixture(root)
            )
            queues = (root / "screen", root / "remaining")
            with (
                mock.patch.object(
                    completion, "TABLE_C_PRETRAINING_RECOVERY_RECEIPT", receipt
                ),
                mock.patch.object(matrix_queue, "DEFAULT_TRAINING_QUEUE_DIRS", queues),
                mock.patch.object(
                    recovery, "verify_recovery", return_value=verification
                ) as replay,
            ):
                observed_record, observed_verifier, observed = (
                    completion._verify_table_c_pretraining_recovery(
                        queue, specification
                    )
                )
            self.assertEqual(observed_record, receipt_record)
            self.assertEqual(
                observed_verifier,
                completion._table_c_mtime_file_record(Path(recovery.__file__)),
            )
            self.assertEqual(observed, verification)
            replay.assert_called_once_with(queues[1], receipt)

    def test_table_c_adapter_rejects_extra_recovery_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt, _, specification, queue, verification = self._recovery_fixture(root)
            queue["events"].append(dict(queue["events"][0]))
            with (
                mock.patch.object(
                    completion, "TABLE_C_PRETRAINING_RECOVERY_RECEIPT", receipt
                ),
                mock.patch.object(
                    matrix_queue,
                    "DEFAULT_TRAINING_QUEUE_DIRS",
                    (root / "screen", root / "remaining"),
                ),
                mock.patch.object(
                    recovery, "verify_recovery", return_value=verification
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError, "single sealed"
                ),
            ):
                completion._verify_table_c_pretraining_recovery(queue, specification)

    def test_table_c_adapter_treats_current_policy_attestation_as_supplemental(self):
        from tools import audit_stageb_table_c_dependency_closure as dependency

        with mock.patch.object(
            dependency,
            "verify_attestation",
            side_effect=AssertionError("live auditor must not run"),
        ) as replay:
            result, source = (
                completion._verify_table_c_supplemental_final_dependency_attestation()
            )
        self.assertEqual(result["status"], "passed_supplemental_self_hashed")
        self.assertFalse(result["authoritative_dependency_proof"])
        self.assertFalse(result["live_source_parity_required"])
        self.assertEqual(result["archived_auditor_record_count"], 2)
        self.assertEqual(result["unarchived_newer_auditor_record_count"], 1)
        self.assertEqual(
            source,
            completion.file_record(
                completion.TABLE_C_CURRENT_POLICY_FINAL_DEPENDENCY_ATTESTATION
            ),
        )
        replay.assert_not_called()

        original_read = completion._read_json
        changed = original_read(
            completion.TABLE_C_CURRENT_POLICY_FINAL_DEPENDENCY_ATTESTATION,
            label="fixture",
        )
        changed["auditor_sources"][0]["sha256"] = "0" * 64

        def tampered_read(path, *, label):
            if Path(path).resolve() == (
                completion.TABLE_C_CURRENT_POLICY_FINAL_DEPENDENCY_ATTESTATION.resolve()
            ):
                return changed
            return original_read(path, label=label)

        with mock.patch.object(
            completion, "_read_json", side_effect=tampered_read
        ), self.assertRaisesRegex(
            completion.CompletionReceiptError, "fixed identity or self-hash"
        ):
            completion._verify_table_c_supplemental_final_dependency_attestation()

    def test_table_c_adapter_replays_archived_finalization_lineage(self):
        result, evidence = completion._verify_table_c_archived_finalization_lineage()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["historical_auditor_sha256"],
            "8a30be0c00d5b180c46a71923cc7dbb5b9998c8a7652d0d8c26b2feb8894dacb",
        )
        self.assertTrue(result["historical_auditor_archived"])
        self.assertTrue(result["historical_finalizer_evidence_bound"])
        self.assertFalse(result["historical_finalizer_live_bytes_required"])
        self.assertTrue(result["staged_upgrade_replayed"])
        self.assertTrue(result["all_authoritative_sources_archived"])
        self.assertEqual(result["dependency_record_count"], 85)
        self.assertEqual(result["static_source_record_count"], 9)
        self.assertEqual(result["auditor_record_count"], 3)
        self.assertEqual(result["source_count"], 89)
        self.assertEqual(result["object_count"], 89)
        self.assertEqual(len(result["_archived_object_records"]), 89)
        self.assertEqual(
            evidence["archived_object_records_sha256"],
            result["archived_object_records_sha256"],
        )
        self.assertNotIn("_archived_object_records", evidence)
        self.assertEqual(
            evidence["schema"],
            completion.TABLE_C_FINALIZATION_LINEAGE_EVIDENCE_SCHEMA,
        )
        self.assertFalse(evidence["historical_finalizer_live_bytes_required"])
        self.assertEqual(
            evidence["historical_finalizer_source_record"],
            dict(completion.TABLE_C_HISTORICAL_FINALIZER_SOURCE_IDENTITY),
        )
        evidence_without_hash = copy.deepcopy(evidence)
        evidence_sha256 = evidence_without_hash.pop("evidence_sha256")
        self.assertEqual(
            evidence_sha256,
            completion.canonical_json_sha256(evidence_without_hash),
        )

    def test_table_c_adapter_rejects_archived_source_and_object_tamper(self):
        preflight = json.loads(
            completion.TABLE_C_PREFLIGHT_DEPENDENCY_ATTESTATION.read_text(
                encoding="utf-8"
            )
        )
        source_snapshot = json.loads(
            (
                completion.TABLE_C_TRAINING_SNAPSHOT / "source_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        changed = copy.deepcopy(source_snapshot)
        changed["sources"][0]["memberships"] = ["auditor_source"]
        with self.assertRaisesRegex(
            completion.CompletionReceiptError, "archived source record drifted"
        ):
            completion._verify_table_c_archived_source_objects(preflight, changed)

        first = source_snapshot["sources"][0]
        target = (
            completion.TABLE_C_TRAINING_SNAPSHOT / first["archive_object"]
        ).resolve()
        original_sha = completion._sha256_file

        def forged_object_sha(path):
            if Path(path).resolve() == target:
                return "0" * 64
            return original_sha(path)

        with mock.patch.object(
            completion, "_sha256_file", side_effect=forged_object_sha
        ), self.assertRaisesRegex(
            completion.CompletionReceiptError, "archived object bytes drifted"
        ):
            completion._verify_table_c_archived_source_objects(
                preflight, source_snapshot
            )

    def test_table_c_archived_objects_reject_persistent_post_hash_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository_root = root / "repository"
            source_path = repository_root / "tools/source.py"
            source_bytes = b"print('archived')\n"
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            source_identity = {
                "path": str(source_path.resolve()),
                "sha256": source_sha256,
                "size_bytes": len(source_bytes),
                "mtime_ns": 1,
            }
            preflight = {
                "repository_root": str(repository_root.resolve()),
                "dependency_closure": {"file_records": [source_identity]},
                "training_evidence": {"static_repository_sources": []},
                "auditor_sources": [],
            }
            archive_object = (
                f"objects/sha256/{source_sha256[:2]}/{source_sha256}"
            )
            source_snapshot_sha256 = "a" * 64
            source_snapshot = {
                "schema": (
                    "pivot.stageb.table_c_u1000_training_source_snapshot/v1"
                ),
                "status": "retrospective_training_source_snapshot",
                "source_snapshot_sha256": source_snapshot_sha256,
                "sources": [
                    {
                        **source_identity,
                        "relative_path": "tools/source.py",
                        "memberships": ["dependency_closure"],
                        "archive_object": archive_object,
                    }
                ],
            }
            snapshot_root = root / "snapshot"
            object_path = snapshot_root / archive_object
            object_path.parent.mkdir(parents=True)
            object_path.write_bytes(source_bytes)
            counts = {
                "dependency_closure": 1,
                "static_repository_source": 0,
                "auditor_source": 0,
                "deduplicated_source": 1,
            }
            snapshot_identity = {
                **completion.TABLE_C_TRAINING_SNAPSHOT_IDENTITY,
                "source_snapshot_sha256": source_snapshot_sha256,
            }
            original_record = completion.file_record
            mutated = False

            def mutate_after_first_hash(path):
                nonlocal mutated
                record = original_record(path)
                if Path(path).resolve() == object_path.resolve() and not mutated:
                    changed = object_path.read_bytes()
                    object_path.write_bytes(bytes([changed[0] ^ 1]) + changed[1:])
                    mutated = True
                return record

            with (
                mock.patch.object(
                    completion, "TABLE_C_ARCHIVED_SOURCE_COUNTS", counts
                ),
                mock.patch.object(
                    completion,
                    "TABLE_C_TRAINING_SNAPSHOT_IDENTITY",
                    snapshot_identity,
                ),
                mock.patch.object(
                    completion,
                    "file_record",
                    side_effect=mutate_after_first_hash,
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "archived source objects artifacts changed during semantic replay",
                ),
            ):
                completion._verify_table_c_archived_source_objects(
                    preflight,
                    source_snapshot,
                    snapshot_root=snapshot_root,
                )
            self.assertTrue(mutated)

    def test_table_c_adapter_rejects_rehashed_final_lineage_tamper(self):
        original_read = completion._read_json
        changed = original_read(
            completion.TABLE_C_FINAL_DEPENDENCY_ATTESTATION,
            label="fixture",
        )
        changed["finalization"]["transformation"]["to"] = "running_or_completed"
        changed["attestation_sha256"] = completion._table_c_attestation_digest(
            changed
        )
        identities = copy.deepcopy(
            completion.TABLE_C_DEPENDENCY_ATTESTATION_IDENTITIES
        )
        identities["final"]["semantic_sha256"] = changed["attestation_sha256"]

        def tampered_read(path, *, label):
            if Path(path).resolve() == (
                completion.TABLE_C_FINAL_DEPENDENCY_ATTESTATION.resolve()
            ):
                return copy.deepcopy(changed)
            return original_read(path, label=label)

        with (
            mock.patch.object(
                completion,
                "TABLE_C_DEPENDENCY_ATTESTATION_IDENTITIES",
                identities,
            ),
            mock.patch.object(completion, "_read_json", side_effect=tampered_read),
            self.assertRaisesRegex(
                completion.CompletionReceiptError,
                "auditor/final transformation lineage drifted",
            ),
        ):
            completion._verify_table_c_archived_finalization_lineage()

    def test_table_c_adapter_rejects_rehashed_historical_finalizer_record_tamper(self):
        original_read = completion._read_json
        changed = original_read(
            completion.TABLE_C_FINAL_DEPENDENCY_ATTESTATION,
            label="fixture",
        )
        changed["finalization"]["finalizer_source"]["sha256"] = "0" * 64
        changed["attestation_sha256"] = completion._table_c_attestation_digest(
            changed
        )
        identities = copy.deepcopy(
            completion.TABLE_C_DEPENDENCY_ATTESTATION_IDENTITIES
        )
        identities["final"]["semantic_sha256"] = changed["attestation_sha256"]

        def tampered_read(path, *, label):
            if Path(path).resolve() == (
                completion.TABLE_C_FINAL_DEPENDENCY_ATTESTATION.resolve()
            ):
                return copy.deepcopy(changed)
            return original_read(path, label=label)

        with (
            mock.patch.object(
                completion,
                "TABLE_C_DEPENDENCY_ATTESTATION_IDENTITIES",
                identities,
            ),
            mock.patch.object(completion, "_read_json", side_effect=tampered_read),
            self.assertRaisesRegex(
                completion.CompletionReceiptError,
                "archived finalizer source identity drifted",
            ),
        ):
            completion._verify_table_c_archived_finalization_lineage()

    def test_table_c_archived_finalization_requires_every_canonical_evidence_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("TABLE_C_FINAL_DEPENDENCY_ATTESTATION", root / "missing-final.json"),
                (
                    "TABLE_C_PREFLIGHT_DEPENDENCY_ATTESTATION",
                    root / "missing-preflight.json",
                ),
                ("TABLE_C_TRAINING_SNAPSHOT", root / "missing-snapshot"),
            )
            for attribute, missing in cases:
                with (
                    self.subTest(attribute=attribute),
                    mock.patch.object(completion, attribute, missing),
                    self.assertRaisesRegex(
                        completion.CompletionReceiptError,
                        "cannot bind completion artifact",
                    ),
                ):
                    completion._verify_table_c_archived_finalization_lineage()

    def test_table_c_archived_finalization_rejects_canonical_evidence_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            final_attestation = Path(temporary) / "final.json"
            final_attestation.write_bytes(
                completion.TABLE_C_FINAL_DEPENDENCY_ATTESTATION.read_bytes()
            )
            original_verify = completion._verify_table_c_archived_source_objects

            def mutate_after_source_replay(*args, **kwargs):
                result = original_verify(*args, **kwargs)
                final_attestation.write_bytes(final_attestation.read_bytes() + b"\n")
                return result

            with (
                mock.patch.object(
                    completion,
                    "TABLE_C_FINAL_DEPENDENCY_ATTESTATION",
                    final_attestation,
                ),
                mock.patch.object(
                    completion,
                    "_verify_table_c_archived_source_objects",
                    side_effect=mutate_after_source_replay,
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "artifacts changed during semantic replay",
                ),
            ):
                completion._verify_table_c_archived_finalization_lineage()

    def test_table_c_adapter_rejects_supplemental_outer_window_race(self):
        from tools import aggregate_stageb_matrix_validation as aggregate
        from tools import build_stageb_table_c_u1000_training_snapshot as snapshot
        from tools import (
            recover_stageb_matrix_validation_interruption as validation_recovery,
        )
        from tools import run_stageb_serial_matrix_queue as serial_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_dirs = (root / "training-0", root / "training-1")
            validation_dir = root / "validation"
            snapshot_root = root / "snapshot"
            for directory in (*training_dirs, validation_dir, snapshot_root):
                directory.mkdir(parents=True)

            final_attestation = root / "final.json"
            preflight_attestation = root / "preflight.json"
            supplemental_attestation = root / "supplemental.json"
            recovery_receipt_path = root / "recovery.json"
            validation_recovery_receipt_path = root / "validation-recovery.json"
            aggregation_spec_path = root / "aggregation-spec.json"
            aggregation_report_path = root / "aggregation-report.json"
            for path, kind in (
                (final_attestation, "final"),
                (preflight_attestation, "preflight"),
                (supplemental_attestation, "supplemental"),
                (recovery_receipt_path, "recovery"),
                (validation_recovery_receipt_path, "validation-recovery"),
                (aggregation_report_path, "aggregate"),
            ):
                _write_json(path, {"kind": kind})
            _write_json(snapshot_root / "completion_subreceipt.json")
            _write_json(snapshot_root / "source_snapshot.json")
            for directory in training_dirs:
                _write_json(directory / "queue.json")
            _write_json(validation_dir / "queue.json")

            object_records = {}
            for index in range(89):
                content = f"archived-object-{index}\n".encode("ascii")
                sha256 = hashlib.sha256(content).hexdigest()
                archive_object = f"objects/sha256/{sha256[:2]}/{sha256}"
                object_path = snapshot_root / archive_object
                object_path.parent.mkdir(parents=True, exist_ok=True)
                object_path.write_bytes(content)
                object_records[archive_object] = completion.file_record(object_path)
            object_records_sha256 = completion.canonical_json_sha256(
                {"objects": object_records}
            )

            expected_ids = list(matrix_queue.EXPECTED_RUN_IDS)
            training_queues = {}
            for directory, specification in zip(
                training_dirs,
                matrix_queue.LOCKED_TRAINING_QUEUES.values(),
            ):
                training_queues[str(directory.resolve())] = {
                    "status": "completed",
                    "plan": {
                        "queue_id": specification["queue_id"],
                        "items": [
                            {"run_id": run_id}
                            for run_id in specification["run_ids"]
                        ],
                    },
                    "plan_sha256": specification["plan_sha256"],
                }

            evaluation_queue_id = "evaluation-queue-id"
            evaluation_plan_sha256 = "e" * 64
            _write_json(
                aggregation_spec_path,
                {
                    "schema": aggregate.INPUT_SCHEMA,
                    "evaluation_queue_id": evaluation_queue_id,
                    "evaluation_plan_sha256": evaluation_plan_sha256,
                },
            )
            aggregation_spec_record = completion.file_record(
                aggregation_spec_path
            )
            evaluation = {
                "status": "completed",
                "plan": {
                    "queue_id": evaluation_queue_id,
                    "items": [{"run_id": run_id} for run_id in expected_ids],
                },
                "plan_sha256": evaluation_plan_sha256,
                "aggregation_input_spec": aggregation_spec_record,
            }

            final_record = completion.file_record(final_attestation)
            preflight_record = completion.file_record(preflight_attestation)
            source_snapshot_record = completion.file_record(
                snapshot_root / "source_snapshot.json"
            )
            finalization_evidence = {
                "schema": completion.TABLE_C_FINALIZATION_LINEAGE_EVIDENCE_SCHEMA,
                "status": "passed",
                "final_attestation": final_record,
                "preflight_attestation": preflight_record,
                "archived_source_snapshot": source_snapshot_record,
                "historical_finalizer_source_record": dict(
                    completion.TABLE_C_HISTORICAL_FINALIZER_SOURCE_IDENTITY
                ),
                "historical_finalizer_live_bytes_required": False,
                "semantic_transformation_replayed": True,
                "archived_source_manifest_sha256": "b" * 64,
                "archived_object_records_sha256": object_records_sha256,
            }
            finalization_evidence["evidence_sha256"] = (
                completion.canonical_json_sha256(finalization_evidence)
            )
            finalization_result = {
                "status": "passed",
                "staged_upgrade_replayed": True,
                "historical_auditor_archived": True,
                "historical_finalizer_evidence_bound": True,
                "historical_finalizer_live_bytes_required": False,
                "all_authoritative_sources_archived": True,
                "source_count": 89,
                "object_count": 89,
                "archived_source_manifest_sha256": "b" * 64,
                "archived_object_records_sha256": object_records_sha256,
                "_archived_object_records": object_records,
            }
            snapshot_identity = {
                **completion.TABLE_C_TRAINING_SNAPSHOT_IDENTITY,
                "completion_subreceipt_file_sha256": completion.file_record(
                    snapshot_root / "completion_subreceipt.json"
                )["sha256"],
                "source_snapshot_file_sha256": source_snapshot_record["sha256"],
            }
            snapshot_result = {
                "status": "passed",
                "live_completion_record_count": 443,
            }
            recovery_record = completion._table_c_mtime_file_record(
                recovery_receipt_path
            )
            recovery_verifier = completion._table_c_mtime_file_record(
                Path(recovery.__file__)
            )
            validation_recovery_record = completion._table_c_mtime_file_record(
                validation_recovery_receipt_path
            )
            validation_recovery_verifier = completion._table_c_mtime_file_record(
                Path(validation_recovery.__file__)
            )
            training_snapshot_verifier = completion.file_record(
                Path(snapshot.__file__)
            )
            supplemental_result = {
                "status": "passed_supplemental_self_hashed",
                "authoritative_dependency_proof": False,
            }
            mutate_supplemental = False

            def load_training_queue(path):
                return training_queues[str(Path(path).resolve())]

            def mutate_after_supplemental_replay():
                stale_record = completion.file_record(supplemental_attestation)
                if mutate_supplemental:
                    supplemental_attestation.write_text(
                        supplemental_attestation.read_text(encoding="ascii") + " ",
                        encoding="ascii",
                    )
                return supplemental_result, stale_record

            patchers = (
                mock.patch.object(
                    completion,
                    "TABLE_C_FINAL_DEPENDENCY_ATTESTATION",
                    final_attestation,
                ),
                mock.patch.object(
                    completion,
                    "TABLE_C_PREFLIGHT_DEPENDENCY_ATTESTATION",
                    preflight_attestation,
                ),
                mock.patch.object(
                    completion,
                    "TABLE_C_CURRENT_POLICY_FINAL_DEPENDENCY_ATTESTATION",
                    supplemental_attestation,
                ),
                mock.patch.object(
                    completion,
                    "TABLE_C_PRETRAINING_RECOVERY_RECEIPT",
                    recovery_receipt_path,
                ),
                mock.patch.object(
                    completion,
                    "TABLE_C_VALIDATION_RECOVERY_RECEIPT",
                    validation_recovery_receipt_path,
                ),
                mock.patch.object(
                    completion, "TABLE_C_AGGREGATION_SPEC", aggregation_spec_path
                ),
                mock.patch.object(
                    completion,
                    "TABLE_C_AGGREGATION_REPORT",
                    aggregation_report_path,
                ),
                mock.patch.object(
                    completion, "TABLE_C_TRAINING_SNAPSHOT", snapshot_root
                ),
                mock.patch.object(
                    completion,
                    "TABLE_C_TRAINING_SNAPSHOT_IDENTITY",
                    snapshot_identity,
                ),
                mock.patch.object(
                    matrix_queue,
                    "DEFAULT_TRAINING_QUEUE_DIRS",
                    training_dirs,
                ),
                mock.patch.object(
                    matrix_queue, "DEFAULT_QUEUE_DIR", validation_dir
                ),
                mock.patch.object(
                    completion,
                    "_validate_table_c_sequences",
                    return_value=(expected_ids, [{} for _ in expected_ids]),
                ),
                mock.patch.object(
                    completion,
                    "_verify_table_c_training_snapshot",
                    return_value=(snapshot_result, training_snapshot_verifier),
                ),
                mock.patch.object(
                    completion,
                    "_verify_table_c_archived_finalization_lineage",
                    return_value=(finalization_result, finalization_evidence),
                ),
                mock.patch.object(
                    completion,
                    "_verify_table_c_supplemental_final_dependency_attestation",
                    side_effect=mutate_after_supplemental_replay,
                ),
                mock.patch.object(
                    serial_queue,
                    "load_queue",
                    side_effect=load_training_queue,
                ),
                mock.patch.object(
                    serial_queue,
                    "verify_queue",
                    return_value={"status": "passed"},
                ),
                mock.patch.object(
                    completion,
                    "_verify_table_c_pretraining_recovery",
                    return_value=(
                        recovery_record,
                        recovery_verifier,
                        {"status": "passed"},
                    ),
                ),
                mock.patch.object(
                    completion,
                    "_verify_table_c_validation_recovery",
                    return_value=(
                        validation_recovery_record,
                        validation_recovery_verifier,
                        {"status": "passed"},
                    ),
                ),
                mock.patch.object(
                    matrix_queue, "load_queue", return_value=evaluation
                ),
                mock.patch.object(
                    matrix_queue,
                    "verify_queue",
                    return_value={"status": "passed"},
                ),
                mock.patch.object(
                    completion,
                    "_verify_table_c_matrix_aggregate",
                    return_value={},
                ),
            )
            with ExitStack() as stack:
                for patcher in patchers:
                    stack.enter_context(patcher)
                passed = completion._verify_table_c()
                self.assertEqual(passed["status"], "completed")
                self.assertEqual(
                    passed["artifacts"][
                        "supplemental_current_policy_attestation"
                    ],
                    completion.file_record(supplemental_attestation),
                )
                mutate_supplemental = True
                with self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "Table-C canonical artifacts changed during semantic replay",
                ):
                    completion._verify_table_c()

    def test_table_c_archived_proof_tolerates_live_docs_and_serial_evolution(self):
        from tools import audit_stageb_table_c_dependency_closure as dependency

        preflight = json.loads(
            completion.TABLE_C_PREFLIGHT_DEPENDENCY_ATTESTATION.read_text(
                encoding="utf-8"
            )
        )
        docs = next(
            record
            for record in preflight["training_evidence"][
                "static_repository_sources"
            ]
            if Path(record["path"]).name == "paper_cvpr_ablation_protocol.md"
        )
        serial = next(
            record
            for record in preflight["auditor_sources"]
            if Path(record["path"]).name == "run_stageb_serial_matrix_queue.py"
        )
        historical_finalizer = Path(
            completion.TABLE_C_HISTORICAL_FINALIZER_SOURCE_IDENTITY["path"]
        )
        self.assertNotEqual(
            completion._sha256_file(Path(docs["path"])), docs["sha256"]
        )
        self.assertNotEqual(
            completion._sha256_file(Path(serial["path"])), serial["sha256"]
        )
        forbidden = {
            Path(docs["path"]).resolve(),
            Path(serial["path"]).resolve(),
            historical_finalizer.resolve(),
        }
        original_sha = completion._sha256_file

        def reject_live_source_hash(path):
            if Path(path).resolve() in forbidden:
                raise AssertionError(f"live historical source was hashed: {path}")
            return original_sha(path)

        with (
            mock.patch.object(
                completion, "_sha256_file", side_effect=reject_live_source_hash
            ),
            mock.patch.object(
                dependency,
                "verify_attestation",
                side_effect=AssertionError("live dependency replay is forbidden"),
            ) as replay,
        ):
            archived, _ = completion._verify_table_c_archived_finalization_lineage()
            supplemental, _ = (
                completion._verify_table_c_supplemental_final_dependency_attestation()
            )
        replay.assert_not_called()
        self.assertTrue(archived["all_authoritative_sources_archived"])
        self.assertEqual(
            supplemental["status"], "passed_supplemental_self_hashed"
        )

    def test_table_c_adapter_requires_durable_immutable_training_snapshot(self):
        from tools import build_stageb_table_c_u1000_training_snapshot as snapshot

        identity = {
            "completion_subreceipt_file_sha256": (
                "b7e467d94d9632dc2f5e44b0d5045b6da9241a0a3420ff9e9a4513e78b045d8c"
            ),
            "completion_subreceipt_sha256": (
                "461397059b3fc87256719926b611c2a8a499f44c40598caaba6565c819f0f48f"
            ),
            "source_snapshot_file_sha256": (
                "11e39a431ad73014ad8ca8308b835a21493c45639ae61e2eb90ca401b0b5d8a2"
            ),
            "source_snapshot_sha256": (
                "b28bcd208c6388e4b305db99a3eb2d4c206607b4b31af0759423937938b10538"
            ),
        }
        self.assertEqual(dict(completion.TABLE_C_TRAINING_SNAPSHOT_IDENTITY), identity)
        passed = {
            "status": "passed",
            "run_count": 33,
            "source_count": 89,
            "object_count": 89,
            "live_completion_record_count": 443,
            "live_source_parity_required": False,
            "live_parity_record_count": None,
            "strict_live_final_gates": None,
            **identity,
        }
        with mock.patch.object(
            snapshot, "verify_snapshot", return_value=passed
        ) as replay:
            result, source = completion._verify_table_c_training_snapshot()
        self.assertEqual(result, passed)
        self.assertEqual(source, completion.file_record(Path(snapshot.__file__)))
        replay.assert_called_once_with(
            completion.TABLE_C_TRAINING_SNAPSHOT,
            require_live_source_parity=False,
        )

        incomplete = dict(passed)
        incomplete["live_completion_record_count"] = 442
        with mock.patch.object(
            snapshot, "verify_snapshot", return_value=incomplete
        ), self.assertRaisesRegex(
            completion.CompletionReceiptError, "contract is incomplete"
        ):
            completion._verify_table_c_training_snapshot()

        for field in identity:
            for mutation in ("missing", "substituted"):
                with self.subTest(snapshot_identity=field, mutation=mutation):
                    substituted = dict(passed)
                    if mutation == "missing":
                        substituted.pop(field)
                    else:
                        substituted[field] = "0" * 64
                    with mock.patch.object(
                        snapshot, "verify_snapshot", return_value=substituted
                    ), self.assertRaisesRegex(
                        completion.CompletionReceiptError,
                        "snapshot identity drifted",
                    ):
                        completion._verify_table_c_training_snapshot()

    def test_table_c_adapter_closes_on_aggregate_identity_failure(self):
        from tools import aggregate_stageb_matrix_validation as aggregate

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "table_c_report.json"
            report.write_text("{}\n", encoding="ascii")
            failure = aggregate.MatrixValidationError(
                "formal matrix evaluation queue identity has not been sealed"
            )
            with mock.patch.object(
                completion, "TABLE_C_AGGREGATION_REPORT", report
            ), mock.patch.object(
                aggregate, "aggregate_spec", side_effect=failure
            ), self.assertRaisesRegex(
                completion.CompletionReceiptError,
                "aggregate replay failed: formal matrix evaluation queue identity",
            ):
                completion._verify_table_c_matrix_aggregate(aggregate)

    def test_headline_m0_hook_replays_deep_training_validation_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._headline_fixture(Path(temporary))
            with self._headline_patch_stack(fixture):
                result = completion._verify_headline_m0_adapter()

            self.assertEqual(result["adapter_id"], completion.HEADLINE_M0_ADAPTER_ID)
            self.assertEqual(result["contract"]["training_run_count"], 6)
            self.assertEqual(result["contract"]["validation_run_count"], 6)
            self.assertEqual(result["contract"]["optimizer_updates"], 23_532)
            self.assertEqual(
                result["contract"]["direction"], "M0N_minus_M0"
            )
            self.assertTrue(
                result["semantic_replay"][
                    "checkpoint_ancestry_optimizer_resume_numerical_telemetry_verified"
                ]
            )
            self.assertEqual(
                [call.args[:2] for call in fixture["training_mock"].call_args_list],
                [
                    (fixture["m0_queue"], "M0"),
                    (fixture["m0n_queue"], "M0N"),
                ],
            )
            self.assertTrue(
                all(
                    call.kwargs == {"require_completed": True}
                    for call in fixture["training_mock"].call_args_list
                )
            )
            fixture["report_mock"].assert_called_once_with(fixture["report_path"])

    def test_headline_m0_full_run_telemetry_projection_is_explicitly_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._headline_fixture(Path(temporary))
            training_runner = fixture["training_runner"]
            run = fixture["training"]["M0"]["completed_training_runs"][0]
            evidence_snapshot = completion._verify_headline_evidence_snapshot(
                training_runner,
                run["evidence_snapshot"],
                label="fixture completed evidence",
            )
            completion._verify_headline_full_run_telemetry(
                training_runner,
                run_id=run["run_id"],
                telemetry=run["telemetry"],
                ancestry=run["ancestry"],
                evidence_snapshot=evidence_snapshot,
            )

            mutations = {
                "schema": lambda value: value["full_run"].__setitem__(
                    "schema", "drifted"
                ),
                "attempt_count": lambda value: value["full_run"].__setitem__(
                    "attempt_count", 2
                ),
                "total_rows": lambda value: value["full_run"].__setitem__(
                    "sample_rows", 11
                ),
                "devices": lambda value: value["full_run"].__setitem__(
                    "devices", []
                ),
                "attempt_record": lambda value: value["full_run"]["attempts"][
                    0
                ].__setitem__("attempt_ordinal", 1),
                "attempt_digest": lambda value: value["full_run"]["attempts"][
                    0
                ].__setitem__("evidence_sha256", "0" * 64),
                "semantic_digest": lambda value: value["full_run"].__setitem__(
                    "semantic_sha256", "0" * 64
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    telemetry = copy.deepcopy(run["telemetry"])
                    mutate(telemetry)
                    with self.assertRaisesRegex(
                        completion.CompletionReceiptError, "telemetry"
                    ):
                        completion._verify_headline_full_run_telemetry(
                            training_runner,
                            run_id=run["run_id"],
                            telemetry=telemetry,
                            ancestry=run["ancestry"],
                            evidence_snapshot=evidence_snapshot,
                        )

    def test_headline_m0_hook_rejects_deep_training_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._headline_fixture(Path(temporary))
            tampered = copy.deepcopy(fixture["training"]["M0"])
            tampered["completed_training_runs"][0]["semantic_sha256"] = "0" * 64

            def training_replay(_queue, contract_id, **_kwargs):
                return tampered if contract_id == "M0" else fixture["training"]["M0N"]

            with (
                self._headline_patch_stack(
                    fixture, training_side_effect=training_replay
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "training semantic SHA-256 drifted",
                ),
            ):
                completion._verify_headline_m0_adapter()

    def test_headline_m0_hook_rejects_swapped_training_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._headline_fixture(Path(temporary))

            def swapped(_queue, contract_id, **_kwargs):
                other = "M0N" if contract_id == "M0" else "M0"
                return fixture["training"][other]

            with (
                self._headline_patch_stack(
                    fixture, training_side_effect=swapped
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "M0 training queue completion is incomplete",
                ),
            ):
                completion._verify_headline_m0_adapter()

    def test_headline_m0_hook_rejects_validation_training_binding_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._headline_fixture(Path(temporary))
            manifest = copy.deepcopy(fixture["validation_manifest"])
            manifest["plan"]["training_queues"][0]["queue_id"] = (
                "00000000-0000-0000-0000-000000000099"
            )
            with (
                self._headline_patch_stack(
                    fixture, validation_manifest=manifest
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "not bound to M0 training",
                ),
            ):
                completion._verify_headline_m0_adapter()

    def test_headline_m0_hook_rejects_report_digest_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._headline_fixture(Path(temporary))
            report = copy.deepcopy(fixture["report"])
            report["report_sha256"] = "0" * 64
            with (
                self._headline_patch_stack(fixture, report=report),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "aggregate report contract is incomplete",
                ),
            ):
                completion._verify_headline_m0_adapter()

    def test_headline_m0_hook_rejects_persistent_verifier_source_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._headline_fixture(root)
            aggregate_source = root / "aggregate_verifier.py"
            aggregate_source.write_text("version = 1\n", encoding="ascii")

            def mutate_source(_path):
                aggregate_source.write_text("version = 2\n", encoding="ascii")
                return fixture["report"]

            with (
                self._headline_patch_stack(
                    fixture,
                    aggregate_source=aggregate_source,
                    report_side_effect=mutate_source,
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "artifacts changed during semantic replay",
                ),
            ):
                completion._verify_headline_m0_adapter()

    def test_headline_m0_hook_rejects_training_evidence_mutated_after_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._headline_fixture(Path(temporary))
            snapshot = fixture["training"]["M0"]["completed_training_runs"][0][
                "evidence_snapshot"
            ]
            evidence_path = Path(snapshot["records"][0]["path"])

            def mutate_evidence(_path):
                evidence_path.write_text(
                    "mutated after completed-training replay\n", encoding="ascii"
                )
                return fixture["report"]

            with (
                self._headline_patch_stack(
                    fixture, report_side_effect=mutate_evidence
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "completed-training evidence changed after semantic replay",
                ),
            ):
                completion._verify_headline_m0_adapter()

    def test_table_b_v2_hook_replays_exact_queue_bound_aggregate(self):
        from tools import aggregate_stageb_table_b_v2_validation as aggregate
        from tools import run_stageb_table_b_v2_queue as training_queue
        from tools import run_stageb_table_b_v2_validation_queue as validation_queue
        from util import stage_b_table_b_v2_contract as contract

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_dir = root / "training"
            validation_dir = root / "validation"
            report_path = root / "aggregate.json"
            for path in (
                training_dir / "queue.json",
                training_dir / training_queue.SOURCE_PLAN_NAME,
                training_dir / training_queue.SCOPE_PLAN_NAME,
                training_dir / training_queue.COMPLETION_NAME,
                validation_dir / "queue.json",
                validation_dir / validation_queue.VALIDATION_SPEC_NAME,
            ):
                _write_json(path)
            training_ids = list(contract.FORMAL_RUN_IDS)
            training = {
                "schema": training_queue.COMPLETION_SCHEMA,
                "status": "passed",
                "profile": contract.FORMAL_PROFILE,
                "ordered_run_ids": training_ids,
                "queue": {"queue_id": "training-id", "plan_sha256": "a" * 64},
                "common_input_replay": {
                    "status": "passed",
                    "all_six_runs_share_identical_common_inputs": True,
                    "only_declared_condition_inputs_differ": True,
                },
                "runs": {run_id: {"run_root": run_id} for run_id in training_ids},
            }
            training["semantic_sha256"] = training_queue._semantic_sha256(training)
            validation_ids = list(validation_queue.RUN_IDS)
            validation_manifest = {
                "status": "completed",
                "plan": {
                    "training_queue": {
                        "queue_dir": str(training_dir.resolve()),
                        "queue_id": "training-id",
                        "plan_sha256": "a" * 64,
                        "completion_semantic_sha256": training["semantic_sha256"],
                    }
                },
            }
            validation = {
                "schema": validation_queue.VERIFICATION_SCHEMA,
                "status": "passed",
                "queue_status": "completed",
                "queue_id": "validation-id",
                "plan_sha256": "b" * 64,
                "ordered_seeds": list(validation_queue.SEEDS),
                "phase_order_per_seed": list(validation_queue.PHASE_ORDER),
                "total_phase_count": len(validation_queue.SEEDS)
                * len(validation_queue.PHASE_ORDER),
                "verified_items": [{"run_id": run_id} for run_id in validation_ids],
            }
            report = {
                "schema": aggregate.REPORT_SCHEMA,
                "status": "validated_formal_v2_supplemental_diagnostic",
                "formal_global_fpr_eligible": False,
                "created_at_utc": "persisted",
                "formal_evaluation_protocol": {
                    "training_source_contract": "table_b_v2_formal"
                },
                "validation": {
                    "formal_v2_training_resolver_replayed": True,
                    "exact_three_seed_six_phase_queue_replayed": True,
                    "validation_queue_spec_replayed": True,
                    "shared_gpu_lease_queue_verified": True,
                },
                "inputs": {
                    "formal_v2_validation_queue": {
                        "queue_id": "validation-id",
                        "plan_sha256": "b" * 64,
                        "training_queue_id": "training-id",
                        "training_queue_plan_sha256": "a" * 64,
                        "ordered_seeds": list(validation_queue.SEEDS),
                        "phase_order_per_seed": list(validation_queue.PHASE_ORDER),
                        "total_phase_count": len(validation_queue.SEEDS)
                        * len(validation_queue.PHASE_ORDER),
                    }
                },
            }
            _write_json(report_path, report)
            replay = {**copy.deepcopy(report), "created_at_utc": "replayed"}
            with (
                mock.patch.object(
                    completion, "TABLE_B_V2_TRAINING_QUEUE", training_dir
                ),
                mock.patch.object(
                    completion, "TABLE_B_V2_VALIDATION_QUEUE", validation_dir
                ),
                mock.patch.object(
                    completion, "TABLE_B_V2_VALIDATION_AGGREGATE", report_path
                ),
                mock.patch.object(
                    training_queue, "verify_formal_queue", return_value=training
                ) as train_replay,
                mock.patch.object(
                    validation_queue, "load_queue", return_value=validation_manifest
                ),
                mock.patch.object(
                    validation_queue, "verify_queue", return_value=validation
                ),
                mock.patch.object(
                    aggregate, "aggregate", return_value=replay
                ) as aggregate_replay,
            ):
                result = completion._verify_table_b_v2_adapter()
            self.assertEqual(result["adapter_id"], completion.TABLE_B_V2_ADAPTER_ID)
            self.assertEqual(result["contract"]["training_run_count"], 6)
            self.assertEqual(result["contract"]["validation_phase_count"], 6)
            train_replay.assert_called_once_with(training_dir, persist=False)
            aggregate_replay.assert_called_once_with(validation_dir)

            tampered = copy.deepcopy(replay)
            tampered["inputs"]["formal_v2_validation_queue"]["training_queue_id"] = (
                "other"
            )
            with (
                mock.patch.object(
                    completion, "TABLE_B_V2_TRAINING_QUEUE", training_dir
                ),
                mock.patch.object(
                    completion, "TABLE_B_V2_VALIDATION_QUEUE", validation_dir
                ),
                mock.patch.object(
                    completion, "TABLE_B_V2_VALIDATION_AGGREGATE", report_path
                ),
                mock.patch.object(
                    training_queue, "verify_formal_queue", return_value=training
                ),
                mock.patch.object(
                    validation_queue, "load_queue", return_value=validation_manifest
                ),
                mock.patch.object(
                    validation_queue, "verify_queue", return_value=validation
                ),
                mock.patch.object(aggregate, "aggregate", return_value=tampered),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError, "differs from semantic replay"
                ),
            ):
                completion._verify_table_b_v2_adapter()

            aggregate_source = root / "aggregate_verifier.py"
            aggregate_source.write_text("version = 1\n", encoding="ascii")

            def mutate_verifier_source(_queue_dir):
                aggregate_source.write_text("version = 2\n", encoding="ascii")
                return replay

            with (
                mock.patch.object(
                    completion, "TABLE_B_V2_TRAINING_QUEUE", training_dir
                ),
                mock.patch.object(
                    completion, "TABLE_B_V2_VALIDATION_QUEUE", validation_dir
                ),
                mock.patch.object(
                    completion, "TABLE_B_V2_VALIDATION_AGGREGATE", report_path
                ),
                mock.patch.object(aggregate, "__file__", str(aggregate_source)),
                mock.patch.object(
                    training_queue, "verify_formal_queue", return_value=training
                ),
                mock.patch.object(
                    validation_queue, "load_queue", return_value=validation_manifest
                ),
                mock.patch.object(
                    validation_queue, "verify_queue", return_value=validation
                ),
                mock.patch.object(
                    aggregate, "aggregate", side_effect=mutate_verifier_source
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "artifacts changed during semantic replay",
                ),
            ):
                completion._verify_table_b_v2_adapter()

    def test_table_d_hook_replays_final_receipt_and_queue_bound_aggregate(self):
        from tools import aggregate_stageb_table_d_formal_matrix as aggregate
        from tools import run_stageb_table_d_formal_queue as training_queue
        from tools import run_stageb_table_d_matrix_validation_queue as validation_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_dir = root / "training"
            validation_dir = root / "validation"
            report_path = root / "aggregate.json"
            for path in (
                training_dir / "queue.json",
                training_dir / training_queue.SOURCE_PLAN_NAME,
                training_dir / training_queue.SCOPE_PLAN_NAME,
                training_dir / training_queue.COMPLETION_NAME,
                validation_dir / "queue.json",
                validation_dir / validation_queue.AGGREGATION_SPEC_NAME,
                validation_dir / validation_queue.EVALUATION_SCOPE_PLAN_NAME,
            ):
                _write_json(path)
            training_ids = list(training_queue.RUN_IDS)
            training = {
                "schema": training_queue.COMPLETION_SCHEMA,
                "status": "passed",
                "profile": training_queue.PROFILE,
                "ordered_run_ids": training_ids,
                "formal_training_contract": training_queue.FORMAL_TRAINING_CONTRACT,
                "queue": {"queue_id": "training-id", "plan_sha256": "a" * 64},
                "active_item_identity_replayed": True,
                "runs": {run_id: {"run_root": run_id} for run_id in training_ids},
            }
            training["semantic_sha256"] = training_queue._semantic_sha(training)
            validation_ids = list(validation_queue.JOB_IDS)
            final_verification = {
                "schema": validation_queue.FINAL_VERIFICATION_SCHEMA,
                "queue_id": "validation-id",
                "plan_sha256": "b" * 64,
                "ordered_job_ids": validation_ids,
                "training_completion_semantic_sha256": training["semantic_sha256"],
            }
            validation_manifest = {
                "status": "completed",
                "plan": {
                    "training_queue": {
                        "queue_dir": str(training_dir.resolve()),
                        "queue_id": "training-id",
                        "plan_sha256": "a" * 64,
                        "completion_semantic_sha256": training["semantic_sha256"],
                    }
                },
            }
            validation = {
                "schema": validation_queue.VERIFICATION_SCHEMA,
                "status": "passed",
                "queue_status": "completed",
                "profile": validation_queue.PROFILE,
                "queue_id": "validation-id",
                "plan_sha256": "b" * 64,
                "ordered_job_ids": validation_ids,
                "verified_items": [{"job_id": job_id} for job_id in validation_ids],
                "final_verification": final_verification,
            }
            report = {
                "schema": aggregate.REPORT_SCHEMA,
                "status": "validated_matrix_validation_only",
                "created_at_utc": "persisted",
                "formal_test_or_strict_result": False,
                "protocol": {
                    "paired_bootstrap": {
                        "iterations": aggregate.FORMAL_BOOTSTRAP_ITERATIONS,
                        "confidence": aggregate.FORMAL_BOOTSTRAP_CONFIDENCE,
                        "seed": aggregate.FORMAL_BOOTSTRAP_SEED,
                        "unit": "image cluster within training seed",
                        "seed_first": True,
                    }
                },
                "validation": {
                    "pass": True,
                    "exact_fifteen_final_jobs": True,
                    "exact_three_s3_rank_jobs": True,
                    "record_identities_aligned": True,
                    "runtime_code_data_surface_equal": True,
                    "input_rehash_and_postflight_replayed": True,
                    "training_authority_replayed": True,
                },
                "inputs": {
                    "evaluation_queue": {
                        "queue_id": "validation-id",
                        "plan_sha256": "b" * 64,
                        "final_verification": final_verification,
                    },
                    "final_diagnostics": {
                        "status": "not_bound",
                        "separate_from_matrix_metrics": True,
                        "pooled_into_matrix_results": False,
                    },
                },
                "experiments": {
                    row: {} for row in (*training_queue.ROWS, "S3_rank")
                },
                "comparisons": {
                    "clean_ownership_vs_S0": {},
                    "S2F_minus_S2_full_objective_control": {},
                    "S3_confidence_minus_rank_diagnostic": {},
                },
            }
            _write_json(report_path, report)
            replay = {**copy.deepcopy(report), "created_at_utc": "replayed"}
            with (
                mock.patch.object(completion, "TABLE_D_TRAINING_QUEUE", training_dir),
                mock.patch.object(
                    completion, "TABLE_D_VALIDATION_QUEUE", validation_dir
                ),
                mock.patch.object(
                    completion, "TABLE_D_VALIDATION_AGGREGATE", report_path
                ),
                mock.patch.object(
                    training_queue, "verify_training_queue", return_value=training
                ) as train_replay,
                mock.patch.object(
                    validation_queue, "load_queue", return_value=validation_manifest
                ),
                mock.patch.object(
                    validation_queue, "verify_queue", return_value=validation
                ),
                mock.patch.object(aggregate, "aggregate", return_value=replay),
            ):
                result = completion._verify_table_d_adapter()
            self.assertEqual(result["adapter_id"], completion.TABLE_D_ADAPTER_ID)
            self.assertEqual(result["contract"]["training_run_count"], 15)
            self.assertEqual(result["contract"]["validation_job_count"], 18)
            train_replay.assert_called_once_with(training_dir, persist=False)

            drifted_validation = copy.deepcopy(validation)
            drifted_validation["final_verification"][
                "training_completion_semantic_sha256"
            ] = "0" * 64
            with (
                mock.patch.object(completion, "TABLE_D_TRAINING_QUEUE", training_dir),
                mock.patch.object(
                    completion, "TABLE_D_VALIDATION_QUEUE", validation_dir
                ),
                mock.patch.object(
                    completion, "TABLE_D_VALIDATION_AGGREGATE", report_path
                ),
                mock.patch.object(
                    training_queue, "verify_training_queue", return_value=training
                ),
                mock.patch.object(
                    validation_queue, "load_queue", return_value=validation_manifest
                ),
                mock.patch.object(
                    validation_queue,
                    "verify_queue",
                    return_value=drifted_validation,
                ),
                mock.patch.object(aggregate, "aggregate", return_value=replay),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError, "not exact or training-bound"
                ),
            ):
                completion._verify_table_d_adapter()

    def test_new_formal_hooks_fail_closed_when_canonical_artifacts_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with mock.patch.object(
                completion, "HEADLINE_M0_TRAINING_QUEUE", missing
            ), self.assertRaisesRegex(
                completion.CompletionReceiptError, "cannot bind completion artifact"
            ):
                completion._verify_headline_m0_adapter()
            with mock.patch.object(
                completion, "TABLE_B_V2_TRAINING_QUEUE", missing
            ), self.assertRaisesRegex(
                completion.CompletionReceiptError, "cannot bind completion artifact"
            ):
                completion._verify_table_b_v2_adapter()
            with mock.patch.object(
                completion, "TABLE_D_TRAINING_QUEUE", missing
            ), self.assertRaisesRegex(
                completion.CompletionReceiptError, "cannot bind completion artifact"
            ):
                completion._verify_table_d_adapter()
            with mock.patch.object(
                completion, "G0C_SOAK_QUEUE", missing
            ), self.assertRaisesRegex(
                completion.CompletionReceiptError, "cannot bind completion artifact"
            ):
                completion._verify_g0c_queue_adapter()

    def test_new_formal_hook_canonical_paths_are_fixed(self):
        root = completion.REPO_ROOT / "outputs/paper_cvpr_v1"
        self.assertEqual(
            completion.HEADLINE_M0_TRAINING_QUEUE,
            root / "queues/headline_m0_training_u23532_v1",
        )
        self.assertEqual(
            completion.HEADLINE_M0N_TRAINING_QUEUE,
            root / "queues/headline_m0n_training_u23532_v1",
        )
        self.assertEqual(
            completion.HEADLINE_M0_VALIDATION_QUEUE,
            root / "queues/headline_m0_m0n_validation_v1",
        )
        self.assertEqual(
            completion.HEADLINE_M0_VALIDATION_AGGREGATE,
            root / "aggregates/headline_m0_m0n_validation_report.json",
        )
        self.assertEqual(
            completion.TABLE_B_V2_TRAINING_QUEUE,
            root / "queues/table_b_v2_training_u1000_v1",
        )
        self.assertEqual(
            completion.TABLE_B_V2_VALIDATION_QUEUE,
            root / "queues/table_b_v2_validation_v1",
        )
        self.assertEqual(
            completion.TABLE_B_V2_VALIDATION_AGGREGATE,
            root / "aggregates/table_b_v2_validation_report.json",
        )
        self.assertEqual(
            completion.TABLE_D_TRAINING_QUEUE,
            root / "queues/table_d_formal_training_u1000_v1",
        )
        self.assertEqual(
            completion.TABLE_D_VALIDATION_QUEUE,
            root / "queues/table_d_matrix_validation_v1",
        )
        self.assertEqual(
            completion.TABLE_D_VALIDATION_AGGREGATE,
            root / "aggregates/table_d_formal_matrix_report.json",
        )
        self.assertEqual(
            completion.G0C_SOAK_QUEUE,
            root / "queues/table_a_g0c_soak_u50_v1",
        )

    def test_production_registry_is_exact_and_all_blocks_are_sealed(self):
        self.assertEqual(tuple(completion.BLOCK_ADAPTER_REGISTRY), completion.BLOCKS)
        projection = completion._registry_projection()
        self.assertEqual(set(projection), set(completion.BLOCKS))
        self.assertEqual(projection["C"]["state"], "sealed")
        self.assertEqual(
            projection["C"]["adapter_id"],
            "table_c_training_validation_aggregate/v5",
        )
        self.assertEqual(projection["A"]["state"], "sealed")
        self.assertEqual(
            projection["A"]["adapter_id"],
            "headline_m0_m0n_training_validation_aggregate/v1",
        )
        self.assertIs(
            completion.BLOCK_ADAPTER_REGISTRY["A"].verifier,
            completion._verify_headline_m0_adapter,
        )
        for block in completion.BLOCKS:
            self.assertEqual(projection[block]["state"], "sealed")

    def test_g0c_hook_replays_exact_queues_and_validation_aggregate(self):
        from tools import aggregate_stageb_table_a_results as aggregate
        from tools import run_stageb_table_a_g0c_queues as g0c_queues
        from tools import run_stageb_table_a_g0c_soak_queue as soak_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            soak_dir = root / "soak"
            training_dir = root / "training"
            validation_dir = root / "validation"
            report_path = root / "table_a_three_seed.json"
            soak_artifacts = {
                name: root / f"soak-{name}.json"
                for name in ("plan", "postflight", "seal")
            }
            soak_artifacts["checkpoint"] = root / "soak-checkpoint.pth"
            for path in (
                soak_dir / "queue.json",
                training_dir / "queue.json",
                validation_dir / "queue.json",
                report_path,
                *soak_artifacts.values(),
            ):
                _write_json(path)
            soak_seal_record = completion.file_record(soak_artifacts["seal"])
            soak_evidence = {
                "schema": soak_queue.COMPLETION_SCHEMA,
                "run_id": soak_queue.RUN_ID,
                "seed": soak_queue.SEED,
                "micro_batch_size": soak_queue.MICRO_BATCH_SIZE,
                "gradient_accumulation_steps": (
                    soak_queue.GRADIENT_ACCUMULATION_STEPS
                ),
                "effective_global_batch": soak_queue.EFFECTIVE_GLOBAL_BATCH,
                "optimizer_updates": soak_queue.OPTIMIZER_UPDATES,
                "fresh_only": True,
                "lease_release_gate": (
                    "durable_completion_reload_plus_full_native_replay"
                ),
                "soak_plan": completion.file_record(soak_artifacts["plan"]),
                "postflight": completion.file_record(soak_artifacts["postflight"]),
                "checkpoint": completion.file_record(soak_artifacts["checkpoint"]),
                "soak_seal": soak_seal_record,
            }
            soak = {
                "schema": soak_queue.VERIFICATION_SCHEMA,
                "status": "passed",
                "queue_status": "completed",
                "run_id": soak_queue.RUN_ID,
                "completion_evidence": soak_evidence,
            }
            training_ids = list(g0c_queues.TRAINING_RUN_IDS)
            validation_ids = list(g0c_queues.VALIDATION_RUN_IDS)
            training = {
                "status": "passed",
                "queue_status": "completed",
                "queue_kind": g0c_queues.TRAINING_KIND,
                "queue_id": "g0c-training",
                "plan_sha256": "a" * 64,
                "ordered_run_ids": training_ids,
                "verified_items": [
                    {"run_id": run_id} for run_id in training_ids
                ],
            }
            training_manifest = {
                "plan": {
                    "items": [
                        {
                            "expected_plan": {
                                "inputs": {
                                    "g0c_soak_seal": {
                                        "path": soak_seal_record["path"],
                                        "sha256": soak_seal_record["sha256"],
                                    }
                                },
                                "soak_seal": {
                                    "schema": g0c_queues.training_runner.SOAK_SEAL_SCHEMA,
                                    "path": soak_seal_record["path"],
                                    "sha256": soak_seal_record["sha256"],
                                },
                            }
                        }
                        for _run_id in training_ids
                    ]
                }
            }
            validation = {
                "status": "passed",
                "queue_status": "completed",
                "queue_kind": g0c_queues.VALIDATION_KIND,
                "queue_id": "g0c-validation",
                "plan_sha256": "b" * 64,
                "ordered_run_ids": validation_ids,
                "verified_items": [
                    {"run_id": run_id} for run_id in validation_ids
                ],
            }
            report = {
                "schema": aggregate.SCHEMA,
                "status": "passed",
                "profile": aggregate.table_a.VALIDATION_PROFILE,
                "formal_seeds": list(g0c_queues.FORMAL_SEEDS),
                "report_sha256": "c" * 64,
            }
            with (
                mock.patch.object(completion, "G0C_SOAK_QUEUE", soak_dir),
                mock.patch.object(completion, "G0C_TRAINING_QUEUE", training_dir),
                mock.patch.object(completion, "G0C_VALIDATION_QUEUE", validation_dir),
                mock.patch.object(
                    completion, "G0C_VALIDATION_AGGREGATE", report_path
                ),
                mock.patch.object(
                    soak_queue, "verify_queue", return_value=soak
                ) as verify_soak,
                mock.patch.object(
                    g0c_queues,
                    "verify_queue",
                    side_effect=[training, validation],
                ) as verify_queue,
                mock.patch.object(
                    g0c_queues, "load_queue", return_value=training_manifest
                ),
                mock.patch.object(
                    aggregate, "verify_report", return_value=report
                ) as verify_report,
            ):
                result = completion._verify_g0c_queue_adapter()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["contract"]["training_run_count"], 3)
            self.assertEqual(result["contract"]["validation_run_count"], 6)
            self.assertEqual(result["contract"]["soak_optimizer_updates"], 50)
            self.assertEqual(
                result["contract"]["validation_run_ids"], validation_ids
            )
            self.assertEqual(
                [call.args[0] for call in verify_queue.call_args_list],
                [training_dir, validation_dir],
            )
            verify_report.assert_called_once_with(report_path)
            verify_soak.assert_called_once_with(soak_dir)
            self.assertIs(
                completion.BLOCK_ADAPTER_REGISTRY["G0c"].verifier,
                completion._verify_g0c_queue_adapter,
            )

            incomplete = copy.deepcopy(training)
            incomplete["verified_items"].pop()
            with (
                mock.patch.object(completion, "G0C_SOAK_QUEUE", soak_dir),
                mock.patch.object(completion, "G0C_TRAINING_QUEUE", training_dir),
                mock.patch.object(completion, "G0C_VALIDATION_QUEUE", validation_dir),
                mock.patch.object(
                    completion, "G0C_VALIDATION_AGGREGATE", report_path
                ),
                mock.patch.object(
                    soak_queue, "verify_queue", return_value=soak
                ),
                mock.patch.object(
                    g0c_queues,
                    "verify_queue",
                    side_effect=[incomplete, validation],
                ),
                mock.patch.object(
                    g0c_queues, "load_queue", return_value=training_manifest
                ),
                mock.patch.object(
                    aggregate, "verify_report", return_value=report
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError, "training queue"
                ),
            ):
                completion._verify_g0c_queue_adapter()

            wrong_soak_binding = copy.deepcopy(training_manifest)
            wrong_soak_binding["plan"]["items"][0]["expected_plan"]["inputs"][
                "g0c_soak_seal"
            ]["sha256"] = "0" * 64
            with (
                mock.patch.object(completion, "G0C_SOAK_QUEUE", soak_dir),
                mock.patch.object(completion, "G0C_TRAINING_QUEUE", training_dir),
                mock.patch.object(completion, "G0C_VALIDATION_QUEUE", validation_dir),
                mock.patch.object(
                    completion, "G0C_VALIDATION_AGGREGATE", report_path
                ),
                mock.patch.object(soak_queue, "verify_queue", return_value=soak),
                mock.patch.object(
                    g0c_queues,
                    "verify_queue",
                    side_effect=[training, validation],
                ),
                mock.patch.object(
                    g0c_queues, "load_queue", return_value=wrong_soak_binding
                ),
                mock.patch.object(
                    aggregate, "verify_report", return_value=report
                ),
                self.assertRaisesRegex(
                    completion.CompletionReceiptError,
                    "not bound to the queue-owned U50 soak seal",
                ),
            ):
                completion._verify_g0c_queue_adapter()

    def test_build_is_fresh_only_and_verify_replays_every_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact.json"
            artifact.write_text('{"status":"passed"}\n', encoding="ascii")
            receipt = root / "paper_ablation_completion_receipt.json"
            registry = _fake_registry(artifact)
            with (
                mock.patch.object(completion, "CANONICAL_RECEIPT_PATH", receipt),
                mock.patch.object(completion, "BLOCK_ADAPTER_REGISTRY", registry),
            ):
                built = completion.build_receipt()
                verified = completion.verify_receipt()
                self.assertEqual(built, verified)
                self.assertEqual(set(built["completed_blocks"]), set(completion.BLOCKS))
                self.assertEqual(
                    built["adapter_registry"]["blocks"]["C"]["state"], "sealed"
                )
                with self.assertRaisesRegex(
                    completion.CompletionReceiptError, "already exists"
                ):
                    completion.build_receipt()

    def test_self_hash_tamper_fails_before_adapter_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact.json"
            artifact.write_text('{"status":"passed"}\n', encoding="ascii")
            receipt = root / "paper_ablation_completion_receipt.json"
            registry = _fake_registry(artifact)
            with (
                mock.patch.object(completion, "CANONICAL_RECEIPT_PATH", receipt),
                mock.patch.object(completion, "BLOCK_ADAPTER_REGISTRY", registry),
            ):
                completion.build_receipt()
                value = json.loads(receipt.read_text(encoding="ascii"))
                value["completed_before_final_gate"] = False
                receipt.write_text(json.dumps(value), encoding="ascii")
                with (
                    mock.patch.object(completion, "_derive_receipt_payload") as replay,
                    self.assertRaisesRegex(
                        completion.CompletionReceiptError, "self SHA-256"
                    ),
                ):
                    completion.verify_receipt()
                replay.assert_not_called()

    def test_release_validator_delegates_to_completion_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "paper_ablation_completion_receipt.json"
            receipt.write_text("{}\n", encoding="ascii")
            expected = {"schema": completion.SCHEMA, "status": "completed"}
            with (
                mock.patch.object(
                    completion, "CANONICAL_RECEIPT_PATH", receipt
                ),
                mock.patch.object(
                    release, "PAPER_ABLATION_COMPLETION_RECEIPT_PATH", receipt
                ),
                mock.patch.object(
                    completion, "verify_receipt", return_value=expected
                ) as verify,
            ):
                self.assertEqual(
                    release.validate_paper_ablation_completion_receipt(receipt),
                    expected,
                )
            verify.assert_called_once_with(receipt.resolve())

    def test_self_hashed_handwritten_receipt_cannot_bypass_fresh_adapter_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "paper_ablation_completion_receipt.json"
            payload = {
                "schema": completion.SCHEMA,
                "status": "completed",
                "completed_before_final_gate": True,
                "all_training_validation_diagnostics_completed": True,
                "adapter_registry": {
                    "schema": completion.REGISTRY_SCHEMA,
                    "blocks": {
                        block: {"adapter_id": "forged/v1", "state": "sealed"}
                        for block in completion.BLOCKS
                    },
                    "registry_source": {
                        "path": str(receipt),
                        "sha256": "0" * 64,
                        "size_bytes": 0,
                    },
                },
                "completed_blocks": {
                    block: {
                        "status": "completed",
                        "adapter_id": "forged/v1",
                        "contract": {"seeds": [17, 42, 73]},
                        "artifacts": {"forged": {"path": str(receipt)}},
                        "semantic_replay": {"claimed": True},
                    }
                    for block in completion.BLOCKS
                },
            }
            payload["receipt_sha256"] = completion.canonical_json_sha256(payload)
            receipt.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="ascii"
            )
            with (
                mock.patch.object(completion, "CANONICAL_RECEIPT_PATH", receipt),
                mock.patch.object(
                    release, "PAPER_ABLATION_COMPLETION_RECEIPT_PATH", receipt
                ),
                mock.patch.object(
                    completion,
                    "_derive_receipt_payload",
                    side_effect=completion.CompletionReceiptError(
                        "fresh adapter replay required"
                    ),
                ),
                self.assertRaisesRegex(
                    release.HeadlineReleaseError, "fresh adapter replay required"
                ),
            ):
                release.validate_paper_ablation_completion_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
