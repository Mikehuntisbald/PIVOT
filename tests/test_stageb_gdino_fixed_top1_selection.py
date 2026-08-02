import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.eval_stageb_gdino_fixed_top1_calibration import (
    CalibrationError,
    RUNTIME,
    run_evaluation,
)
from tools.compare_stageb_fpr95_records import exact_fpr95
from tools.stageb_eval_records import RECORD_SCHEMA
from tools.stageb_gdino_fixed_top1_selection import (
    CALIBRATION_COMPLETION_SCHEMA,
    CALIBRATION_SCORE_CONTRACT,
    DEFAULT_DATA_ROOT,
    FIXED_TOP1_SCHEMA,
    P0_SCHEMA,
    PARTITION_SCHEMA,
    SELECTION_SCHEMA,
    SelectionError,
    _checkpoint_run_prefix,
    _choose_partition,
    _image_identity_keys,
    _verify_summary_recomputed_from_records,
    _validate_no_strict_score_path,
    create_partition,
    create_selection,
    file_record,
    verify_partition,
    verify_calibration_completion,
    verify_selection,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _jsonl_record(path: Path, rows: int, images: int):
    return {**file_record(path), "rows": rows, "unique_images": images}


def _fixture_checkpoint_verification(role: str, iteration):
    return {
        "schema": CALIBRATION_COMPLETION_SCHEMA,
        "role": role,
        "fixture_iteration": iteration,
    }


def _accepted_rows(count: int):
    rows = []
    for index in range(count):
        image_id = index if index < count - 10 else count - 11
        dataset = "refcocoplus" if index % 2 == 0 else "refcocog"
        rows.append(
            {
                "sample_id": f"accepted-{index}",
                "image_id": image_id,
                "ann_id": index + 10_000,
                "ref_id": index + 20_000,
                "sent_id": index + 30_000,
                "file_name": f"COCO_train2014_{image_id:012d}.jpg",
                "dataset": dataset,
                "pair_source": "refcoco+_unc" if dataset == "refcocoplus" else "refcocog_google",
                "replace_category": ["color" if index % 3 else "size"],
                "split": "train",
            }
        )
    return rows


class FixedTop1PartitionTests(unittest.TestCase):
    def test_production_partition_rejects_unregistered_seed_or_salt_search(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with self.assertRaisesRegex(SelectionError, "pre-registered seed"):
                create_partition(
                    accepted_path=directory / "accepted.jsonl",
                    verification_audit_path=directory / "verification.json",
                    strict2031_path=directory / "strict-a.jsonl",
                    strict1607_path=directory / "strict-b.jsonl",
                    train_path=directory / "train.jsonl",
                    calibration_path=directory / "calibration.jsonl",
                    audit_path=directory / "partition.json",
                    seed=1,
                    salt_candidates=256,
                )

    def test_loader_resolved_image_identity_rejects_id_aliases(self):
        with tempfile.TemporaryDirectory() as raw:
            image = Path(raw) / "same.jpg"
            image.write_bytes(b"same image")
            rows = _accepted_rows(2)
            rows[0]["image_id"] = 1
            rows[1]["image_id"] = 2
            rows[0]["image_path"] = str(image)
            rows[1]["image_path"] = str(image)
            with self.assertRaisesRegex(SelectionError, "aliases image_ids"):
                _image_identity_keys(rows, context="attack", require_exists=True)

            rows[1]["image_path"] = "/missing/stale-alias.jpg"
            keys, identity = _image_identity_keys(
                rows, context="stale-path", require_exists=False
            )
            self.assertNotEqual(keys[0], keys[1])
            self.assertTrue(identity["image_id_resolved_path_bijection"])

    def test_adaptive_image_partition_is_deterministic_balanced_and_reachable(self):
        rows = _accepted_rows(3000)
        first = _choose_partition(rows, seed=20260712, salt_candidates=64)
        second = _choose_partition(rows, seed=20260712, salt_candidates=64)
        self.assertEqual(first["policy"], second["policy"])
        self.assertEqual(
            [row["sample_id"] for row in first["calibration_rows"]],
            [row["sample_id"] for row in second["calibration_rows"]],
        )
        self.assertGreaterEqual(len(first["calibration_rows"]), 1000)
        self.assertGreaterEqual(len(first["calibration_images"]), 500)
        self.assertEqual(first["recommended_max_target"], 100)
        self.assertEqual(first["readiness_errors"], [])
        self.assertFalse(first["train_images"] & first["calibration_images"])
        duplicate_image = rows[-1]["image_id"]
        placements = {
            "train" if row in first["train_rows"] else "calibration"
            for row in rows
            if row["image_id"] == duplicate_image
        }
        self.assertEqual(len(placements), 1)
        self.assertEqual(first["policy"]["desired_calibration_rows"], 1000)
        self.assertLessEqual(
            first["policy"]["selected_score"]["max_rate_deviation"]["float"],
            0.05,
        )

    def test_partition_files_replay_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            accepted = directory / "accepted.jsonl"
            verification = directory / "verification.json"
            strict2031 = directory / "holdout-a.jsonl"
            strict1607 = directory / "holdout-b.jsonl"
            train = directory / "train.jsonl"
            calibration = directory / "calibration.jsonl"
            audit = directory / "partition.json"
            _write_jsonl(accepted, _accepted_rows(3000))
            _write_json(verification, {"sealed": True})
            _write_jsonl(strict2031, [{"image_id": 999_999}])
            _write_jsonl(strict1607, [{"image_id": 999_999}])

            def fake_strict(path, **_kwargs):
                return (
                    {999_999},
                    {"/tmp/fixed-top1-strict-fixture.jpg"},
                    {
                        **file_record(path),
                        "rows": 1,
                        "unique_images": 1,
                        "image_identity": {"fixture": True},
                    },
                )

            with mock.patch(
                "tools.stageb_gdino_fixed_top1_selection._load_strict_images",
                side_effect=fake_strict,
            ), mock.patch(
                "tools.stageb_gdino_fixed_top1_selection._resolved_source_image",
                side_effect=lambda row, **_kwargs: f"/tmp/image-{int(row['image_id'])}.jpg",
            ):
                payload = create_partition(
                    accepted_path=accepted,
                    verification_audit_path=verification,
                    strict2031_path=strict2031,
                    strict1607_path=strict1607,
                    train_path=train,
                    calibration_path=calibration,
                    audit_path=audit,
                    source_validator=lambda *_args: None,
                )
                verified = verify_partition(audit)
            self.assertEqual(payload["schema"], PARTITION_SCHEMA)
            self.assertTrue(payload["selection_readiness"]["pass"])
            self.assertEqual(verified["recommended_max_target"], 100)
            train.write_text(train.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with mock.patch(
                "tools.stageb_gdino_fixed_top1_selection._load_strict_images",
                side_effect=fake_strict,
            ), mock.patch(
                "tools.stageb_gdino_fixed_top1_selection._resolved_source_image",
                side_effect=lambda row, **_kwargs: f"/tmp/image-{int(row['image_id'])}.jpg",
            ), self.assertRaisesRegex(SelectionError, "file identity drifted"):
                verify_partition(audit)


class FixedTop1SelectionTests(unittest.TestCase):
    @staticmethod
    def _manifest(directory: Path, count: int = 1000):
        path = directory / "calibration.jsonl"
        rows = [
            {
                "sample_id": f"cal-{index}",
                "image_id": index,
                "ann_id": index + 10_000,
                "ref_id": index + 20_000,
                "sent_id": index + 30_000,
                "split": "train",
                "replace_category": ["color" if index % 2 else "size"],
            }
            for index in range(count)
        ]
        _write_jsonl(path, rows)
        return path, rows

    @staticmethod
    def _records(path: Path, manifest: Path, rows, run_id: str, negative: float):
        manifest_sha = file_record(manifest)["sha256"]
        values = []
        for index, row in enumerate(rows):
            values.append(
                {
                    "schema": RECORD_SCHEMA,
                    "task": "tn",
                    "manifest_key": "tn_global",
                    "manifest_sha256": manifest_sha,
                    "manifest_n": len(rows),
                    "manifest_index": index,
                    "sample_id": row["sample_id"],
                    "split": "train",
                    "image_id": row["image_id"],
                    "ann_id": row["ann_id"],
                    "ref_id": row["ref_id"],
                    "sent_id": row["sent_id"],
                    "run_id": run_id,
                    "valid": True,
                    "pos_score": 0.5,
                    "neg_score": negative,
                    "pos_iou": 0.5,
                    "neg_iou": 0.0,
                }
            )
        _write_jsonl(path, values)

    @staticmethod
    def _completion(
        directory: Path,
        *,
        manifest_record,
        checkpoint: Path,
        checkpoint_audit: Path,
        role: str,
        iteration,
        records: Path,
        probe_record,
        partition_record,
        config_record,
    ):
        data_root = DEFAULT_DATA_ROOT.resolve()
        runtime_actual = {
            "device": "cuda:0",
            "device_type": "cuda",
            "cuda_device_index": 0,
            "cuda_device_name": "fixture-gpu",
            "cuda_device_capability": [8, 0],
            "effective_amp": True,
            "num_workers": 0,
            "data_root": str(data_root),
            "image_root": str(data_root / "COCO/coco2014/train2014"),
            "torch_version": "fixture",
            "torch_cuda_version": "fixture",
            "cudnn_version": 1,
            "environment": {"GFLOPS_DEBUG_SHILONG": None},
        }
        preflight = directory / "calibration_eval_preflight.json"
        summary = directory / "summary.json"
        record_rows = [
            json.loads(line)
            for line in records.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        recomputed = exact_fpr95(
            [row["pos_score"] for row in record_rows],
            [row["neg_score"] for row in record_rows],
        )
        forward_calls = 2 * (
            (len(record_rows) + int(RUNTIME["batch_size"]) - 1)
            // int(RUNTIME["batch_size"])
        )
        query_evidence = {
            "hook": "root_model_forward_hook",
            "checked_outputs": [
                "stage_b_gdino_confidence_score",
                "stage_b_gdino_base_score",
                "pred_boxes",
            ],
            "query_count_each_call": 900,
            "observed_forward_calls": forward_calls,
            "expected_forward_calls": forward_calls,
            "observed_examples_across_negative_positive": 2 * len(record_rows),
            "expected_examples_across_negative_positive": 2 * len(record_rows),
            "pass": True,
        }
        _write_json(
            preflight,
            {
                "schema": CALIBRATION_COMPLETION_SCHEMA,
                "kind": "calibration_evaluation_preflight",
                "input_role": role,
                "iteration": iteration,
                "manifest": manifest_record,
                "checkpoint": file_record(checkpoint),
                "checkpoint_audit": file_record(checkpoint_audit),
                "checkpoint_verification": _fixture_checkpoint_verification(
                    role, iteration
                ),
                "probe_preflight": probe_record,
                "partition_audit": partition_record,
                "config": config_record,
                "config_import_chain": [config_record],
                "runtime": RUNTIME,
                "runtime_actual": runtime_actual,
                "score_contract": CALIBRATION_SCORE_CONTRACT,
                "code": [file_record(ROOT / "tools/stageb_eval_records.py")],
                "selection_input_scope": "calibration_only",
                "strict_isolation": {
                    "strict_metric_inputs": [],
                    "strict_result_paths": [],
                    "strict_paths_consumed_for_scoring": False,
                },
            },
        )
        _write_json(
            summary,
            {
                "manifest_sha256": manifest_record["sha256"],
                "manifest_n": manifest_record["rows"],
                "invalid_records": 0,
                "records_jsonl": str(records.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "run_id": _checkpoint_run_prefix(checkpoint.resolve()),
                "batch_size": RUNTIME["batch_size"],
                "seed": RUNTIME["seed"],
                "max_batches": 0,
                "runtime_actual": runtime_actual,
                "score_contract": CALIBRATION_SCORE_CONTRACT,
                "num_pairs": len(record_rows),
                "threshold_at_95tpr": recomputed["threshold"],
                "actual_tpr_at_95tpr": recomputed["actual_tpr"],
                "fpr95tpr": recomputed["fpr"],
                "tn_fpr": recomputed["fpr"],
                "query_geometry_evidence": query_evidence,
            },
        )
        identity = {
            "rank_score_equals_base": True,
            "confidence_score_equals_base": True,
            "rank_residual_exact_zero": True,
            "confidence_gate_exact_zero": True,
        }
        completion = directory / "calibration_eval_complete.json"
        _write_json(
            completion,
            {
                "schema": CALIBRATION_COMPLETION_SCHEMA,
                "kind": "completed_calibration_evaluation",
                "input_role": role,
                "iteration": iteration,
                "selection_input_scope": "calibration_only",
                "strict_isolation": {
                    "strict_metric_inputs": [],
                    "strict_result_paths": [],
                    "strict_paths_consumed_for_scoring": False,
                },
                "checkpoint": file_record(checkpoint),
                "checkpoint_audit": file_record(checkpoint_audit),
                "p0_identity": identity if role == "p0" else None,
                "manifest": manifest_record,
                "manifest_rows": manifest_record["rows"],
                "preflight": file_record(preflight),
                "summary": file_record(summary),
                "records": file_record(records),
            },
        )
        return completion

    def test_records_replay_selects_unique_earliest_tied_milestone(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest, rows = self._manifest(directory)
            manifest_record = _jsonl_record(manifest, len(rows), len(rows))
            partition_audit = directory / "partition.json"
            _write_json(partition_audit, {"sealed": True})
            fake_partition = {
                "audit": file_record(partition_audit),
                "calibration": manifest_record,
                "recommended_max_target": 100,
                "selection_readiness": {"pass": True, "errors": []},
            }
            probe = directory / "probe_preflight.json"
            baseline = directory / "baseline.pth"
            baseline.write_bytes(b"baseline")
            config = directory / "config.py"
            config.write_text("stage_b_gdino_score_adapter = True\n", encoding="utf-8")
            _write_json(
                probe,
                {
                    "schema": FIXED_TOP1_SCHEMA,
                    "kind": "phase_preflight",
                    "phase": "fixed-top1-confidence",
                    "launch": {"max_target": 100, "milestones": [50, 100]},
                    "static": {
                        "config": file_record(config),
                        "partition": {"audit": file_record(partition_audit)},
                    },
                    "fixed_gdino_source_binding": {
                        "checkpoint_sha256": file_record(baseline)["sha256"],
                        "matches_rank_initial_baseline": True,
                    },
                },
            )
            calibration_root = directory / "eval"
            p0_checkpoint = directory / "p0.pth"
            p0_checkpoint.write_bytes(b"p0")
            p0_sidecar = directory / "p0.pth.audit.json"
            identity = {
                "rank_score_equals_base": True,
                "confidence_score_equals_base": True,
                "rank_residual_exact_zero": True,
                "confidence_gate_exact_zero": True,
            }
            _write_json(
                p0_sidecar,
                {
                    "schema": P0_SCHEMA,
                    "kind": "p0_checkpoint_audit",
                    "p0_checkpoint": file_record(p0_checkpoint),
                    "baseline": file_record(baseline),
                    "functional_identity": identity,
                },
            )
            p0_records = (
                calibration_root
                / "p0/per_example_records"
                / f"{_checkpoint_run_prefix(p0_checkpoint)}__tn_global.records.jsonl"
            )
            self._records(
                p0_records,
                manifest,
                rows,
                _checkpoint_run_prefix(p0_checkpoint),
                0.6,
            )
            self._completion(
                calibration_root / "p0",
                manifest_record=manifest_record,
                checkpoint=p0_checkpoint,
                checkpoint_audit=p0_sidecar,
                role="p0",
                iteration=None,
                records=p0_records,
                probe_record=file_record(probe),
                partition_record=file_record(partition_audit),
                config_record=file_record(config),
            )
            selected_checkpoints = {}
            for iteration in (50, 100):
                label = f"s{iteration:06d}"
                checkpoint = (
                    directory
                    / "milestones"
                    / f"checkpoint_iter_{iteration:06d}.pth"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(str(iteration).encode("ascii"))
                audit = checkpoint.with_suffix(".audit.json")
                _write_json(
                    audit,
                    {
                        "schema": FIXED_TOP1_SCHEMA,
                        "kind": "milestone_checkpoint",
                        "iteration": iteration,
                        "checkpoint": file_record(checkpoint),
                    },
                )
                records = (
                    calibration_root
                    / label
                    / "per_example_records"
                    / f"{_checkpoint_run_prefix(checkpoint)}__tn_global.records.jsonl"
                )
                self._records(
                    records,
                    manifest,
                    rows,
                    _checkpoint_run_prefix(checkpoint),
                    0.0,
                )
                self._completion(
                    calibration_root / label,
                    manifest_record=manifest_record,
                    checkpoint=checkpoint,
                    checkpoint_audit=audit,
                    role="milestone",
                    iteration=iteration,
                    records=records,
                    probe_record=file_record(probe),
                    partition_record=file_record(partition_audit),
                    config_record=file_record(config),
                )
                selected_checkpoints[iteration] = (checkpoint, audit)
            output = directory / "selection.json"
            with mock.patch(
                "tools.stageb_gdino_fixed_top1_selection.verify_partition",
                return_value=fake_partition,
            ), mock.patch(
                "tools.stageb_gdino_fixed_top1_selection.BOOTSTRAP_ITERATIONS", 100
            ), mock.patch(
                "tools.stageb_gdino_fixed_top1_selection.calibration_code_records",
                return_value=[file_record(ROOT / "tools/stageb_eval_records.py")],
            ), mock.patch(
                "tools.stageb_gdino_fixed_top1_selection.calibration_config_records",
                return_value=[file_record(config)],
            ), mock.patch(
                "tools.stageb_gdino_fixed_top1_selection.replay_calibration_checkpoint_verification",
                side_effect=lambda **kwargs: _fixture_checkpoint_verification(
                    kwargs["role"], kwargs["milestone_iteration"]
                ),
            ):
                payload = create_selection(
                    probe_preflight_path=probe,
                    calibration_root=calibration_root,
                    output=output,
                )
                verified = verify_selection(
                    output,
                    expected_checkpoint=selected_checkpoints[50][0],
                    expected_milestone_audit=selected_checkpoints[50][1],
                    expected_calibration_root=calibration_root,
                )
                with self.assertRaisesRegex(
                    SelectionError, "unexpected calibration evaluation root"
                ):
                    verify_selection(
                        output,
                        expected_calibration_root=directory / "wrong-calibration-root",
                    )

                p0_completion = (
                    calibration_root / "p0/calibration_eval_complete.json"
                )
                p0_preflight = (
                    calibration_root / "p0/calibration_eval_preflight.json"
                )
                original_completion = json.loads(
                    p0_completion.read_text(encoding="utf-8")
                )
                original_preflight = json.loads(
                    p0_preflight.read_text(encoding="utf-8")
                )
                shallow_preflight = dict(original_preflight)
                shallow_preflight["code"] = [file_record(config)]
                _write_json(p0_preflight, shallow_preflight)
                shallow_completion = dict(original_completion)
                shallow_completion["preflight"] = file_record(p0_preflight)
                _write_json(p0_completion, shallow_completion)
                with self.assertRaisesRegex(
                    SelectionError, "dependency closure drifted"
                ):
                    verify_calibration_completion(
                        p0_completion,
                        expected_manifest=manifest_record,
                        expected_role="p0",
                        expected_iteration=None,
                    )

                _write_json(p0_preflight, original_preflight)
                _write_json(p0_completion, original_completion)
                external_records = directory / "copied-records.jsonl"
                external_records.write_bytes(p0_records.read_bytes())
                external_completion = dict(original_completion)
                external_completion["records"] = file_record(external_records)
                _write_json(p0_completion, external_completion)
                with self.assertRaisesRegex(
                    SelectionError, "records path/name"
                ):
                    verify_calibration_completion(
                        p0_completion,
                        expected_manifest=manifest_record,
                        expected_role="p0",
                        expected_iteration=None,
                    )
            self.assertEqual(payload["schema"], SELECTION_SCHEMA)
            self.assertEqual(payload["selected_iteration"], 50)
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["input_scope"], "calibration_only")
            self.assertFalse(verified["strict_paths_consumed_for_scoring"])
            self.assertEqual(
                payload["selection_input_contract"]["strict_metric_inputs"], []
            )

    def test_strict_paths_are_rejected_as_selection_score_inputs(self):
        paths = (
            "/tmp/strict2031/results.jsonl",
            "/tmp/stageb_vlm_verified_strict_ann_umd_val_20260711/results.jsonl",
        )
        for path in paths:
            with self.subTest(path=path), self.assertRaisesRegex(
                SelectionError, "strict score/result path"
            ):
                _validate_no_strict_score_path(
                    {"records": path},
                    context="test inputs",
                )

    def test_summary_fpr_cannot_be_replaced_after_records_are_sealed(self):
        checkpoint = Path("/tmp/probe/milestones/checkpoint_iter_000050.pth")
        records = SimpleNamespace(
            valid=np.asarray([True, True], dtype=np.bool_),
            positive=np.asarray([0.9, 0.8], dtype=np.float64),
            negative=np.asarray([0.2, 0.1], dtype=np.float64),
            run_ids=(_checkpoint_run_prefix(checkpoint),),
        )
        completion = {
            "checkpoint": {"path": str(checkpoint)},
            "summary_value": {
                "num_pairs": 2,
                "threshold_at_95tpr": 0.8,
                "actual_tpr_at_95tpr": 1.0,
                "fpr95tpr": 1.0,
                "tn_fpr": 1.0,
            },
        }
        with self.assertRaisesRegex(SelectionError, "not exactly recomputed"):
            _verify_summary_recomputed_from_records(completion, records)

    def test_deploy_runtime_and_launcher_do_not_screen_on_formal_strict(self):
        self.assertEqual(RUNTIME["batch_size"], 16)
        self.assertEqual(RUNTIME["resize_short_side"], 800)
        self.assertEqual(RUNTIME["max_size"], 1333)
        self.assertEqual(
            RUNTIME["forward_order"], "negative_then_positive_separate_calls"
        )
        self.assertEqual(RUNTIME["score"], "max_over_900_query_confidence_score")
        launcher = (
            ROOT / "tools/run_stageb_gdino_fixed_top1_calibration.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("run_stageb_fixed_protocol_eval", launcher)
        self.assertNotIn("strict2031", launcher)
        self.assertNotIn("strict1607", launcher)
        self.assertIn("verify-calibration", launcher)
        self.assertIn("verify-selection", launcher)
        self.assertIn("no calibration or selection output was created", launcher)
        dataset_config = json.loads(
            (
                ROOT
                / "config/datasets_stageb_gdino_adapter_fixed_top1_verified_pairs.json"
            ).read_text(encoding="utf-8")
        )
        annotation = dataset_config["train"][0]["anno"]
        self.assertTrue(annotation.endswith("/train_pairs.jsonl"))
        self.assertNotIn("accepted_pairs.jsonl", annotation)

    def test_calibration_rejects_cpu_and_transform_debug_environment_early(self):
        args = SimpleNamespace(device="cpu", num_workers=0)
        with self.assertRaisesRegex(CalibrationError, "requires CUDA"):
            run_evaluation(args)
        with mock.patch.dict(
            os.environ, {"GFLOPS_DEBUG_SHILONG": "INFO"}, clear=False
        ), self.assertRaisesRegex(CalibrationError, "GFLOPS_DEBUG_SHILONG"):
            run_evaluation(args)

        evaluator = (
            ROOT / "tools/eval_stageb_gdino_fixed_top1_calibration.py"
        ).read_text(encoding="utf-8")
        self.assertIn("data_aug_scale_overlap", evaluator)
        self.assertIn("register_forward_hook", evaluator)
        self.assertIn("query_count_each_call", evaluator)


if __name__ == "__main__":
    unittest.main()
