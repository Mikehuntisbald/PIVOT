import copy
import contextlib
import hashlib
import json
import shlex
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import aggregate_stageb_matrix_validation as matrix
from tools import aggregate_stageb_paper_results as paper_aggregate
from tools import run_stageb_paper_evaluations as evaluator
from tools.compare_stageb_fpr95_records import exact_fpr95
from tools.stageb_screen_calibration import build_manifest, summary_fields
from tools.stageb_ref_split_contract import REF_SPLIT_MANIFEST_FILES


PRODUCTION_LOCKED_TRAINING_VALIDATOR = matrix._validate_locked_training_identity
PRODUCTION_EVALUATION_QUEUE_VALIDATOR = (
    matrix._validate_evaluation_queue_binding
)


SMALL_REF_COUNTS = {
    "refcoco_val": 6,
    "refcocop_val": 5,
    "refcocog_val": 4,
}


def _ref_manifest_rows(split: str) -> list[dict]:
    split_index = matrix.REF_VALIDATION_SPLITS.index(split)
    return [
        {
            "image_id": split_index * 100 + index // 2,
            "ann_id": 10 + index,
            "ref_id": 20 + index,
            "sent_id": 30 + index,
        }
        for index in range(SMALL_REF_COUNTS[split])
    ]


def _jsonl_bytes(rows) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


SMALL_REF_CONTRACT = {
    "refcoco_val": {
        "rows": SMALL_REF_COUNTS["refcoco_val"],
        "sha256": hashlib.sha256(
            _jsonl_bytes(_ref_manifest_rows("refcoco_val"))
        ).hexdigest(),
    },
    "refcocop_val": {
        "rows": SMALL_REF_COUNTS["refcocop_val"],
        "sha256": hashlib.sha256(
            _jsonl_bytes(_ref_manifest_rows("refcocop_val"))
        ).hexdigest(),
    },
    "refcocog_val": {
        "rows": SMALL_REF_COUNTS["refcocog_val"],
        "sha256": hashlib.sha256(
            _jsonl_bytes(_ref_manifest_rows("refcocog_val"))
        ).hexdigest(),
    },
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_commands(
    runtime: evaluator.Runtime,
    source: evaluator.EvaluationSource,
    output_root: Path,
) -> list[dict]:
    command = [
        str(runtime.python),
        "fixture_evaluate.py",
        "--training_run_id",
        str(source.training_run_id),
        "--output_dir",
        str(output_root / "validation_calibration"),
        "--ref_splits",
        *matrix.REF_VALIDATION_SPLITS,
    ]
    return [
        {
            "phase_id": "validation_calibration",
            "command": command,
            "command_shell": shlex.join(command),
            "console_log": str(
                output_root / "validation_calibration_console.log"
            ),
        }
    ]


def _fixture_replay_matrix_plan_contract(
    launch,
    *,
    source,
    runtime,
    output_root,
    cache,
):
    if (
        runtime.batch_size != 16
        or runtime.num_workers != 4
        or runtime.amp is not True
        or runtime.log_every != 50
    ):
        raise matrix.MatrixValidationError("matrix runtime fixture drifted")
    if launch.get("commands") != _fixture_commands(
        runtime, source, output_root
    ):
        raise matrix.MatrixValidationError("canonical matrix command drifted")
    inputs = launch.get("inputs")
    records = inputs.get("records") if isinstance(inputs, dict) else None
    if not isinstance(records, list):
        raise matrix.MatrixValidationError("fixture inputs are missing")

    def paths_for(role: str) -> list[Path]:
        return [
            Path(record["path"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("roles"), list)
            and role in record["roles"]
        ]

    code = paths_for("evaluation_code_dependency")
    provenance = paths_for("source_provenance_dependency")
    data = paths_for("evaluation_data_input")
    if len(code) != 1 or len(provenance) != 1 or len(data) != 1:
        raise matrix.MatrixValidationError("fixture code/data inputs drifted")
    protocol = launch["protocol"]["screen_calibration"]
    entries = [
        (source.config, "evaluation_config"),
        (source.config, "config_dependency"),
        (source.checkpoint, "evaluation_checkpoint"),
        (code[0], "evaluation_code_dependency"),
        (provenance[0], "source_provenance_dependency"),
        (data[0], "evaluation_data_input"),
        (Path(protocol["source_manifest"]["path"]), "matrix_calibration_source"),
        (Path(protocol["source_audit"]["path"]), "matrix_calibration_audit"),
    ]
    entries.extend((path, "training_data") for path in source.training_data)
    for path, role in (
        (source.sequence_manifest, "training_sequence_manifest"),
        (source.final_phase_manifest, "training_final_phase_manifest"),
        (source.training_postflight, "training_final_phase_postflight"),
        (source.selected_phase_manifest, "training_selected_phase_manifest"),
        (
            source.selected_training_postflight,
            "training_selected_phase_postflight",
        ),
        (source.training_queue_manifest, "training_queue_manifest"),
        (source.training_queue_detached_launch, "training_queue_detached_launch"),
        (source.training_queue_detached_status, "training_queue_detached_status"),
    ):
        if path is not None:
            entries.append((path, role))
    expected = {
        "algorithm": "sha256",
        "records": evaluator._merge_input_records(entries, cache),
    }
    if inputs != expected:
        raise matrix.MatrixValidationError("canonical formal inputs drifted")
    return {
        "python": str(runtime.python),
        "data_root": str(runtime.data_root),
        "device": runtime.device,
        "batch_size": runtime.batch_size,
        "num_workers": runtime.num_workers,
        "amp": runtime.amp,
        "log_every": runtime.log_every,
        "eval_seed": evaluator.EVAL_SEED,
        "max_ref_batches": 0,
        "max_tn_batches": 0,
    }


def _fixture_evaluation_queue_binding(
    queue_dir,
    *,
    spec_path,
    evaluation_queue_id,
    evaluation_plan_sha256,
    experiments,
    seeds,
):
    cache = evaluator.HashCache()
    sealed = matrix._snapshot_files(
        (
            Path(queue_dir) / "queue.json",
            Path(queue_dir) / "predeclared_contract.json",
            Path(queue_dir) / "training_attestation.json",
        ),
        cache,
    )
    return {
        "queue_dir": str(Path(queue_dir).resolve()),
        "queue_id": evaluation_queue_id,
        "plan_sha256": evaluation_plan_sha256,
        "provenance_scope": matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE,
        "predeclared_contract_sha256": "c" * 64,
        "verification_schema": queue_runner_schema,
        "verification_status": "passed",
        "verified_item_count": len(experiments) * len(seeds),
        "sealed_files": sealed,
    }


queue_runner_schema = "pivot.stageb.matrix_validation_queue_verification/v1"


def _source_row(index: int) -> dict:
    return {
        "sample_id": f"calibration:{index}",
        "image_id": 1000 + index // 2,
        "ann_id": 2000 + index,
        "ref_id": 3000 + index,
        "sent_id": 4000 + index,
        "class_id": 7,
        "file_name": f"COCO_train2014_{index:012d}.jpg",
        "split": "train",
        "dataset": "refcocoplus",
        "pair_source": "refcoco+_unc",
        "category_name": "person",
        "class_norm_name": "person",
        "target_bbox_used": [1.0, 2.0, 30.0, 40.0],
        "sent": "person in a red shirt",
        "try_tn": "person in a blue shirt",
        "try_tn_head": "person",
        "try_tn_head_phrase": "person in a red shirt",
        "replace_category": ["color"],
        "replace_from": ["red"],
        "replace_to": ["blue"],
        "replace_span": [[3, 4]],
        "tn_edits": [
            {
                "category": "color",
                "replace_from": "red",
                "replace_to": "blue",
                "replace_span": [3, 4],
            }
        ],
        "table_b_pair_schema": "stage-b-paper-table-b-scope-preserving-pair-v1",
        "table_b_id": "D3",
        "tn_scope": "proposal_covered_verified",
        "proposal_covered_verified": True,
        "traceable_counterfactual_edit": True,
        "visual_verified_negative": True,
        "global_tn_verified": False,
        "proposalset_proxy_verified": False,
        "cached_proposal_coverage_only": True,
        "all_900_gdino_queries_verified": False,
        "global_max_label_is_semantic_extrapolation": True,
    }


class MatrixFixture:
    def __init__(
        self,
        root: Path,
        *,
        duplicate_checkpoint: bool = False,
        duplicate_training_run: bool = False,
    ) -> None:
        self.root = root
        self.data_root = root / "data"
        self.data_root.mkdir(parents=True)
        self.evaluation_queue = root / "evaluation_queue"
        for name in (
            "queue.json",
            "predeclared_contract.json",
            "training_attestation.json",
        ):
            _write_json(
                self.evaluation_queue / name,
                {"schema": f"fixture-{name}"},
            )
        self.source = root / "shared" / "calibration_source.jsonl"
        self.audit = root / "shared" / "calibration_audit.json"
        self.code = root / "shared" / "evaluation_code.py"
        self.data = root / "shared" / "evaluation_data.json"
        _write_jsonl(
            self.source,
            [_source_row(index) for index in range(matrix.CALIBRATION_ROWS)],
        )
        _write_json(self.audit, {"schema": "synthetic-table-b-audit"})
        self.code.parent.mkdir(parents=True, exist_ok=True)
        self.code.write_text("EVALUATION_CODE = 1\n", encoding="utf-8")
        _write_json(self.data, {"surface": "validation-only"})
        self.roots: dict[str, dict[int, Path]] = {"baseline": {}, "candidate": {}}
        baseline_checkpoints: dict[int, Path] = {}
        for experiment in ("baseline", "candidate"):
            for seed in (11, 22):
                training_root = root / "training" / experiment / f"seed{seed}"
                checkpoint = training_root / "checkpoint_iter.pth"
                config = training_root / "config.py"
                training_root.mkdir(parents=True, exist_ok=True)
                if experiment == "candidate" and duplicate_checkpoint and seed == 11:
                    checkpoint = baseline_checkpoints[seed]
                else:
                    checkpoint.write_bytes(f"{experiment}:{seed}".encode("ascii"))
                config.write_text(f"ROW = {experiment!r}\n", encoding="utf-8")
                if experiment == "baseline":
                    baseline_checkpoints[seed] = checkpoint
                evaluation_root = root / "evaluations" / experiment / f"seed{seed}"
                self._build_evaluation(
                    experiment=experiment,
                    seed=seed,
                    evaluation_root=evaluation_root,
                    training_root=training_root,
                    checkpoint=checkpoint,
                    config=config,
                    duplicate_training_run=(
                        experiment == "candidate"
                        and duplicate_training_run
                        and seed == 11
                    ),
                )
                self.roots[experiment][seed] = evaluation_root
        self.spec = root / "matrix_spec.json"
        self.write_spec(reference="baseline")

    def _build_evaluation(
        self,
        *,
        experiment: str,
        seed: int,
        evaluation_root: Path,
        training_root: Path,
        checkpoint: Path,
        config: Path,
        duplicate_training_run: bool,
    ) -> None:
        section = evaluation_root / "validation_calibration"
        records_dir = section / "per_example_records"
        derived = section / "tn_eval_inputs" / "tn_matrix_calibration.jsonl"
        training_sequence = training_root / "sequence_manifest.json"
        training_launch = training_root / "launch_manifest.json"
        training_postflight = training_root / "postflight.json"
        queue_dir = training_root / "queue"
        queue_manifest = queue_dir / "queue.json"
        detached_launch = queue_dir / "detached_launch.json"
        detached_status = queue_dir / "detached_status.json"
        _write_json(training_sequence, {"status": "completed"})
        _write_json(training_launch, {"status": "completed"})
        _write_json(training_postflight, {"status": "passed"})
        _write_json(queue_manifest, {"status": "completed"})
        _write_json(detached_launch, {"status": "launched"})
        _write_json(detached_status, {"status": "completed"})
        binding = build_manifest(
            source_path=self.source,
            audit_path=self.audit,
            derived_path=derived,
            data_root=self.data_root,
        )
        cache = evaluator.HashCache()
        checkpoint_sha = cache.digest(checkpoint)
        run_id = evaluator._checkpoint_run_id(checkpoint)

        ref_summary = []
        for split_index, split in enumerate(matrix.REF_VALIDATION_SPLITS):
            contract = SMALL_REF_CONTRACT[split]
            count = int(contract["rows"])
            manifest_rows = _ref_manifest_rows(split)
            manifest_path = (
                section
                / "refcoco_eval_inputs"
                / REF_SPLIT_MANIFEST_FILES[split]
            )
            _write_jsonl(manifest_path, manifest_rows)
            record_path = records_dir / f"{run_id}__{split}.records.jsonl"
            records = []
            for index, manifest_row in enumerate(manifest_rows):
                baseline_correct = index < max(1, count // 2)
                correct = baseline_correct or (
                    experiment == "candidate" and index == count - 1
                )
                top1_iou = 0.75 if correct else 0.25
                records.append(
                    {
                        "schema": "stageb-eval-record-v1",
                        "task": "ref",
                        "manifest_key": f"ref:{split}",
                        "manifest_sha256": contract["sha256"],
                        "manifest_n": count,
                        "manifest_index": index,
                        "sample_id": (
                            f"ref:{split}:{manifest_row['image_id']}:"
                            f"{manifest_row['ann_id']}:{manifest_row['ref_id']}:"
                            f"{manifest_row['sent_id']}"
                        ),
                        **manifest_row,
                        "split": split,
                        "run_id": run_id,
                        "valid": True,
                        "correct50": correct,
                        "top1_iou": top1_iou,
                    }
                )
            _write_jsonl(record_path, records)
            ref_summary.append(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_name": checkpoint.name,
                    "run_id": run_id,
                    "seed": evaluator.EVAL_SEED + split_index * 100000,
                    "max_batches": 0,
                    "invalid_records": 0,
                    "dataset": split,
                    "manifest_n": count,
                    "manifest_sha256": contract["sha256"],
                    "num_expressions": count,
                    "records_jsonl": str(record_path),
                    "acc50": float(np.mean([row["correct50"] for row in records])),
                    "mean_iou": float(np.mean([row["top1_iou"] for row in records])),
                }
            )

        derived_rows = [
            json.loads(line)
            for line in derived.read_text(encoding="utf-8").splitlines()
        ]
        positive = []
        negative = []
        tn_records = []
        for index, source in enumerate(derived_rows):
            pos_score = 0.55 + (index % 7) * 0.01
            neg_score = (
                0.52 + (index % 5) * 0.01
                if experiment == "baseline"
                else 0.36 + (index % 5) * 0.01
            )
            positive.append(pos_score)
            negative.append(neg_score)
            tn_records.append(
                {
                    "schema": "stageb-eval-record-v1",
                    "task": "tn",
                    "manifest_key": "tn_global",
                    "manifest_sha256": binding.derived_manifest["sha256"],
                    "manifest_n": matrix.CALIBRATION_ROWS,
                    "manifest_index": index,
                    "sample_id": source["sample_id"],
                    "image_id": source["image_id"],
                    "ann_id": source["ann_id"],
                    "ref_id": source["ref_id"],
                    "sent_id": source["sent_id"],
                    "split": matrix.CALIBRATION_SPLIT,
                    "run_id": run_id,
                    "valid": True,
                    "pos_score": pos_score,
                    "neg_score": neg_score,
                }
            )
        tn_record_path = records_dir / f"{run_id}__tn_global.records.jsonl"
        _write_jsonl(tn_record_path, tn_records)
        fpr = exact_fpr95(positive, negative)
        tn_summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_name": checkpoint.name,
            "run_id": run_id,
            "seed": evaluator.EVAL_SEED,
            "max_batches": 0,
            "invalid_records": 0,
            "manifest_n": matrix.CALIBRATION_ROWS,
            "num_pairs": matrix.CALIBRATION_ROWS,
            "manifest_sha256": binding.derived_manifest["sha256"],
            "records_jsonl": str(tn_record_path),
            "fpr95tpr": float(fpr["fpr"]),
            "threshold_at_95tpr": float(fpr["threshold"]),
            "pair_win_rate": float(
                np.mean(np.asarray(positive) > np.asarray(negative))
            ),
            **summary_fields(binding),
        }
        summary_path = section / "summary.json"
        _write_json(summary_path, {"refcoco": ref_summary, "tn": [tn_summary]})

        source_record = evaluator._file_record(
            self.source, cache, roles=("matrix_calibration_source",)
        )
        source_record["rows"] = matrix.CALIBRATION_ROWS
        audit_record = evaluator._file_record(
            self.audit, cache, roles=("matrix_calibration_audit",)
        )
        calibration_contract = {
            "source_manifest": source_record,
            "source_audit": audit_record,
            "unique_images": matrix.CALIBRATION_ROWS // 2,
            "scope": "proposal_covered_verified",
            "single_edit_provenance": True,
            "strict_union_image_overlap": 0,
            "train_calibration_image_overlap": 0,
        }
        evaluation_id = f"{experiment}_seed{seed}"
        training_run_id = (
            "baseline:11" if duplicate_training_run else f"{experiment}:{seed}"
        )
        source_payload = {
            "kind": "pivot_token_ablation_training_run",
            "evaluation_id": evaluation_id,
            "config": str(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "training_run_id": training_run_id,
            "training_seed": seed,
            "training_run_root": str(training_root),
            "sequence_manifest": str(training_sequence),
            "training_phase": "final",
            "diagnostic_only": False,
            "final_phase_id": "joint",
            "final_phase_manifest": str(training_launch),
            "training_postflight": str(training_postflight),
            "selected_phase_id": "joint",
            "selected_phase_manifest": str(training_launch),
            "selected_training_postflight": str(training_postflight),
            "training_queue_manifest": str(queue_manifest),
            "training_queue_detached_launch": str(detached_launch),
            "training_queue_detached_status": str(detached_status),
            "training_queue_id": f"queue-{experiment}-{seed}",
            "training_queue_plan_sha256": "a" * 64,
            "artifact_repository_root": str(
                evaluator.ARTIFACT_REPOSITORY_ROOT
            ),
            "training_data": [str(self.data)],
        }
        source_object = matrix._evaluation_source_from_launch(source_payload)
        runtime_object = evaluator.Runtime(
            python=Path(sys.executable).resolve(),
            data_root=self.data_root.resolve(),
            device="cuda:0",
            batch_size=16,
            num_workers=4,
            amp=True,
            log_every=50,
        )
        input_entries = [
            (checkpoint, "evaluation_checkpoint"),
            (config, "evaluation_config"),
            (config, "config_dependency"),
            (self.code, "evaluation_code_dependency"),
            (self.code, "source_provenance_dependency"),
            (self.data, "evaluation_data_input"),
            (self.source, "matrix_calibration_source"),
            (self.audit, "matrix_calibration_audit"),
            (self.data, "training_data"),
            (training_sequence, "training_sequence_manifest"),
            (training_launch, "training_final_phase_manifest"),
            (training_postflight, "training_final_phase_postflight"),
            (training_launch, "training_selected_phase_manifest"),
            (training_postflight, "training_selected_phase_postflight"),
            (queue_manifest, "training_queue_manifest"),
            (detached_launch, "training_queue_detached_launch"),
            (detached_status, "training_queue_detached_status"),
        ]
        input_records = evaluator._merge_input_records(input_entries, cache)
        launch = {
            "schema": evaluator.SCHEMA,
            "status": "completed",
            "repository_root": str(evaluator.REPO_ROOT),
            "artifact_repository_root": str(
                evaluator.ARTIFACT_REPOSITORY_ROOT
            ),
            "artifact_outputs_root": str(evaluator.ARTIFACT_OUTPUTS_ROOT),
            "evaluation_id": evaluation_id,
            "output_dir": str(evaluation_root.resolve()),
            "source": source_payload,
            "runtime": {
                "python": str(runtime_object.python),
                "data_root": str(runtime_object.data_root),
                "device": runtime_object.device,
                "batch_size": runtime_object.batch_size,
                "num_workers": runtime_object.num_workers,
                "amp": runtime_object.amp,
                "log_every": runtime_object.log_every,
                "eval_seed": evaluator.EVAL_SEED,
                "max_ref_batches": 0,
                "max_tn_batches": 0,
            },
            "protocol": {
                "profile": evaluator.MATRIX_PROFILE,
                "ref_splits": list(matrix.REF_VALIDATION_SPLITS),
                "strict_manifests": {},
                "screen_calibration": calibration_contract,
                "processes": ["validation_calibration"],
                "strict1607_skip_ref": False,
                "per_example_records": True,
                "release_policy": (
                    "ablation_matrix_validation_only_no_ref_test_or_strict_access"
                ),
            },
            "commands": _fixture_commands(
                runtime_object, source_object, evaluation_root.resolve()
            ),
            "inputs": {"algorithm": "sha256", "records": input_records},
            "completed_phases": [
                {
                    "phase_id": "validation_calibration",
                    "status": "completed",
                    "returncode": 0,
                }
            ],
        }
        input_rehash = evaluator._rehash_inputs(launch)
        input_rehash_path = evaluation_root / "input_rehash.json"
        _write_json(input_rehash_path, input_rehash)
        launch["input_rehash_artifact"] = evaluator._file_record(
            input_rehash_path, cache, roles=("input_rehash",)
        )
        postflight = evaluator._postflight_screen(launch, input_rehash)
        postflight_path = evaluation_root / "postflight.json"
        _write_json(postflight_path, postflight)
        launch["postflight"] = postflight
        launch["postflight_artifact"] = evaluator._file_record(
            postflight_path, cache, roles=("postflight",)
        )
        _write_json(evaluation_root / "launch_manifest.json", launch)

    def write_spec(self, *, reference) -> None:
        payload = {
            "schema": matrix.INPUT_SCHEMA,
            "expected_train_seeds": [11, 22],
            "evaluation_queue_dir": str(self.evaluation_queue),
            "evaluation_queue_id": "fixture-evaluation-queue",
            "evaluation_plan_sha256": "e" * 64,
            "evaluation_provenance_scope": (
                matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE
            ),
            "experiments": [
                {
                    "id": experiment,
                    "label": experiment,
                    "evaluation_roots": {
                        str(seed): str(self.roots[experiment][seed])
                        for seed in (11, 22)
                    },
                }
                for experiment in ("baseline", "candidate")
            ],
        }
        if reference is not None:
            payload["reference_experiment"] = reference
        _write_json(self.spec, payload)

    def refresh_root(self, root: Path) -> None:
        launch_path = root / "launch_manifest.json"
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        input_rehash = evaluator._rehash_inputs(launch)
        input_rehash_path = root / "input_rehash.json"
        _write_json(input_rehash_path, input_rehash)
        cache = evaluator.HashCache()
        launch["input_rehash_artifact"] = evaluator._file_record(
            input_rehash_path, cache, roles=("input_rehash",)
        )
        postflight = evaluator._postflight_screen(launch, input_rehash)
        postflight_path = root / "postflight.json"
        _write_json(postflight_path, postflight)
        launch["postflight"] = postflight
        launch["postflight_artifact"] = evaluator._file_record(
            postflight_path, cache, roles=("postflight",)
        )
        _write_json(launch_path, launch)


class StageBMatrixValidationTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(
            patch.object(matrix, "FORMAL_TRAIN_SEEDS", (11, 22))
        )
        self.stack.enter_context(
            patch.object(
                matrix,
                "FORMAL_EXPERIMENT_IDS",
                ("baseline", "candidate"),
            )
        )
        self.stack.enter_context(
            patch.object(matrix, "FORMAL_REFERENCE_EXPERIMENT", "baseline")
        )
        self.stack.enter_context(
            patch.object(
                matrix,
                "FORMAL_EVALUATION_QUEUE_ID",
                "fixture-evaluation-queue",
            )
        )
        self.stack.enter_context(
            patch.object(matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64)
        )
        self.stack.enter_context(
            patch.object(matrix, "FORMAL_BOOTSTRAP_ITERATIONS", 2)
        )
        self.stack.enter_context(
            patch.object(evaluator, "_revalidate_matrix_source", return_value=None)
        )
        self.stack.enter_context(
            patch.object(
                matrix,
                "_validate_locked_training_identity",
                return_value=None,
            )
        )
        self.stack.enter_context(
            patch.object(
                matrix,
                "_replay_matrix_plan_contract",
                side_effect=_fixture_replay_matrix_plan_contract,
            )
        )
        self.stack.enter_context(
            patch.object(
                matrix,
                "_validate_evaluation_queue_binding",
                side_effect=_fixture_evaluation_queue_binding,
            )
        )
        self.stack.enter_context(
            patch.dict(
                evaluator.REF_SPLIT_CONTRACT,
                SMALL_REF_CONTRACT,
                clear=False,
            )
        )
        self.stack.enter_context(
            patch.dict(
                paper_aggregate.REF_SPLIT_CONTRACT,
                SMALL_REF_CONTRACT,
                clear=False,
            )
        )
        self.stack.enter_context(
            patch.dict(
                matrix.REF_VALIDATION_CONTRACT,
                SMALL_REF_CONTRACT,
                clear=True,
            )
        )

    def tearDown(self):
        self.stack.close()

    def test_replays_metrics_seed_first_aggregation_and_deterministic_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            first = matrix.aggregate_spec(
                fixture.spec, bootstrap_iterations=2
            )
            second = matrix.aggregate_spec(
                fixture.spec, bootstrap_iterations=2
            )
            self.assertEqual(first["status"], "validated_matrix_validation_only")
            self.assertFalse(first["formal_test_or_strict_result"])
            self.assertFalse(first["protocol"]["ref_test_access"])
            self.assertEqual(
                first["inputs"]["aggregator_source"]["path"],
                str(Path(matrix.__file__).resolve()),
            )
            self.assertEqual(
                first["inputs"]["aggregator_source"]["sha256"],
                evaluator.HashCache().digest(Path(matrix.__file__)),
            )
            self.assertEqual(
                len(first["inputs"]["aggregation_source_closure"]), 13
            )
            closure_paths = {
                Path(record["path"]).resolve().relative_to(matrix.REPO_ROOT).as_posix()
                for record in first["inputs"]["aggregation_source_closure"]
            }
            self.assertEqual(
                closure_paths,
                set(matrix.AGGREGATION_EXPECTED_SOURCE_PATHS),
            )
            self.assertIn(
                "tools/stageb_profile_dependency_audit.py", closure_paths
            )
            self.assertFalse(
                {
                    "tools/stageb_headline_release_contract.py",
                    "tools/build_stageb_paper_ablation_completion_receipt.py",
                    "tools/build_stageb_b58_exposure_receipt.py",
                    "tools/aggregate_stageb_table_d_diagnostics.py",
                }
                & closure_paths
            )
            candidate = first["experiments"]["candidate"]
            self.assertEqual(set(candidate["per_seed"]), {"11", "22"})
            self.assertEqual(
                candidate["aggregate"]["ref_validation"]["val_macro"]["acc50"][
                    "ddof"
                ],
                1,
            )
            self.assertEqual(
                candidate["aggregate"]["calibration"]["fpr95"]["ddof"], 1
            )
            comparison = first["comparisons_to_reference"]["candidate"]
            self.assertTrue(comparison["record_identities_aligned"])
            self.assertTrue(
                comparison["calibration"]["fpr95"]["bootstrap"]["seed_first"]
            )
            self.assertEqual(
                comparison,
                second["comparisons_to_reference"]["candidate"],
            )
            with self.assertRaisesRegex(
                matrix.MatrixValidationError, "evaluated evidence root"
            ):
                matrix._assert_output_path_isolated(
                    fixture.roots["candidate"][11] / "report.json",
                    first,
                )

    def test_optional_reference_omits_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            fixture.write_spec(reference=None)
            with patch.object(matrix, "FORMAL_REFERENCE_EXPERIMENT", None):
                report = matrix.aggregate_spec(
                    fixture.spec, bootstrap_iterations=2
                )
            self.assertIsNone(report["reference_experiment"])
            self.assertEqual(report["comparisons_to_reference"], {})

    def test_non_matrix_profile_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            root = fixture.roots["candidate"][11]
            launch_path = root / "launch_manifest.json"
            launch = json.loads(launch_path.read_text())
            launch["protocol"]["profile"] = evaluator.SCREEN_PROFILE
            _write_json(launch_path, launch)
            with self.assertRaisesRegex(matrix.MatrixValidationError, "profile"):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_missing_or_extra_seed_fails_closed(self):
        for mutation in ("missing", "extra"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                spec = json.loads(fixture.spec.read_text())
                roots = spec["experiments"][1]["evaluation_roots"]
                if mutation == "missing":
                    roots.pop("22")
                else:
                    roots["73"] = roots["22"]
                _write_json(fixture.spec, spec)
                with self.assertRaisesRegex(
                    matrix.MatrixValidationError, "seed set mismatch"
                ):
                    matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_formal_seed_contract_rejects_two_or_extra_seeds(self):
        for declared in ([17, 42], [17, 42, 73, 99]):
            with self.subTest(
                declared=declared
            ), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                spec = json.loads(fixture.spec.read_text())
                spec["expected_train_seeds"] = declared
                _write_json(fixture.spec, spec)
                with patch.object(
                    matrix, "FORMAL_TRAIN_SEEDS", (17, 42, 73)
                ), self.assertRaisesRegex(
                    matrix.MatrixValidationError, "must be exactly"
                ):
                    matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_formal_table_c_requires_all_rows_and_l0_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            with patch.object(
                matrix,
                "FORMAL_EXPERIMENT_IDS",
                tuple(f"L{index}" for index in range(11)),
            ), patch.object(
                matrix, "FORMAL_REFERENCE_EXPERIMENT", "L0"
            ), self.assertRaisesRegex(
                matrix.MatrixValidationError, "canonical Table-C rows"
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

        for reference in (None, "candidate"):
            with self.subTest(
                reference=reference
            ), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                fixture.write_spec(reference=reference)
                with self.assertRaisesRegex(
                    matrix.MatrixValidationError,
                    "reference_experiment must be exactly",
                ):
                    matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_declared_experiment_cannot_mislabel_training_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            spec = json.loads(fixture.spec.read_text())
            spec["experiments"][1]["id"] = "mislabel"
            spec["experiments"][1]["label"] = "mislabel"
            spec["reference_experiment"] = "mislabel"
            _write_json(fixture.spec, spec)
            with patch.object(
                matrix,
                "FORMAL_EXPERIMENT_IDS",
                ("baseline", "mislabel"),
            ), patch.object(
                matrix, "FORMAL_REFERENCE_EXPERIMENT", "mislabel"
            ), self.assertRaisesRegex(
                matrix.MatrixValidationError, "training run_id"
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_forged_formal_source_revalidation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            with patch.object(
                evaluator,
                "_revalidate_matrix_source",
                side_effect=evaluator.PaperEvaluationError("forged source"),
            ), self.assertRaisesRegex(
                matrix.MatrixValidationError, "formal training source revalidation"
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_missing_training_queue_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            root = fixture.roots["candidate"][11]
            launch_path = root / "launch_manifest.json"
            launch = json.loads(launch_path.read_text())
            launch["source"]["training_queue_manifest"] = None
            _write_json(launch_path, launch)
            with self.assertRaisesRegex(
                matrix.MatrixValidationError, "incomplete formal training/queue"
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_changed_training_postflight_fails_input_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            root = fixture.roots["candidate"][11]
            launch = json.loads((root / "launch_manifest.json").read_text())
            training_postflight = Path(
                launch["source"]["training_postflight"]
            )
            _write_json(training_postflight, {"status": "drifted"})
            with self.assertRaisesRegex(
                matrix.MatrixValidationError,
                "canonical formal inputs|rehash replay failed",
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_duplicate_root_run_and_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            spec = json.loads(fixture.spec.read_text())
            spec["experiments"][1]["evaluation_roots"]["11"] = spec[
                "experiments"
            ][0]["evaluation_roots"]["11"]
            _write_json(fixture.spec, spec)
            with self.assertRaisesRegex(matrix.MatrixValidationError, "duplicated"):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(
                Path(temporary), duplicate_training_run=True
            )
            with self.assertRaisesRegex(
                matrix.MatrixValidationError, "training run_id"
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary), duplicate_checkpoint=True)
            with self.assertRaisesRegex(matrix.MatrixValidationError, "checkpoint"):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_ref_test_or_strict_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            forbidden = (
                fixture.roots["candidate"][11]
                / "validation_calibration/refcoco_testA.records.jsonl"
            )
            _write_jsonl(forbidden, [{"not": "allowed"}])
            with self.assertRaisesRegex(
                matrix.MatrixValidationError, "test/strict artifact"
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_ref_test_or_strict_command_option_fails_closed(self):
        for token in (
            "--ref_splits=refcoco_testA",
            "--strict_manifest=/tmp/opaque.jsonl",
        ):
            with self.subTest(
                token=token
            ), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                root = fixture.roots["candidate"][11]
                launch_path = root / "launch_manifest.json"
                launch = json.loads(launch_path.read_text())
                launch["commands"][0]["command"].append(token)
                _write_json(launch_path, launch)
                with self.assertRaisesRegex(
                    matrix.MatrixValidationError, "test/strict surface"
                ):
                    matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_output_artifact_is_fresh_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            matrix._write_atomic(output, "first\n")
            with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                matrix._write_atomic(output, "second\n")
            self.assertEqual(output.read_text(), "first\n")

    def test_failed_input_rehash_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            fixture.code.write_text("EVALUATION_CODE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(
                matrix.MatrixValidationError,
                "canonical formal inputs|rehash replay failed|input drift",
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_wrong_ref_split_count_or_manifest_fails_closed(self):
        for mutation in ("split", "count", "manifest"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                root = fixture.roots["candidate"][11]
                summary_path = root / "validation_calibration/summary.json"
                summary = json.loads(summary_path.read_text())
                if mutation == "split":
                    summary["refcoco"][0]["dataset"] = "refcoco_testA"
                elif mutation == "count":
                    summary["refcoco"][0]["manifest_n"] += 1
                else:
                    summary["refcoco"][0]["manifest_sha256"] = "f" * 64
                _write_json(summary_path, summary)
                with self.assertRaisesRegex(
                    matrix.MatrixValidationError,
                    "postflight replay failed|three validation splits",
                ):
                    matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_non_1570_calibration_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            root = fixture.roots["candidate"][11]
            launch_path = root / "launch_manifest.json"
            launch = json.loads(launch_path.read_text())
            launch["protocol"]["screen_calibration"]["source_manifest"]["rows"] = 1569
            _write_json(launch_path, launch)
            with self.assertRaisesRegex(
                matrix.MatrixValidationError,
                "postflight replay failed|1,570|calibration",
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_ref_mean_iou_and_calibration_operating_metrics_are_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            root = fixture.roots["candidate"][11]
            summary_path = root / "validation_calibration/summary.json"
            pristine = json.loads(summary_path.read_text())
            mutations = (
                ("mean_iou", ("refcoco", 0, "mean_iou")),
                ("positive_q05", ("tn", 0, "threshold_at_95tpr")),
                ("pair_win", ("tn", 0, "pair_win_rate")),
            )
            for label, (section, index, field) in mutations:
                with self.subTest(metric=label):
                    payload = copy.deepcopy(pristine)
                    payload[section][index][field] = 0.123456789
                    _write_json(summary_path, payload)
                    fixture.refresh_root(root)
                    with self.assertRaisesRegex(
                        matrix.MatrixValidationError,
                        "mismatch",
                    ):
                        matrix.aggregate_spec(
                            fixture.spec, bootstrap_iterations=2
                        )
                    _write_json(summary_path, pristine)
                    fixture.refresh_root(root)

    def test_binding_or_record_drift_fails_closed(self):
        for mutation in ("binding", "records"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                root = fixture.roots["candidate"][11]
                postflight = json.loads((root / "postflight.json").read_text())
                calibration = postflight["artifacts"]["matrix_calibration"]
                if mutation == "binding":
                    path = Path(calibration["derived_manifest"]["path"])
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["sample_id"] = "drifted-derived"
                else:
                    path = Path(calibration["records"]["path"])
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["sample_id"] = "drifted-record"
                _write_jsonl(path, rows)
                with self.assertRaisesRegex(
                    matrix.MatrixValidationError,
                    "postflight replay failed|binding|record",
                ):
                    matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_inconsistent_record_surface_or_code_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            root = fixture.roots["candidate"][11]
            postflight = json.loads((root / "postflight.json").read_text())
            record_path = Path(
                postflight["artifacts"]["ref_validation"]["refcoco_val"][
                    "records"
                ]["path"]
            )
            rows = [json.loads(line) for line in record_path.read_text().splitlines()]
            rows[0]["sample_id"] = "same-manifest-different-identity"
            _write_jsonl(record_path, rows)
            fixture.refresh_root(root)
            with self.assertRaisesRegex(
                matrix.MatrixValidationError,
                "locked evaluation manifest|record identities",
            ):
                matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)
        for role in ("evaluation_code_dependency", "evaluation_data_input"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                root = fixture.roots["candidate"][11]
                launch_path = root / "launch_manifest.json"
                launch = json.loads(launch_path.read_text())
                extra = Path(temporary) / f"extra-{role}.txt"
                extra.write_text("EXTRA = 1\n", encoding="utf-8")
                launch["inputs"]["records"].append(
                    evaluator._file_record(
                        extra,
                        evaluator.HashCache(),
                        roles=(role,),
                    )
                )
                _write_json(launch_path, launch)
                fixture.refresh_root(root)
                with self.assertRaisesRegex(
                    matrix.MatrixValidationError,
                    "fixture code/data inputs|canonical formal inputs|"
                    "inconsistent runtime/code/data/surface",
                ):
                    matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_label_runtime_command_and_input_contracts_fail_closed(self):
        mutations = (
            "label",
            "batch",
            "amp",
            "command",
            "opaque_input",
            "training_input",
        )
        for mutation in mutations:
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                if mutation == "label":
                    spec = json.loads(fixture.spec.read_text())
                    spec["experiments"][0]["label"] = "L4"
                    _write_json(fixture.spec, spec)
                    expected = "label must equal"
                else:
                    root = fixture.roots["candidate"][11]
                    launch_path = root / "launch_manifest.json"
                    launch = json.loads(launch_path.read_text())
                    if mutation == "batch":
                        launch["runtime"]["batch_size"] = 8
                        expected = "runtime"
                    elif mutation == "amp":
                        launch["runtime"]["amp"] = False
                        expected = "runtime"
                    elif mutation == "command":
                        launch["commands"][0]["command"][1] = "opaque_eval.py"
                        expected = "canonical matrix command"
                    elif mutation == "opaque_input":
                        opaque = Path(temporary) / "opaque.txt"
                        opaque.write_text("opaque\n", encoding="utf-8")
                        launch["inputs"]["records"].append(
                            evaluator._file_record(
                                opaque,
                                evaluator.HashCache(),
                                roles=("opaque_heldout_input",),
                            )
                        )
                        expected = "canonical formal inputs"
                    else:
                        for record in launch["inputs"]["records"]:
                            if "training_data" in record["roles"]:
                                record["roles"].remove("training_data")
                                break
                        expected = "canonical formal inputs"
                    _write_json(launch_path, launch)
                with self.assertRaisesRegex(
                    matrix.MatrixValidationError, expected
                ):
                    matrix.aggregate_spec(fixture.spec, bootstrap_iterations=2)

    def test_formal_bootstrap_overrides_fail_closed(self):
        cases = (
            {"bootstrap_iterations": 1},
            {"bootstrap_iterations": 5000, "confidence": 0.5},
            {"bootstrap_iterations": 5000, "bootstrap_seed": 7},
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            for arguments in cases:
                with self.subTest(arguments=arguments), patch.object(
                    matrix, "FORMAL_BOOTSTRAP_ITERATIONS", 5000
                ), self.assertRaisesRegex(
                    matrix.MatrixValidationError, "formal bootstrap protocol"
                ):
                    matrix.aggregate_spec(fixture.spec, **arguments)

    def test_locked_training_queue_and_root_identity_fail_closed(self):
        source = evaluator.EvaluationSource(
            kind="pivot_token_ablation_training_run",
            evaluation_id="L0_seed17",
            config=Path(__file__),
            checkpoint=Path(__file__),
            checkpoint_sha256="a" * 64,
            training_run_id="L0:17",
            training_seed=17,
            training_run_root=Path("/tmp/wrong-root"),
            training_queue_id="wrong-queue",
            training_queue_plan_sha256="b" * 64,
        )
        with self.assertRaisesRegex(
            matrix.MatrixValidationError, "locked Table-C run/queue identity"
        ):
            PRODUCTION_LOCKED_TRAINING_VALIDATOR(
                source, experiment_id="L0", seed=17
            )

    def test_wrong_evaluation_queue_or_predeclared_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical_queue"
            wrong = root / "wrong_queue"
            canonical.mkdir()
            wrong.mkdir()
            with patch.object(
                matrix, "FORMAL_EVALUATION_QUEUE_DIR", canonical
            ), patch.object(
                matrix, "DEFAULT_INPUT_SPEC", root / "spec.json"
            ), patch.object(
                matrix,
                "FORMAL_EVALUATION_QUEUE_ID",
                "fixture-evaluation-queue",
            ), patch.object(
                matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64
            ), self.assertRaisesRegex(
                matrix.MatrixValidationError, "not the canonical predeclared"
            ):
                PRODUCTION_EVALUATION_QUEUE_VALIDATOR(
                    wrong,
                    spec_path=root / "spec.json",
                    evaluation_queue_id="fixture-evaluation-queue",
                    evaluation_plan_sha256="e" * 64,
                    experiments=[],
                    seeds=[],
                )

            for name in (
                "queue.json",
                "predeclared_contract.json",
                "training_attestation.json",
            ):
                _write_json(canonical / name, {"fixture": name})
            _write_json(root / "spec.json", {"fixture": "canonical-spec"})
            planned = []
            experiments = []
            for experiment_id in ("baseline", "candidate"):
                roots = {}
                for seed in (11, 22):
                    evaluation_root = root / experiment_id / f"seed{seed}"
                    evaluation_root.mkdir(parents=True)
                    roots[seed] = evaluation_root.resolve()
                experiments.append({"id": experiment_id, "roots": roots})
            for seed in (11, 22):
                for experiment in experiments:
                    run_id = f"{experiment['id']}:{seed}"
                    planned.append(
                        {
                            "run_id": run_id,
                            "evaluation_root": str(
                                experiment["roots"][seed]
                            ),
                        }
                    )
            queue = {
                "status": "completed",
                "plan_sha256": "e" * 64,
                "predeclared_contract_sha256": "c" * 64,
                "plan": {
                    "queue_id": "fixture-evaluation-queue",
                    "profile": matrix.PROFILE,
                    "provenance_scope": (
                        matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE
                    ),
                    "items": planned,
                },
            }
            verification = {
                "schema": queue_runner_schema,
                "status": "passed",
                "errors": [],
                "queue_id": "fixture-evaluation-queue",
                "plan_sha256": "e" * 64,
                "provenance_scope": (
                    matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE
                ),
                "verified_items": [{}] * 4,
            }
            experiments[0]["roots"][11] = experiments[0]["roots"][22]
            with patch.object(
                matrix, "FORMAL_EVALUATION_QUEUE_DIR", canonical
            ), patch.object(
                matrix, "DEFAULT_INPUT_SPEC", root / "spec.json"
            ), patch.object(
                matrix,
                "FORMAL_EVALUATION_QUEUE_ID",
                "fixture-evaluation-queue",
            ), patch.object(
                matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64
            ), patch.object(
                matrix, "FORMAL_EXPERIMENT_IDS", ("baseline", "candidate")
            ), patch.object(
                matrix.evaluation_queue, "load_queue", return_value=queue
            ), patch.object(
                matrix.evaluation_queue,
                "verify_queue",
                return_value=verification,
            ), self.assertRaisesRegex(
                matrix.MatrixValidationError, "spec roots differ"
            ):
                PRODUCTION_EVALUATION_QUEUE_VALIDATOR(
                    canonical,
                    spec_path=root / "spec.json",
                    evaluation_queue_id="fixture-evaluation-queue",
                    evaluation_plan_sha256="e" * 64,
                    experiments=experiments,
                    seeds=(11, 22),
                )

    def test_aggregation_requires_the_queue_bound_spec_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            queue_dir.mkdir()
            for name in (
                "queue.json",
                "predeclared_contract.json",
                "training_attestation.json",
            ):
                _write_json(queue_dir / name, {"fixture": name})
            spec_path = root / "table_c_input.json"
            _write_json(spec_path, {"fixture": "predeclared-spec"})
            experiments = []
            items = []
            for experiment_id in ("baseline", "candidate"):
                roots = {}
                for seed in (11, 22):
                    evaluation_root = root / experiment_id / f"seed{seed}"
                    evaluation_root.mkdir(parents=True)
                    roots[seed] = evaluation_root.resolve()
                experiments.append({"id": experiment_id, "roots": roots})
            for seed in (11, 22):
                for experiment in experiments:
                    items.append(
                        {
                            "run_id": f"{experiment['id']}:{seed}",
                            "evaluation_root": str(experiment["roots"][seed]),
                        }
                    )
            cache = evaluator.HashCache()
            full_record = matrix._compact_file_record(spec_path, cache)
            content_record = {
                key: full_record[key]
                for key in ("path", "sha256", "size_bytes")
            }
            queue = {
                "status": "completed",
                "plan_sha256": "e" * 64,
                "predeclared_contract_sha256": "c" * 64,
                "aggregation_input_spec": {
                    **content_record,
                    "sha256": "0" * 64,
                },
                "plan": {
                    "queue_id": "fixture-evaluation-queue",
                    "profile": matrix.PROFILE,
                    "provenance_scope": (
                        matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE
                    ),
                    "aggregation_input_spec": (
                        matrix.evaluation_queue._aggregation_input_spec_binding(
                            spec_path
                        )
                    ),
                    "items": items,
                },
            }
            verification = {
                "schema": queue_runner_schema,
                "status": "passed",
                "errors": [],
                "queue_id": "fixture-evaluation-queue",
                "plan_sha256": "e" * 64,
                "provenance_scope": (
                    matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE
                ),
                "verified_items": [{}] * 4,
            }
            with patch.object(
                matrix, "FORMAL_EVALUATION_QUEUE_DIR", queue_dir
            ), patch.object(
                matrix, "DEFAULT_INPUT_SPEC", spec_path
            ), patch.object(
                matrix,
                "FORMAL_EVALUATION_QUEUE_ID",
                "fixture-evaluation-queue",
            ), patch.object(
                matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64
            ), patch.object(
                matrix.evaluation_queue, "load_queue", return_value=queue
            ), patch.object(
                matrix.evaluation_queue,
                "verify_queue",
                return_value=verification,
            ), self.assertRaisesRegex(
                matrix.MatrixValidationError, "queue-bound predeclared input"
            ):
                PRODUCTION_EVALUATION_QUEUE_VALIDATOR(
                    queue_dir,
                    spec_path=spec_path,
                    evaluation_queue_id="fixture-evaluation-queue",
                    evaluation_plan_sha256="e" * 64,
                    experiments=experiments,
                    seeds=(11, 22),
                )

            queue["aggregation_input_spec"] = content_record
            with patch.object(
                matrix, "FORMAL_EVALUATION_QUEUE_DIR", queue_dir
            ), patch.object(
                matrix, "DEFAULT_INPUT_SPEC", spec_path
            ), patch.object(
                matrix,
                "FORMAL_EVALUATION_QUEUE_ID",
                "fixture-evaluation-queue",
            ), patch.object(
                matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64
            ), patch.object(
                matrix.evaluation_queue, "load_queue", return_value=queue
            ), patch.object(
                matrix.evaluation_queue,
                "verify_queue",
                return_value=verification,
            ):
                binding = PRODUCTION_EVALUATION_QUEUE_VALIDATOR(
                    queue_dir,
                    spec_path=spec_path,
                    evaluation_queue_id="fixture-evaluation-queue",
                    evaluation_plan_sha256="e" * 64,
                    experiments=experiments,
                    seeds=(11, 22),
                )
            self.assertEqual(binding["queue_id"], "fixture-evaluation-queue")

            for label, target in (
                ("queue_plan", queue["plan"]),
                ("verification", verification),
            ):
                original = target["provenance_scope"]
                target["provenance_scope"] = (
                    matrix.evaluation_queue.TEST_ONLY_PROVENANCE_SCOPE
                )
                try:
                    with self.subTest(test_only_scope=label), patch.object(
                        matrix, "FORMAL_EVALUATION_QUEUE_DIR", queue_dir
                    ), patch.object(
                        matrix, "DEFAULT_INPUT_SPEC", spec_path
                    ), patch.object(
                        matrix,
                        "FORMAL_EVALUATION_QUEUE_ID",
                        "fixture-evaluation-queue",
                    ), patch.object(
                        matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64
                    ), patch.object(
                        matrix.evaluation_queue, "load_queue", return_value=queue
                    ), patch.object(
                        matrix.evaluation_queue,
                        "verify_queue",
                        return_value=verification,
                    ), self.assertRaisesRegex(
                        matrix.MatrixValidationError,
                        "not completed/verified exactly",
                    ):
                        PRODUCTION_EVALUATION_QUEUE_VALIDATOR(
                            queue_dir,
                            spec_path=spec_path,
                            evaluation_queue_id="fixture-evaluation-queue",
                            evaluation_plan_sha256="e" * 64,
                            experiments=experiments,
                            seeds=(11, 22),
                        )
                finally:
                    target["provenance_scope"] = original

            identity_mutations = (
                ("queue_id", queue["plan"], "queue_id", "replacement-queue"),
                ("queue_plan", queue, "plan_sha256", "f" * 64),
                (
                    "verification_id",
                    verification,
                    "queue_id",
                    "replacement-queue",
                ),
                (
                    "verification_plan",
                    verification,
                    "plan_sha256",
                    "f" * 64,
                ),
            )
            for label, target, field, replacement in identity_mutations:
                original = target[field]
                target[field] = replacement
                try:
                    with self.subTest(current_identity=label), patch.object(
                        matrix, "FORMAL_EVALUATION_QUEUE_DIR", queue_dir
                    ), patch.object(
                        matrix, "DEFAULT_INPUT_SPEC", spec_path
                    ), patch.object(
                        matrix,
                        "FORMAL_EVALUATION_QUEUE_ID",
                        "fixture-evaluation-queue",
                    ), patch.object(
                        matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64
                    ), patch.object(
                        matrix.evaluation_queue, "load_queue", return_value=queue
                    ), patch.object(
                        matrix.evaluation_queue,
                        "verify_queue",
                        return_value=verification,
                    ), self.assertRaisesRegex(
                        matrix.MatrixValidationError,
                        "not completed/verified exactly",
                    ):
                        PRODUCTION_EVALUATION_QUEUE_VALIDATOR(
                            queue_dir,
                            spec_path=spec_path,
                            evaluation_queue_id="fixture-evaluation-queue",
                            evaluation_plan_sha256="e" * 64,
                            experiments=experiments,
                            seeds=(11, 22),
                        )
                finally:
                    target[field] = original

            alternate_spec = root / "alternate_table_c_input.json"
            alternate_spec.write_bytes(spec_path.read_bytes())
            with patch.object(
                matrix, "FORMAL_EVALUATION_QUEUE_DIR", queue_dir
            ), patch.object(
                matrix, "DEFAULT_INPUT_SPEC", spec_path
            ), patch.object(
                matrix,
                "FORMAL_EVALUATION_QUEUE_ID",
                "fixture-evaluation-queue",
            ), patch.object(
                matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64
            ), self.assertRaisesRegex(
                matrix.MatrixValidationError, "canonical predeclared spec path"
            ):
                PRODUCTION_EVALUATION_QUEUE_VALIDATOR(
                    queue_dir,
                    spec_path=alternate_spec,
                    evaluation_queue_id="fixture-evaluation-queue",
                    evaluation_plan_sha256="e" * 64,
                    experiments=experiments,
                    seeds=(11, 22),
                )

            for field, value in (
                ("FORMAL_EVALUATION_QUEUE_ID", "replacement-queue"),
                ("FORMAL_EVALUATION_PLAN_SHA256", "f" * 64),
            ):
                with self.subTest(replaced_authority=field), patch.object(
                    matrix, "FORMAL_EVALUATION_QUEUE_DIR", queue_dir
                ), patch.object(
                    matrix, "DEFAULT_INPUT_SPEC", spec_path
                ), patch.object(
                    matrix,
                    "FORMAL_EVALUATION_QUEUE_ID",
                    "fixture-evaluation-queue",
                ), patch.object(
                    matrix, "FORMAL_EVALUATION_PLAN_SHA256", "e" * 64
                ), patch.object(
                    matrix, field, value
                ), self.assertRaisesRegex(
                    matrix.MatrixValidationError, "authorized evaluation queue identity"
                ):
                    PRODUCTION_EVALUATION_QUEUE_VALIDATOR(
                        queue_dir,
                        spec_path=spec_path,
                        evaluation_queue_id="fixture-evaluation-queue",
                        evaluation_plan_sha256="e" * 64,
                        experiments=experiments,
                        seeds=(11, 22),
                    )

    def test_input_spec_requires_presealed_queue_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary))
            for field in (
                "FORMAL_EVALUATION_QUEUE_ID",
                "FORMAL_EVALUATION_PLAN_SHA256",
            ):
                with self.subTest(unsealed=field), patch.object(
                    matrix, field, None
                ), self.assertRaisesRegex(
                    matrix.MatrixValidationError,
                    "formal matrix evaluation queue identity has not been sealed",
                ):
                    matrix._parse_spec(fixture.spec)

            payload = json.loads(fixture.spec.read_text(encoding="utf-8"))
            payload["evaluation_provenance_scope"] = (
                matrix.evaluation_queue.TEST_ONLY_PROVENANCE_SCOPE
            )
            _write_json(fixture.spec, payload)
            with self.assertRaisesRegex(
                matrix.MatrixValidationError,
                "evaluation_provenance_scope must be exactly formal",
            ):
                matrix._parse_spec(fixture.spec)

    def test_long_aggregation_rehashes_spec_source_and_evaluation_evidence(self):
        for mutation in ("spec", "source", "record"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                fixture = MatrixFixture(Path(temporary))
                tracked_source = Path(temporary) / "tracked_aggregation_source.py"
                tracked_source.write_text("VALUE = 1\n", encoding="utf-8")
                if mutation == "record":
                    postflight = json.loads(
                        (
                            fixture.roots["candidate"][11]
                            / "postflight.json"
                        ).read_text()
                    )
                    target = Path(
                        postflight["artifacts"]["ref_validation"][
                            "refcoco_val"
                        ]["records"]["path"]
                    )
                elif mutation == "spec":
                    target = fixture.spec
                else:
                    target = tracked_source
                original_comparison = matrix._comparison
                mutated = False

                def mutate_then_compare(**kwargs):
                    nonlocal mutated
                    if not mutated:
                        with target.open("a", encoding="utf-8") as handle:
                            handle.write(" \n")
                        mutated = True
                    return original_comparison(**kwargs)

                source_context = (
                    patch.object(
                        matrix,
                        "_aggregation_source_paths",
                        return_value=(
                            Path(matrix.__file__).resolve(),
                            tracked_source.resolve(),
                        ),
                    )
                    if mutation == "source"
                    else contextlib.nullcontext()
                )
                with source_context, patch.object(
                    matrix, "_comparison", side_effect=mutate_then_compare
                ), self.assertRaisesRegex(
                    matrix.MatrixValidationError, "changed during aggregation"
                ):
                    matrix.aggregate_spec(
                        fixture.spec, bootstrap_iterations=2
                    )


class CanonicalMatrixSpecTest(unittest.TestCase):
    def _fixture(self, root: Path):
        queue_dir = root / "queue"
        output_root = root / "evaluations"
        queue_dir.mkdir()
        items = []
        for seed in matrix.FORMAL_TRAIN_SEEDS:
            for experiment_id in matrix.FORMAL_EXPERIMENT_IDS:
                evaluation_root = output_root / experiment_id / f"seed{seed}"
                evaluation_root.mkdir(parents=True)
                items.append(
                    {
                        "run_id": f"{experiment_id}:{seed}",
                        "evaluation_root": str(evaluation_root),
                    }
                )
        plan = {
            "queue_dir": str(queue_dir),
            "queue_id": "canonical-queue-id",
            "profile": matrix.PROFILE,
            "provenance_scope": matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE,
            "items": items,
        }
        queue = {
            "status": "completed",
            "plan_sha256": "a" * 64,
            "plan": plan,
        }
        verification = {
            "status": "passed",
            "errors": [],
            "queue_id": "canonical-queue-id",
            "plan_sha256": "a" * 64,
            "provenance_scope": matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE,
            "verified_items": [{"run_id": item["run_id"]} for item in items],
        }
        output = root / "attestations" / "table_c_input.json"
        _write_json(
            output,
            matrix.evaluation_queue._aggregation_input_spec_payload(
                plan, "a" * 64
            ),
        )
        return queue_dir, output_root, output, queue, verification

    def test_builder_verifies_predeclared_spec_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, output_root, output, queue, verification = self._fixture(root)
            with patch.object(
                matrix, "FORMAL_EVALUATION_QUEUE_DIR", queue_dir
            ), patch.object(
                matrix.evaluation_queue, "DEFAULT_QUEUE_DIR", queue_dir
            ), patch.object(
                matrix.evaluation_queue, "DEFAULT_OUTPUT_ROOT", output_root
            ), patch.object(
                matrix, "DEFAULT_INPUT_SPEC", output
            ), patch.object(
                matrix, "FORMAL_EVALUATION_QUEUE_ID", "canonical-queue-id"
            ), patch.object(
                matrix, "FORMAL_EVALUATION_PLAN_SHA256", "a" * 64
            ), patch.object(
                matrix.evaluation_queue, "load_queue", return_value=queue
            ), patch.object(
                matrix.evaluation_queue,
                "verify_queue",
                return_value=verification,
            ):
                payload = matrix.build_canonical_spec()
                persisted = json.loads(output.read_text(encoding="ascii"))
                self.assertEqual(payload, persisted)
                self.assertEqual(payload["evaluation_queue_id"], "canonical-queue-id")
                self.assertEqual(
                    payload["evaluation_plan_sha256"], "a" * 64
                )
                self.assertEqual(
                    payload["evaluation_provenance_scope"],
                    matrix.evaluation_queue.FORMAL_PROVENANCE_SCOPE,
                )
                self.assertEqual(
                    [row["id"] for row in payload["experiments"]],
                    list(matrix.FORMAL_EXPERIMENT_IDS),
                )
                self.assertEqual(matrix.build_canonical_spec(), payload)

    def test_builder_rejects_incomplete_verification_and_root_drift(self):
        for mutation in ("verification", "root"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                queue_dir, output_root, output, queue, verification = self._fixture(root)
                if mutation == "verification":
                    verification["verified_items"].pop()
                    expected = "verification count"
                else:
                    queue["plan"]["items"][0]["evaluation_root"] = str(
                        output_root / "wrong"
                    )
                    (output_root / "wrong").mkdir()
                    expected = "root drifted"
                with patch.object(
                    matrix, "FORMAL_EVALUATION_QUEUE_DIR", queue_dir
                ), patch.object(
                    matrix.evaluation_queue, "DEFAULT_QUEUE_DIR", queue_dir
                ), patch.object(
                    matrix.evaluation_queue, "DEFAULT_OUTPUT_ROOT", output_root
                ), patch.object(
                    matrix, "DEFAULT_INPUT_SPEC", output
                ), patch.object(
                    matrix, "FORMAL_EVALUATION_QUEUE_ID", "canonical-queue-id"
                ), patch.object(
                    matrix, "FORMAL_EVALUATION_PLAN_SHA256", "a" * 64
                ), patch.object(
                    matrix.evaluation_queue, "load_queue", return_value=queue
                ), patch.object(
                    matrix.evaluation_queue,
                    "verify_queue",
                    return_value=verification,
                ), self.assertRaisesRegex(matrix.MatrixValidationError, expected):
                    matrix.build_canonical_spec()


if __name__ == "__main__":
    unittest.main()
