import copy
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tools import aggregate_stageb_paper_results as paper_aggregate
from tools import run_stageb_matrix_validation_queue as queue_runner
from tools import run_stageb_paper_evaluations as evaluator


REAL_PRETRAINING_RECOVERY_VERIFIER = queue_runner._verify_pretraining_recovery


SMALL_REF_CONTRACT = {
    "refcoco_val": {
        "rows": 4,
        "sha256": hashlib.sha256(b"refcoco_val").hexdigest(),
    },
    "refcocop_val": {
        "rows": 3,
        "sha256": hashlib.sha256(b"refcocop_val").hexdigest(),
    },
    "refcocog_val": {
        "rows": 2,
        "sha256": hashlib.sha256(b"refcocog_val").hexdigest(),
    },
}


FAKE_EVALUATOR = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__REPO_ROOT__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import aggregate_stageb_paper_results as paper_aggregate
from tools import run_stageb_paper_evaluations as evaluator
from tools.compare_stageb_fpr95_records import exact_fpr95
from tools.stageb_screen_calibration import build_manifest, summary_fields

BEHAVIORS = __BEHAVIORS__
SMALL_REF_CONTRACT = __SMALL_REF_CONTRACT__
evaluator.REF_SPLIT_CONTRACT.update(SMALL_REF_CONTRACT)
paper_aggregate.REF_SPLIT_CONTRACT.update(SMALL_REF_CONTRACT)


def arg(name):
    return sys.argv[sys.argv.index(name) + 1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, roles):
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "roles": roles,
    }


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if sys.argv[1] != "run":
    raise SystemExit(2)
training_root = Path(arg("--training-run-root")).resolve()
training_queue = Path(arg("--training-queue-dir")).resolve()
output = Path(arg("--output-dir")).resolve()
profile = arg("--profile")
data_root = Path(arg("--data-root")).resolve()
identity = json.loads((training_root / "run_identity.json").read_text())
run_id = identity["run_id"]
seed = identity["seed"]
evaluation_id = run_id.replace(":", "_seed")
behavior = BEHAVIORS.get(run_id, "success")
output.mkdir(parents=True, exist_ok=False)
attempts = output.parent / (output.name + ".attempts")
with attempts.open("a", encoding="utf-8") as handle:
    handle.write(run_id + "\n")
if behavior == "spawn_descendant_running":
    descendant = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    write(
        output / "process_tree.json",
        {"leader_pid": os.getpid(), "descendant_pid": descendant.pid},
    )
    time.sleep(30)
queue = json.loads((training_queue / "queue.json").read_text())
checkpoint = training_root / "checkpoint_iter.pth"
cache = evaluator.HashCache()
calibration_source = data_root / "calibration_source.jsonl"
calibration_audit = data_root / "calibration_audit.json"
source_record = evaluator._file_record(
    calibration_source, cache, roles=("matrix_calibration_source",)
)
source_record["rows"] = len(calibration_source.read_text().splitlines())
audit_record = evaluator._file_record(
    calibration_audit, cache, roles=("matrix_calibration_audit",)
)
calibration_contract = {
    "source_manifest": source_record,
    "source_audit": audit_record,
    "unique_images": source_record["rows"],
    "scope": "proposal_covered_verified",
    "single_edit_provenance": True,
    "strict_union_image_overlap": 0,
    "train_calibration_image_overlap": 0,
}
input_records = [record(checkpoint, ["evaluation_checkpoint"])]
if behavior != "missing_eval_source":
    input_records.append(
        record(Path(__file__), ["evaluation_code_dependency"])
    )
inputs = {"algorithm": "sha256", "records": input_records}
launch = {
    "schema": "pivot.stageb.paper_evaluation_launch/v1",
    "status": "running",
    "repository_root": str(evaluator.REPO_ROOT),
    "artifact_repository_root": str(evaluator.ARTIFACT_REPOSITORY_ROOT),
    "artifact_outputs_root": str(evaluator.ARTIFACT_OUTPUTS_ROOT),
    "evaluation_id": evaluation_id,
    "output_dir": str(output),
    "source": {
        "kind": "pivot_token_ablation_training_run",
        "evaluation_id": evaluation_id,
        "training_run_id": run_id,
        "training_seed": seed,
        "training_run_root": str(training_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "training_queue_id": queue["plan"]["queue_id"],
        "training_queue_plan_sha256": queue["plan_sha256"],
    },
    "protocol": {
        "profile": "screen_validation" if behavior == "non_matrix" else profile,
        "screen_calibration": calibration_contract,
    },
    "inputs": inputs,
    "completed_phases": [],
}
write(output / "launch_manifest.json", launch)
if behavior == "failed":
    launch["status"] = "failed"
    launch["error"] = "synthetic failure"
    write(output / "launch_manifest.json", launch)
    raise SystemExit(1)
if behavior == "running":
    time.sleep(0.7)

section = output / "validation_calibration"
records_dir = section / "per_example_records"
checkpoint_run_id = evaluator._checkpoint_run_id(checkpoint)
ref_summary = []
for split_index, split in enumerate(evaluator.SCREEN_REF_SPLITS):
    contract = SMALL_REF_CONTRACT[split]
    count = contract["rows"]
    records_path = records_dir / f"{checkpoint_run_id}__{split}.records.jsonl"
    records = []
    for index in range(count):
        correct = index < max(1, count // 2)
        records.append({
            "schema": "stageb-eval-record-v1",
            "task": "ref",
            "manifest_key": f"ref:{split}",
            "manifest_sha256": contract["sha256"],
            "manifest_n": count,
            "manifest_index": index,
            "sample_id": f"{split}:{index}",
            "image_id": split_index * 100 + index // 2,
            "ann_id": 10 + index,
            "ref_id": 20 + index,
            "sent_id": 30 + index,
            "split": split,
            "run_id": checkpoint_run_id,
            "valid": True,
            "correct50": correct,
            "top1_iou": 0.75 if correct else 0.25,
        })
    write_jsonl(records_path, records)
    ref_summary.append({
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "run_id": checkpoint_run_id,
        "seed": evaluator.EVAL_SEED + split_index * 100000,
        "max_batches": 0,
        "invalid_records": 0,
        "dataset": split,
        "manifest_n": count,
        "manifest_sha256": contract["sha256"],
        "num_expressions": count,
        "records_jsonl": str(records_path),
        "acc50": float(np.mean([row["correct50"] for row in records])),
        "mean_iou": float(np.mean([row["top1_iou"] for row in records])),
    })

derived_path = section / "tn_eval_inputs" / "tn_matrix_calibration.jsonl"
binding = build_manifest(
    source_path=calibration_source,
    audit_path=calibration_audit,
    derived_path=derived_path,
    data_root=data_root,
)
derived_rows = [
    json.loads(line) for line in derived_path.read_text().splitlines()
]
positive = []
negative = []
tn_records = []
for index, source in enumerate(derived_rows):
    pos_score = 0.65 + index * 0.01
    neg_score = 0.25 + index * 0.01
    positive.append(pos_score)
    negative.append(neg_score)
    tn_records.append({
        "schema": "stageb-eval-record-v1",
        "task": "tn",
        "manifest_key": "tn_global",
        "manifest_sha256": binding.derived_manifest["sha256"],
        "manifest_n": len(derived_rows),
        "manifest_index": index,
        "sample_id": source["sample_id"],
        "image_id": source["image_id"],
        "ann_id": source["ann_id"],
        "ref_id": source["ref_id"],
        "sent_id": source["sent_id"],
        "split": binding.eval_split,
        "run_id": checkpoint_run_id,
        "valid": True,
        "pos_score": pos_score,
        "neg_score": neg_score,
    })
tn_records_path = records_dir / f"{checkpoint_run_id}__tn_global.records.jsonl"
write_jsonl(tn_records_path, tn_records)
fpr = exact_fpr95(positive, negative)
tn_summary = {
    "checkpoint": str(checkpoint),
    "checkpoint_name": checkpoint.name,
    "run_id": checkpoint_run_id,
    "seed": evaluator.EVAL_SEED,
    "max_batches": 0,
    "invalid_records": 0,
    "manifest_n": len(derived_rows),
    "num_pairs": len(derived_rows),
    "manifest_sha256": binding.derived_manifest["sha256"],
    "records_jsonl": str(tn_records_path),
    "fpr95tpr": float(fpr["fpr"]),
    "threshold_at_95tpr": float(fpr["threshold"]),
    "pair_win_rate": float(np.mean(np.asarray(positive) > np.asarray(negative))),
    **summary_fields(binding),
}
summary_path = section / "summary.json"
write(summary_path, {"refcoco": ref_summary, "tn": [tn_summary]})

rehash = evaluator._rehash_inputs(launch)
rehash_path = output / "input_rehash.json"
write(rehash_path, rehash)
launch["input_rehash_artifact"] = record(rehash_path, ["input_rehash"])
if behavior == "empty_postflight":
    postflight = {
        "schema": evaluator.POSTFLIGHT_SCHEMA,
        "status": "passed",
        "profile": profile,
        "evaluation_id": evaluation_id,
        "input_rehash": rehash,
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
    }
else:
    postflight = evaluator._postflight_screen(launch, rehash)

if behavior == "corrupt_summary":
    summary = json.loads(summary_path.read_text())
    summary["refcoco"][0]["acc50"] = 0.123456789
    write(summary_path, summary)
elif behavior == "corrupt_records":
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0]["correct50"] = not rows[0]["correct50"]
    write_jsonl(records_path, rows)
elif behavior == "corrupt_contract":
    postflight["contracts"]["zero_invalid_records"] = False
elif behavior == "corrupt_rehash":
    rehash["records"][0]["observed_sha256"] = "0" * 64
    write(rehash_path, rehash)
    launch["input_rehash_artifact"] = record(rehash_path, ["input_rehash"])
    postflight["input_rehash"] = rehash

postflight_path = output / "postflight.json"
write(postflight_path, postflight)
launch["postflight"] = postflight
launch["postflight_artifact"] = record(postflight_path, ["postflight"])
launch["completed_phases"] = [{
    "phase_id": "validation_calibration",
    "status": "completed",
    "returncode": 0,
}]
launch["status"] = "completed"
write(output / "launch_manifest.json", launch)
'''


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _calibration_source_row(index: int) -> dict:
    return {
        "sample_id": f"calibration:{index}",
        "image_id": 1000 + index,
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


class QueueFixture:
    def __init__(self, root: Path, behaviors=None, *, remaining_running=True):
        self.root = root
        self.output_root = root / "evaluations" / "matrix_validation"
        self.training_root = root / "training"
        self.queue_dir = root / "evaluation_queue"
        self.data_root = root / "data"
        self.data_root.mkdir(parents=True)
        _write_jsonl(
            self.data_root / "calibration_source.jsonl",
            [_calibration_source_row(index) for index in range(4)],
        )
        _write_json(
            self.data_root / "calibration_audit.json",
            {"schema": "synthetic-table-b-audit"},
        )
        self.runner = root / "fake_evaluator.py"
        self.runner.write_text(
            FAKE_EVALUATOR.replace("__BEHAVIORS__", repr(behaviors or {}))
            .replace("__REPO_ROOT__", repr(str(queue_runner.REPO_ROOT)))
            .replace("__SMALL_REF_CONTRACT__", repr(SMALL_REF_CONTRACT)),
            encoding="utf-8",
        )
        self.runner.chmod(0o755)
        self.bindings = {
            "completed_l0_l4_seed17": {
                "queue_id": "fixture-first",
                "plan_sha256": "1" * 64,
                "run_ids": tuple(f"L{index}:17" for index in range(5)),
            },
            "remaining_table_c": {
                "queue_id": "fixture-remaining",
                "plan_sha256": "2" * 64,
                "run_ids": (
                    *(f"L{index}:17" for index in range(5, 11)),
                    *(f"L{index}:{seed}" for seed in (42, 73) for index in range(11)),
                ),
            },
        }
        self.training_queues = {}
        self.sources = {}
        for role, binding in self.bindings.items():
            queue_dir = root / f"training_queue_{role}"
            plan_items = [
                {"run_id": run_id, "runner": "token"}
                for run_id in binding["run_ids"]
            ]
            queue = {
                "status": (
                    "running"
                    if role == "remaining_table_c" and remaining_running
                    else "completed"
                ),
                "plan_sha256": binding["plan_sha256"],
                "plan": {
                    "queue_id": binding["queue_id"],
                    "items": plan_items,
                },
                "items": [
                    {"run_id": run_id, "runner": "token", "status": "completed"}
                    for run_id in binding["run_ids"]
                ],
            }
            _write_json(queue_dir / "queue.json", queue)
            self.training_queues[queue_dir.resolve()] = queue
            for run_id in binding["run_ids"]:
                row, raw_seed = run_id.split(":")
                seed = int(raw_seed)
                training_root = self.training_root / row / f"seed{seed}"
                checkpoint = training_root / "checkpoint_iter.pth"
                config = training_root / "config.py"
                sequence = training_root / "sequence_manifest.json"
                launch = training_root / "launch_manifest.json"
                postflight = training_root / "postflight.json"
                detached = queue_dir / "jobs" / run_id.replace(":", "_")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(run_id.encode("ascii"))
                config.write_text("FIXTURE = True\n", encoding="utf-8")
                _write_json(sequence, {"status": "completed", "run_id": run_id})
                _write_json(launch, {"status": "completed", "run_id": run_id})
                _write_json(postflight, {"status": "passed", "run_id": run_id})
                _write_json(detached / "launch.json", {"status": "launched"})
                _write_json(detached / "status.json", {"status": "completed"})
                _write_json(
                    training_root / "run_identity.json",
                    {"run_id": run_id, "seed": seed},
                )
                self.sources[run_id] = evaluator.EvaluationSource(
                    kind="pivot_token_ablation_training_run",
                    evaluation_id=run_id.replace(":", "_seed"),
                    config=config.resolve(),
                    checkpoint=checkpoint.resolve(),
                    checkpoint_sha256=queue_runner._sha256_file(checkpoint),
                    training_run_id=run_id,
                    training_seed=seed,
                    training_run_root=training_root.resolve(),
                    sequence_manifest=sequence.resolve(),
                    training_phase="final",
                    diagnostic_only=False,
                    final_phase_id="joint",
                    final_phase_manifest=launch.resolve(),
                    training_postflight=postflight.resolve(),
                    selected_phase_id="joint",
                    selected_phase_manifest=launch.resolve(),
                    selected_training_postflight=postflight.resolve(),
                    training_queue_manifest=(queue_dir / "queue.json").resolve(),
                    training_queue_detached_launch=(detached / "launch.json").resolve(),
                    training_queue_detached_status=(detached / "status.json").resolve(),
                    training_queue_id=binding["queue_id"],
                    training_queue_plan_sha256=binding["plan_sha256"],
                    training_data=(config.resolve(),),
                )

    def complete_training(self):
        for queue in self.training_queues.values():
            queue["status"] = "completed"

    def load_training_queue(self, path):
        return self.training_queues[Path(path).resolve()]

    def verify_training_queue(self, path):
        queue = self.load_training_queue(path)
        return {
            "status": "passed" if queue["status"] == "completed" else "failed",
            "queue_id": queue["plan"]["queue_id"],
            "plan_sha256": queue["plan_sha256"],
            "errors": [],
            "verified_items": [],
        }

    def resolve_source(self, training_root, training_queue_dir, cache):
        identity = json.loads((Path(training_root) / "run_identity.json").read_text())
        return self.sources[identity["run_id"]]

    def create(
        self,
        queue_dir=None,
        *,
        test_only_capability=queue_runner._TEST_ONLY_CREATE_CAPABILITY,
    ):
        with patch.object(
            queue_runner, "DEFAULT_LEASE_ROOT", self.root / "leases"
        ):
            return queue_runner.create_queue(
                self.queue_dir if queue_dir is None else Path(queue_dir),
                training_queue_dirs=list(self.training_queues),
                output_root=self.output_root,
                runner_python=Path(sys.executable),
                evaluation_runner=self.runner,
                evaluation_python=Path(sys.executable),
                data_root=self.data_root,
                evaluation_source_paths=[self.runner],
                lease_root=self.root / "leases",
                aggregation_input_spec_path=(
                    self.root
                    / "attestations"
                    / "table_c_matrix_validation_input.json"
                ),
                test_only_capability=test_only_capability,
            )


class StageBMatrixValidationQueueTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        queue_runner._VALIDATED_SOURCE_CONTRACT_CACHE.clear()
        self.stack.callback(
            queue_runner._VALIDATED_SOURCE_CONTRACT_CACHE.clear
        )
        self.stack.enter_context(
            patch.object(
                queue_runner,
                "DEFAULT_OUTPUT_ROOT",
                self.root / "evaluations" / "matrix_validation",
            )
        )
        self.stack.enter_context(
            patch.object(
                queue_runner,
                "DEFAULT_QUEUE_DIR",
                self.root / "evaluation_queue",
            )
        )
        self.stack.enter_context(
            patch.object(
                queue_runner,
                "DEFAULT_TRAINING_OUTPUT_ROOT",
                self.root / "training",
            )
        )
        self.stack.enter_context(
            patch.object(
                queue_runner,
                "_verify_pretraining_recovery",
                side_effect=lambda _queue_dir, _queue, role: (
                    {"fixture_recovery": True}
                    if role == "remaining_table_c"
                    else None
                ),
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

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def fixture(self, behaviors=None, *, remaining_running=True):
        fixture = QueueFixture(
            self.root, behaviors, remaining_running=remaining_running
        )
        self.stack.enter_context(
            patch.object(
                queue_runner, "LOCKED_TRAINING_QUEUES", fixture.bindings
            )
        )
        self.stack.enter_context(
            patch.object(
                queue_runner.training_queue,
                "load_queue",
                side_effect=fixture.load_training_queue,
            )
        )
        self.stack.enter_context(
            patch.object(
                queue_runner.training_queue,
                "verify_queue",
                side_effect=fixture.verify_training_queue,
            )
        )
        self.stack.enter_context(
            patch.object(
                queue_runner,
                "_resolve_formal_source",
                side_effect=fixture.resolve_source,
            )
        )
        return fixture

    def ready(self, fixture):
        fixture.complete_training()
        state = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(state["status"], "planned", state.get("failure"))
        self.assertIsNotNone(state["training_attestation"])
        return state

    def orphan_attestation_failure(self, mutate):
        fixture = self.fixture(remaining_running=False)
        created = fixture.create()
        payload = queue_runner._build_training_attestation(created)
        self.assertIsNotNone(payload)
        mutate(payload)
        payload["semantic_sha256"] = queue_runner._canonical_sha(
            queue_runner._training_attestation_semantic_payload(payload)
        )
        _write_json(fixture.queue_dir / "training_attestation.json", payload)
        failed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure"]["stage"], "training_gate")
        self.assertIsNone(failed["training_attestation"])
        self.assertFalse(Path(failed["plan"]["lease_path"]).exists())
        return failed

    def assert_resealed_attestation_rejected(
        self, created, payload, *, pattern
    ):
        payload["semantic_sha256"] = queue_runner._canonical_sha(
            queue_runner._training_attestation_semantic_payload(payload)
        )
        attestation_path = Path(created["plan"]["queue_dir"]) / (
            "training_attestation.json"
        )
        _write_json(attestation_path, payload)
        resealed = copy.deepcopy(created)
        resealed["training_attestation"] = queue_runner._file_record(
            attestation_path
        )
        _write_json(Path(created["plan"]["queue_dir"]) / "queue.json", resealed)
        with self.assertRaisesRegex(queue_runner.MatrixQueueError, pattern):
            loaded = queue_runner._load_queue_structural(
                Path(created["plan"]["queue_dir"])
            )
            observed = queue_runner._training_attestation_payload(loaded)
            queue_runner._replay_training_attestation(loaded, observed)

    def test_create_predeclares_exact_33_while_training_runs(self):
        fixture = self.fixture()
        queue = fixture.create()
        self.assertEqual(queue["status"], "waiting_training")
        self.assertEqual(
            queue["plan"]["provenance_scope"],
            queue_runner.TEST_ONLY_PROVENANCE_SCOPE,
        )
        self.assertEqual(
            [item["run_id"] for item in queue["plan"]["items"]],
            list(queue_runner.EXPECTED_RUN_IDS),
        )
        self.assertFalse(fixture.output_root.exists())
        spec_path = Path(queue["aggregation_input_spec"]["path"])
        self.assertTrue(spec_path.is_file())
        spec = json.loads(spec_path.read_text(encoding="ascii"))
        self.assertEqual(spec["evaluation_queue_id"], queue["plan"]["queue_id"])
        self.assertEqual(spec["evaluation_plan_sha256"], queue["plan_sha256"])
        self.assertEqual(
            spec["evaluation_provenance_scope"],
            queue_runner.TEST_ONLY_PROVENANCE_SCOPE,
        )
        self.assertEqual(
            [row["id"] for row in spec["experiments"]],
            list(queue_runner.ROWS),
        )
        contract = json.loads(
            (fixture.queue_dir / "predeclared_contract.json").read_text()
        )
        self.assertEqual(contract["profile"], evaluator.MATRIX_PROFILE)
        self.assertEqual(
            [record["path"] for record in queue["plan"]["evaluation_sources"]],
            [str(fixture.runner.resolve())],
        )
        controller_paths = tuple(
            Path(record["path"])
            .relative_to(queue_runner.REPO_ROOT)
            .as_posix()
            for record in queue["plan"]["controller_sources"]
        )
        self.assertEqual(
            controller_paths,
            queue_runner.CONTROLLER_SOURCE_RELATIVE_PATHS,
        )
        self.assertNotIn("controller_source", queue["plan"])
        self.assertEqual(
            contract["evaluation_sources"],
            queue["plan"]["evaluation_sources"],
        )
        self.assertEqual(
            contract["controller_sources"],
            queue["plan"]["controller_sources"],
        )
        waiting = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(waiting["status"], "waiting_training")
        self.assertFalse((fixture.queue_dir / "jobs").exists())
        self.assertFalse(Path(queue["plan"]["lease_path"]).exists())

    def test_production_source_profiles_are_exact_and_exclude_late_bound_code(self):
        controller = queue_runner._controller_source_paths()
        child = queue_runner._child_evaluation_source_paths(
            evaluation_runner=queue_runner.DEFAULT_EVALUATION_RUNNER.resolve(),
            injected_paths=None,
        )
        controller_relative = {
            path.relative_to(queue_runner.REPO_ROOT).as_posix()
            for path in controller
        }
        child_relative = {
            path.relative_to(queue_runner.REPO_ROOT).as_posix()
            for path in child
        }
        self.assertEqual(len(controller), 12)
        self.assertEqual(len(child), 75)
        self.assertEqual(len(controller_relative | child_relative), 77)
        self.assertEqual(len(controller_relative & child_relative), 10)
        self.assertEqual(
            tuple(sorted(controller_relative)),
            queue_runner.CONTROLLER_SOURCE_RELATIVE_PATHS,
        )
        self.assertFalse(
            child_relative & queue_runner.LATE_BOUND_SOURCE_RELATIVE_PATHS
        )
        self.assertIn(
            "tools/stageb_profile_dependency_audit.py", controller_relative
        )
        self.assertNotIn(
            "tools/run_stageb_matrix_validation_queue.py", child_relative
        )

    def test_injected_sources_require_nonpersistent_test_capability(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "in-process test-only capability"
        ):
            fixture.create(test_only_capability=None)
        self.assertFalse(fixture.queue_dir.exists())

        created = fixture.create()
        queue_id = created["plan"]["queue_id"]
        queue_runner._AUTHORIZED_TEST_QUEUE_IDS.remove(queue_id)
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError,
            "test-only queue provenance lacks its in-process capability",
        ):
            queue_runner.load_queue(fixture.queue_dir)

    def test_formal_replay_rejects_injected_incomplete_source_closure(self):
        fixture = self.fixture()
        created = fixture.create()
        plan = copy.deepcopy(created["plan"])
        plan["provenance_scope"] = queue_runner.FORMAL_PROVENANCE_SCOPE
        runner = queue_runner.DEFAULT_EVALUATION_RUNNER.resolve(strict=True)
        canonical = queue_runner._child_evaluation_source_paths(
            evaluation_runner=runner,
            injected_paths=None,
        )
        omitted = next(path for path in canonical if path != runner)
        plan["evaluation_runner"] = queue_runner._file_record(runner)
        plan["evaluation_sources"] = [
            queue_runner._file_record(path)
            for path in canonical
            if path != omitted
        ]
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "exact 75-file"
        ):
            queue_runner._validate_source_contract_structure(plan)

    def test_formal_replay_rejects_noncanonical_evaluation_runner(self):
        fixture = self.fixture()
        created = fixture.create()
        plan = copy.deepcopy(created["plan"])
        plan["provenance_scope"] = queue_runner.FORMAL_PROVENANCE_SCOPE
        plan["evaluation_runner"] = queue_runner._file_record(fixture.runner)
        plan["evaluation_sources"] = [
            queue_runner._file_record(fixture.runner)
        ]
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "runner is not canonical"
        ):
            queue_runner._validate_source_contract_structure(plan)

    def test_source_contract_cache_preserves_auth_and_live_hash_checks(self):
        fixture = self.fixture()
        created = fixture.create()
        plan = created["plan"]
        queue_runner._VALIDATED_SOURCE_CONTRACT_CACHE.clear()
        real_controller = queue_runner._controller_source_paths
        real_support = queue_runner._profile_support_source_paths
        with (
            patch.object(
                queue_runner,
                "_controller_source_paths",
                wraps=real_controller,
            ) as controller_paths,
            patch.object(
                queue_runner,
                "_profile_support_source_paths",
                wraps=real_support,
            ) as support_paths,
        ):
            queue_runner._validate_source_contract_structure(plan)
            queue_runner._validate_source_contract_structure(plan)
            self.assertEqual(controller_paths.call_count, 1)
            self.assertEqual(support_paths.call_count, 1)

            incomplete = copy.deepcopy(plan)
            incomplete["controller_sources"] = incomplete[
                "controller_sources"
            ][:-1]
            with self.assertRaisesRegex(
                queue_runner.MatrixQueueError,
                "controller source closure drifted",
            ):
                queue_runner._validate_source_contract_structure(incomplete)
            self.assertEqual(controller_paths.call_count, 2)

        queue_id = plan["queue_id"]
        queue_runner._AUTHORIZED_TEST_QUEUE_IDS.remove(queue_id)
        try:
            with self.assertRaisesRegex(
                queue_runner.MatrixQueueError,
                "test-only queue provenance lacks",
            ):
                queue_runner._validate_source_contract_structure(plan)
        finally:
            queue_runner._AUTHORIZED_TEST_QUEUE_IDS.add(queue_id)

        fixture.runner.write_text(
            fixture.runner.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError,
            "evaluation runner file identity drifted",
        ):
            queue_runner.load_queue(fixture.queue_dir)

    def test_relocated_queue_is_read_only_to_the_live_controller(self):
        fixture = self.fixture()
        created = fixture.create()
        relocated = self.root / "relocated-execution"
        relocated.mkdir()
        os.symlink(
            (queue_runner.REPO_ROOT / "outputs").resolve(),
            relocated / "outputs",
            target_is_directory=True,
        )
        foreign = copy.deepcopy(created)
        foreign["plan"]["repository_root"] = str(relocated.resolve())

        with (
            patch.object(
                queue_runner, "_load_queue_structural", return_value=foreign
            ),
            patch.object(queue_runner, "_advance_training_gate") as gate,
            self.assertRaisesRegex(
                queue_runner.MatrixQueueError,
                "mutation requires execution",
            ),
        ):
            queue_runner.advance_once(fixture.queue_dir)
        gate.assert_not_called()

        with (
            patch.object(queue_runner, "load_queue", return_value=foreign),
            patch.object(queue_runner.subprocess, "Popen") as popen,
            self.assertRaisesRegex(
                queue_runner.MatrixQueueError,
                "mutation requires execution",
            ),
        ):
            queue_runner.detach_queue(fixture.queue_dir, poll_seconds=0.05)
        popen.assert_not_called()

        revision = foreign["revision"]
        with (
            patch.object(queue_runner, "_write_json_atomic") as write,
            self.assertRaisesRegex(
                queue_runner.MatrixQueueError,
                "mutation requires execution",
            ),
        ):
            queue_runner._save_queue(foreign)
        write.assert_not_called()
        self.assertEqual(foreign["revision"], revision)

    def test_production_child_profile_count_guard_is_fail_closed(self):
        common = evaluator.evaluation_common_code_paths()
        provenance = evaluator.evaluation_source_provenance_paths("token")
        cases = (
            (common[:-1], provenance),
            (common, provenance[:-1]),
            (common, [common[0], *provenance[1:]]),
        )
        for malformed_common, malformed_provenance in cases:
            with self.subTest(
                common=len(malformed_common),
                provenance=len(malformed_provenance),
            ), patch.object(
                evaluator,
                "evaluation_common_code_paths",
                return_value=malformed_common,
            ), patch.object(
                evaluator,
                "evaluation_source_provenance_paths",
                return_value=malformed_provenance,
            ), self.assertRaisesRegex(
                queue_runner.MatrixQueueError,
                "exactly 72 disjoint common files plus 3 provenance files",
            ):
                queue_runner._child_evaluation_source_paths(
                    evaluation_runner=(
                        queue_runner.DEFAULT_EVALUATION_RUNNER.resolve()
                    ),
                    injected_paths=None,
                )

    def test_controller_sources_are_required_by_plan_replay(self):
        fixture = self.fixture()
        fixture.create()
        queue_path = fixture.queue_dir / "queue.json"
        payload = json.loads(queue_path.read_text())
        payload["plan"].pop("controller_sources")
        payload["plan_sha256"] = queue_runner._canonical_sha(payload["plan"])
        _write_json(queue_path, payload)
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "controller source closure is empty"
        ):
            queue_runner.load_queue(fixture.queue_dir)

    def test_real_recovery_verifier_binds_receipt_and_source(self):
        from tools import recover_stageb_serial_matrix_pretraining_failure as recovery

        receipt = self.root / "recovery_receipt.json"
        receipt.write_text('{"status":"archived"}\n', encoding="ascii")
        receipt_record = queue_runner._file_record(receipt)
        binding = {
            "queue_id": "fixture-remaining",
            "plan_sha256": "2" * 64,
            "run_ids": ("L2:42",),
        }
        queue = {
            "status": "completed",
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
        replay = {
            "status": "passed",
            "queue_id": binding["queue_id"],
            "plan_sha256": binding["plan_sha256"],
            "run_id": "L2:42",
            "current_item_status": "completed",
            "receipt_sha256": "b" * 64,
            "archived_evidence_verified": True,
            "semantic_replay": recovery.SEMANTIC_REPLAY_PROOF,
            "verifier_source": queue_runner._file_record(Path(recovery.__file__)),
        }
        with (
            patch.object(
                queue_runner,
                "LOCKED_TRAINING_QUEUES",
                {"remaining_table_c": binding},
            ),
            patch.object(
                queue_runner,
                "TABLE_C_PRETRAINING_RECOVERY_RECEIPT",
                receipt,
            ),
            patch.object(recovery, "verify_recovery", return_value=replay) as verify,
        ):
            observed = REAL_PRETRAINING_RECOVERY_VERIFIER(
                self.root / "training_queue", queue, "remaining_table_c"
            )
        self.assertEqual(observed["receipt"], receipt_record)
        self.assertEqual(observed["receipt_sha256"], "b" * 64)
        self.assertEqual(
            observed["verifier_source"]["path"],
            str(Path(recovery.__file__).resolve()),
        )
        verify.assert_called_once_with(self.root / "training_queue", receipt)

    def test_failed_training_queue_fails_waiting_gate(self):
        fixture = self.fixture()
        fixture.create()
        remaining = next(
            queue
            for queue in fixture.training_queues.values()
            if queue["plan"]["queue_id"] == "fixture-remaining"
        )
        remaining["status"] = "failed"
        failed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["items"][0]["status"], "failed")
        self.assertFalse(fixture.output_root.exists())

    def test_waiting_training_source_drift_fails_before_attestation(self):
        fixture = self.fixture()
        created = fixture.create()
        controller_path = Path(
            created["plan"]["controller_sources"][0]["path"]
        ).resolve()
        real_file_record = queue_runner._file_record

        def drifted_file_record(path):
            record = real_file_record(path)
            if Path(path).resolve() == controller_path:
                record["sha256"] = "0" * 64
            return record

        with patch.object(
            queue_runner, "_file_record", side_effect=drifted_file_record
        ):
            failed = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure"]["stage"], "training_gate")
        self.assertIn("controller source 0 file identity drifted", failed["failure"]["error"])
        self.assertIsNone(failed["training_attestation"])
        self.assertFalse((fixture.queue_dir / "training_attestation.json").exists())
        self.assertFalse(Path(failed["plan"]["lease_path"]).exists())
        self.assertFalse(fixture.output_root.exists())

    def test_orphan_attestation_rejects_missing_file_role(self):
        failed = self.orphan_attestation_failure(
            lambda payload: payload["sources"][0]["files"].pop("config")
        )
        self.assertIn("file-role projection is not exact", failed["failure"]["error"])

    def test_orphan_attestation_rejects_source_order_drift(self):
        def swap_sources(payload):
            payload["sources"][0], payload["sources"][1] = (
                payload["sources"][1],
                payload["sources"][0],
            )

        failed = self.orphan_attestation_failure(swap_sources)
        self.assertIn("plan binding drifted", failed["failure"]["error"])

    def test_orphan_attestation_rejects_resolved_semantic_drift(self):
        def mutate_resolved_source(payload):
            payload["sources"][0]["resolved_source"]["formal_contract_id"] = (
                "forged-contract"
            )

        failed = self.orphan_attestation_failure(mutate_resolved_source)
        self.assertIn("canonical 33-source replay", failed["failure"]["error"])

    def test_attestation_replay_rejects_every_resealed_semantic_mutation(self):
        fixture = self.fixture(remaining_running=False)
        created = fixture.create()
        extra_training_data = self.root / "extra_training_data.jsonl"
        extra_training_data.write_text("{}\n", encoding="ascii")
        first_run_id = queue_runner.EXPECTED_RUN_IDS[0]
        first_source = fixture.sources[first_run_id]
        fixture.sources[first_run_id] = replace(
            first_source,
            training_data=(
                *first_source.training_data,
                extra_training_data.resolve(),
            ),
        )
        baseline = queue_runner._build_training_attestation(created)
        self.assertIsNotNone(baseline)

        def changed(value):
            if isinstance(value, bool):
                return not value
            if isinstance(value, int):
                return value + 1
            if isinstance(value, str):
                return value + ".forged"
            if isinstance(value, list):
                return [*value, copy.deepcopy(value[0])]
            if value is None:
                return "forged"
            self.fail(f"unsupported attestation value type: {type(value)}")

        binding_keys = tuple(baseline["sources"][0]["plan_binding"])
        for key in binding_keys:
            with self.subTest(projection="plan_binding", key=key):
                candidate = copy.deepcopy(baseline)
                binding = candidate["sources"][0]["plan_binding"]
                binding[key] = changed(binding[key])
                self.assert_resealed_attestation_rejected(
                    created,
                    candidate,
                    pattern="plan binding drifted",
                )

        identity_keys = {
            "kind",
            "training_run_id",
            "training_seed",
            "training_run_root",
            "training_queue_id",
            "training_queue_plan_sha256",
            "training_phase",
            "diagnostic_only",
        }
        resolved_keys = tuple(baseline["sources"][0]["resolved_source"])
        for key in resolved_keys:
            with self.subTest(projection="resolved_source", key=key):
                candidate = copy.deepcopy(baseline)
                resolved = candidate["sources"][0]["resolved_source"]
                resolved[key] = changed(resolved[key])
                pattern = (
                    "resolved identity drifted"
                    if key in identity_keys
                    else "canonical 33-source replay"
                )
                self.assert_resealed_attestation_rejected(
                    created,
                    candidate,
                    pattern=pattern,
                )

        file_roles = tuple(baseline["sources"][0]["files"])
        for role in file_roles:
            with self.subTest(projection="missing_file_role", role=role):
                candidate = copy.deepcopy(baseline)
                candidate["sources"][0]["files"].pop(role)
                self.assert_resealed_attestation_rejected(
                    created,
                    candidate,
                    pattern="file-role projection is not exact",
                )

        for projection in ("files_only", "files_and_resolved"):
            with self.subTest(projection=projection):
                candidate = copy.deepcopy(baseline)
                files = candidate["sources"][0]["files"]["training_data"]
                files.reverse()
                if projection == "files_and_resolved":
                    resolved = candidate["sources"][0]["resolved_source"]
                    resolved["training_data"].reverse()
                self.assert_resealed_attestation_rejected(
                    created,
                    candidate,
                    pattern="canonical 33-source replay",
                )

        source_mutations = {
            "order": lambda sources: sources.__setitem__(
                slice(0, 2), [sources[1], sources[0]]
            ),
            "missing": lambda sources: sources.pop(),
            "extra_duplicate": lambda sources: sources.append(
                copy.deepcopy(sources[0])
            ),
        }
        for label, mutate in source_mutations.items():
            with self.subTest(projection="sources", mutation=label):
                candidate = copy.deepcopy(baseline)
                mutate(candidate["sources"])
                pattern = (
                    "plan binding drifted"
                    if label == "order"
                    else "source cardinality drifted"
                )
                self.assert_resealed_attestation_rejected(
                    created,
                    candidate,
                    pattern=pattern,
                )

    def test_success_runs_serially_and_final_verify_passes(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        completed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05
        )
        self.assertEqual(
            completed["status"], "completed", completed.get("failure")
        )
        self.assertTrue(
            all(item["status"] == "completed" for item in completed["items"])
        )
        report = queue_runner.verify_queue(fixture.queue_dir)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(len(report["verified_items"]), 33)
        self.assertFalse(Path(completed["plan"]["lease_path"]).exists())

    def test_foreign_gpu_lease_is_busy_without_poisoning_queue(self):
        fixture = self.fixture(remaining_running=False)
        created = fixture.create()
        self.ready(fixture)
        lease_path = Path(created["plan"]["lease_path"])
        _write_json(
            lease_path,
            {
                "schema": queue_runner.training_queue.LEASE_SCHEMA,
                "queue_id": "foreign-queue",
                "queue_dir": str(self.root / "foreign"),
                "plan_sha256": "f" * 64,
                "gpu_key": created["plan"]["gpu_key"],
            },
        )
        with self.assertRaises(queue_runner.MatrixQueueBusy):
            queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        observed = queue_runner.load_queue(fixture.queue_dir)
        self.assertEqual(observed["status"], "planned")
        self.assertTrue(all(item["status"] == "pending" for item in observed["items"]))
        self.assertFalse(fixture.output_root.exists())

    def test_matrix_lease_blocks_serial_queue_and_status_is_read_only(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        self.ready(fixture)
        reserved = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        lease_path = Path(reserved["plan"]["lease_path"])
        competing = {
            "plan": {
                "queue_id": "competing-serial-queue",
                "queue_dir": str(self.root / "competing"),
                "gpu_key": reserved["plan"]["gpu_key"],
                "lease_path": str(lease_path),
            },
            "plan_sha256": "9" * 64,
        }
        with self.assertRaises(queue_runner.training_queue.QueueBusyError):
            queue_runner.training_queue._ensure_lease(
                competing, {"run_id": "S0:17"}, create=True
            )
        before = (fixture.queue_dir / "queue.json").read_bytes()
        status = queue_runner.queue_status(fixture.queue_dir)
        self.assertEqual(status["lease"]["queue_id"], reserved["plan"]["queue_id"])
        self.assertEqual((fixture.queue_dir / "queue.json").read_bytes(), before)

    def test_missing_lease_between_items_is_not_recreated(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        self.ready(fixture)
        reserved = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        lease_path = Path(reserved["plan"]["lease_path"])
        queue_path = fixture.queue_dir / "queue.json"
        payload = json.loads(queue_path.read_text())
        payload["items"][0]["status"] = "completed"
        payload["items"][0]["completed_at_utc"] = queue_runner._utc_now()
        payload["status"] = "running"
        _write_json(queue_path, payload)
        lease_path.unlink()
        failed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["items"][0]["status"], "completed")
        self.assertEqual(failed["items"][1]["status"], "failed")
        self.assertIn("lost its durable GPU lease", failed["failure"]["error"])

    def test_completed_cleanup_failure_is_retryable(self):
        fixture = self.fixture(remaining_running=False)
        created = fixture.create()
        lease_path = Path(created["plan"]["lease_path"])
        with patch.object(
            queue_runner,
            "_clear_owned_gpu_lease",
            side_effect=queue_runner.MatrixQueueBusy("injected cleanup contention"),
        ), self.assertRaises(queue_runner.MatrixQueueBusy):
            queue_runner.run_queue(fixture.queue_dir, poll_seconds=0.05)
        persisted = queue_runner.load_queue(fixture.queue_dir)
        self.assertEqual(persisted["status"], "completed")
        self.assertIsNotNone(persisted["final_verification"])
        self.assertTrue(lease_path.is_file())
        completed = queue_runner.advance_once(fixture.queue_dir)
        self.assertEqual(completed["status"], "completed")
        self.assertFalse(lease_path.exists())

    def test_completed_resume_rejects_live_attestation_receipt_drift(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        completed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05
        )
        self.assertEqual(completed["status"], "completed")
        queue_path = fixture.queue_dir / "queue.json"
        payload = json.loads(queue_path.read_text())
        payload["final_verification"][
            "training_attestation_semantic_sha256"
        ] = "0" * 64
        _write_json(queue_path, payload)
        structurally_valid = queue_runner._load_queue_structural(
            fixture.queue_dir
        )
        self.assertEqual(structurally_valid["status"], "completed")
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "differs from live replay"
        ):
            queue_runner.load_queue(fixture.queue_dir)
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "differs from live replay"
        ):
            queue_runner.advance_once(fixture.queue_dir)

    def test_post_save_receipt_drift_is_detected_before_lease_release(self):
        fixture = self.fixture(remaining_running=False)
        created = fixture.create()
        lease_path = Path(created["plan"]["lease_path"])
        queue_path = fixture.queue_dir / "queue.json"
        real_save_queue = queue_runner._save_queue
        mutated = False

        def save_then_mutate_receipt(queue):
            nonlocal mutated
            real_save_queue(queue)
            if queue["status"] == "completed" and not mutated:
                persisted = json.loads(queue_path.read_text())
                persisted["final_verification"][
                    "training_attestation_semantic_sha256"
                ] = "0" * 64
                _write_json(queue_path, persisted)
                mutated = True

        with patch.object(
            queue_runner,
            "_save_queue",
            side_effect=save_then_mutate_receipt,
        ), self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "differs from live replay"
        ):
            queue_runner.run_queue(fixture.queue_dir, poll_seconds=0.05)
        persisted = json.loads(queue_path.read_text())
        self.assertTrue(mutated)
        self.assertEqual(persisted["status"], "completed")
        self.assertEqual(
            persisted["final_verification"][
                "training_attestation_semantic_sha256"
            ],
            "0" * 64,
        )
        self.assertTrue(lease_path.is_file())

    def test_active_queue_rejects_lost_lease_and_retains_failed_lease(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        self.ready(fixture)
        reserved = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        lease_path = Path(reserved["plan"]["lease_path"])
        self.assertTrue(lease_path.is_file())
        lease_path.unlink()
        failed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(failed["status"], "failed")
        self.assertIn("lost its durable GPU lease", failed["failure"]["error"])
        self.assertFalse(failed["failure"]["lease_retained_fail_closed"])

        second_root = self.root / "failure_case"
        with patch.object(
            queue_runner,
            "DEFAULT_OUTPUT_ROOT",
            second_root / "evaluations" / "matrix_validation",
        ), patch.object(
            queue_runner, "DEFAULT_QUEUE_DIR", second_root / "evaluation_queue"
        ), patch.object(
            queue_runner, "DEFAULT_TRAINING_OUTPUT_ROOT", second_root / "training"
        ):
            failing = QueueFixture(
                second_root, {"L0:17": "failed"}, remaining_running=False
            )
            with patch.object(
                queue_runner, "LOCKED_TRAINING_QUEUES", failing.bindings
            ), patch.object(
                queue_runner.training_queue,
                "load_queue",
                side_effect=failing.load_training_queue,
            ), patch.object(
                queue_runner.training_queue,
                "verify_queue",
                side_effect=failing.verify_training_queue,
            ), patch.object(
                queue_runner,
                "_resolve_formal_source",
                side_effect=failing.resolve_source,
            ):
                failing.create()
                failed_evaluation = queue_runner.run_queue(
                    failing.queue_dir, poll_seconds=0.05
                )
            retained = Path(failed_evaluation["plan"]["lease_path"])
            self.assertTrue(retained.is_file())
            self.assertTrue(
                failed_evaluation["failure"]["lease_retained_fail_closed"]
            )
            self.assertEqual(
                json.loads(retained.read_text())["queue_id"],
                failed_evaluation["plan"]["queue_id"],
            )

    def test_launched_child_is_terminated_before_lost_lease_failure(self):
        fixture = self.fixture({"L0:17": "running"}, remaining_running=False)
        fixture.create()
        self.ready(fixture)
        state = None
        for _ in range(3):
            state = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        assert state is not None
        item = state["items"][0]
        self.assertEqual(item["status"], "launched")
        pid = item["child_pid"]
        process = queue_runner._LOCAL_EVALUATION_PROCESSES[pid]
        lease_path = Path(state["plan"]["lease_path"])
        lease_path.unlink()
        try:
            failed = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        finally:
            if process.poll() is None:
                queue_runner._terminate_spawned_process(process)
            queue_runner._LOCAL_EVALUATION_PROCESSES.pop(pid, None)
        self.assertEqual(failed["status"], "failed")
        self.assertIn("lost its durable GPU lease", failed["failure"]["error"])
        self.assertEqual(
            failed["items"][0]["child_termination"]["status"], "terminated"
        )
        self.assertIsNotNone(process.poll())
        self.assertFalse(queue_runner._process_group_exists(pid))
        self.assertFalse((fixture.output_root / "L1" / "seed17").exists())

    def test_foreign_lease_ownership_loss_terminates_launched_child(self):
        fixture = self.fixture({"L0:17": "running"}, remaining_running=False)
        fixture.create()
        self.ready(fixture)
        state = None
        for _ in range(3):
            state = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        assert state is not None
        item = state["items"][0]
        pid = item["child_pid"]
        process = queue_runner._LOCAL_EVALUATION_PROCESSES[pid]
        lease_path = Path(state["plan"]["lease_path"])
        foreign = {
            "schema": queue_runner.training_queue.LEASE_SCHEMA,
            "status": "owned",
            "created_at_utc": queue_runner._utc_now(),
            "queue_id": "foreign-queue",
            "queue_dir": str(self.root / "foreign"),
            "plan_sha256": "f" * 64,
            "gpu_key": state["plan"]["gpu_key"],
            "first_run_id": "S0:17",
            "policy": "retained_across_items_until_verified_queue_completion",
        }
        _write_json(lease_path, foreign)
        try:
            failed = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        finally:
            if process.poll() is None:
                queue_runner._terminate_spawned_process(process)
            queue_runner._LOCAL_EVALUATION_PROCESSES.pop(pid, None)
        self.assertEqual(failed["status"], "failed")
        self.assertIn("lease ownership was lost", failed["failure"]["error"])
        self.assertFalse(failed["failure"]["lease_retained_fail_closed"])
        self.assertEqual(json.loads(lease_path.read_text()), foreign)
        self.assertIsNotNone(process.poll())
        self.assertFalse(queue_runner._process_group_exists(pid))

    def test_launching_orphan_binds_then_dies_on_foreign_lease_loss(self):
        fixture = self.fixture(
            {"L0:17": "spawn_descendant_running"},
            remaining_running=False,
        )
        fixture.create()
        self.ready(fixture)
        queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        launching = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(launching["items"][0]["status"], "launching")
        planned = launching["plan"]["items"][0]
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        environment.update(launching["plan"]["gpu_environment"])
        process = queue_runner.subprocess.Popen(
            list(planned["command"]),
            cwd=queue_runner.REPO_ROOT,
            env=environment,
            stdin=queue_runner.subprocess.DEVNULL,
            stdout=queue_runner.subprocess.DEVNULL,
            stderr=queue_runner.subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        pid = process.pid
        queue_runner._LOCAL_EVALUATION_PROCESSES[pid] = process
        process_tree_path = (
            fixture.output_root / "L0" / "seed17" / "process_tree.json"
        )
        deadline = time.monotonic() + 5
        while not process_tree_path.is_file():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        lease_path = Path(launching["plan"]["lease_path"])
        foreign = json.loads(lease_path.read_text())
        foreign.update(
            queue_id="foreign-queue",
            queue_dir=str(self.root / "foreign"),
            plan_sha256="f" * 64,
        )
        _write_json(lease_path, foreign)
        try:
            failed = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        finally:
            if process.poll() is None:
                queue_runner._terminate_spawned_process(process)
            queue_runner._LOCAL_EVALUATION_PROCESSES.pop(pid, None)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(
            failed["items"][0]["child_termination"]["status"],
            "terminated",
        )
        self.assertEqual(json.loads(lease_path.read_text()), foreign)
        self.assertIsNotNone(process.poll())
        self.assertFalse(queue_runner._process_group_exists(pid))

    def test_transient_lease_lock_contention_preserves_launched_child(self):
        fixture = self.fixture({"L0:17": "running"}, remaining_running=False)
        fixture.create()
        self.ready(fixture)
        state = None
        for _ in range(3):
            state = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        assert state is not None
        item = state["items"][0]
        pid = item["child_pid"]
        process = queue_runner._LOCAL_EVALUATION_PROCESSES[pid]
        lease_path = Path(state["plan"]["lease_path"])
        try:
            with patch.object(
                queue_runner.training_queue,
                "_ensure_lease",
                side_effect=queue_runner.training_queue.QueueBusyError(
                    "injected lease lock contention"
                ),
            ), self.assertRaises(queue_runner.MatrixQueueBusy):
                queue_runner.run_queue(
                    fixture.queue_dir, poll_seconds=0.05, once=True
                )
            preserved = queue_runner.load_queue(fixture.queue_dir)
            self.assertEqual(preserved["status"], "running")
            self.assertEqual(preserved["items"][0]["status"], "launched")
            self.assertTrue(lease_path.is_file())
            self.assertIsNone(process.poll())
        finally:
            if process.poll() is None:
                queue_runner._terminate_spawned_process(process)
            queue_runner._LOCAL_EVALUATION_PROCESSES.pop(pid, None)

    def test_unprovable_child_shutdown_preserves_launched_state_and_lease(self):
        fixture = self.fixture({"L0:17": "running"}, remaining_running=False)
        fixture.create()
        self.ready(fixture)
        state = None
        for _ in range(3):
            state = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        assert state is not None
        item = state["items"][0]
        pid = item["child_pid"]
        process = queue_runner._LOCAL_EVALUATION_PROCESSES[pid]
        lease_path = Path(state["plan"]["lease_path"])
        try:
            with patch.object(
                queue_runner,
                "_advance_launched",
                side_effect=queue_runner.MatrixQueueError("injected active error"),
            ), patch.object(
                queue_runner,
                "_terminate_launched_process_group",
                side_effect=queue_runner.MatrixQueueError(
                    "injected unobservable process group"
                ),
            ), self.assertRaisesRegex(
                queue_runner.MatrixQueueError, "queue remains launched"
            ):
                queue_runner.run_queue(
                    fixture.queue_dir, poll_seconds=0.05, once=True
                )
            preserved = queue_runner.load_queue(fixture.queue_dir)
            self.assertEqual(preserved["status"], "running")
            self.assertEqual(preserved["items"][0]["status"], "launched")
            self.assertNotIn("failure", preserved)
            self.assertIn(
                "injected unobservable process group",
                preserved["items"][0]["child_termination_blocked"][
                    "termination_error"
                ],
            )
            self.assertTrue(lease_path.is_file())
        finally:
            if process.poll() is None:
                queue_runner._terminate_spawned_process(process)
            queue_runner._LOCAL_EVALUATION_PROCESSES.pop(pid, None)

    def test_launch_uses_sealed_cuda_visibility(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        self.ready(fixture)
        queue_runner.run_queue(fixture.queue_dir, poll_seconds=0.05, once=True)
        queue_runner.run_queue(fixture.queue_dir, poll_seconds=0.05, once=True)
        identity = {
            "pid": 12345,
            "available": True,
            "start_time_ticks": 7,
            "boot_id": "fixture-boot",
            "state": "R",
        }
        process = Mock(pid=12345)
        with patch.object(
            queue_runner, "_matching_processes", return_value=[]
        ), patch.object(
            queue_runner.subprocess, "Popen", return_value=process
        ) as popen, patch.object(
            queue_runner.training_queue,
            "_read_process_identity",
            return_value=identity,
        ):
            launched = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        self.assertEqual(launched["items"][0]["status"], "launched")
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(
            environment["CUDA_VISIBLE_DEVICES"], launched["plan"]["gpu_key"]
        )
        self.assertEqual(
            environment["PIVOT_CUDA_VISIBLE_DEVICES"],
            launched["plan"]["gpu_key"],
        )
        queue_runner._LOCAL_EVALUATION_PROCESSES.pop(12345, None)

    def test_predeclared_spec_tamper_and_staged_recovery(self):
        fixture = self.fixture()
        created = fixture.create()
        spec_path = Path(created["aggregation_input_spec"]["path"])
        stage_dir = queue_runner._creation_stage_dir(fixture.queue_dir)
        fixture.queue_dir.rename(stage_dir)
        recovered = fixture.create()
        self.assertEqual(recovered["plan_sha256"], created["plan_sha256"])
        self.assertFalse(stage_dir.exists())

        payload = json.loads(spec_path.read_text())
        payload["evaluation_plan_sha256"] = "0" * 64
        _write_json(spec_path, payload)
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError,
            "aggregation input.*drifted|content identity drifted",
        ):
            queue_runner.load_queue(fixture.queue_dir)

    def test_empty_interrupted_creation_stage_is_rebuilt(self):
        fixture = self.fixture()
        stage_dir = queue_runner._creation_stage_dir(fixture.queue_dir)
        stage_dir.mkdir(parents=True)
        (stage_dir / ".queue.json.tmp-interrupted").write_text(
            "partial", encoding="ascii"
        )
        created = fixture.create()
        self.assertEqual(created["status"], "waiting_training")
        self.assertTrue(fixture.queue_dir.is_dir())
        self.assertFalse(stage_dir.exists())

    def test_partial_private_stage_artifact_is_rebuilt_from_plan(self):
        fixture = self.fixture()
        created = fixture.create()
        stage_dir = queue_runner._creation_stage_dir(fixture.queue_dir)
        fixture.queue_dir.rename(stage_dir)
        contract_path = stage_dir / "predeclared_contract.json"
        contract_path.write_text('{"partial":', encoding="ascii")
        recovered = fixture.create()
        self.assertEqual(recovered["plan_sha256"], created["plan_sha256"])
        contract = json.loads(
            (fixture.queue_dir / "predeclared_contract.json").read_text()
        )
        self.assertEqual(
            queue_runner._canonical_sha(contract),
            recovered["predeclared_contract_sha256"],
        )

    def test_exclusive_publication_never_exposes_partial_final_path(self):
        path = self.root / "publication" / "artifact.json"
        with patch.object(
            queue_runner,
            "_rename_noreplace",
            side_effect=OSError("injected publish failure"),
        ), self.assertRaisesRegex(OSError, "injected publish failure"):
            queue_runner._write_json_exclusive(path, {"status": "complete"})
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.iterdir()), [])

        real_fsync_directory = queue_runner._fsync_directory
        with patch.object(
            queue_runner,
            "_fsync_directory",
            wraps=real_fsync_directory,
        ) as fsync_directory:
            queue_runner._write_json_exclusive(path, {"status": "complete"})
        self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
        self.assertTrue(
            any(call.args == (path.parent,) for call in fsync_directory.call_args_list)
        )

    def test_runtime_gpu_binding_is_fail_closed(self):
        fixture = self.fixture()
        created = fixture.create()
        plan = copy.deepcopy(created["plan"])
        plan["runtime"]["cuda_visible_devices"] = "different-gpu"
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "runtime/lease GPU binding"
        ):
            queue_runner._validate_lifecycle_contract(
                created,
                plan,
                queue_runner._canonical_sha(plan),
                fixture.queue_dir,
            )
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "device must be exactly"
        ):
            queue_runner._runtime_plan(
                evaluation_python=Path(sys.executable),
                data_root=fixture.data_root,
                device="cuda:1",
                cuda_visible_devices="0",
            )

    def test_failure_and_non_matrix_profile_stop_before_second(self):
        for behavior in ("failed", "non_matrix"):
            with self.subTest(behavior=behavior):
                # Each subtest needs a distinct root because create is fresh-only.
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    with patch.object(
                        queue_runner,
                        "DEFAULT_OUTPUT_ROOT",
                        root / "evaluations" / "matrix_validation",
                    ), patch.object(
                        queue_runner,
                        "DEFAULT_QUEUE_DIR",
                        root / "evaluation_queue",
                    ), patch.object(
                        queue_runner,
                        "DEFAULT_TRAINING_OUTPUT_ROOT",
                        root / "training",
                    ):
                        fixture = QueueFixture(
                            root, {"L0:17": behavior}, remaining_running=False
                        )
                        with patch.object(
                            queue_runner,
                            "LOCKED_TRAINING_QUEUES",
                            fixture.bindings,
                        ), patch.object(
                            queue_runner.training_queue,
                            "load_queue",
                            side_effect=fixture.load_training_queue,
                        ), patch.object(
                            queue_runner.training_queue,
                            "verify_queue",
                            side_effect=fixture.verify_training_queue,
                        ), patch.object(
                            queue_runner,
                            "_resolve_formal_source",
                            side_effect=fixture.resolve_source,
                        ):
                            fixture.create()
                            failed = queue_runner.run_queue(
                                fixture.queue_dir, poll_seconds=0.05
                            )
                        self.assertEqual(failed["status"], "failed")
                        self.assertEqual(failed["items"][0]["status"], "failed")
                        self.assertEqual(failed["items"][1]["status"], "pending")

    def test_passed_but_invalid_evidence_stops_before_second(self):
        behaviors = (
            "empty_postflight",
            "corrupt_summary",
            "corrupt_records",
            "corrupt_contract",
            "corrupt_rehash",
            "missing_eval_source",
        )
        for behavior in behaviors:
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.object(
                    queue_runner,
                    "DEFAULT_OUTPUT_ROOT",
                    root / "evaluations" / "matrix_validation",
                ), patch.object(
                    queue_runner,
                    "DEFAULT_QUEUE_DIR",
                    root / "evaluation_queue",
                ), patch.object(
                    queue_runner,
                    "DEFAULT_TRAINING_OUTPUT_ROOT",
                    root / "training",
                ):
                    fixture = QueueFixture(
                        root, {"L0:17": behavior}, remaining_running=False
                    )
                    with patch.object(
                        queue_runner, "LOCKED_TRAINING_QUEUES", fixture.bindings
                    ), patch.object(
                        queue_runner.training_queue,
                        "load_queue",
                        side_effect=fixture.load_training_queue,
                    ), patch.object(
                        queue_runner.training_queue,
                        "verify_queue",
                        side_effect=fixture.verify_training_queue,
                    ), patch.object(
                        queue_runner,
                        "_resolve_formal_source",
                        side_effect=fixture.resolve_source,
                    ):
                        fixture.create()
                        failed = queue_runner.run_queue(
                            fixture.queue_dir, poll_seconds=0.05
                        )
                    self.assertEqual(failed["status"], "failed")
                    self.assertEqual(failed["items"][0]["status"], "failed")
                    self.assertEqual(failed["items"][1]["status"], "pending")
                    self.assertFalse(
                        (fixture.output_root / "L1" / "seed17").exists()
                    )

    def test_restart_recovers_running_child_without_duplicate_launch(self):
        fixture = self.fixture({"L0:17": "running"}, remaining_running=False)
        fixture.create()
        self.ready(fixture)
        for _ in range(3):
            state = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        self.assertEqual(state["items"][0]["status"], "launched")
        resumed = queue_runner.run_queue(fixture.queue_dir, poll_seconds=0.05)
        self.assertEqual(resumed["status"], "completed", resumed.get("failure"))
        attempts = fixture.output_root / "L0" / "seed17.attempts"
        self.assertEqual(attempts.read_text().splitlines(), ["L0:17"])

    def test_restart_recovers_empty_work_dir_created_before_state_save(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        self.ready(fixture)
        reserved = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(reserved["items"][0]["status"], "reserved")
        work_dir = Path(reserved["items"][0]["work_dir"])
        work_dir.mkdir(parents=True)
        recovered = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(recovered["status"], "running")
        self.assertEqual(recovered["items"][0]["status"], "launching")

    def test_launching_rechecks_evaluation_source_before_spawn(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        self.ready(fixture)
        queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        launching = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(launching["items"][0]["status"], "launching")
        with fixture.runner.open("ab") as handle:
            handle.write(b"drift-before-popen")
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError,
            "evaluation runner file identity drifted",
        ):
            queue_runner.load_queue(fixture.queue_dir)
        failed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(fixture.output_root.exists())

    def test_controller_source_drift_fails_load_launch_and_queue_verify(self):
        fixture = self.fixture(remaining_running=False)
        created = fixture.create()
        controller_record = created["plan"]["controller_sources"][0]
        controller_path = Path(controller_record["path"]).resolve()
        real_file_record = queue_runner._file_record

        def drifted_file_record(path):
            record = real_file_record(path)
            if Path(path).resolve() == controller_path:
                record["sha256"] = "0" * 64
            return record

        with patch.object(
            queue_runner,
            "_file_record",
            side_effect=drifted_file_record,
        ):
            with self.assertRaisesRegex(
                queue_runner.MatrixQueueError,
                "controller source 0 file identity drifted",
            ):
                queue_runner.load_queue(fixture.queue_dir)
            report = queue_runner.verify_queue(fixture.queue_dir)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any(
                error.get("scope") == "queue_provenance"
                and "controller source 0 file identity drifted"
                in error.get("error", "")
                for error in report["errors"]
            )
        )

        self.ready(fixture)
        with patch.object(
            queue_runner,
            "_file_record",
            side_effect=drifted_file_record,
        ):
            failed = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        self.assertEqual(failed["status"], "failed")
        self.assertIn(
            "controller source 0 file identity drifted",
            failed["failure"]["error"],
        )
        self.assertFalse(fixture.output_root.exists())

    def test_launching_rechecks_attested_training_source_before_spawn(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        self.ready(fixture)
        queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        launching = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(launching["items"][0]["status"], "launching")
        checkpoint = fixture.training_root / "L0/seed17/checkpoint_iter.pth"
        with checkpoint.open("ab") as handle:
            handle.write(b"drift-before-popen")
        failed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(fixture.output_root.exists())

    def test_unobservable_child_identity_waits_without_false_failure(self):
        fixture = self.fixture({"L0:17": "running"}, remaining_running=False)
        fixture.create()
        self.ready(fixture)
        for _ in range(3):
            launched = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        item = launched["items"][0]
        self.assertEqual(item["status"], "launched")
        identity = item["child_process_identity"]
        self.assertEqual(identity["pid"], item["child_pid"])
        self.assertTrue(identity["available"])
        self.assertIsInstance(identity["start_time_ticks"], int)
        self.assertTrue(identity["boot_id"])
        process = queue_runner._LOCAL_EVALUATION_PROCESSES[item["child_pid"]]
        with patch.object(
            queue_runner.training_queue, "_process_running", return_value=None
        ):
            waiting = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        self.assertEqual(waiting["status"], "running")
        self.assertEqual(waiting["items"][0]["status"], "launched")
        self.assertIsNone(waiting["items"][0]["last_observation"]["pid_running"])
        process.wait(timeout=5)
        completed_first = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(completed_first["items"][0]["status"], "completed")
        self.assertEqual(completed_first["items"][1]["status"], "pending")

    def test_reused_pid_identity_does_not_count_as_live_child(self):
        fixture = self.fixture({"L0:17": "running"}, remaining_running=False)
        fixture.create()
        self.ready(fixture)
        for _ in range(3):
            launched = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        item = launched["items"][0]
        process = queue_runner._LOCAL_EVALUATION_PROCESSES[item["child_pid"]]
        reused_identity = dict(item["child_process_identity"])
        reused_identity["start_time_ticks"] += 1
        reused_identity["state"] = "R"
        with patch.object(
            queue_runner.training_queue, "_process_running", return_value=True
        ), patch.object(
            queue_runner.training_queue,
            "_read_process_identity",
            return_value=reused_identity,
        ):
            failed = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["items"][0]["status"], "failed")
        process.wait(timeout=5)
        queue_runner._LOCAL_EVALUATION_PROCESSES.pop(item["child_pid"], None)

    def test_bind_rejects_unavailable_child_identity(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        self.ready(fixture)
        queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        launching = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "lacks exact PID/start-time/boot binding"
        ):
            queue_runner._bind_process(
                launching,
                0,
                12345,
                {"pid": 12345, "available": False},
            )
        persisted = queue_runner.load_queue(fixture.queue_dir)
        self.assertEqual(persisted["items"][0]["status"], "launching")

    def test_spawn_to_bind_failure_terminates_and_reaps_process_group(self):
        fixture = self.fixture(
            {"L0:17": "spawn_descendant_running"}, remaining_running=False
        )
        fixture.create()
        self.ready(fixture)
        queue_runner.run_queue(fixture.queue_dir, poll_seconds=0.05, once=True)
        launching = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05, once=True
        )
        self.assertEqual(launching["items"][0]["status"], "launching")
        process_tree_path = fixture.output_root / "L0" / "seed17" / "process_tree.json"
        real_read_identity = queue_runner.training_queue._read_process_identity

        def wait_for_descendant(pid):
            deadline = time.monotonic() + 5
            while not process_tree_path.is_file():
                if time.monotonic() >= deadline:
                    self.fail("fake evaluator did not publish its descendant PID")
                time.sleep(0.01)
            return real_read_identity(pid)

        with patch.object(
            queue_runner.training_queue,
            "_read_process_identity",
            side_effect=wait_for_descendant,
        ), patch.object(
            queue_runner,
            "_bind_process",
            side_effect=queue_runner.MatrixQueueError("injected bind failure"),
        ):
            failed = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        process_tree = json.loads(process_tree_path.read_text())
        leader_pid = process_tree["leader_pid"]
        descendant_pid = process_tree["descendant_pid"]
        try:
            self.assertEqual(failed["status"], "failed")
            self.assertIn("injected bind failure", failed["failure"]["error"])
            self.assertFalse(
                queue_runner.training_queue._process_running(leader_pid)
            )
            self.assertFalse(
                queue_runner.training_queue._process_running(descendant_pid)
            )
            self.assertFalse(queue_runner._process_group_exists(leader_pid))
        finally:
            if queue_runner._process_group_exists(leader_pid) is True:
                os.killpg(leader_pid, queue_runner.signal.SIGKILL)

    def test_waiting_training_interrupt_is_not_persisted_as_failure(self):
        fixture = self.fixture()
        fixture.create()
        with patch.object(
            queue_runner,
            "_advance_training_gate",
            side_effect=KeyboardInterrupt,
        ), self.assertRaises(KeyboardInterrupt):
            queue_runner.advance_once(fixture.queue_dir)
        state = queue_runner.load_queue(fixture.queue_dir)
        self.assertEqual(state["status"], "waiting_training")
        self.assertTrue(all(item["status"] == "pending" for item in state["items"]))

    def test_training_tamper_and_runner_drift_fail_before_launch(self):
        for mutation in ("training", "runner"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.object(
                    queue_runner,
                    "DEFAULT_OUTPUT_ROOT",
                    root / "evaluations" / "matrix_validation",
                ), patch.object(
                    queue_runner,
                    "DEFAULT_QUEUE_DIR",
                    root / "evaluation_queue",
                ), patch.object(
                    queue_runner,
                    "DEFAULT_TRAINING_OUTPUT_ROOT",
                    root / "training",
                ):
                    fixture = QueueFixture(root, remaining_running=False)
                    with patch.object(
                        queue_runner, "LOCKED_TRAINING_QUEUES", fixture.bindings
                    ), patch.object(
                        queue_runner.training_queue,
                        "load_queue",
                        side_effect=fixture.load_training_queue,
                    ), patch.object(
                        queue_runner.training_queue,
                        "verify_queue",
                        side_effect=fixture.verify_training_queue,
                    ), patch.object(
                        queue_runner,
                        "_resolve_formal_source",
                        side_effect=fixture.resolve_source,
                    ):
                        fixture.create()
                        fixture.complete_training()
                        queue_runner.run_queue(
                            fixture.queue_dir, poll_seconds=0.05, once=True
                        )
                        target = (
                            fixture.training_root / "L0/seed17/checkpoint_iter.pth"
                            if mutation == "training"
                            else fixture.runner
                        )
                        with target.open("ab") as handle:
                            handle.write(b"drift")
                        failed = queue_runner.run_queue(
                            fixture.queue_dir, poll_seconds=0.05, once=True
                        )
                    self.assertEqual(failed["status"], "failed")
                    self.assertFalse(fixture.output_root.exists())

    def test_duplicate_output_is_rejected_at_create(self):
        fixture = self.fixture()
        duplicate = fixture.output_root / "L0/seed17"
        duplicate.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            fixture.create()

    def test_alternate_queue_directory_is_rejected_at_create(self):
        fixture = self.fixture()
        alternate = self.root / "alternate_evaluation_queue"
        with self.assertRaisesRegex(
            queue_runner.MatrixQueueError, "queue directory must be canonical"
        ):
            fixture.create(alternate)
        self.assertFalse(alternate.exists())

    def test_final_verify_rejects_postflight_tamper(self):
        fixture = self.fixture(remaining_running=False)
        fixture.create()
        completed = queue_runner.run_queue(
            fixture.queue_dir, poll_seconds=0.05
        )
        self.assertEqual(completed["status"], "completed")
        postflight = fixture.output_root / "L0/seed17/postflight.json"
        payload = json.loads(postflight.read_text())
        payload["profile"] = "screen_validation"
        _write_json(postflight, payload)
        report = queue_runner.verify_queue(fixture.queue_dir)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any(
                "L0:17" == error.get("run_id")
                for error in report["errors"]
            )
        )

    def test_final_provenance_drift_fails_before_lease_release(self):
        fixture = self.fixture(remaining_running=False)
        created = fixture.create()
        with patch.object(
            queue_runner, "_advance_final_verification", side_effect=KeyboardInterrupt
        ), self.assertRaises(KeyboardInterrupt):
            queue_runner.run_queue(fixture.queue_dir, poll_seconds=0.05)
        verifying = queue_runner.load_queue(fixture.queue_dir)
        self.assertEqual(verifying["status"], "verifying")
        self.assertTrue(
            all(item["status"] == "completed" for item in verifying["items"])
        )
        lease_path = Path(created["plan"]["lease_path"])
        self.assertTrue(lease_path.is_file())

        controller_path = Path(
            created["plan"]["controller_sources"][0]["path"]
        ).resolve()
        real_file_record = queue_runner._file_record

        def drifted_file_record(path):
            record = real_file_record(path)
            if Path(path).resolve() == controller_path:
                record["sha256"] = "0" * 64
            return record

        with patch.object(
            queue_runner, "_file_record", side_effect=drifted_file_record
        ):
            failed = queue_runner.run_queue(
                fixture.queue_dir, poll_seconds=0.05, once=True
            )
        self.assertEqual(failed["status"], "failed")
        self.assertIn("controller source 0 file identity drifted", failed["failure"]["error"])
        self.assertTrue(failed["failure"]["lease_retained_fail_closed"])
        self.assertTrue(lease_path.is_file())
        self.assertIsNone(failed["final_verification"])

    def test_detach_is_idempotent_and_status_is_read_only(self):
        fixture = self.fixture()
        fixture.create()
        process = SimpleNamespace(pid=12345)
        before = (fixture.queue_dir / "queue.json").read_bytes()
        with patch.object(
            queue_runner.subprocess, "Popen", return_value=process
        ), patch.object(
            queue_runner.training_queue,
            "_read_process_identity",
            return_value={"available": True, "start_time_ticks": 1},
        ), patch.object(
            queue_runner.training_queue,
            "_process_running",
            return_value=True,
        ):
            first = queue_runner.detach_queue(fixture.queue_dir, poll_seconds=0.05)
            second = queue_runner.detach_queue(fixture.queue_dir, poll_seconds=0.05)
        self.assertEqual(first["status"], "launched")
        self.assertEqual(second["status"], "already_running")
        queue_runner.queue_status(fixture.queue_dir)
        self.assertEqual((fixture.queue_dir / "queue.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
