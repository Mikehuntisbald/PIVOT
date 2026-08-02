import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools import run_stageb_paper_evaluations as paper_eval
from tools import run_stageb_table_a_evaluations as runner
from tools import stageb_headline_release_contract as headline_release


def _source(root: Path) -> paper_eval.EvaluationSource:
    config = root / "config.py"
    checkpoint = root / "checkpoint.pth"
    config.write_text("stage_b = True\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    return paper_eval.EvaluationSource(
        kind="pivot_paper_training_run",
        evaluation_id="L4_seed17",
        config=config.resolve(),
        checkpoint=checkpoint.resolve(),
        checkpoint_sha256=runner.HashCache().digest(checkpoint),
        training_run_id="L4:17",
        training_seed=17,
    )


def _g0c_source(root: Path, *, seed: int = 17) -> runner.G0cEvaluationSource:
    config = root / "g0c_config.py"
    training_root = root / "g0c_training"
    training_root.mkdir(exist_ok=True)
    checkpoint = training_root / "checkpoint_iter.pth"
    training_plan = root / "g0c_training_plan.json"
    training_postflight = training_root / "postflight.json"
    queue_root = root / "g0c_training_queue"
    job_root = queue_root / "jobs" / f"{seed}-G0c" / "job"
    job_root.mkdir(parents=True, exist_ok=True)
    queue_manifest = queue_root / "queue.json"
    job_launch = job_root / "launch.json"
    job_status = job_root / "status.json"
    config.write_text("stage_b = False\n", encoding="utf-8")
    checkpoint.write_bytes(b"trained-g0c-checkpoint")
    training_plan.write_text("{}\n", encoding="utf-8")
    training_postflight.write_text("{}\n", encoding="utf-8")
    queue_manifest.write_text("{}\n", encoding="utf-8")
    job_launch.write_text("{}\n", encoding="utf-8")
    job_status.write_text("{}\n", encoding="utf-8")
    return runner.G0cEvaluationSource(
        kind=runner.G0C_SOURCE_KIND,
        evaluation_id=f"G0c_seed{seed}",
        config=config.resolve(),
        checkpoint=checkpoint.resolve(),
        checkpoint_sha256=runner.HashCache().digest(checkpoint),
        training_run_id=f"G0c:{seed}",
        training_seed=seed,
        training_run_root=training_root.resolve(),
        training_phase="final",
        diagnostic_only=False,
        final_phase_id="formal",
        training_postflight=training_postflight.resolve(),
        selected_phase_id="formal",
        selected_training_postflight=training_postflight.resolve(),
        training_plan=training_plan.resolve(),
        training_plan_contract_sha256="b" * 64,
        source_dependency_tree_sha256="c" * 64,
        source_provenance_dependencies=(config.resolve(),),
        training_queue_manifest=queue_manifest.resolve(),
        training_queue_detached_launch=job_launch.resolve(),
        training_queue_detached_status=job_status.resolve(),
        training_queue_id="g0c-training-queue",
        training_queue_plan_sha256="d" * 64,
    )


def _runtime(root: Path) -> runner.Runtime:
    return runner.Runtime(
        python=Path(sys.executable).resolve(),
        data_root=root.resolve(),
        device="cuda:0",
        batch_size=16,
        num_workers=8,
        amp=True,
    )


def _gate_fixture(root: Path):
    runtime = {
        **runner._jsonable(runner.asdict(_runtime(root))),
        "eval_seed": runner.EVAL_SEED,
    }
    runtime_contract = runner._formal_runtime_contract(runtime)

    def canonical(kind, profile, seed):
        row = "L4" if kind == "candidate" else "G0c"
        return root / profile / kind / row / f"seed{seed}"

    prerequisites = {}
    for name in runner.RELEASE_PREREQUISITE_NAMES:
        path = root / f"{name}.json"
        path.write_text(json.dumps({"status": "sealed"}), encoding="utf-8")
        prerequisites[name] = {
            "path": str(path.resolve()),
            "sha256": runner.HashCache().digest(path),
            "size_bytes": path.stat().st_size,
        }
    instances = []
    provenance = []
    for kind in ("candidate", "g0c"):
        for seed in runner.FORMAL_SEEDS:
            validation_root = canonical(kind, runner.VALIDATION_PROFILE, seed)
            validation_root.mkdir(parents=True)
            validation_instance = {
                "seed": seed,
                "instance_id": f"table_a:validation:{kind}:{seed}",
                "instance_sha256": f"{seed + (0 if kind == 'candidate' else 100):064x}",
            }
            source = {
                "training_run_id": (
                    f"L4:{seed}" if kind == "candidate" else f"G0c:{seed}"
                ),
                "checkpoint_sha256": f"{seed + 1000:064x}",
                "training_queue_id": f"{kind}-queue",
                "training_queue_plan_sha256": "a" * 64,
            }
            launch_path = validation_root / "launch_manifest.json"
            launch_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "kind": kind,
                        "profile": runner.VALIDATION_PROFILE,
                        "evaluation_id": f"{kind}_seed{seed}",
                        "runtime": runtime,
                        "source": source,
                        "instance": validation_instance,
                    }
                ),
                encoding="utf-8",
            )
            postflight_path = validation_root / "postflight.json"
            postflight_path.write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            final_instance = runner._instance_payload(
                kind=kind,
                profile=runner.FINAL_PROFILE,
                seed=seed,
                output_dir=canonical(kind, runner.FINAL_PROFILE, seed),
                source=SimpleNamespace(
                    evaluation_id=f"{kind}_seed{seed}",
                    **source,
                ),
                runtime=runtime,
            )
            instances.append(final_instance)
            provenance.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "validation_instance_id": validation_instance["instance_id"],
                    "validation_instance_sha256": validation_instance[
                        "instance_sha256"
                    ],
                    "validation_launch": runner._file_record(
                        launch_path,
                        runner.HashCache(),
                        "table_a_validation_launch",
                    ),
                    "validation_postflight": runner._file_record(
                        postflight_path,
                        runner.HashCache(),
                        "table_a_validation_postflight",
                    ),
                }
            )
    gate = {
        "schema": runner.FINAL_GATE_SCHEMA,
        "status": "sealed",
        "selection_frozen": True,
        "all_paper_ablations_completed": True,
        "created_before_first_final_evaluation": True,
        "selection_rule": (
            "promote_all_predeclared_candidate_and_g0c_seeds_from_completed_"
            "validation_without_test_metric_selection"
        ),
        "instances": instances,
        "runtime_contract": runtime_contract,
        "release_prerequisites": prerequisites,
        "validation_provenance": provenance,
    }
    gate["gate_sha256"] = runner._canonical_json_sha256(gate)
    return gate, instances, prerequisites, canonical


def _g0c_resolver_fixture(root: Path, *, seed: int = 17):
    cache = runner.HashCache()
    config = root / "formal_g0c_config.py"
    dataset = root / "formal_g0c_dataset.py"
    config.write_text("stage_b = False\n", encoding="utf-8")
    dataset.write_text("datasets = []\n", encoding="utf-8")
    training_jsonls = []
    for index in range(4):
        path = root / f"training_{index}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        training_jsonls.append(path)
    controller = Path(runner.g0c_controls.__file__).resolve()
    dependency_paths = sorted(
        (controller, config.resolve()),
        key=runner.g0c_controls._dependency_label,
    )
    digest = runner.hashlib.sha256()
    dependency_records = []
    for path in dependency_paths:
        relative = runner.g0c_controls._dependency_label(path)
        sha256 = cache.digest(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
        dependency_records.append(
            {
                "path": str(path),
                "relative_path": relative,
                "sha256": sha256,
                "size_bytes": path.stat().st_size,
                "dependency_kind": "python",
            }
        )
    source_tree = {
        "algorithm": "recursive-python-plus-native-runtime-closure-v2",
        "file_count": len(dependency_records),
        "python_file_count": len(dependency_records),
        "native_file_count": 0,
        "sha256": digest.hexdigest(),
        "records": dependency_records,
    }
    output_root = root / "formal_g0c_training"
    output_root.mkdir()
    checkpoint = output_root / "checkpoint_iter.pth"
    checkpoint.write_bytes(b"formal-trained-g0c")
    checkpoint_record = {
        "path": str(checkpoint.resolve()),
        "sha256": cache.digest(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
    }
    inputs = {
        "config": {
            "path": str(config.resolve()),
            "sha256": cache.digest(config),
        },
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256": cache.digest(dataset),
        },
    }
    for index, path in enumerate(training_jsonls):
        inputs[f"training_jsonl_{index}"] = {
            "path": str(path.resolve()),
            "sha256": cache.digest(path),
        }
    plan = {
        "schema": runner.g0c_controls.PLAN_SCHEMA,
        "row_id": "G0c",
        "purpose": "formal",
        "matched_contract": {"seed": seed},
        "output_dir": str(output_root.resolve()),
        "inputs": inputs,
        "source_dependency_tree": source_tree,
    }
    plan["plan_sha256"] = runner.g0c_controls._plan_sha256(plan)
    plan_path = root / "formal_g0c_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    postflight = {
        "schema": runner.g0c_controls.POSTFLIGHT_SCHEMA,
        "status": "PASS",
        "row_id": "G0c",
        "purpose": "formal",
        "plan_sha256": plan["plan_sha256"],
        "checkpoint": checkpoint_record,
        "source_dependency_tree_sha256": source_tree["sha256"],
        "validated_at_utc": "2026-07-19T00:00:00+00:00",
    }
    postflight_path = output_root / "postflight.json"
    postflight_path.write_text(json.dumps(postflight), encoding="utf-8")
    return {
        "config": config,
        "output_root": output_root,
        "checkpoint": checkpoint,
        "plan": plan,
        "plan_path": plan_path,
        "postflight": postflight,
        "postflight_path": postflight_path,
        "source_tree": source_tree,
        "dependency_paths": tuple(dependency_paths),
        "training_data": (dataset, *training_jsonls),
    }


class TableAEvaluationPlanTest(unittest.TestCase):
    def test_list_exposes_complete_table_a_contract(self):
        with mock.patch("builtins.print") as output:
            self.assertEqual(runner.main(["list"]), 0)
        value = json.loads(output.call_args.args[0])
        self.assertEqual(value["candidate_rows"], ["G1", "G2", "G3", "G4", "G5"])
        self.assertEqual(value["g0c_topks"], [1, 5, 10, 50, "all"])
        self.assertEqual(value["category_pairs"], 512)

    def test_candidate_source_is_exact_l4_seed_and_locked_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = (
                root
                / "outputs/paper_cvpr_v1/token_ablation_frozen_v2/L4/seed17"
            )
            queue_root = root / "queue"
            run_root.mkdir(parents=True)
            queue_root.mkdir()
            queue_manifest = queue_root / "queue.json"
            queue_manifest.write_text("{}\n", encoding="utf-8")
            base = _source(root)
            source = paper_eval.EvaluationSource(
                **{
                    **base.__dict__,
                    "training_run_root": run_root.resolve(),
                    "training_queue_manifest": queue_manifest.resolve(),
                    "training_queue_id": "queue-17",
                    "training_queue_plan_sha256": "a" * 64,
                }
            )
            locked = {
                17: {
                    "path": str(queue_root.resolve()),
                    "queue_id": "queue-17",
                    "plan_sha256": "a" * 64,
                }
            }
            with (
                mock.patch.object(runner, "REPO_ROOT", root),
                mock.patch.object(runner, "LOCKED_CANDIDATE_QUEUES", locked),
            ):
                self.assertEqual(
                    runner._validate_candidate_source(
                        source,
                        training_run_root=run_root,
                        training_queue_dir=queue_root,
                    ),
                    17,
                )
                wrong = paper_eval.EvaluationSource(
                    **{**source.__dict__, "training_run_id": "L0:17"}
                )
                with self.assertRaisesRegex(
                    runner.TableAEvaluationError, "locked formal L4"
                ):
                    runner._validate_candidate_source(
                        wrong,
                        training_run_root=run_root,
                        training_queue_dir=queue_root,
                    )

    def test_final_gate_is_self_bound_and_uniquely_authorizes_instance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "gate.json"
            gate, instances, prerequisites, canonical = _gate_fixture(root)
            instance = instances[0]
            path.write_text(json.dumps(gate), encoding="utf-8")
            with (
                mock.patch.object(runner, "FINAL_GATE_PATH", path),
                mock.patch.object(
                    runner, "canonical_output_dir", side_effect=canonical
                ),
                mock.patch.object(
                    runner,
                    "_release_prerequisite_records",
                    return_value=prerequisites,
                ),
            ):
                receipt = runner._validate_final_gate(path, instance)
                self.assertEqual(receipt["payload"], gate)
                with self.assertRaisesRegex(
                    runner.TableAEvaluationError, "runtime contract"
                ):
                    runner._validate_final_gate(
                        path, {**instance, "runtime_contract": {"contract": "drift"}}
                    )
                gate["instances"][1] = instance
                gate["gate_sha256"] = runner._canonical_json_sha256(
                    {key: value for key, value in gate.items() if key != "gate_sha256"}
                )
                path.write_text(json.dumps(gate), encoding="utf-8")
                with self.assertRaisesRegex(
                    runner.TableAEvaluationError, "order/cardinality|unique"
                ):
                    runner._validate_final_gate(path, instance)

    def test_gate_v2_replays_exact_global_release_prerequisites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection.json"
            completion = root / "completion.json"
            selection.write_text("{}\n", encoding="ascii")
            completion.write_text("{}\n", encoding="ascii")
            completed = {
                "status": "completed",
                "completed_before_final_gate": True,
                "all_training_validation_diagnostics_completed": True,
            }
            with (
                mock.patch.object(
                    headline_release, "SELECTION_RECEIPT_PATH", selection
                ),
                mock.patch.object(
                    headline_release,
                    "PAPER_ABLATION_COMPLETION_RECEIPT_PATH",
                    completion,
                ),
                mock.patch.object(
                    headline_release,
                    "validate_selection_receipt",
                    return_value={"status": "eligible"},
                ) as validate_selection,
                mock.patch.object(
                    headline_release,
                    "validate_paper_ablation_completion_receipt",
                    return_value=completed,
                ) as validate_completion,
            ):
                records = runner._release_prerequisite_records(
                    replay_selection=True
                )
            validate_selection.assert_called_once_with(
                selection, replay_validation=True
            )
            validate_completion.assert_called_once_with(completion)
            self.assertEqual(set(records), set(runner.RELEASE_PREREQUISITE_NAMES))

            with (
                mock.patch.object(
                    headline_release, "SELECTION_RECEIPT_PATH", selection
                ),
                mock.patch.object(
                    headline_release,
                    "PAPER_ABLATION_COMPLETION_RECEIPT_PATH",
                    completion,
                ),
                mock.patch.object(
                    headline_release,
                    "validate_selection_receipt",
                    return_value={"status": "ineligible"},
                ),
                mock.patch.object(
                    headline_release,
                    "validate_paper_ablation_completion_receipt",
                    return_value=completed,
                ),
                self.assertRaisesRegex(
                    runner.TableAEvaluationError, "selection is ineligible"
                ),
            ):
                runner._release_prerequisite_records(replay_selection=False)

    def test_final_plan_input_closure_includes_gate_and_both_global_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path = root / "gate.json"
            gate, instances, prerequisites, canonical = _gate_fixture(root)
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            records = {}
            with (
                mock.patch.object(runner, "FINAL_GATE_PATH", gate_path),
                mock.patch.object(
                    runner, "FINAL_CONSUMPTION_ROOT", root / "consumptions"
                ),
                mock.patch.object(
                    runner, "canonical_output_dir", side_effect=canonical
                ),
                mock.patch.object(
                    runner,
                    "_release_prerequisite_records",
                    return_value=prerequisites,
                ),
            ):
                binding = runner._bind_final_gate(
                    profile=runner.FINAL_PROFILE,
                    final_gate=gate_path,
                    instance=instances[0],
                    cache=runner.HashCache(),
                    records=records,
                )
            self.assertEqual(binding["path"], str(gate_path.resolve()))
            expected_paths = {
                str(gate_path.resolve()),
                *(record["path"] for record in prerequisites.values()),
            }
            self.assertEqual(set(records), expected_paths)
            self.assertEqual(
                set().union(*(set(record["roles"]) for record in records.values())),
                {
                    "table_a_final_evaluation_gate",
                    "table_a_final_headline_selection_receipt",
                    "table_a_final_paper_ablation_completion_receipt",
                },
            )

    def test_seal_final_gate_promotes_exact_six_validation_instances_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path = root / "gate.json"

            def canonical(kind, profile, seed):
                return root / profile / kind / f"seed{seed}"

            launches = {}
            for kind in ("candidate", "g0c"):
                for seed in runner.FORMAL_SEEDS:
                    validation_root = canonical(kind, "validation", seed)
                    validation_root.mkdir(parents=True)
                    (validation_root / "postflight.json").write_text(
                        json.dumps({"status": "passed"}), encoding="utf-8"
                    )
                    launches[str(validation_root.resolve())] = {
                        "schema": runner.SCHEMA,
                        "status": "completed",
                        "kind": kind,
                        "profile": "validation",
                        "evaluation_id": f"{kind}_{seed}",
                        "instance": {
                            "seed": seed,
                            "instance_id": f"validation:{kind}:{seed}",
                            "instance_sha256": f"{seed:064x}",
                        },
                        "runtime": {
                            "python": str(Path(sys.executable).resolve()),
                            "data_root": str(root.resolve()),
                            "device": "cuda:0",
                            "batch_size": 16,
                            "num_workers": 8,
                            "amp": True,
                            "eval_seed": 42,
                        },
                        "source": {
                            "training_run_id": (
                                f"L4:{seed}" if kind == "candidate" else None
                            ),
                            "checkpoint_sha256": f"{seed + 1:064x}",
                            "training_queue_id": (
                                "queue" if kind == "candidate" else None
                            ),
                            "training_queue_plan_sha256": (
                                "a" * 64 if kind == "candidate" else None
                            ),
                        },
                    }

            def load_launch(path):
                return launches[str(Path(path).resolve())]

            with (
                mock.patch.object(runner, "FINAL_GATE_PATH", gate_path),
                mock.patch.object(
                    runner, "canonical_output_dir", side_effect=canonical
                ),
                mock.patch.object(runner, "_load_launch", side_effect=load_launch),
                mock.patch.object(
                    runner, "postflight", return_value={"status": "passed"}
                ),
                mock.patch.object(
                    runner,
                    "_release_prerequisite_records",
                    return_value={
                        name: {
                            "path": str((root / f"{name}.json").resolve()),
                            "sha256": "a" * 64,
                            "size_bytes": 1,
                        }
                        for name in runner.RELEASE_PREREQUISITE_NAMES
                    },
                ),
            ):
                for name in runner.RELEASE_PREREQUISITE_NAMES:
                    (root / f"{name}.json").write_text("x", encoding="ascii")
                for validation_root, launch in (
                    (Path(path), value) for path, value in launches.items()
                ):
                    (validation_root / "launch_manifest.json").write_text(
                        json.dumps(launch), encoding="utf-8"
                    )
                first_postflight = (
                    canonical("candidate", "validation", 17) / "postflight.json"
                )
                first_postflight.write_text(
                    json.dumps({"status": "passed", "tampered": True}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    runner.TableAEvaluationError, "differs from fresh replay"
                ):
                    runner.seal_final_gate(gate_path)
                first_postflight.write_text(
                    json.dumps({"status": "passed"}), encoding="utf-8"
                )
                gate = runner.seal_final_gate(gate_path)
                self.assertEqual(len(gate["instances"]), 6)
                self.assertTrue(gate["selection_frozen"])
                self.assertTrue(gate["created_before_first_final_evaluation"])
                with self.assertRaises(FileExistsError):
                    runner.seal_final_gate(gate_path)

    def test_final_gate_consumption_is_single_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path = root / "gate.json"
            gate, instances, prerequisites, canonical = _gate_fixture(root)
            instance = instances[0]
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            plan = {
                "instance": instance,
                "final_gate": {
                    "path": str(gate_path),
                    "sha256": runner.HashCache().digest(gate_path),
                },
                "output_dir": str(root / "output"),
            }
            with (
                mock.patch.object(runner, "FINAL_GATE_PATH", gate_path),
                mock.patch.object(
                    runner, "FINAL_CONSUMPTION_ROOT", root / "consumptions"
                ),
                mock.patch.object(
                    runner, "canonical_output_dir", side_effect=canonical
                ),
                mock.patch.object(
                    runner,
                    "_release_prerequisite_records",
                    return_value=prerequisites,
                ),
            ):
                receipt = runner._consume_final_gate(plan)
                plan["final_consumption"] = receipt
                runner._validate_final_consumption(plan)
                with self.assertRaisesRegex(
                    runner.TableAEvaluationError, "already consumed"
                ):
                    runner._consume_final_gate(plan)

    def test_final_gate_rejects_release_or_validation_identity_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path = root / "gate.json"
            gate, instances, prerequisites, canonical = _gate_fixture(root)
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            with (
                mock.patch.object(runner, "FINAL_GATE_PATH", gate_path),
                mock.patch.object(
                    runner, "canonical_output_dir", side_effect=canonical
                ),
                mock.patch.object(
                    runner,
                    "_release_prerequisite_records",
                    return_value=prerequisites,
                ),
            ):
                runner._validate_final_gate(gate_path, instances[0])
                Path(
                    gate["validation_provenance"][0]["validation_postflight"][
                        "path"
                    ]
                ).write_text(json.dumps({"status": "changed"}), encoding="utf-8")
                with self.assertRaisesRegex(
                    runner.TableAEvaluationError, "changed after gate sealing"
                ):
                    runner._validate_final_gate(gate_path, instances[0])

            changed = {
                **prerequisites,
                "headline_selection_receipt": {
                    **prerequisites["headline_selection_receipt"],
                    "sha256": "0" * 64,
                },
            }
            with (
                mock.patch.object(runner, "FINAL_GATE_PATH", gate_path),
                mock.patch.object(
                    runner, "canonical_output_dir", side_effect=canonical
                ),
                mock.patch.object(
                    runner,
                    "_release_prerequisite_records",
                    return_value=changed,
                ),
                self.assertRaisesRegex(
                    runner.TableAEvaluationError, "release prerequisites"
                ),
            ):
                runner._validate_final_gate(gate_path, instances[0])

    def test_atomic_gate_publication_never_replaces_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            destination = root / "gate.json"
            source.write_text("new\n", encoding="ascii")
            destination.write_text("original\n", encoding="ascii")
            with self.assertRaises(FileExistsError):
                runner._rename_noreplace(source, destination)
            self.assertEqual(destination.read_text(encoding="ascii"), "original\n")
            self.assertEqual(source.read_text(encoding="ascii"), "new\n")

    def test_candidate_plan_has_one_full_causal_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_source = _source(root)
            queue_dir = root / "queue"
            job_dir = queue_dir / "jobs" / "001-L4_17" / "job"
            job_dir.mkdir(parents=True)
            queue_manifest = queue_dir / "queue.json"
            detached_launch = job_dir / "launch.json"
            detached_status = job_dir / "status.json"
            queue_manifest.write_text("{}\n", encoding="utf-8")
            detached_launch.write_text("{}\n", encoding="utf-8")
            detached_status.write_text("{}\n", encoding="utf-8")
            queue_bindings = (
                (
                    str(queue_manifest.resolve()),
                    "training_queue_manifest",
                    runner.HashCache().digest(queue_manifest),
                ),
                (
                    str(detached_launch.resolve()),
                    "training_queue_detached_launch",
                    runner.HashCache().digest(detached_launch),
                ),
                (
                    str(detached_status.resolve()),
                    "training_queue_detached_status",
                    runner.HashCache().digest(detached_status),
                ),
            )
            source = paper_eval.EvaluationSource(
                **{
                    **base_source.__dict__,
                    "training_queue_manifest": queue_manifest.resolve(),
                    "training_queue_detached_launch": detached_launch.resolve(),
                    "training_queue_detached_status": detached_status.resolve(),
                    "training_queue_id": "queue-id",
                    "training_queue_plan_sha256": "a" * 64,
                }
            )
            strict = root / "strict.jsonl"
            strict.write_text("{}\n", encoding="utf-8")
            strict_record = runner._file_record(
                strict, runner.HashCache(), "strict2031"
            )
            strict_record.update({"rows": 2031, "source_counts": {}})
            contract = {
                "candidate_topk": 50,
                "all_query_count": 900,
                "required_rows": ["G1", "G2", "G3", "G4", "G5"],
                "required_edit_taxonomies": list(runner.REQUIRED_EDIT_TAXONOMIES),
                "category_pairs": 512,
                "category_arms": 1024,
            }
            with (
                mock.patch.object(
                    paper_eval, "_resolve_pivot_source", return_value=source
                ) as resolve_source,
                mock.patch.object(runner, "_candidate_contract", return_value=contract),
                mock.patch.object(runner, "_validate_candidate_source", return_value=17),
                mock.patch.object(
                    runner,
                    "canonical_output_dir",
                    return_value=root / "output",
                ),
                mock.patch.object(runner, "_category_rows", return_value=[]),
                mock.patch.object(runner, "_category_asset_paths", return_value=[]),
                mock.patch.object(runner, "_base_data_and_code_records"),
                mock.patch.object(
                    runner,
                    "_profile_surface",
                    return_value=(
                        runner.VALIDATION_REF_SPLITS,
                        strict_record,
                        {"screen_calibration_audit": strict_record},
                    ),
                ),
                mock.patch.object(
                    runner,
                    "_strict_taxonomy_counts",
                    return_value={
                        "color": 2027,
                        "size": 1,
                        "action": 1,
                        "spatial": 1,
                        "relation": 1,
                    },
                ),
            ):
                plan = runner.build_candidate_plan(
                    _runtime(root),
                    root / "training",
                    root / "output",
                    training_queue_dir=queue_dir,
                )
        resolve_source.assert_called_once()
        self.assertEqual(resolve_source.call_args.args[0], root / "training")
        self.assertIsInstance(resolve_source.call_args.args[1], paper_eval.HashCache)
        self.assertEqual(
            resolve_source.call_args.kwargs, {"training_queue_dir": queue_dir}
        )
        self.assertEqual(plan["kind"], "candidate")
        self.assertEqual(len(plan["commands"]), 1)
        command = plan["commands"][0]["command"]
        self.assertIn("--true_role_swap", command)
        self.assertIn("--formal_table_a", command)
        self.assertEqual(
            command[command.index("--ref_splits") + 1 : command.index("--tn_jsonl")],
            list(runner.VALIDATION_REF_SPLITS),
        )
        self.assertIn("--tn_jsonl", command)
        self.assertIn("--category_jsonl", command)
        self.assertEqual(command[command.index("--max_batches") + 1], "0")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        queue_roles = {
            role
            for record in plan["inputs"]["records"]
            for role in record["roles"]
            if role.startswith("training_queue_")
        }
        self.assertEqual(
            queue_roles,
            {
                "training_queue_manifest",
                "training_queue_detached_launch",
                "training_queue_detached_status",
            },
        )
        for path, role, sha256 in queue_bindings:
            records = [
                record
                for record in plan["inputs"]["records"]
                if role in record["roles"]
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["path"], path)
            self.assertEqual(records[0]["sha256"], sha256)
        self.assertEqual(plan["source"]["training_queue_id"], "queue-id")
        self.assertEqual(
            plan["source"]["training_queue_plan_sha256"], "a" * 64
        )

    def test_g0c_cli_forwards_training_queue_attestation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _runtime(root)
            with (
                mock.patch.object(runner, "_resolve_runtime", return_value=runtime),
                mock.patch.object(
                    runner, "build_g0c_plan", return_value={"status": "planned"}
                ) as build_plan,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(
                    runner.main(
                        [
                            "dry-run",
                            "--kind",
                            "g0c",
                            "--profile",
                            "validation",
                            "--g0c-training-plan",
                            str(root / "g0c-plan.json"),
                            "--training-queue-dir",
                            str(root / "queue"),
                            "--output-dir",
                            str(root / "output"),
                        ]
                    ),
                    0,
                )
            build_plan.assert_called_once_with(
                runtime,
                root / "g0c-plan.json",
                root / "output",
                profile="validation",
                training_queue_dir=root / "queue",
                final_gate=None,
            )

    def test_g0c_resolver_preserves_trained_source_identity_and_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _g0c_resolver_fixture(root)
            with (
                mock.patch.object(
                    runner.g0c_controls,
                    "formal_plan_path",
                    return_value=fixture["plan_path"],
                ),
                mock.patch.object(
                    runner.g0c_controls,
                    "formal_output_root",
                    return_value=fixture["output_root"],
                ),
                mock.patch.object(
                    runner.g0c_controls, "CONFIG", fixture["config"]
                ),
                mock.patch.object(
                    runner.g0c_controls,
                    "_source_dependency_tree",
                    return_value=fixture["source_tree"],
                ),
                mock.patch.object(
                    runner.g0c_controls,
                    "verify_checkpoint",
                    return_value=fixture["postflight"],
                ),
                mock.patch.object(
                    paper_eval, "_config_paths", return_value=[fixture["config"]]
                ),
                mock.patch.object(
                    paper_eval, "_resolve_baseline_source"
                ) as baseline_resolver,
            ):
                source, plan, postflight, seed = runner._resolve_g0c_source(
                    fixture["plan_path"], runner.HashCache()
                )
            baseline_resolver.assert_not_called()
            self.assertIsInstance(source, runner.G0cEvaluationSource)
            self.assertEqual(source.kind, runner.G0C_SOURCE_KIND)
            self.assertEqual(source.source_family, runner.G0C_SOURCE_FAMILY)
            self.assertEqual(source.training_run_id, "G0c:17")
            self.assertEqual(source.training_seed, 17)
            self.assertEqual(source.training_plan, fixture["plan_path"].resolve())
            self.assertEqual(source.training_postflight, postflight.resolve())
            self.assertEqual(source.checkpoint, fixture["checkpoint"].resolve())
            self.assertEqual(
                source.source_provenance_dependencies,
                fixture["dependency_paths"],
            )
            self.assertEqual(
                set(source.training_data),
                {path.resolve() for path in fixture["training_data"]},
            )
            self.assertEqual(plan["plan_sha256"], source.training_plan_contract_sha256)
            self.assertEqual(seed, 17)

    def test_g0c_resolver_rejects_missing_or_forged_training_provenance(self):
        for mutation, expected in (
            ("missing_source_tree", "dependency tree is missing"),
            ("forged_config", "config path is not canonical"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = _g0c_resolver_fixture(root)
                live_tree = json.loads(json.dumps(fixture["source_tree"]))
                if mutation == "missing_source_tree":
                    fixture["plan"].pop("source_dependency_tree")
                else:
                    forged = root / "forged_config.py"
                    forged.write_text("stage_b = False\n", encoding="utf-8")
                    fixture["plan"]["inputs"]["config"] = {
                        "path": str(forged.resolve()),
                        "sha256": runner.HashCache().digest(forged),
                    }
                fixture["plan"]["plan_sha256"] = runner.g0c_controls._plan_sha256(
                    fixture["plan"]
                )
                fixture["plan_path"].write_text(
                    json.dumps(fixture["plan"]), encoding="utf-8"
                )
                with (
                    mock.patch.object(
                        runner.g0c_controls,
                        "formal_plan_path",
                        return_value=fixture["plan_path"],
                    ),
                    mock.patch.object(
                        runner.g0c_controls,
                        "formal_output_root",
                        return_value=fixture["output_root"],
                    ),
                    mock.patch.object(
                        runner.g0c_controls, "CONFIG", fixture["config"]
                    ),
                    mock.patch.object(
                        runner.g0c_controls,
                        "_source_dependency_tree",
                        return_value=live_tree,
                    ),
                    mock.patch.object(paper_eval, "_config_paths", return_value=[fixture["config"]]),
                    self.assertRaisesRegex(runner.TableAEvaluationError, expected),
                ):
                    runner._resolve_g0c_source(
                        fixture["plan_path"], runner.HashCache()
                    )

    def test_g0c_fresh_provenance_replay_rejects_launch_relabel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _g0c_source(root)
            training_plan = {
                "plan_sha256": source.training_plan_contract_sha256,
                "matched_contract": {"seed": 17},
            }
            provenance = {"status": "fixture"}
            plan = {
                "kind": "g0c",
                "source": runner._jsonable(runner.asdict(source)),
                "contract": {
                    "training_provenance": provenance,
                    "training_contract": {"seed": 17},
                },
            }
            assert source.training_postflight is not None
            with (
                mock.patch.object(
                    runner,
                    "_resolve_g0c_source",
                    return_value=(
                        source,
                        training_plan,
                        source.training_postflight,
                        17,
                    ),
                ) as resolver,
                mock.patch.object(
                    runner,
                    "_g0c_training_provenance_contract",
                    return_value=provenance,
                ),
            ):
                replay = runner._replay_g0c_training_provenance(plan)
                self.assertEqual(replay["status"], "passed")
                resolver.assert_called_once_with(
                    source.training_plan,
                    mock.ANY,
                    training_queue_dir=source.training_queue_manifest.parent,
                )
                changed = json.loads(json.dumps(plan))
                changed["source"]["checkpoint_sha256"] = "f" * 64
                with self.assertRaisesRegex(
                    runner.TableAEvaluationError,
                    "differs from fresh canonical resolution",
                ):
                    runner._replay_g0c_training_provenance(changed)
                changed = json.loads(json.dumps(plan))
                changed["contract"]["training_contract"]["seed"] = 42
                with self.assertRaisesRegex(
                    runner.TableAEvaluationError,
                    "training contract differs",
                ):
                    runner._replay_g0c_training_provenance(changed)

    def test_table_a_separates_common_code_and_source_family_provenance_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _g0c_source(root)
            common = root / "common.py"
            table_a_only = root / "table_a_only.py"
            common.write_text("x = 1\n", encoding="utf-8")
            table_a_only.write_text("x = 2\n", encoding="utf-8")
            records = {}
            with (
                mock.patch.object(
                    paper_eval, "_config_paths", return_value=[source.config]
                ),
                mock.patch.object(
                    paper_eval,
                    "evaluation_common_code_paths",
                    return_value=[common],
                ),
                mock.patch.object(
                    paper_eval, "_data_input_paths", return_value=[]
                ),
                mock.patch.object(
                    runner, "ROLE_CODE_PATHS", (common, table_a_only)
                ),
                mock.patch.object(
                    runner.g0c_controls,
                    "_native_runtime_dependency_paths",
                    return_value=[],
                ),
            ):
                runner._source_records(source, runner.HashCache(), records)
                runner._base_data_and_code_records(
                    _runtime(root),
                    runner.HashCache(),
                    records,
                    role_mode=True,
                )
            by_path = {Path(path): record for path, record in records.items()}
            self.assertEqual(
                set(by_path[Path(sys.executable).resolve()]["roles"]),
                {"evaluation_python_runtime"},
            )
            common_roles = set(by_path[common.resolve()]["roles"])
            self.assertIn("evaluation_code_dependency", common_roles)
            self.assertNotIn("source_provenance_dependency", common_roles)
            self.assertNotIn(
                "table_a_evaluation_code_dependency", common_roles
            )
            table_a_roles = set(by_path[table_a_only.resolve()]["roles"])
            self.assertEqual(
                table_a_roles, {"table_a_evaluation_code_dependency"}
            )
            provenance_roles = set(by_path[source.config.resolve()]["roles"])
            self.assertIn("config_dependency", provenance_roles)
            self.assertIn("source_provenance_dependency", provenance_roles)
            self.assertIn(
                "source_family_table_a_g0c_provenance_dependency",
                provenance_roles,
            )

            missing = runner.G0cEvaluationSource(
                **{
                    **source.__dict__,
                    "source_provenance_dependencies": (),
                }
            )
            with (
                mock.patch.object(
                    paper_eval, "_config_paths", return_value=[source.config]
                ),
                self.assertRaisesRegex(
                    runner.TableAEvaluationError,
                    "source-family provenance dependencies are missing",
                ),
            ):
                    runner._source_records(missing, runner.HashCache(), {})

    def test_table_a_native_closure_rejects_a_different_child_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            other = Path(temporary) / "other-python"
            other.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            other.chmod(0o755)
            with self.assertRaisesRegex(
                runner.TableAEvaluationError,
                "must run under its selected Python runtime",
            ):
                runner._require_current_python_runtime(
                    other, label="Table-A evaluation"
                )

    def test_candidate_cli_forwards_training_queue_attestation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _runtime(root)
            with (
                mock.patch.object(runner, "_resolve_runtime", return_value=runtime),
                mock.patch.object(
                    runner, "build_candidate_plan", return_value={"status": "planned"}
                ) as build_plan,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(
                    runner.main(
                        [
                            "dry-run",
                            "--kind",
                            "candidate",
                            "--profile",
                            "validation",
                            "--training-run-root",
                            str(root / "training"),
                            "--training-queue-dir",
                            str(root / "queue"),
                            "--output-dir",
                            str(root / "output"),
                        ]
                    ),
                    0,
                )
            build_plan.assert_called_once_with(
                runtime,
                root / "training",
                root / "output",
                profile="validation",
                training_queue_dir=root / "queue",
                final_gate=None,
            )

    def test_g0c_plan_has_ref8_two_strict_manifests_and_all_topks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _g0c_source(root)
            training_plan_path = source.training_plan
            assert training_plan_path is not None
            training_postflight = source.training_postflight
            assert training_postflight is not None
            strict_records = {}
            for label, rows in (("strict2031", 2031), ("strict1607", 1607)):
                path = root / f"{label}.jsonl"
                path.write_text("{}\n", encoding="utf-8")
                record = runner._file_record(path, runner.HashCache(), label)
                record.update({"rows": rows, "source_counts": {}})
                strict_records[label] = record
            training_plan = {
                "plan_sha256": source.training_plan_contract_sha256,
                "matched_contract": {
                    "effective_global_batch": 40,
                    "optimizer_updates": 1000,
                    "seed": 17,
                }
            }
            with (
                mock.patch.object(
                    runner,
                    "_resolve_g0c_source",
                    return_value=(source, training_plan, training_postflight, 17),
                ),
                mock.patch.object(
                    runner, "canonical_output_dir", return_value=root / "output"
                ),
                mock.patch.object(
                    runner.g0c_controls,
                    "formal_plan_path",
                    return_value=training_plan_path,
                ),
                mock.patch.object(
                    runner.g0c_controls,
                    "formal_output_root",
                    return_value=source.training_run_root,
                ),
                mock.patch.object(runner.g0c_controls, "CONFIG", source.config),
                mock.patch.object(runner, "_base_data_and_code_records"),
                mock.patch.object(
                    runner,
                    "_profile_surface",
                    return_value=(
                        tuple(paper_eval.REF_SPLITS),
                        strict_records["strict2031"],
                        strict_records,
                    ),
                ),
                mock.patch.object(runner, "_bind_final_gate", return_value={"path": "gate", "sha256": "a" * 64}),
            ):
                plan = runner.build_g0c_plan(
                    _runtime(root),
                    training_plan_path,
                    root / "output",
                    profile="final",
                    training_queue_dir=source.training_queue_manifest.parent,
                    final_gate=root / "gate.json",
                )
        self.assertEqual(plan["kind"], "g0c")
        self.assertEqual(len(plan["commands"]), 2)
        primary = plan["commands"][0]["command"]
        topk = primary[primary.index("--topk") + 1 : primary.index("--threshold_tprs")]
        self.assertEqual(topk, ["1", "5", "10", "50"])
        self.assertEqual(
            primary[primary.index("--ref_splits") + 1 : primary.index("--tn_jsonl")],
            list(paper_eval.REF_SPLITS),
        )
        self.assertIn("--skip_ref", plan["commands"][1]["command"])

    def test_g0c_command_surface_rejects_diagnostic_or_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            tn_a = root / "strict2031.jsonl"
            tn_b = root / "strict1607.jsonl"
            config.write_text("x = 1\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            tn_a.write_text("{}\n", encoding="utf-8")
            tn_b.write_text("{}\n", encoding="utf-8")
            runtime = _runtime(root)
            source = SimpleNamespace(config=config, checkpoint=checkpoint)
            output = root / "output"
            tn_primary = {"path": str(tn_a)}
            tn_inputs = {
                "strict2031": tn_primary,
                "strict1607": {"path": str(tn_b)},
            }
            commands = runner._g0c_command_specs(
                runtime=runtime,
                source=source,
                output_dir=output,
                profile="final",
                ref_splits=paper_eval.REF_SPLITS,
                tn_primary=tn_primary,
                tn_inputs=tn_inputs,
            )
            plan = {
                "kind": "g0c",
                "profile": "final",
                "output_dir": str(output),
                "source": {
                    "config": str(config),
                    "checkpoint": str(checkpoint),
                },
                "runtime": {**runner._jsonable(runner.asdict(runtime)), "eval_seed": 42},
                "contract": {"ref_splits": list(paper_eval.REF_SPLITS)},
                "tn_manifest": tn_primary,
                "tn_inputs": tn_inputs,
                "commands": commands,
            }
            runner._validate_command_surface(plan)
            for mutation in ("wrong_checkpoint", "diagnostic_command"):
                changed = json.loads(json.dumps(plan))
                if mutation == "wrong_checkpoint":
                    command = changed["commands"][0]["command"]
                    command[command.index("--ckpts") + 1] = str(root / "wrong.pth")
                    changed["commands"][0]["command_shell"] = runner.shlex.join(command)
                else:
                    command = ["python", "diagnostic.py"]
                    changed["commands"] = [
                        {
                            "phase_id": phase,
                            "command": command,
                            "command_shell": runner.shlex.join(command),
                            "console_log": str(output / f"{phase}_console.log"),
                        }
                        for phase in ("ref8_strict2031", "strict1607")
                    ]
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    runner.TableAEvaluationError, "command"
                ):
                    runner._validate_command_surface(changed)


class TableAEvaluationOutputTest(unittest.TestCase):
    def _summary_row(self, dataset, manifest, records):
        return {
            "dataset": dataset,
            "manifest_n": manifest["rows"],
            "manifest_sha256": manifest["sha256"],
            "invalid_records": 0,
            "max_batches": 0,
            "records_jsonl": str(records),
            "acc50": 0.5,
            "acc50@5": 0.5,
            "acc50@10": 0.5,
            "acc50@50": 0.5,
            "recall50@all_queries": 0.5,
            "mean_iou": 0.5,
            "mean_iou@5": 0.5,
            "mean_iou@10": 0.5,
            "mean_iou@50": 0.5,
            "mean_best_iou@all_queries": 0.5,
        }

    def test_g0c_output_verifier_requires_every_topk_metric_and_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            primary = output / "ref8_strict2031"
            supplemental = output / "strict1607"
            primary.mkdir()
            supplemental.mkdir()
            ref_rows = []
            for split in paper_eval.REF_SPLITS:
                contract = paper_eval.REF_SPLIT_CONTRACT[split]
                records = primary / f"{split}.jsonl"
                records.write_text("{}\n" * int(contract["rows"]), encoding="utf-8")
                ref_rows.append(self._summary_row(split, contract, records))
            strict_contracts = {
                label: {
                    "rows": int(paper_eval.STRICT_SPECS[label]["rows"]),
                    "sha256": str(paper_eval.STRICT_SPECS[label]["sha256"]),
                    "path": str(
                        Path(paper_eval.STRICT_SPECS[label]["path"]).resolve()
                    ),
                }
                for label in ("strict2031", "strict1607")
            }
            tn_rows = {}
            for label, directory in (("strict2031", primary), ("strict1607", supplemental)):
                contract = strict_contracts[label]
                records = directory / f"{label}.jsonl"
                records.write_text(
                    (json.dumps({"valid": True, "pos_score": 0.8, "neg_score": 0.2}) + "\n")
                    * int(contract["rows"]),
                    encoding="utf-8",
                )
                row = self._summary_row(label, contract, records)
                row.update(
                    {
                        "source_manifest_n": int(contract["rows"]),
                        "source_manifest_sha256": str(contract["sha256"]),
                        "fpr95tpr": 0.0,
                        "fpr90tpr": 0.0,
                        "pair_win_rate": 1.0,
                        "pair_tie_rate": 0.0,
                        "pos_score_mean": 0.8,
                        "tn_score_mean": 0.2,
                        "score_gap_mean": 0.6,
                        "threshold_at_95tpr": 0.8,
                        "actual_tpr_at_95tpr": 1.0,
                    }
                )
                tn_rows[label] = row
            (primary / "summary.json").write_text(
                json.dumps({"refcoco": ref_rows, "tn": [tn_rows["strict2031"]]}),
                encoding="utf-8",
            )
            (supplemental / "summary.json").write_text(
                json.dumps({"refcoco": [], "tn": [tn_rows["strict1607"]]}),
                encoding="utf-8",
            )
            plan = {
                "output_dir": str(output),
                "profile": "final",
                "tn_manifest": strict_contracts["strict2031"],
                "tn_inputs": strict_contracts,
            }
            metrics = {
                "fpr90tpr": 0.0,
                "fpr95tpr": 0.0,
                "threshold_at_95tpr": 0.8,
                "actual_tpr_at_95tpr": 1.0,
                "pair_win_rate": 1.0,
                "pair_tie_rate": 0.0,
                "pos_score_mean": 0.8,
                "tn_score_mean": 0.2,
                "score_gap_mean": 0.6,
                "roc_auc": 1.0,
                "manifest_binding_mode": "fixture",
            }
            with (
                mock.patch.object(runner, "_tn_record_metrics", return_value=metrics),
                mock.patch.object(runner, "_validate_g0c_summary_provenance"),
                mock.patch.object(runner, "_replay_g0c_ref_records", return_value={}),
            ):
                result = runner._verify_g0c_outputs(plan, runner.HashCache())
            self.assertEqual(result["ref_splits"], 8)
            self.assertEqual(len(result["records"]), 10)
            del ref_rows[0]["acc50@50"]
            (primary / "summary.json").write_text(
                json.dumps({"refcoco": ref_rows, "tn": [tn_rows["strict2031"]]}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(runner, "_tn_record_metrics", return_value=metrics),
                mock.patch.object(runner, "_validate_g0c_summary_provenance"),
                mock.patch.object(runner, "_replay_g0c_ref_records", return_value={}),
                self.assertRaisesRegex(runner.TableAEvaluationError, "acc50@50"),
            ):
                runner._verify_g0c_outputs(plan, runner.HashCache())

    def test_g0c_validation_persists_complete_tn_metric_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            section = output / "validation_calibration"
            section.mkdir()
            ref_rows = []
            for split in runner.VALIDATION_REF_SPLITS:
                records = section / f"{split}.jsonl"
                records.write_text("{}\n", encoding="utf-8")
                ref_rows.append(
                    {"dataset": split, "records_jsonl": str(records)}
                )
            tn_records = section / "calibration.records.jsonl"
            tn_records.write_text("{}\n", encoding="utf-8")
            derived = section / "calibration.derived.jsonl"
            derived.write_text("{}\n", encoding="utf-8")
            metrics = {
                key: 0.25 for key in runner.G0C_TN_AGGREGATE_METRICS
            }
            replay = {
                **metrics,
                "roc_auc": 0.5,
                "manifest_binding_mode": "legacy_direct_source_v1",
            }
            tn_row = {
                **metrics,
                "records_jsonl": str(tn_records),
                "screen_calibration_derived_path": str(derived),
                "run_id": "formal-run",
            }
            (section / "summary.json").write_text(
                json.dumps({"refcoco": ref_rows, "tn": [tn_row]}),
                encoding="utf-8",
            )
            plan = {
                "output_dir": str(output),
                "profile": runner.VALIDATION_PROFILE,
                "evaluation_id": "G0c_seed17",
                "source": {},
            }
            with (
                mock.patch.object(
                    paper_eval, "_screen_calibration_contract", return_value={}
                ),
                mock.patch.object(
                    paper_eval,
                    "_postflight_screen",
                    return_value={"artifacts": {}},
                ),
                mock.patch.object(runner, "_validate_g0c_summary_provenance"),
                mock.patch.object(
                    runner, "_replay_g0c_ref_records", return_value={}
                ),
                mock.patch.object(
                    runner, "_tn_record_metrics", return_value=replay
                ),
            ):
                result = runner._verify_g0c_outputs(plan, runner.HashCache())
        self.assertEqual(
            result["tn_metrics_recomputed"], {"calibration": replay}
        )

    def test_g0c_tn_replay_binds_every_aggregate_metric(self):
        replay = {
            key: 0.25 for key in runner.G0C_TN_AGGREGATE_METRICS
        }
        row = dict(replay)
        runner._validate_tn_metric_replay(row, replay, label="strict2031")
        for key in runner.G0C_TN_AGGREGATE_METRICS:
            changed = {**row, key: 0.5}
            with self.subTest(key=key), self.assertRaisesRegex(
                runner.TableAEvaluationError,
                rf"{key} differs from records",
            ):
                runner._validate_tn_metric_replay(
                    changed, replay, label="strict2031"
                )

    def test_g0c_tn_replay_accepts_exact_tuple_run_identity(self):
        loaded = SimpleNamespace(
            valid=np.asarray([True, True]),
            run_ids=("formal-run",),
            positive=np.asarray([0.8, 0.6], dtype=np.float64),
            negative=np.asarray([0.2, 0.6], dtype=np.float64),
            manifest_binding_mode="source_to_derived_v1",
        )
        with (
            mock.patch(
                "tools.compare_stageb_fpr95_records.load_manifest",
                return_value=object(),
            ),
            mock.patch(
                "tools.compare_stageb_fpr95_records.load_tn_records",
                return_value=loaded,
            ),
        ):
            replay = runner._tn_record_metrics(
                Path("unused-records.jsonl"),
                Path("unused-manifest.jsonl"),
                expected_run_id="formal-run",
            )
        self.assertEqual(replay["pair_win_rate"], 0.5)
        self.assertEqual(replay["pair_tie_rate"], 0.5)
        self.assertAlmostEqual(replay["score_gap_mean"], 0.3)

    def test_g0c_summary_provenance_rejects_checkpoint_or_runtime_relabel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            wrong = root / "wrong.pth"
            config.write_text("x = 1\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            wrong.write_bytes(b"wrong")
            runtime = _runtime(root)
            checkpoint_sha = runner.HashCache().digest(checkpoint)
            plan = {
                "source": {
                    "config": str(config.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_sha,
                },
                "runtime": {**runner._jsonable(runner.asdict(runtime)), "eval_seed": 42},
            }
            row = {
                "config": str(config.resolve()),
                "config_sha256": runner.HashCache().digest(config),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_name": checkpoint.name,
                "run_id": paper_eval._checkpoint_run_id(checkpoint),
                "seed": 42,
                "batch_size": 16,
                "num_workers": 8,
                "amp": True,
                "device": "cuda:0",
                "data_root": str(root.resolve()),
                "max_batches": 0,
            }
            runner._validate_g0c_summary_provenance(
                row, plan=plan, expected_seed=42
            )
            for key, value in (("checkpoint", str(wrong)), ("amp", False)):
                changed = {**row, key: value}
                with self.subTest(key=key), self.assertRaisesRegex(
                    runner.TableAEvaluationError, "provenance/runtime"
                ):
                    runner._validate_g0c_summary_provenance(
                        changed, plan=plan, expected_seed=42
                    )

    def test_g0c_ref_metrics_are_replayed_from_exact_topk_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            values = (0.25, 0.75)
            rows = []
            for index, top1 in enumerate(values):
                rows.append(
                    {
                        "schema": "stageb-eval-record-v1",
                        "task": "ref",
                        "split": "refcoco_val",
                        "manifest_key": "ref:refcoco_val",
                        "manifest_sha256": "a" * 64,
                        "manifest_n": 2,
                        "manifest_index": index,
                        "run_id": "run",
                        "valid": True,
                        "sample_id": f"sample-{index}",
                        "top1_iou": top1,
                        "all_query_best_iou": min(1.0, top1 + 0.2),
                        "correct50": top1 >= 0.5,
                        "ranked_best_iou": {
                            "1": top1,
                            "5": min(1.0, top1 + 0.05),
                            "10": min(1.0, top1 + 0.1),
                            "50": min(1.0, top1 + 0.15),
                        },
                    }
                )
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in rows),
                encoding="utf-8",
            )
            row = {
                "manifest_n": 2,
                "manifest_sha256": "a" * 64,
                "run_id": "run",
                "num_expressions": 2,
                "valid_mask_expressions": 2,
                "invalid_mask_expressions": 0,
                "acc50": 0.5,
                "acc50@5": 0.5,
                "acc50@10": 0.5,
                "acc50@50": 0.5,
                "mean_iou": 0.5,
                "mean_iou@5": 0.55,
                "mean_iou@10": 0.6,
                "mean_iou@50": 0.65,
                "recall50@all_queries": 0.5,
                "mean_best_iou@all_queries": 0.7,
            }
            runner._replay_g0c_ref_records(
                path, row=row, split="refcoco_val"
            )
            rows[0]["ranked_best_iou"]["50"] = 0.0
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                runner.TableAEvaluationError, "not monotonic"
            ):
                runner._replay_g0c_ref_records(
                    path, row=row, split="refcoco_val"
                )

    def test_candidate_record_verifier_enforces_g5_and_patch_invariance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            common = {
                "candidate_source": "patch_topk",
                "true_role_swap": {"supported": True},
                "routes": {runner.role_eval.TRUE_ROLE_SWAP_ROUTE: {}},
            }
            rows = [
                {**common, "task": "ref", "dataset": "split"},
                {**common, "task": "tn_positive"},
                {
                    "task": "tn_counterfactual",
                    "causal_comparison_supported": True,
                    "surfaces": {
                        "patch": {
                            "delta_max_logit_negative_minus_positive": 0.0,
                            "top1_changed": False,
                        }
                    },
                },
            ]
            rows.extend(
                {
                    "task": "category_intervention",
                    "category_causal_evidence_eligible": True,
                    "category_causal_route": "joint_canonical_prompt_plus_support_patch",
                    "patch_only_category_causal_claim_eligible": False,
                    "prompt_and_support_changed_together": True,
                }
                for _ in range(1024)
            )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            receipt = runner._validate_candidate_records(
                path, expected_ref={"split": 1}, expected_tn=1
            )
            self.assertEqual(receipt["task_counts"]["category_intervention"], 1024)
            rows[2]["surfaces"]["patch"]["top1_changed"] = True
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                runner.TableAEvaluationError, "patch invariance"
            ):
                runner._validate_candidate_records(
                    path, expected_ref={"split": 1}, expected_tn=1
                )


if __name__ == "__main__":
    unittest.main()
