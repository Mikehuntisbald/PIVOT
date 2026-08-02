import copy
import hashlib
import json
import math
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler

from tests import test_stageb_table_b_matched_panel as panel_fixture
from tools.aggregate_stageb_table_b_matched_panel import (
    MatchedPanelReportError,
    _require_common_formal_protocol,
    aggregate_matched_panel,
)
from tools.compare_stageb_fpr95_records import exact_fpr95
from tools.eval_stagea_patch_checkpoints import _set_seed
from tools.eval_text_groundingdino_refcoco_tn import (
    _seed_worker,
    _support_input_identity,
)
from tools.run_stageb_paper_evaluations import (
    DATA_INPUT_RELATIVE_PATHS,
    EvaluationSource,
)
from tools.run_stageb_table_b_matched_evaluations import (
    EVAL_SEED,
    EVALUATOR_ENTRY,
    FORMAL_V2_TRAINING_SOURCE_CONTRACT,
    LAUNCH_SCHEMA,
    VALIDATION_QUEUE_SPEC_ROLE,
    MatchedEvaluationError,
    _build_postflight,
    _canonical_sha256,
    _code_records,
    _file_record,
    _input_rehash,
    _resolve_sources,
    _validate_artifact_layout,
    _write_json_atomic,
    formal_protocol_identity,
    prepare_evaluation,
    validate_condition_source,
    verify_completed_output,
)
from tools.stageb_table_b_matched_eval_surface import (
    DECLARED_SCOPE,
    EVAL_SPLIT,
    MatchedEvalSurfaceError,
    _query_image_files,
    build_surface,
    iter_rows,
    load_binding,
    sha256_file,
    summary_fields,
)


def _targets_only(batch):
    return [target for _image, target in batch]


def _surface_fixture(root: Path):
    fixture = panel_fixture.StageBTableBMatchedPanelTest()._fixture(root)
    d3_path = fixture["d3m_source_path"]
    d3_rows = [json.loads(line) for line in d3_path.read_text().splitlines()]
    for index, row in enumerate(d3_rows):
        category = row["matched_parent_key"]["edit_category"]
        row.update(
            {
                "pair_source": "refcocog_umd",
                "category_name": "object",
                "class_norm_name": "object",
                "try_tn_head": "object",
                "try_tn_head_phrase": row["sent"],
                "replace_category": [category],
                "replace_from": ["old"],
                "replace_to": ["new"],
                "replace_span": [[0, 1]],
                "tn_edits": [
                    {
                        "category": category,
                        "replace_from": "old",
                        "replace_to": "new",
                        "replace_span": [0, 1],
                    }
                ],
                "visual_verified_negative": True,
                "proposalset_proxy_verified": False,
                "cached_proposal_coverage_only": True,
                "all_900_gdino_queries_verified": False,
                "global_max_label_is_semantic_extrapolation": True,
            }
        )
        image_path = (
            root
            / "COCO/coco2014/train2014"
            / Path(row["file_name"]).name
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(64 + index, 64, 64)).save(image_path)
    panel_fixture._write_jsonl(d3_path, d3_rows)
    audit = json.loads(fixture["audit_path"].read_text())
    record = panel_fixture._artifact(d3_path, d3_rows)
    audit["outputs"]["d3m_calibration"] = copy.deepcopy(record)
    audit["outputs"]["d3m_train"] = copy.deepcopy(record)
    panel_fixture._write_json(fixture["audit_path"], audit)
    panel_fixture._write_json(
        root / "canonical_classes_with_aliases.json",
        [
            {"id": 7, "raw_name": "fixture-seven"},
            {"id": 8, "raw_name": "fixture-eight"},
        ],
    )
    support_root = root / "patches_quality"
    support_rows = []
    for class_id in (7, 8):
        for candidate in range(2):
            relative = Path("clean/fixture") / f"class-{class_id}-{candidate}.jpg"
            image_path = support_root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB", (32, 32), color=(class_id * 10, candidate * 20, 32)
            ).save(image_path)
            support_rows.append(
                "\t".join(
                    (
                        str(image_path),
                        str(class_id),
                        "0",
                        "0",
                        "0",
                        "clean",
                        str(relative.with_suffix(".npy")),
                        "256",
                        "float16",
                    )
                )
            )
    tsv = root / "patches_quality_emb/emb_index_from_quality.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text(
        "path\tclass\tocclusion\tblur\tclass_confidence\tbucket\temb_rel_path\tdim\tdtype\n"
        + "\n".join(support_rows)
        + "\n",
        encoding="utf-8",
    )
    return fixture


def _make_source(
    root: Path, *, condition: str, seed: int, training_source: Path
) -> EvaluationSource:
    run_root = root / "training" / condition / f"seed{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    config = run_root / "config.py"
    checkpoint = run_root / "checkpoint_iter.pth"
    sequence = run_root / "sequence_manifest.json"
    phase = run_root / "launch_manifest.json"
    postflight = run_root / "postflight.json"
    config.write_text(f"condition = {condition!r}\n", encoding="utf-8")
    checkpoint.write_bytes(f"{condition}:{seed}:checkpoint".encode("ascii"))
    for path, kind in (
        (sequence, "sequence"),
        (phase, "phase"),
        (postflight, "postflight"),
    ):
        path.write_text(json.dumps({"kind": kind}) + "\n", encoding="utf-8")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return EvaluationSource(
        kind="pivot_paper_training_run",
        evaluation_id=f"{condition}_seed{seed}",
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        training_run_id=f"{condition}:{seed}",
        training_seed=seed,
        training_run_root=run_root,
        sequence_manifest=sequence,
        training_phase="final",
        diagnostic_only=False,
        final_phase_id="joint",
        final_phase_manifest=phase,
        training_postflight=postflight,
        selected_phase_id="joint",
        selected_phase_manifest=phase,
        selected_training_postflight=postflight,
        training_data=(training_source,),
    )


def _resolver_for(sources):
    by_root = {
        Path(source.training_run_root).resolve(): source
        for source in sources.values()
    }

    def resolve(root, _cache, *, training_phase):
        if training_phase != "final":
            raise AssertionError(training_phase)
        return by_root[Path(root).resolve()]

    return resolve


def _prepare(root: Path, *, validation_queue_spec: Path | None = None):
    fixture = _surface_fixture(root)
    for relative in DATA_INPUT_RELATIVE_PATHS:
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    audit = json.loads(fixture["audit_path"].read_text())
    seed = 17
    sources = {
        "D2m": _make_source(
            root,
            condition="D2m",
            seed=seed,
            training_source=Path(audit["outputs"]["d2m_train"]["path"]),
        ),
        "D3m": _make_source(
            root,
            condition="D3m",
            seed=seed,
            training_source=Path(audit["outputs"]["d3m_train"]["path"]),
        ),
    }
    output = root / "evaluation"
    launch = prepare_evaluation(
        output_dir=output,
        audit_path=fixture["audit_path"],
        ledger_path=fixture["pair_ledger_path"],
        d3m_source_path=fixture["d3m_source_path"],
        data_root=root,
        d2m_training_run_root=Path(sources["D2m"].training_run_root),
        d3m_training_run_root=Path(sources["D3m"].training_run_root),
        seed=seed,
        python=Path(sys.executable),
        device="cpu",
        batch_size=2,
        num_workers=0,
        amp=False,
        log_every=1,
        resolver=_resolver_for(sources),
        validation_queue_spec=validation_queue_spec,
    )
    binding = load_binding(
        Path(launch["contract"]["surface"]["matched_eval_surface_binding_path"])
    )
    return fixture, sources, launch, binding


def _write_raw_outputs(launch, binding, sources):
    surface_rows = list(iter_rows(Path(binding.derived_manifest["path"])))
    for condition_index, phase in enumerate(launch["contract"]["phases"]):
        condition = phase["condition"]
        raw_dir = Path(phase["raw_output_dir"])
        records_path = raw_dir / "per_example_records" / "raw.records.jsonl"
        records = []
        positives = []
        negatives = []
        for index, surface in enumerate(surface_rows):
            positive = 0.9 - index * 0.05 + condition_index * 0.01
            negative = 0.45 - index * 0.04 - condition_index * 0.05
            positives.append(positive)
            negatives.append(negative)
            records.append(
                {
                    "schema": "stageb-eval-record-v1",
                    "task": "tn",
                    "manifest_key": "tn_global",
                    "manifest_sha256": binding.derived_manifest["sha256"],
                    "manifest_n": len(surface_rows),
                    "manifest_index": index,
                    "sample_id": surface["sample_id"],
                    "image_id": surface["image_id"],
                    "ann_id": surface["ann_id"],
                    "ref_id": surface["ref_id"],
                    "sent_id": surface["sent_id"],
                    "split": EVAL_SPLIT,
                    "run_id": f"raw-{condition}",
                    "valid": True,
                    "train_scope": None,
                    "eval_scope": DECLARED_SCOPE,
                    "support_input_kind": "patches",
                    "support_input_sha256": hashlib.sha256(
                        f"support-input:{index}".encode()
                    ).hexdigest(),
                    "support_class_ids": [int(surface["class_id"])],
                    "pos_score": positive,
                    "neg_score": negative,
                    "pos_iou": 0.75,
                    "neg_iou": 0.75,
                }
            )
        panel_fixture._write_jsonl(records_path, records)
        fpr = exact_fpr95(positives, negatives)
        pair_win = sum(p > n for p, n in zip(positives, negatives)) / len(records)
        summary_row = {
            **summary_fields(binding),
            "run_id": f"raw-{condition}",
            "checkpoint": str(Path(sources[condition].checkpoint).resolve()),
            "seed": EVAL_SEED,
            "max_batches": 0,
            "eval_scope": DECLARED_SCOPE,
            "manifest_sha256": binding.derived_manifest["sha256"],
            "manifest_n": len(records),
            "num_pairs": len(records),
            "invalid_records": 0,
            "invalid_positive_pairs": 0,
            "invalid_negative_pairs": 0,
            "records_jsonl": str(records_path.resolve()),
            "fpr95tpr": float(fpr["fpr"]),
            "threshold_at_95tpr": float(fpr["threshold"]),
            "pair_win_rate": pair_win,
        }
        panel_fixture._write_json(
            raw_dir / "summary.json", {"refcoco": [], "tn": [summary_row]}
        )


class StageBTableBMatchedEvaluationsTest(unittest.TestCase):
    def test_subprocess_evaluator_dependency_closure_is_exact_and_fail_closed(self):
        relative_paths = {
            str(Path(record["path"]).resolve().relative_to(Path.cwd().resolve()))
            for record in _code_records()
        }
        for required in (
            EVALUATOR_ENTRY,
            "tools/eval_stageb_tn_val.py",
            "tools/stageb_eval_records.py",
            "datasets/patch_episode.py",
            "models/GroundingDINO/groundingdino.py",
        ):
            self.assertIn(required, relative_paths)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture, _sources, launch, binding = _prepare(root)
            protocol = formal_protocol_identity(launch)
            closure_paths = {
                record["path"] for record in protocol["evaluation_code_closure"]
            }
            self.assertIn(EVALUATOR_ENTRY, closure_paths)

            missing = copy.deepcopy(launch)
            missing["contract"]["input_records"] = [
                record
                for record in missing["contract"]["input_records"]
                if not str(record["path"]).endswith(EVALUATOR_ENTRY)
            ]
            missing["contract_sha256"] = _canonical_sha256(missing["contract"])
            with self.assertRaisesRegex(
                MatchedEvaluationError, "evaluation code closure identity drifted"
            ):
                _validate_artifact_layout(missing, binding=binding)

            changed = copy.deepcopy(launch)
            code_record = next(
                record
                for record in changed["contract"]["input_records"]
                if "evaluation_code" in record.get("roles", [])
            )
            code_record["sha256"] = "f" * 64
            changed["contract_sha256"] = _canonical_sha256(changed["contract"])
            with self.assertRaisesRegex(
                MatchedEvaluationError, "evaluation code closure identity drifted"
            ):
                _validate_artifact_layout(changed, binding=binding)

    def test_cross_seed_formal_protocol_rejects_runtime_code_and_command_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture, _sources, launch, _binding = _prepare(root)
            reference = formal_protocol_identity(launch)
            self.assertEqual(
                _require_common_formal_protocol(None, reference, seed=17), reference
            )

            runtime_mutations = (
                ("amp", True),
                ("batch_size", 3),
                ("num_workers", 1),
                ("device", "cpu:changed"),
                ("data_root", str(root.parent.resolve())),
            )
            for field, value in runtime_mutations:
                observed = copy.deepcopy(reference)
                observed["common_runtime"][field] = value
                with self.assertRaisesRegex(
                    MatchedPanelReportError, "runtime protocol differs"
                ):
                    _require_common_formal_protocol(reference, observed, seed=42)

            changed_python = copy.deepcopy(reference)
            changed_python["common_runtime"]["python"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(
                MatchedPanelReportError, "runtime protocol differs"
            ):
                _require_common_formal_protocol(reference, changed_python, seed=42)

            extra_flag_launch = copy.deepcopy(launch)
            for phase in extra_flag_launch["contract"]["phases"]:
                phase["command"].append("--unexpected-formal-flag")
            extra_flag = formal_protocol_identity(extra_flag_launch)
            with self.assertRaisesRegex(
                MatchedPanelReportError, "phase command template differs"
            ):
                _require_common_formal_protocol(reference, extra_flag, seed=42)

            changed_closure = copy.deepcopy(reference)
            changed_closure["evaluation_code_closure"] = changed_closure[
                "evaluation_code_closure"
            ][1:]
            with self.assertRaisesRegex(
                MatchedPanelReportError, "evaluation code closure differs"
            ):
                _require_common_formal_protocol(reference, changed_closure, seed=73)

    def test_support_identity_normalizes_single_patch_and_preserves_multi_patch(self):
        patch = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        tensor_single = _support_input_identity(
            {"patch": patch, "support_class": torch.tensor([7])},
            expected_class_id=7,
        )
        integer_single = _support_input_identity(
            {"patch": patch, "support_class": 7}, expected_class_id=7
        )
        self.assertEqual(tensor_single, integer_single)
        self.assertEqual(tensor_single["support_class_ids"], [7])

        multi = _support_input_identity(
            {
                "patches": torch.stack((patch, patch + 1)),
                "support_classes": torch.tensor([7, 8]),
                "support_class": torch.tensor([999]),
            }
        )
        self.assertEqual(multi["support_class_ids"], [7, 8])
        self.assertEqual(multi["support_input_kind"], "patches")

        invalid_targets = (
            {"patch": patch, "support_class": torch.tensor([7.0])},
            {"patch": patch, "support_class": torch.tensor([7, 8])},
            {"patch": patch, "support_classes": [7], "support_class": 7},
        )
        for target in invalid_targets:
            with self.assertRaisesRegex(RuntimeError, "support_class"):
                _support_input_identity(target)
        with self.assertRaisesRegex(RuntimeError, "differs from its manifest row"):
            _support_input_identity(
                {"patch": patch, "support_class": torch.tensor([8])},
                expected_class_id=7,
            )

    def test_single_patch_four_worker_replay_hashes_actual_support_tensors(self):
        from datasets.patch_episode import PatchEpisodeJsonlDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _surface_fixture(root)
            derived = root / "surface" / "matched.jsonl"
            build_surface(
                audit_path=fixture["audit_path"],
                ledger_path=fixture["pair_ledger_path"],
                source_path=fixture["d3m_source_path"],
                derived_path=derived,
                data_root=root,
            )
            rows = list(iter_rows(derived))

            def replay():
                _set_seed(EVAL_SEED)
                dataset = PatchEpisodeJsonlDataset(
                    root="/",
                    anno=str(derived),
                    source="matched_calibration",
                    box_format="xywh",
                    neg_episode_prob=0.0,
                    support_min_count=1,
                    support_num_patches_min=1,
                    support_num_patches_max=1,
                    support_patch_tsv=str(
                        root / "patches_quality_emb/emb_index_from_quality.tsv"
                    ),
                    support_patch_bucket="clean",
                    support_patch_use_embedding=False,
                    support_patch_image_root=str(root / "patches_quality"),
                    support_patch_max_per_class=200,
                    patch_bank_cache=False,
                    patch_bank_cache_write=False,
                    canonical_classes_json=str(
                        root / "canonical_classes_with_aliases.json"
                    ),
                    keep_only_support_gt=True,
                    build_text_token_masks=False,
                    tn_balance_sampling=False,
                )
                generator = torch.Generator()
                generator.manual_seed(EVAL_SEED)
                loader = DataLoader(
                    dataset,
                    batch_size=2,
                    sampler=SequentialSampler(dataset),
                    drop_last=False,
                    collate_fn=_targets_only,
                    num_workers=4,
                    worker_init_fn=_seed_worker,
                    generator=generator,
                )
                targets = [target for batch in loader for target in batch]
                return [
                    _support_input_identity(
                        target, expected_class_id=int(rows[index]["class_id"])
                    )
                    for index, target in enumerate(targets)
                ]

            first = replay()
            second = replay()
            self.assertEqual(len(first), len(rows))
            self.assertEqual(first, second)
            self.assertTrue(
                all(identity["support_input_kind"] == "patch" for identity in first)
            )

    def test_surface_replays_audit_ledger_rows_and_dataset_consumes_it(self):
        from datasets.patch_episode import PatchEpisodeJsonlDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _surface_fixture(root)
            derived = root / "surface" / "matched.jsonl"
            binding = build_surface(
                audit_path=fixture["audit_path"],
                ledger_path=fixture["pair_ledger_path"],
                source_path=fixture["d3m_source_path"],
                derived_path=derived,
                data_root=root,
            )
            rows = list(iter_rows(derived))
            audit_sha = sha256_file(fixture["audit_path"])
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(binding.row_mapping), 4)
            self.assertEqual(len(binding.image_files), 4)
            self.assertEqual(len(binding.support_pool_mapping), 2)
            self.assertEqual(len(binding.support_pool_files), 4)
            replayed = load_binding(binding.path, expected_derived=derived)
            self.assertEqual(
                replayed.support_pool_mapping_sha256,
                binding.support_pool_mapping_sha256,
            )
            self.assertEqual(
                replayed.support_pool_files_sha256,
                binding.support_pool_files_sha256,
            )
            self.assertEqual(rows[0]["tn_eval_split"], EVAL_SPLIT)
            self.assertEqual(rows[0]["table_b_audit_sha256"], audit_sha)
            self.assertEqual(
                rows[0]["instances"][0]["table_b_audit_sha256"], audit_sha
            )
            self.assertEqual(
                rows[0]["matched_pair_id"], binding.row_mapping[0]["matched_pair_id"]
            )
            random_state = random.getstate()
            patch_tsv = root / "patches_quality_emb/emb_index_from_quality.tsv"
            cache_path = Path(str(patch_tsv) + ".bank.clean.img.pkl")
            cache_path.write_bytes(b"malicious-cache-must-be-ignored")
            try:
                random.seed(EVAL_SEED)
                dataset = PatchEpisodeJsonlDataset(
                    root="/",
                    anno=str(derived),
                    source="matched_calibration",
                    box_format="xywh",
                    neg_episode_prob=0.0,
                    support_min_count=1,
                    support_num_patches_min=1,
                    support_num_patches_max=1,
                    support_patch_tsv=str(
                        patch_tsv
                    ),
                    support_patch_bucket="clean",
                    support_patch_use_embedding=False,
                    support_patch_image_root=str(root / "patches_quality"),
                    support_patch_max_per_class=200,
                    patch_bank_cache=False,
                    patch_bank_cache_write=False,
                    canonical_classes_json=str(
                        root / "canonical_classes_with_aliases.json"
                    ),
                    build_text_token_masks=False,
                    tn_balance_sampling=False,
                )
            finally:
                random.setstate(random_state)
            observed_mapping = [
                {
                    "class_id": class_id,
                    "candidate_paths": [
                        str(Path(path).resolve())
                        for path in dataset.patch_bank[class_id]
                    ],
                }
                for class_id in sorted({int(row["class_id"]) for row in rows})
            ]
            self.assertEqual(observed_mapping, list(binding.support_pool_mapping))
            self.assertEqual(
                cache_path.read_bytes(), b"malicious-cache-must-be-ignored"
            )
            _image, target = dataset[0]
            self.assertEqual(target["table_b_id"], "D3m")
            self.assertEqual(target["table_b_audit_sha256"], audit_sha)
            self.assertFalse(target["global_tn_verified"])

            tampered = rows
            tampered[0]["instances"][0]["tn_scope"] = "image_global_topk_verified"
            panel_fixture._write_jsonl(derived, tampered)
            with self.assertRaisesRegex(
                MatchedEvalSurfaceError, "bound derived_manifest changed|derived row drift"
            ):
                load_binding(binding.path, expected_derived=derived)

    def test_training_resolver_condition_seed_and_source_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _surface_fixture(root)
            audit = json.loads(fixture["audit_path"].read_text())
            expected = audit["outputs"]["d2m_train"]
            source = _make_source(
                root,
                condition="D2m",
                seed=17,
                training_source=Path(expected["path"]),
            )
            evidence = validate_condition_source(
                condition="D2m",
                seed=17,
                source=source,
                training_source_record=expected,
            )
            self.assertEqual(evidence["training_run_id"], "D2m:17")
            with self.assertRaisesRegex(MatchedEvaluationError, "completed D3m:17"):
                validate_condition_source(
                    condition="D3m",
                    seed=17,
                    source=source,
                    training_source_record=audit["outputs"]["d3m_train"],
                )
            relabeled = copy.copy(source)
            object.__setattr__(relabeled, "training_seed", 42)
            with self.assertRaisesRegex(MatchedEvaluationError, "completed D2m:17"):
                validate_condition_source(
                    condition="D2m",
                    seed=17,
                    source=relabeled,
                    training_source_record=expected,
                )

            with self.assertRaisesRegex(
                MatchedEvaluationError, "formal Table-B v2 source"
            ):
                validate_condition_source(
                    condition="D2m",
                    seed=17,
                    source=source,
                    training_source_record=expected,
                    formal_v2=True,
                )
            formal_source = replace(
                source,
                formal_contract_id="table_b_v2_formal_b40_u1000_i1000",
                matrix_validation_only=True,
            )
            formal_evidence = validate_condition_source(
                condition="D2m",
                seed=17,
                source=formal_source,
                training_source_record=expected,
                formal_v2=True,
            )
            self.assertEqual(formal_evidence["training_run_id"], "D2m:17")
            missing = copy.copy(source)
            object.__setattr__(missing, "training_data", ())
            with self.assertRaisesRegex(MatchedEvaluationError, "lacks its audited"):
                validate_condition_source(
                    condition="D2m",
                    seed=17,
                    source=missing,
                    training_source_record=expected,
                )

    def test_formal_v2_source_mode_persists_queue_attestation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _surface_fixture(root)
            audit = json.loads(fixture["audit_path"].read_text())
            seed = 17
            sources = {
                condition: replace(
                    _make_source(
                        root,
                        condition=condition,
                        seed=seed,
                        training_source=Path(
                            audit["outputs"][f"{condition.lower()}_train"]["path"]
                        ),
                    ),
                    formal_contract_id="table_b_v2_formal_b40_u1000_i1000",
                    matrix_validation_only=True,
                )
                for condition in ("D2m", "D3m")
            }
            queue_evidence = {
                "queue_id": "queue",
                "plan_sha256": "a" * 64,
                "manifest": {"path": str(root / "queue.json")},
            }

            def formal_evidence(_queue, *, run_id, run_root):
                return {
                    "profile": "table_b_v2_formal_b40_u1000_i1000",
                    "run_id": run_id,
                    "completion_attestation": {
                        "path": str(root / "formal_completion_attestation.json")
                    },
                    "run_root": str(run_root),
                }

            with patch(
                "tools.run_stageb_table_b_matched_evaluations._paper_queue_attestation",
                return_value=queue_evidence,
            ), patch(
                "tools.run_stageb_table_b_v2.matched_evaluation_resolver",
                return_value=_resolver_for(sources),
            ), patch(
                "tools.run_stageb_table_b_v2_queue.formal_evaluation_evidence",
                side_effect=formal_evidence,
            ):
                resolved, evidence = _resolve_sources(
                    d2m_root=Path(sources["D2m"].training_run_root),
                    d3m_root=Path(sources["D3m"].training_run_root),
                    seed=seed,
                    audit=audit,
                    audit_path=fixture["audit_path"],
                    training_queue_dir=root,
                    formal_v2=True,
                )
            self.assertEqual(set(resolved), {"D2m", "D3m"})
            self.assertEqual(
                evidence["D2m"]["formal_v2"]["run_id"], "D2m:17"
            )
            self.assertEqual(
                FORMAL_V2_TRAINING_SOURCE_CONTRACT, "table_b_v2_formal"
            )
    def test_query_image_path_escape_and_identity_collision_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "COCO/coco2014/train2014"
            image_root.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg"):
                Image.new("RGB", (8, 8)).save(image_root / name)
            with self.assertRaisesRegex(MatchedEvalSurfaceError, "canonical basename"):
                _query_image_files(
                    [{"file_name": "../a.jpg", "image_id": 1}], data_root=root
                )
            with self.assertRaisesRegex(MatchedEvalSurfaceError, "multiple canonical"):
                _query_image_files(
                    [
                        {"file_name": "a.jpg", "image_id": 1},
                        {"file_name": "b.jpg", "image_id": 1},
                    ],
                    data_root=root,
                )
            with self.assertRaisesRegex(MatchedEvalSurfaceError, "multiple image IDs"):
                _query_image_files(
                    [
                        {"file_name": "a.jpg", "image_id": 1},
                        {"file_name": "a.jpg", "image_id": 2},
                    ],
                    data_root=root,
                )
    def test_dry_run_plan_uses_one_surface_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, sources, launch, binding = _prepare(root)
            self.assertEqual(launch["schema"], LAUNCH_SCHEMA)
            self.assertTrue(launch["contract"]["same_surface_for_both_conditions"])
            self.assertEqual(binding.derived_manifest["rows"], 4)
            commands = [phase["command"] for phase in launch["contract"]["phases"]]
            manifest_values = [
                command[command.index("--tn_jsonl") + 1] for command in commands
            ]
            binding_values = [
                command[command.index("--direct_prebuilt_tn_binding") + 1]
                for command in commands
            ]
            self.assertEqual(len(set(manifest_values)), 1)
            self.assertEqual(len(set(binding_values)), 1)
            self.assertTrue(all("--skip_ref" in command for command in commands))
            self.assertTrue(all("--direct_prebuilt_tn" in command for command in commands))
            with self.assertRaisesRegex(FileExistsError, "fresh"):
                prepare_evaluation(
                    output_dir=Path(launch["output_dir"]),
                    audit_path=fixture["audit_path"],
                    ledger_path=fixture["pair_ledger_path"],
                    d3m_source_path=fixture["d3m_source_path"],
                    data_root=root,
                    d2m_training_run_root=Path(sources["D2m"].training_run_root),
                    d3m_training_run_root=Path(sources["D3m"].training_run_root),
                    seed=17,
                    python=Path(sys.executable),
                    device="cpu",
                    batch_size=2,
                    num_workers=0,
                    amp=False,
                    log_every=1,
                    resolver=_resolver_for(sources),
                )

    def test_validation_queue_spec_is_hash_bound_and_rehashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / "validation_input_spec.json"
            spec.write_text('{"schema":"fixture-validation-plan"}\n', encoding="ascii")
            _fixture, _sources, launch, _binding = _prepare(
                root, validation_queue_spec=spec
            )
            declared = launch["contract"]["validation_queue_spec"]
            self.assertEqual(declared["role"], VALIDATION_QUEUE_SPEC_ROLE)
            bound = [
                record
                for record in launch["contract"]["input_records"]
                if VALIDATION_QUEUE_SPEC_ROLE in record["roles"]
            ]
            self.assertEqual(len(bound), 1)
            self.assertEqual(bound[0]["sha256"], declared["sha256"])
            spec.write_text('{"schema":"tampered"}\n', encoding="ascii")
            with self.assertRaisesRegex(MatchedEvaluationError, "input record"):
                _input_rehash(launch["contract"]["input_records"])

    def test_resigned_contract_cannot_relabel_condition_seed_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture, sources, launch, binding = _prepare(root)

            relabeled = copy.deepcopy(launch)
            relabeled["contract"]["training"]["D2m"]["condition"] = "D3m"
            relabeled["contract_sha256"] = _canonical_sha256(relabeled["contract"])
            with self.assertRaisesRegex(
                MatchedEvaluationError, "runtime source differs|paths/commands"
            ):
                _build_postflight(
                    launch=relabeled, sources=sources, binding=binding
                )

            reseeded = copy.deepcopy(launch)
            reseeded["contract"]["seed"] = 73
            reseeded["contract_sha256"] = _canonical_sha256(reseeded["contract"])
            with self.assertRaisesRegex(
                MatchedEvaluationError, "runtime source differs|paths/commands"
            ):
                _build_postflight(
                    launch=reseeded, sources=sources, binding=binding
                )

            swapped_phase = copy.deepcopy(launch)
            swapped_phase["contract"]["phases"][0]["condition"] = "D3m"
            swapped_phase["contract_sha256"] = _canonical_sha256(
                swapped_phase["contract"]
            )
            with self.assertRaisesRegex(
                MatchedEvaluationError,
                "runtime condition contract|paths/commands|phase set drifted",
            ):
                _build_postflight(
                    launch=swapped_phase, sources=sources, binding=binding
                )

            changed_checkpoint = copy.deepcopy(launch)
            changed_checkpoint["contract"]["training"]["D2m"]["checkpoint"][
                "sha256"
            ] = "0" * 64
            changed_checkpoint["contract_sha256"] = _canonical_sha256(
                changed_checkpoint["contract"]
            )
            with self.assertRaisesRegex(
                MatchedEvaluationError, "runtime source differs|paths/commands"
            ):
                _build_postflight(
                    launch=changed_checkpoint, sources=sources, binding=binding
                )

    def test_postflight_rebinds_scores_and_verifier_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, sources, launch, binding = _prepare(root)
            _write_raw_outputs(launch, binding, sources)
            for phase in launch["phases"]:
                phase.update({"status": "completed", "returncode": 0})
            launch["completed_conditions"] = ["D2m", "D3m"]
            postflight = _build_postflight(
                launch=launch, sources=sources, binding=binding
            )
            self.assertEqual(postflight["status"], "passed")
            self.assertTrue(postflight["all_records_and_scores_replayed"])
            self.assertFalse(postflight["formal_global_fpr_eligible"])
            final_paths = {
                condition: Path(
                    postflight["conditions"][condition]["final_records"]["path"]
                )
                for condition in ("D2m", "D3m")
            }
            report = aggregate_matched_panel(
                audit_path=fixture["audit_path"],
                pair_ledger_path=fixture["pair_ledger_path"],
                d2m_source_path=fixture["d2m_source_path"],
                d3m_source_path=fixture["d3m_source_path"],
                evaluation_manifest_path=fixture["d3m_source_path"],
                d2m_records={17: final_paths["D2m"]},
                d3m_records={17: final_paths["D3m"]},
                expected_seeds=[17],
            )
            self.assertTrue(report["validation"]["pass"])
            d2_raw = _read_records(
                Path(postflight["conditions"]["D2m"]["raw_records"]["path"])
            )
            d2_final = _read_records(final_paths["D2m"])
            self.assertEqual(
                [row["pos_score"] for row in d2_raw],
                [row["pos_score"] for row in d2_final],
            )
            self.assertTrue(
                all(row["matched_eval_surface_sha256"] == binding.derived_manifest["sha256"] for row in d2_final)
            )

            output = Path(launch["output_dir"])
            postflight_path = output / "postflight.json"
            _write_json_atomic(postflight_path, postflight)
            for phase in launch["phases"]:
                log_path = Path(phase["console_log"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("synthetic evaluator completed\n", encoding="utf-8")
                phase["console_log_record"] = _file_record(
                    log_path, role=f"{phase['condition']}:console_log"
                )
            launch["status"] = "completed"
            launch["postflight"] = postflight
            launch["postflight_artifact"] = _file_record(postflight_path, role="postflight")
            _write_json_atomic(output / "launch.json", launch)
            self.assertEqual(verify_completed_output(output)["status"], "passed")

            tampered = _read_records(final_paths["D2m"])
            tampered[0]["pos_score"] += 0.1
            panel_fixture._write_jsonl(final_paths["D2m"], tampered)
            with self.assertRaisesRegex(MatchedEvaluationError, "artifact drift"):
                verify_completed_output(output)

    def test_query_and_support_pixels_are_rehashed_and_missing_files_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture, _sources, launch, binding = _prepare(root)
            records = launch["contract"]["input_records"]
            self.assertEqual(
                sum("matched_eval_query_image" in row["roles"] for row in records),
                len(binding.image_files),
            )
            self.assertEqual(
                sum("matched_eval_support_patch" in row["roles"] for row in records),
                len(binding.support_pool_files),
            )
            for bound in (binding.image_files[0], binding.support_pool_files[0]):
                path = Path(bound["path"])
                original = path.read_bytes()
                path.write_bytes(b"pixel-drift")
                with self.assertRaisesRegex(MatchedEvaluationError, "input record"):
                    _input_rehash(records)
                path.write_bytes(original)
            support_path = Path(binding.support_pool_files[0]["path"])
            original = support_path.read_bytes()
            support_path.unlink()
            with self.assertRaises((FileNotFoundError, MatchedEvalSurfaceError)):
                load_binding(binding.path)
            support_path.write_bytes(original)

    def test_resigned_artifact_escape_and_symlink_escape_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture, _sources, launch, binding = _prepare(root)
            escaped = copy.deepcopy(launch)
            escaped_path = str(root.parent / "escaped-raw")
            escaped["contract"]["phases"][0]["raw_output_dir"] = escaped_path
            escaped["phases"][0]["raw_output_dir"] = escaped_path
            escaped["contract_sha256"] = _canonical_sha256(escaped["contract"])
            with self.assertRaisesRegex(
                MatchedEvaluationError, "paths/commands|value is not canonical"
            ):
                _validate_artifact_layout(escaped, binding=binding)

            raw_parent = Path(launch["output_dir"]) / "raw"
            outside = root / "outside"
            outside.mkdir()
            raw_parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(
                MatchedEvaluationError, "escaped output_dir|value is not canonical"
            ):
                _validate_artifact_layout(launch, binding=binding)

    def test_raw_scope_upgrade_or_reorder_fails_before_postprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture, sources, launch, binding = _prepare(root)
            _write_raw_outputs(launch, binding, sources)
            for phase in launch["phases"]:
                phase.update({"status": "completed", "returncode": 0})
            launch["completed_conditions"] = ["D2m", "D3m"]
            d2_phase = launch["contract"]["phases"][0]
            summary = json.loads(
                (Path(d2_phase["raw_output_dir"]) / "summary.json").read_text()
            )
            raw_path = Path(summary["tn"][0]["records_jsonl"])
            records = _read_records(raw_path)
            records[0]["eval_scope"] = "image_global_topk_verified"
            records[0]["global_tn_verified"] = True
            panel_fixture._write_jsonl(raw_path, records)
            with self.assertRaisesRegex(MatchedEvaluationError, "identity/scope"):
                _build_postflight(launch=launch, sources=sources, binding=binding)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture, sources, launch, binding = _prepare(root)
            _write_raw_outputs(launch, binding, sources)
            for phase in launch["phases"]:
                phase.update({"status": "completed", "returncode": 0})
            launch["completed_conditions"] = ["D2m", "D3m"]
            d2_phase = launch["contract"]["phases"][0]
            summary = json.loads(
                (Path(d2_phase["raw_output_dir"]) / "summary.json").read_text()
            )
            raw_path = Path(summary["tn"][0]["records_jsonl"])
            records = _read_records(raw_path)
            records[0], records[1] = records[1], records[0]
            panel_fixture._write_jsonl(raw_path, records)
            with self.assertRaisesRegex(MatchedEvaluationError, "identity/scope"):
                _build_postflight(launch=launch, sources=sources, binding=binding)


def _read_records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    unittest.main()
