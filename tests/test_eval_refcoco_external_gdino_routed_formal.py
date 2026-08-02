import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.test_eval_refcoco_external_gdino_formal import _manifest
from tests.test_eval_refcoco_external_gdino_rank_transfer import (
    _external_outputs,
    _patch_outputs,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as text_ref_eval
from tools import stageb_canonical_caption_route_artifact as route_tool
from tools.stageb_eval_records import EvalManifest, write_eval_records
from tools.stageb_external_rank_transfer_artifact import (
    CAPTION_PROVENANCE_CONTRACT,
    REF_SPLIT_ORDER,
    SPLIT_SEED_PROTOCOL,
    stable_ref_split_seed_map,
)


def _descriptor_grid(descriptor_id):
    descriptor = route_tool.DESCRIPTOR_REGISTRY[descriptor_id]
    return {
        "transfer_mode": descriptor["transfer_mode"],
        "iou_power": descriptor["iou_power"],
        "patch_weight": descriptor["patch_weight"],
        "text_weight": descriptor["text_weight"],
    }


def _routed_settings(root: Path, overrides):
    canonical_classes_path = root / "canonical_classes_with_aliases.json"
    canonical_classes_path.write_text(
        json.dumps(
            [
                {"id": 1, "base_name": "car"},
                {"id": 2, "base_name": "person"},
            ]
        ),
        encoding="utf-8",
    )
    descriptor_ids = list(route_tool.CANDIDATE_DESCRIPTOR_IDS)
    descriptor_grid_by_id = {
        descriptor_id: _descriptor_grid(descriptor_id)
        for descriptor_id in descriptor_ids
    }
    fixed_grid = list(descriptor_grid_by_id.values())
    return {
        "formal_artifact_version": 2,
        "transfer_modes": list(
            dict.fromkeys(row["transfer_mode"] for row in fixed_grid)
        ),
        "iou_powers": list(
            dict.fromkeys(
                row["iou_power"]
                for row in fixed_grid
                if row["iou_power"] is not None
            )
        ),
        "patch_weights": list(
            dict.fromkeys(row["patch_weight"] for row in fixed_grid)
        ),
        "text_weights": list(
            dict.fromkeys(row["text_weight"] for row in fixed_grid)
        ),
        "candidate_topk": 2,
        "contract_patch_rank_weight": 1.0,
        "external_query_count": 900,
        "artifact_identity": {"algorithm": "fixture", "sha256": "a" * 64},
        "artifact": {
            "path": str((root / "formal-v2.json").resolve()),
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
        "patch_config": {
            "path": str((root / "patch.py").resolve()),
            "size_bytes": 10,
            "sha256": "1" * 64,
        },
        "patch_checkpoints": [
            {
                "path": str((root / "patch.pth").resolve()),
                "size_bytes": 20,
                "sha256": "2" * 64,
            }
        ],
        "external_config": {
            "path": str((root / "external.py").resolve()),
            "size_bytes": 11,
            "sha256": "3" * 64,
        },
        "external_checkpoint": {
            "path": str((root / "external.pth").resolve()),
            "size_bytes": 21,
            "sha256": "4" * 64,
        },
        "caption_provenance": dict(CAPTION_PROVENANCE_CONTRACT),
        "evaluation_protocol": {
            "seed_protocol": SPLIT_SEED_PROTOCOL,
            "base_seed": 42,
            "split_seed_stride": 100000,
            "canonical_split_order": list(REF_SPLIT_ORDER),
            "split_seeds": stable_ref_split_seed_map(42),
        },
        "route_selection": {
            "artifact": {
                "path": str((root / "selection.json").resolve()),
                "size_bytes": 2,
                "sha256": "c" * 64,
            },
            "artifact_identity": {"algorithm": "fixture", "sha256": "d" * 64},
            "selection_contract": {
                "batch_size": 1,
                "num_workers": 0,
                "amp": "disabled_by_fixture",
            },
            "canonical_classes": {
                "path": str(canonical_classes_path.resolve()),
                "size_bytes": canonical_classes_path.stat().st_size,
                "sha256": ref_eval._sha256_file(canonical_classes_path),
            },
        },
        "route_policy": {
            "default_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
            "overrides": dict(overrides),
        },
        "route_policy_sha256": "e" * 64,
        "route_default_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
        "route_overrides": dict(overrides),
        "descriptor_registry": dict(route_tool.DESCRIPTOR_REGISTRY),
        "descriptor_grid_by_id": descriptor_grid_by_id,
        "fixed_grid": fixed_grid,
    }


def _repeat_outputs(outputs, batch_size):
    repeated = {}
    for key, value in outputs.items():
        if torch.is_tensor(value):
            repeats = [batch_size] + [1] * (value.dim() - 1)
            repeated[key] = value.repeat(*repeats)
        else:
            repeated[key] = value
    return repeated


def _target(box):
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([123], dtype=torch.int64),
    }


class FormalRoutedExternalGDINOTest(unittest.TestCase):
    def test_formal_routed_runtime_must_match_selection_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _routed_settings(Path(tmpdir), {})
            ref_eval._validate_formal_routed_runtime(
                settings,
                batch_size=1,
                num_workers=0,
                amp=False,
            )
            for mismatch in (
                {"batch_size": 2, "num_workers": 0, "amp": False},
                {"batch_size": 1, "num_workers": 1, "amp": False},
                {"batch_size": 1, "num_workers": 0, "amp": True},
            ):
                with self.assertRaisesRegex(ValueError, "frozen selection"):
                    ref_eval._validate_formal_routed_runtime(settings, **mismatch)

    def test_formal_routed_target_stage_a_caption_cannot_fall_back(self):
        manifest = EvalManifest(
            path=Path("/tmp/formal-routed-caption-provenance.jsonl"),
            task="ref",
            manifest_key="ref:refcoco_val",
            split="refcoco_val",
            sha256="f" * 64,
            rows=[
                {
                    "image_id": 10,
                    "ann_id": 20,
                    "ref_id": 30,
                    "sent_id": 40,
                    "instances": [
                        {"raw_phrase": "the red car", "class_id": 1}
                    ],
                }
            ],
        )
        target = {"caption": "the red car ."}
        with self.assertRaisesRegex(ValueError, "lacks a non-empty stage_a_caption"):
            ref_eval._validate_formal_external_caption_provenance(
                [target],
                manifest,
                0,
                settings={
                    "caption_provenance": dict(CAPTION_PROVENANCE_CONTRACT)
                },
                routed_canonical_id_to_caption={1: "car"},
            )
        target["stage_a_caption"] = "person ."
        with self.assertRaisesRegex(ValueError, "canonical-class provenance drift"):
            ref_eval._validate_formal_external_caption_provenance(
                [target],
                manifest,
                0,
                settings={
                    "caption_provenance": dict(CAPTION_PROVENANCE_CONTRACT)
                },
                routed_canonical_id_to_caption={1: "car"},
            )

    def test_evaluate_dataset_dispatches_v2_and_binds_records_file(self):
        class OneBatchLoader:
            def __init__(self, batch):
                self.batch = batch
                self.dataset = [object()]

            def __len__(self):
                return 1

            def __iter__(self):
                yield self.batch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_id": 10,
                        "ann_id": 20,
                        "ref_id": 30,
                        "sent_id": 40,
                        "instances": [
                            {"raw_phrase": "the red car", "class_id": 1}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw_target = {
                "image_id": torch.tensor([10]),
                "ann_id": torch.tensor([20]),
                "ref_id": torch.tensor([30]),
                "sent_id": torch.tensor([40]),
                "caption": "the red car .",
                "stage_a_caption": "car .",
                "boxes": torch.tensor([[0.70, 0.70, 0.20, 0.20]]),
            }
            targets = [{"boxes": raw_target["boxes"].clone()}]
            patch = _patch_outputs()
            patch["stage_a_captions"] = ["car ."]
            settings = _routed_settings(
                root, {"car": "max_patch_p025_w003125_v1"}
            )
            records_dir = root / "records"
            with (
                mock.patch.object(
                    ref_eval,
                    "_build_loader",
                    return_value=OneBatchLoader((None, [raw_target])),
                ),
                mock.patch.object(
                    ref_eval, "_forward", return_value=(patch, targets)
                ),
                mock.patch.object(
                    ref_eval,
                    "_forward_external_gdino_rank_adapter",
                    return_value=(_external_outputs(), targets),
                ),
            ):
                rows = ref_eval.evaluate_dataset(
                    cfg=types.SimpleNamespace(),
                    model=object(),
                    ckpt_path=settings["patch_checkpoints"][0]["path"],
                    datasetinfo={"anno": str(manifest_path)},
                    dataset_name="refcoco_val",
                    device=torch.device("cpu"),
                    betas=[0.0],
                    topks=[1],
                    batch_size=1,
                    num_workers=0,
                    seed=42,
                    amp=False,
                    max_batches=0,
                    max_images=0,
                    log_every=0,
                    records_output_dir=records_dir,
                    formal_external_rank_transfer_settings=settings,
                    external_gdino_model=object(),
                    external_gdino_cfg=types.SimpleNamespace(
                        stage_b_gdino_score_adapter=True
                    ),
                )
            self.assertEqual(rows[0]["formal_route_version"], 2)
            records_path = Path(rows[0]["records_jsonl"])
            self.assertEqual(
                rows[0]["records_sha256"], ref_eval._sha256_file(records_path)
            )
            record = json.loads(records_path.read_text(encoding="utf-8"))
            self.assertIn("all_query_best_iou", record)
            self.assertEqual(
                record["caption_route_descriptor_id"],
                "max_patch_p025_w003125_v1",
            )

    def test_mixed_batch_routes_before_target_and_unknown_uses_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            override_id = "max_patch_p025_w003125_v1"
            settings = _routed_settings(root, {"person": override_id})
            patch = _repeat_outputs(_patch_outputs(), 2)
            patch["stage_a_captions"] = [" Person.__ ", "unknown object ."]
            external = _repeat_outputs(_external_outputs(), 2)
            accumulator = ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                settings,
                manifest=_manifest(root, count=2),
                run_prefix="patch",
            )
            accumulator.update(
                patch,
                external,
                [
                    _target([0.70, 0.70, 0.20, 0.20]),
                    _target([0.20, 0.20, 0.10, 0.10]),
                ],
            )
            first, second = accumulator.eval_records
            self.assertEqual(first["caption_route_caption"], "person")
            self.assertEqual(first["caption_route_descriptor_id"], override_id)
            self.assertIsInstance(first["winner_candidate_index"], int)
            self.assertIsInstance(first["winner_patch_query_index"], int)
            self.assertEqual(
                second["caption_route_descriptor_id"],
                route_tool.DEFAULT_DESCRIPTOR_ID,
            )
            self.assertTrue(second["caption_route_used_default"])
            self.assertIsNone(second["winner_candidate_index"])
            self.assertIsNone(second["winner_patch_query_index"])
            self.assertEqual(second["matched_external_query_index"], 0)
            self.assertEqual(
                accumulator.results()[0]["route_counts_by_descriptor"][override_id],
                1,
            )

    def test_target_changes_do_not_change_route_winner_or_selected_box(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor_id = "max_external_p05_w0046875_v1"
            settings = _routed_settings(root, {"car": descriptor_id})
            patch = _patch_outputs()
            patch["stage_a_captions"] = ["Car ."]
            external = _external_outputs()
            records = []
            for target in (
                _target([0.70, 0.70, 0.20, 0.20]),
                {
                    "boxes": torch.tensor([[0.05, 0.05, 0.03, 0.03]]),
                    "labels": torch.tensor([999]),
                },
            ):
                accumulator = (
                    ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                        settings,
                        manifest=_manifest(root, count=1),
                        run_prefix="patch",
                    )
                )
                accumulator.update(patch, external, [target])
                records.append(accumulator.eval_records[0])
            evidence_fields = (
                "caption_route_caption",
                "caption_route_descriptor_id",
                "caption_route_output_box_source",
                "winner_candidate_index",
                "winner_patch_query_index",
                "matched_external_query_index",
                "selected_box",
            )
            self.assertEqual(
                {key: records[0][key] for key in evidence_fields},
                {key: records[1][key] for key in evidence_fields},
            )
            matched_query = records[0]["matched_external_query_index"]
            expected_external_box = ref_eval._normalized_cxcywh_to_xyxy(
                external["pred_boxes"], name="expected external boxes"
            )[0, matched_query]
            self.assertEqual(
                records[0]["selected_box"],
                [float(value) for value in expected_external_box.tolist()],
            )

    def test_missing_caption_fails_before_targets_are_read(self):
        class ExplodingTargets:
            def __len__(self):
                raise AssertionError("targets were read before caption routing failed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accumulator = ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                _routed_settings(root, {}),
                manifest=_manifest(root, count=1),
                run_prefix="patch",
            )
            with self.assertRaisesRegex(ValueError, "stage_a_captions"):
                accumulator.update(
                    _patch_outputs(), _external_outputs(), ExplodingTargets()
                )

    def test_each_registry_descriptor_matches_diagnostic_top1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = _target([0.70, 0.70, 0.20, 0.20])
            for descriptor_id in route_tool.DESCRIPTOR_REGISTRY:
                with self.subTest(descriptor_id=descriptor_id):
                    overrides = (
                        {}
                        if descriptor_id == route_tool.DEFAULT_DESCRIPTOR_ID
                        else {"car": descriptor_id}
                    )
                    settings = _routed_settings(root, overrides)
                    patch = _patch_outputs()
                    patch["stage_a_captions"] = ["car ."]
                    external = _external_outputs()
                    formal = (
                        ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                            settings,
                            manifest=_manifest(root, count=1),
                            run_prefix="patch",
                        )
                    )
                    formal.update(patch, external, [target])
                    formal_iou = formal.eval_records[0]["top1_iou"]

                    if descriptor_id == route_tool.DEFAULT_DESCRIPTOR_ID:
                        diagnostic_settings = {
                            **settings,
                            "fixed_grid": [
                                {
                                    "transfer_mode": "nearest_iou",
                                    "iou_power": None,
                                    "patch_weight": 1.0,
                                    "text_weight": 1.0,
                                }
                            ],
                            "transfer_modes": ["nearest_iou"],
                            "iou_powers": [],
                            "include_external_gdino_base_identity": True,
                        }
                        diagnostic = (
                            ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
                                diagnostic_settings
                            )
                        )
                        diagnostic.update(patch, external, [target])
                        row = next(
                            row
                            for row in diagnostic.results()
                            if row.get("diagnostic_descriptor_kind")
                            == ref_eval._EXTERNAL_GDINO_BASE_IDENTITY_KIND
                        )
                    else:
                        descriptor_grid = _descriptor_grid(descriptor_id)
                        diagnostic_settings = {
                            **settings,
                            "fixed_grid": [descriptor_grid],
                            "transfer_modes": [descriptor_grid["transfer_mode"]],
                            "iou_powers": (
                                []
                                if descriptor_grid["iou_power"] is None
                                else [descriptor_grid["iou_power"]]
                            ),
                        }
                        diagnostic = (
                            ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
                                diagnostic_settings
                            )
                        )
                        diagnostic.update(patch, external, [target])
                        row = diagnostic.results()[0]
                    self.assertEqual(formal_iou, row["mean_iou_top1"])

    def test_all_query_oracle_exactly_matches_external_900_query_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _routed_settings(root, {})
            patch = _patch_outputs()
            patch["stage_a_captions"] = ["unknown ."]
            external = _external_outputs()
            target = _target([0.70, 0.70, 0.20, 0.20])
            accumulator = ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                settings,
                manifest=_manifest(root, count=1),
                run_prefix="patch",
            )
            accumulator.update(patch, external, [target])
            external_xyxy = ref_eval.box_ops.box_cxcywh_to_xyxy(
                external["pred_boxes"].detach().float()
            ).clamp(0.0, 1.0)
            gt = ref_eval.box_ops.box_cxcywh_to_xyxy(target["boxes"])[0]
            expected = float(
                ref_eval.box_ops.box_iou(external_xyxy[0], gt.view(1, 4))[0]
                .view(-1)
                .max()
                .item()
            )
            record = accumulator.eval_records[0]
            self.assertEqual(record["all_query_best_iou"], expected)
            self.assertIn("patch_candidate_oracle_iou", record)
            baseline = text_ref_eval.RefCocoTextAccumulator(
                [1],
                manifest=_manifest(root, count=1),
                run_id="baseline",
            )
            baseline.update(
                {"pred_boxes": external["pred_boxes"]},
                [target],
                query_scores=external["stage_b_gdino_base_score"],
            )
            self.assertEqual(
                record["all_query_best_iou"],
                baseline.eval_records[0]["all_query_best_iou"],
            )

    def test_summary_recomputes_metrics_and_rejects_record_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _routed_settings(root, {})
            rows = []
            for split_index, dataset in enumerate(REF_SPLIT_ORDER):
                manifest = EvalManifest(
                    path=root / f"{dataset}.manifest.jsonl",
                    task="ref",
                    manifest_key=f"ref:{dataset}",
                    split=dataset,
                    sha256=f"{split_index + 1:x}" * 64,
                    rows=[
                        {
                            "image_id": 10 + split_index,
                            "ann_id": 20 + split_index,
                            "ref_id": 30 + split_index,
                            "sent_id": 40 + split_index,
                            "instances": [{"raw_phrase": "unknown"}],
                        }
                    ],
                )
                patch = _patch_outputs()
                patch["stage_a_captions"] = ["unknown ."]
                accumulator = (
                    ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                        settings,
                        manifest=manifest,
                        run_prefix="patch",
                    )
                )
                accumulator.update(
                    patch,
                    _external_outputs(),
                    [_target([0.20, 0.20, 0.10, 0.10])],
                )
                row = accumulator.results()[0]
                row.update(
                    {
                        "run_id": accumulator.run_id,
                        "dataset": dataset,
                        "batch_size": 1,
                        "num_workers": 0,
                        "amp": False,
                        "seed": settings["evaluation_protocol"]["split_seeds"][
                            dataset
                        ],
                        "checkpoint": settings["patch_checkpoints"][0]["path"],
                        "manifest_sha256": manifest.sha256,
                        "manifest_n": 1,
                        "invalid_records": 0,
                        "patch_checkpoint_sha256": "2" * 64,
                        "patch_config_sha256": "1" * 64,
                        "external_gdino_checkpoint_sha256": "4" * 64,
                        "external_gdino_config_sha256": "3" * 64,
                        "external_gdino_rank_score_key": (
                            "stage_b_gdino_rank_score"
                        ),
                        "external_gdino_query_count": 900,
                        "transfer_contract_version": 2,
                        "external_transfer_artifact_file_sha256": "b" * 64,
                    }
                )
                records_path = root / f"{dataset}.records.jsonl"
                write_eval_records(records_path, accumulator.eval_records)
                row.update(
                    {
                        "records_jsonl": str(records_path),
                        "records_sha256": ref_eval._sha256_file(records_path),
                        "records_size_bytes": records_path.stat().st_size,
                    }
                )
                rows.append(row)

            metadata = (
                ref_eval._formal_routed_external_rank_transfer_summary_metadata(
                    settings, rows
                )
            )
            self.assertEqual(
                metadata["evaluation_kind"],
                "formal_external_gdino_canonical_caption_route_v2",
            )
            self.assertEqual(
                metadata["route_counts_by_descriptor"][
                    route_tool.DEFAULT_DESCRIPTOR_ID
                ],
                len(REF_SPLIT_ORDER),
            )

            rows[0]["amp"] = True
            with self.assertRaisesRegex(ValueError, "identity drift"):
                ref_eval._formal_routed_external_rank_transfer_summary_metadata(
                    settings, rows
                )
            rows[0]["amp"] = False
            rows[0]["acc50"] = 0.0
            with self.assertRaisesRegex(ValueError, "metric drift"):
                ref_eval._formal_routed_external_rank_transfer_summary_metadata(
                    settings, rows
                )
            rows[0]["acc50"] = 1.0
            records_path = Path(rows[0]["records_jsonl"])
            record = json.loads(records_path.read_text(encoding="utf-8"))
            record["caption_route_descriptor_id"] = "top_query_patch_w0_v1"
            records_path.write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )
            rows[0]["records_sha256"] = ref_eval._sha256_file(records_path)
            rows[0]["records_size_bytes"] = records_path.stat().st_size
            with self.assertRaisesRegex(ValueError, "descriptor routing drifted"):
                ref_eval._formal_routed_external_rank_transfer_summary_metadata(
                    settings, rows
                )
            records_path.write_text(
                records_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "records identity drifted"):
                ref_eval._formal_routed_external_rank_transfer_summary_metadata(
                    settings, rows
                )


if __name__ == "__main__":
    unittest.main()
