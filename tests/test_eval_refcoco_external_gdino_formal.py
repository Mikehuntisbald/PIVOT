import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.test_eval_refcoco_external_gdino_rank_transfer import (
    _external_outputs,
    _patch_cfg,
    _patch_outputs,
)
from tools import eval_refcoco_stageb as ref_eval
from tools.stageb_eval_records import EvalManifest
from tools.stageb_external_rank_transfer_artifact import (
    CAPTION_PROVENANCE_CONTRACT,
    SPLIT_SEED_PROTOCOL,
    stable_ref_split_seed_map,
)


def _manifest(root: Path, count: int = 2) -> EvalManifest:
    rows = [
        {
            "image_id": 10 + index,
            "ann_id": 20 + index,
            "ref_id": 30 + index,
            "sent_id": 40 + index,
            "instances": [{"raw_phrase": "the red car"}],
        }
        for index in range(count)
    ]
    return EvalManifest(
        path=root / "manifest.jsonl",
        task="ref",
        manifest_key="ref:refcoco_val",
        split="refcoco_val",
        sha256="a" * 64,
        rows=rows,
    )


def _formal_settings(root: Path, *, candidate_topk: int = 2):
    settings = {
        "transfer_modes": ["max_score_iou_power"],
        "iou_powers": [0.5],
        "patch_weights": [0.046875],
        "text_weights": [1.0],
        "candidate_topk": candidate_topk,
        "contract_patch_rank_weight": 1.0,
        "external_query_count": 900,
        "artifact_identity": {
            "algorithm": "sha256-canonical-json-excluding-artifact-identity-v1",
            "sha256": "b" * 64,
        },
        "artifact": {
            "path": str((root / "artifact.json").resolve()),
            "size_bytes": 123,
            "sha256": "c" * 64,
        },
        "patch_config": {
            "path": str((root / "patch.py").resolve()),
            "size_bytes": 10,
            "sha256": "d" * 64,
        },
        "patch_checkpoints": [
            {
                "path": str((root / "patch.pth").resolve()),
                "size_bytes": 20,
                "sha256": "e" * 64,
            }
        ],
        "external_config": {
            "path": str((root / "external.py").resolve()),
            "size_bytes": 11,
            "sha256": "f" * 64,
        },
        "external_checkpoint": {
            "path": str((root / "external.pth").resolve()),
            "size_bytes": 21,
            "sha256": "1" * 64,
        },
        "caption_provenance": dict(CAPTION_PROVENANCE_CONTRACT),
        "evaluation_protocol": {
            "seed_protocol": SPLIT_SEED_PROTOCOL,
            "base_seed": 42,
            "split_seed_stride": 100000,
            "canonical_split_order": [
                "refcoco_val",
                "refcoco_testA",
                "refcoco_testB",
                "refcocop_val",
                "refcocop_testA",
                "refcocop_testB",
                "refcocog_val",
                "refcocog_test",
            ],
            "split_seeds": stable_ref_split_seed_map(42),
        },
        "transfer_contract": {
            "mode": "max_score_iou_power",
            "iou_power": 0.5,
            "patch_weight": 0.046875,
            "text_weight": 1.0,
        },
    }
    settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
        settings
    )
    return settings


def _patch_with_caption():
    outputs = _patch_outputs()
    outputs["stage_a_captions"] = ["car ."]
    return outputs


def _formal_row(
    settings,
    root: Path,
    dataset: str,
    *,
    count: int = 2,
    run_id: str = "patch:formal",
):
    records_path = root / f"{dataset}.records.jsonl"
    records_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": "stageb-eval-record-v1",
                    "task": "ref",
                    "manifest_index": index,
                    "manifest_sha256": "9" * 64,
                    "manifest_n": count,
                    "run_id": run_id,
                    "external_transfer_artifact_sha256": "b" * 64,
                    "canonical_caption": "car .",
                    "canonical_class_norm": "car",
                }
            )
            for index in range(count)
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "diagnostic_only": False,
        "formal_gate_eligible": True,
        "formal_transfer_mode": "max_score_iou_power",
        "formal_iou_power": 0.5,
        "formal_patch_weight": 0.046875,
        "formal_text_weight": 1.0,
        "candidate_topk": int(settings["candidate_topk"]),
        "num_expressions": count,
        "acc50": 0.5,
        "acc25": 0.5,
        "mean_iou_top1": 0.5,
        "candidate_oracle_recall50": 1.0,
        "run_id": run_id,
        "dataset": dataset,
        "seed": settings["evaluation_protocol"]["split_seeds"][dataset],
        "checkpoint": settings["patch_checkpoints"][0]["path"],
        "records_jsonl": str(records_path),
        "manifest_sha256": "9" * 64,
        "manifest_n": count,
        "patch_checkpoint_sha256": "e" * 64,
        "patch_config_sha256": "d" * 64,
        "external_gdino_checkpoint_sha256": "1" * 64,
        "external_gdino_config_sha256": "f" * 64,
        "external_gdino_rank_score_key": "stage_b_gdino_rank_score",
        "external_gdino_query_count": 900,
        "transfer_contract_version": 1,
        "external_transfer_artifact_sha256": "b" * 64,
        "external_transfer_artifact_file_sha256": "c" * 64,
    }


class FormalExternalGDINORankTransferTest(unittest.TestCase):
    def test_main_artifact_only_path_preserves_formal_final_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            settings = _formal_settings(root)
            for key in ("patch_config", "external_config"):
                Path(settings[key]["path"]).write_text("fixture = True\n")
            for record in (
                settings["patch_checkpoints"][0],
                settings["external_checkpoint"],
            ):
                Path(record["path"]).write_bytes(b"fixture")
            artifact_path = root / "artifact.json"
            artifact_path.write_text("{}\n", encoding="utf-8")
            rows_by_dataset = {
                dataset: _formal_row(settings, root, dataset, count=1)
                for dataset in ref_eval.REF_SPLIT_ORDER
            }
            patch_cfg = _patch_cfg()
            external_cfg = types.SimpleNamespace(
                stage_b_gdino_score_adapter=True,
                stage_b_gdino_adapter_merged_eval_only=True,
                stage_b_gdino_adapter_merged_eval_contract_version=1,
                num_queries=900,
                stage_b_v11_fixed_text=False,
                stage_b_v7=False,
            )
            argv = [
                "eval_refcoco_stageb.py",
                "--config",
                settings["patch_config"]["path"],
                "--ckpts",
                settings["patch_checkpoints"][0]["path"],
                "--output_dir",
                str(output_dir),
                "--data_root",
                str(root),
                "--device",
                "cpu",
                "--splits",
                "all",
                "--formal_external_rank_transfer_artifact",
                str(artifact_path),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ref_eval.SLConfig,
                    "fromfile",
                    return_value=patch_cfg,
                ),
                mock.patch.object(
                    ref_eval,
                    "_load_bound_config",
                    side_effect=[patch_cfg, external_cfg],
                ),
                mock.patch.object(
                    ref_eval,
                    "evaluator_settings_from_artifact",
                    return_value=settings,
                ),
                mock.patch.object(
                    ref_eval, "_load_canonical_name_maps", return_value=({}, {})
                ),
                mock.patch.object(ref_eval, "_load_phrase_maps", return_value={}),
                mock.patch.object(
                    ref_eval, "load_holdout_keys", return_value=(set(), set())
                ),
                mock.patch.object(
                    ref_eval,
                    "_build_split_jsonl",
                    return_value=(root / "eval.jsonl", 1),
                ),
                mock.patch.object(
                    ref_eval,
                    "_make_datasetinfo",
                    return_value={"anno": str(root / "eval.jsonl")},
                ),
                mock.patch.object(
                    ref_eval, "_load_bound_model", return_value=object()
                ),
                mock.patch.object(
                    ref_eval,
                    "evaluate_dataset",
                    side_effect=lambda **kwargs: [
                        dict(rows_by_dataset[kwargs["dataset_name"]])
                    ],
                ),
            ):
                ref_eval.main()
            payload = json.loads((output_dir / "summary.json").read_text())
            self.assertIs(payload["formal_gate_eligible"], True)
            self.assertEqual(
                payload["evaluation_kind"],
                "formal_external_gdino_rank_transfer",
            )
            self.assertNotIn(
                "Diagnostic only", (output_dir / "summary.md").read_text()
            )

    def test_evaluate_dataset_writes_canonical_formal_records(self):
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
                        "instances": [{"raw_phrase": "the red car"}],
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
                "boxes": torch.tensor([[0.70, 0.70, 0.20, 0.20]]),
            }
            targets = [{"boxes": raw_target["boxes"].clone()}]
            loader = OneBatchLoader((None, [raw_target]))
            settings = _formal_settings(root)
            records_dir = root / "records"
            with (
                mock.patch.object(ref_eval, "_build_loader", return_value=loader),
                mock.patch.object(
                    ref_eval,
                    "_forward",
                    return_value=(_patch_with_caption(), targets),
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
            self.assertEqual(len(rows), 1)
            self.assertIs(rows[0]["formal_gate_eligible"], True)
            records_path = Path(rows[0]["records_jsonl"])
            self.assertTrue(records_path.is_file())
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["schema"], "stageb-eval-record-v1")
            self.assertEqual(records[0]["canonical_caption"], "car .")
            self.assertEqual(records[0]["manifest_sha256"], rows[0]["manifest_sha256"])

    def test_formal_records_match_diagnostic_same_point_per_example(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _formal_settings(root)
            diagnostic = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
                settings
            )
            formal = ref_eval.FormalExternalGDINORankTransferAccumulator(
                settings,
                manifest=_manifest(root),
                run_prefix="patch",
            )
            target = {
                "boxes": torch.tensor([[0.70, 0.70, 0.20, 0.20]])
            }

            for choose_second in (True, False):
                external = _external_outputs()
                if not choose_second:
                    external["stage_b_gdino_rank_score"][0, 0] = 10.0
                    external["stage_b_gdino_rank_score"][0, 1] = 1.0
                diagnostic.update(_patch_with_caption(), external, [target])
                formal.update(_patch_with_caption(), external, [target])

            key = ref_eval._external_grid_key(settings["fixed_grid"][0])
            diagnostic_iou = diagnostic.per_example_top1_iou[key]
            formal_iou = [record["top1_iou"] for record in formal.eval_records]
            self.assertEqual(diagnostic_iou, formal_iou)
            self.assertEqual([record["correct50"] for record in formal.eval_records], [True, False])
            self.assertEqual(
                formal.results()[0]["acc50"], diagnostic.results()[0]["acc50"]
            )
            for record in formal.eval_records:
                self.assertEqual(record["schema"], "stageb-eval-record-v1")
                self.assertEqual(record["canonical_caption"], "car .")
                self.assertEqual(record["canonical_class_norm"], "car")
                self.assertEqual(
                    record["external_transfer_artifact_sha256"], "b" * 64
                )

    def test_caption_and_topk_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _manifest(root, count=1)
            with self.assertRaisesRegex(ValueError, "caption/manifest provenance drift"):
                ref_eval._validate_formal_external_caption_provenance(
                    [{"caption": "the blue car ."}],
                    manifest,
                    0,
                    settings=_formal_settings(root),
                )

            settings = _formal_settings(root, candidate_topk=50)
            formal = ref_eval.FormalExternalGDINORankTransferAccumulator(
                settings, manifest=manifest, run_prefix="patch"
            )
            with self.assertRaisesRegex(ValueError, "Top-K|Top50|expected 50"):
                formal.update(
                    _patch_with_caption(),
                    _external_outputs(),
                    [{"boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1]])}],
                )

    def test_formal_summary_binds_records_artifact_and_stable_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _formal_settings(root)
            rows = [
                _formal_row(settings, root, dataset)
                for dataset in ref_eval.REF_SPLIT_ORDER
            ]
            metadata = ref_eval._formal_external_rank_transfer_summary_metadata(
                settings, rows
            )
            ref_eval._write_summary(
                root, rows, "acc50", diagnostic_metadata=metadata
            )
            payload = json.loads((root / "summary.json").read_text())
            self.assertIs(payload["formal_gate_eligible"], True)
            self.assertIs(payload["diagnostic_only"], False)
            self.assertEqual(
                payload["fixed_seeds"], stable_ref_split_seed_map(42)
            )
            self.assertNotIn("Diagnostic only", (root / "summary.md").read_text())

    def test_formal_summary_rejects_empty_partial_and_missing_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _formal_settings(root)
            with self.assertRaisesRegex(ValueError, "every canonical Ref8 split"):
                ref_eval._formal_external_rank_transfer_summary_metadata(
                    settings, []
                )

            partial_rows = [_formal_row(settings, root, "refcoco_val")]
            with self.assertRaisesRegex(ValueError, "every canonical Ref8 split"):
                ref_eval._formal_external_rank_transfer_summary_metadata(
                    settings, partial_rows
                )

            rows = [
                _formal_row(settings, root, dataset)
                for dataset in ref_eval.REF_SPLIT_ORDER
            ]
            rows[0]["num_expressions"] = 1
            with self.assertRaisesRegex(ValueError, "evaluation is partial"):
                ref_eval._formal_external_rank_transfer_summary_metadata(
                    settings, rows
                )

            rows[0]["num_expressions"] = rows[0]["manifest_n"]
            Path(rows[0]["records_jsonl"]).unlink()
            with self.assertRaisesRegex(ValueError, "records are missing"):
                ref_eval._formal_external_rank_transfer_summary_metadata(
                    settings, rows
                )

    def test_formal_cli_contract_rejects_partial_capped_or_filtered_runs(self):
        valid = {
            "splits": ["all"],
            "max_batches": 0,
            "max_images": 0,
            "holdout_level": "none",
            "exclude_train_jsonl": [],
            "no_per_example_records": False,
        }
        ref_eval._validate_formal_cli_contract(types.SimpleNamespace(**valid))
        ref_eval._validate_formal_cli_contract(
            types.SimpleNamespace(
                **{**valid, "splits": list(ref_eval.REF_SPLIT_ORDER)}
            )
        )
        invalid = (
            ({"splits": ["refcoco_val"]}, "canonical Ref8 order"),
            ({"max_batches": 1}, "full manifests"),
            ({"max_images": 1}, "full manifests"),
            ({"holdout_level": "image"}, "holdout/exclusion"),
            ({"exclude_train_jsonl": ["train.jsonl"]}, "holdout/exclusion"),
            ({"no_per_example_records": True}, "per-example records"),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    ref_eval._validate_formal_cli_contract(
                        types.SimpleNamespace(**{**valid, **overrides})
                    )

    def test_bound_config_and_model_loads_reject_file_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            config.write_text("fixture = True\n", encoding="utf-8")
            config_record = ref_eval._file_identity(str(config), label="config")

            def mutate_config(_path):
                config.write_text("fixture = False\n", encoding="utf-8")
                return types.SimpleNamespace()

            with (
                mock.patch.object(
                    ref_eval.SLConfig, "fromfile", side_effect=mutate_config
                ),
                self.assertRaisesRegex(ValueError, "file identity drifted"),
            ):
                ref_eval._load_bound_config(config_record, label="formal config")

            config.write_text("fixture = True\n", encoding="utf-8")
            config_record = ref_eval._file_identity(str(config), label="config")
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_record = ref_eval._file_identity(
                str(checkpoint), label="checkpoint"
            )

            def mutate_checkpoint(*_args):
                checkpoint.write_bytes(b"changed checkpoint")
                return object()

            with (
                mock.patch.object(
                    ref_eval, "_load_model", side_effect=mutate_checkpoint
                ),
                self.assertRaisesRegex(ValueError, "file identity drifted"),
            ):
                ref_eval._load_bound_model(
                    types.SimpleNamespace(),
                    checkpoint_record,
                    config_record,
                    torch.device("cpu"),
                    label="formal model",
                )

    def test_formal_cli_rejects_diagnostic_grid_mixing_before_artifact_load(self):
        for diagnostic_args in (
            ["--diagnostic_patch_rank_weights", "1"],
            ["--diagnostic_external_patch_weights", "1"],
        ):
            with self.subTest(diagnostic_args=diagnostic_args):
                argv = [
                    "eval_refcoco_stageb.py",
                    "--config",
                    "patch.py",
                    "--ckpts",
                    "patch.pth",
                    "--device",
                    "cpu",
                    "--splits",
                    "all",
                    "--formal_external_rank_transfer_artifact",
                    "artifact.json",
                    *diagnostic_args,
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(
                        ref_eval.SLConfig, "fromfile", return_value=_patch_cfg()
                    ),
                    self.assertRaisesRegex(ValueError, "cannot be combined"),
                ):
                    ref_eval.main()


if __name__ == "__main__":
    unittest.main()
