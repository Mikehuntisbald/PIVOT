import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.test_eval_refcoco_external_gdino_rank_transfer import (
    _external_outputs,
    _patch_outputs,
)
from tests.test_eval_refcoco_external_gdino_routed_formal import (
    _repeat_outputs,
    _routed_settings,
    _target,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import stageb_canonical_caption_route_artifact as route_tool
from tools.stageb_eval_records import EvalManifest, write_eval_records
from tools.stageb_external_rank_transfer_artifact import REF_SPLIT_ORDER
from tools.stageb_fulltext_route_gate_artifact import TOKEN_COUNT_CONTRACT


PERSON_TRANSFER = "max_patch_p025_w003125_v1"
BOWL_TRANSFER = "max_external_p05_w0046875_v1"


def _v3_settings(root: Path):
    settings = _routed_settings(
        root,
        {
            "person": PERSON_TRANSFER,
            "bowl": BOWL_TRANSFER,
            "doughnut": BOWL_TRANSFER,
        },
    )
    canonical_route_selection = dict(settings["route_selection"])
    canonical_route_policy = dict(settings["route_policy"])
    gate_artifact = {
        "path": str((root / "fulltext-gate.json").resolve()),
        "size_bytes": 30,
        "sha256": "6" * 64,
    }
    routed_v2_artifact = {
        "artifact": {
            "path": str((root / "formal-v2.json").resolve()),
            "size_bytes": 31,
            "sha256": "7" * 64,
        },
        "artifact_identity": {"algorithm": "fixture", "sha256": "8" * 64},
        "artifact_schema": "fixture-v2",
        "artifact_kind": "fixture-v2",
    }
    conditional = {
        "person": {
            "descriptor_id": PERSON_TRANSFER,
            "fallback_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
            "predicate": {
                "kind": "full_expression_lexical_token_count_lte",
                "max_tokens": 7,
                "token_count_contract": dict(TOKEN_COUNT_CONTRACT),
            },
        }
    }
    unconditional = {"bowl": BOWL_TRANSFER, "doughnut": BOWL_TRANSFER}
    settings.update(
        {
            "formal_artifact_version": 3,
            "artifact_identity": {"algorithm": "fixture", "sha256": "9" * 64},
            "artifact": {
                "path": str((root / "formal-v3.json").resolve()),
                "size_bytes": 32,
                "sha256": "a" * 64,
            },
            "evaluation_protocol": {
                **settings["evaluation_protocol"],
                "batch_size": 1,
                "num_workers": 0,
                "amp": False,
            },
            "canonical_route_selection": canonical_route_selection,
            "canonical_route_policy": canonical_route_policy,
            "canonical_route_policy_sha256": "e" * 64,
            "route_selection": {
                "descriptor_registry": dict(route_tool.DESCRIPTOR_REGISTRY),
                "policy": {
                    "default_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
                    "unconditional_overrides": unconditional,
                    "conditional_overrides": conditional,
                },
                "policy_sha256": "f" * 64,
            },
            "route_policy": {
                "default_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
                "unconditional_overrides": unconditional,
                "conditional_overrides": conditional,
            },
            "route_policy_sha256": "f" * 64,
            "route_overrides": unconditional,
            "route_unconditional_overrides": unconditional,
            "route_conditional_overrides": conditional,
            "fulltext_route_gate": {
                "artifact": gate_artifact,
                "artifact_identity": {
                    "algorithm": "fixture",
                    "sha256": "b" * 64,
                },
                "route": {
                    "default_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
                    "unconditional_overrides": unconditional,
                    "conditional_overrides": conditional,
                },
            },
            "fulltext_route_gate_artifact": gate_artifact,
            "fulltext_route_gate_artifact_identity": {
                "algorithm": "fixture",
                "sha256": "b" * 64,
            },
            "routed_v2_artifact": routed_v2_artifact,
            "routed_v2_artifact_identity": dict(
                routed_v2_artifact["artifact_identity"]
            ),
        }
    )
    return settings


def _manifest(root: Path, count: int, *, split: str = "refcoco_val"):
    return EvalManifest(
        path=root / f"{split}.manifest.jsonl",
        task="ref",
        manifest_key=f"ref:{split}",
        split=split,
        sha256="c" * 64,
        rows=[
            {
                "image_id": 10 + index,
                "ann_id": 20 + index,
                "ref_id": 30 + index,
                "sent_id": 40 + index,
                "instances": [{"raw_phrase": "person", "class_id": 2}],
            }
            for index in range(count)
        ],
    )


class FormalFullTextGatedExternalGDINOTest(unittest.TestCase):
    def test_person_gate_allows_same_caption_to_select_two_descriptors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _v3_settings(root)
            captions = ["person .", "person .", "bowl .", "unknown ."]
            expressions = [
                "the red PERSON beside a blue car .",
                "the red person beside a blue car today .",
                "the blue bowl .",
                "an unknown object .",
            ]
            patch = _repeat_outputs(_patch_outputs(), len(captions))
            patch["stage_a_captions"] = captions
            patch[ref_eval._FULLTEXT_CAPTIONS_OUTPUT_KEY] = expressions
            external = _repeat_outputs(_external_outputs(), len(captions))
            accumulator = ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                settings,
                manifest=_manifest(root, len(captions)),
                run_prefix="patch",
            )
            accumulator.update(
                patch,
                external,
                [_target([0.70, 0.70, 0.20, 0.20])] * len(captions),
            )

            records = accumulator.eval_records
            self.assertEqual(
                [row["caption_route_descriptor_id"] for row in records],
                [
                    PERSON_TRANSFER,
                    route_tool.DEFAULT_DESCRIPTOR_ID,
                    BOWL_TRANSFER,
                    route_tool.DEFAULT_DESCRIPTOR_ID,
                ],
            )
            self.assertEqual(records[0]["fulltext_route_gate_token_count"], 7)
            self.assertEqual(records[1]["fulltext_route_gate_token_count"], 8)
            self.assertTrue(records[0]["fulltext_route_gate_predicate_matched"])
            self.assertTrue(records[1]["fulltext_route_gate_fallback_matched"])
            self.assertIsInstance(records[0]["winner_candidate_index"], int)
            self.assertIsNone(records[1]["winner_candidate_index"])
            self.assertIsInstance(records[1]["matched_external_query_index"], int)
            for row in records:
                self.assertEqual(len(row["selected_box"]), 4)
                self.assertEqual(row["caption_route_contract_version"], 3)
                self.assertEqual(
                    row["fulltext_route_gate_selected_descriptor_id"],
                    row["caption_route_descriptor_id"],
                )
            person_counts = accumulator.results()[0]["route_counts_by_caption"][
                "person"
            ]
            self.assertEqual(
                person_counts["descriptor_counts"],
                {PERSON_TRANSFER: 1, route_tool.DEFAULT_DESCRIPTOR_ID: 1},
            )

    def test_missing_validated_expression_fails_before_targets_are_read(self):
        class ExplodingTargets:
            def __len__(self):
                raise AssertionError("targets read before full-expression routing")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch = _patch_outputs()
            patch["stage_a_captions"] = ["person ."]
            accumulator = ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                _v3_settings(root),
                manifest=_manifest(root, 1),
                run_prefix="patch",
            )
            with self.assertRaisesRegex(ValueError, "manifest-validated full expression"):
                accumulator.update(patch, _external_outputs(), ExplodingTargets())

    def test_evaluate_dataset_attaches_only_provenance_validated_expression(self):
        class OneBatchLoader:
            dataset = [object()]

            def __init__(self, batch):
                self.batch = batch

            def __len__(self):
                return 1

            def __iter__(self):
                yield self.batch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _v3_settings(root)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_id": 10,
                        "ann_id": 20,
                        "ref_id": 30,
                        "sent_id": 40,
                        "instances": [
                            {"raw_phrase": "the red person", "class_id": 2}
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
                "caption": "the red person .",
                "stage_a_caption": "person .",
                "boxes": torch.tensor([[0.70, 0.70, 0.20, 0.20]]),
            }
            patch = _patch_outputs()
            patch["stage_a_captions"] = ["person ."]
            targets = [{"boxes": raw_target["boxes"].clone()}]
            records_dir = root / "records"
            with (
                mock.patch.object(
                    ref_eval,
                    "_build_loader",
                    return_value=OneBatchLoader((None, [raw_target])),
                ),
                mock.patch.object(ref_eval, "_forward", return_value=(patch, targets)),
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
            record = json.loads(Path(rows[0]["records_jsonl"]).read_text())
            self.assertEqual(rows[0]["formal_route_version"], 3)
            self.assertEqual(
                record["fulltext_route_gate_full_expression"],
                "the red person .",
            )

    def test_summary_rebuilds_v3_gate_and_rejects_token_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _v3_settings(root)
            rows = []
            for split_index, dataset in enumerate(REF_SPLIT_ORDER):
                manifest = _manifest(root, 2, split=dataset)
                patch = _repeat_outputs(_patch_outputs(), 2)
                patch["stage_a_captions"] = ["person .", "person ."]
                patch[ref_eval._FULLTEXT_CAPTIONS_OUTPUT_KEY] = [
                    "short red person .",
                    "the person wearing a bright red shirt beside the old table .",
                ]
                accumulator = (
                    ref_eval.FormalRoutedExternalGDINORankTransferAccumulator(
                        settings, manifest=manifest, run_prefix="patch"
                    )
                )
                accumulator.update(
                    patch,
                    _repeat_outputs(_external_outputs(), 2),
                    [_target([0.70, 0.70, 0.20, 0.20])] * 2,
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
                        "manifest_n": 2,
                        "invalid_records": 0,
                        "patch_checkpoint_sha256": "2" * 64,
                        "patch_config_sha256": "1" * 64,
                        "external_gdino_checkpoint_sha256": "4" * 64,
                        "external_gdino_config_sha256": "3" * 64,
                        "external_gdino_rank_score_key": (
                            "stage_b_gdino_rank_score"
                        ),
                        "external_gdino_query_count": 900,
                        "transfer_contract_version": 3,
                        "external_transfer_artifact_file_sha256": "a" * 64,
                    }
                )
                records_path = root / f"{split_index}.records.jsonl"
                write_eval_records(records_path, accumulator.eval_records)
                row.update(
                    {
                        "records_jsonl": str(records_path),
                        "records_sha256": ref_eval._sha256_file(records_path),
                        "records_size_bytes": records_path.stat().st_size,
                    }
                )
                rows.append(row)

            metadata = ref_eval._formal_routed_external_rank_transfer_summary_metadata(
                settings, rows
            )
            self.assertEqual(
                metadata["evaluation_kind"],
                "formal_external_gdino_fulltext_gated_caption_route_v3",
            )
            self.assertEqual(
                metadata["route_counts_by_caption"]["person"][
                    "descriptor_counts"
                ],
                {
                    PERSON_TRANSFER: len(REF_SPLIT_ORDER),
                    route_tool.DEFAULT_DESCRIPTOR_ID: len(REF_SPLIT_ORDER),
                },
            )

            records_path = Path(rows[0]["records_jsonl"])
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            records[0]["fulltext_route_gate_token_count"] += 1
            records_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n",
                encoding="utf-8",
            )
            rows[0]["records_sha256"] = ref_eval._sha256_file(records_path)
            rows[0]["records_size_bytes"] = records_path.stat().st_size
            with self.assertRaisesRegex(ValueError, "gate drifted"):
                ref_eval._formal_routed_external_rank_transfer_summary_metadata(
                    settings, rows
                )


if __name__ == "__main__":
    unittest.main()
