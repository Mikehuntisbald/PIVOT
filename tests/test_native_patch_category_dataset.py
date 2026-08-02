import copy
import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from datasets.patch_episode import (
    PatchEpisodeConfig,
    PatchEpisodeJsonlDataset,
    _validate_native_patch_category_dataset_binding,
    _validate_native_patch_category_meta,
    build_patch_episode,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_canonical_payload(payload: dict) -> None:
    payload.pop("canonical_payload_sha256", None)
    payload["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


class _FakeEncoding:
    def __init__(self, text: str) -> None:
        self.spans = [match.span() for match in re.finditer(r"\S+", text)]

    def char_to_token(self, offset: int):
        for token_index, (start, end) in enumerate(self.spans):
            if start <= offset < end:
                return token_index
        return None


class _FakeFastTokenizer:
    is_fast = True

    def __call__(self, text: str, **_kwargs) -> _FakeEncoding:
        return _FakeEncoding(text)


class NativePatchCategoryDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.support = self.root / "support.jpg"
        Image.new("RGB", (8, 8), color=(50, 100, 150)).save(self.support)
        support_sha = _sha256(self.support)
        self.meta = {
            "image_id": 10,
            "instances": [
                {
                    "bbox": [0.0, 0.0, 4.0, 4.0],
                    "category_complete_primary": True,
                    "class_id": 3,
                    "raw_phrase": "the full text expression",
                },
                {
                    "bbox": [4.0, 4.0, 2.0, 2.0],
                    "category_complete_auxiliary": True,
                    "class_id": 3,
                },
            ],
            "native_patch_category_group_id": "sealed-group",
            "native_patch_category_variant_index": 0,
            "primary_support_instance_index": 0,
            "query_image_witness": {
                "content_sha256": "1" * 64,
                "path": "/query.jpg",
                "size_bytes": 10,
                "source_filename": "COCO_train2014_000000000010.jpg",
            },
            "stage_b_native_patch_category_d1": True,
            "stage_b_native_patch_category_d1_schema": (
                "pivot.stageb.native_patch_category_d1_row/v1"
            ),
            "stage_b_u2_category_complete": True,
            "stage_b_u2_category_complete_schema": (
                "pivot.stageb.u2_category_complete_ref/v1"
            ),
            "support_patch_witness": {
                "candidate_id": "candidate",
                "class_assignment": "sealed_cache_identity_v1",
                "class_id": 3,
                "coco_id": 20,
                "content_sha256": support_sha,
                "path": str(self.support),
                "selection_priority_sha256": "2" * 64,
                "size_bytes": self.support.stat().st_size,
                "source": "lvis",
                "source_cache_class_id": 3,
                "source_class": "class",
                "source_image_id": 20,
                "source_image_identity": "coco_numeric_id:20",
                "source_row_number": 1,
                "source_row_sha256": "3" * 64,
                "support_partition_receipt_sha256": "4" * 64,
                "train_filtered": True,
            },
        }

    def tearDown(self) -> None:
        self.context.cleanup()

    def _write_d2_binding(self, fixture: dict) -> None:
        manifest = fixture["manifest"]
        manifest.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for row in fixture["rows"]
            ),
            encoding="utf-8",
        )
        output = fixture["receipt"]["splits"]["train"]["refcoco"]["output"]
        output.update(
            {
                "path": str(manifest),
                "rows": len(fixture["rows"]),
                "sha256": _sha256(manifest),
                "size_bytes": manifest.stat().st_size,
            }
        )
        _seal_canonical_payload(fixture["receipt"])
        receipt_path = fixture["receipt_path"]
        receipt_path.write_text(
            json.dumps(fixture["receipt"], sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fixture["datasetinfo"].update(
            {
                "stage_b_native_patch_category_manifest_sha256": _sha256(
                    manifest
                ),
                "stage_b_native_patch_category_receipt_sha256": _sha256(
                    receipt_path
                ),
            }
        )

    def _make_d2_fixture(self, name: str) -> dict:
        fixture_root = self.root / name
        fixture_root.mkdir()
        rows = []
        for row_index, weight in enumerate((0.5, 1.5), start=1):
            query = fixture_root / f"query-{row_index}.jpg"
            Image.new(
                "RGB",
                (8, 8),
                color=(10 * row_index, 20 * row_index, 30 * row_index),
            ).save(query)
            row = copy.deepcopy(self.meta)
            for legacy_key in (
                "native_patch_category_variant_index",
                "stage_b_native_patch_category_d1",
                "stage_b_native_patch_category_d1_schema",
                "stage_b_u2_category_complete",
                "stage_b_u2_category_complete_schema",
            ):
                row.pop(legacy_key, None)
            group_id = f"sealed-group-{row_index}"
            source_identity = str(4 + row_index) * 64
            rotation_contract = (
                "pivot.stageb.native_patch_category_d2.support_rotation/v1"
            )
            rotation_key = hashlib.sha256(
                json.dumps(
                    {
                        "namespace": rotation_contract,
                        "group_id": group_id,
                        "source_identity_sha256": source_identity,
                    },
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest()
            support_witness = row["support_patch_witness"]
            support_witness.pop("selection_priority_sha256")
            support_witness.update(
                {
                    "rotation_key_sha256": rotation_key,
                    "rotation_offset": 2,
                    "rotation_pool_size": 7,
                    "rotation_selected_index": 3,
                    "rotation_start_index": 1,
                    "selection_contract": rotation_contract,
                }
            )
            row.update(
                {
                    "filename": query.name,
                    "image_id": 10 + row_index,
                    "instances": [
                        {
                            "bbox": [0.0, 0.0, 4.0, 4.0],
                            "category_complete_primary": True,
                            "class_id": 3,
                            "head_phrase": "class",
                            "raw_phrase": "red class",
                        }
                    ],
                    "native_patch_category_class_id": 3,
                    "native_patch_category_group_id": group_id,
                    "native_patch_category_source_dataset": "refcoco",
                    "native_patch_category_source_group_expression_count": 1,
                    "native_patch_category_source_identity_sha256": source_identity,
                    "native_patch_category_source_line_number": row_index,
                    "native_patch_category_source_mix_weight": 2,
                    "native_patch_category_sampling_contract": (
                        "source_mix_2_2_1_group_dedup_capped_sqrt_class_v1"
                    ),
                    "native_patch_category_sampling_weight": weight,
                    "query_image_witness": {
                        "content_sha256": _sha256(query),
                        "path": str(query),
                        "size_bytes": query.stat().st_size,
                        "source_filename": query.name,
                    },
                    "stage_b_native_patch_category_d2": True,
                    "stage_b_native_patch_category_d2_schema": (
                        "pivot.stageb.native_patch_category_d2_row/v1"
                    ),
                }
            )
            rows.append(row)

        support_receipt = {
            "schema": "pivot.stageb.data_driven.support_partition_receipt/v1",
            "alias_bridges": [],
            "invariants": {
                "alias_bridges_are_unique_canonical_metadata_matches": True,
                "alias_bridges_reuse_only_filtered_base_paths": True,
            },
        }
        _seal_canonical_payload(support_receipt)
        support_receipt_path = fixture_root / "support-receipt.json"
        support_receipt_path.write_text(
            json.dumps(support_receipt, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = fixture_root / "refcoco-train.jsonl"
        receipt_path = fixture_root / "receipt.json"
        receipt = {
            "schema": "pivot.stageb.native_patch_category_d2_receipt/v1",
            "row_schema": "pivot.stageb.native_patch_category_d2_row/v1",
            "inputs": {
                "support_partition_receipt": {
                    "path": str(support_receipt_path),
                    "sha256": _sha256(support_receipt_path),
                    "size_bytes": support_receipt_path.stat().st_size,
                }
            },
            "sampling_contract": {
                "name": "source_mix_2_2_1_group_dedup_capped_sqrt_class_v1",
                "source_mix_weights": {
                    "refcoco": 2,
                    "refcocoplus": 2,
                    "refcocog": 1,
                },
            },
            "splits": {
                "train": {
                    "refcoco": {
                        "mix_weight": 2,
                        "sampling_weight_mean": 1.0,
                        "output": {},
                    }
                }
            },
            "invariants": {"fixture_is_sealed": True},
        }
        datasetinfo = {
            "root": str(fixture_root),
            "anno": str(manifest),
            "anno_cache": False,
            "anno_cache_write": False,
            "build_text_token_masks": True,
            "mix_weight": 2,
            "native_patch_category_row_locked_support": True,
            "neg_episode_prob": 0.0,
            "stage_b_native_patch_category_receipt": str(receipt_path),
            "stage_b_native_patch_category_row_schema": (
                "pivot.stageb.native_patch_category_d2_row/v1"
            ),
            "stage_b_native_patch_category_sampling_contract": (
                "source_mix_2_2_1_group_dedup_capped_sqrt_class_v1"
            ),
            "stage_b_native_patch_category_sampling_weight_field": (
                "native_patch_category_sampling_weight"
            ),
            "stage_b_native_patch_category_source_dataset": "refcoco",
            "stage_b_native_patch_category_split": "train",
            "stage_b_native_patch_category_variant": "d2",
            "strict_sample_identity": True,
            "support_num_patches_max": 1,
            "support_num_patches_min": 1,
            "support_patch_use_embedding": False,
            "tn_balance_sampling": False,
        }
        fixture = {
            "datasetinfo": datasetinfo,
            "manifest": manifest,
            "receipt": receipt,
            "receipt_path": receipt_path,
            "rows": rows,
        }
        self._write_d2_binding(fixture)
        return fixture

    @staticmethod
    def _load_d2_fixture(fixture: dict) -> PatchEpisodeJsonlDataset:
        args = SimpleNamespace(
            output_dir=None,
            stage_b_native_patch_category=True,
        )
        identity_transform = lambda image, target: (image, target)
        with patch(
            "datasets.patch_episode.AutoTokenizer.from_pretrained",
            return_value=_FakeFastTokenizer(),
        ), patch(
            "datasets.patch_episode.make_query_transforms",
            return_value=identity_transform,
        ):
            return build_patch_episode("train", args, fixture["datasetinfo"])

    def test_meta_and_runtime_support_are_exactly_bound(self) -> None:
        _validate_native_patch_category_meta(self.meta, 0)
        dataset = object.__new__(PatchEpisodeJsonlDataset)
        dataset.cfg = PatchEpisodeConfig(
            native_patch_category_row_locked_support=True,
            neg_episode_prob=0.0,
            support_num_patches_min=1,
            support_num_patches_max=1,
            support_patch_use_embedding=False,
            build_text_token_masks=True,
            strict_sample_identity=True,
        )
        dataset.patch_tfm = lambda image: torch.as_tensor(image.size)
        dataset._native_patch_support_sha_cache = {}

        patch = dataset._load_native_patch_category_support(self.meta, 3)

        self.assertTrue(torch.equal(patch, torch.tensor([8, 8])))
        self.support.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "size drifted"):
            dataset._load_native_patch_category_support(self.meta, 3)

    def test_only_receipt_bound_alias_bridge_is_allowed(self) -> None:
        alias_meta = json.loads(json.dumps(self.meta))
        alias_meta["instances"][0]["class_id"] = 1393
        alias_meta["instances"][1]["class_id"] = 1393
        witness = alias_meta["support_patch_witness"]
        witness["class_id"] = 1393
        witness["source_cache_class_id"] = 223
        witness["class_assignment"] = "canonical_compact_alias_bridge_v1"

        _validate_native_patch_category_meta(
            alias_meta, 0, alias_bridges={1393: 223}
        )
        with self.assertRaisesRegex(ValueError, "assignment is not sealed"):
            _validate_native_patch_category_meta(alias_meta, 0)
        with self.assertRaisesRegex(ValueError, "assignment is not sealed"):
            _validate_native_patch_category_meta(
                alias_meta, 0, alias_bridges={1393: 999}
            )

    def test_receipt_binding_fails_closed(self) -> None:
        manifest = self.root / "train.jsonl"
        manifest.write_text(json.dumps(self.meta, sort_keys=True) + "\n")
        manifest_sha = _sha256(manifest)
        support_receipt = {
            "schema": "pivot.stageb.data_driven.support_partition_receipt/v1",
            "alias_bridges": [
                {
                    "candidate_rows": 2,
                    "source_cache_class_id": 223,
                    "target_class_id": 1393,
                }
            ],
            "invariants": {
                "alias_bridges_are_unique_canonical_metadata_matches": True,
                "alias_bridges_reuse_only_filtered_base_paths": True,
            },
        }
        support_receipt["canonical_payload_sha256"] = hashlib.sha256(
            json.dumps(
                support_receipt,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        support_receipt_path = self.root / "support_receipt.json"
        support_receipt_path.write_text(
            json.dumps(support_receipt, sort_keys=True) + "\n"
        )
        receipt = {
            "schema": "pivot.stageb.native_patch_category_d1_receipt/v1",
            "inputs": {
                "support_partition_receipt": {
                    "path": str(support_receipt_path),
                    "sha256": _sha256(support_receipt_path),
                    "size_bytes": support_receipt_path.stat().st_size,
                }
            },
            "splits": {
                "train": {
                    "output": {
                        "path": str(manifest),
                        "rows": 1,
                        "sha256": manifest_sha,
                        "size_bytes": manifest.stat().st_size,
                    }
                }
            },
            "invariants": {"fixture_is_sealed": True},
        }
        receipt["canonical_payload_sha256"] = hashlib.sha256(
            json.dumps(
                receipt,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        datasetinfo = {
            "anno": str(manifest),
            "anno_cache": False,
            "anno_cache_write": False,
            "build_text_token_masks": True,
            "native_patch_category_row_locked_support": True,
            "neg_episode_prob": 0.0,
            "stage_b_native_patch_category_manifest_sha256": manifest_sha,
            "stage_b_native_patch_category_receipt": str(receipt_path),
            "stage_b_native_patch_category_receipt_sha256": _sha256(receipt_path),
            "stage_b_native_patch_category_split": "train",
            "stage_b_native_patch_category_variant": "d1",
            "strict_sample_identity": True,
            "support_num_patches_max": 1,
            "support_num_patches_min": 1,
            "support_patch_use_embedding": False,
        }
        args = SimpleNamespace(stage_b_native_patch_category=True)

        self.assertEqual(
            _validate_native_patch_category_dataset_binding(
                args, datasetinfo, image_set="train"
            ),
            {1393: 223},
        )
        datasetinfo["stage_b_native_patch_category_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "annotation SHA drifted"):
            _validate_native_patch_category_dataset_binding(
                args, datasetinfo, image_set="train"
            )

    def test_d2_loader_exposes_weights_and_only_d2_runtime_marker(self) -> None:
        fixture = self._make_d2_fixture("valid-d2")

        dataset = self._load_d2_fixture(fixture)

        self.assertEqual(dataset.sample_weights, [0.5, 1.5])
        self.assertTrue(
            math.isclose(
                math.fsum(dataset.sample_weights) / len(dataset.sample_weights),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        for row in dataset.metas:
            self.assertTrue(row["stage_b_native_patch_category_d2"])
            self.assertNotIn("stage_b_native_patch_category_d1", row)
            self.assertNotIn("stage_b_native_patch_category_d1_schema", row)
            self.assertNotIn("stage_b_u2_category_complete", row)
            self.assertNotIn("stage_b_u2_category_complete_schema", row)

        _image, target = dataset[0]
        runtime_markers = {
            key
            for key in target
            if key
            in {
                "stage_b_native_patch_category_d2",
                "stage_b_native_patch_category_d1",
                "stage_b_u2_category_complete",
            }
        }
        self.assertEqual(
            runtime_markers,
            {"stage_b_native_patch_category_d2"},
        )
        self.assertTrue(target["stage_b_native_patch_category_d2"].item())

    def test_d2_loader_fails_closed_on_resealed_tampering(self) -> None:
        tamper_cases = {
            "source": lambda fixture: fixture["rows"][0].__setitem__(
                "native_patch_category_source_dataset", "refcocoplus"
            ),
            "mix": lambda fixture: fixture["rows"][0].__setitem__(
                "native_patch_category_source_mix_weight", 1
            ),
            "weight": lambda fixture: fixture["rows"][0].__setitem__(
                "native_patch_category_sampling_weight", 0.75
            ),
            "receipt": lambda fixture: fixture["receipt"][
                "invariants"
            ].__setitem__("fixture_is_sealed", False),
        }
        for name, tamper in tamper_cases.items():
            with self.subTest(name=name):
                fixture = self._make_d2_fixture(f"tampered-{name}")
                tamper(fixture)
                self._write_d2_binding(fixture)

                with self.assertRaises(ValueError):
                    self._load_d2_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
