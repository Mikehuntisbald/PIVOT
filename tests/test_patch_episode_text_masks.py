import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image
import torch
from transformers import AutoTokenizer

from datasets.patch_episode import (
    PatchEpisodeConfig,
    PatchEpisodeJsonlDataset,
    _LazyJsonlRows,
    _read_jsonl,
    build_patch_episode,
)


class PatchEpisodeTextMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True, local_files_only=True
        )

    def _dataset(self):
        dataset = object.__new__(PatchEpisodeJsonlDataset)
        dataset.cfg = PatchEpisodeConfig(
            build_text_token_masks=True,
            max_text_len=32,
            text_mask_warn_limit=0,
        )
        dataset._text_tokenizer = self.tokenizer
        dataset._text_mask_warn_count = 0
        return dataset

    def test_lazy_jsonl_rows_match_eager_rows_and_blank_line_handling(self):
        rows = [
            {"filename": "a.jpg", "value": "caf\u00e9"},
            {"filename": "b.jpg", "nested": {"values": [2, 3]}},
            {"filename": "c.jpg", "value": 4},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rows.jsonl"
            path.write_text(
                json.dumps(rows[0], ensure_ascii=False)
                + "\r\n\r\n"
                + json.dumps(rows[1])
                + "\r\n   \r\n"
                + json.dumps(rows[2]),
                encoding="utf-8",
            )
            lazy = _LazyJsonlRows(path)
            self.assertEqual(len(lazy), 3)
            self.assertEqual(list(lazy), _read_jsonl(path))
            self.assertEqual(lazy[-1], rows[-1])
            self.assertEqual(lazy[1:], rows[1:])
            self.assertFalse(hasattr(lazy, "metas"))
            lazy[0]
            restored = pickle.loads(pickle.dumps(lazy))
            self.assertIsNone(restored._fd)
            self.assertEqual(list(restored), rows)
            restored.close()
            lazy.close()

    def test_lazy_jsonl_rows_fail_closed_after_file_mutation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rows.jsonl"
            path.write_text('{"value": 1}\n', encoding="utf-8")
            lazy = _LazyJsonlRows(path)
            path.write_text('{"value": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after indexing"):
                lazy[0]

    def test_lazy_jsonl_requires_strict_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rows.jsonl"
            path.write_text('{"filename": "a.jpg"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strict-identity"):
                PatchEpisodeJsonlDataset(
                    root=tmp_dir,
                    anno=str(path),
                    lazy_jsonl=True,
                    strict_sample_identity=False,
                )

    def test_lazy_jsonl_requires_exact_boolean(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rows.jsonl"
            path.write_text('{"filename": "a.jpg"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact boolean"):
                PatchEpisodeJsonlDataset(
                    root=tmp_dir,
                    anno=str(path),
                    lazy_jsonl=1,
                    strict_sample_identity=True,
                )

    def test_lazy_and_eager_dataset_samples_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 32), color=(64, 128, 192)).save(image_path)
            row = {
                "filename": str(image_path),
                "image_id": 11,
                "ann_id": 12,
                "ref_id": 13,
                "sent_id": 14,
                "sample_id": "sample-11-12-13-14",
                "primary_support_instance_index": 0,
                "instances": [
                    {
                        "bbox": [4, 5, 16, 17],
                        "class_id": 1,
                        "phrase": "blue object",
                        "head": "object",
                        "text_is_negative": False,
                    }
                ],
            }
            anno_path = root / "rows.jsonl"
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            kwargs = {
                "root": str(root),
                "anno": str(anno_path),
                "box_format": "xywh",
                "neg_episode_prob": 0.0,
                "support_min_count": 1,
                "support_num_patches_min": 1,
                "support_num_patches_max": 1,
                "tn_balance_sampling": False,
                "strict_sample_identity": True,
            }
            eager = PatchEpisodeJsonlDataset(**kwargs, lazy_jsonl=False)
            lazy = PatchEpisodeJsonlDataset(**kwargs, lazy_jsonl=True)

            random.seed(17)
            torch.manual_seed(17)
            eager_image, eager_target = eager[0]
            random.seed(17)
            torch.manual_seed(17)
            lazy_image, lazy_target = lazy[0]

            self.assertEqual(eager_image.tobytes(), lazy_image.tobytes())
            self.assertEqual(set(eager_target), set(lazy_target))
            for key in eager_target:
                with self.subTest(key=key):
                    eager_value = eager_target[key]
                    lazy_value = lazy_target[key]
                    if torch.is_tensor(eager_value):
                        self.assertTrue(torch.equal(eager_value, lazy_value))
                    else:
                        self.assertEqual(eager_value, lazy_value)

    def test_builder_rejects_lazy_jsonl_outside_validated_data_driven_rows(self):
        with self.assertRaisesRegex(ValueError, "validated data-driven"):
            build_patch_episode(
                "train",
                SimpleNamespace(stage_b_data_driven_score=False),
                {
                    "root": "/",
                    "anno": "/tmp/not-used.jsonl",
                    "strict_sample_identity": True,
                    "lazy_jsonl": True,
                },
            )

    def test_strict_sample_identity_fails_instead_of_resampling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            anno_path = root / "invalid.jsonl"
            anno_path.write_text(
                json.dumps(
                    {
                        "filename": "missing.jpg",
                        "instances": [
                            {"bbox": [0, 0, 8, 8], "class_id": 1}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = PatchEpisodeJsonlDataset(
                root=str(root),
                anno=str(anno_path),
                neg_episode_prob=0.0,
                support_min_count=1,
                strict_sample_identity=True,
            )
            with self.assertRaisesRegex(RuntimeError, "resampling is forbidden"):
                dataset[0]

    def test_adapter_no_support_rejects_missing_or_nonzero_negative_episode_prob(self):
        args = SimpleNamespace(stage_b_gdino_score_adapter=True)
        base = {
            "root": "/",
            "anno": "/tmp/unused-adapter-pairs.jsonl",
            "stage_b_gdino_adapter_no_support": True,
            "require_global_tn_verified": True,
        }
        with self.assertRaisesRegex(ValueError, "neg_episode_prob=0.0"):
            build_patch_episode("train", args, dict(base))
        nonzero = {**base, "neg_episode_prob": 0.1}
        with self.assertRaisesRegex(ValueError, "neg_episode_prob=0.0"):
            build_patch_episode("train", args, nonzero)

    def test_content_and_changed_masks_survive_missing_canonical_word(self):
        dataset = self._dataset()
        result = dataset._build_slot_text_masks(
            ["gray shirt wearing glasses", "gray shirt wearing sunglasses"],
            ["person", "person"],
            [[], []],
            slot_records=[
                {"text_is_negative": False},
                {
                    "text_is_negative": True,
                    "replace_from": "glasses",
                    "replace_to": "sunglasses",
                    "replace_category": "attribute",
                    "positive_phrase": "gray shirt wearing glasses",
                },
            ],
        )
        canonical_mask = result[2]
        attr_pos_mask = result[3]
        attr_neg_mask = result[4]
        content_mask = result[6]

        self.assertFalse(canonical_mask.any())
        self.assertTrue(attr_pos_mask[0].any())
        self.assertTrue(content_mask[0].any())
        self.assertTrue(attr_neg_mask[1].any())
        self.assertTrue(content_mask[1].any())

    def test_rank_positive_survives_missing_canonical_word(self):
        dataset = self._dataset()
        result = dataset._build_slot_text_masks(
            ["gray shirt wearing sunglasses"],
            ["person"],
            [[]],
            slot_records=[
                {
                    "text_is_negative": True,
                    "replace_from": "glasses",
                    "replace_to": "sunglasses",
                    "replace_category": "attribute",
                    "positive_phrase": "gray shirt wearing glasses",
                }
            ],
        )
        rank_phrase_mask = result[10]
        rank_canonical_mask = result[11]
        has_rank_positive = result[12]
        rank_positive_captions = result[13]

        self.assertTrue(rank_phrase_mask[0].any())
        self.assertFalse(rank_canonical_mask[0].any())
        self.assertTrue(bool(has_rank_positive[0]))
        self.assertEqual(rank_positive_captions[0], "gray shirt wearing glasses .")

    def test_rank_positive_is_independent_of_invalid_tn_changed_span(self):
        dataset = self._dataset()
        result = dataset._build_slot_text_masks(
            ["girl looking at the phone"],
            ["girl"],
            [[]],
            slot_records=[
                {
                    "text_is_negative": True,
                    "replace_from": "not",
                    "replace_to": "",
                    "replace_category": "attribute",
                    "positive_phrase": "girl not looking at the phone",
                }
            ],
        )

        self.assertFalse(result[4][0].any())
        self.assertTrue(result[10][0].any())
        self.assertTrue(bool(result[12][0]))
        self.assertEqual(
            result[13][0], "girl not looking at the phone ."
        )

    def test_single_patch_sam3_pair_loader_sets_pair_stride(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(image_path)
            anno_path = root / "pairs.jsonl"
            row = {
                "image_path": str(image_path),
                "class_id": 1,
                "sam_bbox": [4, 4, 16, 16],
                "sent": "red car",
                "try_tn": "blue car",
                "class_norm_name": "car",
                "replace_from": "red",
                "replace_to": "blue",
                "replace_category": "color",
            }
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            dataset = PatchEpisodeJsonlDataset(
                root=str(root),
                anno=str(anno_path),
                source="sam3_tn_pair",
                box_format="xywh",
                neg_episode_prob=0.0,
                support_min_count=1,
                support_num_patches_min=1,
                support_num_patches_max=1,
                build_text_token_masks=True,
                text_encoder_type="bert-base-uncased",
                text_mask_warn_limit=0,
                tn_balance_sampling=False,
            )
            _image, target = dataset[0]

        self.assertEqual(target["cap_list"], ["red car", "blue car"])
        self.assertEqual(target["is_tn"].tolist(), [False, True])
        self.assertEqual(target["verifier_pair_stride"].tolist(), [2])
        self.assertEqual(target["verifier_num_patch_slots"].tolist(), [1])
        self.assertEqual(target["global_tn_verified"].tolist(), [False])
        self.assertEqual(target["proposalset_proxy_verified"].tolist(), [False])

    def test_global_tn_verified_is_required_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(image_path)
            anno_path = root / "pairs.jsonl"
            row = {
                "image_path": str(image_path),
                "class_id": 1,
                "sam_bbox": [4, 4, 16, 16],
                "sent": "red car",
                "try_tn": "blue car",
                "class_norm_name": "car",
                "replace_from": "red",
                "replace_to": "blue",
                "replace_category": "color",
            }
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            kwargs = {
                "root": str(root),
                "anno": str(anno_path),
                "source": "sam3_tn_pair",
                "box_format": "xywh",
                "neg_episode_prob": 0.0,
                "support_min_count": 1,
                "support_num_patches_min": 1,
                "support_num_patches_max": 1,
                "build_text_token_masks": True,
                "text_encoder_type": "bert-base-uncased",
                "text_mask_warn_limit": 0,
                "tn_balance_sampling": False,
                "require_global_tn_verified": True,
            }
            with self.assertRaisesRegex(ValueError, "global_tn_verified=true"):
                PatchEpisodeJsonlDataset(**kwargs)

            row["global_tn_verified"] = "false"
            row["tn_scope"] = "image_global_topk_verified"
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact boolean"):
                PatchEpisodeJsonlDataset(**kwargs)

            row["global_tn_verified"] = True
            row["tn_scope"] = "proposal_set_verified"
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "image_global_topk_verified"):
                PatchEpisodeJsonlDataset(**kwargs)

            row["tn_scope"] = "image_global_topk_verified"
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = PatchEpisodeJsonlDataset(**kwargs)
            _image, target = dataset[0]

        self.assertEqual(target["global_tn_verified"].tolist(), [True])
        self.assertEqual(target["proposalset_proxy_verified"].tolist(), [False])

    def test_proposalset_proxy_requires_exact_boolean_and_scope(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(image_path)
            anno_path = root / "pairs.jsonl"
            row = {
                "image_path": str(image_path),
                "class_id": 1,
                "sam_bbox": [4, 4, 16, 16],
                "sent": "red car",
                "try_tn": "blue car",
                "class_norm_name": "car",
                "replace_from": "red",
                "replace_to": "blue",
                "replace_category": "color",
                "proposalset_proxy_verified": "false",
                "tn_scope": "proposal_set_verified",
            }
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            kwargs = {
                "root": str(root),
                "anno": str(anno_path),
                "source": "sam3_tn_pair",
                "box_format": "xywh",
                "neg_episode_prob": 0.0,
                "support_min_count": 1,
                "support_num_patches_min": 1,
                "support_num_patches_max": 1,
                "build_text_token_masks": True,
                "text_encoder_type": "bert-base-uncased",
                "text_mask_warn_limit": 0,
                "tn_balance_sampling": False,
                "require_proposalset_proxy_verified": True,
            }
            with self.assertRaisesRegex(ValueError, "exact boolean"):
                PatchEpisodeJsonlDataset(**kwargs)

            row["proposalset_proxy_verified"] = True
            row["tn_scope"] = "image_global_topk_verified"
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "proposal_set_verified"):
                PatchEpisodeJsonlDataset(**kwargs)

            row["tn_scope"] = "proposal_set_verified"
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = PatchEpisodeJsonlDataset(**kwargs)
            _image, target = dataset[0]

        self.assertEqual(target["proposalset_proxy_verified"].tolist(), [True])
        self.assertEqual(target["global_tn_verified"].tolist(), [False])

    def test_benchmark_dataft_scope_requires_exact_schema_and_propagates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(image_path)
            anno_path = root / "pairs.jsonl"
            row = {
                "filename": str(image_path),
                "benchmark_dataft_alltn": "true",
                "proposalset_proxy_verified": False,
                "tn_scope": "benchmark_dataft_alltn",
                "instances": [
                    {
                        "bbox": [4, 4, 16, 16],
                        "class_id": 1,
                        "raw_phrase": "red car",
                        "phrase": "red car",
                        "head": "car",
                        "positive_phrase": "red car",
                        "negative_phrase": "blue car",
                        "try_tn": "blue car",
                        "text_is_negative": False,
                        "sam3_tn_pair": True,
                    }
                ],
            }
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            kwargs = {
                "root": str(root),
                "anno": str(anno_path),
                "box_format": "xywh",
                "neg_episode_prob": 0.0,
                "support_min_count": 1,
                "support_num_patches_min": 1,
                "support_num_patches_max": 1,
                "build_text_token_masks": True,
                "text_encoder_type": "bert-base-uncased",
                "text_mask_warn_limit": 0,
                "tn_balance_sampling": False,
                "require_benchmark_dataft_alltn": True,
                "stage_b_gdino_adapter_no_support": True,
            }
            with self.assertRaisesRegex(ValueError, "exact boolean"):
                PatchEpisodeJsonlDataset(**kwargs)

            row["benchmark_dataft_alltn"] = True
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = PatchEpisodeJsonlDataset(**kwargs)
            _image, target = dataset[0]

        self.assertEqual(target["cap_list"], ["red car", "blue car"])
        self.assertEqual(target["is_tn"].tolist(), [False, True])
        self.assertEqual(target["benchmark_dataft_alltn"].tolist(), [True])
        self.assertEqual(target["proposalset_proxy_verified"].tolist(), [False])
        self.assertEqual(target["tn_scope"], "benchmark_dataft_alltn")
        self.assertNotIn("patch", target)
        self.assertNotIn("patch_global", target)

    def test_adapter_ref_eval_no_support_preserves_identity_and_caption(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(image_path)
            anno_path = root / "ref.jsonl"
            row = {
                "filename": str(image_path),
                "image_id": 11,
                "ann_id": 12,
                "ref_id": 13,
                "sent_id": 14,
                "instances": [
                    {
                        "bbox": [4, 4, 16, 16],
                        "class_id": 1,
                        "raw_phrase": "red car",
                        "phrase": "red car",
                        "head": "car",
                        "positive_phrase": "red car",
                        "text_is_negative": False,
                    }
                ],
            }
            anno_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = PatchEpisodeJsonlDataset(
                root=str(root),
                anno=str(anno_path),
                box_format="xywh",
                neg_episode_prob=0.0,
                support_min_count=1,
                support_num_patches_min=1,
                support_num_patches_max=1,
                build_text_token_masks=True,
                text_encoder_type="bert-base-uncased",
                text_mask_warn_limit=0,
                tn_balance_sampling=False,
                stage_b_gdino_adapter_ref_eval=True,
                stage_b_gdino_adapter_no_support=True,
            )
            _image, target = dataset[0]

        self.assertEqual(target["caption"], "red car .")
        self.assertEqual(target["boxes"].shape, (1, 4))
        self.assertEqual(
            [int(target[key].item()) for key in ("image_id", "ann_id", "ref_id", "sent_id")],
            [11, 12, 13, 14],
        )
        self.assertNotIn("patch", target)
        self.assertNotIn("patch_global", target)


if __name__ == "__main__":
    unittest.main()
