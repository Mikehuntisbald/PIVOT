import json
import tempfile
import unittest
from pathlib import Path

import torch

from datasets.patch_episode import PatchEpisodeJsonlDataset
from tools import build_stageb_u2_category_complete_ref as builder


class CategoryCompleteRefBuilderTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.inputs = self.root / "inputs"
        self.outputs = self.root / "outputs"
        self.inputs.mkdir()
        self.train = self.root / "instances_train2014.json"
        self.val = self.root / "instances_val2014.json"
        self.train.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "id": 10,
                            "image_id": 7,
                            "category_id": 1,
                            "bbox": [1, 2, 30, 40],
                            "iscrowd": 0,
                        },
                        {
                            "id": 11,
                            "image_id": 7,
                            "category_id": 1,
                            "bbox": [50, 60, 20, 10],
                            "iscrowd": 0,
                        },
                        {
                            "id": 12,
                            "image_id": 7,
                            "category_id": 1,
                            "bbox": [4, 4, 8, 8],
                            "iscrowd": 1,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.val.write_text(json.dumps({"annotations": []}), encoding="utf-8")
        self.row = {
            "filename": "/data/COCO_train2014_000000000007.jpg",
            "image_id": 7,
            "ann_id": 10,
            "ref_id": 3,
            "sent_id": 4,
            "instances": [
                {
                    "bbox": [1, 2, 30, 40],
                    "class_id": 782,
                    "refcoco_category_id": 1,
                    "raw_phrase": "the person",
                    "text_is_negative": False,
                }
            ],
        }
        for name in builder.SOURCE_NAMES:
            (self.inputs / name).write_text(
                json.dumps(self.row) + "\n", encoding="utf-8"
            )

    def tearDown(self):
        self.context.cleanup()

    def test_builds_all_same_category_boxes_and_receipt(self):
        receipt = builder.build_all(
            input_dir=self.inputs,
            output_dir=self.outputs,
            train2014=self.train,
            val2014=self.val,
        )
        output = json.loads(
            (self.outputs / builder.SOURCE_NAMES[0])
            .read_text(encoding="utf-8")
            .strip()
        )
        self.assertEqual(output["primary_support_instance_index"], 0)
        self.assertIs(output["stage_b_u2_category_complete"], True)
        self.assertEqual(len(output["instances"]), 2)
        self.assertEqual(output["instances"][0]["raw_phrase"], "the person")
        self.assertIs(output["instances"][0]["category_complete_primary"], True)
        self.assertIs(output["instances"][1]["category_complete_auxiliary"], True)
        self.assertEqual(output["instances"][1]["class_id"], 782)
        self.assertEqual(receipt["manifests"][builder.SOURCE_NAMES[0]]["rows"], 1)
        self.assertEqual(
            receipt["manifests"][builder.SOURCE_NAMES[0]]["auxiliary_instances"],
            1,
        )
        self.assertTrue((self.outputs / "receipt.json").is_file())

    def test_rejects_missing_target_annotation(self):
        index, _ = builder.build_coco_index(
            {"train2014": self.train, "val2014": self.val}
        )
        bad = dict(self.row)
        bad["ann_id"] = 999
        with self.assertRaisesRegex(builder.CategoryCompleteBuildError, "missing"):
            builder.enrich_row(bad, index, context="fixture")

    def test_primary_support_resolution_is_strict(self):
        labels = torch.tensor([782, 782], dtype=torch.int64)
        self.assertEqual(
            PatchEpisodeJsonlDataset._resolve_primary_support_instance(
                {"primary_support_instance_index": 0}, labels
            ),
            (782, 0),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            PatchEpisodeJsonlDataset._resolve_primary_support_instance(
                {"primary_support_instance_index": 2}, labels
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            PatchEpisodeJsonlDataset._resolve_primary_support_instance(
                {"primary_support_instance_index": True}, labels
            )


if __name__ == "__main__":
    unittest.main()
