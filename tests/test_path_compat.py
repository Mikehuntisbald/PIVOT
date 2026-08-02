import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from util.path_compat import remap_legacy_path


class LegacyPathCompatibilityTest(unittest.TestCase):
    def test_maps_old_repo_only_when_original_is_missing(self):
        repo = Path(__file__).resolve().parents[1]
        mapped = remap_legacy_path(
            "/home/user/PIVOT/data/ablations", repo_root=repo
        )
        self.assertEqual(mapped, repo / "data/ablations")

    def test_maps_old_data_and_nested_coco_train_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            image = (
                data_root
                / "COCO/coco2017/train2017/train2017/000000000001.jpg"
            )
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            mapped = remap_legacy_path(
                "/home/user/datasets/vision_benchmarks/COCO_2017/"
                "train2017/000000000001.jpg",
                data_root=data_root,
            )
            self.assertEqual(mapped, image)

            pivot_data = remap_legacy_path(
                "/home/user/datasets/pivot_data/COCO/coco2014/train2014",
                data_root=data_root,
            )
            self.assertEqual(pivot_data, data_root / "COCO/coco2014/train2014")

    def test_relative_path_is_not_rebased(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "/tmp/ignored"}):
            self.assertEqual(
                remap_legacy_path("data/example.jsonl"), Path("data/example.jsonl")
            )

    def test_maps_legacy_refcoco_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            expected = data_root / "COCO/refcocog/refs(umd).p"
            self.assertEqual(
                remap_legacy_path(
                    "/home/user/datasets/vision_benchmarks/RefCOCO/"
                    "refcocog/refs(umd).p",
                    data_root=data_root,
                ),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
