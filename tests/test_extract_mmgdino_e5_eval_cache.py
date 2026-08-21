import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.extract_mmgdino_e5_eval_cache import parse_tn_eval_requests


class MMGDinoE5EvalCacheTests(unittest.TestCase):
    def test_strict_pair_is_positive_negative_without_fake_negative_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "COCO_train2014_000000000123.jpg"
            Image.new("RGB", (100, 80), color="white").save(image)
            requests = parse_tn_eval_requests(
                [
                    {
                        "sample_id": "strict:123:1",
                        "image_id": 123,
                        "filename": str(image),
                        "positive_phrase": "yellow shirt",
                        "negative_phrase": "red shirt",
                        "bbox": [10.0, 20.0, 30.0, 40.0],
                    }
                ],
                image_root=root,
            )
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0].pair_role, "positive")
            self.assertEqual(requests[1].pair_role, "negative")
            self.assertEqual(tuple(requests[0].gt_boxes.shape), (1, 4))
            self.assertEqual(tuple(requests[1].gt_boxes.shape), (0, 4))
            self.assertEqual(requests[0].caption, "yellow shirt")
            self.assertEqual(requests[1].caption, "red shirt")


if __name__ == "__main__":
    unittest.main()
