import json
import tempfile
import unittest
from pathlib import Path

from tools.stageb_eval_holdout import is_excluded, load_holdout_keys


class StageBEvalHoldoutTest(unittest.TestCase):
    def test_load_and_filter_ann_or_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                json.dumps({"image_id": 10, "ann_id": 20}) + "\n"
                + json.dumps({"image_id": 11, "ann_id": 21}) + "\n",
                encoding="utf-8",
            )
            ann_keys, image_ids = load_holdout_keys([str(path)])

        self.assertTrue(
            is_excluded(
                image_id=10, ann_id=20, level="ann", ann_keys=ann_keys, image_ids=image_ids
            )
        )
        self.assertFalse(
            is_excluded(
                image_id=10, ann_id=99, level="ann", ann_keys=ann_keys, image_ids=image_ids
            )
        )
        self.assertTrue(
            is_excluded(
                image_id=10, ann_id=99, level="image", ann_keys=ann_keys, image_ids=image_ids
            )
        )


if __name__ == "__main__":
    unittest.main()
