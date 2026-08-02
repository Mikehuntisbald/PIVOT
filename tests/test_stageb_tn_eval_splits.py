import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import eval_stageb_tn_val as tn_eval


class StageBTnEvalSplitTest(unittest.TestCase):
    def test_google_compatibility_and_umd_specs_coexist(self):
        specs = {row["name"]: row for row in tn_eval._split_specs()}
        self.assertEqual(
            specs["refcocog_val"],
            {
                "name": "refcocog_val",
                "pair_source": "refcocog_google",
                "dataset": "refcocog",
                "splitby": "google",
                "split": "val",
            },
        )
        self.assertEqual(
            specs["refcocog_umd_val"],
            {
                "name": "refcocog_umd_val",
                "pair_source": "refcocog_umd",
                "dataset": "refcocog",
                "splitby": "umd",
                "split": "val",
            },
        )

    def test_build_tn_eval_jsonl_selects_umd_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tn.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "image_id": 3,
                        "ann_id": 2,
                        "ref_id": 1,
                        "sent_id": 4,
                        "instances": [
                            {
                                "pair_source": "refcocog_umd",
                                "raw_phrase": "blue car",
                                "positive_phrase": "red car",
                                "replace_category": "color",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            split_map = {(1, 2, 3): "val"}
            with mock.patch.object(
                tn_eval, "_load_ref_split_map", return_value=split_map
            ) as load_split:
                output, metas, counts = tn_eval._build_tn_eval_jsonl(
                    data_root=root,
                    output_dir=root / "out",
                    tn_jsonl=source,
                    splits=["refcocog_umd_val"],
                    max_pairs=0,
                )

            load_split.assert_called_once_with(root, "refcocog", "umd")
            self.assertEqual(counts, {"refcocog_umd_val": 1})
            self.assertEqual(len(metas), 1)
            self.assertEqual(metas[0]["eval_split"], "refcocog_umd_val")
            self.assertEqual(metas[0]["pair_source"], "refcocog_umd")
            written = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["tn_eval_split"], "refcocog_umd_val")


if __name__ == "__main__":
    unittest.main()
