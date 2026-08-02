import argparse
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_stageb_gdino_adapter_dataft_pairs import build


class StageBGDINODataFTBuilderTest(unittest.TestCase):
    @staticmethod
    def _source(index):
        return {
            "filename": f"/images/{index}.jpg",
            "source": "refcoco_unc_train",
            "image_id": 10 + index,
            "ann_id": 20 + index,
            "ref_id": 30 + index,
            "sent_id": 40 + index,
            "split": "train",
            "instances": [
                {
                    "bbox": [1, 2, 10, 20],
                    "class_id": 7,
                    "raw_phrase": "blue car",
                    "positive_phrase": "red car",
                    "head": "car",
                    "text_is_negative": True,
                    "replace_from": "red",
                    "replace_to": "blue",
                    "replace_category": "color",
                }
            ],
        }

    @staticmethod
    def _dataft(index):
        return {
            "filename": f"/images/{index}.jpg",
            "image_id": 10 + index,
            "grounding": {
                "regions": [],
                "caption": "blue car .",
                "caption_list": ["blue car"],
                "is_negative": True,
                "tn_records": [
                    {"phrase": "blue car", "positive_phrase": "red car"}
                ],
            },
        }

    def _write_jsonl(self, path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_builder_requires_lockstep_identity_and_writes_explicit_scope(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.jsonl"
            dataft = root / "dataft.jsonl"
            output = root / "pairs.jsonl"
            audit = root / "audit.json"
            self._write_jsonl(source, [self._source(0), self._source(1)])
            self._write_jsonl(dataft, [self._dataft(0), self._dataft(1)])
            result = build(
                argparse.Namespace(
                    source_pairs=str(source),
                    dataft_tn=str(dataft),
                    output=str(output),
                    audit=str(audit),
                    expected_rows=2,
                )
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(result["rows"], 2)
        self.assertEqual(rows[0]["tn_scope"], "benchmark_dataft_alltn")
        self.assertIs(rows[0]["benchmark_dataft_alltn"], True)
        self.assertIs(rows[0]["proposalset_proxy_verified"], False)
        instance = rows[0]["instances"][0]
        self.assertEqual(instance["raw_phrase"], "red car")
        self.assertEqual(instance["negative_phrase"], "blue car")
        self.assertIs(instance["sam3_tn_pair"], True)

    def test_builder_fails_instead_of_guessing_a_drifted_pair(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.jsonl"
            dataft = root / "dataft.jsonl"
            output = root / "pairs.jsonl"
            audit = root / "audit.json"
            self._write_jsonl(source, [self._source(0)])
            bad = self._dataft(0)
            bad["grounding"]["tn_records"][0]["positive_phrase"] = "green car"
            self._write_jsonl(dataft, [bad])

            with self.assertRaisesRegex(ValueError, "positive text drift"):
                build(
                    argparse.Namespace(
                        source_pairs=str(source),
                        dataft_tn=str(dataft),
                        output=str(output),
                        audit=str(audit),
                        expected_rows=1,
                    )
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
