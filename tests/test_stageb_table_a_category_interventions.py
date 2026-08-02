import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools import build_stageb_table_a_category_interventions as builder


class CategoryInterventionBuilderTest(unittest.TestCase):
    def test_builds_two_arms_and_one_support_asset_per_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "scene.jpg"
            support_a = root / "apple.jpg"
            support_b = root / "banana.jpg"
            Image.new("RGB", (100, 100), "white").save(image)
            Image.new("RGB", (16, 16), "red").save(support_a)
            Image.new("RGB", (16, 16), "yellow").save(support_b)

            canonical = root / "canonical.json"
            canonical.write_text(
                json.dumps({"1": "apple", "2": "banana"}), encoding="utf-8"
            )
            source = root / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "filename": str(image),
                        "image_id": 7,
                        "width": 100,
                        "height": 100,
                        "detection": {
                            "instances": [
                                {"bbox": [0, 0, 20, 20], "label": 1},
                                {"bbox": [60, 60, 90, 90], "label": 2},
                            ]
                        },
                        "not_exhaustive_labels": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            support = root / "support.tsv"
            with support.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["path", "class", "bucket", "emb_rel_path"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "path": str(support_a),
                        "class": "apple",
                        "bucket": "clean",
                        "emb_rel_path": "",
                    }
                )
                writer.writerow(
                    {
                        "path": str(support_b),
                        "class": "banana",
                        "bucket": "clean",
                        "emb_rel_path": "",
                    }
                )

            output = root / "pairs.jsonl"
            output_support = root / "unique.tsv"
            audit = root / "audit.json"
            result = builder.build(
                argparse.Namespace(
                    source=str(source),
                    canonical_map=str(canonical),
                    support_tsv=str(support),
                    support_image_root=str(root / "missing_mirror"),
                    output=str(output),
                    output_support_tsv=str(output_support),
                    audit=str(audit),
                    seed=17,
                    max_pairs=10,
                    max_cross_iou=0.1,
                )
            )
            self.assertEqual(result["summary"]["pairs"], 1)
            self.assertEqual(result["summary"]["rows"], 2)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                {row["category_intervention"]["arm"] for row in rows},
                {"A", "B"},
            )
            self.assertNotEqual(
                rows[0]["category_intervention"]["active_support_sha256"],
                rows[1]["category_intervention"]["active_support_sha256"],
            )
            verified = builder.verify(
                output=output,
                output_support_tsv=output_support,
                audit_path=audit,
            )
            self.assertEqual(verified, result["summary"])
            with output_support.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 2)

            audit_value = json.loads(audit.read_text(encoding="utf-8"))
            audit_value["contract"]["same_image"] = False
            audit.write_text(json.dumps(audit_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audit contract"):
                builder.verify(
                    output=output,
                    output_support_tsv=output_support,
                    audit_path=audit,
                )

    def test_pair_validator_rejects_support_reuse(self):
        pair = {
            "schema": builder.SCHEMA,
            "image_id": 1,
            "category_intervention": {
                "pair_id": "p",
                "arm": "A",
                "image_path": "x",
                "image_sha256": "i",
                "image_width": 1,
                "image_height": 1,
                "active_class_id": 1,
                "counterfactual_class_id": 2,
                "active_support_sha256": "same",
            },
        }
        other = json.loads(json.dumps(pair))
        other["category_intervention"].update(
            {"arm": "B", "active_class_id": 2, "counterfactual_class_id": 1}
        )
        with self.assertRaisesRegex(ValueError, "same support asset"):
            builder._validate_rows([pair, other])

    def test_formal_verify_rejects_noncanonical_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "pairs.jsonl"
            support = root / "support.tsv"
            audit = root / "audit.json"
            output.write_text("", encoding="utf-8")
            support.write_text("path\tclass\n", encoding="utf-8")
            audit.write_text(
                json.dumps(
                    {
                        "schema": builder.AUDIT_SCHEMA,
                        "contract": builder.CONTRACT,
                        "evidence_status": "runtime_inputs_built_no_model_results",
                        "seed": 17,
                        "max_pairs": 1,
                        "max_cross_iou": 0.1,
                        "inputs": {},
                        "outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                builder.verify(
                    output=output,
                    output_support_tsv=support,
                    audit_path=audit,
                    require_canonical=True,
                )


if __name__ == "__main__":
    unittest.main()
