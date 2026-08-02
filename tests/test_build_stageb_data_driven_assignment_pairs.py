import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools import build_stageb_data_driven_assignment_pairs as builder


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


class AssignmentPairBuilderTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.rows = self._fixture_rows()
        for manifest_index, name in enumerate(builder.MANIFESTS):
            rows = []
            for row in self.rows:
                copied = json.loads(json.dumps(row))
                copied["source"] = f"fixture_{manifest_index}_train"
                rows.append(copied)
            (self.inputs / name).write_text(
                "".join(_canonical(row) + "\n" for row in rows),
                encoding="ascii",
            )
        self.expected_hashes = {
            name: builder._sha256_file(self.inputs / name)
            for name in builder.MANIFESTS
        }
        self.expected_rows = {
            name: len(self.rows) for name in builder.MANIFESTS
        }
        category_complete_receipt = {
            "schema": "pivot.stageb.u2_category_complete_receipt/v1",
            "row_schema": "pivot.stageb.u2_category_complete_ref/v1",
            "manifests": {
                name: {
                    "rows": self.expected_rows[name],
                    "output": {"sha256": self.expected_hashes[name]},
                }
                for name in builder.MANIFESTS
            },
        }
        (self.inputs / "receipt.json").write_text(
            json.dumps(category_complete_receipt, sort_keys=True) + "\n",
            encoding="ascii",
        )
        self.expected_receipt_sha256 = builder._sha256_file(
            self.inputs / "receipt.json"
        )

    def tearDown(self):
        self.context.cleanup()

    @staticmethod
    def _official_row(
        *,
        image_id,
        ann_id,
        ref_id,
        sent_id,
        phrase,
        bbox,
        all_instances,
        category_id=1,
    ):
        instances = [
            {
                "bbox": list(bbox),
                "class_id": 782,
                "refcoco_category_id": category_id,
                "coco_ann_id": ann_id,
                "category_complete_primary": True,
                "raw_phrase": phrase,
                "positive_phrase": phrase,
                "text_is_negative": False,
            }
        ]
        for other_ann_id, other_bbox in all_instances:
            if other_ann_id == ann_id:
                continue
            instances.append(
                {
                    "bbox": list(other_bbox),
                    "class_id": 782,
                    "refcoco_category_id": category_id,
                    "coco_ann_id": other_ann_id,
                    "category_complete_auxiliary": True,
                }
            )
        return {
            "source": "fixture_train",
            "image_id": image_id,
            "ann_id": ann_id,
            "ref_id": ref_id,
            "sent_id": sent_id,
            "split": "train",
            "filename": f"/coco/COCO_train2014_{image_id:012d}.jpg",
            "category_complete_coco_split": "train2014",
            "category_complete_coco_category_id": category_id,
            "category_complete_instance_count": len(instances),
            "instances": instances,
            "primary_support_instance_index": 0,
            "stage_b_u2_category_complete": True,
            "stage_b_u2_category_complete_schema": (
                "pivot.stageb.u2_category_complete_ref/v1"
            ),
        }

    def _fixture_rows(self):
        rows = []
        balanced_instances = (
            (100, (0, 0, 10, 10)),
            (200, (20, 0, 10, 10)),
            (300, (40, 0, 10, 10)),
        )
        specifications = (
            (100, 1, 1, "red person left"),
            (100, 1, 2, "red person left side"),
            (200, 2, 3, "red person right"),
            (200, 2, 4, "blue person right"),
            (300, 3, 5, "red person center"),
        )
        bbox_by_ann = dict(balanced_instances)
        for ann_id, ref_id, sent_id, phrase in specifications:
            rows.append(
                self._official_row(
                    image_id=10,
                    ann_id=ann_id,
                    ref_id=ref_id,
                    sent_id=sent_id,
                    phrase=phrase,
                    bbox=bbox_by_ann[ann_id],
                    all_instances=balanced_instances,
                )
            )

        only = ((400, (0, 0, 10, 10)),)
        rows.append(
            self._official_row(
                image_id=11,
                ann_id=400,
                ref_id=4,
                sent_id=6,
                phrase="only green person",
                bbox=only[0][1],
                all_instances=only,
            )
        )

        overlap = ((500, (0, 0, 10, 10)), (600, (0, 0, 10, 10)))
        rows.extend(
            [
                self._official_row(
                    image_id=12,
                    ann_id=ann_id,
                    ref_id=ref_id,
                    sent_id=sent_id,
                    phrase=phrase,
                    bbox=dict(overlap)[ann_id],
                    all_instances=overlap,
                )
                for ann_id, ref_id, sent_id, phrase in (
                    (500, 5, 7, "front person"),
                    (600, 6, 8, "back person"),
                )
            ]
        )

        same_phrase = ((700, (0, 0, 10, 10)), (800, (20, 0, 10, 10)))
        rows.extend(
            [
                self._official_row(
                    image_id=13,
                    ann_id=ann_id,
                    ref_id=ref_id,
                    sent_id=sent_id,
                    phrase=phrase,
                    bbox=dict(same_phrase)[ann_id],
                    all_instances=same_phrase,
                )
                for ann_id, ref_id, sent_id, phrase in (
                    (700, 7, 9, "The Red Person!"),
                    (800, 8, 10, "the red person"),
                )
            ]
        )
        return rows

    def _build(self, output_name="output"):
        return builder.build_all(
            input_root=self.inputs,
            output_root=self.root / output_name,
            expected_input_sha256=self.expected_hashes,
            expected_rows_by_manifest=self.expected_rows,
            expected_category_complete_receipt_sha256=(
                self.expected_receipt_sha256
            ),
        )

    def test_builds_hard_balanced_official_pairs_and_retains_invalid_rows(self):
        receipt = self._build()
        output_root = self.root / "output"
        self.assertEqual(receipt["schema"], builder.RECEIPT_SCHEMA)
        self.assertEqual(receipt["row_schema"], builder.ROW_SCHEMA)
        self.assertEqual(receipt["rows"], len(self.rows) * 3)
        self.assertEqual(receipt["unique_identities"], len(self.rows) * 3)
        self.assertEqual(receipt["valid_rows"], 5 * 3)
        self.assertEqual(receipt["invalid_rows"], 5 * 3)
        self.assertTrue((output_root / "receipt.json").is_file())
        self.assertEqual(
            receipt["category_complete_receipt"]["sha256"],
            self.expected_receipt_sha256,
        )

        name = builder.MANIFESTS[0]
        source_rows = [
            json.loads(line)
            for line in (self.inputs / name).read_text(encoding="ascii").splitlines()
        ]
        output_rows = [
            json.loads(line)
            for line in (output_root / name).read_text(encoding="ascii").splitlines()
        ]
        self.assertEqual(len(output_rows), len(source_rows))
        for source, output in zip(source_rows, output_rows, strict=True):
            base = dict(output)
            for key in builder.ADDED_KEYS:
                base.pop(key)
            self.assertEqual(base, source)
            self.assertIs(output["stage_b_data_driven_assignment_pair"], True)
            self.assertEqual(
                output["stage_b_data_driven_assignment_pair_schema"],
                builder.ROW_SCHEMA,
            )

        first = output_rows[0]
        self.assertIs(first["assignment_pair_valid"], True)
        first_pair = first["assignment_pair"]
        self.assertEqual(first_pair["anchor"]["ann_id"], 100)
        self.assertEqual(first_pair["partner"]["ann_id"], 200)
        self.assertEqual(first_pair["partner"]["expression"], "red person right")
        self.assertLess(first_pair["partner"]["target_iou"], 0.3)
        self.assertTrue(first_pair["selection"]["model_score_free"])

        second = output_rows[1]
        self.assertEqual(second["assignment_pair"]["partner"]["ann_id"], 300)
        self.assertNotEqual(
            second["assignment_pair"]["anchor"]["normalized_expression"],
            second["assignment_pair"]["partner"]["normalized_expression"],
        )
        self.assertEqual(
            [row["assignment_pair_invalid_reason"] for row in output_rows[5:]],
            [
                "no_distinct_official_annotation",
                "no_partner_below_target_iou",
                "no_partner_below_target_iou",
                "no_distinct_normalized_official_expression",
                "no_distinct_normalized_official_expression",
            ],
        )
        for row in output_rows[5:]:
            self.assertIs(row["assignment_pair_valid"], False)
            self.assertIsNone(row["assignment_pair"]["partner"])
            self.assertEqual(
                row["assignment_pair"]["anchor"]["ann_id"], row["ann_id"]
            )

        manifest_receipt = receipt["manifests"][name]
        self.assertEqual(manifest_receipt["rows"], len(self.rows))
        self.assertEqual(manifest_receipt["valid_rows"], 5)
        self.assertEqual(
            manifest_receipt["ordered_identity_stream_sha256"],
            manifest_receipt["output_ordered_identity_stream_sha256"],
        )
        self.assertEqual(
            manifest_receipt["source_row_stream_sha256"],
            manifest_receipt["output_base_row_stream_sha256"],
        )
        self.assertEqual(
            manifest_receipt["output"]["sha256"],
            builder._sha256_file(output_root / name),
        )
        self.assertIn(
            "partner_annotation_use_count_histogram",
            manifest_receipt["partner_usage"],
        )
        self.assertTrue(all(receipt["invariants"].values()))

    def test_empty_normalized_expression_is_retained_but_never_paired(self):
        instances = ((900, (0, 0, 10, 10)), (1000, (20, 0, 10, 10)))
        rows = [
            self._official_row(
                image_id=14,
                ann_id=ann_id,
                ref_id=ref_id,
                sent_id=sent_id,
                phrase=phrase,
                bbox=dict(instances)[ann_id],
                all_instances=instances,
            )
            for ann_id, ref_id, sent_id, phrase in (
                (900, 9, 11, "{}"),
                (1000, 10, 12, "visible person"),
            )
        ]
        metas = [
            builder._row_meta(row, line_number=index + 1, context=f"row-{index}")
            for index, row in enumerate(rows)
        ]
        groups = {metas[0].group_key: [0, 1]}
        assignments = builder.select_assignments(metas, groups)
        self.assertEqual(
            assignments[0].invalid_reason,
            "empty_normalized_official_expression",
        )
        self.assertEqual(
            assignments[1].invalid_reason,
            "no_distinct_normalized_official_expression",
        )
    def test_is_deterministic_and_refuses_overwrite(self):
        self._build("first")
        self._build("second")
        for name in builder.MANIFESTS:
            self.assertEqual(
                builder._sha256_file(self.root / "first" / name),
                builder._sha256_file(self.root / "second" / name),
            )
        before = builder._sha256_file(self.root / "first" / builder.MANIFESTS[0])
        with self.assertRaisesRegex(
            builder.AssignmentPairBuildError, "refusing to replace"
        ):
            self._build("first")
        self.assertEqual(
            before,
            builder._sha256_file(self.root / "first" / builder.MANIFESTS[0]),
        )

    def test_input_hash_drift_fails_before_output(self):
        bad_hashes = dict(self.expected_hashes)
        bad_hashes[builder.MANIFESTS[0]] = "0" * 64
        category_complete_receipt = json.loads(
            (self.inputs / "receipt.json").read_text(encoding="ascii")
        )
        category_complete_receipt["manifests"][builder.MANIFESTS[0]][
            "output"
        ]["sha256"] = bad_hashes[builder.MANIFESTS[0]]
        (self.inputs / "receipt.json").write_text(
            json.dumps(category_complete_receipt, sort_keys=True) + "\n",
            encoding="ascii",
        )
        receipt_sha256 = builder._sha256_file(self.inputs / "receipt.json")
        output = self.root / "hash-failure"
        with self.assertRaisesRegex(
            builder.AssignmentPairBuildError, "input SHA256 mismatch"
        ):
            builder.build_all(
                input_root=self.inputs,
                output_root=output,
                expected_input_sha256=bad_hashes,
                expected_rows_by_manifest=self.expected_rows,
                expected_category_complete_receipt_sha256=(
                    receipt_sha256
                ),
            )
        self.assertFalse(output.exists())
        self.assertFalse(any(self.root.glob(".hash-failure.tmp-*")))

    def test_malformed_late_manifest_leaves_no_partial_output(self):
        late = self.inputs / builder.MANIFESTS[-1]
        late.write_text(
            late.read_text(encoding="ascii") + "not-json\n", encoding="ascii"
        )
        hashes = dict(self.expected_hashes)
        hashes[builder.MANIFESTS[-1]] = builder._sha256_file(late)
        rows = dict(self.expected_rows)
        rows[builder.MANIFESTS[-1]] += 1
        category_complete_receipt = json.loads(
            (self.inputs / "receipt.json").read_text(encoding="ascii")
        )
        category_complete_receipt["manifests"][builder.MANIFESTS[-1]][
            "rows"
        ] += 1
        category_complete_receipt["manifests"][builder.MANIFESTS[-1]][
            "output"
        ]["sha256"] = hashes[builder.MANIFESTS[-1]]
        (self.inputs / "receipt.json").write_text(
            json.dumps(category_complete_receipt, sort_keys=True) + "\n",
            encoding="ascii",
        )
        receipt_sha256 = builder._sha256_file(self.inputs / "receipt.json")
        output = self.root / "parse-failure"
        with self.assertRaisesRegex(builder.AssignmentPairBuildError, "invalid JSON"):
            builder.build_all(
                input_root=self.inputs,
                output_root=output,
                expected_input_sha256=hashes,
                expected_rows_by_manifest=rows,
                expected_category_complete_receipt_sha256=receipt_sha256,
            )
        self.assertFalse(output.exists())
        self.assertFalse(any(self.root.glob(".parse-failure.tmp-*")))


if __name__ == "__main__":
    unittest.main()
