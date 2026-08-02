import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_stageb_gdino_adapter_o64_direct_rank as builder


def _jsonl_bytes(rows):
    return b"".join(builder._canonical_bytes(row) + b"\n" for row in rows)


class O64DirectRankBuilderTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.source_manifest = self.root / "overfit64.jsonl"
        self.source_receipt = self.root / "receipt.json"
        self.expected_invariants = frozenset({"fixture_source_is_valid"})
        self.rows = [self._source_row(index) for index in range(2)]
        self.source_manifest.write_bytes(_jsonl_bytes(self.rows))
        self._write_source_receipt()

    def tearDown(self):
        self.context.cleanup()

    @staticmethod
    def _source_row(index):
        image_id = 100 + index
        anchor_id = 1000 + index * 2
        partner_id = anchor_id + 1
        source = f"fixture_{index}_train"
        anchor_expression = f"left fixture object {index}"
        partner_expression = f"right fixture object {index}"
        anchor_box = [1 + index, 2, 3, 4]
        partner_box = [10, 20 + index, 5, 6]
        return {
            "ann_id": anchor_id,
            "assignment_pair": {
                "anchor": {
                    "bbox": anchor_box,
                    "coco_ann_id": anchor_id,
                    "expression": anchor_expression,
                    "image_id": image_id,
                    "source": source,
                },
                "partner": {
                    "bbox": partner_box,
                    "coco_ann_id": partner_id,
                    "expression": partner_expression,
                    "image_id": image_id,
                    "source": source,
                    "target_iou": 0.0,
                },
                "schema": builder.SOURCE_ROW_SCHEMA,
            },
            "assignment_pair_invalid_reason": None,
            "assignment_pair_valid": True,
            "filename": f"/images/COCO_train2014_{image_id:012d}.jpg",
            "image_id": image_id,
            "instances": [
                {
                    "bbox": anchor_box,
                    "category_complete_primary": True,
                    "class_id": 7,
                    "coco_ann_id": anchor_id,
                    "raw_phrase": anchor_expression,
                    "text_is_negative": False,
                },
                {
                    "bbox": partner_box,
                    "category_complete_auxiliary": True,
                    "class_id": 7,
                    "coco_ann_id": partner_id,
                },
            ],
            "primary_support_instance_index": 0,
            "ref_id": 2000 + index,
            "sent_id": 3000 + index,
            "source": source,
            "split": "train",
            "stage_b_data_driven_assignment_pair": True,
            "stage_b_data_driven_assignment_pair_schema": builder.SOURCE_ROW_SCHEMA,
            "stage_b_u2_category_complete": True,
            "stage_b_u2_category_complete_schema": (
                "pivot.stageb.u2_category_complete_ref/v1"
            ),
        }

    def _source_members(self):
        members = []
        for index, (row, raw) in enumerate(
            zip(self.rows, self.source_manifest.read_bytes().splitlines(), strict=True)
        ):
            pair = row["assignment_pair"]
            members.append(
                {
                    "anchor_coco_ann_id": pair["anchor"]["coco_ann_id"],
                    "class_id": 7,
                    "image_id": row["image_id"],
                    "manifest": "fixture_source.jsonl",
                    "output_index": index,
                    "pair_id": builder._sha256_bytes(f"pair-{index}".encode("ascii")),
                    "partner_coco_ann_id": pair["partner"]["coco_ann_id"],
                    "priority_sha256": builder._sha256_bytes(
                        f"priority-{index}".encode("ascii")
                    ),
                    "ref_id": row["ref_id"],
                    "sent_id": row["sent_id"],
                    "source": row["source"],
                    "source_line_number": 50 + index,
                    "source_row_sha256": builder._sha256_bytes(raw),
                }
            )
        return members

    def _write_source_receipt(self, *, invariant_value=True):
        manifest_record = builder._file_record(self.source_manifest)
        members = self._source_members()
        pair_ids = [member["pair_id"] for member in members]
        image_ids = [str(member["image_id"]) for member in members]
        edges = []
        endpoints = []
        for member in members:
            edge = sorted(
                (
                    member["anchor_coco_ann_id"],
                    member["partner_coco_ann_id"],
                )
            )
            edges.append(f"{edge[0]}\t{edge[1]}")
            endpoints.append(
                f"{member['anchor_coco_ann_id']}\t{member['partner_coco_ann_id']}"
            )
        receipt = {
            "schema": builder.SOURCE_RECEIPT_SCHEMA,
            "row_schema": builder.SOURCE_ROW_SCHEMA,
            "rows": 2,
            "valid_rows": 2,
            "invalid_rows": 0,
            "unique_images": 2,
            "unique_unordered_annotation_edges": 2,
            "unique_annotation_endpoints": 4,
            "output_manifest": self.source_manifest.name,
            "output": manifest_record,
            "selection_contract": {
                "model_score_free": True,
                "forbidden_inputs": sorted(builder.FORBIDDEN_SOURCE_KEYS),
            },
            "ordered_member_stream_encoding": builder.STREAM_ENCODING,
            "ordered_member_pair_id_stream_sha256": builder._record_stream_sha256(
                pair_ids
            ),
            "ordered_image_id_stream_sha256": builder._record_stream_sha256(
                image_ids
            ),
            "ordered_unordered_edge_stream_sha256": builder._record_stream_sha256(
                edges
            ),
            "ordered_endpoint_stream_sha256": builder._record_stream_sha256(
                endpoints
            ),
            "members": members,
            "invariants": {"fixture_source_is_valid": invariant_value},
        }
        receipt["canonical_payload_sha256"] = builder._canonical_payload_sha256(
            receipt
        )
        self.source_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _kwargs(self, output_name="artifact"):
        return {
            "source_manifest": self.source_manifest,
            "source_receipt": self.source_receipt,
            "output_root": self.root / output_name,
            "expected_source_manifest_sha256": builder._sha256_file(
                self.source_manifest
            ),
            "expected_source_receipt_sha256": builder._sha256_file(
                self.source_receipt
            ),
            "expected_pairs": 2,
            "expected_invariants": self.expected_invariants,
        }

    def test_expands_each_pair_to_two_single_region_xyxy_rows(self):
        kwargs = self._kwargs()
        plan = builder.make_plan(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

        receipt = builder.build(**kwargs)
        output_path = kwargs["output_root"] / builder.OUTPUT_MANIFEST
        rows = [json.loads(line) for line in output_path.read_bytes().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [(row["pair_index"], row["direction"]) for row in rows],
            [(0, "anchor"), (0, "partner"), (1, "anchor"), (1, "partner")],
        )
        self.assertEqual(
            [row["grounding"]["regions"][0]["bbox"] for row in rows],
            [
                [1.0, 2.0, 4.0, 6.0],
                [10.0, 20.0, 15.0, 26.0],
                [2.0, 2.0, 5.0, 6.0],
                [10.0, 21.0, 15.0, 27.0],
            ],
        )
        self.assertEqual(
            [row["grounding"]["regions"][0]["phrase"] for row in rows],
            [
                "left fixture object 0",
                "right fixture object 0",
                "left fixture object 1",
                "right fixture object 1",
            ],
        )
        for row in rows:
            self.assertEqual(len(row["grounding"]["regions"]), 1)
            self.assertEqual(row["row_schema"], builder.OUTPUT_ROW_SCHEMA)
            self.assertEqual(
                row["source_manifest_sha256"],
                kwargs["expected_source_manifest_sha256"],
            )
            self.assertEqual(
                row["source_receipt_sha256"],
                kwargs["expected_source_receipt_sha256"],
            )
            self.assertEqual(
                row["source_row_sha256"],
                self._source_members()[row["pair_index"]]["source_row_sha256"],
            )

        self.assertEqual(receipt, plan.receipt)
        self.assertEqual(receipt["pairs"], 2)
        self.assertEqual(receipt["rows"], 4)
        self.assertEqual(receipt["direction_counts"], {"anchor": 2, "partner": 2})
        self.assertEqual(receipt["unique_images"], 2)
        self.assertEqual(receipt["unique_target_annotation_ids"], 4)
        self.assertTrue(all(receipt["invariants"].values()))
        self.assertEqual(receipt["output"]["sha256"], builder._sha256_file(output_path))
        self.assertEqual(builder.verify(**kwargs), receipt)

    def test_dataset_config_targets_the_default_odvg_artifact(self):
        config_path = (
            builder.REPO_ROOT
            / "config/datasets_stageb_gdino_adapter_rank_o64_direct.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["val"], [])
        self.assertEqual(len(config["train"]), 1)
        entry = config["train"][0]
        self.assertEqual(entry["dataset_mode"], "odvg")
        self.assertEqual(entry["root"], "/")
        self.assertEqual(entry["anno"], str(builder.OUTPUT_ROOT / builder.OUTPUT_MANIFEST))
        self.assertEqual(entry["mix_weight"], 1.0)

    def test_refuses_to_overwrite_an_existing_output_root(self):
        kwargs = self._kwargs("no_overwrite")
        builder.build(**kwargs)
        before = {
            path.name: path.read_bytes()
            for path in kwargs["output_root"].iterdir()
            if path.is_file()
        }
        with self.assertRaisesRegex(
            builder.O64DirectRankBuildError, "refusing to replace"
        ):
            builder.build(**kwargs)
        after = {
            path.name: path.read_bytes()
            for path in kwargs["output_root"].iterdir()
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_rejects_false_source_invariant_even_when_receipt_is_resealed(self):
        self._write_source_receipt(invariant_value=False)
        kwargs = self._kwargs("false_invariant")
        with self.assertRaisesRegex(
            builder.O64DirectRankBuildError, "counts or invariants drifted"
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_rejects_source_manifest_hash_drift_before_writing(self):
        kwargs = self._kwargs("source_drift")
        self.source_manifest.write_bytes(self.source_manifest.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(
            builder.O64DirectRankBuildError, "source manifest SHA-256 mismatch"
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_failed_atomic_publish_removes_the_temporary_directory(self):
        kwargs = self._kwargs("atomic")
        with patch.object(builder.os, "rename", side_effect=OSError("fixture failure")):
            with self.assertRaisesRegex(OSError, "fixture failure"):
                builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())
        self.assertEqual(list(self.root.glob(".atomic.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
