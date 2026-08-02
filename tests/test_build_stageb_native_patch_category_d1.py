import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_stageb_native_patch_category_d1 as builder


class NativePatchCategoryD1BuilderTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.partition_root = self.root / "partition"
        self.coco_root = self.root / "coco2014"
        self.support_root = self.root / "support"
        self.official_path = self.root / "official.jsonl"
        for path in (
            self.partition_root,
            self.coco_root / "train2014",
            self.coco_root / "val2014",
            self.support_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

        for image_id in (10, 11, 20, 30):
            path = (
                self.coco_root
                / "train2014"
                / f"COCO_train2014_{image_id:012d}.jpg"
            )
            path.write_bytes(f"query-image-{image_id}".encode("ascii"))

        self.rows = {split: {name: [] for name, _ in builder.SOURCE_MANIFESTS} for split in builder.SPLITS}
        for source_index, (manifest, source) in enumerate(builder.SOURCE_MANIFESTS):
            self.rows["train"][manifest].append(
                self._row(
                    source=source,
                    image_id=10,
                    category_id=1,
                    class_id=100,
                    ann_id=1000,
                    ref_id=100 + source_index,
                    sent_id=200 + source_index,
                    phrase=f"full text from {source}",
                )
            )
            dev_row = self._row(
                source=source,
                image_id=20,
                category_id=1,
                class_id=100,
                ann_id=2000,
                ref_id=300 + source_index,
                sent_id=400 + source_index,
                phrase=f"dev full text from {source}",
            )
            self.rows["dev_full"][manifest].append(dev_row)
            self.rows["dev_screen"][manifest].append(dict(dev_row))

        ref_manifest = builder.SOURCE_MANIFESTS[0][0]
        for variant in range(2):
            self.rows["train"][ref_manifest].append(
                self._row(
                    source="refcoco",
                    image_id=11,
                    category_id=2,
                    class_id=200,
                    ann_id=1100,
                    ref_id=500 + variant,
                    sent_id=600 + variant,
                    phrase=f"single-source variant {variant}",
                )
            )

        self.manifest_paths = {}
        for split in builder.SPLITS:
            split_root = self.partition_root / split
            split_root.mkdir()
            for manifest, _source in builder.SOURCE_MANIFESTS:
                path = split_root / manifest
                self._write_jsonl(path, self.rows[split][manifest])
                self.manifest_paths[(split, manifest)] = path

        self._write_jsonl(
            self.official_path,
            [
                {
                    "filename": "/fixture/COCO_train2014_000000000030.jpg",
                    "image_id": 30,
                }
            ],
        )
        self.partition_receipt = self.root / "partition_receipt.json"
        self._write_partition_receipt()

        self.support_tsv = self.root / "filtered_support.tsv"
        self.support_specs = [
            (100, 10, "same-query-identity"),
            (100, 40, "person-support-a"),
            (100, 41, "person-support-b"),
            (100, 42, "person-support-c"),
            (200, 50, "single-car-support"),
        ]
        self._write_support_tsv()
        self.support_receipt = self.root / "support_receipt.json"
        self._write_support_receipt()

    def tearDown(self):
        self.context.cleanup()

    @staticmethod
    def _row(
        *, source, image_id, category_id, class_id, ann_id, ref_id, sent_id, phrase
    ):
        filename = f"/fixture/COCO_train2014_{image_id:012d}.jpg"
        return {
            "ann_id": ann_id,
            "category_complete_coco_category_id": category_id,
            "category_complete_coco_split": "train2014",
            "category_complete_instance_count": 2,
            "filename": filename,
            "image_id": image_id,
            "instances": [
                {
                    "bbox": [1.0, 2.0, 10.0, 12.0],
                    "category_complete_primary": True,
                    "class_id": class_id,
                    "coco_ann_id": ann_id,
                    "positive_phrase": phrase,
                    "raw_phrase": phrase,
                    "refcoco_category_id": category_id,
                    "text_is_negative": False,
                },
                {
                    "bbox": [20.0, 22.0, 5.0, 6.0],
                    "category_complete_auxiliary": True,
                    "class_id": class_id,
                    "coco_ann_id": ann_id + 900000,
                    "refcoco_category_id": category_id,
                },
            ],
            "primary_support_instance_index": 0,
            "ref_id": ref_id,
            "sent_id": sent_id,
            "source": f"{source}_train",
            "split": "train",
            "stage_b_u2_category_complete": True,
            "stage_b_u2_category_complete_schema": builder.UPSTREAM_ROW_SCHEMA,
        }

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_bytes(
            b"".join(builder._canonical_bytes(row) + b"\n" for row in rows)
        )

    def _write_partition_receipt(self):
        outputs = {"d1_category_complete": {}}
        summaries = {}
        for split in builder.SPLITS:
            outputs["d1_category_complete"][split] = {}
            image_keys = set()
            total_rows = 0
            for manifest, _source in builder.SOURCE_MANIFESTS:
                path = self.manifest_paths[(split, manifest)]
                rows = self.rows[split][manifest]
                for row in rows:
                    image_keys.add(("train2014", row["image_id"]))
                outputs["d1_category_complete"][split][manifest] = {
                    **builder._file_record(path),
                    "rows": len(rows),
                }
                total_rows += len(rows)
            summaries[split] = {
                "rows": total_rows,
                "rows_by_manifest": {
                    name: len(self.rows[split][name])
                    for name, _source in builder.SOURCE_MANIFESTS
                },
                "unique_image_keys": len(image_keys),
                "ordered_image_key_stream_sha256": builder._record_stream_sha256(
                    [builder._image_key_text(key) for key in sorted(image_keys)]
                ),
            }
        official_rows = self._read_jsonl(self.official_path)
        official_keys = sorted(
            {
                builder._image_key(row, context="fixture official")[0]
                for row in official_rows
            }
        )
        official_record = builder._file_record(self.official_path)
        receipt = {
            "schema": builder.UPSTREAM_PARTITION_SCHEMA,
            "source_manifest_order": [
                name for name, _source in builder.SOURCE_MANIFESTS
            ],
            "partition_summary": summaries,
            "dev_full_members": [
                {
                    "coco_split": "train2014",
                    "image_id": 20,
                    "image_key": "train2014:000000000020",
                }
            ],
            "dev_screen_members": [
                {
                    "coco_split": "train2014",
                    "image_id": 20,
                    "image_key": "train2014:000000000020",
                }
            ],
            "official_ref8": {
                "split_order": ["fixture_ref"],
                "splits": {
                    "fixture_ref": {
                        "rows": len(official_rows),
                        "manifest": official_record,
                    }
                },
                "rows": len(official_rows),
                "unique_image_keys": len(official_keys),
                "ordered_image_key_stream_sha256": builder._record_stream_sha256(
                    [builder._image_key_text(key) for key in official_keys]
                ),
            },
            "outputs": outputs,
            "invariants": {
                "fixture_image_partition_is_sealed": True,
                "fixture_category_rows_are_complete": True,
            },
        }
        receipt["canonical_payload_sha256"] = hashlib.sha256(
            builder._canonical_bytes(receipt)
        ).hexdigest()
        self.partition_receipt.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _write_support_tsv(self):
        lines = ["\t".join(builder.SUPPORT_COLUMNS) + "\n"]
        for index, (class_id, coco_id, payload) in enumerate(self.support_specs):
            path = self.support_root / f"support_{index}.jpg"
            path.write_bytes(payload.encode("ascii"))
            row = {
                "path": str(path.resolve()),
                "class_id": str(class_id),
                "source_cache_class_id": str(class_id),
                "class_assignment": "sealed_cache_identity_v1",
                "source": "lvis",
                "source_image_id": str(coco_id),
                "coco_id": str(coco_id),
                "source_class": f"class_{class_id}",
                "source_row_number": str(1000 + index),
                "source_row_sha256": hashlib.sha256(
                    f"raw-source-{index}".encode("ascii")
                ).hexdigest(),
            }
            lines.append("\t".join(row[column] for column in builder.SUPPORT_COLUMNS) + "\n")
        self.support_tsv.write_text("".join(lines), encoding="utf-8")

    def _write_support_receipt(self):
        receipt = {
            "schema": builder.UPSTREAM_SUPPORT_SCHEMA,
            "inputs": {
                "partition_receipt": builder._file_record(self.partition_receipt)
            },
            "outputs": {
                "runtime_support_tsv": {
                    **builder._file_record(self.support_tsv),
                    "rows": len(self.support_specs),
                }
            },
            "exclusion": {
                "union_numeric_coco_ids": [20, 30],
                "union_numeric_coco_id_count": 2,
            },
            "invariants": {
                "fixture_support_is_train_filtered": True,
                "fixture_support_is_runtime_exact": True,
            },
        }
        receipt["canonical_payload_sha256"] = hashlib.sha256(
            builder._canonical_bytes(receipt)
        ).hexdigest()
        self.support_receipt.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _refresh_receipts(self):
        self._write_partition_receipt()
        self._write_support_receipt()

    def _kwargs(self, output_name="artifact"):
        return {
            "partition_receipt": self.partition_receipt,
            "support_receipt": self.support_receipt,
            "support_tsv": self.support_tsv,
            "coco_image_root": self.coco_root,
            "output_root": self.root / output_name,
            "expected_partition_receipt_sha256": builder._sha256_file(
                self.partition_receipt
            ),
            "expected_support_receipt_sha256": builder._sha256_file(
                self.support_receipt
            ),
            "expected_support_tsv_sha256": builder._sha256_file(self.support_tsv),
        }

    @staticmethod
    def _read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_builds_three_variants_per_group_with_explicit_support(self):
        kwargs = self._kwargs()
        receipt = builder.build(**kwargs)
        output_root = kwargs["output_root"]
        train = self._read_jsonl(output_root / builder.OUTPUT_FILES["train"])
        dev_screen = self._read_jsonl(
            output_root / builder.OUTPUT_FILES["dev_screen"]
        )
        dev_full = self._read_jsonl(output_root / builder.OUTPUT_FILES["dev_full"])
        self.assertEqual(len(train), 6)
        self.assertEqual(len(dev_screen), 3)
        self.assertEqual(len(dev_full), 3)

        by_group = {}
        for row in train:
            by_group.setdefault(row["native_patch_category_group_id"], []).append(row)
            self.assertEqual(row["primary_support_instance_index"], 0)
            self.assertTrue(row["instances"][0]["category_complete_primary"])
            self.assertEqual(len(row["instances"]), 2)
            witness = row["support_patch_witness"]
            self.assertEqual(witness["class_id"], row["instances"][0]["class_id"])
            self.assertNotEqual(witness["coco_id"], row["image_id"])
            self.assertNotEqual(
                witness["content_sha256"],
                row["query_image_witness"]["content_sha256"],
            )
            self.assertTrue(witness["train_filtered"])
        self.assertEqual(sorted(map(len, by_group.values())), [3, 3])

        all_source_group = next(
            rows for rows in by_group.values() if rows[0]["image_id"] == 10
        )
        self.assertEqual(
            {row["native_patch_category_source_dataset"] for row in all_source_group},
            set(builder.PREFERRED_SOURCES),
        )
        single_source_group = next(
            rows for rows in by_group.values() if rows[0]["image_id"] == 11
        )
        self.assertEqual(
            {row["native_patch_category_source_dataset"] for row in single_source_group},
            {"refcoco"},
        )
        self.assertIn(
            "hash_cycle_repeat",
            {row["native_patch_category_variant_selection"] for row in single_source_group},
        )
        self.assertEqual(
            {row["native_patch_category_group_id"] for row in dev_screen},
            {row["native_patch_category_group_id"] for row in dev_full},
        )
        self.assertTrue(all(receipt["invariants"].values()))
        self.assertEqual(receipt["splits"]["train"]["groups"], 2)
        self.assertGreaterEqual(
            receipt["splits"]["train"]["support_reuse_fallback_rows"], 2
        )

    def test_create_new_verify_and_tamper_detection(self):
        kwargs = self._kwargs("replay")
        first = builder.build(**kwargs)
        self.assertEqual(builder.verify(**kwargs), first)
        with self.assertRaisesRegex(builder.NativePatchCategoryD1Error, "refusing"):
            builder.build(**kwargs)
        train_path = kwargs["output_root"] / builder.OUTPUT_FILES["train"]
        train_path.write_bytes(train_path.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error, "does not replay byte-for-byte"
        ):
            builder.verify(**kwargs)

    def test_recursive_model_derived_field_is_rejected(self):
        manifest = builder.SOURCE_MANIFESTS[0][0]
        self.rows["train"][manifest][0]["nested"] = {"teacher_score_v2": [0.9]}
        self._write_jsonl(
            self.manifest_paths[("train", manifest)], self.rows["train"][manifest]
        )
        self._refresh_receipts()
        kwargs = self._kwargs("forbidden")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error, "forbidden model-derived field"
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_boolean_false_cannot_impersonate_primary_index_zero(self):
        manifest = builder.SOURCE_MANIFESTS[0][0]
        self.rows["train"][manifest][0]["primary_support_instance_index"] = False
        self._write_jsonl(
            self.manifest_paths[("train", manifest)], self.rows["train"][manifest]
        )
        self._refresh_receipts()
        kwargs = self._kwargs("boolean_primary")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error,
            "category-complete marker contract drifted",
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_support_must_differ_from_query_content_even_with_other_id(self):
        query_content = (
            self.coco_root
            / "train2014"
            / "COCO_train2014_000000000010.jpg"
        ).read_bytes()
        self.support_specs = [(100, 40, query_content.decode("ascii")), (200, 50, "ok")]
        self._write_support_tsv()
        self._write_support_receipt()
        kwargs = self._kwargs("content_collision")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error, "different-image/content support"
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_support_exclusion_union_must_replay_dev_and_official(self):
        receipt = json.loads(self.support_receipt.read_text(encoding="ascii"))
        receipt["exclusion"]["union_numeric_coco_ids"] = [30]
        receipt["exclusion"]["union_numeric_coco_id_count"] = 1
        del receipt["canonical_payload_sha256"]
        receipt["canonical_payload_sha256"] = hashlib.sha256(
            builder._canonical_bytes(receipt)
        ).hexdigest()
        self.support_receipt.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        kwargs = self._kwargs("bad_support_union")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error,
            "support exclusion union does not equal",
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_official_ref8_overlap_is_rejected(self):
        self._write_jsonl(
            self.official_path,
            [
                {
                    "filename": "/fixture/COCO_train2014_000000000010.jpg",
                    "image_id": 10,
                }
            ],
        )
        self._refresh_receipts()
        kwargs = self._kwargs("official_overlap")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error,
            "D1 train contains an official Ref8 image",
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_receipt_canonical_payload_drift_is_rejected(self):
        receipt = json.loads(self.partition_receipt.read_text(encoding="ascii"))
        receipt["partition_summary"]["train"]["rows"] += 1
        self.partition_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        kwargs = self._kwargs("receipt_drift")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error, "canonical payload does not replay"
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_failed_atomic_publish_removes_temporary_directory(self):
        kwargs = self._kwargs("atomic")
        with patch.object(
            builder, "_rename_noreplace", side_effect=OSError("fixture failure")
        ):
            with self.assertRaisesRegex(OSError, "fixture failure"):
                builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())
        self.assertEqual(list(self.root.glob(".atomic.tmp-*")), [])

    def test_dev_screen_group_content_must_equal_dev_full(self):
        for manifest, _source in builder.SOURCE_MANIFESTS:
            self.rows["dev_screen"][manifest][0]["instances"][1]["bbox"][0] = 99.0
            self._write_jsonl(
                self.manifest_paths[("dev_screen", manifest)],
                self.rows["dev_screen"][manifest],
            )
        self._refresh_receipts()
        kwargs = self._kwargs("screen_drift")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error,
            "dev_screen group content/source rows differ",
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_atomic_publish_never_replaces_concurrent_destination(self):
        kwargs = self._kwargs("concurrent")

        def create_destination(_plan):
            kwargs["output_root"].mkdir()
            (kwargs["output_root"] / "sentinel").write_text("keep", encoding="ascii")

        with patch.object(builder, "_assert_inputs_unchanged", side_effect=create_destination):
            with self.assertRaisesRegex(
                builder.NativePatchCategoryD1Error, "concurrent overwrite"
            ):
                builder.build(**kwargs)
        self.assertEqual(
            (kwargs["output_root"] / "sentinel").read_text(encoding="ascii"), "keep"
        )
        self.assertEqual(list(self.root.glob(".concurrent.tmp-*")), [])

    def test_verify_rejects_extra_artifact_entry(self):
        kwargs = self._kwargs("extra_entry")
        builder.build(**kwargs)
        (kwargs["output_root"] / "extra.bin").write_bytes(b"not sealed")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD1Error, "entry set does not replay exactly"
        ):
            builder.verify(**kwargs)


if __name__ == "__main__":
    unittest.main()
