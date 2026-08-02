import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from tools import build_stageb_native_patch_category_d2 as builder


class NativePatchCategoryD2BuilderTest(unittest.TestCase):
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

        for image_id in (10, 11, 12, 20, 30):
            query_path = (
                self.coco_root
                / "train2014"
                / f"COCO_train2014_{image_id:012d}.jpg"
            )
            query_path.write_bytes(f"query-image-{image_id}".encode("ascii"))

        self.rows = {
            split: {name: [] for name, _source in builder.SOURCE_MANIFESTS}
            for split in builder.SPLITS
        }
        ref_manifest = builder.SOURCE_MANIFESTS[0][0]
        plus_manifest = builder.SOURCE_MANIFESTS[1][0]
        g_manifest = builder.SOURCE_MANIFESTS[2][0]

        for variant in range(3):
            self.rows["train"][ref_manifest].append(
                self._row(
                    source="refcoco",
                    image_id=10,
                    category_id=1,
                    class_id=100,
                    ann_id=1000,
                    ref_id=100 + variant,
                    sent_id=200 + variant,
                    phrase=f"ref person expression {variant}",
                )
            )
        for variant in range(2):
            self.rows["train"][ref_manifest].append(
                self._row(
                    source="refcoco",
                    image_id=11,
                    category_id=2,
                    class_id=200,
                    ann_id=1100,
                    ref_id=300 + variant,
                    sent_id=400 + variant,
                    phrase=f"ref vehicle expression {variant}",
                )
            )
        self.rows["train"][ref_manifest].append(
            self._row(
                source="refcoco",
                image_id=12,
                category_id=1,
                class_id=100,
                ann_id=1200,
                ref_id=500,
                sent_id=600,
                phrase="second ref person group",
            )
        )

        for manifest, source in (
            (plus_manifest, "refcocoplus"),
            (g_manifest, "refcocog"),
        ):
            self.rows["train"][manifest].extend(
                [
                    self._row(
                        source=source,
                        image_id=10,
                        category_id=1,
                        class_id=100,
                        ann_id=1000,
                        ref_id=700 + len(self.rows["train"][manifest]),
                        sent_id=800 + len(self.rows["train"][manifest]),
                        phrase=f"{source} person expression",
                    ),
                    self._row(
                        source=source,
                        image_id=11,
                        category_id=2,
                        class_id=200,
                        ann_id=1100,
                        ref_id=900 + len(self.rows["train"][manifest]),
                        sent_id=1000 + len(self.rows["train"][manifest]),
                        phrase=f"{source} vehicle expression",
                    ),
                ]
            )

        for source_index, (manifest, source) in enumerate(
            builder.SOURCE_MANIFESTS
        ):
            row = self._row(
                source=source,
                image_id=20,
                category_id=1,
                class_id=100,
                ann_id=2000,
                ref_id=1200 + source_index,
                sent_id=1300 + source_index,
                phrase=f"dev expression from {source}",
            )
            self.rows["dev_full"][manifest].append(row)
            self.rows["dev_screen"][manifest].append(dict(row))

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
            (200, 11, "same-vehicle-query-identity"),
            (200, 50, "vehicle-support-a"),
            (200, 51, "vehicle-support-b"),
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
        return {
            "ann_id": ann_id,
            "category_complete_coco_category_id": category_id,
            "category_complete_coco_split": "train2014",
            "category_complete_instance_count": 2,
            "filename": f"/fixture/COCO_train2014_{image_id:012d}.jpg",
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

    @staticmethod
    def _read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def _write_partition_receipt(self):
        outputs = {"d1_category_complete": {}}
        summaries = {}
        for split in builder.SPLITS:
            outputs["d1_category_complete"][split] = {}
            image_keys = set()
            total_rows = 0
            for manifest, _source in builder.SOURCE_MANIFESTS:
                rows = self.rows[split][manifest]
                path = self.manifest_paths[(split, manifest)]
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
                ("train2014", int(row["image_id"]))
                for row in official_rows
            }
        )
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
                        "manifest": builder._file_record(self.official_path),
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
            lines.append(
                "\t".join(row[column] for column in builder.SUPPORT_COLUMNS) + "\n"
            )
        self.support_tsv.write_text("".join(lines), encoding="utf-8")

    def _write_support_receipt(self):
        receipt = {
            "schema": builder.UPSTREAM_SUPPORT_SCHEMA,
            "inputs": {
                "partition_receipt": builder._file_record(
                    self.partition_receipt
                )
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
            "expected_support_tsv_sha256": builder._sha256_file(
                self.support_tsv
            ),
        }

    def test_preserves_every_expression_in_source_manifests(self):
        kwargs = self._kwargs()
        receipt = builder.build(**kwargs)
        output_root = kwargs["output_root"]
        expected_train_rows = {
            "refcoco": 6,
            "refcocoplus": 2,
            "refcocog": 2,
        }
        for source, expected in expected_train_rows.items():
            rows = self._read_jsonl(
                output_root / builder.OUTPUT_FILES["train"][source]
            )
            self.assertEqual(len(rows), expected)
            self.assertEqual(
                {row["native_patch_category_source_dataset"] for row in rows},
                {source},
            )
            for row in rows:
                self.assertTrue(row["stage_b_native_patch_category_d2"])
                self.assertEqual(
                    row["stage_b_native_patch_category_d2_schema"],
                    builder.ROW_SCHEMA,
                )
                self.assertNotIn("stage_b_u2_category_complete", row)
                self.assertNotIn("stage_b_native_patch_category_d1", row)
                witness = row["support_patch_witness"]
                self.assertEqual(
                    witness["class_id"], row["native_patch_category_class_id"]
                )
                self.assertNotEqual(witness["coco_id"], row["image_id"])
                self.assertNotEqual(
                    witness["content_sha256"],
                    row["query_image_witness"]["content_sha256"],
                )
                self.assertEqual(
                    witness["rotation_key_sha256"],
                    builder._support_rotation_key(
                        row["native_patch_category_group_id"],
                        row["native_patch_category_source_identity_sha256"],
                    ),
                )
        self.assertEqual(
            receipt["sampling_contract"]["source_mix_weights"],
            builder.SOURCE_MIX_WEIGHTS,
        )
        self.assertTrue(all(receipt["invariants"].values()))

    def test_group_dedup_weights_and_source_mix_are_exact(self):
        kwargs = self._kwargs("weights")
        receipt = builder.build(**kwargs)
        rows = self._read_jsonl(
            kwargs["output_root"] / builder.OUTPUT_FILES["train"]["refcoco"]
        )
        self.assertTrue(
            math.isclose(
                math.fsum(row[builder.SAMPLING_WEIGHT_FIELD] for row in rows),
                len(rows),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        by_group = {}
        for row in rows:
            by_group.setdefault(row["native_patch_category_group_id"], []).append(
                row
            )
        group10 = next(group for group in by_group.values() if group[0]["image_id"] == 10)
        group12 = next(group for group in by_group.values() if group[0]["image_id"] == 12)
        self.assertEqual(len(group10), 3)
        self.assertEqual(len(group12), 1)
        self.assertTrue(
            math.isclose(
                math.fsum(row[builder.SAMPLING_WEIGHT_FIELD] for row in group10),
                group12[0][builder.SAMPLING_WEIGHT_FIELD],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            all(
                row["native_patch_category_source_group_expression_count"] == 3
                for row in group10
            )
        )
        normalization = 6.0 / (2.0 + math.sqrt(2.0))
        self.assertTrue(
            all(
                math.isclose(
                    row[builder.SAMPLING_WEIGHT_FIELD],
                    normalization / 3.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for row in group10
            )
        )
        expected_weight_sha = hashlib.sha256(
            b"".join(
                struct.pack("<d", row[builder.SAMPLING_WEIGHT_FIELD])
                for row in rows
            )
        ).hexdigest()
        self.assertEqual(
            receipt["splits"]["train"]["refcoco"]
            ["sampling_weight_float64_le_stream_sha256"],
            expected_weight_sha,
        )

        source_masses = {}
        for source in builder.SOURCES:
            source_rows = self._read_jsonl(
                kwargs["output_root"] / builder.OUTPUT_FILES["train"][source]
            )
            source_masses[source] = (
                builder.SOURCE_MIX_WEIGHTS[source]
                / len(source_rows)
                * math.fsum(
                    row[builder.SAMPLING_WEIGHT_FIELD] for row in source_rows
                )
            )
        self.assertEqual(source_masses, builder.SOURCE_MIX_WEIGHTS)

    def test_capped_sqrt_class_multiplier(self):
        groups = {}
        for index in range(26):
            class_id = 1 if index < 25 else 2
            group_key = ("train2014", index + 100, class_id)
            group = builder.GroupState(
                key=group_key,
                group_id=builder._group_id(group_key),
                filename=f"image-{index}.jpg",
                query_path=Path(f"/unused/{index}.jpg"),
                class_id=class_id,
                instance_set_sha256=hashlib.sha256(
                    f"instance-{index}".encode("ascii")
                ).hexdigest(),
                instance_count=1,
            )
            ref = builder.ExpressionRef(
                partition="train",
                manifest=builder.SOURCE_MANIFESTS[0][0],
                source_dataset="refcoco",
                line_number=index + 1,
                raw_sha256=hashlib.sha256(
                    f"raw-{index}".encode("ascii")
                ).hexdigest(),
                identity_sha256=hashlib.sha256(
                    f"identity-{index}".encode("ascii")
                ).hexdigest(),
                full_text=f"expression {index}",
                ann_id=index,
                ref_id=index,
                sent_id=index,
            )
            group.expressions["refcoco"].append(ref)
            # The helper requires all formal sources to be represented.
            for source_index, source in enumerate(builder.SOURCES[1:], start=1):
                extra = builder.ExpressionRef(
                    partition="train",
                    manifest=builder.SOURCE_MANIFESTS[source_index][0],
                    source_dataset=source,
                    line_number=index + 1,
                    raw_sha256=hashlib.sha256(
                        f"raw-{source}-{index}".encode("ascii")
                    ).hexdigest(),
                    identity_sha256=hashlib.sha256(
                        f"identity-{source}-{index}".encode("ascii")
                    ).hexdigest(),
                    full_text=f"{source} expression {index}",
                    ann_id=index,
                    ref_id=index,
                    sent_id=index,
                )
                group.expressions[source].append(extra)
            groups[group_key] = group
        weights, summary = builder._sampling_weights_for_groups(groups)
        rare_ref = groups[("train2014", 125, 2)].expressions["refcoco"][0]
        self.assertEqual(weights[rare_ref.key].class_balance_multiplier, 4.0)
        self.assertEqual(
            summary["refcoco"]["class_group_counts"], {"1": 25, "2": 1}
        )

    def test_build_verify_and_tamper_detection(self):
        kwargs = self._kwargs("replay")
        first = builder.build(**kwargs)
        self.assertEqual(builder.verify(**kwargs), first)
        path = kwargs["output_root"] / builder.OUTPUT_FILES["train"]["refcoco"]
        path.write_bytes(path.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD2Error, "does not replay byte-for-byte"
        ):
            builder.verify(**kwargs)

    def test_content_duplicate_is_quarantined_and_fully_accounted(self):
        train_image = (
            self.coco_root
            / "train2014"
            / "COCO_train2014_000000000010.jpg"
        )
        dev_image = (
            self.coco_root
            / "train2014"
            / "COCO_train2014_000000000020.jpg"
        )
        train_image.write_bytes(dev_image.read_bytes())

        kwargs = self._kwargs("quarantine")
        receipt = builder.build(**kwargs)
        output_root = kwargs["output_root"]
        expected_train_rows = {
            "refcoco": 3,
            "refcocoplus": 1,
            "refcocog": 1,
        }
        dev_content_sha = hashlib.sha256(dev_image.read_bytes()).hexdigest()
        for source, expected in expected_train_rows.items():
            rows = self._read_jsonl(
                output_root / builder.OUTPUT_FILES["train"][source]
            )
            self.assertEqual(len(rows), expected)
            self.assertNotIn(
                dev_content_sha,
                {row["query_image_witness"]["content_sha256"] for row in rows},
            )

        quarantine_rows = self._read_jsonl(
            output_root / builder.QUARANTINE_FILE
        )
        self.assertEqual(len(quarantine_rows), 5)
        self.assertEqual({row["image_id"] for row in quarantine_rows}, {10})
        for row in quarantine_rows:
            self.assertTrue(
                row[
                    "stage_b_native_patch_category_d2_content_overlap_quarantine"
                ]
            )
            self.assertNotIn("stage_b_native_patch_category_d2", row)
            self.assertNotIn(builder.SAMPLING_WEIGHT_FIELD, row)
            self.assertNotIn("support_patch_witness", row)

        quarantine = receipt["content_overlap_quarantine"]
        self.assertEqual(quarantine["rows"], 5)
        self.assertEqual(
            quarantine["source_rows"],
            {"refcoco": 3, "refcocoplus": 1, "refcocog": 1},
        )
        relationships = receipt["split_relationships"]
        self.assertEqual(
            relationships[
                "detected_raw_train_dev_full_content_sha256_overlap"
            ],
            1,
        )
        self.assertEqual(
            relationships["eligible_train_dev_full_content_sha256_overlap"],
            0,
        )
        self.assertTrue(all(receipt["invariants"].values()))
        self.assertEqual(builder.verify(**kwargs), receipt)

    def test_recursive_model_derived_field_is_rejected(self):
        manifest = builder.SOURCE_MANIFESTS[0][0]
        self.rows["train"][manifest][0]["nested"] = {"teacher_score_v2": 0.9}
        self._write_jsonl(
            self.manifest_paths[("train", manifest)], self.rows["train"][manifest]
        )
        self._refresh_receipts()
        kwargs = self._kwargs("forbidden")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD2Error,
            "forbidden model-derived field",
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_support_content_cannot_match_any_output_query(self):
        collision = (
            self.coco_root
            / "train2014"
            / "COCO_train2014_000000000011.jpg"
        ).read_text(encoding="ascii")
        self.support_specs = [
            (100, 40, collision),
            (100, 41, collision),
            (200, 50, "vehicle-support"),
        ]
        self._write_support_tsv()
        self._write_support_receipt()
        kwargs = self._kwargs("content_leak")
        with self.assertRaisesRegex(
            builder.NativePatchCategoryD2Error,
            "different-image/content support",
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_official_overlap_is_rejected(self):
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
            builder.NativePatchCategoryD2Error,
            "contains an official Ref8 image",
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())


if __name__ == "__main__":
    unittest.main()
