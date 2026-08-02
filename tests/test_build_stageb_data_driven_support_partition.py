import csv
import hashlib
import io
import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools import build_stageb_data_driven_support_partition as builder


def _jsonl_bytes(rows):
    return b"".join(builder._canonical_bytes(row) + b"\n" for row in rows)


class SupportPartitionBuilderTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.support_image_root = self.root / "patches_quality"
        self.upstream_patch_root = self.root / "upstream_patches"
        self.canonical_classes = self.root / "canonical.json"
        self.canonical_classes.write_text(
            json.dumps(
                [
                    {
                        "id": 10,
                        "raw_name": "apple",
                        "norm_name": "apple",
                        "base_name": None,
                        "synset": "apple.n.01",
                        "synonyms": ["apples"],
                        "aliases": [{"name": "apple", "norm_name": "apple"}],
                    },
                    {
                        "id": 20,
                        "raw_name": "car",
                        "norm_name": "car",
                        "base_name": None,
                        "synset": "car.n.01",
                        "synonyms": ["automobile"],
                        "aliases": [{"name": "car", "norm_name": "car"}],
                    },
                    {
                        "id": 30,
                        "raw_name": "automobile",
                        "norm_name": "automobile",
                        "base_name": None,
                        "synset": None,
                        "synonyms": [],
                        "aliases": [],
                    },
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )

        self.specs = [
            # class, source, source image id, raw/cache order label
            ("apple", "vg_patches", 2, "vg_null_keep"),
            ("apple", "lvis_patches", 101, "lvis_dev_drop"),
            ("car", "vg_patches", 1, "vg_ref8_drop"),
            ("apple", "lvis_patches", 999, "lvis_keep"),
            ("car", "vg_patches", 3, "vg_keep"),
        ]
        self.mirror_paths = {}
        raw_rows = []
        for index, (class_name, source_dir, image_id, label) in enumerate(self.specs):
            basename = f"{class_name}_{image_id}_{5000 + index}_{index}.jpg"
            mirror = (
                self.support_image_root
                / "clean"
                / source_dir
                / class_name
                / basename
            )
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_bytes(f"image-{label}".encode("ascii"))
            self.mirror_paths[label] = mirror.resolve()
            original = self.upstream_patch_root / source_dir / class_name / basename
            raw_rows.append(
                [
                    str(original),
                    class_name,
                    "0",
                    "0",
                    "1",
                    "clean",
                    f"clean/{source_dir}/{class_name}/{Path(basename).stem}.npy",
                    "256",
                    "float16",
                ]
            )
        raw_rows.append(
            [
                str(self.upstream_patch_root / "vg_patches/apple/apple_2_9999_9.jpg"),
                "apple",
                "0",
                "0",
                "1",
                "borderline",
                "borderline/vg_patches/apple/apple_2_9999_9.npy",
                "256",
                "float16",
            ]
        )
        self.support_tsv = self.root / "support.tsv"
        header = (
            b"path\tclass\tocclusion\tblur\tclass_confidence\tbucket\t"
            b"emb_rel_path\tdim\tdtype\r\n"
        )
        encoded_rows = [
            ("\t".join(row) + "\r\n").encode("utf-8") for row in raw_rows
        ]
        self.raw_input_rows = encoded_rows
        self.support_tsv.write_bytes(header + b"".join(encoded_rows))

        self.support_cache = self.root / "support.tsv.bank.clean.img.pkl"
        bank = {
            10: [
                str(self.mirror_paths["lvis_dev_drop"]),
                str(self.mirror_paths["lvis_keep"]),
                str(self.mirror_paths["vg_null_keep"]),
            ],
            20: [
                str(self.mirror_paths["vg_ref8_drop"]),
                str(self.mirror_paths["vg_keep"]),
            ],
        }
        cache = {
            "meta": {
                "version": 3,
                "tsv_path": str(self.support_tsv.resolve()),
                "bucket": "clean",
                "use_embedding": False,
                "max_per_class": 200,
                "support_patch_image_root": str(self.support_image_root.resolve()),
                "canonical_classes_json": str(self.canonical_classes.resolve()),
                "patch_class_map_json": None,
            },
            "bank": bank,
        }
        with self.support_cache.open("wb") as handle:
            pickle.dump(cache, handle, protocol=4)

        self.vg_json = self.root / "image_data.json"
        self.vg_zip = self.root / "image_data.zip"
        self._write_vg_metadata(
            [
                {"image_id": 1, "coco_id": 202},
                {"image_id": 2, "coco_id": None},
                {"image_id": 3, "coco_id": 303},
            ]
        )
        self.partition_receipt = self._write_partition_receipt()

    def tearDown(self):
        self.context.cleanup()

    def _write_vg_metadata(self, rows):
        payload = (json.dumps(rows, sort_keys=True) + "\n").encode("ascii")
        self.vg_json.write_bytes(payload)
        with zipfile.ZipFile(self.vg_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("image_data.json", payload)

    def _write_partition_receipt(self):
        official_path = self.root / "official_ref.jsonl"
        official_path.write_bytes(
            _jsonl_bytes(
                [
                    {
                        "image_id": 202,
                        "filename": "/coco/COCO_val2014_000000000202.jpg",
                    }
                ]
            )
        )
        train_path = self.root / "d1_train.jsonl"
        train_path.write_bytes(
            _jsonl_bytes(
                [
                    {
                        "primary_support_instance_index": 0,
                        "instances": [{"class_id": 10, "canonical_name": "apple"}],
                    },
                    {
                        "primary_support_instance_index": 0,
                        "instances": [{"class_id": 20, "canonical_name": "car"}],
                    },
                    {
                        "primary_support_instance_index": 0,
                        "instances": [
                            {"class_id": 30, "canonical_name": "automobile"}
                        ],
                    },
                ]
            )
        )
        receipt = {
            "schema": builder.PARTITION_SCHEMA,
            "source_manifest_order": ["fixture.jsonl"],
            "selection_contract": {"dev_full_target_images": 1},
            "partition_summary": {
                "dev_full": {"unique_image_keys": 1, "rows": 1},
                "train": {"unique_image_keys": 2, "rows": 3},
            },
            "dev_full_members": [
                {
                    "coco_split": "train2014",
                    "image_id": 101,
                    "image_key": "train2014:000000000101",
                }
            ],
            "official_ref8": {
                "split_order": ["fixture_ref"],
                "splits": {
                    "fixture_ref": {
                        "rows": 1,
                        "manifest": builder._file_record(official_path),
                    }
                },
                "rows": 1,
                "unique_image_keys": 1,
            },
            "outputs": {
                "d1_category_complete": {
                    "train": {
                        "fixture.jsonl": {
                            **builder._file_record(train_path),
                            "rows": 3,
                        }
                    }
                }
            },
            "invariants": {"fixture_partition_is_valid": True},
        }
        receipt["canonical_payload_sha256"] = hashlib.sha256(
            builder._canonical_bytes(receipt)
        ).hexdigest()
        path = self.root / "partition_receipt.json"
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        return path

    def _kwargs(self, output_name="output"):
        return {
            "support_tsv": self.support_tsv,
            "support_cache": self.support_cache,
            "support_image_root": self.support_image_root,
            "canonical_classes": self.canonical_classes,
            "vg_metadata_zip": self.vg_zip,
            "vg_metadata_json": self.vg_json,
            "partition_receipt": self.partition_receipt,
            "output_root": self.root / output_name,
            "expected_sha256": {
                "support_tsv": builder._sha256_file(self.support_tsv),
                "support_cache": builder._sha256_file(self.support_cache),
                "canonical_classes": builder._sha256_file(self.canonical_classes),
                "vg_metadata_zip": builder._sha256_file(self.vg_zip),
                "vg_metadata_json": builder._sha256_file(self.vg_json),
            },
            "expected_partition_receipt_sha256": builder._sha256_file(
                self.partition_receipt
            ),
            "expected_raw_rows": len(self.raw_input_rows),
            "expected_raw_clean_rows": len(self.raw_input_rows) - 1,
            "expected_cache_classes": 2,
            "expected_cache_candidates": 5,
            "expected_training_classes": 3,
        }

    def test_filters_lvis_and_vg_keeps_null_and_emits_runtime_exact_paths(self):
        kwargs = self._kwargs()
        receipt = builder.build(**kwargs)
        output_root = kwargs["output_root"]
        runtime_path = output_root / builder.OUTPUT_RUNTIME_TSV
        with runtime_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(
            [Path(row["path"]) for row in rows],
            [
                self.mirror_paths["lvis_keep"],
                self.mirror_paths["vg_null_keep"],
                self.mirror_paths["vg_keep"],
                self.mirror_paths["vg_keep"],
            ],
        )
        self.assertEqual([int(row["class_id"]) for row in rows], [10, 10, 20, 30])
        self.assertEqual(
            [int(row["source_cache_class_id"]) for row in rows],
            [10, 10, 20, 20],
        )
        self.assertEqual(
            [row["class_assignment"] for row in rows],
            [
                "sealed_cache_identity_v1",
                "sealed_cache_identity_v1",
                "sealed_cache_identity_v1",
                "canonical_compact_alias_bridge_v1",
            ],
        )
        self.assertEqual([row["source"] for row in rows], ["lvis", "vg", "vg", "vg"])
        self.assertEqual([row["coco_id"] for row in rows], ["999", "", "303", "303"])
        self.assertTrue(all(Path(row["path"]).is_file() for row in rows))
        self.assertEqual(receipt["exclusion"]["excluded_candidate_rows"], 2)
        self.assertEqual(
            receipt["exclusion"]["actually_excluded_numeric_coco_ids"],
            [101, 202],
        )
        self.assertEqual(receipt["retained"]["null_coco_id_rows"], 1)
        self.assertEqual(receipt["runtime_bank"]["candidate_rows"], 4)
        self.assertEqual(receipt["runtime_bank"]["unique_paths"], 3)
        self.assertEqual(
            receipt["alias_bridges"],
            [
                {
                    "target_class_id": 30,
                    "source_cache_class_id": 20,
                    "compact_aliases": ["automobile"],
                    "target_canonical_names": ["automobile"],
                    "source_canonical_names": ["automobile", "car", "car n 01"],
                    "candidate_rows": 1,
                    "ordered_reused_path_stream_sha256": hashlib.sha256(
                        (str(self.mirror_paths["vg_keep"]) + "\n").encode("utf-8")
                    ).hexdigest(),
                    "new_unique_paths": 0,
                }
            ],
        )
        self.assertNotIn(
            self.mirror_paths["vg_ref8_drop"],
            {Path(row["path"]) for row in rows},
        )
        self.assertTrue(
            receipt["filter_contract"]["D0_and_D1_share_identical_runtime_bank"]
        )
        self.assertEqual(receipt["training_class_coverage"]["missing_class_ids"], [])
        self.assertTrue(all(receipt["invariants"].values()))
        self.assertEqual(builder.verify(**kwargs), receipt)

    def test_audit_output_is_original_header_and_row_subsequence(self):
        kwargs = self._kwargs()
        receipt = builder.build(**kwargs)
        actual = (kwargs["output_root"] / builder.OUTPUT_AUDIT_TSV).read_bytes()
        header = self.support_tsv.read_bytes().splitlines(keepends=True)[0]
        expected = header + b"".join(
            self.raw_input_rows[index] for index in (0, 3, 4)
        )
        self.assertEqual(actual, expected)
        self.assertIn(b"\r\n", actual)
        self.assertEqual(receipt["outputs"]["audit_raw_tsv"]["rows"], 3)

    def test_ambiguous_compact_alias_bridge_fails_closed(self):
        common = {
            "candidate_index": 0,
            "class_assignment": "sealed_cache_identity_v1",
            "source": "lvis",
            "source_dir": "lvis_patches",
            "source_class": "fixture",
            "source_image_id": 1,
            "coco_id": 1,
        }
        retained = (
            builder.Candidate(
                class_id=20,
                source_cache_class_id=20,
                path=self.mirror_paths["lvis_keep"],
                **common,
            ),
            builder.Candidate(
                class_id=40,
                source_cache_class_id=40,
                path=self.mirror_paths["vg_keep"],
                **common,
            ),
        )
        with self.assertRaisesRegex(
            builder.SupportPartitionError,
            "does not have exactly one compact canonical alias source",
        ):
            builder._build_runtime_candidates(
                retained_base=retained,
                required_class_ids=frozenset({30}),
                canonical_names_by_id={
                    20: frozenset({"automobile"}),
                    30: frozenset({"auto mobile"}),
                    40: frozenset({"auto-mobile"}),
                },
            )

    def test_missing_vg_mapping_fails_closed(self):
        self._write_vg_metadata(
            [
                {"image_id": 1, "coco_id": 202},
                {"image_id": 2, "coco_id": None},
            ]
        )
        kwargs = self._kwargs("missing-map")
        with self.assertRaisesRegex(
            builder.SupportPartitionError,
            "VG support image id lacks official mapping: 3",
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_no_overwrite_and_tamper_detection(self):
        kwargs = self._kwargs("sealed")
        builder.build(**kwargs)
        with self.assertRaisesRegex(builder.SupportPartitionError, "refusing to replace"):
            builder.build(**kwargs)
        runtime_path = kwargs["output_root"] / builder.OUTPUT_RUNTIME_TSV
        with runtime_path.open("ab") as handle:
            handle.write(b"tampered\n")
        with self.assertRaisesRegex(builder.SupportPartitionError, "output drifted"):
            builder.verify(**kwargs)


if __name__ == "__main__":
    unittest.main()
