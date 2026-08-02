import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_stageb_native_residual_fresh_o64 as builder


def _jsonl_bytes(rows):
    return b"".join(builder._canonical_bytes(row) + b"\n" for row in rows)


class FreshNativeResidualO64BuilderTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.inputs = self.root / "inputs"
        self.heldout = self.root / "heldout"
        self.support_root = self.root / "support"
        self.inputs.mkdir()
        self.heldout.mkdir()
        self.support_root.mkdir()

        self.support_image = self.support_root / "clean/class_1/support.jpg"
        self.support_image.parent.mkdir(parents=True)
        self.support_image.write_bytes(b"external-clean-support")
        self.support_tsv = self.root / "support.tsv"
        self.support_tsv.write_text(
            "class_id\tbucket\tpath\temb_rel_path\n"
            f"1\tclean\t{self.support_image}\tclean/class_1/support.npy\n",
            encoding="utf-8",
        )
        self.canonical = self.root / "canonical.json"
        self.canonical.write_text("[]\n", encoding="ascii")
        self.support_cache = self.root / "support.bank.clean.img.pkl"
        with self.support_cache.open("wb") as handle:
            pickle.dump(
                {
                    "meta": {
                        "version": 3,
                        "tsv_path": str(self.support_tsv),
                        "canonical_classes_json": str(self.canonical),
                        "support_patch_image_root": str(self.support_root),
                        "bucket": "clean",
                        "use_embedding": False,
                        "max_per_class": 200,
                    },
                    "bank": {1: [str(self.support_image)]},
                },
                handle,
                protocol=4,
            )

        self.manifest_quotas = tuple(
            (name, quota)
            for (name, _production_quota), quota in zip(
                builder.MANIFEST_QUOTAS, (2, 1, 1), strict=True
            )
        )
        self.old_images = []
        self.dev_images = []
        self.rows_by_manifest = {}
        for manifest_index, (name, _quota) in enumerate(self.manifest_quotas):
            base_image = 1000 + manifest_index * 100
            old_image = base_image
            dev_image = base_image + 1
            self.old_images.append(old_image)
            self.dev_images.append(dev_image)
            images_and_classes = [
                (old_image, 1),
                (dev_image, 1),
                (999, 1),
                (base_image + 2, 9),
                (base_image + 3, 1),
                (base_image + 4, 1),
                (base_image + 5, 1),
                (base_image + 6, 1),
            ]
            rows = [
                self._row(
                    manifest_index=manifest_index,
                    local_index=local_index,
                    image_id=image_id,
                    class_id=class_id,
                )
                for local_index, (image_id, class_id) in enumerate(images_and_classes)
            ]
            self.rows_by_manifest[name] = rows
            (self.inputs / name).write_bytes(_jsonl_bytes(rows))

        self.category_receipt = self.root / "category_receipt.json"
        self.category_receipt.write_text(
            json.dumps({"schema": "fixture.category/v1"}) + "\n",
            encoding="ascii",
        )
        self.input_receipt = self.inputs / "receipt.json"
        self._write_input_receipt()

        self.heldout_splits = tuple(f"split_{index}" for index in range(8))
        self.heldout_files = {}
        self.heldout_contract = {}
        for split in self.heldout_splits:
            filename = f"{split}.jsonl"
            path = self.heldout / filename
            path.write_bytes(_jsonl_bytes([{"image_id": 999, "split": split}]))
            self.heldout_files[split] = filename
            self.heldout_contract[split] = {
                "rows": 1,
                "sha256": builder._sha256_file(path),
            }

        self.old_receipt = self.root / "old_o64_receipt.json"
        self._write_old_receipt()
        self.new_head_receipt = self.root / "new_head_receipt.json"
        self._write_new_head_receipt()

    def tearDown(self):
        self.context.cleanup()

    @staticmethod
    def _row(*, manifest_index, local_index, image_id, class_id):
        anchor = 100000 + manifest_index * 10000 + local_index * 2
        partner = anchor + 1
        source = f"fixture_{manifest_index}_train"
        anchor_box = [float(local_index), 1.0, 10.0, 12.0]
        partner_box = [20.0 + local_index, 2.0, 8.0, 9.0]
        anchor_phrase = f"left fixture {manifest_index} {local_index}"
        partner_phrase = f"right fixture {manifest_index} {local_index}"
        return {
            "source": source,
            "image_id": image_id,
            "ann_id": anchor,
            "ref_id": manifest_index * 1000 + local_index,
            "sent_id": manifest_index * 10000 + local_index,
            "split": "train",
            "filename": f"/fixture/COCO_train2014_{image_id:012d}.jpg",
            "primary_support_instance_index": 0,
            "stage_b_u2_category_complete": True,
            "stage_b_u2_category_complete_schema": (
                "pivot.stageb.u2_category_complete_ref/v1"
            ),
            "stage_b_data_driven_assignment_pair": True,
            "stage_b_data_driven_assignment_pair_schema": (
                builder.ASSIGNMENT_ROW_SCHEMA
            ),
            "assignment_pair_valid": True,
            "assignment_pair_invalid_reason": None,
            "assignment_pair": {
                "schema": builder.ASSIGNMENT_ROW_SCHEMA,
                "anchor": {
                    "source": source,
                    "image_id": image_id,
                    "coco_ann_id": anchor,
                    "bbox": anchor_box,
                    "expression": anchor_phrase,
                },
                "partner": {
                    "source": source,
                    "image_id": image_id,
                    "coco_ann_id": partner,
                    "bbox": partner_box,
                    "expression": partner_phrase,
                    "target_iou": 0.0,
                },
            },
            "instances": [
                {
                    "bbox": anchor_box,
                    "class_id": class_id,
                    "coco_ann_id": anchor,
                    "category_complete_primary": True,
                    "raw_phrase": anchor_phrase,
                    "positive_phrase": anchor_phrase,
                    "text_is_negative": False,
                },
                {
                    "bbox": partner_box,
                    "class_id": class_id,
                    "coco_ann_id": partner,
                    "category_complete_auxiliary": True,
                },
            ],
        }

    def _write_input_receipt(self):
        expected_rows = {
            name: len(rows) for name, rows in self.rows_by_manifest.items()
        }
        receipt = {
            "schema": builder.selection.UPSTREAM_RECEIPT_SCHEMA,
            "row_schema": builder.ASSIGNMENT_ROW_SCHEMA,
            "rows": sum(expected_rows.values()),
            "unique_identities": sum(expected_rows.values()),
            "manifest_order": [name for name, _quota in self.manifest_quotas],
            "manifests": {
                name: {
                    "rows": expected_rows[name],
                    "output": builder._file_record(self.inputs / name),
                }
                for name, _quota in self.manifest_quotas
            },
            "category_complete_receipt": builder._file_record(
                self.category_receipt
            ),
            "invariants": {"fixture_assignment_receipt_is_valid": True},
        }
        self.input_receipt.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _write_old_receipt(self):
        members = []
        for index, image_id in enumerate(self.old_images):
            members.append(
                {
                    "output_index": index,
                    "image_id": image_id,
                    "pair_id": hashlib.sha256(f"old-pair-{image_id}".encode()).hexdigest(),
                    "source_row_sha256": hashlib.sha256(
                        f"old-row-{image_id}".encode()
                    ).hexdigest(),
                }
            )
        receipt = {
            "schema": builder.OLD_O64_RECEIPT_SCHEMA,
            "row_schema": builder.ASSIGNMENT_ROW_SCHEMA,
            "rows": len(members),
            "valid_rows": len(members),
            "invalid_rows": 0,
            "unique_images": len(members),
            "unique_unordered_annotation_edges": len(members),
            "unique_annotation_endpoints": 2 * len(members),
            "members": members,
            "selection_contract": {
                "model_score_free": True,
                "namespace": builder.selection.SELECTION_NAMESPACE,
                "policy": builder.selection.SELECTION_POLICY,
                "forbidden_inputs": sorted(builder.LEGACY_FORBIDDEN_INPUTS),
            },
            "ordered_member_stream_encoding": builder.STREAM_ENCODING,
            "ordered_image_id_stream_sha256": builder._record_stream_sha256(
                [str(value) for value in self.old_images]
            ),
            "invariants": {"fixture_old_o64_is_valid": True},
        }
        receipt["canonical_payload_sha256"] = builder._canonical_payload_sha256(
            receipt
        )
        self.old_receipt.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _write_new_head_receipt(self):
        def member(image_id):
            return {
                "image_id": image_id,
                "image_key": f"train2014:{image_id:012d}",
                "selection_priority_sha256": hashlib.sha256(
                    f"dev-{image_id}".encode()
                ).hexdigest(),
            }

        full = [member(value) for value in self.dev_images]
        screen = full[:1]
        full_keys = [item["image_key"] for item in full]
        screen_keys = [item["image_key"] for item in screen]
        receipt = {
            "schema": builder.NEW_HEAD_RECEIPT_SCHEMA,
            "selection_contract": {
                "model_score_free": True,
                "dev_full_target_images": len(full),
                "dev_screen_target_images": len(screen),
                "dev_screen_is_nested_in_dev_full": True,
                "forbidden_inputs": sorted(builder.LEGACY_FORBIDDEN_INPUTS),
            },
            "partition_summary": {
                "dev_full": {
                    "unique_image_keys": len(full),
                    "ordered_image_key_stream_sha256": (
                        builder._record_stream_sha256(full_keys)
                    ),
                },
                "dev_screen": {
                    "unique_image_keys": len(screen),
                    "ordered_image_key_stream_sha256": (
                        builder._record_stream_sha256(screen_keys)
                    ),
                },
            },
            "dev_full_members": full,
            "dev_screen_members": screen,
            "invariants": {"fixture_new_head_partition_is_valid": True},
        }
        receipt["canonical_payload_sha256"] = builder._canonical_payload_sha256(
            receipt
        )
        self.new_head_receipt.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _kwargs(self, output_name="artifact"):
        expected_input_sha = {
            name: builder._sha256_file(self.inputs / name)
            for name, _quota in self.manifest_quotas
        }
        expected_input_rows = {
            name: len(self.rows_by_manifest[name])
            for name, _quota in self.manifest_quotas
        }
        return {
            "input_root": self.inputs,
            "input_receipt": self.input_receipt,
            "heldout_root": self.heldout,
            "old_o64_receipt": self.old_receipt,
            "new_head_receipt": self.new_head_receipt,
            "support_tsv": self.support_tsv,
            "support_bank_cache": self.support_cache,
            "support_image_root": self.support_root,
            "canonical_classes": self.canonical,
            "output_root": self.root / output_name,
            "manifest_quotas": self.manifest_quotas,
            "expected_input_receipt_sha256": builder._sha256_file(
                self.input_receipt
            ),
            "expected_input_sha256": expected_input_sha,
            "expected_input_rows": expected_input_rows,
            "expected_category_complete_receipt_sha256": builder._sha256_file(
                self.category_receipt
            ),
            "heldout_contract": self.heldout_contract,
            "heldout_manifest_files": self.heldout_files,
            "heldout_splits": self.heldout_splits,
            "expected_heldout_union_images": 1,
            "expected_support_sha256": {
                "support_tsv": builder._sha256_file(self.support_tsv),
                "support_bank_cache": builder._sha256_file(self.support_cache),
                "canonical_classes": builder._sha256_file(self.canonical),
            },
            "expected_selection_library_sha256": builder._sha256_file(
                builder.SELECTION_LIBRARY
            ),
            "expected_old_o64_receipt_sha256": builder._sha256_file(
                self.old_receipt
            ),
            "expected_new_head_receipt_sha256": builder._sha256_file(
                self.new_head_receipt
            ),
            "expected_old_o64_images": len(self.old_images),
            "expected_new_head_dev_full_images": len(self.dev_images),
            "expected_new_head_dev_screen_images": 1,
            "expected_selected_streams": None,
        }

    def test_builds_directed_rows_with_both_exposure_blacklists(self):
        kwargs = self._kwargs()
        plan = builder.make_plan(**kwargs)
        receipt = builder.build(**kwargs)
        manifest_path = kwargs["output_root"] / builder.OUTPUT_MANIFEST
        rows = [json.loads(line) for line in manifest_path.read_bytes().splitlines()]

        self.assertEqual(receipt, plan.receipt)
        self.assertEqual(receipt["pairs"], 4)
        self.assertEqual(receipt["rows"], 8)
        self.assertEqual(receipt["direction_counts"], {"anchor": 4, "partner": 4})
        self.assertEqual(receipt["unique_images"], 4)
        self.assertEqual(receipt["unique_target_annotation_ids"], 8)
        self.assertTrue(all(receipt["invariants"].values()))
        self.assertEqual(
            [(row["pair_index"], row["direction"]) for row in rows],
            [
                (0, "anchor"),
                (0, "partner"),
                (1, "anchor"),
                (1, "partner"),
                (2, "anchor"),
                (2, "partner"),
                (3, "anchor"),
                (3, "partner"),
            ],
        )
        selected_images = {row["image_id"] for row in rows}
        self.assertFalse(selected_images.intersection(self.old_images))
        self.assertFalse(selected_images.intersection(self.dev_images))
        self.assertNotIn(999, selected_images)
        for row in rows:
            self.assertEqual(row["row_schema"], builder.OUTPUT_ROW_SCHEMA)
            self.assertEqual(len(row["grounding"]["regions"]), 1)
            self.assertEqual(len(row["grounding"]["regions"][0]["bbox"]), 4)
            self.assertIn(row["source_assignment_manifest"], dict(self.manifest_quotas))
            self.assertEqual(len(row["source_member_pair_id"]), 64)
        first_stats = receipt["source_statistics"][self.manifest_quotas[0][0]]
        self.assertEqual(first_stats["old_o64_image_excluded_rows"], 1)
        self.assertEqual(first_stats["evaluated_dev_full_image_excluded_rows"], 1)
        self.assertEqual(first_stats["official_ref8_image_excluded_rows"], 1)
        self.assertEqual(first_stats["external_support_uncovered_rows"], 1)
        support = receipt["inputs"]["external_support"]
        self.assertTrue(support["selection_only_not_consumed_by_odvg_loader"])
        self.assertEqual(support["selected_class_count"], 1)
        self.assertEqual(
            support["selected_class_witnesses"][0]["image"]["sha256"],
            builder._sha256_file(self.support_image),
        )
        self.assertEqual(builder.verify(**kwargs), receipt)

    def test_create_new_and_byte_replay_reject_manifest_tampering(self):
        kwargs = self._kwargs("create_new")
        builder.build(**kwargs)
        before = {
            path.name: path.read_bytes()
            for path in kwargs["output_root"].iterdir()
        }
        with self.assertRaisesRegex(builder.FreshO64BuildError, "refusing to replace"):
            builder.build(**kwargs)
        after = {
            path.name: path.read_bytes()
            for path in kwargs["output_root"].iterdir()
        }
        self.assertEqual(after, before)
        manifest = kwargs["output_root"] / builder.OUTPUT_MANIFEST
        manifest.write_bytes(manifest.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(builder.FreshO64BuildError, "does not replay"):
            builder.verify(**kwargs)

    def test_rejects_recursive_teacher_model_or_checkpoint_fields(self):
        name = self.manifest_quotas[0][0]
        self.rows_by_manifest[name][-1]["nested_audit"] = {
            "checkpoint_outputs": [0.1]
        }
        (self.inputs / name).write_bytes(_jsonl_bytes(self.rows_by_manifest[name]))
        self._write_input_receipt()
        kwargs = self._kwargs("forbidden")
        with self.assertRaisesRegex(
            builder.FreshO64BuildError, "forbidden model-derived field"
        ):
            builder.make_plan(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_rejects_exposure_receipt_hash_drift_before_writing(self):
        kwargs = self._kwargs("receipt_drift")
        self.old_receipt.write_bytes(self.old_receipt.read_bytes() + b" ")
        with self.assertRaisesRegex(
            builder.FreshO64BuildError, "old O64 receipt SHA-256 mismatch"
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_missing_external_support_never_falls_back_to_target_crop(self):
        self.support_image.unlink()
        kwargs = self._kwargs("no_support")
        with self.assertRaisesRegex(builder.FreshO64BuildError, "could not satisfy quota"):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())

    def test_failed_atomic_publish_removes_temporary_directory(self):
        kwargs = self._kwargs("atomic")
        with patch.object(builder.os, "rename", side_effect=OSError("fixture failure")):
            with self.assertRaisesRegex(OSError, "fixture failure"):
                builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())
        self.assertEqual(list(self.root.glob(".atomic.tmp-*")), [])

    def test_dataset_config_targets_content_bound_default_artifact(self):
        path = builder.REPO_ROOT / "config/datasets_stageb_native_residual_fresh_o64.json"
        config = json.loads(path.read_text(encoding="ascii"))
        self.assertEqual(config["val"], [])
        self.assertEqual(len(config["train"]), 1)
        entry = config["train"][0]
        self.assertEqual(entry["dataset_mode"], "odvg")
        self.assertEqual(entry["root"], "/")
        self.assertEqual(entry["anno"], str(builder.OUTPUT_ROOT / builder.OUTPUT_MANIFEST))
        self.assertEqual(entry["mix_weight"], 1.0)
        binding = config["artifact_binding"]
        self.assertEqual(binding["manifest"]["path"], entry["anno"])
        self.assertEqual(
            binding["receipt"]["path"], str(builder.OUTPUT_ROOT / "receipt.json")
        )
        self.assertEqual(binding["manifest"]["rows"], 128)
        self.assertEqual(binding["receipt"]["schema"], builder.RECEIPT_SCHEMA)
        self.assertEqual(
            binding["manifest"]["sha256"],
            builder._sha256_file(Path(binding["manifest"]["path"])),
        )
        self.assertEqual(
            binding["receipt"]["sha256"],
            builder._sha256_file(Path(binding["receipt"]["path"])),
        )

    def test_production_selected_stream_constants_are_complete(self):
        self.assertEqual(
            set(builder.EXPECTED_SELECTED_STREAMS),
            {
                "ordered_pair_id_stream_sha256",
                "ordered_image_id_stream_sha256",
                "sorted_image_id_json_sha256",
                "ordered_unordered_edge_stream_sha256",
                "ordered_endpoint_stream_sha256",
            },
        )
        self.assertTrue(
            all(len(value) == 64 for value in builder.EXPECTED_SELECTED_STREAMS.values())
        )


if __name__ == "__main__":
    unittest.main()
