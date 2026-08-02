import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from tools import build_stageb_data_driven_assignment_overfit64 as builder


def _jsonl_bytes(rows):
    return b"".join(builder._canonical_bytes(row) + b"\n" for row in rows)


class AssignmentOverfit64BuilderTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.heldout = self.root / "heldout"
        self.heldout.mkdir()
        self.support_root = self.root / "support_images"
        self.support_image = self.support_root / "clean/class_1/support.jpg"
        self.support_image.parent.mkdir(parents=True)
        self.support_image.write_bytes(b"fixed-external-support-image")
        self.support_tsv = self.root / "upstream_support.tsv"
        self.support_tsv.write_text(
            "class_id\tbucket\tpath\temb_rel_path\n"
            f"1\tclean\t{self.support_image}\tclean/class_1/support.npy\n",
            encoding="utf-8",
        )
        self.canonical = self.root / "canonical.json"
        self.canonical.write_text("[]\n", encoding="ascii")
        self.support_cache = self.root / "support.bank.clean.img.pkl"
        support_payload = {
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
        }
        with self.support_cache.open("wb") as handle:
            pickle.dump(support_payload, handle, protocol=4)

        self.manifest_quotas = tuple(
            (name, quota)
            for (name, _default), quota in zip(
                builder.MANIFEST_QUOTAS, (2, 1, 1), strict=True
            )
        )
        self.rows_by_manifest = {}
        next_image = 100
        for manifest_index, (name, _quota) in enumerate(self.manifest_quotas):
            rows = []
            for local_index in range(5):
                rows.append(
                    self._row(
                        manifest_index=manifest_index,
                        local_index=local_index,
                        image_id=next_image,
                        class_id=1,
                    )
                )
                next_image += 1
            rows.append(
                self._row(
                    manifest_index=manifest_index,
                    local_index=90,
                    image_id=999,
                    class_id=1,
                )
            )
            rows.append(
                self._row(
                    manifest_index=manifest_index,
                    local_index=91,
                    image_id=next_image,
                    class_id=9,
                )
            )
            next_image += 1
            invalid = self._row(
                manifest_index=manifest_index,
                local_index=92,
                image_id=next_image,
                class_id=1,
            )
            invalid["assignment_pair_valid"] = False
            invalid["assignment_pair"]["partner"] = None
            invalid["assignment_pair_invalid_reason"] = "fixture_invalid"
            rows.append(invalid)
            next_image += 1
            self.rows_by_manifest[name] = rows
            (self.inputs / name).write_bytes(_jsonl_bytes(rows))

        self.expected_input_sha = {
            name: builder._sha256_file(self.inputs / name)
            for name, _quota in self.manifest_quotas
        }
        self.expected_input_rows = {
            name: len(self.rows_by_manifest[name])
            for name, _quota in self.manifest_quotas
        }
        self.category_receipt = self.root / "category_receipt.json"
        self.category_receipt.write_text(
            json.dumps({"schema": "fixture-category-receipt/v1"}) + "\n",
            encoding="ascii",
        )
        upstream_receipt = {
            "schema": builder.UPSTREAM_RECEIPT_SCHEMA,
            "row_schema": builder.ROW_SCHEMA,
            "rows": sum(self.expected_input_rows.values()),
            "unique_identities": sum(self.expected_input_rows.values()),
            "manifest_order": [name for name, _quota in self.manifest_quotas],
            "manifests": {
                name: {
                    "rows": self.expected_input_rows[name],
                    "output": builder._file_record(self.inputs / name),
                }
                for name, _quota in self.manifest_quotas
            },
            "category_complete_receipt": builder._file_record(
                self.category_receipt
            ),
            "invariants": {"fixture_upstream_valid": True},
        }
        self.input_receipt = self.inputs / "receipt.json"
        self.input_receipt.write_text(
            json.dumps(upstream_receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

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

        self.expected_support_sha = {
            "support_tsv": builder._sha256_file(self.support_tsv),
            "support_bank_cache": builder._sha256_file(self.support_cache),
            "canonical_classes": builder._sha256_file(self.canonical),
        }

    def tearDown(self):
        self.context.cleanup()

    @staticmethod
    def _row(*, manifest_index, local_index, image_id, class_id):
        anchor = 100000 + manifest_index * 10000 + local_index * 2
        partner = anchor + 1
        source = f"fixture_{manifest_index}_train"
        expression = f"anchor phrase {manifest_index} {local_index}"
        partner_expression = f"partner phrase {manifest_index} {local_index}"
        return {
            "source": source,
            "image_id": image_id,
            "ann_id": anchor,
            "ref_id": manifest_index * 1000 + local_index,
            "sent_id": manifest_index * 1000 + local_index + 100,
            "split": "train",
            "filename": f"/query/COCO_train2014_{image_id:012d}.jpg",
            "primary_support_instance_index": 0,
            "stage_b_u2_category_complete": True,
            "stage_b_u2_category_complete_schema": (
                "pivot.stageb.u2_category_complete_ref/v1"
            ),
            "stage_b_data_driven_assignment_pair": True,
            "stage_b_data_driven_assignment_pair_schema": builder.ROW_SCHEMA,
            "assignment_pair_valid": True,
            "assignment_pair_invalid_reason": None,
            "assignment_pair": {
                "schema": builder.ROW_SCHEMA,
                "anchor": {
                    "source": source,
                    "image_id": image_id,
                    "coco_ann_id": anchor,
                    "expression": expression,
                },
                "partner": {
                    "source": source,
                    "image_id": image_id,
                    "coco_ann_id": partner,
                    "expression": partner_expression,
                    "target_iou": 0.0,
                },
            },
            "instances": [
                {
                    "bbox": [0, 0, 10, 10],
                    "class_id": class_id,
                    "coco_ann_id": anchor,
                    "category_complete_primary": True,
                    "raw_phrase": expression,
                    "positive_phrase": expression,
                    "text_is_negative": False,
                },
                {
                    "bbox": [20, 0, 10, 10],
                    "class_id": class_id,
                    "coco_ann_id": partner,
                    "category_complete_auxiliary": True,
                },
            ],
            "opaque_fixture_payload": {
                "must_survive": True,
                "values": [manifest_index, local_index],
            },
        }

    def _kwargs(self, output_name="artifact"):
        return {
            "input_root": self.inputs,
            "input_receipt": self.input_receipt,
            "heldout_root": self.heldout,
            "support_tsv": self.support_tsv,
            "support_bank_cache": self.support_cache,
            "support_image_root": self.support_root,
            "canonical_classes": self.canonical,
            "output_root": self.root / output_name,
            "manifest_quotas": self.manifest_quotas,
            "expected_input_receipt_sha256": builder._sha256_file(
                self.input_receipt
            ),
            "expected_input_sha256": self.expected_input_sha,
            "expected_input_rows": self.expected_input_rows,
            "heldout_contract": self.heldout_contract,
            "heldout_manifest_files": self.heldout_files,
            "heldout_splits": self.heldout_splits,
            "expected_heldout_union_images": 1,
            "expected_support_sha256": self.expected_support_sha,
            "expected_category_complete_receipt_sha256": (
                builder._sha256_file(self.category_receipt)
            ),
        }

    def test_builds_exact_quota_batch_and_singleton_external_support(self):
        kwargs = self._kwargs()
        receipt = builder.build(**kwargs)
        output_root = kwargs["output_root"]
        output_path = output_root / builder.OUTPUT_MANIFEST
        support_path = output_root / builder.OUTPUT_SUPPORT_TSV
        rows = [json.loads(line) for line in output_path.read_bytes().splitlines()]

        self.assertEqual(receipt["schema"], builder.RECEIPT_SCHEMA)
        self.assertEqual(receipt["row_schema"], builder.ROW_SCHEMA)
        self.assertEqual(receipt["rows"], 4)
        self.assertEqual(receipt["unique_images"], 4)
        self.assertEqual(receipt["unique_unordered_annotation_edges"], 4)
        self.assertEqual(receipt["unique_annotation_endpoints"], 8)
        self.assertEqual(
            receipt["source_counts"],
            {
                self.manifest_quotas[0][0]: 2,
                self.manifest_quotas[1][0]: 1,
                self.manifest_quotas[2][0]: 1,
            },
        )
        self.assertTrue(all(receipt["invariants"].values()))
        self.assertTrue(all(row["opaque_fixture_payload"]["must_survive"] for row in rows))
        self.assertNotIn(999, {row["image_id"] for row in rows})
        self.assertEqual({row["instances"][0]["class_id"] for row in rows}, {1})
        self.assertEqual(
            receipt["source_manifests"][self.manifest_quotas[0][0]][
                "heldout_image_excluded_rows"
            ],
            1,
        )
        self.assertEqual(
            receipt["source_manifests"][self.manifest_quotas[0][0]][
                "external_support_uncovered_rows"
            ],
            1,
        )

        support_lines = support_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(support_lines), 2)
        self.assertEqual(support_lines[0], "class_id\tbucket\tpath\temb_rel_path")
        self.assertEqual(support_lines[1].split("\t")[:2], ["1", "clean"])
        self.assertEqual(receipt["support"]["mini_support_rows"], 1)
        self.assertEqual(
            receipt["support"]["mini_support_candidates_per_class"], 1
        )
        self.assertFalse(receipt["support"]["target_crop_fallback_allowed"])
        self.assertEqual(
            receipt["support"]["mini_support_tsv"]["sha256"],
            builder._sha256_file(support_path),
        )
        witness = receipt["support"]["selected_class_witnesses"][0]
        self.assertEqual(witness["class_id"], 1)
        self.assertEqual(
            witness["image"]["sha256"], builder._sha256_file(self.support_image)
        )
        self.assertEqual(
            receipt["output"]["sha256"], builder._sha256_file(output_path)
        )
        self.assertEqual(builder.verify(**kwargs), receipt)

    def test_dry_plan_writes_nothing_and_build_replays_same_plan(self):
        kwargs = self._kwargs("dry_then_build")
        plan = builder.make_plan(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())
        receipt = builder.build(**kwargs)
        self.assertEqual(receipt, plan.receipt)
        self.assertEqual(
            (kwargs["output_root"] / builder.OUTPUT_MANIFEST).read_bytes(),
            plan.manifest_bytes,
        )
        self.assertEqual(
            (kwargs["output_root"] / builder.OUTPUT_SUPPORT_TSV).read_bytes(),
            plan.support_tsv_bytes,
        )

    def test_selection_is_stable_and_verify_rejects_output_tampering(self):
        first = self._kwargs("first")
        second = self._kwargs("second")
        receipt_first = builder.build(**first)
        receipt_second = builder.build(**second)
        self.assertEqual(
            receipt_first["ordered_member_pair_id_stream_sha256"],
            receipt_second["ordered_member_pair_id_stream_sha256"],
        )
        self.assertEqual(
            (first["output_root"] / builder.OUTPUT_MANIFEST).read_bytes(),
            (second["output_root"] / builder.OUTPUT_MANIFEST).read_bytes(),
        )
        path = first["output_root"] / builder.OUTPUT_MANIFEST
        path.write_bytes(path.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(
            builder.Overfit64BuildError, "does not replay exactly"
        ):
            builder.verify(**first)

    def test_missing_external_support_fails_instead_of_using_target_crop(self):
        self.support_image.unlink()
        kwargs = self._kwargs("no_support")
        with self.assertRaisesRegex(
            builder.Overfit64BuildError, "could not satisfy quota"
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())
        self.assertFalse(any(self.root.glob(".no_support.tmp-*")))

    def test_heldout_hash_drift_fails_before_output(self):
        split = self.heldout_splits[-1]
        path = self.heldout / self.heldout_files[split]
        path.write_bytes(path.read_bytes() + b"{}\n")
        kwargs = self._kwargs("heldout_drift")
        with self.assertRaisesRegex(
            builder.Overfit64BuildError, "heldout manifest SHA-256 drifted"
        ):
            builder.build(**kwargs)
        self.assertFalse(kwargs["output_root"].exists())


if __name__ == "__main__":
    unittest.main()
