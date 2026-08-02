import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools import build_stageb_data_driven_new_head_partition as builder


class NewHeadPartitionBuilderTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.d0_root = self.root / "d0"
        self.d1_root = self.root / "d1"
        self.source_root = self.root / "source"
        self.heldout_root = self.root / "heldout"
        for path in (
            self.d0_root,
            self.d1_root,
            self.source_root,
            self.heldout_root,
        ):
            path.mkdir()

        image_layout = {
            builder.MANIFESTS[0]: [10, 100, 101, 101, 102],
            builder.MANIFESTS[1]: [10, 100, 103, 104],
            builder.MANIFESTS[2]: [10, 100, 105, 106, 107],
        }
        self.rows = {}
        for manifest_index, name in enumerate(builder.MANIFESTS):
            d0_rows = []
            d1_rows = []
            source_rows = []
            for local_index, image_id in enumerate(image_layout[name]):
                d0, d1, source = self._row_pair(
                    manifest_index=manifest_index,
                    local_index=local_index,
                    image_id=image_id,
                )
                d0_rows.append(d0)
                d1_rows.append(d1)
                source_rows.append(source)
            self.rows[name] = {"d0": d0_rows, "d1": d1_rows}
            self._write_rows(self.d0_root / name, d0_rows, compact=True)
            self._write_rows(self.d1_root / name, d1_rows, compact=False)
            self._write_rows(self.source_root / name, source_rows, compact=True)

        self.category_receipt = self.d1_root / "receipt.json"
        self.category_receipt.write_text(
            json.dumps({"schema": "fixture.category_complete_receipt/v1"}) + "\n",
            encoding="ascii",
        )
        self.input_receipt = self.root / "receipt.json"
        self._write_pair_receipt()

        self.heldout_splits = tuple(f"ref_split_{index}" for index in range(8))
        self.heldout_files = {}
        self.heldout_contract = {}
        heldout_row = {
            "filename": "/fixture/COCO_train2014_000000000010.jpg",
            "image_id": 10,
        }
        for split in self.heldout_splits:
            filename = f"{split}.jsonl"
            path = self.heldout_root / filename
            self._write_rows(path, [{**heldout_row, "split": split}], compact=False)
            self.heldout_files[split] = filename
            self.heldout_contract[split] = {
                "rows": 1,
                "sha256": builder._sha256_file(path),
            }

    def tearDown(self):
        self.context.cleanup()

    @staticmethod
    def _row_pair(*, manifest_index, local_index, image_id):
        ann_id = 100000 + manifest_index * 1000 + local_index
        class_id = 20 + (image_id % 3)
        source_name = builder.SOURCE_LABELS[manifest_index]
        primary = {
            "bbox": [float(local_index), 1.0, 10.0, 12.0],
            "class_id": class_id,
            "raw_phrase": f"fixture phrase {manifest_index} {local_index}",
            "positive_phrase": f"fixture phrase {manifest_index} {local_index}",
            "text_is_negative": False,
        }
        identity = {
            "source": f"{source_name}_train",
            "image_id": image_id,
            "ann_id": ann_id,
            "ref_id": manifest_index * 1000 + local_index,
            "sent_id": manifest_index * 10000 + local_index,
            "split": "train",
            "filename": (
                f"/fixture/train2014/COCO_train2014_{image_id:012d}.jpg"
            ),
        }
        source = {**identity, "instances": [dict(primary)]}
        source_sha = hashlib.sha256(builder._canonical_bytes(source)).hexdigest()
        d0 = {
            **identity,
            "instances": [dict(primary)],
            "primary_support_instance_index": 0,
            "stage_b_data_driven_ordinary_primary": True,
            "stage_b_data_driven_ordinary_primary_schema": builder.D0_ROW_SCHEMA,
            "stage_b_data_driven_source_row_sha256": source_sha,
            "opaque_d0_payload": [manifest_index, local_index],
        }
        d1_primary = {
            **primary,
            "category_complete_primary": True,
            "coco_ann_id": ann_id,
        }
        d1 = {
            **identity,
            "instances": [
                d1_primary,
                {
                    "bbox": [20.0, 20.0, 5.0, 5.0],
                    "class_id": class_id,
                    "coco_ann_id": ann_id + 500000,
                    "category_complete_auxiliary": True,
                },
            ],
            "primary_support_instance_index": 0,
            "stage_b_u2_category_complete": True,
            "stage_b_u2_category_complete_schema": builder.D1_ROW_SCHEMA,
            "category_complete_coco_split": "train2014",
            "opaque_d1_payload": {"manifest": manifest_index, "row": local_index},
        }
        return d0, d1, source

    @staticmethod
    def _write_rows(path, rows, *, compact):
        if compact:
            payload = b"".join(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                + b"\n"
                for row in rows
            )
        else:
            payload = b"".join(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=False,
                    separators=(", ", ": "),
                ).encode("ascii")
                + b"\n"
                for row in rows
            )
        path.write_bytes(payload)

    def _write_pair_receipt(self):
        identity_digest = hashlib.sha256()
        primary_digest = hashlib.sha256()
        source_row_digest = hashlib.sha256()
        complete_instances = 0
        manifests = {}
        total_rows = 0
        for name in builder.MANIFESTS:
            d0_rows = self.rows[name]["d0"]
            d1_rows = self.rows[name]["d1"]
            for d0 in d0_rows:
                identity = tuple(d0[key] for key in builder.IDENTITY_KEYS)
                identity_digest.update(builder._canonical_bytes(identity) + b"\n")
                primary_digest.update(
                    builder._canonical_bytes(d0["instances"][0]) + b"\n"
                )
                source_row_digest.update(
                    d0["stage_b_data_driven_source_row_sha256"].encode("ascii")
                    + b"\n"
                )
            manifest_complete_instances = sum(
                len(row["instances"]) for row in d1_rows
            )
            complete_instances += manifest_complete_instances
            total_rows += len(d0_rows)
            manifests[name] = {
                "rows": len(d0_rows),
                "complete_instances": manifest_complete_instances,
                "source": builder._file_record(self.source_root / name),
                "ordinary_primary": builder._file_record(self.d0_root / name),
                "category_complete": builder._file_record(self.d1_root / name),
            }
        receipt = {
            "schema": builder.UPSTREAM_RECEIPT_SCHEMA,
            "ordinary_row_schema": builder.D0_ROW_SCHEMA,
            "rows": total_rows,
            "unique_identities": total_rows,
            "category_complete_instances": complete_instances,
            "ordered_identity_stream_sha256": identity_digest.hexdigest(),
            "source_primary_stream_sha256": primary_digest.hexdigest(),
            "source_row_sha256_stream_sha256": source_row_digest.hexdigest(),
            "category_complete_receipt": builder._file_record(
                self.category_receipt
            ),
            "manifests": manifests,
            "invariants": {
                "fixture_pair_identity": True,
                "fixture_primary_match": True,
            },
        }
        self.input_receipt.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _refresh_pair_file_bindings(self):
        receipt = json.loads(self.input_receipt.read_text(encoding="ascii"))
        for name in builder.MANIFESTS:
            receipt["manifests"][name]["ordinary_primary"] = builder._file_record(
                self.d0_root / name
            )
            receipt["manifests"][name]["category_complete"] = builder._file_record(
                self.d1_root / name
            )
        self.input_receipt.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _kwargs(self, output_name="partition"):
        return {
            "input_receipt": self.input_receipt,
            "d0_root": self.d0_root,
            "d1_root": self.d1_root,
            "heldout_root": self.heldout_root,
            "output_root": self.root / output_name,
            "seed": builder.SEED,
            "dev_images": 4,
            "screen_images": 2,
            "expected_input_receipt_sha256": builder._sha256_file(
                self.input_receipt
            ),
            "expected_category_receipt_sha256": builder._sha256_file(
                self.category_receipt
            ),
            "expected_rows": {
                name: len(self.rows[name]["d0"]) for name in builder.MANIFESTS
            },
            "expected_d0_sha256": {
                name: builder._sha256_file(self.d0_root / name)
                for name in builder.MANIFESTS
            },
            "expected_d1_sha256": {
                name: builder._sha256_file(self.d1_root / name)
                for name in builder.MANIFESTS
            },
            "heldout_contract": self.heldout_contract,
            "heldout_manifest_files": self.heldout_files,
            "heldout_splits": self.heldout_splits,
            "expected_heldout_union_images": 1,
        }

    @staticmethod
    def _key(row):
        return ("train2014", row["image_id"])

    @staticmethod
    def _member_keys(receipt, field):
        return {
            (member["coco_split"], member["image_id"])
            for member in receipt[field]
        }

    def _assert_output_identity_receipt(self, receipt, output_root):
        for partition in builder.PARTITIONS:
            for name in builder.MANIFESTS:
                variant_metadata = []
                for variant in builder.VARIANTS:
                    path = output_root / variant / partition / name
                    identities = []
                    image_keys = set()
                    digest = hashlib.sha256()
                    for raw in path.read_bytes().splitlines():
                        row = json.loads(raw)
                        identity = tuple(row[key] for key in builder.IDENTITY_KEYS)
                        identity_bytes = builder._canonical_bytes(identity)
                        identities.append(identity_bytes)
                        digest.update(identity_bytes + b"\n")
                        image_keys.add(self._key(row))
                    record = receipt["outputs"][variant][partition][name]
                    self.assertEqual(record["rows"], len(identities))
                    self.assertEqual(
                        record["unique_identities"], len(set(identities))
                    )
                    self.assertEqual(record["unique_image_keys"], len(image_keys))
                    self.assertEqual(
                        record["ordered_identity_stream_sha256"],
                        digest.hexdigest(),
                    )
                    variant_metadata.append(
                        (
                            record["rows"],
                            record["unique_identities"],
                            record["unique_image_keys"],
                            record["ordered_identity_stream_sha256"],
                        )
                    )
                self.assertEqual(variant_metadata[0], variant_metadata[1])

    def test_global_partition_ref8_quarantine_nested_screen_and_raw_subsequences(self):
        kwargs = self._kwargs()
        receipt = builder.build(**kwargs)
        output_root = kwargs["output_root"]
        dev_full = self._member_keys(receipt, "dev_full_members")
        dev_screen = self._member_keys(receipt, "dev_screen_members")
        quarantine = {("train2014", 10)}

        self.assertEqual(len(dev_full), 4)
        self.assertEqual(len(dev_screen), 2)
        self.assertTrue(dev_screen.issubset(dev_full))
        self.assertFalse(dev_full.intersection(quarantine))
        self.assertEqual(
            receipt["partition_summary"]["quarantine"]["unique_image_keys"], 1
        )
        self.assertTrue(all(receipt["invariants"].values()))

        train = {
            ("train2014", image_id)
            for image_id in range(100, 108)
        } - dev_full
        partition_keys = {
            "train": train,
            "dev_full": dev_full,
            "dev_screen": dev_screen,
            "quarantine": quarantine,
        }
        for variant, source_root in (
            (builder.VARIANTS[0], self.d0_root),
            (builder.VARIANTS[1], self.d1_root),
        ):
            for partition, keys in partition_keys.items():
                for name in builder.MANIFESTS:
                    expected = b"".join(
                        raw
                        for raw in (source_root / name).read_bytes().splitlines(
                            keepends=True
                        )
                        if self._key(json.loads(raw)) in keys
                    )
                    observed = (
                        output_root / variant / partition / name
                    ).read_bytes()
                    self.assertEqual(observed, expected)

        image_100_main_partitions = []
        for partition in ("train", "dev_full", "quarantine"):
            count = 0
            for name in builder.MANIFESTS:
                path = output_root / builder.VARIANTS[0] / partition / name
                count += sum(
                    json.loads(raw)["image_id"] == 100
                    for raw in path.read_bytes().splitlines()
                )
            if count:
                image_100_main_partitions.append((partition, count))
        self.assertEqual(len(image_100_main_partitions), 1)
        self.assertEqual(image_100_main_partitions[0][1], 3)
        self._assert_output_identity_receipt(receipt, output_root)
        self.assertTrue(
            receipt["invariants"][
                "D0_and_D1_partition_identity_streams_and_counts_match"
            ]
        )
        canonical_payload_sha256 = receipt["canonical_payload_sha256"]
        canonical_payload = dict(receipt)
        del canonical_payload["canonical_payload_sha256"]
        self.assertEqual(
            canonical_payload_sha256,
            hashlib.sha256(builder._canonical_bytes(canonical_payload)).hexdigest(),
        )
        self.assertEqual(builder.verify(**kwargs), receipt)

    def test_selection_is_deterministic_and_build_refuses_overwrite(self):
        first = self._kwargs("first")
        second = self._kwargs("second")
        first_receipt = builder.build(**first)
        second_receipt = builder.build(**second)
        self.assertEqual(
            [member["image_key"] for member in first_receipt["dev_full_members"]],
            [member["image_key"] for member in second_receipt["dev_full_members"]],
        )
        self.assertEqual(
            [member["image_key"] for member in first_receipt["dev_screen_members"]],
            [member["image_key"] for member in second_receipt["dev_screen_members"]],
        )
        for variant in builder.VARIANTS:
            for partition in builder.PARTITIONS:
                for name in builder.MANIFESTS:
                    self.assertEqual(
                        (
                            first["output_root"] / variant / partition / name
                        ).read_bytes(),
                        (
                            second["output_root"] / variant / partition / name
                        ).read_bytes(),
                    )
        with self.assertRaisesRegex(
            builder.NewHeadPartitionError, "refusing to replace existing output root"
        ):
            builder.build(**first)

    def test_verify_rejects_tampering_and_official_contract_drift(self):
        kwargs = self._kwargs("tamper")
        builder.build(**kwargs)
        path = (
            kwargs["output_root"]
            / builder.VARIANTS[0]
            / "train"
            / builder.MANIFESTS[0]
        )
        path.write_bytes(path.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(
            builder.NewHeadPartitionError, "does not replay exactly"
        ):
            builder.verify(**kwargs)

        split = self.heldout_splits[-1]
        heldout_path = self.heldout_root / self.heldout_files[split]
        heldout_path.write_bytes(heldout_path.read_bytes() + b"{}\n")
        drift = self._kwargs("official_drift")
        with self.assertRaisesRegex(
            builder.NewHeadPartitionError,
            "official Ref8 manifest SHA-256 drifted",
        ):
            builder.build(**drift)
        self.assertFalse(drift["output_root"].exists())

    def test_paired_identity_and_primary_drift_fail_closed(self):
        name = builder.MANIFESTS[0]
        d1_path = self.d1_root / name
        rows = [json.loads(raw) for raw in d1_path.read_bytes().splitlines()]
        rows[0]["sent_id"] += 1
        self._write_rows(d1_path, rows, compact=False)
        self._refresh_pair_file_bindings()
        identity_kwargs = self._kwargs("identity_drift")
        with self.assertRaisesRegex(
            builder.NewHeadPartitionError, "paired identity drifted"
        ):
            builder.build(**identity_kwargs)
        self.assertFalse(identity_kwargs["output_root"].exists())

        rows[0]["sent_id"] -= 1
        rows[0]["instances"][0]["bbox"][0] += 0.25
        self._write_rows(d1_path, rows, compact=False)
        self._refresh_pair_file_bindings()
        primary_kwargs = self._kwargs("primary_drift")
        with self.assertRaisesRegex(
            builder.NewHeadPartitionError, "paired primary instance drifted"
        ):
            builder.build(**primary_kwargs)
        self.assertFalse(primary_kwargs["output_root"].exists())


if __name__ == "__main__":
    unittest.main()
