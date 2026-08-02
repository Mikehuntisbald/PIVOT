import json
import tempfile
import unittest
from pathlib import Path

from tools.partition_stageb_semantic_tn import (
    AUDIT_NAME,
    OUTPUT_NAMES,
    SemanticTNPartitionError,
    canonical_sha256,
    create_partition,
    sha256_file,
    verify_partition,
)


def _edit(category, replace_from, replace_to, span):
    return {
        "category": category,
        "replace_from": replace_from,
        "replace_to": replace_to,
        "replace_span": list(span),
    }


def _semantic_row(sample_id, image_id, *, dataset="refcocoplus", edits=None):
    edits = list(edits or [_edit("color", "red", "blue", [0, 1])])
    return {
        "sample_id": sample_id,
        "image_id": image_id,
        "dataset": dataset,
        "sent": "red object",
        "try_tn": "blue object",
        "replace_category": [edit["category"] for edit in edits],
        "replace_from": [edit["replace_from"] for edit in edits],
        "replace_to": [edit["replace_to"] for edit in edits],
        "replace_span": [edit["replace_span"] for edit in edits],
        "tn_edits": edits,
    }


def _write_source(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_by_sample = {}
    payload = []
    for index, row in enumerate(rows):
        if index % 2:
            rendered = json.dumps(row, ensure_ascii=True, sort_keys=True)
            ending = "\r\n"
        else:
            rendered = json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=False,
                separators=(",", ":"),
            )
            ending = "\n"
        raw = (rendered + ending).encode("utf-8")
        raw_by_sample[row["sample_id"]] = raw
        payload.append(raw)
    path.write_bytes(b"".join(payload))
    return raw_by_sample


def _write_manifest(path, image_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"image_id": value}) + "\n" for value in image_ids),
        encoding="utf-8",
    )


def _read_output(path):
    return [
        json.loads(raw.decode("utf-8"))
        for raw in path.read_bytes().splitlines(keepends=True)
        if raw.strip()
    ]


class SemanticTNPartitionTests(unittest.TestCase):
    def _full_fixture(self, root):
        source = root / "semantic_verified_pairs.jsonl"
        strict2031 = root / "strict2031.jsonl"
        strict1607 = root / "strict1607.jsonl"
        rows = [
            _semantic_row(
                f"single-{image_id}",
                image_id,
                dataset="refcocoplus" if image_id % 2 else "refcocog",
            )
            for image_id in range(100, 124)
        ]
        rows.append(_semantic_row("same-image-second-row", 110))
        rows.append(
            _semantic_row(
                "valid-multi-edit",
                124,
                edits=[
                    _edit("color", "red", "blue", [0, 1]),
                    _edit("size", "large", "small", [2, 3]),
                ],
            )
        )
        invalid = _semantic_row("invalid-single-edit", 125)
        invalid["replace_to"] = ["green"]
        rows.append(invalid)
        raw_by_sample = _write_source(source, rows)
        _write_manifest(strict2031, [102, 104, 900])
        _write_manifest(strict1607, [103, 104, 901])
        return source, strict2031, strict1607, rows, raw_by_sample

    def test_partition_filters_union_groups_images_and_preserves_raw_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (
                source,
                strict2031,
                strict1607,
                source_rows,
                raw_by_sample,
            ) = self._full_fixture(root)
            output = root / "partition"
            audit = create_partition(
                source_path=source,
                strict2031_path=strict2031,
                strict1607_path=strict1607,
                output_dir=output,
                seed="paper-seed",
                calibration_ratio="1/2",
                expected_input_sha256=sha256_file(source),
                expected_strict2031_sha256=sha256_file(strict2031),
                expected_strict1607_sha256=sha256_file(strict1607),
            )

            train = _read_output(output / OUTPUT_NAMES["train"])
            calibration = _read_output(output / OUTPUT_NAMES["calibration"])
            all_rows = train + calibration
            all_ids = {row["sample_id"] for row in all_rows}
            expected_ids = {
                row["sample_id"]
                for row in source_rows
                if row["image_id"] not in {102, 103, 104}
            }
            self.assertEqual(all_ids, expected_ids)
            self.assertTrue(train)
            self.assertTrue(calibration)
            self.assertFalse(
                {row["image_id"] for row in train}
                & {row["image_id"] for row in calibration}
            )
            same_image_splits = {
                "train" if row in train else "calibration"
                for row in all_rows
                if row["image_id"] == 110
            }
            self.assertEqual(len(same_image_splits), 1)

            for name in OUTPUT_NAMES:
                path = output / OUTPUT_NAMES[name]
                for output_raw in path.read_bytes().splitlines(keepends=True):
                    sample_id = json.loads(output_raw.decode("utf-8"))["sample_id"]
                    self.assertEqual(output_raw, raw_by_sample[sample_id])

            single_rows = _read_output(
                output / OUTPUT_NAMES["single_edit_train"]
            ) + _read_output(output / OUTPUT_NAMES["single_edit_calibration"])
            single_ids = {row["sample_id"] for row in single_rows}
            self.assertNotIn("valid-multi-edit", single_ids)
            self.assertNotIn("invalid-single-edit", single_ids)
            self.assertIn("same-image-second-row", single_ids)
            self.assertTrue(single_ids.issubset(all_ids))

            self.assertEqual(audit["filtered_overlap"]["filtered_semantic_rows"]["union"], 3)
            self.assertEqual(
                audit["filtered_overlap"]["semantic_source_image_overlap"],
                {
                    "strict2031_only": 1,
                    "strict1607_only": 1,
                    "both": 1,
                    "union": 3,
                },
            )
            provenance = audit["distributions"]["eligible"]["edit_provenance"]
            self.assertEqual(provenance["multi_edit_rows"], 1)
            self.assertEqual(provenance["valid_tn_edits_rows"], len(expected_ids))
            self.assertEqual(
                provenance["single_edit_token_eligible_rows"], len(expected_ids) - 2
            )
            self.assertEqual(provenance["consistent_replace_to_rows"], len(expected_ids) - 1)
            self.assertIn(
                "refcocoplus",
                audit["distributions"]["eligible"]["dataset_rows"],
            )
            self.assertIn(
                "color",
                audit["distributions"]["eligible"]["taxonomy"]["row_membership"],
            )
            self.assertEqual(
                audit["partition_contract_sha256"],
                canonical_sha256(audit["partition_contract"]),
            )
            self.assertEqual(
                verify_partition(output / AUDIT_NAME)["verified"], True
            )

    def test_same_inputs_and_seed_reproduce_membership_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, strict2031, strict1607, _rows, _raw = self._full_fixture(root)
            first = create_partition(
                source_path=source,
                strict2031_path=strict2031,
                strict1607_path=strict1607,
                output_dir=root / "first",
                seed=77,
                calibration_ratio="0.1",
            )
            second = create_partition(
                source_path=source,
                strict2031_path=strict2031,
                strict1607_path=strict1607,
                output_dir=root / "second",
                seed=77,
                calibration_ratio="1/10",
            )
            self.assertEqual(
                first["partition_contract_sha256"],
                second["partition_contract_sha256"],
            )
            for name in OUTPUT_NAMES:
                self.assertEqual(
                    first["outputs"][name]["sha256"],
                    second["outputs"][name]["sha256"],
                )

    def test_duplicate_sample_id_and_missing_critical_fields_fail_before_outputs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            strict2031 = root / "strict2031.jsonl"
            strict1607 = root / "strict1607.jsonl"
            _write_manifest(strict2031, [900])
            _write_manifest(strict1607, [901])

            duplicate_source = root / "duplicate.jsonl"
            _write_source(
                duplicate_source,
                [_semantic_row("duplicate", 1), _semantic_row("duplicate", 2)],
            )
            duplicate_output = root / "duplicate-output"
            with self.assertRaisesRegex(
                SemanticTNPartitionError, "duplicate sample_id"
            ):
                create_partition(
                    source_path=duplicate_source,
                    strict2031_path=strict2031,
                    strict1607_path=strict1607,
                    output_dir=duplicate_output,
                )
            self.assertFalse(any(duplicate_output.glob("*")))

            for missing in ("sample_id", "image_id", "dataset", "sent", "try_tn"):
                with self.subTest(missing=missing):
                    row = _semantic_row(f"missing-{missing}", 10)
                    del row[missing]
                    source = root / f"missing-{missing}.jsonl"
                    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    output = root / f"output-{missing}"
                    with self.assertRaisesRegex(
                        SemanticTNPartitionError,
                        "critical field|invalid image_id",
                    ):
                        create_partition(
                            source_path=source,
                            strict2031_path=strict2031,
                            strict1607_path=strict1607,
                            output_dir=output,
                        )
                    self.assertFalse(any(output.glob("*")))

    def test_optional_expected_hashes_fail_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            strict2031 = root / "strict2031.jsonl"
            strict1607 = root / "strict1607.jsonl"
            _write_source(source, [_semantic_row("one", 1)])
            _write_manifest(strict2031, [900])
            _write_manifest(strict1607, [901])
            output = root / "output"
            with self.assertRaisesRegex(
                SemanticTNPartitionError, "input hash drift"
            ):
                create_partition(
                    source_path=source,
                    strict2031_path=strict2031,
                    strict1607_path=strict1607,
                    output_dir=output,
                    expected_input_sha256="0" * 64,
                )
            self.assertFalse(any(output.glob("*")))

    def test_verifier_rejects_output_and_contract_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, strict2031, strict1607, _rows, _raw = self._full_fixture(root)
            output = root / "output"
            create_partition(
                source_path=source,
                strict2031_path=strict2031,
                strict1607_path=strict1607,
                output_dir=output,
                calibration_ratio="1/2",
            )
            train = output / OUTPUT_NAMES["train"]
            train.write_bytes(train.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(
                SemanticTNPartitionError, "byte-for-byte|identity drifted"
            ):
                verify_partition(output / AUDIT_NAME)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, strict2031, strict1607, _rows, _raw = self._full_fixture(root)
            output = root / "output"
            create_partition(
                source_path=source,
                strict2031_path=strict2031,
                strict1607_path=strict1607,
                output_dir=output,
            )
            audit_path = output / AUDIT_NAME
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["partition_contract"]["partition"]["seed"] = "tampered"
            audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                SemanticTNPartitionError, "contract hash mismatch"
            ):
                verify_partition(audit_path)


if __name__ == "__main__":
    unittest.main()
