import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tools.aggregate_stageb_paper_results as paper_aggregator
from tools.stageb_eval_records import (
    RECORD_SCHEMA,
    RefRecordContractError,
    RefRecords,
    load_formal_ref_records,
)


class FormalRefRecordContractTest(unittest.TestCase):
    split = "refcoco_val"
    manifest_sha256 = hashlib.sha256(split.encode("ascii")).hexdigest()

    def _fixture(self, root: Path) -> tuple[Path, Path, dict, dict]:
        records_path = root / "refcoco_val.records.jsonl"
        summary_path = root / "summary.json"
        rows = []
        for index, correct in enumerate((True, False)):
            rows.append(
                {
                    "schema": RECORD_SCHEMA,
                    "task": "ref",
                    "manifest_key": f"ref:{self.split}",
                    "manifest_sha256": self.manifest_sha256,
                    "manifest_n": 2,
                    "manifest_index": index,
                    "sample_id": f"sample:{index}",
                    "image_id": index,
                    "ann_id": 10 + index,
                    "ref_id": 20 + index,
                    "sent_id": 30 + index,
                    "split": self.split,
                    "run_id": "formal-run",
                    "valid": True,
                    "correct50": correct,
                    "top1_iou": 0.75 if correct else 0.25,
                }
            )
        records_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary_path.write_text("{}\n", encoding="utf-8")
        summary_row = {
            "records_jsonl": str(records_path),
            "manifest_n": 2,
            "manifest_sha256": self.manifest_sha256,
            "num_expressions": 2,
            "max_batches": 0,
            "run_id": "formal-run",
            "acc50": 0.5,
            "invalid_records": 0,
        }
        contract = {
            self.split: {"rows": 2, "sha256": self.manifest_sha256}
        }
        return records_path, summary_path, summary_row, contract

    def test_leaf_and_paper_compatibility_adapter_return_identical_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_path, summary_path, summary_row, contract = self._fixture(root)
            artifact = {
                "path": str(records_path),
                "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                "size_bytes": records_path.stat().st_size,
            }

            direct = load_formal_ref_records(
                artifact,
                base_dir=root,
                label="formal ref",
                split=self.split,
                summary_row=summary_row,
                summary_path=summary_path,
                split_contract=contract,
            )
            with patch.object(paper_aggregator, "REF_SPLIT_CONTRACT", contract):
                compatible = paper_aggregator._load_ref_records(
                    artifact,
                    base_dir=root,
                    label="formal ref",
                    split=self.split,
                    summary_row=summary_row,
                    summary_path=summary_path,
                )

            self.assertIs(paper_aggregator.RefRecords, RefRecords)
            self.assertIs(
                paper_aggregator.load_formal_ref_records,
                load_formal_ref_records,
            )
            self.assertEqual(direct.path, compatible.path)
            self.assertEqual(direct.file_record, compatible.file_record)
            self.assertEqual(direct.identities, compatible.identities)
            np.testing.assert_array_equal(direct.image_ids, compatible.image_ids)
            np.testing.assert_array_equal(direct.correct50, compatible.correct50)
            self.assertEqual(direct.manifest_sha256, compatible.manifest_sha256)
            self.assertEqual(direct.manifest_n, compatible.manifest_n)

    def test_paper_adapter_preserves_leaf_error_message_and_translates_type(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_path, summary_path, summary_row, contract = self._fixture(root)
            rows = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["top1_iou"] = 0.25
            records_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaises(RefRecordContractError) as direct_context:
                load_formal_ref_records(
                    str(records_path),
                    base_dir=root,
                    label="formal ref",
                    split=self.split,
                    summary_row=summary_row,
                    summary_path=summary_path,
                    split_contract=contract,
                )
            with patch.object(paper_aggregator, "REF_SPLIT_CONTRACT", contract):
                with self.assertRaises(
                    paper_aggregator.PaperAggregationError
                ) as compatible_context:
                    paper_aggregator._load_ref_records(
                        str(records_path),
                        base_dir=root,
                        label="formal ref",
                        split=self.split,
                        summary_row=summary_row,
                        summary_path=summary_path,
                    )

            self.assertEqual(
                str(compatible_context.exception),
                str(direct_context.exception),
            )
            self.assertIsInstance(
                compatible_context.exception.__cause__,
                RefRecordContractError,
            )


if __name__ == "__main__":
    unittest.main()
