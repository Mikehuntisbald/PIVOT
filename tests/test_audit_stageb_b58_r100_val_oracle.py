import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_stageb_b58_r100_val_oracle import (
    OracleAuditError,
    build_oracle_receipt,
    write_receipt,
)
from tools.select_stageb_u2_category_gate import SWEEP_CONTRACT, VAL_SPLITS
from tools.stageb_eval_records import RECORD_SCHEMA


class B58R100ValOracleAuditTest(unittest.TestCase):
    GAPS = (0.0, 10.0)
    SEEDS = {
        "refcoco_val": 42,
        "refcocop_val": 300042,
        "refcocog_val": 600042,
    }

    def _fixture(self, root: Path):
        count = 4
        query_count = 3
        contract = {
            split: {
                "rows": count,
                "sha256": hashlib.sha256(split.encode("ascii")).hexdigest(),
            }
            for split in VAL_SPLITS
        }
        base_correct = [True, True, False, False]
        r100_correct = [True, False, True, False]

        def write_records(path, split, run_id, correct, gap=None):
            rows = []
            for index, is_correct in enumerate(correct):
                row = {
                    "schema": RECORD_SCHEMA,
                    "task": "ref",
                    "manifest_key": f"ref:{split}",
                    "manifest_sha256": contract[split]["sha256"],
                    "manifest_n": count,
                    "manifest_index": index,
                    "sample_id": f"{split}:{index}",
                    "image_id": index,
                    "ann_id": 100 + index,
                    "ref_id": 200 + index,
                    "sent_id": 300 + index,
                    "split": split,
                    "run_id": run_id,
                    "valid": True,
                    "correct50": bool(is_correct),
                    "top1_iou": 0.75 if is_correct else 0.25,
                    "all_query_best_iou": 0.9,
                }
                if gap is not None:
                    row.update(
                        {
                            "category_gate_sweep_contract": SWEEP_CONTRACT,
                            "category_gate_max_gap": gap,
                            "category_gate_eligible_queries": (
                                query_count if gap == 10.0 else 1
                            ),
                            "category_gate_winner_query": 0,
                            "category_gate_teacher_winner_query": 0,
                            "category_gate_patch_winner_query": 1,
                        }
                    )
                rows.append(row)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

        baseline_rows = []
        sweep_rows = []
        for split in VAL_SPLITS:
            baseline_path = root / "baseline" / f"{split}.jsonl"
            write_records(baseline_path, split, "b58", base_correct)
            baseline_rows.append(
                {
                    "dataset": split,
                    "seed": self.SEEDS[split],
                    "max_batches": 0,
                    "manifest_n": count,
                    "manifest_sha256": contract[split]["sha256"],
                    "num_expressions": count,
                    "invalid_records": 0,
                    "run_id": "b58",
                    "records_jsonl": str(baseline_path),
                    "acc50": sum(base_correct) / count,
                }
            )
            for gap in self.GAPS:
                correct = r100_correct if gap == 10.0 else base_correct
                record_path = root / "sweep" / f"{split}_{gap:g}.jsonl"
                write_records(record_path, split, f"u2_gap_{gap:g}", correct, gap)
                sweep_rows.append(
                    {
                        "dataset": split,
                        "seed": self.SEEDS[split],
                        "max_batches": 0,
                        "manifest_n": count,
                        "manifest_sha256": contract[split]["sha256"],
                        "num_expressions": count,
                        "invalid_records": 0,
                        "run_id": f"u2_gap_{gap:g}",
                        "records_jsonl": str(record_path),
                        "acc50": sum(correct) / count,
                        "category_gate_sweep_contract": SWEEP_CONTRACT,
                        "category_gate_max_gap": gap,
                        "category_gate_single_forward_gap_count": len(self.GAPS),
                        "checkpoint": "u2.pth",
                        "checkpoint_sha256": "a" * 64,
                        "config": "gate.py",
                        "config_sha256": "b" * 64,
                        "amp": True,
                        "device": "cuda:0",
                    }
                )
        baseline_summary = root / "baseline_summary.json"
        sweep_summary = root / "sweep_summary.json"
        baseline_summary.write_text(
            json.dumps({"refcoco": baseline_rows, "tn": []}), encoding="utf-8"
        )
        sweep_summary.write_text(
            json.dumps({"refcoco": sweep_rows, "tn": []}), encoding="utf-8"
        )
        return baseline_summary, sweep_summary, contract, query_count

    def _build(self, root: Path):
        baseline, sweep, contract, query_count = self._fixture(root)
        return build_oracle_receipt(
            baseline_summary=baseline,
            sweep_summary=sweep,
            split_contract=contract,
            baseline_splits=VAL_SPLITS,
            val_splits=VAL_SPLITS,
            gaps=self.GAPS,
            canonical_seeds=self.SEEDS,
            expected_query_count=query_count,
        )

    def test_seals_exact_paired_oracle_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = self._build(root)
            for result in receipt["split_results"].values():
                self.assertEqual(result["base_only"], 1)
                self.assertEqual(result["r100_only"], 1)
                self.assertEqual(result["both_correct"], 1)
                self.assertEqual(result["both_wrong"], 1)
                self.assertEqual(result["oracle_correct"], 3)
            self.assertEqual(receipt["aggregate"]["n"], 12)
            self.assertEqual(receipt["aggregate"]["base_only"], 3)
            self.assertEqual(receipt["aggregate"]["r100_only"], 3)
            self.assertEqual(receipt["aggregate"]["oracle_correct"], 9)
            self.assertFalse(receipt["test_records_loaded"])
            self.assertTrue(receipt["not_a_selector"])

            output = root / "receipt.json"
            write_receipt(output, receipt)
            self.assertEqual(write_receipt(output, receipt), output.resolve())

    def test_rejects_non_val_sweep_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, sweep, contract, query_count = self._fixture(root)
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            payload["refcoco"][0]["dataset"] = "refcoco_testA"
            sweep.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(OracleAuditError, "test/non-val"):
                build_oracle_receipt(
                    baseline_summary=baseline,
                    sweep_summary=sweep,
                    split_contract=contract,
                    baseline_splits=VAL_SPLITS,
                    val_splits=VAL_SPLITS,
                    gaps=self.GAPS,
                    canonical_seeds=self.SEEDS,
                    expected_query_count=query_count,
                )

    def test_rejects_gap10_that_does_not_admit_every_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, sweep, contract, query_count = self._fixture(root)
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            row = next(
                item
                for item in payload["refcoco"]
                if item["dataset"] == "refcoco_val"
                and item["category_gate_max_gap"] == 10.0
            )
            path = Path(row["records_jsonl"])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]["category_gate_eligible_queries"] = query_count - 1
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OracleAuditError, "did not admit all queries"):
                build_oracle_receipt(
                    baseline_summary=baseline,
                    sweep_summary=sweep,
                    split_contract=contract,
                    baseline_splits=VAL_SPLITS,
                    val_splits=VAL_SPLITS,
                    gaps=self.GAPS,
                    canonical_seeds=self.SEEDS,
                    expected_query_count=query_count,
                )


if __name__ == "__main__":
    unittest.main()
