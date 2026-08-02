import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.select_stageb_u2_category_gate import (
    EXACT_GAPS,
    FORMAL_EVAL_RUNTIME,
    SWEEP_CONTRACT,
    VAL_SPLITS,
    SelectionError,
    build_selection_receipt,
    write_receipt,
)
from tools.stageb_eval_records import RECORD_SCHEMA


class CategoryGateSelectionTest(unittest.TestCase):
    def _fixture(self, root: Path):
        count = 6
        contract = {
            split: {
                "rows": count,
                "sha256": hashlib.sha256(split.encode("ascii")).hexdigest(),
            }
            for split in VAL_SPLITS
        }
        seeds = {
            "refcoco_val": 42,
            "refcocop_val": 300042,
            "refcocog_val": 600042,
        }
        baseline_correct = [True, True, False, False, False, False]
        teacher_correct = [True, True, True, False, False, False]
        safe_feasible = [True, True, True, True, False, False]
        regressing_feasible = [True, True, False, True, True, False]

        def correctness(gap):
            if gap == 0.25:
                return regressing_feasible
            if gap in {0.5, 0.75, 1.0, 1.25}:
                return safe_feasible
            if gap >= 1.5:
                return teacher_correct
            return baseline_correct

        def write_records(path, split, run_id, correct, gap=None, gap_index=0):
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
                    eligible = min(11, gap_index + 1)
                    if gap == 10.0:
                        eligible = 11
                    row.update(
                        {
                            "category_gate_sweep_contract": SWEEP_CONTRACT,
                            "category_gate_max_gap": gap,
                            "category_gate_eligible_queries": eligible,
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
            baseline_run = "b58"
            baseline_path = root / "baseline_records" / f"{split}.jsonl"
            write_records(
                baseline_path, split, baseline_run, baseline_correct
            )
            baseline_rows.append(
                {
                    "dataset": split,
                    "seed": seeds[split],
                    **FORMAL_EVAL_RUNTIME,
                    "manifest_n": count,
                    "manifest_sha256": contract[split]["sha256"],
                    "num_expressions": count,
                    "invalid_records": 0,
                    "run_id": baseline_run,
                    "records_jsonl": str(baseline_path),
                    "acc50": sum(baseline_correct) / count,
                }
            )
            for gap_index, gap in enumerate(EXACT_GAPS):
                correct = correctness(gap)
                run_id = f"u2_gap_{gap:g}"
                record_path = root / "sweep_records" / f"{split}_{gap:g}.jsonl"
                write_records(
                    record_path,
                    split,
                    run_id,
                    correct,
                    gap=gap,
                    gap_index=gap_index,
                )
                sweep_rows.append(
                    {
                        "dataset": split,
                        "seed": seeds[split],
                        **FORMAL_EVAL_RUNTIME,
                        "manifest_n": count,
                        "manifest_sha256": contract[split]["sha256"],
                        "num_expressions": count,
                        "invalid_records": 0,
                        "run_id": run_id,
                        "records_jsonl": str(record_path),
                        "acc50": sum(correct) / count,
                        "category_gate_sweep_contract": SWEEP_CONTRACT,
                        "category_gate_max_gap": gap,
                        "category_gate_single_forward_gap_count": len(EXACT_GAPS),
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
        return sweep_summary, baseline_summary, contract, seeds

    def _build(self, root: Path):
        sweep, baseline, contract, seeds = self._fixture(root)
        return build_selection_receipt(
            sweep_summary=sweep,
            baseline_summary=baseline,
            split_contract=contract,
            baseline_splits=VAL_SPLITS,
            canonical_seeds=seeds,
            expected_query_count=11,
        )

    def test_selects_by_fixed_objective_and_seals_all_gap_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = self._build(root)

            self.assertEqual(receipt["selection"]["max_gap"], 1.25)
            self.assertEqual(
                receipt["contract"]["evaluation_runtime"],
                FORMAL_EVAL_RUNTIME,
            )
            self.assertEqual(len(receipt["gap_results"]), 11)
            self.assertEqual(receipt["feasible_gaps"], [0.25, 0.5, 0.75, 1.0, 1.25])
            selected = receipt["selection"]["aggregate"]
            self.assertEqual(selected["teacher_correct_to_gate_wrong"], 0)
            self.assertEqual(selected["minimum_split_gain_over_baseline"], 2)
            self.assertEqual(selected["gate_minus_teacher"], 3)
            self.assertEqual(len(receipt["payload_sha256"]), 64)

            output = root / "selection_receipt.json"
            write_receipt(output, receipt)
            self.assertEqual(write_receipt(output, receipt), output.resolve())
            observed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(observed["payload_sha256"], receipt["payload_sha256"])

    def test_rejects_test_or_non_val_sweep_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep, baseline, contract, seeds = self._fixture(root)
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            payload["refcoco"][0]["dataset"] = "refcoco_testA"
            sweep.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(SelectionError, "test/non-val"):
                build_selection_receipt(
                    sweep_summary=sweep,
                    baseline_summary=baseline,
                    split_contract=contract,
                    baseline_splits=VAL_SPLITS,
                    canonical_seeds=seeds,
                    expected_query_count=11,
                )

    def test_rejects_gap10_that_is_not_frozen_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep, baseline, contract, seeds = self._fixture(root)
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            row = next(
                row
                for row in payload["refcoco"]
                if row["dataset"] == "refcoco_val"
                and row["category_gate_max_gap"] == 10.0
            )
            record_path = Path(row["records_jsonl"])
            records = [
                json.loads(line)
                for line in record_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["category_gate_winner_query"] = 2
            record_path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SelectionError, "gap10 is not the frozen teacher"):
                build_selection_receipt(
                    sweep_summary=sweep,
                    baseline_summary=baseline,
                    split_contract=contract,
                    baseline_splits=VAL_SPLITS,
                    canonical_seeds=seeds,
                    expected_query_count=11,
                )

    def test_rejects_baseline_batch_size_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep, baseline, contract, seeds = self._fixture(root)
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            payload["refcoco"][0]["batch_size"] = 32
            baseline.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                SelectionError, "fixed evaluation runtime batch_size drifted"
            ):
                build_selection_receipt(
                    sweep_summary=sweep,
                    baseline_summary=baseline,
                    split_contract=contract,
                    baseline_splits=VAL_SPLITS,
                    canonical_seeds=seeds,
                    expected_query_count=11,
                )

    def test_rejects_sweep_fixed_runtime_drift_or_missing_field(self):
        mutations = (
            ("batch_size", 32),
            ("num_workers", 8),
            ("max_batches", 1),
            ("batch_size", None),
            ("num_workers", None),
            ("max_batches", None),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    sweep, baseline, contract, seeds = self._fixture(root)
                    payload = json.loads(sweep.read_text(encoding="utf-8"))
                    if value is None:
                        del payload["refcoco"][0][field]
                    else:
                        payload["refcoco"][0][field] = value
                    sweep.write_text(json.dumps(payload), encoding="utf-8")

                    with self.assertRaises(SelectionError):
                        build_selection_receipt(
                            sweep_summary=sweep,
                            baseline_summary=baseline,
                            split_contract=contract,
                            baseline_splits=VAL_SPLITS,
                            canonical_seeds=seeds,
                            expected_query_count=11,
                        )

    def test_rejects_sweep_canonical_seed_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep, baseline, contract, seeds = self._fixture(root)
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            payload["refcoco"][0]["seed"] += 1
            sweep.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(SelectionError, "canonical seed drifted"):
                build_selection_receipt(
                    sweep_summary=sweep,
                    baseline_summary=baseline,
                    split_contract=contract,
                    baseline_splits=VAL_SPLITS,
                    canonical_seeds=seeds,
                    expected_query_count=11,
                )


if __name__ == "__main__":
    unittest.main()
