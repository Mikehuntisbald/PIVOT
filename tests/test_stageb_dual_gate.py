import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.verify_stageb_dual_gate import (
    evaluate_dual_gate,
    exact_fpr_at_tpr,
    load_record_set,
    load_record_set_with_inputs,
    main as gate_main,
)


def _record(
    *,
    task,
    split,
    index,
    n,
    manifest_hash,
    image_id,
    valid=True,
    pos_score=None,
    neg_score=None,
    correct50=None,
    all_query_best_iou=None,
):
    row = {
        "schema": "stageb-eval-record-v1",
        "task": task,
        "manifest_key": "tn_global" if task == "tn" else f"ref:{split}",
        "manifest_sha256": manifest_hash,
        "manifest_n": n,
        "manifest_index": index,
        "sample_id": f"{task}:{split}:{index}",
        "image_id": image_id,
        "split": split,
        "valid": valid,
    }
    if task == "tn":
        row.update({"pos_score": pos_score, "neg_score": neg_score})
    else:
        row.update(
            {
                "top1_iou": 0.75 if correct50 else 0.25,
                "correct50": bool(correct50),
            }
        )
        if all_query_best_iou is not None:
            row["all_query_best_iou"] = all_query_best_iou
    return row


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class StageBDualGateTest(unittest.TestCase):
    def _write_run(self, root, name, *, candidate, worse_split=False):
        output = root / name
        output.mkdir()
        paths = []

        tn_n = 20
        tn_hash = "a" * 64
        tn_rows = []
        for index in range(tn_n):
            # The exact 95%-TPR threshold is 0.6 for both runs. Candidate has
            # fewer negatives above that threshold.
            baseline_neg = 0.8 if index < 10 else 0.2
            candidate_neg = 0.8 if index < 4 else 0.2
            tn_rows.append(
                _record(
                    task="tn",
                    split="global",
                    index=index,
                    n=tn_n,
                    manifest_hash=tn_hash,
                    image_id=index // 2,
                    pos_score=0.6,
                    neg_score=candidate_neg if candidate else baseline_neg,
                )
            )
        tn_path = output / "tn.records.jsonl"
        _write_jsonl(tn_path, tn_rows)
        paths.append(str(tn_path))

        for split_index, split in enumerate(("refcoco_val", "refcocop_val")):
            n = 8
            manifest_hash = str(split_index + 1) * 64
            baseline_hits = 3
            candidate_hits = 2 if worse_split and split == "refcocop_val" else 5
            rows = [
                _record(
                    task="ref",
                    split=split,
                    index=index,
                    n=n,
                    manifest_hash=manifest_hash,
                    image_id=1000 + split_index * 100 + index // 2,
                    correct50=index < (candidate_hits if candidate else baseline_hits),
                    all_query_best_iou=0.75 if index < 6 else 0.25,
                )
                for index in range(n)
            ]
            path = output / f"{split}.records.jsonl"
            _write_jsonl(path, rows)
            paths.append(str(path))
        return paths

    def test_exact_fpr_uses_order_statistic_and_ge_tie_policy(self):
        result = exact_fpr_at_tpr(list(range(20)), [0, 1, 2], 0.95)
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["actual_tpr"], 0.95)
        self.assertAlmostEqual(result["fpr"], 2.0 / 3.0)

    def test_loader_reports_identity_from_the_same_record_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._write_run(Path(temporary), "baseline", candidate=False)
            groups, inputs = load_record_set_with_inputs(paths)
            self.assertIn("tn_global", groups)
            self.assertEqual(len(inputs), len(paths))
            for path, record in zip(paths, inputs):
                source = Path(path)
                raw = source.read_bytes()
                self.assertEqual(
                    record,
                    {
                        "path": str(source.resolve()),
                        "size_bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    },
                )

    def test_cli_report_includes_same_byte_input_file_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write_run(root, "baseline", candidate=False)
            candidate = self._write_run(root, "candidate", candidate=True)
            output = root / "report.json"
            with redirect_stdout(io.StringIO()):
                status = gate_main(
                    [
                        "--baseline_records",
                        *baseline,
                        "--candidate_records",
                        *candidate,
                        "--required_ref_splits",
                        "refcoco_val",
                        "refcocop_val",
                        "--bootstrap_iterations",
                        "10",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(
                report["input_files"][
                    "identity_is_from_the_same_bytes_used_for_metrics"
                ]
            )
            self.assertEqual(len(report["input_files"]["baseline"]), 3)
            self.assertEqual(len(report["input_files"]["candidate"]), 3)

    def test_gate_passes_only_when_fpr_and_every_ref_split_improve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths = self._write_run(root, "baseline", candidate=False)
            candidate_paths = self._write_run(root, "candidate", candidate=True)
            report = evaluate_dual_gate(
                load_record_set(baseline_paths),
                load_record_set(candidate_paths),
                required_ref_splits=["refcoco_val", "refcocop_val"],
                bootstrap_iterations=100,
                seed=7,
            )
        self.assertTrue(report["validation"]["pass"])
        self.assertTrue(report["gate"]["pass"])
        self.assertLess(report["tn_global"]["candidate_minus_baseline_fpr"], 0.0)
        self.assertEqual(report["tn_global"]["bootstrap"]["num_image_clusters"], 10)
        self.assertTrue(all(row["improved"] for row in report["refcoco"].values()))
        transitions = report["refcoco"]["refcoco_val"]["transitions"]
        self.assertEqual(
            transitions,
            {
                "both_correct": 3,
                "baseline_correct_candidate_wrong": 0,
                "baseline_wrong_candidate_correct": 2,
                "both_wrong": 3,
                "regressions": 0,
                "fixes": 2,
                "net_fixes": 2,
                "net_fixes_rate": 0.25,
            },
        )
        self.assertAlmostEqual(
            transitions["net_fixes_rate"],
            report["refcoco"]["refcoco_val"][
                "candidate_minus_baseline_acc50"
            ],
        )
        self.assertEqual(
            report["refcoco"]["refcoco_val"]["baseline_oracle_recall50"],
            0.75,
        )

    def test_gate_fails_when_one_observed_ref_split_is_not_higher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths = self._write_run(root, "baseline", candidate=False)
            candidate_paths = self._write_run(
                root,
                "candidate",
                candidate=True,
                worse_split=True,
            )
            report = evaluate_dual_gate(
                load_record_set(baseline_paths),
                load_record_set(candidate_paths),
                required_ref_splits=["refcoco_val"],
                bootstrap_iterations=50,
                seed=11,
            )
        self.assertTrue(report["validation"]["pass"])
        self.assertFalse(report["gate"]["pass"])
        self.assertFalse(report["refcoco"]["refcocop_val"]["improved"])
        transitions = report["refcoco"]["refcocop_val"]["transitions"]
        self.assertEqual(transitions["regressions"], 1)
        self.assertEqual(transitions["fixes"], 0)
        self.assertEqual(transitions["net_fixes"], -1)
        self.assertEqual(transitions["net_fixes_rate"], -0.125)

    def test_ref_transitions_count_fixes_and_regressions_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths = self._write_run(root, "baseline", candidate=False)
            candidate_paths = self._write_run(root, "candidate", candidate=True)
            baseline = load_record_set(baseline_paths)
            candidate = load_record_set(candidate_paths)
            # Baseline is correct on 0,1,2. Candidate becomes correct on 1,2,3,4:
            # one regression, two fixes, and therefore one net fix.
            candidate_rows = candidate["ref:refcoco_val"]
            for index, row in enumerate(candidate_rows):
                is_correct = index in {1, 2, 3, 4}
                row["correct50"] = is_correct
                row["top1_iou"] = 0.75 if is_correct else 0.25
            report = evaluate_dual_gate(
                baseline,
                candidate,
                required_ref_splits=["refcoco_val", "refcocop_val"],
                bootstrap_iterations=20,
                seed=13,
            )
        transitions = report["refcoco"]["refcoco_val"]["transitions"]
        self.assertEqual(transitions["both_correct"], 2)
        self.assertEqual(transitions["regressions"], 1)
        self.assertEqual(transitions["fixes"], 2)
        self.assertEqual(transitions["both_wrong"], 3)
        self.assertEqual(transitions["net_fixes"], 1)
        self.assertAlmostEqual(transitions["net_fixes_rate"], 1.0 / 8.0)
        self.assertAlmostEqual(
            report["refcoco"]["refcoco_val"][
                "candidate_minus_baseline_acc50"
            ],
            1.0 / 8.0,
        )

    def test_ref_oracle_scalar_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths = self._write_run(root, "baseline", candidate=False)
            candidate_paths = self._write_run(root, "candidate", candidate=True)
            baseline = load_record_set(baseline_paths)
            candidate = load_record_set(candidate_paths)
            report = evaluate_dual_gate(
                baseline,
                candidate,
                required_ref_splits=["refcoco_val", "refcocop_val"],
                bootstrap_iterations=20,
                seed=17,
            )
        split_report = report["refcoco"]["refcoco_val"]
        self.assertEqual(split_report["baseline_oracle_recall50"], 0.75)
        self.assertEqual(split_report["candidate_oracle_recall50"], 0.75)
        self.assertEqual(split_report["baseline_oracle_headroom"], 0.375)
        self.assertEqual(split_report["candidate_oracle_headroom"], 0.125)
        self.assertEqual(
            report["refcoco"]["refcocop_val"]["baseline_oracle_recall50"],
            0.75,
        )

        candidate["ref:refcoco_val"][0]["all_query_best_iou"] = 0.5
        drifted = evaluate_dual_gate(
            baseline,
            candidate,
            required_ref_splits=["refcoco_val", "refcocop_val"],
            bootstrap_iterations=10,
            seed=19,
        )
        self.assertFalse(drifted["validation"]["pass"])
        self.assertIn(
            "all_query_best_iou oracle scalar drift",
            "\n".join(drifted["validation"]["errors"]),
        )

    def test_validation_rejects_hash_order_duplicates_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths = self._write_run(root, "baseline", candidate=False)
            candidate_paths = self._write_run(root, "candidate", candidate=True)
            candidate = load_record_set(candidate_paths)
            candidate["tn_global"][0]["manifest_sha256"] = "f" * 64
            candidate["tn_global"][1]["sample_id"] = candidate["tn_global"][0]["sample_id"]
            candidate["tn_global"][2]["valid"] = False
            candidate["ref:refcoco_val"][0], candidate["ref:refcoco_val"][1] = (
                candidate["ref:refcoco_val"][1],
                candidate["ref:refcoco_val"][0],
            )
            report = evaluate_dual_gate(
                load_record_set(baseline_paths),
                candidate,
                required_ref_splits=["refcoco_val", "refcocop_val"],
                bootstrap_iterations=10,
            )
        self.assertFalse(report["validation"]["pass"])
        joined = "\n".join(report["validation"]["errors"])
        self.assertIn("multiple manifest hashes", joined)
        self.assertIn("duplicates=1", joined)
        self.assertIn("invalid=1", joined)
        self.assertIn("sample ID order mismatch", joined)


if __name__ == "__main__":
    unittest.main()
