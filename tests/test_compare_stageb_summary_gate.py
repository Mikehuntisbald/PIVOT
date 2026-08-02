import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools import compare_stageb_summary_gate as gate


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path):
    raw = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class StageBSummaryGateTest(unittest.TestCase):
    def _fixture(self, root: Path):
        baseline_checkpoint = root / "baseline.pth"
        baseline_checkpoint.write_bytes(b"baseline checkpoint")
        candidate_checkpoint = str(root / "candidate.pth")

        contracts = {}
        baseline_ref_rows = []
        candidate_ref_rows = []
        for index, split in enumerate(gate.REF_SPLITS):
            manifest_sha = f"{index + 1:064x}"
            n = 10 + index
            seed = 42 + index * 100000
            contracts[split] = {
                "manifest_n": n,
                "manifest_sha256": manifest_sha,
                "seed": seed,
            }
            common = {
                "dataset": split,
                "manifest_n": n,
                "manifest_sha256": manifest_sha,
                "num_expressions": n,
                "seed": seed,
                "max_batches": 0,
                "invalid_records": 0,
            }
            baseline_ref_rows.append(
                dict(
                    common,
                    acc50=0.50,
                    checkpoint=str(baseline_checkpoint),
                    valid_mask_expressions=n,
                    invalid_mask_expressions=0,
                    records_jsonl=f"baseline-{split}.records.jsonl",
                )
            )
            candidate_ref_rows.append(
                dict(
                    common,
                    acc50=0.60,
                    checkpoint=candidate_checkpoint,
                    valid_mask_expressions=n,
                    invalid_mask_expressions=0,
                    records_jsonl=f"{split}.records.jsonl",
                )
            )

        baseline_ref = root / "baseline_ref.json"
        candidate_ref = root / "candidate_ref.json"
        _write_json(baseline_ref, {"refcoco": baseline_ref_rows, "tn": []})
        _write_json(candidate_ref, {"refcoco": candidate_ref_rows, "tn": []})

        section = "strict-small"
        tn_sha = "f" * 64
        derived_tn_sha = "e" * 64
        binding_sha = "d" * 64
        baseline_tn_row = {
            "manifest_n": 12,
            "manifest_sha256": derived_tn_sha,
            "source_manifest_sha256": tn_sha,
            "manifest_binding_sha256": binding_sha,
            "num_pairs": 12,
            "seed": 42,
            "max_batches": 0,
            "invalid_positive_pairs": 0,
            "invalid_negative_pairs": 0,
            "invalid_records": 0,
            "fpr95tpr": 0.20,
            "actual_tpr_at_95tpr": 0.95,
            "records_jsonl": "baseline-strict-small.records.jsonl",
        }
        candidate_tn_row = dict(
            baseline_tn_row,
            checkpoint=candidate_checkpoint,
            fpr95tpr=0.10,
            actual_tpr_at_95tpr=0.95,
            records_jsonl="strict-small.records.jsonl",
        )
        baseline_tn = root / "baseline_tn.json"
        candidate_tn = root / "candidate_tn.json"
        _write_json(baseline_tn, {"refcoco": [], "tn": [baseline_tn_row]})
        _write_json(candidate_tn, {"refcoco": [], "tn": [candidate_tn_row]})

        manifest = {
            "schema": gate.BASELINE_SCHEMA,
            "baseline_id": "fixture-baseline",
            "checkpoint": _file_record(baseline_checkpoint),
            "ref8": {
                "status": gate.FORMAL_STATUS,
                "source_summary": _file_record(baseline_ref),
                "protocol": {"id": "fixture-ref-v1", "splits": contracts},
                "metrics": {split: {"acc50": 0.50} for split in gate.REF_SPLITS},
            },
            "tn": {
                section: {
                    "status": gate.FORMAL_STATUS,
                    "source_summary": _file_record(baseline_tn),
                    "protocol": {
                        "id": "fixture-tn-v1",
                        "manifest_n": 12,
                        "manifest_sha256": tn_sha,
                        "seed": 42,
                        "target_tpr": 0.95,
                    },
                    "fpr95tpr": 0.20,
                }
            },
        }
        manifest_path = root / "baseline_manifest.json"
        _write_json(manifest_path, manifest)
        return {
            "manifest": manifest_path,
            "candidate_ref": candidate_ref,
            "candidate_tn": {section: candidate_tn},
            "section": section,
        }

    def test_formal_matching_protocol_strictly_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            report = gate.compare_summary_gate(
                baseline_manifest=fixture["manifest"],
                candidate_ref_summary=fixture["candidate_ref"],
                candidate_tn_summaries=fixture["candidate_tn"],
            )
        self.assertTrue(report["validation"]["pass"])
        self.assertTrue(report["gate"]["eligible"])
        self.assertTrue(report["gate"]["pass"])
        self.assertTrue(report["ref8"]["all_strictly_higher"])
        self.assertTrue(report["tn"][fixture["section"]]["strictly_lower"])

    def test_equality_fails_the_strict_metric_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            summary = json.loads(fixture["candidate_ref"].read_text(encoding="utf-8"))
            summary["refcoco"][0]["acc50"] = 0.50
            _write_json(fixture["candidate_ref"], summary)
            report = gate.compare_summary_gate(
                baseline_manifest=fixture["manifest"],
                candidate_ref_summary=fixture["candidate_ref"],
                candidate_tn_summaries=fixture["candidate_tn"],
            )
            with redirect_stdout(io.StringIO()):
                status = gate.main(
                    [
                        "--baseline-manifest",
                        str(fixture["manifest"]),
                        "--candidate-ref-summary",
                        str(fixture["candidate_ref"]),
                        "--candidate-tn-summary",
                        f"{fixture['section']}={fixture['candidate_tn'][fixture['section']]}",
                    ]
                )
        self.assertTrue(report["gate"]["eligible"])
        self.assertFalse(report["gate"]["pass"])
        self.assertFalse(report["ref8"]["splits"][gate.REF_SPLITS[0]]["strictly_higher"])
        self.assertEqual(status, 1)

    def test_protocol_mismatch_is_ineligible_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            summary = json.loads(fixture["candidate_ref"].read_text(encoding="utf-8"))
            summary["refcoco"][0]["manifest_sha256"] = "e" * 64
            _write_json(fixture["candidate_ref"], summary)
            report = gate.compare_summary_gate(
                baseline_manifest=fixture["manifest"],
                candidate_ref_summary=fixture["candidate_ref"],
                candidate_tn_summaries=fixture["candidate_tn"],
            )
        split = report["ref8"]["splits"][gate.REF_SPLITS[0]]
        self.assertTrue(split["numeric_strictly_higher"])
        self.assertFalse(split["protocol_match"])
        self.assertIsNone(split["strictly_higher"])
        self.assertFalse(report["gate"]["eligible"])
        self.assertFalse(report["gate"]["pass"])

    def test_nonformal_baseline_never_passes_even_when_numbers_improve(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
            manifest["ref8"]["status"] = "diagnostic_only"
            manifest["ref8"]["reason"] = "historical summary is not formally sealed"
            _write_json(fixture["manifest"], manifest)
            report = gate.compare_summary_gate(
                baseline_manifest=fixture["manifest"],
                candidate_ref_summary=fixture["candidate_ref"],
                candidate_tn_summaries=fixture["candidate_tn"],
            )
        self.assertTrue(report["ref8"]["numeric_all_strictly_higher"])
        self.assertIsNone(report["ref8"]["all_strictly_higher"])
        self.assertFalse(report["gate"]["eligible"])
        self.assertFalse(report["gate"]["pass"])

    def test_unavailable_formal_tn_does_not_compare_diagnostic_fpr(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
            manifest["tn"][fixture["section"]] = {
                "status": "formal_baseline_unavailable",
                "reason": "legacy diagnostic used a different manifest",
            }
            _write_json(fixture["manifest"], manifest)
            report = gate.compare_summary_gate(
                baseline_manifest=fixture["manifest"],
                candidate_ref_summary=fixture["candidate_ref"],
                candidate_tn_summaries=fixture["candidate_tn"],
            )
        tn = report["tn"][fixture["section"]]
        self.assertIsNone(tn["baseline_fpr95tpr"])
        self.assertIsNone(tn["strictly_lower"])
        self.assertFalse(report["gate"]["eligible"])

    def test_missing_ref_split_is_a_validation_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            summary = json.loads(fixture["candidate_ref"].read_text(encoding="utf-8"))
            summary["refcoco"].pop()
            _write_json(fixture["candidate_ref"], summary)
            with self.assertRaisesRegex(gate.SummaryGateError, "split order/completeness"):
                gate.compare_summary_gate(
                    baseline_manifest=fixture["manifest"],
                    candidate_ref_summary=fixture["candidate_ref"],
                    candidate_tn_summaries=fixture["candidate_tn"],
                )

    def test_missing_tn_section_is_a_validation_error_and_cli_returns_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            with self.assertRaisesRegex(gate.SummaryGateError, "candidate TN sections differ"):
                gate.compare_summary_gate(
                    baseline_manifest=fixture["manifest"],
                    candidate_ref_summary=fixture["candidate_ref"],
                    candidate_tn_summaries={},
                )
            with redirect_stderr(io.StringIO()):
                status = gate.main(
                    [
                        "--baseline-manifest",
                        str(fixture["manifest"]),
                        "--candidate-ref-summary",
                        str(fixture["candidate_ref"]),
                    ]
                )
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
