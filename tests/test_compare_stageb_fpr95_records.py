import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_stageb_fpr95_records import (
    RecordComparisonError,
    compare_record_files,
    exact_binary_auroc,
    exact_fpr_at_tpr,
    exact_fpr95,
    render_markdown,
)
from tools.stageb_eval_records import (
    load_eval_manifest,
    make_eval_record,
    sha256_file,
    write_tn_derived_manifest_binding,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class CompareStageBFPR95RecordsTest(unittest.TestCase):
    def _fixture(self, root: Path):
        manifest = root / "manifest.jsonl"
        manifest_rows = []
        for index in range(20):
            split = "refcocog_umd_val" if index < 10 else "refcocoplus_unc_val"
            manifest_rows.append(
                {
                    "sample_id": f"sample:{index}",
                    "image_id": index // 2,
                    "ann_id": 100 + index,
                    "ref_id": 200 + index,
                    "sent_id": 300 + index,
                    "eval_split": split,
                    "instances": [
                        {
                            "replace_category": [
                                "color" if index % 2 == 0 else "spatial"
                            ]
                        }
                    ],
                }
            )
        _write_jsonl(manifest, manifest_rows)
        manifest_hash = sha256_file(manifest)

        def records(candidate):
            rows = []
            for index, manifest_row in enumerate(manifest_rows):
                valid = index != 19
                rows.append(
                    {
                        "schema": "stageb-eval-record-v1",
                        "task": "tn",
                        "manifest_key": "tn_global",
                        "manifest_sha256": manifest_hash,
                        "manifest_n": len(manifest_rows),
                        "manifest_index": index,
                        "sample_id": manifest_row["sample_id"],
                        "image_id": manifest_row["image_id"],
                        "split": manifest_row["eval_split"],
                        "run_id": "candidate" if candidate else "baseline",
                        "valid": valid,
                        "pos_score": 0.6,
                        "neg_score": 0.2 if candidate else 0.8,
                    }
                )
            return rows

        baseline = root / "baseline.records.jsonl"
        candidate = root / "candidate.records.jsonl"
        _write_jsonl(baseline, records(candidate=False))
        _write_jsonl(candidate, records(candidate=True))
        return manifest, baseline, candidate

    def test_exact_fpr95_uses_order_statistic_and_ge_ties(self):
        result = exact_fpr95(list(range(20)), [0, 1, 2])
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["actual_tpr"], 0.95)
        self.assertAlmostEqual(result["fpr"], 2.0 / 3.0)
        self.assertEqual(result["tie_policy"], ">=")

        fpr90 = exact_fpr_at_tpr(list(range(20)), [0, 1, 2], target_tpr=0.90)
        self.assertEqual(fpr90["threshold"], 2.0)
        self.assertAlmostEqual(fpr90["fpr"], 1.0 / 3.0)
        self.assertEqual(exact_binary_auroc([0.5, 0.7], [0.5, 0.1]), 0.875)

    def test_comparison_reports_quantiles_groups_and_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, baseline, candidate = self._fixture(Path(temporary))
            report = compare_record_files(
                baseline_records=baseline,
                candidate_records=candidate,
                manifest_path=manifest,
                bootstrap_iterations=100,
                seed=7,
            )
            markdown = render_markdown(report)
            expected_inputs = {}
            for key, path in (
                ("manifest", manifest),
                ("baseline_records", baseline),
                ("candidate_records", candidate),
            ):
                raw = path.read_bytes()
                expected_inputs[key] = {
                    "path": str(path.resolve()),
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }

        self.assertTrue(report["validation"]["pass"])
        self.assertTrue(
            report["input_files"][
                "identity_is_from_the_same_bytes_used_for_metrics"
            ]
        )
        for key, expected in expected_inputs.items():
            self.assertEqual(report["input_files"][key], expected)
        self.assertEqual(report["validation"]["valid_n"], 19)
        self.assertEqual(report["validation"]["invalid_n"], 1)
        self.assertEqual(
            report["validation"]["baseline_manifest_binding_mode"],
            "legacy_direct_source_v1",
        )
        self.assertEqual(
            report["validation"]["candidate_manifest_binding_mode"],
            "legacy_direct_source_v1",
        )
        self.assertEqual(report["global"]["baseline"]["fpr95"]["fpr"], 1.0)
        self.assertEqual(report["global"]["candidate"]["fpr95"]["fpr"], 0.0)
        self.assertEqual(report["global"]["baseline"]["fpr90"]["fpr"], 1.0)
        self.assertEqual(report["global"]["candidate"]["auroc"], 1.0)
        self.assertEqual(report["global"]["candidate"]["pair_win_rate"], 1.0)
        self.assertEqual(
            report["global"]["baseline"]["positive_quantiles"]["q50"], 0.6
        )
        self.assertEqual(set(report["by_split"]), {"refcocog_umd_val", "refcocoplus_unc_val"})
        self.assertEqual(set(report["by_taxonomy"]), {"color", "spatial"})
        self.assertEqual(report["paired_bootstrap"]["delta_ci_low"], -1.0)
        self.assertEqual(report["paired_bootstrap"]["delta_ci_high"], -1.0)
        self.assertTrue(
            report["paired_bootstrap"]["recomputes_each_model_q05_per_resample"]
        )
        self.assertIn("By Taxonomy", markdown)
        self.assertIn("candidate-minus-baseline", markdown)

    def test_validation_rejects_valid_mask_hash_order_and_sample_drift(self):
        mutations = (
            ("valid mask", lambda rows: rows[0].update(valid=False), "valid mask mismatch"),
            (
                "hash",
                lambda rows: rows[0].update(manifest_sha256="f" * 64),
                "manifest hash mismatch",
            ),
            (
                "order",
                lambda rows: rows[0].update(manifest_index=1),
                "manifest indices are not exact",
            ),
            (
                "sample",
                lambda rows: rows[0].update(sample_id="wrong"),
                "sample_id/order mismatch",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                manifest, baseline, candidate = self._fixture(Path(temporary))
                rows = [json.loads(line) for line in candidate.read_text().splitlines()]
                mutate(rows)
                _write_jsonl(candidate, rows)
                with self.assertRaisesRegex(RecordComparisonError, expected):
                    compare_record_files(
                        baseline_records=baseline,
                        candidate_records=candidate,
                        manifest_path=manifest,
                        bootstrap_iterations=10,
                    )

    def test_two_layer_records_compare_against_locked_source_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source_rows = [
                {
                    "sample_id": f"sample:{index}",
                    "image_id": index,
                    "ann_id": 10 + index,
                    "ref_id": 20 + index,
                    "sent_id": 30 + index,
                    # Preserve the source dataset spelling in the derived row;
                    # the two-layer binding normalizes this to refcocop_val.
                    "eval_split": "refcocoplus_unc_val",
                    "instances": [
                        {
                            "pair_source": "refcoco+_unc",
                            "positive_phrase": "red object",
                            "raw_phrase": "blue object",
                            "replace_category": ["color"],
                        }
                    ],
                }
                for index in range(2)
            ]
            _write_jsonl(source, source_rows)
            derived = root / "derived.jsonl"
            derived_rows = json.loads(json.dumps(source_rows))
            mapping = []
            for index, row in enumerate(derived_rows):
                row["instances"][0]["text_is_negative"] = True
                row["tn_eval_split"] = "refcocop_val"
                row["tn_eval_pair_source"] = "refcoco+_unc"
                row["tn_eval_source_split"] = "val"
                mapping.append(
                    {
                        "derived_index": index,
                        "source_index": index,
                        "sample_id": row["sample_id"],
                        "pair_source": "refcoco+_unc",
                        "source_split": "val",
                        "eval_split": "refcocop_val",
                    }
                )
            _write_jsonl(derived, derived_rows)
            write_tn_derived_manifest_binding(
                source_manifest_path=source,
                derived_manifest_path=derived,
                row_mapping=mapping,
                requested_splits=["refcocop_val"],
                max_pairs=0,
                max_pairs_per_split=0,
                holdout_level="none",
            )
            manifest = load_eval_manifest(
                derived,
                task="tn",
                split="global",
                manifest_key="tn_global",
            )
            paths = []
            for run_id, neg_score in (("baseline", 0.8), ("candidate", 0.2)):
                path = root / f"{run_id}.records.jsonl"
                _write_jsonl(
                    path,
                    [
                        make_eval_record(
                            manifest,
                            index=index,
                            run_id=run_id,
                            valid=True,
                            values={"pos_score": 0.6, "neg_score": neg_score},
                        )
                        for index in range(2)
                    ],
                )
                paths.append(path)
            report = compare_record_files(
                baseline_records=paths[0],
                candidate_records=paths[1],
                manifest_path=source,
                bootstrap_iterations=10,
            )
            emitted = [
                json.loads(line)
                for line in paths[0].read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(
            report["validation"]["baseline_manifest_binding_mode"],
            "source_to_derived_v1",
        )
        self.assertEqual({row["split"] for row in emitted}, {"refcocop_val"})
        self.assertEqual(report["global"]["candidate"]["fpr95"]["fpr"], 0.0)


if __name__ == "__main__":
    unittest.main()
