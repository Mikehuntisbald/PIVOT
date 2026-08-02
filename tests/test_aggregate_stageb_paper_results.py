import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tools.aggregate_stageb_paper_results as paper_aggregator
import tools.stageb_eval_records as eval_records

from tools.aggregate_stageb_paper_results import (
    REF_SPLITS,
    PaperAggregationError,
    RefRecords,
    _headline_acceptance,
    aggregate_manifest,
    paired_ref_seed_first_bootstrap,
    render_csv,
    render_markdown,
    write_report,
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path):
    return {"path": str(path), "sha256": _sha256(path)}


class AggregateStageBPaperResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_contract = {
            split: {
                "rows": 6,
                "sha256": hashlib.sha256(split.encode("ascii")).hexdigest(),
            }
            for split in REF_SPLITS
        }
        contract_patch = patch(
            "tools.aggregate_stageb_paper_results.REF_SPLIT_CONTRACT",
            fixture_contract,
        )
        contract_patch.start()
        self.addCleanup(contract_patch.stop)

    def _fixture(self, root: Path) -> Path:
        tn_manifests = {}
        tn_rows_by_split = {}
        for strict_split, eval_split in (
            ("strict2031", "refcocop_val"),
            ("strict1607", "refcocog_umd_val"),
        ):
            path = root / "protocol" / f"{strict_split}.jsonl"
            rows = [
                {
                    "sample_id": f"{strict_split}:{index}",
                    "image_id": index // 2,
                    "ann_id": 1000 + index,
                    "ref_id": 2000 + index,
                    "sent_id": 3000 + index,
                    "eval_split": eval_split,
                    "instances": [{"replace_category": ["color"]}],
                }
                for index in range(20)
            ]
            _write_jsonl(path, rows)
            tn_manifests[strict_split] = path
            tn_rows_by_split[strict_split] = rows

        config = root / "train" / "config.py"
        data = root / "train" / "data.jsonl"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("seed = 0\n", encoding="utf-8")
        _write_jsonl(data, [{"row": 1}])

        experiments = []
        for experiment_id in ("baseline", "candidate"):
            runs = []
            for seed in (11, 22):
                run_root = root / "runs" / experiment_id / str(seed)
                checkpoint = run_root / "checkpoint.pth"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"{experiment_id}:{seed}".encode("ascii"))
                run_id = f"{experiment_id}_seed{seed}"

                ref_summary_path = run_root / "ref8" / "summary.json"
                ref_rows = []
                ref_records = {}
                for split_index, split in enumerate(REF_SPLITS):
                    record_path = run_root / "ref8" / f"{split}.records.jsonl"
                    manifest_hash = hashlib.sha256(split.encode("ascii")).hexdigest()
                    if experiment_id == "baseline":
                        hits = 3 if seed == 11 else 4
                    else:
                        hits = 5 if seed == 11 else 6
                    records = []
                    for index in range(6):
                        records.append(
                            {
                                "schema": "stageb-eval-record-v1",
                                "task": "ref",
                                "manifest_key": f"ref:{split}",
                                "manifest_sha256": manifest_hash,
                                "manifest_n": 6,
                                "manifest_index": index,
                                "sample_id": f"{split}:{index}",
                                "image_id": split_index * 100 + index // 2,
                                "ann_id": 10 + index,
                                "ref_id": 20 + index,
                                "sent_id": 30 + index,
                                "split": split,
                                "run_id": run_id,
                                "valid": True,
                                "correct50": index < hits,
                                "top1_iou": 0.8 if index < hits else 0.1,
                            }
                        )
                    _write_jsonl(record_path, records)
                    ref_records[split] = _artifact(record_path)
                    ref_rows.append(
                        {
                            "run_id": run_id,
                            "checkpoint": str(checkpoint),
                            "dataset": split,
                            "acc50": hits / 6.0,
                            "num_expressions": 6,
                            "manifest_n": 6,
                            "manifest_sha256": manifest_hash,
                            "records_jsonl": str(record_path),
                            "invalid_records": 0,
                            "max_batches": 0,
                        }
                    )
                _write_json(ref_summary_path, {"refcoco": ref_rows, "tn": []})

                tn_results = {}
                for strict_split, manifest_path in tn_manifests.items():
                    tn_root = run_root / strict_split
                    record_path = tn_root / "tn.records.jsonl"
                    summary_path = tn_root / "summary.json"
                    manifest_hash = _sha256(manifest_path)
                    neg_score = 0.8 if experiment_id == "baseline" else 0.2
                    records = []
                    for index, source in enumerate(tn_rows_by_split[strict_split]):
                        records.append(
                            {
                                "schema": "stageb-eval-record-v1",
                                "task": "tn",
                                "manifest_key": "tn_global",
                                "manifest_sha256": manifest_hash,
                                "manifest_n": 20,
                                "manifest_index": index,
                                "sample_id": source["sample_id"],
                                "image_id": source["image_id"],
                                "split": source["eval_split"],
                                "run_id": run_id,
                                "valid": True,
                                "pos_score": 0.6,
                                "neg_score": neg_score,
                            }
                        )
                    _write_jsonl(record_path, records)
                    _write_json(
                        summary_path,
                        {
                            "refcoco": [],
                            "tn": [
                                {
                                    "run_id": run_id,
                                    "checkpoint": str(checkpoint),
                                    "fpr95tpr": 1.0 if experiment_id == "baseline" else 0.0,
                                    "fpr90tpr": 1.0 if experiment_id == "baseline" else 0.0,
                                    "threshold_at_95tpr": 0.6,
                                    "actual_tpr_at_95tpr": 1.0,
                                    "pair_win_rate": 0.0 if experiment_id == "baseline" else 1.0,
                                    "pair_tie_rate": 0.0,
                                    "pos_score_mean": 0.6,
                                    "tn_score_mean": neg_score,
                                    "score_gap_mean": 0.6 - neg_score,
                                    "num_pairs": 20,
                                    "manifest_n": 20,
                                    "manifest_sha256": manifest_hash,
                                    "records_jsonl": str(record_path),
                                    "invalid_records": 0,
                                    "max_batches": 0,
                                }
                            ],
                        },
                    )
                    tn_results[strict_split] = {
                        "summary": _artifact(summary_path),
                        "records": _artifact(record_path),
                    }
                runs.append(
                    {
                        "train_seed": seed,
                        "artifacts": {
                            "checkpoint": _artifact(checkpoint),
                            "config": _artifact(config),
                            "data": [_artifact(data)],
                        },
                        "results": {
                            "ref": {
                                "summary": _artifact(ref_summary_path),
                                "records": ref_records,
                            },
                            "tn": tn_results,
                        },
                    }
                )
            experiments.append(
                {"id": experiment_id, "label": experiment_id.title(), "runs": runs}
            )

        manifest = root / "paper_manifest.json"
        _write_json(
            manifest,
            {
                "schema": "stageb-paper-results-manifest-v1",
                "expected_train_seeds": [11, 22],
                "baseline_experiment": "baseline",
                "protocol": {
                    "ref_splits": list(REF_SPLITS),
                    "tn_splits": {
                        split: {
                            "manifest": _artifact(path),
                            "expected_n": 20,
                        }
                        for split, path in tn_manifests.items()
                    },
                    "bootstrap": {
                        "iterations": 40,
                        "confidence": 0.95,
                        "seed": 7,
                    },
                },
                "experiments": experiments,
            },
        )
        return manifest

    def test_aggregates_seed_statistics_exact_fpr_and_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            report = aggregate_manifest(manifest)
            outputs = write_report(report, root / "aggregate")
            markdown = render_markdown(report)
            csv_text = render_csv(report)

            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["validation"]["pass"])
            baseline = report["experiments"]["baseline"]["aggregate"]
            candidate = report["experiments"]["candidate"]["aggregate"]
            self.assertAlmostEqual(baseline["ref"]["mean8_acc50"]["mean"], 7.0 / 12.0)
            self.assertAlmostEqual(candidate["ref"]["mean8_acc50"]["mean"], 11.0 / 12.0)
            self.assertGreater(baseline["ref"]["mean8_acc50"]["std"], 0.0)
            self.assertEqual(baseline["tn"]["strict2031"]["mean"], 1.0)
            self.assertEqual(candidate["tn"]["strict1607"]["mean"], 0.0)
            self.assertEqual(
                baseline["tn_metrics"]["strict2031"]["fpr90"]["mean"],
                1.0,
            )
            self.assertEqual(
                candidate["tn_metrics"]["strict2031"]["auroc"]["mean"],
                1.0,
            )
            self.assertEqual(
                candidate["tn_metrics"]["strict2031"]["pair_win_rate"]["mean"],
                1.0,
            )
            color = candidate["tn_taxonomy"]["strict2031"]["color"]
            self.assertEqual(color["records_n"], 20)
            self.assertEqual(color["metrics"]["fpr95"]["mean"], 0.0)

            paired = report["comparisons_to_baseline"]["candidate"]["per_seed"]["11"]
            self.assertAlmostEqual(
                paired["ref"]["mean8_acc50"]["observed_candidate_minus_baseline"],
                1.0 / 3.0,
            )
            self.assertEqual(
                paired["tn"]["strict2031"][
                    "observed_candidate_minus_baseline_fpr95"
                ],
                -1.0,
            )
            self.assertEqual(paired["tn"]["strict2031"]["unit"], "image_cluster")
            self.assertEqual(
                paired["tn"]["strict2031"]["observed_metric_deltas"]["auroc"],
                1.0,
            )
            self.assertIn("Stage-B Paper Results", markdown)
            self.assertIn("TN Secondary Metrics", markdown)
            self.assertIn("TN Taxonomy", markdown)
            self.assertNotIn("NOT FOR PAPER TABLES", markdown)
            self.assertIn("auroc", csv_text)
            self.assertIn("paired_bootstrap", csv_text)
            for path in outputs.values():
                self.assertTrue(Path(path).is_file())
            with self.assertRaises(FileExistsError):
                write_report(report, root / "aggregate")

    def test_missing_seed_fails_closed_and_allow_incomplete_is_watermarked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["experiments"][1]["runs"].pop()
            _write_json(manifest, payload)

            with self.assertRaisesRegex(PaperAggregationError, r"missing=\[22\]"):
                aggregate_manifest(manifest)
            report = aggregate_manifest(manifest, allow_incomplete=True)
            self.assertEqual(report["status"], "incomplete")
            self.assertFalse(report["validation"]["pass"])
            self.assertIn("NOT FOR PAPER TABLES", render_markdown(report))
            self.assertIn("incomplete", render_csv(report))

    def test_verified_builder_receipt_bundle_unlocks_provenance_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._fixture(Path(temporary))
            verified = {
                "passed": True,
                "verdict": "verified_one_time_final_release",
                "data_contract_status": "complete",
                "unmet": [],
            }
            with patch(
                "tools.stageb_headline_release_contract."
                "verify_manifest_release_provenance",
                return_value=verified,
            ) as replay:
                report = aggregate_manifest(manifest)
            self.assertEqual(replay.call_count, 1)
            self.assertTrue(report["headline_provenance"]["passed"])
            self.assertTrue(
                report["comparisons_to_baseline"]["candidate"]
                ["headline_acceptance"]["gates"]["provenance"]["passed"]
            )
            self.assertIsNone(report["headline_watermark"])

    def test_fixed_historical_baseline_compares_each_candidate_seed_without_replication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            baseline = payload["experiments"][0]
            baseline["runs"] = baseline["runs"][:1]
            baseline["expected_train_seeds"] = [11]
            baseline["reference_role"] = "fixed_historical_checkpoint"
            candidate = payload["experiments"][1]
            candidate["expected_train_seeds"] = [11, 22]
            candidate["reference_role"] = "training_seed_distribution"
            _write_json(manifest, payload)

            report = aggregate_manifest(manifest)
            comparison = report["comparisons_to_baseline"]["candidate"]

            self.assertEqual(report["status"], "complete")
            self.assertEqual(
                report["experiments"]["baseline"]["aggregate"]["ref"][
                    "mean8_acc50"
                ]["n"],
                1,
            )
            self.assertEqual(
                report["experiments"]["candidate"]["aggregate"]["ref"][
                    "mean8_acc50"
                ]["n"],
                2,
            )
            self.assertEqual(
                comparison["comparison_mode"],
                "candidate_seeds_vs_fixed_historical_checkpoint",
            )
            self.assertTrue(comparison["baseline_is_not_pseudo_replicated"])
            self.assertEqual(comparison["baseline_reference_seeds"], [11])
            self.assertEqual(comparison["candidate_train_seeds"], [11, 22])
            self.assertEqual(set(comparison["per_seed"]), {"11", "22"})
            self.assertEqual(
                comparison["per_seed"]["11"]["baseline_reference_seed"], 11
            )
            self.assertEqual(
                comparison["per_seed"]["22"]["baseline_reference_seed"], 11
            )
            self.assertEqual(
                comparison["observed_delta_across_train_seeds"]["ref"][
                    "mean8_acc50"
                ]["n"],
                2,
            )
            headline = comparison["headline_seed_first_bootstrap"]
            self.assertEqual(
                headline["draw_contract"]["ref"]["unit"],
                "global_canonical_coco_image_cluster",
            )
            self.assertEqual(
                headline["draw_contract"]["ref"]["candidate_train_seeds"],
                [11, 22],
            )
            self.assertEqual(
                headline["draw_contract"]["tn"]["strict2031"][
                    "candidate_train_seeds"
                ],
                [11, 22],
            )
            self.assertTrue(
                headline["draw_contract"]["tn"]["strict2031"][
                    "recomputes_each_model_q05_per_resample"
                ]
            )
            self.assertEqual(
                headline["tn"]["strict2031"]["positive_q05_threshold"][
                    "observed_candidate_minus_baseline"
                ],
                0.0,
            )
            self.assertTrue(
                comparison["headline_acceptance"]["gates"][
                    "positive_q05_noninferiority"
                ]["passed"]
            )
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["headline_provenance"]["passed"])
            self.assertIn(
                "one_time_final_evaluation_gate_receipt",
                report["headline_provenance"]["unmet"],
            )
            self.assertFalse(
                comparison["headline_acceptance"]["gates"]["provenance"][
                    "passed"
                ]
            )
            self.assertFalse(comparison["headline_acceptance"]["pass"])
            self.assertIn(
                "HEADLINE ACCEPTANCE UNAVAILABLE",
                render_markdown(report),
            )

    def test_ref8_bootstrap_preserves_cross_dataset_same_image_anticorrelation(self):
        def records(split: str, correct: list[bool]) -> RefRecords:
            return RefRecords(
                path=Path(f"/{split}.jsonl"),
                file_record={},
                identities=(
                    (split, 0, 1),
                    (split, 1, 2),
                ),
                image_ids=np.asarray([1, 2], dtype=np.int64),
                correct50=np.asarray(correct, dtype=np.bool_),
                manifest_sha256="a" * 64,
                manifest_n=2,
            )

        splits = ("dataset_a", "dataset_b")
        baseline = {
            "dataset_a": records("dataset_a", [False, True]),
            "dataset_b": records("dataset_b", [True, False]),
        }
        candidate = {
            "dataset_a": records("dataset_a", [True, False]),
            "dataset_b": records("dataset_b", [False, True]),
        }
        result = paired_ref_seed_first_bootstrap(
            baseline,
            {17: candidate, 42: candidate, 73: candidate},
            splits,
            iterations=200,
            confidence=0.95,
            seed=19,
        )
        headline = result["mean_across_candidate_train_seeds"][
            "mean8_acc50"
        ]
        self.assertEqual(headline["observed_candidate_minus_baseline"], 0.0)
        self.assertEqual(headline["delta_std"], 0.0)
        self.assertEqual(headline["delta_ci_low"], 0.0)
        self.assertEqual(headline["delta_ci_high"], 0.0)

        # The former split-independent draw destroys the exact anticorrelation.
        rng = np.random.default_rng(19)
        independent = []
        for _ in range(200):
            draw_a = rng.integers(0, 2, size=2)
            draw_b = rng.integers(0, 2, size=2)
            delta_a = np.asarray([1.0, -1.0])[draw_a].mean()
            delta_b = np.asarray([-1.0, 1.0])[draw_b].mean()
            independent.append(0.5 * (delta_a + delta_b))
        self.assertGreater(float(np.std(independent)), 0.0)

    def test_positive_q05_collapse_fails_headline_gate_despite_lower_fpr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            baseline = payload["experiments"][0]
            baseline["runs"] = baseline["runs"][:1]
            baseline["expected_train_seeds"] = [11]
            baseline["reference_role"] = "fixed_historical_checkpoint"
            candidate = payload["experiments"][1]
            candidate["expected_train_seeds"] = [11, 22]
            candidate["reference_role"] = "training_seed_distribution"
            for run in candidate["runs"]:
                for split in ("strict2031", "strict1607"):
                    result = run["results"]["tn"][split]
                    record_path = Path(result["records"]["path"])
                    rows = [
                        json.loads(line)
                        for line in record_path.read_text(encoding="utf-8").splitlines()
                    ]
                    for row in rows:
                        row["pos_score"] = 0.5
                    _write_jsonl(record_path, rows)
                    result["records"]["sha256"] = _sha256(record_path)
                    summary_path = Path(result["summary"]["path"])
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    row = summary["tn"][0]
                    row["threshold_at_95tpr"] = 0.5
                    row["pos_score_mean"] = 0.5
                    row["score_gap_mean"] = 0.3
                    _write_json(summary_path, summary)
                    result["summary"]["sha256"] = _sha256(summary_path)
            _write_json(manifest, payload)

            report = aggregate_manifest(manifest)
            comparison = report["comparisons_to_baseline"]["candidate"]
            self.assertTrue(
                comparison["headline_acceptance"]["gates"][
                    "strict_fpr95_superiority"
                ]["passed"]
            )
            q05 = comparison["headline_acceptance"]["gates"][
                "positive_q05_noninferiority"
            ]
            self.assertFalse(q05["passed"])
            self.assertAlmostEqual(
                q05["splits"]["strict2031"]["observed_delta"], -0.1
            )
            self.assertFalse(comparison["headline_acceptance"]["pass"])

    def test_headline_acceptance_uses_predeclared_strict_ci_margins(self):
        headline = {
            "ref": {
                "mean8_acc50": {
                    "observed_candidate_minus_baseline": 0.02,
                    "delta_ci_low": 0.001,
                },
                "splits": {
                    split: {
                        "observed_candidate_minus_baseline": -0.005,
                        "delta_ci_low": -0.009,
                    }
                    for split in REF_SPLITS
                },
            },
            "tn": {
                split: {
                    "fpr95": {
                        "observed_candidate_minus_baseline_fpr95": -0.03,
                        "delta_ci_high": -0.001,
                    },
                    "positive_q05_threshold": {
                        "observed_candidate_minus_baseline": -0.01,
                        "delta_ci_low": -0.019,
                    },
                }
                for split in ("strict2031", "strict1607")
            },
        }
        accepted = _headline_acceptance(
            headline, provenance={"passed": True, "unmet": []}
        )
        self.assertTrue(accepted["pass"])

        headline["ref"]["splits"]["refcoco_val"]["delta_ci_low"] = -0.01
        rejected = _headline_acceptance(
            headline, provenance={"passed": True, "unmet": []}
        )
        self.assertFalse(rejected["pass"])
        self.assertFalse(
            rejected["gates"]["ref_split_noninferiority"]["splits"][
                "refcoco_val"
            ]["passed"]
        )

    def test_unequal_training_seed_distributions_fail_without_fixed_reference_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            baseline = payload["experiments"][0]
            baseline["runs"] = baseline["runs"][:1]
            baseline["expected_train_seeds"] = [11]
            payload["experiments"][1]["expected_train_seeds"] = [11, 22]
            _write_json(manifest, payload)

            with self.assertRaisesRegex(
                PaperAggregationError,
                "training-seed distributions require identical seed sets",
            ):
                aggregate_manifest(manifest)

    def test_identity_hash_metric_and_split_drift_fail_closed(self):
        for mutation in ("identity", "hash", "metric", "split"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self._fixture(root)
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                candidate_run = payload["experiments"][1]["runs"][0]
                if mutation == "identity":
                    record_spec = candidate_run["results"]["ref"]["records"]["refcoco_val"]
                    record_path = Path(record_spec["path"])
                    rows = [json.loads(line) for line in record_path.read_text().splitlines()]
                    rows[0]["sample_id"] = "drifted-sample"
                    _write_jsonl(record_path, rows)
                    record_spec["sha256"] = _sha256(record_path)
                    ref_summary = Path(candidate_run["results"]["ref"]["summary"]["path"])
                    candidate_run["results"]["ref"]["summary"]["sha256"] = _sha256(ref_summary)
                    _write_json(manifest, payload)
                    expected = "record identity/order mismatch"
                else:
                    if mutation == "hash":
                        config_path = Path(candidate_run["artifacts"]["config"]["path"])
                        config_path.write_text("seed = 999\n", encoding="utf-8")
                        expected = "SHA-256 mismatch"
                    elif mutation == "metric":
                        summary_spec = candidate_run["results"]["ref"]["summary"]
                        summary_path = Path(summary_spec["path"])
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        summary["refcoco"][0]["acc50"] = 0.0
                        _write_json(summary_path, summary)
                        summary_spec["sha256"] = _sha256(summary_path)
                        _write_json(manifest, payload)
                        expected = "summary acc50"
                    else:
                        candidate_run["results"]["ref"]["records"].pop(
                            "refcoco_val"
                        )
                        _write_json(manifest, payload)
                        expected = "missing records for required split"
                with self.assertRaisesRegex(PaperAggregationError, expected):
                    aggregate_manifest(manifest)

    def test_ref_record_score_run_and_duplicate_identity_replay_fail_closed(self):
        for mutation, expected in (
            ("score", "correct50 does not replay from top1_iou"),
            ("run_id", "run_id mismatch"),
            ("duplicate", "duplicate record identity"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self._fixture(root)
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                run = payload["experiments"][1]["runs"][0]
                artifact = run["results"]["ref"]["records"]["refcoco_val"]
                path = Path(artifact["path"])
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                if mutation == "score":
                    rows[0]["top1_iou"] = 0.1
                elif mutation == "run_id":
                    rows[0]["run_id"] = "wrong_run"
                else:
                    for field in (
                        "sample_id",
                        "image_id",
                        "ann_id",
                        "ref_id",
                        "sent_id",
                    ):
                        rows[1][field] = rows[0][field]
                _write_jsonl(path, rows)
                artifact["sha256"] = _sha256(path)
                _write_json(manifest, payload)
                with self.assertRaisesRegex(PaperAggregationError, expected):
                    aggregate_manifest(manifest)

    def test_record_mutation_after_identity_hash_fails_toctou_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            original = eval_records._read_formal_ref_jsonl
            mutated = False

            def read_then_mutate(path: Path, *, label: str):
                nonlocal mutated
                rows = original(path, label=label)
                if not mutated and path.name == "refcoco_val.records.jsonl":
                    path.write_text(
                        path.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8",
                    )
                    mutated = True
                return rows

            with patch.object(
                eval_records,
                "_read_formal_ref_jsonl",
                side_effect=read_then_mutate,
            ):
                with self.assertRaisesRegex(
                    PaperAggregationError,
                    "changed between identity verification and parsing",
                ):
                    aggregate_manifest(manifest)

    def test_tn_secondary_summary_metric_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            run = payload["experiments"][1]["runs"][0]
            summary_spec = run["results"]["tn"]["strict2031"]["summary"]
            summary_path = Path(summary_spec["path"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["tn"][0]["fpr90tpr"] = 0.5
            _write_json(summary_path, summary)
            summary_spec["sha256"] = _sha256(summary_path)
            _write_json(manifest, payload)

            with self.assertRaisesRegex(PaperAggregationError, "summary fpr90tpr"):
                aggregate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
