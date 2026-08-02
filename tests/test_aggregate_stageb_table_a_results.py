import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import aggregate_stageb_table_a_results as aggregate
from tools import run_stageb_table_a_g0c_queues as queues


class TableAThreeSeedAggregateTest(unittest.TestCase):
    def test_mean_and_ddof1_are_exact(self):
        values = {
            17: {"ref/x/G4/acc50": 1.0},
            42: {"ref/x/G4/acc50": 2.0},
            73: {"ref/x/G4/acc50": 3.0},
        }
        result = aggregate.aggregate_seed_metrics(values)["ref/x/G4/acc50"]
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["std_ddof1"], 1.0)
        self.assertEqual(result["display"], "2.000000 +/- 1.000000")

    def test_seed_or_metric_surface_drift_fails_closed(self):
        with self.assertRaisesRegex(aggregate.TableAAggregationError, "exact seeds"):
            aggregate.aggregate_seed_metrics({17: {"a": 1.0}})
        with self.assertRaisesRegex(aggregate.TableAAggregationError, "surfaces"):
            aggregate.aggregate_seed_metrics(
                {17: {"a": 1.0}, 42: {"a": 2.0}, 73: {"b": 3.0}}
            )

    def test_paired_image_cluster_bootstrap_is_fixed_and_deterministic(self):
        rows = [
            {"cluster_id": "image1", "delta": 1.0},
            {"cluster_id": "image1", "delta": 1.0},
            {"cluster_id": "image2", "delta": -1.0},
        ]
        first = aggregate.paired_image_cluster_bootstrap(rows)
        second = aggregate.paired_image_cluster_bootstrap(rows)
        self.assertEqual(first, second)
        self.assertEqual(first["iterations"], 5000)
        self.assertEqual(first["num_clusters"], 2)
        self.assertEqual(first["num_paired_expressions"], 3)
        self.assertAlmostEqual(first["point_delta_acc50_g4_minus_g0c"], 1.0 / 3.0)
        with self.assertRaisesRegex(aggregate.TableAAggregationError, "5000"):
            aggregate.paired_image_cluster_bootstrap(rows, iterations=10)

    def test_paired_records_require_exact_sample_and_image_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "g0c.jsonl"
            baseline = {
                "sample_id": "s1",
                "image_id": 9,
                "top1_iou": 0.2,
            }
            path.write_text(json.dumps(baseline) + "\n", encoding="utf-8")
            candidate = {
                "sample_id": "s1",
                "image_id": 9,
                "routes": {
                    "patch_admission_text_rank": {"selected_iou": 0.8}
                },
            }
            paired = aggregate._paired_rows(
                seed=17,
                split="refcoco_val",
                candidate_rows=[candidate],
                g0c_path=path,
            )
            self.assertEqual(
                paired,
                [{"cluster_id": "image9", "training_seed": 17, "delta": 1.0}],
            )
            candidate["image_id"] = 10
            with self.assertRaisesRegex(
                aggregate.TableAAggregationError, "image identity"
            ):
                aggregate._paired_rows(
                    seed=17,
                    split="refcoco_val",
                    candidate_rows=[candidate],
                    g0c_path=path,
                )

    def test_instance_loader_replays_postflight_at_canonical_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = {
                "status": "completed",
                "kind": "candidate",
                "profile": "validation",
                "instance": {
                    "seed": 17,
                    "instance_id": "i",
                    "instance_sha256": "a" * 64,
                    "checkpoint_sha256": "b" * 64,
                    "training_queue_id": "q",
                    "training_queue_plan_sha256": "c" * 64,
                },
            }
            postflight = {"status": "passed", "verified_at_utc": "old"}
            (root / "launch_manifest.json").write_text(
                json.dumps(launch), encoding="utf-8"
            )
            (root / "postflight.json").write_text(
                json.dumps(postflight), encoding="utf-8"
            )
            with (
                mock.patch.object(
                    aggregate.table_a, "canonical_output_dir", return_value=root
                ),
                mock.patch.object(
                    aggregate.table_a,
                    "postflight",
                    return_value={"status": "passed", "verified_at_utc": "new"},
                ) as replay,
            ):
                result = aggregate._load_instance(
                    "candidate", "validation", 17
                )
            replay.assert_called_once_with(launch)
            self.assertEqual(result["root"], root.resolve())
            self.assertEqual(result["instance"]["instance_id"], "i")

    def test_g0c_summary_keeps_both_strict_tn_metric_surfaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "ref8_strict2031"
            supplemental = root / "strict1607"
            primary.mkdir()
            supplemental.mkdir()
            records = primary / "ref.jsonl"
            records.write_text("{}\n", encoding="utf-8")
            ref = {
                "dataset": "refcoco_val",
                "records_jsonl": str(records),
                **{
                    key: 0.5
                    for key in (
                        "acc50",
                        "acc50@5",
                        "acc50@10",
                        "acc50@50",
                        "recall50@all_queries",
                        "mean_iou",
                        "mean_iou@5",
                        "mean_iou@10",
                        "mean_iou@50",
                        "mean_best_iou@all_queries",
                    )
                },
            }
            tn = {
                key: 0.25
                for key in (
                    "fpr95tpr",
                    "fpr90tpr",
                    "pair_win_rate",
                    "pair_tie_rate",
                    "pos_score_mean",
                    "tn_score_mean",
                    "score_gap_mean",
                    "threshold_at_95tpr",
                    "actual_tpr_at_95tpr",
                )
            }
            tn["pair_tie_rate"] = 0.75
            (primary / "summary.json").write_text(
                json.dumps({"refcoco": [ref], "tn": [tn]}), encoding="utf-8"
            )
            (supplemental / "summary.json").write_text(
                json.dumps({"refcoco": [], "tn": [tn]}), encoding="utf-8"
            )
            replay = {
                label: {
                    key: 0.25
                    for key in aggregate.table_a.G0C_TN_AGGREGATE_METRICS
                }
                for label in ("strict2031", "strict1607")
            }
            metrics, _ = aggregate._g0c_summary(
                {
                    "root": root,
                    "postflight": {
                        "artifacts": {"tn_metrics_recomputed": replay}
                    },
                },
                aggregate.table_a.FINAL_PROFILE,
            )
            self.assertIn("tn/strict2031/G0c/fpr95tpr", metrics)
            self.assertIn("tn/strict1607/G0c/fpr95tpr", metrics)
            self.assertEqual(
                metrics["tn/strict2031/G0c/pair_tie_rate"], 0.25
            )

    def test_final_queue_binding_requires_exact_consumption_backed_six_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue_dir = Path(temporary) / "final-queue"
            queue_dir.mkdir()
            (queue_dir / "queue.json").write_text("{}\n", encoding="ascii")
            verified_items = []
            for index, run_id in enumerate(queues.FINAL_RUN_IDS):
                kind, seed_text = run_id.split(":")
                record = {
                    "path": f"/artifact/{index}",
                    "sha256": f"{index + 1:064x}",
                    "size_bytes": index + 1,
                }
                verified_items.append(
                    {
                        "run_id": run_id,
                        "native_completion": {
                            "queue_kind": queues.FINAL_KIND,
                            "run_id": run_id,
                            "evaluation_kind": kind,
                            "evaluation_profile": aggregate.table_a.FINAL_PROFILE,
                            "seed": int(seed_text),
                            "instance_sha256": f"{index + 11:064x}",
                            "launch_manifest": record,
                            "postflight": record,
                            "final_gate": record,
                            "final_consumption": record,
                        },
                    }
                )
            verification = {
                "status": "passed",
                "queue_status": "completed",
                "queue_kind": queues.FINAL_KIND,
                "queue_id": "final-queue-id",
                "plan_sha256": "a" * 64,
                "ordered_run_ids": list(queues.FINAL_RUN_IDS),
                "verified_items": verified_items,
            }
            queue = {
                "status": "completed",
                "plan_sha256": "a" * 64,
                "plan": {"queue_id": "final-queue-id"},
            }
            with (
                mock.patch.object(queues, "DEFAULT_FINAL_QUEUE_DIR", queue_dir),
                mock.patch.object(
                    queues, "verify_queue", return_value=verification
                ),
                mock.patch.object(queues, "load_queue", return_value=queue),
            ):
                binding = aggregate._final_queue_binding()
                self.assertEqual(binding["ordered_run_ids"], list(queues.FINAL_RUN_IDS))
                self.assertEqual(len(binding["items"]), 6)
                self.assertTrue(binding["single_use_consumptions_verified"])

                broken = json.loads(json.dumps(verification))
                del broken["verified_items"][0]["native_completion"][
                    "final_consumption"
                ]
                with (
                    mock.patch.object(queues, "verify_queue", return_value=broken),
                    self.assertRaisesRegex(
                        aggregate.TableAAggregationError,
                        "completion evidence is incomplete",
                    ),
                ):
                    aggregate._final_queue_binding()

    def test_final_report_binds_same_queue_before_and_after_metric_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.json"
            records = root / "records.jsonl"
            summary.write_text("{}\n", encoding="ascii")
            records.write_text("{}\n", encoding="ascii")
            binding = {
                "schema": aggregate.FINAL_QUEUE_BINDING_SCHEMA,
                "queue_id": "queue",
                "binding_sha256": "a" * 64,
            }

            def instance(kind, _profile, seed):
                return {
                    "kind": kind,
                    "seed": seed,
                    "root": root,
                    "postflight": {},
                    "instance": {
                        "instance_sha256": f"{kind}:{seed}".encode("ascii").hex().ljust(64, "0")
                    },
                    "provenance": {"root": str(root)},
                }

            with (
                mock.patch.object(
                    aggregate,
                    "_final_queue_binding",
                    side_effect=[binding, binding],
                ) as queue_binding,
                mock.patch.object(aggregate, "_load_instance", side_effect=instance),
                mock.patch.object(
                    aggregate,
                    "_candidate_summary_and_records",
                    return_value=({}, {}, summary, records),
                ),
                mock.patch.object(
                    aggregate,
                    "_candidate_metrics",
                    return_value={"candidate/metric": 0.5},
                ),
                mock.patch.object(
                    aggregate,
                    "_g0c_summary",
                    return_value=({"g0c/metric": 0.4}, {}),
                ),
            ):
                report = aggregate.build_report(aggregate.table_a.FINAL_PROFILE)
            self.assertEqual(queue_binding.call_count, 2)
            self.assertEqual(report["provenance"]["final_queue"], binding)
            self.assertTrue(
                report["provenance"]["no_rerun_contract"][
                    "canonical_final_queue_required"
                ]
            )


if __name__ == "__main__":
    unittest.main()
