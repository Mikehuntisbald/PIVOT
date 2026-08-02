import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.aggregate_stageb_table_b_matched_panel import (
    DECLARED_SURFACE,
    METRIC_LABEL,
    MatchedPanelReportError,
    _parse_seed_record,
    aggregate_formal_matched_panel,
    aggregate_matched_panel,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, rows) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "rows": len(rows),
        "unique_images": len({int(row["image_id"]) for row in rows}),
    }


def _plain_artifact(path: Path, *, rows=None) -> dict:
    result = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _parent(index: int) -> dict:
    return {
        "dataset": "refcocog",
        "image_id": 100 + index,
        "sent_id": 200 + index,
        "edit_category": "color" if index < 2 else "size",
        "positive_phrase_normalized": f"positive phrase {index}",
    }


def _parent_hash(parent: dict) -> str:
    raw = json.dumps(
        parent, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _fixture_rows():
    ledger = []
    d2_rows = []
    d3_rows = []
    relations = ("identical", "identical", "different", "different")
    class_matches = (True, False, True, False)
    for index, (relation, class_match) in enumerate(
        zip(relations, class_matches)
    ):
        parent = _parent(index)
        parent_sha = _parent_hash(parent)
        pair_id = f"c2-parent:calibration:{parent_sha[:24]}"
        d2_class = 7
        d3_class = 7 if class_match else 8
        negative_exact = relation == "identical"
        components = {
            "image_id": True,
            "file_name": True,
            "target_bbox_used": True,
            "positive_phrase": True,
            "negative_text": negative_exact,
            "canonical_class_id": class_match,
        }
        complete = all(components.values())
        if complete:
            causal = "class_aligned_identical_complete_input"
        elif negative_exact:
            causal = "identical_text_class_mismatch"
        elif class_match:
            causal = "different_text_class_aligned"
        else:
            causal = "different_text_class_mismatch"
        stratum = {
            "dataset": "refcocog",
            "edit_category": parent["edit_category"],
            "negative_text_relation": relation,
            "canonical_class_relation": "aligned" if class_match else "mismatched",
            "causal_input_relation": causal,
        }
        shared = {
            "matched_pair_schema": "stage-b-paper-c2-parent-matched-pair-v2",
            "matched_pair_id": pair_id,
            "matched_split": "calibration",
            "matched_parent_key": parent,
            "matched_parent_key_sha256": parent_sha,
            "matched_stratum": stratum,
            "matched_selection": {
                "seed": "fixture",
                "method": "minimum sha256 fixture",
                "candidate_count": 1,
                "selected_candidate_rank": 0,
                "selected_d2_source_line": index + 1,
                "selected_d2_source_row_sha256": hashlib.sha256(
                    f"d2-{index}".encode()
                ).hexdigest(),
            },
            "positive_phrase_exact_match": True,
            "positive_phrase_normalized_match": True,
            "negative_text_exact_match": negative_exact,
            "negative_text_normalized_match": negative_exact,
            "d2_canonical_class_id": d2_class,
            "d3_canonical_class_id": d3_class,
            "canonical_class_id_match": class_match,
            "model_input_component_exact_matches": components,
            "base_parent_input_exact_match": True,
            "complete_model_input_exact_match": complete,
            "class_aligned_identical_complete_input": complete,
        }
        d2_sample = f"d2:{index}"
        d3_sample = f"d3:{index}"
        d2_negative = f"negative {index}" if relation == "different" else f"same {index}"
        d3_negative = f"other {index}" if relation == "different" else f"same {index}"
        ledger.append(
            {
                **shared,
                "image_id": parent["image_id"],
                "dataset": parent["dataset"],
                "sent_id": parent["sent_id"],
                "edit_category": parent["edit_category"],
                "d2m": {
                    "sample_id": d2_sample,
                    "class_id": d2_class,
                    "file_name": f"image-{index}.jpg",
                    "target_bbox_used": [1, 2, 3, 4],
                    "sent": f"positive phrase {index}",
                    "try_tn": d2_negative,
                    "tn_scope": "traceable_counterfactual_edit",
                    "source_line": index + 1,
                    "source_row_sha256": hashlib.sha256(
                        f"d2-{index}".encode()
                    ).hexdigest(),
                },
                "d3m": {
                    "sample_id": d3_sample,
                    "class_id": d3_class,
                    "file_name": f"image-{index}.jpg",
                    "target_bbox_used": [1, 2, 3, 4],
                    "sent": f"positive phrase {index}",
                    "try_tn": d3_negative,
                    "tn_scope": DECLARED_SURFACE,
                    "source_line": index + 1,
                    "source_row_sha256": hashlib.sha256(
                        f"d3-{index}".encode()
                    ).hexdigest(),
                },
            }
        )
        base = {
            **shared,
            "image_id": parent["image_id"],
            "ann_id": 300 + index,
            "ref_id": 400 + index,
            "sent_id": parent["sent_id"],
            "split": "train",
            "file_name": f"image-{index}.jpg",
            "target_bbox_used": [1, 2, 3, 4],
            "sent": f"positive phrase {index}",
        }
        d2_rows.append(
            {
                **base,
                "sample_id": d2_sample,
                "class_id": d2_class,
                "try_tn": d2_negative,
                "table_b_id": "D2m",
                "tn_scope": "traceable_counterfactual_edit",
                "global_tn_verified": False,
                "traceable_counterfactual_edit": True,
                "proposal_covered_verified": False,
            }
        )
        d3_rows.append(
            {
                **base,
                "sample_id": d3_sample,
                "class_id": d3_class,
                "try_tn": d3_negative,
                "table_b_id": "D3m",
                "tn_scope": DECLARED_SURFACE,
                "global_tn_verified": False,
                "traceable_counterfactual_edit": True,
                "proposal_covered_verified": True,
            }
        )
    return ledger, d2_rows, d3_rows


def _records(source, *, run_id, positive, negative):
    manifest_hash = None
    rows = []
    for index, row in enumerate(source):
        rows.append(
            {
                "schema": "stageb-eval-record-v1",
                "task": "tn",
                "manifest_key": "tn_global",
                "manifest_sha256": manifest_hash,
                "manifest_n": len(source),
                "manifest_index": index,
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "ann_id": row["ann_id"],
                "ref_id": row["ref_id"],
                "sent_id": row["sent_id"],
                "split": row["split"],
                "run_id": run_id,
                "valid": True,
                "eval_scope": DECLARED_SURFACE,
                "support_input_kind": "patches",
                "support_input_sha256": hashlib.sha256(
                    f"support-input:{index}".encode()
                ).hexdigest(),
                "support_class_ids": [int(row["class_id"])],
                "pos_score": positive[index],
                "neg_score": negative[index],
            }
        )
    return rows


class StageBTableBMatchedPanelTest(unittest.TestCase):
    def _fixture(self, root: Path):
        ledger, d2_rows, d3_rows = _fixture_rows()
        ledger_path = root / "matched_pairs_calibration.jsonl"
        d2_path = root / "d2m_calibration.jsonl"
        d3_path = root / "d3m_calibration.jsonl"
        _write_jsonl(ledger_path, ledger)
        _write_jsonl(d2_path, d2_rows)
        _write_jsonl(d3_path, d3_rows)

        stats = {
            "pairs": 4,
            "unique_images": 4,
            "dataset_pairs": {"refcocog": 4},
            "edit_category_pairs": {"color": 2, "size": 2},
            "negative_text_relation_pairs": {"different": 2, "identical": 2},
            "negative_text_exact_relation_pairs": {
                "different": 2,
                "identical": 2,
            },
            "positive_text_exact_relation_pairs": {"identical": 4},
            "canonical_class_relation_pairs": {"aligned": 2, "mismatched": 2},
            "canonical_class_id_mismatch_pairs": 2,
            "identical_negative_text_class_id_mismatch_pairs": 1,
            "canonical_class_id_mismatch_direction_pairs": {"7->8": 2},
            "causal_input_relation_pairs": {
                "class_aligned_identical_complete_input": 1,
                "different_text_class_aligned": 1,
                "different_text_class_mismatch": 1,
                "identical_text_class_mismatch": 1,
            },
            "class_aligned_identical_complete_input_pairs": 1,
            "model_input_component_mismatch_pairs": {
                "canonical_class_id": 2,
                "negative_text": 2,
            },
            "d2_candidate_count_histogram": {"1": 4},
        }
        input_dir = root / "audit-inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_records = {}
        for key, row_count in (
            ("D2_raw", 4),
            ("D3_single_edit_train", 6),
            ("D3_single_edit_calibration", 6),
            ("D3_partition_audit", None),
            ("strict2031", 1),
            ("strict1607", 1),
        ):
            path = input_dir / f"{key}.json"
            _write_json(path, {"key": key})
            input_records[key] = _plain_artifact(path, rows=row_count)
        config_dir = root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        dataset_configs = {}
        for table_id in ("D2m", "D3m"):
            path = config_dir / f"{table_id}.json"
            _write_json(path, {"table_b_id": table_id})
            dataset_configs[table_id] = _plain_artifact(path)

        matching_yield = {
            "d3_parent_rows": 6,
            "matched_pairs": 4,
            "unmatched_d3_parent_rows": 2,
            "matched_fraction": 4 / 6,
            "unmatched_fraction": 2 / 6,
            "matched_pairwise_claim_denominator": 4,
            "class_aligned_identical_claim_denominator": 1,
            "unmatched_d3_parent_rows_excluded_from_pairwise_claims": 2,
            "parent_row_partition_is_exact": True,
        }
        split_invariants = {
            "equal_rows": True,
            "aligned_pair_ids": True,
            "aligned_parent_key_sha256": True,
            "positive_phrase_normalized_match": True,
            "unique_pair_ids": True,
            "unique_parent_keys": True,
            "negative_text_relation_is_partition": True,
            "canonical_class_relation_is_partition": True,
            "causal_input_relation_is_partition": True,
            "class_mismatch_count_is_audited": True,
            "class_aligned_identical_stratum_is_exact": True,
            "class_aligned_identical_boolean_is_exact": True,
            "runtime_global_verified_true_rows": 0,
        }
        audit = {
            "schema": "stage-b-paper-c2-parent-matched-tn-v2",
            "kind": "completed_c2_parent_matched_tn_panel",
            "seed": "fixture",
            "inputs": input_records,
            "statistics": {"train": copy.deepcopy(stats), "calibration": stats},
            "matching_yield": {
                "train": copy.deepcopy(matching_yield),
                "calibration": matching_yield,
            },
            "claim_scope": {
                "pairwise_effect_population": "matched_pairs_only",
                "primary_causal_stratum": "class_aligned_identical_complete_input",
                "primary_causal_stratum_requires_exact_model_input": True,
                "canonical_class_id_equality_required": True,
                "unmatched_d3_parent_rows_are_out_of_scope": True,
                "generalization_to_unmatched_d3_parent_rows_supported": False,
            },
            "outputs": {
                "pairs_calibration": _artifact(ledger_path, ledger),
                "d2m_calibration": _artifact(d2_path, d2_rows),
                "d3m_calibration": _artifact(d3_path, d3_rows),
                "pairs_train": _artifact(ledger_path, ledger),
                "d2m_train": _artifact(d2_path, d2_rows),
                "d3m_train": _artifact(d3_path, d3_rows),
            },
            "dataset_configs": dataset_configs,
            "scope_contract": {
                "D2m": {
                    "tn_scope": "traceable_counterfactual_edit",
                    "global_tn_verified": False,
                },
                "D3m": {
                    "tn_scope": DECLARED_SURFACE,
                    "global_tn_verified": False,
                },
            },
            "invariants": {
                "train": copy.deepcopy(split_invariants),
                "calibration": split_invariants,
                "strict_union_image_overlap": 0,
                "train_calibration_image_overlap": 0,
                "train_calibration_pair_id_overlap": 0,
                "unique_train_parent_keys": True,
                "unique_calibration_parent_keys": True,
                "train_matching_yield_partition": True,
                "calibration_matching_yield_partition": True,
            },
            "runtime_contract": {
                "D2m_D3m_supported_by_current_v24": True,
            },
        }
        audit_path = root / "audit.json"
        _write_json(audit_path, audit)
        audit_sha = _sha256(audit_path)

        positives = [0.9, 0.8, 0.7, 0.6]
        paths = {"D2m": {}, "D3m": {}}
        for seed in (11, 22):
            d2_records = _records(
                d3_rows,
                run_id=f"d2m-seed-{seed}",
                positive=[value + seed * 0.0001 for value in positives],
                negative=[0.1, 0.3, 0.65, 0.7],
            )
            d3_records = _records(
                d3_rows,
                run_id=f"d3m-seed-{seed}",
                positive=[value + 0.01 + seed * 0.0001 for value in positives],
                negative=[0.05, 0.2, 0.4, 0.5],
            )
            for condition, rows, training_source in (
                ("D2m", d2_records, d2_path),
                ("D3m", d3_records, d3_path),
            ):
                for row in rows:
                    row["manifest_sha256"] = _sha256(d3_path)
                    row.update(
                        {
                            "provenance_schema": (
                                "stage-b-table-b-matched-eval-provenance-v1"
                            ),
                            "table_b_id": condition,
                            "train_seed": seed,
                            "checkpoint_sha256": hashlib.sha256(
                                f"{condition}:{seed}:checkpoint".encode()
                            ).hexdigest(),
                            "training_source_sha256": _sha256(training_source),
                            "training_source_n": 4,
                            "matched_panel_audit_sha256": audit_sha,
                            "evaluation_source_sha256": _sha256(d3_path),
                            "evaluation_source_n": 4,
                            "evaluation_manifest_sha256": _sha256(d3_path),
                            "declared_evaluation_surface": DECLARED_SURFACE,
                        }
                    )
            d2_record_path = root / "records" / f"d2m-{seed}.jsonl"
            d3_record_path = root / "records" / f"d3m-{seed}.jsonl"
            _write_jsonl(d2_record_path, d2_records)
            _write_jsonl(d3_record_path, d3_records)
            paths["D2m"][seed] = d2_record_path
            paths["D3m"][seed] = d3_record_path
        return {
            "audit_path": audit_path,
            "pair_ledger_path": ledger_path,
            "d2m_source_path": d2_path,
            "d3m_source_path": d3_path,
            "evaluation_manifest_path": d3_path,
            "d2m_records": paths["D2m"],
            "d3m_records": paths["D3m"],
            "expected_seeds": [11, 22],
        }

    def _reseal_output(self, fixture, key, path):
        audit = json.loads(fixture["audit_path"].read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        updated = _artifact(path, rows)
        sealed_path = str(path.resolve())
        for output_key, record in audit["outputs"].items():
            if str(Path(record["path"]).resolve()) == sealed_path:
                audit["outputs"][output_key] = copy.deepcopy(updated)
        audit["outputs"][key] = copy.deepcopy(updated)
        _write_json(fixture["audit_path"], audit)

    def test_reports_scoped_strata_provenance_and_paired_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            report = aggregate_matched_panel(**fixture)

            self.assertEqual(
                report["status"], "validated_internal_records_diagnostic"
            )
            self.assertFalse(report["formal_global_fpr_eligible"])
            self.assertEqual(report["metric_label"], METRIC_LABEL)
            self.assertEqual(report["validation"]["exact_seed_set"], [11, 22])
            accounting = report["validation"]["canonical_class_accounting"]
            self.assertEqual(accounting["mismatch_n"], 2)
            self.assertEqual(
                accounting["mismatch_by_negative_text_relation"],
                {"identical": 1, "different": 1},
            )
            population = report["validation"]["population_accounting"]
            self.assertEqual(population["matched_pairs"], 4)
            self.assertEqual(population["unmatched_d3_parent_rows"], 2)
            seed = report["per_seed"]["11"]
            self.assertEqual(
                set(seed["strata"]),
                {
                    "identical",
                    "different",
                    "class_aligned_identical_complete_input",
                },
            )
            self.assertAlmostEqual(
                seed["strata"]["identical"]["D2m"]["positive_q05"],
                0.8011,
            )
            self.assertAlmostEqual(
                seed["strata"]["identical"]["D3m_minus_D2m"][
                    "positive_q05"
                ],
                0.01,
            )
            self.assertAlmostEqual(
                seed["strata"]["different"]["D3m_minus_D2m"][
                    "positive_over_negative_win_rate"
                ],
                0.5,
            )
            for relation in ("identical", "different"):
                stratum = seed["strata"][relation]
                self.assertEqual(stratum["n"], 2)
                self.assertEqual(stratum["unique_images"], 2)
                self.assertEqual(stratum["canonical_class_mismatch_n"], 1)
                self.assertFalse(stratum["D2m"]["formal_global_fpr_eligible"])
                self.assertIn("positive_q05", stratum["D2m"])
                self.assertIn("auroc", stratum["D2m"])
                self.assertIn("fpr95_like_diagnostic", stratum["D2m"])
                delta = stratum["D3m_minus_D2m"]
                self.assertEqual(delta["direction"], "D3m_minus_D2m")
                self.assertTrue(delta["paired_on_exact_manifest_rows"])
                for row in stratum["paired_records"]:
                    self.assertIn("matched_pair_id", row)
                    self.assertIn("matched_parent_key_sha256", row)
                    self.assertEqual(row["negative_text_relation"], relation)
                    self.assertIn("canonical_class_match", row)
            primary = seed["strata"]["class_aligned_identical_complete_input"]
            self.assertEqual(primary["stratum_kind"], "primary_clean_causal_input")
            self.assertEqual(primary["n"], 1)
            self.assertEqual(primary["canonical_class_mismatch_n"], 0)
            self.assertEqual(primary["complete_model_input_exact_match_n"], 1)
            self.assertIn(
                "d3m_larger_score_gap_rate",
                primary["D3m_minus_D2m"]["paired_model_comparison_rates"],
            )
            self.assertNotIn("pair_win_rate", primary["D2m"])
            self.assertEqual(
                report["inputs"]["record_files"]["D2m"]["11"]["provenance"][
                    "table_b_id"
                ],
                "D2m",
            )
            self.assertEqual(
                report["across_seed_summary"]["identical"]["seed_set"],
                [11, 22],
            )

    def test_exact_seed_sets_and_seed_artifact_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            missing = dict(fixture)
            missing["d3m_records"] = {11: fixture["d3m_records"][11]}
            with self.assertRaisesRegex(MatchedPanelReportError, "exact expected"):
                aggregate_matched_panel(**missing)

            duplicated = dict(fixture)
            duplicated["expected_seeds"] = [11, 11]
            with self.assertRaisesRegex(MatchedPanelReportError, "duplicates"):
                aggregate_matched_panel(**duplicated)

            with self.assertRaisesRegex(MatchedPanelReportError, "duplicate seed"):
                _parse_seed_record(["11=a.jsonl", "11=b.jsonl"], label="records")

    def test_condition_swap_seed_relabel_and_missing_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            swapped = dict(fixture)
            swapped["d2m_records"] = fixture["d3m_records"]
            swapped["d3m_records"] = fixture["d2m_records"]
            with self.assertRaisesRegex(MatchedPanelReportError, "provenance table_b_id"):
                aggregate_matched_panel(**swapped)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            relabeled = dict(fixture)
            relabeled["d2m_records"] = {
                11: fixture["d2m_records"][22],
                22: fixture["d2m_records"][11],
            }
            with self.assertRaisesRegex(MatchedPanelReportError, "provenance train_seed"):
                aggregate_matched_panel(**relabeled)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["d2m_records"][11]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                del row["checkpoint_sha256"]
            _write_jsonl(path, rows)
            with self.assertRaisesRegex(MatchedPanelReportError, "checkpoint_sha256"):
                aggregate_matched_panel(**fixture)

    def test_reordered_record_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["d2m_records"][11]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0], rows[1] = rows[1], rows[0]
            _write_jsonl(path, rows)
            with self.assertRaisesRegex(
                MatchedPanelReportError, "record/manifest binding failed"
            ):
                aggregate_matched_panel(**fixture)

    def test_missing_and_duplicate_record_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["d2m_records"][11]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            _write_jsonl(path, rows[:-1])
            with self.assertRaisesRegex(
                MatchedPanelReportError, "record/manifest binding failed"
            ):
                aggregate_matched_panel(**fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["d2m_records"][11]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["sample_id"] = rows[0]["sample_id"]
            _write_jsonl(path, rows)
            with self.assertRaisesRegex(
                MatchedPanelReportError, "record/manifest binding failed"
            ):
                aggregate_matched_panel(**fixture)

    def test_scope_upgrade_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["d3m_records"][11]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["eval_scope"] = "image_global_topk_verified"
            rows[0]["formal_global_fpr_eligible"] = True
            _write_jsonl(path, rows)
            with self.assertRaisesRegex(MatchedPanelReportError, "declared surface|scope upgrade"):
                aggregate_matched_panel(**fixture)

    def test_support_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["d3m_records"][11]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["support_input_sha256"] = "f" * 64
            _write_jsonl(path, rows)
            with self.assertRaisesRegex(
                MatchedPanelReportError, "identical support inputs"
            ):
                aggregate_matched_panel(**fixture)

    def test_formal_entrypoint_rejects_self_reported_record_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            record_paths = list(fixture["d2m_records"].values()) + list(
                fixture["d3m_records"].values()
            )
            with self.assertRaisesRegex(
                MatchedPanelReportError, "evaluation replay failed"
            ):
                aggregate_formal_matched_panel(
                    audit_path=fixture["audit_path"],
                    pair_ledger_path=fixture["pair_ledger_path"],
                    d2m_source_path=fixture["d2m_source_path"],
                    d3m_source_path=fixture["d3m_source_path"],
                    evaluation_manifest_path=fixture["evaluation_manifest_path"],
                    evaluation_outputs={
                        17: record_paths[0],
                        42: record_paths[1],
                        73: record_paths[2],
                    },
                )

    def test_arbitrary_causal_label_fails_component_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            for key, fixture_key in (
                ("pairs_calibration", "pair_ledger_path"),
                ("d2m_calibration", "d2m_source_path"),
                ("d3m_calibration", "d3m_source_path"),
            ):
                path = fixture[fixture_key]
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[2]["matched_stratum"]["causal_input_relation"] = (
                    "different_text_class_mismatch"
                )
                _write_jsonl(path, rows)
                self._reseal_output(fixture, key, path)
            audit = json.loads(fixture["audit_path"].read_text())
            for split in ("train", "calibration"):
                counts = audit["statistics"][split]["causal_input_relation_pairs"]
                del counts["different_text_class_aligned"]
                counts["different_text_class_mismatch"] = 2
            _write_json(fixture["audit_path"], audit)
            with self.assertRaisesRegex(
                MatchedPanelReportError,
                "causal_input_relation is not implied by component facts",
            ):
                aggregate_matched_panel(**fixture)

    def test_unmatched_population_denominator_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            audit = json.loads(fixture["audit_path"].read_text())
            audit["matching_yield"]["calibration"][
                "unmatched_d3_parent_rows"
            ] = 3
            _write_json(fixture["audit_path"], audit)
            with self.assertRaisesRegex(
                MatchedPanelReportError, "matched/unmatched claim denominator"
            ):
                aggregate_matched_panel(**fixture)

    def test_audited_source_ledger_identity_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["d2m_source_path"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["matched_pair_id"] = "wrong-pair"
            _write_jsonl(path, rows)
            self._reseal_output(fixture, "d2m_calibration", path)
            with self.assertRaisesRegex(MatchedPanelReportError, "differs from ledger"):
                aggregate_matched_panel(**fixture)

    def test_manifest_identity_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            surface = root / "different-surface.jsonl"
            rows = [
                {**json.loads(line), "surface_nonce": "different"}
                for line in fixture["d3m_source_path"].read_text().splitlines()
            ]
            _write_jsonl(surface, rows)
            fixture["evaluation_manifest_path"] = surface
            with self.assertRaisesRegex(MatchedPanelReportError, "declared evaluation manifest"):
                aggregate_matched_panel(**fixture)

    def test_missing_canonical_mismatch_accounting_is_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            audit = json.loads(fixture["audit_path"].read_text())
            del audit["statistics"]["calibration"][
                "canonical_class_id_mismatch_pairs"
            ]
            _write_json(fixture["audit_path"], audit)
            with self.assertRaisesRegex(
                MatchedPanelReportError,
                "calibration statistics|omit mandatory canonical_class_id_mismatch_pairs",
            ):
                aggregate_matched_panel(**fixture)

    def test_duplicate_pair_id_is_rejected_even_under_resealed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["pair_ledger_path"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["matched_pair_id"] = rows[0]["matched_pair_id"]
            _write_jsonl(path, rows)
            self._reseal_output(fixture, "pairs_calibration", path)
            with self.assertRaisesRegex(MatchedPanelReportError, "duplicate matched_pair_id"):
                aggregate_matched_panel(**fixture)


if __name__ == "__main__":
    unittest.main()
