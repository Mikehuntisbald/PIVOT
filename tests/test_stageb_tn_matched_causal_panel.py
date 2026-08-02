import json
import tempfile
import unittest
from pathlib import Path

from tools.build_stageb_tn_data_ablation_matrix import sha256_file
from tools.build_stageb_tn_matched_causal_panel import (
    CONFIG_NAMES,
    DEFAULT_OUTPUT_DIR,
    LEGACY_SCHEMA,
    OUTPUT_NAMES,
    PRIMARY_CAUSAL_STRATUM,
    SCHEMA,
    MatchedPanelError,
    build_panel,
    verify_panel,
)
from datasets.patch_episode import PatchEpisodeJsonlDataset


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _d2(
    index,
    *,
    image_id,
    sent_id,
    category,
    positive,
    negative,
    dataset,
    class_id=7,
):
    return {
        "image_id": image_id,
        "ann_id": index,
        "ref_id": index,
        "sent_id": sent_id,
        "split": "train",
        "filename": f"COCO_train2014_{image_id:012d}.jpg",
        "source": dataset,
        "instances": [
            {
                "bbox": [1, 2, 30, 40],
                "class_id": class_id,
                "category_name": "object",
                "head": "object",
                "head_phrase": "object",
                "raw_phrase": negative,
                "positive_phrase": positive,
                "try_tn": negative,
                "text_is_negative": True,
                "replace_category": category,
                "replace_from": "old",
                "replace_to": "new",
                "replace_token": "old",
                "try_tn_method": "synthetic_rule",
                "try_tn_rule": "single_token_attribute_swap",
                "pair_source": dataset,
            }
        ],
    }


def _d3(
    index,
    *,
    image_id,
    sent_id,
    category,
    positive,
    negative,
    dataset,
    class_id=7,
):
    edit = {
        "category": category,
        "replace_from": "old",
        "replace_to": "new",
        "replace_span": [0, 1],
    }
    return {
        "sample_id": f"d3:{index}",
        "image_id": image_id,
        "ann_id": index,
        "ref_id": index,
        "sent_id": sent_id,
        "split": "train",
        "file_name": f"COCO_train2014_{image_id:012d}.jpg",
        "dataset": dataset,
        "pair_source": dataset,
        "class_id": class_id,
        "category_name": "object",
        "class_norm_name": "object",
        "target_bbox_used": [1, 2, 30, 40],
        "sent": positive,
        "try_tn": negative,
        "try_tn_head": "object",
        "try_tn_head_phrase": positive,
        "replace_category": [category],
        "replace_from": ["old"],
        "replace_to": ["new"],
        "replace_span": [[0, 1]],
        "tn_edits": [edit],
        "cached_proposal_coverage_only": True,
        "all_900_gdino_queries_verified": False,
        "global_max_label_is_semantic_extrapolation": True,
        "global_tn_verified": True,
        "tn_scope": "image_global_topk_verified",
        "proposal_count": 1,
        "coverage_policy": "all_proposals_all_no",
        "verification_contract": "target_plus_all_cached_proposals_no",
        "source": "semantic-source",
    }


def _write_partition_audit(path, train, calibration):
    payload = {
        "schema": "stage-b-semantic-tn-leakage-isolated-partition-v1",
        "invariants": {
            "eligible_strict_union_image_overlap": 0,
            "train_calibration_image_overlap": 0,
            "single_edit_invalid_metadata_rows_excluded": True,
        },
        "outputs": {
            "single_edit_train": {"sha256": sha256_file(train)},
            "single_edit_calibration": {"sha256": sha256_file(calibration)},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class MatchedCausalPanelTests(unittest.TestCase):
    def _fixture(self, root):
        d2_path = root / "d2.jsonl"
        d3_train = root / "d3_train.jsonl"
        d3_calibration = root / "d3_calibration.jsonl"
        strict2031 = root / "strict2031.jsonl"
        strict1607 = root / "strict1607.jsonl"
        d2_rows = [
            _d2(
                1,
                image_id=10,
                sent_id=100,
                category="color",
                positive="blue object",
                negative="red object",
                dataset="refcoco+_unc_train",
            ),
            # A second candidate for the same parent exercises deterministic selection.
            _d2(
                2,
                image_id=10,
                sent_id=100,
                category="color",
                positive="blue object",
                negative="red object",
                dataset="refcoco+_unc_train",
            ),
            _d2(
                3,
                image_id=11,
                sent_id=101,
                category="size",
                positive="large object",
                negative="small object",
                dataset="refcocog_umd_train",
            ),
            _d2(
                4,
                image_id=12,
                sent_id=102,
                category="spatial",
                positive="left object",
                negative="right object",
                dataset="refcoco+_unc_train",
                class_id=8,
            ),
            _d2(
                5,
                image_id=20,
                sent_id=200,
                category="color",
                positive="white object",
                negative="black object",
                dataset="refcocog_umd_train",
            ),
        ]
        train_rows = [
            _d3(
                1,
                image_id=10,
                sent_id=100,
                category="color",
                positive="blue object",
                negative="red object",
                dataset="refcocoplus",
            ),
            _d3(
                2,
                image_id=11,
                sent_id=101,
                category="size",
                positive="large object",
                negative="tiny object",
                dataset="refcocog",
            ),
            _d3(
                3,
                image_id=12,
                sent_id=102,
                category="spatial",
                positive="left object",
                negative="right object",
                dataset="refcocoplus",
            ),
            # Unmatched semantic category is intentionally excluded.
            _d3(
                4,
                image_id=13,
                sent_id=103,
                category="material",
                positive="wood object",
                negative="metal object",
                dataset="refcocoplus",
            ),
        ]
        calibration_rows = [
            _d3(
                5,
                image_id=20,
                sent_id=200,
                category="color",
                positive="white object",
                negative="black object",
                dataset="refcocog",
            )
        ]
        _write_jsonl(d2_path, d2_rows)
        _write_jsonl(d3_train, train_rows)
        _write_jsonl(d3_calibration, calibration_rows)
        _write_jsonl(strict2031, [{"image_id": 900}])
        _write_jsonl(strict1607, [{"image_id": 901}])
        partition_audit = root / "partition_audit.json"
        _write_partition_audit(partition_audit, d3_train, d3_calibration)
        return {
            "d2_path": d2_path,
            "d3_train_path": d3_train,
            "d3_calibration_path": d3_calibration,
            "d3_partition_audit_path": partition_audit,
            "strict2031_path": strict2031,
            "strict1607_path": strict1607,
        }

    def test_parent_matching_alignment_scope_and_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "out"
            configs = root / "configs"
            audit = build_panel(
                **self._fixture(root),
                output_dir=output,
                config_dir=configs,
                seed="unit-match",
            )
            self.assertEqual(audit["schema"], SCHEMA)
            self.assertNotEqual(SCHEMA, LEGACY_SCHEMA)
            self.assertIn("class_aligned", DEFAULT_OUTPUT_DIR.name)
            self.assertNotEqual(
                DEFAULT_OUTPUT_DIR.name, "stageb_tn_c2_parent_matched_20260717"
            )
            self.assertTrue(
                all("class_aligned_v2" in name for name in CONFIG_NAMES.values())
            )
            self.assertEqual(audit["statistics"]["train"]["pairs"], 3)
            self.assertEqual(audit["statistics"]["calibration"]["pairs"], 1)
            self.assertEqual(
                audit["statistics"]["train"]["edit_category_pairs"],
                {"color": 1, "size": 1, "spatial": 1},
            )
            self.assertEqual(
                audit["statistics"]["train"]["d2_candidate_count_histogram"],
                {"1": 2, "2": 1},
            )
            self.assertEqual(
                audit["statistics"]["train"]["canonical_class_relation_pairs"],
                {"aligned": 2, "mismatched": 1},
            )
            self.assertEqual(
                audit["statistics"]["train"]["causal_input_relation_pairs"],
                {
                    PRIMARY_CAUSAL_STRATUM: 1,
                    "different_text_class_aligned": 1,
                    "identical_text_class_mismatch": 1,
                },
            )
            self.assertEqual(
                audit["statistics"]["train"][
                    "class_aligned_identical_complete_input_pairs"
                ],
                1,
            )
            self.assertEqual(
                audit["statistics"]["train"][
                    "canonical_class_id_mismatch_pairs"
                ],
                1,
            )
            self.assertEqual(
                audit["statistics"]["train"][
                    "identical_negative_text_class_id_mismatch_pairs"
                ],
                1,
            )
            self.assertEqual(
                audit["statistics"]["train"][
                    "canonical_class_id_mismatch_direction_pairs"
                ],
                {"8->7": 1},
            )
            self.assertEqual(audit["invariants"]["strict_union_image_overlap"], 0)
            self.assertTrue(audit["invariants"]["train"]["aligned_pair_ids"])
            self.assertTrue(audit["invariants"]["train"]["unique_pair_ids"])
            self.assertTrue(audit["invariants"]["train"]["unique_parent_keys"])
            self.assertEqual(
                sum(
                    audit["statistics"]["train"][
                        "negative_text_relation_pairs"
                    ].values()
                ),
                3,
            )
            self.assertEqual(audit["matching_yield"]["train"]["d3_parent_rows"], 4)
            self.assertEqual(audit["matching_yield"]["train"]["matched_pairs"], 3)
            self.assertEqual(
                audit["matching_yield"]["train"]["unmatched_d3_parent_rows"], 1
            )
            self.assertEqual(
                audit["matching_yield"]["train"][
                    "matched_pairwise_claim_denominator"
                ],
                3,
            )
            self.assertEqual(
                audit["matching_yield"]["train"][
                    "class_aligned_identical_claim_denominator"
                ],
                1,
            )
            self.assertEqual(
                audit["matching_yield"]["train"][
                    "unmatched_d3_parent_rows_excluded_from_pairwise_claims"
                ],
                1,
            )
            self.assertEqual(
                audit["claim_scope"]["primary_causal_stratum"],
                PRIMARY_CAUSAL_STRATUM,
            )
            self.assertTrue(
                audit["claim_scope"]["unmatched_d3_parent_rows_are_out_of_scope"]
            )
            self.assertFalse(
                audit["claim_scope"][
                    "generalization_to_unmatched_d3_parent_rows_supported"
                ]
            )
            self.assertTrue(
                audit["runtime_contract"]["D2m_D3m_supported_by_current_v24"]
            )

            d2m = [
                json.loads(line)
                for line in (output / OUTPUT_NAMES["d2m_train"])
                .read_text()
                .splitlines()
            ]
            d3m = [
                json.loads(line)
                for line in (output / OUTPUT_NAMES["d3m_train"])
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [row["matched_pair_id"] for row in d2m],
                [row["matched_pair_id"] for row in d3m],
            )
            self.assertTrue(all(row["table_b_id"] == "D2m" for row in d2m))
            self.assertTrue(all(row["table_b_id"] == "D3m" for row in d3m))
            self.assertTrue(
                all(row["global_tn_verified"] is False for row in d2m + d3m)
            )
            self.assertTrue(
                all(row["positive_phrase_normalized_match"] for row in d2m + d3m)
            )
            relations = {
                row["matched_stratum"]["causal_input_relation"] for row in d2m
            }
            self.assertEqual(
                relations,
                {
                    PRIMARY_CAUSAL_STRATUM,
                    "different_text_class_aligned",
                    "identical_text_class_mismatch",
                },
            )
            clean = [
                row
                for row in d2m
                if row["matched_stratum"]["causal_input_relation"]
                == PRIMARY_CAUSAL_STRATUM
            ]
            self.assertEqual(len(clean), 1)
            self.assertTrue(clean[0]["canonical_class_id_match"])
            self.assertTrue(clean[0]["complete_model_input_exact_match"])
            mismatch = [row for row in d2m if not row["canonical_class_id_match"]]
            self.assertEqual(len(mismatch), 1)
            self.assertEqual(mismatch[0]["d2_canonical_class_id"], 8)
            self.assertEqual(mismatch[0]["d3_canonical_class_id"], 7)
            verify_panel(output / "audit.json")

            for table_id in ("D2m", "D3m"):
                config = json.loads(
                    (configs / CONFIG_NAMES[table_id]).read_text(encoding="utf-8")
                )
                self.assertEqual(config["train"][-1]["paper_table_b_id"], table_id)
                self.assertEqual(config["train"][-1]["mix_weight"], 3.0)
                self.assertFalse(config["train"][-1]["tn_balance_sampling"])
                self.assertTrue(
                    config["train"][-1]["paper_matched_causal_panel"]
                )
                self.assertTrue(config["train"][-1]["paper_runtime_supported"])
                self.assertEqual(
                    config["train"][-1]["paper_runtime_contract"],
                    "v24_parent_matched_class_aligned_v2_fail_closed",
                )

            for output_key, table_id, scope in (
                ("d2m_train", "D2m", "traceable_counterfactual_edit"),
                ("d3m_train", "D3m", "proposal_covered_verified"),
            ):
                dataset = PatchEpisodeJsonlDataset(
                    root=str(root),
                    anno=str(output / OUTPUT_NAMES[output_key]),
                    source="sam3_tn_pair",
                    box_format="xywh",
                    neg_episode_prob=0.0,
                    build_text_token_masks=False,
                    tn_balance_sampling=False,
                    table_b_id=table_id,
                    table_b_scope=scope,
                    table_b_audit_sha256="unit-audit",
                )
                self.assertEqual(len(dataset.metas), 3)
                instance = dataset.metas[0]["instances"][0]
                self.assertTrue(instance["sam3_tn_pair"])
                self.assertFalse(instance["global_tn_verified"])
                self.assertEqual(instance["tn_scope"], scope)

    def test_rebuild_is_deterministic(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "out"
            configs = root / "configs"
            kwargs = dict(
                **self._fixture(root),
                output_dir=output,
                config_dir=configs,
                seed="deterministic",
            )
            build_panel(**kwargs)
            before = {
                name: sha256_file(output / name) for name in OUTPUT_NAMES.values()
            }
            audit_before = sha256_file(output / "audit.json")
            build_panel(**kwargs)
            self.assertEqual(
                before,
                {name: sha256_file(output / name) for name in OUTPUT_NAMES.values()},
            )
            self.assertEqual(audit_before, sha256_file(output / "audit.json"))

    def test_v2_refuses_to_overwrite_sealed_v1_namespace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "sealed-v1"
            output.mkdir()
            sentinel = output / OUTPUT_NAMES["pairs_train"]
            sentinel.write_bytes(b"sealed-v1-bytes\n")
            (output / "audit.json").write_text(
                json.dumps({"schema": LEGACY_SCHEMA}), encoding="utf-8"
            )
            before = sha256_file(sentinel)
            with self.assertRaisesRegex(
                MatchedPanelError, "refusing to overwrite sealed v1"
            ):
                build_panel(
                    **self._fixture(root),
                    output_dir=output,
                    config_dir=root / "configs",
                    seed="must-not-overwrite-v1",
                )
            self.assertEqual(sha256_file(sentinel), before)
            with self.assertRaisesRegex(
                MatchedPanelError, "lacks canonical-class alignment"
            ):
                verify_panel(output / "audit.json")

    def test_missing_or_noninteger_class_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = self._fixture(root)
            rows = [
                json.loads(line)
                for line in fixture["d2_path"].read_text().splitlines()
            ]
            rows[0]["instances"][0]["class_id"] = True
            _write_jsonl(fixture["d2_path"], rows)
            with self.assertRaisesRegex(
                MatchedPanelError, "class_id must be an exact non-negative integer"
            ):
                build_panel(
                    **fixture,
                    output_dir=root / "out",
                    config_dir=root / "configs",
                    seed="invalid-class",
                )

    def test_verifier_rejects_overstated_class_or_unmatched_denominators(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "out"
            build_panel(
                **self._fixture(root),
                output_dir=output,
                config_dir=root / "configs",
                seed="claim-audit",
            )
            audit_path = output / "audit.json"
            pristine = json.loads(audit_path.read_text())

            overstated = json.loads(json.dumps(pristine))
            overstated["statistics"]["train"][
                "class_aligned_identical_complete_input_pairs"
            ] += 1
            audit_path.write_text(json.dumps(overstated), encoding="utf-8")
            with self.assertRaisesRegex(
                MatchedPanelError, "statistics do not match sealed pair records"
            ):
                verify_panel(audit_path)

            overstated = json.loads(json.dumps(pristine))
            overstated["matching_yield"]["train"][
                "unmatched_d3_parent_rows"
            ] -= 1
            audit_path.write_text(json.dumps(overstated), encoding="utf-8")
            with self.assertRaisesRegex(
                MatchedPanelError, "matched/unmatched claim denominator drifted"
            ):
                verify_panel(audit_path)

    def test_strict_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = self._fixture(root)
            _write_jsonl(fixture["strict2031_path"], [{"image_id": 10}])
            with self.assertRaisesRegex(MatchedPanelError, "strict evaluation"):
                build_panel(
                    **fixture,
                    output_dir=root / "out",
                    config_dir=root / "configs",
                    seed="leak",
                )


if __name__ == "__main__":
    unittest.main()
