import json
import tempfile
import unittest
from pathlib import Path

from tools.build_stageb_tn_data_ablation_matrix import (
    CONFIG_NAMES,
    OUTPUT_NAMES,
    TableBDataError,
    build_matrix,
    sha256_file,
    verify_matrix,
)
from datasets.patch_episode import PatchEpisodeJsonlDataset


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _d1_row(index, image_id):
    positive = f"blue object {index}"
    negative = f"red object {index}"
    instance = {
        "bbox": [1, 2, 30, 40],
        "class_id": 7,
        "canonical_name": "object",
        "head": "object",
        "positive_phrase": positive,
        "phrase": positive,
        "negative_phrase": negative,
        "try_tn": negative,
        "replace_category": ("color", "spatial", "size")[index % 3],
        "replace_from": positive,
        "replace_to": negative,
        "sam3_tn_pair": True,
        "benchmark_dataft_alltn": True,
        "global_tn_verified": False,
        "proposalset_proxy_verified": False,
        "tn_scope": "benchmark_dataft_alltn",
    }
    return {
        "sample_id": f"d1:{index}",
        "image_id": image_id,
        "ann_id": index,
        "ref_id": index,
        "sent_id": index,
        "filename": f"COCO_train2014_{image_id:012d}.jpg",
        "pair_source": ("refcoco", "refcocoplus", "refcocog")[index % 3],
        "source": "recovered",
        "instances": [instance],
        "benchmark_dataft_alltn": True,
        "global_tn_verified": False,
        "proposalset_proxy_verified": False,
        "tn_scope": "benchmark_dataft_alltn",
    }


def _d2_row(index, image_id):
    positive = f"blue object {index}"
    negative = f"red object {index}"
    return {
        "image_id": image_id,
        "ann_id": index,
        "ref_id": index,
        "sent_id": index,
        "filename": f"COCO_train2014_{image_id:012d}.jpg",
        "source": ("refcoco", "refcocoplus", "refcocog")[index % 3],
        "instances": [
            {
                "bbox": [1, 2, 30, 40],
                "class_id": 7,
                "category_name": "object",
                "head": "object",
                "head_phrase": "object",
                "raw_phrase": negative,
                "positive_phrase": positive,
                "try_tn": negative,
                "text_is_negative": True,
                "replace_category": ("color", "spatial", "size")[index % 3],
                "replace_from": "blue",
                "replace_to": "red",
                "replace_token": "blue",
                "try_tn_method": "synthetic_rule",
                "try_tn_rule": "single_token_attribute_swap",
                "pair_source": "synthetic",
            }
        ],
    }


def _d3_row(index, image_id):
    edit = {
        "category": ("color", "spatial", "size")[index % 3],
        "replace_from": "blue",
        "replace_to": "red",
        "replace_span": [0, 1],
    }
    return {
        "sample_id": f"d3:{index}",
        "image_id": image_id,
        "ann_id": index,
        "ref_id": index,
        "sent_id": index,
        "file_name": f"COCO_train2014_{image_id:012d}.jpg",
        "dataset": ("refcocoplus", "refcocog")[index % 2],
        "pair_source": "semantic",
        "class_id": 7,
        "category_name": "object",
        "class_norm_name": "object",
        "target_bbox_used": [1, 2, 30, 40],
        "sent": f"blue object {index}",
        "try_tn": f"red object {index}",
        "try_tn_head": "object",
        "try_tn_head_phrase": f"blue object {index}",
        "replace_category": [edit["category"]],
        "replace_from": [edit["replace_from"]],
        "replace_to": [edit["replace_to"]],
        "replace_span": [edit["replace_span"]],
        "tn_edits": [edit],
        "cached_proposal_coverage_only": True,
        "all_900_gdino_queries_verified": False,
        "global_max_label_is_semantic_extrapolation": True,
        "global_tn_verified": True,
        "tn_scope": "image_global_topk_verified",
        "proposal_count": 2,
        "coverage_policy": "all_proposals_all_no",
        "verification_contract": "target_plus_all_cached_proposals_no",
        "source": "semantic-source",
    }


def _make_d3_audit(path, train_path, cal_path):
    payload = {
        "schema": "stage-b-semantic-tn-leakage-isolated-partition-v1",
        "partition_contract_sha256": "a" * 64,
        "invariants": {
            "eligible_strict_union_image_overlap": 0,
            "train_calibration_image_overlap": 0,
            "single_edit_invalid_metadata_rows_excluded": True,
        },
        "outputs": {
            "single_edit_train": {"sha256": sha256_file(train_path)},
            "single_edit_calibration": {"sha256": sha256_file(cal_path)},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TableBDataMatrixTests(unittest.TestCase):
    def _fixture(self, root, *, target=12):
        d1 = root / "d1.jsonl"
        d2 = root / "d2.jsonl"
        d3_train = root / "d3_train.jsonl"
        d3_cal = root / "d3_cal.jsonl"
        strict2031 = root / "strict2031.jsonl"
        strict1607 = root / "strict1607.jsonl"
        # Many source images make both sides of the deterministic split non-empty.
        _write_jsonl(d1, [_d1_row(i, 1000 + i // 2) for i in range(120)])
        _write_jsonl(d2, [_d2_row(i, 2000 + i // 2) for i in range(120)])
        _write_jsonl(
            d3_train, [_d3_row(i, 3000 + i) for i in range(target)]
        )
        _write_jsonl(
            d3_cal, [_d3_row(100 + i, 4000 + i) for i in range(4)]
        )
        _write_jsonl(strict2031, [{"image_id": 1001}, {"image_id": 2001}])
        _write_jsonl(strict1607, [{"image_id": 1002}, {"image_id": 2002}])
        d3_audit = root / "d3_audit.json"
        _make_d3_audit(d3_audit, d3_train, d3_cal)
        return {
            "d1_path": d1,
            "d2_path": d2,
            "d3_train_path": d3_train,
            "d3_calibration_path": d3_cal,
            "d3_audit_path": d3_audit,
            "strict2031_path": strict2031,
            "strict1607_path": strict1607,
            "target_rows": target,
        }

    def test_equal_size_leakage_scope_and_configs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            kwargs = self._fixture(root)
            output = root / "out"
            configs = root / "configs"
            audit = build_matrix(
                **kwargs,
                output_dir=output,
                config_dir=configs,
                seed="unit-table-b",
                calibration_ratio="1/5",
            )

            self.assertTrue(audit["invariants"]["equal_train_rows_D1_D3"])
            self.assertTrue(audit["invariants"]["D1_D2_taxonomy_counts_equal"])
            self.assertEqual(
                audit["invariants"]["runtime_global_verified_true_rows"],
                {key: 0 for key in OUTPUT_NAMES},
            )
            self.assertTrue(
                all(
                    value == 0
                    for value in audit["invariants"][
                        "strict_union_overlap_D1_D3"
                    ].values()
                )
            )
            for key in ("d1_train", "d2_train", "d3_train"):
                rows = [
                    json.loads(line)
                    for line in (output / OUTPUT_NAMES[key]).read_text().splitlines()
                ]
                self.assertEqual(len(rows), kwargs["target_rows"])
                self.assertTrue(all(row["global_tn_verified"] is False for row in rows))

            d2_rows = [
                json.loads(line)
                for line in (output / OUTPUT_NAMES["d2_train"])
                .read_text()
                .splitlines()
            ]
            self.assertTrue(
                all(row["tn_scope"] == "traceable_counterfactual_edit" for row in d2_rows)
            )
            self.assertTrue(
                all(row["source_provenance"]["try_tn_rule"] for row in d2_rows)
            )

            d3_rows = [
                json.loads(line)
                for line in (output / OUTPUT_NAMES["d3_train"])
                .read_text()
                .splitlines()
            ]
            self.assertTrue(
                all(row["tn_scope"] == "proposal_covered_verified" for row in d3_rows)
            )
            self.assertTrue(
                all(
                    row["source_provenance"]["source_global_tn_verified"] is True
                    for row in d3_rows
                )
            )

            configs_json = {
                key: json.loads((configs / name).read_text())
                for key, name in CONFIG_NAMES.items()
            }
            positives = configs_json["D0"]["train"]
            self.assertEqual([item["mix_weight"] for item in positives], [1.0] * 3)
            for key in ("D1", "D2", "D3"):
                self.assertEqual(configs_json[key]["train"][:3], positives)
                tn = configs_json[key]["train"][3]
                self.assertEqual(tn["mix_weight"], 3.0)
                self.assertFalse(tn["tn_balance_sampling"])
                self.assertFalse(tn["require_global_tn_verified"])
            verify_matrix(audit_path=output / "audit.json")

    def test_rebuild_is_deterministic(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            kwargs = self._fixture(root)
            output = root / "out"
            configs = root / "configs"
            common = dict(
                **kwargs,
                output_dir=output,
                config_dir=configs,
                seed="determinism",
                calibration_ratio="1/5",
            )
            build_matrix(**common)
            hashes_before = {
                name: sha256_file(output / name) for name in OUTPUT_NAMES.values()
            }
            audit_hash_before = sha256_file(output / "audit.json")
            build_matrix(**common)
            self.assertEqual(
                hashes_before,
                {name: sha256_file(output / name) for name in OUTPUT_NAMES.values()},
            )
            self.assertEqual(audit_hash_before, sha256_file(output / "audit.json"))

    def test_scope_preserving_rows_normalize_to_explicit_pairs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            kwargs = self._fixture(root)
            output = root / "out"
            build_matrix(
                **kwargs,
                output_dir=output,
                config_dir=root / "configs",
                seed="loader",
                calibration_ratio="1/5",
            )
            for output_key, expected_scope in (
                ("d1_train", "unverified_all_negative"),
                ("d2_train", "traceable_counterfactual_edit"),
                ("d3_train", "proposal_covered_verified"),
            ):
                dataset = PatchEpisodeJsonlDataset(
                    root=str(root),
                    anno=str(output / OUTPUT_NAMES[output_key]),
                    source="sam3_tn_pair",
                    box_format="xywh",
                    neg_episode_prob=0.0,
                    build_text_token_masks=False,
                    tn_balance_sampling=False,
                )
                self.assertEqual(len(dataset.metas), kwargs["target_rows"])
                instance = dataset.metas[0]["instances"][0]
                self.assertTrue(instance["sam3_tn_pair"])
                self.assertTrue(instance["positive_phrase"])
                self.assertTrue(instance["negative_phrase"])
                self.assertFalse(instance["global_tn_verified"])
                self.assertEqual(instance["tn_scope"], expected_scope)

    def test_d3_strict_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            kwargs = self._fixture(root)
            _write_jsonl(kwargs["strict2031_path"], [{"image_id": 3000}])
            with self.assertRaisesRegex(TableBDataError, "D3.*overlaps"):
                build_matrix(
                    **kwargs,
                    output_dir=root / "out",
                    config_dir=root / "configs",
                    seed="leak",
                    calibration_ratio="1/5",
                )

    def test_unverified_d2_cannot_claim_global(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            kwargs = self._fixture(root)
            rows = [json.loads(line) for line in kwargs["d2_path"].read_text().splitlines()]
            rows[0]["instances"][0]["global_tn_verified"] = True
            _write_jsonl(kwargs["d2_path"], rows)
            with self.assertRaisesRegex(TableBDataError, "claims global verification"):
                build_matrix(
                    **kwargs,
                    output_dir=root / "out",
                    config_dir=root / "configs",
                    seed="scope",
                    calibration_ratio="1/5",
                )


if __name__ == "__main__":
    unittest.main()
