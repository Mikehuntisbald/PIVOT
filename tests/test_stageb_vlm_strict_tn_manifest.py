import json
import unittest

from tools.build_stageb_vlm_strict_tn_manifest import (
    COVERAGE_ALL_PROPOSALS_ALL_NO,
    COVERAGE_TARGET_PLUS_PROPOSAL,
    SCHEMA_VERSION,
    _manifest_bytes,
    _is_generic_stage_a_detection_source,
    _resolve_g_umd_identity,
    _validate_manifest_records,
    passes_coverage_policy,
    proposal_coverage,
)


def _row(proposal_count, answers, *, target_answer="no"):
    return {
        "proposal_num": proposal_count,
        "proposal_cache": [
            {"proposal_id": index} for index in range(proposal_count)
        ],
        "visual_local_judgment": {"answer": target_answer},
        "visual_proposal_judgments": [
            {
                "proposal_id": index,
                "judgment": {"answer": answer},
            }
            for index, answer in enumerate(answers)
        ],
    }


def _manifest_record(sent_id=1):
    return {
        "ann_id": 2,
        "coverage_pass": True,
        "eval_split": "refcocog_umd_val",
        "image_id": 3,
        "box": [0.0, 0.0, 10.0, 10.0],
        "canonical": "car",
        "class_id": 3,
        "filename": "/tmp/COCO_train2014_000000000003.jpg",
        "instances": [
            {
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "pair_source": "refcocog_umd",
            }
        ],
        "manifest_schema": SCHEMA_VERSION,
        "negative_phrase": "blue car",
        "pair_source": "refcocog_umd",
        "positive_phrase": "red car",
        "ref_id": 4,
        "sample_id": f"sample-{sent_id}",
        "sent_id": sent_id,
    }


class StageBVlmStrictTnManifestTest(unittest.TestCase):
    def test_semantic_union_ignores_only_named_generic_stage_a_detection_sources(self):
        self.assertTrue(
            _is_generic_stage_a_detection_source(
                {
                    "dataset_mode": "odvg",
                    "path": "/tmp/stagea_odvg_train_1_coco.jsonl",
                }
            )
        )
        self.assertFalse(
            _is_generic_stage_a_detection_source(
                {
                    "dataset_mode": "odvg",
                    "path": "/tmp/stageb_gdino_ft_refexp_tn_stageb_v1_vg_empty.jsonl",
                }
            )
        )

    def test_umd_identity_resolution_uses_sent_id_to_disambiguate(self):
        candidates = [(10, 7, "val"), (11, 8, "val")]
        self.assertEqual(
            _resolve_g_umd_identity(
                candidates,
                accepted_sent_id=8,
                context="test",
            ),
            (11, 8, "val", "accepted_sent_id"),
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            _resolve_g_umd_identity(
                candidates,
                accepted_sent_id=9,
                context="test",
            )

    def test_target_plus_proposal_reproduces_one_target_allowance(self):
        coverage = proposal_coverage(_row(2, ["no"]))
        self.assertTrue(coverage["target_plus_proposal_covered"])
        self.assertFalse(coverage["all_proposals_all_no"])
        self.assertTrue(
            passes_coverage_policy(coverage, COVERAGE_TARGET_PLUS_PROPOSAL)
        )
        self.assertFalse(
            passes_coverage_policy(coverage, COVERAGE_ALL_PROPOSALS_ALL_NO)
        )

        uncovered = proposal_coverage(_row(2, []))
        self.assertFalse(uncovered["target_plus_proposal_covered"])

    def test_all_proposals_all_no_is_strict_and_nonempty(self):
        trusted = proposal_coverage(_row(2, ["no", "no"]))
        self.assertTrue(trusted["all_proposals_all_no"])

        unknown = proposal_coverage(_row(2, ["no", "unknown"]))
        self.assertFalse(unknown["all_proposals_all_no"])

        empty = proposal_coverage(_row(0, []))
        self.assertFalse(empty["all_proposals_all_no"])

    def test_manifest_validation_rejects_duplicates_and_unsorted_rows(self):
        first = _manifest_record(1)
        duplicate = dict(first)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _validate_manifest_records(
                [first, duplicate],
                require_coverage_pass=True,
            )

        second = _manifest_record(2)
        with self.assertRaisesRegex(ValueError, "not stably sorted"):
            _validate_manifest_records(
                [second, first],
                require_coverage_pass=True,
            )

    def test_manifest_serialization_is_deterministic_ascii_jsonl(self):
        rows = [_manifest_record(1), _manifest_record(2)]
        first = _manifest_bytes(rows)
        second = _manifest_bytes(rows)
        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        for line in first.splitlines():
            self.assertIsInstance(json.loads(line), dict)


if __name__ == "__main__":
    unittest.main()
