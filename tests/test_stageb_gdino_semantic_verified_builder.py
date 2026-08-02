import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from datasets.patch_episode import PatchEpisodeJsonlDataset
from tools.build_stageb_gdino_adapter_semantic_verified_pairs import (
    PAIR_SCHEMA,
    SemanticPairError,
    _make_pair,
    _validate_verification,
    verify,
)


class StageBGDINOSemanticVerifiedBuilderTest(unittest.TestCase):
    @staticmethod
    def _row():
        return {
            "dataset": "refcocoplus",
            "pair_source": "refcoco+_unc",
            "split": "train",
            "image_id": 10,
            "ann_id": 20,
            "ref_id": 30,
            "sent_id": 40,
            "class_id": 7,
            "category_name": "car",
            "class_norm_name": "car",
            "target_bbox_used": [1, 2, 10, 20],
            "sent": "red car",
            "try_tn": "blue car",
            "try_tn_head": "car",
            "replace_from": ["red"],
            "replace_to": ["blue"],
            "replace_category": ["color"],
            "visual_filter_status": "accept",
            "visual_filter_reason": "verified_negative",
            "visual_local_judgment": {"answer": "no"},
            "proposal_num": 1,
            "proposal_cache": [{"proposal_id": 0}],
            "visual_proposal_judgments": [
                {"proposal_id": 0, "judgment": {"answer": "no"}}
            ],
            "tn_scope": "image_global_proposal_verified",
            "global_tn_verified": True,
        }

    def test_pair_contract_is_semantic_topk_and_not_proposal_proxy(self):
        row = self._row()
        coverage = _validate_verification(row, context="unit-row")
        pair = _make_pair(
            row,
            dataset="refcocoplus",
            source_path=Path("source.jsonl"),
            source_line=1,
            coverage=coverage,
        )
        self.assertEqual(pair["adapter_pair_schema"], PAIR_SCHEMA)
        self.assertEqual(pair["tn_scope"], "image_global_topk_verified")
        self.assertEqual(pair["coverage_policy"], "all_proposals_all_no")
        self.assertIs(pair["global_tn_verified"], True)
        self.assertIs(pair["proposalset_proxy_verified"], False)
        self.assertIs(pair["cached_proposal_coverage_only"], True)
        self.assertIs(pair["all_900_gdino_queries_verified"], False)
        self.assertIs(pair["global_max_label_is_semantic_extrapolation"], True)
        self.assertEqual(pair["sent"], "red car")
        self.assertEqual(pair["try_tn"], "blue car")

    def test_verification_fails_closed_on_incomplete_proposals(self):
        row = self._row()
        row["visual_proposal_judgments"] = []
        with self.assertRaisesRegex(SemanticPairError, "not verified no"):
            _validate_verification(row, context="unit-row")

    def test_no_support_dataset_emits_exact_positive_tn_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(
                root / "COCO_train2014_000000000010.jpg"
            )
            row = self._row()
            coverage = _validate_verification(row, context="unit-row")
            pair = _make_pair(
                row,
                dataset="refcocoplus",
                source_path=root / "source.jsonl",
                source_line=1,
                coverage=coverage,
            )
            annotation = root / "pairs.jsonl"
            annotation.write_text(json.dumps(pair) + "\n", encoding="utf-8")
            dataset = PatchEpisodeJsonlDataset(
                root=str(root),
                anno=str(annotation),
                source="sam3_tn_pair",
                sam3_tn_image_root=str(root),
                sam3_tn_bbox_key="target_bbox_used",
                box_format="xywh",
                neg_episode_prob=0.0,
                support_min_count=1,
                support_num_patches_min=1,
                support_num_patches_max=1,
                build_text_token_masks=True,
                text_encoder_type="bert-base-uncased",
                text_mask_warn_limit=0,
                tn_balance_sampling=False,
                require_global_tn_verified=True,
                stage_b_gdino_adapter_no_support=True,
            )
            _image, target = dataset[0]

        self.assertEqual(target["cap_list"], ["red car", "blue car"])
        self.assertEqual(target["is_tn"].tolist(), [False, True])
        self.assertEqual(target["global_tn_verified"].tolist(), [True])
        self.assertEqual(target["proposalset_proxy_verified"].tolist(), [False])
        self.assertEqual(target["tn_scope"], "image_global_topk_verified")
        self.assertNotIn("patch", target)
        self.assertNotIn("patch_global", target)

    def test_frozen_full_output_verifies(self):
        root = Path(__file__).resolve().parents[1]
        output = (
            root
            / "data/ablations/stageb_gdino_adapter_semantic_verified_20260711/"
            "semantic_verified_pairs.jsonl"
        )
        audit = output.parent / "audit.json"

        class Args:
            expected_rows = 17_829

        args = Args()
        args.output = output
        args.audit = audit
        result = verify(args)
        self.assertEqual(result["rows"], 17_829)
        self.assertEqual(
            result["output_sha256"],
            "bea2aca85d207d883da85cb219420f748a65a840516218731811e8e46449b645",
        )


if __name__ == "__main__":
    unittest.main()
