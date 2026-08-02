import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from datasets.patch_episode import PatchEpisodeJsonlDataset
from models.GroundingDINO.groundingdino import GroundingDINO
from tools.build_stageb_v15_fixed_stagea_topk_exact_pairs import (
    CANDIDATE_SELECTION_SCHEMA,
    JUDGE_CONTRACT_SCHEMA,
    ExactTopKBuildError,
    _judgment_payload,
    build,
)
from tools.extract_stageb_v15_fixed_stagea_topk import (
    ExtractionError,
    make_candidates,
)
from tools.render_stageb_v15_fixed_stagea_topk_evidence import (
    EvidenceError,
    render as render_evidence,
    verify as verify_evidence,
)
from tools.seal_stageb_v15_fixed_stagea_topk_reviews import (
    DECISION_SCHEMA as EXTERNAL_DECISION_SCHEMA,
    SealError,
    seal as seal_reviews,
    verify as verify_sealed_reviews,
)
from util.stageb_exact_topk_contract import (
    EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA,
    EXACT_TOPK_EXTRACTION_SCHEMA,
    EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA,
    EXACT_TOPK_JUDGMENT_SCHEMA,
    EXACT_TOPK_PROTOCOL,
    EXACT_TOPK_TN_SCOPE,
    ExactTopKContractError,
    canonical_sha256,
    file_record,
    sha256_file,
    validate_exact_pair_collection,
    validate_exact_pair_row,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class StageBV15ExactTopKProtocolTest(unittest.TestCase):
    def _artifacts(self, root: Path, *, drop_last_judgment: bool = False):
        checkpoint = root / "stagea.pth"
        model_config = root / "stagea.py"
        data_config = root / "data.json"
        canonical = root / "canonical.json"
        checkpoint.write_bytes(b"fixed-stage-a")
        model_config.write_text("candidate_topk = 2\n", encoding="utf-8")
        data_config.write_text("{}\n", encoding="utf-8")
        canonical.write_text(
            json.dumps([{"id": 7, "norm_name": "car", "synonyms": ["car"]}])
            + "\n",
            encoding="utf-8",
        )
        image_path = root / "image.jpg"
        patch_path = root / "support.jpg"
        Image.new("RGB", (32, 32), color=(80, 100, 120)).save(image_path)
        Image.new("RGB", (24, 24), color=(150, 20, 20)).save(patch_path)

        query_transform_payload = {
            "schema": "stage-b-v15-query-transform-contract-v1",
            "deterministic": True,
            "resize": [800, 1333],
            "hflip_probability": 0.0,
        }
        support_transform_payload = {
            "schema": "stage-b-v15-support-transform-contract-v1",
            "resize": 256,
            "center_crop": 224,
        }
        selection_payload = {
            "schema": CANDIDATE_SELECTION_SCHEMA,
            "candidate_topk": 2,
            "score_source": "score_patch_logits",
            "selection": "torch.topk(largest=true,sorted=true)",
            "candidate_order": "descending_patch_logit",
            "candidate_box_space": "normalized_cxcywh",
            "fixed_support_patch_per_row": True,
            "deterministic_query_transform": True,
            "dynamic_candidate_replay_must_match": True,
            "candidate_box_atol": 1.0e-5,
        }
        exact_contract = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_config_sha256": sha256_file(model_config),
            "data_config_sha256": sha256_file(data_config),
            "canonical_classes_sha256": sha256_file(canonical),
            "query_transform_contract_sha256": canonical_sha256(
                query_transform_payload
            ),
            "support_transform_contract_sha256": canonical_sha256(
                support_transform_payload
            ),
            "candidate_selection_contract_sha256": canonical_sha256(
                selection_payload
            ),
            "candidate_topk": 2,
            "candidate_box_atol": 1.0e-5,
        }
        candidates = [
            {
                "rank": 0,
                "query_index": 17,
                "bbox_cxcywh_normalized": [0.5, 0.5, 0.4, 0.4],
                "patch_logit": 4.0,
            },
            {
                "rank": 1,
                "query_index": 8,
                "bbox_cxcywh_normalized": [0.2, 0.3, 0.1, 0.2],
                "patch_logit": 3.0,
            },
        ]
        for candidate in candidates:
            candidate["candidate_sha256"] = canonical_sha256(
                {key: value for key, value in candidate.items() if key != "candidate_sha256"}
            )
        candidate_payloads = [
            {key: value for key, value in candidate.items() if key != "candidate_sha256"}
            for candidate in candidates
        ]
        candidate_set_sha = canonical_sha256(candidate_payloads)
        source_pair = {
            "dataset": "refcocoplus",
            "pair_source": "unit",
            "split": "train",
            "sample_id": "exact:1",
            "image_path": str(image_path),
            "file_name": image_path.name,
            "image_id": 1,
            "ann_id": 2,
            "ref_id": 3,
            "sent_id": 4,
            "class_id": 7,
            "class_norm_name": "car",
            "category_name": "car",
            "target_bbox_used": [4, 4, 16, 16],
            "sent": "red car",
            "try_tn": "blue car",
            "try_tn_head": "car",
            "replace_from": ["red"],
            "replace_to": ["blue"],
            "replace_category": ["color"],
        }
        extraction = {
            "schema": EXACT_TOPK_EXTRACTION_SCHEMA,
            "protocol": EXACT_TOPK_PROTOCOL,
            "sample_id": "exact:1",
            "source_pair": source_pair,
            "source_pair_sha256": canonical_sha256(source_pair),
            "image": file_record(image_path),
            "exact_contract": exact_contract,
            "query_transform_trace": {
                "schema": "stage-b-v15-fixed-stagea-query-transform-trace-v1",
                "original_hw": [32, 32],
                "output_hw": [800, 800],
                "scale_xy": [25.0, 25.0],
                "offset_xy": [0.0, 0.0],
                "operations": [],
            },
            "fixed_support_patch": {
                "path": str(patch_path),
                "sha256": sha256_file(patch_path),
                "class_id": 7,
                "transform_contract_sha256": exact_contract[
                    "support_transform_contract_sha256"
                ],
            },
            "candidate_set_sha256": candidate_set_sha,
            "candidates": candidates,
        }
        extraction["query_transform_trace_sha256"] = canonical_sha256(
            extraction["query_transform_trace"]
        )
        extraction_path = root / "extractions.jsonl"
        _write_jsonl(extraction_path, [extraction])
        extraction_audit = {
            "schema": EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA,
            "protocol": EXACT_TOPK_PROTOCOL,
            "complete": True,
            "rows": 1,
            "exact_contract": exact_contract,
            "extractions": file_record(extraction_path, rows=1),
            "provenance": {
                "checkpoint": file_record(checkpoint),
                "model_config": file_record(model_config),
                "data_config": file_record(data_config),
                "canonical_classes": file_record(canonical),
            },
            "query_transform_contract": {
                **query_transform_payload,
                "sha256": canonical_sha256(query_transform_payload),
            },
            "support_transform_contract": {
                **support_transform_payload,
                "sha256": canonical_sha256(support_transform_payload),
            },
            "candidate_selection_contract": {
                **selection_payload,
                "sha256": canonical_sha256(selection_payload),
            },
        }
        extraction_audit_path = root / "extraction_audit.json"
        _write_json(extraction_audit_path, extraction_audit)

        judge_payload = {
            "schema": JUDGE_CONTRACT_SCHEMA,
            "judge_type": "human",
            "reviewer_policy": "two-pass boxed-region review",
            "prompt_template_sha256": "a" * 64,
            "evidence_asset_policy_sha256": "b" * 64,
            "min_no_confidence": 0.9,
        }
        judge_sha = canonical_sha256(judge_payload)
        extraction_row_sha = canonical_sha256(extraction)
        judgments = []
        for candidate in candidates:
            judgment = {
                "schema": EXACT_TOPK_JUDGMENT_SCHEMA,
                "protocol": EXACT_TOPK_PROTOCOL,
                "status": "complete",
                "sample_id": "exact:1",
                "candidate_rank": candidate["rank"],
                "query_index": candidate["query_index"],
                "candidate_sha256": candidate["candidate_sha256"],
                "candidate_set_sha256": candidate_set_sha,
                "extraction_row_sha256": extraction_row_sha,
                "answer": "no",
                "confidence": 0.99,
                "evidence_sha256": str(candidate["rank"] + 1) * 64,
                "judge_contract_sha256": judge_sha,
            }
            judgment["judgment_sha256"] = canonical_sha256(
                _judgment_payload(judgment)
            )
            judgments.append(judgment)
        if drop_last_judgment:
            judgments.pop()
        judgment_path = root / "judgments.jsonl"
        _write_jsonl(judgment_path, judgments)
        judgment_audit = {
            "schema": EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA,
            "protocol": EXACT_TOPK_PROTOCOL,
            "complete": True,
            "rows": len(judgments),
            "extractions": file_record(extraction_path, rows=1),
            "judgments": file_record(judgment_path, rows=len(judgments)),
            "judge_contract": {**judge_payload, "sha256": judge_sha},
        }
        judgment_audit_path = root / "judgment_audit.json"
        _write_json(judgment_audit_path, judgment_audit)
        return {
            "exact_contract": exact_contract,
            "canonical": canonical,
            "extractions": extraction_path,
            "extraction_audit": extraction_audit_path,
            "judgments": judgment_path,
            "judgment_audit": judgment_audit_path,
            "output": root / "pairs.jsonl",
            "decisions": root / "decisions.jsonl",
            "audit": root / "audit.json",
        }

    @staticmethod
    def _args(paths):
        return argparse.Namespace(
            extractions=paths["extractions"],
            extraction_audit=paths["extraction_audit"],
            judgments=paths["judgments"],
            judgment_audit=paths["judgment_audit"],
            output=paths["output"],
            decisions=paths["decisions"],
            audit=paths["audit"],
        )

    def test_builder_and_loader_require_complete_exact_candidate_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._artifacts(root)
            result = build(self._args(paths))
            self.assertEqual(result["accepted_rows"], 1)
            rows = [json.loads(line) for line in paths["output"].read_text().splitlines()]
            validated = validate_exact_pair_collection(
                rows,
                annotation_path=paths["output"],
                audit_path=paths["audit"],
                expected_contract=paths["exact_contract"],
            )
            self.assertEqual(validated[0]["candidate_indices"], [17, 8])
            self.assertEqual(rows[0]["tn_scope"], EXACT_TOPK_TN_SCOPE)
            self.assertIs(rows[0]["global_max_label_is_semantic_extrapolation"], False)

            dataset = PatchEpisodeJsonlDataset(
                root=str(root),
                anno=str(paths["output"]),
                source="sam3_tn_pair",
                sam3_tn_image_root=str(root),
                sam3_tn_bbox_key="target_bbox_used",
                box_format="xywh",
                neg_episode_prob=0.0,
                support_min_count=1,
                support_num_patches_min=1,
                support_num_patches_max=1,
                canonical_classes_json=str(paths["canonical"]),
                build_text_token_masks=True,
                text_encoder_type="bert-base-uncased",
                text_mask_warn_limit=0,
                tn_balance_sampling=False,
                require_fixed_stagea_topk_exact_verified=True,
                fixed_stagea_topk_exact_audit=str(paths["audit"]),
                fixed_stagea_topk_expected_contract=paths["exact_contract"],
            )
            _image, target = dataset[0]
            self.assertEqual(target["fixed_stagea_candidate_indices"].tolist(), [17, 8])
            self.assertEqual(target["fixed_stagea_candidate_boxes"].shape, (2, 4))
            self.assertTrue(target["fixed_stagea_topk_exact_verified"].item())
            self.assertEqual(tuple(target["patch"].shape), (3, 224, 224))

            support_path = Path(rows[0]["fixed_stagea_support_patch"]["path"])
            support_path.write_bytes(b"tampered-support")
            with self.assertRaisesRegex(RuntimeError, "support patch hash drifted"):
                dataset[0]

            paths["judgments"].write_text(
                paths["judgments"].read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ExactTopKContractError, "drifted"):
                validate_exact_pair_collection(
                    rows,
                    annotation_path=paths["output"],
                    audit_path=paths["audit"],
                    expected_contract=paths["exact_contract"],
                )

    def test_builder_fails_closed_when_one_candidate_judgment_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._artifacts(Path(temporary), drop_last_judgment=True)
            with self.assertRaisesRegex(ExactTopKBuildError, "rows.K"):
                build(self._args(paths))

    def test_legacy_semantic_row_cannot_claim_exact_scope(self):
        row = {
            "adapter_pair_schema": "stage-b-gdino-adapter-semantic-verified-pair-v1",
            "tn_scope": "image_global_topk_verified",
            "global_tn_verified": True,
        }
        contract = {key: "a" * 64 for key in (
            "checkpoint_sha256",
            "model_config_sha256",
            "data_config_sha256",
            "canonical_classes_sha256",
            "query_transform_contract_sha256",
            "support_transform_contract_sha256",
            "candidate_selection_contract_sha256",
        )}
        contract.update(candidate_topk=50, candidate_box_atol=1.0e-5)
        with self.assertRaisesRegex(ExactTopKContractError, "schema"):
            validate_exact_pair_row(row, expected_contract=contract)

    def test_runtime_replay_rejects_index_or_box_drift(self):
        indices = torch.tensor([[2, 1], [4, 3]], dtype=torch.int64)
        boxes = torch.tensor(
            [
                [[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]],
                [[0.4, 0.4, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]],
            ],
            dtype=torch.float32,
        )
        exact_mask = torch.tensor([False, True])
        GroundingDINO.assert_stage_b_v15_exact_candidates(
            indices,
            boxes,
            exact_mask=exact_mask,
            expected_indices=indices.clone(),
            expected_boxes=boxes.clone(),
            box_atol=1.0e-5,
        )
        bad_indices = indices.clone()
        bad_indices[1, 0] = 8
        with self.assertRaisesRegex(RuntimeError, "ordered Top-K"):
            GroundingDINO.assert_stage_b_v15_exact_candidates(
                indices,
                boxes,
                exact_mask=exact_mask,
                expected_indices=bad_indices,
                expected_boxes=boxes,
                box_atol=1.0e-5,
            )
        bad_boxes = boxes.clone()
        bad_boxes[1, 0, 0] += 1.0e-3
        with self.assertRaisesRegex(RuntimeError, "boxes drifted"):
            GroundingDINO.assert_stage_b_v15_exact_candidates(
                indices,
                boxes,
                exact_mask=exact_mask,
                expected_indices=indices,
                expected_boxes=bad_boxes,
                box_atol=1.0e-5,
            )

    def test_extractor_selects_and_hashes_the_complete_ordered_top50(self):
        logits = torch.linspace(-4.0, 4.0, steps=900)
        boxes = torch.full((900, 4), 0.25, dtype=torch.float32)
        candidates, candidate_set_sha = make_candidates(logits, boxes, topk=50)
        self.assertEqual(len(candidates), 50)
        self.assertEqual(
            [row["query_index"] for row in candidates],
            list(range(899, 849, -1)),
        )
        payloads = [
            {key: value for key, value in row.items() if key != "candidate_sha256"}
            for row in candidates
        ]
        self.assertEqual(candidate_set_sha, canonical_sha256(payloads))
        self.assertTrue(
            all(
                row["candidate_sha256"] == canonical_sha256(payload)
                for row, payload in zip(candidates, payloads)
            )
        )
        with self.assertRaisesRegex(ExtractionError, "900 decoder queries"):
            make_candidates(logits[:-1], boxes[:-1], topk=50)

    @staticmethod
    def _review_args(paths, root: Path):
        prompt = root / "review_prompt.txt"
        prompt.write_text(
            "Judge only whether the boxed candidate satisfies the negative expression.\n",
            encoding="utf-8",
        )
        return argparse.Namespace(
            extractions=paths["extractions"],
            extraction_audit=paths["extraction_audit"],
            prompt_template=prompt,
            judge_type="human",
            min_no_confidence=0.9,
            review_manifest=root / "pending_reviews.jsonl",
            audit=root / "pending_reviews.audit.json",
            assets_dir=root / "evidence_assets",
            work_dir=root / "evidence_work",
            log_every=0,
            dry_run=False,
            list=False,
            verify_only=False,
        )

    @staticmethod
    def _seal_args(paths, review_args, root: Path, decisions: Path):
        return argparse.Namespace(
            extractions=paths["extractions"],
            extraction_audit=paths["extraction_audit"],
            review_manifest=review_args.review_manifest,
            review_audit=review_args.audit,
            completed_reviews=decisions,
            judgments=root / "sealed_judgments.jsonl",
            judgment_audit=root / "sealed_judgments.audit.json",
            dry_run=False,
            verify_only=False,
        )

    def test_render_external_review_seal_and_builder_are_one_closed_cpu_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._artifacts(root)
            review_args = self._review_args(paths, root)
            review_audit = render_evidence(review_args)
            self.assertTrue(review_audit["evidence_complete"])
            self.assertFalse(review_audit["review_complete"])
            self.assertEqual(review_audit["answers_present"], 0)
            pending = [
                json.loads(line)
                for line in review_args.review_manifest.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(pending), 2)
            self.assertTrue(all(row["answer"] is None for row in pending))
            self.assertTrue(
                all(Path(row["evidence_path"]).is_file() for row in pending)
            )
            verified = verify_evidence(review_args)
            self.assertEqual(verified["rows"], 2)
            self.assertFalse(verified["review_complete"])

            decisions_path = root / "explicit_external_decisions.jsonl"
            _write_jsonl(
                decisions_path,
                [
                    {
                        "schema": EXTERNAL_DECISION_SCHEMA,
                        "sample_id": row["sample_id"],
                        "candidate_rank": row["candidate_rank"],
                        "answer": "no",
                        "confidence": 0.99,
                        "reviewer_note": f"explicit synthetic rank {row['candidate_rank']}",
                    }
                    for row in pending
                ],
            )
            seal_args = self._seal_args(paths, review_args, root, decisions_path)
            judgment_audit = seal_reviews(seal_args)
            self.assertTrue(judgment_audit["review_complete"])
            self.assertEqual(judgment_audit["rows"], 2)
            self.assertEqual(verify_sealed_reviews(seal_args)["rows"], 2)

            build_args = self._args(paths)
            build_args.judgments = seal_args.judgments
            build_args.judgment_audit = seal_args.judgment_audit
            result = build(build_args)
            self.assertEqual(result["accepted_rows"], 1)
            self.assertEqual(result["candidate_judgments"], 2)

    def test_sealer_rejects_missing_decision_and_tampered_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._artifacts(root)
            review_args = self._review_args(paths, root)
            render_evidence(review_args)
            pending = [
                json.loads(line)
                for line in review_args.review_manifest.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            decisions_path = root / "incomplete_decisions.jsonl"
            _write_jsonl(
                decisions_path,
                [
                    {
                        "schema": EXTERNAL_DECISION_SCHEMA,
                        "sample_id": pending[0]["sample_id"],
                        "candidate_rank": pending[0]["candidate_rank"],
                        "answer": "no",
                        "confidence": 0.99,
                    }
                ],
            )
            seal_args = self._seal_args(paths, review_args, root, decisions_path)
            with self.assertRaisesRegex(SealError, "coverage is not exact"):
                seal_reviews(seal_args)

            _write_jsonl(
                decisions_path,
                [
                    {
                        "schema": EXTERNAL_DECISION_SCHEMA,
                        "sample_id": row["sample_id"],
                        "candidate_rank": row["candidate_rank"],
                        "answer": "no",
                        "confidence": 0.99,
                    }
                    for row in pending
                ],
            )
            Path(pending[0]["evidence_path"]).write_bytes(b"tampered-evidence")
            with self.assertRaisesRegex(
                (EvidenceError, ExactTopKContractError),
                "evidence|asset",
            ):
                seal_reviews(seal_args)


if __name__ == "__main__":
    unittest.main()
