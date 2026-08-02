import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from tools.judge_stageb_fixed_gdino_top1_qwen import (
    ASSET_POLICY_SHA256,
    EXTRACTION_SCHEMA,
    GENERATION_CONFIG,
    GENERATION_CONFIG_SHA256,
    INFERENCE_BATCH_SIZE,
    JUDGE_RUNTIME_POLICY,
    JUDGE_RUNTIME_POLICY_SHA256,
    JUDGMENT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_TEMPLATE_SHA256,
    VISION_PROCESSOR_CONFIG,
    VISION_PROCESSOR_CONFIG_SHA256,
    build_boxed_assets,
    canonical_sha256,
    file_record,
    judgment_cache_key,
    region_bbox_xyxy_original,
    render_prompt,
    sha256_file,
)
from tools.verify_stageb_fixed_gdino_top1_vlm_results import (
    ACCEPTED_PAIR_SCHEMA,
    AUDIT_KIND,
    EXPECTED_FORWARD_CONTRACT,
    EXTRACTION_AUDIT_KIND,
    EXTRACTION_AUDIT_SCHEMA,
    VerificationError,
    verify,
)


class StageBFixedGDINOTop1ResultsTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows):
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _fixture(self, root: Path, *, union=False):
        image_path = root / "image.jpg"
        Image.new("RGB", (100, 80), color=(100, 110, 120)).save(image_path)
        checkpoint = root / "checkpoint.pth"
        checkpoint.write_bytes(b"fixed checkpoint")
        completion = root / "completion.json"
        completion.write_text('{"status":"complete"}\n', encoding="utf-8")
        config = root / "config.py"
        config.write_text("value = 1\n", encoding="utf-8")
        data_config = root / "data_config.py"
        data_config.write_text("data = 1\n", encoding="utf-8")

        def config_record(path: Path):
            record = file_record(path)
            chain = [file_record(path)]
            record.update(
                import_chain=chain,
                import_chain_sha256=canonical_sha256(chain),
            )
            return record

        def static_contract(name: str):
            contract = {
                "schema": f"unit-{name}-transform-v1",
                "image_set": "train" if name == "train" else "val",
                "fix_size": name == "train",
                "hflip_prob": 0.0,
            }
            contract["canonical_json"] = json.dumps(
                contract, sort_keys=True, separators=(",", ":")
            )
            contract["sha256"] = canonical_sha256(contract)
            return contract

        def row_transform(contract, output_hw):
            trace = {
                "schema": "stage-b-image-affine-trace-v1",
                "original_hw": [80, 100],
                "output_hw": output_hw,
                "scale_xy": [1.0, 1.0],
                "offset_xy": [0.0, 0.0],
                "operations": [],
            }
            trace["canonical_json"] = json.dumps(
                trace, sort_keys=True, separators=(",", ":")
            )
            trace["sha256"] = canonical_sha256(trace)
            value = {
                "image_set": contract["image_set"],
                "fix_size": contract["fix_size"],
                "hflip_prob": 0.0,
                "output_hw": output_hw,
                "amp": True,
                "dtype": "float16",
                "static_contract_sha256": contract["sha256"],
                "affine_trace": trace,
            }
            value["canonical_json"] = json.dumps(
                value, sort_keys=True, separators=(",", ":")
            )
            value["sha256"] = canonical_sha256(value)
            return value

        target_judgment = {
            "answer": "no",
            "confidence": 0.95,
            "short_reason": "wrong color",
        }
        proposal_judgment = {
            "answer": "no",
            "confidence": 0.95,
            "short_reason": "wrong object",
        }
        raw_source = {
            "dataset": "refcocoplus",
            "pair_source": "refcoco+_unc",
            "split": "train",
            "image_id": 1,
            "ann_id": 2,
            "ref_id": 3,
            "sent_id": 4,
            "target_bbox_used": [10.0, 10.0, 30.0, 40.0],
            "visual_local_judgment": target_judgment,
            "proposal_cache": [
                {"proposal_id": 0, "bbox": [5.0, 5.0, 20.0, 20.0]}
            ],
            "visual_proposal_judgments": [
                {"proposal_id": 0, "judgment": proposal_judgment}
            ],
            "proposal_num": 1,
            "candidate_cache_version": "v1",
            "visual_filter_status": "accept",
            "visual_filter_reason": "verified_negative",
            "tn_scope": "image_global_proposal_verified",
            "global_tn_verified": True,
        }
        raw_path = root / "raw_verified.jsonl"
        self._write_jsonl(raw_path, [raw_source])
        raw_hash = canonical_sha256(raw_source)

        pair = {
            "adapter_pair_schema": "stage-b-gdino-adapter-semantic-verified-pair-v1",
            "sample_id": "sample-1",
            "dataset": "refcocoplus",
            "split": "train",
            "image_id": 1,
            "ann_id": 2,
            "ref_id": 3,
            "sent_id": 4,
            "sent": "the red car",
            "try_tn": "the blue car",
            "target_bbox_used": [10.0, 10.0, 30.0, 40.0],
            "tn_scope": "image_global_topk_verified",
            "global_tn_verified": True,
            "proposalset_proxy_verified": False,
            "cached_proposal_coverage_only": True,
            "all_900_gdino_queries_verified": False,
            "global_max_label_is_semantic_extrapolation": True,
            "source_file": str(raw_path.resolve()),
            "source_line": 1,
            "source_row_sha256": raw_hash,
        }
        pair_path = root / "pairs.jsonl"
        self._write_jsonl(pair_path, [pair])

        primary = {
            "region_id": "primary-region",
            "origins": ["deploy", "primary", "shadow"] if not union else ["primary"],
            "query_ids": [10],
            "bbox_xyxy_original": [10.0, 10.0, 40.0, 50.0],
            "max_overlap": {
                "kind": "target",
                "proposal_id": None,
                "iou": 1.0,
                "source_answer": "NO",
                "source_confidence": 0.95,
                "source_judgment_sha256": canonical_sha256(target_judgment),
            },
            "inherit_eligible": True,
            "judgment": {"status": "pending", "cache_key": None},
        }
        regions = [primary]
        queries = {
            "primary": {"query_id": 10},
            "shadow": {"query_id": 10},
            "deploy": {"query_id": 10},
        }
        stability = {
            "epsilon": 0.001,
            "primary_shadow_agree": True,
            "primary_deploy_agree": True,
            "query_ids_by_origin": {"primary": 10, "shadow": 10, "deploy": 10},
            "near_tie_query_ids": [],
        }
        if union:
            regions.append(
                {
                    "region_id": "shadow-region",
                    "origins": ["deploy", "near_tie", "shadow"],
                    "query_ids": [11],
                    "bbox_xyxy_original": [55.0, 10.0, 90.0, 60.0],
                    "max_overlap": {
                        "kind": "target",
                        "proposal_id": None,
                        "iou": 0.0,
                        "source_answer": "NO",
                        "source_confidence": 0.95,
                        "source_judgment_sha256": canonical_sha256(target_judgment),
                    },
                    "inherit_eligible": False,
                    "judgment": {"status": "pending", "cache_key": None},
                }
            )
            queries["shadow"] = {"query_id": 11}
            queries["deploy"] = {"query_id": 11}
            stability = {
                "epsilon": 0.001,
                "primary_shadow_agree": False,
                "primary_deploy_agree": False,
                "query_ids_by_origin": {"primary": 10, "shadow": 11, "deploy": 11},
                "near_tie_query_ids": [11],
            }
        train_contract = static_contract("train")
        deploy_contract = static_contract("deploy")
        model_config_record = config_record(config)
        data_config_record = config_record(data_config)
        extraction = {
            "schema": EXTRACTION_SCHEMA,
            "sample_id": "sample-1",
            "dataset": "refcocoplus",
            "split": "train",
            "image_id": 1,
            "ann_id": 2,
            "ref_id": 3,
            "sent_id": 4,
            "source_pair": {
                "path": str(pair_path),
                "line": 1,
                "sha256": sha256_file(pair_path),
                "row_sha256": canonical_sha256(pair),
                "sample_id": "sample-1",
            },
            "source_verified_row": {
                "path": str(raw_path),
                "line": 1,
                "sha256": sha256_file(raw_path),
                "row_sha256": raw_hash,
            },
            "positive_expression": "the red car",
            "negative_expression": "the blue car",
            "negative_caption_model": "the blue car .",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "model_sha256": "1" * 64,
                "base_sha256": "2" * 64,
                "rank_sha256": None,
                "confidence_sha256": None,
                "protocol_train_complete": file_record(completion),
                "size_bytes": checkpoint.stat().st_size,
            },
            "config": model_config_record,
            "data_config": data_config_record,
            "code_sha256": "5" * 64,
            "transform": row_transform(train_contract, [800, 1333]),
            "deploy_transform": row_transform(deploy_contract, [800, 1000]),
            "forward_contract": dict(EXPECTED_FORWARD_CONTRACT),
            "image": {
                "path": str(image_path),
                "width": 100,
                "height": 80,
                "sha256": sha256_file(image_path),
            },
            "num_queries": 900,
            "valid_query_count": 900,
            **queries,
            "stability": stability,
            "regions": regions,
            "source_verification": dict(raw_source),
            "claims": {
                "frozen_gdino_global_max_regions_extracted": True,
                "train_path_and_deploy_transform_regions_extracted": True,
                "all_900_gdino_queries_verified": False,
                "image_global_semantic_absence_proven": False,
                "portable_to_other_checkpoint_or_transform": False,
            },
            "_test_static_contracts": {
                "train": train_contract,
                "deploy": deploy_contract,
            },
        }
        for region in regions:
            region["assets"] = build_boxed_assets(
                extraction,
                region,
                asset_root=root / "assets",
                cache_key=canonical_sha256(
                    {"sample_id": extraction["sample_id"], "region_id": region["region_id"]}
                ),
            )
        return extraction

    def _judgment(self, root: Path, extraction, region, answer, confidence=0.95):
        key = judgment_cache_key(extraction, region)
        assets = dict(region["assets"])
        raw = json.dumps(
            {
                "answer": answer,
                "confidence": confidence,
                "short_reason": "unit-test evidence",
            }
        )
        prompt = render_prompt(extraction["negative_expression"])
        return {
            "schema": JUDGMENT_SCHEMA,
            "extraction_schema": EXTRACTION_SCHEMA,
            "sample_id": extraction["sample_id"],
            "identity": {
                key: extraction[key]
                for key in (
                    "sample_id",
                    "dataset",
                    "split",
                    "image_id",
                    "ann_id",
                    "ref_id",
                    "sent_id",
                )
            },
            "region_id": region["region_id"],
            "cache_key": key,
            "status": "complete",
            "answer": answer,
            "confidence": confidence,
            "short_reason": "unit-test evidence",
            "raw_output": raw,
            "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "error": None,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "prompt": {
                "template_sha256": PROMPT_TEMPLATE_SHA256,
                "rendered_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            },
            "asset_policy_sha256": ASSET_POLICY_SHA256,
            "generation_config": GENERATION_CONFIG,
            "generation_config_sha256": GENERATION_CONFIG_SHA256,
            "vision_processor_config": VISION_PROCESSOR_CONFIG,
            "vision_processor_config_sha256": VISION_PROCESSOR_CONFIG_SHA256,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "judge_runtime_policy": JUDGE_RUNTIME_POLICY,
            "judge_runtime_policy_sha256": JUDGE_RUNTIME_POLICY_SHA256,
            "bbox_xyxy_original": region_bbox_xyxy_original(region),
            "negative_expression": extraction["negative_expression"],
            "assets": assets,
            "runtime": {"test": True},
            "created_at_utc": "2026-07-12T00:00:00+00:00",
        }

    def _run(
        self,
        root: Path,
        extraction,
        judgments,
        *,
        strict_image=999,
        extraction_audit_mutator=None,
    ):
        extraction_path = root / "extractions.jsonl"
        extraction_audit_path = root / "extraction_audit.json"
        judgment_path = root / "judgments.jsonl"
        strict2031 = root / "strict2031.jsonl"
        strict1607 = root / "strict1607.jsonl"
        self._write_jsonl(strict2031, [{"image_id": strict_image}])
        self._write_jsonl(strict1607, [{"image_id": strict_image}])
        extraction["holdout"] = {
            "strict2031_manifest_sha256": sha256_file(strict2031),
            "strict1607_manifest_sha256": sha256_file(strict1607),
            "image_disjoint": True,
        }
        contracts = extraction.pop("_test_static_contracts")
        self._write_jsonl(extraction_path, [extraction])
        self._write_jsonl(judgment_path, judgments)
        manifest_record = {**file_record(extraction_path), "rows": 1}
        strict2031_record = {
            **file_record(strict2031),
            "rows": 1,
            "unique_images": 1,
        }
        strict1607_record = {
            **file_record(strict1607),
            "rows": 1,
            "unique_images": 1,
        }
        extraction_audit = {
            "schema": EXTRACTION_AUDIT_SCHEMA,
            "kind": EXTRACTION_AUDIT_KIND,
            "rows": 1,
            "regions": len(extraction["regions"]),
            "counts": {"rows": 1, "regions": len(extraction["regions"])},
            "manifest": manifest_record,
            "output": {
                key: manifest_record[key]
                for key in ("path", "sha256", "size_bytes")
            },
            "checkpoint": extraction["checkpoint"],
            "model_config": extraction["config"],
            "data_config": extraction["data_config"],
            "code": {"code_sha256": extraction["code_sha256"]},
            "transform_contracts": contracts,
            "holdout": {
                "manifests": {
                    "strict2031": strict2031_record,
                    "strict1607": strict1607_record,
                }
            },
            "runtime": {"tie_epsilon": 0.001},
            "claims": extraction["claims"],
        }
        if extraction_audit_mutator is not None:
            extraction_audit_mutator(extraction_audit)
        extraction_audit_path.write_text(
            json.dumps(extraction_audit, sort_keys=True) + "\n", encoding="utf-8"
        )
        args = argparse.Namespace(
            extractions=str(extraction_path),
            extraction_audit=str(extraction_audit_path),
            judgments=str(judgment_path),
            accepted_output=str(root / "accepted.jsonl"),
            rejected_output=str(root / "rejected.jsonl"),
            quarantine_output=str(root / "quarantine.jsonl"),
            audit=str(root / "audit.json"),
            strict2031=str(strict2031),
            strict1607=str(strict1607),
            expected_strict2031_sha256=sha256_file(strict2031),
            expected_strict1607_sha256=sha256_file(strict1607),
        )
        with mock.patch(
            "tools.verify_stageb_fixed_gdino_top1_vlm_results.EXPECTED_EXTRACTION_ROWS",
            1,
        ):
            return verify(args)

    def test_high_iou_source_no_is_inherited_and_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self._fixture(root)
            audit = self._run(root, extraction, [])
            accepted = json.loads((root / "accepted.jsonl").read_text())
        self.assertEqual(audit["decisions"]["accepted"], 1)
        self.assertEqual(audit["kind"], AUDIT_KIND)
        self.assertEqual(accepted["adapter_pair_schema"], ACCEPTED_PAIR_SCHEMA)
        self.assertIs(accepted["fixed_gdino_global_max_verified"], True)
        self.assertIs(accepted["all_900_gdino_queries_verified"], False)
        self.assertIs(accepted["portable_to_other_checkpoint_or_transform"], False)
        self.assertIs(accepted["global_max_label_is_semantic_extrapolation"], False)
        self.assertEqual(
            accepted["frozen_gdino_train_transform_contract_sha256"],
            audit["train_transform_contract_sha256"],
        )
        self.assertEqual(
            accepted["frozen_gdino_deploy_transform_contract_sha256"],
            audit["deploy_transform_contract_sha256"],
        )
        self.assertEqual(audit["inputs"]["extractions"]["rows"], 1)
        self.assertEqual(audit["inputs"]["judgments"]["rows"], 0)
        self.assertEqual(
            audit["locked_contract"]["judge_runtime_policy"],
            JUDGE_RUNTIME_POLICY,
        )
        self.assertEqual(
            audit["locked_contract"]["judge_runtime_policy_sha256"],
            JUDGE_RUNTIME_POLICY_SHA256,
        )
        self.assertIs(audit["scope"]["global_max_label_is_semantic_extrapolation"], False)

    def test_qwen_runtime_policy_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self._fixture(root, union=True)
            judgment = self._judgment(
                root, extraction, extraction["regions"][1], "NO"
            )
            judgment["judge_runtime_policy"] = {
                **JUDGE_RUNTIME_POLICY,
                "implementation_revision": "drifted",
            }
            with self.assertRaisesRegex(VerificationError, "runtime policy drifted"):
                self._run(root, extraction, [judgment])

    def test_every_shadow_near_tie_region_must_be_no(self):
        for answer, expected in (("NO", "accepted"), ("UNKNOWN", "quarantine"), ("YES", "rejected")):
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                extraction = self._fixture(root, union=True)
                judgment = self._judgment(
                    root, extraction, extraction["regions"][1], answer
                )
                audit = self._run(root, extraction, [judgment])
                self.assertEqual(audit["decisions"][expected], 1)

    def test_missing_noninherited_region_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self._fixture(root, union=True)
            audit = self._run(root, extraction, [])
        self.assertEqual(audit["decisions"]["quarantine"], 1)
        self.assertEqual(audit["row_reason_counts"]["missing_qwen_judgment"], 1)

    def test_strict_image_overlap_fails_the_whole_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self._fixture(root)
            with self.assertRaisesRegex(VerificationError, "not strict-image-disjoint"):
                self._run(root, extraction, [], strict_image=1)

    def test_asset_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self._fixture(root, union=True)
            judgment = self._judgment(
                root, extraction, extraction["regions"][1], "NO"
            )
            Path(judgment["assets"]["full_boxed"]["path"]).write_bytes(b"drift")
            with self.assertRaisesRegex(VerificationError, "asset hash drifted"):
                self._run(root, extraction, [judgment])

    def test_inherited_region_qwen_audit_overrides_source_no(self):
        for answer, expected_comparison, expected_decision in (
            ("NO", "agree_no", "accepted"),
            ("YES", "conflict_yes", "rejected"),
        ):
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                extraction = self._fixture(root)
                judgment = self._judgment(
                    root, extraction, extraction["regions"][0], answer
                )
                audit = self._run(root, extraction, [judgment])
                self.assertEqual(audit["decisions"][expected_decision], 1)
                self.assertEqual(
                    audit["source_qwen_comparison_counts"][expected_comparison], 1
                )

    def test_deploy_query_cannot_be_omitted_from_union(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self._fixture(root, union=True)
            extraction["shadow"]["query_id"] = 10
            extraction["stability"]["primary_shadow_agree"] = True
            extraction["stability"]["query_ids_by_origin"]["shadow"] = 10
            extraction["regions"][0]["origins"].append("shadow")
            extraction["regions"] = extraction["regions"][:1]
            with self.assertRaisesRegex(VerificationError, "deploy query is absent"):
                self._run(root, extraction, [])

    def test_extraction_audit_manifest_count_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self._fixture(root)

            def corrupt(audit):
                audit["manifest"]["rows"] = 0

            with self.assertRaisesRegex(
                VerificationError, "manifest record drifted"
            ):
                self._run(
                    root,
                    extraction,
                    [],
                    extraction_audit_mutator=corrupt,
                )

    def test_extraction_audit_tie_epsilon_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self._fixture(root)

            def corrupt(audit):
                audit["runtime"]["tie_epsilon"] = 1.0e-7

            with self.assertRaisesRegex(VerificationError, "tie epsilon drifted"):
                self._run(
                    root,
                    extraction,
                    [],
                    extraction_audit_mutator=corrupt,
                )


if __name__ == "__main__":
    unittest.main()
