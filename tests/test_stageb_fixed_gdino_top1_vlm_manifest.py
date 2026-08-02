import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from models.GroundingDINO.stage_b_gdino_score_adapter import (
    aggregate_gdino_full_expression_score,
)
from tools.extract_stageb_fixed_gdino_top1_vlm_manifest import (
    CLAIMS,
    ASSET_POLICY,
    EXPECTED_QUERIES,
    ExtractionError,
    SCHEMA,
    TraceableConfidenceTrainTransform,
    TraceableDeployEvalTransform,
    aggregate_full_expression_scores,
    build_regions,
    canonical_sha256,
    deploy_transform_contract_from_cfg,
    exclude_holdout_image_union,
    forward_paired_pos_neg,
    forward_separate,
    generate_region_assets,
    inverse_transform_box,
    make_query_observation,
    sha256_file,
    source_max_overlap,
    summarize_stability,
    transform_contract_from_cfg,
    validate_manifest_row,
    validate_strict_manifests,
    _checkpoint_provenance,
    _code_provenance,
    _manifest_row,
)


class _Nested:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask


class _Model:
    def __init__(self):
        self.calls = []

    def __call__(self, samples, *, captions):
        self.calls.append(list(captions))
        batch = len(captions)
        logits = torch.full((batch, EXPECTED_QUERIES, 3), -5.0)
        boxes = torch.full((batch, EXPECTED_QUERIES, 4), 0.5)
        boxes[..., 2:] = 0.2
        mask = torch.zeros((batch, 1, 3), dtype=torch.bool)
        mask[..., :2] = True
        for index in range(batch):
            logits[index, index, :2] = 5.0
        return {
            "pred_logits_text": logits,
            "pred_boxes": boxes,
            "phrase_to_token_mask": mask,
        }


def _identity_trace():
    return {
        "original_hw": [100, 200],
        "output_hw": [100, 200],
        "scale_xy": [1.0, 1.0],
        "offset_xy": [0.0, 0.0],
    }


def _scores(best=0, second=1, gap=0.1):
    values = torch.zeros(EXPECTED_QUERIES)
    values[best] = 0.9
    values[second] = 0.9 - gap
    return values


def _boxes():
    result = torch.full((EXPECTED_QUERIES, 4), 0.5)
    result[:, 2:] = 0.2
    result[1] = torch.tensor([0.2, 0.2, 0.1, 0.1])
    return result


class StageBFixedGdinoTop1VlmManifestTest(unittest.TestCase):
    def test_code_provenance_recursively_binds_formal_paths_and_native_extension(self):
        code = _code_provenance()
        records = code["files"]
        by_relative_path = {}
        for record in records:
            path = Path(record["path"])
            try:
                relative = path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
            except ValueError:
                continue
            by_relative_path[relative] = record
        for relative in (
            "engine.py",
            "tools/eval_text_groundingdino_refcoco_tn.py",
            "tools/run_stageb_fixed_protocol_eval.sh",
            "models/GroundingDINO/transformer.py",
            "models/GroundingDINO/bertwarper.py",
            "models/GroundingDINO/ms_deform_attn.py",
        ):
            self.assertIn(relative, by_relative_path)
        self.assertEqual(
            by_relative_path["tools/run_stageb_fixed_protocol_eval.sh"]["kind"],
            "orchestration",
        )
        extensions = [
            record for record in records if record["kind"] == "native_extension"
        ]
        self.assertEqual(len(extensions), 1)
        self.assertEqual(extensions[0]["module"], "MultiScaleDeformableAttention")
        self.assertTrue(Path(extensions[0]["path"]).name.endswith(".so"))
        self.assertEqual(extensions[0]["sha256"], sha256_file(Path(extensions[0]["path"])))
        self.assertEqual(code["code_sha256"], canonical_sha256(records))

    def test_checkpoint_provenance_fails_on_training_preflight_drift_first(self):
        from tools.stageb_fixed_protocol_audit import ProtocolError

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint0000.pth"
            checkpoint.write_bytes(b"not-read-after-preflight-failure")
            with mock.patch(
                "tools.stageb_fixed_protocol_audit._verify_train_completion",
                side_effect=ProtocolError("training preflight dependency drift"),
            ), mock.patch(
                "tools.stageb_gdino_adapter_probe_audit._validate_fixed_baseline"
            ) as validate:
                with self.assertRaisesRegex(
                    ExtractionError, "training preflight dependency drift"
                ):
                    _checkpoint_provenance(checkpoint)
            validate.assert_not_called()

    def test_checkpoint_provenance_keeps_complete_file_record_after_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint0000.pth"
            checkpoint.write_bytes(b"checkpoint")
            completion = root / "protocol_train_complete.json"
            completion.write_text('{"kind":"complete"}\n', encoding="utf-8")
            completion_record = {
                "path": str(completion.resolve()),
                "size_bytes": completion.stat().st_size,
                "sha256": sha256_file(completion),
            }
            baseline_record = {
                "sha256": sha256_file(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "base_model_sha256": "a" * 64,
                "protocol_train_complete": completion_record,
            }
            with mock.patch(
                "tools.stageb_fixed_protocol_audit._verify_train_completion"
            ) as verify, mock.patch(
                "tools.stageb_gdino_adapter_probe_audit._validate_fixed_baseline",
                return_value=baseline_record,
            ) as validate:
                observed = _checkpoint_provenance(checkpoint)
            verify.assert_called_once_with(
                checkpoint.resolve().parent, checkpoint=checkpoint.resolve()
            )
            validate.assert_called_once_with(checkpoint.resolve())
            self.assertEqual(observed["protocol_train_complete"], completion_record)

    def test_score_matches_authoritative_float32_aggregation(self):
        torch.manual_seed(3)
        logits = torch.randn(2, EXPECTED_QUERIES, 7, dtype=torch.float16)
        phrase_mask = torch.zeros((2, 3, 7), dtype=torch.bool)
        phrase_mask[0, 0, [1, 3]] = True
        phrase_mask[0, 2, [5]] = True
        phrase_mask[1, 1, [0, 6]] = True
        observed = aggregate_full_expression_scores(logits, phrase_mask)
        expected = aggregate_gdino_full_expression_score(
            logits, phrase_mask.any(dim=1)
        )
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(observed.dtype, torch.float32)

    def test_score_fails_closed_on_query_count_empty_mask_and_nonfinite(self):
        with self.assertRaisesRegex(ExtractionError, "exactly 900"):
            aggregate_full_expression_scores(
                torch.zeros(1, 899, 2), torch.ones(1, 1, 2, dtype=torch.bool)
            )
        with self.assertRaisesRegex(ExtractionError, "at least one"):
            aggregate_full_expression_scores(
                torch.zeros(1, EXPECTED_QUERIES, 2),
                torch.zeros(1, 1, 2, dtype=torch.bool),
            )
        logits = torch.zeros(1, EXPECTED_QUERIES, 2)
        logits[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ExtractionError, "non-finite"):
            aggregate_full_expression_scores(
                logits, torch.ones(1, 1, 2, dtype=torch.bool)
            )

    def test_paired_forward_is_exact_pos_then_neg_single_2b_call(self):
        model = _Model()
        samples = _Nested(
            torch.zeros(4, 3, 8, 8), torch.zeros(4, 8, 8, dtype=torch.bool)
        )
        positives = [f"positive {index} ." for index in range(4)]
        negatives = [f"negative {index} ." for index in range(4)]
        positive, negative = forward_paired_pos_neg(
            model, samples, positives, negatives, amp=False
        )
        self.assertEqual(model.calls, [positives + negatives])
        self.assertEqual(tuple(positive["pred_logits_text"].shape), (4, 900, 3))
        self.assertEqual(tuple(negative["pred_logits_text"].shape), (4, 900, 3))
        self.assertEqual(
            int(positive["pred_logits_text"][0, :, 0].argmax().item()), 0
        )
        self.assertEqual(
            int(negative["pred_logits_text"][0, :, 0].argmax().item()), 4
        )

    def test_deploy_forward_is_separate_b16_negative_then_positive(self):
        model = _Model()
        samples = _Nested(
            torch.zeros(16, 3, 8, 8), torch.zeros(16, 8, 8, dtype=torch.bool)
        )
        negatives = [f"negative {index} ." for index in range(16)]
        positives = [f"positive {index} ." for index in range(16)]
        negative = forward_separate(model, samples, negatives, amp=False)
        positive = forward_separate(model, samples, positives, amp=False)
        self.assertEqual(model.calls, [negatives, positives])
        self.assertEqual(tuple(negative["pred_boxes"].shape), (16, 900, 4))
        self.assertEqual(tuple(positive["pred_boxes"].shape), (16, 900, 4))

    def test_crop_resize_trace_inverts_to_original_coordinates(self):
        # Original -> 2x resize -> crop(left=30, top=20) -> 0.5x resize.
        trace = {
            "original_hw": [100, 200],
            "output_hw": [80, 100],
            "scale_xy": [1.0, 1.0],
            "offset_xy": [-15.0, -10.0],
        }
        # Transformed xyxy is [25,20,75,60], hence original [40,30,90,70].
        geometry = inverse_transform_box([0.5, 0.5, 0.5, 0.5], trace)
        self.assertEqual(geometry["bbox_xyxy_transformed"], [25.0, 20.0, 75.0, 60.0])
        self.assertEqual(geometry["bbox_xyxy_original"], [40.0, 30.0, 90.0, 70.0])
        self.assertEqual(geometry["bbox_xywh_original"], [40.0, 30.0, 50.0, 40.0])

    def test_traceable_train_transform_is_seed_replayable(self):
        from datasets.coco import make_coco_transforms

        cfg = SimpleNamespace(
            fix_size=True,
            strong_aug=False,
            data_aug_hflip_prob=0.0,
        )
        contract = transform_contract_from_cfg(cfg)
        transform = TraceableConfidenceTrainTransform(contract)
        image = Image.new("RGB", (200, 120), color=(40, 80, 120))
        target = {
            "boxes": torch.tensor([[20.0, 10.0, 100.0, 80.0]]),
            "labels": torch.tensor([1]),
            "orig_size": torch.tensor([120, 200]),
            "size": torch.tensor([120, 200]),
        }
        random.seed(42)
        torch.manual_seed(42)
        first_image, first_target = transform(image, dict(target))
        random.seed(42)
        torch.manual_seed(42)
        second_image, second_target = transform(image, dict(target))
        self.assertTrue(torch.equal(first_image, second_image))
        self.assertEqual(
            first_target["_stageb_extraction_transform_trace"],
            second_target["_stageb_extraction_transform_trace"],
        )
        trace = first_target["_stageb_extraction_transform_trace"]
        self.assertEqual(trace["output_hw"], [800, 1333])
        self.assertEqual(trace["operations"][-1], {"op": "fixed_resize", "size_wh": [1333, 800]})
        random.seed(42)
        torch.manual_seed(42)
        standard_image, standard_target = make_coco_transforms(
            "train", fix_size=True, strong_aug=False, args=cfg
        )(image, dict(target))
        self.assertTrue(torch.equal(first_image, standard_image))
        self.assertTrue(torch.equal(first_target["boxes"], standard_target["boxes"]))

    def test_near_tie_and_shadow_flip_produce_query_union(self):
        boxes = _boxes()
        # One float32 ULP at this magnitude: distinct argmax, still inside epsilon.
        primary_scores = _scores(gap=8.0e-8)
        shadow_scores = _scores(best=1, second=0, gap=8.0e-8)
        primary = make_query_observation(
            primary_scores,
            boxes,
            trace=_identity_trace(),
            target_bbox_xywh=[80.0, 40.0, 40.0, 20.0],
            tie_epsilon=1.0e-7,
        )
        shadow = make_query_observation(
            shadow_scores,
            boxes,
            trace=_identity_trace(),
            target_bbox_xywh=[80.0, 40.0, 40.0, 20.0],
            tie_epsilon=1.0e-7,
        )
        observations = {"primary": primary, "shadow": shadow, "deploy": primary}
        stability = summarize_stability(observations, epsilon=1.0e-7)
        self.assertFalse(stability["primary_shadow_agree"])
        self.assertFalse(stability["exact_top1_stable"])
        self.assertEqual(stability["near_tie_query_ids"], [0, 1])
        regions = build_regions(
            "sample", observations, {"target_bbox_used": [80, 40, 40, 20]}
        )
        self.assertEqual(sorted({q for region in regions for q in region["query_ids"]}), [0, 1])

    def test_argmax_uses_first_index_on_exact_tie(self):
        scores = _scores(gap=0.0)
        observation = make_query_observation(
            scores,
            _boxes(),
            trace=_identity_trace(),
            target_bbox_xywh=[1, 1, 10, 10],
            tie_epsilon=0.0,
        )
        self.assertEqual(observation["summary"]["query_id"], 0)
        self.assertEqual(observation["near_tie_query_ids"], [1])

    def test_deploy_transform_query_disagreement_is_in_region_union(self):
        boxes = _boxes()
        boxes[2] = torch.tensor([0.8, 0.7, 0.1, 0.15])
        primary = make_query_observation(
            _scores(best=0, second=1, gap=0.2),
            boxes,
            trace=_identity_trace(),
            target_bbox_xywh=[10, 10, 20, 20],
        )
        deploy = make_query_observation(
            _scores(best=2, second=3, gap=0.2),
            boxes,
            trace=_identity_trace(),
            target_bbox_xywh=[10, 10, 20, 20],
        )
        observations = {"primary": primary, "shadow": primary, "deploy": deploy}
        stability = summarize_stability(observations)
        self.assertFalse(stability["primary_deploy_agree"])
        regions = build_regions(
            "sample", observations, {"target_bbox_used": [10, 10, 20, 20]}
        )
        self.assertEqual(
            sorted({query for region in regions for query in region["query_ids"]}),
            [0, 2],
        )
        deploy_regions = [region for region in regions if "deploy" in region["origins"]]
        self.assertTrue(any(2 in region["query_ids"] for region in deploy_regions))

    def test_source_overlap_is_bound_to_exact_old_judgment(self):
        source = {
            "target_bbox_used": [10, 20, 30, 40],
            "visual_local_judgment": {"answer": "no", "confidence": 0.95},
            "proposal_cache": [{"proposal_id": 7, "bbox": [100, 100, 20, 20]}],
            "visual_proposal_judgments": [
                {"proposal_id": 7, "judgment": {"answer": "no", "confidence": 0.8}}
            ],
        }
        overlap = source_max_overlap([10, 20, 30, 40], source)
        self.assertEqual(overlap["kind"], "target")
        self.assertEqual(overlap["source_answer"], "no")
        self.assertAlmostEqual(overlap["iou"], 1.0)
        self.assertEqual(len(overlap["source_judgment_sha256"]), 64)

    def test_assets_are_deterministic_and_cover_three_views(self):
        region = {
            "region_id": "a" * 64,
            "bbox_xyxy_original": [3.2, 4.1, 20.8, 18.2],
        }
        image = Image.new("RGB", (32, 24), color=(120, 140, 160))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = generate_region_assets(
                image,
                region,
                temporary_root=root / "tmp1",
                final_root=root / "final1",
                sample_key="sample",
            )
            second = generate_region_assets(
                image,
                region,
                temporary_root=root / "tmp2",
                final_root=root / "final2",
                sample_key="sample",
            )
            for key in ("tight", "context_2x_boxed", "full_boxed"):
                self.assertEqual(first[key]["sha256"], second[key]["sha256"])
                self.assertEqual(len(first[key]["sha256"]), 64)
            self.assertEqual(first["raster_bbox_xyxy"], [3, 4, 21, 19])

    def test_holdout_filter_excludes_by_image_union_before_batching(self):
        bindings = [
            {"pair": {"image_id": 1, "dataset": "refcocoplus"}},
            {"pair": {"image_id": 2, "dataset": "refcocog"}},
            {"pair": {"image_id": 1, "dataset": "refcocog"}},
        ]
        kept, stats = exclude_holdout_image_union(
            bindings, {1}, enforce_frozen_counts=False
        )
        self.assertEqual([row["pair"]["image_id"] for row in kept], [2])
        self.assertEqual(stats["excluded_rows"], 2)
        self.assertEqual(stats["excluded_unique_images"], 1)

    def test_locked_strict_manifest_image_union_verifies(self):
        records, images = validate_strict_manifests()
        self.assertEqual(len(images), 1045)
        self.assertEqual(records["strict2031"]["rows"], 2031)
        self.assertEqual(records["strict1607"]["rows"], 1607)

    def test_manifest_row_requires_pending_regions_and_no_holdout_leak(self):
        asset = {"path": "/tmp/asset", "sha256": "0" * 64}
        transform = {"image_set": "train"}
        transform["canonical_json"] = "unused-but-hash-bound"
        transform["sha256"] = canonical_sha256(transform)
        deploy_transform = {"image_set": "val"}
        deploy_transform["canonical_json"] = "unused-but-hash-bound"
        deploy_transform["sha256"] = canonical_sha256(deploy_transform)
        row = {
            "schema": SCHEMA,
            "sample_id": "sample",
            "image_id": 10,
            "num_queries": EXPECTED_QUERIES,
            "valid_query_count": EXPECTED_QUERIES,
            "score_contract": "float32_mean_sigmoid_over_generated_full_expression_tokens",
            "forward_contract": {
                "primary": "confidence_train_paired_2b_pos_then_neg_local_b4",
                "shadow": "formal_eval_separate_negative_then_positive_local_b4",
                "deploy": "formal_eval_val_resize_separate_negative_then_positive_b16",
                "local_batch_size": 4,
                "paired_batch_size": 8,
                "deploy_batch_size": 16,
            },
            "transform": transform,
            "deploy_transform": deploy_transform,
            "primary": {"query_id": 1},
            "shadow": {"query_id": 1},
            "deploy": {"query_id": 1},
            "positive_primary": {"query_id": 1},
            "positive_shadow": {"query_id": 1},
            "positive_deploy": {"query_id": 1},
            "regions": [
                {
                    "region_id": canonical_sha256("region"),
                    "query_ids": [1],
                    "judgment": {"status": "pending", "cache_key": None},
                    "assets": {
                        "tight": asset,
                        "context_2x_boxed": asset,
                        "full_boxed": asset,
                    },
                }
            ],
            "proposal_num": 1,
            "stability": {
                "primary_shadow_agree": True,
                "primary_deploy_agree": True,
                "epsilon": 1.0e-3,
            },
            "holdout": {"image_disjoint": True},
            "claims": dict(CLAIMS),
        }
        validate_manifest_row(row, excluded_images={11})
        with self.assertRaisesRegex(ExtractionError, "leakage"):
            validate_manifest_row(row, excluded_images={10})
        row["regions"][0]["judgment"] = {"status": "accepted"}
        with self.assertRaisesRegex(ExtractionError, "not pending"):
            validate_manifest_row(row)

    def test_actual_manifest_row_schema_binds_raw_file_and_deploy(self):
        from tools.judge_stageb_fixed_gdino_top1_qwen import validate_extraction_row
        from tools.verify_stageb_fixed_gdino_top1_vlm_results import (
            _claims,
            _validate_region_union,
        )

        observation = make_query_observation(
            _scores(),
            _boxes(),
            trace=_identity_trace(),
            target_bbox_xywh=[80, 40, 40, 20],
        )
        source = {
            "target_bbox_used": [80, 40, 40, 20],
            "visual_local_judgment": {"answer": "no", "confidence": 0.95},
            "proposal_cache": [{"proposal_id": 0, "bbox": [10, 10, 20, 20]}],
            "visual_proposal_judgments": [
                {"proposal_id": 0, "judgment": {"answer": "no", "confidence": 0.95}}
            ],
            "proposal_num": 1,
            "visual_filter_status": "accept",
            "visual_filter_reason": "verified_negative",
            "tn_scope": "image_global_proposal_verified",
            "global_tn_verified": True,
        }
        pair = {
            "sample_id": "sample",
            "dataset": "refcocoplus",
            "image_id": 1,
            "ann_id": 2,
            "ref_id": 3,
            "sent_id": 4,
            "split": "train",
            "sent": "red car",
            "try_tn": "blue car",
            "target_bbox_used": [80, 40, 40, 20],
            "source_row_sha256": canonical_sha256(source),
        }
        context = {
            "binding": {
                "pair_path": Path("/tmp/pairs.jsonl"),
                "pair_line": 7,
                "pair": pair,
                "pair_row_sha256": canonical_sha256(pair),
                "source_path": Path("/tmp/raw.jsonl"),
                "source_file_sha256": "a" * 64,
                "source_line": 9,
                "source_row": source,
            },
            "trace": _identity_trace(),
            "deploy_trace": _identity_trace(),
            "positive_caption": "red car .",
            "negative_caption": "blue car .",
            "observations": {
                "primary": observation,
                "shadow": observation,
                "deploy": observation,
            },
            "positive_primary": observation,
            "positive_shadow": observation,
            "positive_deploy": observation,
        }
        asset = {"path": "/tmp/asset", "sha256": "0" * 64}
        regions = [
            {
                "region_id": "b" * 64,
                "origins": ["deploy", "primary", "shadow"],
                "query_ids": [0],
                "base_scores": {"primary:q0": 0.9},
                "bbox_xyxy_norm": [0.4, 0.4, 0.6, 0.6],
                "bbox_xyxy_original": [80, 40, 120, 60],
                "bbox_xywh_original": [80, 40, 40, 20],
                "bbox_sha256": canonical_sha256([80, 40, 40, 20]),
                "max_overlap": source_max_overlap([80, 40, 40, 20], source),
                "inherit_eligible": True,
                "assets": {
                    "asset_policy_sha256": canonical_sha256(ASSET_POLICY),
                    "tight": asset,
                    "context_2x_boxed": asset,
                    "full_boxed": asset,
                },
                "judgment": {"status": "pending", "cache_key": None},
            }
        ]
        transform_contract = {"fix_size": True, "hflip_prob": 0.0, "sha256": "c" * 64}
        deploy_contract = {"fix_size": False, "hflip_prob": 0.0, "sha256": "d" * 64}
        row = _manifest_row(
            context,
            checkpoint={"path": "/tmp/checkpoint", "sha256": "e" * 64},
            model_config={"path": "/tmp/config", "sha256": "f" * 64},
            data_config={"path": "/tmp/data-config", "sha256": "1" * 64},
            code={"code_sha256": "2" * 64},
            transform_contract=transform_contract,
            deploy_transform_contract=deploy_contract,
            holdout={"image_disjoint": True},
            image_record={"path": "/tmp/image", "width": 200, "height": 100, "sha256": "3" * 64},
            regions=regions,
            stability=summarize_stability(context["observations"]),
            amp=True,
        )
        validate_manifest_row(row)
        validate_extraction_row(row)
        _claims(row)
        _validate_region_union(row)
        self.assertEqual(row["source_pair"]["line"], 7)
        self.assertEqual(row["source_verified_row"]["sha256"], "a" * 64)
        self.assertEqual(row["deploy"]["query_id"], 0)
        self.assertEqual(row["forward_contract"]["deploy_batch_size"], 16)
        self.assertIs(row["claims"]["train_path_and_deploy_transform_regions_extracted"], True)

    def test_deploy_transform_is_aspect_preserving_val_resize(self):
        from datasets.coco import make_coco_transforms

        cfg = SimpleNamespace(data_aug_scales=[480, 800], data_aug_max_size=1333)
        contract = deploy_transform_contract_from_cfg(cfg)
        transform = TraceableDeployEvalTransform(contract)
        image = Image.new("RGB", (2000, 1000), color=(10, 20, 30))
        target = {
            "boxes": torch.tensor([[100.0, 100.0, 500.0, 500.0]]),
            "labels": torch.tensor([1]),
            "orig_size": torch.tensor([1000, 2000]),
            "size": torch.tensor([1000, 2000]),
        }
        random.seed(42)
        transformed, transformed_target = transform(image, target)
        # min-side 800 would make width 1600, capped to width 1333.
        self.assertEqual(list(transformed.shape[-2:]), [666, 1332])
        trace = transformed_target["_stageb_extraction_transform_trace"]
        self.assertEqual(trace["original_hw"], [1000, 2000])
        self.assertEqual(trace["output_hw"], [666, 1332])
        self.assertEqual(trace["offset_xy"], [0.0, 0.0])
        random.seed(42)
        standard, standard_target = make_coco_transforms(
            "val", fix_size=True, strong_aug=False, args=cfg
        )(image, dict(target))
        self.assertTrue(torch.equal(transformed, standard))
        self.assertTrue(torch.equal(transformed_target["boxes"], standard_target["boxes"]))


if __name__ == "__main__":
    unittest.main()
