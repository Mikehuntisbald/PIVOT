import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

import tools.extract_mmgdino_responsibility_cache as extractor
from tools.extract_mmgdino_responsibility_cache import (
    EXPECTED_QUERY_COUNT,
    HookBatch,
    MMGroundingDinoExtractionError,
    MMGroundingDinoHookRecorder,
    extract_cached_candidate_shard,
    parse_d3_pair_requests,
    parse_mmgdino_hook_outputs,
    parse_refcoco_rank_requests,
    validate_native_prediction_parity,
    validate_pinned_runtime_assets,
)
from tools.responsibility_isolation_cache import (
    CACHE_FEATURE_DIM,
    CACHE_SHARD_SCHEMA,
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    CachedCandidateContractError,
    cached_candidate_content_sha256,
    file_sha256,
    load_cached_candidate_shard,
    validate_cached_candidate_shard,
)


def _raw_hook_outputs():
    hidden = torch.zeros(6, 1, EXPECTED_QUERY_COUNT, CACHE_FEATURE_DIM)
    hidden[-1].fill_(5.0)
    references = torch.full((7, 1, EXPECTED_QUERY_COUNT, 4), 0.5)
    token_logits = torch.full(
        (6, 1, EXPECTED_QUERY_COUNT, 256), float("-inf")
    )
    token_logits[..., 0] = -2.0
    token_logits[-1, 0, 0, 0] = 4.0
    boxes = torch.full((6, 1, EXPECTED_QUERY_COUNT, 4), 0.1)
    boxes[..., :2] = 0.8
    boxes[-1, 0, 0] = torch.tensor([0.3, 0.3, 0.2, 0.2])
    return (hidden, references), (token_logits, boxes)


def _valid_hook_batch(*, oracle=True):
    features = torch.zeros(EXPECTED_QUERY_COUNT, CACHE_FEATURE_DIM)
    scores = torch.linspace(0.1, 0.9, EXPECTED_QUERY_COUNT)
    boxes = torch.empty(EXPECTED_QUERY_COUNT, 4)
    boxes[:] = torch.tensor([0.8, 0.8, 0.1, 0.1])
    if oracle:
        boxes[0] = torch.tensor([0.3, 0.3, 0.2, 0.2])
    return HookBatch(
        query_features=features.contiguous(),
        native_score=scores.contiguous(),
        boxes=boxes.contiguous(),
        candidate_mask=torch.ones(EXPECTED_QUERY_COUNT, dtype=torch.bool),
    )


class _FakeRuntime:
    def __init__(self, hook=None):
        self.hook = _valid_hook_batch() if hook is None else hook
        self.calls = []

    def infer(self, image_path, caption):
        self.calls.append((image_path, caption))
        return self.hook


class MMGroundingDinoExtractorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "COCO_train2014_000000000123.jpg"
        Image.new("RGB", (100, 80), color=(20, 40, 60)).save(self.image)
        self.image_sha = file_sha256(self.image)
        self.rank_row = {
            "filename": str(self.image),
            "file_name": self.image.name,
            "source": "refcoco_unc_val",
            "image_id": 123,
            "ann_id": 7,
            "ref_id": 11,
            "sent_id": 13,
            "instances": [
                {
                    "bbox": [20, 16, 20, 16],
                    "positive_phrase": "the complete referring expression",
                    "text_is_negative": False,
                }
            ],
        }
        self.d3_row = {
            "sample_id": "d3-row-1",
            "image_id": 123,
            "ann_id": 7,
            "ref_id": 11,
            "sent_id": 13,
            "filename": str(self.image),
            "file_name": self.image.name,
            "sent": "the complete referring expression",
            "try_tn": "the traceable counterfactual expression",
            "target_bbox_used": [20, 16, 20, 16],
            "instances": [
                {
                    "bbox": [20, 16, 20, 16],
                    "positive_phrase": "the complete referring expression",
                    "negative_phrase": "the traceable counterfactual expression",
                    "text_is_negative": True,
                }
            ],
            "proposal_covered_verified": True,
            "visual_verified_negative": True,
            "traceable_counterfactual_edit": True,
            "cached_proposal_coverage_only": True,
            "verification_contract": "target_plus_all_cached_proposals_no",
            "tn_scope": "proposal_covered_verified",
            "table_b_id": "D3",
            "split": "train",
            "tn_eval_split": "screen_calibration",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _requests(self):
        image_cache = {}
        rank = parse_refcoco_rank_requests(
            [self.rank_row], image_cache=image_cache
        )
        pairs = parse_d3_pair_requests([self.d3_row], image_cache=image_cache)
        return rank, pairs

    def test_formal_config_binding_is_b8a4_partial_fusion4(self):
        self.assertEqual(
            extractor.PINNED_FORMAL_CONFIG_PATH,
            Path(
                "/media/haoyi/T9/external/mmgdino_l_baseline/"
                "mmgdino_t_refcoco5e_formal_b8a4_partial_fusion4_"
                "seed20260819.py"
            ),
        )
        self.assertEqual(
            extractor.PINNED_FORMAL_CONFIG_SHA256,
            "c36ef5e02842cc276e4017396fbaf5c1c37622b19a76cb0691b2dd80fef3fb3a",
        )

    def test_hook_parser_selects_final_900_queries_and_native_scores(self):
        decoder, head = _raw_hook_outputs()
        decoder[0].requires_grad_(True)
        hook = parse_mmgdino_hook_outputs(
            decoder, head, feature_dtype=torch.float16
        )
        self.assertEqual(tuple(hook.query_features.shape), (900, 256))
        self.assertEqual(hook.query_features.dtype, torch.float16)
        self.assertTrue(torch.equal(hook.query_features, torch.full_like(hook.query_features, 5)))
        self.assertAlmostEqual(hook.native_score[0].item(), torch.sigmoid(torch.tensor(4.0)).item())
        self.assertAlmostEqual(hook.native_score[1].item(), torch.sigmoid(torch.tensor(-2.0)).item())
        self.assertTrue(hook.candidate_mask.all())
        for value in (
            hook.query_features,
            hook.native_score,
            hook.boxes,
            hook.candidate_mask,
        ):
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(value.is_contiguous())
            self.assertFalse(value.requires_grad)

    def test_hook_parser_fails_closed_on_shape_nonfinite_mask_and_boxes(self):
        decoder, head = _raw_hook_outputs()
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "shape"):
            parse_mmgdino_hook_outputs(
                (decoder[0][:-1], decoder[1]), head
            )

        decoder, head = _raw_hook_outputs()
        decoder[0][-1, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "nonfinite"):
            parse_mmgdino_hook_outputs(decoder, head)

        decoder, head = _raw_hook_outputs()
        head[0][-1, 0, 0].fill_(float("-inf"))
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "all 900"):
            parse_mmgdino_hook_outputs(decoder, head)

        decoder, head = _raw_hook_outputs()
        head[1][-1, 0, 0, 2] = 0.0
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "positive"):
            parse_mmgdino_hook_outputs(decoder, head)

    def test_hook_recorder_requires_exactly_one_call_per_module(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Identity()
                self.bbox_head = torch.nn.Identity()

        model = Model()
        decoder, head = _raw_hook_outputs()
        with MMGroundingDinoHookRecorder(model) as recorder:
            with self.assertRaisesRegex(MMGroundingDinoExtractionError, "exactly one"):
                recorder.consume(feature_dtype=torch.float16)
            model.decoder(decoder)
            model.bbox_head(head)
            hook = recorder.consume(feature_dtype=torch.float32)
            self.assertEqual(hook.query_features.dtype, torch.float32)
            model.decoder(decoder)
            model.decoder(decoder)
            model.bbox_head(head)
            with self.assertRaisesRegex(MMGroundingDinoExtractionError, "decoder=2"):
                recorder.consume(feature_dtype=torch.float32)

    def test_rank_and_d3_parsers_bind_paths_ids_boxes_and_closed_pairs(self):
        rank, pairs = self._requests()
        self.assertEqual(len(rank), 1)
        self.assertEqual(rank[0].sample_id, "refcoco:refcoco_unc_val:123:7:11:13")
        self.assertEqual(rank[0].image_sha256, self.image_sha)
        self.assertTrue(
            torch.equal(
                rank[0].gt_boxes,
                torch.tensor([[0.3, 0.3, 0.2, 0.2]]),
            )
        )
        self.assertEqual([request.pair_role for request in pairs], ["positive", "negative"])
        self.assertEqual(pairs[0].pair_id, pairs[1].pair_id)
        self.assertEqual(tuple(pairs[1].gt_boxes.shape), (0, 4))

    def test_input_parsers_fail_closed_on_path_gt_flags_and_duplicate_ids(self):
        wrong_id = copy.deepcopy(self.rank_row)
        wrong_id["image_id"] = 124
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "filename/image_id"):
            parse_refcoco_rank_requests([wrong_id])

        outside = copy.deepcopy(self.rank_row)
        outside["instances"][0]["bbox"] = [90, 70, 20, 20]
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "outside"):
            parse_refcoco_rank_requests([outside])

        unverified = copy.deepcopy(self.d3_row)
        unverified["visual_verified_negative"] = False
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "must be true"):
            parse_d3_pair_requests([unverified])

        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "unique"):
            parse_refcoco_rank_requests([self.rank_row, self.rank_row])

    def test_explicit_image_roots_resolve_stale_rank_and_raw_d3_train_rows(self):
        stale_rank = copy.deepcopy(self.rank_row)
        stale_rank["filename"] = f"/stale/training/root/{self.image.name}"
        rank = parse_refcoco_rank_requests(
            [stale_rank], image_root=self.root
        )
        self.assertEqual(rank[0].image_path, self.image.resolve())

        raw_d3 = copy.deepcopy(self.d3_row)
        raw_d3.pop("filename")
        raw_d3.pop("instances")
        raw_d3.pop("tn_eval_split")
        pairs = parse_d3_pair_requests([raw_d3], image_root=self.root)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].image_path, self.image.resolve())

        traversal = copy.deepcopy(raw_d3)
        traversal["file_name"] = f"../{self.image.name}"
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "without traversal"):
            parse_d3_pair_requests([traversal], image_root=self.root)

    def test_bound_jsonl_rejects_hash_drift_and_duplicate_json_keys(self):
        source = self.root / "rows.jsonl"
        source.write_text(json.dumps(self.rank_row) + "\n", encoding="utf-8")
        rows = extractor._read_bound_jsonl(
            source,
            expected_sha256=file_sha256(source),
            name="rank_jsonl",
        )
        self.assertEqual(len(rows), 1)
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "mismatch"):
            extractor._read_bound_jsonl(
                source, expected_sha256="0" * 64, name="rank_jsonl"
            )
        duplicate = self.root / "duplicate.jsonl"
        duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "duplicate JSON key"):
            extractor._read_bound_jsonl(
                duplicate,
                expected_sha256=file_sha256(duplicate),
                name="rank_jsonl",
            )

    def test_mocked_extraction_emits_existing_shard_contract_and_code_binding(self):
        rank, pairs = self._requests()
        runtime = _FakeRuntime()
        shard, counters = extract_cached_candidate_shard(
            rank_requests=rank,
            pair_requests=pairs,
            runtime=runtime,
            shard_id="cpu-mock-shard",
            checkpoint_sha256="1" * 64,
            extractor_sha256="2" * 64,
        )
        self.assertEqual(shard["schema"], CACHE_SHARD_SCHEMA)
        self.assertEqual(shard["source"]["extractor_code_sha256"], "2" * 64)
        self.assertEqual(shard["source"]["checkpoint_sha256"], "1" * 64)
        self.assertEqual(
            counters,
            {
                "rank_input": 1,
                "rank_kept": 1,
                "rank_no_positive": 0,
                "confidence_rows": 2,
            },
        )
        self.assertEqual(len(runtime.calls), 3)
        self.assertEqual(
            [caption for _, caption in runtime.calls],
            [
                "the complete referring expression",
                "the complete referring expression",
                "the traceable counterfactual expression",
            ],
        )
        validate_cached_candidate_shard(shard)
        self.assertEqual(
            cached_candidate_content_sha256(shard),
            cached_candidate_content_sha256(shard),
        )

    def test_extraction_rejects_rank_without_oracle_and_malformed_runtime(self):
        rank, pairs = self._requests()
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "no IoU>=0.5"):
            extract_cached_candidate_shard(
                rank_requests=rank,
                pair_requests=pairs,
                runtime=_FakeRuntime(_valid_hook_batch(oracle=False)),
                shard_id="no-oracle",
                checkpoint_sha256="1" * 64,
                extractor_sha256="2" * 64,
            )

        shard, counters = extract_cached_candidate_shard(
            rank_requests=rank,
            pair_requests=pairs,
            runtime=_FakeRuntime(_valid_hook_batch(oracle=False)),
            shard_id="no-oracle-preserved",
            checkpoint_sha256="1" * 64,
            extractor_sha256="2" * 64,
            allow_rank_rows_without_positive=True,
        )
        self.assertEqual(counters["rank_input"], 1)
        self.assertEqual(counters["rank_kept"], 0)
        self.assertEqual(counters["rank_no_positive"], 1)
        validate_cached_candidate_shard(
            shard, allow_rank_rows_without_positive=True
        )
        with self.assertRaisesRegex(
            CachedCandidateContractError, "positives and hard negatives"
        ):
            validate_cached_candidate_shard(shard)

        malformed = replace(
            _valid_hook_batch(),
            query_features=torch.zeros(899, CACHE_FEATURE_DIM),
        )
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "shape"):
            extract_cached_candidate_shard(
                rank_requests=rank,
                pair_requests=pairs,
                runtime=_FakeRuntime(malformed),
                shard_id="bad-hook",
                checkpoint_sha256="1" * 64,
                extractor_sha256="2" * 64,
            )

    def test_extraction_rehashes_images_before_every_forward(self):
        rank, pairs = self._requests()
        Image.new("RGB", (100, 80), color=(90, 80, 70)).save(self.image)
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "changed before forward"):
            extract_cached_candidate_shard(
                rank_requests=rank,
                pair_requests=pairs,
                runtime=_FakeRuntime(),
                shard_id="changed-image",
                checkpoint_sha256="1" * 64,
                extractor_sha256="2" * 64,
            )

    def test_native_prediction_parity_uses_raw_hook_top_query_and_box(self):
        hook = _valid_hook_batch()
        top = int(hook.native_score.argmax())
        box = hook.boxes[top]
        center = box[:2]
        size = box[2:]
        predicted_box = torch.cat((center - size / 2, center + size / 2))
        predicted_box *= torch.tensor([100.0, 80.0, 100.0, 80.0])
        validate_native_prediction_parity(
            hook,
            predicted_scores=hook.native_score[top].reshape(1),
            predicted_boxes=predicted_box.reshape(1, 4),
            image_width=100,
            image_height=80,
        )
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "top score"):
            validate_native_prediction_parity(
                hook,
                predicted_scores=torch.zeros(1),
                predicted_boxes=predicted_box.reshape(1, 4),
                image_width=100,
                image_height=80,
            )

    def test_pinned_asset_validator_rejects_commit_config_and_checkpoint_drift(self):
        mmdet = self.root / "mmdetection"
        mmdet.mkdir()
        config = self.root / "formal.py"
        checkpoint = self.root / "epoch_5.pth"
        config.write_text("config", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        expected_checkpoint = file_sha256(checkpoint)

        def bound_sha(path):
            path = Path(path)
            if path == config:
                return extractor.PINNED_FORMAL_CONFIG_SHA256
            return file_sha256(path)

        with mock.patch.object(
            extractor, "PINNED_FORMAL_CONFIG_PATH", config
        ), mock.patch.object(
            extractor, "_git_head", return_value=extractor.PINNED_MMDET_COMMIT
        ), mock.patch.object(extractor, "file_sha256", side_effect=bound_sha):
            binding = validate_pinned_runtime_assets(
                mmdet_root=mmdet,
                config_path=config,
                checkpoint_path=checkpoint,
                expected_checkpoint_sha256=expected_checkpoint,
            )
            self.assertEqual(binding["checkpoint_sha256"], expected_checkpoint)

        with mock.patch.object(extractor, "_git_head", return_value="0" * 40):
            with self.assertRaisesRegex(MMGroundingDinoExtractionError, "commit drift"):
                validate_pinned_runtime_assets(
                    mmdet_root=mmdet,
                    config_path=config,
                    checkpoint_path=checkpoint,
                    expected_checkpoint_sha256=expected_checkpoint,
                )

        with mock.patch.object(
            extractor,
            "_git_head",
            return_value=extractor.PINNED_MMDET_COMMIT,
        ):
            with self.assertRaisesRegex(
                MMGroundingDinoExtractionError, "config path drift"
            ):
                validate_pinned_runtime_assets(
                    mmdet_root=mmdet,
                    config_path=config,
                    checkpoint_path=checkpoint,
                    expected_checkpoint_sha256=expected_checkpoint,
                )

        with mock.patch.object(
            extractor, "PINNED_FORMAL_CONFIG_PATH", config
        ), mock.patch.object(
            extractor,
            "_git_head",
            return_value=extractor.PINNED_MMDET_COMMIT,
        ), mock.patch.object(extractor, "file_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(
                MMGroundingDinoExtractionError, "config SHA-256 drift"
            ):
                validate_pinned_runtime_assets(
                    mmdet_root=mmdet,
                    config_path=config,
                    checkpoint_path=checkpoint,
                    expected_checkpoint_sha256=expected_checkpoint,
                )

    def test_cli_writes_atomic_shard_and_receipt_with_resolved_asset_paths(self):
        mmdet = self.root / "mmdetection"
        mmdet.mkdir()
        config = self.root / "formal.py"
        checkpoint = self.root / "epoch_5.pth"
        config.write_text("config", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        rank_jsonl = self.root / "rank.jsonl"
        d3_jsonl = self.root / "d3.jsonl"
        rank_jsonl.write_text(json.dumps(self.rank_row) + "\n", encoding="utf-8")
        d3_jsonl.write_text(json.dumps(self.d3_row) + "\n", encoding="utf-8")
        output = self.root / "cache" / "shard.pt"
        receipt_path = self.root / "receipts" / "extract.json"
        checkpoint_sha = file_sha256(checkpoint)

        class Runtime(_FakeRuntime):
            def __init__(self, **_kwargs):
                super().__init__()
                self.closed = False

            def close(self):
                self.closed = True

        binding = {
            "mmdetection_commit": extractor.PINNED_MMDET_COMMIT,
            "config_sha256": extractor.PINNED_FORMAL_CONFIG_SHA256,
            "checkpoint_sha256": checkpoint_sha,
        }
        argv = [
            "--mmdet-root", str(mmdet),
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--checkpoint-sha256", checkpoint_sha,
            "--rank-jsonl", str(rank_jsonl),
            "--rank-jsonl-sha256", file_sha256(rank_jsonl),
            "--rank-image-root", str(self.root),
            "--d3-jsonl", str(d3_jsonl),
            "--d3-jsonl-sha256", file_sha256(d3_jsonl),
            "--d3-image-root", str(self.root),
            "--output", str(output),
            "--receipt", str(receipt_path),
            "--shard-id", "cli-mock-shard",
            "--device", "cpu",
        ]
        with mock.patch.object(
            extractor, "validate_pinned_runtime_assets", return_value=binding
        ), mock.patch.object(extractor, "MMDetectionFrozenRuntime", Runtime):
            extractor.main(argv)

        shard = load_cached_candidate_shard(output)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(shard["shard_id"], "cli-mock-shard")
        self.assertEqual(
            shard["source"]["extractor_code_sha256"],
            file_sha256(Path(extractor.__file__)),
        )
        self.assertEqual(receipt["output"]["row_count"], 3)
        self.assertEqual(receipt["assets"]["mmdet_root"], str(mmdet.resolve()))
        self.assertEqual(receipt["assets"]["config_path"], str(config.resolve()))
        self.assertEqual(
            receipt["assets"]["checkpoint_path"], str(checkpoint.resolve())
        )
        self.assertEqual(receipt["runtime"]["tokens_positive"], -1)
        with self.assertRaisesRegex(MMGroundingDinoExtractionError, "must not already exist"):
            extractor.main(argv)


if __name__ == "__main__":
    unittest.main()
