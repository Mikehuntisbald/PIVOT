import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from tools.judge_stageb_fixed_gdino_top1_qwen import (
    ASSET_POLICY_SHA256,
    EXTRACTION_AUDIT_KIND,
    EXTRACTION_AUDIT_SCHEMA,
    EXTRACTION_SCHEMA,
    JUDGE_RUNTIME_POLICY,
    JUDGE_RUNTIME_POLICY_SHA256,
    MODEL_REVISION,
    PROMPT_TEMPLATE_SHA256,
    QwenJudgeError,
    VISION_PROCESSOR_CONFIG,
    _validate_fixed_cli_policy,
    _validate_retry_error_coverage,
    build_boxed_assets,
    build_plan,
    canonical_sha256,
    file_record,
    judgment_cache_key,
    make_parser,
    parse_structured_answer,
    run,
    sha256_file,
    validate_runtime_environment,
)


class StageBFixedGDINOTop1QwenTest(unittest.TestCase):
    def _write_extraction_audit(self, root: Path, extraction: Path) -> Path:
        record = {**file_record(extraction), "rows": 1}
        path = root / "extraction-audit.json"
        path.write_text(
            json.dumps(
                {
                    "schema": EXTRACTION_AUDIT_SCHEMA,
                    "kind": EXTRACTION_AUDIT_KIND,
                    "rows": 1,
                    "manifest": record,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _row(self, root: Path):
        image_path = root / "image.jpg"
        Image.new("RGB", (100, 80), color=(120, 130, 140)).save(image_path)
        return {
            "schema": EXTRACTION_SCHEMA,
            "identity": {
                "sample_id": "sample-1",
                "dataset": "refcocoplus",
                "split": "train",
                "image_id": 1,
                "ann_id": 2,
                "ref_id": 3,
                "sent_id": 4,
            },
            "negative_expression": "the blue car",
            "num_queries": 900,
            "valid_query_count": 900,
            "image": {
                "path": str(image_path),
                "width": 100,
                "height": 80,
                "sha256": sha256_file(image_path),
            },
            "regions": [
                {
                    "region_id": "region-inherited",
                    "origins": ["primary"],
                    "query_ids": [10],
                    "bbox_xyxy_original": [10.0, 10.0, 40.0, 50.0],
                    "max_overlap": {
                        "kind": "target",
                        "proposal_id": None,
                        "iou": 1.0,
                        "source_answer": "NO",
                        "source_confidence": 0.95,
                    },
                    "inherit_eligible": True,
                },
                {
                    "region_id": "region-qwen",
                    "origins": ["near_tie"],
                    "query_ids": [11],
                    "bbox_xyxy_original": [50.0, 10.0, 90.0, 60.0],
                    "max_overlap": {
                        "kind": "none",
                        "proposal_id": None,
                        "iou": 0.0,
                        "source_answer": "",
                        "source_confidence": -1.0,
                    },
                    "inherit_eligible": False,
                },
            ],
            "claims": {
                "frozen_gdino_global_max_regions_extracted": True,
                "train_path_and_deploy_transform_regions_extracted": True,
                "all_900_gdino_queries_verified": False,
                "image_global_semantic_absence_proven": False,
                "portable_to_other_checkpoint_or_transform": False,
            },
        }

    def test_structured_parser_accepts_json_and_rejects_invalid_answers(self):
        parsed = parse_structured_answer(
            'prefix {"answer":"no","confidence":0.93,"short_reason":"wrong color"}'
        )
        self.assertEqual(parsed["answer"], "NO")
        self.assertEqual(parsed["confidence"], 0.93)
        with self.assertRaisesRegex(QwenJudgeError, "invalid Qwen answer"):
            parse_structured_answer(
                '{"answer":"maybe","confidence":0.5,"short_reason":"unclear"}'
            )

    def test_plan_skips_only_high_iou_high_confidence_source_inheritance(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self._row(Path(temporary))
            plan, stats = build_plan([row])
        self.assertEqual(stats["regions"], 2)
        self.assertEqual(stats["source_inheritable_regions"], 1)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][1]["region_id"], "region-qwen")

    def test_cache_key_is_content_addressed_by_expression_and_bbox(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self._row(Path(temporary))
            first = judgment_cache_key(row, row["regions"][1])
            changed = json.loads(json.dumps(row))
            changed["negative_expression"] = "the green car"
            second = judgment_cache_key(changed, changed["regions"][1])
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_boxed_assets_are_hashed_and_context_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            region = row["regions"][1]
            key = judgment_cache_key(row, region)
            assets = build_boxed_assets(
                row, region, asset_root=root / "assets", cache_key=key
            )
            self.assertEqual(assets["asset_policy_sha256"], ASSET_POLICY_SHA256)
            for name in ("full_boxed", "context_2x_boxed"):
                record = assets[name]
                self.assertEqual(sha256_file(Path(record["path"])), record["sha256"])
            self.assertLessEqual(assets["context_2x_boxed"]["width"], 100)
            self.assertLessEqual(assets["context_2x_boxed"]["height"], 80)

    def test_dry_run_does_not_load_model_or_write_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            extraction = root / "extraction.jsonl"
            extraction.write_text(json.dumps(row) + "\n", encoding="utf-8")
            extraction_audit = self._write_extraction_audit(root, extraction)
            output = root / "judgments.jsonl"
            args = argparse.Namespace(
                input=str(extraction),
                extraction_audit=str(extraction_audit),
                output=str(output),
                cache_dir=str(root / "cache"),
                asset_dir=str(root / "assets"),
                dry_run=True,
                limit=0,
                resume=False,
                retry_errors=False,
                device="cuda:0",
                dtype="bfloat16",
                attn_implementation="sdpa",
                model_cache_dir=None,
                allow_download=False,
                batch_size=1,
                selection_seed="unit-test",
                audit_inherited=0,
            )
            with mock.patch(
                "tools.judge_stageb_fixed_gdino_top1_qwen.LocalQwenRunner._load",
                side_effect=AssertionError("model must not load"),
            ):
                with mock.patch(
                    "tools.judge_stageb_fixed_gdino_top1_qwen.EXPECTED_EXTRACTION_ROWS",
                    1,
                ):
                    summary = run(args)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["planned_qwen_regions"], 1)
            self.assertFalse(output.exists())

    def test_formal_runtime_validation_precedes_existing_output_cache_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            extraction = root / "extraction.jsonl"
            extraction.write_text(json.dumps(row) + "\n", encoding="utf-8")
            extraction_audit = self._write_extraction_audit(root, extraction)
            output = root / "judgments.jsonl"
            output.write_text("not-yet-trusted\n", encoding="utf-8")
            args = argparse.Namespace(
                input=str(extraction),
                extraction_audit=str(extraction_audit),
                output=str(output),
                cache_dir=str(root / "cache"),
                dry_run=False,
                limit=0,
                audit_inherited=0,
                selection_seed="unit-test",
                batch_size=1,
                resume=True,
                retry_errors=False,
                device="cuda:0",
                dtype="bfloat16",
                attn_implementation="sdpa",
                model_cache_dir=None,
                allow_download=False,
            )
            with mock.patch(
                "tools.judge_stageb_fixed_gdino_top1_qwen.EXPECTED_EXTRACTION_ROWS",
                1,
            ), mock.patch(
                "tools.judge_stageb_fixed_gdino_top1_qwen.validate_runtime_environment",
                side_effect=QwenJudgeError("runtime-stop"),
            ), mock.patch(
                "tools.judge_stageb_fixed_gdino_top1_qwen._read_existing"
            ) as read_existing, self.assertRaisesRegex(QwenJudgeError, "runtime-stop"):
                run(args)
            read_existing.assert_not_called()

    def test_model_and_prompt_are_pinned(self):
        self.assertEqual(
            MODEL_REVISION,
            "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        )
        self.assertRegex(PROMPT_TEMPLATE_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(VISION_PROCESSOR_CONFIG["max_pixels"], 1280 * 28 * 28)
        self.assertEqual(
            JUDGE_RUNTIME_POLICY_SHA256,
            canonical_sha256(JUDGE_RUNTIME_POLICY),
        )
        self.assertEqual(JUDGE_RUNTIME_POLICY["inference"]["dtype"], "bfloat16")
        self.assertEqual(
            JUDGE_RUNTIME_POLICY["inference"]["attn_implementation"], "sdpa"
        )

    def test_runtime_policy_is_part_of_cache_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self._row(Path(temporary))
            region = row["regions"][1]
            first = judgment_cache_key(row, region)
            with mock.patch(
                "tools.judge_stageb_fixed_gdino_top1_qwen.JUDGE_RUNTIME_POLICY_SHA256",
                "f" * 64,
            ):
                second = judgment_cache_key(row, region)
        self.assertNotEqual(first, second)

    def test_fixed_cli_policy_rejects_device_dtype_and_attention_drift(self):
        fixed = dict(
            device="cuda:0",
            dtype="bfloat16",
            attn_implementation="sdpa",
            batch_size=1,
            allow_download=False,
        )
        _validate_fixed_cli_policy(argparse.Namespace(**fixed))
        for key, value in (
            ("device", "cuda:1"),
            ("dtype", "float16"),
            ("attn_implementation", "eager"),
        ):
            drifted = dict(fixed)
            drifted[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                QwenJudgeError, f"runtime {key} must remain exactly"
            ):
                _validate_fixed_cli_policy(argparse.Namespace(**drifted))

    def test_runtime_environment_fails_closed_on_observed_policy_drift(self):
        observed = json.loads(json.dumps(JUDGE_RUNTIME_POLICY))
        observed["software"]["transformers"] = "drifted"
        with mock.patch(
            "tools.judge_stageb_fixed_gdino_top1_qwen._observe_runtime_policy",
            return_value=(observed, Path("/tmp/pinned-model")),
        ), self.assertRaisesRegex(QwenJudgeError, "runtime policy drifted in: software"):
            validate_runtime_environment(model_cache_dir=None)

    def test_hash_selection_is_order_independent_and_can_audit_inheritance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(4):
                row = self._row(root)
                row["identity"]["sample_id"] = f"sample-{index}"
                rows.append(row)
            first, first_stats = build_plan(
                rows,
                limit=1,
                selection_seed="fixed-seed",
                audit_inherited=2,
            )
            second, _ = build_plan(
                list(reversed(rows)),
                limit=1,
                selection_seed="fixed-seed",
                audit_inherited=2,
            )
        first_ids = [(item[0]["identity"]["sample_id"], item[1]["region_id"]) for item in first]
        second_ids = [(item[0]["identity"]["sample_id"], item[1]["region_id"]) for item in second]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_stats["planned_non_inherited_regions"], 1)
        self.assertEqual(first_stats["planned_inherited_audit_regions"], 2)

    def test_requested_pilot_pool_sizes_must_exist_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self._row(Path(temporary))
            with self.assertRaisesRegex(QwenJudgeError, "non-inherited limit=2"):
                build_plan([row], limit=2)
            with self.assertRaisesRegex(QwenJudgeError, "inherited audit=2"):
                build_plan([row], limit=1, audit_inherited=2)

    def test_retry_errors_preserves_pilot_inherited_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self._row(Path(temporary))
            inherited_key = ("sample-1", "region-inherited")
            existing = {inherited_key: {"status": "error"}}
            full_without_inherited, _ = build_plan([row], limit=0, audit_inherited=0)
            with self.assertRaisesRegex(
                QwenJudgeError, "preserve the pilot --audit-inherited"
            ):
                _validate_retry_error_coverage(
                    existing,
                    full_without_inherited,
                    retry_errors=True,
                )
            full_with_inherited, _ = build_plan([row], limit=0, audit_inherited=1)
            _validate_retry_error_coverage(
                existing,
                full_with_inherited,
                retry_errors=True,
            )

    def test_batch_cli_defaults_fit_single_24gb_gpu(self):
        args = make_parser().parse_args(
            [
                "--input",
                "in.jsonl",
                "--extraction-audit",
                "extraction-audit.json",
                "--output",
                "out.jsonl",
                "--cache-dir",
                "cache",
            ]
        )
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.attn_implementation, "sdpa")
        self.assertEqual(args.batch_size, 1)

    def test_batch_size_above_one_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            extraction = root / "extraction.jsonl"
            extraction.write_text(json.dumps(row) + "\n", encoding="utf-8")
            extraction_audit = self._write_extraction_audit(root, extraction)
            output = root / "judgments.jsonl"
            args = argparse.Namespace(
                input=str(extraction),
                extraction_audit=str(extraction_audit),
                output=str(output),
                cache_dir=str(root / "cache"),
                dry_run=True,
                limit=0,
                audit_inherited=0,
                selection_seed="unit-test",
                batch_size=2,
                resume=False,
                retry_errors=False,
                device="cuda:0",
                dtype="bfloat16",
                attn_implementation="sdpa",
                model_cache_dir=None,
                allow_download=False,
            )

            with mock.patch(
                "tools.judge_stageb_fixed_gdino_top1_qwen.EXPECTED_EXTRACTION_ROWS",
                1,
            ), self.assertRaisesRegex(
                QwenJudgeError, "runtime batch_size must remain exactly 1"
            ):
                run(args)


if __name__ == "__main__":
    unittest.main()
