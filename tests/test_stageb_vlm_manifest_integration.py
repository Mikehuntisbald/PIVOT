import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.build_stageb_vlm_strict_tn_manifest import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_MANIFEST_NAME,
)


RUN_DATA_INTEGRATION = os.environ.get("PIVOT_RUN_DATA_INTEGRATION") == "1"


@unittest.skipUnless(
    RUN_DATA_INTEGRATION,
    "set PIVOT_RUN_DATA_INTEGRATION=1 to exercise the local RefCOCO/image assets",
)
class StageBVlmManifestIntegrationTest(unittest.TestCase):
    def test_adapter_positive_counterfactual_masks_cover_all_strict_rows(self):
        from datasets import build_dataset
        from tools.eval_stageb_tn_val import _make_datasetinfo
        from util.slconfig import SLConfig

        data_root = DEFAULT_DATA_ROOT.resolve()
        manifest = (DEFAULT_OUTPUT_DIR / "eval_manifest.jsonl").resolve()
        cfg = SLConfig.fromfile(
            "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py"
        )
        cfg.output_dir = ""
        dataset = build_dataset(
            image_set="val",
            args=cfg,
            datasetinfo=_make_datasetinfo(
                data_root,
                manifest,
                adapter_eval_scope="image_global_topk_verified",
                adapter_eval_protocol="stageb_vlm_verified_strict_tn_v2",
            ),
        )
        dataset.transforms = None
        manifest_rows = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
        ]
        focus = {395, 825, 991, 1031, 1292}
        bad = []
        checked_focus = set()
        for index in range(len(dataset)):
            _, target = dataset[index]
            row = manifest_rows[index]
            observed_identity = tuple(
                int(target[key].view(-1)[0].item())
                for key in ("image_id", "ann_id", "ref_id", "sent_id")
            )
            expected_identity = tuple(
                int(row[key])
                for key in ("image_id", "ann_id", "ref_id", "sent_id")
            )
            self.assertEqual(observed_identity, expected_identity)
            self.assertEqual(target.get("sample_id"), row.get("sample_id"))
            valid = bool(target["has_rank_positive"].view(-1)[0].item()) and bool(
                target["rank_positive_phrase_to_token_mask"][0].any().item()
            )
            if not valid:
                bad.append(index)
            if index in focus:
                checked_focus.add(index)
                self.assertTrue(valid)
                self.assertTrue(target["rank_positive_captions"][0])
            self.assertNotIn("patch", target)
            self.assertNotIn("patch_global", target)

        self.assertEqual(len(dataset), 2_031)
        self.assertEqual(checked_focus, focus)
        self.assertEqual(bad, [])

    def test_evaluator_selects_and_dataset_loads_every_manifest_row_without_resampling(self):
        from datasets import build_dataset
        from tools.eval_stageb_tn_val import _build_tn_eval_jsonl, _make_datasetinfo
        from util.slconfig import SLConfig

        data_root = DEFAULT_DATA_ROOT.resolve()
        manifest = (DEFAULT_OUTPUT_DIR / "eval_manifest.jsonl").resolve()
        semantic_manifest = (
            DEFAULT_OUTPUT_DIR / SEMANTIC_STAGEB_UNION_IMAGE_DISJOINT_MANIFEST_NAME
        ).resolve()
        self.assertTrue(manifest.is_file())
        self.assertTrue(semantic_manifest.is_file())

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            eval_jsonl, meta_rows, counts = _build_tn_eval_jsonl(
                data_root=data_root,
                output_dir=output_dir,
                tn_jsonl=manifest,
                splits=["refcocop_val", "refcocog_umd_val"],
                max_pairs=0,
            )
            self.assertEqual(
                counts,
                {"refcocop_val": 1_249, "refcocog_umd_val": 782},
            )
            self.assertEqual(len(meta_rows), 2_031)

            _, semantic_meta_rows, semantic_counts = _build_tn_eval_jsonl(
                data_root=data_root,
                output_dir=output_dir / "semantic",
                tn_jsonl=semantic_manifest,
                splits=["refcocop_val", "refcocog_umd_val"],
                max_pairs=0,
            )
            self.assertEqual(
                semantic_counts,
                {"refcocop_val": 965, "refcocog_umd_val": 642},
            )
            self.assertEqual(len(semantic_meta_rows), 1_607)

            cfg = SLConfig.fromfile(
                "config/ablations/cfg_stageb_v14_phrase_validity_cvar_globaltn.py"
            )
            cfg.output_dir = str(output_dir)
            dataset = build_dataset(
                image_set="val",
                args=cfg,
                datasetinfo=_make_datasetinfo(data_root, eval_jsonl),
            )
            self.assertEqual(len(dataset), 2_031)
            self.assertEqual(len(dataset.metas), 2_031)

            # Query transforms are orthogonal to this contract test. Image opening,
            # bbox parsing, support-patch loading/fallback, and text masks remain active.
            dataset.transforms = None
            failures = []
            with patch(
                "datasets.patch_episode.random.randrange",
                side_effect=AssertionError("hidden dataset resample"),
            ):
                for index in range(len(dataset)):
                    try:
                        _, target = dataset[index]
                        self.assertEqual(tuple(target["boxes"].shape), (1, 4))
                        self.assertIn("support_class", target)
                        self.assertIn("patch", target)
                        self.assertIn("has_rank_positive", target)
                    except Exception as exc:  # Continue so the test reports every bad index.
                        failures.append((index, type(exc).__name__, str(exc)))

            self.assertEqual(failures, [], f"invalid rows: {len(failures)}; first={failures[:5]}")


if __name__ == "__main__":
    unittest.main()
