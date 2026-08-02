import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import stageb_screen_calibration as calibration
from tools.stageb_screen_calibration import (
    EVAL_SPLIT,
    ScreenCalibrationError,
    build_manifest,
    derive_row,
    load_binding,
    meta_rows,
    sha256_file,
    summary_fields,
)


AUDIT_SHA256 = "a" * 64


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(index: int) -> dict:
    return {
        "sample_id": f"screen:{index}",
        "image_id": 10 + index,
        "ann_id": 20 + index,
        "ref_id": 30 + index,
        "sent_id": 40 + index,
        "class_id": 7,
        "file_name": f"COCO_train2014_{index:012d}.jpg",
        "split": "train",
        "dataset": "refcocoplus",
        "pair_source": "refcoco+_unc",
        "category_name": "person",
        "class_norm_name": "person",
        "target_bbox_used": [1.0, 2.0, 30.0, 40.0],
        "sent": "person in a red shirt",
        "try_tn": "person in a blue shirt",
        "try_tn_head": "person",
        "try_tn_head_phrase": "person in a red shirt",
        "replace_category": ["color"],
        "replace_from": ["red"],
        "replace_to": ["blue"],
        "replace_span": [[3, 4]],
        "tn_edits": [
            {
                "category": "color",
                "replace_from": "red",
                "replace_to": "blue",
                "replace_span": [3, 4],
            }
        ],
        "table_b_pair_schema": "stage-b-paper-table-b-scope-preserving-pair-v1",
        "table_b_id": "D3",
        "tn_scope": "proposal_covered_verified",
        "proposal_covered_verified": True,
        "traceable_counterfactual_edit": True,
        "visual_verified_negative": True,
        "global_tn_verified": False,
        "proposalset_proxy_verified": False,
        "cached_proposal_coverage_only": True,
        "all_900_gdino_queries_verified": False,
        "global_max_label_is_semantic_extrapolation": True,
    }


class StageBScreenCalibrationTest(unittest.TestCase):
    def test_relocated_execution_root_resolves_artifact_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifact_repository"
            (artifact_root / "outputs").mkdir(parents=True)
            (artifact_root / "data").mkdir()
            execution_root = root / "execution_snapshot"
            execution_root.mkdir()
            (execution_root / "outputs").symlink_to(
                artifact_root / "outputs", target_is_directory=True
            )

            with mock.patch.object(calibration, "REPO_ROOT", execution_root):
                self.assertEqual(
                    calibration._artifact_repository_root(),
                    artifact_root.resolve(strict=True),
                )

    def test_derives_negative_instance_without_scope_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived = derive_row(
                _row(0),
                data_root=root,
                index=0,
                audit_sha256=AUDIT_SHA256,
            )
            self.assertEqual(derived["tn_eval_split"], EVAL_SPLIT)
            self.assertFalse(derived["global_tn_verified"])
            self.assertEqual(derived["tn_scope"], "proposal_covered_verified")
            instance = derived["instances"][0]
            self.assertTrue(instance["text_is_negative"])
            self.assertEqual(instance["phrase"], "person in a blue shirt")
            self.assertEqual(instance["positive_phrase"], "person in a red shirt")
            self.assertFalse(instance["global_tn_verified"])
            self.assertEqual(derived["table_b_audit_sha256"], AUDIT_SHA256)
            self.assertEqual(instance["table_b_audit_sha256"], AUDIT_SHA256)
            self.assertTrue(Path(derived["filename"]).is_absolute())

    def test_binding_replays_every_row_and_detects_derived_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            audit = root / "audit.json"
            derived = root / "out" / "calibration.jsonl"
            _write_jsonl(source, [_row(0), _row(1)])
            audit.write_text('{"schema": "fixture"}\n', encoding="utf-8")
            binding = build_manifest(
                source_path=source,
                audit_path=audit,
                derived_path=derived,
                data_root=root,
            )
            self.assertEqual(binding.derived_manifest["rows"], 2)
            self.assertEqual(len(binding.row_mapping), 2)
            self.assertEqual(len(meta_rows(binding)), 2)
            fields = summary_fields(binding)
            self.assertEqual(fields["screen_calibration_source_n"], 2)
            self.assertEqual(
                load_binding(binding.path, expected_derived=derived).eval_split,
                EVAL_SPLIT,
            )

            rows = [json.loads(line) for line in derived.read_text().splitlines()]
            rows[0]["instances"][0]["positive_phrase"] = "drift"
            _write_jsonl(derived, rows)
            with self.assertRaisesRegex(
                ScreenCalibrationError, "bound file changed|derived row drift"
            ):
                load_binding(binding.path, expected_derived=derived)

    def test_derived_manifest_is_consumable_by_patch_episode_dataset(self):
        from PIL import Image

        from datasets.patch_episode import PatchEpisodeJsonlDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            audit = root / "audit.json"
            derived = root / "out" / "calibration.jsonl"
            image_path = (
                root
                / "COCO/coco2014/train2014/COCO_train2014_000000000000.jpg"
            )
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (64, 64), color=(128, 128, 128)).save(image_path)
            _write_jsonl(source, [_row(0)])
            audit.write_text('{"schema": "fixture"}\n', encoding="utf-8")
            audit_sha = sha256_file(audit)
            build_manifest(
                source_path=source,
                audit_path=audit,
                derived_path=derived,
                data_root=root,
            )
            dataset = PatchEpisodeJsonlDataset(
                root="/",
                anno=str(derived),
                source="screen_calibration",
                box_format="xywh",
                neg_episode_prob=0.0,
                support_min_count=1,
                support_num_patches_min=1,
                support_num_patches_max=1,
                build_text_token_masks=False,
                tn_balance_sampling=False,
            )
            _image, target = dataset[0]

        self.assertEqual(target["table_b_id"], "D3")
        self.assertEqual(target["table_b_audit_sha256"], audit_sha)
        self.assertEqual(target["tn_scope"], "proposal_covered_verified")

    def test_rejects_global_scope_upgrade_or_multiple_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upgraded = _row(0)
            upgraded["global_tn_verified"] = True
            with self.assertRaisesRegex(ScreenCalibrationError, "scope"):
                derive_row(
                    upgraded,
                    data_root=root,
                    index=0,
                    audit_sha256=AUDIT_SHA256,
                )

            multiple = _row(0)
            multiple["tn_edits"].append(dict(multiple["tn_edits"][0]))
            with self.assertRaisesRegex(ScreenCalibrationError, "single edit"):
                derive_row(
                    multiple,
                    data_root=root,
                    index=0,
                    audit_sha256=AUDIT_SHA256,
                )


if __name__ == "__main__":
    unittest.main()
