import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import materialize_stageb_fulltext_gated_records as materialize


class MaterializeFullTextGatedRecordsTest(unittest.TestCase):
    def _fixture(self, root: Path):
        gate_path = root / "gate.json"
        gate_path.write_text("{}\n", encoding="utf-8")
        gate_payload = {
            "artifact_identity": {"sha256": "f" * 64},
            "caption_route_artifact_identity": {"sha256": "a" * 64},
            "formal_routed_artifact_identity": {"sha256": "b" * 64},
            "route": {
                "conditional_overrides": {
                    "person": {"predicate": {"max_tokens": 1}}
                }
            },
        }
        base_paths = {}
        routed_paths = {}
        manifest_paths = {}
        for split in materialize.REF_SPLITS:
            manifest_path = root / f"{split}.manifest.jsonl"
            base_path = root / f"{split}.base.jsonl"
            routed_path = root / f"{split}.routed.jsonl"
            manifests = [
                {"instances": [{"raw_phrase": "short"}]},
                {"instances": [{"raw_phrase": "two words"}]},
                {"instances": [{"raw_phrase": "bowl"}]},
            ]
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifests),
                encoding="utf-8",
            )
            manifest_sha = materialize._sha256(manifest_path)
            base_rows = []
            routed_rows = []
            for index, caption in enumerate(("person", "person", "bowl")):
                common = {
                    "schema": "stageb-eval-record-v1",
                    "task": "ref",
                    "split": split,
                    "manifest_index": index,
                    "manifest_sha256": manifest_sha,
                    "manifest_n": 3,
                    "valid": True,
                    "sample_id": f"{split}:{index}",
                    "image_id": index,
                    "ann_id": index,
                    "ref_id": index,
                    "sent_id": index,
                    "all_query_best_iou": 0.9,
                }
                base_correct = index == 1
                routed_correct = index != 1
                base_rows.append(
                    {**common, "top1_iou": 0.8 if base_correct else 0.1, "correct50": base_correct}
                )
                routed_rows.append(
                    {
                        **common,
                        "top1_iou": 0.8 if routed_correct else 0.1,
                        "correct50": routed_correct,
                        "canonical_class_norm": caption,
                        "caption_route_descriptor_id": (
                            "max_patch_p025_w003125_v1"
                            if caption == "person"
                            else "max_external_p05_w0046875_v1"
                        ),
                        "caption_route_selection_artifact_identity_sha256": "a" * 64,
                        "external_transfer_artifact_sha256": "b" * 64,
                        "selected_box": [0.1, 0.1, 0.2, 0.2],
                        "selected_box_format": "normalized_xyxy",
                        "winner_candidate_index": 0,
                        "winner_patch_query_index": 0,
                        "matched_external_query_index": 0,
                    }
                )
            base_path.write_text(
                "".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8"
            )
            routed_path.write_text(
                "".join(json.dumps(row) + "\n" for row in routed_rows), encoding="utf-8"
            )
            base_paths[split] = base_path
            routed_paths[split] = routed_path
            manifest_paths[split] = manifest_path
        return gate_path, gate_payload, base_paths, routed_paths, manifest_paths

    def test_short_person_routes_and_long_person_falls_back(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            materialize,
            "EXPECTED_MANIFEST_N",
            {split: 3 for split in materialize.REF_SPLITS},
        ):
            root = Path(temporary)
            gate_path, gate_payload, base, routed, manifests = self._fixture(root)
            with mock.patch.object(
                materialize,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=gate_payload,
            ):
                summary = materialize.materialize_fulltext_gated_records(
                    gate_artifact_path=gate_path,
                    base_record_paths=base,
                    routed_record_paths=routed,
                    manifest_paths=manifests,
                    output_dir=root / "output",
                )
            self.assertTrue(all(row["correct50"] == 3 for row in summary["splits"].values()))
            path = Path(summary["splits"][materialize.REF_SPLITS[0]]["records"]["path"])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows[0]["fulltext_route_gate_predicate_matched"])
            self.assertEqual(rows[0]["fulltext_route_gate_source"], "formal_routed_v2")
            self.assertTrue(rows[1]["fulltext_route_gate_fallback_matched"])
            self.assertEqual(rows[1]["fulltext_route_gate_source"], "external_base_direct")
            self.assertNotIn("selected_box", rows[1])

    def test_pair_order_drift_is_rejected_and_partial_output_removed(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            materialize,
            "EXPECTED_MANIFEST_N",
            {split: 3 for split in materialize.REF_SPLITS},
        ):
            root = Path(temporary)
            gate_path, gate_payload, base, routed, manifests = self._fixture(root)
            first = materialize.REF_SPLITS[0]
            rows = [json.loads(line) for line in routed[first].read_text(encoding="utf-8").splitlines()]
            rows[0]["sample_id"] = "drift"
            routed[first].write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "output"
            with mock.patch.object(
                materialize,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=gate_payload,
            ), self.assertRaisesRegex(materialize.FullTextGatedRecordError, "sample_id"):
                materialize.materialize_fulltext_gated_records(
                    gate_artifact_path=gate_path,
                    base_record_paths=base,
                    routed_record_paths=routed,
                    manifest_paths=manifests,
                    output_dir=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
