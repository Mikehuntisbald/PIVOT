import unittest

import torch

from tools.eval_stageb_data_driven_new_head_dev import IDENTITY_KEYS, ManifestRow
from tools.eval_stageb_data_driven_rank_patch_geometry import (
    QUERY_COUNT,
    ROUTE_NAMES,
    _route_protocol,
    cluster_bootstrap,
    evaluate_batch,
    summarize_records,
)


def _manifest_row(source="refcoco", image_id=123):
    identity = {
        "source": f"{source}_unc_train",
        "image_id": image_id,
        "ann_id": 456,
        "ref_id": 789,
        "sent_id": 7,
        "split": "train",
        "filename": f"COCO_train2014_{image_id:012d}.jpg",
    }
    return ManifestRow(
        identity=tuple(identity[key] for key in IDENTITY_KEYS),
        identity_object=identity,
        coco_split="train2014",
        image_id=image_id,
    )


def _target(row):
    identity = row.identity_object
    return {
        "boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2]], dtype=torch.float32),
        "primary_instance_mask": torch.tensor([True]),
        "image_id": torch.tensor([identity["image_id"]], dtype=torch.int64),
        "ann_id": torch.tensor([identity["ann_id"]], dtype=torch.int64),
        "ref_id": torch.tensor([identity["ref_id"]], dtype=torch.int64),
        "sent_id": torch.tensor([identity["sent_id"]], dtype=torch.int64),
        "dataset_name": identity["source"],
    }


def _outputs():
    candidate = torch.ones((1, QUERY_COUNT), dtype=torch.bool)
    patch = torch.zeros((1, QUERY_COUNT), dtype=torch.float32)
    patch[0, 0] = -100.0
    canonical_data = torch.zeros((1, QUERY_COUNT), dtype=torch.float32)
    canonical_data[0, 0] = 10.0
    canonical_data[0, 1] = 9.0
    raw = torch.zeros_like(canonical_data)
    raw[0, 1] = 5.0
    fused = torch.zeros_like(canonical_data)
    fused[0, 2] = 5.0
    boxes = torch.tensor([0.8, 0.8, 0.1, 0.1]).repeat(1, QUERY_COUNT, 1)
    boxes[0, 1] = torch.tensor([0.2, 0.2, 0.2, 0.2])
    canonical = {
        "stage_b_data_driven_text_rank_score": canonical_data,
        "stage_b_data_driven_rank_score": canonical_data.clone(),
        "stage_b_data_driven_raw_expression_native_score": raw,
        "stage_b_data_driven_encoder_fused_expression_fixed_query_native_score": fused,
        "stage_b_data_driven_candidate_mask": candidate,
        "pred_logits_patch": patch,
        "pred_boxes": boxes,
    }

    fulltext_data = torch.zeros_like(canonical_data)
    fulltext_data[0, 3] = 5.0
    text_logits = torch.full((1, QUERY_COUNT, 4), -10.0)
    text_logits[0, 4, 0] = 10.0
    main_mask = torch.tensor([[True, False, False, False]])
    fulltext_boxes = torch.tensor([0.8, 0.8, 0.1, 0.1]).repeat(
        1, QUERY_COUNT, 1
    )
    fulltext_boxes[0, 4] = torch.tensor([0.2, 0.2, 0.2, 0.2])
    fulltext = {
        "stage_b_data_driven_text_rank_score": fulltext_data,
        "stage_b_data_driven_rank_score": fulltext_data.clone(),
        "stage_b_data_driven_candidate_mask": candidate.clone(),
        "pred_logits_patch": torch.zeros_like(patch),
        "pred_logits_text": text_logits,
        "pred_logits": text_logits.clone(),
        "stage_b_diagnostic_main_phrase_token_mask": main_mask,
        "pred_boxes": fulltext_boxes,
    }
    return canonical, fulltext


class GeometryRouteEvaluationTest(unittest.TestCase):
    def test_routes_share_each_universe_and_apply_exact_gap_mask(self):
        row = _manifest_row()
        canonical, fulltext = _outputs()
        record = evaluate_batch(
            canonical,
            fulltext,
            [_target(row)],
            [row],
            source="refcoco",
            partition="dev_screen",
        )[0]

        self.assertEqual(set(record["routes"]), set(ROUTE_NAMES))
        self.assertFalse(record["routes"]["canonical_data_no_gate"]["acc50"])
        self.assertTrue(record["routes"]["canonical_data_gap3"]["acc50"])
        self.assertEqual(
            record["routes"]["canonical_data_gap3"]["winner_query_index"], 1
        )
        self.assertTrue(
            record["routes"]["canonical_unfused_native_no_gate"]["acc50"]
        )
        self.assertTrue(record["routes"]["fulltext_b58_no_gate"]["acc50"])
        self.assertFalse(record["routes"]["fulltext_data_no_gate"]["acc50"])
        self.assertEqual(
            record["routes"]["fulltext_b58_no_gate"]["winner_query_index"], 4
        )
        self.assertEqual(
            record["query_universes"]["canonical"]["positive_query_count"], 1
        )
        self.assertLess(
            record["query_universes"]["canonical"]["gap3_eligible_query_count"],
            QUERY_COUNT,
        )

    def test_summary_reports_transitions_and_confounded_contrast(self):
        records = {}
        for source in ("refcoco", "refcocoplus", "refcocog"):
            row = _manifest_row(source)
            canonical, fulltext = _outputs()
            records[source] = evaluate_batch(
                canonical,
                fulltext,
                [_target(row)],
                [row],
                source=source,
                partition="dev_screen",
            )
        summary = summarize_records(records)
        gate = summary["contrasts"]["canonical_data_gate_effect"]
        self.assertEqual(gate["macro_ref3_acc50_delta"], 1.0)
        self.assertEqual(gate["sources"]["refcoco"]["transitions"]["fixed"], 1)
        system = summary["contrasts"][
            "system_fulltext_b58_minus_canonical_data_no_gate"
        ]
        self.assertEqual(system["claim"], "different_query_universes_confounded")
        self.assertEqual(system["macro_ref3_acc50_delta"], 1.0)

    def test_cluster_bootstrap_is_deterministic_and_paired(self):
        records = {}
        for source in ("refcoco", "refcocoplus", "refcocog"):
            row = _manifest_row(source)
            canonical, fulltext = _outputs()
            records[source] = evaluate_batch(
                canonical,
                fulltext,
                [_target(row)],
                [row],
                source=source,
                partition="dev_screen",
            )
        first = cluster_bootstrap(records, iterations=20, seed=9, chunk_size=7)
        second = cluster_bootstrap(records, iterations=20, seed=9, chunk_size=7)
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 1)
        result = first["contrasts"]["canonical_data_gate_effect"]
        self.assertEqual(result["ci95"], [1.0, 1.0])
        self.assertEqual(result["probability_delta_positive"], 1.0)

    def test_protocol_declares_query_universe_boundary(self):
        protocol = _route_protocol()
        self.assertFalse(protocol["formal_headline_eligible"])
        self.assertIn(
            "not an equivalent full-expression b58 forward",
            protocol["query_universes"]["canonical"]["native_fixed_query_claim"],
        )
        self.assertEqual(
            len({route["name"] for route in protocol["routes"]}),
            len(ROUTE_NAMES),
        )


if __name__ == "__main__":
    unittest.main()
