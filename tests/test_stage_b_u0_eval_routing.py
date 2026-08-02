import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools import eval_stageb_tn_val as tn_eval
from tools import eval_text_groundingdino_refcoco_tn as joint_eval
from util.misc import NestedTensor


def _u0_cfg():
    return types.SimpleNamespace(
        stage_b_gdino_score_adapter=True,
        stage_b_u0_patch_rank=True,
        stage_b_v11_fixed_text=False,
        stage_b_v7=False,
    )


def _tn_target(*, include_patch=True):
    target = {
        "caption": "blue car .",
        "cap_list": ["red car", "blue car"],
        "is_tn": torch.tensor([False, True], dtype=torch.bool),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
    }
    if include_patch:
        target["patch"] = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    return target


def _batch(target=None):
    samples = NestedTensor(
        torch.ones((1, 3, 4, 4), dtype=torch.float32),
        torch.zeros((1, 4, 4), dtype=torch.bool),
    )
    return samples, [target if target is not None else _tn_target()]


class _DummyU0Model:
    def __init__(self):
        self.stage_b_gdino_score_adapter = object()
        self.stage_b_u0_patch_rank_adapter = object()
        self.calls = []

    def __call__(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        batch_size = int(samples.tensors.shape[0])
        query_count = 2
        token_count = 3
        logits = torch.zeros(
            (batch_size, query_count, token_count), dtype=torch.float32
        )
        expression_mask = torch.tensor(
            [[False, True, True]], dtype=torch.bool
        ).expand(batch_size, -1)
        base = torch.full((batch_size, query_count), 0.5, dtype=torch.float32)
        confidence = torch.arange(
            batch_size * query_count, dtype=torch.float32
        ).reshape(batch_size, query_count)
        u0_rank = 1000.0 - confidence
        return {
            "pred_logits": logits,
            "pred_boxes": torch.zeros((batch_size, query_count, 4)),
            "stage_b_gdino_expression_token_mask": expression_mask,
            "stage_b_gdino_base_score": base,
            "stage_b_gdino_rank_score": base,
            "stage_b_gdino_confidence_score": confidence,
            "stage_b_u0_rank_score": u0_rank,
        }


class StageBU0EvalRoutingTest(unittest.TestCase):
    def test_joint_ref_subset_keeps_canonical_split_seed(self):
        subset = joint_eval._requested_ref_split_seed_map(
            ["refcocog_test", "refcoco_val", "refcocog_val"], 42
        )
        self.assertEqual(subset["refcoco_val"], 42)
        self.assertEqual(subset["refcocog_val"], 600042)
        self.assertEqual(subset["refcocog_test"], 700042)

    def test_category_gate_sweep_mode_is_opt_in_and_ref_val_only(self):
        disabled = types.SimpleNamespace(category_gate_max_gaps=None)
        self.assertIsNone(
            joint_eval._validate_category_gate_sweep_mode(
                disabled, types.SimpleNamespace(), ["refcoco_testA"]
            )
        )
        with self.assertRaisesRegex(ValueError, "base expert requires"):
            joint_eval._validate_category_gate_sweep_mode(
                types.SimpleNamespace(
                    category_gate_max_gaps=None,
                    category_gate_include_base_expert=True,
                ),
                types.SimpleNamespace(),
                ["refcoco_val"],
            )

        args = types.SimpleNamespace(
            category_gate_max_gaps=[0.0, 1.0],
            skip_ref=False,
            skip_tn=True,
            no_per_example_records=False,
        )
        cfg = types.SimpleNamespace(
            stage_b_u0_patch_rank=True,
            stage_b_u0_category_preserving_patch_gate=True,
        )
        self.assertEqual(
            joint_eval._validate_category_gate_sweep_mode(
                args, cfg, ["refcoco_val", "refcocog_val"]
            ),
            (0.0, 1.0),
        )

        bad_cases = (
            (dict(skip_tn=False), cfg, ["refcoco_val"], "TN evaluation"),
            (
                {},
                types.SimpleNamespace(stage_b_u0_patch_rank=True),
                ["refcoco_val"],
                "gate config",
            ),
            ({}, cfg, ["refcoco_testA"], "only Ref val"),
            (
                dict(no_per_example_records=True),
                cfg,
                ["refcoco_val"],
                "records are required",
            ),
        )
        for overrides, bad_cfg, splits, message in bad_cases:
            values = vars(args).copy()
            values.update(overrides)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    joint_eval._validate_category_gate_sweep_mode(
                        types.SimpleNamespace(**values), bad_cfg, splits
                    )

    def test_category_gate_sweep_gap_values_are_strict(self):
        for values, message in (
            ([1.0], "at least two"),
            ([0.0, -1.0], "finite and non-negative"),
            ([0.0, float("inf")], "finite and non-negative"),
            ([1.0, 1.0], "unique"),
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    joint_eval._normalize_category_gate_sweep_gaps(values)

    def test_category_gate_sweep_reuses_one_forward_and_writes_gap_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "refcocog_val.jsonl"
            manifest_row = {
                "sample_id": "ref:g:1",
                "image_id": 1,
                "ann_id": 2,
                "ref_id": 3,
                "sent_id": 4,
                "eval_split": "refcocog_val",
            }
            manifest_path.write_text(
                json.dumps(manifest_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            target = {
                "sample_id": "ref:g:1",
                "image_id": torch.tensor(1),
                "ann_id": torch.tensor(2),
                "ref_id": torch.tensor(3),
                "sent_id": torch.tensor(4),
                "boxes": torch.tensor([[0.8, 0.8, 0.1, 0.1]]),
            }
            batch = (object(), [target])

            class OneBatchLoader:
                dataset = [object()]

                def __iter__(self):
                    yield batch

                def __len__(self):
                    return 1

            outputs = {
                "pred_logits": torch.tensor([[[0.0], [2.0], [-2.0]]]),
                "pred_boxes": torch.tensor(
                    [[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]]
                ),
                "stage_b_gdino_expression_token_mask": torch.ones(
                    (1, 1), dtype=torch.bool
                ),
                "stage_b_u0_teacher_rank_score": torch.tensor(
                    [[0.9, 0.8, 0.1]]
                ),
                "stage_b_u0_category_gate_patch_score": torch.tensor(
                    [[0.0, 1.0, -1.0]]
                ),
                "stage_b_u0_candidate_mask": torch.ones(
                    (1, 3), dtype=torch.bool
                ),
            }
            outputs["stage_b_gdino_base_score"] = (
                outputs["pred_logits"].sigmoid().mean(dim=-1)
            )
            configured = joint_eval._category_gate_sweep_query_scores(
                outputs, [0.0, 3.0]
            )[0][1]
            outputs["stage_b_u0_rank_score"] = configured
            cfg = types.SimpleNamespace(
                stage_b_u0_patch_rank=True,
                stage_b_u0_category_preserving_patch_gate=True,
                stage_b_u0_category_gate_max_gap=0.0,
            )

            with patch.object(
                joint_eval, "_build_loader", return_value=OneBatchLoader()
            ), patch.object(
                joint_eval,
                "_forward_ref_batch",
                return_value=(outputs, [target], None),
            ) as forward:
                rows = joint_eval.evaluate_refcoco_category_gate_sweep(
                    cfg=cfg,
                    model=object(),
                    ckpt_path="u2.pth",
                    datasetinfo={"anno": str(manifest_path)},
                    dataset_name="refcocog_val",
                    device=torch.device("cpu"),
                    batch_size=1,
                    num_workers=0,
                    seed=600042,
                    topks=[1, 3],
                    max_gaps=[0.0, 3.0],
                    amp=False,
                    max_batches=0,
                    log_every=0,
                    records_output_dir=root / "records",
                    include_base_expert=True,
                )

            self.assertEqual(forward.call_count, 1)
            self.assertEqual(
                [row["category_gate_max_gap"] for row in rows],
                [0.0, 3.0, 0.0, 3.0],
            )
            self.assertEqual([row["acc50"] for row in rows], [1.0, 0.0, 1.0, 1.0])
            self.assertEqual(
                [row["seed"] for row in rows],
                [600042, 600042, 600042, 600042],
            )
            self.assertNotEqual(rows[0]["run_id"], rows[1]["run_id"])
            self.assertEqual(
                [row["category_gate_rank_expert"] for row in rows],
                ["teacher_r100", "teacher_r100", "base_b58", "base_b58"],
            )
            records = []
            for row in rows:
                path = Path(row["records_jsonl"])
                self.assertTrue(path.is_file())
                records.append(json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(records[0]["category_gate_eligible_queries"], 1)
            self.assertEqual(records[0]["category_gate_winner_query"], 1)
            self.assertEqual(records[1]["category_gate_eligible_queries"], 3)
            self.assertEqual(records[1]["category_gate_winner_query"], 0)
            self.assertEqual(records[1]["category_gate_teacher_winner_query"], 0)
            self.assertEqual(records[1]["category_gate_patch_winner_query"], 1)
            self.assertEqual(records[2]["category_gate_rank_expert"], "base_b58")
            self.assertEqual(records[3]["category_gate_winner_query"], 1)
            self.assertEqual(records[3]["category_gate_base_winner_query"], 1)

    def test_u0_datasetinfos_keep_support_while_old_adapter_stays_no_support(self):
        data_root = Path("/tmp/pivot-data")
        anno = Path("/tmp/tn.jsonl")
        old_tn = tn_eval._make_datasetinfo(
            data_root,
            anno,
            adapter_eval_scope="benchmark_dataft_alltn",
        )
        self.assertTrue(old_tn["stage_b_gdino_adapter_no_support"])
        self.assertNotIn("support_patch_tsv", old_tn)

        u0_tn = tn_eval._make_datasetinfo(
            data_root,
            anno,
            adapter_eval_scope="benchmark_dataft_alltn",
            u0_patch_rank=True,
        )
        self.assertNotIn("stage_b_gdino_adapter_no_support", u0_tn)
        self.assertIn("support_patch_tsv", u0_tn)

        old_ref = joint_eval._make_datasetinfo(
            data_root,
            "refcoco_val",
            anno,
            adapter_no_support=True,
        )
        self.assertTrue(old_ref["stage_b_gdino_adapter_no_support"])
        self.assertNotIn("support_patch_tsv", old_ref)

        u0_ref = joint_eval._make_datasetinfo(
            data_root,
            "refcoco_val",
            anno,
            u0_patch_rank=True,
        )
        self.assertNotIn("stage_b_gdino_adapter_no_support", u0_ref)
        self.assertTrue(u0_ref["stage_b_gdino_adapter_ref_eval"])
        self.assertIn("support_patch_tsv", u0_ref)
        with self.assertRaisesRegex(ValueError, "requires its support patch"):
            joint_eval._make_datasetinfo(
                data_root,
                "refcoco_val",
                anno,
                adapter_no_support=True,
                u0_patch_rank=True,
            )

    def test_u0_tn_pair_duplicates_the_same_support_and_is_not_patch_only(self):
        model = _DummyU0Model()
        negative, positive, _targets, valid = tn_eval._forward_pair(
            model, _batch(), torch.device("cpu"), amp=False
        )
        self.assertEqual(valid.tolist(), [True])
        self.assertEqual(len(model.calls), 1)
        samples, kwargs = model.calls[0]
        self.assertEqual(int(samples.tensors.shape[0]), 2)
        self.assertIs(kwargs["patch_only"], False)
        self.assertTrue(kwargs["disable_patch_dn"])
        self.assertEqual(len(kwargs["targets"]), 2)
        self.assertEqual(tuple(kwargs["patches"].shape), (2, 3, 2, 2))
        self.assertTrue(torch.equal(kwargs["patches"][0], kwargs["patches"][1]))
        self.assertTrue(
            torch.equal(
                positive["stage_b_gdino_confidence_score"],
                torch.tensor([[0.0, 1.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                negative["stage_b_gdino_confidence_score"],
                torch.tensor([[2.0, 3.0]]),
            )
        )

    def test_u0_joint_ref_forwards_support_and_selects_only_u0_rank(self):
        model = _DummyU0Model()
        outputs, targets, query_scores = joint_eval._forward_ref_batch(
            _u0_cfg(), model, _batch(), torch.device("cpu"), amp=False
        )
        self.assertIsNone(query_scores)
        self.assertEqual(len(targets), 1)
        _samples, kwargs = model.calls[0]
        self.assertIs(kwargs["patch_only"], False)
        self.assertEqual(tuple(kwargs["patches"].shape), (1, 3, 2, 2))
        self.assertEqual(
            joint_eval._adapter_ref_score_key(_u0_cfg()),
            "stage_b_u0_rank_score",
        )
        observed, valid = joint_eval._phrase_scores(
            outputs,
            targets,
            "phrase_to_token_mask",
            adapter_score_key=joint_eval._adapter_ref_score_key(_u0_cfg()),
        )
        self.assertTrue(torch.equal(observed, outputs["stage_b_u0_rank_score"]))
        self.assertEqual(valid.tolist(), [True])
        outputs.pop("stage_b_u0_rank_score")
        with self.assertRaisesRegex(KeyError, "stage_b_u0_rank_score"):
            joint_eval._phrase_scores(
                outputs,
                targets,
                "phrase_to_token_mask",
                adapter_score_key=joint_eval._adapter_ref_score_key(_u0_cfg()),
            )

    def test_u0_joint_tn_uses_only_confidence_even_when_rank_disagrees(self):
        model = _DummyU0Model()
        result = joint_eval._forward_tn_batch(
            _u0_cfg(), model, _batch(), torch.device("cpu"), amp=False
        )
        negative, positive = result[0], result[1]
        self.assertTrue(
            torch.equal(result[4], negative["stage_b_gdino_confidence_score"])
        )
        self.assertTrue(
            torch.equal(result[6], positive["stage_b_gdino_confidence_score"])
        )
        self.assertFalse(torch.equal(result[4], negative["stage_b_u0_rank_score"]))
        self.assertFalse(torch.equal(result[6], positive["stage_b_u0_rank_score"]))

        slot_outputs = {
            "stage_b_gdino_confidence_score": torch.tensor([[0.2, 0.8]]),
            "stage_b_u0_rank_score": torch.tensor([[99.0, -99.0]]),
        }
        self.assertTrue(
            torch.equal(
                tn_eval._slot_scores(slot_outputs, _u0_cfg(), 1.0),
                torch.tensor([[[0.2], [0.8]]]),
            )
        )
        slot_outputs.pop("stage_b_gdino_confidence_score")
        with self.assertRaisesRegex(KeyError, "stage_b_gdino_confidence_score"):
            tn_eval._slot_scores(slot_outputs, _u0_cfg(), 1.0)

    def test_u0_missing_support_or_caption_fails_closed(self):
        model = _DummyU0Model()
        with self.assertRaisesRegex(KeyError, "missing patch"):
            tn_eval._forward_pair(
                model,
                _batch(_tn_target(include_patch=False)),
                torch.device("cpu"),
                amp=False,
            )
        missing_caption = _tn_target()
        missing_caption.pop("caption")
        with self.assertRaisesRegex(KeyError, "requires a non-empty caption"):
            joint_eval._forward_ref_batch(
                _u0_cfg(),
                model,
                _batch(missing_caption),
                torch.device("cpu"),
                amp=False,
            )


if __name__ == "__main__":
    unittest.main()
