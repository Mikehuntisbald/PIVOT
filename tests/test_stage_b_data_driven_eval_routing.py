import types
import unittest
from unittest import mock

import torch
from torch import nn

from engine import _set_stage_b_data_driven_training_mode
from models.GroundingDINO.stage_b_data_driven_score import (
    DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
    StageBDataDrivenCriterion,
    StageBDataDrivenScoreHeads,
    validate_data_driven_trained_checkpoint_payload,
    validate_stage_b_data_driven_score_checkpoint,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_stageb_tn_val as tn_eval
from tools import eval_text_groundingdino_refcoco_tn as joint_eval
from util.misc import NestedTensor


def _cfg(**overrides):
    values = {
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_confidence_trained": True,
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_v7": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _batch(*, paired=False):
    samples = NestedTensor(
        torch.ones(1, 3, 4, 4),
        torch.zeros(1, 4, 4, dtype=torch.bool),
    )
    target = {
        "stage_a_caption": "car .",
        "caption": "blue car .",
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "patch": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
    }
    if paired:
        target.update(
            cap_list=["red car", "blue car"],
            is_tn=torch.tensor([False, True]),
        )
    return samples, [target]


class _RecordingDataDrivenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_b_data_driven_score_heads = nn.Identity()
        self.calls = []

    def forward(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        batch_size = int(samples.tensors.shape[0])
        query_count = 3
        expressions = kwargs["stage_b_data_driven_expression_captions"]
        rank = torch.tensor([[3.0, 2.0, 1.0]]).expand(batch_size, -1).clone()
        confidence = torch.tensor([[1.0, 2.0, 3.0]]).expand(
            batch_size, -1
        ).clone()
        for index, expression in enumerate(expressions):
            if "blue" in expression:
                confidence[index].add_(10.0)
        return {
            "pred_boxes": torch.zeros(batch_size, query_count, 4),
            "pred_logits": torch.zeros(batch_size, query_count, 1),
            "pred_logits_patch": torch.zeros(batch_size, query_count),
            ref_eval._DATA_DRIVEN_RANK_SCORE_KEY: rank,
            tn_eval._DATA_DRIVEN_CONFIDENCE_SCORE_KEY: confidence,
            "stage_b_data_driven_expression_token_mask": torch.ones(
                batch_size, 2, dtype=torch.bool
            ),
        }


class _PatchEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.9))
        self.norm = nn.LayerNorm(4)


class _ModeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.9))
        self.bert = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.9))
        self.transformer = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.9))
        self.patch_encoder = _PatchEncoder(self.backbone)
        self.query_proj_for_patch = nn.Linear(4, 4)
        self.stage_b_data_driven_score_heads = StageBDataDrivenScoreHeads(
            4, rank_dim=3, confidence_dim=3, gate_hidden_dim=4
        )


class StageBDataDrivenEvalRoutingTest(unittest.TestCase):
    def test_gap3_query_diagnostics_record_gate_and_rank_failures(self):
        outputs = {
            ref_eval._DATA_DRIVEN_RANK_SCORE_KEY: torch.tensor(
                [[1.0, 3.0, 2.0]]
            ),
            "stage_b_data_driven_text_rank_score": torch.tensor(
                [[4.0, 1.0, 2.0]]
            ),
            "stage_b_data_driven_candidate_mask": torch.tensor(
                [[True, True, True]]
            ),
            "stage_b_data_driven_category_gate_eligible_mask": torch.tensor(
                [[False, True, True]]
            ),
            "stage_b_data_driven_category_gate_patch_score": torch.tensor(
                [[0.0, 3.0, 2.0]]
            ),
            "pred_logits_patch": torch.tensor([[[0.0], [3.0], [2.0]]]),
        }
        boxes = torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.2, 0.2],
                    [0.0, 0.0, 0.7, 0.7],
                ]
            ]
        )
        values = ref_eval._data_driven_query_diagnostic_values(
            outputs,
            boxes,
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            batch_index=0,
            ranked_query_indices=torch.tensor([1, 2, 0]),
        )
        self.assertEqual(
            values["data_driven_query_diagnostic_contract"],
            ref_eval._DATA_DRIVEN_QUERY_DIAGNOSTIC_CONTRACT,
        )
        self.assertEqual(values["data_driven_gap3_eligible_queries"], 2)
        self.assertEqual(values["data_driven_primary_positive_queries"], 1)
        self.assertEqual(values["data_driven_gap3_positive_queries"], 0)
        self.assertFalse(values["data_driven_gap3_oracle_correct50"])
        self.assertEqual(values["data_driven_final_winner_query"], 1)
        self.assertEqual(values["data_driven_raw_text_winner_query"], 0)
        self.assertEqual(values["data_driven_patch_winner_query"], 1)
        self.assertEqual(values["data_driven_gt_best_query"], 0)
        self.assertEqual(values["data_driven_gap3_oracle_query"], 2)
        self.assertEqual(
            values["data_driven_gap3_oracle_role"], "primary_ambiguous"
        )

        tied_outputs = dict(outputs)
        tied_outputs[ref_eval._DATA_DRIVEN_RANK_SCORE_KEY] = torch.tensor(
            [[1.0, 3.0, 3.0]]
        )
        tied_values = ref_eval._data_driven_query_diagnostic_values(
            tied_outputs,
            boxes,
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            batch_index=0,
            ranked_query_indices=torch.tensor([2, 1, 0]),
        )
        self.assertEqual(tied_values["data_driven_final_winner_query"], 2)
        self.assertEqual(
            tied_values["data_driven_final_score_argmax_query"], 1
        )
        self.assertEqual(
            tied_values["data_driven_final_score_tied_max_queries"], 2
        )

    def test_training_mode_keeps_frozen_generators_deterministic(self):
        model = _ModeModel().train()
        _set_stage_b_data_driven_training_mode(model, "rank_patch_only")
        heads = model.stage_b_data_driven_score_heads
        self.assertFalse(model.training)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.bert.training)
        self.assertFalse(model.transformer.training)
        self.assertTrue(heads.rank_branch.training)
        self.assertFalse(heads.confidence_branch.training)
        self.assertTrue(model.patch_encoder.input_proj.training)
        value = torch.ones(2, 4)
        self.assertTrue(torch.equal(model.backbone(value), model.backbone(value)))

        model.train()
        _set_stage_b_data_driven_training_mode(model, "confidence_pair")
        self.assertFalse(model.training)
        self.assertFalse(heads.rank_branch.training)
        self.assertTrue(heads.confidence_branch.training)
        self.assertTrue(heads.confidence_gate.training)
        self.assertFalse(model.patch_encoder.input_proj.training)

    def test_ref_forward_separates_canonical_and_full_expression(self):
        model = _RecordingDataDrivenModel()
        outputs, targets = ref_eval._forward(
            model, _batch(), torch.device("cpu"), amp=False, cfg=_cfg()
        )
        self.assertEqual(len(targets), 1)
        self.assertEqual(len(model.calls), 1)
        _samples, call = model.calls[0]
        self.assertEqual(call["captions"], ["car ."])
        self.assertEqual(
            call["stage_b_data_driven_expression_captions"], ["blue car ."]
        )
        self.assertNotIn("targets", call)
        self.assertFalse(call["patch_only"])
        self.assertEqual(tuple(call["patches"].shape), (1, 3, 2, 2))
        score = ref_eval._slot_scores(outputs, _cfg(), beta=999.0)
        self.assertTrue(
            torch.equal(
                score[..., 0], outputs[ref_eval._DATA_DRIVEN_RANK_SCORE_KEY]
            )
        )

    def test_tn_pair_uses_same_canonical_and_support_but_confidence_score(self):
        model = _RecordingDataDrivenModel()
        neg, pos, _targets, valid = tn_eval._forward_pair(
            model, _batch(paired=True), torch.device("cpu"), amp=False
        )
        self.assertTrue(valid.tolist() == [True])
        self.assertEqual(len(model.calls), 1)
        samples, call = model.calls[0]
        self.assertEqual(call["captions"], ["car .", "car ."])
        self.assertEqual(
            call["stage_b_data_driven_expression_captions"],
            ["red car .", "blue car ."],
        )
        self.assertTrue(torch.equal(samples.tensors[0], samples.tensors[1]))
        self.assertTrue(torch.equal(call["patches"][0], call["patches"][1]))
        observed = tn_eval._slot_scores(neg, _cfg(), beta=0.0)[..., 0]
        self.assertTrue(
            torch.equal(
                observed, neg[tn_eval._DATA_DRIVEN_CONFIDENCE_SCORE_KEY]
            )
        )
        self.assertFalse(
            torch.equal(
                observed,
                neg[ref_eval._DATA_DRIVEN_RANK_SCORE_KEY],
            )
        )
        self.assertGreater(
            float(neg[tn_eval._DATA_DRIVEN_CONFIDENCE_SCORE_KEY].max()),
            float(pos[tn_eval._DATA_DRIVEN_CONFIDENCE_SCORE_KEY].max()),
        )

    def test_joint_score_path_does_not_require_gdino_base_identity(self):
        outputs = {
            "pred_boxes": torch.zeros(1, 3, 4),
            "stage_b_data_driven_expression_token_mask": torch.tensor(
                [[True, True, False]]
            ),
            joint_eval._DATA_DRIVEN_RANK_SCORE_KEY: torch.tensor(
                [[0.1, 0.8, 0.2]]
            ),
        }
        score, valid = joint_eval._phrase_scores(
            outputs,
            [{}],
            "unused",
            adapter_score_key=joint_eval._DATA_DRIVEN_RANK_SCORE_KEY,
        )
        self.assertEqual(score.argmax(dim=1).tolist(), [1])
        self.assertTrue(valid.tolist() == [True])
        self.assertEqual(
            joint_eval._adapter_ref_score_key(_cfg()),
            joint_eval._DATA_DRIVEN_RANK_SCORE_KEY,
        )

    def test_score_routes_and_checkpoint_validation_fail_closed(self):
        with self.assertRaisesRegex(ValueError, r"\(B,Q\)"):
            ref_eval._slot_scores(
                {ref_eval._DATA_DRIVEN_RANK_SCORE_KEY: torch.zeros(1, 2, 1)},
                _cfg(),
                0.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            tn_eval._slot_scores(
                {
                    tn_eval._DATA_DRIVEN_CONFIDENCE_SCORE_KEY: torch.tensor(
                        [[float("nan")]]
                    )
                },
                _cfg(),
                0.0,
            )

        model = _ModeModel()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        validate_stage_b_data_driven_score_checkpoint(
            model, state, checkpoint_label="complete"
        )
        missing = dict(state)
        missing.pop(
            "stage_b_data_driven_score_heads.confidence_branch.query_proj.weight"
        )
        with self.assertRaisesRegex(ValueError, "key coverage"):
            validate_stage_b_data_driven_score_checkpoint(
                model, missing, checkpoint_label="missing"
            )

    def test_eval_checkpoint_metadata_rejects_cross_phase_attribution(self):
        model = _ModeModel()
        state = {key: value.clone() for key, value in model.state_dict().items()}

        def payload(experiment_id, mode, confidence_trained, token_weight=None):
            args = {
                "stage_b_data_driven_score": True,
                "stage_b_data_driven_experiment_id": experiment_id,
                "stage_b_data_driven_train_mode": mode,
                "stage_b_data_driven_category_complete": experiment_id != "DD0",
                "stage_b_data_driven_confidence_trained": confidence_trained,
                "seed": 42,
            }
            if token_weight is not None:
                args["stage_b_data_driven_token_weight"] = token_weight
                args[
                    "stage_b_data_driven_confidence_initializer_sha256"
                ] = "a" * 64
            return {"model": state, "args": args, "optimizer_updates": 10}

        dd1 = payload("DD1", "rank_patch_only", False)
        with self.assertRaisesRegex(ValueError, "phase metadata"):
            validate_data_driven_trained_checkpoint_payload(
                model,
                dd1,
                checkpoint_label="DD1-as-DD2",
                expected_experiment_id="DD2",
                expected_confidence_trained=True,
                expected_token_weight=0.0,
                expected_confidence_initializer_sha256="a" * 64,
            )

        dd2 = payload("DD2", "confidence_pair", True, token_weight=0.0)
        with self.assertRaisesRegex(ValueError, "phase metadata"):
            validate_data_driven_trained_checkpoint_payload(
                model,
                dd2,
                checkpoint_label="DD2-as-DD3",
                expected_experiment_id="DD3",
                expected_confidence_trained=True,
                expected_token_weight=1.0,
                expected_confidence_initializer_sha256="a" * 64,
            )

    def test_eval_checkpoint_binds_assignment_objective_weights(self):
        model = _ModeModel()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
            assignment_weight=1.0,
            deployment_weight=1.0,
        )
        payload = {
            "model": state,
            "criterion": criterion.state_dict(),
            "args": {
                "stage_b_data_driven_score": True,
                "stage_b_data_driven_experiment_id": "DD1",
                "stage_b_data_driven_train_mode": "rank_patch_only",
                "stage_b_data_driven_category_complete": True,
                "stage_b_data_driven_confidence_trained": False,
                "stage_b_data_driven_rank_supervision": (
                    DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT
                ),
                "stage_b_data_driven_assignment_weight": 1.0,
                "stage_b_data_driven_deployment_weight": 1.0,
                "seed": 42,
            },
            "optimizer_updates": 10,
        }
        validate_data_driven_trained_checkpoint_payload(
            model,
            payload,
            checkpoint_label="hardgap3",
            expected_experiment_id="DD1",
            expected_confidence_trained=False,
            expected_assignment_weight=1.0,
            expected_deployment_weight=1.0,
        )

        payload["args"]["stage_b_data_driven_deployment_weight"] = 0.0
        with self.assertRaisesRegex(
            ValueError, "deployment objective weight drifted"
        ):
            validate_data_driven_trained_checkpoint_payload(
                model,
                payload,
                checkpoint_label="pairtop1-as-hardgap3",
                expected_experiment_id="DD1",
                expected_confidence_trained=False,
                expected_assignment_weight=1.0,
                expected_deployment_weight=1.0,
            )

        payload["args"]["stage_b_data_driven_deployment_weight"] = 1.0
        del payload["args"]["stage_b_data_driven_assignment_weight"]
        with self.assertRaisesRegex(
            ValueError, "assignment objective weight drifted"
        ):
            validate_data_driven_trained_checkpoint_payload(
                model,
                payload,
                checkpoint_label="missing-assignment-weight",
                expected_experiment_id="DD1",
                expected_confidence_trained=False,
                expected_assignment_weight=1.0,
                expected_deployment_weight=1.0,
            )

    def test_ref_loader_forwards_assignment_objective_weights(self):
        model = _ModeModel()
        checkpoint = {
            "model": {
                key: value.clone() for key, value in model.state_dict().items()
            }
        }
        cfg = _cfg(
            modelname="test_data_driven",
            stage_b_data_driven_experiment_id="DD1",
            stage_b_data_driven_variant_id="DD1-PairTop1-HardGap3",
            stage_b_data_driven_rank_supervision=(
                "official_same_image_same_category_assignment_v1"
            ),
            stage_b_data_driven_rank_negative_iou_threshold=0.3,
            stage_b_data_driven_assignment_weight=1.0,
            stage_b_data_driven_deployment_weight=1.0,
        )

        def build(_cfg):
            return model, None, None

        validator_target = (
            "models.GroundingDINO.stage_b_data_driven_score."
            "validate_data_driven_trained_checkpoint_payload"
        )
        with (
            mock.patch.object(
                ref_eval.MODULE_BUILD_FUNCS, "get", return_value=build
            ),
            mock.patch.object(
                ref_eval, "_torch_load_compat", return_value=checkpoint
            ),
            mock.patch(validator_target) as validator,
        ):
            loaded = ref_eval._load_model(
                cfg, "/tmp/hardgap3.pth", torch.device("cpu")
            )

        self.assertIs(loaded, model)
        self.assertEqual(
            validator.call_args.kwargs["expected_assignment_weight"], 1.0
        )
        self.assertEqual(
            validator.call_args.kwargs["expected_deployment_weight"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
