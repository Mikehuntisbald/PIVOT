import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
)


def _ddp_tail_queue_worker(rank, world_size, init_file, result_queue):
    from engine import _sync_bool_all

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            predicate_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=8,
            tail_queue_min_count=4,
            tail_queue_positive_quantile=0.25,
            tail_queue_negative_quantile=0.75,
            tail_queue_temperature=0.5,
        )
        ious = torch.tensor([[0.9, 0.1], [0.8, 0.2]])

        history_positive = (
            torch.tensor([[1.0, -1.0], [0.8, -1.0]], requires_grad=True)
            if rank == 0
            else torch.tensor([[0.6, -1.0], [0.4, -1.0]], requires_grad=True)
        )
        history_negative = (
            torch.tensor([[0.1, -1.0], [0.2, -1.0]], requires_grad=True)
            if rank == 0
            else torch.tensor([[0.3, -1.0], [0.35, -1.0]], requires_grad=True)
        )
        history_gate = (
            torch.tensor([10.0, 11.0])
            if rank == 0
            else torch.tensor([12.0, 13.0])
        )
        initial_losses = criterion(
            history_positive,
            ious,
            local_tn_logits=history_negative,
            positive_confidence_gate_logits=history_gate,
            local_tn_mask=torch.tensor([True, True]),
        )
        gathered_history_gate = criterion._pending_tail_payload[:, 4].tolist()
        if float(initial_losses["loss_fixed_text_tail_queue"].detach()) != 0.0:
            raise AssertionError("tail loss must remain off until history is committed")
        criterion.commit_tail_queue(
            _sync_bool_all(rank == 0, torch.device("cpu"))
        )
        if int(criterion.tail_positive_count) != 0:
            raise AssertionError("one-rank AMP skip must discard the payload globally")
        criterion(
            history_positive,
            ious,
            local_tn_logits=history_negative,
            positive_confidence_gate_logits=history_gate,
            local_tn_mask=torch.tensor([True, True]),
        )
        criterion.commit_tail_queue(
            _sync_bool_all(True, torch.device("cpu"))
        )

        current_positive = (
            torch.tensor([[0.2, -1.0], [0.1, -1.0]], requires_grad=True)
            if rank == 0
            else torch.tensor([[0.0, -1.0], [-0.1, -1.0]], requires_grad=True)
        )
        current_negative = (
            torch.tensor([[0.5, -1.0], [0.6, -1.0]], requires_grad=True)
            if rank == 0
            else torch.tensor([[0.7, -1.0], [0.8, -1.0]], requires_grad=True)
        )
        losses = criterion(
            current_positive,
            ious,
            local_tn_logits=current_negative,
            positive_confidence_gate_logits=torch.zeros(2),
            local_tn_mask=torch.tensor([True, True]),
        )
        tail_loss = losses["loss_fixed_text_tail_queue"]
        tail_loss.backward()
        positive_gradient = float(current_positive.grad[:, 0].sum())
        negative_gradient = float(current_negative.grad[:, 0].sum())
        criterion.commit_tail_queue(True)

        state_equal = True
        for value in criterion.state_dict().values():
            gathered = [torch.empty_like(value) for _ in range(world_size)]
            dist.all_gather(gathered, value)
            state_equal = state_equal and all(
                torch.equal(gathered[0], remote) for remote in gathered[1:]
            )
        result_queue.put(
            {
                "rank": rank,
                "loss": float(tail_loss.detach()),
                "positive_gradient": positive_gradient,
                "negative_gradient": negative_gradient,
                "state_equal": state_equal,
                "gathered_history_gate": gathered_history_gate,
                "positive_queue": criterion.tail_positive_queue.tolist(),
                "negative_queue": criterion.tail_negative_queue.tolist(),
                "positive_ptr": int(criterion.tail_positive_ptr),
                "negative_ptr": int(criterion.tail_negative_ptr),
                "positive_count": int(criterion.tail_positive_count),
                "negative_count": int(criterion.tail_negative_count),
            }
        )
    finally:
        dist.destroy_process_group()


class StageBV14CriterionTest(unittest.TestCase):
    @staticmethod
    def _assert_no_gradient(tensor):
        if tensor.grad is not None:
            if float(tensor.grad.abs().sum()) != 0.0:
                raise AssertionError(f"unexpected gradient: {tensor.grad}")

    def test_absolute_losses_are_class_balanced_and_target_local(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            predicate_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=1.0,
            local_absolute_gamma=0.0,
            predicate_absolute_weight=1.0,
            predicate_absolute_gamma=0.0,
        )
        positive = torch.tensor([[0.0, 8.0, -3.0]], requires_grad=True)
        tn = torch.tensor([[0.0, -8.0, 3.0]], requires_grad=True)
        positive_predicate = torch.tensor([[0.0, 7.0, -2.0]], requires_grad=True)
        tn_predicate = torch.tensor([[0.0, -7.0, 2.0]], requires_grad=True)
        ious = torch.tensor([[0.9, 0.2, 0.4]])

        losses = criterion(
            positive,
            ious,
            local_tn_logits=tn,
            local_tn_mask=torch.tensor([True]),
            positive_predicate_logits=positive_predicate,
            local_tn_predicate_logits=tn_predicate,
            predicate_pair_valid=torch.tensor([True]),
        )
        losses["loss_stage_b_fixed_text"].backward()

        self.assertAlmostEqual(
            float(losses["loss_fixed_text_local_absolute"].detach()),
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )
        self.assertLess(float(positive.grad[0, 0]), 0.0)
        self.assertGreater(float(tn.grad[0, 0]), 0.0)
        self.assertEqual(float(positive.grad[0, 1:].abs().sum()), 0.0)
        self.assertEqual(float(tn.grad[0, 1:].abs().sum()), 0.0)
        self.assertLess(float(positive_predicate.grad[0, 0]), 0.0)
        self.assertGreater(float(tn_predicate.grad[0, 0]), 0.0)
        self.assertEqual(float(positive_predicate.grad[0, 1:].abs().sum()), 0.0)
        self.assertEqual(float(tn_predicate.grad[0, 1:].abs().sum()), 0.0)

    def test_omitted_confidence_arguments_are_strict_legacy_aliases(self):
        criterion = StageBFixedTextCriterion(
            batch_tail_separation_weight=1.0,
            local_absolute_weight=1.0,
            predicate_absolute_weight=1.0,
        )
        rank = torch.tensor([[0.4, -0.2, -0.3], [0.1, -0.4, -0.5]])
        rank_tn = torch.tensor([[0.2, 0.1, 0.0], [-0.1, -0.2, -0.3]])
        predicate = torch.tensor([[0.3, 0.0, 0.0], [0.2, 0.0, 0.0]])
        predicate_tn = torch.tensor([[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]])
        global_tn = torch.tensor([[0.2, 0.1, 0.0], [0.3, 0.2, 0.1]])
        ious = torch.tensor([[0.9, 0.2, 0.1], [0.8, 0.2, 0.1]])
        common = dict(
            local_tn_logits=rank_tn,
            positive_predicate_logits=predicate,
            local_tn_predicate_logits=predicate_tn,
            predicate_pair_valid=torch.tensor([True, True]),
            global_tn_logits=global_tn,
            global_tn_verified=torch.tensor([True, True]),
        )
        legacy = criterion(rank, ious, **common)
        explicit_alias = criterion(
            rank,
            ious,
            confidence_logits=rank,
            local_tn_confidence_logits=rank_tn,
            global_tn_confidence_logits=global_tn,
            **common,
        )
        self.assertEqual(legacy.keys(), explicit_alias.keys())
        for name in legacy:
            self.assertTrue(torch.equal(legacy[name], explicit_alias[name]), name)

    def test_tail_queue_commits_only_successful_steps_and_roundtrips(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            predicate_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=4,
            tail_queue_min_count=2,
            tail_queue_temperature=0.2,
        )
        ious = torch.tensor([[0.9, 0.1], [0.8, 0.2]])

        def run(values):
            positive = torch.tensor(
                [[values[0], -1.0], [values[1], -1.0]], requires_grad=True
            )
            tn = torch.tensor(
                [[values[2], -1.0], [values[3], -1.0]], requires_grad=True
            )
            losses = criterion(
                positive,
                ious,
                local_tn_logits=tn,
                local_tn_mask=torch.tensor([True, True]),
            )
            return positive, tn, losses

        _positive, _tn, losses = run((1.0, 0.8, 0.4, 0.3))
        self.assertEqual(int(criterion.tail_positive_count), 0)
        self.assertEqual(float(losses["loss_fixed_text_tail_queue"].detach()), 0.0)
        criterion.commit_tail_queue(False)
        self.assertEqual(int(criterion.tail_positive_count), 0)

        _positive, _tn, _losses = run((1.0, 0.8, 0.4, 0.3))
        criterion.commit_tail_queue(True)
        self.assertEqual(int(criterion.tail_positive_count), 2)
        self.assertEqual(int(criterion.tail_negative_count), 2)

        positive, tn, losses = run((0.7, 0.6, 0.6, 0.5))
        queue_loss = losses["loss_fixed_text_tail_queue"]
        self.assertTrue(bool(torch.isfinite(queue_loss).item()))
        self.assertGreater(float(queue_loss.detach()), 0.0)
        queue_loss.backward()
        self.assertLess(float(positive.grad[:, 0].sum()), 0.0)
        self.assertGreater(float(tn.grad[:, 0].sum()), 0.0)
        criterion.commit_tail_queue(True)
        self.assertEqual(int(criterion.tail_positive_count), 4)

        state = criterion.state_dict()
        restored = StageBFixedTextCriterion(
            tail_queue_size=4,
            tail_queue_min_count=2,
        )
        restored.load_state_dict(state, strict=True)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_tail_queue_requires_commit_between_forwards(self):
        criterion = StageBFixedTextCriterion(
            tail_queue_size=4,
            tail_queue_min_count=1,
        )
        logits = torch.tensor([[0.0, -1.0]])
        ious = torch.tensor([[0.9, 0.1]])
        tn = torch.tensor([[0.0, -1.0]])
        criterion(logits, ious, local_tn_logits=tn)
        with self.assertRaisesRegex(RuntimeError, "not committed"):
            criterion(logits, ious, local_tn_logits=tn)
        criterion.commit_tail_queue(False)
        criterion(logits, ious, local_tn_logits=tn)
        criterion.commit_tail_queue(True)

    def test_tail_queue_zero_min_count_waits_for_nonempty_history(self):
        criterion = StageBFixedTextCriterion(
            tail_queue_weight=1.0,
            tail_queue_size=4,
            tail_queue_min_count=0,
        )
        logits = torch.tensor([[0.0, -1.0]], requires_grad=True)
        ious = torch.tensor([[0.9, 0.1]])
        tn = torch.tensor([[0.2, -1.0]], requires_grad=True)
        losses = criterion(logits, ious, local_tn_logits=tn)
        self.assertEqual(float(losses["loss_fixed_text_tail_queue"].detach()), 0.0)
        criterion.commit_tail_queue(True)
        losses = criterion(logits, ious, local_tn_logits=tn)
        self.assertGreater(float(losses["loss_fixed_text_tail_queue"].detach()), 0.0)
        criterion.commit_tail_queue(False)

    def test_tail_queue_best_iou_selection_ignores_nonfinite_candidates(self):
        criterion = StageBFixedTextCriterion(
            tail_queue_size=4,
            tail_queue_min_count=1,
        )
        logits = torch.tensor([[9.0, 0.7]])
        ious = torch.tensor([[float("nan"), 0.9]])
        tn = torch.tensor([[8.0, 0.2]])
        criterion(logits, ious, local_tn_logits=tn)
        criterion.commit_tail_queue(True)
        self.assertEqual(int(criterion.tail_positive_count), 1)
        self.assertEqual(int(criterion.tail_negative_count), 1)
        self.assertEqual(float(criterion.tail_positive_queue[0]), float(logits[0, 1]))
        self.assertEqual(float(criterion.tail_negative_queue[0]), float(tn[0, 1]))

    def test_two_rank_gloo_tail_queue_is_bitwise_synced_and_differentiable(self):
        context = mp.get_context("spawn")
        result_queue = context.SimpleQueue()
        with tempfile.TemporaryDirectory() as temporary_directory:
            init_file = f"{temporary_directory}/gloo_init"
            mp.spawn(
                _ddp_tail_queue_worker,
                args=(2, init_file, result_queue),
                nprocs=2,
                join=True,
            )
            results = sorted((result_queue.get(), result_queue.get()), key=lambda x: x["rank"])

        for result in results:
            self.assertTrue(result["state_equal"])
            self.assertEqual(
                result["gathered_history_gate"], [10.0, 11.0, 12.0, 13.0]
            )
            self.assertGreater(result["loss"], 0.0)
            self.assertLess(result["positive_gradient"], 0.0)
            self.assertGreater(result["negative_gradient"], 0.0)
            self.assertEqual(result["positive_ptr"], 0)
            self.assertEqual(result["negative_ptr"], 0)
            self.assertEqual(result["positive_count"], 8)
            self.assertEqual(result["negative_count"], 8)

        self.assertEqual(results[0]["positive_queue"], results[1]["positive_queue"])
        self.assertEqual(results[0]["negative_queue"], results[1]["negative_queue"])
        self.assertTrue(
            torch.equal(
                torch.tensor(results[0]["positive_queue"]),
                torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, -0.1]),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(results[0]["negative_queue"]),
                torch.tensor([0.1, 0.2, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8]),
            )
        )

    def test_decoupled_rank_and_confidence_losses_have_disjoint_gradients(self):
        criterion = StageBFixedTextCriterion(
            batch_tail_separation_weight=1.0,
            global_tn_negative_weight=1.0,
            global_tn_tail_weight=1.0,
            local_absolute_weight=1.0,
        )
        ious = torch.tensor([[0.9, 0.2, 0.1]])

        rank = torch.tensor([[0.1, -0.2, -0.4]], requires_grad=True)
        rank_tn = torch.tensor([[0.3, 0.0, -0.1]], requires_grad=True)
        confidence = torch.tensor([[-0.5, 0.2, 0.1]], requires_grad=True)
        confidence_tn = torch.tensor([[0.7, 0.4, 0.3]], requires_grad=True)
        global_rank_tn = torch.tensor([[-9.0, -8.0, -7.0]], requires_grad=True)
        global_confidence_tn = torch.tensor([[0.8, 0.6, 0.4]], requires_grad=True)
        losses = criterion(
            rank,
            ious,
            local_tn_logits=rank_tn,
            confidence_logits=confidence,
            local_tn_confidence_logits=confidence_tn,
            global_tn_logits=global_rank_tn,
            global_tn_confidence_logits=global_confidence_tn,
            global_tn_verified=torch.tensor([True]),
        )
        confidence_loss = (
            losses["loss_fixed_text_local_absolute"]
            + losses["loss_fixed_text_global_tn_negative"]
            + losses["loss_fixed_text_global_tn_tail"]
            + losses["loss_fixed_text_batch_tail"]
        )
        confidence_loss.backward()
        self._assert_no_gradient(rank)
        self._assert_no_gradient(rank_tn)
        self._assert_no_gradient(global_rank_tn)
        self.assertGreater(float(confidence.grad.abs().sum()), 0.0)
        self.assertGreater(float(confidence_tn.grad.abs().sum()), 0.0)
        self.assertGreater(float(global_confidence_tn.grad.abs().sum()), 0.0)

        rank = torch.tensor([[0.1, -0.2, -0.4]], requires_grad=True)
        rank_tn = torch.tensor([[0.3, 0.0, -0.1]], requires_grad=True)
        confidence = torch.tensor([[-0.5, 0.2, 0.1]], requires_grad=True)
        confidence_tn = torch.tensor([[0.7, 0.4, 0.3]], requires_grad=True)
        positive_predicate = torch.tensor([[0.2, 0.0, 0.0]], requires_grad=True)
        tn_predicate = torch.tensor([[0.4, 0.0, 0.0]], requires_grad=True)
        losses = criterion(
            rank,
            ious,
            local_tn_logits=rank_tn,
            confidence_logits=confidence,
            local_tn_confidence_logits=confidence_tn,
            positive_predicate_logits=positive_predicate,
            local_tn_predicate_logits=tn_predicate,
            predicate_pair_valid=torch.tensor([True]),
        )
        rank_loss = (
            losses["loss_fixed_text_listwise"]
            + losses["loss_fixed_text_local_tn_rank"]
            + losses["loss_fixed_text_predicate_tn_rank"]
        )
        rank_loss.backward()
        self.assertGreater(float(rank.grad.abs().sum()), 0.0)
        self.assertGreater(float(rank_tn.grad.abs().sum()), 0.0)
        self.assertGreater(float(positive_predicate.grad.abs().sum()), 0.0)
        self.assertGreater(float(tn_predicate.grad.abs().sum()), 0.0)
        self._assert_no_gradient(confidence)
        self._assert_no_gradient(confidence_tn)

    def test_global_tail_queue_uses_confidence_max_and_backpropagates_to_max(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            predicate_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=4,
            tail_queue_min_count=1,
            tail_queue_positive_quantile=0.25,
            tail_queue_negative_quantile=0.75,
            tail_queue_temperature=0.5,
            tail_queue_global_scores=True,
        )
        ious = torch.tensor([[0.9, 0.1, 0.1]])
        candidate_mask = torch.tensor([[True, True, False]])
        history_rank = torch.tensor([[9.0, -9.0, 20.0]], requires_grad=True)
        history_rank_tn = torch.tensor([[8.0, -8.0, 20.0]], requires_grad=True)
        history_confidence = torch.tensor([[0.2, 1.5, 100.0]], requires_grad=True)
        history_confidence_tn = torch.tensor([[0.1, 1.2, 100.0]], requires_grad=True)
        criterion(
            history_rank,
            ious,
            candidate_mask,
            local_tn_logits=history_rank_tn,
            confidence_logits=history_confidence,
            local_tn_confidence_logits=history_confidence_tn,
        )
        criterion.commit_tail_queue(True)
        self.assertAlmostEqual(float(criterion.tail_positive_queue[0]), 1.5)
        self.assertAlmostEqual(float(criterion.tail_negative_queue[0]), 1.2)

        rank = torch.tensor([[9.0, -9.0, 20.0]], requires_grad=True)
        rank_tn = torch.tensor([[8.0, -8.0, 20.0]], requires_grad=True)
        confidence = torch.tensor([[0.3, 1.0, 100.0]], requires_grad=True)
        confidence_tn = torch.tensor([[0.2, 1.4, 100.0]], requires_grad=True)
        losses = criterion(
            rank,
            ious,
            candidate_mask,
            local_tn_logits=rank_tn,
            confidence_logits=confidence,
            local_tn_confidence_logits=confidence_tn,
        )
        tail_loss = losses["loss_fixed_text_tail_queue"]
        self.assertGreater(float(tail_loss.detach()), 0.0)
        tail_loss.backward()
        self._assert_no_gradient(rank)
        self._assert_no_gradient(rank_tn)
        self.assertLess(float(confidence.grad[0, 1]), 0.0)
        self.assertGreater(float(confidence_tn.grad[0, 1]), 0.0)
        self.assertEqual(float(confidence.grad[0, [0, 2]].abs().sum()), 0.0)
        self.assertEqual(float(confidence_tn.grad[0, [0, 2]].abs().sum()), 0.0)
        criterion.commit_tail_queue(False)

    def test_fpr95_queue_penalizes_every_verified_negative_global_max(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=16,
            tail_queue_min_count=1,
            tail_queue_positive_quantile=0.25,
            tail_queue_negative_quantile=0.75,
            tail_queue_temperature=0.2,
            tail_queue_margin=0.0,
            tail_queue_global_scores=True,
            tail_queue_objective="fpr95",
            tail_queue_pair_weight=0.25,
            tail_queue_pair_margin=0.0,
        )
        ious = torch.tensor([[0.1, 0.0], [0.2, 0.0]])
        history_positive = torch.tensor([[0.0, -1.0], [1.0, -1.0]])
        history_negative = torch.tensor([[-0.5, -1.0], [0.5, -1.0]])
        criterion(
            history_positive,
            ious,
            local_tn_logits=history_negative,
            confidence_logits=history_positive,
            positive_confidence_gate_logits=torch.zeros(2),
            local_tn_confidence_logits=history_negative,
            global_tn_verified=torch.tensor([True, True]),
        )
        criterion.commit_tail_queue(True)

        positive = torch.tensor([[0.2, -1.0], [0.8, -1.0]], requires_grad=True)
        negative = torch.tensor([[-0.8, -1.0], [0.4, -1.0]], requires_grad=True)
        positive_gate = torch.zeros(2, requires_grad=True)
        losses = criterion(
            positive,
            ious,
            local_tn_logits=negative,
            confidence_logits=positive,
            positive_confidence_gate_logits=positive_gate,
            local_tn_confidence_logits=negative,
            global_tn_verified=torch.tensor([True, True]),
        )
        loss = losses["loss_fixed_text_tail_queue"]
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        # Both samples' global TN maxima contribute, including the one below
        # the historical positive operating threshold.
        self.assertGreater(float(negative.grad[0, 0]), 0.0)
        self.assertGreater(float(negative.grad[1, 0]), 0.0)
        self.assertLess(float(positive.grad[:, 0].sum()), 0.0)
        self.assertLess(float(positive_gate.grad.sum()), 0.0)
        criterion.commit_tail_queue(False)

    def test_confidence_tn_training_scope_preserves_provenance_and_masks_losses(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=1.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=1.0,
            tail_queue_weight=1.0,
            tail_queue_size=8,
            tail_queue_min_count=8,
            tail_queue_global_scores=True,
            tail_queue_objective="fpr95",
        )
        candidate = torch.zeros(2, 1)
        ious = torch.ones(2, 1)
        positive_confidence = torch.ones(2, 1, requires_grad=True)
        tn_confidence = torch.tensor([[0.5], [8.0]], requires_grad=True)
        verified = torch.tensor([True, True])
        train_eligible = torch.tensor([True, False])

        losses = criterion(
            candidate,
            ious,
            local_tn_logits=candidate,
            confidence_logits=positive_confidence,
            positive_confidence_gate_logits=torch.zeros(2),
            local_tn_confidence_logits=tn_confidence,
            local_tn_mask=torch.ones(2, 1, dtype=torch.bool),
            global_tn_logits=candidate,
            global_tn_confidence_logits=tn_confidence,
            global_tn_verified=verified,
            confidence_tn_train_eligible=train_eligible,
            global_tn_candidate_mask=torch.ones(2, 1, dtype=torch.bool),
        )
        losses["loss_stage_b_fixed_text"].backward()

        self.assertGreater(float(tn_confidence.grad[0, 0]), 0.0)
        self.assertEqual(float(tn_confidence.grad[1, 0]), 0.0)
        self.assertEqual(float(losses["fixed_text_global_tn_sample_count"]), 1.0)
        self.assertEqual(
            float(losses["fixed_text_local_absolute_tn_sample_count"]), 1.0
        )
        self.assertEqual(
            float(losses["fixed_text_confidence_tn_train_eligible_count"]), 1.0
        )
        self.assertEqual(
            float(losses["fixed_text_confidence_tn_train_excluded_count"]), 1.0
        )
        self.assertTrue(bool(criterion._pending_tail_payload[0, 3] > 0.5))
        self.assertFalse(bool(criterion._pending_tail_payload[1, 3] > 0.5))
        self.assertTrue(bool((criterion._pending_tail_payload[:, 1] > 0.5).all()))
        criterion.commit_tail_queue(False)

    def test_fpr95_history_q05_uses_exact_evaluator_order_statistic(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=32,
            tail_queue_min_count=20,
            tail_queue_positive_quantile=0.05,
            tail_queue_temperature=0.2,
            tail_queue_global_scores=True,
            tail_queue_objective="fpr95",
        )
        ious = torch.zeros(20, 1)
        history_positive = torch.arange(20, dtype=torch.float32).view(20, 1)
        history_negative = torch.full((20, 1), -10.0)
        criterion(
            history_positive,
            ious,
            local_tn_logits=history_negative,
            positive_confidence_gate_logits=torch.zeros(20),
            global_tn_verified=torch.ones(20, dtype=torch.bool),
        )
        criterion.commit_tail_queue(True)

        losses = criterion(
            torch.zeros(20, 1),
            ious,
            local_tn_logits=torch.zeros(20, 1),
            positive_confidence_gate_logits=torch.zeros(20),
            global_tn_verified=torch.ones(20, dtype=torch.bool),
        )

        # At 95% TPR with 20 positives, the evaluator accepts 19 values and
        # therefore selects the second-smallest score. Linear q05 is 0.95.
        self.assertEqual(
            float(losses["fixed_text_tail_queue_positive_threshold"]), 1.0
        )
        criterion.commit_tail_queue(False)

    def test_fpr95_p3_cancels_common_confidence_translation(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=16,
            tail_queue_min_count=4,
            tail_queue_positive_quantile=0.05,
            tail_queue_temperature=0.1,
            tail_queue_margin=0.0,
            tail_queue_global_scores=True,
            tail_queue_objective="fpr95",
            tail_queue_pair_weight=0.25,
            tail_queue_pair_margin=0.05,
            tail_queue_positive_trust_weight=1.0,
            tail_queue_positive_trust_margin=0.02,
        )
        ious = torch.zeros(4, 1)
        verified = torch.ones(4, dtype=torch.bool)
        criterion(
            torch.tensor([[0.35], [0.45], [0.55], [0.65]]),
            ious,
            local_tn_logits=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
            positive_confidence_gate_logits=torch.zeros(4),
            global_tn_verified=verified,
        )
        criterion.commit_tail_queue(True)

        common_bias = torch.tensor(0.0, requires_grad=True)
        positive = torch.tensor([[0.4], [0.5], [0.6], [0.7]]) + common_bias
        negative = torch.tensor([[0.2], [0.3], [0.55], [0.8]]) + common_bias
        losses = criterion(
            positive,
            ious,
            local_tn_logits=negative,
            positive_confidence_gate_logits=common_bias.expand(4),
            global_tn_verified=verified,
        )
        losses["loss_fixed_text_tail_queue"].backward()

        self.assertAlmostEqual(float(common_bias.grad), 0.0, places=6)
        criterion.commit_tail_queue(False)

    def test_fpr95_positive_gate_trust_hinge_and_fail_closed_contract(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=8,
            tail_queue_min_count=2,
            tail_queue_positive_quantile=0.05,
            tail_queue_temperature=0.2,
            tail_queue_margin=0.0,
            tail_queue_global_scores=True,
            tail_queue_objective="fpr95",
            tail_queue_positive_trust_weight=1.0,
            tail_queue_positive_trust_margin=0.02,
        )
        ious = torch.zeros(3, 1)
        verified = torch.ones(3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "positive_confidence_gate_logits"):
            criterion(
                torch.tensor([[0.5], [0.7], [0.9]]),
                ious,
                local_tn_logits=torch.zeros(3, 1),
                global_tn_verified=verified,
            )

        criterion(
            torch.tensor([[0.5], [0.7], [0.9]]),
            ious,
            local_tn_logits=torch.zeros(3, 1),
            positive_confidence_gate_logits=torch.zeros(3),
            global_tn_verified=verified,
        )
        criterion.commit_tail_queue(True)

        positive = torch.tensor([[-4.0], [-3.0], [-2.0]], requires_grad=True)
        negative = torch.tensor([[0.6], [0.1], [-0.3]], requires_grad=True)
        positive_gate = torch.tensor([-0.01, -0.02, -0.03], requires_grad=True)
        losses = criterion(
            positive,
            ious,
            local_tn_logits=negative,
            positive_confidence_gate_logits=positive_gate,
            global_tn_verified=verified,
        )
        losses["loss_fixed_text_tail_queue"].backward()

        self.assertEqual(float(positive.grad.abs().sum()), 0.0)
        self.assertTrue(bool((negative.grad[:, 0] > 0.0).all().item()))
        self.assertAlmostEqual(
            float(positive_gate.grad[0]), float(positive_gate.grad[1]), places=7
        )
        self.assertAlmostEqual(
            float(positive_gate.grad[2] - positive_gate.grad[1]),
            -1.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(losses["fixed_text_tail_queue_positive_trust_loss"]),
            0.01 / 3.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(
                losses[
                    "fixed_text_tail_queue_positive_trust_violation_rate"
                ]
            ),
            1.0 / 3.0,
            places=6,
        )
        criterion.commit_tail_queue(False)

    def test_fpr95_positive_trust_top_quarter_targets_only_low_tail(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=16,
            tail_queue_min_count=8,
            tail_queue_positive_quantile=0.05,
            tail_queue_temperature=0.2,
            tail_queue_margin=0.0,
            tail_queue_global_scores=True,
            tail_queue_objective="fpr95",
            tail_queue_positive_trust_weight=1.0,
            tail_queue_positive_trust_margin=0.02,
            tail_queue_positive_trust_reduction_contract=(
                "top_quarter_cvar_v2"
            ),
        )
        ious = torch.zeros(8, 1)
        criterion(
            torch.linspace(0.2, 0.9, 8).view(8, 1),
            ious,
            local_tn_logits=torch.linspace(-0.8, -0.1, 8).view(8, 1),
            positive_confidence_gate_logits=torch.zeros(8),
            global_tn_verified=torch.ones(8, dtype=torch.bool),
        )
        criterion.commit_tail_queue(True)

        positive_gate = torch.tensor(
            [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09],
            requires_grad=True,
        )
        losses = criterion(
            torch.linspace(0.2, 0.9, 8).view(8, 1),
            ious,
            local_tn_logits=torch.zeros(8, 1),
            positive_confidence_gate_logits=positive_gate,
            global_tn_verified=torch.zeros(8, dtype=torch.bool),
        )

        losses["loss_fixed_text_tail_queue"].backward()
        self.assertAlmostEqual(
            float(losses["fixed_text_tail_queue_positive_trust_loss"]),
            0.065,
            places=6,
        )
        self.assertEqual(float(positive_gate.grad[:6].abs().sum()), 0.0)
        self.assertTrue(bool((positive_gate.grad[6:] < 0.0).all().item()))
        criterion.commit_tail_queue(False)

    def test_fpr95_exact_batch_lower_tail_routes_threshold_gradient_to_q05(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            local_absolute_weight=0.0,
            tail_queue_weight=1.0,
            tail_queue_size=32,
            tail_queue_min_count=16,
            tail_queue_positive_quantile=0.05,
            tail_queue_temperature=0.2,
            tail_queue_margin=0.0,
            tail_queue_global_scores=True,
            tail_queue_objective="fpr95",
            tail_queue_positive_gradient_contract=(
                "exact_batch_lower_tail_st_v2"
            ),
        )
        ious = torch.zeros(16, 1)
        verified = torch.ones(16, dtype=torch.bool)
        criterion(
            torch.linspace(0.0, 1.5, 16).view(16, 1),
            ious,
            local_tn_logits=torch.full((16, 1), -0.5),
            positive_confidence_gate_logits=torch.zeros(16),
            global_tn_verified=verified,
        )
        criterion.commit_tail_queue(True)

        positive = torch.linspace(0.2, 1.7, 16).requires_grad_()
        negative = torch.full((16, 1), 0.4, requires_grad=True)
        losses = criterion(
            positive.view(16, 1),
            ious,
            local_tn_logits=negative,
            positive_confidence_gate_logits=positive,
            global_tn_verified=verified,
        )
        losses["loss_fixed_text_tail_queue"].backward()

        # At B=16, exact 95% TPR accepts 16 values and therefore routes the
        # threshold gradient only through the minimum positive score.
        self.assertLess(float(positive.grad[0]), 0.0)
        self.assertEqual(float(positive.grad[1:].abs().sum()), 0.0)
        self.assertTrue(bool((negative.grad[:, 0] > 0.0).all().item()))
        criterion.commit_tail_queue(False)

    def test_fpr95_mean_plus_lower_tail_preserves_mean_and_tail_gradients(self):
        def run_contract(contract):
            criterion = StageBFixedTextCriterion(
                listwise_weight=0.0,
                local_tn_rank_weight=0.0,
                local_anchor_weight=0.0,
                global_tn_negative_weight=0.0,
                global_tn_tail_weight=0.0,
                local_absolute_weight=0.0,
                tail_queue_weight=1.0,
                tail_queue_size=32,
                tail_queue_min_count=16,
                tail_queue_positive_quantile=0.05,
                tail_queue_temperature=0.2,
                tail_queue_margin=0.0,
                tail_queue_global_scores=True,
                tail_queue_objective="fpr95",
                tail_queue_positive_gradient_contract=contract,
            )
            ious = torch.zeros(16, 1)
            verified = torch.ones(16, dtype=torch.bool)
            criterion(
                torch.linspace(0.0, 1.5, 16).view(16, 1),
                ious,
                local_tn_logits=torch.full((16, 1), -0.5),
                positive_confidence_gate_logits=torch.zeros(16),
                global_tn_verified=verified,
            )
            criterion.commit_tail_queue(True)

            positive = torch.linspace(0.2, 1.7, 16).requires_grad_()
            negative = torch.full((16, 1), 0.4, requires_grad=True)
            loss = criterion(
                positive.view(16, 1),
                ious,
                local_tn_logits=negative,
                positive_confidence_gate_logits=positive,
                global_tn_verified=verified,
            )["loss_fixed_text_tail_queue"]
            loss.backward()
            criterion.commit_tail_queue(False)
            return loss.detach(), positive.grad.detach(), negative.grad.detach()

        mean_loss, mean_positive_grad, mean_negative_grad = run_contract(
            "mean_translation_v1"
        )
        tail_loss, tail_positive_grad, tail_negative_grad = run_contract(
            "exact_batch_lower_tail_st_v2"
        )
        combined_loss, combined_positive_grad, combined_negative_grad = run_contract(
            "mean_plus_exact_lower_tail_st_v3"
        )
        quarter_loss, quarter_positive_grad, quarter_negative_grad = run_contract(
            "mean_plus_quarter_exact_lower_tail_st_v4"
        )
        bounded_loss, bounded_positive_grad, bounded_negative_grad = run_contract(
            "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5"
        )
        elementwise_loss, elementwise_positive_grad, elementwise_negative_grad = (
            run_contract(
                "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
            )
        )

        torch.testing.assert_close(combined_loss, mean_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(combined_loss, tail_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            combined_negative_grad, mean_negative_grad, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            combined_negative_grad, tail_negative_grad, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            combined_positive_grad,
            mean_positive_grad + tail_positive_grad,
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(quarter_loss, mean_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            quarter_negative_grad, mean_negative_grad, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            quarter_positive_grad,
            mean_positive_grad + 0.25 * tail_positive_grad,
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(bounded_loss, mean_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            bounded_negative_grad, mean_negative_grad, rtol=0.0, atol=0.0
        )
        gate_mean = torch.linspace(0.2, 1.7, 16).mean()
        bounded_mean_scale = 1.0 - torch.tanh(gate_mean).square()
        torch.testing.assert_close(
            bounded_positive_grad,
            bounded_mean_scale * mean_positive_grad
            + 0.0625 * tail_positive_grad,
            rtol=1e-6,
            atol=1e-7,
        )
        gate_values = torch.linspace(0.2, 1.7, 16)
        elementwise_mean_scale = 1.0 - torch.tanh(gate_values).square()
        torch.testing.assert_close(
            elementwise_loss, mean_loss, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            elementwise_negative_grad, mean_negative_grad, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            elementwise_positive_grad,
            elementwise_mean_scale * mean_positive_grad
            + 0.0625 * tail_positive_grad,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_fpr95_exact_batch_lower_tail_requires_positive_carrier(self):
        for contract in (
            "exact_batch_lower_tail_st_v2",
            "mean_plus_exact_lower_tail_st_v3",
            "mean_plus_quarter_exact_lower_tail_st_v4",
            "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5",
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6",
        ):
            with self.subTest(contract=contract):
                criterion = StageBFixedTextCriterion(
                    tail_queue_weight=1.0,
                    tail_queue_size=8,
                    tail_queue_min_count=1,
                    tail_queue_global_scores=True,
                    tail_queue_objective="fpr95",
                    tail_queue_positive_trust_weight=0.0,
                    tail_queue_positive_gradient_contract=contract,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "lower-tail gradient routing",
                ):
                    criterion(
                        torch.zeros(2, 1),
                        torch.ones(2, 1),
                        local_tn_logits=torch.zeros(2, 1),
                    )


if __name__ == "__main__":
    unittest.main()
