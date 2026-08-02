import types
import unittest
from unittest import mock

import torch

import engine as engine_module
from engine import GracefulTrainingExit, train_one_epoch
from util.misc import NestedTensor


class _LinearLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, samples, captions=None, patches=None, patch_global=None, **kwargs):
        del captions, patches, patch_global, kwargs
        return {"linear_loss": self.weight * samples.tensors.float().mean()}


class _LinearCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_dict = {"loss_linear": 1.0}

    def forward(self, outputs, targets, cap_list, captions):
        del targets, cap_list, captions
        return {"loss_linear": outputs["linear_loss"]}


class _DeferredQueueCriterion(_LinearCriterion):
    def __init__(self):
        super().__init__()
        self.pending = None
        self.deferred = []
        self.committed = []

    def forward(self, outputs, targets, cap_list, captions):
        result = super().forward(outputs, targets, cap_list, captions)
        self.pending = outputs["linear_loss"].detach().reshape(1)
        return result

    def defer_tail_queue_payload(self):
        if self.pending is not None:
            self.deferred.append(self.pending)
            self.pending = None

    def commit_tail_queue(self, step_succeeded):
        payloads = list(self.deferred)
        if self.pending is not None:
            payloads.append(self.pending)
        if step_succeeded:
            self.committed.append(torch.cat(payloads).tolist())
        self.pending = None
        self.deferred.clear()


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _FakeGradScaler:
    def __init__(self, *, skip=False):
        self.scale_value = 8.0
        self.skip = bool(skip)
        self.scale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        self.scale_calls += 1
        return loss

    def unscale_(self, optimizer):
        del optimizer

    def step(self, optimizer):
        self.step_calls += 1
        if not self.skip:
            optimizer.step()

    def update(self):
        self.update_calls += 1
        if self.skip:
            self.scale_value /= 2.0


def _batch(value):
    samples = NestedTensor(
        torch.tensor([[[[float(value)]]]]),
        torch.zeros((1, 1, 1), dtype=torch.bool),
    )
    targets = [{"caption": "object .", "cap_list": ["object"]}]
    return samples, targets


def _args(
    *,
    accumulation_steps,
    max_updates=0,
    checkpoint_interval=0,
    amp=False,
    onecycle=True,
):
    return types.SimpleNamespace(
        amp=bool(amp),
        amp_max_consecutive_skips=0,
        debug=False,
        gradient_accumulation_steps=accumulation_steps,
        iter_checkpoint_interval=checkpoint_interval,
        max_train_iters=max_updates,
        onecyclelr=bool(onecycle),
        patch_only=False,
        stage_b=False,
        stage_b_gdino_score_adapter=False,
        stage_b_legacy_global_gate=False,
        stage_b_v11_fixed_text=False,
        stage_b_v15_separate_grad_clip=False,
        stage_b_v7=False,
    )


class GradientAccumulationTest(unittest.TestCase):
    def _run(
        self,
        values,
        *,
        accumulation_steps,
        max_updates=0,
        start_iter=0,
        start_optimizer_updates=0,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        amp=False,
        onecycle=True,
        callback=None,
    ):
        model = model or _LinearLossModel()
        optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = scheduler or _CountingScheduler()
        result = train_one_epoch(
            model,
            criterion or _LinearCriterion(),
            [_batch(value) for value in values],
            optimizer,
            torch.device("cpu"),
            epoch=0,
            max_norm=0.0,
            wo_class_error=True,
            lr_scheduler=scheduler,
            args=_args(
                accumulation_steps=accumulation_steps,
                max_updates=max_updates,
                checkpoint_interval=max_updates,
                amp=amp,
                onecycle=onecycle,
            ),
            scaler=scaler,
            start_iter=start_iter,
            start_optimizer_updates=start_optimizer_updates,
            iter_checkpoint_fn=callback,
        )
        return model, optimizer, scheduler, result

    def test_two_micro_batches_form_one_optimizer_update(self):
        checkpoints = []
        model = _LinearLossModel()
        scheduler = _CountingScheduler()
        with self.assertRaises(GracefulTrainingExit):
            self._run(
                [1.0, 3.0, 9.0, 11.0],
                accumulation_steps=2,
                max_updates=1,
                model=model,
                scheduler=scheduler,
                callback=lambda **kwargs: checkpoints.append(kwargs),
            )
        self.assertAlmostEqual(float(model.weight.detach()), 0.8, places=6)
        self.assertEqual(scheduler.steps, 1)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["iteration"], 2)
        self.assertEqual(checkpoints[0]["optimizer_updates"], 1)
        self.assertEqual(checkpoints[0]["reason"], "max_train_iters")

    def test_default_one_preserves_one_update_per_micro_batch(self):
        checkpoints = []
        model = _LinearLossModel()
        scheduler = _CountingScheduler()
        with self.assertRaises(GracefulTrainingExit):
            self._run(
                [1.0, 3.0, 9.0],
                accumulation_steps=1,
                max_updates=2,
                model=model,
                scheduler=scheduler,
                callback=lambda **kwargs: checkpoints.append(kwargs),
            )
        self.assertAlmostEqual(float(model.weight.detach()), 0.6, places=6)
        self.assertEqual(scheduler.steps, 2)
        self.assertEqual(checkpoints[0]["iteration"], 2)
        self.assertEqual(checkpoints[0]["optimizer_updates"], 2)

    def test_resume_uses_micro_iteration_and_optimizer_update_separately(self):
        model = _LinearLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = _CountingScheduler()
        checkpoints = []
        with self.assertRaises(GracefulTrainingExit):
            self._run(
                [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                accumulation_steps=2,
                max_updates=2,
                start_iter=2,
                start_optimizer_updates=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                callback=lambda **kwargs: checkpoints.append(kwargs),
            )
        self.assertAlmostEqual(float(model.weight.detach()), 0.4, places=6)
        self.assertEqual(scheduler.steps, 1)
        self.assertEqual(checkpoints[0]["iteration"], 4)
        self.assertEqual(checkpoints[0]["optimizer_updates"], 2)

    def test_epoch_tail_is_normalized_by_actual_micro_batch_count(self):
        model, _, scheduler, result = self._run(
            [1.0, 3.0, 5.0], accumulation_steps=2
        )
        self.assertAlmostEqual(float(model.weight.detach()), 0.3, places=6)
        self.assertEqual(scheduler.steps, 2)
        self.assertEqual(result["optimizer_updates"], 2)

    def test_resume_rejects_non_boundary_micro_iteration(self):
        with self.assertRaisesRegex(ValueError, "optimizer-step boundary"):
            self._run(
                [1.0, 3.0, 5.0],
                accumulation_steps=2,
                start_iter=1,
                start_optimizer_updates=0,
            )

    def test_amp_scaler_and_scheduler_step_once_per_accumulated_update(self):
        model = _LinearLossModel()
        scheduler = _CountingScheduler()
        scaler = _FakeGradScaler()
        checkpoints = []
        with self.assertRaises(GracefulTrainingExit):
            self._run(
                [1.0, 3.0],
                accumulation_steps=2,
                max_updates=1,
                model=model,
                scheduler=scheduler,
                scaler=scaler,
                amp=True,
                callback=lambda **kwargs: checkpoints.append(kwargs),
            )
        self.assertEqual(scaler.scale_calls, 2)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)
        self.assertEqual(scheduler.steps, 1)
        self.assertEqual(checkpoints[0]["optimizer_updates"], 1)

    def test_amp_skip_does_not_advance_update_budget_or_scheduler(self):
        model = _LinearLossModel()
        scheduler = _CountingScheduler()
        scaler = _FakeGradScaler(skip=True)
        _, _, _, result = self._run(
            [1.0, 3.0],
            accumulation_steps=2,
            max_updates=1,
            model=model,
            scheduler=scheduler,
            scaler=scaler,
            amp=True,
        )
        self.assertAlmostEqual(float(model.weight.detach()), 1.0, places=6)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(scheduler.steps, 0)

    def test_update_budget_can_continue_across_epoch_boundary(self):
        model = _LinearLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = _CountingScheduler()
        _, _, _, first_epoch = self._run(
            [1.0, 1.0],
            accumulation_steps=1,
            max_updates=3,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.assertEqual(first_epoch["optimizer_updates"], 2)
        checkpoints = []
        with self.assertRaises(GracefulTrainingExit):
            self._run(
                [1.0, 1.0],
                accumulation_steps=1,
                max_updates=3,
                start_optimizer_updates=2,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                callback=lambda **kwargs: checkpoints.append(kwargs),
            )
        self.assertEqual(checkpoints[0]["iteration"], 1)
        self.assertEqual(checkpoints[0]["optimizer_updates"], 3)

    def test_stateful_queue_payloads_commit_only_after_full_update(self):
        criterion = _DeferredQueueCriterion()
        with self.assertRaises(GracefulTrainingExit):
            self._run(
                [1.0, 3.0],
                accumulation_steps=2,
                max_updates=1,
                criterion=criterion,
                callback=lambda **kwargs: None,
            )
        self.assertEqual(criterion.committed, [[1.0, 3.0]])
        self.assertEqual(criterion.deferred, [])
        self.assertIsNone(criterion.pending)

    def test_epoch_tail_limit_advances_step_scheduler_before_checkpoint(self):
        scheduler = _CountingScheduler()
        checkpoints = []
        with self.assertRaises(GracefulTrainingExit):
            self._run(
                [1.0, 3.0],
                accumulation_steps=1,
                max_updates=2,
                scheduler=scheduler,
                onecycle=False,
                callback=lambda **kwargs: checkpoints.append(kwargs),
            )
        self.assertEqual(scheduler.steps, 1)
        self.assertEqual(checkpoints[0]["iteration"], 0)
        self.assertEqual(checkpoints[0]["optimizer_updates"], 2)
        self.assertTrue(checkpoints[0]["epoch_finished"])
        self.assertEqual(checkpoints[0]["reason"], "max_train_iters")

    def test_remote_rank_stop_request_exits_all_ranks_at_update_boundary(self):
        checkpoints = []
        with mock.patch.object(engine_module, "_sync_bool_any", return_value=True):
            with self.assertRaises(GracefulTrainingExit):
                self._run(
                    [1.0, 3.0],
                    accumulation_steps=1,
                    callback=lambda **kwargs: checkpoints.append(kwargs),
                )
        self.assertEqual(checkpoints[0]["iteration"], 1)
        self.assertEqual(checkpoints[0]["reason"], "signal")

    def test_distributed_amp_step_disagreement_aborts_before_scheduler(self):
        scheduler = _CountingScheduler()
        with mock.patch.object(engine_module, "_sync_bool_any", return_value=True), \
             mock.patch.object(engine_module, "_sync_bool_all", return_value=False):
            with self.assertRaisesRegex(FloatingPointError, "differed across"):
                self._run(
                    [1.0, 3.0],
                    accumulation_steps=1,
                    scheduler=scheduler,
                )
        self.assertEqual(scheduler.steps, 0)


if __name__ == "__main__":
    unittest.main()
