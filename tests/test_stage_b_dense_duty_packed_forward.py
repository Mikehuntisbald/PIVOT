import torch

from engine import (
    _PackedStageBDataLoader,
    _call_fixed_text_criterion_in_logical_batches,
    _scale_loss_for_logical_accumulation,
    _stage_b_token_edit_carrier_logits,
    _stage_b_token_role_carrier_logits,
)
from main import _stage_b_training_forward_batches_per_epoch
from util.misc import NestedTensor


def _batch(start: int, count: int, height: int, width: int):
    tensors = torch.arange(
        start,
        start + count * 3 * height * width,
        dtype=torch.float32,
    ).reshape(count, 3, height, width)
    mask = torch.zeros((count, height, width), dtype=torch.bool)
    targets = [{"row": start + index} for index in range(count)]
    return NestedTensor(tensors, mask), targets


def test_packed_loader_preserves_order_and_tail_logical_batch():
    logical_batches = [
        _batch(0, 2, 3, 4),
        _batch(1000, 2, 5, 2),
        _batch(2000, 2, 2, 3),
    ]
    loader = _PackedStageBDataLoader(logical_batches, pack_factor=2)

    assert len(loader) == 2
    packed = list(loader)
    first_samples, first_targets = packed[0]
    assert tuple(first_samples.tensors.shape) == (4, 3, 5, 4)
    assert [target["row"] for target in first_targets] == [0, 1, 1000, 1001]
    assert not bool(first_samples.mask[:2, :3, :4].any())
    assert bool(first_samples.mask[:2, 3:, :].all())
    assert bool(first_samples.mask[2:, :, 2:].all())

    tail_samples, tail_targets = packed[1]
    assert tuple(tail_samples.tensors.shape) == (2, 3, 2, 3)
    assert [target["row"] for target in tail_targets] == [2000, 2001]


def test_packed_epoch_geometry_uses_physical_forward_count():
    args = type(
        "Args",
        (),
        {
            "stage_b_dense_duty": True,
            "stage_b_dense_duty_forward_pack_factor": 2,
        },
    )()
    assert _stage_b_training_forward_batches_per_epoch(args, 887) == 444


class _QueueAwareCriterion:
    def __init__(self):
        self.pending = False
        self.calls = []
        self.deferred = []

    def __call__(self, *, candidate_logits, candidate_ious):
        assert not self.pending
        self.pending = True
        self.calls.append(candidate_logits.detach().clone())
        return {
            "loss": candidate_logits.mean(),
            "count": candidate_ious.new_tensor(float(candidate_logits.shape[0])),
        }

    def defer_tail_queue_payload(self):
        assert self.pending
        self.deferred.append(len(self.calls))
        self.pending = False


def test_logical_criterion_keeps_b16_statistics_and_queue_order():
    criterion = _QueueAwareCriterion()
    logits = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    ious = torch.ones_like(logits)

    losses, logical_batches = _call_fixed_text_criterion_in_logical_batches(
        criterion,
        logical_batch_size=2,
        candidate_logits=logits,
        candidate_ious=ious,
    )

    assert logical_batches == 2
    assert [int(call.shape[0]) for call in criterion.calls] == [2, 2]
    assert criterion.deferred == [1, 2]
    assert torch.equal(criterion.calls[0], logits[:2])
    assert torch.equal(criterion.calls[1], logits[2:])
    assert torch.equal(losses["loss"], logits.mean())
    assert float(losses["count"].item()) == 2.0


def test_engine_selects_detached_tn_slot_from_query_specific_base_logits():
    base_logits = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    base_logits.requires_grad_()
    outputs = {
        "stage_b_dense_duty_confidence_base_logits": base_logits,
        "stage_b_dense_duty_global_confidence_logits": torch.full_like(
            base_logits, 99.0
        ),
    }

    carrier = _stage_b_token_edit_carrier_logits(outputs)

    assert torch.equal(carrier, base_logits[..., 1])
    assert carrier.requires_grad is False
    assert _stage_b_token_edit_carrier_logits({}) is None


def test_engine_selects_detached_full_pair_query_specific_base_logits():
    base_logits = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    base_logits.requires_grad_()
    outputs = {
        "stage_b_dense_duty_confidence_base_logits": base_logits,
        "stage_b_dense_duty_global_confidence_logits": torch.full_like(
            base_logits, 99.0
        ),
    }

    carrier = _stage_b_token_role_carrier_logits(outputs)

    assert torch.equal(carrier, base_logits)
    assert carrier.requires_grad is False
    assert _stage_b_token_role_carrier_logits({}) is None


class _CarrierAwareCriterion:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        candidate_logits,
        candidate_ious,
        token_edit_carrier_logits,
        global_tn_candidate_mask,
    ):
        self.calls.append(
            (
                token_edit_carrier_logits.clone(),
                global_tn_candidate_mask.clone(),
            )
        )
        return {
            "loss": candidate_logits.mean(),
            "count": candidate_ious.new_tensor(float(candidate_logits.shape[0])),
        }

    def defer_tail_queue_payload(self):
        return None


def test_packed_logical_batches_keep_carrier_and_mask_rows_aligned():
    criterion = _CarrierAwareCriterion()
    base_logits = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
    base_logits.requires_grad_()
    carrier = _stage_b_token_edit_carrier_logits(
        {"stage_b_dense_duty_confidence_base_logits": base_logits}
    )
    candidate_mask = torch.tensor(
        [[True, False], [False, True], [True, True], [False, False]]
    )

    losses, logical_batches = _call_fixed_text_criterion_in_logical_batches(
        criterion,
        logical_batch_size=2,
        candidate_logits=torch.zeros((4, 2), requires_grad=True),
        candidate_ious=torch.ones((4, 2)),
        token_edit_carrier_logits=carrier,
        global_tn_candidate_mask=candidate_mask,
    )

    assert logical_batches == 2
    assert torch.equal(criterion.calls[0][0], carrier[:2])
    assert torch.equal(criterion.calls[1][0], carrier[2:])
    assert torch.equal(criterion.calls[0][1], candidate_mask[:2])
    assert torch.equal(criterion.calls[1][1], candidate_mask[2:])
    losses["loss"].backward()
    assert base_logits.grad is None


class _RoleCarrierAwareCriterion:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        candidate_logits,
        candidate_ious,
        candidate_mask,
        global_tn_candidate_mask,
        global_tn_verified,
        confidence_tn_train_eligible,
        token_role_carrier_logits,
        token_logits,
        score_token_mask,
        expression_valid_mask,
        token_supervision_valid,
        token_positive_mask,
        token_shared_mask,
        token_changed_mask,
        token_direct_trace_valid,
    ):
        self.calls.append(
            {
                name: value.detach().clone()
                for name, value in locals().items()
                if name not in {"self", "candidate_ious"}
            }
        )
        return {
            "loss": token_logits.mean(),
            "count": candidate_ious.new_tensor(float(candidate_logits.shape[0])),
        }

    def defer_tail_queue_payload(self):
        return None


def test_packed_role_complete_carrier_keeps_all_row_contracts_aligned():
    criterion = _RoleCarrierAwareCriterion()
    batch_size, candidates, tokens = 4, 2, 3
    base_logits = torch.arange(
        batch_size * candidates * 2, dtype=torch.float32
    ).reshape(batch_size, candidates, 2)
    base_logits.requires_grad_()
    carrier = _stage_b_token_role_carrier_logits(
        {"stage_b_dense_duty_confidence_base_logits": base_logits}
    )
    token_logits = torch.arange(
        batch_size * candidates * 2 * tokens, dtype=torch.float32
    ).reshape(batch_size, candidates, 2, tokens)
    token_logits.requires_grad_()
    positive_roles = torch.tensor(
        [
            [[True, False, False], [False, False, False]],
            [[False, True, False], [False, False, False]],
            [[False, False, True], [False, False, False]],
            [[True, True, False], [False, False, False]],
        ]
    )
    shared_roles = torch.tensor(
        [
            [[False, False, False], [False, True, False]],
            [[False, False, False], [True, False, False]],
            [[False, False, False], [True, True, False]],
            [[False, False, False], [False, False, True]],
        ]
    )
    changed_roles = torch.tensor(
        [
            [[False, False, False], [True, False, False]],
            [[False, False, False], [False, True, False]],
            [[False, False, False], [False, False, True]],
            [[False, False, False], [True, False, True]],
        ]
    )
    kwargs = {
        "candidate_logits": torch.arange(
            batch_size * candidates, dtype=torch.float32
        ).reshape(batch_size, candidates),
        "candidate_ious": torch.ones((batch_size, candidates)),
        "candidate_mask": torch.tensor(
            [[True, True], [True, False], [False, True], [True, True]]
        ),
        "global_tn_candidate_mask": torch.tensor(
            [[True, False], [False, True], [True, True], [False, False]]
        ),
        "global_tn_verified": torch.tensor([True, False, True, True]),
        "confidence_tn_train_eligible": torch.tensor(
            [True, False, False, True]
        ),
        "token_role_carrier_logits": carrier,
        "token_logits": token_logits,
        "score_token_mask": torch.tensor(
            [
                [[True, True, False], [True, True, False]],
                [[True, False, True], [True, False, True]],
                [[False, True, True], [False, True, True]],
                [[True, True, True], [True, True, True]],
            ]
        ),
        "expression_valid_mask": torch.tensor(
            [[True, True], [True, False], [False, True], [True, True]]
        ),
        "token_supervision_valid": torch.tensor([True, True, False, True]),
        "token_positive_mask": positive_roles,
        "token_shared_mask": shared_roles,
        "token_changed_mask": changed_roles,
        "token_direct_trace_valid": torch.tensor([True, False, False, True]),
    }

    losses, logical_batches = _call_fixed_text_criterion_in_logical_batches(
        criterion,
        logical_batch_size=2,
        **kwargs,
    )

    assert logical_batches == 2
    assert len(criterion.calls) == 2
    for logical_index, call in enumerate(criterion.calls):
        start = logical_index * 2
        end = start + 2
        for name, value in kwargs.items():
            if name == "candidate_ious":
                continue
            assert torch.equal(call[name], value[start:end]), name
    losses["loss"].backward()
    assert base_logits.grad is None
    assert token_logits.grad is not None


def test_logical_accumulation_weights_full_and_epoch_tail_updates():
    full = _scale_loss_for_logical_accumulation(
        torch.tensor(6.0),
        logical_batches_in_forward=2,
        logical_batches_in_update=4,
    )
    tail_two = _scale_loss_for_logical_accumulation(
        torch.tensor(6.0),
        logical_batches_in_forward=2,
        logical_batches_in_update=3,
    )
    tail_one = _scale_loss_for_logical_accumulation(
        torch.tensor(3.0),
        logical_batches_in_forward=1,
        logical_batches_in_update=3,
    )

    assert float(full.item()) == 3.0
    assert float((tail_two + tail_one).item()) == 5.0


def _reference_logical_loss(logits, logical_batch_size):
    return sum(
        logits[start : start + logical_batch_size].mean()
        for start in range(0, int(logits.shape[0]), logical_batch_size)
    ) / float(int(logits.shape[0]) // logical_batch_size)


def test_packed_accumulation_matches_four_logical_batch_gradients():
    packed_logits = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    packed_logits.requires_grad_()
    packed_criterion = _QueueAwareCriterion()

    first_losses, first_count = _call_fixed_text_criterion_in_logical_batches(
        packed_criterion,
        logical_batch_size=2,
        candidate_logits=packed_logits[:4],
        candidate_ious=torch.ones_like(packed_logits[:4]),
    )
    second_losses, second_count = _call_fixed_text_criterion_in_logical_batches(
        packed_criterion,
        logical_batch_size=2,
        candidate_logits=packed_logits[4:],
        candidate_ious=torch.ones_like(packed_logits[4:]),
    )
    packed_loss = _scale_loss_for_logical_accumulation(
        first_losses["loss"],
        logical_batches_in_forward=first_count,
        logical_batches_in_update=4,
    ) + _scale_loss_for_logical_accumulation(
        second_losses["loss"],
        logical_batches_in_forward=second_count,
        logical_batches_in_update=4,
    )
    packed_loss.backward()

    reference_logits = packed_logits.detach().clone().requires_grad_()
    reference_loss = _reference_logical_loss(reference_logits, 2)
    reference_loss.backward()

    assert torch.equal(packed_loss, reference_loss)
    assert torch.equal(packed_logits.grad, reference_logits.grad)
    assert packed_criterion.deferred == [1, 2, 3, 4]


def test_packed_epoch_tail_matches_three_logical_batch_gradients():
    packed_logits = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    packed_logits.requires_grad_()
    packed_criterion = _QueueAwareCriterion()

    first_losses, first_count = _call_fixed_text_criterion_in_logical_batches(
        packed_criterion,
        logical_batch_size=2,
        candidate_logits=packed_logits[:4],
        candidate_ious=torch.ones_like(packed_logits[:4]),
    )
    tail_losses, tail_count = _call_fixed_text_criterion_in_logical_batches(
        packed_criterion,
        logical_batch_size=2,
        candidate_logits=packed_logits[4:],
        candidate_ious=torch.ones_like(packed_logits[4:]),
    )
    packed_loss = _scale_loss_for_logical_accumulation(
        first_losses["loss"],
        logical_batches_in_forward=first_count,
        logical_batches_in_update=3,
    ) + _scale_loss_for_logical_accumulation(
        tail_losses["loss"],
        logical_batches_in_forward=tail_count,
        logical_batches_in_update=3,
    )
    packed_loss.backward()

    reference_logits = packed_logits.detach().clone().requires_grad_()
    reference_loss = _reference_logical_loss(reference_logits, 2)
    reference_loss.backward()

    assert torch.equal(packed_loss, reference_loss)
    assert torch.equal(packed_logits.grad, reference_logits.grad)
    assert packed_criterion.deferred == [1, 2]
