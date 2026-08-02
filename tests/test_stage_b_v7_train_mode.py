import unittest

import torch
from torch import nn

from engine import _set_stage_b_v7_training_mode


class _DummyVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.9))
        self.head = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))


class _DummyStageBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_a = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.9))
        self.stage_b_verifier = _DummyVerifier()


class StageBV7TrainModeTest(unittest.TestCase):
    def test_frozen_towers_stay_eval_while_verifier_head_trains(self):
        model = _DummyStageBModel()
        model.train()
        _set_stage_b_v7_training_mode(model)

        self.assertFalse(model.training)
        self.assertFalse(model.stage_a.training)
        self.assertTrue(model.stage_b_verifier.training)
        self.assertFalse(model.stage_b_verifier.bert.training)
        self.assertTrue(model.stage_b_verifier.head.training)

        x = torch.ones((2, 4))
        first = model.stage_a(x)
        second = model.stage_a(x)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
