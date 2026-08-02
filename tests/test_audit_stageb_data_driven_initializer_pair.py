import unittest
from collections import OrderedDict

import torch

from tools.audit_stageb_data_driven_initializer_pair import (
    InitializerPairAuditError,
    _assert_equal_tensors,
    _common_non_rank_keys,
)


class AuditStageBDataDrivenInitializerPairTest(unittest.TestCase):
    def _states(self):
        common = [
            ("backbone.weight", torch.tensor([1.0, 2.0])),
            (
                "stage_b_data_driven_score_heads.confidence_branch.weight",
                torch.tensor([3.0]),
            ),
        ]
        absolute = OrderedDict(
            common
            + [
                (
                    "stage_b_data_driven_score_heads.rank_branch.absolute",
                    torch.tensor([4.0]),
                ),
                (
                    "stage_b_data_driven_score_heads._contract_version",
                    torch.tensor(1),
                ),
            ]
        )
        relational = OrderedDict(
            common
            + [
                (
                    "stage_b_data_driven_score_heads.rank_branch.relational",
                    torch.tensor([5.0]),
                ),
                (
                    "stage_b_data_driven_score_heads._contract_version",
                    torch.tensor(3),
                ),
            ]
        )
        return absolute, relational

    def test_common_partition_excludes_rank_and_contract(self):
        absolute, relational = self._states()
        keys = _common_non_rank_keys(absolute, relational)
        self.assertEqual(
            keys,
            [
                "backbone.weight",
                "stage_b_data_driven_score_heads.confidence_branch.weight",
            ],
        )
        _assert_equal_tensors(absolute, relational, keys)

    def test_common_value_or_key_drift_fails_closed(self):
        absolute, relational = self._states()
        relational["backbone.weight"] = torch.tensor([9.0, 2.0])
        with self.assertRaisesRegex(InitializerPairAuditError, "tensors differ"):
            _assert_equal_tensors(
                absolute,
                relational,
                _common_non_rank_keys(absolute, relational),
            )
        relational["extra.weight"] = torch.ones(1)
        with self.assertRaisesRegex(InitializerPairAuditError, "key coverage"):
            _common_non_rank_keys(absolute, relational)


if __name__ == "__main__":
    unittest.main()
