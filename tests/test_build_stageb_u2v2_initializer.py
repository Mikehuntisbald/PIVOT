import unittest

import torch

from tools.build_stageb_u2v2_initializer import (
    U2V2InitializerError,
    _audit_sources,
)


def _sources():
    trunk = {}
    for index in range(187):
        trunk[f"backbone.tensor_{index}"] = torch.tensor([float(index)])
    for index in range(751):
        trunk[f"trunk.tensor_{index}"] = torch.tensor([float(index)])
    rank = {}
    for stem, count in (("rank_norm", 2), ("rank_trunk", 4), ("rank_output", 2)):
        for index in range(count):
            rank[f"stage_b_gdino_score_adapter.{stem}.tensor_{index}"] = torch.tensor([1.0])
    confidence = {}
    for index in range(12):
        confidence[f"stage_b_gdino_score_adapter.confidence_gate.tensor_{index}"] = torch.tensor([1.0])
    r100 = {**trunk, **rank, **confidence}
    c100 = {key: value.clone() for key, value in r100.items()}
    for key in confidence:
        c100[key] += 1.0
    stagea = {key: value.clone() for key, value in trunk.items()}
    for index in range(187):
        stagea[f"patch_encoder.backbone.tensor_{index}"] = trunk[f"backbone.tensor_{index}"].clone()
    for index in range(9):
        stagea[f"patch_extra.tensor_{index}"] = torch.tensor([float(index)])
    return stagea, r100, c100


class U2V2SourceAuditTest(unittest.TestCase):
    def test_exact_ownership_is_accepted(self):
        roles = _audit_sources(*_sources())
        self.assertEqual({key: len(value) for key, value in roles.items()}, {
            "trunk": 938, "rank": 8, "confidence": 12,
            "patch": 196, "patch_backbone": 187,
        })

    def test_shared_trunk_drift_is_rejected(self):
        stagea, r100, c100 = _sources()
        stagea["trunk.tensor_0"] += 1
        with self.assertRaisesRegex(U2V2InitializerError, "bitwise tensor drift"):
            _audit_sources(stagea, r100, c100)

    def test_rank_ownership_drift_is_rejected(self):
        stagea, r100, c100 = _sources()
        c100["stage_b_gdino_score_adapter.rank_norm.tensor_0"] += 1
        with self.assertRaisesRegex(U2V2InitializerError, "confidence12 only"):
            _audit_sources(stagea, r100, c100)


if __name__ == "__main__":
    unittest.main()
