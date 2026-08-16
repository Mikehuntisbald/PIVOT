import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.stageb_u2v3_category_admission_contract import (
    SCHEMA,
    TRAINABLE_KEYS,
    U2V3ContractError,
    build_training_contract,
    validate_runtime_payload,
)


class _StateModel(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def state_dict(self, *args, **kwargs):
        return self._state


def _payload():
    state = {key: torch.zeros(2) for key in TRAINABLE_KEYS}
    state["backbone.frozen"] = torch.arange(3, dtype=torch.float32)
    return {"model": state, "u2v2_initializer": {"schema": "pivot.stageb.u2v2_initializer/v1"}}


class U2V3CategoryAdmissionContractTest(unittest.TestCase):
    def test_contract_allows_only_declared_admission_changes(self):
        initializer = _payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c0.pth"
            torch.save(initializer, path)
            with mock.patch(
                "tools.stageb_u2v3_category_admission_contract.validate_initializer_payload",
                return_value={"schema": "pivot.stageb.u2v2_initializer/v1"},
            ):
                from tools.stageb_u2v3_category_admission_contract import _sha256_file

                digest = _sha256_file(path)
                contract = build_training_contract(
                    initializer, initializer_path=path, initializer_sha256=digest
                )
                self.assertEqual(contract["schema"], SCHEMA)
                trained = _payload()
                trained["model"][TRAINABLE_KEYS[0]] = torch.ones(2)
                trained["u2v3_category_admission"] = contract
                trained["args"] = {
                    "stage_b_u2v3_runtime_audit": {
                        "schema": "pivot.stageb.u2v3_category_admission_runtime/v1"
                    }
                }
                validated = validate_runtime_payload(
                    _StateModel(trained["model"]),
                    trained,
                    checkpoint_label="unit",
                    initializer_path=path,
                    initializer_sha256=digest,
                )
                self.assertEqual(validated["trainable_keys"], list(TRAINABLE_KEYS))

                trained["model"]["backbone.frozen"] = torch.ones(3)
                with self.assertRaisesRegex(U2V3ContractError, "changed frozen"):
                    validate_runtime_payload(
                        _StateModel(trained["model"]),
                        trained,
                        checkpoint_label="unit",
                        initializer_path=path,
                        initializer_sha256=digest,
                    )


if __name__ == "__main__":
    unittest.main()
