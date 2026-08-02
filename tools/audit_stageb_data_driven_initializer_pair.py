#!/usr/bin/env python3
"""Seal a fair absolute/relational b58-only initializer pair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    data_driven_tensor_state_sha256,
    validate_data_driven_initializer_payload,
    validate_data_driven_relational_initializer_payload,
)
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    _safe_load_checkpoint,
    stable_file_record,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


PAIR_SCHEMA = "pivot.stageb.data_driven_initializer_pair/v1"
RANK_PREFIX = "stage_b_data_driven_score_heads.rank_branch."
CONFIDENCE_PREFIXES = (
    "stage_b_data_driven_score_heads.confidence_branch.",
    "stage_b_data_driven_score_heads.confidence_gate.",
)
CONTRACT_KEY = "stage_b_data_driven_score_heads._contract_version"
EXPECTED_COMMON_KEY_COUNT = 1149
EXPECTED_ABSOLUTE_RANK_KEY_COUNT = 9
EXPECTED_RELATIONAL_RANK_KEY_COUNT = 40


class InitializerPairAuditError(RuntimeError):
    pass


def _config_record(path: Path) -> dict[str, Any]:
    return {
        "leaf": stable_file_record(path, label="paired initializer config"),
        "import_chain": [
            stable_file_record(item, label=f"config dependency {item.name}")
            for item in config_import_chain(path.resolve(), root=REPO_ROOT)
        ],
    }


def _build_expected_model(config: Path):
    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    for key in (
        "stage_b_data_driven_base_initializer_path",
        "stage_b_data_driven_base_initializer_sha256",
    ):
        if str(getattr(cfg, key, "") or "").strip():
            raise InitializerPairAuditError(
                f"paired initializer config must leave {key} blank"
            )
    torch.manual_seed(42)
    model, _criterion, _postprocessors = build_model_main(cfg)
    return model.eval(), cfg


def _common_non_rank_keys(
    absolute_state: Mapping[str, Any], relational_state: Mapping[str, Any]
) -> list[str]:
    def retained(state: Mapping[str, Any]) -> set[str]:
        return {
            str(key)
            for key in state
            if key != CONTRACT_KEY and not str(key).startswith(RANK_PREFIX)
        }

    absolute_keys = retained(absolute_state)
    relational_keys = retained(relational_state)
    if absolute_keys != relational_keys:
        raise InitializerPairAuditError(
            "paired common-state key coverage differs: "
            f"absolute_only={sorted(absolute_keys - relational_keys)[:8]}, "
            f"relational_only={sorted(relational_keys - absolute_keys)[:8]}"
        )
    return sorted(absolute_keys)


def _assert_equal_tensors(
    absolute_state: Mapping[str, Any],
    relational_state: Mapping[str, Any],
    keys: list[str],
) -> None:
    unequal = [
        key
        for key in keys
        if not (
            torch.is_tensor(absolute_state.get(key))
            and torch.is_tensor(relational_state.get(key))
            and torch.equal(absolute_state[key], relational_state[key])
        )
    ]
    if unequal:
        raise InitializerPairAuditError(
            f"paired common tensors differ: {unequal[:8]}"
        )


def _contract_value(state: Mapping[str, Any]) -> int:
    value = state.get(CONTRACT_KEY)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise InitializerPairAuditError("initializer score contract is malformed")
    return int(value.detach().cpu().item())


def build_receipt(
    *,
    absolute_path: Path,
    relational_path: Path,
    absolute_config: Path,
    relational_config: Path,
    b58_path: Path,
) -> dict[str, Any]:
    absolute_record = stable_file_record(
        absolute_path, label="absolute initializer"
    )
    relational_record = stable_file_record(
        relational_path, label="relational initializer"
    )
    if absolute_record["sha256"] == relational_record["sha256"]:
        raise InitializerPairAuditError("paired initializer files must differ")

    absolute_payload = _safe_load_checkpoint(
        absolute_path, label="absolute initializer"
    )
    absolute_model, absolute_cfg = _build_expected_model(absolute_config)
    validate_data_driven_initializer_payload(
        absolute_model,
        absolute_payload,
        checkpoint_label="paired absolute initializer",
    )
    del absolute_model

    relational_payload = _safe_load_checkpoint(
        relational_path, label="relational initializer"
    )
    relational_model, relational_cfg = _build_expected_model(relational_config)
    validate_data_driven_relational_initializer_payload(
        relational_model,
        relational_payload,
        checkpoint_label="paired relational initializer",
    )
    del relational_model

    if str(absolute_cfg.stage_b_data_driven_rank_architecture) != "absolute_token":
        raise InitializerPairAuditError("absolute pair config architecture drifted")
    if str(relational_cfg.stage_b_data_driven_rank_architecture) != "relational_v1":
        raise InitializerPairAuditError("relational pair config architecture drifted")
    if int(absolute_cfg.stage_b_data_driven_head_init_seed) != 42 or int(
        relational_cfg.stage_b_data_driven_head_init_seed
    ) != 42:
        raise InitializerPairAuditError("paired head initialization seed drifted")

    absolute_contract = absolute_payload["data_driven_initializer"]
    relational_contract = relational_payload[
        "data_driven_relational_initializer"
    ]
    if absolute_contract.get("config") != _config_record(
        absolute_config
    ) or relational_contract.get("config") != _config_record(relational_config):
        raise InitializerPairAuditError(
            "paired initializer artifact/config binding drifted"
        )
    b58_record = stable_file_record(b58_path, label="paired b58 source")
    if absolute_contract.get("b58_source") != b58_record or relational_contract.get(
        "b58_source"
    ) != b58_record:
        raise InitializerPairAuditError(
            "paired initializers do not bind the same requested b58 source"
        )
    if absolute_contract.get("invariants", {}).get(
        "b58_is_only_checkpoint_source"
    ) is not True or relational_contract.get("invariants", {}).get(
        "b58_is_only_checkpoint_source"
    ) is not True:
        raise InitializerPairAuditError("paired checkpoint-source boundary drifted")

    absolute_state = absolute_payload["model"]
    relational_state = relational_payload["model"]
    common_keys = _common_non_rank_keys(absolute_state, relational_state)
    if len(common_keys) != EXPECTED_COMMON_KEY_COUNT:
        raise InitializerPairAuditError(
            "paired common-state count drifted: "
            f"expected={EXPECTED_COMMON_KEY_COUNT}, observed={len(common_keys)}"
        )
    _assert_equal_tensors(absolute_state, relational_state, common_keys)
    absolute_common_hash = data_driven_tensor_state_sha256(
        absolute_state, common_keys
    )
    relational_common_hash = data_driven_tensor_state_sha256(
        relational_state, common_keys
    )
    if absolute_common_hash != relational_common_hash:
        raise InitializerPairAuditError("paired common-state hashes differ")

    absolute_rank_keys = sorted(
        key for key in absolute_state if key.startswith(RANK_PREFIX)
    )
    relational_rank_keys = sorted(
        key for key in relational_state if key.startswith(RANK_PREFIX)
    )
    if len(absolute_rank_keys) != EXPECTED_ABSOLUTE_RANK_KEY_COUNT or len(
        relational_rank_keys
    ) != EXPECTED_RELATIONAL_RANK_KEY_COUNT:
        raise InitializerPairAuditError("paired rank-state key count drifted")
    contract_values = {
        "absolute": _contract_value(absolute_state),
        "relational": _contract_value(relational_state),
    }
    if contract_values != {"absolute": 1, "relational": 3}:
        raise InitializerPairAuditError(
            f"paired score contracts drifted: {contract_values}"
        )

    confidence_keys = sorted(
        key for key in common_keys if key.startswith(CONFIDENCE_PREFIXES)
    )
    patch_keys = relational_contract["role_keys"]["random_patch_projection"]
    receipt = {
        "schema": PAIR_SCHEMA,
        "status": "passed",
        "seed": 42,
        "absolute_initializer": absolute_record,
        "relational_initializer": relational_record,
        "absolute_config": _config_record(absolute_config),
        "relational_config": _config_record(relational_config),
        "b58_source": b58_record,
        "common_non_rank_non_contract": {
            "key_count": len(common_keys),
            "tensor_sha256": absolute_common_hash,
        },
        "paired_patch": {
            "key_count": len(patch_keys),
            "tensor_sha256": data_driven_tensor_state_sha256(
                absolute_state, patch_keys
            ),
        },
        "paired_confidence": {
            "key_count": len(confidence_keys),
            "tensor_sha256": data_driven_tensor_state_sha256(
                absolute_state, confidence_keys
            ),
        },
        "architecture_intervention": {
            "absolute_rank_key_count": len(absolute_rank_keys),
            "absolute_rank_tensor_sha256": data_driven_tensor_state_sha256(
                absolute_state, absolute_rank_keys
            ),
            "relational_rank_key_count": len(relational_rank_keys),
            "relational_rank_tensor_sha256": data_driven_tensor_state_sha256(
                relational_state, relational_rank_keys
            ),
            "score_contract_values": contract_values,
        },
        "invariants": {
            "b58_is_only_tensor_checkpoint_source": True,
            "all_common_non_rank_non_contract_tensors_bitwise_equal": True,
            "patch_and_confidence_initialization_bitwise_equal": True,
            "rank_subtree_is_the_only_parameterized_architecture_intervention": True,
            "no_teacher_u1000_u5020_or_old_initializer_tensor_source": True,
        },
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute", type=Path, required=True)
    parser.add_argument("--relational", type=Path, required=True)
    parser.add_argument("--absolute-config", type=Path, required=True)
    parser.add_argument("--relational-config", type=Path, required=True)
    parser.add_argument("--b58", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"pair receipt output must be fresh: {output}")
    receipt = build_receipt(
        absolute_path=args.absolute.resolve(strict=True),
        relational_path=args.relational.resolve(strict=True),
        absolute_config=args.absolute_config.resolve(strict=True),
        relational_config=args.relational_config.resolve(strict=True),
        b58_path=args.b58.resolve(strict=True),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    record = stable_file_record(output, label="initializer pair receipt")
    print(
        f"[OK] wrote {record['path']} size={record['size_bytes']} "
        f"sha256={record['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
