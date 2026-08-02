#!/usr/bin/env python3
"""Build and verify the b58-only native-residual rank initializer."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    data_driven_tensor_state_sha256,
)
from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    StageBGDINOScoreAdapter,
    validate_stage_b_gdino_score_adapter_checkpoint,
)
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    _safe_load_checkpoint,
    stable_file_record,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.native_residual_data_only_initializer/v1"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_gdino_score_adapter_rank_o64_direct_u500.py"
)
DEFAULT_B58 = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)
EXPECTED_B58_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
ADAPTER_PREFIX = "stage_b_gdino_score_adapter."
RANK_PARTS = ("rank_norm.", "rank_trunk.", "rank_output.")
CONFIDENCE_PARTS = (
    "confidence_norm.",
    "confidence_trunk.",
    "confidence_gate.",
)
FORBIDDEN_SOURCE_PREFIXES = (
    ADAPTER_PREFIX,
    "stage_b_u0_patch_rank_adapter.",
    "stage_b_data_driven_score_heads.",
    "stage_b_fixed_text_scorer.",
    "patch_encoder.",
    "query_proj_for_patch.",
)
FORBIDDEN_SOURCE_KEYS = frozenset({"patch_logit_scale"})


class NativeResidualInitializerError(RuntimeError):
    pass


def _source_model(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise NativeResidualInitializerError("b58 checkpoint has no model mapping")
    cleaned = utils.clean_state_dict(state)
    invalid = [
        key
        for key, value in cleaned.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise NativeResidualInitializerError(
            f"b58 model contains non-tensor values: {invalid[:8]}"
        )
    forbidden = sorted(
        key
        for key in cleaned
        if key in FORBIDDEN_SOURCE_KEYS
        or any(key.startswith(prefix) for prefix in FORBIDDEN_SOURCE_PREFIXES)
    )
    if forbidden:
        raise NativeResidualInitializerError(
            "b58 source already contains forbidden adapter/patch tensors: "
            f"{forbidden[:8]}"
        )
    return cleaned


def compose_native_residual_state(
    template: Mapping[str, torch.Tensor],
    b58: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, list[str]]]:
    """Exhaustively bind every template tensor to b58 or the new adapter."""

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    roles = {"b58_base": [], "random_identity_adapter": []}
    for key, template_value in template.items():
        if key in b58:
            value = b58[key]
            role = "b58_base"
        elif key.startswith(ADAPTER_PREFIX):
            value = template_value
            role = "random_identity_adapter"
        else:
            raise NativeResidualInitializerError(
                f"native-residual template contains an unbound tensor: {key}"
            )
        if not torch.is_tensor(value):
            raise NativeResidualInitializerError(f"state value is not a tensor: {key}")
        if value.dtype != template_value.dtype or tuple(value.shape) != tuple(
            template_value.shape
        ):
            raise NativeResidualInitializerError(
                f"initializer tensor shape/dtype mismatch at {key}"
            )
        result[key] = value.detach().cpu().clone()
        roles[role].append(key)
    if set(roles["b58_base"]) != set(b58):
        raise NativeResidualInitializerError(
            "b58 keys do not map exactly into the native-residual template: "
            f"missing={sorted(set(b58) - set(roles['b58_base']))[:8]}"
        )
    adapter_keys = {
        key for key in template if key.startswith(ADAPTER_PREFIX)
    }
    if set(roles["random_identity_adapter"]) != adapter_keys or not adapter_keys:
        raise NativeResidualInitializerError(
            "native-residual adapter tensor role is incomplete"
        )
    return result, roles


def _adapter_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(ADAPTER_PREFIX): value
        for key, value in state.items()
        if key.startswith(ADAPTER_PREFIX)
    }


def _require_zero_terminals(adapter: StageBGDINOScoreAdapter) -> None:
    terminal = {
        "rank_output.weight": adapter.rank_output.weight,
        "rank_output.bias": adapter.rank_output.bias,
        "confidence_gate.4.weight": adapter.confidence_gate[-1].weight,
        "confidence_gate.4.bias": adapter.confidence_gate[-1].bias,
    }
    nonzero = [
        name
        for name, value in terminal.items()
        if int(torch.count_nonzero(value.detach()).item()) != 0
    ]
    if nonzero:
        raise NativeResidualInitializerError(
            f"native-residual terminal tensors are not exact zero: {nonzero}"
        )


def functional_identity_check(adapter: StageBGDINOScoreAdapter) -> dict[str, bool]:
    """Prove exact identity, including ties and a nontrivial candidate mask."""

    _require_zero_terminals(adapter)
    generator = torch.Generator(device="cpu").manual_seed(20_260_723)
    query = torch.randn(
        3, 11, adapter.hidden_dim, generator=generator, dtype=torch.float32
    )
    base = torch.rand(3, 11, generator=generator, dtype=torch.float32)
    base[0, 0] = base[0, 1]
    base[1, 4] = base[1, 5]
    mask = torch.ones_like(base, dtype=torch.bool)
    mask[0, -2:] = False
    mask[2, -1] = False
    adapter.eval()
    with torch.inference_mode():
        output = adapter(query, base, mask)
    checks = {
        "rank_score_equals_base": bool(torch.equal(output["rank_score"], base)),
        "confidence_score_equals_base": bool(
            torch.equal(output["confidence_score"], base)
        ),
        "rank_residual_exact_zero": bool(
            int(torch.count_nonzero(output["rank_residual"]).item()) == 0
        ),
        "confidence_gate_exact_zero": bool(
            int(torch.count_nonzero(output["confidence_gate"]).item()) == 0
        ),
        "rank_winner_equals_base_winner": bool(
            torch.equal(
                output["rank_score"].argmax(dim=1), base.argmax(dim=1)
            )
        ),
        "confidence_winner_equals_base_winner": bool(
            torch.equal(
                output["confidence_score"].argmax(dim=1), base.argmax(dim=1)
            )
        ),
    }
    if not all(checks.values()):
        raise NativeResidualInitializerError(
            f"native-residual functional identity failed: {checks}"
        )
    return checks


def _config_binding(path: Path) -> dict[str, Any]:
    chain = config_import_chain(path.resolve(), root=REPO_ROOT)
    return {
        "leaf": stable_file_record(path, label="native-residual initializer config"),
        "import_chain": [
            stable_file_record(item, label=f"config dependency {item.name}")
            for item in chain
        ],
    }


def _build_template(config: Path, seed: int):
    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    required = {
        "stage_b_gdino_score_adapter": True,
        "stage_b_native_residual_data_only": True,
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "enable_patch_branch": False,
    }
    for key, expected in required.items():
        if getattr(cfg, key, None) != expected:
            raise NativeResidualInitializerError(
                f"initializer config requires {key}={expected!r}"
            )
    if float(getattr(cfg, "stage_b_gdino_confidence_weight", -1.0)) != 0.0:
        raise NativeResidualInitializerError(
            "initializer config must disable the confidence loss"
        )
    torch.manual_seed(int(seed))
    model, _criterion, _postprocessors = build_model_main(cfg)
    return model.eval(), cfg


def _expected_role_hashes(
    state: Mapping[str, torch.Tensor], roles: Mapping[str, Sequence[str]]
) -> dict[str, str]:
    return {
        f"{role}_tensor_sha256": data_driven_tensor_state_sha256(state, keys)
        for role, keys in roles.items()
    }


def validate_initializer_payload(
    expected_model: nn.Module,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "native_residual_initializer",
    }:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: top-level initializer keys drifted"
        )
    state = payload.get("model")
    contract = payload.get("native_residual_initializer")
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: initializer state/contract is malformed"
        )
    if contract.get("schema") != SCHEMA:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: initializer schema drifted"
        )
    expected_state = expected_model.state_dict()
    if set(state) != set(expected_state):
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: model key set differs from the template"
        )
    for key, expected in expected_state.items():
        value = state[key]
        if not torch.is_tensor(value) or value.dtype != expected.dtype or tuple(
            value.shape
        ) != tuple(expected.shape):
            raise NativeResidualInitializerError(
                f"{checkpoint_label}: tensor shape/dtype drift at {key}"
            )
    roles = contract.get("role_keys")
    if not isinstance(roles, Mapping) or set(roles) != {
        "b58_base",
        "random_identity_adapter",
    }:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: initializer roles drifted"
        )
    normalized_roles = {
        role: [str(key) for key in keys]
        for role, keys in roles.items()
        if isinstance(keys, (list, tuple))
    }
    if set(normalized_roles) != set(roles):
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: initializer role values are malformed"
        )
    flat = [key for keys in normalized_roles.values() for key in keys]
    if len(flat) != len(set(flat)) or set(flat) != set(state):
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: initializer roles are not disjoint/exhaustive"
        )
    if any(
        key.startswith(ADAPTER_PREFIX)
        for key in normalized_roles["b58_base"]
    ) or any(
        not key.startswith(ADAPTER_PREFIX)
        for key in normalized_roles["random_identity_adapter"]
    ):
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: base/adapter role prefix mismatch"
        )
    expected_hashes = _expected_role_hashes(state, normalized_roles)
    expected_hashes["full_model_tensor_sha256"] = (
        data_driven_tensor_state_sha256(state, sorted(state))
    )
    for key, expected in expected_hashes.items():
        if contract.get(key) != expected:
            raise NativeResidualInitializerError(
                f"{checkpoint_label}: {key} drifted"
            )
    source = contract.get("b58_source")
    if not isinstance(source, Mapping) or source.get("sha256") != EXPECTED_B58_SHA256:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: b58 source binding drifted"
        )
    if contract.get("checkpoint_sources") != [source]:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: b58 must be the sole checkpoint source"
        )
    if contract.get("training_allowed") is not True or contract.get(
        "teacher_logits_or_scores_consumed"
    ) is not False:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: training/teacher-source contract drifted"
        )
    invariants = contract.get("invariants")
    required_invariants = {
        "b58_is_only_checkpoint_source": True,
        "no_r100_p50_u0_stagea_or_teacher_logit_source": True,
        "rank_and_confidence_parameters_are_disjoint": True,
        "rank_output_exact_zero": True,
        "confidence_output_exact_zero": True,
        "native_rank_and_confidence_are_bitwise_identity": True,
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not value
        for key, value in required_invariants.items()
    ):
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: initializer invariants drifted"
        )
    validate_stage_b_gdino_score_adapter_checkpoint(
        expected_model, state, checkpoint_label=checkpoint_label
    )
    adapter = expected_model.stage_b_gdino_score_adapter
    adapter.load_state_dict(_adapter_state(state), strict=True)
    identity = functional_identity_check(adapter)
    if contract.get("functional_identity") != identity:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: functional identity receipt drifted"
        )


def verify_external_bindings(
    payload: Mapping[str, Any],
    *,
    config: Path,
    b58_path: Path,
    checkpoint_label: str,
) -> None:
    """Reopen external inputs and compare every claimed base tensor."""

    contract = payload["native_residual_initializer"]
    state = payload["model"]
    b58_record = stable_file_record(b58_path, label="b58 source checkpoint")
    if b58_record["sha256"] != EXPECTED_B58_SHA256 or contract.get(
        "b58_source"
    ) != b58_record:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: external b58 file binding drifted"
        )
    if contract.get("config") != _config_binding(config):
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: external config binding drifted"
        )
    source_payload = _safe_load_checkpoint(b58_path, label="b58 source checkpoint")
    source_state = _source_model(source_payload)
    role_keys = contract["role_keys"]["b58_base"]
    if set(role_keys) != set(source_state):
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: b58 role key set drifted"
        )
    mismatched = [
        key for key in role_keys if not torch.equal(state[key], source_state[key])
    ]
    if mismatched:
        raise NativeResidualInitializerError(
            f"{checkpoint_label}: b58 tensors changed: {mismatched[:8]}"
        )
    del source_payload


def build_payload(
    *, config: Path, b58_path: Path, seed: int
) -> tuple[dict[str, Any], nn.Module]:
    b58_record = stable_file_record(b58_path, label="b58 source checkpoint")
    if b58_record["sha256"] != EXPECTED_B58_SHA256:
        raise NativeResidualInitializerError(
            "b58 source SHA256 mismatch: "
            f"expected {EXPECTED_B58_SHA256}, got {b58_record['sha256']}"
        )
    model, cfg = _build_template(config, seed)
    source_payload = _safe_load_checkpoint(b58_path, label="b58 source checkpoint")
    source_state = _source_model(source_payload)
    state, roles = compose_native_residual_state(model.state_dict(), source_state)
    model.load_state_dict(state, strict=True)
    adapter = model.stage_b_gdino_score_adapter
    identity = functional_identity_check(adapter)
    rank_ids = {id(parameter) for parameter in adapter.rank_parameters()}
    confidence_ids = {id(parameter) for parameter in adapter.gate_parameters()}
    if not rank_ids or not confidence_ids or rank_ids & confidence_ids:
        raise NativeResidualInitializerError(
            "rank and confidence parameter sets are empty or overlap"
        )
    contract: dict[str, Any] = {
        "schema": SCHEMA,
        "seed": int(seed),
        "training_allowed": True,
        "b58_source": b58_record,
        "checkpoint_sources": [b58_record],
        "teacher_logits_or_scores_consumed": False,
        "config": _config_binding(config),
        "architecture": {
            "hidden_dim": int(cfg.hidden_dim),
            "num_queries": int(cfg.num_queries),
            "adapter_dim": int(cfg.stage_b_gdino_adapter_dim),
            "gate_hidden_dim": int(cfg.stage_b_gdino_gate_hidden_dim),
            "enable_patch_branch": bool(cfg.enable_patch_branch),
            "train_mode": str(cfg.stage_b_gdino_adapter_train_mode),
        },
        "role_keys": roles,
        "functional_identity": identity,
        "full_model_tensor_sha256": data_driven_tensor_state_sha256(
            state, sorted(state)
        ),
        "invariants": {
            "b58_is_only_checkpoint_source": True,
            "no_r100_p50_u0_stagea_or_teacher_logit_source": True,
            "rank_and_confidence_parameters_are_disjoint": True,
            "rank_output_exact_zero": True,
            "confidence_output_exact_zero": True,
            "native_rank_and_confidence_are_bitwise_identity": True,
        },
    }
    contract.update(_expected_role_hashes(state, roles))
    payload = {"model": state, "native_residual_initializer": contract}
    validate_initializer_payload(
        model, payload, checkpoint_label="in-memory native-residual initializer"
    )
    del source_payload
    gc.collect()
    return payload, model


def _default_output(seed: int) -> Path:
    return (
        REPO_ROOT
        / "outputs/paper_cvpr_v1/native_residual_initializers"
        / f"b58_only_seed{int(seed)}"
        / "checkpoint_nr_d0_init.pth"
    )


def _write_fresh(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"initializer output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale initializer temporary exists: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"), nargs="?", default="create")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--b58", type=Path, default=DEFAULT_B58)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = (args.output or _default_output(args.seed)).resolve()
    config = args.config.resolve(strict=True)
    b58 = args.b58.resolve(strict=True)
    if args.command == "create":
        payload, model = build_payload(
            config=config, b58_path=b58, seed=int(args.seed)
        )
        _write_fresh(output, payload)
    else:
        model, _cfg = _build_template(config, int(args.seed))
    reloaded = _safe_load_checkpoint(output, label="native-residual initializer")
    validate_initializer_payload(
        model,
        reloaded,
        checkpoint_label=f"published native-residual initializer {output}",
    )
    verify_external_bindings(
        reloaded,
        config=config,
        b58_path=b58,
        checkpoint_label=f"published native-residual initializer {output}",
    )
    record = stable_file_record(output, label="native-residual initializer")
    print(
        f"[OK] verified {record['path']} size={record['size_bytes']} "
        f"sha256={record['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
