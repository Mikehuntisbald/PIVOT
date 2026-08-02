#!/usr/bin/env python3
"""Build and verify the single-model U0 initializer from sealed teacher weights."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    U0_INITIALIZER_SCHEMA,
    U0_PATCH_BACKBONE_PREFIX,
    U0_PATCH_SOURCE_KEYS,
    U0_SEALED_TEACHER_ARCHITECTURE_FIELDS,
    U0_TEACHER_FUNCTIONAL_FIELDS,
    stage_b_u0_tensor_state_sha256,
    validate_stage_b_u0_initializer_payload,
    validate_stage_b_u0_patch_rank_checkpoint,
)
from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    StageBGDINOScoreAdapter,
    validate_stage_b_gdino_score_adapter_checkpoint,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from tools.stageb_gdino_adapter_probe_audit import (  # noqa: E402
    ProbeAuditError,
    file_record,
    load_checkpoint,
)
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = U0_INITIALIZER_SCHEMA
DEFAULT_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_u0_r100p50_patch_rank.py"
PATCH_BACKBONE_PREFIX = U0_PATCH_BACKBONE_PREFIX
U0_PREFIX = "stage_b_u0_patch_rank_adapter."
PATCH_SOURCE_KEYS = U0_PATCH_SOURCE_KEYS
tensor_state_sha256 = stage_b_u0_tensor_state_sha256


class U0InitializerError(RuntimeError):
    pass


def _require_hash(path: Path, expected: str, *, label: str) -> dict[str, Any]:
    record = file_record(path.resolve(strict=True))
    if record["sha256"] != str(expected).strip().lower():
        raise U0InitializerError(
            f"{label} SHA256 mismatch: expected {expected}, got {record['sha256']}"
        )
    return record


def _model_state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise U0InitializerError(f"{label} has no model mapping")
    invalid = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise U0InitializerError(f"{label} contains non-tensor model values: {invalid[:8]}")
    return state


def _exact_tensor(value: torch.Tensor, wanted: torch.Tensor, *, key: str) -> None:
    if tuple(value.shape) != tuple(wanted.shape) or value.dtype != wanted.dtype:
        raise U0InitializerError(
            f"tensor shape/dtype mismatch for {key}: "
            f"{tuple(value.shape)}/{value.dtype} != {tuple(wanted.shape)}/{wanted.dtype}"
        )


def compose_u0_model_state(
    template_state: Mapping[str, torch.Tensor],
    merged_state: Mapping[str, torch.Tensor],
    stagea_state: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, list[str]]]:
    observed_patch_keys = {key for key in stagea_state if key in PATCH_SOURCE_KEYS}
    if observed_patch_keys != set(PATCH_SOURCE_KEYS):
        raise U0InitializerError(
            "Stage-A patch-specific key contract mismatch: "
            f"missing={sorted(set(PATCH_SOURCE_KEYS) - observed_patch_keys)}, "
            f"unexpected={sorted(observed_patch_keys - set(PATCH_SOURCE_KEYS))}"
        )
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    roles = {"merged": [], "shared_backbone_alias": [], "stagea_patch": [], "u0_zero": []}
    for key, template_value in template_state.items():
        if key in merged_state:
            value = merged_state[key]
            role = "merged"
        elif key.startswith(PATCH_BACKBONE_PREFIX):
            base_key = key.removeprefix("patch_encoder.")
            if base_key not in merged_state:
                raise U0InitializerError(
                    f"patch-backbone alias has no sealed b58 source: {key} -> {base_key}"
                )
            value = merged_state[base_key]
            role = "shared_backbone_alias"
        elif key in PATCH_SOURCE_KEYS:
            value = stagea_state[key]
            role = "stagea_patch"
        elif key.startswith(U0_PREFIX):
            value = template_value
            role = "u0_zero"
        else:
            raise U0InitializerError(f"U0 template contains an unbound tensor: {key}")
        _exact_tensor(value, template_value, key=key)
        result[key] = value.detach().cpu().clone()
        roles[role].append(key)
    if set(merged_state) != set(roles["merged"]):
        raise U0InitializerError(
            "sealed merged teacher keys do not map exactly into U0: "
            f"missing={sorted(set(merged_state) - set(roles['merged']))[:8]}"
        )
    if not roles["shared_backbone_alias"] or set(roles["stagea_patch"]) != set(
        PATCH_SOURCE_KEYS
    ):
        raise U0InitializerError("U0 patch/shared-backbone roles are incomplete")
    if not roles["u0_zero"]:
        raise U0InitializerError("U0 template has no patch-rank adapter state")
    return result, roles


def _config_binding(config: Path) -> dict[str, Any]:
    chain = config_import_chain(config.resolve(), root=REPO_ROOT)
    return {
        "leaf": file_record(config.resolve()),
        "import_chain": [file_record(path) for path in chain],
    }


def _build_template(config: Path):
    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    if not bool(getattr(cfg, "stage_b_u0_patch_rank", False)):
        raise U0InitializerError("config does not enable stage_b_u0_patch_rank")
    if not bool(getattr(cfg, "enable_patch_branch", False)):
        raise U0InitializerError("config does not enable the patch branch")
    torch.manual_seed(0)
    model, _criterion, _postprocessors = build_model_main(cfg)
    return model.eval(), cfg


def _architecture_contract(
    cfg: Any,
    merged_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = set(U0_SEALED_TEACHER_ARCHITECTURE_FIELDS)
    teacher = merged_contract.get("architecture")
    if not isinstance(teacher, Mapping) or set(teacher) != fields:
        raise U0InitializerError("merged teacher architecture contract drifted")
    teacher = {key: teacher[key] for key in U0_SEALED_TEACHER_ARCHITECTURE_FIELDS}
    u0 = {
        key: getattr(cfg, key, None)
        for key in U0_SEALED_TEACHER_ARCHITECTURE_FIELDS
    }
    for key in fields - {"enable_patch_branch"}:
        if u0[key] != teacher[key]:
            raise U0InitializerError(
                f"U0 config differs from sealed teacher architecture at {key}: "
                f"teacher={teacher[key]!r}, u0={u0[key]!r}"
            )
    if teacher["enable_patch_branch"] is not False or u0["enable_patch_branch"] is not True:
        raise U0InitializerError(
            "U0 requires the sole teacher architecture transition "
            "enable_patch_branch=False -> True"
        )
    return teacher, u0


def _teacher_functional_equivalence(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    teacher_architecture: Mapping[str, Any],
) -> dict[str, bool]:
    source = StageBGDINOScoreAdapter(
        hidden_dim=int(teacher_architecture["hidden_dim"]),
        adapter_dim=int(teacher_architecture["stage_b_gdino_adapter_dim"]),
        gate_hidden_dim=int(
            teacher_architecture["stage_b_gdino_gate_hidden_dim"]
        ),
        gate_pool_temperature=float(
            teacher_architecture["stage_b_gdino_gate_pool_temperature"]
        ),
        gate_topk=int(teacher_architecture["stage_b_gdino_gate_topk"]),
    ).eval()
    prefix = "stage_b_gdino_score_adapter."
    source_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    source.load_state_dict(source_state, strict=True)
    generator = torch.Generator(device="cpu").manual_seed(20_260_720)
    query = torch.randn(2, 900, 256, generator=generator)
    base = torch.rand(2, 900, generator=generator)
    with torch.inference_mode():
        expected = source(query, base)
        observed = model.stage_b_gdino_score_adapter(query, base)
    result = {
        key: bool(torch.equal(observed[key], expected[key]))
        for key in U0_TEACHER_FUNCTIONAL_FIELDS
    }
    if not all(result.values()):
        raise U0InitializerError(
            f"U0 sealed teacher functional equivalence failed: {result}"
        )
    return result


def _contract(
    *,
    state: Mapping[str, torch.Tensor],
    roles: Mapping[str, Sequence[str]],
    merged_record: Mapping[str, Any],
    stagea_record: Mapping[str, Any],
    config_binding: Mapping[str, Any],
    teacher_architecture: Mapping[str, Any],
    u0_architecture: Mapping[str, Any],
    teacher_functional_bitwise: Mapping[str, bool],
) -> dict[str, Any]:
    merged_hash = tensor_state_sha256(state, roles["merged"])
    patch_hash = tensor_state_sha256(state, roles["stagea_patch"])
    u0_hash = tensor_state_sha256(state, roles["u0_zero"])
    alias_hash = tensor_state_sha256(state, roles["shared_backbone_alias"])
    return {
        "schema": SCHEMA,
        "single_model_root": True,
        "single_shared_backbone": True,
        "resumable": False,
        "training_initializer": True,
        "sources": {
            "merged_r100_p50": dict(merged_record),
            "stagea_patch": dict(stagea_record),
        },
        "config": dict(config_binding),
        "sealed_teacher_architecture": dict(teacher_architecture),
        "u0_architecture": dict(u0_architecture),
        "teacher_functional_bitwise": dict(teacher_functional_bitwise),
        "model_state_keys": len(state),
        "role_key_counts": {key: len(value) for key, value in roles.items()},
        "role_keys": {key: list(value) for key, value in roles.items()},
        "merged_teacher_tensor_sha256": merged_hash,
        "stagea_patch_tensor_sha256": patch_hash,
        "shared_backbone_alias_tensor_sha256": alias_hash,
        "u0_zero_tensor_sha256": u0_hash,
        "full_model_tensor_sha256": tensor_state_sha256(state, state.keys()),
        "invariants": {
            "merged_teacher_copied_bitwise": True,
            "stagea_patch_specific_keys_only": True,
            "stagea_patch_backbone_imported": False,
            "patch_backbone_aliases_source_b58": True,
            "u0_output_exactly_zero": True,
            "u0_rank_equals_r100_at_initialization": True,
            "p50_confidence_unchanged": True,
        },
    }


def build_initializer_payload(
    *,
    merged_checkpoint: Path,
    merged_sha256: str,
    stagea_checkpoint: Path,
    stagea_sha256: str,
    config: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    merged_record = _require_hash(merged_checkpoint, merged_sha256, label="merged teacher")
    stagea_record = _require_hash(stagea_checkpoint, stagea_sha256, label="Stage-A patch source")
    merged_payload = load_checkpoint(merged_checkpoint)
    stagea_payload = load_checkpoint(stagea_checkpoint)
    merged_state = _model_state(merged_payload, label="merged teacher")
    stagea_state = _model_state(stagea_payload, label="Stage-A patch source")
    embedded = merged_payload.get("contract")
    if not isinstance(embedded, Mapping) or embedded.get("eval_only") is not True:
        raise U0InitializerError("merged teacher is missing its eval-only contract")
    if embedded.get("full_model_tensor_sha256") != tensor_state_sha256(
        merged_state, merged_state.keys()
    ):
        raise U0InitializerError("merged teacher embedded tensor hash drifted")
    teacher_architecture, u0_architecture = _architecture_contract(
        SLConfig.fromfile(str(config)), embedded
    )
    model, _cfg = _build_template(config)
    state, roles = compose_u0_model_state(
        model.state_dict(), merged_state, stagea_state
    )
    load_result = model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise U0InitializerError("strict U0 initializer load unexpectedly drifted")
    validate_stage_b_gdino_score_adapter_checkpoint(
        model, state, checkpoint_label="U0 initializer"
    )
    validate_stage_b_u0_patch_rank_checkpoint(
        model, state, checkpoint_label="U0 initializer"
    )
    if int(torch.count_nonzero(model.stage_b_u0_patch_rank_adapter.output.weight)):
        raise U0InitializerError("U0 output weight is not exactly zero")
    if int(torch.count_nonzero(model.stage_b_u0_patch_rank_adapter.output.bias)):
        raise U0InitializerError("U0 output bias is not exactly zero")
    generator = torch.Generator(device="cpu").manual_seed(20_260_720)
    query = torch.randn(2, 900, 256, generator=generator)
    base = torch.rand(2, 900, generator=generator)
    patch = torch.randn(2, 900, generator=generator)
    with torch.inference_mode():
        teacher = model.stage_b_gdino_score_adapter(query, base)
        u0 = model.stage_b_u0_patch_rank_adapter(patch, teacher["rank_score"])
    if not torch.equal(u0["rank_score"], teacher["rank_score"]):
        raise U0InitializerError("zero U0 residual is not bitwise R100 identity")
    teacher_functional_bitwise = _teacher_functional_equivalence(
        model, state, teacher_architecture
    )
    contract = _contract(
        state=state,
        roles=roles,
        merged_record=merged_record,
        stagea_record=stagea_record,
        config_binding=_config_binding(config),
        teacher_architecture=teacher_architecture,
        u0_architecture=u0_architecture,
        teacher_functional_bitwise=teacher_functional_bitwise,
    )
    del model, merged_payload, stagea_payload
    gc.collect()
    return {"model": state, "u0_initializer": contract}


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise U0InitializerError(f"refusing to overwrite U0 initializer: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        raise U0InitializerError(f"stale U0 initializer temporary exists: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_initializer(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    payload = load_checkpoint(checkpoint)
    if set(payload) != {"model", "u0_initializer"}:
        raise U0InitializerError("U0 initializer top-level keys drifted")
    contract = payload.get("u0_initializer")
    state = _model_state(payload, label="U0 initializer")
    validate_stage_b_u0_initializer_payload(
        state, payload, checkpoint_label="serialized U0 initializer"
    )
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise U0InitializerError("U0 initializer contract is missing")
    roles = contract.get("role_keys")
    if not isinstance(roles, Mapping):
        raise U0InitializerError("U0 initializer role keys are missing")
    if contract.get("full_model_tensor_sha256") != tensor_state_sha256(
        state, state.keys()
    ):
        raise U0InitializerError("U0 initializer full tensor hash drifted")
    for field, role in (
        ("merged_teacher_tensor_sha256", "merged"),
        ("stagea_patch_tensor_sha256", "stagea_patch"),
        ("shared_backbone_alias_tensor_sha256", "shared_backbone_alias"),
        ("u0_zero_tensor_sha256", "u0_zero"),
    ):
        if contract.get(field) != tensor_state_sha256(state, roles[role]):
            raise U0InitializerError(f"U0 initializer {field} drifted")
    for alias in roles["shared_backbone_alias"]:
        base_key = str(alias).removeprefix("patch_encoder.")
        if not torch.equal(state[alias], state[base_key]):
            raise U0InitializerError(f"shared backbone alias differs: {alias}")
    if int(torch.count_nonzero(state[U0_PREFIX + "output.weight"])) or int(
        torch.count_nonzero(state[U0_PREFIX + "output.bias"])
    ):
        raise U0InitializerError("serialized U0 residual output is not zero")
    return {
        "schema": SCHEMA,
        "status": "verified",
        "checkpoint": file_record(checkpoint),
        "contract": dict(contract),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--merged-checkpoint", required=True)
    create.add_argument("--merged-sha256", required=True)
    create.add_argument("--stagea-checkpoint", required=True)
    create.add_argument("--stagea-sha256", required=True)
    create.add_argument("--config", default=str(DEFAULT_CONFIG))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "create":
            output = Path(args.output)
            payload = build_initializer_payload(
                merged_checkpoint=Path(args.merged_checkpoint).resolve(),
                merged_sha256=args.merged_sha256,
                stagea_checkpoint=Path(args.stagea_checkpoint).resolve(),
                stagea_sha256=args.stagea_sha256,
                config=Path(args.config).resolve(),
            )
            _write_checkpoint(output, payload)
            result = verify_initializer(output)
        else:
            result = verify_initializer(Path(args.checkpoint))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    except (OSError, ProbeAuditError, U0InitializerError, ValueError) as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
