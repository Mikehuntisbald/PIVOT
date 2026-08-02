#!/usr/bin/env python3
"""Build the deterministic b58-only initializer shared by DD0-DD3."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    DATA_DRIVEN_INITIALIZER_SCHEMA,
    data_driven_tensor_state_sha256,
    validate_data_driven_initializer_payload,
)
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    _safe_load_checkpoint,
    stable_file_record,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_data_driven_a0_v2_initializer.py"
)
DEFAULT_B58 = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)
EXPECTED_B58_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
HEAD_PREFIX = "stage_b_data_driven_score_heads."
PATCH_BACKBONE_PREFIX = "patch_encoder.backbone."
FORBIDDEN_SOURCE_PREFIXES = (
    "stage_b_gdino_score_adapter.",
    "stage_b_u0_patch_rank_adapter.",
    "stage_b_data_driven_score_heads.",
    "patch_encoder.",
    "query_proj_for_patch.",
)


class DataDrivenInitializerError(RuntimeError):
    pass


def _build_template(config: Path, seed: int):
    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    if not bool(getattr(cfg, "stage_b_data_driven_score", False)):
        raise DataDrivenInitializerError(
            "initializer config must enable stage_b_data_driven_score"
        )
    if str(getattr(cfg, "stage_b_data_driven_train_mode", "")) != "rank_patch_only":
        raise DataDrivenInitializerError(
            "initializer config must use rank_patch_only mode"
        )
    if str(getattr(cfg, "stage_b_data_driven_rank_architecture", "")) != "absolute_token":
        raise DataDrivenInitializerError(
            "initializer config must use absolute_token rank"
        )
    if str(getattr(cfg, "stage_b_data_driven_base_initializer_path", "") or "").strip():
        raise DataDrivenInitializerError(
            "initializer config must not bind a base initializer path"
        )
    if str(getattr(cfg, "stage_b_data_driven_base_initializer_sha256", "") or "").strip():
        raise DataDrivenInitializerError(
            "initializer config must not bind a base initializer SHA256"
        )
    torch.manual_seed(int(seed))
    model, _criterion, _postprocessors = build_model_main(cfg)
    return model.eval(), cfg


def _source_model(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise DataDrivenInitializerError("b58 checkpoint has no model mapping")
    cleaned = utils.clean_state_dict(state)
    invalid = [
        key
        for key, value in cleaned.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise DataDrivenInitializerError(
            f"b58 model contains non-tensor values: {invalid[:8]}"
        )
    forbidden = sorted(
        key
        for key in cleaned
        if any(key.startswith(prefix) for prefix in FORBIDDEN_SOURCE_PREFIXES)
    )
    if forbidden:
        raise DataDrivenInitializerError(
            f"b58 source already contains forbidden scoring/patch tensors: {forbidden[:8]}"
        )
    return cleaned


def compose_data_driven_state(
    template: Mapping[str, torch.Tensor],
    b58: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, list[str]]]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    roles = {
        "b58_base": [],
        "shared_backbone_alias": [],
        "random_patch_projection": [],
        "random_absolute_heads": [],
    }
    for key, template_value in template.items():
        if key in b58:
            value = b58[key]
            role = "b58_base"
        elif key.startswith(PATCH_BACKBONE_PREFIX):
            source_key = key.removeprefix("patch_encoder.")
            if source_key not in b58:
                raise DataDrivenInitializerError(
                    f"patch backbone alias has no b58 source: {key} -> {source_key}"
                )
            value = b58[source_key]
            role = "shared_backbone_alias"
        elif key.startswith(HEAD_PREFIX):
            value = template_value
            role = "random_absolute_heads"
        else:
            value = template_value
            role = "random_patch_projection"
        if value.dtype != template_value.dtype or tuple(value.shape) != tuple(
            template_value.shape
        ):
            raise DataDrivenInitializerError(
                f"initializer tensor shape/dtype mismatch at {key}"
            )
        result[key] = value.detach().cpu().clone()
        roles[role].append(key)
    if set(roles["b58_base"]) != set(b58):
        raise DataDrivenInitializerError(
            "b58 keys do not map exactly into the data-driven template: "
            f"missing={sorted(set(b58) - set(roles['b58_base']))[:8]}"
        )
    if any(not keys for keys in roles.values()):
        raise DataDrivenInitializerError("initializer contains an empty tensor role")
    for alias in roles["shared_backbone_alias"]:
        source_key = alias.removeprefix("patch_encoder.")
        if not torch.equal(result[alias], result[source_key]):
            raise DataDrivenInitializerError(
                f"shared patch backbone differs from b58 at {alias}"
            )
    return result, roles


def _config_binding(path: Path) -> dict[str, Any]:
    chain = config_import_chain(path.resolve(), root=REPO_ROOT)
    return {
        "leaf": stable_file_record(path, label="data-driven initializer config"),
        "import_chain": [
            stable_file_record(item, label=f"config dependency {item.name}")
            for item in chain
        ],
    }


def build_payload(
    *,
    config: Path,
    b58_path: Path,
    seed: int,
) -> tuple[dict[str, Any], torch.nn.Module]:
    b58_record = stable_file_record(b58_path, label="b58 source checkpoint")
    if b58_record["sha256"] != EXPECTED_B58_SHA256:
        raise DataDrivenInitializerError(
            "b58 source SHA256 mismatch: "
            f"expected {EXPECTED_B58_SHA256}, got {b58_record['sha256']}"
        )
    model, cfg = _build_template(config, seed)
    source_payload = _safe_load_checkpoint(b58_path, label="b58 source checkpoint")
    source_state = _source_model(source_payload)
    state, roles = compose_data_driven_state(model.state_dict(), source_state)
    model.load_state_dict(state, strict=True)
    heads = model.stage_b_data_driven_score_heads
    rank_ids = {id(parameter) for parameter in heads.rank_parameters()}
    confidence_ids = {id(parameter) for parameter in heads.confidence_parameters()}
    if not rank_ids or not confidence_ids or rank_ids & confidence_ids:
        raise DataDrivenInitializerError(
            "rank and confidence parameter sets are empty or overlap"
        )
    contract = {
        "schema": DATA_DRIVEN_INITIALIZER_SCHEMA,
        "seed": int(seed),
        "b58_source": b58_record,
        "config": _config_binding(config),
        "architecture": {
            "hidden_dim": int(cfg.hidden_dim),
            "num_queries": int(cfg.num_queries),
            "rank_dim": int(cfg.stage_b_data_driven_rank_dim),
            "confidence_dim": int(cfg.stage_b_data_driven_confidence_dim),
            "enable_patch_branch": bool(cfg.enable_patch_branch),
        },
        "role_keys": roles,
        "full_model_tensor_sha256": data_driven_tensor_state_sha256(
            state, sorted(state)
        ),
        "invariants": {
            "b58_is_only_checkpoint_source": True,
            "no_r100_p50_u0_or_stagea_tensor_source": True,
            "canonical_query_and_full_text_heads_are_separate": True,
            "rank_and_confidence_parameters_are_disjoint": True,
            "patch_backbone_aliases_b58": True,
        },
    }
    for role, keys in roles.items():
        contract[f"{role}_tensor_sha256"] = data_driven_tensor_state_sha256(
            state, keys
        )
    payload = {
        "model": state,
        "data_driven_initializer": contract,
    }
    validate_data_driven_initializer_payload(
        model, payload, checkpoint_label="in-memory data-driven initializer"
    )
    del source_payload
    gc.collect()
    return payload, model


def _default_output(seed: int) -> Path:
    return (
        REPO_ROOT
        / "outputs/paper_cvpr_v1/data_driven_initializers"
        / f"fair_v2_seed{int(seed)}"
        / "checkpoint_dd_a0_absolute_v2_init.pth"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--b58", type=Path, default=DEFAULT_B58)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or _default_output(args.seed)
    if output.exists():
        raise FileExistsError(f"initializer output must be fresh: {output}")
    payload, model = build_payload(
        config=args.config.resolve(strict=True),
        b58_path=args.b58.resolve(strict=True),
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    reloaded = _safe_load_checkpoint(output, label="published data-driven initializer")
    validate_data_driven_initializer_payload(
        model,
        reloaded,
        checkpoint_label=f"published data-driven initializer {output}",
    )
    record = stable_file_record(output, label="published data-driven initializer")
    print(
        f"[OK] wrote {record['path']} size={record['size_bytes']} "
        f"sha256={record['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
