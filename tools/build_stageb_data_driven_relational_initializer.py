#!/usr/bin/env python3
"""Build the deterministic b58-only relational-v1 DD1 initializer."""

from __future__ import annotations

import argparse
import gc
import os
import platform
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
    DATA_DRIVEN_RELATIONAL_INITIALIZER_SCHEMA,
    data_driven_tensor_state_sha256,
    validate_data_driven_relational_initializer_payload,
)
from tools.build_stageb_data_driven_initializer import (  # noqa: E402
    DEFAULT_B58,
    EXPECTED_B58_SHA256,
    _source_model,
)
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    _safe_load_checkpoint,
    stable_file_record,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_data_driven_a1_v2_initializer.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_initializers/fair_v2_seed42/"
    "checkpoint_dd_a1_relational_v2_init.pth"
)
EXPECTED_A0_PATCH_TENSOR_SHA256 = (
    "fa40f84b6c2105bededc5b3df3ff1161cabfbe76059ca79ae31fd3a82d0bfcb9"
)
EXPECTED_A0_CONFIDENCE_TENSOR_SHA256 = (
    "e09f95a26bd64982a83bb64c93f87873c4b1eb646fdfed9736c6aa4dac9e5305"
)
PATCH_KEYS = {
    "patch_logit_scale",
    "patch_encoder.input_proj.0.weight",
    "patch_encoder.input_proj.0.bias",
    "patch_encoder.input_proj.1.weight",
    "patch_encoder.input_proj.1.bias",
    "patch_encoder.norm.weight",
    "patch_encoder.norm.bias",
    "query_proj_for_patch.weight",
    "query_proj_for_patch.bias",
}
RANK_PREFIX = "stage_b_data_driven_score_heads.rank_branch."
CONFIDENCE_PREFIXES = (
    "stage_b_data_driven_score_heads.confidence_branch.",
    "stage_b_data_driven_score_heads.confidence_gate.",
)
CONTRACT_KEY = "stage_b_data_driven_score_heads._contract_version"
PATCH_BACKBONE_PREFIX = "patch_encoder.backbone."


class RelationalInitializerError(RuntimeError):
    pass


def _build_template(config: Path, seed: int):
    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    if not bool(getattr(cfg, "stage_b_data_driven_score", False)):
        raise RelationalInitializerError("template must enable data-driven scoring")
    if str(getattr(cfg, "stage_b_data_driven_train_mode", "")) != "rank_patch_only":
        raise RelationalInitializerError("template must use rank_patch_only")
    if str(getattr(cfg, "stage_b_data_driven_rank_architecture", "")) != "relational_v1":
        raise RelationalInitializerError("template must use relational_v1 rank")
    if str(getattr(cfg, "stage_b_data_driven_base_initializer_path", "") or "").strip():
        raise RelationalInitializerError(
            "initializer template must not bind a base initializer path"
        )
    if str(getattr(cfg, "stage_b_data_driven_base_initializer_sha256", "") or "").strip():
        raise RelationalInitializerError(
            "initializer template must not bind a base initializer SHA256"
        )
    torch.manual_seed(int(seed))
    model, _criterion, _postprocessors = build_model_main(cfg)
    return model.eval(), cfg


def compose_relational_state(
    template: Mapping[str, torch.Tensor],
    b58: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, list[str]]]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    roles = {
        "b58_base": [],
        "shared_backbone_alias": [],
        "random_patch_projection": [],
        "random_relational_rank": [],
        "random_absolute_confidence": [],
        "score_contract_buffer": [],
    }
    for key, template_value in template.items():
        if key in b58:
            value = b58[key]
            role = "b58_base"
        elif key.startswith(PATCH_BACKBONE_PREFIX):
            source_key = key.removeprefix("patch_encoder.")
            if source_key not in b58:
                raise RelationalInitializerError(
                    f"patch backbone alias has no b58 source: {key}"
                )
            value = b58[source_key]
            role = "shared_backbone_alias"
        elif key in PATCH_KEYS:
            value = template_value
            role = "random_patch_projection"
        elif key.startswith(RANK_PREFIX):
            value = template_value
            role = "random_relational_rank"
        elif key.startswith(CONFIDENCE_PREFIXES):
            value = template_value
            role = "random_absolute_confidence"
        elif key == CONTRACT_KEY:
            value = template_value
            role = "score_contract_buffer"
        else:
            raise RelationalInitializerError(
                f"relational initializer has an unbound template tensor: {key}"
            )
        if value.dtype != template_value.dtype or tuple(value.shape) != tuple(
            template_value.shape
        ):
            raise RelationalInitializerError(
                f"relational tensor shape/dtype mismatch at {key}"
            )
        result[key] = value.detach().cpu().clone()
        roles[role].append(key)
    if set(roles["b58_base"]) != set(b58):
        raise RelationalInitializerError(
            "b58 keys do not map exactly into the relational template"
        )
    if any(not keys for keys in roles.values()):
        raise RelationalInitializerError("relational initializer has an empty role")
    for alias in roles["shared_backbone_alias"]:
        source_key = alias.removeprefix("patch_encoder.")
        if not torch.equal(result[alias], result[source_key]):
            raise RelationalInitializerError(
                f"shared patch backbone differs from b58 at {alias}"
            )
    patch_hash = data_driven_tensor_state_sha256(
        result, roles["random_patch_projection"]
    )
    if patch_hash != EXPECTED_A0_PATCH_TENSOR_SHA256:
        raise RelationalInitializerError(
            "relational patch initialization differs from the absolute A0 control: "
            f"expected={EXPECTED_A0_PATCH_TENSOR_SHA256}, observed={patch_hash}"
        )
    confidence_hash = data_driven_tensor_state_sha256(
        result, roles["random_absolute_confidence"]
    )
    if confidence_hash != EXPECTED_A0_CONFIDENCE_TENSOR_SHA256:
        raise RelationalInitializerError(
            "relational confidence initialization differs from the absolute A0 "
            "control: "
            f"expected={EXPECTED_A0_CONFIDENCE_TENSOR_SHA256}, "
            f"observed={confidence_hash}"
        )
    return result, roles


def _config_binding(path: Path) -> dict[str, Any]:
    return {
        "leaf": stable_file_record(path, label="relational initializer config"),
        "import_chain": [
            stable_file_record(item, label=f"config dependency {item.name}")
            for item in config_import_chain(path.resolve(), root=REPO_ROOT)
        ],
    }


def build_payload(
    *, config: Path, b58_path: Path, seed: int
) -> tuple[dict[str, Any], torch.nn.Module]:
    b58_record = stable_file_record(b58_path, label="b58 source checkpoint")
    if b58_record["sha256"] != EXPECTED_B58_SHA256:
        raise RelationalInitializerError(
            "b58 source SHA256 mismatch: "
            f"expected={EXPECTED_B58_SHA256}, observed={b58_record['sha256']}"
        )
    model, cfg = _build_template(config, seed)
    source_payload = _safe_load_checkpoint(b58_path, label="b58 source checkpoint")
    source_state = _source_model(source_payload)
    state, roles = compose_relational_state(model.state_dict(), source_state)
    model.load_state_dict(state, strict=True)
    heads = model.stage_b_data_driven_score_heads
    rank_ids = {id(parameter) for parameter in heads.rank_parameters()}
    confidence_ids = {id(parameter) for parameter in heads.confidence_parameters()}
    if not rank_ids or not confidence_ids or rank_ids & confidence_ids:
        raise RelationalInitializerError("rank/confidence parameter partition drifted")
    rank = heads.rank_branch
    architecture = {
        "hidden_dim": int(cfg.hidden_dim),
        "num_queries": int(cfg.num_queries),
        "rank_architecture": "relational_v1",
        "rank_dim": int(cfg.stage_b_data_driven_rank_dim),
        "rank_num_heads": int(cfg.stage_b_data_driven_rank_num_heads),
        "rank_image_level_policy": str(
            cfg.stage_b_data_driven_rank_image_level_policy
        ),
        "rank_image_levels": int(cfg.stage_b_data_driven_rank_image_levels),
        "rank_image_pool_size": int(
            cfg.stage_b_data_driven_rank_image_pool_size
        ),
        "rank_image_pool_policy": str(
            cfg.stage_b_data_driven_rank_image_pool_policy
        ),
        "rank_box_fourier_bands": int(
            cfg.stage_b_data_driven_rank_box_fourier_bands
        ),
        "rank_ffn_dim": int(cfg.stage_b_data_driven_rank_ffn_dim),
        "rank_dropout": float(cfg.stage_b_data_driven_rank_dropout),
        "head_init_seed": int(cfg.stage_b_data_driven_head_init_seed),
        "confidence_dim": int(cfg.stage_b_data_driven_confidence_dim),
        "enable_patch_branch": bool(cfg.enable_patch_branch),
    }
    if (
        architecture["rank_image_level_policy"] != rank.image_level_policy
        or architecture["rank_image_pool_policy"] != rank.image_pool_policy
        or any(
            architecture[key] != value
            for key, value in {
                "rank_num_heads": rank.num_heads,
                "rank_image_levels": rank.image_levels,
                "rank_image_pool_size": rank.image_pool_size,
                "rank_box_fourier_bands": rank.box_fourier_bands,
                "rank_ffn_dim": rank.ffn_dim,
                "rank_dropout": rank.dropout,
            }.items()
        )
    ):
        raise RelationalInitializerError("relational model/config contract drifted")
    contract: dict[str, Any] = {
        "schema": DATA_DRIVEN_RELATIONAL_INITIALIZER_SCHEMA,
        "seed": int(seed),
        "b58_source": b58_record,
        "b58_model_tensor_sha256": data_driven_tensor_state_sha256(
            source_state, sorted(source_state)
        ),
        "config": _config_binding(config),
        "implementation": {
            "builder": stable_file_record(
                Path(__file__).resolve(), label="relational initializer builder"
            ),
            "score_heads": stable_file_record(
                REPO_ROOT / "models/GroundingDINO/stage_b_data_driven_score.py",
                label="relational score heads",
            ),
            "model_integration": stable_file_record(
                REPO_ROOT / "models/GroundingDINO/groundingdino.py",
                label="relational model integration",
            ),
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "architecture": architecture,
        "role_keys": roles,
        "role_key_counts": {role: len(keys) for role, keys in roles.items()},
        "full_model_tensor_sha256": data_driven_tensor_state_sha256(
            state, sorted(state)
        ),
        "invariants": {
            "b58_is_only_checkpoint_source": True,
            "no_u1000_u5020_tensor_source": True,
            "no_teacher_adapter_tensor_source": True,
            "canonical_query_and_full_text_paths_are_separate": True,
            "rank_and_confidence_parameters_are_disjoint": True,
            "patch_backbone_aliases_b58": True,
            "patch_initialization_matches_absolute_a0": True,
            "confidence_initialization_matches_absolute_a0": True,
            "image_pooling_is_padding_invariant": True,
        },
    }
    for role, keys in roles.items():
        contract[f"{role}_tensor_sha256"] = data_driven_tensor_state_sha256(
            state, keys
        )
    payload = {
        "model": state,
        "data_driven_relational_initializer": contract,
    }
    validate_data_driven_relational_initializer_payload(
        model, payload, checkpoint_label="in-memory relational initializer"
    )
    del source_payload
    gc.collect()
    return payload, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--b58", type=Path, default=DEFAULT_B58)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
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
    reloaded = _safe_load_checkpoint(output, label="published relational initializer")
    validate_data_driven_relational_initializer_payload(
        model,
        reloaded,
        checkpoint_label=f"published relational initializer {output}",
    )
    record = stable_file_record(output, label="published relational initializer")
    print(
        f"[OK] wrote {record['path']} size={record['size_bytes']} "
        f"sha256={record['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
