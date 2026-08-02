from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from tools.build_stageb_native_patch_category_initializer import (
    EXPECTED_B58_SHA256,
    RANDOM_FROZEN_PATCH_KEYS,
    RANDOM_TRAINABLE_PATCH_KEYS,
    NativePatchCategoryInitializerError,
    build_native_patch_category_contract,
    compose_native_patch_category_state,
    extract_b58_source_state,
    validate_native_patch_category_initializer_payload,
)


def _template_and_source():
    source = OrderedDict(
        {
            "backbone.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "decoder.bias": torch.arange(2, dtype=torch.float32),
        }
    )
    template = OrderedDict(
        (key, torch.zeros_like(value)) for key, value in source.items()
    )
    template["patch_encoder.backbone.weight"] = torch.full((2, 3), -1.0)
    for index, key in enumerate(sorted(RANDOM_TRAINABLE_PATCH_KEYS), start=1):
        template[key] = torch.tensor([float(index)], dtype=torch.float32)
    template[next(iter(RANDOM_FROZEN_PATCH_KEYS))] = torch.tensor(
        2.0, dtype=torch.float32
    )
    return template, source


def _record(*, sha256: str = EXPECTED_B58_SHA256):
    return {
        "path": "/sealed/checkpoint0001.pth",
        "size_bytes": 123,
        "sha256": sha256,
    }


def _payload():
    template, source = _template_and_source()
    state, roles = compose_native_patch_category_state(template, source)
    contract = build_native_patch_category_contract(
        state=state,
        roles=roles,
        b58_record=_record(),
        config_binding={"leaf": {}, "import_chain": []},
        seed=42,
        architecture={
            "enable_patch_branch": True,
            "patch_gate_with_text": False,
        },
    )
    return template, source, {
        "model": state,
        "native_patch_category_initializer": contract,
    }


def _clone_payload(payload):
    return {
        "model": OrderedDict(
            (key, value.clone()) for key, value in payload["model"].items()
        ),
        "native_patch_category_initializer": {
            **payload["native_patch_category_initializer"],
            "sources": {
                "b58": dict(
                    payload["native_patch_category_initializer"]["sources"]["b58"]
                )
            },
            "role_keys": {
                role: list(keys)
                for role, keys in payload[
                    "native_patch_category_initializer"
                ]["role_keys"].items()
            },
            "role_key_counts": dict(
                payload["native_patch_category_initializer"]["role_key_counts"]
            ),
            "random_initialization": dict(
                payload["native_patch_category_initializer"][
                    "random_initialization"
                ]
            ),
            "invariants": dict(
                payload["native_patch_category_initializer"]["invariants"]
            ),
        },
    }


def test_compose_is_exhaustive_and_copies_only_approved_sources():
    template, source = _template_and_source()
    state, roles = compose_native_patch_category_state(template, source)

    assert set(state) == set(template)
    assert set(roles["b58_base"]) == set(source)
    assert set(roles["random_trainable_patch_projection"]) == set(
        RANDOM_TRAINABLE_PATCH_KEYS
    )
    assert set(roles["random_frozen_patch_scale"]) == set(
        RANDOM_FROZEN_PATCH_KEYS
    )
    assert roles["shared_backbone_alias"] == ["patch_encoder.backbone.weight"]
    for key, value in source.items():
        assert torch.equal(state[key], value)
        assert state[key].data_ptr() != value.data_ptr()
    assert torch.equal(
        state["patch_encoder.backbone.weight"], source["backbone.weight"]
    )
    for key in RANDOM_TRAINABLE_PATCH_KEYS | RANDOM_FROZEN_PATCH_KEYS:
        assert torch.equal(state[key], template[key])


def test_compose_rejects_unbound_template_and_unconsumed_b58_keys():
    template, source = _template_and_source()
    template["unexpected.weight"] = torch.ones(1)
    with pytest.raises(NativePatchCategoryInitializerError, match="unbound tensor"):
        compose_native_patch_category_state(template, source)

    template, source = _template_and_source()
    source["stale.weight"] = torch.ones(1)
    with pytest.raises(
        NativePatchCategoryInitializerError, match="do not map exactly"
    ):
        compose_native_patch_category_state(template, source)


def test_source_extractor_rejects_u2_r100_p50_stagea_or_teacher_derivatives():
    forbidden_keys = (
        "stage_b_gdino_score_adapter.rank_output.weight",
        "stage_b_u0_patch_rank_adapter.output.weight",
        "stage_b_data_driven_score_heads.rank.weight",
        "patch_encoder.input_proj.0.weight",
        "query_proj_for_patch.weight",
        "patch_logit_scale",
    )
    for key in forbidden_keys:
        with pytest.raises(
            NativePatchCategoryInitializerError, match="forbidden derived/patch"
        ):
            extract_b58_source_state(
                {"model": {"backbone.weight": torch.ones(1), key: torch.ones(1)}},
                checkpoint_label="forbidden source",
            )


def test_validator_accepts_exact_payload_and_external_b58_anchor():
    template, source, payload = _payload()
    validate_native_patch_category_initializer_payload(
        template,
        payload,
        checkpoint_label="valid",
        expected_b58_state=source,
    )


def test_validator_rejects_tensor_and_alias_tampering():
    template, source, payload = _payload()
    drift = _clone_payload(payload)
    drift["model"]["decoder.bias"].add_(1)
    with pytest.raises(ValueError, match="tensor hash drifted|full-model"):
        validate_native_patch_category_initializer_payload(
            template, drift, checkpoint_label="tensor drift"
        )

    drift = _clone_payload(payload)
    drift["model"]["patch_encoder.backbone.weight"].add_(1)
    with pytest.raises(ValueError, match="shared-backbone alias drifted"):
        validate_native_patch_category_initializer_payload(
            template, drift, checkpoint_label="alias drift"
        )


def test_external_b58_anchor_rejects_rehashed_base_tampering():
    template, source, payload = _payload()
    drift = _clone_payload(payload)
    drift["model"]["decoder.bias"].add_(1)
    roles = drift["native_patch_category_initializer"]["role_keys"]
    drift["native_patch_category_initializer"] = (
        build_native_patch_category_contract(
            state=drift["model"],
            roles=roles,
            b58_record=_record(),
            config_binding={"leaf": {}, "import_chain": []},
            seed=42,
            architecture={
                "enable_patch_branch": True,
                "patch_gate_with_text": False,
            },
        )
    )
    with pytest.raises(ValueError, match="b58-anchored tensor drifted"):
        validate_native_patch_category_initializer_payload(
            template,
            drift,
            checkpoint_label="rehashed drift",
            expected_b58_state=source,
        )


def test_validator_rejects_non_b58_source_provenance():
    template, _source, payload = _payload()
    drift = _clone_payload(payload)
    drift["native_patch_category_initializer"]["sources"]["b58"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="b58 source SHA256 mismatch"):
        validate_native_patch_category_initializer_payload(
            template, drift, checkpoint_label="teacher source"
        )
