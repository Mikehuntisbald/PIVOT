from collections import OrderedDict

import pytest
import torch
from torch import nn

from models.GroundingDINO.stage_b_dense_duty_scorer import (
    AbsoluteConfidencePool,
    CONFIDENCE_PHRASE_AGGREGATION_LEGACY,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
    CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_AFFINE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_OFF,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE,
    TokenAwareConfidenceAdapter,
)
from util.stage_b_confidence_adapter_migration import (
    ABSOLUTE_CAP_FRESH_CONFIDENCE_CONTRACT,
    ABSOLUTE_CAP_MIGRATION_SCHEMA,
    CARRIER_SLOPE_FRESH_CONFIDENCE_CONTRACT,
    CARRIER_SLOPE_MIGRATION_SCHEMA,
    CARRIER_SLOPE_RESIDUAL_CONTRACT,
    CARRIER_AFFINE_FRESH_CONFIDENCE_CONTRACT,
    CARRIER_AFFINE_MIGRATION_SCHEMA,
    CARRIER_AFFINE_RESIDUAL_CONTRACT,
    CANDIDATE_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
    CANDIDATE_ABSOLUTE_MIGRATION_SCHEMA,
    CANDIDATE_CALIBRATED_FRESH_CONFIDENCE_CONTRACT,
    CANDIDATE_CALIBRATED_MIGRATION_SCHEMA,
    CANDIDATE_SAMPLE_CALIBRATOR_FRESH_CONFIDENCE_CONTRACT,
    CANDIDATE_SAMPLE_CALIBRATOR_HEAD_GRADIENT_CONTRACT,
    CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA,
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_ADAPTER_TENSOR_COUNT,
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_ELEMENT_COUNT,
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_SHA256,
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_STORAGE_BYTES,
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_TENSOR_COUNT,
    EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_STRICT_TARGET_TENSOR_COUNT,
    FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA,
    FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT,
    FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
    FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT,
    FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
    FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT,
    FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA,
    FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT,
    EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_CANDIDATE_CALIBRATED_ADAPTER_TENSOR_COUNT,
    EXPECTED_CANDIDATE_CALIBRATED_FRESH_ELEMENT_COUNT,
    EXPECTED_CANDIDATE_CALIBRATED_FRESH_SHA256,
    EXPECTED_CANDIDATE_CALIBRATED_FRESH_STORAGE_BYTES,
    EXPECTED_CANDIDATE_CALIBRATED_FRESH_TENSOR_COUNT,
    EXPECTED_CANDIDATE_CALIBRATED_STRICT_TARGET_TENSOR_COUNT,
    CANDIDATE_NORMALIZED_FRESH_CONFIDENCE_CONTRACT,
    CANDIDATE_NORMALIZED_MIGRATION_SCHEMA,
    EXPECTED_CANDIDATE_NORMALIZED_ADAPTER_TENSOR_COUNT,
    EXPECTED_CANDIDATE_NORMALIZED_FRESH_ELEMENT_COUNT,
    EXPECTED_CANDIDATE_NORMALIZED_FRESH_SHA256,
    EXPECTED_CANDIDATE_NORMALIZED_FRESH_STORAGE_BYTES,
    EXPECTED_CANDIDATE_NORMALIZED_FRESH_TENSOR_COUNT,
    EXPECTED_CANDIDATE_NORMALIZED_STRICT_TARGET_TENSOR_COUNT,
    CANDIDATE_ASYMMETRIC_FRESH_CONFIDENCE_CONTRACT,
    CANDIDATE_ASYMMETRIC_MIGRATION_SCHEMA,
    EXPECTED_CANDIDATE_ASYMMETRIC_ADAPTER_TENSOR_COUNT,
    EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_ELEMENT_COUNT,
    EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_SHA256,
    EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_STORAGE_BYTES,
    EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_TENSOR_COUNT,
    EXPECTED_CANDIDATE_ASYMMETRIC_STRICT_TARGET_TENSOR_COUNT,
    CANDIDATE_SET_ATTENTION_FRESH_CONFIDENCE_CONTRACT,
    CANDIDATE_SET_ATTENTION_MIGRATION_SCHEMA,
    GLOBAL_TRUST_VETO_FRESH_CONFIDENCE_CONTRACT,
    GLOBAL_TRUST_VETO_HEAD_GRADIENT_CONTRACT,
    GLOBAL_TRUST_VETO_MIGRATION_SCHEMA,
    DEPLOYED_ROUTER_FRESH_CONFIDENCE_CONTRACT,
    DEPLOYED_ROUTER_HEAD_GRADIENT_CONTRACT,
    DEPLOYED_ROUTER_MIGRATION_SCHEMA,
    EXPECTED_DEPLOYED_ROUTER_ADAPTER_TENSOR_COUNT,
    EXPECTED_DEPLOYED_ROUTER_FRESH_ELEMENT_COUNT,
    EXPECTED_DEPLOYED_ROUTER_FRESH_SHA256,
    EXPECTED_DEPLOYED_ROUTER_FRESH_STORAGE_BYTES,
    EXPECTED_DEPLOYED_ROUTER_FRESH_TENSOR_COUNT,
    EXPECTED_DEPLOYED_ROUTER_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_GLOBAL_TRUST_VETO_ADAPTER_TENSOR_COUNT,
    EXPECTED_GLOBAL_TRUST_VETO_FRESH_ELEMENT_COUNT,
    EXPECTED_GLOBAL_TRUST_VETO_FRESH_SHA256,
    EXPECTED_GLOBAL_TRUST_VETO_FRESH_STORAGE_BYTES,
    EXPECTED_GLOBAL_TRUST_VETO_FRESH_TENSOR_COUNT,
    EXPECTED_GLOBAL_TRUST_VETO_POOL_TENSOR_COUNT,
    EXPECTED_GLOBAL_TRUST_VETO_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_GLOBAL_TRUST_VETO_VETO_POOL_TENSOR_COUNT,
    EXPECTED_CANDIDATE_SET_ATTENTION_ADAPTER_TENSOR_COUNT,
    EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_ELEMENT_COUNT,
    EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_SHA256,
    EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_STORAGE_BYTES,
    EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_TENSOR_COUNT,
    EXPECTED_CANDIDATE_SET_ATTENTION_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_CANDIDATE_ABSOLUTE_ADAPTER_TENSOR_COUNT,
    EXPECTED_CANDIDATE_ABSOLUTE_FRESH_ELEMENT_COUNT,
    EXPECTED_CANDIDATE_ABSOLUTE_FRESH_SHA256,
    EXPECTED_CANDIDATE_ABSOLUTE_FRESH_STORAGE_BYTES,
    EXPECTED_CANDIDATE_ABSOLUTE_FRESH_TENSOR_COUNT,
    EXPECTED_ABSOLUTE_CAP_ADAPTER_TENSOR_COUNT,
    EXPECTED_ABSOLUTE_CAP_FRESH_ELEMENT_COUNT,
    EXPECTED_ABSOLUTE_CAP_FRESH_SHA256,
    EXPECTED_ABSOLUTE_CAP_FRESH_STORAGE_BYTES,
    EXPECTED_ABSOLUTE_CAP_FRESH_TENSOR_COUNT,
    EXPECTED_ABSOLUTE_CAP_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_FRESH_ELEMENT_COUNT,
    EXPECTED_FRESH_SHA256,
    EXPECTED_FRESH_STORAGE_BYTES,
    EXPECTED_FRESH_TENSOR_COUNT,
    EXPECTED_CARRIER_SLOPE_ADAPTER_TENSOR_COUNT,
    EXPECTED_CARRIER_SLOPE_FRESH_ELEMENT_COUNT,
    EXPECTED_CARRIER_SLOPE_FRESH_SHA256,
    EXPECTED_CARRIER_SLOPE_FRESH_STORAGE_BYTES,
    EXPECTED_CARRIER_SLOPE_FRESH_TENSOR_COUNT,
    EXPECTED_CARRIER_SLOPE_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_CARRIER_AFFINE_ADAPTER_TENSOR_COUNT,
    EXPECTED_CARRIER_AFFINE_FRESH_ELEMENT_COUNT,
    EXPECTED_CARRIER_AFFINE_FRESH_SHA256,
    EXPECTED_CARRIER_AFFINE_FRESH_STORAGE_BYTES,
    EXPECTED_CARRIER_AFFINE_FRESH_TENSOR_COUNT,
    EXPECTED_CARRIER_AFFINE_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_GATE_MARGIN_ADAPTER_TENSOR_COUNT,
    EXPECTED_GATE_MARGIN_FRESH_ELEMENT_COUNT,
    EXPECTED_GATE_MARGIN_FRESH_SHA256,
    EXPECTED_GATE_MARGIN_FRESH_STORAGE_BYTES,
    EXPECTED_GATE_MARGIN_FRESH_TENSOR_COUNT,
    EXPECTED_GATE_MARGIN_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_RANK_AFFINE_ADAPTER_TENSOR_COUNT,
    EXPECTED_RANK_AFFINE_FRESH_ELEMENT_COUNT,
    EXPECTED_RANK_AFFINE_FRESH_SHA256,
    EXPECTED_RANK_AFFINE_FRESH_STORAGE_BYTES,
    EXPECTED_RANK_AFFINE_FRESH_TENSOR_COUNT,
    EXPECTED_RANK_AFFINE_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_RANK_EVIDENCE_ADAPTER_TENSOR_COUNT,
    EXPECTED_RANK_EVIDENCE_FRESH_ELEMENT_COUNT,
    EXPECTED_RANK_EVIDENCE_FRESH_SHA256,
    EXPECTED_RANK_EVIDENCE_FRESH_STORAGE_BYTES,
    EXPECTED_RANK_EVIDENCE_FRESH_TENSOR_COUNT,
    EXPECTED_RANK_EVIDENCE_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_SPARSE_RANK_CHANNEL_ADAPTER_TENSOR_COUNT,
    EXPECTED_SPARSE_RANK_CHANNEL_FRESH_ELEMENT_COUNT,
    EXPECTED_SPARSE_RANK_CHANNEL_FRESH_SHA256,
    EXPECTED_SPARSE_RANK_CHANNEL_FRESH_STORAGE_BYTES,
    EXPECTED_SPARSE_RANK_CHANNEL_FRESH_TENSOR_COUNT,
    EXPECTED_SPARSE_RANK_CHANNEL_STRICT_TARGET_TENSOR_COUNT,
    EXPECTED_SIGNED_RANK_QUERY_POOL_ADAPTER_TENSOR_COUNT,
    EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_ELEMENT_COUNT,
    EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_SHA256,
    EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_STORAGE_BYTES,
    EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_TENSOR_COUNT,
    EXPECTED_SIGNED_RANK_QUERY_POOL_STRICT_TARGET_TENSOR_COUNT,
    RANK_EVIDENCE_FRESH_CONFIDENCE_CONTRACT,
    RANK_EVIDENCE_MIGRATION_SCHEMA,
    RANK_EVIDENCE_RESIDUAL_CONTRACT,
    RANK_AFFINE_FRESH_CONFIDENCE_CONTRACT,
    RANK_AFFINE_MIGRATION_SCHEMA,
    RANK_AFFINE_RESIDUAL_CONTRACT,
    GATE_MARGIN_FRESH_CONFIDENCE_CONTRACT,
    GATE_MARGIN_MIGRATION_SCHEMA,
    GATE_MARGIN_RESIDUAL_CONTRACT,
    SPARSE_RANK_CHANNEL_FRESH_CONFIDENCE_CONTRACT,
    SPARSE_RANK_CHANNEL_MIGRATION_SCHEMA,
    SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
    SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACT,
    SIGNED_RANK_QUERY_POOL_FRESH_CONFIDENCE_CONTRACT,
    SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA,
    TOKEN_CONDITIONED_POOL_FEATURE_CONTRACT,
    migrate_legacy_rank_to_confidence_adapter,
    validate_confidence_adapter_migration_audit,
)
from util.stage_b_dense_duty_audit import fingerprint_named_tensors


class _ProductionFreshScorer(nn.Module):
    def __init__(
        self,
        phrase_aggregation: str,
        *,
        rank_evidence_contract: str = CONFIDENCE_RANK_EVIDENCE_CONTRACT_OFF,
        pool_feature_contract: str = "patch_statistics_only_v1",
        residual_parameterization_gain: float = 1.0,
        gate_gradient_contract: str = "hard_detached_v1",
        head_gradient_contract: str = "shared_token_veto_global_absolute_v1",
    ) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.confidence_adapter = TokenAwareConfidenceAdapter(
                256,
                adapter_dim=64,
                max_text_len=256,
                patch_hidden_dim=64,
                score_topk=10,
                patch_score_clip=5.0,
                phrase_aggregation=phrase_aggregation,
                veto_cap_initial_ceiling=-0.1,
                rank_evidence_contract=rank_evidence_contract,
                pool_feature_contract=pool_feature_contract,
                residual_parameterization_gain=residual_parameterization_gain,
                gate_gradient_contract=gate_gradient_contract,
                head_gradient_contract=head_gradient_contract,
            )
            self.confidence_pool = AbsoluteConfidencePool(
                256,
                pool_hidden_dim=256,
                score_topk=10,
                pool_temperature=0.2,
                set_attention=(
                    pool_feature_contract
                    == CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION
                ),
            )
            if (
                head_gradient_contract
                == CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO
            ):
                self.confidence_veto_pool = AbsoluteConfidencePool(
                    256,
                    pool_hidden_dim=256,
                    score_topk=10,
                    pool_temperature=0.2,
                    set_attention=False,
                )
        self.register_buffer(
            "_dense_duty_contract_version", torch.tensor(3, dtype=torch.int64)
        )


class _ProductionFreshModel(nn.Module):
    def __init__(
        self,
        phrase_aggregation: str,
        *,
        rank_evidence_contract: str = CONFIDENCE_RANK_EVIDENCE_CONTRACT_OFF,
        pool_feature_contract: str = "patch_statistics_only_v1",
        residual_parameterization_gain: float = 1.0,
        gate_gradient_contract: str = "hard_detached_v1",
        head_gradient_contract: str = "shared_token_veto_global_absolute_v1",
    ) -> None:
        super().__init__()
        self.stage_b_fixed_text_scorer = _ProductionFreshScorer(
            phrase_aggregation,
            rank_evidence_contract=rank_evidence_contract,
            pool_feature_contract=pool_feature_contract,
            residual_parameterization_gain=residual_parameterization_gain,
            gate_gradient_contract=gate_gradient_contract,
            head_gradient_contract=head_gradient_contract,
        )


def _v53_production_model() -> _ProductionFreshModel:
    return _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=(
            CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
        ),
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
        head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
        ),
    )


def _v54_production_model() -> _ProductionFreshModel:
    return _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=(
            CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE
        ),
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
        head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
        ),
    )


def _v55_production_model() -> _ProductionFreshModel:
    return _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=(
            CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
        ),
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
        head_gradient_contract=(
            CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
        ),
    )


class _ProductionV53MigrationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(5, 4)
        self.stage_b_fixed_text_scorer = _ProductionFreshScorer(
            CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
            rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
            ),
            residual_parameterization_gain=0.25 / 0.03,
            gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
            ),
            head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
            ),
        )
        self.stage_b_fixed_text_scorer.rank_tower = nn.Linear(4, 3)


class _ProductionV54MigrationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(5, 4)
        self.stage_b_fixed_text_scorer = _ProductionFreshScorer(
            CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
            rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_REFERENCE
            ),
            residual_parameterization_gain=0.25 / 0.03,
            gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
            ),
            head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
            ),
        )
        self.stage_b_fixed_text_scorer.rank_tower = nn.Linear(4, 3)


class _ProductionV55MigrationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(5, 4)
        self.stage_b_fixed_text_scorer = _ProductionFreshScorer(
            CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
            rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
            ),
            residual_parameterization_gain=0.25 / 0.03,
            gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
            ),
            head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE
            ),
        )
        self.stage_b_fixed_text_scorer.rank_tower = nn.Linear(4, 3)


class _TargetScorer(nn.Module):
    def __init__(
        self,
        *,
        absolute_cap: bool = False,
        rank_evidence: bool = False,
        carrier_slope: bool = False,
        carrier_affine: bool = False,
        rank_channel: bool = False,
        signed_pool: bool = False,
    ) -> None:
        super().__init__()
        self.rank_tower = nn.Linear(4, 3)
        self.confidence_adapter = nn.Linear(4, 2)
        if absolute_cap:
            self.confidence_adapter.register_parameter(
                "veto_cap_raw_ceiling", nn.Parameter(torch.tensor(-2.25))
            )
        if rank_evidence:
            self.confidence_adapter.register_parameter(
                "rank_evidence_residual_scale",
                nn.Parameter(torch.zeros((), dtype=torch.float32)),
            )
        if carrier_slope or carrier_affine or rank_channel:
            self.confidence_adapter.carrier_rank_slope = nn.Linear(
                64, 1, bias=(carrier_affine or rank_channel)
            )
            nn.init.zeros_(self.confidence_adapter.carrier_rank_slope.weight)
            if self.confidence_adapter.carrier_rank_slope.bias is not None:
                nn.init.zeros_(self.confidence_adapter.carrier_rank_slope.bias)
            self.confidence_adapter.rank_evidence_contract = (
                SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
                if rank_channel
                else CARRIER_SLOPE_RESIDUAL_CONTRACT
            )
            if carrier_affine:
                self.confidence_adapter.rank_evidence_contract = (
                    CARRIER_AFFINE_RESIDUAL_CONTRACT
                )
        if rank_channel:
            self.confidence_adapter.rank_channel_norm = nn.LayerNorm(256)
            self.confidence_adapter.rank_channel_projection = nn.Linear(256, 64)
            self.confidence_adapter.rank_channel_logit_projection = nn.Linear(1, 64)
            self.confidence_adapter.rank_channel_output = nn.Linear(
                64, 1, bias=False
            )
            nn.init.zeros_(self.confidence_adapter.rank_channel_output.weight)
        self.confidence_adapter.pool_feature_contract = (
            SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACT
            if signed_pool
            else "patch_statistics_only_v1"
        )
        if signed_pool:
            self.confidence_adapter.global_query_norm = nn.LayerNorm(256)
            self.confidence_adapter.global_query_trunk = nn.Sequential(
                nn.Linear(257, 64),
                nn.GELU(),
                nn.Linear(64, 256),
                nn.GELU(),
            )
        self.confidence_pool = nn.Sequential(
            nn.Linear(3, 2), nn.GELU(), nn.Linear(2, 1)
        )
        self.register_buffer(
            "_dense_duty_contract_version", torch.tensor(3, dtype=torch.int64)
        )


class _TargetModel(nn.Module):
    def __init__(
        self,
        *,
        absolute_cap: bool = False,
        rank_evidence: bool = False,
        carrier_slope: bool = False,
        carrier_affine: bool = False,
        rank_channel: bool = False,
        signed_pool: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = nn.Linear(5, 4)
        self.stage_b_fixed_text_scorer = _TargetScorer(
            absolute_cap=absolute_cap,
            rank_evidence=rank_evidence,
            carrier_slope=carrier_slope,
            carrier_affine=carrier_affine,
            rank_channel=rank_channel,
            signed_pool=signed_pool,
        )


def _legacy_state(model: _TargetModel):
    target = model.state_dict()
    source = OrderedDict()
    for name, value in target.items():
        if (
            ".confidence_adapter." in name
            or ".confidence_pool." in name
            or ".confidence_veto_pool." in name
        ):
            continue
        source[name] = value.detach().clone()
    rank_prefix = "stage_b_fixed_text_scorer.rank_tower."
    legacy_prefix = "stage_b_fixed_text_scorer.confidence_tower."
    for name, value in target.items():
        if name.startswith(rank_prefix):
            source[legacy_prefix + name[len(rank_prefix) :]] = value.detach().clone()
    for name, value in target.items():
        if name.startswith("stage_b_fixed_text_scorer.confidence_pool."):
            source[name] = torch.randn_like(value) if value.is_floating_point() else value
    source["stage_b_fixed_text_scorer._dense_duty_contract_version"] = torch.tensor(
        1, dtype=torch.int64
    )
    return source


def _fingerprints(model, source):
    runtime = model.state_dict()
    rank_names = sorted(
        name
        for name in runtime
        if name.startswith("stage_b_fixed_text_scorer.rank_tower.")
    )
    transferred_names = sorted(
        name
        for name in runtime
        if not name.startswith("stage_b_fixed_text_scorer.")
        or name.startswith("stage_b_fixed_text_scorer.rank_tower.")
    )
    return (
        fingerprint_named_tensors(source, rank_names)["sha256"],
        fingerprint_named_tensors(source, transferred_names)["sha256"],
    )


def test_migration_transfers_rank_bitwise_and_fresh_initializes_confidence():
    torch.manual_seed(7)
    model = _TargetModel()
    runtime = {name: value.clone() for name, value in model.state_dict().items()}
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="unit legacy rank",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=11,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )
    model.load_state_dict(migrated, strict=True)
    for name, value in migrated.items():
        if name.startswith("stage_b_fixed_text_scorer.rank_tower."):
            assert torch.equal(value, source[name])
        if ".confidence_adapter." in name or ".confidence_pool." in name:
            assert torch.equal(value, runtime[name])
    assert audit["retired_confidence_loaded_tensor_count"] == 0
    assert audit["rank"]["sha256"] == rank_sha


def test_migration_marks_absolute_cap_fresh_surface():
    torch.manual_seed(7)
    model = _TargetModel(absolute_cap=True)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="unit legacy rank to absolute cap",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=11,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    cap_name = (
        "stage_b_fixed_text_scorer.confidence_adapter.veto_cap_raw_ceiling"
    )
    assert torch.equal(migrated[cap_name], model.state_dict()[cap_name])
    assert audit["schema"] == ABSOLUTE_CAP_MIGRATION_SCHEMA
    assert (
        audit["fresh_confidence_contract"]
        == ABSOLUTE_CAP_FRESH_CONFIDENCE_CONTRACT
    )


def test_migration_marks_rank_evidence_v4_fresh_surface():
    torch.manual_seed(7)
    model = _TargetModel(absolute_cap=True, rank_evidence=True)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="unit legacy rank to rank evidence",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=11,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    scale_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "rank_evidence_residual_scale"
    )
    assert torch.equal(migrated[scale_name], torch.zeros((), dtype=torch.float32))
    assert audit["schema"] == RANK_EVIDENCE_MIGRATION_SCHEMA
    assert (
        audit["fresh_confidence_contract"]
        == RANK_EVIDENCE_FRESH_CONFIDENCE_CONTRACT
    )
    assert audit["rank_evidence_contract"] == RANK_EVIDENCE_RESIDUAL_CONTRACT


def test_migration_marks_carrier_slope_v7_fresh_surface():
    torch.manual_seed(7)
    model = _TargetModel(absolute_cap=True, carrier_slope=True)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="unit legacy rank to carrier slope",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=11,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    weight_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "carrier_rank_slope.weight"
    )
    assert torch.equal(migrated[weight_name], torch.zeros((1, 64)))
    assert audit["schema"] == CARRIER_SLOPE_MIGRATION_SCHEMA
    assert (
        audit["fresh_confidence_contract"]
        == CARRIER_SLOPE_FRESH_CONFIDENCE_CONTRACT
    )
    assert audit["rank_evidence_contract"] == CARRIER_SLOPE_RESIDUAL_CONTRACT


def test_migration_marks_carrier_affine_v8_fresh_surface():
    torch.manual_seed(7)
    model = _TargetModel(absolute_cap=True, carrier_affine=True)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="unit legacy rank to carrier affine",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=11,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    weight_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "carrier_rank_slope.weight"
    )
    bias_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "carrier_rank_slope.bias"
    )
    assert torch.equal(migrated[weight_name], torch.zeros((1, 64)))
    assert torch.equal(migrated[bias_name], torch.zeros((1,)))
    assert audit["schema"] == CARRIER_AFFINE_MIGRATION_SCHEMA
    assert (
        audit["fresh_confidence_contract"]
        == CARRIER_AFFINE_FRESH_CONFIDENCE_CONTRACT
    )
    assert audit["rank_evidence_contract"] == CARRIER_AFFINE_RESIDUAL_CONTRACT


def test_migration_marks_sparse_rank_channel_v9_fresh_surface():
    torch.manual_seed(7)
    model = _TargetModel(absolute_cap=True, rank_channel=True)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="unit legacy rank to sparse rank channel",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=11,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    prefix = "stage_b_fixed_text_scorer.confidence_adapter."
    assert torch.equal(
        migrated[prefix + "carrier_rank_slope.weight"], torch.zeros((1, 64))
    )
    assert torch.equal(
        migrated[prefix + "carrier_rank_slope.bias"], torch.zeros((1,))
    )
    assert torch.equal(
        migrated[prefix + "rank_channel_output.weight"], torch.zeros((1, 64))
    )
    assert audit["schema"] == SPARSE_RANK_CHANNEL_MIGRATION_SCHEMA
    assert (
        audit["fresh_confidence_contract"]
        == SPARSE_RANK_CHANNEL_FRESH_CONFIDENCE_CONTRACT
    )
    assert (
        audit["rank_evidence_contract"]
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    )


def test_migration_marks_signed_rank_query_pool_v10_fresh_surface():
    torch.manual_seed(7)
    model = _TargetModel(
        absolute_cap=True,
        rank_channel=True,
        signed_pool=True,
    )
    runtime = model.state_dict()
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="unit legacy rank to signed rank-query pool",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=11,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    prefix = "stage_b_fixed_text_scorer.confidence_adapter."
    global_query_names = sorted(
        name
        for name in runtime
        if name.startswith(prefix + "global_query_")
    )
    assert global_query_names == [
        prefix + "global_query_norm.bias",
        prefix + "global_query_norm.weight",
        prefix + "global_query_trunk.0.bias",
        prefix + "global_query_trunk.0.weight",
        prefix + "global_query_trunk.2.bias",
        prefix + "global_query_trunk.2.weight",
    ]
    assert all(torch.equal(migrated[name], runtime[name]) for name in global_query_names)
    assert audit["schema"] == SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA
    assert (
        audit["fresh_confidence_contract"]
        == SIGNED_RANK_QUERY_POOL_FRESH_CONFIDENCE_CONTRACT
    )
    assert audit["pool_feature_contract"] == SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACT
    assert audit["rank_evidence_contract"] == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT


def test_migration_accepts_token_conditioned_pool_on_identical_fresh_surface():
    torch.manual_seed(7)
    model = _TargetModel(
        absolute_cap=True,
        rank_channel=True,
        signed_pool=True,
    )
    model.stage_b_fixed_text_scorer.confidence_adapter.pool_feature_contract = (
        TOKEN_CONDITIONED_POOL_FEATURE_CONTRACT
    )
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)
    _migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="unit legacy rank to token-conditioned pool",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=11,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )
    assert audit["schema"] == SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA
    assert audit["pool_feature_contract"] == TOKEN_CONDITIONED_POOL_FEATURE_CONTRACT


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("missing", "parameter surface is incomplete"),
        ("contract", "contract and parameter surface disagree"),
        ("shape", "tensor contract drifted"),
        ("missing_rank_channel", "requires sparse token evidence"),
    ),
)
def test_migration_rejects_signed_rank_query_pool_surface_drift(mutation, match):
    torch.manual_seed(7)
    model = _TargetModel(
        absolute_cap=True,
        rank_channel=mutation != "missing_rank_channel",
        signed_pool=True,
    )
    adapter = model.stage_b_fixed_text_scorer.confidence_adapter
    if mutation == "missing":
        adapter.global_query_norm = None
    elif mutation == "contract":
        adapter.pool_feature_contract = "patch_statistics_only_v1"
    elif mutation == "shape":
        adapter.global_query_trunk[0] = nn.Linear(258, 64)
    elif mutation != "missing_rank_channel":  # pragma: no cover
        raise AssertionError(mutation)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    with pytest.raises(RuntimeError, match=match):
        migrate_legacy_rank_to_confidence_adapter(
            model,
            source,
            checkpoint_label="drifted signed rank-query pool",
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=11,
            source_checkpoint_reason="signal",
            expected_rank_sha256=rank_sha,
            expected_transferred_sha256=transferred_sha,
        )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("unexpected_bias", "bias surface disagrees"),
        ("contract_mismatch", "contract and parameter surface disagree"),
        ("nonzero", "requires exactly a zero-initialized"),
        ("scalar_scale", "requires exactly a zero-initialized"),
    ),
)
def test_migration_rejects_invalid_carrier_slope_runtime_surface(mutation, match):
    torch.manual_seed(7)
    model = _TargetModel(absolute_cap=True, carrier_slope=True)
    adapter = model.stage_b_fixed_text_scorer.confidence_adapter
    if mutation == "unexpected_bias":
        adapter.carrier_rank_slope.bias = nn.Parameter(torch.zeros((1,)))
    elif mutation == "contract_mismatch":
        adapter.rank_evidence_contract = CONFIDENCE_RANK_EVIDENCE_CONTRACT_OFF
    elif mutation == "nonzero":
        with torch.no_grad():
            adapter.carrier_rank_slope.weight.fill_(1.0)
    elif mutation == "scalar_scale":
        adapter.register_parameter(
            "rank_evidence_residual_scale",
            nn.Parameter(torch.zeros((), dtype=torch.float32)),
        )
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(mutation)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    with pytest.raises(RuntimeError, match=match):
        migrate_legacy_rank_to_confidence_adapter(
            model,
            source,
            checkpoint_label="invalid carrier slope",
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=11,
            source_checkpoint_reason="signal",
            expected_rank_sha256=rank_sha,
            expected_transferred_sha256=transferred_sha,
        )


def test_migration_rejects_carrier_slope_without_absolute_cap_surface():
    torch.manual_seed(7)
    model = _TargetModel(carrier_slope=True)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    with pytest.raises(RuntimeError, match="requires exactly a zero-initialized"):
        migrate_legacy_rank_to_confidence_adapter(
            model,
            source,
            checkpoint_label="carrier slope without absolute cap",
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=11,
            source_checkpoint_reason="signal",
            expected_rank_sha256=rank_sha,
            expected_transferred_sha256=transferred_sha,
        )


def test_migration_rejects_rank_evidence_without_absolute_cap_surface():
    torch.manual_seed(7)
    model = _TargetModel(rank_evidence=True)
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    with pytest.raises(RuntimeError, match="requires.*absolute-cap surface"):
        migrate_legacy_rank_to_confidence_adapter(
            model,
            source,
            checkpoint_label="undeclared rank-evidence surface",
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=11,
            source_checkpoint_reason="signal",
            expected_rank_sha256=rank_sha,
            expected_transferred_sha256=transferred_sha,
        )


def test_v5_gated_pool_reuses_exact_v4_absolute_cap_fresh_surface():
    v4_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP
    ).state_dict()
    v5_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
    ).state_dict()
    v4_names = sorted(v4_state)
    v5_names = sorted(v5_state)

    assert v5_names == v4_names
    assert all(torch.equal(v5_state[name], v4_state[name]) for name in v4_names)
    assert (
        "stage_b_fixed_text_scorer.confidence_adapter.veto_cap_raw_ceiling"
        in v5_names
    )
    expected = {
        "sha256": EXPECTED_ABSOLUTE_CAP_FRESH_SHA256,
        "tensor_count": EXPECTED_ABSOLUTE_CAP_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_ABSOLUTE_CAP_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_ABSOLUTE_CAP_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }
    assert fingerprint_named_tensors(v4_state, v4_names) == expected
    assert fingerprint_named_tensors(v5_state, v5_names) == expected


def test_rank_evidence_v4_adds_one_zero_scalar_to_exact_production_surface():
    v2_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_LEGACY
    ).state_dict()
    v3_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
    ).state_dict()
    v4_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE,
    ).state_dict()
    v2_names = sorted(v2_state)
    v3_names = sorted(v3_state)
    v4_names = sorted(v4_state)
    scale_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "rank_evidence_residual_scale"
    )

    assert v4_names == sorted((*v3_names, scale_name))
    assert all(torch.equal(v4_state[name], v3_state[name]) for name in v3_names)
    assert torch.equal(v4_state[scale_name], torch.zeros((), dtype=torch.float32))
    assert sum(".confidence_adapter." in name for name in v4_names) == (
        EXPECTED_RANK_EVIDENCE_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(v2_state, v2_names) == {
        "sha256": EXPECTED_FRESH_SHA256,
        "tensor_count": EXPECTED_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }
    assert fingerprint_named_tensors(v3_state, v3_names) == {
        "sha256": EXPECTED_ABSOLUTE_CAP_FRESH_SHA256,
        "tensor_count": EXPECTED_ABSOLUTE_CAP_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_ABSOLUTE_CAP_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_ABSOLUTE_CAP_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }
    assert fingerprint_named_tensors(v4_state, v4_names) == {
        "sha256": EXPECTED_RANK_EVIDENCE_FRESH_SHA256,
        "tensor_count": EXPECTED_RANK_EVIDENCE_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_RANK_EVIDENCE_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_RANK_EVIDENCE_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_rank_affine_v5_adds_one_zero_bias_to_v4_production_surface():
    v4_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE,
    ).state_dict()
    v5_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CONFIDENCE_RANK_EVIDENCE_CONTRACT_AFFINE,
    ).state_dict()
    v4_names = sorted(v4_state)
    v5_names = sorted(v5_state)
    bias_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "rank_evidence_residual_bias"
    )

    assert v5_names == sorted((*v4_names, bias_name))
    assert all(torch.equal(v5_state[name], v4_state[name]) for name in v4_names)
    assert torch.equal(v5_state[bias_name], torch.zeros((), dtype=torch.float32))
    assert sum(".confidence_adapter." in name for name in v5_names) == (
        EXPECTED_RANK_AFFINE_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(v5_state, v5_names) == {
        "sha256": EXPECTED_RANK_AFFINE_FRESH_SHA256,
        "tensor_count": EXPECTED_RANK_AFFINE_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_RANK_AFFINE_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_RANK_AFFINE_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_gate_margin_v6_reuses_v4_zero_surface_under_distinct_contract():
    v4_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE,
    ).state_dict()
    v6_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN,
        residual_parameterization_gain=0.25 / 0.03,
    ).state_dict()

    assert sorted(v6_state) == sorted(v4_state)
    assert all(torch.equal(v6_state[name], v4_state[name]) for name in v4_state)
    assert sum(".confidence_adapter." in name for name in v6_state) == (
        EXPECTED_GATE_MARGIN_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(v6_state, sorted(v6_state)) == {
        "sha256": EXPECTED_GATE_MARGIN_FRESH_SHA256,
        "tensor_count": EXPECTED_GATE_MARGIN_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_GATE_MARGIN_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_GATE_MARGIN_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_carrier_slope_v7_adds_zero_linear_to_absolute_cap_surface():
    absolute_cap_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
    ).state_dict()
    carrier_slope_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CARRIER_SLOPE_RESIDUAL_CONTRACT,
        residual_parameterization_gain=0.25 / 0.03,
    ).state_dict()
    absolute_cap_names = sorted(absolute_cap_state)
    carrier_slope_names = sorted(carrier_slope_state)
    weight_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "carrier_rank_slope.weight"
    )
    scale_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "rank_evidence_residual_scale"
    )
    affine_bias_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "rank_evidence_residual_bias"
    )

    assert carrier_slope_names == sorted((*absolute_cap_names, weight_name))
    assert scale_name not in carrier_slope_names
    assert affine_bias_name not in carrier_slope_names
    assert torch.equal(carrier_slope_state[weight_name], torch.zeros((1, 64)))
    assert sum(
        ".confidence_adapter." in name for name in carrier_slope_names
    ) == EXPECTED_CARRIER_SLOPE_ADAPTER_TENSOR_COUNT
    assert fingerprint_named_tensors(
        carrier_slope_state, carrier_slope_names
    ) == {
        "sha256": EXPECTED_CARRIER_SLOPE_FRESH_SHA256,
        "tensor_count": EXPECTED_CARRIER_SLOPE_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_CARRIER_SLOPE_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_CARRIER_SLOPE_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_carrier_affine_v8_adds_only_zero_intercept_to_v7_surface():
    slope_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CARRIER_SLOPE_RESIDUAL_CONTRACT,
        residual_parameterization_gain=0.25 / 0.03,
    ).state_dict()
    affine_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CARRIER_AFFINE_RESIDUAL_CONTRACT,
        residual_parameterization_gain=0.25 / 0.03,
    ).state_dict()
    slope_names = sorted(slope_state)
    affine_names = sorted(affine_state)
    bias_name = (
        "stage_b_fixed_text_scorer.confidence_adapter."
        "carrier_rank_slope.bias"
    )
    assert affine_names == sorted((*slope_names, bias_name))
    assert all(torch.equal(affine_state[name], value) for name, value in slope_state.items())
    assert torch.equal(affine_state[bias_name], torch.zeros((1,)))
    assert sum(
        ".confidence_adapter." in name for name in affine_names
    ) == EXPECTED_CARRIER_AFFINE_ADAPTER_TENSOR_COUNT
    assert fingerprint_named_tensors(affine_state, affine_names) == {
        "sha256": EXPECTED_CARRIER_AFFINE_FRESH_SHA256,
        "tensor_count": EXPECTED_CARRIER_AFFINE_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_CARRIER_AFFINE_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_CARRIER_AFFINE_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_sparse_rank_channel_v9_adds_seven_tensors_to_v8_surface():
    affine_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=CARRIER_AFFINE_RESIDUAL_CONTRACT,
        residual_parameterization_gain=0.25 / 0.03,
    ).state_dict()
    rank_channel_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        residual_parameterization_gain=0.25 / 0.03,
    ).state_dict()
    added = sorted(set(rank_channel_state).difference(affine_state))
    assert added == [
        "stage_b_fixed_text_scorer.confidence_adapter."
        "rank_channel_logit_projection.bias",
        "stage_b_fixed_text_scorer.confidence_adapter."
        "rank_channel_logit_projection.weight",
        "stage_b_fixed_text_scorer.confidence_adapter.rank_channel_norm.bias",
        "stage_b_fixed_text_scorer.confidence_adapter.rank_channel_norm.weight",
        "stage_b_fixed_text_scorer.confidence_adapter.rank_channel_output.weight",
        "stage_b_fixed_text_scorer.confidence_adapter.rank_channel_projection.bias",
        "stage_b_fixed_text_scorer.confidence_adapter.rank_channel_projection.weight",
    ]
    assert all(
        torch.equal(rank_channel_state[name], value)
        for name, value in affine_state.items()
    )
    output_name = (
        "stage_b_fixed_text_scorer.confidence_adapter.rank_channel_output.weight"
    )
    assert torch.equal(rank_channel_state[output_name], torch.zeros((1, 64)))
    assert sum(
        ".confidence_adapter." in name for name in rank_channel_state
    ) == EXPECTED_SPARSE_RANK_CHANNEL_ADAPTER_TENSOR_COUNT
    assert fingerprint_named_tensors(
        rank_channel_state, sorted(rank_channel_state)
    ) == {
        "sha256": EXPECTED_SPARSE_RANK_CHANNEL_FRESH_SHA256,
        "tensor_count": EXPECTED_SPARSE_RANK_CHANNEL_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_SPARSE_RANK_CHANNEL_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_SPARSE_RANK_CHANNEL_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_signed_rank_query_pool_v10_adds_only_six_tensors_to_v9_surface():
    v9_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        residual_parameterization_gain=0.25 / 0.03,
    ).state_dict()
    v10_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACT,
        residual_parameterization_gain=0.25 / 0.03,
    ).state_dict()
    prefix = "stage_b_fixed_text_scorer.confidence_adapter."
    added = sorted(set(v10_state).difference(v9_state))
    assert added == [
        prefix + "global_query_norm.bias",
        prefix + "global_query_norm.weight",
        prefix + "global_query_trunk.0.bias",
        prefix + "global_query_trunk.0.weight",
        prefix + "global_query_trunk.2.bias",
        prefix + "global_query_trunk.2.weight",
    ]
    assert all(torch.equal(v10_state[name], value) for name, value in v9_state.items())
    assert sum(
        ".confidence_adapter." in name for name in v10_state
    ) == EXPECTED_SIGNED_RANK_QUERY_POOL_ADAPTER_TENSOR_COUNT
    assert fingerprint_named_tensors(v10_state, sorted(v10_state)) == {
        "sha256": EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_SHA256,
        "tensor_count": EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_candidate_absolute_v12_binds_zero_initialized_query_logit_surface():
    state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT
        ),
    ).state_dict()
    prefix = "stage_b_fixed_text_scorer.confidence_adapter."
    assert torch.equal(
        state[prefix + "candidate_absolute_head.3.weight"],
        torch.zeros((1, 256)),
    )
    assert torch.equal(
        state[prefix + "candidate_absolute_head.3.bias"], torch.zeros((1,))
    )
    assert sum(".confidence_adapter." in name for name in state) == (
        EXPECTED_CANDIDATE_ABSOLUTE_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(state, sorted(state)) == {
        "sha256": EXPECTED_CANDIDATE_ABSOLUTE_FRESH_SHA256,
        "tensor_count": EXPECTED_CANDIDATE_ABSOLUTE_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_CANDIDATE_ABSOLUTE_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_CANDIDATE_ABSOLUTE_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_candidate_calibrated_v13_binds_zero_initialized_monotone_surface():
    state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT
        ),
    ).state_dict()
    prefix = "stage_b_fixed_text_scorer.confidence_adapter."
    for suffix in (
        "candidate_patch_scale_raw",
        "candidate_veto_depth_raw",
        "candidate_coverage_depth_raw",
    ):
        value = state[prefix + suffix]
        assert value.dtype == torch.float32
        assert tuple(value.shape) == ()
        assert value.item() == 0.0
    assert sum(".confidence_adapter." in name for name in state) == (
        EXPECTED_CANDIDATE_CALIBRATED_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(state, sorted(state)) == {
        "sha256": EXPECTED_CANDIDATE_CALIBRATED_FRESH_SHA256,
        "tensor_count": EXPECTED_CANDIDATE_CALIBRATED_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_CANDIDATE_CALIBRATED_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_CANDIDATE_CALIBRATED_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_candidate_calibrated_v13_audit_validates_exact_surface():
    audit = {
        "schema": CANDIDATE_CALIBRATED_MIGRATION_SCHEMA,
        "fresh_confidence_contract": (
            CANDIDATE_CALIBRATED_FRESH_CONFIDENCE_CONTRACT
        ),
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": (
            CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED
        ),
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_CANDIDATE_CALIBRATED_FRESH_SHA256,
            "tensor_count": EXPECTED_CANDIDATE_CALIBRATED_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_CANDIDATE_CALIBRATED_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_CANDIDATE_CALIBRATED_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_CANDIDATE_CALIBRATED_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": 6,
        "strict_target_tensor_count": (
            EXPECTED_CANDIDATE_CALIBRATED_STRICT_TARGET_TENSOR_COUNT
        ),
    }
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit


def test_candidate_normalized_v14_binds_two_zero_initialized_veto_scalars():
    state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT
        ),
    ).state_dict()
    prefix = "stage_b_fixed_text_scorer.confidence_adapter."
    assert prefix + "candidate_patch_scale_raw" not in state
    for suffix in ("candidate_veto_depth_raw", "candidate_coverage_depth_raw"):
        value = state[prefix + suffix]
        assert value.dtype == torch.float32
        assert tuple(value.shape) == ()
        assert value.item() == 0.0
    assert sum(".confidence_adapter." in name for name in state) == (
        EXPECTED_CANDIDATE_NORMALIZED_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(state, sorted(state)) == {
        "sha256": EXPECTED_CANDIDATE_NORMALIZED_FRESH_SHA256,
        "tensor_count": EXPECTED_CANDIDATE_NORMALIZED_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_CANDIDATE_NORMALIZED_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_CANDIDATE_NORMALIZED_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_candidate_normalized_v14_audit_validates_exact_surface():
    audit = {
        "schema": CANDIDATE_NORMALIZED_MIGRATION_SCHEMA,
        "fresh_confidence_contract": CANDIDATE_NORMALIZED_FRESH_CONFIDENCE_CONTRACT,
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_CANDIDATE_NORMALIZED_FRESH_SHA256,
            "tensor_count": EXPECTED_CANDIDATE_NORMALIZED_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_CANDIDATE_NORMALIZED_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_CANDIDATE_NORMALIZED_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_CANDIDATE_NORMALIZED_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": 6,
        "strict_target_tensor_count": EXPECTED_CANDIDATE_NORMALIZED_STRICT_TARGET_TENSOR_COUNT,
    }
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit


def test_candidate_asymmetric_v15_binds_two_zero_initialized_veto_scalars():
    state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
        ),
    ).state_dict()
    prefix = "stage_b_fixed_text_scorer.confidence_adapter."
    assert prefix + "candidate_patch_scale_raw" not in state
    for suffix in ("candidate_veto_depth_raw", "candidate_coverage_depth_raw"):
        value = state[prefix + suffix]
        assert value.dtype == torch.float32
        assert tuple(value.shape) == ()
        assert value.item() == 0.0
    assert sum(".confidence_adapter." in name for name in state) == (
        EXPECTED_CANDIDATE_ASYMMETRIC_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(state, sorted(state)) == {
        "sha256": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_SHA256,
        "tensor_count": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_candidate_asymmetric_v15_audit_validates_exact_surface():
    audit = {
        "schema": CANDIDATE_ASYMMETRIC_MIGRATION_SCHEMA,
        "fresh_confidence_contract": CANDIDATE_ASYMMETRIC_FRESH_CONFIDENCE_CONTRACT,
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_SHA256,
            "tensor_count": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_CANDIDATE_ASYMMETRIC_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": 6,
        "strict_target_tensor_count": EXPECTED_CANDIDATE_ASYMMETRIC_STRICT_TARGET_TENSOR_COUNT,
    }
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit


def test_fulltext_global_absolute_v20_binds_production_fresh_surface_and_parameters():
    torch.manual_seed(7)
    first = _v53_production_model()
    torch.manual_seed(999)
    second = _v53_production_model()
    first_state = first.state_dict()
    second_state = second.state_dict()

    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert fingerprint_named_tensors(first_state, sorted(first_state)) == {
        "sha256": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
        "tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }
    confidence_parameters = [
        parameter
        for name, parameter in first.named_parameters()
        if ".confidence_adapter." in name or ".confidence_pool." in name
    ]
    assert len(confidence_parameters) == (
        EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT
    )
    assert sum(parameter.numel() for parameter in confidence_parameters) == (
        EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT
    )
    assert sum(".confidence_adapter." in name for name in first_state) == (
        EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT
    )
    assert sum(".confidence_pool." in name for name in first_state) == (
        EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT
    )
    forbidden_fragments = (
        "patch_residual",
        "global_query_norm",
        "veto_cap_raw_ceiling",
        "candidate_patch_scale_raw",
        "candidate_veto_depth_raw",
        "candidate_coverage_depth_raw",
    )
    assert not any(
        fragment in name for name in first_state for fragment in forbidden_fragments
    )


def test_fulltext_global_absolute_v20_audit_validates_exact_surface():
    audit = {
        "schema": FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
        "fresh_confidence_contract": (
            FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
        ),
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "head_gradient_contract": FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT,
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
            "tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT,
        "confidence_parameter_tensor_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT
        ),
        "confidence_parameter_element_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT
        ),
        "strict_target_tensor_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT
        ),
    }
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit

    for field in (
        "confidence_parameter_tensor_count",
        "confidence_parameter_element_count",
    ):
        drifted = dict(audit)
        drifted[field] += 1
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                drifted,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )


def test_fulltext_global_absolute_v20_migrates_u6551_without_rank_drift():
    model = _ProductionV53MigrationModel()
    runtime = model.state_dict()
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="U6551 to V53 fresh confidence",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    rank_names = sorted(
        name
        for name in runtime
        if name.startswith("stage_b_fixed_text_scorer.rank_tower.")
    )
    assert all(torch.equal(migrated[name], source[name]) for name in rank_names)
    assert fingerprint_named_tensors(migrated, rank_names)["sha256"] == rank_sha
    assert audit["rank"]["sha256"] == rank_sha
    assert audit["source_optimizer_updates"] == 6551
    assert audit["schema"] == FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
    assert (
        audit["fresh_confidence_contract"]
        == FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
    )
    assert audit["confidence_parameter_tensor_count"] == 65
    assert audit["confidence_parameter_element_count"] == 534_725


def test_fulltext_global_absolute_v20_rejects_v52_confidence_continuation():
    model = _ProductionV53MigrationModel()
    source = _legacy_state(model)
    v52_state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
        ),
        head_gradient_contract=CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE,
    ).state_dict()
    source.update(
        (name, value.detach().clone())
        for name, value in v52_state.items()
        if ".confidence_adapter." in name
    )
    rank_sha, transferred_sha = _fingerprints(model, source)

    with pytest.raises(RuntimeError, match="already contains a confidence adapter"):
        migrate_legacy_rank_to_confidence_adapter(
            model,
            source,
            checkpoint_label="invalid V52 confidence continuation",
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            expected_rank_sha256=rank_sha,
            expected_transferred_sha256=transferred_sha,
        )


def test_candidate_sample_calibrator_v19_reuses_exact_no_router_fresh_surface():
    state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
        ),
        head_gradient_contract=CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE,
    ).state_dict()
    assert not any("deployed_router" in name for name in state)
    assert sum(".confidence_adapter." in name for name in state) == (
        EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(state, sorted(state)) == {
        "sha256": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_SHA256,
        "tensor_count": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_candidate_sample_calibrator_v19_audit_validates_exact_surface():
    audit = {
        "schema": CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA,
        "fresh_confidence_contract": (
            CANDIDATE_SAMPLE_CALIBRATOR_FRESH_CONFIDENCE_CONTRACT
        ),
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "head_gradient_contract": (
            CANDIDATE_SAMPLE_CALIBRATOR_HEAD_GRADIENT_CONTRACT
        ),
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_SHA256,
            "tensor_count": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": (
            EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_ADAPTER_TENSOR_COUNT
        ),
        "pool_tensor_count": 6,
        "strict_target_tensor_count": (
            EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_STRICT_TARGET_TENSOR_COUNT
        ),
    }
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit


def test_deployed_router_v18_binds_exact_zero_init_independent_surface():
    state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
        ),
        head_gradient_contract=CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER,
    ).state_dict()
    prefix = "stage_b_fixed_text_scorer.confidence_adapter.deployed_router_"
    router_names = sorted(name for name in state if name.startswith(prefix))
    assert len(router_names) == 6
    assert tuple(state[prefix + "norm.weight"].shape) == (10,)
    assert tuple(state[prefix + "residual.0.weight"].shape) == (64, 10)
    assert tuple(state[prefix + "residual.2.weight"].shape) == (1, 64)
    assert torch.count_nonzero(state[prefix + "residual.2.weight"]).item() == 0
    assert torch.count_nonzero(state[prefix + "residual.2.bias"]).item() == 0
    assert sum(".confidence_adapter." in name for name in state) == (
        EXPECTED_DEPLOYED_ROUTER_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(state, sorted(state)) == {
        "sha256": EXPECTED_DEPLOYED_ROUTER_FRESH_SHA256,
        "tensor_count": EXPECTED_DEPLOYED_ROUTER_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_DEPLOYED_ROUTER_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_DEPLOYED_ROUTER_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_deployed_router_v18_audit_validates_exact_surface():
    audit = {
        "schema": DEPLOYED_ROUTER_MIGRATION_SCHEMA,
        "fresh_confidence_contract": DEPLOYED_ROUTER_FRESH_CONFIDENCE_CONTRACT,
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "head_gradient_contract": DEPLOYED_ROUTER_HEAD_GRADIENT_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_DEPLOYED_ROUTER_FRESH_SHA256,
            "tensor_count": EXPECTED_DEPLOYED_ROUTER_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_DEPLOYED_ROUTER_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_DEPLOYED_ROUTER_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_DEPLOYED_ROUTER_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": 6,
        "strict_target_tensor_count": EXPECTED_DEPLOYED_ROUTER_STRICT_TARGET_TENSOR_COUNT,
    }
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            {**audit, "head_gradient_contract": "split_token_veto_global_absolute_v2"},
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )


def test_global_trust_veto_v17_binds_the_independent_zero_init_pool():
    state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
        ),
        head_gradient_contract=CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO,
    ).state_dict()
    veto_prefix = "stage_b_fixed_text_scorer.confidence_veto_pool."
    veto_names = sorted(name for name in state if name.startswith(veto_prefix))
    assert len(veto_names) == EXPECTED_GLOBAL_TRUST_VETO_VETO_POOL_TENSOR_COUNT
    assert torch.count_nonzero(
        state[veto_prefix + "residual.4.weight"]
    ).item() == 0
    assert torch.count_nonzero(
        state[veto_prefix + "residual.4.bias"]
    ).item() == 0
    assert sum(".confidence_adapter." in name for name in state) == (
        EXPECTED_GLOBAL_TRUST_VETO_ADAPTER_TENSOR_COUNT
    )
    assert sum(".confidence_pool." in name for name in state) == (
        EXPECTED_GLOBAL_TRUST_VETO_POOL_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(state, sorted(state)) == {
        "sha256": EXPECTED_GLOBAL_TRUST_VETO_FRESH_SHA256,
        "tensor_count": EXPECTED_GLOBAL_TRUST_VETO_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_GLOBAL_TRUST_VETO_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_GLOBAL_TRUST_VETO_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_global_trust_veto_v17_audit_validates_exact_surface():
    audit = {
        "schema": GLOBAL_TRUST_VETO_MIGRATION_SCHEMA,
        "fresh_confidence_contract": GLOBAL_TRUST_VETO_FRESH_CONFIDENCE_CONTRACT,
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "head_gradient_contract": GLOBAL_TRUST_VETO_HEAD_GRADIENT_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_GLOBAL_TRUST_VETO_FRESH_SHA256,
            "tensor_count": EXPECTED_GLOBAL_TRUST_VETO_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_GLOBAL_TRUST_VETO_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_GLOBAL_TRUST_VETO_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_GLOBAL_TRUST_VETO_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": EXPECTED_GLOBAL_TRUST_VETO_POOL_TENSOR_COUNT,
        "veto_pool_tensor_count": (
            EXPECTED_GLOBAL_TRUST_VETO_VETO_POOL_TENSOR_COUNT
        ),
        "strict_target_tensor_count": (
            EXPECTED_GLOBAL_TRUST_VETO_STRICT_TARGET_TENSOR_COUNT
        ),
    }
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit
    for field, value in (
        ("veto_pool_tensor_count", 0),
        ("head_gradient_contract", "split_token_veto_global_absolute_v2"),
    ):
        drifted = dict(audit)
        drifted[field] = value
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                drifted,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )


def test_candidate_set_attention_v16_binds_expanded_pool_surface():
    state = _ProductionFreshModel(
        CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
        rank_evidence_contract=(
            CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
        ),
        pool_feature_contract=(
            CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION
        ),
        residual_parameterization_gain=0.25 / 0.03,
        gate_gradient_contract=(
            CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT
        ),
    ).state_dict()
    prefix = "stage_b_fixed_text_scorer.confidence_pool."
    assert prefix + "set_seed" in state
    assert sum(".confidence_adapter." in name for name in state) == (
        EXPECTED_CANDIDATE_SET_ATTENTION_ADAPTER_TENSOR_COUNT
    )
    assert fingerprint_named_tensors(state, sorted(state)) == {
        "sha256": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_SHA256,
        "tensor_count": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_candidate_set_attention_v16_audit_validates_exact_surface():
    audit = {
        "schema": CANDIDATE_SET_ATTENTION_MIGRATION_SCHEMA,
        "fresh_confidence_contract": CANDIDATE_SET_ATTENTION_FRESH_CONFIDENCE_CONTRACT,
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_SHA256,
            "tensor_count": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_CANDIDATE_SET_ATTENTION_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": 25,
        "strict_target_tensor_count": EXPECTED_CANDIDATE_SET_ATTENTION_STRICT_TARGET_TENSOR_COUNT,
    }
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit


def test_migration_rejects_missing_or_undeclared_rank_state():
    model = _TargetModel()
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)
    source.pop("stage_b_fixed_text_scorer.rank_tower.weight")
    with pytest.raises(RuntimeError, match="rank key drift"):
        migrate_legacy_rank_to_confidence_adapter(
            model,
            source,
            checkpoint_label="broken legacy rank",
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=11,
            source_checkpoint_reason="signal",
            expected_rank_sha256=rank_sha,
            expected_transferred_sha256=transferred_sha,
        )


def _formal_migration_audit(
    *,
    absolute_cap: bool = False,
    rank_evidence: bool = False,
    rank_affine: bool = False,
    gate_margin: bool = False,
    carrier_slope: bool = False,
    carrier_affine: bool = False,
    rank_channel: bool = False,
    signed_pool: bool = False,
):
    if (
        rank_evidence
        or rank_affine
        or gate_margin
        or carrier_slope
        or carrier_affine
        or rank_channel
        or signed_pool
    ) and not absolute_cap:
        raise ValueError("rank-evidence audit requires the absolute-cap surface")
    if signed_pool and not rank_channel:
        raise ValueError("signed-pool audit requires the sparse rank channel")
    if signed_pool:
        schema = SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA
        fresh_sha256 = EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_SHA256
        fresh_tensor_count = EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_STORAGE_BYTES
        adapter_tensor_count = EXPECTED_SIGNED_RANK_QUERY_POOL_ADAPTER_TENSOR_COUNT
        strict_target_tensor_count = (
            EXPECTED_SIGNED_RANK_QUERY_POOL_STRICT_TARGET_TENSOR_COUNT
        )
    elif rank_channel:
        schema = SPARSE_RANK_CHANNEL_MIGRATION_SCHEMA
        fresh_sha256 = EXPECTED_SPARSE_RANK_CHANNEL_FRESH_SHA256
        fresh_tensor_count = EXPECTED_SPARSE_RANK_CHANNEL_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_SPARSE_RANK_CHANNEL_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_SPARSE_RANK_CHANNEL_FRESH_STORAGE_BYTES
        adapter_tensor_count = EXPECTED_SPARSE_RANK_CHANNEL_ADAPTER_TENSOR_COUNT
        strict_target_tensor_count = (
            EXPECTED_SPARSE_RANK_CHANNEL_STRICT_TARGET_TENSOR_COUNT
        )
    elif carrier_affine:
        schema = CARRIER_AFFINE_MIGRATION_SCHEMA
        fresh_sha256 = EXPECTED_CARRIER_AFFINE_FRESH_SHA256
        fresh_tensor_count = EXPECTED_CARRIER_AFFINE_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_CARRIER_AFFINE_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_CARRIER_AFFINE_FRESH_STORAGE_BYTES
        adapter_tensor_count = EXPECTED_CARRIER_AFFINE_ADAPTER_TENSOR_COUNT
        strict_target_tensor_count = (
            EXPECTED_CARRIER_AFFINE_STRICT_TARGET_TENSOR_COUNT
        )
    elif carrier_slope:
        schema = CARRIER_SLOPE_MIGRATION_SCHEMA
        fresh_sha256 = EXPECTED_CARRIER_SLOPE_FRESH_SHA256
        fresh_tensor_count = EXPECTED_CARRIER_SLOPE_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_CARRIER_SLOPE_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_CARRIER_SLOPE_FRESH_STORAGE_BYTES
        adapter_tensor_count = EXPECTED_CARRIER_SLOPE_ADAPTER_TENSOR_COUNT
        strict_target_tensor_count = (
            EXPECTED_CARRIER_SLOPE_STRICT_TARGET_TENSOR_COUNT
        )
    elif gate_margin:
        schema = GATE_MARGIN_MIGRATION_SCHEMA
        fresh_sha256 = EXPECTED_GATE_MARGIN_FRESH_SHA256
        fresh_tensor_count = EXPECTED_GATE_MARGIN_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_GATE_MARGIN_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_GATE_MARGIN_FRESH_STORAGE_BYTES
        adapter_tensor_count = EXPECTED_GATE_MARGIN_ADAPTER_TENSOR_COUNT
        strict_target_tensor_count = EXPECTED_GATE_MARGIN_STRICT_TARGET_TENSOR_COUNT
    elif rank_affine:
        schema = RANK_AFFINE_MIGRATION_SCHEMA
        fresh_sha256 = EXPECTED_RANK_AFFINE_FRESH_SHA256
        fresh_tensor_count = EXPECTED_RANK_AFFINE_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_RANK_AFFINE_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_RANK_AFFINE_FRESH_STORAGE_BYTES
        adapter_tensor_count = EXPECTED_RANK_AFFINE_ADAPTER_TENSOR_COUNT
        strict_target_tensor_count = EXPECTED_RANK_AFFINE_STRICT_TARGET_TENSOR_COUNT
    elif rank_evidence:
        schema = RANK_EVIDENCE_MIGRATION_SCHEMA
        fresh_sha256 = EXPECTED_RANK_EVIDENCE_FRESH_SHA256
        fresh_tensor_count = EXPECTED_RANK_EVIDENCE_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_RANK_EVIDENCE_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_RANK_EVIDENCE_FRESH_STORAGE_BYTES
        adapter_tensor_count = EXPECTED_RANK_EVIDENCE_ADAPTER_TENSOR_COUNT
        strict_target_tensor_count = (
            EXPECTED_RANK_EVIDENCE_STRICT_TARGET_TENSOR_COUNT
        )
    elif absolute_cap:
        schema = ABSOLUTE_CAP_MIGRATION_SCHEMA
        fresh_sha256 = EXPECTED_ABSOLUTE_CAP_FRESH_SHA256
        fresh_tensor_count = EXPECTED_ABSOLUTE_CAP_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_ABSOLUTE_CAP_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_ABSOLUTE_CAP_FRESH_STORAGE_BYTES
        adapter_tensor_count = EXPECTED_ABSOLUTE_CAP_ADAPTER_TENSOR_COUNT
        strict_target_tensor_count = EXPECTED_ABSOLUTE_CAP_STRICT_TARGET_TENSOR_COUNT
    else:
        schema = "pivot.stageb.rank_to_token_confidence_adapter/v2"
        fresh_sha256 = EXPECTED_FRESH_SHA256
        fresh_tensor_count = EXPECTED_FRESH_TENSOR_COUNT
        fresh_element_count = EXPECTED_FRESH_ELEMENT_COUNT
        fresh_storage_bytes = EXPECTED_FRESH_STORAGE_BYTES
        adapter_tensor_count = 22
        strict_target_tensor_count = 1617
    audit = {
        "schema": schema,
        "token_logit_contract": (
            "detached_rank_token_minus_zero_init_residual_v1"
        ),
        "pool_feature_contract": (
            SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACT
            if signed_pool
            else "patch_statistics_only_v1"
        ),
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {
            "sha256": "b" * 64,
            "tensor_count": 453,
            "nonfinite_count": 0,
        },
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": fresh_sha256,
            "tensor_count": fresh_tensor_count,
            "element_count": fresh_element_count,
            "storage_bytes": fresh_storage_bytes,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": adapter_tensor_count,
        "pool_tensor_count": 6,
        "strict_target_tensor_count": strict_target_tensor_count,
    }
    if signed_pool:
        audit["fresh_confidence_contract"] = (
            SIGNED_RANK_QUERY_POOL_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif rank_channel:
        audit["fresh_confidence_contract"] = (
            SPARSE_RANK_CHANNEL_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif carrier_affine:
        audit["fresh_confidence_contract"] = (
            CARRIER_AFFINE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = CARRIER_AFFINE_RESIDUAL_CONTRACT
    elif carrier_slope:
        audit["fresh_confidence_contract"] = (
            CARRIER_SLOPE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = CARRIER_SLOPE_RESIDUAL_CONTRACT
    elif gate_margin:
        audit["fresh_confidence_contract"] = (
            GATE_MARGIN_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = GATE_MARGIN_RESIDUAL_CONTRACT
    elif rank_affine:
        audit["fresh_confidence_contract"] = (
            RANK_AFFINE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = RANK_AFFINE_RESIDUAL_CONTRACT
    elif rank_evidence:
        audit["fresh_confidence_contract"] = (
            RANK_EVIDENCE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = RANK_EVIDENCE_RESIDUAL_CONTRACT
    elif absolute_cap:
        audit["fresh_confidence_contract"] = (
            ABSOLUTE_CAP_FRESH_CONFIDENCE_CONTRACT
        )
    return audit


def test_migration_audit_binds_exact_fresh_confidence_fingerprint():
    audit = _formal_migration_audit()
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    drifted = dict(audit)
    drifted["fresh_confidence"] = dict(audit["fresh_confidence"])
    drifted["fresh_confidence"]["sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            drifted,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )

    drifted_contract = dict(audit)
    drifted_contract["pool_feature_contract"] = "full_query_bypass"
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            drifted_contract,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )


def test_migration_audit_binds_absolute_cap_surface_and_fingerprint():
    audit = _formal_migration_audit(absolute_cap=True)
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    missing_contract = dict(audit)
    missing_contract.pop("fresh_confidence_contract")
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            missing_contract,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )

    legacy_fingerprint = dict(audit)
    legacy_fingerprint["fresh_confidence"] = dict(audit["fresh_confidence"])
    legacy_fingerprint["fresh_confidence"]["sha256"] = EXPECTED_FRESH_SHA256
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            legacy_fingerprint,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )


def test_migration_audit_binds_rank_evidence_v4_surface_and_fingerprint():
    audit = _formal_migration_audit(absolute_cap=True, rank_evidence=True)
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    for missing_name in (
        "fresh_confidence_contract",
        "rank_evidence_contract",
    ):
        missing_contract = dict(audit)
        missing_contract.pop(missing_name)
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                missing_contract,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )

    legacy_fingerprint = dict(audit)
    legacy_fingerprint["fresh_confidence"] = dict(audit["fresh_confidence"])
    legacy_fingerprint["fresh_confidence"]["sha256"] = (
        EXPECTED_ABSOLUTE_CAP_FRESH_SHA256
    )
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            legacy_fingerprint,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )


def test_migration_audit_binds_rank_affine_v5_surface_and_fingerprint():
    audit = _formal_migration_audit(absolute_cap=True, rank_affine=True)
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    for missing_name in (
        "fresh_confidence_contract",
        "rank_evidence_contract",
    ):
        missing_contract = dict(audit)
        missing_contract.pop(missing_name)
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                missing_contract,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )


def test_migration_audit_binds_gate_margin_v6_contract_and_fingerprint():
    audit = _formal_migration_audit(absolute_cap=True, gate_margin=True)
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    for missing_name in ("fresh_confidence_contract", "rank_evidence_contract"):
        drifted = dict(audit)
        drifted.pop(missing_name)
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                drifted,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )


def test_migration_audit_binds_carrier_slope_v7_contract_and_fingerprint():
    audit = _formal_migration_audit(absolute_cap=True, carrier_slope=True)
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    for missing_name in ("fresh_confidence_contract", "rank_evidence_contract"):
        drifted = dict(audit)
        drifted.pop(missing_name)
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                drifted,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )

    legacy_surface = dict(audit)
    legacy_surface["fresh_confidence"] = dict(audit["fresh_confidence"])
    legacy_surface["fresh_confidence"]["sha256"] = (
        EXPECTED_GATE_MARGIN_FRESH_SHA256
    )
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            legacy_surface,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )


def test_migration_audit_binds_carrier_affine_v8_contract_and_fingerprint():
    audit = _formal_migration_audit(absolute_cap=True, carrier_affine=True)
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    legacy_surface = dict(audit)
    legacy_surface["fresh_confidence"] = dict(audit["fresh_confidence"])
    legacy_surface["fresh_confidence"]["sha256"] = (
        EXPECTED_CARRIER_SLOPE_FRESH_SHA256
    )
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            legacy_surface,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )


def test_migration_audit_binds_sparse_rank_channel_v9_contract_and_fingerprint():
    audit = _formal_migration_audit(absolute_cap=True, rank_channel=True)
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    drifted = dict(audit)
    drifted["fresh_confidence"] = dict(audit["fresh_confidence"])
    drifted["fresh_confidence"]["sha256"] = EXPECTED_CARRIER_AFFINE_FRESH_SHA256
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            drifted,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )


def test_migration_audit_binds_signed_rank_query_pool_v10_contract():
    audit = _formal_migration_audit(
        absolute_cap=True,
        rank_channel=True,
        signed_pool=True,
    )
    validated = validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    )
    assert validated == audit

    drifted_values = (
        ("schema", SPARSE_RANK_CHANNEL_MIGRATION_SCHEMA),
        ("fresh_confidence_contract", SPARSE_RANK_CHANNEL_FRESH_CONFIDENCE_CONTRACT),
        ("rank_evidence_contract", CARRIER_AFFINE_RESIDUAL_CONTRACT),
        ("pool_feature_contract", "patch_statistics_only_v1"),
    )
    for field, value in drifted_values:
        drifted = dict(audit)
        drifted[field] = value
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                drifted,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )

    drifted_fingerprint = dict(audit)
    drifted_fingerprint["fresh_confidence"] = dict(audit["fresh_confidence"])
    drifted_fingerprint["fresh_confidence"]["sha256"] = (
        EXPECTED_SPARSE_RANK_CHANNEL_FRESH_SHA256
    )
    with pytest.raises(RuntimeError, match="migration audit is invalid"):
        validate_confidence_adapter_migration_audit(
            drifted_fingerprint,
            source_checkpoint_sha256="a" * 64,
            source_optimizer_updates=6551,
            source_checkpoint_reason="signal",
            rank_sha256="b" * 64,
            transferred_sha256="c" * 64,
        )


def _v54_exact_residual_migration_audit() -> dict:
    return {
        "schema": FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA,
        "fresh_confidence_contract": (
            FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
        ),
        "rank_evidence_contract": SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT,
        "head_gradient_contract": FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT,
        "token_logit_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "pool_feature_contract": (
            FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
        ),
        "source_checkpoint_sha256": "a" * 64,
        "source_optimizer_updates": 6551,
        "source_checkpoint_reason": "signal",
        "rank": {"sha256": "b" * 64, "tensor_count": 453, "nonfinite_count": 0},
        "transferred": {
            "sha256": "c" * 64,
            "tensor_count": 1588,
            "nonfinite_count": 0,
        },
        "fresh_confidence": {
            "sha256": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
            "tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
            "element_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
            "storage_bytes": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
            "nonfinite_count": 0,
        },
        "retired_confidence_tower_tensor_count": 453,
        "retired_confidence_pool_tensor_count": 6,
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT,
        "pool_tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT,
        "confidence_parameter_tensor_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT
        ),
        "confidence_parameter_element_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT
        ),
        "strict_target_tensor_count": (
            EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT
        ),
    }


def test_v54_exact_residual_v21_preserves_v53_parameter_surface():
    v53_state = _v53_production_model().state_dict()
    v54_state = _v54_production_model().state_dict()

    assert tuple(v53_state) == tuple(v54_state)
    assert all(torch.equal(v53_state[name], v54_state[name]) for name in v53_state)
    assert fingerprint_named_tensors(v54_state, sorted(v54_state)) == {
        "sha256": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
        "tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_v54_exact_residual_v21_audit_rejects_v53_contract_mixing():
    audit = _v54_exact_residual_migration_audit()
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit

    for field, value in (
        ("schema", FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA),
        (
            "fresh_confidence_contract",
            FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT,
        ),
        ("pool_feature_contract", FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT),
    ):
        drifted = dict(audit)
        drifted[field] = value
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                drifted,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )


def test_v54_exact_residual_v21_migrates_u6551_without_rank_drift():
    model = _ProductionV54MigrationModel()
    runtime = model.state_dict()
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="U6551 to V54 fresh exact-residual confidence",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    rank_names = sorted(
        name
        for name in runtime
        if name.startswith("stage_b_fixed_text_scorer.rank_tower.")
    )
    assert all(torch.equal(migrated[name], source[name]) for name in rank_names)
    assert fingerprint_named_tensors(migrated, rank_names)["sha256"] == rank_sha
    assert audit["schema"] == (
        FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
    )
    assert audit["fresh_confidence_contract"] == (
        FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
    )
    assert audit["pool_feature_contract"] == (
        FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
    )
    assert audit["confidence_parameter_tensor_count"] == 65
    assert audit["confidence_parameter_element_count"] == 534_725


def _v55_independent_absolute_migration_audit() -> dict:
    audit = _v54_exact_residual_migration_audit()
    audit.update(
        {
            "schema": FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA,
            "fresh_confidence_contract": (
                FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
            ),
            "head_gradient_contract": (
                FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            ),
            "pool_feature_contract": (
                FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT
            ),
        }
    )
    return audit


def test_v55_independent_absolute_v22_preserves_v54_parameter_surface():
    v54_state = _v54_production_model().state_dict()
    v55_state = _v55_production_model().state_dict()

    assert tuple(v54_state) == tuple(v55_state)
    assert all(torch.equal(v54_state[name], v55_state[name]) for name in v54_state)
    assert fingerprint_named_tensors(v55_state, sorted(v55_state)) == {
        "sha256": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
        "tensor_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT,
        "element_count": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT,
        "storage_bytes": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES,
        "nonfinite_count": 0,
    }


def test_v55_independent_absolute_v22_audit_rejects_v54_contract_mixing():
    audit = _v55_independent_absolute_migration_audit()
    assert validate_confidence_adapter_migration_audit(
        audit,
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        rank_sha256="b" * 64,
        transferred_sha256="c" * 64,
    ) == audit

    for field, value in (
        ("schema", FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA),
        (
            "fresh_confidence_contract",
            FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT,
        ),
        (
            "head_gradient_contract",
            FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT,
        ),
        (
            "pool_feature_contract",
            FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT,
        ),
    ):
        drifted = dict(audit)
        drifted[field] = value
        with pytest.raises(RuntimeError, match="migration audit is invalid"):
            validate_confidence_adapter_migration_audit(
                drifted,
                source_checkpoint_sha256="a" * 64,
                source_optimizer_updates=6551,
                source_checkpoint_reason="signal",
                rank_sha256="b" * 64,
                transferred_sha256="c" * 64,
            )


def test_v55_independent_absolute_v22_migrates_u6551_without_rank_drift():
    model = _ProductionV55MigrationModel()
    runtime = model.state_dict()
    source = _legacy_state(model)
    rank_sha, transferred_sha = _fingerprints(model, source)

    migrated, audit = migrate_legacy_rank_to_confidence_adapter(
        model,
        source,
        checkpoint_label="U6551 to V55 fresh independent confidence",
        source_checkpoint_sha256="a" * 64,
        source_optimizer_updates=6551,
        source_checkpoint_reason="signal",
        expected_rank_sha256=rank_sha,
        expected_transferred_sha256=transferred_sha,
    )

    rank_names = sorted(
        name
        for name in runtime
        if name.startswith("stage_b_fixed_text_scorer.rank_tower.")
    )
    assert all(torch.equal(migrated[name], source[name]) for name in rank_names)
    assert audit["schema"] == FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA
    assert audit["fresh_confidence_contract"] == (
        FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
    )
    assert audit["head_gradient_contract"] == (
        FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT
    )
    assert audit["pool_feature_contract"] == (
        FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT
    )
    assert audit["confidence_parameter_tensor_count"] == 65
    assert audit["confidence_parameter_element_count"] == 534_725
