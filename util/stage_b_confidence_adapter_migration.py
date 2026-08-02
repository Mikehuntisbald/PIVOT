"""Strict U6551 rank-to-confidence-adapter state migration."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch
from torch import nn

from util.stage_b_dense_duty_audit import fingerprint_named_tensors


MIGRATION_SCHEMA = "pivot.stageb.rank_to_token_confidence_adapter/v2"
ABSOLUTE_CAP_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_absolute_cap/v3"
)
RANK_EVIDENCE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_rank_evidence/v4"
)
RANK_AFFINE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_rank_affine/v5"
)
GATE_MARGIN_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_gate_margin/v6"
)
CARRIER_SLOPE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_carrier_slope/v7"
)
CARRIER_AFFINE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_carrier_affine/v8"
)
SPARSE_RANK_CHANNEL_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_sparse_rank_channel/v9"
)
SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_signed_rank_query_pool/v10"
)
CROSS_ATTENTION_ABSOLUTE_POOL_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_cross_attention_absolute_pool/v11"
)
CANDIDATE_ABSOLUTE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_candidate_absolute/v12"
)
CANDIDATE_CALIBRATED_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_candidate_calibrated/v13"
)
CANDIDATE_NORMALIZED_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_candidate_normalized/v14"
)
CANDIDATE_ASYMMETRIC_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_candidate_asymmetric/v15"
)
CANDIDATE_SET_ATTENTION_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_candidate_set_attention/v16"
)
GLOBAL_TRUST_VETO_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_global_trust_veto/v17"
)
DEPLOYED_ROUTER_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_deployed_router/v18"
)
CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_candidate_sample_calibrator/v19"
)
FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_fulltext_global_absolute/v20"
)
FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_"
    "fulltext_global_absolute_exact_residual/v21"
)
FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_"
    "fulltext_global_independent_absolute/v22"
)
DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_"
    "deployment_owned_global_absolute/v23"
)
DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_"
    "deployment_owned_query_global_absolute/v24"
)
DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_"
    "deployment_owned_query_veto_global_absolute/v25"
)
ABSOLUTE_CAP_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_patch_pool_trainable_absolute_cap_v1"
)
RANK_EVIDENCE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_patch_pool_trainable_absolute_cap_rank_evidence_scale_v2"
)
RANK_EVIDENCE_RESIDUAL_CONTRACT = "zero_init_rank_logit_scale_v1"
RANK_AFFINE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_patch_pool_trainable_absolute_cap_rank_affine_v3"
)
RANK_AFFINE_RESIDUAL_CONTRACT = "zero_init_rank_logit_affine_v2"
GATE_MARGIN_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_patch_pool_trainable_absolute_cap_gate_margin_v4"
)
GATE_MARGIN_RESIDUAL_CONTRACT = "zero_init_rank_logit_gate_margin_scale_v3"
CARRIER_SLOPE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_patch_pool_trainable_absolute_cap_carrier_token_rank_slope_v5"
)
CARRIER_SLOPE_RESIDUAL_CONTRACT = "zero_init_carrier_token_rank_slope_v4"
CARRIER_AFFINE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_patch_pool_trainable_absolute_cap_carrier_token_rank_affine_v6"
)
CARRIER_AFFINE_RESIDUAL_CONTRACT = "zero_init_carrier_token_rank_affine_v5"
SPARSE_RANK_CHANNEL_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_patch_pool_trainable_absolute_cap_sparse_rank_channel_v7"
)
SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT = (
    "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
)
SIGNED_RANK_QUERY_POOL_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_query_patch_signed_pool_absolute_cap_v8"
)
CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_query_modifier_cross_attention_absolute_pool_v9"
)
CANDIDATE_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_query_modifier_candidate_absolute_logits_v10"
)
CANDIDATE_CALIBRATED_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_candidate_absolute_patch_invariant_monotone_veto_v11"
)
CANDIDATE_NORMALIZED_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_candidate_normalized_patch_amplified_monotone_veto_v12"
)
CANDIDATE_ASYMMETRIC_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_raw_patch_asymmetric_monotone_veto_v13"
)
CANDIDATE_SET_ATTENTION_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_candidate_set_attention_asymmetric_veto_v14"
)
GLOBAL_TRUST_VETO_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_raw_patch_asymmetric_split_global_trust_veto_v15"
)
DEPLOYED_ROUTER_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_raw_patch_asymmetric_independent_deployed_router_v16"
)
CANDIDATE_SAMPLE_CALIBRATOR_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_raw_patch_asymmetric_candidate_sample_calibrator_v17"
)
FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_candidate_residual_global_absolute_v18"
)
FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_candidate_residual_"
    "global_absolute_exact_rank_max_residual_trust_v19"
)
FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_local_candidate_"
    "global_independent_absolute_v20"
)
DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_deployment_owned_"
    "global_absolute_v21"
)
DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_deployment_owned_monotone_"
    "query_global_absolute_v22"
)
DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_deployment_owned_bounded_"
    "query_veto_global_absolute_v23"
)
GLOBAL_TRUST_VETO_HEAD_GRADIENT_CONTRACT = (
    "split_token_veto_global_trust_veto_v4"
)
DEPLOYED_ROUTER_HEAD_GRADIENT_CONTRACT = (
    "split_token_veto_deployed_router_global_absolute_v5"
)
CANDIDATE_SAMPLE_CALIBRATOR_HEAD_GRADIENT_CONTRACT = (
    "split_token_veto_candidate_absolute_sample_calibrator_v6"
)
FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT = (
    "split_token_veto_fulltext_global_absolute_v7"
)
FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT = (
    "split_token_veto_local_candidate_global_absolute_v8"
)
DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT = (
    "split_token_veto_deployment_owned_global_absolute_v9"
)
DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT = (
    "split_token_veto_deployment_owned_query_global_absolute_v10"
)
DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT = (
    "split_token_veto_deployment_owned_query_veto_global_absolute_v11"
)
FULLTEXT_GLOBAL_ABSOLUTE_GATE_GRADIENT_CONTRACT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
TOKEN_LOGIT_CONTRACT = "detached_rank_token_minus_zero_init_residual_v1"
POOL_FEATURE_CONTRACT = "patch_statistics_only_v1"
SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACT = (
    "detached_rank_query_plus_patch_statistics_signed_residual_v2"
)
TOKEN_CONDITIONED_POOL_FEATURE_CONTRACT = (
    "detached_rank_query_token_context_plus_patch_statistics_monotone_v3"
)
CROSS_ATTENTION_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_rank_query_modifier_cross_attention_plus_patch_statistics_absolute_v4"
)
CANDIDATE_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_query_modifier_cross_attention_candidate_absolute_logits_v5"
)
CANDIDATE_CALIBRATED_POOL_FEATURE_CONTRACT = (
    "detached_candidate_absolute_patch_invariant_monotone_veto_logits_v6"
)
CANDIDATE_NORMALIZED_POOL_FEATURE_CONTRACT = (
    "detached_candidate_absolute_normalized_patch_amplified_veto_logits_v7"
)
CANDIDATE_ASYMMETRIC_POOL_FEATURE_CONTRACT = (
    "detached_candidate_absolute_raw_patch_asymmetric_veto_logits_v8"
)
CANDIDATE_SET_ATTENTION_POOL_FEATURE_CONTRACT = (
    "detached_candidate_set_attention_absolute_asymmetric_veto_logits_v9"
)
FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_candidate_residual_global_pool_v10"
)
FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_candidate_residual_"
    "global_pool_exact_rank_max_reference_v11"
)
FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_local_candidate_"
    "frozen_rank_global_pool_v12"
)
DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_deployment_owned_global_pool_v13"
)
DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_monotone_query_"
    "deployment_owned_global_pool_v14"
)
DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_token_conditioned_query_veto_"
    "deployment_owned_global_pool_v15"
)
SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACTS = frozenset(
    {
        SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACT,
        TOKEN_CONDITIONED_POOL_FEATURE_CONTRACT,
    }
)
EXPECTED_ADAPTER_TENSOR_COUNT = 22
EXPECTED_POOL_TENSOR_COUNT = 6
EXPECTED_FRESH_TENSOR_COUNT = 29
EXPECTED_FRESH_ELEMENT_COUNT = 185_925
EXPECTED_FRESH_STORAGE_BYTES = 743_704
EXPECTED_FRESH_SHA256 = (
    "3647c3e359a834d920c5bdca24f046bb3491ffa80ce070fe9af8bdfeda3291ec"
)
EXPECTED_STRICT_TARGET_TENSOR_COUNT = 1_617
EXPECTED_ABSOLUTE_CAP_ADAPTER_TENSOR_COUNT = 23
EXPECTED_ABSOLUTE_CAP_FRESH_TENSOR_COUNT = 30
EXPECTED_ABSOLUTE_CAP_FRESH_ELEMENT_COUNT = 185_926
EXPECTED_ABSOLUTE_CAP_FRESH_STORAGE_BYTES = 743_708
EXPECTED_ABSOLUTE_CAP_FRESH_SHA256 = (
    "bb5b8b048064b4510f87cbf0248fcdc4b07bf55ea7f85d21db8f98a400d533a7"
)
EXPECTED_ABSOLUTE_CAP_STRICT_TARGET_TENSOR_COUNT = 1_618
EXPECTED_RANK_EVIDENCE_ADAPTER_TENSOR_COUNT = 24
EXPECTED_RANK_EVIDENCE_FRESH_TENSOR_COUNT = 31
EXPECTED_RANK_EVIDENCE_FRESH_ELEMENT_COUNT = 185_927
EXPECTED_RANK_EVIDENCE_FRESH_STORAGE_BYTES = 743_712
EXPECTED_RANK_EVIDENCE_FRESH_SHA256 = (
    "1e8e997e6758a8c3036ed4b8c2a692fedb85781828d03073c2e7d7f8938b8eb8"
)
EXPECTED_RANK_EVIDENCE_STRICT_TARGET_TENSOR_COUNT = 1_619
EXPECTED_RANK_AFFINE_ADAPTER_TENSOR_COUNT = 25
EXPECTED_RANK_AFFINE_FRESH_TENSOR_COUNT = 32
EXPECTED_RANK_AFFINE_FRESH_ELEMENT_COUNT = 185_928
EXPECTED_RANK_AFFINE_FRESH_STORAGE_BYTES = 743_716
EXPECTED_RANK_AFFINE_FRESH_SHA256 = (
    "6002f80a673e214a9416d4348c3df050dc6df5e190aa6d69d41ff50ca000da51"
)
EXPECTED_RANK_AFFINE_STRICT_TARGET_TENSOR_COUNT = 1_620
EXPECTED_GATE_MARGIN_ADAPTER_TENSOR_COUNT = EXPECTED_RANK_EVIDENCE_ADAPTER_TENSOR_COUNT
EXPECTED_GATE_MARGIN_FRESH_TENSOR_COUNT = EXPECTED_RANK_EVIDENCE_FRESH_TENSOR_COUNT
EXPECTED_GATE_MARGIN_FRESH_ELEMENT_COUNT = EXPECTED_RANK_EVIDENCE_FRESH_ELEMENT_COUNT
EXPECTED_GATE_MARGIN_FRESH_STORAGE_BYTES = EXPECTED_RANK_EVIDENCE_FRESH_STORAGE_BYTES
EXPECTED_GATE_MARGIN_FRESH_SHA256 = EXPECTED_RANK_EVIDENCE_FRESH_SHA256
EXPECTED_GATE_MARGIN_STRICT_TARGET_TENSOR_COUNT = (
    EXPECTED_RANK_EVIDENCE_STRICT_TARGET_TENSOR_COUNT
)
EXPECTED_CARRIER_SLOPE_ADAPTER_TENSOR_COUNT = 24
EXPECTED_CARRIER_SLOPE_FRESH_TENSOR_COUNT = 31
EXPECTED_CARRIER_SLOPE_FRESH_ELEMENT_COUNT = 185_990
EXPECTED_CARRIER_SLOPE_FRESH_STORAGE_BYTES = 743_964
EXPECTED_CARRIER_SLOPE_FRESH_SHA256 = (
    "4557940593c3d24a04634497f27872c43e0b4760d9b11f97d1d33fc750162280"
)
EXPECTED_CARRIER_SLOPE_STRICT_TARGET_TENSOR_COUNT = 1_619
EXPECTED_CARRIER_AFFINE_ADAPTER_TENSOR_COUNT = 25
EXPECTED_CARRIER_AFFINE_FRESH_TENSOR_COUNT = 32
EXPECTED_CARRIER_AFFINE_FRESH_ELEMENT_COUNT = 185_991
EXPECTED_CARRIER_AFFINE_FRESH_STORAGE_BYTES = 743_968
EXPECTED_CARRIER_AFFINE_FRESH_SHA256 = (
    "cc4665411d020412c0bc3905fb10d634d0c02270537403bda556585a4a5c296c"
)
EXPECTED_CARRIER_AFFINE_STRICT_TARGET_TENSOR_COUNT = 1_620
EXPECTED_SPARSE_RANK_CHANNEL_ADAPTER_TENSOR_COUNT = 32
EXPECTED_SPARSE_RANK_CHANNEL_FRESH_TENSOR_COUNT = 39
EXPECTED_SPARSE_RANK_CHANNEL_FRESH_ELEMENT_COUNT = 203_143
EXPECTED_SPARSE_RANK_CHANNEL_FRESH_STORAGE_BYTES = 812_576
EXPECTED_SPARSE_RANK_CHANNEL_FRESH_SHA256 = (
    "986f11973cec7827a4f9cd0274bbf5742b7008cae1042a9b3538a28a4d9f90b1"
)
EXPECTED_SPARSE_RANK_CHANNEL_STRICT_TARGET_TENSOR_COUNT = 1_627
EXPECTED_SIGNED_RANK_QUERY_POOL_ADAPTER_TENSOR_COUNT = 38
EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_TENSOR_COUNT = 45
EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_ELEMENT_COUNT = 236_807
EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_STORAGE_BYTES = 947_232
EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_SHA256 = (
    "85f59029344f37e24fdac7beb4b9c0030aea1e7c3aa47e2f9c3c14ecbb933a22"
)
EXPECTED_SIGNED_RANK_QUERY_POOL_STRICT_TARGET_TENSOR_COUNT = 1_633
EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_ADAPTER_TENSOR_COUNT = 60
EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_TENSOR_COUNT = 67
EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_ELEMENT_COUNT = 469_383
EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_STORAGE_BYTES = 1_877_536
EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_SHA256 = (
    "a79afda2a17522f08129ef1e87ad172b3806a531a032eb1f59cdab51af841166"
)
EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_STRICT_TARGET_TENSOR_COUNT = 1_655
EXPECTED_CANDIDATE_ABSOLUTE_ADAPTER_TENSOR_COUNT = 66
EXPECTED_CANDIDATE_ABSOLUTE_FRESH_TENSOR_COUNT = 73
EXPECTED_CANDIDATE_ABSOLUTE_FRESH_ELEMENT_COUNT = 535_944
EXPECTED_CANDIDATE_ABSOLUTE_FRESH_STORAGE_BYTES = 2_143_780
EXPECTED_CANDIDATE_ABSOLUTE_FRESH_SHA256 = (
    "dc18ca03b622fc3e2dbd26bcceee8c04d0244a34076aee1ef717c7e66dae0900"
)
EXPECTED_CANDIDATE_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT = 1_661
EXPECTED_CANDIDATE_CALIBRATED_ADAPTER_TENSOR_COUNT = 69
EXPECTED_CANDIDATE_CALIBRATED_FRESH_TENSOR_COUNT = 76
EXPECTED_CANDIDATE_CALIBRATED_FRESH_ELEMENT_COUNT = 535_947
EXPECTED_CANDIDATE_CALIBRATED_FRESH_STORAGE_BYTES = 2_143_792
EXPECTED_CANDIDATE_CALIBRATED_FRESH_SHA256 = (
    "703baf1a3ea3605b0c4b094bb698c012a2ad509ed04caaa29afe8f8dc812a936"
)
EXPECTED_CANDIDATE_CALIBRATED_STRICT_TARGET_TENSOR_COUNT = 1_664
EXPECTED_CANDIDATE_NORMALIZED_ADAPTER_TENSOR_COUNT = 68
EXPECTED_CANDIDATE_NORMALIZED_FRESH_TENSOR_COUNT = 75
EXPECTED_CANDIDATE_NORMALIZED_FRESH_ELEMENT_COUNT = 535_946
EXPECTED_CANDIDATE_NORMALIZED_FRESH_STORAGE_BYTES = 2_143_788
EXPECTED_CANDIDATE_NORMALIZED_FRESH_SHA256 = (
    "c61139abf68f25b719213e69ee7e9a33e6201897a8944572ec9478b9e016f01a"
)
EXPECTED_CANDIDATE_NORMALIZED_STRICT_TARGET_TENSOR_COUNT = 1_663
EXPECTED_CANDIDATE_ASYMMETRIC_ADAPTER_TENSOR_COUNT = 68
EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_TENSOR_COUNT = 75
EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_ELEMENT_COUNT = 535_946
EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_STORAGE_BYTES = 2_143_788
EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_SHA256 = (
    "c61139abf68f25b719213e69ee7e9a33e6201897a8944572ec9478b9e016f01a"
)
EXPECTED_CANDIDATE_ASYMMETRIC_STRICT_TARGET_TENSOR_COUNT = 1_663
EXPECTED_CANDIDATE_SET_ATTENTION_ADAPTER_TENSOR_COUNT = 68
EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_TENSOR_COUNT = 94
EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_ELEMENT_COUNT = 1_329_034
EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_STORAGE_BYTES = 5_316_140
EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_SHA256 = (
    "fdca185ef64b2467127de7fc28df4bde2460f6861a44e8805a6f740b1927328a"
)
EXPECTED_CANDIDATE_SET_ATTENTION_STRICT_TARGET_TENSOR_COUNT = 1_682
EXPECTED_GLOBAL_TRUST_VETO_ADAPTER_TENSOR_COUNT = 68
EXPECTED_GLOBAL_TRUST_VETO_POOL_TENSOR_COUNT = 6
EXPECTED_GLOBAL_TRUST_VETO_VETO_POOL_TENSOR_COUNT = 6
EXPECTED_GLOBAL_TRUST_VETO_FRESH_TENSOR_COUNT = 81
EXPECTED_GLOBAL_TRUST_VETO_FRESH_ELEMENT_COUNT = 669_323
EXPECTED_GLOBAL_TRUST_VETO_FRESH_STORAGE_BYTES = 2_677_296
EXPECTED_GLOBAL_TRUST_VETO_FRESH_SHA256 = (
    "a502ae9d912404fcc62c172616482a7eeaa24132d51a2c37024e84001398d2ab"
)
EXPECTED_GLOBAL_TRUST_VETO_STRICT_TARGET_TENSOR_COUNT = 1_669
EXPECTED_DEPLOYED_ROUTER_ADAPTER_TENSOR_COUNT = 74
EXPECTED_DEPLOYED_ROUTER_FRESH_TENSOR_COUNT = 81
EXPECTED_DEPLOYED_ROUTER_FRESH_ELEMENT_COUNT = 536_735
EXPECTED_DEPLOYED_ROUTER_FRESH_STORAGE_BYTES = 2_146_944
EXPECTED_DEPLOYED_ROUTER_FRESH_SHA256 = (
    "2dac1bb68c9c6850381b00ad494cd7b53ea7e1ea4f1faa406bd034e72fe4039d"
)
EXPECTED_DEPLOYED_ROUTER_STRICT_TARGET_TENSOR_COUNT = 1_669
EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_ADAPTER_TENSOR_COUNT = (
    EXPECTED_CANDIDATE_ASYMMETRIC_ADAPTER_TENSOR_COUNT
)
EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_TENSOR_COUNT = (
    EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_TENSOR_COUNT
)
EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_ELEMENT_COUNT = (
    EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_ELEMENT_COUNT
)
EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_STORAGE_BYTES = (
    EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_STORAGE_BYTES
)
EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_SHA256 = (
    EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_SHA256
)
EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_STRICT_TARGET_TENSOR_COUNT = (
    EXPECTED_CANDIDATE_ASYMMETRIC_STRICT_TARGET_TENSOR_COUNT
)
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT = 59
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT = 6
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT = 65
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT = 534_725
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT = 66
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT = 534_726
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES = 2_138_908
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256 = (
    "7d4da05e70e4f2a15ca2c3265f82a36369c2c9732cdedace63ef24c4e97646b4"
)
EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT = 1_654
EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_TENSOR_COUNT = 59
EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_ELEMENT_COUNT = 468_164
EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_TENSOR_COUNT = 6
EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_ELEMENT_COUNT = 66_561
EXPECTED_RANK_TENSOR_COUNT = 453
EXPECTED_TRANSFERRED_TENSOR_COUNT = 1_588
SCORER_PREFIX = "stage_b_fixed_text_scorer."
RANK_PREFIX = SCORER_PREFIX + "rank_tower."
LEGACY_CONFIDENCE_PREFIX = SCORER_PREFIX + "confidence_tower."
LEGACY_POOL_PREFIX = SCORER_PREFIX + "confidence_pool."
ADAPTER_PREFIX = SCORER_PREFIX + "confidence_adapter."
POOL_PREFIX = SCORER_PREFIX + "confidence_pool."
VETO_POOL_PREFIX = SCORER_PREFIX + "confidence_veto_pool."
CONTRACT_KEY = SCORER_PREFIX + "_dense_duty_contract_version"
ABSOLUTE_CAP_PARAMETER_KEY = ADAPTER_PREFIX + "veto_cap_raw_ceiling"
RANK_EVIDENCE_PARAMETER_KEY = ADAPTER_PREFIX + "rank_evidence_residual_scale"
RANK_AFFINE_BIAS_PARAMETER_KEY = ADAPTER_PREFIX + "rank_evidence_residual_bias"
CARRIER_RANK_SLOPE_WEIGHT_KEY = ADAPTER_PREFIX + "carrier_rank_slope.weight"
CARRIER_RANK_SLOPE_BIAS_KEY = ADAPTER_PREFIX + "carrier_rank_slope.bias"
RANK_CHANNEL_NORM_WEIGHT_KEY = ADAPTER_PREFIX + "rank_channel_norm.weight"
RANK_CHANNEL_NORM_BIAS_KEY = ADAPTER_PREFIX + "rank_channel_norm.bias"
RANK_CHANNEL_PROJECTION_WEIGHT_KEY = (
    ADAPTER_PREFIX + "rank_channel_projection.weight"
)
RANK_CHANNEL_PROJECTION_BIAS_KEY = ADAPTER_PREFIX + "rank_channel_projection.bias"
RANK_CHANNEL_LOGIT_PROJECTION_WEIGHT_KEY = (
    ADAPTER_PREFIX + "rank_channel_logit_projection.weight"
)
RANK_CHANNEL_LOGIT_PROJECTION_BIAS_KEY = (
    ADAPTER_PREFIX + "rank_channel_logit_projection.bias"
)
RANK_CHANNEL_OUTPUT_WEIGHT_KEY = ADAPTER_PREFIX + "rank_channel_output.weight"
RANK_CHANNEL_PARAMETER_KEYS = frozenset(
    {
        RANK_CHANNEL_NORM_WEIGHT_KEY,
        RANK_CHANNEL_NORM_BIAS_KEY,
        RANK_CHANNEL_PROJECTION_WEIGHT_KEY,
        RANK_CHANNEL_PROJECTION_BIAS_KEY,
        RANK_CHANNEL_LOGIT_PROJECTION_WEIGHT_KEY,
        RANK_CHANNEL_LOGIT_PROJECTION_BIAS_KEY,
        RANK_CHANNEL_OUTPUT_WEIGHT_KEY,
    }
)
GLOBAL_QUERY_NORM_WEIGHT_KEY = ADAPTER_PREFIX + "global_query_norm.weight"
GLOBAL_QUERY_NORM_BIAS_KEY = ADAPTER_PREFIX + "global_query_norm.bias"
GLOBAL_QUERY_TRUNK_INPUT_WEIGHT_KEY = (
    ADAPTER_PREFIX + "global_query_trunk.0.weight"
)
GLOBAL_QUERY_TRUNK_INPUT_BIAS_KEY = ADAPTER_PREFIX + "global_query_trunk.0.bias"
GLOBAL_QUERY_TRUNK_OUTPUT_WEIGHT_KEY = (
    ADAPTER_PREFIX + "global_query_trunk.2.weight"
)
GLOBAL_QUERY_TRUNK_OUTPUT_BIAS_KEY = ADAPTER_PREFIX + "global_query_trunk.2.bias"
GLOBAL_QUERY_PARAMETER_KEYS = frozenset(
    {
        GLOBAL_QUERY_NORM_WEIGHT_KEY,
        GLOBAL_QUERY_NORM_BIAS_KEY,
        GLOBAL_QUERY_TRUNK_INPUT_WEIGHT_KEY,
        GLOBAL_QUERY_TRUNK_INPUT_BIAS_KEY,
        GLOBAL_QUERY_TRUNK_OUTPUT_WEIGHT_KEY,
        GLOBAL_QUERY_TRUNK_OUTPUT_BIAS_KEY,
    }
)
FULLTEXT_GLOBAL_QUERY_PARAMETER_KEYS = frozenset(
    {
        GLOBAL_QUERY_TRUNK_INPUT_WEIGHT_KEY,
        GLOBAL_QUERY_TRUNK_INPUT_BIAS_KEY,
        GLOBAL_QUERY_TRUNK_OUTPUT_WEIGHT_KEY,
        GLOBAL_QUERY_TRUNK_OUTPUT_BIAS_KEY,
    }
)
CROSS_ATTENTION_PARAMETER_KEYS = frozenset(
    {
        ADAPTER_PREFIX + "cross_query_norm.weight",
        ADAPTER_PREFIX + "cross_query_norm.bias",
        ADAPTER_PREFIX + "cross_text_norm.weight",
        ADAPTER_PREFIX + "cross_text_norm.bias",
        ADAPTER_PREFIX + "cross_query_projection.weight",
        ADAPTER_PREFIX + "cross_query_projection.bias",
        ADAPTER_PREFIX + "cross_text_projection.weight",
        ADAPTER_PREFIX + "cross_text_projection.bias",
        ADAPTER_PREFIX + "cross_evidence_projection.weight",
        ADAPTER_PREFIX + "cross_evidence_projection.bias",
        ADAPTER_PREFIX + "cross_attention.in_proj_weight",
        ADAPTER_PREFIX + "cross_attention.in_proj_bias",
        ADAPTER_PREFIX + "cross_attention.out_proj.weight",
        ADAPTER_PREFIX + "cross_attention.out_proj.bias",
        ADAPTER_PREFIX + "cross_ffn.0.weight",
        ADAPTER_PREFIX + "cross_ffn.0.bias",
        ADAPTER_PREFIX + "cross_ffn.1.weight",
        ADAPTER_PREFIX + "cross_ffn.1.bias",
        ADAPTER_PREFIX + "cross_ffn.3.weight",
        ADAPTER_PREFIX + "cross_ffn.3.bias",
        ADAPTER_PREFIX + "cross_output_projection.weight",
        ADAPTER_PREFIX + "cross_output_projection.bias",
    }
)
CANDIDATE_ABSOLUTE_PARAMETER_KEYS = frozenset(
    {
        ADAPTER_PREFIX + "candidate_absolute_head.0.weight",
        ADAPTER_PREFIX + "candidate_absolute_head.0.bias",
        ADAPTER_PREFIX + "candidate_absolute_head.1.weight",
        ADAPTER_PREFIX + "candidate_absolute_head.1.bias",
        ADAPTER_PREFIX + "candidate_absolute_head.3.weight",
        ADAPTER_PREFIX + "candidate_absolute_head.3.bias",
    }
)
CANDIDATE_CALIBRATION_PARAMETER_KEYS = frozenset(
    {
        ADAPTER_PREFIX + "candidate_patch_scale_raw",
        ADAPTER_PREFIX + "candidate_veto_depth_raw",
        ADAPTER_PREFIX + "candidate_coverage_depth_raw",
    }
)
CANDIDATE_NORMALIZED_CALIBRATION_PARAMETER_KEYS = frozenset(
    {
        ADAPTER_PREFIX + "candidate_veto_depth_raw",
        ADAPTER_PREFIX + "candidate_coverage_depth_raw",
    }
)
DEPLOYED_ROUTER_PARAMETER_KEYS = frozenset(
    {
        ADAPTER_PREFIX + "deployed_router_norm.weight",
        ADAPTER_PREFIX + "deployed_router_norm.bias",
        ADAPTER_PREFIX + "deployed_router_residual.0.weight",
        ADAPTER_PREFIX + "deployed_router_residual.0.bias",
        ADAPTER_PREFIX + "deployed_router_residual.2.weight",
        ADAPTER_PREFIX + "deployed_router_residual.2.bias",
    }
)


def _migration_surface_contract(audit: Mapping[str, Any]) -> dict[str, Any]:
    schema = audit.get("schema")
    fulltext_global_absolute_surface = (
        schema == FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
    ) or (
        schema == FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
    ) or (
        schema == FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
    ) or (
        schema == DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
    ) or (
        schema == DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
    ) or (
        schema == DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
    )
    if (
        fulltext_global_absolute_surface
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
        and audit.get("head_gradient_contract")
        == (
            DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            if schema
            == DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
            else
            DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            if schema == DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
            else
            DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            if schema == DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
            else
            FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            if schema == FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA
            else FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
        )
    ):
        surface = {
            "adapter_tensor_count": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT
            ),
            "pool_tensor_count": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT
            ),
            "confidence_parameter_tensor_count": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT
            ),
            "confidence_parameter_element_count": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT
            ),
            "fresh_tensor_count": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT
            ),
            "fresh_element_count": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT
            ),
            "fresh_storage_bytes": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES
            ),
            "fresh_sha256": EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT
            ),
        }
        if schema == DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA:
            surface.update(
                {
                    "active_confidence_parameter_tensor_count": (
                        EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_TENSOR_COUNT
                    ),
                    "active_confidence_parameter_element_count": (
                        EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_ELEMENT_COUNT
                    ),
                    "diagnostic_candidate_parameter_tensor_count": (
                        EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_TENSOR_COUNT
                    ),
                    "diagnostic_candidate_parameter_element_count": (
                        EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_ELEMENT_COUNT
                    ),
                }
            )
        elif schema in {
            DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
            DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA,
        }:
            surface.update(
                {
                    "active_confidence_parameter_tensor_count": (
                        EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT
                    ),
                    "active_confidence_parameter_element_count": (
                        EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT
                    ),
                    "deployed_query_parameter_tensor_count": (
                        EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_TENSOR_COUNT
                    ),
                    "deployed_query_parameter_element_count": (
                        EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_ELEMENT_COUNT
                    ),
                }
            )
        return surface
    if (
        schema == CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CANDIDATE_SAMPLE_CALIBRATOR_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
        and audit.get("head_gradient_contract")
        == CANDIDATE_SAMPLE_CALIBRATOR_HEAD_GRADIENT_CONTRACT
    ):
        return {
            "adapter_tensor_count": (
                EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_ADAPTER_TENSOR_COUNT
            ),
            "fresh_tensor_count": (
                EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_TENSOR_COUNT
            ),
            "fresh_element_count": (
                EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_ELEMENT_COUNT
            ),
            "fresh_storage_bytes": (
                EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_STORAGE_BYTES
            ),
            "fresh_sha256": EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == DEPLOYED_ROUTER_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == DEPLOYED_ROUTER_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
        and audit.get("head_gradient_contract")
        == DEPLOYED_ROUTER_HEAD_GRADIENT_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_DEPLOYED_ROUTER_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_DEPLOYED_ROUTER_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_DEPLOYED_ROUTER_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_DEPLOYED_ROUTER_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_DEPLOYED_ROUTER_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_DEPLOYED_ROUTER_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == GLOBAL_TRUST_VETO_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == GLOBAL_TRUST_VETO_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
        and audit.get("head_gradient_contract")
        == GLOBAL_TRUST_VETO_HEAD_GRADIENT_CONTRACT
    ):
        return {
            "adapter_tensor_count": (
                EXPECTED_GLOBAL_TRUST_VETO_ADAPTER_TENSOR_COUNT
            ),
            "fresh_tensor_count": EXPECTED_GLOBAL_TRUST_VETO_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_GLOBAL_TRUST_VETO_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": (
                EXPECTED_GLOBAL_TRUST_VETO_FRESH_STORAGE_BYTES
            ),
            "fresh_sha256": EXPECTED_GLOBAL_TRUST_VETO_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_GLOBAL_TRUST_VETO_STRICT_TARGET_TENSOR_COUNT
            ),
            "pool_tensor_count": EXPECTED_GLOBAL_TRUST_VETO_POOL_TENSOR_COUNT,
            "veto_pool_tensor_count": (
                EXPECTED_GLOBAL_TRUST_VETO_VETO_POOL_TENSOR_COUNT
            ),
        }
    if (
        schema == CANDIDATE_SET_ATTENTION_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CANDIDATE_SET_ATTENTION_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_CANDIDATE_SET_ATTENTION_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CANDIDATE_SET_ATTENTION_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == CANDIDATE_ASYMMETRIC_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CANDIDATE_ASYMMETRIC_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_CANDIDATE_ASYMMETRIC_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CANDIDATE_ASYMMETRIC_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == CANDIDATE_NORMALIZED_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CANDIDATE_NORMALIZED_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_CANDIDATE_NORMALIZED_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_CANDIDATE_NORMALIZED_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_CANDIDATE_NORMALIZED_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_CANDIDATE_NORMALIZED_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_CANDIDATE_NORMALIZED_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CANDIDATE_NORMALIZED_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == CANDIDATE_CALIBRATED_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CANDIDATE_CALIBRATED_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_CANDIDATE_CALIBRATED_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_CANDIDATE_CALIBRATED_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_CANDIDATE_CALIBRATED_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_CANDIDATE_CALIBRATED_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_CANDIDATE_CALIBRATED_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CANDIDATE_CALIBRATED_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == CANDIDATE_ABSOLUTE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CANDIDATE_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_CANDIDATE_ABSOLUTE_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_CANDIDATE_ABSOLUTE_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_CANDIDATE_ABSOLUTE_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_CANDIDATE_ABSOLUTE_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_CANDIDATE_ABSOLUTE_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CANDIDATE_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == CROSS_ATTENTION_ABSOLUTE_POOL_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": (
                EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_ADAPTER_TENSOR_COUNT
            ),
            "fresh_tensor_count": (
                EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_TENSOR_COUNT
            ),
            "fresh_element_count": (
                EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_ELEMENT_COUNT
            ),
            "fresh_storage_bytes": (
                EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_STORAGE_BYTES
            ),
            "fresh_sha256": EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == SIGNED_RANK_QUERY_POOL_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": (
                EXPECTED_SIGNED_RANK_QUERY_POOL_ADAPTER_TENSOR_COUNT
            ),
            "fresh_tensor_count": (
                EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_TENSOR_COUNT
            ),
            "fresh_element_count": (
                EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_ELEMENT_COUNT
            ),
            "fresh_storage_bytes": (
                EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_STORAGE_BYTES
            ),
            "fresh_sha256": EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_SIGNED_RANK_QUERY_POOL_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == SPARSE_RANK_CHANNEL_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == SPARSE_RANK_CHANNEL_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": (
                EXPECTED_SPARSE_RANK_CHANNEL_ADAPTER_TENSOR_COUNT
            ),
            "fresh_tensor_count": EXPECTED_SPARSE_RANK_CHANNEL_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_SPARSE_RANK_CHANNEL_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": (
                EXPECTED_SPARSE_RANK_CHANNEL_FRESH_STORAGE_BYTES
            ),
            "fresh_sha256": EXPECTED_SPARSE_RANK_CHANNEL_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_SPARSE_RANK_CHANNEL_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if schema == MIGRATION_SCHEMA:
        return {
            "adapter_tensor_count": EXPECTED_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_FRESH_SHA256,
            "strict_target_tensor_count": EXPECTED_STRICT_TARGET_TENSOR_COUNT,
        }
    if (
        schema == ABSOLUTE_CAP_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == ABSOLUTE_CAP_FRESH_CONFIDENCE_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_ABSOLUTE_CAP_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_ABSOLUTE_CAP_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_ABSOLUTE_CAP_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_ABSOLUTE_CAP_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_ABSOLUTE_CAP_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_ABSOLUTE_CAP_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == RANK_EVIDENCE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == RANK_EVIDENCE_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == RANK_EVIDENCE_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_RANK_EVIDENCE_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_RANK_EVIDENCE_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_RANK_EVIDENCE_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_RANK_EVIDENCE_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_RANK_EVIDENCE_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_RANK_EVIDENCE_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == GATE_MARGIN_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == GATE_MARGIN_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == GATE_MARGIN_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_GATE_MARGIN_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_GATE_MARGIN_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_GATE_MARGIN_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_GATE_MARGIN_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_GATE_MARGIN_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_GATE_MARGIN_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == CARRIER_SLOPE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CARRIER_SLOPE_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == CARRIER_SLOPE_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_CARRIER_SLOPE_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_CARRIER_SLOPE_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_CARRIER_SLOPE_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_CARRIER_SLOPE_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_CARRIER_SLOPE_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CARRIER_SLOPE_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == CARRIER_AFFINE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == CARRIER_AFFINE_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == CARRIER_AFFINE_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_CARRIER_AFFINE_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_CARRIER_AFFINE_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_CARRIER_AFFINE_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_CARRIER_AFFINE_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_CARRIER_AFFINE_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_CARRIER_AFFINE_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    if (
        schema == RANK_AFFINE_MIGRATION_SCHEMA
        and audit.get("fresh_confidence_contract")
        == RANK_AFFINE_FRESH_CONFIDENCE_CONTRACT
        and audit.get("rank_evidence_contract")
        == RANK_AFFINE_RESIDUAL_CONTRACT
    ):
        return {
            "adapter_tensor_count": EXPECTED_RANK_AFFINE_ADAPTER_TENSOR_COUNT,
            "fresh_tensor_count": EXPECTED_RANK_AFFINE_FRESH_TENSOR_COUNT,
            "fresh_element_count": EXPECTED_RANK_AFFINE_FRESH_ELEMENT_COUNT,
            "fresh_storage_bytes": EXPECTED_RANK_AFFINE_FRESH_STORAGE_BYTES,
            "fresh_sha256": EXPECTED_RANK_AFFINE_FRESH_SHA256,
            "strict_target_tensor_count": (
                EXPECTED_RANK_AFFINE_STRICT_TARGET_TENSOR_COUNT
            ),
        }
    raise RuntimeError("confidence-adapter migration audit is invalid")


def validate_confidence_adapter_migration_audit(
    value: Any,
    *,
    source_checkpoint_sha256: str,
    source_optimizer_updates: int,
    source_checkpoint_reason: str,
    rank_sha256: str,
    transferred_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("confidence-adapter checkpoint lacks its migration audit")
    audit = dict(value)
    surface = _migration_surface_contract(audit)
    signed_pool_schema = (
        audit.get("schema") == SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA
    )
    cross_attention_pool_schema = (
        audit.get("schema") == CROSS_ATTENTION_ABSOLUTE_POOL_MIGRATION_SCHEMA
    )
    candidate_absolute_schema = (
        audit.get("schema") == CANDIDATE_ABSOLUTE_MIGRATION_SCHEMA
    )
    candidate_calibrated_schema = (
        audit.get("schema") == CANDIDATE_CALIBRATED_MIGRATION_SCHEMA
    )
    candidate_normalized_schema = (
        audit.get("schema") == CANDIDATE_NORMALIZED_MIGRATION_SCHEMA
    )
    candidate_asymmetric_schema = (
        audit.get("schema") == CANDIDATE_ASYMMETRIC_MIGRATION_SCHEMA
    )
    candidate_set_attention_schema = (
        audit.get("schema") == CANDIDATE_SET_ATTENTION_MIGRATION_SCHEMA
    )
    global_trust_veto_schema = (
        audit.get("schema") == GLOBAL_TRUST_VETO_MIGRATION_SCHEMA
    )
    deployed_router_schema = (
        audit.get("schema") == DEPLOYED_ROUTER_MIGRATION_SCHEMA
    )
    candidate_sample_calibrator_schema = (
        audit.get("schema") == CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA
    )
    fulltext_global_absolute_schema = (
        audit.get("schema") == FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
    )
    fulltext_global_absolute_exact_residual_schema = (
        audit.get("schema")
        == FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
    )
    fulltext_global_independent_absolute_schema = (
        audit.get("schema")
        == FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA
    )
    deployment_owned_global_absolute_schema = (
        audit.get("schema")
        == DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
    )
    deployment_owned_query_global_absolute_schema = (
        audit.get("schema")
        == DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
    )
    deployment_owned_query_veto_global_absolute_schema = (
        audit.get("schema")
        == DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
    )
    fulltext_global_absolute_family_schema = (
        fulltext_global_absolute_schema
        or fulltext_global_absolute_exact_residual_schema
        or fulltext_global_independent_absolute_schema
        or deployment_owned_global_absolute_schema
        or deployment_owned_query_global_absolute_schema
        or deployment_owned_query_veto_global_absolute_schema
    )
    pool_feature_contract = audit.get("pool_feature_contract")
    rank = audit.get("rank")
    transferred = audit.get("transferred")
    fresh = audit.get("fresh_confidence")
    if (
        audit.get("token_logit_contract") != TOKEN_LOGIT_CONTRACT
        or (
            (
                pool_feature_contract
                != FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
            )
            if fulltext_global_absolute_schema
            else (
                pool_feature_contract
                != FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
            )
            if fulltext_global_absolute_exact_residual_schema
            else (
                pool_feature_contract
                != FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT
            )
            if fulltext_global_independent_absolute_schema
            else (
                pool_feature_contract
                != DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
            )
            if deployment_owned_query_veto_global_absolute_schema
            else (
                pool_feature_contract
                != DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
            )
            if deployment_owned_query_global_absolute_schema
            else (
                pool_feature_contract
                != DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
            )
            if deployment_owned_global_absolute_schema
            else (
                pool_feature_contract
                != CANDIDATE_SET_ATTENTION_POOL_FEATURE_CONTRACT
            )
            if candidate_set_attention_schema
            else (
                pool_feature_contract
                != CANDIDATE_ASYMMETRIC_POOL_FEATURE_CONTRACT
            )
            if (
                candidate_asymmetric_schema
                or global_trust_veto_schema
                or deployed_router_schema
                or candidate_sample_calibrator_schema
            )
            else (
                pool_feature_contract
                != CANDIDATE_NORMALIZED_POOL_FEATURE_CONTRACT
            )
            if candidate_normalized_schema
            else (
                pool_feature_contract
                != CANDIDATE_CALIBRATED_POOL_FEATURE_CONTRACT
            )
            if candidate_calibrated_schema
            else (
                pool_feature_contract
                != CANDIDATE_ABSOLUTE_POOL_FEATURE_CONTRACT
            )
            if candidate_absolute_schema
            else (
                pool_feature_contract
                != CROSS_ATTENTION_ABSOLUTE_POOL_FEATURE_CONTRACT
            )
            if cross_attention_pool_schema
            else (
                pool_feature_contract
                not in SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACTS
                if signed_pool_schema
                else pool_feature_contract != POOL_FEATURE_CONTRACT
            )
        )
        or audit.get("source_checkpoint_sha256") != source_checkpoint_sha256
        or audit.get("source_optimizer_updates") != int(source_optimizer_updates)
        or audit.get("source_checkpoint_reason") != source_checkpoint_reason
        or not isinstance(rank, Mapping)
        or rank.get("sha256") != rank_sha256
        or rank.get("tensor_count") != EXPECTED_RANK_TENSOR_COUNT
        or rank.get("nonfinite_count") != 0
        or not isinstance(transferred, Mapping)
        or transferred.get("sha256") != transferred_sha256
        or transferred.get("tensor_count") != EXPECTED_TRANSFERRED_TENSOR_COUNT
        or transferred.get("nonfinite_count") != 0
        or not isinstance(fresh, Mapping)
        or fresh.get("sha256") != surface["fresh_sha256"]
        or fresh.get("tensor_count") != surface["fresh_tensor_count"]
        or fresh.get("element_count") != surface["fresh_element_count"]
        or fresh.get("storage_bytes") != surface["fresh_storage_bytes"]
        or fresh.get("nonfinite_count") != 0
        or audit.get("retired_confidence_tower_tensor_count") != 453
        or audit.get("retired_confidence_pool_tensor_count") != 6
        or audit.get("retired_confidence_loaded_tensor_count") != 0
        or audit.get("adapter_tensor_count") != surface["adapter_tensor_count"]
        or audit.get("pool_tensor_count")
        != surface.get(
            "pool_tensor_count",
            25 if candidate_set_attention_schema else EXPECTED_POOL_TENSOR_COUNT,
        )
        or audit.get("veto_pool_tensor_count", 0)
        != surface.get("veto_pool_tensor_count", 0)
        or (
            "confidence_parameter_tensor_count" in surface
            and audit.get("confidence_parameter_tensor_count")
            != surface["confidence_parameter_tensor_count"]
        )
        or (
            "confidence_parameter_element_count" in surface
            and audit.get("confidence_parameter_element_count")
            != surface["confidence_parameter_element_count"]
        )
        or any(
            audit.get(field) != expected
            for field, expected in surface.items()
            if field.startswith(
                (
                    "active_confidence_parameter_",
                    "diagnostic_candidate_parameter_",
                    "deployed_query_parameter_",
                )
            )
        )
        or (
            global_trust_veto_schema
            and audit.get("head_gradient_contract")
            != GLOBAL_TRUST_VETO_HEAD_GRADIENT_CONTRACT
        )
        or (
            deployed_router_schema
            and audit.get("head_gradient_contract")
            != DEPLOYED_ROUTER_HEAD_GRADIENT_CONTRACT
        )
        or (
            candidate_sample_calibrator_schema
            and audit.get("head_gradient_contract")
            != CANDIDATE_SAMPLE_CALIBRATOR_HEAD_GRADIENT_CONTRACT
        )
        or (
            fulltext_global_independent_absolute_schema
            and audit.get("head_gradient_contract")
            != FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT
        )
        or (
            deployment_owned_query_veto_global_absolute_schema
            and audit.get("head_gradient_contract")
            != DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
        )
        or (
            deployment_owned_query_global_absolute_schema
            and audit.get("head_gradient_contract")
            != DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
        )
        or (
            (
                deployment_owned_query_global_absolute_schema
                or deployment_owned_query_veto_global_absolute_schema
            )
            and audit.get("deployed_query_requires_grad_count")
            != EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_TENSOR_COUNT
        )
        or (
            deployment_owned_global_absolute_schema
            and audit.get("head_gradient_contract")
            != DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
        )
        or (
            deployment_owned_global_absolute_schema
            and audit.get("diagnostic_candidate_requires_grad_count") != 0
        )
        or (
            (
                fulltext_global_absolute_schema
                or fulltext_global_absolute_exact_residual_schema
            )
            and audit.get("head_gradient_contract")
            != FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
        )
        or audit.get("strict_target_tensor_count")
        != surface["strict_target_tensor_count"]
    ):
        raise RuntimeError("confidence-adapter migration audit is invalid")
    return audit


def _compatible_tensor(source: Any, target: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(source):
        raise RuntimeError(f"rank-adapter migration source is not a tensor: {name}")
    if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
        raise RuntimeError(
            "rank-adapter migration tensor contract drifted: "
            f"{name}, source=({tuple(source.shape)}, {source.dtype}), "
            f"target=({tuple(target.shape)}, {target.dtype})"
        )
    return source


def migrate_legacy_rank_to_confidence_adapter(
    model: nn.Module,
    source_state: Mapping[str, Any],
    *,
    checkpoint_label: str,
    source_checkpoint_sha256: str,
    source_optimizer_updates: int,
    source_checkpoint_reason: str,
    expected_rank_sha256: str,
    expected_transferred_sha256: str,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Build an exact target state while retiring every legacy confidence tensor."""
    if not isinstance(source_state, Mapping):
        raise TypeError(f"{checkpoint_label}: model state must be a mapping")
    root = model.module if hasattr(model, "module") else model
    scorer = getattr(root, "stage_b_fixed_text_scorer", None)
    if scorer is None or not hasattr(scorer, "confidence_adapter"):
        raise RuntimeError("rank-adapter migration requires the v2 dense-duty scorer")
    runtime = root.state_dict()
    source = {str(name): value for name, value in source_state.items()}

    rank_names = sorted(name for name in runtime if name.startswith(RANK_PREFIX))
    legacy_confidence_names = sorted(
        name for name in source if name.startswith(LEGACY_CONFIDENCE_PREFIX)
    )
    legacy_pool_names = sorted(name for name in source if name.startswith(LEGACY_POOL_PREFIX))
    adapter_names = sorted(name for name in runtime if name.startswith(ADAPTER_PREFIX))
    pool_names = sorted(name for name in runtime if name.startswith(POOL_PREFIX))
    veto_pool_names = sorted(
        name for name in runtime if name.startswith(VETO_POOL_PREFIX)
    )
    runtime_parameters = dict(root.named_parameters())
    confidence_parameter_names = sorted(
        name
        for name in runtime_parameters
        if name.startswith((ADAPTER_PREFIX, POOL_PREFIX, VETO_POOL_PREFIX))
    )
    confidence_parameter_element_count = sum(
        int(runtime_parameters[name].numel()) for name in confidence_parameter_names
    )
    active_confidence_ids = (
        {id(parameter) for parameter in scorer.confidence_parameters()}
        if hasattr(scorer, "confidence_parameters")
        else {id(runtime_parameters[name]) for name in confidence_parameter_names}
    )
    active_confidence_parameter_names = sorted(
        name
        for name in confidence_parameter_names
        if id(runtime_parameters[name]) in active_confidence_ids
    )
    diagnostic_candidate_ids = (
        {
            id(parameter)
            for parameter in scorer.candidate_diagnostic_parameters()
        }
        if hasattr(scorer, "candidate_diagnostic_parameters")
        else set()
    )
    diagnostic_candidate_parameter_names = sorted(
        name
        for name in confidence_parameter_names
        if id(runtime_parameters[name]) in diagnostic_candidate_ids
    )
    if not rank_names or not adapter_names or not pool_names:
        raise RuntimeError("rank-adapter migration found an incomplete target scorer")
    if any(name.startswith(ADAPTER_PREFIX) for name in source):
        raise RuntimeError("rank source unexpectedly already contains a confidence adapter")
    if not legacy_confidence_names or not legacy_pool_names:
        raise RuntimeError("rank source lacks the declared legacy confidence surface")

    provided_rank_names = sorted(name for name in source if name.startswith(RANK_PREFIX))
    if provided_rank_names != rank_names:
        missing = sorted(set(rank_names).difference(provided_rank_names))
        unexpected = sorted(set(provided_rank_names).difference(rank_names))
        raise RuntimeError(
            f"rank-adapter migration rank key drift: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    expected_legacy_suffixes = {
        name[len(RANK_PREFIX) :] for name in rank_names
    }
    observed_legacy_suffixes = {
        name[len(LEGACY_CONFIDENCE_PREFIX) :] for name in legacy_confidence_names
    }
    if observed_legacy_suffixes != expected_legacy_suffixes:
        raise RuntimeError("legacy confidence tower does not mirror the rank tower schema")

    allowed_source_scorer = set(rank_names)
    allowed_source_scorer.update(legacy_confidence_names)
    allowed_source_scorer.update(legacy_pool_names)
    allowed_source_scorer.add(CONTRACT_KEY)
    unexpected_scorer = sorted(
        name
        for name in source
        if name.startswith(SCORER_PREFIX) and name not in allowed_source_scorer
    )
    if unexpected_scorer:
        raise RuntimeError(
            "rank source has undeclared scorer tensors: " f"{unexpected_scorer[:8]}"
        )
    old_contract = source.get(CONTRACT_KEY)
    if (
        not torch.is_tensor(old_contract)
        or old_contract.numel() != 1
        or int(old_contract.item()) != 1
    ):
        raise RuntimeError("rank source does not carry the legacy v1 scorer contract")

    transferred_names = sorted(
        name
        for name in runtime
        if not name.startswith(SCORER_PREFIX) or name.startswith(RANK_PREFIX)
    )
    migrated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, target in runtime.items():
        if name in transferred_names:
            value = _compatible_tensor(source.get(name), target, name=name)
            migrated[name] = value.detach().clone()
        else:
            migrated[name] = target.detach().clone()

    rank_fingerprint = fingerprint_named_tensors(source, rank_names)
    if rank_fingerprint["sha256"] != expected_rank_sha256:
        raise RuntimeError(
            "rank source fingerprint drifted: "
            f"expected={expected_rank_sha256}, observed={rank_fingerprint['sha256']}"
        )
    transferred_fingerprint = fingerprint_named_tensors(source, transferred_names)
    if transferred_fingerprint["sha256"] != expected_transferred_sha256:
        raise RuntimeError(
            "transferred Stage-A/rank fingerprint drifted: "
            f"expected={expected_transferred_sha256}, "
            f"observed={transferred_fingerprint['sha256']}"
        )
    migrated_rank = fingerprint_named_tensors(migrated, rank_names)
    if migrated_rank != rank_fingerprint:
        raise RuntimeError("rank tensors changed during confidence-adapter migration")

    fresh_names = sorted(
        (*adapter_names, *pool_names, *veto_pool_names, CONTRACT_KEY)
    )
    fresh_fingerprint = fingerprint_named_tensors(migrated, fresh_names)
    absolute_cap_surface = ABSOLUTE_CAP_PARAMETER_KEY in adapter_names
    rank_evidence_surface = RANK_EVIDENCE_PARAMETER_KEY in adapter_names
    rank_affine_surface = RANK_AFFINE_BIAS_PARAMETER_KEY in adapter_names
    carrier_slope_weight_surface = CARRIER_RANK_SLOPE_WEIGHT_KEY in adapter_names
    carrier_slope_bias_surface = CARRIER_RANK_SLOPE_BIAS_KEY in adapter_names
    rank_channel_parameter_names = set(adapter_names).intersection(
        RANK_CHANNEL_PARAMETER_KEYS
    )
    global_query_parameter_names = set(adapter_names).intersection(
        GLOBAL_QUERY_PARAMETER_KEYS
    )
    cross_attention_parameter_names = set(adapter_names).intersection(
        CROSS_ATTENTION_PARAMETER_KEYS
    )
    candidate_absolute_parameter_names = set(adapter_names).intersection(
        CANDIDATE_ABSOLUTE_PARAMETER_KEYS
    )
    candidate_calibration_parameter_names = set(adapter_names).intersection(
        CANDIDATE_CALIBRATION_PARAMETER_KEYS
    )
    deployed_router_parameter_names = set(adapter_names).intersection(
        DEPLOYED_ROUTER_PARAMETER_KEYS
    )
    rank_channel_surface = bool(rank_channel_parameter_names)
    carrier_surface = carrier_slope_weight_surface
    runtime_rank_evidence_contract = str(
        getattr(scorer.confidence_adapter, "rank_evidence_contract", "") or ""
    ).strip().lower()
    gate_margin_surface = (
        runtime_rank_evidence_contract == GATE_MARGIN_RESIDUAL_CONTRACT
    )
    carrier_slope_contract = (
        runtime_rank_evidence_contract == CARRIER_SLOPE_RESIDUAL_CONTRACT
    )
    carrier_affine_contract = (
        runtime_rank_evidence_contract == CARRIER_AFFINE_RESIDUAL_CONTRACT
    )
    rank_channel_contract = (
        runtime_rank_evidence_contract == SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    )
    runtime_pool_feature_contract = str(
        getattr(scorer.confidence_adapter, "pool_feature_contract", "") or ""
    ).strip().lower()
    signed_rank_query_pool_contract = (
        runtime_pool_feature_contract
        in SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACTS
    )
    cross_attention_pool_contract = (
        runtime_pool_feature_contract
        == CROSS_ATTENTION_ABSOLUTE_POOL_FEATURE_CONTRACT
    )
    candidate_absolute_contract = (
        runtime_pool_feature_contract
        == CANDIDATE_ABSOLUTE_POOL_FEATURE_CONTRACT
    )
    candidate_calibrated_contract = (
        runtime_pool_feature_contract
        == CANDIDATE_CALIBRATED_POOL_FEATURE_CONTRACT
    )
    candidate_normalized_contract = (
        runtime_pool_feature_contract
        == CANDIDATE_NORMALIZED_POOL_FEATURE_CONTRACT
    )
    candidate_asymmetric_contract = (
        runtime_pool_feature_contract
        == CANDIDATE_ASYMMETRIC_POOL_FEATURE_CONTRACT
    )
    candidate_set_attention_contract = (
        runtime_pool_feature_contract
        == CANDIDATE_SET_ATTENTION_POOL_FEATURE_CONTRACT
    )
    fulltext_global_absolute_pool_contract = (
        runtime_pool_feature_contract
        == FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
    )
    fulltext_global_absolute_exact_residual_pool_contract = (
        runtime_pool_feature_contract
        == FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
    )
    fulltext_global_independent_absolute_pool_contract = (
        runtime_pool_feature_contract
        == FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT
    )
    deployment_owned_global_absolute_pool_contract = (
        runtime_pool_feature_contract
        == DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
    )
    deployment_owned_query_global_absolute_pool_contract = (
        runtime_pool_feature_contract
        == DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
    )
    deployment_owned_query_veto_global_absolute_pool_contract = (
        runtime_pool_feature_contract
        == DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
    )
    fulltext_global_absolute_family_pool_contract = (
        fulltext_global_absolute_pool_contract
        or fulltext_global_absolute_exact_residual_pool_contract
        or fulltext_global_independent_absolute_pool_contract
        or deployment_owned_global_absolute_pool_contract
        or deployment_owned_query_global_absolute_pool_contract
        or deployment_owned_query_veto_global_absolute_pool_contract
    )
    runtime_head_gradient_contract = str(
        getattr(scorer.confidence_adapter, "head_gradient_contract", "") or ""
    ).strip().lower()
    global_trust_veto_contract = (
        runtime_head_gradient_contract
        == GLOBAL_TRUST_VETO_HEAD_GRADIENT_CONTRACT
    )
    deployed_router_contract = (
        runtime_head_gradient_contract
        == DEPLOYED_ROUTER_HEAD_GRADIENT_CONTRACT
    )
    candidate_sample_calibrator_contract = (
        runtime_head_gradient_contract
        == CANDIDATE_SAMPLE_CALIBRATOR_HEAD_GRADIENT_CONTRACT
    )
    fulltext_global_absolute_head_contract = (
        runtime_head_gradient_contract
        == FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
    )
    fulltext_global_independent_absolute_head_contract = (
        runtime_head_gradient_contract
        == FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT
    )
    deployment_owned_global_absolute_head_contract = (
        runtime_head_gradient_contract
        == DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
    )
    deployment_owned_query_global_absolute_head_contract = (
        runtime_head_gradient_contract
        == DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
    )
    deployment_owned_query_veto_global_absolute_head_contract = (
        runtime_head_gradient_contract
        == DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
    )
    fulltext_global_absolute_contract = (
        fulltext_global_absolute_family_pool_contract
        and (
            fulltext_global_absolute_head_contract
            or fulltext_global_independent_absolute_head_contract
            or deployment_owned_global_absolute_head_contract
            or deployment_owned_query_global_absolute_head_contract
            or deployment_owned_query_veto_global_absolute_head_contract
        )
    )
    if (
        fulltext_global_absolute_family_pool_contract
        != (
            fulltext_global_absolute_head_contract
            or fulltext_global_independent_absolute_head_contract
            or deployment_owned_global_absolute_head_contract
            or deployment_owned_query_global_absolute_head_contract
            or deployment_owned_query_veto_global_absolute_head_contract
        )
    ):
        raise RuntimeError(
            "full-text global-absolute pool and head contracts must be selected "
            "together"
        )
    if (
        fulltext_global_independent_absolute_pool_contract
        != fulltext_global_independent_absolute_head_contract
    ):
        raise RuntimeError(
            "independent full-text global-absolute pool and head contracts must "
            "be selected together"
        )
    if (
        deployment_owned_global_absolute_pool_contract
        != deployment_owned_global_absolute_head_contract
    ):
        raise RuntimeError(
            "deployment-owned global-absolute pool and head contracts must be "
            "selected together"
        )
    if (
        deployment_owned_query_global_absolute_pool_contract
        != deployment_owned_query_global_absolute_head_contract
    ):
        raise RuntimeError(
            "deployment-owned query-global pool and head contracts must be "
            "selected together"
        )
    if (
        deployment_owned_query_veto_global_absolute_pool_contract
        != deployment_owned_query_veto_global_absolute_head_contract
    ):
        raise RuntimeError(
            "deployment-owned query-veto pool and head contracts must be "
            "selected together"
        )
    if fulltext_global_absolute_contract:
        runtime_gate_gradient_contract = str(
            getattr(scorer.confidence_adapter, "gate_gradient_contract", "") or ""
        ).strip().lower()
        if (
            runtime_gate_gradient_contract
            != FULLTEXT_GLOBAL_ABSOLUTE_GATE_GRADIENT_CONTRACT
        ):
            raise RuntimeError(
                "full-text global-absolute migration requires the asymmetric "
                "candidate gate contract"
            )
    if candidate_sample_calibrator_contract and not candidate_asymmetric_contract:
        raise RuntimeError(
            "candidate/sample calibrator migration requires the asymmetric "
            "candidate confidence surface"
        )
    cross_attention_feature_contract = (
        cross_attention_pool_contract
        or candidate_absolute_contract
        or candidate_calibrated_contract
        or candidate_normalized_contract
        or candidate_asymmetric_contract
        or candidate_set_attention_contract
        or fulltext_global_absolute_contract
    )
    if rank_evidence_surface and not absolute_cap_surface:
        raise RuntimeError(
            "rank-evidence migration requires the production absolute-cap surface"
        )
    if rank_affine_surface and not rank_evidence_surface:
        raise RuntimeError(
            "rank-affine migration requires its rank-evidence scale"
        )
    if gate_margin_surface and (
        not rank_evidence_surface or rank_affine_surface or not absolute_cap_surface
    ):
        raise RuntimeError(
            "gate-margin migration requires exactly the rank scale and absolute-cap "
            "surfaces"
        )
    if carrier_slope_bias_surface != (
        carrier_affine_contract or rank_channel_contract
    ):
        raise RuntimeError(
            "carrier rank migration bias surface disagrees with its contract"
        )
    if carrier_surface != (
        carrier_slope_contract or carrier_affine_contract or rank_channel_contract
    ):
        raise RuntimeError(
            "carrier rank runtime contract and parameter surface disagree"
        )
    if carrier_surface:
        carrier_weight = runtime[CARRIER_RANK_SLOPE_WEIGHT_KEY]
        carrier_bias = (
            runtime[CARRIER_RANK_SLOPE_BIAS_KEY]
            if carrier_affine_contract or rank_channel_contract
            else None
        )
        if (
                (not absolute_cap_surface and not fulltext_global_absolute_contract)
            or rank_evidence_surface
            or rank_affine_surface
            or tuple(carrier_weight.shape) != (1, 64)
            or carrier_weight.dtype != torch.float32
            or bool(torch.count_nonzero(carrier_weight).item())
            or (
                carrier_bias is not None
                and (
                    tuple(carrier_bias.shape) != (1,)
                    or carrier_bias.dtype != torch.float32
                    or bool(torch.count_nonzero(carrier_bias).item())
                )
            )
        ):
            raise RuntimeError(
                "carrier rank migration requires exactly a zero-initialized "
                "Linear(64, 1), the declared global constraint surface, and no "
                "scalar rank scale/bias"
            )
    if rank_channel_surface != rank_channel_contract:
        raise RuntimeError(
            "sparse rank-channel runtime contract and parameter surface disagree"
        )
    if rank_channel_contract:
        if rank_channel_parameter_names != set(RANK_CHANNEL_PARAMETER_KEYS):
            raise RuntimeError("sparse rank-channel parameter surface is incomplete")
        expected_shapes = {
            RANK_CHANNEL_NORM_WEIGHT_KEY: (256,),
            RANK_CHANNEL_NORM_BIAS_KEY: (256,),
            RANK_CHANNEL_PROJECTION_WEIGHT_KEY: (64, 256),
            RANK_CHANNEL_PROJECTION_BIAS_KEY: (64,),
            RANK_CHANNEL_LOGIT_PROJECTION_WEIGHT_KEY: (64, 1),
            RANK_CHANNEL_LOGIT_PROJECTION_BIAS_KEY: (64,),
            RANK_CHANNEL_OUTPUT_WEIGHT_KEY: (1, 64),
        }
        for name, shape in expected_shapes.items():
            value = runtime[name]
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise RuntimeError(
                    f"sparse rank-channel tensor contract drifted: {name}"
                )
        if bool(torch.count_nonzero(runtime[RANK_CHANNEL_OUTPUT_WEIGHT_KEY]).item()):
            raise RuntimeError(
                "sparse rank-channel output must be exactly zero initialized"
            )
    query_pool_contract = (
        signed_rank_query_pool_contract or cross_attention_feature_contract
    )
    if bool(global_query_parameter_names) != query_pool_contract:
        raise RuntimeError(
            "query-conditioned pool runtime contract and parameter surface disagree"
        )
    if query_pool_contract:
        expected_global_query_keys = (
            FULLTEXT_GLOBAL_QUERY_PARAMETER_KEYS
            if fulltext_global_absolute_contract
            else GLOBAL_QUERY_PARAMETER_KEYS
        )
        if global_query_parameter_names != set(expected_global_query_keys):
            raise RuntimeError("query-conditioned pool parameter surface is incomplete")
        expected_shapes = {
            GLOBAL_QUERY_NORM_WEIGHT_KEY: (256,),
            GLOBAL_QUERY_NORM_BIAS_KEY: (256,),
            GLOBAL_QUERY_TRUNK_INPUT_WEIGHT_KEY: (64, 257),
            GLOBAL_QUERY_TRUNK_INPUT_BIAS_KEY: (64,),
            GLOBAL_QUERY_TRUNK_OUTPUT_WEIGHT_KEY: (256, 64),
            GLOBAL_QUERY_TRUNK_OUTPUT_BIAS_KEY: (256,),
        }
        if fulltext_global_absolute_contract:
            expected_shapes.pop(GLOBAL_QUERY_NORM_WEIGHT_KEY)
            expected_shapes.pop(GLOBAL_QUERY_NORM_BIAS_KEY)
        for name, shape in expected_shapes.items():
            value = runtime[name]
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise RuntimeError(
                    f"query-conditioned pool tensor contract drifted: {name}"
                )
        if not rank_channel_contract or (
            not absolute_cap_surface and not fulltext_global_absolute_contract
        ):
            raise RuntimeError(
                "query-conditioned pool requires sparse token evidence and the "
                "absolute-cap surface"
            )
    if bool(cross_attention_parameter_names) != cross_attention_feature_contract:
        raise RuntimeError(
            "cross-attention pool runtime contract and parameter surface disagree"
        )
    if cross_attention_feature_contract:
        if cross_attention_parameter_names != set(CROSS_ATTENTION_PARAMETER_KEYS):
            raise RuntimeError("cross-attention pool parameter surface is incomplete")
        expected_cross_shapes = {
            ADAPTER_PREFIX + "cross_query_norm.weight": (256,),
            ADAPTER_PREFIX + "cross_query_norm.bias": (256,),
            ADAPTER_PREFIX + "cross_text_norm.weight": (256,),
            ADAPTER_PREFIX + "cross_text_norm.bias": (256,),
            ADAPTER_PREFIX + "cross_query_projection.weight": (128, 256),
            ADAPTER_PREFIX + "cross_query_projection.bias": (128,),
            ADAPTER_PREFIX + "cross_text_projection.weight": (128, 256),
            ADAPTER_PREFIX + "cross_text_projection.bias": (128,),
            ADAPTER_PREFIX + "cross_evidence_projection.weight": (128, 3),
            ADAPTER_PREFIX + "cross_evidence_projection.bias": (128,),
            ADAPTER_PREFIX + "cross_attention.in_proj_weight": (384, 128),
            ADAPTER_PREFIX + "cross_attention.in_proj_bias": (384,),
            ADAPTER_PREFIX + "cross_attention.out_proj.weight": (128, 128),
            ADAPTER_PREFIX + "cross_attention.out_proj.bias": (128,),
            ADAPTER_PREFIX + "cross_ffn.0.weight": (128,),
            ADAPTER_PREFIX + "cross_ffn.0.bias": (128,),
            ADAPTER_PREFIX + "cross_ffn.1.weight": (256, 128),
            ADAPTER_PREFIX + "cross_ffn.1.bias": (256,),
            ADAPTER_PREFIX + "cross_ffn.3.weight": (128, 256),
            ADAPTER_PREFIX + "cross_ffn.3.bias": (128,),
            ADAPTER_PREFIX + "cross_output_projection.weight": (256, 128),
            ADAPTER_PREFIX + "cross_output_projection.bias": (256,),
        }
        for name, shape in expected_cross_shapes.items():
            value = runtime[name]
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise RuntimeError(
                    f"cross-attention pool tensor contract drifted: {name}"
                )
    candidate_head_contract = (
        candidate_absolute_contract
        or candidate_calibrated_contract
        or candidate_normalized_contract
        or candidate_asymmetric_contract
        or candidate_set_attention_contract
        or fulltext_global_absolute_contract
    )
    if bool(candidate_absolute_parameter_names) != candidate_head_contract:
        raise RuntimeError(
            "candidate-absolute runtime contract and parameter surface disagree"
        )
    if candidate_head_contract:
        if candidate_absolute_parameter_names != set(
            CANDIDATE_ABSOLUTE_PARAMETER_KEYS
        ):
            raise RuntimeError("candidate-absolute parameter surface is incomplete")
        expected_candidate_shapes = {
            ADAPTER_PREFIX + "candidate_absolute_head.0.weight": (256,),
            ADAPTER_PREFIX + "candidate_absolute_head.0.bias": (256,),
            ADAPTER_PREFIX + "candidate_absolute_head.1.weight": (256, 256),
            ADAPTER_PREFIX + "candidate_absolute_head.1.bias": (256,),
            ADAPTER_PREFIX + "candidate_absolute_head.3.weight": (1, 256),
            ADAPTER_PREFIX + "candidate_absolute_head.3.bias": (1,),
        }
        for name, shape in expected_candidate_shapes.items():
            value = runtime[name]
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise RuntimeError(
                    f"candidate-absolute tensor contract drifted: {name}"
                )
        for name in (
            ADAPTER_PREFIX + "candidate_absolute_head.3.weight",
            ADAPTER_PREFIX + "candidate_absolute_head.3.bias",
        ):
            if bool(torch.count_nonzero(runtime[name]).item()):
                raise RuntimeError(
                    "candidate-absolute output must be exactly zero initialized"
                )
    candidate_calibration_contract = (
        candidate_calibrated_contract
        or candidate_normalized_contract
        or candidate_asymmetric_contract
        or candidate_set_attention_contract
    )
    if bool(candidate_calibration_parameter_names) != candidate_calibration_contract:
        raise RuntimeError(
            "candidate-calibrated runtime contract and parameter surface disagree"
        )
    if candidate_calibration_contract:
        expected_calibration_keys = (
            CANDIDATE_NORMALIZED_CALIBRATION_PARAMETER_KEYS
            if (
                candidate_normalized_contract
                or candidate_asymmetric_contract
                or candidate_set_attention_contract
            )
            else CANDIDATE_CALIBRATION_PARAMETER_KEYS
        )
        if candidate_calibration_parameter_names != set(expected_calibration_keys):
            raise RuntimeError("candidate-calibrated parameter surface is incomplete")
        for name in expected_calibration_keys:
            value = runtime[name]
            if (
                tuple(value.shape) != ()
                or value.dtype != torch.float32
                or bool(torch.count_nonzero(value).item())
            ):
                raise RuntimeError(
                    f"candidate-calibrated scalar must be FP32 zero: {name}"
                )
    if bool(deployed_router_parameter_names) != deployed_router_contract:
        raise RuntimeError(
            "independent deployed-router runtime contract and parameter surface "
            "disagree"
        )
    if deployed_router_contract:
        if not candidate_asymmetric_contract:
            raise RuntimeError(
                "independent deployed-router migration requires the asymmetric "
                "candidate confidence surface"
            )
        if deployed_router_parameter_names != set(DEPLOYED_ROUTER_PARAMETER_KEYS):
            raise RuntimeError("independent deployed-router surface is incomplete")
        expected_router_shapes = {
            ADAPTER_PREFIX + "deployed_router_norm.weight": (10,),
            ADAPTER_PREFIX + "deployed_router_norm.bias": (10,),
            ADAPTER_PREFIX + "deployed_router_residual.0.weight": (64, 10),
            ADAPTER_PREFIX + "deployed_router_residual.0.bias": (64,),
            ADAPTER_PREFIX + "deployed_router_residual.2.weight": (1, 64),
            ADAPTER_PREFIX + "deployed_router_residual.2.bias": (1,),
        }
        for name, shape in expected_router_shapes.items():
            value = runtime[name]
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise RuntimeError(
                    f"independent deployed-router tensor contract drifted: {name}"
                )
        for name in (
            ADAPTER_PREFIX + "deployed_router_residual.2.weight",
            ADAPTER_PREFIX + "deployed_router_residual.2.bias",
        ):
            if bool(torch.count_nonzero(runtime[name]).item()):
                raise RuntimeError(
                    "independent deployed-router output must be exactly zero "
                    "initialized"
                )
    if bool(veto_pool_names) != global_trust_veto_contract:
        raise RuntimeError(
            "global trust/veto runtime contract and veto-pool surface disagree"
        )
    if fulltext_global_absolute_contract and absolute_cap_surface:
        raise RuntimeError(
            "full-text global-absolute confidence must not expose the retired "
            "token-veto cap parameter"
        )
    if global_trust_veto_contract:
        if not candidate_asymmetric_contract:
            raise RuntimeError(
                "global trust/veto migration requires the asymmetric candidate "
                "confidence surface"
            )
        expected_veto_shapes = {
            VETO_POOL_PREFIX + "residual.0.weight": (256, 262),
            VETO_POOL_PREFIX + "residual.0.bias": (256,),
            VETO_POOL_PREFIX + "residual.2.weight": (256, 256),
            VETO_POOL_PREFIX + "residual.2.bias": (256,),
            VETO_POOL_PREFIX + "residual.4.weight": (1, 256),
            VETO_POOL_PREFIX + "residual.4.bias": (1,),
        }
        if set(veto_pool_names) != set(expected_veto_shapes):
            raise RuntimeError("global veto-pool parameter surface is incomplete")
        for name, shape in expected_veto_shapes.items():
            value = runtime[name]
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise RuntimeError(
                    f"global veto-pool tensor contract drifted: {name}"
                )
        for name in (
            VETO_POOL_PREFIX + "residual.4.weight",
            VETO_POOL_PREFIX + "residual.4.bias",
        ):
            if bool(torch.count_nonzero(runtime[name]).item()):
                raise RuntimeError(
                    "global veto-pool output must be exactly zero initialized"
                )
    audit = {
        "schema": (
            DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
            if deployment_owned_query_veto_global_absolute_pool_contract
            else
            DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
            if deployment_owned_query_global_absolute_pool_contract
            else DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
            if deployment_owned_global_absolute_pool_contract
            else FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA
            if fulltext_global_independent_absolute_pool_contract
            else FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
            if fulltext_global_absolute_exact_residual_pool_contract
            else FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
            if fulltext_global_absolute_pool_contract
            else CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA
            if candidate_sample_calibrator_contract
            else DEPLOYED_ROUTER_MIGRATION_SCHEMA
            if deployed_router_contract
            else GLOBAL_TRUST_VETO_MIGRATION_SCHEMA
            if global_trust_veto_contract
            else CANDIDATE_SET_ATTENTION_MIGRATION_SCHEMA
            if candidate_set_attention_contract
            else (
                CANDIDATE_ASYMMETRIC_MIGRATION_SCHEMA
                if candidate_asymmetric_contract
                else (
                    CANDIDATE_NORMALIZED_MIGRATION_SCHEMA
                    if candidate_normalized_contract
                    else (
                        CANDIDATE_CALIBRATED_MIGRATION_SCHEMA
                        if candidate_calibrated_contract
                        else (
                            CANDIDATE_ABSOLUTE_MIGRATION_SCHEMA
                            if candidate_absolute_contract
                            else (
                                CROSS_ATTENTION_ABSOLUTE_POOL_MIGRATION_SCHEMA
                                if cross_attention_pool_contract
                                else (
                                    SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA
                                    if signed_rank_query_pool_contract
                                    else (
                                        SPARSE_RANK_CHANNEL_MIGRATION_SCHEMA
                                        if rank_channel_contract
                                        else (
                                            CARRIER_AFFINE_MIGRATION_SCHEMA
                                            if carrier_affine_contract
                                            else (
                                                CARRIER_SLOPE_MIGRATION_SCHEMA
                                                if carrier_slope_contract
                                                else (
                                                    GATE_MARGIN_MIGRATION_SCHEMA
                                                    if gate_margin_surface
                                                    else (
                                                        RANK_AFFINE_MIGRATION_SCHEMA
                                                        if rank_affine_surface
                                                        else (
                                                            RANK_EVIDENCE_MIGRATION_SCHEMA
                                                            if rank_evidence_surface
                                                            else (
                                                                ABSOLUTE_CAP_MIGRATION_SCHEMA
                                                                if absolute_cap_surface
                                                                else MIGRATION_SCHEMA
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "token_logit_contract": TOKEN_LOGIT_CONTRACT,
        "pool_feature_contract": runtime_pool_feature_contract,
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "source_optimizer_updates": int(source_optimizer_updates),
        "source_checkpoint_reason": str(source_checkpoint_reason),
        "rank": rank_fingerprint,
        "transferred": transferred_fingerprint,
        "fresh_confidence": fresh_fingerprint,
        "retired_confidence_tower_tensor_count": len(legacy_confidence_names),
        "retired_confidence_pool_tensor_count": len(legacy_pool_names),
        "retired_confidence_loaded_tensor_count": 0,
        "adapter_tensor_count": len(adapter_names),
        "pool_tensor_count": len(pool_names),
        "strict_target_tensor_count": len(migrated),
    }
    if fulltext_global_absolute_contract:
        audit["fresh_confidence_contract"] = (
            DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
            if deployment_owned_query_veto_global_absolute_pool_contract
            else
            DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
            if deployment_owned_query_global_absolute_pool_contract
            else DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
            if deployment_owned_global_absolute_pool_contract
            else FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
            if fulltext_global_independent_absolute_pool_contract
            else FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
            if fulltext_global_absolute_exact_residual_pool_contract
            else FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
        audit["head_gradient_contract"] = (
            DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            if deployment_owned_query_veto_global_absolute_pool_contract
            else
            DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            if deployment_owned_query_global_absolute_pool_contract
            else DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            if deployment_owned_global_absolute_pool_contract
            else FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT
            if fulltext_global_independent_absolute_pool_contract
            else FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT
        )
        audit["confidence_parameter_tensor_count"] = len(
            confidence_parameter_names
        )
        audit["confidence_parameter_element_count"] = (
            confidence_parameter_element_count
        )
        if deployment_owned_global_absolute_pool_contract:
            audit.update(
                {
                    "active_confidence_parameter_tensor_count": len(
                        active_confidence_parameter_names
                    ),
                    "active_confidence_parameter_element_count": sum(
                        int(runtime_parameters[name].numel())
                        for name in active_confidence_parameter_names
                    ),
                    "diagnostic_candidate_parameter_tensor_count": len(
                        diagnostic_candidate_parameter_names
                    ),
                    "diagnostic_candidate_parameter_element_count": sum(
                        int(runtime_parameters[name].numel())
                        for name in diagnostic_candidate_parameter_names
                    ),
                    "diagnostic_candidate_requires_grad_count": sum(
                        int(runtime_parameters[name].requires_grad)
                        for name in diagnostic_candidate_parameter_names
                    ),
                }
            )
        elif (
            deployment_owned_query_global_absolute_pool_contract
            or deployment_owned_query_veto_global_absolute_pool_contract
        ):
            audit.update(
                {
                    "active_confidence_parameter_tensor_count": len(
                        active_confidence_parameter_names
                    ),
                    "active_confidence_parameter_element_count": sum(
                        int(runtime_parameters[name].numel())
                        for name in active_confidence_parameter_names
                    ),
                    "deployed_query_parameter_tensor_count": len(
                        diagnostic_candidate_parameter_names
                    ),
                    "deployed_query_parameter_element_count": sum(
                        int(runtime_parameters[name].numel())
                        for name in diagnostic_candidate_parameter_names
                    ),
                    "deployed_query_requires_grad_count": sum(
                        int(runtime_parameters[name].requires_grad)
                        for name in diagnostic_candidate_parameter_names
                    ),
                }
            )
    elif candidate_sample_calibrator_contract:
        audit["fresh_confidence_contract"] = (
            CANDIDATE_SAMPLE_CALIBRATOR_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
        audit["head_gradient_contract"] = (
            CANDIDATE_SAMPLE_CALIBRATOR_HEAD_GRADIENT_CONTRACT
        )
    elif deployed_router_contract:
        audit["fresh_confidence_contract"] = (
            DEPLOYED_ROUTER_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
        audit["head_gradient_contract"] = DEPLOYED_ROUTER_HEAD_GRADIENT_CONTRACT
    elif global_trust_veto_contract:
        audit["veto_pool_tensor_count"] = len(veto_pool_names)
        audit["fresh_confidence_contract"] = (
            GLOBAL_TRUST_VETO_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
        audit["head_gradient_contract"] = (
            GLOBAL_TRUST_VETO_HEAD_GRADIENT_CONTRACT
        )
    elif candidate_set_attention_contract:
        audit["fresh_confidence_contract"] = (
            CANDIDATE_SET_ATTENTION_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif candidate_asymmetric_contract:
        audit["fresh_confidence_contract"] = (
            CANDIDATE_ASYMMETRIC_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif candidate_normalized_contract:
        audit["fresh_confidence_contract"] = (
            CANDIDATE_NORMALIZED_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif candidate_calibrated_contract:
        audit["fresh_confidence_contract"] = (
            CANDIDATE_CALIBRATED_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif candidate_absolute_contract:
        audit["fresh_confidence_contract"] = (
            CANDIDATE_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif cross_attention_pool_contract:
        audit["fresh_confidence_contract"] = (
            CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif signed_rank_query_pool_contract:
        audit["fresh_confidence_contract"] = (
            SIGNED_RANK_QUERY_POOL_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif rank_channel_contract:
        audit["fresh_confidence_contract"] = (
            SPARSE_RANK_CHANNEL_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT
    elif carrier_affine_contract:
        audit["fresh_confidence_contract"] = (
            CARRIER_AFFINE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = CARRIER_AFFINE_RESIDUAL_CONTRACT
    elif carrier_slope_contract:
        audit["fresh_confidence_contract"] = (
            CARRIER_SLOPE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = CARRIER_SLOPE_RESIDUAL_CONTRACT
    elif gate_margin_surface:
        audit["fresh_confidence_contract"] = (
            GATE_MARGIN_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = GATE_MARGIN_RESIDUAL_CONTRACT
    elif rank_affine_surface:
        audit["fresh_confidence_contract"] = (
            RANK_AFFINE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = RANK_AFFINE_RESIDUAL_CONTRACT
    elif rank_evidence_surface:
        audit["fresh_confidence_contract"] = (
            RANK_EVIDENCE_FRESH_CONFIDENCE_CONTRACT
        )
        audit["rank_evidence_contract"] = RANK_EVIDENCE_RESIDUAL_CONTRACT
    elif absolute_cap_surface:
        audit["fresh_confidence_contract"] = (
            ABSOLUTE_CAP_FRESH_CONFIDENCE_CONTRACT
        )
    return migrated, audit


__all__ = [
    "DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT",
    "DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT",
    "DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA",
    "DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT",
    "DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT",
    "DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT",
    "DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA",
    "DEPLOYMENT_OWNED_QUERY_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT",
    "DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT",
    "DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT",
    "DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA",
    "DEPLOYMENT_OWNED_QUERY_VETO_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT",
    "EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_ELEMENT_COUNT",
    "EXPECTED_DEPLOYMENT_OWNED_ACTIVE_PARAMETER_TENSOR_COUNT",
    "EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_ELEMENT_COUNT",
    "EXPECTED_DEPLOYMENT_OWNED_DIAGNOSTIC_PARAMETER_TENSOR_COUNT",
    "FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT",
    "FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_GRADIENT_CONTRACT",
    "FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA",
    "FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT",
    "FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT",
    "FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA",
    "FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT",
    "FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT",
    "FULLTEXT_GLOBAL_ABSOLUTE_GATE_GRADIENT_CONTRACT",
    "FULLTEXT_GLOBAL_ABSOLUTE_HEAD_GRADIENT_CONTRACT",
    "FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA",
    "FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_ADAPTER_TENSOR_COUNT",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_ELEMENT_COUNT",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_PARAMETER_TENSOR_COUNT",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_ELEMENT_COUNT",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_SHA256",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_STORAGE_BYTES",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_TENSOR_COUNT",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_POOL_TENSOR_COUNT",
    "EXPECTED_FULLTEXT_GLOBAL_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT",
    "CANDIDATE_SAMPLE_CALIBRATOR_FRESH_CONFIDENCE_CONTRACT",
    "CANDIDATE_SAMPLE_CALIBRATOR_HEAD_GRADIENT_CONTRACT",
    "CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA",
    "EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_ELEMENT_COUNT",
    "EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_SHA256",
    "EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_STORAGE_BYTES",
    "EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_FRESH_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_SAMPLE_CALIBRATOR_STRICT_TARGET_TENSOR_COUNT",
    "DEPLOYED_ROUTER_FRESH_CONFIDENCE_CONTRACT",
    "DEPLOYED_ROUTER_HEAD_GRADIENT_CONTRACT",
    "DEPLOYED_ROUTER_MIGRATION_SCHEMA",
    "EXPECTED_DEPLOYED_ROUTER_ADAPTER_TENSOR_COUNT",
    "EXPECTED_DEPLOYED_ROUTER_FRESH_ELEMENT_COUNT",
    "EXPECTED_DEPLOYED_ROUTER_FRESH_SHA256",
    "EXPECTED_DEPLOYED_ROUTER_FRESH_STORAGE_BYTES",
    "EXPECTED_DEPLOYED_ROUTER_FRESH_TENSOR_COUNT",
    "EXPECTED_DEPLOYED_ROUTER_STRICT_TARGET_TENSOR_COUNT",
    "ABSOLUTE_CAP_FRESH_CONFIDENCE_CONTRACT",
    "ABSOLUTE_CAP_MIGRATION_SCHEMA",
    "CARRIER_SLOPE_FRESH_CONFIDENCE_CONTRACT",
    "CARRIER_SLOPE_MIGRATION_SCHEMA",
    "CARRIER_SLOPE_RESIDUAL_CONTRACT",
    "CARRIER_AFFINE_FRESH_CONFIDENCE_CONTRACT",
    "CARRIER_AFFINE_MIGRATION_SCHEMA",
    "CARRIER_AFFINE_RESIDUAL_CONTRACT",
    "SPARSE_RANK_CHANNEL_FRESH_CONFIDENCE_CONTRACT",
    "SPARSE_RANK_CHANNEL_MIGRATION_SCHEMA",
    "SPARSE_RANK_CHANNEL_RESIDUAL_CONTRACT",
    "SIGNED_RANK_QUERY_POOL_FRESH_CONFIDENCE_CONTRACT",
    "SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACT",
    "SIGNED_RANK_QUERY_POOL_FEATURE_CONTRACTS",
    "TOKEN_CONDITIONED_POOL_FEATURE_CONTRACT",
    "SIGNED_RANK_QUERY_POOL_MIGRATION_SCHEMA",
    "CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_CONFIDENCE_CONTRACT",
    "CROSS_ATTENTION_ABSOLUTE_POOL_FEATURE_CONTRACT",
    "CROSS_ATTENTION_ABSOLUTE_POOL_MIGRATION_SCHEMA",
    "EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_ELEMENT_COUNT",
    "EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_SHA256",
    "EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_STORAGE_BYTES",
    "EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_FRESH_TENSOR_COUNT",
    "EXPECTED_CROSS_ATTENTION_ABSOLUTE_POOL_STRICT_TARGET_TENSOR_COUNT",
    "CANDIDATE_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT",
    "CANDIDATE_ABSOLUTE_POOL_FEATURE_CONTRACT",
    "CANDIDATE_ABSOLUTE_MIGRATION_SCHEMA",
    "EXPECTED_CANDIDATE_ABSOLUTE_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_ABSOLUTE_FRESH_ELEMENT_COUNT",
    "EXPECTED_CANDIDATE_ABSOLUTE_FRESH_SHA256",
    "EXPECTED_CANDIDATE_ABSOLUTE_FRESH_STORAGE_BYTES",
    "EXPECTED_CANDIDATE_ABSOLUTE_FRESH_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_ABSOLUTE_STRICT_TARGET_TENSOR_COUNT",
    "CANDIDATE_CALIBRATED_FRESH_CONFIDENCE_CONTRACT",
    "CANDIDATE_CALIBRATED_POOL_FEATURE_CONTRACT",
    "CANDIDATE_CALIBRATED_MIGRATION_SCHEMA",
    "EXPECTED_CANDIDATE_CALIBRATED_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_CALIBRATED_FRESH_ELEMENT_COUNT",
    "EXPECTED_CANDIDATE_CALIBRATED_FRESH_SHA256",
    "EXPECTED_CANDIDATE_CALIBRATED_FRESH_STORAGE_BYTES",
    "EXPECTED_CANDIDATE_CALIBRATED_FRESH_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_CALIBRATED_STRICT_TARGET_TENSOR_COUNT",
    "CANDIDATE_NORMALIZED_FRESH_CONFIDENCE_CONTRACT",
    "CANDIDATE_NORMALIZED_POOL_FEATURE_CONTRACT",
    "CANDIDATE_NORMALIZED_MIGRATION_SCHEMA",
    "EXPECTED_CANDIDATE_NORMALIZED_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_NORMALIZED_FRESH_ELEMENT_COUNT",
    "EXPECTED_CANDIDATE_NORMALIZED_FRESH_SHA256",
    "EXPECTED_CANDIDATE_NORMALIZED_FRESH_STORAGE_BYTES",
    "EXPECTED_CANDIDATE_NORMALIZED_FRESH_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_NORMALIZED_STRICT_TARGET_TENSOR_COUNT",
    "CANDIDATE_ASYMMETRIC_FRESH_CONFIDENCE_CONTRACT",
    "CANDIDATE_ASYMMETRIC_POOL_FEATURE_CONTRACT",
    "CANDIDATE_ASYMMETRIC_MIGRATION_SCHEMA",
    "EXPECTED_CANDIDATE_ASYMMETRIC_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_ELEMENT_COUNT",
    "EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_SHA256",
    "EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_STORAGE_BYTES",
    "EXPECTED_CANDIDATE_ASYMMETRIC_FRESH_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_ASYMMETRIC_STRICT_TARGET_TENSOR_COUNT",
    "CANDIDATE_SET_ATTENTION_FRESH_CONFIDENCE_CONTRACT",
    "CANDIDATE_SET_ATTENTION_POOL_FEATURE_CONTRACT",
    "CANDIDATE_SET_ATTENTION_MIGRATION_SCHEMA",
    "EXPECTED_CANDIDATE_SET_ATTENTION_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_ELEMENT_COUNT",
    "EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_SHA256",
    "EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_STORAGE_BYTES",
    "EXPECTED_CANDIDATE_SET_ATTENTION_FRESH_TENSOR_COUNT",
    "EXPECTED_CANDIDATE_SET_ATTENTION_STRICT_TARGET_TENSOR_COUNT",
    "GLOBAL_TRUST_VETO_FRESH_CONFIDENCE_CONTRACT",
    "GLOBAL_TRUST_VETO_HEAD_GRADIENT_CONTRACT",
    "GLOBAL_TRUST_VETO_MIGRATION_SCHEMA",
    "EXPECTED_GLOBAL_TRUST_VETO_ADAPTER_TENSOR_COUNT",
    "EXPECTED_GLOBAL_TRUST_VETO_FRESH_ELEMENT_COUNT",
    "EXPECTED_GLOBAL_TRUST_VETO_FRESH_SHA256",
    "EXPECTED_GLOBAL_TRUST_VETO_FRESH_STORAGE_BYTES",
    "EXPECTED_GLOBAL_TRUST_VETO_FRESH_TENSOR_COUNT",
    "EXPECTED_GLOBAL_TRUST_VETO_POOL_TENSOR_COUNT",
    "EXPECTED_GLOBAL_TRUST_VETO_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_GLOBAL_TRUST_VETO_VETO_POOL_TENSOR_COUNT",
    "EXPECTED_SIGNED_RANK_QUERY_POOL_ADAPTER_TENSOR_COUNT",
    "EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_ELEMENT_COUNT",
    "EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_SHA256",
    "EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_STORAGE_BYTES",
    "EXPECTED_SIGNED_RANK_QUERY_POOL_FRESH_TENSOR_COUNT",
    "EXPECTED_SIGNED_RANK_QUERY_POOL_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_SPARSE_RANK_CHANNEL_ADAPTER_TENSOR_COUNT",
    "EXPECTED_SPARSE_RANK_CHANNEL_FRESH_ELEMENT_COUNT",
    "EXPECTED_SPARSE_RANK_CHANNEL_FRESH_SHA256",
    "EXPECTED_SPARSE_RANK_CHANNEL_FRESH_STORAGE_BYTES",
    "EXPECTED_SPARSE_RANK_CHANNEL_FRESH_TENSOR_COUNT",
    "EXPECTED_SPARSE_RANK_CHANNEL_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_CARRIER_AFFINE_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CARRIER_AFFINE_FRESH_ELEMENT_COUNT",
    "EXPECTED_CARRIER_AFFINE_FRESH_SHA256",
    "EXPECTED_CARRIER_AFFINE_FRESH_STORAGE_BYTES",
    "EXPECTED_CARRIER_AFFINE_FRESH_TENSOR_COUNT",
    "EXPECTED_CARRIER_AFFINE_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_CARRIER_SLOPE_ADAPTER_TENSOR_COUNT",
    "EXPECTED_CARRIER_SLOPE_FRESH_ELEMENT_COUNT",
    "EXPECTED_CARRIER_SLOPE_FRESH_SHA256",
    "EXPECTED_CARRIER_SLOPE_FRESH_STORAGE_BYTES",
    "EXPECTED_CARRIER_SLOPE_FRESH_TENSOR_COUNT",
    "EXPECTED_CARRIER_SLOPE_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_ABSOLUTE_CAP_ADAPTER_TENSOR_COUNT",
    "EXPECTED_ABSOLUTE_CAP_FRESH_ELEMENT_COUNT",
    "EXPECTED_ABSOLUTE_CAP_FRESH_SHA256",
    "EXPECTED_ABSOLUTE_CAP_FRESH_STORAGE_BYTES",
    "EXPECTED_ABSOLUTE_CAP_FRESH_TENSOR_COUNT",
    "EXPECTED_ABSOLUTE_CAP_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_RANK_AFFINE_ADAPTER_TENSOR_COUNT",
    "EXPECTED_RANK_AFFINE_FRESH_ELEMENT_COUNT",
    "EXPECTED_RANK_AFFINE_FRESH_SHA256",
    "EXPECTED_RANK_AFFINE_FRESH_STORAGE_BYTES",
    "EXPECTED_RANK_AFFINE_FRESH_TENSOR_COUNT",
    "EXPECTED_RANK_AFFINE_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_RANK_EVIDENCE_ADAPTER_TENSOR_COUNT",
    "EXPECTED_RANK_EVIDENCE_FRESH_ELEMENT_COUNT",
    "EXPECTED_RANK_EVIDENCE_FRESH_SHA256",
    "EXPECTED_RANK_EVIDENCE_FRESH_STORAGE_BYTES",
    "EXPECTED_RANK_EVIDENCE_FRESH_TENSOR_COUNT",
    "EXPECTED_RANK_EVIDENCE_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_ADAPTER_TENSOR_COUNT",
    "EXPECTED_FRESH_ELEMENT_COUNT",
    "EXPECTED_FRESH_SHA256",
    "EXPECTED_FRESH_STORAGE_BYTES",
    "EXPECTED_FRESH_TENSOR_COUNT",
    "EXPECTED_GATE_MARGIN_ADAPTER_TENSOR_COUNT",
    "EXPECTED_GATE_MARGIN_FRESH_ELEMENT_COUNT",
    "EXPECTED_GATE_MARGIN_FRESH_SHA256",
    "EXPECTED_GATE_MARGIN_FRESH_STORAGE_BYTES",
    "EXPECTED_GATE_MARGIN_FRESH_TENSOR_COUNT",
    "EXPECTED_GATE_MARGIN_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_POOL_TENSOR_COUNT",
    "EXPECTED_RANK_TENSOR_COUNT",
    "EXPECTED_STRICT_TARGET_TENSOR_COUNT",
    "EXPECTED_TRANSFERRED_TENSOR_COUNT",
    "MIGRATION_SCHEMA",
    "GATE_MARGIN_FRESH_CONFIDENCE_CONTRACT",
    "GATE_MARGIN_MIGRATION_SCHEMA",
    "GATE_MARGIN_RESIDUAL_CONTRACT",
    "POOL_FEATURE_CONTRACT",
    "RANK_AFFINE_FRESH_CONFIDENCE_CONTRACT",
    "RANK_AFFINE_MIGRATION_SCHEMA",
    "RANK_AFFINE_RESIDUAL_CONTRACT",
    "RANK_EVIDENCE_FRESH_CONFIDENCE_CONTRACT",
    "RANK_EVIDENCE_MIGRATION_SCHEMA",
    "RANK_EVIDENCE_RESIDUAL_CONTRACT",
    "TOKEN_LOGIT_CONTRACT",
    "migrate_legacy_rank_to_confidence_adapter",
    "validate_confidence_adapter_migration_audit",
]
