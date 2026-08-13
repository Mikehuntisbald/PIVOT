#!/usr/bin/env python3
"""Auditor profile for the Stage-B total-trust confidence branch.

The historical two-phase auditor remains the default P3 contract.  This
profile reuses its fail-closed lineage checks while changing only the
confidence objective/configuration and publishing a distinct schema so that
the resulting checkpoints cannot be mistaken for the older experiment.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import stageb_gdino_adapter_probe_audit as _base


_HISTORICAL_SCHEMA = _base.SCHEMA
_HISTORICAL_PHASE_SPECS = copy.deepcopy(_base.PHASE_SPECS)
_LEGACY_REPLAY_SCHEMA = "pivot.stageb.legacy_replay_receipt/v1"
_LEGACY_REPLAY_RECEIPT_SHA256 = (
    "f8bd5104960eca19b90353168577a126a5577ea9aac511c6b0e3ab01b4bf2bfc"
)
_LEGACY_REPLAY_SELF_SHA256 = (
    "7979cea6caf551889b0fd7cf2bc334294cb4152ec0ef3b5f4e069acb7ce606a6"
)
_LEGACY_B58_FILE_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
_LEGACY_B58_BASE_SHA256 = (
    "889ea86458e5c2a3a92a2cf4cbfc006662e9dd268f0d9e22bd9abbdafaa2ec14"
)
_LEGACY_R100_FILE_SHA256 = (
    "a933725aab2226d35aa1cd94992c154b332bbbbc3406060bbda061cef4959dd5"
)
_LEGACY_R100_RANK_SHA256 = (
    "90f8d970ffeb2b906acedc6b908ecc96802acc88362ee1a51d04b5c93e0fe335"
)
_LEGACY_R100_CONFIDENCE_SHA256 = (
    "521c33d6ccbd408b9935f41a3ef986f23676803e6931ccbf1b50fbc9650e5be0"
)
LEGACY_R100_INITIALIZATION = (
    "sealed_legacy_b58_r100_to_total_trust_confidence_pretrain_model_path"
)
_base.SCHEMA = "stageb-gdino-adapter-total-trust-probe-v1"
_base.PROBE_WORLD_SIZE = 1
_base.PROBE_PER_GPU_BATCH = 8
_base.PHASE_SPECS = copy.deepcopy(_HISTORICAL_PHASE_SPECS)
_base.PHASE_SPECS["confidence"].update(
    {
        "confidence_objective_code": 3,
        "confidence_objective": "detached_recent_q05_total_trust",
        "paired_margin_weight": 0.0,
        "config": (
            "config/ablations/"
            "cfg_stageb_gdino_score_adapter_dataft_total_trust.py"
        ),
    }
)
_base.TRAIN_CODE_INCLUDE = tuple(_base.TRAIN_CODE_INCLUDE) + (
    "tools/build_stageb_legacy_replay_receipt.py",
    "tools/stageb_gdino_adapter_total_trust_probe_audit.py",
)
_base.ORCHESTRATION_PATHS = tuple(_base.ORCHESTRATION_PATHS) + (
    "tools/run_stageb_gdino_adapter_total_trust_probe.sh",
)

# A total-trust confidence run can consume an already audited historical P3
# rank milestone.  Temporarily replay that input under the historical schema;
# all newly emitted confidence lineage remains total-trust schema tagged.
_TOTAL_VALIDATE_RANK_INITIAL = _base._validate_rank_initial


def _require_mapping(value, *, label):
    if not isinstance(value, dict):
        raise _base.ProbeAuditError(f"legacy replay receipt has no {label} mapping")
    return value


def _validate_legacy_r100_initial(initial_checkpoint, receipt_path, receipt):
    from tools.build_stageb_legacy_replay_receipt import (
        LegacyReplayReceiptError,
        verify_receipt,
    )

    receipt_record = _base.file_record(_base.resolve_path(receipt_path))
    if receipt_record["sha256"] != _LEGACY_REPLAY_RECEIPT_SHA256:
        raise _base.ProbeAuditError("legacy replay receipt file hash drifted")
    if receipt.get("receipt_sha256") != _LEGACY_REPLAY_SELF_SHA256:
        raise _base.ProbeAuditError("legacy replay receipt self-hash drifted")
    try:
        verified = verify_receipt(_base.resolve_path(receipt_path))
    except (LegacyReplayReceiptError, _base.ProbeAuditError) as error:
        raise _base.ProbeAuditError(
            f"legacy replay receipt verification failed: {error}"
        ) from error
    if (
        verified.get("status") != "verified"
        or verified.get("receipt_sha256") != _LEGACY_REPLAY_SELF_SHA256
        or verified.get("receipt") != receipt_record
    ):
        raise _base.ProbeAuditError("legacy replay receipt verification is incomplete")

    checkpoints = _require_mapping(receipt.get("checkpoints"), label="checkpoints")
    invariants = _require_mapping(receipt.get("invariants"), label="invariants")
    expected_hashes = _require_mapping(
        receipt.get("expected_hashes"), label="expected_hashes"
    )
    b58 = _require_mapping(checkpoints.get("b58"), label="b58 checkpoint")
    rank = _require_mapping(
        checkpoints.get("rank_r100"), label="rank_r100 checkpoint"
    )
    b58_file = _require_mapping(b58.get("file"), label="b58 file")
    b58_model = _require_mapping(b58.get("model"), label="b58 model")
    rank_file = _require_mapping(rank.get("file"), label="rank_r100 file")
    rank_model = _require_mapping(rank.get("model"), label="rank_r100 model")
    rank_training = _require_mapping(
        rank.get("training"), label="rank_r100 training"
    )
    rank_args = _require_mapping(
        rank_training.get("args_summary"), label="rank_r100 args"
    )

    locked = {
        "b58_file": (b58_file.get("sha256"), _LEGACY_B58_FILE_SHA256),
        "b58_base": (b58_model.get("base_tensor_sha256"), _LEGACY_B58_BASE_SHA256),
        "rank_file": (rank_file.get("sha256"), _LEGACY_R100_FILE_SHA256),
        "rank_base": (rank_model.get("base_tensor_sha256"), _LEGACY_B58_BASE_SHA256),
        "rank_branch": (rank_model.get("rank_tensor_sha256"), _LEGACY_R100_RANK_SHA256),
        "rank_confidence": (
            rank_model.get("confidence_tensor_sha256"),
            _LEGACY_R100_CONFIDENCE_SHA256,
        ),
    }
    for label, (observed, expected) in locked.items():
        if observed != expected:
            raise _base.ProbeAuditError(
                f"legacy replay receipt {label} hash drifted: {observed!r}"
            )
    for role, field, expected in (
        ("b58", "file_sha256", _LEGACY_B58_FILE_SHA256),
        ("b58", "base_tensor_sha256", _LEGACY_B58_BASE_SHA256),
        ("rank_r100", "file_sha256", _LEGACY_R100_FILE_SHA256),
        ("rank_r100", "rank_tensor_sha256", _LEGACY_R100_RANK_SHA256),
    ):
        role_hashes = _require_mapping(expected_hashes.get(role), label=f"{role} hashes")
        if role_hashes.get(field) != expected:
            raise _base.ProbeAuditError(
                f"legacy replay receipt does not lock {role}.{field}"
            )
    required_invariants = (
        "b58_has_no_adapter",
        "rank_r100_rank_trained_confidence_untouched",
        "b58_rank_confidence_shared_base_bitwise",
    )
    if any(invariants.get(key) is not True for key in required_invariants):
        raise _base.ProbeAuditError("legacy replay receipt rank/base invariants are incomplete")
    if (
        invariants.get("rank_r100_iteration_and_updates") != 100
        or rank_training.get("iteration") != 100
        or rank_training.get("optimizer_updates") != 100
        or rank_model.get("rank_final_zero") is not False
        or rank_model.get("confidence_final_zero") is not True
    ):
        raise _base.ProbeAuditError("legacy R100 branch-isolation contract drifted")

    expected_args = {
        "config_file": "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py",
        "datasets": "config/datasets_stageb_gdino_adapter_rank_three_ref.json",
        "seed": 42,
        "batch_size": 32,
        "world_size": 1,
        "distributed": False,
        "amp": True,
        "max_train_iters": 100,
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "stage_b_gdino_tn_scope": "",
        "stage_b_gdino_rank_weight": 1.0,
        "stage_b_gdino_confidence_weight": 0.0,
        "stage_b_gdino_queue_size": 0,
        "stage_b_gdino_queue_min_count": 0,
    }
    for key, expected in expected_args.items():
        if rank_args.get(key) != expected:
            raise _base.ProbeAuditError(
                f"legacy R100 training arg {key} drifted: {rank_args.get(key)!r}"
            )

    initial_checkpoint = _base.resolve_path(initial_checkpoint)
    if _base.resolve_path(rank_file.get("path", "")) != initial_checkpoint:
        raise _base.ProbeAuditError(
            "confidence initial checkpoint is not receipt-bound legacy R100"
        )
    record = _base.checkpoint_record(initial_checkpoint)
    if (
        record.get("sha256") != _LEGACY_R100_FILE_SHA256
        or record.get("base_model_sha256") != _LEGACY_B58_BASE_SHA256
        or record.get("rank_sha256") != _LEGACY_R100_RANK_SHA256
        or record.get("confidence_sha256") != _LEGACY_R100_CONFIDENCE_SHA256
        or record.get("rank_final_zero") is not False
        or record.get("confidence_final_zero") is not True
    ):
        raise _base.ProbeAuditError("legacy R100 checkpoint payload drifted from receipt")
    return record


def _validate_rank_initial_compat(initial_checkpoint, initial_audit):
    audit = _base.read_json(_base.resolve_path(initial_audit))
    if audit.get("schema") == _LEGACY_REPLAY_SCHEMA:
        return _validate_legacy_r100_initial(
            initial_checkpoint, initial_audit, audit
        )
    if audit.get("schema") != _HISTORICAL_SCHEMA:
        return _TOTAL_VALIDATE_RANK_INITIAL(initial_checkpoint, initial_audit)
    saved_schema = _base.SCHEMA
    saved_specs = _base.PHASE_SPECS
    saved_preflight_validator = _base._verify_current_preflight

    def _sealed_historical_preflight(path):
        # The rank sidecar is an already-sealed input artifact.  Revalidate its
        # immutable rank/data contract, while allowing code-closure hashes to
        # remain historical because this confidence profile changes runtime
        # code after the rank checkpoint was produced.
        preflight = _base.read_json(path)
        if (
            preflight.get("schema") != _HISTORICAL_SCHEMA
            or preflight.get("kind") != "phase_preflight"
            or preflight.get("phase") != "rank"
            or not isinstance(preflight.get("static"), dict)
            or not isinstance(preflight.get("launch"), dict)
        ):
            raise _base.ProbeAuditError(
                "historical rank preflight is missing sealed structural fields"
            )
        launch = preflight["launch"]
        if {
            "world_size": launch.get("world_size"),
            "per_gpu_batch": launch.get("per_gpu_batch"),
            "global_batch": launch.get("global_batch"),
        } != {"world_size": 2, "per_gpu_batch": 4, "global_batch": 8}:
            raise _base.ProbeAuditError("historical rank launch contract drifted")
        if tuple(launch.get("milestones", ())) != _base.RANK_MILESTONES:
            raise _base.ProbeAuditError("historical rank milestone contract drifted")
        if (
            launch.get("initialization")
            != "fixed_pure_stageb_dataft_to_rank_pretrain_model_path"
            or launch.get("first_invocation") != "pretrain_model_path"
            or launch.get("later_invocations") != "resume"
            or preflight.get("initial_audit") is not None
        ):
            raise _base.ProbeAuditError(
                "historical rank initialization contract drifted"
            )
        initial = preflight.get("initial_checkpoint")
        if not isinstance(initial, dict):
            raise _base.ProbeAuditError(
                "historical rank preflight has no initial checkpoint"
            )
        initial_path = _base.resolve_path(initial.get("path", ""))
        if _base.file_record(initial_path) != initial:
            raise _base.ProbeAuditError("historical rank initial checkpoint drifted")

        static = preflight["static"]
        if (
            static.get("phase") != "rank"
            or static.get("train_mode") != "rank_only"
            or static.get("tn_scope") != ""
        ):
            raise _base.ProbeAuditError("historical rank static contract drifted")
        historical_spec = _HISTORICAL_PHASE_SPECS["rank"]
        for key in ("config", "datasets"):
            record = static.get(key)
            if not isinstance(record, dict):
                raise _base.ProbeAuditError(
                    f"historical rank static contract has no {key} record"
                )
            expected_path = _base.resolve_path(historical_spec[key])
            if _base.resolve_path(record.get("path", "")) != expected_path:
                raise _base.ProbeAuditError(
                    f"historical rank {key} path does not match the sealed protocol"
                )
            if _base.file_record(expected_path) != record:
                raise _base.ProbeAuditError(
                    f"historical rank {key} file record drifted"
                )
        resolved_config = static.get("resolved_config")
        if not isinstance(resolved_config, dict):
            raise _base.ProbeAuditError(
                "historical rank resolved config contract is missing"
            )
        expected_resolved = {
            "stage_b_gdino_score_adapter": True,
            "stage_b_gdino_adapter_train_mode": "rank_only",
            "stage_b_gdino_tn_scope": "",
            "batch_size": 4,
            "epochs": 1,
            "skip_eval": True,
            "data_aug_hflip_prob": 0.0,
            "stage_b_gdino_confidence_objective": historical_spec[
                "confidence_objective"
            ],
            "stage_b_gdino_paired_margin_weight": 0.0,
            "stage_b_gdino_rank_weight": 1.0,
            "stage_b_gdino_confidence_weight": 0.0,
        }
        for key, expected in expected_resolved.items():
            if resolved_config.get(key) != expected:
                raise _base.ProbeAuditError(
                    f"historical rank resolved config drifted at {key}"
                )
        sources = static.get("sources")
        if not isinstance(sources, list) or len(sources) != len(
            historical_spec["sources"]
        ):
            raise _base.ProbeAuditError("historical rank source contract drifted")
        for observed, expected in zip(sources, historical_spec["sources"]):
            if not isinstance(observed, dict) or any(
                observed.get(key) != expected[key]
                for key in ("rows", "sha256", "dataset_mode", "mix_weight")
            ) or _base.resolve_path(observed.get("path", "")) != _base.resolve_path(
                expected["path"]
            ):
                raise _base.ProbeAuditError("historical rank source contract drifted")
        for section in ("config_import_chain", "code", "orchestration"):
            records = static.get(section)
            if not isinstance(records, list) or not records:
                raise _base.ProbeAuditError(
                    f"historical rank {section} records are missing"
                )
            for record in records:
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("path"), str)
                    or int(record.get("size_bytes", -1)) < 0
                    or not isinstance(record.get("sha256"), str)
                    or len(record["sha256"]) != 64
                    or not _base.resolve_path(record["path"]).is_file()
                ):
                    raise _base.ProbeAuditError(
                        f"historical rank {section} record is malformed"
                    )
        # These records deliberately remain sealed to the historical rank
        # runtime.  Their paths must still exist and their digest fields must
        # be well-formed, but they are not compared with the current closure:
        # the total-trust profile necessarily changes that closure.
        return preflight

    try:
        _base.SCHEMA = _HISTORICAL_SCHEMA
        _base.PHASE_SPECS = copy.deepcopy(_HISTORICAL_PHASE_SPECS)
        _base._verify_current_preflight = _sealed_historical_preflight
        return _TOTAL_VALIDATE_RANK_INITIAL(initial_checkpoint, initial_audit)
    finally:
        _base.SCHEMA = saved_schema
        _base.PHASE_SPECS = saved_specs
        _base._verify_current_preflight = saved_preflight_validator


_base._validate_rank_initial = _validate_rank_initial_compat

_TOTAL_STABLE_PREFLIGHT_PAYLOAD = _base._stable_preflight_payload


def _stable_preflight_payload_compat(**kwargs):
    payload = _TOTAL_STABLE_PREFLIGHT_PAYLOAD(**kwargs)
    initial_audit = kwargs.get("initial_audit")
    if kwargs.get("phase") == "confidence" and initial_audit is not None:
        audit = _base.read_json(_base.resolve_path(initial_audit))
        if audit.get("schema") == _LEGACY_REPLAY_SCHEMA:
            payload["launch"]["initialization"] = LEGACY_R100_INITIALIZATION
    return payload


_base._stable_preflight_payload = _stable_preflight_payload_compat


if __name__ == "__main__":
    _base.main()
