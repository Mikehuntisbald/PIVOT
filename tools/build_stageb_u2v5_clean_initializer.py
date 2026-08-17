#!/usr/bin/env python3
"""Build the leakage-clean Stage-A + positive-only R100 admission initializer.

The builder deliberately has no C100 input.  The twelve confidence tensors are
copied from the sealed R100 checkpoint where the Stage-A/R100 receipt proves
that they are still at their identity initialization.  The legacy U0 shell is
copied only to restore the admission auxiliary initialization used by U2-v4.
"""

from __future__ import annotations

import argparse
import json
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
from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    stage_b_u0_tensor_state_sha256,
)
from tools.build_stageb_u0_initializer import verify_initializer as verify_u0  # noqa: E402
from tools.stageb_gdino_adapter_probe_audit import file_record, load_checkpoint  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.u2v5_clean_initializer/v1"
RECEIPT_SCHEMA = "pivot.stagea_b58_r100_receipt/v3"
DEFAULT_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_u2v2_c0.py"
U0_PREFIX = "stage_b_u0_patch_rank_adapter."
ADAPTER_PREFIX = "stage_b_gdino_score_adapter."
RANK_PREFIXES = (
    ADAPTER_PREFIX + "rank_norm.",
    ADAPTER_PREFIX + "rank_trunk.",
    ADAPTER_PREFIX + "rank_output.",
)
CONFIDENCE_PREFIXES = (
    ADAPTER_PREFIX + "confidence_norm.",
    ADAPTER_PREFIX + "confidence_trunk.",
    ADAPTER_PREFIX + "confidence_gate.",
)


class U2V5CleanInitializerError(RuntimeError):
    pass


def _state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise U2V5CleanInitializerError(f"{label} has no tensor-only model state")
    return state


def _load_source(path: Path, sha256: str, *, label: str):
    path = path.resolve(strict=True)
    record = file_record(path)
    if record["sha256"] != str(sha256).strip().lower():
        raise U2V5CleanInitializerError(f"{label} SHA256 mismatch")
    payload = load_checkpoint(path)
    return record, payload, _state(payload, label=label)


def _equal(left: torch.Tensor, right: torch.Tensor, *, key: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
        raise U2V5CleanInitializerError(f"bitwise tensor drift at {key}")


def _read_receipt(
    path: Path,
    *,
    stagea_record: Mapping[str, Any],
    r100_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve(strict=True)
    record = file_record(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise U2V5CleanInitializerError("Stage-A/R100 receipt schema drifted")
    invariants = receipt.get("invariants")
    required = (
        "r100_base_bitwise_equal_stagea_nonpatch_trunk",
        "r100_exactly_100_optimizer_updates",
        "r100_rank_trained_confidence_zero_initialized",
        "stagea_nonpatch_trunk_bitwise_equal_b58",
    )
    if not isinstance(invariants, Mapping) or any(invariants.get(k) is not True for k in required):
        raise U2V5CleanInitializerError("Stage-A/R100 receipt invariants are incomplete")
    if receipt.get("stagea", {}).get("checkpoint") != dict(stagea_record):
        raise U2V5CleanInitializerError("receipt Stage-A source does not match")
    if receipt.get("rank_r100", {}).get("checkpoint") != dict(r100_record):
        raise U2V5CleanInitializerError("receipt R100 source does not match")
    if int(receipt.get("rank_r100", {}).get("iteration", -1)) != 100:
        raise U2V5CleanInitializerError("R100 receipt is not the sealed U100 milestone")
    return record, receipt


def _template(config: Path):
    cfg = SLConfig.fromfile(str(config.resolve(strict=True)))
    cfg.device = "cpu"
    if not bool(getattr(cfg, "stage_b_u2v2", False)):
        raise U2V5CleanInitializerError("config must enable the U2 compatibility stack")
    torch.manual_seed(0)
    model, _criterion, _post = build_model_main(cfg)
    return model.eval()


def build_initializer_payload(
    *,
    stagea_checkpoint: Path,
    stagea_sha256: str,
    r100_checkpoint: Path,
    r100_sha256: str,
    stagea_r100_receipt: Path,
    legacy_u0_initializer: Path,
    legacy_u0_sha256: str,
    config: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    stagea_record, _stagea_payload, stagea = _load_source(
        stagea_checkpoint, stagea_sha256, label="Stage-A"
    )
    r100_record, _r100_payload, r100 = _load_source(
        r100_checkpoint, r100_sha256, label="positive-only R100"
    )
    receipt_record, receipt = _read_receipt(
        stagea_r100_receipt, stagea_record=stagea_record, r100_record=r100_record
    )
    u0_record, _u0_payload, u0 = _load_source(
        legacy_u0_initializer, legacy_u0_sha256, label="legacy U0 initializer"
    )
    u0_verification = verify_u0(legacy_u0_initializer)
    if u0_verification["checkpoint"] != u0_record:
        raise U2V5CleanInitializerError("legacy U0 verification record drifted")

    rank = sorted(key for key in r100 if key.startswith(RANK_PREFIXES))
    confidence = sorted(key for key in r100 if key.startswith(CONFIDENCE_PREFIXES))
    trunk = sorted(set(r100) - set(rank) - set(confidence))
    if (len(trunk), len(rank), len(confidence), len(r100)) != (938, 8, 12, 958):
        raise U2V5CleanInitializerError("R100 ownership must be trunk938/rank8/confidence12")
    shared = sorted(set(stagea) & set(r100))
    patch = sorted(set(stagea) - set(r100))
    if shared != trunk or len(patch) != 196:
        raise U2V5CleanInitializerError("Stage-A/R100 ownership must be shared938/patch196")
    for key in shared:
        _equal(stagea[key], r100[key], key=key)
    patch_backbone = sorted(key for key in patch if key.startswith("patch_encoder.backbone."))
    if len(patch_backbone) != 187:
        raise U2V5CleanInitializerError("Stage-A patch backbone must contain 187 tensors")
    for key in patch_backbone:
        _equal(stagea[key], stagea[key.removeprefix("patch_encoder.")], key=key)
    expected_conf_sha = receipt["rank_r100"].get("confidence_tensor_sha256")
    observed_conf_sha = stage_b_u0_tensor_state_sha256(r100, confidence)
    if expected_conf_sha != observed_conf_sha:
        raise U2V5CleanInitializerError("R100 identity confidence12 hash drifted")

    model = _template(config)
    template = model.state_dict()
    u0_keys = sorted(key for key in template if key.startswith(U0_PREFIX))
    if len(template) != 1165 or len(u0_keys) != 11 or set(u0_keys) - set(u0):
        raise U2V5CleanInitializerError("runtime template/U0 shell drifted")
    output_keys = (U0_PREFIX + "output.weight", U0_PREFIX + "output.bias")
    if any(int(torch.count_nonzero(u0[key])) for key in output_keys):
        raise U2V5CleanInitializerError("legacy U0 output must be exactly zero")

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, wanted in template.items():
        if key in r100:
            value = r100[key]
        elif key in patch:
            value = stagea[key]
        elif key in u0_keys:
            value = u0[key]
        else:
            raise U2V5CleanInitializerError(f"unowned clean tensor: {key}")
        if value.dtype != wanted.dtype or value.shape != wanted.shape:
            raise U2V5CleanInitializerError(f"shape/dtype mismatch at {key}")
        result[key] = value.detach().cpu().clone()
    model.load_state_dict(result, strict=True)

    roles = {
        "trunk": trunk,
        "rank": rank,
        "identity_confidence": confidence,
        "patch": patch,
        "u0_admission_shell": u0_keys,
    }
    contract = {
        "schema": SCHEMA,
        "training_initializer": True,
        "resumable": False,
        "model_state_keys": len(result),
        "sources": {
            "stagea": stagea_record,
            "positive_only_r100": r100_record,
            "stagea_r100_receipt": receipt_record,
            "legacy_u0": u0_record,
        },
        "forbidden_sources": ["c100_confidence12", "legacy_p50_confidence12"],
        "role_keys": roles,
        "role_key_counts": {role: len(keys) for role, keys in roles.items()},
        "role_tensor_sha256": {
            role: stage_b_u0_tensor_state_sha256(result, keys)
            for role, keys in roles.items()
        },
        "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(result, result.keys()),
        "routes": {
            "text": "full_referring_expression",
            "ref": "legacy_u2v4_admission_then_raw_positive_only_r100",
            "confidence": "identity_untrained_requires_fresh_d3_phase",
            "b58_top1_guard": False,
        },
        "invariants": {
            "stagea_r100_trunk938_bitwise": True,
            "positive_only_r100_rank8": True,
            "r100_confidence12_identity_untrained": True,
            "patch196_from_stagea": True,
            "patch_backbone187_equals_main": True,
            "legacy_u0_admission_shell11": True,
            "c100_confidence_imported": False,
        },
    }
    return {"model": result, "u2v5_clean_initializer": contract}


def validate_initializer_payload(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> dict[str, Any]:
    if set(payload) != {"model", "u2v5_clean_initializer"}:
        raise U2V5CleanInitializerError("clean initializer top-level keys drifted")
    state = _state(payload, label="clean initializer")
    contract = payload.get("u2v5_clean_initializer")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise U2V5CleanInitializerError("clean initializer schema drifted")
    roles = contract.get("role_keys")
    hashes = contract.get("role_tensor_sha256")
    if len(state) != 1165 or not isinstance(roles, Mapping) or not isinstance(hashes, Mapping):
        raise U2V5CleanInitializerError("clean initializer role contract is missing")
    partition: list[str] = []
    for role, keys in roles.items():
        if not isinstance(keys, list) or hashes.get(role) != stage_b_u0_tensor_state_sha256(state, keys):
            raise U2V5CleanInitializerError(f"clean role hash drifted: {role}")
        partition.extend(keys)
    if len(partition) != len(set(partition)) or set(partition) != set(state):
        raise U2V5CleanInitializerError("clean roles do not partition model state")
    if contract.get("full_model_tensor_sha256") != stage_b_u0_tensor_state_sha256(state, state.keys()):
        raise U2V5CleanInitializerError("clean full-state hash drifted")
    if contract.get("invariants", {}).get("c100_confidence_imported") is not False:
        raise U2V5CleanInitializerError("clean initializer does not forbid C100 confidence")
    if any("c100" in key.lower() for key in contract.get("sources", {})):
        raise U2V5CleanInitializerError("clean initializer contains a C100 source")
    if verify_sources:
        sources = contract["sources"]
        for source in sources.values():
            path = Path(str(source.get("path", ""))).resolve(strict=True)
            if file_record(path) != dict(source):
                raise U2V5CleanInitializerError("clean initializer source changed")
    return dict(contract)


def validate_confidence_runtime_payload(
    model,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> dict[str, Any]:
    """Validate a phase-2 checkpoint without weakening legacy U2 schemas."""

    state = _state(payload, label=checkpoint_label)
    contract = payload.get("u2v5_clean_confidence")
    if not isinstance(contract, Mapping) or contract.get("schema") != (
        "pivot.stageb.u2v5_clean_confidence_handoff/v1"
    ):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} lacks clean-confidence provenance"
        )
    ablation_row_id = contract.get("ablation_row_id")
    expected_scope = "proposal_covered_verified"
    expected_table = "D3"
    if ablation_row_id is not None:
        from tools.stageb_u2v5_ablation_registry import get_row

        try:
            row = get_row(str(ablation_row_id))
        except KeyError as exc:
            raise U2V5CleanInitializerError(
                f"{checkpoint_label} names an unknown confidence row"
            ) from exc
        if row.phase != "confidence":
            raise U2V5CleanInitializerError(
                f"{checkpoint_label} confidence row has wrong phase"
            )
        if row.row_id in {"D1"}:
            expected_scope, expected_table = "unverified_all_negative", "D1"
        elif row.row_id in {"D2", "D2m"}:
            expected_scope, expected_table = (
                "traceable_counterfactual_edit", row.row_id
            )
        elif row.row_id == "D3m":
            expected_scope, expected_table = "proposal_covered_verified", "D3m"
    if not (
        contract.get("c100_confidence_imported") is False
        and contract.get("scope") == expected_scope
        and contract.get("table_b_id") == expected_table
        and (
            ablation_row_id is None
            or (
                isinstance(contract.get("table_b_audit_sha256"), str)
                and len(contract["table_b_audit_sha256"]) == 64
            )
        )
    ):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} changed the clean confidence ownership/scope"
        )
    args = payload.get("args")
    if isinstance(args, Mapping):
        c100_path = args.get("stage_b_u2v2_c100_checkpoint")
        c100_sha = args.get("stage_b_u2v2_c100_sha256")
    else:
        c100_path = getattr(args, "stage_b_u2v2_c100_checkpoint", None)
        c100_sha = getattr(args, "stage_b_u2v2_c100_sha256", None)
    if c100_path is not None or c100_sha is not None:
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} serializes forbidden C100 provenance"
        )

    frozen_keys = contract.get("frozen_keys")
    confidence_keys = contract.get("identity_confidence_keys")
    if not (
        isinstance(frozen_keys, list)
        and isinstance(confidence_keys, list)
        and len(frozen_keys) == 1153
        and len(confidence_keys) == 12
        and set(frozen_keys).isdisjoint(confidence_keys)
        and set(frozen_keys) | set(confidence_keys) == set(state)
    ):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} is not an exact frozen1153/confidence12 partition"
        )
    if contract.get("frozen_tensor_sha256") != stage_b_u0_tensor_state_sha256(
        state, frozen_keys
    ):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} changed a frozen admission tensor"
        )

    admission = contract.get("admission_checkpoint")
    initializer = contract.get("initializer")
    if not isinstance(admission, Mapping) or not isinstance(initializer, Mapping):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} has incomplete source bindings"
        )
    admission_path = Path(str(admission.get("path", ""))).resolve(strict=True)
    initializer_path = Path(str(initializer.get("path", ""))).resolve(strict=True)
    admission_record = file_record(admission_path)
    if not (
        admission_record["path"] == str(admission_path)
        and admission_record["sha256"] == admission.get("sha256")
    ):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} admission source changed"
        )
    initializer_record = file_record(initializer_path)
    if not (
        initializer_record["path"] == str(initializer_path)
        and initializer_record["sha256"] == initializer.get("sha256")
    ):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} initializer source changed"
        )
    initializer_payload = load_checkpoint(initializer_path)
    initializer_contract = validate_initializer_payload(initializer_payload)
    if initializer.get("schema") != initializer_contract.get("schema"):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} initializer schema binding drifted"
        )
    initial_state = _state(initializer_payload, label="clean initializer")
    if contract.get(
        "identity_confidence_tensor_sha256"
    ) != stage_b_u0_tensor_state_sha256(initial_state, confidence_keys):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} identity confidence binding drifted"
        )
    admission_state = _state(
        load_checkpoint(admission_path), label="clean admission checkpoint"
    )
    for key in frozen_keys:
        if not torch.equal(state[key].detach().cpu(), admission_state[key].detach().cpu()):
            raise U2V5CleanInitializerError(
                f"{checkpoint_label} frozen tensor differs from admission: {key}"
            )

    runtime = contract.get("runtime_audit")
    if not (
        isinstance(runtime, Mapping)
        and int(runtime.get("successful_optimizer_steps", 0)) > 0
        and runtime.get("successful_optimizer_steps")
        == payload.get("optimizer_updates")
        and runtime.get("amp_skipped_optimizer_steps") == 0
        and runtime.get("nonfinite_gradient_boundaries") == 0
        and runtime.get("zero_gradient_successful_steps") == 0
    ):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} has an invalid confidence runtime audit"
        )
    model_state = model.state_dict()
    if set(model_state) != set(state):
        raise U2V5CleanInitializerError(
            f"{checkpoint_label} model keyset differs at runtime"
        )
    return dict(contract)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise U2V5CleanInitializerError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--stagea-checkpoint", required=True)
    create.add_argument("--stagea-sha256", required=True)
    create.add_argument("--r100-checkpoint", required=True)
    create.add_argument("--r100-sha256", required=True)
    create.add_argument("--stagea-r100-receipt", required=True)
    create.add_argument("--legacy-u0-initializer", required=True)
    create.add_argument("--legacy-u0-sha256", required=True)
    create.add_argument("--config", default=str(DEFAULT_CONFIG))
    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "create":
        payload = build_initializer_payload(
            stagea_checkpoint=Path(args.stagea_checkpoint),
            stagea_sha256=args.stagea_sha256,
            r100_checkpoint=Path(args.r100_checkpoint),
            r100_sha256=args.r100_sha256,
            stagea_r100_receipt=Path(args.stagea_r100_receipt),
            legacy_u0_initializer=Path(args.legacy_u0_initializer),
            legacy_u0_sha256=args.legacy_u0_sha256,
            config=Path(args.config),
        )
        _write(Path(args.output), payload)
        checkpoint = Path(args.output).resolve(strict=True)
    else:
        checkpoint = Path(args.checkpoint).resolve(strict=True)
    contract = validate_initializer_payload(load_checkpoint(checkpoint))
    print(json.dumps({"status": "verified", "checkpoint": file_record(checkpoint), "contract": contract}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, U2V5CleanInitializerError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
