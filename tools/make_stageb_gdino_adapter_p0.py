#!/usr/bin/env python3
"""Create or verify an evaluation-only zero-init adapter checkpoint (P0)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    StageBGDINOScoreAdapter,
)
from tools.stageb_gdino_adapter_probe_audit import (  # noqa: E402
    ADAPTER_PREFIX,
    PHASE_SPECS,
    ProbeAuditError,
    _validate_fixed_baseline,
    checkpoint_record,
    file_record,
    load_checkpoint,
    read_json,
    resolve_path,
    sha256_file,
    validate_phase_static,
    write_json,
)
from util.slconfig import SLConfig  # noqa: E402


P0_SCHEMA = "stageb-gdino-adapter-p0-v1"
DEFAULT_CONFIG = PHASE_SPECS["confidence"]["config"]


def build_adapter_from_config(config: Path, *, seed: int) -> StageBGDINOScoreAdapter:
    cfg = SLConfig.fromfile(str(config))
    torch.manual_seed(int(seed))
    adapter = StageBGDINOScoreAdapter(
        hidden_dim=int(getattr(cfg, "hidden_dim")),
        adapter_dim=int(getattr(cfg, "stage_b_gdino_adapter_dim", 128)),
        gate_hidden_dim=int(getattr(cfg, "stage_b_gdino_gate_hidden_dim", 128)),
        gate_pool_temperature=float(
            getattr(cfg, "stage_b_gdino_gate_pool_temperature", 0.1)
        ),
        gate_topk=int(getattr(cfg, "stage_b_gdino_gate_topk", 10)),
    )
    return adapter


def build_p0_model_state(
    baseline_state: Mapping[str, Any],
    adapter: StageBGDINOScoreAdapter,
) -> OrderedDict[str, Any]:
    if any(str(key).startswith(ADAPTER_PREFIX) for key in baseline_state):
        raise ProbeAuditError("P0 source baseline already contains adapter parameters")
    merged = OrderedDict((str(key), value) for key, value in baseline_state.items())
    for key, value in adapter.state_dict().items():
        merged[ADAPTER_PREFIX + str(key)] = value.detach().cpu()
    return merged


def _functional_identity_check(adapter: StageBGDINOScoreAdapter) -> None:
    generator = torch.Generator(device="cpu").manual_seed(1729)
    query = torch.randn(
        3,
        11,
        adapter.hidden_dim,
        generator=generator,
        dtype=torch.float32,
    )
    base = torch.rand(3, 11, generator=generator, dtype=torch.float32)
    mask = torch.ones_like(base, dtype=torch.bool)
    mask[0, -2:] = False
    output = adapter(query, base, mask)
    if not torch.equal(output["rank_score"], base):
        raise ProbeAuditError("P0 rank score is not bitwise identical to the base score")
    if not torch.equal(output["confidence_score"], base):
        raise ProbeAuditError("P0 confidence score is not bitwise identical to the base score")
    if int(torch.count_nonzero(output["rank_residual"]).item()) != 0:
        raise ProbeAuditError("P0 rank residual is not exactly zero")
    if int(torch.count_nonzero(output["confidence_gate"]).item()) != 0:
        raise ProbeAuditError("P0 confidence gate is not exactly zero")


def verify_p0(
    *,
    baseline_checkpoint: Path,
    p0_checkpoint: Path,
    config: Path,
) -> Dict[str, Any]:
    validate_phase_static("confidence")
    expected_config = resolve_path(DEFAULT_CONFIG)
    if config.resolve() != expected_config:
        raise ProbeAuditError(
            f"P0 parity config must be the confidence/evaluation config: {expected_config}"
        )
    baseline_record = _validate_fixed_baseline(baseline_checkpoint)
    p0_record = checkpoint_record(p0_checkpoint)
    if p0_record.get("base_model_sha256") != baseline_record.get("base_model_sha256"):
        raise ProbeAuditError("P0 checkpoint changed pure baseline model tensors")
    if p0_record.get("rank_final_zero") is not True:
        raise ProbeAuditError("P0 rank output layer is not exactly zero")
    if p0_record.get("confidence_final_zero") is not True:
        raise ProbeAuditError("P0 confidence output layer is not exactly zero")

    payload = load_checkpoint(p0_checkpoint)
    metadata = payload.get("p0_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != P0_SCHEMA:
        raise ProbeAuditError("P0 checkpoint is missing its evaluation-only metadata")
    if metadata.get("source_baseline_sha256") != baseline_record.get("sha256"):
        raise ProbeAuditError("P0 metadata does not link to the fixed baseline file hash")
    if metadata.get("config_sha256") != sha256_file(config):
        raise ProbeAuditError("P0 metadata config hash drifted")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ProbeAuditError("P0 checkpoint has no model state")
    adapter = build_adapter_from_config(config, seed=int(metadata.get("seed", -1)))
    adapter_state = {
        str(key).removeprefix(ADAPTER_PREFIX): value
        for key, value in state.items()
        if str(key).startswith(ADAPTER_PREFIX)
    }
    adapter.load_state_dict(adapter_state, strict=True)
    adapter.eval()
    _functional_identity_check(adapter)
    del payload
    return {
        "schema": P0_SCHEMA,
        "kind": "p0_checkpoint_audit",
        "baseline": baseline_record,
        "config": file_record(config),
        "p0_checkpoint": p0_record,
        "functional_identity": {
            "rank_score_equals_base": True,
            "confidence_score_equals_base": True,
            "rank_residual_exact_zero": True,
            "confidence_gate_exact_zero": True,
        },
        "intended_use": "evaluation_only_same_records_parity",
    }


def verify_p0_sidecar(
    *,
    p0_checkpoint: Path,
    audit: Mapping[str, Any],
    sidecar: Path,
) -> Dict[str, Any]:
    recorded = read_json(sidecar)
    if recorded != dict(audit):
        raise ProbeAuditError(
            f"P0 sidecar does not exactly match the recomputed checkpoint audit: {sidecar}"
        )
    return {
        "schema": P0_SCHEMA,
        "kind": "p0_checkpoint_and_sidecar_verified",
        "checkpoint": file_record(p0_checkpoint),
        "sidecar": file_record(sidecar),
        "audit": dict(audit),
    }


def create_p0(
    *,
    baseline_checkpoint: Path,
    output: Path,
    config: Path,
    seed: int,
) -> Dict[str, Any]:
    if output.exists():
        raise ProbeAuditError(f"refusing to overwrite P0 checkpoint: {output}")
    sidecar = Path(str(output) + ".audit.json")
    if sidecar.exists():
        raise ProbeAuditError(f"refusing to overwrite P0 audit: {sidecar}")
    baseline_record = _validate_fixed_baseline(baseline_checkpoint)
    validate_phase_static("confidence")
    if config.resolve() != resolve_path(DEFAULT_CONFIG):
        raise ProbeAuditError("P0 must use the fixed confidence/evaluation config")
    baseline_payload = load_checkpoint(baseline_checkpoint)
    baseline_state = baseline_payload.get("model")
    if not isinstance(baseline_state, Mapping):
        raise ProbeAuditError("fixed baseline checkpoint has no model state")
    adapter = build_adapter_from_config(config, seed=seed)
    adapter.eval()
    _functional_identity_check(adapter)
    merged = build_p0_model_state(baseline_state, adapter)
    checkpoint_payload: Dict[str, Any] = {
        "model": merged,
        "args": {
            "config_file": str(config),
            "pretrain_model_path": str(baseline_checkpoint),
            "resume": "",
            "stage_b_gdino_score_adapter": True,
            "stage_b_gdino_adapter_train_mode": "confidence_only",
            "stage_b_gdino_tn_scope": "benchmark_dataft_alltn",
            "p0_eval_only": True,
        },
        "p0_metadata": {
            "schema": P0_SCHEMA,
            "seed": int(seed),
            "source_baseline": str(baseline_checkpoint),
            "source_baseline_sha256": baseline_record["sha256"],
            "config": str(config),
            "config_sha256": sha256_file(config),
            "training_allowed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    try:
        torch.save(checkpoint_payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    del checkpoint_payload, merged, baseline_payload
    audit = verify_p0(
        baseline_checkpoint=baseline_checkpoint,
        p0_checkpoint=output,
        config=config,
    )
    write_json(sidecar, audit)
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--baseline-checkpoint", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--config", default=DEFAULT_CONFIG)
        if name == "create":
            command.add_argument("--seed", type=int, default=42)
        else:
            command.add_argument("--output")
            command.add_argument(
                "--sidecar",
                help="P0 sidecar audit (default: CHECKPOINT.audit.json)",
            )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        baseline = resolve_path(args.baseline_checkpoint)
        checkpoint = resolve_path(args.checkpoint)
        config = resolve_path(args.config)
        if args.command == "create":
            audit = create_p0(
                baseline_checkpoint=baseline,
                output=checkpoint,
                config=config,
                seed=int(args.seed),
            )
            output = Path(str(checkpoint) + ".audit.json")
            print(f"[OK] created zero-init P0 checkpoint: {checkpoint}")
            print(f"[OK] wrote P0 audit: {output}")
        else:
            audit = verify_p0(
                baseline_checkpoint=baseline,
                p0_checkpoint=checkpoint,
                config=config,
            )
            sidecar = (
                resolve_path(args.sidecar)
                if args.sidecar
                else Path(str(checkpoint) + ".audit.json")
            )
            verification = verify_p0_sidecar(
                p0_checkpoint=checkpoint,
                audit=audit,
                sidecar=sidecar,
            )
            if args.output:
                write_json(resolve_path(args.output), verification)
            print(json.dumps(verification, indent=2, sort_keys=True, ensure_ascii=True))
    except ProbeAuditError as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
