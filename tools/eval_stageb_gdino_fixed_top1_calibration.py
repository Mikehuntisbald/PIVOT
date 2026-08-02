#!/usr/bin/env python3
"""Evaluate one fixed-top1 checkpoint on the sealed calibration manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_refcoco_stageb import _load_model  # noqa: E402
from tools.eval_text_groundingdino_refcoco_tn import evaluate_tn_dataset  # noqa: E402
from tools.eval_stageb_tn_val import (  # noqa: E402
    _make_datasetinfo,
    _validate_adapter_tn_eval_manifest,
)
from tools.stageb_dependency_audit import (  # noqa: E402
    DependencyAuditError,
    config_import_chain,
)
from tools.stageb_gdino_adapter_probe_audit import file_record  # noqa: E402
from tools.stageb_gdino_fixed_top1_selection import (  # noqa: E402
    CALIBRATION_COMPLETION_SCHEMA,
    CALIBRATION_RUNTIME,
    CALIBRATION_SCORE_CONTRACT,
    DEFAULT_DATA_ROOT,
    SelectionError,
    _atomic_write_json,
    _iter_jsonl,
    calibration_code_records,
    read_json,
    replay_calibration_checkpoint_verification,
    verify_calibration_completion,
    verify_partition,
)
from util.slconfig import SLConfig  # noqa: E402


RUNTIME = dict(CALIBRATION_RUNTIME)
SCORE_CONTRACT = dict(CALIBRATION_SCORE_CONTRACT)


class CalibrationError(RuntimeError):
    pass


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _meta_rows(rows: list[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    result = []
    for index, row in enumerate(rows):
        split = str(
            row.get("eval_split")
            or row.get("tn_eval_split")
            or row.get("split")
            or ""
        ).strip()
        if not split:
            raise CalibrationError(f"calibration row {index} has no split")
        category = row.get("replace_category", "unknown")
        if isinstance(category, list):
            category = "+".join(str(value) for value in category)
        result.append(
            {
                "eval_split": split,
                "category": str(category),
                "pair_source": str(row.get("pair_source", "")),
                "image_id": int(row["image_id"]),
                "ann_id": int(row["ann_id"]),
                "ref_id": int(row["ref_id"]),
                "sent_id": int(row["sent_id"]),
                "sample_id": str(row["sample_id"]),
            }
        )
    return result


def _checkpoint_verification(
    *,
    role: str,
    checkpoint: Path,
    checkpoint_audit: Path,
    baseline_checkpoint: Path | None,
    milestone_iteration: int | None,
) -> Dict[str, Any]:
    return replay_calibration_checkpoint_verification(
        role=role,
        checkpoint=checkpoint,
        checkpoint_audit=checkpoint_audit,
        baseline_checkpoint=baseline_checkpoint,
        milestone_iteration=milestone_iteration,
    )


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    if "GFLOPS_DEBUG_SHILONG" in os.environ:
        raise CalibrationError(
            "GFLOPS_DEBUG_SHILONG is forbidden for sealed deploy-transform evaluation"
        )
    if int(args.num_workers) < 0:
        raise CalibrationError("num_workers must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise CalibrationError("sealed calibration requires CUDA with effective AMP")
    if not torch.cuda.is_available():
        raise CalibrationError("CUDA calibration requested but CUDA is unavailable")
    torch.cuda.set_device(device)
    device_index = int(torch.cuda.current_device() if device.index is None else device.index)
    properties = torch.cuda.get_device_properties(device_index)
    data_root = _resolve(args.data_root)
    if data_root != DEFAULT_DATA_ROOT.resolve():
        raise CalibrationError("calibration data root differs from the partition image-identity root")
    image_root = (data_root / "COCO/coco2014/train2014").resolve()
    if not image_root.is_dir():
        raise CalibrationError(f"calibration image root is missing: {image_root}")
    runtime_actual = {
        "device": str(device),
        "device_type": device.type,
        "cuda_device_index": device_index,
        "cuda_device_name": str(properties.name),
        "cuda_device_capability": [int(properties.major), int(properties.minor)],
        "effective_amp": True,
        "num_workers": int(args.num_workers),
        "data_root": str(data_root),
        "image_root": str(image_root),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "environment": {"GFLOPS_DEBUG_SHILONG": None},
    }
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CalibrationError(f"calibration output directory is not empty: {output_dir}")
    checkpoint = _resolve(args.checkpoint)
    checkpoint_audit = _resolve(args.checkpoint_audit)
    partition = verify_partition(_resolve(args.partition_audit))
    if partition["selection_readiness"].get("pass") is not True:
        raise CalibrationError("partition is not ready for held-out selection")
    manifest_record = dict(partition["calibration"])
    manifest = Path(manifest_record["path"])
    probe_preflight = read_json(_resolve(args.probe_preflight))
    static = probe_preflight.get("static")
    if not isinstance(static, Mapping):
        raise CalibrationError("probe preflight has no static section")
    config_value = static.get("config")
    if not isinstance(config_value, Mapping) or not config_value.get("path"):
        raise CalibrationError("probe preflight has no config record")
    config = Path(str(config_value["path"])).resolve()
    if file_record(config) != dict(config_value):
        raise CalibrationError("probe evaluation config drifted")
    iteration = int(args.iteration) if args.iteration is not None else None
    verification = _checkpoint_verification(
        role=args.role,
        checkpoint=checkpoint,
        checkpoint_audit=checkpoint_audit,
        baseline_checkpoint=(
            _resolve(args.baseline_checkpoint) if args.baseline_checkpoint else None
        ),
        milestone_iteration=iteration,
    )
    cfg = SLConfig.fromfile(str(config))
    if (
        float(getattr(cfg, "stage_b_gdino_gate_pool_temperature", -1.0)) != 0.01
        or int(getattr(cfg, "stage_b_gdino_gate_topk", -1)) != 3
        or int(getattr(cfg, "num_queries", -1)) != 900
        or max(list(getattr(cfg, "data_aug_scales", []))) != 800
        or int(getattr(cfg, "data_aug_max_size", -1)) != 1333
        or getattr(cfg, "data_aug_scale_overlap", None) is not None
    ):
        raise CalibrationError("fixed-top1 pool/query/deploy geometry contract drifted")
    try:
        config_paths = config_import_chain(config, root=REPO_ROOT)
    except DependencyAuditError as error:
        raise CalibrationError(str(error)) from error
    output_dir.mkdir(parents=True, exist_ok=False)
    preflight_path = output_dir / "calibration_eval_preflight.json"
    preflight = {
        "schema": CALIBRATION_COMPLETION_SCHEMA,
        "kind": "calibration_evaluation_preflight",
        "input_role": args.role,
        "iteration": iteration,
        "checkpoint": file_record(checkpoint),
        "checkpoint_audit": file_record(checkpoint_audit),
        "checkpoint_verification": verification,
        "probe_preflight": file_record(_resolve(args.probe_preflight)),
        "partition_audit": partition["audit"],
        "manifest": manifest_record,
        "config": file_record(config),
        "config_import_chain": [file_record(path) for path in config_paths],
        "runtime": dict(RUNTIME),
        "runtime_actual": runtime_actual,
        "score_contract": dict(SCORE_CONTRACT),
        "code": calibration_code_records(),
        "selection_input_scope": "calibration_only",
        "strict_isolation": {
            "strict_metric_inputs": [],
            "strict_result_paths": [],
            "strict_paths_consumed_for_scoring": False,
        },
    }
    _atomic_write_json(preflight_path, preflight)
    rows = [row for _line, row in _iter_jsonl(manifest)]
    eval_scope = _validate_adapter_tn_eval_manifest(cfg, rows)
    if eval_scope != "image_global_topk_verified":
        raise CalibrationError(f"unexpected calibration TN scope: {eval_scope!r}")
    datasetinfo = _make_datasetinfo(
        data_root,
        manifest,
        adapter_eval_scope=eval_scope,
        adapter_eval_protocol="adapter_training_pair_schema",
    )
    datasetinfo["root"] = str(image_root)
    datasetinfo["sam3_tn_image_root"] = str(image_root)
    cfg.device = str(device)
    cfg.patch_only = False
    cfg.use_coco_eval = False
    cfg.batch_size = int(RUNTIME["batch_size"])
    cfg.build_text_token_masks = True
    cfg.text_mask_warn_limit = 0
    _set_seed(int(RUNTIME["seed"]))
    model = _load_model(cfg, str(checkpoint), device)
    if int(getattr(model, "num_queries", -1)) != int(SCORE_CONTRACT["query_count"]):
        raise CalibrationError("loaded model does not produce the locked 900-query geometry")
    observed_forward = {"calls": 0, "examples": 0}

    def validate_query_geometry(_module, _inputs, outputs):
        if not isinstance(outputs, Mapping):
            raise CalibrationError("calibration model forward did not return a mapping")
        tensors = {
            key: outputs.get(key)
            for key in (
                "stage_b_gdino_confidence_score",
                "stage_b_gdino_base_score",
                "pred_boxes",
            )
        }
        if any(not torch.is_tensor(value) for value in tensors.values()):
            raise CalibrationError("calibration model output lacks score/box query tensors")
        shapes = {key: tuple(value.shape) for key, value in tensors.items()}
        if any(len(shape) < 2 or int(shape[1]) != 900 for shape in shapes.values()):
            raise CalibrationError(f"calibration forward query geometry drifted: {shapes}")
        batch_sizes = {int(shape[0]) for shape in shapes.values()}
        if len(batch_sizes) != 1:
            raise CalibrationError(f"calibration forward batch geometry drifted: {shapes}")
        observed_forward["calls"] += 1
        observed_forward["examples"] += next(iter(batch_sizes))

    hook = model.register_forward_hook(validate_query_geometry)
    try:
        row = evaluate_tn_dataset(
            cfg=cfg,
            model=model,
            ckpt_path=str(checkpoint),
            datasetinfo=datasetinfo,
            meta_rows=_meta_rows(rows),
            device=device,
            batch_size=int(RUNTIME["batch_size"]),
            num_workers=int(args.num_workers),
            seed=int(RUNTIME["seed"]),
            threshold_tprs=[float(RUNTIME["threshold_tpr"])],
            score_thresholds=[],
            amp=bool(RUNTIME["amp"]),
            max_batches=0,
            log_every=int(args.log_every),
            records_output_dir=output_dir / "per_example_records",
        )
    finally:
        hook.remove()
    del model
    expected_calls = 2 * int(
        math.ceil(int(manifest_record["rows"]) / int(RUNTIME["batch_size"]))
    )
    if (
        observed_forward["calls"] != expected_calls
        or observed_forward["examples"] != 2 * int(manifest_record["rows"])
    ):
        raise CalibrationError(
            "calibration negative/positive forward count differs from full-manifest execution"
        )
    row["query_geometry_evidence"] = {
        "hook": "root_model_forward_hook",
        "checked_outputs": [
            "stage_b_gdino_confidence_score",
            "stage_b_gdino_base_score",
            "pred_boxes",
        ],
        "query_count_each_call": 900,
        "observed_forward_calls": int(observed_forward["calls"]),
        "expected_forward_calls": expected_calls,
        "observed_examples_across_negative_positive": int(
            observed_forward["examples"]
        ),
        "expected_examples_across_negative_positive": 2
        * int(manifest_record["rows"]),
        "pass": True,
    }
    row["runtime_actual"] = runtime_actual
    row["score_contract"] = dict(SCORE_CONTRACT)
    summary_path = output_dir / "summary.json"
    _atomic_write_json(summary_path, row)
    records_path = Path(str(row["records_jsonl"])).resolve()
    if int(row.get("manifest_n", -1)) != int(manifest_record["rows"]):
        raise CalibrationError("calibration evaluator did not cover the full manifest")
    if int(row.get("invalid_records", -1)) != 0:
        raise CalibrationError("calibration evaluator produced invalid records")
    completion_path = output_dir / "calibration_eval_complete.json"
    completion = {
        "schema": CALIBRATION_COMPLETION_SCHEMA,
        "kind": "completed_calibration_evaluation",
        "input_role": args.role,
        "iteration": iteration,
        "selection_input_scope": "calibration_only",
        "strict_isolation": {
            "strict_metric_inputs": [],
            "strict_result_paths": [],
            "strict_paths_consumed_for_scoring": False,
        },
        "checkpoint": file_record(checkpoint),
        "checkpoint_audit": file_record(checkpoint_audit),
        "p0_identity": (
            verification.get("functional_identity") if args.role == "p0" else None
        ),
        "manifest": manifest_record,
        "manifest_rows": int(manifest_record["rows"]),
        "preflight": file_record(preflight_path),
        "summary": file_record(summary_path),
        "records": file_record(records_path),
    }
    _atomic_write_json(completion_path, completion)
    verify_calibration_completion(
        completion_path,
        expected_manifest=manifest_record,
        expected_role=args.role,
        expected_iteration=iteration,
    )
    return completion


def verify_existing(args: argparse.Namespace) -> Dict[str, Any]:
    partition = verify_partition(_resolve(args.partition_audit))
    iteration = int(args.iteration) if args.iteration is not None else None
    completion_path = _resolve(args.output_dir) / "calibration_eval_complete.json"
    result = verify_calibration_completion(
        completion_path,
        expected_manifest=partition["calibration"],
        expected_role=args.role,
        expected_iteration=iteration,
    )
    if result["checkpoint"] != file_record(_resolve(args.checkpoint)):
        raise CalibrationError("calibration completion belongs to a different checkpoint")
    if result["checkpoint_audit"] != file_record(_resolve(args.checkpoint_audit)):
        raise CalibrationError("calibration completion belongs to a different checkpoint audit")
    verification = _checkpoint_verification(
        role=args.role,
        checkpoint=_resolve(args.checkpoint),
        checkpoint_audit=_resolve(args.checkpoint_audit),
        baseline_checkpoint=(
            _resolve(args.baseline_checkpoint) if args.baseline_checkpoint else None
        ),
        milestone_iteration=iteration,
    )
    preflight = read_json(Path(result["preflight"]["path"]))
    if preflight.get("probe_preflight") != file_record(_resolve(args.probe_preflight)):
        raise CalibrationError("calibration completion belongs to a different probe preflight")
    if preflight.get("partition_audit") != partition["audit"]:
        raise CalibrationError("calibration completion belongs to a different partition")
    if preflight.get("checkpoint_verification") != verification:
        raise CalibrationError("calibration checkpoint verification replay drifted")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--role", choices=("p0", "milestone"), required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--probe-preflight", required=True)
    parser.add_argument("--partition-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="/home/user/datasets/pivot_data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.role == "p0":
            if args.iteration is not None or not args.baseline_checkpoint:
                raise CalibrationError("P0 requires --baseline-checkpoint and forbids --iteration")
        elif args.iteration is None or args.baseline_checkpoint:
            raise CalibrationError("milestone requires --iteration and forbids --baseline-checkpoint")
        result = verify_existing(args) if args.verify_only else run_evaluation(args)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    except (
        CalibrationError,
        SelectionError,
        DependencyAuditError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
