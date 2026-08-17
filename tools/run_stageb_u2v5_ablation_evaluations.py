#!/usr/bin/env python3
"""Run val3/calibration mechanism profiles for completed U2-v5 rows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stageb_u2v5_ablation_registry import SEEDS, get_row
from tools.stageb_u2v4_legacy_training_contract import AUXILIARY_RESIDUAL_KEYS
import torch


PYTHON = Path("/home/haoyi/miniconda/envs/cvpr/bin/python")
TRAINING = ROOT / "outputs/u2v5_cvpr_ablation_20260817/training"
EVALUATIONS = ROOT / "outputs/u2v5_cvpr_ablation_20260817/evaluations/mechanism"
EVAL = "tools/eval_text_groundingdino_refcoco_tn.py"
AUDIT = "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
MATCHED_AUDIT = "data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/audit.json"


def _checkpoints(row_id: str, *, require: bool = True) -> list[str]:
    paths = [TRAINING / row_id / f"seed{seed}/checkpoint_iter.pth" for seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing and require:
        raise FileNotFoundError(f"row {row_id} lacks checkpoints: {missing}")
    return [str(path) for path in paths]


def _base(row_id: str, profile: str, *, require: bool = True) -> list[str]:
    return [str(PYTHON), EVAL, "--ckpts", *_checkpoints(row_id, require=require), "--output_dir", str(EVALUATIONS / row_id / profile), "--data_root", "/media/haoyi/T9/data", "--device", "cuda:0", "--batch_size", "16", "--num_workers", "4", "--seed", "42", "--amp", "--holdout_level", "none"]


def commands(row_id: str, *, require: bool = True) -> list[list[str]]:
    row = get_row(row_id)
    if not row.formal_training:
        raise ValueError(f"row {row_id} is not a new formal row")
    if row.block == "A":
        if row_id == "A2":
            return []
        return [[
            *_base(row_id, "val3", require=require),
            "--config", "config/ablations/cfg_stageb_u2v5_ablation_admission_eval_gap3.py",
            "--ref_splits", "refcoco_val", "refcocop_val", "refcocog_val",
            "--skip_tn", "--topk", "1",
        ]]
    if row.block == "C":
        return [[
            *_base(row_id, "d3_calibration", require=require), "--config", str(row.config),
            "--tn_jsonl", "data/ablations/stageb_tn_table_b_equal_exposure_20260717/d3_proposal_covered_calibration.jsonl",
            "--screen_calibration_manifest", "--screen_calibration_audit", AUDIT,
            "--skip_ref",
        ]]
    if row.block == "D":
        source_map = {
            "D1": ("data/ablations/stageb_tn_table_b_equal_exposure_20260717/d1_unverified_allneg_calibration.jsonl", AUDIT),
            "D2": ("data/ablations/stageb_tn_table_b_equal_exposure_20260717/d2_traceable_edit_calibration.jsonl", AUDIT),
            "D2m": ("data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/d2m_traceable_edit_calibration.jsonl", MATCHED_AUDIT),
            "D3m": ("data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/d3m_proposal_covered_calibration.jsonl", MATCHED_AUDIT),
        }
        source, audit = source_map[row_id]
        return [[
            *_base(row_id, "calibration", require=require), "--config", str(row.config),
            "--tn_jsonl", source,
            "--u2v5_ablation_calibration_manifest",
            "--u2v5_ablation_calibration_audit", audit,
            "--skip_ref",
        ]]
    if row.block == "O":
        config = f"config/ablations/cfg_stageb_u2v5_ablation_{row_id.lower()}_eval.py"
        return [
            [
                *_base(row_id, "val3", require=require), "--config", config,
                "--ref_splits", "refcoco_val", "refcocop_val", "refcocog_val",
                "--skip_tn", "--topk", "1",
            ],
            [
                *_base(row_id, "d3_calibration", require=require), "--config", config,
                "--tn_jsonl", "data/ablations/stageb_tn_table_b_equal_exposure_20260717/d3_proposal_covered_calibration.jsonl",
                "--screen_calibration_manifest", "--screen_calibration_audit", AUDIT,
                "--skip_ref",
            ],
        ]
    raise ValueError(f"row {row_id} has no mechanism evaluation profile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dry-run", "run"])
    parser.add_argument("--row-id", required=True)
    args = parser.parse_args()
    planned = commands(args.row_id, require=args.command == "run")
    if args.command == "dry-run":
        print(json.dumps({"row_id": args.row_id, "commands": planned}, indent=2))
        return
    env = dict(os.environ)
    env.setdefault("DATA_ROOT", "/media/haoyi/T9/data")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.row_id == "A2":
        output = EVALUATIONS / "A2" / "deployment_parity.json"
        if output.exists():
            raise FileExistsError(f"parity output must be fresh: {output}")
        initializer = torch.load(
            ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/initializer/checkpoint_clean_init.pth",
            map_location="cpu", weights_only=False,
        )["model"]
        deploy_keys = sorted(set(initializer) - set(AUXILIARY_RESIDUAL_KEYS))
        rows = []
        for seed, checkpoint_path in zip(SEEDS, _checkpoints("A2")):
            state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )["model"]
            unequal = [
                key for key in deploy_keys
                if not torch.equal(state[key], initializer[key])
            ]
            if unequal:
                raise RuntimeError(
                    f"A2 seed{seed} changed deployed tensors: {unequal[:8]}"
                )
            digest = hashlib.sha256()
            for key in deploy_keys:
                digest.update(key.encode("utf-8"))
                digest.update(state[key].contiguous().numpy().tobytes())
            rows.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint_path,
                    "deploy_tensor_count": len(deploy_keys),
                    "deploy_tensor_sha256": digest.hexdigest(),
                    "bitwise_equal_initializer": True,
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema": "pivot.stageb.u2v5_a2_deployment_parity/v1",
                    "auxiliary_residual_deployed": False,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    for command in planned:
        output = Path(command[command.index("--output_dir") + 1])
        if output.exists():
            raise FileExistsError(f"evaluation output must be fresh: {output}")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        summary = output / "summary.json"
        if not summary.is_file():
            raise RuntimeError(f"evaluation did not produce {summary}")


if __name__ == "__main__":
    main()
