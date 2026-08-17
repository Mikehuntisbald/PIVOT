#!/usr/bin/env python3
"""Execute the preregistered U2-v5 Test5/strict2031 contrasts once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/haoyi/miniconda/envs/cvpr/bin/python")
EVAL = "tools/eval_text_groundingdino_refcoco_tn.py"
TRAINING = ROOT / "outputs/u2v5_cvpr_ablation_20260817/training"
STRICT = "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/eval_manifest.jsonl"
TEST5 = ["refcoco_testA", "refcoco_testB", "refcocop_testA", "refcocop_testB", "refcocog_test"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _ckpts(row: str) -> list[str]:
    paths = [TRAINING / row / f"seed{seed}/checkpoint_iter.pth" for seed in (17, 42, 73)]
    for path in paths:
        path.resolve(strict=True)
    return [str(path) for path in paths]


def _run(command: list[str], output: Path, env: dict[str, str]) -> Path:
    if output.exists():
        raise FileExistsError(f"confirmatory output must be fresh: {output}")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    summary = output / "summary.json"
    if not summary.is_file():
        raise RuntimeError(f"confirmatory evaluation lacks {summary}")
    return summary


def _test5(row: str, config: str, root: Path, env: dict[str, str]) -> Path:
    output = root / row / "test5"
    command = [
        str(PYTHON), EVAL, "--config", config, "--ckpts", *_ckpts(row),
        "--output_dir", str(output), "--data_root", "/media/haoyi/T9/data",
        "--device", "cuda:0", "--batch_size", "16", "--num_workers", "4",
        "--seed", "42", "--amp", "--ref_splits", *TEST5,
        "--skip_tn", "--topk", "1", "--holdout_level", "none",
    ]
    return _run(command, output, env)


def _strict(row: str, config: str, root: Path, env: dict[str, str]) -> Path:
    output = root / row / "strict2031"
    command = [
        str(PYTHON), EVAL, "--config", config, "--ckpts", *_ckpts(row),
        "--output_dir", str(output), "--data_root", "/media/haoyi/T9/data",
        "--tn_jsonl", STRICT, "--device", "cuda:0", "--batch_size", "16",
        "--num_workers", "4", "--seed", "42", "--amp", "--skip_ref",
        "--tn_splits", "refcocop_val", "refcocog_umd_val",
        "--holdout_level", "none",
    ]
    return _run(command, output, env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    args = parser.parse_args()
    prereg_path = Path(args.preregistration).resolve(strict=True)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("schema") != "pivot.stageb.u2v5_ablation_preregistration/v1" or prereg.get("status") != "locked_before_confirmatory_evaluation":
        raise RuntimeError("invalid ablation preregistration")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty or current != prereg["git"]["commit"]:
        raise RuntimeError("confirmatory evaluation code differs from preregistration")
    final_root = Path(prereg["final_output_root"])
    if final_root.exists():
        raise FileExistsError(f"confirmatory root already exists: {final_root}")
    env = dict(os.environ)
    env.setdefault("DATA_ROOT", "/media/haoyi/T9/data")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    outputs = {}
    outputs["A1_test5"] = _test5("A1", "config/ablations/cfg_stageb_u2v5_ablation_admission_eval_gap3.py", final_root, env)
    outputs["C2_strict2031"] = _strict("C2", "config/ablations/cfg_stageb_u2v5_ablation_c2_no_positive_trust.py", final_root, env)
    for row in ("O0", "O2"):
        config = f"config/ablations/cfg_stageb_u2v5_ablation_{row.lower()}_eval.py"
        outputs[f"{row}_test5"] = _test5(row, config, final_root, env)
        outputs[f"{row}_strict2031"] = _strict(row, config, final_root, env)
    for row in ("D2m", "D3m"):
        config = f"config/ablations/cfg_stageb_u2v5_ablation_{row.lower()}_matched.py"
        outputs[f"{row}_strict2031"] = _strict(row, config, final_root, env)
    reused = {
        "A5_O3_test5": ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/ref8_u50/summary.json",
        "C3_O3_strict2031": ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/strict2031_u50/summary.json",
    }
    manifest = {
        "schema": "pivot.stageb.u2v5_ablation_confirmatory_results/v1",
        "preregistration": _record(prereg_path),
        "new_summaries": {key: _record(path) for key, path in outputs.items()},
        "reused_anchor_summaries": {key: _record(path) for key, path in reused.items()},
        "strict1607_forwarded_separately": False,
        "status": "complete",
    }
    manifest_path = final_root / "results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": _record(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
