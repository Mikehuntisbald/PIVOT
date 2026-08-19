#!/usr/bin/env python3
"""Measure ARROW inference efficiency without changing model weights.

The formal protocol uses 200 fixed, unique FineCops source images, batch size
one, 50 warm-up forwards, and one timed forward per image.  Images and support
patches are decoded/transformed before timing.  CUDA events include tensor
transfer and model execution but exclude file I/O, metric computation, and
visualization.  Visual-support caching is measured both with the support
encoder inside the timed forward and with its frozen embedding resident on
the GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_arrow_finecops import _apply_route_admission_inputs, _datasetinfo
from tools.eval_stageb_tn_val import _prepare_stage_b_u0_patch_batch
from tools.eval_text_groundingdino_refcoco_tn import (
    _build_loader,
    _forward_ref_batch,
    _load_model,
    _load_model_with_checkpoint_contract,
)
from util.slconfig import SLConfig


PREREG = ROOT / "outputs/arrow_finecops_20260819/preregistration.json"
BASE_CONFIG = ROOT / "config/ablations/cfg_stageb_from_gdino_ft_with_tn.py"
BASE_CHECKPOINT = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/checkpoint0001.pth"
)
RANK_CONFIG = ROOT / "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py"
RANK_CHECKPOINT = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/initializer/checkpoint_clean_init.pth"
FORMAL_OUTPUT = ROOT / "paper/data/efficiency_receipt.json"
MANIFEST_OUTPUT = ROOT / "paper/data/efficiency_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    try:
        display = str(path.relative_to(ROOT))
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": sha256(path), "size_bytes": path.stat().st_size}


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RankOnlyScoreAdapter(nn.Module):
    """Inference-only wrapper over the sealed rank owner.

    It executes exactly the original rank path and returns identity placeholders
    for confidence fields that the surrounding model serializes but does not
    consume in this route.  A preflight requires bitwise rank-score parity.
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.hidden_dim = int(source.hidden_dim)
        self.rank_norm = source.rank_norm
        self.rank_trunk = source.rank_trunk
        self.rank_output = source.rank_output

    def forward(
        self,
        query_hs: torch.Tensor,
        base_score: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = (
            torch.ones_like(base_score, dtype=torch.bool)
            if candidate_mask is None
            else candidate_mask.to(device=base_score.device, dtype=torch.bool)
        )
        hs = query_hs.detach()
        base = base_score.detach().to(device=hs.device)
        safe_base = base.masked_fill(~mask, 0.0).to(dtype=hs.dtype)
        rank_input = torch.cat((self.rank_norm(hs), safe_base.unsqueeze(-1)), dim=-1)
        feature = self.rank_trunk(rank_input)
        residual = self.rank_output(feature).squeeze(-1).to(dtype=base.dtype)
        residual = residual.masked_fill(~mask, 0.0)
        score = base + residual
        gate = base.new_zeros((base.shape[0],))
        return {
            "base_score": base,
            "rank_feature": feature,
            "rank_residual": residual,
            "rank_score": score,
            "confidence_feature": feature.new_zeros(feature.shape),
            "confidence_gate": gate,
            "confidence_score": base,
            "candidate_mask": mask,
        }


def config(path: Path, device: torch.device) -> Any:
    cfg = SLConfig.fromfile(str(path))
    cfg.device = str(device)
    cfg.aux_loss = False
    cfg.use_checkpoint = False
    cfg.use_transformer_ckpt = False
    return cfg


def fixed_rows(source_a: Path, source_b: Path, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_a = jsonl(source_a)
    selected: list[dict[str, Any]] = []
    seen_images: set[int] = set()
    for row in rows_a:
        if row.get("finecops_kind") != "positive" or not row.get("finecops_support_covered"):
            continue
        cluster = int(row["finecops_cluster_gqa_image_id"])
        if cluster in seen_images:
            continue
        selected.append(row)
        seen_images.add(cluster)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only found {len(selected)} unique supported positive images")
    ids = [row["sample_id"] for row in selected]
    rows_b_by_id = {row["sample_id"]: row for row in jsonl(source_b)}
    selected_b = [rows_b_by_id[sample_id] for sample_id in ids]
    for row_a, row_b in zip(selected, selected_b):
        stable = ("sample_id", "filename", "finecops_expression", "finecops_cluster_gqa_image_id")
        if any(row_a[key] != row_b[key] for key in stable):
            raise RuntimeError("A/B efficiency manifests do not share the input universe")
    return selected, selected_b


def loader_batches(
    cfg: Any,
    dataset_payload: Mapping[str, Any],
    route: str,
    rows: list[dict[str, Any]],
    directory: Path,
    device: torch.device,
) -> list[Any]:
    manifest = directory / f"efficiency_{route.lower()}.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    loader = _build_loader(
        cfg,
        _datasetinfo(dataset_payload, route=route, manifest_path=manifest),
        1,
        0,
        device,
        20260821,
    )
    batches = list(loader)
    if len(batches) != len(rows):
        raise RuntimeError("efficiency loader length drifted")
    if route == "B":
        updated = []
        for batch, row in zip(batches, rows):
            targets = list(batch[1])
            _apply_route_admission_inputs("B", targets, [row])
            updated.append((batch[0], targets))
        batches = updated
    return batches


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def device_metadata(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    return {
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "driver": query,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": "torch.cuda.amp autocast (fp16 kernels where enabled)",
    }


def summarize(times_ms: list[float]) -> dict[str, float]:
    array = np.asarray(times_ms, dtype=np.float64)
    q25, q75 = np.quantile(array, [0.25, 0.75])
    mean = float(np.mean(array))
    return {
        "iterations": int(array.size),
        "median_ms": float(np.median(array)),
        "mean_ms": mean,
        "p90_ms": float(np.quantile(array, 0.90)),
        "q25_ms": float(q25),
        "q75_ms": float(q75),
        "iqr_ms": float(q75 - q25),
        "throughput_images_per_second": 1000.0 / mean,
    }


def run_forward(cfg: Any, model: nn.Module, batch: Any, device: torch.device) -> None:
    outputs, _, _ = _forward_ref_batch(cfg, model, batch, device, amp=True)
    boxes = outputs.get("pred_boxes")
    if not torch.is_tensor(boxes) or tuple(boxes.shape[:2]) != (1, 900):
        raise RuntimeError("efficiency forward output contract drifted")


def benchmark(
    name: str,
    cfg: Any,
    model: nn.Module,
    batches: list[Any],
    device: torch.device,
    warmup: int,
    iterations: int,
    *,
    notes: str,
) -> dict[str, Any]:
    if len(batches) < iterations:
        raise ValueError("timed iterations exceed fixed manifest rows")
    model.eval()
    with torch.inference_mode():
        for index in range(warmup):
            run_forward(cfg, model, batches[index % len(batches)], device)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        for index in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_forward(cfg, model, batches[index], device)
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
    result = summarize(times)
    result.update({
        "route": name,
        "batch_size": 1,
        "model_parameters_loaded": parameter_count(model),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "notes": notes,
    })
    return result


def cached_support_batches(
    model: nn.Module,
    batches: list[Any],
    device: torch.device,
) -> list[Any]:
    root = model.module if hasattr(model, "module") else model
    result: list[Any] = []
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        for batch in batches:
            _, _, _, patches, patch_global, _ = _prepare_stage_b_u0_patch_batch(batch, device)
            if patch_global is None:
                if patches is None:
                    raise RuntimeError("uncached visual route yielded no support tensor")
                patch_global = root.encode_patches(patches)["patch_global"]
            if tuple(patch_global.shape[:1]) != (1,):
                raise RuntimeError("cached support batch is not batch-one")
            targets = []
            for target in batch[1]:
                copied = dict(target)
                for key in ("patch", "patches", "patch_global"):
                    copied.pop(key, None)
                copied["patch_global"] = patch_global[0].detach()
                targets.append(copied)
            result.append((batch[0], targets))
    return result


def rank_wrapper_parity(cfg: Any, model: nn.Module, batch: Any, device: torch.device) -> dict[str, Any]:
    root = model.module if hasattr(model, "module") else model
    source = root.stage_b_gdino_score_adapter
    with torch.inference_mode():
        original, _, _ = _forward_ref_batch(cfg, model, batch, device, amp=True)
        wanted = original["stage_b_gdino_rank_score"].detach().clone()
        root.stage_b_gdino_score_adapter = RankOnlyScoreAdapter(source).to(device).eval()
        wrapped, _, _ = _forward_ref_batch(cfg, model, batch, device, amp=True)
        observed = wrapped["stage_b_gdino_rank_score"]
    if not torch.equal(wanted, observed):
        raise RuntimeError("rank-only efficiency wrapper changed the sealed rank score")
    return {
        "bitwise_rank_score_parity": True,
        "compared_elements": int(wanted.numel()),
        "original_adapter_params": sum(p.numel() for p in source.parameters()),
        "rank_only_adapter_params": sum(p.numel() for p in root.stage_b_gdino_score_adapter.parameters()),
    }


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=FORMAL_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("formal efficiency measurement requires CUDA")
    if not args.engineering_smoke and (args.samples < 200 or args.warmup < 50 or args.iterations < 200):
        raise ValueError("formal protocol requires >=200 samples, >=50 warmup, >=200 timings")
    if args.iterations > args.samples or min(args.samples, args.warmup, args.iterations) <= 0:
        raise ValueError("invalid benchmark sizes")

    device = torch.device(args.device)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    dataset = json.loads(Path(prereg["dataset"]["path"]).read_text(encoding="utf-8"))
    source_a = Path(dataset["manifests"]["a_eval"]["path"])
    source_b = Path(dataset["manifests"]["bc_full"]["path"])
    rows_a, rows_b = fixed_rows(source_a, source_b, args.samples)
    manifest_payload = {
        "schema": "arrow.paper.efficiency_manifest/v1",
        "selection": "first positive exact-support row per unique FineCops source image in sealed manifest order",
        "rows": args.samples,
        "unique_images": len({row["finecops_cluster_gqa_image_id"] for row in rows_a}),
        "sample_ids": [row["sample_id"] for row in rows_a],
        "source_a": file_record(source_a),
        "source_b": file_record(source_b),
        "input_contract": {"batch_size": 1, "resize_short_side": 800, "max_long_side": 1333, "horizontal_flip": False},
    }
    manifest_path = MANIFEST_OUTPUT if not args.engineering_smoke else args.output.with_name("efficiency_manifest_smoke.json")
    write_json(manifest_path, manifest_payload)

    with tempfile.TemporaryDirectory(prefix="arrow-efficiency-") as temporary:
        temp = Path(temporary)
        cfg_a = config(Path(prereg["configs"]["A"]["path"]), device)
        cfg_b = config(Path(prereg["configs"]["B"]["path"]), device)
        batches_a = loader_batches(cfg_a, dataset, "A", rows_a, temp, device)
        batches_b = loader_batches(cfg_b, dataset, "B", rows_b, temp, device)
        transformed_shapes = sorted({tuple(batch[0].tensors.shape[-2:]) for batch in batches_a})

        variants: list[dict[str, Any]] = []
        checkpoints: dict[str, Any] = {}

        cfg_base = config(BASE_CONFIG, device)
        base = _load_model(cfg_base, str(BASE_CHECKPOINT), device).eval()
        checkpoints["frozen_base"] = file_record(BASE_CHECKPOINT)
        base_params = parameter_count(base)
        variants.append(benchmark(
            "Frozen base", cfg_base, base, batches_a, device, args.warmup, args.iterations,
            notes="complete-expression candidate generator and native base scoring; support target ignored",
        ))
        del base
        release_cuda()

        cfg_rank = config(RANK_CONFIG, device)
        rank_model = _load_model(cfg_rank, str(RANK_CHECKPOINT), device).eval()
        checkpoints["rank_initializer"] = file_record(RANK_CHECKPOINT)
        parity = rank_wrapper_parity(cfg_rank, rank_model, batches_a[0], device)
        variants.append(benchmark(
            "+ complete-expression ranker", cfg_rank, rank_model, batches_a, device, args.warmup, args.iterations,
            notes="bitwise-equivalent rank-only wrapper omits the independently owned rejector",
        ))
        del rank_model
        release_cuda()

        text_checkpoint = Path(prereg["checkpoints"]["B"]["42"]["path"])
        text_model, _ = _load_model_with_checkpoint_contract(cfg_b, text_checkpoint, device)
        text_model.eval(); checkpoints["arrow_t"] = file_record(text_checkpoint)
        variants.append(benchmark(
            "ARROW-T", cfg_b, text_model, batches_b, device, args.warmup, args.iterations,
            notes="canonical category phrase encoded inside the timed forward; no support patch read",
        ))
        full_params = parameter_count(text_model)
        del text_model
        release_cuda()

        visual_checkpoint = Path(prereg["checkpoints"]["A"]["42"]["path"])
        visual_model, _ = _load_model_with_checkpoint_contract(cfg_a, visual_checkpoint, device)
        visual_model.eval(); checkpoints["arrow_v"] = file_record(visual_checkpoint)
        variants.append(benchmark(
            "ARROW-V, support uncached", cfg_a, visual_model, batches_a, device, args.warmup, args.iterations,
            notes="frozen support encoder runs inside every timed forward",
        ))
        cached = cached_support_batches(visual_model, batches_a, device)
        variants.append(benchmark(
            "ARROW-V, support cached", cfg_a, visual_model, cached, device, args.warmup, args.iterations,
            notes="one frozen 256-D support embedding per sample is GPU-resident before timing",
        ))
        del visual_model
        release_cuda()

    rank_params = int(parity["rank_only_adapter_params"])
    rejector_params = 83969
    admission_surface_params = 263680
    admission_auxiliary_params = 4487
    cumulative_owner_params = rank_params + rejector_params + admission_surface_params + admission_auxiliary_params
    receipt = {
        "schema": "arrow.paper.efficiency_receipt/v1",
        "status": "engineering_smoke" if args.engineering_smoke else "formal_zero_training_measurement",
        "no_training": True,
        "optimizer_created": False,
        "model_weights_changed": False,
        "protocol": {
            "batch_size": 1,
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "cuda_events": True,
            "synchronize_each_iteration": True,
            "file_io_inside_timing": False,
            "visualization_inside_timing": False,
            "host_to_device_transfer_inside_timing": True,
            "manifest": file_record(manifest_path),
            "transformed_tensor_shapes_hw": [list(shape) for shape in transformed_shapes],
        },
        "environment": device_metadata(device),
        "sources": {
            "script": file_record(Path(__file__)),
            "finecops_preregistration": file_record(PREREG),
            "configs": {
                "base": file_record(BASE_CONFIG), "rank": file_record(RANK_CONFIG),
                "arrow_a": file_record(Path(prereg["configs"]["A"]["path"])),
                "arrow_b": file_record(Path(prereg["configs"]["B"]["path"])),
            },
            "checkpoints": checkpoints,
        },
        "parameter_accounting": {
            "frozen_base_total": base_params,
            "arrow_runtime_total": full_params,
            "runtime_loaded_increase": full_params - base_params,
            "cumulative_ever_trained_decision_owner_params": cumulative_owner_params,
            "phase_active": {
                "complete_expression_ranker": rank_params,
                "admission_surface_plus_training_only_auxiliary": admission_surface_params + admission_auxiliary_params,
                "isolated_rejector": rejector_params,
            },
            "deployed_decision_owner_params": rank_params + admission_surface_params + rejector_params,
            "training_only_auxiliary_params": admission_auxiliary_params,
            "frozen_admission_scale_param": full_params - base_params - cumulative_owner_params,
            "notes": "runtime increase includes one frozen patch-logit-scale scalar; the auxiliary is serialized and currently computed but never owns the deployed gate/rank score",
        },
        "rank_only_wrapper_preflight": parity,
        "variants": variants,
    }
    write_json(args.output, receipt)
    print(json.dumps({row["route"]: {key: row[key] for key in ("median_ms", "p90_ms", "peak_allocated_bytes")} for row in variants}, indent=2))


if __name__ == "__main__":
    main()
