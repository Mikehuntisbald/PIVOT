#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util import get_tokenlizer  # noqa: E402
from tools.eval_stagea_patch_checkpoints import (  # noqa: E402
    PatchEvalAccumulator,
    _as_patch_logits,
    _build_loader,
    _checkpoint_label,
    _dataset_name,
    _load_model_and_criterion,
    _prepare_patch_batch,
    _set_seed,
)
from tools.eval_text_stagea_caliber_checkpoints import (  # noqa: E402
    _build_pos_map,
    _class_names_from_target,
    _load_cid_to_name,
    _make_prompt,
    _support_classes_from_target,
)
from util.slconfig import SLConfig  # noqa: E402


def _pad_pos_maps(pos_maps: List[torch.Tensor], kmax: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros((len(pos_maps), int(kmax), 256), dtype=torch.float32, device=device)
    for i, pos_map in enumerate(pos_maps):
        rows = min(int(kmax), int(pos_map.shape[0]))
        cols = min(256, int(pos_map.shape[1]))
        if rows > 0 and cols > 0:
            out[i, :rows, :cols] = pos_map[:rows, :cols].to(device=device, dtype=torch.float32)
    denom = out.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return out / denom


def _make_prompts_and_pos_maps(raw_targets, tokenizer, cid_to_name) -> Tuple[List[str], List[torch.Tensor]]:
    prompts: List[str] = []
    pos_maps: List[torch.Tensor] = []
    for target in raw_targets:
        support_classes = _support_classes_from_target(target)
        names = _class_names_from_target(target, support_classes, cid_to_name)
        prompt = _make_prompt(names)
        prompts.append(prompt)
        pos_maps.append(_build_pos_map(tokenizer, prompt, names))
    return prompts, pos_maps


@torch.no_grad()
def _forward_patch_text(model, batch, device: torch.device, *, tokenizer, cid_to_name, amp: bool):
    raw_targets = list(batch[1])
    prompts, pos_maps = _make_prompts_and_pos_maps(raw_targets, tokenizer, cid_to_name)
    samples, targets, _captions, patches, patch_global, patch_mask = _prepare_patch_batch(*batch, device)
    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        outputs = model(
            samples,
            targets=targets,
            captions=prompts,
            patches=patches,
            patch_global=patch_global,
            patch_mask=patch_mask,
            patch_only=True,
            patch_only_compute_text_logits=True,
        )
    patch_logits = _as_patch_logits(outputs)
    text_logits = outputs.get("pred_logits_text", None)
    if text_logits is None:
        raise KeyError("Model did not return pred_logits_text; patch_only_compute_text_logits=True is required.")
    text_logits = text_logits.detach().float()
    pos_map = _pad_pos_maps(pos_maps, patch_logits.shape[-1], device)
    text_prob = torch.einsum("bqt,bkt->bqk", text_logits.sigmoid(), pos_map).clamp(1e-5, 1.0 - 1e-5)
    text_logit = torch.logit(text_prob)
    return outputs, targets, patch_logits, text_logit


def _mean_metric(results: List[Dict[str, Any]], run_id: str, metric: str) -> float:
    vals = [float(r.get(metric, 0.0)) for r in results if r["run_id"] == run_id]
    return sum(vals) / max(1, len(vals))


def _write_outputs(output_dir: Path, results: List[Dict[str, Any]], primary_metric: str) -> None:
    run_ids = []
    seen = set()
    for r in results:
        run_id = r["run_id"]
        if run_id not in seen:
            seen.add(run_id)
            run_ids.append(run_id)
    ranking = [
        {
            "rank": i + 1,
            "run_id": run_id,
            f"mean_{primary_metric}": _mean_metric(results, run_id, primary_metric),
            "mean_box_recall@50": _mean_metric(results, run_id, "box_recall@50"),
            "mean_matched_query_recall@50": _mean_metric(results, run_id, "matched_query_recall@50"),
        }
        for i, run_id in enumerate(
            sorted(run_ids, key=lambda rid: _mean_metric(results, rid, primary_metric), reverse=True)
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"primary_metric": primary_metric, "ranking": ranking, "results": results}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    datasets = []
    seen_ds = set()
    for r in results:
        ds = r["dataset"]
        if ds not in seen_ds:
            seen_ds.add(ds)
            datasets.append(ds)
    by_run_ds = {(r["run_id"], r["dataset"]): r for r in results}
    lines = [
        "# Stage-A patch/text fusion probe",
        "",
        f"Primary metric: mean `{primary_metric}` across evaluated datasets.",
        "",
        "| rank | run | mean patch_ap50 | mean box_recall@50 | mean matched_query_recall@50 | "
        + " | ".join(f"{ds} patch_ap50" for ds in datasets)
        + " |",
        "|---:|---|---:|---:|---:|" + "|".join("---:" for _ in datasets) + "|",
    ]
    for row in ranking:
        run_id = row["run_id"]
        ds_vals = [f"{float(by_run_ds.get((run_id, ds), {}).get('patch_ap50', 0.0)):.6f}" for ds in datasets]
        lines.append(
            f"| {row['rank']} | `{run_id}` | "
            f"{float(row[f'mean_{primary_metric}']):.6f} | "
            f"{float(row['mean_box_recall@50']):.6f} | "
            f"{float(row['mean_matched_query_recall@50']):.6f} | "
            + " | ".join(ds_vals)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate_one(
    *,
    cfg,
    model,
    criterion,
    datasetinfo: Dict[str, Any],
    dataset_name: str,
    ckpt_path: str,
    device: torch.device,
    tokenizer,
    betas: List[float],
    batch_size: int,
    num_workers: int,
    seed: int,
    topks: List[int],
    ap_max_dets_per_image: int,
    amp: bool,
    max_batches: int,
    max_images: int,
    log_every: int,
) -> List[Dict[str, Any]]:
    loader = _build_loader(
        cfg=cfg,
        datasetinfo=datasetinfo,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )
    cid_to_name = _load_cid_to_name(datasetinfo.get("canonical_classes_json", None))
    accs = {
        float(beta): PatchEvalAccumulator(topks, ap_max_dets_per_image=ap_max_dets_per_image)
        for beta in betas
    }
    start = time.time()
    total_batches = len(loader)
    print(
        f"[INFO] fusion eval ckpt={Path(ckpt_path).name} dataset={dataset_name} "
        f"images={len(loader.dataset)} batches={total_batches} batch_size={batch_size} betas={betas}"
    )
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= max_batches:
            break
        if max_images > 0 and next(iter(accs.values())).num_images >= max_images:
            break
        outputs, targets, patch_logits, text_logit = _forward_patch_text(
            model, batch, device, tokenizer=tokenizer, cid_to_name=cid_to_name, amp=amp
        )
        for beta, acc in accs.items():
            fused_outputs = dict(outputs)
            fused_outputs["pred_logits_patch"] = patch_logits + float(beta) * text_logit
            acc.update(fused_outputs, targets, criterion)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and ((batch_i + 1) % log_every == 0 or batch_i == 0):
            elapsed = time.time() - start
            done_batches = batch_i + 1
            target_batches = min(total_batches, max_batches) if max_batches > 0 else total_batches
            if max_images > 0:
                target_batches = min(target_batches, (max_images + max(1, batch_size) - 1) // max(1, batch_size))
            eta = elapsed / max(1, done_batches) * max(0, target_batches - done_batches)
            print(
                f"[INFO] {dataset_name} {Path(ckpt_path).name}: "
                f"batch {done_batches}/{target_batches}, images={next(iter(accs.values())).num_images}, "
                f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
            )

    results = []
    ckpt_label = _checkpoint_label(ckpt_path)
    for beta, acc in accs.items():
        result = acc.result()
        run_id = f"{ckpt_label}:patch_plus_text_b{beta:g}"
        result.update(
            {
                "run_id": run_id,
                "checkpoint": str(ckpt_path),
                "checkpoint_name": Path(ckpt_path).name,
                "dataset": dataset_name,
                "score_mode": "patch_plus_text_logit",
                "text_beta": float(beta),
                "seconds": time.time() - start,
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "seed": int(seed),
                "max_batches": int(max_batches),
                "max_images": int(max_images),
                "ap_max_dets_per_image": int(ap_max_dets_per_image),
            }
        )
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Stage-A patch/text score fusion on patch episodes.")
    parser.add_argument("--config", default="config/cfg_patch_stage_a.py")
    parser.add_argument("--datasets", default="config/datasets_patch_stage_a_lvis_coco2017_eval_local.json")
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/stageA_patch_text_fusion_probe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--ap_max_dets_per_image", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--dataset_names", nargs="*", default=None)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--primary_metric", default="patch_ap50")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.batch_size = int(args.batch_size)
    cfg.patch_only = True
    cfg.use_coco_eval = False

    with open(args.datasets, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)
    val_infos = list(dataset_meta.get("val", []))
    if args.dataset_names:
        wanted = set(args.dataset_names)
        val_infos = [d for i, d in enumerate(val_infos) if _dataset_name(d, i) in wanted]
    if not val_infos:
        raise ValueError(f"No val datasets found in {args.datasets}")

    tokenizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    results: List[Dict[str, Any]] = []
    for ckpt_i, ckpt_path in enumerate(args.ckpts):
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(args.ckpts)}: {ckpt_path}")
        _set_seed(int(args.seed))
        model, criterion = _load_model_and_criterion(cfg, ckpt_path, device)
        for ds_i, datasetinfo in enumerate(val_infos):
            ds_name = _dataset_name(datasetinfo, ds_i)
            ds_results = evaluate_one(
                cfg=cfg,
                model=model,
                criterion=criterion,
                datasetinfo=datasetinfo,
                dataset_name=ds_name,
                ckpt_path=ckpt_path,
                device=device,
                tokenizer=tokenizer,
                betas=list(args.betas),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(args.seed) + ds_i * 100000,
                topks=list(args.topk),
                ap_max_dets_per_image=int(args.ap_max_dets_per_image),
                amp=bool(args.amp),
                max_batches=int(args.max_batches),
                max_images=int(args.max_images),
                log_every=int(args.log_every),
            )
            results.extend(ds_results)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{_checkpoint_label(ckpt_path)}__{ds_name}.json").write_text(
                json.dumps(ds_results, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _write_outputs(output_dir, results, str(args.primary_metric))
            for r in ds_results:
                print(
                    f"[RESULT] {r['run_id']} {ds_name}: patch_ap50={r['patch_ap50']:.6f} "
                    f"box_recall@50={r.get('box_recall@50', 0.0):.6f} "
                    f"matched_query_recall@50={r.get('matched_query_recall@50', 0.0):.6f}"
                )
        del model
        del criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_outputs(output_dir, results, str(args.primary_metric))
    print(f"[INFO] wrote {output_dir / 'summary.json'}")
    print(f"[INFO] wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
