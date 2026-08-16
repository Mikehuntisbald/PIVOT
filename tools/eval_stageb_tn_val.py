#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from tools.stageb_eval_holdout import is_excluded, load_holdout_keys  # noqa: E402
from groundingdino.util import box_ops  # noqa: E402
from models.GroundingDINO.stage_b_score import compute_stage_b_slot_logits  # noqa: E402
from tools.eval_refcoco_stageb import (  # noqa: E402
    _ckpt_run_prefix,
    _load_model,
    _safe_name,
    _uses_stage_b_post_candidate_scorer,
)
from tools.eval_stagea_patch_checkpoints import _prepare_patch_batch, _set_seed  # noqa: E402
from tools.stageb_eval_records import (  # noqa: E402
    EvalManifest,
    extract_adapter_tn_pair_captions,
    load_eval_manifest,
    make_eval_record,
    sample_id_from_meta,
    tn_manifest_binding_summary_fields,
    validate_eval_manifest_batch_alignment as _validate_eval_manifest_batch_alignment,
    write_tn_derived_manifest_binding,
    write_eval_records,
)
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


_DATA_DRIVEN_CONFIDENCE_SCORE_KEY = "stage_b_data_driven_confidence_score"
_NATIVE_PATCH_CONFIDENCE_SCORE_KEY = (
    "stage_b_native_patch_confidence_score"
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _load_ref_split_map(data_root: Path, dataset: str, splitby: str) -> Dict[Tuple[int, int, int], str]:
    refs_path = data_root / "COCO" / dataset / f"refs({splitby}).p"
    with refs_path.open("rb") as handle:
        refs = pickle.load(handle)
    out: Dict[Tuple[int, int, int], str] = {}
    for ref in refs:
        try:
            key = (int(ref["ref_id"]), int(ref["ann_id"]), int(ref["image_id"]))
        except Exception:
            continue
        out[key] = str(ref.get("split", ""))
    return out


def _split_specs() -> List[Dict[str, str]]:
    return [
        {
            "name": "refcoco_val",
            "pair_source": "refcoco_unc",
            "dataset": "refcoco",
            "splitby": "unc",
            "split": "val",
        },
        {
            "name": "refcoco_testA",
            "pair_source": "refcoco_unc",
            "dataset": "refcoco",
            "splitby": "unc",
            "split": "testA",
        },
        {
            "name": "refcoco_testB",
            "pair_source": "refcoco_unc",
            "dataset": "refcoco",
            "splitby": "unc",
            "split": "testB",
        },
        {
            "name": "refcocop_val",
            "pair_source": "refcoco+_unc",
            "dataset": "refcoco+",
            "splitby": "unc",
            "split": "val",
        },
        {
            "name": "refcocop_testA",
            "pair_source": "refcoco+_unc",
            "dataset": "refcoco+",
            "splitby": "unc",
            "split": "testA",
        },
        {
            "name": "refcocop_testB",
            "pair_source": "refcoco+_unc",
            "dataset": "refcoco+",
            "splitby": "unc",
            "split": "testB",
        },
        {
            "name": "refcocog_val",
            "pair_source": "refcocog_google",
            "dataset": "refcocog",
            "splitby": "google",
            "split": "val",
        },
        {
            "name": "refcocog_umd_val",
            "pair_source": "refcocog_umd",
            "dataset": "refcocog",
            "splitby": "umd",
            "split": "val",
        },
    ]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_category(value: Any) -> str:
    values = _as_list(value)
    if not values:
        return "unknown"
    text = str(values[0]).strip()
    return text or "unknown"


def _build_tn_eval_jsonl(
    *,
    data_root: Path,
    output_dir: Path,
    tn_jsonl: Path,
    splits: List[str],
    max_pairs: int,
    max_pairs_per_split: int = 0,
    holdout_level: str = "none",
    holdout_ann_keys=None,
    holdout_image_ids=None,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, int]]:
    specs = {spec["name"]: spec for spec in _split_specs()}
    wanted = list(splits)
    if wanted == ["all"]:
        wanted = list(specs)
    unknown = [name for name in wanted if name not in specs]
    if unknown:
        raise KeyError(f"Unknown TN split names: {unknown}; available={list(specs)} or all")

    split_maps: Dict[Tuple[str, str, str], Dict[Tuple[int, int, int], str]] = {}
    wanted_by_source_split: Dict[Tuple[str, str], str] = {}
    for name in wanted:
        spec = specs[name]
        map_key = (spec["dataset"], spec["splitby"], spec["pair_source"])
        if map_key not in split_maps:
            split_maps[map_key] = _load_ref_split_map(data_root, spec["dataset"], spec["splitby"])
        wanted_by_source_split[(spec["pair_source"], spec["split"])] = name

    out_dir = output_dir / "tn_eval_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_slug = "_".join(_safe_name(x) for x in wanted)
    out_path = out_dir / f"tn_{split_slug}.jsonl"
    meta_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {name: 0 for name in wanted}
    row_mapping: List[Dict[str, Any]] = []
    seen = 0
    holdout_ann_keys = holdout_ann_keys or set()
    holdout_image_ids = holdout_image_ids or set()

    with out_path.open("w", encoding="utf-8") as out_f:
        for source_index, row in enumerate(_iter_jsonl(tn_jsonl)):
            instances = row.get("instances")
            if not isinstance(instances, list) or not instances:
                continue
            inst = instances[0]
            if not isinstance(inst, dict):
                continue
            pair_source = str(inst.get("pair_source") or row.get("pair_source") or row.get("source") or "")
            if not pair_source:
                continue
            try:
                key = (int(row["ref_id"]), int(row["ann_id"]), int(row["image_id"]))
            except Exception:
                continue

            source_split = None
            for (dataset, splitby, source), split_map in split_maps.items():
                if source != pair_source:
                    continue
                source_split = split_map.get(key)
                if source_split is not None:
                    break
            if source_split is None:
                continue
            eval_split = wanted_by_source_split.get((pair_source, source_split))
            if eval_split is None:
                continue
            if int(max_pairs_per_split) > 0 and counts.get(eval_split, 0) >= int(max_pairs_per_split):
                continue
            if is_excluded(
                image_id=int(row["image_id"]),
                ann_id=int(row["ann_id"]),
                level=holdout_level,
                ann_keys=holdout_ann_keys,
                image_ids=holdout_image_ids,
            ):
                continue

            positive_phrase = inst.get("positive_phrase")
            if not isinstance(positive_phrase, str) or not positive_phrase.strip():
                continue
            inst["text_is_negative"] = True
            out_row = dict(row)
            out_row["tn_eval_split"] = eval_split
            out_row["tn_eval_pair_source"] = pair_source
            out_row["tn_eval_source_split"] = source_split
            out_row["instances"] = [inst]
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            row_mapping.append(
                {
                    "derived_index": int(seen),
                    "source_index": int(source_index),
                    "sample_id": sample_id_from_meta(
                        row,
                        task="tn",
                        split="global",
                        index=source_index,
                    ),
                    "pair_source": pair_source,
                    "source_split": source_split,
                    "eval_split": eval_split,
                }
            )

            meta_rows.append(
                {
                    "eval_split": eval_split,
                    "pair_source": pair_source,
                    "source_split": source_split,
                    "image_id": int(row["image_id"]),
                    "ann_id": int(row["ann_id"]),
                    "ref_id": int(row["ref_id"]),
                    "sent_id": int(row.get("sent_id", -1)),
                    "negative_phrase": inst.get("raw_phrase") or inst.get("phrase"),
                    "positive_phrase": positive_phrase,
                    "category": _first_category(inst.get("replace_category")),
                }
            )
            counts[eval_split] = counts.get(eval_split, 0) + 1
            seen += 1
            if max_pairs > 0 and seen >= int(max_pairs):
                break
    write_tn_derived_manifest_binding(
        source_manifest_path=tn_jsonl,
        derived_manifest_path=out_path,
        row_mapping=row_mapping,
        requested_splits=wanted,
        max_pairs=max_pairs,
        max_pairs_per_split=max_pairs_per_split,
        holdout_level=holdout_level,
    )
    return out_path, meta_rows, counts


def _validate_adapter_tn_eval_manifest(
    cfg,
    rows: List[Dict[str, Any]],
    *,
    allow_proposal_covered_calibration: bool = False,
) -> Optional[str]:
    adapter_enabled = bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
    data_driven_enabled = bool(getattr(cfg, "stage_b_data_driven_score", False))
    if not (adapter_enabled or data_driven_enabled):
        return None
    from models.GroundingDINO.stage_b_gdino_score_adapter import (
        stage_b_gdino_tn_scope_code,
    )

    train_scope = str(
        getattr(
            cfg,
            (
                "stage_b_gdino_tn_scope"
                if adapter_enabled
                else "stage_b_data_driven_tn_scope"
            ),
            "",
        )
    ).strip()
    if train_scope:
        stage_b_gdino_tn_scope_code(train_scope)
    eval_scopes = set()
    protocols = set()
    for index, row in enumerate(rows):
        if row.get("manifest_schema", None) == "stageb_vlm_verified_strict_tn_v2":
            audit = row.get("proposal_audit", None)
            if (
                row.get("visual_verified_negative", None) is not True
                or row.get("coverage_pass", None) is not True
                or row.get("coverage_policy", None) != "target_plus_proposal"
                or not isinstance(audit, dict)
                or audit.get("target_verified_no", None) is not True
                or audit.get("target_plus_proposal_covered", None) is not True
            ):
                raise ValueError(
                    "Stage-B GDINO adapter strict-v2 TN manifest contract failed "
                    f"at row {index}"
                )
            # proposal_count=0 is valid for this audited policy, so
            # all_proposals_all_no is deliberately not required.
            scope = "image_global_topk_verified"
            protocols.add("stageb_vlm_verified_strict_tn_v2")
        else:
            scope = row.get("tn_scope", None)
            stage_b_gdino_tn_scope_code(scope)
            if scope == "proposal_covered_verified":
                expected_audit = str(
                    getattr(cfg, "stage_b_v19_table_b_audit_sha256", "")
                ).strip()
                if not (
                    allow_proposal_covered_calibration
                    and bool(getattr(cfg, "stage_b_u2v5_clean_confidence", False))
                    and row.get("table_b_id") == "D3"
                    and row.get("tn_eval_split") == "screen_calibration"
                    and row.get("tn_eval_source_split")
                    == "sealed_image_disjoint_calibration"
                    and row.get("proposal_covered_verified") is True
                    and row.get("global_tn_verified") is False
                    and row.get("benchmark_dataft_alltn") is False
                    and row.get("proposalset_proxy_verified") is False
                    and row.get("cached_proposal_coverage_only") is True
                    and row.get("all_900_gdino_queries_verified") is False
                    and row.get("global_max_label_is_semantic_extrapolation") is True
                    and len(expected_audit) == 64
                    and row.get("table_b_audit_sha256") == expected_audit
                ):
                    raise ValueError(
                        "proposal-covered adapter evaluation is allowed only for "
                        "the audit-bound U2-v5 D3 screen calibration"
                    )
                protocols.add("u2v5_d3_screen_calibration")
                eval_scopes.add(str(scope))
                continue
            verification_key = (
                "global_tn_verified"
                if scope == "image_global_topk_verified"
                else "benchmark_dataft_alltn"
            )
            if (
                row.get(verification_key, None) is not True
                or row.get("proposalset_proxy_verified", None) is not False
            ):
                raise ValueError(
                    "Stage-B GDINO adapter TN evaluation manifest is not strictly "
                    f"verified at row {index}: scope={scope!r}, requires exact "
                    f"boolean {verification_key}=true and proposal proxy=false"
                )
            protocols.add("adapter_training_pair_schema")
        eval_scopes.add(str(scope))
    if len(eval_scopes) != 1 or len(protocols) != 1:
        raise ValueError(
            "Stage-B GDINO adapter TN evaluation requires one uniform manifest "
            f"protocol/scope, got protocols={sorted(protocols)}, "
            f"scopes={sorted(eval_scopes)}"
        )
    # Training and evaluation scopes are intentionally recorded separately.
    return next(iter(eval_scopes))


def _make_datasetinfo(
    data_root: Path,
    anno: Path,
    *,
    adapter_eval_scope: Optional[str] = None,
    adapter_eval_protocol: Optional[str] = None,
    u0_patch_rank: bool = False,
    data_driven_score: bool = False,
) -> Dict[str, Any]:
    info = {
        "name": "tn_val",
        "dataset_mode": "patch_episode",
        "root": "/",
        "anno": str(anno),
        "box_format": "xywh",
        "canonical_classes_json": str(data_root / "canonical_classes_with_aliases.json"),
        "keep_only_support_gt": True,
        "neg_episode_prob": 0.0,
        "support_min_count": 1 if adapter_eval_scope else 2,
        "support_patch_size": 224,
        "support_num_patches_min": 1,
        "support_num_patches_max": 1,
        "build_text_token_masks": True,
        "text_mask_skip_invalid_canonical": False,
        "text_mask_warn_limit": 0,
        # Evaluation records are positional against the immutable manifest.
        # Disable training-only balancing explicitly to keep the contract fixed,
        # avoid extra statistics, and prevent future sampler misuse.
        "tn_balance_sampling": False,
    }
    if (
        adapter_eval_scope is not None
        and not u0_patch_rank
        and not data_driven_score
    ):
        info["stage_b_gdino_adapter_no_support"] = True
    else:
        info.update(
            {
                "support_patch_tsv": str(
                    data_root / "patches_quality_emb" / "emb_index_from_quality.tsv"
                ),
                "support_patch_bucket": "clean",
                "support_patch_use_embedding": False,
                "support_patch_image_root": str(data_root / "patches_quality"),
                "support_patch_max_per_class": 200,
                "patch_emb_cache_size": 4096,
            }
        )
    if adapter_eval_scope is not None:
        if adapter_eval_protocol == "stageb_vlm_verified_strict_tn_v2":
            info["require_vlm_strict_tn"] = True
        else:
            info["require_global_tn_verified"] = (
                adapter_eval_scope == "image_global_topk_verified"
            )
            info["require_benchmark_dataft_alltn"] = (
                adapter_eval_scope == "benchmark_dataft_alltn"
            )
    return info


def _build_loader(
    cfg,
    datasetinfo: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    _set_seed(seed)
    dataset = build_dataset(image_set="val", args=cfg, datasetinfo=datasetinfo)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _pad_target_mask(
    targets: List[Dict[str, Any]],
    key: str,
    kmax: int,
    tmax: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not any(key in target for target in targets):
        return None
    out = torch.zeros((len(targets), kmax, tmax), dtype=torch.bool, device=device)
    for i, target in enumerate(targets):
        mask = target.get(key)
        if not torch.is_tensor(mask):
            continue
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.dim() != 2:
            continue
        rows = min(kmax, int(mask.shape[0]))
        cols = min(tmax, int(mask.shape[-1]))
        if rows > 0 and cols > 0:
            out[i, :rows, :cols] = mask[:rows, :cols].to(device=device, dtype=torch.bool)
    return out


def _inject_text_masks(
    outputs: Dict[str, torch.Tensor],
    raw_targets: List[Dict[str, Any]],
    *,
    phrase_key: str,
    canonical_key: str,
    device: torch.device,
) -> None:
    patch_logits = outputs["pred_logits_patch"]
    kmax = 1 if patch_logits.dim() == 2 else int(patch_logits.shape[-1])
    tmax = int(outputs["pred_logits_text"].shape[-1])
    phrase_mask = _pad_target_mask(raw_targets, phrase_key, kmax, tmax, device)
    canonical_mask = _pad_target_mask(raw_targets, canonical_key, kmax, tmax, device)
    if phrase_mask is not None:
        outputs["phrase_to_token_mask"] = phrase_mask
    if canonical_mask is not None:
        outputs["canonical_to_token_mask"] = canonical_mask


def _target_texts(raw_targets: List[Dict[str, Any]], key: str, fallback_key: str = "caption") -> List[str]:
    texts: List[str] = []
    for target in raw_targets:
        value = target.get(key, None)
        if not isinstance(value, str) or not value.strip():
            value = target.get(fallback_key, "object .")
        texts.append(str(value or "object ."))
    return texts


def _rank_positive_captions(raw_targets: List[Dict[str, Any]]) -> Tuple[List[str], torch.Tensor]:
    captions: List[str] = []
    valid: List[bool] = []
    for target in raw_targets:
        rank_caps = target.get("rank_positive_captions", None)
        has_rank = target.get("has_rank_positive", None)
        cap = None
        ok = False
        if isinstance(rank_caps, list) and rank_caps:
            maybe = rank_caps[0]
            if isinstance(maybe, str) and maybe.strip():
                cap = maybe
                ok = True
        if torch.is_tensor(has_rank):
            ok = ok and bool(has_rank.view(-1)[0].item()) if has_rank.numel() > 0 else False
        captions.append(cap if cap is not None else str(target.get("caption", "object .")))
        valid.append(bool(ok))
    return captions, torch.as_tensor(valid, dtype=torch.bool)


def _split_paired_output(value, batch_size: int, *, negative: bool):
    if torch.is_tensor(value):
        if value.dim() > 0 and int(value.shape[0]) == 2 * int(batch_size):
            start = int(batch_size) if negative else 0
            return value.narrow(0, start, int(batch_size))
        return value
    if isinstance(value, dict):
        return {
            key: _split_paired_output(item, batch_size, negative=negative)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _split_paired_output(item, batch_size, negative=negative)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _split_paired_output(item, batch_size, negative=negative)
            for item in value
        )
    return value


def _is_stage_b_u0_model(model) -> bool:
    root = model.module if hasattr(model, "module") else model
    return getattr(root, "stage_b_u0_patch_rank_adapter", None) is not None


def _prepare_stage_b_u0_patch_batch(batch, device: torch.device):
    raw_targets = list(batch[1])
    if not raw_targets:
        raise ValueError("Stage-B U0 evaluation requires a non-empty target batch")
    support_keys = ("patch", "patches", "patch_global")
    batch_support_key: Optional[str] = None
    for index, target in enumerate(raw_targets):
        present = [key for key in support_keys if key in target]
        if not present:
            raise KeyError(
                f"Stage-B U0 target {index} is missing patch/patches/patch_global"
            )
        if len(present) != 1:
            raise ValueError(
                f"Stage-B U0 target {index} must contain exactly one support input key, "
                f"got {present}"
            )
        key = present[0]
        if batch_support_key is None:
            batch_support_key = key
        elif key != batch_support_key:
            raise ValueError(
                "Stage-B U0 target batch mixes support input representations: "
                f"{batch_support_key!r} and {key!r}"
            )
        value = target[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Stage-B U0 target {index} {key} must be a tensor")
        if key == "patch":
            valid_shape = value.dim() == 3
        elif key == "patches":
            valid_shape = value.dim() == 4 and int(value.shape[0]) == 1
        else:
            valid_shape = value.dim() == 1 or (
                value.dim() == 2 and int(value.shape[0]) == 1
            )
        if not valid_shape:
            raise ValueError(
                f"Stage-B U0 target {index} {key} must encode exactly one support slot, "
                f"got shape={tuple(value.shape)}"
            )

    prepared = _prepare_patch_batch(*batch, device)
    _samples, _targets, _captions, patches, patch_global, patch_mask = prepared
    supports = [value for value in (patches, patch_global) if value is not None]
    if len(supports) != 1:
        raise RuntimeError("Stage-B U0 patch preparation did not yield one support tensor")
    support = supports[0]
    if int(support.shape[0]) != len(raw_targets):
        raise RuntimeError("Stage-B U0 support batch does not align with targets")
    if patch_mask is not None:
        if tuple(patch_mask.shape) != (len(raw_targets), 1) or not bool(
            patch_mask.all().item()
        ):
            raise RuntimeError("Stage-B U0 requires one valid support slot per target")
    return prepared


def _duplicate_stage_b_u0_support(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.cat((value, value), dim=0)


@torch.no_grad()
def _forward_pair(model, batch, device: torch.device, *, amp: bool):
    raw_targets = list(batch[1])
    root = model.module if hasattr(model, "module") else model
    if bool(getattr(root, "stage_b_native_patch_category", False)):
        if not bool(
            getattr(root, "stage_b_native_patch_confidence_trained", False)
        ):
            raise RuntimeError(
                "native patch-category TN/FPR evaluation is forbidden until an "
                "independent confidence head is explicitly marked trained"
            )
        (
            samples,
            targets,
            _captions,
            patches,
            patch_global,
            patch_mask,
        ) = _prepare_stage_b_u0_patch_batch(batch, device)
        pos_captions, neg_captions, valid_pos = extract_adapter_tn_pair_captions(
            raw_targets
        )
        batch_size = len(raw_targets)
        paired_samples = utils.NestedTensor(
            torch.cat((samples.tensors, samples.tensors), dim=0),
            (
                torch.cat((samples.mask, samples.mask), dim=0)
                if samples.mask is not None
                else None
            ),
        )
        with torch.cuda.amp.autocast(
            enabled=bool(amp) and device.type == "cuda"
        ):
            paired_outputs = model(
                paired_samples,
                captions=list(pos_captions) + list(neg_captions),
                patches=_duplicate_stage_b_u0_support(patches),
                patch_global=_duplicate_stage_b_u0_support(patch_global),
                patch_mask=_duplicate_stage_b_u0_support(patch_mask),
                patch_only=False,
                disable_patch_dn=True,
            )
        if not isinstance(paired_outputs, dict):
            raise TypeError(
                "native patch-category confidence forward must return a dictionary"
            )
        if _NATIVE_PATCH_CONFIDENCE_SCORE_KEY not in paired_outputs:
            raise KeyError(
                "trained native patch-category confidence forward is missing "
                f"{_NATIVE_PATCH_CONFIDENCE_SCORE_KEY}"
            )
        pos_outputs = _split_paired_output(
            paired_outputs, batch_size, negative=False
        )
        neg_outputs = _split_paired_output(
            paired_outputs, batch_size, negative=True
        )
        return neg_outputs, pos_outputs, targets, valid_pos.to(device=device)
    if getattr(root, "stage_b_data_driven_score_heads", None) is not None:
        (
            samples,
            targets,
            _captions,
            patches,
            patch_global,
            patch_mask,
        ) = _prepare_stage_b_u0_patch_batch(batch, device)
        pos_captions, neg_captions, valid_pos = extract_adapter_tn_pair_captions(
            raw_targets
        )
        canonical_captions = []
        for index, target in enumerate(raw_targets):
            canonical = target.get("stage_a_caption")
            if not isinstance(canonical, str) or not canonical.strip():
                raise KeyError(
                    f"data-driven TN target {index} requires stage_a_caption"
                )
            canonical_captions.append(canonical)
        batch_size = len(raw_targets)
        paired_samples = utils.NestedTensor(
            torch.cat((samples.tensors, samples.tensors), dim=0),
            (
                torch.cat((samples.mask, samples.mask), dim=0)
                if samples.mask is not None
                else None
            ),
        )
        with torch.cuda.amp.autocast(
            enabled=bool(amp) and device.type == "cuda"
        ):
            paired_outputs = model(
                paired_samples,
                captions=canonical_captions + canonical_captions,
                patches=_duplicate_stage_b_u0_support(patches),
                patch_global=_duplicate_stage_b_u0_support(patch_global),
                patch_mask=_duplicate_stage_b_u0_support(patch_mask),
                patch_only=False,
                disable_patch_dn=True,
                stage_b_data_driven_expression_captions=(
                    list(pos_captions) + list(neg_captions)
                ),
            )
        pos_outputs = _split_paired_output(
            paired_outputs, batch_size, negative=False
        )
        neg_outputs = _split_paired_output(
            paired_outputs, batch_size, negative=True
        )
        return neg_outputs, pos_outputs, targets, valid_pos.to(device=device)
    if getattr(root, "stage_b_gdino_score_adapter", None) is not None:
        u0_patch_rank = _is_stage_b_u0_model(model)
        if u0_patch_rank:
            (
                samples,
                targets,
                _captions,
                patches,
                patch_global,
                patch_mask,
            ) = _prepare_stage_b_u0_patch_batch(batch, device)
        else:
            samples = batch[0].to(device)
            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                    if torch.is_tensor(value)
                    and key not in {"patch", "patches", "patch_global"}
                }
                for target in raw_targets
            ]
            patches = None
            patch_global = None
            patch_mask = None
        (
            pos_captions,
            neg_captions,
            valid_pos,
        ) = extract_adapter_tn_pair_captions(raw_targets)
        batch_size = len(raw_targets)
        paired_samples = utils.NestedTensor(
            torch.cat((samples.tensors, samples.tensors), dim=0),
            (
                torch.cat((samples.mask, samples.mask), dim=0)
                if samples.mask is not None
                else None
            ),
        )
        with torch.cuda.amp.autocast(
            enabled=bool(amp) and device.type == "cuda"
        ):
            forward_kwargs: Dict[str, Any] = {
                "captions": list(pos_captions) + list(neg_captions)
            }
            if u0_patch_rank:
                forward_kwargs.update(
                    targets=list(targets) + list(targets),
                    patches=_duplicate_stage_b_u0_support(patches),
                    patch_global=_duplicate_stage_b_u0_support(patch_global),
                    patch_mask=_duplicate_stage_b_u0_support(patch_mask),
                    patch_only=False,
                    disable_patch_dn=True,
                )
            paired_outputs = model(paired_samples, **forward_kwargs)
        pos_outputs = _split_paired_output(
            paired_outputs, batch_size, negative=False
        )
        neg_outputs = _split_paired_output(
            paired_outputs, batch_size, negative=True
        )
        return neg_outputs, pos_outputs, targets, valid_pos.to(device=device)

    samples, targets, neg_captions, patches, patch_global, patch_mask = _prepare_patch_batch(*batch, device)
    pos_captions, valid_pos = _rank_positive_captions(raw_targets)
    stage_a_captions = _target_texts(raw_targets, "stage_a_caption")
    is_v7 = (
        getattr(model, "stage_b_verifier", None) is not None
        or getattr(model, "stage_b_fixed_text_scorer", None) is not None
    )
    if is_v7:
        kmax = int(patch_mask.shape[1]) if patch_mask is not None else 1
        neg_phrase_mask = _pad_target_mask(raw_targets, "phrase_to_token_mask", kmax, 256, device)
        neg_canonical_mask = _pad_target_mask(raw_targets, "canonical_to_token_mask", kmax, 256, device)
        pos_phrase_mask = _pad_target_mask(raw_targets, "rank_positive_phrase_to_token_mask", kmax, 256, device)
        pos_canonical_mask = _pad_target_mask(raw_targets, "rank_positive_canonical_to_token_mask", kmax, 256, device)
        with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
            neg_outputs = model(
                samples,
                targets=targets,
                captions=stage_a_captions,
                patches=patches,
                patch_global=patch_global,
                patch_mask=patch_mask,
                patch_only=True,
                patch_only_compute_text_logits=False,
                disable_patch_dn=True,
                return_stage_b_v7_features=True,
                stage_b_v7_verifier_captions=neg_captions,
                phrase_to_token_mask=neg_phrase_mask,
                canonical_to_token_mask=neg_canonical_mask,
            )
            pos_outputs = model(
                samples,
                targets=targets,
                captions=stage_a_captions,
                patches=patches,
                patch_global=patch_global,
                patch_mask=patch_mask,
                patch_only=True,
                patch_only_compute_text_logits=False,
                disable_patch_dn=True,
                return_stage_b_v7_features=True,
                stage_b_v7_verifier_captions=pos_captions,
                phrase_to_token_mask=pos_phrase_mask,
                canonical_to_token_mask=pos_canonical_mask,
            )
        return neg_outputs, pos_outputs, targets, valid_pos.to(device=device)

    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        neg_outputs = model(
            samples,
            targets=targets,
            captions=neg_captions,
            patches=patches,
            patch_global=patch_global,
            patch_mask=patch_mask,
            patch_only=True,
            patch_only_compute_text_logits=True,
            disable_patch_dn=True,
        )
        pos_outputs = model(
            samples,
            targets=targets,
            captions=pos_captions,
            patches=patches,
            patch_global=patch_global,
            patch_mask=patch_mask,
            patch_only=True,
            patch_only_compute_text_logits=True,
            disable_patch_dn=True,
        )
    _inject_text_masks(
        neg_outputs,
        raw_targets,
        phrase_key="phrase_to_token_mask",
        canonical_key="canonical_to_token_mask",
        device=device,
    )
    _inject_text_masks(
        pos_outputs,
        raw_targets,
        phrase_key="rank_positive_phrase_to_token_mask",
        canonical_key="rank_positive_canonical_to_token_mask",
        device=device,
    )
    return neg_outputs, pos_outputs, targets, valid_pos.to(device=device)


def _slot_scores(outputs: Dict[str, torch.Tensor], cfg, beta: float) -> torch.Tensor:
    if bool(getattr(cfg, "stage_b_native_patch_category", False)):
        if not bool(
            getattr(cfg, "stage_b_native_patch_confidence_trained", False)
        ):
            raise RuntimeError(
                "native patch-category TN/FPR evaluation is forbidden until an "
                "independent confidence head is explicitly marked trained"
            )
        score = outputs.get(_NATIVE_PATCH_CONFIDENCE_SCORE_KEY)
        if score is None:
            raise KeyError(
                "native patch-category TN/FPR evaluation is missing "
                f"{_NATIVE_PATCH_CONFIDENCE_SCORE_KEY}; rank-score fallback is forbidden"
            )
        if not torch.is_tensor(score) or score.dim() != 2:
            shape = (
                tuple(score.shape)
                if torch.is_tensor(score)
                else type(score).__name__
            )
            raise ValueError(
                f"{_NATIVE_PATCH_CONFIDENCE_SCORE_KEY} must be a (B,Q) tensor, got {shape}"
            )
        boxes = outputs.get("pred_boxes")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 3
            or int(boxes.shape[-1]) != 4
            or tuple(score.shape) != tuple(boxes.shape[:2])
        ):
            shape = (
                tuple(boxes.shape)
                if torch.is_tensor(boxes)
                else type(boxes).__name__
            )
            raise ValueError(
                f"{_NATIVE_PATCH_CONFIDENCE_SCORE_KEY} must align with "
                f"pred_boxes (B,Q,4), got {shape}"
            )
        score = score.detach().float()
        if not bool(torch.isfinite(score).all().item()):
            raise ValueError(
                f"{_NATIVE_PATCH_CONFIDENCE_SCORE_KEY} must contain only finite values"
            )
        return score.unsqueeze(-1)

    if bool(getattr(cfg, "stage_b_data_driven_score", False)):
        if not bool(
            getattr(cfg, "stage_b_data_driven_confidence_trained", False)
        ):
            raise RuntimeError(
                "data-driven TN/FPR evaluation is forbidden before DD2 trains "
                "the independent confidence head"
            )
        score = outputs.get(_DATA_DRIVEN_CONFIDENCE_SCORE_KEY)
        if not torch.is_tensor(score) or score.dim() != 2:
            shape = tuple(score.shape) if torch.is_tensor(score) else type(score).__name__
            raise ValueError(
                f"{_DATA_DRIVEN_CONFIDENCE_SCORE_KEY} must be a (B,Q) tensor, got {shape}"
            )
        score = score.detach().float()
        if not bool(torch.isfinite(score).all().item()):
            raise ValueError(
                f"{_DATA_DRIVEN_CONFIDENCE_SCORE_KEY} must contain only finite values"
            )
        return score.unsqueeze(-1)

    if bool(getattr(cfg, "stage_b_u0_patch_rank", False)) and not bool(
        getattr(cfg, "stage_b_gdino_score_adapter", False)
    ):
        raise ValueError(
            "stage_b_u0_patch_rank TN evaluation requires "
            "stage_b_gdino_score_adapter and its confidence score"
        )
    gdino_confidence = outputs.get("stage_b_gdino_confidence_score", None)
    if (
        bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
        and gdino_confidence is None
    ):
        raise KeyError(
            "Stage-B GDINO adapter evaluation is missing "
            "stage_b_gdino_confidence_score"
        )
    if gdino_confidence is not None:
        score = gdino_confidence.detach().float()
        if score.dim() == 2:
            score = score.unsqueeze(-1)
        if score.dim() != 3:
            raise ValueError(
                "stage_b_gdino_confidence_score must be (B,Q) or (B,Q,K), "
                f"got {tuple(score.shape)}"
            )
        return score
    legacy_gate_score = outputs.get("stage_b_legacy_global_confidence", None)
    if legacy_gate_score is not None:
        score = legacy_gate_score.detach().float()
        if score.dim() == 2:
            score = score.unsqueeze(-1)
        if score.dim() != 3:
            raise ValueError(
                "stage_b_legacy_global_confidence must be (B,Q) or (B,Q,K), "
                f"got {tuple(score.shape)}"
            )
        return score
    if _uses_stage_b_post_candidate_scorer(cfg):
        score = outputs.get("stage_b_v7_final_score", None)
        if score is None:
            score = outputs.get("stage_b_v7_predicate_score", None)
        if score is None:
            raise KeyError(
                "Stage-B post-candidate eval requires stage_b_v7_final_score "
                "or stage_b_v7_predicate_score."
            )
        score = score.detach().float()
        if score.dim() == 2:
            score = score.unsqueeze(-1)
        if score.dim() != 3:
            raise ValueError(f"stage_b_v7 score must be (B,Q) or (B,Q,K), got {tuple(score.shape)}")
        return score

    return compute_stage_b_slot_logits(
        outputs,
        beta=float(beta),
        canonical_weight=float(getattr(cfg, "stage_b_infer_canonical_weight", 1.0)),
        text_agg=str(getattr(cfg, "stage_b_infer_text_agg", "mean")),
        softmin_tau=float(getattr(cfg, "stage_b_infer_softmin_tau", 0.7)),
        mean_softmin_alpha=float(getattr(cfg, "stage_b_infer_mean_softmin_alpha", 0.5)),
        normalize_fused_score=bool(getattr(cfg, "stage_b_infer_normalize_fused_score", True)),
        score_mode=str(getattr(cfg, "stage_b_score_mode", "patch_text")),
    )


def _best_scores_and_iou(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], cfg, beta: float):
    slot_logits = _slot_scores(outputs, cfg, beta)
    bsz, _q, k = slot_logits.shape
    flat = slot_logits.reshape(bsz, -1)
    score, flat_idx = flat.max(dim=1)
    query_idx = torch.div(flat_idx, k, rounding_mode="floor")

    pred_boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
    ious: List[float] = []
    for b, target in enumerate(targets):
        gt_boxes = target.get("boxes")
        if (not torch.is_tensor(gt_boxes)) or gt_boxes.numel() == 0:
            ious.append(float("nan"))
            continue
        gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1].detach().float()).clamp(0.0, 1.0)[0]
        q = int(query_idx[b].item())
        iou = box_ops.box_iou(pred_boxes[b, q : q + 1], gt.view(1, 4))[0].view(-1)[0]
        ious.append(float(iou.item()))
    return score.detach().float().cpu().numpy(), np.asarray(ious, dtype=np.float32)


def _score_at_best_iou(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], cfg, beta: float):
    slot_logits = _slot_scores(outputs, cfg, beta).detach().float()
    pred_boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
    candidate_mask = outputs.get("stage_b_v7_candidate_mask", None)
    if torch.is_tensor(candidate_mask):
        candidate_mask = candidate_mask.detach().to(device=pred_boxes.device, dtype=torch.bool)
        if candidate_mask.dim() == 3:
            candidate_mask = candidate_mask[..., 0]
        if candidate_mask.shape != pred_boxes.shape[:2]:
            candidate_mask = None
    scores: List[float] = []
    ious: List[float] = []
    for b, target in enumerate(targets):
        gt_boxes = target.get("boxes")
        if (not torch.is_tensor(gt_boxes)) or gt_boxes.numel() == 0:
            scores.append(float("nan"))
            ious.append(float("nan"))
            continue
        gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1].detach().float()).clamp(0.0, 1.0)[0]
        query_ious = box_ops.box_iou(pred_boxes[b], gt.view(1, 4))[0].view(-1)
        if candidate_mask is not None and bool(candidate_mask[b].any().item()):
            query_ious = query_ious.masked_fill(~candidate_mask[b], -1.0)
        q = int(query_ious.argmax().item())
        scores.append(float(slot_logits[b, q, 0].item()))
        ious.append(float(query_ious[q].item()))
    return np.asarray(scores, dtype=np.float32), np.asarray(ious, dtype=np.float32)


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _safe_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _threshold_for_tpr(pos_scores: np.ndarray, target_tpr: float) -> float:
    pos_scores = pos_scores[np.isfinite(pos_scores)]
    if pos_scores.size == 0:
        return float("inf")
    target_tpr = min(1.0, max(0.0, float(target_tpr)))
    if target_tpr <= 0.0:
        return float(np.nextafter(pos_scores.max(), np.inf))
    accepted = max(1, int(math.ceil(target_tpr * int(pos_scores.size))))
    ascending_index = int(pos_scores.size) - accepted
    return float(np.partition(pos_scores, ascending_index)[ascending_index])


def _summarize_arrays(
    *,
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    pos_iou: np.ndarray,
    neg_iou: np.ndarray,
    pos_iou_score: np.ndarray,
    neg_iou_score: np.ndarray,
    threshold_tprs: List[float],
) -> Dict[str, Any]:
    valid = np.isfinite(pos_scores) & np.isfinite(neg_scores)
    pos_scores = pos_scores[valid]
    neg_scores = neg_scores[valid]
    pos_iou = pos_iou[valid]
    neg_iou = neg_iou[valid]
    pos_iou_score = pos_iou_score[valid]
    neg_iou_score = neg_iou_score[valid]
    gap = pos_scores - neg_scores
    out: Dict[str, Any] = {
        "num_pairs": int(pos_scores.size),
        "pair_win_rate": float(np.mean(pos_scores > neg_scores)) if pos_scores.size else 0.0,
        "pair_tie_rate": float(np.mean(pos_scores == neg_scores)) if pos_scores.size else 0.0,
        "score_gap_mean": _safe_mean(gap),
        "score_gap_median": _safe_median(gap),
        "pos_score_mean": _safe_mean(pos_scores),
        "tn_score_mean": _safe_mean(neg_scores),
        "pos_score_median": _safe_median(pos_scores),
        "tn_score_median": _safe_median(neg_scores),
        "pos_top1_iou50": float(np.mean(pos_iou >= 0.5)) if pos_iou.size else 0.0,
        "tn_top1_iou50": float(np.mean(neg_iou >= 0.5)) if neg_iou.size else 0.0,
        "pos_best_iou_query_score_mean": _safe_mean(pos_iou_score),
        "tn_best_iou_query_score_mean": _safe_mean(neg_iou_score),
        "best_iou_query_pair_win_rate": (
            float(np.mean(pos_iou_score > neg_iou_score)) if pos_iou_score.size else 0.0
        ),
    }
    target_valid = np.isfinite(pos_iou_score) & np.isfinite(neg_iou_score)
    target_pos = pos_iou_score[target_valid]
    target_neg = neg_iou_score[target_valid]
    target_gap = target_pos - target_neg
    out.update(
        {
            "target_pair_win_rate": (
                float(np.mean(target_pos > target_neg)) if target_pos.size else 0.0
            ),
            "target_pair_tie_rate": (
                float(np.mean(target_pos == target_neg)) if target_pos.size else 0.0
            ),
            "target_score_gap_mean": _safe_mean(target_gap),
            "target_score_gap_median": _safe_median(target_gap),
            "target_pos_score_mean": _safe_mean(target_pos),
            "target_tn_score_mean": _safe_mean(target_neg),
        }
    )
    for tpr in threshold_tprs:
        key = f"{int(round(float(tpr) * 100)):02d}"
        threshold = _threshold_for_tpr(pos_scores, float(tpr))
        actual_tpr = float(np.mean(pos_scores >= threshold)) if pos_scores.size else 0.0
        fpr = float(np.mean(neg_scores >= threshold)) if neg_scores.size else 0.0
        out[f"threshold_at_{key}tpr"] = threshold
        out[f"actual_tpr_at_{key}tpr"] = actual_tpr
        out[f"fpr{key}tpr"] = fpr
        target_threshold = _threshold_for_tpr(target_pos, float(tpr))
        out[f"target_threshold_at_{key}tpr"] = target_threshold
        out[f"target_actual_tpr_at_{key}tpr"] = (
            float(np.mean(target_pos >= target_threshold)) if target_pos.size else 0.0
        )
        out[f"target_fpr{key}tpr"] = (
            float(np.mean(target_neg >= target_threshold)) if target_neg.size else 0.0
        )
    out.setdefault("fpr95tpr", 0.0)
    out.setdefault("target_fpr95tpr", 0.0)
    out["tn_fpr"] = float(out.get("fpr95tpr", 0.0))
    return out


def _summarize_group(
    records: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
    threshold_tprs: List[float],
    key: str,
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[int]] = {}
    for i, meta in enumerate(metas):
        value = str(meta.get(key, "unknown"))
        groups.setdefault(value, []).append(i)
    out: Dict[str, Dict[str, Any]] = {}
    for value, idxs in groups.items():
        if not idxs:
            continue
        idx = np.asarray(idxs, dtype=np.int64)
        out[value] = _summarize_arrays(
            pos_scores=np.asarray([records[i]["pos_score"] for i in idx], dtype=np.float32),
            neg_scores=np.asarray([records[i]["tn_score"] for i in idx], dtype=np.float32),
            pos_iou=np.asarray([records[i]["pos_iou"] for i in idx], dtype=np.float32),
            neg_iou=np.asarray([records[i]["tn_iou"] for i in idx], dtype=np.float32),
            pos_iou_score=np.asarray([records[i]["pos_best_iou_query_score"] for i in idx], dtype=np.float32),
            neg_iou_score=np.asarray([records[i]["tn_best_iou_query_score"] for i in idx], dtype=np.float32),
            threshold_tprs=threshold_tprs,
        )
    return out


class TnPairAccumulator:
    def __init__(
        self,
        betas: Iterable[float],
        *,
        manifest: EvalManifest,
        run_prefix: str,
        train_scope: Optional[str] = None,
        eval_scope: Optional[str] = None,
    ) -> None:
        self.betas = [float(beta) for beta in betas]
        self.manifest = manifest
        self.run_prefix = str(run_prefix)
        self.train_scope = train_scope
        self.eval_scope = eval_scope
        self.records: Dict[float, List[Dict[str, float]]] = {beta: [] for beta in self.betas}
        self.eval_records: Dict[float, List[Dict[str, Any]]] = {beta: [] for beta in self.betas}
        self.metas: List[Dict[str, Any]] = []
        self.invalid_positive = 0

    def update(
        self,
        *,
        neg_outputs: Dict[str, torch.Tensor],
        pos_outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        valid_pos: torch.Tensor,
        metas: List[Dict[str, Any]],
        cfg,
        manifest_start_index: int,
    ) -> None:
        valid_np = valid_pos.detach().cpu().numpy().astype(bool)
        self.invalid_positive += int((~valid_np).sum())

        def diagnostic_value(
            outputs: Dict[str, torch.Tensor], key: str, index: int
        ) -> Optional[float]:
            value = outputs.get(key)
            if not torch.is_tensor(value) or value.dim() < 1:
                return None
            row = value[index].detach().float().reshape(-1)
            if row.numel() == 0 or not bool(torch.isfinite(row[0]).item()):
                return None
            return float(row[0].item())

        def diagnostic_argmax_value(
            outputs: Dict[str, torch.Tensor],
            score_key: str,
            value_key: str,
            index: int,
        ) -> Optional[float]:
            score = outputs.get(score_key)
            value = outputs.get(value_key)
            if not torch.is_tensor(score) or not torch.is_tensor(value):
                return None
            score_row = score[index].detach().float().reshape(-1)
            value_row = value[index].detach().float().reshape(-1)
            if score_row.numel() == 0 or score_row.numel() != value_row.numel():
                return None
            argmax = int(score_row.argmax().item())
            selected = value_row[argmax]
            if not bool(torch.isfinite(selected).item()):
                return None
            return float(selected.item())

        def diagnostic_max_value(
            outputs: Dict[str, torch.Tensor], key: str, index: int
        ) -> Optional[float]:
            value = outputs.get(key)
            if not torch.is_tensor(value) or value.dim() < 1:
                return None
            row = value[index].detach().float().reshape(-1)
            finite = row[torch.isfinite(row)]
            if finite.numel() == 0:
                return None
            return float(finite.max().item())

        for beta in self.betas:
            neg_score, neg_iou = _best_scores_and_iou(neg_outputs, targets, cfg, beta)
            pos_score, pos_iou = _best_scores_and_iou(pos_outputs, targets, cfg, beta)
            neg_iou_score, _neg_iou_best = _score_at_best_iou(neg_outputs, targets, cfg, beta)
            pos_iou_score, _pos_iou_best = _score_at_best_iou(pos_outputs, targets, cfg, beta)
            for i, ok in enumerate(valid_np):
                finite = bool(
                    ok
                    and np.isfinite(pos_score[i])
                    and np.isfinite(neg_score[i])
                    and np.isfinite(pos_iou[i])
                    and np.isfinite(neg_iou[i])
                    and np.isfinite(pos_iou_score[i])
                    and np.isfinite(neg_iou_score[i])
                )
                self.eval_records[beta].append(
                    make_eval_record(
                        self.manifest,
                        index=int(manifest_start_index) + i,
                        run_id=f"{self.run_prefix}:b{beta:g}",
                        valid=finite,
                        meta=metas[i],
                        values={
                            "train_scope": self.train_scope,
                            "eval_scope": self.eval_scope,
                            "beta": float(beta),
                            "pos_score": float(pos_score[i]),
                            "neg_score": float(neg_score[i]),
                            "pos_iou": float(pos_iou[i]),
                            "neg_iou": float(neg_iou[i]),
                            "pos_best_iou_query_score": float(pos_iou_score[i]),
                            "neg_best_iou_query_score": float(neg_iou_score[i]),
                            "pos_reference_global_logit": diagnostic_value(
                                pos_outputs,
                                "stage_b_dense_duty_reference_global_confidence_logits",
                                i,
                            ),
                            "neg_reference_global_logit": diagnostic_value(
                                neg_outputs,
                                "stage_b_dense_duty_reference_global_confidence_logits",
                                i,
                            ),
                            "pos_frozen_rank_full_expression_global_logit": diagnostic_value(
                                pos_outputs,
                                "stage_b_dense_duty_frozen_rank_full_expression_global_logits",
                                i,
                            ),
                            "neg_frozen_rank_full_expression_global_logit": diagnostic_value(
                                neg_outputs,
                                "stage_b_dense_duty_frozen_rank_full_expression_global_logits",
                                i,
                            ),
                            "pos_global_logit": diagnostic_value(
                                pos_outputs,
                                "stage_b_dense_duty_global_confidence_logits",
                                i,
                            ),
                            "neg_global_logit": diagnostic_value(
                                neg_outputs,
                                "stage_b_dense_duty_global_confidence_logits",
                                i,
                            ),
                            "pos_pool_absolute_logit": diagnostic_value(
                                pos_outputs,
                                "stage_b_dense_duty_confidence_pool_absolute_logits",
                                i,
                            ),
                            "neg_pool_absolute_logit": diagnostic_value(
                                neg_outputs,
                                "stage_b_dense_duty_confidence_pool_absolute_logits",
                                i,
                            ),
                            "pos_deployed_query_veto_depth": diagnostic_value(
                                pos_outputs,
                                "stage_b_dense_duty_deployed_query_veto_depth",
                                i,
                            ),
                            "neg_deployed_query_veto_depth": diagnostic_value(
                                neg_outputs,
                                "stage_b_dense_duty_deployed_query_veto_depth",
                                i,
                            ),
                            "pos_deployed_query_veto_gate": diagnostic_value(
                                pos_outputs,
                                "stage_b_dense_duty_deployed_query_veto_gate",
                                i,
                            ),
                            "neg_deployed_query_veto_gate": diagnostic_value(
                                neg_outputs,
                                "stage_b_dense_duty_deployed_query_veto_gate",
                                i,
                            ),
                            "pos_veto_sample_gate": diagnostic_value(
                                pos_outputs,
                                "stage_b_dense_duty_confidence_veto_sample_gate",
                                i,
                            ),
                            "neg_veto_sample_gate": diagnostic_value(
                                neg_outputs,
                                "stage_b_dense_duty_confidence_veto_sample_gate",
                                i,
                            ),
                            "pos_veto_coverage": diagnostic_value(
                                pos_outputs,
                                "stage_b_dense_duty_confidence_veto_coverage",
                                i,
                            ),
                            "neg_veto_coverage": diagnostic_value(
                                neg_outputs,
                                "stage_b_dense_duty_confidence_veto_coverage",
                                i,
                            ),
                            "pos_candidate_max_base_logit": diagnostic_argmax_value(
                                pos_outputs,
                                "stage_b_dense_duty_confidence_base_logits",
                                "stage_b_dense_duty_confidence_base_logits",
                                i,
                            ),
                            "neg_candidate_max_base_logit": diagnostic_argmax_value(
                                neg_outputs,
                                "stage_b_dense_duty_confidence_base_logits",
                                "stage_b_dense_duty_confidence_base_logits",
                                i,
                            ),
                            "pos_candidate_max_mismatch_gate": diagnostic_argmax_value(
                                pos_outputs,
                                "stage_b_dense_duty_confidence_base_logits",
                                "stage_b_dense_duty_confidence_mismatch_gate",
                                i,
                            ),
                            "neg_candidate_max_mismatch_gate": diagnostic_argmax_value(
                                neg_outputs,
                                "stage_b_dense_duty_confidence_base_logits",
                                "stage_b_dense_duty_confidence_mismatch_gate",
                                i,
                            ),
                            "pos_patch_candidate_max_logit": diagnostic_max_value(
                                pos_outputs,
                                "stage_b_v15_candidate_patch_logits",
                                i,
                            ),
                            "neg_patch_candidate_max_logit": diagnostic_max_value(
                                neg_outputs,
                                "stage_b_v15_candidate_patch_logits",
                                i,
                            ),
                        },
                    )
                )
                if not ok:
                    continue
                self.records[beta].append(
                    {
                        "pos_score": float(pos_score[i]),
                        "tn_score": float(neg_score[i]),
                        "pos_iou": float(pos_iou[i]),
                        "tn_iou": float(neg_iou[i]),
                        "pos_best_iou_query_score": float(pos_iou_score[i]),
                        "tn_best_iou_query_score": float(neg_iou_score[i]),
                    }
                )
        for i, ok in enumerate(valid_np):
            if ok:
                self.metas.append(metas[i])

    def results(
        self,
        *,
        checkpoint: str,
        threshold_tprs: List[float],
        elapsed: float,
        batch_size: int,
        num_workers: int,
        seed: int,
        max_batches: int,
        max_pairs: int,
    ) -> List[Dict[str, Any]]:
        run_prefix = _ckpt_run_prefix(checkpoint)
        rows: List[Dict[str, Any]] = []
        for beta in self.betas:
            recs = self.records[beta]
            summary = _summarize_arrays(
                pos_scores=np.asarray([r["pos_score"] for r in recs], dtype=np.float32),
                neg_scores=np.asarray([r["tn_score"] for r in recs], dtype=np.float32),
                pos_iou=np.asarray([r["pos_iou"] for r in recs], dtype=np.float32),
                neg_iou=np.asarray([r["tn_iou"] for r in recs], dtype=np.float32),
                pos_iou_score=np.asarray([r["pos_best_iou_query_score"] for r in recs], dtype=np.float32),
                neg_iou_score=np.asarray([r["tn_best_iou_query_score"] for r in recs], dtype=np.float32),
                threshold_tprs=threshold_tprs,
            )
            summary.update(
                {
                    "run_id": f"{run_prefix}:b{beta:g}",
                    "checkpoint": str(checkpoint),
                    "checkpoint_name": Path(checkpoint).name,
                    "checkpoint_run_prefix": run_prefix,
                    "beta": float(beta),
                    "seconds": float(elapsed),
                    "batch_size": int(batch_size),
                    "num_workers": int(num_workers),
                    "seed": int(seed),
                    "max_batches": int(max_batches),
                    "max_pairs": int(max_pairs),
                    "train_scope": self.train_scope,
                    "eval_scope": self.eval_scope,
                    "invalid_positive_pairs": int(self.invalid_positive),
                    "by_split": _summarize_group(recs, self.metas, threshold_tprs, "eval_split"),
                    "by_category": _summarize_group(recs, self.metas, threshold_tprs, "category"),
                }
            )
            rows.append(summary)
        return rows


@torch.no_grad()
def evaluate_checkpoint(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    meta_rows: List[Dict[str, Any]],
    device: torch.device,
    betas: List[float],
    threshold_tprs: List[float],
    batch_size: int,
    num_workers: int,
    seed: int,
    amp: bool,
    max_batches: int,
    max_pairs: int,
    log_every: int,
    records_output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    manifest = load_eval_manifest(
        Path(datasetinfo["anno"]),
        task="tn",
        split="global",
        manifest_key="tn_global",
    )
    eval_scope = _validate_adapter_tn_eval_manifest(cfg, manifest.rows)
    train_scope = (
        str(getattr(cfg, "stage_b_gdino_tn_scope", "")).strip() or None
        if bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
        else None
    )
    run_prefix = _ckpt_run_prefix(ckpt_path)
    total_batches = len(loader)
    if max_pairs > 0:
        total_batches = min(total_batches, math.ceil(int(max_pairs) / max(1, int(batch_size))))
    if max_batches > 0:
        total_batches = min(total_batches, int(max_batches))
    acc = TnPairAccumulator(
        betas,
        manifest=manifest,
        run_prefix=run_prefix,
        train_scope=train_scope,
        eval_scope=eval_scope,
    )
    start = time.time()
    offset = 0
    print(
        f"[INFO] TN eval ckpt={ckpt_path} pairs={len(loader.dataset)} batches={len(loader)} "
        f"batch_size={batch_size} betas={betas}"
    )
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= int(max_batches):
            break
        raw_bsz = len(batch[1])
        if max_pairs > 0 and offset >= int(max_pairs):
            break
        manifest_start_index = offset
        _validate_eval_manifest_batch_alignment(
            list(batch[1]), manifest, manifest_start_index
        )
        metas = meta_rows[offset : offset + raw_bsz]
        offset += raw_bsz
        neg_outputs, pos_outputs, targets, valid_pos = _forward_pair(model, batch, device, amp=amp)
        acc.update(
            neg_outputs=neg_outputs,
            pos_outputs=pos_outputs,
            targets=targets,
            valid_pos=valid_pos,
            metas=metas,
            cfg=cfg,
            manifest_start_index=manifest_start_index,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (batch_i == 0 or (batch_i + 1) % int(log_every) == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            eta = elapsed / max(1, done) * max(0, total_batches - done)
            used = sum(len(v) for v in acc.records.values()) // max(1, len(acc.records))
            print(
                f"[INFO] {Path(ckpt_path).parent.name}/{Path(ckpt_path).name}: "
                f"batch {done}/{total_batches}, valid_pairs={used}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
            )
    rows = acc.results(
        checkpoint=ckpt_path,
        threshold_tprs=threshold_tprs,
        elapsed=time.time() - start,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        max_batches=max_batches,
        max_pairs=max_pairs,
    )
    for row in rows:
        if records_output_dir is None:
            continue
        beta = float(row["beta"])
        records_path = Path(records_output_dir) / (
            f"{run_prefix}__tn_global__{_safe_name(f'b{beta:g}')}.records.jsonl"
        )
        write_eval_records(records_path, acc.eval_records[beta])
        row.update(
            {
                "records_jsonl": str(records_path),
                "manifest_sha256": manifest.sha256,
                "manifest_n": manifest.size,
                **tn_manifest_binding_summary_fields(manifest),
                "invalid_records": int(
                    sum(not bool(record.get("valid")) for record in acc.eval_records[beta])
                ),
            }
        )
    return rows


def _write_summary(output_dir: Path, rows: List[Dict[str, Any]], primary_metric: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reverse = primary_metric in {"pair_win_rate", "score_gap_mean", "score_gap_median"}
    ranking = [
        dict(row, rank=i + 1)
        for i, row in enumerate(
            sorted(
                rows,
                key=lambda r: (
                    float(r.get(primary_metric, 0.0)),
                    -float(r.get("pair_win_rate", 0.0)),
                    float(r.get("tn_score_mean", 0.0)),
                ),
                reverse=reverse,
            )
        )
    ]
    payload = {"primary_metric": primary_metric, "ranking": ranking, "results": rows}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    split_names: List[str] = []
    seen = set()
    for row in rows:
        for split in row.get("by_split", {}):
            if split not in seen:
                seen.add(split)
                split_names.append(split)
    header = [
        "# Stage-B TN-Val Rejection Evaluation",
        "",
        f"Primary metric: `{primary_metric}`. Lower is better for FPR metrics.",
        "",
        "| rank | run | beta | global fpr95 | global fpr90 | global pair win | target fpr95 | target fpr90 | target pair win | target gap | pos IoU50 | TN IoU50 | pairs |"
        + "".join(f" {split} fpr95 |" for split in split_names),
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        + "".join("---:|" for _ in split_names),
    ]
    lines = list(header)
    for row in ranking:
        split_cells = []
        by_split = row.get("by_split", {})
        for split in split_names:
            split_cells.append(f"{float(by_split.get(split, {}).get('fpr95tpr', 0.0)):.6f}")
        lines.append(
            f"| {int(row['rank'])} | `{row['run_id']}` | {float(row['beta']):g} | "
            f"{float(row.get('fpr95tpr', 0.0)):.6f} | "
            f"{float(row.get('fpr90tpr', 0.0)):.6f} | "
            f"{float(row.get('pair_win_rate', 0.0)):.6f} | "
            f"{float(row.get('target_fpr95tpr', 0.0)):.6f} | "
            f"{float(row.get('target_fpr90tpr', 0.0)):.6f} | "
            f"{float(row.get('target_pair_win_rate', 0.0)):.6f} | "
            f"{float(row.get('target_score_gap_mean', 0.0)):.6f} | "
            f"{float(row.get('pos_top1_iou50', 0.0)):.6f} | "
            f"{float(row.get('tn_top1_iou50', 0.0)):.6f} | "
            f"{int(row.get('num_pairs', 0))} | "
            + " | ".join(split_cells)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        checkpoint = str(row.get("checkpoint", ""))
        if not run_id:
            continue
        by_key[(run_id, checkpoint)] = row
    return list(by_key.values())


def _load_existing_rows(output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not output_dir.exists():
        return rows
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            rows.extend([row for row in data if isinstance(row, dict)])
    return _dedupe_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-B true-negative phrase rejection on TN val pairs.")
    parser.add_argument("--config", default="config/cfg_patch_stage_b.py")
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/stageb_tn_val_compare")
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))
    parser.add_argument(
        "--tn_jsonl",
        default=None,
        help="TN patch_episode jsonl. Defaults to DATA_ROOT/patch_episode_prebuilt/refexp_tn_stageb_v1.jsonl.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--threshold_tprs", nargs="+", type=float, default=[0.75, 0.9, 0.95])
    parser.add_argument("--splits", nargs="+", default=["refcocop_val", "refcocog_val"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=0, help="Maximum TN pairs after split filtering; 0 means full.")
    parser.add_argument("--max_pairs_per_split", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--primary_metric", default="fpr95tpr")
    parser.add_argument("--append_existing", action="store_true", help="Include existing per-checkpoint JSON rows in summary.")
    parser.add_argument("--stage_b_v7_candidate_topk", type=int, default=None)
    parser.add_argument("--stage_b_v11_candidate_topk", type=int, default=None)
    parser.add_argument("--stage_b_v7_patch_prior_weight", type=float, default=None)
    parser.add_argument("--stage_b_v7_phrase_agg", default=None)
    parser.add_argument("--stage_b_v7_phrase_mean_weight", type=float, default=None)
    parser.add_argument("--stage_b_v7_phrase_softmin_tau", type=float, default=None)
    parser.add_argument("--exclude_train_jsonl", nargs="*", default=[])
    parser.add_argument("--holdout_level", choices=["none", "ann", "image"], default="none")
    parser.add_argument(
        "--no_per_example_records",
        action="store_true",
        help="Disable canonical *.records.jsonl output used by the paired final gate.",
    )
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    tn_jsonl = Path(args.tn_jsonl) if args.tn_jsonl else data_root / "patch_episode_prebuilt" / "refexp_tn_stageb_v1.jsonl"
    if not tn_jsonl.exists():
        raise FileNotFoundError(tn_jsonl)

    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    adapter_enabled = bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
    data_driven_score = bool(getattr(cfg, "stage_b_data_driven_score", False))
    u0_patch_rank = bool(getattr(cfg, "stage_b_u0_patch_rank", False))
    native_patch_category = bool(
        getattr(cfg, "stage_b_native_patch_category", False)
    )
    if native_patch_category:
        if len(args.ckpts) != 1:
            raise ValueError(
                "stage_b_native_patch_category TN/FPR evaluation requires "
                "exactly one checkpoint"
            )
        if not bool(
            getattr(cfg, "stage_b_native_patch_confidence_trained", False)
        ):
            raise RuntimeError(
                "native patch-category TN/FPR evaluation is forbidden until an "
                "independent confidence head is explicitly marked trained"
            )
    if u0_patch_rank and not adapter_enabled:
        raise ValueError(
            "stage_b_u0_patch_rank evaluation requires stage_b_gdino_score_adapter"
        )
    cfg.patch_only = not (
        adapter_enabled or data_driven_score or native_patch_category
    )
    cfg.patch_only_compute_text_logits = cfg.patch_only
    cfg.build_text_token_masks = True
    cfg.use_coco_eval = False
    # Evaluation runs without gradients, so checkpointing only increases
    # allocator pressure and can trigger large transient CUDA allocations.
    cfg.use_checkpoint = False
    cfg.use_transformer_ckpt = False
    cfg.batch_size = int(args.batch_size)
    cfg.text_mask_warn_limit = 0
    for key in (
        "stage_b_v7_candidate_topk",
        "stage_b_v11_candidate_topk",
        "stage_b_v7_patch_prior_weight",
        "stage_b_v7_phrase_agg",
        "stage_b_v7_phrase_mean_weight",
        "stage_b_v7_phrase_softmin_tau",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    holdout_ann_keys, holdout_image_ids = load_holdout_keys(args.exclude_train_jsonl)
    if args.holdout_level != "none":
        print(
            f"[INFO] holdout level={args.holdout_level} "
            f"ann_keys={len(holdout_ann_keys)} image_ids={len(holdout_image_ids)}"
        )
    tn_eval_jsonl, meta_rows, counts = _build_tn_eval_jsonl(
        data_root=data_root,
        output_dir=output_dir,
        tn_jsonl=tn_jsonl,
        splits=list(args.splits),
        max_pairs=int(args.max_pairs),
        max_pairs_per_split=int(args.max_pairs_per_split),
        holdout_level=args.holdout_level,
        holdout_ann_keys=holdout_ann_keys,
        holdout_image_ids=holdout_image_ids,
    )
    if not meta_rows:
        raise RuntimeError(f"No TN rows selected from {tn_jsonl} for splits={args.splits}")
    selected_eval_rows = list(_iter_jsonl(tn_eval_jsonl))
    eval_scope = _validate_adapter_tn_eval_manifest(cfg, selected_eval_rows)
    eval_protocol = (
        "stageb_vlm_verified_strict_tn_v2"
        if eval_scope is not None
        and all(
            row.get("manifest_schema", None)
            == "stageb_vlm_verified_strict_tn_v2"
            for row in selected_eval_rows
        )
        else ("adapter_training_pair_schema" if eval_scope is not None else None)
    )
    datasetinfo = _make_datasetinfo(
        data_root,
        tn_eval_jsonl,
        adapter_eval_scope=eval_scope,
        adapter_eval_protocol=eval_protocol,
        u0_patch_rank=u0_patch_rank,
        data_driven_score=data_driven_score,
    )
    print(f"[INFO] built TN eval jsonl: {tn_eval_jsonl} rows={len(meta_rows)} split_counts={counts}")

    rows: List[Dict[str, Any]] = _load_existing_rows(output_dir) if bool(args.append_existing) else []
    if rows:
        print(f"[INFO] loaded {len(rows)} existing result rows from {output_dir}")
    for ckpt_i, ckpt_path in enumerate(args.ckpts):
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(args.ckpts)}: {ckpt_path}")
        _set_seed(int(args.seed))
        model = _load_model(cfg, ckpt_path, device)
        ckpt_rows = evaluate_checkpoint(
            cfg=cfg,
            model=model,
            ckpt_path=ckpt_path,
            datasetinfo=datasetinfo,
            meta_rows=meta_rows,
            device=device,
            betas=list(args.betas),
            threshold_tprs=list(args.threshold_tprs),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            amp=bool(args.amp),
            max_batches=int(args.max_batches),
            max_pairs=int(args.max_pairs),
            log_every=int(args.log_every),
            records_output_dir=(
                None if args.no_per_example_records else output_dir / "per_example_records"
            ),
        )
        rows = _dedupe_rows(rows + ckpt_rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{_ckpt_run_prefix(ckpt_path)}.json").write_text(
            json.dumps(ckpt_rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_summary(output_dir, rows, str(args.primary_metric))
        for row in ckpt_rows:
            print(
                f"[RESULT] {row['run_id']}: fpr95={row.get('fpr95tpr', 0.0):.6f} "
                f"fpr90={row.get('fpr90tpr', 0.0):.6f} pair_win={row.get('pair_win_rate', 0.0):.6f} "
                f"gap={row.get('score_gap_mean', 0.0):.6f} "
                f"tn_iou50={row.get('tn_top1_iou50', 0.0):.6f} pairs={row.get('num_pairs', 0)}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_summary(output_dir, rows, str(args.primary_metric))
    print(f"[INFO] wrote {output_dir / 'summary.json'}")
    print(f"[INFO] wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
