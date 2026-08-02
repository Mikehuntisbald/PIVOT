#!/usr/bin/env python3
"""Read-only C1 role and counterfactual diagnostics for the v19 scorer.

This evaluator deliberately separates three score surfaces on the unchanged
Stage-A patch Top-K candidates:

* the canonical patch logit;
* the pure full-text logit, recovered exactly by removing the configured patch
  prior from the scorer's fused logit; and
* the deployed patch-prior plus full-text rank logit.

``text_only`` always means text-only ranking on the fixed patch candidate set.
When ``--true_role_swap`` is requested, G5 is evaluated by two deterministic
forwards over the same image/support episode.  The first forward uses the
canonical-text GroundingDINO logits over all frozen queries to choose Top-K;
the second forces those exact query indices into the existing full-text scorer.
The evaluator fails closed unless all-query states, boxes, and patch logits are
bitwise identical across the two forwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util import box_ops  # noqa: E402
from tools.eval_refcoco_stageb import (  # noqa: E402
    _build_loader as _build_ref_loader,
    _build_split_jsonl,
    _default_splits,
    _diagnostic_patch_rank_candidate_logits,
    _forward as _forward_ref,
    _load_canonical_name_maps,
    _load_model,
    _load_phrase_maps,
    _make_datasetinfo as _make_ref_datasetinfo,
    _normalized_cxcywh_to_xyxy,
    _pad_target_mask as _pad_ref_target_mask,
    _prepare_patch_batch as _prepare_ref_patch_batch,
    _target_texts as _ref_target_texts,
)
from tools.eval_stageb_tn_val import (  # noqa: E402
    _build_loader as _build_tn_loader,
    _forward_pair,
    _make_datasetinfo as _make_tn_datasetinfo,
    _pad_target_mask as _pad_tn_target_mask,
    _prepare_patch_batch as _prepare_tn_patch_batch,
    _rank_positive_captions,
    _target_texts as _tn_target_texts,
)
from tools.stageb_ref_split_contract import REF_SPLITS  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "stageb-v19-role-causal-v4"
ORACLE_TOPKS = (1, 5, 10, 50)
ROLE_SWAP_STATUS = {
    "supported": False,
    "reason": (
        "v19 exposes patch-canonical Top-K admission and full-text scores only "
        "for those admitted candidates; it does not expose text-canonical "
        "Top-K admission with a faithful patch/full-expression rank consumer"
    ),
}
TRUE_ROLE_SWAP_STATUS = {
    "supported": True,
    "id": "G5",
    "candidate_source": "canonical_text_allquery_topk",
    "admission_score": "max_logit_over_canonical_phrase_tokens",
    "rank_score": "patch_prior_plus_fulltext_rank_logit",
    "frozen_tensor_check": "bitwise_hs_boxes_patch_logits",
}
TRUE_ROLE_SWAP_ROUTE = "text_canonical_admission_patch_fulltext_rank"
VALIDATION_REF_SPLITS = ("refcoco_val", "refcocop_val", "refcocog_val")
FORMAL_PROFILES = ("validation", "final")


def _new_g5_equality_receipt() -> Dict[str, int]:
    return {
        "forced_forward_count": 0,
        "selector_no_grad_count": 0,
        "hs_bitwise_equal_count": 0,
        "boxes_bitwise_equal_count": 0,
        "patch_logits_bitwise_equal_count": 0,
        "candidate_order_bitwise_equal_count": 0,
    }


def _finalize_g5_equality_receipt(receipt: Mapping[str, int]) -> Dict[str, Any]:
    count = int(receipt.get("forced_forward_count", 0))
    equality_fields = (
        "selector_no_grad_count",
        "hs_bitwise_equal_count",
        "boxes_bitwise_equal_count",
        "patch_logits_bitwise_equal_count",
        "candidate_order_bitwise_equal_count",
    )
    passed = count > 0 and all(int(receipt.get(key, -1)) == count for key in equality_fields)
    if not passed:
        raise RuntimeError("G5 tensor-equality receipt is incomplete")
    return {
        "status": "passed",
        "comparison": "torch.equal_bitwise",
        "no_grad_required": True,
        **{key: int(value) for key, value in receipt.items()},
    }


def _require_tensor(outputs: Mapping[str, Any], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"role diagnostic requires tensor output {key!r}")
    return value.detach()


def _target_candidate_ious(
    candidate_boxes_xyxy: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Return maximum IoU to any target box for each fixed candidate."""

    batch_size, candidate_count, _ = candidate_boxes_xyxy.shape
    if len(targets) != batch_size:
        raise ValueError(
            f"target count {len(targets)} does not match batch size {batch_size}"
        )
    out = candidate_boxes_xyxy.new_full((batch_size, candidate_count), float("nan"))
    for batch_index, target in enumerate(targets):
        boxes = target.get("boxes")
        if not torch.is_tensor(boxes) or boxes.numel() == 0:
            continue
        if boxes.dim() != 2 or int(boxes.shape[-1]) != 4:
            raise ValueError("target boxes must have shape (G,4)")
        gt_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach().to(
                device=candidate_boxes_xyxy.device, dtype=torch.float32
            )
        ).clamp(0.0, 1.0)
        iou = box_ops.box_iou(candidate_boxes_xyxy[batch_index], gt_xyxy)[0]
        out[batch_index] = iou.max(dim=1).values.to(dtype=out.dtype)
    return out


def extract_role_components(
    outputs: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    *,
    patch_rank_weight: float,
    candidate_source: str = "patch_topk",
) -> Dict[str, Any]:
    """Validate v19 outputs and expose patch, pure-text, and fused surfaces."""

    patch_rank_weight = float(patch_rank_weight)
    if candidate_source not in {"patch_topk", "canonical_text_topk"}:
        raise ValueError(f"unsupported role candidate source {candidate_source!r}")
    logits_by_weight, candidate_idx, expression_valid = (
        _diagnostic_patch_rank_candidate_logits(
            dict(outputs),
            weights=(0.0, patch_rank_weight),
            contract_weight=patch_rank_weight,
            require_patch_topk_order=(candidate_source == "patch_topk"),
        )
    )
    fused_logits = _require_tensor(outputs, "stage_b_v11_final_phrase_logits")
    text_logits = logits_by_weight[0.0]
    # At the contract point the shared validator returns the authoritative
    # tensor bitwise, avoiding a subtract/add round trip.
    contract_logits = logits_by_weight[patch_rank_weight]
    if not torch.equal(contract_logits, fused_logits):
        raise RuntimeError("contract fused logits are not bitwise authoritative")

    pred_boxes_xyxy = _normalized_cxcywh_to_xyxy(
        _require_tensor(outputs, "pred_boxes"), name="role diagnostic pred_boxes"
    )
    all_query_ious = _target_candidate_ious(pred_boxes_xyxy, targets)
    candidate_boxes_xyxy = torch.gather(
        pred_boxes_xyxy,
        1,
        candidate_idx.unsqueeze(-1).expand(-1, -1, 4),
    )
    emitted_candidate_boxes = _require_tensor(
        outputs, "stage_b_v11_candidate_boxes"
    ).float()
    gathered_cxcywh = torch.gather(
        _require_tensor(outputs, "pred_boxes").float(),
        1,
        candidate_idx.unsqueeze(-1).expand(-1, -1, 4),
    )
    if not torch.equal(emitted_candidate_boxes.float(), gathered_cxcywh):
        raise ValueError(
            "stage_b_v11_candidate_boxes are not an exact gather of pred_boxes"
        )

    patch_logits = _require_tensor(
        outputs, "stage_b_v15_candidate_patch_logits"
    ).float()
    all_query_patch_logits = _require_tensor(outputs, "pred_logits_patch").float()
    if all_query_patch_logits.dim() == 3:
        if int(all_query_patch_logits.shape[-1]) != 1:
            raise ValueError(
                "role diagnostic requires exactly one canonical support patch"
            )
        all_query_patch_logits = all_query_patch_logits[..., 0]
    if tuple(all_query_patch_logits.shape) != tuple(pred_boxes_xyxy.shape[:2]):
        raise ValueError("all-query patch logits do not align with pred_boxes")
    gathered_patch_logits = torch.gather(all_query_patch_logits, 1, candidate_idx)
    if not torch.equal(gathered_patch_logits, patch_logits):
        raise ValueError(
            "candidate patch logits are not an exact gather of pred_logits_patch"
        )
    if candidate_source == "patch_topk":
        expected_idx = torch.topk(
            all_query_patch_logits,
            int(candidate_idx.shape[1]),
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        if not torch.equal(candidate_idx, expected_idx):
            raise ValueError("patch candidate indices are not exact dense patch Top-K")
    return {
        "candidate_source": candidate_source,
        "candidate_idx": candidate_idx,
        "candidate_boxes_xyxy": candidate_boxes_xyxy,
        "candidate_ious": _target_candidate_ious(candidate_boxes_xyxy, targets),
        "all_query_ious": all_query_ious,
        "all_query_patch_logits": all_query_patch_logits,
        "expression_valid": expression_valid,
        "patch_logits": patch_logits,
        "text_logits": text_logits.float(),
        "fused_logits": fused_logits.float(),
    }


def canonical_text_admission_scores(
    text_logits: torch.Tensor,
    canonical_phrase_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the canonical-text all-query score used by the G5 admission.

    GroundingDINO category evaluation uses the maximum logit over the token
    positions assigned to a category phrase.  Keeping this surface in logit
    space preserves exact query ordering and avoids an arbitrary probability
    pooling choice.
    """

    if text_logits.dim() != 3:
        raise ValueError("canonical text logits must have shape (B,Q,T)")
    if canonical_phrase_mask.dim() != 2 or tuple(canonical_phrase_mask.shape) != (
        int(text_logits.shape[0]),
        int(text_logits.shape[2]),
    ):
        raise ValueError("canonical phrase mask must have shape (B,T)")
    mask = canonical_phrase_mask.to(device=text_logits.device, dtype=torch.bool)
    if bool((~mask.any(dim=-1)).any().item()):
        raise ValueError("every canonical caption must expose at least one phrase token")
    masked = text_logits.detach().float().masked_fill(
        ~mask[:, None, :], torch.finfo(torch.float32).min
    )
    scores = masked.max(dim=-1).values
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("canonical text admission scores must be finite")
    return scores


def _canonical_phrase_mask_from_output(
    outputs: Mapping[str, Any],
) -> torch.Tensor:
    """Read the model-generated canonical phrase mask before dataset overrides."""

    logits = _require_tensor(outputs, "pred_logits_text")
    mask = _require_tensor(outputs, "phrase_to_token_mask")
    if mask.dim() != 3 or int(mask.shape[0]) != int(logits.shape[0]):
        raise ValueError("model-generated phrase_to_token_mask must be (B,P,T)")
    if int(mask.shape[1]) != 1:
        raise ValueError(
            "G5 canonical admission requires exactly one canonical phrase per row"
        )
    token_count = int(logits.shape[-1])
    if int(mask.shape[-1]) < token_count:
        raise ValueError("canonical phrase mask is narrower than pred_logits_text")
    return mask[:, 0, :token_count].to(dtype=torch.bool)


def _text_topk_candidate_indices(
    outputs: Mapping[str, Any], *, candidate_count: int
) -> torch.Tensor:
    logits = _require_tensor(outputs, "pred_logits_text")
    mask = _canonical_phrase_mask_from_output(outputs)
    scores = canonical_text_admission_scores(logits, mask)
    count = min(int(candidate_count), int(scores.shape[1]))
    if count <= 0:
        raise ValueError("G5 candidate_count must be positive")
    return torch.topk(scores, count, dim=1, largest=True, sorted=True).indices


@torch.no_grad()
def _call_fixed_text_model(
    model,
    *,
    samples,
    targets,
    stage_a_captions: Sequence[str],
    verifier_captions: Sequence[str],
    patches,
    patch_global,
    patch_mask,
    phrase_mask,
    canonical_mask,
    amp: bool,
    device: torch.device,
    compute_canonical_text: bool,
):
    # Dataset token masks are built against verifier captions.  Passing them
    # under the canonical Stage-A forward would overwrite the model-generated
    # canonical phrase mask needed for G5 admission, while the fixed scorer
    # already builds its own full-text masks from ``verifier_captions``.
    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        return model(
            samples,
            targets=targets,
            captions=list(stage_a_captions),
            patches=patches,
            patch_global=patch_global,
            patch_mask=patch_mask,
            patch_only=True,
            patch_only_compute_text_logits=bool(compute_canonical_text),
            disable_patch_dn=True,
            return_stage_b_v7_features=True,
            stage_b_v7_verifier_captions=list(verifier_captions),
        )


def _forced_candidate_forward(
    model,
    *,
    expected_outputs: Mapping[str, Any],
    forced_indices: torch.Tensor,
    call_kwargs: Mapping[str, Any],
    equality_receipt: Optional[Dict[str, int]] = None,
):
    """Run the scorer on forced indices and prove the all-query tensor is fixed."""

    root = model.module if hasattr(model, "module") else model
    name = "select_stage_b_v11_candidates"
    had_instance_override = name in getattr(root, "__dict__", {})
    old_instance_value = root.__dict__.get(name) if had_instance_override else None
    expected_hs = _require_tensor(expected_outputs, "hs")
    expected_boxes = _require_tensor(expected_outputs, "pred_boxes")
    expected_patch = _require_tensor(expected_outputs, "pred_logits_patch")
    forced_indices = forced_indices.detach().to(
        device=expected_boxes.device, dtype=torch.int64
    )
    selector_calls = 0

    def forced_selector(query_hs, pred_boxes, patch_score, topk):
        nonlocal selector_calls
        selector_calls += 1
        if torch.is_grad_enabled():
            raise RuntimeError("G5 forced selector executed with gradients enabled")
        if not torch.equal(query_hs.detach(), expected_hs):
            raise RuntimeError("G5 rerun changed the frozen all-query states")
        if not torch.equal(pred_boxes.detach(), expected_boxes):
            raise RuntimeError("G5 rerun changed the frozen all-query boxes")
        if not torch.equal(patch_score.detach(), expected_patch):
            raise RuntimeError("G5 rerun changed the frozen all-query patch logits")
        if int(topk) != int(forced_indices.shape[1]):
            raise RuntimeError(
                "G5 forced candidate count differs from configured scorer Top-K"
            )
        if equality_receipt is not None:
            equality_receipt["selector_no_grad_count"] += 1
            equality_receipt["hs_bitwise_equal_count"] += 1
            equality_receipt["boxes_bitwise_equal_count"] += 1
            equality_receipt["patch_logits_bitwise_equal_count"] += 1
        candidate_hs = torch.gather(
            query_hs.detach(),
            1,
            forced_indices.unsqueeze(-1).expand(-1, -1, query_hs.shape[-1]),
        )
        candidate_boxes = torch.gather(
            pred_boxes.detach(),
            1,
            forced_indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        return forced_indices, candidate_hs, candidate_boxes

    setattr(root, name, forced_selector)
    try:
        outputs = _call_fixed_text_model(model, **dict(call_kwargs))
    finally:
        if had_instance_override:
            setattr(root, name, old_instance_value)
        else:
            delattr(root, name)
    observed = _require_tensor(outputs, "stage_b_v11_candidate_idx")
    if selector_calls != 1:
        raise RuntimeError(
            f"G5 candidate selector must run exactly once, observed {selector_calls}"
        )
    if not torch.equal(observed, forced_indices):
        raise RuntimeError("G5 scorer did not preserve the forced candidate order")
    if equality_receipt is not None:
        equality_receipt["forced_forward_count"] += 1
        equality_receipt["candidate_order_bitwise_equal_count"] += 1
    return outputs


def _finite_float(value: torch.Tensor | float) -> Optional[float]:
    result = float(value.item() if torch.is_tensor(value) else value)
    return result if math.isfinite(result) else None


def _ranked_oracle(
    scores: torch.Tensor,
    candidate_ious: torch.Tensor,
    *,
    oracle_topks: Iterable[int] = ORACLE_TOPKS,
) -> Dict[str, Dict[str, Any]]:
    if scores.dim() != 1 or candidate_ious.dim() != 1:
        raise ValueError("route scores and IoUs must be one-dimensional")
    if int(scores.numel()) != int(candidate_ious.numel()) or not scores.numel():
        raise ValueError("route scores and IoUs must be non-empty and aligned")
    order = torch.argsort(scores, descending=True, stable=True)
    result: Dict[str, Dict[str, Any]] = {}
    for topk in tuple(int(value) for value in oracle_topks):
        if topk <= 0:
            raise ValueError("oracle Top-K values must be positive")
        effective = min(topk, int(order.numel()))
        best = candidate_ious.index_select(0, order[:effective]).max()
        result[str(topk)] = {
            "effective_k": effective,
            "best_iou": _finite_float(best),
            "recall_iou50": bool(best >= 0.5),
        }
    best = candidate_ious.index_select(0, order).max()
    result["all"] = {
        "effective_k": int(order.numel()),
        "best_iou": _finite_float(best),
        "recall_iou50": bool(best >= 0.5),
    }
    return result


def _selected_route(
    scores: torch.Tensor,
    candidate_idx: torch.Tensor,
    candidate_ious: torch.Tensor,
    *,
    oracle_topks: Iterable[int] = ORACLE_TOPKS,
) -> Dict[str, Any]:
    ranked_oracle = _ranked_oracle(
        scores, candidate_ious, oracle_topks=oracle_topks
    )
    selected = int(torch.argsort(scores, descending=True, stable=True)[0].item())
    return {
        "candidate_position": selected,
        "query_index": int(candidate_idx[selected].item()),
        "selected_iou": _finite_float(candidate_ious[selected]),
        "score_logit": _finite_float(scores[selected]),
        "score_probability": _finite_float(scores[selected].sigmoid()),
        "ranked_oracle": ranked_oracle,
    }


def route_role_records(
    components: Mapping[str, Any],
    *,
    metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    oracle_topks: Iterable[int] = ORACLE_TOPKS,
) -> List[Dict[str, Any]]:
    """Create one JSON-safe role record per valid expression slot."""

    candidate_idx = components["candidate_idx"]
    candidate_ious = components["candidate_ious"]
    expression_valid = components["expression_valid"]
    patch_logits = components["patch_logits"]
    text_logits = components["text_logits"]
    fused_logits = components["fused_logits"]
    all_query_ious = components["all_query_ious"]
    all_query_patch_logits = components["all_query_patch_logits"]
    candidate_source = str(components["candidate_source"])
    batch_size, candidate_count = candidate_idx.shape
    slot_count = int(expression_valid.shape[1])
    if metadata is not None and len(metadata) != batch_size:
        raise ValueError("metadata must contain exactly one row per batch sample")
    topks = tuple(int(value) for value in oracle_topks)
    if any(value <= 0 for value in topks):
        raise ValueError("oracle Top-K values must be positive")

    records: List[Dict[str, Any]] = []
    for batch_index in range(batch_size):
        base_meta = dict(metadata[batch_index]) if metadata is not None else {}
        for slot_index in range(slot_count):
            if not bool(expression_valid[batch_index, slot_index].item()):
                continue
            ious = candidate_ious[batch_index]
            oracle: Dict[str, Dict[str, Any]] = {}
            for topk in topks:
                effective = min(topk, candidate_count)
                best_iou = ious[:effective].max()
                oracle[str(topk)] = {
                    "effective_k": effective,
                    "best_iou": _finite_float(best_iou),
                    "recall_iou50": bool(best_iou >= 0.5),
                }
            all_candidate_best = ious.max()
            oracle["all"] = {
                "effective_k": candidate_count,
                "best_iou": _finite_float(all_candidate_best),
                "recall_iou50": bool(all_candidate_best >= 0.5),
            }
            all_query_ids = torch.arange(
                int(all_query_patch_logits.shape[1]),
                device=candidate_idx.device,
                dtype=candidate_idx.dtype,
            )
            all_query_patch_route = _selected_route(
                all_query_patch_logits[batch_index],
                all_query_ids,
                all_query_ious[batch_index],
                oracle_topks=topks,
            )
            patch_route = _selected_route(
                patch_logits[batch_index],
                candidate_idx[batch_index],
                ious,
                oracle_topks=topks,
            )
            text_route = _selected_route(
                text_logits[batch_index, :, slot_index],
                candidate_idx[batch_index],
                ious,
                oracle_topks=topks,
            )
            fused_route = _selected_route(
                fused_logits[batch_index, :, slot_index],
                candidate_idx[batch_index],
                ious,
                oracle_topks=topks,
            )
            if (
                candidate_source == "patch_topk"
                and patch_route["query_index"]
                != all_query_patch_route["query_index"]
            ):
                raise RuntimeError(
                    "patch Top-K route changed the all-query patch winner"
                )
            records.append(
                {
                    **base_meta,
                    "expression_slot": slot_index,
                    "candidate_source": candidate_source,
                    "all_query_count": int(all_query_patch_logits.shape[1]),
                    "candidate_count": candidate_count,
                    "candidate_oracle": oracle,
                    "patch_canonical_top1_logit": patch_route["score_logit"],
                    "fulltext_top1_logit": text_route["score_logit"],
                    "fused_top1_logit": fused_route["score_logit"],
                    "routes": {
                        "patch_all_queries": all_query_patch_route,
                        "patch_only": patch_route,
                        "text_only": {
                            **text_route,
                            "scope": "fixed_patch_topk_candidates",
                            "global_text_candidate_generation_supported": False,
                        },
                        "patch_admission_text_rank": fused_route,
                    },
                    "true_role_swap": dict(ROLE_SWAP_STATUS),
                }
            )
    return records


def merge_true_role_swap_records(
    patch_records: Sequence[Mapping[str, Any]],
    swap_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach the true G5 route to aligned G2-G4 records."""

    if len(patch_records) != len(swap_records):
        raise ValueError("G5 records must align one-to-one with patch-route records")
    merged: List[Dict[str, Any]] = []
    identity_keys = (
        "image_id",
        "ann_id",
        "ref_id",
        "sent_id",
        "sample_id",
        "expression_slot",
    )
    for patch_record, swap_record in zip(patch_records, swap_records):
        for key in identity_keys:
            if patch_record.get(key) != swap_record.get(key):
                raise ValueError(f"G5 record identity drifted at {key!r}")
        if int(patch_record["candidate_count"]) != int(
            swap_record["candidate_count"]
        ):
            raise ValueError("G5 and patch routes must use the same candidate count")
        if int(patch_record["all_query_count"]) != int(
            swap_record["all_query_count"]
        ):
            raise ValueError("G5 rerun changed the all-query count")
        if patch_record["routes"]["patch_all_queries"] != swap_record["routes"][
            "patch_all_queries"
        ]:
            raise ValueError("G5 rerun changed the all-query patch route")
        out = dict(patch_record)
        routes = dict(patch_record["routes"])
        g5_route = dict(swap_record["routes"]["patch_admission_text_rank"])
        g5_route.update(
            {
                "scope": "canonical_text_allquery_topk_candidates",
                "candidate_source": "canonical_text_allquery_topk",
                "candidate_count_matched_to_patch_route": True,
            }
        )
        routes[TRUE_ROLE_SWAP_ROUTE] = g5_route
        out["routes"] = routes
        out["true_role_swap_candidate_oracle"] = dict(
            swap_record["candidate_oracle"]
        )
        out["true_role_swap"] = dict(TRUE_ROLE_SWAP_STATUS)
        merged.append(out)
    return merged


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else 0.0


def _oracle_key(value: str) -> Tuple[int, int]:
    return (1, 0) if str(value) == "all" else (0, int(value))


def _aggregate_ranked_oracle(
    routes: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not routes:
        return {}
    keys = sorted(routes[0]["ranked_oracle"], key=_oracle_key)
    if any(set(route["ranked_oracle"]) != set(keys) for route in routes):
        raise ValueError("route-ranked oracle keys changed across records")
    result: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        rows = [route["ranked_oracle"][key] for route in routes]
        effective_values = {int(row["effective_k"]) for row in rows}
        result[key] = {
            "recall_iou50": _mean(
                float(row["recall_iou50"]) for row in rows
            ),
            "mean_best_iou": _mean(
                row["best_iou"]
                for row in rows
                if row["best_iou"] is not None
            ),
            "effective_k_min": min(effective_values),
            "effective_k_max": max(effective_values),
        }
    return result


def aggregate_role_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate oracle recall and route localization without model state."""

    role_swap_supported = bool(records) and all(
        bool(record.get("true_role_swap", {}).get("supported"))
        for record in records
    )
    result: Dict[str, Any] = {
        "num_expressions": len(records),
        "candidate_oracle": {},
        "routes": {},
        "true_role_swap": dict(
            TRUE_ROLE_SWAP_STATUS if role_swap_supported else ROLE_SWAP_STATUS
        ),
    }
    if not records:
        return result
    oracle_keys = sorted(records[0]["candidate_oracle"], key=_oracle_key)
    for key in oracle_keys:
        rows = [record["candidate_oracle"][key] for record in records]
        result["candidate_oracle"][key] = {
            "recall_iou50": _mean(float(row["recall_iou50"]) for row in rows),
            "mean_best_iou": _mean(
                row["best_iou"] for row in rows if row["best_iou"] is not None
            ),
        }
    if role_swap_supported:
        result["true_role_swap_candidate_oracle"] = {}
        swap_oracle_keys = sorted(
            records[0]["true_role_swap_candidate_oracle"],
            key=_oracle_key,
        )
        for key in swap_oracle_keys:
            rows = [record["true_role_swap_candidate_oracle"][key] for record in records]
            result["true_role_swap_candidate_oracle"][key] = {
                "recall_iou50": _mean(
                    float(row["recall_iou50"]) for row in rows
                ),
                "mean_best_iou": _mean(
                    row["best_iou"] for row in rows if row["best_iou"] is not None
                ),
            }
    patch_queries = [record["routes"]["patch_only"]["query_index"] for record in records]
    route_names = [
        "patch_all_queries",
        "patch_only",
        "text_only",
        "patch_admission_text_rank",
    ]
    if role_swap_supported:
        route_names.append(TRUE_ROLE_SWAP_ROUTE)
    fused_queries = [
        record["routes"]["patch_admission_text_rank"]["query_index"]
        for record in records
    ]
    for route_name in route_names:
        routes = [record["routes"][route_name] for record in records]
        result["routes"][route_name] = {
            "acc50": _mean(
                float(route["selected_iou"] >= 0.5)
                for route in routes
                if route["selected_iou"] is not None
            ),
            "mean_selected_iou": _mean(
                route["selected_iou"]
                for route in routes
                if route["selected_iou"] is not None
            ),
            "mean_selected_logit": _mean(
                route["score_logit"]
                for route in routes
                if route["score_logit"] is not None
            ),
            "top1_query_churn_vs_patch_only": _mean(
                float(route["query_index"] != patch_queries[index])
                for index, route in enumerate(routes)
            ),
            "ranked_oracle": _aggregate_ranked_oracle(routes),
        }
        if route_name == TRUE_ROLE_SWAP_ROUTE:
            result["routes"][route_name][
                "top1_query_churn_vs_patch_admission_text_rank"
            ] = _mean(
                float(route["query_index"] != fused_queries[index])
                for index, route in enumerate(routes)
            )
    row_routes = {
        "G1": (
            "stage_a_all_queries",
            "patch_canonical",
            "patch_all_queries",
        ),
        "G2": ("patch_top50", "patch_canonical_only", "patch_only"),
        "G3": ("patch_top50", "full_text_only", "text_only"),
        "G4": (
            "patch_top50",
            "patch_admission_plus_fulltext_rank",
            "patch_admission_text_rank",
        ),
    }
    if role_swap_supported:
        row_routes["G5"] = (
            "canonical_text_allquery_top50",
            "patch_plus_fulltext_rank",
            TRUE_ROLE_SWAP_ROUTE,
        )
    result["table_a_rows"] = {}
    for row_id, (candidate_source, score_surface, route_name) in row_routes.items():
        route = result["routes"][route_name]
        result["table_a_rows"][row_id] = {
            "row_id": row_id,
            "candidate_source": candidate_source,
            "score_surface": score_surface,
            "candidate_count": (
                int(records[0]["all_query_count"])
                if row_id == "G1"
                else int(records[0]["candidate_count"])
            ),
            "num_expressions": len(records),
            "acc50": route["acc50"],
            "mean_selected_iou": route["mean_selected_iou"],
            "ranked_oracle": route["ranked_oracle"],
            "top1_query_churn_vs_patch_only": route[
                "top1_query_churn_vs_patch_only"
            ],
        }
        if row_id == "G5":
            result["table_a_rows"][row_id][
                "top1_query_churn_vs_patch_admission_text_rank"
            ] = route["top1_query_churn_vs_patch_admission_text_rank"]
    return result


def _raw_edit_categories(meta: Mapping[str, Any]) -> List[str]:
    edits = meta.get("tn_edits")
    values: List[Any] = []
    if isinstance(edits, list):
        for edit in edits:
            if isinstance(edit, Mapping):
                values.append(edit.get("category"))
    if not values:
        value = meta.get("replace_category")
        values = list(value) if isinstance(value, (list, tuple)) else [value]
    return [str(value).strip() for value in values if str(value or "").strip()]


def _contains_noun_category_edit(categories: Sequence[str]) -> bool:
    return any(normalize_edit_taxonomy([value]) == "noun_category" for value in categories)


def normalize_edit_taxonomy(categories: Sequence[str]) -> str:
    """Map free-form edit labels to the five paper groups or mixed/other."""

    groups = set()
    for category in categories:
        value = str(category).lower().replace("_", " ").replace("/", " ")
        words = set(value.split())
        if "relation" in words or "relational" in words:
            groups.add("relation")
        elif words.intersection(
            {
                "spatial",
                "position",
                "location",
                "direction",
                "orientation",
                "distance",
                "side",
                "placement",
                "proximity",
                "depth",
            }
        ):
            groups.add("spatial")
        if words.intersection({"color", "colour", "shade", "brightness", "tone"}):
            groups.add("color")
        if words.intersection(
            {
                "size",
                "height",
                "width",
                "length",
                "scale",
                "quantity",
                "count",
                "number",
                "dimension",
            }
        ):
            groups.add("size")
        if words.intersection(
            {
                "action",
                "activity",
                "motion",
                "behavior",
                "behaviour",
                "gesture",
                "posture",
                "pose",
                "sport",
            }
        ):
            groups.add("action")
        if words.intersection(
            {
                "noun",
                "category",
                "object",
                "type",
                "species",
                "animal",
                "vehicle",
                "food",
                "identity",
                "entity",
                "item",
                "subject",
                "person",
                "breed",
            }
        ):
            groups.add("noun_category")
    if not groups:
        return "other"
    if len(groups) > 1:
        return "mixed"
    return next(iter(groups))


def _rank_vector(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(
        1, int(scores.numel()) + 1, device=scores.device, dtype=order.dtype
    )
    return ranks


def _counterfactual_surface(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    query_ids: torch.Tensor,
) -> Dict[str, Any]:
    positive_top = int(torch.argmax(positive_scores).item())
    negative_top = int(torch.argmax(negative_scores).item())
    positive_ranks = _rank_vector(positive_scores)
    negative_ranks = _rank_vector(negative_scores)
    return {
        "positive_top1_query": int(query_ids[positive_top].item()),
        "negative_top1_query": int(query_ids[negative_top].item()),
        "top1_changed": positive_top != negative_top,
        "delta_max_logit_negative_minus_positive": _finite_float(
            negative_scores.max() - positive_scores.max()
        ),
        "delta_at_positive_top1_negative_minus_positive": _finite_float(
            negative_scores[positive_top] - positive_scores[positive_top]
        ),
        "delta_at_negative_top1_negative_minus_positive": _finite_float(
            negative_scores[negative_top] - positive_scores[negative_top]
        ),
        "positive_top1_rank_under_negative": int(negative_ranks[positive_top].item()),
        "negative_top1_rank_under_positive": int(positive_ranks[negative_top].item()),
        "mean_absolute_rank_change": _finite_float(
            (positive_ranks - negative_ranks).abs().float().mean()
        ),
    }


def counterfactual_role_records(
    positive: Mapping[str, torch.Tensor],
    negative: Mapping[str, torch.Tensor],
    metadata: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare aligned positive/TN score surfaces for controlled text edits."""

    pos_idx = positive["candidate_idx"]
    neg_idx = negative["candidate_idx"]
    if pos_idx.shape != neg_idx.shape:
        raise ValueError("positive and negative candidate shapes differ")
    if len(metadata) != int(pos_idx.shape[0]):
        raise ValueError("counterfactual metadata must align with the batch")
    pos_valid = positive["expression_valid"]
    neg_valid = negative["expression_valid"]
    if not torch.equal(pos_valid, neg_valid):
        raise ValueError("positive and negative expression-valid masks differ")

    records: List[Dict[str, Any]] = []
    for batch_index in range(int(pos_idx.shape[0])):
        exact_admission = torch.equal(pos_idx[batch_index], neg_idx[batch_index])
        same_set = set(pos_idx[batch_index].tolist()) == set(
            neg_idx[batch_index].tolist()
        )
        if not same_set:
            alignment = "candidate_set_changed"
        elif exact_admission:
            alignment = "exact"
        else:
            alignment = "same_set_reordered"

        # Reorder negative surfaces onto positive query order when possible.
        if same_set:
            neg_position = {
                int(query): index
                for index, query in enumerate(neg_idx[batch_index].tolist())
            }
            reorder = torch.as_tensor(
                [neg_position[int(query)] for query in pos_idx[batch_index].tolist()],
                device=neg_idx.device,
                dtype=torch.long,
            )
        else:
            reorder = None

        raw_categories = _raw_edit_categories(metadata[batch_index])
        contains_noun_category = _contains_noun_category_edit(raw_categories)
        for slot_index in range(int(pos_valid.shape[1])):
            if not bool(pos_valid[batch_index, slot_index].item()):
                continue
            base: Dict[str, Any] = {
                **dict(metadata[batch_index]),
                "expression_slot": slot_index,
                "edit_categories": raw_categories,
                "edit_taxonomy": normalize_edit_taxonomy(raw_categories),
                "contains_noun_category_edit": contains_noun_category,
                "candidate_alignment": alignment,
                "candidate_admission_changed": not exact_admission,
                "delta_direction": "negative_minus_positive",
                "canonical_prompt_held_fixed": True,
                "patch_category_causal_supported": (
                    False if contains_noun_category else None
                ),
                "patch_invariance_interpretable_for_noncategory_edit": (
                    not contains_noun_category
                ),
                "patch_category_causal_reason": (
                    "the Stage-A canonical prompt and support patch are held "
                    "fixed between positive and TN forwards; a noun edit in "
                    "the full-text caption does not intervene on the patch "
                    "category input"
                    if contains_noun_category
                    else None
                ),
                "true_role_swap": dict(ROLE_SWAP_STATUS),
            }
            if reorder is None:
                base["causal_comparison_supported"] = False
                base["unsupported_reason"] = (
                    "positive and negative forwards produced different candidate sets"
                )
                records.append(base)
                continue
            base["causal_comparison_supported"] = True
            surfaces: Dict[str, Dict[str, Any]] = {}
            for name, key in (
                ("patch", "patch_logits"),
                ("fulltext", "text_logits"),
                ("patch_admission_text_rank", "fused_logits"),
            ):
                pos_scores = positive[key][batch_index]
                neg_scores = negative[key][batch_index].index_select(0, reorder)
                if pos_scores.dim() == 2:
                    pos_scores = pos_scores[:, slot_index]
                    neg_scores = neg_scores[:, slot_index]
                surfaces[name] = _counterfactual_surface(
                    pos_scores, neg_scores, pos_idx[batch_index]
                )
            base["surfaces"] = surfaces
            records.append(base)
    return records


def merge_true_role_swap_counterfactual_records(
    patch_records: Sequence[Mapping[str, Any]],
    swap_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach G5 positive/TN sensitivity to the aligned causal records."""

    if len(patch_records) != len(swap_records):
        raise ValueError("G5 causal records must align one-to-one")
    merged: List[Dict[str, Any]] = []
    for patch_record, swap_record in zip(patch_records, swap_records):
        for key in ("image_id", "ann_id", "ref_id", "sent_id", "sample_id", "expression_slot"):
            if patch_record.get(key) != swap_record.get(key):
                raise ValueError(f"G5 causal record identity drifted at {key!r}")
        if not bool(swap_record.get("causal_comparison_supported")):
            raise ValueError(
                "G5 canonical admission must be identical for positive/TN forwards"
            )
        out = dict(patch_record)
        if bool(out.get("causal_comparison_supported")):
            surfaces = dict(out["surfaces"])
            surfaces[TRUE_ROLE_SWAP_ROUTE] = dict(
                swap_record["surfaces"]["patch_admission_text_rank"]
            )
            out["surfaces"] = surfaces
        out["true_role_swap"] = dict(TRUE_ROLE_SWAP_STATUS)
        out["true_role_swap_candidate_alignment"] = swap_record[
            "candidate_alignment"
        ]
        merged.append(out)
    return merged


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _aggregate_counterfactual_group(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    supported = [record for record in records if record.get("causal_comparison_supported")]
    contains_noun_category = any(
        bool(record.get("contains_noun_category_edit")) for record in records
    )
    out: Dict[str, Any] = {
        "num_pairs": len(records),
        "num_supported": len(supported),
        "canonical_prompt_held_fixed": True,
        "contains_noun_category_pairs": sum(
            bool(record.get("contains_noun_category_edit")) for record in records
        ),
        "patch_category_evidence_eligible": not contains_noun_category,
        "candidate_admission_change_rate": _mean(
            float(record["candidate_admission_changed"]) for record in records
        ),
        "surfaces": {},
    }
    surface_names = ["patch", "fulltext", "patch_admission_text_rank"]
    if supported and all(
        TRUE_ROLE_SWAP_ROUTE in record.get("surfaces", {}) for record in supported
    ):
        surface_names.append(TRUE_ROLE_SWAP_ROUTE)
    for surface in surface_names:
        rows = [record["surfaces"][surface] for record in supported]
        deltas = [
            float(row["delta_max_logit_negative_minus_positive"])
            for row in rows
            if row["delta_max_logit_negative_minus_positive"] is not None
        ]
        out["surfaces"][surface] = {
            "delta_max_logit_mean": _mean(deltas),
            "delta_max_logit_median": _median(deltas),
            "top1_change_rate": _mean(float(row["top1_changed"]) for row in rows),
            "mean_absolute_rank_change": _mean(
                row["mean_absolute_rank_change"]
                for row in rows
                if row["mean_absolute_rank_change"] is not None
            ),
            "invariant_rate_at_1e-8": _mean(
                float(abs(value) <= 1e-8) for value in deltas
            ),
            "category_causal_evidence_eligible": not (
                surface == "patch" and contains_noun_category
            ),
        }
    return out


def aggregate_counterfactual_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("edit_taxonomy", "other")), []).append(record)
    role_swap_supported = bool(records) and all(
        bool(record.get("true_role_swap", {}).get("supported"))
        for record in records
    )
    return {
        "overall": _aggregate_counterfactual_group(records),
        "by_edit_taxonomy": {
            key: _aggregate_counterfactual_group(value)
            for key, value in sorted(groups.items())
        },
        "causal_contract": {
            "canonical_prompt_held_fixed": True,
            "noun_category_fulltext_edits_test_patch_category_role": False,
            "category_evidence_requirement": (
                "use Stage-A patch AP/admission evidence or rerun with a "
                "changed canonical prompt and category-matched support patch"
            ),
        },
        "true_role_swap": dict(
            TRUE_ROLE_SWAP_STATUS if role_swap_supported else ROLE_SWAP_STATUS
        ),
    }


def _normalized_category_boxes(
    intervention: Mapping[str, Any], key: str, *, device: torch.device
) -> torch.Tensor:
    category = intervention.get(key)
    if not isinstance(category, Mapping):
        raise ValueError(f"category intervention is missing {key}")
    boxes = category.get("boxes_xyxy")
    width = int(intervention.get("image_width", 0))
    height = int(intervention.get("image_height", 0))
    if not isinstance(boxes, list) or not boxes or width <= 0 or height <= 0:
        raise ValueError("category intervention has invalid GT geometry")
    tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if tensor.dim() != 2 or int(tensor.shape[-1]) != 4:
        raise ValueError("category intervention boxes must have shape (G,4)")
    scale = torch.tensor([width, height, width, height], device=device)
    normalized = tensor / scale
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("category intervention boxes are non-finite")
    return normalized.clamp(0.0, 1.0)


def category_intervention_arm_records(
    outputs: Mapping[str, Any],
    metadata: Sequence[Mapping[str, Any]],
    *,
    oracle_topks: Iterable[int] = ORACLE_TOPKS,
) -> List[Dict[str, Any]]:
    """Measure patch-category admission for synchronized prompt/support swaps."""

    patch_logits = _require_tensor(outputs, "pred_logits_patch").float()
    if patch_logits.dim() == 3:
        if int(patch_logits.shape[-1]) != 1:
            raise ValueError("category intervention requires one support patch")
        patch_logits = patch_logits[..., 0]
    pred_boxes = _normalized_cxcywh_to_xyxy(
        _require_tensor(outputs, "pred_boxes"), name="category intervention boxes"
    )
    candidate_idx = _require_tensor(outputs, "stage_b_v11_candidate_idx").long()
    if tuple(patch_logits.shape) != tuple(pred_boxes.shape[:2]):
        raise ValueError("category patch logits do not align with all-query boxes")
    if len(metadata) != int(patch_logits.shape[0]):
        raise ValueError("category metadata does not align with model batch")
    expected_idx = torch.topk(
        patch_logits,
        min(int(candidate_idx.shape[1]), int(patch_logits.shape[1])),
        dim=1,
        largest=True,
        sorted=True,
    ).indices
    if not torch.equal(candidate_idx, expected_idx):
        raise RuntimeError("runtime patch candidates drifted from dense patch Top-K")
    topks = tuple(int(value) for value in oracle_topks)
    if any(value <= 0 for value in topks):
        raise ValueError("category oracle Top-K values must be positive")

    records: List[Dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        intervention = meta.get("category_intervention")
        if not isinstance(intervention, Mapping):
            raise ValueError("category intervention metadata is absent")
        if intervention.get("schema") != "stageb-table-a-category-intervention-pair-v1":
            raise ValueError("category intervention row schema mismatch")
        arm = str(intervention.get("arm"))
        if arm not in {"A", "B"}:
            raise ValueError("category intervention arm must be A/B")
        active_key = "class_a" if arm == "A" else "class_b"
        counterfactual_key = "class_b" if arm == "A" else "class_a"
        if int(intervention["active_class_id"]) != int(
            intervention[active_key]["id"]
        ):
            raise ValueError("active category ID is inconsistent with the arm")
        expected_prompt = f"{intervention[active_key]['name']} ."
        if str(intervention.get("canonical_prompt")) != expected_prompt:
            raise ValueError("canonical prompt is inconsistent with the active category")
        if str(intervention.get("active_support_sha256")) != str(
            intervention[active_key]["support_sha256"]
        ):
            raise ValueError("support patch is inconsistent with the active category")
        active_gt = _normalized_category_boxes(
            intervention, active_key, device=pred_boxes.device
        )
        counterfactual_gt = _normalized_category_boxes(
            intervention, counterfactual_key, device=pred_boxes.device
        )
        active_iou = box_ops.box_iou(pred_boxes[index], active_gt)[0].max(dim=1).values
        counterfactual_iou = box_ops.box_iou(
            pred_boxes[index], counterfactual_gt
        )[0].max(dim=1).values
        oracle: Dict[str, Any] = {}
        for topk in topks:
            effective = min(topk, int(patch_logits.shape[1]))
            indices = torch.topk(
                patch_logits[index], effective, largest=True, sorted=True
            ).indices
            active_best = active_iou[indices].max()
            counterfactual_best = counterfactual_iou[indices].max()
            oracle[str(topk)] = {
                "effective_k": effective,
                "active_best_iou": _finite_float(active_best),
                "active_recall_iou50": bool(active_best >= 0.5),
                "counterfactual_best_iou": _finite_float(counterfactual_best),
                "counterfactual_recall_iou50": bool(counterfactual_best >= 0.5),
            }
        active_best = active_iou.max()
        counterfactual_best = counterfactual_iou.max()
        oracle["all"] = {
            "effective_k": int(patch_logits.shape[1]),
            "active_best_iou": _finite_float(active_best),
            "active_recall_iou50": bool(active_best >= 0.5),
            "counterfactual_best_iou": _finite_float(counterfactual_best),
            "counterfactual_recall_iou50": bool(counterfactual_best >= 0.5),
        }
        top1 = int(torch.argmax(patch_logits[index]).item())
        records.append(
            {
                **dict(meta),
                "task": "category_intervention",
                "pair_id": intervention["pair_id"],
                "arm": arm,
                "active_class_id": int(intervention["active_class_id"]),
                "active_class_name": intervention["active_class_name"],
                "counterfactual_class_id": int(
                    intervention["counterfactual_class_id"]
                ),
                "counterfactual_class_name": intervention[
                    "counterfactual_class_name"
                ],
                "canonical_prompt": intervention["canonical_prompt"],
                "active_support_path": intervention["active_support_path"],
                "active_support_sha256": intervention["active_support_sha256"],
                "prompt_and_support_changed_together": True,
                "category_causal_route": (
                    "joint_canonical_prompt_plus_support_patch"
                ),
                "patch_only_category_causal_claim_eligible": False,
                "all_query_count": int(patch_logits.shape[1]),
                "candidate_count": int(candidate_idx.shape[1]),
                "candidate_admission": oracle,
                "patch_top1": {
                    "query_index": top1,
                    "patch_logit": _finite_float(patch_logits[index, top1]),
                    "box_xyxy_normalized": [
                        float(value) for value in pred_boxes[index, top1].tolist()
                    ],
                    "active_iou": _finite_float(active_iou[top1]),
                    "counterfactual_iou": _finite_float(counterfactual_iou[top1]),
                },
                "category_causal_evidence_eligible": True,
            }
        )
    return records


def _box_iou_list(a: Sequence[float], b: Sequence[float]) -> float:
    tensor_a = torch.as_tensor(a, dtype=torch.float32).view(1, 4)
    tensor_b = torch.as_tensor(b, dtype=torch.float32).view(1, 4)
    return float(box_ops.box_iou(tensor_a, tensor_b)[0].item())


def aggregate_category_intervention_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("pair_id")), []).append(record)
    pair_rows: List[Dict[str, Any]] = []
    for pair_id, rows in groups.items():
        by_arm = {str(row.get("arm")): row for row in rows}
        if len(rows) != 2 or set(by_arm) != {"A", "B"}:
            raise ValueError(f"category pair {pair_id!r} is not exactly A/B")
        a, b = by_arm["A"], by_arm["B"]
        if int(a["image_id"]) != int(b["image_id"]):
            raise ValueError(f"category pair {pair_id!r} changed the image")
        if int(a["active_class_id"]) != int(b["counterfactual_class_id"]):
            raise ValueError(f"category pair {pair_id!r} has inconsistent classes")
        if int(b["active_class_id"]) != int(a["counterfactual_class_id"]):
            raise ValueError(f"category pair {pair_id!r} has inconsistent classes")
        if str(a["active_support_sha256"]) == str(b["active_support_sha256"]):
            raise ValueError(f"category pair {pair_id!r} did not change support")
        pair_rows.append(
            {
                "pair_id": pair_id,
                "top1_both_match_active": bool(
                    float(a["patch_top1"]["active_iou"]) >= 0.5
                    and float(b["patch_top1"]["active_iou"]) >= 0.5
                ),
                "top1_query_changed": int(a["patch_top1"]["query_index"])
                != int(b["patch_top1"]["query_index"]),
                "top1_box_iou_between_arms": _box_iou_list(
                    a["patch_top1"]["box_xyxy_normalized"],
                    b["patch_top1"]["box_xyxy_normalized"],
                ),
            }
        )
    summary: Dict[str, Any] = {
        "num_pairs": len(groups),
        "num_arms": len(records),
        "category_causal_evidence_eligible": bool(records),
        "category_causal_route": "joint_canonical_prompt_plus_support_patch",
        "patch_only_category_causal_claim_eligible": False,
        "intervention_contract": {
            "same_image": True,
            "canonical_prompt_changed": True,
            "support_patch_changed": True,
            "both_categories_have_ground_truth": True,
            "model_weights_frozen": True,
            "causal_factor": "canonical_prompt_and_support_patch_jointly_changed",
            "patch_only_attribution_forbidden": True,
        },
        "candidate_admission": {},
        "top1_both_match_active_rate": _mean(
            float(row["top1_both_match_active"]) for row in pair_rows
        ),
        "top1_query_change_rate": _mean(
            float(row["top1_query_changed"]) for row in pair_rows
        ),
        "mean_top1_box_iou_between_arms": _mean(
            row["top1_box_iou_between_arms"] for row in pair_rows
        ),
    }
    if records:
        for key in sorted(
            records[0]["candidate_admission"], key=_oracle_key
        ):
            rows = [record["candidate_admission"][key] for record in records]
            summary["candidate_admission"][key] = {
                "matched_active_recall_iou50": _mean(
                    float(row["active_recall_iou50"]) for row in rows
                ),
                "counterfactual_category_recall_iou50": _mean(
                    float(row["counterfactual_recall_iou50"]) for row in rows
                ),
                "matched_active_mean_best_iou": _mean(
                    row["active_best_iou"] for row in rows
                ),
                "counterfactual_mean_best_iou": _mean(
                    row["counterfactual_best_iou"] for row in rows
                ),
            }
    return summary


def verify_category_runtime_assets(
    rows: Sequence[Mapping[str, Any]], support_tsv: Path
) -> Dict[str, Any]:
    """Rehash the actual images/supports consumed by the causal intervention."""

    support_tsv = support_tsv.expanduser().resolve(strict=True)
    with support_tsv.open("r", encoding="utf-8", newline="") as handle:
        support_rows = list(csv.DictReader(handle, delimiter="\t"))
    tsv_paths = {
        str(Path(str(row.get("path", ""))).expanduser().resolve(strict=True))
        for row in support_rows
    }
    expected_supports: Dict[str, str] = {}
    expected_images: Dict[str, str] = {}
    for row in rows:
        intervention = row.get("category_intervention")
        if not isinstance(intervention, Mapping):
            raise ValueError("category intervention metadata is missing")
        image_path = str(
            Path(str(intervention["image_path"])).expanduser().resolve(strict=True)
        )
        image_sha = str(intervention["image_sha256"])
        previous_image = expected_images.setdefault(image_path, image_sha)
        if previous_image != image_sha:
            raise ValueError("category image path is bound to multiple SHA-256 values")
        for key in ("class_a", "class_b"):
            category = intervention.get(key)
            if not isinstance(category, Mapping):
                raise ValueError(f"category intervention is missing {key}")
            support_path = str(
                Path(str(category["support_path"])).expanduser().resolve(strict=True)
            )
            support_sha = str(category["support_sha256"])
            previous_support = expected_supports.setdefault(support_path, support_sha)
            if previous_support != support_sha:
                raise ValueError(
                    "category support path is bound to multiple SHA-256 values"
                )
    if set(expected_supports) != tsv_paths:
        missing = sorted(set(expected_supports) - tsv_paths)
        extra = sorted(tsv_paths - set(expected_supports))
        raise ValueError(
            "category support TSV/path binding mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    for label, bindings in (
        ("image", expected_images),
        ("support", expected_supports),
    ):
        for path, expected_sha in bindings.items():
            observed_sha = _file_sha256(Path(path))
            if observed_sha != expected_sha:
                raise ValueError(
                    f"category {label} asset SHA-256 mismatch: {path}"
                )
    digest = hashlib.sha256()
    for kind, bindings in (("image", expected_images), ("support", expected_supports)):
        for path, sha256 in sorted(bindings.items()):
            digest.update(f"{kind}\0{path}\0{sha256}\n".encode("utf-8"))
    return {
        "status": "passed",
        "algorithm": "sorted-kind-path-sha256-v1",
        "images_rehashed": len(expected_images),
        "supports_rehashed": len(expected_supports),
        "binding_sha256": digest.hexdigest(),
        "support_tsv_sha256": _file_sha256(support_tsv),
    }


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def _identity_from_row(row: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    return tuple(
        int(row.get(key, -1)) for key in ("image_id", "ann_id", "ref_id", "sent_id")
    )  # type: ignore[return-value]


def _identity_from_target(target: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    values = []
    for key in ("image_id", "ann_id", "ref_id", "sent_id"):
        value = target.get(key)
        if torch.is_tensor(value) and value.numel():
            values.append(int(value.reshape(-1)[0].item()))
        else:
            values.append(-1)
    return tuple(values)  # type: ignore[return-value]


def _metadata_index(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[int, int, int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for row in rows:
        key = _identity_from_row(row)
        if key in out:
            raise ValueError(f"duplicate expression identity in diagnostic input: {key}")
        instance = (row.get("instances") or [{}])[0]
        if not isinstance(instance, Mapping):
            instance = {}
        out[key] = {
            "image_id": key[0],
            "ann_id": key[1],
            "ref_id": key[2],
            "sent_id": key[3],
            "sample_id": row.get("sample_id"),
            "positive_phrase": row.get("sent") or instance.get("positive_phrase"),
            "negative_phrase": row.get("try_tn") or instance.get("raw_phrase"),
            "replace_category": row.get("replace_category") or instance.get("replace_category"),
            "tn_edits": row.get("tn_edits"),
            "category_intervention": row.get("category_intervention"),
        }
    return out


def _batch_metadata(
    targets: Sequence[Mapping[str, Any]],
    metadata: Mapping[Tuple[int, int, int, int], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for target in targets:
        key = _identity_from_target(target)
        if key not in metadata:
            raise KeyError(f"batch target identity is absent from input metadata: {key}")
        rows.append(dict(metadata[key]))
    return rows


def _negative_only_pair_batch(
    batch,
    metadata: Sequence[Mapping[str, Any]],
):
    """Replace a training-style paired caption with the TN expression only.

    SAM3 pair datasets expose a combined ``positive . negative .`` training
    caption.  The shared paired evaluator expects its first forward caption to
    be the negative expression alone, with the exact positive carried in
    ``rank_positive_captions``.  This adapter changes only the batch-local
    target dictionaries; source records remain immutable.
    """

    targets = list(batch[1])
    if len(targets) != len(metadata):
        raise ValueError("TN pair metadata must align with collated targets")
    rewritten = []
    for target, meta in zip(targets, metadata):
        negative = meta.get("negative_phrase")
        if not isinstance(negative, str) or not negative.strip():
            raise ValueError("TN causal diagnostic requires a negative expression")
        caption = " ".join(negative.replace(".", " ").split())
        if not caption:
            raise ValueError("TN causal diagnostic normalized to an empty caption")
        copied = dict(target)
        copied["caption"] = f"{caption} ."
        copied["verifier_caption"] = copied["caption"]
        # These masks were tokenized against the combined training caption.
        # v19 retokenizes its verifier captions, so stale masks must not be
        # forwarded under the new caption.
        copied.pop("phrase_to_token_mask", None)
        copied.pop("canonical_to_token_mask", None)
        rewritten.append(copied)
    return batch[0], rewritten


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_eval(cfg, args: argparse.Namespace) -> None:
    cfg.device = str(args.device)
    cfg.patch_only = True
    cfg.build_text_token_masks = True
    cfg.use_coco_eval = False
    cfg.use_checkpoint = False
    cfg.use_transformer_ckpt = False
    cfg.batch_size = int(args.batch_size)
    cfg.text_mask_warn_limit = 0
    if not bool(getattr(cfg, "stage_b_v11_fixed_text", False)):
        raise ValueError("C1 role diagnostic requires stage_b_v11_fixed_text=true")
    if not bool(getattr(cfg, "stage_b_v15_patch_rank_fusion", False)):
        raise ValueError("C1 role diagnostic requires explicit patch-rank fusion")
    if not bool(
        getattr(cfg, "stage_b_v19_explicit_confidence_output_contract", False)
    ):
        raise ValueError("C1 role diagnostic is restricted to the v19 contract")


def _official_ref_inputs(
    *,
    requested: Sequence[str],
    data_root: Path,
    output_dir: Path,
) -> List[Tuple[str, Path]]:
    if not requested:
        return []
    specs = _default_splits()
    by_name = {str(spec["name"]): spec for spec in specs}
    names = list(by_name) if list(requested) == ["all"] else list(requested)
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise KeyError(f"unknown Ref split(s): {unknown}")
    name_to_id, id_to_name = _load_canonical_name_maps(
        data_root / "canonical_classes_with_aliases.json"
    )
    phrase_maps = _load_phrase_maps(
        [
            data_root / "refcoco_text_pairs" / "refcoco_unc_pairs.jsonl",
            data_root / "data_proc" / "refcoco_text_pairs" / "refcoco+_unc_pairs.jsonl",
            data_root / "data_proc" / "refcoco_text_pairs" / "refcocog_google_pairs.jsonl",
            data_root / "patch_episode_prebuilt" / "refcocoplus_stageb_phrase_v1.jsonl",
            data_root / "patch_episode_prebuilt" / "refcocog_stageb_phrase_v1.jsonl",
        ]
    )
    result = []
    for name in names:
        spec = by_name[name]
        path, _count = _build_split_jsonl(
            data_root=data_root,
            output_dir=output_dir,
            dataset=spec["dataset"],
            splitby=spec["splitby"],
            split=spec["split"],
            phrase_sources=list(spec["sources"]),
            phrase_maps=phrase_maps,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
        )
        result.append((name, path))
    return result


def _run_ref_forward_with_role_swap(
    *,
    model,
    cfg,
    batch,
    device: torch.device,
    args: argparse.Namespace,
    equality_receipt: Optional[Dict[str, int]] = None,
):
    """Return ordinary patch admission and true G5 outputs for one Ref batch."""

    raw_targets = list(batch[1])
    (
        samples,
        targets,
        _captions,
        patches,
        patch_global,
        patch_mask,
    ) = _prepare_ref_patch_batch(*batch, device)
    stage_a_captions = _ref_target_texts(raw_targets, "stage_a_caption")
    verifier_captions = _ref_target_texts(raw_targets, "verifier_caption")
    kmax = int(patch_mask.shape[1]) if patch_mask is not None else 1
    phrase_mask = _pad_ref_target_mask(
        raw_targets, "phrase_to_token_mask", kmax, device
    )
    canonical_mask = _pad_ref_target_mask(
        raw_targets, "canonical_to_token_mask", kmax, device
    )
    call_kwargs = {
        "samples": samples,
        "targets": targets,
        "stage_a_captions": stage_a_captions,
        "verifier_captions": verifier_captions,
        "patches": patches,
        "patch_global": patch_global,
        "patch_mask": patch_mask,
        "phrase_mask": phrase_mask,
        "canonical_mask": canonical_mask,
        "amp": bool(args.amp),
        "device": device,
        "compute_canonical_text": True,
    }
    patch_outputs = _call_fixed_text_model(model, **call_kwargs)
    patch_indices = _require_tensor(
        patch_outputs, "stage_b_v11_candidate_idx"
    )
    text_indices = _text_topk_candidate_indices(
        patch_outputs, candidate_count=int(patch_indices.shape[1])
    )
    swap_outputs = _forced_candidate_forward(
        model,
        expected_outputs=patch_outputs,
        forced_indices=text_indices,
        call_kwargs=call_kwargs,
        equality_receipt=equality_receipt,
    )
    return patch_outputs, swap_outputs, targets


def _run_tn_forward_with_role_swap(
    *,
    model,
    batch,
    device: torch.device,
    args: argparse.Namespace,
    equality_receipt: Optional[Dict[str, int]] = None,
):
    """Return patch and G5 positive/TN forwards under one canonical admission."""

    raw_targets = list(batch[1])
    (
        samples,
        targets,
        _neg_caption_from_batch,
        patches,
        patch_global,
        patch_mask,
    ) = _prepare_tn_patch_batch(*batch, device)
    pos_captions, valid_pos = _rank_positive_captions(raw_targets)
    neg_captions = _tn_target_texts(raw_targets, "caption")
    stage_a_captions = _tn_target_texts(raw_targets, "stage_a_caption")
    kmax = int(patch_mask.shape[1]) if patch_mask is not None else 1
    neg_phrase_mask = _pad_tn_target_mask(
        raw_targets, "phrase_to_token_mask", kmax, 256, device
    )
    neg_canonical_mask = _pad_tn_target_mask(
        raw_targets, "canonical_to_token_mask", kmax, 256, device
    )
    pos_phrase_mask = _pad_tn_target_mask(
        raw_targets, "rank_positive_phrase_to_token_mask", kmax, 256, device
    )
    pos_canonical_mask = _pad_tn_target_mask(
        raw_targets, "rank_positive_canonical_to_token_mask", kmax, 256, device
    )

    common = {
        "samples": samples,
        "targets": targets,
        "stage_a_captions": stage_a_captions,
        "patches": patches,
        "patch_global": patch_global,
        "patch_mask": patch_mask,
        "amp": bool(args.amp),
        "device": device,
        "compute_canonical_text": True,
    }
    neg_kwargs = {
        **common,
        "verifier_captions": neg_captions,
        "phrase_mask": neg_phrase_mask,
        "canonical_mask": neg_canonical_mask,
    }
    pos_kwargs = {
        **common,
        "verifier_captions": pos_captions,
        "phrase_mask": pos_phrase_mask,
        "canonical_mask": pos_canonical_mask,
    }
    neg_outputs = _call_fixed_text_model(model, **neg_kwargs)
    pos_outputs = _call_fixed_text_model(model, **pos_kwargs)
    for key in ("hs", "pred_boxes", "pred_logits_patch"):
        if not torch.equal(_require_tensor(neg_outputs, key), _require_tensor(pos_outputs, key)):
            raise RuntimeError(
                f"positive/TN canonical forwards changed frozen all-query tensor {key!r}"
            )
    patch_indices = _require_tensor(neg_outputs, "stage_b_v11_candidate_idx")
    text_indices = _text_topk_candidate_indices(
        neg_outputs, candidate_count=int(patch_indices.shape[1])
    )
    pos_text_indices = _text_topk_candidate_indices(
        pos_outputs, candidate_count=int(patch_indices.shape[1])
    )
    if not torch.equal(text_indices, pos_text_indices):
        raise RuntimeError("positive/TN canonical text admission indices differ")
    neg_swap = _forced_candidate_forward(
        model,
        expected_outputs=neg_outputs,
        forced_indices=text_indices,
        call_kwargs=neg_kwargs,
        equality_receipt=equality_receipt,
    )
    pos_swap = _forced_candidate_forward(
        model,
        expected_outputs=pos_outputs,
        forced_indices=text_indices,
        call_kwargs=pos_kwargs,
        equality_receipt=equality_receipt,
    )
    return (
        neg_outputs,
        pos_outputs,
        neg_swap,
        pos_swap,
        targets,
        valid_pos.to(device=device),
    )


def _run_ref_input(
    *,
    model,
    cfg,
    path: Path,
    name: str,
    data_root: Path,
    device: torch.device,
    args: argparse.Namespace,
    equality_receipt: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    source_rows = list(_iter_jsonl(path))
    metadata_index = _metadata_index(source_rows)
    datasetinfo = _make_ref_datasetinfo(data_root, name, path)
    loader = _build_ref_loader(
        cfg, datasetinfo, args.batch_size, args.num_workers, device, args.seed
    )
    records: List[Dict[str, Any]] = []
    weight = float(getattr(cfg, "stage_b_v15_patch_rank_weight", 0.0))
    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        swap_outputs = None
        if bool(args.true_role_swap):
            outputs, swap_outputs, targets = _run_ref_forward_with_role_swap(
                model=model,
                cfg=cfg,
                batch=batch,
                device=device,
                args=args,
                equality_receipt=equality_receipt,
            )
        else:
            outputs, targets = _forward_ref(
                model, batch, device, amp=args.amp, cfg=cfg
            )
        metadata = _batch_metadata(targets, metadata_index)
        for row in metadata:
            row.update({"task": "ref", "dataset": name})
        components = extract_role_components(
            outputs,
            targets,
            patch_rank_weight=weight,
            candidate_source="patch_topk",
        )
        batch_records = route_role_records(components, metadata=metadata)
        if swap_outputs is not None:
            swap_components = extract_role_components(
                swap_outputs,
                targets,
                patch_rank_weight=weight,
                candidate_source="canonical_text_topk",
            )
            swap_records = route_role_records(
                swap_components, metadata=metadata
            )
            batch_records = merge_true_role_swap_records(
                batch_records, swap_records
            )
        records.extend(batch_records)
    return records


def _run_category_input(
    *,
    model,
    cfg,
    path: Path,
    support_tsv: Path,
    audit_path: Path,
    data_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from tools.build_stageb_table_a_category_interventions import (
        verify as verify_category_interventions,
    )

    verify_category_interventions(
        output=path,
        output_support_tsv=support_tsv,
        audit_path=audit_path,
        require_canonical=bool(args.formal_table_a),
    )
    source_rows = list(_iter_jsonl(path))
    asset_rehash = verify_category_runtime_assets(source_rows, support_tsv)
    metadata_index = _metadata_index(source_rows)
    datasetinfo = _make_ref_datasetinfo(data_root, "table_a_category", path)
    datasetinfo.update(
        {
            "support_min_count": 1,
            "support_patch_tsv": str(support_tsv),
            "support_patch_bucket": "clean",
            "support_patch_use_embedding": False,
            "support_patch_max_per_class": 1,
            "support_patch_image_root": str(data_root / "patches_quality"),
            "keep_only_support_gt": True,
        }
    )
    loader = _build_ref_loader(
        cfg, datasetinfo, args.batch_size, args.num_workers, device, args.seed
    )
    records: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        outputs, targets = _forward_ref(
            model, batch, device, amp=args.amp, cfg=cfg
        )
        metadata = _batch_metadata(targets, metadata_index)
        captions = outputs.get("stage_a_captions")
        if not isinstance(captions, list) or len(captions) != len(metadata):
            raise RuntimeError("category intervention did not emit Stage-A captions")
        for index, (target, meta) in enumerate(zip(targets, metadata)):
            intervention = meta.get("category_intervention")
            if not isinstance(intervention, Mapping):
                raise ValueError("category intervention metadata is missing")
            if str(captions[index]) != str(intervention["canonical_prompt"]):
                raise RuntimeError(
                    "runtime canonical prompt differs from the intervention binding"
                )
            support_class = target.get("support_class")
            if not torch.is_tensor(support_class) or support_class.numel() != 1:
                raise RuntimeError("category intervention requires one support class")
            if int(support_class.item()) != int(intervention["active_class_id"]):
                raise RuntimeError(
                    "runtime support patch class differs from the active category"
                )
        records.extend(category_intervention_arm_records(outputs, metadata))
    return records, asset_rehash


def _run_tn_input(
    *,
    model,
    cfg,
    path: Path,
    data_root: Path,
    device: torch.device,
    args: argparse.Namespace,
    equality_receipt: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_rows = list(_iter_jsonl(path))
    metadata_index = _metadata_index(source_rows)
    datasetinfo = _make_tn_datasetinfo(data_root, path)
    if source_rows and not isinstance(source_rows[0].get("instances"), list):
        # The shared TN evaluator normally consumes an already-derived
        # patch-episode manifest.  Traceable semantic pair files retain the
        # original SAM3-style schema, so supply the same explicit replay fields
        # used by their training manifests before calling the shared loader.
        coco_train_root = data_root / "COCO" / "coco2014" / "train2014"
        datasetinfo.update(
            {
                "source": "sam3_tn_pair",
                "root": str(coco_train_root),
                "sam3_tn_image_root": str(coco_train_root),
                "sam3_tn_bbox_key": "target_bbox_used",
                "support_min_count": 1,
            }
        )
    loader = _build_tn_loader(
        cfg, datasetinfo, args.batch_size, args.num_workers, device, args.seed
    )
    positive_route_records: List[Dict[str, Any]] = []
    causal_records: List[Dict[str, Any]] = []
    weight = float(getattr(cfg, "stage_b_v15_patch_rank_weight", 0.0))
    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        metadata = _batch_metadata(list(batch[1]), metadata_index)
        pair_batch = _negative_only_pair_batch(batch, metadata)
        neg_swap_outputs = None
        pos_swap_outputs = None
        if bool(args.true_role_swap):
            (
                neg_outputs,
                pos_outputs,
                neg_swap_outputs,
                pos_swap_outputs,
                targets,
                valid_pos,
            ) = _run_tn_forward_with_role_swap(
                model=model,
                batch=pair_batch,
                device=device,
                args=args,
                equality_receipt=equality_receipt,
            )
        else:
            neg_outputs, pos_outputs, targets, valid_pos = _forward_pair(
                model, pair_batch, device, amp=args.amp
            )
        if not bool(valid_pos.all().item()):
            raise ValueError("TN diagnostic requires an exact positive caption per row")
        for row in metadata:
            row.update({"task": "tn_positive", "dataset": path.stem})
        positive = extract_role_components(
            pos_outputs,
            targets,
            patch_rank_weight=weight,
            candidate_source="patch_topk",
        )
        negative = extract_role_components(
            neg_outputs,
            targets,
            patch_rank_weight=weight,
            candidate_source="patch_topk",
        )
        positive_batch_records = route_role_records(positive, metadata=metadata)
        causal_batch_records = counterfactual_role_records(
            positive, negative, metadata
        )
        if neg_swap_outputs is not None and pos_swap_outputs is not None:
            positive_swap = extract_role_components(
                pos_swap_outputs,
                targets,
                patch_rank_weight=weight,
                candidate_source="canonical_text_topk",
            )
            negative_swap = extract_role_components(
                neg_swap_outputs,
                targets,
                patch_rank_weight=weight,
                candidate_source="canonical_text_topk",
            )
            positive_batch_records = merge_true_role_swap_records(
                positive_batch_records,
                route_role_records(positive_swap, metadata=metadata),
            )
            causal_batch_records = merge_true_role_swap_counterfactual_records(
                causal_batch_records,
                counterfactual_role_records(
                    positive_swap, negative_swap, metadata
                ),
            )
        positive_route_records.extend(positive_batch_records)
        causal_records.extend(causal_batch_records)
    return positive_route_records, causal_records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate v19 patch/text roles and TN edit causality."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--data_root", default=os.environ.get("DATA_ROOT", "/media/haoyi/T9/data")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--formal_table_a",
        action="store_true",
        help="Emit the fail-closed formal Table-A evidence contract.",
    )
    parser.add_argument(
        "--evaluation_profile",
        choices=("diagnostic", *FORMAL_PROFILES),
        default="diagnostic",
    )
    parser.add_argument(
        "--true_role_swap",
        action="store_true",
        help=(
            "Run G5 with canonical-text all-query Top-K admission and the "
            "existing patch/full-expression rank scorer."
        ),
    )
    parser.add_argument(
        "--ref_splits",
        nargs="*",
        default=[],
        help="Official Ref split names or 'all'.",
    )
    parser.add_argument(
        "--ref_jsonl",
        action="append",
        default=[],
        help="Additional prebuilt patch-episode Ref JSONL (repeatable).",
    )
    parser.add_argument(
        "--tn_jsonl", default=None, help="Traceable positive/TN pair JSONL."
    )
    parser.add_argument("--screen_calibration_manifest", action="store_true")
    parser.add_argument("--screen_calibration_audit", default=None)
    parser.add_argument(
        "--category_jsonl",
        default=None,
        help="Paired synchronized category-intervention episode JSONL.",
    )
    parser.add_argument(
        "--category_support_tsv",
        default=None,
        help="Single-support-per-category TSV bound by the category audit.",
    )
    parser.add_argument(
        "--category_audit",
        default=None,
        help="Category-intervention build audit binding JSONL and support TSV.",
    )
    parser.add_argument(
        "--per_example_jsonl",
        default=None,
        help="Defaults to OUTPUT_DIR/role_causal.records.jsonl.",
    )
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Defaults to OUTPUT_DIR/role_causal.summary.json.",
    )
    args = parser.parse_args()

    if bool(args.formal_table_a):
        errors = []
        expected_ref = (
            VALIDATION_REF_SPLITS
            if args.evaluation_profile == "validation"
            else tuple(REF_SPLITS)
        )
        if args.evaluation_profile not in FORMAL_PROFILES:
            errors.append("a formal evaluation profile is required")
        if tuple(args.ref_splits) != tuple(expected_ref):
            errors.append("the profile-specific Ref split list is not exact")
        if not args.true_role_swap:
            errors.append("--true_role_swap is required")
        if int(args.seed) != 42 or int(args.max_batches) != 0:
            errors.append("seed=42 and full batches are required")
        if args.ref_jsonl or args.tn_jsonl is None:
            errors.append("custom Ref JSONL is forbidden and TN input is required")
        if not all(
            value is not None
            for value in (
                args.category_jsonl,
                args.category_support_tsv,
                args.category_audit,
            )
        ):
            errors.append("the full category intervention is required")
        if args.per_example_jsonl is not None or args.summary_json is not None:
            errors.append("custom formal output paths are forbidden")
        if args.evaluation_profile == "validation":
            if not args.screen_calibration_manifest or args.screen_calibration_audit is None:
                errors.append("validation requires the sealed screen calibration binding")
        elif args.screen_calibration_manifest or args.screen_calibration_audit is not None:
            errors.append("final evaluation cannot use the validation calibration adapter")
        if errors:
            parser.error("invalid formal Table-A contract: " + "; ".join(errors))
    elif args.evaluation_profile != "diagnostic":
        parser.error("non-diagnostic profile requires --formal_table_a")
    elif args.screen_calibration_manifest or args.screen_calibration_audit is not None:
        parser.error("screen calibration mode is restricted to formal validation")

    if (
        not args.ref_splits
        and not args.ref_jsonl
        and args.tn_jsonl is None
        and args.category_jsonl is None
    ):
        parser.error(
            "provide --ref_splits, --ref_jsonl, --tn_jsonl, and/or --category_jsonl"
        )
    category_values = (
        args.category_jsonl,
        args.category_support_tsv,
        args.category_audit,
    )
    if any(value is not None for value in category_values) and not all(
        value is not None for value in category_values
    ):
        parser.error(
            "--category_jsonl, --category_support_tsv, and --category_audit "
            "must be provided together"
        )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if bool(args.formal_table_a) and output_dir.exists():
        raise FileExistsError(f"formal Table-A output must be fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()
    cfg = SLConfig.fromfile(args.config)
    _configure_eval(cfg, args)
    model = _load_model(cfg, args.checkpoint, device)

    started = time.time()
    equality_receipt = _new_g5_equality_receipt()
    ref_inputs = _official_ref_inputs(
        requested=args.ref_splits,
        data_root=data_root,
        output_dir=output_dir,
    )
    ref_inputs.extend(
        (Path(value).stem, Path(value).expanduser().resolve())
        for value in args.ref_jsonl
    )
    all_records: List[Dict[str, Any]] = []
    ref_summaries: Dict[str, Any] = {}
    for name, path in ref_inputs:
        records = _run_ref_input(
            model=model,
            cfg=cfg,
            path=path,
            name=name,
            data_root=data_root,
            device=device,
            args=args,
            equality_receipt=equality_receipt,
        )
        ref_summaries[name] = aggregate_role_records(records)
        all_records.extend(records)

    causal_records: List[Dict[str, Any]] = []
    tn_role_summary = None
    screen_calibration_binding = None
    if args.tn_jsonl is not None:
        tn_path = Path(args.tn_jsonl).expanduser().resolve()
        tn_runtime_path = tn_path
        if args.screen_calibration_manifest:
            from tools.stageb_screen_calibration import build_manifest

            tn_runtime_path = output_dir / "tn_eval_inputs/tn_screen_calibration.jsonl"
            screen_calibration_binding = build_manifest(
                source_path=tn_path,
                audit_path=Path(args.screen_calibration_audit),
                derived_path=tn_runtime_path,
                data_root=data_root,
            )
        tn_routes, causal_records = _run_tn_input(
            model=model,
            cfg=cfg,
            path=tn_runtime_path,
            data_root=data_root,
            device=device,
            args=args,
            equality_receipt=equality_receipt,
        )
        tn_role_summary = aggregate_role_records(tn_routes)
        all_records.extend(tn_routes)
        all_records.extend(
            {**row, "task": "tn_counterfactual"} for row in causal_records
        )

    category_records: List[Dict[str, Any]] = []
    category_summary = None
    category_asset_rehash = None
    if args.category_jsonl is not None:
        category_records, category_asset_rehash = _run_category_input(
            model=model,
            cfg=cfg,
            path=Path(args.category_jsonl).expanduser().resolve(),
            support_tsv=Path(args.category_support_tsv).expanduser().resolve(),
            audit_path=Path(args.category_audit).expanduser().resolve(),
            data_root=data_root,
            device=device,
            args=args,
        )
        category_summary = aggregate_category_intervention_records(
            category_records
        )
        category_summary["asset_rehash"] = category_asset_rehash
        all_records.extend(category_records)

    input_manifests: Dict[str, Any] = {
        f"ref:{name}": {
            "path": str(path),
            "sha256": _file_sha256(path),
            "rows": int(ref_summaries[name]["num_expressions"]),
        }
        for name, path in ref_inputs
    }
    if args.tn_jsonl is not None:
        input_manifests["tn"] = {
            "path": str(tn_path),
            "sha256": _file_sha256(tn_path),
            "rows": len(causal_records),
        }
        if screen_calibration_binding is not None:
            from tools.stageb_screen_calibration import (
                binding_path as screen_binding_path,
                summary_fields as screen_summary_fields,
            )

            derived_path = Path(
                str(screen_calibration_binding.derived_manifest["path"])
            ).resolve(strict=True)
            binding = screen_binding_path(derived_path).resolve(strict=True)
            input_manifests["tn_runtime_derived"] = {
                "path": str(derived_path),
                "sha256": _file_sha256(derived_path),
                "rows": len(causal_records),
            }
            input_manifests["tn_screen_binding"] = {
                "path": str(binding),
                "sha256": _file_sha256(binding),
            }
            screen_calibration_summary = screen_summary_fields(
                screen_calibration_binding
            )
        else:
            screen_calibration_summary = None
    else:
        screen_calibration_summary = None
    if args.category_jsonl is not None:
        for label, value in (
            ("category", args.category_jsonl),
            ("category_support", args.category_support_tsv),
            ("category_audit", args.category_audit),
        ):
            input_path = Path(value).expanduser().resolve()
            input_manifests[label] = {
                "path": str(input_path),
                "sha256": _file_sha256(input_path),
            }

    summary = {
        "schema": SCHEMA,
        "diagnostic_only": not bool(args.formal_table_a),
        "formal_gate_eligible": bool(args.formal_table_a),
        "formal_table_a": bool(args.formal_table_a),
        "evaluation_profile": str(args.evaluation_profile),
        "formal_contract": (
            {
                "full_dataset": True,
                "fixed_eval_seed_42": True,
                "true_role_swap_required": True,
                "profile_surface_isolated": True,
                "category_claim": "joint_canonical_prompt_plus_support_route",
                "patch_only_category_claim_eligible": False,
            }
            if args.formal_table_a
            else None
        ),
        "config": {
            "path": str(Path(args.config).expanduser().resolve()),
            "sha256": _file_sha256(Path(args.config).expanduser().resolve()),
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).expanduser().resolve()),
            "sha256": _file_sha256(Path(args.checkpoint).expanduser().resolve()),
        },
        "patch_rank_weight": float(
            getattr(cfg, "stage_b_v15_patch_rank_weight", 0.0)
        ),
        "fulltext_score_contract": (
            "exactly fused_rank_logit - patch_rank_weight * patch_canonical_logit"
        ),
        "text_only_scope": "ranking within fixed patch Top-K candidates",
        "true_role_swap": dict(
            TRUE_ROLE_SWAP_STATUS if args.true_role_swap else ROLE_SWAP_STATUS
        ),
        "oracle_topks": list(ORACLE_TOPKS),
        "oracle_includes_all": True,
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "max_batches": int(args.max_batches),
        "input_manifests": input_manifests,
        "screen_calibration": screen_calibration_summary,
        "ref": ref_summaries,
        "tn_positive_roles": tn_role_summary,
        "tn_counterfactual": aggregate_counterfactual_records(causal_records),
        "category_intervention": category_summary,
        "g5_tensor_equality_receipt": (
            _finalize_g5_equality_receipt(equality_receipt)
            if args.true_role_swap
            else None
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    records_path = (
        Path(args.per_example_jsonl).expanduser().resolve()
        if args.per_example_jsonl
        else output_dir / "role_causal.records.jsonl"
    )
    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_dir / "role_causal.summary.json"
    )
    _write_jsonl(records_path, all_records)
    summary["outputs"] = {
        "records": {
            "path": str(records_path),
            "sha256": _file_sha256(records_path),
            "rows": len(all_records),
        }
    }
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "records": str(records_path),
                "summary": str(summary_path),
                "num_records": len(all_records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
