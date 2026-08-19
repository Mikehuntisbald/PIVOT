#!/usr/bin/env python3
"""Select and render the deterministic ARROW qualitative appendix.

This script is deliberately separate from ``select_qualitative_examples.py``.
It performs no model forward, training, threshold fitting, or checkpoint
selection.  It scans only already-sealed FineCops, Admission-panel, and Test5
records; every selection predicate must hold for seeds 17, 42, and 73.

The output is a machine-readable selection receipt, a flat figure-source CSV,
and a single 3x3 supplement PDF.  Test5 records do not store predicted box
coordinates, so those panels draw the ground-truth box only and say so rather
than reconstructing an unsealed prediction.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SEEDS = (17, 42, 73)
NUMBER_REGISTRY = PAPER / "data/paper_numbers.json"


def load_sealed_thresholds() -> dict[int, float]:
    payload = json.loads(NUMBER_REGISTRY.read_text(encoding="utf-8"))
    numbers = payload["numbers"]
    return {
        seed: float(numbers[f"cross_benchmark.sealed_source_tau.seed{seed}"]["value"])
        for seed in SEEDS
    }


TAU = load_sealed_thresholds()

FINE_MANIFEST = Path(
    "/media/haoyi/T9/data/FineCops-Ref/v1/manifests/finecops_test_all.jsonl"
)
FINE_RECORDS = {
    seed: ROOT
    / f"outputs/arrow_finecops_20260819/evaluations/A/seed{seed}/records.jsonl"
    for seed in SEEDS
}

PANEL_MANIFEST = (
    ROOT
    / "data/ablations/stageb_table_a_category_intervention_20260717"
    / "category_intervention_pairs.jsonl"
)
PANEL_ROUTE_DIRS = {
    "V": "AR_A_PATCH",
    "T": "AR_B_TEXT",
    "N": "AR_C_NULL",
}
PANEL_RECORDS = {
    (route, seed): ROOT
    / "outputs/arrow_admission_input_20260818/evaluations"
    / directory
    / "fresh_panel"
    / f"seed{seed}.records.jsonl"
    for route, directory in PANEL_ROUTE_DIRS.items()
    for seed in SEEDS
}

TEST5_SPLITS = {
    "refcoco_testA": "refcoco_unc_testA.jsonl",
    "refcoco_testB": "refcoco_unc_testB.jsonl",
    "refcocop_testA": "refcocoplus_unc_testA.jsonl",
    "refcocop_testB": "refcocoplus_unc_testB.jsonl",
    "refcocog_test": "refcocog_umd_test.jsonl",
}
TEST5_ROOT = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/ref8_u50"
TEST5_INPUTS = {
    split: TEST5_ROOT / "refcoco_eval_inputs" / filename
    for split, filename in TEST5_SPLITS.items()
}
TEST5_BASE_RECORDS = {
    split: ROOT
    / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/per_example_records"
    / (
        "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch_"
        f"checkpoint0001__{split}.records.jsonl"
    )
    for split in TEST5_SPLITS
}
TEST5_ARROW_RECORDS = {
    (split, seed): TEST5_ROOT
    / "per_example_records"
    / f"confidence_seed{seed}_u50_checkpoint_iter__{split}.records.jsonl"
    for split in TEST5_SPLITS
    for seed in SEEDS
}

OUT_JSON = PAPER / "data/qualitative_appendix.json"
OUT_CSV = PAPER / "data/plot_sources/qualitative_appendix.csv"
OUT_PDF = PAPER / "qualitative/qualitative_appendix_grid.pdf"

COLORS = {
    "base": "#D55E00",
    "arrow": "#0072B2",
    "rank": "#CC79A7",
    "gt": "#009E73",
    "class_a": "#56B4E9",
    "class_b": "#E69F00",
    "failure": "#D55E00",
}


PUBLIC_NAME_REPLACEMENTS = (
    ("B58", "Frozen base"),
    ("b58", "frozen_base"),
    ("R100", "complete-expression ranker"),
    ("r100", "complete_expression_ranker"),
    ("D3", "isolated rejector"),
    ("d3", "isolated_rejector"),
)


def paper_facing(value: Any) -> Any:
    """Translate legacy identifiers in the committed, public plot-source CSV.

    The JSON receipt deliberately retains byte-level source/provenance names;
    this conversion is applied only to human-facing figure metadata.
    """

    if isinstance(value, dict):
        return {paper_facing(key): paper_facing(item) for key, item in value.items()}
    if isinstance(value, list):
        return [paper_facing(item) for item in value]
    if isinstance(value, tuple):
        return tuple(paper_facing(item) for item in value)
    if isinstance(value, str):
        result = value
        for legacy, public in PUBLIC_NAME_REPLACEMENTS:
            result = result.replace(legacy, public)
        return result
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def route_copy(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_probability": route["official_probability"],
        "raw_confidence": route["raw_confidence"],
        "top1_box_xyxy": route["top1_box_xyxy"],
        "top1_iou": route["top1_iou"],
        "top1_query_index": route["top1_query_index"],
    }


class ArtifactRegistry:
    """Hash source bytes once and reject conflicting registrations."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self._hash_cache: dict[Path, str] = {}

    def add(
        self,
        artifact_id: str,
        path: Path | str,
        role: str,
        *,
        expected_sha256: str | None = None,
    ) -> str:
        resolved = Path(path).resolve()
        require(resolved.is_file(), f"missing source artifact: {resolved}")
        digest = self._hash_cache.setdefault(resolved, sha256(resolved))
        if expected_sha256 is not None:
            require(
                digest == expected_sha256,
                f"SHA-256 mismatch for {resolved}: {digest} != {expected_sha256}",
            )
        item = {
            "path": str(resolved),
            "sha256": digest,
            "size_bytes": resolved.stat().st_size,
            "role": role,
        }
        if artifact_id in self.items:
            require(self.items[artifact_id] == item, f"artifact ID collision: {artifact_id}")
        else:
            self.items[artifact_id] = item
        return artifact_id


def add_fixed_sources(registry: ArtifactRegistry) -> dict[str, str]:
    ids: dict[str, str] = {}
    ids["selector"] = registry.add(
        "qualitative_appendix_selector", Path(__file__), "deterministic_selection_code"
    )
    ids["number_registry"] = registry.add(
        "qualitative_number_registry",
        NUMBER_REGISTRY,
        "sealed_threshold_and_metric_registry",
    )
    ids["fine_manifest"] = registry.add(
        "finecops_manifest", FINE_MANIFEST, "selection_population_manifest"
    )
    for seed, path in FINE_RECORDS.items():
        key = f"fine_seed{seed}"
        ids[key] = registry.add(
            f"finecops_arrow_v_seed{seed}_records", path, "sealed_model_records"
        )

    ids["panel_manifest"] = registry.add(
        "admission_panel_manifest", PANEL_MANIFEST, "selection_population_manifest"
    )
    for (route, seed), path in PANEL_RECORDS.items():
        key = f"panel_{route}_seed{seed}"
        ids[key] = registry.add(
            f"admission_panel_{route}_seed{seed}_records", path, "sealed_model_records"
        )

    for split in TEST5_SPLITS:
        ids[f"test5_input_{split}"] = registry.add(
            f"test5_{split}_input_manifest",
            TEST5_INPUTS[split],
            "selection_population_manifest",
        )
        ids[f"test5_base_{split}"] = registry.add(
            f"test5_{split}_b58_records",
            TEST5_BASE_RECORDS[split],
            "sealed_model_records",
        )
        for seed in SEEDS:
            ids[f"test5_arrow_{split}_seed{seed}"] = registry.add(
                f"test5_{split}_arrow_seed{seed}_records",
                TEST5_ARROW_RECORDS[split, seed],
                "sealed_model_records",
            )
    return ids


def load_finecops() -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, dict[str, Any]]],
]:
    manifests = load_jsonl(FINE_MANIFEST)
    manifest_by_id = {row["sample_id"]: row for row in manifests}
    require(len(manifest_by_id) == len(manifests), "FineCops sample_id is not unique")
    records: dict[int, dict[str, dict[str, Any]]] = {}
    manifest_order = [row["sample_id"] for row in manifests]
    for seed, path in FINE_RECORDS.items():
        rows = load_jsonl(path)
        require(
            [row["sample_id"] for row in rows] == manifest_order,
            f"FineCops record/manifest order mismatch for seed {seed}",
        )
        require(all(row["seed"] == seed for row in rows), f"FineCops seed tag mismatch: {seed}")
        records[seed] = {row["sample_id"]: row for row in rows}
    return manifests, records


def load_panel() -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]],
]:
    manifest_rows = load_jsonl(PANEL_MANIFEST)
    manifests: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifest_rows:
        meta = row["category_intervention"]
        require(row["sample_id"] == f"{meta['pair_id']}:{meta['arm']}", "panel sample ID drift")
        require(meta["arm"] not in manifests[meta["pair_id"]], "duplicate panel arm")
        manifests[meta["pair_id"]][meta["arm"]] = row
    require(all(set(arms) == {"A", "B"} for arms in manifests.values()), "panel pair missing arm")

    records: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]] = {}
    expected_ids = {row["sample_id"] for row in manifest_rows}
    for route in PANEL_ROUTE_DIRS:
        records[route] = {}
        for seed in SEEDS:
            rows = load_jsonl(PANEL_RECORDS[route, seed])
            require(len(rows) == len(expected_ids), f"panel row-count mismatch: {route}/{seed}")
            require({row["sample_id"] for row in rows} == expected_ids, f"panel identity drift: {route}/{seed}")
            require(all(row["seed"] == seed for row in rows), f"panel seed drift: {route}/{seed}")
            by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for row in rows:
                require(row["arm"] not in by_pair[row["pair_id"]], "duplicate panel record arm")
                by_pair[row["pair_id"]][row["arm"]] = row
            require(all(set(arms) == {"A", "B"} for arms in by_pair.values()), "panel record missing arm")
            records[route][seed] = by_pair
    return dict(manifests), records


def load_test5() -> dict[str, dict[str, Any]]:
    joined: dict[str, dict[str, Any]] = {}
    for split in TEST5_SPLITS:
        inputs = load_jsonl(TEST5_INPUTS[split])
        base_rows = load_jsonl(TEST5_BASE_RECORDS[split])
        arrow_rows = {
            seed: load_jsonl(TEST5_ARROW_RECORDS[split, seed]) for seed in SEEDS
        }
        require(len(inputs) == len(base_rows), f"Test5 input/base length mismatch: {split}")
        require(
            all(len(rows) == len(inputs) for rows in arrow_rows.values()),
            f"Test5 input/ARROW length mismatch: {split}",
        )
        actual_manifest_sha = sha256(TEST5_INPUTS[split])
        for index, (input_row, base) in enumerate(zip(inputs, base_rows)):
            require(base["manifest_index"] == index, f"Test5 base index drift: {split}/{index}")
            require(base["manifest_sha256"] == actual_manifest_sha, f"Test5 base manifest SHA drift: {split}")
            require(base["split"] == split and base["valid"], f"invalid Test5 base row: {split}/{index}")
            per_seed: dict[int, dict[str, Any]] = {}
            for seed in SEEDS:
                row = arrow_rows[seed][index]
                require(row["sample_id"] == base["sample_id"], f"Test5 sample drift: {split}/{seed}/{index}")
                require(row["manifest_index"] == index, f"Test5 ARROW index drift: {split}/{seed}/{index}")
                require(row["manifest_sha256"] == actual_manifest_sha, f"Test5 ARROW manifest SHA drift: {split}/{seed}")
                require(row["split"] == split and row["valid"], f"invalid Test5 ARROW row: {split}/{seed}/{index}")
                per_seed[seed] = row
            for key in ("image_id", "ann_id", "ref_id", "sent_id"):
                require(input_row[key] == base[key], f"Test5 input identity drift: {split}/{index}/{key}")
            require(base["sample_id"] not in joined, "duplicate Test5 sample identity")
            joined[base["sample_id"]] = {
                "split": split,
                "input": input_row,
                "base": base,
                "arrow": per_seed,
            }
    return joined


def all_seed_rows(
    records: dict[int, dict[str, dict[str, Any]]], sample_id: str
) -> list[dict[str, Any]]:
    return [records[seed][sample_id] for seed in SEEDS]


def require_nonempty(category: str, rows: list[Any]) -> None:
    require(rows, f"fail-close: no exact candidate satisfies {category}")


def select_fine_admission_rescue(
    manifests: list[dict[str, Any]], records: dict[int, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], int]:
    candidates = []
    for manifest in manifests:
        rows = all_seed_rows(records, manifest["sample_id"])
        if (
            manifest["finecops_kind"] == "positive"
            and manifest["finecops_support_covered"]
            and all(
                row["routes"]["r100_d3"]["top1_iou"] < 0.5
                <= row["routes"]["deployed"]["top1_iou"]
                for row in rows
            )
        ):
            candidates.append(manifest)
    require_nonempty("FineCops Admission rescue", candidates)

    def key(manifest: dict[str, Any]) -> tuple[float, float, str]:
        rows = all_seed_rows(records, manifest["sample_id"])
        return (
            mean(row["eligible_queries"] for row in rows),
            -mean(
                row["routes"]["deployed"]["top1_iou"]
                - row["routes"]["r100_d3"]["top1_iou"]
                for row in rows
            ),
            manifest["sample_id"],
        )

    return min(candidates, key=key), len(candidates)


def select_fine_admission_failure(
    manifests: list[dict[str, Any]], records: dict[int, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], int]:
    candidates = []
    for manifest in manifests:
        rows = all_seed_rows(records, manifest["sample_id"])
        if (
            manifest["finecops_kind"] == "positive"
            and manifest["finecops_support_covered"]
            and all(
                row["routes"]["deployed"]["top1_iou"] < 0.5
                <= row["routes"]["r100_d3"]["top1_iou"]
                for row in rows
            )
        ):
            candidates.append(manifest)
    require_nonempty("FineCops Admission failure", candidates)

    def key(manifest: dict[str, Any]) -> tuple[float, str]:
        rows = all_seed_rows(records, manifest["sample_id"])
        drop = mean(
            row["routes"]["r100_d3"]["top1_iou"]
            - row["routes"]["deployed"]["top1_iou"]
            for row in rows
        )
        return -drop, manifest["sample_id"]

    return min(candidates, key=key), len(candidates)


def select_fine_fixed_false_rejection(
    manifests: list[dict[str, Any]], records: dict[int, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], int]:
    candidates = []
    for manifest in manifests:
        rows = all_seed_rows(records, manifest["sample_id"])
        if (
            manifest["finecops_kind"] == "positive"
            and manifest["finecops_support_covered"]
            and all(
                row["routes"]["deployed"]["top1_iou"] >= 0.5
                and row["routes"]["deployed"]["raw_confidence"] < TAU[seed]
                for seed, row in zip(SEEDS, rows)
            )
        ):
            candidates.append(manifest)
    require_nonempty("FineCops fixed-threshold false rejection", candidates)

    def key(manifest: dict[str, Any]) -> tuple[float, float, str]:
        rows = all_seed_rows(records, manifest["sample_id"])
        return (
            mean(row["routes"]["deployed"]["raw_confidence"] / TAU[seed] for seed, row in zip(SEEDS, rows)),
            -mean(row["routes"]["deployed"]["top1_iou"] for row in rows),
            manifest["sample_id"],
        )

    return min(candidates, key=key), len(candidates)


def select_fine_hard_false_positive(
    manifests: list[dict[str, Any]], records: dict[int, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], int]:
    candidates = []
    for manifest in manifests:
        rows = all_seed_rows(records, manifest["sample_id"])
        level = manifest["finecops_negative_level"]
        if (
            manifest["finecops_kind"] == "text"
            and manifest["finecops_support_covered"]
            and level is not None
            and level >= 2
            and all(
                row["routes"]["deployed"]["raw_confidence"] >= TAU[seed]
                for seed, row in zip(SEEDS, rows)
            )
        ):
            candidates.append(manifest)
    require_nonempty("FineCops hard compositional false positive", candidates)

    def key(manifest: dict[str, Any]) -> tuple[float, float, str]:
        rows = all_seed_rows(records, manifest["sample_id"])
        return (
            -manifest["finecops_negative_level"],
            -mean(row["routes"]["deployed"]["raw_confidence"] / TAU[seed] for seed, row in zip(SEEDS, rows)),
            manifest["sample_id"],
        )

    return min(candidates, key=key), len(candidates)


def select_fine_tn_downrank(
    manifests: list[dict[str, Any]],
    records: dict[int, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    positives_by_annotation = {
        row["finecops_annotation_id"]: row
        for row in manifests
        if row["finecops_kind"] == "positive"
    }
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for negative in manifests:
        if negative["finecops_kind"] == "positive" or not negative["finecops_support_covered"]:
            continue
        parent = positives_by_annotation.get(negative["finecops_parent_positive_id"])
        if parent is None or not parent["finecops_support_covered"]:
            continue
        conditions = []
        for seed in SEEDS:
            neg = records[seed][negative["sample_id"]]
            pos = records[seed][parent["sample_id"]]
            conditions.append(
                neg["routes"]["b58"]["official_probability"]
                >= pos["routes"]["b58"]["official_probability"]
                and neg["routes"]["deployed"]["raw_confidence"]
                < pos["routes"]["deployed"]["raw_confidence"]
                and neg["routes"]["deployed"]["raw_confidence"] < TAU[seed]
            )
        if all(conditions):
            candidates.append((negative, parent))
    require_nonempty("FineCops paired external TN down-rank", candidates)

    def key(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, str]:
        negative, parent = pair
        gains = []
        for seed in SEEDS:
            neg = records[seed][negative["sample_id"]]
            pos = records[seed][parent["sample_id"]]
            gains.append(
                neg["routes"]["b58"]["official_probability"]
                - pos["routes"]["b58"]["official_probability"]
                + pos["routes"]["deployed"]["raw_confidence"]
                - neg["routes"]["deployed"]["raw_confidence"]
            )
        return -mean(gains), negative["sample_id"]

    selected_negative, selected_parent = min(candidates, key=key)
    return selected_negative, selected_parent, len(candidates)


def pair_success(
    panel_records: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]],
    route: str,
    seed: int,
    pair_id: str,
) -> bool:
    arms = panel_records[route][seed][pair_id]
    return all(
        arms[arm]["has_both_oracle_query_sets"] and arms[arm]["active_score_wins"]
        for arm in ("A", "B")
    )


def select_panel_contrast(
    panel_manifests: dict[str, dict[str, dict[str, Any]]],
    panel_records: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]],
    winner: str,
    loser: str,
) -> tuple[str, int]:
    candidates = []
    for pair_id in panel_manifests:
        oracle_complete = all(
            panel_records[route][seed][pair_id][arm]["has_both_oracle_query_sets"]
            for route in (winner, loser)
            for seed in SEEDS
            for arm in ("A", "B")
        )
        if (
            oracle_complete
            and all(pair_success(panel_records, winner, seed, pair_id) for seed in SEEDS)
            and all(not pair_success(panel_records, loser, seed, pair_id) for seed in SEEDS)
        ):
            candidates.append(pair_id)
    require_nonempty(f"panel {winner} success / {loser} failure", candidates)

    def key(pair_id: str) -> tuple[float, str]:
        min_margin = min(
            panel_records[winner][seed][pair_id][arm]["active_max_admission_score"]
            - panel_records[winner][seed][pair_id][arm]["counterfactual_max_admission_score"]
            for seed in SEEDS
            for arm in ("A", "B")
        )
        return -min_margin, pair_id

    return min(candidates, key=key), len(candidates)


def select_test5_base_rescue(test5: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], int]:
    candidates = [
        row
        for row in test5.values()
        if row["base"]["top1_iou"] < 0.5
        and all(row["arrow"][seed]["top1_iou"] >= 0.5 for seed in SEEDS)
    ]
    require_nonempty("Test5 base wrong / ARROW correct", candidates)

    def key(row: dict[str, Any]) -> tuple[float, str]:
        gain = mean(row["arrow"][seed]["top1_iou"] - row["base"]["top1_iou"] for seed in SEEDS)
        return -gain, row["base"]["sample_id"]

    return min(candidates, key=key), len(candidates)


def select_test5_rank_failure(test5: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], int]:
    candidates = [
        row
        for row in test5.values()
        if all(
            row["arrow"][seed]["eligible_query_best_iou"] >= 0.5
            and row["arrow"][seed]["top1_iou"] < 0.5
            for seed in SEEDS
        )
    ]
    require_nonempty("Test5 within-eligible rank failure", candidates)

    def key(row: dict[str, Any]) -> tuple[float, str]:
        gap = mean(
            row["arrow"][seed]["eligible_query_best_iou"]
            - row["arrow"][seed]["top1_iou"]
            for seed in SEEDS
        )
        return -gap, row["base"]["sample_id"]

    return min(candidates, key=key), len(candidates)


def fine_metrics(
    sample_id: str,
    records: dict[int, dict[str, dict[str, Any]]],
    *,
    parent_id: str | None = None,
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        row = records[seed][sample_id]
        item = {
            "eligible_queries": row["eligible_queries"],
            "tau": TAU[seed],
            "b58": route_copy(row["routes"]["b58"]),
            "r100_d3": route_copy(row["routes"]["r100_d3"]),
            "deployed": route_copy(row["routes"]["deployed"]),
            "deployed_accepts": row["routes"]["deployed"]["raw_confidence"] >= TAU[seed],
        }
        if parent_id is not None:
            parent = records[seed][parent_id]
            item["paired_parent"] = {
                "sample_id": parent_id,
                "b58": route_copy(parent["routes"]["b58"]),
                "deployed": route_copy(parent["routes"]["deployed"]),
            }
        per_seed[str(seed)] = item
    return per_seed


def panel_metrics(
    pair_id: str,
    panel_records: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route in PANEL_ROUTE_DIRS:
        result[route] = {}
        for seed in SEEDS:
            arms: dict[str, Any] = {}
            for arm in ("A", "B"):
                row = panel_records[route][seed][pair_id][arm]
                arms[arm] = {
                    "has_both_oracle_query_sets": row["has_both_oracle_query_sets"],
                    "active_score_wins": row["active_score_wins"],
                    "active_max_admission_score": row["active_max_admission_score"],
                    "counterfactual_max_admission_score": row["counterfactual_max_admission_score"],
                    "active_minus_counterfactual_margin": (
                        row["active_max_admission_score"]
                        - row["counterfactual_max_admission_score"]
                    ),
                    "eligible_query_count": row["eligible_query_count"],
                }
            result[route][str(seed)] = {
                "pair_switch_success": pair_success(panel_records, route, seed, pair_id),
                "arms": arms,
            }
    return result


def test5_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "b58_seed42_invariant": {
            "top1_iou": row["base"]["top1_iou"],
            "all_query_best_iou": row["base"]["all_query_best_iou"],
        },
        "arrow": {
            str(seed): {
                "top1_iou": row["arrow"][seed]["top1_iou"],
                "eligible_query_best_iou": row["arrow"][seed]["eligible_query_best_iou"],
                "all_query_best_iou": row["arrow"][seed]["all_query_best_iou"],
                "eligible_queries": row["arrow"][seed]["stage_b_u2v2_eligible_queries"],
            }
            for seed in SEEDS
        },
    }


def add_selected_image(
    registry: ArtifactRegistry,
    artifact_id: str,
    path: str,
    expected_sha: str | None,
) -> str:
    return registry.add(artifact_id, path, "qualitative_source_image", expected_sha256=expected_sha)


def add_selected_support(
    registry: ArtifactRegistry,
    artifact_id: str,
    path: str,
    expected_sha: str,
) -> str:
    return registry.add(artifact_id, path, "qualitative_support_crop", expected_sha256=expected_sha)


def build_examples(
    registry: ArtifactRegistry,
    fixed_ids: dict[str, str],
    fine_manifests: list[dict[str, Any]],
    fine_records: dict[int, dict[str, dict[str, Any]]],
    panel_manifests: dict[str, dict[str, dict[str, Any]]],
    panel_records: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]],
    test5: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    base_rescue, base_count = select_test5_base_rescue(test5)
    admission_rescue, admission_count = select_fine_admission_rescue(fine_manifests, fine_records)
    v_pair, v_count = select_panel_contrast(panel_manifests, panel_records, "V", "T")
    t_pair, t_count = select_panel_contrast(panel_manifests, panel_records, "T", "N")
    tn_negative, tn_parent, tn_count = select_fine_tn_downrank(fine_manifests, fine_records)
    fixed_reject, fixed_count = select_fine_fixed_false_rejection(fine_manifests, fine_records)
    hard_fp, hard_count = select_fine_hard_false_positive(fine_manifests, fine_records)
    admission_failure, admission_failure_count = select_fine_admission_failure(
        fine_manifests, fine_records
    )
    rank_failure, rank_failure_count = select_test5_rank_failure(test5)

    examples: list[dict[str, Any]] = []

    def test_example(
        *,
        order: int,
        category: str,
        label: str,
        selected: dict[str, Any],
        predicate_id: str,
        predicate: str,
        selection_order: str,
        candidate_count: int,
        display_note: str,
    ) -> None:
        sample_id = selected["base"]["sample_id"]
        split = selected["split"]
        image_path = selected["input"]["filename"]
        image_id = add_selected_image(
            registry,
            f"qual_image_test5_{hashlib.sha256(sample_id.encode()).hexdigest()[:16]}",
            image_path,
            None,
        )
        source_ids = [
            fixed_ids["selector"],
            fixed_ids["number_registry"],
            fixed_ids[f"test5_input_{split}"],
            fixed_ids[f"test5_base_{split}"],
            *(fixed_ids[f"test5_arrow_{split}_seed{seed}"] for seed in SEEDS),
            image_id,
        ]
        instance = selected["input"]["instances"][0]
        examples.append(
            {
                "order": order,
                "category": category,
                "label": label,
                "surface": "Test5",
                "sample_id": sample_id,
                "predicate_id": predicate_id,
                "predicate": predicate,
                "selection_order": selection_order,
                "candidate_count": candidate_count,
                "seeds": list(SEEDS),
                "source_artifact_ids": source_ids,
                "image_artifact_id": image_id,
                "expression": instance.get("positive_phrase") or instance.get("raw_phrase"),
                "expression_scope": "sealed evaluation-input positive_phrase",
                "ground_truth_bbox_xywh": instance["bbox"],
                "metrics": test5_metrics(selected),
                "display_note": display_note,
            }
        )

    def fine_example(
        *,
        order: int,
        category: str,
        label: str,
        manifest: dict[str, Any],
        predicate_id: str,
        predicate: str,
        selection_order: str,
        candidate_count: int,
        display_note: str,
        parent: dict[str, Any] | None = None,
    ) -> None:
        sample_id = manifest["sample_id"]
        image_artifact = manifest["finecops_image_artifact"]
        image_id = add_selected_image(
            registry,
            f"qual_image_fine_{hashlib.sha256(sample_id.encode()).hexdigest()[:16]}",
            image_artifact["path"],
            image_artifact["sha256"],
        )
        source_ids = [
            fixed_ids["selector"],
            fixed_ids["number_registry"],
            fixed_ids["fine_manifest"],
            *(fixed_ids[f"fine_seed{seed}"] for seed in SEEDS),
            image_id,
        ]
        support_id = None
        support = manifest.get("finecops_support")
        if support and support.get("path"):
            support_id = add_selected_support(
                registry,
                f"qual_support_{support['sha256'][:16]}",
                support["path"],
                support["sha256"],
            )
            source_ids.append(support_id)
        parent_id = parent["sample_id"] if parent is not None else None
        examples.append(
            {
                "order": order,
                "category": category,
                "label": label,
                "surface": "FineCops-Ref",
                "sample_id": sample_id,
                "paired_parent_sample_id": parent_id,
                "predicate_id": predicate_id,
                "predicate": predicate,
                "selection_order": selection_order,
                "candidate_count": candidate_count,
                "seeds": list(SEEDS),
                "source_artifact_ids": source_ids,
                "image_artifact_id": image_id,
                "support_artifact_id": support_id,
                "kind": manifest["finecops_kind"],
                "expression": manifest["finecops_expression"],
                "ground_truth_bbox_xywh": manifest["finecops_bbox_xywh"],
                "negative_type": manifest["finecops_negative_type"],
                "negative_level": manifest["finecops_negative_level"],
                "tuple_type": manifest["finecops_tuple_type"],
                "metrics": fine_metrics(sample_id, fine_records, parent_id=parent_id),
                "display_note": display_note,
            }
        )

    def panel_example(
        *,
        order: int,
        category: str,
        label: str,
        pair_id: str,
        predicate_id: str,
        predicate: str,
        selection_order: str,
        candidate_count: int,
        routes: tuple[str, str],
        display_note: str,
    ) -> None:
        manifest_a = panel_manifests[pair_id]["A"]["category_intervention"]
        manifest_b = panel_manifests[pair_id]["B"]["category_intervention"]
        require(manifest_a["image_sha256"] == manifest_b["image_sha256"], "panel arm image drift")
        image_id = add_selected_image(
            registry,
            f"qual_image_panel_{manifest_a['image_sha256'][:16]}",
            manifest_a["image_path"],
            manifest_a["image_sha256"],
        )
        source_ids = [
            fixed_ids["selector"],
            fixed_ids["number_registry"],
            fixed_ids["panel_manifest"],
            image_id,
        ]
        for route in routes:
            source_ids.extend(fixed_ids[f"panel_{route}_seed{seed}"] for seed in SEEDS)
        support_ids = []
        for class_name in ("class_a", "class_b"):
            class_meta = manifest_a[class_name]
            support_id = add_selected_support(
                registry,
                f"qual_support_{class_meta['support_sha256'][:16]}",
                class_meta["support_path"],
                class_meta["support_sha256"],
            )
            support_ids.append(support_id)
            source_ids.append(support_id)
        examples.append(
            {
                "order": order,
                "category": category,
                "label": label,
                "surface": "Admission category-intervention panel",
                "sample_id": pair_id,
                "pair_id": pair_id,
                "predicate_id": predicate_id,
                "predicate": predicate,
                "selection_order": selection_order,
                "candidate_count": candidate_count,
                "seeds": list(SEEDS),
                "source_artifact_ids": source_ids,
                "image_artifact_id": image_id,
                "support_artifact_ids": support_ids,
                "class_a": manifest_a["class_a"],
                "class_b": manifest_a["class_b"],
                "metrics": panel_metrics(pair_id, panel_records),
                "display_note": display_note,
            }
        )

    test_example(
        order=1,
        category="base_wrong_arrow_correct",
        label="Base wrong, ARROW correct",
        selected=base_rescue,
        predicate_id="test5_base_wrong_arrow_correct_all_seeds",
        predicate="B58 top-1 IoU < 0.5 and ARROW top-1 IoU >= 0.5 for each seed 17/42/73.",
        selection_order="maximum mean(ARROW IoU - B58 IoU), then lexical sample_id",
        candidate_count=base_count,
        display_note="GT only: sealed Test5 records bind IoU but do not contain prediction coordinates.",
    )
    fine_example(
        order=2,
        category="admission_rescue",
        label="Admission rescue",
        manifest=admission_rescue,
        predicate_id="finecops_admission_rescue_all_seeds",
        predicate="Covered positive with raw-R100 IoU < 0.5 and deployed ARROW-V IoU >= 0.5 for every seed.",
        selection_order="minimum mean eligible-query count, maximum mean IoU recovery, then lexical sample_id",
        candidate_count=admission_count,
        display_note="Raw R100 and deployed ARROW-V boxes shown for seed 42; predicate holds for all seeds.",
    )
    panel_example(
        order=3,
        category="visual_switch_success_text_failure",
        label="V switch succeeds, T fails",
        pair_id=v_pair,
        predicate_id="panel_v_success_t_failure_all_seeds",
        predicate="Both visual-support arms switch correctly for every seed; the canonical-text pair fails in every seed; all oracle query sets exist.",
        selection_order="maximum minimum V active-minus-counterfactual margin over two arms and three seeds, then lexical pair_id",
        candidate_count=v_count,
        routes=("V", "T"),
        display_note="Class GT boxes and the two visual supports are shown; no predicted box is reconstructed.",
    )
    panel_example(
        order=4,
        category="text_switch_success_null_failure",
        label="T switch succeeds, N fails",
        pair_id=t_pair,
        predicate_id="panel_t_success_n_failure_all_seeds",
        predicate="Both canonical-text arms switch correctly for every seed; the category-agnostic null pair fails in every seed; all oracle query sets exist.",
        selection_order="maximum minimum T active-minus-counterfactual margin over two arms and three seeds, then lexical pair_id",
        candidate_count=t_count,
        routes=("T", "N"),
        display_note="Class GT boxes visualize the controlled category pair; N is invariant to category input by construction.",
    )
    fine_example(
        order=5,
        category="external_tn_downrank",
        label="External TN down-rank",
        manifest=tn_negative,
        parent=tn_parent,
        predicate_id="finecops_paired_tn_ordering_flip_and_fixed_rejection_all_seeds",
        predicate="For each seed, B58 ranks the negative at least as high as its paired positive, while D3 ranks it lower and its score is below the sealed D3 threshold.",
        selection_order="maximum mean paired ordering improvement across B58 and D3, then lexical negative sample_id",
        candidate_count=tn_count,
        display_note="Negative record boxes shown for seed 42; paired score inequalities hold for all seeds.",
    )
    fine_example(
        order=6,
        category="fixed_threshold_external_false_rejection",
        label="Fixed-tau false rejection",
        manifest=fixed_reject,
        predicate_id="finecops_fixed_tau_false_rejection_all_seeds",
        predicate="Covered positive is localized at IoU >= 0.5 but D3 raw confidence is below that seed's sealed D3 calibration threshold for every seed.",
        selection_order="minimum mean confidence/tau ratio, maximum mean deployed IoU, then lexical sample_id",
        candidate_count=fixed_count,
        display_note="A domain-shift failure: localization is correct but the fixed source-domain operating point rejects it.",
    )
    fine_example(
        order=7,
        category="hard_compositional_false_positive",
        label="Hard compositional false positive",
        manifest=hard_fp,
        predicate_id="finecops_hard_compositional_false_positive_all_seeds",
        predicate="Covered FineCops text negative at level >= 2 has D3 raw confidence at or above the sealed threshold for every seed.",
        selection_order="maximum negative level, maximum mean confidence/tau ratio, then lexical sample_id",
        candidate_count=hard_count,
        display_note="ARROW failure: the compositional text negative is falsely accepted for all seeds.",
    )
    fine_example(
        order=8,
        category="arrow_failure_admission_drop",
        label="ARROW failure: Admission drop",
        manifest=admission_failure,
        predicate_id="finecops_admission_drops_correct_r100_all_seeds",
        predicate="Covered positive has raw-R100 IoU >= 0.5 but deployed ARROW-V IoU < 0.5 for every seed.",
        selection_order="maximum mean raw-R100-to-ARROW IoU drop, then lexical sample_id",
        candidate_count=admission_failure_count,
        display_note="A useful R100 winner is removed by Admission for every seed.",
    )
    test_example(
        order=9,
        category="arrow_failure_within_eligible_rank",
        label="ARROW failure: within-eligible rank",
        selected=rank_failure,
        predicate_id="test5_within_eligible_rank_failure_all_seeds",
        predicate="For every seed, an eligible query reaches IoU >= 0.5 but ARROW's selected top-1 has IoU < 0.5.",
        selection_order="maximum mean(eligible-best IoU - top-1 IoU), then lexical sample_id",
        candidate_count=rank_failure_count,
        display_note="GT only: sealed Test5 records bind IoU but do not contain prediction coordinates.",
    )

    require([example["order"] for example in examples] == list(range(1, 10)), "example order drift")
    require(len({example["sample_id"] for example in examples}) == len(examples), "duplicate selected identity")
    require(
        sum(example["category"].startswith("arrow_failure") for example in examples) >= 2,
        "fewer than two explicitly labeled ARROW failures",
    )
    require(
        len({registry.items[example["image_artifact_id"]]["sha256"] for example in examples})
        == len(examples),
        "selected examples do not have distinct source images",
    )
    return examples


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def draw_box(
    axis: Any,
    xyxy: Iterable[float],
    color: str,
    label: str,
    linewidth: float = 1.8,
) -> None:
    x1, y1, x2, y2 = xyxy
    axis.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )
    axis.text(
        x1,
        max(1, y1),
        label,
        color="white",
        fontsize=6.4,
        va="top",
        ha="left",
        bbox={"facecolor": color, "edgecolor": "none", "pad": 1.1, "alpha": 0.92},
    )


def short_text(value: str, width: int = 43, lines: int = 2) -> str:
    return "\n".join(textwrap.wrap(value, width=width, max_lines=lines, placeholder="..."))


def render_fine(axis: Any, example: dict[str, Any], registry: ArtifactRegistry) -> str:
    metrics = example["metrics"]["42"]
    kind = example["kind"]
    if kind == "positive":
        draw_box(axis, xywh_to_xyxy(example["ground_truth_bbox_xywh"]), COLORS["gt"], "GT")

    category = example["category"]
    if category == "admission_rescue":
        draw_box(axis, metrics["r100_d3"]["top1_box_xyxy"], COLORS["rank"], "Ranker")
        draw_box(axis, metrics["deployed"]["top1_box_xyxy"], COLORS["arrow"], "ARROW")
        note = f"seed42 IoU: R={metrics['r100_d3']['top1_iou']:.2f}, A={metrics['deployed']['top1_iou']:.2f}"
    elif category == "external_tn_downrank":
        draw_box(axis, metrics["b58"]["top1_box_xyxy"], COLORS["base"], "Frozen")
        draw_box(axis, metrics["deployed"]["top1_box_xyxy"], COLORS["arrow"], "Rejector")
        parent = metrics["paired_parent"]
        note = (
            f"seed42 Frozen n/p={metrics['b58']['official_probability']:.2f}/{parent['b58']['official_probability']:.2f}\n"
            f"Rejector n/p={metrics['deployed']['raw_confidence']:.2f}/{parent['deployed']['raw_confidence']:.2f}; reject"
        )
    elif category == "fixed_threshold_external_false_rejection":
        draw_box(axis, metrics["deployed"]["top1_box_xyxy"], COLORS["arrow"], "ARROW")
        note = (
            f"correct IoU={metrics['deployed']['top1_iou']:.2f}, but "
            f"c={metrics['deployed']['raw_confidence']:.2f}<tau={metrics['tau']:.2f}"
        )
    elif category == "hard_compositional_false_positive":
        draw_box(axis, metrics["deployed"]["top1_box_xyxy"], COLORS["failure"], "false accept")
        note = (
            f"{example['negative_type']} L{example['negative_level']}: "
            f"c={metrics['deployed']['raw_confidence']:.2f}>=tau={metrics['tau']:.2f}"
        )
    elif category == "arrow_failure_admission_drop":
        draw_box(axis, metrics["r100_d3"]["top1_box_xyxy"], COLORS["rank"], "Ranker")
        draw_box(axis, metrics["deployed"]["top1_box_xyxy"], COLORS["failure"], "ARROW")
        note = f"seed42 IoU: R={metrics['r100_d3']['top1_iou']:.2f}, A={metrics['deployed']['top1_iou']:.2f}"
    else:
        raise RuntimeError(f"unknown FineCops category: {category}")
    return note


def render_panel(axis: Any, example: dict[str, Any], registry: ArtifactRegistry) -> str:
    class_a = example["class_a"]
    class_b = example["class_b"]
    for box in class_a["boxes_xyxy"]:
        draw_box(axis, box, COLORS["class_a"], class_a["name"])
    for box in class_b["boxes_xyxy"]:
        draw_box(axis, box, COLORS["class_b"], class_b["name"])

    if example["category"] == "visual_switch_success_text_failure":
        for index, (support_id, label) in enumerate(
            zip(example["support_artifact_ids"], ("support A", "support B"))
        ):
            inset = axis.inset_axes([0.69, 0.69 - index * 0.25, 0.29, 0.22])
            inset.imshow(Image.open(registry.items[support_id]["path"]).convert("RGB"))
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(label, fontsize=6.4, pad=0.5, color="white", backgroundcolor="#333333")
        winner, loser = "V", "T"
    else:
        winner, loser = "T", "N"
    margins = [
        example["metrics"][winner][str(seed)]["arms"][arm]["active_minus_counterfactual_margin"]
        for seed in SEEDS
        for arm in ("A", "B")
    ]
    return f"{winner}: 6/6 switches (min margin {min(margins):.2f}); {loser}: fails each seed"


def render_test5(axis: Any, example: dict[str, Any]) -> str:
    draw_box(axis, xywh_to_xyxy(example["ground_truth_bbox_xywh"]), COLORS["gt"], "GT only")
    metrics = example["metrics"]
    if example["category"] == "base_wrong_arrow_correct":
        arrow_ious = [metrics["arrow"][str(seed)]["top1_iou"] for seed in SEEDS]
        return f"Frozen IoU={metrics['b58_seed42_invariant']['top1_iou']:.2f}; ARROW IoU={min(arrow_ious):.2f}-{max(arrow_ious):.2f}"
    gaps = [
        metrics["arrow"][str(seed)]["eligible_query_best_iou"]
        - metrics["arrow"][str(seed)]["top1_iou"]
        for seed in SEEDS
    ]
    return f"eligible good, top-1 wrong; IoU gap={min(gaps):.2f}-{max(gaps):.2f}"


def render_pdf(examples: list[dict[str, Any]], registry: ArtifactRegistry) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Match CVPR's 6.875in text width exactly.  The supplement places this PDF
    # at ``width=\textwidth``; an exact source width avoids hidden TeX downscale
    # of otherwise readable type.
    # Keep the source below 7.8in tall so the full-width supplement float and
    # its caption fit on one CVPR page.  We compact only vertical whitespace;
    # font sizes stay in physical points and are not scaled down.
    figure, axes = plt.subplots(3, 3, figsize=(6.875, 7.75))
    labels = "abcdefghi"
    for label, example, axis in zip(labels, examples, axes.flat):
        image_path = registry.items[example["image_artifact_id"]]["path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(
            f"({label}) {example['label']}", fontsize=7.7, fontweight="bold", pad=1.8
        )
        if example["surface"] == "FineCops-Ref":
            metrics_note = render_fine(axis, example, registry)
            expression = example["expression"]
        elif example["surface"] == "Admission category-intervention panel":
            metrics_note = render_panel(axis, example, registry)
            expression = f"switch: {example['class_a']['name']} <-> {example['class_b']['name']}"
        else:
            metrics_note = render_test5(axis, example)
            expression = example["expression"]

        axis.text(
            0.01,
            0.015,
            short_text(metrics_note, width=48, lines=2),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.4,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
        axis.text(
            0.5,
            -0.025,
            short_text(expression, width=48, lines=2),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=6.4,
        )

    figure.suptitle(
        "Deterministic qualitative handoffs and failures (all predicates hold for seeds 17/42/73)",
        fontsize=9.0,
        fontweight="bold",
        y=0.991,
    )
    figure.text(
        0.5,
        0.006,
        "Selection uses sealed records only; seed 42 boxes are drawn where stored. Test5 stores IoU but not prediction coordinates, so (a,i) show GT only.",
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.955,
        bottom=0.045,
        hspace=0.24,
        wspace=0.08,
    )
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fixed_time = dt.datetime(2026, 8, 19, 0, 0, 0, tzinfo=dt.timezone.utc)
    figure.savefig(
        OUT_PDF,
        format="pdf",
        metadata={
            "Title": "ARROW deterministic qualitative appendix",
            "Author": "ARROW authors",
            "Subject": "Zero-training selection from sealed records",
            "Keywords": "ARROW, qualitative appendix, visual grounding",
            "Creator": "paper/scripts/select_qualitative_appendix.py",
            "Producer": "Matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    plt.close(figure)


def write_outputs(
    examples: list[dict[str, Any]], registry: ArtifactRegistry
) -> None:
    receipt = {
        "schema": "arrow.paper.qualitative_appendix/v1",
        "status": "post_hoc_zero_training_deterministic_selection",
        "generated_by": "paper/scripts/select_qualitative_appendix.py",
        "seeds": list(SEEDS),
        "sealed_d3_raw_thresholds": {str(seed): TAU[seed] for seed in SEEDS},
        "selection_scope": {
            "finecops": "sealed ARROW-V A-route records",
            "admission_panel": "sealed V/T/N category-intervention records",
            "test5": "sealed B58 and leakage-clean ARROW records",
        },
        "protocol_guards": [
            "No model forward, training, checkpoint selection, Gap tuning, or threshold fitting is performed.",
            "Every predicate is required independently for seeds 17, 42, and 73.",
            "Candidates are ranked by the declared numeric key and lexical identity; no visual judgment enters selection.",
            "FineCops fixed thresholds come verbatim from sealed D3 calibration.",
            "Test5 prediction coordinates are absent from sealed records and are not reconstructed.",
        ],
        "source_artifacts": dict(sorted(registry.items.items())),
        "selection": examples,
        "selection_count": len(examples),
        "all_requested_categories_exact": True,
        "fallbacks_used": [],
        "failure_count": sum(
            example["category"].startswith("arrow_failure") for example in examples
        ),
        "claim_boundary": (
            "These deterministically selected examples illustrate registered mechanisms and failures; "
            "they are not an estimator, a checkpoint-selection surface, or evidence beyond the bound records."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "order",
        "category",
        "label",
        "surface",
        "sample_id",
        "image_path",
        "image_sha256",
        "predicate_id",
        "predicate",
        "candidate_count",
        "selection_order",
        "seeds",
        "source_artifact_ids",
        "kind",
        "expression",
        "metrics_json",
        "display_note",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            image = registry.items[example["image_artifact_id"]]
            writer.writerow(
                {
                    "order": example["order"],
                    "category": example["category"],
                    "label": example["label"],
                    "surface": example["surface"],
                    "sample_id": example["sample_id"],
                    "image_path": image["path"],
                    "image_sha256": image["sha256"],
                    "predicate_id": paper_facing(example["predicate_id"]),
                    "predicate": paper_facing(example["predicate"]),
                    "candidate_count": example["candidate_count"],
                    "selection_order": paper_facing(example["selection_order"]),
                    "seeds": ";".join(str(seed) for seed in example["seeds"]),
                    "source_artifact_ids": ";".join(
                        paper_facing(example["source_artifact_ids"])
                    ),
                    "kind": example.get("kind", ""),
                    "expression": example.get("expression", ""),
                    "metrics_json": json.dumps(
                        paper_facing(example["metrics"]),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "display_note": paper_facing(example["display_note"]),
                }
            )


def main() -> None:
    registry = ArtifactRegistry()
    fixed_ids = add_fixed_sources(registry)
    fine_manifests, fine_records = load_finecops()
    panel_manifests, panel_records = load_panel()
    test5 = load_test5()
    examples = build_examples(
        registry,
        fixed_ids,
        fine_manifests,
        fine_records,
        panel_manifests,
        panel_records,
        test5,
    )
    write_outputs(examples, registry)
    render_pdf(examples, registry)
    print(
        json.dumps(
            {
                "selection_count": len(examples),
                "json": str(OUT_JSON),
                "csv": str(OUT_CSV),
                "pdf": str(OUT_PDF),
                "pdf_sha256": sha256(OUT_PDF),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
