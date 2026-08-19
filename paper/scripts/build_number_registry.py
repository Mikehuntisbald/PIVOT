#!/usr/bin/env python3
"""Build ARROW's source and semantic-number registries from sealed artifacts.

The paper, tables, and plots consume ``paper_numbers.json`` rather than
copying values from prose.  Every numeric entry records the exact artifact,
SHA-256 digest, JSON path, evaluation surface, and evidentiary status.

This script is intentionally read-only with respect to model outputs.  It
never opens checkpoints or per-example records and it performs no evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "paper" / "data"


SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "handoff": {
        "path": "ARROW_Codex_Handoff_v1.md",
        "kind": "specification",
        "owns": ["paper deliverables", "claim prohibitions", "validation gates"],
    },
    "blueprint": {
        "path": "ARROW_CVPR_Paper_Blueprint_v1.md",
        "kind": "prose_blueprint",
        "owns": ["paper framing", "section plan", "claim boundary"],
    },
    "ablation_design": {
        "path": "docs/paper_cvpr_u2v5_complete_ablation_design_20260817.md",
        "kind": "protocol",
        "owns": ["ablation registry", "selection policy", "statistics contract"],
    },
    "ablation_results_doc": {
        "path": "docs/paper_cvpr_u2v5_ablation_results_20260818.md",
        "kind": "results_narrative",
        "owns": ["main anchor", "ownership conclusions", "mechanism caveats"],
    },
    "admission_results_doc": {
        "path": "docs/paper_cvpr_arrow_admission_input_results_20260818.md",
        "kind": "results_narrative",
        "owns": ["ARROW-V/T/N naming", "category-switch interpretation"],
    },
    "finecops_protocol_doc": {
        "path": "docs/paper_cvpr_arrow_finecops_protocol_20260819.md",
        "kind": "protocol",
        "owns": ["FineCops evaluation surface", "support coverage contract"],
    },
    "finecops_results_doc": {
        "path": "docs/paper_cvpr_arrow_finecops_results_20260819.md",
        "kind": "results_narrative",
        "owns": ["FineCops results", "official-comparison claim boundary"],
    },
    "gref_results_doc": {
        "path": "docs/arrow_grefcoco_rejection_transfer_20260820.md",
        "kind": "results_narrative",
        "owns": ["gRefCOCO restricted scope", "overlap caveat", "threshold transfer"],
    },
    "clean_anchor_doc": {
        "path": "docs/stageb_u2v5_leakage_clean_anchor_20260817.md",
        "kind": "lineage_receipt",
        "owns": ["clean initializer", "checkpoint selection", "one-time held-out read"],
    },
    "anchor_bootstrap": {
        "path": "outputs/u2v5_cvpr_ablation_20260817/bootstrap/anchor_vs_b58.json",
        "kind": "sealed_result",
        "expected_sha256": "22210948d03d03e11cf7bb77a01ec7f31c6e737dded8c018663a73715f32aad7",
        "owns": ["Frozen-base to ARROW Test5 and Strict-TN gains"],
    },
    "ablation_tables": {
        "path": "outputs/u2v5_cvpr_ablation_20260817/paper_tables_v3.json",
        "kind": "sealed_result",
        "expected_sha256": "b5ca62d73075feb7760c79eac8d1d86fd3b4b487a2092424df7ceff2fc2be14f",
        "owns": ["A/C/O mechanism rows", "confirmatory contrasts"],
    },
    "zero_training": {
        "path": "outputs/u2v5_cvpr_ablation_20260817/zero_training_supplement_v3.json",
        "kind": "sealed_result",
        "expected_sha256": "a4a6d43fd705bbabaf578595fed46a8af632676e1c41cd0a92981bb009a43b1c",
        "owns": ["M routes", "error attribution", "sensitivity receipts"],
    },
    "ps_supplement": {
        "path": "outputs/u2v5_cvpr_ablation_20260817/zero_training_ps/summary.json",
        "kind": "sealed_result",
        "expected_sha256": "c459bd3323fff5b263bb2210d7c6af20e64491b18be8a54d642e25c6ea59560c",
        "owns": ["query/rank text interventions", "support interventions"],
    },
    "gap_sensitivity": {
        "path": "paper/data/gap_sensitivity.json",
        "kind": "zero_training_derived_receipt",
        "expected_sha256": "83bc3cebd3c74be515c1fc54de6b7cc69e331a497d16319be2528d689f7bd507",
        "owns": ["relative-gap validation sensitivity", "eligible-set size sensitivity"],
    },
    "admission_results": {
        "path": "outputs/arrow_admission_input_20260818/results.json",
        "kind": "sealed_result",
        "expected_sha256": "398e14081c9c226bf6136a4d5a5578ac385e63dd8dda63258e43aab1cf59f92b",
        "owns": ["ARROW-V/T/N Test5", "fresh category-switch panel"],
    },
    "finecops_results": {
        "path": "outputs/arrow_finecops_20260819/results.json",
        "kind": "sealed_result",
        "expected_sha256": "22e714a987ab88de749fd793e43c3d1194f161890d181c83bfe32b8b07b5ba11",
        "owns": ["FineCops official-byte metrics", "matched contrasts", "fixed-threshold transfer"],
    },
    "gref_results": {
        "path": "outputs/arrow_grefcoco_20260820/results.json",
        "kind": "sealed_result",
        "expected_sha256": "3b4841e0c87adb911b219f7bc7588b740e2b9ed346ec9630f4995ffa7ce4bde7",
        "owns": ["gRefCOCO restricted single/no-target metrics", "bootstrap gates"],
    },
    "confidence_anatomy": {
        "path": "paper/data/confidence_anatomy.json",
        "kind": "zero_training_derived_receipt",
        "owns": [
            "cross-benchmark rejection ordering summary",
            "sealed-threshold positive TPR transfer",
        ],
    },
    "efficiency_receipt": {
        "path": "paper/data/efficiency_receipt.json",
        "kind": "zero_training_measurement",
        "expected_sha256": "0cd0ea5acde4827a0bf21ad554d28258800ca9595c094264d6bfcdf8b8809498",
        "owns": [
            "parameter accounting",
            "batch-one latency and throughput",
            "peak allocated and reserved GPU memory",
        ],
    },
    "gref_dataset_manifest": {
        "path": "/media/haoyi/T9/data/gRefCOCO/v1/manifests/dataset_manifest.json",
        "kind": "sealed_dataset_manifest",
        "expected_sha256": "b8ae1ca5000677c007c609fd4c8a0e586f7a41658fa02bbd42ffb73b9df92a6a",
        "owns": ["gRefCOCO split counts", "single/no-target/multi scope"],
    },
    "gref_overlap_audit": {
        "path": "/media/haoyi/T9/data/gRefCOCO/v1/manifests/overlap_audit.json",
        "kind": "sealed_overlap_audit",
        "expected_sha256": "1b69209ab5feca1bbb07812393ffe770d876741e6718281b2d0daf1e655af36c",
        "owns": ["Rejector-supervision-disjoint counts", "Stage-A/R100/D3 exposure caveats"],
    },
    "arrow_release": {
        "path": "outputs/arrow_release_20260818/release_manifest.json",
        "kind": "release_manifest",
        "expected_sha256": "ebe587bee63ed4288f464d1a4872184735a2d0ba0b3ce8cba121973cf1bc49a7",
        "owns": ["public method identity", "sealed checkpoint hashes", "legacy ABI mapping"],
    },
}


OWNERSHIP_RECEIPTS = {
    seed: f"outputs/u2v5_cvpr_ablation_20260817/training/O0/seed{seed}/ownership_receipt.json"
    for seed in (17, 42, 73)
}


CLAIM_BOUNDARIES = {
    "task": "single-target selective visual grounding; emit one box or abstain",
    "not_full_grec": "multi-target gRefCOCO expressions are excluded",
    "finecops": "FineCops-specific annotation/task zero-shot, not globally image-disjoint",
    "finecops_visual_surface": "ARROW-V uses the preregistered 95.60%-covered exact-support surface",
    "gref": "previously exposed COCO imagery; Rejector-supervision-disjoint excludes only rejector train/calibration images",
    "verified_negatives": "proposal-covered verified negatives, not image-global or all-query verification",
    "calibration": "relative rejection discrimination transfers; the source operating threshold does not",
    "ownership": "exclusive parameter ownership is supported; phased scheduling is not independently superior",
    "positive_trust": "no standalone held-out causal gain is supported",
    "support": "visual support controls Admission but is not necessary for every correct top-1 prediction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, text=True, capture_output=True
    )
    value = result.stdout.strip()
    return value or None


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def load_json(relative: str) -> Any:
    with resolve_path(relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def at(value: Any, dotted_path: str) -> Any:
    current = value
    for token in dotted_path.split("."):
        if token == "":
            continue
        if isinstance(current, list):
            current = current[int(token)]
        else:
            current = current[token]
    return current


def mean_mapping(value: dict[str, float]) -> float:
    return statistics.fmean(value.values())


def source_registry() -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for key, spec in SOURCE_SPECS.items():
        relative = spec["path"]
        path = resolve_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"required paper source missing: {relative}")
        actual_sha = sha256(path)
        expected = spec.get("expected_sha256")
        if expected is not None and actual_sha != expected:
            raise RuntimeError(
                f"sealed source drift for {relative}: expected {expected}, got {actual_sha}"
            )
        repo_relative = not Path(relative).is_absolute()
        tracked = repo_relative and git_value("ls-files", "--error-unmatch", relative) is not None
        sources[key] = {
            "path": relative,
            "kind": spec["kind"],
            "sha256": actual_sha,
            "size_bytes": path.stat().st_size,
            "git_commit": git_value("log", "-1", "--format=%H", "--", relative)
            if tracked
            else None,
            "git_tracked": tracked,
            "claims_owned": spec["owns"],
        }
    return {
        "schema": "arrow.paper.source_registry/v1",
        "repository_root": str(ROOT),
        "sources": sources,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }


class NumberBuilder:
    def __init__(self, sources: dict[str, Any]) -> None:
        self.sources = sources
        self.payloads = {
            key: load_json(spec["path"])
            for key, spec in SOURCE_SPECS.items()
            if spec["path"].endswith(".json")
        }
        self.numbers: dict[str, dict[str, Any]] = {}

    def add(
        self,
        key: str,
        source: str,
        json_path: str,
        *,
        unit: str,
        surface: str,
        status: str,
        direction: str = "descriptive",
        notes: str = "",
        ci_path: str | None = None,
        sd_path: str | None = None,
        by_seed_path: str | None = None,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        raw = at(self.payloads[source], json_path)
        value = transform(raw) if transform else raw
        source_meta = self.sources["sources"][source]
        item: dict[str, Any] = {
            "key": key,
            "value": value,
            "unit": unit,
            "surface": surface,
            "direction": direction,
            "source_path": source_meta["path"],
            "source_sha256": source_meta["sha256"],
            "source_json_path": json_path,
            "status": status,
            "notes": notes,
        }
        if ci_path:
            item["ci95"] = at(self.payloads[source], ci_path)
        if sd_path:
            item["sample_sd"] = at(self.payloads[source], sd_path)
        if by_seed_path:
            item["by_seed"] = at(self.payloads[source], by_seed_path)
        if key in self.numbers:
            raise KeyError(f"duplicate semantic number key: {key}")
        self.numbers[key] = item

    def add_literal_from_receipt(
        self,
        key: str,
        source_path: str,
        json_path: str,
        *,
        unit: str,
        surface: str,
        status: str,
        direction: str = "descriptive",
        notes: str = "",
    ) -> None:
        path = resolve_path(source_path)
        payload = load_json(source_path)
        self.numbers[key] = {
            "key": key,
            "value": at(payload, json_path),
            "unit": unit,
            "surface": surface,
            "direction": direction,
            "source_path": source_path,
            "source_sha256": sha256(path),
            "source_json_path": json_path,
            "status": status,
            "notes": notes,
        }


def build_numbers(registry: dict[str, Any]) -> dict[str, Any]:
    b = NumberBuilder(registry)

    # Frozen-base -> final ARROW headline endpoints.
    for prefix, path, surface, direction in (
        ("main.test5", "test5", "Test5", "higher_is_better"),
        ("main.strict2031", "strict2031", "Strict-TN2031", "lower_is_better"),
    ):
        b.add(f"{prefix}.arrow", "anchor_bootstrap", f"{path}.candidate_seed_mean", unit="fraction", surface=surface, status="confirmatory", direction=direction, by_seed_path=f"{path}.candidate_by_seed")
        b.add(f"{prefix}.frozen_base", "anchor_bootstrap", f"{path}.reference_seed_mean", unit="fraction", surface=surface, status="confirmatory", direction=direction, by_seed_path=f"{path}.reference_by_seed")
        b.add(f"{prefix}.gain", "anchor_bootstrap", f"{path}.gain", unit="absolute", surface=surface, status="confirmatory", direction="higher_is_better_gain", ci_path=f"{path}.ci95", notes="paired image-cluster bootstrap, 5,000 replicates")

    # Cumulative route decomposition and candidate/error anatomy.
    route_map = {
        "m0_frozen_base": "M_cumulative_routes.M0_B58",
        "m1_complete_expression_ranker": "M_cumulative_routes.M1_positive_only_R100",
        "m3_trained_admission": "M_cumulative_routes.M3_trained_admission_identity_confidence",
        "m4_full_arrow_v": "M_cumulative_routes.M4_full_U2v5",
    }
    for name, path in route_map.items():
        b.add(f"cumulative.{name}.val3_acc50", "zero_training", f"{path}.mean", unit="fraction", surface="val3", status="mechanism", direction="higher_is_better", sd_path=f"{path}.sample_sd", by_seed_path=f"{path}.by_seed")
    b.add("cumulative.m2_static_admission.val3_acc50_seed17", "zero_training", "M_cumulative_routes.M2_static_admission_R100.seed17_val3_micro.by_seed.17", unit="fraction", surface="val3 seed17", status="mechanism", direction="higher_is_better", notes="single-seed parity route; not a three-seed mean")
    for key, path, unit in (
        ("anatomy.all_query_oracle_recall50", "variants.P0_S0.metrics.all_query_oracle_recall50", "fraction"),
        ("anatomy.eligible_gt_recall50", "variants.P0_S0.metrics.eligible_recall50", "fraction"),
        ("anatomy.mean_eligible_queries", "variants.P0_S0.metrics.eligible_query_count_mean", "queries"),
        ("anatomy.canonical_rank_acc50", "variants.P3.metrics.acc50", "fraction"),
        ("anatomy.canonical_rank_top1_churn", "variants.P3.metrics.top1_query_churn", "fraction"),
    ):
        b.add(key, "ps_supplement", path, unit=unit, surface="val3 seed42", status="exploratory_mechanism", direction="descriptive")
    for seed in (17, 42, 73):
        for error in ("admission", "rank", "correct"):
            b.add(f"anatomy.seed{seed}.{error}_rate", "zero_training", f"G_error_attribution.by_seed.{seed}.rate.{error}", unit="fraction", surface="val3", status="mechanism", direction="descriptive")

    # Admission training and ownership controls.
    for row in ("A1", "A3", "A4", "A5"):
        b.add(f"admission_training.{row.lower()}.val3_macro", "ablation_tables", f"mechanism.admission.{row}.macro.mean", unit="fraction", surface="val3", status="mechanism", direction="higher_is_better", sd_path=f"mechanism.admission.{row}.macro.sample_sd", by_seed_path=f"mechanism.admission.{row}.macro.by_seed")
        b.add(f"admission_training.{row.lower()}.val3_micro", "ablation_tables", f"mechanism.admission.{row}.micro.mean", unit="fraction", surface="val3", status="validation_mechanism", direction="higher_is_better", sd_path=f"mechanism.admission.{row}.micro.sample_sd", by_seed_path=f"mechanism.admission.{row}.micro.by_seed")
        for split in ("refcoco_val", "refcocop_val", "refcocog_val"):
            b.add(
                f"admission_training.{row.lower()}.{split}.mean", "ablation_tables",
                f"mechanism.admission.{row}.by_seed", unit="fraction",
                surface=f"{split}, mean over seeds", status="validation_mechanism",
                direction="higher_is_better",
                transform=lambda rows, split=split: statistics.fmean(
                    float(seed_row[split]) for seed_row in rows.values()
                ),
            )
        for seed in (17, 42, 73):
            b.add(
                f"admission_training.{row.lower()}.seed{seed}.val3_micro",
                "ablation_tables", f"mechanism.admission.{row}.by_seed.{seed}.micro",
                unit="fraction", surface=f"val3 seed {seed}",
                status="validation_mechanism", direction="higher_is_better",
            )
    b.add("admission_training.a0_a2.seed17_val3_micro", "zero_training", "M_cumulative_routes.M2_static_admission_R100.seed17_val3_micro.by_seed.17", unit="fraction", surface="val3 seed17", status="single_seed_parity_route", direction="higher_is_better", notes="A2 deployment is bitwise equal to the A0 initializer; not a three-seed aggregate")
    b.add("admission_training.a1.test5", "ablation_tables", "confirmatory.bootstrap.admission.test5.reference_seed_mean", unit="fraction", surface="Test5", status="confirmatory", direction="higher_is_better", by_seed_path="confirmatory.bootstrap.admission.test5.reference_by_seed")
    b.add("admission_training.a5.test5", "ablation_tables", "confirmatory.bootstrap.admission.test5.candidate_seed_mean", unit="fraction", surface="Test5", status="confirmatory", direction="higher_is_better", by_seed_path="confirmatory.bootstrap.admission.test5.candidate_by_seed")
    b.add("admission_training.a5_minus_a1.test5_gain", "ablation_tables", "confirmatory.bootstrap.admission.test5.gain", unit="absolute", surface="Test5", status="confirmatory", direction="higher_is_better_gain", ci_path="confirmatory.bootstrap.admission.test5.ci95")
    for row in ("C1", "C2", "C3", "C4", "D1"):
        for metric, direction in (("fpr95", "lower_is_better"), ("pair_win", "higher_is_better")):
            b.add(
                f"rejection_training.{row.lower()}.{metric}", "ablation_tables",
                f"mechanism.confidence_data.{row}.{metric}.mean", unit="fraction",
                surface="D3 calibration", status="calibration_mechanism",
                direction=direction,
                sd_path=f"mechanism.confidence_data.{row}.{metric}.sample_sd",
                by_seed_path=f"mechanism.confidence_data.{row}.{metric}.by_seed",
            )
    for row, role in (("c2", "reference"), ("c3", "candidate")):
        b.add(
            f"rejection_training.{row}.strict2031_fpr95", "ablation_tables",
            f"confirmatory.bootstrap.confidence.strict2031.{role}_seed_mean",
            unit="fraction", surface="Strict-TN2031", status="confirmatory",
            direction="lower_is_better",
            by_seed_path=f"confirmatory.bootstrap.confidence.strict2031.{role}_by_seed",
        )
    for row in ("O0", "O1", "O2", "O3"):
        b.add(f"ownership.{row.lower()}.val3_macro", "ablation_tables", f"mechanism.ownership.{row}.ref.macro.mean", unit="fraction", surface="val3", status="mechanism", direction="higher_is_better", sd_path=f"mechanism.ownership.{row}.ref.macro.sample_sd")
        b.add(f"ownership.{row.lower()}.val3_micro", "ablation_tables", f"mechanism.ownership.{row}.ref.micro.mean", unit="fraction", surface="val3", status="validation_mechanism", direction="higher_is_better", sd_path=f"mechanism.ownership.{row}.ref.micro.sample_sd", by_seed_path=f"mechanism.ownership.{row}.ref.micro.by_seed")
        b.add(f"ownership.{row.lower()}.calibration_fpr95", "ablation_tables", f"mechanism.ownership.{row}.confidence.fpr95.mean", unit="fraction", surface="D3 calibration", status="mechanism", direction="lower_is_better", sd_path=f"mechanism.ownership.{row}.confidence.fpr95.sample_sd")
        b.add(f"ownership.{row.lower()}.calibration_pair_win", "ablation_tables", f"mechanism.ownership.{row}.confidence.pair_win.mean", unit="fraction", surface="D3 calibration", status="calibration_mechanism", direction="higher_is_better", sd_path=f"mechanism.ownership.{row}.confidence.pair_win.sample_sd", by_seed_path=f"mechanism.ownership.{row}.confidence.pair_win.by_seed")
        for split in ("refcoco_val", "refcocop_val", "refcocog_val"):
            b.add(
                f"ownership.{row.lower()}.{split}.mean", "ablation_tables",
                f"mechanism.ownership.{row}.ref.by_seed", unit="fraction",
                surface=f"{split}, mean over seeds", status="validation_mechanism",
                direction="higher_is_better",
                transform=lambda rows, split=split: statistics.fmean(
                    float(seed_row[split]) for seed_row in rows.values()
                ),
            )
        for seed in (17, 42, 73):
            b.add(
                f"ownership.{row.lower()}.seed{seed}.val3_micro", "ablation_tables",
                f"mechanism.ownership.{row}.ref.by_seed.{seed}.micro",
                unit="fraction", surface=f"val3 seed {seed}",
                status="validation_mechanism", direction="higher_is_better",
            )
    b.add("ownership.o2_minus_o0.test5_gain", "ablation_tables", "confirmatory.bootstrap.ownership_isolation.test5.gain", unit="absolute", surface="Test5", status="confirmatory_route_preservation", direction="higher_is_better_gain", ci_path="confirmatory.bootstrap.ownership_isolation.test5.ci95")
    b.add("ownership.o2_minus_o0.strict_fpr95_reduction", "ablation_tables", "confirmatory.bootstrap.ownership_isolation.strict2031.gain", unit="absolute", surface="Strict-TN2031", status="confirmatory", direction="higher_is_better_reduction", ci_path="confirmatory.bootstrap.ownership_isolation.strict2031.ci95")
    b.add("ownership.o3_minus_o2.strict_fpr95_reduction", "ablation_tables", "confirmatory.bootstrap.ownership_schedule.strict2031.gain", unit="absolute", surface="Strict-TN2031", status="confirmatory_not_significant", direction="higher_is_better_reduction", ci_path="confirmatory.bootstrap.ownership_schedule.strict2031.ci95")
    for row, role in (("reference", "o0"), ("candidate", "o2")):
        b.add(f"ownership.{role}.test5", "ablation_tables", f"confirmatory.bootstrap.ownership_isolation.test5.{row}_seed_mean", unit="fraction", surface="Test5", status="confirmatory_route_preservation", direction="higher_is_better", by_seed_path=f"confirmatory.bootstrap.ownership_isolation.test5.{row}_by_seed")
        b.add(f"ownership.{role}.strict2031_fpr95", "ablation_tables", f"confirmatory.bootstrap.ownership_isolation.strict2031.{row}_seed_mean", unit="fraction", surface="Strict-TN2031", status="confirmatory", direction="lower_is_better", by_seed_path=f"confirmatory.bootstrap.ownership_isolation.strict2031.{row}_by_seed")
    b.add("ownership.o3.test5", "ablation_tables", "confirmatory.bootstrap.ownership_schedule.test5.candidate_seed_mean", unit="fraction", surface="Test5", status="confirmatory_noninferiority", direction="higher_is_better", by_seed_path="confirmatory.bootstrap.ownership_schedule.test5.candidate_by_seed")
    b.add("ownership.o3.strict2031_fpr95", "ablation_tables", "confirmatory.bootstrap.ownership_schedule.strict2031.candidate_seed_mean", unit="fraction", surface="Strict-TN2031", status="confirmatory_not_significant", direction="lower_is_better", by_seed_path="confirmatory.bootstrap.ownership_schedule.strict2031.candidate_by_seed")
    for seed, relative in OWNERSHIP_RECEIPTS.items():
        for metric in ("cosine_mean", "negative_cosine_fraction", "sign_conflict_mean"):
            b.add_literal_from_receipt(f"ownership.o0.seed{seed}.{metric}", relative, f"gradient_audit.{metric}", unit="fraction", surface="training trajectory", status="mechanism", notes="shared-score owner diagnostic")

    # Zero-training prompt and support interventions (seed 42 mechanism
    # surface).  They preserve sealed weights and change only the named input.
    for variant in ("P0_S0", "P1", "P2", "P3", "S1", "S2", "S3", "S4"):
        for metric, unit, direction in (
            ("acc50", "fraction", "higher_is_better"),
            ("all_query_oracle_recall50", "fraction", "higher_is_better"),
            ("eligible_recall50", "fraction", "higher_is_better"),
            ("eligible_query_count_mean", "queries", "descriptive"),
            ("eligible_mask_hamming_mean", "queries", "descriptive"),
            ("top1_query_churn", "fraction", "descriptive"),
        ):
            b.add(
                f"intervention.{variant.lower()}.{metric}", "ps_supplement",
                f"variants.{variant}.metrics.{metric}", unit=unit,
                surface="val3 seed42", status="zero_training_mechanism",
                direction=direction, notes="sealed-weight input intervention",
            )

    # Single-forward relative-gap sensitivity.  The derived receipt verifies
    # all 24 per-example files against the sealed summary and binds their
    # hashes before pooling the three validation splits.
    for index, label in enumerate(("0", "0p5", "1", "2", "3", "5", "10", "infinity")):
        b.add(
            f"gap_sensitivity.gap_{label}.gap", "gap_sensitivity",
            f"rows.{index}.gap", unit="relative_margin", surface="val3 seed42",
            status="exploratory_validation_only", direction="descriptive",
            notes="single-forward sweep setting",
        )
        b.add(
            f"gap_sensitivity.gap_{label}.n", "gap_sensitivity",
            f"rows.{index}.n", unit="expressions", surface="val3 seed42",
            status="exploratory_validation_only", direction="descriptive",
            notes="pooled validation expression count",
        )
        for metric, unit, direction in (
            ("acc50", "fraction", "higher_is_better"),
            ("eligible_recall50", "fraction", "higher_is_better"),
            ("mean_eligible_queries", "queries", "descriptive"),
        ):
            b.add(
                f"gap_sensitivity.gap_{label}.{metric}", "gap_sensitivity",
                f"rows.{index}.{metric}", unit=unit, surface="val3 seed42",
                status="exploratory_validation_only", direction=direction,
                notes="single-forward sweep; cannot change the sealed Gap3 model",
            )

    # Capacity-matched Admission-input routes.
    b.add("admission_input.arrow_v.test5", "anchor_bootstrap", "test5.candidate_seed_mean", unit="fraction", surface="Test5", status="post_release_frozen", direction="higher_is_better", by_seed_path="test5.candidate_by_seed")
    b.add("admission_input.arrow_t.test5", "admission_results", "test5.B_minus_A.candidate_seed_mean", unit="fraction", surface="Test5", status="post_release_frozen", direction="higher_is_better", by_seed_path="test5.B_minus_A.candidate_by_seed")
    b.add("admission_input.arrow_n.test5", "admission_results", "test5.C_minus_A.candidate_seed_mean", unit="fraction", surface="Test5", status="post_release_frozen", direction="higher_is_better", by_seed_path="test5.C_minus_A.candidate_by_seed")
    b.add("admission_input.arrow_v.switch_success", "admission_results", "panel.contrasts.visual_over_text.candidate_by_seed", unit="fraction", surface="fresh category-switch panel", status="confirmatory_mechanism", direction="higher_is_better", transform=mean_mapping)
    b.add("admission_input.arrow_t.switch_success", "admission_results", "panel.contrasts.visual_over_text.reference_by_seed", unit="fraction", surface="fresh category-switch panel", status="confirmatory_mechanism", direction="higher_is_better", transform=mean_mapping)
    b.add("admission_input.arrow_n.switch_success", "admission_results", "panel.contrasts.category_over_null.reference_by_seed", unit="fraction", surface="fresh category-switch panel", status="confirmatory_mechanism", direction="higher_is_better", transform=mean_mapping)
    b.add("admission_input.visual_over_text.switch_gain", "admission_results", "panel.contrasts.visual_over_text.gain", unit="absolute", surface="fresh category-switch panel", status="confirmatory_mechanism", direction="higher_is_better_gain", ci_path="panel.contrasts.visual_over_text.ci95")
    b.add("admission_input.text_over_null.switch_gain", "admission_results", "panel.contrasts.category_over_null.gain", unit="absolute", surface="fresh category-switch panel", status="confirmatory_mechanism", direction="higher_is_better_gain", ci_path="panel.contrasts.category_over_null.ci95")
    for route in ("A", "B", "C"):
        paper_name = {"A": "arrow_v", "B": "arrow_t", "C": "arrow_n"}[route]
        b.add(f"admission_input.{paper_name}.val3", "admission_results", f"val3.{route}.mean", unit="fraction", surface="val3", status="mechanism", direction="higher_is_better", sd_path=f"val3.{route}.sample_sd", by_seed_path=f"val3.{route}.by_seed")

    # FineCops official-byte results.  ARROW-V is on its exact-support surface;
    # all remaining route aggregates are full-test.
    b.add("finecops.arrow_v.support_coverage", "finecops_results", "a_support_contract.coverage", unit="fraction", surface="FineCops positives", status="external", direction="higher_is_better")
    for route, paper_name, scope in (
        ("A", "arrow_v", "exact-support matched surface"),
        ("B", "arrow_t", "full test"),
        ("C", "arrow_n", "full test"),
        ("B58", "frozen_base", "full test"),
        ("R100_D3", "ranker_rejector", "full test"),
    ):
        for metric, semantic in (
            ("precision1_macro", "positive_p1_macro"),
            ("precision1_micro", "positive_p1_micro"),
            ("text_recall1", "negative_text_recall1"),
            ("image_recall1", "negative_image_recall1"),
        ):
            b.add(f"finecops.{paper_name}.{semantic}", "finecops_results", f"seed_aggregates.{route}.{metric}.mean", unit="fraction", surface=f"FineCops {scope}", status="external", direction="higher_is_better", sd_path=f"seed_aggregates.{route}.{metric}.sample_sd", by_seed_path=f"seed_aggregates.{route}.{metric}.by_seed")

    # FineCops subgroup statistics that are explicitly present in the sealed
    # per-seed result object.  We report only the frozen-base and ARROW-T
    # full-test routes; no unregistered subgroup distribution is reconstructed.
    for route, paper_name in (("B58", "frozen_base"), ("B", "arrow_t")):
        for level in ("1", "2", "3"):
            b.add(
                f"finecops.subgroup.{paper_name}.positive_l{level}_p1",
                "finecops_results", f"metrics.{route}", unit="fraction",
                surface=f"FineCops positive level {level}", status="external_descriptive",
                direction="higher_is_better",
                transform=lambda rows, level=level: statistics.fmean(
                    float(seed_row["precision1_by_level"][level]) for seed_row in rows.values()
                ),
            )
        subgroup_names = {
            "text": (
                "attribute|L1", "attribute|L2", "object|L1", "object|L2",
                "order|L1", "order|L2", "relation|L1", "relation|L2",
                "swap_attr|L1", "swap_attr|L2",
            ),
            "image": (
                "attribute|L1", "attribute|L2", "flip|L1", "flip|L2",
                "object|L1", "object|L2", "order|L1", "swap_attr|L1",
                "swap_attr|L2",
            ),
        }
        for kind, names in subgroup_names.items():
            for subgroup in names:
                safe = subgroup.lower().replace("|", "_")
                b.add(
                    f"finecops.subgroup.{paper_name}.{kind}.{safe}.recall1",
                    "finecops_results", f"metrics.{route}", unit="fraction",
                    surface=f"FineCops negative-{kind} {subgroup}",
                    status="external_descriptive", direction="higher_is_better",
                    transform=lambda rows, kind=kind, subgroup=subgroup: statistics.fmean(
                        float(seed_row["rejection"][kind]["recall1_by_type_level"][subgroup]["recall1"])
                        for seed_row in rows.values()
                    ),
                )
                if route == "B":
                    b.add(
                        f"finecops.subgroup.count.{kind}.{safe}",
                        "finecops_results", f"metrics.{route}", unit="examples",
                        surface=f"FineCops negative-{kind} {subgroup}",
                        status="external_descriptive", direction="descriptive",
                        transform=lambda rows, kind=kind, subgroup=subgroup: next(iter({
                            int(seed_row["rejection"][kind]["recall1_by_type_level"][subgroup]["count"])
                            for seed_row in rows.values()
                        })),
                    )
    for contrast, semantic in (
        ("positive_A_minus_B", "visual_over_text.positive_p1_gain"),
        ("text_A_minus_B", "visual_over_text.negative_text_recall_gain"),
        ("image_A_minus_B", "visual_over_text.negative_image_recall_gain"),
        ("positive_B_minus_C", "text_over_null.positive_p1_gain"),
        ("text_B_minus_C", "text_over_null.negative_text_recall_gain"),
        ("image_B_minus_C", "text_over_null.negative_image_recall_gain"),
    ):
        b.add(f"finecops.{semantic}", "finecops_results", f"bootstrap.contrasts.{contrast}.gain", unit="absolute", surface="FineCops matched exact-support surface", status="external_confirmatory", direction="higher_is_better_gain", ci_path=f"bootstrap.contrasts.{contrast}.ci95")

    # Materialize matched-surface row values from the registered paired gains.
    # ARROW-V's route metrics are already evaluated on that exact-support
    # surface; subtracting the paired gains yields ARROW-T and ARROW-N on the
    # identical universe without mixing them with their full-test aggregates.
    matched_metrics = {
        "positive_p1_macro": ("precision1_macro", "positive_p1_gain"),
        "negative_text_recall1": ("text_recall1", "negative_text_recall_gain"),
        "negative_image_recall1": ("image_recall1", "negative_image_recall_gain"),
    }
    for semantic, (aggregate_name, gain_name) in matched_metrics.items():
        visual = b.numbers[f"finecops.arrow_v.{semantic}"]["value"]
        text_gain = b.numbers[f"finecops.visual_over_text.{gain_name}"]["value"]
        null_gain = b.numbers[f"finecops.text_over_null.{gain_name}"]["value"]
        for route, value, note in (
            ("arrow_v", visual, "direct exact-support route metric"),
            ("arrow_t", visual - text_gain, "derived from paired ARROW-V minus ARROW-T gain"),
            ("arrow_n", visual - text_gain - null_gain, "derived from paired ARROW-V/T and ARROW-T/N gains"),
        ):
            source_meta = registry["sources"]["finecops_results"]
            b.numbers[f"finecops.matched.{route}.{semantic}"] = {
                "key": f"finecops.matched.{route}.{semantic}",
                "value": value,
                "unit": "fraction",
                "surface": "FineCops matched exact-support surface",
                "direction": "higher_is_better",
                "source_path": source_meta["path"],
                "source_sha256": source_meta["sha256"],
                "source_json_path": "seed_aggregates.A and bootstrap.contrasts",
                "status": "external_confirmatory_derived",
                "notes": note,
            }

    # Official FineCops reference values are external literature numbers, not
    # ARROW record replay.  They are bound to the committed official-comparison
    # result narrative until the BibTeX/closest-work source package is sealed.
    for metric, value in (
        ("positive_p1_macro", 0.4845),
        ("negative_text_recall1", 0.3869),
        ("negative_image_recall1", 0.4314),
        ("negative_text_auroc_type_macro", 0.5398),
        ("negative_image_auroc_type_macro", 0.5652),
    ):
        source_meta = registry["sources"]["finecops_results_doc"]
        b.numbers[f"finecops.official_mm_gdino_t.{metric}"] = {
            "key": f"finecops.official_mm_gdino_t.{metric}",
            "value": value,
            "unit": "fraction",
            "surface": "FineCops full test (published reference)",
            "direction": "higher_is_better",
            "source_path": source_meta["path"],
            "source_sha256": source_meta["sha256"],
            "source_json_path": None,
            "status": "published_reference",
            "notes": "MM-GDINO-T zero-shot reference reported by FineCops-Ref",
        }
    official_source = registry["sources"]["finecops_results_doc"]
    b.numbers["finecops.official_mm_gdino_t.rejection_auroc_type_macro"] = {
        "key": "finecops.official_mm_gdino_t.rejection_auroc_type_macro",
        "value": statistics.fmean((0.5398, 0.5652)),
        "unit": "fraction",
        "surface": "FineCops full test (published type macro)",
        "direction": "higher_is_better",
        "source_path": official_source["path"],
        "source_sha256": official_source["sha256"],
        "source_json_path": None,
        "status": "published_reference_derived",
        "notes": "unweighted mean of published negative-text and negative-image AUROC type macros",
    }
    # Byte-exact ARROW replay through the pinned official evaluator. Keep these
    # distinct from the audited all-positive diagnostics below: the official
    # evaluator's historical overall rejection path uses level-1 positives.
    for route, text_auc, image_auc in (
        ("frozen_base", 0.5887, 0.5955),
        ("isolated_rejector", 0.6088, 0.6069),
    ):
        for negative_type, value in (
            ("negative_text", text_auc),
            ("negative_image", image_auc),
        ):
            key = f"finecops.official_exact.{route}.{negative_type}_auroc_type_macro"
            b.numbers[key] = {
                "key": key,
                "value": value,
                "unit": "fraction",
                "surface": "FineCops full test (official historical level-1-positive scope)",
                "direction": "higher_is_better",
                "source_path": official_source["path"],
                "source_sha256": official_source["sha256"],
                "source_json_path": None,
                "status": "external_official_exact",
                "notes": "byte-exact replay through the pinned official evaluator; not the audited all-positive metric",
            }
    finecops_fixed_by_seed = {
        str(seed): at(
            b.payloads["finecops_results"],
            f"metrics.B.{seed}.rejection.text.sealed_d3_threshold.positive_tpr",
        )
        for seed in (17, 42, 73)
    }
    fine_source = registry["sources"]["finecops_results"]
    b.numbers["finecops.arrow_t.fixed_tpr"] = {
        "key": "finecops.arrow_t.fixed_tpr",
        "value": statistics.fmean(finecops_fixed_by_seed.values()),
        "by_seed": finecops_fixed_by_seed,
        "sample_sd": statistics.stdev(finecops_fixed_by_seed.values()),
        "unit": "fraction",
        "surface": "FineCops full test",
        "direction": "target_0.95",
        "source_path": fine_source["path"],
        "source_sha256": fine_source["sha256"],
        "source_json_path": "metrics.B.*.rejection.text.sealed_d3_threshold.positive_tpr",
        "status": "external_fixed_source_threshold",
        "notes": "sealed D3 threshold; no FineCops threshold fitting",
    }

    # gRefCOCO restricted single/no-target transfer.  Paper-facing output must
    # use Rejector-supervision-disjoint, never a global image-disjoint label.
    for artifact_name, paper_name in (("full", "full"), ("d3_disjoint", "rejector_supervision_disjoint")):
        for metric in ("auroc", "aupr", "fpr95"):
            b.add(f"gref.{paper_name}.frozen_base.{metric}", "gref_results", f"surfaces.{artifact_name}.b58.{metric}", unit="fraction", surface=f"gRefCOCO {paper_name} restricted slice", status="external", direction="lower_is_better" if metric == "fpr95" else "higher_is_better")
            b.add(f"gref.{paper_name}.isolated_rejector.{metric}", "gref_results", f"surfaces.{artifact_name}.d3_summary.{metric}.mean", unit="fraction", surface=f"gRefCOCO {paper_name} restricted slice", status="external", direction="lower_is_better" if metric == "fpr95" else "higher_is_better", sd_path=f"surfaces.{artifact_name}.d3_summary.{metric}.sample_sd", by_seed_path=f"surfaces.{artifact_name}.d3_summary.{metric}.by_seed")
        b.add(f"gref.{paper_name}.isolated_rejector.fixed_tpr", "gref_results", f"surfaces.{artifact_name}.fixed_threshold_summary.positive_tpr.mean", unit="fraction", surface=f"gRefCOCO {paper_name} restricted slice", status="external_fixed_source_threshold", direction="target_0.95", sd_path=f"surfaces.{artifact_name}.fixed_threshold_summary.positive_tpr.sample_sd", by_seed_path=f"surfaces.{artifact_name}.fixed_threshold_summary.positive_tpr.by_seed", ci_path=f"bootstrap.surfaces.{artifact_name}.fixed_tpr.ci95")
        for metric in ("auroc_gain", "fpr95_gain"):
            b.add(f"gref.{paper_name}.{metric}", "gref_results", f"bootstrap.surfaces.{artifact_name}.{metric}.ci95", unit="absolute", surface=f"gRefCOCO {paper_name} restricted slice", status="external_confirmatory", direction="higher_is_better_gain", transform=lambda ci: None, notes="point gain is derived from the separately registered model means; CI stored below")
            b.numbers[f"gref.{paper_name}.{metric}"]["ci95"] = b.numbers[f"gref.{paper_name}.{metric}"]["source_json_path"] and at(b.payloads["gref_results"], f"bootstrap.surfaces.{artifact_name}.{metric}.ci95")
            b.numbers[f"gref.{paper_name}.{metric}"]["value"] = (
                b.numbers[f"gref.{paper_name}.isolated_rejector.{metric.removesuffix('_gain')}"]["value"]
                - b.numbers[f"gref.{paper_name}.frozen_base.{metric.removesuffix('_gain')}"]["value"]
                if metric == "auroc_gain"
                else b.numbers[f"gref.{paper_name}.frozen_base.fpr95"]["value"]
                - b.numbers[f"gref.{paper_name}.isolated_rejector.fpr95"]["value"]
            )
    for count_key, source, json_path in (
        ("full.images", "gref_overlap_audit", "surfaces.full.images"),
        ("full.positive", "gref_overlap_audit", "surfaces.full.positive"),
        ("full.no_target", "gref_overlap_audit", "surfaces.full.negative"),
        ("rejector_supervision_disjoint.images", "gref_overlap_audit", "surfaces.d3_disjoint.images"),
        ("rejector_supervision_disjoint.positive", "gref_overlap_audit", "surfaces.d3_disjoint.positive"),
        ("rejector_supervision_disjoint.no_target", "gref_overlap_audit", "surfaces.d3_disjoint.negative"),
        ("excluded_multi_target.testA", "gref_dataset_manifest", "counts.testA.multi"),
        ("excluded_multi_target.testB", "gref_dataset_manifest", "counts.testB.multi"),
    ):
        b.add(f"gref.scope.{count_key}", source, json_path, unit="count", surface="gRefCOCO dataset manifest", status="scope_contract", direction="descriptive", notes="multi-target rows are excluded")

    # Common external-transfer figure surface.  These values are recomputed
    # by build_confidence_anatomy.py from sealed per-example records.  FineCops
    # has no D3-vs-base rejection bootstrap receipt, so it remains explicitly
    # descriptive; the figure must not synthesize an interval for it.
    cross_specs = {
        "internal": {
            "surface": "Strict-TN2031",
            "status": "confirmatory_or_derived",
            "notes": "internal anchor; FPR95 contrast has paired image-cluster bootstrap",
            "fixed_tpr_path": "surfaces.internal.isolated_rejector_mean.fixed_threshold_positive_tpr",
        },
        "finecops": {
            "surface": "FineCops audited all-positive type macro",
            "status": "external_descriptive",
            "notes": "point estimate only; no D3-vs-base rejection bootstrap receipt exists",
            "fixed_tpr_path": "surfaces.finecops.isolated_rejector.fixed_threshold_positive_tpr",
        },
        "grefcoco": {
            "surface": "gRefCOCO restricted Full single/no-target",
            "status": "external_confirmatory",
            "notes": "paired stratified image-cluster bootstrap is available for ordering gains",
            "fixed_tpr_path": "surfaces.grefcoco.isolated_rejector.fixed_threshold_positive_tpr",
        },
    }
    for artifact_name, spec in cross_specs.items():
        for metric, path, direction in (
            ("auroc_gain", f"surfaces.{artifact_name}.auroc_gain", "higher_is_better_gain"),
            ("fpr95_reduction", f"surfaces.{artifact_name}.fpr95_reduction", "higher_is_better_reduction"),
            ("fixed_tpr", spec["fixed_tpr_path"], "target_0.95"),
            ("fixed_target_tpr", f"surfaces.{artifact_name}.fixed_target_tpr", "descriptive"),
        ):
            b.add(
                f"cross_benchmark.{artifact_name}.{metric}",
                "confidence_anatomy",
                path,
                unit="fraction",
                surface=spec["surface"],
                status=spec["status"],
                direction=direction,
                notes=spec["notes"],
            )
    for seed in (17, 42, 73):
        b.add(
            f"cross_benchmark.sealed_source_tau.seed{seed}",
            "confidence_anatomy",
            f"surfaces.internal.isolated_rejector_by_seed.{seed}.sealed_d3_threshold",
            unit="raw_score",
            surface="sealed 1,570-row D3 calibration",
            status="calibration_contract",
            direction="descriptive",
            notes="copied from sealed D3 calibration; never fitted on an external benchmark",
        )
    for route, artifact_route in (("frozen_base", "frozen_base"), ("isolated_rejector", "isolated_rejector")):
        for metric, direction in (("auroc_type_macro", "higher_is_better"), ("fpr95_type_macro", "lower_is_better")):
            b.add(
                f"finecops.rejection.{route}.{metric}", "confidence_anatomy",
                f"surfaces.finecops.{artifact_route}.{metric}", unit="fraction",
                surface="FineCops audited all-positive type macro",
                status="external_descriptive", direction=direction,
                notes="point estimate only; no D3-vs-base rejection bootstrap receipt exists",
            )
    # Reuse only already-registered bootstrap intervals; never invent an
    # interval for a point-estimate-only external surface.
    for cross_key, source_key in (
        ("cross_benchmark.internal.fpr95_reduction", "main.strict2031.gain"),
        ("cross_benchmark.grefcoco.auroc_gain", "gref.full.auroc_gain"),
        ("cross_benchmark.grefcoco.fpr95_reduction", "gref.full.fpr95_gain"),
        ("cross_benchmark.grefcoco.fixed_tpr", "gref.full.isolated_rejector.fixed_tpr"),
    ):
        source_item = b.numbers[source_key]
        if "ci95" in source_item:
            b.numbers[cross_key]["ci95"] = source_item["ci95"]
            b.numbers[cross_key]["ci_source_key"] = source_key

    # Formal zero-training efficiency measurement.  Latency is batch-one CUDA
    # event time with H2D included and file I/O excluded.  The runtime increase
    # includes the serialized training-only auxiliary and one frozen scale
    # scalar; the deployed-owner count excludes both.
    for key, path, notes in (
        ("total_params", "parameter_accounting.arrow_runtime_total", "all parameters loaded by the full ARROW runtime"),
        ("frozen_base_params", "parameter_accounting.frozen_base_total", "frozen candidate-generator/base parameter count"),
        ("cumulative_ever_trained_params", "parameter_accounting.cumulative_ever_trained_decision_owner_params", "sum of disjoint decision-owner parameters across phases"),
        ("deployed_owner_params", "parameter_accounting.deployed_decision_owner_params", "deployed rank, Admission-surface, and rejection owners"),
        ("inference_added_params", "parameter_accounting.runtime_loaded_increase", "increase over the frozen base, including serialized auxiliary and frozen scale"),
        ("training_only_auxiliary_params", "parameter_accounting.training_only_auxiliary_params", "serialized and computed but never used by the deployed gate/rank score"),
    ):
        b.add(
            f"efficiency.{key}", "efficiency_receipt", path,
            unit="parameters", surface="ARROW runtime", status="formal_zero_training_measurement",
            direction="descriptive", notes=notes,
        )
    for owner, path in (
        ("ranker", "parameter_accounting.phase_active.complete_expression_ranker"),
        ("admission", "parameter_accounting.phase_active.admission_surface_plus_training_only_auxiliary"),
        ("rejector", "parameter_accounting.phase_active.isolated_rejector"),
    ):
        b.add(
            f"efficiency.active.{owner}_params", "efficiency_receipt", path,
            unit="parameters", surface=f"{owner} training phase",
            status="formal_zero_training_measurement", direction="descriptive",
            notes="only this phase owner receives gradients",
        )

    # Cumulative route accounting is derived exactly from the disjoint owners
    # above.  These values populate the cumulative-route table without mixing
    # phase-active and runtime-loaded definitions.
    efficiency_source = registry["sources"]["efficiency_receipt"]
    ranker_params = int(b.numbers["efficiency.active.ranker_params"]["value"])
    admission_params = int(b.numbers["efficiency.active.admission_params"]["value"])
    rejector_params = int(b.numbers["efficiency.active.rejector_params"]["value"])
    for route, active, cumulative, loaded in (
        ("frozen_base", 0, 0, 0),
        ("ranker", ranker_params, ranker_params, ranker_params),
        ("static_admission", 0, ranker_params, None),
        ("learned_admission", admission_params, ranker_params + admission_params, ranker_params + admission_params + 1),
        ("full_arrow", rejector_params, ranker_params + admission_params + rejector_params, int(b.numbers["efficiency.inference_added_params"]["value"])),
    ):
        for metric, value in (("active_params", active), ("cumulative_params", cumulative), ("inference_added_params", loaded)):
            if value is None:
                continue
            key = f"efficiency.route.{route}.{metric}"
            b.numbers[key] = {
                "key": key,
                "value": value,
                "unit": "parameters",
                "surface": f"cumulative route: {route}",
                "direction": "descriptive",
                "source_path": efficiency_source["path"],
                "source_sha256": efficiency_source["sha256"],
                "source_json_path": "derived from parameter_accounting.phase_active",
                "status": "formal_zero_training_measurement_derived",
                "notes": "exact sum of disjoint parameter-owner counts",
            }

    variant_names = {
        0: "frozen_base",
        1: "ranker",
        2: "arrow_t",
        3: "arrow_v_uncached",
        4: "arrow_v_cached",
    }
    for index, name in variant_names.items():
        for metric, unit, direction in (
            ("median_ms", "milliseconds", "lower_is_better"),
            ("mean_ms", "milliseconds", "lower_is_better"),
            ("p90_ms", "milliseconds", "lower_is_better"),
            ("iqr_ms", "milliseconds", "lower_is_better"),
            ("throughput_images_per_second", "images_per_second", "higher_is_better"),
            ("peak_allocated_bytes", "bytes", "lower_is_better"),
            ("peak_reserved_bytes", "bytes", "lower_is_better"),
            ("model_parameters_loaded", "parameters", "descriptive"),
        ):
            b.add(
                f"efficiency.variant.{name}.{metric}", "efficiency_receipt",
                f"variants.{index}.{metric}", unit=unit,
                surface="batch-one inference, RTX 5090", status="formal_zero_training_measurement",
                direction=direction,
                notes="200 timed iterations after 50 warm-up iterations; CUDA events; H2D included; file I/O excluded",
            )

    return {
        "schema": "arrow.paper.semantic_number_registry/v1",
        "numbers": dict(sorted(b.numbers.items())),
        "claim_boundaries": CLAIM_BOUNDARIES,
        "pending_measurements": [],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files differ")
    args = parser.parse_args()

    sources = source_registry()
    numbers = build_numbers(sources)
    outputs = {
        DATA_DIR / "source_registry.json": sources,
        DATA_DIR / "paper_numbers.json": numbers,
    }
    if args.check:
        mismatches: list[str] = []
        for path, payload in outputs.items():
            expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(str(path.relative_to(ROOT)))
        if mismatches:
            raise SystemExit("stale paper registries: " + ", ".join(mismatches))
        return
    for path, payload in outputs.items():
        write_json(path, payload)


if __name__ == "__main__":
    main()
