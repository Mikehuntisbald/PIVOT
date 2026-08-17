#!/usr/bin/env python3
"""Paired image-cluster bootstrap for sealed U2-v5 Ref/TN records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA = "pivot.stageb.u2v5_paired_bootstrap/v1"
TEST5 = {
    "refcoco_testA", "refcoco_testB", "refcocop_testA",
    "refcocop_testB", "refcocog_test",
}
DEFAULT_STRICT1607 = Path(
    "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/"
    "semantic_stageb_union_image_disjoint_manifest.jsonl"
)


class BootstrapError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BootstrapError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise BootstrapError(f"invalid JSONL records: {path}")
    return rows


def _summary_record_paths(summary_path: Path, *, task: str) -> list[tuple[dict, Path]]:
    summary = _read_json(summary_path)
    key = "refcoco" if task == "ref" else "tn"
    result = []
    for row in summary.get(key, []):
        if task == "ref" and row.get("dataset") not in TEST5:
            continue
        raw = row.get("records_jsonl")
        if not isinstance(raw, str):
            raise BootstrapError(f"summary row lacks records_jsonl: {row.get('run_id')}")
        path = Path(raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        result.append((row, path.resolve(strict=True)))
    if not result:
        raise BootstrapError(f"summary has no {task} records: {summary_path}")
    return result


def _seed_from_run_id(run_id: str) -> int | None:
    for seed in (17, 42, 73):
        if f"seed{seed}" in str(run_id):
            return seed
    return None


def _index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise BootstrapError("records contain missing/duplicate sample IDs")
        if row.get("valid") is not True:
            raise BootstrapError(f"invalid evaluation record: {sample_id}")
        result[sample_id] = row
    return result


def _load_ref(summary_path: Path) -> dict[int | None, dict[str, Mapping[str, Any]]]:
    grouped: dict[int | None, list[dict]] = {}
    for row, path in _summary_record_paths(summary_path, task="ref"):
        grouped.setdefault(_seed_from_run_id(str(row.get("run_id", ""))), []).extend(
            _read_jsonl(path)
        )
    return {seed: _index(rows) for seed, rows in grouped.items()}


def _load_tn(summary_path: Path) -> dict[int | None, dict[str, Mapping[str, Any]]]:
    grouped: dict[int | None, list[dict]] = {}
    for row, path in _summary_record_paths(summary_path, task="tn"):
        grouped.setdefault(_seed_from_run_id(str(row.get("run_id", ""))), []).extend(
            _read_jsonl(path)
        )
    return {seed: _index(rows) for seed, rows in grouped.items()}


def _candidate_seeds(values: Mapping[int | None, Any]) -> tuple[int, ...]:
    seeds = tuple(sorted(seed for seed in values if seed is not None))
    if seeds != (17, 42, 73):
        raise BootstrapError(f"candidate records must contain seeds 17/42/73: {seeds}")
    return seeds


def _reference(values: Mapping[int | None, Any]) -> Any:
    if set(values) == {None}:
        return values[None]
    if len(values) == 1:
        return next(iter(values.values()))
    raise BootstrapError("reference summary must contain exactly one run")


def _align(candidate: Mapping[int | None, dict], reference: dict) -> tuple[tuple[int, ...], list[str]]:
    seeds = _candidate_seeds(candidate)
    ids = sorted(reference)
    for seed in seeds:
        if sorted(candidate[seed]) != ids:
            raise BootstrapError(f"seed {seed} sample IDs do not align to reference")
        for sample_id in ids:
            left, right = candidate[seed][sample_id], reference[sample_id]
            identity = ("image_id", "ann_id", "ref_id", "sent_id")
            if any(left.get(key) != right.get(key) for key in identity):
                raise BootstrapError(f"record identity drifted: {sample_id}")
    return seeds, ids


def _clusters(index: Mapping[str, Mapping[str, Any]], ids: list[str]) -> tuple[list[int], dict[int, list[str]]]:
    groups: dict[int, list[str]] = {}
    for sample_id in ids:
        groups.setdefault(int(index[sample_id]["image_id"]), []).append(sample_id)
    return sorted(groups), groups


def _percentile(draws: np.ndarray) -> list[float]:
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _pvalue(draws: np.ndarray) -> float:
    return float((1 + int(np.count_nonzero(draws <= 0.0))) / (draws.size + 1))


def _bootstrap_ref(candidate: dict, reference: dict, *, iterations: int, rng: np.random.Generator) -> dict[str, Any]:
    seeds, ids = _align(candidate, reference)
    images, groups = _clusters(reference, ids)
    ref_counts = np.asarray([len(groups[image]) for image in images], dtype=np.float64)
    ref_correct = np.asarray([
        sum(bool(reference[sid]["correct50"]) for sid in groups[image])
        for image in images
    ], dtype=np.float64)
    candidate_correct = {
        seed: np.asarray([
            sum(bool(candidate[seed][sid]["correct50"]) for sid in groups[image])
            for image in images
        ], dtype=np.float64)
        for seed in seeds
    }
    observed_ref = float(ref_correct.sum() / ref_counts.sum())
    observed_seeds = {
        seed: float(candidate_correct[seed].sum() / ref_counts.sum()) for seed in seeds
    }
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(images), size=len(images))
        count = ref_counts[sampled].sum()
        ref_metric = ref_correct[sampled].sum() / count
        candidate_metric = np.mean([
            candidate_correct[seed][sampled].sum() / count for seed in seeds
        ])
        draws[iteration] = candidate_metric - ref_metric
    observed_mean = float(np.mean(list(observed_seeds.values())))
    return {
        "metric": "test5_micro_acc50_gain",
        "expressions": len(ids),
        "unique_images": len(images),
        "candidate_by_seed": {str(k): v for k, v in observed_seeds.items()},
        "candidate_seed_mean": observed_mean,
        "reference": observed_ref,
        "gain": observed_mean - observed_ref,
        "ci95": _percentile(draws),
        "one_sided_p": _pvalue(draws),
    }


def _reference_by_seed(values: dict, seeds: tuple[int, ...]) -> dict[int, dict]:
    if set(values) == {None}:
        return {seed: values[None] for seed in seeds}
    if set(values) == set(seeds):
        return {seed: values[seed] for seed in seeds}
    raise BootstrapError("reference records must be shared or seed-aligned")


def _bootstrap_ref_comparison(candidate: dict, reference_values: dict, *, iterations: int, rng: np.random.Generator) -> dict[str, Any]:
    seeds = _candidate_seeds(candidate)
    references = _reference_by_seed(reference_values, seeds)
    ids = sorted(references[seeds[0]])
    for seed in seeds:
        _align({seed: candidate[seed], **{other: candidate[other] for other in seeds if other != seed}}, references[seed])
        if sorted(references[seed]) != ids:
            raise BootstrapError("reference seed sample IDs differ")
    images, groups = _clusters(references[seeds[0]], ids)
    counts = np.asarray([len(groups[image]) for image in images], dtype=np.float64)
    candidate_correct = {
        seed: np.asarray([sum(bool(candidate[seed][sid]["correct50"]) for sid in groups[image]) for image in images], dtype=np.float64)
        for seed in seeds
    }
    reference_correct = {
        seed: np.asarray([sum(bool(references[seed][sid]["correct50"]) for sid in groups[image]) for image in images], dtype=np.float64)
        for seed in seeds
    }
    candidate_observed = {seed: float(candidate_correct[seed].sum() / counts.sum()) for seed in seeds}
    reference_observed = {seed: float(reference_correct[seed].sum() / counts.sum()) for seed in seeds}
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(images), size=len(images))
        count = counts[sampled].sum()
        draws[iteration] = float(np.mean([
            candidate_correct[seed][sampled].sum() / count
            - reference_correct[seed][sampled].sum() / count
            for seed in seeds
        ]))
    candidate_mean = float(np.mean(list(candidate_observed.values())))
    reference_mean = float(np.mean(list(reference_observed.values())))
    return {
        "metric": "test5_micro_acc50_gain",
        "expressions": len(ids),
        "unique_images": len(images),
        "candidate_by_seed": {str(k): v for k, v in candidate_observed.items()},
        "reference_by_seed": {str(k): v for k, v in reference_observed.items()},
        "candidate_seed_mean": candidate_mean,
        "reference_seed_mean": reference_mean,
        "gain": candidate_mean - reference_mean,
        "ci95": _percentile(draws),
        "one_sided_p": _pvalue(draws),
    }


def _threshold(pos: np.ndarray, target_tpr: float = 0.95) -> float:
    pos = pos[np.isfinite(pos)]
    if not pos.size:
        return float("inf")
    accepted = max(1, int(math.ceil(float(target_tpr) * int(pos.size))))
    index = int(pos.size) - accepted
    return float(np.partition(pos, index)[index])


def _fpr(rows: Mapping[str, Mapping[str, Any]], ids: Iterable[str]) -> float:
    ids = list(ids)
    pos = np.asarray([float(rows[sid]["pos_score"]) for sid in ids], dtype=np.float64)
    neg = np.asarray([float(rows[sid]["neg_score"]) for sid in ids], dtype=np.float64)
    threshold = _threshold(pos)
    return float(np.mean(neg >= threshold))


def _strict1607_ids(path: Path) -> set[str]:
    return {str(row["sample_id"]) for row in _read_jsonl(path.resolve(strict=True))}


def _bootstrap_tn_surface(candidate: dict, reference: dict, ids: list[str], *, iterations: int, rng: np.random.Generator, name: str) -> dict[str, Any]:
    seeds, aligned = _align(candidate, reference)
    if set(ids) - set(aligned):
        raise BootstrapError(f"{name} IDs are not a subset of strict2031 records")
    ids = sorted(ids)
    images, groups = _clusters(reference, ids)
    observed_ref = _fpr(reference, ids)
    observed_seeds = {seed: _fpr(candidate[seed], ids) for seed in seeds}
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(images), size=len(images))
        sampled_ids = [sid for index in sampled for sid in groups[images[int(index)]]]
        ref_metric = _fpr(reference, sampled_ids)
        candidate_metric = np.mean([
            _fpr(candidate[seed], sampled_ids) for seed in seeds
        ])
        draws[iteration] = ref_metric - candidate_metric
    observed_mean = float(np.mean(list(observed_seeds.values())))
    return {
        "metric": f"{name}_fpr95_reduction",
        "pairs": len(ids),
        "unique_images": len(images),
        "candidate_by_seed": {str(k): v for k, v in observed_seeds.items()},
        "candidate_seed_mean": observed_mean,
        "reference": observed_ref,
        "gain": observed_ref - observed_mean,
        "ci95": _percentile(draws),
        "one_sided_p": _pvalue(draws),
    }


def _bootstrap_tn_comparison(candidate: dict, reference_values: dict, ids: list[str], *, iterations: int, rng: np.random.Generator, name: str) -> dict[str, Any]:
    seeds = _candidate_seeds(candidate)
    references = _reference_by_seed(reference_values, seeds)
    aligned = sorted(references[seeds[0]])
    if set(ids) - set(aligned):
        raise BootstrapError(f"{name} IDs are outside the aligned universe")
    ids = sorted(ids)
    for seed in seeds:
        if sorted(candidate[seed]) != sorted(references[seed]) or sorted(references[seed]) != aligned:
            raise BootstrapError("seeded TN records do not align")
    images, groups = _clusters(references[seeds[0]], ids)
    candidate_observed = {seed: _fpr(candidate[seed], ids) for seed in seeds}
    reference_observed = {seed: _fpr(references[seed], ids) for seed in seeds}
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(images), size=len(images))
        sampled_ids = [sid for index in sampled for sid in groups[images[int(index)]]]
        draws[iteration] = float(np.mean([
            _fpr(references[seed], sampled_ids) - _fpr(candidate[seed], sampled_ids)
            for seed in seeds
        ]))
    candidate_mean = float(np.mean(list(candidate_observed.values())))
    reference_mean = float(np.mean(list(reference_observed.values())))
    return {
        "metric": f"{name}_fpr95_reduction",
        "pairs": len(ids),
        "unique_images": len(images),
        "candidate_by_seed": {str(k): v for k, v in candidate_observed.items()},
        "reference_by_seed": {str(k): v for k, v in reference_observed.items()},
        "candidate_seed_mean": candidate_mean,
        "reference_seed_mean": reference_mean,
        "gain": reference_mean - candidate_mean,
        "ci95": _percentile(draws),
        "one_sided_p": _pvalue(draws),
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise BootstrapError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ref-summary", required=True)
    parser.add_argument("--reference-ref-summary", required=True)
    parser.add_argument("--candidate-strict2031-summary", required=True)
    parser.add_argument("--reference-strict2031-summary", required=True)
    parser.add_argument("--strict1607-manifest", default=str(DEFAULT_STRICT1607))
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise BootstrapError("iterations must be positive")
    paths = {
        "candidate_ref": Path(args.candidate_ref_summary),
        "reference_ref": Path(args.reference_ref_summary),
        "candidate_strict2031": Path(args.candidate_strict2031_summary),
        "reference_strict2031": Path(args.reference_strict2031_summary),
        "strict1607_manifest": Path(args.strict1607_manifest),
    }
    ref_candidate, ref_reference = _load_ref(paths["candidate_ref"]), _load_ref(paths["reference_ref"])
    tn_candidate, tn_reference = _load_tn(paths["candidate_strict2031"]), _load_tn(paths["reference_strict2031"])
    strict_ids = sorted(next(iter(tn_candidate.values())))
    seed_sequence = np.random.SeedSequence(args.seed)
    ref_rng, strict2031_rng, strict1607_rng = [np.random.default_rng(child) for child in seed_sequence.spawn(3)]
    subset = _strict1607_ids(paths["strict1607_manifest"])
    payload = {
        "schema": SCHEMA,
        "bootstrap": {"iterations": args.iterations, "base_seed": args.seed, "generator": "PCG64", "cluster": "image_id", "same_draw_all_seeds": True},
        "inputs": {name: _record(path) for name, path in paths.items()},
        "test5": _bootstrap_ref_comparison(ref_candidate, ref_reference, iterations=args.iterations, rng=ref_rng),
        "strict2031": _bootstrap_tn_comparison(tn_candidate, tn_reference, strict_ids, iterations=args.iterations, rng=strict2031_rng, name="strict2031"),
        "strict1607": _bootstrap_tn_comparison(tn_candidate, tn_reference, sorted(subset), iterations=args.iterations, rng=strict1607_rng, name="strict1607"),
    }
    _write(Path(args.output).resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, BootstrapError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
