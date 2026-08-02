#!/usr/bin/env python3
"""Aggregate the six immutable three-seed Table-A evaluation instances."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_table_a_evaluations as table_a  # noqa: E402


SCHEMA = "pivot.stageb.table_a_three_seed_aggregate/v1"
FINAL_QUEUE_BINDING_SCHEMA = "pivot.stageb.table_a_final_queue_binding/v1"
FORMAL_SEEDS = (17, 42, 73)
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 170718


class TableAAggregationError(RuntimeError):
    """Raised when the immutable Table-A aggregation contract cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TableAAggregationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TableAAggregationError(f"JSON artifact is not an object: {path}")
    return value


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_volatile(item)
            for key, item in value.items()
            if key not in {"verified_at_utc", "validated_at_utc"}
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def canonical_report_path(profile: str) -> Path:
    if profile not in table_a.PROFILES:
        raise TableAAggregationError(f"unsupported Table-A profile: {profile!r}")
    return (
        REPO_ROOT
        / "outputs/paper_cvpr_v1/table_a/aggregates"
        / profile
        / "table_a_three_seed.json"
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise TableAAggregationError(f"artifact is not a file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _final_queue_binding() -> dict[str, Any]:
    from tools import run_stageb_table_a_g0c_queues as queues

    try:
        queue_dir = queues.DEFAULT_FINAL_QUEUE_DIR.resolve(strict=True)
        verification = queues.verify_queue(queue_dir)
        queue = queues.load_queue(queue_dir)
    except (
        queues.G0cQueueError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise TableAAggregationError(
            f"canonical Table-A final queue replay failed: {exc}"
        ) from exc
    verified_items = verification.get("verified_items")
    if (
        verification.get("status") != "passed"
        or verification.get("queue_status") != "completed"
        or verification.get("queue_kind") != queues.FINAL_KIND
        or verification.get("ordered_run_ids") != list(queues.FINAL_RUN_IDS)
        or not isinstance(verified_items, list)
        or len(verified_items) != len(queues.FINAL_RUN_IDS)
        or queue.get("status") != "completed"
        or queue.get("plan_sha256") != verification.get("plan_sha256")
        or queue.get("plan", {}).get("queue_id") != verification.get("queue_id")
    ):
        raise TableAAggregationError(
            "canonical Table-A final queue is not exactly completed and verified"
        )
    items = []
    for run_id, evidence in zip(queues.FINAL_RUN_IDS, verified_items):
        native = evidence.get("native_completion") if isinstance(evidence, Mapping) else None
        if (
            not isinstance(native, Mapping)
            or evidence.get("run_id") != run_id
            or native.get("run_id") != run_id
            or native.get("queue_kind") != queues.FINAL_KIND
            or native.get("evaluation_profile") != table_a.FINAL_PROFILE
            or not isinstance(native.get("final_gate"), Mapping)
            or not isinstance(native.get("final_consumption"), Mapping)
        ):
            raise TableAAggregationError(
                f"Table-A final queue completion evidence is incomplete: {run_id}"
            )
        items.append(
            {
                "run_id": run_id,
                "evaluation_kind": native.get("evaluation_kind"),
                "seed": int(native.get("seed", -1)),
                "instance_sha256": native.get("instance_sha256"),
                "launch_manifest": copy.deepcopy(native.get("launch_manifest")),
                "postflight": copy.deepcopy(native.get("postflight")),
                "final_gate": copy.deepcopy(native.get("final_gate")),
                "final_consumption": copy.deepcopy(native.get("final_consumption")),
            }
        )
    binding = {
        "schema": FINAL_QUEUE_BINDING_SCHEMA,
        "queue_kind": queues.FINAL_KIND,
        "queue_id": verification["queue_id"],
        "plan_sha256": verification["plan_sha256"],
        "ordered_run_ids": list(queues.FINAL_RUN_IDS),
        "queue_manifest": _artifact_record(queue_dir / "queue.json"),
        "items": items,
        "shared_gpu_lease_released": True,
        "single_use_consumptions_verified": True,
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TableAAggregationError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, Mapping):
                raise TableAAggregationError(
                    f"{path}:{line_number}: record is not an object"
                )
            rows.append(row)
    return rows


def _load_instance(kind: str, profile: str, seed: int) -> dict[str, Any]:
    expected_root = table_a.canonical_output_dir(kind, profile, seed)
    try:
        root = expected_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TableAAggregationError(
            f"required canonical evaluation instance is incomplete: "
            f"{kind}/{profile}/seed{seed} at {expected_root}"
        ) from exc
    launch_path = (root / "launch_manifest.json").resolve(strict=True)
    postflight_path = (root / "postflight.json").resolve(strict=True)
    launch = _read_json(launch_path)
    persisted = _read_json(postflight_path)
    if (
        launch.get("status") != "completed"
        or launch.get("kind") != kind
        or launch.get("profile") != profile
        or int(launch.get("instance", {}).get("seed", -1)) != seed
    ):
        raise TableAAggregationError(
            f"Table-A launch identity/status mismatch: {kind}/{seed}"
        )
    try:
        replayed = table_a.postflight(launch)
    except Exception as exc:
        raise TableAAggregationError(
            f"Table-A postflight replay failed for {kind}/{seed}: {exc}"
        ) from exc
    if _strip_volatile(replayed) != _strip_volatile(persisted):
        raise TableAAggregationError(
            f"persisted postflight differs from replay for {kind}/{seed}"
        )
    instance = launch.get("instance")
    if not isinstance(instance, Mapping):
        raise TableAAggregationError(f"missing immutable instance: {kind}/{seed}")
    return {
        "kind": kind,
        "seed": seed,
        "root": root,
        "launch": launch,
        "postflight": persisted,
        "instance": dict(instance),
        "provenance": {
            "root": str(root),
            "launch_manifest": {
                "path": str(launch_path),
                "sha256": _sha256(launch_path),
            },
            "postflight": {
                "path": str(postflight_path),
                "sha256": _sha256(postflight_path),
            },
            "instance_id": instance.get("instance_id"),
            "instance_sha256": instance.get("instance_sha256"),
            "checkpoint_sha256": instance.get("checkpoint_sha256"),
            "training_queue_id": instance.get("training_queue_id"),
            "training_queue_plan_sha256": instance.get(
                "training_queue_plan_sha256"
            ),
        },
    }


def _put(metrics: dict[str, float], path: str, value: Any) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TableAAggregationError(f"metric {path} is not numeric") from exc
    if not math.isfinite(numeric):
        raise TableAAggregationError(f"metric {path} is non-finite")
    metrics[path] = numeric


def _candidate_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ref = summary.get("ref")
    if not isinstance(ref, Mapping):
        raise TableAAggregationError("candidate summary has no Ref metrics")
    for split, split_summary in ref.items():
        rows = split_summary.get("table_a_rows")
        if not isinstance(rows, Mapping) or set(rows) != {"G1", "G2", "G3", "G4", "G5"}:
            raise TableAAggregationError(f"candidate {split} lacks G1-G5")
        for row_id, row in rows.items():
            prefix = f"ref/{split}/{row_id}"
            for key in ("acc50", "mean_selected_iou", "top1_query_churn_vs_patch_only"):
                _put(metrics, f"{prefix}/{key}", row.get(key))
            if row_id == "G5":
                _put(
                    metrics,
                    f"{prefix}/top1_query_churn_vs_patch_admission_text_rank",
                    row.get("top1_query_churn_vs_patch_admission_text_rank"),
                )
            oracle = row.get("ranked_oracle")
            if not isinstance(oracle, Mapping) or set(oracle) != {"1", "5", "10", "50", "all"}:
                raise TableAAggregationError(f"candidate {split}/{row_id} oracle drifted")
            for topk, values in oracle.items():
                for key in ("recall_iou50", "mean_best_iou"):
                    _put(metrics, f"{prefix}/oracle@{topk}/{key}", values.get(key))
    tn = summary.get("tn_counterfactual")
    groups = tn.get("by_edit_taxonomy") if isinstance(tn, Mapping) else None
    if not isinstance(groups, Mapping):
        raise TableAAggregationError("candidate summary has no TN taxonomy metrics")
    all_groups = {"overall": tn.get("overall"), **dict(groups)}
    for taxonomy, group in all_groups.items():
        if not isinstance(group, Mapping):
            raise TableAAggregationError(f"candidate TN group {taxonomy} is invalid")
        _put(
            metrics,
            f"tn/{taxonomy}/candidate_admission_change_rate",
            group.get("candidate_admission_change_rate"),
        )
        for surface, values in group.get("surfaces", {}).items():
            for key in (
                "delta_max_logit_mean",
                "delta_max_logit_median",
                "top1_change_rate",
                "mean_absolute_rank_change",
                "invariant_rate_at_1e-8",
            ):
                _put(metrics, f"tn/{taxonomy}/{surface}/{key}", values.get(key))
    category = summary.get("category_intervention")
    if not isinstance(category, Mapping):
        raise TableAAggregationError("candidate summary has no category metrics")
    for key in (
        "top1_both_match_active_rate",
        "top1_query_change_rate",
        "mean_top1_box_iou_between_arms",
    ):
        _put(metrics, f"category/{key}", category.get(key))
    for topk, values in category.get("candidate_admission", {}).items():
        for key in (
            "matched_active_recall_iou50",
            "counterfactual_category_recall_iou50",
            "matched_active_mean_best_iou",
            "counterfactual_mean_best_iou",
        ):
            _put(metrics, f"category/admission@{topk}/{key}", values.get(key))
    return metrics


def _g0c_summary(instance: Mapping[str, Any], profile: str) -> tuple[dict[str, float], dict[str, Path]]:
    root = Path(instance["root"])
    section = root / (
        "validation_calibration" if profile == table_a.VALIDATION_PROFILE else "ref8_strict2031"
    )
    summary_path = (section / "summary.json").resolve(strict=True)
    summary = _read_json(summary_path)
    ref_rows = summary.get("refcoco")
    if not isinstance(ref_rows, list):
        raise TableAAggregationError("G0c summary has no Ref rows")
    metrics: dict[str, float] = {}
    records: dict[str, Path] = {}
    for row in ref_rows:
        if not isinstance(row, Mapping):
            raise TableAAggregationError("G0c Ref summary row is invalid")
        split = str(row.get("dataset", ""))
        for key in (
            "acc50",
            "acc50@5",
            "acc50@10",
            "acc50@50",
            "recall50@all_queries",
            "mean_iou",
            "mean_iou@5",
            "mean_iou@10",
            "mean_iou@50",
            "mean_best_iou@all_queries",
        ):
            _put(metrics, f"ref/{split}/G0c/{key}", row.get(key))
        records[split] = table_a._resolve_records_path(
            row.get("records_jsonl"), section_dir=section
        )
    tn_sections = [("calibration", summary, section)]
    if profile == table_a.FINAL_PROFILE:
        tn_sections = [("strict2031", summary, section)]
        supplemental_section = Path(instance["root"]) / "strict1607"
        supplemental = _read_json(
            (supplemental_section / "summary.json").resolve(strict=True)
        )
        tn_sections.append(("strict1607", supplemental, supplemental_section))
    for label, payload, _section in tn_sections:
        rows = payload.get("tn")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise TableAAggregationError(f"G0c {label} summary has no exact TN row")
    postflight = instance.get("postflight")
    artifacts = postflight.get("artifacts") if isinstance(postflight, Mapping) else None
    replay = (
        artifacts.get("tn_metrics_recomputed")
        if isinstance(artifacts, Mapping)
        else None
    )
    expected_labels = {label for label, _, _ in tn_sections}
    if not isinstance(replay, Mapping) or set(replay) != expected_labels:
        raise TableAAggregationError(
            "G0c postflight lacks the exact verified TN replay surface"
        )
    for label in sorted(expected_labels):
        values = replay[label]
        if not isinstance(values, Mapping):
            raise TableAAggregationError(f"G0c {label} TN replay is invalid")
        for key in table_a.G0C_TN_AGGREGATE_METRICS:
            if key not in values:
                raise TableAAggregationError(
                    f"G0c {label} TN replay lacks {key}"
                )
            _put(metrics, f"tn/{label}/G0c/{key}", values[key])
    return metrics, records


def _candidate_summary_and_records(
    instance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, list[Mapping[str, Any]]], Path, Path]:
    root = Path(instance["root"])
    summary_path = (root / "role_causal/role_causal.summary.json").resolve(strict=True)
    records_path = (root / "role_causal/role_causal.records.jsonl").resolve(strict=True)
    summary = _read_json(summary_path)
    grouped = table_a._load_candidate_record_groups(records_path)
    return summary, grouped, summary_path, records_path


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(FORMAL_SEEDS):
        raise TableAAggregationError("three-seed statistic received the wrong cardinality")
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values))
    return {
        "n": len(values),
        "mean": mean,
        "std_ddof1": std,
        "display": f"{mean:.6f} +/- {std:.6f}",
    }


def aggregate_seed_metrics(
    seed_metrics: Mapping[int, Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    if tuple(sorted(seed_metrics)) != FORMAL_SEEDS:
        raise TableAAggregationError("aggregate requires exact seeds 17/42/73")
    key_sets = {tuple(sorted(values)) for values in seed_metrics.values()}
    if len(key_sets) != 1:
        raise TableAAggregationError("metric surfaces differ across seeds")
    keys = next(iter(key_sets))
    return {
        key: _stats([float(seed_metrics[seed][key]) for seed in FORMAL_SEEDS])
        for key in keys
    }


def paired_image_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if int(iterations) != BOOTSTRAP_ITERATIONS:
        raise TableAAggregationError("formal bootstrap must use exactly 5000 iterations")
    clusters: dict[str, list[float]] = {}
    for row in rows:
        cluster = str(row.get("cluster_id", ""))
        delta = float(row.get("delta", math.nan))
        if not cluster or not math.isfinite(delta):
            raise TableAAggregationError("paired bootstrap row is invalid")
        clusters.setdefault(cluster, []).append(delta)
    if not clusters:
        raise TableAAggregationError("paired bootstrap has no image clusters")
    ordered = sorted(clusters)
    sums = np.asarray([sum(clusters[key]) for key in ordered], dtype=np.float64)
    counts = np.asarray([len(clusters[key]) for key in ordered], dtype=np.float64)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=np.float64)
    cluster_count = len(ordered)
    batch = 64
    for start in range(0, int(iterations), batch):
        width = min(batch, int(iterations) - start)
        indices = rng.integers(0, cluster_count, size=(width, cluster_count))
        draws[start : start + width] = sums[indices].sum(axis=1) / counts[
            indices
        ].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    return {
        "iterations": int(iterations),
        "rng": "numpy.PCG64",
        "seed": int(seed),
        "resample_unit": "image_id_cluster_with_all_training_seeds",
        "num_clusters": cluster_count,
        "num_paired_expressions": int(counts.sum()),
        "point_delta_acc50_g4_minus_g0c": point,
        "ci95_percentile": [float(lower), float(upper)],
    }


def _paired_rows(
    *,
    seed: int,
    split: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    g0c_path: Path,
) -> list[dict[str, Any]]:
    candidate_by_id: dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in candidate_by_id:
            raise TableAAggregationError(f"candidate sample IDs are not unique: {split}")
        candidate_by_id[sample_id] = row
    g0c_by_id: dict[str, Mapping[str, Any]] = {}
    for row in _load_jsonl(g0c_path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in g0c_by_id:
            raise TableAAggregationError(f"G0c sample IDs are not unique: {split}")
        g0c_by_id[sample_id] = row
    if set(candidate_by_id) != set(g0c_by_id):
        raise TableAAggregationError(f"G4/G0c records do not align exactly: {split}")
    paired = []
    for sample_id in sorted(candidate_by_id):
        candidate = candidate_by_id[sample_id]
        baseline = g0c_by_id[sample_id]
        selected_iou = float(
            candidate["routes"]["patch_admission_text_rank"]["selected_iou"]
        )
        top1_iou = float(baseline["top1_iou"])
        if not math.isfinite(selected_iou) or not math.isfinite(top1_iou):
            raise TableAAggregationError(f"non-finite paired IoU: {sample_id}")
        if int(candidate.get("image_id", -1)) != int(baseline.get("image_id", -2)):
            raise TableAAggregationError(f"paired image identity drifted: {sample_id}")
        paired.append(
            {
                "cluster_id": f"image{int(candidate['image_id'])}",
                "training_seed": int(seed),
                "delta": float(selected_iou >= 0.5) - float(top1_iou >= 0.5),
            }
        )
    return paired


def build_report(profile: str) -> dict[str, Any]:
    if profile not in table_a.PROFILES:
        raise TableAAggregationError(f"unsupported Table-A profile: {profile!r}")
    final_queue_before = (
        _final_queue_binding() if profile == table_a.FINAL_PROFILE else None
    )
    instances = {
        kind: {seed: _load_instance(kind, profile, seed) for seed in FORMAL_SEEDS}
        for kind in ("candidate", "g0c")
    }
    instance_ids = [
        value["instance"]["instance_sha256"]
        for by_seed in instances.values()
        for value in by_seed.values()
    ]
    if len(instance_ids) != 6 or len(set(instance_ids)) != 6:
        raise TableAAggregationError("six unique immutable instances are required")

    seed_metrics: dict[int, dict[str, float]] = {}
    bootstrap_rows: dict[str, list[dict[str, Any]]] = {}
    artifact_provenance: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        candidate = instances["candidate"][seed]
        g0c = instances["g0c"][seed]
        summary, grouped, candidate_summary_path, candidate_records_path = (
            _candidate_summary_and_records(candidate)
        )
        candidate_values = _candidate_metrics(summary)
        g0c_values, g0c_records = _g0c_summary(g0c, profile)
        overlap = set(candidate_values).intersection(g0c_values)
        if overlap:
            raise TableAAggregationError(f"candidate/G0c metric path collision: {overlap}")
        seed_metrics[seed] = {**candidate_values, **g0c_values}
        for split, path in g0c_records.items():
            candidate_rows = grouped.get(f"ref:{split}", [])
            bootstrap_rows.setdefault(split, []).extend(
                _paired_rows(
                    seed=seed,
                    split=split,
                    candidate_rows=candidate_rows,
                    g0c_path=path,
                )
            )
        artifact_provenance.extend(
            [
                {
                    "kind": "candidate",
                    "seed": seed,
                    **candidate["provenance"],
                    "summary_sha256": _sha256(candidate_summary_path),
                    "records_sha256": _sha256(candidate_records_path),
                },
                {
                    "kind": "g0c",
                    "seed": seed,
                    **g0c["provenance"],
                    "ref_record_sha256": {
                        split: _sha256(path) for split, path in sorted(g0c_records.items())
                    },
                },
            ]
        )
    final_queue_after = (
        _final_queue_binding() if profile == table_a.FINAL_PROFILE else None
    )
    if final_queue_before != final_queue_after:
        raise TableAAggregationError(
            "Table-A final queue changed while building the aggregate"
        )
    report = {
        "schema": SCHEMA,
        "status": "passed",
        "profile": profile,
        "formal_seeds": list(FORMAL_SEEDS),
        "statistics_contract": {
            "center": "arithmetic_mean_across_training_seeds",
            "dispersion": "sample_standard_deviation",
            "ddof": 1,
            "display": "mean +/- std_ddof1",
        },
        "seed_metrics": {
            str(seed): dict(sorted(seed_metrics[seed].items()))
            for seed in FORMAL_SEEDS
        },
        "metric_aggregates": aggregate_seed_metrics(seed_metrics),
        "paired_bootstrap_g4_minus_g0c": {
            split: paired_image_cluster_bootstrap(
                rows,
                iterations=BOOTSTRAP_ITERATIONS,
                seed=BOOTSTRAP_SEED
                + int(hashlib.sha256(split.encode("utf-8")).hexdigest()[:8], 16),
            )
            for split, rows in sorted(bootstrap_rows.items())
        },
        "provenance": {
            "instances": artifact_provenance,
            "no_rerun_contract": {
                "canonical_roots_only": True,
                "exactly_one_instance_per_kind_seed_profile": True,
                "postflights_replayed": True,
                "aggregator_spawns_evaluations": False,
                "canonical_final_queue_required": (
                    profile == table_a.FINAL_PROFILE
                ),
            },
        },
    }
    if final_queue_after is not None:
        report["provenance"]["final_queue"] = final_queue_after
    report["report_sha256"] = _canonical_sha256(report)
    return report


def verify_report(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    value = _read_json(path)
    payload = copy.deepcopy(value)
    expected_sha = str(payload.pop("report_sha256", ""))
    if expected_sha != _canonical_sha256(payload):
        raise TableAAggregationError("aggregate report self SHA-256 mismatch")
    profile = str(value.get("profile", ""))
    if path != canonical_report_path(profile).resolve(strict=True):
        raise TableAAggregationError("aggregate report path is not canonical")
    observed = build_report(profile)
    if value != observed:
        raise TableAAggregationError("aggregate report differs from full replay")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve(strict=False)
    if path.exists():
        raise FileExistsError(f"aggregate report must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("aggregate", "verify"))
    parser.add_argument("--profile", choices=table_a.PROFILES, required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    expected = canonical_report_path(args.profile).resolve(strict=False)
    output = Path(args.output).expanduser().resolve(strict=False) if args.output else expected
    if output != expected:
        parser.error("--output must be the canonical profile-specific report path")
    if args.mode == "verify":
        report = verify_report(output)
    else:
        report = build_report(args.profile)
        _write_new(output, report)
    print(json.dumps({"status": "passed", "path": str(output), "sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
