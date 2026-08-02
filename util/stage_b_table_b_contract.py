"""Fail-closed contract for weak-scope Table-B confidence supervision."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from util.path_compat import remap_legacy_path


REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_B_AUDIT_SCHEMA = "stage-b-paper-table-b-equal-exposure-v1"
TABLE_B_MATCHED_AUDIT_SCHEMA = "stage-b-paper-c2-parent-matched-tn-v1"
TABLE_B_PAIR_SCHEMA = "stage-b-paper-table-b-scope-preserving-pair-v1"
TABLE_B_SCOPE_BY_ID = {
    "D1": "unverified_all_negative",
    "D2": "traceable_counterfactual_edit",
    "D3": "proposal_covered_verified",
    "D2m": "traceable_counterfactual_edit",
    "D3m": "proposal_covered_verified",
}
TABLE_B_AUDIT_SCOPE_BY_ID = {
    "D1": "unverified all-negative",
    "D2": "traceable counterfactual edit only",
    "D3": "target plus cached proposals (proposal-covered)",
}
TABLE_B_STANDARD_IDS = frozenset(TABLE_B_AUDIT_SCOPE_BY_ID)
TABLE_B_MATCHED_IDS = frozenset({"D2m", "D3m"})
TABLE_B_EXPECTED_TRAIN_ROWS = {
    "D1": 14_196,
    "D2": 14_196,
    "D3": 14_196,
    "D2m": 7_074,
    "D3m": 7_074,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIRED_SOURCES = {"sam3_tn_pair", "sam3_paired_tn", "sam3_and_tn"}


class TableBContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class TableBConfidenceContract:
    table_b_id: str
    scope: str
    scope_allowlist: tuple[str, ...]
    audit_path: Path
    audit_sha256: str
    train_path: Path
    train_sha256: str
    allow_single_edit_token_provenance: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TableBContractError(f"{field} must be a non-empty path")
    path = remap_legacy_path(value, repo_root=REPO_ROOT)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as error:
        raise TableBContractError(f"{field} does not exist: {path}") from error


def _exact_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TableBContractError(f"{field} must be an exact boolean")
    return value


def _zero_mapping(value: Any, *, field: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise TableBContractError(f"Table-B audit {field} must be a non-empty mapping")
    bad = {str(key): count for key, count in value.items() if count != 0}
    if bad:
        raise TableBContractError(f"Table-B audit {field} is nonzero: {bad}")


@lru_cache(maxsize=16)
def _validate_audit_cached(
    audit_path_text: str,
    expected_sha256: str,
    table_b_id: str,
    scope: str,
) -> tuple[str, str]:
    audit_path = Path(audit_path_text)
    observed_sha256 = sha256_file(audit_path)
    if observed_sha256 != expected_sha256:
        raise TableBContractError(
            "Table-B audit SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TableBContractError(f"invalid Table-B audit {audit_path}: {error}") from error
    if not isinstance(audit, Mapping):
        raise TableBContractError("Table-B audit root must be an object")
    invariants = audit.get("invariants")
    if not isinstance(invariants, Mapping):
        raise TableBContractError("Table-B audit is missing invariants")
    if table_b_id in TABLE_B_STANDARD_IDS:
        if audit.get("schema") != TABLE_B_AUDIT_SCHEMA:
            raise TableBContractError("Table-B equal-exposure audit schema mismatch")
        if audit.get("kind") != "completed_table_b_data_matrix":
            raise TableBContractError("Table-B audit is not a completed data matrix")
        if audit.get("target_train_rows_per_tn_source") != 14_196:
            raise TableBContractError(
                "Table-B audit target row count is not sealed at 14196"
            )
        for key in (
            "equal_train_rows_D1_D3",
            "scope_preserved_without_global_upgrade",
        ):
            if invariants.get(key) is not True:
                raise TableBContractError(
                    f"Table-B audit invariant {key!r} did not pass"
                )
        _zero_mapping(
            invariants.get("strict_union_overlap_D1_D3"),
            field="strict_union_overlap_D1_D3",
        )
        _zero_mapping(
            invariants.get("train_calibration_image_overlap"),
            field="train_calibration_image_overlap",
        )
        _zero_mapping(
            invariants.get("runtime_global_verified_true_rows"),
            field="runtime_global_verified_true_rows",
        )
        source = audit.get("sources", {}).get(table_b_id)
        if not isinstance(source, Mapping):
            raise TableBContractError(
                f"Table-B audit has no source contract for {table_b_id}"
            )
        if source.get("verification_scope") != TABLE_B_AUDIT_SCOPE_BY_ID[table_b_id]:
            raise TableBContractError(
                f"Table-B audit source scope drifted for {table_b_id}"
            )
    elif table_b_id in TABLE_B_MATCHED_IDS:
        if audit.get("schema") != TABLE_B_MATCHED_AUDIT_SCHEMA:
            raise TableBContractError("Table-B matched-panel audit schema mismatch")
        if audit.get("kind") != "completed_c2_parent_matched_tn_panel":
            raise TableBContractError("Table-B matched audit is incomplete")
        runtime_contract = audit.get("runtime_contract")
        if not isinstance(runtime_contract, Mapping) or runtime_contract.get(
            "D2m_D3m_supported_by_current_v24"
        ) is not True:
            raise TableBContractError(
                "Table-B matched audit does not enable the v24 fail-closed runtime"
            )
        for split in ("train", "calibration"):
            split_invariants = invariants.get(split)
            if not isinstance(split_invariants, Mapping) or not all(
                split_invariants.get(key) is True
                for key in (
                    "aligned_pair_ids",
                    "aligned_parent_key_sha256",
                    "equal_rows",
                    "positive_phrase_normalized_match",
                    "unique_pair_ids",
                    "unique_parent_keys",
                )
            ):
                raise TableBContractError(
                    f"Table-B matched audit {split} invariants are incomplete"
                )
            if split_invariants.get("runtime_global_verified_true_rows") != 0:
                raise TableBContractError(
                    f"Table-B matched audit {split} upgraded global TN labels"
                )
        for key in (
            "strict_union_image_overlap",
            "train_calibration_image_overlap",
            "train_calibration_pair_id_overlap",
        ):
            if invariants.get(key) != 0:
                raise TableBContractError(
                    f"Table-B matched audit invariant {key!r} is nonzero"
                )
        scope_contract = audit.get("scope_contract", {}).get(table_b_id)
        if not isinstance(scope_contract, Mapping):
            raise TableBContractError(
                f"Table-B matched audit lacks scope contract for {table_b_id}"
            )
        if scope_contract.get("tn_scope") != scope:
            raise TableBContractError(
                f"Table-B matched audit scope drifted for {table_b_id}"
            )
        if scope_contract.get("global_tn_verified") is not False:
            raise TableBContractError(
                "Table-B matched audit must retain global_tn_verified=false"
            )
    else:
        raise TableBContractError(f"unsupported Table-B ID {table_b_id!r}")
    if TABLE_B_SCOPE_BY_ID[table_b_id] != scope:
        raise TableBContractError(
            f"Table-B runtime scope {scope!r} does not match ID {table_b_id}"
        )

    output_key = f"{table_b_id.lower()}_train"
    output = audit.get("outputs", {}).get(output_key)
    if not isinstance(output, Mapping):
        raise TableBContractError(f"Table-B audit is missing output {output_key}")
    expected_rows = TABLE_B_EXPECTED_TRAIN_ROWS[table_b_id]
    if output.get("rows") != expected_rows:
        raise TableBContractError(f"Table-B output {output_key} row count drifted")
    train_path = _resolve_path(output.get("path"), field=f"outputs.{output_key}.path")
    train_sha256 = output.get("sha256")
    if not isinstance(train_sha256, str) or not _SHA256_RE.fullmatch(train_sha256):
        raise TableBContractError(f"Table-B output {output_key} has invalid SHA-256")
    observed_train_sha256 = sha256_file(train_path)
    if observed_train_sha256 != train_sha256:
        raise TableBContractError(
            f"Table-B output {output_key} SHA-256 mismatch: "
            f"expected {train_sha256}, observed {observed_train_sha256}"
        )
    return str(train_path), train_sha256


def table_b_contract_from_args(args: Any) -> TableBConfidenceContract | None:
    enabled_raw = getattr(
        args, "stage_b_v19_allow_scope_labeled_tn_ablation", False
    )
    enabled = _exact_bool(
        enabled_raw, field="stage_b_v19_allow_scope_labeled_tn_ablation"
    )
    if not enabled:
        return None

    data_driven_confidence = bool(
        getattr(args, "stage_b_data_driven_score", False)
    ) and str(
        getattr(args, "stage_b_data_driven_train_mode", "") or ""
    ).strip().lower() == "confidence_pair"
    if data_driven_confidence:
        if not bool(
            getattr(args, "stage_b_data_driven_category_complete", False)
        ):
            raise TableBContractError(
                "data-driven confidence Table-B binding requires the DD1 "
                "category-complete phase"
            )
    else:
        if getattr(args, "stage_b_v14_global_tn_all_candidates", None) is not True:
            raise TableBContractError(
                "enabled scope-labeled confidence supervision requires "
                "stage_b_v14_global_tn_all_candidates=True"
            )
        score_ownership = str(
            getattr(args, "stage_b_v22_score_ownership", "") or ""
        ).strip()
        if not score_ownership:
            for field in (
                "stage_b_v15_decoupled_confidence",
                "stage_b_v19_explicit_confidence_output_contract",
            ):
                if getattr(args, field, None) is not True:
                    raise TableBContractError(
                        "enabled v19 scope-labeled confidence supervision "
                        f"requires {field}=True"
                    )
            if getattr(args, "stage_b_v16_confidence_output_mode", None) != "base_plus_gate":
                raise TableBContractError(
                    "enabled v19 scope-labeled confidence supervision requires "
                    "the base_plus_gate output contract"
                )

    allow_token_provenance = _exact_bool(
        getattr(
            args,
            "stage_b_v19_table_b_allow_single_edit_token_provenance",
            False,
        ),
        field="stage_b_v19_table_b_allow_single_edit_token_provenance",
    )

    table_b_id = getattr(args, "stage_b_v19_table_b_id", None)
    if table_b_id not in TABLE_B_SCOPE_BY_ID:
        raise TableBContractError(
            "enabled Table-B confidence supervision requires exact "
            "stage_b_v19_table_b_id in the sealed D1/D2/D3 or D2m/D3m blocks"
        )
    expected_scope = TABLE_B_SCOPE_BY_ID[table_b_id]
    raw_allowlist = getattr(args, "stage_b_v19_table_b_scope_allowlist", None)
    if not isinstance(raw_allowlist, (list, tuple)) or any(
        not isinstance(value, str) for value in raw_allowlist
    ):
        raise TableBContractError(
            "stage_b_v19_table_b_scope_allowlist must be a string list/tuple"
        )
    scope_allowlist = tuple(raw_allowlist)
    if scope_allowlist != (expected_scope,):
        raise TableBContractError(
            f"Table-B {table_b_id} scope allowlist must be exactly "
            f"[{expected_scope!r}], got {list(scope_allowlist)!r}"
        )

    expected_sha256 = getattr(args, "stage_b_v19_table_b_audit_sha256", None)
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise TableBContractError(
            "stage_b_v19_table_b_audit_sha256 must be exactly 64 lowercase hex characters"
        )
    audit_path = _resolve_path(
        getattr(args, "stage_b_v19_table_b_audit", None),
        field="stage_b_v19_table_b_audit",
    )
    train_path_text, train_sha256 = _validate_audit_cached(
        str(audit_path), expected_sha256, table_b_id, expected_scope
    )
    return TableBConfidenceContract(
        table_b_id=table_b_id,
        scope=expected_scope,
        scope_allowlist=scope_allowlist,
        audit_path=audit_path,
        audit_sha256=expected_sha256,
        train_path=Path(train_path_text),
        train_sha256=train_sha256,
        allow_single_edit_token_provenance=allow_token_provenance,
    )


def validate_table_b_dataset_binding(
    args: Any, datasetinfo: Mapping[str, Any]
) -> TableBConfidenceContract | None:
    """Validate one dataset entry and return its TN contract when applicable."""
    contract = table_b_contract_from_args(args)
    if contract is None:
        return None

    source = str(datasetinfo.get("source", "")).strip().lower()
    declared_id = datasetinfo.get("paper_table_b_id")
    has_binding = declared_id is not None or any(
        key in datasetinfo for key in ("paper_tn_scope", "paper_contract_audit")
    )
    if not has_binding:
        if source in _PAIRED_SOURCES:
            raise TableBContractError(
                "enabled Table-B config received a paired TN dataset without "
                "paper_table_b_id/paper_tn_scope/paper_contract_audit"
            )
        return None
    if source not in _PAIRED_SOURCES:
        raise TableBContractError("Table-B binding is allowed only on a paired TN source")
    if declared_id != contract.table_b_id:
        raise TableBContractError(
            f"dataset Table-B ID {declared_id!r} does not match config "
            f"{contract.table_b_id!r}"
        )
    if datasetinfo.get("paper_tn_scope") != contract.scope:
        raise TableBContractError(
            f"dataset Table-B scope {datasetinfo.get('paper_tn_scope')!r} does not "
            f"match {contract.scope!r}"
        )
    dataset_audit = _resolve_path(
        datasetinfo.get("paper_contract_audit"), field="paper_contract_audit"
    )
    if dataset_audit != contract.audit_path:
        raise TableBContractError(
            f"dataset audit path {dataset_audit} does not match config {contract.audit_path}"
        )
    if datasetinfo.get("require_global_tn_verified") is not False:
        raise TableBContractError(
            "scope-labeled Table-B data must keep require_global_tn_verified=false"
        )
    if datasetinfo.get("require_single_edit_token_provenance") is not (
        contract.allow_single_edit_token_provenance
    ):
        raise TableBContractError(
            "scope-labeled dataset token-provenance mode does not match its "
            "config-bound authorization"
        )
    annotation = _resolve_path(datasetinfo.get("anno"), field="dataset annotation")
    if annotation != contract.train_path:
        raise TableBContractError(
            f"dataset annotation {annotation} is not the audited "
            f"{contract.table_b_id} train output {contract.train_path}"
        )
    observed_sha256 = sha256_file(annotation)
    if observed_sha256 != contract.train_sha256:
        raise TableBContractError(
            f"dataset annotation hash drift: expected {contract.train_sha256}, "
            f"observed {observed_sha256}"
        )
    return contract


def _target_scalar_bool(target: Mapping[str, Any], key: str, *, index: int) -> bool:
    value = target.get(key)
    if not (
        torch.is_tensor(value)
        and value.dtype == torch.bool
        and value.numel() == 1
    ):
        raise TableBContractError(
            f"Table-B target {index} requires scalar bool tensor {key!r}"
        )
    return bool(value.reshape(-1)[0].item())


def build_confidence_ablation_eligible(
    args: Any,
    targets: Sequence[Mapping[str, Any]],
    paired_tn: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Create the distinct weak-scope confidence mask after per-batch checks."""
    contract = table_b_contract_from_args(args)
    if contract is None:
        return None
    if not torch.is_tensor(paired_tn) or paired_tn.dtype != torch.bool:
        raise TableBContractError("paired_tn must be a boolean tensor")
    paired_tn = paired_tn.to(device=device, dtype=torch.bool).reshape(-1)
    if paired_tn.numel() != len(targets):
        raise TableBContractError(
            f"paired_tn has {paired_tn.numel()} values for {len(targets)} targets"
        )

    eligible = torch.zeros(len(targets), dtype=torch.bool, device=device)
    for index, (target, is_paired) in enumerate(zip(targets, paired_tn.tolist())):
        if not is_paired:
            if target.get("table_b_id") is not None:
                raise TableBContractError(
                    f"non-paired target {index} unexpectedly carries a Table-B ID"
                )
            continue
        if target.get("table_b_id") != contract.table_b_id:
            raise TableBContractError(
                f"paired target {index} has unknown/mixed Table-B ID "
                f"{target.get('table_b_id')!r}; expected {contract.table_b_id!r}"
            )
        if target.get("tn_scope") not in contract.scope_allowlist:
            raise TableBContractError(
                f"paired target {index} scope {target.get('tn_scope')!r} is not in "
                f"the exact allowlist {list(contract.scope_allowlist)!r}"
            )
        if target.get("table_b_audit_sha256") != contract.audit_sha256:
            raise TableBContractError(
                f"paired target {index} Table-B audit SHA-256 binding mismatch"
            )
        if _target_scalar_bool(target, "global_tn_verified", index=index):
            raise TableBContractError(
                f"paired Table-B target {index} must retain global_tn_verified=false; "
                "weak-scope eligibility is a distinct contract"
            )
        eligible[index] = True
    return eligible


__all__ = [
    "TABLE_B_AUDIT_SCHEMA",
    "TABLE_B_PAIR_SCHEMA",
    "TABLE_B_SCOPE_BY_ID",
    "TableBConfidenceContract",
    "TableBContractError",
    "build_confidence_ablation_eligible",
    "table_b_contract_from_args",
    "validate_table_b_dataset_binding",
]
