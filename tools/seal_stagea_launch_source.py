#!/usr/bin/env python3
"""Seal the exact tracked source state observed by a running Stage-A job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "pivot.stagea.launch_source_manifest/v1"
MANIFEST_NAME = "stagea_launch_source_manifest.json"
PATCH_NAME = "stagea_launch_worktree.patch"
REPO_ROOT = Path(__file__).resolve().parents[1]
_INFO_HEADER = re.compile(
    r"^INFO\s+(?P<launched>[^|]+)\| git:\n"
    r"\s+sha: (?P<sha>[0-9a-f]{40}), status: (?P<status>[^,]+), branch: (?P<branch>[^\n]+)\n"
    r"\nINFO\s+[^|]+\| Command: (?P<command>[^\n]+)",
    re.MULTILINE,
)
_RUNTIME_SOURCES = (
    "main.py",
    "engine.py",
    "config/ablations/cfg_stagea_b58_trunk_patch0006_realign_20260814.py",
    "config/datasets_patch_stage_a_lvis_coco2017_local.json",
    "models/GroundingDINO/groundingdino.py",
)


class LaunchSourceSealError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def _parse_info(info_path: Path) -> dict[str, str]:
    text = info_path.read_text(encoding="utf-8")
    match = _INFO_HEADER.search(text)
    if match is None:
        raise LaunchSourceSealError(f"could not parse Stage-A launch header: {info_path}")
    return {key: value.strip() for key, value in match.groupdict().items()}


def _parse_launch_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError as error:
        raise LaunchSourceSealError(f"invalid Stage-A launch timestamp: {value}") from error
    return parsed.astimezone()


def _dirty_entries(launch_sha: str, launch_ns: int) -> list[dict[str, Any]]:
    raw = str(_git("diff", "--name-status", "--find-renames", launch_sha))
    entries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        fields = line.split("\t")
        status = fields[0]
        paths = fields[1:]
        if not paths:
            raise LaunchSourceSealError(f"unparseable git diff name-status line: {line!r}")
        entry: dict[str, Any] = {"status": status, "paths": paths}
        destination = REPO_ROOT / paths[-1]
        if destination.exists():
            stat = destination.stat()
            if stat.st_mtime_ns > launch_ns:
                raise LaunchSourceSealError(
                    f"tracked dirty file changed after Stage-A launch: {destination}"
                )
            entry["file"] = _file_record(destination)
            entry["mtime_not_after_launch"] = True
        else:
            entry["deleted"] = True
        entries.append(entry)
    return entries


def build_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=True)
    info_path = output_dir / "info.txt"
    launch_files = (
        info_path,
        output_dir / "config_cfg.py",
        output_dir / "config_args_raw.json",
        output_dir / "config_args_all.json",
    )
    for path in launch_files:
        if not path.is_file():
            raise LaunchSourceSealError(f"missing Stage-A launch artifact: {path}")

    header = _parse_info(info_path)
    launched_at = _parse_launch_timestamp(header["launched"])
    launch_ns = int(launched_at.timestamp() * 1_000_000_000)
    current_sha = str(_git("rev-parse", "HEAD"))
    if current_sha != header["sha"]:
        raise LaunchSourceSealError(
            f"repository HEAD moved since Stage-A launch: {current_sha} != {header['sha']}"
        )

    allowed_sealer = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    untracked = [
        path
        for path in str(_git("ls-files", "--others", "--exclude-standard")).splitlines()
        if path and path != allowed_sealer
    ]
    if untracked:
        raise LaunchSourceSealError(
            "cannot prove whether untracked files existed at launch: " + ", ".join(untracked)
        )

    patch_path = output_dir / PATCH_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if patch_path.exists() or manifest_path.exists():
        raise LaunchSourceSealError(
            f"refusing to overwrite an existing Stage-A source seal in {output_dir}"
        )
    patch_bytes = bytes(_git("diff", "--binary", header["sha"], binary=True))
    temporary_patch = Path(str(patch_path) + ".tmp")
    temporary_manifest = Path(str(manifest_path) + ".tmp")
    try:
        temporary_patch.write_bytes(patch_bytes)
        os.replace(temporary_patch, patch_path)
        runtime_sources = []
        for relative in _RUNTIME_SOURCES:
            source = REPO_ROOT / relative
            record = _file_record(source)
            if record["mtime_ns"] > launch_ns:
                raise LaunchSourceSealError(
                    f"runtime source changed after Stage-A launch: {source}"
                )
            record["relative_path"] = relative
            record["mtime_not_after_launch"] = True
            runtime_sources.append(record)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "launch": {
                "launched_at": launched_at.isoformat(),
                "git_sha": header["sha"],
                "git_status": header["status"],
                "git_branch": header["branch"],
                "command": header["command"],
                "output_dir": str(output_dir),
            },
            "launch_artifacts": [_file_record(path) for path in launch_files],
            "tracked_worktree_patch": _file_record(patch_path),
            "tracked_dirty_entries": _dirty_entries(header["sha"], launch_ns),
            "runtime_sources": runtime_sources,
            "untracked_source_contract": {
                "observed_untracked_paths": [],
                "sealer_path_excluded_because_added_for_post_launch_audit": allowed_sealer,
            },
            "invariants": {
                "head_matches_logged_launch_sha": True,
                "tracked_dirty_mtimes_not_after_launch": True,
                "runtime_source_mtimes_not_after_launch": True,
                "tracked_diff_is_reconstructible_from_launch_sha": True,
            },
        }
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_patch.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return verify_manifest(manifest_path)


def verify_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchSourceSealError(f"could not read launch source manifest: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise LaunchSourceSealError("launch source manifest schema is invalid")
    if payload.get("manifest_sha256") != _canonical_sha256(payload):
        raise LaunchSourceSealError("launch source manifest self-hash mismatch")
    for record in [
        *payload.get("launch_artifacts", []),
        payload.get("tracked_worktree_patch", {}),
    ]:
        if not isinstance(record, Mapping) or not record.get("path"):
            raise LaunchSourceSealError("launch source manifest has an invalid file record")
        current = _file_record(Path(str(record["path"])))
        for key in ("size_bytes", "sha256"):
            if current[key] != record.get(key):
                raise LaunchSourceSealError(
                    f"sealed launch artifact drifted ({key}): {record['path']}"
                )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        payload = build_manifest(args.output_dir)
        manifest = args.output_dir.expanduser().resolve() / MANIFEST_NAME
    else:
        payload = verify_manifest(args.manifest)
        manifest = args.manifest.expanduser().resolve(strict=True)
    print(
        json.dumps(
            {
                "status": "verified",
                "schema": SCHEMA,
                "manifest": _file_record(manifest),
                "manifest_sha256": payload["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
