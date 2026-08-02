#!/usr/bin/env python3
"""Deterministic local-Python dependency discovery for Stage-B audit hashes."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


class DependencyAuditError(RuntimeError):
    pass


def _module_parts(path: Path, root: Path) -> tuple[str, ...]:
    relative = path.resolve().relative_to(root.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return tuple(parts)


def _module_path(parts: Sequence[str], root: Path) -> Path | None:
    if not parts:
        return None
    file_candidate = root.joinpath(*parts).with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate.resolve()
    package_candidate = root.joinpath(*parts, "__init__.py")
    if package_candidate.is_file():
        return package_candidate.resolve()
    return None


def _import_targets(path: Path, root: Path) -> set[Path]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise DependencyAuditError(f"could not parse Python dependency {path}: {error}") from error
    current_parts = _module_parts(path, root)
    current_is_package = path.name == "__init__.py"
    current_package = current_parts if current_is_package else current_parts[:-1]
    targets: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[tuple[str, ...]] = []
        if isinstance(node, ast.Import):
            modules.extend(tuple(alias.name.split(".")) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(current_package) - (int(node.level) - 1)
                if keep < 0:
                    continue
                base = current_package[:keep]
            else:
                base = ()
            if node.module:
                base = base + tuple(node.module.split("."))
                modules.append(base)
            for alias in node.names:
                if alias.name != "*":
                    modules.append(base + tuple(alias.name.split(".")))
        for parts in modules:
            candidate = _module_path(parts, root)
            if candidate is not None:
                targets.add(candidate)
    return targets


def local_python_dependency_paths(
    entries: Iterable[str | Path],
    *,
    root: str | Path = REPO_ROOT,
    include: Iterable[str | Path] = (),
) -> list[Path]:
    """Return the sorted recursive closure of repo-local Python imports."""

    root_path = Path(root).resolve()
    pending: set[Path] = set()
    for raw in tuple(entries) + tuple(include):
        path = Path(raw)
        if not path.is_absolute():
            path = root_path / path
        path = path.resolve()
        if not path.is_file():
            raise DependencyAuditError(f"Python dependency entry is missing: {path}")
        try:
            path.relative_to(root_path)
        except ValueError as error:
            raise DependencyAuditError(f"dependency escapes root {root_path}: {path}") from error
        pending.add(path)

    visited: set[Path] = set()
    while pending:
        path = min(pending, key=lambda value: value.relative_to(root_path).as_posix())
        pending.remove(path)
        if path in visited:
            continue
        visited.add(path)
        pending.update(_import_targets(path, root_path).difference(visited))
    return sorted(visited, key=lambda value: value.relative_to(root_path).as_posix())


def config_import_chain(
    config: str | Path,
    *,
    root: str | Path = REPO_ROOT,
) -> list[Path]:
    root_path = Path(root).resolve()
    paths = local_python_dependency_paths([config], root=root_path)
    config_root = (root_path / "config").resolve()
    return [path for path in paths if path == config_root or config_root in path.parents]


__all__ = [
    "DependencyAuditError",
    "config_import_chain",
    "local_python_dependency_paths",
]
