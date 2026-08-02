#!/usr/bin/env python3
"""Profile-aware, exact-edge pruning for repo-local Python dependencies."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

from tools.stageb_dependency_audit import (
    REPO_ROOT,
    DependencyAuditError,
    _import_targets,
)


PathLike = str | Path
PrunedEdge = tuple[PathLike, PathLike]


class ProfileDependencyAuditError(DependencyAuditError):
    """Raised when a dependency profile or its exact-edge prunes are invalid."""


def _relative_label(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _resolve_file(raw: PathLike, root: Path, *, role: str) -> Path:
    try:
        path = Path(raw)
    except TypeError as error:
        raise ProfileDependencyAuditError(f"{role} is not a path: {raw!r}") from error
    if not path.is_absolute():
        path = root / path
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ProfileDependencyAuditError(f"{role} is missing: {path}") from error
    if not path.is_file():
        raise ProfileDependencyAuditError(f"{role} is not a file: {path}")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ProfileDependencyAuditError(
            f"{role} escapes repository root {root}: {path}"
        ) from error
    return path


def _resolve_roots(
    entry_paths: Iterable[PathLike],
    include_paths: Iterable[PathLike],
    root: Path,
) -> tuple[Path, ...]:
    resolved = {
        _resolve_file(raw, root, role=role)
        for role, values in (
            ("Python dependency entry", entry_paths),
            ("Python dependency include", include_paths),
        )
        for raw in values
    }
    return tuple(sorted(resolved, key=lambda path: _relative_label(path, root)))


def _reachable_graph(
    roots: tuple[Path, ...],
    root: Path,
) -> dict[Path, tuple[Path, ...]]:
    """Build the unpruned graph in deterministic breadth-first order."""

    pending = deque(roots)
    queued = set(roots)
    graph: dict[Path, tuple[Path, ...]] = {}
    while pending:
        source = pending.popleft()
        if source in graph:
            continue
        try:
            targets = tuple(
                sorted(
                    _import_targets(source, root),
                    key=lambda path: _relative_label(path, root),
                )
            )
        except DependencyAuditError as error:
            raise ProfileDependencyAuditError(str(error)) from error
        graph[source] = targets
        for target in targets:
            if target not in graph and target not in queued:
                pending.append(target)
                queued.add(target)
    return graph


def _resolve_pruned_edges(
    raw_edges: Iterable[PrunedEdge],
    root: Path,
    graph: dict[Path, tuple[Path, ...]],
) -> frozenset[tuple[Path, Path]]:
    edges: set[tuple[Path, Path]] = set()
    for index, raw_edge in enumerate(raw_edges):
        if isinstance(raw_edge, (str, Path)):
            raise ProfileDependencyAuditError(
                f"pruned edge {index} must be a (source, target) pair"
            )
        try:
            source_raw, target_raw = raw_edge
        except (TypeError, ValueError) as error:
            raise ProfileDependencyAuditError(
                f"pruned edge {index} must be a (source, target) pair"
            ) from error
        source = _resolve_file(
            source_raw,
            root,
            role=f"pruned edge {index} source",
        )
        target = _resolve_file(
            target_raw,
            root,
            role=f"pruned edge {index} target",
        )
        if source not in graph:
            raise ProfileDependencyAuditError(
                "pruned edge source is not reachable from the dependency roots: "
                f"{_relative_label(source, root)}"
            )
        if target not in graph[source]:
            raise ProfileDependencyAuditError(
                "pruned edge is not a direct resolved local import: "
                f"{_relative_label(source, root)} -> "
                f"{_relative_label(target, root)}"
            )
        edges.add((source, target))
    return frozenset(edges)


def recursive_local_python_dependencies(
    entry_paths: Iterable[PathLike],
    *,
    repository_root: PathLike = REPO_ROOT,
    include_paths: Iterable[PathLike] = (),
    pruned_edges: Iterable[PrunedEdge] = (),
) -> list[Path]:
    """Return a sorted local-Python closure with only declared edges removed.

    Every prune is validated against the complete unpruned graph. A target omitted
    from one source remains in the closure when another reachable source imports it.
    """

    try:
        root = Path(repository_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ProfileDependencyAuditError(
            f"repository root is missing: {repository_root}"
        ) from error
    if not root.is_dir():
        raise ProfileDependencyAuditError(f"repository root is not a directory: {root}")

    roots = _resolve_roots(entry_paths, include_paths, root)
    graph = _reachable_graph(roots, root)
    excluded_edges = _resolve_pruned_edges(pruned_edges, root, graph)

    pending = deque(roots)
    queued = set(roots)
    visited: set[Path] = set()
    while pending:
        source = pending.popleft()
        if source in visited:
            continue
        visited.add(source)
        for target in graph[source]:
            if (source, target) in excluded_edges:
                continue
            if target not in visited and target not in queued:
                pending.append(target)
                queued.add(target)

    unreachable_sources = sorted(
        {source for source, _ in excluded_edges if source not in visited},
        key=lambda path: _relative_label(path, root),
    )
    if unreachable_sources:
        rendered = ", ".join(
            _relative_label(path, root) for path in unreachable_sources
        )
        raise ProfileDependencyAuditError(
            "pruned edge source is unreachable after applying the dependency "
            f"profile: {rendered}"
        )
    return sorted(visited, key=lambda path: _relative_label(path, root))


__all__ = [
    "ProfileDependencyAuditError",
    "PrunedEdge",
    "recursive_local_python_dependencies",
]
