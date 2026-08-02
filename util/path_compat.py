"""Compatibility mapping for sealed artifacts created on the original host."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


PathLike = Union[str, os.PathLike[str]]

_LEGACY_REPO_ROOT = Path("/home/user/PIVOT")
_LEGACY_DATA_ROOT = Path("/home/user/datasets/pivot_data")
_LEGACY_COCO_ROOT = Path("/home/user/datasets/vision_benchmarks/COCO_2017")
_LEGACY_REFCOCO_ROOT = Path("/home/user/datasets/vision_benchmarks/RefCOCO")


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def default_data_root() -> Path:
    media_user = os.environ.get("MEDIA_USER", "haoyi")
    t9_root = os.environ.get("T9_ROOT", f"/media/{media_user}/T9")
    return Path(os.environ.get("DATA_ROOT", str(Path(t9_root) / "data"))).expanduser()


def remap_legacy_path(
    value: PathLike,
    *,
    repo_root: PathLike | None = None,
    data_root: PathLike | None = None,
) -> Path:
    """Map missing paths from the sealed `/home/user` layout to this checkout.

    Existing paths always win. Relative paths are returned unchanged so callers
    retain control of their own base directory.
    """

    path = Path(os.path.expandvars(os.fspath(value))).expanduser()
    if not path.is_absolute() or path.exists():
        return path

    current_repo = (
        Path(repo_root).expanduser()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    current_data = (
        Path(data_root).expanduser() if data_root is not None else default_data_root()
    )

    relative = _relative_to(path, _LEGACY_REPO_ROOT)
    if relative is not None:
        return current_repo / relative

    relative = _relative_to(path, _LEGACY_DATA_ROOT)
    if relative is not None:
        return current_data / relative

    relative = _relative_to(path, _LEGACY_COCO_ROOT)
    if relative is not None:
        coco_root = current_data / "COCO/coco2017"
        candidates = [coco_root / relative]
        if relative.parts and relative.parts[0] == "train2017":
            candidates.append(coco_root / "train2017" / relative)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    relative = _relative_to(path, _LEGACY_REFCOCO_ROOT)
    if relative is not None:
        return current_data / "COCO" / relative

    return path
