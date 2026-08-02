from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Set, Tuple


def load_holdout_keys(paths: Iterable[str]) -> Tuple[Set[Tuple[int, int]], Set[int]]:
    ann_keys: Set[Tuple[int, int]] = set()
    image_ids: Set[int] = set()
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_id = row.get("image_id")
                ann_id = row.get("ann_id")
                if image_id is None:
                    continue
                image_id = int(image_id)
                image_ids.add(image_id)
                if ann_id is not None:
                    ann_keys.add((image_id, int(ann_id)))
    return ann_keys, image_ids


def is_excluded(
    *,
    image_id: int,
    ann_id: int,
    level: str,
    ann_keys: Set[Tuple[int, int]],
    image_ids: Set[int],
) -> bool:
    level = str(level).lower().strip()
    if level == "ann":
        return (int(image_id), int(ann_id)) in ann_keys
    if level == "image":
        return int(image_id) in image_ids
    if level == "none":
        return False
    raise ValueError(f"Unknown holdout level: {level}")
