from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_model_registry(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(project_root: Path, spec: dict[str, Any]) -> Path:
    path = project_root / spec["path"]
    if path.exists():
        return path
    for legacy in spec.get("legacy_paths", []):
        legacy_path = project_root / legacy
        if legacy_path.exists():
            return legacy_path
    return path

