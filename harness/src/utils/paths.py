from __future__ import annotations

from pathlib import Path


def project_root_from_script(script_path: str | Path) -> Path:
    return Path(script_path).resolve().parents[2]


def to_posix_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()

