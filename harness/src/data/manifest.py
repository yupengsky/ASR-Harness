from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestItem:
    utt_id: str
    split: str
    wav_path: str
    text: str


def read_jsonl(path: Path) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                items.append(
                    ManifestItem(
                        utt_id=obj["utt_id"],
                        split=obj["split"],
                        wav_path=obj["wav_path"],
                        text=obj["text"],
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing field {exc}") from exc
    return items


def write_jsonl(path: Path, items: Iterable[ManifestItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(
                json.dumps(
                    {
                        "utt_id": item.utt_id,
                        "split": item.split,
                        "wav_path": item.wav_path,
                        "text": item.text,
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")


def validate_items(project_root: Path, items: Iterable[ManifestItem]) -> list[str]:
    errors: list[str] = []
    for item in items:
        wav = project_root / item.wav_path
        if not wav.is_file():
            errors.append(f"{item.utt_id}: missing wav {item.wav_path}")
        if not item.text:
            errors.append(f"{item.utt_id}: empty text")
        if Path(item.wav_path).is_absolute():
            errors.append(f"{item.utt_id}: wav_path must be relative")
    return errors

