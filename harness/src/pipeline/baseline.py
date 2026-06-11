from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from harness.src.data.manifest import ManifestItem, read_jsonl, validate_items
from harness.src.metrics.cer import cer_stats, normalize_chinese_text
from harness.src.models.base import ASRBackend


@dataclass(frozen=True)
class BaselineSummary:
    model: str
    device: str
    manifest: str
    count: int
    total_ref_chars: int
    total_edit_distance: int
    cer: float
    model_load_time_s: float | None
    total_audio_time_s: float | None
    total_wall_time_s: float
    avg_wall_time_s: float


def run_baseline(
    backend: ASRBackend,
    project_root: Path,
    manifest_path: Path,
    prediction_path: Path,
    summary_path: Path,
    device: str = "cpu",
    model_load_time_s: float | None = None,
    limit: int | None = None,
) -> BaselineSummary:
    items = read_jsonl(manifest_path)
    if limit is not None:
        items = items[:limit]

    errors = validate_items(project_root, items)
    if errors:
        raise ValueError("Invalid manifest:\n" + "\n".join(errors[:20]))

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    total_distance = 0
    total_ref_chars = 0
    total_audio_time: float | None = 0.0
    started = time.perf_counter()

    with prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        for index, item in enumerate(items, start=1):
            row, audio_time = _evaluate_one(backend, project_root, item, index)
            if audio_time is None:
                total_audio_time = None
            elif total_audio_time is not None:
                total_audio_time += audio_time
            total_distance += row["edit_distance"]
            total_ref_chars += row["ref_chars"]
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    total_wall_time = time.perf_counter() - started
    summary = BaselineSummary(
        model=backend.model_name,
        device=device,
        manifest=_relative_or_absolute(manifest_path, project_root),
        count=len(items),
        total_ref_chars=total_ref_chars,
        total_edit_distance=total_distance,
        cer=(total_distance / total_ref_chars) if total_ref_chars else 0.0,
        model_load_time_s=round(model_load_time_s, 4) if model_load_time_s is not None else None,
        total_audio_time_s=round(total_audio_time, 4) if total_audio_time is not None else None,
        total_wall_time_s=round(total_wall_time, 4),
        avg_wall_time_s=round(total_wall_time / len(items), 4) if items else 0.0,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _evaluate_one(
    backend: ASRBackend,
    project_root: Path,
    item: ManifestItem,
    index: int,
) -> tuple[dict[str, object], float | None]:
    wav_path = project_root / item.wav_path
    started = time.perf_counter()
    result = backend.transcribe(wav_path)
    wall_time = time.perf_counter() - started
    stats = cer_stats(item.text, result.text)
    return (
        {
            "index": index,
            "utt_id": item.utt_id,
            "split": item.split,
            "wav_path": item.wav_path,
            "ref": item.text,
            "hyp": result.text,
            "ref_norm": normalize_chinese_text(item.text),
            "hyp_norm": normalize_chinese_text(result.text),
            "edit_distance": stats.distance,
            "ref_chars": stats.ref_chars,
            "cer": stats.cer,
            "wall_time_s": round(wall_time, 4),
        },
        _wav_duration_s(wav_path),
    )


def _wav_duration_s(path: Path) -> float | None:
    try:
        import wave

        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return round(frames / rate, 4) if rate else None
    except (OSError, wave.Error):
        return None


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
