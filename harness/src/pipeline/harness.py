from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from harness.src.audio.variants import DEFAULT_VARIANTS, generate_audio_variants
from harness.src.data.manifest import ManifestItem, read_jsonl, validate_items
from harness.src.lm.char_ngram import CharNGramLM
from harness.src.metrics.cer import cer_stats, normalize_chinese_text
from harness.src.models.base import ASRBackend
from harness.src.pipeline.baseline import _relative_or_absolute, _wav_duration_s


@dataclass(frozen=True)
class HarnessSummary:
    model: str
    device: str
    manifest: str
    train_manifest: str
    count: int
    variants: list[str]
    total_ref_chars: int
    total_edit_distance: int
    cer: float
    model_load_time_s: float | None
    total_audio_time_s: float | None
    total_wall_time_s: float
    avg_wall_time_s: float
    score_margin: float
    changed_from_orig: int
    improved_vs_orig: int
    worsened_vs_orig: int


def run_harness(
    backend: ASRBackend,
    project_root: Path,
    manifest_path: Path,
    train_manifest_path: Path,
    prediction_path: Path,
    summary_path: Path,
    lm: CharNGramLM,
    device: str = "cpu",
    model_load_time_s: float | None = None,
    limit: int | None = None,
    variant_names: tuple[str, ...] = DEFAULT_VARIANTS,
    score_margin: float = 0.0,
) -> HarnessSummary:
    items = read_jsonl(manifest_path)
    if limit is not None:
        items = items[:limit]

    errors = validate_items(project_root, items)
    if errors:
        raise ValueError("Invalid manifest:\n" + "\n".join(errors[:20]))

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    variant_root = prediction_path.parent / "audio_variants"
    total_distance = 0
    total_ref_chars = 0
    total_audio_time: float | None = 0.0
    changed = 0
    improved = 0
    worsened = 0
    started = time.perf_counter()

    with prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        for index, item in enumerate(items, start=1):
            row, audio_time = _evaluate_one(
                backend=backend,
                project_root=project_root,
                item=item,
                index=index,
                lm=lm,
                variant_root=variant_root,
                variant_names=variant_names,
                score_margin=score_margin,
            )
            if audio_time is None:
                total_audio_time = None
            elif total_audio_time is not None:
                total_audio_time += audio_time
            total_distance += row["edit_distance"]
            total_ref_chars += row["ref_chars"]
            changed += int(row["changed_from_orig"])
            improved += int(row["improved_vs_orig"])
            worsened += int(row["worsened_vs_orig"])
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    total_wall_time = time.perf_counter() - started
    summary = HarnessSummary(
        model=backend.model_name,
        device=device,
        manifest=_relative_or_absolute(manifest_path, project_root),
        train_manifest=_relative_or_absolute(train_manifest_path, project_root),
        count=len(items),
        variants=list(variant_names),
        total_ref_chars=total_ref_chars,
        total_edit_distance=total_distance,
        cer=(total_distance / total_ref_chars) if total_ref_chars else 0.0,
        model_load_time_s=round(model_load_time_s, 4) if model_load_time_s is not None else None,
        total_audio_time_s=round(total_audio_time, 4) if total_audio_time is not None else None,
        total_wall_time_s=round(total_wall_time, 4),
        avg_wall_time_s=round(total_wall_time / len(items), 4) if items else 0.0,
        score_margin=score_margin,
        changed_from_orig=changed,
        improved_vs_orig=improved,
        worsened_vs_orig=worsened,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _evaluate_one(
    backend: ASRBackend,
    project_root: Path,
    item: ManifestItem,
    index: int,
    lm: CharNGramLM,
    variant_root: Path,
    variant_names: tuple[str, ...],
    score_margin: float,
) -> tuple[dict[str, object], float | None]:
    wav_path = project_root / item.wav_path
    variants = generate_audio_variants(
        wav_path=wav_path,
        output_dir=variant_root / item.utt_id,
        variant_names=variant_names,
    )

    candidates: list[dict[str, object]] = []
    seen: dict[str, int] = {}
    for variant_index, variant in enumerate(variants):
        started = time.perf_counter()
        result = backend.transcribe(variant.path)
        wall_time = time.perf_counter() - started
        norm = normalize_chinese_text(result.text)
        lm_score = lm.score_avg_logprob(norm)
        candidate = {
            "variant": variant.name,
            "text": result.text,
            "norm": norm,
            "lm_score": lm_score,
            "wall_time_s": round(wall_time, 4),
            "variant_order": variant_index,
        }
        candidates.append(candidate)
        if norm not in seen:
            seen[norm] = len(candidates) - 1

    orig = candidates[0]
    target_len = len(str(orig["norm"]))
    for candidate in candidates:
        length = len(str(candidate["norm"]))
        length_penalty = abs(length - target_len) / max(target_len, 1)
        candidate["score"] = round(float(candidate["lm_score"]) - 1.5 * length_penalty, 6)

    orig = candidates[0]
    best = max(candidates, key=lambda c: (float(c["score"]), -int(c["variant_order"])))
    if str(best["norm"]) != str(orig["norm"]) and float(best["score"]) - float(orig["score"]) < score_margin:
        best = orig
    stats = cer_stats(item.text, str(best["text"]))
    orig_stats = cer_stats(item.text, str(orig["text"]))
    return (
        {
            "index": index,
            "utt_id": item.utt_id,
            "split": item.split,
            "wav_path": item.wav_path,
            "ref": item.text,
            "hyp": best["text"],
            "ref_norm": normalize_chinese_text(item.text),
            "hyp_norm": normalize_chinese_text(str(best["text"])),
            "edit_distance": stats.distance,
            "ref_chars": stats.ref_chars,
            "cer": stats.cer,
            "orig_hyp": orig["text"],
            "orig_cer": orig_stats.cer,
            "orig_edit_distance": orig_stats.distance,
            "selected_variant": best["variant"],
            "selected_score": best["score"],
            "changed_from_orig": str(best["norm"]) != str(orig["norm"]),
            "improved_vs_orig": stats.distance < orig_stats.distance,
            "worsened_vs_orig": stats.distance > orig_stats.distance,
            "candidates": candidates,
        },
        _wav_duration_s(wav_path),
    )
