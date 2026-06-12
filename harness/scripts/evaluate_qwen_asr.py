from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.src.data.manifest import read_jsonl, validate_items
from harness.src.metrics.cer import cer_stats, normalize_chinese_text
from harness.src.pipeline.baseline import _relative_or_absolute, _wav_duration_s


@dataclass(frozen=True)
class QwenASRSummary:
    model: str
    backend: str
    device: str
    dtype: str
    language: str | None
    manifest: str
    count: int
    total_ref_chars: int
    total_edit_distance: int
    cer: float
    model_load_time_s: float
    total_audio_time_s: float | None
    total_wall_time_s: float
    avg_wall_time_s: float


def default_output_paths(model: str, manifest: Path, output_root: Path) -> tuple[Path, Path]:
    set_name = manifest.parent.name
    split_name = manifest.stem
    out_dir = output_root / set_name / "qwen_asr" / model
    return out_dir / f"{split_name}_predictions.jsonl", out_dir / f"{split_name}_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-ASR using the official qwen-asr package.")
    parser.add_argument("--model-path", default="models/qwen3-asr-0.6b")
    parser.add_argument("--model-name", default="qwen3-asr-0.6b")
    parser.add_argument("--manifest", default="data/manifests/smoke/test.jsonl")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    prediction_path, summary_path = default_output_paths(args.model_name, manifest_path, ROOT / args.output_root)

    if args.dry_run:
        print(json.dumps({
            "model_path": args.model_path,
            "manifest": _relative_or_absolute(manifest_path, ROOT),
            "predictions": _relative_or_absolute(prediction_path, ROOT),
            "summary": _relative_or_absolute(summary_path, ROOT),
            "device": args.device,
            "dtype": args.dtype,
            "language": args.language,
            "limit": args.limit,
        }, ensure_ascii=False, indent=2))
        return

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit("Qwen3-ASR evaluation requires package: qwen-asr") from exc

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    items = read_jsonl(manifest_path)
    if args.limit is not None:
        items = items[: args.limit]
    errors = validate_items(ROOT, items)
    if errors:
        raise SystemExit("Invalid manifest:\n" + "\n".join(errors[:20]))

    load_started = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        str(ROOT / args.model_path),
        dtype=dtype,
        device_map=args.device,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    model_load_time = time.perf_counter() - load_started

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    total_distance = 0
    total_ref_chars = 0
    total_audio_time: float | None = 0.0
    started = time.perf_counter()

    with prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        for index, item in enumerate(items, start=1):
            wav_path = ROOT / item.wav_path
            infer_started = time.perf_counter()
            result = model.transcribe(audio=str(wav_path), language=args.language)
            wall_time = time.perf_counter() - infer_started
            hyp = result[0].text if result else ""
            detected_language = getattr(result[0], "language", None) if result else None
            stats = cer_stats(item.text, hyp)
            audio_time = _wav_duration_s(wav_path)
            if audio_time is None:
                total_audio_time = None
            elif total_audio_time is not None:
                total_audio_time += audio_time
            total_distance += stats.distance
            total_ref_chars += stats.ref_chars
            f.write(json.dumps({
                "index": index,
                "utt_id": item.utt_id,
                "split": item.split,
                "wav_path": item.wav_path,
                "ref": item.text,
                "hyp": hyp,
                "ref_norm": normalize_chinese_text(item.text),
                "hyp_norm": normalize_chinese_text(hyp),
                "edit_distance": stats.distance,
                "ref_chars": stats.ref_chars,
                "cer": stats.cer,
                "detected_language": detected_language,
                "wall_time_s": round(wall_time, 4),
            }, ensure_ascii=False))
            f.write("\n")
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(items)):
                elapsed = time.perf_counter() - started
                print(
                    f"progress {index}/{len(items)} cer={total_distance / total_ref_chars:.6f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    total_wall_time = time.perf_counter() - started
    summary = QwenASRSummary(
        model=args.model_name,
        backend="qwen-asr",
        device=args.device,
        dtype=args.dtype,
        language=args.language,
        manifest=_relative_or_absolute(manifest_path, ROOT),
        count=len(items),
        total_ref_chars=total_ref_chars,
        total_edit_distance=total_distance,
        cer=(total_distance / total_ref_chars) if total_ref_chars else 0.0,
        model_load_time_s=round(model_load_time, 4),
        total_audio_time_s=round(total_audio_time, 4) if total_audio_time is not None else None,
        total_wall_time_s=round(total_wall_time, 4),
        avg_wall_time_s=round(total_wall_time / len(items), 4) if items else 0.0,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print(f"predictions: {prediction_path.relative_to(ROOT)}")
    print(f"summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
