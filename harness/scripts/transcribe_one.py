from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path


os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.src.metrics.cer import cer_stats, normalize_chinese_text
from harness.src.models.factory import create_backend


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def round_time(value: float) -> float:
    return round(value, 4)


@contextlib.contextmanager
def quiet_third_party_logs(enabled: bool):
    if not enabled:
        yield
        return

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                yield
            finally:
                logging.disable(previous_disable_level)


def transcribe_qwen(
    audio_path: Path,
    model_name: str,
    model_path: str,
    device: str,
    dtype_name: str,
    language: str | None,
    max_new_tokens: int,
    quiet: bool,
) -> dict[str, object]:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit("Qwen3-ASR transcription requires package: qwen-asr") from exc

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    load_started = time.perf_counter()
    with quiet_third_party_logs(quiet):
        model = Qwen3ASRModel.from_pretrained(
            str(resolve_project_path(model_path)),
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=max_new_tokens,
        )
    load_time = time.perf_counter() - load_started

    infer_started = time.perf_counter()
    with quiet_third_party_logs(quiet):
        result = model.transcribe(audio=str(audio_path), language=language)
    wall_time = time.perf_counter() - infer_started
    text = result[0].text if result else ""
    detected_language = getattr(result[0], "language", None) if result else None

    return {
        "called": True,
        "model": model_name,
        "backend": "qwen-asr",
        "text": text,
        "norm": normalize_chinese_text(text),
        "detected_language": detected_language,
        "load_time_s": round_time(load_time),
        "wall_time_s": round_time(wall_time),
    }


def transcribe_registered_model(
    audio_path: Path,
    model_name: str,
    registry: str,
    device: str,
    quiet: bool,
) -> dict[str, object]:
    load_started = time.perf_counter()
    with quiet_third_party_logs(quiet):
        backend = create_backend(model_name, ROOT, resolve_project_path(registry), device=device)
    load_time = time.perf_counter() - load_started

    infer_started = time.perf_counter()
    with quiet_third_party_logs(quiet):
        result = backend.transcribe(audio_path)
    wall_time = time.perf_counter() - infer_started
    text = result.text

    return {
        "called": True,
        "model": model_name,
        "backend": "registered",
        "text": text,
        "norm": normalize_chinese_text(text),
        "load_time_s": round_time(load_time),
        "wall_time_s": round_time(wall_time),
    }


def build_result(
    audio_path: Path,
    qwen06: dict[str, object],
    paraformer: dict[str, object],
    qwen17: dict[str, object],
    reference: str | None,
) -> dict[str, object]:
    fast_agree = qwen06["norm"] == paraformer["norm"]
    if fast_agree:
        final_text = str(qwen06["text"])
        selected_source = "fast_exact:qwen3-asr-0.6b"
    else:
        final_text = str(qwen17["text"])
        selected_source = "qwen3-asr-1.7b"

    result: dict[str, object] = {
        "audio_path": relative_or_absolute(audio_path),
        "final_text": final_text,
        "final_norm": normalize_chinese_text(final_text),
        "selected_source": selected_source,
        "qwen17_called": not fast_agree,
        "decision": {
            "strategy": "cascade_fast_exact",
            "fast_models_agree": fast_agree,
            "fast_model_norms": {
                "qwen3-asr-0.6b": qwen06["norm"],
                "paraformer-zh": paraformer["norm"],
            },
        },
        "models": {
            "qwen3-asr-0.6b": qwen06,
            "paraformer-zh": paraformer,
            "qwen3-asr-1.7b": qwen17,
        },
    }

    if reference is not None:
        stats = cer_stats(reference, final_text)
        result["reference"] = reference
        result["reference_norm"] = normalize_chinese_text(reference)
        result["metrics"] = {
            "edit_distance": stats.distance,
            "ref_chars": stats.ref_chars,
            "cer": stats.cer,
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with the final ASR cascade system.")
    parser.add_argument("--audio", required=True, help="Audio file path. Relative paths are resolved from project root.")
    parser.add_argument("--ref", "--reference", dest="reference", help="Optional reference text for CER calculation.")
    parser.add_argument("--output", help="Optional JSON output path. If omitted, only prints JSON to stdout.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--registry", default="harness/configs/models.json")
    parser.add_argument("--qwen06-model-path", default="models/qwen3-asr-0.6b")
    parser.add_argument("--qwen17-model-path", default="models/qwen3-asr-1.7b")
    parser.add_argument("--paraformer-model", default="paraformer-zh")
    parser.add_argument("--verbose", action="store_true", help="Show third-party model logs and warnings.")
    args = parser.parse_args()

    audio_path = resolve_project_path(args.audio)
    if not audio_path.is_file():
        raise SystemExit(f"Audio file not found: {audio_path}")

    qwen06 = transcribe_qwen(
        audio_path=audio_path,
        model_name="qwen3-asr-0.6b",
        model_path=args.qwen06_model_path,
        device=args.device,
        dtype_name=args.dtype,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        quiet=not args.verbose,
    )
    paraformer = transcribe_registered_model(
        audio_path=audio_path,
        model_name=args.paraformer_model,
        registry=args.registry,
        device=args.device,
        quiet=not args.verbose,
    )

    if qwen06["norm"] == paraformer["norm"]:
        qwen17 = {
            "called": False,
            "model": "qwen3-asr-1.7b",
            "backend": "qwen-asr",
            "text": None,
            "norm": None,
            "reason": "skipped because fast model normalized outputs match",
        }
    else:
        qwen17 = transcribe_qwen(
            audio_path=audio_path,
            model_name="qwen3-asr-1.7b",
            model_path=args.qwen17_model_path,
            device=args.device,
            dtype_name=args.dtype,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            quiet=not args.verbose,
        )

    result = build_result(
        audio_path=audio_path,
        qwen06=qwen06,
        paraformer=paraformer,
        qwen17=qwen17,
        reference=args.reference,
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
