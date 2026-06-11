from __future__ import annotations

from pathlib import Path

from harness.src.models.base import ASRResult, BackendDependencyError


class TransformersASRBackend:
    def __init__(self, model_name: str, model_path: Path, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise BackendDependencyError("Transformers ASR backend requires packages: torch, transformers") from exc

        self.model_name = model_name
        device_arg = -1
        if device.startswith("cuda"):
            device_arg = int(device.split(":", 1)[1]) if ":" in device else 0
        dtype = torch.float16 if device_arg >= 0 else torch.float32
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=str(model_path),
            tokenizer=str(model_path),
            feature_extractor=str(model_path),
            device=device_arg,
            dtype=dtype,
            trust_remote_code=True,
        )

    def transcribe(self, wav_path: Path) -> ASRResult:
        raw = self._pipeline(str(wav_path))
        text = raw.get("text", "") if isinstance(raw, dict) else str(raw)
        return ASRResult(text=text, raw=raw)

