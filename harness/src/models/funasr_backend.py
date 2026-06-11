from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.src.models.base import ASRResult, BackendDependencyError


class FunASRBackend:
    def __init__(self, model_name: str, model_path: Path, device: str = "cpu") -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise BackendDependencyError("FunASR backend requires package: funasr") from exc

        self.model_name = model_name
        self._model = AutoModel(model=str(model_path), device=device)

    def transcribe(self, wav_path: Path) -> ASRResult:
        raw = self._model.generate(input=str(wav_path))
        return ASRResult(text=_extract_text(raw), raw=raw)


def _extract_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("text", "sentence", "pred"):
            value = raw.get(key)
            if isinstance(value, str):
                return value
    if isinstance(raw, list):
        texts = [_extract_text(item) for item in raw]
        return "".join(texts)
    return str(raw)

