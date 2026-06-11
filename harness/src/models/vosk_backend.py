from __future__ import annotations

import json
import wave
from pathlib import Path

from harness.src.models.base import ASRResult, BackendDependencyError


class VoskBackend:
    def __init__(self, model_name: str, model_path: Path, device: str = "cpu") -> None:
        if device != "cpu":
            raise ValueError("Vosk backend only supports CPU inference")
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as exc:
            raise BackendDependencyError("Vosk backend requires package: vosk") from exc

        SetLogLevel(-1)
        self.model_name = model_name
        self._recognizer_cls = KaldiRecognizer
        self._model = Model(str(model_path))

    def transcribe(self, wav_path: Path) -> ASRResult:
        with wave.open(str(wav_path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError(f"Vosk expects mono 16-bit PCM WAV: {wav_path}")
            recognizer = self._recognizer_cls(self._model, wav.getframerate())
            chunks: list[dict[str, object]] = []
            while True:
                data = wav.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    chunks.append(json.loads(recognizer.Result()))
            final = json.loads(recognizer.FinalResult())
            chunks.append(final)
        text = "".join(str(chunk.get("text", "")) for chunk in chunks)
        return ASRResult(text=text, raw=chunks)

