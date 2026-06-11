from __future__ import annotations

from pathlib import Path

from harness.src.models.base import ASRResult, BackendDependencyError


class SherpaOnnxParaformerBackend:
    def __init__(self, model_name: str, model_path: Path, device: str = "cpu") -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise BackendDependencyError("Sherpa-ONNX backend requires package: sherpa-onnx") from exc

        provider = "cuda" if device.startswith("cuda") else "cpu"
        self.model_name = model_name
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(model_path / "model.int8.onnx"),
            tokens=str(model_path / "tokens.txt"),
            provider=provider,
        )

    def transcribe(self, wav_path: Path) -> ASRResult:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise BackendDependencyError("Sherpa-ONNX backend requires package: soundfile") from exc

        samples, sample_rate = sf.read(str(wav_path), dtype="float32")
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        result = stream.result
        return ASRResult(text=result.text, raw={"tokens": result.tokens, "timestamps": result.timestamps})

