from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class AudioVariant:
    name: str
    path: Path


DEFAULT_VARIANTS = (
    "orig",
    "peak_norm",
    "trim_peak_norm",
    "preemph",
    "speed_0_95",
)


def generate_audio_variants(
    wav_path: Path,
    output_dir: Path,
    variant_names: tuple[str, ...] = DEFAULT_VARIANTS,
    sample_rate: int = 16000,
) -> list[AudioVariant]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = signal.resample_poly(audio, sample_rate, sr).astype(np.float32)
        sr = sample_rate

    variants: list[AudioVariant] = []
    for name in variant_names:
        if name == "orig":
            variants.append(AudioVariant(name=name, path=wav_path))
            continue
        transformed = _transform(audio, name)
        out_path = output_dir / f"{wav_path.stem}.{name}.wav"
        sf.write(str(out_path), transformed, sr, subtype="PCM_16")
        variants.append(AudioVariant(name=name, path=out_path))
    return variants


def _transform(audio: np.ndarray, name: str) -> np.ndarray:
    if name == "peak_norm":
        return _peak_norm(audio)
    if name == "trim_peak_norm":
        return _trim_silence(_peak_norm(audio))
    if name == "preemph":
        return _peak_norm(np.append(audio[:1], audio[1:] - 0.97 * audio[:-1]))
    if name == "highpass":
        sos = signal.butter(4, 80, btype="highpass", fs=16000, output="sos")
        return _peak_norm(signal.sosfilt(sos, audio).astype(np.float32))
    if name == "speed_0_95":
        return _peak_norm(_resample_length(audio, 1 / 0.95))
    if name == "speed_1_05":
        return _peak_norm(_resample_length(audio, 1 / 1.05))
    raise ValueError(f"Unknown audio variant: {name}")


def _peak_norm(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-6:
        return audio.astype(np.float32)
    return np.clip(audio * (target_peak / peak), -0.99, 0.99).astype(np.float32)


def _trim_silence(audio: np.ndarray, threshold_ratio: float = 0.02, pad_samples: int = 1600) -> np.ndarray:
    if audio.size == 0:
        return audio
    threshold = max(float(np.max(np.abs(audio))) * threshold_ratio, 1e-4)
    active = np.flatnonzero(np.abs(audio) > threshold)
    if active.size == 0:
        return audio.astype(np.float32)
    start = max(int(active[0]) - pad_samples, 0)
    end = min(int(active[-1]) + pad_samples + 1, audio.size)
    return audio[start:end].astype(np.float32)


def _resample_length(audio: np.ndarray, length_ratio: float) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    new_len = max(int(round(audio.size * length_ratio)), 1)
    return signal.resample(audio, new_len).astype(np.float32)
