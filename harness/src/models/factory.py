from __future__ import annotations

from pathlib import Path

from harness.src.models.base import ASRBackend
from harness.src.models.funasr_backend import FunASRBackend
from harness.src.models.registry import load_model_registry, resolve_model_path
from harness.src.models.vosk_backend import VoskBackend
from harness.src.models.whisper_backend import TransformersWhisperBackend


def create_backend(
    model_name: str,
    project_root: Path,
    registry_path: Path,
    device: str = "cpu",
) -> ASRBackend:
    registry = load_model_registry(registry_path)
    if model_name not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")

    spec = registry[model_name]
    model_path = resolve_model_path(project_root, spec)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found for {model_name}: {model_path}")

    model_type = spec["type"]
    if model_type == "funasr":
        return FunASRBackend(model_name, model_path, device=device)
    if model_type == "vosk":
        return VoskBackend(model_name, model_path, device=device)
    if model_type == "transformers-whisper":
        return TransformersWhisperBackend(model_name, model_path, device=device)
    raise NotImplementedError(f"No baseline backend implemented for model type: {model_type}")

