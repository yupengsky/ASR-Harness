from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ASRResult:
    text: str
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ASRBackend(Protocol):
    model_name: str

    def transcribe(self, wav_path: Path) -> ASRResult:
        """Transcribe one audio file."""


class BackendDependencyError(RuntimeError):
    """Raised when an optional ASR backend dependency is unavailable."""

