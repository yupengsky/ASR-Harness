from __future__ import annotations

import re
from dataclasses import dataclass


_SPACE_RE = re.compile(r"\s+")
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def normalize_chinese_text(text: str) -> str:
    """Normalize reference and prediction text for character-level CER."""
    compact = _SPACE_RE.sub("", text.strip())
    return "".join(_CHINESE_CHAR_RE.findall(compact))


def edit_distance(ref: str, hyp: str) -> int:
    if ref == hyp:
        return 0
    prev = list(range(len(hyp) + 1))
    for i, r_char in enumerate(ref, start=1):
        curr = [i]
        for j, h_char in enumerate(hyp, start=1):
            cost = 0 if r_char == h_char else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref_norm = normalize_chinese_text(ref)
    hyp_norm = normalize_chinese_text(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    return edit_distance(ref_norm, hyp_norm) / len(ref_norm)


@dataclass(frozen=True)
class CerStats:
    distance: int
    ref_chars: int

    @property
    def cer(self) -> float:
        if self.ref_chars == 0:
            return 0.0 if self.distance == 0 else 1.0
        return self.distance / self.ref_chars


def cer_stats(ref: str, hyp: str) -> CerStats:
    ref_norm = normalize_chinese_text(ref)
    hyp_norm = normalize_chinese_text(hyp)
    return CerStats(distance=edit_distance(ref_norm, hyp_norm), ref_chars=len(ref_norm))
