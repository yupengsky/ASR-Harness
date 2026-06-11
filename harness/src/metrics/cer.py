from __future__ import annotations

import re


_SPACE_RE = re.compile(r"\s+")


def normalize_chinese_text(text: str) -> str:
    """Normalize reference and prediction text for character-level CER."""
    return _SPACE_RE.sub("", text.strip())


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

