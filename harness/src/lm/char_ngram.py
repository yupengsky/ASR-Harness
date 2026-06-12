from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from harness.src.data.manifest import read_jsonl
from harness.src.metrics.cer import normalize_chinese_text


@dataclass
class CharNGramLM:
    order: int
    alpha: float
    vocab: set[str]
    context_counts: dict[str, int]
    ngram_counts: dict[tuple[str, str], int]

    def score_avg_logprob(self, text: str) -> float:
        chars = list(normalize_chinese_text(text))
        if not chars:
            return -100.0
        padded = ["<s>"] * (self.order - 1) + chars + ["</s>"]
        total = 0.0
        steps = 0
        vocab_size = max(len(self.vocab), 1)
        for idx in range(self.order - 1, len(padded)):
            char = padded[idx]
            prob = self._prob_with_backoff(padded, idx, char, vocab_size)
            total += math.log(prob)
            steps += 1
        return total / max(steps, 1)

    def _prob_with_backoff(self, padded: list[str], idx: int, char: str, vocab_size: int) -> float:
        for order in range(self.order, 0, -1):
            context = "\t".join(padded[idx - order + 1 : idx]) if order > 1 else ""
            context_count = self.context_counts.get(context, 0)
            if context_count == 0 and order > 1:
                continue
            count = self.ngram_counts.get((context, char), 0)
            return (count + self.alpha) / (context_count + self.alpha * vocab_size)
        return 1.0 / vocab_size

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "order": self.order,
            "alpha": self.alpha,
            "vocab": sorted(self.vocab),
            "context_counts": self.context_counts,
            "ngram_counts": [list(key) + [value] for key, value in self.ngram_counts.items()],
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharNGramLM":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            order=int(data["order"]),
            alpha=float(data["alpha"]),
            vocab=set(data["vocab"]),
            context_counts={str(k): int(v) for k, v in data["context_counts"].items()},
            ngram_counts={(str(context), str(char)): int(value) for context, char, value in data["ngram_counts"]},
        )


def build_char_ngram_lm(manifest_path: Path, order: int = 4, alpha: float = 0.1) -> CharNGramLM:
    vocab: set[str] = {"</s>"}
    context_counts: Counter[str] = Counter()
    ngram_counts: Counter[tuple[str, str]] = Counter()

    for item in read_jsonl(manifest_path):
        chars = list(normalize_chinese_text(item.text))
        vocab.update(chars)
        padded = ["<s>"] * (order - 1) + chars + ["</s>"]
        for idx in range(order - 1, len(padded)):
            char = padded[idx]
            for current_order in range(1, order + 1):
                context = (
                    "\t".join(padded[idx - current_order + 1 : idx])
                    if current_order > 1
                    else ""
                )
                context_counts[context] += 1
                ngram_counts[(context, char)] += 1

    return CharNGramLM(
        order=order,
        alpha=alpha,
        vocab=vocab,
        context_counts=dict(context_counts),
        ngram_counts=dict(ngram_counts),
    )
