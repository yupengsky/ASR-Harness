from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SPLITS = ("train", "dev", "test")
TEXT_LINE_RE = re.compile(r"[\u4e00-\u9fff]")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def resolve_trn_path(trn_path: Path) -> Path:
    raw = read_text(trn_path)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) == 1 and lines[0].endswith(".trn"):
        target = (trn_path.parent / lines[0]).resolve()
        if target.is_file():
            return target
    return trn_path


def read_reference_text(trn_path: Path) -> str:
    resolved = resolve_trn_path(trn_path)
    lines = [line.strip() for line in read_text(resolved).splitlines() if line.strip()]
    for line in lines:
        if TEXT_LINE_RE.search(line):
            return "".join(CHINESE_CHAR_RE.findall(line))
    raise ValueError(f"No Chinese reference line found in {trn_path}")


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def collect_split(raw_root: Path, split: str, root: Path) -> list[dict[str, str]]:
    split_dir = raw_root / split
    canonical_trn_dir = raw_root / "data"
    items: list[dict[str, str]] = []
    for wav in sorted(split_dir.glob("*.wav")):
        trn = canonical_trn_dir / f"{wav.name}.trn"
        if not trn.is_file():
            trn = split_dir / f"{wav.name}.trn"
        if not trn.is_file():
            raise FileNotFoundError(f"Missing transcription for {wav}")
        items.append(
            {
                "utt_id": wav.stem,
                "split": split,
                "wav_path": relative(wav, root),
                "text": read_reference_text(trn),
            }
        )
    return items


def write_jsonl(path: Path, items: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_vocab(split_items: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    chars = sorted({char for items in split_items.values() for item in items for char in item["text"]})
    return {char: idx for idx, char in enumerate(chars)}


def stats_for(items: list[dict[str, str]]) -> dict[str, float | int]:
    lengths = [len(item["text"]) for item in items]
    if not lengths:
        return {"count": 0, "min_chars": 0, "max_chars": 0, "avg_chars": 0.0}
    return {
        "count": len(items),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
    }


def subset(items: list[dict[str, str]], size: int | None) -> list[dict[str, str]]:
    if size is None:
        return items
    return items[:size]


def emit_manifest_set(
    root: Path,
    manifest_root: Path,
    name: str,
    split_items: dict[str, list[dict[str, str]]],
    size: int | None,
) -> None:
    out_dir = manifest_root / name
    selected = {split: subset(items, size) for split, items in split_items.items()}
    for split, items in selected.items():
        write_jsonl(out_dir / f"{split}.jsonl", items)
    write_json(out_dir / "vocab.json", build_vocab(selected))
    stats = {split: stats_for(items) for split, items in selected.items()}
    stats["vocab"] = {
        "size": len(build_vocab(selected)),
        "path": relative(out_dir / "vocab.json", root),
    }
    write_json(out_dir / "stats.json", stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate THCHS30 manifests with project-relative paths.")
    parser.add_argument("--raw-root", default="data/raw/thchs30/data_thchs30")
    parser.add_argument("--manifest-root", default="data/manifests")
    parser.add_argument("--smoke-size", type=int, default=3)
    parser.add_argument("--debug-size", type=int, default=100)
    args = parser.parse_args()

    root = project_root()
    raw_root = root / args.raw_root
    manifest_root = root / args.manifest_root
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw THCHS30 directory not found: {raw_root}")

    split_items = {split: collect_split(raw_root, split, root) for split in SPLITS}
    emit_manifest_set(root, manifest_root, "smoke", split_items, args.smoke_size)
    emit_manifest_set(root, manifest_root, "debug100", split_items, args.debug_size)
    emit_manifest_set(root, manifest_root, "full", split_items, None)

    for name in ("smoke", "debug100", "full"):
        stats_path = manifest_root / name / "stats.json"
        print(f"wrote {stats_path.relative_to(root)}")


if __name__ == "__main__":
    main()
