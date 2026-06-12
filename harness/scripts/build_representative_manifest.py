from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.src.data.manifest import ManifestItem, read_jsonl, write_jsonl


@dataclass(frozen=True)
class RunTarget:
    name: str
    full_cer: float
    selected_cer: float
    absolute_delta: float
    relative_delta: float


@dataclass(frozen=True)
class FeatureSummary:
    full_mean: float
    selected_mean: float
    relative_delta: float


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("NAME cannot be empty.")
    return name, ROOT / raw_path


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def read_prediction_stats(path: Path) -> dict[str, tuple[int, int]]:
    stats: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            stats[item["utt_id"]] = (int(item["edit_distance"]), int(item["ref_chars"]))
    return stats


def speaker_group(utt_id: str) -> str:
    return utt_id.split("_", 1)[0]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def rel_delta(selected: float, full: float) -> float:
    return (selected - full) / full if full else 0.0


def proportions(values: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = len(values)
    return {key: count / total for key, count in counts.items()}


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def make_decile_buckets(records: list[dict[str, object]], bucket_count: int) -> list[list[int]]:
    ranked = sorted(range(len(records)), key=lambda i: (records[i]["difficulty"], records[i]["index"]))
    buckets: list[list[int]] = []
    for bucket_index in range(bucket_count):
        start = round(bucket_index * len(ranked) / bucket_count)
        end = round((bucket_index + 1) * len(ranked) / bucket_count)
        buckets.append(ranked[start:end])
    return buckets


def target_bucket_counts(total_size: int, bucket_count: int) -> list[int]:
    base = total_size // bucket_count
    remainder = total_size % bucket_count
    return [base + (1 if index < remainder else 0) for index in range(bucket_count)]


def score_subset(
    selected_indices: list[int],
    records: list[dict[str, object]],
    run_names: list[str],
    full_run_cers: dict[str, float],
    full_features: dict[str, tuple[float, float]],
    full_group_props: dict[str, float],
) -> float:
    selected = [records[index] for index in selected_indices]
    ref_total = sum(int(record["ref_chars"]) for record in selected)

    score = 0.0
    for run_name in run_names:
        selected_edit = sum(int(record[f"{run_name}_edit"]) for record in selected)
        selected_cer = selected_edit / ref_total if ref_total else 0.0
        score += 80.0 * rel_delta(selected_cer, full_run_cers[run_name]) ** 2

    for feature_name, (full_mean, full_std) in full_features.items():
        values = [float(record[feature_name]) for record in selected]
        score += 2.0 * rel_delta(mean(values), full_mean) ** 2
        if full_std:
            score += 0.5 * rel_delta(stdev(values), full_std) ** 2

    selected_group_props = proportions([str(record["group"]) for record in selected])
    score += 1.5 * total_variation(selected_group_props, full_group_props) ** 2
    return score


def build_representative_subset(
    records: list[dict[str, object]],
    run_names: list[str],
    size: int,
    seed: int,
    iterations: int,
    bucket_count: int,
) -> tuple[list[int], float]:
    rng = random.Random(seed)
    buckets = make_decile_buckets(records, bucket_count)
    counts = target_bucket_counts(size, bucket_count)

    full_ref = sum(int(record["ref_chars"]) for record in records)
    full_run_cers = {
        run_name: sum(int(record[f"{run_name}_edit"]) for record in records) / full_ref
        for run_name in run_names
    }
    feature_names = ["duration_s", "ref_chars", "chars_per_second", "difficulty"]
    full_features = {
        name: (
            mean([float(record[name]) for record in records]),
            stdev([float(record[name]) for record in records]),
        )
        for name in feature_names
    }
    full_group_props = proportions([str(record["group"]) for record in records])

    best_indices: list[int] | None = None
    best_score = math.inf
    for _ in range(iterations):
        selected_indices: list[int] = []
        for bucket, count in zip(buckets, counts):
            selected_indices.extend(rng.sample(bucket, count))
        current_score = score_subset(
            selected_indices,
            records,
            run_names,
            full_run_cers,
            full_features,
            full_group_props,
        )
        if current_score < best_score:
            best_score = current_score
            best_indices = selected_indices

    if best_indices is None:
        raise RuntimeError("Failed to build representative subset.")
    return sorted(best_indices, key=lambda index: int(records[index]["index"])), best_score


def summarize(
    records: list[dict[str, object]],
    selected_indices: list[int],
    run_names: list[str],
    score: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    selected = [records[index] for index in selected_indices]
    full_ref = sum(int(record["ref_chars"]) for record in records)
    selected_ref = sum(int(record["ref_chars"]) for record in selected)

    run_targets: list[RunTarget] = []
    for run_name in run_names:
        full_cer = sum(int(record[f"{run_name}_edit"]) for record in records) / full_ref
        selected_cer = sum(int(record[f"{run_name}_edit"]) for record in selected) / selected_ref
        run_targets.append(
            RunTarget(
                name=run_name,
                full_cer=full_cer,
                selected_cer=selected_cer,
                absolute_delta=selected_cer - full_cer,
                relative_delta=rel_delta(selected_cer, full_cer),
            )
        )

    feature_names = ["duration_s", "ref_chars", "chars_per_second", "difficulty"]
    feature_summary = {
        name: asdict(
            FeatureSummary(
                full_mean=mean([float(record[name]) for record in records]),
                selected_mean=mean([float(record[name]) for record in selected]),
                relative_delta=rel_delta(
                    mean([float(record[name]) for record in selected]),
                    mean([float(record[name]) for record in records]),
                ),
            )
        )
        for name in feature_names
    }

    return {
        "source_manifest": args.source_manifest,
        "exclude_manifest": args.exclude_manifest,
        "output_manifest": args.output_manifest,
        "size": len(selected),
        "seed": args.seed,
        "iterations": args.iterations,
        "bucket_count": args.bucket_count,
        "score": score,
        "run_targets": [asdict(target) for target in run_targets],
        "feature_summary": feature_summary,
        "full_group_proportions": proportions([str(record["group"]) for record in records]),
        "selected_group_proportions": proportions([str(record["group"]) for record in selected]),
        "selected_utt_ids": [str(record["utt_id"]) for record in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact representative ASR manifest.")
    parser.add_argument("--source-manifest", default="data/manifests/full/test.jsonl")
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--output-manifest", default="data/manifests/representative100/test.jsonl")
    parser.add_argument("--diagnostics", default="outputs/logs/representative100_diagnostics.json")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--prediction", action="append", type=parse_named_path, required=True)
    args = parser.parse_args()

    manifest_path = ROOT / args.source_manifest
    items = read_jsonl(manifest_path)
    excluded_utt_ids: set[str] = set()
    for exclude_manifest in args.exclude_manifest:
        excluded_utt_ids.update(item.utt_id for item in read_jsonl(ROOT / exclude_manifest))
    if excluded_utt_ids:
        items = [item for item in items if item.utt_id not in excluded_utt_ids]
    prediction_stats = [(name, read_prediction_stats(path)) for name, path in args.prediction]
    run_names = [name for name, _ in prediction_stats]

    records: list[dict[str, object]] = []
    for index, item in enumerate(items):
        duration = wav_duration_s(ROOT / item.wav_path)
        ref_chars = len(item.text)
        record: dict[str, object] = {
            "index": index,
            "utt_id": item.utt_id,
            "duration_s": duration,
            "ref_chars": ref_chars,
            "chars_per_second": ref_chars / duration if duration else 0.0,
            "group": speaker_group(item.utt_id),
        }
        item_difficulties: list[float] = []
        for run_name, stats in prediction_stats:
            edit, ref = stats[item.utt_id]
            record[f"{run_name}_edit"] = edit
            record[f"{run_name}_ref"] = ref
            item_difficulties.append(edit / ref if ref else 0.0)
        record["difficulty"] = mean(item_difficulties)
        records.append(record)

    selected_indices, score = build_representative_subset(
        records=records,
        run_names=run_names,
        size=args.size,
        seed=args.seed,
        iterations=args.iterations,
        bucket_count=args.bucket_count,
    )
    selected_items = [items[int(records[index]["index"])] for index in selected_indices]
    output_manifest = ROOT / args.output_manifest
    write_jsonl(output_manifest, selected_items)

    diagnostics = summarize(records, selected_indices, run_names, score, args)
    diagnostics_path = ROOT / args.diagnostics
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(diagnostics["run_targets"], ensure_ascii=False, indent=2))
    print(f"manifest: {output_manifest.relative_to(ROOT)}")
    print(f"diagnostics: {diagnostics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
