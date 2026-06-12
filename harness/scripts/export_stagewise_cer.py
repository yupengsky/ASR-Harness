from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for each --input value.")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Input NAME cannot be empty.")
    return name, (ROOT / path).resolve()


def read_prediction_stats(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                distance = int(item["edit_distance"])
                ref_chars = int(item["ref_chars"])
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} lacks {exc.args[0]}") from exc
            rows.append((distance, ref_chars))
    return rows


def export_stagewise(inputs: list[tuple[str, Path]], output_path: Path, step: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "sample_count",
                "cumulative_edit_distance",
                "cumulative_ref_chars",
                "cumulative_cer",
                "window_edit_distance",
                "window_ref_chars",
                "window_cer",
            ],
        )
        writer.writeheader()
        for run_name, path in inputs:
            stats = read_prediction_stats(path)
            total_distance = 0
            total_ref_chars = 0
            window_distance = 0
            window_ref_chars = 0
            for index, (distance, ref_chars) in enumerate(stats, start=1):
                total_distance += distance
                total_ref_chars += ref_chars
                window_distance += distance
                window_ref_chars += ref_chars
                if index % step == 0 or index == len(stats):
                    writer.writerow(
                        {
                            "run": run_name,
                            "sample_count": index,
                            "cumulative_edit_distance": total_distance,
                            "cumulative_ref_chars": total_ref_chars,
                            "cumulative_cer": total_distance / total_ref_chars if total_ref_chars else 0.0,
                            "window_edit_distance": window_distance,
                            "window_ref_chars": window_ref_chars,
                            "window_cer": window_distance / window_ref_chars if window_ref_chars else 0.0,
                        }
                    )
                    window_distance = 0
                    window_ref_chars = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cumulative/window CER for report plots.")
    parser.add_argument("--input", action="append", type=parse_named_path, required=True, help="NAME=predictions.jsonl")
    parser.add_argument("--output", default="outputs/logs/stagewise_cer_full_test.csv")
    parser.add_argument("--step", type=int, default=50)
    args = parser.parse_args()

    if args.step <= 0:
        raise SystemExit("--step must be positive")

    output_path = ROOT / args.output
    export_stagewise(args.input, output_path, args.step)
    print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
