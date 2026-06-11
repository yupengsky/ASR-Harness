from __future__ import annotations

import argparse
import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect baseline summary files into one JSON table.")
    parser.add_argument("--summary-root", default="outputs")
    parser.add_argument("--output", default="outputs/logs/baseline_summary.json")
    args = parser.parse_args()

    root = project_root()
    summary_root = root / args.summary_root
    rows = []
    for path in sorted(summary_root.glob("*/baseline/*/*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["summary_path"] = path.relative_to(root).as_posix()
        rows.append(data)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['model']}\t{row['manifest']}\t{row['device']}\t"
            f"count={row['count']}\tcer={row['cer']:.4f}\twall={row['total_wall_time_s']}s"
        )
    print(f"wrote {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()

