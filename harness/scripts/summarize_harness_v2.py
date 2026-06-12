from __future__ import annotations

import argparse
import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Harness v2 fusion summary files.")
    parser.add_argument("--summary-root", default="outputs")
    parser.add_argument("--output", default="outputs/logs/harness_v2_summary.json")
    args = parser.parse_args()

    root = project_root()
    rows = []
    for path in sorted((root / args.summary_root).glob("*/harness_v2/*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["experiment"] = path.parent.name
        data["summary_path"] = path.relative_to(root).as_posix()
        rows.append(data)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['experiment']}\t{row['split']}\t{row['candidate_mode']}\t{row['strategy']}\t"
            f"anchor={row['anchor_cer']:.4f}\tcer={row['cer']:.4f}\toracle={row['oracle_cer']:.4f}"
        )
    print(f"wrote {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()

